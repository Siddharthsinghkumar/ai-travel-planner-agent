# core/api_key_manager.py
# JSON-backed APIKeyManager: safe, atomic writes, async locks, .env reload
# Stores only env var names + fingerprints, never raw keys.
# All datetimes are timezone-aware (UTC) when stored as ISO strings, but exhaustion timestamps
# are kept as float (epoch) for fast comparison. The state file persists only fingerprints and
# exhaustion timestamps (as float). The actual keys are kept in memory only.
import os
import re
import json
import time
import hashlib
import asyncio
import logging
import contextlib
import fcntl
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from pathlib import Path
from dotenv import dotenv_values, find_dotenv
from typing import Dict, List, Optional, Tuple, Any, Callable, Awaitable, Union
from core.env_config import get_env_int, get_env_str

logger = logging.getLogger(__name__)

# File path (override with env if you want)
STATE_FILE = Path(get_env_str("KEY_STATE_FILE", "data/api_key_state.json"))
RELOAD_INTERVAL_SECONDS = get_env_int("API_KEY_RELOAD_INTERVAL", 86400)  # 24h (for full env reload)
# How often to refresh environment variables in the background
REFRESH_INTERVAL = get_env_int("KEY_REFRESH_INTERVAL", 30)  # seconds
# Lockfile path for multi‑process safety (override via env)
LOCKFILE_PATH = get_env_str("KEY_REFRESH_LOCKFILE", "/tmp/llm_key_refresh.lock")

# policy mapping for fallback exhaustion durations
POLICIES = {
    "serpapi": "monthly",
    "openai": "credit",
    "gemini": "daily",
    "weather": "daily"
}

# patterns to find numbered keys in env
ENV_PATTERNS = {
    "serpapi": re.compile(r"SERPAPI_KEY_(\d+)"),
    "openai": re.compile(r"OPENAI_KEY_(\d+)"),
    "gemini": re.compile(r"GEMINI_KEY_(\d+)"),
    "weather": re.compile(r"WEATHER_KEY_(\d+)")
}

def _now():
    """Return current UTC datetime (aware)."""
    return datetime.now(UTC)

def _now_ts() -> float:
    """Return current epoch timestamp (UTC)."""
    return time.time()

def _first_of_next_month(dt: datetime) -> datetime:
    """Return timezone-aware UTC datetime for the first day of the next month."""
    year = dt.year + (1 if dt.month == 12 else 0)
    month = 1 if dt.month == 12 else dt.month + 1
    return datetime(year, month, 1, tzinfo=UTC)

def _compute_exhaustion_until(service: str, reset_at: Optional[Union[datetime, float]] = None) -> float:
    """Return epoch timestamp for exhaustion 'until' time.
       If reset_at is given (datetime or float) it is used. Otherwise use policy defaults.
    """
    if reset_at is not None:
        if isinstance(reset_at, datetime):
            if reset_at.tzinfo is None:
                reset_at = reset_at.replace(tzinfo=UTC)
            return reset_at.timestamp()
        else:
            return float(reset_at)

    policy = POLICIES.get(service, "daily")
    now_dt = _now()
    if policy == "monthly":
        reset_dt = _first_of_next_month(now_dt)
    elif policy == "daily":
        reset_dt = now_dt + timedelta(days=1)
    elif policy == "credit":
        reset_dt = datetime.max.replace(tzinfo=UTC)
    else:
        reset_dt = now_dt + timedelta(days=1)
    return reset_dt.timestamp()

def _exhaustion_ttl_for_error(service: str, reason: str) -> float:
    """Return an appropriate exhaustion timestamp based on the error reason."""
    now_dt = _now()
    if reason in ("unauthorized", "invalid_key"):
        # Permanent until manually fixed
        return datetime.max.replace(tzinfo=UTC).timestamp()
    if reason == "rate_limit":
        # Short backoff (1 hour)
        return (now_dt + timedelta(hours=1)).timestamp()
    if reason == "quota_exceeded":
        # Use service's normal policy (daily/monthly)
        return _compute_exhaustion_until(service)
    # Default fallback
    return _compute_exhaustion_until(service)

def _try_acquire_lockfile(lockfile_path: str):
    """Acquire an exclusive lock on the given file; returns file descriptor or None."""
    fd = None
    try:
        fd = os.open(lockfile_path, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        return fd
    except BlockingIOError:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        return None
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        return None

@dataclass
class KeyEntry:
    """In-memory representation of a single API key."""
    value: str                     # the actual key (never persisted)
    fingerprint: str               # sha256 of value
    exhausted_until: Optional[float] = None   # epoch timestamp, or None
    in_use: int = 0                # number of current reservations
    last_used: float = field(default_factory=time.monotonic)  # monotonic time of last use
    # Pending exhaustion (set when mark_exhausted is called while in_use > 0)
    _pending_exhaust: bool = False
    _pending_exhaust_until: Optional[float] = None
    # Pending clear (set when a listener wants to clear/remove key after usage)
    _pending_clear: bool = False
    # Per‑key lock for atomic in_use operations
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False)

class APIKeyManager:
    def __init__(self, refresh_interval: int = REFRESH_INTERVAL):
        self._lock = asyncio.Lock()
        self._keys: Dict[str, List[KeyEntry]] = {}          # service -> list of KeyEntry
        self._rr_index: Dict[str, int] = {}                 # round-robin index per service
        self._refresh_task: Optional[asyncio.Task] = None
        self._stop_refresh = asyncio.Event()
        self.refresh_interval = refresh_interval
        self._last_state_write: float = 0.0                  # for throttling writes (not used now)
        self._callbacks: List[Callable[[], Awaitable[None]]] = []  # legacy change callbacks
        self._key_event_listeners: List[Callable[[str, dict], Any]] = []  # event listeners
        self._lockfile_fd: Optional[int] = None              # file descriptor for refresh lock

        # ensure state directory exists
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # initial load: first from state file, then from environment
        self._load_initial_state()
        # No auto-start of refresh loop – caller must start explicitly

    # ---------- fingerprint ----------
    @staticmethod
    def _fingerprint(key: str) -> str:
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    # ---------- state file persistence ----------
    def _load_state_file(self) -> Dict[str, Any]:
        """Load persisted exhaustion data from JSON file.
           Returns a dict mapping service -> fingerprint -> exhausted_until (float)."""
        if not STATE_FILE.exists():
            return {}
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to read key state file; starting with empty state")
            return {}

    def _write_state_file(self, state: Dict[str, Any]):
        """Atomically write state (fingerprint -> exhausted_until) to disk, with secure perms."""
        tmp = STATE_FILE.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, sort_keys=True)
            tmp.replace(STATE_FILE)
            try:
                STATE_FILE.chmod(0o600)
            except Exception:
                logger.debug("Could not set permission on key state file")
            self._last_state_write = _now_ts()
        except Exception:
            logger.exception("Failed to write key state file")

    async def _persist_exhaustion(self):
        """Snapshot current exhaustion and write to disk safely.
           Snapshot is taken while holding self._lock, then the write happens
           without holding the lock (to avoid blocking other ops)."""
        # Build snapshot under global lock
        async with self._lock:
            state = {}
            for service, entries in self._keys.items():
                svc_state = {}
                for ke in entries:
                    if ke.exhausted_until is not None:
                        svc_state[ke.fingerprint] = {"exhausted_until": ke.exhausted_until}
                if svc_state:
                    state[service] = svc_state
        # write file (sync) outside lock
        self._write_state_file(state)

    def _load_initial_state(self):
        """Load persisted exhaustion data and merge with environment (blocking, called at init)."""
        persisted = self._load_state_file()
        # Parse environment
        env_keys = self._parse_env_keys()
        # Build initial _keys from env, using persisted exhaustion where fingerprint matches
        self._keys.clear()
        for service, key_list in env_keys.items():
            entries = []
            for key in key_list:
                fp = self._fingerprint(key)
                exhausted = None
                # check persisted state for this fingerprint
                svc_state = persisted.get(service, {})
                if fp in svc_state:
                    exhausted = svc_state[fp].get("exhausted_until")
                entries.append(KeyEntry(value=key, fingerprint=fp, exhausted_until=exhausted))
            self._keys[service] = entries
            self._rr_index[service] = 0

    # ---------- environment parsing ----------
    def _parse_env_keys(self) -> Dict[str, List[str]]:
        """Read .env file directly and return a dict service -> list of keys (in index order).
           Uses dotenv_values to avoid polluting os.environ."""
        # find_dotenv() returns the path to the .env file; if not found, it returns ''
        env_path = find_dotenv()
        if not env_path:
            logger.debug("No .env file found, using empty environment")
            env = {}
        else:
            env = dotenv_values(env_path)  # reads the file without modifying os.environ
        services = {}
        for service, prog in ENV_PATTERNS.items():
            # collect keys with their numeric index
            indexed = []
            for name, value in env.items():
                m = prog.match(name)
                if m and value and value.strip():
                    idx = int(m.group(1))
                    indexed.append((idx, value.strip()))
            # sort by index and store values in order
            if indexed:
                indexed.sort(key=lambda x: x[0])
                services[service] = [v for _, v in indexed]
        return services

    # ---------- legacy change notification ----------
    def register_on_change(self, callback: Callable[[], Awaitable[None]]):
        """Register an async callback to be invoked when the set of keys (fingerprints) changes."""
        self._callbacks.append(callback)

    async def _notify_change(self):
        """Call all registered change callbacks (schedule each so they cannot block the caller)."""
        loop = asyncio.get_running_loop()
        for cb in list(self._callbacks):
            try:
                if asyncio.iscoroutinefunction(cb):
                    async def _run_cb(c=cb):
                        try:
                            await c()
                        except Exception:
                            logger.exception("Key change callback failed")
                    asyncio.create_task(_run_cb())
                else:
                    def _run_sync_cb(c=cb):
                        try:
                            c()
                        except Exception:
                            logger.exception("Key change callback (sync) failed")
                    loop.run_in_executor(None, _run_sync_cb)
            except Exception:
                logger.exception("Failed to schedule key change callback")

    # ---------- event listener API (handles both sync and async) ----------
    def register_key_event_listener(self, listener: Callable[[str, dict], Any]) -> None:
        """Register a callback to be notified of key-related events.
           The callback can be sync or async and receives (event_name, payload).
           Events: "key_exhausted", "env_changed", "key_no_longer_in_use".
        """
        self._key_event_listeners.append(listener)

    async def _notify_listeners(self, event_name: str, payload: dict):
        """Notify all registered event listeners (fire-and-forget). Each listener is executed
           in its own small task / executor wrapper and exceptions are caught and logged.
        """
        loop = asyncio.get_running_loop()

        for listener in list(self._key_event_listeners):
            try:
                if asyncio.iscoroutinefunction(listener):
                    # run async listener in its own task and catch/log exceptions
                    async def _run_async_listener(l=listener):
                        try:
                            await l(event_name, payload)
                        except Exception:
                            logger.exception("Async key event listener raised an exception",
                                             extra={"event": event_name})
                    asyncio.create_task(_run_async_listener())
                else:
                    # wrap sync listener in a callable that logs exceptions, then run in executor
                    def _run_sync_listener(l=listener):
                        try:
                            l(event_name, payload)
                        except Exception:
                            logger.exception("Sync key event listener raised an exception",
                                             extra={"event": event_name})
                    loop.run_in_executor(None, _run_sync_listener)
            except Exception:
                # This should be rare (scheduling error), but log it
                logger.exception("Failed to schedule key event listener", extra={"event": event_name})

    # ---------- background refresh with lockfile ----------
    def start_refresh_loop(self, interval_seconds: int = REFRESH_INTERVAL,
                           lockfile: str = LOCKFILE_PATH,
                           skip_lock_check: bool = False):
        """Start the background refresh task if not already running.
           If skip_lock_check is True, the caller is responsible for ensuring only one process
           runs the loop (e.g., via an external distributed lock). Otherwise a POSIX file lock
           is used to prevent multiple loops on the same host.
        """
        if self._refresh_task is not None and not self._refresh_task.done():
            return

        if not skip_lock_check:
            # Try to acquire the lockfile
            fd = _try_acquire_lockfile(lockfile)
            if fd is None:
                logger.info("Another process owns the refresh loop lock - not starting a second loop.")
                return
            self._lockfile_fd = fd
        else:
            self._lockfile_fd = None

        self.refresh_interval = interval_seconds
        self._stop_refresh.clear()

        # Verify we have a running event loop before creating the task
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("start_refresh_loop called without a running event loop; skipping.")
            return

        self._refresh_task = loop.create_task(self._refresh_loop())
        logger.info("Started API key refresh loop (interval=%ds)", interval_seconds)

    def stop_refresh_loop(self):
        """Stop the background refresh task and release the lockfile."""
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None
            logger.info("Stopped API key refresh loop")
        if self._lockfile_fd is not None:
            try:
                fcntl.flock(self._lockfile_fd, fcntl.LOCK_UN)
                os.close(self._lockfile_fd)
            except Exception:
                logger.exception("Error releasing lockfile")
            finally:
                self._lockfile_fd = None

    async def _refresh_loop(self):
        """Periodically refresh keys from the .env file."""
        try:
            while not self._stop_refresh.is_set():
                try:
                    changed = await self._reload_env_keys_if_changed()
                    if changed:
                        # notify legacy listeners
                        await self._notify_change()
                except Exception:
                    logger.exception("Error during key refresh loop")
                try:
                    # Wait for interval or stop signal
                    await asyncio.wait_for(self._stop_refresh.wait(), timeout=self.refresh_interval)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info("Key manager refresh loop cancelled")
        finally:
            # Release lockfile when loop ends
            if self._lockfile_fd is not None:
                try:
                    fcntl.flock(self._lockfile_fd, fcntl.LOCK_UN)
                    os.close(self._lockfile_fd)
                except Exception:
                    logger.exception("Error releasing lockfile")
                finally:
                    self._lockfile_fd = None

    async def _reload_env_keys_if_changed(self) -> bool:
        """Reload keys from environment, merge with current entries, and return True if the set of fingerprints changed."""
        # Build snapshot while holding the lock
        async with self._lock:
            new_env = self._parse_env_keys()

            # Build old fingerprint maps for detailed payload
            old_fingerprint_maps = {}
            for svc, entries in self._keys.items():
                old_fingerprint_maps[svc] = {i: ke.fingerprint for i, ke in enumerate(entries)}

            # Build new keys dict and new fingerprint maps
            new_keys = {}
            new_fingerprint_maps = {}
            for service, key_list in new_env.items():
                entries = []
                # Get current entries for this service (if any)
                current_entries = {ke.fingerprint: ke for ke in self._keys.get(service, [])}
                for key in key_list:
                    fp = self._fingerprint(key)
                    if fp in current_entries:
                        # reuse existing entry (preserves exhausted_until, in_use, last_used, pending flags)
                        ke = current_entries[fp]
                        # update value in case it changed (fingerprint same implies value same, but safe)
                        ke.value = key
                        entries.append(ke)
                    else:
                        # brand new key
                        entries.append(KeyEntry(value=key, fingerprint=fp))
                new_keys[service] = entries
                new_fingerprint_maps[service] = {i: ke.fingerprint for i, ke in enumerate(entries)}
                # update round-robin index (if service existed, keep index; otherwise start at 0)
                if service not in self._rr_index:
                    self._rr_index[service] = 0

            # Remove services no longer present
            for service in list(self._rr_index.keys()):
                if service not in new_env:
                    self._rr_index.pop(service, None)

            # Determine if any service's fingerprint set changed
            old_fp_sets = {svc: set(m.values()) for svc, m in old_fingerprint_maps.items()}
            new_fp_sets = {svc: set(m.values()) for svc, m in new_fingerprint_maps.items()}
            changed = (old_fp_sets != new_fp_sets)

            # Compute affected (service, idx) pairs where fingerprint changed
            affected = []
            all_services = set(old_fingerprint_maps.keys()) | set(new_fingerprint_maps.keys())
            for svc in all_services:
                old_fps = old_fingerprint_maps.get(svc, {})
                new_fps = new_fingerprint_maps.get(svc, {})
                all_idxs = set(old_fps.keys()) | set(new_fps.keys())
                for idx in all_idxs:
                    if old_fps.get(idx) != new_fps.get(idx):
                        affected.append((svc, idx))

            self._keys = new_keys

            # Build persistence snapshot if changed
            persist_state: Dict[str, Any] = {}
            if changed:
                for svc, entries in new_keys.items():
                    svc_state = {}
                    for ke in entries:
                        if ke.exhausted_until is not None:
                            svc_state[ke.fingerprint] = {"exhausted_until": ke.exhausted_until}
                    if svc_state:
                        persist_state[svc] = svc_state

        # Outside lock: write state file and notify if changed
        if changed:
            if persist_state:
                self._write_state_file(persist_state)
            logger.info("Key set changed – triggering callbacks")
            asyncio.create_task(self._notify_listeners("env_changed", {
                "old_fingerprint_maps": old_fingerprint_maps,
                "new_fingerprint_maps": new_fingerprint_maps,
                "affected": affected,
            }))

        return changed

    # ---------- key selection (round-robin, skip exhausted/in_use/pending_clear) ----------
    async def _pick_active_key(self, service: str) -> Tuple[Optional[int], Optional[KeyEntry]]:
        """Round-robin choose a non-exhausted key entry that is not in_use and not pending_clear. Returns (index, entry)."""
        entries = self._keys.get(service)
        if not entries:
            return None, None
        n = len(entries)
        start = self._rr_index.get(service, 0) % n
        now = _now_ts()
        for i in range(n):
            idx = (start + i) % n
            ke = entries[idx]
            if ke.exhausted_until and ke.exhausted_until > now:
                continue
            if ke.in_use > 0:
                continue
            if ke._pending_clear:
                continue
            # found available key
            self._rr_index[service] = (idx + 1) % n
            ke.last_used = time.monotonic()
            return idx, ke
        return None, None

    # ---------- public API: reserve_key (async context manager) ----------
    @contextlib.asynccontextmanager
    async def reserve_key(self, service: str):
        """Async context manager that reserves a key for exclusive use.
           Usage:
               async with manager.reserve_key("serpapi") as (idx, key_value):
                   # use the key
           Raises RuntimeError if no key available.
        """
        # Step 1: pick a key (under global lock)
        async with self._lock:
            idx, ke = await self._pick_active_key(service)
            if ke is None:
                raise RuntimeError(f"No available keys for service: {service}")

        # Step 2: acquire the key's lock and increment in_use
        async with ke.lock:
            ke.in_use += 1

        try:
            yield (idx, ke.value)
        finally:
            # Step 3: decrement in_use under the key's lock
            async with ke.lock:
                ke.in_use = max(0, ke.in_use - 1)
                # If there was a pending exhaustion, apply it now that no one is using the key
                if ke._pending_exhaust and ke.in_use == 0:
                    ke.exhausted_until = ke._pending_exhaust_until
                    ke._pending_exhaust = False
                    ke._pending_exhaust_until = None
                    await self._persist_exhaustion()
                    logger.debug("Applied pending exhaustion for key", extra={"service": service, "index": idx})
                    # Notify listeners that exhaustion was applied
                    asyncio.create_task(self._notify_listeners("key_exhausted", {
                        "service": service,
                        "index": idx,
                        "reason": "(pending applied)",
                        "until": datetime.fromtimestamp(ke.exhausted_until, tz=UTC).isoformat(),
                        "pending": False
                    }))
                # If there is a pending clear, notify now that usage has ended
                if ke._pending_clear and ke.in_use == 0:
                    ke._pending_clear = False
                    asyncio.create_task(self._notify_listeners("key_no_longer_in_use", {
                        "service": service,
                        "index": idx
                    }))

    # ---------- legacy: simple get_key (does not reserve) ----------
    async def get_key(self, service: str) -> Tuple[Optional[str], Optional[int]]:
        """Legacy method: returns (key_value, index) for the first available key,
           without incrementing in_use. Prefer reserve_key() for new code."""
        async with self._lock:
            entries = self._keys.get(service)
            if not entries:
                return None, None
            now = _now_ts()
            for idx, ke in enumerate(entries):
                if ke.exhausted_until and ke.exhausted_until > now:
                    continue
                if ke._pending_clear:
                    continue
                return ke.value, idx
            return None, None

    # ---------- exhaustion marking ----------
    async def mark_exhausted(self, service: str, idx: int, reason: str = "",
                             reset_at: Optional[Union[datetime, float]] = None,
                             until: Optional[float] = None):
        """Mark a key as exhausted until a specific time (reset_at) or until a policy-based default.
           If the key is currently in use, the exhaustion is recorded as pending and will be applied
           after all users release the key.
           reset_at can be a datetime (aware) or a float epoch timestamp.
           until is an alias for reset_at used by airline_api.
           The state is persisted when exhaustion actually takes effect.
        """
        # 'until' is an alias for reset_at
        if until is not None and reset_at is None:
            reset_at = until

        persist_state = None  # will hold snapshot if we need to write disk

        async with self._lock:
            try:
                ke = self._keys[service][idx]
            except (KeyError, IndexError):
                logger.warning("mark_exhausted: invalid service/index")
                return

            # Determine exhaustion timestamp
            if reset_at is not None:
                if isinstance(reset_at, datetime):
                    if reset_at.tzinfo is None:
                        reset_at = reset_at.replace(tzinfo=UTC)
                    until_ts = reset_at.timestamp()
                else:
                    until_ts = float(reset_at)
            else:
                until_ts = _exhaustion_ttl_for_error(service, reason)

            until_iso = datetime.fromtimestamp(until_ts, tz=UTC).isoformat()

            if ke.in_use > 0:
                # Defer exhaustion until key is released
                ke._pending_exhaust = True
                ke._pending_exhaust_until = until_ts
                logger.info("Key marked exhausted (pending)", extra={
                    "service": service, "index": idx, "until": until_iso, "reason": reason
                })
                asyncio.create_task(self._notify_listeners("key_exhausted", {
                    "service": service,
                    "index": idx,
                    "reason": reason,
                    "until": until_iso,
                    "pending": True
                }))
            else:
                # Apply immediately
                ke.exhausted_until = until_ts

                # Build disk snapshot NOW while lock is held (avoids re-entering lock)
                persist_state = {}
                for svc, entries in self._keys.items():
                    svc_state = {}
                    for e in entries:
                        if e.exhausted_until is not None:
                            svc_state[e.fingerprint] = {"exhausted_until": e.exhausted_until}
                    if svc_state:
                        persist_state[svc] = svc_state

                logger.info("Key marked exhausted", extra={
                    "service": service, "index": idx, "until": until_iso, "reason": reason
                })
                asyncio.create_task(self._notify_listeners("key_exhausted", {
                    "service": service,
                    "index": idx,
                    "reason": reason,
                    "until": until_iso,
                    "pending": False
                }))

        # ── Outside the lock: write disk safely, no deadlock ──
        if persist_state is not None:
            self._write_state_file(persist_state)

    # ---------- clear exhaustion / pending clear ----------
    async def clear_exhausted(self, service: str, idx: int):
        """Manually re-enable a key by clearing its exhaustion flag (and any pending flag)."""
        async with self._lock:
            try:
                ke = self._keys[service][idx]
                changed = False
                if ke.exhausted_until is not None:
                    ke.exhausted_until = None
                    changed = True
                if ke._pending_exhaust:
                    ke._pending_exhaust = False
                    ke._pending_exhaust_until = None
                    changed = True
                if changed:
                    await self._persist_exhaustion()
                    logger.info("Key cleared from exhausted state", extra={"service": service, "index": idx})
            except (KeyError, IndexError):
                pass

    async def mark_key_pending_clear(self, service: str, idx: int):
        """Mark a key for later cleanup (e.g., remove from rotation). When the key is no longer in use,
           the manager will trigger a 'key_no_longer_in_use' event so listeners can act."""
        async with self._lock:
            try:
                ke = self._keys[service][idx]
                if not ke._pending_clear:
                    ke._pending_clear = True
                    logger.info("Key marked pending clear", extra={"service": service, "index": idx})
            except (KeyError, IndexError):
                pass

    async def clear_pending_flag(self, service: str, idx: int):
        """Manually clear the pending_clear flag (if set)."""
        async with self._lock:
            try:
                ke = self._keys[service][idx]
                if ke._pending_clear:
                    ke._pending_clear = False
                    logger.info("Cleared pending_clear flag", extra={"service": service, "index": idx})
            except (KeyError, IndexError):
                pass

    # ---------- in-use query helpers ----------
    async def get_in_use_count(self, service: str, index: int) -> int:
        """Return the current number of reservations for a specific key."""
        async with self._lock:
            try:
                return self._keys[service][index].in_use
            except (KeyError, IndexError):
                return 0

    async def is_any_key_in_use(self, service: str) -> bool:
        """Return True if any key of the given service is currently reserved."""
        async with self._lock:
            entries = self._keys.get(service, [])
            return any(ke.in_use > 0 for ke in entries)

    async def wait_for_no_usage(self, service: str, index: int, timeout: float = 10.0) -> bool:
        """Wait until the specified key has zero in-use count, or until timeout.
           Returns True if usage reached zero, False on timeout."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            async with self._lock:
                try:
                    if self._keys[service][index].in_use == 0:
                        return True
                except (KeyError, IndexError):
                    return True  # key doesn't exist, consider done
            await asyncio.sleep(0.1)
        return False

    # ---------- debug status ----------
    async def status(self) -> Dict[str, List[Dict]]:
        """Return safe debug info: per service a list of dicts with index, active, in_use, exhausted_until."""
        async with self._lock:
            out = {}
            for svc, entries in self._keys.items():
                lst = []
                for i, ke in enumerate(entries):
                    active = not (ke.exhausted_until and ke.exhausted_until > _now_ts())
                    exhausted_iso = None
                    if ke.exhausted_until:
                        exhausted_iso = datetime.fromtimestamp(ke.exhausted_until, tz=UTC).isoformat()
                    lst.append({
                        "index": i,
                        "fingerprint": ke.fingerprint,
                        "active": active,
                        "in_use": ke.in_use,
                        "exhausted_until": exhausted_iso,
                        "pending_exhaust": ke._pending_exhaust,
                        "pending_clear": ke._pending_clear,
                    })
                out[svc] = lst
            return out

    # ---------- public aliases (used by app.py and health.py) ----------
    async def load_env_keys(self):
        """Force-reload keys from environment. Called at startup."""
        await self._reload_env_keys_if_changed()

    async def get_status(self) -> Dict[str, List[Dict]]:
        """Public alias for status(). Used by health.py routes."""
        return await self.status()

    async def refresh_from_env(self, sync: bool = False):
        """Reload keys from .env. Used by health endpoints."""
        await self._reload_env_keys_if_changed()

    async def record_usage(self, service: str, idx: int):
        """
        Record that a key was successfully used.
        Updates last_used timestamp. This is a lightweight tracking call —
        it does not increment in_use (that is handled by reserve_key).
        """
        async with self._lock:
            try:
                ke = self._keys[service][idx]
                ke.last_used = time.monotonic()
            except (KeyError, IndexError):
                logger.debug("record_usage: unknown service/index %s:%s", service, idx)

# Global singleton instance
key_manager = APIKeyManager()
