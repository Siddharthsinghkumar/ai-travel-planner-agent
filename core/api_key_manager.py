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
STATE_LOCKFILE_PATH = get_env_str("KEY_STATE_LOCKFILE", "/tmp/llm_key_state.lock")
# How often to refresh environment variables in the background
REFRESH_INTERVAL = get_env_int("KEY_REFRESH_INTERVAL", 30)  # seconds
# Lockfile path for multi‑process safety (override via env)
LOCKFILE_PATH = get_env_str("KEY_REFRESH_LOCKFILE", "/tmp/llm_key_refresh.lock")

# policy mapping for fallback exhaustion durations
POLICIES = {
    "serpapi": "monthly",
    "openai": "credit",
    "anthropic": "credit",
    "gemini": "daily",
    "weather": "daily"
}

RATE_LIMIT_COOLDOWN_SECONDS = max(30, get_env_int("KEY_RATE_LIMIT_COOLDOWN_SECONDS", 3600))
TRANSIENT_COOLDOWN_SECONDS = max(5, get_env_int("KEY_TRANSIENT_COOLDOWN_SECONDS", 300))
CIRCUIT_OPEN_COOLDOWN_SECONDS = max(5, get_env_int("KEY_CIRCUIT_OPEN_COOLDOWN_SECONDS", 120))
AUTH_FAILURE_COOLDOWN_SECONDS = max(300, get_env_int("KEY_AUTH_FAILURE_COOLDOWN_SECONDS", 86400))

# patterns to find numbered keys in env
ENV_PATTERNS = {
    "serpapi": re.compile(r"SERPAPI_KEY_(\d+)"),
    "openai": re.compile(r"OPENAI_KEY_(\d+)"),
    "gemini": re.compile(r"GEMINI_KEY_(\d+)"),
    "anthropic": re.compile(r"ANTHROPIC_KEY_(\d+)"),
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


def _normalize_reason_class(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "unknown"

    if any(token in text for token in ("circuit_open", "circuit breaker open", "circuit breaker")):
        return "circuit_open"
    if any(token in text for token in ("unauthorized", "invalid_key", "invalid api key", "forbidden", "access denied", "401", "403")):
        return "auth"
    if any(token in text for token in ("quota", "insufficient_quota", "billing", "payment required", "plan_searches_left")):
        return "quota"
    if any(token in text for token in ("rate_limit", "rate limit", "ratelimit", "too many requests", "429", "http_429")):
        return "rate_limit"
    if any(token in text for token in ("timeout", "timed out", "network", "connect", "temporar", "transient", "unavailable", "http_5xx", "5xx", "503", "502", "504")):
        return "transient"
    return "unknown"


def _future_ts_or_none(value: Optional[Union[float, int]], now_ts: float) -> Optional[float]:
    if value is None:
        return None
    try:
        ts = float(value)
    except Exception:
        return None
    if ts <= now_ts:
        return None
    return ts


def _merge_exhaustion_until(
    current_until: Optional[float],
    candidate_until: Optional[float],
    now_ts: float,
) -> Optional[float]:
    """
    Keep exhaustion timestamps monotonic for active quarantines.
    Never shortens an active exhaustion window due to a later weaker signal.
    """
    current_future = _future_ts_or_none(current_until, now_ts)
    candidate_future = _future_ts_or_none(candidate_until, now_ts)
    if current_future and candidate_future:
        return max(current_future, candidate_future)
    return current_future or candidate_future

def _exhaustion_ttl_for_error(service: str, reason: str) -> float:
    """Return an appropriate exhaustion timestamp based on the error reason."""
    now_dt = _now()
    reason_class = _normalize_reason_class(reason)
    if reason_class == "auth":
        # Auth failures are usually configuration/key issues and should not flap rapidly.
        return (now_dt + timedelta(seconds=AUTH_FAILURE_COOLDOWN_SECONDS)).timestamp()
    if reason_class == "rate_limit":
        return (now_dt + timedelta(seconds=RATE_LIMIT_COOLDOWN_SECONDS)).timestamp()
    if reason_class == "quota":
        # Quota tracks service policy horizon (daily/monthly/etc).
        return _compute_exhaustion_until(service)
    if reason_class == "circuit_open":
        return (now_dt + timedelta(seconds=CIRCUIT_OPEN_COOLDOWN_SECONDS)).timestamp()
    if reason_class == "transient":
        return (now_dt + timedelta(seconds=TRANSIENT_COOLDOWN_SECONDS)).timestamp()
    # Unknown defaults to service policy.
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


@contextlib.contextmanager
def _exclusive_state_file_lock(lockfile_path: str):
    """
    Cross-process lock for state file read/write operations.
    Keeps persistence updates serialized across workers.
    """
    fd = None
    try:
        fd = os.open(lockfile_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(fd)
            except Exception:
                pass

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
    last_exhausted_reason: Optional[str] = None
    last_exhausted_reason_class: Optional[str] = None
    last_exhausted_at: Optional[float] = None
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
        self._last_env_scan_meta: Dict[str, Any] = {}

        # ensure state directory exists
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # initial load: first from state file, then from environment
        self._load_initial_state()
        # No auto-start of refresh loop – caller must start explicitly

    def _snapshot_exhaustion_state_locked(self) -> Dict[str, Any]:
        now_ts = _now_ts()
        state: Dict[str, Any] = {}
        for service, entries in self._keys.items():
            svc_state = {}
            for ke in entries:
                until_ts = _future_ts_or_none(ke.exhausted_until, now_ts)
                if until_ts is None:
                    continue
                svc_state[ke.fingerprint] = {"exhausted_until": until_ts}
            if svc_state:
                state[service] = svc_state
        return state

    def _sweep_key_invariants_locked(self) -> bool:
        """
        Normalize key state for long-lived stability.
        Returns True when any in-memory state changed.
        Must be called with self._lock held.
        """
        changed = False
        now_ts = _now_ts()

        for service, entries in self._keys.items():
            for idx, ke in enumerate(entries):
                if ke.in_use < 0:
                    logger.warning(
                        "Corrected negative in_use count",
                        extra={"service": service, "index": idx, "in_use_before": ke.in_use},
                    )
                    ke.in_use = 0
                    changed = True

                normalized_exhausted = _future_ts_or_none(ke.exhausted_until, now_ts)
                if normalized_exhausted != ke.exhausted_until:
                    ke.exhausted_until = normalized_exhausted
                    changed = True

                if ke._pending_exhaust:
                    pending_until = _future_ts_or_none(ke._pending_exhaust_until, now_ts)
                    if pending_until is None:
                        ke._pending_exhaust = False
                        ke._pending_exhaust_until = None
                        changed = True
                    elif ke.in_use == 0:
                        merged = _merge_exhaustion_until(ke.exhausted_until, pending_until, now_ts)
                        if merged != ke.exhausted_until:
                            ke.exhausted_until = merged
                        ke._pending_exhaust = False
                        ke._pending_exhaust_until = None
                        changed = True
                elif ke._pending_exhaust_until is not None:
                    ke._pending_exhaust_until = None
                    changed = True

        for service, entries in self._keys.items():
            if entries:
                self._rr_index[service] = int(self._rr_index.get(service, 0)) % len(entries)
            else:
                self._rr_index[service] = 0

        return changed

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
            with _exclusive_state_file_lock(STATE_LOCKFILE_PATH):
                if not STATE_FILE.exists():
                    return {}
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            logger.exception("Failed to read key state file; starting with empty state")
            return {}

    def _write_state_file(self, state: Dict[str, Any]):
        """Atomically write state (fingerprint -> exhausted_until) to disk, with secure perms."""
        tmp = STATE_FILE.with_name(f"{STATE_FILE.name}.tmp.{os.getpid()}")
        try:
            with _exclusive_state_file_lock(STATE_LOCKFILE_PATH):
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
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    async def _persist_exhaustion(self):
        """Snapshot current exhaustion and write to disk safely.
           Snapshot is taken while holding self._lock, then the write happens
           without holding the lock (to avoid blocking other ops)."""
        # Build snapshot under global lock
        async with self._lock:
            self._sweep_key_invariants_locked()
            state = self._snapshot_exhaustion_state_locked()
        # write file (sync) outside lock
        self._write_state_file(state)

    def _load_initial_state(self):
        """Load persisted exhaustion data and merge with environment (blocking, called at init)."""
        persisted = self._load_state_file()
        now_ts = _now_ts()
        # Parse environment
        env_keys = self._parse_env_keys()
        if self._last_env_scan_meta:
            logger.info(
                "API key ingest snapshot",
                extra=self._last_env_scan_meta,
            )
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
                    exhausted = _future_ts_or_none(svc_state[fp].get("exhausted_until"), now_ts)
                entries.append(KeyEntry(value=key, fingerprint=fp, exhausted_until=exhausted))
            self._keys[service] = entries
            self._rr_index[service] = 0

    # ---------- environment parsing ----------
    def _parse_env_keys(self) -> Dict[str, List[str]]:
        """Read merged key config and return service -> key list (in index order).

        Merge strategy:
        1) .env file values (if present)
        2) runtime os.environ overrides .env for matching names

        This keeps key discovery consistent with runtime env injection while retaining
        .env compatibility for local development.
        """
        # Use cwd-based lookup so we read the active deployment/project .env.
        env_path = find_dotenv(usecwd=True)
        file_env: Dict[str, Any] = {}
        if env_path:
            file_env = dotenv_values(env_path)  # reads file without mutating os.environ
        runtime_env: Dict[str, str] = dict(os.environ)
        merged_env: Dict[str, Any] = dict(file_env)
        merged_env.update(runtime_env)  # runtime env wins

        services = {}
        source_counts: Dict[str, Dict[str, int]] = {
            service: {"runtime_env": 0, "dotenv": 0}
            for service in ENV_PATTERNS.keys()
        }
        for service, prog in ENV_PATTERNS.items():
            # collect keys with their numeric index
            indexed = []
            for name, value in merged_env.items():
                m = prog.fullmatch(name)
                if m and value and value.strip():
                    idx = int(m.group(1))
                    indexed.append((idx, value.strip()))
                    if name in runtime_env and runtime_env.get(name, "").strip():
                        source_counts[service]["runtime_env"] += 1
                    else:
                        source_counts[service]["dotenv"] += 1
            # sort by index and store values in order
            if indexed:
                indexed.sort(key=lambda x: x[0])
                services[service] = [v for _, v in indexed]
        self._last_env_scan_meta = {
            "config_source": "merged_env",
            "env_file_path": env_path or None,
            "service_key_counts": {svc: len(vals) for svc, vals in services.items()},
            "service_source_counts": source_counts,
        }
        logger.debug("Parsed keys from merged env config", extra=self._last_env_scan_meta)
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
            if self._lockfile_fd is not None:
                try:
                    fcntl.flock(self._lockfile_fd, fcntl.LOCK_UN)
                    os.close(self._lockfile_fd)
                except Exception:
                    logger.exception("Error releasing lockfile after loop-start failure")
                finally:
                    self._lockfile_fd = None
            return

        self._refresh_task = loop.create_task(self._refresh_loop())
        logger.info("Started API key refresh loop (interval=%ds)", interval_seconds)

    def stop_refresh_loop(self):
        """Stop the background refresh task and release the lockfile."""
        self._stop_refresh.set()
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
        """Periodically refresh keys from the merged runtime/.env environment."""
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

            # Treat index shifts as real changes. We compare index->fingerprint maps
            # directly (not only set equality) so listeners can evict stale idx caches.
            fingerprint_changed = (old_fingerprint_maps != new_fingerprint_maps)

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
            state_sanitized = self._sweep_key_invariants_locked()
            changed = bool(fingerprint_changed or state_sanitized)
            persist_state: Dict[str, Any] = {}
            if changed:
                persist_state = self._snapshot_exhaustion_state_locked()

        # Outside lock: write state file and notify if changed
        if changed:
            # Always write changed snapshots, including {}.
            # This prevents stale on-disk exhaustion state from lingering after clears.
            self._write_state_file(persist_state)
            if fingerprint_changed:
                logger.info(
                    "Key set changed – triggering callbacks",
                    extra=(self._last_env_scan_meta or None),
                )
                asyncio.create_task(self._notify_listeners("env_changed", {
                    "old_fingerprint_maps": old_fingerprint_maps,
                    "new_fingerprint_maps": new_fingerprint_maps,
                    "affected": affected,
                }))
            else:
                logger.debug("Key state normalized without keyset fingerprint changes")

        return bool(fingerprint_changed)

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
        persist_state: Optional[Dict[str, Any]] = None
        pending_exhaust_event: Optional[dict] = None
        key_released_event: Optional[dict] = None

        # Step 1: sweep invariants and pick a key (under global lock)
        async with self._lock:
            if self._sweep_key_invariants_locked():
                persist_state = self._snapshot_exhaustion_state_locked()
            idx, ke = await self._pick_active_key(service)
            if ke is None:
                raise RuntimeError(f"No available keys for service: {service}")

        if persist_state is not None:
            self._write_state_file(persist_state)

        # Step 2: acquire the key's lock and increment in_use
        async with ke.lock:
            ke.in_use += 1

        try:
            yield (idx, ke.value)
        finally:
            # Step 3: decrement in_use under the key's lock
            async with ke.lock:
                ke.in_use -= 1
                if ke.in_use < 0:
                    logger.warning(
                        "reserve_key release corrected negative in_use",
                        extra={"service": service, "index": idx, "in_use_before_correction": ke.in_use},
                    )
                    ke.in_use = 0
                # If there was a pending exhaustion, apply it now that no one is using the key
                if ke._pending_exhaust and ke.in_use == 0:
                    now_ts = _now_ts()
                    ke.exhausted_until = _merge_exhaustion_until(
                        ke.exhausted_until,
                        ke._pending_exhaust_until,
                        now_ts,
                    )
                    ke._pending_exhaust = False
                    ke._pending_exhaust_until = None
                    if ke.exhausted_until is not None:
                        pending_exhaust_event = {
                            "service": service,
                            "index": idx,
                            "reason": ke.last_exhausted_reason or "(pending applied)",
                            "reason_class": ke.last_exhausted_reason_class or "unknown",
                            "until": datetime.fromtimestamp(ke.exhausted_until, tz=UTC).isoformat(),
                            "pending": False,
                        }
                        logger.debug(
                            "Applied pending exhaustion for key",
                            extra={"service": service, "index": idx, "reason_class": pending_exhaust_event["reason_class"]},
                        )
                # If there is a pending clear, notify now that usage has ended
                if ke._pending_clear and ke.in_use == 0:
                    ke._pending_clear = False
                    key_released_event = {
                        "service": service,
                        "index": idx,
                    }

            if pending_exhaust_event is not None:
                async with self._lock:
                    self._sweep_key_invariants_locked()
                    persist_state = self._snapshot_exhaustion_state_locked()
                self._write_state_file(persist_state)
                asyncio.create_task(self._notify_listeners("key_exhausted", pending_exhaust_event))

            if key_released_event is not None:
                asyncio.create_task(self._notify_listeners("key_no_longer_in_use", key_released_event))

    # ---------- legacy: simple get_key (does not reserve) ----------
    async def get_key(self, service: str) -> Tuple[Optional[str], Optional[int]]:
        """Legacy method: returns (key_value, index) for the first available key,
           without incrementing in_use. Prefer reserve_key() for new code."""
        persist_state: Optional[Dict[str, Any]] = None
        result: Tuple[Optional[str], Optional[int]] = (None, None)
        async with self._lock:
            if self._sweep_key_invariants_locked():
                persist_state = self._snapshot_exhaustion_state_locked()
            entries = self._keys.get(service)
            if not entries:
                result = (None, None)
            else:
                now = _now_ts()
                for idx, ke in enumerate(entries):
                    if ke.exhausted_until and ke.exhausted_until > now:
                        continue
                    if ke._pending_clear:
                        continue
                    result = (ke.value, idx)
                    break
        if persist_state is not None:
            self._write_state_file(persist_state)
        return result

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
        now_ts = _now_ts()
        reason_class = _normalize_reason_class(reason)

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

            merged_until = _merge_exhaustion_until(ke.exhausted_until, until_ts, now_ts)
            pending_merged_until = _merge_exhaustion_until(ke._pending_exhaust_until, until_ts, now_ts)
            requested_until_iso = datetime.fromtimestamp(until_ts, tz=UTC).isoformat()
            ke.last_exhausted_reason = reason
            ke.last_exhausted_reason_class = reason_class
            ke.last_exhausted_at = now_ts

            if ke.in_use > 0:
                # Defer exhaustion until key is released
                ke._pending_exhaust = pending_merged_until is not None
                ke._pending_exhaust_until = pending_merged_until
                pending_until_iso = (
                    datetime.fromtimestamp(pending_merged_until, tz=UTC).isoformat()
                    if pending_merged_until is not None
                    else requested_until_iso
                )
                logger.info("Key marked exhausted (pending)", extra={
                    "service": service, "index": idx, "until": pending_until_iso, "reason": reason, "reason_class": reason_class
                })
                asyncio.create_task(self._notify_listeners("key_exhausted", {
                    "service": service,
                    "index": idx,
                    "reason": reason,
                    "reason_class": reason_class,
                    "until": pending_until_iso,
                    "pending": True
                }))
            else:
                # Apply immediately
                ke.exhausted_until = merged_until
                effective_until_iso = (
                    datetime.fromtimestamp(ke.exhausted_until, tz=UTC).isoformat()
                    if ke.exhausted_until is not None
                    else requested_until_iso
                )

                logger.info(
                    "Key marked exhausted",
                    extra={
                        "service": service,
                        "index": idx,
                        "until": effective_until_iso,
                        "reason": reason,
                        "reason_class": reason_class,
                        "existing_extended": bool(
                            _future_ts_or_none(ke.exhausted_until, now_ts)
                            and _future_ts_or_none(until_ts, now_ts)
                            and ke.exhausted_until != until_ts
                        ),
                    },
                )
                asyncio.create_task(self._notify_listeners("key_exhausted", {
                    "service": service,
                    "index": idx,
                    "reason": reason,
                    "reason_class": reason_class,
                    "until": effective_until_iso,
                    "pending": False
                }))

            self._sweep_key_invariants_locked()
            persist_state = self._snapshot_exhaustion_state_locked()

        # ── Outside the lock: write disk safely, no deadlock ──
        if persist_state is not None:
            self._write_state_file(persist_state)

    # ---------- clear exhaustion / pending clear ----------
    async def clear_exhausted(self, service: str, idx: int):
        """Manually re-enable a key by clearing its exhaustion flag (and any pending flag)."""
        persist_state: Optional[Dict[str, Any]] = None
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
                    self._sweep_key_invariants_locked()
                    persist_state = self._snapshot_exhaustion_state_locked()
                    logger.info("Key cleared from exhausted state", extra={"service": service, "index": idx})
            except (KeyError, IndexError):
                pass
        if persist_state is not None:
            self._write_state_file(persist_state)

    async def mark_key_pending_clear(self, service: str, idx: int):
        """Mark a key for later cleanup (e.g., remove from rotation). When the key is no longer in use,
           the manager will trigger a 'key_no_longer_in_use' event so listeners can act."""
        release_event: Optional[dict] = None
        async with self._lock:
            try:
                ke = self._keys[service][idx]
                if not ke._pending_clear:
                    ke._pending_clear = True
                    logger.info("Key marked pending clear", extra={"service": service, "index": idx})
                if ke.in_use == 0:
                    # Do not leave a ghost pending flag when no active holders exist.
                    ke._pending_clear = False
                    release_event = {"service": service, "index": idx}
            except (KeyError, IndexError):
                pass
        if release_event is not None:
            asyncio.create_task(self._notify_listeners("key_no_longer_in_use", release_event))

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
        persist_state: Optional[Dict[str, Any]] = None
        async with self._lock:
            if self._sweep_key_invariants_locked():
                persist_state = self._snapshot_exhaustion_state_locked()
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
                        "last_exhausted_reason_class": ke.last_exhausted_reason_class,
                        "last_exhausted_at": (
                            datetime.fromtimestamp(ke.last_exhausted_at, tz=UTC).isoformat()
                            if ke.last_exhausted_at
                            else None
                        ),
                    })
                out[svc] = lst
        if persist_state is not None:
            self._write_state_file(persist_state)
        return out

    # ---------- public aliases (used by app.py and health.py) ----------
    async def load_env_keys(self):
        """Force-reload keys from environment. Called at startup."""
        await self._reload_env_keys_if_changed()

    async def get_status(self) -> Dict[str, List[Dict]]:
        """Public alias for status(). Used by health.py routes."""
        return await self.status()

    async def refresh_from_env(self, sync: bool = False):
        """Reload keys from merged runtime/.env environment. Used by health endpoints."""
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
