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
from core.env_config import get_env_int, get_env_str, get_env_float
import core.metrics as app_metrics
from cachetools import TTLCache

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
    # OpenWeather limits vary by product-plan; keep default fallback short/manual-first.
    "weather": "provider_plan"
}

SERPAPI_UNKNOWN_RESET_DEFERRAL_SECONDS = max(
    300,
    get_env_int("SERPAPI_UNKNOWN_RESET_DEFERRAL_SECONDS", 7 * 24 * 3600),
)
KEY_RESERVATION_WAIT_SECONDS = max(
    0.0,
    get_env_float("KEY_RESERVATION_WAIT_SECONDS", 15.0),
)
KEY_RESERVATION_POLL_SECONDS = max(
    0.01,
    get_env_float("KEY_RESERVATION_POLL_SECONDS", 0.05),
)

RATE_LIMIT_COOLDOWN_SECONDS = max(30, get_env_int("KEY_RATE_LIMIT_COOLDOWN_SECONDS", 3600))
TRANSIENT_COOLDOWN_SECONDS = max(5, get_env_int("KEY_TRANSIENT_COOLDOWN_SECONDS", 300))
CIRCUIT_OPEN_COOLDOWN_SECONDS = max(5, get_env_int("KEY_CIRCUIT_OPEN_COOLDOWN_SECONDS", 120))
AUTH_FAILURE_COOLDOWN_SECONDS = max(300, get_env_int("KEY_AUTH_FAILURE_COOLDOWN_SECONDS", 86400))
SERPAPI_ACCOUNT_RECONCILE_INTERVAL_SECONDS = max(
    300,
    get_env_int("SERPAPI_ACCOUNT_RECONCILE_INTERVAL_SECONDS", 1800),
)
SERPAPI_ACCOUNT_RECONCILE_TIMEOUT_SECONDS = max(
    2.0,
    get_env_float("SERPAPI_ACCOUNT_RECONCILE_TIMEOUT_SECONDS", 4.0),
)
SERPAPI_RECONCILE_STALE_SECONDS = max(
    300,
    get_env_int("SERPAPI_RECONCILE_STALE_SECONDS", 21600),
)
SERPAPI_RECONCILE_GRACE_SECONDS = max(
    30,
    get_env_int("SERPAPI_RECONCILE_GRACE_SECONDS", 300),
)
PROVIDER_OVERRIDE_CACHE_SECONDS = max(
    3,
    get_env_int("PROVIDER_OVERRIDE_CACHE_SECONDS", 10),
)
WEATHER_PROVIDER_POLICY_FALLBACK_SECONDS = max(
    300,
    get_env_int("WEATHER_PROVIDER_POLICY_FALLBACK_SECONDS", 3600),
)

OVERRIDE_SCOPE_KEY = "key"
OVERRIDE_SCOPE_PROVIDER_ACCOUNT = "provider_account"
OVERRIDE_SCOPE_PROJECT = "project"
VALID_OVERRIDE_SCOPES = {
    OVERRIDE_SCOPE_KEY,
    OVERRIDE_SCOPE_PROVIDER_ACCOUNT,
    OVERRIDE_SCOPE_PROJECT,
}

OVERRIDE_FORCE_EXHAUSTED_UNTIL = "force_exhausted_until"
OVERRIDE_CLEAR_EXHAUSTION = "clear_exhaustion"
OVERRIDE_FORCE_ACTIVE_UNTIL = "force_active_until"
OVERRIDE_SKIP_RECONCILE_UNTIL = "skip_reconcile_until"
VALID_OVERRIDE_TYPES = {
    OVERRIDE_FORCE_EXHAUSTED_UNTIL,
    OVERRIDE_CLEAR_EXHAUSTION,
    OVERRIDE_FORCE_ACTIVE_UNTIL,
    OVERRIDE_SKIP_RECONCILE_UNTIL,
}

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
    elif policy == "provider_plan":
        reset_dt = now_dt + timedelta(seconds=WEATHER_PROVIDER_POLICY_FALLBACK_SECONDS)
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
    key_name: Optional[str] = None
    name_fingerprint: Optional[str] = None
    searches_left: Optional[int] = None
    expected_reset_basis: Optional[str] = None
    expected_reset_at: Optional[float] = None
    last_checked_at: Optional[float] = None
    last_provider_error: Optional[str] = None
    last_provider_reason: Optional[str] = None
    failure_classification: Optional[str] = None
    retry_after: Optional[float] = None
    default_reset_day: Optional[int] = None
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
        self._serpapi_reconcile_task: Optional[asyncio.Task] = None
        self._stop_serpapi_reconcile = asyncio.Event()
        self._serpapi_force_reconcile_name_fps: set[str] = set()
        self._serpapi_reconcile_meta: Dict[str, Any] = {
            "last_started_at": None,
            "last_completed_at": None,
            "last_status": "never_run",
            "last_error": None,
            "last_checked": 0,
            "last_skipped": 0,
            "last_forced": 0,
        }
        self._provider_state_schema_ready: bool = False
        PROVIDER_OVERRIDE_CACHE_MAXSIZE = 100  # max providers to cache
        self._provider_override_cache: TTLCache = TTLCache(
            maxsize=PROVIDER_OVERRIDE_CACHE_MAXSIZE,
            ttl=PROVIDER_OVERRIDE_CACHE_SECONDS,
        )
        self._exhaustion_dedup: Dict[str, float] = {}  # dedup key -> last timestamp

        # ensure state directory exists
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # initial load: first from state file, then from environment
        self._load_initial_state()
        # No auto-start of refresh loop – caller must start explicitly

    def _snapshot_exhaustion_state_locked(self) -> Dict[str, Any]:
        # Legacy JSON state snapshots are retired. Keep method for compatibility with
        # existing call-sites/tests, but canonical persistence is DB-backed provider_key_states.
        return {}

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
                    app_metrics.record_key_state_event(
                        service=service,
                        event="in_use_corrected",
                        reason_class=ke.last_exhausted_reason_class or "unknown",
                    )
                    ke.in_use = 0
                    changed = True

                normalized_exhausted = _future_ts_or_none(ke.exhausted_until, now_ts)
                if normalized_exhausted != ke.exhausted_until:
                    if ke.exhausted_until is not None and normalized_exhausted is None:
                        app_metrics.record_key_state_event(
                            service=service,
                            event="cooldown_expired",
                            reason_class=ke.last_exhausted_reason_class or "unknown",
                        )
                    ke.exhausted_until = normalized_exhausted
                    changed = True

                normalized_retry_after = _future_ts_or_none(ke.retry_after, now_ts)
                if normalized_retry_after is None and normalized_exhausted is not None:
                    normalized_retry_after = normalized_exhausted
                if normalized_retry_after != ke.retry_after:
                    ke.retry_after = normalized_retry_after
                    changed = True

                if ke._pending_exhaust:
                    pending_until = _future_ts_or_none(ke._pending_exhaust_until, now_ts)
                    if pending_until is None:
                        app_metrics.record_key_state_event(
                            service=service,
                            event="pending_exhaust_cleared",
                            reason_class=ke.last_exhausted_reason_class or "unknown",
                        )
                        ke._pending_exhaust = False
                        ke._pending_exhaust_until = None
                        changed = True
                    elif ke.in_use == 0:
                        merged = _merge_exhaustion_until(ke.exhausted_until, pending_until, now_ts)
                        if merged != ke.exhausted_until:
                            ke.exhausted_until = merged
                            ke.retry_after = merged
                        app_metrics.record_key_state_event(
                            service=service,
                            event="pending_exhaust_applied",
                            reason_class=ke.last_exhausted_reason_class or "unknown",
                        )
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

    @staticmethod
    def _fingerprint_name(name: str) -> str:
        return hashlib.sha256(str(name or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _classify_serpapi_failure(reason: str, reason_class: Optional[str] = None) -> str:
        normalized_reason_class = reason_class or _normalize_reason_class(reason)
        lowered = str(reason or "").strip().lower()
        if normalized_reason_class == "quota":
            return "monthly_quota"
        if normalized_reason_class == "rate_limit":
            return "rate_limit"
        if normalized_reason_class == "auth":
            return "auth"
        if normalized_reason_class == "transient":
            return "transient"
        if "account_reconcile" in lowered:
            return "account_reconcile"
        return "unknown"

    def _serpapi_expected_reset_basis(
        self,
        *,
        reason: str,
        reason_class: str,
        has_reset_at: bool,
        from_account: bool = False,
    ) -> str:
        if from_account and has_reset_at:
            return "account_inferred_cycle_boundary"
        if from_account:
            return "account_reconcile_without_reset_timestamp"
        if reason_class == "quota" and has_reset_at:
            return "provider_or_policy_reset_at"
        if reason_class == "quota":
            return "policy_inferred_cycle_boundary"
        if reason_class == "rate_limit":
            return "transient_backoff_window"
        if reason_class == "auth":
            return "auth_cooldown_window"
        if reason_class == "transient":
            return "transient_backoff_window"
        return "unknown"

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
        # Retained as a no-op compatibility shim.
        # Canonical persistence is provider_key_states via DB.
        _ = state

    async def _persist_exhaustion(self):
        """Persist current key state to canonical DB store (DB write runs off the event loop)."""
        snapshots = []
        async with self._lock:
            self._sweep_key_invariants_locked()
            snapshots = self._snapshot_all_entries_locked()
        for snap in snapshots:
            await self._persist_entry_snapshot_to_db(**snap)

    def _load_initial_state(self):
        """Load environment key slots and hydrate persisted provider state from DB."""
        env_records = self._parse_env_key_records()
        if self._last_env_scan_meta:
            logger.info(
                "API key ingest snapshot",
                extra=self._last_env_scan_meta,
            )
        legacy_state = self._load_state_file()

        # Build initial _keys from env. Durable state is hydrated from DB afterwards.
        self._keys.clear()
        for service, key_records in env_records.items():
            entries = []
            for record in key_records:
                key = str(record.get("value") or "")
                fp = self._fingerprint(key)
                key_name = str(record.get("name") or "")
                entries.append(
                    KeyEntry(
                        value=key,
                        fingerprint=fp,
                        exhausted_until=None,
                        key_name=key_name,
                        name_fingerprint=self._fingerprint_name(key_name),
                        default_reset_day=record.get("default_reset_day"),
                    )
                )
            self._keys[service] = entries
            self._rr_index[service] = 0
        if legacy_state:
            self._migrate_legacy_state_file_to_db(legacy_state)
        # DB hydration is deferred to explicit async startup/load_env_keys to keep module import light.

    # ---------- environment parsing ----------
    def _parse_env_key_records(self) -> Dict[str, List[Dict[str, Any]]]:
        """Read merged key config and return service -> key records (name/value/index).

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

        services: Dict[str, List[Dict[str, Any]]] = {}
        serpapi_reset_days: Dict[int, int] = {}
        for env_name, env_value in merged_env.items():
            m = re.fullmatch(r"SERPAPI_KEY_RESET_DAY_(\d+)", str(env_name or ""))
            if not m:
                continue
            raw_day = str(env_value or "").strip()
            if not raw_day:
                continue
            try:
                parsed_day = int(raw_day)
            except Exception:
                logger.warning(
                    "Ignoring invalid SerpAPI reset day override",
                    extra={"env_name": str(env_name), "raw_value": raw_day},
                )
                continue
            if parsed_day < 1 or parsed_day > 28:
                logger.warning(
                    "Ignoring out-of-range SerpAPI reset day override",
                    extra={"env_name": str(env_name), "raw_value": raw_day},
                )
                continue
            serpapi_reset_days[int(m.group(1))] = parsed_day
        source_counts: Dict[str, Dict[str, int]] = {
            service: {"runtime_env": 0, "dotenv": 0}
            for service in ENV_PATTERNS.keys()
        }
        for service, prog in ENV_PATTERNS.items():
            # collect keys with their numeric index
            indexed: List[Tuple[int, str, str]] = []
            for name, value in merged_env.items():
                m = prog.fullmatch(name)
                if m and value and value.strip():
                    idx = int(m.group(1))
                    indexed.append((idx, value.strip(), str(name)))
                    if name in runtime_env and runtime_env.get(name, "").strip():
                        source_counts[service]["runtime_env"] += 1
                    else:
                        source_counts[service]["dotenv"] += 1
            # sort by index and store values in order
            if indexed:
                indexed.sort(key=lambda x: x[0])
                service_entries = [
                    {"index": idx, "value": val, "name": env_name}
                    for idx, val, env_name in indexed
                ]
                if service == "serpapi":
                    for row in service_entries:
                        try:
                            row_idx = int(row.get("index") or 0)
                        except Exception:
                            row_idx = 0
                        if row_idx in serpapi_reset_days:
                            row["default_reset_day"] = int(serpapi_reset_days[row_idx])
                services[service] = service_entries
        self._last_env_scan_meta = {
            "config_source": "merged_env",
            "env_file_path": env_path or None,
            "service_key_counts": {svc: len(vals) for svc, vals in services.items()},
            "service_source_counts": source_counts,
        }
        logger.debug("Parsed keys from merged env config", extra=self._last_env_scan_meta)
        return services

    def _parse_env_keys(self) -> Dict[str, List[str]]:
        records = self._parse_env_key_records()
        return {
            service: [str(item.get("value") or "") for item in items]
            for service, items in records.items()
        }

    def _exhaustion_dedup_key(self, service: str, idx: int, reason_class: str) -> str:
        return f"{service}:{idx}:{reason_class}"

    def _should_log_exhaustion(self, dedup_key: str, now_ts: float) -> bool:
        ttl = max(30.0, get_env_float("KEY_EXHAUST_LOG_DEDUP_SECONDS", 300.0))
        previous = self._exhaustion_dedup.get(dedup_key)
        if previous is not None and (now_ts - previous) < ttl:
            return False
        self._exhaustion_dedup[dedup_key] = now_ts
        if len(self._exhaustion_dedup) > 512:
            stale_before = now_ts - (ttl * 2.0)
            for key in list(self._exhaustion_dedup.keys()):
                if self._exhaustion_dedup.get(key, now_ts) < stale_before:
                    self._exhaustion_dedup.pop(key, None)
        return True

    @staticmethod
    def _sanitize_reason_for_log(reason: str) -> str:
        text = str(reason or "")
        if len(text) > 200:
            text = text[:200] + "..."
        return text

    @contextlib.contextmanager
    def _provider_state_session(self):
        session = None
        try:
            if not self._provider_state_schema_ready:
                from agents.database import init_db
                init_db()
                self._provider_state_schema_ready = True
            from agents.database import SessionLocal
            session = SessionLocal()
        except Exception:
            logger.exception("provider_state_session_failed")
            yield None
            return

        try:
            yield session
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass

    @staticmethod
    def _datetime_to_ts(value: Optional[datetime]) -> Optional[float]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.timestamp()

    @staticmethod
    def _datetime_to_iso_utc(value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()

    @staticmethod
    def _override_until_semantics(override_type: Optional[str]) -> Optional[str]:
        normalized = str(override_type or "").strip().lower()
        if normalized == OVERRIDE_FORCE_EXHAUSTED_UNTIL:
            return "forces_exhaustion_until"
        if normalized == OVERRIDE_FORCE_ACTIVE_UNTIL:
            return "forces_active_until"
        if normalized == OVERRIDE_SKIP_RECONCILE_UNTIL:
            return "skips_reconcile_until"
        return None

    @staticmethod
    def _parse_override_active_until(raw_value: Optional[str]) -> Optional[datetime]:
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _invalidate_provider_override_cache(self, provider: Optional[str] = None) -> None:
        if provider:
            self._provider_override_cache.pop(str(provider).strip().lower(), None)
            return
        self._provider_override_cache.clear()

    def _load_active_provider_overrides(self, provider: str) -> List[Dict[str, Any]]:
        normalized_provider = str(provider or "").strip().lower()
        if not normalized_provider:
            return []
        now_utc = _now()
        with self._provider_state_session() as session:
            if session is None:
                return []
            try:
                from agents.database import ProviderStateOverride
                rows = (
                    session.query(ProviderStateOverride)
                    .filter(
                        ProviderStateOverride.provider == normalized_provider,
                        ProviderStateOverride.is_enabled.is_(True),
                    )
                    .all()
                )
            except Exception:
                logger.exception("provider_override_load_failed")
                return []
        out: List[Dict[str, Any]] = []
        for row in rows:
            active_until = row.active_until
            if isinstance(active_until, datetime):
                if active_until.tzinfo is None:
                    active_until = active_until.replace(tzinfo=UTC)
                if active_until < now_utc:
                    continue
            out.append(
                {
                    "id": int(row.id),
                    "provider": normalized_provider,
                    "scope_type": str(row.scope_type or ""),
                    "scope_identifier": row.scope_identifier,
                    "key_name_fingerprint": str(row.key_name_fingerprint or "") or None,
                    "key_value_fingerprint": str(row.key_value_fingerprint or "") or None,
                    "override_type": str(row.override_type or ""),
                    "active_until": active_until,
                    "note": row.note,
                    "is_enabled": bool(row.is_enabled),
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
            )
        return out

    def _provider_overrides_cached(self, provider: str) -> List[Dict[str, Any]]:
        normalized_provider = str(provider or "").strip().lower()
        if not normalized_provider:
            return []
        # TTLCache handles expiration and eviction automatically
        cached = self._provider_override_cache.get(normalized_provider)
        if cached is not None:
            return cached
        rows = self._load_active_provider_overrides(normalized_provider)
        self._provider_override_cache[normalized_provider] = rows
        return rows

    def _override_binding_matches_current_key(self, row: Dict[str, Any]) -> bool:
        scope_type = str((row or {}).get("scope_type") or "").strip().lower()
        if scope_type != OVERRIDE_SCOPE_KEY:
            return True
        provider = str((row or {}).get("provider") or "").strip().lower()
        if provider != "serpapi":
            return True
        name_fp = str((row or {}).get("key_name_fingerprint") or "").strip() or str((row or {}).get("scope_identifier") or "").strip()
        value_fp = str((row or {}).get("key_value_fingerprint") or "").strip()
        if not name_fp or not value_fp:
            return False
        for ke in self._keys.get("serpapi", []) or []:
            if str(ke.name_fingerprint or "") == name_fp and str(ke.fingerprint or "") == value_fp:
                return True
        return False

    def _override_applies_to_key(
        self,
        *,
        provider: str,
        override_row: Dict[str, Any],
        key_name_fingerprint: Optional[str],
        key_value_fingerprint: Optional[str],
    ) -> bool:
        normalized_provider = str(provider or "").strip().lower()
        scope_type = str((override_row or {}).get("scope_type") or "").strip().lower()
        scope_identifier = str((override_row or {}).get("scope_identifier") or "").strip()
        if scope_type == OVERRIDE_SCOPE_KEY:
            if normalized_provider == "serpapi":
                row_name_fp = str((override_row or {}).get("key_name_fingerprint") or "").strip() or scope_identifier
                row_value_fp = str((override_row or {}).get("key_value_fingerprint") or "").strip()
                return bool(
                    key_name_fingerprint
                    and key_value_fingerprint
                    and row_name_fp
                    and row_value_fp
                    and row_name_fp == key_name_fingerprint
                    and row_value_fp == key_value_fingerprint
                )
            return bool(key_name_fingerprint and scope_identifier and scope_identifier == key_name_fingerprint)
        if scope_type in {OVERRIDE_SCOPE_PROVIDER_ACCOUNT, OVERRIDE_SCOPE_PROJECT}:
            return True
        return False

    def _resolve_provider_override_effects(
        self,
        *,
        provider: str,
        key_name_fingerprint: Optional[str] = None,
        key_value_fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        rows = self._provider_overrides_cached(provider)
        now_ts = _now_ts()
        effect = {
            "clear_exhaustion": False,
            "force_exhausted_until": None,
            "force_active_until": None,
            "skip_reconcile_until": None,
            "override_ids": [],
        }
        for row in rows:
            if not self._override_applies_to_key(
                provider=provider,
                override_row=row,
                key_name_fingerprint=key_name_fingerprint,
                key_value_fingerprint=key_value_fingerprint,
            ):
                continue
            override_id = row.get("id")
            if override_id is not None:
                effect["override_ids"].append(int(override_id))
            override_type = str(row.get("override_type") or "").strip().lower()
            active_until = row.get("active_until")
            active_until_ts = self._datetime_to_ts(active_until) if isinstance(active_until, datetime) else None
            if override_type == OVERRIDE_CLEAR_EXHAUSTION:
                effect["clear_exhaustion"] = True
                continue
            if active_until_ts is None or active_until_ts <= now_ts:
                continue
            if override_type == OVERRIDE_FORCE_EXHAUSTED_UNTIL:
                current = effect.get("force_exhausted_until")
                effect["force_exhausted_until"] = max(float(current or 0.0), float(active_until_ts))
            elif override_type == OVERRIDE_FORCE_ACTIVE_UNTIL:
                current = effect.get("force_active_until")
                effect["force_active_until"] = max(float(current or 0.0), float(active_until_ts))
            elif override_type == OVERRIDE_SKIP_RECONCILE_UNTIL:
                current = effect.get("skip_reconcile_until")
                effect["skip_reconcile_until"] = max(float(current or 0.0), float(active_until_ts))
        return effect

    def _current_serpapi_recheck_gate_ts(self, ke: KeyEntry) -> Optional[float]:
        now_ts = _now_ts()
        expected_reset_ts = _future_ts_or_none(ke.expected_reset_at, now_ts)
        exhausted_ts = _future_ts_or_none(ke.exhausted_until, now_ts)
        candidates = [ts for ts in (expected_reset_ts, exhausted_ts) if ts is not None]
        if not candidates:
            return None
        return min(candidates)

    def _should_skip_serpapi_reconcile_key(
        self,
        *,
        ke: KeyEntry,
        is_forced: bool,
        override_effects: Dict[str, Any],
    ) -> bool:
        if is_forced:
            return False
        now_ts = _now_ts()
        skip_until = _future_ts_or_none(override_effects.get("skip_reconcile_until"), now_ts)
        if skip_until is not None:
            return True
        if override_effects.get("clear_exhaustion"):
            return False
        force_active_until = _future_ts_or_none(override_effects.get("force_active_until"), now_ts)
        if force_active_until is not None:
            return False
        force_exhausted_until = _future_ts_or_none(override_effects.get("force_exhausted_until"), now_ts)
        if force_exhausted_until is not None:
            return True

        exhausted_until = _future_ts_or_none(ke.exhausted_until, now_ts)
        retry_after = _future_ts_or_none(ke.retry_after, now_ts)
        if exhausted_until is not None or retry_after is not None:
            # Exhausted keys should not be rechecked before the retry window opens.
            return True

        try:
            last_checked_at = float(ke.last_checked_at) if ke.last_checked_at is not None else None
        except Exception:
            last_checked_at = None
        if last_checked_at is None:
            return False
        stale_cutoff_seconds = float(SERPAPI_RECONCILE_STALE_SECONDS)
        # Active keys use stale-window refresh checks.
        return (now_ts - last_checked_at) < stale_cutoff_seconds

    def _upsert_provider_state_override(
        self,
        *,
        provider: str,
        scope_type: str,
        scope_identifier: Optional[str],
        key_name_fingerprint: Optional[str],
        key_value_fingerprint: Optional[str],
        override_type: str,
        active_until: Optional[datetime],
        note: Optional[str],
    ) -> Dict[str, Any]:
        normalized_provider = str(provider or "").strip().lower()
        normalized_scope_type = str(scope_type or "").strip().lower()
        normalized_override_type = str(override_type or "").strip().lower()
        normalized_scope_identifier = str(scope_identifier or "").strip() or None
        normalized_key_name_fp = str(key_name_fingerprint or "").strip() or None
        normalized_key_value_fp = str(key_value_fingerprint or "").strip() or None
        if normalized_scope_type not in VALID_OVERRIDE_SCOPES:
            raise ValueError(f"unsupported scope_type: {normalized_scope_type}")
        if normalized_override_type not in VALID_OVERRIDE_TYPES:
            raise ValueError(f"unsupported override_type: {normalized_override_type}")
        if normalized_scope_type == OVERRIDE_SCOPE_KEY and not normalized_scope_identifier:
            raise ValueError("scope_identifier is required when scope_type=key")
        if (
            normalized_provider == "serpapi"
            and normalized_scope_type == OVERRIDE_SCOPE_KEY
            and (not normalized_key_name_fp or not normalized_key_value_fp)
        ):
            raise ValueError("serpapi key overrides require both key_name_fingerprint and key_value_fingerprint")
        if normalized_override_type in {
            OVERRIDE_FORCE_EXHAUSTED_UNTIL,
            OVERRIDE_FORCE_ACTIVE_UNTIL,
            OVERRIDE_SKIP_RECONCILE_UNTIL,
        } and active_until is None:
            raise ValueError("active_until is required for timed override types")
        now_utc = _now()
        if isinstance(active_until, datetime):
            if active_until.tzinfo is None:
                active_until = active_until.replace(tzinfo=UTC)
            active_until = active_until.astimezone(UTC)
            if active_until <= now_utc:
                raise ValueError("active_until must be in the future")

        with self._provider_state_session() as session:
            if session is None:
                raise RuntimeError("provider override storage unavailable")
            try:
                from agents.database import ProviderStateOverride
                row = ProviderStateOverride(
                    provider=normalized_provider,
                    scope_type=normalized_scope_type,
                    scope_identifier=normalized_scope_identifier,
                    key_name_fingerprint=normalized_key_name_fp,
                    key_value_fingerprint=normalized_key_value_fp,
                    override_type=normalized_override_type,
                    active_until=active_until,
                    note=note,
                    is_enabled=True,
                )
                session.add(row)
                session.commit()
                session.refresh(row)
                result = {
                    "id": int(row.id),
                    "provider": normalized_provider,
                    "scope_type": normalized_scope_type,
                    "scope_identifier": normalized_scope_identifier,
                    "key_name_fingerprint": normalized_key_name_fp,
                    "key_value_fingerprint": normalized_key_value_fp,
                    "override_type": normalized_override_type,
                    "override_until": self._datetime_to_iso_utc(row.active_until),
                    "active_until": self._datetime_to_iso_utc(row.active_until),
                    "override_until_semantics": self._override_until_semantics(normalized_override_type),
                    "note": row.note,
                    "is_enabled": bool(row.is_enabled),
                }
            except Exception:
                session.rollback()
                logger.exception("provider_override_upsert_failed")
                raise
        self._invalidate_provider_override_cache(normalized_provider)
        return result

    def _list_provider_state_overrides(
        self,
        *,
        provider: Optional[str] = None,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        normalized_provider = str(provider or "").strip().lower() or None
        now_utc = _now()
        with self._provider_state_session() as session:
            if session is None:
                return []
            try:
                from agents.database import ProviderStateOverride
                query = session.query(ProviderStateOverride)
                if normalized_provider:
                    query = query.filter(ProviderStateOverride.provider == normalized_provider)
                rows = query.order_by(ProviderStateOverride.updated_at.desc()).all()
            except Exception:
                logger.exception("provider_override_list_failed")
                return []
        out: List[Dict[str, Any]] = []
        for row in rows:
            active_until = row.active_until
            if isinstance(active_until, datetime) and active_until.tzinfo is None:
                active_until = active_until.replace(tzinfo=UTC)
            row_payload = {
                "provider": str(row.provider or ""),
                "scope_type": str(row.scope_type or ""),
                "scope_identifier": row.scope_identifier,
                "key_name_fingerprint": str(row.key_name_fingerprint or "") or None,
                "key_value_fingerprint": str(row.key_value_fingerprint or "") or None,
            }
            time_active = bool(
                row.is_enabled and (active_until is None or active_until >= now_utc)
            )
            binding_match = self._override_binding_matches_current_key(row_payload)
            is_currently_active = bool(time_active and binding_match)
            if not include_inactive and not is_currently_active:
                continue
            out.append(
                {
                    "id": int(row.id),
                    "provider": str(row.provider or ""),
                    "scope_type": str(row.scope_type or ""),
                    "scope_identifier": row.scope_identifier,
                    "key_name_fingerprint": str(row.key_name_fingerprint or "") or None,
                    "key_value_fingerprint": str(row.key_value_fingerprint or "") or None,
                    "override_type": str(row.override_type or ""),
                    "override_until": self._datetime_to_iso_utc(active_until),
                    "active_until": self._datetime_to_iso_utc(active_until),
                    "override_until_semantics": self._override_until_semantics(str(row.override_type or "")),
                    "note": row.note,
                    "is_enabled": bool(row.is_enabled),
                    "binding_matches_current_key": binding_match,
                    "is_currently_active": is_currently_active,
                    "created_at": self._datetime_to_iso_utc(row.created_at),
                    "updated_at": self._datetime_to_iso_utc(row.updated_at),
                }
            )
        return out

    def _disable_provider_state_override(self, override_id: int) -> bool:
        with self._provider_state_session() as session:
            if session is None:
                return False
            try:
                from agents.database import ProviderStateOverride
                row = (
                    session.query(ProviderStateOverride)
                    .filter(ProviderStateOverride.id == int(override_id))
                    .first()
                )
                if row is None:
                    return False
                row.is_enabled = False
                row.updated_at = _now()
                provider = str(row.provider or "").strip().lower()
                session.commit()
            except Exception:
                session.rollback()
                logger.exception("provider_override_disable_failed")
                return False
        self._invalidate_provider_override_cache(provider)
        return True

    async def set_provider_state_override(
        self,
        *,
        provider: str,
        scope_type: str,
        scope_identifier: Optional[str],
        override_type: str,
        active_until: Optional[str],
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        parsed_until = self._parse_override_active_until(active_until)
        normalized_provider = str(provider or "").strip().lower()
        normalized_scope_type = str(scope_type or "").strip().lower()
        normalized_override_type = str(override_type or "").strip().lower()
        normalized_scope_identifier = str(scope_identifier or "").strip() or None

        key_name_fp: Optional[str] = None
        key_value_fp: Optional[str] = None
        if normalized_provider == "serpapi" and normalized_scope_type == OVERRIDE_SCOPE_KEY:
            key_name_fp = normalized_scope_identifier
            if key_name_fp:
                key_value_fp = await self._serpapi_key_value_fingerprint_for_name(key_name_fp)
            if not key_name_fp or not key_value_fp:
                raise ValueError(
                    "serpapi key override binding failed: key must exist with matching name fingerprint"
                )

        row = await asyncio.to_thread(
            self._upsert_provider_state_override,
            provider=provider,
            scope_type=scope_type,
            scope_identifier=scope_identifier,
            key_name_fingerprint=key_name_fp,
            key_value_fingerprint=key_value_fp,
            override_type=override_type,
            active_until=parsed_until,
            note=note,
        )
        # SerpAPI operator intent: known-reset "exhausted until" must cap the durable horizon so the
        # key becomes usable/recheckable at that known timestamp.
        if (
            normalized_provider == "serpapi"
            and normalized_scope_type == OVERRIDE_SCOPE_KEY
            and normalized_override_type == OVERRIDE_FORCE_EXHAUSTED_UNTIL
            and parsed_until is not None
            and normalized_scope_identifier
            and key_value_fp
        ):
            await self._apply_serpapi_known_reset_override(
                key_name_fingerprint=normalized_scope_identifier,
                key_value_fingerprint=key_value_fp,
                override_until=parsed_until,
                note=note,
            )
        return row

    async def _serpapi_key_value_fingerprint_for_name(self, key_name_fingerprint: str) -> Optional[str]:
        async with self._lock:
            for ke in self._keys.get("serpapi", []) or []:
                if str(ke.name_fingerprint or "") == str(key_name_fingerprint or ""):
                    return str(ke.fingerprint or "") or None
        return None

    async def _apply_serpapi_known_reset_override(
        self,
        *,
        key_name_fingerprint: str,
        key_value_fingerprint: str,
        override_until: datetime,
        note: Optional[str] = None,
    ) -> None:
        if override_until.tzinfo is None:
            override_until = override_until.replace(tzinfo=UTC)
        override_until = override_until.astimezone(UTC)
        target_ts = override_until.timestamp()
        persist_state: Optional[Dict[str, Any]] = None
        matched: List[Tuple[str, str, Optional[int]]] = []

        async with self._lock:
            for idx, ke in enumerate(self._keys.get("serpapi", []) or []):
                if str(ke.name_fingerprint or "") != str(key_name_fingerprint or ""):
                    continue
                if str(ke.fingerprint or "") != str(key_value_fingerprint or ""):
                    continue
                ke.exhausted_until = target_ts
                ke.retry_after = target_ts
                ke.expected_reset_at = target_ts
                ke.expected_reset_basis = "operator_known_reset_datetime"
                ke.last_checked_at = _now_ts()
                ke.last_provider_reason = "manual_override_known_reset"
                ke.last_provider_error = "manual_override_known_reset"
                ke.failure_classification = "manual_override"
                matched.append((str(ke.fingerprint or ""), str(ke.name_fingerprint or ""), idx))
            if matched:
                self._sweep_key_invariants_locked()
                persist_state = self._snapshot_exhaustion_state_locked()

        if persist_state is not None:
            self._write_state_file(persist_state)

        if matched:
            for key_value_fp, key_name_fp, idx in matched:
                await self._upsert_serpapi_provider_state_to_db(
                    key_name_fingerprint=key_name_fp,
                    key_value_fingerprint=key_value_fp,
                    is_exhausted=True,
                    searches_left=None,
                    last_checked_at=_now(),
                    expected_reset_basis="operator_known_reset_datetime",
                    expected_reset_at=override_until,
                    last_error=(note or "manual_override_known_reset")[:1000],
                    last_reason="manual_override_known_reset",
                    failure_classification="manual_override",
                )
                logger.info(
                    "Applied SerpAPI known-reset override to key slot",
                    extra={
                        "service": "serpapi",
                        "index": idx,
                        "until": override_until.isoformat(),
                        "scope_type": "key",
                    },
                )
        else:
            await asyncio.to_thread(
                self._apply_serpapi_known_reset_override_fallback_sync,
                key_name_fingerprint=str(key_name_fingerprint or ""),
                key_value_fingerprint=str(key_value_fingerprint or ""),
                override_until=override_until,
                note=note,
            )

    def _apply_serpapi_known_reset_override_fallback_sync(
        self,
        key_name_fingerprint: str,
        key_value_fingerprint: str,
        override_until: datetime,
        note: Optional[str] = None,
    ) -> None:
        with self._provider_state_session() as session:
            if session is None:
                return
            try:
                from agents.database import ProviderKeyState
                row = (
                    session.query(ProviderKeyState)
                    .filter(
                        ProviderKeyState.provider == "serpapi",
                        ProviderKeyState.key_name_fingerprint == key_name_fingerprint,
                    )
                    .first()
                )
                if row is not None and str(row.key_value_fingerprint or "") == key_value_fingerprint:
                    row.is_exhausted = True
                    row.expected_reset_basis = "operator_known_reset_datetime"
                    row.expected_reset_at = override_until
                    row.last_checked_at = _now()
                    row.last_reason = "manual_override_known_reset"
                    row.last_error = (note or "manual_override_known_reset")[:1000]
                    row.failure_classification = "manual_override"
                    session.commit()
            except Exception:
                session.rollback()
                logger.exception("serpapi_known_reset_override_db_patch_failed")

    async def list_provider_state_overrides(
        self,
        *,
        provider: Optional[str] = None,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._list_provider_state_overrides,
            provider=provider,
            include_inactive=include_inactive,
        )

    async def disable_provider_state_override(self, override_id: int) -> bool:
        return await asyncio.to_thread(
            self._disable_provider_state_override, override_id
        )

    async def key_scope_identifier(self, provider: str, index: int) -> Optional[str]:
        normalized_provider = str(provider or "").strip().lower()
        async with self._lock:
            try:
                ke = self._keys[normalized_provider][int(index)]
            except Exception:
                return None
            scope_identifier = str(ke.name_fingerprint or "").strip()
            return scope_identifier or None

    def _upsert_provider_key_state(
        self,
        *,
        provider: str,
        key_name_fingerprint: str,
        key_value_fingerprint: str,
        is_exhausted: bool,
        exhausted_until: Optional[datetime] = None,
        retry_after: Optional[datetime] = None,
        searches_left: Optional[int] = None,
        last_checked_at: Optional[datetime] = None,
        last_used_at: Optional[datetime] = None,
        expected_reset_basis: Optional[str] = None,
        expected_reset_at: Optional[datetime] = None,
        last_error: Optional[str] = None,
        last_reason: Optional[str] = None,
        failure_classification: Optional[str] = None,
        state_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized_provider = str(provider or "").strip().lower()
        if not normalized_provider or not key_name_fingerprint:
            return
        with self._provider_state_session() as session:
            if session is None:
                return
            try:
                from agents.database import ProviderKeyState
                row = session.query(ProviderKeyState).filter(
                    ProviderKeyState.provider == normalized_provider,
                    ProviderKeyState.key_name_fingerprint == key_name_fingerprint,
                ).first()
                if row is None:
                    row = ProviderKeyState(
                        provider=normalized_provider,
                        key_name_fingerprint=key_name_fingerprint,
                        key_value_fingerprint=key_value_fingerprint,
                    )
                    session.add(row)
                effective_exhausted_until = exhausted_until
                effective_retry_after = retry_after
                if bool(is_exhausted):
                    if effective_exhausted_until is None and isinstance(row.exhausted_until, datetime):
                        effective_exhausted_until = row.exhausted_until
                    if effective_retry_after is None and isinstance(row.retry_after, datetime):
                        effective_retry_after = row.retry_after
                    if effective_retry_after is None:
                        effective_retry_after = effective_exhausted_until
                row.key_value_fingerprint = key_value_fingerprint
                row.is_exhausted = bool(is_exhausted)
                row.exhausted_until = effective_exhausted_until if bool(is_exhausted) else None
                row.retry_after = effective_retry_after if bool(is_exhausted) else None
                if searches_left is not None or normalized_provider == "serpapi":
                    row.searches_left = searches_left
                if last_checked_at is not None:
                    row.last_checked_at = last_checked_at
                if last_used_at is not None:
                    row.last_used_at = last_used_at
                if expected_reset_basis is not None or not bool(is_exhausted):
                    row.expected_reset_basis = expected_reset_basis
                if expected_reset_at is not None or not bool(is_exhausted):
                    row.expected_reset_at = expected_reset_at
                row.last_error = (last_error or "")[:1000] if last_error is not None else None
                row.last_reason = (last_reason or "")[:128] if last_reason is not None else None
                row.failure_classification = (failure_classification or "")[:64] if failure_classification is not None else None
                row.state_meta = dict(state_meta or {}) if state_meta is not None else None
                session.commit()
            except Exception:
                session.rollback()
                logger.exception("provider_state_upsert_failed", extra={"provider": normalized_provider})

    def _upsert_serpapi_provider_state(
        self,
        *,
        key_name_fingerprint: str,
        key_value_fingerprint: str,
        is_exhausted: bool,
        exhausted_until: Optional[datetime] = None,
        retry_after: Optional[datetime] = None,
        searches_left: Optional[int] = None,
        last_checked_at: Optional[datetime] = None,
        expected_reset_basis: Optional[str] = None,
        expected_reset_at: Optional[datetime] = None,
        last_error: Optional[str] = None,
        last_reason: Optional[str] = None,
        failure_classification: Optional[str] = None,
    ) -> None:
        exhausted_dt = exhausted_until if isinstance(exhausted_until, datetime) else None
        if exhausted_dt is None and is_exhausted and isinstance(expected_reset_at, datetime):
            exhausted_dt = expected_reset_at
        retry_dt = retry_after if isinstance(retry_after, datetime) else exhausted_dt
        self._upsert_provider_key_state(
            provider="serpapi",
            key_name_fingerprint=key_name_fingerprint,
            key_value_fingerprint=key_value_fingerprint,
            is_exhausted=bool(is_exhausted),
            exhausted_until=exhausted_dt,
            retry_after=retry_dt,
            searches_left=searches_left,
            last_checked_at=last_checked_at,
            expected_reset_basis=expected_reset_basis,
            expected_reset_at=expected_reset_at,
            last_error=last_error,
            last_reason=last_reason,
            failure_classification=failure_classification,
        )

    async def _upsert_serpapi_provider_state_to_db(
        self,
        *,
        key_name_fingerprint: str,
        key_value_fingerprint: str,
        is_exhausted: bool,
        exhausted_until: Optional[datetime] = None,
        retry_after: Optional[datetime] = None,
        searches_left: Optional[int] = None,
        last_checked_at: Optional[datetime] = None,
        expected_reset_basis: Optional[str] = None,
        expected_reset_at: Optional[datetime] = None,
        last_error: Optional[str] = None,
        last_reason: Optional[str] = None,
        failure_classification: Optional[str] = None,
    ) -> None:
        exhausted_dt = exhausted_until if isinstance(exhausted_until, datetime) else None
        if exhausted_dt is None and is_exhausted and isinstance(expected_reset_at, datetime):
            exhausted_dt = expected_reset_at
        retry_dt = retry_after if isinstance(retry_after, datetime) else exhausted_dt
        await asyncio.to_thread(
            self._upsert_provider_key_state,
            provider="serpapi",
            key_name_fingerprint=key_name_fingerprint,
            key_value_fingerprint=key_value_fingerprint,
            is_exhausted=bool(is_exhausted),
            exhausted_until=exhausted_dt,
            retry_after=retry_dt,
            searches_left=searches_left,
            last_checked_at=last_checked_at,
            expected_reset_basis=expected_reset_basis,
            expected_reset_at=expected_reset_at,
            last_error=last_error,
            last_reason=last_reason,
            failure_classification=failure_classification,
        )

    def _load_provider_state_map(self, provider: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        normalized_provider = str(provider or "").strip().lower() or None
        with self._provider_state_session() as session:
            if session is None:
                return {}
            try:
                from agents.database import ProviderKeyState
                query = session.query(ProviderKeyState)
                if normalized_provider:
                    query = query.filter(ProviderKeyState.provider == normalized_provider)
                rows = query.all()
            except Exception:
                logger.exception("provider_state_load_failed")
                return {}
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in rows:
            provider_name = str(row.provider or "").strip().lower()
            if not provider_name:
                continue
            bucket = out.setdefault(provider_name, {})
            bucket[str(row.key_name_fingerprint or "")] = {
                "key_value_fingerprint": str(row.key_value_fingerprint or ""),
                "is_exhausted": bool(row.is_exhausted),
                "exhausted_until": row.exhausted_until,
                "retry_after": row.retry_after,
                "searches_left": row.searches_left,
                "last_checked_at": row.last_checked_at,
                "last_used_at": row.last_used_at,
                "expected_reset_basis": row.expected_reset_basis,
                "expected_reset_at": row.expected_reset_at,
                "last_error": row.last_error,
                "last_reason": row.last_reason,
                "failure_classification": row.failure_classification,
                "state_meta": row.state_meta if isinstance(row.state_meta, dict) else {},
            }
        return out

    def _load_serpapi_provider_state_map(self) -> Dict[str, Dict[str, Any]]:
        return self._load_provider_state_map("serpapi").get("serpapi", {})

    def _next_monthly_reset_for_day(self, reset_day: Optional[int], now_dt: Optional[datetime] = None) -> Optional[datetime]:
        if reset_day is None:
            return None
        try:
            day = int(reset_day)
        except Exception:
            return None
        if day < 1 or day > 28:
            return None
        now_utc = (now_dt or _now()).astimezone(UTC)
        candidate = datetime(now_utc.year, now_utc.month, day, tzinfo=UTC)
        if candidate <= now_utc:
            next_month_base = _first_of_next_month(now_utc)
            candidate = datetime(next_month_base.year, next_month_base.month, day, tzinfo=UTC)
        return candidate

    def _serpapi_weekly_retry_datetime(self) -> datetime:
        return _now() + timedelta(seconds=SERPAPI_UNKNOWN_RESET_DEFERRAL_SECONDS)

    def _serpapi_retry_target(
        self,
        ke: KeyEntry,
        *,
        provider_reset_at: Optional[datetime] = None,
    ) -> Tuple[datetime, str]:
        if isinstance(provider_reset_at, datetime):
            if provider_reset_at.tzinfo is None:
                provider_reset_at = provider_reset_at.replace(tzinfo=UTC)
            return provider_reset_at.astimezone(UTC), "account_inferred_cycle_boundary"
        monthly_target = self._next_monthly_reset_for_day(ke.default_reset_day)
        if monthly_target is not None:
            return monthly_target, "default_key_monthly_reset_day"
        return self._serpapi_weekly_retry_datetime(), "weekly_unknown_reset_fallback"

    async def _clear_rotated_serpapi_state(self, key_name_fingerprint: str, key_value_fingerprint: str) -> None:
        # Rotation-safe: preserve slot identity but reset stale exhaustion metadata.
        await asyncio.to_thread(
            self._upsert_provider_key_state,
            provider="serpapi",
            key_name_fingerprint=key_name_fingerprint,
            key_value_fingerprint=key_value_fingerprint,
            is_exhausted=False,
            exhausted_until=None,
            retry_after=None,
            searches_left=None,
            last_checked_at=_now(),
            expected_reset_basis="cleared_on_key_rotation",
            expected_reset_at=None,
            last_error=None,
            last_reason="key_rotated",
            failure_classification="rotation",
            state_meta={},
        )

    def _account_data_searches_left(self, payload: Dict[str, Any]) -> Optional[int]:
        for candidate in (
            payload.get("plan_searches_left"),
            payload.get("total_searches_left"),
            payload.get("searches_left"),
        ):
            if candidate is None:
                continue
            try:
                return int(candidate)
            except Exception:
                continue
        return None

    def _infer_expected_reset_from_account(self, payload: Dict[str, Any]) -> Optional[datetime]:
        # SerpAPI account docs expose counts/limits but not a documented exact reset timestamp.
        # Keep a conservative inference aligned with monthly-cycle behavior when plan looks monthly.
        for key in ("next_reset", "reset_at", "next_reset_at"):
            raw = payload.get(key)
            if not raw:
                continue
            try:
                parsed = datetime.fromisoformat(str(raw))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return parsed.astimezone(UTC)
            except Exception:
                continue
        plan_name = str(payload.get("plan_name") or "").lower()
        if "month" in plan_name or "monthly" in plan_name:
            return _first_of_next_month(_now())
        return None

    def _snapshot_entry_state_locked(self, service: str, idx: int, ke: KeyEntry, *, checked_now: bool = False) -> Optional[Dict[str, Any]]:
        provider = str(service or "").strip().lower()
        if not provider:
            return None
        key_name_fp = str(ke.name_fingerprint or "")
        if not key_name_fp:
            return None
        now_ts = _now_ts()
        exhausted_until_ts = _future_ts_or_none(ke.exhausted_until, now_ts)
        retry_after_ts = _future_ts_or_none(ke.retry_after, now_ts)
        exhausted_dt = datetime.fromtimestamp(exhausted_until_ts, tz=UTC) if exhausted_until_ts is not None else None
        retry_after_dt = datetime.fromtimestamp(retry_after_ts, tz=UTC) if retry_after_ts is not None else exhausted_dt
        last_checked_dt = _now() if checked_now else None
        if not checked_now and ke.last_checked_at is not None:
            try:
                last_checked_dt = datetime.fromtimestamp(float(ke.last_checked_at), tz=UTC)
            except Exception:
                last_checked_dt = None
        expected_reset_dt = None
        if ke.expected_reset_at is not None:
            try:
                expected_reset_dt = datetime.fromtimestamp(float(ke.expected_reset_at), tz=UTC)
            except Exception:
                expected_reset_dt = None
        last_used_dt = _now() if not checked_now else None
        return dict(
            provider=provider,
            key_name_fingerprint=key_name_fp,
            key_value_fingerprint=str(ke.fingerprint or ""),
            is_exhausted=bool(exhausted_until_ts is not None),
            exhausted_until=exhausted_dt,
            retry_after=retry_after_dt,
            searches_left=ke.searches_left if provider == "serpapi" else None,
            last_checked_at=last_checked_dt,
            last_used_at=last_used_dt,
            expected_reset_basis=ke.expected_reset_basis,
            expected_reset_at=expected_reset_dt,
            last_error=ke.last_provider_error,
            last_reason=ke.last_provider_reason,
            failure_classification=ke.failure_classification or ke.last_exhausted_reason_class,
            state_meta={"default_reset_day": ke.default_reset_day} if provider == "serpapi" and ke.default_reset_day else None,
        )

    async def _persist_entry_snapshot_to_db(self, **kwargs: Any) -> None:
        if not kwargs.get("provider") or not kwargs.get("key_name_fingerprint"):
            return
        await asyncio.to_thread(
            self._upsert_provider_key_state,
            provider=kwargs["provider"],
            key_name_fingerprint=kwargs["key_name_fingerprint"],
            key_value_fingerprint=kwargs.get("key_value_fingerprint", ""),
            is_exhausted=kwargs.get("is_exhausted", False),
            exhausted_until=kwargs.get("exhausted_until"),
            retry_after=kwargs.get("retry_after"),
            searches_left=kwargs.get("searches_left"),
            last_checked_at=kwargs.get("last_checked_at"),
            last_used_at=kwargs.get("last_used_at"),
            expected_reset_basis=kwargs.get("expected_reset_basis"),
            expected_reset_at=kwargs.get("expected_reset_at"),
            last_error=kwargs.get("last_error"),
            last_reason=kwargs.get("last_reason"),
            failure_classification=kwargs.get("failure_classification"),
            state_meta=kwargs.get("state_meta"),
        )

    def _snapshot_all_entries_locked(self) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        for service, entries in self._keys.items():
            for idx, ke in enumerate(entries):
                snap = self._snapshot_entry_state_locked(service, idx, ke, checked_now=False)
                if snap is not None:
                    snapshots.append(snap)
        return snapshots

    def _persist_entry_state_locked(self, service: str, idx: int, ke: KeyEntry, *, checked_now: bool = False) -> Optional[Dict[str, Any]]:
        return self._snapshot_entry_state_locked(service, idx, ke, checked_now=checked_now)

    def _persist_all_entries_locked(self) -> List[Dict[str, Any]]:
        return self._snapshot_all_entries_locked()

    def _hydrate_all_provider_state_from_db_sync(self) -> List[Tuple[str, str]]:
        rows_by_provider = self._load_provider_state_map()
        cleared_rotations: List[Tuple[str, str]] = []
        if not rows_by_provider:
            return cleared_rotations
        now_ts = _now_ts()
        for service, entries in self._keys.items():
            provider_rows = rows_by_provider.get(str(service or "").strip().lower(), {})
            if not provider_rows:
                continue
            for ke in entries:
                name_fp = str(ke.name_fingerprint or "")
                if not name_fp:
                    continue
                row = provider_rows.get(name_fp)
                if not isinstance(row, dict):
                    continue
                persisted_value_fp = str(row.get("key_value_fingerprint") or "")
                if persisted_value_fp and persisted_value_fp != ke.fingerprint:
                    ke.exhausted_until = None
                    ke.retry_after = None
                    ke.searches_left = None
                    ke.expected_reset_basis = "cleared_on_key_rotation"
                    ke.expected_reset_at = None
                    ke.last_provider_reason = "key_rotated"
                    ke.failure_classification = "rotation"
                    if service == "serpapi":
                        self._serpapi_force_reconcile_name_fps.add(name_fp)
                        cleared_rotations.append((name_fp, ke.fingerprint))
                    continue

                exhausted_dt = row.get("exhausted_until")
                retry_dt = row.get("retry_after")
                checked_at = row.get("last_checked_at")
                reset_at = row.get("expected_reset_at")
                ke.searches_left = row.get("searches_left")
                ke.last_checked_at = self._datetime_to_ts(checked_at)
                ke.expected_reset_basis = row.get("expected_reset_basis")
                ke.expected_reset_at = self._datetime_to_ts(reset_at)
                ke.last_provider_error = row.get("last_error")
                ke.last_provider_reason = row.get("last_reason")
                ke.failure_classification = row.get("failure_classification")
                state_meta = row.get("state_meta") if isinstance(row.get("state_meta"), dict) else {}
                if service == "serpapi" and ke.default_reset_day is None:
                    maybe_default_day = state_meta.get("default_reset_day")
                    if isinstance(maybe_default_day, int):
                        ke.default_reset_day = maybe_default_day

                exhausted_ts = _future_ts_or_none(self._datetime_to_ts(exhausted_dt), now_ts)
                retry_ts = _future_ts_or_none(self._datetime_to_ts(retry_dt), now_ts)
                if bool(row.get("is_exhausted")) and exhausted_ts is None:
                    if service == "serpapi":
                        if ke.default_reset_day is not None:
                            inferred = self._next_monthly_reset_for_day(ke.default_reset_day)
                            exhausted_ts = inferred.timestamp() if inferred is not None else None
                            if exhausted_ts is not None:
                                ke.expected_reset_basis = "default_key_monthly_reset_day"
                                ke.expected_reset_at = exhausted_ts
                        if exhausted_ts is None:
                            exhausted_ts = self._serpapi_weekly_retry_datetime().timestamp()
                            ke.expected_reset_basis = "weekly_unknown_reset_fallback"
                            ke.expected_reset_at = exhausted_ts
                    else:
                        exhausted_ts = _future_ts_or_none(
                            _exhaustion_ttl_for_error(service, str(ke.last_provider_reason or "")),
                            now_ts,
                        )
                ke.exhausted_until = exhausted_ts
                ke.retry_after = retry_ts if retry_ts is not None else exhausted_ts

                # CRITICAL: restore last_exhausted_reason_class from DB so the
                # already-exhausted guard in mark_exhausted works after restart.
                # Without this, every caller re-announces the key as a fresh incident.
                if exhausted_ts is not None:
                    fc = row.get("failure_classification")
                    if fc:
                        ke.last_exhausted_reason_class = _normalize_reason_class(str(fc))

        self._sweep_key_invariants_locked()
        return cleared_rotations

    async def _hydrate_all_provider_state_from_db(self) -> None:
        cleared_rotations: List[Tuple[str, str]] = []
        async with self._lock:
            cleared_rotations = self._hydrate_all_provider_state_from_db_sync()
        for name_fp, value_fp in cleared_rotations:
            await self._clear_rotated_serpapi_state(name_fp, value_fp)

    async def _hydrate_serpapi_state_from_db(self) -> None:
        await self._hydrate_all_provider_state_from_db()

    def _migrate_legacy_state_file_to_db(self, legacy_state: Dict[str, Any]) -> None:
        if not isinstance(legacy_state, dict):
            return
        now_ts = _now_ts()
        if not self._keys:
            return
        for service, entries in self._keys.items():
            if not entries:
                continue
            if service == "serpapi":
                continue
            service_state = legacy_state.get(service, {})
            if not isinstance(service_state, dict):
                continue
            fp_to_until: Dict[str, Optional[float]] = {}
            for fp, payload in service_state.items():
                if not isinstance(payload, dict):
                    continue
                fp_to_until[str(fp)] = _future_ts_or_none(payload.get("exhausted_until"), now_ts)
            if not fp_to_until:
                continue
            for ke in entries:
                until_ts = fp_to_until.get(str(ke.fingerprint or ""))
                if until_ts is None:
                    continue
                exhausted_dt = datetime.fromtimestamp(until_ts, tz=UTC)
                self._upsert_provider_key_state(
                    provider=service,
                    key_name_fingerprint=str(ke.name_fingerprint or ""),
                    key_value_fingerprint=str(ke.fingerprint or ""),
                    is_exhausted=True,
                    exhausted_until=exhausted_dt,
                    retry_after=exhausted_dt,
                    last_checked_at=_now(),
                    expected_reset_basis="legacy_json_migration",
                    expected_reset_at=None,
                    last_error=None,
                    last_reason="legacy_state_file_migration",
                    failure_classification="migration",
                )
        try:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
        except Exception:
            logger.debug("legacy_key_state_file_cleanup_skipped")

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
                        await self._hydrate_serpapi_state_from_db()
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
        serpapi_rotated: List[Tuple[str, str]] = []
        async with self._lock:
            new_env_records = self._parse_env_key_records()

            # Build old fingerprint maps for detailed payload
            old_fingerprint_maps = {}
            old_serpapi_by_name: Dict[str, str] = {}
            for svc, entries in self._keys.items():
                old_fingerprint_maps[svc] = {i: ke.fingerprint for i, ke in enumerate(entries)}
                if svc == "serpapi":
                    for ke in entries:
                        if ke.name_fingerprint:
                            old_serpapi_by_name[str(ke.name_fingerprint)] = ke.fingerprint

            # Build new keys dict and new fingerprint maps
            new_keys = {}
            new_fingerprint_maps = {}
            new_serpapi_by_name: Dict[str, str] = {}
            for service, key_records in new_env_records.items():
                entries = []
                # Get current entries for this service (if any)
                current_entries_by_value = {ke.fingerprint: ke for ke in self._keys.get(service, [])}
                current_entries_by_name = {
                    str(ke.name_fingerprint): ke for ke in self._keys.get(service, [])
                    if ke.name_fingerprint
                }
                for record in key_records:
                    key = str(record.get("value") or "")
                    key_name = str(record.get("name") or "")
                    default_reset_day = record.get("default_reset_day")
                    name_fp = self._fingerprint_name(key_name)
                    fp = self._fingerprint(key)
                    if fp in current_entries_by_value:
                        # reuse existing entry (preserves exhausted_until, in_use, last_used, pending flags)
                        ke = current_entries_by_value[fp]
                        if service == "serpapi" and str(ke.name_fingerprint or "") != name_fp:
                            # Slot identity changed while key value stayed the same:
                            # do not carry old slot-bound override/exhaustion state forward.
                            entries.append(
                                KeyEntry(
                                    value=key,
                                    fingerprint=fp,
                                    key_name=key_name,
                                    name_fingerprint=name_fp,
                                    default_reset_day=default_reset_day,
                                )
                            )
                            self._serpapi_force_reconcile_name_fps.add(name_fp)
                        else:
                            # update value in case it changed (fingerprint same implies value same, but safe)
                            ke.value = key
                            ke.key_name = key_name
                            ke.name_fingerprint = name_fp
                            if service == "serpapi":
                                ke.default_reset_day = default_reset_day
                            entries.append(ke)
                    elif service == "serpapi" and name_fp in current_entries_by_name:
                        # Value changed while slot identity stayed stable => key rotation.
                        entries.append(
                                KeyEntry(
                                    value=key,
                                    fingerprint=fp,
                                    key_name=key_name,
                                    name_fingerprint=name_fp,
                                    default_reset_day=default_reset_day,
                                )
                            )
                    else:
                        # brand new key
                        entries.append(
                            KeyEntry(
                                value=key,
                                fingerprint=fp,
                                key_name=key_name,
                                name_fingerprint=name_fp,
                                default_reset_day=default_reset_day,
                            )
                        )
                    if service == "serpapi":
                        new_serpapi_by_name[name_fp] = fp
                new_keys[service] = entries
                new_fingerprint_maps[service] = {i: ke.fingerprint for i, ke in enumerate(entries)}
                # update round-robin index (if service existed, keep index; otherwise start at 0)
                if service not in self._rr_index:
                    self._rr_index[service] = 0

            # Remove services no longer present
            for service in list(self._rr_index.keys()):
                if service not in new_env_records:
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
            reload_snapshots: List[Dict[str, Any]] = []
            if changed:
                reload_snapshots = self._persist_all_entries_locked()

            for name_fp, old_value_fp in old_serpapi_by_name.items():
                new_value_fp = new_serpapi_by_name.get(name_fp)
                if not new_value_fp:
                    continue
                if old_value_fp != new_value_fp:
                    serpapi_rotated.append((name_fp, new_value_fp))
                    self._serpapi_force_reconcile_name_fps.add(name_fp)

        # Outside lock: persist DB writes off the event loop
        for snap in reload_snapshots:
            await self._persist_entry_snapshot_to_db(**snap)

        # Outside lock: notify if changed
        if changed:
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

        for name_fp, new_value_fp in serpapi_rotated:
            await self._clear_rotated_serpapi_state(name_fp, new_value_fp)

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
            override_effects = self._resolve_provider_override_effects(
                provider=service,
                key_name_fingerprint=str(ke.name_fingerprint or "") or None,
                key_value_fingerprint=str(ke.fingerprint or "") or None,
            )
            forced_exhausted_until = _future_ts_or_none(override_effects.get("force_exhausted_until"), now)
            if forced_exhausted_until is not None:
                continue
            force_active_until = _future_ts_or_none(override_effects.get("force_active_until"), now)
            clear_exhaustion = bool(override_effects.get("clear_exhaustion"))
            if force_active_until is None and not clear_exhaustion and ke.exhausted_until and ke.exhausted_until > now:
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
    async def reserve_key(self, service: str, *, wait_timeout: Optional[float] = None):
        """Async context manager that reserves a key for exclusive use.
           Usage:
               async with manager.reserve_key("serpapi") as (idx, key_value):
                   # use the key
           Raises RuntimeError if no key available.
        """
        if wait_timeout is None:
            wait_timeout = float(KEY_RESERVATION_WAIT_SECONDS)
        wait_timeout = max(0.0, float(wait_timeout))
        started_wait = time.monotonic()
        pending_exhaust_event: Optional[dict] = None
        key_released_event: Optional[dict] = None

        idx: Optional[int] = None
        ke: Optional[KeyEntry] = None
        # Step 1: sweep invariants and pick a key (under global lock), waiting if needed.
        while True:
            async with self._lock:
                self._sweep_key_invariants_locked()
                entries = self._keys.get(service)
                if not entries:
                    raise RuntimeError(f"No keys configured for service: {service}")
                idx, ke = await self._pick_active_key(service)
                if ke is not None and idx is not None:
                    break

            if (time.monotonic() - started_wait) >= wait_timeout:
                app_metrics.record_key_state_event(
                    service=service,
                    event="reservation_timeout",
                    reason_class="all_exhausted",
                )
                raise RuntimeError(f"No available keys for service: {service}")
            await asyncio.sleep(float(KEY_RESERVATION_POLL_SECONDS))

        assert ke is not None and idx is not None

        app_metrics.record_key_state_event(
            service=service,
            event="reserved",
            reason_class="ok",
        )

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
                    ke.retry_after = ke.exhausted_until
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
                release_snapshot = self._persist_entry_state_locked(service, idx, ke, checked_now=False)

            # DB write outside all held locks
            if release_snapshot is not None:
                await self._persist_entry_snapshot_to_db(**release_snapshot)

            if pending_exhaust_event is not None:
                asyncio.create_task(self._notify_listeners("key_exhausted", pending_exhaust_event))

            if key_released_event is not None:
                asyncio.create_task(self._notify_listeners("key_no_longer_in_use", key_released_event))

    # ---------- legacy: simple get_key (does not reserve) ----------
    async def get_key(self, service: str) -> Tuple[Optional[str], Optional[int]]:
        """Legacy method: returns (key_value, index) for the first available key,
           without incrementing in_use. Prefer reserve_key() for new code."""
        result: Tuple[Optional[str], Optional[int]] = (None, None)
        async with self._lock:
            self._sweep_key_invariants_locked()
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
        return result

    # ---------- exhaustion marking ----------
    async def mark_exhausted(self, service: str, idx: int, reason: str = "",
                             reset_at: Optional[Union[datetime, float]] = None,
                             until: Optional[float] = None) -> bool:
        """Mark a key as exhausted until a specific time (reset_at) or until a policy-based default.
           If the key is currently in use, the exhaustion is recorded as pending and will be applied
           after all users release the key.
           reset_at can be a datetime (aware) or a float epoch timestamp.
           until is an alias for reset_at used by airline_api.
           The state is persisted (off the event loop) when exhaustion actually takes effect.

           Returns True if the exhaustion was newly applied or was a meaningful state change.
           Returns False if the key was already exhausted with the same reason_class within
           the active window (caller should not log or emit events).
        """
        # 'until' is an alias for reset_at
        if until is not None and reset_at is None:
            reset_at = until

        now_ts = _now_ts()
        reason_class = _normalize_reason_class(reason)
        mark_snapshot: Optional[Dict[str, Any]] = None
        result: bool = True

        async with self._lock:
            try:
                ke = self._keys[service][idx]
            except (KeyError, IndexError):
                logger.warning("mark_exhausted: invalid service/index")
                return False

            # Determine exhaustion timestamp
            if reset_at is not None:
                if isinstance(reset_at, datetime):
                    if reset_at.tzinfo is None:
                        reset_at = reset_at.replace(tzinfo=UTC)
                    until_ts = reset_at.timestamp()
                else:
                    until_ts = float(reset_at)
            else:
                if service == "serpapi" and reason_class == "quota":
                    retry_dt, retry_basis = self._serpapi_retry_target(ke)
                    until_ts = retry_dt.timestamp()
                    ke.expected_reset_basis = retry_basis
                    ke.expected_reset_at = until_ts
                else:
                    until_ts = _exhaustion_ttl_for_error(service, reason)
                    if service == "serpapi":
                        ke.expected_reset_basis = self._serpapi_expected_reset_basis(
                            reason=reason,
                            reason_class=reason_class,
                            has_reset_at=True,
                            from_account="account_reconcile" in str(reason or "").lower(),
                        )
                        ke.expected_reset_at = float(until_ts)

            # --- Guard: key already exhausted within active window, same reason_class ---
            existing_until = _future_ts_or_none(ke.exhausted_until, now_ts)
            if existing_until is not None and reason_class == ke.last_exhausted_reason_class:
                ke.last_checked_at = now_ts
                mark_snapshot = self._persist_entry_state_locked(service, idx, ke, checked_now=True)
                result = False

            if result:
                # Cap exhaustion horizon: never extend beyond 2x the requested TTL for the same reason_class.
                if existing_until is not None:
                    requested_ttl = until_ts - now_ts
                    max_allowed = existing_until + requested_ttl
                    if until_ts > max_allowed:
                        until_ts = max_allowed

                merged_until = _merge_exhaustion_until(ke.exhausted_until, until_ts, now_ts)
                pending_merged_until = _merge_exhaustion_until(ke._pending_exhaust_until, until_ts, now_ts)
                requested_until_iso = datetime.fromtimestamp(until_ts, tz=UTC).isoformat()
                ke.last_exhausted_reason = reason
                ke.last_exhausted_reason_class = reason_class
                ke.last_exhausted_at = now_ts
                ke.last_provider_reason = reason
                ke.last_provider_error = reason
                if service == "serpapi":
                    ke.failure_classification = self._classify_serpapi_failure(reason, reason_class)
                else:
                    ke.failure_classification = reason_class or "unknown"
                ke.last_checked_at = now_ts

                dedup_key = self._exhaustion_dedup_key(service, idx, reason_class)
                should_log = self._should_log_exhaustion(dedup_key, now_ts)
                sanitized_reason = self._sanitize_reason_for_log(reason)
                was_already_exhausted = existing_until is not None

                if ke.in_use > 0:
                    ke._pending_exhaust = pending_merged_until is not None
                    ke._pending_exhaust_until = pending_merged_until
                    app_metrics.record_key_state_event(
                        service=service,
                        event="exhausted_pending",
                        reason_class=reason_class,
                    )
                    pending_until_iso = (
                        datetime.fromtimestamp(pending_merged_until, tz=UTC).isoformat()
                        if pending_merged_until is not None
                        else requested_until_iso
                    )
                    if should_log:
                        log_fn = logger.warning if reason_class == "auth" else logger.info
                        log_fn(
                            "Key exhaustion deferred (pending release)",
                            extra={
                                "service": service,
                                "index": idx,
                                "until": pending_until_iso,
                                "reason_class": reason_class,
                                "pending": True,
                            },
                        )
                    asyncio.create_task(self._notify_listeners("key_exhausted", {
                        "service": service,
                        "index": idx,
                        "reason": sanitized_reason,
                        "reason_class": reason_class,
                        "until": pending_until_iso,
                        "pending": True
                    }))
                else:
                    ke.exhausted_until = merged_until
                    ke.retry_after = merged_until
                    app_metrics.record_key_state_event(
                        service=service,
                        event="exhausted",
                        reason_class=reason_class,
                    )
                    effective_until_iso = (
                        datetime.fromtimestamp(ke.exhausted_until, tz=UTC).isoformat()
                        if ke.exhausted_until is not None
                        else requested_until_iso
                    )

                    if should_log:
                        if reason_class == "auth":
                            log_fn = logger.warning
                        elif was_already_exhausted:
                            log_fn = logger.debug
                        else:
                            log_fn = logger.info
                        log_fn(
                            "Key exhaustion applied",
                            extra={
                                "service": service,
                                "index": idx,
                                "until": effective_until_iso,
                                "reason_class": reason_class,
                                "was_already_exhausted": was_already_exhausted,
                            },
                        )
                    asyncio.create_task(self._notify_listeners("key_exhausted", {
                        "service": service,
                        "index": idx,
                        "reason": sanitized_reason,
                        "reason_class": reason_class,
                        "until": effective_until_iso,
                        "pending": False
                    }))
                if service == "serpapi" and ke.expected_reset_at is None:
                    ke.expected_reset_basis = self._serpapi_expected_reset_basis(
                        reason=reason,
                        reason_class=reason_class,
                        has_reset_at=bool(until_ts),
                        from_account="account_reconcile" in str(reason or "").lower(),
                    )
                    ke.expected_reset_at = float(until_ts)

                self._sweep_key_invariants_locked()
                mark_snapshot = self._persist_entry_state_locked(service, idx, ke, checked_now=True)

        # DB write outside held lock
        if mark_snapshot is not None:
            await self._persist_entry_snapshot_to_db(**mark_snapshot)
        return result

    # ---------- clear exhaustion / pending clear ----------
    async def clear_exhausted(self, service: str, idx: int):
        """Manually re-enable a key by clearing its exhaustion flag (and any pending flag)."""
        clear_snapshot: Optional[Dict[str, Any]] = None
        async with self._lock:
            try:
                ke = self._keys[service][idx]
                changed = False
                if ke.exhausted_until is not None:
                    ke.exhausted_until = None
                    ke.retry_after = None
                    changed = True
                if ke._pending_exhaust:
                    ke._pending_exhaust = False
                    ke._pending_exhaust_until = None
                    changed = True
                if changed:
                    if service == "serpapi":
                        ke.expected_reset_basis = "cleared_after_reconcile"
                        ke.expected_reset_at = None
                    ke.last_provider_error = None
                    ke.last_provider_reason = "cleared"
                    ke.failure_classification = "recovered"
                    ke.last_checked_at = _now_ts()
                    app_metrics.record_key_state_event(
                        service=service,
                        event="recovered",
                        reason_class=ke.last_exhausted_reason_class or "unknown",
                    )
                    self._sweep_key_invariants_locked()
                    clear_snapshot = self._persist_entry_state_locked(service, idx, ke, checked_now=True)
                    logger.info("Key cleared from exhausted state", extra={"service": service, "index": idx})
            except (KeyError, IndexError):
                pass
        if clear_snapshot is not None:
            await self._persist_entry_snapshot_to_db(**clear_snapshot)

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
                    app_metrics.record_key_state_event(
                        service=service,
                        event="pending_clear_release_ready",
                        reason_class=ke.last_exhausted_reason_class or "unknown",
                    )
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

    async def reconcile_serpapi_account_state(
        self,
        *,
        key_name_fingerprints: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        started_at = _now()
        self._serpapi_reconcile_meta["last_started_at"] = started_at.isoformat()
        self._serpapi_reconcile_meta["last_status"] = "running"
        self._serpapi_reconcile_meta["last_error"] = None

        forced_names: set[str] = set(str(item or "") for item in (key_name_fingerprints or set()) if str(item or ""))
        async with self._lock:
            if self._serpapi_force_reconcile_name_fps:
                forced_names |= set(self._serpapi_force_reconcile_name_fps)
                self._serpapi_force_reconcile_name_fps -= forced_names

            serpapi_entries: List[Tuple[int, str, str, str, bool, float, float, float]] = []
            skipped = 0
            for idx, ke in enumerate(self._keys.get("serpapi", []) or []):
                name_fp = str(ke.name_fingerprint or "")
                if key_name_fingerprints and name_fp not in key_name_fingerprints:
                    continue
                override_effects = self._resolve_provider_override_effects(
                    provider="serpapi",
                    key_name_fingerprint=name_fp or None,
                    key_value_fingerprint=str(ke.fingerprint or "") or None,
                )
                is_forced = bool(name_fp and name_fp in forced_names)
                if self._should_skip_serpapi_reconcile_key(
                    ke=ke,
                    is_forced=is_forced,
                    override_effects=override_effects,
                ):
                    skipped += 1
                    continue
                serpapi_entries.append(
                    (
                        idx,
                        str(ke.value or ""),
                        str(ke.fingerprint or ""),
                        name_fp,
                        bool(ke.exhausted_until and ke.exhausted_until > _now_ts()),
                        float(ke.exhausted_until or 0.0),
                        float(ke.expected_reset_at or 0.0),
                        float(ke.last_checked_at or 0.0),
                    )
                )

        if not serpapi_entries:
            self._serpapi_reconcile_meta["last_completed_at"] = _now().isoformat()
            self._serpapi_reconcile_meta["last_status"] = "ok"
            self._serpapi_reconcile_meta["last_checked"] = 0
            self._serpapi_reconcile_meta["last_skipped"] = int(skipped if 'skipped' in locals() else 0)
            self._serpapi_reconcile_meta["last_forced"] = len(forced_names)
            return {
                "checked": 0,
                "updated": 0,
                "exhausted": 0,
                "recovered": 0,
                "errors": 0,
                "skipped": int(skipped if 'skipped' in locals() else 0),
                "forced": len(forced_names),
            }

        from core.http_client import get_client

        client = get_client()
        checked = 0
        updated = 0
        exhausted = 0
        recovered = 0
        errors = 0
        skipped = int(skipped if 'skipped' in locals() else 0)

        for idx, key_value, key_value_fp, key_name_fp, was_exhausted, _old_exhausted_until, _old_expected_reset_at, _old_last_checked_at in serpapi_entries:
            checked += 1
            now_utc = _now()
            try:
                response = await client.get(
                    "https://serpapi.com/account.json",
                    params={"api_key": key_value},
                    timeout=SERPAPI_ACCOUNT_RECONCILE_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                errors += 1
                error_text = str(exc)
                failure_classification = self._classify_serpapi_failure(error_text, "transient")
                await self._upsert_serpapi_provider_state_to_db(
                    key_name_fingerprint=key_name_fp,
                    key_value_fingerprint=key_value_fp,
                    is_exhausted=was_exhausted,
                    searches_left=None,
                    last_checked_at=now_utc,
                    expected_reset_basis="account_check_exception",
                    expected_reset_at=None,
                    last_error=error_text,
                    last_reason="account_reconcile_exception",
                    failure_classification=failure_classification,
                )
                async with self._lock:
                    try:
                        ke = self._keys["serpapi"][idx]
                        ke.last_provider_error = error_text
                        ke.last_provider_reason = "account_reconcile_exception"
                        ke.failure_classification = failure_classification
                        ke.last_checked_at = now_utc.timestamp()
                    except Exception:
                        pass
                continue

            if response.status_code != 200:
                errors += 1
                reason = f"account_reconcile_http_{response.status_code}"
                reason_class = "auth" if response.status_code in {401, 403} else "transient"
                if response.status_code in {401, 403}:
                    await self.mark_exhausted("serpapi", idx, reason=reason)
                    exhausted += 1
                    was_exhausted = True
                await self._upsert_serpapi_provider_state_to_db(
                    key_name_fingerprint=key_name_fp,
                    key_value_fingerprint=key_value_fp,
                    is_exhausted=was_exhausted,
                    searches_left=None,
                    last_checked_at=now_utc,
                    expected_reset_basis="account_http_error",
                    expected_reset_at=None,
                    last_error=f"HTTP {response.status_code}",
                    last_reason=reason,
                    failure_classification=self._classify_serpapi_failure(reason, reason_class),
                )
                continue

            try:
                data = response.json()
            except Exception as exc:
                errors += 1
                await self._upsert_serpapi_provider_state_to_db(
                    key_name_fingerprint=key_name_fp,
                    key_value_fingerprint=key_value_fp,
                    is_exhausted=was_exhausted,
                    searches_left=None,
                    last_checked_at=now_utc,
                    expected_reset_basis="account_json_parse_error",
                    expected_reset_at=None,
                    last_error=str(exc),
                    last_reason="account_reconcile_parse_error",
                    failure_classification="transient",
                )
                continue

            searches_left = self._account_data_searches_left(data if isinstance(data, dict) else {})
            reset_at = self._infer_expected_reset_from_account(data if isinstance(data, dict) else {})
            reset_ts: Optional[float] = reset_at.timestamp() if isinstance(reset_at, datetime) else None

            if searches_left is not None and searches_left <= 0:
                # Only call mark_exhausted if key is NOT already exhausted.
                # mark_exhausted has its own guard, but skip the call entirely
                # to avoid unnecessary DB writes and event noise.
                if not was_exhausted:
                    await self.mark_exhausted(
                        "serpapi",
                        idx,
                        reason="account_reconcile_quota_exhausted",
                        until=reset_ts,
                    )
                    exhausted += 1
                else:
                    # Quietly confirm — already exhausted, just persist state bump.
                    async with self._lock:
                        try:
                            ke = self._keys["serpapi"][idx]
                            ke.last_checked_at = now_utc.timestamp()
                            ke.searches_left = searches_left
                            self._persist_entry_state_locked("serpapi", idx, ke, checked_now=True)
                        except Exception:
                            pass
            elif searches_left is not None and searches_left > 0:
                await self.clear_exhausted("serpapi", idx)
                if was_exhausted:
                    recovered += 1

            async with self._lock:
                try:
                    ke = self._keys["serpapi"][idx]
                    ke.searches_left = searches_left
                    ke.last_checked_at = now_utc.timestamp()
                    ke.last_provider_error = None
                    ke.last_provider_reason = "account_reconcile"
                    if searches_left is not None and searches_left <= 0:
                        if isinstance(reset_at, datetime):
                            ke.expected_reset_basis = "account_inferred_cycle_boundary"
                            ke.expected_reset_at = reset_at.timestamp()
                        elif ke.expected_reset_at is None:
                            retry_dt, retry_basis = self._serpapi_retry_target(ke)
                            ke.expected_reset_basis = retry_basis
                            ke.expected_reset_at = retry_dt.timestamp()
                        ke.failure_classification = "monthly_quota"
                    else:
                        ke.expected_reset_basis = self._serpapi_expected_reset_basis(
                            reason="account_reconcile",
                            reason_class="unknown",
                            has_reset_at=bool(reset_at),
                            from_account=True,
                        )
                        ke.expected_reset_at = reset_at.timestamp() if isinstance(reset_at, datetime) else None
                        ke.failure_classification = "ok"
                    is_exhausted_now = bool(ke.exhausted_until and ke.exhausted_until > _now_ts())
                    exhausted_until_dt = (
                        datetime.fromtimestamp(float(ke.exhausted_until), tz=UTC)
                        if is_exhausted_now
                        else None
                    )
                    retry_after_dt = (
                        datetime.fromtimestamp(float(ke.retry_after), tz=UTC)
                        if ke.retry_after and ke.retry_after > _now_ts()
                        else exhausted_until_dt
                    )
                    expected_reset_basis = ke.expected_reset_basis
                    expected_reset_at_dt = (
                        datetime.fromtimestamp(float(ke.expected_reset_at), tz=UTC)
                        if ke.expected_reset_at
                        else None
                    )
                    failure_classification = ke.failure_classification
                except Exception:
                    is_exhausted_now = was_exhausted
                    exhausted_until_dt = None
                    retry_after_dt = None
                    expected_reset_basis = None
                    expected_reset_at_dt = None
                    failure_classification = (
                        "monthly_quota"
                        if (searches_left is not None and searches_left <= 0)
                        else "ok"
                    )

            await self._upsert_serpapi_provider_state_to_db(
                key_name_fingerprint=key_name_fp,
                key_value_fingerprint=key_value_fp,
                is_exhausted=is_exhausted_now,
                exhausted_until=exhausted_until_dt,
                retry_after=retry_after_dt,
                searches_left=searches_left,
                last_checked_at=now_utc,
                expected_reset_basis=expected_reset_basis,
                expected_reset_at=expected_reset_at_dt,
                last_error=None,
                last_reason="account_reconcile",
                failure_classification=failure_classification,
            )
            updated += 1

        completed_at = _now()
        self._serpapi_reconcile_meta["last_completed_at"] = completed_at.isoformat()
        self._serpapi_reconcile_meta["last_status"] = "ok" if errors == 0 else "degraded"
        self._serpapi_reconcile_meta["last_error"] = None if errors == 0 else f"errors={errors}"
        self._serpapi_reconcile_meta["last_checked"] = checked
        self._serpapi_reconcile_meta["last_skipped"] = skipped
        self._serpapi_reconcile_meta["last_forced"] = len(forced_names)
        return {
            "checked": checked,
            "updated": updated,
            "exhausted": exhausted,
            "recovered": recovered,
            "errors": errors,
            "skipped": skipped,
            "forced": len(forced_names),
        }

    def start_serpapi_reconcile_loop(self, interval_seconds: int = SERPAPI_ACCOUNT_RECONCILE_INTERVAL_SECONDS) -> None:
        if self._serpapi_reconcile_task is not None and not self._serpapi_reconcile_task.done():
            return
        self._stop_serpapi_reconcile.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("start_serpapi_reconcile_loop called without running event loop; skipping.")
            return
        self._serpapi_reconcile_task = loop.create_task(
            self._serpapi_reconcile_loop(interval_seconds=max(30, int(interval_seconds)))
        )
        logger.info(
            "Started SerpAPI account reconciliation loop",
            extra={"interval_seconds": max(30, int(interval_seconds))},
        )

    def stop_serpapi_reconcile_loop(self) -> None:
        self._stop_serpapi_reconcile.set()
        task = self._serpapi_reconcile_task
        if task is not None:
            task.cancel()
        self._serpapi_reconcile_task = None

    async def _serpapi_reconcile_loop(self, interval_seconds: int) -> None:
        try:
            while not self._stop_serpapi_reconcile.is_set():
                try:
                    await self.reconcile_serpapi_account_state()
                except Exception:
                    logger.exception("SerpAPI reconcile loop iteration failed")
                try:
                    await asyncio.wait_for(
                        self._stop_serpapi_reconcile.wait(),
                        timeout=max(30, int(interval_seconds)),
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info("SerpAPI reconcile loop cancelled")
        finally:
            self._serpapi_reconcile_task = None

    def serpapi_reconcile_status(self) -> Dict[str, Any]:
        running = bool(self._serpapi_reconcile_task and not self._serpapi_reconcile_task.done())
        return {
            **self._serpapi_reconcile_meta,
            "running": running,
            "forced_key_count": len(self._serpapi_force_reconcile_name_fps),
        }

    # ---------- debug status ----------
    @staticmethod
    def _iso_from_timestamp(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
        except Exception:
            return None

    async def status(self) -> Dict[str, List[Dict]]:
        """Return sanitized debug info (no key names, no fingerprints, no raw provider errors)."""
        async with self._lock:
            self._sweep_key_invariants_locked()
            out = {}
            for svc, entries in self._keys.items():
                lst = []
                for i, ke in enumerate(entries):
                    now_ts = _now_ts()
                    override_effects = self._resolve_provider_override_effects(
                        provider=svc,
                        key_name_fingerprint=str(ke.name_fingerprint or "") or None,
                        key_value_fingerprint=str(ke.fingerprint or "") or None,
                    )
                    forced_exhausted_until = _future_ts_or_none(override_effects.get("force_exhausted_until"), now_ts)
                    force_active_until = _future_ts_or_none(override_effects.get("force_active_until"), now_ts)
                    clear_exhaustion = bool(override_effects.get("clear_exhaustion"))
                    active = True
                    if forced_exhausted_until is not None:
                        active = False
                    elif force_active_until is None and not clear_exhaustion and (ke.exhausted_until and ke.exhausted_until > now_ts):
                        active = False
                    exhausted_iso = None
                    if ke.exhausted_until:
                        exhausted_iso = datetime.fromtimestamp(ke.exhausted_until, tz=UTC).isoformat()
                    row = {
                        "index": i,
                        "active": active,
                        "in_use": ke.in_use,
                        "exhausted_until": exhausted_iso,
                        "pending_exhaust": ke._pending_exhaust,
                        "pending_clear": ke._pending_clear,
                        "failure_classification": ke.failure_classification,
                        "last_exhausted_reason_class": ke.last_exhausted_reason_class,
                        "last_exhausted_at": self._iso_from_timestamp(ke.last_exhausted_at),
                        "has_manual_override": bool(override_effects.get("override_ids")),
                    }
                    # SerpAPI-specific operational metadata kept coarse and non-secret.
                    if svc == "serpapi":
                        row["searches_left"] = ke.searches_left
                        row["last_checked_at"] = self._iso_from_timestamp(ke.last_checked_at)
                        row["default_reset_day"] = ke.default_reset_day
                    lst.append(row)
                out[svc] = lst
        return out

    # ---------- public aliases (used by app.py and health.py) ----------
    async def load_env_keys(self):
        """Force-reload keys from environment. Called at startup."""
        await self._reload_env_keys_if_changed()
        await self._hydrate_serpapi_state_from_db()
        await self._persist_exhaustion()

    async def get_status(self) -> Dict[str, List[Dict]]:
        """Public alias for status(). Used by health.py routes."""
        return await self.status()

    async def refresh_from_env(self, sync: bool = False):
        """Reload keys from merged runtime/.env environment. Used by health endpoints."""
        await self._reload_env_keys_if_changed()
        await self._hydrate_serpapi_state_from_db()
        await self._persist_exhaustion()

    async def record_usage(self, service: str, idx: int):
        """
        Record that a key was successfully used.
        Updates last_used timestamp. This is a lightweight tracking call —
        it does not increment in_use (that is handled by reserve_key).
        """
        snap: Optional[Dict[str, Any]] = None
        async with self._lock:
            try:
                ke = self._keys[service][idx]
                ke.last_used = time.monotonic()
                snap = self._persist_entry_state_locked(service, idx, ke, checked_now=False)
            except (KeyError, IndexError):
                logger.debug("record_usage: unknown service/index %s:%s", service, idx)
        if snap is not None:
            await self._persist_entry_snapshot_to_db(**snap)

# Global singleton instance
key_manager = APIKeyManager()
