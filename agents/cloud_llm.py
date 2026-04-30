# NOTE: The application must call close_client() on shutdown, e.g.:
# @app.on_event("shutdown")
# async def shutdown():
#     await cloud_llm.close_client()


# NOTE:
# Prometheus metrics (LLM_REQUESTS and LLM_LATENCY) are intentionally
# instrumented at the public API layer (generate/stream methods).
# This ensures:
#   - One metric emission per logical LLM request
#   - Correct provider labeling (cloud vs ollama fallback)
#   - Accurate end-to-end latency measurement
# Metrics are NOT added to lower-level transport methods to avoid
# double-counting retries or internal fallback attempts.
import time
import asyncio
import logging
import importlib
import hashlib
import threading
from datetime import datetime, UTC, timedelta
from contextlib import asynccontextmanager
from typing import Optional, Any, List, Tuple, Dict
from zoneinfo import ZoneInfo

# Core infrastructure
from core.retry import retry_async, RetryConfig
from core.circuit_breaker import get_circuit_breaker, CircuitBreakerOpenError, is_open as circuit_is_open
from core.request_context import get_request_id
from core.api_key_manager import key_manager
from core.env_config import get_env_bool, get_env_float, get_env_int, get_env_str, parse_csv_env
from core.llm_mode import get_cloud_provider_chain_resolution, get_llm_mode_default, LLM_MODE_OLLAMA_ONLY
import core.metrics as app_metrics

# Configure logging
logger = logging.getLogger(__name__)

# Environment configuration defaults (dynamic reads happen at runtime where required)
ENABLE_GEMINI_HELPER = get_env_bool("ENABLE_GEMINI_HELPER", default=False)
DEFAULT_MODEL = get_env_str("CLOUD_LLM_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT = get_env_float("CLOUD_LLM_TIMEOUT", 30.0)
DEFAULT_TEMPERATURE = get_env_float("CLOUD_LLM_TEMPERATURE", 0.7)
DEFAULT_MAX_TOKENS = get_env_int("CLOUD_LLM_MAX_TOKENS", 1024)
COST_PER_1K_TOKENS = get_env_float("CLOUD_LLM_COST_PER_1K_TOKENS", 0.0)
MAX_COST_PER_REQUEST = get_env_float("CLOUD_LLM_MAX_COST", 0.0)  # 0 = disabled
FALLBACK_MODELS = parse_csv_env("CLOUD_LLM_FALLBACK_MODELS")
STREAM_CHUNK_TIMEOUT = get_env_float("CLOUD_LLM_STREAM_CHUNK_TIMEOUT", 5.0)

# ---- Cooldown globals for health check ----
PROVIDER_FAIL_COOLDOWN = get_env_int("PROVIDER_FAIL_COOLDOWN", 300)  # seconds
_PROVIDER_FAIL_COOLDOWNS = {}  # provider_name -> unix timestamp until which we skip health checks
PROVIDER_AUTH_FAIL_COOLDOWN = get_env_int("PROVIDER_AUTH_FAIL_COOLDOWN", 900)
PROVIDER_TRANSIENT_FAIL_COOLDOWN = get_env_int("PROVIDER_TRANSIENT_FAIL_COOLDOWN", 60)
PROVIDER_CIRCUIT_OPEN_FAIL_COOLDOWN = get_env_int("PROVIDER_CIRCUIT_OPEN_FAIL_COOLDOWN", 45)
_FALLBACK_WARNING_STATE: Dict[str, Dict[str, float]] = {}
_EXHAUSTION_EVENT_DEDUP: Dict[str, float] = {}
# --------------------------------------------------

_CACHE_MANAGED_PROVIDERS = {"openai", "anthropic"}

_provider_chain_lock = threading.Lock()

# ---- Cloud enablement flag ----
def is_cloud_admin_enabled() -> bool:
    """
    Administrative cloud enablement switch.
    USE_CLOUD_LLM=0 disables cloud routing even when provider keys are usable.
    """
    return get_env_bool("USE_CLOUD_LLM", default=True)


# Backward-compatible snapshot retained for legacy log/debug surfaces.
USE_CLOUD_LLM = is_cloud_admin_enabled()
# --------------------------------------------------


def _fallback_warning_window_seconds() -> float:
    return max(5.0, get_env_float("CLOUD_LLM_FALLBACK_LOG_WINDOW_SECONDS", 30.0))


def _model_attempt_budget() -> int:
    return max(1, min(8, get_env_int("CLOUD_LLM_MODEL_ATTEMPT_BUDGET", 3)))


def _bounded_models_to_try(primary_model: str, fallback_models: List[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for candidate in [primary_model, *(fallback_models or [])]:
        normalized = str(candidate or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    if not ordered:
        ordered.append(primary_model)
    return ordered[:_model_attempt_budget()]


def _fallback_reason_bucket(exc: Exception) -> str:
    text = str(exc or "").lower()
    if _is_no_available_keys_error(exc):
        return "no_active_key"
    if "circuit breaker open" in text or "circuit_open" in text:
        return "circuit_open"
    if "rate limit" in text or "too many requests" in text or "429" in text:
        return "rate_limit"
    if "quota" in text or "insufficient_quota" in text:
        return "quota"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "unauthorized" in text or "forbidden" in text or "invalid api key" in text:
        return "auth"
    if "network" in text or "connect" in text or "unavailable" in text:
        return "transient"
    return type(exc).__name__.lower()


def _should_emit_fallback_warning(scope: str, bucket: str) -> tuple[bool, int]:
    key = f"{scope}:{bucket}"
    now = time.monotonic()
    window_seconds = _fallback_warning_window_seconds()
    state = _FALLBACK_WARNING_STATE.get(key)
    if state is None or (now - float(state.get("window_start", now))) > window_seconds:
        _FALLBACK_WARNING_STATE[key] = {"window_start": now, "count": 1}
        if len(_FALLBACK_WARNING_STATE) > 512:
            stale_before = now - (window_seconds * 2.0)
            for stale_key in list(_FALLBACK_WARNING_STATE.keys()):
                started = float(_FALLBACK_WARNING_STATE.get(stale_key, {}).get("window_start", now))
                if started < stale_before:
                    _FALLBACK_WARNING_STATE.pop(stale_key, None)
        return True, 1

    state["count"] = int(state.get("count", 1)) + 1
    count = int(state["count"])
    # Keep the first warning, then sparse milestone warnings only.
    if count in {5, 20, 50}:
        return True, count
    return False, count


def _should_emit_exhaustion_reaction_log(event_key: str) -> bool:
    now = time.monotonic()
    ttl = max(5.0, get_env_float("KEY_EVENT_DEDUP_WINDOW_SECONDS", 30.0))
    previous = _EXHAUSTION_EVENT_DEDUP.get(event_key)
    if previous is not None and (now - previous) < ttl:
        return False
    _EXHAUSTION_EVENT_DEDUP[event_key] = now
    if len(_EXHAUSTION_EVENT_DEDUP) > 1024:
        stale_before = now - (ttl * 2.0)
        for key in list(_EXHAUSTION_EVENT_DEDUP.keys()):
            if _EXHAUSTION_EVENT_DEDUP.get(key, now) < stale_before:
                _EXHAUSTION_EVENT_DEDUP.pop(key, None)
    return True


def _is_cache_managed_provider(provider: Any) -> bool:
    normalized = str(provider or "").strip().lower()
    return normalized in _CACHE_MANAGED_PROVIDERS


def _openai_exhaustion_reason(exc: Exception) -> str:
    """
    Distinguish account/billing exhaustion from ordinary rate-limit cooldowns.
    OpenAI billing exhaustion should be treated as quota/billing domain, not daily-reset style.
    """
    text = str(exc or "").lower()
    if any(token in text for token in ("insufficient_quota", "billing", "credit", "payment required", "quota")):
        return "billing_quota_exhausted"
    return "rate_limit"


def _gemini_rate_scope_mode() -> str:
    mode = get_env_str("GEMINI_QUOTA_SCOPE_MODE", "project_or_provider").strip().lower()
    if mode in {"project_or_provider", "provider_account", "project", "key"}:
        return mode
    return "project_or_provider"


def _gemini_scope_binding(idx: int) -> tuple[str, str]:
    # Prefer explicit per-key project mapping when provided.
    project_by_key = get_env_str(f"GEMINI_PROJECT_ID_{int(idx) + 1}", "").strip()
    if not project_by_key:
        project_by_key = get_env_str(f"GEMINI_PROJECT_{int(idx) + 1}", "").strip()
    if project_by_key:
        return "project", project_by_key

    # Then global project hints.
    global_project = get_env_str("GEMINI_PROJECT_ID", "").strip()
    if not global_project:
        global_project = get_env_str("GOOGLE_CLOUD_PROJECT", "").strip()
    if global_project:
        return "project", global_project

    # Fallback: provider-account scoped hold (safe default when project mapping is unknown).
    provider_account = get_env_str("GEMINI_PROVIDER_ACCOUNT_ID", "").strip()
    if not provider_account:
        provider_account = get_env_str("GEMINI_ACCOUNT_ID", "").strip()
    return "provider_account", (provider_account or "default")


def _next_midnight_pacific_utc() -> datetime:
    try:
        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        next_midnight_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return next_midnight_pt.astimezone(UTC)
    except Exception:
        # Fallback if timezone database is unavailable.
        return datetime.now(UTC) + timedelta(days=1)


def _gemini_exhaustion_bucket(exc: Exception) -> str:
    text = str(exc or "").lower()
    if any(token in text for token in ("unauthorized", "invalid api key", "invalid key", "permission denied", "403", "401", "auth")):
        return "auth"
    if any(token in text for token in ("insufficient_quota", "quota", "resource exhausted", "billing")):
        return "quota"
    if any(token in text for token in ("rate limit", "too many requests", "429", "ratelimit")):
        return "rate_limit"
    return "unknown"


async def _apply_gemini_runtime_exhaustion(idx: int, bucket: str, error_text: str) -> None:
    """
    Gemini default policy is project/provider-account scoped unless explicitly set to key mode.
    This avoids assuming independent per-key quota truth when project/account mapping is unknown.
    """
    mode = _gemini_rate_scope_mode()
    normalized_bucket = str(bucket or "unknown").strip().lower()
    if mode == "key":
        reason = "rate_limit" if normalized_bucket in {"quota", "rate_limit"} else "unauthorized"
        await key_manager.mark_exhausted("gemini", idx, reason=reason)
        return

    scope_type, scope_identifier = _gemini_scope_binding(idx)
    if mode == "project" and scope_type != "project":
        scope_type, scope_identifier = ("project", "default")
    if mode == "provider_account":
        scope_type, scope_identifier = ("provider_account", "default")

    now_utc = datetime.now(UTC)
    if normalized_bucket == "quota":
        until = _next_midnight_pacific_utc()
        note = "auto_runtime_quota_project_scoped"
    else:
        until = now_utc + timedelta(seconds=max(60, int(PROVIDER_FAIL_COOLDOWN)))
        note = "auto_runtime_rate_limit_project_scoped"

    # Dedup: avoid re-applying the same scope+type override within the dedup window
    dedup_key = f"gemini_override|{scope_type}|{scope_identifier}|{note}"
    if not _should_emit_exhaustion_reaction_log(dedup_key):
        logger.debug(
            "Gemini runtime exhaustion override deduped (scope=%s identifier=%s)",
            scope_type,
            scope_identifier,
        )
        return

    try:
        await key_manager.set_provider_state_override(
            provider="gemini",
            scope_type=scope_type,
            scope_identifier=scope_identifier,
            override_type="force_exhausted_until",
            active_until=until.isoformat(),
            note=note,
        )
        logger.info(
            "Applied Gemini runtime exhaustion override",
            extra={
                "provider": "gemini",
                "scope_type": scope_type,
                "reason_bucket": normalized_bucket,
                "until": until.isoformat(),
            },
        )
    except Exception:
        logger.exception(
            "gemini_runtime_override_failed_falling_back_to_key",
            extra={"provider": "gemini", "reason_bucket": normalized_bucket, "idx": idx},
        )
        fallback_reason = "rate_limit" if normalized_bucket in {"quota", "rate_limit"} else "unauthorized"
        await key_manager.mark_exhausted("gemini", idx, reason=fallback_reason)

# Lazy imports for optional provider SDKs
try:
    from openai import AsyncOpenAI, RateLimitError, APIConnectionError, AuthenticationError as OpenAIAuthError
except ImportError:
    AsyncOpenAI = None
    RateLimitError = None
    APIConnectionError = None
    OpenAIAuthError = None

try:
    from anthropic import AsyncAnthropic, RateLimitError as AnthroRateLimitError, APIConnectionError as AnthroConnErr, AuthenticationError as AnthroAuthError
except ImportError:
    AsyncAnthropic = None
    AnthroRateLimitError = None
    AnthroConnErr = None
    AnthroAuthError = None


# ----------------------------------------------------------------------
# Client cache with in‑use tracking, per‑entry locks, and fingerprints
# ----------------------------------------------------------------------
_clients: Dict[Tuple[str, int], dict] = {}       # (provider, idx) -> metadata dict
_client_lock = asyncio.Lock()

async def _create_client(provider: str, idx: int, key: str):
    """Create a new client for the given provider and key."""
    if provider == "openai":
        if AsyncOpenAI is None:
            raise CloudLLMError("OpenAI SDK not installed")
        return AsyncOpenAI(api_key=key)
    elif provider == "anthropic":
        if AsyncAnthropic is None:
            raise CloudLLMError("Anthropic SDK not installed")
        return AsyncAnthropic(api_key=key)
    else:
        raise CloudLLMError(f"Unsupported provider for client creation: {provider}")

async def _close_client(client):
    """Safely close a client if it has a close method."""
    try:
        if hasattr(client, "aclose"):
            await client.aclose()
        elif hasattr(client, "close"):
            maybe = client.close()
            if asyncio.iscoroutine(maybe):
                await maybe
    except Exception:
        logger.exception("Error closing cached client")

@asynccontextmanager
async def get_client(provider: str, idx: int, key: str):
    """
    Async context manager that yields a cached client for (provider, idx).
    Increments _in_use while the client is held; refuses if _pending_clear is set.
    Uses per‑entry locks for atomicity.
    """
    cache_key = (provider, idx)
    entry_obj = None
    created_local_client = None

    # Fast path: try to reuse an existing entry
    async with _client_lock:
        existing = _clients.get(cache_key)

    if existing:
        # Ensure the per-entry lock exists (create if missing) under global lock to avoid races
        if "_lock" not in existing:
            async with _client_lock:
                if "_lock" not in existing:
                    existing["_lock"] = asyncio.Lock()

        entry_lock = existing["_lock"]
        # Acquire the per-entry lock to check pending_clear and bump _in_use atomically
        async with entry_lock:
            # It's possible another coroutine cleared the entry concurrently; re-check presence
            async with _client_lock:
                if cache_key not in _clients:
                    existing = None
                else:
                    existing = _clients[cache_key]

            if existing is None:
                # fall through to creation path
                pass
            else:
                if existing.get("_pending_clear"):
                    raise CloudLLMError(f"Client for {provider}:{idx} is pending clear and cannot be used")
                existing["_in_use"] = existing.get("_in_use", 0) + 1
                entry_obj = existing

    # If absent, create client outside the lock (avoid blocking other coros)
    if entry_obj is None:
        created_local_client = await _create_client(provider, idx, key)
        # Compute a stable fingerprint for this key so env-change events can target exact instances
        key_fingerprint = hashlib.sha256(key.encode()).hexdigest()
        new_entry = {
            "client": created_local_client,
            "_in_use": 1,
            "_pending_clear": False,
            "created_at": time.time(),
            "_lock": asyncio.Lock(),
            "fingerprint": key_fingerprint,
        }

        async with _client_lock:
            # double-check whether another coroutine created it meanwhile
            existing = _clients.get(cache_key)
            if existing:
                # ensure per-entry lock exists
                if "_lock" not in existing:
                    existing["_lock"] = asyncio.Lock()
                entry_lock = existing["_lock"]

                async with entry_lock:
                    if existing.get("_pending_clear"):
                        # If theirs is pending_clear, drop our client and error
                        await _close_client(created_local_client)
                        raise CloudLLMError(f"Client for {provider}:{idx} is pending clear and cannot be used")
                    existing["_in_use"] = existing.get("_in_use", 0) + 1
                    entry_obj = existing
                    # close our unused created client to avoid leaks
                    await _close_client(created_local_client)
                    created_local_client = None
            else:
                _clients[cache_key] = new_entry
                entry_obj = new_entry

    try:
        yield entry_obj["client"]
    finally:
        # decrement the specific entry object we incremented earlier using its per-entry lock
        if entry_obj:
            entry_lock = entry_obj.get("_lock")
            if entry_lock:
                async with entry_lock:
                    entry_obj["_in_use"] = max(0, entry_obj.get("_in_use", 1) - 1)
            else:
                # fallback (should not happen)
                async with _client_lock:
                    entry_obj["_in_use"] = max(0, entry_obj.get("_in_use", 1) - 1)

async def clear_client_cache(provider: str, idx: int, timeout: float = 30.0, expected_fingerprint: str | None = None) -> bool:
    """
    Safely remove the cached client for (provider, idx).
    Marks it pending_clear, waits for _in_use to become zero (with timeout), then closes and removes it.
    If expected_fingerprint is provided, it must match the current entry's fingerprint to proceed.
    Returns True if removed, False if timed out or fingerprint mismatch.
    """
    cache_key = (provider, idx)
    start = time.monotonic()

    async with _client_lock:
        entry = _clients.get(cache_key)
        if not entry:
            return True  # already gone
        # If caller provided an expected_fingerprint, ensure it matches the current entry
        if expected_fingerprint is not None:
            actual_fp = entry.get("fingerprint")
            if actual_fp != expected_fingerprint:
                logger.debug(
                    "clear_client_cache skipped for %s:%d due to fingerprint mismatch (expected=%s actual=%s)",
                    provider, idx, expected_fingerprint, actual_fp
                )
                return False
        # Now mark pending_clear while holding client global lock (we will also acquire per-entry lock below)
        entry["_pending_clear"] = True

    # Wait for in_use to drop to zero (exponential backoff)
    backoff = 0.1
    while True:
        async with _client_lock:
            current_entry = _clients.get(cache_key)
        if not current_entry:
            # already removed
            break

        entry_lock = current_entry.get("_lock")
        if entry_lock is None:
            # create a lock if somehow missing (under global lock)
            async with _client_lock:
                current_entry = _clients.get(cache_key)
                if current_entry and "_lock" not in current_entry:
                    current_entry["_lock"] = asyncio.Lock()
                entry_lock = current_entry.get("_lock")

        async with entry_lock:
            # Re-read in_use while holding per-entry lock
            in_use = current_entry.get("_in_use", 0)
            if in_use == 0:
                break

        if time.monotonic() - start > timeout:
            # Snapshot in_use under per-entry lock for the log
            async with entry_lock:
                final_in_use = current_entry.get("_in_use", 0)
            logger.warning(
                "clear_client_cache timed out for %s:%d (in_use=%d), force-closing",
                provider, idx, final_in_use
            )
            break

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 1.0)

    # Now close and remove – hold BOTH locks to ensure atomic pop
    async with _client_lock:
        entry = _clients.get(cache_key)
        if not entry:
            return True

        entry_lock = entry.get("_lock")
        if entry_lock:
            async with entry_lock:
                # verify fingerprint once more
                if expected_fingerprint is not None and entry.get("fingerprint") != expected_fingerprint:
                    logger.debug("clear_client_cache aborted at removal: fingerprint changed for %s:%d", provider, idx)
                    return False
                popped = _clients.pop(cache_key, None)
        else:
            # fallback if no per-entry lock (should not happen)
            popped = _clients.pop(cache_key, None)

    # Close outside both locks to avoid blocking
    if popped:
        await _close_client(popped["client"])
        logger.debug("Cleared cached client for %s:%d", provider, idx)
        return True
    return False

async def close_all_clients():
    """Close all cached clients (used during shutdown)."""
    # Mark all as pending clear to block new acquisitions
    async with _client_lock:
        for entry in _clients.values():
            entry["_pending_clear"] = True

    # Wait briefly for in_use to drop
    for cache_key, entry in list(_clients.items()):
        waited = 0.0
        while waited < 5.0:
            async with _client_lock:
                if entry["_in_use"] == 0:
                    break
            await asyncio.sleep(0.1)
            waited += 0.1

    # Now close all clients
    async with _client_lock:
        for cache_key, entry in list(_clients.items()):
            await _close_client(entry["client"])
            _clients.pop(cache_key, None)
    logger.info("All cached clients closed")


# ----------------------------------------------------------------------
# Key event listener (to be registered by application startup)
# ----------------------------------------------------------------------
async def on_key_event(event: str, payload: dict):
    """
    Handle key-manager events for cloud provider client-cache hygiene.
    Expected events: "key_exhausted", "env_changed".
    This listener only owns cached SDK clients for cloud LLM providers
    (currently openai/anthropic). Non-cloud services are intentionally ignored.
    """
    try:
        if event == "key_exhausted":
            provider = payload.get("service")
            idx = payload.get("index")
            if provider and idx is not None:
                # Dedup by service+idx+reason_class only — pending status is a
                # lifecycle detail, not a distinct event worth logging twice.
                dedup_key = "|".join(
                    [
                        str(provider),
                        str(idx),
                        str(payload.get("reason_class") or ""),
                    ]
                )
                if not _is_cache_managed_provider(provider):
                    if _should_emit_exhaustion_reaction_log(dedup_key):
                        logger.debug(
                            "api_key_manager key_exhausted for %s:%d; cloud client-cache listener skipped (provider has no cached cloud SDK client)",
                            provider,
                            idx,
                        )
                    return
                if _should_emit_exhaustion_reaction_log(dedup_key):
                    logger.info(
                        "api_key_manager key_exhausted for %s:%d; cloud client-cache listener clearing cached cloud SDK client state",
                        provider,
                        idx,
                    )
                else:
                    logger.debug(
                        "Deduped key_exhausted reaction log for %s:%d; cloud client-cache listener clearing cache silently",
                        provider,
                        idx,
                    )
                asyncio.create_task(clear_client_cache(provider, idx))
        elif event == "env_changed":
            # payload may include new_fingerprint_maps / old_fingerprint_maps
            new_maps = payload.get("new_fingerprint_maps", {}) or {}
            old_maps = payload.get("old_fingerprint_maps", {}) or {}
            affected = payload.get("affected", [])
            # Keep provider init/runtime capability aligned with current env config.
            refresh_provider_chain_from_env(force=True)
            for provider, idx in affected:
                if not _is_cache_managed_provider(provider):
                    logger.debug(
                        "api_key_manager env_changed includes %s:%d; cloud client-cache listener skipped (provider has no cached cloud SDK client)",
                        provider,
                        idx,
                    )
                    continue
                # try to find the new fingerprint for this provider/idx (fall back to old)
                expected_fp = None
                try:
                    expected_fp = new_maps.get(provider, {}).get(idx)
                    if expected_fp is None:
                        expected_fp = old_maps.get(provider, {}).get(idx)
                except Exception:
                    # if maps use string keys or other shape, do a best-effort
                    pass

                logger.debug(
                    "api_key_manager env_changed for %s:%d; cloud client-cache listener clearing cache (expected_fp=%s)",
                    provider,
                    idx,
                    expected_fp,
                )
                asyncio.create_task(clear_client_cache(provider, idx, expected_fingerprint=expected_fp))
        else:
            logger.debug("Ignoring unknown key event: %s", event)
    except Exception:
        logger.exception("on_key_event raised unexpectedly (event=%s)", event)


# ----------------------------------------------------------------------
# Helper: per‑chunk timeout wrapper for async iterators
# ----------------------------------------------------------------------
async def _aiter_with_timeout(aiter, per_item_timeout: float):
    """
    Wrap an async iterable so each __anext__ is awaited with a timeout.
    Yields items until StopAsyncIteration or a per-item timeout occurs.
    """
    ait = aiter.__aiter__()
    while True:
        try:
            item = await asyncio.wait_for(ait.__anext__(), timeout=per_item_timeout)
        except asyncio.TimeoutError:
            raise
        except StopAsyncIteration:
            break
        yield item


# ----------------------------------------------------------------------
# Provider Adapter – normalizes API responses to a consistent shape
# ----------------------------------------------------------------------
class ProviderAdapter:
    def __init__(self, provider: str, gemini_module=None, gemini_instance=None):
        self.provider = provider
        # For Gemini we keep the module and instance (if any) for reuse, but keys come from key_manager
        self.gemini_module = gemini_module
        self.gemini_instance = gemini_instance

    async def ping(self, model: str) -> None:
        """Perform a minimal health check call. Raises exception on failure."""
        # Use a minimal completion
        await self.create_completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=1,
            timeout=5.0
        )

    async def create_completion(self, *, model, messages, temperature, max_tokens, timeout):
        """Non‑streaming completion. Returns a normalized object with .choices[0].message.content and .usage."""
        provider = self.provider

        if provider == "openai":
            async with _reserve_provider_key("openai") as (idx, key):
                async with get_client(provider, idx, key) as client:
                    try:
                        raw = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            ),
                            timeout=timeout
                        )
                        await key_manager.record_usage("openai", idx)
                        return raw
                    except OpenAIAuthError as e:
                        await key_manager.mark_exhausted("openai", idx, reason="unauthorized")
                        raise
                    except RateLimitError as e:
                        await key_manager.mark_exhausted("openai", idx, reason=_openai_exhaustion_reason(e))
                        raise
                    except (APIConnectionError, asyncio.TimeoutError) as e:
                        # Transient errors, don't mark exhausted
                        raise
                    except Exception as e:
                        raise

        elif provider == "anthropic":
            async with _reserve_provider_key("anthropic") as (idx, key):
                async with get_client(provider, idx, key) as client:
                    # Convert messages to Anthropic prompt format (simplified)
                    system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
                    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
                    prompt = "\n\n".join(user_msgs)

                    try:
                        raw = await asyncio.wait_for(
                            client.completions.create(
                                model=model,
                                prompt=prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            ),
                            timeout=timeout
                        )
                        await key_manager.record_usage("anthropic", idx)
                        # Normalize to match OpenAI's response structure
                        class FakeChoice:
                            def __init__(self, text):
                                self.message = type("Message", (), {"content": text})
                        class FakeResponse:
                            def __init__(self, text, usage):
                                self.choices = [FakeChoice(text)]
                                self.usage = usage
                        text = raw.completion if hasattr(raw, "completion") else raw.choices[0].text
                        usage = getattr(raw, "usage", None)
                        return FakeResponse(text, usage)
                    except AnthroAuthError as e:
                        await key_manager.mark_exhausted("anthropic", idx, reason="unauthorized")
                        raise
                    except AnthroRateLimitError as e:
                        await key_manager.mark_exhausted("anthropic", idx, reason="rate_limit")
                        raise
                    except (AnthroConnErr, asyncio.TimeoutError) as e:
                        raise
                    except Exception as e:
                        raise

        elif provider == "gemini":
            # Use the Gemini helper module stored during init
            if self.gemini_module is None:
                raise CloudLLMError("Gemini helper not available")
            async with _reserve_provider_key("gemini") as (idx, key):
                # Build a prompt from messages
                system_part = ""
                user_part = ""
                for msg in messages:
                    if msg["role"] == "system":
                        system_part = msg["content"]
                    elif msg["role"] == "user":
                        user_part = msg["content"]
                prompt = f"{system_part}\n\n{user_part}" if system_part else user_part

                # Define a sync function that calls the helper with the key
                def sync_call() -> str:
                    # Try client method first if we have an instance that can accept a key
                    if self.gemini_instance is not None and hasattr(self.gemini_instance, "generate"):
                        return self.gemini_instance.generate(prompt, model=model, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                    # Then module-level generate() with api_key param
                    if hasattr(self.gemini_module, "generate"):
                        return self.gemini_module.generate(prompt, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                    # Finally generate_single()
                    if hasattr(self.gemini_module, "generate_single"):
                        return self.gemini_module.generate_single(prompt, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                    raise RuntimeError("No usable Gemini entrypoint found")

                try:
                    content = await asyncio.to_thread(sync_call)
                    await key_manager.record_usage("gemini", idx)
                    # Build a fake response
                    class FakeChoice:
                        def __init__(self, text):
                            self.message = type("Message", (), {"content": text})
                    class FakeResponse:
                        def __init__(self, text):
                            self.choices = [FakeChoice(text)]
                            self.usage = None
                    return FakeResponse(content)
                except Exception as e:
                    bucket = _gemini_exhaustion_bucket(e)
                    if bucket in {"quota", "rate_limit"}:
                        await _apply_gemini_runtime_exhaustion(idx, bucket, str(e))
                    elif bucket == "auth":
                        await key_manager.mark_exhausted("gemini", idx, reason="unauthorized")
                    raise

        else:
            raise RuntimeError(f"Unsupported provider: {provider}")

    async def open_stream(self, *, model, messages, temperature, max_tokens, timeout):
        """Open a streaming connection. Returns an async iterable yielding normalized chunks."""
        provider = self.provider

        if provider == "openai":
            async def _stream_generator():
                async with _reserve_provider_key("openai") as (idx, key):
                    async with get_client(provider, idx, key) as client:
                        try:
                            stream = await asyncio.wait_for(
                                client.chat.completions.create(
                                    model=model,
                                    messages=messages,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    stream=True,
                                ),
                                timeout=timeout
                            )
                        except OpenAIAuthError as e:
                            await key_manager.mark_exhausted("openai", idx, reason="unauthorized")
                            raise
                        except RateLimitError as e:
                            await key_manager.mark_exhausted("openai", idx, reason=_openai_exhaustion_reason(e))
                            raise
                        except (APIConnectionError, asyncio.TimeoutError) as e:
                            raise
                        except Exception as e:
                            raise

                        async for chunk in stream:
                            yield chunk

                        await key_manager.record_usage("openai", idx)
            return _stream_generator()

        elif provider == "anthropic":
            async def _stream_generator():
                async with _reserve_provider_key("anthropic") as (idx, key):
                    async with get_client(provider, idx, key) as client:
                        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
                        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
                        prompt = "\n\n".join(user_msgs)
                        try:
                            raw_stream = await asyncio.wait_for(
                                client.completions.create(
                                    model=model,
                                    prompt=prompt,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    stream=True,
                                ),
                                timeout=timeout
                            )
                        except AnthroAuthError as e:
                            await key_manager.mark_exhausted("anthropic", idx, reason="unauthorized")
                            raise
                        except AnthroRateLimitError as e:
                            await key_manager.mark_exhausted("anthropic", idx, reason="rate_limit")
                            raise
                        except (AnthroConnErr, asyncio.TimeoutError) as e:
                            raise
                        except Exception as e:
                            raise

                        async for chunk in raw_stream:
                            delta_text = getattr(chunk, "completion", None) or chunk.choices[0].text
                            if delta_text is None:
                                continue
                            # Build fake chunk object
                            class FakeDelta:
                                def __init__(self, content): self.content = content
                            class FakeChoice:
                                def __init__(self, delta): self.delta = delta
                            class FakeChunk:
                                def __init__(self, delta): self.choices = [FakeChoice(FakeDelta(delta))]
                            yield FakeChunk(delta_text)

                        await key_manager.record_usage("anthropic", idx)
            return _stream_generator()

        elif provider == "gemini":
            # Gemini streaming not natively supported; fallback to non-streaming
            async def _stream_generator():
                async with _reserve_provider_key("gemini") as (idx, key):
                    # Build prompt
                    system_part = ""
                    user_part = ""
                    for msg in messages:
                        if msg["role"] == "system":
                            system_part = msg["content"]
                        elif msg["role"] == "user":
                            user_part = msg["content"]
                    prompt = f"{system_part}\n\n{user_part}" if system_part else user_part

                    def sync_call() -> str:
                        if self.gemini_instance is not None and hasattr(self.gemini_instance, "generate"):
                            return self.gemini_instance.generate(prompt, model=model, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                        if hasattr(self.gemini_module, "generate"):
                            return self.gemini_module.generate(prompt, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                        if hasattr(self.gemini_module, "generate_single"):
                            return self.gemini_module.generate_single(prompt, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                        raise RuntimeError("No usable Gemini entrypoint found")

                    try:
                        content = await asyncio.to_thread(sync_call)
                    except Exception as e:
                        bucket = _gemini_exhaustion_bucket(e)
                        if bucket in {"quota", "rate_limit"}:
                            await _apply_gemini_runtime_exhaustion(idx, bucket, str(e))
                        elif bucket == "auth":
                            await key_manager.mark_exhausted("gemini", idx, reason="unauthorized")
                        raise

                    # Yield as one chunk
                    class FakeDelta:
                        def __init__(self, content): self.content = content
                    class FakeChoice:
                        def __init__(self, delta): self.delta = delta
                    class FakeChunk:
                        def __init__(self, delta): self.choices = [FakeChoice(FakeDelta(delta))]
                    yield FakeChunk(content)

                    await key_manager.record_usage("gemini", idx)
            return _stream_generator()

        else:
            raise RuntimeError(f"Unsupported provider: {provider}")

    async def close(self):
        """Close any persistent resources (Gemini client may have its own close)."""
        try:
            if self.provider == "gemini" and self.gemini_instance is not None:
                close_method = getattr(self.gemini_instance, "close", None)
                if callable(close_method):
                    maybe = close_method()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            logger.info("Cloud provider adapter closed", extra={"provider": self.provider})
        except Exception as e:
            logger.exception("cloud_adapter_close_failed", extra={"provider": self.provider, "error": str(e)})


# ----------------------------------------------------------------------
# Build the provider chain
# ----------------------------------------------------------------------
_provider_chain_signature: Optional[Tuple[Tuple[str, ...], bool, str]] = None
_configured_provider_order: List[str] = []
_configured_provider_source: str = "default"
_provider_init_status: Dict[str, Dict[str, Any]] = {}


def _configured_provider_chain_from_env() -> List[str]:
    resolution = get_cloud_provider_chain_resolution()
    return [part.strip().lower() for part in resolution["providers"] if str(part).strip()]


def _gemini_helper_enabled() -> bool:
    # Runtime env is authoritative; keep module snapshot only as fallback default.
    return get_env_bool("ENABLE_GEMINI_HELPER", default=ENABLE_GEMINI_HELPER)


def _cloud_startup_routing_relevant() -> bool:
    mode = str(get_llm_mode_default() or "").strip().lower()
    return mode != LLM_MODE_OLLAMA_ONLY


def _set_provider_init_status(provider: str, *, initialized: bool, reason: str) -> None:
    _provider_init_status[provider] = {
        "configured": True,
        "initialized": bool(initialized),
        "reason": reason,
    }


def _provider_runtime_capability_snapshot() -> Dict[str, Any]:
    return {
        "configured_providers": list(_configured_provider_order),
        "configured_provider_source": _configured_provider_source,
        "available_providers": [name for name, _, _ in provider_chain],
        "provider_init_status": dict(_provider_init_status),
        "default_provider": DEFAULT_PROVIDER,
        "gemini_default_intent": (_configured_provider_order[0] if _configured_provider_order else None) == "gemini",
    }


def _parse_provider_chain() -> List[str]:
    """Return configured provider order (deduplicated, runtime env aware)."""
    seen: set[str] = set()
    out: List[str] = []
    for provider in _configured_provider_chain_from_env():
        if provider and provider not in seen:
            seen.add(provider)
            out.append(provider)
    return out


def _init_provider(provider: str):
    """Initialise a single provider. Returns (adapter, retry_exceptions) or None if skipped."""
    if provider == "openai":
        if AsyncOpenAI is None:
            _set_provider_init_status(provider, initialized=False, reason="sdk_missing_openai")
            if _cloud_startup_routing_relevant():
                logger.warning("OpenAI package not installed - skipping provider.")
            else:
                logger.debug("OpenAI package not installed - skipping provider (non-routing mode).")
            return None
        # No client created here; key_manager will provide keys at call time
        _set_provider_init_status(provider, initialized=True, reason="ok")
        return ProviderAdapter(provider), (RateLimitError, APIConnectionError, asyncio.TimeoutError, OpenAIAuthError)

    elif provider == "anthropic":
        # Skip if SDK not installed (but we already have lazy imports)
        if AsyncAnthropic is None:
            _set_provider_init_status(provider, initialized=False, reason="sdk_missing_anthropic")
            if _cloud_startup_routing_relevant():
                logger.warning("Anthropic package not installed - skipping provider.")
            else:
                logger.debug("Anthropic package not installed - skipping provider (non-routing mode).")
            return None
        _set_provider_init_status(provider, initialized=True, reason="ok")
        return ProviderAdapter(provider), (AnthroRateLimitError, AnthroConnErr, asyncio.TimeoutError, AnthroAuthError)

    elif provider == "gemini":
        # Legacy Gemini helper path is opt-in only.
        # This avoids noisy startup warnings when the historical helper module
        # has already been removed from the deployment.
        if not _gemini_helper_enabled():
            _set_provider_init_status(provider, initialized=False, reason="gemini_helper_disabled")
            logger.debug("Gemini legacy helper disabled (ENABLE_GEMINI_HELPER=0) - skipping provider.")
            return None
        try:
            gemini_mod = importlib.import_module("gemini_multikey_9_3_helper_script")
        except ModuleNotFoundError:
            _set_provider_init_status(provider, initialized=False, reason="gemini_helper_module_missing")
            logger.debug("Gemini legacy helper module not found - skipping provider.")
            return None
        except Exception as e:
            _set_provider_init_status(provider, initialized=False, reason="gemini_helper_import_failed")
            if _cloud_startup_routing_relevant():
                logger.warning("Gemini legacy helper import failed - skipping provider: %s", e)
            else:
                logger.debug("Gemini legacy helper import failed - skipping provider in non-routing mode: %s", e)
            return None

        # Look for a GeminiClient class, otherwise use module-level generate functions
        GeminiClient = getattr(gemini_mod, "GeminiClient", None)
        gemini_instance = None
        if GeminiClient is not None:
            # Instantiate without a key; we'll pass key per call
            try:
                gemini_instance = GeminiClient(api_key=None)
            except TypeError:
                try:
                    gemini_instance = GeminiClient()
                except Exception:
                    gemini_instance = None  # adjust if constructor requires key

        adapter = ProviderAdapter(provider, gemini_module=gemini_mod, gemini_instance=gemini_instance)
        # Only retry on timeout for Gemini (or other exceptions if we detect)
        gemini_retry_exc = (asyncio.TimeoutError,)
        _set_provider_init_status(provider, initialized=True, reason="ok")
        return adapter, gemini_retry_exc

    else:
        _set_provider_init_status(provider, initialized=False, reason="unsupported_provider")
        if _cloud_startup_routing_relevant():
            logger.warning("Unsupported provider %s - skipping.", provider)
        else:
            logger.debug("Unsupported provider %s - skipping in non-routing mode.", provider)
        return None


provider_chain: List[Tuple[str, ProviderAdapter, Tuple]] = []
_adapters = {}  # for closing later
DEFAULT_PROVIDER = None
_adapter = None


def refresh_provider_chain_from_env(*, force: bool = False) -> Dict[str, Any]:
    """Refresh configured/initialised providers from current runtime env."""
    global provider_chain, _adapters, DEFAULT_PROVIDER, _adapter
    global _provider_chain_signature, _configured_provider_order, _configured_provider_source

    provider_resolution = get_cloud_provider_chain_resolution()
    configured = _parse_provider_chain()
    configured_source = str(provider_resolution.get("source") or "unknown")
    signature = (tuple(configured), _gemini_helper_enabled(), configured_source)
    if not force and _provider_chain_signature == signature:
        return _provider_runtime_capability_snapshot()

    with _provider_chain_lock:
        # Re-check inside lock to avoid duplicate rebuilds under concurrency.
        if not force and _provider_chain_signature == signature:
            return _provider_runtime_capability_snapshot()

        _configured_provider_order = list(configured)
        _configured_provider_source = configured_source
        _provider_init_status.clear()

        new_provider_chain: List[Tuple[str, ProviderAdapter, Tuple]] = []
        new_adapters: Dict[str, ProviderAdapter] = {}

        for prov in configured:
            try:
                res = _init_provider(prov)
                if res is None:
                    logger.debug("Provider %s intentionally skipped during refresh.", prov)
                    continue
                adapter, retry_exc = res
                new_provider_chain.append((prov, adapter, retry_exc))
                new_adapters[prov] = adapter
                logger.info("Initialised cloud provider", extra={"provider": prov})
            except Exception as e:
                _set_provider_init_status(prov, initialized=False, reason="init_exception")
                logger.warning("Skipping provider %s due to initialization error: %s", prov, e, exc_info=True)
                continue

        provider_chain = new_provider_chain
        _adapters = new_adapters
        DEFAULT_PROVIDER = provider_chain[0][0] if provider_chain else None
        _adapter = _adapters.get(DEFAULT_PROVIDER) if DEFAULT_PROVIDER else None
        _provider_chain_signature = signature

        snapshot = _provider_runtime_capability_snapshot()
        routing_relevant = _cloud_startup_routing_relevant()
        if not provider_chain:
            if is_cloud_admin_enabled():
                if routing_relevant:
                    logger.warning(
                        "No cloud providers initialised from configured chain in this worker process; cloud routing unavailable.",
                        extra=snapshot,
                    )
                else:
                    logger.debug(
                        "No cloud providers initialised in this worker process (non-routing in llm_mode=ollama_only).",
                        extra=snapshot,
                    )
            else:
                logger.debug(
                    "Cloud providers not initialised in this worker process because cloud routing is disabled by configuration.",
                    extra=snapshot,
                )
        else:
            if routing_relevant:
                logger.info(
                    "Cloud provider runtime refresh complete",
                    extra=snapshot,
                )
            else:
                logger.debug(
                    "Cloud provider runtime refresh complete (non-routing in llm_mode=ollama_only).",
                    extra=snapshot,
                )
        return snapshot


def get_provider_runtime_status() -> Dict[str, Any]:
    refresh_provider_chain_from_env(force=False)
    return _provider_runtime_capability_snapshot()

# Initial refresh at import to preserve existing module-level behavior.
refresh_provider_chain_from_env(force=True)


class CloudLLMError(Exception):
    """Custom exception for cloud LLM errors."""
    pass


class CloudProviderUnavailableError(CloudLLMError):
    """Raised when a cloud provider has no currently available keys."""
    pass


def _is_no_available_keys_error(exc: Exception) -> bool:
    return "no available keys for service" in str(exc).lower()


@asynccontextmanager
async def _reserve_provider_key(service: str):
    try:
        async with key_manager.reserve_key(service) as reservation:
            yield reservation
    except RuntimeError as exc:
        if _is_no_available_keys_error(exc):
            raise CloudProviderUnavailableError(str(exc)) from exc
        raise


def _resolve_provider_entries(
    selected_provider: Optional[str],
    allow_provider_fallback: bool,
) -> List[Tuple[str, "ProviderAdapter", Tuple]]:
    entries = list(provider_chain)
    if not entries:
        return []

    if not selected_provider:
        return entries

    normalized = selected_provider.strip().lower()
    prioritized = [entry for entry in entries if entry[0] == normalized]
    if not prioritized:
        available = ", ".join(name for name, _, _ in entries) or "(none)"
        if allow_provider_fallback:
            logger.warning(
                "Selected cloud provider '%s' is not initialised; falling back to available providers: %s",
                selected_provider,
                available,
            )
            return entries
        raise CloudLLMError(
            f"Selected cloud provider '{selected_provider}' is not configured. "
            f"Available providers: {available}"
        )

    if not allow_provider_fallback:
        return prioritized

    remainder = [entry for entry in entries if entry[0] != normalized]
    return prioritized + remainder


def get_available_providers() -> List[str]:
    return [name for name, _, _ in provider_chain]


_PROVIDER_KEY_SERVICE_MAP = {
    "openai": "openai",
    "gemini": "gemini",
    "anthropic": "anthropic",
}


def _is_key_entry_usable(entry: Any, now_ts: float) -> bool:
    if isinstance(entry, str):
        return entry.strip().lower() == "active"

    if not isinstance(entry, dict):
        return False

    if bool(entry.get("pending_clear", False) or entry.get("_pending_clear", False)):
        return False

    if "active" in entry and not bool(entry.get("active")):
        return False

    exhausted_until = entry.get("exhausted_until")
    if exhausted_until in (None, ""):
        return True

    try:
        if isinstance(exhausted_until, (int, float)):
            return float(exhausted_until) <= now_ts
    except Exception:
        return False

    # When exhausted_until is already ISO text, key_manager.status() has typically
    # already folded this into `active`; if active==True we treat it as usable.
    return "active" in entry and bool(entry.get("active"))


async def get_provider_usability() -> Dict[str, bool]:
    await asyncio.to_thread(refresh_provider_chain_from_env, force=False)
    providers = [name for name, _, _ in provider_chain]
    if not providers:
        return {}

    try:
        status = await key_manager.get_status()
    except Exception:
        logger.exception("provider_usability_status_failed")
        return {provider: False for provider in providers}

    now_ts = time.time()
    usability: Dict[str, bool] = {}

    for provider in providers:
        service = _PROVIDER_KEY_SERVICE_MAP.get(provider)
        if not service:
            usability[provider] = False
            continue

        entries = (status or {}).get(service, [])
        if isinstance(entries, dict):
            iterable = list(entries.values())
        elif isinstance(entries, list):
            iterable = entries
        else:
            iterable = []

        usability[provider] = any(_is_key_entry_usable(entry, now_ts) for entry in iterable)

    return usability


async def get_usable_providers() -> List[str]:
    usability = await get_provider_usability()
    providers = [name for name, _, _ in provider_chain]
    return [provider for provider in providers if usability.get(provider, False)]


async def cloud_backend_is_usable(*, respect_admin_flag: bool = True) -> bool:
    if respect_admin_flag and not is_cloud_admin_enabled():
        return False
    return bool(await get_usable_providers())


async def _check_circuit_breaker(provider: str):
    """Raise CloudLLMError if circuit breaker for this provider is open."""
    if await circuit_is_open(f"cloud_llm_{provider}"):
        raise CloudLLMError(f"Circuit breaker open for provider {provider}")


def _estimate_cost(usage) -> float | None:
    """Calculate estimated cost based on token usage."""
    cost_per_1k = get_env_float("CLOUD_LLM_COST_PER_1K_TOKENS", COST_PER_1K_TOKENS)
    if cost_per_1k <= 0 or not usage:
        return None
    total_tokens = getattr(usage, "total_tokens", 0)
    return (total_tokens / 1000) * cost_per_1k


async def _call_adapter_completion(
    adapter: ProviderAdapter,
    retry_exc: Tuple,
    messages: list,
    model: str,
    temperature: float,
    timeout: float,
    max_tokens: int | None = None,
) -> Any:
    """
    Internal function to call a specific provider with retry, circuit breaker, and timeout.
    Returns the normalized response object.
    """
    start = time.monotonic()
    request_id = get_request_id()
    provider = adapter.provider

    async def _do_call():
        return await adapter.create_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    try:
        cfg = RetryConfig(retries=3, base_delay=1.0)
        response = await retry_async(
            _do_call,
            config=cfg,
            request_id=request_id,
            retry_exceptions=retry_exc   # use provider-specific retry exceptions
        )
        latency = time.monotonic() - start

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        log_extra = {
            "request_id": request_id,
            "provider": provider,
            "model": model,
            "latency_sec": round(latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        cost = _estimate_cost(usage)
        if cost is not None:
            log_extra["estimated_cost_usd"] = round(cost, 6)

        logger.debug("Cloud LLM success", extra=log_extra)
        return response

    except Exception as e:
        logger.error("Cloud LLM failure", extra={
            "request_id": request_id,
            "provider": provider,
            "error": str(e),
            "model": model,
        })
        raise


async def generate(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
    cloud_provider: str | None = None,
    allow_provider_fallback: bool = True,
) -> str:
    """
    Generate a non‑streaming completion with provider‑level and model‑level fallback,
    and cost guardrail.
    """
    if not is_cloud_admin_enabled():
        raise CloudLLMError("Cloud LLM is disabled by configuration (USE_CLOUD_LLM=0)")
    await asyncio.to_thread(refresh_provider_chain_from_env, force=False)

    # Handle defaults
    primary_model = (get_env_str("CLOUD_LLM_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL) if model is None else model
    temperature = get_env_float("CLOUD_LLM_TEMPERATURE", DEFAULT_TEMPERATURE) if temperature is None else temperature
    timeout = get_env_float("CLOUD_LLM_TIMEOUT", DEFAULT_TIMEOUT) if timeout is None else timeout
    max_tokens = get_env_int("CLOUD_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS) if max_tokens is None else max_tokens
    fallback_models = parse_csv_env("CLOUD_LLM_FALLBACK_MODELS", default=FALLBACK_MODELS)
    max_cost_per_request = get_env_float("CLOUD_LLM_MAX_COST", MAX_COST_PER_REQUEST)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    provider_entries = _resolve_provider_entries(cloud_provider, allow_provider_fallback)
    if not provider_entries:
        raise CloudLLMError("No cloud providers available")

    # For each provider in chain
    last_exception = None
    for prov_name, adapter, retry_exc in provider_entries:
        # Check circuit breaker for this provider
        try:
            await _check_circuit_breaker(prov_name)
        except CloudLLMError as e:
            reason_bucket = _fallback_reason_bucket(e)
            should_warn, occurrence = _should_emit_fallback_warning(
                scope=f"generate:{prov_name}:circuit",
                bucket=reason_bucket,
            )
            if should_warn:
                logger.warning(
                    "Skipping provider %s due to open circuit (occurrence=%d)",
                    prov_name,
                    occurrence,
                )
            last_exception = e
            continue

        # Build list of models to try: primary + fallbacks
        models_to_try = _bounded_models_to_try(primary_model, fallback_models)
        for attempt_model in models_to_try:
            try:
                response = await _call_adapter_completion(
                    adapter=adapter,
                    retry_exc=retry_exc,
                    messages=messages,
                    model=attempt_model,
                    temperature=temperature,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )

                # Validate response structure
                if not hasattr(response, "choices") or not response.choices:
                    raise CloudLLMError("Malformed response: missing choices")
                if not hasattr(response.choices[0], "message") or not hasattr(response.choices[0].message, "content"):
                    raise CloudLLMError("Malformed response: missing message content")
                content = response.choices[0].message.content
                if content is None:
                    raise CloudLLMError("Empty completion (content is None)")

                # Cost guardrail
                if max_cost_per_request > 0:
                    usage = getattr(response, "usage", None)
                    cost = _estimate_cost(usage)
                    if cost and cost > max_cost_per_request:
                        raise CloudLLMError(f"Cost exceeded: ${cost:.6f} > ${max_cost_per_request:.6f}")

                return content

            except Exception as e:
                last_exception = e
                if _is_no_available_keys_error(e):
                    logger.warning(
                        "Provider %s unavailable due to key exhaustion/unavailability",
                        prov_name,
                        extra={"provider": prov_name},
                    )
                    break
                reason_bucket = _fallback_reason_bucket(e)
                should_warn, occurrence = _should_emit_fallback_warning(
                    scope=f"generate:{prov_name}:model",
                    bucket=reason_bucket,
                )
                if should_warn:
                    logger.warning(
                        "Provider %s model %s failed before fallback (reason_bucket=%s, occurrence=%d)",
                        prov_name,
                        attempt_model,
                        reason_bucket,
                        occurrence,
                        exc_info=(occurrence == 1),
                    )
                continue

        # If all models for this provider failed, log and move to next provider
        should_warn, occurrence = _should_emit_fallback_warning(
            scope=f"generate:{prov_name}:provider",
            bucket="all_models_failed",
        )
        if should_warn:
            logger.warning(
                "All models for provider %s failed, trying next provider (occurrence=%d)",
                prov_name,
                occurrence,
            )

    # If all providers and models failed
    raise last_exception or CloudLLMError("All providers and models failed")


async def generate_stream(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
    cloud_provider: str | None = None,
    allow_provider_fallback: bool = True,
):
    """
    Generate a streaming completion with provider‑level and model‑level fallback
    (before first token) and cost guardrail.
    """
    if not is_cloud_admin_enabled():
        raise CloudLLMError("Cloud LLM is disabled by configuration (USE_CLOUD_LLM=0)")
    await asyncio.to_thread(refresh_provider_chain_from_env, force=False)

    primary_model = (get_env_str("CLOUD_LLM_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL) if model is None else model
    temperature = get_env_float("CLOUD_LLM_TEMPERATURE", DEFAULT_TEMPERATURE) if temperature is None else temperature
    timeout = get_env_float("CLOUD_LLM_TIMEOUT", DEFAULT_TIMEOUT) if timeout is None else timeout
    max_tokens = get_env_int("CLOUD_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS) if max_tokens is None else max_tokens
    fallback_models = parse_csv_env("CLOUD_LLM_FALLBACK_MODELS", default=FALLBACK_MODELS)
    stream_chunk_timeout = get_env_float("CLOUD_LLM_STREAM_CHUNK_TIMEOUT", STREAM_CHUNK_TIMEOUT)
    max_cost_per_request = get_env_float("CLOUD_LLM_MAX_COST", MAX_COST_PER_REQUEST)
    cost_per_1k_tokens = get_env_float("CLOUD_LLM_COST_PER_1K_TOKENS", COST_PER_1K_TOKENS)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_exception = None
    start_time = time.monotonic()
    request_id = get_request_id()

    provider_entries = _resolve_provider_entries(cloud_provider, allow_provider_fallback)
    if not provider_entries:
        raise CloudLLMError("No cloud providers available")

    for prov_name, adapter, retry_exc in provider_entries:
        # Check circuit breaker for this provider
        try:
            await _check_circuit_breaker(prov_name)
        except CloudLLMError as e:
            reason_bucket = _fallback_reason_bucket(e)
            should_warn, occurrence = _should_emit_fallback_warning(
                scope=f"stream:{prov_name}:circuit",
                bucket=reason_bucket,
            )
            if should_warn:
                logger.warning(
                    "Skipping provider %s due to open circuit (occurrence=%d)",
                    prov_name,
                    occurrence,
                )
            last_exception = e
            continue

        models_to_try = _bounded_models_to_try(primary_model, fallback_models)
        tokens_emitted = False

        for attempt_model in models_to_try:
            try:
                # Get circuit breaker instance for this attempt (provider-specific)
                breaker = await get_circuit_breaker(f"cloud_llm_{prov_name}")

                # Configure retry for opening the stream
                cfg = RetryConfig(retries=3, base_delay=1.0)

                # Generator factory: opens the stream and yields chunks with timeout
                async def _stream_generator_factory():
                    # Open stream with retry (before first token)
                    stream = await retry_async(
                        lambda: adapter.open_stream(
                            model=attempt_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=timeout,
                        ),
                        config=cfg,
                        request_id=request_id,
                        retry_exceptions=retry_exc,
                    )
                    # Now consume stream with per‑chunk timeout
                    async for chunk in _aiter_with_timeout(stream, stream_chunk_timeout):
                        yield chunk

                # Run the whole generator under breaker protection
                estimated_tokens = 0
                first_token = True

                async for chunk in breaker.run_generator_protected(
                    _stream_generator_factory,
                    non_failure_exceptions=(CloudProviderUnavailableError,),
                ):
                    # ----- defensive delta extraction -----
                    choice0 = getattr(chunk, "choices", [None])[0]
                    if not choice0:
                        continue

                    delta = None
                    delta_obj = getattr(choice0, "delta", None)
                    if delta_obj:
                        delta = getattr(delta_obj, "content", None)

                    if delta is None:
                        delta = (
                            getattr(choice0, "text", None)
                            or getattr(getattr(choice0, "message", None), "content", None)
                        )

                    if not delta:
                        continue

                    # Cost guardrail
                    estimated_tokens += max(1, len(delta) // 4)
                    if max_cost_per_request > 0 and cost_per_1k_tokens > 0:
                        estimated_cost = (estimated_tokens / 1000) * cost_per_1k_tokens
                        if estimated_cost > max_cost_per_request:
                            raise CloudLLMError("Streaming cost exceeded")

                    if first_token:
                        tokens_emitted = True
                        ttft = time.monotonic() - start_time
                        logger.debug("Cloud LLM first token", extra={
                            "request_id": request_id,
                            "provider": prov_name,
                            "model": attempt_model,
                            "ttft_sec": round(ttft, 3),
                        })
                        first_token = False

                    yield delta

                # Generator finished normally – success (breaker already recorded success)
                total_latency = time.monotonic() - start_time
                logger.debug("Cloud LLM stream completed", extra={
                    "request_id": request_id,
                    "provider": prov_name,
                    "model": attempt_model,
                    "total_latency_sec": round(total_latency, 3),
                    "estimated_tokens": estimated_tokens,
                })
                return

            except CircuitBreakerOpenError:
                should_warn, occurrence = _should_emit_fallback_warning(
                    scope=f"stream:{prov_name}:breaker_open",
                    bucket="circuit_open",
                )
                if should_warn:
                    logger.warning("cloud_llm circuit open", extra={
                        "request_id": request_id,
                        "provider": prov_name,
                        "model": attempt_model,
                        "occurrence": occurrence,
                    })
                raise CloudLLMError("Circuit breaker open")

            except Exception as e:
                # If tokens were already emitted, we must escalate (no fallback mid‑stream)
                if tokens_emitted:
                    logger.error("Stream failed after first token", extra={
                        "request_id": request_id,
                        "provider": prov_name,
                        "model": attempt_model,
                        "error": str(e),
                    })
                    raise

                # Otherwise (failure before first token) – try next model in this provider
                last_exception = e
                if _is_no_available_keys_error(e):
                    logger.warning(
                        "Stream skipped provider %s due to unavailable keys",
                        prov_name,
                        extra={"provider": prov_name},
                    )
                    break
                reason_bucket = _fallback_reason_bucket(e)
                should_warn, occurrence = _should_emit_fallback_warning(
                    scope=f"stream:{prov_name}:model",
                    bucket=reason_bucket,
                )
                if should_warn:
                    logger.warning(
                        "Stream open failed for provider %s model %s before first token (reason_bucket=%s, occurrence=%d)",
                        prov_name,
                        attempt_model,
                        reason_bucket,
                        occurrence,
                        exc_info=(occurrence == 1),
                    )
                continue

        # All models for this provider failed before first token – move to next provider
        should_warn, occurrence = _should_emit_fallback_warning(
            scope=f"stream:{prov_name}:provider",
            bucket="all_models_failed",
        )
        if should_warn:
            logger.warning(
                "All models for provider %s failed before first token, trying next provider (occurrence=%d)",
                prov_name,
                occurrence,
            )

    # All providers and models failed without emitting a single token
    raise last_exception or CloudLLMError("All streaming providers and models failed")


def _classify_provider_health_failure(exc: Exception) -> str:
    text = str(exc or "").lower()
    if "circuit breaker open" in text or "circuit_open" in text:
        return "circuit_open"
    if any(token in text for token in ("unauthorized", "invalid api key", "invalid key", "forbidden", "401", "403", "auth")):
        return "auth"
    if any(token in text for token in ("insufficient_quota", "quota", "billing", "payment required")):
        return "quota"
    if any(token in text for token in ("rate limit", "ratelimit", "too many requests", "429")):
        return "rate_limit"
    if any(token in text for token in ("timeout", "timed out", "network", "connect", "temporar", "unavailable", "503", "502", "504")):
        return "transient"
    return "unknown"


def _cooldown_seconds_for_failure_class(reason_class: str) -> int:
    cls = str(reason_class or "unknown").strip().lower()
    if cls == "auth":
        return max(30, int(PROVIDER_AUTH_FAIL_COOLDOWN))
    if cls in {"quota", "rate_limit"}:
        return max(30, int(PROVIDER_FAIL_COOLDOWN))
    if cls == "transient":
        return max(5, int(PROVIDER_TRANSIENT_FAIL_COOLDOWN))
    if cls == "circuit_open":
        return max(5, int(PROVIDER_CIRCUIT_OPEN_FAIL_COOLDOWN))
    return max(5, int(min(PROVIDER_FAIL_COOLDOWN, 60)))


def _prune_provider_fail_cooldowns(now_ts: float, *, configured_providers: Optional[List[str]] = None) -> None:
    configured = set(configured_providers or [])
    stale = []
    for name, until in list(_PROVIDER_FAIL_COOLDOWNS.items()):
        until_ts = float(until or 0)
        if until_ts <= now_ts:
            stale.append(name)
            continue
        if configured and name not in configured:
            stale.append(name)
    for name in stale:
        _PROVIDER_FAIL_COOLDOWNS.pop(name, None)


async def health_check() -> str:
    """
    Cloud health check.
    - If USE_CLOUD_LLM=0 → cloud is administratively disabled → return "disabled"
    - If USE_CLOUD_LLM=1 with no usable provider keys → return "unavailable"
    - If USE_CLOUD_LLM=1 with usable providers → require at least one provider ping healthy
    """
    if not is_cloud_admin_enabled():
        logger.info("Cloud LLM disabled via USE_CLOUD_LLM=0; skipping cloud health probe.")
        return "disabled"
    await asyncio.to_thread(refresh_provider_chain_from_env, force=False)

    usable_providers = await get_usable_providers()
    if not usable_providers:
        logger.info(
            "Cloud LLM enabled but no usable cloud providers available.",
            extra=get_provider_runtime_status(),
        )
        return "unavailable"

    now = time.time()
    ping_model = get_env_str("CLOUD_LLM_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    configured_providers = [name for name, _, _ in provider_chain]
    _prune_provider_fail_cooldowns(now, configured_providers=configured_providers)

    for prov_name, adapter, _ in provider_chain:
        if prov_name not in usable_providers:
            continue
        if adapter is None:
            continue

        cooldown_until = float(_PROVIDER_FAIL_COOLDOWNS.get(prov_name, 0) or 0)
        if now < cooldown_until:
            app_metrics.record_provider_health_cooldown_skip(prov_name)
            logger.debug(
                "Health check skipping provider due to cooldown",
                extra={"provider": prov_name, "cooldown_until": cooldown_until},
            )
            continue

        try:
            await adapter.ping(ping_model)
            _PROVIDER_FAIL_COOLDOWNS.pop(prov_name, None)
            logger.info("Health check ok", extra={"provider": prov_name})
            return "ok"
        except Exception as e:
            reason_class = _classify_provider_health_failure(e)
            app_metrics.record_provider_health_failure(prov_name, reason_class)
            cooldown_seconds = _cooldown_seconds_for_failure_class(reason_class)
            proposed_until = now + float(cooldown_seconds)
            current_until = float(_PROVIDER_FAIL_COOLDOWNS.get(prov_name, 0) or 0)
            _PROVIDER_FAIL_COOLDOWNS[prov_name] = max(current_until, proposed_until)
            logger.warning(
                "Health check provider failure recorded",
                extra={
                    "provider": prov_name,
                    "reason_class": reason_class,
                    "cooldown_seconds": cooldown_seconds,
                    "cooldown_until": _PROVIDER_FAIL_COOLDOWNS[prov_name],
                    "error": str(e),
                },
            )
            continue

    return "fail"


async def close_client():
    """Close all underlying provider clients gracefully."""
    await close_all_clients()
    # Close adapters (Gemini instance if any)
    for adapter in _adapters.values():
        await adapter.close()
    logger.info("Cloud client shutdown complete")


class CloudLLMClient:
    """
    Wrapper around cloud LLM module-level functions
    to match router expectations.
    """

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
        request_id: Optional[str] = None,
        cloud_provider: Optional[str] = None,
        allow_provider_fallback: bool = True,
    ):
        if stream:
            return generate_stream(
                prompt=prompt,
                system=system or "",
                model=model,
                cloud_provider=cloud_provider,
                allow_provider_fallback=allow_provider_fallback,
            )
        return await generate(
            prompt=prompt,
            system=system or "",
            model=model,
            cloud_provider=cloud_provider,
            allow_provider_fallback=allow_provider_fallback,
        )

    async def health_check(self) -> bool:
        return await health_check() == "ok"

    async def get_provider_usability(self) -> Dict[str, bool]:
        return await get_provider_usability()

    async def has_usable_provider(self) -> bool:
        return await cloud_backend_is_usable(respect_admin_flag=True)
