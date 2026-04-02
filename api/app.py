# api/app.py
# NOTE:
# We intentionally rely on FastAPI's default exception handling for
# unexpected errors. The planner layer already converts operational
# failures (timeouts, tool errors, LLM failures) into structured
# JSON responses. Only truly unexpected exceptions propagate as 500,
# which is desirable for visibility and debugging.
# NOTE:
# The /ask endpoint now supports both non‑streaming (JSON) and streaming (SSE)
# responses. Streaming is enabled by passing ?stream=true in the query string.
# Background jobs are triggered by ?async_job=true; they return a 202 with a job_id
# that can be polled via GET /jobs/{job_id} or streamed via GET /jobs/{job_id}/events.

import uuid
import json
import logging
import os
import asyncio
import time
import html
import hashlib
import fcntl                     # for process‑level locking
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException, Query, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Use module import instead of direct function import for better testability
import agents.planner_agent as planner_agent

# Import specific tool exceptions for granular error handling
from tools.airline_api import AirlineAPIError
from tools.weather_api import WeatherAPIError
from core.http_client import close_client
from core.request_context import set_request_id, get_request_id
from core.logging_config import setup_logging
from core.health import full_health_check
from core.async_llm_client import init_llm_client, close_llm_client
import core.metrics as app_metrics
from core.env_config import get_env_bool, get_env_float, get_env_int, get_env_str, is_env_set
from core.llm_mode import (
    llm_routing_context,
    normalize_cloud_provider,
    normalize_llm_mode,
    get_cloud_provider_chain_resolution,
    get_configured_cloud_providers,
    get_default_cloud_provider,
    get_llm_mode_default,
    get_llm_mode_resolution,
    get_mode_dependency_map,
    LLM_MODE_CLOUD_ONLY,
    LLM_MODE_CLOUD_FIRST,
    LLM_MODE_OLLAMA_ONLY,
    LLM_MODE_OLLAMA_FIRST,
    VALID_LLM_MODES,
)
from core import job_queue                     # background job worker
from core.api_key_manager import key_manager    # key rotation manager
from agents.cloud_llm import (
    on_key_event,
    get_available_providers,
    get_provider_usability,
    get_provider_runtime_status,
    get_usable_providers,
    is_cloud_admin_enabled,
    refresh_provider_chain_from_env,
)  # callback for key changes
from agents import ollama_client

logger = logging.getLogger(__name__)
LOG_REQUEST_BODY_DEBUG = get_env_bool("LOG_REQUEST_BODY_DEBUG", default=False)

ASK_MAX_INFLIGHT_DEFAULT = max(1, get_env_int("ASK_MAX_INFLIGHT", 16))
ASK_DUPLICATE_RETRY_AFTER_DEFAULT = max(1, get_env_int("ASK_DUPLICATE_RETRY_AFTER_SECONDS", 2))
ASK_OVERLOAD_RETRY_AFTER_DEFAULT = max(1, get_env_int("ASK_OVERLOAD_RETRY_AFTER_SECONDS", 1))


def _resolve_ask_max_inflight() -> int:
    return max(1, get_env_int("ASK_MAX_INFLIGHT", ASK_MAX_INFLIGHT_DEFAULT))


def _resolve_ask_duplicate_retry_after_seconds() -> int:
    return max(
        1,
        get_env_int("ASK_DUPLICATE_RETRY_AFTER_SECONDS", ASK_DUPLICATE_RETRY_AFTER_DEFAULT),
    )


def _resolve_ask_overload_retry_after_seconds() -> int:
    return max(
        1,
        get_env_int("ASK_OVERLOAD_RETRY_AFTER_SECONDS", ASK_OVERLOAD_RETRY_AFTER_DEFAULT),
    )


def _resolve_ask_inflight_stale_seconds() -> float:
    configured = get_env_float("ASK_INFLIGHT_STALE_SECONDS", 0.0)
    if configured > 0:
        return max(10.0, configured)
    return max(10.0, float(_resolve_request_timeout()) + 10.0)


def _build_ask_request_fingerprint(
    *,
    origin: Optional[str],
    destination: Optional[str],
    date: Optional[str],
    user_query: str,
    trip_type: Optional[str],
    llm_mode: Optional[str],
    cloud_provider: Optional[str],
    stream: bool,
) -> str:
    normalized = {
        "origin": str(origin or "").strip().upper(),
        "destination": str(destination or "").strip().upper(),
        "date": str(date or "").strip(),
        "user_query": str(user_query or "").strip(),
        "trip_type": str(trip_type or "").strip().lower(),
        "llm_mode": str(llm_mode or "").strip().lower(),
        "cloud_provider": str(cloud_provider or "").strip().lower(),
        "stream": bool(stream),
    }
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _ensure_ask_runtime_state(app: FastAPI) -> Dict[str, Any]:
    state = getattr(app.state, "ask_runtime_state", None)
    if isinstance(state, dict) and isinstance(state.get("inflight"), dict) and state.get("lock") is not None:
        return state
    state = {
        "lock": asyncio.Lock(),
        "inflight": {},
    }
    app.state.ask_runtime_state = state
    return state


def _prune_stale_ask_inflight_locked(
    *,
    inflight: Dict[str, Dict[str, Any]],
    now_monotonic: float,
    stale_after_seconds: float,
) -> int:
    stale_keys = []
    for key, value in list(inflight.items()):
        started_at = float((value or {}).get("started_at") or now_monotonic)
        if (now_monotonic - started_at) > stale_after_seconds:
            stale_keys.append(key)
    for key in stale_keys:
        inflight.pop(key, None)
    return len(stale_keys)


async def _release_ask_inflight_key(request_fingerprint: Optional[str]) -> None:
    if not request_fingerprint:
        return
    runtime_state = _ensure_ask_runtime_state(app)
    lock = runtime_state["lock"]
    inflight = runtime_state["inflight"]
    async with lock:
        inflight.pop(request_fingerprint, None)
        app_metrics.set_ask_inflight(len(inflight))


def _legacy_async_llm_client_enabled() -> bool:
    return get_env_bool("ENABLE_LEGACY_ASYNC_LLM_CLIENT", default=False)

# Deprecated configuration variables that are still tolerated for backward compatibility.
_DEPRECATED_ENV_GUIDANCE = {
    "OPENAI_API_KEY": {
        "replacement": "OPENAI_KEY_n",
        "note": "Legacy async client path only; modern cloud routing uses key-manager numbered keys.",
    },
    "ANTHROPIC_API_KEY": {
        "replacement": "provider-specific numbered key pool (not OPENAI_API_KEY style)",
        "note": "Legacy async client path only; modern runtime routes through cloud provider chain and key manager.",
    },
    "CLOUD_BASE_URL": {
        "replacement": "CLOUD_PROVIDER_CHAIN/CLOUD_PROVIDER with provider adapters",
        "note": "Only legacy async client init reads CLOUD_BASE_URL directly.",
    },
    "LLM_PRIORITY": {
        "replacement": "LLM_MODE",
        "note": "LLM_PRIORITY is kept only for legacy hybrid alias mapping.",
    },
    "LLM_PREWARM": {
        "replacement": "PLANNER_PREWARM",
        "note": "LLM_PREWARM is no longer read by the active backend startup path and is ignored.",
    },
    "PLANNER_STREAMING_ENABLED": {
        "replacement": "none (removed)",
        "note": "PLANNER_STREAMING_ENABLED has no active read path and is ignored.",
    },
}


def _is_env_set(var_name: str) -> bool:
    return is_env_set(var_name)


def _emit_deprecated_config_warnings() -> list[str]:
    warned: list[str] = []
    legacy_client_enabled = _legacy_async_llm_client_enabled()
    for var_name, guidance in _DEPRECATED_ENV_GUIDANCE.items():
        if not _is_env_set(var_name):
            continue
        # CLOUD_BASE_URL is only consumed by the legacy async client path.
        # When legacy path is disabled, warning on this variable is noisy and misleading.
        if var_name == "CLOUD_BASE_URL" and not legacy_client_enabled:
            continue
        warned.append(var_name)
        logger.warning(
            "Config deprecation: %s is deprecated. Use %s instead. %s Compatibility support remains active for now.",
            var_name,
            guidance["replacement"],
            guidance["note"],
        )
    return warned


def _startup_log_level_for_worker(refresh_owner: bool) -> str:
    workers = _declared_worker_count()
    if refresh_owner or workers in (None, 1):
        return "info"
    return "debug"


def _worker_runtime_role(refresh_owner: bool) -> str:
    if refresh_owner:
        return "refresh_owner"
    workers = _declared_worker_count()
    if workers is not None and workers > 1:
        return "follower"
    return "single_or_undeclared"


def _should_emit_startup_summary(refresh_owner: bool) -> bool:
    workers = _declared_worker_count()
    return bool(refresh_owner or workers in (None, 1))


def _is_cloud_startup_relevant_for_mode(llm_mode: str) -> bool:
    return str(llm_mode or "").strip().lower() != LLM_MODE_OLLAMA_ONLY


def _is_cloud_startup_relevant_now() -> bool:
    return _is_cloud_startup_relevant_for_mode(get_llm_mode_default())


async def _log_startup_config_summary(*, deprecated_env_detected: list[str], log_level: str = "info") -> None:
    """
    Log a one-time non-secret startup config summary for operational visibility.
    This intentionally avoids logging any API keys or secret values.
    """
    mode_resolution = get_llm_mode_resolution()
    llm_mode = str(mode_resolution["mode"])
    mode_source = str(mode_resolution["source"])
    legacy_priority_used = bool(mode_resolution.get("legacy_priority_used", False))
    mode_dependency = get_mode_dependency_map(llm_mode)

    provider_chain_resolution = get_cloud_provider_chain_resolution()
    configured_chain = list(provider_chain_resolution["providers"])
    default_provider = get_default_cloud_provider()
    provider_chain_source = str(provider_chain_resolution["source"])

    cloud_enabled = is_cloud_admin_enabled()
    cloud_runtime_relevant = _is_cloud_startup_relevant_for_mode(llm_mode)
    planner_prewarm_enabled = get_env_bool("PLANNER_PREWARM", default=False)
    ollama_base_url = get_env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = get_env_str("OLLAMA_MODEL", "openhermes")
    planner_timeout = max(5.0, get_env_float("PLANNER_LLM_TIMEOUT", 45.0))
    router_timeout = max(1.0, get_env_float("ROUTER_TIMEOUT", 90.0))
    local_timeout = max(1.0, get_env_float("LOCAL_LLM_TIMEOUT", max(get_env_float("OLLAMA_TIMEOUT", 30.0), planner_timeout)))
    cloud_timeout = max(1.0, get_env_float("CLOUD_LLM_TIMEOUT", 60.0))
    request_timeout = _resolve_request_timeout()

    usable_providers: list[str] = []
    provider_runtime_status: dict = {}
    if cloud_enabled and cloud_runtime_relevant:
        try:
            usable_providers = await get_usable_providers()
            provider_runtime_status = get_provider_runtime_status()
        except Exception:
            logger.warning("startup_config_summary_cloud_usability_unavailable")

    deprecated_text = ",".join(sorted(deprecated_env_detected)) if deprecated_env_detected else "none"
    usable_text = ",".join(usable_providers) if usable_providers else ("none" if cloud_runtime_relevant else "n/a_non_routing")
    chain_text = ",".join(configured_chain) if configured_chain else "none"
    initialized_text = ",".join(provider_runtime_status.get("available_providers") or []) or ("none" if cloud_runtime_relevant else "n/a_non_routing")
    ignored_vars = ",".join(mode_dependency.get("ignored_for_routing", [])) or "none"

    log_fn = logger.info if str(log_level).lower() == "info" else logger.debug
    log_fn(
        (
            "Startup config summary | llm_mode=%s | llm_mode_source=%s | llm_priority_compat_used=%s "
            "| cloud_enabled=%s | cloud_default_provider=%s | cloud_provider_chain=%s | cloud_provider_chain_source=%s "
            "| cloud_runtime_relevant=%s | cloud_initialized_providers=%s | cloud_usable_providers=%s | ollama_base_url=%s | ollama_model=%s "
            "| planner_timeout_sec=%.2f | local_llm_timeout_sec=%.2f | cloud_llm_timeout_sec=%.2f | router_timeout_sec=%.2f | request_timeout_sec=%s "
            "| mode_ignored_for_routing=%s | key_manager_lock_backend=%s | planner_prewarm=%s | deprecated_env_detected=%s"
        ),
        llm_mode,
        mode_source,
        legacy_priority_used,
        cloud_enabled,
        default_provider,
        chain_text,
        provider_chain_source,
        cloud_runtime_relevant,
        initialized_text,
        usable_text,
        ollama_base_url,
        ollama_model,
        planner_timeout,
        local_timeout,
        cloud_timeout,
        router_timeout,
        request_timeout,
        ignored_vars,
        KEY_MANAGER_LOCK_BACKEND,
        planner_prewarm_enabled,
        deprecated_text,
    )


# --- Pluggable lock helpers (file-based default, redis optional) ---
try:
    import redis.asyncio as redis_async  # optional dependency for redis lock backend
except ImportError:
    redis_async = None

KEY_MANAGER_LOCK_BACKEND = (get_env_str("KEY_MANAGER_LOCK_BACKEND", "file") or "file").lower()
KEY_MANAGER_REDIS_URL = get_env_str("KEY_MANAGER_REDIS_URL", "redis://localhost:6379/0")
KEY_MANAGER_LOCK_NAME = get_env_str("KEY_MANAGER_LOCK_NAME", "llm:key_refresh_lock")
KEY_MANAGER_LOCK_TTL = get_env_int("KEY_MANAGER_LOCK_TTL_SECONDS", 60)  # lock TTL for redis
KEY_MANAGER_LOCK_PATH = get_env_str("KEY_MANAGER_LOCK_PATH", "/tmp/llm_key_refresh.lock")
KEY_MANAGER_REDIS_RENEW_INTERVAL = get_env_float("KEY_MANAGER_REDIS_RENEW_INTERVAL_SECONDS", 0.0)

ASYNC_JOB_REQUIRE_SINGLE_WORKER = get_env_bool("ASYNC_JOB_REQUIRE_SINGLE_WORKER", default=True)
ALLOW_UNSAFE_ASYNC_JOBS = get_env_bool("ALLOW_UNSAFE_ASYNC_JOBS", default=False)
ASYNC_JOB_FAIL_FAST_ON_UNSUPPORTED_TOPOLOGY = get_env_bool(
    "ASYNC_JOB_FAIL_FAST_ON_UNSUPPORTED_TOPOLOGY",
    default=False,
)


def _acquire_process_lock(path: str) -> Optional[int]:
    """
    Try to acquire an exclusive lock on the given file.
    Returns a file descriptor if successful, None if another process holds the lock.
    """
    fd = None
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except (IOError, OSError, BlockingIOError):
        # Lock already held by another process
        if fd is not None:
            os.close(fd)
        return None
    except Exception:
        # Unexpected error; fall back to no lock
        if fd is not None:
            os.close(fd)
        return None


async def _acquire_redis_lock(redis_url: str, name: str, ttl: int):
    """Try to acquire a Redis-based distributed lock. Returns a tuple (client, lock) on success, or (None, None)."""
    if redis_async is None:
        return None, None
    client = None
    try:
        client = redis_async.from_url(redis_url)
        lock = client.lock(name, timeout=ttl)
        acquired = await lock.acquire(blocking=False)
        if acquired:
            return client, lock
        else:
            await client.close()
            return None, None
    except Exception:
        # Any redis error -> don't acquire
        if client:
            try:
                await client.close()
            except Exception:
                pass
        return None, None


def _safe_int_env(name: str) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _declared_worker_count() -> Optional[int]:
    for env_name in ("UVICORN_WORKERS", "WEB_CONCURRENCY", "GUNICORN_WORKERS", "WORKERS"):
        value = _safe_int_env(env_name)
        if value is not None:
            return value
    return None


def _compute_async_job_support() -> dict:
    guard_enabled = get_env_bool(
        "ASYNC_JOB_REQUIRE_SINGLE_WORKER",
        default=ASYNC_JOB_REQUIRE_SINGLE_WORKER,
    )
    allow_unsafe = get_env_bool(
        "ALLOW_UNSAFE_ASYNC_JOBS",
        default=ALLOW_UNSAFE_ASYNC_JOBS,
    )
    fail_fast = get_env_bool(
        "ASYNC_JOB_FAIL_FAST_ON_UNSUPPORTED_TOPOLOGY",
        default=ASYNC_JOB_FAIL_FAST_ON_UNSUPPORTED_TOPOLOGY,
    )
    workers = _declared_worker_count()
    support = {
        "declared_workers": workers,
        "guard_active": bool(guard_enabled),
        "allow_unsafe_override": bool(allow_unsafe),
        "fail_fast_on_unsupported_topology": bool(fail_fast),
        "contract": "single_worker_required_process_local_queue",
    }
    if allow_unsafe:
        return {
            **support,
            "enabled": True,
            "reason": "unsafe_override_enabled",
        }
    if not guard_enabled:
        support["guard_active"] = False
        return {
            **support,
            "enabled": True,
            "reason": "single_worker_guard_disabled",
        }
    if workers is not None and workers > 1:
        return {
            **support,
            "enabled": False,
            "reason": "unsupported_multi_worker_topology",
        }
    return {
        **support,
        "enabled": True,
        "reason": "single_worker_or_undeclared_topology",
    }


def _get_async_job_support_state(app: FastAPI) -> dict:
    support = getattr(app.state, "async_job_support", None)
    if isinstance(support, dict):
        return support
    return _compute_async_job_support()


def _should_run_prewarm(prewarm_enabled: bool, refresh_owner: bool) -> bool:
    if not prewarm_enabled:
        return False
    if get_env_bool("PLANNER_PREWARM_ALL_WORKERS", default=False):
        return True
    if refresh_owner:
        return True
    workers = _declared_worker_count()
    # In single/undeclared topology, keep prewarm enabled even if lock ownership
    # wasn't acquired (for example, transient stale lockfile during local runs).
    return workers in (None, 1)


async def _stop_key_refresh_for_lost_ownership(app: FastAPI, reason: str) -> None:
    if not getattr(app.state, "key_manager_refresh_owner", False):
        return
    app.state.key_manager_refresh_owner = False
    logger.error("Lost key manager refresh lock ownership (%s); stopping refresh loop.", reason)
    try:
        key_manager.stop_refresh_loop()
    except Exception:
        logger.exception("key_manager_stop_refresh_loop_after_ownership_loss_failed")
    task = getattr(app.state, "key_manager_task", None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("key_manager_task_cancel_after_ownership_loss_failed")
    app.state.key_manager_task = None


def _redis_lease_interval_seconds(ttl_seconds: int) -> float:
    if KEY_MANAGER_REDIS_RENEW_INTERVAL > 0:
        return max(0.2, KEY_MANAGER_REDIS_RENEW_INTERVAL)
    ttl = max(2, int(ttl_seconds))
    return max(0.5, min(ttl / 3.0, ttl - 1.0))


async def _run_redis_lock_lease_keeper(app: FastAPI, lock, ttl_seconds: int) -> None:
    interval = _redis_lease_interval_seconds(ttl_seconds)
    while getattr(app.state, "key_manager_refresh_owner", False):
        await asyncio.sleep(interval)
        if not getattr(app.state, "key_manager_refresh_owner", False):
            break
        try:
            extended = await lock.extend(ttl_seconds, replace_ttl=True)
        except Exception as exc:
            await _stop_key_refresh_for_lost_ownership(
                app,
                f"lease_extend_exception:{type(exc).__name__}",
            )
            break
        if not extended:
            await _stop_key_refresh_for_lost_ownership(app, "lease_extend_rejected_not_owner")
            break


async def _acquire_pluggable_lock():
    """Return a tuple (backend, handle) where backend is 'file' or 'redis' and handle is fd or (client, lock)."""
    if KEY_MANAGER_LOCK_BACKEND == "redis":
        client, lock = await _acquire_redis_lock(KEY_MANAGER_REDIS_URL, KEY_MANAGER_LOCK_NAME, KEY_MANAGER_LOCK_TTL)
        if lock:
            return "redis", (client, lock)
        return "redis", None
    # default: file lock
    fd = _acquire_process_lock(KEY_MANAGER_LOCK_PATH)
    if fd is not None:
        return "file", fd
    return "file", None


async def prewarm_llm():
    """
    Ollama prewarm with retries and exponential backoff.
    Will not crash startup if Ollama is slow or unavailable.
    """
    from agents import ollama_client
    from agents.ollama_client import OllamaError

    base_timeout = max(
        float(get_env_int("OLLAMA_PREWARM_TIMEOUT", 60)),
        get_env_float("OLLAMA_TIMEOUT", 30.0),
    )
    timeout_step = max(0, get_env_int("OLLAMA_PREWARM_TIMEOUT_STEP", 20))
    max_retries = get_env_int("OLLAMA_PREWARM_RETRIES", 3)
    backoff = 1
    model_name = get_env_str("OLLAMA_MODEL", "openhermes")
    last_error: Optional[str] = None
    last_error_bucket: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        attempt_timeout = base_timeout + (attempt - 1) * timeout_step
        attempt_started = time.monotonic()
        try:
            await ollama_client.prewarm(
                model=model_name,
                timeout=attempt_timeout,
                request_id=f"prewarm-{attempt}",
            )
            logger.info(
                "Ollama prewarm OK",
                extra={
                    "model": model_name,
                    "attempt": attempt,
                    "timeout_sec": attempt_timeout,
                    "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                    "recovered_from_prior_failure": attempt > 1,
                    "prior_error_bucket": last_error_bucket,
                },
            )
            return {
                "status": "ok",
                "attempts": attempt,
                "model": model_name,
                "timeout_sec": attempt_timeout,
            }
        except OllamaError as e:
            last_error = str(e)
            msg = str(e).lower()
            if "timed out" in msg or "timeout" in msg:
                last_error_bucket = "timeout"
            elif "circuit breaker" in msg:
                last_error_bucket = "circuit_open"
            else:
                last_error_bucket = "backend_error"
            logger.warning(
                "Ollama prewarm attempt failed",
                extra={
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "timeout_sec": attempt_timeout,
                    "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                    "error": str(e),
                    "error_bucket": last_error_bucket,
                },
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                logger.warning(
                    "Ollama prewarm failed after %d attempts — continuing without prewarm",
                    max_retries
                )
                return {
                    "status": "failed",
                    "attempts": attempt,
                    "model": model_name,
                    "timeout_sec": attempt_timeout,
                    "error": last_error,
                }
        except Exception as e:
            # Catch any other unexpected errors (e.g., import issues) and treat as failure
            last_error = str(e)
            last_error_bucket = "unexpected"
            logger.warning(
                "Unexpected error during prewarm attempt",
                extra={
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "timeout_sec": attempt_timeout,
                    "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                    "error": str(e),
                    "error_bucket": last_error_bucket,
                },
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                logger.warning("Prewarm aborted after %d attempts", max_retries)
                return {
                    "status": "failed",
                    "attempts": attempt,
                    "model": model_name,
                    "timeout_sec": attempt_timeout,
                    "error": last_error,
                }


def require_admin_token(x_admin_token: str = Header(...)):
    """Dependency to protect admin endpoints with a token from environment."""
    expected = get_env_str("ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_complete = False
    app.state.legacy_llm_client_initialized = False
    prewarm_enabled = get_env_bool("PLANNER_PREWARM", default=False)
    app.state.llm_prewarm = {
        "enabled": prewarm_enabled,
        "best_effort": True,
        "status": "scheduled" if prewarm_enabled else "disabled",
        "model": get_env_str("OLLAMA_MODEL", "openhermes"),
        "attempts": 0,
        "last_error": None,
        "last_updated": datetime.utcnow().isoformat() + "Z",
    }
    app.state.llm_prewarm_task = None
    app.state.async_job_support = _compute_async_job_support()
    app.state.key_manager_lease_task = None
    app.state.key_manager_refresh_owner = False
    app.state.ask_runtime_state = {
        "lock": asyncio.Lock(),
        "inflight": {},
    }

    # Startup: configure structured JSON logging
    setup_logging()
    deprecated_env_detected = _emit_deprecated_config_warnings()

    # Legacy async LLM client path is compatibility-only and opt-in.
    # Modern runtime uses llm_router + provider adapters + key-manager pools.
    if _legacy_async_llm_client_enabled():
        try:
            await init_llm_client()
            app.state.legacy_llm_client_initialized = True
        except Exception as e:
            logger.info("legacy_llm_client_init_skipped: %s", str(e))
    else:
        logger.debug("legacy_llm_client_disabled_by_config")

    # Load API keys from environment into the key manager
    try:
        await key_manager.load_env_keys()
    except Exception:
        logger.exception("key_manager_load_failed")

    if is_cloud_admin_enabled():
        try:
            if _is_cloud_startup_relevant_now():
                # Avoid forcing duplicate provider-refresh startup logs in follower workers.
                # Module import has already performed an initial refresh in-process.
                refresh_provider_chain_from_env(force=False)
            else:
                logger.debug("cloud_provider_refresh_skipped_non_routing_ollama_only_mode")
        except Exception:
            logger.exception("cloud_provider_refresh_failed")
    else:
        logger.debug("cloud_provider_refresh_skipped_cloud_disabled")

    # ---- Register key event listener early (idempotent) ----
    try:
        already_registered = False
        # best-effort detection to avoid duplicate registration in same process
        listeners = getattr(key_manager, "_key_event_listeners", None)
        if listeners is not None:
            try:
                if on_key_event in listeners:
                    already_registered = True
            except Exception:
                # fall back to identity scan
                for item in list(listeners):
                    if getattr(item, "__name__", None) == getattr(on_key_event, "__name__", None):
                        already_registered = True
                        break

        if not already_registered:
            key_manager.register_key_event_listener(on_key_event)
            app.state.cloud_llm_listener_registered = True
            logger.debug("Registered cloud LLM key event listener")
        else:
            logger.debug("Cloud LLM key event listener already registered in this process")
    except Exception:
        logger.exception("Failed to register cloud LLM key event listener")

    # ---- Pluggable lock to ensure only one process/replica runs the refresh loop ----
    lock_backend, lock_handle = await _acquire_pluggable_lock()
    should_run_refresh = lock_handle is not None

    # Fallback: env var override (useful for containers where you set one replica manually)
    if not should_run_refresh and get_env_bool("RUN_KEY_REFRESH", default=False):
        if lock_backend == "redis":
            logger.error(
                "RUN_KEY_REFRESH=true ignored for redis backend when lock is not acquired; "
                "refusing unsafe refresh-loop ownership."
            )
        else:
            logger.warning(
                "RUN_KEY_REFRESH=true but lock not acquired; starting refresh loop anyway. "
                "Ensure only one replica has this variable set."
            )
            should_run_refresh = True

    if should_run_refresh:
        logger.info("Starting key manager background refresh loop (lock_backend=%s).", lock_backend)
        # Save lock handle for shutdown cleanup
        app.state.key_manager_lock_backend = lock_backend
        app.state.key_manager_lock_handle = lock_handle

        # Start the key manager's background refresh loop (interval configurable)
        refresh_interval = get_env_int("KEY_ENV_MONITOR_TICK", 60)
        # start_refresh_loop is synchronous; it creates an internal task.
        key_manager.start_refresh_loop(
            interval_seconds=refresh_interval,
            skip_lock_check=True      # we already acquired the lock ourselves
        )
        # Store the internal task so we can cancel it on shutdown
        app.state.key_manager_task = key_manager._refresh_task
        app.state.key_manager_refresh_owner = True
        if lock_backend == "redis" and lock_handle is not None:
            client, lock = lock_handle
            app.state.key_manager_lease_task = asyncio.create_task(
                _run_redis_lock_lease_keeper(app, lock, KEY_MANAGER_LOCK_TTL)
            )
    else:
        logger.debug("Another process/replica holds the key manager lock; not starting refresh loop.")
        app.state.key_manager_lock_backend = None
        app.state.key_manager_lock_handle = None
        app.state.key_manager_task = None
        app.state.key_manager_refresh_owner = False

    if _should_emit_startup_summary(bool(app.state.key_manager_refresh_owner)):
        try:
            await _log_startup_config_summary(
                deprecated_env_detected=deprecated_env_detected,
                log_level="info",
            )
        except Exception:
            logger.exception("startup_config_summary_failed")
    else:
        logger.debug(
            "startup_config_summary_suppressed_for_follower_worker",
            extra={"pid": os.getpid(), "refresh_owner": bool(app.state.key_manager_refresh_owner)},
        )

    runtime_role = _worker_runtime_role(bool(app.state.key_manager_refresh_owner))
    async_job_support = _get_async_job_support_state(app)
    logger.info(
        "Worker runtime role resolved",
        extra={
            "pid": os.getpid(),
            "refresh_owner": bool(app.state.key_manager_refresh_owner),
            "worker_role": runtime_role,
            "lock_backend": lock_backend,
            "async_jobs_enabled": bool(async_job_support.get("enabled", True)),
            "async_jobs_reason": async_job_support.get("reason"),
        },
    )

    # Start the background job worker loop (always needed)
    app.state.job_worker = asyncio.create_task(job_queue.worker_loop())

    if not async_job_support.get("enabled", True):
        if async_job_support.get("fail_fast_on_unsupported_topology"):
            if bool(app.state.key_manager_refresh_owner):
                logger.error(
                    "Async job startup fail-fast requested but suppressed to avoid worker respawn loops; request-time guard remains active",
                    extra=async_job_support,
                )
            else:
                logger.debug(
                    "Async job startup fail-fast requested but suppressed (follower worker); request-time guard remains active",
                    extra=async_job_support,
                )
        if bool(app.state.key_manager_refresh_owner):
            logger.warning(
                "Async jobs disabled due to unsupported topology",
                extra=async_job_support,
            )
        else:
            logger.debug(
                "Async jobs disabled due to unsupported topology (follower worker)",
                extra=async_job_support,
            )
    elif async_job_support.get("reason") == "unsafe_override_enabled":
        logger.warning(
            "Async jobs enabled via unsafe override; correctness is not guaranteed in multi-worker topology",
            extra=async_job_support,
        )
    else:
        logger.debug("Async job topology check", extra=async_job_support)

    # Optional prewarm (non‑blocking, best-effort)
    if _should_run_prewarm(prewarm_enabled, should_run_refresh):
        async def background_prewarm():
            app.state.llm_prewarm["status"] = "running"
            app.state.llm_prewarm["last_updated"] = datetime.utcnow().isoformat() + "Z"
            try:
                result = await prewarm_llm()
                app.state.llm_prewarm["attempts"] = int(result.get("attempts", 0))
                app.state.llm_prewarm["status"] = str(result.get("status", "failed"))
                app.state.llm_prewarm["last_error"] = result.get("error")
                app.state.llm_prewarm["last_updated"] = datetime.utcnow().isoformat() + "Z"
            except Exception:
                logger.exception("Background prewarm failed")
                app.state.llm_prewarm["status"] = "failed"
                app.state.llm_prewarm["last_error"] = "background_prewarm_exception"
                app.state.llm_prewarm["last_updated"] = datetime.utcnow().isoformat() + "Z"
        app.state.llm_prewarm_task = asyncio.create_task(background_prewarm())
    elif prewarm_enabled:
        app.state.llm_prewarm["status"] = "skipped_non_owner_worker"
        app.state.llm_prewarm["last_updated"] = datetime.utcnow().isoformat() + "Z"
        logger.debug(
            "Skipping prewarm on non-owner worker",
            extra={"pid": os.getpid(), "refresh_owner": bool(should_run_refresh)},
        )

    app.state.startup_complete = True
    yield

    app.state.startup_complete = False

    # Shutdown: gracefully stop the job worker
    try:
        await job_queue.stop_worker()
        await app.state.job_worker
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

    prewarm_task = getattr(app.state, "llm_prewarm_task", None)
    if prewarm_task and not prewarm_task.done():
        prewarm_task.cancel()
        try:
            await prewarm_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("llm_prewarm_task_cancel_failed")

    # Stop the key manager's background refresh loop only if we started it
    if getattr(app.state, "key_manager_task", None):
        try:
            # stop_refresh_loop is synchronous; it cancels the internal task.
            key_manager.stop_refresh_loop()
        except Exception:
            logger.exception("key_manager_stop_refresh_loop_failed")

        # Cancel background task if still running (though stop_refresh_loop should have done it)
        task = app.state.key_manager_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("key_manager_task_cancel_failed")

    lease_task = getattr(app.state, "key_manager_lease_task", None)
    if lease_task and not lease_task.done():
        lease_task.cancel()
        try:
            await lease_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("key_manager_lease_task_cancel_failed")

    # Release the lock we acquired (backend-specific)
    backend = getattr(app.state, "key_manager_lock_backend", None)
    handle = getattr(app.state, "key_manager_lock_handle", None)
    if backend == "file" and handle is not None:
        try:
            os.close(handle)
            logger.info("Released file lock for key manager refresh.")
        except Exception:
            logger.exception("failed_to_release_file_lock")
    elif backend == "redis" and handle is not None:
        client, lock = handle
        try:
            # release the redis lock and close connection
            await lock.release()
        except Exception:
            logger.exception("failed_to_release_redis_lock")
        try:
            await client.close()
        except Exception:
            pass

    # Clean up legacy async client only when it was enabled.
    if getattr(app.state, "legacy_llm_client_initialized", False):
        await close_llm_client()
    await close_client()

    # Ensure cloud_llm provider adapters are closed (safe even if none initialised)
    try:
        import agents.cloud_llm as cloud_llm
        await cloud_llm.close_client()
    except Exception:
        logger.exception("cloud_llm_close_failed_during_lifespan_shutdown")


app = FastAPI(
    title="LLM Travel Agent",
    lifespan=lifespan
)

# Add CORS middleware – now configurable via environment variable
# Read production origins from environment, fallback to localhost for dev
env_origins = get_env_str(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
)
allowed_origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to log raw request bodies for debugging 422 errors
@app.middleware("http")
async def log_request_body(request: Request, call_next):
    if not LOG_REQUEST_BODY_DEBUG:
        return await call_next(request)
    try:
        body_bytes = await request.body()
        body_text = body_bytes.decode(errors="replace")
        body_keys = []
        # Best-effort redaction for known secret-like fields when debug logging is enabled.
        try:
            body_json = json.loads(body_text)
            if isinstance(body_json, dict):
                body_keys = sorted(str(k) for k in body_json.keys())[:50]
                redacted = {}
                for key, value in body_json.items():
                    key_l = str(key).lower()
                    if any(token in key_l for token in ("token", "secret", "key", "password", "authorization")):
                        redacted[key] = "***REDACTED***"
                    else:
                        redacted[key] = value
                body_text = json.dumps(redacted, ensure_ascii=False)
        except Exception:
            pass
        logger.debug(
            "request_body_observed",
            extra={
                "path": request.url.path,
                "method": request.method,
                "body_size_bytes": len(body_bytes),
                "body_keys": body_keys,
            },
        )
        # Reattach the body so FastAPI can still read it
        request._body = body_bytes
    except Exception:
        # Logging must never break the request
        pass
    response = await call_next(request)
    return response


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Generate a unique request ID and store it in the context."""
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _route_template_for_metrics(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return "unmatched"


@app.middleware("http")
async def observe_http_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    method = (request.method or "GET").upper()
    status_code = 500
    start = time.monotonic()
    response = None

    app_metrics.inc_http_inflight()
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.monotonic() - start
        route = _route_template_for_metrics(request)
        app_metrics.dec_http_inflight()
        app_metrics.record_http_request(route=route, method=method, status_code=status_code, duration_sec=duration)

        request_id = (
            response.headers.get("X-Request-ID") if response is not None else None
        ) or get_request_id() or "unknown"
        logger.debug(
            "http_request_complete | request_id=%s | method=%s | route=%s | status=%s | duration_ms=%.1f",
            request_id,
            method,
            route,
            status_code,
            duration * 1000,
        )


class AskRequest(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    date: Optional[str] = None
    user_query: Optional[str] = None
    trip_type: Optional[str] = None          # now optional, planner may default to "Business"
    llm_mode: Optional[str] = None
    cloud_provider: Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        """If date is provided, ensure it's in YYYY-MM-DD format."""
        if v is None or v == "":
            return None
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be YYYY-MM-DD")

        return v

    @model_validator(mode="after")
    def normalize_and_validate(self):
        # Normalize: strip whitespace, convert empty strings to None
        self.origin = self.origin.strip() if self.origin else None
        self.destination = self.destination.strip() if self.destination else None
        self.date = self.date.strip() if self.date else None
        self.user_query = self.user_query.strip() if self.user_query else None
        self.trip_type = self.trip_type.strip() if self.trip_type else None
        self.llm_mode = self.llm_mode.strip() if self.llm_mode else None
        self.cloud_provider = self.cloud_provider.strip() if self.cloud_provider else None

        if self.trip_type:
            # Normalize both semantic trip intent and UI trip-mode labels.
            # Planner resolves semantic intent separately so routing labels do not override it.
            trip_type_map = {
                "business": "Business",
                "holiday": "Holiday",
                "flexible": "Flexible",
                "urgent": "Urgent",
                "one-way": "one-way",
                "one way": "one-way",
                "oneway": "one-way",
                "round-trip": "round-trip",
                "round trip": "round-trip",
                "roundtrip": "round-trip",
                "return": "round-trip",
                "via-stopover": "via-stopover",
                "via stopover": "via-stopover",
                "viastopover": "via-stopover",
                "via / stopover": "via-stopover",
                "stopover": "via-stopover",
            }
            normalized_trip_type = trip_type_map.get(self.trip_type.lower())
            if not normalized_trip_type:
                allowed = sorted(set(trip_type_map.keys()))
                raise ValueError(
                    "trip_type must be one of the supported semantic or route-mode values: "
                    + ", ".join(allowed)
                )
            self.trip_type = normalized_trip_type

        if self.llm_mode:
            self.llm_mode = normalize_llm_mode(self.llm_mode)

        if self.cloud_provider:
            self.cloud_provider = normalize_cloud_provider(self.cloud_provider)

        origin = self.origin
        destination = self.destination
        date = self.date
        user_query = self.user_query

        # Rule 1 — reject completely empty
        if not origin and not destination and not date and not user_query:
            raise ValueError(
                "At least one of user_query or origin/destination must be provided."
            )

        # Rule 2 — structured must include both origin and destination together
        if (origin or destination) and not (origin and destination):
            raise ValueError(
                "Both origin and destination must be provided together."
            )

        return self


@app.post("/ask")
async def ask(
    req: AskRequest,
    stream: bool = False,
    async_job: bool = Query(False, description="Enqueue request as background job")
):
    """
    Plan a trip based on the user's request.
    - If `async_job=true`, the request is enqueued and returns a 202 with a job_id.
    - Otherwise:
        - If `stream=false` (default), returns a single JSON response.
        - If `stream=true`, returns a Server‑Sent Events (SSE) stream of tokens.
    """
    # Define global timeout early so it's available in exception handlers
    GLOBAL_TIMEOUT = _resolve_request_timeout()

    # Use the already normalized values from the model
    origin = req.origin
    destination = req.destination
    llm_mode = req.llm_mode
    cloud_provider = req.cloud_provider
    # Detect structured‑only mode (no user query, both origin/destination present, date missing)
    is_structured_only = (
        not req.user_query
        and origin
        and destination
        and not req.date
    )

    # Compute effective date (structured default rule applies to all branches)
    DEFAULT_STRUCTURED_OFFSET_DAYS = get_env_int("DEFAULT_STRUCTURED_OFFSET_DAYS", 15)
    effective_date = req.date

    if is_structured_only:
        effective_date = (
            datetime.now() + timedelta(days=DEFAULT_STRUCTURED_OFFSET_DAYS)
        ).strftime("%Y-%m-%d")

    # Determine the user query to send to the planner
    if is_structured_only:
        # For pure structured requests, give a sensible default prompt
        planner_user_query = "Provide best available option."
    else:
        planner_user_query = req.user_query or ""

    ask_request_fingerprint: Optional[str] = None
    ask_slot_acquired = False
    stream_cleanup_owner = False

    try:
        def _failure_domain_for_reason(reason: Optional[str]) -> str:
            r = str(reason or "").strip().lower()
            if r in {"upstream_timeout", "upstream_unavailable", "provider_failure"}:
                return "upstream_provider"
            if r in {"invalid_route", "invalid_past_date", "invalid_date_order"}:
                return "request_validation"
            if r in {"no_flights"}:
                return "search_outcome"
            if r in {"stream_contract_violation", "planner_incomplete", "planner_error"}:
                return "internal_backend"
            return "internal_backend"

        def _to_sse_data_frame(payload: str) -> str:
            """
            Format a payload as a valid SSE data frame.
            Each logical line must be prefixed with `data:` to preserve embedded newlines.
            """
            text = str(payload).replace("\r\n", "\n").replace("\r", "\n")
            lines = text.split("\n")
            return "".join(f"data: {line}\n" for line in lines) + "\n"

        def _is_preformatted_sse_frame(payload: str) -> bool:
            """
            Detect whether payload is already a fully formatted SSE frame.
            This enables planner-emitted typed events (event: ... / data: ...) to pass through unchanged.
            """
            if not isinstance(payload, str):
                return False
            return payload.startswith("event: ") and payload.endswith("\n\n")

        if not async_job:
            ask_request_fingerprint = _build_ask_request_fingerprint(
                origin=origin,
                destination=destination,
                date=effective_date,
                user_query=planner_user_query,
                trip_type=req.trip_type,
                llm_mode=llm_mode,
                cloud_provider=cloud_provider,
                stream=stream,
            )
            owner_request_id = get_request_id() or "unknown"
            runtime_state = _ensure_ask_runtime_state(app)
            lock = runtime_state["lock"]
            inflight: Dict[str, Dict[str, Any]] = runtime_state["inflight"]
            now_monotonic = time.monotonic()
            stale_after_seconds = _resolve_ask_inflight_stale_seconds()
            max_inflight = _resolve_ask_max_inflight()
            duplicate_retry_after = _resolve_ask_duplicate_retry_after_seconds()
            overload_retry_after = _resolve_ask_overload_retry_after_seconds()
            duplicate_meta: Optional[Dict[str, Any]] = None
            overload_meta: Optional[Dict[str, Any]] = None

            async with lock:
                stale_removed = _prune_stale_ask_inflight_locked(
                    inflight=inflight,
                    now_monotonic=now_monotonic,
                    stale_after_seconds=stale_after_seconds,
                )
                if stale_removed:
                    logger.warning(
                        "ask_inflight_stale_pruned | removed=%s | stale_after_sec=%.2f",
                        stale_removed,
                        stale_after_seconds,
                    )
                existing = inflight.get(ask_request_fingerprint)
                if existing is not None:
                    duplicate_meta = {
                        "leader_request_id": str(existing.get("owner_request_id") or "unknown"),
                        "leader_stream": bool(existing.get("stream")),
                        "started_at_monotonic": float(existing.get("started_at") or now_monotonic),
                        "retry_after_seconds": duplicate_retry_after,
                        "inflight_size": len(inflight),
                    }
                elif len(inflight) >= max_inflight:
                    overload_meta = {
                        "retry_after_seconds": overload_retry_after,
                        "inflight_size": len(inflight),
                        "max_inflight": max_inflight,
                    }
                else:
                    inflight[ask_request_fingerprint] = {
                        "owner_request_id": owner_request_id,
                        "stream": bool(stream),
                        "started_at": now_monotonic,
                    }
                    ask_slot_acquired = True
                app_metrics.set_ask_inflight(len(inflight))

            if duplicate_meta is not None:
                app_metrics.record_ask_duplicate(stream=stream, outcome="in_progress")
                app_metrics.record_ask_admission(outcome="rejected_duplicate", stream=stream)
                response = JSONResponse(
                    status_code=409,
                    content={
                        "detail": "Duplicate /ask request already in progress in this process.",
                        "error": "duplicate_request_in_progress",
                        "retry_after_seconds": duplicate_meta["retry_after_seconds"],
                        "leader_request_id": duplicate_meta["leader_request_id"],
                        "contract": "single_node_process_local_duplicate_guard",
                        "result_status": "error",
                    },
                )
                response.headers["Retry-After"] = str(duplicate_meta["retry_after_seconds"])
                response.headers["X-Ask-Admission"] = "duplicate_in_progress"
                response.headers["X-Ask-Fingerprint"] = ask_request_fingerprint[:16]
                response.headers["X-Ask-Contract"] = "single-node-process-local"
                return response

            if overload_meta is not None:
                app_metrics.record_ask_admission(outcome="rejected_overload", stream=stream)
                response = JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Request admission temporarily saturated; please retry shortly.",
                        "error": "ask_overloaded",
                        "retry_after_seconds": overload_meta["retry_after_seconds"],
                        "inflight_requests": overload_meta["inflight_size"],
                        "max_inflight_requests": overload_meta["max_inflight"],
                        "contract": "single_node_process_local_backpressure",
                        "result_status": "error",
                    },
                )
                response.headers["Retry-After"] = str(overload_meta["retry_after_seconds"])
                response.headers["X-Ask-Admission"] = "overloaded"
                response.headers["X-Ask-Contract"] = "single-node-process-local"
                return response

            app_metrics.record_ask_admission(outcome="accepted", stream=stream)

        # Background job branch
        if async_job:
            async_job_support = _get_async_job_support_state(app)
            if not async_job_support.get("enabled", True):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "async_job_topology_unsupported",
                        "reason": async_job_support.get("reason"),
                        "declared_workers": async_job_support.get("declared_workers"),
                        "hint": "Use non-async /ask or run single-worker topology for async_job.",
                    },
                )
            from core.job_queue import enqueue_job
            # Exclude None fields for a cleaner payload
            payload = req.model_dump(exclude_none=True)
            # Override with processed values
            payload["origin"] = origin
            payload["destination"] = destination
            payload["date"] = effective_date
            payload["user_query"] = planner_user_query
            job_id = await enqueue_job(payload)
            return Response(
                status_code=202,
                content=json.dumps({"job_id": job_id}),
                media_type="application/json"
            )

        if stream:
            # Streaming branch: call planner with stream=True
            # No outer timeout – planner handles streaming timeouts internally

            async def event_stream():
                saw_done_json = False
                done_json_count = 0
                completion_enforced_by = "planner"
                try:
                    with llm_routing_context(llm_mode=llm_mode, cloud_provider=cloud_provider):
                        agen_or_result = await planner_agent.plan_trip(
                            origin=origin,
                            destination=destination,
                            date=effective_date,
                            user_query=planner_user_query,
                            trip_type=req.trip_type,
                            stream=True
                        )
                        # If the planner returns an async generator, iterate and yield SSE frames
                        if hasattr(agen_or_result, "__aiter__"):
                            async for chunk in agen_or_result:
                                chunk_text = str(chunk)
                                if "[DONE_JSON]" in chunk_text:
                                    done_json_count += 1
                                    if not saw_done_json:
                                        saw_done_json = True
                                    else:
                                        # Deterministic enforcement: suppress duplicate completion payloads.
                                        completion_enforced_by = "api_wrapper_duplicate_suppressed"
                                        logger.debug(
                                            "stream_duplicate_done_json_suppressed | request_id=%s | done_json_count=%s",
                                            get_request_id() or "unknown",
                                            done_json_count,
                                        )
                                        continue
                                elif saw_done_json:
                                    # Planner emitted data after completion marker; suppress trailing chunks.
                                    completion_enforced_by = "api_wrapper_trailing_chunk_suppressed"
                                    continue

                                if "[DONE_JSON]" in chunk_text and done_json_count > 1:
                                    continue

                                if "[DONE_JSON]" in chunk_text:
                                    saw_done_json = True
                                if _is_preformatted_sse_frame(chunk):
                                    yield chunk_text
                                else:
                                    yield _to_sse_data_frame(chunk_text)
                            if not saw_done_json:
                                app_metrics.record_stream_done_json("missing")
                                completion_enforced_by = "api_wrapper_missing_done_json"
                                logger.warning(
                                    "stream_missing_done_json | request_id=%s | llm_mode=%s | cloud_provider=%s",
                                    get_request_id() or "unknown",
                                    llm_mode or "default",
                                    cloud_provider or "default",
                                )
                                err_payload = {
                                    "error": "Streaming response ended without completion payload.",
                                    "stage": "api_stream_wrapper",
                                    "failure_reason": "stream_contract_violation",
                                    "failure_domain": _failure_domain_for_reason("stream_contract_violation"),
                                    "result_status": "error",
                                    "stream_completion_enforcement": completion_enforced_by,
                                }
                                yield _to_sse_data_frame("[ERROR] Streaming response ended without completion payload.")
                                yield _to_sse_data_frame("[DONE_JSON]" + json.dumps(err_payload, ensure_ascii=False))
                            # Final done event
                            logger.debug(
                                "stream_completion_audit | request_id=%s | done_json_count=%s | completion_enforced_by=%s",
                                get_request_id() or "unknown",
                                done_json_count,
                                completion_enforced_by,
                            )
                            yield "event: done\ndata: \n\n"
                        else:
                            # Fallback: if planner returned a dict (non-streaming), still honor stream completion contract.
                            done_payload = "[DONE_JSON]" + json.dumps(agen_or_result, ensure_ascii=False)
                            saw_done_json = True
                            done_json_count = 1
                            completion_enforced_by = "api_wrapper_non_generator_payload"
                            yield _to_sse_data_frame(done_payload)
                            logger.debug(
                                "stream_completion_audit | request_id=%s | done_json_count=%s | completion_enforced_by=%s",
                                get_request_id() or "unknown",
                                done_json_count,
                                completion_enforced_by,
                            )
                            yield "event: done\ndata: \n\n"
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("stream_wrapper_unexpected_error")
                    if not saw_done_json:
                        app_metrics.record_stream_done_json("api_wrapper_error")
                        completion_enforced_by = "api_wrapper_exception"
                        err_payload = {
                            "error": "Streaming pipeline interrupted before completion.",
                            "failure_reason": "stream_contract_violation",
                            "failure_domain": _failure_domain_for_reason("stream_contract_violation"),
                            "result_status": "error",
                            "stream_completion_enforcement": completion_enforced_by,
                        }
                        yield _to_sse_data_frame("[ERROR] Streaming pipeline interrupted before completion.")
                        yield _to_sse_data_frame("[DONE_JSON]" + json.dumps(err_payload, ensure_ascii=False))
                    yield "event: done\ndata: \n\n"
                finally:
                    if ask_slot_acquired:
                        await _release_ask_inflight_key(ask_request_fingerprint)

            stream_cleanup_owner = True
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Non‑streaming branch: apply global timeout
        with llm_routing_context(llm_mode=llm_mode, cloud_provider=cloud_provider):
            result = await asyncio.wait_for(
                planner_agent.plan_trip(
                    origin=origin,
                    destination=destination,
                    date=effective_date,
                    user_query=planner_user_query,
                    trip_type=req.trip_type,
                ),
                timeout=GLOBAL_TIMEOUT
            )

        # If the planner returns a dict with a terminal error/warning payload, treat it as non-success.
        if isinstance(result, dict):
            if "error" in result:
                detail = str(result.get("error") or "").strip() or "Planner failed to produce a complete response."
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": detail,
                        "failure_reason": result.get("failure_reason") or "planner_error",
                        "failure_domain": _failure_domain_for_reason(result.get("failure_reason") or "planner_error"),
                        "no_flights_reason": result.get("no_flights_reason"),
                        "flight_counts": result.get("flight_counts"),
                        "search_date": result.get("search_date"),
                        "result_status": result.get("result_status") or "error",
                    },
                )
            if result.get("fallback") and "warning" in result:
                detail = str(result.get("warning") or "").strip() or "No live flights found."
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": detail,
                        "failure_reason": result.get("failure_reason") or "no_flights",
                        "failure_domain": _failure_domain_for_reason(result.get("failure_reason") or "no_flights"),
                        "no_flights_reason": result.get("no_flights_reason") or "unknown",
                        "flight_counts": result.get("flight_counts"),
                        "search_date": result.get("search_date"),
                        "result_status": result.get("result_status") or "error",
                    },
                )
        return result

    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {GLOBAL_TIMEOUT} seconds")
        raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        # Re-raise HTTPExceptions that we intentionally throw
        raise
    except AirlineAPIError as e:
        # Upstream tool failed: 502 Bad Gateway is appropriate
        logger.exception("Airline API failure")
        text = str(e).lower()
        reason = "upstream_timeout" if ("timed out" in text or "timeout" in text) else "provider_failure"
        return JSONResponse(
            status_code=502,
            content={
                "detail": str(e),
                "failure_reason": reason,
                "failure_domain": _failure_domain_for_reason(reason),
                "result_status": "error",
            },
        )
    except WeatherAPIError as e:
        # Upstream tool failed: 502 Bad Gateway is appropriate
        logger.exception("Weather API failure")
        text = str(e).lower()
        reason = "upstream_timeout" if ("timed out" in text or "timeout" in text) else "provider_failure"
        return JSONResponse(
            status_code=502,
            content={
                "detail": str(e),
                "failure_reason": reason,
                "failure_domain": _failure_domain_for_reason(reason),
                "result_status": "error",
            },
        )
    except ValueError as e:
        # Defensive: bad data formatting inside planner
        logger.exception("Bad request data")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error in /ask")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if ask_slot_acquired and not stream_cleanup_owner:
            await _release_ask_inflight_key(ask_request_fingerprint)


@app.get("/booking/handoff/post/{artifact_id}", response_class=HTMLResponse)
async def booking_post_handoff_bridge(artifact_id: str):
    """
    One-time bridge for provider-managed POST booking artifacts.
    Accepts only server-issued artifact ids (no open redirect/proxy behavior).
    """
    from tools.booking_handoff import consume_post_handoff_artifact_with_diagnostics

    artifact, consume_meta = consume_post_handoff_artifact_with_diagnostics(artifact_id)
    if not artifact:
        lookup_result = str((consume_meta or {}).get("lookup_result") or "not_found")
        detail_message = {
            "already_consumed": "booking handoff artifact already consumed",
            "expired": "booking handoff artifact expired",
            "not_found": "booking handoff artifact not found",
            "consume_race_lost": "booking handoff artifact already consumed",
            "invalid_artifact_id": "booking handoff artifact id is invalid",
            "lookup_failed": "booking handoff artifact unavailable due to lookup failure",
        }.get(lookup_result, "booking handoff artifact not found or expired")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "booking_handoff_artifact_unavailable",
                "message": detail_message,
                "lookup_result": lookup_result,
                "artifact_id_prefix": (consume_meta or {}).get("artifact_id_prefix"),
            },
        )

    action_url = artifact.get("url")
    post_data = artifact.get("post_data")
    if not isinstance(action_url, str) or not action_url:
        raise HTTPException(status_code=400, detail="booking handoff artifact is invalid")

    hidden_inputs: list[str] = []

    def _append_input(name: str, value: str) -> None:
        hidden_inputs.append(
            f'<input type="hidden" name="{html.escape(name, quote=True)}" value="{html.escape(value, quote=True)}" />'
        )

    if isinstance(post_data, dict):
        for key, value in post_data.items():
            if isinstance(value, list):
                for item in value:
                    _append_input(str(key), str(item))
            elif value is not None:
                _append_input(str(key), str(value))
    elif isinstance(post_data, list):
        _append_input("__payload_json", json.dumps(post_data, ensure_ascii=False))
    elif post_data is not None:
        _append_input("__payload", str(post_data))

    html_body = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Redirecting to booking</title></head>"
        "<body>"
        "<p>Redirecting to booking provider...</p>"
        f"<form id='handoff' method='post' action='{html.escape(action_url, quote=True)}'>"
        + "".join(hidden_inputs) +
        "</form>"
        "<script>document.getElementById('handoff').submit();</script>"
        "<noscript><button type='submit' form='handoff'>Continue to booking</button></noscript>"
        "</body></html>"
    )

    return HTMLResponse(
        content=html_body,
        headers={
            "Cache-Control": "no-store",
            "X-Booking-Bridge-Consume-Result": str((consume_meta or {}).get("lookup_result") or "hit"),
        },
    )


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Retrieve the current status and result of a background job."""
    from core.job_queue import get_job
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str):
    """SSE stream of events for a background job."""
    queue = await job_queue.get_job_event_queue(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_stream():
        while True:
            # Stop if client disconnected
            if await request.is_disconnected():
                break
            try:
                evt = await queue.get()
            except asyncio.CancelledError:
                break
            if evt is None:
                break

            # Deep‑safe JSON serialization
            def to_serializable(obj):
                if hasattr(obj, "model_dump"):          # Pydantic v2
                    return obj.model_dump()
                if hasattr(obj, "dict"):                # Pydantic v1
                    return obj.dict()
                if isinstance(obj, dict):
                    return {k: to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [to_serializable(i) for i in obj]
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                return str(obj)                          # fallback

            evt = to_serializable(evt)

            # Send event as SSE data (client will parse JSON)
            yield f"data: {json.dumps(evt)}\n\n"

            # Close stream on terminal event
            if evt.get("type") in ("closed", "done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Admin‑protected debug endpoints
@app.get("/debug/keys", dependencies=[Depends(require_admin_token)])
async def debug_keys():
    """Return masked keys and their status (active/exhausted until). Requires admin token."""
    try:
        # key_manager.status() may be async or sync; handle both
        status = key_manager.status()
        if asyncio.iscoroutine(status):
            status = await status
        return status
    except Exception:
        logger.exception("debug_keys_failed")
        raise HTTPException(status_code=500, detail="key manager error")


@app.post("/debug/keys/reload", dependencies=[Depends(require_admin_token)])
async def reload_keys_endpoint():
    """Force a reload of API keys from environment variables. Requires admin token."""
    try:
        await key_manager.load_env_keys()
        return {"status": "reloaded"}
    except Exception:
        logger.exception("debug_keys_reload_failed")
        raise HTTPException(status_code=500, detail="reload failed")


@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe."""
    if not getattr(app.state, "startup_complete", False):
        health = {"status": "starting"}
        return Response(
            content=json.dumps(health),
            status_code=503,
            media_type="application/json"
        )
    prewarm = getattr(app.state, "llm_prewarm", {}) or {}
    if prewarm.get("enabled") and prewarm.get("status") in {"scheduled", "running"}:
        health = {
            "status": "warming",
            "llm_prewarm": {
                "enabled": True,
                "best_effort": bool(prewarm.get("best_effort", True)),
                "status": prewarm.get("status"),
            },
        }
        return Response(
            content=json.dumps(health),
            status_code=503,
            media_type="application/json"
        )
    return {
        "status": "ok",
        "llm_prewarm": {
            "enabled": bool(prewarm.get("enabled", False)),
            "best_effort": bool(prewarm.get("best_effort", True)),
            "status": prewarm.get("status", "disabled"),
        },
    }


@app.get("/health")
async def health():
    """Lightweight health check for container probes (no external API calls)."""
    logger.debug("lightweight health check")

    async def _check_key_manager() -> tuple[str, dict]:
        try:
            status = key_manager.status()
            if asyncio.iscoroutine(status):
                status = await status
        except Exception:
            logger.exception("lightweight_health_key_manager_failed")
            return "fail", {"reason": "status_exception"}

        if not isinstance(status, dict):
            return "degraded", {"reason": "status_not_dict"}

        service_count = len(status)
        key_entry_count = 0
        for entries in status.values():
            if isinstance(entries, list):
                key_entry_count += len(entries)
            elif isinstance(entries, dict):
                key_entry_count += len(entries)
            elif entries:
                key_entry_count += 1

        if service_count == 0 or key_entry_count == 0:
            return "degraded", {
                "reason": "empty_key_status",
                "service_count": service_count,
                "key_entry_count": key_entry_count,
            }

        return "ok", {
            "reason": "ok",
            "service_count": service_count,
            "key_entry_count": key_entry_count,
        }

    async def _check_database() -> str:
        try:
            from agents.database import SessionLocal  # local import keeps startup behavior unchanged
            from sqlalchemy import text
        except Exception:
            # Database layer not available in this runtime.
            return "unavailable"

        def _ping_db() -> None:
            db = SessionLocal()
            try:
                db.execute(text("SELECT 1"))
            finally:
                db.close()

        try:
            await asyncio.wait_for(asyncio.to_thread(_ping_db), timeout=1.0)
            return "ok"
        except asyncio.TimeoutError:
            logger.warning("lightweight_health_database_degraded_timeout")
            return "degraded"
        except Exception:
            logger.warning("lightweight_health_database_degraded")
            return "fail"

    async def _check_cloud_availability() -> tuple[str, list[str]]:
        cloud_enabled_by_config = is_cloud_admin_enabled()
        if not cloud_enabled_by_config:
            return "disabled", []
        try:
            usable = await get_usable_providers()
        except Exception:
            logger.exception("lightweight_health_cloud_probe_failed")
            return "unavailable", []
        return ("ok" if usable else "unavailable"), usable

    dependencies = {
        "app": "ok" if getattr(app.state, "startup_complete", False) else "fail",
        "key_manager": "fail",
        "database": "unavailable",
        "ollama": "not_relevant",
        "cloud": "not_relevant",
    }

    key_manager_probe, database_status, ollama_available, cloud_probe = await asyncio.gather(
        _check_key_manager(),
        _check_database(),
        _check_ollama_availability_for_options(),
        _check_cloud_availability(),
    )
    key_manager_status, key_manager_basis = key_manager_probe
    dependencies["key_manager"] = key_manager_status
    dependencies["database"] = database_status
    cloud_dep_status, usable_cloud_providers = cloud_probe

    llm_mode = get_llm_mode_default()
    primary_llm_backend = "ollama"
    fallback_llm_backend = None
    required_dependencies: list[str] = []
    fallback_dependencies: list[str] = []
    required_unavailable: list[str] = []
    fallback_unavailable: list[str] = []
    not_relevant_dependencies: list[str] = []

    ollama_status = "ok" if ollama_available else "unavailable"

    if llm_mode == LLM_MODE_OLLAMA_ONLY:
        primary_llm_backend = "ollama"
        required_dependencies = ["ollama"]
        dependencies["ollama"] = ollama_status
        dependencies["cloud"] = "not_relevant"
        not_relevant_dependencies = ["cloud"]
    elif llm_mode == LLM_MODE_CLOUD_ONLY:
        primary_llm_backend = "cloud"
        required_dependencies = ["cloud"]
        dependencies["cloud"] = cloud_dep_status
        dependencies["ollama"] = "not_relevant"
        not_relevant_dependencies = ["ollama"]
    elif llm_mode == LLM_MODE_CLOUD_FIRST:
        primary_llm_backend = "cloud"
        fallback_llm_backend = "ollama"
        required_dependencies = ["cloud"]
        fallback_dependencies = ["ollama"]
        dependencies["cloud"] = cloud_dep_status
        dependencies["ollama"] = ollama_status
    else:
        # ollama_first default
        primary_llm_backend = "ollama"
        fallback_llm_backend = "cloud"
        required_dependencies = ["ollama"]
        fallback_dependencies = ["cloud"]
        dependencies["ollama"] = ollama_status
        dependencies["cloud"] = cloud_dep_status

    for dep in required_dependencies:
        if dependencies.get(dep) != "ok":
            required_unavailable.append(dep)
    for dep in fallback_dependencies:
        if dependencies.get(dep) != "ok":
            fallback_unavailable.append(dep)

    # /health should remain stable and avoid external-API-triggered failures.
    hard_fail_fields = ("app", "key_manager")
    base_fail = any(dependencies[f] == "fail" for f in hard_fail_fields)
    if base_fail:
        status = "fail"
    elif required_unavailable and not fallback_dependencies:
        status = "fail"
    elif required_unavailable and fallback_dependencies and not all(
        dependencies.get(dep) == "ok" for dep in fallback_dependencies
    ):
        status = "fail"
    elif required_unavailable or fallback_unavailable:
        status = "degraded"
    else:
        status = "ok"

    if key_manager_status != "ok" and status == "ok":
        status = "degraded"

    prewarm = getattr(app.state, "llm_prewarm", {}) or {}
    prewarm_enabled = bool(prewarm.get("enabled", False))
    prewarm_status = str(prewarm.get("status", "disabled"))
    provider_runtime_status = get_provider_runtime_status()
    if (
        prewarm_enabled
        and primary_llm_backend == "ollama"
        and prewarm_status in {"scheduled", "running", "failed"}
        and status == "ok"
    ):
        status = "degraded"

    async_job_support = _get_async_job_support_state(app)
    refresh_owner = bool(getattr(app.state, "key_manager_refresh_owner", False))
    return {
        "status": status,
        "dependencies": dependencies,
        "llm_mode": llm_mode,
        "primary_llm_backend": primary_llm_backend,
        "fallback_llm_backend": fallback_llm_backend,
        "health_basis": {
            "required_dependencies": required_dependencies,
            "fallback_dependencies": fallback_dependencies,
            "required_unavailable": required_unavailable,
            "fallback_unavailable": fallback_unavailable,
            "not_relevant_dependencies": not_relevant_dependencies,
            "usable_cloud_providers": usable_cloud_providers,
            "cloud_provider_runtime": provider_runtime_status,
            "key_manager": key_manager_basis,
            "llm_prewarm_enabled": prewarm_enabled,
            "llm_prewarm_status": prewarm_status,
        },
        "llm_prewarm": {
            "enabled": prewarm_enabled,
            "best_effort": bool(prewarm.get("best_effort", True)),
            "status": prewarm_status,
            "model": prewarm.get("model"),
            "attempts": int(prewarm.get("attempts", 0) or 0),
            "last_error": prewarm.get("last_error"),
            "last_updated": prewarm.get("last_updated"),
        },
        "external_dependency_checks": {
            "checked": False,
            "note": "Use /health/deep for external provider health (cloud, airline, weather).",
            "deep_endpoint": "/health/deep",
        },
        "runtime_topology": {
            "pid": os.getpid(),
            "refresh_owner": refresh_owner,
            "worker_role": _worker_runtime_role(refresh_owner),
            "async_jobs_enabled": bool(async_job_support.get("enabled", True)),
            "async_job_support": async_job_support,
        },
    }


async def _check_ollama_availability_for_options() -> bool:
    probe_timeout = max(0.5, get_env_float("OLLAMA_HEALTHCHECK_TIMEOUT", 1.5))
    try:
        status = await asyncio.wait_for(ollama_client.health_check(), timeout=probe_timeout)
        if isinstance(status, bool):
            return status
        return str(status).lower() == "ok"
    except Exception:
        return False


def _derive_effective_mode_for_options(
    requested_mode: str,
    *,
    cloud_available: bool,
    ollama_available: bool,
) -> str:
    mode = (requested_mode or get_llm_mode_default()).lower()
    # Keep strict modes strict so UI/status messaging remains truthful.
    if mode in {LLM_MODE_CLOUD_ONLY, LLM_MODE_OLLAMA_ONLY}:
        return mode
    if cloud_available and not ollama_available:
        return LLM_MODE_CLOUD_ONLY
    if ollama_available and not cloud_available:
        return LLM_MODE_OLLAMA_ONLY
    return mode


def _resolve_request_timeout() -> int:
    """
    API-level guardrail timeout for non-stream requests.
    This should be a broad envelope and must not race backend/router explanation owners.
    """
    router_timeout = max(1.0, get_env_float("ROUTER_TIMEOUT", 90.0))
    request_floor = max(30.0, router_timeout + 10.0)
    configured = get_env_int("PLANNER_GLOBAL_TIMEOUT", 0)
    if configured > 0:
        return max(10, int(max(float(configured), request_floor)))

    # Derived default keeps API guardrail comfortably above router ownership.
    return int(max(90.0, router_timeout + 30.0))


def _request_timeout_source() -> str:
    configured = get_env_int("PLANNER_GLOBAL_TIMEOUT", 0)
    if configured > 0:
        router_timeout = max(1.0, get_env_float("ROUTER_TIMEOUT", 90.0))
        request_floor = max(30.0, router_timeout + 10.0)
        if float(configured) < request_floor:
            return "PLANNER_GLOBAL_TIMEOUT_clamped_to_ROUTER_TIMEOUT_plus_10s"
        return "PLANNER_GLOBAL_TIMEOUT"
    return "derived_from_ROUTER_TIMEOUT_plus_30s"


def _timeout_ownership_map() -> dict:
    stream_total_timeout = get_env_float("PLANNER_STREAM_TOTAL_TIMEOUT", 0.0)
    return {
        "non_stream_explanation": {
            "owner": "llm_router_backend_timeout",
            "driver": "LOCAL_LLM_TIMEOUT (or derived default)",
        },
        "stream_first_token": {
            "owner": "llm_router_first_chunk_timeout",
            "driver": "LOCAL_LLM_TIMEOUT (or derived default)",
        },
        "stream_chunk": {
            "owner": "llm_router_per_chunk_timeout",
            "driver": "LOCAL_LLM_TIMEOUT/CLOUD_LLM_TIMEOUT",
        },
        "stream_total": {
            "owner": "planner_stream_total_timeout",
            "enabled": stream_total_timeout > 0,
            "driver": "PLANNER_STREAM_TOTAL_TIMEOUT",
        },
        "request_guardrail_non_stream": {
            "owner": "api_wait_for_guardrail",
            "driver": _request_timeout_source(),
        },
    }


def _effective_timeout_snapshot() -> dict:
    planner_timeout = max(5.0, get_env_float("PLANNER_LLM_TIMEOUT", 45.0))
    ollama_timeout = max(1.0, get_env_float("OLLAMA_TIMEOUT", 30.0))
    local_timeout = max(1.0, get_env_float("LOCAL_LLM_TIMEOUT", max(ollama_timeout, planner_timeout)))
    cloud_timeout = max(1.0, get_env_float("CLOUD_LLM_TIMEOUT", 60.0))
    router_timeout = max(1.0, get_env_float("ROUTER_TIMEOUT", 90.0))
    stream_init_timeout = max(1.0, get_env_float("PLANNER_STREAM_INIT_TIMEOUT", planner_timeout))
    stream_total_timeout = get_env_float("PLANNER_STREAM_TOTAL_TIMEOUT", 0.0)
    return {
        "planner_llm_timeout_sec": planner_timeout,
        "ollama_timeout_sec": ollama_timeout,
        "local_llm_timeout_sec": local_timeout,
        "cloud_llm_timeout_sec": cloud_timeout,
        "router_timeout_sec": router_timeout,
        "planner_stream_init_timeout_sec": stream_init_timeout,
        "planner_stream_total_timeout_sec": stream_total_timeout,
        "request_timeout_sec": _resolve_request_timeout(),
        "request_timeout_source": _request_timeout_source(),
    }


@app.get("/llm/options")
async def llm_options():
    refresh_provider_chain_from_env(force=False)
    runtime_status = get_provider_runtime_status()
    provider_init_status = runtime_status.get("provider_init_status", {}) or {}

    mode_resolution = get_llm_mode_resolution()
    provider_chain_resolution = get_cloud_provider_chain_resolution()
    configured_providers = get_configured_cloud_providers()
    available_providers = get_available_providers()
    usable_providers = await get_usable_providers()
    provider_usability = await get_provider_usability()

    providers = [p for p in configured_providers if p in available_providers]
    if not providers:
        providers = available_providers or configured_providers

    cloud_enabled_by_config = is_cloud_admin_enabled()
    cloud_usable = cloud_enabled_by_config and len(usable_providers) > 0
    provider_switch_enabled = cloud_enabled_by_config and len(usable_providers) > 1
    requested_mode = get_llm_mode_default()
    mode_resolution = {**mode_resolution, "mode": requested_mode}
    ollama_available = await _check_ollama_availability_for_options()
    effective_mode = _derive_effective_mode_for_options(
        requested_mode,
        cloud_available=cloud_usable,
        ollama_available=ollama_available,
    )

    default_provider = get_default_cloud_provider()
    effective_default_provider = (
        default_provider
        if provider_usability.get(default_provider, False)
        else (usable_providers[0] if usable_providers else default_provider)
    )

    provider_status = {
        provider: {
            "configured": provider in configured_providers,
            "initialized": provider in available_providers,
            "usable": provider in usable_providers,
            "init_reason": (provider_init_status.get(provider) or {}).get("reason"),
        }
        for provider in sorted(set(configured_providers + available_providers))
    }
    effective_mode_dependency = get_mode_dependency_map(effective_mode)
    deprecated_env_active = sorted([var for var in _DEPRECATED_ENV_GUIDANCE if _is_env_set(var)])

    return {
        "llm_modes": list(VALID_LLM_MODES),
        "cloud_providers": providers,
        "defaults": {
            "llm_mode": requested_mode,
            "cloud_provider": default_provider,
        },
        "provider_status": provider_status,
        "usable_cloud_providers": usable_providers,
        "cloud_usable": cloud_usable,
        "cloud_enabled_by_config": cloud_enabled_by_config,
        "provider_switch_enabled": provider_switch_enabled,
        "effective_default_provider": effective_default_provider,
        "effective_mode": effective_mode,
        "backend_availability": {
            "cloud": cloud_usable,
            "ollama": ollama_available,
        },
        "config_authority": {
            "llm_mode": mode_resolution,
            "cloud_provider_chain": provider_chain_resolution,
            "mode_dependency": effective_mode_dependency,
            "effective_timeouts": _effective_timeout_snapshot(),
            "timeout_ownership": _timeout_ownership_map(),
            "deprecated_env_active": deprecated_env_active,
        },
    }


@app.get("/health/deep")
async def health_deep():
    """Deep health check (includes external API checks)."""
    logger.debug("deep health check (external APIs)")
    start = time.monotonic()
    result = await full_health_check()
    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.debug(
        "deep health check complete",
        extra={"elapsed_ms": elapsed_ms, "status": result.get("status")},
    )
    return result


@app.get("/health/keys")
async def health_keys():
    """Return key manager metadata status (no secret values)."""
    status = await key_manager.get_status()

    out = {}
    for service, entries in (status or {}).items():
        rows = []
        if isinstance(entries, list):
            iterable = enumerate(entries)
        elif isinstance(entries, dict):
            # Backward compatibility if a dict shape is returned.
            iterable = []
            for k, v in entries.items():
                try:
                    idx = int(k)
                except Exception:
                    idx = len(iterable)
                iterable.append((idx, v))
        else:
            iterable = [(0, entries)]

        for idx, entry in iterable:
            if isinstance(entry, dict):
                rows.append(
                    {
                        "index": entry.get("index", idx),
                        "active": bool(entry.get("active", False)),
                        "in_use": int(entry.get("in_use", 0) or 0),
                        "exhausted_until": entry.get("exhausted_until"),
                    }
                )
            elif isinstance(entry, str):
                rows.append(
                    {
                        "index": idx,
                        "active": entry == "active",
                        "in_use": 0,
                        "exhausted_until": None,
                    }
                )
            else:
                rows.append(
                    {
                        "index": idx,
                        "active": False,
                        "in_use": 0,
                        "exhausted_until": None,
                    }
                )
        out[service] = rows

    return out


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/version")
async def version():
    """
    Return version information to help debug deployment consistency.
    - git_commit: set via environment variable GIT_COMMIT (optional)
    - timestamp: last modification time of this file
    """
    return {
        "git_commit": get_env_str("GIT_COMMIT", "unknown"),
        "file_mtime": os.path.getmtime(__file__)
    }
