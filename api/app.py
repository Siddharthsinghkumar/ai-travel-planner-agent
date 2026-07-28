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
# Async jobs/tracking state are memory-only in this runtime: lost on restart/crash
# and not durable persistence.

import uuid
import json
import logging
import os
import asyncio
import time
import html
import hashlib
import secrets
import urllib.parse
import fcntl                     # for process‑level locking
import contextlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException, Query, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field, field_validator, model_validator
from agents.v2_graph import v2_agent
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, start_http_server

# Use module import instead of direct function import for better testability
import agents.planner_agent as planner_agent

# Import exceptions through service layer to avoid direct tool imports at API layer
from api.services_exceptions import AirlineAPIError, WeatherAPIError
from core.http_client import close_client
from core.request_context import set_request_id, get_request_id
from core.structured_logging import setup_structlog
from core.health import full_health_check
from core.async_llm_client import init_llm_client, close_llm_client
import core.metrics as app_metrics
from core.auth import AuthenticatedPrincipal, get_current_principal, get_optional_principal
from core.signal_handler import setup_signal_handlers
from core.env_config import get_env_bool, get_env_float, get_env_int, get_env_str, is_env_set
from core.ollama_context import RUNTIME_NUM_CTX_DEFAULT, resolve_runtime_num_ctx
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
    VALID_LLM_MODES,
)
from core import job_queue                     # background job worker
from core.rate_limiter import SlidingWindowRateLimiter
from core.api_key_manager import key_manager    # key rotation manager
from agents.database import init_db
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
from api.contracts import LLMOptionsResponseContract, VersionResponseContract
from api.runtime_legacy_llm import shutdown_legacy_llm_client, startup_legacy_llm_client
from api.routes_booking_tracking import build_booking_tracking_router
from api.runtime_key_manager import shutdown_key_manager_runtime, startup_key_manager_runtime
from api.runtime_price_tracker import run_price_tracker_loop

logger = logging.getLogger(__name__)
LOG_REQUEST_BODY_DEBUG = get_env_bool("LOG_REQUEST_BODY_DEBUG", default=False)

ASK_MAX_INFLIGHT_DEFAULT = max(1, get_env_int("ASK_MAX_INFLIGHT", 16))
ASK_DUPLICATE_RETRY_AFTER_DEFAULT = max(1, get_env_int("ASK_DUPLICATE_RETRY_AFTER_SECONDS", 2))
ASK_OVERLOAD_RETRY_AFTER_DEFAULT = max(1, get_env_int("ASK_OVERLOAD_RETRY_AFTER_SECONDS", 1))
ASK_RECENT_COMPLETION_TTL_DEFAULT = max(
    0.0,
    get_env_float("ASK_RECENT_COMPLETION_TTL_SECONDS", 3.0),
)
ASK_RECENT_COMPLETION_MAX_ENTRIES_DEFAULT = max(
    16,
    get_env_int("ASK_RECENT_COMPLETION_MAX_ENTRIES", 256),
)
ASK_RATE_LIMIT_WINDOW_SECONDS_DEFAULT = max(1, get_env_int("ASK_RATE_LIMIT_WINDOW_SECONDS", 60))
ASK_RATE_LIMIT_PER_WINDOW_DEFAULT = max(1, get_env_int("ASK_RATE_LIMIT_PER_WINDOW", 30))
ASK_ASYNC_RATE_LIMIT_PER_WINDOW_DEFAULT = max(
    1,
    get_env_int("ASK_ASYNC_RATE_LIMIT_PER_WINDOW", 10),
)
ADMIN_RATE_LIMIT_WINDOW_SECONDS_DEFAULT = max(
    1,
    get_env_int("ADMIN_RATE_LIMIT_WINDOW_SECONDS", 60),
)
ADMIN_RATE_LIMIT_PER_WINDOW_DEFAULT = max(
    1,
    get_env_int("ADMIN_RATE_LIMIT_PER_WINDOW", 30),
)
DIAGNOSTIC_RATE_LIMIT_PER_WINDOW_DEFAULT = max(
    1,
    get_env_int("DIAGNOSTIC_RATE_LIMIT_PER_WINDOW", 15),
)
JSON_REQUEST_BODY_MAX_BYTES_DEFAULT = max(
    16 * 1024,
    get_env_int("JSON_REQUEST_BODY_MAX_BYTES", 256 * 1024),
)
APP_SECURITY_HEADERS_ENABLE_HSTS = get_env_bool("APP_SECURITY_HEADERS_ENABLE_HSTS", default=False)
APP_SECURITY_HEADERS_HSTS_VALUE = (
    get_env_str("APP_SECURITY_HEADERS_HSTS_VALUE", "max-age=31536000; includeSubDomains") or ""
).strip() or "max-age=31536000; includeSubDomains"
APP_SECURITY_HEADERS_X_FRAME_OPTIONS = (
    get_env_str("APP_SECURITY_HEADERS_X_FRAME_OPTIONS", "DENY") or ""
).strip() or "DENY"
APP_SECURITY_HEADERS_REFERRER_POLICY = (
    get_env_str("APP_SECURITY_HEADERS_REFERRER_POLICY", "no-referrer") or ""
).strip() or "no-referrer"
APP_SECURITY_HEADERS_CSP = (
    get_env_str(
        "APP_SECURITY_HEADERS_CSP",
        "frame-ancestors 'none'; base-uri 'self'; object-src 'none'",
    )
    or ""
).strip() or "frame-ancestors 'none'; base-uri 'self'; object-src 'none'"


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


def _resolve_ask_recent_completion_ttl_seconds() -> float:
    configured = get_env_float(
        "ASK_RECENT_COMPLETION_TTL_SECONDS",
        ASK_RECENT_COMPLETION_TTL_DEFAULT,
    )
    return max(0.0, configured)


def _resolve_ask_recent_completion_max_entries() -> int:
    configured = get_env_int(
        "ASK_RECENT_COMPLETION_MAX_ENTRIES",
        ASK_RECENT_COMPLETION_MAX_ENTRIES_DEFAULT,
    )
    return max(16, configured)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_ask_rate_limit_window_seconds() -> int:
    return max(1, get_env_int("ASK_RATE_LIMIT_WINDOW_SECONDS", ASK_RATE_LIMIT_WINDOW_SECONDS_DEFAULT))


def _resolve_ask_rate_limit_per_window() -> int:
    return max(1, get_env_int("ASK_RATE_LIMIT_PER_WINDOW", ASK_RATE_LIMIT_PER_WINDOW_DEFAULT))


def _resolve_ask_async_rate_limit_per_window() -> int:
    return max(
        1,
        get_env_int("ASK_ASYNC_RATE_LIMIT_PER_WINDOW", ASK_ASYNC_RATE_LIMIT_PER_WINDOW_DEFAULT),
    )


def _resolve_admin_rate_limit_window_seconds() -> int:
    return max(
        1,
        get_env_int("ADMIN_RATE_LIMIT_WINDOW_SECONDS", ADMIN_RATE_LIMIT_WINDOW_SECONDS_DEFAULT),
    )


def _resolve_admin_rate_limit_per_window() -> int:
    return max(
        1,
        get_env_int("ADMIN_RATE_LIMIT_PER_WINDOW", ADMIN_RATE_LIMIT_PER_WINDOW_DEFAULT),
    )


def _resolve_diagnostic_rate_limit_per_window() -> int:
    return max(
        1,
        get_env_int(
            "DIAGNOSTIC_RATE_LIMIT_PER_WINDOW",
            DIAGNOSTIC_RATE_LIMIT_PER_WINDOW_DEFAULT,
        ),
    )


def _ensure_ask_rate_limiter(app: FastAPI) -> SlidingWindowRateLimiter:
    limiter = getattr(app.state, "ask_rate_limiter", None)
    if isinstance(limiter, SlidingWindowRateLimiter):
        return limiter
    limiter = SlidingWindowRateLimiter(
        max_keys=max(500, get_env_int("ASK_RATE_LIMIT_MAX_KEYS", 10000)),
        sensitive=True,
    )
    app.state.ask_rate_limiter = limiter
    return limiter


def _ensure_admin_rate_limiter(app: FastAPI) -> SlidingWindowRateLimiter:
    limiter = getattr(app.state, "admin_rate_limiter", None)
    if isinstance(limiter, SlidingWindowRateLimiter):
        return limiter
    limiter = SlidingWindowRateLimiter(
        max_keys=max(200, get_env_int("ADMIN_RATE_LIMIT_MAX_KEYS", 5000)),
        sensitive=True,
    )
    app.state.admin_rate_limiter = limiter
    return limiter


def _public_error_schema(
    *,
    code: str,
    message: str,
    request_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "code": str(code or "internal_error"),
        "message": str(message or "Request failed."),
        "request_id": str(request_id or get_request_id() or "unknown"),
    }
    if job_id:
        payload["job_id"] = str(job_id)
    return payload


def _ask_public_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    failure_reason: Optional[str] = None,
    failure_domain: Optional[str] = None,
) -> JSONResponse:
    payload: Dict[str, Any] = {
        "detail": message,
        "error": code,
        "public_error": _public_error_schema(code=code, message=message),
        "result_status": "error",
    }
    if failure_reason:
        payload["failure_reason"] = failure_reason
    if failure_domain:
        payload["failure_domain"] = failure_domain
    return JSONResponse(status_code=status_code, content=payload)


def _resolve_json_request_body_max_bytes() -> int:
    return max(
        256,
        get_env_int("JSON_REQUEST_BODY_MAX_BYTES", JSON_REQUEST_BODY_MAX_BYTES_DEFAULT),
    )


def _is_json_request(request: Request) -> bool:
    method = str(request.method or "").upper()
    if method not in {"POST", "PUT", "PATCH"}:
        return False
    content_type = str(request.headers.get("content-type") or "").lower()
    return "application/json" in content_type


def _oversized_request_response(*, max_bytes: int) -> JSONResponse:
    message = f"JSON request body exceeds the {max_bytes} byte limit."
    return JSONResponse(
        status_code=413,
        content={
            "error": "payload_too_large",
            "detail": message,
            "public_error": _public_error_schema(
                code="payload_too_large",
                message=message,
            ),
            "max_bytes": max_bytes,
        },
    )


async def _reject_oversized_json_request(request: Request) -> Optional[JSONResponse]:
    if not _is_json_request(request):
        return None
    max_bytes = _resolve_json_request_body_max_bytes()
    content_length = str(request.headers.get("content-length") or "").strip()
    if content_length:
        try:
            if int(content_length) > max_bytes:
                return _oversized_request_response(max_bytes=max_bytes)
        except Exception:
            logger.debug("invalid_content_length_header_ignored", extra={"value": content_length})
    body_bytes = await request.body()
    if len(body_bytes) > max_bytes:
        return _oversized_request_response(max_bytes=max_bytes)
    request._body = body_bytes
    return None


def _is_docs_or_openapi_path(path: str) -> bool:
    return path in {"/docs", "/redoc", "/openapi.json"} or path.startswith("/docs/")


def _apply_app_security_headers(request: Request, response: Response) -> None:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", APP_SECURITY_HEADERS_X_FRAME_OPTIONS)
    response.headers.setdefault("Referrer-Policy", APP_SECURITY_HEADERS_REFERRER_POLICY)
    if not _is_docs_or_openapi_path(str(request.url.path or "")):
        response.headers.setdefault("Content-Security-Policy", APP_SECURITY_HEADERS_CSP)
    if APP_SECURITY_HEADERS_ENABLE_HSTS and str(request.url.scheme or "").lower() == "https":
        response.headers.setdefault("Strict-Transport-Security", APP_SECURITY_HEADERS_HSTS_VALUE)


def _ask_rate_limit_subject(
    *,
    request: Request,
    principal: Optional[AuthenticatedPrincipal],
) -> str:
    if principal is not None:
        return f"principal:{principal.principal_id}"
    client_host = (getattr(request.client, "host", None) or "unknown").strip()
    return f"ip:{client_host}"


async def _build_ask_rate_limit_response(
    *,
    request: Request,
    principal: Optional[AuthenticatedPrincipal],
    async_job: bool,
) -> Optional[JSONResponse]:
    limiter = _ensure_ask_rate_limiter(app)
    window_seconds = _resolve_ask_rate_limit_window_seconds()
    limit = (
        _resolve_ask_async_rate_limit_per_window()
        if async_job
        else _resolve_ask_rate_limit_per_window()
    )
    subject = _ask_rate_limit_subject(request=request, principal=principal)
    key = f"ask:{'async' if async_job else 'sync'}:{subject}"
    decision = await limiter.check(key, limit=limit, window_seconds=window_seconds)
    if decision.allowed:
        return None

    app_metrics.record_ask_admission(outcome="rejected_rate_limited", stream=False)
    error_code = "async_job_rate_limited" if async_job else "ask_rate_limited"
    content: Dict[str, Any] = {
        "detail": "Rate limit exceeded for this caller.",
        "error": error_code,
        "retry_after_seconds": decision.retry_after_seconds,
        "limit": decision.limit,
        "window_seconds": decision.window_seconds,
        "contract": "single_node_process_local_rate_limit",
        "result_status": "error",
    }
    if async_job:
        content["job_contract"] = _job_contract_payload()
        content["job_runtime_warning"] = _job_runtime_warning_payload()
    response = JSONResponse(
        status_code=429,
        content=content,
    )
    response.headers["Retry-After"] = str(decision.retry_after_seconds)
    response.headers["X-Ask-Admission"] = "rate_limited"
    response.headers["X-Ask-Contract"] = "single-node-process-local"
    if async_job:
        _apply_job_runtime_headers(response)
    return response


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
        if not isinstance(state.get("recent_completed"), dict):
            state["recent_completed"] = {}
        return state
    state = {
        "lock": asyncio.Lock(),
        "inflight": {},
        "recent_completed": {},
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


def _prune_recent_ask_completions_locked(
    *,
    recent_completed: Dict[str, Dict[str, Any]],
    now_monotonic: float,
    ttl_seconds: float,
    max_entries: int,
) -> int:
    removed = 0
    if ttl_seconds <= 0:
        removed = len(recent_completed)
        recent_completed.clear()
        return removed

    stale_keys = []
    for key, value in list(recent_completed.items()):
        completed_at = float((value or {}).get("completed_at") or now_monotonic)
        if (now_monotonic - completed_at) > ttl_seconds:
            stale_keys.append(key)
    for key in stale_keys:
        recent_completed.pop(key, None)
        removed += 1

    if len(recent_completed) > max_entries:
        overflow = len(recent_completed) - max_entries
        oldest = sorted(
            recent_completed.items(),
            key=lambda item: float((item[1] or {}).get("completed_at") or now_monotonic),
        )[:overflow]
        for key, _ in oldest:
            recent_completed.pop(key, None)
            removed += 1

    return removed


async def _release_ask_inflight_key(request_fingerprint: Optional[str]) -> None:
    if not request_fingerprint:
        return
    runtime_state = _ensure_ask_runtime_state(app)
    lock = runtime_state["lock"]
    inflight = runtime_state["inflight"]
    async with lock:
        inflight.pop(request_fingerprint, None)
        app_metrics.set_ask_inflight(len(inflight))


async def _record_recent_ask_completion(
    request_fingerprint: Optional[str],
    completion_snapshot: Optional[Dict[str, Any]],
) -> None:
    if not request_fingerprint or not isinstance(completion_snapshot, dict):
        return
    runtime_state = _ensure_ask_runtime_state(app)
    lock = runtime_state["lock"]
    recent_completed = runtime_state["recent_completed"]
    now_monotonic = time.monotonic()
    ttl_seconds = _resolve_ask_recent_completion_ttl_seconds()
    max_entries = _resolve_ask_recent_completion_max_entries()
    async with lock:
        _prune_recent_ask_completions_locked(
            recent_completed=recent_completed,
            now_monotonic=now_monotonic,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
        )
        if ttl_seconds <= 0:
            return
        recent_completed[request_fingerprint] = {
            "completed_at": now_monotonic,
            "status_code": int(completion_snapshot.get("status_code") or 200),
            "payload": completion_snapshot.get("payload"),
        }


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
    ollama_model = get_env_str("OLLAMA_MODEL", "qwen2.5:3b")
    runtime_num_ctx = resolve_runtime_num_ctx(
        process_env=os.environ,
        dotenv_paths=[Path(__file__).resolve().parent.parent / ".env"],
        minimum_value=1,
        fallback_default=RUNTIME_NUM_CTX_DEFAULT,
    )
    ollama_num_ctx_effective = runtime_num_ctx.get("effective_num_ctx")
    ollama_num_ctx_source = str(runtime_num_ctx.get("source") or "unset")
    ollama_num_ctx_process = str(runtime_num_ctx.get("process_raw") or "") or "<unset>"
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
            "| ollama_num_ctx_process=%s | ollama_num_ctx_effective=%s | ollama_num_ctx_source=%s "
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
        ollama_num_ctx_process,
        str(ollama_num_ctx_effective) if ollama_num_ctx_effective is not None else "<unset>",
        ollama_num_ctx_source,
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
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        with contextlib.suppress(OSError):
            os.fchmod(fd, 0o600)
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
        "job_runtime_warning": _job_runtime_warning_payload(),
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


def _get_startup_complete_state(app: FastAPI) -> bool:
    return bool(getattr(app.state, "startup_complete", False))


def _get_llm_prewarm_state(app: FastAPI) -> Dict[str, Any]:
    prewarm = getattr(app.state, "llm_prewarm", {}) or {}
    if isinstance(prewarm, dict):
        return prewarm
    return {}


def _get_key_manager_refresh_owner_state(app: FastAPI) -> bool:
    return bool(getattr(app.state, "key_manager_refresh_owner", False))


def _job_contract_payload() -> Dict[str, Any]:
    # Contract is intentionally process-local/single-worker for this runtime.
    # Multi-worker shared-state/distributed async semantics are explicitly deferred.
    warning_payload = job_queue.job_runtime_warning_payload()
    return {
        "durability": "memory_only_ephemeral",
        "queue": "in_memory_single_worker",
        "contract": "single_worker_required_process_local_queue",
        "jobs_tracking_memory_only": bool(warning_payload.get("jobs_tracking_memory_only", True)),
        "lost_on_restart": bool(warning_payload.get("lost_on_restart", True)),
        "durable_persistence": bool(warning_payload.get("durable_persistence", False)),
        "warning": str(warning_payload.get("warning") or ""),
    }


def _job_runtime_warning_payload() -> Dict[str, Any]:
    return job_queue.job_runtime_warning_payload()


SSE_KEEPALIVE_INTERVAL = 20
SSE_PING_FRAME = ": ping\n\n"


async def _with_keepalive_pings(agen, *, interval_seconds=None, ping_frame=None):
    if interval_seconds is None:
        interval_seconds = SSE_KEEPALIVE_INTERVAL
    if ping_frame is None:
        ping_frame = SSE_PING_FRAME
    try:
        while True:
            task = asyncio.ensure_future(agen.__anext__())
            done, _pending = await asyncio.wait([task], timeout=interval_seconds)
            if task in done:
                try:
                    chunk = task.result()
                except StopAsyncIteration:
                    return
                yield chunk
            else:
                yield ping_frame
                try:
                    chunk = await task
                except StopAsyncIteration:
                    return
                yield chunk
    except StopAsyncIteration:
        pass


def _apply_job_runtime_headers(response: Response) -> None:
    response.headers["X-Async-Job-Durability"] = "memory-only-ephemeral"
    response.headers["X-Async-Job-Lost-On-Restart"] = "true"
    response.headers["X-Async-Job-Not-Durable"] = "true"


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


async def _run_price_tracker_loop(app: FastAPI) -> None:
    await run_price_tracker_loop(app, logger=logger)


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
    model_name = get_env_str("OLLAMA_MODEL", "qwen2.5:3b")
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


def _validate_admin_token(x_admin_token: Optional[str]) -> str:
    from core.config import TESTING
    testing = TESTING
    if testing and get_env_bool("AUTH_DISABLE_ADMIN", False):
        return "test-admin-token"
    expected = (get_env_str("ADMIN_TOKEN") or "").strip()
    provided = (x_admin_token or "").strip()
    if not expected or not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail="Forbidden")
    return provided


def _admin_rate_limit_subject(*, request: Request, token: str) -> str:
    client_host = (getattr(request.client, "host", None) or "unknown").strip()
    token_fp = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"{client_host}:{token_fp}"


async def _enforce_admin_rate_limit(*, request: Request, token: str, scope: str) -> None:
    limiter = _ensure_admin_rate_limiter(app)
    window_seconds = _resolve_admin_rate_limit_window_seconds()
    limit = (
        _resolve_diagnostic_rate_limit_per_window()
        if scope == "diagnostic"
        else _resolve_admin_rate_limit_per_window()
    )
    subject = _admin_rate_limit_subject(request=request, token=token)
    decision = await limiter.check(
        f"admin:{scope}:{subject}",
        limit=limit,
        window_seconds=window_seconds,
    )
    if decision.allowed:
        return
    raise HTTPException(
        status_code=429,
        detail={
            "error": "admin_rate_limited",
            "message": "Rate limit exceeded for admin/diagnostic endpoint access.",
            "public_error": _public_error_schema(
                code="admin_rate_limited",
                message="Rate limit exceeded for admin/diagnostic endpoint access.",
            ),
            "retry_after_seconds": decision.retry_after_seconds,
            "limit": decision.limit,
            "window_seconds": decision.window_seconds,
        },
        headers={"Retry-After": str(decision.retry_after_seconds)},
    )


async def require_admin_token(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
):
    """Dependency to protect admin endpoints with a token from environment."""
    _validate_admin_token(x_admin_token)


async def require_admin_access(
    request: Request,
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
):
    token = _validate_admin_token(x_admin_token)
    await _enforce_admin_rate_limit(request=request, token=token, scope="debug")


async def require_admin_diagnostic_access(
    request: Request,
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
):
    token = _validate_admin_token(x_admin_token)
    await _enforce_admin_rate_limit(request=request, token=token, scope="diagnostic")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_complete = False
    app.state.legacy_llm_client_initialized = False
    prewarm_enabled = get_env_bool("PLANNER_PREWARM", default=False)
    app.state.llm_prewarm = {
        "enabled": prewarm_enabled,
        "best_effort": True,
        "status": "scheduled" if prewarm_enabled else "disabled",
        "model": get_env_str("OLLAMA_MODEL", "qwen2.5:3b"),
        "attempts": 0,
        "last_error": None,
        "last_updated": _utc_now_iso(),
    }
    app.state.llm_prewarm_task = None
    app.state.async_job_support = _compute_async_job_support()
    app.state.key_manager_lease_task = None
    app.state.key_manager_refresh_owner = False
    app.state.key_manager_hydration_task = None
    app.state.ask_runtime_state = {
        "lock": asyncio.Lock(),
        "inflight": {},
        "recent_completed": {},
    }
    app.state.ask_rate_limiter = SlidingWindowRateLimiter(
        max_keys=max(500, get_env_int("ASK_RATE_LIMIT_MAX_KEYS", 10000))
    )
    app.state.price_tracker_enabled = get_env_bool("PRICE_TRACKER_ENABLED", default=True)
    app.state.price_tracker_status = {
        "enabled": bool(app.state.price_tracker_enabled),
        "last_started_at": None,
        "last_completed_at": None,
        "last_alert_count": 0,
        "last_error": None,
    }
    app.state.price_tracker_task = None

    # Startup: configure structured JSON logging
    setup_structlog()
    setup_signal_handlers()
    deprecated_env_detected = _emit_deprecated_config_warnings()

    def _validate_secret_runtime_config() -> None:
        if not get_env_bool("ENFORCE_SECRET_ENV_VALIDATION", default=False):
            return
        database_url = str(get_env_str("DATABASE_URL", "") or "").strip()
        if not database_url:
            raise RuntimeError("DATABASE_URL must be configured when ENFORCE_SECRET_ENV_VALIDATION=1")
        lowered = database_url.lower()
        insecure_markers = ("strongpassword", "changeme", "example", "admin:admin", "password@")
        if any(marker in lowered for marker in insecure_markers):
            raise RuntimeError(
                "DATABASE_URL contains a placeholder-like credential; supply a real secret at runtime."
            )

    _validate_secret_runtime_config()

    # Legacy async LLM client path is compatibility-only and opt-in.
    # Modern runtime uses llm_router + provider adapters + key-manager pools.
    await startup_legacy_llm_client(
        app,
        enabled=_legacy_async_llm_client_enabled(),
        logger=logger,
        init_client_fn=init_llm_client,
    )

    # Ensure ORM tables (including provider_key_states) exist before key-manager provider-state IO.
    try:
        init_db()
    except Exception:
        logger.exception("database_init_failed")
        raise

    key_load_timeout = max(
        0.1,
        float(get_env_float("KEY_MANAGER_STARTUP_LOAD_TIMEOUT_SECONDS", 1.5)),
    )
    lock_backend = await startup_key_manager_runtime(
        app,
        logger=logger,
        key_load_timeout=key_load_timeout,
        cloud_admin_enabled=bool(is_cloud_admin_enabled()),
        cloud_startup_relevant=bool(_is_cloud_startup_relevant_now()),
        refresh_provider_chain_from_env_fn=refresh_provider_chain_from_env,
        key_event_listener=on_key_event,
        acquire_pluggable_lock_fn=_acquire_pluggable_lock,
        run_redis_lock_lease_keeper_fn=_run_redis_lock_lease_keeper,
        key_manager_lock_ttl=KEY_MANAGER_LOCK_TTL,
        run_key_refresh_override=get_env_bool("RUN_KEY_REFRESH", default=False),
        key_env_monitor_tick=get_env_int("KEY_ENV_MONITOR_TICK", 60),
        serpapi_reconcile_interval=0,  # removed — was burning quota
    )
    should_run_refresh = bool(app.state.key_manager_refresh_owner)

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

    # Initialize the job queue (SQLite persistence)
    try:
        await job_queue.initialize_job_queue()
    except Exception:
        logger.exception("job_queue_init_failed")
        # For durability, we might want to fail startup if we can't init the queue
        raise

    # Start the background job worker loop (always needed)
    app.state.job_worker = asyncio.create_task(job_queue.worker_loop())

    # Start price-tracker loop (single-node, best-effort)
    if app.state.price_tracker_enabled:
        app.state.price_tracker_task = asyncio.create_task(_run_price_tracker_loop(app))
    else:
        app.state.price_tracker_status["note"] = "disabled_by_config"

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
            app.state.llm_prewarm["last_updated"] = _utc_now_iso()
            try:
                result = await prewarm_llm()
                app.state.llm_prewarm["attempts"] = int(result.get("attempts", 0))
                app.state.llm_prewarm["status"] = str(result.get("status", "failed"))
                app.state.llm_prewarm["last_error"] = result.get("error")
                app.state.llm_prewarm["last_updated"] = _utc_now_iso()
            except Exception:
                logger.exception("Background prewarm failed")
                app.state.llm_prewarm["status"] = "failed"
                app.state.llm_prewarm["last_error"] = "background_prewarm_exception"
                app.state.llm_prewarm["last_updated"] = _utc_now_iso()
        app.state.llm_prewarm_task = asyncio.create_task(background_prewarm())
    elif prewarm_enabled:
        app.state.llm_prewarm["status"] = "skipped_non_owner_worker"
        app.state.llm_prewarm["last_updated"] = _utc_now_iso()
        logger.debug(
            "Skipping prewarm on non-owner worker",
            extra={"pid": os.getpid(), "refresh_owner": bool(should_run_refresh)},
        )

    app.state.startup_complete = True

    # Expose Prometheus metrics on a dedicated internal port (8765) so the scrape
    # config can reach them without the admin-token gate on the public /metrics route
    # (F-006). Guarded: never bind in TESTING (tests/CI), and never fail startup if the
    # port is unavailable.
    if not get_env_bool("TESTING", default=False):
        try:
            start_http_server(8765)
            logger.info("prometheus_metrics_server_started", extra={"port": 8765})
        except Exception:
            logger.exception("prometheus_metrics_server_start_failed", extra={"port": 8765})

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

    price_tracker_task = getattr(app.state, "price_tracker_task", None)
    if price_tracker_task and not price_tracker_task.done():
        app.state.price_tracker_enabled = False
        price_tracker_task.cancel()
        try:
            await price_tracker_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("price_tracker_task_cancel_failed")

    await shutdown_key_manager_runtime(app, logger=logger)

    await shutdown_legacy_llm_client(
        app,
        close_client_fn=close_llm_client,
    )
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
app.include_router(
    build_booking_tracking_router(
        app,
        logger=logger,
        job_contract_payload_fn=_job_contract_payload,
    )
)

# Add CORS middleware with strict origin parsing.
# Production must provide explicit trusted origins (scheme + host + optional port).
_DEV_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def _normalize_cors_origin(origin: str) -> Optional[str]:
    candidate = str(origin or "").strip()
    if not candidate or candidate == "*":
        return None
    parsed = urllib.parse.urlsplit(candidate)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.hostname or parsed.query or parsed.fragment or parsed.username or parsed.password:
        return None
    if parsed.path not in {"", "/"}:
        return None

    host = str(parsed.hostname or "").lower()
    if not host:
        return None
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if parsed.port is None else f"{host}:{int(parsed.port)}"
    return f"{parsed.scheme.lower()}://{netloc}"


def _build_allowed_origins() -> list[str]:
    env_was_set = is_env_set("ALLOWED_ORIGINS")
    env_origins = get_env_str("ALLOWED_ORIGINS", ",".join(_DEV_ALLOWED_ORIGINS)) or ""
    allowed: list[str] = []
    seen: set[str] = set()

    for raw in env_origins.split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        if candidate == "*":
            logger.warning("ALLOWED_ORIGINS wildcard is not allowed; ignoring '*'")
            continue
        normalized = _normalize_cors_origin(candidate)
        if not normalized:
            logger.warning(
                "Ignoring invalid CORS origin in ALLOWED_ORIGINS; expected scheme://host[:port]",
                extra={"origin": candidate},
            )
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        allowed.append(normalized)

    if not allowed and env_was_set:
        logger.warning("No valid ALLOWED_ORIGINS entries parsed; browser cross-origin requests will be denied.")
        return []
    return allowed or list(_DEV_ALLOWED_ORIGINS)


allowed_origins = _build_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Middleware to log raw request bodies for debugging 422 errors
@app.middleware("http")
async def log_request_body(request: Request, call_next):
    oversized = await _reject_oversized_json_request(request)
    if oversized is not None:
        _apply_app_security_headers(request, oversized)
        return oversized
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
            logger.exception("request_body_debug_redaction_failed")
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
        logger.exception("request_body_debug_observe_failed")
    response = await call_next(request)
    return response


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Generate a unique request ID and store it in the context."""
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    _apply_app_security_headers(request, response)
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
    model_config = {"extra": "forbid"}
    origin: Optional[str] = Field(default=None, max_length=8)
    destination: Optional[str] = Field(default=None, max_length=8)
    date: Optional[str] = Field(default=None, max_length=32)
    user_query: Optional[str] = Field(default=None, max_length=4000)
    trip_type: Optional[str] = Field(default=None, max_length=32)          # now optional, planner may default to "Business"
    llm_mode: Optional[str] = Field(default=None, max_length=32)
    cloud_provider: Optional[str] = Field(default=None, max_length=32)

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


class ProviderStateOverrideRequest(BaseModel):
    provider: str
    scope_type: str
    scope_identifier: Optional[str] = None
    key_index: Optional[int] = None
    override_type: str
    override_until: Optional[str] = None
    active_until: Optional[str] = None
    note: Optional[str] = None

    @model_validator(mode="after")
    def normalize(self):
        self.provider = str(self.provider or "").strip().lower()
        self.scope_type = str(self.scope_type or "").strip().lower()
        self.scope_identifier = str(self.scope_identifier or "").strip() or None
        if self.key_index is not None:
            self.key_index = int(self.key_index)
        self.override_type = str(self.override_type or "").strip().lower()
        self.override_until = str(self.override_until or "").strip() or None
        self.active_until = str(self.active_until or "").strip() or None
        if self.override_until and self.active_until and self.override_until != self.active_until:
            raise ValueError("override_until and active_until must match when both are provided")
        if self.override_until and not self.active_until:
            self.active_until = self.override_until
        self.note = str(self.note or "").strip() or None
        if not self.provider:
            raise ValueError("provider is required")
        if not self.scope_type:
            raise ValueError("scope_type is required")
        if not self.override_type:
            raise ValueError("override_type is required")
        return self




class V2AskRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

@app.post("/v2/ask")
async def ask_v2(req: V2AskRequest, stream: bool = False):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    if not stream:
        result = await v2_agent.ainvoke(
            {"user_query": req.query, "thread_id": thread_id, "session_id": "api"},
            config=config
        )
        return {"result": result, "thread_id": thread_id}
        
    async def sse_generator():
        try:
            async for event in v2_agent.astream(
                {"user_query": req.query, "thread_id": thread_id, "session_id": "api"},
                config=config,
                stream_mode="updates"
            ):
                yield f"data: {json.dumps(event)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post("/ask")
async def ask(
    req: AskRequest,
    request: Request,
    stream: bool = False,
    async_job: bool = Query(False, description="Enqueue request as background job"),
    principal: Optional[AuthenticatedPrincipal] = Depends(get_optional_principal),
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
    non_stream_completion_snapshot: Optional[Dict[str, Any]] = None
    warmup_probe = str(request.headers.get("X-Validation-Warmup-Probe") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    warmup_replay_bypassed = False

    if principal is not None and not isinstance(principal, AuthenticatedPrincipal):
        principal = None

    if async_job and principal is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required for async jobs.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    rate_limit_response = await _build_ask_rate_limit_response(
        request=request,
        principal=principal,
        async_job=bool(async_job),
    )
    if rate_limit_response is not None:
        return rate_limit_response

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
            recent_completed: Dict[str, Dict[str, Any]] = runtime_state["recent_completed"]
            now_monotonic = time.monotonic()
            stale_after_seconds = _resolve_ask_inflight_stale_seconds()
            recent_ttl_seconds = _resolve_ask_recent_completion_ttl_seconds()
            recent_max_entries = _resolve_ask_recent_completion_max_entries()
            max_inflight = _resolve_ask_max_inflight()
            duplicate_retry_after = _resolve_ask_duplicate_retry_after_seconds()
            overload_retry_after = _resolve_ask_overload_retry_after_seconds()
            duplicate_meta: Optional[Dict[str, Any]] = None
            overload_meta: Optional[Dict[str, Any]] = None
            replay_meta: Optional[Dict[str, Any]] = None

            async with lock:
                stale_removed = _prune_stale_ask_inflight_locked(
                    inflight=inflight,
                    now_monotonic=now_monotonic,
                    stale_after_seconds=stale_after_seconds,
                )
                _prune_recent_ask_completions_locked(
                    recent_completed=recent_completed,
                    now_monotonic=now_monotonic,
                    ttl_seconds=recent_ttl_seconds,
                    max_entries=recent_max_entries,
                )
                if stale_removed:
                    logger.warning(
                        "ask_inflight_stale_pruned | removed=%s | stale_after_sec=%.2f",
                        stale_removed,
                        stale_after_seconds,
                    )
                    app_metrics.record_ask_inflight_stale_pruned(stale_removed)
                if not stream and recent_ttl_seconds > 0:
                    if warmup_probe:
                        if recent_completed.get(ask_request_fingerprint) is not None:
                            warmup_replay_bypassed = True
                    else:
                        recent = recent_completed.get(ask_request_fingerprint)
                        if recent is not None:
                            replay_meta = {
                                "status_code": int(recent.get("status_code") or 200),
                                "payload": recent.get("payload"),
                                "age_seconds": round(
                                    max(0.0, now_monotonic - float(recent.get("completed_at") or now_monotonic)),
                                    3,
                                ),
                            }
                existing = inflight.get(ask_request_fingerprint)
                if replay_meta is not None:
                    pass
                elif existing is not None:
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

            if replay_meta is not None:
                logger.debug(
                    "ask_admission_recent_replay",
                    extra={
                        "request_id": owner_request_id,
                        "stream": bool(stream),
                        "replay_age_seconds": replay_meta["age_seconds"],
                        "contract": "single_node_process_local_recent_replay",
                    },
                )
                app_metrics.record_ask_duplicate(stream=False, outcome="recent_replay")
                app_metrics.record_ask_admission(outcome="replayed_recent", stream=False)
                response = JSONResponse(
                    status_code=replay_meta["status_code"],
                    content=replay_meta["payload"],
                )
                response.headers["X-Ask-Admission"] = "replayed_recent"
                response.headers["X-Ask-Contract"] = "single-node-process-local"
                response.headers["X-Ask-Replay-Age-Seconds"] = str(replay_meta["age_seconds"])
                if warmup_probe:
                    response.headers["X-Validation-Warmup-Execution"] = "replayed_recent"
                    response.headers["X-Validation-Warmup-Replay-Bypassed"] = "false"
                return response

            if duplicate_meta is not None:
                logger.info(
                    "ask_admission_duplicate_rejected",
                    extra={
                        "request_id": owner_request_id,
                        "stream": bool(stream),
                        "leader_request_id": duplicate_meta["leader_request_id"],
                        "inflight_size": duplicate_meta["inflight_size"],
                        "retry_after_seconds": duplicate_meta["retry_after_seconds"],
                        "contract": "single_node_process_local_duplicate_guard",
                    },
                )
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
                if warmup_probe:
                    response.headers["X-Validation-Warmup-Execution"] = "duplicate_in_progress"
                    response.headers["X-Validation-Warmup-Replay-Bypassed"] = (
                        "true" if warmup_replay_bypassed else "false"
                    )
                return response

            if overload_meta is not None:
                logger.warning(
                    "ask_admission_overload_rejected",
                    extra={
                        "request_id": owner_request_id,
                        "stream": bool(stream),
                        "inflight_size": overload_meta["inflight_size"],
                        "max_inflight": overload_meta["max_inflight"],
                        "retry_after_seconds": overload_meta["retry_after_seconds"],
                        "contract": "single_node_process_local_backpressure",
                    },
                )
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
                if warmup_probe:
                    response.headers["X-Validation-Warmup-Execution"] = "overloaded"
                    response.headers["X-Validation-Warmup-Replay-Bypassed"] = (
                        "true" if warmup_replay_bypassed else "false"
                    )
                return response

            app_metrics.record_ask_admission(outcome="accepted", stream=stream)
            logger.debug(
                "ask_admission_accepted",
                extra={
                    "request_id": owner_request_id,
                    "stream": bool(stream),
                    "inflight_size": len(inflight),
                    "max_inflight": max_inflight,
                    "contract": "single_node_process_local_backpressure",
                },
            )

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
                        "job_contract": _job_contract_payload(),
                        "job_runtime_warning": _job_runtime_warning_payload(),
                    },
                )
            from core.job_queue import enqueue_job
            from core.job_queue import JobQueueBackpressureError
            # Exclude None fields for a cleaner payload
            payload = req.model_dump(exclude_none=True)
            # Override with processed values
            payload["origin"] = origin
            payload["destination"] = destination
            payload["date"] = effective_date
            payload["user_query"] = planner_user_query
            try:
                job_id = await enqueue_job(payload, owner_principal_id=principal.principal_id)
            except JobQueueBackpressureError as exc:
                response = JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Async job admission is saturated. Please retry.",
                        "error": "async_job_backpressure",
                        "reason": exc.reason,
                        "retry_after_seconds": exc.retry_after_seconds,
                        "contract": "single_worker_required_process_local_queue",
                        "job_contract": _job_contract_payload(),
                        "job_runtime_warning": _job_runtime_warning_payload(),
                        "result_status": "error",
                    },
                )
                response.headers["Retry-After"] = str(exc.retry_after_seconds)
                response.headers["X-Ask-Admission"] = "async_backpressure"
                response.headers["X-Ask-Contract"] = "single-node-process-local"
                _apply_job_runtime_headers(response)
                return response
            response = JSONResponse(
                status_code=202,
                content={
                    "job_id": job_id,
                    "job_contract": _job_contract_payload(),
                    "job_runtime_warning": _job_runtime_warning_payload(),
                },
            )
            _apply_job_runtime_headers(response)
            return response

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
                            stream=True,
                            owner_principal_id=principal.principal_id if principal else None,
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
            release_failsafe = None
            if ask_slot_acquired:
                # Defensive release if stream iteration exits early before generator cleanup.
                release_failsafe = BackgroundTask(_release_ask_inflight_key, ask_request_fingerprint)
            response = StreamingResponse(
                _with_keepalive_pings(event_stream()),
                media_type="text/event-stream",
                background=release_failsafe,
            )
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["Cache-Control"] = "no-cache"
            return response

        # Non‑streaming branch: apply global timeout
        with llm_routing_context(llm_mode=llm_mode, cloud_provider=cloud_provider):
            result = await asyncio.wait_for(
                planner_agent.plan_trip(
                    origin=origin,
                    destination=destination,
                    date=effective_date,
                    user_query=planner_user_query,
                    trip_type=req.trip_type,
                    owner_principal_id=principal.principal_id if principal else None,
                ),
                timeout=GLOBAL_TIMEOUT
            )

        response_status_code = 200
        response_payload: Any = result

        # If the planner returns a dict with a terminal error/warning payload, treat it as non-success.
        if isinstance(result, dict):
            handoff_meta = result.get("booking_handoff")
            handoff_rows = result.get("top_flights")
            contract_payload: Dict[str, Any] = {}
            if isinstance(handoff_meta, dict):
                contract_payload["booking_handoff"] = handoff_meta
            if isinstance(handoff_rows, list):
                contract_payload["top_flights"] = handoff_rows
            if contract_payload.get("booking_handoff") or contract_payload.get("top_flights") is not None:
                compact_debug: Dict[str, Any] = {}
                if contract_payload.get("top_flights") is not None:
                    compact_debug["top_flights"] = contract_payload.get("top_flights")
                if compact_debug:
                    contract_payload["debug_info"] = compact_debug

            if "error" in result:
                detail = str(result.get("error") or "").strip() or "Planner failed to produce a complete response."
                # Use 502 for provider failures, 500 for other internal planner errors. 
                # 400 is for client-side bad requests.
                failure_reason = result.get("failure_reason") or "planner_error"
                if failure_reason in {"provider_failure", "upstream_timeout", "upstream_unavailable"}:
                    response_status_code = 502
                else:
                    response_status_code = 500
                    
                response_payload = {
                    "detail": detail,
                    "failure_reason": failure_reason,
                    "failure_domain": _failure_domain_for_reason(failure_reason),
                    "no_flights_reason": result.get("no_flights_reason"),
                    "flight_counts": result.get("flight_counts"),
                    "search_date": result.get("search_date"),
                    "result_status": result.get("result_status") or "error",
                }
                if contract_payload:
                    response_payload.update(contract_payload)
            elif result.get("fallback") and "warning" in result:
                detail = str(result.get("warning") or "").strip() or "No live flights found."
                # Fallback warnings (no flights) are successful responses (200) with a specific status.
                response_status_code = 200
                response_payload = {
                    "detail": detail,
                    "failure_reason": result.get("failure_reason") or "no_flights",
                    "failure_domain": _failure_domain_for_reason(result.get("failure_reason") or "no_flights"),
                    "no_flights_reason": result.get("no_flights_reason") or "unknown",
                    "flight_counts": result.get("flight_counts"),
                    "search_date": result.get("search_date"),
                    "result_status": result.get("result_status") or "success",
                }
                if contract_payload:
                    response_payload.update(contract_payload)

        encoded_payload = jsonable_encoder(response_payload)
        non_stream_completion_snapshot = {
            "status_code": response_status_code,
            "payload": encoded_payload,
        }
        response = JSONResponse(status_code=response_status_code, content=encoded_payload)
        if warmup_probe:
            response.headers["X-Ask-Admission"] = response.headers.get("X-Ask-Admission") or "warmup_fresh"
            response.headers["X-Validation-Warmup-Execution"] = "fresh_execution"
            response.headers["X-Validation-Warmup-Replay-Bypassed"] = (
                "true" if warmup_replay_bypassed else "false"
            )
        return response

    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {GLOBAL_TIMEOUT} seconds")
        return _ask_public_error_response(
            status_code=504,
            code="request_timeout",
            message="Request timed out.",
            failure_reason="upstream_timeout",
            failure_domain=_failure_domain_for_reason("upstream_timeout"),
        )
    except HTTPException:
        # Re-raise HTTPExceptions that we intentionally throw
        raise
    except AirlineAPIError as e:
        # Upstream tool failed: 502 Bad Gateway is appropriate
        logger.exception("Airline API failure")
        text = str(e).lower()
        reason = "upstream_timeout" if ("timed out" in text or "timeout" in text) else "provider_failure"
        return _ask_public_error_response(
            status_code=502,
            code="upstream_provider_error",
            message="Upstream provider request failed.",
            failure_reason=reason,
            failure_domain=_failure_domain_for_reason(reason),
        )
    except WeatherAPIError as e:
        # Upstream tool failed: 502 Bad Gateway is appropriate
        logger.exception("Weather API failure")
        text = str(e).lower()
        reason = "upstream_timeout" if ("timed out" in text or "timeout" in text) else "provider_failure"
        return _ask_public_error_response(
            status_code=502,
            code="upstream_provider_error",
            message="Upstream provider request failed.",
            failure_reason=reason,
            failure_domain=_failure_domain_for_reason(reason),
        )
    except ValueError:
        # Defensive: bad data formatting inside planner
        logger.exception("Bad request data")
        return _ask_public_error_response(
            status_code=400,
            code="invalid_request_payload",
            message="Request payload is invalid.",
            failure_reason="invalid_request_payload",
            failure_domain=_failure_domain_for_reason("invalid_route"),
        )
    except Exception:
        logger.exception("Unexpected error in /ask")
        return _ask_public_error_response(
            status_code=500,
            code="internal_server_error",
            message="Internal server error.",
            failure_reason="planner_error",
            failure_domain=_failure_domain_for_reason("planner_error"),
        )
    finally:
        if ask_slot_acquired and not stream_cleanup_owner:
            await _release_ask_inflight_key(ask_request_fingerprint)
        if ask_slot_acquired and non_stream_completion_snapshot is not None and not warmup_probe:
            await _record_recent_ask_completion(
                ask_request_fingerprint,
                non_stream_completion_snapshot,
            )


@app.get("/booking/handoff/post/{artifact_id}", response_class=HTMLResponse)
async def booking_post_handoff_bridge_get(artifact_id: str, request: Request):
    """
    Non-mutating landing endpoint for one-time booking handoff.
    The artifact is consumed only via POST to preserve HTTP semantics.
    """
    accept_header = str(request.headers.get("accept") or "").lower()
    browser_prefers_html = "text/html" in accept_header or "*/*" in accept_header
    if not browser_prefers_html:
        raise HTTPException(
            status_code=405,
            detail={
                "error": "booking_handoff_post_required",
                "message": "Use POST to consume booking handoff artifacts.",
            },
        )
    escaped_artifact_id = html.escape(artifact_id, quote=True).replace("'", "&#x27;")
    html_body = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Continue to booking</title></head>"
        "<body>"
        "<p>Continue to securely submit your booking handoff.</p>"
        f"<form id='handoff-consume' method='post' action='/booking/handoff/post/{escaped_artifact_id}'>"
        "<button type='submit'>Continue to booking</button>"
        "</form>"
        "<script>document.getElementById('handoff-consume').submit();</script>"
        "</body></html>"
    )
    return HTMLResponse(content=html_body, headers={"Cache-Control": "no-store"})


@app.post("/booking/handoff/post/{artifact_id}", response_class=HTMLResponse)
async def booking_post_handoff_bridge_post(artifact_id: str, request: Request):
    """
    One-time bridge for provider-managed POST booking artifacts.
    Accepts only server-issued artifact ids (no open redirect/proxy behavior).
    """
    from api.services_exceptions import get_consume_post_handoff_artifact
    consume_post_handoff_artifact_with_diagnostics = get_consume_post_handoff_artifact()

    artifact, consume_meta = consume_post_handoff_artifact_with_diagnostics(artifact_id)
    accept_header = str(request.headers.get("accept") or "").lower()
    browser_prefers_html = "text/html" in accept_header
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
        status_code = {
            "already_consumed": 410,
            "consume_race_lost": 410,
            "expired": 410,
            "lookup_failed": 503,
            "invalid_artifact_id": 404,
            "not_found": 404,
        }.get(lookup_result, 404)
        if browser_prefers_html:
            html_body = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Booking Link Unavailable</title></head><body>"
                "<h1>Booking Link Unavailable</h1>"
                "<p>This one-time booking handoff link is no longer available.</p>"
                f"<p><strong>Reason:</strong> {html.escape(detail_message)}</p>"
                "<p>Please return to your latest search results and open a fresh booking link.</p>"
                "</body></html>"
            )
            return HTMLResponse(
                content=html_body,
                status_code=status_code,
                headers={
                    "Cache-Control": "no-store",
                    "X-Booking-Bridge-Consume-Result": lookup_result,
                },
            )
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
        escaped_name = html.escape(name, quote=True).replace("'", "&#x27;")
        escaped_value = html.escape(value, quote=True).replace("'", "&#x27;")
        hidden_inputs.append(
            f'<input type="hidden" name="{escaped_name}" value="{escaped_value}" />'
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

    escaped_action_url = html.escape(action_url, quote=True).replace("'", "&#x27;")
    html_body = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Redirecting to booking</title>"
        "<style>body{font-family:system-ui,sans-serif;display:flex;align-items:center;justify-content:center;"
        "min-height:100vh;margin:0;background:#f9fafb;color:#111827}"
        ".card{background:#fff;padding:2rem;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.1);text-align:center;"
        "max-width:400px}.spinner{width:32px;height:32px;border:3px solid #e5e7eb;border-top-color:#3b82f6;"
        "border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 1rem}"
        "@keyframes spin{to{transform:rotate(360deg)}}</style></head>"
        "<body><div class='card'><div class='spinner'></div>"
        "<h2 style='margin:0 0 .5rem;font-size:1.1rem'>Opening booking provider</h2>"
        "<p style='margin:0;color:#6b7280;font-size:.9rem'>Securely redirecting to complete your booking...</p>"
        "</div>"
        f"<form id='handoff' method='post' action='{escaped_action_url}' style='display:none'>"
        + "".join(hidden_inputs) +
        "</form>"
        "<script>setTimeout(function(){document.getElementById('handoff').submit()},300)</script>"
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


class PlanApprovalRequest(BaseModel):
    model_config = {"extra": "forbid"}
    approved: bool


@app.post("/plan/{plan_id}/approve")
async def approve_plan(
    plan_id: str,
    req: PlanApprovalRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
):
    """HITL approval gate — approve or reject a pending booking plan."""
    import time as _time
    from agents.planner_agent import _approval_store
    from core.hitl_audit import HITLAuditLogger

    _audit = HITLAuditLogger()
    _start = _time.monotonic()

    ok, reason = await _approval_store.set_decision(plan_id, req.approved, principal_id=principal.principal_id)
    latency_ms = (_time.monotonic() - _start) * 1000

    if not ok:
        _audit.log_decision(
            plan_id=plan_id,
            user_id=principal.principal_id,
            approved=req.approved,
            latency_ms=latency_ms,
            details={"status": reason or "not_found"},
        )
        if reason == "principal_mismatch":
            raise HTTPException(status_code=403, detail="HITL approval requires the plan owner principal.")
        raise HTTPException(status_code=404, detail="No pending approval found for this plan_id.")

    _audit.log_decision(
        plan_id=plan_id,
        user_id=principal.principal_id,
        approved=req.approved,
        latency_ms=latency_ms,
        details={"action": "booking_handoff"},
    )
    return {"plan_id": plan_id, "approved": req.approved, "principal_id": principal.principal_id}


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    response: Response,
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
):
    """Retrieve current status/result of an in-memory, non-durable background job."""
    from core.job_queue import get_job
    job = await get_job(job_id, owner_principal_id=principal.principal_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    payload = dict(job)
    payload.pop("event_seq", None)
    payload.pop("owner_principal_id", None)
    payload["contract"] = _job_contract_payload()
    payload["job_runtime_warning"] = _job_runtime_warning_payload()
    _apply_job_runtime_headers(response)
    return payload


@app.get("/jobs")
async def list_jobs(
    response: Response,
    limit: int = Query(100, ge=1, le=500),
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
):
    """List background jobs for the caller (in-memory, non-durable runtime state)."""
    from core.job_queue import list_jobs as list_jobs_impl

    rows = await list_jobs_impl(owner_principal_id=principal.principal_id, limit=limit)
    items: list[Dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload.pop("event_seq", None)
        payload.pop("owner_principal_id", None)
        payload["contract"] = _job_contract_payload()
        payload["job_runtime_warning"] = _job_runtime_warning_payload()
        items.append(payload)

    _apply_job_runtime_headers(response)
    return {
        "count": len(items),
        "items": items,
        "contract": _job_contract_payload(),
        "job_runtime_warning": _job_runtime_warning_payload(),
    }


@app.get("/jobs/{job_id}/events")
async def job_events(
    request: Request,
    job_id: str,
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
):
    """SSE stream of events for a background job."""
    queue = await job_queue.get_job_event_queue(job_id, owner_principal_id=principal.principal_id)
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
            event_name = str(evt.get("event") or "job_event")

            # Send event as SSE data (client will parse JSON)
            yield f"event: {event_name}\n" + f"data: {json.dumps(evt)}\n\n"

            # Close stream on terminal event
            if evt.get("event") in ("closed", "done", "error", "cancelled"):
                break

    response = StreamingResponse(_with_keepalive_pings(event_stream()), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    _apply_job_runtime_headers(response)
    return response


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    response: Response,
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
):
    """Request cancellation for a queued/running in-memory, non-durable job."""
    result = await job_queue.request_cancel_job(job_id, owner_principal_id=principal.principal_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="job not found")
    payload = dict(result)
    job = dict(payload.get("job") or {})
    if isinstance(job, dict):
        job.pop("event_seq", None)
        job.pop("owner_principal_id", None)
        job["contract"] = _job_contract_payload()
        job["job_runtime_warning"] = _job_runtime_warning_payload()
        payload["job"] = job
    payload["job_runtime_warning"] = _job_runtime_warning_payload()
    _apply_job_runtime_headers(response)
    return payload


# Admin‑protected debug endpoints
def _iter_key_status_entries(status_payload: Dict[str, Any]):
    for service, entries in (status_payload or {}).items():
        if isinstance(entries, list):
            iterable = enumerate(entries)
        elif isinstance(entries, dict):
            iterable = []
            for k, v in entries.items():
                try:
                    idx = int(k)
                except Exception:
                    idx = len(iterable)
                iterable.append((idx, v))
        else:
            iterable = [(0, entries)]
        yield service, iterable


def _sanitize_debug_key_status(status_payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for service, iterable in _iter_key_status_entries(status_payload):
        rows = []
        for idx, entry in iterable:
            if isinstance(entry, dict):
                row = {
                    "index": int(entry.get("index", idx) or idx),
                    "active": bool(entry.get("active", False)),
                    "in_use": int(entry.get("in_use", 0) or 0),
                    "exhausted_until": entry.get("exhausted_until"),
                    "pending_exhaust": bool(entry.get("pending_exhaust", False)),
                    "pending_clear": bool(entry.get("pending_clear", False)),
                    "failure_classification": entry.get("failure_classification"),
                }
                if service == "serpapi":
                    row["searches_left"] = entry.get("searches_left")
                    row["last_checked_at"] = entry.get("last_checked_at")
                rows.append(row)
            elif isinstance(entry, str):
                rows.append(
                    {
                        "index": idx,
                        "active": entry == "active",
                        "in_use": 0,
                        "exhausted_until": None,
                        "pending_exhaust": False,
                        "pending_clear": False,
                        "failure_classification": None,
                    }
                )
            else:
                rows.append(
                    {
                        "index": idx,
                        "active": False,
                        "in_use": 0,
                        "exhausted_until": None,
                        "pending_exhaust": False,
                        "pending_clear": False,
                        "failure_classification": None,
                    }
                )
        sanitized[service] = rows
    return sanitized


def _sanitize_serpapi_reconcile_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    return {
        "running": bool(meta.get("running", False)),
        "last_status": meta.get("last_status"),
        "last_started_at": meta.get("last_started_at"),
        "last_completed_at": meta.get("last_completed_at"),
    }


def _sanitize_health_key_status(status_payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for service, iterable in _iter_key_status_entries(status_payload):
        rows = []
        for idx, entry in iterable:
            if isinstance(entry, dict):
                active = bool(entry.get("active", False))
                rows.append(
                    {
                        "index": int(entry.get("index", idx) or idx),
                        "active": active,
                        "state": "active" if active else "exhausted",
                    }
                )
            elif isinstance(entry, str):
                active = entry == "active"
                rows.append(
                    {
                        "index": idx,
                        "active": active,
                        "state": "active" if active else "exhausted",
                    }
                )
            else:
                rows.append({"index": idx, "active": False, "state": "exhausted"})
        out[service] = rows
    return out


@app.get("/debug/keys", dependencies=[Depends(require_admin_access)])
async def debug_keys():
    """Return masked keys and their status (active/exhausted until). Requires admin token."""
    try:
        # key_manager.status() may be async or sync; handle both
        status = key_manager.status()
        if asyncio.iscoroutine(status):
            status = await status
        return {
            "services": _sanitize_debug_key_status(status),
            "serpapi_reconciliation": _sanitize_serpapi_reconcile_meta(
                key_manager.serpapi_reconcile_status()
            ),
        }
    except Exception:
        logger.exception("debug_keys_failed")
        raise HTTPException(status_code=500, detail="key manager error")


@app.post("/debug/keys/reload", dependencies=[Depends(require_admin_access)])
async def reload_keys_endpoint():
    """Force a reload of API keys from environment variables. Requires admin token."""
    try:
        await key_manager.load_env_keys()
        return {"status": "reloaded"}
    except Exception:
        logger.exception("debug_keys_reload_failed")
        raise HTTPException(status_code=500, detail="reload failed")


@app.get("/debug/provider-state/overrides", dependencies=[Depends(require_admin_access)])
async def debug_provider_state_overrides(
    provider: Optional[str] = Query(None),
    include_inactive: bool = Query(False),
):
    try:
        rows = await key_manager.list_provider_state_overrides(
            provider=provider,
            include_inactive=include_inactive,
        )
        return {"overrides": rows}
    except Exception:
        logger.exception("debug_provider_state_overrides_failed")
        raise HTTPException(status_code=500, detail="provider state override query failed")


@app.post("/debug/provider-state/overrides", dependencies=[Depends(require_admin_access)])
async def debug_provider_state_override_upsert(req: ProviderStateOverrideRequest):
    try:
        scope_identifier = req.scope_identifier
        if req.scope_type == "key" and not scope_identifier:
            if req.key_index is None:
                raise HTTPException(status_code=400, detail="scope_identifier or key_index is required for key scope")
            scope_identifier = await key_manager.key_scope_identifier(req.provider, int(req.key_index))
            if not scope_identifier:
                raise HTTPException(status_code=404, detail="provider key index not found for key scope override")
        row = await key_manager.set_provider_state_override(
            provider=req.provider,
            scope_type=req.scope_type,
            scope_identifier=scope_identifier,
            override_type=req.override_type,
            active_until=req.active_until,
            note=req.note,
        )
        return {"status": "ok", "override": row}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("debug_provider_state_override_upsert_failed")
        raise HTTPException(status_code=500, detail="provider state override upsert failed")


@app.post("/debug/provider-state/overrides/{override_id}/disable", dependencies=[Depends(require_admin_access)])
async def debug_provider_state_override_disable(override_id: int):
    try:
        disabled = await key_manager.disable_provider_state_override(override_id)
        if not disabled:
            raise HTTPException(status_code=404, detail="override not found")
        return {"status": "disabled", "override_id": int(override_id)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("debug_provider_state_override_disable_failed")
        raise HTTPException(status_code=500, detail="provider state override disable failed")


@app.post("/debug/provider-state/reconcile/serpapi", dependencies=[Depends(require_admin_access)])
async def debug_provider_state_reconcile_serpapi(
    key_name_fingerprints: Optional[list[str]] = None,
):
    try:
        normalized = {
            str(item or "").strip()
            for item in (key_name_fingerprints or [])
            if str(item or "").strip()
        }
        result = await key_manager.reconcile_serpapi_account_state(
            key_name_fingerprints=normalized or None
        )
        return {"status": "ok", "result": result}
    except Exception:
        logger.exception("debug_provider_state_reconcile_serpapi_failed")
        raise HTTPException(status_code=500, detail="serpapi reconcile failed")


@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe."""
    if not _get_startup_complete_state(app):
        health = {"status": "starting"}
        return Response(
            content=json.dumps(health),
            status_code=503,
            media_type="application/json"
        )
    prewarm = _get_llm_prewarm_state(app)
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
        "app": "ok" if _get_startup_complete_state(app) else "fail",
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
    key_manager_status, _key_manager_basis = key_manager_probe
    dependencies["key_manager"] = key_manager_status
    dependencies["database"] = database_status
    cloud_dep_status, _usable_cloud_providers = cloud_probe

    llm_mode = get_llm_mode_default()
    primary_llm_backend = "ollama"
    required_dependencies: list[str] = []
    fallback_dependencies: list[str] = []
    required_unavailable: list[str] = []
    fallback_unavailable: list[str] = []

    ollama_status = "ok" if ollama_available else "unavailable"

    if llm_mode == LLM_MODE_OLLAMA_ONLY:
        primary_llm_backend = "ollama"
        required_dependencies = ["ollama"]
        dependencies["ollama"] = ollama_status
        dependencies["cloud"] = "not_relevant"
    elif llm_mode == LLM_MODE_CLOUD_ONLY:
        primary_llm_backend = "cloud"
        required_dependencies = ["cloud"]
        dependencies["cloud"] = cloud_dep_status
        dependencies["ollama"] = "not_relevant"
    elif llm_mode == LLM_MODE_CLOUD_FIRST:
        primary_llm_backend = "cloud"
        required_dependencies = ["cloud"]
        fallback_dependencies = ["ollama"]
        dependencies["cloud"] = cloud_dep_status
        dependencies["ollama"] = ollama_status
    else:
        # ollama_first default
        primary_llm_backend = "ollama"
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

    prewarm = _get_llm_prewarm_state(app)
    prewarm_enabled = bool(prewarm.get("enabled", False))
    prewarm_status = str(prewarm.get("status", "disabled"))
    if (
        prewarm_enabled
        and primary_llm_backend == "ollama"
        and prewarm_status in {"scheduled", "running", "failed"}
        and status == "ok"
    ):
        status = "degraded"

    async_job_support = _get_async_job_support_state(app)
    return {
        "status": status,
        "dependencies": dependencies,
        "llm_prewarm": {
            "enabled": prewarm_enabled,
            "best_effort": bool(prewarm.get("best_effort", True)),
            "status": prewarm_status,
        },
        "external_dependency_checks": {
            "checked": False,
            "note": "Use /health/deep for external provider health (cloud, airline, weather).",
            "deep_endpoint": "/health/deep",
        },
        "async_jobs_enabled": bool(async_job_support.get("enabled", True)),
        "async_job_contract": _job_contract_payload(),
        "async_job_runtime_warning": _job_runtime_warning_payload(),
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


@app.get(
    "/llm/options",
    response_model=LLMOptionsResponseContract,
    dependencies=[Depends(require_admin_diagnostic_access)],
)
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


@app.get("/health/deep", dependencies=[Depends(require_admin_diagnostic_access)])
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


@app.get("/health/keys", dependencies=[Depends(require_admin_diagnostic_access)])
async def health_keys():
    """Return high-level key status (public, no detailed operational metadata)."""
    status = await key_manager.get_status()
    return _sanitize_health_key_status(status)


@app.get("/metrics", dependencies=[Depends(require_admin_diagnostic_access)])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/version", response_model=VersionResponseContract)
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
