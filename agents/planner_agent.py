"""
Planner Agent (Brain Layer)

Responsibilities:
- Parse user intent
- Retrieve and validate flight & weather data
- Apply preference-aware scoring
- Generate LLM explanations
- Persist full audit trail to PostgreSQL

UI-agnostic, FastAPI-ready.
Fully async with production-grade improvements.
Supports both blocking (full result) and streaming (token-by-token) responses.
"""

import asyncio
import json
import logging
import os
import re
import weakref
import time
import random
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import dateutil.parser
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from core.env_config import get_env_bool, get_env_float, get_env_int, get_env_str

# Optional: better date parsing
try:
    import dateparser
    HAS_DATEPARSER = True
except ImportError:
    HAS_DATEPARSER = False

# Use TTLCache for bounded cache
try:
    from cachetools import TTLCache
except ImportError:
    TTLCache = None

# Local imports – use wrappers for testability
from tools.airline_api import search_flights as _search_flights_impl, AirlineAPIError
from tools.weather_api import check_weather as _check_weather_impl, get_forecast_for_date as _get_forecast_for_date_impl
from agents.llm_router import generate, AllBackendsFailed
from core.llm_mode import (
    get_llm_mode_and_priority,
    LLM_MODE_OLLAMA_ONLY,
    LLM_MODE_CLOUD_ONLY,
)

# New centralised location resolver
from core.iata_resolver import (
    city_for_iata,
    is_iata_token,
    label_for_iata,
    resolve_location,
    resolve_location_with_trace,
)
from core.api_key_manager import key_manager as api_key_manager

# Wrappers (unit‑test seam)
def search_flights(*args, **kwargs):
    return _search_flights_impl(*args, **kwargs)

def check_weather(*args, **kwargs):
    return _check_weather_impl(*args, **kwargs)

def get_forecast_for_date(*args, **kwargs):
    return _get_forecast_for_date_impl(*args, **kwargs)

default_flight_tool = search_flights
default_weather_tool = get_forecast_for_date   # Use forecast-for-date as the default weather tool

# Metrics instrumentation
import core.metrics as metrics

load_dotenv()

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logger = logging.getLogger("planner_agent")
_FLIGHT_GET_WARNING_LOGGED = False

# ----------------------------------------------------------------------
# Environment flags & configurable timeouts
# ----------------------------------------------------------------------
USE_CLOUD_FALLBACK = get_env_bool("USE_CLOUD_LLM", default=True)
PLANNER_LLM_MODEL = get_env_str("PLANNER_LLM_MODEL", "gpt-4o-mini")
PLANNER_LLM_TIMEOUT = get_env_float("PLANNER_LLM_TIMEOUT", 45.0)  # seconds
PLANNER_LLM_PROMPT_SOFT_LIMIT = max(2000, get_env_int("PLANNER_LLM_PROMPT_SOFT_LIMIT", 9500))
PLANNER_LLM_PROMPT_HARD_LIMIT = max(2500, get_env_int("PLANNER_LLM_PROMPT_HARD_LIMIT", 7800))
PLANNER_LLM_TRIP_DESCRIPTION_MAX_CHARS = max(
    300,
    get_env_int("PLANNER_LLM_TRIP_DESCRIPTION_MAX_CHARS", 1400),
)
PLANNER_LLM_WARNINGS_MAX_CHARS = max(
    200,
    get_env_int("PLANNER_LLM_WARNINGS_MAX_CHARS", 1200),
)
PLANNER_LLM_MAX_FLIGHTS_ONE_WAY = max(3, get_env_int("PLANNER_LLM_MAX_FLIGHTS_ONE_WAY", 8))
PLANNER_LLM_MAX_FLIGHTS_ROUND_TRIP = max(2, get_env_int("PLANNER_LLM_MAX_FLIGHTS_ROUND_TRIP", 4))
def _resolve_planner_llm_timeout() -> float:
    return max(5.0, get_env_float("PLANNER_LLM_TIMEOUT", PLANNER_LLM_TIMEOUT))


def _resolve_router_local_timeout_hint(planner_timeout: Optional[float] = None) -> float:
    planner_timeout_hint = (
        _resolve_planner_llm_timeout()
        if planner_timeout is None
        else max(5.0, float(planner_timeout))
    )
    ollama_timeout = max(1.0, get_env_float("OLLAMA_TIMEOUT", 30.0))
    local_timeout_default = max(ollama_timeout, planner_timeout_hint)
    return max(1.0, get_env_float("LOCAL_LLM_TIMEOUT", local_timeout_default))


def _resolve_planner_llm_model() -> str:
    return (get_env_str("PLANNER_LLM_MODEL", PLANNER_LLM_MODEL) or PLANNER_LLM_MODEL).strip()


def _apply_prompt_hard_limit(prompt_text: str, *, hard_limit: int) -> Tuple[str, bool]:
    """
    Keep prompt size within a strict upper bound to reduce local-model timeout pressure.
    Preserves both the beginning and end sections (facts + user question/instructions).
    """
    if hard_limit <= 0:
        return prompt_text, False
    if len(prompt_text) <= hard_limit:
        return prompt_text, False

    marker = "\n\n[...prompt trimmed for runtime stability...]\n\n"
    min_head = int(hard_limit * 0.55)
    min_tail = hard_limit - min_head - len(marker)
    if min_tail < 120:
        # Degenerate case: hard limit is very small; keep leading section only.
        return prompt_text[:hard_limit], True

    head = prompt_text[:min_head]
    tail = prompt_text[-min_tail:]
    return f"{head}{marker}{tail}", True


def _planner_include_backend_status() -> bool:
    return get_env_bool("PLANNER_INCLUDE_BACKEND_STATUS", default=PLANNER_INCLUDE_BACKEND_STATUS)


def _stream_init_timeout_floor() -> float:
    # Keep planner stream-init timeout aligned with router local backend handshake timeout
    # to avoid planner-side premature cancellation while router still waits for first token.
    return max(20.0, _resolve_router_local_timeout_hint())


STREAM_INIT_TIMEOUT_FLOOR = _stream_init_timeout_floor()  # exported for tests/back-compat


def _resolve_stream_init_timeout() -> float:
    """
    Resolve the planner stream-init timeout from env with a safety floor.
    Prevents overly aggressive configuration from causing false stream cancellations.
    """
    floor = _stream_init_timeout_floor()
    configured = get_env_float("PLANNER_STREAM_INIT_TIMEOUT", floor)
    return max(configured, floor)


# Stream-init timeout must allow backend/router first-token handshake on healthy but slower requests.
STREAM_INIT_TIMEOUT = _resolve_stream_init_timeout()


def _resolve_stream_total_timeout(planner_timeout: float) -> Optional[float]:
    """
    Optional total stream timeout for planner consume loop.
    Default disabled (0) to avoid cancelling healthy long responses while chunks are flowing.
    """
    configured = get_env_float("PLANNER_STREAM_TOTAL_TIMEOUT", 0.0)
    if configured <= 0:
        return None
    return max(configured, planner_timeout)
PLANNER_INCLUDE_BACKEND_STATUS = get_env_bool("PLANNER_INCLUDE_BACKEND_STATUS", default=False)

# Per‑call timeouts
FLIGHT_TOOL_TIMEOUT = get_env_float("FLIGHT_TOOL_TIMEOUT", 8.0)
WEATHER_TOOL_TIMEOUT = get_env_float("WEATHER_TOOL_TIMEOUT", 5.0)
LLM_CORRECTION_TIMEOUT = get_env_float("LLM_CORRECTION_TIMEOUT", 5.0)
FLIGHT_SEARCH_BASE_RESULTS = 10
FLIGHT_SEARCH_MAX_RESULTS_CAP = 14
FLIGHT_SEARCH_WEAK_ROUTE_BONUS = 2
FLIGHT_SEARCH_ROUND_TRIP_BONUS = 2
FLIGHT_SEARCH_DEEP_SEARCH_BONUS = 2

# Return trip timeout (covers flight + weather + LLM)
RETURN_TRIP_TIMEOUT = get_env_float("RETURN_TRIP_TIMEOUT", 40.0)
BOOKING_HANDOFF_TIMEOUT = get_env_float("BOOKING_HANDOFF_TIMEOUT", 1.2)
PER_FLIGHT_HANDOFF_LIMIT = max(1, get_env_int("PER_FLIGHT_HANDOFF_LIMIT", 3))
PER_FLIGHT_HANDOFF_PROBE_LIMIT = max(
    PER_FLIGHT_HANDOFF_LIMIT,
    min(8, get_env_int("PER_FLIGHT_HANDOFF_PROBE_LIMIT", PER_FLIGHT_HANDOFF_LIMIT + 2)),
)
PER_FLIGHT_HANDOFF_PROBE_MAX = max(
    PER_FLIGHT_HANDOFF_PROBE_LIMIT,
    min(12, get_env_int("PER_FLIGHT_HANDOFF_PROBE_MAX", PER_FLIGHT_HANDOFF_PROBE_LIMIT + 3)),
)
PER_FLIGHT_HANDOFF_SCAN_LIMIT = max(
    PER_FLIGHT_HANDOFF_PROBE_LIMIT,
    min(14, get_env_int("PER_FLIGHT_HANDOFF_SCAN_LIMIT", PER_FLIGHT_HANDOFF_PROBE_LIMIT + 5)),
)
ROUND_TRIP_HANDOFF_PROBE_BONUS = max(0, get_env_int("ROUND_TRIP_HANDOFF_PROBE_BONUS", 2))
WEAK_ROUTE_HANDOFF_PROBE_BONUS = max(0, get_env_int("WEAK_ROUTE_HANDOFF_PROBE_BONUS", 2))
ROUND_TRIP_HANDOFF_TIMEOUT_BONUS = max(0.0, get_env_float("ROUND_TRIP_HANDOFF_TIMEOUT_BONUS", 0.35))
WEAK_ROUTE_HANDOFF_SIGNAL_THRESHOLD = max(1, get_env_int("WEAK_ROUTE_HANDOFF_SIGNAL_THRESHOLD", 6))
WEAK_ROUTE_ROUND_TRIP_SCAN_BONUS = max(0, get_env_int("WEAK_ROUTE_ROUND_TRIP_SCAN_BONUS", 2))
BOOKING_OPTIONS_HTTP_TIMEOUT_HINT = get_env_float("BOOKING_OPTIONS_HTTP_TIMEOUT", 2.2)
BOOKING_OPTIONS_RETRIES_HINT = max(1, get_env_int("BOOKING_OPTIONS_RETRIES", 3))
BOOKING_OPTIONS_RETRY_BACKOFF_HINT = get_env_float("BOOKING_OPTIONS_RETRY_BACKOFF", 0.15)
BOOKING_OPTIONS_ATTEMPTS_BUDGET_HINT = min(2, BOOKING_OPTIONS_RETRIES_HINT)
BOOKING_TOKEN_RESOLVE_TIMEOUT_HINT = get_env_float(
    "BOOKING_TOKEN_RESOLVE_TIMEOUT",
    (
        BOOKING_OPTIONS_HTTP_TIMEOUT_HINT * BOOKING_OPTIONS_ATTEMPTS_BUDGET_HINT
        + BOOKING_OPTIONS_RETRY_BACKOFF_HINT * max(0, BOOKING_OPTIONS_ATTEMPTS_BUDGET_HINT - 1)
        + 0.5
    ),
)
PER_FLIGHT_HANDOFF_TIMEOUT_FLOOR = min(
    6.0,
    max(
        BOOKING_TOKEN_RESOLVE_TIMEOUT_HINT + 0.25,
        BOOKING_OPTIONS_HTTP_TIMEOUT_HINT
        + BOOKING_OPTIONS_RETRY_BACKOFF_HINT * max(0, BOOKING_OPTIONS_ATTEMPTS_BUDGET_HINT - 1)
        + 0.9,
    ),
)
PER_FLIGHT_HANDOFF_TIMEOUT = max(
    get_env_float("PER_FLIGHT_HANDOFF_TIMEOUT", BOOKING_HANDOFF_TIMEOUT),
    PER_FLIGHT_HANDOFF_TIMEOUT_FLOOR,
)

# Retry configuration for flight tool
FLIGHT_RETRY_ATTEMPTS = get_env_int("FLIGHT_RETRY_ATTEMPTS", 3)
FLIGHT_RETRY_BASE = get_env_float("FLIGHT_RETRY_BASE", 0.5)      # seconds
FLIGHT_RETRY_MAX_BACKOFF = get_env_float("FLIGHT_RETRY_MAX_BACKOFF", 5.0)
FLIGHT_RETRY_JITTER = get_env_float("FLIGHT_RETRY_JITTER", 0.25)   # fraction

# Cache control
DISABLE_CACHE     = get_env_bool("DISABLE_CACHE", default=False)
CACHE_FLIGHT_TTL  = get_env_int("CACHE_FLIGHT_TTL", 900)    # default 15 min
CACHE_WEATHER_TTL = get_env_int("CACHE_WEATHER_TTL", 3600)  # default 1 hour

# NEW: Weather forecast max days limit (default to 5 for free OpenWeatherMap)
WEATHER_FORECAST_MAX_DAYS = get_env_int("WEATHER_FORECAST_MAX_DAYS", 5)

logger.info(
    f"LLM Configuration: USE_CLOUD_FALLBACK={USE_CLOUD_FALLBACK}, "
    f"MODEL={PLANNER_LLM_MODEL}, TIMEOUT={PLANNER_LLM_TIMEOUT}s"
)


def _sse_event(event_name: str, payload: Dict[str, Any]) -> str:
    """
    Return a preformatted SSE frame.
    These frames are passed through unchanged by the API streaming wrapper.
    """
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


_WEATHER_TEMP_FIELDS = {"temperature_c", "feels_like_c", "temp_min_c", "temp_max_c"}


def _normalized_weather_display(weather: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert enum-like values to plain values and normalize display-facing temperatures
    to at most one decimal place.
    """
    normalized: Dict[str, Any] = {}
    for key, raw_value in weather.items():
        value = raw_value.value if hasattr(raw_value, "value") else raw_value
        if key in _WEATHER_TEMP_FIELDS and value not in (None, "", "N/A"):
            try:
                value = round(float(value), 1)
            except Exception:
                pass
        normalized[key] = value
    return normalized

# ----------------------------------------------------------------------
# Database session logging
# ----------------------------------------------------------------------
try:
    from agents.database import SessionLocal, SessionHistory
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("Database module not available. Session logging disabled.")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MAX_RECURSION_DEPTH = 3

def _record_price_snapshot_safe(*args, **kwargs):
    """
    Lazy wrapper to avoid eager DB initialization during module import.
    """
    try:
        from tools.price_tracker import record_price_snapshot
        return record_price_snapshot(*args, **kwargs)
    except Exception:
        return None


def _flight_value_safe(flight_obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely read flight fields from dict-like payloads or Flight/Pydantic objects.
    Logs a warning once when dict-style access is attempted on non-dict objects.
    """
    global _FLIGHT_GET_WARNING_LOGGED
    if isinstance(flight_obj, dict):
        return flight_obj.get(key, default)
    if hasattr(flight_obj, key):
        if not _FLIGHT_GET_WARNING_LOGGED:
            logger.warning("Non-dict flight object encountered; using attribute access fallback")
            _FLIGHT_GET_WARNING_LOGGED = True
        return getattr(flight_obj, key, default)
    return default


async def _build_booking_handoff_url_safe(*args, **kwargs):
    try:
        from tools.booking_handoff import build_booking_handoff_url
        return await build_booking_handoff_url(*args, **kwargs)
    except Exception:
        if kwargs.get("return_details"):
            return {
                "url": None,
                "source": "unavailable",
                "reason": "booking_handoff_exception",
                "status": "unavailable",
                "handoff_mode": "unavailable",
                "landing_guarantee": "none",
                "is_exact_handoff": False,
                "is_search_fallback": False,
                "is_provider_managed": False,
                "is_booking_quality_exit": False,
                "booking_exit_quality": "unavailable",
                "failure_bucket": "exception",
                "cache_hit": False,
            }
        return None


def _booking_handoff_timeout_or_exception_fallback(
    *,
    origin: Optional[str],
    destination: Optional[str],
    depart_date: Optional[str],
    return_date: Optional[str],
    flight: Optional[Dict[str, Any]],
    failure_reason: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Build a deterministic fallback handoff payload for timeout/exception failures.
    """
    def _legacy_quality_from_tier(quality_tier: str) -> str:
        normalized = str(quality_tier or "")
        if normalized in {
            "round_trip_route_seeded_with_itinerary_hints",
            "route_seeded_with_itinerary_hints",
        }:
            return "route_seeded_with_itinerary_hints"
        if normalized in {"round_trip_route_seeded_with_return_leg", "route_seeded_basic"}:
            return "route_seeded_basic"
        return "generic"

    fallback_url: Optional[str] = None
    fallback_quality_tier = (
        "round_trip_route_seeded_with_return_leg"
        if (return_date and origin and destination and depart_date)
        else "route_seeded_basic"
    )
    fallback_quality = _legacy_quality_from_tier(fallback_quality_tier)
    fallback_context: Dict[str, Any] = {
        "route_type": "round_trip" if return_date else "one_way",
        "has_route_seed": bool(origin and destination and depart_date),
        "has_itinerary_hints": False,
        "includes_return_leg_hint": bool(return_date and origin and destination and depart_date),
        "quality_tier": fallback_quality_tier,
    }
    try:
        from tools.booking_handoff import (
            _search_fallback_context,
            _search_fallback_quality,
            build_google_flights_fallback,
        )
        if origin and destination and depart_date:
            fallback_url = build_google_flights_fallback(
                origin=origin,
                destination=destination,
                depart_date=depart_date,
                return_date=return_date,
                flight=flight,
            )
            fallback_quality_tier = _search_fallback_quality(
                origin=origin,
                destination=destination,
                depart_date=depart_date,
                return_date=return_date,
                flight=flight,
            )
            fallback_quality = _legacy_quality_from_tier(fallback_quality_tier)
            fallback_context = _search_fallback_context(
                origin=origin,
                destination=destination,
                depart_date=depart_date,
                return_date=return_date,
                flight=flight,
            )
            fallback_context["quality_tier"] = fallback_quality_tier
    except Exception:
        fallback_url = None
        if isinstance(flight, dict) and (
            str(flight.get("airline") or "").strip()
            or str(flight.get("flight_no") or "").strip()
            or str(flight.get("departure_time") or "").strip()
            or str(flight.get("arrival_time") or "").strip()
        ):
            if return_date and origin and destination and depart_date:
                fallback_quality_tier = "round_trip_route_seeded_with_itinerary_hints"
            elif return_date:
                fallback_quality_tier = "round_trip_route_seeded_with_return_leg"
            else:
                fallback_quality_tier = "route_seeded_with_itinerary_hints"
            fallback_context["has_itinerary_hints"] = True
        fallback_quality = _legacy_quality_from_tier(fallback_quality_tier)
        fallback_context["quality_tier"] = fallback_quality_tier

    if fallback_url:
        fallback_context["quality_tier"] = fallback_quality_tier
        return fallback_url, {
            "source": "google_flights_fallback",
            "reason": f"{failure_reason}_google_search_fallback",
            "provider": "google_flights",
            "status": "ok",
            "handoff_mode": "search_fallback",
            "landing_guarantee": "best_effort_search",
            "search_fallback_quality": fallback_quality,
            "search_fallback_context": fallback_context,
            "is_exact_handoff": False,
            "is_search_fallback": True,
            "is_provider_managed": False,
            "is_booking_quality_exit": False,
            "booking_exit_quality": "search_assist",
            "failure_bucket": "timeout" if "timeout" in failure_reason else "exception",
            "cache_hit": False,
        }

    if failure_reason == "booking_handoff_timeout":
        return None, {
            "source": "timeout",
            "reason": failure_reason,
            "status": "unavailable",
            "handoff_mode": "unavailable",
            "landing_guarantee": "none",
            "is_exact_handoff": False,
            "is_search_fallback": False,
            "is_provider_managed": False,
            "is_booking_quality_exit": False,
            "booking_exit_quality": "unavailable",
            "failure_bucket": "timeout",
            "cache_hit": False,
        }

    return None, {
        "source": "unavailable",
        "reason": failure_reason,
        "status": "unavailable",
        "handoff_mode": "unavailable",
        "landing_guarantee": "none",
        "is_exact_handoff": False,
        "is_search_fallback": False,
        "is_provider_managed": False,
        "is_booking_quality_exit": False,
        "booking_exit_quality": "unavailable",
        "failure_bucket": "exception",
        "cache_hit": False,
    }


async def _resolve_flight_booking_handoff(
    *,
    flight_obj: Any,
    origin: Optional[str],
    destination: Optional[str],
    depart_date: Optional[str],
    return_date: Optional[str],
    timeout_sec: float,
    candidate_rank: Optional[int] = None,
    probe_signal: Optional[int] = None,
    route_type: str = "one_way",
    cache_mode_hint: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], Optional[str]]:
    """
    Resolve booking handoff for one flight and preserve an explicit classification payload.
    Returns (possibly-updated flight object, handoff info, handoff URL).
    """
    booking_handoff_info: Dict[str, Any] = {
        "source": "unavailable",
        "reason": "not_attempted",
        "status": "unavailable",
    }
    booking_url: Optional[str] = None
    handoff_started = time.monotonic()

    flight_payload = (
        flight_obj.model_dump()
        if hasattr(flight_obj, "model_dump")
        else (dict(vars(flight_obj)) if hasattr(flight_obj, "__dict__") else {})
    )
    has_booking_token = bool(isinstance(flight_payload, dict) and flight_payload.get("booking_token"))
    has_shareable = bool(
        isinstance(flight_payload, dict)
        and (
            flight_payload.get("shareable_link")
            or flight_payload.get("provider_link")
            or flight_payload.get("partner_booking_link")
            or flight_payload.get("booking_url")
        )
    )

    booking_task = asyncio.create_task(
        _build_booking_handoff_url_safe(
            flight=flight_payload,
            origin=origin,
            destination=destination,
            depart_date=depart_date,
            return_date=return_date,
            return_details=True,
        )
    )

    try:
        booking_result = await asyncio.wait_for(booking_task, timeout=timeout_sec)
        if isinstance(booking_result, dict):
            booking_url = booking_result.get("url")
            reason_value = booking_result.get("reason")
            cache_hit_value = booking_result.get("cache_hit")
            if cache_hit_value is None and isinstance(reason_value, str):
                cache_hit_value = reason_value.endswith("_cache")
            failure_bucket_value = booking_result.get("failure_bucket")
            if not failure_bucket_value and isinstance(reason_value, str) and reason_value:
                failure_bucket_value = reason_value
            booking_handoff_info = {
                "source": booking_result.get("source") or "unavailable",
                "reason": reason_value or "unknown",
                "provider": booking_result.get("provider"),
                "status": booking_result.get("status") or ("ok" if booking_url else "unavailable"),
                "handoff_mode": booking_result.get("handoff_mode"),
                "landing_guarantee": booking_result.get("landing_guarantee"),
                "search_fallback_quality": booking_result.get("search_fallback_quality"),
                "search_fallback_context": booking_result.get("search_fallback_context"),
                "artifact_field": booking_result.get("artifact_field"),
                "requires_browser_post": booking_result.get("requires_browser_post"),
                "is_exact_handoff": booking_result.get("is_exact_handoff"),
                "is_search_fallback": booking_result.get("is_search_fallback"),
                "is_provider_managed": booking_result.get("is_provider_managed"),
                "is_booking_quality_exit": booking_result.get("is_booking_quality_exit"),
                "booking_exit_quality": booking_result.get("booking_exit_quality"),
                "failure_bucket": failure_bucket_value,
                "cache_hit": cache_hit_value,
            }
        else:
            booking_url = booking_result
            if booking_url:
                booking_handoff_info = {"source": "unknown", "reason": "legacy_url_only", "status": "ok"}
            else:
                booking_handoff_info = {"source": "unavailable", "reason": "legacy_url_missing", "status": "unavailable"}
    except asyncio.TimeoutError:
        booking_url, booking_handoff_info = _booking_handoff_timeout_or_exception_fallback(
            origin=origin,
            destination=destination,
            depart_date=depart_date,
            return_date=return_date,
            flight=flight_payload,
            failure_reason="booking_handoff_timeout",
        )
        booking_handoff_info["failure_bucket"] = "timeout"
        booking_handoff_info["cache_hit"] = False
    except Exception as exc:
        logger.info("build_booking_handoff_url failed; continuing without handoff URL: %s", str(exc))
        booking_url, booking_handoff_info = _booking_handoff_timeout_or_exception_fallback(
            origin=origin,
            destination=destination,
            depart_date=depart_date,
            return_date=return_date,
            flight=flight_payload,
            failure_reason="booking_handoff_exception",
        )
        booking_handoff_info["failure_bucket"] = "exception"
        booking_handoff_info["cache_hit"] = False

    if booking_url and hasattr(flight_obj, "model_copy"):
        flight_obj = flight_obj.model_copy(update={"handoff_url": booking_url})
    elif booking_url and isinstance(flight_obj, dict):
        updated = dict(flight_obj)
        updated["handoff_url"] = booking_url
        flight_obj = updated

    flight_no = None
    if isinstance(flight_payload, dict):
        flight_no = flight_payload.get("flight_no")
    logger.info(
        "booking_handoff_flight_resolution",
        extra={
            "flight_no": flight_no,
            "route_type": route_type,
            "candidate_rank": candidate_rank,
            "probe_signal": probe_signal,
            "cache_mode_hint": cache_mode_hint,
            "has_booking_token": has_booking_token,
            "has_shareable_hint": has_shareable,
            "timeout_sec": timeout_sec,
            "duration_ms": int((time.monotonic() - handoff_started) * 1000),
            "result_bucket": _booking_handoff_bucket(booking_handoff_info),
            "is_booking_ready": bool(booking_handoff_info.get("is_booking_quality_exit")),
            "reason": booking_handoff_info.get("reason"),
            "failure_bucket": booking_handoff_info.get("failure_bucket"),
            "cache_hit": booking_handoff_info.get("cache_hit"),
        },
    )

    return flight_obj, booking_handoff_info, booking_url


def _booking_handoff_strength(meta: Dict[str, Any]) -> int:
    """
    Higher score = better booking-quality handoff for top-level promotion.
    Keeps per-flight rows unchanged; only used for primary booking_handoff selection.
    """
    if not isinstance(meta, dict):
        return 0

    if meta.get("is_booking_quality_exit"):
        if meta.get("requires_browser_post"):
            return 500
        if meta.get("handoff_mode") == "direct_booking":
            return 480
        return 460

    # Backward-compatible promotion for legacy compact metadata that may not
    # carry explicit is_booking_quality_exit flags.
    source = str(meta.get("source") or "").strip().lower()
    status = str(meta.get("status") or "").strip().lower()
    if status == "ok" and source in {"booking_token", "shareable_link"}:
        return 440

    if meta.get("is_provider_managed"):
        return 420

    if meta.get("is_search_fallback"):
        quality = str(meta.get("search_fallback_quality") or "")
        context_payload = meta.get("search_fallback_context")
        if isinstance(context_payload, dict):
            quality_tier = str(context_payload.get("quality_tier") or "")
            if quality_tier:
                quality = quality_tier
        if quality == "round_trip_route_seeded_with_itinerary_hints":
            return 280
        if quality == "round_trip_route_seeded_with_return_leg":
            return 240
        if quality == "route_seeded_with_itinerary_hints":
            return 260
        if quality == "route_seeded_basic":
            return 220
        return 180

    if status == "ok":
        return 120

    if source in {"timeout", "unavailable"}:
        return 20

    return 0


def _booking_handoff_bucket(meta: Dict[str, Any]) -> str:
    if not isinstance(meta, dict):
        return "unknown"
    if meta.get("is_booking_quality_exit"):
        return "booking_ready"
    reason = str(meta.get("reason") or "")
    if reason.startswith("booking_handoff_timeout"):
        return "timeout"
    if reason.startswith("booking_options_provider_error"):
        return "provider_error"
    if reason.startswith("booking_options_http_error"):
        return "http_error"
    if reason.startswith("booking_options_request_exception"):
        return "request_exception"
    if reason.startswith("booking_options_parse_error"):
        return "parse_error"
    if str(meta.get("source") or "") == "google_flights_fallback":
        return "search_fallback"
    if str(meta.get("status") or "") == "unavailable":
        return "unavailable"
    return "other"


def _align_top_level_booking_handoff_with_rows(
    top_level_meta: Dict[str, Any],
    top_flights_payload: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str], bool]:
    """
    Ensure top-level booking handoff is never weaker than the strongest per-flight row
    already present in the response payload.
    """
    current_meta = dict(top_level_meta or {})
    current_score = _booking_handoff_strength(current_meta)
    best_meta = current_meta
    best_score = current_score
    best_rank = int(current_meta.get("selected_flight_rank", 1) or 1)
    best_url: Optional[str] = None

    for idx, row in enumerate(top_flights_payload or []):
        if not isinstance(row, dict):
            continue
        candidate_meta = row.get("booking_handoff") or {}
        if not isinstance(candidate_meta, dict):
            continue
        candidate_score = _booking_handoff_strength(candidate_meta)
        candidate_rank = int(row.get("rank", idx + 1) or (idx + 1))
        if candidate_score > best_score:
            best_score = candidate_score
            best_meta = dict(candidate_meta)
            best_rank = candidate_rank
            best_url = (
                row.get("handoff_url")
                or row.get("booking_url")
                or None
            )

    if best_score <= current_score:
        return current_meta, None, False

    if best_rank != 1:
        best_meta["selected_flight_rank"] = best_rank
    return best_meta, best_url, True


def _booking_handoff_cache_snapshot() -> Dict[str, Any]:
    """
    Returns compact cache stats for cold-vs-hot booking handoff diagnostics.
    """
    try:
        from tools.booking_handoff import booking_resolution_cache_stats

        stats = booking_resolution_cache_stats()
        entries = int(stats.get("entries") or 0)
        ttl_sec = int(stats.get("ttl_sec") or 0)
        return {
            "cache_mode": "hot" if entries > 0 else "cold",
            "cache_entries": entries,
            "cache_ttl_sec": ttl_sec,
        }
    except Exception:
        return {
            "cache_mode": "unknown",
            "cache_entries": None,
            "cache_ttl_sec": None,
        }


def _flight_payload_for_handoff_signal(flight_obj: Any) -> Dict[str, Any]:
    if hasattr(flight_obj, "model_dump"):
        payload = flight_obj.model_dump()
        return payload if isinstance(payload, dict) else {}
    if isinstance(flight_obj, dict):
        return dict(flight_obj)
    if hasattr(flight_obj, "__dict__"):
        try:
            payload = dict(vars(flight_obj))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _booking_handoff_candidate_signal(flight_obj: Any) -> int:
    """
    Heuristic signal for booking-artifact richness.
    Higher scores are more likely to yield booking-ready exits.
    """
    payload = _flight_payload_for_handoff_signal(flight_obj)
    if not payload:
        return 0

    score = 0
    if payload.get("booking_token"):
        score += 8
    booking_request = payload.get("booking_request")
    if isinstance(booking_request, dict):
        score += 6
        if booking_request.get("url") or booking_request.get("endpoint") or booking_request.get("booking_url"):
            score += 2
        if booking_request.get("post_data") not in (None, "", {}, []):
            score += 2
    options = payload.get("booking_options")
    if isinstance(options, list) and options:
        score += 2
        has_option_link = False
        for opt in options[:8]:
            if not isinstance(opt, dict):
                continue
            if any(opt.get(key) for key in ("booking_url", "link", "url", "deeplink", "redirect_link")):
                has_option_link = True
                break
        if has_option_link:
            score += 3
    if any(
        payload.get(field)
        for field in ("shareable_link", "provider_link", "partner_booking_link", "booking_url")
    ):
        score += 4
    if any(isinstance(payload.get(field), (dict, list)) for field in ("offer", "offers", "providers", "book_with")):
        score += 1
    return score


async def _hold_booking_safe(*args, **kwargs):
    from tools.booking_handoff import hold_booking
    return await hold_booking(*args, **kwargs)


def _confirm_booking_safe(*args, **kwargs):
    from tools.booking_handoff import confirm_booking
    return confirm_booking(*args, **kwargs)


def _cancel_booking_safe(*args, **kwargs):
    from tools.booking_handoff import cancel_booking
    return cancel_booking(*args, **kwargs)


def _parse_price_insights_safe(*args, **kwargs):
    try:
        from tools.price_tracker import parse_price_insights
        return parse_price_insights(*args, **kwargs)
    except Exception:
        return None


def _format_price_insights_for_llm_safe(*args, **kwargs):
    try:
        from tools.price_tracker import format_price_insights_for_llm
        return format_price_insights_for_llm(*args, **kwargs)
    except Exception:
        return ""


def _analyze_price_trend_safe(*args, **kwargs):
    try:
        from tools.price_tracker import analyze_price_trend
        return analyze_price_trend(*args, **kwargs)
    except Exception:
        return None


def _predict_future_price_safe(*args, **kwargs):
    try:
        from tools.price_tracker import predict_future_price
        return predict_future_price(*args, **kwargs)
    except Exception:
        return None


def _detect_booking_or_tracking_action(user_query: str) -> Optional[str]:
    """
    Detect explicit booking lifecycle / price tracking intents.
    Returns one of: confirm_booking, cancel_booking, hold_booking, track_price, or None.
    """
    q = (user_query or "").lower()
    if not q:
        return None

    if ("confirm" in q and ("booking" in q or "reservation" in q)):
        return "confirm_booking"
    if any(x in q for x in ("cancel booking", "cancel my reservation", "cancel reservation", "cancel my booking")):
        return "cancel_booking"
    if any(x in q for x in ("notify me if", "track this flight price", "track price", "price drops", "price decreases", "alert me if price")):
        return "track_price"
    if any(x in q for x in ("hold this flight", "hold booking", "hold my booking", "reserve this flight", "reserve my booking")):
        return "hold_booking"
    return None


def _extract_booking_id(user_query: str) -> Optional[int]:
    """
    Extract booking id from phrases like:
      - booking 123
      - booking id: 123
      - reservation #123
      - id 123
    """
    q = user_query or ""
    patterns = [
        r"\b(?:booking|reservation)\s*(?:id)?\s*[:#]?\s*(\d+)\b",
        r"\b(?:id|#)\s*(\d+)\b",
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None

def _resolve_city_to_iata_with_trace(city_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Resolve a free-form city string to an IATA code. Priorities:
      1) explicit valid 3-letter IATA token (uppercase token from user input)
      2) resolve_location(...) over cleaned candidate variants
      3) return None
    """
    trace: Dict[str, Any] = {
        "input": city_text,
        "candidates": [],
        "selected_iata": None,
        "resolution_basis": "unresolved",
        "resolver_trace": None,
    }
    if not city_text:
        trace["resolution_basis"] = "empty_input"
        return None, trace
    token = city_text.strip().lower()
    cleaned = " ".join(
        p for p in re.findall(r"[a-z]+", token)
        if p not in {
            "flight", "flights", "from", "to", "on", "at", "for", "via", "through",
            "stopover", "stop", "in", "return", "returning", "leaving", "departing",
            "coming", "back", "after", "before", "trip", "business", "holiday",
            "urgent", "flexible", "book", "booking", "ticket", "tickets", "find",
            "cheapest", "cheap", "and", "with", "under", "tomorrow", "today", "next",
            "this", "day", "days", "week", "weeks",
        }
    ).strip()
    candidates = [c for c in [cleaned, token] if c]
    trace["candidates"] = list(candidates)

    # 1) explicit uppercase IATA token present anywhere in the fragment.
    for token_match in re.findall(r"\b([A-Z]{3})\b", city_text):
        if is_iata_token(token_match):
            trace["selected_iata"] = token_match
            trace["resolution_basis"] = "explicit_iata_token"
            return token_match, trace

    # 2) explicit 3-letter uppercase token when it is a real IATA code.
    explicit_token = city_text.strip()
    if len(explicit_token) == 3 and explicit_token.isalpha() and explicit_token.isupper():
        if is_iata_token(explicit_token):
            trace["selected_iata"] = explicit_token
            trace["resolution_basis"] = "explicit_iata_token"
            return explicit_token, trace

    # 3) fallback to resolver (external module)
    try:
        for candidate in candidates:
            resolved, resolver_trace = resolve_location_with_trace(candidate)
            sanitized = _sanitize_iata_code(resolved)
            if sanitized:
                trace["selected_iata"] = sanitized
                trace["resolver_trace"] = resolver_trace
                trace["resolution_basis"] = resolver_trace.get("match_basis") or "resolver"
                return sanitized, trace
    except Exception:
        logger.debug("resolve_location failed", exc_info=True)

    return None, trace


def _resolve_city_to_iata(city_text: str) -> Optional[str]:
    resolved, _trace = _resolve_city_to_iata_with_trace(city_text)
    return resolved


def _sanitize_iata_code(value: Optional[str]) -> Optional[str]:
    """
    Convert any resolver output into a strict valid 3-letter IATA code.
    Returns None if the value cannot be validated.
    """
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    if len(raw) == 3 and raw.isalpha() and is_iata_token(raw.upper()):
        return raw.upper()

    resolved = resolve_location(raw)
    if resolved and len(resolved) == 3 and is_iata_token(resolved):
        return resolved

    return None


def _iata_city_label(iata_code: Optional[str]) -> Dict[str, Optional[str]]:
    normalized = _sanitize_iata_code(iata_code)
    if not normalized:
        return {"iata": None, "city": None, "label": None}
    city = city_for_iata(normalized)
    label = label_for_iata(normalized) or normalized
    return {"iata": normalized, "city": city, "label": label}


def _infer_route_pair_from_query(user_query: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Deterministically infer origin/destination from free-form text when route regex misses.
    Priority:
      1) explicit IATA token pair (e.g., "DEL BOM")
      2) resolver-backed city phrase pair in text order (generic, no planner-local aliases)
    """
    trace: Dict[str, Any] = {
        "source": None,
        "iata_candidates": [],
        "city_candidates": [],
    }
    if not user_query:
        return None, None, trace

    # 1) explicit IATA tokens (uppercase only to avoid treating normal words like "via"/"new" as airport codes)
    iata_tokens: List[str] = []
    for tok in re.findall(r"\b([A-Z]{3})\b", user_query):
        if is_iata_token(tok) and tok not in iata_tokens:
            iata_tokens.append(tok)
    trace["iata_candidates"] = iata_tokens
    if len(iata_tokens) >= 2:
        trace["source"] = "iata_pair"
        return iata_tokens[0], iata_tokens[1], trace

    q_lower = user_query.lower()
    # If query already carries explicit "from ... to ..." phrasing and that path
    # failed earlier, do not invent a route from unrelated words.
    if re.search(r"\bfrom\b.+\bto\b", q_lower):
        trace["source"] = "skipped_unresolved_from_to"
        return None, None, trace

    # 2) resolver-backed city phrase scan in appearance order.
    noise_words = {
        "flight", "flights", "from", "to", "on", "at", "for", "via", "through",
        "stopover", "stop", "in", "return", "returning", "leaving", "departing",
        "coming", "back", "after", "before", "trip", "business", "holiday",
        "urgent", "flexible", "book", "booking", "ticket", "tickets", "find",
        "cheapest", "cheap", "and", "with", "under", "tomorrow", "today", "next",
        "this", "day", "days", "week", "weeks", "plan",
    }
    words = [(m.start(), m.group(0), m.group(0).lower()) for m in re.finditer(r"[A-Za-z]{2,}", user_query)]
    meaningful_tokens = [w for _, _, w in words if w not in noise_words and len(w) > 1]
    if len(meaningful_tokens) > 4:
        trace["source"] = "skipped_ambiguous_phrase"
        return None, None, trace

    best_hits_by_pos: Dict[int, Dict[str, Any]] = {}

    max_ngram = min(4, len(words))
    for size in range(max_ngram, 0, -1):
        for i in range(0, len(words) - size + 1):
            segment = words[i:i + size]
            lowers = [w[2] for w in segment]
            if any(tok in noise_words for tok in lowers):
                continue
            if size == 1 and len(lowers[0]) <= 3:
                continue
            if size > 1 and max(len(tok) for tok in lowers) <= 3:
                continue
            phrase = " ".join(w[1] for w in segment)
            resolved_raw, resolver_trace = resolve_location_with_trace(phrase)
            resolved = _sanitize_iata_code(resolved_raw)
            if not resolved:
                continue
            pos = segment[0][0]
            existing = best_hits_by_pos.get(pos)
            if existing is None or size > existing["size"]:
                best_hits_by_pos[pos] = {
                    "pos": pos,
                    "city": phrase,
                    "code": resolved,
                    "size": size,
                    "resolution_basis": resolver_trace.get("match_basis"),
                    "is_fuzzy": resolver_trace.get("is_fuzzy"),
                    "confidence": resolver_trace.get("confidence"),
                    "runner_up_confidence": resolver_trace.get("runner_up_confidence"),
                }

    ordered_hits = [best_hits_by_pos[pos] for pos in sorted(best_hits_by_pos.keys())]
    trace["city_candidates"] = [
        {
            "city": h["city"],
            "code": h["code"],
            "pos": h["pos"],
            "resolution_basis": h.get("resolution_basis"),
            "is_fuzzy": bool(h.get("is_fuzzy")),
            "confidence": h.get("confidence"),
            "runner_up_confidence": h.get("runner_up_confidence"),
        }
        for h in ordered_hits
    ]

    ordered_codes: List[str] = []
    for hit in ordered_hits:
        code = hit["code"]
        if code not in ordered_codes:
            ordered_codes.append(code)

    if len(ordered_codes) >= 2:
        trace["source"] = "resolver_phrase_pair"
        return ordered_codes[0], ordered_codes[1], trace

    return None, None, trace


def _build_flight_search_profile(intent: "ParsedIntent", normalization_debug: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact, bounded search-breadth profile for weak-route/round-trip/deep-search paths.
    Keeps strong-route defaults unchanged.
    """
    final = (normalization_debug or {}).get("final") or {}
    route_inference = (normalization_debug or {}).get("route_inference") or {}
    origin_basis = str(final.get("origin_resolution_basis") or "").lower()
    destination_basis = str(final.get("destination_resolution_basis") or "").lower()
    weak_route_confidence = bool(
        final.get("origin_resolution_is_fuzzy")
        or final.get("destination_resolution_is_fuzzy")
        or "fuzzy" in origin_basis
        or "fuzzy" in destination_basis
        or route_inference.get("source") in {"resolver_phrase_pair", "skipped_ambiguous_phrase"}
    )

    breadth_bump = 0
    if weak_route_confidence:
        breadth_bump += FLIGHT_SEARCH_WEAK_ROUTE_BONUS
    if bool(intent.return_date):
        breadth_bump += FLIGHT_SEARCH_ROUND_TRIP_BONUS
    if bool(intent.deep_search):
        breadth_bump += FLIGHT_SEARCH_DEEP_SEARCH_BONUS

    max_results = min(FLIGHT_SEARCH_MAX_RESULTS_CAP, FLIGHT_SEARCH_BASE_RESULTS + breadth_bump)
    return {
        "max_results": max_results,
        "breadth_bump": max_results - FLIGHT_SEARCH_BASE_RESULTS,
        "deep_search": bool(intent.deep_search),
        "weak_route_confidence": weak_route_confidence,
        "is_round_trip": bool(intent.return_date),
    }

# ----------------------------------------------------------------------
# LLM circuit breaker with auto-recovery
# ----------------------------------------------------------------------
_llm_failures = 0
_llm_failure_lock = asyncio.Lock()
LLM_FAILURE_THRESHOLD = max(1, get_env_int("LLM_FAILURE_THRESHOLD", 5))
LLM_CIRCUIT_OPEN = False
LLM_CIRCUIT_RESET_TIMEOUT = max(10, get_env_int("LLM_CIRCUIT_RESET_TIMEOUT", 120))  # seconds
LLM_FAILURE_IDLE_RESET_TIMEOUT = max(30.0, get_env_float("LLM_FAILURE_IDLE_RESET_TIMEOUT", 180.0))
_llm_circuit_reset_time: Optional[float] = None
_llm_last_failure_at: Optional[float] = None
_llm_last_failure_stage: Optional[str] = None
_llm_last_failure_reason: Optional[str] = None
_NON_COUNTING_FAILURE_REASONS = {"circuit_open", "cancelled", "stream_no_visible_tokens"}


def _is_ollama_only_mode(llm_mode: Optional[str], effective_mode: Optional[str]) -> bool:
    mode = (effective_mode or llm_mode or "").strip().lower()
    return mode == LLM_MODE_OLLAMA_ONLY


def _counter_reason_from_router_reason(reason: Optional[str]) -> str:
    normalized = (reason or "").strip().lower()
    if normalized in {"timeout", "stream_timeout"}:
        return "upstream_timeout"
    if normalized in _NON_COUNTING_FAILURE_REASONS:
        return normalized
    return "upstream_unavailable"


async def check_llm_circuit(*, llm_mode: Optional[str] = None, effective_mode: Optional[str] = None) -> bool:
    """Return True if circuit is open (skip LLM). Handles auto-recovery."""
    global _llm_failures, LLM_CIRCUIT_OPEN, _llm_circuit_reset_time
    global _llm_last_failure_at, _llm_last_failure_stage, _llm_last_failure_reason
    mode_hint = (effective_mode or llm_mode or "").strip().lower()
    if not mode_hint:
        try:
            resolved_mode, _ = await get_llm_mode_and_priority()
            mode_hint = (resolved_mode or "").strip().lower()
        except Exception:
            try:
                mode_hint = (get_llm_mode_default() or "").strip().lower()
            except Exception:
                mode_hint = ""
    if _is_ollama_only_mode(mode_hint, mode_hint):
        # In strict ollama_only mode, rely on backend-local circuiting in ollama_client.
        # Reset any stale planner-global state so logs/metrics don't look sticky across mode changes.
        async with _llm_failure_lock:
            if LLM_CIRCUIT_OPEN or _llm_failures:
                logger.info(
                    "LLM planner circuit state reset in ollama_only mode",
                    extra={
                        "failure_scope": "planner_process_consecutive",
                        "previous_failure_count": _llm_failures,
                        "was_open": LLM_CIRCUIT_OPEN,
                    },
                )
            LLM_CIRCUIT_OPEN = False
            _llm_failures = 0
            _llm_circuit_reset_time = None
            _llm_last_failure_at = None
            _llm_last_failure_stage = None
            _llm_last_failure_reason = None
        return False
    async with _llm_failure_lock:
        now = time.monotonic()
        # Auto-recover if timeout elapsed
        if LLM_CIRCUIT_OPEN and _llm_circuit_reset_time and now > _llm_circuit_reset_time:
            logger.info(
                "LLM circuit breaker reset after timeout",
                extra={
                    "failure_scope": "planner_process_consecutive",
                    "reset_timeout_sec": LLM_CIRCUIT_RESET_TIMEOUT,
                },
            )
            LLM_CIRCUIT_OPEN = False
            _llm_failures = 0
            _llm_circuit_reset_time = None
            _llm_last_failure_at = None
            _llm_last_failure_stage = None
            _llm_last_failure_reason = None

        if _llm_failures >= LLM_FAILURE_THRESHOLD:
            if not LLM_CIRCUIT_OPEN:
                logger.warning(
                    "LLM circuit breaker OPEN",
                    extra={
                        "failure_scope": "planner_process_consecutive",
                        "failure_count": _llm_failures,
                        "threshold": LLM_FAILURE_THRESHOLD,
                        "reset_timeout_sec": LLM_CIRCUIT_RESET_TIMEOUT,
                        "last_failure_stage": _llm_last_failure_stage,
                        "last_failure_reason": _llm_last_failure_reason,
                    },
                )
                LLM_CIRCUIT_OPEN = True
                _llm_circuit_reset_time = now + LLM_CIRCUIT_RESET_TIMEOUT
        return LLM_CIRCUIT_OPEN

async def record_llm_success():
    """Reset failure count on success."""
    global _llm_failures, LLM_CIRCUIT_OPEN, _llm_circuit_reset_time
    global _llm_last_failure_at, _llm_last_failure_stage, _llm_last_failure_reason
    async with _llm_failure_lock:
        _llm_failures = 0
        LLM_CIRCUIT_OPEN = False
        _llm_circuit_reset_time = None
        _llm_last_failure_at = None
        _llm_last_failure_stage = None
        _llm_last_failure_reason = None

async def record_llm_failure(
    *,
    stage: str = "unknown",
    reason: Optional[str] = None,
    llm_mode: Optional[str] = None,
    effective_mode: Optional[str] = None,
    attempt_count: Optional[int] = None,
    backend: Optional[str] = None,
):
    """Increment process-wide consecutive failure count."""
    global _llm_failures, _llm_last_failure_at, _llm_last_failure_stage, _llm_last_failure_reason
    resolved_llm_mode = (llm_mode or "").strip().lower() or None
    resolved_effective_mode = (effective_mode or "").strip().lower() or None
    if not resolved_llm_mode and not resolved_effective_mode:
        try:
            resolved_mode, _ = await get_llm_mode_and_priority()
            resolved_llm_mode = (resolved_mode or "").strip().lower() or None
            resolved_effective_mode = resolved_effective_mode or resolved_llm_mode
        except Exception:
            try:
                fallback_mode = (get_llm_mode_default() or "").strip().lower()
                if fallback_mode:
                    resolved_llm_mode = fallback_mode
                    resolved_effective_mode = resolved_effective_mode or fallback_mode
            except Exception:
                pass
    async with _llm_failure_lock:
        normalized_reason = (reason or "").strip().lower()
        counted_toward_circuit = normalized_reason not in _NON_COUNTING_FAILURE_REASONS
        strict_ollama_only = _is_ollama_only_mode(resolved_llm_mode, resolved_effective_mode)
        if counted_toward_circuit and strict_ollama_only:
            counted_toward_circuit = False
        now = time.monotonic()
        if counted_toward_circuit:
            if (
                _llm_last_failure_at is not None
                and (now - _llm_last_failure_at) > LLM_FAILURE_IDLE_RESET_TIMEOUT
                and _llm_failures > 0
            ):
                logger.info(
                    "LLM failure counter reset after idle interval",
                    extra={
                        "failure_scope": "planner_process_consecutive",
                        "idle_reset_timeout_sec": LLM_FAILURE_IDLE_RESET_TIMEOUT,
                        "previous_failure_count": _llm_failures,
                    },
                )
                _llm_failures = 0
            _llm_failures += 1
        elif strict_ollama_only and _llm_failures:
            # Keep planner-level counter neutral in strict local mode.
            _llm_failures = 0
        _llm_last_failure_at = now
        _llm_last_failure_stage = stage
        _llm_last_failure_reason = normalized_reason or reason
        if counted_toward_circuit:
            logger.warning(
                "LLM failure count: %s",
                _llm_failures,
                extra={
                    "failure_scope": "planner_process_consecutive",
                    "failure_stage": stage,
                    "failure_reason": normalized_reason or reason,
                    "llm_mode": resolved_llm_mode,
                    "effective_mode": resolved_effective_mode,
                    "attempt_count": attempt_count,
                    "backend": backend,
                    "counted_toward_circuit": True,
                },
            )
        else:
            logger.info(
                "LLM failure observed (not counted toward planner circuit)",
                extra={
                    "failure_scope": "planner_process_consecutive",
                    "failure_stage": stage,
                    "failure_reason": normalized_reason or reason,
                    "llm_mode": resolved_llm_mode,
                    "effective_mode": resolved_effective_mode,
                    "attempt_count": attempt_count,
                    "backend": backend,
                    "counted_toward_circuit": False,
                    "failure_count": _llm_failures,
                },
            )

# ----------------------------------------------------------------------
# Async-safe cache decorator with per-key locks and bounded TTL cache
# ----------------------------------------------------------------------
class CacheLock(asyncio.Lock):
    """Simple subclass so Lock objects support weak references."""
    pass

_cache_locks = weakref.WeakValueDictionary()

def _get_cache_lock(key):
    """Lazily create/return a per-key asyncio.Lock stored as a weak ref.

    Locks are kept alive by strong references in tasks that hold/wait on them.
    When no task holds or waits on a lock any more, GC removes it from the dict.
    This removes the need for manual clear() and avoids orphaning in-flight locks.
    """
    lock = _cache_locks.get(key)
    if lock is None:
        lock = CacheLock()
        _cache_locks[key] = lock
    return lock

def async_cache(ttl: int, maxsize: int = 1000):
    """
    Decorator that caches the result of an async function for `ttl` seconds.
    Uses per-key locks to prevent cache stampede and bounded cache to limit memory.
    """
    def decorator(func):
        # Use TTLCache if available, else simple dict (unbounded)
        if TTLCache:
            cache = TTLCache(maxsize=maxsize, ttl=ttl)
        else:
            cache = {}
            logger.warning("cachetools not installed, using unbounded cache (memory may grow)")

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            key = (func.__name__, args, frozenset(kwargs.items()))
            lock = _get_cache_lock(key)

            async with lock:
                if not DISABLE_CACHE:                    # skip entire cache block in dev
                    if TTLCache:
                        if key in cache:
                            logger.debug(f"Cache hit for {func.__name__}")
                            return cache[key]
                    else:
                        now = time.monotonic()
                        if key in cache:
                            result, timestamp = cache[key]
                            if now - timestamp < ttl:
                                logger.debug(f"Cache hit for {func.__name__}")
                                return result

                logger.debug(f"Cache miss for {func.__name__}")
                result = await func(*args, **kwargs)

                if not DISABLE_CACHE:                    # don't store if cache disabled
                    if TTLCache:
                        cache[key] = result
                    else:
                        cache[key] = (result, now)

                return result

        return wrapper
    return decorator

def create_cached_fetcher(ttl: int, maxsize: int, fetch_func: Callable):
    """
    Create an async cached version of a fetch function.
    The fetch function must accept the same arguments each time.
    """
    @async_cache(ttl=ttl, maxsize=maxsize)
    async def cached(*args, **kwargs):
        return await fetch_func(*args, **kwargs)
    return cached

# ----------------------------------------------------------------------
# Shared module-level cached tool instances (system-wide cache)
# ----------------------------------------------------------------------
_shared_cached_search = create_cached_fetcher(
    ttl=CACHE_FLIGHT_TTL,
    maxsize=500,
    fetch_func=default_flight_tool
)

_shared_cached_weather = create_cached_fetcher(
    ttl=CACHE_WEATHER_TTL,
    maxsize=500,
    fetch_func=default_weather_tool
)

# ----------------------------------------------------------------------
# Retry helper for flight searches (with exponential backoff + jitter)
# ----------------------------------------------------------------------
async def _call_with_retries(fn: Callable, attempts: int = FLIGHT_RETRY_ATTEMPTS) -> Any:
    """
    Generic async retry with exponential backoff + jitter.
    Calls a zero-argument coroutine function `fn` repeatedly until success or max attempts.
    Raises the last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except AirlineAPIError as e:
            # Upstream advertised 'unavailable' — try retrying a couple of times
            last_exc = e
            logger.warning("AirlineAPIError on attempt %d/%d: %s", attempt, attempts, str(e))
        except asyncio.TimeoutError as e:
            last_exc = e
            logger.warning("Timeout on airline API attempt %d/%d", attempt, attempts)
        except Exception as e:
            # transient network or parser error — treat as retryable for first N attempts
            last_exc = e
            logger.exception("Unexpected exception during airline fetch attempt %d/%d", attempt, attempts)

        if attempt == attempts:
            break

        # exponential backoff with jitter
        backoff = min(FLIGHT_RETRY_BASE * (2 ** (attempt - 1)), FLIGHT_RETRY_MAX_BACKOFF)
        jitter = backoff * (random.uniform(-FLIGHT_RETRY_JITTER, FLIGHT_RETRY_JITTER))
        sleep_for = max(0.05, backoff + jitter)
        await asyncio.sleep(sleep_for)

    # all attempts failed — re-raise last exception (caller will convert to fallback)
    raise last_exc

# ----------------------------------------------------------------------
# Helper to normalize raw flight data into Flight objects
# ----------------------------------------------------------------------
def normalize_flights(raw_flights: List[Any], default_date: str) -> List['Flight']:
    normalized = []
    for f in raw_flights:

        # Case 1: dict
        if isinstance(f, dict):
            flight_data = dict(f)

        # Case 2: tool Flight or any object with __dict__
        elif hasattr(f, "__dict__"):
            flight_data = dict(vars(f))

        else:
            logger.warning(f"Skipping unknown flight type: {type(f)}")
            continue

        # Ensure date
        if 'date' not in flight_data or not flight_data.get('date'):
            flight_data['date'] = default_date

        try:
            normalized.append(Flight(**flight_data))
        except ValidationError as e:
            logger.warning(f"Skipping invalid flight after conversion: {e}")

    return normalized

# ----------------------------------------------------------------------
# Pydantic models for validation and structured output
# ----------------------------------------------------------------------
class Flight(BaseModel):
    """Validated flight data model."""
    airline: str
    flight_no: str
    departure_time: str
    arrival_time: str
    duration_min: int
    price_inr: Union[str, int]
    price_unavailable: bool = False
    stops: Union[str, int] = 0           # int from API; "N/A" when unknown
    layover_info: str = ""               # e.g. "1h 30m at BOM"
    baggage: str = "Check airline"       # Extracted from SerpAPI extensions
    booking_token: Optional[str] = None  # For booking handoff
    shareable_link: Optional[str] = None
    provider_link: Optional[str] = None
    partner_booking_link: Optional[str] = None
    booking_url: Optional[str] = None
    booking_request: Optional[Dict[str, Any]] = None
    booking_options: Optional[List[Dict[str, Any]]] = None
    carbon_emissions_g: Optional[int] = None  # CO2 in grams
    date: Optional[str] = None
    handoff_url: Optional[str] = None    # Resolved booking deep-link (set after ranking)
    # New fields for multi-leg/layover tracking (added by airline_api)
    layover_durations_min: Optional[List[int]] = Field(default_factory=list)
    layover_airports: Optional[List[str]] = Field(default_factory=list)

    @field_validator('price_inr', mode='before')
    @classmethod
    def validate_price(cls, v):
        if isinstance(v, int):
            return f"₹{v:,}"
        if isinstance(v, str):
            raw = v.strip()
            if not raw:
                return "Price unavailable"
            lowered = raw.lower()
            if "unavailable" in lowered or lowered in {"n/a", "na", "unknown"}:
                return "Price unavailable"
            if not raw.startswith('₹'):
                try:
                    price_int = int(str(raw).replace(',', '').replace('₹', '').strip())
                    return f"₹{price_int:,}"
                except Exception:
                    return "Price unavailable"
            return raw
        if v is None:
            return "Price unavailable"
        return v

    @field_validator('departure_time', 'arrival_time', mode='before')
    @classmethod
    def validate_time_format(cls, v):
        if isinstance(v, str):
            match = re.search(r'(\d{1,2}):(\d{2})', v)
            if match:
                hour, minute = match.groups()
                return f"{int(hour):02d}:{minute}"
        return "00:00"

    @field_validator('duration_min', mode='before')
    @classmethod
    def validate_duration(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                numbers = re.findall(r'\d+', v)
                if numbers:
                    return int(numbers[0])
            except:
                pass
        return 999

class PlanResult(BaseModel):
    """Structured output of the planning process."""
    llm_response: Optional[str]  # None when skip_llm=True
    best_flight: Dict[str, Any]
    weather: Dict[str, Any]
    search_date: str
    warnings: Optional[List[str]] = None
    debug_info: Optional[Dict[str, Any]] = None  # Internal metrics + extra data for streaming
    return_trip: Optional['PlanResult'] = None
    fallback_note: str = ""
    # New fields for weather presence tracking
    weather_present: bool = True
    weather_reason: Optional[str] = None
    flight_counts: Optional[Dict[str, int]] = None
    stopover_filter: Optional[Dict[str, Any]] = None
    result_status: str = "success"
    degradation: Optional[Dict[str, Any]] = None
    booking_handoff: Optional[Dict[str, Any]] = None
    top_flights: Optional[List[Dict[str, Any]]] = None

class MultiCityResult(BaseModel):
    """Structured output for multi-city trips."""
    multicity: bool = True
    legs: List[PlanResult]

PlanResult.model_rebuild()

# ----------------------------------------------------------------------
# Route extraction from natural language (pure regex, no IATA logic)
# ----------------------------------------------------------------------
def extract_stopover(query_text: str) -> Dict[str, Optional[str]]:
    """
    Robust extraction for 'via' stopover phrases.
    Returns dict with keys: origin_text, destination_text, via_text
    Accepts multiword city names and many phrasings.
    """
    q = query_text.strip()
    tail_guard = (
        r'(?=\s+(?:via|through|stopover(?:\s+in)?|connecting\s+(?:via|through)|stop\s+in|'
        r'on|for|at|tomorrow|today|next|this|return(?:ing)?|coming\s+back|'
        r'leaving|departing|after|before|under|with|by)\b|$)'
    )

    def _clean_fragment(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        cleaned = re.sub(r"\s+", " ", text).strip(" ,.-")
        # Strip directive/boilerplate prefixes so route fragments remain city-focused.
        for _ in range(3):
            next_cleaned = cleaned
            next_cleaned = re.sub(
                r"^(?:please\s+|kindly\s+)?(?:find|search|show|get|book)(?:\s+me)?\s+",
                "",
                next_cleaned,
                flags=re.IGNORECASE,
            )
            next_cleaned = re.sub(
                r"^(?:a|an|the|flight|flights|trip|from|to|round[\s-]*trip|one[\s-]*way|return(?:\s+flight)?|multi[\s-]*city)\s+",
                "",
                next_cleaned,
                flags=re.IGNORECASE,
            )
            if next_cleaned == cleaned:
                break
            cleaned = next_cleaned
        cleaned = re.sub(r"\s+(?:on|for|at)\s+\d{4}-\d{2}-\d{2}\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned or None

    # from <A> to <B> [via <C>]
    m = re.search(
        rf'\bfrom\s+([A-Za-z][A-Za-z\s-]{{1,80}}?)\s+to\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
        q,
        re.IGNORECASE,
    )
    if m:
        origin = _clean_fragment(m.group(1))
        destination = _clean_fragment(m.group(2))
        via_match = re.search(
            rf'\b(?:via|through|stopover(?:\s+in)?|connecting\s+(?:via|through)|stop\s+in)\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
            q,
            re.IGNORECASE,
        )
        via = _clean_fragment(via_match.group(1)) if via_match else None
        return {"origin_text": origin, "destination_text": destination, "via_text": via}

    # <A> to <B> [via <C>]
    m = re.search(
        rf'\b([A-Za-z][A-Za-z\s-]{{1,80}}?)\s+to\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
        q,
        re.IGNORECASE,
    )
    if m:
        origin = _clean_fragment(m.group(1))
        destination = _clean_fragment(m.group(2))
        via_match = re.search(
            rf'\b(?:via|through|stopover(?:\s+in)?|connecting\s+(?:via|through)|stop\s+in)\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
            q,
            re.IGNORECASE,
        )
        via = _clean_fragment(via_match.group(1)) if via_match else None
        return {"origin_text": origin, "destination_text": destination, "via_text": via}

    # via <C> alone
    m = re.search(
        rf'\b(?:via|through|stopover(?:\s+in)?|connecting\s+(?:via|through)|stop\s+in)\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
        q,
        re.IGNORECASE,
    )
    if m:
        return {"origin_text": None, "destination_text": None, "via_text": _clean_fragment(m.group(1))}

    return {"origin_text": None, "destination_text": None, "via_text": None}

def normalize_trip(user_query: str, include_trace: bool = False) -> Dict[str, Any]:
    """
    Build a canonical trip object from raw user_query using regex and centralised resolver.
    Returns dict with keys: origin_iata, destination_iata, via_iata.
    """
    parts = extract_stopover(user_query)

    origin_iata = None
    dest_iata = None
    via_iata = None

    route_trace: Dict[str, Any] = {
        "raw_fragments": parts,
        "origin_resolution": None,
        "destination_resolution": None,
        "via_resolution": None,
    }

    if parts["origin_text"]:
        origin_iata, origin_trace = _resolve_city_to_iata_with_trace(parts["origin_text"])
        route_trace["origin_resolution"] = origin_trace
        logger.debug(f"normalize_trip: origin_text='{parts['origin_text']}' -> origin_iata={origin_iata}")

    if parts["destination_text"]:
        dest_iata, destination_trace = _resolve_city_to_iata_with_trace(parts["destination_text"])
        route_trace["destination_resolution"] = destination_trace
        logger.debug(f"normalize_trip: dest_text='{parts['destination_text']}' -> dest_iata={dest_iata}")

    if parts["via_text"]:
        via_iata, via_trace = _resolve_city_to_iata_with_trace(parts["via_text"])
        route_trace["via_resolution"] = via_trace
        logger.debug(f"normalize_trip: via_text='{parts['via_text']}' -> via_iata={via_iata}")

    payload: Dict[str, Any] = {
        "origin_iata": origin_iata,
        "destination_iata": dest_iata,
        "via_iata": via_iata,
    }
    if include_trace:
        payload["route_trace"] = route_trace
    return payload

# ----------------------------------------------------------------------
# Preference extraction constants
# ----------------------------------------------------------------------
TIME_WINDOWS = {
    "morning": ("04:00", "11:59"),
    "afternoon": ("12:00", "17:59"),
    "evening": ("18:00", "23:59"),
    "night": ("00:00", "03:59")
}
AIRLINES = ["indigo", "air india", "vistara", "goair", "spicejet", "akasa", "airasia"]

# ----------------------------------------------------------------------
# Intent parsing (pure logic, no IO)
# ----------------------------------------------------------------------
class ParsedIntent(BaseModel):
    """All extracted information from the user query."""
    origin_iata: Optional[str] = None
    destination_iata: Optional[str] = None
    date: Optional[str] = None
    return_date: Optional[str] = None
    time_pref: Optional[str] = None
    price_limit: Optional[int] = None
    wants_direct: bool = False
    preferred_airlines: List[str] = Field(default_factory=list)
    layover_limit_minutes: Optional[int] = None
    baggage_pref: Optional[str] = None
    trip_duration_days: Optional[int] = None
    stopover_city: Optional[str] = None
    flight_pref: str = "default"
    wants_eco: bool = False           # True when user asks for green/eco/low-carbon flights
    trip_type: Optional[str] = None   # No default – fallback applied in business logic
    deep_search: bool = False         # NEW: user wants absolute cheapest, exhaustive search
    date_parse_trace: Optional[Dict[str, Any]] = None
    route_parse_trace: Optional[Dict[str, Any]] = None

def parse_intent(user_query: str) -> ParsedIntent:
    """Extract all structured data from the natural language query."""
    intent = ParsedIntent()

    # --- First, use the robust normalize_trip to get IATA codes + route trace ---
    trip = normalize_trip(user_query, include_trace=True)
    if trip["origin_iata"]:
        intent.origin_iata = trip["origin_iata"]
    if trip["destination_iata"]:
        intent.destination_iata = trip["destination_iata"]
    intent.route_parse_trace = trip.get("route_trace")

    # Deterministic fallback for compact route formats like "DEL BOM ..." or "Delhi Mumbai ..."
    if not intent.origin_iata or not intent.destination_iata:
        inferred_origin, inferred_dest, _ = _infer_route_pair_from_query(user_query)
        if not intent.origin_iata and inferred_origin:
            intent.origin_iata = inferred_origin
        if not intent.destination_iata and inferred_dest:
            intent.destination_iata = inferred_dest

    # Always sanitize to strict valid IATA codes.
    intent.origin_iata = _sanitize_iata_code(intent.origin_iata)
    intent.destination_iata = _sanitize_iata_code(intent.destination_iata)
    # via_iata is not stored directly in intent; we only use city name for stopover if needed
    # (we could keep via_city for potential later use, but for now we ignore)

    q = user_query.lower()
    WORD_TO_NUM = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
        'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
        'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'fortnight': 14,
    }

    def _replace_word_numbers(text: str) -> str:
        normalized = text
        for word, num in WORD_TO_NUM.items():
            normalized = re.sub(rf'\b{word}\b', str(num), normalized)
        return normalized

    q_num = _replace_word_numbers(q)

    def _has_explicit_calendar_date(text: str) -> bool:
        """Detect explicit calendar-like dates while excluding duration phrases."""
        month_names = (
            "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            "jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        )
        return bool(
            re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
            or re.search(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month_names})(?:\s+\d{{4}})?\b", text)
            or re.search(rf"\b(?:{month_names})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b", text)
            or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})\b", text)
        )

    def _has_ambiguous_calendar_without_year(text: str) -> bool:
        """Detect month/day style dates without an explicit year."""
        month_names = (
            "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            "jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        )
        return bool(
            re.search(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month_names})\b(?!\s+\d{{4}}\b)", text)
            or re.search(rf"\b(?:{month_names})\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!,\s*\d{{4}}\b)", text)
            or re.search(r"\b\d{1,2}[/-]\d{1,2}\b(?![/-]\d{2,4}\b)", text)
        )

    today = datetime.now().date()
    _relative_date_set = False   # track if we set date via relative rules
    date_parse_trace: Dict[str, Any] = {
        "source": "none",
        "raw_match": None,
        "candidate_source": None,
        "candidate_date": None,
        "year_inferred": False,
        "discard_reason": None,
    }

    # --- "starting DATE" match (e.g., "starting March 20") ---
    starting_match = re.search(
        r'\bstarting\s+(?:on\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+\w+|\w+\s+\d{1,2}(?:,\s*\d{4})?|\d{4}-\d{2}-\d{2})',
        q, re.IGNORECASE
    )
    if starting_match and not _relative_date_set:
        try:
            parsed_start = dateutil.parser.parse(starting_match.group(1), dayfirst=True)
            # Keep explicit "starting <date>" literal date even if it is in the past.
            intent.date = parsed_start.strftime("%Y-%m-%d")
            date_parse_trace["source"] = "starting_clause"
            date_parse_trace["raw_match"] = starting_match.group(1)
            _relative_date_set = True
        except Exception:
            pass

    # --- Explicit ISO departure date in query should be preserved exactly ---
    if not _relative_date_set:
        iso_departure_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
        if iso_departure_match:
            intent.date = iso_departure_match.group(1)
            date_parse_trace["source"] = "explicit_iso"
            date_parse_trace["raw_match"] = iso_departure_match.group(1)
            _relative_date_set = True

    # --- Enhanced relative date parsing (with word numbers + weeks) ---
    if not _relative_date_set:
        q_rel = q_num

        # Explicit relative anchors first (avoid fuzzy parsing on trailing duration numbers).
        if re.search(r"\bday after tomorrow\b", q_rel):
            intent.date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
            date_parse_trace["source"] = "relative_day_after_tomorrow"
            _relative_date_set = True
        elif re.search(r"\btomorrow\b", q_rel):
            intent.date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
            date_parse_trace["source"] = "relative_tomorrow"
            _relative_date_set = True

        # "in 3 days", "3 days later", "3 days after today", "3 days from now"
        if not _relative_date_set:
            rel_days = re.search(
                r"\bin\s+(\d+)\s+days?\b|(\d+)\s+days?\s+later\b|(\d+)\s+days?\s+(?:after|from)\s+today\b|(\d+)\s+days?\s+from\s+now\b",
                q_rel,
            )
            if rel_days:
                n = int(next(g for g in rel_days.groups() if g))
                intent.date = (today + timedelta(days=n)).strftime("%Y-%m-%d")
                date_parse_trace["source"] = "relative_days_offset"
                date_parse_trace["raw_match"] = rel_days.group(0)
                _relative_date_set = True

        # "in 2 weeks", "2 weeks from now / after today"
        if not _relative_date_set:
            rel_weeks = re.search(
                r"\bin\s+(\d+)\s+weeks?\b|after\s+(\d+)\s+weeks?\b|(\d+)\s+weeks?\s+(?:from\s+(?:now|today)|after\s+(?:today|now))\b",
                q_rel,
            )
            if rel_weeks:
                n = int(next(g for g in rel_weeks.groups() if g))
                intent.date = (today + timedelta(weeks=n)).strftime("%Y-%m-%d")
                date_parse_trace["source"] = "relative_weeks_offset"
                date_parse_trace["raw_match"] = rel_weeks.group(0)
                _relative_date_set = True

        # Keep broad "today" last so specific offsets like "14 days after today" win.
        if not _relative_date_set and re.search(r"\btoday\b", q_rel):
            intent.date = today.strftime("%Y-%m-%d")
            date_parse_trace["source"] = "relative_today"
            _relative_date_set = True

    # --- If relative date not set, fall back to other date parsers ---
    if not _relative_date_set:
        # Strip price expressions so "₹3000" / "under 5000 INR" don't pollute dateutil
        q_clean = re.sub(
            r'under\s*[₹$€£]?\s*\d+|[₹$€£]\s*\d+|\d[\d,]*\s*(?:rupees?|inr|usd|eur)\b',
            '', q, flags=re.IGNORECASE
        )

        parsed_date = None
        parsed_source = None
        if HAS_DATEPARSER:
            settings = {'PREFER_DATES_FROM': 'future', 'DATE_ORDER': 'DMY'}
            parsed_date = dateparser.parse(q_clean, settings=settings)
            if parsed_date:
                parsed_source = "dateparser"
        else:
            # Fallback regex
            date_match = re.search(r'\b(\d{1,2})(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b', q)
            if date_match:
                day, _, month = date_match.groups()
                year = today.year
                try:
                    parsed_date = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
                    parsed_source = "fallback_month_day_regex"
                except:
                    pass

        # Guard against fuzzy parser anchoring to a bare duration number (e.g., "returning 3 days later" -> Jan 3).
        relative_trip_tokens = re.search(r"\b(return(?:ing)?|later|days?|weeks?|tomorrow|today|from now|after today|in \d+)\b", q_clean)
        if not parsed_date and not (relative_trip_tokens and not _has_explicit_calendar_date(q_clean)):
            try:
                parsed_date = dateutil.parser.parse(
                    q_clean,
                    fuzzy=True, dayfirst=True,
                    default=datetime.now().replace(month=1, day=1)
                )
                parsed_source = "dateutil_fuzzy"
            except:
                pass

        if parsed_date:
            date_parse_trace["candidate_source"] = parsed_source
            date_parse_trace["candidate_date"] = parsed_date.strftime("%Y-%m-%d")
            # Sanity check: reject absurd years and any past date
            if parsed_date.year > today.year + 2 or parsed_date.year < 2000:
                parsed_date = None
                date_parse_trace["discard_reason"] = "year_out_of_range"
            elif parsed_date.date() < today:
                explicit_calendar = _has_explicit_calendar_date(q)
                ambiguous_without_year = _has_ambiguous_calendar_without_year(q)
                if explicit_calendar and ambiguous_without_year:
                    # Conservative inference: only infer next year for explicit month/day style dates
                    # that did not specify a year.
                    try:
                        bumped = parsed_date.replace(year=parsed_date.year + 1)
                    except ValueError:
                        bumped = None
                    if bumped and bumped.date() >= today:
                        intent.date = bumped.strftime("%Y-%m-%d")
                        date_parse_trace["source"] = "explicit_ambiguous_year_inferred"
                        date_parse_trace["year_inferred"] = True
                    else:
                        intent.date = parsed_date.strftime("%Y-%m-%d")
                        date_parse_trace["source"] = "explicit_calendar_preserved_past"
                else:
                    # Preserve explicit dated user text exactly; do not silently rewrite.
                    if explicit_calendar:
                        intent.date = parsed_date.strftime("%Y-%m-%d")
                        date_parse_trace["source"] = "explicit_calendar_preserved_past"
                    else:
                        # Fuzzy implicit date parsed in the past: drop it instead of silently bumping year.
                        date_parse_trace["discard_reason"] = "implicit_past_candidate"
                        date_parse_trace["source"] = "fuzzy_past_discarded"
            else:
                intent.date = parsed_date.strftime("%Y-%m-%d")
                date_parse_trace["source"] = parsed_source or "parser_candidate"

    if intent.date and date_parse_trace["source"] == "none":
        date_parse_trace["source"] = "resolved_without_trace"
    intent.date_parse_trace = date_parse_trace

    # --- Time preference ---
    for key in TIME_WINDOWS:
        if key in q:
            intent.time_pref = key
            break

    # --- Price limit ---
    price_pattern = re.compile(
        r"\b(?:under|below|within|less\s+than|up\s+to)\s*"
        r"(?:₹\s*|rs\.?\s*|inr\s*|rupees?\s*)?"
        r"(?P<amount>\d{1,7})"
        r"(?:\s*(?:₹|rs\.?|inr|rupees?))?\b",
        re.IGNORECASE,
    )
    for price_match in price_pattern.finditer(q):
        trailing = q[price_match.end(): price_match.end() + 24]
        # Avoid treating duration constraints as budgets (for example, "layover under 2 hours").
        if re.match(r"\s*(?:hours?|hrs?|h|minutes?|mins?|days?|weeks?)\b", trailing):
            continue
        intent.price_limit = int(price_match.group("amount"))
        break

    # --- Direct flights ---
    intent.wants_direct = "direct" in q

    # --- Eco/green preference ---
    intent.wants_eco = any(kw in q for kw in ("eco", "green", "low carbon", "low-carbon", "sustainable", "environment"))

    # --- Airline preference ---
    intent.preferred_airlines = [a for a in AIRLINES if a in q]

    # --- Layover limit (explicit number) ---
    layover_match = re.search(r'layover.*?(\d{1,2})\s*hours?', q)
    if layover_match:
        intent.layover_limit_minutes = int(layover_match.group(1)) * 60

    # --- NEW: detect "short layover" or "quick connection" and set a default limit if not already set ---
    if not intent.layover_limit_minutes:
        if re.search(r'\b(short layover|quick connection)\b', q):
            intent.layover_limit_minutes = 180  # 3 hours

    # --- Baggage preference ---
    if "hand baggage" in q or "cabin only" in q:
        intent.baggage_pref = "hand"
    elif "check-in" in q:
        intent.baggage_pref = "checked"

    # --- Trip duration (for return) ---
    def _unit_to_days(count: int, unit: str) -> int:
        unit_l = (unit or "").lower()
        if unit_l.startswith("week"):
            return count * 7
        return count

    return_duration_patterns = (
        r"\b(?:return(?:ing)?|coming\s+back|come\s+back|back)\b[^.]{0,80}?\b(?:after|in)\s+(?P<count>\d+)\s+(?P<unit>day|days|night|nights|week|weeks)\b",
        r"\b(?:return(?:ing)?|coming\s+back|come\s+back|back)\b[^.]{0,80}?\b(?P<count>\d+)\s+(?P<unit>day|days|night|nights|week|weeks)\s+(?:later|after\s+that)\b",
    )
    for pattern in return_duration_patterns:
        match = re.search(pattern, q_num)
        if match:
            intent.trip_duration_days = _unit_to_days(int(match.group("count")), match.group("unit"))
            break

    if intent.trip_duration_days is None:
        duration_match = re.search(
            r"\b(?:for|trip\s+for|stay(?:ing)?\s+for)\s+(\d+)\s*(day|days|night|nights|week|weeks)\b"
            r"|\b(\d+)\s*(day|days|night|nights|week|weeks)\s+trip\b",
            q_num,
        )
        if duration_match:
            count_raw = duration_match.group(1) or duration_match.group(3)
            unit_raw = duration_match.group(2) or duration_match.group(4)
            if count_raw and unit_raw:
                intent.trip_duration_days = _unit_to_days(int(count_raw), unit_raw)

    # --- Return date explicit ---
    return_match = re.search(
        r'return(?:ing)?(?:\s+on)?\s+(\d{4}-\d{2}-\d{2}|\d{1,2}[\-/]\d{1,2}(?:[\-/]\d{2,4})?)',
        q,
    )
    if return_match:
        try:
            dt = dateutil.parser.parse(return_match.group(1), dayfirst=True)
            intent.return_date = dt.strftime("%Y-%m-%d")
        except:
            pass

    # --- Stopover city ---
    # Parse from original query to preserve multi-word city names (for example, "New Delhi", "Abu Dhabi").
    via_match = re.search(
        r'\b(?:via|through|stopover(?:\s+in)?|connecting\s+(?:via|through)|stop\s+in)\s+([A-Za-z][A-Za-z\s-]{1,80}?)(?=\s+(?:on|for|at|tomorrow|today|next|this|return(?:ing)?|coming\s+back|leaving|departing|after|before|under|with|by)\b|$)',
        user_query,
        flags=re.IGNORECASE,
    )
    if via_match:
        stopover_text = re.sub(r'\s+', ' ', via_match.group(1)).strip(" ,.-")
        stopover_text = re.sub(r'^(?:in)\s+', '', stopover_text, flags=re.IGNORECASE)
        if re.fullmatch(r"[A-Za-z]{3}", stopover_text) and is_iata_token(stopover_text.upper()):
            intent.stopover_city = stopover_text.upper()
        else:
            intent.stopover_city = stopover_text.title()

    # --- Flight preference ---
    if "cheapest" in q or "cheap" in q or "lowest price" in q or "budget" in q:
        intent.flight_pref = "cheapest"
    elif "shortest" in q or "fastest" in q or "least time" in q or "quickest" in q:
        intent.flight_pref = "shortest"
    elif "balanced" in q or ("price" in q and "duration" in q):
        intent.flight_pref = "balanced"
    elif intent.wants_eco:
        intent.flight_pref = "eco"

    # --- NEW: deep search (absolute cheapest) ---
    if any(x in q for x in ["absolute cheapest", "cheapest possible", "lowest price ever"]):
        intent.deep_search = True

    # --- Trip type ---
    if "flexible" in q or "any day" in q or "around" in q:
        intent.trip_type = "Flexible"
    elif "business" in q:
        intent.trip_type = "Business"
    elif "holiday" in q or "vacation" in q:
        intent.trip_type = "Holiday"
    elif "urgent" in q or "emergency" in q:
        intent.trip_type = "Urgent"

    return intent

# ----------------------------------------------------------------------
# Filtering and ranking (pure logic, using Flight objects)
# ----------------------------------------------------------------------
def price_to_int(price: Union[str, int]) -> int:
    if isinstance(price, int):
        return price
    try:
        return int(str(price).replace('₹', '').replace(',', '').strip())
    except:
        return 10**9

def normalize_flight_field(value: Any) -> str:
    """Convert flight field to a normalized string for matching."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return str(value).strip().lower()

def filter_flights(flights: List[Flight], intent: ParsedIntent) -> List[Flight]:
    """Apply all user filters to the flight list with tolerant matching."""
    filtered = []
    # Local warnings list (passed to LLM later)
    filter_warnings = []
    warned_unknown_price_budget = False
    warned_unknown_stops_direct = False
    for f in flights:
        reasons = []
        # Normalize fields
        stops = normalize_flight_field(getattr(f, "stops", ""))
        baggage = normalize_flight_field(getattr(f, "baggage", ""))
        airline = normalize_flight_field(getattr(f, "airline", ""))

        if intent.time_pref:
            start, end = TIME_WINDOWS[intent.time_pref]
            dep = f.departure_time[-5:]
            if not (start <= dep <= end):
                reasons.append("time")
        if intent.price_limit:
            price_unavailable = bool(getattr(f, "price_unavailable", False))
            if price_unavailable:
                if not warned_unknown_price_budget:
                    filter_warnings.append(
                        "Some flights had unavailable prices, so budget filtering could not be strictly enforced for those options."
                    )
                    warned_unknown_price_budget = True
            else:
                price = price_to_int(f.price_inr)
                if price > intent.price_limit:
                    reasons.append("price")
        if intent.wants_direct:
            # Tolerant direct detection
            stops_clean = stops.replace("stops", "").strip()
            direct = False
            # Unknown stops → assume it might be direct (benefit of doubt)
            if stops in ("n/a", "", "unknown"):
                direct = True
                if not warned_unknown_stops_direct:
                    filter_warnings.append("Stop data unavailable for recommended flight; directness cannot be confirmed.")
                    warned_unknown_stops_direct = True
            elif any(x in stops for x in ("non-stop", "nonstop", "direct")):
                direct = True
            else:
                try:
                    if int(stops_clean) == 0:
                        direct = True
                except:
                    pass
            if not direct:
                reasons.append("not direct")
        if intent.preferred_airlines:
            if not any(pref in airline for pref in intent.preferred_airlines):
                reasons.append("airline")
        # --- Layover limit filter (using true layover durations) ---
        if intent.layover_limit_minutes:
            layover_durs = f.layover_durations_min or []   # assume flight has this field
            if layover_durs and any(d > intent.layover_limit_minutes for d in layover_durs):
                reasons.append("layover")
        # --- Stopover filter: handled separately in main flow, not here ---
        if intent.baggage_pref:
            if intent.baggage_pref == "hand":
                if not any(x in baggage for x in (
                    "hand", "cabin", "carry", "7kg", "8kg", "10kg"
                )):
                    reasons.append("baggage")
            if intent.baggage_pref == "checked":
                if not any(x in baggage for x in (
                    "checked", "check", "hold", "1pc", "2pc", "15kg", "20kg"
                )):
                    reasons.append("baggage")

        if not reasons:
            filtered.append(f)
        else:
            logger.debug(f"Flight {f.flight_no} rejected: {reasons}")
    return filtered, filter_warnings

def get_weights(pref: str) -> Tuple[float, float, float]:
    """Return (price_weight, duration_weight, carbon_weight)."""
    if pref == "cheapest":
        return 0.8, 0.2, 0.0
    elif pref == "shortest":
        return 0.2, 0.8, 0.0
    elif pref == "balanced":
        return 0.5, 0.5, 0.0
    elif pref == "eco":
        return 0.3, 0.2, 0.5   # heavily weight low carbon
    return 0.6, 0.4, 0.0

def rank_flights(flights: List[Flight], intent: ParsedIntent) -> List[Flight]:
    """
    Rank flights by normalized price, duration, and (when eco mode) carbon emissions.
    Normalization values computed once to avoid O(n²).
    """
    if not flights:
        return []
    prices    = [price_to_int(f.price_inr) for f in flights]
    durations = [f.duration_min for f in flights]
    carbons   = [f.carbon_emissions_g for f in flights if f.carbon_emissions_g is not None]

    min_price, max_price = min(prices), max(prices)
    min_dur,   max_dur   = min(durations), max(durations)
    min_co2,   max_co2   = (min(carbons), max(carbons)) if carbons else (0, 1)

    wp, wd, wc = get_weights(intent.flight_pref)

    # If deep_search, we can increase price weight even further (but already 0.8 for cheapest)
    # This could be used to adjust search parameters earlier, but for ranking we keep as is.

    def score(f: Flight) -> float:
        price    = price_to_int(f.price_inr)
        duration = f.duration_min
        price_norm = (max_price - price)    / (max_price - min_price) if max_price > min_price else 1.0
        dur_norm   = (max_dur   - duration) / (max_dur   - min_dur)   if max_dur   > min_dur   else 1.0
        # Carbon norm: lower CO2 scores higher; flights without data score 0.5 (neutral)
        if wc > 0 and f.carbon_emissions_g is not None and max_co2 > min_co2:
            co2_norm = (max_co2 - f.carbon_emissions_g) / (max_co2 - min_co2)
        else:
            co2_norm = 0.5
        return price_norm * wp + dur_norm * wd + co2_norm * wc

    scored = [(f, score(f)) for f in flights]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scored]

# ----------------------------------------------------------------------
# LLM explanation generation with timeout and circuit breaker (non-streaming)
# ----------------------------------------------------------------------
def _enforce_narrative_consistency(
    llm_text: str,
    best_flight: Flight,
    weather: Dict[str, Any],
) -> str:
    """
    Post-process LLM text to keep narrative aligned with structured data.
    Targeted fixes only:
    1) prevent layover wording for non-stop flights
    2) prevent min/max temperature inversion wording
    """
    if not llm_text:
        return llm_text

    text = llm_text

    # 1) Non-stop flights should not be described as having layovers.
    if best_flight.stops == 0 or str(best_flight.stops).strip() == "0":
        text = re.sub(
            r"\bwith a layover\s+(?:less than|under|below)\s+\d+\s*(?:hours?|hrs?|h|minutes?|mins?)\b",
            "with no layover",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bhas a layover\s+(?:less than|under|below)\s+\d+\s*(?:hours?|hrs?|h|minutes?|mins?)\b",
            "has no layover",
            text,
            flags=re.IGNORECASE,
        )
        replacements = (
            (r"\bwith a layover\b", "with no layover"),
            (r"\bhas a layover\b", "has no layover"),
            (r"\blayover of\b", "max-layover preference of"),
            (r"\bconnecting time\b", "connection preference"),
        )
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 2) Keep weather low/high labels consistent with structured min/max.
    if isinstance(weather, dict):
        temp_min = weather.get("temp_min_c")
        temp_max = weather.get("temp_max_c")
        try:
            if temp_min is not None and temp_max is not None:
                temp_min_f = float(temp_min)
                temp_max_f = float(temp_max)
                if temp_max_f > temp_min_f:
                    # If text describes max as low/min, rewrite that numeric mention to min.
                    max_int = str(int(temp_max_f))
                    min_str = str(temp_min)
                    low_inversion_re = re.compile(
                        rf'(?i)\b(low|minimum|min)(?P<middle>(?:(?!(?:high|max)).){{0,30}}?)(?P<value>{re.escape(max_int)}(?:\.0+)?)'
                    )
                    text = low_inversion_re.sub(
                        lambda m: f"{m.group(1)}{m.group('middle')}{min_str}",
                        text,
                    )
        except Exception:
            # Never fail planner response due to sanitization.
            pass

    return text


def _ensure_route_grounding(
    llm_text: str,
    origin_iata: Optional[str],
    destination_iata: Optional[str],
) -> str:
    """
    Ensure final prose explicitly grounds the recommendation to canonical route labels.
    This keeps misspelled-query narratives from drifting into non-canonical city spellings.
    """
    text = (llm_text or "").strip()
    origin = _sanitize_iata_code(origin_iata)
    destination = _sanitize_iata_code(destination_iata)
    if not origin or not destination:
        return text

    origin_city = city_for_iata(origin) or origin
    destination_city = city_for_iata(destination) or destination
    text_lower = text.lower()

    def _mentions(label: str, iata: str) -> bool:
        token = (label or "").strip().lower()
        if token and token in text_lower:
            return True
        return bool(re.search(rf"\b{re.escape(iata.lower())}\b", text_lower))

    if _mentions(origin_city, origin) and _mentions(destination_city, destination):
        return text

    route_line = f"Route confirmation: {origin_city} ({origin}) to {destination_city} ({destination})."
    if not text:
        return route_line
    return f"{text}\n\n{route_line}"


def _safe_llm_error_message(error: Exception) -> str:
    if isinstance(error, AllBackendsFailed):
        status = error.as_dict() if hasattr(error, "as_dict") else {}
        failures = status.get("failures") or []
        mode = (status.get("mode") or "").strip().lower()
        effective_mode = (status.get("effective_mode") or "").strip().lower()
        strict_mode = effective_mode or mode
        single_backend_scope = (
            len(failures) <= 1
            and (
                mode in {LLM_MODE_OLLAMA_ONLY, LLM_MODE_CLOUD_ONLY}
                or effective_mode in {LLM_MODE_OLLAMA_ONLY, LLM_MODE_CLOUD_ONLY}
            )
        )
        if single_backend_scope:
            if strict_mode == LLM_MODE_OLLAMA_ONLY:
                return "Configured Ollama backend temporarily unavailable"
            if strict_mode == LLM_MODE_CLOUD_ONLY:
                return "Configured cloud backend temporarily unavailable"
            return "Configured LLM backend temporarily unavailable"
        return "LLM backends temporarily unavailable"

    message = str(error).lower()
    if "timed out" in message or "timeout" in message:
        return "LLM request timed out"
    if "no available keys for service" in message:
        return "Selected cloud provider is temporarily unavailable"
    return "LLM response unavailable"


def _is_single_backend_unavailable(error: AllBackendsFailed) -> bool:
    status = error.as_dict() if hasattr(error, "as_dict") else {}
    failures = status.get("failures") or []
    mode = (status.get("mode") or "").strip().lower()
    effective_mode = (status.get("effective_mode") or "").strip().lower()
    return (
        len(failures) <= 1
        and (
            mode in {LLM_MODE_OLLAMA_ONLY, LLM_MODE_CLOUD_ONLY}
            or effective_mode in {LLM_MODE_OLLAMA_ONLY, LLM_MODE_CLOUD_ONLY}
        )
    )


def _primary_router_failure_reason(status: Dict[str, Any]) -> str:
    failures = status.get("failures") or []
    if failures and isinstance(failures[0], dict):
        return _counter_reason_from_router_reason(str(failures[0].get("reason") or ""))
    return "upstream_unavailable"


def _truncate_for_prompt(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 18)].rstrip() + " ...[trimmed]"


def _backend_status_payload(error: Exception) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"message": _safe_llm_error_message(error)}
    if _planner_include_backend_status() and isinstance(error, AllBackendsFailed):
        payload["backend_status"] = error.as_dict()
    return payload


def _classify_tool_failure_reason(error: Any) -> str:
    text = str(error or "").lower()
    if "timed out" in text or "timeout" in text:
        return "upstream_timeout"
    if "no available keys" in text or "temporarily unavailable" in text or "all serpapi keys exhausted" in text:
        return "upstream_unavailable"
    return "provider_failure"


def _failure_domain_from_reason(reason: Optional[str]) -> str:
    r = (reason or "").strip().lower()
    if r in {"upstream_timeout", "upstream_unavailable", "provider_failure"}:
        return "upstream_provider"
    if r in {"invalid_route", "invalid_past_date", "invalid_date_order"}:
        return "request_validation"
    if r in {"no_flights"}:
        return "search_outcome"
    return "internal_backend"


def _llm_degradation_payload(
    *,
    reason: str,
    message: str,
    provider: Optional[str] = None,
    backend_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "component": "llm_explanation",
        "reason": reason,
        "message": message,
        "domain": _failure_domain_from_reason(reason),
    }
    if provider:
        payload["provider"] = provider
    if backend_status:
        payload["backend_status"] = backend_status
    return payload


def _explanation_degradation_note(reason: Optional[str], message: Optional[str]) -> str:
    safe_reason = (reason or "upstream_unavailable").strip().lower()
    safe_message = (
        (message or "").strip()
        or "LLM explanation degraded; structured flight and weather data remain available."
    )
    return f"LLM explanation degraded ({safe_reason}): {safe_message}"


async def generate_explanation(
    user_query: str,
    intent: ParsedIntent,
    best_flight: Flight,
    weather: Dict,
    all_flights: List[Flight],
    filters_applied: str,
    trip_description: str,
    warnings: Optional[List[str]] = None,
    price_insights_str: str = "",   # pre-formatted price intelligence sentence
    price_analysis_str: str = "",   # NEW: structured price trend info
    price_prediction_str: str = "", # NEW: price prediction advice
    booking_url: Optional[str] = None,  # NEW: optional booking link
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Call LLM to produce a natural language response, with timeout and circuit breaker."""
    try:
        llm_mode_hint, llm_priority_hint = await get_llm_mode_and_priority()
    except Exception:
        llm_mode_hint, llm_priority_hint = "unknown", "unknown"

    if await check_llm_circuit(llm_mode=llm_mode_hint, effective_mode=llm_mode_hint):
        logger.warning("LLM circuit breaker open, returning deterministic summary")
        return generate_deterministic_summary(
            best_flight, weather, filters_applied,
            error="circuit breaker open", location=intent.destination_iata
        ), _llm_degradation_payload(
            reason="upstream_unavailable",
            message="LLM circuit breaker is open; explanation generated via deterministic fallback.",
            provider="router",
        )

    # Cap flights shown in prompt to reduce token usage.
    max_flights_in_prompt = (
        PLANNER_LLM_MAX_FLIGHTS_ROUND_TRIP
        if intent.return_date
        else PLANNER_LLM_MAX_FLIGHTS_ONE_WAY
    )
    flights_str = "\n".join([
        f"- {f.airline} {f.flight_no} on {f.date or 'N/A'} | "
        f"{f.departure_time} → {f.arrival_time} | "
        f"{f.duration_min} min | {f.price_inr} | "
        f"Stops: {f.stops} | Baggage: {f.baggage}"
        for f in all_flights[:max_flights_in_prompt]
    ])

    # Format warnings if present, escalating "relaxed" warnings to mandatory instructions
    if warnings:
        processed = []
        for w in warnings:
            if "relaxed" in w.lower():
                processed.append(
                    "MANDATORY: Your baggage or airline preference could not be matched. "
                    "You MUST explicitly tell the user their preference was not available and "
                    "you are showing the closest alternative. Do NOT claim the flight meets their baggage requirement."
                )
            else:
                processed.append(w)
        warnings_str = "\nSystem Notes/Warnings:\n- " + "\n- ".join(processed)
    else:
        warnings_str = ""
    warnings_str = _truncate_for_prompt(warnings_str, PLANNER_LLM_WARNINGS_MAX_CHARS)
    trip_description_for_prompt = _truncate_for_prompt(
        trip_description,
        PLANNER_LLM_TRIP_DESCRIPTION_MAX_CHARS,
    )

    # Ensure weather values are plain (not enum objects) and build readable string
    weather_display = _normalized_weather_display(weather)

    # Determine forecast proximity to travel date
    forecast_date_str = weather_display.get("forecast_date")
    travel_date_str = best_flight.date or intent.date or "your travel date"
    forecast_is_approximate = False
    approx_note = ""
    if forecast_date_str and travel_date_str and travel_date_str != "your travel date":
        try:
            from datetime import datetime as _dt
            t_dt = _dt.strptime(travel_date_str, "%Y-%m-%d")
            f_dt = _dt.strptime(forecast_date_str, "%Y-%m-%d")
            delta = abs((t_dt - f_dt).days)
            if delta > 3:
                forecast_is_approximate = True
                approx_note = f" (closest available forecast; actual forecast for {forecast_date_str})"
        except Exception:
            pass

    # Build weather string — clearly labelled as FORECAST (not current conditions)
    forecast_label = travel_date_str
    weather_str = (
        f"Condition: {weather_display.get('condition', 'N/A')}, "
        f"Temperature: {weather_display.get('temperature_c', 'N/A')}°C "
        f"(feels like {weather_display.get('feels_like_c', 'N/A')}°C)"
    )
    if weather_display.get("temp_min_c") is not None and weather_display.get("temp_max_c") is not None:
        weather_str += (
            f", Daily low: {weather_display['temp_min_c']}°C, Daily high: {weather_display['temp_max_c']}°C"
        )
    weather_str += (
        f", Humidity: {weather_display.get('humidity', 'N/A')}%, "
        f"Wind: {weather_display.get('wind_kph', 'N/A')} kph, "
        f"AQI: {weather_display.get('air_quality_index', 'N/A')}"
    )
    if weather_display.get("precipitation_chance") is not None:
        weather_str += f", Precipitation chance: {weather_display['precipitation_chance']}%"
    if weather_display.get("has_rain"):
        weather_str += " ⚠️ Rain expected"
    if weather_display.get("has_snow"):
        weather_str += " ⚠️ Snow expected"

    # Handle unknown stops gracefully
    stops_val = best_flight.stops
    if str(stops_val) in ("N/A", "n/a", "", "unknown"):
        stops_display = "unknown (no data available)"
    elif isinstance(stops_val, int):
        stops_display = "non-stop" if stops_val == 0 else f"{stops_val} stop(s)"
    else:
        stops_display = str(stops_val)

    # Carbon display
    carbon_display = "N/A"
    if best_flight.carbon_emissions_g is not None:
        carbon_kg = round(best_flight.carbon_emissions_g / 1000, 1)
        carbon_display = f"{carbon_kg} kg CO₂"

    # CRITICAL CONSTRAINT for unknown stops
    constraint_note = ""
    if "unknown" in stops_display:
        constraint_note = "You MUST NOT state this is a non-stop or direct flight. If stops are unknown, say 'stop availability unknown'."

    # --- Baggage constraint for hand baggage mismatch ---
    baggage_constraint = ""
    if intent.baggage_pref == "hand":
        bf_baggage_low = best_flight.baggage.lower()
        if any(x in bf_baggage_low for x in ("checked", "check", "free bag", "hold")):
            baggage_constraint = (
                f"\nCRITICAL BAGGAGE CONSTRAINT: User requested hand/cabin-baggage only, "
                f"but this flight's baggage is '{best_flight.baggage}' (checked baggage). "
                "You MUST NOT open with or imply this flight meets the hand baggage requirement. "
                "Your first sentence MUST be something like: 'No hand-baggage-only flight was "
                "available; here is the closest alternative.' Then disclose the baggage mismatch clearly."
            )

    # --- Layover constraint for non-stop flights when user has a layover limit ---
    layover_constraint = ""
    if intent.layover_limit_minutes and best_flight.stops == 0:
        layover_constraint = (
            f"\nLAYOVER NOTE: This is a non-stop flight — it has NO layover at all, which "
            f"automatically satisfies the user's max layover requirement of {intent.layover_limit_minutes} minutes. "
            "CRITICAL PHRASING RULES — apply to every sentence including the opening and summary:\n"
            "  1. NEVER write 'with a layover less than X hours' or 'layover of less than X' "
            "anywhere in your response — those phrases imply the flight HAS a layover.\n"
            "  2. In your opening sentence, do NOT echo the user's constraint with 'layover of less than'. "
            "Instead write: 'Based on your max-layover preference, the best option is...'\n"
            f"  3. In your summary sentence, write something like: "
            f"'{best_flight.airline} {best_flight.flight_no} is the best non-stop option — it has no layover "
            "whatsoever, which exceeds your requirement.' Do not echo the user's constraint phrase."
        )

    # --- Temperature constraint for correct ordering ---
    temp_constraint = ""
    if weather_display.get("temp_min_c") is not None and weather_display.get("temp_max_c") is not None:
        temp_constraint = (
            f"\nTEMP RULE: Daily low is {weather_display['temp_min_c']}°C, daily high is "
            f"{weather_display['temp_max_c']}°C. Never swap these — do not describe "
            f"{weather_display['temp_max_c']}°C as the 'low' temperature."
        )

    # Construct system prompt with airline rule only if the best flight does NOT match the preference
    airline_rule = ""
    if intent.preferred_airlines:
        preferred_lower = [a.lower() for a in intent.preferred_airlines]
        best_airline_lower = best_flight.airline.lower()
        airline_matched = any(p in best_airline_lower or best_airline_lower in p for p in preferred_lower)
        if not airline_matched:
            airline_rule = (
                f"AIRLINE RULE: The user's preferred airline ({', '.join(intent.preferred_airlines)}) is not available. "
                "Your FIRST sentence MUST disclose this — something like: "
                "'No flights were found for [airline]; here is the closest available alternative.' "
                "Do NOT open with flight details and then mention the airline mismatch later. "
                "Do NOT invent or fabricate alternative flights — only present flights from the data above."
            )

    # Prepend a facts block to ensure origin/destination appear explicitly
    facts_block = (
        f"Origin: {intent.origin_iata or 'unknown'}\n"
        f"Destination: {intent.destination_iata or 'unknown'}\n"
        f"Departure date: {intent.date or 'not specified'}\n"
    )
    if intent.return_date:
        facts_block += f"Return date: {intent.return_date}\n"

    system = (
        "You are a professional travel planning assistant. "
        "CRITICAL: NEVER invent, fabricate, or suggest flight numbers, airline codes, prices, "
        "departure times, or any flight details that are not explicitly present in the flight "
        "data provided to you. If no matching airline is found, present only the available flight "
        "with a note that it differs from the preference — do NOT create fictional alternatives. "
        "RULE: If a flight data field is 'unknown (no data available)', you are PROHIBITED from "
        "stating or implying its value. "
        "IATA RULE: Whenever you mention a city's weather, always include its IATA code in "
        "parentheses, e.g. 'Weather for Bangalore (BLR)' or 'Mumbai (BOM)'. "
        "CITY NAME RULE: When writing about the flight destination, use ONLY the correct city name "
        "for the destination IATA code. Examples of correct mappings: MAA = Chennai (NOT Mumbai), "
        "BLR = Bangalore, BOM = Mumbai, DEL = Delhi. Never call MAA 'Mumbai' or BOM 'Chennai'. "
        + airline_rule
    )

    def _render_prompt(
        *,
        trip_description_block: str,
        warnings_block: str,
        flights_block: str,
    ) -> str:
        return f"""
CRITICAL CONSTRAINT: The stops field for this flight is '{stops_display}'.
{constraint_note}{baggage_constraint}{layover_constraint}{temp_constraint}

You are a helpful travel assistant helping a user plan {trip_description_block}.

User preferences:
- {filters_applied}
{warnings_block}

Flight options from {intent.origin_iata} to {intent.destination_iata} around {intent.date}:
{flights_block}

Best matching flight:
- {best_flight.airline} {best_flight.flight_no} on {best_flight.date or 'N/A'} |
  {best_flight.departure_time} → {best_flight.arrival_time} |
  Duration: {best_flight.duration_min} minutes |
  Price: {best_flight.price_inr} |
  Stops: {stops_display}{f" ({best_flight.layover_info})" if best_flight.layover_info else ""} |
  Baggage: {best_flight.baggage} |
  Carbon emissions: {carbon_display}
{f"{chr(10)}{price_insights_str}" if price_insights_str else ""}
{f"{chr(10)}{price_analysis_str}" if price_analysis_str else ""}
{f"{chr(10)}{price_prediction_str}" if price_prediction_str else ""}
Weather FORECAST for {intent.destination_iata} on {forecast_label}{approx_note}:
{weather_str}

IMPORTANT: Only reference the exact flights listed above. Do not create or suggest any other flights, codes, or prices.
User's question: {user_query}

Please recommend the best flight, explain why it matches their preferences, mention the weather forecast suitability (including packing advice based on min/max temperature and any rain or snow alerts), and answer the user's query helpfully.
"""

    prompt = _render_prompt(
        trip_description_block=trip_description_for_prompt,
        warnings_block=warnings_str,
        flights_block=flights_str,
    )

    # Combine facts block and prompt with an instruction to echo the facts
    full_prompt = (
        facts_block
        + "\nPlease include the above origin and destination clearly at the start of your summary.\n\n"
        + prompt
    )
    prompt_chars_before_trim = len(full_prompt)
    prompt_trimmed = False
    prompt_hard_trimmed = False
    trimmed_flights_in_prompt = max_flights_in_prompt
    if prompt_chars_before_trim > PLANNER_LLM_PROMPT_SOFT_LIMIT:
        prompt_trimmed = True
        trimmed_flights_in_prompt = max(2, max_flights_in_prompt // 2)
        trimmed_flights_str = "\n".join([
            f"- {f.airline} {f.flight_no} on {f.date or 'N/A'} | "
            f"{f.departure_time} → {f.arrival_time} | "
            f"{f.duration_min} min | {f.price_inr} | "
            f"Stops: {f.stops} | Baggage: {f.baggage}"
            for f in all_flights[:trimmed_flights_in_prompt]
        ])
        prompt = _render_prompt(
            trip_description_block=_truncate_for_prompt(
                trip_description_for_prompt,
                max(320, PLANNER_LLM_TRIP_DESCRIPTION_MAX_CHARS // 2),
            ),
            warnings_block=_truncate_for_prompt(
                warnings_str,
                max(240, PLANNER_LLM_WARNINGS_MAX_CHARS // 2),
            ),
            flights_block=trimmed_flights_str,
        )
        full_prompt = (
            facts_block
            + "\nPlease include the above origin and destination clearly at the start of your summary.\n\n"
            + prompt
        )
        logger.info(
            "LLM prompt trimmed for timeout protection",
            extra={
                "llm_mode": llm_mode_hint,
                "llm_priority": llm_priority_hint,
                "prompt_chars_before": prompt_chars_before_trim,
                "prompt_chars_after": len(full_prompt),
                "flights_before": max_flights_in_prompt,
                "flights_after": trimmed_flights_in_prompt,
                "is_round_trip": bool(intent.return_date),
            },
        )

    full_prompt, prompt_hard_trimmed = _apply_prompt_hard_limit(
        full_prompt,
        hard_limit=PLANNER_LLM_PROMPT_HARD_LIMIT,
    )
    if prompt_hard_trimmed:
        prompt_trimmed = True
        logger.info(
            "LLM prompt hard-capped for runtime stability",
            extra={
                "llm_mode": llm_mode_hint,
                "llm_priority": llm_priority_hint,
                "hard_limit_chars": PLANNER_LLM_PROMPT_HARD_LIMIT,
                "prompt_chars_after_cap": len(full_prompt),
            },
        )

    logger.info("Sending prompt to LLM")
    logger.debug(
        "LLM prompt prepared",
        extra={"prompt_chars": len(full_prompt)},
    )
    logger.debug(
        "LLM trip description prepared",
        extra={"trip_description_chars": len(trip_description)},
    )
    logger.info(
        "LLM explanation request context",
        extra={
            "llm_mode": llm_mode_hint,
            "llm_priority": llm_priority_hint,
            "prompt_chars": len(full_prompt),
            "prompt_trimmed": prompt_trimmed,
            "prompt_hard_trimmed": prompt_hard_trimmed,
            "flights_in_prompt": trimmed_flights_in_prompt,
            "all_flights_count": len(all_flights),
            "warnings_count": len(warnings or []),
            "has_return_date": bool(intent.return_date),
            "has_weather_payload": bool(weather_display),
            "has_price_context": bool(
                (price_insights_str or "").strip()
                or (price_analysis_str or "").strip()
                or (price_prediction_str or "").strip()
            ),
            "has_booking_url": bool((booking_url or "").strip()),
            "trip_description_chars": len(trip_description or ""),
        },
    )

    LLMROUTER_FALLBACK_MARKER = "All LLM backends failed"
    planner_timeout = _resolve_planner_llm_timeout()
    planner_model = _resolve_planner_llm_model()
    router_timeout_hint = get_env_float("ROUTER_TIMEOUT", 90.0)
    router_local_timeout_hint = _resolve_router_local_timeout_hint(planner_timeout)
    logger.info(
        "LLM timeout ownership (non-stream)",
        extra={
            "timeout_owner": "router_backend_timeout",
            "planner_timeout_hint_sec": planner_timeout,
            "router_local_timeout_sec": router_local_timeout_hint,
            "router_timeout_sec": router_timeout_hint,
        },
    )
    try:
        llm_text = await generate(
            prompt=full_prompt,
            system=system,
            model=planner_model,
            stream=False,
        )
        # Detect when llm_router itself returned an internal fallback string instead of raising
        if LLMROUTER_FALLBACK_MARKER in (llm_text or ""):
            logger.warning("generate() returned internal backend-failure fallback; retrying once")
            await asyncio.sleep(1)
            try:
                llm_text = await generate(
                    prompt=full_prompt,
                    system=system,
                    model=planner_model,
                    stream=False,
                )
            except Exception:
                llm_text = ""
            if not llm_text or LLMROUTER_FALLBACK_MARKER in llm_text:
                await record_llm_failure(
                    stage="generate_explanation",
                    reason="router_fallback_marker",
                    llm_mode=llm_mode_hint,
                    effective_mode=llm_mode_hint,
                    attempt_count=2,
                )
                return generate_deterministic_summary(
                    best_flight, weather, filters_applied,
                    error="All LLM backends failed", location=intent.destination_iata
                ), _llm_degradation_payload(
                    reason="upstream_unavailable",
                    message="All configured LLM backends failed; explanation generated via deterministic fallback.",
                    provider="router",
                )
        llm_text = _enforce_narrative_consistency(llm_text, best_flight, weather)
        await record_llm_success()
        return llm_text, None
    except (TimeoutError, asyncio.TimeoutError):
        timeout_budget = max(router_local_timeout_hint, planner_timeout)
        logger.error("LLM call timed out after %.2fs", timeout_budget)
        await record_llm_failure(
            stage="generate_explanation",
            reason="upstream_timeout",
            llm_mode=llm_mode_hint,
            effective_mode=llm_mode_hint,
            attempt_count=1,
            backend="ollama_or_router",
        )
        return generate_deterministic_summary(
            best_flight, weather, filters_applied,
            error="timed out", location=intent.destination_iata
        ), _llm_degradation_payload(
            reason="upstream_timeout",
            message=f"LLM explanation timed out after {timeout_budget}s; deterministic fallback used.",
            provider="router",
        )
    except AllBackendsFailed as e:
        status = e.as_dict()
        failures = status.get("failures") or []
        counter_reason = _primary_router_failure_reason(status)
        first_failure = failures[0] if failures else {}
        primary_backend = (
            str(first_failure.get("backend") or "")
            if isinstance(first_failure, dict)
            else None
        )
        single_backend_scope = _is_single_backend_unavailable(e)
        logger.warning(
            "LLM backend unavailable for explanation"
            if single_backend_scope
            else "LLM backends unavailable for explanation",
            extra={
                **status,
                "configured_llm_mode": llm_mode_hint,
                "failure_count_reported": len(failures),
            },
        )
        await record_llm_failure(
            stage="generate_explanation",
            reason=counter_reason,
            llm_mode=llm_mode_hint,
            effective_mode=status.get("effective_mode"),
            attempt_count=max(1, len(failures)),
            backend=primary_backend,
        )
        return generate_deterministic_summary(
            best_flight,
            weather,
            filters_applied,
            error=_safe_llm_error_message(e),
            location=intent.destination_iata
        ), _llm_degradation_payload(
            reason="upstream_unavailable",
            message=(
                "Configured LLM backend unavailable; deterministic explanation fallback used."
                if single_backend_scope
                else "LLM backends unavailable; deterministic explanation fallback used."
            ),
            provider="router",
            backend_status=status,
        )
    except Exception as e:
        logger.error("LLM call failed", extra={"error": str(e), "error_type": type(e).__name__})
        msg = str(e).lower()
        counter_reason = (
            "circuit_open"
            if ("circuit breaker open" in msg or "circuit breaker is open" in msg)
            else "upstream_timeout"
            if ("timed out" in msg or "timeout" in msg)
            else "upstream_unavailable"
        )
        await record_llm_failure(
            stage="generate_explanation",
            reason=counter_reason,
            llm_mode=llm_mode_hint,
            effective_mode=llm_mode_hint,
            attempt_count=1,
            backend="ollama_or_router",
        )
        return generate_deterministic_summary(
            best_flight, weather, filters_applied,
            error=_safe_llm_error_message(e), location=intent.destination_iata
        ), _llm_degradation_payload(
            reason="upstream_unavailable",
            message="LLM explanation unavailable; deterministic fallback used.",
            provider="router",
        )

def generate_deterministic_summary(
    best_flight: Flight, weather: Dict, filters: str,
    error: str = "", location: str = ""
) -> str:
    """Fallback summary when LLM is unavailable."""
    if isinstance(weather, dict):
        condition = weather.get("condition", weather.get("description", "unknown"))
        temp = weather.get("temperature_c", weather.get("temp", "unknown"))
    else:
        # Weather object
        condition = getattr(weather, "condition", "unknown")
        temp = getattr(weather, "temperature_c", getattr(weather, "temp", "unknown"))

    weather_str = f"{condition}, {temp}°C" if temp != "unknown" else condition
    loc_display = f" ({location})" if location else ""

    base = (f"I recommend {best_flight.airline} {best_flight.flight_no} at "
            f"{best_flight.departure_time} arriving {best_flight.arrival_time}. "
            f"Duration: {best_flight.duration_min} minutes, Price: {best_flight.price_inr}. "
            f"Weather at destination{loc_display}: {weather_str}. ")
    if error:
        base += f"(Explanation degraded: {error}.)"
    return base

# ----------------------------------------------------------------------
# City correction using LLM (with circuit breaker + tolerant extraction)
# ----------------------------------------------------------------------
async def correct_cities_with_llm(user_query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempt to extract origin/destination IATA codes using LLM with a structured JSON prompt.
    Returns (origin_iata, destination_iata, explanation) – explanation may be None if not needed.
    """
    if await check_llm_circuit():
        logger.warning("LLM circuit breaker open, skipping city correction")
        return None, None, None

    mini_prompt = f"""
Extract the route from this travel query.

Return ONLY valid JSON with the following fields:
- "origin": the origin city or IATA code (or null if unknown)
- "destination": the destination city or IATA code (or null if unknown)
- "via": any stopover city or IATA code (or null if none)
- "explanation": a one-sentence human summary of what you understood

Do not add any other text.

Query: {user_query}
"""
    try:
        fixed_text = await asyncio.wait_for(
            generate(
                prompt=mini_prompt,
                system="You are a precise travel assistant. Always output JSON only.",
                model=_resolve_planner_llm_model(),
                stream=False
            ),
            timeout=LLM_CORRECTION_TIMEOUT
        )

        # Attempt to find a JSON object in the response (allows the LLM to add commentary)
        parsed_json = None
        try:
            # Try direct JSON load first
            parsed_json = json.loads(fixed_text)
        except Exception:
            # Search for first { ... } block
            m = re.search(r'\{.*\}', fixed_text, flags=re.DOTALL)
            if m:
                try:
                    parsed_json = json.loads(m.group(0))
                except Exception:
                    parsed_json = None

        if parsed_json:
            origin_val = parsed_json.get("origin")
            dest_val = parsed_json.get("destination")
            via_val = parsed_json.get("via")
            explanation = parsed_json.get("explanation", fixed_text)
            # If the model gave IATA codes, normalize them; else keep raw for normalize_trip
            return (
                resolve_location(origin_val) if origin_val else None,
                resolve_location(dest_val) if dest_val else None,
                explanation
            )

        # --- tolerant regex extraction for common LLM free-text outputs ---
        def extract_field(txt: str, keys):
            for key in keys:
                # e.g., "origin: Delhi" or "origin - Delhi" or "origin = Delhi"
                m = re.search(rf'{key}\s*[:=\-]\s*([A-Za-z ]{{2,40}})', txt, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            # try "from X to Y" pattern:
            m2 = re.search(r'from\s+([A-Za-z ]{{2,40}})\s+to\s+([A-Za-z ]{{2,40}})', txt, flags=re.IGNORECASE)
            if m2:
                return m2.group(1).strip()
            return None

        origin_val = extract_field(fixed_text, ["origin", "from"])
        dest_val = extract_field(fixed_text, ["destination", "to", "dest"])
        # Also check "from ... to ..." for destination
        m_to = re.search(r'from\s+([A-Za-z ]{{2,40}})\s+to\s+([A-Za-z ]{{2,40}})', fixed_text, flags=re.IGNORECASE)
        if m_to:
            origin_val = origin_val or m_to.group(1).strip()
            dest_val = dest_val or m_to.group(2).strip()

        via_val = None
        m_via = re.search(r'(via|stopover)\s+([A-Za-z ]{{2,40}})', fixed_text, flags=re.IGNORECASE)
        if m_via:
            via_val = m_via.group(2).strip()

        if origin_val or dest_val or via_val:
            # Normalize if they look like IATA codes
            return (
                resolve_location(origin_val) if origin_val else None,
                resolve_location(dest_val) if dest_val else None,
                fixed_text  # explanation
            )

        # If no JSON found and no regex extracted, fall back to older behavior
        parsed = normalize_trip(user_query)  # note: we ignore llm_correction_text here
        return parsed["origin_iata"], parsed["destination_iata"], parsed.get("via_iata")  # explanation? maybe fixed_text

    except asyncio.TimeoutError:
        logger.warning("LLM city correction timed out")
    except Exception as e:
        if isinstance(e, AllBackendsFailed):
            logger.warning("LLM city correction skipped: all backends unavailable", extra=e.as_dict())
            return None, None, None
        logger.warning(f"LLM city correction failed: {e}")
    return None, None, None

# ----------------------------------------------------------------------
# Main internal planning function (async, layered, with dependency injection)
# ----------------------------------------------------------------------
async def _plan_trip_internal(
    *,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    date: Optional[str] = None,
    user_query: str,
    trip_type: Optional[str] = None,
    flights: Optional[List[Union[Dict, Flight]]] = None,
    depth: int = 0,
    flight_tool: Callable = default_flight_tool,
    weather_tool: Callable = default_weather_tool,
    skip_llm: bool = False   # if True, return data without LLM explanation
) -> Union[PlanResult, MultiCityResult, Dict]:
    """Internal implementation without top-level timeout. Used for non‑streaming mode."""
    # Prevent excessive recursion
    if depth >= MAX_RECURSION_DEPTH:
        logger.error("Max recursion depth reached")
        return {
            "error": "Too deep recursion in trip planning",
            "failure_reason": "planner_error",
            "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
        }

    # Use shared module-level cache unless custom tools are injected
    cached_search = (
        _shared_cached_search
        if flight_tool is default_flight_tool
        else create_cached_fetcher(900, 500, flight_tool)
    )

    cached_weather = (
        _shared_cached_weather
        if weather_tool is default_weather_tool
        else create_cached_fetcher(3600, 500, weather_tool)
    )
    weather_cache: Dict[str, Any] = {}
    weather_inflight: Dict[str, asyncio.Task] = {}

    def _normalize_weather_for_display(weather_value: Any, requested_location: Optional[str] = None) -> Any:
        """
        Keep weather payload shape intact and attach display labels without
        altering source numeric weather values.
        """
        if weather_value is None or isinstance(weather_value, Exception):
            return weather_value

        payload: Any = weather_value
        if isinstance(weather_value, dict):
            payload = dict(weather_value)
        elif hasattr(weather_value, "model_dump"):
            payload = weather_value.model_dump()
        elif hasattr(weather_value, "to_dict"):
            payload = weather_value.to_dict()
        elif hasattr(weather_value, "__dict__"):
            payload = dict(vars(weather_value))

        if not isinstance(payload, dict):
            return weather_value

        if requested_location and not payload.get("location"):
            payload["location"] = requested_location

        normalized_location = _sanitize_iata_code(str(requested_location or payload.get("location") or ""))
        if normalized_location:
            payload["location"] = normalized_location
            location_city = city_for_iata(normalized_location)
            location_label = label_for_iata(normalized_location) or normalized_location
            if location_city:
                payload["location_city"] = location_city
            payload["location_label"] = location_label

        return payload

    async def get_weather_once(location: str, travel_date: str) -> Any:
        cache_key = f"{location}_{travel_date}"
        if cache_key in weather_cache:
            return weather_cache[cache_key]

        inflight = weather_inflight.get(cache_key)
        if inflight is not None:
            return await inflight

        async def _fetch_weather() -> Any:
            weather_query_location = location
            normalized_loc = _sanitize_iata_code(location)
            if normalized_loc:
                weather_query_location = city_for_iata(normalized_loc) or location
            result = await cached_weather(location=weather_query_location, travel_date=travel_date, units="metric")
            normalized = _normalize_weather_for_display(result, requested_location=location)
            weather_cache[cache_key] = normalized
            return normalized

        task = asyncio.create_task(_fetch_weather())
        weather_inflight[cache_key] = task
        try:
            return await task
        finally:
            weather_inflight.pop(cache_key, None)

    # Phase timing
    phases = {}
    warnings = []
    # Always provide a debug_info dict so later code paths can safely augment it.
    debug_info: Dict[str, Any] = {}
    # Ensure filtered_count always exists to avoid UnboundLocalError later
    filtered_count: int = 0
    normalization_debug: Dict[str, Any] = {
        "input": {
            "origin": origin,
            "destination": destination,
            "date": date,
            "user_query": user_query,
        }
    }
    relaxation_attempts: List[Dict[str, Any]] = []
    tool_search_meta: Dict[str, Any] = {}
    pre_filter_count = 0
    post_filter_count = 0

    # ------------------------------------------------------------------
    # 1. Parse intent (overrides explicit params)
    # ------------------------------------------------------------------
    start = time.monotonic()
    # Only parse if there is meaningful user input; otherwise start with empty intent.
    if user_query:
        intent = parse_intent(user_query)
    else:
        intent = ParsedIntent()

    explicit_route_resolution: Dict[str, Any] = {}
    # Override with explicit parameters if provided, using central resolver
    if origin:
        resolved_origin, origin_trace = _resolve_city_to_iata_with_trace(origin)
        intent.origin_iata = resolved_origin
        explicit_route_resolution["origin"] = origin_trace
    if destination:
        resolved_destination, destination_trace = _resolve_city_to_iata_with_trace(destination)
        intent.destination_iata = resolved_destination
        explicit_route_resolution["destination"] = destination_trace
    if date:
        intent.date = date

    # Never propagate malformed airport values downstream.
    intent.origin_iata = _sanitize_iata_code(intent.origin_iata)
    intent.destination_iata = _sanitize_iata_code(intent.destination_iata)
    normalization_debug["after_initial_parse"] = {
        "origin_iata": intent.origin_iata,
        "destination_iata": intent.destination_iata,
        "route_parse_trace": intent.route_parse_trace,
        "date_parse_trace": intent.date_parse_trace,
        "explicit_route_resolution": explicit_route_resolution,
    }

    # Trip-type resolution keeps semantic intent separate from route-mode labels.
    normalized_trip_type = (trip_type or "").strip().lower()
    semantic_trip_map = {
        "business": "Business",
        "holiday": "Holiday",
        "flexible": "Flexible",
        "urgent": "Urgent",
    }
    route_trip_map = {
        "one-way": "one-way",
        "one way": "one-way",
        "oneway": "one-way",
        "round-trip": "round-trip",
        "round trip": "round-trip",
        "return": "round-trip",
        "via-stopover": "via-stopover",
        "via stopover": "via-stopover",
        "via / stopover": "via-stopover",
        "stopover": "via-stopover",
    }
    semantic_trip_override = semantic_trip_map.get(normalized_trip_type)
    requested_trip_mode = route_trip_map.get(normalized_trip_type)

    resolved_trip_type = semantic_trip_override or intent.trip_type or "Business"
    intent.trip_type = resolved_trip_type

    # Structured round-trip mode should trigger return-leg planning even when query text omits it.
    if requested_trip_mode == "round-trip" and not intent.return_date and not intent.trip_duration_days:
        intent.trip_duration_days = 3
    elif requested_trip_mode == "one-way":
        # Structured one-way mode should suppress return-leg inference from free-text.
        intent.return_date = None
        intent.trip_duration_days = None

    # Sanity check: stopover city cannot be same as origin or destination
    if intent.stopover_city:
        stopover_lower = intent.stopover_city.lower()
        if (intent.origin_iata and stopover_lower == intent.origin_iata.lower()) or \
           (intent.destination_iata and stopover_lower == intent.destination_iata.lower()):
            logger.warning(f"Stopover city '{intent.stopover_city}' same as origin/destination; ignoring.")
            intent.stopover_city = None

    phases['intent_parsing'] = time.monotonic() - start

    llm_correction_explanation = None

    # Deterministic non-LLM fallback first for compact/weak route forms
    # to reduce model-dependent variability during route recovery.
    if user_query and (not intent.origin_iata or not intent.destination_iata):
        inferred_origin, inferred_dest, infer_trace = _infer_route_pair_from_query(user_query)
        if not intent.origin_iata and inferred_origin:
            intent.origin_iata = _sanitize_iata_code(inferred_origin)
        if not intent.destination_iata and inferred_dest:
            intent.destination_iata = _sanitize_iata_code(inferred_dest)
        normalization_debug["route_inference"] = infer_trace

    # If we still lack origin/destination, use LLM correction as fallback.
    if user_query and (not intent.origin_iata or not intent.destination_iata):
        logger.info("Missing origin/destination after deterministic recovery, attempting LLM correction")
        start = time.monotonic()
        corrected_origin, corrected_dest, explanation = await correct_cities_with_llm(user_query)
        normalized_corrected_origin = _sanitize_iata_code(corrected_origin)
        normalized_corrected_dest = _sanitize_iata_code(corrected_dest)
        # Non-destructive merge: never clobber an already-resolved side with
        # null/invalid LLM output. LLM correction is only allowed to fill missing
        # route sides when normalization yields a valid IATA code.
        if not intent.origin_iata and normalized_corrected_origin:
            intent.origin_iata = normalized_corrected_origin
        if not intent.destination_iata and normalized_corrected_dest:
            intent.destination_iata = normalized_corrected_dest
        if explanation:
            llm_correction_explanation = explanation
        normalization_debug["llm_correction"] = {
            "attempted": True,
            "raw_origin": corrected_origin,
            "raw_destination": corrected_dest,
            "normalized_origin_iata": normalized_corrected_origin,
            "normalized_destination_iata": normalized_corrected_dest,
            "explanation": llm_correction_explanation,
        }
        phases['city_correction'] = time.monotonic() - start

    normalization_debug["route_trace"] = {
        "raw_fragments": (intent.route_parse_trace or {}).get("raw_fragments"),
        "query_resolution": {
            "origin": (intent.route_parse_trace or {}).get("origin_resolution"),
            "destination": (intent.route_parse_trace or {}).get("destination_resolution"),
            "via": (intent.route_parse_trace or {}).get("via_resolution"),
        },
        "explicit_param_resolution": explicit_route_resolution,
        "route_inference": normalization_debug.get("route_inference"),
    }

    query_route_trace = intent.route_parse_trace or {}
    explicit_origin_basis = (explicit_route_resolution.get("origin") or {}).get("resolution_basis")
    explicit_destination_basis = (explicit_route_resolution.get("destination") or {}).get("resolution_basis")
    query_origin_basis = (query_route_trace.get("origin_resolution") or {}).get("resolution_basis")
    query_destination_basis = (query_route_trace.get("destination_resolution") or {}).get("resolution_basis")
    normalization_debug["final"] = {
        "origin_iata": intent.origin_iata,
        "destination_iata": intent.destination_iata,
        "origin_resolution_basis": explicit_origin_basis or query_origin_basis,
        "destination_resolution_basis": explicit_destination_basis or query_destination_basis,
        "origin_resolution_is_fuzzy": bool(
            ((explicit_route_resolution.get("origin") or {}).get("resolver_trace") or {}).get("is_fuzzy")
            or ((query_route_trace.get("origin_resolution") or {}).get("resolver_trace") or {}).get("is_fuzzy")
        ),
        "destination_resolution_is_fuzzy": bool(
            ((explicit_route_resolution.get("destination") or {}).get("resolver_trace") or {}).get("is_fuzzy")
            or ((query_route_trace.get("destination_resolution") or {}).get("resolver_trace") or {}).get("is_fuzzy")
        ),
    }

    if not intent.origin_iata or not intent.destination_iata:
        logger.warning(
            "Route normalization failed after correction",
            extra={
                "origin_iata": intent.origin_iata,
                "destination_iata": intent.destination_iata,
                "user_query": user_query,
            },
        )
        return {
            "error": "Could not determine origin or destination airport after AI correction.",
            "failure_reason": "invalid_route",
            "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
            "debug_info": {
                "phases": phases.copy(),
                "intent": intent.model_dump(),
                "normalization": normalization_debug,
                "relaxation_attempts": relaxation_attempts,
            },
        }

    # ------------------------------------------------------------------
    # 2. Determine search date and return date
    # ------------------------------------------------------------------
    if intent.date:
        try:
            base_date = datetime.strptime(intent.date, "%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"Invalid date format '{intent.date}'; expected YYYY-MM-DD") from e
    else:
        # Default to tomorrow if no date provided
        base_date = datetime.now() + timedelta(days=1)
    search_date = base_date.strftime("%Y-%m-%d")

    normalization_debug["date_interpretation"] = {
        "provided_date_param": date,
        "parsed_intent_date": intent.date,
        "search_date": search_date,
        "source": (intent.date_parse_trace or {}).get("source"),
        "year_inferred": bool((intent.date_parse_trace or {}).get("year_inferred")),
        "discard_reason": (intent.date_parse_trace or {}).get("discard_reason"),
    }

    if not intent.return_date and intent.trip_duration_days:
        intent.return_date = (base_date + timedelta(days=intent.trip_duration_days)).strftime("%Y-%m-%d")
        normalization_debug["date_interpretation"]["return_date_derived_from_duration"] = {
            "trip_duration_days": intent.trip_duration_days,
            "return_date": intent.return_date,
        }

    # Reject past dates truthfully; do not silently rewrite calendar intent.
    today_date = datetime.now().date()
    outbound_dt = None
    try:
        outbound_dt = datetime.strptime(search_date, "%Y-%m-%d").date()
        if outbound_dt < today_date:
            return {
                "error": f"Travel date {search_date} is in the past. Please provide today or a future date.",
                "failure_reason": "invalid_past_date",
                "search_date": search_date,
                "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
                "debug_info": {"phases": phases.copy(), "intent": intent.model_dump()},
            }
    except Exception:
        pass

    if intent.return_date:
        try:
            return_dt = datetime.strptime(intent.return_date, "%Y-%m-%d").date()
            if return_dt < today_date:
                return {
                    "error": f"Return date {intent.return_date} is in the past. Please provide a future return date.",
                    "failure_reason": "invalid_past_date",
                    "search_date": search_date,
                    "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
                    "debug_info": {"phases": phases.copy(), "intent": intent.model_dump()},
                }
            if outbound_dt and return_dt < outbound_dt:
                return {
                    "error": f"Return date {intent.return_date} must be on or after departure date {search_date}.",
                    "failure_reason": "invalid_date_order",
                    "search_date": search_date,
                    "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
                    "debug_info": {"phases": phases.copy(), "intent": intent.model_dump()},
                }
            if outbound_dt:
                # Keep duration telemetry aligned with the actual outbound→return delta.
                intent.trip_duration_days = max((return_dt - outbound_dt).days, 0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Handle stopover (multicity) – sequential legs, but note: this is for multi-segment, not true stopover
    # For true stopover, we should filter flights after search.
    # ------------------------------------------------------------------
    if intent.stopover_city:
        via_iata = _resolve_city_to_iata(intent.stopover_city) if intent.stopover_city else None
        if via_iata and via_iata not in (intent.origin_iata, intent.destination_iata):
            # Fetch leg1
            leg1 = await _plan_trip_internal(
                origin=intent.origin_iata,
                destination=via_iata,
                date=search_date,
                user_query="",
                trip_type=intent.trip_type,
                depth=depth+1,
                flight_tool=flight_tool,
                weather_tool=weather_tool,
                skip_llm=True
            )
            # Fetch leg2
            leg2 = await _plan_trip_internal(
                origin=via_iata,
                destination=intent.destination_iata,
                date=search_date,
                user_query="",
                trip_type=intent.trip_type,
                depth=depth+1,
                flight_tool=flight_tool,
                weather_tool=weather_tool,
                skip_llm=True
            )

            if isinstance(leg1, PlanResult) and isinstance(leg2, PlanResult):
                # Single LLM call covering both legs at once
                l1_flight = leg1.best_flight if isinstance(leg1.best_flight, Flight) else Flight(**leg1.best_flight)
                l2_flight = leg2.best_flight if isinstance(leg2.best_flight, Flight) else Flight(**leg2.best_flight)
                l1_weather = leg1.weather or {}
                l2_weather = leg2.weather or {}

                def _fmt_weather(w):
                    return f"{w.get('condition','?')}, {w.get('temperature_c','?')}°C"

                combined_trip_description = (
                    f"a multi-city trip via {intent.stopover_city}\n\n"
                    f"LEG 1: {intent.origin_iata} → {via_iata} on {search_date}\n"
                    f"  Flight: {l1_flight.airline} {l1_flight.flight_no} | "
                    f"{l1_flight.departure_time} → {l1_flight.arrival_time} | "
                    f"{l1_flight.duration_min}min | {l1_flight.price_inr} | Stops: {l1_flight.stops}\n"
                    f"  Weather at {via_iata} (stopover city): {_fmt_weather(l1_weather)}\n"
                    f"  NOTE: The {_fmt_weather(l1_weather)} temperature is FOR {via_iata} ONLY.\n\n"
                    f"LEG 2: {via_iata} → {intent.destination_iata} on {search_date}\n"
                    f"  Flight: {l2_flight.airline} {l2_flight.flight_no} | "
                    f"{l2_flight.departure_time} → {l2_flight.arrival_time} | "
                    f"{l2_flight.duration_min}min | {l2_flight.price_inr} | Stops: {l2_flight.stops}\n"
                    f"  Weather at {intent.destination_iata} (final destination): {_fmt_weather(l2_weather)}\n"
                    f"  NOTE: The {_fmt_weather(l2_weather)} temperature is FOR {intent.destination_iata} ONLY.\n\n"
                )

                # Add per-leg temperature ordering constraints
                def _temp_rule(w, loc):
                    t_min = w.get("temp_min_c")
                    t_max = w.get("temp_max_c")
                    if t_min is not None and t_max is not None and t_max != t_min:
                        return (
                            f"TEMP RULE FOR {loc}: Daily low={t_min}°C, daily high={t_max}°C. "
                            f"NEVER describe {t_max}°C as the 'low' temperature.\n"
                        )
                    return ""
                combined_trip_description += _temp_rule(l1_weather, via_iata)
                combined_trip_description += _temp_rule(l2_weather, intent.destination_iata)
                combined_trip_description += (
                    "Label your response clearly as 'Leg 1:' and 'Leg 2:' sections. "
                    "Do NOT mix temperatures between legs."
                )

                # --- Enrich with stopover filter info ---
                # In explicit multi-city mode we intentionally split the route into
                # two legs around `via_iata`, so this request is considered matched
                # by construction (layover_airports is not the right signal here).
                matched_leg1 = 1
                matched_leg2 = 1
                matched_count = 2
                debug_info_stopover = {
                    "requested": intent.stopover_city,
                    "resolved_iata": via_iata,
                    "matched_count": matched_count,
                    "match_strategy": "multicity_leg_split",
                    "match_reason": "explicit_multicity_leg_split",
                }
                logger.info("Stopover filter applied", extra=debug_info_stopover)

                if matched_count > 0:
                    # Use first matching flight's layover_info
                    top = l1_flight if matched_leg1 else l2_flight
                    layover_text = top.layover_info or "No layover details available"
                    leg_lines = []
                    # We don't have full leg list in Flight; but we have layover_info string. Use that.
                    combined_trip_description += f"\nTop matched itinerary has layovers: {layover_text}\nPlease explain the layovers and whether this matches the user's 'via {intent.stopover_city}' request in plain language.\n"
                else:
                    combined_trip_description += f"\nNo itineraries were found that stop via {intent.stopover_city}. Please explain that no flights matched the stopover requirement and suggest nearby alternatives or changing the stopover city.\n"

                # ADD DEBUG LOGGING FOR COMBINED TRIP DESCRIPTION
                logger.debug(
                    "Multi-city combined_trip_description (first 500 chars): %s",
                    combined_trip_description[:500].replace('\n', ' ')
                )

                # FIX: Use final destination weather (l2_weather) for the standard weather block,
                # while the combined_trip_description already contains per-leg weather.
                combined_llm, combined_llm_degradation = await generate_explanation(
                    user_query=user_query,
                    intent=intent,
                    best_flight=l1_flight,
                    weather=l2_weather,          # final destination weather for the standard block
                    all_flights=[l1_flight, l2_flight],
                    filters_applied="multi-city stopover trip",
                    trip_description=combined_trip_description,
                    warnings=warnings,
                    price_insights_str="",       # Not used in stopover for now
                    price_analysis_str="",
                    price_prediction_str="",
                )
                if combined_llm_degradation:
                    warnings.append(
                        f"Detailed explanation degraded ({combined_llm_degradation.get('reason')}); "
                        "structured itinerary data is complete."
                    )

                # Split the combined response on the "Leg 2" boundary so each leg gets its own slice.
                split_match = re.search(r'\bLeg\s*2\b', combined_llm, re.IGNORECASE)
                if split_match and split_match.start() > 30:
                    leg1.llm_response = combined_llm[:split_match.start()].strip()
                    leg2.llm_response = combined_llm[split_match.start():].strip()
                    # If either slice is still too short, augment with template rather than fully replacing
                    if len(leg1.llm_response) < 300:
                        leg1.llm_response += (
                            f"\n\nWeather details for {via_iata}: "
                            f"Daily low: {l1_weather.get('temp_min_c', 'N/A')}°C, "
                            f"daily high: {l1_weather.get('temp_max_c', 'N/A')}°C. "
                            f"Pack suitable clothing for {str(l1_weather.get('condition', 'the forecast')).lower()} "
                            f"weather with temperatures around {l1_weather.get('temperature_c', 'N/A')}°C at the stopover. "
                            f"Precipitation chance: {l1_weather.get('precipitation_chance', 'N/A')}%."
                        )
                    if len(leg2.llm_response) < 300:
                        leg2.llm_response += (
                            f"\n\nWeather details for {intent.destination_iata}: "
                            f"Daily low: {l2_weather.get('temp_min_c', 'N/A')}°C, "
                            f"daily high: {l2_weather.get('temp_max_c', 'N/A')}°C. "
                            f"Pack suitable clothing for {str(l2_weather.get('condition', 'the forecast')).lower()} "
                            f"weather with temperatures around {l2_weather.get('temperature_c', 'N/A')}°C at your destination. "
                            f"Precipitation chance: {l2_weather.get('precipitation_chance', 'N/A')}%."
                        )
                else:
                    # Combined call failed or was too short. Use deterministic summaries for BOTH legs.
                    l1_wx = l1_weather
                    leg1.llm_response = (
                        f"Leg 1: {intent.origin_iata} to {via_iata} on {search_date}\n\n"
                        f"Flight: {l1_flight.airline} {l1_flight.flight_no} | "
                        f"{l1_flight.departure_time} → {l1_flight.arrival_time} | "
                        f"{l1_flight.duration_min} min | {l1_flight.price_inr} | Stops: {l1_flight.stops}\n\n"
                        f"Weather at {via_iata} (stopover city): {_fmt_weather(l1_wx)}. "
                        f"Daily low: {l1_wx.get('temp_min_c', 'N/A')}°C, "
                        f"daily high: {l1_wx.get('temp_max_c', 'N/A')}°C. "
                        f"Pack suitable clothing for {str(l1_wx.get('condition', 'the forecast')).lower()} "
                        f"weather with temperatures around {l1_wx.get('temperature_c', 'N/A')}°C "
                        f"at the stopover. "
                        f"Precipitation chance: {l1_wx.get('precipitation_chance', 'N/A')}%."
                    )
                    l2_wx = l2_weather
                    leg2.llm_response = (
                        f"Leg 2: {via_iata} to {intent.destination_iata} on {search_date}\n\n"
                        f"Flight: {l2_flight.airline} {l2_flight.flight_no} | "
                        f"{l2_flight.departure_time} → {l2_flight.arrival_time} | "
                        f"{l2_flight.duration_min} min | {l2_flight.price_inr} | Stops: {l2_flight.stops}\n\n"
                        f"Weather at {intent.destination_iata} (final destination): {_fmt_weather(l2_wx)}. "
                        f"Daily low: {l2_wx.get('temp_min_c', 'N/A')}°C, "
                        f"daily high: {l2_wx.get('temp_max_c', 'N/A')}°C. "
                        f"Pack suitable clothing for {str(l2_wx.get('condition', 'the forecast')).lower()} "
                        f"weather with temperatures around {l2_wx.get('temperature_c', 'N/A')}°C "
                        f"at your final destination. "
                        f"Precipitation chance: {l2_wx.get('precipitation_chance', 'N/A')}%."
                    )

                # Add API trace and stopover filter to each leg's debug_info
                leg1.debug_info = leg1.debug_info or {}
                leg1.debug_info["stopover_filter"] = debug_info_stopover
                leg1.debug_info["api_trace"] = {
                    "flight": {
                        "request": {
                            "departure": intent.origin_iata,
                            "arrival": via_iata,
                            "date": search_date,
                            "intent_date": intent.date,
                            "return_date": intent.return_date if intent.return_date else None,
                        },
                        "raw_count": 1,
                        "filtered_count": 1,
                        "best_flight_no": l1_flight.flight_no,
                        "raw_response": [dict(l1_flight)],
                    },
                    "weather": {
                        "request": {
                            "location": via_iata,
                            "date": search_date,
                        },
                        "forecast_date": l1_weather.get("forecast_date"),
                        "condition": l1_weather.get("condition"),
                        "temperature_c": l1_weather.get("temperature_c"),
                        "raw_response": l1_weather,
                    }
                }
                leg2.debug_info = leg2.debug_info or {}
                leg2.debug_info["stopover_filter"] = debug_info_stopover
                leg2.debug_info["api_trace"] = {
                    "flight": {
                        "request": {
                            "departure": via_iata,
                            "arrival": intent.destination_iata,
                            "date": search_date,
                            "intent_date": intent.date,
                            "return_date": intent.return_date if intent.return_date else None,
                        },
                        "raw_count": 1,
                        "filtered_count": 1,
                        "best_flight_no": l2_flight.flight_no,
                        "raw_response": [dict(l2_flight)],
                    },
                    "weather": {
                        "request": {
                            "location": intent.destination_iata,
                            "date": search_date,
                        },
                        "forecast_date": l2_weather.get("forecast_date"),
                        "condition": l2_weather.get("condition"),
                        "temperature_c": l2_weather.get("temperature_c"),
                        "raw_response": l2_weather,
                    }
                }
                leg1 = leg1.model_copy(update={"stopover_filter": debug_info_stopover})
                leg2 = leg2.model_copy(update={"stopover_filter": debug_info_stopover})
                return MultiCityResult(legs=[leg1, leg2])
        # fallback: if via_iata not resolved or is same as origin/destination, treat as normal trip

    # ------------------------------------------------------------------
    # 2 & 3. Fetch ALL API data — flights and weather, with smart reuse and parallelism
    # ------------------------------------------------------------------
    start_apis = time.monotonic()
    search_profile = _build_flight_search_profile(intent, normalization_debug)
    normalization_debug["flight_search_profile"] = search_profile

    async def _call_flight_tool_safe(**kwargs):
        """
        Forward deep_search/max_results when supported.
        Older injected test doubles may not accept deep_search.
        """
        try:
            return await flight_tool(**kwargs)
        except TypeError as e:
            if "deep_search" in str(e):
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("deep_search", None)
                return await flight_tool(**fallback_kwargs)
            raise

    # Outbound flight task
    flight_task = asyncio.create_task(
        _call_flight_tool_safe(
            departure=intent.origin_iata,
            arrival=intent.destination_iata,
            date=search_date,
            max_results=search_profile["max_results"],
            return_date=intent.return_date if intent.return_date else None,
            deep_search=search_profile["deep_search"],
        )
    )

    # We'll gather flight and outbound weather first (only outbound weather for now)
    gather_tasks = [flight_task]
    # We may add weather tasks later depending on key availability and date limits
    # For now, we just await the flight result.
    try:
        gather_results = await asyncio.gather(*gather_tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Parallel data fetch failed: {e}")
        return {
            "error": str(e),
            "failure_reason": _classify_tool_failure_reason(e),
            "search_date": search_date,
            "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
            "debug_info": {"phases": phases.copy()},
        }

    result_iter = iter(gather_results)
    flight_result = next(result_iter)

    # --- Attempt to reuse return flights from the initial API response ---
    return_flight_result = None
    if intent.return_date and not intent.stopover_city:
        try:
            # flight_result may be tuple (flights, price_insights) or just flights
            raw_flights = flight_result[0] if isinstance(flight_result, tuple) else flight_result
            if isinstance(raw_flights, list) and raw_flights:
                # Try to extract flights that match return date
                return_hits = []
                outbound_hits = []
                for f in raw_flights:
                    # Some flight payloads are dicts while others are Flight/Pydantic objects.
                    flight_date = _flight_value_safe(f, "date") or _flight_value_safe(f, "search_date")
                    if flight_date == intent.return_date:
                        return_hits.append(f)
                    elif flight_date == search_date:
                        outbound_hits.append(f)
                if return_hits:
                    logger.info(f"Reusing {len(return_hits)} return flights from initial API response")
                    return_flight_result = return_hits
                else:
                    logger.debug("No return flights in initial response; will perform separate search")
            else:
                logger.debug("Flight result not a list, cannot extract return legs")
        except Exception as e:
            logger.warning(f"Error while inspecting flight_result for return leg: {e}; will fall back to separate call")
            return_flight_result = None

    # Now, after flight result, we can decide on weather fetching.
    today = datetime.now().date()
    return_date_str = intent.return_date if intent.return_date else None

    # Determine which weather dates we need (within forecast limit)
    dates_to_fetch = []
    # Outbound weather
    try:
        outbound_dt = datetime.strptime(search_date, "%Y-%m-%d").date()
        outbound_delta_days = (outbound_dt - today).days
        if 0 <= outbound_delta_days <= WEATHER_FORECAST_MAX_DAYS:
            dates_to_fetch.append(("outbound", intent.destination_iata, search_date))
    except Exception:
        # if date invalid, skip weather
        pass

    # Return weather (if we have return date and not stopover)
    if intent.return_date and not intent.stopover_city:
        try:
            return_dt = datetime.strptime(intent.return_date, "%Y-%m-%d").date()
            return_delta_days = (return_dt - today).days
            if 0 <= return_delta_days <= WEATHER_FORECAST_MAX_DAYS:
                dates_to_fetch.append(("return", intent.origin_iata, intent.return_date))
        except Exception:
            pass

    # Fetch weather intelligently: parallel if enough keys, else sequential
    weather_out = None
    weather_ret = None
    weather_present_out = False
    weather_present_ret = False
    weather_reason_out = None
    weather_reason_ret = None

    if dates_to_fetch:
        # Check how many weather keys are active
        status = await api_key_manager.get_status()
        weather_keys_active = sum(1 for e in status.get("weather", []) if e["active"])

        if len(dates_to_fetch) == 2 and weather_keys_active >= 2:
            # Run both weather fetches in parallel
            logger.info("Running parallel weather fetches (2 keys available)")
            tasks = []
            for _, loc, dt in dates_to_fetch:
                tasks.append(asyncio.create_task(
                    get_weather_once(location=loc, travel_date=dt)
                ))
            weather_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Map back: first task is outbound (if both), second return
            if weather_results[0] is not None and not isinstance(weather_results[0], Exception):
                weather_out = weather_results[0]
                weather_present_out = True
            else:
                weather_out = {}
                weather_present_out = False
                weather_reason_out = "api_failure"
            if len(weather_results) > 1:
                if weather_results[1] is not None and not isinstance(weather_results[1], Exception):
                    weather_ret = weather_results[1]
                    weather_present_ret = True
                else:
                    weather_ret = {}
                    weather_present_ret = False
                    weather_reason_ret = "api_failure"
        else:
            # Sequential (safe)
            logger.info("Running sequential weather fetches (<=1 key available or only one date)")
            for leg, loc, dt in dates_to_fetch:
                try:
                    res = await get_weather_once(location=loc, travel_date=dt)
                    if leg == "outbound":
                        weather_out = res
                        weather_present_out = True
                    else:
                        weather_ret = res
                        weather_present_ret = True
                except Exception as e:
                    logger.warning(f"Weather fetch for {leg} failed: {e}")
                    if leg == "outbound":
                        weather_out = {}
                        weather_present_out = False
                        weather_reason_out = "api_failure"
                    else:
                        weather_ret = {}
                        weather_present_ret = False
                        weather_reason_ret = "api_failure"
    else:
        # No weather within forecast window, use placeholders and set reason
        if not weather_out and (not dates_to_fetch or dates_to_fetch[0][0] != "outbound"):
            weather_out = {"condition": "Unpredictable this far in the future", "temperature_c": "N/A"}
            weather_present_out = False
            weather_reason_out = "forecast_horizon_exceeded"
        if intent.return_date and not intent.stopover_city and not weather_ret:
            weather_ret = {"condition": "Unpredictable this far in the future", "temperature_c": "N/A"}
            weather_present_ret = False
            weather_reason_ret = "forecast_horizon_exceeded"

    # PATCH: If return date exists but was beyond forecast window (not added to dates_to_fetch),
    # set the placeholder now. The else branch above only runs when dates_to_fetch is entirely empty.
    if intent.return_date and not intent.stopover_city and weather_ret is None:
        weather_ret = {"condition": "Unpredictable this far in the future", "temperature_c": "N/A"}
        weather_present_ret = False
        weather_reason_ret = "forecast_horizon_exceeded"

    def _align_weather_date(
        payload: Any,
        *,
        requested_date: Optional[str],
        present: bool,
        reason: Optional[str],
        leg: str,
    ) -> Tuple[Any, bool, Optional[str]]:
        if not requested_date or payload is None or isinstance(payload, Exception):
            return payload, present, reason

        if isinstance(payload, dict):
            weather_payload = dict(payload)
        elif hasattr(payload, "model_dump"):
            weather_payload = payload.model_dump()
        elif hasattr(payload, "__dict__"):
            weather_payload = dict(vars(payload))
        else:
            return payload, present, reason

        forecast_date = weather_payload.get("forecast_date")
        if forecast_date and str(forecast_date) != requested_date:
            logger.info(
                "Weather forecast date mismatch for %s leg; using unavailable placeholder",
                leg,
                extra={
                    "requested_date": requested_date,
                    "provider_forecast_date": str(forecast_date),
                },
            )
            return (
                {
                    "condition": "Forecast unavailable for requested travel date",
                    "temperature_c": "N/A",
                    "forecast_date": requested_date,
                    "requested_date": requested_date,
                    "provider_forecast_date": str(forecast_date),
                    "forecast_exact_match": False,
                },
                False,
                "forecast_date_mismatch",
            )

        if requested_date:
            weather_payload["forecast_exact_match"] = bool(forecast_date and str(forecast_date) == requested_date)
        return weather_payload, present, reason

    if intent.return_date and not intent.stopover_city:
        weather_ret, weather_present_ret, weather_reason_ret = _align_weather_date(
            weather_ret,
            requested_date=intent.return_date,
            present=weather_present_ret,
            reason=weather_reason_ret,
            leg="return",
        )

    # If we still don't have return_flight_result and we need it, fetch it now sequentially
    if intent.return_date and not intent.stopover_city and return_flight_result is None:
        logger.info("Performing separate return flight search")
        try:
            return_flight_result = await _call_flight_tool_safe(
                departure=intent.destination_iata,
                arrival=intent.origin_iata,
                date=return_date_str,
                max_results=search_profile["max_results"],
                deep_search=search_profile["deep_search"],
            )
        except Exception as e:
            logger.warning(f"Return flight search failed: {e}")
            return_flight_result = e

    phases['api_parallel'] = time.monotonic() - start_apis

    # --- Process flight_result and extract outbound flights and price_insights ---
    if isinstance(flight_result, Exception):
        logger.error(f"Flight search failed: {flight_result}")
        failure_reason = _classify_tool_failure_reason(flight_result)
        return {
            "error": str(flight_result),
            "failure_reason": failure_reason,
            "search_date": search_date,
            "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
            "debug_info": {"phases": phases.copy()},
        }
    if isinstance(flight_result, tuple):
        all_flights, price_payload = flight_result
        if isinstance(price_payload, dict) and "_search_meta" in price_payload:
            tool_search_meta = dict(price_payload.get("_search_meta") or {})
            price_insights_raw = price_payload.get("price_insights")
        else:
            price_insights_raw = price_payload
    else:
        all_flights = flight_result
        price_insights_raw = None
    if not all_flights:
        raw_candidate_count = int(tool_search_meta.get("raw_candidate_count") or 0)
        no_flights_reason = "filtered_out" if raw_candidate_count > 0 else "no_inventory"
        return {
            "warning": "No live flights found.",
            "fallback": True,
            "failure_reason": "no_flights",
            "no_flights_reason": no_flights_reason,
            "search_date": search_date,
            "weather": {},
            "flight_counts": {
                "pre_filter": 0,
                "post_filter": 0,
                "filtered_out": raw_candidate_count,
                "raw_provider": raw_candidate_count,
            },
            "debug_info": {
                "phases": phases.copy(),
                "tool_search_meta": tool_search_meta,
            },
        }

    # If we extracted return flights earlier, we need to separate them from all_flights.
    if return_flight_result and isinstance(return_flight_result, list) and not isinstance(return_flight_result, Exception):
        # Remove any flights in all_flights that have return date
        original_len = len(all_flights)
        all_flights = [f for f in all_flights if _flight_value_safe(f, "date") != intent.return_date]
        if len(all_flights) != original_len:
            logger.info(f"Removed {original_len - len(all_flights)} return-leg flights from outbound list")

    weather_data = weather_out if weather_out and not isinstance(weather_out, Exception) else {}

    # Capture raw flights before normalization (all_flights now only outbound)
    _raw_flights_before_normalize = [
        dict(f) if isinstance(f, dict) else vars(f)
        for f in all_flights
    ]

    # At this point, all_flights is a list of raw flight dicts; normalize them
    all_flights = normalize_flights(all_flights, search_date)
    pre_filter_count = len(all_flights)

    # ------------------------------------------------------------------
    # 4. Apply filters and rank (with fallback) – update effective intent
    # ------------------------------------------------------------------
    start = time.monotonic()
    effective_intent = intent.model_copy(deep=True)  # start with original intent
    ranking_intent = intent

    # Apply standard filters first
    filtered, filter_warnings = filter_flights(all_flights, effective_intent)
    warnings.extend(filter_warnings)
    filtered_count = len(filtered) if filtered is not None else 0
    relaxation_attempts.append({
        "step": "strict_filters",
        "matched_count": filtered_count,
    })

    # If user requested a stopover, we need to filter flights that actually have that stopover
    # and enrich the prompt.
    stopover_matched_itins = []
    stopover_iata = None
    stopover_filter_payload: Optional[Dict[str, Any]] = None
    if intent.stopover_city:
        stopover_text = intent.stopover_city
        stopover_match_strategy = None
        stopover_match_reason = None
        try:
            stopover_iata = _resolve_city_to_iata(stopover_text)
        except Exception:
            stopover_iata = None

        # Filter flights by layover_airports
        if stopover_iata:
            stopover_matched_itins = [f for f in all_flights if stopover_iata in (f.layover_airports or [])]
            stopover_match_strategy = "iata_layover_airports"
            if stopover_matched_itins:
                stopover_match_reason = "matched_resolved_stopover_iata"
            else:
                stopover_match_reason = "resolved_stopover_iata_not_present_in_layovers"
        else:
            # fallback to substring match in layover_info
            lower_v = stopover_text.lower()
            stopover_matched_itins = [f for f in all_flights if lower_v in (f.layover_info or "").lower()]
            stopover_match_strategy = "layover_info_substring"
            if stopover_matched_itins:
                stopover_match_reason = "matched_stopover_text_in_layover_info"
            else:
                stopover_match_reason = "unresolved_stopover_city"

        # Update debug info
        debug_info_stopover = {
            "requested": stopover_text,
            "resolved_iata": stopover_iata,
            "matched_count": len(stopover_matched_itins),
            "match_strategy": stopover_match_strategy,
            "match_reason": stopover_match_reason,
        }
        stopover_filter_payload = debug_info_stopover
        debug_info = debug_info or {}
        debug_info["stopover"] = debug_info_stopover
        debug_info["stopover_filter"] = debug_info_stopover
        logger.info("Stopover filter applied", extra=debug_info_stopover)

        # If we have matches, we may want to prioritize them in ranking, but we'll handle in prompt enrichment
        # For now, we keep filtered list unchanged but note the matches.
        # We'll later use stopover_matched_itins to enrich prompt.

    # If strict filtering yields nothing, apply staged relaxation in a safe priority order.
    if not filtered:
        logger.warning("No flights after strict filtering; applying staged relaxation fallback")
        relaxed_intent = intent.model_copy(deep=True)

        def _snapshot_relaxed_state() -> Dict[str, Any]:
            return {
                "preferred_airlines": list(relaxed_intent.preferred_airlines or []),
                "layover_limit_minutes": relaxed_intent.layover_limit_minutes,
                "price_limit": relaxed_intent.price_limit,
                "time_pref": relaxed_intent.time_pref,
                "wants_direct": relaxed_intent.wants_direct,
                "baggage_pref": relaxed_intent.baggage_pref,
            }

        def _try_relaxed(step: str) -> bool:
            nonlocal filtered, effective_intent, filtered_count
            candidate, candidate_warnings = filter_flights(all_flights, relaxed_intent)
            warnings.extend(candidate_warnings)
            matched_count = len(candidate) if candidate else 0
            relaxation_attempts.append({
                "step": step,
                "matched_count": matched_count,
                "state": _snapshot_relaxed_state(),
            })
            if candidate:
                filtered = candidate
                effective_intent = relaxed_intent.model_copy(deep=True)
                filtered_count = len(filtered)
                return True
            return False

        # 1) Remove preferred airline constraint
        if not filtered and relaxed_intent.preferred_airlines:
            preferred = ", ".join(relaxed_intent.preferred_airlines)
            relaxed_intent.preferred_airlines = []
            warnings.append(
                f"No exact flights were available for preferred airline(s) {preferred}; "
                "relaxed airline preference."
            )
            _try_relaxed("remove_preferred_airline")

        # 2) Remove layover duration constraint
        if not filtered and relaxed_intent.layover_limit_minutes:
            prev_layover = relaxed_intent.layover_limit_minutes
            relaxed_intent.layover_limit_minutes = None
            warnings.append(
                f"No flights met the max layover limit ({prev_layover} minutes); "
                "relaxed layover-duration constraint."
            )
            _try_relaxed("remove_layover_limit")

        # 3) Increase price cap by 25–50%
        if not filtered and relaxed_intent.price_limit:
            original_cap = relaxed_intent.price_limit
            for factor in (1.25, 1.50):
                new_cap = max(int(original_cap * factor), original_cap + 1)
                if new_cap <= relaxed_intent.price_limit:
                    continue
                relaxed_intent.price_limit = new_cap
                warnings.append(
                    f"No flights were found under ₹{original_cap}. "
                    f"Increased budget cap to ₹{new_cap} and retried."
                )
                if _try_relaxed(f"increase_price_cap_{int(factor*100)}pct"):
                    break

        # 4) Increase allowed duration (broaden departure time window if set)
        if not filtered and relaxed_intent.time_pref:
            prev_time_pref = relaxed_intent.time_pref
            relaxed_intent.time_pref = None
            warnings.append(
                f"No flights matched the {prev_time_pref} departure window; "
                "expanded the allowed time window."
            )
            _try_relaxed("remove_time_window")

        # 5) Remove direct-only requirement
        if not filtered and relaxed_intent.wants_direct:
            relaxed_intent.wants_direct = False
            warnings.append("No direct flights matched all constraints; allowing connecting options.")
            _try_relaxed("remove_direct_only")

        # Existing behavior retained: relax baggage filter if still empty.
        if not filtered and relaxed_intent.baggage_pref:
            prev_baggage = relaxed_intent.baggage_pref
            relaxed_intent.baggage_pref = None
            warnings.append(
                f"No flights matched the {prev_baggage} baggage preference; "
                "relaxed baggage constraint."
            )
            _try_relaxed("remove_baggage_filter")

        # 6) Final fallback: choose cheapest available flights.
        if not filtered:
            effective_intent = relaxed_intent.model_copy(deep=True)
            effective_intent.flight_pref = "cheapest"
            ranking_intent = effective_intent.model_copy(deep=True)

            filtered = sorted(all_flights, key=lambda f: price_to_int(f.price_inr))
            filtered_count = len(filtered)
            relaxation_attempts.append({
                "step": "fallback_cheapest_available",
                "matched_count": filtered_count,
                "state": _snapshot_relaxed_state(),
            })

            if filtered:
                cheapest_price = price_to_int(filtered[0].price_inr)
                if intent.price_limit and cheapest_price > intent.price_limit:
                    warnings.append(
                        f"No flights were found under ₹{intent.price_limit}. "
                        f"The cheapest available option is ₹{cheapest_price}."
                    )
                else:
                    warnings.append(
                        "No exact flights were found with current constraints; "
                        "showing the cheapest available option."
                    )

    if not filtered:
        # Even all_flights was empty, but we already handled that earlier
        return {
            "error": "Sorry, I couldn't find any flights matching your preferences.",
            "failure_reason": "no_flights",
            "no_flights_reason": "filtered_out",
            "search_date": search_date,
            "flight_counts": {
                "pre_filter": pre_filter_count,
                "post_filter": 0,
                "filtered_out": max(pre_filter_count, 0),
            },
            "debug_info": {"phases": phases.copy(), "tool_search_meta": tool_search_meta},
        }

    ranked = rank_flights(filtered, ranking_intent)
    best_flight = ranked[0]
    ranked_count = len(ranked)
    post_filter_count = ranked_count
    phases['filter_rank'] = time.monotonic() - start

    # ------------------------------------------------------------------
    # Resolve booking handoff per top-N ranked flights and keep best-flight summary.
    # ------------------------------------------------------------------
    booking_handoff_info: Dict[str, Any] = {
        "source": "unavailable",
        "reason": "not_attempted",
        "status": "unavailable",
    }
    top_flights_payload: List[Dict[str, Any]] = []
    booking_url: Optional[str] = None

    top_ranked = ranked[:PER_FLIGHT_HANDOFF_LIMIT]
    top_signal_max = max((_booking_handoff_candidate_signal(f) for f in top_ranked), default=0)
    weak_route_confidence = bool((search_profile or {}).get("weak_route_confidence"))
    round_trip_low_signal = bool(intent.return_date and top_signal_max < WEAK_ROUTE_HANDOFF_SIGNAL_THRESHOLD)
    weak_signal_candidates = bool(
        top_signal_max <= 0
        or (top_signal_max < WEAK_ROUTE_HANDOFF_SIGNAL_THRESHOLD and (weak_route_confidence or bool(intent.return_date)))
    )
    probe_limit = PER_FLIGHT_HANDOFF_PROBE_LIMIT
    if intent.return_date:
        probe_limit += ROUND_TRIP_HANDOFF_PROBE_BONUS
    if weak_signal_candidates:
        probe_limit += WEAK_ROUTE_HANDOFF_PROBE_BONUS
    if round_trip_low_signal:
        # Small extra probe on weak round-trip signal to avoid early fallback lock-in.
        probe_limit += 1
    probe_limit = max(len(top_ranked), min(PER_FLIGHT_HANDOFF_PROBE_MAX, probe_limit))
    scan_limit = min(len(ranked), max(probe_limit, PER_FLIGHT_HANDOFF_SCAN_LIMIT))
    if round_trip_low_signal:
        scan_limit = min(len(ranked), scan_limit + WEAK_ROUTE_ROUND_TRIP_SCAN_BONUS)

    probe_entries: List[Tuple[int, Any, int]] = [
        (idx, ranked[idx], _booking_handoff_candidate_signal(ranked[idx]))
        for idx in range(len(top_ranked))
    ]
    if scan_limit > len(top_ranked):
        extra_entries: List[Tuple[int, Any, int]] = [
            (idx, ranked[idx], _booking_handoff_candidate_signal(ranked[idx]))
            for idx in range(len(top_ranked), scan_limit)
        ]
        # Prefer artifact-rich candidates while keeping rank proximity as tiebreaker.
        extra_entries.sort(key=lambda item: (item[2], -item[0]), reverse=True)
        for entry in extra_entries:
            if len(probe_entries) >= probe_limit:
                break
            probe_entries.append(entry)

    probe_ranked = [entry[1] for entry in probe_entries]
    per_flight_handoff_timeout = PER_FLIGHT_HANDOFF_TIMEOUT + (
        ROUND_TRIP_HANDOFF_TIMEOUT_BONUS if intent.return_date else 0.0
    )
    booking_stage_started = time.monotonic()
    if probe_ranked:
        cache_snapshot = _booking_handoff_cache_snapshot()
        logger.info(
            "booking_handoff_stage_started",
            extra={
                "flight_candidates": len(top_ranked),
                "probe_candidates": len(probe_ranked),
                "probe_limit": probe_limit,
                "scan_limit": scan_limit,
                "top_signal_max": top_signal_max,
                "weak_route_confidence": weak_route_confidence,
                "weak_signal_candidates": weak_signal_candidates,
                "is_round_trip": bool(intent.return_date),
                "per_flight_timeout_sec": per_flight_handoff_timeout,
                **cache_snapshot,
            },
        )
        resolved_handoffs = await asyncio.gather(
            *[
                _resolve_flight_booking_handoff(
                    flight_obj=flight,
                    origin=intent.origin_iata,
                    destination=intent.destination_iata,
                    depart_date=search_date,
                    return_date=intent.return_date,
                    timeout_sec=per_flight_handoff_timeout,
                    candidate_rank=ranked_idx + 1,
                    probe_signal=signal,
                    route_type="round_trip" if intent.return_date else "one_way",
                    cache_mode_hint=cache_snapshot.get("cache_mode"),
                )
                for ranked_idx, flight, signal in probe_entries
            ]
        )

        promotion_pool: List[Dict[str, Any]] = []
        for idx, (resolved_flight, handoff_meta, resolved_url) in enumerate(resolved_handoffs):
            rank = probe_entries[idx][0] + 1
            promotion_pool.append(
                {
                    "rank": rank,
                    "booking_handoff": handoff_meta,
                    "handoff_url": resolved_url,
                    "is_in_top_payload": idx < len(top_ranked),
                }
            )
            if idx >= len(top_ranked):
                continue
            payload = (
                resolved_flight.model_dump()
                if hasattr(resolved_flight, "model_dump")
                else dict(resolved_flight)
            )
            payload["booking_handoff"] = handoff_meta
            payload["rank"] = rank
            top_flights_payload.append(payload)

            if idx == 0:
                if hasattr(resolved_flight, "model_dump"):
                    best_flight = resolved_flight
                booking_handoff_info = handoff_meta
                booking_url = resolved_url or payload.get("handoff_url")

        # Promote best booking-quality exit to top-level handoff summary even if
        # fare-ranked row 1 is a weaker fallback.
        if promotion_pool:
            scored: List[Tuple[int, int]] = []
            for i, payload in enumerate(promotion_pool):
                score = _booking_handoff_strength(payload.get("booking_handoff") or {})
                scored.append((score, i))
            best_handoff_idx = max(scored, key=lambda x: (x[0], -x[1]))[1]
            promoted_payload = promotion_pool[best_handoff_idx]
            promoted_meta = promoted_payload.get("booking_handoff") or {}
            if isinstance(promoted_meta, dict):
                booking_handoff_info = dict(promoted_meta)
                selected_rank = promoted_payload.get("rank", best_handoff_idx + 1)
                if selected_rank != 1:
                    booking_handoff_info["selected_flight_rank"] = selected_rank
                booking_url = promoted_payload.get("handoff_url") or booking_url

            bucket_counts: Dict[str, int] = {}
            for payload in promotion_pool:
                bucket = _booking_handoff_bucket(payload.get("booking_handoff") or {})
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            has_booking_ready = any(
                bool((payload.get("booking_handoff") or {}).get("is_booking_quality_exit"))
                for payload in promotion_pool
            )
            cache_assisted_successes = sum(
                1
                for payload in promotion_pool
                if str((payload.get("booking_handoff") or {}).get("reason") or "").endswith("_cache")
            )
            search_fallback_quality_counts: Dict[str, int] = {}
            best_search_fallback_rank: Optional[int] = None
            best_search_fallback_score = -1
            probe_signals: List[int] = [entry[2] for entry in probe_entries]
            positive_signal_candidates = sum(1 for sig in probe_signals if sig > 0)
            artifact_signal_candidates = sum(
                1 for sig in probe_signals if sig >= WEAK_ROUTE_HANDOFF_SIGNAL_THRESHOLD
            )
            max_probe_signal = max(probe_signals, default=0)
            for payload in promotion_pool:
                meta = payload.get("booking_handoff") or {}
                if not isinstance(meta, dict) or not meta.get("is_search_fallback"):
                    continue
                quality = str(meta.get("search_fallback_quality") or "generic")
                search_fallback_quality_counts[quality] = search_fallback_quality_counts.get(quality, 0) + 1
                score = _booking_handoff_strength(meta)
                if score > best_search_fallback_score:
                    best_search_fallback_score = score
                    best_search_fallback_rank = int(payload.get("rank") or 1)
            if isinstance(booking_handoff_info, dict) and booking_handoff_info.get("is_search_fallback"):
                booking_handoff_info["search_fallback_pool"] = {
                    "fallback_candidates": sum(search_fallback_quality_counts.values()),
                    "quality_counts": search_fallback_quality_counts,
                    "best_fallback_rank": best_search_fallback_rank or 1,
                    "probed_candidates": len(probe_entries),
                    "positive_signal_candidates": positive_signal_candidates,
                    "artifact_signal_candidates": artifact_signal_candidates,
                    "max_probe_signal": max_probe_signal,
                    "signal_threshold": WEAK_ROUTE_HANDOFF_SIGNAL_THRESHOLD,
                }
            logger.info(
                "booking_handoff_stage_completed",
                extra={
                    "flight_candidates": len(top_ranked),
                    "probe_candidates": len(probe_ranked),
                    "probe_limit": probe_limit,
                    "scan_limit": scan_limit,
                    "top_signal_max": top_signal_max,
                    "weak_route_confidence": weak_route_confidence,
                    "weak_signal_candidates": weak_signal_candidates,
                    "is_round_trip": bool(intent.return_date),
                    "duration_ms": int((time.monotonic() - booking_stage_started) * 1000),
                    "bucket_counts": bucket_counts,
                    "has_booking_ready": has_booking_ready,
                    "cache_assisted_successes": cache_assisted_successes,
                    "search_fallback_quality_counts": search_fallback_quality_counts,
                    "selected_primary_rank": booking_handoff_info.get("selected_flight_rank", 1),
                    **cache_snapshot,
                },
            )

        # Preserve ranked ordering while carrying handoff URLs for top-N flights.
        if top_flights_payload:
            updated_ranked = []
            for idx, original in enumerate(ranked):
                if idx < len(top_ranked):
                    updated_ranked.append(resolved_handoffs[idx][0])
                else:
                    updated_ranked.append(original)
            ranked = updated_ranked

    if not top_flights_payload and hasattr(best_flight, "model_dump"):
        # Defensive fallback: should be rare, but keep best-flight handoff summary intact.
        resolved_best, booking_handoff_info, booking_url = await _resolve_flight_booking_handoff(
            flight_obj=best_flight,
            origin=intent.origin_iata,
            destination=intent.destination_iata,
            depart_date=search_date,
            return_date=intent.return_date,
            timeout_sec=PER_FLIGHT_HANDOFF_TIMEOUT,
            candidate_rank=1,
            probe_signal=_booking_handoff_candidate_signal(best_flight),
            route_type="round_trip" if intent.return_date else "one_way",
            cache_mode_hint=_booking_handoff_cache_snapshot().get("cache_mode"),
        )
        if hasattr(resolved_best, "model_dump"):
            best_flight = resolved_best
            top_flights_payload = [
                {
                    **best_flight.model_dump(),
                    "booking_handoff": booking_handoff_info,
                    "rank": 1,
                }
            ]
        else:
            top_flights_payload = []
            booking_url = None
            booking_handoff_info = {
                "source": "unavailable",
                "reason": "booking_handoff_unresolved",
                "status": "unavailable",
            }

    aligned_handoff_meta, aligned_handoff_url, handoff_aligned = _align_top_level_booking_handoff_with_rows(
        booking_handoff_info,
        top_flights_payload,
    )
    if handoff_aligned:
        previous_bucket = _booking_handoff_bucket(booking_handoff_info or {})
        aligned_bucket = _booking_handoff_bucket(aligned_handoff_meta or {})
        booking_handoff_info = aligned_handoff_meta
        if aligned_handoff_url:
            booking_url = aligned_handoff_url
        logger.info(
            "booking_handoff_top_level_aligned_with_per_flight_quality",
            extra={
                "previous_bucket": previous_bucket,
                "aligned_bucket": aligned_bucket,
                "selected_flight_rank": booking_handoff_info.get("selected_flight_rank", 1),
            },
        )

    # ------------------------------------------------------------------
    # 5. Build description for LLM using effective_intent
    # ------------------------------------------------------------------
    filter_parts = []
    if effective_intent.time_pref: filter_parts.append(f"{effective_intent.time_pref} flights")
    if effective_intent.price_limit: filter_parts.append(f"under ₹{effective_intent.price_limit}")
    if effective_intent.wants_direct: filter_parts.append("direct flights only")
    # Always include original airline preference so LLM knows what was asked, even if relaxed
    if intent.preferred_airlines:
        filter_parts.append(f"preferred airlines: {', '.join(intent.preferred_airlines)}")
    elif effective_intent.preferred_airlines:
        filter_parts.append(f"preferred airlines: {', '.join(effective_intent.preferred_airlines)}")
    if effective_intent.layover_limit_minutes: filter_parts.append(f"max layover: {effective_intent.layover_limit_minutes//60}h")
    if effective_intent.baggage_pref: filter_parts.append(f"{effective_intent.baggage_pref} baggage only")
    if effective_intent.wants_eco:
        filter_parts.append("eco-friendly / lowest carbon emissions")
    # Include flight preference (cheapest/shortest/balanced) in filters
    if effective_intent.flight_pref == "cheapest":
        filter_parts.append("cheapest / lowest price")
    elif effective_intent.flight_pref == "shortest":
        filter_parts.append("shortest flight duration")
    elif effective_intent.flight_pref == "balanced":
        filter_parts.append("balanced price and duration")
    # NEW: deep search flag
    if intent.deep_search:
        filter_parts.append("deep search for absolute cheapest price")
    filters_applied = "; ".join(filter_parts) if filter_parts else "no specific filters"

    trip_description = f"a {intent.trip_type} trip"
    if intent.stopover_city:
        trip_description += f" via {intent.stopover_city}"
    if intent.return_date:
        trip_description += f", returning on {intent.return_date}"

    # ------------------------------------------------------------------
    # 5a. Enrich trip_description with stopover filter details (if any)
    # ------------------------------------------------------------------
    if intent.stopover_city and stopover_matched_itins is not None:
        if stopover_matched_itins:
            # Use top matched flight (first in filtered list? or from stopover_matched_itins)
            top = stopover_matched_itins[0]
            layover_text = top.layover_info or "No layover details available"
            # Build a small leg table if possible (but Flight may not have legs; use layover_info)
            leg_lines = []
            # Since we don't have full legs in Flight, we'll rely on layover_info.
            trip_description += (
                f"\n\nSTOPOVER REQUEST: {intent.stopover_city} (resolved: {stopover_iata})\n"
                f"Top matched itinerary layover(s): {layover_text}\n"
                "Please explain – in clear, helpful detail (about 80–120 words) – whether this itinerary satisfies the user's 'via' request; "
                "if not, explain why and suggest concrete alternatives (e.g., change stopover to X, allow 1 stop, or search nearby airports)."
            )
        else:
            # No matches: give the LLM the top few candidate flights and instruct it to explain failure and alternatives
            sample_lines = []
            for idx, f in enumerate(filtered[:3]):
                sample_lines.append(
                    f"{idx+1}. {f.airline} {f.flight_no} {f.departure_time}->{f.arrival_time} duration {f.duration_min}m layovers: {f.layover_airports or '[]'}"
                )
            trip_description += (
                f"\n\nUser asked for flights 'via {intent.stopover_city}', but no flights in the returned results stop via {stopover_text}.\n"
                "Here are the top 3 candidate flights returned by the search:\n" + "\n".join(sample_lines) + "\n"
                "Please write a clear, helpful explanation (about 80–120 words) that says no flights matched the requested stopover, why that may be (e.g., no itineraries with that layover in results), and suggest concrete alternatives the user can try (e.g., search for flights with 1 stop, allow a different via city, or change date)."
            )

    # ------------------------------------------------------------------
    # 6. Convert weather to dict for consistent serialization (safe version)
    # ------------------------------------------------------------------
    if hasattr(weather_data, "model_dump"):
        weather_dict = weather_data.model_dump()
    elif hasattr(weather_data, "__dict__"):
        weather_dict = dict(vars(weather_data))
    elif isinstance(weather_data, dict):
        weather_dict = weather_data
    else:
        # Last-resort safe serialization
        weather_dict = json.loads(json.dumps(weather_data, default=str))

    # ------------------------------------------------------------------
    # 7. Prepare debug_info with extra data needed for streaming (ALWAYS populated)
    # ------------------------------------------------------------------
    # Parse price_insights once here so both non-streaming and streaming paths can use it
    _price_insights_obj = None
    price_insights_str = ""
    price_analysis_str = ""
    price_prediction_str = ""
    if price_insights_raw:
        try:
            # Existing formatting
            _price_insights_obj = _parse_price_insights_safe(
                {"price_insights": price_insights_raw},
                current_price_inr=price_to_int(best_flight.price_inr),
            )
            if _price_insights_obj:
                price_insights_str = _format_price_insights_for_llm_safe(_price_insights_obj)

            # NEW: price analysis and prediction
            price_history = price_insights_raw.get("price_history", [])
            if price_history:
                analysis = _analyze_price_trend_safe(price_history)
                if analysis:
                    price_analysis_str = (
                        f"Price trend: {analysis.get('trend', 'stable')}. "
                        f"Average price for this route: ₹{analysis.get('average_price', 'N/A')}. "
                        f"Current price is {analysis.get('price_level', 'average')}."
                    )
                prediction = _predict_future_price_safe(price_history)
                if prediction:
                    price_prediction_str = (
                        f"Price prediction: {prediction.get('trend', 'prices may change')}. "
                        f"{prediction.get('advice', '')}"
                    )
        except Exception as _pi_err:
            logger.warning("price_insights formatting failed", extra={"error": str(_pi_err)})

    # ------------------------------------------------------------------
    # Add API trace to debug_info (now defined)
    # ------------------------------------------------------------------
    debug_info = debug_info or {}
    origin_route = _iata_city_label(intent.origin_iata)
    destination_route = _iata_city_label(intent.destination_iata)
    debug_info.update({
        "phases": phases.copy(),
        "intent": intent.model_dump(),
        "effective_intent": effective_intent.model_dump(),
        "requested_trip_mode": requested_trip_mode,
        "route_labels": {
            "origin_iata": origin_route["iata"],
            "origin_city": origin_route["city"],
            "origin_label": origin_route["label"],
            "destination_iata": destination_route["iata"],
            "destination_city": destination_route["city"],
            "destination_label": destination_route["label"],
        },
        "filters_applied": filters_applied,
        "trip_description": trip_description,
        "all_flights": [f.model_dump() for f in all_flights],
        "top_flights": top_flights_payload,
        "per_flight_handoff_limit": PER_FLIGHT_HANDOFF_LIMIT,
        "flight_counts": {
            "pre_filter": pre_filter_count,
            "post_filter": post_filter_count,
            "filtered_out": max(pre_filter_count - post_filter_count, 0),
            "raw_provider": int(tool_search_meta.get("raw_candidate_count") or pre_filter_count),
        },
        "tool_search_meta": tool_search_meta,
        "filtered_count": filtered_count,
        "ranked_count": ranked_count,
        "price_insights_str": price_insights_str,
        "price_analysis_str": price_analysis_str,
        "price_prediction_str": price_prediction_str,
        "booking_handoff": booking_handoff_info,
        "normalization": normalization_debug,
        "relaxation_attempts": relaxation_attempts,
        # NEW — full API traceability including raw responses
        "api_trace": {
            "flight": {
                "request": {
                    "departure": intent.origin_iata,
                    "arrival": intent.destination_iata,
                    "date": search_date,
                    "intent_date": intent.date,
                    "return_date": intent.return_date if intent.return_date else None,
                },
                "raw_count": len(all_flights),
                "filtered_count": filtered_count,
                "best_flight_no": best_flight.flight_no,
                "raw_response": _raw_flights_before_normalize,
            },
            "weather": {
                "request": {
                    "location": intent.destination_iata,
                    "date": search_date,
                },
                "forecast_date": weather_dict.get("forecast_date"),
                "condition": weather_dict.get("condition"),
                "temperature_c": weather_dict.get("temperature_c"),
                "raw_response": weather_dict,
            },
        },
    })

    # Log a stripped api_trace (no raw_response, no booking tokens) for clean log diffs
    _log_trace = {
        "flight": {
            "request": debug_info["api_trace"]["flight"]["request"],
            "raw_count": debug_info["api_trace"]["flight"]["raw_count"],
            "filtered_count": debug_info["api_trace"]["flight"]["filtered_count"],
            "best_flight_no": debug_info["api_trace"]["flight"]["best_flight_no"],
        },
        "weather": {
            "request": debug_info["api_trace"]["weather"]["request"],
            "forecast_date": debug_info["api_trace"]["weather"]["forecast_date"],
            "condition": debug_info["api_trace"]["weather"]["condition"],
            "temperature_c": debug_info["api_trace"]["weather"]["temperature_c"],
        }
    }
    logger.debug("api_trace (stripped): %s", _log_trace)

    # Build return trip data for LLM prompt injection (no recursive call needed)
    return_trip_result = None
    return_flight_data = None
    return_weather_data = None
    rw_dict = {}  # ensure rw_dict is always defined for later use
    if return_flight_result and not isinstance(return_flight_result, Exception):
        raw_rt_flights = return_flight_result[0] if isinstance(return_flight_result, tuple) else return_flight_result
        rt_flights_norm = normalize_flights(raw_rt_flights, return_date_str)
        if rt_flights_norm:
            # Rank return flights using the same intent (original preferences)
            rt_ranked = rank_flights(rt_flights_norm, intent)
            if rt_ranked:
                return_flight_data = rt_ranked[0]
                return_weather_data = weather_ret if weather_ret and not isinstance(weather_ret, Exception) else {}
                # Convert return weather to dict safely
                if isinstance(return_weather_data, dict):
                    rw_dict = return_weather_data
                elif hasattr(return_weather_data, "model_dump"):
                    rw_dict = return_weather_data.model_dump()
                elif hasattr(return_weather_data, "__dict__"):
                    rw_dict = dict(vars(return_weather_data))
                else:
                    rw_dict = {}
                # Build return trip PlanResult for the response payload (no LLM yet — will be filled later)
                return_trip_result = PlanResult(
                    llm_response=None,   # filled in by combined LLM below
                    best_flight=return_flight_data.model_dump(),
                    weather=rw_dict,
                    search_date=return_date_str,
                    fallback_note="",
                    debug_info={"phases": {}},
                    warnings=None,
                    weather_present=weather_present_ret,
                    weather_reason=weather_reason_ret,
                )
                # Enhanced return leg injection with MANDATORY instruction and brevity request
                trip_description += (
                    f"\n\nRETURN FLIGHT (on {return_date_str}):\n"
                    f"- Flight: {return_flight_data.airline} {return_flight_data.flight_no} | "
                    f"{return_flight_data.departure_time} → {return_flight_data.arrival_time} | "
                    f"Price: {return_flight_data.price_inr} | Stops: {return_flight_data.stops}\n"
                    f"- Weather at {intent.origin_iata} on {return_date_str}: "
                    f"{rw_dict.get('condition')}, {rw_dict.get('temperature_c')}°C\n\n"
                    f"MANDATORY: Your response MUST cover BOTH legs but you must be EXTREMELY BRIEF.\n"
                    f"CRITICAL RULE: Your ENTIRE response must be 4 sentences maximum. Do not write long paragraphs."
                )
                # Add TEMP RULE for return leg to prevent temperature inversion
                rw_t_min = rw_dict.get("temp_min_c")
                rw_t_max = rw_dict.get("temp_max_c")
                if rw_t_min is not None and rw_t_max is not None and rw_t_max > rw_t_min:
                    trip_description += (
                        f"\nTEMP RULE FOR RETURN ({intent.origin_iata}): "
                        f"Daily low={rw_t_min}°C, daily high={rw_t_max}°C. "
                        f"NEVER describe {rw_t_max}°C as the 'low' temperature in the return section."
                    )

    if skip_llm:
        # Provide a clean summary when LLM is skipped
        llm_text = f"Flight: {best_flight.airline} {best_flight.flight_no} ({best_flight.departure_time} - {best_flight.arrival_time}). Price: {best_flight.price_inr}. Weather: {weather_dict.get('condition')}, {weather_dict.get('temperature_c')}°C."
        llm_text = _ensure_route_grounding(llm_text, intent.origin_iata, intent.destination_iata)
        llm_degradation = None
        llm_degradation_note = ""
    else:
        start = time.monotonic()
        llm_text, llm_degradation = await generate_explanation(
            user_query=user_query,
            intent=intent,
            best_flight=best_flight,
            weather=weather_dict,
            all_flights=all_flights,
            filters_applied=filters_applied,
            trip_description=trip_description,
            warnings=warnings,
            price_insights_str=price_insights_str,
            price_analysis_str=price_analysis_str,
            price_prediction_str=price_prediction_str,
            booking_url=booking_url,  # NEW: pass booking URL if available
        )
        phases['llm_generation'] = time.monotonic() - start
        debug_info["phases"] = phases.copy()
        llm_degradation_note = ""
        if llm_degradation:
            llm_degradation_note = _explanation_degradation_note(
                llm_degradation.get("reason"),
                llm_degradation.get("message"),
            )
            warnings.append(llm_degradation_note)
            debug_info["degradation"] = llm_degradation
        llm_text = _ensure_route_grounding(llm_text, intent.origin_iata, intent.destination_iata)

        # No second LLM call — the combined trip_description already covers both legs.
        # Split llm_text at the RETURN section to give return_trip its own slice.
        if return_trip_result and return_flight_data:
            rt_split = re.search(r'\bRETURN\b', llm_text, re.IGNORECASE)
            if rt_split and rt_split.start() > 50:
                return_trip_result.llm_response = llm_text[rt_split.start():].strip()
            else:
                # LLM ignored the return section — build a deterministic return summary
                # so return_trip.llm_response is always meaningful and validator-compliant
                rw_cond = rw_dict.get('condition', 'N/A')
                rw_temp = rw_dict.get('temperature_c', 'N/A')
                return_trip_result.llm_response = (
                    f"Return flight on {return_date_str}: "
                    f"{return_flight_data.airline} {return_flight_data.flight_no} | "
                    f"{return_flight_data.departure_time} → {return_flight_data.arrival_time} | "
                    f"Price: {return_flight_data.price_inr} | Stops: {return_flight_data.stops}. "
                    f"Weather at {intent.origin_iata} on {return_date_str}: {rw_cond}, {rw_temp}°C. "
                    f"Pack accordingly for your return journey."
                )

    # ------------------------------------------------------------------
    # 9. Prepare result and handle round-trip
    # ------------------------------------------------------------------
    # Ensure debug_info always contains price_insights_str (Fix #5)
    result_debug = debug_info or {}
    result_debug["price_insights_str"] = price_insights_str or ""
    result_debug["price_analysis_str"] = price_analysis_str or ""
    result_debug["price_prediction_str"] = price_prediction_str or ""

    result = PlanResult(
        llm_response=llm_text,
        best_flight=best_flight.model_dump(),
        weather=weather_dict,
        search_date=search_date,
        fallback_note=llm_degradation_note if llm_degradation else "",
        debug_info=result_debug,
        warnings=warnings if warnings else None,
        return_trip=return_trip_result if isinstance(return_trip_result, PlanResult) else None,
        weather_present=weather_present_out,
        weather_reason=weather_reason_out,
        flight_counts={
            "pre_filter": pre_filter_count,
            "post_filter": post_filter_count,
            "filtered_out": max(pre_filter_count - post_filter_count, 0),
            "raw_provider": int(tool_search_meta.get("raw_candidate_count") or pre_filter_count),
        },
        stopover_filter=stopover_filter_payload,
        result_status="degraded" if llm_degradation else "success",
        degradation=llm_degradation,
        booking_handoff=booking_handoff_info,
        top_flights=top_flights_payload,
    )

    # ------------------------------------------------------------------
    # 10. Log session (only if not skipping LLM)
    # ------------------------------------------------------------------
    if DB_AVAILABLE and not skip_llm:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    save_session,
                    user_query=user_query,
                    agent_reasoning={
                        "version": "planner-v7-normalized",
                        "intent": intent.model_dump(),
                        "effective_intent": effective_intent.model_dump(),
                        "filters_applied": filters_applied,
                        "ranked_count": len(ranked),
                        "flight_pref": intent.flight_pref,
                        "trip_type": intent.trip_type,
                        "use_cloud_llm": USE_CLOUD_FALLBACK,
                        "phases": phases,
                        "warnings": warnings
                    },
                    tool_output={
                        "all_flights_count": len(all_flights),
                        "filtered_count": filtered_count,
                        "weather": weather_dict,
                        "origin": intent.origin_iata,
                        "destination": intent.destination_iata,
                        "search_date": search_date
                    },
                    final_response=llm_text or ""
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.error("Database write timed out")

    # ------------------------------------------------------------------
    # 11. Final safety: ensure llm_response exists when not skipped
    # ------------------------------------------------------------------
    if not skip_llm and not result.llm_response:
        result.llm_response = "I found a flight matching your criteria, but the detailed explanation is currently unavailable."

    return result

# ----------------------------------------------------------------------
# Public entry point with streaming support
# ----------------------------------------------------------------------
async def plan_trip(
    *,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    date: Optional[str] = None,
    user_query: str,
    trip_type: Optional[str] = None,
    stream: bool = False,
    flights: Optional[List[Union[Dict, Flight]]] = None,
    depth: int = 0,
    flight_tool: Callable = default_flight_tool,
    weather_tool: Callable = default_weather_tool
) -> Union[PlanResult, MultiCityResult, Dict, AsyncGenerator[str, None]]:
    """
    Public entry point for planning a trip.

    If stream=False (default): returns a PlanResult (or MultiCityResult/error dict) with the full
    response, as in previous versions.

    If stream=True: returns an asynchronous generator that yields tokens as they are produced
    by the LLM. The generator will also produce a final JSON payload prefixed with "[DONE_JSON]"
    containing the structured result (best flight, weather, etc.). This mode is intended for
    SSE (Server‑Sent Events) or similar incremental delivery.

    Args:
        origin: IATA code (optional, overrides parsing)
        destination: IATA code (optional, overrides parsing)
        date: YYYY-MM-DD (optional, overrides parsing)
        user_query: Natural language query
        trip_type: Business/Holiday/Flexible/Urgent (optional, overrides parsing)
        stream: If True, return a token generator instead of a full PlanResult.
        flights: Pre-fetched flight list (for testing, can be dicts or Flight objects)
        depth: Recursion depth (internal use)
        flight_tool: Async function to search flights (for injection)
        weather_tool: Async function to fetch weather (for injection)

    Returns:
        - If stream=False: PlanResult, MultiCityResult, or error dict
        - If stream=True: AsyncGenerator[str, None]
    """
    try:
        llm_mode_hint, _ = await get_llm_mode_and_priority()
    except Exception:
        llm_mode_hint = "unknown"

    action = _detect_booking_or_tracking_action(user_query)

    async def _handle_action_intent() -> Dict[str, Any]:
        # Confirm / cancel are direct booking lifecycle actions and do not require route parsing.
        if action in ("confirm_booking", "cancel_booking"):
            booking_id = _extract_booking_id(user_query)
            if booking_id is None:
                return {
                    "error": "Please provide a numeric booking id to confirm/cancel.",
                    "action": action,
                }
            if action == "confirm_booking":
                try:
                    ok = await asyncio.to_thread(_confirm_booking_safe, booking_id)
                except Exception as e:
                    return {
                        "action": action,
                        "booking_id": booking_id,
                        "success": False,
                        "error": str(e),
                    }
                return {
                    "action": action,
                    "booking_id": booking_id,
                    "success": ok,
                    "message": "Booking confirmed." if ok else "Booking could not be confirmed.",
                }
            try:
                ok = await asyncio.to_thread(_cancel_booking_safe, booking_id)
            except Exception as e:
                return {
                    "action": action,
                    "booking_id": booking_id,
                    "success": False,
                    "error": str(e),
                }
            return {
                "action": action,
                "booking_id": booking_id,
                "success": ok,
                "message": "Booking cancelled." if ok else "Booking could not be cancelled.",
            }

        # Hold / track intents require selecting a flight first.
        planned = await _plan_trip_internal(
            origin=origin,
            destination=destination,
            date=date,
            user_query=user_query,
            trip_type=trip_type,
            flights=flights,
            depth=depth,
            flight_tool=flight_tool,
            weather_tool=weather_tool,
            skip_llm=True,
        )
        if isinstance(planned, dict):
            return planned
        if isinstance(planned, MultiCityResult):
            return {
                "error": "Booking actions currently support single-leg selections only.",
                "action": action,
            }

        best_flight = planned.best_flight or {}
        dbg = planned.debug_info or {}
        parsed_intent = dbg.get("intent", {}) if isinstance(dbg, dict) else {}
        resolved_origin = parsed_intent.get("origin_iata") or origin
        resolved_destination = parsed_intent.get("destination_iata") or destination
        depart_date = planned.search_date or date

        if not resolved_origin or not resolved_destination or not depart_date:
            return {
                "error": "Could not resolve route/date required for booking action.",
                "action": action,
            }

        default_hold = get_env_int("BOOKING_HOLD_MINUTES", 15)
        track_hold = get_env_int("PRICE_TRACK_HOLD_MINUTES", 43200)  # 30 days
        hold_minutes = track_hold if action == "track_price" else default_hold

        try:
            held = await _hold_booking_safe(
                flight=best_flight,
                origin=resolved_origin,
                destination=resolved_destination,
                depart_date=depart_date,
                hold_minutes=hold_minutes,
            )
        except Exception as e:
            return {
                "action": action,
                "success": False,
                "error": str(e),
            }

        if action == "track_price":
            # Seed a baseline snapshot so the monitor has an initial reference point.
            try:
                _record_price_snapshot_safe(
                    origin=resolved_origin,
                    destination=resolved_destination,
                    travel_date=depart_date,
                    price_inr=float(price_to_int(_flight_value_safe(best_flight, "price_inr", 0))),
                )
            except Exception as e:
                logger.warning("record_price_snapshot failed for tracking setup", extra={"error": str(e)})

            return {
                "action": action,
                "success": True,
                "message": "We will notify you if the price of this flight drops.",
                "booking": held,
                "best_flight": best_flight,
                "monitoring_active": True,
            }

        return {
            "action": action,
            "success": True,
            "message": "Flight held successfully.",
            "booking": held,
            "best_flight": best_flight,
        }

    if action:
        if stream:
            async def action_stream() -> AsyncGenerator[str, None]:
                payload = await _handle_action_intent()
                msg = payload.get("message") or payload.get("error") or "Action processed."
                yield msg
                metrics.record_stream_done_json("action")
                yield "[DONE_JSON]" + json.dumps(payload)
            return action_stream()
        return await _handle_action_intent()

    # Non‑streaming branch – existing behaviour
    if not stream:
        return await _plan_trip_internal(
            origin=origin,
            destination=destination,
            date=date,
            user_query=user_query,
            trip_type=trip_type,
            flights=flights,
            depth=depth,
            flight_tool=flight_tool,
            weather_tool=weather_tool,
            skip_llm=False
        )

    # --- Streaming branch ---
    async def stream_generator() -> AsyncGenerator[str, None]:
        stream_provider = "unknown"
        got_first_token = False
        data_result: Optional[Union[PlanResult, MultiCityResult, Dict[str, Any]]] = None

        def done_json_frame(payload: Dict[str, Any], status: str) -> str:
            metrics.record_stream_done_json(status)
            return "[DONE_JSON]" + json.dumps(payload)

        def _fallback_llm_text_from_structured_result() -> str:
            if isinstance(data_result, PlanResult):
                return str(data_result.llm_response or "").strip()
            return ""

        def _degraded_done_payload(
            *,
            reason: str,
            message: str,
            provider: Optional[str] = None,
            backend_status: Optional[Dict[str, Any]] = None,
            partial_llm_response: Optional[str] = None,
        ) -> Dict[str, Any]:
            degradation = _llm_degradation_payload(
                reason=reason,
                message=message,
                provider=provider,
                backend_status=backend_status,
            )
            if isinstance(data_result, PlanResult):
                payload = data_result.model_dump()
                degradation_note = _explanation_degradation_note(reason, message)
                warnings = list(payload.get("warnings") or [])
                warnings.append(degradation_note)
                payload["warnings"] = warnings
                payload["result_status"] = "degraded"
                payload["degradation"] = degradation
                payload["fallback_note"] = payload.get("fallback_note") or degradation_note
                if partial_llm_response:
                    payload["llm_response"] = partial_llm_response
                return payload

            if isinstance(data_result, MultiCityResult):
                payload = data_result.model_dump()
                degradation_note = _explanation_degradation_note(reason, message)
                payload["result_status"] = "degraded"
                payload["degradation"] = degradation
                payload["fallback_note"] = payload.get("fallback_note") or degradation_note
                for leg in payload.get("legs", []):
                    leg_warnings = list(leg.get("warnings") or [])
                    leg_warnings.append(degradation_note)
                    leg["warnings"] = leg_warnings
                    leg["result_status"] = "degraded"
                    leg["degradation"] = degradation
                    leg["fallback_note"] = leg.get("fallback_note") or degradation_note
                return payload

            return {
                "error": message,
                    "failure_reason": reason,
                    "failure_domain": _failure_domain_from_reason(reason),
                    "result_status": "error",
                }

        async def _emit_structured_fallback_chunk_if_available() -> None:
            fallback_text = _fallback_llm_text_from_structured_result()
            if fallback_text:
                yield_text = fallback_text if fallback_text.endswith(" ") else f"{fallback_text} "
                yield yield_text

        try:
            # Emit an immediate non-technical progress signal so clients can distinguish a healthy
            # stream from a stalled connection before LLM token generation begins.
            yield _sse_event("reasoning_step", {"step": "Gathering live flight options and destination weather."})

            # 1. Get all data without LLM explanation (skip_llm=True)
            data_result = await _plan_trip_internal(
                origin=origin,
                destination=destination,
                date=date,
                user_query=user_query,
                trip_type=trip_type,
                flights=flights,
                depth=depth,
                flight_tool=flight_tool,
                weather_tool=weather_tool,
                skip_llm=True
            )

            # Any dict payload at this stage is non-success for streaming.
            # Stream consumers must receive explicit error semantics, never a soft success.
            if isinstance(data_result, dict):
                planner_error = str(
                    data_result.get("error")
                    or data_result.get("warning")
                    or "Planner returned an incomplete response payload."
                ).strip()
                if not planner_error:
                    planner_error = "Planner returned an incomplete response payload."
                yield f"[ERROR] {planner_error}"
                done_payload: Dict[str, Any] = {"error": planner_error}
                for key in ("failure_reason", "no_flights_reason", "flight_counts", "search_date"):
                    if key in data_result:
                        done_payload[key] = data_result.get(key)
                if "failure_reason" not in done_payload:
                    done_payload["failure_reason"] = "planner_incomplete"
                done_payload["failure_domain"] = _failure_domain_from_reason(done_payload.get("failure_reason"))
                done_payload["result_status"] = "error"
                yield done_json_frame(done_payload, status="error")
                return

            if not isinstance(data_result, (PlanResult, MultiCityResult)):
                yield "[ERROR] Planner returned an unsupported response type."
                yield done_json_frame(
                    {
                        "error": "Planner returned an unsupported response type.",
                        "failure_reason": "planner_incomplete",
                        "failure_domain": _failure_domain_from_reason("planner_incomplete"),
                        "result_status": "error",
                    },
                    status="error",
                )
                return

            # 3. Extract data needed for prompt
            if isinstance(data_result, MultiCityResult):
                # For multi-city trips, we currently only stream a simple message and final JSON.
                # (Could be extended to stream per leg, but omitted for brevity.)
                yield "This is a multi-city trip. "
                final_json = data_result.model_dump()
                # Ensure llm_response is None for all legs
                for leg in final_json.get("legs", []):
                    leg["llm_response"] = None
                yield done_json_frame(final_json, status="success")
                return

            # 2. Check circuit breaker before calling LLM for single-leg explanation stream.
            if await check_llm_circuit(llm_mode=llm_mode_hint, effective_mode=llm_mode_hint):
                metrics.record_stream_failure("unknown")
                metrics.record_stream_fallback("circuit_open", "unknown")
                yield _sse_event(
                    "reasoning_step",
                    {"step": "Explanation generation degraded: LLM temporarily unavailable, returning structured flight and weather data."},
                )
                yield done_json_frame(
                    _degraded_done_payload(
                        reason="upstream_unavailable",
                        message="LLM circuit breaker is open; explanation stream skipped.",
                        provider="router",
                    ),
                    status="degraded",
                )
                return

            # Single leg
            best_flight = Flight(**data_result.best_flight)
            weather = data_result.weather
            # Warnings are at the top level of PlanResult
            warnings = data_result.warnings or []
            debug_info = data_result.debug_info or {}
            intent_dict = debug_info.get("intent", {})
            route_labels = debug_info.get("route_labels", {}) if isinstance(debug_info, dict) else {}
            # Use effective_intent if available for filters description
            effective_intent_dict = debug_info.get("effective_intent", intent_dict)
            all_flights_dicts = debug_info.get("all_flights", [])
            filters_applied = debug_info.get("filters_applied", "")
            trip_description = debug_info.get("trip_description", "")
            price_insights_str = debug_info.get("price_insights_str", "")
            price_analysis_str = debug_info.get("price_analysis_str", "")
            price_prediction_str = debug_info.get("price_prediction_str", "")

            # Emit structured hydration events as soon as data is available.
            if all_flights_dicts:
                yield _sse_event(
                    "flights",
                    {
                        "all_flights": all_flights_dicts,
                        "top_flights": data_result.top_flights or [],
                        "best_flight": data_result.best_flight,
                        "origin_iata": intent_dict.get("origin_iata"),
                        "destination_iata": intent_dict.get("destination_iata"),
                        "origin_city": route_labels.get("origin_city"),
                        "destination_city": route_labels.get("destination_city"),
                        "origin_label": route_labels.get("origin_label"),
                        "destination_label": route_labels.get("destination_label"),
                    },
                )

            if isinstance(weather, dict) and weather:
                weather_payload: Dict[str, Any] = {}
                for key, value in weather.items():
                    weather_payload[key] = value.value if hasattr(value, "value") else value
                destination_iata = _sanitize_iata_code(str(intent_dict.get("destination_iata") or ""))
                if destination_iata:
                    if not weather_payload.get("location"):
                        weather_payload["location"] = destination_iata
                    if not weather_payload.get("location_city"):
                        city = city_for_iata(destination_iata)
                        if city:
                            weather_payload["location_city"] = city
                    if not weather_payload.get("location_label"):
                        weather_payload["location_label"] = label_for_iata(destination_iata) or destination_iata
                yield _sse_event("weather", {"weather": weather_payload})

            # Emit progressive reasoning steps from known structured data.
            yield _sse_event(
                "reasoning_step",
                {"step": f"Ranked options and selected {best_flight.airline} {best_flight.flight_no} as the strongest overall fit."},
            )
            if best_flight.stops == 0:
                yield _sse_event(
                    "reasoning_step",
                    {"step": "Non-stop routing helped reduce transfer risk and overall travel complexity."},
                )
            else:
                yield _sse_event(
                    "reasoning_step",
                    {"step": f"Accepted {best_flight.stops} stop(s) to keep overall value and timing strong."},
                )

            weather_condition = weather.get("condition") if isinstance(weather, dict) else None
            weather_temp = weather.get("temperature_c") if isinstance(weather, dict) else None
            if weather_condition or weather_temp is not None:
                temp_text = ""
                if weather_temp is not None and str(weather_temp).strip() != "":
                    try:
                        temp_text = f" around {round(float(weather_temp), 1)}°C"
                    except Exception:
                        temp_text = f" around {weather_temp}°C"
                yield _sse_event(
                    "reasoning_step",
                    {"step": f"Destination weather ({intent_dict.get('destination_iata') or 'destination'}) looks {str(weather_condition).lower() if weather_condition else 'stable'}{temp_text}, so comfort and packing guidance were included."},
                )

            # Inject return leg details if present
            if data_result.return_trip:
                rt = data_result.return_trip
                rt_flight = rt.best_flight
                rt_weather = rt.weather
                trip_description += (
                    f"\n\nRETURN LEG (on {rt.search_date}):\n"
                    f"- Flight: {rt_flight.get('airline')} {rt_flight.get('flight_no')} | {rt_flight.get('departure_time')} → {rt_flight.get('arrival_time')} | Price: {rt_flight.get('price_inr')}\n"
                    f"- Weather: {rt_weather.get('condition')}, {rt_weather.get('temperature_c')}°C\n"
                    "Please summarize the round trip, mentioning both outbound and return flights and their respective weather."
                )

            # Build prompt — kept in sync with generate_explanation (non-streaming path)
            stream_max_flights_in_prompt = (
                PLANNER_LLM_MAX_FLIGHTS_ROUND_TRIP
                if intent_dict.get("return_date")
                else PLANNER_LLM_MAX_FLIGHTS_ONE_WAY
            )
            stream_min_flights = 2 if intent_dict.get("return_date") else 3
            stream_max_flights_in_prompt = max(stream_min_flights, stream_max_flights_in_prompt)
            stream_flights_in_prompt = max(
                1,
                min(stream_max_flights_in_prompt, len(all_flights_dicts)),
            )
            flights_str = "\n".join([
                f"- {f['airline']} {f['flight_no']} on {f.get('date','N/A')} | "
                f"{f['departure_time']} → {f['arrival_time']} | "
                f"{f['duration_min']} min | {f['price_inr']} | "
                f"Stops: {f.get('stops', 'N/A')} | Baggage: {f.get('baggage', 'N/A')}"
                for f in all_flights_dicts[:stream_flights_in_prompt]
            ])

            # Format warnings if present, escalating "relaxed" warnings to mandatory instructions
            if warnings:
                processed = []
                for w in warnings:
                    if "relaxed" in w.lower():
                        processed.append(
                            "MANDATORY: Your baggage or airline preference could not be matched. "
                            "You MUST explicitly tell the user their preference was not available and "
                            "you are showing the closest alternative. Do NOT claim the flight meets their baggage requirement."
                        )
                    else:
                        processed.append(w)
                warnings_str = "\nSystem Notes/Warnings:\n- " + "\n- ".join(processed)
            else:
                warnings_str = ""
            warnings_for_prompt = _truncate_for_prompt(warnings_str, PLANNER_LLM_WARNINGS_MAX_CHARS)
            trip_description_for_prompt = _truncate_for_prompt(
                trip_description,
                PLANNER_LLM_TRIP_DESCRIPTION_MAX_CHARS,
            )
            stream_prompt_trimmed = (
                stream_flights_in_prompt < min(10, len(all_flights_dicts))
                or warnings_for_prompt != warnings_str
                or trip_description_for_prompt != trip_description
            )

            # Ensure weather values are plain
            weather_display = _normalized_weather_display(weather)

            # Determine forecast proximity to travel date
            forecast_date_str = weather_display.get("forecast_date")
            travel_date_str = best_flight.date or intent_dict.get("date") or "your travel date"
            forecast_is_approximate = False
            approx_note = ""
            if forecast_date_str and travel_date_str and travel_date_str != "your travel date":
                try:
                    from datetime import datetime as _dt
                    t_dt = _dt.strptime(travel_date_str, "%Y-%m-%d")
                    f_dt = _dt.strptime(forecast_date_str, "%Y-%m-%d")
                    delta = abs((t_dt - f_dt).days)
                    if delta > 3:
                        forecast_is_approximate = True
                        approx_note = f" (closest available forecast; actual forecast for {forecast_date_str})"
                except Exception:
                    pass

            # Full weather string with min/max, rain/snow alerts — same as non-streaming
            forecast_label = travel_date_str
            weather_str = (
                f"Condition: {weather_display.get('condition', 'N/A')}, "
                f"Temperature: {weather_display.get('temperature_c', 'N/A')}°C "
                f"(feels like {weather_display.get('feels_like_c', 'N/A')}°C)"
            )
            if weather_display.get("temp_min_c") is not None and weather_display.get("temp_max_c") is not None:
                weather_str += f", Daily low: {weather_display['temp_min_c']}°C, Daily high: {weather_display['temp_max_c']}°C"
            weather_str += (
                f", Humidity: {weather_display.get('humidity', 'N/A')}%, "
                f"Wind: {weather_display.get('wind_kph', 'N/A')} kph, "
                f"AQI: {weather_display.get('air_quality_index', 'N/A')}"
            )
            if weather_display.get("precipitation_chance") is not None:
                weather_str += f", Precipitation chance: {weather_display['precipitation_chance']}%"
            if weather_display.get("has_rain"):
                weather_str += " ⚠️ Rain expected"
            if weather_display.get("has_snow"):
                weather_str += " ⚠️ Snow expected"

            # Human-readable stops
            stops_val = best_flight.stops
            if str(stops_val) in ("N/A", "n/a", "", "unknown"):
                stops_display = "unknown (no data available)"
            elif isinstance(stops_val, int):
                stops_display = "non-stop" if stops_val == 0 else f"{stops_val} stop(s)"
            else:
                stops_display = str(stops_val)

            # Carbon display
            carbon_display = "N/A"
            if best_flight.carbon_emissions_g is not None:
                carbon_kg = round(best_flight.carbon_emissions_g / 1000, 1)
                carbon_display = f"{carbon_kg} kg CO₂"

            # CRITICAL CONSTRAINT for unknown stops
            constraint_note = ""
            if "unknown" in stops_display:
                constraint_note = "You MUST NOT state this is a non-stop or direct flight. If stops are unknown, say 'stop availability unknown'."

            # --- Layover constraint for non-stop flights when user has a layover limit ---
            layover_constraint = ""
            if intent_dict.get("layover_limit_minutes") and best_flight.stops == 0:
                layover_constraint = (
                    f"\nLAYOVER NOTE: This is a non-stop flight — it has NO layover at all, which "
                    f"automatically satisfies the user's max layover requirement of {intent_dict['layover_limit_minutes']} minutes. "
                    "CRITICAL PHRASING RULES — apply to every sentence including the opening and summary:\n"
                    "  1. NEVER write 'with a layover less than X hours' or 'layover of less than X' "
                    "anywhere in your response — those phrases imply the flight HAS a layover.\n"
                    "  2. In your opening sentence, do NOT echo the user's constraint with 'layover of less than'. "
                    "Instead write: 'Based on your max-layover preference, the best option is...'\n"
                    f"  3. In your summary sentence, write something like: "
                    f"'{best_flight.airline} {best_flight.flight_no} is the best non-stop option — it has no layover "
                    "whatsoever, which exceeds your requirement.' Do not echo the user's constraint phrase."
                )

            # --- Temperature constraint for correct ordering ---
            temp_constraint = ""
            if weather_display.get("temp_min_c") is not None and weather_display.get("temp_max_c") is not None:
                temp_constraint = (
                    f"\nTEMP RULE: Daily low is {weather_display['temp_min_c']}°C, daily high is "
                    f"{weather_display['temp_max_c']}°C. Never swap these — do not describe "
                    f"{weather_display['temp_max_c']}°C as the 'low' temperature."
                )

            # Construct system prompt with airline rule only if preferences exist
            airline_rule = ""
            if intent_dict.get("preferred_airlines"):
                preferred_lower = [a.lower() for a in intent_dict.get("preferred_airlines", [])]
                best_airline_lower = best_flight.airline.lower()
                airline_matched = any(p in best_airline_lower or best_airline_lower in p for p in preferred_lower)
                if not airline_matched:
                    airline_rule = (
                        "AIRLINE RULE: The user's preferred airline is not available. "
                        "Your FIRST sentence MUST disclose this — something like: "
                        "'No flights were found for [airline]; here is the closest available alternative.' "
                        "Do NOT open with flight details and then mention the airline mismatch later. "
                        "Do NOT invent or fabricate alternative flights — only present flights from the data above."
                    )

            # Facts block for streaming branch
            facts_block = (
                f"Origin: {intent_dict.get('origin_iata') or 'unknown'}\n"
                f"Destination: {intent_dict.get('destination_iata') or 'unknown'}\n"
                f"Departure date: {intent_dict.get('date') or 'not specified'}\n"
            )
            if intent_dict.get("return_date"):
                facts_block += f"Return date: {intent_dict['return_date']}\n"

            system = (
                "You are a professional travel planning assistant. "
                "CRITICAL: NEVER invent, fabricate, or suggest flight numbers, airline codes, prices, "
                "departure times, or any flight details that are not explicitly present in the flight "
                "data provided to you. If no matching airline is found, present only the available flight "
                "with a note that it differs from the preference — do NOT create fictional alternatives. "
                "RULE: If a flight data field is 'unknown (no data available)', you are PROHIBITED from "
                "stating or implying its value. "
                "IATA RULE: Whenever you mention a city's weather, always include its IATA code in "
                "parentheses, e.g. 'Weather for Bangalore (BLR)' or 'Mumbai (BOM)'. "
                "CITY NAME RULE: When writing about the flight destination, use ONLY the correct city name "
                "for the destination IATA code. Examples of correct mappings: MAA = Chennai (NOT Mumbai), "
                "BLR = Bangalore, BOM = Mumbai, DEL = Delhi. Never call MAA 'Mumbai' or BOM 'Chennai'. "
                + airline_rule
            )

            prompt = f"""
CRITICAL CONSTRAINT: The stops field for this flight is '{stops_display}'.
{constraint_note}{layover_constraint}{temp_constraint}

You are a helpful travel assistant helping a user plan {trip_description_for_prompt}.

User preferences:
- {filters_applied}
{warnings_for_prompt}

Flight options from {intent_dict.get('origin_iata')} to {intent_dict.get('destination_iata')} around {intent_dict.get('date')}:
{flights_str}

Best matching flight:
- {best_flight.airline} {best_flight.flight_no} on {best_flight.date or 'N/A'} |
  {best_flight.departure_time} → {best_flight.arrival_time} |
  Duration: {best_flight.duration_min} minutes |
  Price: {best_flight.price_inr} |
  Stops: {stops_display}{f" ({best_flight.layover_info})" if best_flight.layover_info else ""} |
  Baggage: {best_flight.baggage} |
  Carbon emissions: {carbon_display}
{f"{chr(10)}{price_insights_str}" if price_insights_str else ""}
{f"{chr(10)}{price_analysis_str}" if price_analysis_str else ""}
{f"{chr(10)}{price_prediction_str}" if price_prediction_str else ""}
Weather FORECAST for {intent_dict.get('destination_iata')} on {forecast_label}{approx_note}:
{weather_str}

IMPORTANT: Only reference the exact flights listed above. Do not create or suggest any other flights, codes, or prices.
User's question: {user_query}

Please recommend the best flight, explain why it matches their preferences, mention the weather forecast suitability (including packing advice based on min/max temperature and any rain or snow alerts), and answer the user's query helpfully.
"""

            full_prompt = facts_block + "\nPlease include the above origin and destination clearly at the start of your summary.\n\n" + prompt
            full_prompt, stream_prompt_hard_trimmed = _apply_prompt_hard_limit(
                full_prompt,
                hard_limit=PLANNER_LLM_PROMPT_HARD_LIMIT,
            )
            if stream_prompt_hard_trimmed:
                stream_prompt_trimmed = True

            logger.debug(
                "Streaming LLM prompt prepared",
                extra={"prompt_chars": len(full_prompt)},
            )
            logger.info(
                "Streaming LLM explanation request context",
                extra={
                    "llm_mode": llm_mode_hint if "llm_mode_hint" in locals() else "unknown",
                    "prompt_chars": len(full_prompt),
                    "prompt_trimmed": stream_prompt_trimmed,
                    "prompt_hard_trimmed": stream_prompt_hard_trimmed,
                    "flights_in_prompt": stream_flights_in_prompt,
                    "flights_cap": stream_max_flights_in_prompt,
                    "all_flights_count": len(all_flights_dicts),
                    "warnings_count": len(warnings or []),
                    "has_return_date": bool(intent_dict.get("return_date")),
                    "has_weather_payload": bool(weather_display),
                    "has_price_context": bool(
                        (price_insights_str or "").strip()
                        or (price_analysis_str or "").strip()
                        or (price_prediction_str or "").strip()
                    ),
                    "trip_description_chars": len(trip_description_for_prompt or ""),
                },
            )

            stream_init_timeout = _resolve_stream_init_timeout()
            planner_llm_timeout = _resolve_planner_llm_timeout()
            stream_total_timeout = _resolve_stream_total_timeout(planner_llm_timeout)
            planner_model = _resolve_planner_llm_model()
            router_timeout_hint = get_env_float("ROUTER_TIMEOUT", 90.0)
            router_local_timeout_hint = _resolve_router_local_timeout_hint(planner_llm_timeout)
            logger.info(
                "LLM timeout ownership (stream)",
                extra={
                    "timeout_owner": "router_stream_first_chunk_timeout",
                    "stream_init_timeout_hint_sec": stream_init_timeout,
                    "stream_total_timeout_sec": stream_total_timeout,
                    "planner_timeout_hint_sec": planner_llm_timeout,
                    "router_local_timeout_sec": router_local_timeout_hint,
                    "router_timeout_sec": router_timeout_hint,
                },
            )

            # 4. Call LLM in streaming mode.
            # Router owns first-token and per-chunk timeout boundaries.
            llm_start = time.monotonic()
            try:
                token_stream = await generate(
                    prompt=full_prompt,
                    system=system,
                    model=planner_model,
                    stream=True,
                )
            except (TimeoutError, asyncio.TimeoutError):
                await record_llm_failure(
                    stage="plan_trip_stream_init",
                    reason="upstream_timeout",
                    llm_mode=llm_mode_hint if "llm_mode_hint" in locals() else None,
                    effective_mode=llm_mode_hint if "llm_mode_hint" in locals() else None,
                    attempt_count=1,
                    backend="ollama_or_router",
                )
                metrics.record_stream_failure("unknown")  # provider unknown at this point
                metrics.record_stream_init_timeout("unknown")
                metrics.record_stream_fallback("stream_init_timeout", "unknown")
                yield _sse_event(
                    "reasoning_step",
                    {"step": "Explanation generation degraded: LLM stream initialization timed out, returning structured flight and weather data."},
                )
                async for chunk in _emit_structured_fallback_chunk_if_available():
                    yield chunk
                yield done_json_frame(
                    _degraded_done_payload(
                        reason="upstream_timeout",
                        message=f"LLM stream initialization timed out after {router_local_timeout_hint}s.",
                        provider="router",
                    ),
                    status="degraded",
                )
                return
            except AllBackendsFailed as e:
                status = e.as_dict()
                failures = status.get("failures") or []
                single_backend_scope = _is_single_backend_unavailable(e)
                counter_reason = _primary_router_failure_reason(status)
                timeout_like = counter_reason in {"timeout", "stream_timeout"}
                await record_llm_failure(
                    stage="plan_trip_stream_init",
                    reason="upstream_timeout" if timeout_like else counter_reason,
                    llm_mode=status.get("mode"),
                    effective_mode=status.get("effective_mode"),
                    attempt_count=max(1, len(failures)),
                    backend=(
                        str((failures[0] or {}).get("backend") or "")
                        if failures and isinstance(failures[0], dict)
                        else None
                    ),
                )
                metrics.record_stream_failure("unknown")
                metrics.record_stream_fallback("all_backends_failed", "unknown")
                logger.warning(
                    (
                        "LLM stream initialization timed out for configured backend"
                        if single_backend_scope and timeout_like
                        else "LLM stream initialization timed out across backends"
                        if timeout_like
                        else "LLM stream unavailable for configured backend"
                        if single_backend_scope
                        else "LLM stream unavailable across backends"
                    ),
                    extra=status,
                )
                yield _sse_event(
                    "reasoning_step",
                    {
                        "step": (
                            "Explanation generation degraded: configured LLM backend stream initialization timed out, returning structured flight and weather data."
                            if single_backend_scope and timeout_like
                            else "Explanation generation degraded: LLM backend stream initialization timed out, returning structured flight and weather data."
                            if timeout_like
                            else "Explanation generation degraded: configured LLM backend unavailable, returning structured flight and weather data."
                            if single_backend_scope
                            else "Explanation generation degraded: LLM backends unavailable, returning structured flight and weather data."
                        )
                    },
                )
                async for chunk in _emit_structured_fallback_chunk_if_available():
                    yield chunk
                if timeout_like:
                    stream_init_message = (
                        f"Configured LLM backend stream initialization timed out after {router_local_timeout_hint}s."
                        if single_backend_scope
                        else f"LLM backend stream initialization timed out after {router_local_timeout_hint}s."
                    )
                else:
                    stream_init_message = (
                        "Configured LLM backend unavailable for streaming explanation."
                        if single_backend_scope
                        else "LLM backends unavailable for streaming explanation."
                    )

                yield done_json_frame(
                    _degraded_done_payload(
                        reason="upstream_timeout" if timeout_like else "upstream_unavailable",
                        message=stream_init_message,
                        provider="router",
                        backend_status=status,
                    ),
                    status="degraded",
                )
                return

            # Try to extract provider from token_stream (if available)
            provider = getattr(token_stream, "provider", "unknown")
            stream_provider = provider
            metrics.record_stream_start(provider)

            got_first_token = False
            first_token_time = None
            full_response = ""
            stream_activity_chunks = 0
            thinking_heartbeat_chunks = 0

            # 5. Consume stream with optional total timeout.
            try:
                async def _consume_stream() -> None:
                    nonlocal got_first_token, first_token_time, full_response
                    nonlocal stream_activity_chunks, thinking_heartbeat_chunks
                    # Handle async stream, sync iterable, or single string gracefully
                    if hasattr(token_stream, "__aiter__"):
                        async for token in token_stream:
                            if not isinstance(token, str):
                                token = str(token)
                            if token == "":
                                # Heartbeat chunk (e.g., model thinking signal) – keep stream alive,
                                # but don't count as first visible answer token.
                                stream_activity_chunks += 1
                                thinking_heartbeat_chunks += 1
                                continue
                            stream_activity_chunks += 1
                            if not got_first_token:
                                first_token_time = time.monotonic() - llm_start
                                try:
                                    metrics.LLM_LATENCY.labels(provider=provider).observe(first_token_time)
                                    metrics.observe_llm_first_token(provider, first_token_time)
                                except Exception:
                                    pass
                                got_first_token = True
                            yield token
                            full_response += token
                    elif hasattr(token_stream, "__iter__") and not isinstance(token_stream, (str, bytes)):
                        for token in token_stream:
                            if not isinstance(token, str):
                                token = str(token)
                            if token == "":
                                stream_activity_chunks += 1
                                thinking_heartbeat_chunks += 1
                                continue
                            stream_activity_chunks += 1
                            if not got_first_token:
                                first_token_time = time.monotonic() - llm_start
                                try:
                                    metrics.LLM_LATENCY.labels(provider=provider).observe(first_token_time)
                                    metrics.observe_llm_first_token(provider, first_token_time)
                                except Exception:
                                    pass
                                got_first_token = True
                            yield token
                            full_response += token
                    else:
                        # single non-iterable result (string) — yield it as one chunk
                        token = token_stream
                        if not isinstance(token, str):
                            token = str(token)
                        if token == "":
                            stream_activity_chunks += 1
                            thinking_heartbeat_chunks += 1
                            return
                        stream_activity_chunks += 1
                        if not got_first_token:
                            first_token_time = time.monotonic() - llm_start
                            try:
                                metrics.LLM_LATENCY.labels(provider=provider).observe(first_token_time)
                                metrics.observe_llm_first_token(provider, first_token_time)
                            except Exception:
                                pass
                            got_first_token = True
                        yield token
                        full_response += token

                if stream_total_timeout is not None:
                    async with asyncio.timeout(stream_total_timeout):
                        async for token in _consume_stream():
                            yield token
                else:
                    async for token in _consume_stream():
                        yield token
            except asyncio.CancelledError:
                metrics.record_stream_cancellation(provider, "planner_consume")
                metrics.record_stream_failure(provider)
                raise
            except asyncio.TimeoutError as e:
                await record_llm_failure(
                    stage="plan_trip_stream_consume",
                    reason="upstream_timeout",
                    llm_mode=llm_mode_hint if "llm_mode_hint" in locals() else None,
                    effective_mode=llm_mode_hint if "llm_mode_hint" in locals() else None,
                    attempt_count=1,
                    backend=provider,
                )
                metrics.record_stream_failure(provider)
                metrics.record_stream_fallback("stream_timeout", provider)
                yield _sse_event(
                    "reasoning_step",
                    {"step": "Explanation generation degraded: LLM stream timed out, returning structured flight and weather data."},
                )
                async for chunk in _emit_structured_fallback_chunk_if_available():
                    yield chunk
                yield done_json_frame(
                    _degraded_done_payload(
                        reason="upstream_timeout",
                        message=(
                            str(e).strip()
                            if str(e).strip()
                            else (
                                f"LLM streaming timed out after {stream_total_timeout}s."
                                if stream_total_timeout is not None
                                else f"LLM streaming timed out after {planner_llm_timeout}s."
                            )
                        ),
                        provider=provider,
                        partial_llm_response=(full_response or None),
                    ),
                    status="degraded",
                )
                return

            saw_stream_activity = stream_activity_chunks > 0
            if not (full_response or "").strip():
                if saw_stream_activity:
                    failure_reason = "stream_no_visible_tokens"
                    degradation_reason = "upstream_stream_no_visible_tokens"
                    degradation_message = (
                        "LLM stream stayed alive but produced no visible answer text; deterministic summary preserved."
                    )
                else:
                    failure_reason = "upstream_unavailable"
                    degradation_reason = "upstream_unavailable"
                    degradation_message = "LLM stream completed without visible response text; structured fallback preserved."
                await record_llm_failure(
                    stage="plan_trip_stream_empty_response",
                    reason=failure_reason,
                    llm_mode=llm_mode_hint if "llm_mode_hint" in locals() else None,
                    effective_mode=llm_mode_hint if "llm_mode_hint" in locals() else None,
                    attempt_count=1,
                    backend=provider,
                )
                metrics.record_stream_failure(provider)
                metrics.record_stream_fallback(
                    "stream_no_visible_tokens" if saw_stream_activity else "empty_stream_response",
                    provider,
                )
                logger.info(
                    "LLM stream completed without visible answer tokens",
                    extra={
                        "provider": provider,
                        "stream_activity_chunks": stream_activity_chunks,
                        "thinking_heartbeat_chunks": thinking_heartbeat_chunks,
                        "first_visible_token_seen": got_first_token,
                    },
                )
                yield _sse_event(
                    "reasoning_step",
                    {
                        "step": (
                            "Explanation generation degraded: LLM stream stayed active but produced no visible answer text; returning structured flight and weather data."
                            if saw_stream_activity
                            else "Explanation generation degraded: LLM stream produced no visible answer text; returning structured flight and weather data."
                        )
                    },
                )
                async for chunk in _emit_structured_fallback_chunk_if_available():
                    yield chunk
                yield done_json_frame(
                    _degraded_done_payload(
                        reason=degradation_reason,
                        message=degradation_message,
                        provider=provider,
                    ),
                    status="degraded",
                )
                return

            # 6. Success – record metrics, success, build final JSON, and log
            total_time = time.monotonic() - llm_start
            try:
                metrics.LLM_LATENCY.labels(provider=provider).observe(total_time)
                metrics.observe_llm_full_response(provider, stream=True, duration_sec=total_time)
            except Exception:
                pass
            metrics.record_stream_success(provider, total_time)
            await record_llm_success()

            final_result = data_result.model_dump()
            final_result["llm_response"] = full_response
            # Remove debug_info if you don't want to expose internal data to the client
            # final_result.pop("debug_info", None)
            yield done_json_frame(final_result, status="success")

            # 7. Log session asynchronously (non‑blocking) using stored filtered_count
            if DB_AVAILABLE:
                try:
                    filtered_count = debug_info.get("filtered_count", len(all_flights_dicts))
                    phases = debug_info.get("phases", {})
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            save_session,
                            user_query=user_query,
                            agent_reasoning={
                                "version": "planner-v7-streaming",
                                "intent": intent_dict,
                                "effective_intent": effective_intent_dict,
                                "filters_applied": filters_applied,
                                "ranked_count": debug_info.get("ranked_count", 0),
                                "flight_pref": intent_dict.get("flight_pref"),
                                "trip_type": intent_dict.get("trip_type"),
                                "use_cloud_llm": USE_CLOUD_FALLBACK,
                                "phases": phases,
                                "warnings": warnings
                            },
                            tool_output={
                                "all_flights_count": len(all_flights_dicts),
                                "filtered_count": filtered_count,
                                "weather": weather,
                                "origin": intent_dict.get("origin_iata"),
                                "destination": intent_dict.get("destination_iata"),
                                "search_date": data_result.search_date
                            },
                            final_response=full_response
                        ),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.error("Database write timed out")

        except asyncio.CancelledError:
            metrics.record_stream_cancellation(stream_provider, "planner_generator")
            raise
        except Exception as e:
            err_text = str(e).lower()
            counter_reason = (
                "circuit_open"
                if ("circuit breaker open" in err_text or "circuit breaker is open" in err_text)
                else "upstream_timeout"
                if "timeout" in err_text
                else "upstream_unavailable"
            )
            await record_llm_failure(
                stage="plan_trip_stream",
                reason=counter_reason,
                llm_mode=llm_mode_hint if "llm_mode_hint" in locals() else None,
                effective_mode=llm_mode_hint if "llm_mode_hint" in locals() else None,
                attempt_count=1,
                backend=stream_provider,
            )
            metrics.record_stream_failure(stream_provider)
            fallback_reason = "mid_stream_interruption" if got_first_token else "planner_exception"
            metrics.record_stream_fallback(fallback_reason, stream_provider)
            logger.exception("Error in streaming plan_trip")
            safe_error = _safe_llm_error_message(e)
            if isinstance(data_result, (PlanResult, MultiCityResult)):
                yield _sse_event(
                    "reasoning_step",
                    {"step": "Explanation generation degraded due to an LLM/backend interruption; structured flight and weather data are still available."},
                )
                async for chunk in _emit_structured_fallback_chunk_if_available():
                    yield chunk
                reason = "upstream_timeout" if "timeout" in safe_error.lower() else "upstream_unavailable"
                backend_status = e.as_dict() if isinstance(e, AllBackendsFailed) else None
                yield done_json_frame(
                    _degraded_done_payload(
                        reason=reason,
                        message=safe_error,
                        provider=stream_provider,
                        backend_status=backend_status,
                    ),
                    status="degraded",
                )
            else:
                yield f"[ERROR] {safe_error}"
                done_payload = {
                    "error": safe_error,
                    "failure_reason": "upstream_unavailable",
                    "failure_domain": _failure_domain_from_reason("upstream_unavailable"),
                    "result_status": "error",
                    **_backend_status_payload(e),
                }
                yield done_json_frame(done_payload, status="error")

    return stream_generator()

# ----------------------------------------------------------------------
# Session logging (sync)
# ----------------------------------------------------------------------
def save_session(user_query: str, agent_reasoning: dict, tool_output: dict, final_response: str, user_id: Optional[str] = None):
    if not DB_AVAILABLE:
        logger.info("Session logging skipped (database not available)")
        return

    db = SessionLocal()
    try:
        sh = SessionHistory(
            user_id=user_id,
            user_query=user_query,
            agent_reasoning=agent_reasoning,
            tool_output=tool_output,
            final_response=final_response
        )
        db.add(sh)
        db.commit()
        logger.info("Session saved to database")
    except Exception as e:
        logger.error(f"Failed to save session: {e}")
        db.rollback()
    finally:
        db.close()

# ----------------------------------------------------------------------
# Helper: normalize airport for tests (simple deterministic fallback)
# ----------------------------------------------------------------------
def normalize_airport(text: Optional[str]) -> Optional[str]:
    """
    Normalize a free-text airport/city token into an IATA code or return None.
    Tries (in order):
      1. Direct resolver (city name, airport name, phrase) -> resolve_location(...)
      2. If token is a 3-letter alpha code, return uppercased token (best-effort)
      3. Otherwise None

    This is intentionally simple and deterministic for tests.
    """
    if not text:
        return None

    tok = text.strip()
    # Fast path for explicit 3-letter tokens (case-insensitive).
    if len(tok) == 3 and tok.isalpha():
        up = tok.upper()
        if is_iata_token(up):
            return up

    # First try the central resolver which already supports phrases/tokens/n-grams
    try:
        iata = resolve_location(text)
    except Exception:
        iata = None

    if iata:
        return iata

    return None
