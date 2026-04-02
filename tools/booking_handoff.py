# tools/booking_handoff.py
"""
Booking Handoff Tool

Responsibilities:
- Hold, confirm, cancel, and expire booking records in the database
- Resolve the best possible deep-link for a flight using (in priority order):
    1. SerpAPI booking_token  → calls /search?engine=google_flights_booking to get
                                airline-native checkout URL (exact itinerary, best UX)
    2. shareable_link         → direct Google Flights shareable link (still pre-filled)
    3. Google Flights fallback→ clean HTTPS search URL (last resort, no guessing)

NOTE: The old AIRLINE_BOOKING_URLS dict that guessed airline homepages is removed.
      Those URLs just dumped the user on a generic search page, which is worse than
      the Google Flights fallback. The SerpAPI token gives us the real checkout page.
"""

import asyncio
import contextlib
import hashlib
import logging
import re
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union

import httpx
from cachetools import TTLCache
from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, and_, or_
from agents.database import Base, SessionLocal, get_engine
from core.api_key_manager import key_manager as api_key_manager
from core.env_config import get_env_bool, get_env_float, get_env_int
from core.http_client import get_client
from core.request_context import get_request_id

logger = logging.getLogger(__name__)
BOOKING_OPTIONS_HTTP_TIMEOUT = get_env_float("BOOKING_OPTIONS_HTTP_TIMEOUT", 2.2)
BOOKING_OPTIONS_RETRIES = max(1, get_env_int("BOOKING_OPTIONS_RETRIES", 3))
BOOKING_OPTIONS_RETRY_BACKOFF = get_env_float("BOOKING_OPTIONS_RETRY_BACKOFF", 0.15)
BOOKING_OPTIONS_ROUND_TRIP_TIMEOUT_BONUS = max(
    0.0,
    get_env_float("BOOKING_OPTIONS_ROUND_TRIP_TIMEOUT_BONUS", 0.35),
)
BOOKING_OPTIONS_ATTEMPTS_BUDGET = min(2, BOOKING_OPTIONS_RETRIES)
BOOKING_TOKEN_RESOLVE_TIMEOUT_FLOOR = (
    BOOKING_OPTIONS_HTTP_TIMEOUT * BOOKING_OPTIONS_ATTEMPTS_BUDGET
    + BOOKING_OPTIONS_RETRY_BACKOFF * max(0, BOOKING_OPTIONS_ATTEMPTS_BUDGET - 1)
    + 0.5
)
BOOKING_TOKEN_RESOLVE_TIMEOUT = max(
    get_env_float("BOOKING_TOKEN_RESOLVE_TIMEOUT", 1.4),
    BOOKING_TOKEN_RESOLVE_TIMEOUT_FLOOR,
)
BOOKING_REQUEST_HTTP_TIMEOUT = get_env_float("BOOKING_REQUEST_HTTP_TIMEOUT", 2.0)
BOOKING_REQUEST_RETRIES = max(1, get_env_int("BOOKING_REQUEST_RETRIES", 2))
BOOKING_REQUEST_RETRY_BACKOFF = get_env_float("BOOKING_REQUEST_RETRY_BACKOFF", 0.12)
POST_HANDOFF_TTL_SECONDS = max(180, get_env_int("POST_HANDOFF_TTL_SECONDS", 900))
POST_HANDOFF_MAX_ENTRIES = max(50, get_env_int("POST_HANDOFF_MAX_ENTRIES", 1000))
POST_HANDOFF_REQUIRE_PERSISTENCE = get_env_bool("POST_HANDOFF_REQUIRE_PERSISTENCE", default=True)
_post_handoff_artifacts: TTLCache[str, Dict[str, Any]] = TTLCache(
    maxsize=POST_HANDOFF_MAX_ENTRIES,
    ttl=POST_HANDOFF_TTL_SECONDS,
)
BOOKING_RESOLUTION_CACHE_TTL_SECONDS = max(30, get_env_int("BOOKING_RESOLUTION_CACHE_TTL_SECONDS", 180))
BOOKING_RESOLUTION_CACHE_MAX_ENTRIES = max(50, get_env_int("BOOKING_RESOLUTION_CACHE_MAX_ENTRIES", 500))
_booking_resolution_cache: TTLCache[str, Dict[str, Any]] = TTLCache(
    maxsize=BOOKING_RESOLUTION_CACHE_MAX_ENTRIES,
    ttl=BOOKING_RESOLUTION_CACHE_TTL_SECONDS,
)
_candidate_fallback_log_counts: TTLCache[str, int] = TTLCache(
    maxsize=5000,
    ttl=max(60, get_env_int("BOOKING_CANDIDATE_FALLBACK_LOG_TTL_SECONDS", 600)),
)
_handoff_cache_lock = threading.RLock()


class BookingOptionsFetchError(RuntimeError):
    """
    Raised when booking-options fetch fails after bounded retries.
    Carries compact, non-sensitive request-shape context for diagnostics.
    """

    def __init__(self, reason: str, *, context: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.context = context or {}


def _token_fingerprint(token: str) -> str:
    if not token:
        return "none"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def _candidate_fallback_log_key(*, exception_bucket: str, token_fp: str, route_type: str) -> str:
    request_id = get_request_id() or "unknown"
    return f"{request_id}:{exception_bucket}:{token_fp}:{route_type}"


def _should_emit_candidate_fallback_log(
    *,
    exception_bucket: str,
    token_fp: str,
    route_type: str,
    candidate_probe_context: bool,
) -> tuple[bool, int]:
    """
    Throttle repeated expected candidate fallback logs per request/bucket/token/route.
    Returns (should_emit, occurrence_count).
    """
    if not candidate_probe_context:
        return True, 1
    key = _candidate_fallback_log_key(
        exception_bucket=exception_bucket,
        token_fp=token_fp,
        route_type=route_type,
    )
    with _handoff_cache_lock:
        occurrence = int(_candidate_fallback_log_counts.get(key, 0)) + 1
        _candidate_fallback_log_counts[key] = occurrence
    if occurrence == 1:
        return True, occurrence
    if occurrence in {5, 10, 20}:
        return True, occurrence
    return False, occurrence


def _key_fingerprint(key: str) -> str:
    if not key:
        return "none"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def _classify_booking_options_exception(exc: Exception) -> str:
    if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
        return "timeout"
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
        return "network"
    if isinstance(exc, ValueError):
        return "response_parse"
    text = str(exc or "").lower()
    if "no available keys for service" in text or "no usable keys for provider" in text:
        return "no_active_key"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "provider_rate_limited"
    if "403" in text or "401" in text or "unauthorized" in text:
        return "provider_auth"
    return "unexpected"


def _get_booking_http_client() -> httpx.AsyncClient:
    """
    Keep booking-options transport aligned with flight-search transport.
    """
    return get_client()


def _booking_resolution_cache_key(
    *,
    booking_token: str,
    departure_id: Optional[str],
    arrival_id: Optional[str],
    outbound_date: Optional[str],
    return_date: Optional[str],
) -> str:
    raw = "|".join(
        [
            str(booking_token or ""),
            str(departure_id or ""),
            str(arrival_id or ""),
            str(outbound_date or ""),
            str(return_date or ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _cache_booking_resolution(cache_key: str, payload: Dict[str, Any]) -> None:
    with _handoff_cache_lock:
        _booking_resolution_cache[cache_key] = payload


def booking_resolution_cache_stats() -> Dict[str, int]:
    with _handoff_cache_lock:
        entries = len(_booking_resolution_cache)
    return {
        "entries": entries,
        "ttl_sec": BOOKING_RESOLUTION_CACHE_TTL_SECONDS,
    }


def _canonicalize_handoff_url(value: Optional[str], *, _depth: int = 0) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None

    if raw.startswith("//"):
        raw = f"https:{raw}"
    elif raw.lower().startswith("www."):
        raw = f"https://{raw}"

    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return None

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    host = parsed.netloc.lower()
    if _depth < 2 and host.endswith("google.com") and parsed.path == "/url":
        qs = urllib.parse.parse_qs(parsed.query or "")
        saw_wrapped_target = False
        for key in ("q", "url"):
            candidate = qs.get(key, [None])[0]
            if not candidate:
                continue
            saw_wrapped_target = True
            unquoted = urllib.parse.unquote(str(candidate)).strip()
            normalized = _canonicalize_handoff_url(unquoted, _depth=_depth + 1)
            if normalized:
                return normalized
        if saw_wrapped_target:
            return None

    return raw


def _is_usable_handoff_url(value: Optional[str]) -> bool:
    return _canonicalize_handoff_url(value) is not None


def _is_google_search_fallback_url(value: Optional[str]) -> bool:
    canonical = _canonicalize_handoff_url(value)
    if not canonical:
        return False
    try:
        parsed = urllib.parse.urlparse(canonical)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    return host.endswith("google.com") and parsed.path == "/travel/flights"


def _build_post_handoff_bridge_url(artifact_id: str) -> str:
    return f"/booking/handoff/post/{artifact_id}"


def _artifact_log_id(artifact_id: str) -> str:
    if not artifact_id:
        return "none"
    return artifact_id[:12]


def _store_post_handoff_artifact_persistent(
    *,
    artifact_id: str,
    url: str,
    post_data: Any,
    headers: Optional[Dict[str, str]] = None,
) -> bool:
    session = SessionLocal()
    try:
        expires_at = datetime.utcnow() + timedelta(seconds=POST_HANDOFF_TTL_SECONDS)
        row = PostHandoffArtifact(
            artifact_id=artifact_id,
            url=url,
            post_data=post_data,
            headers=headers or {},
            expires_at=expires_at,
        )
        session.merge(row)
        session.commit()
        logger.info(
            "booking_post_bridge_artifact_stored",
            extra={
                "artifact_id_prefix": _artifact_log_id(artifact_id),
                "ttl_seconds": POST_HANDOFF_TTL_SECONDS,
                "one_time_use": True,
                "persistent_store": True,
            },
        )
        return True
    except Exception as e:
        session.rollback()
        logger.warning(
            "booking_post_bridge_artifact_store_failed",
            extra={
                "artifact_id_prefix": _artifact_log_id(artifact_id),
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            },
        )
        return False
    finally:
        session.close()


def _consume_post_handoff_artifact_persistent_with_result(
    artifact_id: str,
) -> tuple[Optional[Dict[str, Any]], str]:
    session = SessionLocal()
    try:
        row = session.query(PostHandoffArtifact).filter(
            PostHandoffArtifact.artifact_id == artifact_id
        ).first()
        if not row:
            logger.debug(
                "booking_post_bridge_lookup_miss",
                extra={"artifact_id_prefix": _artifact_log_id(artifact_id), "lookup_result": "not_found"},
            )
            return None, "not_found"
        now = datetime.utcnow()
        updated = session.query(PostHandoffArtifact).filter(
            and_(
                PostHandoffArtifact.artifact_id == artifact_id,
                PostHandoffArtifact.consumed_at.is_(None),
                or_(PostHandoffArtifact.expires_at.is_(None), PostHandoffArtifact.expires_at > now),
            )
        ).update(
            {PostHandoffArtifact.consumed_at: now},
            synchronize_session=False,
        )
        if updated != 1:
            session.rollback()
            fresh = session.query(PostHandoffArtifact).filter(
                PostHandoffArtifact.artifact_id == artifact_id
            ).first()
            lookup_result = "not_found"
            if fresh is not None:
                if fresh.consumed_at is not None:
                    lookup_result = "already_consumed"
                elif fresh.expires_at is not None and fresh.expires_at <= now:
                    lookup_result = "expired"
                else:
                    lookup_result = "consume_race_lost"
            logger.debug(
                "booking_post_bridge_lookup_miss",
                extra={"artifact_id_prefix": _artifact_log_id(artifact_id), "lookup_result": lookup_result},
            )
            return None, lookup_result

        session.commit()
        logger.debug(
            "booking_post_bridge_lookup_hit",
            extra={"artifact_id_prefix": _artifact_log_id(artifact_id), "lookup_result": "persistent_hit"},
        )
        return {
            "url": row.url,
            "post_data": row.post_data,
            "headers": row.headers or {},
        }, "persistent_hit"
    except Exception as e:
        session.rollback()
        logger.warning(
            "booking_post_bridge_lookup_failed",
            extra={
                "artifact_id_prefix": _artifact_log_id(artifact_id),
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            },
        )
        return None, "lookup_failed"
    finally:
        session.close()


def _consume_post_handoff_artifact_persistent(artifact_id: str) -> Optional[Dict[str, Any]]:
    artifact, _ = _consume_post_handoff_artifact_persistent_with_result(artifact_id)
    return artifact


def _mark_post_handoff_artifact_consumed_persistent(artifact_id: str) -> None:
    session = SessionLocal()
    try:
        now = datetime.utcnow()
        session.query(PostHandoffArtifact).filter(
            and_(
                PostHandoffArtifact.artifact_id == artifact_id,
                PostHandoffArtifact.consumed_at.is_(None),
            )
        ).update(
            {PostHandoffArtifact.consumed_at: now},
            synchronize_session=False,
        )
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def register_post_handoff_artifact(
    *,
    url: str,
    post_data: Any,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Register an internal POST-capable handoff artifact and return a short-lived bridge URL.
    """
    canonical_url = _canonicalize_handoff_url(url)
    if not canonical_url or _is_google_search_fallback_url(canonical_url):
        return None
    if post_data in (None, "", {}, []):
        return None

    artifact_id = uuid.uuid4().hex
    with _handoff_cache_lock:
        _post_handoff_artifacts[artifact_id] = {
            "url": canonical_url,
            "post_data": post_data,
            "headers": headers or {},
        }
    persisted = _store_post_handoff_artifact_persistent(
        artifact_id=artifact_id,
        url=canonical_url,
        post_data=post_data,
        headers=headers or {},
    )
    if not persisted and POST_HANDOFF_REQUIRE_PERSISTENCE:
        with _handoff_cache_lock:
            _post_handoff_artifacts.pop(artifact_id, None)
        logger.warning(
            "booking_post_bridge_artifact_rejected_non_persistent",
            extra={
                "artifact_id_prefix": _artifact_log_id(artifact_id),
                "ttl_seconds": POST_HANDOFF_TTL_SECONDS,
                "require_persistence": True,
            },
        )
        return None
    logger.info(
        "booking_post_bridge_artifact_created",
        extra={
            "artifact_id_prefix": _artifact_log_id(artifact_id),
            "ttl_seconds": POST_HANDOFF_TTL_SECONDS,
            "one_time_use": True,
            "stored_in_memory": True,
            "stored_persistent": persisted,
        },
    )
    return _build_post_handoff_bridge_url(artifact_id)


def consume_post_handoff_artifact(artifact_id: str) -> Optional[Dict[str, Any]]:
    artifact, _ = consume_post_handoff_artifact_with_diagnostics(artifact_id)
    return artifact


def consume_post_handoff_artifact_with_diagnostics(
    artifact_id: str,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not artifact_id:
        diagnostics = {
            "artifact_id_prefix": _artifact_log_id(artifact_id),
            "lookup_result": "invalid_artifact_id",
            "consume_outcome": "miss",
            "request_id": get_request_id() or "unknown",
        }
        logger.info("booking_post_bridge_consume_outcome", extra=diagnostics)
        return None, diagnostics

    logger.debug(
        "booking_post_bridge_lookup_requested",
        extra={"artifact_id_prefix": _artifact_log_id(artifact_id)},
    )
    with _handoff_cache_lock:
        artifact = _post_handoff_artifacts.pop(artifact_id, None)
    lookup_result = "not_found"
    if isinstance(artifact, dict):
        _mark_post_handoff_artifact_consumed_persistent(artifact_id)
        logger.debug(
            "booking_post_bridge_lookup_hit",
            extra={"artifact_id_prefix": _artifact_log_id(artifact_id), "lookup_result": "memory_hit"},
        )
        lookup_result = "memory_hit"
        consumed = {
            "url": artifact.get("url"),
            "post_data": artifact.get("post_data"),
            "headers": artifact.get("headers") or {},
        }
        diagnostics = {
            "artifact_id_prefix": _artifact_log_id(artifact_id),
            "lookup_result": lookup_result,
            "consume_outcome": "hit",
            "request_id": get_request_id() or "unknown",
        }
        logger.info("booking_post_bridge_consume_outcome", extra=diagnostics)
        return consumed, diagnostics

    artifact, lookup_result = _consume_post_handoff_artifact_persistent_with_result(artifact_id)
    diagnostics = {
        "artifact_id_prefix": _artifact_log_id(artifact_id),
        "lookup_result": lookup_result,
        "consume_outcome": "hit" if isinstance(artifact, dict) else "miss",
        "request_id": get_request_id() or "unknown",
    }
    logger.info("booking_post_bridge_consume_outcome", extra=diagnostics)
    return artifact, diagnostics


def _search_fallback_quality(
    *,
    origin: Optional[str],
    destination: Optional[str],
    depart_date: Optional[str],
    return_date: Optional[str],
    flight: Optional[Dict[str, Any]],
) -> str:
    if not origin or not destination or not depart_date:
        return "generic"
    has_itinerary_hints = bool(
        isinstance(flight, dict)
        and (
            str(flight.get("airline") or "").strip()
            or str(flight.get("flight_no") or "").strip()
            or str(flight.get("departure_time") or "").strip()
            or str(flight.get("arrival_time") or "").strip()
        )
    )
    if return_date:
        if has_itinerary_hints:
            return "round_trip_route_seeded_with_itinerary_hints"
        return "round_trip_route_seeded_with_return_leg"
    if has_itinerary_hints:
        return "route_seeded_with_itinerary_hints"
    return "route_seeded_basic"


def _search_fallback_context(
    *,
    origin: Optional[str],
    destination: Optional[str],
    depart_date: Optional[str],
    return_date: Optional[str],
    flight: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    has_route_seed = bool(origin and destination and depart_date)
    has_itinerary_hints = bool(
        isinstance(flight, dict)
        and (
            str(flight.get("airline") or "").strip()
            or str(flight.get("flight_no") or "").strip()
            or str(flight.get("departure_time") or "").strip()
            or str(flight.get("arrival_time") or "").strip()
        )
    )
    quality_tier = _search_fallback_quality(
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        return_date=return_date,
        flight=flight,
    )
    return {
        "route_type": "round_trip" if return_date else "one_way",
        "has_route_seed": has_route_seed,
        "has_itinerary_hints": has_itinerary_hints,
        "includes_return_leg_hint": bool(return_date and has_route_seed),
        "quality_tier": quality_tier,
    }


def _legacy_search_fallback_quality_for_clients(quality_tier: str) -> str:
    """
    Backward-compatible quality string for clients/tests expecting legacy values.
    Keep richer tier details in `search_fallback_context`.
    """
    normalized = str(quality_tier or "")
    if normalized in {
        "round_trip_route_seeded_with_itinerary_hints",
        "route_seeded_with_itinerary_hints",
    }:
        return "route_seeded_with_itinerary_hints"
    if normalized in {"round_trip_route_seeded_with_return_leg", "route_seeded_basic"}:
        return "route_seeded_basic"
    return "generic"


def _extract_booking_request_payload(flight: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract a normalized booking_request follow-up payload from flight artifacts.
    Supports direct dict payloads and flattened fallback keys.
    """
    request_obj = flight.get("booking_request")
    if isinstance(request_obj, dict):
        raw = request_obj
    else:
        maybe_url = flight.get("booking_request_url")
        if not maybe_url:
            return None
        raw = {
            "url": maybe_url,
            "method": flight.get("booking_request_method"),
            "post_data": flight.get("booking_request_post_data"),
            "headers": flight.get("booking_request_headers"),
        }

    raw_url = raw.get("url") or raw.get("endpoint") or raw.get("booking_url") or raw.get("link")
    canonical_url = _canonicalize_handoff_url(raw_url)
    if not canonical_url:
        return None
    # booking_request should never "win" via generic Google search URL.
    if _is_google_search_fallback_url(canonical_url):
        return None

    method = str(raw.get("method") or "").strip().upper()
    post_data = raw.get("post_data")
    if not method:
        method = "POST" if post_data not in (None, "", {}, []) else "GET"
    if method not in {"GET", "POST"}:
        return None

    headers = raw.get("headers")
    safe_headers: Dict[str, str] = {}
    if isinstance(headers, dict):
        for key, value in headers.items():
            if not isinstance(key, str):
                continue
            if value is None:
                continue
            key_l = key.lower().strip()
            if key_l in {"authorization", "cookie"}:
                continue
            safe_headers[key] = str(value)

    return {
        "url": canonical_url,
        "method": method,
        "post_data": post_data,
        "headers": safe_headers,
    }


def _extract_link_from_response_payload(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("booking_url", "booking_link", "deeplink", "redirect_link", "link", "url"):
            candidate = _canonicalize_handoff_url(payload.get(key))
            if candidate and not _is_google_search_fallback_url(candidate):
                return candidate
        for nested_key in ("data", "result", "booking", "provider", "seller"):
            if nested_key in payload:
                nested = _extract_link_from_response_payload(payload.get(nested_key))
                if nested:
                    return nested
        return None
    if isinstance(payload, list):
        for item in payload[:10]:
            nested = _extract_link_from_response_payload(item)
            if nested:
                return nested
    return None


async def _resolve_booking_request_handoff(flight: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve provider/shareable handoff from booking_request payload.
    Returns (url, artifact_field).
    """
    payload = _extract_booking_request_payload(flight)
    if not payload:
        return None, None

    method = payload["method"]
    url = payload["url"]
    headers = payload.get("headers") or {}
    post_data = payload.get("post_data")

    if method == "GET":
        return url, "booking_request.url"

    bridge_url = register_post_handoff_artifact(
        url=url,
        post_data=post_data,
        headers=headers,
    )
    if bridge_url:
        return bridge_url, "booking_request.post_followup_bridge"
    return None, None


def _iter_booking_option_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    root_rows = data.get("booking_options")
    if isinstance(root_rows, list):
        rows.extend([r for r in root_rows if isinstance(r, dict)])

    for container in ("best_flights", "other_flights", "flights", "selected_flights"):
        section = data.get(container)
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            nested_rows = item.get("booking_options")
            if isinstance(nested_rows, list):
                rows.extend([r for r in nested_rows if isinstance(r, dict)])

    return rows

# ----------------------------------------------------------------------
# Database model
# ----------------------------------------------------------------------

class Booking(Base):
    """
    Represents a single flight booking record.

    Status lifecycle:
        HELD  →  CONFIRMED  (user confirmed within hold window)
        HELD  →  EXPIRED    (hold_minutes elapsed without confirmation)
        HELD  →  CANCELLED  (user explicitly cancelled before expiry)
    """
    __tablename__ = "bookings"

    id           = Column(Integer, primary_key=True, index=True)
    status       = Column(String,  nullable=False)           # HELD | CONFIRMED | CANCELLED | EXPIRED
    flight       = Column(JSON,    nullable=False)           # Full flight dict (includes booking_token, shareable_link)
    passenger    = Column(JSON,    nullable=True)            # Passenger info dict (name, DOB, passport…)
    booking_token = Column(Text,   nullable=True)            # SerpAPI booking_token (top-level, for quick access)
    shareable_link = Column(Text,  nullable=True)            # SerpAPI shareable_link (top-level, for quick access)
    handoff_url  = Column(Text,    nullable=True)            # Resolved deep-link written at hold time
    created_at   = Column(DateTime, default=datetime.utcnow)
    expires_at   = Column(DateTime, nullable=True)


class PostHandoffArtifact(Base):
    """
    Cross-process safe storage for short-lived POST handoff artifacts.
    """
    __tablename__ = "post_handoff_artifacts"

    artifact_id = Column(String, primary_key=True, index=True)
    url = Column(Text, nullable=False)
    post_data = Column(JSON, nullable=False)
    headers = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    consumed_at = Column(DateTime, nullable=True)


def ensure_tables():
    """Create any missing tables (safe to call multiple times)."""
    Base.metadata.create_all(bind=get_engine())

ensure_tables()


# ----------------------------------------------------------------------
# SerpAPI booking resolution
# ----------------------------------------------------------------------

SERPAPI_BOOKING_ENDPOINT = "https://serpapi.com/search"

def _extract_booking_options_from_payload(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    def _coerce_price(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        compact = text.replace(",", "")
        match = re.search(r"(\d+(?:\.\d+)?)", compact)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    def _extract_nested_link(opt: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        nested_sources = ("book_with", "providers", "seller", "book", "booking")
        link_keys = ("link", "url", "booking_url", "booking_link", "deeplink", "redirect_link")
        for key in nested_sources:
            nested = opt.get(key)
            if isinstance(nested, dict):
                candidate = None
                for lk in link_keys:
                    if nested.get(lk):
                        candidate = nested.get(lk)
                        break
                provider = nested.get("name") or nested.get("provider")
                canonical = _canonicalize_handoff_url(candidate)
                if canonical:
                    return canonical, provider
            if isinstance(nested, str):
                canonical = _canonicalize_handoff_url(nested)
                if canonical:
                    return canonical, None
            if isinstance(nested, list):
                for item in nested:
                    if not isinstance(item, dict):
                        continue
                    candidate = None
                    for lk in link_keys:
                        if item.get(lk):
                            candidate = item.get(lk)
                            break
                    provider = item.get("name") or item.get("provider")
                    canonical = _canonicalize_handoff_url(candidate)
                    if canonical:
                        return canonical, provider
        return None, None

    options = []
    seen_links = set()
    for opt in _iter_booking_option_rows(data):
        if not isinstance(opt, dict):
            continue
        provider = opt.get("name") or opt.get("provider") or "unknown"
        price = opt.get("price")
        link = (
            opt.get("link")
            or opt.get("url")
            or opt.get("booking_url")
            or opt.get("booking_link")
            or opt.get("deeplink")
            or opt.get("redirect_link")
        )
        canonical = _canonicalize_handoff_url(link)
        if not canonical:
            nested_link, nested_provider = _extract_nested_link(opt)
            if nested_link:
                canonical = nested_link
                provider = nested_provider or provider
        if not canonical:
            continue
        if canonical in seen_links:
            continue
        seen_links.add(canonical)

        price_float = _coerce_price(price)
        options.append({
            "provider": provider,
            "price": price_float,
            "link": canonical.strip(),
            "price_available": price_float is not None,
        })
    return options


def _extract_booking_request_artifact_from_payload(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    booking_request = data.get("booking_request")
    if not isinstance(booking_request, dict):
        selected = data.get("selected_flights")
        if isinstance(selected, list):
            for row in selected:
                if isinstance(row, dict) and isinstance(row.get("booking_request"), dict):
                    booking_request = row.get("booking_request")
                    break
    if not isinstance(booking_request, dict):
        # Some payloads nest booking_request under buckets like:
        # booking_options[*].together.booking_request
        queue: List[Any] = []
        for key in ("booking_options", "selected_flights", "best_flights", "other_flights", "flights"):
            value = data.get(key)
            if isinstance(value, (list, dict)):
                queue.append(value)
        while queue and not isinstance(booking_request, dict):
            node = queue.pop(0)
            if isinstance(node, dict):
                direct = node.get("booking_request")
                if isinstance(direct, dict):
                    booking_request = direct
                    break
                together = node.get("together")
                if isinstance(together, dict):
                    nested = together.get("booking_request")
                    if isinstance(nested, dict):
                        booking_request = nested
                        break
                for child in node.values():
                    if isinstance(child, (list, dict)):
                        queue.append(child)
            elif isinstance(node, list):
                for child in node:
                    if isinstance(child, (list, dict)):
                        queue.append(child)
    if not isinstance(booking_request, dict):
        return None

    url = (
        booking_request.get("url")
        or booking_request.get("endpoint")
        or booking_request.get("booking_url")
        or booking_request.get("link")
    )
    canonical_url = _canonicalize_handoff_url(url)
    if not canonical_url or _is_google_search_fallback_url(canonical_url):
        return None

    post_data = booking_request.get("post_data")
    method = str(booking_request.get("method") or "").strip().upper()
    if not method:
        method = "POST" if post_data not in (None, "", {}, []) else "GET"

    safe_headers: Dict[str, str] = {}
    raw_headers = booking_request.get("headers")
    if isinstance(raw_headers, dict):
        for key, value in raw_headers.items():
            if not isinstance(key, str):
                continue
            if value is None:
                continue
            key_l = key.lower().strip()
            if key_l in {"authorization", "cookie"}:
                continue
            safe_headers[key] = str(value)

    return {
        "url": canonical_url,
        "method": method,
        "post_data": post_data,
        "headers": safe_headers,
    }


async def _fetch_booking_options_payload(
    *,
    booking_token: str,
    departure_id: Optional[str] = None,
    arrival_id: Optional[str] = None,
    outbound_date: Optional[str] = None,
    return_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch raw google_flights_booking payload from SerpAPI.
    Includes original route/date search context when available.
    """

    route_type = "round_trip" if return_date else "one_way"
    max_attempts = max(1, min(BOOKING_OPTIONS_RETRIES, BOOKING_OPTIONS_ATTEMPTS_BUDGET))

    for attempt in range(1, max_attempts + 1):
        attempt_started = time.monotonic()
        response_flags: Dict[str, Optional[bool]] = {
            "response_has_booking_options": None,
            "response_has_selected_flights": None,
            "response_has_booking_request_url": None,
            "response_has_booking_request_post_data": None,
            "response_top_keys": None,
        }
        try:
            async with api_key_manager.reserve_key("serpapi") as (idx, key):
                params = {
                    # Booking options are selected by booking_token on google_flights engine.
                    "engine": "google_flights",
                    "booking_token": booking_token,
                    "api_key": key,
                    "hl": "en",
                    "gl": "in",
                    "currency": "INR",
                }
                # We intentionally never mix booking_token with departure_token.
                params.pop("departure_token", None)
                # Proven runtime requirement: token-only requests can fail.
                # Preserve original search context when available.
                if departure_id:
                    params["departure_id"] = departure_id
                if arrival_id:
                    params["arrival_id"] = arrival_id
                if outbound_date:
                    params["outbound_date"] = outbound_date
                if return_date:
                    params["return_date"] = return_date
                    params["type"] = "1"
                elif outbound_date:
                    params["type"] = "2"

                request_timeout = BOOKING_OPTIONS_HTTP_TIMEOUT + (
                    BOOKING_OPTIONS_ROUND_TRIP_TIMEOUT_BONUS if return_date else 0.0
                )
                client = _get_booking_http_client()
                resp = await client.get(
                    SERPAPI_BOOKING_ENDPOINT,
                    params=params,
                    timeout=request_timeout,
                )

            request_shape = {
                "attempt": attempt,
                "has_booking_token": bool(params.get("booking_token")),
                "has_departure_id": bool(params.get("departure_id")),
                "has_arrival_id": bool(params.get("arrival_id")),
                "has_outbound_date": bool(params.get("outbound_date")),
                "has_return_date": bool(params.get("return_date")),
                "has_departure_token": bool(params.get("departure_token")),
                "token_fp": _token_fingerprint(booking_token),
                "key_fp": _key_fingerprint(key),
                "key_source": "api_key_manager.reserve_key:serpapi",
                "key_index": idx,
                "engine": str(params.get("engine") or ""),
                "request_timeout_sec": request_timeout,
                "retry_limit": max_attempts,
                "attempt_budget": BOOKING_OPTIONS_ATTEMPTS_BUDGET,
                "client_mode": "shared_get_client",
                "route_type": route_type,
            }

            if resp.status_code != 200:
                transient = resp.status_code in {408, 429} or resp.status_code >= 500
                if transient and attempt < max_attempts:
                    await asyncio.sleep(BOOKING_OPTIONS_RETRY_BACKOFF * attempt)
                    continue
                if resp.status_code in {401, 403}:
                    with contextlib.suppress(Exception):
                        await api_key_manager.mark_exhausted(
                            "serpapi",
                            idx,
                            reason=f"booking_options_http_{resp.status_code}",
                        )
                logger.info(
                    "booking_options unavailable after shaped request",
                    extra={
                        **request_shape,
                        "http_status": resp.status_code,
                        "fetch_duration_ms": int((time.monotonic() - attempt_started) * 1000),
                        "result_bucket": "http_error",
                        **response_flags,
                    },
                )
                raise BookingOptionsFetchError(
                    "booking_options_http_error",
                    context={
                        **request_shape,
                        "http_status": resp.status_code,
                        **response_flags,
                    },
                )

            try:
                data = resp.json()
            except Exception as parse_exc:
                logger.info(
                    "booking_options parse error after shaped request",
                    extra={
                        **request_shape,
                        "fetch_duration_ms": int((time.monotonic() - attempt_started) * 1000),
                        "result_bucket": "response_parse_error",
                        "exception_type": type(parse_exc).__name__,
                        "exception_message": str(parse_exc),
                        **response_flags,
                    },
                )
                raise BookingOptionsFetchError(
                    "booking_options_parse_error",
                    context={
                        **request_shape,
                        "exception_type": type(parse_exc).__name__,
                        "exception_message": str(parse_exc),
                        "exception_bucket": "response_parse",
                        **response_flags,
                    },
                ) from parse_exc
            booking_request_artifact = _extract_booking_request_artifact_from_payload(data)
            top_keys = list(data.keys())[:12] if isinstance(data, dict) else []
            response_flags = {
                "response_has_booking_options": bool(_extract_booking_options_from_payload(data)),
                "response_has_selected_flights": bool(isinstance(data.get("selected_flights"), list) and len(data.get("selected_flights") or []) > 0) if isinstance(data, dict) else False,
                "response_has_booking_request_url": bool(
                    isinstance(booking_request_artifact, dict) and booking_request_artifact.get("url")
                ),
                "response_has_booking_request_post_data": bool(
                    isinstance(booking_request_artifact, dict)
                    and booking_request_artifact.get("post_data") not in (None, "", {}, [])
                ),
                "response_top_keys": ",".join(top_keys),
            }

            if "error" in data:
                error_text = str(data.get("error") or "").lower()
                transient_error = any(
                    token in error_text
                    for token in (
                        "rate limit",
                        "too many requests",
                        "temporarily",
                        "timeout",
                        "timed out",
                        "unavailable",
                        "try again",
                    )
                )
                if transient_error and attempt < max_attempts:
                    await asyncio.sleep(BOOKING_OPTIONS_RETRY_BACKOFF * attempt)
                    continue
                if any(tok in error_text for tok in ("unauthorized", "invalid api key", "invalid key", "access denied")):
                    with contextlib.suppress(Exception):
                        await api_key_manager.mark_exhausted(
                            "serpapi",
                            idx,
                            reason="booking_options_provider_unauthorized",
                        )
                logger.info(
                    "booking_options provider error after shaped request",
                    extra={
                        **request_shape,
                        "provider_error": str(data.get("error") or "")[:180],
                        "fetch_duration_ms": int((time.monotonic() - attempt_started) * 1000),
                        "result_bucket": "provider_error",
                        **response_flags,
                    },
                )
                raise BookingOptionsFetchError(
                    "booking_options_provider_error",
                    context={
                        **request_shape,
                        "provider_error": str(data.get("error") or "")[:180],
                        **response_flags,
                    },
                )
            with contextlib.suppress(Exception):
                await api_key_manager.record_usage("serpapi", idx)
            logger.info(
                "booking_options fetch succeeded",
                extra={
                    **request_shape,
                    "fetch_duration_ms": int((time.monotonic() - attempt_started) * 1000),
                    "result_bucket": "artifact_payload",
                    **response_flags,
                },
            )
            return data

        except BookingOptionsFetchError:
            raise
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            if attempt < max_attempts:
                await asyncio.sleep(BOOKING_OPTIONS_RETRY_BACKOFF * attempt)
                continue
            exception_bucket = _classify_booking_options_exception(e)
            token_fp = _token_fingerprint(booking_token)
            candidate_probe_context = bool(
                booking_token or (departure_id and arrival_id and outbound_date)
            )
            should_emit, occurrence = _should_emit_candidate_fallback_log(
                exception_bucket=exception_bucket,
                token_fp=token_fp,
                route_type=route_type,
                candidate_probe_context=candidate_probe_context,
            )
            if should_emit:
                logger.debug(
                    "fetch_booking_options expected transient exception; candidate fallback",
                    extra={
                        "attempt": attempt,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "exception_bucket": exception_bucket,
                        "fetch_duration_ms": int((time.monotonic() - attempt_started) * 1000),
                        "result_bucket": "request_exception",
                        "has_booking_token": bool(booking_token),
                        "has_departure_id": bool(departure_id),
                        "has_arrival_id": bool(arrival_id),
                        "has_outbound_date": bool(outbound_date),
                        "has_return_date": bool(return_date),
                        "has_departure_token": False,
                        "route_type": route_type,
                        "token_fp": token_fp,
                        "key_source": "api_key_manager.reserve_key:serpapi",
                        "candidate_probe_context": candidate_probe_context,
                        "occurrence": occurrence,
                        "suppressed_prior_occurrences": max(0, occurrence - 1),
                        **response_flags,
                    },
                )
            raise BookingOptionsFetchError(
                "booking_options_request_exception",
                context={
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "exception_bucket": exception_bucket,
                    "has_booking_token": bool(booking_token),
                    "has_departure_id": bool(departure_id),
                    "has_arrival_id": bool(arrival_id),
                    "has_outbound_date": bool(outbound_date),
                    "has_return_date": bool(return_date),
                    "has_departure_token": False,
                    "route_type": route_type,
                    "token_fp": token_fp,
                    "key_source": "api_key_manager.reserve_key:serpapi",
                    **response_flags,
                },
            )
        except Exception as e:
            exception_bucket = _classify_booking_options_exception(e)
            token_fp = _token_fingerprint(booking_token)
            candidate_probe_context = bool(
                booking_token or (departure_id and arrival_id and outbound_date)
            )
            if exception_bucket == "unexpected":
                if candidate_probe_context:
                    # In bounded candidate probing, isolated unexpected per-candidate failures
                    # can be expected while other candidates still resolve successfully.
                    log_severity = "info"
                    log_fn = logger.info
                else:
                    log_severity = "warning"
                    log_fn = logger.warning
            elif exception_bucket in {"no_active_key", "provider_rate_limited", "provider_auth"}:
                log_severity = "debug" if candidate_probe_context else "info"
                log_fn = logger.debug if candidate_probe_context else logger.info
            else:
                log_severity = "debug"
                log_fn = logger.debug
            should_emit, occurrence = _should_emit_candidate_fallback_log(
                exception_bucket=exception_bucket,
                token_fp=token_fp,
                route_type=route_type,
                candidate_probe_context=candidate_probe_context,
            )
            if should_emit:
                log_fn(
                    (
                        "fetch_booking_options unexpected exception; candidate fallback"
                        if exception_bucket == "unexpected" and candidate_probe_context
                        else "fetch_booking_options exception; falling back"
                        if exception_bucket == "unexpected"
                        else "fetch_booking_options expected exception; candidate fallback"
                    ),
                    extra={
                        "attempt": attempt,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "exception_bucket": exception_bucket,
                        "log_severity": log_severity,
                        "candidate_probe_context": candidate_probe_context,
                        "fetch_duration_ms": int((time.monotonic() - attempt_started) * 1000),
                        "result_bucket": "request_exception",
                        "has_booking_token": bool(booking_token),
                        "has_departure_id": bool(departure_id),
                        "has_arrival_id": bool(arrival_id),
                        "has_outbound_date": bool(outbound_date),
                        "has_return_date": bool(return_date),
                        "has_departure_token": False,
                        "route_type": route_type,
                        "token_fp": token_fp,
                        "key_source": "api_key_manager.reserve_key:serpapi",
                        "occurrence": occurrence,
                        "suppressed_prior_occurrences": max(0, occurrence - 1),
                        **response_flags,
                    },
                )
            raise BookingOptionsFetchError(
                "booking_options_request_exception",
                context={
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "exception_bucket": exception_bucket,
                    "has_booking_token": bool(booking_token),
                    "has_departure_id": bool(departure_id),
                    "has_arrival_id": bool(arrival_id),
                    "has_outbound_date": bool(outbound_date),
                    "has_return_date": bool(return_date),
                    "has_departure_token": False,
                    "route_type": route_type,
                    "token_fp": token_fp,
                    "key_source": "api_key_manager.reserve_key:serpapi",
                    **response_flags,
                },
            )

    raise BookingOptionsFetchError(
        "booking_options_exhausted",
        context={
            "has_booking_token": bool(booking_token),
            "has_departure_id": bool(departure_id),
            "has_arrival_id": bool(arrival_id),
            "has_outbound_date": bool(outbound_date),
            "has_return_date": bool(return_date),
            "has_departure_token": False,
            "route_type": route_type,
            "token_fp": _token_fingerprint(booking_token),
        },
    )


async def fetch_booking_options(
    booking_token: str,
    *,
    departure_id: Optional[str] = None,
    arrival_id: Optional[str] = None,
    outbound_date: Optional[str] = None,
    return_date: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Call SerpAPI's google_flights_booking engine and return normalized booking options.
    """
    try:
        data = await _fetch_booking_options_payload(
            booking_token=booking_token,
            departure_id=departure_id,
            arrival_id=arrival_id,
            outbound_date=outbound_date,
            return_date=return_date,
        )
    except BookingOptionsFetchError:
        return None
    if not data:
        return None

    options = _extract_booking_options_from_payload(data)
    if not options:
        logger.info("No valid booking options found; falling back")
        return None

    logger.info("Fetched %d booking options from SerpAPI", len(options))
    return options


async def best_booking_option(
    booking_token: str,
    *,
    departure_id: Optional[str] = None,
    arrival_id: Optional[str] = None,
    outbound_date: Optional[str] = None,
    return_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch all booking options and return the cheapest one.

    Returns a dict with 'provider', 'price', 'link' keys, or None if no options.
    """
    options = await fetch_booking_options(
        booking_token,
        departure_id=departure_id,
        arrival_id=arrival_id,
        outbound_date=outbound_date,
        return_date=return_date,
    )
    if not options:
        return None
    priced = [opt for opt in options if opt.get("price") is not None]
    if priced:
        best = min(priced, key=lambda x: x["price"])
        logger.info(f"Best booking option: {best['provider']} at {best['price']}")
    else:
        best = options[0]
        logger.info("Best booking option selected without numeric price; using first valid booking link")
    return best


async def resolve_booking_token(
    booking_token: str,
    *,
    departure_id: Optional[str] = None,
    arrival_id: Optional[str] = None,
    outbound_date: Optional[str] = None,
    return_date: Optional[str] = None,
) -> Optional[str]:
    """
    Return the direct booking URL from the cheapest option found via SerpAPI.
    Returns None if resolution fails (including 401/429 or no options).
    """
    best = await best_booking_option(
        booking_token,
        departure_id=departure_id,
        arrival_id=arrival_id,
        outbound_date=outbound_date,
        return_date=return_date,
    )
    return best["link"] if best else None


async def resolve_booking_token_with_details(
    booking_token: str,
    *,
    departure_id: Optional[str] = None,
    arrival_id: Optional[str] = None,
    outbound_date: Optional[str] = None,
    return_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve booking token with explicit machine-readable classification.
    """
    resolution_started = time.monotonic()
    cache_key = _booking_resolution_cache_key(
        booking_token=booking_token,
        departure_id=departure_id,
        arrival_id=arrival_id,
        outbound_date=outbound_date,
        return_date=return_date,
    )
    with _handoff_cache_lock:
        cached = _booking_resolution_cache.get(cache_key)
    if isinstance(cached, dict):
        cached_kind = str(cached.get("kind") or "")
        if cached_kind == "direct_booking":
            cached_url = str(cached.get("url") or "").strip()
            if _is_usable_handoff_url(cached_url):
                logger.info(
                    "booking_token_resolution cache hit",
                    extra={
                        "cache_key": cache_key,
                        "cache_kind": cached_kind,
                        "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                    },
                )
                return {
                    "url": cached_url,
                    "source": "booking_token",
                    "reason": "resolved_booking_token_cache",
                    "status": "ok",
                    "provider": "serpapi",
                    "handoff_mode": "direct_booking",
                    "landing_guarantee": "partner_specific",
                    "is_exact_handoff": True,
                    "is_search_fallback": False,
                    "is_provider_managed": True,
                    "is_booking_quality_exit": True,
                    "booking_exit_quality": "booking_ready",
                    "cache_hit": True,
                }
        if cached_kind == "booking_request_post":
            request_url = str(cached.get("url") or "").strip()
            post_data = cached.get("post_data")
            request_headers = cached.get("headers") or {}
            bridge_url = register_post_handoff_artifact(
                url=request_url,
                post_data=post_data,
                headers=request_headers,
            )
            if bridge_url:
                logger.info(
                    "booking_token_resolution cache hit",
                    extra={
                        "cache_key": cache_key,
                        "cache_kind": cached_kind,
                        "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                    },
                )
                return {
                    "url": bridge_url,
                    "source": "booking_token",
                    "reason": "resolved_booking_request_post_cache",
                    "status": "ok",
                    "provider": "serpapi",
                    "handoff_mode": "provider_or_shareable",
                    "landing_guarantee": "provider_managed_post_bridge",
                    "artifact_field": "booking_request.post_followup_bridge",
                    "requires_browser_post": True,
                    "is_exact_handoff": False,
                    "is_search_fallback": False,
                    "is_provider_managed": True,
                    "is_booking_quality_exit": True,
                    "booking_exit_quality": "booking_ready",
                    "cache_hit": True,
                }
        if cached_kind == "booking_request_get":
            cached_url = str(cached.get("url") or "").strip()
            if _is_usable_handoff_url(cached_url):
                logger.info(
                    "booking_token_resolution cache hit",
                    extra={
                        "cache_key": cache_key,
                        "cache_kind": cached_kind,
                        "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                    },
                )
                return {
                    "url": cached_url,
                    "source": "booking_token",
                    "reason": "resolved_booking_request_cache",
                    "status": "ok",
                    "provider": "serpapi",
                    "handoff_mode": "provider_or_shareable",
                    "landing_guarantee": "provider_managed",
                    "artifact_field": "booking_request.url",
                    "is_exact_handoff": False,
                    "is_search_fallback": False,
                    "is_provider_managed": True,
                    "is_booking_quality_exit": True,
                    "booking_exit_quality": "booking_ready",
                    "cache_hit": True,
                }

    resolve_timeout = BOOKING_TOKEN_RESOLVE_TIMEOUT + (
        BOOKING_OPTIONS_ROUND_TRIP_TIMEOUT_BONUS if return_date else 0.0
    )
    try:
        payload = await asyncio.wait_for(
            _fetch_booking_options_payload(
                booking_token=booking_token,
                departure_id=departure_id,
                arrival_id=arrival_id,
                outbound_date=outbound_date,
                return_date=return_date,
            ),
            timeout=resolve_timeout,
        )
    except asyncio.TimeoutError:
        logger.info(
            "booking_token_resolution completed",
            extra={
                "result_bucket": "timeout",
                "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
            },
        )
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_resolution_timeout",
            "status": "unavailable",
            "provider": "serpapi",
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
    except BookingOptionsFetchError as e:
        logger.info(
            "booking_token_resolution completed",
            extra={
                "result_bucket": e.reason or "request_exception",
                "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
            },
        )
        return {
            "url": None,
            "source": "booking_token",
            "reason": e.reason or "booking_options_request_exception",
            "status": "unavailable",
            "provider": "serpapi",
            "handoff_mode": "unavailable",
            "landing_guarantee": "none",
            "is_exact_handoff": False,
            "is_search_fallback": False,
            "is_provider_managed": False,
            "is_booking_quality_exit": False,
            "booking_exit_quality": "unavailable",
            "failure_bucket": e.context.get("exception_bucket") or e.reason,
            "cache_hit": False,
        }
    except Exception:
        logger.info(
            "booking_token_resolution completed",
            extra={
                "result_bucket": "exception",
                "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
            },
        )
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_resolution_exception",
            "status": "unavailable",
            "provider": "serpapi",
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

    options = _extract_booking_options_from_payload(payload or {})
    best = None
    if options:
        priced = [opt for opt in options if opt.get("price") is not None]
        best = min(priced, key=lambda x: x["price"]) if priced else options[0]

    if best and best.get("link"):
        link = str(best.get("link")).strip()
        if not _is_usable_handoff_url(link):
            logger.info(
                "booking_token_resolution completed",
                extra={
                    "result_bucket": "invalid_direct_link",
                    "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                },
            )
            return {
                "url": None,
                "source": "booking_token",
                "reason": "booking_token_invalid_url",
                "status": "unavailable",
                "provider": "serpapi",
                "handoff_mode": "unavailable",
                "landing_guarantee": "none",
                "is_exact_handoff": False,
                "is_search_fallback": False,
                "is_provider_managed": False,
                "is_booking_quality_exit": False,
                "booking_exit_quality": "unavailable",
            }
        _cache_booking_resolution(cache_key, {"kind": "direct_booking", "url": link})
        logger.info(
            "booking_token_resolution completed",
            extra={
                "result_bucket": "direct_booking",
                "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
            },
        )
        return {
            "url": link,
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "status": "ok",
            "provider": "serpapi",
            "handoff_mode": "direct_booking",
            "landing_guarantee": "partner_specific",
            "is_exact_handoff": True,
            "is_search_fallback": False,
            "is_provider_managed": True,
            "is_booking_quality_exit": True,
            "booking_exit_quality": "booking_ready",
            "cache_hit": False,
        }

    booking_request = _extract_booking_request_artifact_from_payload(payload or {})
    if booking_request and booking_request.get("url"):
        request_url = str(booking_request.get("url")).strip()
        post_data = booking_request.get("post_data")
        request_headers = booking_request.get("headers") or {}
        method = str(booking_request.get("method") or "").upper()

        if method == "POST" and post_data not in (None, "", {}, []):
            _cache_booking_resolution(
                cache_key,
                {
                    "kind": "booking_request_post",
                    "url": request_url,
                    "post_data": post_data,
                    "headers": request_headers,
                },
            )
            bridge_url = register_post_handoff_artifact(
                url=request_url,
                post_data=post_data,
                headers=request_headers,
            )
            if bridge_url:
                logger.info(
                    "booking_token_resolution completed",
                    extra={
                        "result_bucket": "booking_request_post",
                        "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                    },
                )
                return {
                    "url": bridge_url,
                    "source": "booking_token",
                    "reason": "resolved_booking_request_post",
                    "status": "ok",
                    "provider": "serpapi",
                    "handoff_mode": "provider_or_shareable",
                    "landing_guarantee": "provider_managed_post_bridge",
                    "artifact_field": "booking_request.post_followup_bridge",
                    "requires_browser_post": True,
                    "is_exact_handoff": False,
                    "is_search_fallback": False,
                    "is_provider_managed": True,
                    "is_booking_quality_exit": True,
                    "booking_exit_quality": "booking_ready",
                    "cache_hit": False,
                }
        if _is_usable_handoff_url(request_url):
            _cache_booking_resolution(
                cache_key,
                {
                    "kind": "booking_request_get",
                    "url": request_url,
                },
            )
            logger.info(
                "booking_token_resolution completed",
                extra={
                    "result_bucket": "booking_request_get",
                    "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                },
            )
            return {
                "url": request_url,
                "source": "booking_token",
                "reason": "resolved_booking_request",
                "status": "ok",
                "provider": "serpapi",
                "handoff_mode": "provider_or_shareable",
                "landing_guarantee": "provider_managed",
                "artifact_field": "booking_request.url",
                "is_exact_handoff": False,
                "is_search_fallback": False,
                "is_provider_managed": True,
                "is_booking_quality_exit": True,
                "booking_exit_quality": "booking_ready",
                "cache_hit": False,
            }

    logger.info(
        "booking_token_resolution completed",
        extra={
            "result_bucket": "no_usable_artifact",
            "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
        },
    )
    return {
        "url": None,
        "source": "booking_token",
        "reason": "booking_token_unresolved",
        "status": "unavailable",
        "provider": "serpapi",
        "handoff_mode": "unavailable",
        "landing_guarantee": "none",
        "is_exact_handoff": False,
        "is_search_fallback": False,
        "is_provider_managed": False,
        "is_booking_quality_exit": False,
        "booking_exit_quality": "unavailable",
        "cache_hit": False,
    }


def build_google_flights_fallback(
    *,
    origin: str,
    destination: str,
    depart_date: str,
    return_date: Optional[str] = None,
    flight: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a clean, HTTPS Google Flights search URL as a last-resort fallback.

    Uses the /travel/flights?q= format which is stable and works without
    any special parameters.
    """
    query_parts: List[str] = [
        f"Flights from {origin} airport to {destination} airport on {depart_date}",
        f"IATA {origin} {destination}",
    ]
    if return_date:
        query_parts.append("round trip")
        query_parts.append(f"returning {return_date}")
        # Keep return-leg phrasing explicit so fallback remains useful on weak
        # round-trip routes while staying compact for parser compatibility.
        query_parts.append(f"return flight {destination} to {origin} on {return_date}")

    # Add itinerary hints so fallback remains differentiated/useful per flight.
    # Keep phrasing parser-friendly to avoid generic search landing pages.
    if isinstance(flight, dict):
        airline = str(flight.get("airline") or "").strip()
        flight_no = str(flight.get("flight_no") or "").strip()
        departure_time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", str(flight.get("departure_time") or ""))
        arrival_time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", str(flight.get("arrival_time") or ""))
        layover_airports_raw = flight.get("layover_airports")
        layover_airports: List[str] = []
        if isinstance(layover_airports_raw, list):
            for item in layover_airports_raw:
                token = str(item or "").strip().upper()
                if re.fullmatch(r"[A-Z]{3}", token):
                    layover_airports.append(token)
        stops_value = flight.get("stops")
        stops: Optional[int]
        try:
            stops = int(stops_value) if stops_value is not None else None
        except (TypeError, ValueError):
            stops = None

        if airline and flight_no:
            query_parts.append(f"{airline} {flight_no}")
        elif airline:
            query_parts.append(airline)
        elif flight_no:
            query_parts.append(flight_no)

        dep_time = (
            f"{int(departure_time_match.group(1)):02d}:{departure_time_match.group(2)}"
            if departure_time_match
            else None
        )
        arr_time = (
            f"{int(arrival_time_match.group(1)):02d}:{arrival_time_match.group(2)}"
            if arrival_time_match
            else None
        )
        if dep_time and arr_time:
            query_parts.append(f"{dep_time} to {arr_time}")
        elif dep_time:
            query_parts.append(f"depart {dep_time}")
        elif arr_time:
            query_parts.append(f"arrive {arr_time}")

        if layover_airports:
            # Keep only first two IATA hints so query stays compact and parseable.
            query_parts.append("via " + " ".join(layover_airports[:2]))
        elif stops and stops > 0:
            query_parts.append(f"{stops} stop{'s' if stops != 1 else ''}")

    params = {
        "q": " ".join(query_parts),
        "hl": "en",
        "gl": "in",
        "curr": "INR",
    }
    return f"https://www.google.com/travel/flights?{urllib.parse.urlencode(params)}"


def _select_shareable_or_partner_link(flight: Dict[str, Any]) -> tuple[Optional[str], Optional[str], bool]:
    """
    Returns (url, source_field, had_invalid_shareable_flag).
    """
    candidate_fields = (
        "shareable_link",
        "provider_link",
        "partner_booking_link",
        "booking_url",
        "partner_url",
        "handoff_url",
        "link",
        "url",
    )
    had_invalid_shareable = False
    for field in candidate_fields:
        raw = flight.get(field)
        if not raw:
            continue
        canonical = _canonicalize_handoff_url(str(raw))
        if not canonical:
            if field == "shareable_link":
                had_invalid_shareable = True
            continue
        # Treat generic Google travel search URLs as fallback-equivalent unless
        # they were explicitly provided as shareable_link.
        if field != "shareable_link" and _is_google_search_fallback_url(canonical):
            continue
        return canonical, field, had_invalid_shareable

    def _extract_nested_link(node: Any, path: str, depth: int) -> tuple[Optional[str], Optional[str]]:
        if depth > 3:
            return None, None
        link_keys = ("booking_url", "booking_link", "deeplink", "redirect_link", "link", "url")
        container_keys = ("book_with", "providers", "seller", "booking", "options", "booking_options", "tickets", "offers", "offer")

        if isinstance(node, dict):
            for lk in link_keys:
                raw_link = node.get(lk)
                if not raw_link:
                    continue
                canonical_link = _canonicalize_handoff_url(str(raw_link))
                if not canonical_link:
                    continue
                if _is_google_search_fallback_url(canonical_link):
                    continue
                return canonical_link, f"{path}.{lk}" if path else lk
            for ck in container_keys:
                if ck not in node:
                    continue
                nested_link, nested_field = _extract_nested_link(
                    node[ck],
                    f"{path}.{ck}" if path else ck,
                    depth + 1,
                )
                if nested_link:
                    return nested_link, nested_field
            return None, None

        if isinstance(node, list):
            for idx, item in enumerate(node[:10]):
                nested_link, nested_field = _extract_nested_link(item, f"{path}[{idx}]", depth + 1)
                if nested_link:
                    return nested_link, nested_field
        return None, None

    for root in ("booking_options", "book_with", "providers", "seller", "tickets", "offers", "offer", "booking"):
        if root not in flight:
            continue
        nested_link, nested_field = _extract_nested_link(flight[root], root, 0)
        if nested_link:
            return nested_link, nested_field or root, had_invalid_shareable

    return None, None, had_invalid_shareable


async def build_booking_handoff_url(
    *,
    flight: dict,
    origin: str,
    destination: str,
    depart_date: str,
    return_date: Optional[str] = None,
    passengers: int = 1,
    return_details: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Resolve the best possible booking deep-link for a flight, in priority order:

    1. SerpAPI booking_token  → calls the SerpAPI booking engine to get the
                                cheapest airline-native checkout URL.
    2. shareable_link         → the SerpAPI-provided shareable Google Flights link
                                which pre-fills the exact itinerary.
    3. Google Flights fallback→ a clean /travel/flights?q= search URL.

    Args:
        flight: Full flight dict from planner (must include 'booking_token'
                and/or 'shareable_link' keys populated by airline_api.py).
        origin, destination: IATA codes.
        depart_date: YYYY-MM-DD.
        return_date: YYYY-MM-DD or None for one-way.
        passengers: Number of adult passengers.

    Returns:
        str: The best available booking URL (default behavior).
        dict: When return_details=True, returns compact source classification metadata.
    """
    def _detail_payload(
        *,
        url: Optional[str],
        source: str,
        reason: str,
        provider: Optional[str] = None,
        handoff_mode: Optional[str] = None,
        landing_guarantee: Optional[str] = None,
        artifact_field: Optional[str] = None,
        search_fallback_quality: Optional[str] = None,
        search_fallback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        status = "ok" if url else "unavailable"
        mode = handoff_mode or "unavailable"
        payload: Dict[str, Any] = {
            "url": url,
            "source": source,
            "reason": reason,
            "status": status,
            "is_exact_handoff": mode == "direct_booking",
            "is_search_fallback": mode == "search_fallback",
            "is_provider_managed": mode in {"direct_booking", "provider_or_shareable"},
            "is_booking_quality_exit": mode in {"direct_booking", "provider_or_shareable"},
            "booking_exit_quality": (
                "booking_ready"
                if mode in {"direct_booking", "provider_or_shareable"}
                else ("search_assist" if mode == "search_fallback" else "unavailable")
            ),
        }
        if provider:
            payload["provider"] = provider
        payload["handoff_mode"] = mode
        payload["landing_guarantee"] = landing_guarantee or "none"
        if artifact_field:
            payload["artifact_field"] = artifact_field
        if mode == "search_fallback" and search_fallback_quality:
            payload["search_fallback_quality"] = search_fallback_quality
        if mode == "search_fallback" and isinstance(search_fallback_context, dict):
            payload["search_fallback_context"] = search_fallback_context
        return payload

    # ── Priority 1: SerpAPI booking_token ────────────────────────────────
    booking_token = flight.get("booking_token")
    token_result: Optional[Dict[str, Any]] = None
    if booking_token:
        try:
            token_result = await resolve_booking_token_with_details(
                booking_token,
                departure_id=origin,
                arrival_id=destination,
                outbound_date=depart_date,
                return_date=return_date,
            )
        except TypeError:
            # Backward-compatible path for tests that monkeypatch this helper
            # with older signatures.
            token_result = await resolve_booking_token_with_details(booking_token)
        resolved = token_result.get("url") if isinstance(token_result, dict) else None
        if token_result and token_result.get("requires_browser_post") and isinstance(resolved, str) and resolved.startswith("/"):
            logger.info("Handoff URL resolved via POST-capable booking artifact bridge")
            if return_details:
                token_result["artifact_field"] = token_result.get("artifact_field") or "booking_request.post_followup_bridge"
                return token_result
            return resolved
        canonical_resolved = _canonicalize_handoff_url(str(resolved)) if resolved else None
        if canonical_resolved:
            logger.info("Handoff URL resolved via SerpAPI booking_token (cheapest option)")
            if return_details:
                token_result["url"] = canonical_resolved
                if token_result.get("handoff_mode") != "provider_or_shareable":
                    token_result["handoff_mode"] = "direct_booking"
                    token_result["landing_guarantee"] = "partner_specific"
                    token_result["is_exact_handoff"] = True
                    token_result["is_search_fallback"] = False
                    token_result["is_provider_managed"] = True
                    token_result["is_booking_quality_exit"] = True
                    token_result["booking_exit_quality"] = "booking_ready"
                return token_result
            return canonical_resolved
        if resolved and not canonical_resolved:
            token_result = {
                "url": None,
                "source": "booking_token",
                "reason": "booking_token_invalid_url",
                "status": "unavailable",
                "provider": "serpapi",
            }
        logger.info("booking_token resolution unavailable; falling through to shareable_link/google fallback")

    # ── Priority 2: shareable/provider link artifacts ─────────────────────
    canonical_shareable, shareable_field, had_invalid_shareable = _select_shareable_or_partner_link(flight)
    if canonical_shareable:
        logger.info("Handoff URL resolved via %s", shareable_field or "shareable_link")
        if return_details:
            if token_result:
                token_reason = token_result.get("reason") or "booking_token_unresolved"
                if shareable_field == "shareable_link":
                    reason = f"{token_reason}_fallback_shareable"
                else:
                    reason = f"{token_reason}_fallback_partner_link"
            else:
                reason = "shareable_link_available" if shareable_field == "shareable_link" else f"partner_link_{shareable_field}_available"
            return _detail_payload(
                url=canonical_shareable,
                source="shareable_link",
                reason=reason,
                provider="serpapi",
                handoff_mode="provider_or_shareable",
                landing_guarantee="provider_managed",
                artifact_field=shareable_field,
            )
        return canonical_shareable
    if had_invalid_shareable:
        logger.info("shareable_link is present but invalid; falling through to stronger links/google fallback")
        if not token_result:
            token_result = {
                "reason": "invalid_shareable_link",
                "source": "shareable_link",
                "status": "unavailable",
                "provider": "serpapi",
            }

    # ── Priority 2.5: booking_request follow-up artifacts ────────────────
    booking_request_url, booking_request_field = await _resolve_booking_request_handoff(flight)
    if booking_request_url:
        logger.info("Handoff URL resolved via %s", booking_request_field or "booking_request")
        if return_details:
            if token_result:
                token_reason = token_result.get("reason") or "booking_token_unresolved"
                reason = f"{token_reason}_fallback_partner_link"
            else:
                reason = "partner_link_booking_request_available"
            return _detail_payload(
                url=booking_request_url,
                source="shareable_link",
                reason=reason,
                provider="serpapi",
                handoff_mode="provider_or_shareable",
                landing_guarantee="provider_managed",
                artifact_field=booking_request_field or "booking_request",
            )
        return booking_request_url

    # ── Priority 3: Google Flights fallback ──────────────────────────────
    logger.info("Handoff URL falling back to Google Flights search URL")
    fallback_url = build_google_flights_fallback(
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        return_date=return_date,
        flight=flight,
    )
    if return_details:
        search_fallback_context = _search_fallback_context(
            origin=origin,
            destination=destination,
            depart_date=depart_date,
            return_date=return_date,
            flight=flight,
        )
        search_fallback_quality = _legacy_search_fallback_quality_for_clients(
            search_fallback_context.get("quality_tier", "")
        )
        if token_result:
            token_reason = token_result.get("reason") or "booking_token_unresolved"
            reason = f"{token_reason}_google_search_fallback"
        else:
            reason = "no_booking_artifacts_google_search_fallback"
        return _detail_payload(
            url=fallback_url,
            source="google_flights_fallback",
            reason=reason,
            provider="google_flights",
            handoff_mode="search_fallback",
            landing_guarantee="best_effort_search",
            search_fallback_quality=search_fallback_quality,
            search_fallback_context=search_fallback_context,
        )
    return fallback_url


# ----------------------------------------------------------------------
# Core booking CRUD
# ----------------------------------------------------------------------

async def hold_booking(
    *,
    flight: dict,
    origin: str,
    destination: str,
    depart_date: str,
    passenger: Optional[dict] = None,
    hold_minutes: int = 15,
    return_date: Optional[str] = None,
    passengers: int = 1,
) -> dict:
    """
    Create a HELD booking record and resolve + store the best handoff URL.

    Returns a dict with:
        id            – database booking id
        status        – "HELD"
        handoff_url   – resolved deep-link for the user to complete purchase
        expires_at    – ISO-8601 string of when the hold expires
    """
    # Resolve the URL before writing to DB so it's immediately available
    handoff_url = await build_booking_handoff_url(
        flight=flight,
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        return_date=return_date,
        passengers=passengers,
    )

    def _db_operations():
        db = SessionLocal()
        try:
            booking = Booking(
                status="HELD",
                flight=flight,
                passenger=passenger,
                booking_token=flight.get("booking_token"),
                shareable_link=flight.get("shareable_link"),
                handoff_url=handoff_url,
                expires_at=datetime.utcnow() + timedelta(minutes=hold_minutes),
            )
            db.add(booking)
            db.commit()
            db.refresh(booking)

            logger.info(
                "Booking held",
                extra={
                    "booking_id": booking.id,
                    "flight_no": flight.get("flight_no"),
                    "handoff_url_source": (
                        "serpapi_token" if flight.get("booking_token") else
                        "shareable_link" if flight.get("shareable_link") else
                        "google_fallback"
                    ),
                }
            )

            return booking.id, booking.status, booking.expires_at.isoformat()
        finally:
            db.close()

    b_id, b_status, b_expires = await asyncio.to_thread(_db_operations)

    return {
        "id": b_id,
        "status": b_status,
        "handoff_url": handoff_url,
        "expires_at": b_expires,
    }


def get_booking(booking_id: int) -> Optional[dict]:
    """Retrieve a booking by id. Returns None if not found."""
    db = SessionLocal()
    try:
        b = db.get(Booking, booking_id)
        if not b:
            return None
        return {
            "id":             b.id,
            "status":         b.status,
            "flight":         b.flight,
            "passenger":      b.passenger,
            "booking_token":  b.booking_token,
            "shareable_link": b.shareable_link,
            "handoff_url":    b.handoff_url,
            "created_at":     b.created_at.isoformat() if b.created_at else None,
            "expires_at":     b.expires_at.isoformat() if b.expires_at else None,
        }
    finally:
        db.close()


def confirm_booking(booking_id: int) -> bool:
    """
    Confirm a HELD booking. Returns False if the hold has expired or the
    booking is not in HELD status.
    """
    db = SessionLocal()
    try:
        b = db.get(Booking, booking_id)
        if not b:
            return False
        if b.expires_at and b.expires_at < datetime.utcnow():
            b.status = "EXPIRED"
            db.commit()
            logger.info("Booking auto-expired during confirm attempt", extra={"booking_id": booking_id})
            return False
        if b.status != "HELD":
            return False
        b.status = "CONFIRMED"
        db.commit()
        logger.info("Booking confirmed", extra={"booking_id": booking_id})
        return True
    finally:
        db.close()


def cancel_booking(booking_id: int) -> bool:
    """
    Cancel a HELD booking. Cannot cancel a CONFIRMED or EXPIRED booking.
    """
    db = SessionLocal()
    try:
        b = db.get(Booking, booking_id)
        if not b:
            return False
        if b.status in ("CONFIRMED", "EXPIRED"):
            return False
        b.status = "CANCELLED"
        db.commit()
        logger.info("Booking cancelled", extra={"booking_id": booking_id})
        return True
    finally:
        db.close()


def expire_bookings() -> int:
    """
    Bulk-expire all HELD bookings whose hold window has elapsed.
    Intended to be called by a periodic scheduler (e.g. every 5 minutes).

    Returns:
        int: Number of bookings transitioned to EXPIRED.
    """
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        stale = (
            db.query(Booking)
            .filter(Booking.status == "HELD", Booking.expires_at < now)
            .all()
        )
        count = len(stale)
        for b in stale:
            b.status = "EXPIRED"
        db.commit()
        if count:
            logger.info("Bulk-expired stale bookings", extra={"count": count})
        return count
    finally:
        db.close()


def get_active_held_bookings() -> list[dict]:
    """
    Return all HELD bookings that have not yet expired.
    Used by the price tracker to know which routes to monitor.
    """
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        rows = (
            db.query(Booking)
            .filter(Booking.status == "HELD", Booking.expires_at > now)
            .all()
        )
        return [
            {
                "id":            b.id,
                "flight":        b.flight,
                "booking_token": b.booking_token,
                "handoff_url":   b.handoff_url,
                "expires_at":    b.expires_at.isoformat() if b.expires_at else None,
            }
            for b in rows
        ]
    finally:
        db.close()
