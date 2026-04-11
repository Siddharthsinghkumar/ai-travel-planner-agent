# tools/booking_handoff.py
"""
Booking Handoff Tool

Responsibilities:
- Hold, cancel, and expire local booking-follow-up records in the database
- Resolve booking handoff strictly through SerpAPI booking-token artifacts:
    1. SerpAPI booking_token  → /search?engine=google_flights_booking follow-ups
       that produce a replayable non-Google provider checkout URL.
    2. If provider checkout cannot be resolved, booking handoff is unavailable.

No Google Flights fallback/search-assist URL is emitted by booking flows.
"""

import asyncio
import contextlib
import html
import hashlib
import logging
import re
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Tuple

import httpx
from cachetools import TTLCache
from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, and_, or_
from agents.database import Base, SessionLocal, get_engine
from core.api_key_manager import key_manager as api_key_manager
from core.env_config import get_env_bool, get_env_float, get_env_int
from core.http_client import get_client
from core.request_context import get_request_id
import core.metrics as app_metrics

logger = logging.getLogger(__name__)
BOOKING_OPTIONS_HTTP_TIMEOUT = get_env_float("BOOKING_OPTIONS_HTTP_TIMEOUT", 2.2)
BOOKING_OPTIONS_RETRIES = max(1, get_env_int("BOOKING_OPTIONS_RETRIES", 2))
BOOKING_OPTIONS_RETRY_BACKOFF = get_env_float("BOOKING_OPTIONS_RETRY_BACKOFF", 0.15)
BOOKING_OPTIONS_ROUND_TRIP_TIMEOUT_BONUS = max(
    0.0,
    get_env_float("BOOKING_OPTIONS_ROUND_TRIP_TIMEOUT_BONUS", 0.35),
)
BOOKING_OPTIONS_ATTEMPTS_BUDGET = max(2, min(4, BOOKING_OPTIONS_RETRIES + 1))
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
BOOKING_REQUEST_RESPONSE_SNIPPET_BYTES = max(
    1200,
    get_env_int("BOOKING_REQUEST_RESPONSE_SNIPPET_BYTES", 12000),
)
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
_META_REFRESH_TAG_RE = re.compile(
    r"<meta[^>]*http-equiv\s*=\s*['\"]?\s*refresh\s*['\"]?[^>]*>",
    re.IGNORECASE,
)


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
    # Keep throttle scope request-level so candidate bursts do not spam logs.
    # `token_fp` is still recorded in log extras for diagnostics.
    if request_id == "unknown":
        return f"{request_id}:{exception_bucket}:{route_type}:{token_fp}"
    return f"{request_id}:{exception_bucket}:{route_type}"


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
    if occurrence in {8, 16, 32}:
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


def _classify_http_status(status_code: Optional[int]) -> str:
    if status_code in {401, 403}:
        return "auth_failure"
    if status_code == 429:
        return "rate_limit"
    if status_code is None:
        return "unknown_failure"
    if 400 <= int(status_code) < 500:
        return "request_failure"
    if int(status_code) >= 500:
        return "provider_failure"
    return "unknown_failure"


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
    include_airlines: Optional[str] = None,
    deep_search: Optional[bool] = None,
    travel_class: Optional[str] = None,
    adults: Optional[int] = None,
    currency: Optional[str] = None,
    hl: Optional[str] = None,
) -> str:
    raw = "|".join(
        [
            str(booking_token or ""),
            str(departure_id or ""),
            str(arrival_id or ""),
            str(outbound_date or ""),
            str(return_date or ""),
            str(include_airlines or ""),
            "1" if deep_search else "0",
            str(travel_class or ""),
            str(adults or ""),
            str(currency or ""),
            str(hl or ""),
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


def _is_brittle_google_tracker_url(value: Optional[str]) -> bool:
    canonical = _canonicalize_handoff_url(value)
    if not canonical:
        return False
    try:
        parsed = urllib.parse.urlparse(canonical)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    if not _is_google_domain(host):
        return False
    return bool(re.match(r"^/travel/clk(?:/|$)", path))


def _is_usable_handoff_url(value: Optional[str]) -> bool:
    canonical = _canonicalize_handoff_url(value)
    if not canonical:
        return False
    if _is_brittle_google_tracker_url(canonical):
        return False
    if _is_google_search_fallback_url(canonical):
        return False
    return True


def _bridge_target_domain_path(value: Optional[str]) -> Tuple[str, str]:
    canonical = _canonicalize_handoff_url(value)
    if not canonical:
        return "unknown", "unknown"
    try:
        parsed = urllib.parse.urlparse(canonical)
    except Exception:
        return "unknown", "unknown"
    domain = (parsed.netloc or "unknown").lower()
    path = parsed.path or "/"
    return domain, path


def _is_google_domain(host: Optional[str]) -> bool:
    normalized = str(host or "").strip().lower()
    if not normalized:
        return False
    return bool(re.search(r"(^|\.)google\.[a-z.]+$", normalized))


def _domain_for_url(value: Optional[str]) -> Optional[str]:
    canonical = _canonicalize_handoff_url(value)
    if not canonical:
        return None
    try:
        parsed = urllib.parse.urlparse(canonical)
    except Exception:
        return None
    host = (parsed.netloc or "").strip().lower()
    return host or None


def _iter_link_domains(payload: Any) -> Dict[str, int]:
    link_keys = {"url", "link", "booking_url", "booking_link", "deeplink", "redirect_link", "endpoint"}
    domain_counts: Dict[str, int] = {}
    queue: List[Any] = [payload]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key.lower() in link_keys:
                    domain = _domain_for_url(value)
                    if domain:
                        domain_counts[domain] = int(domain_counts.get(domain, 0)) + 1
                elif isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    queue.append(item)
    return domain_counts


def _option_sort_key(option: Dict[str, Any]) -> Tuple[int, float]:
    price = option.get("price")
    if isinstance(price, (int, float)):
        return 0, float(price)
    return 1, float("inf")


def _select_best_replayable_partner_option(options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not options:
        return None
    sorted_options = sorted(options, key=_option_sort_key)
    for option in sorted_options:
        link = str(option.get("link") or "").strip()
        if not _is_usable_handoff_url(link):
            continue
        domain = _domain_for_url(link)
        if not domain or _is_google_domain(domain):
            continue
        return option
    return None


def _summarize_booking_artifact_graph(
    *,
    payload: Dict[str, Any],
    options: List[Dict[str, Any]],
    booking_request: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    option_domains: Dict[str, int] = {}
    replayable_partner_domains: Dict[str, int] = {}
    google_tracker_options = 0
    for option in options:
        link = str(option.get("link") or "").strip()
        domain = _domain_for_url(link)
        if domain:
            option_domains[domain] = int(option_domains.get(domain, 0)) + 1
        if _is_brittle_google_tracker_url(link):
            google_tracker_options += 1
        if _is_usable_handoff_url(link) and domain and not _is_google_domain(domain):
            replayable_partner_domains[domain] = int(replayable_partner_domains.get(domain, 0)) + 1

    booking_request_domain = _domain_for_url((booking_request or {}).get("url"))
    all_domain_counts = _iter_link_domains(payload)
    non_google_domains = sorted([d for d in all_domain_counts.keys() if not _is_google_domain(d)])

    return {
        "inspected_sources": [
            "booking_options[].link",
            "booking_options[].together.link",
            "booking_options[].booking_request",
            "selected_flights[].booking_request",
            "nested_provider_url_fields",
        ],
        "booking_options_count": len(options),
        "booking_option_domains": sorted(option_domains.items(), key=lambda item: item[1], reverse=True)[:10],
        "booking_option_google_tracker_count": google_tracker_options,
        "replayable_partner_option_domains": sorted(
            replayable_partner_domains.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10],
        "booking_request_present": bool(booking_request),
        "booking_request_domain": booking_request_domain,
        "booking_request_method": str((booking_request or {}).get("method") or "").upper() or None,
        "booking_request_has_post_data": bool(
            booking_request and booking_request.get("post_data") not in (None, "", {}, [])
        ),
        "all_url_domain_counts": sorted(all_domain_counts.items(), key=lambda item: item[1], reverse=True)[:12],
        "all_non_google_domains": non_google_domains[:20],
        "has_replayable_partner_option": bool(replayable_partner_domains),
        "has_any_non_google_domain": bool(non_google_domains),
        "only_google_click_or_google_domains": bool(all_domain_counts) and not bool(non_google_domains),
    }


def _build_unverified_google_tracker_post_bridge_payload(
    *,
    bridge_url: str,
    reason: str,
    request_url: str,
    cache_hit: bool,
    artifact_inspection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_domain, target_path = _bridge_target_domain_path(request_url)
    payload = {
        "url": None,
        "diagnostic_handoff_url": bridge_url,
        "source": "booking_token",
        "reason": reason,
        "status": "unavailable",
        "provider": "serpapi",
        "handoff_mode": "unavailable",
        "landing_guarantee": "none",
        "artifact_field": "booking_request.post_followup_bridge",
        "requires_browser_post": True,
        "browser_landing_verdict": "unknown_unverified",
        "bridge_target_domain": target_domain,
        "bridge_target_path": target_path,
        "is_exact_handoff": False,
        "is_search_fallback": False,
        "is_provider_managed": False,
        "is_booking_quality_exit": False,
        "booking_exit_quality": "unavailable",
        "booking_availability": "unavailable_from_upstream_artifacts",
        "booking_unavailability_reason": "no_replayable_partner_booking_url_from_upstream",
        "provider_data_limited": True,
        "cache_hit": cache_hit,
    }
    if isinstance(artifact_inspection, dict):
        payload["artifact_inspection"] = artifact_inspection
        payload["proof_only_google_artifacts"] = bool(
            artifact_inspection.get("only_google_click_or_google_domains")
        ) and not bool(artifact_inspection.get("has_replayable_partner_option"))
    return payload


def _is_google_search_fallback_url(value: Optional[str]) -> bool:
    canonical = _canonicalize_handoff_url(value)
    if not canonical:
        return False
    try:
        parsed = urllib.parse.urlparse(canonical)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    return _is_google_domain(host) and parsed.path == "/travel/flights"


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
    if (
        not canonical_url
        or _is_google_search_fallback_url(canonical_url)
    ):
        return None
    if post_data in (None, "", {}, []):
        return None
    # booking_request POST artifacts often use Google travel/clk endpoints.
    # They are not exposed directly to clients; they are consumed through our
    # one-time bridge and then auto-submitted as POST.

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
        app_metrics.record_booking_handoff_consume(
            lookup_result=diagnostics["lookup_result"],
            outcome=diagnostics["consume_outcome"],
        )
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
        app_metrics.record_booking_handoff_consume(
            lookup_result=diagnostics["lookup_result"],
            outcome=diagnostics["consume_outcome"],
        )
        logger.info("booking_post_bridge_consume_outcome", extra=diagnostics)
        return consumed, diagnostics

    artifact, lookup_result = _consume_post_handoff_artifact_persistent_with_result(artifact_id)
    diagnostics = {
        "artifact_id_prefix": _artifact_log_id(artifact_id),
        "lookup_result": lookup_result,
        "consume_outcome": "hit" if isinstance(artifact, dict) else "miss",
        "request_id": get_request_id() or "unknown",
    }
    app_metrics.record_booking_handoff_consume(
        lookup_result=diagnostics["lookup_result"],
        outcome=diagnostics["consume_outcome"],
    )
    logger.info("booking_post_bridge_consume_outcome", extra=diagnostics)
    return artifact, diagnostics


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

    raw_url = raw.get("url") or raw.get("endpoint") or raw.get("booking_url")
    canonical_url = _canonicalize_handoff_url(raw_url)
    if not canonical_url:
        return None

    method = str(raw.get("method") or "").strip().upper()
    post_data = raw.get("post_data")
    if not method:
        method = "POST" if post_data not in (None, "", {}, []) else "GET"
    if method not in {"GET", "POST"}:
        return None
    # booking_request should never "win" via generic Google search URL.
    if _is_google_search_fallback_url(canonical_url):
        return None
    # Allow tracker URLs only when they carry a concrete POST artifact that is
    # consumed via our one-time bridge endpoint.
    if _is_brittle_google_tracker_url(canonical_url) and not (
        method == "POST" and post_data not in (None, "", {}, [])
    ):
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
            if candidate and not _is_google_search_fallback_url(candidate) and not _is_brittle_google_tracker_url(candidate):
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


def _extract_meta_refresh_url_from_html(html_text: str, *, base_url: str) -> Optional[str]:
    if not isinstance(html_text, str) or not html_text:
        return None
    for tag_match in _META_REFRESH_TAG_RE.finditer(html_text[:BOOKING_REQUEST_RESPONSE_SNIPPET_BYTES]):
        tag = str(tag_match.group(0) or "")
        content_match = re.search(
            r"content\s*=\s*(['\"])(.*?)\1",
            tag,
            re.IGNORECASE | re.DOTALL,
        )
        if content_match:
            content_value = str(content_match.group(2) or "")
        else:
            unquoted_match = re.search(r"content\s*=\s*([^>]+)", tag, re.IGNORECASE)
            content_value = str(unquoted_match.group(1) or "") if unquoted_match else ""
        if not content_value:
            continue
        content_url_match = re.search(r"url\s*=\s*", content_value, re.IGNORECASE)
        if not content_url_match:
            continue
        candidate_tail = str(content_value[content_url_match.end():] or "").strip()
        if not candidate_tail:
            continue
        if candidate_tail[0] in {"'", '"'}:
            quote = candidate_tail[0]
            closing = candidate_tail.find(quote, 1)
            candidate_raw = (
                candidate_tail[1:closing]
                if closing > 0
                else candidate_tail[1:]
            )
        else:
            candidate_raw = candidate_tail.split()[0]
        candidate_raw = html.unescape(str(candidate_raw or "").strip().strip("'\""))
        if not candidate_raw:
            continue
        absolute = urllib.parse.urljoin(base_url, candidate_raw)
        canonical = _canonicalize_handoff_url(absolute)
        if canonical:
            return canonical
    return None


def _prepare_booking_request_post_body(post_data: Any) -> Dict[str, Any]:
    if post_data is None:
        return {}
    if isinstance(post_data, bytes):
        return {"content": post_data}
    if isinstance(post_data, str):
        return {"content": post_data}
    if isinstance(post_data, (dict, list, tuple)):
        return {"data": post_data}
    return {"content": str(post_data)}


def _build_booking_request_headers(headers: Dict[str, Any]) -> Dict[str, str]:
    safe: Dict[str, str] = {}
    for key, value in (headers or {}).items():
        if not isinstance(key, str) or value is None:
            continue
        normalized = key.strip()
        if not normalized:
            continue
        if normalized.lower() in {"authorization", "cookie"}:
            continue
        safe[normalized] = str(value)
    safe.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    return safe


async def _resolve_booking_request_post_to_provider_url(
    *,
    request_url: str,
    post_data: Any,
    request_headers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    headers = _build_booking_request_headers(request_headers or {})
    post_kwargs = _prepare_booking_request_post_body(post_data)
    client = _get_booking_http_client()

    for attempt in range(1, BOOKING_REQUEST_RETRIES + 1):
        attempt_started = time.monotonic()
        try:
            response = await client.post(
                request_url,
                headers=headers,
                timeout=BOOKING_REQUEST_HTTP_TIMEOUT,
                follow_redirects=True,
                **post_kwargs,
            )
            status_code = int(response.status_code)
            content_type = str(response.headers.get("content-type") or "").lower()
            final_url_raw = str(getattr(response, "url", "") or "").strip()
            final_url = _canonicalize_handoff_url(final_url_raw)

            html_snippet = ""
            meta_refresh_url = None
            if "html" in content_type or content_type == "":
                with contextlib.suppress(Exception):
                    html_snippet = str(response.text or "")[:BOOKING_REQUEST_RESPONSE_SNIPPET_BYTES]
                meta_refresh_url = _extract_meta_refresh_url_from_html(
                    html_snippet,
                    base_url=final_url or request_url,
                )

            json_candidate = None
            if "json" in content_type:
                with contextlib.suppress(Exception):
                    json_candidate = _extract_link_from_response_payload(response.json())

            candidates: List[Tuple[str, str]] = []
            if meta_refresh_url:
                candidates.append(("meta_refresh", meta_refresh_url))
            if json_candidate:
                canonical_json = _canonicalize_handoff_url(json_candidate)
                if canonical_json:
                    candidates.append(("json_payload", canonical_json))
            if final_url:
                candidates.append(("final_response_url", final_url))

            for resolver_source, candidate in candidates:
                domain = _domain_for_url(candidate)
                if not _is_usable_handoff_url(candidate):
                    continue
                if domain and _is_google_domain(domain):
                    continue
                return {
                    "status": "resolved",
                    "resolved_url": candidate,
                    "resolver_source": resolver_source,
                    "status_code": status_code,
                    "api_base_url": urllib.parse.urljoin(request_url, "/"),
                    "final_response_url": final_url,
                    "content_type": content_type or None,
                    "attempt": attempt,
                    "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                    "provider_error_classification": "none",
                }

            return {
                "status": "unresolved",
                "reason": "booking_request_post_no_replayable_provider_url",
                "status_code": status_code,
                "api_base_url": urllib.parse.urljoin(request_url, "/"),
                "final_response_url": final_url,
                "content_type": content_type or None,
                "meta_refresh_url": meta_refresh_url,
                "response_excerpt": html_snippet[:350] if html_snippet else None,
                "attempt": attempt,
                "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                "provider_error_classification": _classify_http_status(status_code),
            }
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            if attempt < BOOKING_REQUEST_RETRIES:
                await asyncio.sleep(BOOKING_REQUEST_RETRY_BACKOFF * attempt)
                continue
            return {
                "status": "unresolved",
                "reason": "booking_request_post_network_error",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "api_base_url": urllib.parse.urljoin(request_url, "/"),
                "provider_error_classification": "network_failure",
            }
        except Exception as exc:
            return {
                "status": "unresolved",
                "reason": "booking_request_post_exception",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "api_base_url": urllib.parse.urljoin(request_url, "/"),
                "provider_error_classification": "unknown_failure",
            }

    return {
        "status": "unresolved",
        "reason": "booking_request_post_resolution_exhausted",
        "api_base_url": urllib.parse.urljoin(request_url, "/"),
        "provider_error_classification": "unknown_failure",
    }


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
        if _is_brittle_google_tracker_url(url):
            return None, None
        return url, "booking_request.url"

    resolver = await _resolve_booking_request_post_to_provider_url(
        request_url=url,
        post_data=post_data,
        request_headers=headers,
    )
    if resolver.get("status") == "resolved":
        resolved_url = _canonicalize_handoff_url(str(resolver.get("resolved_url") or ""))
        if resolved_url and _is_usable_handoff_url(resolved_url):
            return resolved_url, "booking_request.post_resolved_provider_url"

    # Keep one-time bridge artifact available for diagnostics only.
    register_post_handoff_artifact(
        url=url,
        post_data=post_data,
        headers=headers,
    )
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
    Represents a single local booking-follow-up record.

    Status lifecycle:
        HELD  →  CANCELLED  (user explicitly cancelled before expiry)
        HELD  →  EXPIRED    (hold_minutes elapsed without cancellation)

    Important product semantics:
    - Real booking/payment completion happens on external airline/OTA/provider sites.
    - This local record must not imply provider-side booking completion.
    """
    __tablename__ = "bookings"

    id           = Column(Integer, primary_key=True, index=True)
    status       = Column(String,  nullable=False)           # HELD | CANCELLED | EXPIRED
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
        together = opt.get("together")
        together_payload = together if isinstance(together, dict) else {}
        provider = (
            together_payload.get("book_with")
            or together_payload.get("name")
            or opt.get("book_with")
            or opt.get("name")
            or opt.get("provider")
            or "unknown"
        )
        price = together_payload.get("price") if together_payload.get("price") is not None else opt.get("price")
        link = (
            together_payload.get("link")
            or together_payload.get("url")
            or together_payload.get("booking_url")
            or together_payload.get("booking_link")
            or together_payload.get("deeplink")
            or together_payload.get("redirect_link")
            or opt.get("link")
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
        marketed_as = together_payload.get("marketed_as")
        if isinstance(marketed_as, (str, int, float)):
            marketed_as = [str(marketed_as)]
        elif isinstance(marketed_as, list):
            marketed_as = [str(item) for item in marketed_as if item is not None][:8]
        else:
            marketed_as = None
        local_prices = together_payload.get("local_prices")
        baggage_prices = together_payload.get("baggage_prices")
        booking_request = together_payload.get("booking_request")
        if not isinstance(booking_request, dict):
            booking_request = opt.get("booking_request")
        options.append({
            "provider": provider,
            "price": price_float,
            "link": canonical.strip(),
            "price_available": price_float is not None,
            "book_with": provider,
            "option_title": together_payload.get("option_title") or opt.get("option_title"),
            "airline": together_payload.get("airline") or opt.get("airline"),
            "airline_logos": together_payload.get("airline_logos") or opt.get("airline_logos"),
            "marketed_as": marketed_as,
            "local_prices": local_prices if isinstance(local_prices, (list, dict, str, int, float)) else None,
            "baggage_prices": baggage_prices if isinstance(baggage_prices, (list, dict, str, int, float)) else None,
            "separate_tickets": bool(opt.get("separate_tickets")),
            "booking_request": booking_request if isinstance(booking_request, dict) else None,
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
    )
    canonical_url = _canonicalize_handoff_url(url)

    post_data = booking_request.get("post_data")
    method = str(booking_request.get("method") or "").strip().upper()
    if not method:
        method = "POST" if post_data not in (None, "", {}, []) else "GET"
    if method not in {"GET", "POST"}:
        return None
    if not canonical_url or _is_google_search_fallback_url(canonical_url):
        return None
    if _is_brittle_google_tracker_url(canonical_url) and not (
        method == "POST" and post_data not in (None, "", {}, [])
    ):
        return None

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


def _summarize_booking_option(option: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(option, dict):
        return None
    summary: Dict[str, Any] = {
        "book_with": option.get("book_with") or option.get("provider"),
        "price": option.get("price"),
        "option_title": option.get("option_title"),
        "link_domain": _domain_for_url(option.get("link")),
        "airline": option.get("airline"),
        "airline_logos": option.get("airline_logos"),
        "marketed_as": option.get("marketed_as"),
        "separate_tickets": bool(option.get("separate_tickets")),
        "local_prices": option.get("local_prices"),
        "baggage_prices": option.get("baggage_prices"),
    }
    cleaned = {k: v for k, v in summary.items() if v not in (None, "", [], {})}
    return cleaned or None


def _seller_label_from_url(value: Optional[str]) -> Optional[str]:
    domain = _domain_for_url(value)
    if not domain:
        return None
    if "goindigo" in domain:
        return "IndiGo"
    if "airindia" in domain:
        return "Air India"
    token = domain.split(".")[0]
    if token == "www":
        parts = domain.split(".")
        token = parts[1] if len(parts) > 1 else token
    cleaned = token.replace("-", " ").strip()
    if not cleaned:
        return None
    return cleaned.title()


def _coalesce_booking_option_summary(
    summary: Optional[Dict[str, Any]],
    *,
    resolved_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    domain = _domain_for_url(resolved_url)
    if not isinstance(summary, dict):
        summary = {}
    merged = dict(summary)
    if domain and not merged.get("link_domain"):
        merged["link_domain"] = domain
    if not merged.get("book_with"):
        seller = _seller_label_from_url(resolved_url)
        if seller:
            merged["book_with"] = seller
    cleaned = {k: v for k, v in merged.items() if v not in (None, "", [], {})}
    return cleaned or None


def _summarize_selected_flights(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    selected = payload.get("selected_flights")
    if not isinstance(selected, list) or not selected:
        return None
    first = selected[0]
    if not isinstance(first, dict):
        return None
    flights = first.get("flights")
    if not isinstance(flights, list) or not flights:
        return None
    legs: List[Dict[str, Any]] = []
    for leg in flights[:6]:
        if not isinstance(leg, dict):
            continue
        dep = leg.get("departure_airport") if isinstance(leg.get("departure_airport"), dict) else {}
        arr = leg.get("arrival_airport") if isinstance(leg.get("arrival_airport"), dict) else {}
        legs.append(
            {
                "airline": leg.get("airline"),
                "flight_number": leg.get("flight_number"),
                "operating_flight_number": leg.get("operating_flight_number"),
                "travel_class": leg.get("travel_class"),
                "departure_airport_id": dep.get("id"),
                "departure_time": dep.get("time"),
                "arrival_airport_id": arr.get("id"),
                "arrival_time": arr.get("time"),
            }
        )
    if not legs:
        return None
    return {"legs": legs}


async def _fetch_booking_options_payload(
    *,
    booking_token: str,
    departure_id: Optional[str] = None,
    arrival_id: Optional[str] = None,
    outbound_date: Optional[str] = None,
    return_date: Optional[str] = None,
    include_airlines: Optional[str] = None,
    deep_search: Optional[bool] = None,
    travel_class: Optional[str] = None,
    adults: Optional[int] = None,
    currency: Optional[str] = None,
    hl: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch raw google_flights_booking payload from SerpAPI.
    Includes original route/date search context when available.
    """

    route_type = "round_trip" if return_date else "one_way"
    max_attempts = max(1, min(BOOKING_OPTIONS_RETRIES, BOOKING_OPTIONS_ATTEMPTS_BUDGET))
    relaxed_shape_enabled = bool(
        departure_id
        or arrival_id
        or outbound_date
        or include_airlines
        or deep_search is not None
        or travel_class
        or adults
    )
    if relaxed_shape_enabled:
        max_attempts = max(2, max_attempts)
    use_relaxed_shape = False

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
                    "hl": (hl or "en"),
                    "gl": "in",
                    "currency": (currency or "INR"),
                }
                # We intentionally never mix booking_token with departure_token.
                params.pop("departure_token", None)
                # Proven runtime requirement: token-only requests can fail.
                # Preserve original search context when available.
                if not use_relaxed_shape:
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
                    if include_airlines:
                        params["include_airlines"] = include_airlines
                    if deep_search is not None:
                        params["deep_search"] = "true" if bool(deep_search) else "false"
                    if travel_class:
                        params["travel_class"] = str(travel_class)
                    if adults and int(adults) > 0:
                        params["adults"] = str(int(adults))

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
                "has_include_airlines": bool(params.get("include_airlines")),
                "has_deep_search": bool(params.get("deep_search")),
                "travel_class": str(params.get("travel_class") or ""),
                "adults": str(params.get("adults") or ""),
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
                "shape_mode": "token_only_relaxed" if use_relaxed_shape else "route_shaped",
            }

            if resp.status_code != 200:
                transient = resp.status_code in {408, 429} or resp.status_code >= 500
                if transient and attempt < max_attempts:
                    await asyncio.sleep(BOOKING_OPTIONS_RETRY_BACKOFF * attempt)
                    continue
                if transient and attempt >= max_attempts:
                    app_metrics.record_retry_budget_exhausted("booking_options_fetch")
                if resp.status_code in {401, 403}:
                    with contextlib.suppress(Exception):
                        await api_key_manager.mark_exhausted(
                            "serpapi",
                            idx,
                            reason=f"booking_options_http_{resp.status_code}",
                        )
                if (not use_relaxed_shape) and relaxed_shape_enabled and resp.status_code in {400, 404, 422}:
                    use_relaxed_shape = True
                    logger.warning(
                        "booking_options shaped request degraded; retrying token-only request",
                        extra={
                            **request_shape,
                            "http_status": resp.status_code,
                            "fetch_duration_ms": int((time.monotonic() - attempt_started) * 1000),
                            "result_bucket": "http_error_recoverable",
                            **response_flags,
                        },
                    )
                    continue
                logger.warning(
                    "booking_options unavailable after request",
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
                logger.warning(
                    "booking_options parse error after request",
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
                if transient_error and attempt >= max_attempts:
                    app_metrics.record_retry_budget_exhausted("booking_options_fetch")
                if any(tok in error_text for tok in ("unauthorized", "invalid api key", "invalid key", "access denied")):
                    with contextlib.suppress(Exception):
                        await api_key_manager.mark_exhausted(
                            "serpapi",
                            idx,
                            reason="booking_options_provider_unauthorized",
                        )
                if (not use_relaxed_shape) and relaxed_shape_enabled:
                    use_relaxed_shape = True
                    logger.warning(
                        "booking_options shaped request provider error; retrying token-only request",
                        extra={
                            **request_shape,
                            "provider_error": str(data.get("error") or "")[:180],
                            "fetch_duration_ms": int((time.monotonic() - attempt_started) * 1000),
                            "result_bucket": "provider_error_recoverable",
                            **response_flags,
                        },
                    )
                    continue
                logger.warning(
                    "booking_options provider error after request",
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
            logger.debug(
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
            app_metrics.record_retry_budget_exhausted("booking_options_fetch")
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
                    log_severity = "debug"
                    log_fn = logger.debug
                else:
                    log_severity = "warning"
                    log_fn = logger.warning
            elif exception_bucket in {"no_active_key", "provider_rate_limited", "provider_auth"}:
                log_severity = "debug" if candidate_probe_context else "warning"
                log_fn = logger.debug if candidate_probe_context else logger.warning
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
    include_airlines: Optional[str] = None,
    deep_search: Optional[bool] = None,
    travel_class: Optional[str] = None,
    adults: Optional[int] = None,
    currency: Optional[str] = None,
    hl: Optional[str] = None,
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
            include_airlines=include_airlines,
            deep_search=deep_search,
            travel_class=travel_class,
            adults=adults,
            currency=currency,
            hl=hl,
        )
    except BookingOptionsFetchError:
        return None
    if not data:
        return None

    options = _extract_booking_options_from_payload(data)
    if not options:
        logger.warning("No valid booking options found after provider fetch")
        return None

    logger.debug("Fetched %d booking options from SerpAPI", len(options))
    return options


async def best_booking_option(
    booking_token: str,
    *,
    departure_id: Optional[str] = None,
    arrival_id: Optional[str] = None,
    outbound_date: Optional[str] = None,
    return_date: Optional[str] = None,
    include_airlines: Optional[str] = None,
    deep_search: Optional[bool] = None,
    travel_class: Optional[str] = None,
    adults: Optional[int] = None,
    currency: Optional[str] = None,
    hl: Optional[str] = None,
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
        include_airlines=include_airlines,
        deep_search=deep_search,
        travel_class=travel_class,
        adults=adults,
        currency=currency,
        hl=hl,
    )
    if not options:
        return None
    priced = [opt for opt in options if opt.get("price") is not None]
    if priced:
        best = min(priced, key=lambda x: x["price"])
        logger.debug("Best booking option selected with numeric price")
    else:
        best = options[0]
        logger.debug("Best booking option selected without numeric price; using first valid booking link")
    return best


async def resolve_booking_token(
    booking_token: str,
    *,
    departure_id: Optional[str] = None,
    arrival_id: Optional[str] = None,
    outbound_date: Optional[str] = None,
    return_date: Optional[str] = None,
    include_airlines: Optional[str] = None,
    deep_search: Optional[bool] = None,
    travel_class: Optional[str] = None,
    adults: Optional[int] = None,
    currency: Optional[str] = None,
    hl: Optional[str] = None,
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
        include_airlines=include_airlines,
        deep_search=deep_search,
        travel_class=travel_class,
        adults=adults,
        currency=currency,
        hl=hl,
    )
    return best["link"] if best else None


async def resolve_booking_token_with_details(
    booking_token: str,
    *,
    departure_id: Optional[str] = None,
    arrival_id: Optional[str] = None,
    outbound_date: Optional[str] = None,
    return_date: Optional[str] = None,
    include_airlines: Optional[str] = None,
    deep_search: Optional[bool] = None,
    travel_class: Optional[str] = None,
    adults: Optional[int] = None,
    currency: Optional[str] = None,
    hl: Optional[str] = None,
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
        include_airlines=include_airlines,
        deep_search=deep_search,
        travel_class=travel_class,
        adults=adults,
        currency=currency,
        hl=hl,
    )
    with _handoff_cache_lock:
        cached = _booking_resolution_cache.get(cache_key)
    if isinstance(cached, dict):
        cached_kind = str(cached.get("kind") or "")
        if cached_kind == "direct_booking":
            cached_url = str(cached.get("url") or "").strip()
            artifact_inspection = cached.get("artifact_inspection")
            booking_option_summary = cached.get("booking_option_summary")
            selected_flights_summary = cached.get("selected_flights_summary")
            if _is_usable_handoff_url(cached_url):
                logger.debug(
                    "booking_token_resolution cache hit",
                    extra={
                        "cache_key": cache_key,
                        "cache_kind": cached_kind,
                        "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                    },
                )
                payload = {
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
                if isinstance(artifact_inspection, dict):
                    payload["artifact_inspection"] = artifact_inspection
                if isinstance(booking_option_summary, dict):
                    payload["booking_option_summary"] = booking_option_summary
                if isinstance(selected_flights_summary, dict):
                    payload["selected_flights_summary"] = selected_flights_summary
                return payload
        if cached_kind in {"booking_request_post", "booking_request_post_resolved"}:
            request_url = str(cached.get("url") or "").strip()
            post_data = cached.get("post_data")
            request_headers = cached.get("headers") or {}
            artifact_inspection = cached.get("artifact_inspection")
            booking_option_summary = cached.get("booking_option_summary")
            selected_flights_summary = cached.get("selected_flights_summary")
            cached_resolved_provider_url = _canonicalize_handoff_url(
                str(cached.get("resolved_provider_url") or "")
            )
            if cached_resolved_provider_url and _is_usable_handoff_url(cached_resolved_provider_url):
                resolved_summary = _coalesce_booking_option_summary(
                    booking_option_summary if isinstance(booking_option_summary, dict) else None,
                    resolved_url=cached_resolved_provider_url,
                )
                payload = {
                    "url": cached_resolved_provider_url,
                    "source": "booking_token",
                    "reason": "resolved_booking_request_post_cache",
                    "status": "ok",
                    "provider": "serpapi",
                    "handoff_mode": "provider_or_shareable",
                    "landing_guarantee": "provider_managed",
                    "artifact_field": "booking_request.post_data_resolved_provider_url",
                    "is_exact_handoff": False,
                    "is_search_fallback": False,
                    "is_provider_managed": True,
                    "is_booking_quality_exit": True,
                    "booking_exit_quality": "booking_ready",
                    "cache_hit": True,
                    "booking_request_resolution": cached.get("booking_request_resolution"),
                }
                if isinstance(artifact_inspection, dict):
                    payload["artifact_inspection"] = artifact_inspection
                if isinstance(resolved_summary, dict):
                    payload["booking_option_summary"] = resolved_summary
                if isinstance(selected_flights_summary, dict):
                    payload["selected_flights_summary"] = selected_flights_summary
                return payload

            resolver = await _resolve_booking_request_post_to_provider_url(
                request_url=request_url,
                post_data=post_data,
                request_headers=request_headers,
            )
            resolved_provider_url = _canonicalize_handoff_url(str(resolver.get("resolved_url") or ""))
            if resolver.get("status") == "resolved" and resolved_provider_url and _is_usable_handoff_url(resolved_provider_url):
                _cache_booking_resolution(
                    cache_key,
                    {
                        "kind": "booking_request_post_resolved",
                        "url": request_url,
                        "post_data": post_data,
                        "headers": request_headers,
                        "resolved_provider_url": resolved_provider_url,
                        "booking_request_resolution": resolver,
                        "artifact_inspection": artifact_inspection,
                        "booking_option_summary": booking_option_summary,
                        "selected_flights_summary": selected_flights_summary,
                    },
                )
                resolved_summary = _coalesce_booking_option_summary(
                    booking_option_summary if isinstance(booking_option_summary, dict) else None,
                    resolved_url=resolved_provider_url,
                )
                payload = {
                    "url": resolved_provider_url,
                    "source": "booking_token",
                    "reason": "resolved_booking_request_post_cache",
                    "status": "ok",
                    "provider": "serpapi",
                    "handoff_mode": "provider_or_shareable",
                    "landing_guarantee": "provider_managed",
                    "artifact_field": "booking_request.post_data_resolved_provider_url",
                    "is_exact_handoff": False,
                    "is_search_fallback": False,
                    "is_provider_managed": True,
                    "is_booking_quality_exit": True,
                    "booking_exit_quality": "booking_ready",
                    "cache_hit": True,
                    "booking_request_resolution": resolver,
                }
                if isinstance(artifact_inspection, dict):
                    payload["artifact_inspection"] = artifact_inspection
                if isinstance(resolved_summary, dict):
                    payload["booking_option_summary"] = resolved_summary
                if isinstance(selected_flights_summary, dict):
                    payload["selected_flights_summary"] = selected_flights_summary
                return payload

            bridge_url = register_post_handoff_artifact(
                url=request_url,
                post_data=post_data,
                headers=request_headers,
            )
            if bridge_url:
                target_domain, target_path = _bridge_target_domain_path(request_url)
                logger.debug(
                    "booking_post_bridge_target_resolved",
                    extra={
                        "target_domain": target_domain,
                        "target_path": target_path,
                        "cache_key": cache_key,
                        "cache_kind": cached_kind,
                        "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                    },
                )
                if _is_brittle_google_tracker_url(request_url):
                    logger.info(
                        "booking_post_bridge_booking_ready_downgraded",
                        extra={
                            "target_domain": target_domain,
                            "target_path": target_path,
                            "downgrade_reason": "google_tracker_post_bridge_unverified",
                            "browser_landing_verdict": "unknown_unverified",
                            "cache_hit": True,
                        },
                    )
                    downgraded = _build_unverified_google_tracker_post_bridge_payload(
                        bridge_url=bridge_url,
                        reason="resolved_booking_request_post_unverified_google_tracker_cache",
                        request_url=request_url,
                        cache_hit=True,
                        artifact_inspection=artifact_inspection if isinstance(artifact_inspection, dict) else None,
                    )
                    downgraded["booking_request_resolution"] = resolver
                    if isinstance(booking_option_summary, dict):
                        downgraded["booking_option_summary"] = booking_option_summary
                    if isinstance(selected_flights_summary, dict):
                        downgraded["selected_flights_summary"] = selected_flights_summary
                    return downgraded
                payload = {
                    "url": None,
                    "diagnostic_handoff_url": bridge_url,
                    "source": "booking_token",
                    "reason": "booking_request_post_resolution_failed_cache",
                    "status": "unavailable",
                    "provider": "serpapi",
                    "handoff_mode": "unavailable",
                    "landing_guarantee": "none",
                    "artifact_field": "booking_request.post_followup_bridge",
                    "requires_browser_post": True,
                    "is_exact_handoff": False,
                    "is_search_fallback": False,
                    "is_provider_managed": False,
                    "is_booking_quality_exit": False,
                    "booking_exit_quality": "unavailable",
                    "cache_hit": True,
                    "booking_request_resolution": resolver,
                }
                if isinstance(artifact_inspection, dict):
                    payload["artifact_inspection"] = artifact_inspection
                    payload["proof_only_google_artifacts"] = bool(
                        artifact_inspection.get("only_google_click_or_google_domains")
                    ) and not bool(artifact_inspection.get("has_replayable_partner_option"))
                if isinstance(booking_option_summary, dict):
                    payload["booking_option_summary"] = booking_option_summary
                if isinstance(selected_flights_summary, dict):
                    payload["selected_flights_summary"] = selected_flights_summary
                return payload
        if cached_kind == "booking_request_get":
            cached_url = str(cached.get("url") or "").strip()
            artifact_inspection = cached.get("artifact_inspection")
            booking_option_summary = cached.get("booking_option_summary")
            selected_flights_summary = cached.get("selected_flights_summary")
            if _is_usable_handoff_url(cached_url):
                logger.debug(
                    "booking_token_resolution cache hit",
                    extra={
                        "cache_key": cache_key,
                        "cache_kind": cached_kind,
                        "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                    },
                )
                payload = {
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
                if isinstance(artifact_inspection, dict):
                    payload["artifact_inspection"] = artifact_inspection
                if isinstance(booking_option_summary, dict):
                    payload["booking_option_summary"] = booking_option_summary
                if isinstance(selected_flights_summary, dict):
                    payload["selected_flights_summary"] = selected_flights_summary
                return payload

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
                include_airlines=include_airlines,
                deep_search=deep_search,
                travel_class=travel_class,
                adults=adults,
                currency=currency,
                hl=hl,
            ),
            timeout=resolve_timeout,
        )
    except asyncio.TimeoutError:
        logger.debug(
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
        logger.debug(
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
        logger.debug(
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
    booking_request = _extract_booking_request_artifact_from_payload(payload or {})
    selected_flights_summary = _summarize_selected_flights(payload or {})
    artifact_inspection = _summarize_booking_artifact_graph(
        payload=payload or {},
        options=options,
        booking_request=booking_request,
    )
    logger.debug(
        "booking_artifact_graph_inspected",
        extra={
            "booking_options_count": artifact_inspection.get("booking_options_count"),
            "booking_option_domains": artifact_inspection.get("booking_option_domains"),
            "replayable_partner_option_domains": artifact_inspection.get("replayable_partner_option_domains"),
            "booking_request_domain": artifact_inspection.get("booking_request_domain"),
            "booking_request_method": artifact_inspection.get("booking_request_method"),
            "booking_request_has_post_data": artifact_inspection.get("booking_request_has_post_data"),
            "only_google_click_or_google_domains": artifact_inspection.get("only_google_click_or_google_domains"),
            "has_replayable_partner_option": artifact_inspection.get("has_replayable_partner_option"),
        },
    )

    invalid_direct_link_encountered = False
    if options:
        invalid_direct_link_encountered = True

    selected_partner_option = _select_best_replayable_partner_option(options)
    summary_option = selected_partner_option
    if summary_option is None:
        summary_option = next(
            (
                option
                for option in options
                if isinstance(option, dict) and isinstance(option.get("booking_request"), dict)
            ),
            None,
        )
    if summary_option is None and options:
        summary_option = sorted(options, key=_option_sort_key)[0]
    booking_option_summary = _summarize_booking_option(summary_option)
    if selected_partner_option and selected_partner_option.get("link"):
        link = str(selected_partner_option.get("link")).strip()
        _cache_booking_resolution(
            cache_key,
            {
                "kind": "direct_booking",
                "url": link,
                "artifact_inspection": artifact_inspection,
                "booking_option_summary": booking_option_summary,
                "selected_flights_summary": selected_flights_summary,
            },
        )
        logger.debug(
            "booking_token_resolution completed",
            extra={
                "result_bucket": "direct_booking",
                "selected_domain": _domain_for_url(link),
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
            "artifact_inspection": artifact_inspection,
            "booking_option_summary": booking_option_summary,
            "selected_flights_summary": selected_flights_summary,
        }

    if options:
        logger.debug(
            "booking_token_no_replayable_partner_option",
            extra={
                "booking_option_domains": artifact_inspection.get("booking_option_domains"),
                "booking_option_google_tracker_count": artifact_inspection.get("booking_option_google_tracker_count"),
                "replayable_partner_option_domains": artifact_inspection.get("replayable_partner_option_domains"),
                "only_google_click_or_google_domains": artifact_inspection.get("only_google_click_or_google_domains"),
            },
        )

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
                    "artifact_inspection": artifact_inspection,
                    "booking_option_summary": booking_option_summary,
                    "selected_flights_summary": selected_flights_summary,
                },
            )
            resolver = await _resolve_booking_request_post_to_provider_url(
                request_url=request_url,
                post_data=post_data,
                request_headers=request_headers,
            )
            resolved_provider_url = _canonicalize_handoff_url(str(resolver.get("resolved_url") or ""))
            if resolver.get("status") == "resolved" and resolved_provider_url and _is_usable_handoff_url(resolved_provider_url):
                _cache_booking_resolution(
                    cache_key,
                    {
                        "kind": "booking_request_post_resolved",
                        "url": request_url,
                        "post_data": post_data,
                        "headers": request_headers,
                        "resolved_provider_url": resolved_provider_url,
                        "booking_request_resolution": resolver,
                        "artifact_inspection": artifact_inspection,
                        "booking_option_summary": booking_option_summary,
                        "selected_flights_summary": selected_flights_summary,
                    },
                )
                resolved_summary = _coalesce_booking_option_summary(
                    booking_option_summary if isinstance(booking_option_summary, dict) else None,
                    resolved_url=resolved_provider_url,
                )
                return {
                    "url": resolved_provider_url,
                    "source": "booking_token",
                    "reason": "resolved_booking_request_post",
                    "status": "ok",
                    "provider": "serpapi",
                    "handoff_mode": "provider_or_shareable",
                    "landing_guarantee": "provider_managed",
                    "artifact_field": "booking_request.post_data_resolved_provider_url",
                    "is_exact_handoff": False,
                    "is_search_fallback": False,
                    "is_provider_managed": True,
                    "is_booking_quality_exit": True,
                    "booking_exit_quality": "booking_ready",
                    "cache_hit": False,
                    "artifact_inspection": artifact_inspection,
                    "booking_request_resolution": resolver,
                    "booking_option_summary": resolved_summary,
                    "selected_flights_summary": selected_flights_summary,
                }

            bridge_url = register_post_handoff_artifact(
                url=request_url,
                post_data=post_data,
                headers=request_headers,
            )
            if bridge_url:
                target_domain, target_path = _bridge_target_domain_path(request_url)
                logger.debug(
                    "booking_post_bridge_target_resolved",
                    extra={
                        "target_domain": target_domain,
                        "target_path": target_path,
                        "result_bucket": "booking_request_post",
                        "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
                    },
                )
                if _is_brittle_google_tracker_url(request_url):
                    logger.info(
                        "booking_post_bridge_booking_ready_downgraded",
                        extra={
                            "target_domain": target_domain,
                            "target_path": target_path,
                            "downgrade_reason": "google_tracker_post_bridge_unverified",
                            "browser_landing_verdict": "unknown_unverified",
                            "cache_hit": False,
                            "only_google_click_or_google_domains": artifact_inspection.get(
                                "only_google_click_or_google_domains"
                            ),
                            "all_non_google_domains": artifact_inspection.get("all_non_google_domains"),
                        },
                    )
                    downgraded = _build_unverified_google_tracker_post_bridge_payload(
                        bridge_url=bridge_url,
                        reason="resolved_booking_request_post_unverified_google_tracker",
                        request_url=request_url,
                        cache_hit=False,
                        artifact_inspection=artifact_inspection,
                    )
                    downgraded["booking_request_resolution"] = resolver
                    downgraded["booking_option_summary"] = booking_option_summary
                    downgraded["selected_flights_summary"] = selected_flights_summary
                    return downgraded
                return {
                    "url": None,
                    "diagnostic_handoff_url": bridge_url,
                    "source": "booking_token",
                    "reason": "booking_request_post_resolution_failed",
                    "status": "unavailable",
                    "provider": "serpapi",
                    "handoff_mode": "unavailable",
                    "landing_guarantee": "none",
                    "artifact_field": "booking_request.post_followup_bridge",
                    "requires_browser_post": True,
                    "is_exact_handoff": False,
                    "is_search_fallback": False,
                    "is_provider_managed": False,
                    "is_booking_quality_exit": False,
                    "booking_exit_quality": "unavailable",
                    "cache_hit": False,
                    "artifact_inspection": artifact_inspection,
                    "booking_request_resolution": resolver,
                    "proof_only_google_artifacts": bool(
                        artifact_inspection.get("only_google_click_or_google_domains")
                    ) and not bool(artifact_inspection.get("has_replayable_partner_option")),
                    "booking_option_summary": booking_option_summary,
                    "selected_flights_summary": selected_flights_summary,
                }
        if _is_usable_handoff_url(request_url):
            _cache_booking_resolution(
                cache_key,
                {
                    "kind": "booking_request_get",
                    "url": request_url,
                    "artifact_inspection": artifact_inspection,
                    "booking_option_summary": booking_option_summary,
                    "selected_flights_summary": selected_flights_summary,
                },
            )
            logger.debug(
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
                "artifact_inspection": artifact_inspection,
                "booking_option_summary": booking_option_summary,
                "selected_flights_summary": selected_flights_summary,
            }

    logger.debug(
        "booking_token_resolution completed",
        extra={
            "result_bucket": "no_usable_artifact",
            "resolution_duration_ms": int((time.monotonic() - resolution_started) * 1000),
        },
    )
    return {
        "url": None,
        "source": "booking_token",
        "reason": "booking_token_invalid_url" if invalid_direct_link_encountered else "booking_token_unresolved",
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
        "artifact_inspection": artifact_inspection,
        "proof_only_google_artifacts": bool(
            artifact_inspection.get("only_google_click_or_google_domains")
        ) and not bool(artifact_inspection.get("has_replayable_partner_option")),
        "booking_option_summary": booking_option_summary,
        "selected_flights_summary": selected_flights_summary,
    }


def _extract_airline_code_hint(flight: Dict[str, Any]) -> Optional[str]:
    if not isinstance(flight, dict):
        return None
    candidates: List[str] = []
    flight_no = str(flight.get("flight_no") or "").strip().upper()
    if flight_no:
        candidates.append(flight_no)
    marketed = flight.get("marketed_as")
    if isinstance(marketed, list):
        candidates.extend([str(item).strip().upper() for item in marketed if item])
    legs = flight.get("legs")
    if isinstance(legs, list):
        for leg in legs[:2]:
            if isinstance(leg, dict):
                leg_no = str(leg.get("flight_number") or "").strip().upper()
                if leg_no:
                    candidates.append(leg_no)
    for candidate in candidates:
        match = re.search(r"\b([A-Z0-9]{2})\s*\d", candidate)
        if match:
            return match.group(1)
    return None


def _build_serpapi_booking_resolution_hints(flight: Dict[str, Any]) -> Dict[str, Any]:
    hints: Dict[str, Any] = {
        "currency": "INR",
        "hl": "en",
        "adults": 1,
    }
    include_airlines = _extract_airline_code_hint(flight)
    if include_airlines:
        hints["include_airlines"] = include_airlines
    stops = flight.get("stops")
    try:
        stop_count = int(stops) if stops is not None else 0
    except Exception:
        stop_count = 0
    if stop_count > 0:
        # For connecting itineraries widen seller graph lookup on booking options.
        hints["deep_search"] = True
    return hints


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
    Resolve booking handoff through the booking-token provider-resolution path only.
    No Google fallback/search-assist link is emitted.
    """
    def _detail_payload(
        *,
        status: str,
        reason: str,
        source: str,
        url: Optional[str] = None,
        provider: Optional[str] = None,
        cache_hit: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": status,
            "reason": reason,
            "source": source,
            "url": url if status == "booking_ready" else None,
            "booking_exit_quality": "booking_ready" if status == "booking_ready" else "unavailable",
        }
        if provider:
            payload["provider"] = provider
        if cache_hit is not None:
            payload["cache_hit"] = bool(cache_hit)
        return payload

    booking_token = flight.get("booking_token")
    if not booking_token:
        unavailable = _detail_payload(
            status="unavailable",
            reason="booking_token_missing",
            source="unavailable",
            provider="serpapi",
        )
        return unavailable if return_details else None

    serpapi_hints = _build_serpapi_booking_resolution_hints(flight)
    try:
        token_result = await resolve_booking_token_with_details(
            booking_token,
            departure_id=origin,
            arrival_id=destination,
            outbound_date=depart_date,
            return_date=return_date,
            include_airlines=serpapi_hints.get("include_airlines"),
            deep_search=serpapi_hints.get("deep_search"),
            adults=serpapi_hints.get("adults"),
            currency=serpapi_hints.get("currency"),
            hl=serpapi_hints.get("hl"),
        )
    except TypeError:
        # Backward-compatible tests may monkeypatch old signatures.
        token_result = await resolve_booking_token_with_details(booking_token)
    except Exception:
        token_result = {
            "url": None,
            "reason": "booking_handoff_exception",
            "source": "booking_token",
            "provider": "serpapi",
        }

    if not isinstance(token_result, dict):
        token_result = {
            "url": None,
            "reason": "booking_token_unresolved",
            "source": "booking_token",
            "provider": "serpapi",
        }

    resolved_candidate = _canonicalize_handoff_url(token_result.get("url"))
    resolved_is_booking_ready = bool(resolved_candidate) and str(
        token_result.get("booking_exit_quality") or ""
    ).strip().lower() == "booking_ready"

    if resolved_is_booking_ready:
        ready = _detail_payload(
            status="booking_ready",
            reason=str(token_result.get("reason") or "resolved_booking_token"),
            source=str(token_result.get("source") or "booking_token"),
            url=resolved_candidate,
            provider=str(token_result.get("provider") or "serpapi"),
            cache_hit=token_result.get("cache_hit"),
        )
        return ready if return_details else str(ready["url"])

    unavailable = _detail_payload(
        status="unavailable",
        reason=str(token_result.get("reason") or "booking_token_unresolved"),
        source=str(token_result.get("source") or "booking_token"),
        provider=str(token_result.get("provider") or "serpapi"),
        cache_hit=token_result.get("cache_hit"),
    )
    return unavailable if return_details else None


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
        checkout_ready – true only when provider checkout URL exists
        checkout_status – booking_ready | provider_handoff_unavailable
        hold_outcome  – held_with_checkout | held_local_only
        booking_handoff – compact handoff metadata for truthful status display
    """
    def _is_booking_ready_meta(meta: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(meta, dict):
            return False
        return str(meta.get("booking_exit_quality") or "").strip().lower() == "booking_ready"

    def _normalized_checkout_status(
        *,
        handoff_url_value: Optional[str],
        handoff_meta_value: Optional[Dict[str, Any]],
    ) -> str:
        if _is_usable_handoff_url(handoff_url_value):
            return "booking_ready"
        status_text = str((handoff_meta_value or {}).get("status") or "").strip().lower()
        if status_text == "booking_ready":
            return "booking_ready"
        return "provider_handoff_unavailable"

    selected_meta = flight.get("booking_handoff") if isinstance(flight.get("booking_handoff"), dict) else None
    pre_resolved_url_candidates = [
        flight.get("handoff_url"),
        selected_meta.get("url") if isinstance(selected_meta, dict) else None,
        selected_meta.get("handoff_url") if isinstance(selected_meta, dict) else None,
        selected_meta.get("resolved_url") if isinstance(selected_meta, dict) else None,
        selected_meta.get("provider_url") if isinstance(selected_meta, dict) else None,
    ]
    pre_resolved_handoff_url = next(
        (
            _canonicalize_handoff_url(str(candidate or "").strip())
            for candidate in pre_resolved_url_candidates
            if _is_usable_handoff_url(_canonicalize_handoff_url(str(candidate or "").strip()))
        ),
        None,
    )

    handoff_source = "resolved_during_hold"
    handoff_meta: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "booking_handoff_not_ready",
        "source": "booking_token",
        "url": None,
        "booking_exit_quality": "unavailable",
    }
    if pre_resolved_handoff_url and _is_booking_ready_meta(selected_meta):
        handoff_url = pre_resolved_handoff_url
        handoff_source = "selected_handoff_reuse"
        handoff_meta = {
            "status": "booking_ready",
            "reason": "selected_handoff_reuse",
            "source": str(selected_meta.get("source") or "booking_token") if isinstance(selected_meta, dict) else "booking_token",
            "url": pre_resolved_handoff_url,
            "booking_exit_quality": "booking_ready",
        }
        if isinstance(selected_meta, dict):
            provider = selected_meta.get("provider")
            if provider:
                handoff_meta["provider"] = provider
            if isinstance(selected_meta.get("cache_hit"), bool):
                handoff_meta["cache_hit"] = selected_meta.get("cache_hit")
    else:
        # Resolve only when no reusable booking-ready handoff is already attached to the selected flight.
        handoff_result = await build_booking_handoff_url(
            flight=flight,
            origin=origin,
            destination=destination,
            depart_date=depart_date,
            return_date=return_date,
            passengers=passengers,
            return_details=True,
        )
        if isinstance(handoff_result, dict):
            handoff_meta = {
                "status": str(handoff_result.get("status") or "unavailable"),
                "reason": str(handoff_result.get("reason") or "booking_token_unresolved"),
                "source": str(handoff_result.get("source") or "booking_token"),
                "url": _canonicalize_handoff_url(handoff_result.get("url")),
                "booking_exit_quality": str(handoff_result.get("booking_exit_quality") or "unavailable"),
            }
            provider = handoff_result.get("provider")
            if provider:
                handoff_meta["provider"] = provider
            if isinstance(handoff_result.get("cache_hit"), bool):
                handoff_meta["cache_hit"] = handoff_result.get("cache_hit")
            handoff_url = handoff_meta["url"] if handoff_meta.get("status") == "booking_ready" else None
        else:
            handoff_url = _canonicalize_handoff_url(handoff_result)
            handoff_meta = {
                "status": "booking_ready" if handoff_url else "unavailable",
                "reason": "resolved_booking_token" if handoff_url else "booking_token_unresolved",
                "source": "booking_token",
                "url": handoff_url,
                "booking_exit_quality": "booking_ready" if handoff_url else "unavailable",
            }

    persisted_flight = dict(flight or {})
    # Ensure held records always carry route/date primitives required by tracking.
    persisted_flight["origin"] = str(origin or "").strip().upper()
    persisted_flight["destination"] = str(destination or "").strip().upper()
    persisted_flight["departure_iata"] = persisted_flight.get("departure_iata") or persisted_flight["origin"]
    persisted_flight["arrival_iata"] = persisted_flight.get("arrival_iata") or persisted_flight["destination"]
    persisted_flight["date"] = str(depart_date or "").strip()
    if return_date:
        persisted_flight["return_date"] = str(return_date).strip()
    persisted_flight["passengers"] = int(max(1, passengers or 1))
    persisted_flight["booking_handoff"] = dict(handoff_meta)
    if handoff_url:
        persisted_flight["handoff_url"] = handoff_url
    else:
        persisted_flight.pop("handoff_url", None)
        persisted_flight.pop("search_assist_url", None)
        persisted_flight.pop("fallback_search_url", None)

    def _db_operations():
        db = SessionLocal()
        try:
            booking = Booking(
                status="HELD",
                flight=persisted_flight,
                passenger=passenger,
                booking_token=persisted_flight.get("booking_token"),
                shareable_link=persisted_flight.get("shareable_link"),
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
                    "flight_no": persisted_flight.get("flight_no"),
                    "handoff_url_source": handoff_source,
                    "checkout_ready": bool(handoff_url),
                    "checkout_status": _normalized_checkout_status(
                        handoff_url_value=handoff_url,
                        handoff_meta_value=handoff_meta,
                    ),
                }
            )

            return booking.id, booking.status, booking.expires_at.isoformat()
        finally:
            db.close()

    b_id, b_status, b_expires = await asyncio.to_thread(_db_operations)
    checkout_ready = bool(handoff_url)
    checkout_status = _normalized_checkout_status(
        handoff_url_value=handoff_url,
        handoff_meta_value=handoff_meta,
    )

    return {
        "id": b_id,
        "status": b_status,
        "handoff_url": handoff_url,
        "expires_at": b_expires,
        "checkout_ready": checkout_ready,
        "checkout_status": checkout_status,
        "hold_outcome": "held_with_checkout" if checkout_ready else "held_local_only",
        "booking_handoff": handoff_meta,
    }


def get_booking(booking_id: int) -> Optional[dict]:
    """Retrieve a booking by id. Returns None if not found."""
    db = SessionLocal()
    try:
        b = db.get(Booking, booking_id)
        if not b:
            return None
        flight_payload = b.flight if isinstance(b.flight, dict) else {}
        booking_handoff = (
            flight_payload.get("booking_handoff")
            if isinstance(flight_payload.get("booking_handoff"), dict)
            else None
        )
        checkout_ready = bool(_is_usable_handoff_url(b.handoff_url))
        checkout_status = "booking_ready" if checkout_ready else "provider_handoff_unavailable"
        return {
            "id":             b.id,
            "status":         b.status,
            "flight":         b.flight,
            "passenger":      b.passenger,
            "booking_token":  b.booking_token,
            "shareable_link": b.shareable_link,
            "handoff_url":    b.handoff_url,
            "checkout_ready": checkout_ready,
            "checkout_status": checkout_status,
            "hold_outcome": "held_with_checkout" if checkout_ready else "held_local_only",
            "booking_handoff": booking_handoff,
            "created_at":     b.created_at.isoformat() if b.created_at else None,
            "expires_at":     b.expires_at.isoformat() if b.expires_at else None,
        }
    finally:
        db.close()


def list_bookings(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    """
    List recent bookings, optionally filtered by status.
    Intended for lightweight UI status surfaces (not large-scale pagination).
    """
    db = SessionLocal()
    try:
        q = db.query(Booking)
        if status:
            q = q.filter(Booking.status == str(status).upper())
        rows = q.order_by(Booking.created_at.desc()).limit(max(1, int(limit))).all()
        payload_rows: list[dict] = []
        for b in rows:
            flight_payload = b.flight if isinstance(b.flight, dict) else {}
            booking_handoff = (
                flight_payload.get("booking_handoff")
                if isinstance(flight_payload.get("booking_handoff"), dict)
                else None
            )
            checkout_ready = bool(_is_usable_handoff_url(b.handoff_url))
            payload_rows.append(
                {
                    "id":             b.id,
                    "status":         b.status,
                    "flight":         b.flight,
                    "passenger":      b.passenger,
                    "booking_token":  b.booking_token,
                    "shareable_link": b.shareable_link,
                    "handoff_url":    b.handoff_url,
                    "checkout_ready": checkout_ready,
                    "checkout_status": "booking_ready" if checkout_ready else "provider_handoff_unavailable",
                    "hold_outcome": "held_with_checkout" if checkout_ready else "held_local_only",
                    "booking_handoff": booking_handoff,
                    "created_at":     b.created_at.isoformat() if b.created_at else None,
                    "expires_at":     b.expires_at.isoformat() if b.expires_at else None,
                }
            )
        return payload_rows
    finally:
        db.close()


def cancel_booking(booking_id: int) -> bool:
    """
    Cancel a local booking-follow-up record.

    Cancel is allowed for active local records. EXPIRED rows remain immutable.
    """
    db = SessionLocal()
    try:
        b = db.get(Booking, booking_id)
        if not b:
            return False
        if b.status == "EXPIRED":
            return False
        if b.status == "CANCELLED":
            return True
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
        payload_rows: list[dict] = []
        for b in rows:
            checkout_ready = bool(_is_usable_handoff_url(b.handoff_url))
            payload_rows.append(
                {
                    "id":            b.id,
                    "flight":        b.flight,
                    "booking_token": b.booking_token,
                    "handoff_url":   b.handoff_url,
                    "checkout_ready": checkout_ready,
                    "checkout_status": "booking_ready" if checkout_ready else "provider_handoff_unavailable",
                    "expires_at":    b.expires_at.isoformat() if b.expires_at else None,
                }
            )
        return payload_rows
    finally:
        db.close()


def expire_held_booking_for_tracking_invalid_data(
    booking_id: int,
    *,
    reason: str,
    emit_warning: bool = True,
) -> bool:
    """
    Mark an active HELD booking as EXPIRED when tracking prerequisites are invalid.
    Used to quarantine legacy malformed rows so tracker warnings do not repeat forever.
    """
    db = SessionLocal()
    try:
        booking = db.get(Booking, booking_id)
        if not booking:
            return False
        if booking.status != "HELD":
            return False
        booking.status = "EXPIRED"
        if isinstance(booking.flight, dict):
            flight_payload = dict(booking.flight)
            tracking_meta = (
                dict(flight_payload.get("tracking_meta"))
                if isinstance(flight_payload.get("tracking_meta"), dict)
                else {}
            )
            tracking_meta["invalidated_reason"] = str(reason or "invalid_tracking_data")
            tracking_meta["invalidated_at"] = datetime.utcnow().isoformat() + "Z"
            flight_payload["tracking_meta"] = tracking_meta
            booking.flight = flight_payload
        db.commit()
        if emit_warning:
            logger.warning(
                "Expired held booking due to invalid tracking data",
                extra={"booking_id": booking_id, "reason": reason},
            )
        return True
    finally:
        db.close()
