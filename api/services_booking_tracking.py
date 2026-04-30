"""Booking + tracking service boundary extracted from api.app.

Keeps route handlers thin while preserving existing API behavior.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from typing import Any, Dict, Optional

from fastapi import HTTPException

from core.auth import AuthenticatedPrincipal
from core.env_config import get_env_int

_BOOKING_RESOLVE_INFLIGHT_LOCK = asyncio.Lock()
_BOOKING_RESOLVE_INFLIGHT: Dict[str, asyncio.Task] = {}


def _booking_resolve_request_key(req: Any) -> str:
    flight = req.flight if isinstance(getattr(req, "flight", None), dict) else {}
    fingerprint_payload = {
        "origin": str(getattr(req, "origin", "") or "").strip().upper(),
        "destination": str(getattr(req, "destination", "") or "").strip().upper(),
        "depart_date": str(getattr(req, "depart_date", "") or "").strip(),
        "return_date": str(getattr(req, "return_date", "") or "").strip(),
        "passengers": int(getattr(req, "passengers", 1) or 1),
        "booking_token": str(flight.get("booking_token") or "").strip(),
        "flight_no": str(flight.get("flight_no") or "").strip(),
        "airline": str(flight.get("airline") or "").strip(),
        "departure_time": str(flight.get("departure_time") or "").strip(),
        "arrival_time": str(flight.get("arrival_time") or "").strip(),
        "price_inr": str(flight.get("price_inr") or "").strip(),
    }
    serialized = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


async def _resolve_handoff_detail_coalesced(req: Any) -> Any:
    from tools.booking_handoff import build_booking_handoff_url

    key = _booking_resolve_request_key(req)
    existing: Optional[asyncio.Task] = None
    async with _BOOKING_RESOLVE_INFLIGHT_LOCK:
        existing = _BOOKING_RESOLVE_INFLIGHT.get(key)
        if existing is None:
            task = asyncio.create_task(
                build_booking_handoff_url(
                    flight=req.flight,
                    origin=req.origin,
                    destination=req.destination,
                    depart_date=req.depart_date,
                    return_date=req.return_date,
                    passengers=req.passengers,
                    return_details=True,
                )
            )
            _BOOKING_RESOLVE_INFLIGHT[key] = task
            existing = task
    try:
        return await existing
    finally:
        async with _BOOKING_RESOLVE_INFLIGHT_LOCK:
            current = _BOOKING_RESOLVE_INFLIGHT.get(key)
            if current is existing:
                _BOOKING_RESOLVE_INFLIGHT.pop(key, None)


def _coerce_price_inr(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            parsed = float(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace("₹", "").replace(",", "").strip()
    try:
        parsed = float(cleaned)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _tracking_detail(*, error: str, reason: str, message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error": error,
        "reason": reason,
        "message": message,
        "contract": "price_tracking_requires_supported_selected_flight",
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    return payload


def _normalize_booking_handoff_detail(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        status = str(payload.get("status") or "unavailable").strip().lower()
        url_candidate = str(payload.get("url") or "").strip() or None
        handoff_url = url_candidate if status == "booking_ready" else None
        normalized: Dict[str, Any] = {
            "status": "booking_ready" if handoff_url else "unavailable",
            "reason": str(payload.get("reason") or ("resolved_booking_token" if handoff_url else "booking_token_unresolved")),
            "source": str(payload.get("source") or "booking_token"),
            "url": handoff_url,
            "booking_exit_quality": "booking_ready" if handoff_url else "unavailable",
        }
        provider = payload.get("provider")
        if provider:
            normalized["provider"] = provider
        if isinstance(payload.get("cache_hit"), bool):
            normalized["cache_hit"] = payload.get("cache_hit")
        blocked_domain = str(payload.get("blocked_domain") or "").strip()
        if blocked_domain:
            normalized["blocked_domain"] = blocked_domain
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics:
            normalized["diagnostics"] = diagnostics
        return normalized
    return {
        "status": "unavailable",
        "reason": "booking_handoff_unavailable",
        "source": "booking_token",
        "url": None,
        "booking_exit_quality": "unavailable",
    }


async def booking_resolve_handoff(
    req: Any,
    *,
    principal: Optional[AuthenticatedPrincipal],
    auth_mode: str,
    logger,
) -> Dict[str, Any]:
    try:
        handoff_detail_raw = await _resolve_handoff_detail_coalesced(req)
    except Exception as exc:
        logger.warning(
            "booking_handoff_resolve_failed",
            extra={"exception_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "booking_handoff_resolve_failed",
                "reason": "provider_resolution_failed",
                "message": "Provider handoff resolution failed for this selected flight.",
            },
        )

    handoff_detail = _normalize_booking_handoff_detail(handoff_detail_raw)
    handoff_url = str(handoff_detail.get("url") or "").strip() or None
    booking_ready = bool(handoff_url) and str(handoff_detail.get("status") or "").strip().lower() == "booking_ready"
    reason = str(handoff_detail.get("reason") or "")
    blocked_category = None
    retryable: Optional[bool] = None
    if booking_ready:
        message = "Provider handoff resolved for this selected flight."
        retryable = False
    elif reason == "booking_options_request_exception":
        # Drill into diagnostics to distinguish key exhaustion from network/timeout failures.
        _error_ctx = (handoff_detail.get("diagnostics") or {}).get("booking_options_error_context") or {}
        _exception_bucket = str(_error_ctx.get("exception_bucket") or "").strip()
        if _exception_bucket == "no_active_key":
            blocked_category = "provider_key_exhausted"
            message = (
                "No active SerpAPI key is available — all configured keys are exhausted or on cooldown. "
                "Wait for cooldown to expire or add more SERPAPI_KEY_n entries."
            )
            # Keys recover from cooldown quickly; allow retry after a short wait.
            retryable = True
        else:
            blocked_category = "request_exception"
            message = "Provider booking options request failed due to timeout/network exception."
            retryable = True
    elif reason == "booking_options_http_error":
        _error_ctx = (handoff_detail.get("diagnostics") or {}).get("booking_options_error_context") or {}
        _http_status = int(_error_ctx.get("http_status") or 0)
        _is_client_error = 400 <= _http_status < 500 and _http_status not in {408, 429}
        blocked_category = "provider_client_error" if _is_client_error else "http_error"
        # HTTP 400 means the provider definitively rejected the token — retrying the same
        # token will produce the same result.  Other client errors (401/403/422) may be
        # transient (auth rotation, rate-limit adjacent) so remain retryable.
        if _http_status == 400:
            retryable = False
            blocked_category = "provider_unavailable"
        else:
            retryable = True
        if _is_client_error:
            message = (
                f"Provider booking options request returned HTTP {_http_status}. "
                + ("Provider does not support booking for this flight."
                   if _http_status == 400
                   else "The booking token may be expired or invalid. Try again.")
            )
        else:
            message = "Provider booking options request returned an HTTP error."
    elif reason == "provider_domain_not_allowlisted":
        blocked_category = "allowlist_policy"
        blocked_domain = str((handoff_detail.get("blocked_domain") or "")).strip()
        domain_hint = f" (domain: {blocked_domain})" if blocked_domain else ""
        message = (
            f"Provider handoff was blocked by booking-domain allowlist policy{domain_hint}. "
            "If this is a legitimate checkout domain, add it to BOOKING_HANDOFF_ALLOWED_DOMAIN_SUFFIXES."
        )
        retryable = False
    elif reason in {
        "resolved_booking_request_post_unverified_google_tracker",
        "resolved_booking_request_post_unverified_google_tracker_cache",
        "booking_request_post_resolution_failed",
        "booking_request_post_resolution_failed_cache",
        "booking_token_invalid_url",
    }:
        blocked_category = "stale_or_unusable_booking_artifact"
        message = "Provider booking artifact is stale or unusable for a safe handoff URL."
        # Allow one retry: the token may still be fresh enough for a token-only fallback.
        retryable = True
    elif reason == "booking_token_missing":
        blocked_category = "provider_unavailable"
        message = "This flight row has no provider booking token."
        retryable = False
    elif reason == "booking_options_exhausted":
        # All retry budget consumed — the shaped and relaxed attempts both failed.
        # A subsequent attempt may succeed if the provider recovers.
        blocked_category = "resolution_budget_exhausted"
        message = "Provider booking options fetch exhausted its retry budget. Try again later."
        retryable = True
    elif reason in {"booking_token_resolution_exception", "booking_handoff_exception"}:
        blocked_category = "resolution_exception"
        message = "An unexpected error occurred during provider handoff resolution."
        # Unexpected exceptions can be transient; allow retry.
        retryable = True
    elif reason in {"booking_token_unresolved", "booking_token_invalid_url"}:
        # No usable provider URL was obtained — not a transient failure.
        blocked_category = "stale_or_unusable_booking_artifact"
        message = "Provider booking artifact is stale or unusable for a safe handoff URL."
        retryable = False
    elif reason == "booking_token_resolution_timeout":
        # Outer wait_for timed out — the provider was too slow.  Allow a retry after delay.
        blocked_category = "resolution_timeout"
        message = "Provider handoff resolution timed out. Try again after a short wait."
        retryable = True
    elif reason == "booking_options_provider_error":
        blocked_category = "provider_unavailable"
        message = "Provider booking options returned a provider-side error. Try again."
        retryable = True
    else:
        blocked_category = "provider_unavailable"
        message = "Provider handoff is unavailable for this selected flight."
        retryable = False

    # Persist the resolved URL to any existing HELD booking for this flight identity
    # so the checkout link survives frontend state resets.
    if booking_ready and handoff_url and principal is not None:
        from tools.booking_handoff import patch_booking_handoff_url

        flight = req.flight if isinstance(req.flight, dict) else {}
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                patch_booking_handoff_url,
                handoff_url,
                owner_principal_id=principal.principal_id,
                flight_no=str(flight.get("flight_no") or "").strip(),
                airline=str(flight.get("airline") or "").strip(),
                departure_time=str(flight.get("departure_time") or "").strip(),
                arrival_time=str(flight.get("arrival_time") or "").strip(),
                date=str(flight.get("date") or "").strip(),
            )

    # If resolution failed but an existing held record already has a persisted handoff URL
    # for this flight identity, surface it so the frontend stays synced.
    effective_handoff_url = handoff_url if booking_ready else None
    if not effective_handoff_url and principal is not None:
        from tools.booking_handoff import get_persisted_handoff_url_for_flight

        flight = req.flight if isinstance(req.flight, dict) else {}
        with contextlib.suppress(Exception):
            persisted_url = await asyncio.to_thread(
                get_persisted_handoff_url_for_flight,
                owner_principal_id=principal.principal_id,
                flight_no=str(flight.get("flight_no") or "").strip(),
                airline=str(flight.get("airline") or "").strip(),
                departure_time=str(flight.get("departure_time") or "").strip(),
                arrival_time=str(flight.get("arrival_time") or "").strip(),
                date=str(flight.get("date") or req.depart_date or "").strip(),
            )
            if persisted_url:
                effective_handoff_url = persisted_url
                booking_ready = True
                message = "Provider handoff resolved from persisted held record."
                blocked_category = None
                retryable = None

    # Structured decisive log: exactly why this row failed and whether quota was consumed.
    if not booking_ready:
        _diag = handoff_detail.get("diagnostics") or {}
        _failure_bucket = _diag.get("failure_bucket") or _diag.get("artifact_field") or "unknown"
        _http_status = (_diag.get("booking_options_error_context") or {}).get("http_status")
        _exception_bucket = (_diag.get("booking_options_error_context") or {}).get("exception_bucket")
        _quota_wasted = blocked_category in ("provider_key_exhausted", "resolution_budget_exhausted")
        logger.info(
            "booking_resolve_row_failure",
            extra={
                "row_failure": True,
                "reason": reason,
                "blocked_category": blocked_category,
                "failure_bucket": _failure_bucket,
                "http_status": _http_status,
                "exception_bucket": _exception_bucket,
                "retryable": retryable,
                "quota_wasted": _quota_wasted,
                "airline": str((req.flight or {}).get("airline") or ""),
                "flight_no": str((req.flight or {}).get("flight_no") or ""),
                "route": f"{req.origin}->{req.destination}",
                "depart_date": str(req.depart_date or ""),
            },
        )

    return {
        "action": "resolve_booking_handoff",
        "success": booking_ready,
        "handoff_url": effective_handoff_url,
        "booking_handoff": handoff_detail,
        "blocked_reason": None if booking_ready else (reason or "booking_handoff_unavailable"),
        "blocked_category": None if booking_ready else blocked_category,
        "retryable": None if booking_ready else retryable,
        "message": message,
        "auth_mode": auth_mode,
        "auth_required": auth_mode == "authenticated_token",
        "owner_principal_id": principal.principal_id if principal else None,
        "best_flight": req.flight,
    }


async def booking_hold(req: Any, *, principal: AuthenticatedPrincipal, logger) -> Dict[str, Any]:
    from tools.booking_handoff import hold_booking

    hold_minutes = req.hold_minutes or get_env_int("BOOKING_HOLD_MINUTES", 15)
    try:
        held = await hold_booking(
            flight=req.flight,
            origin=req.origin,
            destination=req.destination,
            depart_date=req.depart_date,
            return_date=req.return_date,
            passengers=req.passengers,
            passenger=req.passenger,
            hold_minutes=hold_minutes,
            owner_principal_id=principal.principal_id,
        )
    except Exception as exc:
        logger.warning(
            "booking_hold_creation_failed",
            extra={"exception_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "booking_hold_failed",
                "reason": "hold_creation_failed",
                "message": "Could not create a local hold record for this selection.",
            },
        )

    checkout_ready = bool((held or {}).get("checkout_ready"))
    checkout_status = str((held or {}).get("checkout_status") or ("booking_ready" if checkout_ready else "provider_handoff_unavailable"))
    hold_outcome = str((held or {}).get("hold_outcome") or ("held_with_checkout" if checkout_ready else "held_local_only"))
    return {
        "action": "hold_booking",
        "success": True,
        "hold_created": True,
        "checkout_ready": checkout_ready,
        "checkout_status": checkout_status,
        "hold_outcome": hold_outcome,
        "message": (
            "Flight held successfully. Provider checkout link is ready."
            if checkout_ready
            else "Flight held locally, but provider checkout is currently unavailable."
        ),
        "booking": held,
        "best_flight": req.flight,
    }


async def booking_track_price(
    req: Any,
    *,
    principal: AuthenticatedPrincipal,
    price_tracker_enabled: bool,
    logger,
) -> Dict[str, Any]:
    from tools.booking_handoff import cancel_booking
    from tools.booking_handoff import get_booking
    from tools.booking_handoff import hold_booking
    from tools.price_tracker import record_price_snapshot

    if not price_tracker_enabled:
        raise HTTPException(
            status_code=503,
            detail=_tracking_detail(
                error="price_tracking_disabled",
                reason="disabled_by_configuration",
                message="Price tracking is disabled by configuration.",
            ),
        )

    hold_minutes = req.hold_minutes or get_env_int("PRICE_TRACK_HOLD_MINUTES", 43200)
    baseline_price = _coerce_price_inr(req.flight.get("price_inr"))
    if baseline_price is None:
        raise HTTPException(
            status_code=422,
            detail=_tracking_detail(
                error="price_tracking_unsupported_selection",
                reason="selected_flight_price_unavailable",
                message="Price tracking requires a selected flight with a numeric fare.",
            ),
        )

    held = await hold_booking(
        flight=req.flight,
        origin=req.origin,
        destination=req.destination,
        depart_date=req.depart_date,
        return_date=req.return_date,
        passengers=req.passengers,
        passenger=req.passenger,
        hold_minutes=hold_minutes,
        owner_principal_id=principal.principal_id,
    )
    booking_id_raw = held.get("id")
    booking_id: Optional[int] = None
    try:
        if booking_id_raw is not None:
            booking_id = int(booking_id_raw)
    except Exception:
        booking_id = None

    persisted_booking = None
    if booking_id is not None:
        with contextlib.suppress(Exception):
            persisted_booking = await asyncio.to_thread(
                get_booking,
                booking_id,
                principal.principal_id,
            )

    persisted_flight = (
        persisted_booking.get("flight")
        if isinstance(persisted_booking, dict) and isinstance(persisted_booking.get("flight"), dict)
        else {}
    )

    tracking_missing_fields: list[str] = []
    if not (persisted_flight.get("origin") or persisted_flight.get("departure_iata")):
        tracking_missing_fields.append("origin")
    if not (persisted_flight.get("destination") or persisted_flight.get("arrival_iata")):
        tracking_missing_fields.append("destination")
    if not persisted_flight.get("date"):
        tracking_missing_fields.append("travel_date")
    if _coerce_price_inr(persisted_flight.get("price_inr")) is None:
        tracking_missing_fields.append("held_price")
    if booking_id is None:
        tracking_missing_fields.append("booking_id")
    if not persisted_booking:
        tracking_missing_fields.append("held_booking_record")

    if tracking_missing_fields:
        cancelled = False
        if booking_id is not None:
            try:
                cancelled = bool(
                    await asyncio.to_thread(
                        cancel_booking,
                        booking_id,
                        principal.principal_id,
                    )
                )
            except Exception:
                logger.exception("tracking_setup_cleanup_cancel_failed")
        raise HTTPException(
            status_code=503,
            detail=_tracking_detail(
                error="price_tracking_setup_failed",
                reason="held_tracking_prerequisites_missing",
                message="Price tracking setup failed because HELD booking prerequisites were incomplete.",
                booking_id=booking_id,
                missing_fields=tracking_missing_fields,
                cleanup_cancelled=cancelled,
            ),
        )

    try:
        snapshot_id = record_price_snapshot(
            origin=req.origin,
            destination=req.destination,
            travel_date=req.depart_date,
            price_inr=baseline_price,
        )
        if not isinstance(snapshot_id, int) or snapshot_id <= 0:
            raise RuntimeError("snapshot_persist_failed")
    except Exception as exc:
        booking_id = held.get("id")
        cancelled = False
        if booking_id is not None:
            try:
                cancelled = bool(
                    await asyncio.to_thread(
                        cancel_booking,
                        int(booking_id),
                        principal.principal_id,
                    )
                )
            except Exception:
                logger.exception("tracking_setup_cleanup_cancel_failed")
        logger.warning(
            "record_price_snapshot failed for tracking setup",
            extra={
                "booking_id": booking_id,
                "exception_type": type(exc).__name__,
                "cleanup_cancelled": cancelled,
            },
        )
        raise HTTPException(
            status_code=503,
            detail=_tracking_detail(
                error="price_tracking_setup_failed",
                reason="snapshot_persist_failed",
                message="Price tracking setup failed before monitoring could start.",
                booking_id=booking_id,
                cleanup_cancelled=cancelled,
            ),
        )

    return {
        "action": "track_price",
        "success": True,
        "message": "Price tracking activated for this itinerary.",
        "booking": held,
        "best_flight": req.flight,
        "monitoring_active": True,
        "tracking_state": {
            "booking_id": booking_id,
            "baseline_snapshot_id": snapshot_id,
            "route_tracking_ready": True,
            "checkout_dependency": "not_required",
        },
    }


async def booking_cancel(booking_id: int, *, principal: AuthenticatedPrincipal) -> Dict[str, Any]:
    from tools.booking_handoff import cancel_booking

    ok = await asyncio.to_thread(cancel_booking, booking_id, principal.principal_id)
    if not ok:
        raise HTTPException(status_code=404, detail="booking not found")
    return {
        "action": "cancel_booking",
        "booking_id": booking_id,
        "success": True,
        "message": "Booking cancelled.",
    }


async def booking_get(booking_id: int, *, principal: AuthenticatedPrincipal) -> Dict[str, Any]:
    from tools.booking_handoff import get_booking

    booking = await asyncio.to_thread(get_booking, booking_id, principal.principal_id)
    if booking is None:
        raise HTTPException(status_code=404, detail="booking not found")
    return booking


async def booking_list(
    status: Optional[str],
    limit: int,
    *,
    principal: AuthenticatedPrincipal,
) -> Dict[str, Any]:
    from tools.booking_handoff import list_bookings

    bookings = await asyncio.to_thread(
        list_bookings,
        status,
        limit,
        principal.principal_id,
    )
    return {"count": len(bookings), "items": bookings}


def price_tracking_status_payload(
    *,
    price_tracker_enabled: bool,
    price_tracker_status: Optional[Dict[str, Any]],
    job_contract_payload: Dict[str, Any],
) -> Dict[str, Any]:
    job_runtime_warning = {
        "jobs_tracking_memory_only": bool(job_contract_payload.get("jobs_tracking_memory_only", True)),
        "lost_on_restart": bool(job_contract_payload.get("lost_on_restart", True)),
        "durable_persistence": bool(job_contract_payload.get("durable_persistence", False)),
        "warning": str(job_contract_payload.get("warning") or ""),
    }
    return {
        "enabled": bool(price_tracker_enabled),
        "status": price_tracker_status or {},
        "contract": job_contract_payload,
        "job_runtime_warning": job_runtime_warning,
    }


async def price_tracking_alerts(
    *,
    principal: AuthenticatedPrincipal,
    booking_id: Optional[int] = None,
    job_contract_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from tools.price_tracker import get_unacknowledged_alerts

    alerts = await asyncio.to_thread(
        get_unacknowledged_alerts,
        booking_id,
        principal.principal_id,
    )
    payload: Dict[str, Any] = {"count": len(alerts), "items": alerts}
    if isinstance(job_contract_payload, dict):
        payload["contract"] = job_contract_payload
        payload["job_runtime_warning"] = {
            "jobs_tracking_memory_only": bool(job_contract_payload.get("jobs_tracking_memory_only", True)),
            "lost_on_restart": bool(job_contract_payload.get("lost_on_restart", True)),
            "durable_persistence": bool(job_contract_payload.get("durable_persistence", False)),
            "warning": str(job_contract_payload.get("warning") or ""),
        }
    return payload


async def price_tracking_ack(
    alert_id: int,
    *,
    principal: AuthenticatedPrincipal,
    job_contract_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from tools.price_tracker import acknowledge_alert

    ok = await asyncio.to_thread(acknowledge_alert, alert_id, principal.principal_id)
    if not ok:
        raise HTTPException(status_code=404, detail="alert not found")
    payload: Dict[str, Any] = {"alert_id": alert_id, "acknowledged": True}
    if isinstance(job_contract_payload, dict):
        payload["contract"] = job_contract_payload
        payload["job_runtime_warning"] = {
            "jobs_tracking_memory_only": bool(job_contract_payload.get("jobs_tracking_memory_only", True)),
            "lost_on_restart": bool(job_contract_payload.get("lost_on_restart", True)),
            "durable_persistence": bool(job_contract_payload.get("durable_persistence", False)),
            "warning": str(job_contract_payload.get("warning") or ""),
        }
    return payload
