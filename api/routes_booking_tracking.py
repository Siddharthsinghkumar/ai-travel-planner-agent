"""Booking + price-tracking route boundary.

Keeps API orchestration in app.py while isolating booking/tracking HTTP surfaces.
Booking handoff policy remains strict: no Google search-assist fallback URL is
valid for booking flows; only booking-token/provider-resolution outcomes are.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import secrets
import time
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from api import services_booking_tracking as booking_tracking_service
from core.auth import (
    AuthenticatedPrincipal,
    OptionalPrincipalDiagnostics,
    get_current_principal,
    get_optional_principal,
    get_optional_principal_diagnostics,
)
from core.env_config import get_env_bool, get_env_int, get_env_str

_IDEMPOTENCY_LOCK = asyncio.Lock()
_IDEMPOTENCY_RECORDS: Dict[str, Dict[str, Any]] = {}
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
BOOKING_IDEMPOTENCY_WINDOW_SECONDS = max(
    1,
    get_env_int("BOOKING_IDEMPOTENCY_WINDOW_SECONDS", 900),
)
BOOKING_IDEMPOTENCY_MAX_ENTRIES = max(
    100,
    get_env_int("BOOKING_IDEMPOTENCY_MAX_ENTRIES", 5000),
)
BOOKING_HANDOFF_RESOLVE_ALLOW_LOCAL_DEV_WITHOUT_AUTH = get_env_bool(
    "BOOKING_HANDOFF_RESOLVE_ALLOW_LOCAL_DEV_WITHOUT_AUTH",
    default=False,
)
BOOKING_HANDOFF_LOCAL_DEV_SECRET = get_env_str(
    "BOOKING_HANDOFF_LOCAL_DEV_SECRET",
    default="",
).strip() or None


def _monotonic_now() -> float:
    return time.monotonic()


def _reset_booking_idempotency_state_for_tests() -> None:
    _IDEMPOTENCY_RECORDS.clear()


def _normalize_idempotency_key(raw: Optional[str]) -> Optional[str]:
    key = str(raw or "").strip()
    if not key:
        return None
    if not _IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_idempotency_key",
                "message": "Idempotency-Key must be 8-128 chars of A-Z, a-z, 0-9, '.', '_', ':', or '-'.",
            },
        )
    return key


def _normalized_host(raw: Optional[str]) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return ""
    if value.startswith("[") and "]" in value:
        value = value[1 : value.index("]")]
    elif value.count(":") == 1:
        value = value.split(":", 1)[0]
    return value.strip()


def _request_is_loopback(request: Request) -> bool:
    client_host = _normalized_host(getattr(getattr(request, "client", None), "host", None))
    return client_host in _LOOPBACK_HOSTS


def _resolve_handoff_access_mode(
    *,
    request: Request,
    principal: Optional[AuthenticatedPrincipal],
    x_local_dev_secret: Optional[str] = None,
) -> tuple[str, Optional[AuthenticatedPrincipal]]:
    if principal is not None:
        return "authenticated_token", principal
    if BOOKING_HANDOFF_RESOLVE_ALLOW_LOCAL_DEV_WITHOUT_AUTH and _request_is_loopback(request):
        if BOOKING_HANDOFF_LOCAL_DEV_SECRET is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "local_dev_mode_not_configured",
                    "reason": "local_dev_unauthed_mode_requires_secret",
                    "message": (
                        "Local-dev unauthenticated handoff is not properly configured. "
                        "Set BOOKING_HANDOFF_LOCAL_DEV_SECRET for local development."
                    ),
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        provided = (x_local_dev_secret or "").strip()
        if not provided or not secrets.compare_digest(provided, BOOKING_HANDOFF_LOCAL_DEV_SECRET):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "local_dev_auth_required",
                    "reason": "local_dev_secret_required",
                    "message": (
                        "Local-dev unauthenticated handoff requires a valid secret. "
                        "Provide X-Local-Dev-Secret header with the configured secret."
                    ),
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        return "local_dev_unauthed", None
    raise HTTPException(
        status_code=401,
        detail={
            "error": "booking_handoff_auth_required",
            "reason": "auth_required_for_handoff_resolution",
            "message": (
                "Authentication required for booking handoff resolution. "
                "Configure a bearer token, or enable local-dev unauthenticated mode explicitly."
            ),
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


def _booking_handoff_capability_payload(
    *,
    request: Request,
    auth_diagnostics: OptionalPrincipalDiagnostics,
) -> Dict[str, Any]:
    principal = auth_diagnostics.principal
    token_present = bool(auth_diagnostics.token_present)
    token_valid = bool(auth_diagnostics.token_valid)
    auth_rejected = bool(auth_diagnostics.auth_rejected)
    auth_error = str(auth_diagnostics.auth_error or "").strip() or None
    loopback_request = _request_is_loopback(request)
    local_dev_unauth_configured = bool(BOOKING_HANDOFF_RESOLVE_ALLOW_LOCAL_DEV_WITHOUT_AUTH)
    local_dev_unauth_available = bool(
        local_dev_unauth_configured
        and loopback_request
        and BOOKING_HANDOFF_LOCAL_DEV_SECRET is not None
    )

    if principal is not None:
        return {
            "action": "booking_handoff_capabilities",
            "resolve_available_now": True,
            "auth_mode": "authenticated_token",
            "resolve_auth_mode": "auto",
            "auth_required": True,
            "has_valid_token": True,
            "token_present": token_present,
            "auth_rejected": False,
            "auth_error": None,
            "blocked_reason": None,
            "local_dev_unauth_configured": local_dev_unauth_configured,
            "local_dev_unauth_enabled": local_dev_unauth_configured,
            "loopback_request": loopback_request,
            "loopback_eligible": loopback_request,
            "local_dev_unauth_available": local_dev_unauth_available,
            "message": "Authenticated bearer token is active for booking handoff resolution.",
        }
    if local_dev_unauth_available:
        message = "Local-dev unauthenticated handoff resolution is enabled for loopback requests."
        if auth_rejected:
            message = (
                "Provided bearer token was rejected. "
                "Local-dev unauthenticated handoff resolution is still available for loopback requests."
            )
        return {
            "action": "booking_handoff_capabilities",
            "resolve_available_now": True,
            "auth_mode": "local_dev_unauthed",
            "resolve_auth_mode": "omit",
            "auth_required": False,
            "has_valid_token": False,
            "token_present": token_present,
            "auth_rejected": auth_rejected,
            "auth_error": auth_error,
            "blocked_reason": None,
            "local_dev_unauth_configured": local_dev_unauth_configured,
            "local_dev_unauth_enabled": local_dev_unauth_configured,
            "loopback_request": loopback_request,
            "loopback_eligible": loopback_request,
            "local_dev_unauth_available": local_dev_unauth_available,
            "message": message,
        }
    if auth_rejected:
        return {
            "action": "booking_handoff_capabilities",
            "resolve_available_now": False,
            "auth_mode": "auth_required",
            "resolve_auth_mode": "auto",
            "auth_required": True,
            "has_valid_token": False,
            "token_present": token_present,
            "auth_rejected": True,
            "auth_error": auth_error,
            "blocked_reason": "invalid_token",
            "local_dev_unauth_configured": local_dev_unauth_configured,
            "local_dev_unauth_enabled": local_dev_unauth_configured,
            "loopback_request": loopback_request,
            "loopback_eligible": loopback_request,
            "local_dev_unauth_available": local_dev_unauth_available,
            "message": (
                "Configured bearer token was rejected for booking handoff resolution. "
                "Use a valid backend token, or enable local-dev unauthenticated mode explicitly."
            ),
        }
    if local_dev_unauth_configured and not loopback_request:
        return {
            "action": "booking_handoff_capabilities",
            "resolve_available_now": False,
            "auth_mode": "auth_required",
            "resolve_auth_mode": "auto",
            "auth_required": True,
            "has_valid_token": False,
            "token_present": token_present,
            "auth_rejected": False,
            "auth_error": None,
            "blocked_reason": "loopback_required_for_local_dev",
            "local_dev_unauth_configured": local_dev_unauth_configured,
            "local_dev_unauth_enabled": local_dev_unauth_configured,
            "loopback_request": loopback_request,
            "loopback_eligible": loopback_request,
            "local_dev_unauth_available": local_dev_unauth_available,
            "message": (
                "Local-dev unauthenticated handoff resolution is configured, "
                "but this request is not loopback-eligible."
            ),
        }
    return {
        "action": "booking_handoff_capabilities",
        "resolve_available_now": False,
        "auth_mode": "auth_required",
        "resolve_auth_mode": "auto",
        "auth_required": True,
        "has_valid_token": False,
        "token_present": token_present,
        "auth_rejected": False,
        "auth_error": None,
        "blocked_reason": "missing_token",
        "local_dev_unauth_configured": local_dev_unauth_configured,
        "local_dev_unauth_enabled": local_dev_unauth_configured,
        "loopback_request": loopback_request,
        "loopback_eligible": loopback_request,
        "local_dev_unauth_available": local_dev_unauth_available,
        "message": (
            "No bearer token was provided for booking handoff resolution. "
            "Configure a valid token, or enable local-dev unauthenticated mode explicitly."
        ),
    }


def _idempotency_fingerprint(
    *,
    principal_id: str,
    operation: str,
    payload: Dict[str, Any],
) -> str:
    normalized_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    raw = "|".join(
        [
            str(principal_id or "").strip(),
            str(operation or "").strip(),
            normalized_payload,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _idempotency_entry_key(*, principal_id: str, operation: str, key: str) -> str:
    return f"{str(principal_id or '').strip()}::{str(operation or '').strip()}::{str(key or '').strip()}"


def _prune_expired_idempotency_locked(now_monotonic: float) -> None:
    expired = [
        record_key
        for record_key, record in _IDEMPOTENCY_RECORDS.items()
        if float(record.get("expires_at", 0.0) or 0.0) <= now_monotonic
    ]
    for record_key in expired:
        _IDEMPOTENCY_RECORDS.pop(record_key, None)
    if len(_IDEMPOTENCY_RECORDS) <= BOOKING_IDEMPOTENCY_MAX_ENTRIES:
        return
    ordered = sorted(
        _IDEMPOTENCY_RECORDS.items(),
        key=lambda item: float(item[1].get("expires_at", 0.0) or 0.0),
    )
    overflow = len(_IDEMPOTENCY_RECORDS) - BOOKING_IDEMPOTENCY_MAX_ENTRIES
    for record_key, _ in ordered[:overflow]:
        _IDEMPOTENCY_RECORDS.pop(record_key, None)


async def _run_with_idempotency(
    *,
    principal: AuthenticatedPrincipal,
    operation: str,
    idempotency_key: Optional[str],
    request_payload: Dict[str, Any],
    execute_fn,
) -> Dict[str, Any]:
    normalized_key = _normalize_idempotency_key(idempotency_key)
    if not normalized_key:
        return await execute_fn()

    principal_id = str(principal.principal_id or "").strip()
    if not principal_id:
        raise HTTPException(status_code=401, detail="Authentication required.")
    fingerprint = _idempotency_fingerprint(
        principal_id=principal_id,
        operation=operation,
        payload=request_payload,
    )
    entry_key = _idempotency_entry_key(
        principal_id=principal_id,
        operation=operation,
        key=normalized_key,
    )
    now_monotonic = _monotonic_now()
    async with _IDEMPOTENCY_LOCK:
        _prune_expired_idempotency_locked(now_monotonic)
        existing = _IDEMPOTENCY_RECORDS.get(entry_key)
        if existing is not None:
            if str(existing.get("fingerprint") or "") != fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "idempotency_key_conflict",
                        "message": "Idempotency-Key has already been used with a different request payload.",
                    },
                )
            if str(existing.get("state") or "") == "completed":
                response_payload = existing.get("response_payload")
                if isinstance(response_payload, dict):
                    replay = copy.deepcopy(response_payload)
                    replay["idempotency_replayed"] = True
                    return replay
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "idempotency_request_in_progress",
                    "message": "Another request with this Idempotency-Key is still being processed.",
                },
            )
        _IDEMPOTENCY_RECORDS[entry_key] = {
            "state": "in_progress",
            "fingerprint": fingerprint,
            "expires_at": now_monotonic + max(1.0, BOOKING_IDEMPOTENCY_WINDOW_SECONDS),
            "created_at": now_monotonic,
        }

    try:
        result = await execute_fn()
    except Exception:
        async with _IDEMPOTENCY_LOCK:
            current = _IDEMPOTENCY_RECORDS.get(entry_key)
            if current and str(current.get("fingerprint") or "") == fingerprint:
                _IDEMPOTENCY_RECORDS.pop(entry_key, None)
        raise

    async with _IDEMPOTENCY_LOCK:
        _IDEMPOTENCY_RECORDS[entry_key] = {
            "state": "completed",
            "fingerprint": fingerprint,
            "response_payload": copy.deepcopy(result),
            "expires_at": _monotonic_now() + max(1.0, BOOKING_IDEMPOTENCY_WINDOW_SECONDS),
            "created_at": now_monotonic,
        }
        _prune_expired_idempotency_locked(_monotonic_now())
    return result


class BookingHoldRequest(BaseModel):
    flight: Dict[str, Any] = Field(default_factory=dict)
    origin: str = Field(min_length=3, max_length=8)
    destination: str = Field(min_length=3, max_length=8)
    depart_date: str = Field(min_length=6, max_length=32)
    return_date: Optional[str] = Field(default=None, max_length=32)
    passengers: int = Field(default=1, ge=1, le=9)
    hold_minutes: Optional[int] = Field(default=None, ge=1, le=10080)
    passenger: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def normalize(self):
        self.origin = str(self.origin or "").strip().upper()
        self.destination = str(self.destination or "").strip().upper()
        self.depart_date = str(self.depart_date or "").strip()
        if self.return_date:
            self.return_date = str(self.return_date).strip()
        self.passengers = max(1, int(self.passengers or 1))
        if not self.origin or not self.destination or not self.depart_date:
            raise ValueError("origin, destination, and depart_date are required")
        return self

    @field_validator("flight")
    @classmethod
    def validate_flight_payload_size(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if len(encoded.encode("utf-8")) > 65536:
            raise ValueError("flight payload exceeds maximum supported size")
        return value

    @field_validator("passenger")
    @classmethod
    def validate_passenger_payload_size(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if len(encoded.encode("utf-8")) > 16384:
            raise ValueError("passenger payload exceeds maximum supported size")
        return value


class BookingCancelRequest(BaseModel):
    booking_id: int

    @field_validator("booking_id")
    @classmethod
    def validate_booking_id(cls, value):
        if value is None or int(value) <= 0:
            raise ValueError("booking_id must be a positive integer")
        return int(value)


class BookingTrackRequest(BookingHoldRequest):
    pass


class BookingHandoffResolveRequest(BaseModel):
    flight: Dict[str, Any] = Field(default_factory=dict)
    origin: str = Field(min_length=3, max_length=8)
    destination: str = Field(min_length=3, max_length=8)
    depart_date: str = Field(min_length=6, max_length=32)
    return_date: Optional[str] = Field(default=None, max_length=32)
    passengers: int = Field(default=1, ge=1, le=9)

    @model_validator(mode="after")
    def normalize(self):
        self.origin = str(self.origin or "").strip().upper()
        self.destination = str(self.destination or "").strip().upper()
        self.depart_date = str(self.depart_date or "").strip()
        if self.return_date:
            self.return_date = str(self.return_date).strip()
        self.passengers = max(1, int(self.passengers or 1))
        if not self.origin or not self.destination or not self.depart_date:
            raise ValueError("origin, destination, and depart_date are required")
        return self

    @field_validator("flight")
    @classmethod
    def validate_flight_payload_size(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if len(encoded.encode("utf-8")) > 65536:
            raise ValueError("flight payload exceeds maximum supported size")
        return value


def build_booking_tracking_router(
    app: FastAPI,
    *,
    logger: logging.Logger,
    job_contract_payload_fn: Callable[[], Dict[str, Any]],
) -> APIRouter:
    router = APIRouter()

    @router.get("/booking/handoff/capabilities")
    async def booking_handoff_capabilities(
        request: Request,
        response: Response,
        auth_diagnostics: OptionalPrincipalDiagnostics = Depends(get_optional_principal_diagnostics),
    ):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "Authorization"
        return _booking_handoff_capability_payload(
            request=request,
            auth_diagnostics=auth_diagnostics,
        )

    @router.post("/booking/hold")
    async def booking_hold(
        req: BookingHoldRequest,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(get_current_principal),
    ):
        """Create a HELD booking for the selected flight."""
        request_payload = req.model_dump(mode="json", exclude_none=False)
        return await _run_with_idempotency(
            principal=principal,
            operation="booking_hold",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
            execute_fn=lambda: booking_tracking_service.booking_hold(req, principal=principal, logger=logger),
        )

    @router.post("/booking/handoff/resolve")
    async def booking_handoff_resolve(
        req: BookingHandoffResolveRequest,
        request: Request,
        x_local_dev_secret: Optional[str] = Header(default=None, alias="X-Local-Dev-Secret"),
        principal: Optional[AuthenticatedPrincipal] = Depends(get_optional_principal),
    ):
        """Resolve provider booking handoff for one selected flight row."""
        auth_mode, effective_principal = _resolve_handoff_access_mode(
            request=request,
            principal=principal,
            x_local_dev_secret=x_local_dev_secret,
        )
        return await booking_tracking_service.booking_resolve_handoff(
            req,
            principal=effective_principal,
            auth_mode=auth_mode,
            logger=logger,
        )

    @router.post("/booking/track-price")
    async def booking_track_price(
        req: BookingTrackRequest,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(get_current_principal),
    ):
        """Create a HELD local record and enable price tracking for the selected flight."""
        request_payload = req.model_dump(mode="json", exclude_none=False)
        return await _run_with_idempotency(
            principal=principal,
            operation="booking_track_price",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
            execute_fn=lambda: booking_tracking_service.booking_track_price(
                req,
                principal=principal,
                price_tracker_enabled=bool(getattr(app.state, "price_tracker_enabled", False)),
                logger=logger,
            ),
        )

    @router.post("/booking/cancel")
    async def booking_cancel(
        req: BookingCancelRequest,
        principal: AuthenticatedPrincipal = Depends(get_current_principal),
    ):
        """Cancel a local booking follow-up record."""
        return await booking_tracking_service.booking_cancel(req.booking_id, principal=principal)

    @router.get("/bookings/{booking_id}")
    async def booking_get(
        booking_id: int,
        principal: AuthenticatedPrincipal = Depends(get_current_principal),
    ):
        return await booking_tracking_service.booking_get(booking_id, principal=principal)

    @router.get("/bookings")
    async def booking_list(
        status: Optional[str] = None,
        limit: int = Query(100, ge=1, le=500),
        principal: AuthenticatedPrincipal = Depends(get_current_principal),
    ):
        return await booking_tracking_service.booking_list(status, limit, principal=principal)

    @router.get("/price-tracking/status")
    async def price_tracking_status(
        principal: AuthenticatedPrincipal = Depends(get_current_principal),
    ):
        return booking_tracking_service.price_tracking_status_payload(
            price_tracker_enabled=bool(getattr(app.state, "price_tracker_enabled", False)),
            price_tracker_status=getattr(app.state, "price_tracker_status", {}) or {},
            job_contract_payload=job_contract_payload_fn(),
        )

    @router.get("/price-tracking/alerts")
    async def price_tracking_alerts(
        booking_id: Optional[int] = None,
        principal: AuthenticatedPrincipal = Depends(get_current_principal),
    ):
        return await booking_tracking_service.price_tracking_alerts(
            principal=principal,
            booking_id=booking_id,
            job_contract_payload=job_contract_payload_fn(),
        )

    @router.post("/price-tracking/alerts/{alert_id}/ack")
    async def price_tracking_ack(
        alert_id: int,
        principal: AuthenticatedPrincipal = Depends(get_current_principal),
    ):
        return await booking_tracking_service.price_tracking_ack(
            alert_id,
            principal=principal,
            job_contract_payload=job_contract_payload_fn(),
        )

    return router
