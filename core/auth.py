"""Minimal authenticated-principal seam for protected API surfaces.

This module intentionally provides a small, pluggable dependency layer:
- one canonical principal object
- one canonical required-auth dependency
- one optional-auth dependency (for mixed public/protected endpoints)
"""

from __future__ import annotations

import json
import secrets
from typing import Dict, Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from core.env_config import get_env_str

_bearer = HTTPBearer(auto_error=False)


class AuthenticatedPrincipal(BaseModel):
    principal_id: str
    auth_source: str = "static_bearer"


class OptionalPrincipalDiagnostics(BaseModel):
    principal: Optional[AuthenticatedPrincipal] = None
    token_present: bool = False
    token_valid: bool = False
    auth_rejected: bool = False
    auth_error: Optional[str] = None
    auth_error_message: Optional[str] = None


def _unauthorized(detail: str = "Authentication required.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _tokens_from_json(raw: str) -> Dict[str, str]:
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    mapping: Dict[str, str] = {}
    for principal_id, token in data.items():
        p = str(principal_id or "").strip()
        t = str(token or "").strip()
        if p and t:
            mapping[t] = p
    return mapping


def _tokens_from_csv(raw: str) -> Dict[str, str]:
    """Parse AUTH_BEARER_TOKENS as `principal_a:token_a,principal_b:token_b`."""
    mapping: Dict[str, str] = {}
    for chunk in str(raw or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        principal, sep, token = item.partition(":")
        if sep != ":":
            continue
        p = principal.strip()
        t = token.strip()
        if p and t:
            mapping[t] = p
    return mapping


def _token_to_principal() -> Dict[str, str]:
    # Preferred explicit mappings
    json_map = _tokens_from_json(get_env_str("AUTH_BEARER_TOKENS_JSON", "") or "")
    csv_map = _tokens_from_csv(get_env_str("AUTH_BEARER_TOKENS", "") or "")
    merged = {**json_map, **csv_map}

    # Minimal single-token fallback for local/dev hardening rollout
    single = (get_env_str("AUTH_TOKEN", "") or "").strip()
    if single:
        principal = (get_env_str("AUTH_DEFAULT_PRINCIPAL_ID", "default_principal") or "default_principal").strip()
        merged.setdefault(single, principal)
    return merged


def _resolve_principal_for_token(token: str) -> Optional[str]:
    token_value = str(token or "").strip()
    if not token_value:
        return None
    for known_token, principal_id in _token_to_principal().items():
        if secrets.compare_digest(token_value, known_token):
            return principal_id
    return None


def _extract_bearer_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> Optional[str]:
    auth_header = (request.headers.get("Authorization") or "").strip()
    if credentials is None:
        if auth_header:
            # Header was supplied but not parsed as valid bearer credentials.
            raise _unauthorized("Invalid authorization header.")
        return None
    token = (credentials.credentials or "").strip()
    if not token:
        raise _unauthorized("Missing bearer token.")
    return token


async def get_optional_principal(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> Optional[AuthenticatedPrincipal]:
    diagnostics = await get_optional_principal_diagnostics(request, credentials)
    if diagnostics.principal is not None:
        return diagnostics.principal
    if diagnostics.auth_rejected:
        raise _unauthorized(diagnostics.auth_error_message or "Invalid authentication token.")
    return None


async def get_optional_principal_diagnostics(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> OptionalPrincipalDiagnostics:
    auth_header = (request.headers.get("Authorization") or "").strip()
    if credentials is None:
        if auth_header:
            return OptionalPrincipalDiagnostics(
                principal=None,
                token_present=True,
                token_valid=False,
                auth_rejected=True,
                auth_error="invalid_authorization_header",
                auth_error_message="Invalid authorization header.",
            )
        return OptionalPrincipalDiagnostics()

    token = (credentials.credentials or "").strip()
    if not token:
        return OptionalPrincipalDiagnostics(
            principal=None,
            token_present=True,
            token_valid=False,
            auth_rejected=True,
            auth_error="missing_bearer_token",
            auth_error_message="Missing bearer token.",
        )

    principal_id = _resolve_principal_for_token(token)
    if not principal_id:
        return OptionalPrincipalDiagnostics(
            principal=None,
            token_present=True,
            token_valid=False,
            auth_rejected=True,
            auth_error="invalid_token",
            auth_error_message="Invalid authentication token.",
        )

    return OptionalPrincipalDiagnostics(
        principal=AuthenticatedPrincipal(principal_id=principal_id, auth_source="static_bearer"),
        token_present=True,
        token_valid=True,
        auth_rejected=False,
        auth_error=None,
        auth_error_message=None,
    )


async def get_current_principal(
    principal: Optional[AuthenticatedPrincipal] = Security(get_optional_principal),
) -> AuthenticatedPrincipal:
    if principal is None:
        raise _unauthorized()
    return principal
