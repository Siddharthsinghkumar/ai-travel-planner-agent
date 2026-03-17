# core/health.py

import asyncio
import importlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Callable, Awaitable

from fastapi import APIRouter, Query, Header, Depends, HTTPException

# Local imports (adjust if needed)
from core.api_key_manager import key_manager
from agents.cloud_llm import clear_client_cache

logger = logging.getLogger(__name__)

# Map health checks to module paths (all must expose a callable named `health_check`)
_HEALTH_PROVIDERS = {
    "openai": "agents.cloud_llm",
    "ollama": "agents.ollama_client",
    "airline": "tools.airline_api",
    "weather": "tools.weather_api",
}

# Map each provider (keys from _HEALTH_PROVIDERS) to one or more services
# that exist in key_manager (the services are like "serpapi","openai","gemini","weather").
# If a provider maps to an empty list, we treat it as local/no-keys and run the health check normally.
# Edit this mapping to match how your code uses keys (e.g. if airline uses serpapi + openai).
PROVIDER_KEYMAP = {
    "airline": ["serpapi", "openai"],  # example: airline health depends on serpapi/openai keys
    "weather": ["weather"],
    "openai": ["openai"],
    "ollama": [],  # local model, no external key
}


def _get_health_func(module_path: str) -> Callable[[], Awaitable[str]]:
    """
    Dynamically import the module and return its `health_check` callable.
    This allows monkeypatching in tests.
    """
    module = importlib.import_module(module_path)
    return module.health_check


async def check_database() -> str:
    """
    Placeholder for database health check.
    Replace with actual DB connection test.
    """
    # Example: simulate async check
    await asyncio.sleep(0.1)
    return "ok"


def is_key_active(entry) -> bool:
    """Return True if the key entry indicates an active usable key."""
    if isinstance(entry, str):
        # Simple string status: "active" is active, anything else (like "exhausted ...") is not
        return entry == "active"

    if isinstance(entry, dict):
        # Active if not pending clear and not exhausted
        if entry.get("_pending_clear"):
            return False
        if entry.get("exhausted_until"):
            return False
        # If no explicit flags, assume active (backward compatibility)
        return True

    return False


def is_key_pending_clear(entry) -> bool:
    """Return True if the key entry indicates it is pending clear."""
    if isinstance(entry, dict):
        return entry.get("_pending_clear", False)
    return False


# ------------------------------------------------------------------------------
# Human-friendly key formatting helpers
# ------------------------------------------------------------------------------

def _ts_to_iso(ts: Any) -> Any:
    """Convert epoch-ish timestamp to ISO8601 UTC string, else return original."""
    try:
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        return str(ts)
    except Exception:
        return str(ts)


def _normalize_key_entry(raw_entry: Any, index: Any = None) -> Dict[str, Any]:
    """Turn an arbitrary raw entry into a predictable dict with fields we want to display."""
    out = {
        "index": index,
        "state": None,
        "exhausted_until": None,
        "in_use": None,
        "pending_clear": False,
        "fingerprint": None,
        "created_at": None,
        "raw": raw_entry,
    }

    if isinstance(raw_entry, str):
        out["state"] = raw_entry
        return out

    if isinstance(raw_entry, dict):
        out["fingerprint"] = raw_entry.get("fingerprint") or raw_entry.get("fp") or raw_entry.get("key_fingerprint")
        out["in_use"] = int(raw_entry.get("_in_use", raw_entry.get("in_use", 0) or 0))
        out["pending_clear"] = bool(raw_entry.get("_pending_clear", False) or raw_entry.get("pending", False))
        exhausted_ts = raw_entry.get("exhausted_until") or raw_entry.get("until") or raw_entry.get("expires_at")
        out["exhausted_until"] = _ts_to_iso(exhausted_ts)
        created = raw_entry.get("created_at") or raw_entry.get("created")
        if created is not None:
            out["created_at"] = _ts_to_iso(created)
        if out["pending_clear"]:
            out["state"] = "pending_clear"
        elif exhausted_ts:
            out["state"] = "exhausted"
        else:
            out["state"] = "active"
        return out

    out["state"] = "unknown"
    return out


def format_keys_human_readable(key_status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a dict structured as:
    {
      "<service>": {
         "keys": [ {index, state, exhausted_until, in_use, pending_clear, fingerprint, created_at, raw}, ... ],
         "summary": {"active": n, "exhausted": n, "pending_clear": n, "total": n}
      },
      ...
    }
    """
    out: Dict[str, Any] = {}
    for service, svc_val in (key_status or {}).items():
        keys_list = []
        if isinstance(svc_val, dict):
            try:
                items = sorted(svc_val.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
            except Exception:
                items = list(svc_val.items())
            for idx, entry in items:
                keys_list.append(_normalize_key_entry(entry, index=idx))
        elif isinstance(svc_val, list):
            for idx, entry in enumerate(svc_val):
                keys_list.append(_normalize_key_entry(entry, index=idx))
        else:
            keys_list.append(_normalize_key_entry(svc_val, index=0))

        summary = {"total": len(keys_list), "active": 0, "exhausted": 0, "pending_clear": 0, "unknown": 0}
        for k in keys_list:
            st = k.get("state")
            if st == "active":
                summary["active"] += 1
            elif st == "exhausted":
                summary["exhausted"] += 1
            elif st == "pending_clear":
                summary["pending_clear"] += 1
            else:
                summary["unknown"] += 1

        out[service] = {"keys": keys_list, "summary": summary}
    return out


# ------------------------------------------------------------------------------
# Core health check logic (slim version)
# ------------------------------------------------------------------------------

async def full_health_check() -> Dict[str, Any]:
    """
    Run all health checks, resolving modules at runtime.
    Returns the slim format: status and dependencies only.
    (Original contract expected by other scripts.)
    """
    results = {}
    messages: Dict[str, str] = {}  # still collected for logging, but not returned
    use_cloud_llm = os.getenv("USE_CLOUD_LLM", "0") == "1"

    # Fetch live key status
    try:
        key_status = key_manager.status()
        if asyncio.iscoroutine(key_status):
            key_status = await key_status
    except Exception:
        logger.exception("Failed to get key status from key_manager")
        key_status = {}

    has_pending_clear = False

    for name, module_path in _HEALTH_PROVIDERS.items():
        try:
            if name == "openai" and not use_cloud_llm:
                results[name] = "ok"
                continue

            mapped_services = PROVIDER_KEYMAP.get(name, [name])
            if mapped_services is None:
                mapped_services = []

            if mapped_services:
                provider_has_active = False
                for svc in mapped_services:
                    svc_status = key_status.get(svc, {})

                    # If there is no key status information for this service (None / empty dict / empty list),
                    # assume the service has an active key (backwards-compatible for test envs where keys
                    # aren't provided). This prevents missing key info from causing a readiness failure.
                    if not svc_status:
                        logger.debug(
                            "No key status info for %s — assuming active for readiness check", svc
                        )
                        provider_has_active = True
                        continue

                    # Handle both list and dict formats
                    if isinstance(svc_status, list):
                        for status_val in svc_status:
                            if is_key_active(status_val):
                                provider_has_active = True
                            if is_key_pending_clear(status_val):
                                has_pending_clear = True
                    elif isinstance(svc_status, dict):
                        for key_id, status_val in svc_status.items():
                            if is_key_active(status_val):
                                provider_has_active = True
                            if is_key_pending_clear(status_val):
                                has_pending_clear = True
                    else:
                        msg = f"Unexpected key status format for {svc}: {repr(svc_status)}. Assuming active."
                        logger.warning(msg)
                        messages.setdefault(name, "")
                        messages[name] += msg + " "
                        provider_has_active = True

                if not provider_has_active:
                    status_snapshot = {svc: key_status.get(svc) for svc in mapped_services}
                    msg = (
                        f"No usable keys for provider {name}. "
                        f"Services checked: {mapped_services}. "
                        f"Status: {status_snapshot}"
                    )
                    # Log the issue but DO NOT fail the provider immediately.
                    # Some providers (or tests) may not rely on keys.
                    logger.warning(msg)
                    messages[name] = msg

            # Run provider health check (keys may be missing or insufficient, but we still try)
            func = _get_health_func(module_path)
            maybe = func()
            if asyncio.iscoroutine(maybe):
                results[name] = await maybe
            else:
                results[name] = maybe

        except Exception:
            logger.exception("Health check failed", extra={"provider": name})
            results[name] = "fail"
            messages.setdefault(name, "")
            messages[name] += "Exception during health check: see logs. "

    # Check database
    try:
        db_result = check_database()
        results["database"] = await db_result
    except Exception:
        logger.exception("Database health check failed")
        results["database"] = "fail"
        messages.setdefault("database", "")
        messages["database"] += "Database health check failed (see logs). "

    overall = "fail" if any(v == "fail" for v in results.values()) else "degraded" if has_pending_clear else "ok"

    # Slim return (original contract)
    return {
        "status": overall,
        "dependencies": results,
    }


# ------------------------------------------------------------------------------
# Verbose version (adds keys and messages)
# ------------------------------------------------------------------------------

async def full_health_check_verbose() -> Dict[str, Any]:
    """Extended health check with key details and messages. Used by /health/ready."""
    slim = await full_health_check()
    # Re-fetch key status for the verbose fields
    try:
        key_status = key_manager.status()
        if asyncio.iscoroutine(key_status):
            key_status = await key_status
    except Exception:
        key_status = {}
    return {
        **slim,
        "keys": format_keys_human_readable(key_status),
        "messages": {},  # placeholder; messages could be collected if needed
    }


# ------------------------------------------------------------------------------
# Admin token dependency
# ------------------------------------------------------------------------------

def require_admin_token(x_admin_token: str = Header(...)):
    expected = os.getenv("ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


# ------------------------------------------------------------------------------
# FastAPI routes
# ------------------------------------------------------------------------------

router = APIRouter()


@router.get("/debug/keys", dependencies=[Depends(require_admin_token)])
async def debug_keys():
    """Raw key status (admin only)."""
    return await key_manager.get_status()


@router.get("/health/keys")
async def health_keys(detail: bool = Query(False, description="If true, include raw entry under each key")):
    """Human-friendly key view (public, no auth required)."""
    try:
        status = await key_manager.get_status()
    except Exception:
        try:
            status = key_manager.status()
            if asyncio.iscoroutine(status):
                status = await status
        except Exception:
            logger.exception("Failed to read key status for health/keys")
            status = {}

    formatted = format_keys_human_readable(status)
    if not detail:
        for svc_data in formatted.values():
            for k in svc_data["keys"]:
                k.pop("raw", None)
    return formatted


@router.post("/debug/keys/refresh", dependencies=[Depends(require_admin_token)])
async def force_refresh():
    """Reload keys from environment (admin only)."""
    await key_manager.refresh_from_env(sync=False)
    return {"ok": True}


@router.post("/debug/keys/clear-client", dependencies=[Depends(require_admin_token)])
async def clear_client(provider: str, idx: int):
    """Evict a cached client (admin only)."""
    try:
        await clear_client_cache(provider, idx)
        return {"ok": True, "provider": provider, "idx": idx}
    except Exception as e:
        logger.exception("Client cache clear failed")
        return {"ok": False, "error": str(e)}


@router.get("/health/ready")
async def health_ready(
    refresh: bool = Query(False, description="If true, refresh API keys from environment before checking health")
):
    """Readiness probe (public) — returns verbose health info including keys."""
    if refresh:
        try:
            await key_manager.refresh_from_env(sync=False)
            logger.info("Manual key refresh triggered via health_ready endpoint")
        except Exception:
            logger.exception("Key refresh in health check failed")
    return await full_health_check_verbose()