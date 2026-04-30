# core/health.py

import asyncio
import importlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Callable, Awaitable

# Local imports (adjust if needed)
from core.api_key_manager import key_manager
from core.env_config import get_env_float
from agents.cloud_llm import is_cloud_admin_enabled, get_usable_providers
from core.llm_mode import get_llm_mode_default, LLM_MODE_OLLAMA_ONLY

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
    # "openai" health dependency points to agents.cloud_llm (shared cloud path).
    # Treat either OpenAI or Gemini usable keys as satisfying cloud-key availability.
    "openai": ["openai", "gemini"],
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
    Run a lightweight DB readiness probe.
    Returns: ok | degraded | unavailable | fail
    """
    timeout_seconds = max(0.2, float(get_env_float("HEALTH_DB_TIMEOUT_SECONDS", 1.0)))
    try:
        from agents.database import SessionLocal
        from sqlalchemy import text
    except Exception:
        return "unavailable"

    def _ping() -> None:
        session = SessionLocal()
        try:
            session.execute(text("SELECT 1"))
        finally:
            session.close()

    try:
        await asyncio.wait_for(asyncio.to_thread(_ping), timeout=timeout_seconds)
        return "ok"
    except asyncio.TimeoutError:
        logger.warning("health_db_check_timeout", extra={"timeout_seconds": timeout_seconds})
        return "degraded"
    except Exception:
        logger.exception("health_db_check_failed")
        return "fail"


def is_key_active(entry) -> bool:
    """Return True if the key entry indicates an active usable key."""
    if isinstance(entry, str):
        # Simple string status: "active" is active, anything else (like "exhausted ...") is not
        return entry == "active"

    if isinstance(entry, dict):
        pending_clear = bool(entry.get("pending_clear", False) or entry.get("_pending_clear", False))
        if pending_clear:
            return False

        if "active" in entry:
            return bool(entry.get("active"))

        exhausted_until = entry.get("exhausted_until")
        if exhausted_until in (None, ""):
            return True

        # Numeric timestamp support.
        try:
            if isinstance(exhausted_until, (int, float)):
                return float(exhausted_until) <= datetime.now(timezone.utc).timestamp()
        except Exception:
            return False

        # ISO timestamp support.
        try:
            dt = datetime.fromisoformat(str(exhausted_until))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp() <= datetime.now(timezone.utc).timestamp()
        except Exception:
            return False

    return False


def is_key_pending_clear(entry) -> bool:
    """Return True if the key entry indicates it is pending clear."""
    if isinstance(entry, dict):
        return bool(entry.get("pending_clear", False) or entry.get("_pending_clear", False))
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
    llm_mode = (get_llm_mode_default() or "").strip().lower()
    mode_non_relevant_dependencies = {"openai"} if llm_mode == LLM_MODE_OLLAMA_ONLY else set()
    cloud_admin_enabled = is_cloud_admin_enabled()
    cloud_usable_providers = []
    if cloud_admin_enabled:
        try:
            cloud_usable_providers = await get_usable_providers()
        except Exception:
            logger.exception("Failed to fetch cloud provider usability during health check")
            cloud_usable_providers = []
    cloud_usable = bool(cloud_usable_providers)

    # Fetch live key status
    try:
        key_status = key_manager.status()
        if asyncio.iscoroutine(key_status):
            key_status = await key_status
    except Exception:
        logger.exception("Failed to get key status from key_manager")
        key_status = {}

    has_pending_clear = False
    key_gate_issues: list[Dict[str, Any]] = []
    key_status_assumptions: list[Dict[str, Any]] = []

    for name, module_path in _HEALTH_PROVIDERS.items():
        try:
            if name in mode_non_relevant_dependencies:
                results[name] = "not_relevant"
                messages[name] = f"Dependency {name} is not required for llm_mode={llm_mode}."
                continue

            if name == "openai" and not cloud_admin_enabled:
                results[name] = "disabled"
                messages[name] = "Cloud LLM disabled by configuration (USE_CLOUD_LLM=0)."
                continue

            if name == "openai" and not cloud_usable:
                results[name] = "unavailable"
                messages[name] = "Cloud LLM enabled but no usable cloud provider keys are currently active."
                continue

            mapped_services = PROVIDER_KEYMAP.get(name, [name])
            if mapped_services is None:
                mapped_services = []

            if mapped_services:
                provider_has_active = False
                missing_key_status_services: list[str] = []
                unknown_format_services: list[str] = []
                for svc in mapped_services:
                    svc_status = key_status.get(svc, {})

                    # Missing key-status for a mapped service is treated as unknown/unavailable.
                    if not svc_status:
                        missing_key_status_services.append(svc)
                        logger.debug("No key status info for %s (provider=%s)", svc, name)
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
                        unknown_format_services.append(svc)
                        msg = f"Unexpected key status format for {svc}: {repr(svc_status)}."
                        logger.warning(msg)
                        messages.setdefault(name, "")
                        messages[name] += msg + " "

                if not provider_has_active:
                    status_snapshot = {svc: key_status.get(svc) for svc in mapped_services}
                    reason = (
                        "missing_key_status"
                        if missing_key_status_services
                        else "unknown_key_status_format"
                        if unknown_format_services
                        else "no_usable_keys"
                    )
                    msg = (
                        f"Provider {name} unavailable due to key state ({reason}). "
                        f"Services checked: {mapped_services}. "
                        f"Status: {status_snapshot}"
                    )
                    logger.warning(msg)
                    messages[name] = msg
                    key_gate_issues.append(
                        {
                            "provider": name,
                            "services_checked": mapped_services,
                            "status_snapshot": status_snapshot,
                            "reason": reason,
                        }
                    )
                    results[name] = "unavailable"
                    continue

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

    failed_dependencies = sorted([name for name, value in results.items() if value == "fail"])
    degraded_dependencies = sorted([name for name, value in results.items() if value == "degraded"])
    unavailable_dependencies = sorted([name for name, value in results.items() if value == "unavailable"])
    disabled_dependencies = sorted([name for name, value in results.items() if value == "disabled"])
    not_relevant_dependencies = sorted([name for name, value in results.items() if value == "not_relevant"])

    dependency_degraded = bool(degraded_dependencies or unavailable_dependencies)
    keying_degraded = bool(has_pending_clear or key_gate_issues or key_status_assumptions)

    overall = (
        "fail"
        if failed_dependencies
        else "degraded"
        if dependency_degraded or keying_degraded
        else "ok"
    )

    status_reasons: list[str] = []
    if failed_dependencies:
        status_reasons.append("dependency_failed")
    if degraded_dependencies:
        status_reasons.append("dependency_degraded")
    if unavailable_dependencies:
        status_reasons.append("dependency_unavailable")
    if has_pending_clear:
        status_reasons.append("key_pending_clear")
    if key_gate_issues:
        status_reasons.append("provider_key_gate_issue")
    if key_status_assumptions:
        status_reasons.append("key_status_assumed")

    # Slim return (original contract)
    return {
        "status": overall,
        "dependencies": results,
        "dependency_summary": {
            "failed": failed_dependencies,
            "degraded": degraded_dependencies,
            "unavailable": unavailable_dependencies,
            "disabled": disabled_dependencies,
            "not_relevant": not_relevant_dependencies,
        },
        "status_reasons": status_reasons,
        "key_gate_issues": key_gate_issues,
        "key_status_assumptions": key_status_assumptions,
        "messages": messages,
    }


# ------------------------------------------------------------------------------
# Verbose version (adds keys and messages)
# ------------------------------------------------------------------------------

async def full_health_check_verbose() -> Dict[str, Any]:
    """Extended health check with key details and messages (for diagnostics)."""
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
        "messages": slim.get("messages", {}),
    }
