# core/health.py

import asyncio
import importlib
import logging
from sqlalchemy import text
from typing import Dict
from agents.database import SessionLocal

logger = logging.getLogger(__name__)


async def check_database() -> str:
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return "ok"
    except Exception:
        return "fail"


# Map health checks to module paths (all must expose a callable named `health_check`)
_HEALTH_PROVIDERS = {
    "openai": "agents.cloud_llm",
    "ollama": "agents.ollama_client",
    "airline": "tools.airline_api",
    "weather": "tools.weather_api",
}


def _get_health_func(module_path: str):
    """Dynamically import a module and return its `health_check` function."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, "health_check")
    except Exception:
        logger.exception("Failed to import health module", extra={"module": module_path})
        # Return a dummy failing function to keep the health check running
        async def _fail():
            return "fail"
        return _fail


async def full_health_check() -> Dict:
    """
    Run all health checks, resolving modules at runtime so monkeypatching works.
    Returns:
        {
            "status": "ok" or "degraded",
            "dependencies": {
                "openai": "ok"/"fail",
                "ollama": ...,
                "database": ...,
                ...
            }
        }
    """
    results = {}

    # Check external API services
    for name, module_path in _HEALTH_PROVIDERS.items():
        func = _get_health_func(module_path)
        try:
            maybe = func()
            if asyncio.iscoroutine(maybe):
                results[name] = await maybe
            else:
                results[name] = maybe
        except Exception:
            logger.exception("Health check failed", extra={"provider": name})
            results[name] = "fail"

    # Check database separately (defined locally)
    try:
        db_result = check_database()          # returns coroutine
        results["database"] = await db_result
    except Exception:
        logger.exception("Database health check failed")
        results["database"] = "fail"

    # Determine overall status
    overall = "ok" if all(v == "ok" for v in results.values()) else "degraded"

    return {
        "status": overall,
        "dependencies": results
    }