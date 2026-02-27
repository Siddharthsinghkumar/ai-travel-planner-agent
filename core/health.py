# core/health.py

import asyncio
from sqlalchemy import text
from typing import Dict
from agents.database import SessionLocal
from agents.cloud_llm import health_check as cloud_health
from agents.ollama_client import health_check as ollama_health
from tools.airline_api import health_check as airline_health
from tools.weather_api import health_check as weather_health


async def check_database() -> str:
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return "ok"
    except Exception:
        return "fail"

async def full_health_check() -> Dict:
    checks = {
        "openai": cloud_health(),
        "ollama": ollama_health(),
        "airline": airline_health(),
        "weather": weather_health(),
        "database": check_database(),
    }

    # Run async ones concurrently
    results = {}
    for name, check in checks.items():
        try:
            if asyncio.iscoroutine(check):
                results[name] = await check
            else:
                results[name] = check
        except Exception:
            results[name] = "fail"

    overall = "ok" if all(v == "ok" for v in results.values()) else "degraded"

    return {
        "status": overall,
        "dependencies": results
    }
