import asyncio
import pytest
import core.health as health_module
from core.health import full_health_check

@pytest.mark.asyncio
async def test_health_degraded(monkeypatch):
    # Define a failing health check coroutine
    async def fail():
        return "fail"

    # Save the original _get_health_func
    orig_get = health_module._get_health_func

    # Replacement that returns fail for the weather module, otherwise original
    def fake_get(path):
        if path == "tools.weather_api":
            return fail
        return orig_get(path)

    # Apply the patch
    monkeypatch.setattr(health_module, "_get_health_func", fake_get)

    # Run the full health check
    result = await full_health_check()

    # Verify the overall status is degraded
    assert result["status"] == "degraded"