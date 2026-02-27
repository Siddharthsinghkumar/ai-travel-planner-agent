import pytest
from core.health import full_health_check

@pytest.mark.asyncio
async def test_health_degraded(monkeypatch):

    async def fail():
        return "fail"

    monkeypatch.setattr("tools.weather_api.health_check", fail)

    result = await full_health_check()

    assert result["status"] == "degraded"
