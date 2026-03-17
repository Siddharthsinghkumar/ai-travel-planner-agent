import asyncio
import pytest
import core.health as health_module
from core.health import full_health_check

@pytest.mark.asyncio
async def test_health_all_ok(monkeypatch):
    """All providers healthy → overall ok."""
    async def ok():
        return "ok"

    monkeypatch.setattr(health_module, "_get_health_func", lambda path: ok)
    result = await full_health_check()
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_health_fail_when_provider_fails(monkeypatch):
    """A provider returning fail → overall fail."""
    async def fail():
        return "fail"

    orig_get = health_module._get_health_func

    def fake_get(path):
        if path == "tools.weather_api":
            return fail
        return orig_get(path)

    monkeypatch.setattr(health_module, "_get_health_func", fake_get)
    result = await full_health_check()
    assert result["status"] == "fail"
    assert result["dependencies"]["weather"] == "fail"


@pytest.mark.asyncio
async def test_health_degraded_when_pending_clear(monkeypatch):
    """Keys with pending_clear and all checks passing → overall degraded."""
    async def ok():
        return "ok"

    monkeypatch.setattr(health_module, "_get_health_func", lambda path: ok)

    # Must include ALL services referenced in PROVIDER_KEYMAP:
    # openai → ["openai"], airline → ["serpapi", "openai"], weather → ["weather"]
    # Give every service at least one active key.
    # Put pending_clear=True on one openai key to trigger "degraded".
    mock_status = {
        "openai": [
            {"active": True, "in_use": 0, "exhausted_until": None,
             "_pending_clear": True, "pending_exhaust": False, "fingerprint": "abc"}
        ],
        "serpapi": [
            {"active": True, "in_use": 0, "exhausted_until": None,
             "_pending_clear": False, "pending_exhaust": False, "fingerprint": "def"}
        ],
        "weather": [
            {"active": True, "in_use": 0, "exhausted_until": None,
             "_pending_clear": False, "pending_exhaust": False, "fingerprint": "ghi"}
        ],
    }
    monkeypatch.setattr(health_module.key_manager, "status", lambda: mock_status)

    result = await full_health_check()
    assert result["status"] == "degraded"