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

    async def ok():
        return "ok"

    def fake_get(path):
        if path == "tools.weather_api":
            return fail
        return ok

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


@pytest.mark.asyncio
async def test_health_marks_cloud_disabled_when_admin_flag_off(monkeypatch):
    async def ok():
        return "ok"

    monkeypatch.setattr(health_module, "_get_health_func", lambda path: ok)
    monkeypatch.setattr(health_module, "is_cloud_admin_enabled", lambda: False)

    async def fake_usable():
        return ["gemini"]

    monkeypatch.setattr(health_module, "get_usable_providers", fake_usable)

    result = await full_health_check()
    assert result["dependencies"]["openai"] == "disabled"


@pytest.mark.asyncio
async def test_health_marks_cloud_unavailable_when_enabled_without_usable_provider(monkeypatch):
    async def ok():
        return "ok"

    monkeypatch.setattr(health_module, "_get_health_func", lambda path: ok)
    monkeypatch.setattr(health_module, "is_cloud_admin_enabled", lambda: True)

    async def fake_usable():
        return []

    monkeypatch.setattr(health_module, "get_usable_providers", fake_usable)

    result = await full_health_check()
    assert result["dependencies"]["openai"] == "unavailable"


@pytest.mark.asyncio
async def test_health_runs_cloud_probe_when_enabled_and_usable(monkeypatch):
    async def ok():
        return "ok"

    monkeypatch.setattr(health_module, "_get_health_func", lambda path: ok)
    monkeypatch.setattr(health_module, "is_cloud_admin_enabled", lambda: True)

    async def fake_usable():
        return ["gemini"]

    monkeypatch.setattr(health_module, "get_usable_providers", fake_usable)

    result = await full_health_check()
    assert result["dependencies"]["openai"] == "ok"
