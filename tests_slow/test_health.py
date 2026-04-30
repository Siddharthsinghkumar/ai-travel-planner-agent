import asyncio
import pytest
import core.health as health_module
from core.health import full_health_check

REAL_CHECK_DATABASE = health_module.check_database


@pytest.fixture(autouse=True)
def _stub_database_probe_for_full_health(monkeypatch):
    async def _ok():
        return "ok"

    monkeypatch.setattr(health_module, "check_database", _ok)


def _healthy_key_status():
    return {
        "openai": [{"active": True, "pending_clear": False}],
        "gemini": [{"active": True, "pending_clear": False}],
        "serpapi": [{"active": True, "pending_clear": False}],
        "weather": [{"active": True, "pending_clear": False}],
    }


@pytest.mark.asyncio
async def test_health_all_ok(monkeypatch):
    """All providers healthy → overall ok."""
    async def ok():
        return "ok"

    monkeypatch.setattr(health_module, "_get_health_func", lambda path: ok)
    monkeypatch.setattr(health_module.key_manager, "status", lambda: _healthy_key_status())
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
    monkeypatch.setattr(health_module.key_manager, "status", lambda: _healthy_key_status())
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
    monkeypatch.setattr(health_module.key_manager, "status", lambda: _healthy_key_status())
    monkeypatch.setattr(health_module, "is_cloud_admin_enabled", lambda: False)
    monkeypatch.setattr(health_module, "get_llm_mode_default", lambda: "ollama_first")

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
    monkeypatch.setattr(health_module.key_manager, "status", lambda: _healthy_key_status())
    monkeypatch.setattr(health_module, "is_cloud_admin_enabled", lambda: True)
    monkeypatch.setattr(health_module, "get_llm_mode_default", lambda: "ollama_first")

    async def fake_usable():
        return []

    monkeypatch.setattr(health_module, "get_usable_providers", fake_usable)

    result = await full_health_check()
    assert result["dependencies"]["openai"] == "unavailable"
    assert result["status"] == "degraded"
    assert "openai" in result.get("dependency_summary", {}).get("unavailable", [])
    assert "dependency_unavailable" in (result.get("status_reasons") or [])


@pytest.mark.asyncio
async def test_health_runs_cloud_probe_when_enabled_and_usable(monkeypatch):
    async def ok():
        return "ok"

    monkeypatch.setattr(health_module, "_get_health_func", lambda path: ok)
    monkeypatch.setattr(health_module.key_manager, "status", lambda: _healthy_key_status())
    monkeypatch.setattr(health_module, "is_cloud_admin_enabled", lambda: True)
    monkeypatch.setattr(health_module, "get_llm_mode_default", lambda: "ollama_first")

    async def fake_usable():
        return ["gemini"]

    monkeypatch.setattr(health_module, "get_usable_providers", fake_usable)

    result = await full_health_check()
    assert result["dependencies"]["openai"] == "ok"


@pytest.mark.asyncio
async def test_health_degraded_when_dependency_reports_degraded(monkeypatch):
    async def degraded():
        return "degraded"

    async def ok():
        return "ok"

    def fake_get(path):
        if path == "tools.airline_api":
            return degraded
        return ok

    monkeypatch.setattr(health_module, "_get_health_func", fake_get)
    monkeypatch.setattr(health_module.key_manager, "status", lambda: _healthy_key_status())

    result = await full_health_check()
    assert result["status"] == "degraded"
    assert "airline" in result.get("dependency_summary", {}).get("degraded", [])
    assert "dependency_degraded" in (result.get("status_reasons") or [])


@pytest.mark.asyncio
async def test_health_degraded_when_key_status_missing(monkeypatch):
    async def ok():
        return "ok"

    monkeypatch.setattr(health_module, "_get_health_func", lambda path: ok)
    monkeypatch.setattr(health_module, "is_cloud_admin_enabled", lambda: False)
    monkeypatch.setattr(health_module.key_manager, "status", lambda: {})

    result = await full_health_check()

    assert result["status"] == "degraded"
    assert result["dependencies"]["airline"] == "unavailable"
    assert result["dependencies"]["weather"] == "unavailable"
    key_gate_issues = result.get("key_gate_issues") or []
    assert key_gate_issues
    assert any(issue.get("reason") == "missing_key_status" for issue in key_gate_issues)
    assert result.get("key_status_assumptions") == []


@pytest.mark.asyncio
async def test_health_deep_ollama_only_treats_openai_as_not_relevant(monkeypatch):
    async def ok():
        return "ok"

    monkeypatch.setattr(health_module, "_get_health_func", lambda path: ok)
    monkeypatch.setattr(health_module.key_manager, "status", lambda: _healthy_key_status())
    monkeypatch.setattr(health_module, "get_llm_mode_default", lambda: "ollama_only")
    monkeypatch.setattr(health_module, "is_cloud_admin_enabled", lambda: True)

    async def fake_usable():
        return []

    monkeypatch.setattr(health_module, "get_usable_providers", fake_usable)

    result = await full_health_check()
    assert result["status"] == "ok"
    assert result["dependencies"]["openai"] == "not_relevant"
    assert "openai" in result.get("dependency_summary", {}).get("not_relevant", [])


@pytest.mark.asyncio
async def test_check_database_returns_ok_for_successful_ping(monkeypatch):
    class _FakeSession:
        def execute(self, _stmt):
            return None

        def close(self):
            return None

    monkeypatch.setattr("agents.database.SessionLocal", lambda: _FakeSession())
    result = await REAL_CHECK_DATABASE()
    assert result == "ok"


@pytest.mark.asyncio
async def test_check_database_returns_fail_on_query_error(monkeypatch):
    class _FakeSession:
        def execute(self, _stmt):
            raise RuntimeError("db down")

        def close(self):
            return None

    monkeypatch.setattr("agents.database.SessionLocal", lambda: _FakeSession())
    result = await REAL_CHECK_DATABASE()
    assert result == "fail"
