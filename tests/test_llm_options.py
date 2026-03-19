import httpx
import pytest

from api.app import app


@pytest.mark.asyncio
async def test_llm_options_back_compatible_with_extended_metadata(monkeypatch):
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: True)
    monkeypatch.setattr("api.app.get_configured_cloud_providers", lambda: ["gemini", "openai"])
    monkeypatch.setattr("api.app.get_available_providers", lambda: ["gemini", "openai"])
    monkeypatch.setattr("api.app.get_default_cloud_provider", lambda: "gemini")
    monkeypatch.setattr("api.app.get_llm_mode_default", lambda: "ollama_first")

    async def fake_usable():
        return ["gemini", "openai"]

    async def fake_usability():
        return {"gemini": True, "openai": True}

    async def fake_ollama():
        return True

    monkeypatch.setattr("api.app.get_usable_providers", fake_usable)
    monkeypatch.setattr("api.app.get_provider_usability", fake_usability)
    monkeypatch.setattr("api.app._check_ollama_availability_for_options", fake_ollama)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/llm/options")

    assert resp.status_code == 200
    body = resp.json()

    # Backward-compatible keys
    assert "llm_modes" in body
    assert "cloud_providers" in body
    assert "defaults" in body

    # New metadata
    assert body["usable_cloud_providers"] == ["gemini", "openai"]
    assert body["cloud_enabled_by_config"] is True
    assert body["cloud_usable"] is True
    assert body["provider_switch_enabled"] is True
    assert body["effective_default_provider"] == "gemini"
    assert body["effective_mode"] == "ollama_first"
    assert body["backend_availability"] == {"cloud": True, "ollama": True}
    assert body["provider_status"]["gemini"]["usable"] is True
    assert body["provider_status"]["openai"]["usable"] is True


@pytest.mark.asyncio
async def test_llm_options_disables_provider_switch_when_single_provider_usable(monkeypatch):
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: True)
    monkeypatch.setattr("api.app.get_configured_cloud_providers", lambda: ["gemini", "openai"])
    monkeypatch.setattr("api.app.get_available_providers", lambda: ["gemini", "openai"])
    monkeypatch.setattr("api.app.get_default_cloud_provider", lambda: "gemini")
    monkeypatch.setattr("api.app.get_llm_mode_default", lambda: "cloud_only")

    async def fake_usable():
        return ["openai"]

    async def fake_usability():
        return {"gemini": False, "openai": True}

    async def fake_ollama():
        return True

    monkeypatch.setattr("api.app.get_usable_providers", fake_usable)
    monkeypatch.setattr("api.app.get_provider_usability", fake_usability)
    monkeypatch.setattr("api.app._check_ollama_availability_for_options", fake_ollama)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/llm/options")

    assert resp.status_code == 200
    body = resp.json()
    assert body["usable_cloud_providers"] == ["openai"]
    assert body["cloud_enabled_by_config"] is True
    assert body["provider_switch_enabled"] is False
    # Default was gemini but unusable; effective default should be usable openai.
    assert body["effective_default_provider"] == "openai"
    # Both backends are usable here, so mode remains as requested.
    assert body["effective_mode"] == "cloud_only"


@pytest.mark.asyncio
async def test_llm_options_reports_cloud_unavailable_and_effective_mode_degradation(monkeypatch):
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: True)
    monkeypatch.setattr("api.app.get_configured_cloud_providers", lambda: ["gemini", "openai"])
    monkeypatch.setattr("api.app.get_available_providers", lambda: ["gemini", "openai"])
    monkeypatch.setattr("api.app.get_default_cloud_provider", lambda: "gemini")
    monkeypatch.setattr("api.app.get_llm_mode_default", lambda: "cloud_first")

    async def fake_usable():
        return []

    async def fake_usability():
        return {"gemini": False, "openai": False}

    async def fake_ollama():
        return True

    monkeypatch.setattr("api.app.get_usable_providers", fake_usable)
    monkeypatch.setattr("api.app.get_provider_usability", fake_usability)
    monkeypatch.setattr("api.app._check_ollama_availability_for_options", fake_ollama)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/llm/options")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cloud_usable"] is False
    assert body["cloud_enabled_by_config"] is True
    assert body["provider_switch_enabled"] is False
    assert body["usable_cloud_providers"] == []
    assert body["effective_mode"] == "ollama_only"


@pytest.mark.asyncio
async def test_llm_options_reports_cloud_disabled_by_config(monkeypatch):
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: False)
    monkeypatch.setattr("api.app.get_configured_cloud_providers", lambda: ["gemini", "openai"])
    monkeypatch.setattr("api.app.get_available_providers", lambda: ["gemini", "openai"])
    monkeypatch.setattr("api.app.get_default_cloud_provider", lambda: "gemini")
    monkeypatch.setattr("api.app.get_llm_mode_default", lambda: "cloud_first")

    async def fake_usable():
        return ["gemini", "openai"]

    async def fake_usability():
        return {"gemini": True, "openai": True}

    async def fake_ollama():
        return True

    monkeypatch.setattr("api.app.get_usable_providers", fake_usable)
    monkeypatch.setattr("api.app.get_provider_usability", fake_usability)
    monkeypatch.setattr("api.app._check_ollama_availability_for_options", fake_ollama)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/llm/options")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cloud_enabled_by_config"] is False
    assert body["cloud_usable"] is False
    assert body["provider_switch_enabled"] is False
    assert body["usable_cloud_providers"] == ["gemini", "openai"]
