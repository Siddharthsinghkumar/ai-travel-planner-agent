import pytest

import api.app as api_app


def _patch_lifespan_dependencies(monkeypatch):
    async def _noop_async(*args, **kwargs):
        return None

    async def _no_lock():
        return "file", None

    async def _no_usable_providers():
        return []

    monkeypatch.setattr(api_app, "init_llm_client", _noop_async)
    monkeypatch.setattr(api_app.key_manager, "load_env_keys", _noop_async)
    monkeypatch.setattr(api_app.key_manager, "register_key_event_listener", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_app, "_acquire_pluggable_lock", _no_lock)
    monkeypatch.setattr(api_app, "get_usable_providers", _no_usable_providers)


@pytest.mark.asyncio
async def test_lifespan_logs_deprecation_warnings_when_legacy_vars_present(monkeypatch, capsys):
    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "legacy-anthropic-key")
    monkeypatch.setenv("CLOUD_BASE_URL", "https://legacy.example.com")
    monkeypatch.setenv("LLM_PRIORITY", "local-first")

    async with api_app.app.router.lifespan_context(api_app.app):
        assert api_app.app.state.startup_complete is True

    stdout = capsys.readouterr().out
    messages = [line for line in stdout.splitlines() if "Config deprecation:" in line]
    for var_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CLOUD_BASE_URL", "LLM_PRIORITY"):
        assert any("Config deprecation:" in message and var_name in message for message in messages)


@pytest.mark.asyncio
async def test_lifespan_has_no_deprecation_warning_when_vars_absent(monkeypatch, capsys):
    _patch_lifespan_dependencies(monkeypatch)
    for var_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CLOUD_BASE_URL", "LLM_PRIORITY"):
        monkeypatch.delenv(var_name, raising=False)

    async with api_app.app.router.lifespan_context(api_app.app):
        assert api_app.app.state.startup_complete is True

    stdout = capsys.readouterr().out
    assert "Config deprecation:" not in stdout


@pytest.mark.asyncio
async def test_lifespan_logs_startup_config_summary(monkeypatch, capsys):
    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setenv("LLM_MODE", "cloud_first")
    monkeypatch.setenv("CLOUD_PROVIDER_CHAIN", "gemini,openai")
    monkeypatch.setenv("CLOUD_PROVIDER", "gemini")
    monkeypatch.setenv("USE_CLOUD_LLM", "1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "openhermes")

    async with api_app.app.router.lifespan_context(api_app.app):
        assert api_app.app.state.startup_complete is True

    stdout = capsys.readouterr().out
    assert "Startup config summary | llm_mode=cloud_first" in stdout
    assert "cloud_provider_chain=gemini,openai" in stdout
    assert "cloud_enabled=True" in stdout
