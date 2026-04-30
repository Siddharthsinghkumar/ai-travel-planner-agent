# tests/test_prewarm.py
import asyncio
import pytest
from api.app import app, prewarm_llm
from agents.ollama_client import OllamaError

@pytest.fixture(autouse=True)
def env_set(monkeypatch):
    # Ensure the env var is set for this test
    monkeypatch.setenv("PLANNER_PREWARM", "1")
    yield


@pytest.fixture(autouse=True)
def _stub_shutdown_clients(monkeypatch):
    async def _noop_async(*_args, **_kwargs):
        return None

    monkeypatch.setattr("api.app.close_client", _noop_async)
    monkeypatch.setattr("api.app.close_llm_client", _noop_async)
    monkeypatch.setattr("agents.cloud_llm.close_client", _noop_async)

@pytest.mark.asyncio
async def test_prewarm_invoked(monkeypatch):
    called = {"count": 0}

    async def fake_prewarm(*args, **kwargs):
        called["count"] += 1
        return "warmup-ok"

    # patch the ollama_client.prewarm used by api.app.prewarm_llm
    monkeypatch.setattr("agents.ollama_client.prewarm", fake_prewarm)

    # Enter lifespan context directly and allow the background prewarm task to run.
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.01)
        assert called["count"] >= 1


@pytest.mark.asyncio
async def test_prewarm_progressive_timeout_recovers_after_initial_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_PREWARM_TIMEOUT", "10")
    monkeypatch.setenv("OLLAMA_PREWARM_TIMEOUT_STEP", "5")
    monkeypatch.setenv("OLLAMA_PREWARM_RETRIES", "3")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "8")

    seen_timeouts = []
    calls = {"count": 0}

    async def fake_prewarm(*args, **kwargs):
        calls["count"] += 1
        seen_timeouts.append(kwargs.get("timeout"))
        if calls["count"] == 1:
            raise OllamaError("Request timed out after 10s")
        return "warmup-ok"

    monkeypatch.setattr("agents.ollama_client.prewarm", fake_prewarm)

    result = await prewarm_llm()

    assert result["status"] == "ok"
    assert result["attempts"] == 2
    assert seen_timeouts == [10, 15]
