# tests/test_prewarm.py
import asyncio
import pytest
from api.app import app

@pytest.fixture(autouse=True)
def env_set(monkeypatch):
    # Ensure the env var is set for this test
    monkeypatch.setenv("PLANNER_PREWARM", "1")
    yield

@pytest.mark.asyncio
async def test_prewarm_invoked(monkeypatch):
    called = {"count": 0}

    async def fake_generate(*args, **kwargs):
        called["count"] += 1
        return "warmup-ok"

    # patch the ollama_client.generate used by api.app.prewarm_llm
    monkeypatch.setattr("agents.ollama_client.generate", fake_generate)

    # Enter lifespan context directly and allow the background prewarm task to run.
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.01)
        assert called["count"] >= 1
