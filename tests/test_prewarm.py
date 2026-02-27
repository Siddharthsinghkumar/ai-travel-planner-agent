# tests/test_prewarm.py
import time
import pytest
from fastapi.testclient import TestClient
from api.app import app

@pytest.fixture(autouse=True)
def env_set(monkeypatch):
    # Ensure the env var is set for this test
    monkeypatch.setenv("PLANNER_PREWARM", "1")
    yield

def test_prewarm_invoked(monkeypatch):
    called = {"count": 0}

    async def fake_generate(*args, **kwargs):
        called["count"] += 1
        return "warmup-ok"

    # patch the ollama_client.generate used by api.app.prewarm_llm
    monkeypatch.setattr("agents.ollama_client.generate", fake_generate)

    # Starting the TestClient will run lifespan() and should call prewarm task
    with TestClient(app) as client:
        # give the background prewarm tiny time to run (it should be scheduled immediately)
        time.sleep(0.01)  # allow loop scheduling (TestClient waits for startup)
        # Now assert our fake generate was called at least once
        assert called["count"] >= 1