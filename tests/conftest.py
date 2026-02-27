# core/conftest.py
import pytest
from fastapi.testclient import TestClient
from api.app import app
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_planner_llm_state():
    """
    Reset global LLM circuit breaker state before each test.
    """
    from agents import planner_agent

    planner_agent._llm_failures = 0
    planner_agent.LLM_CIRCUIT_OPEN = False


@pytest.fixture(autouse=True)
def disable_real_llm(monkeypatch):
    """
    Prevent real LLM calls during tests.
    """

    async def fake_generate(*args, **kwargs):
        return "fake-llm-response"

    monkeypatch.setattr("agents.planner_agent.generate", fake_generate)


# conftest.py (append or merge into your existing file)
import asyncio

async def _fake_stream_generator():
    yield "Searching flights...\n"
    await asyncio.sleep(0)
    yield "Found options...\n"
    await asyncio.sleep(0)
    for t in ["Stream", " ", "token", "."]:
        yield t

async def _fake_generate(prompt: str, system: str = "", model: str = None, stream: bool = False, **kwargs):
    if stream:
        return _fake_stream_generator()
    return "non-stream response"

@pytest.fixture(autouse=True)
def patch_llm_router(monkeypatch):
    monkeypatch.setattr("agents.llm_router.generate", _fake_generate)
    yield