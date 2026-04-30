import asyncio
import os

import pytest
from fastapi.testclient import TestClient
from api.app import app
from api import routes_booking_tracking
from core import job_queue
from core.rate_limiter import SlidingWindowRateLimiter


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


@pytest.fixture(autouse=True)
def set_test_auth_tokens(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLE", "true")
    monkeypatch.setenv(
        "AUTH_BEARER_TOKENS",
        "test-user:test-user-token,other-user:other-user-token",
    )
    monkeypatch.setenv("ADMIN_TOKEN", "admin-test-token")


@pytest.fixture(autouse=True)
def reset_ask_rate_limiter_state():
    app.state.ask_rate_limiter = SlidingWindowRateLimiter(max_keys=5000)
    yield
    app.state.ask_rate_limiter = SlidingWindowRateLimiter(max_keys=5000)


@pytest.fixture(autouse=True)
def reset_admin_rate_limiter_state():
    app.state.admin_rate_limiter = SlidingWindowRateLimiter(max_keys=5000)
    yield
    app.state.admin_rate_limiter = SlidingWindowRateLimiter(max_keys=5000)


@pytest.fixture(autouse=True)
def reset_job_queue_state():
    # Ensure the shared DB tables exist (in-memory SQLite for tests)
    from agents.database import init_db
    init_db()

    job_queue._jobs.clear()
    job_queue._job_event_queues.clear()
    job_queue._job_tasks.clear()
    job_queue._queue = asyncio.Queue(maxsize=job_queue.JOB_QUEUE_MAXSIZE)
    job_queue._last_prune_at = 0.0
    job_queue.JOB_RETENTION_SECONDS = 3600
    job_queue.JOB_PRUNE_INTERVAL_SECONDS = 300
    job_queue._async_state_lock = None
    yield
    job_queue._jobs.clear()
    job_queue._job_event_queues.clear()
    job_queue._job_tasks.clear()
    job_queue._queue = asyncio.Queue(maxsize=job_queue.JOB_QUEUE_MAXSIZE)
    job_queue._last_prune_at = 0.0
    job_queue.JOB_RETENTION_SECONDS = 3600
    job_queue.JOB_PRUNE_INTERVAL_SECONDS = 300
    job_queue._async_state_lock = None


@pytest.fixture(autouse=True)
def reset_booking_idempotency_state():
    routes_booking_tracking._reset_booking_idempotency_state_for_tests()
    yield
    routes_booking_tracking._reset_booking_idempotency_state_for_tests()
