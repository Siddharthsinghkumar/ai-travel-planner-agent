import asyncio
import time
from types import SimpleNamespace

import pytest

from api.app import app
import api.app as api_app


@pytest.mark.asyncio
async def test_lifespan_tolerates_legacy_llm_client_init_failure(monkeypatch):
    async def fail_init():
        raise ValueError("OPENAI_API_KEY missing")

    monkeypatch.setenv("ENABLE_LEGACY_ASYNC_LLM_CLIENT", "1")
    monkeypatch.setattr("api.app.init_llm_client", fail_init)

    async with app.router.lifespan_context(app):
        assert app.state.startup_complete is True


@pytest.mark.asyncio
async def test_lifespan_skips_legacy_llm_client_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_LEGACY_ASYNC_LLM_CLIENT", "0")

    async def should_not_run():
        raise AssertionError("legacy init should be skipped when disabled")

    monkeypatch.setattr("api.app.init_llm_client", should_not_run)

    async with app.router.lifespan_context(app):
        assert app.state.startup_complete is True


def test_compute_async_job_support_exposes_contract_flags(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("ASYNC_JOB_REQUIRE_SINGLE_WORKER", "1")
    monkeypatch.delenv("ALLOW_UNSAFE_ASYNC_JOBS", raising=False)
    monkeypatch.setenv("ASYNC_JOB_FAIL_FAST_ON_UNSUPPORTED_TOPOLOGY", "1")

    support = api_app._compute_async_job_support()

    assert support["enabled"] is False
    assert support["reason"] == "unsupported_multi_worker_topology"
    assert support["contract"] == "single_worker_required_process_local_queue"
    assert support["guard_active"] is True
    assert support["allow_unsafe_override"] is False
    assert support["fail_fast_on_unsupported_topology"] is True


def test_fail_fast_does_not_trigger_when_unsafe_override_enables_async_jobs(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("ASYNC_JOB_REQUIRE_SINGLE_WORKER", "1")
    monkeypatch.setenv("ALLOW_UNSAFE_ASYNC_JOBS", "1")
    monkeypatch.setenv("ASYNC_JOB_FAIL_FAST_ON_UNSUPPORTED_TOPOLOGY", "1")

    support = api_app._compute_async_job_support()

    assert support["enabled"] is True
    assert support["reason"] == "unsafe_override_enabled"


@pytest.mark.asyncio
async def test_lifespan_does_not_crash_for_unsupported_async_topology_even_when_fail_fast_configured(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("ASYNC_JOB_REQUIRE_SINGLE_WORKER", "1")
    monkeypatch.delenv("ALLOW_UNSAFE_ASYNC_JOBS", raising=False)
    monkeypatch.setenv("ASYNC_JOB_FAIL_FAST_ON_UNSUPPORTED_TOPOLOGY", "1")
    monkeypatch.setenv("USE_CLOUD_LLM", "0")
    monkeypatch.setattr(app.state, "async_job_support", None, raising=False)

    async with app.router.lifespan_context(app):
        assert app.state.startup_complete is True
        support = app.state.async_job_support
        assert support["enabled"] is False
        assert support["reason"] == "unsupported_multi_worker_topology"
        assert support["fail_fast_on_unsupported_topology"] is True


@pytest.mark.asyncio
async def test_lifespan_does_not_block_on_background_serpapi_reconcile(monkeypatch):
    async def _fake_lock():
        return "file", None

    async def _fake_load_env_keys():
        return None

    async def _fake_noop_async(*_args, **_kwargs):
        return None

    async def _slow_reconcile_once():
        await asyncio.sleep(0.35)
        return {"checked": 1, "errors": 0}

    reconcile_task_holder = {"task": None}

    def _fake_start_serpapi_reconcile_loop(interval_seconds: int = 0):
        reconcile_task_holder["task"] = asyncio.create_task(_slow_reconcile_once())
        api_app.key_manager._serpapi_reconcile_task = reconcile_task_holder["task"]

    def _fake_stop_serpapi_reconcile_loop():
        task = reconcile_task_holder.get("task")
        if task and not task.done():
            task.cancel()
        api_app.key_manager._serpapi_reconcile_task = None

    monkeypatch.setenv("RUN_KEY_REFRESH", "1")
    monkeypatch.setenv("USE_CLOUD_LLM", "0")
    monkeypatch.setattr(api_app, "_acquire_pluggable_lock", _fake_lock)
    monkeypatch.setattr(api_app, "init_db", lambda: None)
    monkeypatch.setattr(api_app, "_should_emit_startup_summary", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(api_app.key_manager, "load_env_keys", _fake_load_env_keys)
    monkeypatch.setattr(api_app.key_manager, "start_refresh_loop", lambda **_kwargs: None)
    monkeypatch.setattr(api_app.key_manager, "stop_refresh_loop", lambda: None)
    monkeypatch.setattr(api_app.key_manager, "start_serpapi_reconcile_loop", _fake_start_serpapi_reconcile_loop)
    monkeypatch.setattr(api_app.key_manager, "stop_serpapi_reconcile_loop", _fake_stop_serpapi_reconcile_loop)
    monkeypatch.setattr(api_app.key_manager, "register_key_event_listener", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("api.app.refresh_provider_chain_from_env", lambda **_kwargs: None)
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: False)
    monkeypatch.setattr("api.app.close_client", _fake_noop_async)
    monkeypatch.setattr("api.app.close_llm_client", _fake_noop_async)
    monkeypatch.setattr("agents.cloud_llm.close_client", _fake_noop_async)

    started = time.monotonic()
    async with app.router.lifespan_context(app):
        startup_elapsed = time.monotonic() - started
        assert app.state.startup_complete is True
        assert startup_elapsed < 0.3
        await asyncio.sleep(0.01)
        assert reconcile_task_holder["task"] is not None


@pytest.mark.asyncio
async def test_redis_lock_lease_keeper_stops_refresh_on_owner_loss(monkeypatch):
    class _FakeLock:
        def __init__(self):
            self.calls = 0

        async def extend(self, *_args, **_kwargs):
            self.calls += 1
            return False

    fake_app = SimpleNamespace(state=SimpleNamespace(
        key_manager_refresh_owner=True,
        key_manager_task=None,
    ))
    stop_calls = {"n": 0}
    monkeypatch.setattr(api_app.key_manager, "stop_refresh_loop", lambda: stop_calls.__setitem__("n", stop_calls["n"] + 1))
    monkeypatch.setattr(api_app, "_redis_lease_interval_seconds", lambda _ttl: 0.01)

    await asyncio.wait_for(
        api_app._run_redis_lock_lease_keeper(fake_app, _FakeLock(), ttl_seconds=60),
        timeout=0.3,
    )

    assert fake_app.state.key_manager_refresh_owner is False
    assert stop_calls["n"] == 1


@pytest.mark.asyncio
async def test_redis_lock_lease_keeper_keeps_owner_when_renewals_succeed(monkeypatch):
    class _FakeLock:
        def __init__(self):
            self.calls = 0

        async def extend(self, *_args, **_kwargs):
            self.calls += 1
            return True

    fake_app = SimpleNamespace(state=SimpleNamespace(
        key_manager_refresh_owner=True,
        key_manager_task=None,
    ))
    stop_calls = {"n": 0}
    monkeypatch.setattr(api_app.key_manager, "stop_refresh_loop", lambda: stop_calls.__setitem__("n", stop_calls["n"] + 1))
    monkeypatch.setattr(api_app, "_redis_lease_interval_seconds", lambda _ttl: 0.01)
    lock = _FakeLock()

    task = asyncio.create_task(api_app._run_redis_lock_lease_keeper(fake_app, lock, ttl_seconds=60))
    await asyncio.sleep(0.04)
    fake_app.state.key_manager_refresh_owner = False
    await asyncio.wait_for(task, timeout=0.3)

    assert lock.calls >= 1
    assert stop_calls["n"] == 0


def test_should_run_prewarm_on_single_or_undeclared_topology_by_default(monkeypatch):
    monkeypatch.delenv("PLANNER_PREWARM_ALL_WORKERS", raising=False)
    monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("GUNICORN_WORKERS", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)
    assert api_app._should_run_prewarm(True, True) is True
    assert api_app._should_run_prewarm(True, False) is True
    assert api_app._should_run_prewarm(False, True) is False
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    assert api_app._should_run_prewarm(True, False) is False


def test_should_run_prewarm_on_all_workers_when_enabled(monkeypatch):
    monkeypatch.setenv("PLANNER_PREWARM_ALL_WORKERS", "1")
    assert api_app._should_run_prewarm(True, False) is True


def test_startup_log_level_for_worker_follower_is_debug_in_multi_worker(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    assert api_app._startup_log_level_for_worker(refresh_owner=False) == "debug"
    assert api_app._startup_log_level_for_worker(refresh_owner=True) == "info"


def test_startup_log_level_for_worker_single_worker_is_info(monkeypatch):
    monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("GUNICORN_WORKERS", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)
    assert api_app._startup_log_level_for_worker(refresh_owner=False) == "info"


def test_worker_runtime_role_for_refresh_owner(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    assert api_app._worker_runtime_role(refresh_owner=True) == "refresh_owner"


def test_worker_runtime_role_for_follower_in_multi_worker(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    assert api_app._worker_runtime_role(refresh_owner=False) == "follower"


def test_worker_runtime_role_for_single_or_undeclared(monkeypatch):
    monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("GUNICORN_WORKERS", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)
    assert api_app._worker_runtime_role(refresh_owner=False) == "single_or_undeclared"


def test_should_emit_startup_summary_only_for_owner_in_multi_worker(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    assert api_app._should_emit_startup_summary(refresh_owner=True) is True
    assert api_app._should_emit_startup_summary(refresh_owner=False) is False


def test_should_emit_startup_summary_for_single_worker(monkeypatch):
    monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("GUNICORN_WORKERS", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)
    assert api_app._should_emit_startup_summary(refresh_owner=False) is True


def test_cloud_startup_relevance_helper(monkeypatch):
    assert api_app._is_cloud_startup_relevant_for_mode("ollama_only") is False
    assert api_app._is_cloud_startup_relevant_for_mode("ollama_first") is True
