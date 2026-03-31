import pytest

import agents.planner_agent as planner_agent
from core.llm_mode import llm_routing_context


@pytest.mark.asyncio
async def test_record_llm_failure_resets_after_idle_interval(monkeypatch):
    await planner_agent.record_llm_success()
    monkeypatch.setattr(planner_agent, "LLM_FAILURE_IDLE_RESET_TIMEOUT", 1.0)

    await planner_agent.record_llm_failure(
        stage="first",
        reason="upstream_timeout",
        llm_mode="cloud_first",
        effective_mode="cloud_first",
    )
    assert planner_agent._llm_failures == 1

    planner_agent._llm_last_failure_at = (planner_agent._llm_last_failure_at or 0.0) - 5.0
    await planner_agent.record_llm_failure(
        stage="second",
        reason="upstream_timeout",
        llm_mode="cloud_first",
        effective_mode="cloud_first",
    )
    assert planner_agent._llm_failures == 1

    await planner_agent.record_llm_success()


@pytest.mark.asyncio
async def test_record_llm_failure_does_not_count_circuit_open_reason():
    await planner_agent.record_llm_success()

    await planner_agent.record_llm_failure(stage="breaker", reason="circuit_open")
    await planner_agent.record_llm_failure(stage="breaker", reason="cancelled")

    assert planner_agent._llm_failures == 0

    await planner_agent.record_llm_success()


@pytest.mark.asyncio
async def test_check_llm_circuit_is_bypassed_in_ollama_only_mode():
    await planner_agent.record_llm_success()
    planner_agent._llm_failures = planner_agent.LLM_FAILURE_THRESHOLD + 2
    planner_agent.LLM_CIRCUIT_OPEN = True

    assert (
        await planner_agent.check_llm_circuit(
            llm_mode="ollama_only",
            effective_mode="ollama_only",
        )
        is False
    )

    await planner_agent.record_llm_success()


@pytest.mark.asyncio
async def test_record_llm_failure_ollama_only_mode_is_not_counted():
    await planner_agent.record_llm_success()

    await planner_agent.record_llm_failure(
        stage="generate_explanation",
        reason="upstream_unavailable",
        llm_mode="ollama_only",
        effective_mode="ollama_only",
        backend="ollama",
    )

    assert planner_agent._llm_failures == 0

    await planner_agent.record_llm_success()


@pytest.mark.asyncio
async def test_check_llm_circuit_respects_context_override_without_explicit_mode(monkeypatch):
    await planner_agent.record_llm_success()
    planner_agent._llm_failures = planner_agent.LLM_FAILURE_THRESHOLD + 1
    planner_agent.LLM_CIRCUIT_OPEN = True
    monkeypatch.setenv("LLM_MODE", "cloud_first")

    with llm_routing_context(llm_mode="ollama_only"):
        assert await planner_agent.check_llm_circuit() is False

    await planner_agent.record_llm_success()


@pytest.mark.asyncio
async def test_record_llm_failure_without_mode_hints_respects_context_override(monkeypatch):
    await planner_agent.record_llm_success()
    monkeypatch.setenv("LLM_MODE", "cloud_first")

    with llm_routing_context(llm_mode="ollama_only"):
        await planner_agent.record_llm_failure(
            stage="generate_explanation",
            reason="upstream_timeout",
            backend="ollama",
        )

    assert planner_agent._llm_failures == 0

    await planner_agent.record_llm_success()


@pytest.mark.asyncio
async def test_check_llm_circuit_ollama_only_clears_stale_open_state():
    await planner_agent.record_llm_success()
    planner_agent._llm_failures = planner_agent.LLM_FAILURE_THRESHOLD + 3
    planner_agent.LLM_CIRCUIT_OPEN = True

    is_open = await planner_agent.check_llm_circuit(
        llm_mode="ollama_only",
        effective_mode="ollama_only",
    )

    assert is_open is False
    assert planner_agent._llm_failures == 0
    assert planner_agent.LLM_CIRCUIT_OPEN is False


@pytest.mark.asyncio
async def test_record_llm_failure_stream_no_visible_tokens_is_non_counting():
    await planner_agent.record_llm_success()

    await planner_agent.record_llm_failure(
        stage="plan_trip_stream_empty_response",
        reason="stream_no_visible_tokens",
        llm_mode="ollama_first",
        effective_mode="ollama_first",
        backend="ollama",
    )

    assert planner_agent._llm_failures == 0
