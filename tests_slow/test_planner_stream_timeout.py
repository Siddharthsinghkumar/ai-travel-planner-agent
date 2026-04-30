import pytest

import agents.planner_agent as planner_agent


def test_resolve_stream_init_timeout_applies_safe_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_STREAM_INIT_TIMEOUT", "5")
    assert planner_agent._resolve_stream_init_timeout() == pytest.approx(planner_agent.STREAM_INIT_TIMEOUT_FLOOR)


def test_resolve_stream_init_timeout_respects_higher_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_LLM_TIMEOUT", "30")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
    monkeypatch.delenv("LOCAL_LLM_TIMEOUT", raising=False)
    monkeypatch.setenv("PLANNER_STREAM_INIT_TIMEOUT", "35")
    assert planner_agent._resolve_stream_init_timeout() == pytest.approx(35.0)


def test_resolve_stream_total_timeout_defaults_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLANNER_STREAM_TOTAL_TIMEOUT", raising=False)
    assert planner_agent._resolve_stream_total_timeout(45.0) is None


def test_resolve_stream_total_timeout_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_STREAM_TOTAL_TIMEOUT", "120")
    assert planner_agent._resolve_stream_total_timeout(45.0) == pytest.approx(120.0)


def test_stream_init_timeout_floor_tracks_planner_timeout_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_LLM_TIMEOUT", "50")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
    monkeypatch.delenv("LOCAL_LLM_TIMEOUT", raising=False)
    assert planner_agent._stream_init_timeout_floor() == pytest.approx(50.0)
