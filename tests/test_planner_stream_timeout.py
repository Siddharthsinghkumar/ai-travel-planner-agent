import pytest

import agents.planner_agent as planner_agent


def test_resolve_stream_init_timeout_applies_safe_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_STREAM_INIT_TIMEOUT", "5")
    assert planner_agent._resolve_stream_init_timeout() == pytest.approx(planner_agent.STREAM_INIT_TIMEOUT_FLOOR)


def test_resolve_stream_init_timeout_respects_higher_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_STREAM_INIT_TIMEOUT", "35")
    assert planner_agent._resolve_stream_init_timeout() == pytest.approx(35.0)

