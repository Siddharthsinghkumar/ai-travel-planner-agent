from agents.planner_agent import _apply_prompt_hard_limit


def test_apply_prompt_hard_limit_preserves_critical_tail_anchors():
    head = "Origin: DEL\nDestination: BOM\nDeparture date: 2026-05-01\n"
    middle = ("Flight options:\n" + ("- sample-flight-line\n" * 200))
    tail = (
        "\nIMPORTANT: Only reference the exact flights listed above. "
        "Do not create or suggest any other flights.\n"
        "User's question: cheapest direct option?\n"
        "Please recommend the best flight.\n"
    )
    prompt = head + middle + tail

    trimmed, did_trim = _apply_prompt_hard_limit(prompt, hard_limit=520)

    assert did_trim is True
    assert len(trimmed) <= 520
    assert "Origin: DEL" in trimmed
    assert "IMPORTANT: Only reference the exact flights listed above." in trimmed
    assert "User's question: cheapest direct option?" in trimmed


def test_apply_prompt_hard_limit_fallback_without_known_anchors():
    prompt = "A" * 2000
    trimmed, did_trim = _apply_prompt_hard_limit(prompt, hard_limit=400)

    assert did_trim is True
    assert len(trimmed) <= 400
    assert "[...prompt trimmed for runtime stability...]" in trimmed
