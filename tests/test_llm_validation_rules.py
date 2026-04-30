from validation.llm_validation_rules import (
    detect_layover_contradiction,
    detect_relaxed_preferred_airline_contradiction,
)


def test_layover_nonstop_awkward_duration_wording_is_not_contradiction():
    llm_text = (
        "This is a non-stop flight with no layover. "
        "The flight duration of 120 minutes is perfect for your requirement of a layover less than 2 hours."
    )
    best_flight = {"stops": 0, "layover_durations_min": [], "duration_min": 120}
    reason = detect_layover_contradiction(
        llm_text=llm_text,
        best_flight=best_flight,
        layover_limit_minutes=120,
    )
    assert reason is None


def test_layover_nonstop_explicit_layover_claim_fails():
    llm_text = "Best option is TA123. It has a layover of 90 minutes."
    best_flight = {"stops": 0, "layover_durations_min": [], "duration_min": 120}
    reason = detect_layover_contradiction(
        llm_text=llm_text,
        best_flight=best_flight,
        layover_limit_minutes=120,
    )
    assert reason == "LLM implies non-stop flight has an actual layover"


def test_layover_structured_violation_with_compliance_claim_fails():
    llm_text = "This option meets your max layover requirement and stays within your layover limit."
    best_flight = {"stops": 1, "layover_durations_min": [150], "duration_min": 300}
    reason = detect_layover_contradiction(
        llm_text=llm_text,
        best_flight=best_flight,
        layover_limit_minutes=120,
    )
    assert reason == (
        "LLM claims layover-limit compliance but structured layover durations exceed the limit"
    )


def test_relaxed_preferred_airline_disclosure_not_contradiction():
    llm_text = (
        "No flights were found for your preferred airline Indigo. "
        "Here is the closest alternative: TestAir TA123."
    )
    reason = detect_relaxed_preferred_airline_contradiction(
        llm_text=llm_text,
        preferred_airlines=["indigo"],
        selected_airline="TestAir",
    )
    assert reason is None


def test_relaxed_preferred_airline_false_satisfaction_fails():
    llm_text = "Your best recommended Indigo flight is TA123 for this route."
    reason = detect_relaxed_preferred_airline_contradiction(
        llm_text=llm_text,
        preferred_airlines=["indigo"],
        selected_airline="TestAir",
    )
    assert reason is not None

