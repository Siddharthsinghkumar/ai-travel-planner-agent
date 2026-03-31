#tests/test_planner_logic.py
import pytest
from datetime import datetime, timedelta
from agents.planner_agent import parse_intent, filter_flights, Flight
import agents.planner_agent as planner_agent
from agents.llm_router import AllBackendsFailed
from core.iata_resolver import is_iata_token

def test_parse_intent_complex_query():
    # Test that the regex engine properly extracts complex user intents
    query = "Business flight from Delhi to Mumbai under ₹5000 direct only preferably indigo"
    intent = parse_intent(query)
    
    assert intent.origin_iata == "DEL"
    assert intent.destination_iata == "BOM"
    assert intent.price_limit == 5000
    assert intent.wants_direct is True
    assert "indigo" in intent.preferred_airlines
    assert intent.trip_type == "Business"

def test_filter_flights_logic():
    # Setup mock intent requiring a direct flight under ₹6000
    intent = parse_intent("DEL to BOM direct under 6000")
    
    # Create test flights
    perfect_flight = Flight(
        airline="IndiGo", flight_no="6E123", departure_time="10:00", 
        arrival_time="12:00", duration_min=120, price_inr=5000, 
        stops=0, layover_info=""
    )
    expensive_flight = Flight(
        airline="Vistara", flight_no="UK123", departure_time="10:00", 
        arrival_time="12:00", duration_min=120, price_inr=8000, 
        stops=0, layover_info=""
    )
    layover_flight = Flight(
        airline="Air India", flight_no="AI123", departure_time="10:00", 
        arrival_time="14:00", duration_min=240, price_inr=4500, 
        stops=1, layover_info="2h at BLR"
    )
    
    flights = [perfect_flight, expensive_flight, layover_flight]
    
    # Run the filter
    filtered, warnings = filter_flights(flights, intent)
    
    # Assertions: Should only keep the perfect flight
    assert len(filtered) == 1
    assert filtered[0].flight_no == "6E123"
    
    # If we change the intent to allow layovers, it should return 2 flights
    intent.wants_direct = False
    filtered_relaxed, _ = filter_flights(flights, intent)
    assert len(filtered_relaxed) == 2


def test_filter_flights_keeps_price_unavailable_candidates_under_budget():
    intent = parse_intent("DEL to BOM under 6000")

    unknown_price_flight = Flight(
        airline="UnknownPriceAir",
        flight_no="UP100",
        departure_time="09:00",
        arrival_time="11:10",
        duration_min=130,
        price_inr="Price unavailable",
        price_unavailable=True,
        stops=0,
        layover_info="",
    )
    expensive_flight = Flight(
        airline="ExpAir",
        flight_no="EX200",
        departure_time="12:00",
        arrival_time="14:15",
        duration_min=135,
        price_inr=9000,
        stops=0,
        layover_info="",
    )

    filtered, warnings = filter_flights([unknown_price_flight, expensive_flight], intent)

    assert any(f.flight_no == "UP100" for f in filtered)
    assert all(f.flight_no != "EX200" for f in filtered)
    assert any("unavailable prices" in w.lower() for w in warnings)


def test_parse_intent_preserves_explicit_year_date_without_bumping():
    intent = parse_intent("Business trip from MAA to DEL under ₹12000 on March 20, 2026")
    assert intent.date == "2026-03-20"


def test_parse_intent_keeps_starting_month_day_literal_date():
    intent = parse_intent("Business trip from Delhi to Mumbai for 3 days starting March 20")
    assert intent.date is not None
    parsed = datetime.strptime(intent.date, "%Y-%m-%d")
    assert parsed.month == 3
    assert parsed.day == 20


def test_parse_intent_preserves_multiword_via_city():
    intent = parse_intent("Find flights from Lucknow to Mumbai via New Delhi tomorrow")
    assert intent.stopover_city == "New Delhi"


def test_parse_intent_preserves_multiword_via_city_with_middle_east_name():
    intent = parse_intent("Find flights DEL to BOM via Abu Dhabi on 2026-03-20")
    assert intent.stopover_city == "Abu Dhabi"


def test_parse_intent_normalizes_stopover_in_phrase():
    intent = parse_intent("Find flights DEL to BOM stopover in New Delhi tomorrow")
    assert intent.stopover_city == "New Delhi"


def test_parse_intent_handles_connecting_through_phrase():
    intent = parse_intent("Find flights DEL to BOM connecting through Abu Dhabi on 2026-03-20")
    assert intent.stopover_city == "Abu Dhabi"


def test_parse_intent_round_trip_relative_dates_do_not_anchor_to_january():
    intent = parse_intent("Round-trip from Delhi to Mumbai departing tomorrow and returning 3 days later")
    expected_departure = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    assert intent.date == expected_departure
    assert intent.trip_duration_days == 3


def test_parse_intent_duration_phrase_is_not_treated_as_explicit_calendar_date():
    intent = parse_intent("Round-trip DEL to BOM returning 3 days later")
    if intent.date:
        parsed = datetime.strptime(intent.date, "%Y-%m-%d")
        assert parsed.month != 1 or parsed.day != 3
    assert intent.trip_duration_days == 3


def test_parse_intent_handles_noisy_route_phrase_without_iata_corruption():
    intent = parse_intent("Flight from Jaipur to Goa on 2025-01-10")
    assert intent.origin_iata == "JAI"
    assert intent.destination_iata in {"GOI", "GOX"}
    assert intent.date == "2025-01-10"


def test_parse_intent_noisy_oneway_phrase_still_extracts_route_and_relative_date():
    intent = parse_intent("Hey, can you quickly find me a cheap flight from Pune to Ahmedabad tomorrow, cabin only?")
    expected_departure = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    assert intent.origin_iata is not None and is_iata_token(intent.origin_iata)
    assert intent.destination_iata is not None and is_iata_token(intent.destination_iata)
    assert intent.origin_iata != intent.destination_iata
    assert intent.date == expected_departure
    assert intent.baggage_pref == "hand"
    assert isinstance(intent.route_parse_trace, dict)
    assert "raw_fragments" in intent.route_parse_trace


def test_parse_intent_ambiguous_month_day_is_conservative_and_traceable():
    intent = parse_intent("Please find flights from Pune to Jaipur on March 2")
    assert intent.date is not None
    parsed = datetime.strptime(intent.date, "%Y-%m-%d").date()
    assert parsed.month == 3 and parsed.day == 2
    assert parsed >= datetime.now().date()
    assert isinstance(intent.date_parse_trace, dict)
    assert intent.date_parse_trace.get("source") is not None


def test_parse_intent_relative_roundtrip_keeps_correct_route():
    intent = parse_intent("Return flight from Lucknow to Bengaluru leaving tomorrow and coming back after 4 days")
    assert intent.origin_iata == "LKO"
    assert intent.destination_iata == "BLR"
    assert intent.trip_duration_days == 4


def test_parse_intent_relative_roundtrip_word_number_duration_alignment():
    intent = parse_intent(
        "Round-trip from Jaipur to Goa leaving the day after tomorrow and returning four days later"
    )
    expected_departure = (datetime.now().date() + timedelta(days=2)).strftime("%Y-%m-%d")
    assert intent.origin_iata == "JAI"
    assert intent.destination_iata in {"GOI", "GOX"}
    assert intent.date == expected_departure
    assert intent.trip_duration_days == 4


def test_parse_intent_route_trace_does_not_leak_roundtrip_directive_into_origin_fragment():
    intent = parse_intent("Round-trip Jaipur to Goa leaving tomorrow and returning four days later")
    raw_fragments = (intent.route_parse_trace or {}).get("raw_fragments") or {}
    assert raw_fragments.get("origin_text") == "Jaipur"
    assert raw_fragments.get("destination_text") == "Goa"


def test_parse_intent_stopover_parsing_survives_directive_cleanup():
    intent = parse_intent(
        "Round-trip from Jaipur to Goa via New Delhi leaving tomorrow and coming back after four days"
    )
    assert intent.origin_iata == "JAI"
    assert intent.destination_iata in {"GOI", "GOX"}
    assert intent.stopover_city == "New Delhi"


def test_parse_intent_preserves_long_multiword_stopover_phrase():
    intent = parse_intent("Find flights from Kolkata to Bangkok connecting through Ho Chi Minh City next week")
    assert intent.stopover_city == "Ho Chi Minh City"


def test_parse_intent_common_city_name_does_not_jump_to_unrelated_airport_name():
    intent = parse_intent("Find flights from Pune to Kochi on April 22, 2026")
    assert intent.origin_iata == "PNQ"
    assert intent.destination_iata == "COK"


def test_parse_intent_explicit_iso_return_date_is_preserved():
    intent = parse_intent("Business round trip from Chennai to Delhi on 2026-04-15 returning 2026-04-18")
    assert intent.date == "2026-04-15"
    assert intent.return_date == "2026-04-18"


@pytest.mark.parametrize(
    "query, expected_date",
    [
        ("Need lowest fares from Pune to Ahmedabad on 2026-08-14", "2026-08-14"),
        ("Please find flights from Kochi to Jaipur on 2026-09-02 with hand baggage", "2026-09-02"),
    ],
)
def test_parse_intent_extracts_sane_route_from_noisy_city_phrase(query, expected_date):
    intent = parse_intent(query)
    assert intent.date == expected_date
    assert intent.origin_iata is not None and is_iata_token(intent.origin_iata)
    assert intent.destination_iata is not None and is_iata_token(intent.destination_iata)
    assert intent.origin_iata != intent.destination_iata


def test_parse_intent_does_not_promote_noise_plainwords_to_iata_route():
    intent = parse_intent("Need quick trip from qqq to rrr tomorrow with cheap options")
    assert intent.origin_iata is None
    assert intent.destination_iata is None


def test_parse_intent_preserves_embedded_uppercase_iata_tokens_in_noisy_phrase():
    intent = parse_intent("Find cheapest DEL to BOM on 2026-04-18")
    assert intent.origin_iata == "DEL"
    assert intent.destination_iata == "BOM"
    assert intent.date == "2026-04-18"


def test_safe_llm_error_message_is_mode_aware_for_single_backend_scope():
    err = AllBackendsFailed(
        mode="ollama_only",
        effective_mode="ollama_only",
        failures=[{"backend": "ollama", "stage": "timeout", "reason": "timeout", "error": "ollama timeout"}],
    )
    assert planner_agent._safe_llm_error_message(err) == "Configured Ollama backend temporarily unavailable"


def test_safe_llm_error_message_is_mode_aware_for_cloud_only_scope():
    err = AllBackendsFailed(
        mode="cloud_only",
        effective_mode="cloud_only",
        failures=[{"backend": "cloud", "stage": "backend_error", "reason": "provider_unreachable", "error": "unreachable"}],
    )
    assert planner_agent._safe_llm_error_message(err) == "Configured cloud backend temporarily unavailable"


def test_safe_llm_error_message_keeps_plural_for_multi_backend_scope():
    err = AllBackendsFailed(
        mode="ollama_first",
        effective_mode="ollama_first",
        failures=[
            {"backend": "ollama", "stage": "timeout", "reason": "timeout", "error": "ollama timeout"},
            {"backend": "cloud", "stage": "backend_error", "reason": "provider_unreachable", "error": "cloud unreachable"},
        ],
    )
    assert planner_agent._safe_llm_error_message(err) == "LLM backends temporarily unavailable"


def test_explanation_degradation_note_is_explicit():
    note = planner_agent._explanation_degradation_note(
        "upstream_timeout",
        "LLM explanation timed out after 50s; deterministic fallback used.",
    )
    assert note.startswith("LLM explanation degraded (upstream_timeout):")


def test_generate_deterministic_summary_marks_explanation_degradation():
    flight = Flight(
        airline="TestAir",
        flight_no="TA101",
        departure_time="10:00",
        arrival_time="12:00",
        duration_min=120,
        price_inr=5000,
        stops=0,
        layover_info="",
    )
    text = planner_agent.generate_deterministic_summary(
        flight,
        {"condition": "Sunny", "temperature_c": 30},
        "none",
        error="timed out",
        location="BOM",
    )
    assert "Explanation degraded: timed out" in text


def test_build_flight_search_profile_applies_bounded_weak_route_bump():
    intent = planner_agent.ParsedIntent(
        origin_iata="DEL",
        destination_iata="BOM",
        date="2026-06-20",
        trip_type="Business",
    )
    profile = planner_agent._build_flight_search_profile(
        intent,
        {
            "final": {
                "origin_resolution_basis": "fuzzy_city",
                "destination_resolution_basis": "exact_city",
                "origin_resolution_is_fuzzy": True,
                "destination_resolution_is_fuzzy": False,
            },
            "route_inference": {"source": "resolver_phrase_pair"},
        },
    )
    assert profile["weak_route_confidence"] is True
    assert profile["deep_search"] is False
    assert profile["is_round_trip"] is False
    assert profile["max_results"] == planner_agent.FLIGHT_SEARCH_BASE_RESULTS + planner_agent.FLIGHT_SEARCH_WEAK_ROUTE_BONUS


@pytest.mark.asyncio
async def test_plan_trip_internal_prefers_deterministic_route_recovery_before_llm(monkeypatch):
    def fake_parse_intent(_query: str):
        return planner_agent.ParsedIntent(
            origin_iata=None,
            destination_iata=None,
            date="2026-06-20",
            trip_type="Business",
        )

    async def fail_llm_correction(_query: str):
        raise AssertionError("LLM correction should not run when deterministic route inference succeeds")

    async def fake_flight_tool(**_kwargs):
        return [
            Flight(
                airline="TestAir",
                flight_no="TA001",
                departure_time="09:00",
                arrival_time="11:00",
                duration_min=120,
                price_inr=5200,
                stops=0,
                layover_info="",
            )
        ]

    async def fake_weather_tool(*, location: str, date: str):
        return {"condition": "Clear", "temperature_c": 29, "forecast_date": date, "location": location}

    monkeypatch.setattr(planner_agent, "parse_intent", fake_parse_intent)
    monkeypatch.setattr(planner_agent, "_infer_route_pair_from_query", lambda _q: ("DEL", "BOM", {"source": "resolver_phrase_pair"}))
    monkeypatch.setattr(planner_agent, "correct_cities_with_llm", fail_llm_correction)

    result = await planner_agent._plan_trip_internal(
        user_query="Delhi Mumbai tomorrow",
        trip_type="Business",
        skip_llm=True,
        flight_tool=fake_flight_tool,
        weather_tool=fake_weather_tool,
    )

    assert isinstance(result, planner_agent.PlanResult)
    assert result.best_flight["flight_no"] == "TA001"


@pytest.mark.asyncio
async def test_plan_trip_internal_forwards_deep_search_and_bounded_breadth(monkeypatch):
    calls = []

    async def fake_flight_tool(**kwargs):
        calls.append(dict(kwargs))
        return [
            Flight(
                airline="TestAir",
                flight_no="TA900",
                departure_time="06:00",
                arrival_time="08:00",
                duration_min=120,
                price_inr=4500,
                stops=0,
                layover_info="",
            )
        ]

    async def fake_weather_tool(*, location: str, date: str):
        return {"condition": "Sunny", "temperature_c": 31, "forecast_date": date, "location": location}

    result = await planner_agent._plan_trip_internal(
        user_query=(
            "Round-trip from DEL to BOM on 2026-06-20 returning on 2026-06-24 "
            "absolute cheapest possible"
        ),
        trip_type="Business",
        skip_llm=True,
        flight_tool=fake_flight_tool,
        weather_tool=fake_weather_tool,
    )

    assert isinstance(result, planner_agent.PlanResult)
    assert calls, "flight tool should be called at least once"
    assert calls[0].get("deep_search") is True
    assert calls[0].get("max_results") == planner_agent.FLIGHT_SEARCH_MAX_RESULTS_CAP


@pytest.mark.asyncio
async def test_plan_trip_internal_llm_correction_does_not_overwrite_resolved_side_with_invalid_value(monkeypatch):
    calls = []

    def fake_parse_intent(_query: str):
        return planner_agent.ParsedIntent(
            origin_iata="DEL",
            destination_iata=None,
            date="2026-06-20",
            trip_type="Business",
        )

    async def fake_llm_correction(_query: str):
        # Origin text is intentionally invalid/unresolvable while destination is recoverable.
        return "not a real airport token", "Mumbai", "Recovered route with low confidence"

    async def fake_flight_tool(**kwargs):
        calls.append(dict(kwargs))
        return [
            Flight(
                airline="TestAir",
                flight_no="TA777",
                departure_time="07:00",
                arrival_time="09:00",
                duration_min=120,
                price_inr=4700,
                stops=0,
                layover_info="",
            )
        ]

    async def fake_weather_tool(*, location: str, date: str):
        return {"condition": "Clear", "temperature_c": 30, "forecast_date": date, "location": location}

    monkeypatch.setattr(planner_agent, "parse_intent", fake_parse_intent)
    monkeypatch.setattr(planner_agent, "_infer_route_pair_from_query", lambda _q: (None, None, {"source": "none"}))
    monkeypatch.setattr(planner_agent, "correct_cities_with_llm", fake_llm_correction)

    result = await planner_agent._plan_trip_internal(
        user_query="Delhi to maybe Mumbai tomorrow",
        trip_type="Business",
        skip_llm=True,
        flight_tool=fake_flight_tool,
        weather_tool=fake_weather_tool,
    )

    assert isinstance(result, planner_agent.PlanResult)
    assert calls, "flight tool should be called"
    assert calls[0].get("departure") == "DEL"
    assert calls[0].get("arrival") == "BOM"

    normalization = (result.debug_info or {}).get("normalization") or {}
    final = normalization.get("final") or {}
    assert final.get("origin_iata") == "DEL"
    assert final.get("destination_iata") == "BOM"
