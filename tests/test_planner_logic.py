#tests/test_planner_logic.py
import asyncio
import pytest
import logging
from datetime import datetime, timedelta
from agents.planner_agent import parse_intent, filter_flights, Flight, _ensure_route_grounding
import agents.planner_agent as planner_agent
from agents.llm_router import AllBackendsFailed
from core.iata_resolver import is_iata_token


@pytest.mark.asyncio
async def test_action_intent_confirm_booking_is_not_a_supported_action():
    assert planner_agent._detect_booking_or_tracking_action("confirm booking 42") is None

    result = await planner_agent.plan_trip(user_query="confirm booking 42", stream=False)
    assert isinstance(result, dict)
    assert "action" not in result
    assert result.get("failure_reason") in {"invalid_route", "planner_error"}


@pytest.mark.asyncio
async def test_action_intent_track_price_surfaces_snapshot_setup_failure(monkeypatch):
    async def fake_plan_trip_internal(**_kwargs):
        return planner_agent.PlanResult(
            llm_response="ok",
            best_flight={
                "airline": "TestAir",
                "flight_no": "TA500",
                "departure_time": "09:00",
                "arrival_time": "11:00",
                "duration_min": 120,
                "price_inr": 5200,
                "stops": 0,
                "baggage": "7kg cabin",
            },
            weather={"condition": "Clear", "temperature_c": 29},
            search_date="2026-06-12",
            debug_info={"intent": {"origin_iata": "DEL", "destination_iata": "BOM"}},
        )

    async def fake_hold_booking_safe(**_kwargs):
        return {"id": 42, "status": "HELD"}

    def fake_record_snapshot_fail(*_args, **_kwargs):
        raise RuntimeError("snapshot_persist_failed")

    def fake_cancel_booking_safe(_booking_id: int):
        return True

    monkeypatch.setattr(planner_agent, "_plan_trip_internal", fake_plan_trip_internal)
    monkeypatch.setattr(planner_agent, "_hold_booking_safe", fake_hold_booking_safe)
    monkeypatch.setattr(planner_agent, "_record_price_snapshot_safe", fake_record_snapshot_fail)
    monkeypatch.setattr(planner_agent, "_cancel_booking_safe", fake_cancel_booking_safe)

    result = await planner_agent.plan_trip(
        user_query="track price for this flight",
        stream=False,
    )

    assert result["action"] == "track_price"
    assert result["success"] is False
    assert result["error"] == "price_tracking_setup_failed"
    assert result["reason"] == "snapshot_persist_failed"
    assert result["cleanup_cancelled"] is True


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


def test_rank_flights_default_preserves_source_order():
    intent = planner_agent.ParsedIntent(flight_pref="default")
    first = Flight(
        airline="A1",
        flight_no="A100",
        departure_time="08:00",
        arrival_time="10:00",
        duration_min=120,
        price_inr=5000,
        stops=0,
        layover_info="",
        baggage="15kg check-in",
        airline_logo="https://img.example/a1.png",
        marketed_as=["A100"],
        itinerary_type="One way",
    )
    second = Flight(
        airline="A2",
        flight_no="A200",
        departure_time="08:00",
        arrival_time="10:00",
        duration_min=120,
        price_inr=5000,
        stops=0,
        layover_info="",
        baggage="15kg check-in",
        airline_logo="https://img.example/a2.png",
        marketed_as=["A200", "Codeshare"],
        itinerary_type="Round trip candidate",
    )

    ranked = planner_agent.rank_flights([first, second], intent)
    assert [f.flight_no for f in ranked] == ["A100", "A200"]


def test_rank_flights_cheapest_sorts_by_price_then_source_order():
    intent = planner_agent.ParsedIntent(flight_pref="cheapest")
    first = Flight(
        airline="A1",
        flight_no="A100",
        departure_time="08:00",
        arrival_time="10:00",
        duration_min=120,
        price_inr=5000,
        stops=0,
        layover_info="",
    )
    second = Flight(
        airline="A2",
        flight_no="A200",
        departure_time="08:10",
        arrival_time="10:10",
        duration_min=120,
        price_inr=4500,
        stops=0,
        layover_info="",
    )
    third_same_price = Flight(
        airline="A3",
        flight_no="A300",
        departure_time="08:20",
        arrival_time="10:20",
        duration_min=120,
        price_inr=4500,
        stops=0,
        layover_info="",
    )

    ranked = planner_agent.rank_flights([first, second, third_same_price], intent)
    assert [f.flight_no for f in ranked] == ["A200", "A300", "A100"]


def test_rank_flights_shortest_sorts_by_duration_then_source_order():
    intent = planner_agent.ParsedIntent(flight_pref="shortest")
    first = Flight(
        airline="A1",
        flight_no="A100",
        departure_time="08:00",
        arrival_time="10:20",
        duration_min=140,
        price_inr=5000,
        stops=0,
        layover_info="",
    )
    second = Flight(
        airline="A2",
        flight_no="A200",
        departure_time="08:10",
        arrival_time="10:00",
        duration_min=110,
        price_inr=5200,
        stops=0,
        layover_info="",
    )
    third_same_duration = Flight(
        airline="A3",
        flight_no="A300",
        departure_time="08:30",
        arrival_time="10:20",
        duration_min=110,
        price_inr=5300,
        stops=0,
        layover_info="",
    )

    ranked = planner_agent.rank_flights([first, second, third_same_duration], intent)
    assert [f.flight_no for f in ranked] == ["A200", "A300", "A100"]


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


def test_parse_intent_nonstop_maps_to_direct_filter():
    intent = parse_intent("Find nonstop flights from Delhi to Mumbai tomorrow")
    assert intent.wants_direct is True


def test_parse_intent_least_travel_time_maps_to_shortest_pref():
    intent = parse_intent("Find flights from Mumbai to Bengaluru with least travel time on 2026-05-15")
    assert intent.flight_pref == "shortest"


def test_parse_intent_detects_explicit_cabin_preference():
    intent = parse_intent("Need business class flight from Delhi to Mumbai on 2026-04-20")
    assert intent.cabin_pref == "business"


def test_filter_flights_explicit_cabin_pref_preserves_source_order_within_subset():
    intent = planner_agent.ParsedIntent(cabin_pref="business")
    flights = [
        Flight(
            airline="A1",
            flight_no="A100",
            departure_time="08:00",
            arrival_time="10:00",
            duration_min=120,
            price_inr=5000,
            stops=0,
            layover_info="",
            travel_class="Business",
        ),
        Flight(
            airline="A2",
            flight_no="A200",
            departure_time="08:20",
            arrival_time="10:20",
            duration_min=120,
            price_inr=5200,
            stops=0,
            layover_info="",
            travel_class="Economy",
        ),
        Flight(
            airline="A3",
            flight_no="A300",
            departure_time="08:40",
            arrival_time="10:40",
            duration_min=120,
            price_inr=5400,
            stops=0,
            layover_info="",
            travel_class="Business Saver",
        ),
    ]

    filtered, _ = filter_flights(flights, intent)
    assert [f.flight_no for f in filtered] == ["A100", "A300"]


def test_parse_intent_preserves_explicit_year_date_without_bumping():
    intent = parse_intent("Business trip from MAA to DEL under ₹12000 on March 20, 2026")
    assert intent.date == "2026-03-20"


def test_parse_intent_keeps_starting_month_day_literal_date():
    intent = parse_intent("Business trip from Delhi to Mumbai for 3 days starting March 20")
    assert intent.date is not None
    parsed = datetime.strptime(intent.date, "%Y-%m-%d")
    assert parsed.month == 3
    assert parsed.day == 20


def test_parse_intent_relative_weeks_supports_after_prefix():
    intent = parse_intent("Flight DEL BOM after 2 weeks")
    expected_departure = (datetime.now().date() + timedelta(days=14)).strftime("%Y-%m-%d")
    assert intent.date == expected_departure


def test_parse_intent_layover_constraint_does_not_create_price_limit():
    intent = parse_intent("Flight from Delhi to Mumbai on 2026-05-15, layover under 2 hours")
    assert intent.layover_limit_minutes == 120
    assert intent.price_limit is None


def test_parse_intent_preserves_multiword_via_city():
    intent = parse_intent("Find flights from Lucknow to Mumbai via New Delhi tomorrow")
    assert intent.stopover_city == "New Delhi"


def test_parse_intent_preserves_uppercase_iata_stopover_token():
    intent = parse_intent("DEL to MAA via BLR on 2026-05-15")
    assert intent.stopover_city == "BLR"


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


def test_parse_intent_round_trip_leave_return_phrase_resolves_route_and_dates():
    query = (
        "Round-trip Delhi to Mumbai, leave 2026-05-15 and return 2026-05-17, "
        "prioritize cheapest acceptable option."
    )
    intent = parse_intent(query)
    assert intent.origin_iata == "DEL"
    assert intent.destination_iata == "BOM"
    assert intent.date == "2026-05-15"
    assert intent.return_date == "2026-05-17"
    raw_fragments = (intent.route_parse_trace or {}).get("raw_fragments") or {}
    assert raw_fragments.get("origin_text") == "Delhi"
    assert raw_fragments.get("destination_text") == "Mumbai"


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


@pytest.mark.asyncio
async def test_correct_cities_with_llm_timeout_logs_info_not_warning(monkeypatch, caplog):
    async def fake_check_llm_circuit():
        return False

    async def slow_generate(**_kwargs):
        await asyncio.sleep(0.01)
        return "{\"origin\": \"Delhi\", \"destination\": \"Mumbai\"}"

    monkeypatch.setattr(planner_agent, "check_llm_circuit", fake_check_llm_circuit)
    monkeypatch.setattr(planner_agent, "generate", slow_generate)
    monkeypatch.setattr(planner_agent, "LLM_CORRECTION_TIMEOUT", 0.001)

    with caplog.at_level(logging.INFO):
        resolved = await planner_agent.correct_cities_with_llm("Round-trip Delhi to Mumbai")

    assert resolved == (None, None, None)
    assert any(
        "LLM city correction timed out; using deterministic route recovery fallback" in record.message
        and record.levelname == "INFO"
        for record in caplog.records
    )
    assert not any(
        "LLM city correction timed out" in record.message and record.levelno >= logging.WARNING
        for record in caplog.records
    )


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


def test_parse_intent_preserves_embedded_uppercase_iata_tokens_in_noisy_phrase():
    intent = parse_intent("Find cheapest DEL to BOM on 2026-04-18")
    assert intent.origin_iata == "DEL"
    assert intent.destination_iata == "BOM"


def test_ensure_route_grounding_appends_canonical_route_for_misspelled_narrative():
    narrative = "Best option is from Dehli to Bombay on 2026-05-15."
    grounded = _ensure_route_grounding(narrative, "DEL", "BOM")
    grounded_lower = grounded.lower()
    assert "new delhi (del)" in grounded_lower
    assert "mumbai (bom)" in grounded_lower


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
async def test_plan_trip_internal_explicit_fastest_selects_shortest_duration(monkeypatch):
    async def fake_flight_tool(**_kwargs):
        return [
            Flight(
                airline="TestAir",
                flight_no="TA110",
                departure_time="08:00",
                arrival_time="09:50",
                duration_min=110,
                price_inr=5100,
                stops=0,
                layover_info="",
                travel_class="Economy",
            ),
            Flight(
                airline="TestAir",
                flight_no="TA100",
                departure_time="10:00",
                arrival_time="11:40",
                duration_min=100,
                price_inr=5400,
                stops=0,
                layover_info="",
                travel_class="Economy",
            ),
        ]

    async def fake_weather_tool(*, location: str, date: str):
        return {"condition": "Clear", "temperature_c": 29, "forecast_date": date, "location": location}

    result = await planner_agent._plan_trip_internal(
        user_query="BOM to BLR with least travel time on 2026-05-15",
        trip_type="Business",
        skip_llm=True,
        flight_tool=fake_flight_tool,
        weather_tool=fake_weather_tool,
    )

    assert isinstance(result, planner_agent.PlanResult)
    assert result.best_flight["flight_no"] == "TA100"
    assert result.best_flight["duration_min"] == 100


@pytest.mark.asyncio
async def test_plan_trip_internal_explicit_business_request_is_truthful_when_unavailable(monkeypatch):
    async def fake_flight_tool(**_kwargs):
        return [
            Flight(
                airline="EcoAir",
                flight_no="EA101",
                departure_time="09:00",
                arrival_time="11:00",
                duration_min=120,
                price_inr=5600,
                stops=0,
                layover_info="",
                travel_class="Economy",
            ),
            Flight(
                airline="EcoAir",
                flight_no="EA102",
                departure_time="12:00",
                arrival_time="14:05",
                duration_min=125,
                price_inr=5800,
                stops=0,
                layover_info="",
                travel_class="Economy",
            ),
        ]

    async def fake_weather_tool(*, location: str, date: str):
        return {"condition": "Sunny", "temperature_c": 32, "forecast_date": date, "location": location}

    result = await planner_agent._plan_trip_internal(
        user_query="Need business class flight from MAA to DEL on 2026-05-15",
        trip_type="Business",
        skip_llm=True,
        flight_tool=fake_flight_tool,
        weather_tool=fake_weather_tool,
    )

    assert isinstance(result, planner_agent.PlanResult)
    assert result.best_flight["travel_class"] == "Economy"
    assert isinstance(result.constraint_outcomes, dict)
    cabin_meta = result.constraint_outcomes.get("cabin")
    assert isinstance(cabin_meta, dict)
    assert cabin_meta.get("requested") == "business"
    assert cabin_meta.get("matched") is False
    assert cabin_meta.get("fallback_applied") is True
    assert any("No Business cabin inventory matched this search" in w for w in (result.warnings or []))


def test_infer_route_pair_from_query_recovers_explicit_from_to_phrase():
    origin, destination, trace = planner_agent._infer_route_pair_from_query(
        "Need a round trip from Delhi to Mumbai returning next week"
    )
    assert origin == "DEL"
    assert destination == "BOM"
    assert trace.get("source") in {
        "deterministic_from_to_phrase",
        "resolver_phrase_pair_from_to_fallback",
    }


@pytest.mark.asyncio
async def test_plan_trip_internal_invalid_route_still_surfaces_handoff_contract(monkeypatch):
    def fake_parse_intent(_query: str):
        return planner_agent.ParsedIntent(
            origin_iata=None,
            destination_iata=None,
            date="2026-06-20",
            trip_type="Business",
        )

    monkeypatch.setattr(planner_agent, "parse_intent", fake_parse_intent)
    monkeypatch.setattr(planner_agent, "_infer_route_pair_from_query", lambda _q: (None, None, {"source": "none"}))
    async def fake_llm_correction(_query: str):
        return None, None, None

    monkeypatch.setattr(planner_agent, "correct_cities_with_llm", fake_llm_correction)

    result = await planner_agent._plan_trip_internal(
        user_query="from nowhere to ???",
        trip_type="Business",
        skip_llm=True,
    )

    assert isinstance(result, dict)
    assert result["failure_reason"] == "invalid_route"
    assert result["top_flights"] == []
    assert result["booking_handoff"]["status"] == "unavailable"
    assert result["booking_handoff"]["source"] == "unavailable"
    assert result["booking_handoff"]["booking_exit_quality"] == "unavailable"
    assert result["debug_info"]["top_flights"] == []
    assert "booking_handoff_quality_context" not in result["debug_info"]


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
    expected_max = min(
        planner_agent.FLIGHT_SEARCH_MAX_RESULTS_CAP,
        planner_agent.FLIGHT_SEARCH_BASE_RESULTS
        + planner_agent.FLIGHT_SEARCH_ROUND_TRIP_BONUS
        + planner_agent.FLIGHT_SEARCH_DEEP_SEARCH_BONUS,
    )
    assert calls[0].get("max_results") == expected_max


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


@pytest.mark.asyncio
async def test_plan_trip_internal_natural_language_round_trip_phrase_populates_contract(monkeypatch):
    query = (
        "Round-trip Delhi to Mumbai, leave 2026-05-15 and return 2026-05-17, "
        "prioritize cheapest acceptable option."
    )

    async def fake_flight_tool(**kwargs):
        date = kwargs.get("date")
        departure = kwargs.get("departure")
        arrival = kwargs.get("arrival")
        return [
            Flight(
                airline="TestAir",
                flight_no=f"TA-{departure}-{arrival}",
                departure_time="09:00",
                arrival_time="11:00",
                duration_min=120,
                price_inr=4800,
                stops=0,
                layover_info="",
                baggage="7kg cabin",
                date=date,
            )
        ], {"_search_meta": {"raw_candidate_count": 1}}

    async def fake_weather_tool(*, location: str, travel_date: str):
        return {
            "condition": "Clear",
            "temperature_c": 29,
            "forecast_date": travel_date,
            "location": location,
        }

    result = await planner_agent._plan_trip_internal(
        user_query=query,
        trip_type="round-trip",
        skip_llm=True,
        flight_tool=fake_flight_tool,
        weather_tool=fake_weather_tool,
    )

    assert isinstance(result, planner_agent.PlanResult)
    assert result.search_date == "2026-05-15"
    assert result.return_trip is not None
    assert result.debug_info["route_labels"]["origin_iata"] == "DEL"
    assert result.debug_info["route_labels"]["destination_iata"] == "BOM"
    assert isinstance(result.top_flights, list) and len(result.top_flights) >= 1
    assert isinstance(result.booking_handoff, dict)
    assert result.booking_handoff.get("booking_exit_quality") in {"unavailable", "booking_ready", "deferred"}
    assert isinstance(result.best_flight.get("booking_handoff"), dict)
    rt_block = result.booking_handoff.get("round_trip") or {}
    assert rt_block.get("return_search_outcome") in {"ok", "failed", "not_attempted"}
    assert rt_block.get("return_handoff_status") in {"booking_ready", "deferred", "unavailable"}
    assert isinstance(rt_block.get("is_outbound_only_handoff"), bool)


@pytest.mark.asyncio
async def test_round_trip_block_marks_outbound_only_when_return_leg_fails(monkeypatch):
    query = (
        "Round-trip Delhi to Mumbai, leave 2026-05-15 and return 2026-05-17, "
        "prioritize cheapest acceptable option."
    )

    async def flaky_round_trip_flight_tool(**kwargs):
        departure = kwargs.get("departure")
        arrival = kwargs.get("arrival")
        if departure == "BOM" and arrival == "DEL":
            raise RuntimeError("return leg upstream unavailable")
        date = kwargs.get("date")
        return [
            Flight(
                airline="TestAir",
                flight_no="TA-OUT",
                departure_time="09:00",
                arrival_time="11:00",
                duration_min=120,
                price_inr=4800,
                stops=0,
                layover_info="",
                baggage="7kg cabin",
                date=date,
            )
        ], {"_search_meta": {"raw_candidate_count": 1}}

    async def fake_weather_tool(*, location: str, travel_date: str):
        return {
            "condition": "Clear",
            "temperature_c": 29,
            "forecast_date": travel_date,
            "location": location,
        }

    result = await planner_agent._plan_trip_internal(
        user_query=query,
        trip_type="round-trip",
        skip_llm=True,
        flight_tool=flaky_round_trip_flight_tool,
        weather_tool=fake_weather_tool,
    )

    assert isinstance(result, planner_agent.PlanResult)
    contract = result.booking_handoff.get("round_trip") or {}
    assert contract.get("return_search_outcome") == "failed"
    assert contract.get("return_handoff_status") == "unavailable"
    assert isinstance(contract.get("is_outbound_only_handoff"), bool)
