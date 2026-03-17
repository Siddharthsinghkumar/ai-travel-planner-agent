#tests/test_planner_logic.py
import pytest
from agents.planner_agent import parse_intent, filter_flights, Flight

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