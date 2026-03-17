#tests/test_booking_handoff.py
import pytest
from tools.booking_handoff import build_google_flights_fallback, build_booking_handoff_url

def test_build_google_flights_fallback():
    url = build_google_flights_fallback(
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20"
    )
    
    # Ensure it uses the correct base
    assert url.startswith("https://www.google.com/travel/flights?q=")
    # Ensure spaces are properly URL-encoded (no raw spaces allowed in valid URLs)
    assert " " not in url
    assert "Flights%20from%20DEL%20to%20BOM" in url
    assert "2026-03-20" in url

@pytest.mark.asyncio
async def test_build_booking_handoff_url_priorities():
    # Test that shareable_link is prioritized over the fallback
    flight_mock = {
        "shareable_link": "https://google.com/flights/shareable_test"
    }
    
    url = await build_booking_handoff_url(
        flight=flight_mock,
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20"
    )
    
    assert url == "https://google.com/flights/shareable_test"