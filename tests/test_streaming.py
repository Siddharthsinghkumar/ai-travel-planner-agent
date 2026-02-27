# tests/test_streaming.py
import asyncio
from fastapi.testclient import TestClient
import pytest

from api.app import app

client = TestClient(app)

# Fake async generator to simulate LLM streaming
async def _fake_stream_generator():
    # Simulate status tokens then content tokens then a final done marker
    yield "Searching flights...\n"
    await asyncio.sleep(0)  # allow event loop scheduling
    yield "Found 3 options. Choosing best one...\n"
    await asyncio.sleep(0)
    # Simulate content tokens from the model
    for token in ["Hello", ", ", "this ", "is ", "a ", "streamed ", "response."]:
        yield token
    # generator naturally ends

# Wrapper that returns the async generator (matches agents.llm_router.generate signature)
async def fake_generate(prompt: str, system: str = "", model: str = None, stream: bool = False, **kwargs):
    if stream:
        return _fake_stream_generator()
    # In case code calls generate(...) with stream=False during tests, return a simple string
    return "non-stream response"

@pytest.fixture(autouse=True)
def patch_llm_router(monkeypatch):
    """
    Autouse fixture: patch the router's generate to avoid network I/O during tests.
    """
    # patch the function used by planner_agent (module path may vary based on your package layout)
    monkeypatch.setattr("agents.planner_agent.generate", fake_generate)
    yield
    # monkeypatch will undo automatically

def test_streaming_endpoint_returns_sse(monkeypatch):
    # Patch flight search tool to avoid real API call
    async def fake_search_flights(*args, **kwargs):
        return [
            {
                "airline": "TestAir",
                "flight_no": "TA123",
                "departure_time": "06:00",
                "arrival_time": "08:00",
                "duration_min": 120,
                "price_inr": "₹1000",
                "stops": "N/A",
                "baggage": "N/A",
                "layover_time": "0",
                "date": "2026-03-15"
            }
        ]

    monkeypatch.setattr(
        "tools.airline_api.search_flights",
        fake_search_flights
    )

    # Patch weather tool if used (optional)
    async def fake_weather(*args, **kwargs):
        return {
            "location": "BOM",
            "condition": "Clear",
            "temperature_c": 25,
            "feels_like_c": 25,
            "humidity": 50,
            "wind_kph": 5,
            "air_quality_index": 1,
            "timestamp": None
        }

    monkeypatch.setattr(
        "tools.weather_api.get_weather",
        fake_weather
    )

    payload = {"date": "2026-03-15", "user_query": "test from delhi to mumbai", "trip_type": "Business"}
    with client.stream("POST", "/ask?stream=true", json=payload) as resp:
        assert resp.status_code == 200
        content = ""
        # read a few chunks (non-blocking) to ensure we got streaming data
        for chunk in resp.iter_text():
            if not chunk:
                break
            content += chunk
            # stop early once we saw the done event or the done JSON prefix
            if "[DONE_JSON]" in content or "event: done" in content:
                break

        text = content
        # check for at least one token and SSE framing
        assert "data:" in text or "Searching flights" in text
        # check we received model tokens
        assert "streamed response" in text.lower()
        # ensure final marker present
        assert "[DONE_JSON]" in text or "event: done" in text