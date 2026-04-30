import contextlib
import time
from unittest.mock import AsyncMock

import pytest

import tools.weather_api
from tools.weather_api import Weather, WeatherAPIError, _redact_request_params, get_forecast_for_date


class _DummyResponse:
    def __init__(self, *, status_code: int, json_data=None, text: str = "", headers=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json_data


class _DummyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def request(self, *_args, **_kwargs):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


class _DummyKeyManager:
    def __init__(self):
        self.mark_calls = []
        self.record_calls = []

    @contextlib.asynccontextmanager
    async def reserve_key(self, _service):
        yield 0, "weather-key"

    async def mark_exhausted(self, service, idx, *, until=None, reason=""):
        self.mark_calls.append(
            {
                "service": service,
                "idx": idx,
                "until": until,
                "reason": reason,
            }
        )

    async def record_usage(self, service, idx):
        self.record_calls.append((service, idx))


@pytest.mark.asyncio
async def test_get_forecast_for_date(monkeypatch):
    mock_forecasts = [
        Weather(
            location="BOM",
            condition="Clear",
            temperature_c=25,
            feels_like_c=25,
            humidity=50,
            wind_kph=5,
            forecast_date="2026-03-14",
        ),
        Weather(
            location="BOM",
            condition="Rain",
            temperature_c=22,
            feels_like_c=22,
            humidity=80,
            wind_kph=10,
            forecast_date="2026-03-15",
        ),
        Weather(
            location="BOM",
            condition="Cloudy",
            temperature_c=24,
            feels_like_c=24,
            humidity=60,
            wind_kph=8,
            forecast_date="2026-03-16",
        ),
    ]

    monkeypatch.setattr(tools.weather_api, "get_forecast", AsyncMock(return_value=mock_forecasts))
    monkeypatch.setattr(tools.weather_api, "get_current_weather", AsyncMock(return_value=mock_forecasts[0]))

    result = await get_forecast_for_date("BOM", "2026-03-15")

    assert result.forecast_date == "2026-03-15"
    assert result.condition == "Rain"
    assert result.temperature_c == 22


def test_redact_request_params_masks_appid_and_auth_fields():
    redacted = _redact_request_params(
        {
            "appid": "secret-weather-key",
            "authorization": "Bearer secret-token",
            "x-api-key": "secret-header-key",
            "api_key": "secret-generic-key",
            "q": "Mumbai",
        }
    )

    assert redacted["appid"] == "***REDACTED***"
    assert redacted["authorization"] == "***REDACTED***"
    assert redacted["x-api-key"] == "***REDACTED***"
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["q"] == "Mumbai"


@pytest.mark.asyncio
async def test_make_request_raw_marks_key_exhausted_on_payload_unauthorized(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient(
        [
            _DummyResponse(
                status_code=200,
                json_data={"cod": "401", "message": "Invalid API key"},
                text='{"cod":"401","message":"Invalid API key"}',
            )
        ]
    )
    monkeypatch.setattr(tools.weather_api, "key_manager", dummy_km)
    monkeypatch.setattr(tools.weather_api, "get_client", lambda: client)
    monkeypatch.setattr(tools.weather_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(tools.weather_api.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(tools.weather_api, "MAX_RETRIES", 1)

    with pytest.raises(WeatherAPIError, match="unauthorized or pending activation"):
        await tools.weather_api._make_request_raw("GET", "https://example.test/weather", {"q": "BOM"})

    assert len(dummy_km.mark_calls) == 1
    mark = dummy_km.mark_calls[0]
    assert mark["service"] == "weather"
    assert mark["idx"] == 0
    assert "unauthorized" in mark["reason"]


@pytest.mark.asyncio
async def test_make_request_raw_marks_key_exhausted_on_http_401(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient([_DummyResponse(status_code=401, text="unauthorized")])

    monkeypatch.setattr(tools.weather_api, "key_manager", dummy_km)
    monkeypatch.setattr(tools.weather_api, "get_client", lambda: client)
    monkeypatch.setattr(tools.weather_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(tools.weather_api.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(tools.weather_api, "MAX_RETRIES", 1)

    with pytest.raises(WeatherAPIError, match="unauthorized or pending activation"):
        await tools.weather_api._make_request_raw("GET", "https://example.test/weather", {"q": "DEL"})

    assert len(dummy_km.mark_calls) == 1
    assert "unauthorized_http_401" in dummy_km.mark_calls[0]["reason"]


@pytest.mark.asyncio
async def test_make_request_raw_http_429_uses_retry_after_and_marks_exhausted(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient(
        [
            _DummyResponse(
                status_code=429,
                text="too many requests",
                headers={"Retry-After": "7"},
            )
        ]
    )

    monkeypatch.setattr(tools.weather_api, "key_manager", dummy_km)
    monkeypatch.setattr(tools.weather_api, "get_client", lambda: client)
    monkeypatch.setattr(tools.weather_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(tools.weather_api.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(tools.weather_api, "MAX_RETRIES", 1)

    with pytest.raises(WeatherAPIError, match="Max retries exceeded"):
        await tools.weather_api._make_request_raw("GET", "https://example.test/weather", {"q": "GOI"})

    assert len(dummy_km.mark_calls) == 1
    mark = dummy_km.mark_calls[0]
    assert "http_429" in mark["reason"]
    assert isinstance(mark["until"], int)
    assert mark["until"] >= int(time.time()) + 5


@pytest.mark.asyncio
async def test_make_request_raw_honors_total_attempt_budget_under_repeated_429(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient(
        [
            _DummyResponse(status_code=429, text="too many requests", headers={"Retry-After": "1"}),
            _DummyResponse(status_code=429, text="too many requests", headers={"Retry-After": "1"}),
            _DummyResponse(status_code=429, text="too many requests", headers={"Retry-After": "1"}),
        ]
    )

    monkeypatch.setattr(tools.weather_api, "key_manager", dummy_km)
    monkeypatch.setattr(tools.weather_api, "get_client", lambda: client)
    monkeypatch.setattr(tools.weather_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(tools.weather_api.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(tools.weather_api, "MAX_RETRIES", 5)
    monkeypatch.setattr(tools.weather_api, "WEATHER_TOTAL_ATTEMPT_BUDGET", 2)

    with pytest.raises(WeatherAPIError, match="within budget \\(2 attempts\\)"):
        await tools.weather_api._make_request_raw("GET", "https://example.test/weather", {"q": "GOI"})

    assert client.calls == 2
