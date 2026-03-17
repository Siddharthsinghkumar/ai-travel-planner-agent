#tests/test_weather_api.py
import pytest
from unittest.mock import AsyncMock
import tools.weather_api
from tools.weather_api import get_forecast_for_date, Weather

@pytest.mark.asyncio
async def test_get_forecast_for_date(monkeypatch):
    # 1. Create a mock 3-day forecast
    mock_forecasts = [
        Weather(location="BOM", condition="Clear", temperature_c=25, feels_like_c=25, humidity=50, wind_kph=5, forecast_date="2026-03-14"),
        Weather(location="BOM", condition="Rain", temperature_c=22, feels_like_c=22, humidity=80, wind_kph=10, forecast_date="2026-03-15"),
        Weather(location="BOM", condition="Cloudy", temperature_c=24, feels_like_c=24, humidity=60, wind_kph=8, forecast_date="2026-03-16")
    ]
    
    # 2. Mock the internal dependencies
    monkeypatch.setattr(tools.weather_api, "get_forecast", AsyncMock(return_value=mock_forecasts))
    # Mock current weather for the AQI fallback we added
    monkeypatch.setattr(tools.weather_api, "get_current_weather", AsyncMock(return_value=mock_forecasts[0]))
    
    # 3. Request weather for a specific date in the middle of the array
    result = await get_forecast_for_date("BOM", "2026-03-15")
    
    # 4. Assert it picked the correct day's weather, not the first item
    assert result.forecast_date == "2026-03-15"
    assert result.condition == "Rain"
    assert result.temperature_c == 22