import os
import requests
from bs4 import BeautifulSoup

API_KEY = os.getenv("WEATHER_API_KEY")

def get_coordinates(city):
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        "q": city,
        "limit": 1,
        "appid": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    if not data:
        raise ValueError(f"City '{city}' not found.")
    return data[0]["lat"], data[0]["lon"]

def get_weather(lat, lon):
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    return response.json()

def get_air_quality(lat, lon):
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_weather_from_wttr(city):
    try:
        url = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)

        if res.status_code != 200:
            print(f"[DEBUG] wttr.in request failed: {res.status_code}")
            return None

        data = res.json()
        current = data["current_condition"][0]

        desc = current["weatherDesc"][0]["value"]
        temp = current["temp_C"]
        feels = current["FeelsLikeC"]
        humidity = current["humidity"]

        return (
            f"🌤️ Weather in {city} (via wttr.in):\n"
            f"• Condition: {desc}\n"
            f"• Temperature: {temp}°C (Feels like {feels}°C)\n"
            f"• Humidity: {humidity}%"
        )
    except Exception as e:
        print(f"[DEBUG] wttr.in fallback failed: {e}")
        return None


def check_weather(city, date=None):
    try:
        lat, lon = get_coordinates(city)
        weather = get_weather(lat, lon)
        air_quality = get_air_quality(lat, lon)

        desc = weather["weather"][0]["description"].capitalize()
        temp = weather["main"]["temp"]
        feels_like = weather["main"]["feels_like"]
        humidity = weather["main"]["humidity"]

        aqi_index = air_quality["list"][0]["main"]["aqi"]
        aqi_status = {
            1: "Good",
            2: "Fair",
            3: "Moderate",
            4: "Poor",
            5: "Very Poor"
        }.get(aqi_index, "Unknown")

        return (
            f"🌤️ Weather in {city}:\n"
            f"• Condition: {desc}\n"
            f"• Temperature: {temp}°C (Feels like {feels_like}°C)\n"
            f"• Humidity: {humidity}%\n"
            f"• Air Quality Index: {aqi_index} — {aqi_status}"
        )

    except Exception as e:
        print(f"[DEBUG] OpenWeatherMap error: {e}")
        print(f"[DEBUG] Falling back to wttr.in for '{city}'…")
        fallback = get_weather_from_wttr(city)
        return fallback or f"❌ Error fetching weather for '{city}' from both sources."


