import os
import requests
from dotenv import load_dotenv
load_dotenv()
def search_flights(departure: str, arrival: str, date: str, max_results=5):
    """
    Fetches flights from Google Flights via SerpApi.

    Args:
        departure (str): IATA code (e.g., 'DEL')
        arrival (str): IATA code (e.g., 'BOM')
        date (str): Date in YYYY-MM-DD format
        max_results (int): Limit number of results returned

    Returns:
        List[Dict]: Flights with airline, flight number, timing, duration, and price.
    """
    API_KEY = os.getenv("SERPAPI_KEY")
    if not API_KEY:
        raise RuntimeError("Set SERPAPI_KEY in your environment before running")
    params = {
        "engine": "google_flights",
        "departure_id": departure.upper(),
        "arrival_id": arrival.upper(),
        "outbound_date": date,
        "type": "2",  # One-way
        "travel_class": "1",  # Economy
        "adults": "1",
        "hl": "en",
        "gl": "in",
        "currency": "INR",
        "deep_search": "true",
        "api_key": API_KEY
    }

    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    flights = data.get("other_flights", [])

    results = []
    print("🔍 Flights fetched (showing key fields):")
    for idx, result in enumerate(flights[:max_results], start=1):
        flight = result.get("flights", [{}])[0]
        airline     = flight.get("airline", "–")
        fno         = flight.get("flight_number", "–")
        dep_time    = flight.get("departure_airport", {}).get("time", "–")
        arr_time    = flight.get("arrival_airport", {}).get("time", "–")
        price       = result.get("price", "–")
        print(f"{idx}. {airline} {fno} | Dep: {dep_time} | Arr: {arr_time} | Price: {price}")

    for result in flights[:max_results]:
        try:
            flight = result["flights"][0]
            price = result.get("price")
            if not price:
                print(f"⚠️ Missing price in result: {airline} {fno} @ Dep {dep_time}")
                continue # Skip this flight

            results.append({
                "airline": flight["airline"],
                "flight_no": flight["flight_number"],
                "departure_time": flight["departure_airport"]["time"],
                "arrival_time": flight["arrival_airport"]["time"],
                "duration_min": flight["duration"],
                "price_inr": price
            })
        except Exception as e:
            print("⚠️ Skipped a result due to parsing error:", e)

    print("📡 Calling SerpAPI with date:", date)
    return results
