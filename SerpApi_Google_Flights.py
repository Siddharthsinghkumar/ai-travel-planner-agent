import requests

# Replace this with your actual API key from SerpApi
API_KEY = "a2be66706d1f122f9ff7ef2da779d9f91c28a3880d6d9f1bd7dcfcb559c32247"

params = {
    "engine": "google_flights",
    "departure_id": "DEL",
    "arrival_id": "BOM",
    "outbound_date": "2025-06-15",
    "type": "2",  # One way
    "travel_class": "1",  # Economy
    "adults": "1",
    "hl": "en",
    "gl": "in",
    "currency": "INR",
    "deep_search": "true",  # identical results to browser
    "api_key": API_KEY
}

response = requests.get("https://serpapi.com/search", params=params)
data = response.json()

flights = data.get("other_flights", [])
if not flights:
    print("⚠️ No flights found.")
else:
    print(f"✅ Found {len(flights)} flights.\n")
    for result in flights[:5]:
        try:
            flight = result["flights"][0]
            airline = flight["airline"]
            flight_no = flight["flight_number"]
            dep_time = flight["departure_airport"]["time"]
            arr_time = flight["arrival_airport"]["time"]
            duration = flight["duration"]
            price = result["price"]

            print(f"{airline} {flight_no}")
            print(f"  Departure: {dep_time}")
            print(f"  Arrival  : {arr_time}")
            print(f"  Duration : {duration} minutes")
            print(f"  Price    : ₹{price}")
            print("-" * 50)
        except Exception as e:
            print("⚠️ Skipped a result due to error:", e)
