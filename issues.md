(venv) sidd@sidd-ASUS-TUF-Gaming-F15-FX507ZE-FX577ZE:~/project/llm-travel-agent$ git status
On branch main
Your branch is ahead of 'origin/main' by 1 commit.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   README.md
	modified:   agents/llm_router.py
	modified:   agents/planner_agent.py
	modified:   api/app.py
	modified:   core/health.py
	modified:   core/iata_resolver.py
	modified:   core/metrics.py
	modified:   frontend/README.md
	modified:   frontend/package.json
	modified:   frontend/src/App.tsx
	modified:   frontend/src/components/AIReasoningPanel.tsx
	modified:   frontend/src/components/AuroraCanvas.tsx
	modified:   frontend/src/components/FeatureCapabilities.tsx
	modified:   frontend/src/components/FlightCard.tsx
	modified:   frontend/src/components/FlightsTicker.tsx
	modified:   frontend/src/components/QueryForm.tsx
	modified:   frontend/src/components/StreamPane.tsx
	modified:   frontend/src/components/WeatherSummary.tsx
	modified:   frontend/src/components/flights-ticker.css
	modified:   frontend/src/hooks/useStreamingPlan.tsx
	modified:   frontend/src/index.css
	modified:   frontend/src/lib/capabilities.ts
	modified:   frontend/src/lib/format.ts
	modified:   frontend/src/lib/types.ts
	deleted:    gemini_multikey_9_3_helper_script.py
	modified:   tests/test_airline_api.py
	modified:   tests/test_api.py
	modified:   tests/test_full_pipeline.py
	modified:   tests/test_full_validation_soft_pass.py
	modified:   tests/test_health.py
	modified:   tests/test_llm_router_modes.py
	modified:   tests/test_streaming.py
	modified:   tests/test_weather_api.py
	modified:   tools/airline_api.py
	modified:   tools/booking_handoff.py
	modified:   tools/price_tracker.py
	modified:   tools/weather_api.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.env
	.env.laptopdocker
	.playwright-browsers/
	.vscode/
	COMMIT
	Failed api&scrapper script/
	docs/
	frontend/scripts/export-frozen-demo.mjs
	frontend/src/components/MultiCitySummary.tsx
	full_validation.py
	monitoring/
	tests/test_frontend_validator_mode.py
	tests/test_iata_resolver_labels.py
	tests/test_observability_metrics.py
	tests/test_planner_stream_error_contract.py
	ui_dummy_prototype.html
	validation/
	validation_logs/
	venv/

no changes added to commit (use "git add" and/or "git commit -a")
(venv) sidd@sidd-ASUS-TUF-Gaming-F15-FX507ZE-FX577ZE:~/project/llm-travel-agent$ git diff --stat
 README.md                                       | 187 +++++++++++++++
 agents/llm_router.py                            |  35 ++-
 agents/planner_agent.py                         | 112 +++++++--
 api/app.py                                      | 176 +++++++++++---
 core/health.py                                  |  32 ++-
 core/iata_resolver.py                           |  38 ++-
 core/metrics.py                                 | 126 +++++++++-
 frontend/README.md                              |  26 ++-
 frontend/package.json                           |   3 +-
 frontend/src/App.tsx                            | 295 +++++++++++++++++++-----
 frontend/src/components/AIReasoningPanel.tsx    | 115 +++------
 frontend/src/components/AuroraCanvas.tsx        |  65 +++++-
 frontend/src/components/FeatureCapabilities.tsx |  15 +-
 frontend/src/components/FlightCard.tsx          |  10 +-
 frontend/src/components/FlightsTicker.tsx       |  39 +++-
 frontend/src/components/QueryForm.tsx           |  16 +-
 frontend/src/components/StreamPane.tsx          |  95 +-------
 frontend/src/components/WeatherSummary.tsx      |  44 +++-
 frontend/src/components/flights-ticker.css      |  12 +-
 frontend/src/hooks/useStreamingPlan.tsx         | 144 +++++++++++-
 frontend/src/index.css                          | 235 +++++++++++++++++--
 frontend/src/lib/capabilities.ts                |   6 +-
 frontend/src/lib/format.ts                      |   4 +-
(venv) sidd@sidd-ASUS-TUF-Gaming-F15-FX507ZE-FX577ZE:~/project/llm-travel-agent$ curl -sS -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"user_query":"Cheap flight from Delhi to Mumbai","trip_type":"one-way"}' | jq
{
  "llm_response": "I recommend IndiGo 6E 6218 at 06:05 arriving 08:15. Duration: 130 minutes, Price: ₹6,725. Weather at destination (BOM): Clear sky, 18°C. (Note: Enhanced explanation unavailable due to LLM backends temporarily unavailable.)",
  "best_flight": {
    "airline": "IndiGo",
    "flight_no": "6E 6218",
    "departure_time": "06:05",
    "arrival_time": "08:15",
    "duration_min": 130,
    "price_inr": "₹6,725",
    "stops": 0,
    "layover_info": "",
    "baggage": "Check airline",
    "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZeU1UZ2FDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjIxOCJdXV0=",
    "carbon_emissions_g": 85000,
    "date": "2026-03-22",
    "handoff_url": "https://www.google.com/travel/flights?q=Flights%20from%20DEL%20to%20BOM%20on%202026-03-22",
    "layover_durations_min": [],
    "layover_airports": []
  },
  "weather": {
    "location": "BOM",
    "condition": "Clear sky",
    "temperature_c": 18,
    "feels_like_c": 7.12625,
    "humidity": 66,
    "wind_kph": 11.3,
    "air_quality_index": null,
    "timestamp": 1774180800,
    "temp_min_c": 4.35,
    "temp_max_c": 14.52,
    "has_rain": false,
    "has_snow": false,
    "precipitation_chance": 0,
    "forecast_date": "2026-03-22",
    "location_city": "Mumbai",
    "location_label": "Mumbai (BOM)"
  },
  "search_date": "2026-03-22",
  "warnings": null,
  "debug_info": {
    "phases": {
      "intent_parsing": 0.001768801001162501,
      "api_parallel": 4.114576711002883,
      "filter_rank": 0.00005623199831461534,
      "llm_generation": 30.003351188002853
    },
    "intent": {
      "origin_iata": "DEL",
      "destination_iata": "BOM",
      "date": null,
      "return_date": null,
      "time_pref": null,
      "price_limit": null,
      "wants_direct": false,
      "preferred_airlines": [],
      "layover_limit_minutes": null,
      "baggage_pref": null,
      "trip_duration_days": null,
      "stopover_city": null,
      "flight_pref": "cheapest",
      "wants_eco": false,
      "trip_type": "Business",
      "deep_search": false
    },
    "effective_intent": {
      "origin_iata": "DEL",
      "destination_iata": "BOM",
      "date": null,
      "return_date": null,
      "time_pref": null,
      "price_limit": null,
      "wants_direct": false,
      "preferred_airlines": [],
      "layover_limit_minutes": null,
      "baggage_pref": null,
      "trip_duration_days": null,
      "stopover_city": null,
      "flight_pref": "cheapest",
      "wants_eco": false,
      "trip_type": "Business",
      "deep_search": false
    },
    "route_labels": {
      "origin_iata": "DEL",
      "origin_city": "New Delhi",
      "origin_label": "New Delhi (DEL)",
      "destination_iata": "BOM",
      "destination_city": "Mumbai",
      "destination_label": "Mumbai (BOM)"
    },
    "filters_applied": "cheapest / lowest price",
    "trip_description": "a Business trip",
    "all_flights": [
      {
        "airline": "IndiGo",
        "flight_no": "6E 6218",
        "departure_time": "06:05",
        "arrival_time": "08:15",
        "duration_min": 130,
        "price_inr": "₹6,725",
        "stops": 0,
        "layover_info": "",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZeU1UZ2FDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjIxOCJdXV0=",
        "carbon_emissions_g": 85000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [],
        "layover_airports": []
      },
      {
        "airline": "Air India",
        "flight_no": "AI 2951",
        "departure_time": "13:20",
        "arrival_time": "15:45",
        "duration_min": 145,
        "price_inr": "₹7,470",
        "stops": 0,
        "layover_info": "",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1pCU1RJNU5URWFDZ2l1T2hBQUdnTkpUbEk0SEhDS1BnPT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIkFJIiwiMjk1MSJdXV0=",
        "carbon_emissions_g": 99000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [],
        "layover_airports": []
      },
      {
        "airline": "SpiceJet",
        "flight_no": "SG 385",
        "departure_time": "06:30",
        "arrival_time": "08:35",
        "duration_min": 125,
        "price_inr": "₹8,290",
        "stops": 0,
        "layover_info": "",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1ZUUnpNNE5Sb0tDT0pBRUFBYUEwbE9VamdjY1BKRSIsW1siREVMIiwiMjAyNi0wMy0yMiIsIkJPTSIsbnVsbCwiU0ciLCIzODUiXV1d",
        "carbon_emissions_g": 98000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [],
        "layover_airports": []
      },
      {
        "airline": "IndiGo",
        "flight_no": "6E 6787",
        "departure_time": "00:30",
        "arrival_time": "02:45",
        "duration_min": 135,
        "price_inr": "₹6,725",
        "stops": 0,
        "layover_info": "",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZM09EY2FDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjc4NyJdXV0=",
        "carbon_emissions_g": 85000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [],
        "layover_airports": []
      },
      {
        "airline": "IndiGo",
        "flight_no": "6E 2766",
        "departure_time": "04:00",
        "arrival_time": "06:20",
        "duration_min": 140,
        "price_inr": "₹6,725",
        "stops": 0,
        "layover_info": "",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRJM05qWWFDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiMjc2NiJdXV0=",
        "carbon_emissions_g": 85000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [],
        "layover_airports": []
      },
      {
        "airline": "IndiGo",
        "flight_no": "6E 449",
        "departure_time": "05:00",
        "arrival_time": "07:15",
        "duration_min": 135,
        "price_inr": "₹6,725",
        "stops": 0,
        "layover_info": "",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1UyUlRRME9Sb0tDTVUwRUFBYUEwbE9VamdjY1BFMyIsW1siREVMIiwiMjAyNi0wMy0yMiIsIkJPTSIsbnVsbCwiNkUiLCI0NDkiXV1d",
        "carbon_emissions_g": 85000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [],
        "layover_airports": []
      },
      {
        "airline": "IndiGo",
        "flight_no": "6E 6814",
        "departure_time": "07:05",
        "arrival_time": "09:15",
        "duration_min": 130,
        "price_inr": "₹6,725",
        "stops": 0,
        "layover_info": "",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZNE1UUWFDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjgxNCJdXV0=",
        "carbon_emissions_g": 85000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [],
        "layover_airports": []
      },
      {
        "airline": "IndiGo",
        "flight_no": "6E 2033",
        "departure_time": "07:10",
        "arrival_time": "11:40",
        "duration_min": 270,
        "price_inr": "₹6,851",
        "stops": 1,
        "layover_info": "1h 20m at AMD",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZzAyUlRJd016TjhOa1UyTnpFeUdnb0l3elVRQUJvRFNVNVNPQnh3OXpnPSIsW1siREVMIiwiMjAyNi0wMy0yMiIsIkFNRCIsbnVsbCwiNkUiLCIyMDMzIl0sWyJBTUQiLCIyMDI2LTAzLTIyIiwiQk9NIixudWxsLCI2RSIsIjY3MTIiXV1d",
        "carbon_emissions_g": 109000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [
          80
        ],
        "layover_airports": [
          "AMD"
        ]
      },
      {
        "airline": "Air India",
        "flight_no": "AI 1745",
        "departure_time": "05:25",
        "arrival_time": "07:50",
        "duration_min": 145,
        "price_inr": "₹7,165",
        "stops": 0,
        "layover_info": "",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1pCU1RFM05EVWFDZ2o5TnhBQUdnTkpUbEk0SEhERk93PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIkFJIiwiMTc0NSJdXV0=",
        "carbon_emissions_g": 99000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [],
        "layover_airports": []
      },
      {
        "airline": "Air India",
        "flight_no": "AI 2975",
        "departure_time": "06:25",
        "arrival_time": "08:45",
        "duration_min": 140,
        "price_inr": "₹7,165",
        "stops": 0,
        "layover_info": "",
        "baggage": "Check airline",
        "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1pCU1RJNU56VWFDZ2o5TnhBQUdnTkpUbEk0SEhERk93PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIkFJIiwiMjk3NSJdXV0=",
        "carbon_emissions_g": 99000,
        "date": "2026-03-22",
        "handoff_url": null,
        "layover_durations_min": [],
        "layover_airports": []
      }
    ],
    "filtered_count": 10,
    "ranked_count": 10,
    "price_insights_str": "",
    "price_analysis_str": "",
    "price_prediction_str": "",
    "normalization": {
      "input": {
        "origin": null,
        "destination": null,
        "date": null,
        "user_query": "Cheap flight from Delhi to Mumbai"
      },
      "after_initial_parse": {
        "origin_iata": "DEL",
        "destination_iata": "BOM"
      },
      "final": {
        "origin_iata": "DEL",
        "destination_iata": "BOM"
      }
    },
    "relaxation_attempts": [
      {
        "step": "strict_filters",
        "matched_count": 10
      }
    ],
    "api_trace": {
      "flight": {
        "request": {
          "departure": "DEL",
          "arrival": "BOM",
          "date": "2026-03-22",
          "intent_date": null,
          "return_date": null
        },
        "raw_count": 10,
        "filtered_count": 10,
        "best_flight_no": "6E 6218",
        "raw_response": [
          {
            "airline": "IndiGo",
            "flight_no": "6E 6218",
            "departure_time": "2026-03-22 06:05",
            "arrival_time": "2026-03-22 08:15",
            "duration_min": 130,
            "price_inr": 6725,
            "stops": 0,
            "layover_info": "",
            "layover_airports": [],
            "layover_durations_min": [],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZeU1UZ2FDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjIxOCJdXV0=",
            "shareable_link": null,
            "carbon_emissions_g": 85000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 06:05"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 08:15"
                },
                "duration": 130,
                "airplane": "Airbus A321neo",
                "airline": "IndiGo",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/6E.png",
                "travel_class": "Economy",
                "flight_number": "6E 6218",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "Carbon emissions estimate: 85 kg"
                ]
              }
            ]
          },
          {
            "airline": "Air India",
            "flight_no": "AI 2951",
            "departure_time": "2026-03-22 13:20",
            "arrival_time": "2026-03-22 15:45",
            "duration_min": 145,
            "price_inr": 7470,
            "stops": 0,
            "layover_info": "",
            "layover_airports": [],
            "layover_durations_min": [],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1pCU1RJNU5URWFDZ2l1T2hBQUdnTkpUbEk0SEhDS1BnPT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIkFJIiwiMjk1MSJdXV0=",
            "shareable_link": null,
            "carbon_emissions_g": 99000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 13:20"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 15:45"
                },
                "duration": 145,
                "airplane": "Airbus A320neo",
                "airline": "Air India",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/AI.png",
                "travel_class": "Economy",
                "flight_number": "AI 2951",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "In-seat USB outlet",
                  "Stream media to your device",
                  "Carbon emissions estimate: 98 kg"
                ],
                "often_delayed_by_over_30_min": true
              }
            ]
          },
          {
            "airline": "SpiceJet",
            "flight_no": "SG 385",
            "departure_time": "2026-03-22 06:30",
            "arrival_time": "2026-03-22 08:35",
            "duration_min": 125,
            "price_inr": 8290,
            "stops": 0,
            "layover_info": "",
            "layover_airports": [],
            "layover_durations_min": [],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1ZUUnpNNE5Sb0tDT0pBRUFBYUEwbE9VamdjY1BKRSIsW1siREVMIiwiMjAyNi0wMy0yMiIsIkJPTSIsbnVsbCwiU0ciLCIzODUiXV1d",
            "shareable_link": null,
            "carbon_emissions_g": 98000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 06:30"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 08:35"
                },
                "duration": 125,
                "airplane": "Boeing 737",
                "airline": "SpiceJet",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/SG.png",
                "travel_class": "Economy",
                "flight_number": "SG 385",
                "legroom": "29 in",
                "extensions": [
                  "Below average legroom (29 in)",
                  "Stream media to your device",
                  "Carbon emissions estimate: 97 kg"
                ],
                "often_delayed_by_over_30_min": true
              }
            ]
          },
          {
            "airline": "IndiGo",
            "flight_no": "6E 6787",
            "departure_time": "2026-03-22 00:30",
            "arrival_time": "2026-03-22 02:45",
            "duration_min": 135,
            "price_inr": 6725,
            "stops": 0,
            "layover_info": "",
            "layover_airports": [],
            "layover_durations_min": [],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZM09EY2FDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjc4NyJdXV0=",
            "shareable_link": null,
            "carbon_emissions_g": 85000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 00:30"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 02:45"
                },
                "duration": 135,
                "airplane": "Airbus A321neo",
                "airline": "IndiGo",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/6E.png",
                "travel_class": "Economy",
                "flight_number": "6E 6787",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "Carbon emissions estimate: 85 kg"
                ],
                "overnight": true,
                "often_delayed_by_over_30_min": true
              }
            ]
          },
          {
            "airline": "IndiGo",
            "flight_no": "6E 2766",
            "departure_time": "2026-03-22 04:00",
            "arrival_time": "2026-03-22 06:20",
            "duration_min": 140,
            "price_inr": 6725,
            "stops": 0,
            "layover_info": "",
            "layover_airports": [],
            "layover_durations_min": [],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRJM05qWWFDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiMjc2NiJdXV0=",
            "shareable_link": null,
            "carbon_emissions_g": 85000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 04:00"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 06:20"
                },
                "duration": 140,
                "airplane": "Airbus A321neo",
                "airline": "IndiGo",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/6E.png",
                "travel_class": "Economy",
                "flight_number": "6E 2766",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "Carbon emissions estimate: 85 kg"
                ],
                "overnight": true
              }
            ]
          },
          {
            "airline": "IndiGo",
            "flight_no": "6E 449",
            "departure_time": "2026-03-22 05:00",
            "arrival_time": "2026-03-22 07:15",
            "duration_min": 135,
            "price_inr": 6725,
            "stops": 0,
            "layover_info": "",
            "layover_airports": [],
            "layover_durations_min": [],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1UyUlRRME9Sb0tDTVUwRUFBYUEwbE9VamdjY1BFMyIsW1siREVMIiwiMjAyNi0wMy0yMiIsIkJPTSIsbnVsbCwiNkUiLCI0NDkiXV1d",
            "shareable_link": null,
            "carbon_emissions_g": 85000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 05:00"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 07:15"
                },
                "duration": 135,
                "airplane": "Airbus A321neo",
                "airline": "IndiGo",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/6E.png",
                "travel_class": "Economy",
                "flight_number": "6E 449",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "Carbon emissions estimate: 85 kg"
                ]
              }
            ]
          },
          {
            "airline": "IndiGo",
            "flight_no": "6E 6814",
            "departure_time": "2026-03-22 07:05",
            "arrival_time": "2026-03-22 09:15",
            "duration_min": 130,
            "price_inr": 6725,
            "stops": 0,
            "layover_info": "",
            "layover_airports": [],
            "layover_durations_min": [],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZNE1UUWFDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjgxNCJdXV0=",
            "shareable_link": null,
            "carbon_emissions_g": 85000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 07:05"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 09:15"
                },
                "duration": 130,
                "airplane": "Airbus A321neo",
                "airline": "IndiGo",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/6E.png",
                "travel_class": "Economy",
                "flight_number": "6E 6814",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "Carbon emissions estimate: 85 kg"
                ]
              }
            ]
          },
          {
            "airline": "IndiGo",
            "flight_no": "6E 2033",
            "departure_time": "2026-03-22 07:10",
            "arrival_time": "2026-03-22 11:40",
            "duration_min": 270,
            "price_inr": 6851,
            "stops": 1,
            "layover_info": "1h 20m at AMD",
            "layover_airports": [
              "AMD"
            ],
            "layover_durations_min": [
              80
            ],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZzAyUlRJd016TjhOa1UyTnpFeUdnb0l3elVRQUJvRFNVNVNPQnh3OXpnPSIsW1siREVMIiwiMjAyNi0wMy0yMiIsIkFNRCIsbnVsbCwiNkUiLCIyMDMzIl0sWyJBTUQiLCIyMDI2LTAzLTIyIiwiQk9NIixudWxsLCI2RSIsIjY3MTIiXV1d",
            "shareable_link": null,
            "carbon_emissions_g": 109000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 07:10"
                },
                "arrival_airport": {
                  "name": "Sardar Vallabhbhai Patel International Airport",
                  "id": "AMD",
                  "time": "2026-03-22 08:55"
                },
                "duration": 105,
                "airplane": "Airbus A321neo",
                "airline": "IndiGo",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/6E.png",
                "travel_class": "Economy",
                "flight_number": "6E 2033",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "Carbon emissions estimate: 61 kg"
                ]
              },
              {
                "departure_airport": {
                  "name": "Sardar Vallabhbhai Patel International Airport",
                  "id": "AMD",
                  "time": "2026-03-22 10:15"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 11:40"
                },
                "duration": 85,
                "airplane": "Airbus A321neo",
                "airline": "IndiGo",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/6E.png",
                "travel_class": "Economy",
                "flight_number": "6E 6712",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "Carbon emissions estimate: 47 kg"
                ],
                "often_delayed_by_over_30_min": true
              }
            ]
          },
          {
            "airline": "Air India",
            "flight_no": "AI 1745",
            "departure_time": "2026-03-22 05:25",
            "arrival_time": "2026-03-22 07:50",
            "duration_min": 145,
            "price_inr": 7165,
            "stops": 0,
            "layover_info": "",
            "layover_airports": [],
            "layover_durations_min": [],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1pCU1RFM05EVWFDZ2o5TnhBQUdnTkpUbEk0SEhERk93PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIkFJIiwiMTc0NSJdXV0=",
            "shareable_link": null,
            "carbon_emissions_g": 99000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 05:25"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 07:50"
                },
                "duration": 145,
                "airplane": "Airbus A320neo",
                "airline": "Air India",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/AI.png",
                "travel_class": "Economy",
                "flight_number": "AI 1745",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "In-seat USB outlet",
                  "Stream media to your device",
                  "Carbon emissions estimate: 98 kg"
                ]
              }
            ]
          },
          {
            "airline": "Air India",
            "flight_no": "AI 2975",
            "departure_time": "2026-03-22 06:25",
            "arrival_time": "2026-03-22 08:45",
            "duration_min": 140,
            "price_inr": 7165,
            "stops": 0,
            "layover_info": "",
            "layover_airports": [],
            "layover_durations_min": [],
            "baggage": "Check airline",
            "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1pCU1RJNU56VWFDZ2o5TnhBQUdnTkpUbEk0SEhERk93PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIkFJIiwiMjk3NSJdXV0=",
            "shareable_link": null,
            "carbon_emissions_g": 99000,
            "legs": [
              {
                "departure_airport": {
                  "name": "Indira Gandhi International Airport",
                  "id": "DEL",
                  "time": "2026-03-22 06:25"
                },
                "arrival_airport": {
                  "name": "Chhatrapati Shivaji Maharaj International Airport Mumbai",
                  "id": "BOM",
                  "time": "2026-03-22 08:45"
                },
                "duration": 140,
                "airplane": "Airbus A320neo",
                "airline": "Air India",
                "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/AI.png",
                "travel_class": "Economy",
                "flight_number": "AI 2975",
                "legroom": "28 in",
                "extensions": [
                  "Below average legroom (28 in)",
                  "In-seat USB outlet",
                  "Stream media to your device",
                  "Carbon emissions estimate: 98 kg"
                ]
              }
            ]
          }
        ]
      },
      "weather": {
        "request": {
          "location": "BOM",
          "date": "2026-03-22"
        },
        "forecast_date": "2026-03-22",
        "condition": "Clear sky",
        "temperature_c": 18,
        "raw_response": {
          "location": "BOM",
          "condition": "Clear sky",
          "temperature_c": 18,
          "feels_like_c": 7.12625,
          "humidity": 66,
          "wind_kph": 11.3,
          "air_quality_index": null,
          "timestamp": 1774180800,
          "temp_min_c": 4.35,
          "temp_max_c": 14.52,
          "has_rain": false,
          "has_snow": false,
          "precipitation_chance": 0,
          "forecast_date": "2026-03-22",
          "location_city": "Mumbai",
          "location_label": "Mumbai (BOM)"
        }
      }
    }
  },
  "return_trip": null,
  "fallback_note": "",
  "weather_present": true,
  "weather_reason": null
}
(venv) sidd@sidd-ASUS-TUF-Gaming-F15-FX507ZE-FX577ZE:~/project/llm-travel-agent$ curl -N -X POST "http://127.0.0.1:8000/ask?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"user_query":"Cheap flight from Delhi to Mumbai","trip_type":"one-way"}'
event: reasoning_step
data: {"step": "Gathering live flight options and destination weather."}

event: flights
data: {"all_flights": [{"airline": "IndiGo", "flight_no": "6E 6218", "departure_time": "06:05", "arrival_time": "08:15", "duration_min": 130, "price_inr": "₹6,725", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZeU1UZ2FDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjIxOCJdXV0=", "carbon_emissions_g": 85000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [], "layover_airports": []}, {"airline": "Air India", "flight_no": "AI 2951", "departure_time": "13:20", "arrival_time": "15:45", "duration_min": 145, "price_inr": "₹7,470", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1pCU1RJNU5URWFDZ2l1T2hBQUdnTkpUbEk0SEhDS1BnPT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIkFJIiwiMjk1MSJdXV0=", "carbon_emissions_g": 99000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [], "layover_airports": []}, {"airline": "SpiceJet", "flight_no": "SG 385", "departure_time": "06:30", "arrival_time": "08:35", "duration_min": 125, "price_inr": "₹8,290", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1ZUUnpNNE5Sb0tDT0pBRUFBYUEwbE9VamdjY1BKRSIsW1siREVMIiwiMjAyNi0wMy0yMiIsIkJPTSIsbnVsbCwiU0ciLCIzODUiXV1d", "carbon_emissions_g": 98000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [], "layover_airports": []}, {"airline": "IndiGo", "flight_no": "6E 6787", "departure_time": "00:30", "arrival_time": "02:45", "duration_min": 135, "price_inr": "₹6,725", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZM09EY2FDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjc4NyJdXV0=", "carbon_emissions_g": 85000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [], "layover_airports": []}, {"airline": "IndiGo", "flight_no": "6E 2766", "departure_time": "04:00", "arrival_time": "06:20", "duration_min": 140, "price_inr": "₹6,725", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRJM05qWWFDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiMjc2NiJdXV0=", "carbon_emissions_g": 85000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [], "layover_airports": []}, {"airline": "IndiGo", "flight_no": "6E 449", "departure_time": "05:00", "arrival_time": "07:15", "duration_min": 135, "price_inr": "₹6,725", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1UyUlRRME9Sb0tDTVUwRUFBYUEwbE9VamdjY1BFMyIsW1siREVMIiwiMjAyNi0wMy0yMiIsIkJPTSIsbnVsbCwiNkUiLCI0NDkiXV1d", "carbon_emissions_g": 85000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [], "layover_airports": []}, {"airline": "IndiGo", "flight_no": "6E 6814", "departure_time": "07:05", "arrival_time": "09:15", "duration_min": 130, "price_inr": "₹6,725", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZNE1UUWFDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjgxNCJdXV0=", "carbon_emissions_g": 85000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [], "layover_airports": []}, {"airline": "IndiGo", "flight_no": "6E 2033", "departure_time": "07:10", "arrival_time": "11:40", "duration_min": 270, "price_inr": "₹6,851", "stops": 1, "layover_info": "1h 20m at AMD", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZzAyUlRJd016TjhOa1UyTnpFeUdnb0l3elVRQUJvRFNVNVNPQnh3OXpnPSIsW1siREVMIiwiMjAyNi0wMy0yMiIsIkFNRCIsbnVsbCwiNkUiLCIyMDMzIl0sWyJBTUQiLCIyMDI2LTAzLTIyIiwiQk9NIixudWxsLCI2RSIsIjY3MTIiXV1d", "carbon_emissions_g": 109000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [80], "layover_airports": ["AMD"]}, {"airline": "Air India", "flight_no": "AI 1745", "departure_time": "05:25", "arrival_time": "07:50", "duration_min": 145, "price_inr": "₹7,165", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1pCU1RFM05EVWFDZ2o5TnhBQUdnTkpUbEk0SEhERk93PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIkFJIiwiMTc0NSJdXV0=", "carbon_emissions_g": 99000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [], "layover_airports": []}, {"airline": "Air India", "flight_no": "AI 2975", "departure_time": "06:25", "arrival_time": "08:45", "duration_min": 140, "price_inr": "₹7,165", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1pCU1RJNU56VWFDZ2o5TnhBQUdnTkpUbEk0SEhERk93PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIkFJIiwiMjk3NSJdXV0=", "carbon_emissions_g": 99000, "date": "2026-03-22", "handoff_url": null, "layover_durations_min": [], "layover_airports": []}], "best_flight": {"airline": "IndiGo", "flight_no": "6E 6218", "departure_time": "06:05", "arrival_time": "08:15", "duration_min": 130, "price_inr": "₹6,725", "stops": 0, "layover_info": "", "baggage": "Check airline", "booking_token": "WyJDalJJTmtsdFJVVlNjazloYmpSQlNVMUZVbEZDUnkwdExTMHRMUzB0TFMxNWJHaGhOVUZCUVVGQlIyMHRjWGhqVEZoMlZWVkJFZ1kyUlRZeU1UZ2FDZ2pGTkJBQUdnTkpUbEk0SEhEeE53PT0iLFtbIkRFTCIsIjIwMjYtMDMtMjIiLCJCT00iLG51bGwsIjZFIiwiNjIxOCJdXV0=", "carbon_emissions_g": 85000, "date": "2026-03-22", "handoff_url": "https://www.google.com/travel/flights?q=Flights%20from%20DEL%20to%20BOM%20on%202026-03-22", "layover_durations_min": [], "layover_airports": []}, "origin_iata": "DEL", "destination_iata": "BOM", "origin_city": "New Delhi", "destination_city": "Mumbai", "origin_label": "New Delhi (DEL)", "destination_label": "Mumbai (BOM)"}

event: weather
data: {"weather": {"location": "BOM", "condition": "Clear sky", "temperature_c": 18, "feels_like_c": 7.12625, "humidity": 66, "wind_kph": 11.3, "air_quality_index": null, "timestamp": 1774180800, "temp_min_c": 4.35, "temp_max_c": 14.52, "has_rain": false, "has_snow": false, "precipitation_chance": 0, "forecast_date": "2026-03-22", "location_city": "Mumbai", "location_label": "Mumbai (BOM)"}}

event: reasoning_step
data: {"step": "Ranked options and selected IndiGo 6E 6218 as the strongest overall fit."}

event: reasoning_step
data: {"step": "Non-stop routing helped reduce transfer risk and overall travel complexity."}

event: reasoning_step
data: {"step": "Destination weather (BOM) looks clear sky around 18.0°C, so comfort and packing guidance were included."}

data: Based

data:  on

data:  your

data:  preference

data:  for

data:  the

data:  cheap

data: est

data:  flight

data: ,

data:  the

data:  best

data:  option

data:  for

data:  your

data:  trip

data:  from

data:  Delhi

data:  to

data:  M

data: umb

data: ai

data:  is

data:  Ind

data: i

data: Go

data:  

data: 6

data: E

data:  

data: 6

data: 2

data: 1

data: 8

data:  on

data:  

data: 2

data: 0

data: 2

data: 6

data: -

data: 0

data: 3

data: -

data: 2

data: 2

data: .

data:  This

data:  non

data: -

data: stop

data:  flight

data:  depart

data: s

data:  at

data:  

data: 0

data: 6

data: :

data: 0

data: 5

data:  and

data:  arrives

data:  at

data:  

data: 0

data: 8

data: :

data: 1

data: 5

data: ,

data:  with

data:  a

data:  duration

data:  of

data:  

data: 1

data: 3

data: 0

data:  minutes

data:  and

data:  a

data:  price

data:  of

data:  

data: ₹

data: 6

data: ,

data: 7

data: 2

data: 5

data: .

data: 
data: 

data: 
data: 

data: This

data:  flight

data: [ERROR] LLM request timed out

data: [DONE_JSON]{"error": "LLM request timed out", "message": "LLM request timed out"}

event: done
data: 

(venv) sidd@sidd-ASUS-TUF-Gaming-F15-FX507ZE-FX577ZE:~/project/llm-travel-agent$ python full_validation.py --mode machine --r 0
START safe_full_validation_report.py
Validation mode: machine
Running pytest (unit tests)
[global ] pytest                         ... PASSED (5.006 s)
Creating /home/sidd/project/llm-travel-agent/.env.tmp for mode=machine
Validation override: USE_CLOUD_LLM=1 (cloud admin enablement; runtime still requires usable provider keys)
Starting local uvicorn server...
Waiting for service readiness at http://localhost:8000/health/ready ...
Service is ready.
Local server healthy and ready.
Running smoke checks (mode=machine) — logs in /home/sidd/project/llm-travel-agent/validation_logs
Using rotation index: 0
[machine] query basic                    ... FAILED (60.062 s)
[machine] query missing date             ... PASSED (59.940 s)
[global ] query natural language date [0] ... PASSED (55.976 s)
[global ] query misspelled city [0]      ... PASSED (50.887 s)
[global ] query round trip duration [0]  ... FAILED (59.871 s)
[global ] query time morning [0]         ... PASSED (50.646 s)
[global ] query price cap [0]            ... FAILED (47.667 s)
[global ] query direct only [0]          ... FAILED (60.045 s)
[global ] query preferred airline [0]    ... FAILED (60.048 s)
[global ] query layover limit [0]        ... FAILED (60.049 s)
[global ] query hand baggage [0]         ... PASSED (58.678 s)
[global ] query stopover via [0]         ... PASSED (60.045 s)
[global ] eco_flight_machine_0           ... PASSED (53.579 s)
[machine] async_parallel_machine still running... (30s elapsed)
[machine] async_parallel_machine still running... (60s elapsed)
[machine] async_parallel_machine still running... (90s elapsed)
[machine] parallel async queries         ... FAILED (118.026 s)
[machine] stream basic                   ... FAILED (20.049 s)
[machine] stream natural language date   ... FAILED (20.048 s)
[machine] health lightweight             ... PASSED (0.010 s)
[machine] health deep                    ... PASSED (4.900 s)
[machine] health keys                    ... PASSED (0.009 s)
[machine] capability constraints         ... FAILED (60.048 s)
[global ] result_machine_integration     ... PASSED (1.001 s)

Summary (non-pass outcomes):
  machine   query basic                          [Validation] LLM generation timed out / returned deterministic fallback
  machine   query round trip duration [0]        [Validation] search_date mismatch: got 2026-03-22, expected 2026-03-20
  machine   query price cap [0]                  [Validation] search_date mismatch: got 2027-03-20, expected 2026-03-20
  machine   query direct only [0]                [Validation] LLM generation timed out / returned deterministic fallback
  machine   query preferred airline [0]          [Validation] Preferred airline unavailable but LLM did not open with disclosure
  machine   query layover limit [0]              [Validation] LLM generation timed out / returned deterministic fallback
  machine   parallel async queries               [Validation] Parallel query 2 returned deterministic fallback (All LLM backends failed)
  machine   stream basic                         [Validation] Stream DONE_JSON contains server error: LLM stream initialization timed out
  machine   stream natural language date         [Validation] Stream DONE_JSON contains server error: LLM stream initialization timed out
  machine   capability constraints               [Validation] LLM generation timed out / returned deterministic fallback

Totals: 20 total, 10 passed, 0 soft-passed (no credit), 10 failed

Detailed counts from consolidated log:
  PASSED lines: 12
  FAILED lines: 10

Full logs available in: /home/sidd/project/llm-travel-agent/validation_logs/validation_run_20260321T200053.log

=== CAPABILITY REPORT ===
planner: FAIL
airline_api: DEGRADED
weather_api: OK
llm_router: DEGRADED
health_system: OK

=== VALIDATION SUMMARY ===
Mode: machine
Total tests: 20
Passed: 10
Soft-passed (no credit): 0
Failed: 10
Duration: 968.35 sec
(venv) sidd@sidd-ASUS-TUF-Gaming-F15-FX507ZE-FX577ZE:~/project/llm-travel-agent$ i have a idea or theory why they all failed timeout bcz the ollama was running on cpu only no gpu activity as i was manually running the  uvicorn api.app:app --reload
for the above query so this fullvalidation closed it abrupty so something boke and ollama was not using gpu only cpu but i have no idea why broke or what broke or will it happen again or i can or not reporduce it   now i will restart and try again for the full validation   
i: command not found
bash: syntax error near unexpected token `above'
(venv) sidd@sidd-ASUS-TUF-Gaming-F15-FX507ZE-FX577ZE:~/project/llm-travel-agent$ 
and the app terminal   (venv) sidd@sidd-ASUS-TUF-Gaming-F15-FX507ZE-FX577ZE:~/project/llm-travel-agent$ uvicorn api.app:app --reload
INFO:     Will watch for changes in these directories: ['/home/sidd/project/llm-travel-agent']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [436715] using StatReload
Gemini helper module not importable - skipping provider: No module named 'gemini_multikey_9_3_helper_script'
INFO:     Started server process [436717]
INFO:     Waiting for application startup.
2026-03-21 19:56:57,836 | WARNING | api.app | Config deprecation: CLOUD_BASE_URL is deprecated. Use CLOUD_PROVIDER_CHAIN/CLOUD_PROVIDER with provider adapters instead. Only legacy async client init reads CLOUD_BASE_URL directly. Compatibility support remains active for now.
2026-03-21 19:56:57,836 | INFO | core.async_llm_client | legacy_async_llm_client_enabled: compatibility initializer active; modern request routing remains router + cloud adapters + key-manager pools.
2026-03-21 19:56:57,946 | INFO | api.app | Startup config summary | llm_mode=ollama_only | cloud_enabled=False | cloud_default_provider=gemini | cloud_provider_chain=gemini,openai | cloud_usable_providers=none | ollama_base_url=http://0.0.0.0:11434 | ollama_model=openhermes | key_manager_lock_backend=file | planner_prewarm=True | deprecated_env_detected=CLOUD_BASE_URL
2026-03-21 19:56:57,946 | INFO | api.app | Registered cloud LLM key event listener
2026-03-21 19:56:57,946 | INFO | api.app | Starting key manager background refresh loop (lock_backend=file).
2026-03-21 19:56:57,946 | INFO | core.api_key_manager | Started API key refresh loop (interval=60s)
2026-03-21 19:56:57,947 | INFO | agents.ollama_client | Ollama request started
2026-03-21 19:56:57,963 | INFO | uvicorn.error | Application startup complete.
2026-03-21 19:57:00,385 | INFO | agents.ollama_client | Ollama request succeeded
2026-03-21 19:57:00,385 | INFO | api.app | Ollama prewarm OK
2026-03-21 19:58:37,816 | INFO | airline_api | SerpAPI request started
2026-03-21 19:58:38,878 | INFO | core.api_key_manager | Key marked exhausted (pending)
2026-03-21 19:58:38,878 | INFO | airline_api | Key quota exhausted
2026-03-21 19:58:38,879 | INFO | agents.cloud_llm | Key exhausted for serpapi:0 – clearing client cache
2026-03-21 19:58:38,879 | INFO | agents.cloud_llm | Key exhausted for serpapi:0 – clearing client cache
2026-03-21 19:58:40,358 | INFO | airline_api | SerpAPI attempt succeeded
2026-03-21 19:58:40,766 | INFO | airline_api | SerpAPI final success
2026-03-21 19:58:40,768 | INFO | planner_agent | Running sequential weather fetches (<=1 key available or only one date)
2026-03-21 19:58:40,768 | INFO | tools.weather_api | [WEATHER] using key index: 0
2026-03-21 19:58:41,278 | INFO | tools.weather_api | Weather API request attempt
2026-03-21 19:58:41,278 | INFO | tools.weather_api | [WEATHER] response status: 200 (key index: 0)
2026-03-21 19:58:41,279 | INFO | tools.weather_api | Weather API request successful (total)
2026-03-21 19:58:41,768 | INFO | tools.weather_api | [WEATHER] using key index: 1
2026-03-21 19:58:41,929 | INFO | tools.weather_api | Weather API request attempt
2026-03-21 19:58:41,929 | INFO | tools.weather_api | [WEATHER] response status: 200 (key index: 1)
2026-03-21 19:58:41,930 | INFO | tools.weather_api | Weather API request successful (total)
2026-03-21 19:58:41,930 | INFO | tools.weather_api | get_forecast_for_date: matched forecast
2026-03-21 19:58:42,404 | WARNING | tools.booking_handoff | SerpAPI booking options fetch failed
2026-03-21 19:58:42,404 | INFO | tools.booking_handoff | booking_token resolution unavailable; falling through to shareable_link/google fallback
2026-03-21 19:58:42,404 | INFO | tools.booking_handoff | Handoff URL falling back to Google Flights search URL
2026-03-21 19:58:42,405 | INFO | planner_agent | Sending prompt to LLM
2026-03-21 19:58:42,406 | INFO | agents.ollama_client | Ollama request started
2026-03-21 19:59:12,407 | INFO | agents.ollama_client | ollama_request_cancelled
2026-03-21 19:59:12,407 | WARNING | agents.llm_router | LLM backend timeout
2026-03-21 19:59:12,408 | WARNING | planner_agent | LLM backends unavailable for explanation
2026-03-21 19:59:12,408 | WARNING | planner_agent | LLM failure count: 1
2026-03-21 19:59:12,450 | INFO | planner_agent | Session saved to database
2026-03-21 19:59:22,291 | INFO | airline_api | Returning cached flight results
2026-03-21 19:59:22,291 | INFO | planner_agent | Running sequential weather fetches (<=1 key available or only one date)
2026-03-21 19:59:22,291 | INFO | tools.weather_api | [WEATHER] using key index: 0
2026-03-21 19:59:22,603 | INFO | tools.weather_api | Weather API request attempt
2026-03-21 19:59:22,603 | INFO | tools.weather_api | [WEATHER] response status: 200 (key index: 0)
2026-03-21 19:59:22,604 | INFO | tools.weather_api | Weather API request successful (total)
2026-03-21 19:59:22,605 | INFO | tools.weather_api | get_forecast_for_date: matched forecast
2026-03-21 19:59:22,956 | WARNING | tools.booking_handoff | SerpAPI booking options fetch failed
2026-03-21 19:59:22,957 | INFO | tools.booking_handoff | booking_token resolution unavailable; falling through to shareable_link/google fallback
2026-03-21 19:59:22,957 | INFO | tools.booking_handoff | Handoff URL falling back to Google Flights search URL
2026-03-21 19:59:22,961 | INFO | agents.ollama_client | Ollama request started
2026-03-21 19:59:41,590 | INFO | agents.llm_router | LLM streaming started
2026-03-21 19:59:52,962 | INFO | core.circuit_breaker | Circuit breaker observed cancellation
2026-03-21 19:59:52,962 | ERROR | agents.ollama_client | Ollama streaming timed out
2026-03-21 19:59:52,962 | INFO | agents.ollama_client | Ollama streaming completed
2026-03-21 19:59:52,963 | ERROR | agents.llm_router | Streaming error
2026-03-21 19:59:52,963 | WARNING | planner_agent | LLM failure count: 2
2026-03-21 19:59:52,963 | ERROR | planner_agent | Error in streaming plan_trip
Traceback (most recent call last):
  File "/home/sidd/project/llm-travel-agent/agents/ollama_client.py", line 519, in token_generator
    async for token in stream_iter:
  File "/home/sidd/project/llm-travel-agent/agents/ollama_client.py", line 299, in _streaming_call
    async for token in ollama_breaker.run_generator_protected(agen_factory):
  File "/home/sidd/project/llm-travel-agent/core/circuit_breaker.py", line 281, in run_generator_protected
    async for item in agen:
  File "/home/sidd/project/llm-travel-agent/agents/ollama_client.py", line 232, in _streaming_call_internal
    async for line in response.aiter_lines():
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpx/_models.py", line 1031, in aiter_lines
    async for text in self.aiter_text():
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpx/_models.py", line 1018, in aiter_text
    async for byte_content in self.aiter_bytes():
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpx/_models.py", line 997, in aiter_bytes
    async for raw_bytes in self.aiter_raw():
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpx/_models.py", line 1055, in aiter_raw
    async for raw_stream_bytes in self.stream:
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpx/_client.py", line 176, in __aiter__
    async for chunk in self._stream:
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpx/_transports/default.py", line 271, in __aiter__
    async for part in self._httpcore_stream:
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpcore/_async/connection_pool.py", line 407, in __aiter__
    raise exc from None
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpcore/_async/connection_pool.py", line 403, in __aiter__
    async for part in self._stream:
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpcore/_async/http11.py", line 342, in __aiter__
    raise exc
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpcore/_async/http11.py", line 334, in __aiter__
    async for chunk in self._connection._receive_response_body(**kwargs):
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpcore/_async/http11.py", line 203, in _receive_response_body
    event = await self._receive_event(timeout=timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpcore/_async/http11.py", line 217, in _receive_event
    data = await self._network_stream.read(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/httpcore/_backends/anyio.py", line 35, in read
    return await self._stream.receive(max_bytes=max_bytes)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sidd/project/llm-travel-agent/venv/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 1254, in receive
    await self._protocol.read_event.wait()
  File "/usr/lib/python3.12/asyncio/locks.py", line 212, in wait
    await fut
asyncio.exceptions.CancelledError

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/sidd/project/llm-travel-agent/agents/ollama_client.py", line 518, in token_generator
    async with asyncio.timeout(timeout):
  File "/usr/lib/python3.12/asyncio/timeouts.py", line 115, in __aexit__
    raise TimeoutError from exc_val
TimeoutError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/sidd/project/llm-travel-agent/agents/planner_agent.py", line 3460, in stream_generator
    async for token in token_stream:
  File "/home/sidd/project/llm-travel-agent/agents/llm_router.py", line 50, in __anext__
    return await self._agen.__anext__()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sidd/project/llm-travel-agent/agents/llm_router.py", line 325, in stream_with_first
    async for chunk in self._stream_with_timeout(gen, timeout, backend_label, request_id):
  File "/home/sidd/project/llm-travel-agent/agents/llm_router.py", line 553, in _stream_with_timeout
    chunk = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/tasks.py", line 520, in wait_for
    return await fut
           ^^^^^^^^^
  File "/home/sidd/project/llm-travel-agent/agents/ollama_client.py", line 542, in token_generator
    raise OllamaError(f"Streaming timed out after {timeout}s")
agents.ollama_client.OllamaError: Streaming timed out after 30.0s
2026-03-21 20:00:58,517 | INFO | uvicorn.error | Shutting down
2026-03-21 20:00:58,618 | INFO | uvicorn.error | Waiting for application shutdown.
2026-03-21 20:00:58,618 | INFO | core.api_key_manager | Stopped API key refresh loop
2026-03-21 20:00:58,618 | INFO | core.api_key_manager | Key manager refresh loop cancelled
2026-03-21 20:00:58,618 | INFO | api.app | Released file lock for key manager refresh.
2026-03-21 20:00:58,619 | INFO | agents.cloud_llm | All cached clients closed
2026-03-21 20:00:58,619 | INFO | agents.cloud_llm | Cloud provider adapter closed
2026-03-21 20:00:58,619 | INFO | agents.cloud_llm | Cloud client shutdown complete
2026-03-21 20:00:58,619 | INFO | uvicorn.error | Application shutdown complete.
2026-03-21 20:00:58,619 | INFO | uvicorn.error | Finished server process [436717]
INFO:     Stopping reloader process [436715]
(venv) sidd@sidd-ASUS-TUF-Gaming-F15-FX507ZE-FX577ZE:~/project/llm-travel-agent    the backend is flaky after the changes made by codex   it not breaking down but something is wrong 