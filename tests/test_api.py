from fastapi.testclient import TestClient
from api.app import app
from datetime import datetime, timedelta

future_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
client = TestClient(app)

def test_ask_endpoint(monkeypatch):

    async def fake_plan_trip(**kwargs):
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)


    response = client.post("/ask", json={
        "origin": "DEL",
        "destination": "BOM",
        "date": future_date,
        "user_query": "Business trip"
    })

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
