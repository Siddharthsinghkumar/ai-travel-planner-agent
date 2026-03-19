import httpx
import pytest
from datetime import datetime, timedelta
from api.app import app

future_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")

@pytest.mark.asyncio
async def test_ask_endpoint(monkeypatch):

    async def fake_plan_trip(**kwargs):
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/ask", json={
            "origin": "DEL",
            "destination": "BOM",
            "date": future_date,
            "user_query": "Business trip"
        })

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
