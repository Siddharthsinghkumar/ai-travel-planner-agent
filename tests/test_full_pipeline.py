#tests/test_full_pipeline.py
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

payload = {
    "date": "2026-03-15",
    "user_query": "delhi to mumbai",
    "trip_type": "Business"
}

def test_full_blocking_flow():
    """Phase 1 (non-stream) + Phase 3 timeout + Phase 4 metrics"""
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "best_flight" in data
    assert "weather" in data
    assert "llm_response" in data


def test_streaming_flow():
    """Phase 1 streaming end-to-end"""
    with client.stream("POST", "/ask?stream=true", json=payload) as resp:
        assert resp.status_code == 200
        content = b"".join(resp.iter_raw())
        assert b"[DONE_JSON]" in content


def test_async_job_flow():
    """Phase 2 job queue lifecycle"""
    resp = client.post("/ask?async_job=true", json=payload)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status_resp = client.get(f"/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in ["queued", "running", "done"]


def test_metrics_endpoint():
    """Phase 4 metrics exposure"""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "llm_requests_total" in text
    assert "stream_requests_total" in text
    assert "job_queue_size" in text