from pathlib import Path

from validation.frontend_contract import (
    FrontendValidationContext,
    FrontendValidationRequest,
    coerce_frontend_validation_request,
)
from validation.frontend_validator import FrontendValidator


def _validator() -> FrontendValidator:
    return FrontendValidator(
        frontend_url="http://127.0.0.1:5173",
        frontend_dir=Path("frontend"),
        auto_start_frontend=False,
        fixture_mode_default=True,
    )


def test_endpoint_kind_classification() -> None:
    v = _validator()
    assert v._endpoint_kind_for_request("POST", "http://127.0.0.1:8000/ask?stream=true") == "ask_stream"
    assert v._endpoint_kind_for_request("POST", "http://127.0.0.1:8000/ask?async_job=true") == "ask_async"
    assert v._endpoint_kind_for_request("GET", "http://127.0.0.1:8000/bookings?limit=50") == "bookings_list"
    assert (
        v._endpoint_kind_for_request("POST", "http://127.0.0.1:8000/price-tracking/alerts/7/ack")
        == "price_tracking_alert_ack"
    )
    assert v._endpoint_kind_for_request("POST", "http://127.0.0.1:8000/jobs/job-1/cancel") == "jobs_cancel"
    assert v._endpoint_kind_for_request("GET", "http://127.0.0.1:8000/llm/options") == "llm_options"
    assert v._endpoint_kind_for_request("GET", "http://127.0.0.1:8000/version") == "version"


def test_dev_mode_entry_url_is_enabled_via_expectation_flag() -> None:
    v = _validator()
    entry = v._build_frontend_entry_url(validation_expectations={"enable_dev_mode": True})
    assert entry.endswith("?dev=true")


def test_fixture_scenario_resolution_supports_legacy_aliases() -> None:
    v = _validator()
    assert v._resolve_fixture_scenario_name("mock_stream_success_one_way") == "fixture_stream_one_way"
    assert v._resolve_fixture_scenario_name("fixture_tracking_alerts") == "fixture_tracking_alerts"
    assert v._resolve_fixture_scenario_name("unknown") == ""


def test_frontend_validation_request_contract_strips_legacy_keys() -> None:
    request = FrontendValidationRequest(
        payload={
            "user_query": "hello",
            "__validation_scenario": "mock_stream_success_one_way",
            "__validation_expectations": {"enable_dev_mode": True},
            "__validation_case_name": "legacy_case",
        },
        context=FrontendValidationContext(
            scenario="fixture_non_stream_one_way",
            expectations={"allow_live_backend": False},
            case_name="typed_case",
        ),
    )
    normalized = coerce_frontend_validation_request(request)
    assert "__validation_scenario" not in normalized.payload
    assert "__validation_expectations" not in normalized.payload
    assert "__validation_case_name" not in normalized.payload
    assert normalized.context.scenario == "fixture_non_stream_one_way"
    assert normalized.context.case_name == "typed_case"


def test_query_linked_evidence_requires_more_than_single_marker_hit() -> None:
    v = _validator()
    snapshot = {
        "stream_text": "Route DEL to BOM with ranked options and direct-flight rationale.",
        "reasoning_text": "Reasoning includes Mumbai weather, direct availability, and timing.",
        "weather_text": "Mumbai weather: clear skies.",
        "flight_text": "Delhi to Mumbai flights listed here.",
        "highlight_text": "",
        "booking_panel_text": "",
        "tracking_panel_text": "",
    }
    payload = {"origin": "DEL", "destination": "BOM"}
    evidence = v._query_linked_evidence(payload, "Direct Delhi to Mumbai flight", snapshot, is_multi_leg_query=False)
    assert evidence["ok"] is True
    assert evidence["required_hits"] >= 2
    assert evidence["hit_count"] >= 2
    assert "direct" in evidence["intent_hits"]

    weak_snapshot = dict(snapshot)
    weak_snapshot["reasoning_text"] = "Reasoning includes Mumbai weather and timing."
    weak_snapshot["stream_text"] = "Route DEL to BOM with ranked options and rationale."
    weak = v._query_linked_evidence(payload, "Direct Delhi to Mumbai flight", weak_snapshot, is_multi_leg_query=False)
    assert weak["ok"] is False


def test_source_payload_alignment_handles_structured_fields() -> None:
    v = _validator()
    source = {
        "user_query": "Business class direct flight",
        "origin": "DEL",
        "destination": "BOM",
        "date": "2026-07-18",
        "trip_type": "round-trip",
        "return_date": "2026-07-21",
        "direct_only": True,
        "cabin": "business",
        "baggage_pref": "hand",
    }
    form_state = {
        "user_query": "Business class direct flight",
        "origin": "DEL",
        "destination": "BOM",
        "date": "2026-07-18",
        "trip_type": "round-trip",
        "return_date": "2026-07-21",
        "direct_only": True,
        "cabin": "business",
        "baggage_pref": "hand",
    }
    alignment = v._build_source_payload_alignment(source, form_state)
    assert alignment["matches_source"] is True
    assert alignment["missing_keys"] == []
    assert alignment["mismatched_keys"] == []


def test_network_summary_falls_back_to_unmatched_records_for_completion() -> None:
    v = _validator()
    records = [
        {
            "matches_payload": False,
            "response_status": 200,
            "completed": True,
            "failed": False,
            "is_stream": True,
            "stream_done_marker_checked": True,
            "stream_done_marker_seen": True,
            "stream_done_frame_found": True,
            "stream_done_event": "done",
            "stream_done_json_parsed": True,
            "url": "http://127.0.0.1:8000/ask?stream=true",
        }
    ]
    summary = v._network_summary(records)
    assert summary["payload_matched_request_fired"] is False
    assert summary["request_completed_success"] is True
    assert summary["stream_request_success"] is True
    assert summary["matched_statuses"] == [200]
