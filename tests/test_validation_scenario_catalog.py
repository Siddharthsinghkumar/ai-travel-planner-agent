from validation.scenario_catalog import (
    MODE_FRONTEND_FIXTURE_BROWSER,
    MODE_FRONTEND_REAL_BACKEND_BROWSER,
    classify_frontend_endpoint_request,
    frontend_fixture_scenarios,
    frontend_runtime_cases,
)


def _case_map(mode):
    return {case.case_name: case for case in frontend_runtime_cases(mode=mode)}


def test_classify_frontend_endpoint_request_accepts_non_digit_alert_ack_token():
    endpoint = classify_frontend_endpoint_request(
        "POST",
        "http://127.0.0.1:8000/price-tracking/alerts/fixture-alert-501/ack",
    )
    assert endpoint == "price_tracking_alert_ack"


def test_fixture_dev_mode_operator_endpoints_expect_stream_contract():
    case = _case_map(MODE_FRONTEND_FIXTURE_BROWSER)["frontend_fixture_dev_mode_operator_endpoints"]
    expectations = case.expectations
    assert expectations["expect_stream_request"] is True
    assert "ask_stream" in expectations["required_endpoint_calls"]
    assert "ask.non_stream" not in case.contract_assertions
    assert "ask.stream" in case.contract_assertions


def test_real_backend_dev_mode_operator_endpoints_expect_stream_contract():
    case = _case_map(MODE_FRONTEND_REAL_BACKEND_BROWSER)["frontend_real_backend_dev_mode_operator_endpoints"]
    expectations = case.expectations
    assert expectations["expect_stream_request"] is True
    assert "ask_stream" in expectations["required_endpoint_calls"]
    assert "ask.non_stream" not in case.contract_assertions
    assert "ask.stream" in case.contract_assertions


def test_fixture_tracking_alerts_initial_alerts_is_sequence_of_dict_rows():
    scenario = frontend_fixture_scenarios()["fixture_tracking_alerts"]
    assert isinstance(scenario.initial_alerts, (tuple, list))
    assert scenario.initial_alerts
    assert isinstance(scenario.initial_alerts[0], dict)
    assert scenario.initial_alerts[0]["alert_id"] == 501
