from validation.scenario_catalog import (
    MODE_FRONTEND_FIXTURE_BROWSER,
    MODE_FRONTEND_REAL_BACKEND_BROWSER,
    MODE_LIVE_CANARY_BROWSER,
    SOFT_PASS_LIVE_ONLY,
    classify_frontend_endpoint_request,
    frontend_fixture_scenarios,
    frontend_runtime_cases,
    known_features,
    known_mode_buckets,
    resolve_frontend_fixture_scenario_name,
    validation_meta_for_prefix,
)


def test_frontend_fixture_runtime_matrix_has_expected_core_journeys() -> None:
    cases = frontend_runtime_cases(mode=MODE_FRONTEND_FIXTURE_BROWSER, include_live_canary=False)
    names = {case.case_name for case in cases}

    assert "frontend_fixture_stream_one_way" in names
    assert "frontend_fixture_non_stream_one_way" in names
    assert "frontend_fixture_round_trip" in names
    assert "frontend_fixture_booking_hold_cancel" in names
    assert "frontend_fixture_tracking_alerts" in names
    assert "frontend_fixture_async_jobs" in names
    assert "frontend_fixture_dev_mode_operator_endpoints" in names


def test_frontend_real_backend_runtime_matrix_has_broader_browser_journeys() -> None:
    cases = frontend_runtime_cases(mode=MODE_FRONTEND_REAL_BACKEND_BROWSER, include_live_canary=False)
    names = {case.case_name for case in cases}

    assert "frontend_real_backend_ask_non_stream_basic" in names
    assert "frontend_real_backend_ask_stream_basic" in names
    assert "frontend_real_backend_direct_truth" in names
    assert "frontend_real_backend_cabin_truth" in names
    assert "frontend_real_backend_booking_hold" in names
    assert "frontend_real_backend_booking_cancel" in names
    assert "frontend_real_backend_alerts_list" in names
    assert "frontend_real_backend_alert_ack" in names
    assert "frontend_real_backend_async_jobs" in names
    assert "frontend_real_backend_dev_mode_operator_endpoints" in names


def test_frontend_real_backend_matrix_can_include_live_canary_cases() -> None:
    cases = frontend_runtime_cases(mode=MODE_FRONTEND_REAL_BACKEND_BROWSER, include_live_canary=True)
    names = {case.case_name for case in cases}

    assert "frontend_live_canary_direct_one_way" in names
    assert "frontend_live_canary_seller_diversity" in names


def test_frontend_fixture_catalog_contains_runtime_case_fixtures() -> None:
    fixtures = frontend_fixture_scenarios()
    cases = frontend_runtime_cases(mode=MODE_FRONTEND_FIXTURE_BROWSER, include_live_canary=False)

    for case in cases:
        if case.fixture_scenario:
            assert case.fixture_scenario in fixtures, f"Missing fixture scenario: {case.fixture_scenario}"


def test_validation_meta_exposes_mode_bucket_and_soft_pass_policy() -> None:
    fixture_meta = validation_meta_for_prefix("frontend_fixture_stream_one_way_machine")
    assert fixture_meta.validation_type == "frontend-fixture"
    assert fixture_meta.mode_bucket == MODE_FRONTEND_FIXTURE_BROWSER

    live_meta = validation_meta_for_prefix("frontend_live_canary_direct_one_way_machine")
    assert live_meta.validation_type == "live-canary"
    assert live_meta.mode_bucket == MODE_LIVE_CANARY_BROWSER
    assert live_meta.soft_pass_policy == SOFT_PASS_LIVE_ONLY


def test_known_features_and_modes_include_new_dimensions() -> None:
    features = set(known_features())
    modes = set(known_mode_buckets())

    assert "seller.ota_diversity" in features
    assert "booking.navigation.real_provider_browser_proof" in features
    assert "booking.policy.no_google_fallback" in features
    assert "frontend.ui.fields.cabin" in features
    assert "frontend.ui.fields.return_date" in features

    assert MODE_FRONTEND_FIXTURE_BROWSER in modes
    assert MODE_FRONTEND_REAL_BACKEND_BROWSER in modes
    assert MODE_LIVE_CANARY_BROWSER in modes


def test_shared_fixture_alias_resolver_maps_legacy_names() -> None:
    fixtures = frontend_fixture_scenarios()
    assert resolve_frontend_fixture_scenario_name("mock_stream_success_one_way", fixture_catalog=fixtures) == "fixture_stream_one_way"
    assert resolve_frontend_fixture_scenario_name("fixture_tracking_alerts", fixture_catalog=fixtures) == "fixture_tracking_alerts"
    assert resolve_frontend_fixture_scenario_name("does_not_exist", fixture_catalog=fixtures) == ""


def test_shared_endpoint_classifier_covers_operator_paths() -> None:
    assert classify_frontend_endpoint_request("GET", "http://127.0.0.1:8000/llm/options") == "llm_options"
    assert classify_frontend_endpoint_request("GET", "http://127.0.0.1:8000/version") == "version"
