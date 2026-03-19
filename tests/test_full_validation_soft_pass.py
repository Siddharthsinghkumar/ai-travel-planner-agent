import importlib.util
import sys
from pathlib import Path


def _load_full_validation_module(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "full_validation.py"
    monkeypatch.setattr(sys, "argv", ["full_validation.py"])
    spec = importlib.util.spec_from_file_location("full_validation_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_soft_pass_verdict_for_no_credit_tag(monkeypatch):
    module = _load_full_validation_module(monkeypatch)
    verdict = module._determine_validation_verdict(
        "quick_sync_ask_machine",
        124,
        {"provider_no_active_key"},
    )
    assert verdict == module.VERDICT_SOFT_PASS_NO_CREDIT


def test_fail_verdict_for_routing_error_tag(monkeypatch):
    module = _load_full_validation_module(monkeypatch)
    verdict = module._determine_validation_verdict(
        "quick_sync_ask_machine",
        124,
        {"routing_failed"},
    )
    assert verdict == module.VERDICT_FAIL


def test_no_soft_pass_for_non_eligible_test(monkeypatch):
    module = _load_full_validation_module(monkeypatch)
    verdict = module._determine_validation_verdict(
        "health_deep_machine",
        124,
        {"provider_no_active_key"},
    )
    assert verdict == module.VERDICT_FAIL


def test_extracts_backend_status_tags_from_stream_done_json(monkeypatch):
    module = _load_full_validation_module(monkeypatch)
    stream_body = (
        "data: chunk\n\n"
        "data: [DONE_JSON]"
        '{"error":"LLM temporarily unavailable","backend_status":{"failures":[{"reason":"provider_billing_blocked"}]}}'
    )
    tags = module._extract_structured_failure_tags(stream_body, is_stream=True)
    assert "provider_billing_blocked" in tags
