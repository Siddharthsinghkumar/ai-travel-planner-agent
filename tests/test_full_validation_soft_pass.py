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


def test_stream_case_fails_when_done_json_missing(monkeypatch):
    module = _load_full_validation_module(monkeypatch)

    class FakeCurlProcess:
        def __init__(self, cmd, stdout=None, stderr=None, text=None):
            self.cmd = cmd
            self.returncode = 0
            out_path = None
            if "-o" in cmd:
                out_path = cmd[cmd.index("-o") + 1]
            if out_path:
                Path(out_path).write_text("data: token chunk only\n\n")

        def communicate(self):
            return "200", ""

    monkeypatch.setattr(module.subprocess, "Popen", FakeCurlProcess)
    module.REPORT.clear()

    status = module.run_and_log(
        "streaming_test_machine",
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            "http://127.0.0.1:8000/ask?stream=true",
            "-H",
            "Content-Type: application/json",
            "-d",
            '{"user_query":"test"}',
        ],
        is_stream=True,
    )

    assert status == 124
    assert module.REPORT[-1]["reason"] == "Stream missing DONE_JSON completion payload"
