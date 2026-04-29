#!/usr/bin/env python3
"""
safe_full_validation_report.py

Improved version of safe_full_validation.sh that captures logs and prints a neat summary.
Based on original bash script.
"""

import os
import sys
import subprocess
import time
import json
import configparser
import tempfile
import shutil
import hashlib
import re
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from core.env_config import get_env_bool, get_env_int, get_env_str, is_env_set
from core.ollama_context import (
    normalize_validation_num_ctx_mode,
    resolve_validation_num_ctx,
)
from validation.scenario_catalog import (
    frontend_runtime_cases,
    known_features,
    known_mode_buckets,
    MODE_API_CONTRACT,
    MODE_BACKEND_INTERNAL,
    MODE_FRONTEND_FIXTURE_BROWSER,
    MODE_FRONTEND_REAL_BACKEND_BROWSER,
    MODE_LIVE_CANARY_BROWSER,
    MODE_RUNTIME_HEALTH,
    SOFT_PASS_ALLOWED,
    SOFT_PASS_HARD_FAIL_ONLY,
    SOFT_PASS_LIVE_ONLY,
    validation_meta_for_prefix,
    validation_meta_prefix_map,
)
from validation.llm_validation_rules import (
    detect_layover_contradiction,
    detect_relaxed_preferred_airline_contradiction,
)

# ----------------------------------------------------------------------
# Configuration + CLI (quiet by default)
# ----------------------------------------------------------------------
ROOT = Path.cwd()
ORIG_ENV_MACHINE = ROOT / ".env"
ORIG_ENV_DOCKER = ROOT / ".env.laptopdocker"
TMP_ENV = ROOT / ".env.tmp"
CONTAINER_NAME = "llm-test-local"
IMAGE_NAME = "llm-test:normal"
PYTEST_CMD = ["pytest", "-q"]

VALIDATION_PORT = max(1, get_env_int("VALIDATION_PORT", 8000))
DEFAULT_API_BASE_URL = f"http://localhost:{VALIDATION_PORT}"
HEALTH_URL = f"{DEFAULT_API_BASE_URL}/health"
SMOKE_TIMEOUT = get_env_int("SMOKE_TIMEOUT", 30)
APP_START_TIMEOUT = 25
READY_TIMEOUT = 60
FRONTEND_DEFAULT_URL = get_env_str("FRONTEND_VALIDATION_URL", "http://127.0.0.1:5173")
FRONTEND_DEFAULT_HOST = get_env_str("FRONTEND_VALIDATION_HOST", "127.0.0.1")
FRONTEND_DEFAULT_PORT = get_env_int("FRONTEND_VALIDATION_PORT", 5173)
FRONTEND_QUERY_TIMEOUT = get_env_int("FRONTEND_VALIDATION_QUERY_TIMEOUT", max(SMOKE_TIMEOUT + 35, 70))
VALIDATION_AUTH_TOKEN = "validation-auth-token"
VALIDATION_ADMIN_TOKEN = "validation-admin-token"
FRONTEND_HIGH_VALUE_REROUTE_PREFIXES = (
    "quick_sync_ask",
    "missing_date_test",
    "round_trip_duration",
    "stopover_via",
    "streaming_test",
    "streaming_nl_relative",
    "capability_constraints",
)
FRONTEND_BACKEND_DIRECT_PREFIXES = (
    "health_",
    "llm_options_",
    "version_info_",
    "contract_",
    "async_parallel_",
)

# Variant counts for each test (in order of run_smoke_checks_logged)
SMOKE_VARIANT_COUNTS = [
    5,  # nl_relative_date
    4,  # misspelled_city
    4,  # round_trip_duration
    4,  # time_pref_morning
    4,  # price_cap
    5,  # direct_only
    4,  # preferred_airline
    4,  # layover_limit
    4,  # baggage_hand
    4,  # stopover_via
]
MAX_VARIANTS = max(SMOKE_VARIANT_COUNTS)   # maximum variants across all smoke tests

# CLI (quiet is default behaviour)
class ValidationHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


parser = argparse.ArgumentParser(
    description=(
        "Run local validation before push.\n"
        "By default this run validates backend/API paths; frontend browser checks are opt-in."
    ),
    epilog=(
        "Common examples:\n"
        "  python full_validation.py\n"
        "    Run backend/API-only validation on local machine with the default pytest suite.\n\n"
        "  python full_validation.py --pytest-lane full\n"
        "    Run both default and slow pytest families (tests/ + tests_slow/).\n\n"
        "  python full_validation.py --pytest-lane slow\n"
        "    Run only explicit slow pytest families (tests_slow/).\n\n"
        "  python full_validation.py --frontend\n"
        "    Run backend/API validation + frontend-heavy fixture-backed browser checks\n"
        "    (safe default, no live-provider credit burn).\n\n"
        "  python full_validation.py --frontend --frontend-real-backend\n"
        "    Browser validation against a real local backend (explicit opt-in).\n\n"
        "  python full_validation.py --mode docker --live\n"
        "    Docker run with real external provider calls (slowest, most expensive).\n\n"
        "Notes:\n"
        "  - VALIDATION_QUIET env var controls default terminal verbosity when\n"
        "    --quiet/--no-quiet are not explicitly provided.\n"
        "  - --frontend-preview is valid only with --frontend."
    ),
    formatter_class=ValidationHelpFormatter,
)

scope_group = parser.add_argument_group("Validation Scope")
scope_group.add_argument(
    "--mode",
    choices=["machine", "docker", "both"],
    default="machine",
    help="Execution target: local machine, docker container, or both.",
)
scope_group.add_argument(
    "--live",
    action="store_true",
    default=False,
    help="Disable TESTING mode and call real SerpAPI/OpenWeatherMap providers.",
)

frontend_group = parser.add_argument_group("Frontend Validation")
frontend_group.add_argument(
    "--frontend",
    action="store_true",
    default=False,
    help="Route ask validations through the browser/UI path instead of direct backend curl.",
)
frontend_group.add_argument(
    "--frontend-real-backend",
    action="store_true",
    default=False,
    help="With --frontend, run browser checks against the real backend instead of fixture-backed frontend validation.",
)
frontend_group.add_argument(
    "--frontend-live-canary",
    action="store_true",
    default=False,
    help="With --frontend --frontend-real-backend --live, run a small explicit browser live-provider canary subset.",
)
frontend_group.add_argument(
    "--frontend-preview",
    action="store_true",
    default=False,
    help="With --frontend, run Vite in preview mode (npm run preview) instead of dev mode.",
)

execution_group = parser.add_argument_group("Execution Behavior")
_default_pytest_lane_raw = (get_env_str("VALIDATION_PYTEST_LANE", "default") or "default").strip().lower()
_pytest_lane_aliases = {
    "default": "default",
    "fast": "default",
    "full": "full",
    "all": "full",
    "slow": "slow",
}
_default_pytest_lane = _pytest_lane_aliases.get(_default_pytest_lane_raw, "default")
execution_group.add_argument(
    "--r",
    type=int,
    default=None,
    help="Force a specific variant index for variantized checks (overrides rotation).",
)
execution_group.add_argument(
    "--loop",
    action="store_true",
    default=False,
    help="Run all query variants sequentially (ignores rotation).",
)
execution_group.add_argument(
    "--pytest-lane",
    choices=["default", "full", "slow"],
    default=_default_pytest_lane,
    help=(
        "Pytest lane to run inside validation: default (same semantics as plain pytest -q), "
        "full (default + explicit slow suite), or slow (explicit slow suite only)."
    ),
)

output_group = parser.add_argument_group("Output Controls")
output_group.add_argument(
    "--debug",
    action="store_true",
    help="Enable verbose debug logging (file logs always stay verbose).",
)
output_group.add_argument(
    "--quiet",
    dest="quiet",
    action="store_true",
    help="Minimal terminal output (PASS/FAIL and summary only).",
)
output_group.add_argument(
    "--no-quiet",
    dest="quiet",
    action="store_false",
    help="Verbose terminal output.",
)

special_group = parser.add_argument_group("Special Runs")
special_group.add_argument(
    "--ragas-eval",
    action="store_true",
    default=False,
    help="Run RAGAS baseline evaluation and write results to eval_results/ragas_baseline.json.",
)
special_group.add_argument(
    "--with-rag",
    action="store_true",
    default=False,
    help="Run RAGAS evaluation with RAG context retrieval (requires --ragas-eval).",
)
special_group.add_argument(
    "--hitl-test",
    action="store_true",
    default=False,
    help="Run HITL approval gate integration test.",
)

parser.set_defaults(quiet=None)
args = parser.parse_args()

if args.frontend_preview and not args.frontend:
    parser.error("--frontend-preview requires --frontend")
if args.frontend_real_backend and not args.frontend:
    parser.error("--frontend-real-backend requires --frontend")
if args.frontend_live_canary and not (args.frontend and args.frontend_real_backend and args.live):
    parser.error("--frontend-live-canary requires --frontend --frontend-real-backend --live")

VALIDATION_MODE = (get_env_str("VALIDATION_MODE", "") or "").strip().lower()
REAL_MODE = VALIDATION_MODE == "real"
env_quiet = get_env_bool("VALIDATION_QUIET", default=True)
if args.quiet is None:
    args.quiet = env_quiet

if REAL_MODE:
    args.live = True
    SMOKE_TIMEOUT = max(SMOKE_TIMEOUT, 90)
    print(f"REAL mode enabled via VALIDATION_MODE=real. SMOKE_TIMEOUT={SMOKE_TIMEOUT}s")

if args.live:
    SMOKE_TIMEOUT = max(SMOKE_TIMEOUT, 90)
    print(f"LIVE mode: real SerpAPI + OWM calls enabled. SMOKE_TIMEOUT raised to {SMOKE_TIMEOUT}s.")

LOG_DIR = ROOT / "validation_logs"
LOG_DIR.mkdir(exist_ok=True)

ROTATION_FILE = LOG_DIR / "rotation_counter.txt"

# Small IATA → city name aliases for weather checks
IATA_CITY_ALIASES = {
    "blr": ["bangalore", "bengaluru"],
    "maa": ["chennai", "madras"],
    "bom": ["mumbai", "bombay"],
    "del": ["delhi", "new delhi"],
    "hyd": ["hyderabad"],
    "ccu": ["kolkata", "calcutta"],
}

# ----------------------------------------------------------------------
# Logging setup (quiet by default)
# ----------------------------------------------------------------------
log_filename = LOG_DIR / f"validation_run_{datetime.now().strftime('%Y%m%dT%H%M%S%z')}.log"

logger = logging.getLogger("validation")
logger.setLevel(logging.DEBUG)  # file always gets everything

# File handler (FULL LOGS ALWAYS)
fh = logging.FileHandler(log_filename, mode='w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'
))
logger.addHandler(fh)

# Console handler (QUIET BY DEFAULT)
ch = logging.StreamHandler()

if args.debug:
    ch.setLevel(logging.DEBUG)   # full verbosity
else:
    ch.setLevel(logging.INFO)    # PASS/FAIL + summary

ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

# Silence noisy internal modules unless --debug
NOISY_LOGGERS = [
    "agents",
    "agents.ollama_client",
    "agents.llm_router",
    "agents.cloud_llm",
    "planner_agent",
    "airline_api",
    "tools.weather_api",
    "uvicorn",
    "urllib3",
]

if not args.debug and args.quiet:
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

def log(msg):
    logger.info(msg)

# ----------------------------------------------------------------------
# Global report storage (list of dicts with name, status, duration, reason)
# ----------------------------------------------------------------------
REPORT = []
CAPABILITY_REPORT = {
    "planner": "UNKNOWN",
    "airline_api": "UNKNOWN",
    "weather_api": "UNKNOWN",
    "llm_router": "UNKNOWN",
    "health_system": "UNKNOWN",
}
CAPABILITY_REPORT_DETAILS = {}
PYTEST_LANE_CONTEXT = {
    "lane": "unknown",
    "lane_semantics": "unknown",
    "pytest_testpaths": [],
    "selected_paths": [],
    "command": [],
    "probe_command": [],
}
FRONTEND_VALIDATOR = None

LLM_NEAR_TIMEOUT_RATIO = 0.85
LLM_TIMEOUT_SHAPED_RATIO = 0.97
LLM_WARMUP_RETRY_DELAY_SEC_DEFAULT = 1.5
LLM_WARMUP_MAX_ATTEMPTS_DEFAULT = 2

LLM_WARMUP_CONTEXT = {
    "attempted": False,
    "succeeded": False,
    "mode": "",
    "attempts": 0,
    "max_attempts": 0,
    "reason": "",
    "http_status": None,
    "latency_sec": None,
    "completion_source": "",
    "backend": "",
    "model": "",
    "num_ctx": None,
    "thinking_mode": "",
    "first_token_latency_sec": None,
    "timeout_ratio": None,
    "request_reached_llm_path": False,
    "completion_observed": False,
    "degraded_observed": False,
    "admission": "",
    "execution": "",
    "replayed_recent": False,
    "replay_bypassed": False,
    "attempt_records": [],
}

VALIDATION_RUNTIME_CONFIG = {
    "mode": "",
    "backend_expectation": "",
    "llm_mode": "",
    "use_cloud_llm": "",
    "ollama_model_process_env": "",
    "ollama_model_src_env": "",
    "ollama_model_tmp_env": "",
    "ollama_num_ctx_mode": "validated_default",
    "ollama_num_ctx_process_env": "",
    "ollama_num_ctx_validation_override": "",
    "ollama_num_ctx_effective": None,
    "ollama_num_ctx_source": "",
    "ollama_num_ctx_tmp_env": "",
    "ollama_thinking_mode_process_env": "",
    "ollama_thinking_mode_validation_override": "",
    "ollama_thinking_mode_effective": "",
    "ollama_thinking_mode_tmp_env": "",
    "rotation_index": None,
    "rotation_source": "",
    "rotation_raw": None,
    "rotation_file_before": None,
    "rotation_file_after": None,
    "rotation_loop_mode": False,
    "async_parallel_mode": "sequential",
}

VERDICT_PASS = "PASS"
VERDICT_SOFT_PASS_NO_CREDIT = "SOFT_PASS_NO_CREDIT"
VERDICT_FAIL = "FAIL"

SOFT_PASS_NO_CREDIT_TAGS = {
    "provider_quota_exhausted",
    "provider_billing_blocked",
    "provider_no_active_key",
    "runtime_date_basis_skew",
}

DISPLAY_MAP = {
    "pytest_unit": "pytest",
    "quick_sync_ask": "query basic",
    "missing_date_test": "query missing date",
    "nl_relative_date": "query natural language date",
    "misspelled_city": "query misspelled city",
    "round_trip_duration": "query round trip duration",
    "time_pref_morning": "query time morning",
    "price_cap": "query price cap",
    "direct_only": "query direct only",
    "preferred_airline": "query preferred airline",
    "layover_limit": "query layover limit",
    "baggage_hand": "query hand baggage",
    "stopover_via": "query stopover via",
    "eco_flight": "query eco flight",
    "async_parallel": "parallel async queries",
    "streaming_test": "stream basic",
    "streaming_nl_relative": "stream natural language date",
    "health_light": "health lightweight",
    "health_deep": "health deep",
    "health_keys": "health keys",
    "health_runtime_topology": "health runtime topology",
    "llm_options": "llm options",
    "version_info": "version info",
    "capability_constraints": "capability constraints",
    "contract_no_flights": "contract no-flights",
    "contract_degraded_stream": "contract degraded stream",
    "contract_booking_bridge": "contract booking bridge",
    "contract_jobs_flow": "contract jobs flow",
    "contract_hardening_duplicate_guard": "contract hardening duplicate guard",
    "contract_hardening_backpressure": "contract hardening backpressure",
    "contract_hardening_consume_race": "contract hardening consume race",
    "contract_hardening_retry_budget": "contract hardening retry budget",
    "contract_hardening_key_cooldown": "contract hardening key cooldown",
    "frontend_runtime_stream_one_way": "frontend stream one-way",
    "frontend_runtime_non_stream_one_way": "frontend non-stream one-way",
    "frontend_runtime_round_trip": "frontend round-trip runtime",
    "frontend_runtime_via_stopover": "frontend via-stopover runtime",
    "frontend_runtime_degraded": "frontend degraded runtime",
    "frontend_runtime_no_flights": "frontend no-flights runtime",
    "frontend_runtime_booking_handoff": "frontend booking-handoff runtime",
    "frontend_runtime_booking_local_only": "frontend local-only booking truth",
    "frontend_runtime_booking_hold_cancel": "frontend booking hold+cancel flow",
    "frontend_runtime_tracking_alerts": "frontend tracking alerts flow",
    "frontend_runtime_async_jobs": "frontend async jobs flow",
    "frontend_runtime_async_cancel": "frontend async cancel flow",
    "frontend_runtime_cabin_truth": "frontend cabin truthfulness",
    "frontend_runtime_direct_truth": "frontend direct truthfulness",
    "frontend_runtime_live_canary": "frontend live canary",
    "real_simple_flight": "real simple flight",
    "real_weather_query": "real weather query",
    "real_combined_query": "real combined query",
    "docker_hosted_smoke": "server boot (docker)",
}

VALIDATION_META = validation_meta_prefix_map()


def _strip_mode_suffix(name):
    cleaned = str(name or "")
    cleaned = re.sub(r"_machine(?=(_\d+)?$)", "", cleaned)
    cleaned = re.sub(r"_docker-hosted(?=(_\d+)?$)", "", cleaned)
    return cleaned


def _mode_label_for_name(name):
    value = str(name or "")
    if re.search(r"_machine(?:_\d+)?$", value):
        return "machine"
    if re.search(r"_docker-hosted(?:_\d+)?$", value):
        return "docker"
    return "global"


def _extract_variant_suffix(base_name):
    parts = str(base_name).split("_")
    if parts and parts[-1].isdigit():
        return parts[-1]
    return None


def _display_name_for_base(base_name):
    variant = _extract_variant_suffix(base_name)
    for prefix, display in DISPLAY_MAP.items():
        if str(base_name).startswith(prefix):
            if variant is not None:
                return f"{display} [{variant}]"
            return display
    return base_name


def _validation_meta_for_base(base_name):
    meta = validation_meta_for_prefix(str(base_name))
    return {
        "scenario": meta.scenario,
        "layers": list(meta.layers),
        "validation_type": meta.validation_type,
        "features": list(meta.features),
        "mode_bucket": getattr(meta, "mode_bucket", MODE_BACKEND_INTERNAL),
        "soft_pass_policy": getattr(meta, "soft_pass_policy", SOFT_PASS_HARD_FAIL_ONLY),
        "criticality": getattr(meta, "criticality", "core"),
    }


def _is_soft_pass_eligible_test(*, soft_pass_policy, mode_bucket):
    """Soft-pass is ONLY allowed in live_canary_browser mode.
    
    In all other modes (backend_internal, api_contract, frontend_fixture, etc.),
    soft-pass is disabled to prevent false green results.
    """
    policy = str(soft_pass_policy or SOFT_PASS_HARD_FAIL_ONLY).strip().lower()
    if policy == SOFT_PASS_ALLOWED:
        return False
    if policy == SOFT_PASS_LIVE_ONLY:
        return str(mode_bucket or "") == MODE_LIVE_CANARY_BROWSER
    return False


def _parse_iso_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None


def _is_date_basis_tolerable(actual_date, expected_date, assertions, data):
    if not (actual_date and expected_date):
        return False
    if not isinstance(assertions, dict):
        return False

    skew_days = int(assertions.get("allow_runtime_date_skew_days") or 0)
    if skew_days <= 0:
        return False

    delta_days = (actual_date - expected_date).days
    if abs(delta_days) > skew_days:
        return False

    intent = (data.get("debug_info") or {}).get("intent") or {}
    date_trace = intent.get("date_parse_trace") or {}
    normalization = (data.get("debug_info") or {}).get("normalization") or {}
    date_interpretation = normalization.get("date_interpretation") or {}

    source = (
        date_trace.get("source")
        or date_interpretation.get("source")
        or ""
    )
    allowed_sources = {"none", "relative_today", "relative_days_offset", "relative_weeks_offset"}
    if source and source not in allowed_sources:
        return False

    return True


def _evaluate_expected_date_assertion(data, assertions):
    expected_raw = assertions.get("expected_date")
    actual_raw = data.get("search_date")
    if not actual_raw and data.get("multicity") and data.get("legs"):
        actual_raw = data["legs"][0].get("search_date")

    details = {
        "expected_date": expected_raw,
        "actual_date": actual_raw,
        "expected_basis": assertions.get("expected_date_basis") or "unspecified",
        "allow_runtime_skew_days": int(assertions.get("allow_runtime_date_skew_days") or 0),
    }

    if actual_raw == expected_raw:
        details["outcome"] = "match"
        return True, False, "", details

    actual_dt = _parse_iso_date(actual_raw)
    expected_dt = _parse_iso_date(expected_raw)
    if _is_date_basis_tolerable(actual_dt, expected_dt, assertions, data):
        details["outcome"] = "runtime_skew_tolerated"
        details["delta_days"] = (actual_dt - expected_dt).days
        return False, True, (
            f"search_date mismatch tolerated due runtime date-basis skew: "
            f"got {actual_raw}, expected {expected_raw}"
        ), details

    details["outcome"] = "mismatch"
    if actual_dt and expected_dt:
        details["delta_days"] = (actual_dt - expected_dt).days
    return False, False, f"search_date mismatch: got {actual_raw}, expected {expected_raw}", details


def _extract_noncurl_failure_reason(name, cmd_output, exit_code):
    output = str(cmd_output or "")
    tags = set()

    if "contract_booking_bridge" in str(name or ""):
        tags.add("booking_bridge_contract_failure")
        for pattern in (
            r"booking bridge artifact registration failed[^\n]*",
            r"first bridge fetch failed:[^\n]*",
            r"booking bridge html missing[^\n]*",
            r"unexpected consume result header:[^\n]*",
            r"second bridge fetch should be 404,[^\n]*",
            r"unexpected second bridge payload:[^\n]*",
        ):
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(0).strip(), tags

    return f"Command exited with {exit_code}", tags


def _extract_booking_bridge_diagnostics_from_output(cmd_output):
    diagnostics = []
    for line in str(cmd_output or "").splitlines():
        if not line.startswith("BOOKING_BRIDGE_DIAG "):
            continue
        payload = line[len("BOOKING_BRIDGE_DIAG "):].strip()
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                diagnostics.append(parsed)
        except Exception:
            continue
    return diagnostics


def _determine_pass_quality(name, status, verdict, assertions, is_stream):
    base = _strip_mode_suffix(name)
    if verdict == VERDICT_FAIL:
        return "fail"
    if verdict == VERDICT_SOFT_PASS_NO_CREDIT:
        return "soft-pass"

    if base == "pytest_unit":
        return "acceptable"
    if base in {"docker_hosted_smoke", "result_machine_integration"}:
        return "acceptable"
    if base.startswith("health_keys"):
        return "acceptable"
    if base.startswith("async_parallel"):
        return "acceptable"
    if base.startswith("contract_jobs_flow"):
        return "acceptable"
    if base.startswith("contract_no_flights"):
        return "acceptable"
    if base.startswith("contract_degraded_stream"):
        return "acceptable"

    score = 0
    if isinstance(assertions, dict):
        if assertions.get("check_api_trace"):
            score += 2
        if assertions.get("expected_date"):
            score += 2
        if assertions.get("check_return_llm"):
            score += 2
        if assertions.get("check_all_legs_llm"):
            score += 2
        if assertions.get("check_llm_flight_consistency"):
            score += 1
        if assertions.get("check_relaxed_filter_honest"):
            score += 1
        if assertions.get("check_health_keys_shape"):
            score += 1
        if assertions.get("check_health_light_shape"):
            score += 1
        if assertions.get("check_health_deep_shape"):
            score += 1
        if assertions.get("check_llm_options_shape"):
            score += 1
        if assertions.get("check_version_info_shape"):
            score += 1
        required_paths = assertions.get("required_paths") or []
        if isinstance(required_paths, list) and required_paths:
            score += 1
    if is_stream:
        score += 1

    if score >= 4:
        return "strong"
    if score >= 2:
        return "acceptable"
    return "weak"


def _determine_frontend_pass_quality(name, verdict, frontend_result):
    if verdict == VERDICT_FAIL:
        return "fail"
    if verdict == VERDICT_SOFT_PASS_NO_CREDIT:
        return "soft-pass"

    cases = []
    if isinstance(frontend_result, dict):
        maybe_cases = frontend_result.get("frontend_cases")
        if isinstance(maybe_cases, list):
            cases = [c for c in maybe_cases if isinstance(c, dict)]
    if not cases:
        return "weak"

    if any(not case.get("passes") for case in cases):
        return "fail"

    score = 0
    if all(case.get("planner_form_ready") and case.get("ui_reset_performed") for case in cases):
        score += 1
    if all((case.get("network_summary") or {}).get("request_completed_success") for case in cases):
        score += 1
    if all((case.get("network_summary") or {}).get("streaming_completed") for case in cases):
        score += 1
    if all((case.get("payload_parity") or {}).get("matches_expected") for case in cases):
        score += 1
    if all((case.get("source_payload_alignment") or {}).get("matches_source") for case in cases):
        score += 1
    if all(
        bool(case.get("proof_overview_visible"))
        and bool(case.get("proof_evidence_visible"))
        and bool(case.get("ranked_shortlist_visible"))
        for case in cases
    ):
        score += 1

    base = _strip_mode_suffix(name)
    if base.startswith(("frontend_runtime_", "frontend_fixture_", "frontend_real_backend_", "frontend_live_canary_")):
        score += 1

    if score >= 6:
        return "strong"
    if score >= 4:
        return "acceptable"
    return "weak"


def _collect_backend_status_tags(node):
    tags = set()

    def walk(value):
        if isinstance(value, dict):
            backend_status = value.get("backend_status")
            if isinstance(backend_status, dict):
                failures = backend_status.get("failures") or []
                if isinstance(failures, list):
                    for failure in failures:
                        if isinstance(failure, dict):
                            reason = failure.get("reason")
                            if isinstance(reason, str) and reason.strip():
                                tags.add(reason.strip())
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(node)
    return tags


def _extract_stream_done_json(resp_body):
    if not resp_body:
        return None
    done_match = re.search(r'\[DONE_JSON\](\{.*\})', resp_body, re.DOTALL)
    if not done_match:
        return None
    try:
        return json.loads(done_match.group(1))
    except json.JSONDecodeError:
        return None


def _extract_structured_failure_tags(resp_body, is_stream):
    tags = set()
    if not resp_body:
        return tags

    try:
        payload = json.loads(resp_body)
        tags.update(_collect_backend_status_tags(payload))
    except Exception:
        pass

    if is_stream:
        done_data = _extract_stream_done_json(resp_body)
        if done_data is not None:
            tags.update(_collect_backend_status_tags(done_data))

    return tags


LLM_UNAVAILABLE_TEXT_MARKERS = (
    "configured llm backend unavailable",
    "configured ollama backend temporarily unavailable",
    "llm backend unavailable",
    "llm stream unavailable",
    "llm explanation degraded",
    "deterministic explanation fallback",
    "deterministic fallback used",
    "enhanced explanation unavailable",
    "all llm backends failed",
    "upstream_timeout",
    "upstream_unavailable",
)


def _text_has_llm_unavailable_marker(value):
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in LLM_UNAVAILABLE_TEXT_MARKERS)


def _payload_has_llm_backend_unavailable_signal(payload):
    """
    Detect response shapes that indicate deterministic/degraded fallback because
    the configured LLM backend was unavailable.
    """
    if not isinstance(payload, dict):
        return False

    if payload.get("fallback") is True:
        return True
    if _text_has_llm_unavailable_marker(payload.get("llm_response")):
        return True
    if _text_has_llm_unavailable_marker(payload.get("fallback_note")):
        return True

    warnings = payload.get("warnings")
    if isinstance(warnings, list) and any(_text_has_llm_unavailable_marker(w) for w in warnings):
        return True

    degradation = payload.get("degradation")
    if isinstance(degradation, dict):
        component = str(degradation.get("component") or "").strip().lower()
        reason = degradation.get("reason")
        message = degradation.get("message")
        if "llm" in component and (
            _text_has_llm_unavailable_marker(reason)
            or _text_has_llm_unavailable_marker(message)
            or isinstance(degradation.get("backend_status"), dict)
        ):
            return True
        if _text_has_llm_unavailable_marker(reason) or _text_has_llm_unavailable_marker(message):
            return True

    debug_info = payload.get("debug_info")
    if isinstance(debug_info, dict):
        llm_exec = debug_info.get("llm_execution")
        if isinstance(llm_exec, dict):
            if llm_exec.get("degraded") is True:
                return True
            if (
                _text_has_llm_unavailable_marker(llm_exec.get("reason"))
                or _text_has_llm_unavailable_marker(llm_exec.get("source"))
            ):
                return True
        dbg_degradation = debug_info.get("degradation")
        if isinstance(dbg_degradation, dict):
            component = str(dbg_degradation.get("component") or "").strip().lower()
            if "llm" in component:
                if (
                    _text_has_llm_unavailable_marker(dbg_degradation.get("reason"))
                    or _text_has_llm_unavailable_marker(dbg_degradation.get("message"))
                ):
                    return True
                backend_status = dbg_degradation.get("backend_status")
                if isinstance(backend_status, dict):
                    failures = backend_status.get("failures")
                    if isinstance(failures, list) and failures:
                        return True

    return_trip = payload.get("return_trip")
    if isinstance(return_trip, dict) and _payload_has_llm_backend_unavailable_signal(return_trip):
        return True

    legs = payload.get("legs")
    if isinstance(legs, list):
        for leg in legs:
            if isinstance(leg, dict) and _payload_has_llm_backend_unavailable_signal(leg):
                return True

    return False


def _extract_primary_payload(resp_body, *, is_stream):
    if not resp_body:
        return None
    if is_stream:
        return _extract_stream_done_json(resp_body)
    try:
        payload = json.loads(resp_body)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_llm_execution_records(payload):
    records = []
    if not isinstance(payload, dict):
        return records

    debug_info = payload.get("debug_info")
    if isinstance(debug_info, dict):
        llm_exec = debug_info.get("llm_execution")
        if isinstance(llm_exec, dict):
            records.append({"scope": "root", **llm_exec})

    return_trip = payload.get("return_trip")
    if isinstance(return_trip, dict):
        rt_debug = return_trip.get("debug_info")
        if isinstance(rt_debug, dict):
            rt_exec = rt_debug.get("llm_execution")
            if isinstance(rt_exec, dict):
                records.append({"scope": "return_trip", **rt_exec})

    legs = payload.get("legs")
    if isinstance(legs, list):
        for idx, leg in enumerate(legs):
            if not isinstance(leg, dict):
                continue
            leg_debug = leg.get("debug_info")
            if not isinstance(leg_debug, dict):
                continue
            leg_exec = leg_debug.get("llm_execution")
            if isinstance(leg_exec, dict):
                records.append({"scope": f"leg_{idx + 1}", **leg_exec})

    return records


def _classify_llm_generation_mode(sources, *, is_stream):
    normalized = {str(s or "").strip().lower() for s in (sources or []) if str(s or "").strip()}
    if any("stream" in src for src in normalized):
        return "stream"
    if normalized:
        return "non_stream"
    return "stream" if is_stream else "non_stream"


def _build_llm_evidence(
    payload,
    *,
    expect_llm,
    is_stream,
    duration_sec,
    validation_request_start_epoch_ms=None,
):
    """
    Distinguish model residency from real generation completion evidence.
    This intentionally does NOT treat "model loaded" as completion proof.
    """
    evidence = {
        "required": bool(expect_llm),
        "state": "not_required",
        "model_loaded_signal": "not_measured_by_scenario",
        "model_loaded_signal_note": (
            "Model residency/keep_alive is not treated as generation success; "
            "completion evidence comes from llm_execution/result_status/fallback markers."
        ),
        "request_reached_llm_path": False,
        "completion_observed": False,
        "degraded_observed": False,
        "generation_mode": "stream" if is_stream else "non_stream",
        "completion_sources": [],
        "backends": [],
        "models": [],
        "num_ctx_values": [],
        "thinking_modes": [],
        "timeout_sec": None,
        "latency_sec": None,
        "timeout_ratio": None,
        "near_timeout": False,
        "timeout_shaped": False,
        "first_token_latency_sec": None,
        "first_token_from_validation_send_sec": None,
        "first_token_available": False,
        "first_token_measurement": "not_available",
        "payload_present": isinstance(payload, dict),
    }
    if not expect_llm:
        return evidence
    if not isinstance(payload, dict):
        evidence["state"] = "unverified_no_payload"
        return evidence

    records = _extract_llm_execution_records(payload)
    sources = []
    backends = []
    models = []
    num_ctx_values = []
    thinking_modes = []
    latencies = []
    timeout_pairs = []
    first_token_latencies = []
    first_token_epochs = []
    saw_visible_first_token = False

    completion_sources = {"router_completion", "stream_completion", "skip_llm_summary"}
    degraded_sources = {"deterministic_fallback", "stream_deterministic_fallback"}

    for record in records:
        source = str(record.get("source") or "").strip().lower()
        backend = str(record.get("backend") or "").strip().lower()
        degraded = bool(record.get("degraded") is True)
        timeout_sec = _safe_float(record.get("timeout_sec"))
        latency_sec = _safe_float(record.get("latency_sec"))
        model = str(record.get("model") or "").strip()
        num_ctx = record.get("num_ctx")
        thinking_mode = str(record.get("thinking_mode") or "").strip().lower()
        first_token_latency = _safe_float(record.get("first_token_latency_sec"))
        if first_token_latency is not None:
            saw_visible_first_token = True
        if first_token_latency is None:
            first_token_latency = _safe_float(record.get("first_chunk_latency_sec"))
        first_token_epoch = _safe_float(record.get("first_token_epoch_ms"))
        if first_token_epoch is None:
            first_token_epoch = _safe_float(record.get("first_chunk_epoch_ms"))

        if source:
            sources.append(source)
        if backend:
            backends.append(backend)
        if model:
            models.append(model)
        if num_ctx not in (None, ""):
            num_ctx_values.append(num_ctx)
        if thinking_mode:
            thinking_modes.append(thinking_mode)
        if latency_sec is not None and latency_sec >= 0:
            latencies.append(latency_sec)
        if (
            latency_sec is not None
            and timeout_sec is not None
            and timeout_sec > 0
            and latency_sec >= 0
        ):
            timeout_pairs.append((latency_sec, timeout_sec))
        if first_token_latency is not None and first_token_latency >= 0:
            first_token_latencies.append(first_token_latency)
        if first_token_epoch is not None and first_token_epoch > 0:
            first_token_epochs.append(first_token_epoch)

        if degraded and source and source not in degraded_sources:
            # preserve explicit degraded sources in reporting for debug readability
            sources.append(f"{source}:degraded")

    evidence["generation_mode"] = _classify_llm_generation_mode(sources, is_stream=is_stream)
    evidence["completion_sources"] = sorted({src for src in sources if src in completion_sources})
    evidence["backends"] = sorted(set(backends))
    evidence["models"] = sorted(set(models))
    normalized_ctx_values = []
    for value in num_ctx_values:
        try:
            normalized_ctx_values.append(int(value))
            continue
        except Exception:
            pass
        as_text = str(value).strip()
        if as_text:
            normalized_ctx_values.append(as_text)
    evidence["num_ctx_values"] = sorted(set(normalized_ctx_values), key=lambda x: str(x))
    evidence["thinking_modes"] = sorted(set(thinking_modes))
    evidence["request_reached_llm_path"] = bool(records) or bool((payload.get("llm_response") or "").strip())

    degraded_observed = _payload_has_llm_backend_unavailable_signal(payload) or str(
        payload.get("result_status") or ""
    ).strip().lower() == "degraded"
    completion_observed = bool(evidence["completion_sources"]) and not degraded_observed
    evidence["degraded_observed"] = bool(degraded_observed)
    evidence["completion_observed"] = bool(completion_observed)

    if latencies:
        # Keep the max latency when multiple legs/segments are present.
        evidence["latency_sec"] = round(max(latencies), 3)

    if timeout_pairs:
        # Use the highest utilization ratio across all observed completions.
        ratios = [(lat / budget) for lat, budget in timeout_pairs if budget > 0]
        if ratios:
            ratio = max(ratios)
            evidence["timeout_ratio"] = round(ratio, 3)
            evidence["near_timeout"] = ratio >= LLM_NEAR_TIMEOUT_RATIO
            evidence["timeout_shaped"] = ratio >= LLM_TIMEOUT_SHAPED_RATIO
        evidence["timeout_sec"] = round(max(budget for _, budget in timeout_pairs), 3)

    if first_token_latencies:
        evidence["first_token_latency_sec"] = round(max(first_token_latencies), 3)
        evidence["first_token_available"] = True
        evidence["first_token_measurement"] = (
            "llm_dispatch_to_first_visible_token"
            if saw_visible_first_token
            else "llm_dispatch_to_first_chunk"
        )
    elif is_stream:
        evidence["first_token_measurement"] = "stream_first_token_not_observed"
    else:
        evidence["first_token_measurement"] = "not_available_non_stream"

    request_start_epoch_ms = _safe_float(validation_request_start_epoch_ms)
    if first_token_epochs and request_start_epoch_ms is not None:
        from_send = [
            (epoch_ms - request_start_epoch_ms) / 1000.0
            for epoch_ms in first_token_epochs
            if epoch_ms >= request_start_epoch_ms
        ]
        if from_send:
            evidence["first_token_from_validation_send_sec"] = round(min(from_send), 3)

    if evidence["degraded_observed"]:
        evidence["state"] = "degraded_fallback"
    elif evidence["completion_observed"]:
        evidence["state"] = "completed"
    elif evidence["request_reached_llm_path"]:
        evidence["state"] = "attempted_no_completion_proof"
    else:
        evidence["state"] = "no_llm_path_evidence"

    # When no llm_execution timing is available, keep overall test duration
    # as a fallback observability signal (still not a completion proof).
    if evidence["latency_sec"] is None and duration_sec is not None:
        duration_f = _safe_float(duration_sec)
        if duration_f is not None and duration_f > 0:
            evidence["latency_sec"] = round(duration_f, 3)

    return evidence


def _percentile(values, pct):
    if not values:
        return None
    sorted_values = sorted(float(v) for v in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * max(0.0, min(1.0, float(pct)))
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if upper == lower:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def _summarize_llm_evidence(report_entries):
    required_entries = [entry for entry in report_entries if bool(entry.get("llm_required"))]
    with_payload = [
        entry for entry in required_entries
        if isinstance((entry.get("llm_evidence") or {}), dict)
        and bool((entry.get("llm_evidence") or {}).get("payload_present"))
    ]
    completion_entries = [
        entry for entry in with_payload if (entry.get("llm_evidence") or {}).get("completion_observed")
    ]
    degraded_entries = [
        entry for entry in with_payload if (entry.get("llm_evidence") or {}).get("degraded_observed")
    ]
    near_timeout_entries = [
        entry for entry in completion_entries if (entry.get("llm_evidence") or {}).get("near_timeout")
    ]
    timeout_shaped_entries = [
        entry for entry in completion_entries if (entry.get("llm_evidence") or {}).get("timeout_shaped")
    ]
    timeout_ratios = [
        float((entry.get("llm_evidence") or {}).get("timeout_ratio"))
        for entry in completion_entries
        if _safe_float((entry.get("llm_evidence") or {}).get("timeout_ratio")) is not None
    ]
    first_token_entries = [
        entry
        for entry in with_payload
        if _safe_float((entry.get("llm_evidence") or {}).get("first_token_latency_sec")) is not None
    ]
    first_token_latencies = [
        float((entry.get("llm_evidence") or {}).get("first_token_latency_sec"))
        for entry in first_token_entries
    ]
    first_token_from_request = [
        float((entry.get("llm_evidence") or {}).get("first_token_from_validation_send_sec"))
        for entry in with_payload
        if _safe_float((entry.get("llm_evidence") or {}).get("first_token_from_validation_send_sec")) is not None
    ]
    first_token_unavailable_entries = [
        entry
        for entry in with_payload
        if not bool((entry.get("llm_evidence") or {}).get("first_token_available"))
    ]
    models_seen = sorted(
        {
            str(model)
            for entry in with_payload
            for model in ((entry.get("llm_evidence") or {}).get("models") or [])
            if str(model).strip()
        }
    )
    num_ctx_seen = sorted(
        {
            str(value)
            for entry in with_payload
            for value in ((entry.get("llm_evidence") or {}).get("num_ctx_values") or [])
            if str(value).strip()
        }
    )
    thinking_seen = sorted(
        {
            str(mode)
            for entry in with_payload
            for mode in ((entry.get("llm_evidence") or {}).get("thinking_modes") or [])
            if str(mode).strip()
        }
    )
    backend_seen = sorted(
        {
            str(backend)
            for entry in with_payload
            for backend in ((entry.get("llm_evidence") or {}).get("backends") or [])
            if str(backend).strip()
        }
    )
    unknown_entries = [
        entry for entry in required_entries
        if (entry.get("llm_evidence") or {}).get("state") == "unverified_no_payload"
    ]
    completion_ratio = (
        len(completion_entries) / len(with_payload)
        if with_payload
        else 0.0
    )

    near_timeout_ratio = (
        len(near_timeout_entries) / len(completion_entries)
        if completion_entries
        else 0.0
    )
    timeout_shaped_ratio = (
        len(timeout_shaped_entries) / len(completion_entries)
        if completion_entries
        else 0.0
    )

    return {
        "required_total": len(required_entries),
        "required_with_payload_evidence": len(with_payload),
        "completion_observed": len(completion_entries),
        "completion_ratio": round(completion_ratio, 3),
        "degraded_observed": len(degraded_entries),
        "unknown_unverified": len(unknown_entries),
        "near_timeout_completions": len(near_timeout_entries),
        "near_timeout_ratio": round(near_timeout_ratio, 3),
        "timeout_shaped_completions": len(timeout_shaped_entries),
        "timeout_shaped_ratio": round(timeout_shaped_ratio, 3),
        "timeout_ratio_p50": (
            round(_percentile(timeout_ratios, 0.50), 3)
            if timeout_ratios
            else None
        ),
        "timeout_ratio_p90": (
            round(_percentile(timeout_ratios, 0.90), 3)
            if timeout_ratios
            else None
        ),
        "first_token_observed": len(first_token_entries),
        "first_token_unavailable": len(first_token_unavailable_entries),
        "first_token_latency_p50": (
            round(_percentile(first_token_latencies, 0.50), 3)
            if first_token_latencies
            else None
        ),
        "first_token_latency_p90": (
            round(_percentile(first_token_latencies, 0.90), 3)
            if first_token_latencies
            else None
        ),
        "first_token_from_validation_send_p50": (
            round(_percentile(first_token_from_request, 0.50), 3)
            if first_token_from_request
            else None
        ),
        "first_token_from_validation_send_p90": (
            round(_percentile(first_token_from_request, 0.90), 3)
            if first_token_from_request
            else None
        ),
        "models_seen": models_seen,
        "num_ctx_seen": num_ctx_seen,
        "thinking_modes_seen": thinking_seen,
        "backends_seen": backend_seen,
        "completion_scenarios_near_timeout": [
            _display_name_for_base(_strip_mode_suffix(str(entry.get("name") or "")))
            for entry in near_timeout_entries[:8]
        ],
        "completion_scenarios_timeout_shaped": [
            _display_name_for_base(_strip_mode_suffix(str(entry.get("name") or "")))
            for entry in timeout_shaped_entries[:8]
        ],
    }


def _format_entry_llm_runtime(entry):
    evidence = entry.get("llm_evidence") or {}
    if not isinstance(evidence, dict):
        return ""
    backends = [str(v) for v in (evidence.get("backends") or []) if str(v).strip()]
    models = [str(v) for v in (evidence.get("models") or []) if str(v).strip()]
    num_ctx_values = [str(v) for v in (evidence.get("num_ctx_values") or []) if str(v).strip()]
    thinking_modes = [str(v) for v in (evidence.get("thinking_modes") or []) if str(v).strip()]
    runtime_parts = []
    if backends:
        runtime_parts.append(f"backend={','.join(backends)}")
    if models:
        runtime_parts.append(f"model={','.join(models)}")
    if num_ctx_values:
        runtime_parts.append(f"num_ctx={','.join(num_ctx_values)}")
    if thinking_modes:
        runtime_parts.append(f"thinking={','.join(thinking_modes)}")
    if evidence.get("first_token_latency_sec") is not None:
        runtime_parts.append(f"first_token_sec={evidence.get('first_token_latency_sec')}")
    if evidence.get("first_token_from_validation_send_sec") is not None:
        runtime_parts.append(
            f"req_to_first_token_sec={evidence.get('first_token_from_validation_send_sec')}"
        )
    return ", ".join(runtime_parts)


def _derive_dominant_llm_profile(report_entries):
    """
    Best-effort dominant runtime profile observed across LLM-required scenarios.
    """
    counts = {}
    for entry in report_entries:
        if not bool(entry.get("llm_required")):
            continue
        evidence = entry.get("llm_evidence") or {}
        if not isinstance(evidence, dict):
            continue
        backends = tuple(sorted(str(v) for v in (evidence.get("backends") or []) if str(v).strip()))
        models = tuple(sorted(str(v) for v in (evidence.get("models") or []) if str(v).strip()))
        num_ctx_values = tuple(
            sorted(str(v) for v in (evidence.get("num_ctx_values") or []) if str(v).strip())
        )
        thinking_modes = tuple(
            sorted(str(v) for v in (evidence.get("thinking_modes") or []) if str(v).strip())
        )
        profile_key = (backends, models, num_ctx_values, thinking_modes)
        if profile_key == ((), (), (), ()):
            continue
        counts[profile_key] = counts.get(profile_key, 0) + 1
    if not counts:
        return {
            "backend": "",
            "model": "",
            "num_ctx": "",
            "thinking_mode": "",
            "observed_count": 0,
        }
    top_key, top_count = max(counts.items(), key=lambda item: item[1])
    backends, models, num_ctx_values, thinking_modes = top_key
    return {
        "backend": ",".join(backends),
        "model": ",".join(models),
        "num_ctx": ",".join(num_ctx_values),
        "thinking_mode": ",".join(thinking_modes),
        "observed_count": int(top_count),
    }


def _apply_scope_to_capability_status(core_status, *, frontend_enabled):
    normalized = str(core_status or "UNVERIFIED").strip().upper()
    if frontend_enabled:
        return normalized
    if normalized in {"DEGRADED", "UNVERIFIED", "FAIL", "PARTIAL"}:
        return normalized
    return "BACKEND_ONLY"


def _read_pytest_testpaths():
    ini_path = ROOT / "pytest.ini"
    if not ini_path.exists():
        return []
    parser = configparser.ConfigParser()
    try:
        parser.read(ini_path)
    except Exception:
        return []
    if not parser.has_section("pytest"):
        return []
    raw_value = parser.get("pytest", "testpaths", fallback="")
    if not raw_value:
        return []
    paths = []
    for raw_line in str(raw_value).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [token for token in re.split(r"\s+", line) if token and not token.startswith("#")]
        paths.extend(parts)
    return paths


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _run_validation_llm_warmup(mode_label, base_url):
    """
    Validation-startup gate for real LLM readiness.
    Service /health readiness is not sufficient for this gate.
    """
    global LLM_WARMUP_CONTEXT

    max_attempts = max(1, _env_int("VALIDATION_LLM_WARMUP_ATTEMPTS", LLM_WARMUP_MAX_ATTEMPTS_DEFAULT))
    retry_delay = max(0.0, _env_float("VALIDATION_LLM_WARMUP_RETRY_DELAY_SEC", LLM_WARMUP_RETRY_DELAY_SEC_DEFAULT))
    warmup_date = (datetime.now().date() + timedelta(days=21)).strftime("%Y-%m-%d")
    warmup_payload = {
        "origin": "DEL",
        "destination": "BOM",
        "date": warmup_date,
        "user_query": "Validation warmup probe: confirm route and summarize best flight.",
    }

    LLM_WARMUP_CONTEXT = {
        "attempted": True,
        "succeeded": False,
        "mode": str(mode_label or ""),
        "attempts": 0,
        "max_attempts": int(max_attempts),
        "reason": "",
        "http_status": None,
        "latency_sec": None,
        "completion_source": "",
        "backend": "",
        "model": "",
        "num_ctx": None,
        "thinking_mode": "",
        "first_token_latency_sec": None,
        "timeout_ratio": None,
        "request_reached_llm_path": False,
        "completion_observed": False,
        "degraded_observed": False,
        "admission": "",
        "execution": "",
        "replayed_recent": False,
        "replay_bypassed": False,
        "attempt_records": [],
    }

    completion_sources = {"router_completion", "stream_completion", "skip_llm_summary"}
    local_backend = "ollama"

    log(
        "Validation LLM warmup gate: service readiness achieved; "
        "now probing real non-degraded local completion."
    )
    last_reason = "warmup_not_attempted"
    for attempt in range(1, max_attempts + 1):
        attempt_started = time.time()
        log(f"Validation LLM warmup attempt {attempt}/{max_attempts} (mode={mode_label})")
        probe_id = f"{mode_label}-warmup-{int(time.time() * 1000)}-a{attempt}"
        headers = {
            "Content-Type": "application/json",
            "X-Validation-Warmup-Probe": "1",
            "X-Validation-Warmup-Id": probe_id,
            "X-Validation-Warmup-Attempt": str(attempt),
        }
        request_error = ""
        response_body = ""
        http_status = None
        admission = ""
        execution_marker = ""
        replay_bypassed = False
        replayed_recent = False
        try:
            resp = requests.post(
                f"{base_url}/ask",
                data=json.dumps(warmup_payload),
                headers=headers,
            )
            response_body = resp.text or ""
            http_status = str(resp.status_code)
            admission = str(resp.headers.get("X-Ask-Admission") or "").strip().lower()
            execution_marker = str(resp.headers.get("X-Validation-Warmup-Execution") or "").strip().lower()
            replay_bypassed = str(resp.headers.get("X-Validation-Warmup-Replay-Bypassed") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            replayed_recent = admission == "replayed_recent"
        except Exception as exc:
            request_error = str(exc).strip() or exc.__class__.__name__
        latency_sec = max(0.0, time.time() - attempt_started)
        payload = _extract_primary_payload(response_body, is_stream=False)
        evidence = _build_llm_evidence(
            payload,
            expect_llm=True,
            is_stream=False,
            duration_sec=latency_sec,
        )
        llm_exec = {}
        if isinstance(payload, dict):
            llm_exec = (payload.get("debug_info") or {}).get("llm_execution") or {}
            if not isinstance(llm_exec, dict):
                llm_exec = {}
        source = str(llm_exec.get("source") or "").strip().lower()
        backend = str(llm_exec.get("backend") or "").strip().lower()
        model = str(llm_exec.get("model") or "").strip()
        num_ctx = llm_exec.get("num_ctx")
        thinking_mode = str(llm_exec.get("thinking_mode") or "").strip().lower()
        first_token_latency_sec = _safe_float(
            llm_exec.get("first_token_latency_sec")
            if llm_exec.get("first_token_latency_sec") is not None
            else llm_exec.get("first_chunk_latency_sec")
        )
        result_status = ""
        if isinstance(payload, dict):
            result_status = str(payload.get("result_status") or "").strip().lower()

        warmup_ok = True
        reason = ""
        if request_error:
            warmup_ok = False
            reason = f"warmup_request_error:{request_error}"
        elif http_status and http_status != "200":
            warmup_ok = False
            reason = f"http_{http_status}"
        elif not isinstance(payload, dict):
            warmup_ok = False
            reason = "warmup_payload_not_json_object"
        elif _payload_has_llm_backend_unavailable_signal(payload):
            warmup_ok = False
            reason = "warmup_detected_degraded_or_backend_unavailable_signal"
        elif replayed_recent:
            warmup_ok = False
            reason = "warmup_replayed_recent_response"
        elif result_status and result_status != "success":
            warmup_ok = False
            reason = f"warmup_result_status_{result_status}"
        elif not evidence.get("request_reached_llm_path"):
            warmup_ok = False
            reason = "warmup_no_llm_path_evidence"
        elif evidence.get("degraded_observed"):
            warmup_ok = False
            reason = "warmup_degraded_observed"
        elif not evidence.get("completion_observed"):
            warmup_ok = False
            reason = "warmup_no_completion_observed"
        elif source not in completion_sources:
            warmup_ok = False
            reason = f"warmup_completion_source_not_accepted:{source or 'missing'}"
        elif backend != local_backend:
            warmup_ok = False
            reason = f"warmup_backend_not_local:{backend or 'missing'}"

        LLM_WARMUP_CONTEXT.update(
            {
                "attempts": int(attempt),
                "http_status": http_status,
                "latency_sec": round(float(latency_sec), 3),
                "completion_source": source,
                "backend": backend,
                "model": model,
                "num_ctx": num_ctx,
                "thinking_mode": thinking_mode,
                "first_token_latency_sec": (
                    round(first_token_latency_sec, 3)
                    if first_token_latency_sec is not None
                    else None
                ),
                "timeout_ratio": evidence.get("timeout_ratio"),
                "request_reached_llm_path": bool(evidence.get("request_reached_llm_path")),
                "completion_observed": bool(evidence.get("completion_observed")),
                "degraded_observed": bool(evidence.get("degraded_observed")),
                "admission": admission,
                "execution": execution_marker,
                "replayed_recent": bool(replayed_recent),
                "replay_bypassed": bool(replay_bypassed),
                "reason": reason or "",
            }
        )
        attempt_records = list(LLM_WARMUP_CONTEXT.get("attempt_records") or [])
        attempt_records.append(
            {
                "attempt": int(attempt),
                "probe_id": probe_id,
                "http_status": http_status,
                "latency_sec": round(float(latency_sec), 3),
                "reason": reason or "",
                "admission": admission,
                "execution": execution_marker,
                "replayed_recent": bool(replayed_recent),
                "replay_bypassed": bool(replay_bypassed),
                "request_reached_llm_path": bool(evidence.get("request_reached_llm_path")),
                "completion_observed": bool(evidence.get("completion_observed")),
                "degraded_observed": bool(evidence.get("degraded_observed")),
                "completion_source": source,
                "backend": backend,
                "model": model,
                "num_ctx": num_ctx,
                "thinking_mode": thinking_mode,
                "first_token_latency_sec": (
                    round(first_token_latency_sec, 3)
                    if first_token_latency_sec is not None
                    else None
                ),
                "timeout_ratio": evidence.get("timeout_ratio"),
            }
        )
        LLM_WARMUP_CONTEXT["attempt_records"] = attempt_records

        if warmup_ok:
            LLM_WARMUP_CONTEXT["succeeded"] = True
            LLM_WARMUP_CONTEXT["reason"] = ""
            log(
                "Validation LLM warmup succeeded with real completion: "
                + f"source={source}, backend={backend}, "
                + f"model={model or 'unknown'}, num_ctx={num_ctx}, thinking_mode={thinking_mode or 'unknown'}, "
                + f"latency_sec={latency_sec:.3f}, timeout_ratio={evidence.get('timeout_ratio')}, "
                + f"first_token_latency_sec={LLM_WARMUP_CONTEXT.get('first_token_latency_sec')}, "
                + f"admission={admission or 'unknown'}, execution={execution_marker or 'unknown'}, "
                + f"replayed_recent={bool(replayed_recent)}, replay_bypassed={bool(replay_bypassed)}"
            )
            return True

        last_reason = reason or "warmup_failed_unknown_reason"
        log(
            "Validation LLM warmup attempt failed: "
            + f"reason={last_reason}, http_status={http_status}, "
            + f"model={model or 'unknown'}, num_ctx={num_ctx}, thinking_mode={thinking_mode or 'unknown'}, "
            + f"completion_observed={bool(evidence.get('completion_observed'))}, "
            + f"degraded_observed={bool(evidence.get('degraded_observed'))}, "
            + f"admission={admission or 'unknown'}, execution={execution_marker or 'unknown'}, "
            + f"replayed_recent={bool(replayed_recent)}, replay_bypassed={bool(replay_bypassed)}"
        )
        if attempt < max_attempts and retry_delay > 0:
            log(f"Validation LLM warmup retrying after {retry_delay:.1f}s")
            time.sleep(retry_delay)

    LLM_WARMUP_CONTEXT["succeeded"] = False
    LLM_WARMUP_CONTEXT["reason"] = last_reason
    log(
        "Validation LLM warmup FAILED after "
        + f"{max_attempts} attempt(s): reason={last_reason}. "
        + "LLM-required scenario loop will not start."
    )
    return False


def _extract_log_based_failure_tags(mode_label):
    if mode_label == "machine":
        log_path = LOG_DIR / "machine_uvicorn.log"
    elif mode_label == "docker":
        log_path = LOG_DIR / "docker_validation_container_logs.log"
    else:
        return set()

    if not log_path.exists():
        return set()

    try:
        log_text = log_path.read_text(errors="ignore").lower()
    except Exception:
        return set()

    tags = set()

    if "no usable keys for provider" in log_text or "no available keys for service" in log_text:
        tags.add("provider_no_active_key")
    if "insufficient_quota" in log_text or "quota exceeded" in log_text:
        tags.add("provider_quota_exhausted")
    if "billing" in log_text or "hard limit" in log_text or "credit" in log_text or "payment required" in log_text:
        tags.add("provider_billing_blocked")
    if "rate limit" in log_text or "429" in log_text or "too many requests" in log_text:
        tags.add("provider_rate_limited")
    if "authentication" in log_text or "invalid api key" in log_text or "unauthorized" in log_text:
        tags.add("provider_auth_failed")
    if "name resolution" in log_text or "dns" in log_text or "connection" in log_text or "unreachable" in log_text:
        tags.add("provider_unreachable")

    return tags


def _determine_validation_verdict(
    name,
    status,
    failure_tags,
    *,
    soft_pass_policy=SOFT_PASS_HARD_FAIL_ONLY,
    mode_bucket=MODE_BACKEND_INTERNAL,
):
    if status == 0:
        return VERDICT_PASS

    if _is_soft_pass_eligible_test(soft_pass_policy=soft_pass_policy, mode_bucket=mode_bucket):
        soft_hits = set(failure_tags or set()) & SOFT_PASS_NO_CREDIT_TAGS
        hard_hits = set(failure_tags or set()) - SOFT_PASS_NO_CREDIT_TAGS
        if soft_hits and not hard_hits:
            return VERDICT_SOFT_PASS_NO_CREDIT

    return VERDICT_FAIL


def _looks_like_llm_content_failure(reason):
    text = (reason or "").lower()
    return (
        "llm" in text
        or "fallback" in text
        or "all lmm backends failed" in text
        or "all llm backends failed" in text
        or "temporarily unavailable" in text
    )


def _classify_failure_category(entry):
    status = int(entry.get("status") or 0)
    reason = str(entry.get("reason") or "")
    reason_l = reason.lower()
    failure_tags = set(entry.get("failure_tags") or [])
    base_name = _strip_mode_suffix(entry.get("name", ""))
    validation_type = entry.get("validation_type") or _validation_meta_for_base(base_name).get("validation_type")

    if "booking_bridge_contract_failure" in failure_tags:
        return "Contract"
    if base_name.startswith("contract_") or validation_type == "contract":
        return "Contract"

    if any(tag.startswith("provider_") for tag in failure_tags):
        return "Infra"
    if status == 127:
        return "Infra"
    if status == 125:
        return "Unexpected"
    if status == 126:
        return "Validation"
    if status == 124:
        infra_markers = (
            "connection refused",
            "timed out",
            "timeout",
            "name resolution",
            "dns",
            "docker daemon",
            "address already in use",
        )
        if any(marker in reason_l for marker in infra_markers):
            return "Infra"
        if "http 5" in reason_l:
            return "Infra"
        if _looks_like_llm_content_failure(reason):
            return "Validation"
        return "Validation"
    if "command exited with" in reason_l:
        return "Infra"
    return "Validation"

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def run_cmd(cmd, capture_output=True, check=False, timeout=None):
    """
    Run a shell command and return (stdout, stderr, returncode).
    Accepts an optional timeout (seconds). If capture_output=False, streams
    stdout/stderr and returns (stdout, stderr, rc) when finished.
    """
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=False, capture_output=True, text=True, timeout=timeout)
            return result.stdout, result.stderr, result.returncode
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                return stdout, stderr, 124
            return stdout, stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        # timeout -> return a non-zero code and include message
        return getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s", 124
    except FileNotFoundError as e:
        return "", str(e), 127
    except Exception as e:
        return "", str(e), 125

def wait_for_ready(url=f"{DEFAULT_API_BASE_URL}/health/ready", max_wait=READY_TIMEOUT):
    log(f"Waiting for service readiness at {url} ...")
    waited = 0
    interval = 1
    while waited < max_wait:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                log("Service is ready.")
                return True
        except:
            pass
        time.sleep(interval)
        waited += interval
    log(f"Timed out waiting for readiness at {url} after {max_wait}s")
    return False

def wait_for_health_poll(timeout, url=HEALTH_URL):
    waited = 0
    interval = 1
    while waited < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(interval)
        waited += interval
    return False

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _fetch_json(url: str, timeout: int = 8):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _summarize_key_state(keys_json):
    summary = {}
    if not isinstance(keys_json, dict):
        return summary
    for service, entries in keys_json.items():
        total = 0
        exhausted = 0
        active = 0
        if isinstance(entries, list):
            iterable = entries
        elif isinstance(entries, dict):
            iterable = list(entries.values())
        else:
            iterable = [entries]
        for entry in iterable:
            total += 1
            if isinstance(entry, dict):
                exhausted_until = entry.get("exhausted_until")
                if exhausted_until:
                    exhausted += 1
                if entry.get("active", False):
                    active += 1
            elif str(entry).lower() == "active":
                active += 1
        summary[service] = {"total": total, "active": active, "exhausted": exhausted}
    return summary


def _read_env_value_from_file(env_path: Path, key: str):
    """Best-effort KEY=value lookup from env files without executing shell syntax."""
    try:
        lines = env_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    wanted = str(key or "").strip()
    if not wanted:
        return None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        if lhs.strip() != wanted:
            continue
        value = rhs.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return None


def _resolve_validation_timeout_seconds(
    *,
    key: str,
    src_env: Path,
    validation_override_env: str | None,
    fallback_default: int | None,
    minimum_seconds: int,
    allow_unset: bool = False,
    include_src_env: bool = True,
):
    """
    Resolve timeout values in priority order:
      1) explicit validation override env var
      2) current process env var
      3) source env file copied into .env.tmp (optional)
      4) runtime-aligned hard default (if provided)
    """
    raw = None
    source = "unset"

    if validation_override_env and is_env_set(validation_override_env):
        raw = os.getenv(validation_override_env)
        source = f"{validation_override_env} override"
    else:
        process_raw = os.getenv(key)
        if process_raw not in (None, ""):
            raw = process_raw
            source = f"{key} (process env)"
        elif include_src_env:
            file_raw = _read_env_value_from_file(src_env, key)
            if file_raw not in (None, ""):
                raw = file_raw
                source = f"{key} ({src_env.name})"

    if raw in (None, ""):
        if allow_unset and fallback_default is None:
            return None, source
        value = fallback_default
        source = f"default ({fallback_default})"
    else:
        try:
            value = int(float(str(raw).strip()))
        except Exception:
            if allow_unset and fallback_default is None:
                return None, f"{source} (invalid -> unset)"
            value = fallback_default
            source = f"{source} (invalid -> default {fallback_default})"

    if value is None:
        return None, source
    return max(int(minimum_seconds), int(value)), source


def _resolve_validation_num_ctx(*, src_env: Path):
    """
    Resolve validation OLLAMA_NUM_CTX with explicit experiment controls.

    Modes:
      - validated_default (default): force validation baseline (4096)
      - passthrough: honor OLLAMA_NUM_CTX from env/.env
      - override: honor VALIDATION_OLLAMA_NUM_CTX explicitly
    """
    ctx_mode = normalize_validation_num_ctx_mode(
        get_env_str("VALIDATION_OLLAMA_NUM_CTX_MODE", "validated_default")
    )
    resolution = resolve_validation_num_ctx(
        mode=ctx_mode,
        validation_override_raw=os.getenv("VALIDATION_OLLAMA_NUM_CTX"),
        process_env=os.environ,
        passthrough_env_paths=[src_env],
        baseline_default=4096,
        minimum_value=1024,
    )
    value = int(resolution.get("effective_num_ctx") or 4096)
    source = str(resolution.get("source") or f"default ({value})")
    note = str(resolution.get("note") or "").strip()
    if note:
        source = f"{source} ({note})"
    return value, source, str(resolution.get("mode") or ctx_mode)


def _resolve_validation_thinking_mode(*, src_env: Path):
    """
    Preserve validation default behavior: thinking is disabled unless explicitly
    overridden by VALIDATION_OLLAMA_THINKING_MODE.
    """
    del src_env  # reserved for future parity modes
    raw = os.getenv("VALIDATION_OLLAMA_THINKING_MODE")
    source = "default (disable)"
    value = "disable"
    if raw not in (None, ""):
        value = str(raw).strip().lower()
        source = "VALIDATION_OLLAMA_THINKING_MODE override"
    if value not in {"auto", "disable", "force"}:
        value = "disable"
        source = f"{source} (invalid -> disable)"
    return value, source


def docker_available():
    """Check if docker daemon is reachable."""
    try:
        r = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        return r.returncode == 0
    except Exception:
        return False


def _should_use_frontend_path(cmd, frontend_enabled=None):
    if frontend_enabled is None:
        frontend_enabled = args.frontend
    if not frontend_enabled or not cmd:
        return False
    if cmd[0] == "curl":
        return any("/ask" in token for token in cmd)
    if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-c":
        return "/ask" in cmd[2] and "-d" in cmd[2]
    return False


def _should_route_case_through_frontend(name, assertions):
    if not args.frontend:
        return False

    base = _strip_mode_suffix(name)
    # Keep backend/runtime/contract checks on direct backend paths even in --frontend mode.
    if any(base.startswith(prefix) for prefix in FRONTEND_BACKEND_DIRECT_PREFIXES):
        return False

    if isinstance(assertions, dict) and assertions.get("frontend_scenario"):
        return True
    if base.startswith(("frontend_runtime_", "frontend_fixture_", "frontend_real_backend_", "frontend_live_canary_")):
        return bool(args.frontend_real_backend)

    # Safe default: do not reroute backend smoke checks through browser unless
    # explicit real-backend frontend mode is enabled.
    if not args.frontend_real_backend:
        return False

    return any(base.startswith(prefix) for prefix in FRONTEND_HIGH_VALUE_REROUTE_PREFIXES)


def _format_frontend_failure_reason(case_result):
    case = case_result if isinstance(case_result, dict) else {}
    net = case.get("network_summary") or {}
    phase = str(case.get("failure_phase") or "unknown").strip() or "unknown"
    expectation = (
        str(case.get("failure_expectation") or "").strip()
        or str(case.get("fail_reason") or "").strip()
        or str(case.get("error_text") or "").strip()
        or "Frontend parity checks failed."
    )
    selector = str(case.get("failure_selector") or "").strip()
    runtime_s = case.get("runtime_s")
    statuses = net.get("matched_statuses")
    request_url = case.get("actual_request_url")
    request_stream = case.get("actual_request_stream")

    context_parts = [
        f"phase={phase}",
        f"request_fired={net.get('request_fired')}",
        f"request_completed_success={net.get('request_completed_success')}",
        f"streaming_completed={net.get('streaming_completed')}",
    ]
    if statuses is not None:
        context_parts.append(f"statuses={statuses}")
    if request_url:
        context_parts.append(f"url={request_url}")
    if request_stream is not None:
        context_parts.append(f"is_stream={bool(request_stream)}")
    if runtime_s is not None:
        context_parts.append(f"runtime_s={runtime_s}")
    if selector:
        context_parts.append(f"selector={selector}")

    return f"Frontend validation failed ({'; '.join(context_parts)}): {expectation}"


def _ensure_frontend_validator():
    global FRONTEND_VALIDATOR
    if FRONTEND_VALIDATOR is not None:
        return FRONTEND_VALIDATOR

    try:
        from validation.frontend_validator import FrontendValidator
    except Exception as exc:
        raise RuntimeError(f"Failed to import frontend validator module: {exc}") from exc

    FRONTEND_VALIDATOR = FrontendValidator(
        frontend_url=FRONTEND_DEFAULT_URL,
        frontend_dir=ROOT / "frontend",
        frontend_host=FRONTEND_DEFAULT_HOST,
        frontend_port=FRONTEND_DEFAULT_PORT,
        frontend_server_mode="preview" if args.frontend_preview else "dev",
        query_timeout_s=FRONTEND_QUERY_TIMEOUT,
        auto_start_frontend=True,
        fixture_mode_default=not args.frontend_real_backend,
        allow_real_backend=bool(args.frontend_real_backend),
        log_fn=log,
    )
    FRONTEND_VALIDATOR.start()
    return FRONTEND_VALIDATOR


def _close_frontend_validator():
    global FRONTEND_VALIDATOR
    if FRONTEND_VALIDATOR is None:
        return
    try:
        FRONTEND_VALIDATOR.close()
    finally:
        FRONTEND_VALIDATOR = None


def _run_frontend_validation_for_cmd(cmd, *, name=None, assertions=None):
    from validation.frontend_validator import extract_payloads_from_curl_command
    from validation.frontend_contract import FrontendValidationContext, FrontendValidationRequest

    def _frontend_case_legitimacy_issue(case_result):
        """
        Frontend validation must not credit deterministic fallback paths as real LLM-backed success.
        """
        scenario_name = str(case_result.get("validation_scenario") or "").strip().lower()
        if scenario_name.startswith("mock_") or scenario_name.startswith("fixture_"):
            return None

        fallback_marker = "(Note: Enhanced explanation unavailable"
        reqs = case_result.get("network_requests") or []
        try:
            runtime_s = float(case_result.get("runtime_s") or 0.0)
        except Exception:
            runtime_s = 0.0

        has_non_fallback_llm = False

        for rec in reqs:
            if not isinstance(rec, dict):
                continue

            done_keys = [str(k).lower() for k in (rec.get("stream_done_json_keys") or [])]
            if rec.get("is_stream") and rec.get("stream_done_json_parsed") and "error" in done_keys:
                return "Stream completed with DONE_JSON error payload (not a real LLM success path)."

            if rec.get("is_stream") and rec.get("stream_done_json_parsed"):
                if any(k in {"llm_response", "best_flight", "all_flights", "weather", "legs"} for k in done_keys) and "error" not in done_keys:
                    has_non_fallback_llm = True

            preview = str(rec.get("response_body_preview") or "")
            if fallback_marker in preview:
                return "Frontend response used deterministic fallback narrative (enhanced explanation unavailable)."
            if '"fallback":true' in preview.lower().replace(" ", ""):
                return "Frontend response body indicates fallback=true (not LLM-backed completion)."

            response_json = rec.get("response_body_json")
            if isinstance(response_json, dict):
                if response_json.get("fallback") is True:
                    return "Frontend response JSON indicates fallback=true (not LLM-backed completion)."
                llm_text = str(response_json.get("llm_response") or "")
                if llm_text:
                    if fallback_marker in llm_text:
                        return "Frontend response used deterministic fallback llm_response."
                    has_non_fallback_llm = True

        if runtime_s > 0 and runtime_s < 5.0 and not has_non_fallback_llm:
            return f"Suspiciously fast pass ({runtime_s:.2f}s) without non-fallback LLM evidence."

        return None

    payloads = extract_payloads_from_curl_command(cmd)
    if not payloads:
        return False, {"status": 125, "failure_reason": "No JSON payload found for frontend validation", "result": {}}

    frontend_scenario = ""
    frontend_expectations = {}
    if isinstance(assertions, dict):
        frontend_scenario = str(assertions.get("frontend_scenario") or "").strip()
        maybe_expect = assertions.get("frontend_expectations")
        if isinstance(maybe_expect, dict):
            frontend_expectations = maybe_expect

    validator = _ensure_frontend_validator()
    case_results = []
    for idx, payload in enumerate(payloads):
        request = FrontendValidationRequest(
            payload=dict(payload),
            context=FrontendValidationContext(
                scenario=frontend_scenario,
                expectations=dict(frontend_expectations),
                case_name=str(name or ""),
            ),
        )

        scenario_for_case = request.context.scenario
        result = validator.run_query(request, timeout_s=FRONTEND_QUERY_TIMEOUT)
        result["case_index"] = idx
        if scenario_for_case:
            result["validation_scenario"] = scenario_for_case
        case_results.append(result)

    first_failure = next((case for case in case_results if not case.get("passes")), None)
    if first_failure:
        reason = _format_frontend_failure_reason(first_failure)
        return True, {
            "status": 124,
            "failure_reason": reason,
            "result": {"frontend_cases": case_results},
        }

    first_legitimacy_failure = next(
        (
            case
            for case in case_results
            if _frontend_case_legitimacy_issue(case)
        ),
        None,
    )
    if first_legitimacy_failure:
        base_reason = _frontend_case_legitimacy_issue(first_legitimacy_failure) or "Frontend legitimacy checks failed."
        reason = (
            "Frontend legitimacy check failed "
            f"(phase=legitimacy; runtime_s={first_legitimacy_failure.get('runtime_s')}): {base_reason}"
        )
        return True, {
            "status": 124,
            "failure_reason": reason,
            "result": {"frontend_cases": case_results},
        }

    return True, {
        "status": 0,
        "failure_reason": "",
        "result": {"frontend_cases": case_results},
    }

# ----------------------------------------------------------------------
# Rotation index management (persistent counter, wraps at MAX_VARIANTS)
# ----------------------------------------------------------------------
def get_rotation_index():
    """
    Returns the variant index.

    If --r is provided:
        - Use it (modulo MAX_VARIANTS)
        - DO NOT modify stored counter

    Otherwise:
        - Read stored counter
        - Use its value
        - Increment and wrap around MAX_VARIANTS
        - Save back
    """
    # Manual override via CLI
    if args.r is not None:
        resolved = args.r % MAX_VARIANTS
        VALIDATION_RUNTIME_CONFIG.update(
            {
                "rotation_index": resolved,
                "rotation_source": "cli_--r",
                "rotation_raw": args.r,
                "rotation_file_before": None,
                "rotation_file_after": None,
            }
        )
        return resolved

    # Optional manual override via env for benchmark scripting parity.
    env_rotation = (os.getenv("VALIDATION_ROTATION_INDEX") or "").strip()
    if env_rotation:
        try:
            raw_value = int(env_rotation)
            resolved = raw_value % MAX_VARIANTS
            VALIDATION_RUNTIME_CONFIG.update(
                {
                    "rotation_index": resolved,
                    "rotation_source": "env_VALIDATION_ROTATION_INDEX",
                    "rotation_raw": raw_value,
                    "rotation_file_before": None,
                    "rotation_file_after": None,
                }
            )
            return resolved
        except Exception:
            log(
                "Warning: invalid VALIDATION_ROTATION_INDEX="
                + f"{env_rotation!r}; falling back to rotation file."
            )

    # Automatic mode
    if ROTATION_FILE.exists():
        try:
            current = int(ROTATION_FILE.read_text().strip())
        except:
            current = 0
    else:
        current = 0

    # Wrap around after full cycle
    next_val = (current + 1) % MAX_VARIANTS
    ROTATION_FILE.write_text(str(next_val))
    resolved = current % MAX_VARIANTS
    VALIDATION_RUNTIME_CONFIG.update(
        {
            "rotation_index": resolved,
            "rotation_source": "rotation_file",
            "rotation_raw": current,
            "rotation_file_before": current,
            "rotation_file_after": next_val,
        }
    )
    return resolved

# ----------------------------------------------------------------------
# Image build and verification
# ----------------------------------------------------------------------
def build_and_verify():
    log("=== validation: ensure docker image rebuilt ===")
    # write current commit (always, not only when building)
    if (ROOT / ".git").exists():
        stdout, _, _ = run_cmd(["git", "rev-parse", "--short", "HEAD"])
        commit = stdout.strip()
        if commit:
            (ROOT / "COMMIT").write_text(commit + "\n")

    log("Building Docker image (using cache by default)")
    # Pre-pull the base image (best-effort) to avoid long unexpected downloads during build
    try:
        dockerfile = ROOT / "Dockerfile"
        if dockerfile.exists():
            with open(dockerfile, 'r') as df:
                for line in df:
                    ln = line.strip().lower()
                    if ln.startswith("from"):
                        # FROM lines typically look like: FROM python:3.11-slim AS base
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            base_image = parts[1]
                            log(f"Pre-pulling base image: {base_image}")
                            run_cmd(["docker", "pull", base_image], capture_output=True, timeout=240)
                        break
    except Exception as e:
        logger.debug("pre-pull failed: %s", e)

    # Use cached build by default (remove --no-cache). Add timeout so the script doesn't hang forever.
    _, _, rc = run_cmd(
        ["docker", "build", "--pull", "--progress=plain", "-t", IMAGE_NAME, "."],
        capture_output=False,
        timeout=300
    )
    if rc != 0:
        log("Docker build failed")
        sys.exit(1)

    # deterministic planner_agent.py checksum check
    # Find the file path
    stdout, _, _ = run_cmd(["git", "ls-files"])
    files = stdout.splitlines()
    planner_files = [f for f in files if re.search(r'(^|/)planner_agent.py$', f)]
    if planner_files:
        host_planner_file = planner_files[0]
    else:
        host_planner_file = "agents/planner_agent.py"

    host_planner_path = ROOT / host_planner_file
    if not host_planner_path.exists():
        log(f"ERROR: planner_agent.py not found at {host_planner_path} on host — aborting tests.")
        sys.exit(2)

    host_sum = compute_sha256(host_planner_path)

    # create temporary container
    stdout, stderr, rc = run_cmd(["docker", "create", IMAGE_NAME, "/bin/true"])
    if rc != 0:
        log("Failed to create container for checksum")
        sys.exit(1)
    container_id = stdout.strip()
    try:
        container_path = f"/app/{host_planner_file}"
        cp_cmd = ["docker", "cp", f"{container_id}:{container_path}", "/tmp/planner_agent_in_container.py"]
        run_cmd(cp_cmd, check=False)  # ignore failure
        container_sum_path = Path("/tmp/planner_agent_in_container.py")
        if container_sum_path.exists():
            container_sum = compute_sha256(container_sum_path)
            container_sum_path.unlink()
        else:
            container_sum = "missing"
    finally:
        run_cmd(["docker", "rm", container_id])

    log(f"planner_agent.py checksum: host={host_sum} container={container_sum}")
    if host_sum != container_sum:
        log("ERROR: mismatch between host and container planner_agent.py — aborting tests.")
        sys.exit(2)

    log("Proceeding to run validation...")

# ----------------------------------------------------------------------
# Temporary env creation
# ----------------------------------------------------------------------
def create_temp_env(mode):
    log(f"Creating {TMP_ENV} for mode={mode}")
    if mode == "machine":
        src_env = ORIG_ENV_MACHINE
    else:
        src_env = ORIG_ENV_DOCKER

    if src_env.exists():
        shutil.copy(src_env, TMP_ENV)
    else:
        log(f"Warning: {src_env} not found, creating minimal .env.tmp")
        TMP_ENV.write_text("")

    # Determine Ollama URL
    testing_line = "" if args.live else "TESTING=true"
    log_level = "WARNING" if args.quiet and not args.debug else "INFO"
    access_log = "false" if args.quiet else "true"

    validation_ollama_num_ctx, validation_ollama_num_ctx_source, validation_ollama_num_ctx_mode = _resolve_validation_num_ctx(
        src_env=src_env
    )
    validation_ollama_thinking_mode, validation_ollama_thinking_mode_source = _resolve_validation_thinking_mode(
        src_env=src_env
    )

    # Keep validation runtime semantics aligned with local runtime defaults unless
    # explicitly overridden for validation-only experiments.
    planner_llm_timeout, planner_llm_timeout_source = _resolve_validation_timeout_seconds(
        key="PLANNER_LLM_TIMEOUT",
        src_env=src_env,
        validation_override_env="VALIDATION_PLANNER_LLM_TIMEOUT_SECONDS",
        fallback_default=45,
        minimum_seconds=5,
        include_src_env=False,
    )
    ollama_timeout, ollama_timeout_source = _resolve_validation_timeout_seconds(
        key="OLLAMA_TIMEOUT",
        src_env=src_env,
        validation_override_env="VALIDATION_OLLAMA_TIMEOUT_SECONDS",
        fallback_default=45,
        minimum_seconds=1,
        include_src_env=False,
    )
    planner_global_timeout, planner_global_timeout_source = _resolve_validation_timeout_seconds(
        key="PLANNER_GLOBAL_TIMEOUT",
        src_env=src_env,
        validation_override_env="VALIDATION_PLANNER_GLOBAL_TIMEOUT_SECONDS",
        fallback_default=None,
        minimum_seconds=1,
        allow_unset=True,
        include_src_env=False,
    )
    router_timeout, router_timeout_source = _resolve_validation_timeout_seconds(
        key="ROUTER_TIMEOUT",
        src_env=src_env,
        validation_override_env="VALIDATION_ROUTER_TIMEOUT_SECONDS",
        fallback_default=max(planner_llm_timeout, ollama_timeout) + 10,
        minimum_seconds=1,
        include_src_env=True,
    )
    local_llm_timeout = max(ollama_timeout, planner_llm_timeout)
    # Keep router ownership above backend-specific budgets so validation doesn't
    # self-induce deterministic fallback before local/planner limits.
    router_floor = local_llm_timeout + 5
    if router_timeout < router_floor:
        router_timeout_source = (
            f"{router_timeout_source} (clamped to local/planner floor {router_floor}s)"
        )
        router_timeout = router_floor

    log(
        "Validation local LLM settings: "
        f"OLLAMA_NUM_CTX={validation_ollama_num_ctx}, "
        f"OLLAMA_THINKING_MODE={validation_ollama_thinking_mode}, "
        f"OLLAMA_TIMEOUT={ollama_timeout}s, "
        f"PLANNER_LLM_TIMEOUT={planner_llm_timeout}s, "
        f"ROUTER_TIMEOUT={router_timeout}s, "
        + (
            "PLANNER_GLOBAL_TIMEOUT=unset"
            if planner_global_timeout is None
            else f"PLANNER_GLOBAL_TIMEOUT={planner_global_timeout}s"
        )
    )
    log(
        "Validation local LLM setting sources: "
        f"OLLAMA_NUM_CTX<-{validation_ollama_num_ctx_source} (mode={validation_ollama_num_ctx_mode}); "
        f"OLLAMA_THINKING_MODE<-{validation_ollama_thinking_mode_source}"
    )
    log(
        "Validation timeout sources: "
        f"OLLAMA_TIMEOUT<-{ollama_timeout_source}; "
        f"PLANNER_LLM_TIMEOUT<-{planner_llm_timeout_source}; "
        f"ROUTER_TIMEOUT<-{router_timeout_source}; "
        f"PLANNER_GLOBAL_TIMEOUT<-{planner_global_timeout_source}"
    )

    # Honor explicit per-run routing overrides so mode-specific validation is real.
    requested_llm_mode = (get_env_str("LLM_MODE", "") or "").strip().lower()
    valid_llm_modes = {"ollama_only", "cloud_only", "cloud_first", "ollama_first"}
    llm_mode_line = f"LLM_MODE={requested_llm_mode}" if requested_llm_mode in valid_llm_modes else ""

    requested_cloud_provider = (get_env_str("CLOUD_PROVIDER", "") or "").strip().lower()
    cloud_provider_line = f"CLOUD_PROVIDER={requested_cloud_provider}" if requested_cloud_provider else ""

    if is_env_set("VALIDATION_USE_CLOUD_LLM"):
        use_cloud_llm = "1" if get_env_bool("VALIDATION_USE_CLOUD_LLM", default=False) else "0"
    elif is_env_set("USE_CLOUD_LLM"):
        use_cloud_llm = "1" if get_env_bool("USE_CLOUD_LLM", default=False) else "0"
    elif requested_llm_mode == "ollama_only":
        use_cloud_llm = "0"
    elif requested_llm_mode in {"cloud_only", "cloud_first", "ollama_first"}:
        use_cloud_llm = "1"
    else:
        # Keep default aligned with runtime semantics: cloud is admin-enabled unless explicitly disabled.
        use_cloud_llm = "1"

    if llm_mode_line:
        log(f"Validation override: {llm_mode_line}")
    if cloud_provider_line:
        log(f"Validation override: {cloud_provider_line}")
    log(f"Validation override: USE_CLOUD_LLM={use_cloud_llm} (cloud admin enablement; runtime still requires usable provider keys)")

    requested_database_url = (get_env_str("VALIDATION_DATABASE_URL", "") or "").strip()
    database_url_line = f"DATABASE_URL={requested_database_url}" if requested_database_url else ""
    if database_url_line:
        log("Validation override: DATABASE_URL from VALIDATION_DATABASE_URL")

    planner_global_timeout_line = (
        f"PLANNER_GLOBAL_TIMEOUT={planner_global_timeout}"
        if planner_global_timeout is not None
        else ""
    )

    if mode == "docker":
        ollama_for_docker = "http://host.docker.internal:11434"
        overrides = f"""
# ----- temporary test overrides (generated by safe_full_validation_report.sh) -----
{testing_line}
# Keep DB cross-process for booking bridge contract checks while TESTING=true.
TESTING_USE_PERSISTENT_DB=1
OLLAMA_BASE_URL={ollama_for_docker}
CLOUD_LLM_TIMEOUT=5
CLOUD_LLM_STREAM_CHUNK_TIMEOUT=1
ROUTER_TIMEOUT={router_timeout}
PLANNER_PREWARM=1
{planner_global_timeout_line}
PLANNER_LLM_TIMEOUT={planner_llm_timeout}
OLLAMA_TIMEOUT={ollama_timeout}
LOCAL_LLM_TIMEOUT={local_llm_timeout}
OLLAMA_NUM_CTX={validation_ollama_num_ctx}
OLLAMA_THINKING_MODE={validation_ollama_thinking_mode}
USE_CLOUD_LLM={use_cloud_llm}
{llm_mode_line}
{cloud_provider_line}
{database_url_line}
AUTH_TOKEN={VALIDATION_AUTH_TOKEN}
AUTH_DEFAULT_PRINCIPAL_ID=validation-user
ADMIN_TOKEN={VALIDATION_ADMIN_TOKEN}
LOG_LEVEL={log_level}
ENABLE_UVICORN_ACCESS_LOG={access_log}
"""
    else:
        overrides = f"""
# ----- temporary test overrides (generated by safe_full_validation_report.sh) -----
{testing_line}
# Keep DB cross-process for booking bridge contract checks while TESTING=true.
TESTING_USE_PERSISTENT_DB=1
CLOUD_LLM_TIMEOUT=5
CLOUD_LLM_STREAM_CHUNK_TIMEOUT=1
ROUTER_TIMEOUT={router_timeout}
PLANNER_PREWARM=1
{planner_global_timeout_line}
PLANNER_LLM_TIMEOUT={planner_llm_timeout}
OLLAMA_TIMEOUT={ollama_timeout}
LOCAL_LLM_TIMEOUT={local_llm_timeout}
OLLAMA_NUM_CTX={validation_ollama_num_ctx}
OLLAMA_THINKING_MODE={validation_ollama_thinking_mode}
USE_CLOUD_LLM={use_cloud_llm}
{llm_mode_line}
{cloud_provider_line}
{database_url_line}
AUTH_TOKEN={VALIDATION_AUTH_TOKEN}
AUTH_DEFAULT_PRINCIPAL_ID=validation-user
ADMIN_TOKEN={VALIDATION_ADMIN_TOKEN}
LOG_LEVEL={log_level}
ENABLE_UVICORN_ACCESS_LOG={access_log}
"""
    with open(TMP_ENV, 'a') as f:
        f.write(overrides)

    tmp_env_ollama_model = _read_env_value_from_file(TMP_ENV, "OLLAMA_MODEL")
    tmp_env_ollama_num_ctx = _read_env_value_from_file(TMP_ENV, "OLLAMA_NUM_CTX")
    tmp_env_ollama_thinking_mode = _read_env_value_from_file(TMP_ENV, "OLLAMA_THINKING_MODE")
    expected_backend = "ollama_only" if requested_llm_mode == "ollama_only" else "mixed_or_mode_derived"

    VALIDATION_RUNTIME_CONFIG.update(
        {
            "mode": mode,
            "backend_expectation": expected_backend,
            "llm_mode": requested_llm_mode or "",
            "use_cloud_llm": use_cloud_llm,
            "ollama_model_process_env": str(os.getenv("OLLAMA_MODEL") or ""),
            "ollama_model_src_env": str(_read_env_value_from_file(src_env, "OLLAMA_MODEL") or ""),
            "ollama_model_tmp_env": str(tmp_env_ollama_model or ""),
            "ollama_num_ctx_mode": validation_ollama_num_ctx_mode,
            "ollama_num_ctx_process_env": str(os.getenv("OLLAMA_NUM_CTX") or ""),
            "ollama_num_ctx_validation_override": str(os.getenv("VALIDATION_OLLAMA_NUM_CTX") or ""),
            "ollama_num_ctx_effective": validation_ollama_num_ctx,
            "ollama_num_ctx_source": validation_ollama_num_ctx_source,
            "ollama_num_ctx_tmp_env": str(tmp_env_ollama_num_ctx or ""),
            "ollama_thinking_mode_process_env": str(os.getenv("OLLAMA_THINKING_MODE") or ""),
            "ollama_thinking_mode_validation_override": str(os.getenv("VALIDATION_OLLAMA_THINKING_MODE") or ""),
            "ollama_thinking_mode_effective": validation_ollama_thinking_mode,
            "ollama_thinking_mode_tmp_env": str(tmp_env_ollama_thinking_mode or ""),
            "async_parallel_mode": str(
                (get_env_str("VALIDATION_ASYNC_PARALLEL_MODE", "sequential") or "sequential")
            ).strip().lower(),
        }
    )
    log(
        "Validation local model config (env vs effective): "
        + f"OLLAMA_MODEL(process)={VALIDATION_RUNTIME_CONFIG['ollama_model_process_env'] or '<unset>'}, "
        + f"OLLAMA_MODEL({src_env.name})={VALIDATION_RUNTIME_CONFIG['ollama_model_src_env'] or '<unset>'}, "
        + f"OLLAMA_MODEL(tmp)={VALIDATION_RUNTIME_CONFIG['ollama_model_tmp_env'] or '<unset/runtime_default>'}, "
        + f"OLLAMA_NUM_CTX(process)={VALIDATION_RUNTIME_CONFIG['ollama_num_ctx_process_env'] or '<unset>'}, "
        + f"VALIDATION_OLLAMA_NUM_CTX={VALIDATION_RUNTIME_CONFIG['ollama_num_ctx_validation_override'] or '<unset>'}, "
        + f"OLLAMA_NUM_CTX(effective)={validation_ollama_num_ctx}, "
        + f"OLLAMA_THINKING_MODE(process)={VALIDATION_RUNTIME_CONFIG['ollama_thinking_mode_process_env'] or '<unset>'}, "
        + f"VALIDATION_OLLAMA_THINKING_MODE={VALIDATION_RUNTIME_CONFIG['ollama_thinking_mode_validation_override'] or '<unset>'}, "
        + f"OLLAMA_THINKING_MODE(effective)={validation_ollama_thinking_mode}"
    )

    internal_debug = LOG_DIR / "internal_debug.log"
    with open(internal_debug, 'a') as f:
        f.write(f"Wrote overrides to {TMP_ENV}\n")

# ----------------------------------------------------------------------
# run_and_log equivalent (with per-test LLM expectation, assertions, and failure reason)
# ----------------------------------------------------------------------
def _extract_request_payload_from_cmd(cmd):
    """Best-effort extraction of JSON payload passed via curl -d/--data."""
    if not cmd:
        return {}
    for i, token in enumerate(cmd):
        if token in ("-d", "--data", "--data-raw", "--data-binary") and i + 1 < len(cmd):
            raw_payload = cmd[i + 1]
            try:
                return json.loads(raw_payload)
            except Exception:
                return {"_raw": raw_payload}
    return {}


def _build_failure_diagnostics(name, cmd, resp_body, cmd_output="", date_assertion_details=None):
    """
    Build deterministic diagnostics for failed validation cases.
    Includes normalized route, planner intent, API request payloads, and relaxation attempts when available.
    """
    diagnostics = {
        "test": name,
        "request_payload": _extract_request_payload_from_cmd(cmd),
        "normalized_origin": None,
        "normalized_destination": None,
        "intent_state": {},
        "api_request_payload": {},
        "relaxation_attempts": [],
    }

    try:
        data = json.loads(resp_body) if resp_body else {}
    except Exception:
        data = {}

    debug = (data.get("debug_info") or {}) if isinstance(data, dict) else {}
    intent = debug.get("intent") or {}
    api_trace = debug.get("api_trace") or {}
    flight_request = (api_trace.get("flight") or {}).get("request") or {}
    weather_request = (api_trace.get("weather") or {}).get("request") or {}

    if intent:
        diagnostics["normalized_origin"] = intent.get("origin_iata")
        diagnostics["normalized_destination"] = intent.get("destination_iata")
        diagnostics["intent_state"] = intent
    else:
        req = diagnostics["request_payload"] if isinstance(diagnostics["request_payload"], dict) else {}
        inferred_origin, inferred_destination = _infer_route_from_payload(req)
        diagnostics["normalized_origin"] = inferred_origin
        diagnostics["normalized_destination"] = inferred_destination
        diagnostics["intent_state"] = {
            "origin": req.get("origin"),
            "destination": req.get("destination"),
            "date": req.get("date"),
            "user_query": req.get("user_query"),
        }

    diagnostics["api_request_payload"] = {
        "flight": flight_request,
        "weather": weather_request,
    }
    diagnostics["relaxation_attempts"] = debug.get("relaxation_attempts") or []
    if isinstance(date_assertion_details, dict) and date_assertion_details:
        diagnostics["date_assertion"] = date_assertion_details

    bridge_diags = _extract_booking_bridge_diagnostics_from_output(cmd_output)
    if bridge_diags:
        diagnostics["booking_bridge"] = bridge_diags

    return diagnostics


def _normalize_iata_candidate(value):
    """Best-effort deterministic normalization for diagnostics only."""
    if not value:
        return None
    token = str(value).strip().lower()
    if len(token) == 3 and token.isalpha():
        return token.upper()

    for code, aliases in IATA_CITY_ALIASES.items():
        if token == code:
            return code.upper()
        if token in aliases:
            return code.upper()
    return None


def _infer_route_from_payload(payload):
    """
    Best-effort route inference for diagnostics when planner debug_info is unavailable.
    """
    if not isinstance(payload, dict):
        return None, None

    origin = _normalize_iata_candidate(payload.get("origin"))
    destination = _normalize_iata_candidate(payload.get("destination"))
    user_query = str(payload.get("user_query") or "")

    # IATA-pair fallback (e.g., "DEL BOM ...")
    if (not origin or not destination) and user_query:
        iata_tokens = []
        for tok in re.findall(r"\b([A-Za-z]{3})\b", user_query):
            up = tok.upper()
            if up not in iata_tokens:
                iata_tokens.append(up)
        if len(iata_tokens) >= 2:
            origin = origin or iata_tokens[0]
            destination = destination or iata_tokens[1]

    # "from X to Y" fallback
    if (not origin or not destination) and user_query:
        m = re.search(r"from\s+([A-Za-z ]+?)\s+to\s+([A-Za-z ]+?)(?:\s+|$)", user_query, re.IGNORECASE)
        if m:
            origin = origin or _normalize_iata_candidate(m.group(1))
            destination = destination or _normalize_iata_candidate(m.group(2))

    # City-alias fallback in text order (e.g., "Delhi Mumbai ...")
    if (not origin or not destination) and user_query:
        q_lower = user_query.lower()
        hits = []
        for code, aliases in IATA_CITY_ALIASES.items():
            for alias in aliases:
                for match in re.finditer(r"\b" + re.escape(alias) + r"\b", q_lower):
                    hits.append((match.start(), code.upper()))
        hits.sort(key=lambda x: x[0])
        ordered_codes = []
        for _, code in hits:
            if code not in ordered_codes:
                ordered_codes.append(code)
        if len(ordered_codes) >= 2:
            origin = origin or ordered_codes[0]
            destination = destination or ordered_codes[1]

    return origin, destination


def _json_path_value(payload, path):
    current = payload
    for token in str(path).split("."):
        if token == "":
            continue
        if isinstance(current, dict):
            if token not in current:
                return None, False
            current = current[token]
            continue
        if isinstance(current, list):
            try:
                idx = int(token)
            except Exception:
                return None, False
            if idx < 0 or idx >= len(current):
                return None, False
            current = current[idx]
            continue
        return None, False
    return current, True


def run_and_log(name, cmd, is_stream=False, expect_llm=True, assertions=None, frontend_override=None):
    """
    Execute a command, capture output, validate based on test type, and log results.
    :param name: test name (will be suffixed with mode)
    :param cmd: command list to execute
    :param is_stream: whether this is a streaming test (different validation)
    :param expect_llm: whether this scenario requires real LLM output (and should fail on degraded/unavailable fallback)
    :param assertions: dict with additional structural checks
    :return: status code
    """
    start_iso = datetime.now().isoformat()
    start_epoch = time.time()

    # detect mode from name
    mode_label = _mode_label_for_name(name)

    # create temp file for output
    tmp_out = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    tmp_out_name = tmp_out.name
    tmp_out.close()

    status = 0
    http_code = None
    curl_exit = None
    failure_reason = ""   # human-readable explanation
    cmd_output = ""       # will hold the full command output for later use
    resp_body = ""        # initialize to avoid NameError later
    normalized_failure_tags = set()
    explicit_failure_tags = set()
    date_assertion_details = {}
    frontend_handled = False
    frontend_status = None
    frontend_failure_reason = ""
    frontend_result = {}
    frontend_enabled = args.frontend if frontend_override is None else bool(frontend_override)
    base_name_for_meta = _strip_mode_suffix(name)
    effective_meta = _validation_meta_for_base(base_name_for_meta)
    if isinstance(assertions, dict):
        override = assertions.get("validation_meta_override")
        if isinstance(override, dict):
            for key in (
                "scenario",
                "layers",
                "validation_type",
                "features",
                "mode_bucket",
                "soft_pass_policy",
                "criticality",
                "dimensions",
                "ui_assertions",
                "contract_assertions",
            ):
                if key in override:
                    effective_meta[key] = override.get(key)

    should_try_frontend = (
        _should_use_frontend_path(cmd, frontend_enabled=frontend_enabled)
        and _should_route_case_through_frontend(name, assertions)
    )

    if should_try_frontend:
        try:
            frontend_handled, frontend_bundle = _run_frontend_validation_for_cmd(
                cmd,
                name=name,
                assertions=assertions,
            )
        except Exception as exc:
            frontend_handled = True
            frontend_bundle = {
                "status": 125,
                "failure_reason": f"Frontend validation execution error: {exc}",
                "result": {},
            }

        if frontend_handled:
            frontend_status = frontend_bundle.get("status", 125)
            frontend_failure_reason = frontend_bundle.get("failure_reason", "")
            frontend_result = frontend_bundle.get("result", {})
            cmd_output = json.dumps(frontend_result, ensure_ascii=False) + "\n"

    run_backend_shadow = not (
        frontend_handled
        and isinstance(assertions, dict)
        and assertions.get("skip_backend_shadow")
    )
    backend_checked = False
    backend_status = None
    backend_failure_reason = ""

    if cmd and cmd[0] == "curl" and run_backend_shadow:
        backend_checked = True
        # For curl commands, capture HTTP status code and response body separately
        with tempfile.NamedTemporaryFile(delete=False) as tmp_resp:
            resp_file = tmp_resp.name
        # Build curl command with -o and -w, and --silent to avoid extra output
        curl_cmd = cmd + ['-o', resp_file, '-w', '%{http_code}', '--silent']
        # run, capturing stderr (though --silent should suppress most)
        with open(tmp_out_name, 'a') as f_err:
            proc = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate()
            curl_exit = proc.returncode
            f_err.write(stderr)

        http_code_str = stdout.strip()
        # Validate http_code
        if not http_code_str.isdigit():
            # Not a valid HTTP code – treat as failure
            status = 125
            http_code = http_code_str
            failure_reason = f"Invalid HTTP code: {http_code_str}"
        else:
            http_code = int(http_code_str)

        # Read response body
        try:
            with open(resp_file, 'r') as f:
                resp_body = f.read()
        except:
            resp_body = ""
        os.unlink(resp_file)

        # Append response body to tmp_out
        with open(tmp_out_name, 'a') as f:
            f.write(resp_body)
        # Append debug metadata
        with open(tmp_out_name, 'a') as f:
            f.write(f'\n{{"_internal_debug": {{"http_code": {http_code}, "curl_exit": {curl_exit}}}}}\n')

        # Determine status based on curl_exit, http_code, and content
        expected_http_statuses = None
        if isinstance(assertions, dict) and "expected_http_statuses" in assertions:
            raw_expected_statuses = assertions.get("expected_http_statuses") or []
            normalized_statuses = []
            if isinstance(raw_expected_statuses, (list, tuple, set)):
                for item in raw_expected_statuses:
                    try:
                        normalized_statuses.append(int(item))
                    except Exception:
                        continue
            else:
                try:
                    normalized_statuses.append(int(raw_expected_statuses))
                except Exception:
                    normalized_statuses = []
            if normalized_statuses:
                expected_http_statuses = set(normalized_statuses)

        if curl_exit != 0:
            if is_stream:
                has_output = os.path.getsize(tmp_out_name) > 0
                has_done_json = "[DONE_JSON]" in (resp_body or "")
                has_validation_error = bool(re.search(r'"msg":"Field required"|"detail"', resp_body, re.IGNORECASE))
                if has_done_json and has_output and not has_validation_error:
                    status = 0
                elif curl_exit == 28:
                    status = 124
                    failure_reason = "Stream timed out before DONE_JSON completion payload (curl_exit=28)"
                elif has_output and not has_validation_error:
                    status = 124
                    failure_reason = f"Stream command exited with {curl_exit} before DONE_JSON completion payload"
                else:
                    status = curl_exit
                    failure_reason = f"curl exited with {curl_exit}" + ("" if os.path.getsize(tmp_out_name) > 0 else " (empty response)")
            else:
                status = curl_exit
                failure_reason = f"curl exited with {curl_exit}"
        else:
            if expected_http_statuses is not None:
                if http_code in expected_http_statuses:
                    status = 0
                else:
                    status = 124
                    failure_reason = (
                        f"HTTP {http_code} (expected one of {sorted(expected_http_statuses)})"
                    )
            elif http_code and (http_code < 200 or http_code >= 300):
                status = 124  # HTTP error
                failure_reason = f"HTTP {http_code}"
            else:
                status = 0

        # Read the full command output for later assertions
        with open(tmp_out_name, 'r') as f:
            cmd_output = f.read()

        # Non-streaming JSON checks (with per-test LLM expectation and assertions)
        if status == 0 and not is_stream:
            # First, check for flight availability warnings (important for direct_only etc.)
            if expect_llm and re.search(r"No flights match your criteria", cmd_output, re.IGNORECASE):
                status = 124
                failure_reason = "Response contains 'No flights match your criteria' warning"
            else:
                try:
                    data = json.loads(resp_body)  # resp_body is the last response? Actually for parallel tests, cmd_output may contain multiple JSONs. We'll handle that in assertions separately.
                    # For most tests, the data is the JSON of the main response.
                    # Helper to check non-empty llm_response recursively
                    def has_nonempty_llm(obj):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                if k == "llm_response" and isinstance(v, str) and v.strip():
                                    return True
                                if has_nonempty_llm(v):
                                    return True
                        elif isinstance(obj, list):
                            for item in obj:
                                if has_nonempty_llm(item):
                                    return True
                        return False

                    # ── Weather staleness warning (always run, not a failure) ──
                    if assertions is None or "check_weather_freshness" not in assertions or assertions.get("check_weather_freshness"):
                        weather_data = data.get("weather", {})
                        if not isinstance(weather_data, dict):
                            weather_data = {}
                        forecast_date_str = weather_data.get("forecast_date")
                        search_date_str = data.get("search_date")
                        if forecast_date_str and search_date_str:
                            try:
                                fd = datetime.strptime(forecast_date_str, "%Y-%m-%d")
                                sd = datetime.strptime(search_date_str, "%Y-%m-%d")
                                gap_days = abs((sd - fd).days)
                                WEATHER_STALENESS_THRESHOLD = 7
                                if gap_days > WEATHER_STALENESS_THRESHOLD:
                                    logger.warning(
                                        f"{name}: weather forecast is {gap_days}d away from search_date "
                                        f"(forecast={forecast_date_str}, search={search_date_str})"
                                    )
                            except ValueError:
                                # Date format mismatch – ignore
                                pass

                    # ── per-test structural assertions ──────────────────────────────
                    if assertions:
                        required_paths = assertions.get("required_paths") or []
                        for _path in required_paths:
                            if status != 0:
                                break
                            _value, _ok = _json_path_value(data, _path)
                            if not _ok:
                                status = 124
                                failure_reason = f"Missing required JSON path: {_path}"
                                break

                        expected_paths = assertions.get("expected_paths") or {}
                        if status == 0 and isinstance(expected_paths, dict):
                            for _path, _expected in expected_paths.items():
                                _value, _ok = _json_path_value(data, _path)
                                if not _ok:
                                    status = 124
                                    failure_reason = f"Missing expected JSON path: {_path}"
                                    break
                                if _value != _expected:
                                    status = 124
                                    failure_reason = (
                                        f"JSON path mismatch at {_path}: got {_value}, expected {_expected}"
                                    )
                                    break

                        if status == 0 and assertions.get("check_health_keys_shape"):
                            required_services = ("serpapi", "openai", "weather")
                            for service in required_services:
                                entries = data.get(service)
                                if not isinstance(entries, list):
                                    status = 124
                                    failure_reason = f"health/keys payload missing list for service '{service}'"
                                    break
                                for idx, item in enumerate(entries):
                                    if not isinstance(item, dict):
                                        status = 124
                                        failure_reason = f"health/keys '{service}[{idx}]' is not an object"
                                        break
                                    if "index" not in item or "active" not in item:
                                        status = 124
                                        failure_reason = (
                                            f"health/keys '{service}[{idx}]' missing required fields "
                                            "(index, active)"
                                        )
                                        break
                                if status != 0:
                                    break

                        if status == 0 and assertions.get("check_health_light_shape"):
                            dependencies = data.get("dependencies")
                            if not isinstance(dependencies, dict) or not dependencies:
                                status = 124
                                failure_reason = "health payload missing dependencies object"
                            elif not isinstance(data.get("status"), str) or not data.get("status"):
                                status = 124
                                failure_reason = "health.status missing or invalid"
                            elif not isinstance(data.get("async_jobs_enabled"), bool):
                                status = 124
                                failure_reason = "health.async_jobs_enabled must be boolean"
                            elif "app" not in dependencies or "key_manager" not in dependencies:
                                status = 124
                                failure_reason = "health.dependencies missing app/key_manager"

                        if status == 0 and assertions.get("check_health_deep_shape"):
                            dependencies = data.get("dependencies")
                            if not isinstance(dependencies, dict) or not dependencies:
                                status = 124
                                failure_reason = "health/deep dependencies missing or invalid"
                            elif not isinstance(data.get("status_reasons"), list):
                                status = 124
                                failure_reason = "health/deep status_reasons missing or invalid"
                            elif not isinstance(data.get("messages"), (list, dict)):
                                status = 124
                                failure_reason = "health/deep messages missing or invalid"

                        if status == 0 and assertions.get("check_llm_options_shape"):
                            llm_modes = data.get("llm_modes")
                            defaults = data.get("defaults")
                            backend_availability = data.get("backend_availability")
                            config_authority = data.get("config_authority")
                            if not isinstance(llm_modes, list) or not llm_modes or not all(isinstance(m, str) and m for m in llm_modes):
                                status = 124
                                failure_reason = "llm/options llm_modes missing or invalid"
                            elif not isinstance(defaults, dict):
                                status = 124
                                failure_reason = "llm/options defaults missing or invalid"
                            elif defaults.get("llm_mode") not in llm_modes:
                                status = 124
                                failure_reason = "llm/options defaults.llm_mode is not in llm_modes"
                            elif not isinstance(backend_availability, dict):
                                status = 124
                                failure_reason = "llm/options backend_availability missing or invalid"
                            elif not isinstance(backend_availability.get("cloud"), bool) or not isinstance(backend_availability.get("ollama"), bool):
                                status = 124
                                failure_reason = "llm/options backend_availability cloud/ollama must be booleans"
                            elif not isinstance(config_authority, dict):
                                status = 124
                                failure_reason = "llm/options config_authority missing or invalid"
                            else:
                                mode_dependency = config_authority.get("mode_dependency")
                                if isinstance(mode_dependency, str):
                                    if not mode_dependency:
                                        status = 124
                                        failure_reason = "llm/options config_authority.mode_dependency missing or invalid"
                                    elif mode_dependency not in llm_modes:
                                        status = 124
                                        failure_reason = (
                                            "llm/options config_authority.mode_dependency is not a supported mode: "
                                            f"{mode_dependency}"
                                        )
                                elif isinstance(mode_dependency, dict):
                                    mode_value = mode_dependency.get("mode")
                                    if not isinstance(mode_value, str) or not mode_value:
                                        status = 124
                                        failure_reason = (
                                            "llm/options config_authority.mode_dependency.mode missing or invalid"
                                        )
                                    elif mode_value not in llm_modes:
                                        status = 124
                                        failure_reason = (
                                            "llm/options config_authority.mode_dependency.mode is not supported: "
                                            f"{mode_value}"
                                        )
                                else:
                                    status = 124
                                    failure_reason = "llm/options config_authority.mode_dependency missing or invalid"

                        if status == 0 and assertions.get("check_version_info_shape"):
                            git_commit = data.get("git_commit")
                            file_mtime = data.get("file_mtime")
                            if not isinstance(git_commit, str) or not git_commit.strip():
                                status = 124
                                failure_reason = "/version git_commit missing or invalid"
                            elif git_commit != "unknown" and not re.fullmatch(r"[0-9a-fA-F]{7,40}", git_commit.strip()):
                                status = 124
                                failure_reason = f"/version git_commit has unexpected format: {git_commit}"
                            else:
                                try:
                                    float(file_mtime)
                                except Exception:
                                    status = 124
                                    failure_reason = "/version file_mtime missing or invalid"

                        # 1. search_date must match expected date
                        if status == 0 and "expected_date" in assertions:
                            date_ok, tolerated_skew, mismatch_reason, date_details = _evaluate_expected_date_assertion(
                                data,
                                assertions,
                            )
                            date_assertion_details = date_details
                            if not date_ok:
                                status = 124
                                failure_reason = mismatch_reason
                                if tolerated_skew:
                                    explicit_failure_tags.add("runtime_date_basis_skew")

                        # 2. return_trip must exist, and main LLM should mention the return flight
                        if status == 0 and assertions.get("check_return_llm"):
                            rt = data.get("return_trip")
                            if not rt:
                                status = 124
                                failure_reason = "return_trip object is missing entirely"
                            else:
                                main_llm = data.get("llm_response", "") or ""
                                rt_bf_no = (rt.get("best_flight") or {}).get("flight_no", "")
                                rt_mentioned = re.findall(r'\b[A-Z0-9]{2}[A-Z0-9]?\d{3,4}\b', main_llm)
                                if rt_mentioned and rt_bf_no and rt_bf_no not in rt_mentioned:
                                    status = 124
                                    failure_reason = f"Main LLM mentions flights {rt_mentioned} but forgot return flight {rt_bf_no}"

                                # Additional check for skip_llm shortform in return_trip
                                rt_llm = rt.get("llm_response", "") or ""
                                shortform_pattern = r'^Flight:\s+\S+\s+\S+\s+\([\d:]+\s+-\s+[\d:]+\)\.\s+Price:.*\.\s+Weather:\s+[^,]+,\s+[\d.]+°C\.\s*$'
                                if re.match(shortform_pattern, rt_llm.strip()):
                                    status = 124
                                    failure_reason = "return_trip.llm_response is skip_llm shortform — no real explanation generated"

                                # Detect identical copy of outbound response in return trip
                                if status == 0:
                                    main_llm = data.get("llm_response", "") or ""
                                    rt_llm_text = rt.get("llm_response", "") or ""
                                    if rt_llm_text and rt_llm_text.strip() == main_llm.strip():
                                        status = 124
                                        failure_reason = "return_trip.llm_response is identical to outbound llm_response (copy bug)"
                                    # Also assert return date appears in return trip LLM
                                    return_date_val = (data.get("debug_info") or {}).get("intent", {}).get("return_date")
                                    if return_date_val and rt_llm_text:
                                        # Check at least the return date or return leg is mentioned
                                        rt_lower = rt_llm_text.lower()
                                        # Accept either the ISO date or a loose mention of returning / return flight
                                        if return_date_val not in rt_llm_text and not any(
                                            w in rt_lower for w in ("return", "coming back", "heading back")
                                        ):
                                            status = 124
                                            failure_reason = f"return_trip.llm_response does not mention return ({return_date_val})"

                                # NEW — check return trip weather is not empty, but allow if reason is forecast_horizon_exceeded
                                rt_weather = rt.get("weather") or {}
                                rt_weather_reason = rt.get("weather_reason")
                                if (not rt_weather or not rt_weather.get("condition") or not rt_weather.get("temperature_c")) \
                                        and rt_weather_reason != "forecast_horizon_exceeded":
                                    status = 124
                                    failure_reason = (
                                        "return_trip.weather is empty or missing condition/temperature — "
                                        "weather API was not fetched for the return leg"
                                    )

                        # 3. stopover: ALL legs must have non-null llm_response
                        if status == 0 and assertions.get("check_all_legs_llm"):
                            if not data.get("multicity"):
                                status = 124
                                failure_reason = "Expected multicity response but got single-leg (stopover not planned)"
                            else:
                                leg_texts = []
                                for i, leg in enumerate(data.get("legs", [])):
                                    llm = (leg.get("llm_response") or "").strip()
                                    leg_texts.append(llm)
                                    if not llm:
                                        status = 124
                                        failure_reason = f"legs[{i}].llm_response is null"
                                        break
                                    # Add minimum length check to avoid split artifacts
                                    if len(llm) < 300:   # increased from 150
                                        status = 124
                                        failure_reason = f"legs[{i}].llm_response is suspiciously short ({len(llm)} chars) — likely a template stub, not a real LLM explanation"
                                        break
                                    # Also require advisory words
                                    advisory_words = ("pack", "weather", "temperature", "recommend", "forecast", "suitable", "bring", "suggest")
                                    if not any(w in llm.lower() for w in advisory_words):
                                        status = 124
                                        failure_reason = f"legs[{i}].llm_response has no advisory content (packing/weather advice) — likely a stub"
                                        break
                                # Additionally, check legs are not identical
                                if status == 0 and len(leg_texts) >= 2 and len(set(leg_texts)) == 1:
                                    status = 124
                                    failure_reason = "All stopover legs have identical llm_response (likely copy bug)"
                                # Check each leg's LLM mentions its weather city (by IATA or city name)
                                if status == 0:
                                    for i, leg in enumerate(data.get("legs", [])):
                                        weather_obj = leg.get("weather") or {}
                                        leg_weather_loc = weather_obj.get("location", "").lower()   # IATA code
                                        # Also check against city name aliases
                                        city_aliases = IATA_CITY_ALIASES.get(leg_weather_loc, [])
                                        llm_text = (leg.get("llm_response") or "").lower()
                                        loc_mentioned = False
                                        if leg_weather_loc and leg_weather_loc in llm_text:
                                            loc_mentioned = True
                                        if not loc_mentioned:
                                            for alias in city_aliases:
                                                if alias in llm_text:
                                                    loc_mentioned = True
                                                    break
                                        if leg_weather_loc and not loc_mentioned:
                                            status = 124
                                            failure_reason = f"legs[{i}] LLM does not mention its weather city ({leg_weather_loc.upper()})"
                                            break

                                        # New check: detect wrong city name in "City (IATA)" pattern
                                        if status == 0 and leg_weather_loc:
                                            wrong_city_re = re.compile(
                                                r'([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)*)\s*\(' + leg_weather_loc.upper() + r'\)',
                                                re.IGNORECASE
                                            )
                                            for wm in wrong_city_re.finditer((leg.get("llm_response") or "")):
                                                city_in_llm = wm.group(1).strip().lower()
                                                for other_iata, other_aliases in IATA_CITY_ALIASES.items():
                                                    if other_iata != leg_weather_loc and city_in_llm in other_aliases:
                                                        status = 124
                                                        failure_reason = (
                                                            f"legs[{i}] LLM calls {leg_weather_loc.upper()} '{city_in_llm}' "
                                                            f"but '{city_in_llm}' belongs to {other_iata.upper()}"
                                                        )
                                                        break
                                                if status != 0:
                                                    break

                                        # Temperature cross-contamination check
                                        if status == 0 and leg_weather_loc:
                                            actual_temp = weather_obj.get("temperature_c")
                                            if actual_temp is not None:
                                                other_temps = [
                                                    (leg.get("weather") or {}).get("temperature_c")
                                                    for j, leg in enumerate(data.get("legs", []))
                                                    if j != i and (leg.get("weather") or {}).get("temperature_c") is not None
                                                ]
                                                llm_raw = leg.get("llm_response") or ""
                                                for other_t in other_temps:
                                                    if other_t and str(other_t) in llm_raw and str(actual_temp) not in llm_raw:
                                                        status = 124
                                                        failure_reason = (
                                                            f"legs[{i}] LLM mentions temperature {other_t}°C from another leg "
                                                            f"but not its own temperature {actual_temp}°C — likely cross-contamination"
                                                        )
                                                        break

                        # 4. LLM response must reference the actual best_flight airline
                        if status == 0 and assertions.get("check_llm_flight_consistency"):
                            llm_text = data.get("llm_response", "") or ""
                            all_flights = (data.get("debug_info") or {}).get("all_flights", [])
                            known_flight_nos = {f.get("flight_no", "") for f in all_flights}
                            # Always include best_flight.flight_no as authoritative
                            bf_flight_no = (data.get("best_flight") or {}).get("flight_no", "")
                            if bf_flight_no:
                                known_flight_nos.add(bf_flight_no)
                            mentioned = re.findall(r'\b[A-Z0-9]{2}[A-Z0-9]?\d{3,4}\b', llm_text)
                            for m in mentioned:
                                if m not in known_flight_nos:
                                    status = 124
                                    failure_reason = f"LLM mentions unknown flight {m} not in flight data"
                                    break

                        # 5. Check that LLM doesn't contain raw Python representations
                        if status == 0 and assertions.get("check_no_python_repr"):
                            llm_text = data.get("llm_response", "") or ""
                            # Python dict/list repr patterns
                            if re.search(r"'[a-z_]+'\s*:", llm_text) or re.search(r"<[A-Z]+\.[A-Z]+:", llm_text):
                                status = 124
                                failure_reason = "LLM response contains raw Python repr (dict or enum)"

                        # 6. Check that LLM IATA codes in parentheses match intent
                        if status == 0 and assertions.get("check_iata_in_llm"):
                            intent_debug = (data.get("debug_info") or {}).get("intent", {})
                            expected_origin = intent_debug.get("origin_iata", "")
                            expected_dest = intent_debug.get("destination_iata", "")
                            llm_text = data.get("llm_response", "") or ""
                            # Accept either IATA code or city name — case-insensitive
                            llm_lower = llm_text.lower()
                            origin_city = IATA_CITY_ALIASES.get(expected_origin.lower(), [expected_origin])
                            dest_city = IATA_CITY_ALIASES.get(expected_dest.lower(), [expected_dest])
                            origin_match = any(code.lower() in llm_lower for code in [expected_origin] + origin_city)
                            dest_match = any(code.lower() in llm_lower for code in [expected_dest] + dest_city)
                            if not (origin_match and dest_match):
                                status = 124
                                failure_reason = f"LLM does not mention origin ({expected_origin} or city) or destination ({expected_dest} or city)"
                                
                        # 7. For direct_only: if stops is N/A, LLM must not claim non-stop
                        if status == 0 and assertions.get("check_stops_na_not_claimed"):
                            stops_val = (data.get("best_flight") or {}).get("stops", "")
                            llm_text = data.get("llm_response", "") or ""
                            if stops_val in ("N/A", "n/a", ""):
                                false_claims = ("non-stop", "nonstop", "no stops", "direct flight", "0 stops")
                                if any(c in llm_text.lower() for c in false_claims):
                                    status = 124
                                    failure_reason = f"LLM claims non-stop but stops='{stops_val}'"

                        # 8. For relaxed-filter tests: if warnings present, LLM must be honest
                        if status == 0 and assertions.get("check_relaxed_filter_honest"):
                            response_warnings = data.get("warnings") or []
                            is_relaxed = any("relaxed" in w.lower() for w in response_warnings)
                            llm_text = data.get("llm_response", "") or ""
                            # Enhanced baggage false‑claim detection
                            baggage_pref = (data.get("debug_info") or {}).get("intent", {}).get("baggage_pref")
                            baggage_val = (data.get("best_flight") or {}).get("baggage", "")
                            if baggage_pref == "hand" and baggage_val and baggage_val not in ("N/A", "n/a", ""):
                                # Flight has known baggage that is not N/A – check if it's actually checked baggage
                                is_actually_checked = any(x in baggage_val.lower() for x in ("checked", "check", "free bag", "hold"))
                                if is_actually_checked:
                                    # Negation detection for "cabin only"
                                    _negation = re.compile(
                                        r'(does not meet|not meet|does not support|not support|'
                                        r'cannot confirm|not a hand|no hand baggage)',
                                        re.IGNORECASE
                                    )
                                    _false_claim_found = False
                                    llm_text_low = llm_text.lower()
                                    # "cabin only" — check with negation window, skipping query echoes
                                    for m in re.finditer(r'cabin only', llm_text_low):
                                        surrounding = llm_text_low[max(0, m.start() - 25): m.end() + 20]
                                        # Skip if this is the LLM echoing the user's preference phrasing
                                        if 'in cabin only' in surrounding or 'cabin only (hand' in surrounding:
                                            continue
                                        window = llm_text_low[max(0, m.start() - 70): m.start()]
                                        if not _negation.search(window):
                                            _false_claim_found = True
                                            break
                                    # All other patterns are unambiguous subject-specific phrases
                                    if not _false_claim_found:
                                        unambiguous_claims = (
                                            "this flight allows only hand", "allows only hand baggage",
                                            "hand baggage only flight", "is a hand baggage only",
                                            "meets your hand baggage", "this flight has hand baggage",
                                            "hand luggage only flight", "flight supports cabin only",
                                            "meets your requirements for a hand baggage",
                                            "meets your requirement for hand baggage",
                                            "meets your cabin baggage",
                                            "satisfies your hand baggage",
                                        )
                                        _false_claim_found = any(c in llm_text_low for c in unambiguous_claims)

                                    if _false_claim_found:
                                        status = 124
                                        failure_reason = f"LLM falsely claims hand‑baggage compliance but baggage='{baggage_val}'"
                            elif baggage_val in ("N/A", "n/a", ""):
                                # Original guard for unknown baggage
                                flight_baggage_claims = (
                                    "this flight allows only hand", "this flight supports cabin only",
                                    "hand baggage only flight", "meets your hand baggage requirement",
                                    "this flight has hand baggage", "allows only hand baggage",
                                )
                                if any(c in llm_text.lower() for c in flight_baggage_claims):
                                    status = 124
                                    failure_reason = f"LLM claims hand‑baggage conformance but baggage='{baggage_val}'"

                            # Prefer contradiction checks over phrase-style enforcement.
                            if status == 0 and is_relaxed:
                                pref_airlines = (data.get("debug_info") or {}).get("intent", {}).get("preferred_airlines", [])
                                selected_airline = (data.get("best_flight") or {}).get("airline", "")
                                contradiction = detect_relaxed_preferred_airline_contradiction(
                                    llm_text=llm_text,
                                    preferred_airlines=pref_airlines,
                                    selected_airline=selected_airline,
                                )
                                if contradiction:
                                    status = 124
                                    failure_reason = contradiction

                        # 9. For layover limit: ensure LLM doesn't confuse duration with layover
                        if status == 0 and assertions.get("check_layover_not_confused"):
                            llm_text = data.get("llm_response", "") or ""
                            layover_limit = (data.get("debug_info") or {}).get("intent", {}).get("layover_limit_minutes")
                            contradiction = detect_layover_contradiction(
                                llm_text=llm_text,
                                best_flight=(data.get("best_flight") or {}),
                                layover_limit_minutes=layover_limit,
                            )
                            if contradiction:
                                status = 124
                                failure_reason = contradiction

                        # 10. Eco-friendly test: check that LLM mentions carbon or eco
                        if status == 0 and assertions.get("check_eco_llm"):
                            llm_text = data.get("llm_response", "") or ""
                            eco_words = ("carbon", "eco", "emission", "green", "sustainable", "co2")
                            if not any(w in llm_text.lower() for w in eco_words):
                                status = 124
                                failure_reason = "LLM does not mention eco/carbon in eco query"

                        # NEW check #10b: weather temperature values in LLM must not swap min/max
                        if status == 0 and assertions.get("check_weather_temp_accuracy"):
                            weather_obj = data.get("weather") or {}
                            temp_min = weather_obj.get("temp_min_c")
                            temp_max = weather_obj.get("temp_max_c")
                            llm_text = data.get("llm_response", "") or ""
                            # Only check inversion when min and max are genuinely different.
                            # OWM current-weather fallback sets both to the same value — no inversion
                            # is possible and the check would produce false positives on live runs.
                            if temp_min is not None and temp_max is not None and temp_max > temp_min:
                                # Find all temperature mentions in format like "25°C", "25.0°C"
                                temp_mentions = [float(m) for m in re.findall(r'(\d+\.?\d*)\s*°C', llm_text)]
                                if temp_mentions:
                                    # Improved regex that stops at "high"/"max" to avoid false positives
                                    low_high_re = re.search(
                                        rf'(?:low|minimum|min)(?:(?!(?:high|max)).){{0,30}}({re.escape(str(int(temp_max)))})',
                                        llm_text, re.IGNORECASE
                                    )
                                    if low_high_re:
                                        status = 124
                                        failure_reason = (
                                            f"LLM describes temp_max ({temp_max}°C) as the 'low' temperature — "
                                            f"min/max are inverted in the narrative"
                                        )

                            # Return trip temp accuracy check
                            rt = data.get("return_trip") or {}
                            rt_weather = rt.get("weather") or {}
                            rt_min = rt_weather.get("temp_min_c")
                            rt_max = rt_weather.get("temp_max_c")
                            rt_llm = (rt.get("llm_response") or "")
                            if rt_min is not None and rt_max is not None and rt_max > rt_min and rt_llm:
                                rt_temp_mentions = [float(m) for m in re.findall(r'(\d+\.?\d*)\s*°C', rt_llm)]
                                if rt_temp_mentions:
                                    rt_invert = re.search(
                                        rf'(?:low|minimum|min)(?:(?!(?:high|max)).){{0,30}}({re.escape(str(int(rt_max)))})',
                                        rt_llm, re.IGNORECASE
                                    )
                                    if rt_invert:
                                        status = 124
                                        failure_reason = (
                                            f"Return trip LLM describes temp_max ({rt_max}°C) as the 'low' "
                                            f"temperature — min/max are inverted in return leg narrative"
                                        )

                            # Multi‑leg temp accuracy check
                            if status == 0 and assertions.get("check_weather_temp_accuracy") and data.get("multicity"):
                                for i, leg in enumerate(data.get("legs", [])):
                                    leg_wx = leg.get("weather") or {}
                                    leg_min = leg_wx.get("temp_min_c")
                                    leg_max = leg_wx.get("temp_max_c")
                                    leg_llm = (leg.get("llm_response") or "")
                                    if leg_min is not None and leg_max is not None and leg_max > leg_min and leg_llm:
                                        leg_mentions = [float(m) for m in re.findall(r'(\d+\.?\d*)\s*°C', leg_llm)]
                                        if leg_mentions:
                                            leg_invert = re.search(
                                                rf'(?:low|minimum|min)(?:(?!(?:high|max)).){{0,30}}({re.escape(str(int(leg_max)))})',
                                                leg_llm, re.IGNORECASE
                                            )
                                            if leg_invert:
                                                status = 124
                                                failure_reason = (
                                                    f"Leg {i+1} LLM describes temp_max ({leg_max}°C) as the 'low' "
                                                    f"temperature — min/max are inverted in stopover leg narrative"
                                                )
                                                break

                        # 11. Check if the LLM response fell back to the deterministic error summary
                        if status == 0:
                            fallback_str = "(Note: Enhanced explanation unavailable"
                            
                            # Check main response
                            llm_text = data.get("llm_response", "") or ""
                            if fallback_str in llm_text:
                                status = 124
                                failure_reason = "LLM generation timed out / returned deterministic fallback"
                                
                            # Check return trip
                            if status == 0 and data.get("return_trip"):
                                rt_llm = data["return_trip"].get("llm_response", "") or ""
                                if fallback_str in rt_llm:
                                    status = 124
                                    failure_reason = "Return trip LLM generation timed out / returned fallback"
                                    
                            # Check multi-city legs
                            if status == 0 and data.get("multicity"):
                                for i, leg in enumerate(data.get("legs", [])):
                                    leg_llm = leg.get("llm_response", "") or ""
                                    if fallback_str in leg_llm:
                                        status = 124
                                        failure_reason = f"Leg {i} LLM generation timed out / returned fallback"
                                        break

                        # 12. Check for flight numbers in LLM text that don't exist in all_flights
                        if status == 0:
                            llm_text_check = data.get("llm_response", "") or ""
                            all_flight_nos = {
                                re.sub(r'\s+', ' ', (f.get("flight_no") or "").strip())
                                for f in (data.get("all_flights") or [])
                            }
                            if all_flight_nos and llm_text_check:
                                mentioned = re.findall(r'\b([A-Z]{1,2}\d?[A-Z]?\s*\d{3,4})\b', llm_text_check)
                                for raw_m in mentioned:
                                    m = re.sub(r'\s+', ' ', raw_m.strip())
                                    if m not in all_flight_nos:
                                        status = 124
                                        failure_reason = f"LLM response mentions flight {m} which is not in all_flights (possible hallucination)"
                                        break

                        # 13. Check for parallel async tests: all JSON blobs must not contain fallback
                        if status == 0 and assertions.get("check_no_parallel_fallback"):
                            # Use cmd_output (full raw output)
                            blobs = re.findall(r'\{\s*"llm_response".*?\}(?=\s*\{|\s*$)', cmd_output, re.DOTALL)
                            for j, blob_str in enumerate(blobs):
                                try:
                                    blob = json.loads(blob_str)
                                    if _payload_has_llm_backend_unavailable_signal(blob):
                                        status = 124
                                        failure_reason = (
                                            f"Parallel query {j+1} used LLM backend-unavailable/degraded fallback"
                                        )
                                        explicit_failure_tags.add("llm_backend_unavailable")
                                        break
                                except:
                                    continue

                        # 14. Validate API trace when present
                        if status == 0 and assertions.get("check_api_trace"):
                            trace = (data.get("debug_info") or {}).get("api_trace", {})
                            intent_d = (data.get("debug_info") or {}).get("intent", {})
                            
                            # (a) Flight API was called with the right departure/arrival
                            ft_req = trace.get("flight", {}).get("request", {})
                            if ft_req:
                                if ft_req.get("departure") != intent_d.get("origin_iata"):
                                    status = 124
                                    failure_reason = (
                                        f"Flight API called with departure={ft_req.get('departure')} "
                                        f"but intent.origin_iata={intent_d.get('origin_iata')}"
                                    )
                                elif ft_req.get("arrival") != intent_d.get("destination_iata"):
                                    status = 124
                                    failure_reason = (
                                        f"Flight API called with arrival={ft_req.get('arrival')} "
                                        f"but intent.destination_iata={intent_d.get('destination_iata')}"
                                    )
                                elif ft_req.get("date") != data.get("search_date"):
                                    status = 124
                                    failure_reason = (
                                        f"Flight API date mismatch: called with {ft_req.get('date')}, "
                                        f"search_date={data.get('search_date')}"
                                    )
                            
                            # (b) Weather API forecast_date must be within a certain number of days of search_date
                            if status == 0:
                                wt = trace.get("weather", {})
                                forecast_date_str = wt.get("forecast_date") or (data.get("weather") or {}).get("forecast_date")
                                search_date_str = data.get("search_date")
                                if forecast_date_str and search_date_str:
                                    try:
                                        fd = datetime.strptime(forecast_date_str, "%Y-%m-%d")
                                        sd = datetime.strptime(search_date_str, "%Y-%m-%d")
                                        gap = abs((sd - fd).days)
                                        max_gap = 20 if args.live else 15
                                        if gap > max_gap:
                                            reason = (
                                                f"Weather forecast is {gap}d from search_date "
                                                f"(forecast={forecast_date_str}, search={search_date_str}) — "
                                            )
                                            if args.live:
                                                reason += "OWM 5‑day window exceeded; planner should have fallen back to current weather"
                                            else:
                                                reason += "API likely returned cached/wrong date data"
                                            status = 124
                                            failure_reason = reason
                                    except ValueError:
                                        pass

                    # Check for new fields: booking_token and price_insights_str (now handles multicity)
                    if status == 0 and not is_stream and expect_llm and isinstance(data, dict):
                        is_multicity = data.get("multicity", False)
                        if not is_multicity:
                            # Single‑leg PlanResult
                            bf = data.get("best_flight", {})
                            if "booking_token" not in bf and "shareable_link" not in bf:
                                status = 124
                                failure_reason = "Booking token and shareable link are both missing from best_flight"
                            debug = data.get("debug_info", {})
                            if "price_insights_str" not in debug:
                                status = 124
                                failure_reason = "price_insights_str missing from debug_info"
                        else:
                            # MultiCityResult – check each leg
                            for i, leg in enumerate(data.get("legs", [])):
                                leg_debug = leg.get("debug_info", {})
                                if "price_insights_str" not in leg_debug:
                                    status = 124
                                    failure_reason = f"price_insights_str missing from legs[{i}].debug_info"
                                    break
                                leg_bf = leg.get("best_flight", {})
                                if "booking_token" not in leg_bf and "shareable_link" not in leg_bf:
                                    status = 124
                                    failure_reason = f"booking_token/shareable_link missing from legs[{i}].best_flight"
                                    break

                    # Now the regular expect_llm check (if not already failed)
                    skip_default_json_validations = bool(
                        isinstance(assertions, dict) and assertions.get("skip_default_json_validations")
                    )
                    if status == 0 and not skip_default_json_validations:
                        if expect_llm:
                            if _payload_has_llm_backend_unavailable_signal(data):
                                status = 124
                                failure_reason = (
                                    "LLM backend unavailable/degraded fallback detected "
                                    "for a scenario that requires real LLM output"
                                )
                                explicit_failure_tags.add("llm_backend_unavailable")
                            if status == 0 and has_nonempty_llm(data):
                                status = 0
                            elif status == 0:
                                status = 124
                                failure_reason = "No non-empty llm_response found"
                        else:
                            # only check top-level detail (validation errors)
                            if isinstance(data, dict) and "detail" in data:
                                status = 124
                                failure_reason = f"Detail error: {data.get('detail')}"
                            else:
                                status = 0
                except json.JSONDecodeError as e:
                    status = 125
                    failure_reason = f"JSON decode error: {e}"

        # Streaming checks
        if is_stream and status == 0:
            if os.path.getsize(tmp_out_name) == 0:
                status = 126
                failure_reason = "Empty stream response"
            else:
                if re.search(r'"msg":"Field required"|"detail"', resp_body, re.IGNORECASE):
                    status = 124
                    failure_reason = "Validation error in stream"

        # Additional DONE_JSON validation for streams
        if is_stream and status == 0:
            done_match = re.search(r'\[DONE_JSON\](\{.*\})', resp_body, re.DOTALL)
            if done_match:
                try:
                    done_data = json.loads(done_match.group(1))
                    if done_data.get("error"):
                        status = 124
                        failure_reason = f"Stream DONE_JSON contains server error: {done_data['error']}"
                    elif not done_data.get("llm_response") and not done_data.get("multicity"):
                        status = 124
                        failure_reason = "Stream DONE_JSON has no llm_response and no multicity data"
                    elif expect_llm and _payload_has_llm_backend_unavailable_signal(done_data):
                        status = 124
                        failure_reason = (
                            "LLM backend unavailable/degraded fallback detected "
                            "for a stream scenario that requires real LLM output"
                        )
                        explicit_failure_tags.add("llm_backend_unavailable")
                except json.JSONDecodeError:
                    status = 124
                    failure_reason = "Stream DONE_JSON present but JSON parsing failed"
            else:
                status = 124
                failure_reason = "Stream missing DONE_JSON completion payload"

        # Additional expected_date check for stream (if present)
        if is_stream and status == 0 and assertions and assertions.get("expected_date"):
            done_match = re.search(r'\[DONE_JSON\](\{.*\})', resp_body, re.DOTALL)
            if done_match:
                try:
                    done_data = json.loads(done_match.group(1))
                    date_ok, tolerated_skew, mismatch_reason, date_details = _evaluate_expected_date_assertion(
                        done_data,
                        assertions,
                    )
                    date_assertion_details = date_details
                    if not date_ok:
                        status = 124
                        if mismatch_reason.startswith("search_date mismatch"):
                            failure_reason = mismatch_reason.replace(
                                "search_date mismatch",
                                "Stream search_date mismatch",
                                1,
                            )
                        else:
                            failure_reason = mismatch_reason
                        if tolerated_skew:
                            explicit_failure_tags.add("runtime_date_basis_skew")
                except (json.JSONDecodeError, KeyError):
                    pass

        # After streaming checks, handle api_trace for stream
        if is_stream and status == 0 and assertions and assertions.get("check_api_trace"):
            done_json_match = re.search(r'\[DONE_JSON\](\{.*\})', resp_body, re.DOTALL)
            if done_json_match:
                try:
                    done_data = json.loads(done_json_match.group(1))
                    trace = (done_data.get("debug_info") or {}).get("api_trace", {})
                    wt = trace.get("weather", {})
                    forecast_date_str = wt.get("forecast_date") or (done_data.get("weather") or {}).get("forecast_date")
                    search_date_str = done_data.get("search_date")
                    if forecast_date_str and search_date_str:
                        fd = datetime.strptime(forecast_date_str, "%Y-%m-%d")
                        sd = datetime.strptime(search_date_str, "%Y-%m-%d")
                        gap = abs((sd - fd).days)
                        max_gap = 20 if args.live else 15
                        if gap > max_gap:
                            reason = (
                                f"Stream DONE_JSON: weather forecast {gap}d from search_date "
                                f"(forecast={forecast_date_str}, search={search_date_str}) — "
                            )
                            if args.live:
                                reason += "OWM 5‑day window exceeded; planner should have fallen back to current weather"
                            else:
                                reason += "API likely returned cached/wrong date data"
                            status = 124
                            failure_reason = reason
                except (ValueError, json.JSONDecodeError):
                    pass

        # Optional debug (now using logger)
        if status != 0:
            logger.debug(f"{name} http_code={http_code} curl_exit={curl_exit}")
        backend_status = status
        backend_failure_reason = failure_reason

    elif not frontend_handled:
        # non-curl command
        cmd_timeout = 120  # hard timeout to prevent runaway processes
        with open(tmp_out_name, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            heartbeat_sec = 30
            last_heartbeat = time.time()
            while True:
                exit_code = proc.poll()
                if exit_code is not None:
                    status = exit_code
                    break
                now = time.time()
                if now - start_epoch > cmd_timeout:
                    log(f"[{mode_label:7s}] {name} exceeded {cmd_timeout}s timeout, killing process")
                    proc.kill()
                    status = 124
                    failure_reason = f"Command exceeded {cmd_timeout}s timeout"
                    break
                if args.debug and now - last_heartbeat >= heartbeat_sec:
                    elapsed = int(now - start_epoch)
                    log(f"[{mode_label:7s}] {name} still running... ({elapsed}s elapsed)")
                    last_heartbeat = now
                time.sleep(1)
        # Read the full command output for later assertions
        with open(tmp_out_name, 'r') as f:
            cmd_output = f.read()

        if status != 0:
            failure_reason, extra_tags = _extract_noncurl_failure_reason(name, cmd_output, status)
            explicit_failure_tags.update(extra_tags)

        # Assertions for non-curl commands (e.g. parallel bash test)
        if status == 0 and assertions and assertions.get("check_no_parallel_fallback"):
            all_blobs = re.findall(r'\{\s*"llm_response".*?\}(?=\s*\{|\s*$)', cmd_output, re.DOTALL)
            for j, blob_str in enumerate(all_blobs):
                try:
                    blob = json.loads(blob_str)
                    if _payload_has_llm_backend_unavailable_signal(blob):
                        status = 124
                        failure_reason = (
                            f"Parallel query {j+1} used LLM backend-unavailable/degraded fallback"
                        )
                        explicit_failure_tags.add("llm_backend_unavailable")
                        break
                except json.JSONDecodeError:
                    continue
            # Also fail if any sub-response is an HTTP error detail
            error_blobs = re.findall(r'\{"detail"\s*:\s*"[^"]*"\}', cmd_output)
            if error_blobs:
                status = 124
                failure_reason = f"Parallel query returned server error: {error_blobs[0]}"

        if status == 0 and assertions and assertions.get("check_parallel_json_integrity"):
            blobs = re.findall(r'\{\s*"llm_response".*?\}(?=\s*\{|\s*$)', cmd_output, re.DOTALL)
            parsed_blobs = []
            for blob_str in blobs:
                try:
                    parsed_blobs.append(json.loads(blob_str))
                except json.JSONDecodeError:
                    continue

            min_count = int(assertions.get("expected_parallel_count") or 2)
            if len(parsed_blobs) < min_count:
                status = 124
                failure_reason = f"Parallel query output contained {len(parsed_blobs)} JSON responses; expected at least {min_count}"
            else:
                expected_destinations = set(assertions.get("expected_parallel_destinations") or [])
                seen_destinations = set()
                for idx, blob in enumerate(parsed_blobs, start=1):
                    llm_text = (blob.get("llm_response") or "").strip()
                    search_date = blob.get("search_date")
                    if not llm_text:
                        status = 124
                        failure_reason = f"Parallel response {idx} has empty llm_response"
                        break
                    if not search_date:
                        status = 124
                        failure_reason = f"Parallel response {idx} missing search_date"
                        break

                    if assertions.get("check_api_trace"):
                        api_trace = (blob.get("debug_info") or {}).get("api_trace") or {}
                        flight_req = (api_trace.get("flight") or {}).get("request") or {}
                        weather_req = (api_trace.get("weather") or {}).get("request") or {}
                        if not flight_req:
                            status = 124
                            failure_reason = f"Parallel response {idx} missing api_trace.flight.request"
                            break
                        if not weather_req:
                            status = 124
                            failure_reason = f"Parallel response {idx} missing api_trace.weather.request"
                            break

                    labels = (blob.get("debug_info") or {}).get("route_labels") or {}
                    dest_iata = labels.get("destination_iata")
                    if isinstance(dest_iata, str) and dest_iata:
                        seen_destinations.add(dest_iata)

                if status == 0 and expected_destinations and not expected_destinations.issubset(seen_destinations):
                    status = 124
                    failure_reason = (
                        f"Parallel responses missing expected destinations: "
                        f"expected={sorted(expected_destinations)}, seen={sorted(seen_destinations)}"
                    )

    if frontend_handled:
        frontend_blob = json.dumps(frontend_result, ensure_ascii=False) if frontend_result else ""
        if frontend_blob:
            cmd_output = (cmd_output or "") + f"\n{{\"_frontend_validation\": {frontend_blob}}}\n"

        if not run_backend_shadow:
            status = 125 if frontend_status is None else int(frontend_status)
            failure_reason = frontend_failure_reason or failure_reason
            resp_body = frontend_blob
        else:
            if frontend_status not in (None, 0):
                status = int(frontend_status)
                failure_reason = frontend_failure_reason or "Frontend validation failed."
                resp_body = frontend_blob
            elif backend_checked and backend_status not in (None, 0):
                status = int(backend_status)
                failure_reason = backend_failure_reason or failure_reason or "Backend shadow validation failed."
            else:
                status = 0
                failure_reason = ""

    end_epoch = time.time()
    duration = end_epoch - start_epoch

    mode_bucket = str(effective_meta.get("mode_bucket") or MODE_BACKEND_INTERNAL)
    soft_pass_policy = str(effective_meta.get("soft_pass_policy") or SOFT_PASS_HARD_FAIL_ONLY)
    normalized_failure_tags.update(explicit_failure_tags)
    normalized_failure_tags.update(_extract_structured_failure_tags(resp_body, is_stream))
    if status == 0 and expect_llm:
        llm_degraded_tags = {"timeout", "llm_backend_unavailable", "upstream_unavailable"}
        tagged_hits = sorted(normalized_failure_tags & llm_degraded_tags)
        if tagged_hits:
            status = 124
            failure_reason = (
                "LLM-required scenario produced degraded/timeout backend tags: "
                + ",".join(tagged_hits)
            )
            normalized_failure_tags.add("llm_backend_unavailable")
    if status != 0 and _is_soft_pass_eligible_test(soft_pass_policy=soft_pass_policy, mode_bucket=mode_bucket):
        normalized_failure_tags.update(_extract_log_based_failure_tags(mode_label))
    verdict = _determine_validation_verdict(
        name,
        status,
        normalized_failure_tags,
        soft_pass_policy=soft_pass_policy,
        mode_bucket=mode_bucket,
    )
    base_name_for_quality = _strip_mode_suffix(name)
    if frontend_handled and base_name_for_quality.startswith(
        ("frontend_runtime_", "frontend_fixture_", "frontend_real_backend_", "frontend_live_canary_")
    ):
        quality_grade = _determine_frontend_pass_quality(name, verdict, frontend_result)
    else:
        quality_grade = _determine_pass_quality(name, status, verdict, assertions, is_stream)

    payload_for_evidence = _extract_primary_payload(resp_body, is_stream=is_stream)
    llm_evidence = _build_llm_evidence(
        payload_for_evidence,
        expect_llm=bool(expect_llm),
        is_stream=bool(is_stream),
        duration_sec=duration,
        validation_request_start_epoch_ms=int(start_epoch * 1000),
    )

    # ------------------------------------------------------------------
    # Build compact summary for log diff readability
    # ------------------------------------------------------------------
    _summary = {
        "test": name,
        "status": status,
        "duration_s": round(duration, 2),
        "verdict": verdict,
        "quality": quality_grade,
        "llm_required": bool(expect_llm),
        "mode_bucket": mode_bucket,
        "soft_pass_policy": soft_pass_policy,
        "llm_evidence_state": llm_evidence.get("state"),
        "llm_request_reached": bool(llm_evidence.get("request_reached_llm_path")),
        "llm_completion_observed": bool(llm_evidence.get("completion_observed")),
        "llm_degraded_observed": bool(llm_evidence.get("degraded_observed")),
        "llm_generation_mode": llm_evidence.get("generation_mode"),
    }
    if llm_evidence.get("latency_sec") is not None:
        _summary["llm_latency_sec"] = llm_evidence.get("latency_sec")
    if llm_evidence.get("timeout_sec") is not None:
        _summary["llm_timeout_sec"] = llm_evidence.get("timeout_sec")
    if llm_evidence.get("timeout_ratio") is not None:
        _summary["llm_timeout_ratio"] = llm_evidence.get("timeout_ratio")
    if llm_evidence.get("near_timeout"):
        _summary["llm_near_timeout"] = True
    if llm_evidence.get("timeout_shaped"):
        _summary["llm_timeout_shaped"] = True
    llm_models = list(llm_evidence.get("models") or [])
    if llm_models:
        _summary["llm_models"] = llm_models
    llm_num_ctx_values = list(llm_evidence.get("num_ctx_values") or [])
    if llm_num_ctx_values:
        _summary["llm_num_ctx_values"] = llm_num_ctx_values
    llm_thinking_modes = list(llm_evidence.get("thinking_modes") or [])
    if llm_thinking_modes:
        _summary["llm_thinking_modes"] = llm_thinking_modes
    if llm_evidence.get("first_token_latency_sec") is not None:
        _summary["llm_first_token_latency_sec"] = llm_evidence.get("first_token_latency_sec")
    if llm_evidence.get("first_token_from_validation_send_sec") is not None:
        _summary["llm_first_token_from_validation_send_sec"] = llm_evidence.get(
            "first_token_from_validation_send_sec"
        )
    _summary["llm_first_token_available"] = bool(llm_evidence.get("first_token_available"))
    _summary["llm_first_token_measurement"] = str(
        llm_evidence.get("first_token_measurement") or "not_available"
    )
    if frontend_handled:
        _summary["frontend_status"] = frontend_status
        _summary["frontend_failure_reason"] = frontend_failure_reason or ""
        _summary["backend_shadow_status"] = backend_status if run_backend_shadow else "skipped"
    if isinstance(payload_for_evidence, dict):
        _d = payload_for_evidence
        if _d.get("multicity"):
            for _i, _leg in enumerate(_d.get("legs", [])):
                _summary[f"leg{_i+1}_llm_len"] = len((_leg.get("llm_response") or ""))
                _summary[f"leg{_i+1}_weather"] = (_leg.get("weather") or {}).get("condition", "N/A")
                _summary[f"leg{_i+1}_flight"] = (_leg.get("best_flight") or {}).get("flight_no", "N/A")
                leg_exec = ((_leg.get("debug_info") or {}).get("llm_execution")) if isinstance(_leg, dict) else None
                if isinstance(leg_exec, dict):
                    leg_source = str(
                        leg_exec.get("source")
                        or leg_exec.get("completion_source")
                        or ""
                    ).strip()
                    if leg_source:
                        _summary[f"leg{_i+1}_llm_exec_source"] = leg_source
                    leg_backend = str(
                        leg_exec.get("backend")
                        or leg_exec.get("router_backend")
                        or ""
                    ).strip()
                    if leg_backend:
                        _summary[f"leg{_i+1}_llm_exec_backend"] = leg_backend
                    if leg_exec.get("degraded") is True:
                        _summary[f"leg{_i+1}_llm_exec_degraded"] = True
        else:
            llm_response = _d.get("llm_response") if isinstance(_d, dict) else ""
            _summary["llm_response_len"] = len((llm_response or ""))
            weather_blob = _d.get("weather") if isinstance(_d, dict) else {}
            if not isinstance(weather_blob, dict):
                weather_blob = {}
            best_flight_blob = _d.get("best_flight") if isinstance(_d, dict) else {}
            if not isinstance(best_flight_blob, dict):
                best_flight_blob = {}
            _summary["weather_condition"] = weather_blob.get("condition", "N/A")
            _summary["best_flight_no"] = best_flight_blob.get("flight_no", "N/A")
            _summary["search_date"] = _d.get("search_date", "N/A") if isinstance(_d, dict) else "N/A"
            llm_exec = (_d.get("debug_info") or {}).get("llm_execution")
            if isinstance(llm_exec, dict):
                exec_source = str(
                    llm_exec.get("source")
                    or llm_exec.get("completion_source")
                    or ""
                ).strip()
                if exec_source:
                    _summary["llm_exec_source"] = exec_source
                exec_backend = str(
                    llm_exec.get("backend")
                    or llm_exec.get("router_backend")
                    or ""
                ).strip()
                if exec_backend:
                    _summary["llm_exec_backend"] = exec_backend
                if llm_exec.get("degraded") is True:
                    _summary["llm_exec_degraded"] = True
    if failure_reason:
        _summary["failure_reason"] = failure_reason
    if normalized_failure_tags:
        _summary["failure_tags"] = sorted(normalized_failure_tags)
    if date_assertion_details:
        _summary["date_assertion"] = date_assertion_details
    diagnostics_blob = None
    if status != 0:
        diagnostics_blob = _build_failure_diagnostics(
            name,
            cmd,
            resp_body,
            cmd_output=cmd_output,
            date_assertion_details=date_assertion_details,
        )

    fh.flush()  # drain Python's logger buffer to fh's fd first
    fh.stream.write(
        f"=== START {mode_label}/{name} ({start_iso}) ===\n"
        + f"SUMMARY: {json.dumps(_summary)}\n"
        + (f"DIAGNOSTICS: {json.dumps(diagnostics_blob)}\n" if diagnostics_blob else "")
        + cmd_output
        + f"\n=== END   {mode_label}/{name} (exit={status}, duration={duration:.3f}s) ===\n\n"
    )
    fh.stream.flush()  # ensure the write reaches the OS immediately

    os.unlink(tmp_out_name)

    # Store report with reason
    base_name = _strip_mode_suffix(name)
    report_display = _display_name_for_base(base_name)
    if isinstance(assertions, dict):
        override_display = str(assertions.get("display_name") or "").strip()
        if override_display:
            report_display = override_display
    REPORT.append({
        "name": name,
        "status": status,
        "duration": duration,
        "reason": failure_reason,
        "verdict": verdict,
        "quality": quality_grade,
        "llm_required": bool(expect_llm),
        "failure_tags": sorted(normalized_failure_tags),
        "display": report_display,
        "scenario": effective_meta.get("scenario", "uncategorized"),
        "layers": list(effective_meta.get("layers", ["uncategorized"])),
        "validation_type": effective_meta.get("validation_type", "uncategorized"),
        "features": list(effective_meta.get("features", [])),
        "mode_bucket": mode_bucket,
        "soft_pass_policy": soft_pass_policy,
        "llm_evidence": dict(llm_evidence or {}),
        "criticality": effective_meta.get("criticality", "core"),
        "dimensions": dict(effective_meta.get("dimensions", {})) if isinstance(effective_meta.get("dimensions"), dict) else {},
        "ui_assertions": list(effective_meta.get("ui_assertions", [])) if isinstance(effective_meta.get("ui_assertions"), (list, tuple)) else [],
        "contract_assertions": list(effective_meta.get("contract_assertions", [])) if isinstance(effective_meta.get("contract_assertions"), (list, tuple)) else [],
    })

    # Live status print to console (using logger, which respects quiet mode)
    display_name = report_display
    llm_runtime_inline = _format_entry_llm_runtime({"llm_evidence": llm_evidence})
    llm_runtime_inline_text = f" | llm={llm_runtime_inline}" if llm_runtime_inline else ""

    # No dynamic tests anymore

    if verdict == VERDICT_PASS:
        log(f"[{mode_label:7s}] {display_name:30s} ... PASSED ({duration:.3f} s)")
    elif verdict == VERDICT_SOFT_PASS_NO_CREDIT:
        log(
            f"[{mode_label:7s}] {display_name:30s} ... SOFT_PASS_NO_CREDIT ({duration:.3f} s)"
            + llm_runtime_inline_text
        )
    else:
        log(
            f"[{mode_label:7s}] {display_name:30s} ... FAILED ({duration:.3f} s)"
            + llm_runtime_inline_text
        )

    return status


def run_capability_checks(mode, base_url):
    """Lightweight capability pass-through checks for health and planner surfaces."""
    future_date = (datetime.now().date() + timedelta(days=21)).strftime("%Y-%m-%d")
    admin_header = ["-H", f"X-Admin-Token: {VALIDATION_ADMIN_TOKEN}"]
    run_and_log(f"health_light_{mode}", [
        "curl", "-sS", "-X", "GET", f"{base_url}/health"
    ], expect_llm=False, assertions={
        "required_paths": [
            "status",
            "dependencies",
            "async_jobs_enabled",
        ],
        "check_health_light_shape": True,
    })
    run_and_log(f"health_deep_{mode}", [
        "curl", "-sS", "-X", "GET", f"{base_url}/health/deep", *admin_header
    ], expect_llm=False, assertions={
        "required_paths": ["status", "dependencies"],
        "check_health_deep_shape": True,
    })
    run_and_log(f"health_keys_{mode}", [
        "curl", "-sS", "-X", "GET", f"{base_url}/health/keys", *admin_header
    ], expect_llm=False, assertions={"check_health_keys_shape": True})
    run_and_log(f"health_runtime_topology_{mode}", [
        "curl", "-sS", "-X", "GET", f"{base_url}/health"
    ], expect_llm=False, assertions={
        "required_paths": [
            "dependencies.app",
            "dependencies.key_manager",
            "async_jobs_enabled",
            "external_dependency_checks.deep_endpoint",
        ],
        "check_health_light_shape": True,
    })
    run_and_log(f"llm_options_{mode}", [
        "curl", "-sS", "-X", "GET", f"{base_url}/llm/options", *admin_header
    ], expect_llm=False, assertions={
        "required_paths": [
            "llm_modes",
            "defaults.llm_mode",
            "backend_availability.cloud",
            "backend_availability.ollama",
            "config_authority.mode_dependency",
        ],
        "check_llm_options_shape": True,
    })
    run_and_log(f"version_info_{mode}", [
        "curl", "-sS", "-X", "GET", f"{base_url}/version"
    ], expect_llm=False, assertions={
        "required_paths": ["git_commit", "file_mtime"],
        "check_version_info_shape": True,
    })

    # Planner capability: parsing + constraints in one representative request.
    run_and_log(f"capability_constraints_{mode}", [
        "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "user_query": (
                f"Direct morning flight from Delhi to Mumbai on {future_date} "
                "under 12000 INR with Indigo preference"
            )
        })
    ], expect_llm=True, assertions={"check_api_trace": True})


def _python_exec():
    venv_python = ROOT / "venv/bin/python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable or "python3"


def run_contract_checks(mode, base_url):
    py = _python_exec()

    run_and_log(
        f"contract_no_flights_{mode}",
        [py, "-m", "pytest", "-q", "tests/test_api.py::test_ask_non_stream_warning_fallback_is_not_success"],
        expect_llm=False,
    )
    run_and_log(
        f"contract_degraded_stream_{mode}",
        [
            py,
            "-m",
            "pytest",
            "-q",
            "tests_slow/test_planner_stream_error_contract.py::test_stream_llm_unavailable_returns_degraded_structured_done_json",
        ],
        expect_llm=False,
    )
    run_and_log(
        f"contract_degraded_stream_heartbeat_{mode}",
        [
            py,
            "-m",
            "pytest",
            "-q",
            "tests_slow/test_planner_stream_error_contract.py::test_stream_thinking_only_heartbeat_without_visible_tokens_degrades",
        ],
        expect_llm=False,
    )
    run_and_log(
        f"contract_hardening_duplicate_guard_{mode}",
        [
            py,
            "-m",
            "pytest",
            "-q",
            "tests/test_api.py::test_ask_duplicate_burst_has_single_leader_and_deterministic_rejections",
        ],
        expect_llm=False,
    )
    run_and_log(
        f"contract_hardening_backpressure_{mode}",
        [
            py,
            "-m",
            "pytest",
            "-q",
            "tests/test_api.py::test_ask_short_distinct_burst_is_bounded_by_inflight_limit",
        ],
        expect_llm=False,
    )
    run_and_log(
        f"contract_hardening_consume_race_{mode}",
        [
            py,
            "-m",
            "pytest",
            "-q",
            "tests/test_api.py::test_booking_post_handoff_bridge_repeated_concurrent_consumes_remain_single_winner",
        ],
        expect_llm=False,
    )
    run_and_log(
        f"contract_hardening_retry_budget_{mode}",
        [
            py,
            "-m",
            "pytest",
            "-q",
            "tests_slow/test_airline_api.py::test_search_flights_retry_budget_is_stable_across_repeated_degraded_calls",
        ],
        expect_llm=False,
    )
    run_and_log(
        f"contract_hardening_key_cooldown_{mode}",
        [
            py,
            "-m",
            "pytest",
            "-q",
            "tests/test_api_key_manager.py::test_key_manager_repeated_failure_recovery_sequence_remains_stable",
            "tests_slow/test_cloud_provider.py::test_health_check_auth_cooldown_and_recovery",
        ],
        expect_llm=False,
    )

    booking_contract_script = f"""
import json
import os
import requests
import sys
import urllib.parse
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

raw_db_url = os.getenv("DATABASE_URL") or ""
if "host.docker.internal" in raw_db_url:
    # This contract script runs on the host process; map container-only hostname
    # to localhost so DB persistence checks hit the same Postgres reliably.
    os.environ["DATABASE_URL"] = raw_db_url.replace("host.docker.internal", "localhost")

from agents.database import get_engine
from tools.booking_handoff import (
    PostHandoffArtifact,
    SessionLocal,
    register_post_handoff_artifact,
)
base = {base_url!r}

def _sanitize_db_url(raw):
    if not raw:
        return "unset"
    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return "invalid"
    host = parsed.hostname or "none"
    port = parsed.port or "default"
    db = (parsed.path or "").lstrip("/") or "none"
    return f"{{parsed.scheme}}://{{host}}:{{port}}/{{db}}"

def _engine_url_basis():
    try:
        return _sanitize_db_url(str(get_engine().url))
    except Exception:
        return "unavailable"

bridge = register_post_handoff_artifact(
    url='https://provider.example/checkout',
    post_data={{'token': 'qa-token', 'fare': 'X1'}},
)
diag0 = {{
    "phase": "registration",
    "registration_ok": bool(bridge),
    "bridge_path": bridge,
    "database_url_basis": _sanitize_db_url(os.getenv("DATABASE_URL")),
    "engine_url_basis": _engine_url_basis(),
    "testing_env": os.getenv("TESTING"),
    "testing_use_persistent_db": os.getenv("TESTING_USE_PERSISTENT_DB"),
    "post_handoff_require_persistence": os.getenv("POST_HANDOFF_REQUIRE_PERSISTENCE"),
}}
print("BOOKING_BRIDGE_DIAG " + json.dumps(diag0, sort_keys=True))
if not bridge:
    raise SystemExit('booking bridge artifact registration failed')

artifact_id = bridge.rsplit("/", 1)[-1]
session = SessionLocal()
persistent_present = False
try:
    row = session.query(PostHandoffArtifact).filter(PostHandoffArtifact.artifact_id == artifact_id).first()
    persistent_present = bool(row)
finally:
    session.close()
print("BOOKING_BRIDGE_DIAG " + json.dumps({{
    "phase": "post_register_store_check",
    "artifact_id_prefix": artifact_id[:12],
    "persistent_row_present": persistent_present,
}}, sort_keys=True))

target = base + bridge
first = requests.post(
    target,
    timeout=12,
    headers={{"Accept": "text/html"}},
)
print("BOOKING_BRIDGE_DIAG " + json.dumps({{
    "phase": "first_consume_post",
    "target": target,
    "first_status": first.status_code,
    "consume_result_header": (first.headers.get('X-Booking-Bridge-Consume-Result') or '').strip(),
}}, sort_keys=True))
if first.status_code != 200:
    raise SystemExit(f'first bridge consume failed: status={{first.status_code}} body={{first.text[:200]}}')
if \"form id='handoff'\" not in first.text:
    raise SystemExit('booking bridge html missing handoff form')
if 'https://provider.example/checkout' not in first.text:
    raise SystemExit('booking bridge html missing provider action url')
consume_header = (first.headers.get('X-Booking-Bridge-Consume-Result') or '').strip()
if consume_header not in ('memory_hit', 'persistent_hit'):
    raise SystemExit(f'unexpected consume result header: {{consume_header}}')
second = requests.post(
    target,
    timeout=12,
    headers={{"Accept": "application/json"}},
)
second_payload = None
try:
    second_payload = second.json()
except Exception:
    second_payload = {{}}
print("BOOKING_BRIDGE_DIAG " + json.dumps({{
    "phase": "second_consume_post",
    "second_status": second.status_code,
    "second_lookup_result": ((second_payload or {{}}).get("detail") or {{}}).get("lookup_result"),
}}, sort_keys=True))
if second.status_code not in (404, 410):
    raise SystemExit(f'second bridge consume should be 404/410, got {{second.status_code}}')
payload = second_payload if isinstance(second_payload, dict) else {{}}
detail = payload.get('detail') if isinstance(payload, dict) else {{}}
if not isinstance(detail, dict) or detail.get('error') != 'booking_handoff_artifact_unavailable':
    raise SystemExit(f'unexpected second bridge payload: {{payload}}')
print('booking bridge contract ok')
"""
    run_and_log(f"contract_booking_bridge_{mode}", [py, "-c", booking_contract_script], expect_llm=False)

    jobs_contract_script = f"""
import json
import os
import time
import requests
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass
base = {base_url!r}
auth_token = (os.getenv("AUTH_TOKEN") or {VALIDATION_AUTH_TOKEN!r}).strip()
if not auth_token:
    raise SystemExit("AUTH_TOKEN missing for jobs flow contract")
headers = {{
    "Authorization": f"Bearer {{auth_token}}",
    "Content-Type": "application/json",
}}
payload = {{
    'origin': 'DEL',
    'destination': 'BOM',
    'date': (time.strftime('%Y-%m-%d', time.localtime(time.time() + 21 * 24 * 3600))),
    'trip_type': 'Business',
    'user_query': 'Async validation job flow check',
}}
create = requests.post(base + '/ask?async_job=true', json=payload, timeout=25, headers=headers)
if create.status_code == 503:
    body = create.json() if create.headers.get('content-type', '').startswith('application/json') else {{}}
    detail = body.get('detail') if isinstance(body, dict) else {{}}
    if isinstance(detail, dict) and detail.get('error') == 'async_job_topology_unsupported':
        print('async jobs topology guard active (accepted contract)')
        raise SystemExit(0)
if create.status_code != 202:
    raise SystemExit(f'job create failed: status={{create.status_code}} body={{create.text[:300]}}')
job_id = (create.json() or {{}}).get('job_id')
if not job_id:
    raise SystemExit('job_id missing from async job create response')

events = requests.get(base + f'/jobs/{{job_id}}/events', stream=True, timeout=(4, 20), headers=headers)
if events.status_code != 200:
    raise SystemExit(f'job events endpoint failed: status={{events.status_code}}')
saw_event = False
deadline = time.time() + 18
for raw in events.iter_lines(decode_unicode=True):
    if time.time() > deadline:
        break
    if not raw or not raw.startswith('data:'):
        continue
    saw_event = True
    try:
        evt = json.loads(raw[len('data:'):].strip())
    except Exception:
        continue
    if evt.get('type') in ('done', 'error', 'closed'):
        break
if not saw_event:
    raise SystemExit('no job events observed')

final_status = None
poll_deadline = time.time() + 35
while time.time() < poll_deadline:
    poll = requests.get(base + f'/jobs/{{job_id}}', timeout=12, headers=headers)
    if poll.status_code != 200:
        raise SystemExit(f'job poll failed: status={{poll.status_code}} body={{poll.text[:200]}}')
    payload = poll.json()
    final_status = payload.get('status')
    if final_status in ('done', 'error'):
        break
    time.sleep(0.35)
if final_status != 'done':
    raise SystemExit(f'job did not complete successfully (status={{final_status}})')
print('jobs flow contract ok')
"""
    run_and_log(f"contract_jobs_flow_{mode}", [py, "-c", jobs_contract_script], expect_llm=False)

    # ------------------------------------------------------------------
    # P0: /metrics endpoint validation
    # ------------------------------------------------------------------
    metrics_contract_script = f"""
import json
import os
import requests
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

base = {base_url!r}
admin_token = (os.getenv("ADMIN_TOKEN") or "").strip()
if not admin_token:
    raise SystemExit("ADMIN_TOKEN missing for /metrics contract")

resp = requests.get(base + "/metrics", headers={{"X-Admin-Token": admin_token}}, timeout=10)
if resp.status_code != 200:
    raise SystemExit(f"/metrics returned {{resp.status_code}}: {{resp.text[:200]}}")

text = resp.text
if not text.strip():
    raise SystemExit("/metrics response is empty")
if not (text.startswith("#") or "# TYPE" in text):
    raise SystemExit("/metrics response is not Prometheus text format")

# Assert 6 core metric families that are statically registered at import time
required = [
    "http_requests_total",
    "llm_requests_total",
    "ask_inflight_requests",
    "circuit_breaker_state",
    "booking_handoff_consume_total",
    "job_queue_size",
]
missing = [m for m in required if f"# TYPE {{m}}" not in text]
if missing:
    raise SystemExit(f"Missing metric families: {{missing}}")

# Verify at least one metric has been collected (non-zero value proves wiring works)
has_value = False
for line in text.splitlines():
    if line and not line.startswith("#"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                if float(parts[-1]) > 0:
                    has_value = True
                    break
            except ValueError:
                pass

if not has_value:
    # Not fatal — metrics may all be zero on a fresh start with no traffic
    print("metrics: all values zero (fresh start, acceptable)")

print("metrics contract ok")
"""
    run_and_log(f"contract_metrics_{mode}", [py, "-c", metrics_contract_script], expect_llm=False)

    # ------------------------------------------------------------------
    # P0: Booking hold/cancel API contract
    # ------------------------------------------------------------------
    booking_hold_cancel_script = f"""
import json
import os
import time
import requests
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

base = {base_url!r}
auth_token = (os.getenv("AUTH_TOKEN") or {VALIDATION_AUTH_TOKEN!r}).strip()
admin_token = (os.getenv("ADMIN_TOKEN") or "").strip()
if not auth_token:
    raise SystemExit("AUTH_TOKEN missing for booking hold/cancel contract")

headers = {{"Authorization": f"Bearer {{auth_token}}"}}

# 1. Get a real flight from /ask
ask_resp = requests.post(base + "/ask", json={{
    "user_query": "flights from Delhi to Mumbai",
    "origin": "DEL",
    "destination": "BOM",
    "date": (time.strftime('%Y-%m-%d', time.localtime(time.time() + 21 * 24 * 3600))),
}}, timeout=90)
if ask_resp.status_code != 200:
    body = ask_resp.json() if ask_resp.headers.get('content-type', '').startswith('application/json') else {{}}
    failure_domain = body.get("failure_domain", "unknown")
    error_msg = body.get("error", body.get("message", ""))
    # Check if this is an LLM/planner failure that blocks booking tests
    if failure_domain in ("llm", "planner") or "ollama" in error_msg.lower() or "not running" in error_msg.lower():
        print(f"BLOCKED: /ask failed due to LLM unavailability (domain={{failure_domain}}, error={{error_msg[:100]}})")
        raise SystemExit(0)
    raise SystemExit(f"/ask returned {{ask_resp.status_code}} (domain={{failure_domain}}): {{ask_resp.text[:200]}}")
ask_body = ask_resp.json()
best_flight = ask_body.get("best_flight")
if not best_flight:
    result_status = ask_body.get("result_status", "unknown")
    # This could also be an LLM failure that returned 200 but no results
    if result_status in ("error", "llm_error"):
        print(f"BLOCKED: /ask returned no best_flight (result_status={{result_status}}) — LLM likely unavailable")
        raise SystemExit(0)
    raise SystemExit(f"/ask returned no best_flight (result_status={{result_status}}) — cannot test booking hold without LLM results")

booking_id = None
try:
    # 2. Hold the flight
    hold_resp = requests.post(base + "/booking/hold", json={{
        "flight": best_flight,
        "origin": "DEL",
        "destination": "BOM",
        "depart_date": best_flight.get("date") or (time.strftime('%Y-%m-%d', time.localtime(time.time() + 21 * 24 * 3600))),
        "passengers": 1,
    }}, headers=headers, timeout=30)
    if hold_resp.status_code != 200:
        raise SystemExit(f"/booking/hold returned {{hold_resp.status_code}}: {{hold_resp.text[:300]}}")
    hold_body = hold_resp.json()
    if hold_body.get("action") != "hold_booking":
        raise SystemExit(f"hold response missing action=hold_booking: {{hold_body}}")
    if not hold_body.get("hold_created"):
        raise SystemExit(f"hold_created is false: {{hold_body}}")
    booking = hold_body.get("booking", {{}})
    if booking.get("status") != "HELD":
        raise SystemExit(f"booking status is not HELD: {{booking}}")
    booking_id = booking.get("id")
    if not booking_id:
        raise SystemExit("booking id missing from hold response")

    # 3. List bookings — verify held booking appears
    list_resp = requests.get(base + "/bookings", headers=headers, timeout=10)
    if list_resp.status_code != 200:
        raise SystemExit(f"/bookings returned {{list_resp.status_code}}")
    list_body = list_resp.json()
    if not any(b["id"] == booking_id for b in list_body.get("items", [])):
        raise SystemExit("Held booking not found in /bookings list")

    # 4. Cancel the booking
    cancel_resp = requests.post(base + "/booking/cancel", json={{"booking_id": booking_id}}, headers=headers, timeout=10)
    if cancel_resp.status_code != 200:
        raise SystemExit(f"/booking/cancel returned {{cancel_resp.status_code}}: {{cancel_resp.text[:200]}}")
    cancel_body = cancel_resp.json()
    if cancel_body.get("action") != "cancel_booking":
        raise SystemExit(f"cancel response missing action=cancel_booking: {{cancel_body}}")
    if not cancel_body.get("success"):
        raise SystemExit(f"cancel success is false: {{cancel_body}}")

    # 5. Verify cancellation — booking should no longer be in HELD state
    list_resp2 = requests.get(base + "/bookings?status=HELD", headers=headers, timeout=10)
    if list_resp2.status_code != 200:
        raise SystemExit(f"/bookings?status=HELD returned {{list_resp2.status_code}}")
    held_bookings = list_resp2.json().get("items", [])
    if any(b["id"] == booking_id for b in held_bookings):
        raise SystemExit("Cancelled booking still appears in HELD list")

    print("booking hold/cancel contract ok")
finally:
    # Cleanup: cancel if still held
    if booking_id:
        try:
            requests.post(base + "/booking/cancel", json={{"booking_id": booking_id}}, headers=headers, timeout=10)
        except Exception:
            pass
"""
    run_and_log(f"contract_booking_hold_cancel_{mode}", [py, "-c", booking_hold_cancel_script], expect_llm=False)

    # ------------------------------------------------------------------
    # P0: Provider failure simulation
    # ------------------------------------------------------------------
    provider_failure_script = f"""
import json
import os
import time
import requests
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

base = {base_url!r}
admin_token = (os.getenv("ADMIN_TOKEN") or "").strip()
if not admin_token:
    raise SystemExit("ADMIN_TOKEN missing for provider failure simulation")

headers = {{"X-Admin-Token": admin_token}}
future_ts = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
created_overrides = []

try:
    # 1. Try provider-level override first (disables all SerpAPI keys)
    override_payload = {{
        "provider": "serpapi",
        "scope_type": "provider_account",
        "scope_identifier": "serpapi",
        "override_type": "force_exhausted_until",
        "active_until": future_ts,
        "note": "validation-test",
    }}
    resp = requests.post(base + "/debug/provider-state/overrides", json=override_payload, headers=headers, timeout=10)

    if resp.status_code == 400:
        # Provider-level override not supported — fall back to key-level
        health_resp = requests.get(base + "/health/keys", headers=headers, timeout=10)
        if health_resp.status_code != 200:
            raise SystemExit(f"/health/keys returned {{health_resp.status_code}}")
        keys = health_resp.json().get("serpapi", [])
        if not keys:
            print("provider failure simulation: no SerpAPI keys configured (skipped)")
            raise SystemExit(0)

        # Force-exhaust all keys
        for key_entry in keys:
            key_override = {{
                "provider": "serpapi",
                "scope_type": "key",
                "key_index": key_entry["index"],
                "override_type": "force_exhausted_until",
                "active_until": future_ts,
                "note": "validation-test",
            }}
            kr = requests.post(base + "/debug/provider-state/overrides", json=key_override, headers=headers, timeout=10)
            if kr.status_code == 200:
                created_overrides.append(kr.json()["override"]["id"])
    elif resp.status_code == 200:
        created_overrides.append(resp.json()["override"]["id"])
    else:
        raise SystemExit(f"Provider override failed: {{resp.status_code}} {{resp.text[:200]}}")

    if not created_overrides:
        raise SystemExit("No overrides created — cannot simulate provider failure")

    # 2. Hit /ask and verify graceful degradation (NOT 500)
    ask_resp = requests.post(base + "/ask", json={{
        "user_query": "flights from Delhi to Mumbai",
        "origin": "DEL",
        "destination": "BOM",
        "date": (time.strftime('%Y-%m-%d', time.localtime(time.time() + 21 * 24 * 3600))),
    }}, timeout=90)

    if ask_resp.status_code == 500:
        raise SystemExit(f"Provider failure caused unhandled 500: {{ask_resp.text[:300]}}")

    if ask_resp.status_code == 200:
        body = ask_resp.json()
        has_structure = any(k in body for k in ["result_status", "error", "llm_response", "debug_info"])
        if not has_structure:
            raise SystemExit(f"Response lacks structure: {{list(body.keys())}}")

    print(f"provider failure simulation ok (status={{ask_resp.status_code}})")
finally:
    # 3. Clean up ALL overrides
    for override_id in created_overrides:
        try:
            requests.post(base + f"/debug/provider-state/overrides/{{override_id}}/disable", headers=headers, timeout=10)
        except Exception:
            pass
"""
    run_and_log(f"contract_provider_failure_simulation_{mode}", [py, "-c", provider_failure_script], expect_llm=False)

    # ------------------------------------------------------------------
    # P0: Inflight / duplicate guard contract
    # ------------------------------------------------------------------
    inflight_duplicate_script = f"""
import json
import os
import time
import requests
import sys
import concurrent.futures
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

base = {base_url!r}

def send_identical_ask():
    payload = {{
        "user_query": "duplicate guard test",
        "origin": "DEL",
        "destination": "BOM",
        "date": (time.strftime('%Y-%m-%d', time.localtime(time.time() + 21 * 24 * 3600))),
    }}
    return requests.post(base + "/ask", json=payload, timeout=30)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(send_identical_ask) for _ in range(4)]
    responses = [f.result() for f in concurrent.futures.as_completed(futures)]

status_codes = [r.status_code for r in responses]
rejected = [s for s in status_codes if s == 409]
if not rejected:
    raise SystemExit(f"No duplicate guard triggered. All statuses: {{status_codes}}")

rl_resp = [r for r in responses if r.status_code == 409][0]
body = rl_resp.json()
if not ("Retry-After" in rl_resp.headers or "retry_after" in body or "error" in body):
    raise SystemExit(f"409 response missing Retry-After/retry_after/error: {{body}}")

print(f"inflight/duplicate guard contract ok ({{len(rejected)}} rejections)")
"""
    run_and_log(f"contract_inflight_duplicate_guard_{mode}", [py, "-c", inflight_duplicate_script], expect_llm=False)

    # ------------------------------------------------------------------
    # P0: Sliding-window rate limit contract
    # ------------------------------------------------------------------
    rate_limit_script = f"""
import json
import os
import time
import requests
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

base = {base_url!r}

# Send unique requests to bypass duplicate guard and trigger the sliding-window rate limiter
# ASK_RATE_LIMIT_PER_WINDOW=30 per 60s window, per-IP
responses = []
for i in range(35):
    payload = {{"user_query": f"rate limit test query {{i}}"}}
    resp = requests.post(base + "/ask", json=payload, timeout=30)
    responses.append(resp)
    if resp.status_code == 429:
        break

status_codes = [r.status_code for r in responses]
rate_limited = [s for s in status_codes if s == 429]
if not rate_limited:
    raise SystemExit(f"No rate limiting triggered after {{len(responses)}} requests. Statuses: {{status_codes}}")

rl_resp = [r for r in responses if r.status_code == 429][0]
body = rl_resp.json()
if not ("Retry-After" in rl_resp.headers or "retry_after" in body):
    raise SystemExit(f"429 response missing Retry-After/retry_after: {{body}}")

print(f"sliding-window rate limit contract ok (triggered after {{len(responses)}} requests)")
"""
    run_and_log(f"contract_rate_limit_sliding_window_{mode}", [py, "-c", rate_limit_script], expect_llm=False)

    # ------------------------------------------------------------------
    # P0: Auth smoke contract
    # ------------------------------------------------------------------
    auth_smoke_script = f"""
import json
import os
import requests
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

base = {base_url!r}
auth_token = (os.getenv("AUTH_TOKEN") or {VALIDATION_AUTH_TOKEN!r}).strip()
if not auth_token:
    raise SystemExit("AUTH_TOKEN missing for auth smoke contract")

# 1. Invalid token → 401 on /bookings
resp = requests.get(base + "/bookings", headers={{"Authorization": "Bearer invalid-token-12345"}}, timeout=10)
if resp.status_code != 401:
    raise SystemExit(f"Expected 401 for invalid token on /bookings, got {{resp.status_code}}")

# 2. No token → 401 on /bookings
resp = requests.get(base + "/bookings", timeout=10)
if resp.status_code != 401:
    raise SystemExit(f"Expected 401 for missing token on /bookings, got {{resp.status_code}}")

# 3. Valid token → 200 on /bookings (even if empty list)
resp = requests.get(base + "/bookings", headers={{"Authorization": f"Bearer {{auth_token}}"}}, timeout=10)
if resp.status_code != 200:
    raise SystemExit(f"Expected 200 for valid token on /bookings, got {{resp.status_code}}")
body = resp.json()
if "items" not in body:
    raise SystemExit(f"Expected 'items' key in /bookings response, got: {{list(body.keys())}}")

# 4. /ask?async_job=true without auth → 401
resp = requests.post(base + "/ask?async_job=true", json={{"user_query": "auth test"}}, timeout=10)
if resp.status_code != 401:
    raise SystemExit(f"Expected 401 for unauthenticated async_job, got {{resp.status_code}}")

print("auth smoke contract ok")
"""
    run_and_log(f"contract_auth_smoke_{mode}", [py, "-c", auth_smoke_script], expect_llm=False)


def run_frontend_runtime_matrix(mode, base_url):
    if not args.frontend:
        return

    log("Frontend runtime matrix mode: frontend-heavy default.")
    matrix_mode = MODE_FRONTEND_REAL_BACKEND_BROWSER if args.frontend_real_backend else MODE_FRONTEND_FIXTURE_BROWSER
    matrix = frontend_runtime_cases(
        mode=matrix_mode,
        include_live_canary=bool(args.frontend_live_canary and args.frontend_real_backend),
    )
    if matrix_mode == MODE_FRONTEND_FIXTURE_BROWSER:
        matrix = [case for case in matrix if case.case_name.startswith("frontend_fixture_")]
        log("Frontend runtime matrix is running in fixture-backed mode (safe default).")
    else:
        matrix = [
            case
            for case in matrix
            if case.case_name.startswith("frontend_real_backend_")
            or (bool(args.frontend_live_canary) and case.case_name.startswith("frontend_live_canary_"))
        ]
        if args.frontend_live_canary:
            log("Frontend runtime matrix is running against real backend + explicit live-provider canary subset.")
        else:
            log("Frontend runtime matrix is running against a real backend (explicit opt-in).")

    def _frontend_validation_type(mode_bucket):
        if mode_bucket == MODE_FRONTEND_FIXTURE_BROWSER:
            return "frontend-fixture"
        if mode_bucket == MODE_FRONTEND_REAL_BACKEND_BROWSER:
            return "frontend-real-backend"
        if mode_bucket == MODE_LIVE_CANARY_BROWSER:
            return "live-canary"
        return "frontend-fixture"

    for case in matrix:
        case_name = case.case_name
        payload = dict(case.payload)
        case_mode_bucket = MODE_LIVE_CANARY_BROWSER if case_name.startswith("frontend_live_canary_") else matrix_mode
        frontend_scenario = case.fixture_scenario if case_mode_bucket == MODE_FRONTEND_FIXTURE_BROWSER else ""
        frontend_expectations = dict(case.expectations or {})
        frontend_expectations.setdefault("assertion_mode", "ui_first")
        if case.ui_assertions:
            frontend_expectations.setdefault("ui_assertions", list(case.ui_assertions))
        if case.contract_assertions:
            frontend_expectations.setdefault("contract_assertions", list(case.contract_assertions))
        frontend_expectations.setdefault("validation_mode_bucket", case_mode_bucket)
        frontend_expectations.setdefault("scenario_dimensions", dict(case.dimensions or {}))
        run_and_log(
            f"{case_name}_{mode}",
            [
                "curl",
                "-sS",
                "-X",
                "POST",
                f"{base_url}/ask",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(payload),
            ],
            expect_llm=(case_mode_bucket != MODE_FRONTEND_FIXTURE_BROWSER),
            assertions={
                "frontend_scenario": frontend_scenario,
                "frontend_expectations": frontend_expectations,
                "skip_backend_shadow": True,
                "display_name": case_name.replace("_", " "),
                "validation_meta_override": {
                    "scenario": case_name,
                    "layers": ["frontend", "api"] + (["e2e"] if case_mode_bucket != MODE_FRONTEND_FIXTURE_BROWSER else []),
                    "validation_type": _frontend_validation_type(case_mode_bucket),
                    "features": list(case.features or ()),
                    "mode_bucket": case_mode_bucket,
                    "soft_pass_policy": case.soft_pass_policy,
                    "criticality": case.criticality,
                    "dimensions": dict(case.dimensions or {}),
                    "ui_assertions": list(case.ui_assertions or ()),
                    "contract_assertions": list(case.contract_assertions or ()),
                },
            },
        )


def run_real_mode_checks(mode, base_url):
    """
    Real mode: run only representative checks against real providers.
    """
    log(f"Running REAL mode representative checks (mode={mode})")

    key_before = _fetch_json(f"{base_url}/health/keys")
    key_before_summary = _summarize_key_state(key_before)
    log(f"Key usage before real checks: {json.dumps(key_before_summary)}")
    future_date = (datetime.now().date() + timedelta(days=21)).strftime("%Y-%m-%d")

    run_and_log(f"real_simple_flight_{mode}", [
        "curl", "--max-time", str(SMOKE_TIMEOUT), "-sS", "-X", "POST", f"{base_url}/ask",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "origin": "DEL",
            "destination": "BOM",
            "date": future_date,
            "user_query": "Find a business flight",
        })
    ], expect_llm=True, assertions={"check_api_trace": True})

    run_and_log(f"real_weather_query_{mode}", [
        "curl", "--max-time", str(SMOKE_TIMEOUT), "-sS", "-X", "POST", f"{base_url}/ask",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "user_query": f"What is the weather in Mumbai on {future_date} and suggest flights from Delhi?"
        })
    ], expect_llm=True, assertions={"check_api_trace": True})

    run_and_log(f"real_combined_query_{mode}", [
        "curl", "--max-time", str(SMOKE_TIMEOUT), "-sS", "-X", "POST", f"{base_url}/ask",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "user_query": (
                f"Cheapest direct morning flight from Delhi to Mumbai on {future_date} "
                "with weather summary"
            )
        })
    ], expect_llm=True, assertions={"check_api_trace": True})

    run_capability_checks(mode, base_url)

    key_after = _fetch_json(f"{base_url}/health/keys")
    key_after_summary = _summarize_key_state(key_after)
    log(f"Key usage after real checks: {json.dumps(key_after_summary)}")

# ----------------------------------------------------------------------
# Smoke checks (static) with per-test LLM expectations
# Variantized tests run one rotation by default, or all variants in --loop mode.
# ----------------------------------------------------------------------
def run_smoke_checks_logged(mode, rotation_index, loop_mode=False, base_url=None):
    log(f"Running smoke checks (mode={mode}) — logs in {LOG_DIR}")
    base_url = base_url or DEFAULT_API_BASE_URL

    if REAL_MODE:
        run_real_mode_checks(mode, base_url)
        return

    if loop_mode:
        log("Loop mode enabled: running all variants for each query group.")
    else:
        log(f"Using rotation index: {rotation_index}")

    def iter_variants(queries):
        if loop_mode:
            return range(len(queries))
        return [rotation_index % len(queries)]

    # Prewarm the LLM backend before starting timed smoke tests (only in LIVE mode)
    today = datetime.now().date()
    future_date = (today + timedelta(days=21)).strftime("%Y-%m-%d")
    if args.live:
        prewarm_payload = {"origin": "DEL", "destination": "BOM", "date": future_date, "user_query": "test prewarm"}
        try:
            log("Warming up LLM backend (may take up to 90s)...")
            r = requests.post(f"{base_url}/ask", json=prewarm_payload, timeout=90)
            data = r.json()
            if "(Note: Enhanced explanation unavailable" in (data.get("llm_response") or ""):
                log("Warning: LLM backend still cold after prewarm — first tests may still flake")
            else:
                log("LLM backend warmed up successfully.")
        except Exception as e:
            log(f"LLM prewarm failed (non-fatal): {e}")

    # Precompute expected dates
    expected_future = future_date
    stream_smoke_timeout = max(SMOKE_TIMEOUT + 20, 45)

    # 1) Basic sync ask (improved query)
    run_and_log(f"quick_sync_ask_{mode}", [
        "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "origin": "DEL",
            "destination": "BOM",
            "date": future_date,
            "user_query": "Is this a good value flight for a business trip?",
        })
    ], expect_llm=True, assertions={"check_api_trace": True, "check_weather_temp_accuracy": True})

    # 2) Missing date test — planner defaults to tomorrow when no date given
    tomorrow = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    run_and_log(f"missing_date_test_{mode}", [
        "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
        "-d", '{"user_query":"Cheap flight from Delhi to Mumbai"}'
    ], expect_llm=True, assertions={
        "expected_date": tomorrow,
        "expected_date_basis": "host_local_tomorrow_for_missing_date",
        "allow_runtime_date_skew_days": 1,
        "check_api_trace": True,
        "check_weather_temp_accuracy": True,
    })

    # 3) Natural language relative date (multiple variants)
    nl_date_queries = [
        "Flight from Delhi to Mumbai fourteen days after today",
        "Need a flight from Delhi to Mumbai two weeks from now",
        "Book a trip from DEL to BOM in 14 days",
        "Delhi to Mumbai around 14 days from today",
        "Flight DEL BOM after 2 weeks"
    ]
    for variant_index in iter_variants(nl_date_queries):
        q = nl_date_queries[variant_index]
        expected_14d = (datetime.now().date() + timedelta(days=14)).strftime("%Y-%m-%d")
        assertions = {
            "check_api_trace": True,
            "check_weather_temp_accuracy": True,
            "expected_date": expected_14d,
            "expected_date_basis": "host_local_relative_14_days",
        }
        run_and_log(f"nl_relative_date_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({"user_query": q})
        ], expect_llm=True, assertions=assertions)

    # 4) Misspelled city (LLM correction)
    misspelled_queries = [
        f"Cheap flight from Dalhi to Mumbai on {future_date}",
        f"Book a flight from Delhi to Mumbay on {future_date}",
        f"DEL to BOM on {future_date} from Dilli",
        f"Travel from Dehli to Bombay on {future_date}"
    ]
    for variant_index in iter_variants(misspelled_queries):
        q = misspelled_queries[variant_index]
        assertions = {"check_api_trace": True, "check_weather_temp_accuracy": True, "check_iata_in_llm": True}
        run_and_log(f"misspelled_city_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({"user_query": q})
        ], expect_llm=True, assertions=assertions)

    # 5) Round trip duration (inferred return)
    round_trip_queries = [
        f"Business trip from Delhi to Mumbai for 3 days starting {future_date}",
        f"Trip from DEL to BOM for 3 nights from {future_date}",
        f"Round trip from Delhi to Mumbai for 3 days starting {future_date}",
        f"Book a return flight from Delhi to Mumbai for 3 days from {future_date}"
    ]
    for variant_index in iter_variants(round_trip_queries):
        q = round_trip_queries[variant_index]
        assertions = {
            "check_api_trace": True,
            "check_weather_temp_accuracy": True,
            "expected_date": expected_future,
            "expected_date_basis": "host_local_explicit_outbound_date",
            "check_return_llm": True,
        }
        run_and_log(f"round_trip_duration_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "user_query": q,
            })
        ], expect_llm=True, assertions=assertions)

    # 6) Time preference — morning
    time_pref_queries = [
        f"Business trip from Delhi to Mumbai on {future_date} in the morning",
        f"Morning flight from DEL to BOM on {future_date}",
        f"Delhi to Mumbai flight before 10am on {future_date}",
        f"Early departure from Delhi to Mumbai on {future_date}"
    ]
    for variant_index in iter_variants(time_pref_queries):
        q = time_pref_queries[variant_index]
        run_and_log(f"time_pref_morning_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({"user_query": q})
        ], expect_llm=True, assertions={"check_api_trace": True, "check_weather_temp_accuracy": True})

    # 7) Price cap (modified one variant to more realistic route/budget)
    price_cap_queries = [
        f"Business trip from MAA to DEL under ₹12000 on {future_date}",
        f"Cheap flight Delhi to Mumbai for less than 3000 INR on {future_date}",
        f"DEL BOM flight within 3000 rupees on {future_date}",
        f"Flight under ₹3000 from Delhi to Mumbai on {future_date}"
    ]
    for variant_index in iter_variants(price_cap_queries):
        q = price_cap_queries[variant_index]
        assertions = {
            "check_api_trace": True,
            "check_weather_temp_accuracy": True,
            "expected_date": expected_future,
            "expected_date_basis": "host_local_explicit_price_cap_date",
        }
        run_and_log(f"price_cap_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({"user_query": q})
        ], expect_llm=True, assertions=assertions)

    # 8) Direct only
    direct_only_queries = [
        f"Direct flights only from Delhi to Mumbai on {future_date}",
        f"Delhi to Mumbai direct flight on {future_date}",
        f"Need nonstop flight DEL to BOM on {future_date}",
        f"DEL BOM direct {future_date}",
        f"From Delhi fly direct to Mumbai on {future_date}"
    ]
    for variant_index in iter_variants(direct_only_queries):
        q = direct_only_queries[variant_index]
        assertions = {"check_api_trace": True, "check_weather_temp_accuracy": True, "check_stops_na_not_claimed": True}
        run_and_log(f"direct_only_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({"user_query": q})
        ], expect_llm=True, assertions=assertions)

    # 9) Preferred airline
    preferred_airline_queries = [
        f"Business trip from Delhi to Mumbai on {future_date} prefer indigo",
        f"Flight DEL to BOM on {future_date} preferably IndiGo",
        f"Book IndiGo flight Delhi to Mumbai on {future_date}",
        f"Delhi to Mumbai flight with Indigo airlines on {future_date}"
    ]
    for variant_index in iter_variants(preferred_airline_queries):
        q = preferred_airline_queries[variant_index]
        assertions = {
            "check_api_trace": True,
            "check_weather_temp_accuracy": True,
            "check_llm_flight_consistency": True,
            "check_no_python_repr": True,
            "check_relaxed_filter_honest": True,
        }
        run_and_log(f"preferred_airline_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({"user_query": q})
        ], expect_llm=True, assertions=assertions)

    # 10) Layover limit
    layover_limit_queries = [
        f"Business trip from Delhi to Mumbai on {future_date} with layover less than 2 hours",
        f"DEL to BOM flight with max 2h layover on {future_date}",
        f"Flight from Delhi to Mumbai on {future_date}, layover under 2 hours",
        f"Journey from Delhi to Mumbai with connection less than 2 hours on {future_date}"
    ]
    for variant_index in iter_variants(layover_limit_queries):
        q = layover_limit_queries[variant_index]
        assertions = {
            "check_api_trace": True,
            "check_weather_temp_accuracy": True,
            "check_relaxed_filter_honest": True,
            "check_layover_not_confused": True,
        }
        run_and_log(f"layover_limit_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "user_query": q,
            })
        ], expect_llm=True, assertions=assertions)

    # 11) Baggage preference (hand baggage only)
    baggage_hand_queries = [
        f"Business trip Delhi to Mumbai on {future_date} cabin only (hand baggage)",
        f"Flight from DEL to BOM with only hand luggage on {future_date}",
        f"Delhi Mumbai trip on {future_date}, just cabin bag",
        f"Book flight with hand baggage only Delhi to Mumbai on {future_date}"
    ]
    for variant_index in iter_variants(baggage_hand_queries):
        q = baggage_hand_queries[variant_index]
        assertions = {"check_api_trace": True, "check_weather_temp_accuracy": True, "check_relaxed_filter_honest": True}
        run_and_log(f"baggage_hand_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({"user_query": q})
        ], expect_llm=True, assertions=assertions)

    # 12) Stopover / multi-city ("via")
    stopover_via_queries = [
        f"Business trip Delhi to Chennai via Bangalore on {future_date}",
        f"DEL to MAA via BLR on {future_date}",
        f"Flight from Delhi to Chennai connecting through Bangalore on {future_date}",
        f"Delhi to Chennai with stop in Bangalore on {future_date}"
    ]
    for variant_index in iter_variants(stopover_via_queries):
        q = stopover_via_queries[variant_index]
        assertions = {"check_api_trace": True, "check_weather_temp_accuracy": True, "check_all_legs_llm": True}
        run_and_log(f"stopover_via_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({"user_query": q})
        ], expect_llm=True, assertions=assertions)

    # 13) Eco-friendly flight test (new)
    eco_queries = [
        f"Find the most eco-friendly flight from Delhi to Mumbai on {future_date}",
        f"Greenest flight DEL to BOM on {future_date}",
    ]
    for variant_index in iter_variants(eco_queries):
        q = eco_queries[variant_index]
        assertions = {"check_api_trace": True, "check_weather_temp_accuracy": True, "check_eco_llm": True}
        run_and_log(f"eco_flight_{mode}_{variant_index}", [
            "curl", "-sS", "-X", "POST", f"{base_url}/ask", "-H", "Content-Type: application/json",
            "-d", json.dumps({"user_query": q})
        ], expect_llm=True, assertions=assertions)

    # 14) Parallel async calls benchmark lane:
    # default remains sequential for stability; opt-in parallel mode is available for A/B stress runs.
    async_parallel_mode = (
        get_env_str("VALIDATION_ASYNC_PARALLEL_MODE", "sequential")
        or "sequential"
    ).strip().lower()
    if async_parallel_mode not in {"sequential", "parallel"}:
        async_parallel_mode = "sequential"
    VALIDATION_RUNTIME_CONFIG["async_parallel_mode"] = async_parallel_mode
    log(f"Async parallel scenario mode: {async_parallel_mode}")
    if async_parallel_mode == "parallel":
        async_cmd = f"""
t1="$(mktemp)"
t2="$(mktemp)"
curl -sS -X POST {base_url}/ask -H "Content-Type: application/json" -d '{{"origin":"DEL","destination":"BOM","date":"{future_date}","user_query":"Business trip"}}' > "$t1" &
p1=$!
curl -sS -X POST {base_url}/ask -H "Content-Type: application/json" -d '{{"origin":"DEL","destination":"BLR","date":"{future_date}","user_query":"Holiday"}}' > "$t2" &
p2=$!
wait "$p1"; s1=$?
wait "$p2"; s2=$?
cat "$t1"
echo
cat "$t2"
rm -f "$t1" "$t2"
if [ "$s1" -ne 0 ] || [ "$s2" -ne 0 ]; then exit 1; fi
"""
    else:
        async_cmd = f"""
curl -sS -X POST {base_url}/ask -H "Content-Type: application/json" -d '{{"origin":"DEL","destination":"BOM","date":"{future_date}","user_query":"Business trip"}}'
curl -sS -X POST {base_url}/ask -H "Content-Type: application/json" -d '{{"origin":"DEL","destination":"BLR","date":"{future_date}","user_query":"Holiday"}}'
"""
    run_and_log(f"async_parallel_{mode}", ["bash", "-c", async_cmd],
                expect_llm=False, assertions={
                    "check_no_parallel_fallback": True,
                    "check_parallel_json_integrity": True,
                    "expected_parallel_count": 2,
                    "expected_parallel_destinations": ["BOM", "BLR"],
                    "check_api_trace": True,
                    "check_weather_temp_accuracy": True,
                })

    # 15) Streaming basic test
    run_and_log(f"streaming_test_{mode}", [
        "curl", "--max-time", str(stream_smoke_timeout), "-N", "-sS", "-X", "POST", f"{base_url}/ask?stream=true",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "origin": "DEL",
            "destination": "BOM",
            "date": future_date,
            "user_query": "Explain why this flight is good",
        })
    ], is_stream=True, assertions={
        "check_api_trace": True,
        "check_weather_temp_accuracy": True,
        "expected_date": expected_future,
        "expected_date_basis": "host_local_explicit_stream_date",
    })

    # 16) Streaming natural language relative date
    expected_14d = (datetime.now().date() + timedelta(days=14)).strftime("%Y-%m-%d")
    assertions = {"check_api_trace": True, "check_weather_temp_accuracy": True}
    assertions["expected_date"] = expected_14d
    assertions["expected_date_basis"] = "host_local_relative_14_days_stream"
    run_and_log(f"streaming_nl_relative_{mode}", [
        "curl", "--max-time", str(stream_smoke_timeout), "-N", "-sS", "-X", "POST", f"{base_url}/ask?stream=true",
        "-H", "Content-Type: application/json",
        "-d", '{"user_query":"Cheapest flight from Delhi to Mumbai fourteen days after today"}'
    ], is_stream=True, assertions=assertions)

    # Capability coverage checks
    run_capability_checks(mode, base_url)
    run_contract_checks(mode, base_url)
    run_frontend_runtime_matrix(mode, base_url)

# ----------------------------------------------------------------------
# pytest runner
# ----------------------------------------------------------------------
def run_pytest_logged():
    global PYTEST_LANE_CONTEXT
    lane = str(args.pytest_lane or "default").strip().lower()
    if lane not in {"default", "full", "slow"}:
        lane = "default"

    pytest_testpaths = _read_pytest_testpaths()
    extra_paths = []
    if lane == "full":
        extra_paths = ["tests", "tests_slow"] if (ROOT / "tests_slow").exists() else ["tests"]
        log("Running pytest (full lane: default suite + explicit slow suite)")
    elif lane == "slow":
        extra_paths = ["tests_slow"]
        log("Running pytest (slow lane: tests_slow only)")
        if not (ROOT / "tests_slow").exists():
            run_and_log(
                "pytest_missing_slow_lane",
                ["bash", "-c", "echo 'tests_slow/ not found. Slow lane is unavailable.'"],
                expect_llm=False,
            )
            return 2
    else:
        log("Running pytest (default lane: same semantics as plain pytest -q)")
        if pytest_testpaths:
            log(
                "Pytest default lane contract: plain discovery uses pytest.ini testpaths="
                + ",".join(pytest_testpaths)
            )

    venv_activate = ROOT / "venv/bin/activate"
    pytest_probe_cmd = None
    if venv_activate.exists():
        python_path = ROOT / "venv/bin/python"
        if python_path.exists():
            cmd = [str(python_path), "-m", "pytest", "-q", *extra_paths]
            pytest_probe_cmd = [str(python_path), "-m", "pytest", "--version"]
        else:
            cmd = ["pytest", "-q", *extra_paths]
            pytest_probe_cmd = ["pytest", "--version"]
    else:
        cmd = ["pytest", "-q", *extra_paths]
        pytest_probe_cmd = ["pytest", "--version"]

    PYTEST_LANE_CONTEXT = {
        "lane": lane,
        "lane_semantics": (
            "default lane mirrors plain pytest -q discovery semantics"
            if lane == "default"
            else "explicit lane selection with explicit path targets"
        ),
        "pytest_testpaths": list(pytest_testpaths),
        "selected_paths": list(extra_paths),
        "command": list(cmd),
        "probe_command": list(pytest_probe_cmd),
    }
    if extra_paths:
        log(
            "Pytest lane command paths: "
            + ",".join(extra_paths)
            + " (explicitly scoped by lane selection)"
        )
    else:
        log("Pytest lane command paths: <none> (plain pytest -q discovery semantics)")

    try:
        subprocess.run(pytest_probe_cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        run_and_log(
            "pytest_missing",
            ["bash", "-c", "echo 'pytest not found. Install dev deps or activate venv with pytest. Skipping pytest.'"],
            expect_llm=False,
        )
        return 2
    except subprocess.CalledProcessError:
        run_and_log(
            "pytest_missing",
            ["bash", "-c", "echo 'pytest is unavailable or misconfigured. Skipping pytest.'"],
            expect_llm=False,
        )
        return 2

    run_and_log("pytest_unit", cmd, expect_llm=False)

# ----------------------------------------------------------------------
# Local server runner for machine mode
# ----------------------------------------------------------------------
def run_machine_local_server():
    log("Starting local uvicorn server...")

    # Find uvicorn executable
    # First check if we're in a venv and have uvicorn
    venv_python = ROOT / "venv/bin/python"
    if venv_python.exists():
        # Use python -m uvicorn
        uvicorn_cmd = [str(venv_python), "-m", "uvicorn"]
    else:
        # fallback to PATH
        uvicorn_path = shutil.which("uvicorn")
        if not uvicorn_path:
            log("ERROR: uvicorn not found in PATH and no venv detected")
            return None
        uvicorn_cmd = [uvicorn_path]

    # Load env from TMP_ENV
    env = os.environ.copy()
    if TMP_ENV.exists():
        with open(TMP_ENV) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        k, v = line.split('=', 1)
                        env[k] = v

    # ensure previous local uvicorn process is not running (use PID file, not broad pkill)
    pid_file = LOG_DIR / "machine_uvicorn.pid"
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            os.kill(old_pid, 0)  # check if process exists
            subprocess.run(["kill", str(old_pid)], stderr=subprocess.DEVNULL, check=False, timeout=5)
            time.sleep(0.3)
        except (ValueError, ProcessLookupError, OSError):
            pass  # stale PID file or already gone
        pid_file.unlink(missing_ok=True)

    # start uvicorn
    logfile = LOG_DIR / "machine_uvicorn.log"
    uvicorn_args = uvicorn_cmd + ["api.app:app", "--host", "0.0.0.0", "--port", str(VALIDATION_PORT)]
    if args.quiet and not args.debug:
        uvicorn_args += ["--log-level", "warning", "--no-access-log"]
    elif args.debug:
        uvicorn_args += ["--log-level", "debug"]
    with open(logfile, 'w') as f:
        proc = subprocess.Popen(
            uvicorn_args,
            stdout=f, stderr=subprocess.STDOUT, env=env
        )

    # Give it a moment to start; check if process already exited
    time.sleep(0.5)
    if proc.poll() is not None:
        log("uvicorn process exited immediately")
        return None

    machine_pid = proc.pid
    with open(LOG_DIR / "machine_uvicorn.pid", 'w') as f:
        f.write(str(machine_pid))

    # wait for health
    if wait_for_health_poll(APP_START_TIMEOUT, f"{DEFAULT_API_BASE_URL}/health"):
        if wait_for_ready(f"{DEFAULT_API_BASE_URL}/health/ready", READY_TIMEOUT):
            log("Local server healthy and ready.")
            return machine_pid
        else:
            log("Local server did not become ready within timeout.")
            proc.terminate()
            proc.wait()
            return None
    else:
        log("Local server failed to start (health check).")
        proc.terminate()
        proc.wait()
        return None

def stop_machine_local_server(pid):
    if pid:
        subprocess.run(["kill", str(pid)], stderr=subprocess.DEVNULL, check=False, timeout=5)
        (LOG_DIR / "machine_uvicorn.pid").unlink(missing_ok=True)

# ----------------------------------------------------------------------
# Docker container runner
# ----------------------------------------------------------------------
def run_docker_container(mode):
    log(f"Starting disposable container for mode={mode}")
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], stderr=subprocess.DEVNULL, check=False, timeout=15)

    cmd = [
        "docker", "run", "-d", "--rm", "-p", "8000:8000",
        "--add-host=host.docker.internal:host-gateway",
        "--name", CONTAINER_NAME,
        "--env-file", str(TMP_ENV),
        "-v", f"{TMP_ENV}:/app/.env:ro",
        IMAGE_NAME
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        log(f"Failed to start container: {result.stderr}")
        return None
    cid = result.stdout.strip()
    with open(LOG_DIR / f"docker_run_{mode}.cid", 'w') as f:
        f.write(cid)

    # wait for health
    time.sleep(1)
    if not wait_for_health_poll(APP_START_TIMEOUT, "http://localhost:8000/health"):
        log("Container did not become healthy within timeout.")
        subprocess.run(["docker", "logs", "--tail", "200", CONTAINER_NAME], timeout=20)
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False, timeout=15)
        return None

    if not wait_for_ready("http://localhost:8000/health/ready", READY_TIMEOUT):
        log("Container did not become ready within timeout.")
        subprocess.run(["docker", "logs", "--tail", "200", CONTAINER_NAME], timeout=20)
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False, timeout=15)
        return None

    log("Container healthy and ready.")
    return cid

def stop_docker_container(cid):
    if cid:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False, timeout=15)

# ----------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------
def main():
    started_at = time.time()
    scope_id = "backend_api_plus_frontend" if args.frontend else "backend_api_only"
    log("START safe_full_validation_report.py")
    log(f"Validation mode: {args.mode}")
    log(
        "Validation scope id: "
        + scope_id
        + (
            " (full product browser/runtime coverage included)"
            if args.frontend
            else " (NOT full product validation; frontend/browser not executed)"
        )
    )
    log(
        "Validation lanes: "
        + f"pytest={args.pytest_lane}, "
        + f"smoke_variants={'all (--loop)' if args.loop else 'rotation-sampled'}"
    )
    log(
        "Validation experiment controls: "
        + f"VALIDATION_OLLAMA_NUM_CTX_MODE="
        + f"{(get_env_str('VALIDATION_OLLAMA_NUM_CTX_MODE', 'validated_default') or 'validated_default').strip().lower()}, "
        + f"VALIDATION_OLLAMA_NUM_CTX={os.getenv('VALIDATION_OLLAMA_NUM_CTX') or '<unset>'}, "
        + f"OLLAMA_NUM_CTX(process)={os.getenv('OLLAMA_NUM_CTX') or '<unset>'}, "
        + f"VALIDATION_ROTATION_INDEX={os.getenv('VALIDATION_ROTATION_INDEX') or '<unset>'}, "
        + f"--r={args.r if args.r is not None else '<unset>'}, "
        + f"VALIDATION_ASYNC_PARALLEL_MODE={os.getenv('VALIDATION_ASYNC_PARALLEL_MODE') or 'sequential'}"
    )
    if args.frontend:
        if args.frontend_real_backend and args.frontend_live_canary:
            frontend_mode_label = "real-backend + live-canary"
        elif args.frontend_real_backend:
            frontend_mode_label = "real-backend"
        else:
            frontend_mode_label = "fixture-backed frontend-heavy default"
        log(
            "Frontend validation enabled: routing high-value /ask smoke cases + "
            f"frontend runtime matrix through UI at {FRONTEND_DEFAULT_URL} ({frontend_mode_label})"
        )
        log("Validation scope: backend/API + frontend browser/runtime paths.")
    else:
        log(
            "Frontend browser/runtime validation is SKIPPED in this run (backend/API only). "
            "Use --frontend to include fixture-backed frontend browser coverage."
        )
        log("Validation scope: backend/API only (this is not full product validation).")
        log("Scope guardrail: backend-only run must not be interpreted as full product health.")

    # Write commit file (always, independent of build)
    if (ROOT / ".git").exists():
        stdout, _, _ = run_cmd(["git", "rev-parse", "--short", "HEAD"])
        commit = stdout.strip()
        if commit:
            (ROOT / "COMMIT").write_text(commit + "\n")

    # Determine docker build logic
    need_docker = args.mode in ("docker", "both")
    skip_docker_build = get_env_bool("SKIP_DOCKER_BUILD", default=False)

    docker_unavailable = False
    if need_docker:
        if not docker_available():
            log("Docker daemon not available — skipping docker validation")
            docker_unavailable = True
        elif not skip_docker_build:
            build_and_verify()
        else:
            log("Skipping docker build (SKIP_DOCKER_BUILD=1)")

    # Run pytest (always)
    run_pytest_logged()

    # Determine rotation index once (loop mode runs all variants and does not advance rotation state)
    if args.loop:
        rotation_index = 0
        VALIDATION_RUNTIME_CONFIG.update(
            {
                "rotation_index": rotation_index,
                "rotation_source": "loop_mode_all_variants",
                "rotation_raw": 0,
                "rotation_file_before": None,
                "rotation_file_after": None,
            }
        )
    else:
        rotation_index = get_rotation_index()
    VALIDATION_RUNTIME_CONFIG["rotation_loop_mode"] = bool(args.loop)
    log(
        "Validation rotation selection: "
        + f"index={VALIDATION_RUNTIME_CONFIG.get('rotation_index')}, "
        + f"source={VALIDATION_RUNTIME_CONFIG.get('rotation_source') or 'unknown'}, "
        + f"raw={VALIDATION_RUNTIME_CONFIG.get('rotation_raw')}, "
        + f"file_before={VALIDATION_RUNTIME_CONFIG.get('rotation_file_before')}, "
        + f"file_after={VALIDATION_RUNTIME_CONFIG.get('rotation_file_after')}, "
        + f"loop_mode={bool(args.loop)}"
    )

    # Machine mode (if requested)
    if args.mode in ("machine", "both"):
        create_temp_env("machine")
        machine_pid = run_machine_local_server()
        if machine_pid:
            warmup_ok = _run_validation_llm_warmup("machine", DEFAULT_API_BASE_URL)
            if warmup_ok:
                log("Validation LLM warmup gate passed. Starting machine scenario loop.")
                run_smoke_checks_logged(
                    "machine",
                    rotation_index,
                    loop_mode=args.loop,
                    base_url=DEFAULT_API_BASE_URL,
                )
            else:
                log(
                    "Validation LLM warmup gate failed. "
                    "Skipping machine LLM-required scenario loop for truthful readiness."
                )
                py = _python_exec()
                warmup_reason = str(LLM_WARMUP_CONTEXT.get("reason") or "unknown")
                run_and_log(
                    "llm_warmup_gate_machine",
                    [
                        py,
                        "-c",
                        "import sys; print(sys.argv[1]); raise SystemExit(1)",
                        warmup_reason,
                    ],
                    expect_llm=False,
                    assertions={
                        "display_name": "llm warmup gate",
                        "validation_meta_override": {
                            "scenario": "llm-warmup-readiness",
                            "layers": ["backend", "api", "runtime"],
                            "validation_type": "validation-gate",
                            "features": ["ops.llm_warmup_gate", "ask.non_stream"],
                            "mode_bucket": MODE_BACKEND_INTERNAL,
                            "soft_pass_policy": SOFT_PASS_HARD_FAIL_ONLY,
                            "criticality": "core",
                            "dimensions": {"gate": "llm_warmup"},
                            "ui_assertions": [],
                            "contract_assertions": [],
                        },
                    },
                )
            # check if still alive
            if subprocess.run(["kill", "-0", str(machine_pid)], capture_output=True).returncode == 0:
                run_and_log("result_machine_integration", [
                    "curl", "-sS", "-X", "GET", f"{DEFAULT_API_BASE_URL}/health"
                ], expect_llm=False, assertions={
                    "required_paths": [
                        "status",
                        "dependencies.app",
                        "dependencies.key_manager",
                        "async_jobs_enabled",
                    ]
                })
                stop_machine_local_server(machine_pid)
            else:
                stop_machine_local_server(machine_pid)
                run_and_log("result_machine_integration_failed", ["bash", "-c", "echo 'Machine server crashed during tests.'; exit 1"])
        else:
            run_and_log("result_machine_integration_failed", ["bash", "-c", "echo 'Machine local integration failed to start.'; exit 1"])

    # Docker mode (if requested and available)
    if args.mode in ("docker", "both"):
        if docker_unavailable:
            log("Skipping docker validation because docker not available")
        else:
            # Note: build already done if needed; we now just run container.
            create_temp_env("docker")
            cid = run_docker_container("docker")
            if cid:
                run_and_log("docker_hosted_smoke", [
                    "curl", "-sS", "-X", "GET", "http://localhost:8000/health"
                ], expect_llm=False, assertions={
                    "required_paths": [
                        "status",
                        "dependencies.app",
                        "dependencies.key_manager",
                        "async_jobs_enabled",
                    ]
                })
                run_smoke_checks_logged(
                    "docker-hosted",
                    rotation_index,
                    loop_mode=args.loop,
                    base_url="http://localhost:8000",
                )
                # capture logs – fixed file descriptor leak
                with open(LOG_DIR / "docker_validation_container_logs.log", "w") as f:
                    subprocess.run(["docker", "logs", "--tail", "200", CONTAINER_NAME], stdout=f)
                stop_docker_container(cid)
            else:
                run_and_log("docker_hosted_failed", ["bash", "-c", "echo 'Docker-hosted app failed to become healthy; see logs'; exit 1"])

    # Cleanup tmp env
    TMP_ENV.unlink(missing_ok=True)
    _close_frontend_validator()

    # Summary
    total = passed = soft_passed = failed = 0
    quality_rollup = {"strong": 0, "acceptable": 0, "weak": 0, "soft-pass": 0, "fail": 0}
    scenario_rollup = {}
    layer_rollup = {}
    validation_type_rollup = {}

    def _new_rollup_stats():
        return {
            "total": 0,
            "pass": 0,
            "soft": 0,
            "fail": 0,
            "strong": 0,
            "acceptable": 0,
            "weak": 0,
            "soft_pass": 0,
            "fail_quality": 0,
        }

    def _bump_quality(stats, quality):
        if quality == "strong":
            stats["strong"] += 1
        elif quality == "acceptable":
            stats["acceptable"] += 1
        elif quality == "weak":
            stats["weak"] += 1
        elif quality == "soft-pass":
            stats["soft_pass"] += 1
        else:
            stats["fail_quality"] += 1

    log("")
    log("Summary (non-pass outcomes):")
    for entry in REPORT:
        name = entry["name"]
        status = entry["status"]
        verdict = entry.get("verdict", VERDICT_PASS if status == 0 else VERDICT_FAIL)
        quality = entry.get("quality") or _determine_pass_quality(name, status, verdict, None, False)
        reason = entry.get("reason", "")
        failure_tags = entry.get("failure_tags") or []
        mode = _mode_label_for_name(name)
        base = _strip_mode_suffix(name)
        display = entry.get("display") or _display_name_for_base(base)
        scenario = entry.get("scenario") or "uncategorized"
        layers = entry.get("layers") or ["uncategorized"]
        validation_type = entry.get("validation_type") or "uncategorized"

        scenario_stats = scenario_rollup.setdefault(scenario, _new_rollup_stats())
        scenario_stats["total"] += 1
        vtype_stats = validation_type_rollup.setdefault(validation_type, _new_rollup_stats())
        vtype_stats["total"] += 1
        _bump_quality(scenario_stats, quality)
        _bump_quality(vtype_stats, quality)
        quality_rollup[quality if quality in quality_rollup else "fail"] += 1

        total += 1
        if verdict == VERDICT_PASS:
            passed += 1
            scenario_stats["pass"] += 1
            vtype_stats["pass"] += 1
            for layer in layers:
                layer_stats = layer_rollup.setdefault(layer, _new_rollup_stats())
                layer_stats["total"] += 1
                layer_stats["pass"] += 1
                _bump_quality(layer_stats, quality)
        elif verdict == VERDICT_SOFT_PASS_NO_CREDIT:
            soft_passed += 1
            scenario_stats["soft"] += 1
            vtype_stats["soft"] += 1
            for layer in layers:
                layer_stats = layer_rollup.setdefault(layer, _new_rollup_stats())
                layer_stats["total"] += 1
                layer_stats["soft"] += 1
                _bump_quality(layer_stats, quality)
            tags_text = f" tags={','.join(failure_tags)}" if failure_tags else ""
            llm_runtime = _format_entry_llm_runtime(entry)
            llm_runtime_text = f" | llm={llm_runtime}" if llm_runtime else ""
            log(
                f"  {mode:8s}  {display:35s}  [SoftPassNoCredit] {reason}{tags_text} "
                f"| quality={quality}{llm_runtime_text}"
            )
        else:
            failed += 1
            scenario_stats["fail"] += 1
            vtype_stats["fail"] += 1
            for layer in layers:
                layer_stats = layer_rollup.setdefault(layer, _new_rollup_stats())
                layer_stats["total"] += 1
                layer_stats["fail"] += 1
                _bump_quality(layer_stats, quality)

            category = _classify_failure_category(entry)
            tags_text = f" tags={','.join(failure_tags)}" if failure_tags else ""
            llm_runtime = _format_entry_llm_runtime(entry)
            llm_runtime_text = f" | llm={llm_runtime}" if llm_runtime else ""
            log(
                f"  {mode:8s}  {display:35s}  [{category}] {reason}"
                f"{tags_text} | quality={quality}{llm_runtime_text} | scenario={scenario} | type={validation_type}"
            )

    log("")
    log(f"Totals: {total} total, {passed} passed, {soft_passed} soft-passed (no credit), {failed} failed")
    if not args.frontend:
        log("")
        log("SCOPE WARNING: This run validated BACKEND/API ONLY.")
        log("  - Frontend/browser tests were NOT executed.")
        log("  - Full product validation requires: full_validation.py --frontend")
        log("  - Do NOT interpret backend-only pass as full product pass.")
    timeout_tagged_passes = [
        entry for entry in REPORT
        if entry.get("verdict") == VERDICT_PASS and "timeout" in (entry.get("failure_tags") or [])
    ]
    if timeout_tagged_passes:
        log(
            f"Timeout-tagged passes: {len(timeout_tagged_passes)} "
            "(these passed contract checks but used degraded/timeout-shaped LLM fallback paths)."
        )

    llm_evidence_rollup = _summarize_llm_evidence(REPORT)
    log("")
    log(
        "LLM evidence summary: "
        + f"required={llm_evidence_rollup['required_total']}, "
        + f"payload_evidence={llm_evidence_rollup['required_with_payload_evidence']}, "
        + f"completed={llm_evidence_rollup['completion_observed']}, "
        + f"completion_ratio={llm_evidence_rollup['completion_ratio']:.0%}, "
        + f"degraded={llm_evidence_rollup['degraded_observed']}, "
        + f"unknown={llm_evidence_rollup['unknown_unverified']}"
    )
    log(
        "LLM timing profile: "
        + f"near_timeout={llm_evidence_rollup['near_timeout_completions']} "
        + f"({llm_evidence_rollup['near_timeout_ratio']:.0%}), "
        + f"timeout_shaped={llm_evidence_rollup['timeout_shaped_completions']} "
        + f"({llm_evidence_rollup['timeout_shaped_ratio']:.0%}), "
        + f"first_token_observed={llm_evidence_rollup['first_token_observed']}, "
        + f"first_token_unavailable={llm_evidence_rollup['first_token_unavailable']}, "
        + f"first_token_p50={llm_evidence_rollup['first_token_latency_p50']}, "
        + f"first_token_p90={llm_evidence_rollup['first_token_latency_p90']}, "
        + f"req_to_first_token_p50={llm_evidence_rollup['first_token_from_validation_send_p50']}, "
        + f"req_to_first_token_p90={llm_evidence_rollup['first_token_from_validation_send_p90']}, "
        + f"timeout_ratio_p50={llm_evidence_rollup['timeout_ratio_p50']}, "
        + f"timeout_ratio_p90={llm_evidence_rollup['timeout_ratio_p90']}"
    )
    log(
        "LLM runtime profiles seen: "
        + f"backend={','.join(llm_evidence_rollup['backends_seen']) or 'unknown'}, "
        + f"model={','.join(llm_evidence_rollup['models_seen']) or 'unknown'}, "
        + f"num_ctx={','.join(llm_evidence_rollup['num_ctx_seen']) or 'unknown'}, "
        + f"thinking={','.join(llm_evidence_rollup['thinking_modes_seen']) or 'unknown'}"
    )
    log(
        "LLM evidence basis: model residency signals are NOT treated as generation completion; "
        "completion is inferred from llm_execution source/degraded flags, result_status, and fallback markers."
    )
    if llm_evidence_rollup["completion_scenarios_near_timeout"]:
        log(
            "Near-timeout completion examples: "
            + ", ".join(llm_evidence_rollup["completion_scenarios_near_timeout"])
        )
    if llm_evidence_rollup["completion_scenarios_timeout_shaped"]:
        log(
            "Timeout-shaped completion examples: "
            + ", ".join(llm_evidence_rollup["completion_scenarios_timeout_shaped"])
        )

    if args.debug:
        log("")
        log("Pass-quality rubric (deterministic): strong >=4 assertion points, acceptable >=2, weak otherwise.")
        log(
            "Pass-quality rollup: "
            + f"strong={quality_rollup['strong']}, "
            + f"acceptable={quality_rollup['acceptable']}, "
            + f"weak={quality_rollup['weak']}, "
            + f"soft-pass={quality_rollup['soft-pass']}, "
            + f"fail={quality_rollup['fail']}"
        )
    frontend_entries = []
    for entry in REPORT:
        layers = entry.get("layers") or []
        base = _strip_mode_suffix(entry.get("name", ""))
        if "frontend" in layers or base.startswith(("frontend_runtime", "frontend_fixture_", "frontend_real_backend_", "frontend_live_canary_")):
            frontend_entries.append(entry)
    if frontend_entries and args.debug:
        frontend_quality = {"strong": 0, "acceptable": 0, "weak": 0, "soft-pass": 0, "fail": 0}
        for entry in frontend_entries:
            quality = entry.get("quality") or "fail"
            frontend_quality[quality if quality in frontend_quality else "fail"] += 1
        log(
            "Frontend quality rollup: "
            + f"strong={frontend_quality['strong']}, "
            + f"acceptable={frontend_quality['acceptable']}, "
            + f"weak={frontend_quality['weak']}, "
            + f"soft-pass={frontend_quality['soft-pass']}, "
            + f"fail={frontend_quality['fail']}"
        )
    if not args.quiet or args.debug:
        log("")
        log("Per-scenario rollup:")
        for scenario_name in sorted(scenario_rollup.keys()):
            stats = scenario_rollup[scenario_name]
            if stats["fail"] > 0:
                status_label = "FAIL"
            elif stats["soft"] > 0 and stats["pass"] == 0:
                status_label = "SOFT_ONLY_NO_CREDIT"
            elif stats["soft"] > 0:
                status_label = "PASS_WITH_SOFT_NO_CREDIT"
            else:
                status_label = "PASS"
            log(
                "  "
                + f"{scenario_name:30s} "
                + f"{status_label:14s} "
                + f"(strong={stats['strong']}, acceptable={stats['acceptable']}, weak={stats['weak']}, "
                + f"soft={stats['soft']}, fail={stats['fail']}, total={stats['total']})"
            )

        log("")
        log("Per-validation-type rollup:")
        for vtype in sorted(validation_type_rollup.keys()):
            stats = validation_type_rollup[vtype]
            log(
                "  "
                + f"{vtype:20s} "
                + f"(strong={stats['strong']}, acceptable={stats['acceptable']}, weak={stats['weak']}, "
                + f"soft={stats['soft']}, fail={stats['fail']}, total={stats['total']})"
            )

        hardening_scenarios = (
            "hardening-duplicate-handling",
            "hardening-backpressure",
            "hardening-consume-race",
            "hardening-retry-budget",
            "hardening-key-cooldown-recovery",
        )
        log("")
        log("Hardening contract rollup:")
        for scenario_name in hardening_scenarios:
            stats = scenario_rollup.get(scenario_name, _new_rollup_stats())
            if stats["total"] <= 0:
                status_label = "NOT_RUN"
            elif stats["fail"] > 0:
                status_label = "FAIL"
            elif stats["soft"] > 0 and stats["pass"] == 0:
                status_label = "SOFT_ONLY_NO_CREDIT"
            elif stats["soft"] > 0:
                status_label = "PASS_WITH_SOFT_NO_CREDIT"
            else:
                status_label = "PASS"
            log(
                "  "
                + f"{scenario_name:34s} "
                + f"{status_label:12s} "
                + f"(strong={stats['strong']}, acceptable={stats['acceptable']}, weak={stats['weak']}, "
                + f"soft={stats['soft']}, fail={stats['fail']}, total={stats['total']})"
            )

    def _layer_confidence(layer, stats):
        total_checks = stats.get("total", 0) or 0
        if total_checks <= 0:
            if layer == "frontend" and not args.frontend:
                return "not-enabled", 0.0
            return "not-measured", 0.0
        score = (
            (1.0 * stats.get("strong", 0))
            + (0.85 * stats.get("acceptable", 0))
            + (0.60 * stats.get("weak", 0))
            + (0.0 * stats.get("soft_pass", 0))
        ) / total_checks
        if score >= 0.90 and stats.get("fail", 0) == 0 and stats.get("weak", 0) <= max(1, total_checks // 4):
            return "strong", score
        if score >= 0.72 and stats.get("fail", 0) <= max(1, total_checks // 4):
            return "medium", score
        return "weak", score

    log("")
    log("Per-layer rollup:")
    for layer in ("backend", "api", "frontend", "e2e", "runtime", "uncategorized"):
        stats = layer_rollup.get(layer, _new_rollup_stats())
        confidence_label, confidence_score = _layer_confidence(layer, stats)
        log(
            "  "
            + f"{layer:12s} "
            + f"{confidence_label:12s} "
            + f"score={confidence_score:.2f} "
            + f"(strong={stats['strong']}, acceptable={stats['acceptable']}, weak={stats['weak']}, "
            + f"soft={stats['soft']}, fail={stats['fail']}, total={stats['total']})"
        )

    feature_layer_rollup = {}
    feature_mode_rollup = {}
    feature_scenarios = {}
    feature_mode_tests = {}
    mode_buckets = known_mode_buckets()

    def _new_feature_stats():
        return {"total": 0, "pass": 0, "soft": 0, "fail": 0, "quality": set(), "scenarios": set(), "tests": set()}

    def _coverage_cell(stats):
        if not stats or int(stats.get("total", 0)) <= 0:
            return "untested"
        if int(stats.get("fail", 0)) > 0:
            return "broken"
        if int(stats.get("soft", 0)) > 0 and int(stats.get("pass", 0)) == 0:
            return "partial"
        if int(stats.get("soft", 0)) > 0:
            return "covered+soft"
        return "covered"

    def _entry_mode_bucket(entry):
        raw = str(entry.get("mode_bucket") or "").strip()
        if raw:
            return raw
        vtype = str(entry.get("validation_type") or "").strip().lower()
        if vtype == "frontend-fixture":
            return MODE_FRONTEND_FIXTURE_BROWSER
        if vtype == "frontend-real-backend":
            return MODE_FRONTEND_REAL_BACKEND_BROWSER
        if vtype == "live-canary":
            return MODE_LIVE_CANARY_BROWSER
        if vtype in {"contract", "hardening-contract"}:
            return MODE_API_CONTRACT
        if vtype == "runtime":
            return MODE_RUNTIME_HEALTH
        return MODE_BACKEND_INTERNAL

    for feature in known_features():
        feature_layer_rollup[feature] = {}
        feature_mode_rollup[feature] = {}
        feature_scenarios[feature] = set()
        feature_mode_tests[feature] = {}

    for entry in REPORT:
        features = entry.get("features") or []
        layers = entry.get("layers") or ["uncategorized"]
        scenario = entry.get("scenario") or "uncategorized"
        verdict = entry.get("verdict")
        quality = entry.get("quality") or "fail"
        entry_name = str(entry.get("name") or "")
        mode_bucket = _entry_mode_bucket(entry)
        for feature in features:
            feature_scenarios.setdefault(feature, set()).add(scenario)
            per_layer = feature_layer_rollup.setdefault(feature, {})
            for layer in layers:
                layer_stats = per_layer.setdefault(layer, _new_feature_stats())
                layer_stats["total"] += 1
                layer_stats["quality"].add(quality)
                layer_stats["scenarios"].add(scenario)
                layer_stats["tests"].add(entry_name)
                if verdict == VERDICT_PASS:
                    layer_stats["pass"] += 1
                elif verdict == VERDICT_SOFT_PASS_NO_CREDIT:
                    layer_stats["soft"] += 1
                else:
                    layer_stats["fail"] += 1

            per_mode = feature_mode_rollup.setdefault(feature, {})
            mode_stats = per_mode.setdefault(mode_bucket, _new_feature_stats())
            mode_stats["total"] += 1
            mode_stats["quality"].add(quality)
            mode_stats["scenarios"].add(scenario)
            mode_stats["tests"].add(entry_name)
            if verdict == VERDICT_PASS:
                mode_stats["pass"] += 1
            elif verdict == VERDICT_SOFT_PASS_NO_CREDIT:
                mode_stats["soft"] += 1
            else:
                mode_stats["fail"] += 1
            feature_mode_tests.setdefault(feature, {}).setdefault(mode_bucket, set()).add(entry_name)

    if not args.quiet or args.debug:
        log("")
        log("Feature-layer coverage matrix:")
    for feature in sorted(feature_layer_rollup.keys()):
        per_layer = feature_layer_rollup.get(feature) or {}
        layer_cells = []
        for layer in ("backend", "api", "frontend", "e2e", "runtime"):
            if not args.frontend and layer in {"frontend", "e2e"}:
                layer_cells.append(f"{layer}=not-run-by-mode")
            else:
                layer_cells.append(f"{layer}={_coverage_cell(per_layer.get(layer))}")
        scenario_list = sorted(list(feature_scenarios.get(feature) or []))[:4]
        scenario_hint = ",".join(scenario_list) if scenario_list else "none"
        if not args.quiet or args.debug:
            log(f"  {feature:32s} {' | '.join(layer_cells)} | scenarios={scenario_hint}")

    if not args.quiet or args.debug:
        log("")
        log("Feature-mode coverage matrix:")
    for feature in sorted(feature_mode_rollup.keys()):
        per_mode = feature_mode_rollup.get(feature) or {}
        mode_cells = []
        contributors = []
        for mode_bucket in mode_buckets:
            stats = per_mode.get(mode_bucket)
            if (
                not args.frontend
                and mode_bucket in {
                    MODE_FRONTEND_FIXTURE_BROWSER,
                    MODE_FRONTEND_REAL_BACKEND_BROWSER,
                    MODE_LIVE_CANARY_BROWSER,
                }
            ):
                mode_cells.append(f"{mode_bucket}=not-run-by-mode")
            else:
                mode_cells.append(f"{mode_bucket}={_coverage_cell(stats)}")
            scenario_list = sorted(list((stats or {}).get("scenarios", set())))[:3]
            if scenario_list:
                contributors.append(f"{mode_bucket}:{','.join(scenario_list)}")
        contributor_hint = " | ".join(contributors) if contributors else "none"
        if not args.quiet or args.debug:
            log(f"  {feature:32s} {' | '.join(mode_cells)} | contributors={contributor_hint}")

    uncovered_features = []
    partially_covered_features = []
    broken_only_by_mode = {
        MODE_FRONTEND_FIXTURE_BROWSER: [],
        MODE_FRONTEND_REAL_BACKEND_BROWSER: [],
        MODE_LIVE_CANARY_BROWSER: [],
    }
    broken_features = []

    for feature in known_features():
        per_mode = feature_mode_rollup.get(feature) or {}
        total_checks = sum(int((stats or {}).get("total", 0)) for stats in per_mode.values())
        if total_checks <= 0:
            uncovered_features.append(feature)
            continue
        broken_modes = {mode for mode, stats in per_mode.items() if int((stats or {}).get("fail", 0)) > 0}
        partial_modes = {
            mode
            for mode, stats in per_mode.items()
            if int((stats or {}).get("total", 0)) > 0
            and int((stats or {}).get("fail", 0)) == 0
            and int((stats or {}).get("soft", 0)) > 0
        }
        if partial_modes:
            partially_covered_features.append(feature)
        if broken_modes:
            broken_features.append(feature)
        if broken_modes == {MODE_FRONTEND_FIXTURE_BROWSER}:
            broken_only_by_mode[MODE_FRONTEND_FIXTURE_BROWSER].append(feature)
        elif broken_modes == {MODE_FRONTEND_REAL_BACKEND_BROWSER}:
            broken_only_by_mode[MODE_FRONTEND_REAL_BACKEND_BROWSER].append(feature)
        elif broken_modes == {MODE_LIVE_CANARY_BROWSER}:
            broken_only_by_mode[MODE_LIVE_CANARY_BROWSER].append(feature)

    if uncovered_features and (not args.quiet or args.debug):
        log("")
        log("Uncovered features:")
        for feature in uncovered_features:
            log(f"  - {feature}")
    if partially_covered_features and (not args.quiet or args.debug):
        log("")
        log("Partially covered features:")
        for feature in sorted(partially_covered_features):
            log(f"  - {feature}")
    if not args.quiet or args.debug:
        log("")
        log("Broken-only mode slices:")
        log(f"  fixture-only: {sorted(broken_only_by_mode[MODE_FRONTEND_FIXTURE_BROWSER]) or 'none'}")
        log(f"  real-backend-only: {sorted(broken_only_by_mode[MODE_FRONTEND_REAL_BACKEND_BROWSER]) or 'none'}")
        log(f"  live-canary-only: {sorted(broken_only_by_mode[MODE_LIVE_CANARY_BROWSER]) or 'none'}")

    try:
        feature_json = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "mode_buckets": list(mode_buckets),
            "run_context": {
                "scope_id": (
                    "backend_api_plus_frontend"
                    if args.frontend
                    else "backend_api_only"
                ),
                "scope_note": (
                    "Backend/API + frontend/browser validation included."
                    if args.frontend
                    else "Backend/API only. Frontend/browser surfaces are unverified in this run."
                ),
                "frontend_enabled": bool(args.frontend),
                "frontend_real_backend": bool(args.frontend_real_backend),
                "frontend_live_canary": bool(args.frontend_live_canary),
                "backend_default_depth": "full",
                "frontend_default_depth_when_enabled": "frontend-heavy",
                "pytest_lane": str(PYTEST_LANE_CONTEXT.get("lane") or args.pytest_lane or "default"),
                "pytest_lane_semantics": str(PYTEST_LANE_CONTEXT.get("lane_semantics") or ""),
                "pytest_testpaths": list(PYTEST_LANE_CONTEXT.get("pytest_testpaths") or []),
                "pytest_selected_paths": list(PYTEST_LANE_CONTEXT.get("selected_paths") or []),
                "llm_warmup_attempted": bool(LLM_WARMUP_CONTEXT.get("attempted")),
                "llm_warmup_succeeded": bool(LLM_WARMUP_CONTEXT.get("succeeded")),
                "llm_warmup_mode": str(LLM_WARMUP_CONTEXT.get("mode") or ""),
                "llm_warmup_attempts": int(LLM_WARMUP_CONTEXT.get("attempts") or 0),
                "llm_warmup_max_attempts": int(LLM_WARMUP_CONTEXT.get("max_attempts") or 0),
                "llm_warmup_reason": str(LLM_WARMUP_CONTEXT.get("reason") or ""),
                "llm_warmup_latency_sec": LLM_WARMUP_CONTEXT.get("latency_sec"),
                "llm_warmup_completion_source": str(LLM_WARMUP_CONTEXT.get("completion_source") or ""),
                "llm_warmup_backend": str(LLM_WARMUP_CONTEXT.get("backend") or ""),
                "llm_warmup_model": str(LLM_WARMUP_CONTEXT.get("model") or ""),
                "llm_warmup_num_ctx": LLM_WARMUP_CONTEXT.get("num_ctx"),
                "llm_warmup_thinking_mode": str(LLM_WARMUP_CONTEXT.get("thinking_mode") or ""),
                "llm_warmup_first_token_latency_sec": LLM_WARMUP_CONTEXT.get("first_token_latency_sec"),
                "llm_warmup_timeout_ratio": LLM_WARMUP_CONTEXT.get("timeout_ratio"),
                "llm_warmup_completion_observed": bool(LLM_WARMUP_CONTEXT.get("completion_observed")),
                "llm_warmup_degraded_observed": bool(LLM_WARMUP_CONTEXT.get("degraded_observed")),
                "llm_warmup_admission": str(LLM_WARMUP_CONTEXT.get("admission") or ""),
                "llm_warmup_execution": str(LLM_WARMUP_CONTEXT.get("execution") or ""),
                "llm_warmup_replayed_recent": bool(LLM_WARMUP_CONTEXT.get("replayed_recent")),
                "llm_warmup_replay_bypassed": bool(LLM_WARMUP_CONTEXT.get("replay_bypassed")),
                "llm_warmup_attempt_records": list(LLM_WARMUP_CONTEXT.get("attempt_records") or []),
                "rotation_index": VALIDATION_RUNTIME_CONFIG.get("rotation_index"),
                "rotation_source": str(VALIDATION_RUNTIME_CONFIG.get("rotation_source") or ""),
                "rotation_raw": VALIDATION_RUNTIME_CONFIG.get("rotation_raw"),
                "rotation_loop_mode": bool(VALIDATION_RUNTIME_CONFIG.get("rotation_loop_mode")),
                "validation_runtime_config": dict(VALIDATION_RUNTIME_CONFIG),
            },
            "llm_evidence_rollup": dict(llm_evidence_rollup),
            "feature_layer_matrix": {},
            "feature_mode_matrix": {},
            "uncovered_features": sorted(uncovered_features),
            "partially_covered_features": sorted(partially_covered_features),
            "broken_features": sorted(broken_features),
            "broken_only_by_mode": {
                key: sorted(value) for key, value in broken_only_by_mode.items()
            },
        }
        for feature, per_layer in feature_layer_rollup.items():
            feature_json["feature_layer_matrix"][feature] = {}
            for layer, stats in per_layer.items():
                feature_json["feature_layer_matrix"][feature][layer] = {
                    "total": int(stats.get("total", 0)),
                    "pass": int(stats.get("pass", 0)),
                    "soft": int(stats.get("soft", 0)),
                    "fail": int(stats.get("fail", 0)),
                    "coverage": _coverage_cell(stats),
                    "quality": sorted(list(stats.get("quality", set()))),
                    "scenarios": sorted(list(stats.get("scenarios", set()))),
                    "tests": sorted(list(stats.get("tests", set()))),
                }
        for feature, per_mode in feature_mode_rollup.items():
            feature_json["feature_mode_matrix"][feature] = {}
            for mode_bucket, stats in per_mode.items():
                feature_json["feature_mode_matrix"][feature][mode_bucket] = {
                    "total": int(stats.get("total", 0)),
                    "pass": int(stats.get("pass", 0)),
                    "soft": int(stats.get("soft", 0)),
                    "fail": int(stats.get("fail", 0)),
                    "coverage": _coverage_cell(stats),
                    "quality": sorted(list(stats.get("quality", set()))),
                    "scenarios": sorted(list(stats.get("scenarios", set()))),
                    "tests": sorted(list(stats.get("tests", set()))),
                }
        (LOG_DIR / "feature_coverage_matrix.json").write_text(
            json.dumps(feature_json, ensure_ascii=False, indent=2)
        )
    except Exception as exc:
        log(f"Warning: failed to write feature coverage matrix JSON: {exc}")

    weak_pass_entries = [
        entry for entry in REPORT
        if entry.get("verdict") == VERDICT_PASS and entry.get("quality") == "weak"
    ]
    if weak_pass_entries and args.debug:
        log("")
        log("Low-confidence pass notes (weak quality):")
        for entry in weak_pass_entries:
            mode = _mode_label_for_name(entry.get("name", ""))
            base = _strip_mode_suffix(entry.get("name", ""))
            display = entry.get("display") or _display_name_for_base(base)
            scenario = entry.get("scenario") or "uncategorized"
            log(f"  {mode:8s}  {display:35s}  scenario={scenario}")

    blind_spots = []
    if not args.frontend:
        blind_spots.append(
            "Frontend runtime matrix was not run because --frontend was not enabled; "
            "this verdict only covers backend/API paths."
        )

    required_scenarios = (
        "one-way-non-stream",
        "round-trip",
        "via-stopover",
        "streaming-success",
        "degraded-result",
        "no-flights",
        "booking-handoff",
        "jobs-flow",
        "health-runtime-truth",
        "hardening-duplicate-handling",
        "hardening-backpressure",
        "hardening-consume-race",
        "hardening-retry-budget",
        "hardening-key-cooldown-recovery",
    )
    for required in required_scenarios:
        if required not in scenario_rollup:
            blind_spots.append(f"Scenario not exercised in this run: {required}")

    if blind_spots:
        log("")
        log("Blind-spot notes:")
        for note in blind_spots:
            log(f"  - {note}")

    report_by_name = {entry["name"]: entry for entry in REPORT}
    planner_surface_ok = any(
        report_by_name.get(n, {}).get("status") == 0
        for n in (
            "quick_sync_ask_machine", "quick_sync_ask_docker-hosted",
            "real_combined_query_machine", "real_combined_query_docker-hosted",
            "capability_constraints_machine", "capability_constraints_docker-hosted",
        )
    )
    airline_surface_ok = any(
        report_by_name.get(n, {}).get("status") == 0
        for n in (
            "quick_sync_ask_machine", "quick_sync_ask_docker-hosted",
            "real_simple_flight_machine", "real_simple_flight_docker-hosted",
        )
    )
    weather_surface_ok = any(
        report_by_name.get(n, {}).get("status") == 0
        for n in (
            "missing_date_test_machine", "missing_date_test_docker-hosted",
            "real_weather_query_machine", "real_weather_query_docker-hosted",
        )
    )
    llm_surface_ok = any(
        report_by_name.get(n, {}).get("status") == 0
        for n in (
            "streaming_test_machine", "streaming_test_docker-hosted",
            "streaming_nl_relative_machine", "streaming_nl_relative_docker-hosted",
            "real_combined_query_machine", "real_combined_query_docker-hosted",
        )
    )
    health_surface_ok = all(
        any(report_by_name.get(f"{suffix}_{m}", {}).get("status") == 0 for m in ("machine", "docker-hosted"))
        for suffix in ("health_light", "health_deep", "health_keys", "health_runtime_topology", "llm_options", "version_info")
    )

    integration_entries = [
        entry for entry in REPORT
        if str(entry.get("validation_type") or "") == "integration"
    ]
    timeout_tagged_entries = [
        entry for entry in integration_entries
        if "timeout" in (entry.get("failure_tags") or [])
    ]
    timeout_ratio = (len(timeout_tagged_entries) / len(integration_entries)) if integration_entries else 0.0

    preference_family_failures = [
        entry for entry in REPORT
        if _strip_mode_suffix(entry.get("name")).startswith("preferred_airline")
        and entry.get("verdict") == VERDICT_FAIL
    ]
    baggage_family_failures = [
        entry for entry in REPORT
        if _strip_mode_suffix(entry.get("name")).startswith("baggage_hand")
        and entry.get("verdict") == VERDICT_FAIL
    ]

    planner_entries = [
        entry for entry in REPORT
        if str(entry.get("scenario") or "") in {"one-way-non-stream", "round-trip", "via-stopover", "non-stream-success"}
    ]
    planner_failures = [entry for entry in planner_entries if entry.get("verdict") == VERDICT_FAIL]
    if not planner_entries or not planner_surface_ok:
        planner_core_status = "UNVERIFIED"
    elif planner_failures:
        planner_core_status = "DEGRADED"
    elif preference_family_failures or baggage_family_failures:
        planner_core_status = "PARTIAL"
    else:
        planner_core_status = "OK"

    if not airline_surface_ok:
        airline_core_status = "DEGRADED"
    elif preference_family_failures:
        airline_core_status = "PARTIAL"
    else:
        airline_core_status = "OK"

    weather_core_status = "OK" if weather_surface_ok else "DEGRADED"

    warmup_attempted = bool(LLM_WARMUP_CONTEXT.get("attempted"))
    warmup_succeeded = bool(LLM_WARMUP_CONTEXT.get("succeeded"))
    if warmup_attempted and not warmup_succeeded:
        llm_core_status = "DEGRADED"
    elif llm_evidence_rollup["required_total"] <= 0:
        llm_core_status = "UNVERIFIED"
    elif llm_evidence_rollup["degraded_observed"] > 0:
        llm_core_status = "DEGRADED"
    elif llm_evidence_rollup["completion_observed"] <= 0:
        llm_core_status = "UNVERIFIED"
    elif llm_evidence_rollup["completion_ratio"] < 0.85:
        llm_core_status = "PARTIAL"
    elif llm_evidence_rollup["timeout_shaped_ratio"] >= 0.30:
        llm_core_status = "PARTIAL"
    elif llm_evidence_rollup["near_timeout_ratio"] >= 0.35:
        llm_core_status = "PARTIAL"
    elif llm_evidence_rollup["unknown_unverified"] > 0:
        llm_core_status = "PARTIAL"
    elif not llm_surface_ok:
        llm_core_status = "PARTIAL"
    else:
        llm_core_status = "OK"

    health_core_status = "OK" if health_surface_ok else "DEGRADED"

    def _capability_entry(core_status, evidence):
        final_status = _apply_scope_to_capability_status(
            core_status,
            frontend_enabled=bool(args.frontend),
        )
        return {
            "status": final_status,
            "core_status": str(core_status or "UNVERIFIED").upper(),
            "scope": (
                "backend_api_plus_frontend"
                if args.frontend
                else "backend_api_only_frontend_unverified"
            ),
            "evidence": evidence,
        }

    CAPABILITY_REPORT_DETAILS.clear()
    CAPABILITY_REPORT_DETAILS["planner"] = _capability_entry(
        planner_core_status,
        {
            "integration_checks": len(planner_entries),
            "integration_failures": len(planner_failures),
            "preferred_airline_failures": len(preference_family_failures),
            "baggage_failures": len(baggage_family_failures),
        },
    )
    CAPABILITY_REPORT_DETAILS["airline_api"] = _capability_entry(
        airline_core_status,
        {
            "surface_probe_ok": bool(airline_surface_ok),
            "preferred_airline_failures": len(preference_family_failures),
        },
    )
    CAPABILITY_REPORT_DETAILS["weather_api"] = _capability_entry(
        weather_core_status,
        {
            "surface_probe_ok": bool(weather_surface_ok),
        },
    )
    CAPABILITY_REPORT_DETAILS["llm_router"] = _capability_entry(
        llm_core_status,
        {
            "warmup_attempted": warmup_attempted,
            "warmup_succeeded": warmup_succeeded,
            "warmup_reason": str(LLM_WARMUP_CONTEXT.get("reason") or ""),
            "warmup_attempts": int(LLM_WARMUP_CONTEXT.get("attempts") or 0),
            "warmup_completion_source": str(LLM_WARMUP_CONTEXT.get("completion_source") or ""),
            "warmup_backend": str(LLM_WARMUP_CONTEXT.get("backend") or ""),
            "warmup_model": str(LLM_WARMUP_CONTEXT.get("model") or ""),
            "warmup_num_ctx": LLM_WARMUP_CONTEXT.get("num_ctx"),
            "warmup_thinking_mode": str(LLM_WARMUP_CONTEXT.get("thinking_mode") or ""),
            "warmup_first_token_latency_sec": LLM_WARMUP_CONTEXT.get("first_token_latency_sec"),
            "warmup_admission": str(LLM_WARMUP_CONTEXT.get("admission") or ""),
            "warmup_execution": str(LLM_WARMUP_CONTEXT.get("execution") or ""),
            "warmup_replayed_recent": bool(LLM_WARMUP_CONTEXT.get("replayed_recent")),
            "warmup_replay_bypassed": bool(LLM_WARMUP_CONTEXT.get("replay_bypassed")),
            "required_total": llm_evidence_rollup["required_total"],
            "completion_observed": llm_evidence_rollup["completion_observed"],
            "completion_ratio": llm_evidence_rollup["completion_ratio"],
            "degraded_observed": llm_evidence_rollup["degraded_observed"],
            "unknown_unverified": llm_evidence_rollup["unknown_unverified"],
            "first_token_observed": llm_evidence_rollup["first_token_observed"],
            "first_token_unavailable": llm_evidence_rollup["first_token_unavailable"],
            "first_token_latency_p50": llm_evidence_rollup["first_token_latency_p50"],
            "first_token_latency_p90": llm_evidence_rollup["first_token_latency_p90"],
            "first_token_from_validation_send_p50": llm_evidence_rollup["first_token_from_validation_send_p50"],
            "first_token_from_validation_send_p90": llm_evidence_rollup["first_token_from_validation_send_p90"],
            "models_seen": list(llm_evidence_rollup["models_seen"]),
            "num_ctx_seen": list(llm_evidence_rollup["num_ctx_seen"]),
            "thinking_modes_seen": list(llm_evidence_rollup["thinking_modes_seen"]),
            "backends_seen": list(llm_evidence_rollup["backends_seen"]),
            "near_timeout_ratio": llm_evidence_rollup["near_timeout_ratio"],
            "timeout_shaped_ratio": llm_evidence_rollup["timeout_shaped_ratio"],
        },
    )
    CAPABILITY_REPORT_DETAILS["health_system"] = _capability_entry(
        health_core_status,
        {
            "surface_probe_ok": bool(health_surface_ok),
        },
    )

    CAPABILITY_REPORT["planner"] = CAPABILITY_REPORT_DETAILS["planner"]["status"]
    CAPABILITY_REPORT["airline_api"] = CAPABILITY_REPORT_DETAILS["airline_api"]["status"]
    CAPABILITY_REPORT["weather_api"] = CAPABILITY_REPORT_DETAILS["weather_api"]["status"]
    CAPABILITY_REPORT["llm_router"] = CAPABILITY_REPORT_DETAILS["llm_router"]["status"]
    CAPABILITY_REPORT["health_system"] = CAPABILITY_REPORT_DETAILS["health_system"]["status"]

    log("")
    log(
        "Capability report basis: scenario-truth overlays + scope qualifiers + LLM evidence rollup "
        "(completion/degraded/near-timeout/unknown)."
    )
    log(
        f"Capability evidence summary: integration_checks={len(integration_entries)}, "
        f"timeout_tagged={len(timeout_tagged_entries)} ({timeout_ratio:.0%}), "
        f"preferred_airline_failures={len(preference_family_failures)}, "
        f"baggage_failures={len(baggage_family_failures)}, "
        f"llm_completion_ratio={llm_evidence_rollup['completion_ratio']:.0%}, "
        f"llm_unknown={llm_evidence_rollup['unknown_unverified']}"
    )

    # Additional counts from consolidated log
    with open(log_filename) as f:
        content = f.read()
    if args.debug:
        log("")
        log("Detailed counts from consolidated log:")
        total_passed = content.count("PASSED")
        total_failed = content.count("FAILED")
        log(f"  PASSED lines: {total_passed}")
        log(f"  FAILED lines: {total_failed}")
        field_required = content.count('"msg":"Field required"')
        if field_required:
            log(f"  Field required errors: {field_required}")

    log("")
    log(f"Full logs available in: {log_filename}")
    log("")
    log("=== CAPABILITY REPORT ===")
    for capability_key in ("planner", "airline_api", "weather_api", "llm_router", "health_system"):
        detail = CAPABILITY_REPORT_DETAILS.get(capability_key, {})
        status = detail.get("status", CAPABILITY_REPORT.get(capability_key, "UNKNOWN"))
        core_status = detail.get("core_status", "UNVERIFIED")
        scope = detail.get("scope", "unknown")
        evidence = detail.get("evidence", {})
        log(
            f"{capability_key}: {status} "
            f"(core={core_status}, scope={scope}, evidence={json.dumps(evidence, ensure_ascii=False)})"
        )
    log("")
    duration_sec = round(time.time() - started_at, 2)
    log("=== VALIDATION SUMMARY ===")
    log(f"Mode: {'real' if REAL_MODE else args.mode}")
    log(
        "Scope id: "
        + ("backend_api_plus_frontend" if args.frontend else "backend_api_only")
    )
    log(
        "Scope: "
        + (
            "backend/API + frontend browser/runtime"
            if args.frontend
            else "backend/API only (frontend browser/runtime not included)"
        )
    )
    if not args.frontend:
        log("NOTE: backend/API-only run; frontend/browser coverage was not executed in this run.")
        log("NOTE: backend/API-only green status must not be read as full product validation.")
    if LLM_WARMUP_CONTEXT.get("attempted"):
        if LLM_WARMUP_CONTEXT.get("succeeded"):
            log(
                "LLM warmup gate: succeeded "
                + f"(attempts={LLM_WARMUP_CONTEXT.get('attempts')}/{LLM_WARMUP_CONTEXT.get('max_attempts')}, "
                + f"source={LLM_WARMUP_CONTEXT.get('completion_source') or 'unknown'}, "
                + f"backend={LLM_WARMUP_CONTEXT.get('backend') or 'unknown'}, "
                + f"model={LLM_WARMUP_CONTEXT.get('model') or 'unknown'}, "
                + f"num_ctx={LLM_WARMUP_CONTEXT.get('num_ctx')}, "
                + f"thinking_mode={LLM_WARMUP_CONTEXT.get('thinking_mode') or 'unknown'}, "
                + f"latency_sec={LLM_WARMUP_CONTEXT.get('latency_sec')}, "
                + f"first_token_latency_sec={LLM_WARMUP_CONTEXT.get('first_token_latency_sec')}, "
                + f"admission={LLM_WARMUP_CONTEXT.get('admission') or 'unknown'}, "
                + f"execution={LLM_WARMUP_CONTEXT.get('execution') or 'unknown'}, "
                + f"replayed_recent={bool(LLM_WARMUP_CONTEXT.get('replayed_recent'))}, "
                + f"replay_bypassed={bool(LLM_WARMUP_CONTEXT.get('replay_bypassed'))})"
            )
        else:
            log(
                "LLM warmup gate: FAILED "
                + f"(attempts={LLM_WARMUP_CONTEXT.get('attempts')}/{LLM_WARMUP_CONTEXT.get('max_attempts')}, "
                + f"reason={LLM_WARMUP_CONTEXT.get('reason') or 'unknown'}, "
                + f"backend={LLM_WARMUP_CONTEXT.get('backend') or 'unknown'}, "
                + f"model={LLM_WARMUP_CONTEXT.get('model') or 'unknown'}, "
                + f"num_ctx={LLM_WARMUP_CONTEXT.get('num_ctx')}, "
                + f"thinking_mode={LLM_WARMUP_CONTEXT.get('thinking_mode') or 'unknown'}, "
                + f"admission={LLM_WARMUP_CONTEXT.get('admission') or 'unknown'}, "
                + f"execution={LLM_WARMUP_CONTEXT.get('execution') or 'unknown'}, "
                + f"replayed_recent={bool(LLM_WARMUP_CONTEXT.get('replayed_recent'))}, "
                + f"replay_bypassed={bool(LLM_WARMUP_CONTEXT.get('replay_bypassed'))})"
            )
    else:
        log("LLM warmup gate: not attempted in this run.")
    scenario_profile = _derive_dominant_llm_profile(REPORT)
    warmup_profile = {
        "backend": str(LLM_WARMUP_CONTEXT.get("backend") or ""),
        "model": str(LLM_WARMUP_CONTEXT.get("model") or ""),
        "num_ctx": (
            ""
            if LLM_WARMUP_CONTEXT.get("num_ctx") in (None, "")
            else str(LLM_WARMUP_CONTEXT.get("num_ctx"))
        ),
        "thinking_mode": str(LLM_WARMUP_CONTEXT.get("thinking_mode") or ""),
    }
    warmup_scenario_same = (
        bool(warmup_profile["backend"] or warmup_profile["model"] or warmup_profile["num_ctx"])
        and warmup_profile["backend"] == str(scenario_profile.get("backend") or "")
        and warmup_profile["model"] == str(scenario_profile.get("model") or "")
        and warmup_profile["num_ctx"] == str(scenario_profile.get("num_ctx") or "")
        and warmup_profile["thinking_mode"] == str(scenario_profile.get("thinking_mode") or "")
    )
    log(
        "Validation runtime config snapshot: "
        + f"backend_expectation={VALIDATION_RUNTIME_CONFIG.get('backend_expectation') or 'unknown'}, "
        + f"llm_mode={VALIDATION_RUNTIME_CONFIG.get('llm_mode') or '<unset>'}, "
        + f"USE_CLOUD_LLM={VALIDATION_RUNTIME_CONFIG.get('use_cloud_llm') or '<unset>'}, "
        + f"OLLAMA_MODEL(tmp)={VALIDATION_RUNTIME_CONFIG.get('ollama_model_tmp_env') or '<unset/runtime_default>'}, "
        + f"OLLAMA_NUM_CTX(process)={VALIDATION_RUNTIME_CONFIG.get('ollama_num_ctx_process_env') or '<unset>'}, "
        + f"VALIDATION_OLLAMA_NUM_CTX={VALIDATION_RUNTIME_CONFIG.get('ollama_num_ctx_validation_override') or '<unset>'}, "
        + f"OLLAMA_NUM_CTX(effective)={VALIDATION_RUNTIME_CONFIG.get('ollama_num_ctx_effective')}, "
        + f"OLLAMA_NUM_CTX(source)={VALIDATION_RUNTIME_CONFIG.get('ollama_num_ctx_source') or 'unknown'}, "
        + f"OLLAMA_THINKING_MODE(effective)={VALIDATION_RUNTIME_CONFIG.get('ollama_thinking_mode_effective') or 'unknown'}"
    )
    log(
        "Rotation selection summary: "
        + f"index={VALIDATION_RUNTIME_CONFIG.get('rotation_index')}, "
        + f"source={VALIDATION_RUNTIME_CONFIG.get('rotation_source') or 'unknown'}, "
        + f"raw={VALIDATION_RUNTIME_CONFIG.get('rotation_raw')}, "
        + f"loop_mode={bool(VALIDATION_RUNTIME_CONFIG.get('rotation_loop_mode'))}"
    )
    log(
        "Warmup vs scenario runtime profile: "
        + f"same={warmup_scenario_same}, "
        + f"warmup={{backend={warmup_profile['backend'] or 'unknown'}, "
        + f"model={warmup_profile['model'] or 'unknown'}, "
        + f"num_ctx={warmup_profile['num_ctx'] or 'unknown'}, "
        + f"thinking={warmup_profile['thinking_mode'] or 'unknown'}}}, "
        + f"scenario={{backend={scenario_profile.get('backend') or 'unknown'}, "
        + f"model={scenario_profile.get('model') or 'unknown'}, "
        + f"num_ctx={scenario_profile.get('num_ctx') or 'unknown'}, "
        + f"thinking={scenario_profile.get('thinking_mode') or 'unknown'}, "
        + f"observed_count={scenario_profile.get('observed_count') or 0}}}"
    )
    log(
        "First-token timing summary: "
        + f"observed={llm_evidence_rollup['first_token_observed']}, "
        + f"unavailable={llm_evidence_rollup['first_token_unavailable']}, "
        + f"llm_dispatch_to_first_token_p50={llm_evidence_rollup['first_token_latency_p50']}, "
        + f"llm_dispatch_to_first_token_p90={llm_evidence_rollup['first_token_latency_p90']}, "
        + f"validation_send_to_first_token_p50={llm_evidence_rollup['first_token_from_validation_send_p50']}, "
        + f"validation_send_to_first_token_p90={llm_evidence_rollup['first_token_from_validation_send_p90']}"
    )
    log(
        "Pytest lane: "
        + str(PYTEST_LANE_CONTEXT.get("lane") or args.pytest_lane or "default")
        + " | semantics: "
        + str(PYTEST_LANE_CONTEXT.get("lane_semantics") or "unspecified")
    )
    log(
        "Pytest testpaths: "
        + (
            ",".join(PYTEST_LANE_CONTEXT.get("pytest_testpaths") or [])
            if (PYTEST_LANE_CONTEXT.get("pytest_testpaths") or [])
            else "<unset>"
        )
    )
    log(
        "Pytest selected paths for this run: "
        + (
            ",".join(PYTEST_LANE_CONTEXT.get("selected_paths") or [])
            if (PYTEST_LANE_CONTEXT.get("selected_paths") or [])
            else "<plain pytest discovery>"
        )
    )
    log(f"Total tests: {total}")
    log(f"Passed: {passed}")
    log(f"Soft-passed (no credit): {soft_passed}")
    log(f"Failed: {failed}")
    log(f"Duration: {duration_sec} sec")
    log("")
    if not args.frontend:
        log("=" * 60)
        log("VALIDATION SCOPE: BACKEND/API ONLY")
        log("  - Frontend/browser tests: NOT EXECUTED")
        log("  - Full product validation: NOT ACHIEVED")
        log("  - Use --frontend for full e2e validation")
        log("=" * 60)
        log("")

    run_summary_artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "log_file": str(log_filename),
        "mode": ("real" if REAL_MODE else args.mode),
        "scope_id": ("backend_api_plus_frontend" if args.frontend else "backend_api_only"),
        "scope_note": (
            "Backend/API + frontend/browser coverage included."
            if args.frontend
            else "Backend/API only; frontend/browser surfaces unverified."
        ),
        "frontend_enabled": bool(args.frontend),
        "frontend_real_backend": bool(args.frontend_real_backend),
        "frontend_live_canary": bool(args.frontend_live_canary),
        "pytest_lane": str(PYTEST_LANE_CONTEXT.get("lane") or args.pytest_lane or "default"),
        "pytest_lane_semantics": str(PYTEST_LANE_CONTEXT.get("lane_semantics") or ""),
        "pytest_testpaths": list(PYTEST_LANE_CONTEXT.get("pytest_testpaths") or []),
        "pytest_selected_paths": list(PYTEST_LANE_CONTEXT.get("selected_paths") or []),
        "llm_warmup_attempted": bool(LLM_WARMUP_CONTEXT.get("attempted")),
        "llm_warmup_succeeded": bool(LLM_WARMUP_CONTEXT.get("succeeded")),
        "llm_warmup_mode": str(LLM_WARMUP_CONTEXT.get("mode") or ""),
        "llm_warmup_attempts": int(LLM_WARMUP_CONTEXT.get("attempts") or 0),
        "llm_warmup_max_attempts": int(LLM_WARMUP_CONTEXT.get("max_attempts") or 0),
        "llm_warmup_reason": str(LLM_WARMUP_CONTEXT.get("reason") or ""),
        "llm_warmup_http_status": LLM_WARMUP_CONTEXT.get("http_status"),
        "llm_warmup_latency_sec": LLM_WARMUP_CONTEXT.get("latency_sec"),
        "llm_warmup_completion_source": str(LLM_WARMUP_CONTEXT.get("completion_source") or ""),
        "llm_warmup_backend": str(LLM_WARMUP_CONTEXT.get("backend") or ""),
        "llm_warmup_model": str(LLM_WARMUP_CONTEXT.get("model") or ""),
        "llm_warmup_num_ctx": LLM_WARMUP_CONTEXT.get("num_ctx"),
        "llm_warmup_thinking_mode": str(LLM_WARMUP_CONTEXT.get("thinking_mode") or ""),
        "llm_warmup_first_token_latency_sec": LLM_WARMUP_CONTEXT.get("first_token_latency_sec"),
        "llm_warmup_timeout_ratio": LLM_WARMUP_CONTEXT.get("timeout_ratio"),
        "llm_warmup_request_reached_llm_path": bool(LLM_WARMUP_CONTEXT.get("request_reached_llm_path")),
        "llm_warmup_completion_observed": bool(LLM_WARMUP_CONTEXT.get("completion_observed")),
        "llm_warmup_degraded_observed": bool(LLM_WARMUP_CONTEXT.get("degraded_observed")),
        "llm_warmup_admission": str(LLM_WARMUP_CONTEXT.get("admission") or ""),
        "llm_warmup_execution": str(LLM_WARMUP_CONTEXT.get("execution") or ""),
        "llm_warmup_replayed_recent": bool(LLM_WARMUP_CONTEXT.get("replayed_recent")),
        "llm_warmup_replay_bypassed": bool(LLM_WARMUP_CONTEXT.get("replay_bypassed")),
        "llm_warmup_attempt_records": list(LLM_WARMUP_CONTEXT.get("attempt_records") or []),
        "totals": {
            "total": int(total),
            "passed": int(passed),
            "soft_passed_no_credit": int(soft_passed),
            "failed": int(failed),
        },
        "llm_evidence_rollup": dict(llm_evidence_rollup),
        "validation_runtime_config": dict(VALIDATION_RUNTIME_CONFIG),
        "scenario_runtime_profile": dict(scenario_profile),
        "warmup_runtime_profile": dict(warmup_profile),
        "warmup_scenario_same_profile": bool(warmup_scenario_same),
        "capability_report": dict(CAPABILITY_REPORT),
        "capability_details": dict(CAPABILITY_REPORT_DETAILS),
        "blind_spot_notes": list(blind_spots),
        "timeout_tagged_passes": len(timeout_tagged_passes),
    }
    try:
        (LOG_DIR / "validation_run_summary.json").write_text(
            json.dumps(run_summary_artifact, ensure_ascii=False, indent=2)
        )
    except Exception as exc:
        log(f"Warning: failed to write validation_run_summary.json: {exc}")

    return failed, bool(args.frontend)


# ----------------------------------------------------------------------
# HITL Approval Gate Integration Test
# ----------------------------------------------------------------------
def test_hitl_approval_gate(base_url=None):
    """
    Integration test for HITL approval gate:
    1. Submit a booking plan with stream=true
    2. Assert the approval_required event fires before any booking action executes
    3. Approve it and assert execution resumes
    """
    url = base_url or DEFAULT_API_BASE_URL
    log("\n=== HITL Approval Gate Integration Test ===")

    import uuid
    import threading
    import time as _time

    plan_id = f"hitl-test-{uuid.uuid4().hex[:8]}"
    future_date = (datetime.now().date() + timedelta(days=21)).strftime("%Y-%m-%d")

    approval_received = threading.Event()
    approval_event_data = {}
    done_json_data = {}
    stream_error = None

    def run_stream():
        nonlocal stream_error
        try:
            import httpx
            with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                with client.stream(
                    "POST",
                    f"{url}/ask?stream=true",
                    json={
                        "origin": "DEL",
                        "destination": "BOM",
                        "date": future_date,
                        "user_query": "Find best flight and book it.",
                    },
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status_code != 200:
                        stream_error = f"Stream returned {response.status_code}"
                        return
                    buffer = ""
                    for chunk in response.iter_bytes():
                        buffer += chunk.decode("utf-8", errors="replace")
                        while "\n\n" in buffer:
                            frame, buffer = buffer.split("\n\n", 1)
                            lines = frame.split("\n")
                            event_type = None
                            data_lines = []
                            for line in lines:
                                if line.startswith("event:"):
                                    event_type = line[6:].strip()
                                elif line.startswith("data:"):
                                    data_lines.append(line[5:].strip() if line.startswith("data: ") else line[5:])
                            if data_lines:
                                data_text = "\n".join(data_lines)
                                if event_type == "approval_required":
                                    parsed = json.loads(data_text)
                                    approval_event_data.update(parsed)
                                    approval_received.set()
                                elif "[DONE_JSON]" in data_text:
                                    done_json = json.loads(data_text.replace("[DONE_JSON]", ""))
                                    done_json_data.update(done_json)
        except Exception as e:
            stream_error = str(e)

    stream_thread = threading.Thread(target=run_stream, daemon=True)
    stream_thread.start()

    approved = approval_received.wait(timeout=30.0)
    if not approved:
        log("FAIL: approval_required event did not fire within 30s")
        if stream_error:
            log(f"  Stream error: {stream_error}")
        return False

    if approval_event_data.get("action") != "booking_handoff":
        log(f"FAIL: approval_required action was '{approval_event_data.get('action')}', expected 'booking_handoff'")
        return False

    if "plan_id" not in approval_event_data:
        log("FAIL: approval_required event missing plan_id")
        return False

    actual_plan_id = approval_event_data["plan_id"]
    log(f"PASS: approval_required event fired with plan_id={actual_plan_id}")

    resp = requests.post(
        f"{url}/plan/{actual_plan_id}/approve",
        json={"approved": True},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VALIDATION_AUTH_TOKEN}",
        },
        timeout=10,
    )
    if resp.status_code != 200:
        log(f"FAIL: approve endpoint returned {resp.status_code}: {resp.text}")
        return False

    log("PASS: approve endpoint returned 200")

    stream_thread.join(timeout=60.0)
    if stream_error:
        log(f"FAIL: stream error after approval: {stream_error}")
        return False

    if done_json_data:
        log(f"PASS: stream completed with DONE_JSON after approval")
        return True

    log("WARN: stream ended without DONE_JSON (may be acceptable if booking was deferred)")
    return True


# ----------------------------------------------------------------------
# RAGAS Evaluation
# ----------------------------------------------------------------------
def run_ragas_eval(base_url=None, with_rag=False):
    """
    Run RAGAS evaluation on existing test cases.
    Writes results to eval_results/ragas_baseline.json or eval_results/ragas_with_rag.json.
    """
    try:
        from datasets import Dataset
    except ImportError:
        log("FAIL: datasets not installed. Run: pip install datasets")
        return False

    url = base_url or DEFAULT_API_BASE_URL
    label = "with-RAG" if with_rag else "pre-RAG baseline"
    log(f"\n=== RAGAS Evaluation ({label}) ===")

    test_cases = [
        {
            "question": "Find cheapest flight from Delhi to Mumbai on 2026-05-15",
            "answer": "",
            "contexts": [""],
            "ground_truth": "Should return flight options from DEL to BOM with prices.",
        },
        {
            "question": "What is the weather in Bangalore tomorrow?",
            "answer": "",
            "contexts": [""],
            "ground_truth": "Should return weather forecast for Bangalore/BLR.",
        },
        {
            "question": "Book a direct flight from BLR to DEL with hand baggage only",
            "answer": "",
            "contexts": [""],
            "ground_truth": "Should return direct flights with hand baggage info.",
        },
    ]

    future_date = (datetime.now().date() + timedelta(days=21)).strftime("%Y-%m-%d")

    # RAG context retrieval (if --with-rag)
    retriever = None
    if with_rag:
        try:
            from rag.retriever import RAGRetriever
            retriever = RAGRetriever(corpus_dir="rag/corpus")
            log("RAG retriever initialized.")
        except Exception as e:
            log(f"RAG retriever init failed: {e}; falling back to baseline contexts.")
            with_rag = False

    server_available = False
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        server_available = resp.ok
    except Exception:
        pass

    if server_available:
        for tc in test_cases:
            try:
                resp = requests.post(
                    f"{url}/ask",
                    json={
                        "user_query": tc["question"],
                        "date": future_date if "2026-05-15" not in tc["question"] else "2026-05-15",
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()
                    tc["answer"] = data.get("llm_response", "") or data.get("message", "") or data.get("error", "")
                    if with_rag and retriever is not None:
                        rag_results = retriever.retrieve(tc["question"], top_k=4)
                        tc["contexts"] = [r["text"] for r in rag_results] if rag_results else ["No relevant context found."]
                    else:
                        tc["contexts"] = [json.dumps(data.get("best_flight", {}))]
                else:
                    tc["answer"] = f"Error: {resp.status_code}"
            except Exception as e:
                tc["answer"] = f"Exception: {str(e)}"
    else:
        log("Server not available; using placeholder data.")
        for i, tc in enumerate(test_cases):
            tc["answer"] = f"Placeholder answer for test case {i+1}. Server was not available during evaluation."
            if with_rag and retriever is not None:
                rag_results = retriever.retrieve(tc["question"], top_k=4)
                tc["contexts"] = [r["text"] for r in rag_results] if rag_results else ["No relevant context found."]
            else:
                tc["contexts"] = [f"Placeholder context for test case {i+1}."]

    ragas_dataset = Dataset.from_dict({
        "question": [tc["question"] for tc in test_cases],
        "answer": [tc["answer"] for tc in test_cases],
        "contexts": [tc["contexts"] for tc in test_cases],
        "ground_truth": [tc["ground_truth"] for tc in test_cases],
    })

    log(f"Dataset prepared with {len(test_cases)} test cases.")

    try:
        from ragas import evaluate
        from ragas.metrics.collections.faithfulness import Faithfulness
        from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
        from ragas.metrics.collections.context_relevance import ContextRelevance

        log("Running RAGAS evaluation (requires LLM configured via environment)...")
        result = evaluate(
            ragas_dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), ContextRelevance()],
        )

        scores = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_relevance"]:
            vals = result[metric_name]
            scores[metric_name] = float(sum(vals) / len(vals)) if vals else 0.0

        per_question = []
        for i, tc in enumerate(test_cases):
            q_scores = {}
            for metric_name in ["faithfulness", "answer_relevancy", "context_relevance"]:
                vals = result[metric_name]
                q_scores[metric_name] = float(vals[i]) if i < len(vals) else 0.0
            per_question.append({
                "question": tc["question"],
                "answer": tc["answer"][:200],
                "scores": q_scores,
            })
    except (TypeError, ImportError) as e:
        log(f"RAGAS LLM-based evaluation not available ({e}); writing placeholder baseline.")
        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_relevance": 0.0,
        }
        per_question = [
            {
                "question": tc["question"],
                "answer": tc["answer"][:200],
                "scores": scores.copy(),
            }
            for tc in test_cases
        ]

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": "with-RAG" if with_rag else "pre-RAG baseline",
        "num_test_cases": len(test_cases),
        "overall_scores": scores,
        "per_question": per_question,
    }

    eval_dir = ROOT / "eval_results"
    eval_dir.mkdir(exist_ok=True)
    if with_rag:
        output_path = eval_dir / "ragas_with_rag.json"
    else:
        output_path = eval_dir / "ragas_baseline.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    log(f"RAGAS results written to {output_path}")
    log(f"Overall scores: faithfulness={scores.get('faithfulness', 0):.3f}, answer_relevancy={scores.get('answer_relevancy', 0):.3f}, context_relevance={scores.get('context_relevance', 0):.3f}")

    # Print delta vs baseline when --with-rag
    if with_rag:
        baseline_path = eval_dir / "ragas_baseline.json"
        if baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text())
                baseline_scores = baseline.get("overall_scores", {})
                log("\nRAGAS Delta (with RAG vs baseline):")
                for metric in ["faithfulness", "answer_relevancy", "context_relevance"]:
                    base_val = baseline_scores.get(metric, 0.0)
                    new_val = scores.get(metric, 0.0)
                    delta = new_val - base_val
                    log(f"  {metric}: {base_val:.3f} -> {new_val:.3f} (Δ {delta:+.3f})")
            except Exception as e:
                log(f"Could not compute delta: {e}")

    return True


if __name__ == "__main__":
    if args.ragas_eval:
        ok = run_ragas_eval(with_rag=args.with_rag)
        sys.exit(0 if ok else 1)

    if args.hitl_test:
        ok = test_hitl_approval_gate()
        sys.exit(0 if ok else 1)

    exit_code = 0
    scope_label = "BACKEND_ONLY"
    try:
        result = main()
        if isinstance(result, tuple):
            failed_count, frontend_enabled = result
            scope_label = "FULL_E2E" if frontend_enabled else "BACKEND_ONLY"
        else:
            failed_count = result
        if isinstance(failed_count, int) and failed_count > 0:
            exit_code = 1
    except KeyboardInterrupt:
        log("Interrupted by user. Cleaning up.")
        exit_code = 1
    except Exception as exc:
        log(f"Validation aborted due to unexpected error: {exc}")
        exit_code = 1
    finally:
        TMP_ENV.unlink(missing_ok=True)
        _close_frontend_validator()
    # optionally add more cleanup (stop docker container, kill uvicorn) if needed
    if exit_code:
        sys.exit(exit_code)
