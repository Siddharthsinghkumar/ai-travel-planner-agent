from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
from urllib.parse import urlencode

import requests
from validation.frontend_contract import FrontendValidationRequest, coerce_frontend_validation_request
from validation.scenario_catalog import (
    classify_frontend_endpoint_request,
    frontend_fixture_scenarios,
    resolve_frontend_fixture_scenario_name,
)


class FrontendValidationError(RuntimeError):
    """Raised when frontend validation cannot execute safely."""


class FrontendValidator:
    """
    Minimal browser-driven validator for the existing Vite frontend.
    Uses simple DOM checks to validate visible output for a query.
    """

    def __init__(
        self,
        *,
        frontend_url: str,
        frontend_dir: Path,
        frontend_host: str = "127.0.0.1",
        frontend_port: int = 5173,
        frontend_server_mode: str = "dev",
        startup_timeout_s: int = 45,
        query_timeout_s: int = 45,
        auto_start_frontend: bool = True,
        fixture_mode_default: bool = True,
        allow_real_backend: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.frontend_url = frontend_url.rstrip("/")
        self.frontend_dir = Path(frontend_dir)
        self.frontend_host = frontend_host
        self.frontend_port = frontend_port
        mode = (frontend_server_mode or "dev").strip().lower()
        self.frontend_server_mode = "preview" if mode == "preview" else "dev"
        self.startup_timeout_s = startup_timeout_s
        self.query_timeout_s = query_timeout_s
        self.auto_start_frontend = auto_start_frontend
        self.fixture_mode_default = bool(fixture_mode_default)
        self.allow_real_backend = bool(allow_real_backend)
        self.log = log_fn or (lambda _msg: None)
        self.debug = (os.getenv("FRONTEND_VALIDATION_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"})
        self.query_hard_cap_s = max(10, int(os.getenv("FRONTEND_VALIDATION_HARD_CAP_S", "120")))
        self.fixture_catalog = frontend_fixture_scenarios()

        self._frontend_proc: Optional[subprocess.Popen] = None
        self._frontend_started_by_validator = False
        self._frontend_reused_external = False
        self._frontend_pid: Optional[int] = None
        self._playwright = None
        self._browser = None
        self._active_pages = 0

    def start(self) -> None:
        if self._browser is not None:
            self._debug("browser_reuse=true")
            return

        os.environ.setdefault(
            "PLAYWRIGHT_BROWSERS_PATH",
            str(self.frontend_dir.parent / ".playwright-browsers"),
        )

        if not self._is_frontend_up():
            if not self.auto_start_frontend:
                raise FrontendValidationError(
                    f"Frontend is not reachable at {self.frontend_url} and auto-start is disabled."
                )
            self._start_frontend_server()
            self._wait_for_frontend_ready()
        else:
            self._frontend_reused_external = True
            self._debug(f"frontend_server_reused_external=true mode={self.frontend_server_mode}")
            self.log(f"[frontend-validator] frontend_server_reused=true mode={self.frontend_server_mode}")

        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise FrontendValidationError(
                "Playwright is required for --frontend mode. Install with "
                "`pip install playwright` and run `playwright install chromium`."
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._debug("browser_started=true")

    def close(self) -> None:
        cleanup_ok = True

        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                cleanup_ok = False
            finally:
                self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                cleanup_ok = False
            finally:
                self._playwright = None

        if self._frontend_proc is not None:
            if self._frontend_started_by_validator:
                stopped = self._stop_frontend_server()
                cleanup_ok = cleanup_ok and stopped
                if stopped:
                    self.log(
                        f"[frontend-validator] frontend_server_stopped=true mode={self.frontend_server_mode} pid={self._frontend_pid or self._frontend_proc.pid}"
                    )
                else:
                    self.log(
                        f"[frontend-validator] frontend_server_stopped=false mode={self.frontend_server_mode} pid={self._frontend_pid or self._frontend_proc.pid}"
                    )
            else:
                self.log(f"[frontend-validator] frontend_server_stop_skipped=true mode={self.frontend_server_mode}")
            self._frontend_proc = None
            self._frontend_started_by_validator = False
            self._frontend_reused_external = False
            self._frontend_pid = None
        elif self._frontend_reused_external:
            self._debug("frontend_server_reused_external=true")
            self.log(f"[frontend-validator] frontend_server_stop_skipped=true mode={self.frontend_server_mode}")
            self._frontend_reused_external = False
            self._frontend_started_by_validator = False

        self._active_pages = 0
        self._debug(f"cleanup_completed={cleanup_ok}")

    def _frontend_server_state(self) -> Dict[str, Any]:
        pid = self._frontend_pid
        if pid is None and self._frontend_proc is not None:
            pid = self._frontend_proc.pid
        return {
            "frontend_started_by_validator": bool(self._frontend_started_by_validator),
            "frontend_reused_external": bool(self._frontend_reused_external),
            "frontend_pid": pid,
        }

    def _build_frontend_entry_url(self, *, validation_expectations: Dict[str, Any]) -> str:
        base = self.frontend_url
        if not isinstance(validation_expectations, dict):
            return base
        if not bool(validation_expectations.get("enable_dev_mode")):
            return base

        parsed = urlparse(base)
        query = parse_qs(parsed.query or "")
        query["dev"] = ["true"]
        new_query = urlencode({k: v[-1] for k, v in query.items()}, doseq=False)
        return parsed._replace(query=new_query).geturl()

    def _debug(self, message: str) -> None:
        if self.debug:
            self.log(f"[frontend-validator] {message}")

    def run_query(
        self,
        request: FrontendValidationRequest | Dict[str, object],
        *,
        timeout_s: Optional[int] = None,
    ) -> Dict[str, object]:
        if self._browser is None:
            self.start()

        if self._browser is None:
            raise FrontendValidationError("Frontend browser is not available")

        normalized_request = coerce_frontend_validation_request(request)
        payload: Dict[str, object] = dict(normalized_request.payload)
        query_timeout = max(int(timeout_s or self.query_timeout_s), 20)
        effective_timeout = min(query_timeout, self.query_hard_cap_s)
        started_at = time.time()
        user_query = str(payload.get("user_query") or "").strip()
        if not user_query:
            raise FrontendValidationError("Frontend validation requires payload.user_query")
        validation_scenario = normalized_request.context.scenario
        validation_expectations = dict(normalized_request.context.expectations)
        validation_case_name = normalized_request.context.case_name or None
        ui_assertion_checks = list(validation_expectations.get("ui_assertions") or []) if isinstance(validation_expectations.get("ui_assertions"), list) else []
        contract_assertion_checks = (
            list(validation_expectations.get("contract_assertions") or [])
            if isinstance(validation_expectations.get("contract_assertions"), list)
            else []
        )
        fixture_scenario_name = self._resolve_fixture_scenario_name(validation_scenario)
        allow_live_backend = bool(validation_expectations.get("allow_live_backend"))
        use_fixture_mode = bool(self.fixture_mode_default and not allow_live_backend)
        if fixture_scenario_name:
            use_fixture_mode = True
        elif validation_scenario.startswith("mock_"):
            # Legacy compatibility for existing scenario labels.
            fixture_scenario_name = self._resolve_fixture_scenario_name(validation_scenario)
            use_fixture_mode = True
        is_multi_leg_query = self._is_multi_leg_query(user_query)

        page = self._browser.new_page()
        self._active_pages += 1
        self._debug(
            f"page_opened active_pages={self._active_pages} timeout_s={effective_timeout} "
            f"hard_cap_s={self.query_hard_cap_s}"
        )
        ask_records: List[Dict[str, Any]] = []
        endpoint_records: List[Dict[str, Any]] = []
        expected_request_payload: Dict[str, Any] = {"user_query": user_query}
        ui_actions: List[str] = []
        form_state: Dict[str, Any] = {}
        submission_mode = "query-only"
        ui_mode = "textarea"
        applied_structured_fields: List[str] = []
        supported_structured_fields: List[str] = []
        unsupported_structured_fields: List[str] = []
        frontend_state = self._frontend_server_state()
        source_payload_alignment = {
            "matches_source": True,
            "missing_keys": [],
            "mismatched_keys": [],
            "source_payload": payload,
        }

        def find_record(request_obj) -> Optional[Dict[str, Any]]:
            for rec in reversed(endpoint_records):
                if rec.get("_request_obj") is request_obj:
                    return rec
            return None

        def on_request(request) -> None:
            endpoint_kind = self._endpoint_kind_for_request(request.method, request.url)
            if not endpoint_kind:
                return

            raw_body = request.post_data or ""
            parsed_body = self._safe_json_loads(raw_body)
            is_ask = endpoint_kind in {"ask_stream", "ask_non_stream", "ask_async"}
            rec = {
                "_request_obj": request,
                "url": request.url,
                "method": request.method,
                "endpoint_kind": endpoint_kind,
                "is_ask": is_ask,
                "is_stream": endpoint_kind == "ask_stream",
                "request_payload": parsed_body,
                "request_body": raw_body,
                "matches_payload": self._payload_matches_request_payload(expected_request_payload, parsed_body) if is_ask else True,
                "response_status": None,
                "response_ok": False,
                "completed": False,
                "failed": False,
                "failure_text": "",
                "response_body_preview": "",
                "response_body_json": None,
                "stream_done_marker_checked": False,
                "stream_done_marker_seen": False,
                "stream_done_frame_found": False,
                "stream_done_event": "",
                "stream_done_json": None,
                "stream_done_json_parsed": False,
                "stream_done_json_error": "",
            }
            endpoint_records.append(rec)
            if is_ask:
                ask_records.append(rec)
            self._debug(
                f"api_request_started kind={endpoint_kind} url={request.url} "
                f"matched={rec['matches_payload'] if is_ask else 'n/a'} "
                f"payload={json.dumps(parsed_body, ensure_ascii=False) if parsed_body is not None else '<non-json>'}"
            )

        def on_response(response) -> None:
            rec = find_record(response.request)
            if not rec:
                return
            rec["response_status"] = response.status
            rec["response_ok"] = response.ok
            self._debug(
                f"api_response kind={rec.get('endpoint_kind')} status={response.status} ok={response.ok} url={response.url}"
            )

        def on_request_finished(request) -> None:
            rec = find_record(request)
            if not rec:
                return
            rec["completed"] = True
            try:
                response = request.response()
            except Exception:
                response = None
            if response is not None:
                rec["stream_done_marker_checked"] = bool(rec.get("is_stream"))
                body_text = ""
                try:
                    body_text = response.text() or ""
                except Exception:
                    body_text = ""
                if body_text:
                    preview = body_text[:1500]
                    rec["response_body_preview"] = preview
                    if rec.get("is_stream"):
                        parsed_stream = self._parse_done_json_from_sse_body(body_text)
                        rec["stream_done_marker_seen"] = bool(parsed_stream["marker_seen"])
                        rec["stream_done_frame_found"] = bool(parsed_stream["frame_found"])
                        rec["stream_done_event"] = str(parsed_stream["event_name"] or "")
                        rec["stream_done_json"] = parsed_stream["done_json"]
                        rec["stream_done_json_parsed"] = parsed_stream["done_json"] is not None
                        rec["stream_done_json_error"] = str(parsed_stream["error"] or "")
                    else:
                        rec["response_body_json"] = self._safe_json_loads(body_text)
            self._debug(f"api_request_finished kind={rec.get('endpoint_kind')}")

        def on_request_failed(request) -> None:
            rec = find_record(request)
            if not rec:
                return
            rec["failed"] = True
            failure = request.failure
            if isinstance(failure, dict):
                rec["failure_text"] = str(failure.get("errorText") or failure)
            elif failure is not None:
                rec["failure_text"] = str(failure)
            rec["completed"] = True
            self._debug(f"api_request_failed kind={rec.get('endpoint_kind')} error={rec['failure_text']}")

        page.on("request", on_request)
        page.on("response", on_response)
        page.on("requestfinished", on_request_finished)
        page.on("requestfailed", on_request_failed)
        try:
            goto_timeout_ms = max(30_000, int(self.startup_timeout_s * 1000))
            last_goto_error: Optional[Exception] = None
            if use_fixture_mode:
                self._install_fixture_routes(page, fixture_scenario_name, payload)
            entry_url = self._build_frontend_entry_url(validation_expectations=validation_expectations)
            for attempt in range(2):
                try:
                    page.goto(entry_url, wait_until="domcontentloaded", timeout=goto_timeout_ms)
                    last_goto_error = None
                    break
                except Exception as exc:
                    last_goto_error = exc
                    self._debug(f"page_goto_retry attempt={attempt + 1} timeout_ms={goto_timeout_ms}")
                    if attempt == 0:
                        page.wait_for_timeout(500)
            if last_goto_error is not None:
                raise last_goto_error
            self._wait_for_form_ready(page)
            pre_snapshot = self._extract_dom_snapshot(page)
            ui_reset_performed = self._is_clean_page_state(pre_snapshot)

            if not ui_reset_performed:
                network_summary = self._network_summary(ask_records)
                failure_context = self._build_failure_context(
                    fail_reason="UI reset check failed before submit (stale content visible).",
                    snapshot=pre_snapshot,
                    network_summary=network_summary,
                    timed_out=False,
                    payload_parity={"matches_expected": False, "missing_keys": [], "mismatched_keys": []},
                    source_payload_alignment=source_payload_alignment,
                    request_records=ask_records,
                )
                return {
                    "user_query": user_query,
                    "payload": payload,
                    "passes": False,
                    "fail_reason": "UI reset check failed before submit (stale content visible).",
                    "failure_phase": failure_context["phase"],
                    "failure_expectation": failure_context["expectation"],
                    "failure_selector": failure_context["selector"],
                    "failure_evidence": failure_context["evidence"],
                    "failure_dom_excerpt": failure_context["dom_excerpt"],
                    "error_visible": pre_snapshot["error_visible"],
                    "error_text": pre_snapshot["error_text"],
                    "timed_out": False,
                    "ui_reset_performed": ui_reset_performed,
                    "request_fired": False,
                    "request_completed_success": False,
                    "network_summary": network_summary,
                    "network_requests": self._sanitize_request_records(ask_records),
                    "endpoint_summary": self._endpoint_summary(endpoint_records),
                    "endpoint_requests": self._sanitize_endpoint_records(endpoint_records),
                    "runtime_s": round(time.time() - started_at, 3),
                    "timeout_limit_s": effective_timeout,
                    "ui_actions": ui_actions,
                    "form_state": form_state,
                    "submission_mode": submission_mode,
                    "ui_mode": ui_mode,
                    "applied_structured_fields": applied_structured_fields,
                    "supported_structured_fields": supported_structured_fields,
                    "unsupported_structured_fields": unsupported_structured_fields,
                    "source_payload_alignment": source_payload_alignment,
                    "intended_payload": expected_request_payload,
                    "post_actions_result": {"performed": [], "errors": []},
                    "frontend_server": frontend_state,
                    "validation_case_name": validation_case_name,
                    "validation_scenario": validation_scenario or None,
                    "validation_expectations": validation_expectations,
                    "ui_assertion_checks": ui_assertion_checks,
                    "contract_assertion_checks": contract_assertion_checks,
                    "pre_submit_snapshot": pre_snapshot,
                    "dom_snapshot": pre_snapshot,
                }

            (
                expected_request_payload,
                form_state,
                submission_mode,
                ui_mode,
                applied_structured_fields,
                supported_structured_fields,
                unsupported_structured_fields,
            ) = self._apply_payload_to_form(page, payload, user_query, ui_actions, validation_expectations)
            source_payload_alignment = self._build_source_payload_alignment(
                source_payload=payload,
                form_state=form_state,
            )
            self._debug(
                "submission_prepared "
                f"mode={submission_mode} ui_mode={ui_mode} "
                f"intended_payload={json.dumps(expected_request_payload, ensure_ascii=False)} "
                f"form_state={json.dumps(form_state, ensure_ascii=False)}"
            )

            submit_started = self._submit_query(page, ask_records)
            if not submit_started:
                snapshot = self._extract_dom_snapshot(page)
                network_summary = self._network_summary(ask_records)
                submit_fail_reason = "Submit did not trigger a new /ask request."
                if snapshot.get("error_visible") and snapshot.get("error_text"):
                    submit_fail_reason = (
                        "Submit blocked by UI validation: "
                        f"{str(snapshot.get('error_text') or '').strip()}"
                    )
                failure_context = self._build_failure_context(
                    fail_reason=submit_fail_reason,
                    snapshot=snapshot,
                    network_summary=network_summary,
                    timed_out=False,
                    payload_parity={"matches_expected": False, "missing_keys": [], "mismatched_keys": []},
                    source_payload_alignment=source_payload_alignment,
                    request_records=ask_records,
                )
                return {
                    "user_query": user_query,
                    "payload": payload,
                    "passes": False,
                    "fail_reason": submit_fail_reason,
                    "failure_phase": failure_context["phase"],
                    "failure_expectation": failure_context["expectation"],
                    "failure_selector": failure_context["selector"],
                    "failure_evidence": failure_context["evidence"],
                    "failure_dom_excerpt": failure_context["dom_excerpt"],
                    "error_visible": snapshot["error_visible"],
                    "error_text": snapshot["error_text"] or "No /ask request fired after submit.",
                    "timed_out": False,
                    "ui_reset_performed": ui_reset_performed,
                    "request_fired": network_summary["request_fired"],
                    "request_completed_success": network_summary["request_completed_success"],
                    "network_summary": network_summary,
                    "network_requests": self._sanitize_request_records(ask_records),
                    "endpoint_summary": self._endpoint_summary(endpoint_records),
                    "endpoint_requests": self._sanitize_endpoint_records(endpoint_records),
                    "ui_actions": ui_actions,
                    "form_state": form_state,
                    "submission_mode": submission_mode,
                    "ui_mode": ui_mode,
                    "applied_structured_fields": applied_structured_fields,
                    "supported_structured_fields": supported_structured_fields,
                    "unsupported_structured_fields": unsupported_structured_fields,
                    "source_payload_alignment": source_payload_alignment,
                    "intended_payload": expected_request_payload,
                    "post_actions_result": {"performed": [], "errors": []},
                    "frontend_server": frontend_state,
                    "validation_case_name": validation_case_name,
                    "validation_scenario": validation_scenario or None,
                    "validation_expectations": validation_expectations,
                    "ui_assertion_checks": ui_assertion_checks,
                    "contract_assertion_checks": contract_assertion_checks,
                    "runtime_s": round(time.time() - started_at, 3),
                    "timeout_limit_s": effective_timeout,
                    "pre_submit_snapshot": pre_snapshot,
                    "dom_snapshot": snapshot,
                }

            pre_action_result: Dict[str, Any] = {"performed": [], "errors": []}
            pre_settle_actions = validation_expectations.get("pre_settle_actions")
            if isinstance(pre_settle_actions, list) and pre_settle_actions:
                pre_action_result = self._run_post_actions(
                    page=page,
                    validation_expectations={"post_actions": pre_settle_actions},
                    endpoint_records=endpoint_records,
                    ui_actions=ui_actions,
                )
                if pre_action_result.get("performed"):
                    page.wait_for_timeout(250)

            snapshot = self._extract_dom_snapshot(page)
            deadline = time.time() + effective_timeout
            completed_since: Optional[float] = None
            while time.time() < deadline:
                network_summary = self._network_summary(ask_records)

                if snapshot["error_visible"] and network_summary["request_fired"]:
                    break

                if network_summary["request_completed"] and not snapshot["is_busy"]:
                    if completed_since is None:
                        completed_since = time.time()
                    elif time.time() - completed_since >= 0.6:
                        break
                else:
                    completed_since = None

                page.wait_for_timeout(250)
                snapshot = self._extract_dom_snapshot(page)

            timed_out = time.time() >= deadline
            if timed_out:
                self._debug("query_timeout_reached=true")
                cancelled = self._try_cancel_stream(page)
                if cancelled:
                    page.wait_for_timeout(300)
                    snapshot = self._extract_dom_snapshot(page)

            network_summary = self._network_summary(ask_records)
            endpoint_summary = self._endpoint_summary(endpoint_records)
            freshness_ok = self._is_fresh_result(pre_snapshot, snapshot)
            query_evidence = self._query_linked_evidence(payload, user_query, snapshot, is_multi_leg_query)
            payload_parity = self._build_payload_parity(
                expected_payload=expected_request_payload,
                records=ask_records,
            )
            post_actions_result = self._run_post_actions(
                page=page,
                validation_expectations=validation_expectations,
                endpoint_records=endpoint_records,
                ui_actions=ui_actions,
            )
            if pre_action_result.get("performed"):
                post_actions_result["performed"] = list(pre_action_result.get("performed") or []) + list(
                    post_actions_result.get("performed") or []
                )
            if pre_action_result.get("errors"):
                post_actions_result["errors"] = list(pre_action_result.get("errors") or []) + list(
                    post_actions_result.get("errors") or []
                )
            if post_actions_result.get("performed"):
                page.wait_for_timeout(350)
                snapshot = self._extract_dom_snapshot(page)
                network_summary = self._network_summary(ask_records)
                endpoint_summary = self._endpoint_summary(endpoint_records)
            primary_request = self._pick_primary_request_record(ask_records)
            passes, fail_reason = self._evaluate_case(
                snapshot=snapshot,
                network_summary=network_summary,
                endpoint_summary=endpoint_summary,
                ui_reset_performed=ui_reset_performed,
                freshness_ok=freshness_ok,
                query_evidence=query_evidence,
                is_multi_leg_query=is_multi_leg_query,
                timed_out=timed_out,
                payload_parity=payload_parity,
                source_payload_alignment=source_payload_alignment,
                validation_scenario=validation_scenario,
                validation_expectations=validation_expectations,
                request_records=ask_records,
                post_actions_result=post_actions_result,
            )
            failure_context = self._build_failure_context(
                fail_reason=fail_reason,
                snapshot=snapshot,
                network_summary=network_summary,
                timed_out=timed_out,
                payload_parity=payload_parity,
                source_payload_alignment=source_payload_alignment,
                request_records=ask_records,
            )

            if timed_out and not snapshot.get("error_text"):
                snapshot["error_text"] = "Frontend timed out before request+UI completion."

            return {
                "user_query": user_query,
                "payload": payload,
                "passes": passes,
                "fail_reason": fail_reason,
                "failure_phase": failure_context["phase"] if not passes else "passed",
                "failure_expectation": failure_context["expectation"] if not passes else "",
                "failure_selector": failure_context["selector"] if not passes else "",
                "failure_evidence": failure_context["evidence"] if not passes else {},
                "failure_dom_excerpt": failure_context["dom_excerpt"] if not passes else {},
                "error_visible": snapshot["error_visible"],
                "error_text": snapshot["error_text"],
                "timed_out": timed_out,
                "ui_reset_performed": ui_reset_performed,
                "request_fired": network_summary["request_fired"],
                "request_completed_success": network_summary["request_completed_success"],
                "network_summary": network_summary,
                "endpoint_summary": endpoint_summary,
                "network_requests": self._sanitize_request_records(ask_records),
                "endpoint_requests": self._sanitize_endpoint_records(endpoint_records),
                "ui_actions": ui_actions,
                "form_state": form_state,
                "submission_mode": submission_mode,
                "ui_mode": ui_mode,
                "applied_structured_fields": applied_structured_fields,
                "supported_structured_fields": supported_structured_fields,
                "unsupported_structured_fields": unsupported_structured_fields,
                "source_payload_alignment": source_payload_alignment,
                "intended_payload": expected_request_payload,
                "post_actions_result": post_actions_result,
                "payload_parity": payload_parity,
                "actual_request_payload": payload_parity.get("actual_payload"),
                "actual_request_url": primary_request.get("url") if primary_request else None,
                "actual_request_status": primary_request.get("response_status") if primary_request else None,
                "actual_request_stream": bool(primary_request.get("is_stream")) if primary_request else None,
                "query_evidence": query_evidence,
                "freshness_ok": freshness_ok,
                "query_type": "multi_leg" if is_multi_leg_query else "standard",
                "runtime_s": round(time.time() - started_at, 3),
                "timeout_limit_s": effective_timeout,
                "frontend_server": frontend_state,
                "validation_case_name": validation_case_name,
                "validation_scenario": validation_scenario or None,
                "validation_expectations": validation_expectations,
                "ui_assertion_checks": ui_assertion_checks,
                "contract_assertion_checks": contract_assertion_checks,
                "pre_submit_snapshot": pre_snapshot,
                "dom_snapshot": snapshot,
                **snapshot,
            }
        finally:
            for event_name, handler in (
                ("request", on_request),
                ("response", on_response),
                ("requestfinished", on_request_finished),
                ("requestfailed", on_request_failed),
            ):
                try:
                    page.remove_listener(event_name, handler)
                except Exception:
                    self._debug(f"listener_remove_failed event={event_name}")
            try:
                page.close()
            except Exception:
                self._debug("page_close_failed=true")
            finally:
                self._active_pages = max(0, self._active_pages - 1)
                self._debug(f"page_closed active_pages={self._active_pages}")

    def _build_failure_context(
        self,
        *,
        fail_reason: str,
        snapshot: Dict[str, Any],
        network_summary: Dict[str, Any],
        timed_out: bool,
        payload_parity: Dict[str, Any],
        source_payload_alignment: Dict[str, Any],
        request_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        reason = str(fail_reason or "").strip() or "Frontend validation failed."
        reason_l = reason.lower()
        phase = "unknown"
        selector = ""

        phase_map = [
            ("planner form did not load", "form_ready", "form.planner-form, textarea.nl-input, button.nl-send"),
            ("ui reset failed before query submit", "pre_submit_reset", "result surfaces should be cleared before submit"),
            ("source payload", "form_payload_alignment", "input.f-input, input.date-native, trip-tab controls"),
            ("submit blocked by ui validation", "submit_request", "button.nl-send / textarea.nl-input Enter"),
            ("submit did not trigger a new /ask request", "submit_request", "button.nl-send / textarea.nl-input Enter"),
            ("no /ask request was fired", "submit_request", "button.nl-send / textarea.nl-input Enter"),
            ("payload matched", "request_payload_match", "captured /ask payload"),
            ("did not complete successfully", "network_response", "/ask response status and completion"),
            ("streaming started but neither stream completion", "stream_runtime", "SSE completion or fallback request"),
            ("done_json", "stream_completion", "SSE [DONE_JSON] frame"),
            ("fallback non-stream request", "fallback_request", "fallback /ask non-stream request"),
            ("timed out", "wait_timeout", "request completion + UI settle window"),
            ("ui still busy", "render_settle", "button.nl-send enabled state"),
            ("error banner visible", "ui_error_banner", ".notice--error"),
            ("rendered result did not change", "render_refresh", ".result-wrap content diff"),
            ("payload does not match intended", "request_payload_parity", "captured /ask request payload"),
            ("query-linked evidence", "render_query_evidence", "stream/reasoning/weather/flight text corpus"),
            ("proof overview section is missing", "render_proof_overview", ".proof-overview-grid"),
            ("proof evidence list is missing", "render_proof_evidence", ".proof-evidence-list"),
            ("ranked shortlist section label is missing", "render_shortlist", ".r-label:has-text('Ranked shortlist')"),
            ("return leg snapshot", "render_round_trip", ".r-label:has-text('Return leg snapshot')"),
            ("multi-city itinerary section", "render_via_stopover", ".r-label:has-text('Multi-city itinerary')"),
            ("degraded scenario", "render_degraded_state", ".notice.notice--inline contains partial result"),
            ("no-flights scenario", "render_no_flights_state", ".notice--error + proof surfaces"),
            ("booking handoff", "render_booking_handoff", "a.flight-card__link--primary"),
        ]
        for needle, mapped_phase, mapped_selector in phase_map:
            if needle in reason_l:
                phase = mapped_phase
                selector = mapped_selector
                break

        if timed_out and phase == "unknown":
            phase = "wait_timeout"
            selector = "request completion + UI settle window"

        primary_request = self._pick_primary_request_record(request_records)
        primary_summary = {}
        if primary_request:
            primary_summary = {
                "url": primary_request.get("url"),
                "status": primary_request.get("response_status"),
                "is_stream": bool(primary_request.get("is_stream")),
                "failed": bool(primary_request.get("failed")),
                "failure_text": primary_request.get("failure_text"),
                "stream_done_marker_seen": bool(primary_request.get("stream_done_marker_seen")),
                "stream_done_frame_found": bool(primary_request.get("stream_done_frame_found")),
                "stream_done_json_parsed": bool(primary_request.get("stream_done_json_parsed")),
                "stream_done_json_error": primary_request.get("stream_done_json_error"),
            }

        evidence = {
            "matched_statuses": network_summary.get("matched_statuses"),
            "request_count": network_summary.get("request_count"),
            "matched_request_count": network_summary.get("matched_request_count"),
            "request_fired": network_summary.get("request_fired"),
            "request_completed_success": network_summary.get("request_completed_success"),
            "streaming_completed": network_summary.get("streaming_completed"),
            "payload_missing_keys": payload_parity.get("missing_keys"),
            "payload_mismatched_keys": payload_parity.get("mismatched_keys"),
            "source_missing_keys": source_payload_alignment.get("missing_keys"),
            "source_mismatched_keys": source_payload_alignment.get("mismatched_keys"),
            "snapshot_flags": {
                "planner_form_ready": bool(snapshot.get("planner_form_ready")),
                "result_wrap_visible": bool(snapshot.get("result_wrap_visible")),
                "best_flight_visible": bool(snapshot.get("best_flight_visible")),
                "flights_count": snapshot.get("flights_count"),
                "weather_present": bool(snapshot.get("weather_present")),
                "reasoning_present": bool(snapshot.get("reasoning_present")),
                "trip_brief_present": bool(snapshot.get("trip_brief_present")),
                "proof_overview_visible": bool(snapshot.get("proof_overview_visible")),
                "proof_evidence_visible": bool(snapshot.get("proof_evidence_visible")),
                "ranked_shortlist_visible": bool(snapshot.get("ranked_shortlist_visible")),
                "degraded_notice_visible": bool(snapshot.get("degraded_notice_visible")),
                "no_flights_error_visible": bool(snapshot.get("no_flights_error_visible")),
                "booking_link_visible": bool(snapshot.get("booking_link_visible")),
                "seller_signal_visible": bool(snapshot.get("seller_signal_visible")),
                "checkout_unavailable_visible": bool(snapshot.get("checkout_unavailable_visible")),
                "provider_handoff_hint_visible": bool(snapshot.get("provider_handoff_hint_visible")),
                "booking_navigation_state": str(snapshot.get("booking_navigation_state") or ""),
            },
            "primary_request": primary_summary,
        }

        dom_excerpt = {
            "error_text": str(snapshot.get("error_text") or "")[:280],
            "degraded_notice_text": str(snapshot.get("degraded_notice_text") or "")[:280],
            "stream_text_head": str(snapshot.get("stream_text") or "")[:280],
            "reasoning_text_head": str(snapshot.get("reasoning_text") or "")[:280],
            "weather_text_head": str(snapshot.get("weather_text") or "")[:280],
            "flight_text_head": str(snapshot.get("flight_text") or "")[:280],
            "notice_texts": list(snapshot.get("notice_texts") or [])[:3],
        }

        return {
            "phase": phase,
            "expectation": reason,
            "selector": selector,
            "evidence": evidence,
            "dom_excerpt": dom_excerpt,
        }

    def _extract_dom_snapshot(self, page) -> Dict[str, object]:
        error_locator = page.get_by_test_id("notice-error")
        if error_locator.count() == 0:
            error_locator = page.locator(".notice--error")
        error_visible = error_locator.count() > 0
        error_text = error_locator.first.inner_text().strip() if error_visible else ""
        planner_form_ready = (
            page.get_by_test_id("planner-form").count() > 0
            and page.get_by_test_id("query-input").count() > 0
        )

        stream_text = ""
        stream_locator = page.get_by_test_id("stream-pane-body")
        if stream_locator.count() == 0:
            stream_locator = page.locator(".stream-pane__body")
        if stream_locator.count() > 0:
            stream_text = stream_locator.first.inner_text().strip()

        best_flight_highlight = page.get_by_text("Best Flight", exact=False).count() > 0
        best_flight_badge = page.get_by_test_id("flight-card-best").count() > 0

        weather_ready = page.locator("[data-weather-ready='true']").count() > 0 or page.locator(".weather-summary--ready").count() > 0
        weather_items = page.locator(".weather-summary__item").count()
        reasoning_items = page.locator(".reasoning-panel .reasoning-list__item").count()
        flights_count = page.get_by_test_id("flight-card").count()
        if flights_count == 0:
            flights_count = page.locator(".flight-item").count()

        reasoning_text = ""
        reasoning_locator = page.get_by_test_id("reasoning-panel")
        if reasoning_locator.count() == 0:
            reasoning_locator = page.locator(".reasoning-panel")
        if reasoning_locator.count() > 0:
            reasoning_text = reasoning_locator.first.inner_text().strip()

        weather_text = ""
        weather_locator = page.get_by_test_id("weather-summary")
        if weather_locator.count() == 0:
            weather_locator = page.locator(".weather-summary")
        if weather_locator.count() > 0:
            weather_text = weather_locator.first.inner_text().strip()

        highlight_text = ""
        highlight_locator = page.locator(".highlights-row")
        if highlight_locator.count() > 0:
            highlight_text = highlight_locator.first.inner_text().strip()

        flight_text = ""
        flight_locator = page.get_by_test_id("flights-list")
        if flight_locator.count() == 0:
            flight_locator = page.locator(".flights-stack")
        if flight_locator.count() > 0:
            flight_text = flight_locator.first.inner_text().strip()

        ranked_shortlist_visible = page.get_by_test_id("ranked-shortlist").count() > 0 or page.get_by_text("Ranked shortlist", exact=False).count() > 0
        proof_overview_visible = page.get_by_test_id("proof-overview").count() > 0 or page.locator(".proof-overview-grid").count() > 0
        proof_evidence_visible = page.get_by_test_id("proof-evidence").count() > 0 or page.locator(".proof-evidence-list").count() > 0
        result_wrap_visible = page.get_by_test_id("result-wrap").count() > 0 or page.locator(".result-wrap").count() > 0
        return_leg_visible = page.get_by_test_id("return-leg").count() > 0 or page.get_by_text("Return leg snapshot", exact=False).count() > 0
        multicity_visible = page.get_by_test_id("multicity-itinerary").count() > 0 or page.get_by_text("Multi-city itinerary", exact=False).count() > 0

        notice_texts: List[str] = []
        notice_locator = page.get_by_test_id("notice-inline")
        if notice_locator.count() == 0:
            notice_locator = page.locator(".notice.notice--inline")
        notice_count = min(notice_locator.count(), 5)
        for idx in range(notice_count):
            try:
                text = (notice_locator.nth(idx).inner_text() or "").strip()
            except Exception:
                text = ""
            if text:
                notice_texts.append(text)

        degraded_notice_text = ""
        for candidate in notice_texts:
            if "partial result:" in candidate.lower():
                degraded_notice_text = candidate
                break
        degraded_notice_visible = bool(degraded_notice_text)

        booking_link_locator = page.get_by_test_id("booking-link")
        if booking_link_locator.count() == 0:
            booking_link_locator = page.locator("a.flight-card__link--primary")
        booking_link_visible = booking_link_locator.count() > 0
        booking_link_href = ""
        booking_link_target = ""
        booking_link_rel = ""
        booking_link_labels: List[str] = []
        if booking_link_visible:
            try:
                booking_link_href = str(booking_link_locator.first.get_attribute("href") or "")
                booking_link_target = str(booking_link_locator.first.get_attribute("target") or "")
                booking_link_rel = str(booking_link_locator.first.get_attribute("rel") or "")
                link_count = min(booking_link_locator.count(), 3)
                for idx in range(link_count):
                    label = str(booking_link_locator.nth(idx).inner_text() or "").strip()
                    if label:
                        booking_link_labels.append(label)
            except Exception:
                booking_link_href = ""
                booking_link_target = ""
                booking_link_rel = ""
                booking_link_labels = []

        no_flights_error_visible = bool(
            error_visible
            and re.search(
                r"no matching flights|no live flights|no flights found",
                error_text or "",
                re.IGNORECASE,
            )
        )

        send_button = page.get_by_test_id("submit-query")
        if send_button.count() == 0:
            send_button = page.locator("button.nl-send")
        is_busy = False
        if send_button.count() > 0:
            try:
                is_busy = send_button.first.is_disabled()
            except Exception:
                is_busy = False

        booking_panel_text = ""
        booking_panel = page.get_by_test_id("booking-panel")
        booking_panel_visible = booking_panel.count() > 0
        if booking_panel.count() > 0:
            try:
                booking_panel_text = booking_panel.first.inner_text().strip()
            except Exception:
                booking_panel_text = ""

        tracking_panel_text = ""
        tracking_panel = page.get_by_test_id("tracking-panel")
        tracking_panel_visible = tracking_panel.count() > 0
        if tracking_panel.count() > 0:
            try:
                tracking_panel_text = tracking_panel.first.inner_text().strip()
            except Exception:
                tracking_panel_text = ""

        seller_badges = page.get_by_test_id("booking-seller")
        seller_badges_count = seller_badges.count()
        seller_signal_visible = bool(
            seller_badges_count > 0
            or any("book with" in label.lower() for label in booking_link_labels)
        )
        checkout_unavailable_visible = bool(
            re.search(
                r"checkout (?:link )?is currently unavailable|booking unavailable|held locally",
                " ".join(notice_texts + [booking_panel_text, flight_text]).lower(),
            )
        )
        provider_handoff_hint_visible = bool(
            booking_link_visible
            or "provider handoff" in " ".join([booking_panel_text, flight_text]).lower()
        )
        booking_navigation_state = "unknown"
        if booking_link_visible:
            booking_navigation_state = "provider_handoff_present"
        elif checkout_unavailable_visible and "held locally" in " ".join([booking_panel_text] + notice_texts).lower():
            booking_navigation_state = "held_local_only"
        elif checkout_unavailable_visible:
            booking_navigation_state = "checkout_unavailable"

        return {
            "best_flight_visible": bool(best_flight_highlight or best_flight_badge),
            "flights_count": flights_count,
            "weather_present": bool(weather_ready and weather_items > 0),
            "reasoning_present": reasoning_items > 0,
            "trip_brief_present": len(stream_text) > 0,
            "error_visible": error_visible,
            "error_text": error_text,
            "is_busy": is_busy,
            "stream_text": stream_text,
            "reasoning_text": reasoning_text,
            "weather_text": weather_text,
            "highlight_text": highlight_text,
            "flight_text": flight_text,
            "planner_form_ready": planner_form_ready,
            "ranked_shortlist_visible": ranked_shortlist_visible,
            "proof_overview_visible": proof_overview_visible,
            "proof_evidence_visible": proof_evidence_visible,
            "result_wrap_visible": result_wrap_visible,
            "return_leg_visible": return_leg_visible,
            "multicity_visible": multicity_visible,
            "notice_texts": notice_texts,
            "degraded_notice_visible": degraded_notice_visible,
            "degraded_notice_text": degraded_notice_text,
            "booking_link_visible": booking_link_visible,
            "booking_link_href": booking_link_href,
            "booking_link_target": booking_link_target,
            "booking_link_rel": booking_link_rel,
            "booking_link_labels": booking_link_labels,
            "seller_badges_count": seller_badges_count,
            "seller_signal_visible": seller_signal_visible,
            "checkout_unavailable_visible": checkout_unavailable_visible,
            "provider_handoff_hint_visible": provider_handoff_hint_visible,
            "booking_navigation_state": booking_navigation_state,
            "no_flights_error_visible": no_flights_error_visible,
            "booking_panel_visible": booking_panel_visible,
            "booking_panel_text": booking_panel_text,
            "tracking_panel_visible": tracking_panel_visible,
            "tracking_panel_text": tracking_panel_text,
        }

    def _wait_for_form_ready(self, page) -> None:
        page.wait_for_selector("[data-testid='planner-form']", timeout=15_000)
        query_input = page.get_by_role("textbox", name=re.compile("query", re.IGNORECASE))
        if query_input.count() == 0:
            query_input = page.get_by_test_id("query-input")
        query_input.first.wait_for(timeout=15_000)

        submit_button = page.get_by_role("button", name=re.compile("submit query|plan my trip", re.IGNORECASE))
        if submit_button.count() == 0:
            submit_button = page.get_by_test_id("submit-query")
        ready_deadline = time.time() + 10.0
        while time.time() < ready_deadline:
            try:
                if submit_button.count() > 0 and submit_button.first.is_enabled():
                    return
            except Exception:
                pass
            page.wait_for_timeout(120)
        raise FrontendValidationError("Frontend form submit button did not become ready in time.")

    def _submit_query(self, page, ask_records: List[Dict[str, Any]]) -> bool:
        baseline_count = len(ask_records)
        submit_button = page.get_by_role("button", name=re.compile("submit query|plan my trip", re.IGNORECASE))
        if submit_button.count() == 0:
            submit_button = page.get_by_test_id("submit-query")
        query_input = page.get_by_role("textbox", name=re.compile("query", re.IGNORECASE))
        if query_input.count() == 0:
            query_input = page.get_by_test_id("query-input")
        for attempt in range(2):
            try:
                if attempt == 0:
                    submit_button.first.click()
                else:
                    query_input.first.press("Enter")
                self._debug(f"submit_attempt={attempt + 1}")
                started_deadline = time.time() + 8.0
                while time.time() < started_deadline:
                    if len(ask_records) > baseline_count:
                        self._debug("submit_request_detected=true")
                        return True
                    page.wait_for_timeout(120)
            except Exception:
                page.wait_for_timeout(200)
        self._debug("submit_request_detected=false")
        return False

    def _apply_payload_to_form(
        self,
        page,
        payload: Dict[str, object],
        user_query: str,
        ui_actions: List[str],
        validation_expectations: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str, str, List[str], List[str], List[str]]:
        supported_fields = [
            key
            for key in ("origin", "destination", "date", "trip_type", "return_date", "direct_only", "cabin", "baggage_pref")
            if payload.get(key) not in (None, "", False)
        ]
        unsupported_candidates = (
            "price_limit",
            "budget",
            "price_cap",
            "wants_direct",
            "preferred_airlines",
            "airline",
            "constraints",
        )
        unsupported_fields = [key for key in unsupported_candidates if payload.get(key)]

        # Prefer user-facing roles/labels first; keep test-id fallback as stable seam.
        query_input = page.get_by_role("textbox", name=re.compile("query", re.IGNORECASE))
        if query_input.count() == 0:
            query_input = page.get_by_test_id("query-input")
        query_input.first.fill(user_query)
        ui_actions.append("fill:query-input")

        applied_structured_fields: List[str] = []
        source_trip_type = self._normalize_trip_type(str(payload.get("trip_type") or ""))
        inferred_trip_type = self._infer_trip_type_from_query(user_query)
        effective_trip_type = source_trip_type or inferred_trip_type
        should_apply_trip_type_tab = bool(source_trip_type) or effective_trip_type in {"round-trip", "via-stopover"}
        if should_apply_trip_type_tab and self._set_trip_type_tab(page, effective_trip_type):
            ui_actions.append(f"set:trip_type={effective_trip_type}")
            if source_trip_type:
                applied_structured_fields.append("trip_type")

        origin = str(payload.get("origin") or "")
        destination = str(payload.get("destination") or "")
        origin_input = page.get_by_label("Origin", exact=False)
        if origin_input.count() == 0:
            origin_input = page.get_by_test_id("input-origin")
        destination_input = page.get_by_label("Destination", exact=False)
        if destination_input.count() == 0:
            destination_input = page.get_by_test_id("input-destination")
        if origin and origin_input.count() > 0:
            origin_input.first.fill(origin)
            ui_actions.append(f"fill:origin={origin}")
            applied_structured_fields.append("origin")
        if destination and destination_input.count() > 0:
            destination_input.first.fill(destination)
            ui_actions.append(f"fill:destination={destination}")
            applied_structured_fields.append("destination")

        date_value = str(payload.get("date") or "")
        date_input = page.get_by_label("Travel date", exact=False)
        if date_input.count() == 0:
            date_input = page.get_by_test_id("input-date")
        if date_input.count() == 0:
            date_input = page.locator("input.date-native")
        if date_input.count() > 0:
            if date_value:
                date_input.first.fill(date_value)
                ui_actions.append(f"fill:date={date_value}")
                applied_structured_fields.append("date")

        return_date_value = str(payload.get("return_date") or "")
        return_date_input = page.get_by_label("Return date", exact=False)
        if return_date_input.count() == 0:
            return_date_input = page.get_by_test_id("input-return-date")
        if return_date_input.count() > 0 and return_date_value:
            return_date_input.first.fill(return_date_value)
            ui_actions.append(f"fill:return_date={return_date_value}")
            applied_structured_fields.append("return_date")
        elif return_date_value and return_date_input.count() == 0:
            unsupported_fields.append("return_date")

        direct_requested = bool(payload.get("direct_only") or payload.get("wants_direct"))
        direct_toggle = page.get_by_label("Direct only", exact=False)
        if direct_toggle.count() == 0:
            direct_toggle = page.get_by_test_id("toggle-direct-only")
        if direct_toggle.count() > 0:
            try:
                current_checked = bool(direct_toggle.first.is_checked())
                if direct_requested != current_checked:
                    if direct_requested:
                        direct_toggle.first.check()
                    else:
                        direct_toggle.first.uncheck()
                    ui_actions.append(f"toggle:direct_only={str(direct_requested).lower()}")
                if direct_requested:
                    applied_structured_fields.append("direct_only")
            except Exception:
                pass
        elif direct_requested:
            unsupported_fields.append("direct_only")

        cabin_value = str(payload.get("cabin") or payload.get("cabin_pref") or "").strip().lower()
        cabin_select = page.get_by_label("Cabin", exact=False)
        if cabin_select.count() == 0:
            cabin_select = page.get_by_test_id("select-cabin")
        if cabin_select.count() > 0 and cabin_value:
            try:
                cabin_select.first.select_option(cabin_value)
                ui_actions.append(f"select:cabin={cabin_value}")
                applied_structured_fields.append("cabin")
            except Exception:
                pass
        elif cabin_value:
            unsupported_fields.append("cabin")

        baggage_value = str(payload.get("baggage_pref") or payload.get("baggage") or "").strip().lower()
        baggage_select = page.get_by_label("Baggage", exact=False)
        if baggage_select.count() == 0:
            baggage_select = page.get_by_test_id("select-baggage")
        if baggage_select.count() > 0 and baggage_value:
            try:
                baggage_select.first.select_option(baggage_value)
                ui_actions.append(f"select:baggage_pref={baggage_value}")
                applied_structured_fields.append("baggage_pref")
            except Exception:
                pass
        elif baggage_value:
            unsupported_fields.append("baggage_pref")

        async_toggle = page.get_by_label("Run in background", exact=False)
        if async_toggle.count() == 0:
            async_toggle = page.get_by_test_id("toggle-async")
        if bool(validation_expectations.get("enable_async_mode")) and async_toggle.count() > 0:
            toggle = async_toggle.first
            try:
                if not toggle.is_checked():
                    toggle.check()
                    ui_actions.append("toggle:async=true")
            except Exception:
                pass

        form_state = self._read_form_state(page)
        intended_payload: Dict[str, Any] = {}
        if form_state.get("user_query"):
            intended_payload["user_query"] = form_state["user_query"]
        if form_state.get("origin"):
            intended_payload["origin"] = form_state["origin"]
        if form_state.get("destination"):
            intended_payload["destination"] = form_state["destination"]
        if form_state.get("date"):
            intended_payload["date"] = form_state["date"]
        if form_state.get("trip_type"):
            intended_payload["trip_type"] = form_state["trip_type"]
        if form_state.get("return_date"):
            intended_payload["return_date"] = form_state["return_date"]
        if form_state.get("direct_only"):
            intended_payload["direct_only"] = bool(form_state["direct_only"])
        if form_state.get("cabin"):
            intended_payload["cabin"] = form_state["cabin"]
        if form_state.get("baggage_pref"):
            intended_payload["baggage_pref"] = form_state["baggage_pref"]

        ui_mode = "structured" if applied_structured_fields else "textarea"
        submission_mode = "query-plus-structured" if ui_mode == "structured" else "query-only"
        return (
            intended_payload,
            form_state,
            submission_mode,
            ui_mode,
            applied_structured_fields,
            supported_fields,
            unsupported_fields,
        )

    def _read_form_state(self, page) -> Dict[str, Any]:
        def _value(locator) -> str:
            if locator.count() == 0:
                return ""
            try:
                return str(locator.first.input_value() or "").strip()
            except Exception:
                return ""

        active_trip = ""
        active_tab = page.locator("[data-testid^='trip-tab-'].active")
        if active_tab.count() == 0:
            active_tab = page.locator(".trip-tab.active")
        if active_tab.count() > 0:
            try:
                active_trip = str(active_tab.first.inner_text() or "").strip().lower()
            except Exception:
                active_trip = ""

        normalized_trip = self._normalize_trip_type(active_trip)
        direct_only = False
        direct_toggle = page.get_by_label("Direct only", exact=False)
        if direct_toggle.count() == 0:
            direct_toggle = page.get_by_test_id("toggle-direct-only")
        if direct_toggle.count() > 0:
            try:
                direct_only = bool(direct_toggle.first.is_checked())
            except Exception:
                direct_only = False

        cabin_locator = page.get_by_label("Cabin", exact=False)
        if cabin_locator.count() == 0:
            cabin_locator = page.get_by_test_id("select-cabin")
        baggage_locator = page.get_by_label("Baggage", exact=False)
        if baggage_locator.count() == 0:
            baggage_locator = page.get_by_test_id("select-baggage")
        query_locator = page.get_by_role("textbox", name=re.compile("query", re.IGNORECASE))
        if query_locator.count() == 0:
            query_locator = page.get_by_test_id("query-input")
        origin_locator = page.get_by_label("Origin", exact=False)
        if origin_locator.count() == 0:
            origin_locator = page.get_by_test_id("input-origin")
        destination_locator = page.get_by_label("Destination", exact=False)
        if destination_locator.count() == 0:
            destination_locator = page.get_by_test_id("input-destination")
        date_locator = page.get_by_label("Travel date", exact=False)
        if date_locator.count() == 0:
            date_locator = page.get_by_test_id("input-date")
        return_date_locator = page.get_by_label("Return date", exact=False)
        if return_date_locator.count() == 0:
            return_date_locator = page.get_by_test_id("input-return-date")

        cabin = _value(cabin_locator)
        baggage_pref = _value(baggage_locator)
        return {
            "user_query": _value(query_locator),
            "origin": _value(origin_locator),
            "destination": _value(destination_locator),
            "date": _value(date_locator),
            "return_date": _value(return_date_locator),
            "trip_type": normalized_trip or "one-way",
            "direct_only": direct_only,
            "cabin": cabin,
            "baggage_pref": baggage_pref,
        }

    def _normalize_trip_type(self, raw_value: str) -> str:
        v = (raw_value or "").strip().lower()
        if v in {"one-way", "one way", "oneway"}:
            return "one-way"
        if v in {"round-trip", "round trip", "return"}:
            return "round-trip"
        if v in {"via-stopover", "via / stopover", "via stopover", "stopover"}:
            return "via-stopover"
        return ""

    def _infer_trip_type_from_query(self, user_query: str) -> str:
        q = user_query.lower()
        if re.search(r"\b(via|stopover|stop over|connecting through|stop in)\b", q):
            return "via-stopover"
        if re.search(r"\b(round[- ]?trip|return(ing)?|come back)\b", q):
            return "round-trip"
        return "one-way"

    def _set_trip_type_tab(self, page, trip_type: str) -> bool:
        test_ids = {
            "one-way": "trip-tab-one-way",
            "round-trip": "trip-tab-round-trip",
            "via-stopover": "trip-tab-via-stopover",
        }
        target_test_id = test_ids.get(trip_type)
        if not target_test_id:
            return False
        tab = page.get_by_test_id(target_test_id)
        if tab.count() == 0:
            label_fallback = {
                "one-way": "One-way",
                "round-trip": "Round-trip",
                "via-stopover": "Via / Stopover",
            }.get(trip_type, "")
            tab = page.locator(".trip-tab", has_text=label_fallback)
        if tab.count() == 0:
            return False
        try:
            tab.first.click()
            return True
        except Exception:
            return False

    def _build_payload_parity(self, expected_payload: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
        candidate = self._pick_primary_request_record(records)
        actual_payload = candidate.get("request_payload") if candidate else None
        if not isinstance(actual_payload, dict):
            return {
                "matches_expected": False,
                "missing_keys": sorted(list(expected_payload.keys())),
                "mismatched_keys": [],
                "actual_payload": actual_payload,
            }

        missing: List[str] = []
        mismatched: List[str] = []
        for key, expected_value in expected_payload.items():
            if key not in actual_payload:
                missing.append(key)
                continue
            if str(actual_payload.get(key, "")).strip() != str(expected_value).strip():
                mismatched.append(key)

        return {
            "matches_expected": not missing and not mismatched,
            "missing_keys": missing,
            "mismatched_keys": mismatched,
            "actual_payload": actual_payload,
        }

    def _pick_primary_request_record(self, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not records:
            return None

        matched = [rec for rec in records if rec.get("matches_payload")]
        candidates = matched if matched else records

        successful = []
        for rec in candidates:
            status = rec.get("response_status")
            if bool(rec.get("completed")) and not rec.get("failed") and isinstance(status, int) and 200 <= status < 300:
                successful.append(rec)

        if successful:
            stream_first = [rec for rec in successful if rec.get("is_stream")]
            return stream_first[-1] if stream_first else successful[-1]

        return candidates[-1]

    def _try_cancel_stream(self, page) -> bool:
        cancel_button = page.get_by_test_id("stream-cancel")
        if cancel_button.count() == 0:
            cancel_button = page.locator("button.stream-pane__cancel")
        if cancel_button.count() == 0:
            self._debug("stream_cancel_available=false")
            return False
        try:
            cancel_button.first.click(timeout=1_000)
            self._debug("stream_cancel_clicked=true")
            return True
        except Exception:
            self._debug("stream_cancel_clicked=false")
            return False

    def _safe_json_loads(self, raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            return None

    def _resolve_fixture_scenario_name(self, scenario: str) -> str:
        return resolve_frontend_fixture_scenario_name(
            scenario,
            fixture_catalog=self.fixture_catalog,
        )

    def _endpoint_kind_for_request(self, method: str, url: str) -> str:
        return classify_frontend_endpoint_request(method, url)

    def _endpoint_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"calls": {}, "failed_calls": {}}
        for rec in records:
            kind = str(rec.get("endpoint_kind") or "").strip() or "unknown"
            calls = summary["calls"]
            failed_calls = summary["failed_calls"]
            calls[kind] = int(calls.get(kind, 0)) + 1
            if rec.get("failed"):
                failed_calls[kind] = int(failed_calls.get(kind, 0)) + 1
        return summary

    def _sanitize_endpoint_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean: List[Dict[str, Any]] = []
        for rec in records:
            clean.append(
                {
                    "endpoint_kind": rec.get("endpoint_kind"),
                    "url": rec.get("url"),
                    "method": rec.get("method"),
                    "response_status": rec.get("response_status"),
                    "response_ok": rec.get("response_ok"),
                    "completed": rec.get("completed"),
                    "failed": rec.get("failed"),
                    "failure_text": rec.get("failure_text"),
                    "request_payload": rec.get("request_payload"),
                }
            )
        return clean

    def _run_post_actions(
        self,
        *,
        page,
        validation_expectations: Dict[str, Any],
        endpoint_records: List[Dict[str, Any]],
        ui_actions: List[str],
    ) -> Dict[str, Any]:
        actions = validation_expectations.get("post_actions") or []
        if not isinstance(actions, list) or not actions:
            return {"performed": [], "errors": []}

        performed: List[str] = []
        errors: List[str] = []

        def _click_first(locator, action_name: str) -> bool:
            if locator.count() <= 0:
                errors.append(f"{action_name}: target missing")
                return False
            try:
                locator.first.click(timeout=1_500)
                performed.append(action_name)
                ui_actions.append(f"post_action:{action_name}")
                page.wait_for_timeout(250)
                return True
            except Exception as exc:
                errors.append(f"{action_name}: {exc}")
                return False

        for action in [str(a).strip().lower() for a in actions]:
            if action == "hold":
                _click_first(page.get_by_test_id("action-hold"), "hold")
            elif action == "track":
                _click_first(page.get_by_test_id("action-track"), "track")
            elif action == "refresh_bookings":
                _click_first(page.get_by_test_id("booking-refresh"), "refresh_bookings")
            elif action == "refresh_alerts":
                _click_first(page.get_by_test_id("alerts-refresh"), "refresh_alerts")
            elif action == "cancel_latest":
                _click_first(page.get_by_test_id("booking-cancel"), "cancel_latest")
            elif action == "ack_first_alert":
                _click_first(page.get_by_test_id("alert-ack"), "ack_first_alert")
            elif action == "ack_first_alert_if_present":
                locator = page.get_by_test_id("alert-ack")
                if locator.count() <= 0:
                    performed.append("ack_first_alert_if_present:skipped")
                    ui_actions.append("post_action:ack_first_alert_if_present:skipped")
                else:
                    _click_first(locator, "ack_first_alert_if_present")
            elif action == "cancel_async_job":
                _click_first(page.get_by_test_id("stream-cancel"), "cancel_async_job")
            else:
                errors.append(f"{action}: unsupported post action")

        return {"performed": performed, "errors": errors, "endpoint_summary": self._endpoint_summary(endpoint_records)}

    def _mock_base_plan_payload(self, payload: Dict[str, object], *, include_handoff: bool = False) -> Dict[str, Any]:
        origin = str(payload.get("origin") or "DEL").upper()
        destination = str(payload.get("destination") or "BOM").upper()
        search_date = str(payload.get("date") or "2026-03-20")

        best_flight: Dict[str, Any] = {
            "airline": "MockAir",
            "flight_no": "MK101",
            "departure_time": "09:20",
            "arrival_time": "11:35",
            "price_inr": 6200,
            "duration_min": 135,
            "stops": 0,
            "baggage": "7kg cabin",
            "date": search_date,
        }
        if include_handoff:
            best_flight["handoff_url"] = "/booking/handoff/post/mock-artifact-001"

        alt_flight = {
            "airline": "MockJet",
            "flight_no": "MJ240",
            "departure_time": "12:10",
            "arrival_time": "14:50",
            "price_inr": 6750,
            "duration_min": 160,
            "stops": 1,
            "baggage": "15kg check-in",
            "layover_info": "45m HYD",
            "date": search_date,
        }

        all_flights = [best_flight, alt_flight]
        return {
            "best_flight": best_flight,
            "all_flights": all_flights,
            "weather": {
                "location": destination,
                "condition": "Clear",
                "temperature_c": 29,
                "temp_min_c": 24,
                "temp_max_c": 32,
            },
            "llm_response": f"Route {origin} to {destination}: MockAir MK101 is the strongest balance of timing and fare.",
            "search_date": search_date,
            "result_status": "success",
            "debug_info": {
                "intent": {"origin_iata": origin, "destination_iata": destination, "date": search_date},
                "all_flights": all_flights,
                "agent_reasoning": [
                    "Selected non-stop candidate for higher schedule reliability.",
                    "Included weather context for packing guidance.",
                ],
            },
        }

    def _mock_stream_done_payload(self, scenario: str, payload: Dict[str, object]) -> Dict[str, Any]:
        normalized = self._resolve_fixture_scenario_name(scenario) or (scenario or "").strip().lower()
        include_handoff = normalized in {"fixture_booking_handoff", "mock_booking_handoff"}
        base = self._mock_base_plan_payload(payload, include_handoff=include_handoff)
        if normalized in {"fixture_stream_round_trip", "mock_stream_success_round_trip"}:
            base["return_trip"] = {
                "best_flight": {
                    "airline": "MockAir",
                    "flight_no": "MK102",
                    "departure_time": "17:20",
                    "arrival_time": "19:25",
                    "price_inr": 6400,
                    "duration_min": 125,
                    "stops": 0,
                    "baggage": "7kg cabin",
                    "date": "2026-03-23",
                },
                "weather": {"condition": "Partly cloudy", "temperature_c": 27},
                "warnings": [],
            }
            base["llm_response"] += " Return leg is included with a practical evening departure."
            return base

        if normalized in {"fixture_stream_via_stopover", "mock_stream_success_via_stopover"}:
            base["multicity"] = True
            leg_one = {
                "llm_response": "Leg 1 DEL to BLR prioritizes a short-duration morning segment with stable weather.",
                "best_flight": {
                    "airline": "MockAir",
                    "flight_no": "MK301",
                    "departure_time": "07:10",
                    "arrival_time": "09:40",
                    "price_inr": 5100,
                    "duration_min": 150,
                    "stops": 0,
                    "baggage": "7kg cabin",
                    "date": str(payload.get("date") or "2026-03-20"),
                },
                "weather": {"location": "BLR", "condition": "Cloudy", "temperature_c": 24},
                "debug_info": {"intent": {"origin_iata": "DEL", "destination_iata": "BLR"}},
            }
            leg_two = {
                "llm_response": "Leg 2 BLR to MAA continues the route with a low-delay window and weather-aware timing.",
                "best_flight": {
                    "airline": "MockJet",
                    "flight_no": "MJ302",
                    "departure_time": "12:30",
                    "arrival_time": "13:40",
                    "price_inr": 3200,
                    "duration_min": 70,
                    "stops": 0,
                    "baggage": "7kg cabin",
                    "date": str(payload.get("date") or "2026-03-20"),
                },
                "weather": {"location": "MAA", "condition": "Sunny", "temperature_c": 31},
                "debug_info": {"intent": {"origin_iata": "BLR", "destination_iata": "MAA"}},
            }
            base["legs"] = [leg_one, leg_two]
            base["llm_response"] = (
                "Multi-city itinerary is ready with route-aware evidence and weather context for each leg."
            )
            return base

        if normalized in {"fixture_degraded_result", "mock_degraded_result"}:
            base["result_status"] = "degraded"
            base["degradation"] = {
                "reason": "upstream_unavailable",
                "message": "LLM explanation backend unavailable; structured itinerary preserved.",
            }
            base["fallback_note"] = (
                "LLM explanation degraded (upstream_unavailable): structured itinerary remains usable."
            )
            return base

        if normalized == "fixture_booking_local_only":
            if isinstance(base.get("best_flight"), dict):
                base["best_flight"].pop("handoff_url", None)
                base["best_flight"]["booking_handoff"] = {
                    "status": "unavailable",
                    "booking_exit_quality": "deferred",
                    "reason": "provider_handoff_unavailable",
                }
            base["warnings"] = [
                "Checkout is unavailable for the selected provider artifacts; this can only be held locally."
            ]
            return base

        if normalized == "fixture_cabin_business_no_match":
            if isinstance(base.get("best_flight"), dict):
                base["best_flight"]["travel_class"] = "Economy"
            base["warnings"] = [
                "No business class inventory matched your request for this route/date. Showing closest available economy options."
            ]
            return base

        if normalized == "fixture_direct_truthful":
            base["warnings"] = [
                "No true nonstop inventory matched all constraints. Showing best available alternatives."
            ]
            return base

        # default one-way / booking / fallback success payloads
        return base

    def _mock_non_stream_payload(self, scenario: str, payload: Dict[str, object]) -> Tuple[int, Dict[str, Any]]:
        normalized = self._resolve_fixture_scenario_name(scenario) or (scenario or "").strip().lower()
        if normalized in {"fixture_no_flights", "mock_no_flights"}:
            return (
                400,
                {
                    "detail": "No matching flights found for this route/date. Try a nearby date or a different route.",
                    "failure_reason": "no_flights",
                    "failure_domain": "search_outcome",
                    "no_flights_reason": "no_inventory",
                    "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
                    "result_status": "error",
                },
            )
        done_payload = self._mock_stream_done_payload(normalized, payload)
        return 200, done_payload

    def _mock_stream_sse_body(
        self,
        *,
        chunks: List[str],
        done_payload: Optional[Dict[str, Any]] = None,
        error_text: str = "",
    ) -> str:
        frames: List[str] = []
        for chunk in chunks:
            frames.append(f"data: {chunk}\n\n")
        if error_text:
            frames.append(f"data: [ERROR] {error_text}\n\n")
        if done_payload is not None:
            frames.append("data: [DONE_JSON]" + json.dumps(done_payload, ensure_ascii=False) + "\n\n")
        frames.append("event: done\ndata: \n\n")
        return "".join(frames)

    def _install_fixture_routes(self, page, scenario: str, payload: Dict[str, object]) -> None:
        scenario_name = self._resolve_fixture_scenario_name(scenario) or "fixture_stream_one_way"
        fixture = self.fixture_catalog.get(scenario_name) or self.fixture_catalog["fixture_stream_one_way"]
        self._debug(f"fixture_route_enabled scenario={scenario_name} ask_mode={fixture.ask_mode}")

        state: Dict[str, Any] = {
            "bookings": [deepcopy(row) for row in fixture.initial_bookings],
            "alerts": [deepcopy(row) for row in fixture.initial_alerts],
            "next_booking_id": max([int(row.get("id", 0)) for row in fixture.initial_bookings] + [100]),
            "job_id": "job-fixture-001",
            "job_status": "queued",
            "job_poll_count": 0,
        }

        def _json(route, status_code: int, body_obj: Dict[str, Any]) -> None:
            route.fulfill(
                status=status_code,
                headers={"Content-Type": "application/json; charset=utf-8"},
                body=json.dumps(body_obj, ensure_ascii=False),
            )

        def _job_result_payload() -> Dict[str, Any]:
            return self._mock_stream_done_payload("fixture_non_stream_one_way", payload)

        def _create_booking(req_payload: Dict[str, Any], *, for_tracking: bool = False) -> Dict[str, Any]:
            state["next_booking_id"] += 1
            flight = req_payload.get("flight") if isinstance(req_payload.get("flight"), dict) else {}
            booking = {
                "id": state["next_booking_id"],
                "status": "HELD",
                "checkout_ready": False,
                "checkout_status": "provider_handoff_unavailable",
                "hold_outcome": "held_local_only",
                "handoff_url": None,
                "flight": {
                    "airline": str(flight.get("airline") or "MockAir"),
                    "flight_no": str(flight.get("flight_no") or "MK101"),
                    "origin": str(req_payload.get("origin") or "DEL"),
                    "destination": str(req_payload.get("destination") or "BOM"),
                    "departure_time": str(flight.get("departure_time") or "09:20"),
                    "arrival_time": str(flight.get("arrival_time") or "11:35"),
                    "date": str(req_payload.get("depart_date") or req_payload.get("date") or "2026-07-18"),
                    "price_inr": int(flight.get("price_inr") or 6200),
                },
                "monitoring_active": bool(for_tracking),
            }
            state["bookings"].insert(0, booking)
            return booking

        def _route_handler(route, request) -> None:
            method = str(request.method or "").upper()
            parsed = urlparse(request.url)
            path = parsed.path or ""
            query = parse_qs(parsed.query or "")

            if method == "GET" and path.endswith("/health"):
                _json(route, 200, {"status": "ok"})
                return
            if method == "GET" and path.endswith("/llm/options"):
                _json(
                    route,
                    200,
                    {
                        "llm_modes": ["ollama_only", "cloud_only", "cloud_first", "ollama_first"],
                        "cloud_providers": ["gemini", "openai"],
                        "defaults": {
                            "llm_mode": "ollama_first",
                            "cloud_provider": "gemini",
                        },
                        "usable_cloud_providers": ["gemini", "openai"],
                        "backend_availability": {"cloud": True, "ollama": True},
                        "provider_switch_enabled": True,
                        "effective_default_provider": "gemini",
                        "effective_mode": "ollama_first",
                    },
                )
                return
            if method == "GET" and path.endswith("/version"):
                _json(route, 200, {"git_commit": "frontend-fixture", "file_mtime": 1712810000})
                return
            if method == "GET" and path.endswith("/bookings"):
                _json(route, 200, {"items": state["bookings"]})
                return
            if method == "POST" and path.endswith("/booking/hold"):
                req_payload = self._safe_json_loads(request.post_data or "") or {}
                booking = _create_booking(req_payload, for_tracking=False)
                response = deepcopy(fixture.hold_response) if isinstance(fixture.hold_response, dict) else {
                    "action": "hold",
                    "success": True,
                    "checkout_ready": False,
                    "hold_outcome": "held_local_only",
                    "message": "Held locally. Provider checkout is unavailable.",
                }
                response["booking"] = booking
                _json(route, 200, response)
                return
            if method == "POST" and path.endswith("/booking/track-price"):
                req_payload = self._safe_json_loads(request.post_data or "") or {}
                booking = _create_booking(req_payload, for_tracking=True)
                response = deepcopy(fixture.track_response) if isinstance(fixture.track_response, dict) else {
                    "action": "track-price",
                    "success": True,
                    "monitoring_active": True,
                    "message": "Price tracking activated.",
                }
                response["booking"] = booking
                _json(route, 200, response)
                return
            if method == "POST" and path.endswith("/booking/cancel"):
                req_payload = self._safe_json_loads(request.post_data or "") or {}
                booking_id = int(req_payload.get("booking_id") or 0)
                for row in state["bookings"]:
                    if int(row.get("id") or 0) == booking_id:
                        row["status"] = "CANCELLED"
                response = deepcopy(fixture.cancel_response) if isinstance(fixture.cancel_response, dict) else {
                    "action": "cancel",
                    "success": True,
                    "message": "Booking cancelled.",
                }
                _json(route, 200, response)
                return
            if method == "GET" and path.endswith("/price-tracking/status"):
                status_payload = fixture.tracking_status or {"enabled": False, "status": {}}
                _json(route, 200, status_payload)
                return
            if method == "GET" and path.endswith("/price-tracking/alerts"):
                _json(route, 200, {"items": state["alerts"]})
                return
            if method == "POST" and re.search(r"/price-tracking/alerts/[^/]+/ack$", path):
                m = re.search(r"/price-tracking/alerts/([^/]+)/ack$", path)
                token = m.group(1) if m else ""
                alert_id: Optional[int] = None
                try:
                    alert_id = int(token)
                except Exception:
                    for row in state["alerts"]:
                        if not isinstance(row, dict):
                            continue
                        try:
                            alert_id = int(row.get("alert_id") or -1)
                        except Exception:
                            continue
                        if alert_id >= 0:
                            break
                filtered_alerts: List[Dict[str, Any]] = []
                for row in state["alerts"]:
                    if not isinstance(row, dict):
                        continue
                    if alert_id is None:
                        filtered_alerts.append(row)
                        continue
                    try:
                        row_alert_id = int(row.get("alert_id") or -1)
                    except Exception:
                        filtered_alerts.append(row)
                        continue
                    if row_alert_id != alert_id:
                        filtered_alerts.append(row)
                state["alerts"] = filtered_alerts
                _json(route, 200, {"acknowledged": True})
                return
            if method == "POST" and path.endswith("/ask") and "async_job" in query:
                state["job_status"] = "running"
                _json(route, 202, {"job_id": state["job_id"]})
                return
            if method == "GET" and re.search(r"/jobs/[^/]+/events$", path):
                frames = [
                    "event: queued\n"
                    f"data: {json.dumps({'event': 'queued', 'status': 'queued', 'job_id': state['job_id']})}\n\n",
                    "event: running\n"
                    f"data: {json.dumps({'event': 'running', 'status': 'running', 'job_id': state['job_id']})}\n\n",
                ]
                if fixture.ask_mode != "async_job_running":
                    frames.append(
                        "event: done\n"
                        f"data: {json.dumps({'event': 'done', 'status': 'done', 'job_id': state['job_id'], 'result': _job_result_payload()})}\n\n"
                    )
                body = "".join(frames)
                route.fulfill(
                    status=200,
                    headers={"Content-Type": "text/event-stream; charset=utf-8"},
                    body=body,
                )
                return
            if method == "GET" and re.search(r"/jobs/[^/]+$", path):
                state["job_poll_count"] += 1
                if fixture.ask_mode == "async_job_running":
                    state["job_status"] = "running"
                else:
                    state["job_status"] = "done" if state["job_poll_count"] >= 2 else "running"
                _json(
                    route,
                    200,
                    {
                        "job_id": state["job_id"],
                        "status": state["job_status"],
                        "result": _job_result_payload() if state["job_status"] == "done" else None,
                    },
                )
                return
            if method == "POST" and re.search(r"/jobs/[^/]+/cancel$", path):
                state["job_status"] = "cancelled"
                _json(route, 200, {"job": {"job_id": state["job_id"], "status": "cancelled"}})
                return

            if method == "POST" and path.endswith("/ask"):
                is_stream = str(query.get("stream", [""])[0]).lower() in {"true", "1", "yes"}
                if fixture.ask_mode in {"stream_fallback_non_stream", "stream_fallback_no_flights"} and is_stream:
                    body = self._mock_stream_sse_body(chunks=[], error_text="LLM temporarily unavailable")
                    route.fulfill(
                        status=200,
                        headers={"Content-Type": "text/event-stream; charset=utf-8"},
                        body=body,
                    )
                    return

                if fixture.ask_mode in {"non_stream", "stream_fallback_non_stream", "stream_fallback_no_flights"} and not is_stream:
                    status_code, json_payload = self._mock_non_stream_payload(scenario_name, payload)
                    _json(route, status_code, json_payload)
                    return

                if is_stream:
                    done_payload = self._mock_stream_done_payload(scenario_name, payload)
                    body = self._mock_stream_sse_body(
                        chunks=[
                            "Analyzing route options with fixture availability...",
                            "Ranking flights and layering destination weather...",
                        ],
                        done_payload=done_payload,
                    )
                    route.fulfill(
                        status=200,
                        headers={"Content-Type": "text/event-stream; charset=utf-8"},
                        body=body,
                    )
                    return

                _json(route, 200, self._mock_stream_done_payload(scenario_name, payload))
                return

            route.continue_()

        page.route("**/*", _route_handler)

    def _parse_done_json_from_sse_body(self, body_text: str) -> Dict[str, Any]:
        marker = "[DONE_JSON]"
        marker_seen = marker in body_text
        result: Dict[str, Any] = {
            "marker_seen": marker_seen,
            "frame_found": False,
            "event_name": "",
            "done_json": None,
            "error": "",
        }

        if not marker_seen:
            result["error"] = "DONE_JSON marker missing"
            return result

        frames = body_text.replace("\r\n", "\n").split("\n\n")
        for raw_frame in frames:
            frame = raw_frame.strip("\n")
            if not frame.strip():
                continue

            event_name = "message"
            data_lines: List[str] = []
            for line in frame.split("\n"):
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip() or "message"
                    continue
                if line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].lstrip())

            data_payload = "\n".join(data_lines)
            if marker not in data_payload:
                continue

            result["frame_found"] = True
            result["event_name"] = event_name
            done_segment = data_payload.split(marker, 1)[1].strip()
            parsed_done = self._safe_json_loads(done_segment)
            result["done_json"] = parsed_done
            if parsed_done is None:
                result["error"] = f"DONE_JSON marker found in SSE data for event '{event_name}' but JSON parse failed"
            return result

        result["error"] = "DONE_JSON marker found in stream body but not within SSE data frame payload"
        return result

    def _payload_matches_request_payload(self, payload: Dict[str, object], request_payload: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(request_payload, dict):
            return False
        for key, expected in payload.items():
            if str(request_payload.get(key, "")).strip() != str(expected).strip():
                return False
        return True

    def _sanitize_request_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean: List[Dict[str, Any]] = []
        for rec in records:
            done_json = rec.get("stream_done_json")
            done_json_keys: List[str] = []
            if isinstance(done_json, dict):
                done_json_keys = sorted([str(key) for key in done_json.keys()])[:20]
            clean.append({
                "url": rec.get("url"),
                "method": rec.get("method"),
                "is_stream": rec.get("is_stream"),
                "matches_payload": rec.get("matches_payload"),
                "response_status": rec.get("response_status"),
                "response_ok": rec.get("response_ok"),
                "completed": rec.get("completed"),
                "failed": rec.get("failed"),
                "failure_text": rec.get("failure_text"),
                "request_payload": rec.get("request_payload"),
                "response_body_preview": rec.get("response_body_preview"),
                "response_body_json": rec.get("response_body_json"),
                "stream_done_marker_checked": rec.get("stream_done_marker_checked"),
                "stream_done_marker_seen": rec.get("stream_done_marker_seen"),
                "stream_done_frame_found": rec.get("stream_done_frame_found"),
                "stream_done_event": rec.get("stream_done_event"),
                "stream_done_json_keys": done_json_keys,
                "stream_done_json_parsed": rec.get("stream_done_json_parsed"),
                "stream_done_json_error": rec.get("stream_done_json_error"),
            })
        return clean

    def _network_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        matched = [rec for rec in records if rec.get("matches_payload")]
        analysis_records = matched if matched else list(records)
        statuses = [
            rec.get("response_status")
            for rec in analysis_records
            if rec.get("response_status") is not None
        ]

        stream_recs = [rec for rec in analysis_records if rec.get("is_stream")]
        fallback_recs = [rec for rec in analysis_records if not rec.get("is_stream")]

        def successful_completed(rec: Dict[str, Any]) -> bool:
            status = rec.get("response_status")
            return bool(rec.get("completed") and not rec.get("failed") and isinstance(status, int) and 200 <= status < 300)

        summary = {
            "request_fired": len(records) > 0,
            "payload_matched_request_fired": len(matched) > 0,
            "request_completed": any(rec.get("completed") for rec in analysis_records),
            "request_completed_success": any(successful_completed(rec) for rec in analysis_records),
            "stream_request_fired": len(stream_recs) > 0,
            "stream_request_success": any(successful_completed(rec) for rec in stream_recs),
            "stream_done_marker_checked": any(bool(rec.get("stream_done_marker_checked")) for rec in stream_recs),
            "stream_done_marker_seen": any(bool(rec.get("stream_done_marker_seen")) for rec in stream_recs),
            "stream_done_frame_found": any(bool(rec.get("stream_done_frame_found")) for rec in stream_recs),
            "stream_done_events": sorted(
                {
                    str(rec.get("stream_done_event"))
                    for rec in stream_recs
                    if str(rec.get("stream_done_event") or "").strip()
                }
            ),
            "stream_done_json_parsed": any(bool(rec.get("stream_done_json_parsed")) for rec in stream_recs),
            "fallback_request_fired": len(fallback_recs) > 0,
            "fallback_request_success": any(successful_completed(rec) for rec in fallback_recs),
            "request_count": len(records),
            "matched_request_count": len(matched),
            "matched_statuses": statuses,
            "matched_url_statuses": [
                {"url": rec.get("url"), "status": rec.get("response_status")}
                for rec in analysis_records
            ],
        }
        summary["streaming_completed"] = bool(
            not summary["stream_request_fired"]
            or summary["stream_done_json_parsed"]
            or summary["fallback_request_success"]
        )
        return summary

    def _is_clean_page_state(self, snapshot: Dict[str, Any]) -> bool:
        return bool(
            snapshot.get("flights_count", 0) == 0
            and not snapshot.get("weather_present")
            and not snapshot.get("best_flight_visible")
            and not snapshot.get("trip_brief_present")
            and not snapshot.get("reasoning_present")
            and not snapshot.get("error_visible")
        )

    def _result_signature(self, snapshot: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(snapshot.get("flights_count", 0)),
                str(bool(snapshot.get("weather_present"))),
                str(bool(snapshot.get("best_flight_visible"))),
                str(bool(snapshot.get("reasoning_present"))),
                str(bool(snapshot.get("trip_brief_present"))),
                str(snapshot.get("stream_text", ""))[:220],
                str(snapshot.get("flight_text", ""))[:220],
                str(snapshot.get("weather_text", ""))[:220],
                str(snapshot.get("highlight_text", ""))[:220],
                str(snapshot.get("error_text", ""))[:220],
            ]
        )

    def _is_fresh_result(self, pre_snapshot: Dict[str, Any], post_snapshot: Dict[str, Any]) -> bool:
        return self._result_signature(pre_snapshot) != self._result_signature(post_snapshot)

    def _extract_query_markers(self, payload: Dict[str, object], user_query: str) -> List[str]:
        markers: set[str] = set()

        code_to_city = {
            "DEL": "delhi",
            "BOM": "mumbai",
            "BLR": "bangalore",
            "MAA": "chennai",
            "GOI": "goa",
            "HYD": "hyderabad",
            "CCU": "kolkata",
        }

        def push_location(value: str) -> None:
            cleaned = value.strip()
            if not cleaned:
                return
            upper = cleaned.upper()
            lower = cleaned.lower()
            if re.fullmatch(r"[A-Z]{3}", upper):
                markers.add(upper.lower())
                city = code_to_city.get(upper)
                if city:
                    markers.add(city)
            markers.add(lower)
            first_word = re.split(r"[^a-zA-Z]+", lower)[0]
            if first_word:
                markers.add(first_word)

        for key in ("origin", "destination"):
            raw = payload.get(key)
            if isinstance(raw, str):
                push_location(raw)

        known_geo = [
            "delhi", "mumbai", "chennai", "bangalore", "bengaluru", "goa", "hyderabad", "kolkata",
            "bombay", "calcutta", "dilli", "del", "bom", "maa", "blr", "hyd", "ccu",
        ]
        q = user_query.lower()
        for token in known_geo:
            if re.search(rf"\b{re.escape(token)}\b", q):
                markers.add(token)

        # Filter short noisy tokens while keeping IATA codes.
        normalized = [m for m in markers if len(m) >= 4 or re.fullmatch(r"[a-z]{3}", m)]
        return sorted(set(normalized), key=lambda item: (-len(item), item))

    def _build_source_payload_alignment(self, source_payload: Dict[str, object], form_state: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = [
            key
            for key in ("user_query", "origin", "destination", "date", "trip_type", "return_date", "direct_only", "cabin", "baggage_pref")
            if source_payload.get(key) not in (None, "", False)
        ]
        missing: List[str] = []
        mismatched: List[str] = []
        for key in required_keys:
            if key not in form_state or not str(form_state.get(key, "")).strip():
                if key == "direct_only" and key in form_state:
                    # direct_only can legitimately be False, so treat presence as sufficient.
                    pass
                else:
                    missing.append(key)
                    continue
            form_value = form_state.get(key, "")
            source_value = source_payload.get(key, "")
            if key == "direct_only":
                if bool(form_value) != bool(source_value):
                    mismatched.append(key)
                continue
            form_value = str(form_value).strip()
            source_value = str(source_value).strip()
            if key == "trip_type":
                form_value = self._normalize_trip_type(form_value) or form_value
                source_value = self._normalize_trip_type(source_value) or source_value
            if form_value != source_value:
                mismatched.append(key)
        return {
            "matches_source": not missing and not mismatched,
            "missing_keys": missing,
            "mismatched_keys": mismatched,
            "source_payload": source_payload,
        }

    def _query_linked_evidence(
        self,
        payload: Dict[str, object],
        user_query: str,
        snapshot: Dict[str, Any],
        is_multi_leg_query: bool,
    ) -> Dict[str, Any]:
        sections = {
            "stream": str(snapshot.get("stream_text", "")),
            "reasoning": str(snapshot.get("reasoning_text", "")),
            "weather": str(snapshot.get("weather_text", "")),
            "flight": str(snapshot.get("flight_text", "")),
            "highlight": str(snapshot.get("highlight_text", "")),
            "booking_panel": str(snapshot.get("booking_panel_text", "")),
            "tracking_panel": str(snapshot.get("tracking_panel_text", "")),
        }
        corpus_text = " ".join(sections.values())
        corpus = corpus_text.lower()
        markers = self._extract_query_markers(payload, user_query)
        hits = sorted([marker for marker in markers if marker in corpus])
        section_hits: Dict[str, List[str]] = {}
        for section_name, section_text in sections.items():
            section_lower = section_text.lower()
            section_hits[section_name] = [marker for marker in markers if marker in section_lower]
        sections_with_hits = [section for section, vals in section_hits.items() if vals]
        rich_sections = [name for name, value in sections.items() if str(value or "").strip()]
        artifact_signal_count = len(
            [name for name in ("stream", "reasoning", "weather", "flight", "highlight") if str(sections.get(name) or "").strip()]
        )
        route_code_markers = {
            str(payload.get("origin") or "").strip().lower(),
            str(payload.get("destination") or "").strip().lower(),
        }
        route_code_markers = {marker for marker in route_code_markers if marker}
        route_code_hits = sorted([marker for marker in route_code_markers if marker in corpus])

        query_l = user_query.lower()
        intent_markers: List[str] = []
        if re.search(r"\b(direct|nonstop|non-stop)\b", query_l):
            intent_markers.extend(["direct", "nonstop", "non-stop"])
        if re.search(r"\b(business|first class|premium economy|economy class|cabin)\b", query_l):
            intent_markers.extend(["business", "first class", "premium economy", "economy", "cabin"])
        if re.search(r"\b(cabin baggage|hand baggage|checked bag|luggage)\b", query_l):
            intent_markers.extend(["baggage", "cabin baggage", "hand baggage", "checked"])
        intent_hits = sorted({marker for marker in intent_markers if marker in corpus})

        required_hits = 3 if is_multi_leg_query else 2
        required_sections = 2 if is_multi_leg_query else 1
        strict_marker_strength_ok = (
            (len(hits) >= required_hits and len(sections_with_hits) >= required_sections)
            if markers
            else len(corpus.strip()) >= 120
        )
        fallback_marker_strength_ok = bool(
            len(hits) >= 1
            and len(route_code_hits) >= 1
            and artifact_signal_count >= (3 if not is_multi_leg_query else 4)
            and len(rich_sections) >= 3
        )
        marker_strength_ok = bool(strict_marker_strength_ok or fallback_marker_strength_ok)
        intent_strength_ok = True if not intent_markers else len(intent_hits) > 0
        ok = bool(marker_strength_ok and intent_strength_ok)
        excerpt = corpus_text[:320]
        return {
            "markers": markers,
            "hits": hits,
            "hit_count": len(hits),
            "sections_with_hits": sections_with_hits,
            "section_hits": section_hits,
            "required_hits": required_hits,
            "required_sections": required_sections,
            "strict_marker_strength_ok": strict_marker_strength_ok,
            "fallback_marker_strength_ok": fallback_marker_strength_ok,
            "artifact_signal_count": artifact_signal_count,
            "rich_sections": rich_sections,
            "route_code_hits": route_code_hits,
            "intent_markers": sorted(set(intent_markers)),
            "intent_hits": intent_hits,
            "ok": ok,
            "corpus_excerpt": excerpt,
        }

    def _evaluate_case(
        self,
        *,
        snapshot: Dict[str, Any],
        network_summary: Dict[str, Any],
        endpoint_summary: Dict[str, Any],
        ui_reset_performed: bool,
        freshness_ok: bool,
        query_evidence: Dict[str, Any],
        is_multi_leg_query: bool,
        timed_out: bool,
        payload_parity: Dict[str, Any],
        source_payload_alignment: Dict[str, Any],
        validation_scenario: str,
        validation_expectations: Dict[str, Any],
        request_records: List[Dict[str, Any]],
        post_actions_result: Dict[str, Any],
    ) -> tuple[bool, str]:
        scenario = (validation_scenario or "").strip().lower()
        normalized = self._resolve_fixture_scenario_name(scenario) or scenario
        is_no_flights = normalized in {"fixture_no_flights", "mock_no_flights"} or bool(validation_expectations.get("expect_no_flights"))
        is_fallback = normalized in {"fixture_stream_fallback_non_stream", "mock_stream_fallback_non_stream"}
        is_degraded = normalized in {"fixture_degraded_result", "mock_degraded_result"} or bool(validation_expectations.get("expect_degraded"))
        is_round_trip = normalized in {"fixture_stream_round_trip", "mock_stream_success_round_trip"} or bool(validation_expectations.get("expect_round_trip"))
        is_via_stopover = normalized in {"fixture_stream_via_stopover", "mock_stream_success_via_stopover"} or bool(validation_expectations.get("expect_via_stopover"))
        is_booking = normalized in {"fixture_booking_handoff", "mock_booking_handoff"} or bool(validation_expectations.get("expect_booking_link"))
        expects_stream = bool(validation_expectations.get("expect_stream_request", not normalized.startswith("fixture_non_stream"))) or is_fallback or is_degraded or is_no_flights or is_booking
        allow_sparse_result = bool(validation_expectations.get("allow_sparse_result"))
        enforce_payload_parity = bool(validation_expectations.get("enforce_payload_parity"))
        require_seller_or_handoff = bool(validation_expectations.get("require_seller_or_handoff_signal"))
        expect_booking_panel = bool(validation_expectations.get("expect_booking_panel"))
        accept_any_outcome = validation_expectations.get("accept_any_outcome") if isinstance(validation_expectations, dict) else []
        accept_any_outcome_set = set(accept_any_outcome) if isinstance(accept_any_outcome, list) else set()

        if not snapshot.get("planner_form_ready"):
            return False, "Planner form did not load correctly."
        if not ui_reset_performed:
            return False, "UI reset failed before query submit."
        if not network_summary["request_fired"]:
            return False, "No /ask request was fired for this query."
        if enforce_payload_parity and not network_summary["payload_matched_request_fired"]:
            return False, "No /ask request payload matched the submitted query payload."
        if not network_summary["request_completed_success"]:
            return False, (
                "No successful completed /ask request for this query "
                f"(statuses={network_summary['matched_statuses']})."
            )
        if expects_stream and not network_summary["stream_request_fired"]:
            return False, "Expected stream request was not fired."
        if network_summary["stream_request_fired"] and not (
            network_summary["stream_request_success"] or network_summary["fallback_request_success"]
        ):
            return False, "Streaming started but neither stream completion nor fallback completion succeeded."
        if network_summary["stream_request_fired"] and network_summary["stream_done_marker_checked"] and not (
            network_summary["stream_done_json_parsed"] or network_summary["fallback_request_success"]
        ):
            if not (is_no_flights and network_summary["fallback_request_fired"]):
                return False, "Streaming response ended without parseable DONE_JSON completion payload."
        if is_fallback and not network_summary["fallback_request_success"]:
            return False, "Fallback non-stream request did not complete successfully."
        if is_no_flights and not network_summary["fallback_request_fired"]:
            return False, "No-flights scenario did not exercise stream-to-fallback request path."
        if timed_out:
            return False, "Timed out before request+UI completion."
        if snapshot["is_busy"]:
            return False, "UI still busy after request completion window."
        if snapshot["error_visible"] and not is_no_flights and not accept_any_outcome_set:
            return False, snapshot["error_text"] or "Error banner visible in UI."
        if not freshness_ok and not allow_sparse_result:
            return False, "Rendered result did not change from pre-submit state."
        if enforce_payload_parity and not source_payload_alignment.get("matches_source"):
            return False, (
                "Submitted UI fields do not match source payload "
                f"(missing={source_payload_alignment.get('missing_keys')}, "
                f"mismatched={source_payload_alignment.get('mismatched_keys')})."
            )
        if enforce_payload_parity and not payload_parity.get("matches_expected"):
            return False, (
                "Captured /ask payload does not match intended UI submission "
                f"(missing={payload_parity.get('missing_keys')}, mismatched={payload_parity.get('mismatched_keys')})."
            )
        action_errors = post_actions_result.get("errors") if isinstance(post_actions_result, dict) else []
        if isinstance(action_errors, list) and action_errors:
            return False, f"Post-submit UI actions failed: {action_errors}"
        required_endpoint_calls = validation_expectations.get("required_endpoint_calls") if isinstance(validation_expectations, dict) else []
        if isinstance(required_endpoint_calls, list) and required_endpoint_calls:
            calls = (endpoint_summary or {}).get("calls") if isinstance(endpoint_summary, dict) else {}
            missing_calls = [name for name in required_endpoint_calls if int((calls or {}).get(name, 0)) <= 0]
            if missing_calls:
                return False, f"Expected endpoint calls missing: {missing_calls}"
        optional_endpoint_calls = validation_expectations.get("optional_endpoint_calls") if isinstance(validation_expectations, dict) else []
        if isinstance(optional_endpoint_calls, list) and optional_endpoint_calls:
            calls = (endpoint_summary or {}).get("calls") if isinstance(endpoint_summary, dict) else {}
            missing_optional = [name for name in optional_endpoint_calls if int((calls or {}).get(name, 0)) <= 0]
            if missing_optional:
                self._debug(f"optional_endpoint_calls_missing={missing_optional}")
        if not query_evidence["ok"] and not is_no_flights and not allow_sparse_result:
            return False, (
                "Rendered result lacks query-linked evidence "
                f"(markers={query_evidence['markers']}, hits={query_evidence['hits']}, "
                f"required={query_evidence['required_hits']}, sections={query_evidence.get('required_sections')})."
            )
        if require_seller_or_handoff and not (
            snapshot.get("seller_signal_visible")
            or snapshot.get("booking_link_visible")
            or snapshot.get("provider_handoff_hint_visible")
        ):
            return False, "Seller/OTA diversity or provider handoff signal was not visible in UI."
        if expect_booking_panel and not snapshot.get("booking_panel_visible"):
            return False, "Booking panel was expected but not visible."

        if accept_any_outcome_set:
            observed_outcomes = set()
            if snapshot.get("no_flights_error_visible"):
                observed_outcomes.add("no_flights")
            if snapshot.get("degraded_notice_visible"):
                observed_outcomes.add("degraded")
            notice_blob = " ".join(snapshot.get("notice_texts") or []).lower()
            if any(token in notice_blob for token in ("constraint", "unavailable", "closest", "no matching")):
                observed_outcomes.add("success_with_constraint_notice")
            if not (accept_any_outcome_set & observed_outcomes):
                return False, f"Expected one accepted outcome {sorted(accept_any_outcome_set)}, observed={sorted(observed_outcomes)}"

        if allow_sparse_result:
            return True, ""
        if is_no_flights:
            if not snapshot.get("error_visible"):
                return False, "No-flights scenario expected an error notice in UI."
            if not snapshot.get("no_flights_error_visible"):
                return False, "No-flights scenario error text was not explicit/truthful."
            if not snapshot.get("proof_overview_visible"):
                return False, "No-flights scenario did not render proof/status surface."
            if not snapshot.get("ranked_shortlist_visible"):
                return False, "No-flights scenario did not keep shortlist section visible."
            return True, ""

        if not snapshot["trip_brief_present"]:
            return False, "Trip brief/narrative text not visible."
        if not snapshot["reasoning_present"]:
            return False, "Reasoning panel content not visible."
        if not snapshot.get("proof_overview_visible"):
            return False, "Proof overview section is missing."
        if not snapshot.get("proof_evidence_visible"):
            return False, "Proof evidence list is missing."
        if not snapshot.get("ranked_shortlist_visible"):
            return False, "Ranked shortlist section label is missing."

        if is_multi_leg_query:
            # Multi-leg can surface in narrative form; require rich, route-specific narrative evidence.
            rich_narrative = len(snapshot.get("stream_text", "").strip()) >= 140
            has_structured_cards = bool(
                snapshot["best_flight_visible"]
                or snapshot["flights_count"] > 0
                or snapshot["weather_present"]
            )
            if not (rich_narrative or has_structured_cards):
                return False, "Multi-leg query did not produce rich route-specific output."
            if is_via_stopover and not snapshot.get("multicity_visible"):
                return False, "Via/stopover scenario did not render multi-city itinerary section."
            return True, ""

        if not (snapshot["best_flight_visible"] or snapshot["flights_count"] > 0):
            return False, "No flight results or best-flight marker visible."
        if not snapshot["weather_present"]:
            return False, "Weather panel data not visible."

        if is_round_trip and not snapshot.get("return_leg_visible"):
            return False, "Round-trip scenario did not render return leg snapshot."

        if is_degraded:
            if not snapshot.get("degraded_notice_visible"):
                return False, "Degraded scenario did not render explicit partial-result notice."
            saw_degraded_contract = False
            for rec in request_records:
                done_json = rec.get("stream_done_json")
                response_json = rec.get("response_body_json")
                payload_obj = done_json if isinstance(done_json, dict) else response_json
                if isinstance(payload_obj, dict) and str(payload_obj.get("result_status") or "").lower() == "degraded":
                    saw_degraded_contract = True
                    break
            if not saw_degraded_contract:
                return False, "Degraded scenario did not carry result_status=degraded in response contract."

        if is_booking:
            if not snapshot.get("booking_link_visible"):
                return False, "Booking handoff scenario did not render booking link/button."
            href = str(snapshot.get("booking_link_href") or "").strip()
            if not href:
                return False, "Booking handoff link is missing href."
            if "/booking/handoff/post/" not in href and not href.startswith("http"):
                return False, "Booking handoff link target is not a valid handoff/provider URL."
            if str(snapshot.get("booking_link_target") or "").strip() != "_blank":
                return False, "Booking handoff link should open in a new tab (_blank)."
            rel = str(snapshot.get("booking_link_rel") or "").lower()
            if "noreferrer" not in rel:
                return False, "Booking handoff link should include rel=noreferrer."

        if validation_expectations:
            # Lightweight opt-in hook for future checks without changing interface shape.
            require_notice = str(validation_expectations.get("require_notice_contains") or "").strip().lower()
            if require_notice:
                notice_blob = " ".join(snapshot.get("notice_texts") or []).lower()
                haystack = " ".join(
                    [
                        notice_blob,
                        str(snapshot.get("error_text") or "").lower(),
                        str(snapshot.get("flight_text") or "").lower(),
                        str(snapshot.get("booking_panel_text") or "").lower(),
                        str(snapshot.get("tracking_panel_text") or "").lower(),
                    ]
                ).strip()
                if require_notice not in haystack:
                    return False, f"Expected notice text '{require_notice}' was not rendered."
            require_notice_any = validation_expectations.get("require_notice_contains_any")
            if isinstance(require_notice_any, list) and require_notice_any:
                haystack = " ".join(
                    [
                        " ".join(snapshot.get("notice_texts") or []).lower(),
                        str(snapshot.get("error_text") or "").lower(),
                        str(snapshot.get("flight_text") or "").lower(),
                        str(snapshot.get("booking_panel_text") or "").lower(),
                        str(snapshot.get("tracking_panel_text") or "").lower(),
                    ]
                ).strip()
                if not any(str(token).strip().lower() in haystack for token in require_notice_any if str(token).strip()):
                    return False, f"Expected one of notice markers {require_notice_any} was not rendered."

        return True, ""

    def _is_multi_leg_query(self, query: str) -> bool:
        normalized = query.lower()
        markers = (
            " via ",
            "connecting through",
            "stopover in",
            "with stop in",
            "stop in",
        )
        return any(marker in normalized for marker in markers)

    def _is_frontend_up(self) -> bool:
        try:
            r = requests.get(self.frontend_url, timeout=2)
            return 200 <= r.status_code < 500
        except Exception:
            return False

    def _start_frontend_server(self) -> None:
        if self._frontend_proc is not None:
            if self._frontend_proc.poll() is None:
                self._debug(
                    f"frontend_server_reused_internal=true mode={self.frontend_server_mode} pid={self._frontend_proc.pid}"
                )
                return
            self._debug("frontend_server_stale_process_reaped=true")
            self._frontend_proc = None
            self._frontend_started_by_validator = False

        if self._is_frontend_up():
            self._frontend_reused_external = True
            self._debug(f"frontend_server_reused_external=true mode={self.frontend_server_mode}")
            self.log(f"[frontend-validator] frontend_server_reused=true mode={self.frontend_server_mode}")
            return
        run_script = "dev"
        if self.frontend_server_mode == "preview":
            run_script = "preview"
            self._ensure_preview_build()

        self.log(f"Starting frontend {self.frontend_server_mode} server at {self.frontend_url}")
        
        # Inject validation auth token and other needed env vars for the frontend dev server
        frontend_env = os.environ.copy()
        # VITE_ prefixed vars are picked up by Vite/React
        # Use a consistent token for validation
        token = os.getenv("AUTH_TOKEN") or "validation-auth-token"
        frontend_env["VITE_AUTH_TOKEN"] = token
        
        # Determine backend URL for the frontend to hit
        # Default to localhost:8000 if not easily derived
        backend_url = os.getenv("VITE_API_BASE_URL", "http://127.0.0.1:8000")
        frontend_env["VITE_API_BASE_URL"] = backend_url
        
        cmd = ["npm", "run", run_script, "--", "--host", self.frontend_host, "--port", str(self.frontend_port)]
        self._frontend_proc = subprocess.Popen(
            cmd,
            cwd=str(self.frontend_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
            env=frontend_env,
        )
        self._frontend_started_by_validator = True
        self._frontend_reused_external = False
        self._frontend_pid = self._frontend_proc.pid
        self._debug(
            f"frontend_server_started=true mode={self.frontend_server_mode} pid={self._frontend_proc.pid}"
        )
        self.log(
            f"[frontend-validator] frontend_server_started=true mode={self.frontend_server_mode} pid={self._frontend_proc.pid}"
        )

    def _ensure_preview_build(self) -> None:
        cmd = ["npm", "run", "build"]
        self._debug("frontend_preview_build_start=true")
        result = subprocess.run(
            cmd,
            cwd=str(self.frontend_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise FrontendValidationError("Failed to build frontend for preview mode.")
        self._debug("frontend_preview_build_done=true")

    def _stop_frontend_server(self) -> bool:
        proc = self._frontend_proc
        if proc is None:
            return True

        pid = self._frontend_pid or proc.pid

        def _kill_group(sig: signal.Signals) -> bool:
            try:
                os.killpg(pid, sig)
                return True
            except Exception:
                return False

        if proc.poll() is None:
            if _kill_group(signal.SIGTERM):
                self._debug(f"frontend_server_sigterm_group pid={pid}")
            else:
                try:
                    proc.terminate()
                    self._debug(f"frontend_server_sigterm_proc pid={pid}")
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

        if proc.poll() is None:
            if _kill_group(signal.SIGKILL):
                self._debug(f"frontend_server_sigkill_group pid={pid}")
            else:
                try:
                    proc.kill()
                    self._debug(f"frontend_server_sigkill_proc pid={pid}")
                except Exception:
                    pass
            try:
                proc.wait(timeout=3)
            except Exception:
                pass

        if proc.poll() is not None and self._is_frontend_up():
            # Fallback pass for orphaned dev-server children that kept the port.
            _kill_group(signal.SIGTERM)
            time.sleep(0.4)
            if self._is_frontend_up():
                _kill_group(signal.SIGKILL)
                time.sleep(0.4)

        return proc.poll() is not None and not self._is_frontend_up()

    def _wait_for_frontend_ready(self) -> None:
        deadline = time.time() + self.startup_timeout_s
        self._debug(f"frontend_ready_wait_start timeout_s={self.startup_timeout_s}")
        while time.time() < deadline:
            if self._is_frontend_up():
                self._debug("frontend_ready=true")
                return
            time.sleep(0.5)
        self._debug("frontend_ready=false")
        raise FrontendValidationError(f"Frontend did not become ready at {self.frontend_url}")


def extract_payloads_from_curl_command(cmd: List[str]) -> List[Dict[str, object]]:
    """
    Extract JSON payloads from validation curl commands.
    Supports direct curl and simple `bash -c` multi-curl command strings.
    """
    payloads: List[Dict[str, object]] = []
    if not cmd:
        return payloads

    if cmd[0] == "curl":
        for idx, token in enumerate(cmd):
            if token in ("-d", "--data", "--data-raw", "--data-binary") and idx + 1 < len(cmd):
                try:
                    parsed = json.loads(cmd[idx + 1])
                    if isinstance(parsed, dict):
                        payloads.append(parsed)
                except Exception:
                    continue
        return payloads

    if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-c":
        script = cmd[2]
        for match in re_findall_json_payloads(script):
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    payloads.append(parsed)
            except Exception:
                continue
    return payloads


def re_findall_json_payloads(script: str) -> List[str]:
    import re

    results: List[str] = []
    patterns = [
        r"-d\s+'(\{.*?\})'",
        r'-d\s+"(\{.*?\})"',
    ]
    for pattern in patterns:
        results.extend(re.findall(pattern, script, flags=re.DOTALL))
    return results
