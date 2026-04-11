# Operational Runbook (Minimal)

This runbook is for quick triage of the `llm-travel-agent` API.
For broader endpoint usage and copy-paste command sets, also see:
- [operator-sheet.md](./operator-sheet.md)
- [runtime-script-catalog.md](./runtime-script-catalog.md)

## 1) API unhealthy

1. Check liveness and readiness:
   - `GET /health/live`
   - `GET /health/ready`
   - `GET /health`
2. Treat `GET /health` as a lightweight probe signal only (no external provider checks).
   - It is useful for process/startup reachability and local dependency hints.
   - It is not authoritative for cloud/airline/weather dependency truth.
3. For dependency truth, use:
   - `GET /health/deep` for external providers + key-gate context
   - `GET /health/keys` for key-state diagnostics
4. Check request-level errors:
   - `http_requests_total{status_class="5xx"}`
   - `http_request_duration_seconds`
5. Validate metrics endpoint is reachable:
   - `GET /metrics`

## 2) Streaming failures increasing

1. Inspect fallback and router failure counters:
   - `stream_fallback_total`
   - `llm_router_stream_failures_total`
   - `stream_requests_total{status="error"}`
2. Look at likely causes by labels:
   - `reason` on `stream_fallback_total`
   - `cause` on `llm_router_stream_failures_total`
3. Check if failures cluster on one backend (`provider` label).
4. If stream init timeouts spike, verify `PLANNER_STREAM_INIT_TIMEOUT` and backend health.

## 3) First-token latency high

1. Check p95:
   - `histogram_quantile(0.95, sum by (le, provider) (rate(llm_first_token_latency_seconds_bucket[10m])))`
2. Compare with full latency:
   - `llm_full_response_latency_seconds`
3. Correlate with routing/fallback:
   - `llm_route_usage_total`
   - `stream_fallback_total`
4. If mostly cloud-related, verify provider/key health and rate limits.

## 4) Cloud/local LLM unavailable

1. Check health endpoints and routing options:
   - `GET /health`
   - `GET /health/deep`
   - `GET /llm/options`
2. Confirm active mode/provider defaults in environment (`CONFIG.md`).
3. Check router/stream error tags in metrics:
   - `llm_router_stream_failures_total`
   - `llm_route_usage_total`
   - `provider_health_failures_total`
   - `provider_health_cooldown_skips_total`
4. Validate local Ollama reachability and cloud key availability.

## 5) Tool failures (flight/weather)

1. Check dependency status in `GET /health/deep` and `GET /health`.
2. Check tool metrics and retries:
   - `tool_requests_total{tool="airline"| "weather"}`
   - `tool_request_latency_seconds`
   - `airline_retries_total`, `weather_retries_total`
3. If weather or airline failures increase, inspect external API key status:
   - `GET /health/keys`
4. If needed, run in fallback mode (`ollama_only`/`cloud_only`) to reduce routing noise during triage.

## 6) Local confidence checks before push

1. Run unit/integration suite:
   - `venv/bin/pytest -q`
2. Run machine validation harness:
   - `venv/bin/python full_validation.py --mode machine --r 0`
3. If UI/runtime behavior matters for your change, also run:
   - `venv/bin/python full_validation.py --mode machine --profile full --frontend --r 0`

Notes:
- Harness logs are written to `validation_logs/`.
- `--live` mode calls real external providers and is intentionally less deterministic.
