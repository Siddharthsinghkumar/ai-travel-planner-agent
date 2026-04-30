# Logging and Monitoring Defaults (Phase 7D.6)

Status: active  
Scope: practical default logging and monitoring contract for single-node production

## Log Streams: App vs Proxy

1. App runtime logs (FastAPI/Uvicorn)
- Source: app process stdout/stderr.
- Destination: `journald` via `systemd` service capture.
- Content: startup lifecycle, route/service logs, provider degradation, booking/tool diagnostics.

2. Proxy logs (Caddy)
- Access logs: request/response logs from Caddy (`deploy/Caddyfile.example` writes JSON access logs to file).
- Runtime logs: Caddy process logs are configured separately via Caddy global `log` option (not the same as access logs).

Operational split:
- Use Caddy access logs for ingress traffic truth (client, path, status, latency at edge).
- Use app logs for application behavior (planner, provider, booking/handoff logic, dependency degradation).

## Sensitive Data Redaction Baseline (S4)

The app logging pipeline applies a best-effort redaction filter (`core.logging_config.SensitiveDataRedactionFilter`) before console output.

Sensitive fields and values that must never appear in logs:
- API keys/secrets (`SERPAPI_KEY_*`, `OPENAI_KEY_*`, `GEMINI_KEY_*`, `WEATHER_KEY_*`, credentials in URLs).
- Tokens (`ADMIN_TOKEN`, bearer tokens, access/refresh/session tokens).
- Auth headers/cookies (`Authorization`, `X-Api-Key`, `X-Admin-Token`, `Cookie`, `Set-Cookie`).
- Password-like values in payloads or exception strings.
- Provider booking payload secrets (for example values inside `booking_request.post_data` when key names indicate secret/token/auth semantics).

Redaction behavior:
- Key/value strings like `api_key=...`, `token=...`, `authorization=...` are replaced with `***REDACTED***`.
- JSON-style secret fields are masked.
- Bearer token strings are masked.
- `logging` extra attributes with sensitive key names are masked.

Safe logging rule:
- Keep request IDs, route/method/status, duration, and error class metadata.
- Do not log raw request auth headers, cookies, API keys, or full secret-bearing payloads.

Safe example:
- Before: `Authorization=Bearer abc123 token=xyz api_key=live_foo`
- After: `Authorization=***REDACTED*** token=***REDACTED*** api_key=***REDACTED***`

## Default Log Levels by Environment

| Environment | App `LOG_LEVEL` default | Uvicorn access log default | Operator note |
|---|---|---|---|
| Development | `DEBUG` | Optional (`ENABLE_UVICORN_ACCESS_LOG=true` when needed) | Verbose local troubleshooting |
| Staging | `INFO` | `false` (prefer Caddy access logs) | Keep signal high, avoid duplicate access logs |
| Production | `INFO` | `false` (prefer Caddy access logs) | Promote to `WARNING` temporarily only for incident narrowing |

Current runtime behavior:
- `LOG_LEVEL` defaults to `INFO` when unset/invalid.
- `ENABLE_UVICORN_ACCESS_LOG` defaults to `false`.

## Canonical Field Checklist

Minimum fields operators should preserve/query:

1. Request identity
- `X-Request-ID` response header from API middleware.
- request completion log line includes `request_id`, method, route, status, duration.

2. Error classification
- Prefer structured/classified fields when present: `failure_reason`, `reason_class`, `exception_bucket`, `status`.
- Preserve HTTP status class from metrics/access logs.

3. Booking outcome
- Response fields for `/ask`/booking flows: `result_status`, `failure_reason`, booking handoff availability/quality.
- Metrics: `booking_handoff_consume_total{lookup_result,outcome}` for booking-handoff consumption outcomes.

## Noisy-Warning Policy

Not every warning-like signal should page operators.

Treat as informational/diagnostic unless accompanied by user-visible failures:
- `Post-success account check skipped (non-blocking)` (booking flow noise cleanup path).
- Isolated `/health/deep` provider degradation while `/health/live` and `/health/ready` remain healthy.

Treat as warning-worthy:
- Sustained `5xx` growth.
- Repeated retry budget exhaustion.
- Admission overload bursts.
- Booking handoff outcomes shifting persistently toward unresolved/unavailable states.

Polling guidance:
- `/health/live` and `/health/ready` can be frequent.
- `/health/deep` is diagnostic; do not poll aggressively.

## Minimum Monitoring Checklist

Use these as first-pass operational defaults:

1. Request rate
- `sum(rate(http_requests_total[5m]))`

2. Error rate (HTTP)
- `sum(rate(http_requests_total{status_class="5xx"}[5m])) / clamp_min(sum(rate(http_requests_total[5m])), 0.001)`

3. Booking-ready / booking-resolution trend
- `sum by (lookup_result, outcome) (increase(booking_handoff_consume_total[15m]))`
- Operator interpretation: watch ready/successful outcomes vs unresolved/failed outcomes over time.

4. Degraded-route trend
- Review `/ask` response fields (`result_status`, `failure_reason`) and correlate with `/health/deep` dependency degradation.
- Current repo does not provide a single dedicated degraded-route Prometheus metric; use response/log sampling plus existing counters.

## Starter Alert Thresholds (Initial Defaults)

These are intentionally conservative first-pass defaults:

- HTTP 5xx ratio > 5% for 10m when request volume is non-trivial.
- `ask_admission_total{outcome="rejected_overload"}` increase >= 5 over 10m.
- `retry_budget_exhausted_total` increase >= 3 over 10m.
- Repeated stream/router fallback/failure counters rising in short windows.

Reference alert rules:
- `monitoring/alerts.yml`

## Reference Basis

- FastAPI deployment concepts (startup/restart and process-management context): https://fastapi.tiangolo.com/deployment/concepts/
- Caddy request logging (`log` directive) and runtime logging (global `log` option):
  - https://caddyserver.com/docs/caddyfile/directives/log
  - https://caddyserver.com/docs/caddyfile/options
- OWASP Logging Cheat Sheet:
  - https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
