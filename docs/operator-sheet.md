# Operator Sheet (Backend)

Source basis: `api/app.py`, `agents/planner_agent.py`, `core/job_queue.py`, `tools/booking_handoff.py`, `README.md`, `CONFIG.md`, `docs/runbook.md`, `monitoring/README.md`.

Canonical production contracts:
- Topology: `docs/deployment-topology.md`
- Environment/secrets: `docs/environment-secrets-contract.md`

## 1) Endpoint Inventory

| Method | Path | Purpose | Showcase-safe | Notes |
|---|---|---|---|---|
| POST | `/ask` | Main planning endpoint (JSON or SSE via query params) | Yes | Query params: `stream`, `async_job` |
| GET | `/booking/handoff/post/{artifact_id}` | One-time POST booking bridge | Demo only when artifact exists | Single-consume artifact semantics |
| GET | `/jobs/{job_id}` | Async-job status/result polling | Diagnostic | Useful only when async-job accepted |
| GET | `/jobs/{job_id}/events` | Async-job SSE events | Diagnostic | Structured event stream (queued/running/progress/done/error/cancelled/closed) |
| GET | `/health/live` | Liveness probe | Yes | Process is alive |
| GET | `/health/ready` | Readiness state | Yes | Returns 503 only while startup is incomplete |
| GET | `/health` | Lightweight runtime health | Yes | Stable, avoids deep external checks |
| GET | `/health/deep` | External/provider deep health | Diagnostic | Can degrade due to provider/key noise |
| GET | `/health/keys` | Key-manager metadata status | Diagnostic | No secret values returned; includes SerpAPI reconcile metadata |
| GET | `/llm/options` | Routing/provider/options snapshot | Yes | Includes config authority + effective mode info |
| GET | `/metrics` | Prometheus metrics | Diagnostic | Text format, no JSON |
| GET | `/version` | Build/version metadata | Diagnostic | `git_commit` + file mtime |
| GET | `/debug/keys` | Masked key debug status | No (admin) | Requires `X-Admin-Token` |
| POST | `/debug/keys/reload` | Force key reload from env | No (admin) | Requires `X-Admin-Token` |
| GET | `/openapi.json` | OpenAPI schema | Diagnostic | FastAPI default |
| GET | `/docs` | Swagger UI | Diagnostic | FastAPI default |
| GET | `/redoc` | ReDoc UI | Diagnostic | FastAPI default |

## 2) Main Ask/Planner Request Field Table

`POST /ask` JSON body (`AskRequest`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `origin` | string | Optional | If provided, `destination` must also be provided |
| `destination` | string | Optional | If provided, `origin` must also be provided |
| `date` | string (`YYYY-MM-DD`) | Optional | Validated format; invalid format => 422 |
| `user_query` | string | Optional | Natural-language intent; at least `user_query` or structured route pair required |
| `trip_type` | string | Optional | Normalized to supported semantic/route labels |
| `llm_mode` | string | Optional | Normalized mode; valid: `ollama_only`, `cloud_only`, `cloud_first`, `ollama_first` |
| `cloud_provider` | string | Optional | Normalized cloud provider override |

Additional query params on `/ask`:

| Query param | Type | Default | Notes |
|---|---|---|---|
| `stream` | bool | `false` | `true` returns SSE stream |
| `async_job` | bool | `false` | `true` enqueues background job (topology-guarded) |

`trip_type` accepts normalized forms including: `Business`, `Holiday`, `Flexible`, `Urgent`, `one-way`, `round-trip`, `via-stopover` plus case/spacing variants mapped by server.

## 3) Curl Examples By Endpoint

Assume:
```bash
BASE="http://127.0.0.1:8000"
```

### `POST /ask` (non-stream)
```bash
curl -sS -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"origin":"DEL","destination":"BOM","date":"2030-01-15","trip_type":"one-way","user_query":"Cheapest acceptable option with short explanation."}' | jq
```

Sync `/ask` success contract includes:
- `best_flight`
- `top_flights`
- `all_flights` (ranked list, non-null)
- `constraint_outcomes` when explicit constraints (for example cabin availability truth) need structured disclosure

### `POST /ask?stream=true` (SSE)
```bash
curl -N -sS -X POST "$BASE/ask?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"origin":"DEL","destination":"BLR","date":"2030-01-16","user_query":"Recommend best overall option and explain tradeoffs."}'
```

### `POST /ask?async_job=true` (when topology supports)
```bash
curl -sS -X POST "$BASE/ask?async_job=true" \
  -H "Content-Type: application/json" \
  -d '{"origin":"DEL","destination":"BOM","date":"2030-01-15","user_query":"Find best value option."}' | jq
```

### `GET /jobs/{job_id}`
```bash
curl -sS "$BASE/jobs/<job_id>" | jq
```

### `GET /jobs/{job_id}/events`
```bash
curl -N -sS "$BASE/jobs/<job_id>/events"
```

### `GET /booking/handoff/post/{artifact_id}`
```bash
curl -i -sS "$BASE/booking/handoff/post/<artifact_id>"
```

### Health/diagnostic endpoints
```bash
curl -sS "$BASE/health/live" | jq
curl -sS "$BASE/health/ready" | jq
curl -sS "$BASE/health" | jq
curl -sS "$BASE/health/deep" | jq
curl -sS "$BASE/health/keys" | jq
curl -sS "$BASE/llm/options" | jq
curl -sS "$BASE/version" | jq
curl -sS "$BASE/metrics" | head -n 40
```

### Admin/debug endpoints
```bash
ADMIN_TOKEN_VALUE="<your_admin_token>"
curl -sS -H "X-Admin-Token: $ADMIN_TOKEN_VALUE" "$BASE/debug/keys" | jq
curl -sS -X POST -H "X-Admin-Token: $ADMIN_TOKEN_VALUE" "$BASE/debug/keys/reload" | jq
```

## 4) Streaming Guide

- Call pattern: `POST /ask?stream=true` with JSON body and `Content-Type: application/json`.
- Response type: `text/event-stream`.
- Stream may include typed SSE events from planner:
  - `event: reasoning_step`
  - `event: flights`
  - `event: weather`
- Stream includes token text frames and must finish with `[DONE_JSON]{...}` payload.
- API wrapper emits terminal SSE frame: `event: done`.

Save and inspect stream:
```bash
curl -N -sS -X POST "$BASE/ask?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"user_query":"Delhi to Mumbai on 2030-01-18, explain reasoning."}' | tee /tmp/ask_stream.log

rg -n "event:|\\[DONE_JSON\\]|\\[ERROR\\]" /tmp/ask_stream.log
```

Success/degraded hints:
- Success: parseable `[DONE_JSON]` with structured fields and no terminal stream contract error.
- Degraded but truthful: `[DONE_JSON]` with `result_status: "degraded"` and degradation metadata.

## 5) Health/Diagnostics Guide

- `/health`: stable runtime status for app-level checks; includes runtime topology and async-job support state.
- `/health/deep`: external dependency truth (cloud/tool/provider), may show degradation caused by API/key/provider conditions.
- `/health/keys`: key metadata and activity/exhaustion state (masked/no raw secrets).
- `/llm/options`: effective mode/provider usability and config authority snapshot.
- `/metrics`: Prometheus counters/histograms for HTTP, streaming, routing, tools.

Quick triage sequence:
1. `/health`
2. `/llm/options`
3. `/health/deep`
4. `/health/keys`
5. `/metrics` (if persistent errors/latency)

## 6) Async-Job / Jobs / Topology

Current contract:
- `contract = "single_worker_required_process_local_queue"`
- Async jobs can be disabled when declared workers > 1 and single-worker guard is active.
- Override exists via `ALLOW_UNSAFE_ASYNC_JOBS=1` (not recommended for normal operation).

Guard behavior:
- Unsupported topology request to `/ask?async_job=true` returns `503` with detail:
  - `error: async_job_topology_unsupported`
  - `reason`
  - `declared_workers`
  - hint text

When supported:
1. `POST /ask?async_job=true` returns `202` + `job_id`
2. Poll `GET /jobs/{job_id}`
3. Optional SSE on `GET /jobs/{job_id}/events` with structured JSON `data` payloads per event

## 7) Booking/Handoff

- Booking bridge endpoint: `GET /booking/handoff/post/{artifact_id}`.
- Booking lifecycle is local-follow-up only: `HELD` → `CANCELLED`/`EXPIRED`.
- Real booking confirmation/payment is external-only (airline/OTA/provider checkout), not an in-app state transition.
- Hold endpoint semantics are explicit:
  - `hold_created=true` means local hold row was created.
  - `checkout_ready=true` only when a provider checkout URL is available.
  - `checkout_status` is `booking_ready` or `provider_handoff_unavailable`.
  - `hold_outcome` is `held_with_checkout` or `held_local_only`.
- Track endpoint semantics are explicit:
  - `/booking/track-price` success requires valid persisted HELD route/date/fare prerequisites.
  - Success payload includes `tracking_state` with `route_tracking_ready=true`.
  - Tracking is route/selection-based and does not require checkout-ready handoff URL.
- Artifact is short-lived and one-time consume.
- First consume: returns auto-submitting HTML form for provider POST flow.
- Re-consume/expired/not found: `404` with structured detail including `lookup_result`.
- `/ask` is search-only by default: do not expect eager booking-token/booking-options resolution during plain search.
- Booking resolution is lazy and selection-gated: resolve only after explicit `buy`/`hold`/`track` intent for the chosen itinerary.
- Round-trip `/ask` is bounded to outbound search plus at most one additional return search.
- Primary handoff assignment is booking-ready-only:
  - `best_flight.handoff_url` is set only when `booking_handoff.status == booking_ready`.
  - No Google Flights search-assist fallback URL is emitted in booking flow responses.
  - This is a strict product boundary: booking flows must stay booking-token/provider-resolution only.
  - If provider resolution fails, booking handoff stays explicit and unavailable (`status=unavailable`, `url=null`).
- SerpApi-first resolver path (book button backend contract):
  - use itinerary-level `booking_token` from selected flight (`best_flights[]` / `other_flights[]`), not root fallback.
  - booking-options request shape includes route context (`departure_id`, `arrival_id`, `outbound_date`, `type`) plus locale/currency (`hl`, `currency`) and optional hints (`include_airlines`, `deep_search`, `adults`, `travel_class`).
  - for `booking_request.post_data`, backend POST-resolves server-side and extracts provider URL from redirect/meta-refresh response.
  - only resolved non-Google provider URL may populate `handoff_url`; raw `google.com/travel/clk/f` is never emitted as primary booking action.
- Validation truth rules (operator runbooks):
  - Validate booking on the resolved non-Google provider URL in a real browser.
  - Browser landing is authoritative for pass/fail (usable airline/OTA booking page vs error/generic Google page).
  - `curl -L` is diagnostic support only and is not sufficient booking proof.
  - Raw Google click artifacts (`google.com/travel/clk/f` or `/booking/handoff/post/...`) are never accepted as booking-proof targets.
  - Pre-check for live validation: at least one SerpAPI key must have `plan_searches_left > 0`; if all keys are exhausted, new resolver/browser proof runs are externally blocked until quota resets or keys are replaced.
- Booking contract in client responses is intentionally lean:
  - `booking_handoff` contains only availability-critical fields (`status`, `reason`, `source`, `url`, `booking_exit_quality`, optional `provider/cache_hit`).
- Round-trip responses expose compact handoff metadata under:
  - `booking_handoff.round_trip.return_search_outcome`
  - `booking_handoff.round_trip.return_search_reason`
  - `booking_handoff.round_trip.return_handoff_status`
  - `booking_handoff.round_trip.is_outbound_only_handoff`
- Return-leg payloads now include resolved handoff metadata on `return_trip.best_flight.booking_handoff` when available.

Reusable manual parity command:
```bash
set -a; source .env; set +a
venv/bin/python tools/serpapi_manual_resolver.py \
  --origin DEL --destination BOM --date 2026-04-26 \
  --trip-type one-way --itinerary-source best_flights --itinerary-index 0
```
- `tools/serpapi_manual_resolver.py` resolves SerpAPI keys through key-manager pools.
- `--api-key` override is intentionally ignored to keep key access centralized.
- Paid provider APIs (Duffel/Amadeus) are not part of the supported runtime path.
- Operator booking flow is SerpApi-first only: itinerary token -> booking options -> server-side POST resolve -> non-Google provider URL when available.
- Hot-path post-success SerpApi `account.json` checks are disabled by default; rely on bounded background reconciliation for quota-state refresh.

Common `lookup_result` values include:
- `already_consumed`
- `expired`
- `not_found`
- `consume_race_lost`
- `invalid_artifact_id`
- `lookup_failed`

## 8) Admin/Debug

- Admin endpoints require `X-Admin-Token` and are for operations, not showcase.
- Production contract: do not expose admin/debug endpoints on the public internet. Keep them private behind reverse-proxy network controls plus `X-Admin-Token`.
- Use cases:
  - inspect key-manager state (`/debug/keys`)
  - force key reload (`/debug/keys/reload`)
  - inspect/set/disable provider-state overrides (`/debug/provider-state/overrides*`)
  - operator-forced SerpAPI reconcile (`/debug/provider-state/reconcile/serpapi`)
- `/debug/keys` is sanitized to avoid exposing key names, key fingerprints, or raw provider exception text.
- Avoid exposing admin tokens in shared logs/screens.

## 9) Responsibility Map

### Backend / FastAPI
- Request validation (`AskRequest` schema + normalization rules)
- Endpoint routing and query-param mode selection (`stream`, `async_job`)
- Topology guard enforcement for async jobs
- SSE wrapper enforcement (`[DONE_JSON]` completion contract + terminal `event: done`)
- Runtime topology/health payloads
- Admin endpoint auth gate via `X-Admin-Token`

### Planner / Backend Business Logic
- Intent extraction/normalization and orchestration
- Flight + weather data gathering and ranking
- Explanation generation (LLM-backed) and degraded fallback behavior
- Streaming typed events (`reasoning_step`, `flights`, `weather`)
- Final structured response composition (`PlanResult`/`MultiCityResult`)

### Frontend
- UI rendering, stream parsing, and progressive UX
- Fallback UX decisions when stream contract/activity fails
- Presentation of cards/labels/reasoning
- User input collection and request shaping

### External APIs / Providers
- Flight result availability/quality and booking artifacts
- Weather data truth and availability
- Cloud LLM provider uptime/rate-limits/quota behavior

### Local Model / Runtime
- Ollama availability/performance
- Local explanation generation behavior and latency

### Intentionally Unsupported / Deferred
- Distributed async-job semantics for true multi-worker/shared-state topology (explicitly deferred)
- Guaranteed booking-ready artifacts for every provider/route case
- Zero-noise deep health in presence of unstable external providers

## 10) Known Limitations

- Async jobs in multi-worker topology are intentionally guarded unless unsafe override is enabled.
- Job queue/state is process-local in current architecture.
- Booking handoff bridge artifacts are single-consume and can expire.
- `/health/deep` can degrade due to external/provider/key issues even when app process is healthy.
- Stream responses may degrade truthfully when LLM backends/timeouts fail; structured fallback is preserved.
- SerpAPI Account API does not document a guaranteed exact reset timestamp; provider reset timing is tracked via reconcile evidence plus explicit inferred reset basis metadata.
- `/health/keys` is intentionally high-level and does not expose detailed quota/reconcile internals.

## 14) Unified Provider Key-State Persistence

- Key state for all managed providers (`serpapi`, `weather`, `openai`, `gemini`, `anthropic`) is durably persisted in SQL storage (`provider_key_states`) using:
  - `provider`
  - `key_name_fingerprint`
  - `key_value_fingerprint`
  - `is_exhausted`
  - `exhausted_until`
  - `retry_after`
  - `searches_left`
  - `last_checked_at`
  - `last_used_at`
  - `expected_reset_basis`
  - `expected_reset_at`
  - `last_error`
  - `last_reason`
  - `failure_classification`
  - `state_meta`
- Canonical single-node deployment baseline uses SQLite (`DATABASE_URL=sqlite:////var/lib/llm-travel-agent/local.db`).
- PostgreSQL is optional/non-canonical and must be treated as an explicit alternative deployment choice.
- Startup (refresh-owner worker):
  1. key load from env
  2. DB hydration for provider key state (bounded startup wait; overflow continues in background)
  3. background reconcile/refresh loops (startup not blocked on provider account checks)
- Runtime:
  - reservation and exhaustion transitions are written to durable provider state
  - rotated key values clear stale exhaustion state by slot fingerprint
  - SerpAPI exhausted keys are rechecked only after retry window opens (not on every startup)
  - SerpAPI unknown-reset quota keys are deferred for a weekly retry window until recovered
- Manual overrides are durably persisted in `provider_state_overrides` with scope-aware semantics:
  - `scope_type`: `key` | `provider_account` | `project`
  - `override_type`: `force_exhausted_until` | `clear_exhaustion` | `force_active_until` | `skip_reconcile_until`
  - key overrides can be targeted by stored key fingerprint (`scope_identifier`) or admin request `key_index` resolved server-side
  - SerpAPI key-scope overrides bind to both `key_name_fingerprint` and `key_value_fingerprint`; if slot-name fingerprint or key-value fingerprint changes, the override is automatically non-applicable
  - SerpAPI key-scope `force_exhausted_until` also applies a durable known-reset horizon for that key slot, so operator-specified reset datetimes actually release/recheck as expected.
  - override datetimes are normalized to UTC in storage/output (offset-aware input is converted, not discarded); `override_until` is the preferred operator-facing field and `active_until` remains as compatibility alias.

## 15) Provider Policy Semantics

- SerpAPI:
  - automatic reconcile via Account API; monthly-cycle exhaustion can persist across restarts.
- Gemini:
  - runtime quota/rate handling defaults to project/provider-account scoped overrides.
  - key-specific behavior is secondary and should be used only when project/account mapping evidence exists.
- OpenWeather:
  - limits are product-plan specific; default fallback holds should stay short/manual-policy-first unless upstream provides explicit reset windows.
- OpenAI:
  - treat billing/credit exhaustion (`insufficient_quota`/billing domain) separately from ordinary transient rate-limit cooldowns.

## 11) Safe Local Startup Commands

Stable showcase/local command:
```bash
LLM_MODE=ollama_only USE_CLOUD_LLM=0 venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Typical mixed mode local command:
```bash
LLM_MODE=ollama_first USE_CLOUD_LLM=1 venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

## 12) Unsupported Topology Diagnostics (Lab Only)

Multi-worker async-job diagnostics are intentionally out of scope for this repository's supported runtime contract.
Use the canonical single-process startup commands above for all operator workflows.

## 13) Accepted Design vs Bug vs External Noise

- Accepted-by-design:
  - Async-job rejection in unsupported multi-worker topology.
  - One-time booking handoff consume behavior.
  - `/health` lightweight semantics separate from `/health/deep`.

- Bug indicators:
  - Missing terminal stream completion contract (`[DONE_JSON]` + `event: done`) for successful stream transport.
  - Inconsistent `/ask` request validation outcomes for valid schema combinations.
  - `/jobs/{id}` state transitions that never reach terminal state despite worker activity.

- External/provider noise:
  - Deep health degradations due to quota/rate-limit/provider outages.
  - Cloud/provider initialization usability shifts based on key availability.
  - Flight/weather upstream jitter affecting output quality without local app crash.
