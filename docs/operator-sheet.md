# Operator Sheet (Backend)

Source basis: `api/app.py`, `agents/planner_agent.py`, `core/job_queue.py`, `tools/booking_handoff.py`, `README.md`, `CONFIG.md`, `docs/runbook.md`, `monitoring/README.md`.

## 1) Endpoint Inventory

| Method | Path | Purpose | Showcase-safe | Notes |
|---|---|---|---|---|
| POST | `/ask` | Main planning endpoint (JSON or SSE via query params) | Yes | Query params: `stream`, `async_job` |
| GET | `/booking/handoff/post/{artifact_id}` | One-time POST booking bridge | Demo only when artifact exists | Single-consume artifact semantics |
| GET | `/jobs/{job_id}` | Async-job status/result polling | Diagnostic | Useful only when async-job accepted |
| GET | `/jobs/{job_id}/events` | Async-job SSE events | Diagnostic | Event stream for queued/running/token/done/error/closed |
| GET | `/health/live` | Liveness probe | Yes | Process is alive |
| GET | `/health/ready` | Readiness/warming state | Yes | May return 503 during startup/prewarm |
| GET | `/health` | Lightweight runtime health | Yes | Stable, avoids deep external checks |
| GET | `/health/deep` | External/provider deep health | Diagnostic | Can degrade due to provider/key noise |
| GET | `/health/keys` | Key-manager metadata status | Diagnostic | No secret values returned |
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
3. Optional SSE on `GET /jobs/{job_id}/events`

## 7) Booking/Handoff

- Booking bridge endpoint: `GET /booking/handoff/post/{artifact_id}`.
- Artifact is short-lived and one-time consume.
- First consume: returns auto-submitting HTML form for provider POST flow.
- Re-consume/expired/not found: `404` with structured detail including `lookup_result`.

Common `lookup_result` values include:
- `already_consumed`
- `expired`
- `not_found`
- `consume_race_lost`
- `invalid_artifact_id`
- `lookup_failed`

## 8) Admin/Debug

- Admin endpoints require `X-Admin-Token` and are for operations, not showcase.
- Use cases:
  - inspect key-manager state (`/debug/keys`)
  - force key reload (`/debug/keys/reload`)
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
- Correct distributed async-job semantics in true multi-worker topology with process-local queues/state
- Guaranteed booking-ready artifacts for every provider/route case
- Zero-noise deep health in presence of unstable external providers

## 10) Known Limitations

- Async jobs in multi-worker topology are intentionally guarded unless unsafe override is enabled.
- Job queue/state is process-local in current architecture.
- Booking handoff bridge artifacts are single-consume and can expire.
- `/health/deep` can degrade due to external/provider/key issues even when app process is healthy.
- Stream responses may degrade truthfully when LLM backends/timeouts fail; structured fallback is preserved.

## 11) Safe Local Startup Commands

Stable showcase/local command:
```bash
LLM_MODE=ollama_only USE_CLOUD_LLM=0 venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Typical mixed mode local command:
```bash
LLM_MODE=ollama_first USE_CLOUD_LLM=1 venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

## 12) Unsafe / Diagnostic Startup Commands

Multi-worker with guarded async jobs (safe but async jobs intentionally disabled):
```bash
UVICORN_WORKERS=2 ASYNC_JOB_REQUIRE_SINGLE_WORKER=1 venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 2
```

Unsafe async override (diagnostic only):
```bash
UVICORN_WORKERS=2 ASYNC_JOB_REQUIRE_SINGLE_WORKER=1 ALLOW_UNSAFE_ASYNC_JOBS=1 venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 2
```

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
