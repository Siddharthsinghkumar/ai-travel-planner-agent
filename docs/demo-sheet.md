# Demo Sheet (Showcase-Safe)

## Safest Local Startup
```bash
LLM_MODE=ollama_only USE_CLOUD_LLM=0 venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

## 5 Must-Have Demo Commands
```bash
# 1) Lightweight health (stable probe signal)
curl -sS http://127.0.0.1:8000/health | jq

# 2) Runtime options/mode visibility
curl -sS http://127.0.0.1:8000/llm/options | jq

# 3) Non-stream ask (structured route)
curl -sS -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"origin":"DEL","destination":"BOM","date":"2030-01-15","trip_type":"one-way","user_query":"Find the best value option and explain why."}' | jq

# 4) Streaming ask (SSE + DONE_JSON)
curl -N -sS -X POST "http://127.0.0.1:8000/ask?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"origin":"DEL","destination":"BLR","date":"2030-01-16","trip_type":"one-way","user_query":"Give me a concise recommendation with tradeoffs."}'

# 5) Round-trip ask
curl -sS -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"user_query":"Round-trip Delhi to Mumbai, leave 2030-01-20 and return 2030-01-24, prioritize cheapest acceptable option."}' | jq
```

## 5 Fallback Diagnostic Commands
```bash
# 1) Deep health (external dependency truth)
curl -sS http://127.0.0.1:8000/health/deep | jq

# 2) Key manager status (no secret values)
curl -sS http://127.0.0.1:8000/health/keys | jq

# 3) Readiness gate state
curl -sS http://127.0.0.1:8000/health/ready | jq

# 4) Version/build visibility
curl -sS http://127.0.0.1:8000/version | jq

# 5) Prometheus metrics quick check
curl -sS http://127.0.0.1:8000/metrics | head -n 40
```

## 3 Commands/Modes To Avoid During Showcase
```bash
# 1) Avoid async-job in multi-worker topology (intentionally guarded)
curl -sS -X POST "http://127.0.0.1:8000/ask?async_job=true" -H "Content-Type: application/json" -d '{"user_query":"..."}'

# 2) Avoid admin debug endpoints unless intentionally demoing operator internals
curl -sS -H "X-Admin-Token: <token>" http://127.0.0.1:8000/debug/keys

# 3) Avoid unsafe async override startup in normal demo runs
ASYNC_JOB_REQUIRE_SINGLE_WORKER=1 ALLOW_UNSAFE_ASYNC_JOBS=1 UVICORN_WORKERS=2 venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

## Notes
- Async-job topology: async jobs are intentionally disabled when declared workers > 1 unless unsafe override is enabled.
- `/health` vs `/health/deep`: use `/health` for stable app/runtime checks; use `/health/deep` for cloud/airline/weather dependency truth.
- Booking handoff bridge: `GET /booking/handoff/post/{artifact_id}` is one-time consume; repeated opens can return 404 (`already_consumed`/`expired`/`not_found`).
- Good-enough showcase output: for `/ask`, a structured result with `best_flight`, `weather`, and meaningful `llm_response`; for stream mode, visible SSE progress plus final `[DONE_JSON]{...}` and terminal `event: done`.
