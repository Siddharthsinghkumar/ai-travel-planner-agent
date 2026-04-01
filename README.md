# LLM Travel Agent

FastAPI + React travel planning app with:
- non-stream `/ask`
- streaming `/ask?stream=true` (SSE + final `[DONE_JSON]`)
- optional async jobs (`/ask?async_job=true`, `/jobs/...`)
- booking handoff bridge (`/booking/handoff/post/{artifact_id}`)
- local validation harness (`full_validation.py`)

Current maturity: advanced local/demo prototype, not production-hardened.

## What Exists Today

- Backend API: [api/app.py](api/app.py)
- Planner and routing: `agents/planner_agent.py`, `agents/llm_router.py`
- Tools: `tools/airline_api.py`, `tools/weather_api.py`, `tools/booking_handoff.py`
- Frontend app: `frontend/src/App.tsx`
- Frontend stream/fallback logic: `frontend/src/hooks/useStreamingPlan.tsx`
- Validation entrypoint: [full_validation.py](full_validation.py)
- Browser runtime validator: `validation/frontend_validator.py`
- Tests: `tests/` (`pytest -q`)

## Quick Start (Local)

### 1. Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev -- --host 127.0.0.1 --port 5173
```

### 3. Quick Checks

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/llm/options
curl -sS http://127.0.0.1:8000/health/deep
curl -sS http://127.0.0.1:8000/health/keys
```

## Validation Workflow

### Unit/integration test suite

```bash
venv/bin/pytest -q
```

### Full local harness (default pre-push path)

```bash
venv/bin/python full_validation.py --mode machine --r 0
```

### Frontend runtime validation (browser-driven)

```bash
venv/bin/python full_validation.py --mode machine --profile full --frontend --r 0
```

Notes:
- `--frontend` routes selected validations through browser automation and frontend runtime checks.
- Frontend runtime matrix runs in `--profile full` and `--profile frontend-heavy`.
- Default `full_validation.py` mode is `--mode both` (machine + docker).
- Logs are written to `validation_logs/`.

### Live provider mode (non-deterministic, slower)

```bash
venv/bin/python full_validation.py --mode machine --live --r 0
```

Use `--live` only when you intentionally want real external provider behavior (keys, quota, rate limits, network effects).

## Runtime and Health Endpoints

- `GET /health/live`: liveness only
- `GET /health/ready`: readiness/startup status
- `GET /health`: lightweight runtime health (stable probe surface)
- `GET /health/deep`: deeper dependency health
- `GET /health/keys`: key-manager status metadata
- `GET /llm/options`: routing/provider/options visibility
- `GET /version`: commit + file mtime metadata

## Frontend and Review Artifacts

In `frontend/`:
- `npm run dev`: Vite dev server
- `npm run build`: production bundle
- `npm run build:review-demo`: outputs backend-free review artifact at repo root `review-demo.html`
- `npm run export:frozen-demo`: outputs single-file static artifact at `frontend/dist/frozen-demo.html`

These demo/review artifacts are for sharing UI/product review states and are not substitutes for live backend validation.

## Docker Path

Local make targets:

```bash
make build
make run
```

- Makefile image: `sidd/llm-travel-agent:latest`
- Env file for make run: `.env.laptopdocker`

`full_validation.py` uses its own docker test image/container flow for validator runs.

## Production-Like vs Local/Demo

- Production-like surfaces:
  - FastAPI contracts and health/runtime endpoints
  - planner + tool orchestration
  - SSE completion contract (`[DONE_JSON]`)
  - booking bridge one-time artifact flow
- Local/demo-only conveniences:
  - TESTING-mode deterministic behavior
  - standalone review HTML artifacts
  - validation harness scenario mocks for frontend runtime checks

## More Docs

- Config reference: [CONFIG.md](CONFIG.md)
- Frontend usage details: [frontend/README.md](frontend/README.md)
- Operator sheet: [docs/operator-sheet.md](docs/operator-sheet.md)
- Runbook: [docs/runbook.md](docs/runbook.md)
- Demo sheet: [docs/demo-sheet.md](docs/demo-sheet.md)
- Monitoring quickstart: [monitoring/README.md](monitoring/README.md)
