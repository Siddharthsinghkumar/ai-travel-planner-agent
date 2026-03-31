# LLM Travel Agent

Travel planning system with a FastAPI backend, React frontend, SSE streaming, and validation tooling that checks backend and real browser behavior.

This repository is currently an **advanced prototype**: strong for demos and portfolio usage, not yet production-ready for real user traffic without additional hardening.

## What This Project Does

- Accepts natural-language and/or structured trip inputs.
- Searches flights and weather, ranks options, and generates an LLM explanation.
- Supports:
  - Non-streaming responses (`POST /ask`)
  - Streaming SSE responses (`POST /ask?stream=true`)
  - Async job mode (`POST /ask?async_job=true` + `/jobs/...`)
- Includes backend unit/integration tests and frontend browser validation.

## Architecture (High Level)

1. Frontend submits query to backend `/ask`.
2. Backend planner resolves intent, calls tools:
   - Flight search (`tools/airline_api.py`)
   - Weather (`tools/weather_api.py`)
3. LLM routing (`agents/llm_router.py`) chooses cloud/local backend by mode.
4. For streaming mode, backend emits SSE frames and final `[DONE_JSON]{...}` payload.
5. Frontend parses SSE frames, displays progressive UI, and requires final parseable completion payload.

## Prerequisites

- Python 3.12
- Node.js 18+
- npm
- Optional for local model path: Ollama
- API keys in env files for live-provider runs

## Configuration Files

- Backend runtime (local): `.env`
- Frontend runtime: `frontend/.env`
- Docker variant used in this repo: `.env.laptopdocker`
- Validation-generated temp file: `.env.tmp` (auto-created by `full_validation.py`)

For full variable details and meanings, see [CONFIG.md](CONFIG.md).

## Local Run Flow

### 1) Backend

```bash
# from repo root
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### 2) Frontend

```bash
# new terminal
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

### 3) Quick sanity checks

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/llm/options
curl -sS http://127.0.0.1:8000/health/deep
curl -sS http://127.0.0.1:8000/health/keys
```

## Validation Flow

### Unit tests

```bash
venv/bin/pytest -q
```

### Full machine validation (backend + checks)

```bash
python full_validation.py --mode machine --r 0
```

### Full frontend validation (real UI path)

```bash
python full_validation.py --mode machine --frontend --r 0
```

### Live-provider run (real external calls)

```bash
python full_validation.py --mode machine --frontend --live --r 0
```

`--live` increases real dependency risk (quota, rate limit, provider outages).

## Health Endpoints: What They Actually Mean

- `GET /health/live`
  - Process liveness only.
- `GET /health/ready`
  - Startup/readiness gate; does not fully validate external dependencies.
- `GET /health`
  - Lightweight probe-safe status. Intentionally avoids deep external dependency checks.
- `GET /health/deep`
  - Deep dependency check (cloud/tool/backend integration signals).
- `GET /health/keys`
  - Key manager status metadata (no secret key values).

Operational rule: use `/health` for container probe stability, and `/health/deep` for dependency truth.

## Streaming & Routing Operational Truth

- `/ask?stream=true` uses SSE frames and a final `[DONE_JSON]` completion payload.
- `event: done` is terminal framing; completion correctness should be judged by parseable `[DONE_JSON]`.
- Routing modes: `ollama_only`, `cloud_only`, `cloud_first`, `ollama_first`.
- **Important limitation**: no true mid-stream provider failover after first token. If a provider fails after stream start, stream interruption is surfaced explicitly.

See ADRs:
- [docs/adr/0001-llm-routing.md](docs/adr/0001-llm-routing.md)
- [docs/adr/0002-streaming.md](docs/adr/0002-streaming.md)

## Common Failure Modes & Triage

### 1) Keys/quota/rate limit failures

Symptoms:
- provider unavailable
- 401/403/429 errors
- degraded deep health

Check:

```bash
curl -sS http://127.0.0.1:8000/health/keys
curl -sS http://127.0.0.1:8000/health/deep
```

### 2) Frontend stream completion failures

Symptoms:
- missing or unparseable completion payload
- frontend fallback triggered after stream activity

Check:

```bash
FRONTEND_VALIDATION_DEBUG=1 python full_validation.py --mode machine --frontend --r 0
```

### 3) Dev server port conflicts (Vite 5173)

Symptoms:
- validator cannot bind/start frontend

Check:

```bash
ss -ltnp | rg 5173
```

## Safe Demo Flow

1. Start backend and frontend locally.
2. Verify `/health` and `/llm/options`.
3. Run `python full_validation.py --mode machine --frontend --r 0`.
4. Demo queries from frontend UI with debug drawer off unless needed.

## Production Readiness (Current Reality)

Current state: **not production-ready**.

Reasons:
- External provider reliability/quota behavior still drives user-visible degradation.
- Mid-stream continuity is bounded (no true post-first-token failover).
- Operational maturity still requires stronger deployment/alert/SLO process.

## Additional Docs

- Demo quick sheet: [docs/demo-sheet.md](docs/demo-sheet.md)
- Full backend operator sheet: [docs/operator-sheet.md](docs/operator-sheet.md)
- Showcase curl command pack: [docs/showcase-commands.sh](docs/showcase-commands.sh)
- Configuration contract: [CONFIG.md](CONFIG.md)
- Minimal operations runbook: [docs/runbook.md](docs/runbook.md)
- Monitoring quickstart: [monitoring/README.md](monitoring/README.md)
