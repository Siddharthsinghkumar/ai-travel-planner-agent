# LLM Travel Agent

FastAPI + React travel planning app with:
- `POST /ask` (non-stream)
- `POST /ask?stream=true` (SSE + final `[DONE_JSON]`)
- optional async jobs (`/ask?async_job=true`, `/jobs/...`)
- booking handoff bridge (`/booking/handoff/post/{artifact_id}`)
- local validation harness (`full_validation.py`)

Current maturity: advanced local/demo prototype.

## Canonical Runtime Contract

Supported production model:
- single-node only
- Caddy reverse proxy in front of FastAPI
- one `uvicorn` app process (`--workers 1`) on loopback
- no public debug/admin endpoints
- no multi-worker shared-state or distributed async topology

Authoritative deployment/security docs:
- [docs/deployment-topology.md](docs/deployment-topology.md)
- [docs/environment-secrets-contract.md](docs/environment-secrets-contract.md)
- [docs/startup-readiness-liveness.md](docs/startup-readiness-liveness.md)
- [docs/reverse-proxy-caddy.md](docs/reverse-proxy-caddy.md)
- [docs/persistence-backups.md](docs/persistence-backups.md)
- [docs/logging-monitoring.md](docs/logging-monitoring.md)
- [docs/admin-debug-exposure.md](docs/admin-debug-exposure.md)
- [docs/security-s1-s2-hardening.md](docs/security-s1-s2-hardening.md)
- [docs/security-s3-s5-hardening.md](docs/security-s3-s5-hardening.md)
- [docs/security-s6-s7-verification.md](docs/security-s6-s7-verification.md)
- [docs/dependency-image-scanning.md](docs/dependency-image-scanning.md)
- [docs/runtime-script-catalog.md](docs/runtime-script-catalog.md)

## Quick Start (Local)

### Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev -- --host 127.0.0.1 --port 5173
```

### Quick Checks

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/llm/options
curl -sS http://127.0.0.1:8000/health/deep
curl -sS http://127.0.0.1:8000/health/keys
```

## Validation

```bash
# full pytest suite
venv/bin/pytest -q

# default machine validation
venv/bin/python full_validation.py --mode machine --r 0

# browser/runtime validation
venv/bin/python full_validation.py --mode machine --profile full --frontend --r 0
```

Use `--live` only when intentionally testing real providers/quota behavior.

## Runtime Endpoints

- `GET /health/live`: liveness
- `GET /health/ready`: readiness/startup
- `GET /health`: lightweight runtime health
- `GET /health/deep`: deeper dependency diagnostics
- `GET /health/keys`: sanitized key-state summary
- `GET /llm/options`: routing/provider/options snapshot
- `GET /version`: commit + file mtime metadata

## Booking Contract (Canonical Summary)

- Supported booking path is SerpApi-first only.
- Plain `/ask` is search-only; booking resolution is lazy and intent-gated.
- Sync `/ask` success payload includes `best_flight`, `top_flights`, and non-null `all_flights` (ranked result order) for verification.
- Explicit cabin requests remain truthful: when requested cabin inventory is unavailable, response includes `constraint_outcomes.cabin` and warning text rather than silently implying cabin match.
- `booking_ready` requires a resolved non-Google provider URL.
- No Google Flights search-assist fallback URL is returned from booking/hold/track flows.
- `/booking/hold` is explicit about checkout readiness: `hold_created=true` always means local hold record exists, while `checkout_ready`/`checkout_status`/`hold_outcome` indicate whether provider checkout is currently available.
- `/booking/track-price` success means monitoring state is established (`tracking_state.route_tracking_ready=true`) and does not require checkout readiness.
- In-app confirm is not a product feature (`/booking/confirm` is intentionally absent).
- Raw Google click artifacts are not booking-ready proof targets.
- Local booking records are follow-up state only (`HELD` / `CANCELLED` / `EXPIRED`); provider-side booking completion is external and not confirmed in-app.

For full booking and operator behavior details, use [docs/operator-sheet.md](docs/operator-sheet.md).

## Additional Docs

- Config reference: [CONFIG.md](CONFIG.md)
- Operator sheet (canonical ops commands/endpoints): [docs/operator-sheet.md](docs/operator-sheet.md)
- Troubleshooting runbook: [docs/runbook.md](docs/runbook.md)
- Frontend usage details: [frontend/README.md](frontend/README.md)
- Monitoring quickstart: [monitoring/README.md](monitoring/README.md)

Canonical deployment smoke script:
- [scripts/deploy_smoke.sh](scripts/deploy_smoke.sh)

## RAGAS Baseline Scores

Pre-RAG baseline evaluation results (generated via `venv/bin/python full_validation.py --ragas-eval`).
Results are stored in `eval_results/ragas_baseline.json`.

| Metric | Score |
|--------|-------|
| Faithfulness | 0.000 |
| Answer Relevancy | 0.000 |
| Context Relevance | 0.000 |

> **Note:** Scores are 0.0 because no LLM was configured for RAGAS evaluation at baseline time.
> These will be populated with real numeric scores once an LLM backend is connected for RAGAS metric computation.
