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

## Evaluation Results

RAGAS evaluation results comparing baseline (no RAG) vs RAG-enhanced retrieval.
Results stored in `eval_results/ragas_baseline.json` and `eval_results/ragas_with_rag.json`.

| Metric | Baseline (no RAG) | With RAG | Delta |
|--------|-------------------|----------|-------|
| Faithfulness | heuristic | heuristic | varies |
| Answer Relevancy | heuristic | heuristic | varies |
| Context Relevance | heuristic | heuristic | varies |

> When no LLM is configured for RAGAS metric computation, a rule-based heuristic scorer produces scores in the 0.3–0.85 range based on keyword overlap, answer length, and context relevance. Run with `--ragas-eval` (baseline) or `--ragas-eval --with-rag` (RAG-enhanced).
> RAG corpus expanded to 102 chunks across 15 files covering baggage, visa, cancellation, seat/loyalty, disruption rights, airport/transport, insurance/health, booking/pricing, Asia-Pacific, Europe, Americas, Middle East/Africa, technology, family/special needs, cruise/alternative transport, safety/security, currency/money, and photography/etiquette.

## Architecture Decisions

### HITL approval gate

Implemented in `agents/planner_agent.py` (ApprovalState class, pending_approval state before booking handoff), `api/app.py` (POST /plan/{plan_id}/approve endpoint with bearer auth), and frontend `AIReasoningPanel.tsx` (Approve/Reject UI). Triggered automatically before every high-impact booking tool call; the planner blocks on an asyncio.Event until the user approves or rejects via the UI or API. All booking/payment functions are decorated with `@high_impact` from `agents/high_impact.py` with single-use approval gating.

### RAG pipeline

Corpus located at `rag/corpus/` (15 files, 102+ chunks). Embedding model: `all-MiniLM-L6-v2`. Retrieval strategy: numpy cosine similarity over pre-computed chunk embeddings. RAG is wired into all prompt sites: `generate_explanation()` (non-streaming), `stream_generator()` (streaming), with context injected before user query. Eval includes 10 RAG-grounded test cases (RAG_GROUNDED_CASES) with ground-truth answers.

### State machine runtime enforcement

Planner state is tracked via `agents/state_machine.py` (PlannerState enum, VALID_TRANSITIONS, transition function). States: idle → intent_parsing → planning → pending_approval → executing → complete/rejected/error. Transitions are enforced at runtime; illegal transitions raise `IllegalTransition`. See [docs/architecture/planner_state_diagram.md](docs/architecture/planner_state_diagram.md).

### Session memory summarization

Implemented in `core/session_memory.py` (`SessionMemory` class). Token-budgeted window with deterministic summarization: recent messages (~70%) kept intact, older messages truncated proportionally. Configured via `SessionMemory(max_tokens=4000, summary_ratio=0.3, ttl_seconds=1800)`. Wired into `plan_trip()` via optional `session_id` parameter; context is injected into LLM prompts after RAG context. No LLM calls for summarization — fully deterministic and fast. Test with `--session-memory-test`.

### HITL audit logging

Implemented in `core/hitl_audit.py` (`HITLAuditLogger` class). Structured JSONL audit trail under `logs/hitl_audit/audit_YYYY-MM-DD.jsonl`. Logs both approval requests and decisions with latency tracking. Metrics include approval rate, rejection rate, and p50/p95 latency. Wired into `POST /plan/{plan_id}/approve` (`api/app.py`) and planner HITL gate (`agents/planner_agent.py`). Summary CLI: `python scripts/hitl_audit_summary.py [--date YYYY-MM-DD] [--json]`. Test with `--hitl-audit-test`.

### KPI telemetry

Instrumentation in place via `core/kpi_telemetry.py` (JSONL event logger) and `scripts/kpi_summary.py` (baseline analysis). Events emitted at plan_start, approval_requested, approval_decision, plan_complete, and plan_error. Baseline analysis pending production traffic. See [docs/architecture/deferred.md](docs/architecture/deferred.md).

### Intentionally deferred

- **Microservices split**: Reason — single-node is appropriate until multi-instance load requires it. Migration path — extract stateless tools/handlers first, then session state. See [docs/architecture/deferred.md](docs/architecture/deferred.md).
- **Multi-agent framework**: Reason — planner is already modular; LangGraph adds overhead before RAG/eval quality is proven. See [docs/architecture/deferred.md](docs/architecture/deferred.md).

## Future Work

- **RAGAS LLM-based scoring**: Connect an LLM backend to enable actual RAGAS metric computation (faithfulness, answer_relevancy, context_relevancy) for meaningful baseline vs with-RAG comparison. Rule-based heuristic fallback is now available when LLM is unavailable.
- **RAG reranker evaluation**: Evaluate a cross-encoder reranker (e.g., `ms-marco-MiniLM-L-6-v2`) to improve top-k precision beyond cosine similarity.
- **KPI longitudinal tracking**: Connect KPI telemetry to Prometheus/Grafana for real-time dashboard once production traffic exists.
- **Session memory persistence**: Current session memory is in-memory only; consider SQLite-backed persistence for cross-restart session continuity.
- **HITL audit retention policy**: Add configurable retention (e.g., 30-day auto-cleanup) for audit JSONL files.
