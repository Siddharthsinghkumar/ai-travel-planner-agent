# Deferred Architecture Decisions

## Multi-agent framework (LangGraph / AutoGen)

**Current state:** The planner is a modular pipeline with role-separated stages implemented within a single module:

1. **Intent parsing** (`agents/planner_agent.py:3447-3588`) — Extracts origin/destination IATA codes, dates, cabin preferences, and stopover info from the user query using regex and heuristic matching.
2. **Flight search** (`agents/planner_agent.py:3961-4197`) — Queries the airline API (SerpApi or direct) for flights matching the parsed intent, with parallel fetching for multi-leg routes.
3. **Weather fetch** (`agents/planner_agent.py:4045-4299`) — Retrieves forecast data for destination and return origin using the weather API tool.
4. **Scoring and ranking** (`agents/planner_agent.py:4300-4560`) — Applies preference-aware scoring (price, duration, stops, cabin match) and ranks candidates.
5. **HITL approval gate** (`agents/planner_agent.py:4576-4590`) — Blocks on user approval before booking handoff resolution.
6. **Booking handoff resolution** (`agents/planner_agent.py:4590-4855`) — Probes top-ranked flights for booking URLs and artifacts.
7. **LLM explanation generation** (`agents/planner_agent.py:2495-2800`) — Assembles a prompt with flight data, weather, RAG context, and user query; calls the LLM for a natural-language response.

**What LangGraph would change:** An explicit `StateGraph` with named nodes for each stage, framework-managed checkpointing between nodes, parallel branch execution for independent tools (flight search + weather), and built-in retry/fallback routing. The current async/await pattern already achieves parallelism (`asyncio.gather`), but LangGraph would make the graph topology declarative and inspectable.

**Cost:** Estimated 1-2 weeks of refactor. Every stage would need to be converted to a LangGraph node with explicit input/output schemas. The existing test suite (26 booking handoff tests, intent parsing tests, scoring tests) would need to be re-tested against the new graph execution model. Integration tests for the HITL gate would need rewriting since the approval state would live in the graph state rather than an `ApprovalState` store.

**Migration trigger:** "Planning depth exceeds 5 sequential LLM calls" or "parallel tool execution becomes a measured bottleneck on traces > 5s." At current complexity (3-4 sequential stages, 2 parallel tool calls), the async/await pattern is sufficient.

**Decision:** Deferred until a migration trigger is met. The planner is already modular and testable; LangGraph adds framework overhead before RAG/eval quality is proven.

## Microservices split

**Current state:** Single-process FastAPI service (`api/app.py`) running on one `uvicorn` worker. Modules that would become independent services if split:

| Module | Current role | Service boundary rationale |
|--------|-------------|---------------------------|
| `api/` | HTTP routing, SSE streaming, request validation | API gateway — handles auth, rate limiting, response formatting |
| `agents/` | Planner logic, intent parsing, scoring, LLM routing | Planning service — CPU/GPU bound, LLM call orchestration |
| `tools/` | Airline API, weather API, booking handoff, price tracking | Tool service — I/O bound, external API calls, caching |
| `core/` | Health checks, metrics, circuit breaker, key management | Infrastructure service — cross-cutting concerns |

**What a split would change:** Independent scaling (e.g., more tool service replicas for high API call volume), isolated failure domains (a weather API outage wouldn't crash the planner), and language flexibility per service (e.g., a Go tool service for high-throughput API calls).

**Cost:** Distributed state management (session state, approval state, booking state would need Redis or a database), network latency between services (adding 10-50ms per inter-service call), deployment complexity (container orchestration via Kubernetes, service mesh for mTLS, distributed tracing via Jaeger), and observability overhead (aggregating logs/metrics across services).

**Migration trigger:** "Multi-instance deployment becomes required for SLA reasons" (e.g., > 100 concurrent users requiring horizontal scaling) or "team size exceeds the point where shared codebase coordination is the bottleneck" (e.g., > 6 engineers working on distinct modules).

**Decision:** Deferred. Single-node is the correct architecture at current load and team size. The migration path is: extract stateless tool handlers first (airline API, weather API), then session state (approval store, booking state), then the planner itself.

## Operational KPI telemetry (manual-effort reduction)

**Current state:** Instrumentation is in place via `core/kpi_telemetry.py` (JSONL event logger) and `scripts/kpi_summary.py` (baseline analysis script). The planner emits events at five lifecycle points: `plan_start`, `approval_requested`, `approval_decision`, `plan_complete`, and `plan_error`.

**Why the 40% target is not yet measured:** There is no production traffic baseline. All validation runs use placeholder data or mock responses. Without real user sessions, metrics like "time to first response," "approval cycle duration," and "actions per session" are undefined.

**What needs to be added when traffic exists:**

1. **Event timestamps** for approval cycles — record `approval_requested_at`, `approval_responded_at`, and compute `approval_latency_ms`. Currently the `ApprovalState` class (`agents/planner_agent.py:124`) tracks decisions but not timestamps.
2. **Time-to-completion per plan** — emit a metric at the end of `plan_trip()` with total wall-clock time broken down by phase (intent parsing, flight search, weather fetch, scoring, LLM generation). The `phases` dict in `_plan_trip_internal` already tracks this internally but does not export it.
3. **Action count per session** — count how many tool calls (flight searches, weather fetches, booking probes) are made per user query. This is implicitly tracked by the number of API calls but not aggregated.
4. **Longitudinal aggregation** — a dashboard (Grafana) showing trends in approval latency, plan completion rate, and error rate over time. The existing Prometheus metrics (`core/metrics.py`) track request counts and latencies but not plan-level KPIs.

**Decision:** Instrumentation in place; baseline analysis pending production traffic. The 40% manual-effort reduction target requires a comparison baseline that does not yet exist. See `core/kpi_telemetry.py` and `scripts/kpi_summary.py`.
