# TECH-BRIEF — llm-travel-agent — Research Companion

> Written 2026-07-18. Purpose: a dense technical summary of scope + planned work so Sid can
> research technologies/frameworks/papers per domain. Each domain: current → target →
> chosen candidates → SEARCH TERMS. Decisions marked FROZEN are settled — research
> alternatives only as ADR material, don't relitigate (doctrine).
> Sources: STATUS-2026-07-18.md, SHIP-PLAN.md, ENTERPRISE-CHECKLIST.md, research/00+10.

## System in one paragraph

Python 3 / FastAPI monolith serving a multi-agent travel planner: NL query → regex+LLM intent
parsing → planner state machine (6,907-line module) orchestrating SerpAPI (Google Flights) and
OpenWeather tools → multi-provider LLM router (Ollama local → Gemini → OpenAI; priority
fallback, per-backend circuit breakers, atomic API-key rotation/cooldown) → explanation
streamed over SSE (string-sentinel protocol incl. `[DONE_JSON]`) to a React 19 + Vite + Framer
Motion frontend → booking handoff (SerpAPI token → provider checkout URL; NO payments).
Sidecars: custom asyncio job queue w/ SQLAlchemy persistence, price tracker loop, HITL
approval gate, local RAG (sentence-transformers MiniLM-L6-v2, 384-dim, numpy retriever),
session memory w/ deterministic summarization. Persistence: Postgres via sync SQLAlchemy
(psycopg2), models scattered, no migration framework. Infra: Docker Compose (postgres, api,
prometheus+grafana profile). Contract: SINGLE-NODE, process-local state (honest header:
`X-Ask-Contract: single-node-process-local`); multi-worker unsafe. Tests: 207 fast (stub LLM,
SQLite, auth off) + ungated slow suite + 340KB validation harness + RAGAS eval baseline.

## Hard constraints (frame every search with these)

- Solo dev + AI executors; budget ≈ ₹0 (free tiers only, per FREE-STACK-MAP).
- FROZEN: FastAPI stays; custom LLM router stays (it IS the portfolio value — litellm/gateway
  research = comparison ADR only); single-node contract until deliberate Phase-2 ADR.
- Ship first (SHIP-PLAN M0–M3), refactor after (D1). No payments ever in scope.
- Python backend — JS-ecosystem answers don't transfer here (except frontend).

## Domain map: current → target → search terms

### 1. Agent orchestration / planner decomposition (Phase 2)
Current: hand-rolled monolithic state machine; module-global state races; string-sentinel
stream protocol parsed by 3 scanners. Target: 9-module decomposition, typed stream events,
golden-master fixtures first.
SEARCH: `durable execution framework python` (Temporal, DBOS, Restate — study, likely
overkill), `LangGraph state machine vs custom`, `pydantic-ai agent framework`, `typed event
protocol SSE python`, `golden master testing refactor`, `agent orchestration patterns 2026`,
`saga pattern python`.

### 2. LLM routing & provider reliability (ship-adjacent: G3)
Current: 3 providers, breakers, key rotation, token budgeting. Target: +NVIDIA NIM, Groq
(free tiers) = 6-provider router; per-provider cooldown taxonomy stays.
SEARCH: `NVIDIA NIM API free tier`, `Groq API limits`, `OpenRouter free models`, `GitHub
Models API`, `LLM gateway comparison litellm portkey helicone` (ADR only), `semantic caching
LLM GPTCache`, `LLM fallback routing patterns`, `token bucket per-provider quota`.

### 3. Async correctness & data layer (Phase 0 item 0.2; Phase 2 async engine)
Current: sync SQLAlchemy writes ON the event loop (hottest paths); psycopg2; no migrations.
Target: to_thread everywhere now; later SQLAlchemy 2.x asyncio + psycopg3, Alembic baseline,
statement timeouts; CI grep-guard against `SessionLocal(` in `async def`.
SEARCH: `SQLAlchemy 2 asyncio psycopg3 migration`, `Alembic baseline existing database`,
`asyncio event loop blocking detection` (`asyncio debug mode`, `aiodebug`, `yappi`),
`FastAPI sync database threadpool pattern`, `Postgres statement_timeout SQLAlchemy`,
`semgrep custom rule async` (for the CI guard).

### 4. Background jobs & queues (Phase 1 WS-B.3)
Current: custom asyncio queue, serial worker, crash-loop >64 pending, contradictory
durability, SSE progress via in-process events. Target: arq (Redis) or procrastinate
(Postgres-only, no new infra) + pub/sub bridge for SSE progress; outbox thinking for
notification side-effects.
SEARCH: `arq vs procrastinate vs celery 2026`, `transactional outbox pattern postgres`,
`Postgres LISTEN NOTIFY job queue`, `SKIP LOCKED queue postgres`, `SSE progress updates
redis pubsub fastapi`, `idempotent consumer pattern`.

### 5. API contracts & input sanitation (M1 + WS-E)
Current: 2/33 endpoints typed; 3 error envelope shapes; /ask returns branch-dependent dict.
Target: pydantic request models on all public POSTs (M1), response models + ONE error
envelope + schemathesis in CI (WS-E); Idempotency-Key standardization (exemplar exists:
routes_booking_tracking.py).
SEARCH: `schemathesis CI fastapi`, `RFC 7807 problem details fastapi`, `idempotency key
middleware fastapi`, `OpenAPI contract testing`, `API versioning SSE payloads`.

### 6. Caching (WS-B.2 + M2 edge)
Current: 5 bespoke mechanisms; stampede lock released before fetch (F-013). Target: one
implementation (cashews candidate) + verified single-flight; Cloudflare edge cache for static.
SEARCH: `cashews python cache`, `cache stampede single flight python`, `stale-while-revalidate
API responses`, `Cloudflare cache rules free plan`.

### 7. Rate limiting, admission control, load handling (M1/M2)
Current: sliding-window limiter + admission control, process-local, fail-mode unaudited.
Target: fail-CLOSED on sensitive paths (MISTAKES 1.3); k6-documented capacity; later
Redis-backed for multi-worker.
SEARCH: `rate limit fail open vs fail closed`, `k6 SSE streaming load test`, `backpressure
asyncio server`, `GCRA rate limiting redis`, `load shedding admission control web service`.

### 8. Observability (M2 — the flagship fix)
Current: Prometheus/Grafana bundled but NEVER scraped (auth 403); KPI telemetry dies after
first request; structured logging w/ redaction exists. Target: internal metrics port, 6 alert
rules firing, UptimeRobot synthetics, Telegram ops alerts; later OTel traces w/ per-provider
LLM spans + TTFT metric.
SEARCH: `prometheus multiprocess uvicorn metrics port`, `OpenTelemetry FastAPI
instrumentation`, `LLM observability TTFT tokens per second metrics`, `Grafana alerting
telegram`, `SLO burn rate alerts single node`.

### 9. Security (M1/M2 + post-ship)
Current: strong headers/CORS/timing-safe/secret-fingerprints/pip-audit; AUTH_DISABLE master
switch, unbound HITL approvals, no dynamic testing, token-only auth. Target: Phase-0 auth
fixes; ZAP baseline CI → ZAP full+nuclei vs staging; trivy images; GitGuardian; sops+age for
config; optional OIDC later.
SEARCH: `OWASP ZAP baseline scan GitHub Actions`, `nuclei templates web app`, `trivy docker
image CI`, `sops age secrets git`, `fastapi OIDC keycloak zitadel`, `SSRF redirect
revalidation`, `LLM prompt injection tool-calling defenses` (HITL gate is the mitigation —
research current attacks).

### 10. Deployment & infra (M2 + Track B)
Current: never deployed; Compose + Caddy docs written. Target: VPS (Oracle-free ARM or
Hetzner) + Caddy TLS + ufw + Cloudflare proxied (DNS/DDoS/WAF/CDN) + nightly pg_dump w/
tested restore; Track B: Terraform replica vs floci (IAM/S3/SQS/ECS/Route53/CloudFront/WAF
v2/Secrets Mgr/KMS/CloudWatch), spot-validated on AWS free credits.
SEARCH: `docker compose production single VPS hardening`, `Caddy reverse proxy SSE
buffering`, `Cloudflare SSE streaming proxy timeout` (⚠ known gotcha — verify SSE through
CF free), `terraform provider aws endpoint override emulator`, `Oracle cloud always free ARM
docker`, `pg_dump restore verification script`.

### 11. Stateless topology (Phase 2 capstone — checklist #25)
Current: process-local queue/limits/idempotency/approvals; workers>1 unsafe (F-027).
Target ADR: externalize to Redis/Postgres → horizontal workers → disposable containers.
SEARCH: `stateless service externalize state redis postgres`, `sticky sessions vs shared
state SSE`, `twelve factor processes`, `postgres advisory locks distributed coordination`.

### 12. RAG & evaluation (post-ship; resume-1 artifact)
Current: MiniLM/numpy retriever, silently dead in prod (deps undeclared); RAGAS baseline
exists (`--with-rag`). Target: declare-or-remove decision (0.4); then eval write-up
(faithfulness/relevance), maybe hybrid retrieval.
SEARCH: `RAGAS metrics interpretation`, `hybrid search BM25 dense fusion`, `reranker
cross-encoder small local`, `RAG evaluation papers 2025 2026`, `LLM-as-judge calibration`.

### 13. Frontend (M3)
Current: React 19 + Vite + SSE + Framer Motion, unmeasured. Target: Lighthouse CI, TTFT UX,
PostHog analytics post-ship.
SEARCH: `Lighthouse CI vite react`, `SSE EventSource reconnect backoff react`, `streaming
UI perceived performance patterns`, `PostHog react vite setup`.

### 14. Testing depth (M1 + post-ship)
Current: fast suite green-but-shallow (stub LLM/SQLite/auth-off); slow suite ungated;
validation harness. Target: tests_slow + Postgres service in CI (0.7); Testcontainers;
schemathesis; browser-use nightly smoke vs live URL.
SEARCH: `testcontainers python postgres pytest`, `pytest asyncio flaky patterns`,
`browser-use scheduled smoke test`, `contract testing SSE streams`, `mutation testing python
mutmut` (stretch).

## Search-scope guard

Skip categories (decided): Kubernetes, microservices, GraphQL, second BaaS, JS rewrites,
litellm-as-replacement, payments, airline scraping (ToS). New-tech findings route through the
decision-gate habit: cheap prototype → Sid verdict → frozen.
