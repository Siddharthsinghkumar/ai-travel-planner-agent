# RESEARCH-INTAKE — 2026-07-18 — SSE Deep-Dive + 13-Domain Modernization Roadmap

> Sid supplied two external research outputs (SSE/reconnect patterns; 13-domain roadmap with
> DBOS/GPTCache/Procrastinate/Cashews/SlowAPI/OTel/etc.). This file is the VETTED intake:
> each item checked against the codebase, the audit (research/00+10), and frozen decisions.
> Verdicts: **ADOPT-SHIP** (into M0–M3) · **ADOPT-POST** (slotted post-ship) ·
> **MODIFIED** (right idea, wrong diagnosis/timing) · **ADR-ONLY** · **REJECT** · **UNVERIFIED**.
> Raw research preserved in `research/inputs/` conceptually — this file is the actionable truth.

## 0. Code reality check (verified 2026-07-18, greps against the tree)

- **Two stream channels, different rules:**
  - `frontend/src/hooks/useAsyncJob.tsx` → native `EventSource` on `/jobs/{id}/events` —
    browser auto-reconnect + `Last-Event-ID` header apply natively. **Ideal first adopter.**
  - `frontend/src/hooks/useStreamingPlan.tsx` → `fetch` + `response.body.getReader()` on /ask —
    NO native reconnect; client must implement resume + send `Last-Event-ID` itself.
- Server streams via raw `StreamingResponse(media_type="text/event-stream")`
  (`api/app.py:2181, 2615`) — **sse-starlette is NOT used**.
- `[DONE_JSON]` sentinel embedded inside data frames (`api/app.py:2087–2170`), parsed by
  frontend AND job queue (roadmap migration caution stands: golden-master fixtures before retyping).
- **No keep-alive pings, no `X-Accel-Buffering: no`, no explicit `Cache-Control: no-cache`
  on the stream responses** — this WOULD have broken/stalled behind Caddy/Cloudflare at M2.

## 1. SSE research — verdict: the best find of the batch

| Item | Verdict | Placement |
|---|---|---|
| Keep-alive comment pings (every 15–30s) + `X-Accel-Buffering: no` + `Cache-Control: no-cache` on both stream endpoints | **ADOPT-SHIP** | **M2 prerequisite** — additive, low-risk, required for the proxy chain (Caddy + Cloudflare). Added to SHIP-PLAN M2. |
| Typed event framing (`event:` field per message type) + `id:` on every event + `Last-Event-ID` resume with a server-side event buffer (ring buffer / Redis / DB log) | **ADOPT-POST** | New post-ship item: **job-events channel FIRST** (native EventSource, small blast radius), then /ask during WS-A/WS-E with golden-master fixtures — replaces the `[DONE_JSON]` string-sentinel protocol (F-031). Do NOT do pre-ship (D1; two consumers depend on the sentinel). |
| sse-starlette (`EventSourceResponse`, `ServerSentEvent`) as the server implementation | **ADOPT-POST** | Same item as above. Note: research attributed `EventSourceResponse` to FastAPI core docs — it's sse-starlette's; minor citation error, right library. |
| /ask fetch-client resume: manual `Last-Event-ID` header on retry + exponential backoff | **ADOPT-POST** | Required because /ask does NOT use EventSource (see §0) — the research assumed native EventSource everywhere; reality is split. |
| Reconnect validation tests (mid-stream disconnect: no dup/no loss by event-id; proxy-timeout test with short-timeout Nginx/Caddy) | **ADOPT-POST** | Ship the test recipe with the protocol migration; the proxy-timeout smoke also runs at M2 against pings. |
| `misina` client library | **UNVERIFIED** | Repo not verified; treat as pattern reference only. langserve (typed multi-event streams) and ag-ui (keep-alive/buffering lessons) are real and good pattern sources. |

## 2. The 13-domain roadmap — verdicts

| Domain / proposal | Verdict | Notes |
|---|---|---|
| **DBOS Transact** for orchestration (also Temporal, self-hosted) | **ADR-ONLY** | Conflicts with FROZEN custom-planner value + the audit's own decomposition plan (9 modules, golden-master first). DBOS is a legit Phase-2 ADR candidate for the workflow/queue side (Postgres-native, no new infra). Not a ship item; "port user journeys to DBOS" pre-ship violates D1. |
| **GPTCache** semantic caching on LLM calls | **REJECT** (for the product path) | Flight prices/availability are time-sensitive — semantically-similar-query cache = confidently stale answers about money. At most for static informational content later, behind an explicit freshness rule. Research oversold "no code changes, dramatic savings." |
| **NIM + Groq** as router backends | ADOPT (already in plan) | Matches ENTERPRISE-CHECKLIST §3 / G3. |
| **SQLAlchemy 2 asyncio + asyncpg/psycopg3**; `to_thread` interim | ADOPT (already in plan) | = Phase-0 0.2 interim + Phase-2 async engine. Research confirms sequencing. |
| **Alembic `stamp head` baseline** on live schema | **ADOPT-POST** | Concrete technique for post-ship item #1 (WS-D). Good spec detail. |
| **Semgrep rule** for sync-DB-in-async guard | **ADOPT-SHIP** | Upgrades 0.11 from grep to semgrep. Noted in M1. |
| **Procrastinate** (Postgres-only) over arq unless Redis adopted anyway | **ADOPT-POST** | Matches audit WS-B.3's own framing exactly; leaning Procrastinate = zero new infra. Decision finalizes when the Redis question (limits/cache) is settled. |
| **Transactional outbox + `SKIP LOCKED`** patterns | **ADOPT-POST** | Design vocabulary for WS-B.3 + notification side-effects. |
| **Pydantic models all endpoints; RFC 7807 single error envelope; Schemathesis in CI** | ADOPT (already in plan) | M1 (public POSTs) + WS-E. RFC 7807 is a good concretization of "ONE envelope". `fastapi-idempotency` package UNVERIFIED — reuse the in-repo exemplar (`routes_booking_tracking.py`) instead; own pattern first. |
| **Cashews** `@cache.early` / `@cache.locked` | ADOPT (already in plan) | WS-B.2; research confirms the stampede fix mechanics (single-flight + background refresh). |
| **SlowAPI / fastapi-limiter** replacing rate limiting | **MODIFIED** | Misdiagnosis: a custom sliding-window limiter + admission control EXISTS and wasn't flagged for replacement — the M1 task is a fail-mode audit (fail-closed), not a rewrite. SlowAPI/limits becomes relevant only at Phase-2 state externalization (Redis-backed limits). |
| **prometheus-fastapi-instrumentator** ("no metrics exposed") | **MODIFIED** | Misdiagnosis: metrics + 6 alert rules EXIST; the failure is the scrape 403 (F-006). Fix = 0.3. Instrumentator is an optional complement for HTTP-level histograms after 0.3, not the cure. |
| **OTel FastAPI + HTTPX/SQLAlchemy instrumentation; TTFT histogram** | ADOPT (already in plan) | Post-ship observability; TTFT definition (query→first token) confirmed. |
| **ZAP baseline CI, nuclei, trivy, secret scanning** | ADOPT (already in plan) | M1/M2 per ENTERPRISE-CHECKLIST. |
| **Prompt-injection guardrails** (OWASP LLM cheat sheet; input screening; NeMo Guardrails later) | **ADOPT-POST** | New explicit post-ship security item — HITL gate is the existing mitigation; add input screening + the OWASP checklist to the security track. |
| **sops+age, OIDC (Keycloak)** | ADOPT (already in plan) | Post-ship options as previously mapped. |
| **Terraform + LocalStack (`tflocal`)** | **MODIFIED** | We standardize on **floci** (Sid-designated; broader free service list verified). LocalStack noted as fallback if floci gaps appear; the `tflocal`-style endpoint-override technique transfers to floci directly. |
| **Cloudflare SSE buffering caution** | ADOPT-SHIP | Independently confirms the G2 gotcha: verify SSE end-to-end through Cloudflare (bypass cache rule) before committing; pings from §1 are the mitigation. |
| **Advisory locks (`pg_try_advisory_lock`), twelve-factor, 2+ replicas** | ADOPT (already in plan) | = WS-B.5 + Phase-2 topology ADR (checklist #25). Research adds the leader-election framing — good. |
| **Hybrid BM25+dense retrieval, cross-encoder rerank (ms-marco-MiniLM)** | **ADOPT-POST** | AFTER the 0.4 declare-or-remove decision on RAG. Concrete model pick is useful. |
| **Lighthouse CI, PostHog (self-host), EventSource backoff, frontend Sentry** | ADOPT (already in plan) | M3 + post-ship. |
| **Testcontainers (Postgres/Redis), Playwright nightly e2e** | ADOPT + nuance | Testcontainers = post-ship test depth. E2E: **Playwright for deterministic CI** + **browser-use as the AI-driven showcase** — different jobs, keep both. |

## 3. The one big rejection: the research's sequencing

Its "Immediate (1–2 weeks)" list front-loads NEW capability adoption (Cashews, GPTCache,
SlowAPI, instrumentator, Schemathesis, Testcontainers) **while the audit's Phase-0 defects
would still be live** — event loop blocking on every key release, Prometheus scrape 403,
queue crash-loop, AUTH_DISABLE. It never saw findings.json. **Our order stands:**
M0 repo truth → M1 Phase 0 (fix what's broken) → M2 deploy → M3 assets/truth → post-ship
adoption in the slots above. Content adopted; calendar rejected.

## 4. Net-new items added to the plan by this intake

1. **M2:** SSE resilience trio on both stream endpoints — keep-alive comment pings,
   `X-Accel-Buffering: no`, `Cache-Control: no-cache` + proxy-timeout smoke test. (SHIP-PLAN updated)
2. **M1:** 0.11 guard implemented as a Semgrep rule rather than grep.
3. **Post-ship:** SSE protocol migration — sse-starlette + typed `event:` frames + `id:` +
   `Last-Event-ID` resume w/ event buffer; job-events channel first, /ask second behind
   golden-master fixtures; reconnect validation tests (dup/loss/proxy-timeout). (SHIP-PLAN updated)
4. **Post-ship security:** prompt-injection input screening per OWASP LLM guidance.
5. **ADR backlog:** DBOS-vs-custom-planner (Phase 2), litellm/portkey comparison (existing),
   Procrastinate-vs-arq final call tied to the Redis decision.
