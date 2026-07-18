# MVP-ROADMAP — llm-travel-agent — 3 Milestones to Shipped

> Written 2026-07-18. THE consolidated execution roadmap. Supersedes SHIP-PLAN's M0–M3 numbering
> by folding M0 into Milestone 1 (SHIP-PLAN's gates/DON'T-WANTs still bind). Inputs, all vetted:
> `research/10-improvement-roadmap.md` (Phase-0 specs), `ENTERPRISE-CHECKLIST.md`,
> `FREE-STACK-MAP.md` (+§7 matrix), `RESEARCH-INTAKE-2026-07-18.md`, `TECH-BRIEF.md`.
> Owners: **DS** = DeepSeek/GLM executor (per-milestone plan via `executor-plan` skill),
> **C** = Claude (verify/plan), **S** = Sid (decisions, accounts, approvals).

## 0. Readiness inventory — do we have everything? YES, except 3 Sid-items

### Have (vetted and slotted)

| Category | Assets | Status |
|---|---|---|
| Task specs | Phase-0 0.1–0.12 fully specified with findings + efforts in `research/10` + findings.json | ✅ |
| Repos (external) | sse-starlette, Semgrep, OWASP ZAP (+baseline Action), nuclei, trivy, GitGuardian, k6, Caddy, Terraform, floci (service list verified), browser-use, free-for-dev, Testcontainers, Schemathesis, Cashews, Procrastinate/arq, Alembic | ✅ all OSS/free |
| Repos (Sid's) | smart-job-scanner-v2 (Telegram bot transplant), merlin-cli-bridge (dev-only inference), in-repo idempotency exemplar (`routes_booking_tracking.py`) | ✅ |
| Services (free, per FREE-STACK-MAP §7) | Cloudflare (DNS/DDoS/CDN/WAF-rules), UptimeRobot, Neon/self-host PG, R2, Resend, Grafana(self-host in repo), Oracle-free/Hetzner VPS, NIM/Groq/Gemini free LLM tiers | ✅ mapped |
| Research | SSE spec patterns (Last-Event-ID/typed events/pings — vetted in RESEARCH-INTAKE), OWASP LLM prompt-injection guidance, TTFT metrics defs | ✅ |
| Papers | None required for MVP. (RAGAS eval write-up + ADRs are post-ship artifacts; their references are already in TECH-BRIEF §12.) | ✅ n/a |
| Docs in-repo | deployment-topology, reverse-proxy-caddy, persistence-backups, environment-secrets-contract, runbook, security S1–S7 | ✅ |

### Missing (blocks, all Sid's)

1. **⛔ G1** — approve Phase-0 scope (recommend: all 12). Blocks Milestone 1 execution.
2. **⛔ G2 + accounts** — host choice (Oracle-free ARM vs Hetzner ~$5) + Cloudflare account + domain decision. Blocks Milestone 2.
3. **⛔ G3 + keys** — prod LLM routing choice (recommend NIM+Groq+Gemini free) + API keys created by Sid. Blocks Milestone 2 routing task.
4. **⛔ G6 (added 2026-07-18)** — frontend REBUILD placement (Sid decided the whole frontend
   will be rebuilt; M1 is backend-only, current frontend frozen as legacy). Pre-ship = MVP
   ships the NEW UI (M3 captures it, adds a rebuild workstream before M3; the rebuild should
   consume `/openapi.json` + could adopt the typed-SSE protocol from RESEARCH-INTAKE §1
   directly instead of retrofitting later). Post-ship = MVP ships the current working UI,
   rebuild becomes the first post-ship workstream. Blocks M3 asset capture only.

Verify-before-commit items (10 min each, inside M2): Cloudflare free-plan SSE pass-through;
Oracle free-tier ARM capacity actually claimable; current NIM/Groq free limits.

## MVP definition (unchanged from SHIP-PLAN §1)

Live public HTTPS demo a stranger can use end-to-end + working Grafana + fake-green killed +
honest README with REAL captures + uptime alerts + tested backups. All proven by
screenshots/recordings/pasted output. NOT in MVP: §5 list.

---

## MILESTONE 1 — Truth & Hardening (est. ~2 weeks incl. buffer) — starts at ⛔ G1

Goal: repo tells the truth; every known Phase-0 defect fixed; CI means something.

### M1-A: Repo truth (half day) — owner DS, verify C

| ID | Task | Spec | Effort | Proof |
|---|---|---|---|---|
| A1 | Triage + commit dangling changes (README, App.tsx, 2×css, PORTFOLIO-SUMMARY.md) | STATUS git state | 1h | `git status` clean, `git log` pasted |
| A2 | Delete artifacts: README.md.backup, COMMIT, local.db, startup_debug.log + dead modules (services_booking.py, circuit_manager.py after de-shadow check, gemini adapter, gateway shims, async_llm_client) + fix `.env.example` provider chain | roadmap 0.8 | 1d | `ls` proofs; suite still green |
| A3 | README honesty pass: strip "production-grade", remove 3 phantom image embeds, status = STATUS file language | MISTAKES 1.12, D2/D3 | 2h | diff pasted |

### M1-B: Phase-0 fixes (the audit's 12, in order) — owner DS, verify C

| ID | Task | Fixes | Effort |
|---|---|---|---|
| B1 (0.1) | `log_event()` out of swallowed-exception blocks (2 sites) | F-008 KPI loss | 30m |
| B2 (0.2) | `to_thread` ALL loop-blockers: key-manager upsert (outside locks), price tracker, `record_price_snapshot`, RAG retrieve, HITL/KPI writers | F-001/002/024 | 1–2d |
| B3 (0.3) | Prometheus scrape fix: internal metrics port (2nd uvicorn app or start_http_server) or scrape credentials; target UP | F-006 | 0.5d |
| B4 (0.4) | Dependency surgery: remove tornado/pillow/protobuf/orjson/urllib3; declare pydantic; drop starlette pin; dev-extra for requests+validation; RAG deps decision (declare or disable LOUDLY); lockfile workflow. **Then recreate venv at new path** (STATUS ⚠) | F-007/003 | 0.5d |
| B5 (0.5) | Queue safety: `await put()` on rehydrate (kills >64-job boot crash-loop); per-job `asyncio.wait_for` timeout; ONE durability truth in payload+headers | F-005 | 1d |
| B6 (0.6) | AUTH_DISABLE → only under TESTING=1; split admin/bearer bypass; red startup banner | F-009 | 0.5d |
| B7 (0.7) | CI: tests_slow job + Postgres service container + ruff gate + built-image import check | F-010 | 1d |
| B8 (0.9) | `session_memory.cleanup_expired()` on price-tracker cadence; cap per-session messages | F-015 | 2h |
| B9 (0.10) | Fail fast if DATABASE_URL unset outside TESTING; align .env.example w/ compose | F-030 | 2h |
| B10 (0.11) | **Semgrep** CI rule: no `SessionLocal(`/`session.query(` in `async def` without to_thread (INTAKE §4.2) | guards B2 | 2h |
| B11 (0.12) | Bind HITL approvals to owning principal in `/plan/{id}/approve` | authz gap | 0.5d |

### M1-C: Checklist hardening adds — owner DS, verify C

| ID | Task | Spec | Effort | Proof |
|---|---|---|---|---|
| C1 | Pydantic request models on ALL public POST endpoints (start from booking-tracking exemplar) | CHECKLIST #11 | 1d | schemathesis-lite smoke or curl matrix |
| C2 | Rate-limit fail-mode audit → fail-CLOSED on sensitive paths | CHECKLIST #12, MISTAKES 1.3 | 2h | test output: limiter backend down ⇒ 429/503 |
| C3 | ZAP baseline job in GitHub Actions | CHECKLIST #5 | 2h | CI run URL |

**M1 exit proof:** CI green WITH tests_slow+Postgres; before/after /ask latency under
`pg_sleep` DB pressure (B2 proven); Prometheus target UP screenshot; `git log` of all tasks.

---

## MILESTONE 2 — Deploy & Prove (est. ~1 week) — starts at ⛔ G2 + ⛔ G3

Goal: public URL, observable, defended, backed up, cheap-to-free.

| ID | Task | Spec/Input | Effort | Owner |
|---|---|---|---|---|
| D1 | SSE resilience trio on BOTH stream endpoints (/ask, /jobs/{id}/events): keep-alive comment pings 15–30s, `X-Accel-Buffering: no`, `Cache-Control: no-cache` | INTAKE §4.1 (neither endpoint has them today) | 0.5d | DS |
| D2 | Router adapters: NVIDIA NIM + Groq backends behind existing breaker/rotation (keys from Sid, G3). **Gemini adapter must be BUILT, not enabled** — found at M1 STOP-1c (D-M1-11): the only gemini path is a legacy helper whose source module is deleted from the tree (flag off, `.pyc` ghost only); today's runtime router is effectively Ollama→OpenAI. Gemini via official SDK/REST behind the same breaker/rotation (key machinery already exists: `core/api_key_manager.py` GEMINI_KEY_n rotation, quota-scope handling in `agents/cloud_llm.py:173-283`) | CHECKLIST §3; M1 D-M1-11 | 1.5d | DS |
| D3 | Provision VPS (G2): Docker+Compose, ufw (80/443/SSH), SSH keys-only, fail2ban | deployment-topology.md | 0.5d | S+DS |
| D4 | Caddy TLS per `reverse-proxy-caddy.md`; Cloudflare DNS **proxied** (DDoS+CDN+firewall rules); SSE pass-through verified (cache-bypass rule for stream paths) | FREE-STACK-MAP §7 | 0.5d | DS |
| D5 | Secrets via docker secrets per `environment-secrets-contract.md`; `.env` never leaves the box; GitGuardian scan on repo | S1–S7 docs | 2h | DS |
| D6 | Monitoring LIVE: compose monitoring profile up; Grafana dashboards w/ real scrape; 6 alert rules able to fire; UptimeRobot on /health + /ask synthetic; Telegram ops alerts (smart-job-scanner-v2 transplant) | B3 + CHECKLIST #17 | 1d | DS |
| D7 | Backups: nightly pg_dump cron + offsite copy (R2 free 10GB) + **ONE RESTORE ACTUALLY TESTED** | persistence-backups.md, WATCH-OUT A1 | 0.5d | DS |
| D8 | k6 smoke: /ask + SSE profile; document sustained-RPS capacity number | CHECKLIST #4/#6 | 0.5d | DS |
| D9 | Dynamic security pass vs staging: ZAP full + nuclei + trivy on image; Mozilla Observatory + Qualys once DNS live | CHECKLIST #5/#24 | 0.5d | DS |
| D10 | Proxy-timeout smoke: idle stream > proxy timeout through Caddy+CF survives via pings | INTAKE §1 | 2h | DS |

**M2 exit proof:** recording of a stranger-flow query on the public URL; Grafana screenshot
(real data); restore-test output; uptime+Telegram alert screenshots; k6 + ZAP outputs; capacity
number written into README.

---

## MILESTONE 3 — Demo, Truth & Launch (est. ~3–4 days) — ends at ⛔ G5

| ID | Task | Spec | Effort | Owner |
|---|---|---|---|---|
| E1 | Real captures from LIVE URL: streaming GIF, frontend screenshot, architecture SVG (render PORTFOLIO-SUMMARY mermaid). Redaction check vs PORTFOLIO-SUMMARY §8 FIRST | D3/D4 rules | 0.5d | S+C |
| E2 | ⛔ G4: Sid approves each capture before README embed | proof standard | — | S |
| E3 | README final: assets embedded, claims=reality, quickstart verified on CLEAN CLONE | D2 | 0.5d | DS+C |
| E4 | Lighthouse CI on frontend; fix cheap wins only (no scope creep) | CHECKLIST #15 | 0.5d | DS |
| E5 | **Resume truth pass**: resume_1 + resume_2 bullets per CHECKLIST §0 (production claims now TRUE; LangGraph/fcntl bullets reworded); PORTFOLIO-SUMMARY + portfolio-website case page get the live URL | CHECKLIST §0 | 2h | S+C |
| E6 | Launch checklist: signoff-sweep style pass — all M1/M2 proofs linked, git tagged `v1.0-mvp`, main frozen | G5 | 2h | C+S |
| E7 | ⛔ G5: Sid says ship | — | — | S |

**M3 exit = MVP SHIPPED.** Post-ship track (SHIP-PLAN §6, 12 items) begins only after G5.

---

## §4. Dependencies & critical path

```
⛔G1 → M1-A (A1→A2→A3) → M1-B (B1→B2→B3→B4(+venv)→B5→B6→B7→B8..B11) → M1-C → M1 exit
⛔G2/⛔G3 (parallel with late M1) → M2: D1,D2 (code) ∥ D3→D4→D5 (infra) → D6→D7→D8→D9→D10 → M2 exit
→ M3: E1→⛔G4→E3→E4→E5→E6→⛔G5 → SHIPPED
```
Critical path ≈ 3–4 calendar weeks solo+executors. B2 (loop-blockers) and D4 (Cloudflare SSE)
are the riskiest tasks; both have verification steps built in.

## §5. NOT in MVP (frozen out — SHIP-PLAN §6 owns these)

Alembic, Terraform+floci Track B, queue replacement, webhook feature, edge Worker, PostHog,
k6 full profile, browser-use smoke, SSE protocol migration (typed events/Last-Event-ID),
prompt-injection screening, eval write-up + ADRs, Phase-1/2 refactors, stateless topology.

## §6. Execution mechanics

At each gate approval, C converts that milestone into a cold-start executor plan
(`executor-plan` skill; WS-1/T12 quality bar): self-contained task blocks, ONLY files named,
⛔ STOP/SWITCH markers, proof lines required per N19, evidence rules per m5 N23. Model split
per M5.11 precedent: DeepSeek @ HIGH for mechanical batches (M1-B, M2 infra), GLM MAX if a
taste/judgment batch appears. Sid runs `check-verify` (or asks C) on every executor STOP report.
