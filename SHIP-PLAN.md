# SHIP PLAN — llm-travel-agent

> Written 2026-07-18. Inputs: `STATUS-2026-07-18.md`, `research/10-improvement-roadmap.md`
> (Phase 0 is the backbone — the audit already sequenced the work; this plan scopes it to
> "shipped" and adds deploy/demo/stack-expansion), `FREE-STACK-MAP.md`,
> `../EXPAND-STACK.md`, `../WATCH-OUT-MISTAKES.md`.

## 1. Mission & definition of DONE

**A live public demo URL + an honest portfolio-grade repo.**

DONE means, all proven per Sid's proof standards (screenshots/recordings/pasted output, not claims):

1. The app runs on a public HTTPS URL, survives a restart, and a stranger can run a travel
   query end-to-end (query → stream → flight options → handoff URL).
2. Grafana shows real metrics from the running instance (monitoring actually collects — F-006 fixed).
3. Every known event-loop blocker is gone; CI runs the slow suite + Postgres (fake-green killed).
4. README claims match the audit's reality; the three TODO demo assets are REAL captures.
5. Uptime monitor + alert on the money path (/ask). Backups per `docs/persistence-backups.md` actually tested once.

DONE is NOT: Phase 1/2 refactors (planner decomposition, router extraction, queue replacement,
async engine), payments, multi-worker topology, browser-use as a product feature, new features
of any kind. Post-ship track is §6.

## 2. DON'T-WANT (binding, per the doctrine)

| # | Rule |
|---|---|
| D1 | No Phase 1/2 refactor work before the URL is live. Ship the C− app made safe, not a B+ app never shipped. |
| D2 | README/portfolio claims never exceed audit reality. "Production-grade" language goes unless STATUS says so. (MISTAKES.md 1.12) |
| D3 | No placeholder/broken demo assets. A README image that 404s = wireframe presented as done. Real captures or no image. |
| D4 | No secrets in screenshots, commits, or terraform state. `.env` lines listed in PORTFOLIO-SUMMARY §8 are radioactive. |
| D5 | No new dependencies beyond those a task names. (portfolio N18) |
| D6 | Green fast-suite ≠ done. Acceptance evidence is the deployed URL behaving + pasted output. (WATCH-OUT A5, N19/N20) |
| D7 | No `--workers >1` ever on this deploy (F-027). Single node is the honest contract — documented, not drifted past. |
| D8 | Free tier ≠ free of failure modes: anything hosting live data gets backups + the WATCH-OUT A1 checklist before traffic. |

## 3. Decision gates (⛔ = Sid decides, work stops)

| Gate | Decision | Options + recommendation |
|---|---|---|
| ⛔ G1 | Phase-0 scope cut | Recommend: ALL of 0.1–0.12 (it's ~1 sprint and each item is cheap). Minimum ship-blocking subset if time-boxed: 0.2 (loop blockers), 0.3 (Prometheus), 0.4 (deps truth), 0.5 (queue safety), 0.6 (auth), 0.7 (CI), 0.10 (DB fail-fast), 0.12 (approval binding). 0.1/0.8/0.9/0.11 are same-week cheap. |
| ⛔ G2 | Host | Recommend: **Oracle Cloud Always Free ARM VM** (4 OCPU/24GB class — fits Docker Compose + Postgres + monitoring, ₹0; verify current terms). Fallback: Hetzner/DO ~$5 VPS (EXPAND-STACK already planned "a cheap VPS as self-hosting practice"). NOT Vercel/Render free (SSE + long-lived processes + compose fit badly). This gate = the WATCH-OUT A6 practice ground. |
| ⛔ G3 | Ollama in prod? | Local-LLM-first is the demo's selling point but needs the VPS to run a model. Options: (a) small quantized model on the ARM box, (b) cloud-first routing on FREE tiers — **NVIDIA NIM + Groq + Gemini** as router backends (see ENTERPRISE-CHECKLIST §3: prod inference ≈ ₹0, upgrades the resume claim to a 6-provider router) — with Ollama shown in the local demo recording. Recommend (b); revisit post-ship. |
| ⛔ G4 | Demo assets approval | Screenshots/GIF captured from the LIVE URL, Sid approves each before README embed. (Proof standard; D3.) |
| ⛔ G5 | Launch | Smoke checklist green on the public URL → Sid says ship. |

## 4. Milestones

### M0 — Repo truth (half day)
- Triage + commit the dangling changes (README, App.tsx, css; add PORTFOLIO-SUMMARY.md).
- Recreate venv at the new path (see STATUS ⚠ — do after 0.4 lands so the manifest is truthful).
- Roadmap 0.8 artifact deletion (README.md.backup, COMMIT, local.db, startup_debug.log, dead modules).
- README honesty pass: strip "production-grade", state what STATUS says; remove broken image embeds until G4.
- **Proof:** `git log` + `git status` clean, pasted.

### M1 — Phase 0 hardening (per G1; ~1 sprint)
Execute `research/10-improvement-roadmap.md` Phase 0 in its stated order (0.1 → 0.12).
The roadmap file is the task spec — this plan does not restate it.
Checklist adds (ENTERPRISE-CHECKLIST §4): pydantic request models on ALL public POST
endpoints; rate-limit fail-mode audit (fail-CLOSED on sensitive paths — MISTAKES 1.3);
OWASP ZAP baseline job in CI. Research intake: implement the 0.11 async-DB guard as a
Semgrep rule (RESEARCH-INTAKE §4.2).
- **Proof:** CI run URL with tests_slow + Postgres job green; grep output showing no
  `SessionLocal(` in `async def` without to_thread (0.11 guard); before/after latency of /ask
  under `pg_sleep` induced DB pressure.

### M2 — Deploy (2–3 days, after G2/G3)
- VPS provisioned (G2), Docker + Compose, deploy per `docs/deployment-topology.md` +
  `docs/reverse-proxy-caddy.md` (Caddy = HTTPS), DNS on Cloudflare free.
- Secrets via docker secrets per `docs/environment-secrets-contract.md`; `.env` never leaves the box.
- Monitoring profile up; **Grafana screenshot with real scrape data = the F-006 fix proven.**
- Backups: nightly `pg_dump` cron + one RESTORE actually tested (WATCH-OUT A1/D8).
- UptimeRobot (free) on `/health` + /ask synthetic; alert to Sid's phone. (WATCH-OUT A2)
- Checklist adds (ENTERPRISE-CHECKLIST §4): Cloudflare **proxied** mode = DDoS + firewall
  rules + CDN ticks; k6 smoke run + documented capacity number; ZAP full scan + nuclei vs
  staging URL; Telegram ops-alert transplant from `smart-job-scanner-v2`; GitGuardian repo
  scan; Mozilla Observatory + Qualys once live.
- **SSE resilience trio (RESEARCH-INTAKE §4.1 — deploy prerequisite):** keep-alive comment
  pings (15–30s) + `X-Accel-Buffering: no` + `Cache-Control: no-cache` on BOTH stream
  endpoints (/ask and /jobs/{id}/events — neither has them today); proxy-timeout smoke test
  through Caddy + Cloudflare (verify SSE survives the proxied free plan — G2 gotcha).
- **Proof:** public URL answering a real query (recording); Grafana screenshot; restore-test output; uptime monitor screenshot; k6 + ZAP outputs.

### M3 — Demo assets + portfolio (1–2 days, gates G4→G5)
- Real captures from the live URL: streaming GIF, frontend screenshot, architecture diagram
  (render the PORTFOLIO-SUMMARY mermaid). Redaction check against PORTFOLIO-SUMMARY §8 first.
- README final: assets embedded, claims = reality, quickstart verified on a clean clone.
- Lighthouse CI on the frontend (tooling already owned from portfolio-website).
- **Resume truth pass:** update resume_1 + resume_2 bullets per ENTERPRISE-CHECKLIST §0 —
  overclaims become verified claims, "LangGraph"/"fcntl" bullets reworded to what's true.
- PORTFOLIO-SUMMARY updated with the live URL; portfolio-website case page can now link a
  living demo instead of claims.
- **Proof:** Sid-approved captures (G4); clean-clone quickstart output pasted.
- ⛔ G5 → ship.

## 5. What shipping this covers from EXPAND-STACK (the point of doing it this way)

| EXPAND-STACK priority | Covered by |
|---|---|
| 1. Docker | Already real here (Dockerfile/Compose) — M2 makes it production practice, not local-only. |
| 2. SQL depth | Phase-0 DB work + backup/restore discipline; Python-side analog (Alembic) is post-ship WS-D. |
| 3. AWS core | §6 Terraform+floci track — practiced against THIS app. |
| 4. Queue+Redis | Post-ship WS-B.3 (arq + Upstash/self-host Redis) — the audit already chose the candidates. |
| 7. Observability | M2 turns on the existing Prometheus/Grafana investment + uptime alerts — priority 7 done for real. |
| (5. Stripe / 6. Auth depth) | Not this project (no payments by design; auth is token-based). Stays on the JS-project track. |

## 6. Post-ship track (parked until G5 — listed so it doesn't creep in early; order per ENTERPRISE-CHECKLIST §4)

1. **Alembic baseline (WS-D):** migration framework + model consolidation — checklist #9
   (futureproof data modeling) and the first post-ship item.
2. **Terraform + floci (AWS Track B, ₹0):** `deploy/terraform/` replicating the topology —
   IAM, S3 backups, SQS, ECS service, Route53, CloudFront, WAF v2 ACL, Secrets Manager + KMS,
   CloudWatch alarms — applied clean against floci, optionally validated once on AWS Free Tier.
3. **Queue replacement (WS-B.3):** arq + Redis (Upstash free or Redis container on the VPS) —
   kills the custom job queue, covers priority 4.
4. **Webhook feature (checklist #22):** outbound price-drop webhook — HMAC-signed,
   timingSafeEqual verification, retry/backoff, replay protection; reuses the price tracker +
   sindhey webhook lessons. Telegram channel from M2 doubles as a consumer.
5. **Edge Worker (checklist #14):** small real Cloudflare Worker (edge status/geo enrichment).
6. **PostHog** on the frontend (checklist #2).
7. **k6 full profile** of /ask + SSE; document the breaking point (WATCH-OUT A3 evidence).
8. **browser-use:** nightly smoke agent driving the LIVE demo (query → stream → handoff link
   present) — automation/testing role ONLY (D-rule: not a product feature). Portfolio piece on
   AI-driven e2e.
9. **SSE protocol migration (RESEARCH-INTAKE §1):** sse-starlette + typed `event:` frames +
   `id:` on every event + `Last-Event-ID` resume with a server-side event buffer — job-events
   channel FIRST (native EventSource), /ask second behind golden-master fixtures (retires the
   `[DONE_JSON]` string-sentinel protocol, F-031); /ask fetch-client gets manual resume +
   backoff; reconnect validation tests (no dup/no loss by event-id; proxy-timeout).
10. **Prompt-injection input screening** per OWASP LLM guidance (HITL gate stays the backstop).
11. **Eval write-up:** RAGAS harness results (faithfulness/relevance), HITL gate design, and
    ADRs: router-vs-litellm, DBOS-vs-custom-planner, Procrastinate-vs-arq (tied to the Redis
    decision) — the resume-1 research artifacts.
12. Phase 1 refactors per roadmap (WS-A router extraction first), then the **Phase-2 topology
    decision: externalize state → workers>1 → honest stateless deployments (checklist #25, the
    capstone tick)** — only if the project earns continued investment.

## 7. Executor split (per standing division of labor)

Claude: this plan, verification at gates, check-verify on executor reports.
DeepSeek/GLM via `executor-plan` skill: M1 Phase-0 items (mechanical, well-specified by the
roadmap + findings.json) and M2 provisioning scripts. Sid: gate decisions, DNS/account
creation, captures approval, launch.
