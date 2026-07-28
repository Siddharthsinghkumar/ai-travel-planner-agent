# ENTERPRISE-CHECKLIST — 24 Capability Ticks × Plan Coverage × Resume Payoff

> Written 2026-07-18 at Sid's request: ship this project with as many enterprise capability
> ticks as possible, feeding **resume_1_ai_backend** and **resume_2_fullstack**
> (`/home/sidd/resume/`). Companion to `SHIP-PLAN.md`, `FREE-STACK-MAP.md`, `STATUS-2026-07-18.md`.

## 0. The resume-truth problem (read first — it's the whole point)

Both resumes ALREADY cite this project (`ai-travel-planner-agent`, linked publicly):

| Resume claim today | Reality today | What makes it TRUE |
|---|---|---|
| "Zero unauthorized tool executions **in production**" (R1) | Never deployed; approval guard can never deny (F-016) | Phase-0 0.12 + M2 live URL |
| "6 Prometheus alert rules" (R2) | Prometheus has never scraped one metric (F-006) | 0.3 fix + M2 Grafana screenshot |
| "Multi-tenant RAG retrieval pipeline" (R2) | RAG silently no-ops in prod (F-003) | 0.4 dependency surgery |
| "207 CI tests (8.4s)" (both) | True but fast-suite only; risky-infra tests never run in CI (F-010) | 0.7 CI gates |
| "LangGraph orchestration" (R2) | Hand-rolled planner state machine, no LangGraph | Fix the RESUME (own doctrine: never upgrade a status label) — "custom agent state machine" is the STRONGER claim anyway |
| "fcntl multi-process locking" (R2) | Listed by the audit as a defect (F-018) | Replace bullet after WS-B.5, or reword to what's true |

**Shipping = converting overclaims into verified claims.** Same rule as the portfolio COPY.md
verified register. Resume edits land at M3 alongside the README honesty pass.

## 1. The 24 ticks — status, coverage, gap action

Legend: ✅ solid today · 🟡 exists-but-broken/partial · ❌ absent.
"Tick at" = when it's honestly claimable. floci column = simulable in the AWS-replica track (§2).

| # | Capability | Today | Tick at | Gap action (free tool) | floci? |
|---|---|---|---|---|---|
| 1 | Authentication | 🟡 token auth, timing-safe; AUTH_DISABLE hole, unbound approvals | M1 | Phase-0 0.6 + 0.12. Enterprise OIDC (Keycloak/Zitadel OSS) = optional post-ship | IAM/STS |
| 2 | Analytics | ❌ | post-ship | PostHog free tier (or self-host Umami) on frontend | — |
| 3 | DNS | ❌ | M2 | Cloudflare free (unlimited domains) | Route53 |
| 4 | Stress testing | ❌ | M2 basic | k6 OSS smoke at M2; full profile + report post-ship (also Loadmill 50-user free) | — |
| 5 | Pen testing | 🟡 static audit only (research/07) | M2 | **ADD:** OWASP ZAP baseline in CI (M1), ZAP full + nuclei vs staging (M2), Mozilla Observatory + Qualys after live; GitGuardian free repo secret-scan; trivy per existing `docs/dependency-image-scanning.md` | — |
| 6 | Load handling | 🟡 admission control exists; loop-blockers negate it | M1–M2 | 0.2 IS the fix; k6 documents real capacity ("sustained N rps" = resume-grade number) | — |
| 7 | Fail tolerance | ✅ breakers/retries/fallback/key-rotation (minus F-021 leak, queue crash-loop) | M1 | 0.5 + WS-B.4 leak fix. Marquee story already | — |
| 8 | Backups | 🟡 doc exists, nothing runs | M2 | Nightly pg_dump + ONE TESTED RESTORE; offsite copy to Cloudflare R2 free 10GB | S3 |
| 9 | Futureproof data modeling | ❌ models scattered, no migrations | post-ship #1 | Alembic baseline (WS-D) — first post-ship item since it's on this list | RDS |
| 10 | DDoS defense | ❌ | M2 | Cloudflare proxied mode (free DDoS + firewall rules) in front of the VPS | WAF v2 |
| 11 | Input sanitation | 🟡 pydantic on ~2/33 endpoints; SSRF designed-out | M1 partial | **ADD to M1:** request models on ALL public POST endpoints; full WS-E + schemathesis post-ship | — |
| 12 | Rate limiting | 🟡 sliding-window limiter exists, process-local; fail-mode unaudited | M1 | **ADD to M1:** verify fail-CLOSED on sensitive paths (MISTAKES 1.3 lesson) | API GW |
| 13 | Caching | 🟡 5 mechanisms, stampede bug F-013 | M2 partial | Cloudflare edge-cache static at M2; cashews consolidation WS-B.2 post-ship | ElastiCache |
| 14 | Edge computing | ❌ | post-ship | Cloudflare Workers free (100k req/day): edge status endpoint / geo header enrichment — small but real | CloudFront |
| 15 | Web performance | 🟡 unmeasured | M3 | **ADD to M3:** Lighthouse CI on frontend (tooling already owned from portfolio) + TTFT metric | — |
| 16 | CDN work | ❌ | M2 | Cloudflare CDN for static/frontend; R2 for demo assets | CloudFront |
| 17 | Monitoring | 🟡 full stack bundled, dark since inception | M2 | 0.3 + M2: Grafana with real data + UptimeRobot + the 6 alert rules actually able to fire | CloudWatch |
| 18 | Network security | 🟡 headers/CORS strong; no TLS/firewall (never deployed) | M2 | Caddy TLS + ufw + SSH keys-only + Cloudflare origin shielding | — |
| 19 | API integrations | ✅ SerpAPI/OpenWeather/multi-LLM, quota-aware | now | Strengthen with NIM/Groq providers (§3) | SES/SNS |
| 20 | Idempotency | 🟡 Idempotency-Key on booking-tracking (exemplary); in-memory records F-011 | M1 partial | 0.5; Postgres-backed records WS-C post-ship | — |
| 21 | Automated testing | 🟡 207 fast + ungated slow suite + RAGAS evals | M1 | 0.7 CI gates; browser-use smoke + schemathesis post-ship | Testcontainers |
| 22 | Webhooks | ❌ nothing inbound or outbound | post-ship | **ADD feature:** outbound price-drop webhook — HMAC-signed, timingSafeEqual verify, retry/backoff, replay protection. Reuses sindhey webhook lessons + existing price tracker. Small, high resume value | SNS/API GW |
| 23 | Secret management | ✅ docker secrets, env contract, redaction, key fingerprints | M2 | Optional polish: Infisical/Doppler free tier, GitGuardian scan | Secrets Mgr + KMS |
| 24 | Audits | ✅ exceptional static corpus + pip-audit CI | M2 stronger | Add the dynamic layer (#5 tools) | — |
| 25 | Stateless deployments | ❌ — explicitly process-local by contract (F-027) | Phase 2 | THE big one: externalize state (Redis/Postgres) → workers>1 → disposable containers. Pairs exactly with ECS-on-floci Terraform. Strongest single enterprise resume line available here | ECS |

**Score:** at G5 ship: ~17/25 honestly claimable (6 already-solid + 11 landed by M1–M3).
Post-ship track closes: data modeling, webhooks, edge, analytics, full caching, full idempotency,
schemathesis, and finally stateless topology. **0-to-end enterprise claim is real after Phase 2.**

## 2. The floci question, answered precisely

**Verified from the repo (2026-07-18):** floci emulates IAM (users/roles/policies/STS), S3,
SQS (std+FIFO+DLQ), SNS, Lambda (real Docker runtimes), ECS, RDS (real Postgres/MySQL
containers, IAM auth), DynamoDB, **Route53, CloudFront, WAF v2, Secrets Manager, ElastiCache
(real Redis/Valkey + SigV4), CloudWatch (metrics/alarms/logs), API Gateway, KMS, SES**.
Known limits: Textract/Transcribe/Bedrock are dummy stubs; needs Docker socket; "local dev
emphasis over production parity."

**So: every AWS-shaped item on the 24-list is simulable** — but hold the two tracks apart:

- **Track A — the LIVE deployment (real ticks):** Cloudflare free (DNS, DDoS, WAF rules, CDN,
  Workers/edge — all confirmed on free-for-dev) + VPS + Caddy + Grafana + UptimeRobot + k6 +
  ZAP. An emulator can never provide these ticks; only a public URL under real traffic can.
- **Track B — the AWS replica (skills ticks, ₹0):** Terraform in `deploy/terraform/` targeting
  floci: IAM policies, S3 backup bucket, SQS job queue, ECS service of this image, Route53
  zone, CloudFront distro, WAF v2 ACL, Secrets Manager + KMS, CloudWatch alarms. Applies
  clean locally → optionally validated once on AWS Free Tier (EC2 750h/mo ×12mo, S3 5GB
  confirmed; GCP/Azure/Oracle equivalents on free-for-dev as alternates).
  Resume line: *"Full AWS IaC replica of the production topology (Terraform), CI-validated
  against a local emulator."* — honest AND enterprise-legible.

## 3. Making it better — repos (Sid's + external), free LLM tiers, research

### Sid's own repos (from `~/project/`, per PROJECT_ECOSYSTEM_MAP.md)

| Repo | Transplant | Value |
|---|---|---|
| `smart-job-scanner-v2` | Battle-tested Telegram bot (`stage10_notification.py`) + Gemini multi-key client | FREE ops alert channel (uptime/error/price-drop alerts to Telegram) — plugs straight into M2 monitoring and the §1.22 webhook feature |
| `merlin-cli` / `merlin-cli-bridge` | Free LLM inference bridge | DEV-TIME ONLY (browser-bridge inference is ToS-gray for prod). Also: it's already a resume-1 project — cross-linking the two demos strengthens both |
| `persona-context-engine` | FAISS embedding patterns | Compare/merge with existing RAG retriever when RAG is un-broken (0.4) |
| `bluesentinel`, Crawl4AI plans | — | NOT this project. Scraping airline sites = ToS risk + scope creep (D5) |

### Free LLM providers → the router (also answers ⛔ G3)

The router (Ollama → Gemini → OpenAI) is the resume centerpiece. Add free-tier providers as
first-class backends — prod inference cost ≈ ₹0 AND upgrades the claim to a **6-provider router**:

| Provider | Free offer (verify current limits) |
|---|---|
| **NVIDIA NIM** (build.nvidia.com) | Free API credits on hosted NIM endpoints (Llama, Mistral etc.) — the "NIM" Sid asked about; job-discovery-engine already has NIM skills to reuse |
| **Groq** | Free tier, extremely fast Llama/Mixtral — ideal "fast lane" backend |
| **Google Gemini** | Free tier — ALREADY integrated, keep |
| **OpenRouter** | Rotating `:free` models — good chaos-testing backend |
| **GitHub Models** | Free with GitHub account — CI-friendly |
| **Cerebras / Mistral La Plateforme** | Free tiers — optional breadth |

G3 recommendation upgrade: prod routing = NIM/Groq/Gemini free tiers (cloud-first), Ollama
featured in the local demo recording. Circuit breakers + key rotation ALREADY handle
multi-provider failure — this is a config + adapter task, not architecture.

### External repos / research already in or adjacent to the plan

ZAP, nuclei, trivy, GitGuardian (pen/audit — §1); k6, schemathesis, arq, cashews, Alembic,
Testcontainers (roadmap); browser-use (smoke agent). **Deliberately NOT adopted: litellm** —
it would replace the hand-built router that IS the portfolio value; instead write a short ADR
comparing the custom router to litellm (great interview artifact).
**Research angle with real payoff:** the RAGAS eval harness already in-repo (`eval_results/`,
`--with-rag` baseline) → publish an eval write-up (methodology, faithfulness/relevance numbers,
HITL gate design) as repo doc + blog post. For resume 1 (AI backend), one honest eval report
outranks three new features.

## 4. Plan deltas (applied to SHIP-PLAN — summary)

- **M1 adds:** request models on all public POST endpoints; rate-limit fail-mode audit; ZAP baseline job in CI.
- **M2 adds:** Cloudflare proxied (DDoS/WAF/CDN) explicit; k6 smoke + capacity number; ZAP full + nuclei vs staging; Telegram ops alerts (smart-job-scanner transplant); GitGuardian + Observatory/Qualys post-live.
- **M3 adds:** Lighthouse CI on frontend; **resume 1 + 2 truth pass** (§0 table) shipped with the README pass.
- **G3 updated:** cloud-first = NIM/Groq/Gemini free tiers.
- **Post-ship reorder:** Alembic first (checklist #9), then Terraform+floci Track B, arq+Redis, webhook feature, edge Worker, PostHog, k6 full, browser-use smoke, schemathesis; **Phase-2 stateless topology as the capstone tick (#25).**
