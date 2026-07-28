# FREE-STACK-MAP — Covering EXPAND-STACK for ₹0 (or near it)

> Written 2026-07-18. Maps the heavy-lifting repos onto every gap in `../EXPAND-STACK.md`:
> - **free-for-dev** (github.com/ripienaar/free-for-dev) — the master list of free tiers; treat
>   it as the CATALOG. Named tiers below are from knowledge that may be stale — **verify current
>   limits against free-for-dev before committing to any service.**
> - **floci** (github.com/floci-io/floci) — local AWS emulator, MIT, 69 services (S3, SQS,
>   Lambda, RDS, ECS…), works with AWS SDKs/CLI/**Terraform**. = AWS practice with zero bill risk.
> - **browser-use** (github.com/browser-use/browser-use) — AI browser automation (LLM drives a
>   real browser). Role here: agentic e2e/smoke testing + a portfolio piece. Python — same
>   ecosystem as llm-travel-agent.
> - **Terraform** (github.com/hashicorp/terraform) — OSS, free. IaC for the AWS track;
>   pairs with floci for free practice, HCP Terraform free tier for state when needed.

## 1. The 7 priorities — free coverage + practice vehicle

| # | Priority | Free coverage | Practice vehicle |
|---|---|---|---|
| 1 | **Docker** | Docker Engine/Compose OSS. Free registries: GitHub Container Registry, Docker Hub free tier. | Already real in llm-travel-agent; SHIP-PLAN M2 = production practice. |
| 2 | **SQL + migration-first ORM** | Postgres OSS; free hosted PG: Neon / Supabase / Aiven free tiers (check free-for-dev "DBaaS"). Drizzle/Prisma OSS; Alembic OSS (Python side). | Travel agent: Alembic baseline (roadmap WS-D) + backup/restore drills. JS side: next client project on Drizzle. |
| 3 | **AWS core** | **floci locally = ₹0 AWS**; then AWS Free Tier (12-month + always-free lambda/SQS quotas) for the real thing. IAM concepts are identical against floci. | Terraform + floci: S3/SQS/ECS for THIS app (SHIP-PLAN §6.1) before touching a real account. |
| 4 | **Queue + Redis** | Redis/Valkey OSS self-host on the VPS; Upstash free tier for managed. arq/BullMQ/procrastinate all OSS. | Travel agent WS-B.3: custom queue → arq (audit already scoped it). |
| 5 | **Stripe** | No fee until real transactions; test mode is a full free sandbox. | Not this project (no payments). Next JS client project, or a toy checkout on the portfolio. |
| 6 | **Auth without Clerk** | Auth.js OSS; Keycloak/Zitadel OSS if a client needs SSO (free-for-dev "Authentication" section). | Travel agent's token auth hardening (Phase-0 0.6/0.12) is real auth-depth work already. |
| 7 | **Observability** | pino/structlog OSS; **Grafana+Prometheus already IN this repo** (fix = F-006); Grafana Cloud free tier; UptimeRobot free monitors; Better Stack free tier; Sentry free tier (already used on JS projects). | SHIP-PLAN M2 turns the existing stack on + uptime alerts = priority 7 done for real. |

## 2. Frameworks list — completion path

| Framework | Free resources | Concrete completion step |
|---|---|---|
| **NestJS** (first) | OSS + official docs/courses free. | Build the next JS-project backend (or a rewrite of one project-2 API) in Nest with Drizzle — one real deliverable, not a tutorial. |
| **Spring Boot** (enterprise track, 6–12 mo) | Adoptium JDK (free OpenJDK), Spring Initializr, spring.io guides, Spring Academy free courses, Maven/Gradle OSS. Free deploy targets for practice: the same VPS, or Oracle Always Free (JVM fits the 24GB ARM box easily). Testcontainers + **floci has Java-first Testcontainers support** — Spring Boot + S3/SQS integration tests run free and local. | Milestone ladder: (1) Java syntax→idioms, (2) Spring Boot REST + JPA/Hibernate + Postgres, (3) Spring Security, (4) one portfolio artifact: re-implement llm-travel-agent's booking-tracking API (the exemplary module) in Spring Boot with Testcontainers — direct comparison piece, enterprise-legible. Start only when deliberately targeting enterprise clients (EXPAND-STACK verdict stands). |
| **Fastify** | OSS. | Absorbed while doing NestJS (Nest can run on Fastify adapter) — a weekend, not a project. |
| **Astro** | OSS; free deploy on Cloudflare Pages/Netlify/GitHub Pages. | Next content/marketing brief ships on Astro instead of Next. |
| **React Native / Expo** | Expo free tier (EAS free builds are limited — check free-for-dev). | Demand-driven only. No pre-investment. |
| Django / Rails / Laravel | — | Explicit skips (unchanged). |

## 3. Dependencies list — completion path

| Dependency | Free? | Concrete completion step |
|---|---|---|
| Drizzle (or Prisma) | OSS | Next JS project day-zero; add to `reusable-components/` starter once proven. |
| BullMQ / arq | OSS | arq in travel agent (WS-B.3) covers the concept; BullMQ on next JS project. |
| pino / structlog | OSS | pino → JS starter template; travel agent already has structured redaction logging to study. |
| TanStack Query | OSS | Next JS client project; kills hand-rolled fetch (MISTAKES 1.23). |
| react-hook-form | OSS | Same project, first form. |
| shadcn/ui | OSS | Commit to it on the next UI build (learned.md §5 already ordered this). |
| ioredis | OSS | Arrives with the Redis/queue work. |
| **k6** | OSS local; k6 Cloud has a small free tier | Load-test the LIVE travel-agent URL post-ship (SHIP-PLAN §6.4); document the breaking point. |
| Renovate / npm-audit | Renovate free for OSS/GitHub; npm audit + **pip-audit already gating this repo's CI** | Turn Renovate on for portfolio-website + this repo this week — it's a checkbox. |
| OpenTelemetry | OSS; free backends: Grafana Cloud free / Jaeger self-host | Post-ship: OTel traces on /ask → SerpAPI/LLM spans; Grafana already deployed by then. |

## 4. Services list — completion path

| Service | Free route (verify on free-for-dev) | Note |
|---|---|---|
| **AWS** | floci (all learning) → AWS Free Tier (validation) | IAM/S3/SQS/ECS against floci first; bill risk ₹0 until deliberate. |
| **Terraform** | OSS CLI; HCP Terraform free tier for remote state (small resource count) | `deploy/terraform/` in this repo is the first real artifact (SHIP-PLAN §6.1). State stays local or HCP free; NEVER commit state (contains secrets — D4). |
| **Stripe** | Test mode free | Next payments project. |
| **Cloudflare** | Free plan: DNS, CDN, TLS; Workers/Pages/R2 free tiers | Travel-agent DNS at M2 = first hands-on. R2 later for demo assets (S3-compatible = same SDK skills as floci practice). |
| **Better Stack / Checkly / UptimeRobot** | All have free tiers; UptimeRobot's is the roomiest historically | M2 uptime + alerting. Kills WATCH-OUT A2. |
| **Grafana** | Already self-hosted in this repo's compose; Grafana Cloud free as managed alternative | M2 proves it with real data. |
| **Cheap VPS** | **Oracle Cloud Always Free ARM** (the famous ₹0 "real server"; verify terms + capacity availability) → else Hetzner/DO ~$5/mo | G2 decision. This IS the self-hosting practice ground from EXPAND-STACK. |
| **PostHog** | Free tier (generous event count) | Add to portfolio-website + travel-agent frontend post-ship; client upsell skill. |
| **Supabase paid tier** | NOT free — deliberately | The one PAID checkbox: live client DBs (sindhey) need PITR today. Do not free-tier this one. (WATCH-OUT A1) |
| Neon / Aiven / Upstash / GHCR / Doppler-Infisical (secrets) | Free tiers per free-for-dev | Pick when the need arrives; catalog covers them. |

## 5. browser-use — where it actually fits (scope-guarded)

1. **Post-ship smoke agent (SHIP-PLAN §6.3):** nightly run against the live URL — submit a
   query, verify stream completes and a handoff link renders. Screenshots on failure. This is
   the freelancer-grade demo of "AI e2e testing" and it watches the demo for free.
2. **Portfolio piece:** short recorded run = content for the portfolio site's travel-agent case page.
3. **NOT (pre-ship):** a product feature inside the travel agent (e.g., agentic browsing of
   airline sites violates D1/D5 and most airline ToS — if ever pursued, that's a deliberate
   post-ship decision with its own gate).

## 6. Order of operations (merges with SHIP-PLAN)

```
Now: SHIP-PLAN M0–M3 (ship the travel agent; covers priorities 1, 7, half of 2)
├─ during M2: Cloudflare DNS + UptimeRobot + Oracle/Hetzner VPS   (services: 3 checked off)
├─ this week, parallel, checkbox-sized: Renovate on 2 repos; Supabase paid tier for sindhey
Post-ship: Terraform+floci track (priority 3) → arq+Redis (priority 4) → k6 + OTel + PostHog + browser-use smoke
Next JS client project: Drizzle, TanStack Query, react-hook-form, shadcn/ui, pino, NestJS backend (priorities 2, 6 + deps list)
Deliberate market decision, later: Spring Boot ladder (§2) — with floci-backed Testcontainers
```

**Standing rule (from WATCH-OUT B1):** every service adopted from this map gets its config
captured in the starter template / `reusable-components/`, so the next project starts with it.

## 7. Per-service matrix: floci practice → real for free (added 2026-07-18)

> Answers: "can I use each floci-emulated service for real, for free?" — YES for all 17, via
> the always-free AWS path or a free alternative. ⚠ AWS restructured its free tier mid-2025
> (new accounts: credits-based free plan ~$100–200 / ~6 months instead of the classic
> 12-month tier for many services) — **verify current terms at signup**; "always-free"
> quotas below are the historically stable ones.

| Service | floci (practice) | Real AWS free path | Free alternative (production Track A) |
|---|---|---|---|
| IAM | ✅ full | ✅ IAM itself is always free | — (concept transfers everywhere) |
| S3 | ✅ | 5GB (classic 12-mo / credits) | **Cloudflare R2 10GB free — S3-COMPATIBLE API, same SDK skills** · Backblaze B2 10GB |
| SQS | ✅ | ~1M requests/mo always-free (verify) | Redis+arq on the VPS · Upstash |
| SNS | ✅ | ~1M publishes free (SMS costs) | **ntfy.sh (free/self-host)** · own Telegram bot (smart-job-scanner transplant) |
| Lambda | ✅ real Docker | ~1M req + 400k GB-s/mo always-free (verify) | **Cloudflare Workers 100k req/day** (already planned for edge) |
| ECS | ✅ real Docker | Control plane free; compute is NOT | The VPS Docker Compose IS the free equivalent · Oracle Always-Free ARM |
| RDS | ✅ real Postgres | 750h micro (classic 12-mo / credits) | **Neon free tier (has branch/restore!)** · self-host PG on VPS (current plan) |
| DynamoDB | ✅ | 25GB always-free | Not needed here; Cloudflare KV / Upstash if KV wanted |
| Route53 | ✅ | ❌ NOT free ($0.50/zone/mo) | **Cloudflare DNS free, unlimited** (already the plan) |
| CloudFront | ✅ | ✅ ~1TB egress/mo always-free (verify) | Cloudflare CDN free (already the plan) |
| WAF v2 | ✅ | ❌ NOT free (~$5/ACL/mo) | **Cloudflare free firewall rules + managed DDoS** |
| Secrets Manager | ✅ | ❌ ($0.40/secret/mo) — BUT **SSM Parameter Store standard tier = free** | Docker secrets (current) · Doppler/Infisical free ≤5 users |
| KMS | ✅ | AWS-managed keys free; customer keys $1/mo | **sops + age (OSS)** for repo/config encryption |
| ElastiCache | ✅ real Redis | 750h micro (classic 12-mo / credits) | **Upstash Redis free** · Redis container on VPS (planned) |
| CloudWatch | ✅ | ~10 metrics / 10 alarms / 5GB logs always-free (verify) | **Self-hosted Prometheus+Grafana — already in this repo** · Grafana Cloud free |
| API Gateway | ✅ | ~1M calls/mo (classic 12-mo) | Caddy on VPS (current) · Workers routes |
| SES | ✅ | ~3k msgs/mo free tier (terms shifted — verify) | **Resend free 3k/mo (already used on sindhey)** · Brevo 300/day |

**Cost-guard rules (binding before ANY real AWS account):** (1) dedicated practice account,
never a client's; (2) billing alarm + AWS Budget set in the first 10 minutes, before the first
resource; (3) Terraform `destroy` verified clean after every practice session; (4) nothing
with a card-on-file autoscales (no NAT gateways, no unmonitored Fargate). The classic AWS
mistake is the surprise bill — it goes in WATCH-OUT the day an account exists.
