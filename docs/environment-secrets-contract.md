# Environment and Secrets Contract (Phase 7D.2)

Status: active  
Scope: backend runtime environment and secret ownership for canonical production topology

## Canonical Configuration Source

- Local development: root `.env` (from `.env.example`).
- Production: environment injection by deployment platform/service manager (for example `systemd` `EnvironmentFile` or platform secret/env injection).
- Do not commit real secrets to repo files.

Related references:
- `docs/deployment-topology.md` (topology contract)
- `CONFIG.md` (broader runtime variable catalog, including legacy compatibility notes)
- `docs/security-s1-s2-hardening.md` (canonical S1/S2 secrets + transport/API-surface policy)

## Required and Optional Variables

### Required for canonical production deployment

| Variable | Required | Secret | Purpose | Default if omitted | Startup behavior if omitted |
|---|---|---|---|---|---|
| `DATABASE_URL` | Yes (contract) | No | Persistent DB location | Falls back to `sqlite:///./local.db` | Process still starts; fallback path is non-canonical for production persistence |
| `ADMIN_TOKEN` | Yes (contract) | Yes | Auth gate for admin/debug endpoints | No default token | Process still starts; admin/debug routes reject access (`403`) |
| `ALLOWED_ORIGINS` | Yes (contract) | No | CORS allowlist for frontend/proxy origins | localhost dev origins | Process still starts; default dev origins may be wrong for production clients |
| `SERPAPI_KEY_n` (`SERPAPI_KEY_1`+) | Yes for live flight/booking | Yes | SerpApi flight + booking resolver access | none | Process still starts; live flight/booking requests degrade/fail due to no usable SerpApi key |

### Required only for specific runtime modes/features

| Variable | Required when | Secret | Purpose | Default if omitted | Startup behavior if omitted |
|---|---|---|---|---|---|
| `GEMINI_KEY_n` | `LLM_MODE` uses cloud + Gemini configured | Yes | Gemini cloud LLM access | none | Process still starts; Gemini provider unusable |
| `OPENAI_KEY_n` | `LLM_MODE` uses cloud + OpenAI configured | Yes | OpenAI cloud LLM access | none | Process still starts; OpenAI provider unusable |
| `WEATHER_KEY_n` | Live weather enrichment expected | Yes | OpenWeather access | none | Process still starts; weather path degrades/fails upstream |

### Optional baseline variables

| Variable | Required | Secret | Purpose | Default if omitted | Startup behavior if omitted |
|---|---|---|---|---|---|
| `LLM_MODE` | Optional | No | Routing mode | `ollama_first` | Process starts with default mode |
| `USE_CLOUD_LLM` | Optional | No | Cloud enable/disable admin gate | enabled | Process starts; cloud may still be unusable without keys |
| `CLOUD_PROVIDER_CHAIN` / `CLOUD_PROVIDER` | Optional | No | Cloud provider ordering/default | defaults to `gemini` | Process starts with default provider chain |
| `OLLAMA_BASE_URL` | Optional | No | Local Ollama endpoint | `http://localhost:11434` | Process starts; may degrade if Ollama unreachable |
| `LOG_LEVEL` | Optional | No | Log verbosity | `INFO` | Process starts with default log level |
| `KEY_ENV_MONITOR_TICK` | Optional | No | Key refresh cadence (seconds) | `60` | Process starts with default cadence |

### Non-canonical runtime toggles (diagnostic/manual only)

These are intentionally omitted from `.env.example` and canonical deployment setup:

| Variable | Category | Why non-canonical |
|---|---|---|
| `ASYNC_JOB_REQUIRE_SINGLE_WORKER` | Runtime safety override | Default `true` already enforces contract; set only for targeted diagnostics |
| `ALLOW_UNSAFE_ASYNC_JOBS` | Runtime safety override | Unsafe bypass for lab-only diagnostics; never required for production setup |
| `KEY_MANAGER_LOCK_BACKEND` | Runtime internals tuning | Canonical topology is single-node file-lock path; backend switching is non-canonical |
| `KEY_MANAGER_LOCK_PATH` | Runtime internals tuning | Optional local lock-path override only |
| `KEY_MANAGER_LOCK_TTL_SECONDS` | Runtime internals tuning | Redis-lock TTL tuning; non-canonical for single-node file-lock default |
| `KEY_MANAGER_REDIS_URL` | Unsupported topology-related tuning | Relevant only for redis lock backend, which is outside canonical topology |
| `KEY_MANAGER_LOCK_NAME` | Unsupported topology-related tuning | Relevant only for redis lock backend, which is outside canonical topology |

`ALLOWED_ORIGINS` hardening note:
- Must be explicit `scheme://host[:port]` origins.
- Wildcard `*` is intentionally not accepted by runtime CORS parsing.
- Invalid entries are ignored.

## Startup Failure Behavior (Current Runtime)

The current runtime is intentionally startup-resilient:
- Missing provider keys do not hard-stop startup.
- Missing `DATABASE_URL` does not hard-stop startup (falls back to local sqlite path).
- Missing `ADMIN_TOKEN` does not hard-stop startup; admin/debug endpoints remain access-denied.
- External-provider readiness is reflected via health/status endpoints (`/health`, `/health/deep`, `/llm/options`) rather than startup crash.

Important implication:
- Production operators must treat the required-contract table above as deployment policy, because many misconfigurations degrade at runtime instead of terminating process startup.

## Secrets Contract

1. Secret sources
- Secrets must come from managed secret injection or protected environment files owned by operations.
- Do not store real secrets in committed `.env` files, docs, tests, or screenshots.

2. Secret vs non-secret boundary
- Secret: `*_KEY_n`, `ADMIN_TOKEN`, provider credentials/tokens.
- Non-secret: mode flags, timeouts, CORS origins, non-sensitive URLs/paths.

3. Rotation ownership (role-based)
- Rotation owner: deployment operator / platform owner.
- Application owner provides required variable list, validates app behavior after rotation, and coordinates rollout timing.

Detailed rotation and leak-response procedures:
- `docs/security-s1-s2-hardening.md` (`S1.4 Rotation Procedure`, `S1.5 Secret Leak Incident Procedure`)

## Deprecated / Non-Canonical Variables

- Legacy single-key vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are compatibility-only and not canonical.
- Legacy control vars (`LLM_PRIORITY`, `LLM_PREWARM`) are non-canonical for current startup path.
- Docker/validation-only env files (`.env.laptopdocker`, `.env.tmp`) are not canonical production env sources.
