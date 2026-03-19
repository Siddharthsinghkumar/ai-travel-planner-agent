# Configuration Guide

## 0. Which File Should I Edit?
- Backend local runtime: edit root `.env` (start from `.env.example`).
- Frontend local runtime (Vite): edit `frontend/.env` (start from `frontend/.env.example`).
- Docker validation/runtime variant in this repo: `.env.laptopdocker` (only for that deployment path).
- One-off validator generated environment: `.env.tmp` (created by `full_validation.py`; do not treat as canonical).

## 1. Purpose
This project has multiple configuration surfaces:

- Runtime/deployment config: environment variables consumed by backend services at startup/runtime.
- Request-level overrides: `llm_mode` and `cloud_provider` fields on `/ask` payloads.
- Frontend config: Vite `VITE_*` variables used by the React UI.
- Test/validation-only config: flags used by `pytest` and `full_validation.py`.
- Docker/runtime overrides: container `ENV` and env-file injection can override root `.env`.

This file defines the canonical configuration contract for current runtime behavior and marks legacy/dead flags explicitly.

## 2. Canonical Runtime Variables

### Routing and cloud selection

`LLM_MODE`
- Purpose: canonical backend routing mode.
- Allowed: `ollama_only`, `cloud_only`, `cloud_first`, `ollama_first`.
- Default: `hybrid` alias is normalized in `core/llm_mode.py` to canonical mode based on `LLM_PRIORITY`.
- Used in: `core/llm_mode.py`, applied per request in `api/app.py`.
- Required: recommended.

`CLOUD_PROVIDER_CHAIN`
- Purpose: ordered cloud provider preference chain.
- Format: comma-separated provider ids (example: `gemini,openai`).
- Default: falls back to `CLOUD_PROVIDER`.
- Used in: `core/llm_mode.py`, `agents/cloud_llm.py`.
- Required: recommended for deterministic provider order.

`CLOUD_PROVIDER`
- Purpose: default cloud provider when no request override is provided.
- Allowed: configured provider ids (for example `gemini`, `openai`).
- Default: `gemini`.
- Used in: `core/llm_mode.py` and `agents/cloud_llm.py`.
- Required: optional when `CLOUD_PROVIDER_CHAIN` is set.

`USE_CLOUD_LLM`
- Purpose: administrative cloud enable/disable gate.
- Allowed: truthy (`1`, `true`, `yes`, `on`) or falsy (`0`, `false`, `no`, `off`).
- Default: enabled when unset.
- Used in: `agents/cloud_llm.py`, `core/health.py`, validation env generation in `full_validation.py`.
- Semantics:
  - `0`: cloud is intentionally disabled, regardless of key availability.
  - `1`: cloud is allowed, but actual cloud readiness still requires usable provider keys (`get_usable_providers()`).

### Ollama runtime

`OLLAMA_BASE_URL`
- Purpose: Ollama endpoint URL.
- Default: `http://localhost:11434` in `agents/ollama_client.py`.
- Used in: `agents/ollama_client.py`.
- Required: recommended for non-default network topology.

`OLLAMA_MODEL`
- Purpose: default local model name.
- Default: `openhermes`.
- Used in: `agents/ollama_client.py`.
- Required: optional.

`OLLAMA_TIMEOUT`
- Purpose: Ollama request timeout (seconds).
- Default: `30.0`.
- Used in: `agents/ollama_client.py`, indirectly by router local timeout defaults.
- Required: optional.

### API and infra

`DATABASE_URL`
- Purpose: database connection string.
- Default: `sqlite:///./local.db` fallback if unset (non-testing path).
- Used in: `agents/database.py`.
- Required: recommended in non-local deployments.

`ADMIN_TOKEN`
- Purpose: admin auth for protected debug endpoints.
- Used in: `api/app.py`, `core/health.py`.
- Required: recommended when admin endpoints are exposed.

`LOG_LEVEL`
- Purpose: runtime logging level.
- Allowed: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- Default: `INFO`.
- Used in: `core/logging_config.py`.
- Required: optional.

### Key manager / refresh / locking

`KEY_ENV_MONITOR_TICK`
- Purpose: key-refresh loop interval (seconds) started by app lifespan.
- Default: `60`.
- Used in: `api/app.py`.
- Required: optional.

`KEY_MANAGER_LOCK_BACKEND`
- Purpose: refresh lock backend selection.
- Allowed: `file`, `redis`.
- Default: `file`.
- Used in: `api/app.py`.
- Required: optional.

`KEY_MANAGER_REDIS_URL`
- Purpose: Redis connection for distributed lock when backend is `redis`.
- Default: `redis://localhost:6379/0`.
- Used in: `api/app.py`.
- Required: optional (redis backend only).

`KEY_MANAGER_LOCK_NAME`
- Purpose: Redis lock key name.
- Default: `llm:key_refresh_lock`.
- Used in: `api/app.py`.
- Required: optional.

`KEY_MANAGER_LOCK_TTL_SECONDS`
- Purpose: Redis lock TTL.
- Default: `60`.
- Used in: `api/app.py`.
- Required: optional.

`KEY_MANAGER_LOCK_PATH`
- Purpose: file lock path when lock backend is `file`.
- Default: `/tmp/llm_key_refresh.lock`.
- Used in: `api/app.py`.
- Required: optional.

### Numbered key families (canonical modern key surface)

`OPENAI_KEY_n`
- Purpose: OpenAI key pool entries.
- Example: `OPENAI_KEY_1=...`, `OPENAI_KEY_2=...`.
- Used in: `core/api_key_manager.py` pattern parsing.
- Required: optional (only if OpenAI provider is in use).

`GEMINI_KEY_n`
- Purpose: Gemini key pool entries.
- Used in: `core/api_key_manager.py`.
- Required: optional (only if Gemini provider is in use).

`SERPAPI_KEY_n`
- Purpose: SerpAPI key pool for flight tool.
- Used in: `core/api_key_manager.py`.
- Required: required for live flight search.

`WEATHER_KEY_n`
- Purpose: OpenWeather key pool for weather tool.
- Used in: `core/api_key_manager.py`.
- Required: required for live weather.

## 3. Request-Level Overrides
`/ask` accepts optional request fields:
- `llm_mode`
- `cloud_provider`

These are runtime request controls, not deployment secrets.

High-level precedence:
1. Request override (if valid) is applied in `api/app.py` via `llm_routing_context`.
2. Otherwise env defaults from `core/llm_mode.py` are used.
3. Router may derive an effective mode from backend availability (`agents/llm_router.py`) to avoid hard failure when one backend is unavailable.

## 4. Frontend Variables
Frontend variables are consumed by Vite and should be treated separately from backend runtime config.

`VITE_API_BASE_URL`
- Purpose: frontend API base URL.
- Used in: `frontend/src/lib/api.ts`.

`VITE_STREAM_SOFT_DELAY_MS`
- Purpose: soft delay marker for frontend streaming; no forced abort.
- Used in: `frontend/src/hooks/useStreamingPlan.tsx`.

`VITE_STREAM_HARD_NO_ACTIVITY_MS`
- Purpose: hard no-activity timeout before frontend falls back from streaming.
- Used in: `frontend/src/hooks/useStreamingPlan.tsx`.

`VITE_UI_MODE`
- Purpose: preview vs dev wording/labels in UI.
- Used in: `frontend/src/lib/uiMode.ts`.

`VITE_DEBUG_MODE`
- Purpose: enables debug-only UI surfaces.
- Used in: `frontend/src/components/DebugDrawer.tsx`.

Note:
- `frontend/.env.example` must contain only `VITE_*` variables.
- Backend variables like `LLM_MODE`, `USE_CLOUD_LLM`, or key vars should never be placed in frontend env files.

## 5. Docker and Runtime Overrides
- Primary container startup path is `Dockerfile` `CMD` (gunicorn), not `app/entrypoint.sh`.
- `app/entrypoint.sh` is retained for compatibility but is not authoritative in the default Docker path.
- Runtime env precedence follows normal process environment rules: container-injected vars override values loaded from `.env`.
- Validation flow (`full_validation.py`) may generate `.env.tmp` to run scenario-specific checks; this is test harness behavior, not production config ownership.

## 5a. Timeout Variables (Current)
Runtime timeout variables currently used in active backend/frontend paths:

`PLANNER_STREAM_INIT_TIMEOUT`
- Purpose: planner streaming init wait budget before returning stream-init timeout.
- Current behavior: clamped to a minimum safe floor in `agents/planner_agent.py`.

`PLANNER_LLM_TIMEOUT`
- Purpose: total planner LLM stream/non-stream budget for the explanation phase.
- Used in: `agents/planner_agent.py`.

`ROUTER_TIMEOUT`
- Purpose: overall LLM router request budget.
- Used in: `agents/llm_router.py`.

`LOCAL_LLM_TIMEOUT`
- Purpose: router first-chunk/per-chunk timeout budget for local backend path.
- Used in: `agents/llm_router.py`.

`CLOUD_LLM_TIMEOUT`
- Purpose: router first-chunk/per-chunk timeout budget for cloud backend path.
- Used in: `agents/llm_router.py` and cloud adapter defaults.

## 6. Deprecated / Legacy Variables

`OPENAI_API_KEY`
- Current behavior: read by legacy `core/async_llm_client.py`.
- Why deprecated: modern routing uses key-manager numbered keys (`OPENAI_KEY_n`), not single key env.
- Replacement: `OPENAI_KEY_n`.
- Status: temporarily supported for legacy init path only.

`ANTHROPIC_API_KEY`
- Current behavior: read by legacy `core/async_llm_client.py`.
- Why deprecated: modern runtime cloud path is adapter + key-manager centric.
- Replacement: provider-specific modern key-manager pattern (do not rely on legacy single-key path).
- Status: legacy compatibility only.

`CLOUD_BASE_URL`
- Current behavior: read by legacy `core/async_llm_client.py`.
- Why deprecated: modern runtime cloud routing is provider/adaptor based (`CLOUD_PROVIDER_CHAIN` / `CLOUD_PROVIDER`).
- Replacement: provider chain/default provider configuration.
- Status: legacy compatibility only.

`LLM_PRIORITY`
- Current behavior: only used to resolve legacy `LLM_MODE=hybrid` alias in `core/llm_mode.py`.
- Why deprecated: canonical control is `LLM_MODE`.
- Replacement: set `LLM_MODE` directly to canonical values.
- Status: temporarily supported for legacy compatibility.

`LLM_PREWARM`
- Current behavior: checked only in `app/entrypoint.sh`.
- Caveat: main Docker startup path uses `Dockerfile` `CMD` directly (gunicorn), so `LLM_PREWARM` is stale in that path.
- Replacement: `PLANNER_PREWARM` (used in `api/app.py` lifespan).
- Status: stale/deprecated in primary Docker runtime path.

## 7. Removed / Dead Variables
These variables are not part of the active canonical surface:

`PLANNER_STREAMING_ENABLED`
- Present in `.env` files.
- No backend/frontend read site found.

`VITE_TOTAL_LLM_TIMEOUT_MS`
- Removed from tracked frontend env config.
- No frontend usage exists.

`HOST` (Dockerfile ENV)
- Removed from Dockerfile.
- Gunicorn bind remains hardcoded to `0.0.0.0:${PORT}`.

## 8. Minimal Setup Examples

Ollama only:
```env
LLM_MODE=ollama_only
USE_CLOUD_LLM=0
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=openhermes
```

Cloud only (Gemini):
```env
LLM_MODE=cloud_only
USE_CLOUD_LLM=1
CLOUD_PROVIDER_CHAIN=gemini
CLOUD_PROVIDER=gemini
GEMINI_KEY_1=your_key_here
```

Ollama + cloud fallback (Ollama first):
```env
LLM_MODE=ollama_first
USE_CLOUD_LLM=1
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=openhermes
CLOUD_PROVIDER_CHAIN=gemini,openai
CLOUD_PROVIDER=gemini
GEMINI_KEY_1=your_gemini_key
OPENAI_KEY_1=your_openai_key
```

## 9. Provider-Specific Examples

Gemini-only cloud:
```env
LLM_MODE=cloud_only
CLOUD_PROVIDER_CHAIN=gemini
CLOUD_PROVIDER=gemini
USE_CLOUD_LLM=1
GEMINI_KEY_1=your_key_here
```

OpenAI-only cloud:
```env
LLM_MODE=cloud_only
CLOUD_PROVIDER_CHAIN=openai
CLOUD_PROVIDER=openai
USE_CLOUD_LLM=1
OPENAI_KEY_1=your_key_here
```

Gemini with OpenAI fallback:
```env
LLM_MODE=cloud_first
CLOUD_PROVIDER_CHAIN=gemini,openai
CLOUD_PROVIDER=gemini
USE_CLOUD_LLM=1
GEMINI_KEY_1=your_gemini_key
OPENAI_KEY_1=your_openai_key
```

Ollama + cloud hybrid fallback:
```env
LLM_MODE=ollama_first
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=openhermes
USE_CLOUD_LLM=1
CLOUD_PROVIDER_CHAIN=gemini,openai
GEMINI_KEY_1=your_gemini_key
OPENAI_KEY_1=your_openai_key
```

## 10. Migration Notes

From `OPENAI_API_KEY` to `OPENAI_KEY_n`
- Move from single legacy key var to numbered key pool entries.
- Keep old var only during transition.

From `LLM_PRIORITY` to `LLM_MODE`
- Replace legacy hybrid-priority combinations with explicit canonical mode.
- Example: `LLM_MODE=ollama_first` instead of `LLM_MODE=hybrid` + `LLM_PRIORITY=local-first`.

From entrypoint prewarm assumption to active startup path
- Primary startup path uses `api/app.py` lifespan and `PLANNER_PREWARM`.
- `LLM_PREWARM` in `app/entrypoint.sh` does not affect the default Docker `CMD` path.

## 11. Validation/Test-Only Variables
Do not treat these as production runtime config:

- `VALIDATION_USE_CLOUD_LLM`
- `VALIDATION_MODE`
- `VALIDATION_QUIET`
- `SMOKE_TIMEOUT`
- `SKIP_DOCKER_BUILD`
- `TESTING`

Also note `pytest.ini` sets test-only defaults (for example `PLANNER_PREWARM`, `LLM_MODE`, `USE_CLOUD_LLM`, `TESTING`) that intentionally do not represent production behavior.
