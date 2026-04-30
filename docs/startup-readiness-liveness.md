# Startup / Readiness / Liveness Contract (Phase 7D.3)

Status: active  
Scope: canonical single-node production runtime behavior

## Startup Sequence (Boot To Ready)

Canonical process model:
- `systemd` starts one `uvicorn` process on loopback (`127.0.0.1:8000`, single worker).
- Reverse proxy sits in front and sends traffic only after readiness.

### 1) Preconditions before app boot

- Env/secrets injected (see `docs/environment-secrets-contract.md`).
- Persistent DB path exists and is writable for `DATABASE_URL` (canonical: `sqlite:////var/lib/llm-travel-agent/local.db`).
- Reverse proxy is configured but should not route external traffic until `/health/ready` returns `200`.

### 2) Startup actions in app lifespan

At startup, `api/app.py` performs (best-effort, resilient) initialization:
1. Configure logging and emit deprecation warnings.
2. Run DB init (`init_db()`), including table creation.
3. Load provider keys from env into key manager.
4. Refresh cloud-provider chain from env (when cloud is admin-enabled).
5. Register key event listeners.
6. Acquire refresh-owner lock (single refresh-owner semantics) and, if owner:
   - start key refresh loop
   - start periodic SerpAPI reconcile loop in background (startup does not block on reconcile completion)
7. Start background job worker loop.
8. Optionally run planner prewarm (best-effort, non-blocking).
9. Set `startup_complete=True`.

Important truth:
- Startup is resilient by design. Several config/provider problems are logged and surfaced as degraded/unavailable at runtime instead of hard process exit.

### 3) Boot complete vs ready-to-serve

- **Boot complete**: process is alive (`/health/live` = alive).
- **Ready to serve**: `/health/ready` returns `200` with status `ok`.
- `/health/ready` returns `503` only while core startup is incomplete (`status=starting`).
- Best-effort prewarm can continue after readiness becomes `ok`.

## Health Endpoint Semantics

| Endpoint | Purpose | External provider calls | LB/liveness suitability | Notes |
|---|---|---|---|---|
| `/health/live` | Process liveness only | No | Best for frequent liveness probes | Returns `{"status":"alive"}` when process is up |
| `/health/ready` | Startup/readiness gate | No | Best readiness gate before routing traffic | `503` only while `startup_complete` is false |
| `/health` | Lightweight runtime status | No external airline/weather/cloud API pings; includes local dependency checks | Safe for periodic platform monitoring (not the deepest diagnostic) | Returns `ok/degraded/fail` plus dependency/topology context |
| `/health/deep` | Diagnostic deep dependency truth | Yes, can touch external providers | Not for high-frequency LB probing | Use for operator diagnosis and low-frequency checks |
| `/health/keys` | High-level key status | No | Diagnostic | Sanitized key-state visibility |

Operational guidance:
- Liveness probe target: `/health/live`
- Readiness probe target: `/health/ready`
- Frequent service status (internal): `/health`
- Deep diagnostics only: `/health/deep`

## Deployment Smoke Test

Canonical smoke script:
- `scripts/deploy_smoke.sh`

What it does:
1. Waits for `/health/ready` to become `200`.
2. Checks `/health` and prints top-level status/dependencies.
3. Checks `/health/deep` and prints deep status (diagnostic).
4. Runs one `/ask` route/date case and prints contract-relevant output.

Usage:

```bash
# Run from trusted/internal operator path behind reverse proxy policy
BASE_URL="https://travel.example.com" scripts/deploy_smoke.sh

# If using temporary/self-signed certs during pre-cutover only
BASE_URL="https://travel.example.com" INSECURE_TLS=1 scripts/deploy_smoke.sh
```

Optional inputs:
- `SMOKE_ORIGIN` (default `DEL`)
- `SMOKE_DESTINATION` (default `BOM`)
- `SMOKE_DATE` (default UTC today + 21 days)

## Failure Matrix

| Failure class | Likely symptom | Affected endpoints | Startup behavior | What to check first | Expected health behavior |
|---|---|---|---|---|---|
| DB unavailable/path wrong | DB warnings/errors in logs; DB-backed features fail | `/health`, `/health/deep`, DB-dependent flows | App usually boots (resilient startup) | `DATABASE_URL`, DB file path/permissions, app logs | `/health` includes `dependencies.database=fail/degraded`; `/health/deep` often `degraded/fail` |
| Missing/incorrect env (critical contract vars) | Runtime misbehavior (e.g., missing admin auth token, wrong CORS, missing provider keys) | `/debug/*`, `/ask`, `/health*` | App usually boots (degrade rather than hard fail) | Env injection source, `docs/environment-secrets-contract.md`, startup logs | `/health/ready` can still be `ok`; `/health`/`/health/deep` reveal degraded/unavailable dependencies |
| Upstream provider degraded (SerpAPI/Ollama/weather/cloud) | `/ask` degraded fallback, timeout/unavailable reasons | `/ask`, `/health/deep`, sometimes `/health` dependency fields | App boots and stays alive | Provider key state (`/health/keys`), `/health/deep`, network/provider status | `/health/live` and `/health/ready` can remain healthy while `/health/deep` is degraded/fail |

Exposure note:
- `/health/deep` and `/health/keys` are diagnostic surfaces and are protected/internal by default in the canonical proxy policy (`docs/admin-debug-exposure.md`).

## Reference Basis

- FastAPI Deployment Concepts: startup managers, restarts, pre-start steps: https://fastapi.tiangolo.com/deployment/concepts/
- FastAPI HTTPS and forwarded headers behind TLS termination proxy: https://fastapi.tiangolo.com/deployment/https/
