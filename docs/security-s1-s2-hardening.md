# Security Hardening Contract (S1/S2)

Status: active  
Scope: secrets/config handling and transport/API-surface policy for canonical single-node deployment

This document is the canonical S1/S2 security contract for this repo.

## S1. Secrets and Configuration Hardening

### 1) Secret Inventory (Canonical)

| Secret category | Env names / source | Purpose | Production storage | Must never appear | Rotation owner | Runtime behavior if missing |
|---|---|---|---|---|---|---|
| Admin auth token | `ADMIN_TOKEN` | Protects admin/debug endpoints (`/debug/*`) | Protected environment injection / managed secret store | Committed files, logs, screenshots, CI console output | Deployment operator / platform owner | Admin/debug routes return `403` |
| SerpApi keys | `SERPAPI_KEY_n` | Flight search + booking handoff resolution | Protected environment injection / managed secret store | Repo files, CLI arg history, process-list-visible command args, logs | Deployment operator / platform owner | Live flight/booking degrades/unavailable |
| Weather API keys | `WEATHER_KEY_n` | Weather enrichment | Protected environment injection / managed secret store | Same as above | Deployment operator / platform owner | Weather path degrades/unavailable |
| Cloud LLM keys | `OPENAI_KEY_n`, `GEMINI_KEY_n` | Cloud LLM requests | Protected environment injection / managed secret store | Same as above | Deployment operator / platform owner | Cloud provider unusable; app can still run degraded |
| CI registry credentials | GitHub Actions `secrets.DOCKERHUB_USERNAME`, `secrets.DOCKERHUB_TOKEN` | Docker image publish in CI | CI secret store only | Repository files and workflow plaintext values | Platform owner / deployment operator | Docker push step fails |
| Optional DB credentials (if non-SQLite URL used) | `DATABASE_URL` credential segment | DB auth (only if URL contains credentials) | Protected environment injection | Logs/docs/example files | Deployment operator | DB connectivity failure/degradation |

Additional note:
- Legacy variables like `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` are compatibility-only and non-canonical.

### 2) Secret Storage Rules

Supported production pattern:
- Secrets come from managed secret injection or protected environment files owned by operations.
- `.env.example` is non-secret template only.

Not supported:
- Committing real secrets to repo.
- Printing secrets in scripts/logs/CI output.
- Passing secrets on command lines when avoidable (process-list/shell-history exposure risk).

### 3) Prohibited Exposure Surfaces

Secrets must not be present in:
- Git-tracked files (`.env`, docs, scripts, workflow YAML values).
- Runtime logs (`journald`, proxy logs, validator logs).
- CI stdout/stderr.
- Shell history / process arguments where avoidable.

Operational hygiene:
- Use environment-based secret loading in scripts/tools whenever possible.
- Avoid `--api-key <value>` style command examples for production/operator docs.

### 4) Rotation Procedure (Practical)

1. Prepare replacement secret in secret-management system.
2. Inject new secret into deployment environment (do not remove old yet).
3. Restart/reload service.
4. Validate with `/health`, `/health/deep`, and one targeted functional check.
5. Revoke/remove old secret upstream.
6. Confirm stable runtime after revoke.
7. Record rotation timestamp/owner in operator records.

### 5) Secret Leak Incident Procedure

1. Identify leaked secret category and blast radius.
2. Revoke/disable leaked secret immediately.
3. Issue replacement and redeploy/restart affected service.
4. Validate runtime health and impacted endpoints.
5. Search affected logs/artifacts/history for additional spread.
6. Remove leaked material from docs/scripts/config where possible and rotate again if uncertainty remains.
7. Record incident, root cause, and follow-up controls.

## S2. Transport and API Surface Hardening

### 1) HTTPS-Only Requirement

- Public API ingress is HTTPS-only through Caddy.
- TLS terminates at Caddy.
- Uvicorn remains loopback-only HTTP (`127.0.0.1:<port>`) and is never public.
- HTTP public traffic is redirected to HTTPS by proxy behavior.

### 2) CORS Trusted-Origin Policy

Runtime policy:
- `ALLOWED_ORIGINS` must contain explicit origins (`scheme://host[:port]`).
- Wildcard `*` is not accepted by runtime origin parser.
- Invalid origins are ignored.
- If `ALLOWED_ORIGINS` is explicitly set but no valid origins parse, cross-origin browser access is denied.
- Dev fallback origins are localhost-only and used only when `ALLOWED_ORIGINS` is not set.

Operational rule:
- Production must set `ALLOWED_ORIGINS` explicitly to trusted frontend origins only.

### 3) Admin/Debug/Diagnostic Surface Policy

- `/debug/*`: internal-only + `X-Admin-Token` required.
- `/health/deep`, `/health/keys`, `/metrics`, `/llm/options`, `/docs`, `/redoc`, `/openapi.json`: protected/internal by default.
- Proxy policy blocks these surfaces from non-private source ranges by default (`deploy/Caddyfile.example`).

Reference operational model:
- keep sensitive/admin-capable surfaces behind reverse proxy/firewall controls; do not expose publicly.

### 4) Request Size / Timeout / Rate-Limit Baseline

Current baseline (truthful):
- Request size limit: no explicit app-level body-size cap is currently enforced; proxy template also does not set an explicit max body size.
- Timeouts: app has explicit request/planner/router timeouts; SerpAPI account reconcile runs in background loops so startup is not blocked on reconcile completion.
- Rate limiting: no global public ingress rate limiter is implemented in this phase. Existing protection is request-path admission control for `/ask` (duplicate/inflight guard), not full API-wide rate limiting.

Operational expectation:
- Treat this as baseline contract, not full anti-abuse coverage.
- If deployment threat model requires stricter controls, add explicit proxy request-size/rate-limit controls in a dedicated follow-up phase.

## Cross-Links

- Environment + secrets contract: `docs/environment-secrets-contract.md`
- Reverse proxy + TLS contract: `docs/reverse-proxy-caddy.md`
- Admin/debug exposure policy: `docs/admin-debug-exposure.md`
- Deployment topology contract: `docs/deployment-topology.md`

## Reference Basis

- OWASP Secrets Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html
- OWASP Key Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Key_Management_Cheat_Sheet.html
- OWASP REST Security Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html
- OWASP CI/CD Security Cheat Sheet (secrets handling in pipelines): https://cheatsheetseries.owasp.org/cheatsheets/CI_CD_Security_Cheat_Sheet.html
- FastAPI CORS docs: https://fastapi.tiangolo.com/tutorial/cors/
- FastAPI behind proxy / forwarded headers: https://fastapi.tiangolo.com/advanced/behind-a-proxy/
- FastAPI HTTPS deployment guidance: https://fastapi.tiangolo.com/deployment/https/
- OpenTripPlanner security guidance: https://docs.opentripplanner.org/en/v1.5.0/Security/
