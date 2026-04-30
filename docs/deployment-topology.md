# Deployment Topology Contract (Phase 7D.1)

Status: active  
Scope: production deployment topology for this repo only

## Decision

The canonical production topology is:
- single-node only
- one FastAPI app process
- reverse proxy in front of FastAPI
- process-local queue/state semantics (no distributed async guarantees)

## Supported Topology

1. Node and process model
- One Linux node/VM.
- One application process managed by service manager (`systemd`): `uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers --forwarded-allow-ips=127.0.0.1,::1`.
- No additional app workers/processes for shared request handling.

2. Reverse proxy role
- Reverse proxy terminates TLS and serves as the public ingress.
- Canonical reverse proxy implementation: Caddy (see `deploy/Caddyfile.example` and `docs/reverse-proxy-caddy.md`).
- Reverse proxy forwards application traffic to `127.0.0.1:8000`.
- Reverse proxy keeps admin/debug routes non-public (for example restrict `/debug/*` to private network/operator access).

3. App process role
- FastAPI app serves API endpoints and process-local async job queue.
- Async job contract remains single-worker/process-local.

4. Persistent database location
- Canonical production DB path: `DATABASE_URL=sqlite:////var/lib/llm-travel-agent/local.db`.
- This file must be on persistent storage owned by the deployment operator.

5. Log destination
- App logs: stdout/stderr captured by `systemd` journal (`journald`).
- Reverse-proxy access/error logs: proxy-managed log files (operator-managed retention).

6. Backup destination
- Database backup artifacts: `/var/backups/llm-travel-agent/` on the node.
- Off-host copy destination: platform-owner-managed backup target (object storage or equivalent durable remote store).

## Not Supported

- Multi-worker shared-state topology.
- Distributed async architecture (queue/state split across workers/nodes).
- Public debug/admin endpoints.
- Alternate production server stacks (for example Gunicorn worker-class paths).

## Operational Notes

- This contract is intentionally single-node and low-complexity.
- `docker-compose.yml`, Docker image defaults, and other local/demo paths are not the canonical production topology contract.
- If deployment needs move beyond single-node semantics, that is a separate explicit architecture phase.

Related operational contracts:
- Startup/readiness/liveness + smoke + failure matrix: `docs/startup-readiness-liveness.md`
- Reverse proxy/TLS/host/forwarded-header contract: `docs/reverse-proxy-caddy.md`
- Persistence + backup/restore contract: `docs/persistence-backups.md`
- Logging/monitoring defaults: `docs/logging-monitoring.md`
- Admin/debug exposure policy: `docs/admin-debug-exposure.md`
- S1/S2 secrets + transport/API-surface hardening: `docs/security-s1-s2-hardening.md`
- Runtime/script classification and retained helper catalog: `docs/runtime-script-catalog.md`

## Phase 7D Exit Checklist (Explicit / Checkable)

7D is complete only when all checks below are done and evidenced for the target environment.

| Check | Evidence command or artifact | Pass criteria |
|---|---|---|
| Fresh machine deploy from docs | Topology + env contracts (`docs/deployment-topology.md`, `docs/environment-secrets-contract.md`) | App and proxy come up on clean host without undocumented steps |
| HTTPS/proxy setup documented and tested | `docs/reverse-proxy-caddy.md`, `deploy/Caddyfile.example`, external `https://<domain>/health/live` | TLS terminates at Caddy; HTTP redirects to HTTPS; Uvicorn remains loopback-only |
| Health/readiness semantics clear | `docs/startup-readiness-liveness.md`, `scripts/deploy_smoke.sh` | `/health/live` + `/health/ready` behave as documented; `/health/deep` treated as diagnostic |
| Backups and restore exist | `docs/persistence-backups.md`, `scripts/sqlite_backup.sh` | Daily backup policy configured, restore drill steps documented and runnable |
| Logging/monitoring defaults in place | `docs/logging-monitoring.md`, `monitoring/alerts.yml` | Operator has default log-level policy and minimum metric/alert checklist |
| Admin/debug exposure policy explicit | `docs/admin-debug-exposure.md`, proxy rules in `deploy/Caddyfile.example` | `/debug/*` and diagnostic surfaces are not publicly exposed |
| End-to-end booking smoke in deployment env | `BASE_URL=https://<domain> scripts/deploy_smoke.sh` | Script exits success; `/ask` contract returns valid response payload |
