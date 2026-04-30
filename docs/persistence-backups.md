# Persistence and Backups Contract (Phase 7D.5)

Status: active  
Scope: canonical single-node SQLite persistence, backup, and restore policy

## Production Persistence Choice

- Supported production DB for this deployment contract: **SQLite**.
- Canonical live DB location: `DATABASE_URL=sqlite:////var/lib/llm-travel-agent/local.db`.
- Expected ownership:
  - DB directory (`/var/lib/llm-travel-agent/`) is created and owned by the deployment operator.
  - App service account has read/write permissions to the DB file.
- Durability assumption:
  - Live DB is stored on persistent node storage.
  - Off-host backup copy is required for host-loss recovery.

Journal/WAL operational note:
- The app does not enforce a custom SQLite journal-mode contract at startup.
- Backup policy therefore uses SQLite-supported live backup methods rather than raw file copies of a changing DB.

## Canonical Backup Method

Default method: **SQLite `VACUUM INTO` snapshot backup** executed against the live DB.

Why this method:
- Produces a transactionally consistent backup file.
- Supported by SQLite documentation as a safe online backup approach for file snapshots.

Do not use as default:
- Plain filesystem copy (`cp`) of an actively changing SQLite DB.
- If an operator must do file-level copy, the app must be quiesced first and SQLite sidecar files handled correctly.

Backup helper script:
- `scripts/sqlite_backup.sh`

## Backup Schedule (Default)

- Daily full backup at `02:30` server local time.
- Mandatory pre-deploy backup before any schema-affecting release.
- Backup destination on node: `/var/backups/llm-travel-agent/`.
- Required off-host copy: platform-owner-managed durable store (object storage or equivalent).

Example systemd timer cadence (operator-managed):
- daily run of `scripts/sqlite_backup.sh`.
- optional second run before scheduled deploy windows.

## Retention Policy (Default)

- On-node retention: keep last **14** daily backups.
- Off-host retention: keep last **30** daily backups.
- Older artifacts are pruned by operator policy/tools.

## Restore Procedure

### A) Bad deploy rollback (same host)

1. Stop app process (`systemctl stop llm-travel-agent`).
2. Confirm process is down.
3. Select restore point from `/var/backups/llm-travel-agent/`.
4. Copy backup file to canonical DB path:
   - target: `/var/lib/llm-travel-agent/local.db`
5. Restore ownership/permissions for app service account.
6. Start app process (`systemctl start llm-travel-agent`).
7. Run deployment smoke (`scripts/deploy_smoke.sh`) and verify `/health/ready`, `/health`, `/health/deep`, `/ask`.

### B) Host loss / replacement

1. Provision replacement host with canonical topology from `docs/deployment-topology.md`.
2. Restore environment/secrets per `docs/environment-secrets-contract.md`.
3. Retrieve latest valid off-host DB backup.
4. Place DB at `/var/lib/llm-travel-agent/local.db` with correct ownership.
5. Start Caddy + app service.
6. Run smoke and operator checks.

## Migration and Rollback Notes

- This repo does not currently use a standalone migration framework contract for production rollout.
- Schema-affecting deploy rule:
  - take a pre-deploy DB backup first,
  - deploy app,
  - validate with smoke,
  - restore DB if schema/app mismatch creates runtime failure.
- Code-only rollback (no schema change):
  - rollback app version first,
  - DB restore usually not required.
- Schema-breaking rollback:
  - prefer restore-from-backup to force-consistent app+DB state,
  - avoid ad-hoc manual schema edits during incident response unless operator intentionally chooses that path.

## Operator Commands (Reference)

```bash
# Backup using canonical script and defaults
DATABASE_URL='sqlite:////var/lib/llm-travel-agent/local.db' \
BACKUP_DIR='/var/backups/llm-travel-agent' \
RETAIN_COUNT=14 \
scripts/sqlite_backup.sh

# List backups
ls -1t /var/backups/llm-travel-agent/backup_*.db
```

## Reference Basis

- SQLite backup API / online backup guidance: https://www.sqlite.org/backup.html
- SQLite `VACUUM INTO`: https://www.sqlite.org/lang_vacuum.html
- FastAPI deployment concepts (startup/restart/ops context): https://fastapi.tiangolo.com/deployment/concepts/
