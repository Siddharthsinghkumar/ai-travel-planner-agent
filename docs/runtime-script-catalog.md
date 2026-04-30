# Runtime And Script Catalog (Obsolescence Cleanup)

Status: active

This repository now supports one runtime/server path only:
- `Caddy` reverse proxy
- one systemd-managed `uvicorn` process on loopback (`--workers 1`)

Gunicorn and alternate process-manager runtime paths are not supported.

## Canonical Production / Deployment Artifacts

| Path | Role | Notes |
|---|---|---|
| `deploy/Caddyfile.example` | Canonical reverse-proxy template | TLS termination + ingress policy for production contract |
| `scripts/deploy_smoke.sh` | Canonical deployment smoke check | Readiness + health + one `/ask` contract check |
| `scripts/sqlite_backup.sh` | Canonical SQLite live-backup helper | Uses `VACUUM INTO`; retention pruning |
| `scripts/check_security_headers.sh` | Canonical HTTPS header verification | Verifies S3 header baseline on real HTTPS endpoint |

## Canonical Validation (Non-Production)

| Path | Role | Notes |
|---|---|---|
| `full_validation.py` | End-to-end local validation harness | Machine/docker and optional frontend validation flows |

## Diagnostic / Manual Helpers

| Path | Role | Notes |
|---|---|---|
| `tools/serpapi_manual_resolver.py` | Manual booking resolver parity tool | Operator diagnostics only; not runtime entrypoint |

## Test-Only Helpers

- Test helpers remain under `tests/` and are not runtime/deployment entrypoints.

## Removed As Obsolete In This Cleanup

- `app/entrypoint.sh` (legacy compatibility wrapper tied to old Docker/gunicorn story)
- `Failed api&scrapper script/aviationstack_flight_search.py`
- `Failed api&scrapper script/ixigo_scraper.py`
- `Failed api&scrapper script/selenium_test.py`
- `docs/showcase-commands.sh` (redundant command wrapper)
- `docs/demo-sheet.md` (near-duplicate command catalog; consolidated into `docs/operator-sheet.md` and `docs/runbook.md`)

These removed files were stale and outside the supported runtime/deployment contract.
