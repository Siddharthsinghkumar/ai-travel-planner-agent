# Security Verification and Pipeline Hygiene (S6/S7)

Status: active  
Last updated: 2026-04-07  
Scope: CI/CD + deployment-pipeline hygiene and final compact security verification/signoff

This document is the canonical S6/S7 record for this repository.

## S6. CI/CD and Deployment Pipeline Hygiene

### 1) Pipeline Surface Inventory

Current in-repo CI surface:
- GitHub Actions workflow: `.github/workflows/ci.yml`
  - `test` job: Python tests
  - `docker-build-and-push` job: image build/push + Trivy image scan

No additional CI providers are defined in-repo.

### 2) CI/Deploy Secret Inventory

| Secret | Purpose | Where it should live | Must never appear | Lifetime guidance | Rotation owner |
|---|---|---|---|---|---|
| `secrets.DOCKERHUB_USERNAME` | Docker image namespace/login username for publish step | GitHub Actions repository/environment secrets | Repo files, workflow plaintext values, shell `echo` output | Medium-lived acceptable, prefer environment-scoped | Platform owner / deployment operator |
| `secrets.DOCKERHUB_TOKEN` | Docker Hub auth token for push | GitHub Actions repository/environment secrets | Repo files, logs, shell output, screenshots | Prefer short-lived/least-scope token where available | Platform owner / deployment operator |
| `GITHUB_TOKEN` (ephemeral Actions token) | Checkout/read workflow context | Automatically issued by GitHub Actions | Printed workflow logs or artifacts | Ephemeral by design | GitHub-managed (repo admin controls permissions) |
| Deployment runtime secrets (`ADMIN_TOKEN`, `SERPAPI_KEY_n`, `OPENAI_KEY_n`, `GEMINI_KEY_n`, `WEATHER_KEY_n`) | Runtime auth/provider access | Deployment-time protected env injection | CI plaintext, repo files, logs | Rotated per operator policy | Deployment operator / platform owner |

### 3) Secret Masking / No-Echo Review

Reviewed surfaces:
- `.github/workflows/ci.yml`
- `scripts/deploy_smoke.sh`
- `scripts/sqlite_backup.sh`
- `scripts/check_security_headers.sh`
- Security/deploy docs and command examples

Findings:
- No `set -x` in retained deployment/security scripts.
- No direct secret `echo`/`printenv` usage in retained scripts.
- Workflow uses GitHub secret context for Docker login credentials.

Hygiene fixes applied in this pass:
- Added workflow-level least-privilege permission baseline: `permissions: contents: read`.
- Added `persist-credentials: false` to checkout steps.
- Normalized test dummy env vars toward canonical key naming (`*_KEY_1`) while keeping compatibility dummies for legacy paths.
- Added Trivy output file and artifact retention to preserve scan evidence without exposing secrets.

### 4) Branch / Deploy Protection Baseline (Policy)

Expected control baseline for this repo:
1. Protected default branch (`main`).
2. Required pull-request review before merge.
3. Required passing CI before merge/deploy.
4. Deploy-to-production actions limited to trusted branch + approved maintainers.
5. If GitHub Environments are used for production deploys, require environment reviewers.

Note:
- These are repository governance settings (platform-side), not fully enforceable from source code alone.

### 5) Deployment Identity Scope (Least Privilege)

Current model:
- CI push job authenticates to Docker Hub using `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN`.

Scope expectations:
- Docker token should be scoped to minimum required repository push rights only.
- CI identity should not have broad registry admin or unrelated org-level privileges.
- Runtime application secrets must remain environment-scoped and not shared into build jobs unless strictly needed.

OIDC note:
- This repo currently uses static registry secrets for Docker Hub publishing.
- If registry/platform supports OIDC workload identity in your environment, prefer short-lived federated credentials over long-lived static tokens.

### 6) Artifact Retention and Rollback Hygiene

Pipeline artifact baseline:
- Trivy image scan output retained as CI artifact (`trivy-image-scan`, 14 days).

Operational retention/rollback cross-links:
- DB backups and retention policy: `docs/persistence-backups.md` + `scripts/sqlite_backup.sh`.
- Deploy smoke verification: `scripts/deploy_smoke.sh`.
- Rollback guidance uses:
  - code rollback (git/release process),
  - config rollback (env injection correction),
  - DB restore when schema/data integrity requires it.

## S7. Final Security Verification Pass

### 1) Compact ASVS-Aligned Checklist

| Area (ASVS-aligned) | Status | Evidence |
|---|---|---|
| Configuration and secrets management | Pass | `docs/security-s1-s2-hardening.md`, `docs/environment-secrets-contract.md` |
| Transport security and trusted proxy handling | Pass | `docs/reverse-proxy-caddy.md`, `deploy/Caddyfile.example` |
| Response headers baseline | Pass | `docs/security-s3-s5-hardening.md`, `scripts/check_security_headers.sh` |
| Logging and sensitive-data redaction | Pass | `core/logging_config.py`, `tests/test_logging_redaction.py`, `docs/security-s3-s5-hardening.md` |
| Endpoint exposure/admin-debug controls | Pass | `docs/admin-debug-exposure.md`, `deploy/Caddyfile.example` |
| Dependency/image vulnerability process | Partial | `docs/dependency-image-scanning.md` (triaged findings remain accepted short-term) |
| CI/CD pipeline least privilege and secret hygiene | Partial | `.github/workflows/ci.yml`, this document (policy controls partly platform-governed) |
| Backup/restore operational readiness | Pass | `docs/persistence-backups.md`, `scripts/sqlite_backup.sh` |

### 2) Manual Verification Record

| Check | Method | Evidence | Result |
|---|---|---|---|
| CI secret usage and no-echo review | Workflow/script grep + workflow review | `.github/workflows/ci.yml`, retained scripts | Pass |
| Least-privilege workflow defaults | Manual review + workflow patch | `permissions: contents: read`, checkout `persist-credentials: false` | Pass |
| HTTPS/proxy/header contract present | Doc/template/script review | `docs/reverse-proxy-caddy.md`, `deploy/Caddyfile.example`, `scripts/check_security_headers.sh` | Pass |
| Logging redaction behavior | Existing focused tests | `tests/test_logging_redaction.py` | Pass |
| Admin/debug exposure policy | Doc/template review | `docs/admin-debug-exposure.md`, Caddy route restrictions | Pass |
| Dependency/image baseline triage | Existing scan artifacts + triage doc | `docs/dependency-image-scanning.md`, `validation_logs/*` | Partial |
| Deployment backup/restore readiness | Script + runbook contract review | `scripts/sqlite_backup.sh`, `docs/persistence-backups.md` | Pass |

### 3) Risk Acceptance (Explicit)

Accepted short-term risks (tracked):
1. Some dependency findings remain triaged as accepted short-term in `docs/dependency-image-scanning.md`.
2. Branch protection / environment reviewer controls are policy expectations and must be enforced in GitHub repository settings.
3. Docker publish still depends on static registry secrets (Docker Hub token), not OIDC federation.
4. Global public ingress rate-limiting/body-size hard controls remain deferred per S1/S2 baseline scope.

### 4) Limited-Exposure Conclusion

Conclusion: **ready for limited exposure**.

Basis:
- Core security controls from S1-S5 are documented and implemented.
- S6 pipeline hygiene now has explicit least-privilege workflow defaults, CI secret inventory, and artifact retention notes.
- S7 verification has explicit pass/partial/open statuses and named accepted risks.

Guardrails for this conclusion:
- Keep deployment scope limited and monitored.
- Enforce branch/environment protections in GitHub settings.
- Continue scheduled dependency remediation and periodic re-scan.

## Reference Basis

- OWASP CI/CD Security Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/CI_CD_Security_Cheat_Sheet.html
- OWASP Secrets Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html
- OWASP ASVS project: https://github.com/OWASP/ASVS
- GitHub Actions security hardening (secure use): https://docs.github.com/actions/security-guides/security-hardening-for-github-actions
- GitHub Actions deployments/environments protection rules: https://docs.github.com/actions/reference/workflows-and-actions/deployments-and-environments
- GitHub Actions OIDC guidance: https://docs.github.com/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect
- OWASP Software Supply Chain Security Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Software_Supply_Chain_Security_Cheat_Sheet.html
- OpenTripPlanner security pattern reference: https://docs.opentripplanner.org/en/v1.5.0/Security/
