# Dependency and Image Scanning Baseline (S5)

Status: active  
Last updated: 2026-04-07  
Scope: practical vulnerability baseline for this repo (single-node deployment contract)

## Tools Used

- Python dependencies: `pip-audit`
- Container image and bundled packages: `trivy` (open-source)

Reference-only commercial equivalents:
- Snyk Open Source / Snyk Container
- GitHub Advanced Security dependency review

## Commands Run

1. Local Python environment dependency scan:

```bash
XDG_CACHE_HOME=/tmp venv/bin/pip-audit --progress-spinner off --local --format json -o validation_logs/s5_pip_audit_local.json
```

2. Runtime image scan (HIGH/CRITICAL, ignoring unfixed):

```bash
docker build -t llm-travel-agent:s5scan .
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/validation_logs:/out" \
  aquasec/trivy:0.50.2 image \
  --severity HIGH,CRITICAL --ignore-unfixed \
  --format json -o /out/s5_trivy_image.json \
  llm-travel-agent:s5scan
```

3. Repository filesystem scan attempt (HIGH/CRITICAL, ignore unfixed):

```bash
docker run --rm -v "$PWD:/src" aquasec/trivy:0.50.2 filesystem \
  --severity HIGH,CRITICAL --ignore-unfixed \
  --format json -o /src/validation_logs/s5_trivy_fs.json /src
```

4. Follow-up local dependency re-scan after targeted upgrades:

```bash
XDG_CACHE_HOME=/tmp venv/bin/pip-audit --progress-spinner off --local --format json -o validation_logs/s5_pip_audit_local_after_followup.json
```

5. Follow-up image re-scan after targeted upgrades:

```bash
docker build -t llm-travel-agent:s5followup .
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/validation_logs:/out" \
  aquasec/trivy:0.50.2 image \
  --severity HIGH,CRITICAL --ignore-unfixed \
  --format json -o /out/s5_trivy_image_after_followup.json \
  llm-travel-agent:s5followup
```

## Baseline Summary

- Initial `pip-audit` local environment:
  - 23 vulnerabilities across 11 packages.
  - Artifact: `validation_logs/s5_pip_audit_local.json`
- Follow-up `pip-audit` local environment:
  - 10 vulnerabilities across 6 packages.
  - Artifact: `validation_logs/s5_pip_audit_local_after_followup.json`
- Initial Trivy image scan:
  - 4 HIGH, 0 CRITICAL.
  - Artifact: `validation_logs/s5_trivy_image.json`
- Follow-up Trivy image scan:
  - 5 HIGH, 0 CRITICAL (all in Node frontend packages; no Python package HIGH/CRITICAL findings).
  - Artifact: `validation_logs/s5_trivy_image_after_followup.json`
- Trivy filesystem scan:
  - 0 HIGH/CRITICAL reported in this run (artifact: `validation_logs/s5_trivy_fs.json`).
  - Treat as supplemental; dependency triage should rely on `pip-audit` + image scan outputs above.

## Triage

### Must-Fix Follow-Up Status (Closed)

Previously required must-fix set and result:

1. `langchain-core`
- Status: fixed by removing unused `langchain` runtime dependency from `requirements.txt` and uninstalling `langchain*` packages from local runtime env.
- Verification:
  - Not present in `validation_logs/s5_pip_audit_local_after_followup.json`.
  - Not present in `validation_logs/s5_trivy_image_after_followup.json`.

2. `starlette`
- Status: fixed.
- Version path: `starlette 0.46.2` -> `starlette 1.0.0` (with compatible `fastapi` upgrade).
- Verification: no `starlette` findings in follow-up `pip-audit` output.

3. `urllib3` / `requests`
- Status: fixed.
- Version path:
  - `urllib3 2.4.0` -> `urllib3 2.6.3`
  - `requests 2.32.3` -> `requests 2.33.1`
- Verification: no `urllib3` or `requests` findings in follow-up `pip-audit` output.

### Accepted Short-Term (Track with Timebox)

1. Node package findings in image (`flatted`, `picomatch`, `vite`)
- Evidence: follow-up Trivy image HIGH findings in bundled Node dependencies.
- Rationale: app runtime is Python API; Node packages are not the production serving runtime process.
- Follow-up: tighten image context (`.dockerignore`) and/or pin/upgrade frontend dependency tree in a dedicated dependency-refresh pass.

2. `streamlit` Windows-only advisory
- Evidence: CVE-2026-33682.
- Rationale: deployment contract is Linux single-node.
- Follow-up: keep tracked; update when normal dependency refresh occurs.

3. Optional/non-core packages (`pillow`, `pygments`, `tornado`, `protobuf`, `orjson`)
- Rationale: not all are exercised on the critical runtime path in this deployment profile.
- Follow-up: resolve in scheduled dependency-upgrade pass with compatibility testing.

### False Positives / De-duplication Notes

1. Duplicate `pillow` advisory rows
- `pip-audit` reports duplicated identifier mappings (`PYSEC-2025-61` alias overlap).
- Treat as one underlying issue for remediation planning.

## Operational Policy

- Minimum S5 gate before public exposure:
  - No untriaged CRITICAL findings.
  - All HIGH findings must be either:
    - remediated, or
    - explicitly documented in accepted-risk list with rationale and owner.
- Re-run baseline scans after dependency changes and before release candidate cut.

## References

- OWASP Vulnerable Dependency Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Vulnerable_Dependency_Management_Cheat_Sheet.html
- Trivy docs: https://trivy.dev/latest/docs/
- OWASP Dependency-Check docs: https://jeremylong.github.io/DependencyCheck/
- Snyk Open Source docs: https://docs.snyk.io/scan-with-snyk/snyk-open-source
- Snyk Container docs: https://docs.snyk.io/scan-with-snyk/snyk-container
