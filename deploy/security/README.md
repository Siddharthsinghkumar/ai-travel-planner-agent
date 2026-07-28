# Security Scanning — deploy/security/

## Live commands (§6 M2-T12)

```bash
# ZAP full scan
ZAP_FULL_TARGET=https://<domain> ./deploy/security/zap-full.sh

# nuclei scan
NUCLEI_TARGET=https://<domain> ./deploy/security/nuclei-scan.sh
```

## Reports

All reports write to `plans/qa/` (untracked). Do not commit them — they contain
host-specific findings.


## Rule: new HIGH finding → STOP

Per §6 M2-T12, if a scan produces a new HIGH-severity finding, STOP and report
before proceeding. The ZAP tuning file (`zap-full.tsv`) can IGNORE/WARN false
positives after audit.

## Template set (nuclei)

- `http/exposures` — exposed files, directories, configs
- `http/misconfiguration` — common web misconfigs
- `ssl` — TLS/SSL issues

## ZAP tuning

`zap-full.tsv` is LOCAL to this runner. It is NEVER `.zap/rules.tsv` (CI-owned,
frozen per N-M2.5). Add IGNORE/WARN rules here after auditing the live scan.
