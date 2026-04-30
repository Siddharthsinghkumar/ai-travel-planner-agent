# Security Hardening Contract (S3/S4/S5)

Status: active  
Scope: response headers, logging/redaction, dependency/image scanning baseline

This document is the canonical S3/S4/S5 security contract for this repo.

## S3. Security Headers

Implementation point:
- Canonical policy is proxy-level in `deploy/Caddyfile.example`.
- App middleware remains focused on app behavior; edge security headers are set at Caddy.

Production header baseline:
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: frame-ancestors 'none'; base-uri 'self'; object-src 'none'`
- remove `Server` header (`-Server`)

Explicit decisions:
- HSTS: enabled only on HTTPS-facing production ingress (Caddy).
- Frame protections: both `X-Frame-Options` and CSP `frame-ancestors` baseline enabled.
- Content sniffing protection: enabled via `X-Content-Type-Options`.
- Referrer leakage: constrained by `strict-origin-when-cross-origin`.
- CSP: minimal non-breaking CSP enabled; strict script/style CSP is intentionally deferred to a dedicated compatibility phase.

Header verification:
- Script: `scripts/check_security_headers.sh`
- Run:
  - `scripts/check_security_headers.sh https://<public-domain>/health/live`
- The script fails if required headers are missing or mismatched.

## S4. Logging and Redaction

Sensitive data that must never be logged:
- API keys (`SERPAPI_KEY_*`, `OPENAI_KEY_*`, `GEMINI_KEY_*`, `WEATHER_KEY_*`, and equivalent provider keys).
- Authentication tokens (`ADMIN_TOKEN`, bearer/access/refresh/session tokens).
- Auth-bearing headers (`Authorization`, `X-Api-Key`, `X-Admin-Token`) and cookies.
- Credential-bearing URLs or key/value fragments in exception strings.
- Secret-like fields in request/provider payloads (including booking payloads when fields are token/key/auth-like).

Redaction rules:
- Replace values for secret-like key/value patterns (`...token=...`, `...api_key=...`, `...authorization=...`) with `***REDACTED***`.
- Mask JSON-style secret fields.
- Mask bearer token strings.
- Mask `logging` extra attributes with sensitive names.

Runtime implementation:
- `core.logging_config.SensitiveDataRedactionFilter` is attached to the console handler.
- Redaction is best-effort and must not break request/exception logging.

Request/exception logging policy:
- Allowed: request id, route, method, status, duration, classified failure reason/error class.
- Not allowed: raw secrets, auth headers, cookie values, full secret-bearing payloads.

Safe example:
- Input: `Authorization=Bearer abc123 token=xyz api_key=live_foo`
- Logged: `Authorization=***REDACTED*** token=***REDACTED*** api_key=***REDACTED***`

## S5. Dependency and Image Scanning Baseline

Canonical baseline process:
- Python dependency scan: `pip-audit` against local runtime environment.
- Container image scan: `trivy image` against built runtime image.

Current baseline and triage are documented in:
- `docs/dependency-image-scanning.md`

Scope note:
- This phase provides a practical scan-and-triage baseline, not a full security platform rollout.
- Paid platforms (for example Snyk, GitHub Advanced Security) are optional comparison points, not required by this repo.

## Cross-Links

- Secrets + transport/API-surface baseline (S1/S2): `docs/security-s1-s2-hardening.md`
- Reverse proxy/TLS/headers contract: `docs/reverse-proxy-caddy.md`
- Logging and monitoring defaults: `docs/logging-monitoring.md`
- Admin/debug exposure policy: `docs/admin-debug-exposure.md`

## Reference Basis

- OWASP HTTP Headers Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/HTTP_Headers_Cheat_Sheet.html
- OWASP Logging Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
- OWASP Vulnerable Dependency Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Vulnerable_Dependency_Management_Cheat_Sheet.html
- FastAPI middleware docs: https://fastapi.tiangolo.com/tutorial/middleware/
- FastAPI behind proxy / forwarded headers docs: https://fastapi.tiangolo.com/advanced/behind-a-proxy/
- Caddy header directive docs: https://caddyserver.com/docs/caddyfile/directives/header
- OpenTripPlanner security guidance (sensitive surfaces behind proxy/firewall): https://docs.opentripplanner.org/en/v1.5.0/Security/
- Trivy docs: https://trivy.dev/latest/docs/
- OWASP Dependency-Check docs: https://jeremylong.github.io/DependencyCheck/
- Snyk Container docs: https://docs.snyk.io/scan-with-snyk/snyk-container
- Snyk Open Source docs: https://docs.snyk.io/scan-with-snyk/snyk-open-source
