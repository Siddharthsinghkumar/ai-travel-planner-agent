# Reverse Proxy / TLS / Host Contract (Phase 7D.4)

Status: active  
Scope: canonical single-node production ingress contract

## Canonical Proxy Choice

Chosen proxy: **Caddy**.

Why this one:
- Minimal single-node operational complexity.
- Built-in automatic HTTPS and certificate renewal for public domains.
- Clean reverse-proxy configuration to loopback `uvicorn`.
- Good fit for current architecture: one app process on localhost, one public ingress point.

Canonical template file:
- `deploy/Caddyfile.example`

## TLS Termination Plan

- TLS terminates at Caddy.
- FastAPI/Uvicorn listens on loopback HTTP only (`127.0.0.1:8000`).
- Public traffic reaches only Caddy (ports 80/443), never direct Uvicorn.
- For domain hosts, Caddy obtains and renews certificates automatically.

## Forwarded Headers / Trusted Proxy Notes

Uvicorn launch contract (canonical):

```bash
uvicorn api.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips=127.0.0.1,::1
```

Rationale:
- App must trust forwarded headers only from the local reverse proxy.
- This ensures correct request scheme/host awareness (`https`, domain host) behind proxy.

Caddy behavior:
- `reverse_proxy` sets/augments `X-Forwarded-*` headers.
- Caddy ignores spoofed incoming `X-Forwarded-*` values by default.
- If Caddy is itself behind another proxy/CDN, configure trusted proxy ranges explicitly in Caddy.

## Public Host / Base URL Behavior

Canonical public host behavior:
- API root: `https://<public-domain>/`
- Docs endpoint path: `https://<public-domain>/docs` (protected/internal by default)
- OpenAPI path: `https://<public-domain>/openapi.json` (protected/internal by default)

Canonical path contract:
- Root-path hosting only (`/`).
- Subpath/root-prefix hosting (for example `/api/v1`) is **not** part of this canonical deployment contract.
- If subpath hosting is needed later, it must be an explicit follow-up phase using `root_path` proxy/app coordination.

## HTTP -> HTTPS Redirect Strategy

- HTTP requests on port 80 are permanently redirected to HTTPS by Caddy automatic HTTPS behavior.
- Uvicorn is loopback-only and not publicly exposed.
- Public ingress policy is HTTPS-only through Caddy.

## Security Header Baseline (S3)

Canonical implementation point:
- Proxy-level in Caddy (`deploy/Caddyfile.example`) for production responses.

Headers intentionally set by default:
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: frame-ancestors 'none'; base-uri 'self'; object-src 'none'`
- remove `Server` response header where possible (`-Server`)

CSP decision for this phase:
- A minimal framing/object/base-uri CSP is set at proxy level.
- A strict script/style CSP is intentionally deferred to a later dedicated phase to avoid breaking FastAPI docs tooling and operational pages without a route-by-route compatibility test.

Header verification command:

```bash
scripts/check_security_headers.sh https://<public-domain>/health/live
```

The check script fails if any required S3 baseline header is missing or mismatched.

## Sensitive Surface Exposure Rule

- Public debug/admin endpoints are not supported.
- Reverse proxy must block `/debug/*` from non-private source networks.
- Reverse proxy should also keep diagnostic-heavy surfaces non-public by default:
  - `/health/deep`
  - `/health/keys`
  - `/metrics`
  - `/llm/options`
  - `/docs`, `/redoc`, `/openapi.json`
- Canonical route policy details: `docs/admin-debug-exposure.md`.
- S1/S2 transport/API-surface baseline (including CORS and rate-limit scope): `docs/security-s1-s2-hardening.md`.

## Reference Basis

- FastAPI behind proxy / forwarded header trust and proxy-aware redirects:
  - https://fastapi.tiangolo.com/advanced/behind-a-proxy/
  - https://fastapi.tiangolo.com/deployment/https/
- FastAPI full-stack production pattern (Traefik reverse proxy with automatic HTTPS):
  - https://fastapi.tiangolo.com/project-generation/
  - https://github.com/fastapi/full-stack-fastapi-template
- OpenTripPlanner operational security guidance (put sensitive surfaces behind firewall/reverse proxy):
  - https://docs.opentripplanner.org/en/v1.5.0/Security/
- Caddy official docs:
  - https://caddyserver.com/docs/automatic-https
  - https://caddyserver.com/docs/caddyfile/directives/reverse_proxy
  - https://caddyserver.com/docs/caddyfile/directives/header
- OWASP HTTP Headers Cheat Sheet:
  - https://cheatsheetseries.owasp.org/cheatsheets/HTTP_Headers_Cheat_Sheet.html
