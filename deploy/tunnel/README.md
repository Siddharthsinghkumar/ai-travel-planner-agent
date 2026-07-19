# Cloudflare Tunnel — llm-travel-agent ingress

> **Why a tunnel:** the AWS security group opens NO inbound web ports (80/443).
> All app ingress is the outbound `cloudflared` tunnel → Cloudflare edge → HTTPS.
> The box keeps an auto public IP for egress only (Docker pulls, apt, outbound APIs).

## Topology

```
Internet → Cloudflare Edge (TLS) → cloudflared (outbound) → caddy:80 → api:8000
```

- `cloudflared` runs in the compose network, reaches `caddy:80` by service name.
- Caddy's SSE tuning (`flush_interval -1`, 300s stream timeouts) still applies on `/ask` and `/jobs/*/events`.
- No inbound firewall rules for HTTP/HTTPS on the host.

## Two paths

### Path 1: Named tunnel (stable, needs Cloudflare account)

Sid creates a named tunnel in his Cloudflare Zero Trust dashboard, captures the
`TUNNEL_TOKEN`, and places it in `.env` (chmod 600, never committed — N-M2.1).

```bash
# Set the token in the environment (Sid's hand only)
export TUNNEL_TOKEN="<placeholder — Sid's real token>"

# Bring up the full stack + tunnel
docker compose -f docker-compose.yml \
               -f deploy/compose.prod.yml \
               -f deploy/compose.tunnel.yml \
               up -d
```

The public hostname is configured in Cloudflare's tunnel ingress rules, pointing at
`caddy:80` inside the compose network. When `eu.org` domain is approved, it can be
added as a Cloudflare zone with the tunnel as its origin.

### Path 2: Quick tunnel (ephemeral, zero account, for testing)

No token needed. Run on the host (not inside compose):

```bash
cloudflared tunnel --url http://localhost:80
```

Prints an ephemeral `*.trycloudflare.com` URL. Good for a quick smoke test before
the named tunnel is configured. App traffic flows: cloudflared → localhost:80 (Caddy) → api:8000.

## Compose network note

The existing `deploy/Caddyfile` hardcodes `127.0.0.1:8000` as the reverse-proxy
target (written for caddy-on-host + uvicorn-on-localhost). When Caddy runs inside
the compose network alongside `api`, the correct target is `api:8000` (compose
service name). Before going live, **update the two `reverse_proxy ... 127.0.0.1:8000`
lines in `deploy/Caddyfile` to `reverse_proxy ... api:8000`**. The quoted path
matchers (`/ask`, `/jobs/*/events`) and all SSE tuning remain unchanged.

## Security

- `TUNNEL_TOKEN` is Sid's — PLACEHOLDER only in this repo (N-M2.1).
- The tunnel is outbound-only (WebSocket to Cloudflare edge). No inbound port 80/443
  on the AWS security group.
- Cloudflare provides DDoS, CDN, and WAF at the edge — included with the free plan.
- SSE cache-bypass rules must be configured on the Cloudflare zone (⛔ Sid's step;
  see plan §5 M2-T6).
