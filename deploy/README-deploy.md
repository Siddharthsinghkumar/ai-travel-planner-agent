# Deploy Guide — llm-travel-agent (M2)

Host-agnostic (Ubuntu ARM or x86). Single-node canonical topology per
`docs/deployment-topology.md`.

## 1. Prerequisites (Sid does these)

- [ ] Fresh Ubuntu VPS (Oracle Always-Free ARM or Hetzner CX22)
- [ ] SSH key pair generated, public key ready
- [ ] Domain name pointed to VPS IP (Cloudflare proxied)
- [ ] Cloudflare cache-bypass rules for `/ask` and `/jobs/*/events`
- [ ] `.env` file prepared with all required secrets (see `docs/environment-secrets-contract.md`)
- [ ] Place `.env` at `~/.env` on the VPS (chmod 600), NOT in the repo checkout

## 2. Host Provisioning

```bash
ssh root@<VPS_IP>
curl -O https://raw.githubusercontent.com/.../deploy/provision.sh
chmod +x provision.sh
./provision.sh
```

This installs: Docker, Docker Compose, ufw (22/80/443), fail2ban, sqlite3,
creates `llm-agent` user, and sets up directory structure.

## 3. App Deployment

```bash
ssh llm-agent@<VPS_IP>
git clone <repo-url> ~/app
cd ~/app

# Copy Caddyfile into place (⛔ Sid: update TRAVEL_DOMAIN first)
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy

# Deploy with docker compose prod overlay
docker compose -f docker-compose.yml -f deploy/compose.prod.yml up -d
```

## 4. Backup Setup

```bash
# Install cron job for nightly backups
sudo cp deploy/backup/llm-backup.cron /etc/cron.d/llm-travel-agent-backup
```

## 5. Verification

```bash
# Smoke test
BASE_URL=https://<your-domain> scripts/deploy_smoke.sh

# Health check
curl -fsS https://<your-domain>/health/live

# Restore test
sudo deploy/backup/pg_restore_test.sh
```

## 6. Monitoring

Grafana at `https://<your-domain>/grafana` (basic auth, ⛔ Sid sets credentials).
Prometheus scraping `api:9091` from within the compose network.
