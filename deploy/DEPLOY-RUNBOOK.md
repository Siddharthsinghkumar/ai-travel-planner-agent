# Deploy-Day Runbook — llm-travel-agent

> **Consolidated checklist stitching all `deploy/` artifacts into one ordered sequence.**
> Each step links its source README and gives the literal command. ⛔ = Sid's account/key needed.
> NO new logic, NO secrets — this is an index over what already exists.

## Pre-flight: what you need before starting

- AWS account (new = $100–200 credits, ~7–14 months) per **⛔G2**
- NIM / Groq / Gemini keys placed in `.env` by Sid's hand only per **⛔G3** (`docs/environment-secrets-contract.md`)
- Cloudflare account with a zone (domain) + Zero Trust enabled
- Local machine: `awscli`, `terraform >= 1.5.0`, `ssh`, `scp`, `k6`, `git`

---

## Step 0: AWS account + budget in console FIRST

> Source: `deploy/terraform/aws/README-aws.md` Step 0

⛔ Sid creates the AWS account and sets a billing budget **before touching Terraform**.

```bash
# In AWS Console:
# 1. Create account (or use existing)
# 2. Billing Dashboard → Budgets → Create budget
# 3. Cost budget, monthly, $5, add alert email to your-email@example.com
# 4. Save — budget email fires even if Terraform isn't running
```

Credit burn note: t3.micro + 30GB gp3 + public IPv4 ≈ $14/month → credits last ~7–14 months.
If Oracle card clears later, migrate to permanent ₹0/24GB (`terraform destroy` on AWS, see Step 13).

---

## Step 1: IAM user + programmatic access

> Source: `deploy/terraform/aws/README-aws.md` Step 1

⛔ Sid creates an IAM user and captures access keys.

```bash
# In AWS Console → IAM → Users → Create user: llm-travel-agent-tf
# Attach policies: AmazonEC2FullAccess, AmazonVPCFullAccess
# Security credentials → Create access key → Application running outside AWS
# Save Access Key ID and Secret Access Key
```

---

## Step 2: AWS CLI + `aws configure`

> Source: `deploy/terraform/aws/README-aws.md` Step 2

```bash
sudo apt-get install -y awscli

aws configure
# AWS Access Key ID:     <paste from Step 1>
# AWS Secret Access Key: <paste from Step 1>
# Default region:        ap-south-1
# Default output format: json

aws sts get-caller-identity   # verify
```

---

## Step 3: Terraform variables

> Source: `deploy/terraform/aws/README-aws.md` Step 3

```bash
cd deploy/terraform/aws
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your values:

```hcl
region         = "ap-south-1"
admin_cidr     = "YOUR-IP/32"          # ⛔ your public IP, NOT 0.0.0.0/0
ssh_public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... you@host"  # ⛔ your key
alert_email    = "you@example.com"      # ⛔ your email
budget_usd     = 5
instance_name  = "llm-travel-agent"
```

Generate an SSH key if needed:

```bash
ssh-keygen -t ed25519 -C "you@example" -f ~/.ssh/llm-travel-agent-aws
```

---

## Step 4: Terraform apply

> Source: `deploy/terraform/aws/README-aws.md` Step 4, `deploy/terraform/aws/floci/README.md`
>
> ⚠️ **FIRST `aws_instance` apply is UNVALIDATED.** floci could not validate
> `aws_instance` (floci lacks `DescribeInstanceTypes`; the provider errors before state).
> SG + keypair + wiring were floci-tested and de-risked; the instance's first true
> `terraform apply` is on real AWS. **Watch this step closely.** If it fails on the
> instance, check AMI availability in `ap-south-1` and t3.micro quota.

```bash
cd deploy/terraform/aws

terraform init
terraform plan        # Review EVERY resource. Confirm: NO aws_eip.
terraform apply       # Type "yes"
```

Outputs: `public_ip` + literal `ssh` command.

```bash
# Example output:
ssh -i ~/.ssh/llm-travel-agent-aws llm-agent@<PUBLIC_IP>
```

---

## Step 5: SSH + verify provisioning

> Source: `deploy/terraform/aws/README-aws.md` Step 5, `deploy/terraform/aws/main.tf` `user_data`

The `user_data` block in `main.tf` ran on boot: 2GB swapfile, Docker + Compose plugin,
`llm-agent` user created, docker group membership. Verify:

```bash
ssh -i ~/.ssh/llm-travel-agent-aws llm-agent@<PUBLIC_IP>

# Verify swap
free -h | grep Swap          # expect 2.0Gi

# Verify Docker
docker --version
docker compose version

# Verify docker group works
docker ps
```

---

## Step 6: SCP `.env` to the box

> Source: `docs/environment-secrets-contract.md`, N-M2.1
>
> ⛔ `.env` contains ALL provider keys (NIM, Groq, Gemini, OpenAI) + DB creds +
> Cloudflare Tunnel token. **Never committed.** Placed by Sid's hand only.

```bash
# From local machine:
scp -i ~/.ssh/llm-travel-agent-aws .env llm-agent@<PUBLIC_IP>:~/
```

---

## Step 7: Clone/sync the project

```bash
# Option A — git clone (recommended):
ssh -i ~/.ssh/llm-travel-agent-aws llm-agent@<PUBLIC_IP>
git clone <repo-url> ~/llm-travel-agent
cd ~/llm-travel-agent
cp ~/.env .

# Option B — scp if no remote:
# scp -i ~/.ssh/llm-travel-agent-aws -r . llm-agent@<PUBLIC_IP>:~/llm-travel-agent
```

---

## Step 8: Fix Caddyfile for compose network

> Source: `deploy/tunnel/README.md` "Compose network note"

The checked-in `deploy/Caddyfile` uses `127.0.0.1:8000` (written for caddy-on-host).
Inside compose, the api service is reachable as `api:8000`. **Fix before bringing the stack up.**

```bash
# On the box, inside ~/llm-travel-agent:
sed -i 's|127.0.0.1:8000|api:8000|g' deploy/Caddyfile
grep 'reverse_proxy.*api:8000' deploy/Caddyfile   # verify two hits
```

---

## Step 9: Bring up the full stack

> Source: `deploy/compose.prod.yml`, `deploy/compose.tunnel.yml`, `deploy/compose.alerting.yml`

All four overlays in one command:

```bash
# On the box. Placeholders must already be in .env (TRAVEL_DOMAIN, TUNNEL_TOKEN, etc.)
docker compose -f docker-compose.yml \
               -f deploy/compose.prod.yml \
               -f deploy/compose.tunnel.yml \
               -f deploy/compose.alerting.yml \
               up -d
```

Verify all containers are running:

```bash
docker compose ps
# Expect: api, postgres, caddy, cloudflared, alerting — all "Up"
```

---

## Step 10: Configure Cloudflare Tunnel

> Source: `deploy/tunnel/README.md`
>
> ⛔ Sid creates a named tunnel in Cloudflare Zero Trust and sets `TUNNEL_TOKEN` in `.env`.

**Named tunnel (stable):**

1. Cloudflare Zero Trust dashboard → Networks → Tunnels → Create tunnel
2. Name it `llm-travel-agent`, save, copy the `TUNNEL_TOKEN`
3. Public hostname: point domain → `caddy:80`
4. Set the token in `.env` on the box: `TUNNEL_TOKEN=<real-token>`

**Quick tunnel fallback (ephemeral, zero account, for pre-domain testing):**

```bash
# On the box (outside compose):
cloudflared tunnel --url http://localhost:80
# Prints: https://<random>.trycloudflare.com
```

---

## Step 11: Cloudflare DNS + cache-bypass (⛔ Sid)

> Source: plan §5 M2-T6

⛔ On Cloudflare: set DNS to PROXIED (orange cloud), add cache-bypass Page Rules or
Cache Rules for the SSE stream paths:

| Path | Rule |
|---|---|
| `/ask` | Bypass cache |
| `/jobs/*/events` | Bypass cache |

---

## Step 12: Verify public HTTPS + SSE survives ≥70s

> Source: M2-T1 (SSE keep-alive pings), `deploy/Caddyfile` (flush_interval -1, 300s read_timeout)

```bash
# Health check through the tunnel
curl -fsS https://<your-domain>/health

# SSE longevity test — idle stream must survive ≥70s
# (T1 keep-alive ping fires every 20s; we prove 3+ pings + cleanup)
time curl -fsSN -H "Accept: text/event-stream" \
  "https://<your-domain>/jobs/test-sse-longevity/events" &
sleep 70
kill %1 2>/dev/null || true
# PASS if curl exits cleanly (no premature connection close / 502)
```

---

## Step 13: Grafana — real scrape data screenshot

> Source: M2-T8, `deploy/compose.prod.yml`

```bash
# Grafana is at https://grafana.<your-domain> (basic auth per Caddyfile)
# ⛔ Sid: open in browser, log in, confirm Prometheus data source shows real scrape data
# Take screenshot — this kills F-006 (local empty-Grafana-window claim)
```

---

## Step 14: Backups + EXECUTED restore test

> Source: `deploy/backup/pg_backup.sh`, `deploy/backup/pg_restore_test.sh`, M2-T9

```bash
# On the box:
cd ~/llm-travel-agent

# Set DB creds
export PGHOST=localhost PGPORT=5432 PGUSER=<from .env> PGPASSWORD=<from .env> PGDATABASE=<from .env>

# Take a backup
deploy/backup/pg_backup.sh

# EXECUTE the restore test (restores into scratch DB, compares row counts)
deploy/backup/pg_restore_test.sh /var/backups/llm-travel-agent/pg_backup_*.dump
# Must show row-count match per table and "Restore test PASSED"

# Install cron for nightly backups
sudo cp deploy/backup/llm-backup.cron /etc/cron.d/llm-travel-agent-backup
```

---

## Step 15: UptimeRobot

> Source: M2-T10

⛔ Sid: add `https://<your-domain>/health` to UptimeRobot (free tier).
Optionally add a keyword monitor on a synthetic `/ask` response.

Take screenshot.

---

## Step 16: k6 capacity number

> Source: `deploy/loadtest/README.md`, `deploy/loadtest/travel-agent.js`, M2-T11

```bash
# From local machine; optional AUTH_TOKEN if AUTH_DISABLE is unset in prod
k6 run -e BASE_URL=https://<your-domain> deploy/loadtest/travel-agent.js

# Or with auth:
k6 run -e BASE_URL=https://<your-domain> -e AUTH_TOKEN=<token> deploy/loadtest/travel-agent.js
```

Document the p95 latency + throughput at saturation point. This supersedes the
T9.3(1b) local proof as the citable capacity evidence.

---

## Step 17: ZAP full + nuclei

> Source: `deploy/security/README.md`, `deploy/security/zap-full.sh`, `deploy/security/nuclei-scan.sh`, M2-T12

```bash
# From local machine or a jump box with Docker + nuclei installed:

# ZAP full scan (warn-only; report writes to plans/qa/)
ZAP_FULL_TARGET=https://<your-domain> ./deploy/security/zap-full.sh

# nuclei scan
NUCLEI_TARGET=https://<your-domain> ./deploy/security/nuclei-scan.sh
```

**Rule:** new HIGH-severity finding → STOP and report. ZAP tuning: add IGNORE/WARN
rules to `deploy/security/zap-full.tsv` (NEVER `.zap/rules.tsv` — that is CI-owned, frozen).

---

## Step 18: Telegram ops alerts

> Source: `deploy/alerting/README.md`, `deploy/alerting/notify_telegram.sh`, `deploy/alerting/watch.sh`, M2-T13

⛔ Sid: create a Telegram bot via @BotFather, capture `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
Set them in `.env` on the box.

```bash
# Dry-run first (no network):
./deploy/alerting/notify_telegram.sh --dry-run "deploy test"

# If not already up from Step 9, bring up the alerting overlay:
docker compose -f docker-compose.yml \
               -f deploy/compose.prod.yml \
               -f deploy/compose.alerting.yml \
               up -d

# Verify alerting sidecar is running:
docker logs llm-travel-alerting --tail 10
```

---

## Step 19: Mozilla Observatory + Qualys SSL

> Source: M2-T13

⛔ Sid clicks these against the live domain:
- [Mozilla Observatory](https://observatory.mozilla.org/) — paste the grade
- [Qualys SSL Labs](https://www.ssllabs.com/ssltest/) — paste the grade

---

## Step 20: Teardown (when done)

> Source: `deploy/terraform/aws/README-aws.md` Step 6

```bash
# On the box:
docker compose -f docker-compose.yml \
               -f deploy/compose.prod.yml \
               -f deploy/compose.tunnel.yml \
               -f deploy/compose.alerting.yml \
               down

# From local:
cd deploy/terraform/aws
terraform destroy   # Releases the public IP — no idle-EIP bill
```

---

## Artifact map

| Step(s) | Source README / artifact |
|---|---|
| 0–4, 20 | `deploy/terraform/aws/README-aws.md` + `deploy/terraform/aws/` module |
| 5 | `deploy/terraform/aws/main.tf` (`user_data`) + `deploy/provision.sh` |
| 6 | `docs/environment-secrets-contract.md` |
| 8 | `deploy/tunnel/README.md` (Caddy compose-network note) |
| 9 | `deploy/compose.prod.yml` + `deploy/compose.tunnel.yml` + `deploy/compose.alerting.yml` |
| 10–11 | `deploy/tunnel/README.md` |
| 12 | `deploy/Caddyfile` + M2-T1 SSE resilience (keep-alive pings) |
| 13 | `deploy/compose.prod.yml` (Grafana section) |
| 14 | `deploy/backup/pg_backup.sh` + `deploy/backup/pg_restore_test.sh` |
| 16 | `deploy/loadtest/README.md` + `deploy/loadtest/travel-agent.js` |
| 17 | `deploy/security/README.md` + `deploy/security/zap-full.sh` + `deploy/security/nuclei-scan.sh` |
| 18 | `deploy/alerting/README.md` + `deploy/alerting/notify_telegram.sh` + `deploy/alerting/watch.sh` |

---

> **End of runbook.** After Step 19, ALL M2 proof artifacts are collected. ⛔ STOP-C
> gates the transition to M3 (demo assets, resume pass, frontend at ⛔G6).
