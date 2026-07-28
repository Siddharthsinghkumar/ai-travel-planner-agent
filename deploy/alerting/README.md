# Ops Alerting — deploy/alerting/

## Architecture

Lightweight alpine+curl sidecar in a compose overlay:
- **Primary signal:** `curl -sf http://api:8000/health`
- **Secondary:** `docker inspect` `.State.Running` for api/postgres/caddy
- `postgres:16` and `caddy:2` have NO HEALTHCHECK → key off `.State.Running`, never `.State.Health`
- On transition-to-unhealthy → `notify_telegram.sh` via Bot API
- Debounced with state file (default 5 min) to avoid alert storms

## Configuration (⛔ Sid sets these)

| Env | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot API token from @BotFather |
| `TELEGRAM_CHAT_ID` | Target chat/user ID |
| `ALERTING_POLL_INTERVAL` | Seconds between health checks (default 15) |
| `ALERTING_DEBOUNCE_SEC` | Minimum seconds between alerts (default 300) |

## Live command (§6 M2-T13)

```bash
docker compose -f docker-compose.yml -f deploy/compose.prod.yml -f deploy/compose.alerting.yml up -d
```

## Dry-run (no network)

```bash
./deploy/alerting/notify_telegram.sh --dry-run "test message"
```

## Token security

All token values in this directory are **PLACEHOLDERS**. Real tokens go in `.env`
on the live box (never in any file under version control — N-M2.1).

## Alert message format

HTML parse_mode, includes:
- 🚨 header with stack name
- Failure detail (curl status or container state)
- UTC timestamp
