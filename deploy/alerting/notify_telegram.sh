#!/usr/bin/env bash
# deploy/alerting/notify_telegram.sh — send Telegram alert via Bot API
# Usage: ./notify_telegram.sh "message text"
#   --dry-run: print the JSON payload, NO network call
#
# Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (⛔ Sid sets these; NEVER committed)

set -euo pipefail

DRY_RUN=false
MESSAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      MESSAGE="$1"
      shift
      ;;
  esac
done

if [ -z "$MESSAGE" ]; then
  echo "ERROR: message required" >&2
  echo "Usage: $0 [--dry-run] \"message\"" >&2
  exit 1
fi

TOKEN="${TELEGRAM_BOT_TOKEN:-PLACEHOLDER_BOT_TOKEN}"
CHAT_ID="${TELEGRAM_CHAT_ID:-PLACEHOLDER_CHAT_ID}"

PAYLOAD=$(cat <<PAYLOAD_END
{
  "chat_id": "${CHAT_ID}",
  "text": "${MESSAGE}",
  "parse_mode": "HTML",
  "disable_web_page_preview": true
}
PAYLOAD_END
)

if [ "$DRY_RUN" = true ]; then
  echo "=== DRY RUN — no network call ==="
  echo "URL:  https://api.telegram.org/bot${TOKEN}/sendMessage"
  echo "Payload:"
  echo "$PAYLOAD" | python3 -m json.tool 2>/dev/null || echo "$PAYLOAD"
  echo "=== END DRY RUN ==="
  exit 0
fi

curl -sf -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" \
  > /dev/null
