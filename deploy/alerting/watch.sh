#!/usr/bin/env bash
# deploy/alerting/watch.sh — health poll + container liveness → Telegram alert
# Runs inside the alpine+curl alerting sidecar.
# Primary signal: curl -sf http://api:8000/health (the api container in compose network).
# Secondary: docker inspect .State.Running for api/postgres/caddy.
#   postgres:16 and caddy:2 have NO HEALTHCHECK — key off .State.Running only.
# Debounced via state file to avoid alert storms.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTIFY="$SCRIPT_DIR/notify_telegram.sh"

STATE_FILE="${STATE_FILE:-/tmp/watch_state.txt}"
DEBOUNCE_SEC="${DEBOUNCE_SEC:-300}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
HEALTH_URL="${HEALTH_URL:-http://api:8000/health}"
CONTAINERS_CHECK="${CONTAINERS_CHECK:-llm-travel-api llm-postgres llm-travel-caddy}"

NOW_EPOCH() { date +%s; }

last_alert=0
if [ -f "$STATE_FILE" ]; then
  last_alert=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
fi

echo "watch.sh starting — poll_interval=${POLL_INTERVAL}s, debounce=${DEBOUNCE_SEC}s, health=${HEALTH_URL}"

while true; do
  HEALTHY=false

  # Primary: curl /health
  if curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
    # Secondary: container liveness
    ALL_RUNNING=true
    for c in $CONTAINERS_CHECK; do
      if ! docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null | grep -q 'true'; then
        ALL_RUNNING=false
        break
      fi
    done

    if [ "$ALL_RUNNING" = true ]; then
      HEALTHY=true
    fi
  fi

  if [ "$HEALTHY" = false ]; then
    NOW=$(NOW_EPOCH)
    ELAPSED=$((NOW - last_alert))
    if [ "$ELAPSED" -ge "$DEBOUNCE_SEC" ]; then
      FAIL_DETAIL=""
      if ! curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
        FAIL_DETAIL="curl ${HEALTH_URL} failed"
      fi
      for c in $CONTAINERS_CHECK; do
        RUNNING=$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null || echo "unknown")
        if [ "$RUNNING" != "true" ]; then
          FAIL_DETAIL="${FAIL_DETAIL:+$FAIL_DETAIL; }${c}=${RUNNING}"
        fi
      done

      ALERT_MSG="<b>🚨 llm-travel-agent UNHEALTHY</b>%0A${FAIL_DETAIL}%0A<i>$(date -u +%Y-%m-%dT%H:%M:%SZ)</i>"
      "$NOTIFY" "$ALERT_MSG" 2>&1 || true
      echo "$NOW" > "$STATE_FILE"
      last_alert=$NOW
      echo "[$(date -u)] 🚨 ALERT SENT: $FAIL_DETAIL"
    else
      echo "[$(date -u)] unhealthy, debounced ($((DEBOUNCE_SEC - ELAPSED))s remaining)"
    fi
  fi

  sleep "$POLL_INTERVAL"
done
