#!/usr/bin/env bash
set -euo pipefail

# Optional pre-start actions:
if [ "${LLM_PREWARM:-0}" = "1" ]; then
  echo "Prewarming LLM..."
  # Call local prewarm endpoint if your app exposes one
  curl -sS "http://127.0.0.1:${PORT:-8000}/internal/prewarm" || true
fi

# Exec the CMD (gunicorn) as PID 1
exec "$@"