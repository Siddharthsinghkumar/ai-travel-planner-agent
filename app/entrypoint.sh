#!/usr/bin/env bash
set -euo pipefail

# Legacy note:
# - LLM_PREWARM is deprecated and this script is not the primary runtime path
#   for Docker images in this repository.
# - Active prewarm behavior is controlled by PLANNER_PREWARM in api/app.py lifespan.
if [ -n "${LLM_PREWARM:-}" ]; then
  echo "Warning: LLM_PREWARM is deprecated; use PLANNER_PREWARM in backend runtime config." >&2
fi

# Exec the CMD (gunicorn) as PID 1
exec "$@"
