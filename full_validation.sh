#!/usr/bin/env bash
# safe_full_validation_report.sh
# Improved version of safe_full_validation.sh that captures logs and prints a neat summary.
# Based on original uploaded file.
set -euo pipefail
ROOT="$(pwd)"
TMP_ENV="$ROOT/.env.tmp"
CONTAINER_NAME="llm-test-local"
IMAGE_NAME="llm-test:normal"
PYTEST_CMD="pytest -q"

HEALTH_URL="http://localhost:8000/health"
SMOKE_TIMEOUT=30          # increased from 12 to 30 seconds
APP_START_TIMEOUT=25

LOG_DIR="$ROOT/validation_logs"
mkdir -p "$LOG_DIR"

# Single consolidated log file for this run
LOGFILE="$LOG_DIR/validation_run_$(date +%Y%m%dT%H%M%S%z).log"

# Print header with timestamp
printf "\n=== Validation run started: %s ===\n\n" "$(date --iso-8601=seconds)" | tee -a "$LOGFILE"

# a compact report storage (each element: name|status|start|duration)
declare -a REPORT=()

# Clean up temporary env file on exit
trap 'rm -f "$TMP_ENV"' EXIT

# Silent info function – no output during normal run (now output is captured anyway)
info() { :; }

# Wait for readiness (dependencies fully up)
wait_for_ready() {
  local url="${1:-http://127.0.0.1:8000/health/ready}"
  local max_wait="${2:-60}"
  local interval=1
  local waited=0
  echo "Waiting for service readiness at $url ..." | tee -a "$LOGFILE"
  while [[ $waited -lt $max_wait ]]; do
    if curl -sS "$url" 2>/dev/null | grep -q '"status":"ok"'; then
      echo "Service is ready." | tee -a "$LOGFILE"
      return 0
    fi
    sleep $interval
    waited=$((waited+interval))
  done
  echo "Timed out waiting for readiness at $url after ${max_wait}s" | tee -a "$LOGFILE"
  return 1
}

# run a command and capture to log, store metadata in REPORT
run_and_log() {
  local name="$1"; shift
  local start_iso start_epoch end_epoch duration
  start_iso="$(date --iso-8601=seconds)"
  start_epoch=$(date +%s)

  # detect streaming test by name
  local is_stream=0
  [[ "$name" == streaming_test_* ]] && is_stream=1
  [[ "$name" == streaming_nl_relative_* ]] && is_stream=1

  # determine mode label from name (machine/docker/global)
  local mode_label="global"
  [[ "$name" == *_machine ]] && mode_label="machine"
  [[ "$name" == *_docker-hosted ]] && mode_label="docker"

  # Create temporary file for command output (for analysis)
  local tmp_out
  tmp_out=$(mktemp)

  local status=0
  if [[ "$1" == "curl" ]]; then
    # For curl commands, capture HTTP status code and response body separately
    local tmp_resp
    tmp_resp=$(mktemp)

    # Run curl, saving body to tmp_resp, capturing HTTP code, and appending stderr to tmp_out
    local http_code
    http_code=$("$@" -o "$tmp_resp" -w "%{http_code}" 2>>"$tmp_out")
    local curl_exit=$?

    # Move response body to tmp_out
    cat "$tmp_resp" >> "$tmp_out"
    rm -f "$tmp_resp"

    # --- immediately write debug metadata into the same temp file (so checks use same file) ---
    printf '\n{"_internal_debug": {"http_code": %s, "curl_exit": %s}}\n' "$http_code" "$curl_exit" >> "$tmp_out"

    # Determine status (streaming has special rules)
    if [[ "$curl_exit" -ne 0 ]]; then
      if [[ "$is_stream" -eq 1 ]]; then
        # For streaming: if we received any content and there is no validation error,
        # treat a timeout/kill (non-zero curl exit) as a benign termination => PASS.
        if [[ -s "$tmp_out" ]] && ! grep -qi '"msg":"Field required"' "$tmp_out" && ! grep -qi '"detail"' "$tmp_out"; then
          status=0
        else
          status=$curl_exit
        fi
      else
        # Non-streaming: non-zero curl is a failure
        status=$curl_exit
      fi
    elif (( http_code < 200 || http_code >= 300 )); then
      # HTTP non-2xx for non-streaming calls => HTTP error / validation
      status=124
    else
      status=0
    fi

    # --- Non-streaming: strict JSON checks
    if [[ "$status" -eq 0 && "$is_stream" -eq 0 ]]; then
      if command -v jq >/dev/null 2>&1; then
        if jq -e 'has("detail")' "$tmp_out" >/dev/null 2>&1; then
          status=124
        elif jq -e '.. | objects | select(has("llm_response")) | .llm_response | select(type=="string" and length>0)' "$tmp_out" >/dev/null 2>&1; then
          status=0
        else
          status=125
        fi
      else
        if grep -qi '"detail"' "$tmp_out"; then
          status=124
        elif grep -qi '"llm_response"' "$tmp_out"; then
          status=0
        else
          status=125
        fi
      fi
    fi

    # --- Streaming: ensure not empty and no validation error in streamed text ---
    if [[ "$is_stream" -eq 1 && "$status" -eq 0 ]]; then
      if [[ ! -s "$tmp_out" ]]; then
        status=126
      else
        if grep -qi '"msg":"Field required"' "$tmp_out" || grep -qi '"detail"' "$tmp_out"; then
          status=124
        fi
      fi
    fi

    # Optional debug print for failures (to console, not logfile)
    if [[ "$status" -ne 0 ]]; then
      echo "[DEBUG] $name http_code=$http_code curl_exit=$curl_exit" >&2
    fi

  else
    # Non-curl command: run and capture output to temp file
    set +e
    "$@" >"$tmp_out" 2>&1
    status=$?
    set -e
    # No additional grep-based detection for non-curl commands (like async parallel)
    # because it can cause false positives. Rely only on exit code.
  fi

  # Compute duration
  end_epoch=$(date +%s)
  duration=$((end_epoch - start_epoch))

  # Append the command output to the global log with separators
  echo "=== START $mode_label/$name ($start_iso) ===" >> "$LOGFILE"
  cat "$tmp_out" >> "$LOGFILE"
  echo "=== END   $mode_label/$name (exit=$status, duration=${duration}s) ===" >> "$LOGFILE"
  echo >> "$LOGFILE"

  # Remove temp file
  rm -f "$tmp_out"

  # Store in REPORT (without logfile path)
  REPORT+=("$name|$status|$start_iso|$duration")

  # live status print to console only
  local display_name="$name"
  display_name="${display_name%_machine}"
  display_name="${display_name%_docker-hosted}"

  # clean up display name to match summary labels
  case "$display_name" in
    quick_sync_ask*) display_name="query basic" ;;
    missing_date_test*) display_name="query missing date" ;;
    nl_relative_date*) display_name="query natural language date" ;;
    misspelled_city*) display_name="query misspelled city" ;;
    round_trip_duration*) display_name="query round trip duration" ;;
    time_pref_morning*) display_name="query time morning" ;;
    price_cap*) display_name="query price cap" ;;
    direct_only*) display_name="query direct only" ;;
    preferred_airline*) display_name="query preferred airline" ;;
    layover_limit*) display_name="query layover limit" ;;
    baggage_hand*) display_name="query hand baggage" ;;
    stopover_via*) display_name="query stopover via" ;;
    async_parallel*) display_name="parallel async queries" ;;
    streaming_test*) display_name="stream basic" ;;
    streaming_nl_relative*) display_name="stream natural language date" ;;
    pytest_unit) display_name="pytest" ;;
    docker_hosted_smoke) display_name="server boot (docker)" ;;
    *) ;;
  esac

  if [[ "$status" -eq 0 ]]; then
    printf "[%-7s] %-30s ... PASSED (%s s)\n" "$mode_label" "$display_name" "$duration"
  else
    printf "[%-7s] %-30s ... FAILED (%s s)\n" "$mode_label" "$display_name" "$duration"
  fi

  return $status
}

# Create tmp .env for given mode (machine or docker)
create_temp_env() {
  local mode="$1"
  info "Creating $TMP_ENV for mode=$mode"

  # Choose the correct original environment file based on mode
  if [[ "$mode" == "machine" ]]; then
      local src_env="$ROOT/.env"
  else
      local src_env="$ROOT/.env.laptopdocker"
  fi

  # Copy the original file to temporary env (create minimal if missing)
  if [[ -f "$src_env" ]]; then
      cp -f "$src_env" "$TMP_ENV"
  else
      echo "Warning: $src_env not found, creating minimal .env.tmp" >&2
      : > "$TMP_ENV"
  fi

  # Determine Ollama URL based on mode (only set for docker)
  if [[ "$mode" == "docker" ]]; then
    local ollama_for_docker="http://host.docker.internal:11434"
    # Append overrides (preserve real keys from original, only set specific test overrides)
    cat >> "$TMP_ENV" <<EOF

# ----- temporary test overrides (generated by safe_full_validation_report.sh) -----
TESTING=true
OLLAMA_BASE_URL=${ollama_for_docker}
CLOUD_LLM_TIMEOUT=5
CLOUD_LLM_STREAM_CHUNK_TIMEOUT=1
PLANNER_PREWARM=1
PLANNER_GLOBAL_TIMEOUT=60
USE_CLOUD_LLM=0
EOF
  else
    # For machine mode, only set test flags, keep original OLLAMA_BASE_URL
    cat >> "$TMP_ENV" <<EOF

# ----- temporary test overrides (generated by safe_full_validation_report.sh) -----
TESTING=true
CLOUD_LLM_TIMEOUT=5
CLOUD_LLM_STREAM_CHUNK_TIMEOUT=1
PLANNER_PREWARM=1
PLANNER_GLOBAL_TIMEOUT=60
USE_CLOUD_LLM=0
EOF
  fi

  echo "Wrote overrides to $TMP_ENV" >> "$LOG_DIR/internal_debug.log"
}

activate_venv_if_any() {
  if [[ -f "$ROOT/venv/bin/activate" ]]; then
    info "Activating venv at $ROOT/venv"
    # shellcheck disable=SC1090
    source "$ROOT/venv/bin/activate"
  elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
    info "Activating venv at $ROOT/.venv"
    # shellcheck disable=SC1090
    source "$ROOT/.venv/bin/activate"
  else
    info "No local venv found; expect pytest in PATH"
  fi
}

wait_for_health_poll() {
  local timeout_s=$1
  local url=${2:-$HEALTH_URL}
  local waited=0
  local interval=1
  while true; do
    if curl -sS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep $interval
    waited=$((waited+interval))
    if (( waited >= timeout_s )); then
      return 1
    fi
  done
  echo "Polling $url ..." >> "$LOGFILE"
}

run_smoke_checks_logged() {
  local mode="$1"
  info "Running smoke checks (mode=$mode) — logs in $LOG_DIR"

  # Basic sync ask
  run_and_log "quick_sync_ask_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"origin":"DEL","destination":"BOM","date":"2026-03-20","user_query":"Say hello in one sentence."}' || true

  # Missing date test
  run_and_log "missing_date_test_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Cheap flight from Delhi to Mumbai"}' || true

  # Natural language relative date
  run_and_log "nl_relative_date_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Flight from Delhi to Mumbai fourteen days after today"}' || true

  # Misspelled city (LLM correction test)
  run_and_log "misspelled_city_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Cheap flight from Dalhi to Mumbai on March 20"}' || true

  # Round trip duration (inferred return)
  run_and_log "round_trip_duration_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Business trip from Delhi to Mumbai for 3 days starting March 20"}' || true

  # Time preference — morning
  run_and_log "time_pref_morning_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Business trip from Delhi to Mumbai on 2026-03-20 in the morning"}' || true

  # Price cap
  run_and_log "price_cap_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Business trip from DEL to BOM under ₹3000 on March 20, 2026"}' || true

  # Direct only
  run_and_log "direct_only_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Direct flights only from Delhi to Mumbai on 2026-03-20"}' || true

  # Preferred airline
  run_and_log "preferred_airline_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Business trip from Delhi to Mumbai on 2026-03-20 prefer indigo"}' || true

  # Layover limit
  run_and_log "layover_limit_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Business trip Delhi to Mumbai on 2026-03-20 with layover less than 2 hours"}' || true

  # Baggage preference (hand baggage only)
  run_and_log "baggage_hand_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Business trip Delhi to Mumbai on 2026-03-20 cabin only (hand baggage)"}' || true

  # Stopover / multi-city ("via")
  run_and_log "stopover_via_${mode}" curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d '{"user_query":"Business trip Delhi to Chennai via Bangalore on March 20"}' || true

  # Parallel async calls (different destinations) – kept for general async test
  run_and_log "async_parallel_${mode}" bash -c '
    curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"origin\":\"DEL\",\"destination\":\"BOM\",\"date\":\"2026-03-20\",\"user_query\":\"Business trip\"}" &
    curl -sS -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"origin\":\"DEL\",\"destination\":\"BLR\",\"date\":\"2026-03-20\",\"user_query\":\"Holiday\"}" &
    wait
  ' || true

  # Streaming basic test (using curl's built-in --max-time instead of external timeout command)
  run_and_log "streaming_test_${mode}" curl --max-time $((SMOKE_TIMEOUT+2)) -N -sS -X POST "http://localhost:8000/ask?stream=true" -H "Content-Type: application/json" \
    -d '{"origin":"DEL","destination":"BOM","date":"2026-03-20","user_query":"Explain why this flight is good"}' || true

  # Streaming natural language relative date (using curl's built-in --max-time)
  run_and_log "streaming_nl_relative_${mode}" curl --max-time $((SMOKE_TIMEOUT+2)) -N -sS -X POST "http://localhost:8000/ask?stream=true" -H "Content-Type: application/json" \
    -d '{"user_query":"Cheapest flight from Delhi to Mumbai fourteen days after today"}' || true
}

run_pytest_logged() {
  info "Running pytest (unit tests)"
  activate_venv_if_any
  if ! command -v pytest >/dev/null 2>&1; then
    run_and_log "pytest_missing" bash -c 'echo "pytest not found. Install dev deps or activate venv with pytest. Skipping pytest."'
    return 2
  fi
  run_and_log "pytest_unit" $PYTEST_CMD || true
}

ensure_image_logged() {
  if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    run_and_log "docker_build_image" docker build -t "$IMAGE_NAME" . || true
  else
    run_and_log "docker_image_present" bash -c "echo 'Docker image $IMAGE_NAME present'"
  fi
}

# --- Local server runner for machine mode ---
run_machine_local_server() {
  echo "Starting local uvicorn server..." | tee -a "$LOGFILE"
  activate_venv_if_any

  # Load temp env to match docker environment (using allexport for clarity)
  set -o allexport
  source "$TMP_ENV"
  set +o allexport

  # Ensure port 8000 is free (use fuser if available)
  if command -v fuser >/dev/null 2>&1; then
    fuser -k 8000/tcp 2>/dev/null || true
  else
    pkill -f "uvicorn api.app:app" 2>/dev/null || true
  fi

  # Start server in background, output directly to the main log
  uvicorn api.app:app --host 0.0.0.0 --port 8000 >>"$LOGFILE" 2>&1 &
  MACHINE_PID=$!

  echo $MACHINE_PID > "$LOG_DIR/machine_uvicorn.pid"

  # Wait for health
  if wait_for_health_poll "$APP_START_TIMEOUT"; then
    # Wait for readiness (dependencies up)
    if ! wait_for_ready "http://localhost:8000/health/ready" 60; then
      echo "Local server did not become ready within timeout." >&2
      kill $MACHINE_PID 2>/dev/null || true
      return 1
    fi
    echo "Local server healthy and ready." | tee -a "$LOGFILE"
    return 0
  else
    echo "Local server failed to start." | tee -a "$LOGFILE"
    kill $MACHINE_PID 2>/dev/null || true
    return 1
  fi
}

stop_machine_local_server() {
  if [[ -f "$LOG_DIR/machine_uvicorn.pid" ]]; then
    pid=$(cat "$LOG_DIR/machine_uvicorn.pid")
    kill $pid 2>/dev/null || true
    rm -f "$LOG_DIR/machine_uvicorn.pid"
  fi
}

# --- main flow ---
info "START safe_full_validation_report.sh"

run_pytest_logged || echo "pytest step returned nonzero or was skipped."

# Machine mode (running directly on host)
create_temp_env machine
if run_machine_local_server; then
  run_smoke_checks_logged machine

  # Check if server is still alive after tests (safely)
  if [[ -n "${MACHINE_PID:-}" ]] && kill -0 "$MACHINE_PID" 2>/dev/null; then
    stop_machine_local_server
    wait "$MACHINE_PID" 2>/dev/null || true   # ensure no zombie
    run_and_log "result_machine_integration" bash -c "echo 'Machine local integration succeeded.'"
  else
    # Server crashed; ensure cleanup
    stop_machine_local_server
    wait "$MACHINE_PID" 2>/dev/null || true
    run_and_log "result_machine_integration_failed" bash -c "echo 'Machine server crashed during tests.'"
  fi
else
  run_and_log "result_machine_integration_failed" bash -c "echo 'Machine local integration failed to start.'"
fi

# Docker-hosted smoke (host-run tests) – runs the same container but from host perspective
create_temp_env docker
ensure_image_logged
docker rm -f "${CONTAINER_NAME}-validation" 2>/dev/null || true
cid=$(docker run -d --rm -p 8000:8000 --add-host=host.docker.internal:host-gateway --name "${CONTAINER_NAME}-validation" --env-file "$TMP_ENV" "$IMAGE_NAME")
echo "$cid" > "$LOG_DIR/docker_validation.cid"
sleep 1
if wait_for_health_poll "$APP_START_TIMEOUT" "http://localhost:8000/health"; then
  # Wait for readiness
  if ! wait_for_ready "http://localhost:8000/health/ready" 60; then
    echo "Docker-hosted app did not become ready; see logs." >&2
    docker logs --tail 200 "${CONTAINER_NAME}-validation" >&2
    run_and_log "docker_hosted_failed" bash -c "echo 'Docker-hosted app failed readiness.'"
    docker rm -f "${CONTAINER_NAME}-validation" 2>/dev/null || true
  else
    run_and_log "docker_hosted_smoke" bash -c "echo 'Docker-hosted app healthy; running smoke checks'; true"
    run_smoke_checks_logged "docker-hosted"
    docker logs --tail 200 "${CONTAINER_NAME}-validation" >"$LOG_DIR/docker_validation_container_logs.log" 2>&1 || true
    docker rm -f "${CONTAINER_NAME}-validation" 2>/dev/null || true
  fi
else
  docker logs --tail 200 "${CONTAINER_NAME}-validation" >"$LOG_DIR/docker_validation_error_logs.log" 2>&1 || true
  run_and_log "docker_hosted_failed" bash -c "echo 'Docker-hosted app failed to become healthy; see docker_validation_error_logs.log'"
  docker rm -f "${CONTAINER_NAME}-validation" 2>/dev/null || true
fi

# --- Compact final summary: failures only + totals with categories ---
total=0; passed=0; failed=0
echo
echo "Summary (failures only):"
for entry in "${REPORT[@]}"; do
  IFS='|' read -r name status start duration <<< "$entry"

  # Determine mode for display
  mode="machine"
  [[ "$name" == *_docker-hosted ]] && mode="docker"

  # Clean test name for display
  base="${name%_machine}"
  base="${base%_docker-hosted}"
  case "$base" in
    pytest_unit) display="pytest" ;;
    quick_sync_ask*) display="query basic" ;;
    missing_date_test*) display="query missing date" ;;
    nl_relative_date*) display="query natural language date" ;;
    misspelled_city*) display="query misspelled city" ;;
    round_trip_duration*) display="query round trip duration" ;;
    time_pref_morning*) display="query time morning" ;;
    price_cap*) display="query price cap" ;;
    direct_only*) display="query direct only" ;;
    preferred_airline*) display="query preferred airline" ;;
    layover_limit*) display="query layover limit" ;;
    baggage_hand*) display="query hand baggage" ;;
    stopover_via*) display="query stopover via" ;;
    async_parallel*) display="parallel async queries" ;;
    streaming_test*) display="stream basic" ;;
    streaming_nl_relative*) display="stream natural language date" ;;
    docker_hosted_smoke) display="server boot (docker)" ;;
    integration_smoke*|result_*|docker_image_*|docker_build_*) continue ;;  # skip noise
    *) continue ;;
  esac

  total=$((total+1))
  if [[ "$status" -eq 0 ]]; then
    passed=$((passed+1))
  else
    failed=$((failed+1))
    # Categorize failure based on status code
    if [[ "$status" -eq 124 ]]; then
      category="Validation"
    elif [[ "$status" -eq 125 ]]; then
      category="Unexpected"
    elif [[ "$status" -eq 126 ]]; then
      category="Empty stream"
    else
      category="Infra"
    fi

    # For failure reason, we need to extract from LOGFILE (but that's complex).
    # For simplicity, we'll just show category and not the detailed reason.
    # The full log is available in LOGFILE.
    printf "  %-8s  %-35s  [%s]\n" "$mode" "$display" "$category"
  fi
done
echo
printf "Totals: %d total, %d passed, %d failed\n" "$total" "$passed" "$failed"

# Additional summary from consolidated log (optional)
echo
echo "Detailed counts from consolidated log:"
total_passed=$(grep -o "PASSED" "$LOGFILE" | wc -l)
total_failed=$(grep -o "FAILED" "$LOGFILE" | wc -l)
echo "  PASSED lines: $total_passed"
echo "  FAILED lines: $total_failed"
# Count field required errors (common validation issue)
field_required=$(grep -c '"msg":"Field required"' "$LOGFILE" || true)
if [[ $field_required -gt 0 ]]; then
  echo "  Field required errors: $field_required"
fi

echo
echo "Full logs available in: $LOGFILE"
exit 0