#!/usr/bin/env bash
set -euo pipefail

# Stop every SGLang component we can start. Works whether you launched with
# ./start.sh (EAGLE), ./start-dspark.sh (DSpark), or ./start-mtp-8889.sh.
# Idempotent: anything that isn't running is skipped with a message.

CONTAINER_NAME="qwen3.8-27b-sglang"        # main engine (start.sh / start-dspark.sh)
MTP_CONTAINER_NAME="qwen3.8-27b-sglang-mtp" # MTP engine (start-mtp-8889.sh)
PID_FILE=".sglang.pid"
MTP_PID_FILE=".sglang-mtp.pid"
LOG_FILE=".sglang.log"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

stop_container() {
  local name="$1" pidfile="$2"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    echo "Container ${name} does not exist; nothing to stop"
    rm -f "${pidfile}"
    return
  fi
  if docker ps --format '{{.Names}}' | grep -qx "${name}"; then
    echo "Stopping container ${name}..."
    docker stop "${name}" >/dev/null
    echo "[$(date -Is)] container ${name} stopped" >> "${LOG_FILE}"
    echo "Stopped ${name}."
  else
    echo "Container ${name} is not running"
  fi
  rm -f "${pidfile}"
  # Leave the stopped container in place; the next start script removes it,
  # so `docker logs ${name}` stays available for post-mortem.
}

stop_container "${CONTAINER_NAME}" "${PID_FILE}"
stop_container "${MTP_CONTAINER_NAME}" "${MTP_PID_FILE}"
