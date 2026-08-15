#!/usr/bin/env bash
set -euo pipefail

# Stop the SGLang container started by start.sh (Qwen3.8-27B).

CONTAINER_NAME="qwen3.8-27b-sglang"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} does not exist; nothing to stop"
  rm -f "${PID_FILE}"
  exit 0
fi

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping container ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null
  echo "[$(date -Is)] container stopped" >> "${LOG_FILE}"
  echo "Stopped."
else
  echo "Container ${CONTAINER_NAME} is not running"
fi

rm -f "${PID_FILE}"

# Leave the stopped container in place; start.sh removes it on next launch,
# so `docker logs ${CONTAINER_NAME}` stays available for post-mortem.
