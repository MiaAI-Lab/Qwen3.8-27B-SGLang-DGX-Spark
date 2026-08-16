#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B on SGLang (DGX Spark / GB10, aarch64)
#
# Recipe from the SGLang cookbook, DGX Spark cell:
#   https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
#
# Notes:
#   - Dense hybrid GDN vision-language model (48 linear-attention + 16
#     full-attention layers); SGLang serves it through the Qwen3-VL path,
#     so the vision tower is live.
#   - DGX Spark cell specifics: 128GB unified memory fits all three
#     checkpoints; the recipe uses 8192-token prefill chunks,
#     --mem-fraction-static 0.95 and --disable-prefill-cuda-graph.
#   - --attention-backend flashinfer is required on SM120/SM121
#     (trtllm_mha is SM100-only).
#   - Default checkpoint is RadixArk/Qwen3.8-27B-NVFP4 (W4A4 + FP8
#     projections, ~16.5GB weights). Override with QUANT=bf16|fp8|nvfp4:
#       QUANT=fp8 ./start.sh
#   - KV cache is explicitly FP8 (--kv-cache-dtype fp8_e4m3). The NVFP4
#     checkpoint declares kv_cache_quant_algo: FP8, so auto would apply
#     it anyway (with the checkpoint's calibration scales); the explicit
#     flag also forces fp8 KV when QUANT=bf16|fp8 (dynamic scales).
#     ~32.8 KB/token -> a 1M-token sequence needs ~33GB of KV.
#   - Thinking & tool calling (model card + cookbook section 3):
#       * Thinking mode is ON by default: the chat template defaults
#         enable_thinking=true and preserve_thinking=true (full
#         reasoning trace retained across turns; good for agents and
#         KV cache reuse). --reasoning-parser qwen3 surfaces the
#         <think> block as reasoning_content instead of inline text.
#         Disable per request via chat_template_kwargs
#         {"enable_thinking": false} (then prefer temperature=0.7,
#         top_p=0.8, presence_penalty=1.5 per card). Reasoning depth
#         per request: reasoning_effort=xhigh|medium|low (xhigh
#         default).
#       * --sampling-defaults model (SGLang default, pinned):
#         recommended sampling params come from the checkpoint's
#         generation_config.json; thinking mode wants temperature=1.0,
#         top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0.
#       * Tool calling: --tool-call-parser qwen3_coder decodes the
#         template's <tool_call><function=...>/<parameter=...>
#         payload into structured tool_calls (hermes expects a
#         different JSON shape and would never parse). SGLang needs
#         no vLLM-style --enable-auto-tool-choice; just send `tools`
#         in the request.
#   - Speed: speculative decoding via the checkpoint's own MTP head
#     (the cookbook's DGX Spark cell, EAGLE 3 steps / topk 1 /
#     4 draft tokens). Measured on this box, single stream, 400 tok
#     (bench.sh, essay prompt):
#       MTP 3/1/4 + NVFP4: 16.9 tok/s thinking / 21.0 non-thinking
#       DSpark    + NVFP4: 16.9 tok/s thinking / 16.2 non-thinking
#       DSpark    + FP8:   11.4 tok/s thinking / 11.5 non-thinking
#     (no-spec baseline not measured; accept-rate MTP 0.23-0.52 vs
#     DSpark 0.09-0.26, accept-len ~2.2 vs ~1.9). MTP also verifies
#     only 4 tokens/step vs DSpark's 8 -> cheaper steps, and needs no
#     second checkpoint. MTP with FlashInfer needs a build newer than
#     0.6.15.post1; if spec decode errors at boot, fall back to
#     --attention-backend triton.
#     Alternative: DSpark (cookbook's trained drafter, separate ~2.7GB
#     checkpoint that auto-downloads into the same HF cache):
#       --speculative-algorithm DSPARK \
#       --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
#       --speculative-dspark-block-size 7 \
#       --speculative-draft-model-quantization unquant \
#     plus --max-mamba-cache-size = concurrency x 12 (see below). Block size 7
#     (verify width 8 incl. the bonus token) and the unquantized BF16
#     draft are from the DSpark model card's serving recipe; leaving
#     --speculative-dspark-block-size unset made no measurable
#     difference here. NOTE: the draft was trained against the FP8
#     target, but QUANT=fp8 did NOT lift acceptance on this box
#     (~1.9 accept len either way) while costing ~30% decode speed
#     from the larger weights -- stick to NVFP4. The card also uses
#     --attention-backend fa3 (SM90-only; we keep flashinfer for
#     SM121) and --mamba-scheduler-strategy extra_buffer (older alias
#     of the mamba radix strategy we already set).
#   - GDN state pool sizing (throughput): the --mamba-full-memory-ratio
#     default 0.9 over-provisions the KV pool and silently clamps
#     concurrency. We set --mamba-full-memory-ratio 4.21 (tuned value)
#     and pin the pool explicitly as the authoritative override:
#       --max-mamba-cache-size = concurrency x (S + D),
#     with --mamba-ssm-dtype bfloat16 (state slot 78.4MB; the fp32
#     default 153.9MB/slot would break the math).
#     with S=4 (--mamba-radix-cache-strategy extra_buffer_lazy, state
#     slots per request, no accuracy cost) and D=4 (MTP's
#     speculative_num_draft_tokens) -> 8 slots/request, so the default
#     10 requests => 80 slots (~6.3GB at 78.4MB/slot). Set
#     MAX_CONCURRENT_REQUESTS in .env to change.
#     DSpark instead drafts D=8 tokens/step: 12 slots/request ->
#     --max-mamba-cache-size = concurrency x 12 (see below).
#     --max-running-requests pins the scheduler cap to match
#     (speculative decoding otherwise resets it to 48). After boot,
#     check the max_running_requests line in .sglang.log.
#   - Context: YARN=0|1 in .env, plus CONTEXT_LENGTH (range 262144..1000000).
#     YaRN (rope scaling) is applied when CONTEXT_LENGTH exceeds 262144
#     and either YARN=1 or the length is exactly 1000000 (1M works out
#     of the box; YARN=1 is needed for e.g. 500K). YaRN factor derived:
#     round(CONTEXT_LENGTH/262144) -> 524288 => 2.0, 1000000 => 4.0
#     (the model card's validated values). The rope override lives
#     under text_config in the JSON; SGLang also needs
#     SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 or it ignores the
#     longer --context-length with the "User-specified context_length
#     (...) is greater than the derived context_length" warning and
#     stays at 262K. The override is NOT compatible with DSpark on this
#     build: the draft ModelConfig inherits it, injecting a text_config
#     dict into the draft's flat config and crashing transformers' rope
#     validator (AttributeError: ... 'max_position_embeddings'). If you
#     switch to DSpark, keep YARN=0 and CONTEXT_LENGTH=262144.

# Load optional .env overrides (QUANT, YARN, CONTEXT_LENGTH, MAX_CONCURRENT_REQUESTS).
# Shell env vars already set win; .env fills the gaps; defaults apply last.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  # `|| [[ -n "${key}" ]]` so a final line without a trailing newline is not dropped.
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    # Tolerate CRLF files and surrounding whitespace, and skip indented comments —
    # otherwise `${!key}` below would be an indirect expansion on an invalid name.
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

QUANT="${QUANT:-nvfp4}"
case "${QUANT}" in
  bf16)  MODEL_ID="Qwen/Qwen3.8-27B" ;;
  fp8)   MODEL_ID="Qwen/Qwen3.8-27B-FP8" ;;
  nvfp4) MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4" ;;
  *) echo "Unknown QUANT '${QUANT}' (use bf16|fp8|nvfp4)"; exit 1 ;;
esac

# Context length: any value from native up to the model's validated 1M.
# YaRN controlled explicitly by YARN (0|1), plus auto-on at exactly 1M.
YARN="${YARN:-0}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-10}"
if (( CONTEXT_LENGTH < 262144 || CONTEXT_LENGTH > 1000000 )); then
  echo "CONTEXT_LENGTH '${CONTEXT_LENGTH}' unsupported (use 262144..1000000)"; exit 1
fi
case "${YARN}" in
  0|1) : ;;
  *) echo "YARN must be 0 or 1, got '${YARN}'"; exit 1 ;;
esac
NEED_YARN=0
if (( CONTEXT_LENGTH > 262144 )); then
  if [[ "${YARN}" == "1" ]] || [[ "${CONTEXT_LENGTH}" == "1000000" ]]; then
    NEED_YARN=1
  else
    echo "warning: CONTEXT_LENGTH=${CONTEXT_LENGTH} needs YaRN but YARN=0; build will stay at 262K"
  fi
fi
if (( NEED_YARN )); then
  # Model card's YaRN recipe: rope_parameters override under text_config.
  # Factor derived from CONTEXT_LENGTH (round(len/262144)); the card
  # validates 2.0 for 524288 and 4.0 for 1M.
  YARN_FACTOR="$(awk -v n="${CONTEXT_LENGTH}" 'BEGIN{printf "%.0f", n/262144}')"
  (( YARN_FACTOR < 1 )) && YARN_FACTOR=1
  YARN_OVERRIDE=$(printf '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": %s, "original_max_position_embeddings": 262144}}}' "${YARN_FACTOR}")
  CONTEXT_ARGS=(--json-model-override-args "${YARN_OVERRIDE}" --context-length "${CONTEXT_LENGTH}")
  ALLOW_LONGER_ARGS=(-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)
  YARN_SUFFIX=" (YaRN factor ${YARN_FACTOR})"
else
  CONTEXT_ARGS=(--context-length "${CONTEXT_LENGTH}")
  ALLOW_LONGER_ARGS=()
  YARN_SUFFIX=""
fi


# GDN state pool: S=4 (extra_buffer_lazy) + D=4 (MTP draft tokens) = 8
# slots per request; DSpark would be 12. Sized from MAX_CONCURRENT_REQUESTS.
MAMBA_SLOTS_PER_REQ=8
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ ))

SERVED_MODEL_NAME="qwen3.8-27b-sglang"
IMAGE="lmsysorg/sglang:qwen38-27b"
CONTAINER_NAME="qwen3.8-27b-sglang"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

# Pick up HF_TOKEN from ~/.bashrc (defined without `export` there) so the
# container gets authenticated Hub access (higher rate limits, faster downloads).
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  # Accept `export HF_TOKEN=…` (the idiomatic form: a bare assignment in .bashrc is
  # not exported to child processes) as well as a plain `HF_TOKEN=…`.
  HF_TOKEN="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\2/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting SGLang container for ${MODEL_ID} (${QUANT})"
echo "Context: ${CONTEXT_LENGTH} tokens${YARN_SUFFIX:-}"
echo "Max concurrent requests: ${MAX_CONCURRENT_REQUESTS} (mamba pool ${MAMBA_CACHE_SIZE} slots)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  --shm-size 32g \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${ALLOW_LONGER_ARGS[@]}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_DIR}:/root/.triton" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  "${CONTEXT_ARGS[@]}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --host "${HOST}" \
  --port "${PORT}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Stream the SGLang startup log to the terminal AND record it in .sglang.log.
# $! tracks the pipeline group; killing it on exit SIGPIPEs docker logs.
# The terminal copy is filtered to drop the per-layer "Enabled fused
# SiLU+mul+FP4-quant for dense MLP down_proj input." notices (they fire once
# per MLP layer); .sglang.log keeps the complete, unfiltered stream.
docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" | grep --line-buffered -v "Enabled fused SiLU+mul+FP4-quant for dense MLP down_proj input" &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  # The log itself is streaming above; only a light heartbeat every ~30s.
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "SGLang is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Anthropic-compatible: http://${HOST}:${PORT}/v1/messages (no /v1 suffix in ANTHROPIC_BASE_URL)"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Thinking: ON by default (disable per request: chat_template_kwargs {\"enable_thinking\": false})"

echo "SGLang is ready and responding; shell is now free."
