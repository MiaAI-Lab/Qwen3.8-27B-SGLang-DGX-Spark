#!/usr/bin/env bash
set -euo pipefail

# start-dspark.sh — run the Qwen3.8-27B SGLang service with **DSpark**
# speculative decoding instead of the EAGLE/MTP draft hardcoded in
# start.sh. Same model, same image, same everything else.
#
# Why a wrapper and not a copy: start.sh is the single source of truth
# for the launch config; DSpark only changes the speculative-decode
# flag stack. This script sets that stack as EXTRA_ARGS (a shell env var,
# which start.sh prioritizes over .env), then hands off to start.sh.
# One place to maintain flags, no drift between the two scripts.
#
# Notes:
#   - The stack is the published-and-reproduced GB10 DSpark config
#     (hasso5703/dgx-spark-qwen38, same pinned image digest febfb971):
#     block 7 / gamma from speculative-num-draft-tokens 8 (gamma+1, must
#     override start.sh's hardcoded 4), bf16 (unquant) draft, mem
#     fraction 0.90 (takes effect on next boot; larger KV pool), torch
#     compile + decode-cuda-graph caps 4, two continuous decode steps.
#     caps 4, two continuous decode steps.
#   - DSpark is fast on code / math / tool calls, SLOWER than MTP on
#     free-form prose. Switch back to MTP: ./stop.sh && ./start.sh
#   - YaRN / CONTEXT_LENGTH > 262144 stays incompatible with DSpark on
#     this build (draft config inherits the rope override and crashes);
#     keep YARN=0, CONTEXT_LENGTH=262144.
#   - .env still applies for everything else (QUANT, concurrency,
#     CHUNKED_PREFILL, CPUSET, ...) except EXTRA_ARGS — this script's
#     EXTRA_ARGS wins. If you need extra flags on top, append them to the
#     EXTRA_ARGS export below (space-separated).
#   - DSPARK_EXTRA env var (optional): extra flags appended AFTER the base
#     stack, for per-boot experiments without editing this file, e.g.:
#       DSPARK_EXTRA="--linear-attn-decode-backend flashinfer" ./start-dspark.sh
#
# Usage: ./start-dspark.sh   (stop with ./stop.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA_ARGS="--speculative-algorithm DSPARK \
--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
--speculative-dspark-block-size 7 \
--speculative-draft-model-quantization unquant \
--speculative-num-draft-tokens 8 \
--mem-fraction-static 0.90 \
--enable-torch-compile --torch-compile-max-bs 4 \
--cuda-graph-max-bs-decode 4 \
--num-continuous-decode-steps 2"
EXTRA_ARGS+=" ${DSPARK_EXTRA:-}"
export EXTRA_ARGS

echo "DSpark mode: EXTRA_ARGS=${EXTRA_ARGS}"
echo "Delegating to ${SCRIPT_DIR}/start.sh"
exec "${SCRIPT_DIR}/start.sh"
