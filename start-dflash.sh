#!/usr/bin/env bash
set -euo pipefail

# DFlash2 wrapper. We serve Qwen3.8-27B with the DFlash2 block-diffusion
# draft instead of start.sh's EAGLE/MTP, by injecting EXTRA_ARGS (appended
# last, argparse last-wins). The draft is pinned to DRAFT_MODEL@DRAFT_REVISION
# (z-lab's DFlash2 draft; incoai/... is a mirror of the same weights).
# Self-contained: if the derived image is missing, it is built automatically
# from patch/ (patch/build-dflash2-image.sh + overlay-dflash2/; needs git +
# network on first build).
# CRASH RULES (NVFP4): --mem-fraction-static 0.90 (0.95 hard-rebooted the
# GB10 once at draft-graph capture; fixed since by running the quantized head
# in place, no dense dequant) and modest MAX_CONCURRENT_REQUESTS if the box
# is busy; NVFP4 needs the quantized-head selector support baked into the
# image or the first request dies ("DFlash2 selector requires a dense ...
# target lm_head"); DFLASH requires --mamba-radix-cache-strategy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure the draft (and only the draft) is cached where the container reads it.
HF_CACHE="${SCRIPT_DIR}/.cache/huggingface/hub"
mkdir -p "${HF_CACHE}"

DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.8-27B-DFlash2}"
DRAFT_REVISION="${DRAFT_REVISION:-50307d4c4cde6860d4eee73e2547cd786fe8e8a4}"

snapshot_present() {
  local base="${HF_CACHE}/models--${1//\//--}"
  local rev=""
  [[ -n "${DRAFT_REVISION}" ]] && rev="/snapshots/${DRAFT_REVISION}" || rev="/snapshots"
  [[ -n "$(find -L "${base}${rev}" -maxdepth 2 -type f -print -quit 2>/dev/null)" ]]
}

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  HF_TOKEN="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\2/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN

IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b-dflash2}"
case "${IMAGE}" in *-minoverlay) build_mode=(--minimal) ;; *) build_mode=() ;; esac
ensure_image() {
  if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Using ${IMAGE}"
  else
    echo "${IMAGE} not present locally — building DFlash2 image ..."
    "${SCRIPT_DIR}/patch/build-dflash2-image.sh" "${build_mode[@]}" || { echo "DFLASH2 image build failed"; exit 1; }
  fi
}
ensure_image
export IMAGE

ensure_cached() {
  local repo="$1"
  local label="${repo}${DRAFT_REVISION:+ @ ${DRAFT_REVISION}}"
  if snapshot_present "${repo}"; then
    echo "draft already cached (${label})"
  else
    echo "draft not cached — pulling ${label} ..."
    docker run --rm --network host \
      -e HF_HOME=/root/.cache/huggingface \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -v "${SCRIPT_DIR}/.cache/huggingface:/root/.cache/huggingface" \
      "${IMAGE}" \
      python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${repo}'${DRAFT_REVISION:+, revision='${DRAFT_REVISION}'})" \
      || { echo "pull failed for ${label}"; exit 1; }
    snapshot_present "${repo}" || { echo "pull failed for ${label}"; exit 1; }
  fi
}
ensure_cached "${DRAFT_MODEL}"

DF_TARGET="${DF_TARGET:-nvfp4}"
case "${DF_TARGET}" in
  bf16)  TARGET_PATH="Qwen/Qwen3.8-27B" ;;
  nvfp4) TARGET_PATH="RadixArk/Qwen3.8-27B-NVFP4" ;;
  *) echo "DF_TARGET must be bf16 or nvfp4, got '${DF_TARGET}'"; exit 1 ;;
esac

EXTRA_ARGS="--model-path ${TARGET_PATH} \
--speculative-algorithm DFLASH \
--speculative-draft-model-path ${DRAFT_MODEL}${DRAFT_REVISION:+ --speculative-draft-model-revision ${DRAFT_REVISION}} \
--speculative-num-draft-tokens 8 \
--mamba-radix-cache-strategy extra_buffer"
if [[ "${DF_TARGET}" == "nvfp4" ]]; then
  EXTRA_ARGS+=" --mem-fraction-static 0.90"
fi
EXTRA_ARGS+=" ${DF_EXTRA:-}"
export EXTRA_ARGS

echo "DFlash mode: EXTRA_ARGS=${EXTRA_ARGS}"
echo "Delegating to ${SCRIPT_DIR}/start.sh"
exec "${SCRIPT_DIR}/start.sh"
