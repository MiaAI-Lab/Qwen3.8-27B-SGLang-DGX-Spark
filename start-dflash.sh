#!/usr/bin/env bash
set -euo pipefail

# DFlash2 wrapper. We serve Qwen3.8-27B with the DFlash2 block-diffusion
# draft instead of start.sh's EAGLE/MTP, by injecting EXTRA_ARGS (appended
# last, argparse last-wins). The draft is pinned to DRAFT_MODEL@DRAFT_REVISION
# (z-lab's DFlash2 draft; incoai/... is a mirror of the same weights).
# Image: an official multi-arch lmsysorg/sglang nightly from main (pinned
# by its index digest, pulled from Docker Hub on first run — no git clone,
# no local build). main carries DFlash2 (sglang #35371) and the quantized
# target lm_head selector (#35496), so every DF_TARGET works, including the
# packed-FP4 head, plus #35255 (zombie-request fix — see CHANGELOG
# 2026-09-09). Override with IMAGE=<ref>; a locally present image is used
# as-is. The self-built image machinery (patch/) was retired 2026-09-05;
# commit 751e29e is the last one carrying it.
# CRASH RULES (NVFP4): --mem-fraction-static 0.90 (0.95 hard-rebooted the
# GB10 once at draft-graph capture, on the self-built image; the cookbook
# pins 0.80 on GB10 because 0.85 trips DGX OS earlyoom). Default
# DF_TARGET=nvfp4 is the BF16-lm_head export (dense head).
# DFLASH requires --mamba-radix-cache-strategy extra_buffer on the image
# this was validated on (extra_buffer_lazy was rejected); the official
# image adds lazy support (#34763) but that is untested here.

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

# Official image, pinned by the multi-arch index digest (docker resolves the
# linux/arm64 child). Upstream build: main commit 708f51e44 (2026-09-09),
# nightly-cu134-20260909-708f51e — the first arm64 line carrying sglang
# #35255 (zombie-request fix; dev-qwen38-27b-dflash2 predates it, and
# v0.5.19 was tagged before it). To bump: `docker buildx imagetools
# inspect lmsysorg/sglang:<tag>` prints the index digest;
# update IMAGE_DIGEST, then re-validate on the box before trusting numbers.
IMAGE_REPO="lmsysorg/sglang"
IMAGE_TAG="nightly-cu134-20260909-708f51e"
IMAGE_DIGEST="sha256:00205b89f74691f76a0ffbd6846376d9323971930a5d59bf63a65dadc7d67927"
IMAGE_UPSTREAM_COMMIT="708f51e44"
IMAGE="${IMAGE:-${IMAGE_REPO}@${IMAGE_DIGEST}}"
LEGACY_IMAGE="lmsysorg/sglang:qwen38-27b-dflash2"   # the retired self-built image

ensure_image() {
  local pinned=0
  [[ "${IMAGE}" == "${IMAGE_REPO}@${IMAGE_DIGEST}" ]] && pinned=1
  if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Using ${IMAGE}"
    return
  fi
  echo "${IMAGE} not present locally — pulling from Docker Hub (~14 GB compressed for arm64) ..."
  docker pull "${IMAGE}" \
    || { echo "pull failed for ${IMAGE} — check network / Docker Hub rate limit (docker login helps); a tag that only ever existed locally must be rebuilt or loaded first (the retired builder is at commit 751e29e), or set IMAGE= to an image you have"; exit 1; }
  docker image inspect "${IMAGE}" >/dev/null 2>&1 \
    || { echo "pull reported success but ${IMAGE} is still missing"; exit 1; }
  (( pinned )) || return 0
  # A digest pull shows TAG=<none> in `docker images`; alias it so it reads as
  # what it is and does not look like a prune candidate. Never re-point a tag
  # that already exists (e.g. a manual `docker pull` of the rolling dev tag):
  # the container runs by digest either way.
  if docker image inspect "${IMAGE_REPO}:${IMAGE_TAG}" >/dev/null 2>&1; then
    docker image inspect --format '{{join .RepoDigests ","}}' "${IMAGE_REPO}:${IMAGE_TAG}" | grep -q "${IMAGE_DIGEST}" \
      || echo "note: local tag ${IMAGE_REPO}:${IMAGE_TAG} points at a different build; leaving it alone (this run uses the pinned digest)"
  else
    docker tag "${IMAGE}" "${IMAGE_REPO}:${IMAGE_TAG}" || true
  fi
  # Only now, with the new image safely on disk, mention the retired one.
  if docker image inspect "${LEGACY_IMAGE}" >/dev/null 2>&1; then
    echo "note: the self-built ${LEGACY_IMAGE} is no longer the default (IMAGE=${LEGACY_IMAGE} keeps using it; docker image rm ${LEGACY_IMAGE} frees the space)"
  fi
}
ensure_image
if [[ "${IMAGE}" == "${IMAGE_REPO}@${IMAGE_DIGEST}" ]]; then
  echo "DFlash image: ${IMAGE_REPO}:${IMAGE_TAG} @ ${IMAGE_DIGEST:0:19}… (upstream ${IMAGE_UPSTREAM_COMMIT})"
else
  echo "DFlash image: ${IMAGE} (IMAGE override)"
fi
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
  bf16) TARGET_PATH="Qwen/Qwen3.8-27B" ;;
  nvfp4|nvfp4-bf16|nvfp4-bf16-head)
        TARGET_PATH="RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead" ;;
  nvfp4-fp4|nvfp4-fp4-head)
        TARGET_PATH="RadixArk/Qwen3.8-27B-NVFP4" ;;
  *) echo "DF_TARGET must be bf16, nvfp4, or nvfp4-fp4, got '${DF_TARGET}'"; exit 1 ;;
esac

EXTRA_ARGS="--model-path ${TARGET_PATH} \
--speculative-algorithm DFLASH \
--speculative-draft-model-path ${DRAFT_MODEL}${DRAFT_REVISION:+ --speculative-draft-model-revision ${DRAFT_REVISION}} \
--speculative-num-draft-tokens 8 \
--mamba-radix-cache-strategy extra_buffer"
case "${DF_TARGET}" in
  nvfp4|nvfp4-bf16|nvfp4-bf16-head|nvfp4-fp4|nvfp4-fp4-head)
    EXTRA_ARGS+=" --mem-fraction-static 0.90" ;;
esac
EXTRA_ARGS+=" ${DF_EXTRA:-}"
export EXTRA_ARGS

echo "DFlash mode: EXTRA_ARGS=${EXTRA_ARGS}"
echo "Delegating to ${SCRIPT_DIR}/start.sh"
exec "${SCRIPT_DIR}/start.sh"
