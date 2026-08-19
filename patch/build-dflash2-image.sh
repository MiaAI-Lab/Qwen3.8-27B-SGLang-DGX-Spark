#!/usr/bin/env bash
set -euo pipefail
# Build the local DFlash2 SGLang image used by start-dflash.sh. No released
# SGLang tag ships DFlash2 (upstream merged 2026-08-19, commit c14312a66), so
# DFlash2 support is overlaid onto the pinned qwen38-27b image in one of two
# modes:
#   default / --full  whole mainline python/sglang tree at c14312a66
#                     + dflash2_nvfp4_head.patch  -> :qwen38-27b-dflash2
#   --minimal         only the 5 DFlash2 modules from overlay-dflash2/
#                     (sha256-verified)           -> :qwen38-27b-dflash2-minoverlay
# start-dflash.sh auto-invokes this (--minimal iff IMAGE=...-minoverlay); run
# it manually to rebuild after bumping the upstream pin. Full mode needs git +
# network on first build; minimal mode never does. Docker COPY preserves every
# image-unique lower-layer file (fork kernels etc.).

mode="full"
case "${1:-}" in
  ""|--full) : ;;
  --minimal) mode="minimal" ;;
  *) echo "usage: $0 [--full|--minimal]" >&2; exit 1 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_image=lmsysorg/sglang:qwen38-27b
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

if [[ "${mode}" == "minimal" ]]; then
  overlay="${script_dir}/overlay-dflash2"
  [[ -f "${overlay}/MANIFEST.sha256" ]] || { echo "missing ${overlay}/MANIFEST.sha256"; exit 1; }
  (cd "${overlay}" && sha256sum -c MANIFEST.sha256) || { echo "overlay checksum mismatch"; exit 1; }
  mkdir -p "${stage}/overlay/sglang"
  cp -r "${overlay}/sglang/." "${stage}/overlay/sglang/"
  files=(
    kernels/ops/speculative/dflash.py
    srt/models/dflash.py
    srt/model_executor/model_runner_components/spec_aux_hidden_state.py
    srt/speculative/dflash_utils.py
    srt/speculative/dflash_worker_v2.py
  )
  {
    echo "FROM ${base_image}"
    for rel in "${files[@]}"; do
      echo "COPY overlay/sglang/${rel} /sgl-workspace/sglang/python/sglang/${rel}"
    done
  } > "${stage}/Dockerfile"
  tag=lmsysorg/sglang:qwen38-27b-dflash2-minoverlay
  echo "minimal overlay verified: ${#files[@]} files"
else
  src_commit=c14312a66
  full_sha=c14312a66420b75ca9a11bf1817c4db1fa26b097
  if [[ ! -d /tmp/sglang-src/.git ]]; then
    echo "cloning sglang (blobless) ..."
    git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git /tmp/sglang-src
  fi
  git -C /tmp/sglang-src -c remote.origin.promisor=true -c fetch.filter=blob:none \
      fetch origin "${full_sha}" --depth 1 >/dev/null 2>&1 || true
  git -C /tmp/sglang-src checkout "${full_sha}" >/dev/null 2>&1
  git -C /tmp/sglang-src reset --hard HEAD >/dev/null 2>&1 || true
  # NVFP4 head handling for the DFLASH selector: run the quantized head in
  # place via lm_head.quant_method.apply (see dflash2_nvfp4_head.patch). No
  # dense dequant copy — a previous dequant-once approach allocated ~2.5-5 GB
  # at draft-graph capture and hard-rebooted the box.
  if [[ -f "${script_dir}/dflash2_nvfp4_head.patch" ]]; then
    git -C /tmp/sglang-src apply --whitespace=nowarn "${script_dir}/dflash2_nvfp4_head.patch"
  fi
  mkdir -p "${stage}/python/sglang"
  cp -r /tmp/sglang-src/python/sglang/. "${stage}/python/sglang/"
  {
    echo "FROM ${base_image}"
    echo "COPY python /sgl-workspace/sglang/python"
  } > "${stage}/Dockerfile"
  tag=lmsysorg/sglang:qwen38-27b-dflash2
  echo "overlay @ ${src_commit}"
fi

export DOCKER_BUILDKIT=1
docker build -t "${tag}" -f "${stage}/Dockerfile" "${stage}"
echo "built ${tag}"
