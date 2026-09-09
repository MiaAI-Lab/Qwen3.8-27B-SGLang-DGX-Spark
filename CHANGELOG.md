# Changelog

All notable changes to this project are documented here. Dates are commit dates.

## 2026-09-09 — default image bumped to a post-#35255 nightly (zombie-request fix)

The previous pin (`dev-qwen38-27b-dflash2`, sglang `5f55db35e`, 2026-08-22) predates sglang [#35255](https://github.com/sgl-project/sglang/pull/35255) (merged 2026-09-04). On that build, a streaming client disconnect mid-generation leaves a zombie request: the TokenizerManager pops the state on `CancelledError`, then `abort_request()` early-returns, so the scheduler keeps decoding to `max_tokens` — holding a `--max-running-requests` slot and flooding `Received output for rid=… but the state was deleted in TokenizerManager` (upstream: [sglang#36333](https://github.com/sgl-project/sglang/issues/36333), [#36876](https://github.com/sgl-project/sglang/issues/36876)). `v0.5.19` (tagged 2026-09-03) does not contain the fix.

**Observed on the GB10 (2026-09-09):** one disconnect left a zombie decoding 16.75 min (6,466 flood lines, 98k total tokens); 10+ zombie rids accumulated over the morning with 18 router-side "client hung up" events and an instantaneous throughput collapse to ~1 tok/s under 8–10 queued requests.

**Changed:**

- `start-dflash.sh` defaults `IMAGE` to `lmsysorg/sglang@sha256:00205b89f74691f76a0ffbd6846376d9323971930a5d59bf63a65dadc7d67927` — the index digest of `nightly-cu134-20260909-708f51e` (main `708f51e44`, 2026-09-09; contains #35255 plus everything the old pin carried: #35371, #35496, #34763, #34859). `IMAGE=` override unchanged. Re-pin to `v0.5.20` when tagged.
- DFLASH config unchanged: `--mamba-radix-cache-strategy extra_buffer`, `--mem-fraction-static 0.90`, same draft pin.

**Verified on the GB10 (2026-09-09, after the swap):** kill-client-mid-stream repro produces zero zombie flood and the request leaves the running batch within 15 s (vs 16.75 min of zombie decode + 6,466 flood lines on the old image); direct and router (cloudflared → minirouter → :8888) smoke tests pass; RAM profile unchanged (113/121 GiB).

## 2026-09-05 — DFlash2 serves from the official SGLang image; `patch/` builder removed

Upstream now publishes a multi-arch image with DFlash2 and the quantized-`lm_head` selector: `lmsysorg/sglang:dev-qwen38-27b-dflash2`, the tag the [cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) maps to DGX Spark ([issue #6](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/issues/6)). `v0.5.18` was tagged later but does not contain the DFlash2 commit (GitHub compare: diverged), so the dev tag is pinned by digest instead of waiting for a release.

**Changed:**

- `start-dflash.sh` defaults `IMAGE` to `lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe` — the index digest of `dev-qwen38-27b-dflash2` (upstream build `5f55db35e` on branch `dflash2-pin-1cf2b8c-nccl`, 2026-08-22; contains sglang #35371, #35496, #34763) — and pulls it on first run (~14 GB compressed, arm64). No git clone, no `docker build`. The pulled image is aliased locally as `lmsysorg/sglang:dev-qwen38-27b-dflash2`.
- `IMAGE=<ref>` override unchanged; a locally present image is used as-is, so `IMAGE=lmsysorg/sglang:qwen38-27b-dflash2` keeps running an already-built legacy image. After a successful pull of the pinned image the script notes a leftover legacy image; the local alias tag is only created when no tag of that name exists (never re-pointed).
- `DF_TARGET=nvfp4-fp4` no longer depends on a local patch: the image's selector handles the packed-FP4 head.
- Draft pin, `DF_TARGET` defaults, `--mem-fraction-static 0.90`, forced `extra_buffer`: unchanged.

**Removed:**

- `patch/` (`build-dflash2-image.sh`, `dflash2_nvfp4_head.patch`, `overlay-dflash2/`) and the `-minoverlay` image mode. Last commit carrying them: `751e29e` (on `main`).

**Docs:**

- README no longer claims DFlash2 has no upstream image: Requirements, Quick start, Scripts, Configuration, Notable serving choices, Measured, Logs & troubleshooting (earlyoom, pull failures, upstream watchpoints #36548 / #38009), Repository layout and Credits updated. The 2026-08-19 DFlash2 numbers are labelled as taken on the self-built image; official-image replication is in **Verified** below.
- `docs/brainstorms/*.md` and `docs/plans/*.md` are now tracked (decision record and plan for this change).
- `stop.sh` header: names `start-dflash.sh` and no longer implies `start-mtp-8889.sh` is tracked (comment only).

**Verified on the GB10 (2026-09-05):** four interleaved single-session runs, n=6 per side, concurrency 10, `bench/ab-image.sh`. Both images boot DFLASH with the selector folded into the draft CUDA graph on both checkpoints; no earlyoom kills (earlyoom is inactive on this box).

- Serving image, DFlash2 self-built → official: `RadixArk/…-NVFP4-BF16-LMHead` (modelopt) 54.58 → 54.57 code and 25.69 → 25.70 essay, a tie; a compressed-tensors NVFP4 fine-tune of the same model 46.47 → 53.05 code (+14.1%) and 24.06 → 25.58 essay (+6.3%). Free on modelopt, a real gain on compressed-tensors — plausibly the 69 upstream commits the official image carries, newest being sglang #35455.
- Engine, MTP → DFlash2 in the same session: 2.25× code / 1.41× essay on the default checkpoint, 2.20× / 1.46× on the compressed-tensors fine-tune. Independently consistent with issue #6, and larger than the ~1.5× reported there.
- Stale data found: MTP now measures 24.2 code / 18.2 essay against 34.5 / 24.1 from 2026-08-18. Day and checkpoint export both differ, so the cause is unconfirmed; the README flags the August MTP/DSpark cells as FP4-head-era until someone re-runs `QUANT=nvfp4-fp4`.
- Method: two boots of the same image differed 6.5% on essay, so sides must be interleaved within one session. The first `ndec.py` pass after a boot is unusable (cold start collapses the two-call denominator; readings of 89, 183 and −371 tok/s observed) — `bench/ab-image.sh` discards one pass per boot.

## 2026-08-25 — Default NVFP4 uses the dense BF16 `lm_head`

SGLang now ships two NVFP4 checkpoints ([cookbook split](https://github.com/sgl-project/sglang/pull/36020)): packed-FP4 `lm_head` vs dense BF16. Cookbook recipes were measured against the BF16-head export, so this repo follows that.

**Changed:**

- `QUANT=nvfp4` (default) and `DF_TARGET=nvfp4` now load `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` instead of `RadixArk/Qwen3.8-27B-NVFP4`. That includes `./start-dflash.sh` with no extras.
- Packed-FP4-head remains available as `QUANT=nvfp4-fp4` / `DF_TARGET=nvfp4-fp4`. Full BF16 weights stay `QUANT=bf16` / `DF_TARGET=bf16` (`Qwen/Qwen3.8-27B`).
- Aliases: `nvfp4-bf16` / `nvfp4-bf16-head` and `nvfp4-fp4-head`.
- Dense BF16 head is ~1.7 GB larger on disk and ~3.2 GB larger at runtime than the packed-FP4 twin. Existing mem-fraction pins (DSpark/DFlash2 0.90, MTP 0.95) are unchanged; if capture OOMs, drop `mem-fraction-static`. Prior on-box numbers were taken on the FP4-head export.

## 2026-08-18 — DSpark vs MTP A/B; benches under `bench/`

**Performance (live, same NVFP4 weights, same `lmsysorg/sglang:qwen38-27b`, 2026-08-18 evening):**

- Code — LRUCache + small test (`bench/ndec.py`, n=5, T=0, thinking off): DSpark **51.5 tok/s** (51.38–51.73; `c2` always 518) vs MTP **34.5 tok/s** (34.46–34.57; `c2` always 508) — **~1.5× / +49%**.
- Default chat — “what is a hash map…” (stream, post-first-token, T=1 thinking on): DSpark **23.2** vs MTP **21.0** — wash; DSpark slightly faster. Thinking-off short chat is a small MTP edge (24.6 / 23.4 vs 22.0 / 21.3).
- Long essay — Babbage → GPUs (`bench/ndec.py`, n=5): DSpark **18.3 tok/s** (18.18–18.29) vs MTP **24.1 tok/s** (24.05–24.13) — MTP **~1.3× / DSpark −24%**. That is the real MTP win, not everyday chat.
- Block sweep (same day): block-7 is the code peak; block-5 is **+8% prose / −16% code**. `--speculative-accept-threshold-acc <1` hurt — leave at 1.0.
- Older wall-time figures (`bench/bench.sh`, includes prefill) are still MTP-era: thinking 17.2–20.5, non-thinking 21.6–22.7, tool-call 26–28. Not comparable to the `ndec` / stream table.

**Added:**

- `bench/ndec.py` — two-call net-decode A/B (LRUCache + essay). How the current-era numbers were measured. Treat code deltas &lt;15% as noise.
- Moved `bench.sh` → `bench/bench.sh` (essay / tool-call wall-time + 16K TTFT). Different clock from `ndec.py`.
- `.gitignore` whitelist: `bench/`, `bench/bench.sh`, `bench/ndec.py`. Explicit `HANDOFF.md` — stays local, never un-ignore.

**Docs:**

- README lead, Quick start, Which engine, and Measured tables use the n=5 + stream A/B. Quick start leads with `./start-dspark.sh` (agents / code / normal chat). `./start.sh` stays MTP (long essays, YaRN / 1M).
- DSpark cannot use YaRN / `CONTEXT_LENGTH` &gt; 262144 on this build (rope override leaks into the draft config and crashes). Native 262K only.

## 2026-08-18 — Track `bench.sh`

- `.gitignore` whitelist now includes `bench.sh` (single-stream decode bench against :8888). Later moved under `bench/`.

## 2026-08-18 — Tuned for DGX Spark: core pinning, correct GDN pool sizing, measured spec decoding

**Performance (measured on-device):**

- Single-stream decode up from 16.9 / 21.0 to 17.2–20.5 / 21.6–22.7 tok/s (thinking / non-thinking), TTFT ~8.3 s for a fresh ~16K-token prompt.
- Speculative-decoding sweep (steps 2/3/4/5/6 → draft 3/4/5/6/7): 12.8 / **17.2** / 16.8 / 16.3 / 15.8 tok/s thinking — the MTP 3/1/4 default is this checkpoint's measured peak (the head is trained for exactly 3 steps; deeper chains only add rejected verify work).

**Fixed:**

- GDN state-pool sizing. The pin is now `--max-mamba-cache-size = MAX_CONCURRENT_REQUESTS × S` (S=4 for `extra_buffer_lazy` + overlap scheduler), matching the engine's own formula (verified in this build's `kv_cache_configurator`): the pool is divided by S alone and the speculative verify window is sized as a separate buffer. The previous `× (S+D)` pin over-provisioned the pool 2× (~1.2 GB of state memory returned to the KV pool at concurrency 4; the KV pool grew from ~2.46M to ~2.48M tokens). The README's earlier "× 8 slots / expect 80 slots" guidance was corrected accordingly.

**Added:**

- Container pinned to GB10's ten 3.9 GHz Cortex-X5 cores (`--cpuset-cpus 5-9,15-19`); the scheduler/tokenizer processes no longer float onto the 2.8 GHz A725 efficiency cores. Measured +2–7% decode. Override or disable via `CPUSET`.
- New `.env` / shell-env tuning surface, all with launch-time validation:
  - `SPEC_STEPS` / `SPEC_TOPK` / `SPEC_DRAFT` (default 3/1/4; topk=1 requires `SPEC_DRAFT = SPEC_STEPS + 1`)
  - `CHUNKED_PREFILL` (default 8192; 2048 for smoother decode inter-token latency under mixed load)
  - `MAMBA_SKIP_DECODE_LOCK=1` — opt-in `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK`, frees one GDN state slot per request (S 4→3)
  - `PREFILL_CUDA_GRAPH=1` — drops `--disable-prefill-cuda-graph` (informational: this build auto-disables prefill graphs on this model anyway)
  - `EXTRA_ARGS` — free-form extra SGLang flags appended last (argparse last-wins, so it can override built-ins); the experiment hatch
- README: "Measured on this box" section (decode / tool-call / TTFT numbers, ±7% run-to-run variance warning, bandwidth-ceiling analysis), corrected pool guidance, cold-boot TTFT/Triton-warmup troubleshooting note, mamba-pool boot check.

**Tested and rejected (documented so nobody re-tests):**

- NGRAM speculative drafting: 12.7–15.7 tok/s, ~30% under MTP even on tool-call-style output.
- Prefill CUDA graphs: auto-disabled by the build on this model — GDN layers don't apply standard GQA.
- `--enable-fused-qk-norm-rope`: within run-to-run noise.

## 2026-08-15 — Repository hardening

- Added MIT `LICENSE`.
- Track `.gitignore` itself so the whitelist rules ship with the repo.

## 2026-08-15 — Initial release

- `start.sh` / `stop.sh`: opinionated, ready-to-run serving of Qwen3.8-27B with SGLang in Docker on DGX Spark (GB10, aarch64) — the SGLang cookbook's validated DGX Spark cell (NVFP4 W4A4 checkpoint, flashinfer attention, `--mem-fraction-static 0.95`, 8192-token prefill chunks, FP8 KV cache, BF16 GDN state, `extra_buffer_lazy` radix strategy).
- MTP speculative decoding via the checkpoint's own head (EAGLE 3 steps / topk 1 / 4 draft tokens); DSpark documented as an alternative with measured head-to-head numbers.
- Long context: native 262K, YaRN rope scaling up to a validated 1M (factor derived automatically; `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` handled internally; DSpark incompatibility documented).
- `.env` config surface (`QUANT`, `YARN`, `CONTEXT_LENGTH`, `MAX_CONCURRENT_REQUESTS`) with `.env.sample` template; idempotent start, readiness polling, log streaming to `.sglang.log`.
- Thinking mode on by default (`--reasoning-parser qwen3`) and tool calling (`--tool-call-parser qwen3_coder`); OpenAI- and Anthropic-compatible endpoints.
