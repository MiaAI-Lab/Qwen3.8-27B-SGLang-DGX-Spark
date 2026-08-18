# Changelog

All notable changes to this project are documented here. Dates are commit dates.

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
