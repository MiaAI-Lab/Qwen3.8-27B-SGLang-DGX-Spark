# Qwen3.8 27B on SGLang for DGX Spark

[![SGLang](https://img.shields.io/badge/SGLang-cookbook-blue)](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
[![Model](https://img.shields.io/badge/model-Qwen3.8--27B-informational)](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
[![arch](https://img.shields.io/badge/arch-arm64%20%2F%20GB10-lightgrey)](#)

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Opinionated, ready-to-run scripts to serve **[Qwen3.8-27B](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)** with **[SGLang](https://docs.sglang.io)** in Docker on an NVIDIA DGX Spark (GB10, aarch64). Three swap-in serving modes — EAGLE/MTP, DSpark, or DFlash2 — with every tuning choice measured on-device instead of guessed.

**DSpark and DFlash2 are faster on code.** Versus MTP, DSpark gives the essay back; DFlash2 does not. Everyday chat on the same streamed probe comes out a DFlash2 win once tokens are counted right, and the long-essay probe is on the MTP side of the table. **All measured numbers, ranges, counting notes and caveats live in [Measured on this box](#measured-on-this-box) — one place, nothing repeated.**

The launch flags start from the **[SGLang cookbook's DGX Spark cell](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)** (NVFP4 + DSpark), then pin choices measured on this box: GDN **bf16** (cookbook float32 was −3%), `extra_buffer_lazy`, mem **0.90**, chunk **8192**, DSpark **block 7 / 8 draft tokens**, torch.compile + decode graphs, X5 cpuset.

- **NVFP4 W4A4** checkpoint (default; BF16 and FP8 available via `QUANT=…`)
- **native 262K context, YaRN off, and 10 concurrent requests by default** — optionally extend to a validated 1M via YaRN (see “[Long context & concurrency](#long-context--concurrency-up-to-1m-10-concurrent)”)
- **FP8 KV cache** (`fp8_e4m3`, ~2× KV memory savings; uses the NVFP4 checkpoint's calibration scales)
- **GDN state pool** sized correctly from `MAX_CONCURRENT_REQUESTS` (concurrency × 4 state slots; the spec verify window is a **separate** engine-side buffer — verified in the build's `kv_cache_configurator`)
- **Pinned to GB10's ten 3.9 GHz Cortex-X5 cores** (`--cpuset-cpus 5-9,15-19`) — the scheduler/tokenizer never land on the 2.8 GHz A725 efficiency cores (measured +2–7% decode)
- **Thinking mode on by default** (`--reasoning-parser qwen3` → `reasoning_content`) and **tool calling** (`qwen3_coder` parser)

---

## Requirements

| Component | Detail |
|---|---|
| Hardware | NVIDIA DGX Spark / GB10 (aarch64, SM121; 128 GB unified memory) |
| Docker | With NVIDIA Container Toolkit / GPU passthrough working (`docker run --gpus all`) |
| SGLang image | `lmsysorg/sglang:qwen38-27b` (model-specific build from the cookbook; multi-arch incl. arm64) |
| CLI tools | `docker`, `curl` |
| Hugging Face token | `HF_TOKEN` defined in `~/.bashrc` (picked up automatically; higher rate limits) |

There is no separate download step: the container pulls the checkpoint into `./.cache/huggingface` on first start (~22 GB for the NVFP4 repo; the cookbook cites ~16.5 GB for the NVFP4 LM weights alone, before the MTP head).

## Quick start (ships as native 262K, 10 concurrent)

```bash
# 1. Copy the sample config once (creates ./.env if you don't have one)
cp .env.sample .env

# 2. Start the server
./start-dspark.sh    # DSpark — code ~51.5; default chat ~23; long essay ~18
# ./start.sh         # MTP — code ~34.5; default chat ~21; long essay ~24
# ./start-dflash.sh  # DFlash2, NVFP4 target — code ~50.9; essay ~25.4; chat ~29–67 (streamed)
#                    #   (bf16 base: DF_TARGET=bf16 — unbenched on this box)

# 3. Use it
curl http://127.0.0.1:8888/v1/models

# 4. Stop it
./stop.sh
```

`.env.sample` ships with `YARN=0`, `CONTEXT_LENGTH=262144` (native) and `MAX_CONCURRENT_REQUESTS=10` — so a fresh clone serves **262K context, YaRN off, 10 concurrent** after just the `cp` above. `.env` is the live config (plain `VAR=value` lines read by `start.sh`): shell exports of the same names win, `.env` fills the gaps, start.sh defaults apply last. Changes only take effect on the **next** launch — `./stop.sh && ./start-dspark.sh` (or `./start.sh` for MTP). For anything above native context (e.g. 1M) or a different concurrency, see [Long context & concurrency](#long-context--concurrency-up-to-1m-10-concurrent). Note: DSpark cannot use YaRN / context &gt; 262144 on this build (ditto DFlash2 — same draft-config leak).

`start-dflash.sh` is self-contained but different: no released SGLang image has DFlash2 support yet (it merged upstream 2026-08-19, after every published tag including the pinned `qwen38-27b`), so on a machine without the local `lmsysorg/sglang:qwen38-27b-dflash2` image the script **builds it automatically** (needs git + network once) via `patch/build-dflash2-image.sh`, which overlays the mainline python tree at that commit plus `patch/dflash2_nvfp4_head.patch` (quantized-head selector support for the NVFP4 target via `lm_head.quant_method` — no dense dequant; a dequant-once approach hard-rebooted this box at graph capture) onto the pinned GB10 image. First boot then pulls the ~2.7 GB draft if missing.

All start scripts are idempotent: if the container is already running they say so and exit; if a stopped container exists they remove it first. `./stop.sh` stops whichever engine is up.

## Scripts

| Script | What it does |
|---|---|
| `start.sh` | Launches the SGLang container (`docker run -d`, host network, `--shm-size 32g`), streams logs to `.sglang.log`, records the container ID in `.sglang.pid`, and polls `http://127.0.0.1:8888/v1/models` until the server is ready. **EAGLE/MTP speculative decoding** (`SPEC_STEPS/SPEC_TOPK/SPEC_DRAFT = 3/1/4`). Monitoring on by default: Prometheus `--enable-metrics` + `--enable-cache-report` (per issue #3). |
| `start-dspark.sh` | Same service, **DSpark** instead of EAGLE: block-7 / unquant draft, torch.compile + decode-graph caps, `--num-continuous-decode-steps 2`, mem 0.90. Thin wrapper (`EXTRA_ARGS` → `start.sh`). **Code 51.5 vs MTP 34.5 — see [Measured on this box](#measured-on-this-box) for all numbers.** |
| `start-dflash.sh` | Same service, **DFlash2** block-diffusion draft (default `z-lab/Qwen3.8-27B-DFlash2@50307d4`, pinned — same draft as the `incoai/…` mirror; `DRAFT_MODEL`/`DRAFT_REVISION` env-overridable). **Default target: NVFP4** `RadixArk/Qwen3.8-27B-NVFP4` with `--mem-fraction-static 0.90` and quantized-head selector support baked into the image (in-place `lm_head.quant_method.apply` — no dense dequant; a dequant-once approach hard-rebooted this box); `DF_TARGET=bf16` selects `Qwen/Qwen3.8-27B` instead. Builds the image automatically on first run from `patch/` if missing. Measured numbers: [Measured on this box](#measured-on-this-box). |
| `patch/build-dflash2-image.sh` | Builds the DFlash2 image (`--full` default: pinned mainline python tree + `patch/dflash2_nvfp4_head.patch`, needs git+network first time; `--minimal`: only the 5 DFlash2 modules from `patch/overlay-dflash2/`, sha256-verified, no network). Auto-invoked by `start-dflash.sh` when the image is missing (`--minimal` iff `IMAGE=…-minoverlay`). |
| `bench/ndec.py` | Two-call net-decode A/B (LRUCache + essay, thinking off). How the DSpark vs MTP numbers above were measured. Run twice; trust the second; treat code deltas &lt;15% as noise. |
| `bench/bench.sh` | Essay / tool-call **wall-time** bench + 16K TTFT probe (includes prefill). Different clock from `ndec.py`. |
| `stop.sh` | Stops the serving engine (idempotent; also cleans up any experiment processes still alive). Leaves the stopped container in place for `docker logs` post-mortem. |

Runtime artifacts: `.sglang.log` (server log), `.sglang.pid` (container ID), `.cache/` (HF + Triton caches). All are git-ignored.

> Whitelisted for tracking: `start.sh`, `start-dspark.sh`, `stop.sh`, `start-dflash.sh`, `patch/`, `bench/`, `README.md`, `CHANGELOG.md`, `.env.sample`, `LICENSE`, `.gitignore`. Experiment scripts and analysis docs stay untracked by design.

## Which engine to use

**EAGLE/MTP and DSpark** are the same NVFP4 27B on the same `lmsysorg/sglang:qwen38-27b` image — only the speculative decoder changes. **DFlash2 (`start-dflash.sh`)** defaults to the same NVFP4 weights, on a derived image (see [Scripts](#scripts) / Configuration below) that exists only because DFlash2 support is newer than any released image. DFlash2 can also target the bf16 base (`DF_TARGET=bf16`) — not benched.

Performance numbers for all three engines live in the **single table in [Measured on this box](#measured-on-this-box)** — below is engine selection only (no repeated tok/s):

| | `./start-dspark.sh` (DSpark block-7) | `./start.sh` (EAGLE/MTP 3/1/4) | `./start-dflash.sh` (DFlash2, NVFP4) |
|---|---|---|---|
| Best for | agents, code, tools, **normal chat** (**default here**) | long-form writing | code AND long-form writing (essay holds; chat is a win too) |
| Memory | 22 GB target + ~2.7 GB draft, mem 0.90 | 22 GB target (in-checkpoint MTP), mem 0.95 | NVFP4: 22 GB target + ~2.6 GB draft, **mem 0.90**; bf16: 52 GB weights |

DSpark/MTP columns are live 2026-08-18 evening; DFlash2 on 2026-08-19 (n=5, same single boot, same probes) — numbers and caveats are in [Measured on this box](#measured-on-this-box). **Take-aways, with that caveat firmly in mind:** DFlash2 on NVFP4 ties DSpark on code (inside the <15% noise band), *beats MTP* on the long essay, and *beats both on every short-chat condition* once counted from `completion_tokens` (the earlier ~9.2 “tok/s” reading was an SSE event-counting artifact — see the counting note in [Measured on this box](#measured-on-this-box)). Watchpoints: `mem-fraction-static 0.95` + DFlash2 wedged the box once (hard reboot; see [Logs & troubleshooting](#logs--troubleshooting)) — root-caused and fixed (see the DFlash2 bullet); the current `0.90` + 16-concurrent profile boot-**validated** 2026-08-19 with no reboot (concurrency ladder in [Measured on this box](#measured-on-this-box)). bf16 target is entirely unmeasured. Do not compare these to sparkDash fill-to-max streams.

Tuning history worth knowing: every Tier A (kernel-path) and Tier B (config/host) experiment measured **zero net gain** — these configs are the local optimum on this box. DSpark block-7 is the code peak; block-5 trades −16% code for +8% prose if you want it (`DSPARK_EXTRA`, see Configuration). Local-only write-ups: `TIER_A_RESULTS.md`, `TIER_B_RESULTS.md`, `TIER_C_RESULTS.md`, `DS4F.md`, `KIMI.md`, `GROK.md`, `HANDOFF.md`.

## Configuration

Defaults live at the top of `start.sh`:

| Variable | Default | Notes |
|---|---|---|
| `YARN` | `0` | `0` off / `1` on for `CONTEXT_LENGTH` > 262144; implicitly on at exactly `1000000`. Factor = round(`CONTEXT_LENGTH`/262144) |
| `CONTEXT_LENGTH` | `262144` | Range `262144`..`1000000` (native..1M). Combined with `YARN=1` for values above native; `1M` auto-enables YaRN even with `YARN=0`. Invalid values abort at startup |
| `MAX_CONCURRENT_REQUESTS` | `10` | Sizes `--max-mamba-cache-size` = concurrency × 4 slots and passes `--max-running-requests` |
| `SPEC_STEPS` / `SPEC_TOPK` / `SPEC_DRAFT` | `3` / `1` / `4` | MTP chain drafting; topk=1 requires `SPEC_DRAFT = SPEC_STEPS + 1` (validated at launch). Sweep the pair on your box and pin the winner — 3/1/4 is the measured peak here |
| `CHUNKED_PREFILL` | `8192` | Prefill chunk tokens. Cookbook DGX Spark cell uses `2048`; we keep `8192` (prefill/TTFT, not decode tok/s). |
| `CPUSET` | `5-9,15-19` | Docker `--cpuset-cpus` pin to GB10's Cortex-X5 cores (A725s are 0-4, 10-14). Empty = no pinning |
| `MAMBA_SKIP_DECODE_LOCK` | `0` | `1` sets `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` in the container — frees one GDN state slot per request (S 4→3) |
| `PREFILL_CUDA_GRAPH` | `0` | `1` drops `--disable-prefill-cuda-graph`. Info: this build auto-disables prefill graphs on this model anyway (GDN layers ≠ standard GQA) |
| `EXTRA_ARGS` | — | Free-form extra SGLang flags, appended **last** (argparse last-wins, so they can override built-ins). The experiment hatch: `EXTRA_ARGS="--fp4-gemm-runner-backend triton" ./start.sh` |
| `QUANT` | `nvfp4` | `nvfp4` → `RadixArk/Qwen3.8-27B-NVFP4`, `fp8` → `Qwen/Qwen3.8-27B-FP8`, `bf16` → `Qwen/Qwen3.8-27B`. All three fit in the Spark's 128 GB. |
| (shell overrides) | — | Any variable above can also be set as a shell env var, or put in `.env` |
| `SERVED_MODEL_NAME` | `qwen3.8-27b-sglang` | Name clients use in API requests |
| `IMAGE` | `lmsysorg/sglang:qwen38-27b` | Cookbook-pinned image for this model |
| `CONTAINER_NAME` | `qwen3.8-27b-sglang` | Also used by `stop.sh` |
| `PORT` | `8888` | Listens on `0.0.0.0` via host networking |

> The shipped `.env`/`.env.sample` match the `start.sh` defaults above, so a fresh clone serves **native 262K context, YaRN off, 10 concurrent** out of the box. Raise context above 262K with `YARN=1` + `CONTEXT_LENGTH` (see below).

`start-dspark.sh` adds a couple of knobs (shell-env or `.env`, optional):

| Variable | Default | Notes |
|---|---|---|
| `DSPARK_EXTRA` | — | Extra SGLang flags appended AFTER the base DSpark stack, for per-boot experiments without editing the script. E.g. `DSPARK_EXTRA="--speculative-dspark-block-size 5 --speculative-num-draft-tokens 6" ./start-dspark.sh` (prose-tuned block; see below) |
| `IMAGE` | `lmsysorg/sglang:qwen38-27b` | env override (`IMAGE=tag ./start-dspark.sh`) to run a patched derivative image; roll back by not setting it. |

Note: `--cuda-graph-max-bs` is a deprecated alias in this build; the DSpark stack uses `--cuda-graph-max-bs-decode 4`.

`start-dflash.sh` knobs (shell-env; it doesn't otherwise change `start.sh`/.env behavior except the model-path override):

| Variable | Default | Notes |
|---|---|---|
| `DF_TARGET` | `nvfp4` | `nvfp4` (default) → `RadixArk/Qwen3.8-27B-NVFP4` (+`--mem-fraction-static 0.90`); `bf16` → `Qwen/Qwen3.8-27B`. The NVFP4 path requires the quantized-head selector support baked into the derived image; without it, DFLASH dies at the first request (see [Scripts](#scripts)). |
| `DF_EXTRA` | — | Extra SGLang flags appended AFTER the base DFlash2 stack (last-wins). E.g. `DF_EXTRA="--mem-fraction-static 0.90" ./start-dflash.sh` |
| `IMAGE` | `lmsysorg/sglang:qwen38-27b-dflash2` (local) | The derived image; the script builds it if missing (auto-invokes `patch/build-dflash2-image.sh`). Override with `IMAGE=tag` — e.g. `IMAGE=lmsysorg/sglang:qwen38-27b-dflash2-minoverlay` (built via `patch/build-dflash2-image.sh --minimal`). |
| `DRAFT_MODEL` / `DRAFT_REVISION` | `z-lab/Qwen3.8-27B-DFlash2` / `50307d4…` | Pinned DFlash2 draft (same weights as the `incoai/…` mirror). `DRAFT_MODEL=incoai/Qwen3.8-27B-DFlash2 DRAFT_REVISION=dedf8df…` reproduces the original n=5 baseline; `DRAFT_REVISION=''` follows a branch head. |

DFlash2-specific stack facts (so nobody re-learns them on a crash): `extra_buffer_lazy` is rejected by DFLASH (AssertionError) → the script forces `--mamba-radix-cache-strategy extra_buffer`; DFLASH only supports `speculative_num_steps == 1` (engine auto-overrides start.sh's MTP 3); `--enable-dp-attention` and the overlap scheduler are off in this path; the draft (`incoai`/`z-lab/…-DFlash2`) is a block-diffusion drafter, not a token LLM, so EAGLE knobs (`topk`, `num_steps`) don't apply.

### Long context & concurrency (up to 1M, 10 concurrent)

All long-context and concurrency handling is driven by **three variables** in `.env` (or as shell exports):

| Variable | Meaning |
|---|---|
| `YARN` | `1` = enable YaRN rope scaling — **required** for any `CONTEXT_LENGTH` > 262144; `0` = off (sensible only at/below 262144) |
| `CONTEXT_LENGTH` | desired context in tokens, range `262144`..`1000000` |
| `MAX_CONCURRENT_REQUESTS` | parallel requests; also sets `--max-running-requests`, and sizes the GDN pool = value × 4 slots |

**Step by step — 1M context with 10 concurrent (goes above the shipped default):**

```bash
cp .env.sample .env                      # once, if you have no .env yet
nano .env                                # make sure these are set:
#  YARN=1
#  CONTEXT_LENGTH=1000000
#  MAX_CONCURRENT_REQUESTS=10
./stop.sh && ./start.sh                  # relaunch so new values apply
# verify after boot:
grep -E "context_len|max_running_requests" .sglang.log
expect: context_len=1000000, max_running_requests=10, mamba pool 40 slots
```

| You want | `YARN` | `CONTEXT_LENGTH` | `MAX_CONCURRENT_REQUESTS` | YaRN factor |
|---|---|---|---|---|
| 1M + 10 concurrent | 1 | 1000000 | 10 | 4.0 (also auto-on) |
| 512K + 10 concurrent | 1 | 524288 | 10 | 2.0 |
| 768K + 10 concurrent | 1 | 786432 | 10 | 3.0 |
| native 262K + 10 concurrent | 0 | 262144 | 10 | — |
| 1M + 2 concurrent | 1 | 1000000 | 2 | 4.0 |

Rules of thumb:

- **Above 262144 you must set `YARN=1`** (1M is the one exception — it auto-enables YaRN even with `YARN=0`, so 1M works no matter what). `YARN=0` at e.g. 524288 is allowed but produces a warning and the server stays at 262K.
- The YaRN factor is computed for you: `round(CONTEXT_LENGTH / 262144)` → 524288 gives 2.0, 786432 gives 3.0, 1000000 gives 4.0 (2.0 and 4.0 are the model card's validated points).
- SGLang otherwise fails closed at 262K with "User-specified context_length (...) is greater than the derived context_length"; `start.sh` auto-sets the required `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` env var, so you never touch it.
- Hardware bound (not a config knob): one KV token ≈ 32.8 KB, a full 1M sequence ≈ 33 GB, pool ≈ 75 GB → ~**2 full 1M requests** run at once regardless of `MAX_CONCURRENT_REQUESTS`; extra concurrent requests queue until KV frees.
- **DSpark caveat**: if you switch the speculative algorithm to DSpark, keep `YARN=0` / `CONTEXT_LENGTH=262144` — the YaRN override leaks into the DSpark draft config and crashes at boot.

### Notable serving choices

- **Recipe vs this repo:** the cookbook's DGX Spark + NVFP4 + DSpark cell uses mem 0.85, chunk 2048, GDN float32, radix `extra_buffer`; we measured those (float32 −3%, FP8 ~30% slower) and pin instead: mem 0.90, chunk 8192, GDN bf16, `extra_buffer_lazy`, block 7 / 8 draft tokens, torch.compile + decode graphs, `--num-continuous-decode-steps 2`, prefill graphs off, flashinfer, FP8 KV. MTP (`start.sh`) keeps mem 0.95, EAGLE 3/1/4.
- **Speculative decoding:** MTP = in-checkpoint head (no download); DSpark fetches `RadixArk/Qwen3.8-27B-DSpark` (~2.7 GB) once. Numbers: [Measured on this box](#measured-on-this-box), not the older 16–21 tok/s wall-time figures. MTP 3/1/4 is measured-optimal here (steps sweep 2→12.8, 3→17.2, 4→16.8, 5→16.3, 6→15.8); NGRAM (~30% under MTP) and prefill CUDA graphs were rejected. DSpark's draft was trained on FP8; `QUANT=fp8` did not lift acceptance. If spec decode errors at boot: `--attention-backend triton`.
- **DFlash2 (`start-dflash.sh`):** requires a derived image (DFlash2 merged upstream 2026-08-19, after every released tag) — built automatically from `patch/` on first run. Draft: block-diffusion drafter, pinned `z-lab/Qwen3.8-27B-DFlash2@50307d4` (mirror of `incoai/…`; override `DRAFT_MODEL`/`DRAFT_REVISION`), ~2.6 GB. **Crash history (fixed):** the original head handling dequantized the whole NVFP4 lm_head (~2.5 GB) at draft-graph capture and hard-rebooted the box (0.95, and 0.80 at concurrency ≥ 8–10); the image now runs the quantized head in place (`lm_head.quant_method.apply`) — no big allocation, no capture spike. The 0.90/16 profile boot-**verified** 2026-08-19 (no reboot; ladder in [Measured on this box](#measured-on-this-box)). Operational: NVFP4 at `--mem-fraction-static 0.90` + `MAX_CONCURRENT_REQUESTS=16`; keep `YARN=0`/`CONTEXT_LENGTH=262144` (same draft-config leak as DSpark); unmeasured: bf16 target, long context. 2026-08-19 A/B (minimal 5-file overlay vs whole-tree, same base digest): 61.1/28.4 vs 50.9/25.4 (+20%/+12%), confounded (fresh boot, conc 10 vs 4); details: `bench/_ab-dflash2/SUMMARY.md` (local).
- **CPU pinning (GB10 is big.LITTLE):** container pinned to the ten 3.9 GHz Cortex-X5 cores (`5-9,15-19`); the ten 2.8 GHz A725 cores (`0-4,10-14`) stay free. Without pinning, scheduler/tokenizer processes land on little cores ~half the time. Measured +2–7% decode. Override with `CPUSET` (empty = off).
- **GDN state pool (throughput):** `--max-mamba-cache-size` = concurrency × S; S=4 for `extra_buffer_lazy` + overlap scheduler (no accuracy cost; verified in this build's `kv_cache_configurator.py` — the verify window is a separate buffer, so folding draft tokens in (×8) over-provisions 2×). Default 10 → 40 slots (~3.1 GB at BF16, 78.4 MB/slot). `--max-running-requests` pins the scheduler cap (spec decode otherwise resets it to 48); verify after boot. `MAMBA_SKIP_DECODE_LOCK=1` drops S to 3. The stock `--mamba-full-memory-ratio 0.9` over-provisions KV and clamps concurrency; pinned at 4.21 instead.
- **Context (up to 1M with YaRN):** `.env` `YARN` / `CONTEXT_LENGTH` (262144..1000000). YaRN is applied above 262K (auto-on at exactly 1M), factor = round(len/262144) — 524288→2.0, 1000000→4.0 (the card's validated points). `start.sh` passes the `rope_parameters` override + `--context-length` + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` (this build requires it); verify with `grep context_len .sglang.log`. **Not compatible with DSpark/DFlash2** (the override leaks into the draft config and crashes the rope validator: `AttributeError: … max_position_embeddings`) — keep `YARN=0`/262K for both; YaRN is MTP-only.
- **KV cache:** explicit `--kv-cache-dtype fp8_e4m3` (the NVFP4 checkpoint declares FP8 KV anyway; the flag keeps FP8 KV if you switch quant). ~32.8 KB/token, so a full 1M sequence ≈ 33 GB. Measured pool 2.48M tokens ≈ 81 GB: two 1M requests fit simultaneously; a third is admitted as KV frees.
- **Vision:** the model is a native VLM; SGLang serves the vision tower live (image + video input supported out of the box).

### Measured on this box

**Probe × engine — the one canonical table** (DSpark/MTP measured 2026-08-18, DFlash2 2026-08-19):

| Probe | DSpark (`./start-dspark.sh`, block-7) | MTP (`./start.sh`, EAGLE 3/1/4) | DFlash2 (`./start-dflash.sh`, NVFP4 target) |
|---|---|---|---|
| Code — LRUCache + small test (`bench/ndec.py`, n=5 all) | **51.5 tok/s** (51.4–51.7; `c2` always 518) | **34.5 tok/s** (34.5–34.6; `c2` always 508) | **50.9 tok/s** (50.8–51.1; `c2` always 600) |
| Short chat — “what is a hash map…” (stream) | 22.0 / 21.3 / **23.2** (T=0 off · T=1 off · **T=1 thinking on**) | 24.6 / 23.4 / **21.0** | **31.7 / 28.9 / 66.6** (T=0 off · T=1 off · **T=1 thinking on**) |
| Long essay — Babbage → GPUs (`bench/ndec.py`, n=5 all) | **18.3 tok/s** (18.2–18.3) | **24.1 tok/s** (24.1–24.1) | **25.4 tok/s** (25.3–25.4) |

**Counting + methodology (read before comparing):** DFlash2 column is n=5, 2026-08-19 (same single boot: code 50.77–51.07 median 50.90; essay 25.34–25.39 median 25.39; `c2` always 600 — the earlier n=2 run sits inside these ranges and is superseded); DSpark/MTP are n=5 from the original 2026-08-18 session — same probes, same box, different day, so cross-column deltas are indicative, not a race. Within a day, code deltas <15% are still noise, so call it “ties DSpark”. Short-chat counting: the DFlash2 numbers are taken from the server's own `completion_tokens` (`stream_options.include_usage`), post-first-token — this DFlash2 image batches several tokens per SSE event (~3.75 on average at a fixed ~8 events/s cadence), so a client that counts events as tokens reads ~9 “tok/s”; the DSpark/MTP cells were recorded with that same event-counting script on the older stock image and stand as recorded (events ≈ single tokens there). Non-streamed same-prompt sanity check agrees (28.4 tok/s overall incl. TTFT at T=0).

Engine notes that do **not** repeat the table: DSpark **increases coding speed (~1.5× on LRUCache)**; the MTP prose win is the **long essay**; DSpark slightly beat MTP on default chat (thinking on) with no degrade; thinking-off chat was a small MTP edge. Block sweep: block-7 is the code peak; block-5 is **+8% prose / −16% code**. `--speculative-accept-threshold-acc <1` hurt — leave at 1.0.

**MTP-era wall-time** (`./bench/bench.sh`, includes prefill; not comparable to the table above): thinking 17.2–20.5 tok/s, non-thinking 21.6–22.7, tool-call 26–28. TTFT on a fresh ~16K prompt ~8.3 s warm / ~13 s first boot (Triton warmup). MTP step sweep peaked at 3/1/4 (see above).

**DFlash2 detail (2026-08-19, same single boot — raw runs behind the table above):** ndec code 50.77–51.07, essay 25.34–25.39 (`c2` always 600; DSpark/MTP code `c2` 518/508). Chat raw: 31.6–31.8 / 26.4–34.3 / 45.6–80.2 (T0 off · T1 off · T1 think). Wall-time (non-streamed, incl. prefill — a different clock): thinking 18.8–19.7, non-thinking 22.8–24.1, tool-call 27.3–30.8; TTFT on a fresh ~16K prompt 8.2 s warm. Not measured: bf16 target, long context. Same caveats as the counting note above — different-day vs DSpark/MTP and pre-fix probes (stable at 0.80/4, reboots at 0.80/8–10) mean one replication boot before switching the default remains the honest recommendation.

**Concurrency ladder — DFLASH2 NVFP4 (2026-08-19, the post-fix 0.90/16 boot, synthetic structural-decode fixture; aggregate is total across streams, stream is per-client):**

| Streams | TTFT | Aggregate tok/s | Per-stream tok/s |
|---|---|---|---|
| 1 | 127 ms | 56.6 | 56.6 |
| 2 | 202 ms | 58.4 | 42.4 |
| 4 | 224 ms | 111.6 | 33.4 |
| 8 | 280 ms | 184.9 | 30.8 |
| 16 | 4.18 s | 227.6 | 28.2 |

This run is the first post-fix boot at the 0.90 / 16 profile and it completed all 16 streams **without a reboot** — the draft-capture crash is gone. Per-stream throughput degrades gracefully as concurrency rises (56.6 → 28.2 tok/s); aggregate scales to **227.6 tok/s at 16 concurrent**. TTFT holds at 0.13–0.28 s through ×8 and jumps to **4.18 s at ×16** (16-way admission on this box). One boot, one fixture — indicative, not a guarantee; replicate before relying on it. Another clock again: not comparable to the ndec/stream/table rows.

Run-to-run variance is ~±1.5 tok/s (±7%) on those wall-time numbers. The box **drifts** (essay 19.5 → 18 tok/s over ~an hour of heavy benching — power-cap). The LRUCache two-call is window-dependent (same boot 44–51 by cap); treat code deltas **&lt;15% as noise**. The essay probe (±1% within a boot) is the A/B discriminator. Re-baseline in-session; do not compare across hours. The next step-change needs a newer `lmsysorg/sglang:qwen38-27b` image.

## Thinking & tool calling

- **Thinking mode is ON by default** — the chat template defaults `enable_thinking=true` and `preserve_thinking=true` (the full reasoning trace is retained across turns; good for agents and KV reuse). `--reasoning-parser qwen3` surfaces `<think>…</think>` as `reasoning_content` instead of inline text. Depth is tunable per request with `reasoning_effort=xhigh|medium|low` (xhigh default).
- **Sampling defaults** come from the checkpoint's `generation_config.json` (`--sampling-defaults model`): thinking mode wants `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0`.
- **Tool calling** needs no extra SGLang flag (unlike vLLM's `--enable-auto-tool-choice`): `--tool-call-parser qwen3_coder` decodes the template's `<tool_call><function=…>/<parameter=…>` payload into structured `tool_calls`. Just send `tools` in the request. (The hermes parser expects a different payload and would never parse.)

## Using the API

OpenAI-compatible base URL: `http://127.0.0.1:8888/v1` (model name: `qwen3.8-27b-sglang`).

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-sglang",
    "messages": [{"role": "user", "content": "Explain YaRN in two sentences."}]
  }'
```

Non-thinking / instruct request (per the model card):

```json
{
  "model": "qwen3.8-27b-sglang",
  "messages": [{"role": "user", "content": "Write a haiku about GB10."}],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "presence_penalty": 1.5,
  "chat_template_kwargs": { "enable_thinking": false }
}
```

SGLang also serves an **Anthropic-compatible** endpoint at `http://127.0.0.1:8888/v1/messages` — for Claude Code, set `ANTHROPIC_BASE_URL=http://127.0.0.1:8888` (no `/v1` suffix; Claude Code appends it). The same parser flags apply there. Coding agents that speak plain OpenAI (OpenCode, Pi, …) point at `/v1` and use the served model name.

## Logs & troubleshooting

- Tail the server log: `tail -f .sglang.log` (or `docker logs -f qwen3.8-27b-sglang`)
- After a DSpark boot: `grep -oE "speculative_algorithm='[^']+'|speculative_dspark_block_size=[0-9]+|context_len=[0-9]+" .sglang.log | tail -3` — expect `DSPARK`, block `7`, `262144`
- After a DFlash2 boot: `grep -oE "speculative_algorithm='[^']+'|speculative_draft_model_path='[^']+'|speculative_num_draft_tokens=[0-9]+" .sglang.log | tail -3` — expect `DFLASH`, `incoai/…-DFlash2`, `8`. Also grep for `Initialized DFLASH draft runner` and `DFLASH selector decode … folded into the draft cuda graph` (if you see `kept eager (reason=quantized lm_head)` on a boot, the dequant patch is not in the image — rebuild)
- If the GB10 hard-reboots (kernel log: `task sglang::schedul … blocked …` / `journald … Under memory pressure`) right after `Capture target verify CUDA graph end`, it was DFlash2 + a too-high `--mem-fraction-static` (0.95) at draft-graph capture; relaunch DFlash2 at 0.90. **Root cause found 2026-08-19:** the crashes (0.95, and 0.80 at concurrency ≥ 8–10) were caused by the old dequant-once head handling materializing the full dense NVFP4 lm_head (~2.5–5 GB) during graph capture — fixed in the image (in-place `lm_head.quant_method.apply` selector; no dense dequant). If you still see this signature, it's not the mem fraction per se. If you need mixed-chat benchmarks, run them when nothing else is loaded.
- DFlash2 SSE streams emit ~8 events/s regardless of throughput (measured 2026-08-19: median 126 ms between events) and, on this newer image, each event carries several tokens (~3.75 on average) — so event-counting a DFlash2 stream under-reads tok/s ~4×. Always count `completion_tokens` (`stream_options.include_usage`) for real rates; the DSpark/MTP-era short-chat cells used an event-counting script on the older stock image, where events ≈ single tokens
- If a DFlash2 request dies with `DFlash2 selector requires a dense FP16/BF16/FP32 target lm_head` on an NVFP4 target, the image lacks the quantized-head selector support — rebuild it with `patch/build-dflash2-image.sh` and run that image
- `start.sh` / `start-dspark.sh` print the last 200 log lines and exit if the container dies before becoming ready
- Terminal output filters the harmless per-layer “Enabled fused SiLU+mul+FP4-quant…” notices; `.sglang.log` keeps everything
- Concurrency check: `grep max_running_requests .sglang.log` — should equal your `MAX_CONCURRENT_REQUESTS` (default 10), not a lower clamped value
- Mamba pool check: `grep max_mamba_cache_size .sglang.log` — expect `MAX_CONCURRENT_REQUESTS × 4`
- First long prefill after a cold boot is slow (~13 s for a fresh 16K prompt vs ~8 s warm) — that's Triton kernel warmup, not a regression; the `.cache/triton` volume persists it across restarts
- If startup dies with `AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'`, you're using DSpark with `YARN=1` / `CONTEXT_LENGTH=1000000` — the YaRN override leaks into the draft config. Keep `YARN=0` and `CONTEXT_LENGTH=262144` for DSpark (see Context note)
- First start downloads ~22 GB of weights (plus ~2.7 GB DSpark draft model if you switch to DSpark); subsequent starts reuse `./.cache/huggingface`

## Repository layout

```
.
├── start.sh          # EAGLE/MTP engine (port 8888); tracked
├── stop.sh           # stops whichever engine is up; tracked
├── start-dspark.sh   # DSpark engine (port 8888); whitelisted for versioning
├── start-dflash.sh   # DFlash2 engine (port 8888, bf16 or NVFP4 target); auto-builds the image from patch/ on first run; tracked
├── patch/             # DFlash2 image build machinery (build-dflash2-image.sh, dflash2_nvfp4_head.patch, overlay-dflash2/); tracked
├── bench/
│   ├── bench.sh     # essay / tool-call wall-time bench + TTFT probe
│   └── ndec.py      # two-call net-decode (LRUCache + essay); engine A/B (any engine)
├── .env             # live config (context / concurrency / quant / tuning); not tracked by git
├── .env.sample      # tracked template — copy to .env to configure
├── .gitignore       # whitelist: start scripts, bench/, README, CHANGELOG, .env.sample, LICENSE
├── LICENSE          # MIT
└── README.md        # tracked
```

Experiment write-ups are local-only (untracked): `DS4F.md`, `KIMI.md`, `GROK.md`, `TIER_A_RESULTS.md`, `TIER_B_RESULTS.md`, `TIER_C_RESULTS.md`, `HANDOFF.md`.

## Notes

- `QUANT` values: `nvfp4` → `RadixArk/Qwen3.8-27B-NVFP4`, `fp8` → `Qwen/Qwen3.8-27B-FP8`, `bf16` → `Qwen/Qwen3.8-27B` (all fit in the Spark's 128 GB).
- `SERVED_MODEL_NAME`, `IMAGE`, `CONTAINER_NAME`, `PORT` are set inline in `start.sh` (not `.env`).

## Credits

- [SGLang cookbook — Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) — the DGX Spark serving recipe, MTP and GDN state-pool guidance
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) — YaRN 1M-context SGLang recipe and sampling recommendations
- [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) — NVFP4 W4A4 checkpoint (FP8 KV calibration scales)
- [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) — the DSpark draft model used by `start-dspark.sh`
- [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) / [z-lab mirror](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) — the DFlash2 block-diffusion drafter used by `start-dflash.sh` (trained against the bf16 `Qwen/Qwen3.8-27B`)
- [inco.ai/blog/dflash2](https://inco.ai/blog/dflash2/) — DFlash2 write-up; its SGLang serving recipe is what `start-dflash.sh` pins
- [SGLang DFLASH2 commit](https://github.com/sgl-project/sglang/commit/c14312a66) — upstream mainline DFlash2 support, merged after every released image; this repo's derived-image build tracks it
- [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38) — the published DSpark-on-GB10 config (same pinned image) that the DSpark flag stack builds on
- [SGLang](https://github.com/sgl-project/sglang) — inference engine and OpenAI/Anthropic-compatible server
