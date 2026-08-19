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

**DSpark and DFlash2 are faster on code.** Versus MTP, DSpark gives the essay back; DFlash2 does not. Everyday chat on the same streamed probe comes out a DFlash2 win once tokens are counted right: **31.7 / 28.9 / 66.6** vs DSpark 22.0 / 21.3 / 23.2 and MTP 24.6 / 23.4 / 21.0 (see the counting note under the table — the DFlash2 cells are counted from the server's `completion_tokens`, because on this newer image SGLang batches several tokens per SSE event and a naive chunk-count reads ~9 “tok/s”):

| Probe | DSpark (`./start-dspark.sh`, block-7) | MTP (`./start.sh`, EAGLE 3/1/4) | DFlash2 (`./start-dflash.sh`, NVFP4 target) |
|---|---|---|---|
| Code — LRUCache + small test (`bench/ndec.py`, n=5 all) | **51.5 tok/s** (51.4–51.7; `c2` always 518) | **34.5 tok/s** (34.5–34.6; `c2` always 508) | **50.9 tok/s** (50.8–51.1; `c2` always 600) |
| Short chat — “what is a hash map…” (stream) | 22.0 / 21.3 / **23.2** (T=0 off · T=1 off · **T=1 thinking on**) | 24.6 / 23.4 / **21.0** | **31.7 / 28.9 / 66.6** (T=0 off · T=1 off · **T=1 thinking on**) |
| Long essay — Babbage → GPUs (`bench/ndec.py`, n=5 all) | **18.3 tok/s** (18.2–18.3) | **24.1 tok/s** (24.1–24.1) | **25.4 tok/s** (25.3–25.4) |

The DFlash2 column is **n=5, 2026-08-19** (same single boot: code 50.77–51.07, median 50.90; essay 25.34–25.39, median 25.39; `c2` always 600 for both — the earlier n=2 run, 50.86/50.76 and 25.27/25.30, sits inside these ranges and is superseded); the DSpark/MTP columns are n=5 from the original 2026-08-18 session — same probes, same box, different day. Within a day, code deltas <15% are still noise, so call it “ties DSpark”. The essay number (25.4) is what's interesting: it's on the MTP side of the table. The short-chat cells need a counting note: the DFlash2 numbers above are taken from the server's own `completion_tokens` (`stream_options.include_usage`), post-first-token, n=5 — because this DFlash2 image (newer mainline than the DSpark/MTP image) batches several tokens per SSE event (~3.75 on average, at a fixed ~8 events/s cadence), and a client that counts events as tokens reads ~9 “tok/s”. The DSpark/MTP cells were recorded with that same event-counting script on the older stock image and stand as recorded (on that image events ≈ single tokens). Re-measured today, DFlash2's true streamed decode is 31.7 / 28.9 / 66.6 — faster than both other columns on every chat condition, and the non-streamed same-prompt sanity check agrees (28.4 tok/s overall incl. TTFT at T=0).

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

`start-dflash.sh` is self-contained but different: no released SGLang image has DFlash2 support yet (it merged upstream 2026-08-19, after every published tag including the pinned `qwen38-27b`), so on a machine that doesn't already have the local `lmsysorg/sglang:qwen38-27b-dflash2` image the script **builds it** (needs git + network once) via `build-dflash2-image.sh`, which overlays the mainline python tree at that commit plus the `dflash2_nvfp4_head.patch` (quantized-head selector support for the NVFP4 target via `lm_head.quant_method` — no dense dequant; a dequant-once approach hard-rebooted this box at graph capture) onto the pinned GB10 image. First boot then pulls the ~2.7 GB draft if missing.

All start scripts are idempotent: if the container is already running they say so and exit; if a stopped container exists they remove it first. `./stop.sh` stops whichever engine is up.

## Scripts

| Script | What it does |
|---|---|
| `start.sh` | Launches the SGLang container (`docker run -d`, host network, `--shm-size 32g`), streams logs to `.sglang.log`, records the container ID in `.sglang.pid`, and polls `http://127.0.0.1:8888/v1/models` until the server is ready. **EAGLE/MTP speculative decoding** (`SPEC_STEPS/SPEC_TOPK/SPEC_DRAFT = 3/1/4`). |
| `start-dspark.sh` | Same service, **DSpark** instead of EAGLE: block-7 / unquant draft, torch.compile + decode-graph caps, `--num-continuous-decode-steps 2`, mem 0.90. Thin wrapper (`EXTRA_ARGS` → `start.sh`). **Code 51.5 vs MTP 34.5. Default chat ~23 vs ~21. Long essay 18.3 vs 24.1.** |
| `start-dflash.sh` | Same service, **DFlash2** block-diffusion draft (default `z-lab/Qwen3.8-27B-DFlash2@50307d4`, pinned — same draft as the `incoai/…` mirror; `DRAFT_MODEL`/`DRAFT_REVISION` env-overridable). **Default target: NVFP4** `RadixArk/Qwen3.8-27B-NVFP4` with `--mem-fraction-static 0.90` and the quantized-head selector patch (`dflash2_nvfp4_head.patch`, beams the NVFP4 head into the DFLASH selector via `lm_head.quant_method.apply` — no dense dequant); `DF_TARGET=bf16` selects `Qwen/Qwen3.8-27B` instead. Needs the local derived image (built automatically on first run; see below). **Measured (NVFP4 target, 2026-08-19, n=5): code 50.9, essay 25.4; streamed chat 31.7 / 28.9 / 66.6 tok/s (usage-counted; see the counting note under the top table); wall-time + TTFT in [Measured on this box](#measured-on-this-box). An A/B against a 5-file minimal overlay (below) measured 61.1 / 28.4 — confounded, see [Measured on this box](#measured-on-this-box).** |
| `build-dflash2-image.sh` | Builds the DFlash2 image (`--full` default: pinned mainline python tree + `dflash2_nvfp4_head.patch`, needs git+network first time; `--minimal`: only the 5 DFlash2 modules from `overlay-dflash2/`, sha256-verified via its `MANIFEST.sha256`, no network). Auto-invoked by `start-dflash.sh` (`--minimal` iff `IMAGE=…-minoverlay`); or run manually to rebuild after bumping the upstream pin. The `--minimal` variant measured **+20% code / +12% essay** in the 2026-08-19 A/B (confounded; see [Measured on this box](#measured-on-this-box)). |
| `bench/ndec.py` | Two-call net-decode A/B (LRUCache + essay, thinking off). How the DSpark vs MTP numbers above were measured. Run twice; trust the second; treat code deltas &lt;15% as noise. |
| `bench/bench.sh` | Essay / tool-call **wall-time** bench + 16K TTFT probe (includes prefill). Different clock from `ndec.py`. |
| `stop.sh` | Stops the serving engine (idempotent; also cleans up any experiment processes still alive). Leaves the stopped container in place for `docker logs` post-mortem. |

Runtime artifacts: `.sglang.log` (server log), `.sglang.pid` (container ID), `.cache/` (HF + Triton caches). All are git-ignored.

> Whitelisted for tracking: `start.sh`, `start-dspark.sh`, `stop.sh`, `start-dflash.sh`, `build-dflash2-image.sh`, `overlay-dflash2/`, `dflash2_nvfp4_head.patch`, `bench/`, `README.md`, `CHANGELOG.md`, `.env.sample`, `LICENSE`, `.gitignore`. Experiment scripts and analysis docs stay untracked by design.

## Which engine to use

**EAGLE/MTP and DSpark** are the same NVFP4 27B on the same `lmsysorg/sglang:qwen38-27b` image — only the speculative decoder changes. **DFlash2 (`start-dflash.sh`)** defaults to the same NVFP4 weights, on a derived image (see [Scripts](#scripts) / Configuration below) that exists only because DFlash2 support is newer than any released image. DFlash2 can also target the bf16 base (`DF_TARGET=bf16`) — not benched.

| | `./start-dspark.sh` (DSpark block-7) | `./start.sh` (EAGLE/MTP 3/1/4) | `./start-dflash.sh` (DFlash2, NVFP4) |
|---|---|---|---|
| Code (LRUCache, `bench/ndec.py`, n=5 all) | **51.5 tok/s** | **34.5 tok/s** | **50.9 tok/s** |
| Short chat (stream, thinking on / UI default) | **23.2 tok/s** | **21.0 tok/s** | **66.6 tok/s** (45.6–80.2, bursty) |
| Long essay (`bench/ndec.py`, n=5 all) | **18.3 tok/s** | **24.1 tok/s** | **25.4 tok/s** |
| Best for | agents, code, tools, **normal chat** (**default here**) | long-form writing | code AND long-form writing (essay holds; chat is a win too) |
| Memory | 22 GB target + ~2.7 GB draft, mem 0.90 | 22 GB target (in-checkpoint MTP), mem 0.95 | NVFP4: 22 GB target + ~2.6 GB draft, **mem 0.90**; bf16: 52 GB weights |

DSpark/MTP columns are live 2026-08-18 evening. DFlash2 was benched 2026-08-19 (n=5, same single boot, thinking off, T=0 — the same `bench/ndec.py` probes; short-chat and wall-time now measured too). **Take-aways, with the same-boot/different-day caveat firmly in mind:** DFlash2 on NVFP4 ties DSpark on code (50.9 vs 51.5 — inside the <15% noise band), *beats MTP* on the essay (25.4 vs 24.1), and *beats both on every short-chat condition* (31.7 / 28.9 / 66.6 once counted from `completion_tokens` — the earlier ~9.2 “tok/s” reading for this column was an SSE event-counting artifact on the newer image, see the top-table note). Watchpoints: `mem-fraction-static 0.95` + DFlash2 wedged the box once (hard reboot; see [Logs & troubleshooting](#logs--troubleshooting)) — now fixed (see the DFlash2 bullet); current target profile is `0.90` + 16 concurrent, pending a validation boot. bf16 target power/bandwidth is entirely unmeasured. Do not compare these to sparkDash fill-to-max streams.

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
| `DF_TARGET` | `nvfp4` | `nvfp4` (default) → `RadixArk/Qwen3.8-27B-NVFP4` (+`--mem-fraction-static 0.90`); `bf16` → `Qwen/Qwen3.8-27B`. The NVFP4 path requires the quantized-head selector patch baked into the derived image (see `dflash2_nvfp4_head.patch`); without it, DFLASH dies at the first request (see [Scripts](#scripts)). |
| `DF_EXTRA` | — | Extra SGLang flags appended AFTER the base DFlash2 stack (last-wins). E.g. `DF_EXTRA="--mem-fraction-static 0.90" ./start-dflash.sh` |
| `IMAGE` | `lmsysorg/sglang:qwen38-27b-dflash2` (local) | The derived image; the script builds it if missing (auto-invokes `build-dflash2-image.sh`). Override with `IMAGE=tag` to use your own build — e.g. `IMAGE=lmsysorg/sglang:qwen38-27b-dflash2-minoverlay` (built via `build-dflash2-image.sh --minimal`) |
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

- **Recipe vs this repo:** the cookbook DGX Spark + NVFP4 + DSpark cell uses mem **0.85**, chunk **2048**, GDN **float32**, radix `extra_buffer`, and does not pin block-7 / compile / decode graphs. We measured those defaults (float32 **−3%**; FP8 target **~30% slower** than NVFP4). Live DSpark stack: mem **0.90**, chunk **8192**, GDN **bf16**, `extra_buffer_lazy`, block **7** / 8 draft tokens, torch.compile + `--cuda-graph-max-bs-decode 4`, `--num-continuous-decode-steps 2`, `--disable-prefill-cuda-graph`, flashinfer, FP8 KV. `start.sh` MTP still uses mem **0.95** and EAGLE 3/1/4.
- **Speculative decoding:** MTP (`./start.sh`) is the in-checkpoint head — no second download. DSpark (`./start-dspark.sh`) fetches `RadixArk/Qwen3.8-27B-DSpark` (~2.7 GB) once. Current-era numbers are the table at the top (**51.5 vs 34.5 code, default chat 23.2 vs 21.0, essay 18.3 vs 24.1**; DFlash2's column is in the same table), not the older wall-time 16–21 tok/s figures. The DSpark draft was trained on FP8; `QUANT=fp8` did not lift acceptance here. If spec decode errors at boot, try `--attention-backend triton`.
  **MTP 3/1/4 is measured-optimal** on this silicon (thinking, `bench/bench.sh`-style sweep): steps 2 → 12.8, **3 → 17.2**, 4 → 16.8, 5 → 16.3, 6 → 15.8. Also rejected: **NGRAM** (12.7–15.7, ~30% under MTP even on tool-call-style output) and **prefill CUDA graphs** (this build auto-disables them: GDN layers ≠ standard GQA).
- **DFlash2 (`start-dflash.sh`)**: mainline support merged 2026-08-19 (`c14312a66`) — *after every released image* — so this repo carries a derived local image + `dflash2_nvfp4_head.patch` to make the NVFP4 checkpoint usable (its `lm_head` is NVFP4-quantized). **2026-08-19 root-cause note:** the patch originally dequantized the whole NVFP4 head once into a dense copy (~2.5 GB bf16, ~5 GB peak) at draft-graph capture — which wedged the GB10 at 0.95 and again at 0.80 with any concurrency ≥ 8–10 (the “hard reboot right after Capture target verify CUDA graph end” signature). The patch now runs the quantized head in place via `lm_head.quant_method.apply` (like the minimal-overlay variant does — no big allocation), so the whole-tree image no longer carries that capture-time spike. The draft is a **block-diffusion drafter** (default pinned to `z-lab/Qwen3.8-27B-DFlash2@50307d4`, mirror of `incoai/…`; `DRAFT_MODEL`/`DRAFT_REVISION` override) that shares the target's embed/lm_head; it ships ~2.6 GB and fetched into the same cache. Measured NVFP4 (2026-08-19, n=5): code 50.9 / essay 25.4 tok/s — ties DSpark on code, clears MTP on the essay; streamed short chat 31.7 / 28.9 / 66.6 tok/s, counted from `completion_tokens` (this newer image emits ~8 SSE events/s with ~3.75 tokens/event, so naive event-counting under-reads ~4×; the DSpark/MTP-era cells used event-counting on the older image and stand as recorded). Not yet measured: bf16 target, long context. Operational: `--mem-fraction-static 0.90` + `MAX_CONCURRENT_REQUESTS=16` for the NVFP4 path (0.95 wedged the box once before the patch fix; 0.80/0.85 were interim) and leave YaRN/`CONTEXT_LENGTH=262144` as-is (same draft-config leak as DSpark).
  **2026-08-19 internal A/B of the two image builds** (same base digest `febfb971…`, same NVFP4 target `554ebba9…`, same probes): the **minimal 5-file overlay** (`build-dflash2-image.sh --minimal` + `overlay-dflash2/`, sha256-verified via its `MANIFEST.sha256`) measured **code 61.1 / essay 28.4 tok/s** vs the whole-tree image's 50.9 / 25.4 (+20% / +12%) at the same mem 0.80. Confounds: fresh boot + concurrency 10 vs the baseline's 4, so a same-day matched-concurrency A/B is still owed before swapping the default image. Operational warning from that session: relaunching the whole-tree image at `MAX_CONCURRENT_REQUESTS=10` hard-rebooted the GB10 — since root-caused to the *old* dequant-once patch's ~2.5–5 GB head allocation during draft-graph capture (see the DFlash2 bullet); the minimal-overlay image (no such allocation) booted and benched fine at 10, and the whole-tree image at concurrency 4 never crashed. The fix landed 2026-08-19 in `dflash2_nvfp4_head.patch` — the whole-tree image no longer dequantizes; target profile is now 0.90 / 16 concurrent, to be boot-validated. Details: `bench/_ab-dflash2/SUMMARY.md` (local-only).
- **CPU pinning (GB10 is big.LITTLE):** the container is pinned to the ten 3.9 GHz Cortex-X5 cores (`5-9,15-19`) via `--cpuset-cpus`; the ten 2.8 GHz Cortex-A725 cores (`0-4,10-14`) stay free for everything else on the host. Without pinning, the scheduler/tokenizer Python processes float across all 20 cores and land on little cores ~half the time. Measured +2–7% decode. Override with `CPUSET` (empty string = off).
- **GDN state pool (throughput):** hybrid GDN models reserve a state pool that sets the concurrency ceiling; the default `--mamba-full-memory-ratio 0.9` over-provisions KV and silently clamps concurrency. Pinned instead: `--max-mamba-cache-size` = **concurrency × S**, where S=4 for `--mamba-radix-cache-strategy extra_buffer_lazy` + the overlap scheduler (no accuracy cost). This is the engine's own formula — verified in this build's `kv_cache_configurator.py`: the pool is divided by S alone and the speculative verify window is sized as a **separate** buffer, so folding draft tokens into the pin (early revisions here did: `× 8`) over-provisions the pool 2×. Default 10 requests → 40 slots (~3.1 GB at `--mamba-ssm-dtype bfloat16`'s 78.4 MB/slot; fp32 would be 153.9 MB). `--max-running-requests` pins the scheduler cap to match (spec decode otherwise resets it to 48). After boot, check the `max_running_requests` line in `.sglang.log`. `MAMBA_SKIP_DECODE_LOCK=1` drops S to 3 if you need the headroom.
- **Context: native 262,144 tokens, up to a validated 1M with YaRN.** In `.env` (or as exports): `YARN=0|1` and `CONTEXT_LENGTH` (range 262144..1000000). YaRN rope scaling is applied when `CONTEXT_LENGTH` > 262144 and either `YARN=1` or the length is exactly `1000000` (so 1M works out of the box); the factor is derived as round(`CONTEXT_LENGTH`/262144) — 524288 → 2.0, 1000000 → 4.0, the model card's validated values. `start.sh` then passes `--json-model-override-args` (the `rope_parameters` block under `text_config`) + `--context-length` + `-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, which this SGLang build requires — without it, it logs "User-specified context_length (...) is greater than the derived context_length" and stays at 262K. Verify after boot: `grep context_len .sglang.log`. The override is **not compatible with DSpark** on this build (the draft `ModelConfig` inherits it, injecting a `text_config` dict into the draft's flat config and crashing transformers' rope validator with `AttributeError: ... 'max_position_embeddings'`), so keep `YARN=0`/`CONTEXT_LENGTH=262144` for DSpark; YaRN sits on top of the default MTP setup.
- **KV cache:** explicit `--kv-cache-dtype fp8_e4m3`. The NVFP4 checkpoint declares FP8 KV anyway (`auto` would honor it with its calibration scales); the explicit flag keeps FP8 KV if you switch to `QUANT=bf16|fp8`. At ~32.8 KB/token, a full 1M-token sequence costs ~33 GB of KV. Measured on this box (pool = 2.48M tokens ≈ 81 GB): **two** full-length 1M requests fit simultaneously and a third is admitted as KV frees.
- **Vision:** the model is a native VLM and SGLang serves the vision tower live (image + video input supported out of the box).

### Measured on this box

**Engine A/B (2026-08-18):**

| Probe | Clock | DSpark block-7 | MTP 3/1/4 |
|---|---|---|---|
| Code — LRUCache + OrderedDict + small test | `bench/ndec.py` two-call, T=0, thinking off, n=5 | **51.5 tok/s** (51.38–51.73; `c2` always 518) | **34.5 tok/s** (34.46–34.57; `c2` always 508) |
| Short chat — *What is a hash map, and when would I use one instead of a list?* | stream, post-first-token | **22.0** T=0 off · **21.3** T=1 off · **23.2** T=1 think on | **24.6** T=0 off · **23.4** T=1 off · **21.0** T=1 think on |
| Long essay — history of computing, Babbage → GPUs | `bench/ndec.py` two-call, T=0, thinking off, n=5 | **18.3 tok/s** (18.18–18.29) | **24.1 tok/s** (24.05–24.13) |

DSpark **increases coding speed (~1.5× on LRUCache)**. The MTP prose win is the **long essay** (24.1 vs 18.3). Default UI chat (thinking on) was **23.2 DSpark vs 21.0 MTP** — no degrade; DSpark was slightly faster. Thinking-off chat is a small MTP edge (~24 vs ~22). Block sweep: block-7 is the code peak; block-5 is **+8% prose / −16% code**. `--speculative-accept-threshold-acc <1` hurt — leave at 1.0.

**MTP-era wall-time** (`./bench/bench.sh`, includes prefill; not comparable to the table above): thinking 17.2–20.5 tok/s, non-thinking 21.6–22.7, tool-call 26–28. TTFT on a fresh ~16K prompt ~8.3 s warm / ~13 s first boot (Triton warmup). MTP step sweep peaked at 3/1/4 (see above).

**DFlash2, NVFP4 target (2026-08-19 — `bench/ndec.py` n=5 + short-chat + wall-time, all same single boot, thinking off/T=0 for the ndec probes):** code **50.9 tok/s** (50.77–51.07, median 50.90; `c2` always 600), essay **25.4 tok/s** (25.34–25.39, median 25.39). The earlier n=2 numbers (code 50.86/50.76, essay 25.27/25.30) are superseded — same boot, n=5 just tightens the range and adds the `c2=600` token counts (DSpark/MTP code were 518/508). Short chat (streamed, same prompt “What is a hash map…”, max_tokens 300, n=5, post-first-token, counted from the server's `completion_tokens` via `stream_options.include_usage`): **31.7 / 28.9 / 66.6 tok/s** (T=0 off · T=1 off · T=1 thinking on; raw 31.6–31.8 / 26.4–34.3 / 45.6–80.2). Counting note that matters: on this DFlash2 image the SSE stream runs at a fixed ~8 events/s (median 126 ms between events) with an average of ~3.75 tokens per event, so a naive client that counts SSE events as tokens reads ~8–9 “tok/s” — that earlier reading (9.2/9.1/9.2, from the same event-counting script the DSpark/MTP cells were recorded with) is superseded as a counting artifact, not a real throughput number. Non-streamed sanity check, same prompts: 28.4 tok/s overall incl. TTFT (T=0, 300 tok), 24.0 (thinking on, 400 tok) — consistent with the streamed figures. Wall-time (`bench/bench.sh`, non-streamed, includes prefill — a different clock from the table): thinking **18.8–19.7 tok/s** (3 runs: 19.7/18.8/19.0), non-thinking **22.8–24.1** (23.1/22.8/24.1), tool-call **27.3–30.8** (30.8/27.3/28.6); TTFT on a fresh ~16K prompt **8.2 s warm** (8.18/8.17/8.21; first-boot cold ~13 s per the MTP note). The DSpark/MTP columns are from a *different* day (2026-08-18) — treat the cross-column comparison as indicative, not a race. Not measured for this mode: bf16 target, LongContext (concurrency/mem were partially probed in the A/B session — the honest data pre-fix: stable at 0.80/4, hard-reboot at 0.80/concurrency 8–10; post-fix target profile is 0.90/16 and awaits a validation boot; see the Notable-serving-choices bullet). With n=5 and all three probes plus wall-time now on the same boot, the standing is firmer than the n=2 preliminary: DFlash2 on NVFP4 is the strongest measured config on this box on code, essay, and chat alike — but it is still a different-*day* comparison and concurrency/bf16/LongContext are unmeasured, so one replication boot before switching the default remains the honest recommendation. Everything else below about drift/noise/baselines applies equally.

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
- If the GB10 hard-reboots (kernel log: `task sglang::schedul … blocked …` / `journald … Under memory pressure`) right after `Capture target verify CUDA graph end`, it was DFlash2 + a too-high `--mem-fraction-static` (0.95) at draft-graph capture; relaunch DFlash2 at 0.90. **Root cause found 2026-08-19:** the crashes (0.95, and 0.80 at concurrency ≥ 8–10) were caused by the old dequant-once patch materializing the full dense NVFP4 lm_head (~2.5–5 GB) during graph capture — fixed by `dflash2_nvfp4_head.patch` now using `lm_head.quant_method.apply` instead. If you still see this signature, it's not the mem fraction per se. If you need mixed-chat benchmarks, run them when nothing else is loaded.
- DFlash2 SSE streams emit ~8 events/s regardless of throughput (measured 2026-08-19: median 126 ms between events) and, on this newer image, each event carries several tokens (~3.75 on average) — so event-counting a DFlash2 stream under-reads tok/s ~4×. Always count `completion_tokens` (`stream_options.include_usage`) for real rates; the DSpark/MTP-era short-chat cells used an event-counting script on the older stock image, where events ≈ single tokens
- If a DFlash2 request tool-call dies with `DFlash2 selector requires a dense FP16/BF16/FP32 target lm_head` on an NVFP4 target, the image lacks the dequant patch — `./build-dflash2-image.sh` ships with it and must be the tag you run
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
├── start-dflash.sh   # DFlash2 engine (port 8888, bf16 or NVFP4 target); tracked with the DFlash2 path
├── build-dflash2-image.sh   # builds the DFlash2 image (--full mainline overlay + patch | --minimal 5-file overlay); tracked
├── overlay-dflash2/        # the 5 DFlash2 modules (sha256-verified via MANIFEST.sha256); tracked
├── dflash2_nvfp4_head.patch # quantized-head selector patch for the DFLASH NVFP4 path (replaces a dequant-once approach that hard-rebooted this box); tracked
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
