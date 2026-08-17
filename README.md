# Qwen3.8 27B on SGLang for DGX Spark

[![SGLang](https://img.shields.io/badge/SGLang-cookbook-blue)](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
[![Model](https://img.shields.io/badge/model-Qwen3.8--27B-informational)](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
[![arch](https://img.shields.io/badge/arch-arm64%20%2F%20GB10-lightgrey)](#)

Opinionated, ready-to-run scripts to serve **[Qwen3.8-27B](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)** with **[SGLang](https://docs.sglang.io)** in Docker on an NVIDIA DGX Spark (GB10, aarch64). One script starts an OpenAI-compatible server, one stops it — with every tuning choice measured on-device instead of guessed.

The serving recipe is the **[SGLang cookbook's DGX Spark cell](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)** — the model-specific, validated launch configuration — with the speed and long-context options from the same page turned on.

- **NVFP4 W4A4** checkpoint (default; BF16 and FP8 available via `QUANT=…`)
- **native 262K context, YaRN off, and 10 concurrent requests by default** — optionally extend to a validated 1M via YaRN (see “[Long context & concurrency](#long-context--concurrency-up-to-1m-10-concurrent)”)
- **FP8 KV cache** (`fp8_e4m3`, ~2× KV memory savings; uses the NVFP4 checkpoint's calibration scales)
- **MTP speculative decoding** (the checkpoint's own head; EAGLE 3 steps / topk 1 / 4 draft tokens) — the fastest option measured on this box, and **proven the peak by an on-device sweep** (steps 2–6: 12.8 → **17.2** → 16.8 → 16.3 → 15.8 tok/s; the MTP head is trained for exactly 3 steps). DSpark available as an alternative (see serving notes)
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
./start.sh

# 3. Use it
curl http://127.0.0.1:8888/v1/models

# 4. Stop it
./stop.sh
```

`.env.sample` ships with `YARN=0`, `CONTEXT_LENGTH=262144` (native) and `MAX_CONCURRENT_REQUESTS=10` — so a fresh clone serves **262K context, YaRN off, 10 concurrent** after just the `cp` above. `.env` is the live config (plain `VAR=value` lines read by `start.sh`): shell exports of the same names win, `.env` fills the gaps, start.sh defaults apply last. Changes only take effect on the **next** launch — `./stop.sh && ./start.sh`. For anything above native context (e.g. 1M) or a different concurrency, see [Long context & concurrency](#long-context--concurrency-up-to-1m-10-concurrent).

`start.sh` is idempotent: if the container is already running it says so and exits; if a stopped container exists it removes it first.

## Scripts

| Script | What it does |
|---|---|
| `start.sh` | Launches the SGLang container (`docker run -d`, host network, `--shm-size 32g`), streams logs to `.sglang.log`, records the container ID in `.sglang.pid`, and polls `http://127.0.0.1:8888/v1/models` until the server is ready. |
| `stop.sh` | Stops the container, removes `.sglang.pid`, and leaves the stopped container in place for `docker logs` post-mortem (the next `start.sh` removes it). |

Runtime artifacts: `.sglang.log` (server log), `.sglang.pid` (container ID), `.cache/` (HF + Triton caches). All are git-ignored.

## Configuration

Defaults live at the top of `start.sh`:

| Variable | Default | Notes |
|---|---|---|
| `YARN` | `0` | `0` off / `1` on for `CONTEXT_LENGTH` > 262144; implicitly on at exactly `1000000`. Factor = round(`CONTEXT_LENGTH`/262144) |
| `CONTEXT_LENGTH` | `262144` | Range `262144`..`1000000` (native..1M). Combined with `YARN=1` for values above native; `1M` auto-enables YaRN even with `YARN=0`. Invalid values abort at startup |
| `MAX_CONCURRENT_REQUESTS` | `10` | Sizes `--max-mamba-cache-size` = concurrency × 4 slots and passes `--max-running-requests` |
| `SPEC_STEPS` / `SPEC_TOPK` / `SPEC_DRAFT` | `3` / `1` / `4` | MTP chain drafting; topk=1 requires `SPEC_DRAFT = SPEC_STEPS + 1` (validated at launch). Sweep the pair on your box and pin the winner — 3/1/4 is the measured peak here |
| `CHUNKED_PREFILL` | `8192` | Prefill chunk tokens. `2048` = smoother decode inter-token latency under mixed load (cookbook general advice; the Spark cell is validated at 8192) |
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

- **Recipe (cookbook, DGX Spark cell):** `--mem-fraction-static 0.95`, `--attention-backend flashinfer` (`trtllm_mha` is SM100-only), `--chunked-prefill-size 8192`, `--disable-prefill-cuda-graph`. The Spark's unified memory fits all three checkpoints, and its cells keep the large prefill chunks (the cookbook's general 2048-token advice does not apply here).
- **Speculative decoding (default: MTP):** `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` — the checkpoint's own MTP head; no second download. Alternative: DSpark (`--speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark --speculative-dspark-block-size 7 --speculative-draft-model-quantization unquant`, separate ~2.7 GB checkpoint, fetched automatically). Measured head-to-head on this box (single stream, 400 tok, pre-optimization numbers): MTP **16.9 / 21.0 tok/s** (thinking / non-thinking), DSpark 16.9 / 16.2, DSpark + FP8 target 11.4 / 11.5. MTP wins via far higher per-token acceptance (0.23–0.52 vs DSpark's 0.09–0.26) and a cheaper 4-token verify. The DSpark draft was trained against the FP8 target, but `QUANT=fp8` did not lift acceptance here while costing ~30% decode speed — stay on NVFP4. If either spec-decode path errors at boot, rerun with `--attention-backend triton`.
  **The 3/1/4 default is measured-optimal on this silicon** (thinking tok/s, on-device sweep): steps 2 → 12.8, **3 → 17.2**, 4 → 16.8, 5 → 16.3, 6 → 15.8 — the MTP head is trained for exactly 3 steps; deeper chains only add rejected verify work. Also tested and rejected: **NGRAM** drafting (12.7–15.7 tok/s, ~30% under MTP even on tool-call-style output — the trie has nothing to draft from on generative prose) and **prefill CUDA graphs** (this build auto-disables them on this model: `some layers do not apply Standard GQA` — the GDN layers).
- **CPU pinning (GB10 is big.LITTLE):** the container is pinned to the ten 3.9 GHz Cortex-X5 cores (`5-9,15-19`) via `--cpuset-cpus`; the ten 2.8 GHz Cortex-A725 cores (`0-4,10-14`) stay free for everything else on the host. Without pinning, the scheduler/tokenizer Python processes float across all 20 cores and land on little cores ~half the time. Measured +2–7% decode. Override with `CPUSET` (empty string = off).
- **GDN state pool (throughput):** hybrid GDN models reserve a state pool that sets the concurrency ceiling; the default `--mamba-full-memory-ratio 0.9` over-provisions KV and silently clamps concurrency. Pinned instead: `--max-mamba-cache-size` = **concurrency × S**, where S=4 for `--mamba-radix-cache-strategy extra_buffer_lazy` + the overlap scheduler (no accuracy cost). This is the engine's own formula — verified in this build's `kv_cache_configurator.py`: the pool is divided by S alone and the speculative verify window is sized as a **separate** buffer, so folding draft tokens into the pin (early revisions here did: `× 8`) over-provisions the pool 2×. Default 10 requests → 40 slots (~3.1 GB at `--mamba-ssm-dtype bfloat16`'s 78.4 MB/slot; fp32 would be 153.9 MB). `--max-running-requests` pins the scheduler cap to match (spec decode otherwise resets it to 48). After boot, check the `max_running_requests` line in `.sglang.log`. `MAMBA_SKIP_DECODE_LOCK=1` drops S to 3 if you need the headroom.
- **Context: native 262,144 tokens, up to a validated 1M with YaRN.** In `.env` (or as exports): `YARN=0|1` and `CONTEXT_LENGTH` (range 262144..1000000). YaRN rope scaling is applied when `CONTEXT_LENGTH` > 262144 and either `YARN=1` or the length is exactly `1000000` (so 1M works out of the box); the factor is derived as round(`CONTEXT_LENGTH`/262144) — 524288 → 2.0, 1000000 → 4.0, the model card's validated values. `start.sh` then passes `--json-model-override-args` (the `rope_parameters` block under `text_config`) + `--context-length` + `-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, which this SGLang build requires — without it, it logs "User-specified context_length (...) is greater than the derived context_length" and stays at 262K. Verify after boot: `grep context_len .sglang.log`. The override is **not compatible with DSpark** on this build (the draft `ModelConfig` inherits it, injecting a `text_config` dict into the draft's flat config and crashing transformers' rope validator with `AttributeError: ... 'max_position_embeddings'`), so keep `YARN=0`/`CONTEXT_LENGTH=262144` for DSpark; YaRN sits on top of the default MTP setup.
- **KV cache:** explicit `--kv-cache-dtype fp8_e4m3`. The NVFP4 checkpoint declares FP8 KV anyway (`auto` would honor it with its calibration scales); the explicit flag keeps FP8 KV if you switch to `QUANT=bf16|fp8`. At ~32.8 KB/token, a full 1M-token sequence costs ~33 GB of KV. Measured on this box (pool = 2.48M tokens ≈ 81 GB): **two** full-length 1M requests fit simultaneously and a third is admitted as KV frees.
- **Vision:** the model is a native VLM and SGLang serves the vision tower live (image + video input supported out of the box).

### Measured on this box (after tuning)

| Metric | Value |
|---|---|
| Single-stream decode, thinking mode | 17.2–20.5 tok/s |
| Single-stream decode, non-thinking | 21.6–22.7 tok/s |
| Tool-call request decode | 26–28 tok/s |
| TTFT, fresh ~16K-token prompt (warm) | ~8.3 s (~1.8K tok/s prefill) |
| TTFT, same prompt on a cold-booted server | ~13 s (first prefill pays Triton kernel warmup; the mounted cache persists it across restarts) |
| Spec-decode sweep (steps 2/3/4/5/6) | 3/1/4 is the peak — see above |

Run-to-run variance is ~±1.5 tok/s (±7%) — treat smaller deltas as noise, and re-measure twice after any restart before believing a number. The box is bandwidth-bound (~273 GB/s peak, ~240 sustained): at ~22.5 tok/s the verify passes already consume ~65% of the wall, so config-level tuning is close to exhausted. The next step-change needs a newer `lmsysorg/sglang:qwen38-27b` image — pull it, re-sweep the spec configs (~30 min of restarts and benchmarks), and re-validate.

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
- `start.sh` prints the last 200 log lines and exits if the container dies before becoming ready
- Terminal output filters the harmless per-layer “Enabled fused SiLU+mul+FP4-quant…” notices; `.sglang.log` keeps everything
- Concurrency check: `grep max_running_requests .sglang.log` — should equal your `MAX_CONCURRENT_REQUESTS` (default 10), not a lower clamped value
- Mamba pool check: `grep max_mamba_cache_size .sglang.log` — expect `MAX_CONCURRENT_REQUESTS × 4`
- First long prefill after a cold boot is slow (~13 s for a fresh 16K prompt vs ~8 s warm) — that's Triton kernel warmup, not a regression; the `.cache/triton` volume persists it across restarts
- If startup dies with `AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'`, you're using DSpark with `YARN=1` / `CONTEXT_LENGTH=1000000` — the YaRN override leaks into the draft config. Keep `YARN=0` and `CONTEXT_LENGTH=262144` for DSpark (see Context note)
- First start downloads ~22 GB of weights (plus ~2.7 GB DSpark draft model if you switch to DSpark); subsequent starts reuse `./.cache/huggingface`

## Repository layout

```
.
├── start.sh       # launch SGLang container, wait for readiness
├── stop.sh        # stop the container, clean up pid file
├── .env           # live config (context / concurrency / quant / tuning); not tracked by git
├── .env.sample    # tracked template — copy to .env to configure
├── .gitignore     # whitelist: start.sh, stop.sh, README.md, .env.sample, LICENSE, .gitignore
├── LICENSE        # MIT
└── README.md
```

## Notes

- `QUANT` values: `nvfp4` → `RadixArk/Qwen3.8-27B-NVFP4`, `fp8` → `Qwen/Qwen3.8-27B-FP8`, `bf16` → `Qwen/Qwen3.8-27B` (all fit in the Spark's 128 GB).
- `SERVED_MODEL_NAME`, `IMAGE`, `CONTAINER_NAME`, `PORT` are set inline in `start.sh` (not `.env`).

## Credits

- [SGLang cookbook — Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) — the DGX Spark serving recipe, MTP and GDN state-pool guidance
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) — YaRN 1M-context SGLang recipe and sampling recommendations
- [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) — NVFP4 W4A4 checkpoint (FP8 KV calibration scales)
- [SGLang](https://github.com/sgl-project/sglang) — inference engine and OpenAI/Anthropic-compatible server
