#!/usr/bin/env bash
# Simple single-stream decode benchmark against the local SGLang server.
set -euo pipefail
URL="http://127.0.0.1:8888/v1/chat/completions"
MODEL="${MODEL:-qwen3.8-27b-sglang}"

bench() {
  local label="$1" payload="$2"
  python3 - "$label" "$payload" <<'EOF'
import json, sys, time, urllib.request
label, payload = sys.argv[1], sys.argv[2]
body = json.loads(payload)
if "model" not in body: body["model"] = "qwen3.8-27b-sglang"
req = urllib.request.Request(
    "http://127.0.0.1:8888/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    out = json.load(r)
dt = time.time() - t0
toks = out["usage"]["completion_tokens"]
print(f"{label:26s} tokens={toks:5d}  {dt:6.1f}s  {toks/dt:6.1f} tok/s")
EOF
}

PROMPT="Write a detailed technical essay on the history of computing, from Babbage to GPUs."
bench "$1 thinking"    "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":400}"
bench "$1 non-thinking" "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":400,\"temperature\":0.7,\"top_p\":0.8,\"presence_penalty\":1.5,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
bench "$1 toolcall" "{\"messages\":[{\"role\":\"user\",\"content\":\"Call get_weather for Tokyo, then reserve_table for 4 at 7pm tonight via the reserve_table tool, then send a confirmation via send_email. Emit the tool calls.\"}],\"max_tokens\":400,\"chat_template_kwargs\":{\"enable_thinking\":false}}"

# TTFT probe: ~16k-token prompt with a unique lead (defeats prefix cache),
# streaming; reports time-to-first-token and effective prefill rate.
python3 - <<'EOF'
import json, time, urllib.request, uuid
para = ("The quick brown fox jumps over the lazy dog while the sun sets "
        "over the mountains and rivers flow gently through the valley. ") * 640
prompt = f"Session {uuid.uuid4()}. Summarize in one sentence:\n\n{para}"
body = {"model": "qwen3.8-27b-sglang", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32, "stream": True}
req = urllib.request.Request("http://127.0.0.1:8888/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); first = None
with urllib.request.urlopen(req, timeout=300) as r:
    for line in r:
        if first is None and line.startswith(b"data:") and b"delta" in line:
            first = time.time()
            break
ttft = (first or time.time()) - t0
n = len(prompt.split())
print(f"ttft~16k                   ttft={ttft:6.2f}s  prefill~{n/ttft:7.0f} tok/s")
EOF
