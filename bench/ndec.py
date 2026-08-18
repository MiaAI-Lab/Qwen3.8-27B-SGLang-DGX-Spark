#!/usr/bin/env python3
# Net-decode tok/s via two-call delta method (same prompt, 60 vs 600 tokens).
import json, time, urllib.request, sys

def call(prompt, mt, thinking):
    body = {"model": "qwen3.8-27b-sglang", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": mt,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request("http://127.0.0.1:8888/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read()[:200]!r}")
    return time.time() - t0, d["usage"]["completion_tokens"]

call("Say OK", 16, False)  # warmup

probes = [
    ("code (dspark-favored)", "Write a Python class LRUCache with O(1) get and put using OrderedDict, plus a small test.", False),
    ("essay-prose (mtp-favored)", "Write a detailed technical essay on the history of computing, from Babbage to GPUs.", False),
]
for name, p, th in probes:
    d1, c1 = call(p, 60, th)
    d2, c2 = call(p, 600, th)
    tps = (c2 - c1) / (d2 - d1)
    print(f"{name:30s} net decode = {tps:6.2f} tok/s   (c1={c1},d1={d1:.2f} | c2={c2},d2={d2:.2f})")
