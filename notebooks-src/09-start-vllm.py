# %% [markdown]
# # Lab 09.1 · Start vLLM & talk to it
#
# **Tokens per Second** · Module 09
#
# You will:
# 1. Install vLLM on a free Colab T4.
# 2. Predict, **on paper**, how big the KV cache pool will be.
# 3. Start a real server with `vllm serve`, and read its startup log.
# 4. Talk to it: `/v1/models`, `/v1/completions`, `/v1/chat/completions`, streaming.
# 5. Measure time to first token and time per output token off the stream.
# 6. Check your prediction against the `GPU KV cache size` line the server prints.
#
# Runtime → Change runtime type → **T4 GPU**. Then run the cells top to bottom.
#
# This notebook starts and stops a real HTTP server. That takes a few minutes;
# be patient and read the output of each cell before moving on.

# %% [markdown]
# ## 0. Install
#
# `vllm` pins its own CUDA-enabled PyTorch, so **restart the runtime after this cell**
# (Runtime → Restart runtime) before running the next one.

# %%
!pip install -q vllm

# %% [markdown]
# ## 1. The paper calculation, before anything runs
#
# We use **`Qwen/Qwen2.5-0.5B-Instruct`**: about 0.5 B parameters, so 0.98 GB in FP16,
# which leaves plenty of a T4's 16 GB for a KV cache.
#
# vLLM sizes things like this:
#
# ```
#   budget      = gpu_memory_utilization × total_memory
#   pool        = budget − weights − reserve      (reserve ≈ 4% of total memory)
#   pool_tokens = pool ÷ kv_bytes_per_token
# ```
#
# and `kv_bytes_per_token = 2 × layers × kv_heads × head_dim × bytes`.

# %%
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PARAMS = 0.494e9          # 0.49 B parameters
BYTES = 2                 # float16
TOTAL = 16e9              # T4 has 16 GB
UTIL = 0.85
RESERVE_FRAC = 0.04

weights = PARAMS * BYTES
budget = UTIL * TOTAL
reserve = RESERVE_FRAC * TOTAL
pool = budget - weights - reserve

print(f"weights          {weights/1e9:8.2f} GB   (params × 2 bytes)")
print(f"budget           {budget/1e9:8.2f} GB   (util {UTIL} × 16 GB)")
print(f"reserve          {reserve/1e9:8.2f} GB   (~4% of the card)")
print(f"KV pool          {pool/1e9:8.2f} GB")
print(f"predicted pool at 2048-token requests: {pool/2048:,.0f} sequences")

# %% [markdown]
# KV bytes per token, once we know the model's shape. We read it from the config
# after the weights are downloaded (the next cell does that with `AutoConfig`).

# %%
import json, os, platform, subprocess, sys, time, math
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    print("torch", torch.__version__, "· cuda", torch.version.cuda)
    print(torch.cuda.get_device_name(0), "·", torch.cuda.get_device_properties(0).total_memory/1e9, "GB")
except Exception as e:            # pragma: no cover
    raise SystemExit(f"Install / restart the runtime first: {e}")

from transformers import AutoConfig, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
cfg = AutoConfig.from_pretrained(MODEL)

L, HKV, DH = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
kv_per_token = 2 * L * HKV * DH * BYTES
print(f"\n{MODEL}: {L} layers, {HKV} KV heads, head_dim {DH}")
print(f"KV bytes per token = 2 × {L} × {HKV} × {DH} × {BYTES} = {kv_per_token:,} B "
      f"({kv_per_token/1024:.0f} KB)")
print(f"=> predicted pool holds {pool/kv_per_token:,.0f} tokens of context")
print(f"=> predicted blocks (16 tokens each): {pool/(kv_per_token*16):,.0f}")

# %% [markdown]
# ## 2. Start the server
#
# We run `vllm serve` as a **subprocess**, with its output going to `vllm.log`, so we
# can read the startup log afterwards. Two flags matter on a T4:
#
# - `--dtype float16` — the T4 has no BF16 tensor cores.
# - `--enforce-eager` — skip CUDA graph capture; small benefit here, and it makes
#   small/old GPUs behave.

# %%
PORT = 8000
LOG = "vllm.log"
if os.path.exists(LOG):
    os.remove(LOG)

cmd = [
    "vllm", "serve", MODEL,
    "--dtype", "float16",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", str(UTIL),
    "--port", str(PORT),
    "--enforce-eager",
    "--disable-log-requests",
]
print(" ".join(cmd))
proc = subprocess.Popen(cmd, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
print("pid", proc.pid, "- waiting for /health ...")

# %%
import requests

t0 = time.time()
ready = False
while time.time() - t0 < 900:
    if proc.poll() is not None:
        print("!! the server exited. Last 40 log lines:")
        print(open(LOG).read()[-4000:])
        break
    try:
        if requests.get(f"http://localhost:{PORT}/health", timeout=2).status_code == 200:
            ready = True
            break
    except Exception:
        pass
    time.sleep(3)

print(f"ready={ready} after {time.time()-t0:.0f} s")
print(open(LOG).read()[-2500:])

# %% [markdown]
# ### Read the log against your prediction
#
# Look for the `GPU KV cache size: N tokens` line and compare it with the number you
# printed in section 1. They will not match exactly — vLLM reserves more for non-torch
# memory and for fragmentation than our flat 4% — but they should be the same order of
# magnitude, and the **ratio** is what matters.

# %%
log = open(LOG).read()
for line in log.splitlines():
    low = line.lower()
    if any(k in low for k in ["kv cache size", "maximum concurrency", "memory profiling",
                              "startup complete", "model weights take", "capturing"]):
        print(line)

# %%
import re
m = re.search(r"KV cache size: ([\d,]+) tokens", log)
if m:
    actual = int(m.group(1).replace(",", ""))
    print(f"\npredicted pool tokens : {pool/kv_per_token:,.0f}")
    print(f"actual   pool tokens : {actual:,}")
    print(f"ratio                 : {actual/(pool/kv_per_token):.2f}×")
else:
    print("Could not find the KV cache line — scroll up and read the log yourself.")

# %% [markdown]
# ## 3. Talk to it
#
# `/v1/models` first: what does the server think it is serving?

# %%
base = f"http://localhost:{PORT}/v1"
print(json.dumps(requests.get(base + "/models", timeout=30).json(), indent=2)[:800])

# %% [markdown]
# A plain completion. `temperature: 0` makes it greedy, so the answer is reproducible.

# %%
r = requests.post(base + "/completions", timeout=120, json={
    "model": MODEL,
    "prompt": "The three speed limits of a GPU are compute, bandwidth, and",
    "max_tokens": 24,
    "temperature": 0.0,
})
d = r.json()
print(json.dumps(d, indent=2)[:1200])

# %% [markdown]
# Now the chat endpoint. The server applies the model's chat template, including its
# special role tokens — those count against your context budget and are billed as
# prompt tokens.

# %%
r = requests.post(base + "/chat/completions", timeout=180, json={
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": "Why does batching tokens make decoding faster?"},
    ],
    "max_tokens": 160,
    "temperature": 0.0,
})
d = r.json()
print(d["choices"][0]["message"]["content"])
print("\nfinish_reason :", d["choices"][0]["finish_reason"])
print("usage         :", d["usage"])
print("prompt tokens :", d["usage"]["prompt_tokens"],
      f"but your question alone is {len(tok('Why does batching tokens make decoding faster?')['input_ids'])} tokens — "
      "the system prompt and the chat template's role markers are billed too")

# %% [markdown]
# ## 4. Streaming, and measure the two latencies
#
# With `"stream": true` the response is server-sent events: one `data:` line per token.
# Time the first line (**TTFT**) and the gaps between the rest (**TPOT**).

# %%
def stream_and_time(prompt, max_tokens=80):
    t_send = time.perf_counter()
    times, pieces = [], []
    with requests.post(base + "/chat/completions", stream=True, timeout=180, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }) as resp:
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            body = raw[5:].strip()
            if body == b"[DONE]":
                break
            times.append(time.perf_counter())
            chunk = json.loads(body)
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                pieces.append(delta)
    return "".join(pieces), t_send, times

text, t_send, times = stream_and_time("Explain what a KV cache is, in two sentences.")
ttft = (times[0] - t_send) * 1000
gaps = [(times[i] - times[i - 1]) * 1000 for i in range(1, len(times))]

print(text)
print(f"\ntokens streamed : {len(times)}")
print(f"TTFT            : {ttft:.0f} ms")
print(f"TPOT (mean gap) : {sum(gaps)/len(gaps):.1f} ms  → {1000/(sum(gaps)/len(gaps)):.0f} tokens/s")
print(f"total time      : {(times[-1]-t_send)*1000:.0f} ms")

# %% [markdown]
# **Check it against the arithmetic.** Qwen2.5-0.5B-Instruct in FP16 is 0.98 GB of
# weights. The T4 reads about 320 GB/s at peak, and in practice ~80% of that:
#
# `0.98 GB ÷ 256 GB/s ≈ 3.8 ms`, plus ~1.5 ms of fixed per-step overhead
# (Python, kernel launches, sampling) ≈ **5.3 ms**, or about 190 tokens/s.
#
# If your measured TPOT is far above that, the extra time is overhead, not memory —
# which is exactly why `--enforce-eager` exists, and why the real fix on small models
# is batching many requests into one step (which we measure next).

# %%
for c in (1, 8, 32):
    prompts = ["Explain what a KV cache is, in two sentences."] * c
    t = time.perf_counter()
    outs = requests.post(base + "/chat/completions", timeout=600, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": p} for p in prompts],
        "max_tokens": 80, "temperature": 0.0,
    }).json()
    wall = time.perf_counter() - t
    gen = sum(o["usage"]["completion_tokens"] for o in outs["choices"])
    prompt_tokens = sum(o["usage"]["prompt_tokens"] for o in outs["choices"])
    print(f"concurrency {c:>3}  wall {wall:6.2f} s  "
          f"output {gen/wall:8.1f} tok/s  prompt {prompt_tokens/wall:8.1f} tok/s")

# %% [markdown]
# **Read it.** Total output throughput climbs steeply as you add requests, because the
# 0.98 GB of weights is read once per step and shared by everyone. That single fact is
# the whole argument for batching, and the whole of Module 07.

# %% [markdown]
# ## 5. The engine's own counters
#
# `/metrics` is Prometheus text. These are the lines worth knowing.

# %%
text = requests.get(f"http://localhost:{PORT}/metrics", timeout=30).text
wanted = [l for l in text.splitlines()
          if l and not l.startswith("#")
          and any(k in l for k in ["num_requests_running", "num_requests_waiting",
                                   "gpu_cache_usage", "generation_tokens_total",
                                   "prompt_tokens_total"])]
print("\n".join(wanted) or "(no matching counters — scroll up and read /metrics yourself)")

# %% [markdown]
# ## 6. Shut down
#
# Always stop the server, or the GPU memory stays held and the next cell fails.

# %%
proc.terminate()
try:
    proc.wait(timeout=60)
except Exception:
    proc.kill()
print("server stopped")
print("total GPU memory still held:", torch.cuda.memory_allocated() / 1e9, "GB (this process)")

# %% [markdown]
# ## Your turn
#
# 1. Change `UTIL` to 0.6 and 0.95, restart, and compare the printed `GPU KV cache size`
#    with your prediction. Does the pool track the budget linearly?
# 2. Set `--max-model-len 512` and look at the `Maximum concurrency` line. Then set it to
#    2,048. Which one is your workload's choice, and why is it not a speed setting?
# 3. Run the streaming cell again. How many prompt tokens did the chat template add?
#    That is context you are paying for and did not type.
# 4. Ask for `max_tokens: 4096` on a request that finishes in 80. What does `finish_reason`
#    say, and what does that tell you about the slot that request was holding?
# 5. Write your three numbers into **My notes**: your predicted pool, the actual pool, and
#    your measured TPOT. Module 09.3 starts from exactly these.