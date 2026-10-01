# %% [markdown]
# # Lab 22.1 · Measure a cold start, then watch it drain
#
# **Tokens per Second** · Module 22
#
# You will:
# 1. Time the phases of a real server start-up, on paper and then in reality.
# 2. See why the page cache makes a *second* start so much faster.
# 3. Measure the difference between a first request and a steady-state one
#    (this is why warm-up exists).
# 4. Compute the drain timeout your longest generation needs.
# 5. Watch a real stream get truncated when the drain budget is too short.
#
# Runtime → Change runtime type → **T4 GPU**. Then run the cells top to bottom.
#
# The T4 is not production hardware and the absolute numbers will not match an
# H100. **What transfers is the shape**: which phase dominates, why, and how the
# phases change with model size. Say so explicitly in whatever you write about
# this afterwards — that caveat is part of the deliverable.

# %% [markdown]
# ## 0. Install
#
# `vllm` pins its own CUDA-enabled PyTorch, so **restart the runtime after this
# cell** (Runtime → Restart runtime) before running the next one.

# %%
!pip install -q vllm

# %%
import json, math, os, signal, subprocess, sys, threading, time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOGFILE = "/tmp/vllm_start.log"

print(sys.version.split()[0])

# %% [markdown]
# ## 1. The paper calculation, before anything runs
#
# From Lesson 1: the startup path is
#
# ```
# process launch & imports  →  read checkpoint  →  copy into HBM
#        →  capture CUDA graphs  →  autotune  →  warm-up
# ```
#
# and the phase that scales with your model is the checkpoint read:
#
# $$t_{\text{read}} = \frac{P \times b}{\text{storage throughput}}$$
#
# We will check this against a real server shortly. First, the arithmetic.

# %%
# Qwen2.5-0.5B-Instruct, read from its config after download.
PARAMS = 0.494e9
BYTES = 2            # float16
TOTAL_MEM = 16e9     # T4 has 16 GB

weight_bytes = PARAMS * BYTES
print(f"weights            {weight_bytes/1e9:8.3f} GB   ({PARAMS/1e9:.2f} B params x 2 bytes)")

# Two very different "storage throughputs":
#   - a network-backed object store on a real cluster: ~1.2 GB/s
#   - whatever Colab gives you, which is usually much faster
for name, bps in [("cluster object store", 1.2e9), ("fast local NVMe", 5e9)]:
    print(f"read @ {name:22s} {bps/1e9:.1f} GB/s -> {weight_bytes/bps:6.2f} s")

# Host-to-device copy over PCIe (Gen4 x16 is ~24 GB/s)
print(f"copy into HBM @ PCIe 24 GB/s  -> {weight_bytes/24e9:6.2f} s")

# The ceiling: largest model we could cold-start inside a 120 s budget
for budget, bps in [(120, 1.2e9), (120, 5e9)]:
    p_max = budget * bps / BYTES
    print(f"budget {budget:4d} s @ {bps/1e9:.1f} GB/s -> largest model {p_max/1e9:6.1f} B params")

# %% [markdown]
# ## 2. The model config: KV bytes per token
#
# We need the real shape, not the parameter count, for everything that follows.

# %%
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(MODEL)
L, NKV, DHEAD = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
print(f"layers {L}   kv heads {NKV}   head dim {DHEAD}   hidden {cfg.hidden_size}")

kv_bytes_per_token = 2 * L * NKV * DHEAD * 2   # K and V, 2 bytes each
print(f"KV bytes per token: {kv_bytes_per_token:,} ({kv_bytes_per_token/1024:.0f} KiB)")

# %% [markdown]
# ## 3. Start the server for real, and time it
#
# We start `vllm serve` as a subprocess and record the wall-clock time until the
# server answers a health check. This is your **creation-to-ready** number: the
# one thing the previous lesson said to put on a dashboard.

# %%
PORT = 8000
t0 = time.time()
# Log to a FILE, not a pipe: nobody is draining a pipe, so once the OS buffer
# fills the server would block forever on its own log writes.
log = open(LOGFILE, "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
     "--model", MODEL, "--port", str(PORT), "--max-model-len", "2048",
     "--gpu-memory-utilization", "0.85", "--enforce-eager"],
    stdout=log, stderr=subprocess.STDOUT, text=True,
)
print(f"launched, pid {proc.pid}; waiting for the health endpoint...")

# Poll the OpenAI models endpoint until it answers.
import urllib.request
def health(port=PORT):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

while not health():
    if proc.poll() is not None:
        raise SystemExit("server exited during startup")
    time.sleep(1)

t_ready = time.time() - t0
log.flush()
print(f"\nREADY after {t_ready:.1f} s  (this is your creation-to-ready number)")

# %% [markdown]
# ### Read the startup log
#
# The server prints phase timestamps. Compare the gaps:
#
# - a long gap at **importing / loading weights** is storage cost,
# - a long gap at **capturing CUDA graphs** or **autotuning** is shape count,
# - and everything before that is process start and imports.
#
# We started with `--enforce-eager`, which *disables* graph capture. That makes
# the next comparison meaningful.

# %%
with open(LOGFILE) as f:
    lines = [ln.rstrip() for ln in f if ln.strip()]

print(f"{len(lines)} log lines\n")
# Show the milestone lines: the phases Lesson 1 names.
KEYS = ("Loading safetensors", "model weights took", "graph captur",
        "Capturing", "init engine", "Starting vLLM API", "torch.compile",
        "Available KV cache memory", "GPU KV cache size", "Application startup")
for ln in lines:
    if any(k.lower() in ln.lower() for k in KEYS):
        print("  ", ln[:150])

# %% [markdown]
# Look at the weights line specifically. It usually reports both the byte count
# and the elapsed time, which is `t_read` from Lesson 1 measured for real.

# %%
for ln in lines:
    if "model weights took" in ln.lower() or "loading safetensors" in ln.lower():
        print("  ", ln[:200])

# %% [markdown]
# ## 4. Warm-up: first request vs steady state
#
# This is the measurement that justifies warm-up existing. We send the *same*
# kind of request several times and record time-to-first-token each time.
#
# Expect the first call to be noticeably slower: graphs, caches and allocator
# arenas that were never populated. That gap is what a warm-up phase removes.

# %%
import json as _json

def chat_once(prompt, max_tokens=32, timeout=120):
    body = _json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "stream": False, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = _json.loads(r.read())
    return time.time() - t, out["usage"]["completion_tokens"]

ttfts = []
for i in range(12):
    dt, ntok = chat_once(f"Count from {i} to {i + 5}. Be brief.")
    ttfts.append(dt)
    print(f"request {i:2d}: {dt:6.2f} s  ({ntok} output tokens)")

first, steady = ttfts[0], sum(ttfts[3:]) / len(ttfts[3:])
print(f"\nfirst request : {first:6.2f} s")
print(f"steady state  : {steady:6.2f} s")
print(f"warm-up gap   : {first - steady:6.2f} s  ({(first/steady - 1) * 100:.0f}% slower)")

# %% [markdown]
# **This is the number to argue from.** If your gap is small, warm-up is cheap.
# If it is 2x, then serving real traffic to an unwarmed replica means the first
# few dozen users per replica pay it. Multiply by your replica count and restart
# frequency and you have the answer to "do we need warm-up?".
#
# Note also that this measurement used `--enforce-eager`. Turn it on in
# production and the gap narrows, because graph capture moves work from the
# first request into startup — which is exactly the trade Lesson 1 described.

# %% [markdown]
# ## 5. The page cache, demonstrated
#
# Start a *second* server on the same node with the weights already in the OS
# page cache, and compare creation-to-ready. Expect the second to be
# substantially faster — and remember that this is exactly why you must measure
# cold starts on a node you have deliberately emptied.

# %%
PORT2 = 8001
t1 = time.time()
log2 = open("/tmp/vllm_start2.log", "w")
proc2 = subprocess.Popen(
    [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
     "--model", MODEL, "--port", str(PORT2), "--max-model-len", "2048",
     "--gpu-memory-utilization", "0.85", "--enforce-eager"],
    stdout=log2, stderr=subprocess.STDOUT, text=True,
)

while not health(PORT2):
    if proc2.poll() is not None:
        break
    time.sleep(1)

t_ready2 = time.time() - t1
log2.flush()
print(f"first  start: {t_ready:6.1f} s")
print(f"second start: {t_ready2:6.1f} s   (weights already in page cache)")
print(f"speedup      : {t_ready / t_ready2:.2f}x  on identical hardware and arguments")

# %% [markdown]
# Both servers are now holding 16 GB of GPU memory budget between them, so we
# shut the second one down immediately. (If this OOMs on your Colab session,
# that is itself a useful reminder of Lesson 8: memory is the real ceiling.)

# %%
proc2.send_signal(signal.SIGTERM)
try:
    proc2.wait(timeout=90)
    print("second server stopped cleanly (SIGTERM accepted)")
except subprocess.TimeoutExpired:
    proc2.kill(); print("second server had to be killed")

# %% [markdown]
# ## 6. Compute your drain timeout
#
# From Lesson 2:
#
# $$t_{\text{drain}} \ge t_{\text{TTFT}} + t_{\text{TPOT}} \times \text{max output tokens}$$
#
# Use the *steady-state* per-token time you measured in step 4.

# %%
tpot = (steady / 32) * 1000     # ms per token, from the steady-state runs above
print(f"steady-state TPOT: {tpot:.1f} ms/token\n")
print(f"{'max output':>12}  {'request time':>13}  {'+50% margin':>13}")
for out in [200, 400, 800, 1600]:
    t_req = 0.4 + (tpot / 1000) * out
    print(f"{out:>12}  {t_req:>11.1f} s  {t_req * 1.5:>11.1f} s")

# %% [markdown]
# **Pick the row matching your product's longest supported output and record
# that number.** Then treat it as a constraint: if someone raises the max output
# length, the drain timeout has to rise with it or every deploy truncates those
# generations.
#
# Which row describes you? A chat product with a 400-token cap needs seconds; a
# long-form generation product needs minutes, and its drain timeout will be one
# of the more surprising numbers in your configuration.

# %% [markdown]
# ## 7. Watch a drain truncate a stream (or save it)
#
# This is the experiment that convinces people the drain gate matters, so run it
# even though the outcome is an error message.
#
# We start a **long** generation, then SIGTERM the server while it streams. With
# a generous deadline it finishes. With a deadline shorter than the generation,
# the stream stops early — and note that **there is no error**. The client just
# stops receiving tokens.

# %%
def stream_and_watch(drain_timeout, max_tokens=400):
    """Start a streaming generation; SIGTERM the server after `drain_timeout`."""
    body = _json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a detailed essay about the history of bridges."}],
        "max_tokens": max_tokens, "stream": True, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})

    got = []
    # SIGTERM at the drain deadline: the platform sends it, then kills at the
    # deadline if we have not exited. We model the signal here and let the
    # kernel finish the in-flight request.
    killer = threading.Timer(drain_timeout, lambda: proc.send_signal(signal.SIGTERM))
    killer.start()
    try:
        with urllib.request.urlopen(req, timeout=drain_timeout + 60) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = _json.loads(payload)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        got.append(delta)
                except Exception:
                    pass
    except Exception as e:
        print(f"  stream ended with an exception: {type(e).__name__}")
    finally:
        killer.cancel()

    return "".join(got)

# --- Case A: a generous deadline. The generation should complete. ---
print("Case A: drain timeout 120 s, generating up to 400 tokens")
text_a = stream_and_watch(120)
print(f"  received {len(text_a.split())} words, ends with: ...{text_a[-60:]!r}")

# %% [markdown]
# If Case A completed, the server may now be shutting down. Restart it for the
# interesting case.

# %%
if health() is False:
    time.sleep(3)
    if not health():
        t2 = time.time()
        proc = subprocess.Popen(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
             "--model", MODEL, "--port", str(PORT), "--max-model-len", "2048",
             "--gpu-memory-utilization", "0.85", "--enforce-eager"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        while not health():
            if proc.poll() is not None:
                raise SystemExit("server exited during startup")
            time.sleep(1)
        print(f"restarted in {time.time() - t2:.1f} s")

# %% [markdown]
# --- Case B: a deadline far shorter than the generation. ---
#
# We kill the server **hard** after 4 seconds rather than sending SIGTERM, so
# the process is gone before the generation could finish. This is exactly what a
# drain timeout that is too short looks like from the client's side.

# %%
def stream_and_kill(kill_after=4.0, max_tokens=400):
    body = _json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a detailed essay about the history of bridges."}],
        "max_tokens": max_tokens, "stream": True, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})

    killer = threading.Timer(kill_after, lambda: proc.kill())
    killer.start()
    got = []
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = _json.loads(payload)["choices"][0]["delta"].get("content")
                    if delta:
                        got.append(delta)
                except Exception:
                    pass
    except Exception as e:
        print(f"  stream ended with: {type(e).__name__}")
    finally:
        killer.cancel()

    return "".join(got)

text_b = stream_and_kill(4.0)
print(f"\nCase B: killed after 4 s, generating up to 400 tokens")
print(f"  received {len(text_b.split())} words")
print(f"  ends with: ...{text_b[-60:]!r}")
print("\nCompare with Case A. The client got no error in either case — in Case B")
print("it simply stopped receiving tokens mid-sentence. That is what a drain")
print("timeout that is too short looks like to a user, and why this bug gets")
print("blamed on the client.")

# %% [markdown]
# ## 8. Your numbers
#
# Fill this in. These are the numbers your checklist needs.

# %%
print("""
CREATION-TO-READY (T4, eager mode)   {t_ready:.1f} s first, {t_ready2:.1f} s second
STEADY-STATE TPOT                   {tpot:.1f} ms
WARM-UP GAP                         {gap:.1f} s  ({gappct:.0f}% slower first request)
DRAIN TIMEOUT for your max output   {drain:.1f} s  (at {maxout} output tokens)

Remember: the T4 numbers are shape-only. Report the *phase ordering* and the
*ratios* as what transfers, and say explicitly that absolute milliseconds do not.
""".format(
    t_ready=t_ready, t_ready2=t_ready2, tpot=tpot,
    gap=first - steady, gappct=(first / steady - 1) * 100,
    drain=(0.4 + tpot / 1000 * 800) * 1.5, maxout=800,
))

# %%
# Clean up: make sure no server is left running.
for p in (proc,):
    if p.poll() is None:
        p.kill()
        print("server stopped")