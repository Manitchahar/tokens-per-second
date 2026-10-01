# %% [markdown]
# # Lab 09.3 · Anatomy of a request
#
# **Tokens per Second** · Module 09
#
# You will measure the **three clocks** of a request inside vLLM, one at a time:
#
# 1. **TTFT** — queueing plus prefill plus one decode step. Time the first streamed token.
# 2. **ITL** — the gaps between the rest of the tokens. Time every chunk of the stream.
# 3. **The queue** — send N requests at once and watch how the latency of one of them
#    grows as the others get in front of it.
#
# Then watch the **KV blocks** a request holds, through the engine's own metrics.
#
# Runtime → Change runtime type → **T4 GPU**. Run the cells top to bottom.
#
# We use `Qwen/Qwen2.5-0.5B-Instruct` so the whole thing runs on a free T4, and
# `--enforce-eager` because the T4 will not benefit from CUDA graphs.

# %% [markdown]
# ## 0. Install (restart the runtime afterwards)

# %%
!pip install -q vllm

# %% [markdown]
# ## 1. The harness
#
# Two helpers, the same ones from the tuning lab: start a server, wait for `/health`,
# and fire requests at it.

# %%
import json, math, os, re, statistics, subprocess, time
from concurrent.futures import ThreadPoolExecutor

import requests

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000
MAX_TOKENS = 96
SHORT = "Say hi."
LONG = "You are explaining GPU inference to a new engineer. " * 40


def start_server(extra_args=(), log="vllm_anatomy.log", timeout=900):
    if os.path.exists(log):
        os.remove(log)
    cmd = ["vllm", "serve", MODEL, "--port", str(PORT), "--disable-log-requests",
           "--dtype", "float16", "--enforce-eager",
           "--max-model-len", "4096", "--max-num-seqs", "32"] + list(extra_args)
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            raise SystemExit("server exited:\n" + open(log).read()[-3000:])
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=2).status_code == 200:
                return p, open(log).read()
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("server did not become healthy:\n" + open(log).read()[-3000:])


def stop_server(p):
    p.terminate()
    try:
        p.wait(timeout=60)
    except Exception:
        p.kill()
    time.sleep(5)


def stream_once(prompt, max_tokens=MAX_TOKENS):
    """Stream one completion; return (text, send time, list of per-token arrival times)."""
    t_send = time.perf_counter()
    times, pieces = [], []
    with requests.post(f"http://localhost:{PORT}/v1/completions", stream=True, timeout=300, json={
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True,
    }) as resp:
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            body = raw[5:].strip()
            if body == b"[DONE]":
                break
            times.append(time.perf_counter())
            delta = json.loads(body)["choices"][0].get("text")
            if delta:
                pieces.append(delta)
    return "".join(pieces), t_send, times


def kv_usage_pct():
    """The engine's own number for how full the KV cache pool is, in percent."""
    text = requests.get(f"http://localhost:{PORT}/metrics", timeout=30).text
    for line in text.splitlines():
        if line.startswith("vllm:gpu_cache_usage_perc"):
            return float(line.split()[-1])
    return None


print("harness ready")

# %% [markdown]
# ## 2. Start the server and note the pool

# %%
proc, LOG = start_server()
for line in LOG.splitlines():
    low = line.lower()
    if "kv cache size" in low or "maximum concurrency" in low or "startup complete" in low:
        print(line)

m = re.search(r"KV cache size: ([\d,]+) tokens", LOG)
pool_tokens = int(m.group(1).replace(",", "")) if m else None
print(f"\npool holds {pool_tokens:,} tokens" if pool_tokens else "pool size unknown")

# %% [markdown]
# ## 3. Clocks one and two: TTFT and ITL
#
# The first token and the rest are **different work**. Timing every chunk of a stream
# separates them.

# %%
text, t_send, times = stream_once(LONG)
ttft_ms = (times[0] - t_send) * 1000
gaps_ms = [(times[i] - times[i - 1]) * 1000 for i in range(1, len(times))]
total_ms = (times[-1] - t_send) * 1000
itl_mean = statistics.mean(gaps_ms)

print(f"prompt        a long prompt, {len(times)} tokens streamed back")
print(f"TTFT          {ttft_ms:8.1f} ms   <- queue + prefill + one decode step")
print(f"ITL mean      {itl_mean:8.1f} ms   <- {1000/itl_mean:.0f} tokens/s")
print(f"ITL median    {statistics.median(gaps_ms):8.1f} ms")
print(f"ITL p95       {sorted(gaps_ms)[int(0.95*len(gaps_ms))]:8.1f} ms")
print(f"ITL max       {max(gaps_ms):8.1f} ms   <- the outliers are Python, not the GPU")
print(f"total         {total_ms:8.1f} ms   = TTFT + {len(times)-1} x ITL")
print(f"\ncheck: {ttft_ms:.0f} + {len(times)-1} x {itl_mean:.1f} = "
      f"{ttft_ms + (len(times)-1)*itl_mean:.0f} ms")

# %% [markdown]
# **Read it.** TTFT is much larger than ITL, and ITL has a long tail: the median gap is
# the GPU, the `max` gap is the Python in between. Both facts are worth a lot later — the
# tail is exactly what Module 15 · Compilation & CUDA Graphs attacks.
#
# Now predict the decode step from the arithmetic before touching the next cell.
# Qwen2.5-0.5B is 0.98 GB in FP16, and the T4 delivers about 320 GB/s at peak,
# ~80% of that in practice, plus ~1.5 ms of fixed overhead.

# %%
pred_itl_ms = 1000 * (0.98e9 / (0.8 * 320e9) + 0.0015)
print(f"predicted one decode step: {pred_itl_ms:.1f} ms    measured ITL: {itl_mean:.1f} ms")
print(f"predicted tokens/s:        {1000/pred_itl_ms:.0f}             measured:       {1000/itl_mean:.0f}")
print(f"\nTTFT was {ttft_ms:.0f} ms. One decode step is {itl_mean:.1f} ms of that, so "
      f"{ttft_ms - itl_mean:.0f} ms was queueing plus prefill.")

# %% [markdown]
# ## 4. Clock three: the queue
#
# Fire N identical requests at the same instant. Each one waits for the others' single
# decode steps, so its latency should grow roughly like (1 + N - 1) ITLs. This is the
# arithmetic behind "queueing dominates TTFT under load".

# %%
def fire(_):
    _, t0, ts = stream_once(SHORT, max_tokens=16)
    return {"ttft": (ts[0] - t0) * 1000, "total": (ts[-1] - t0) * 1000, "tokens": len(ts)}


print(f"{'N':>4} {'p50 TTFT ms':>12} {'max TTFT ms':>12} {'p50 total ms':>13} {'tok/s each':>11}")
print("-" * 58)
for n in (1, 4, 8, 16, 32):
    with ThreadPoolExecutor(max_workers=n) as ex:
        rs = list(ex.map(fire, range(n)))
    ttfts = sorted(r["ttft"] for r in rs)
    totals = sorted(r["total"] for r in rs)
    p50_total = totals[len(totals) // 2]
    each = rs[0]["tokens"] * 1000 / p50_total
    print(f"{n:>4} {ttfts[len(ttfts)//2]:>12.0f} {max(ttfts):>12.0f} {p50_total:>13.0f} {each:>11.1f}")

# %% [markdown]
# **Read it.** The `N = 1` row is the floor. As N grows the median TTFT grows by roughly
# one ITL per extra request, and the **max** grows by much more — the unlucky request was
# admitted last. Nobody got slower on the GPU: the same read of the weights now serves
# everyone. That is continuous batching paying for itself, and queueing taxing you for it.

# %% [markdown]
# ## 5. The KV blocks a request holds
#
# The engine publishes how full its pool is. Fill it with short prompts, then with long
# ones, and compare: KV memory is a straight line in context length.

# %%
def hold(prompts, seconds=8):
    """Fire requests and sample the KV pool while they are still running."""
    with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futs = [ex.submit(stream_once, p, 128) for p in prompts]
        time.sleep(seconds)
        used = kv_usage_pct()
        for f in futs:
            try:
                f.result(timeout=300)
            except Exception:
                pass
    time.sleep(4)
    return used


print(f"{'workload':>28} {'KV pool used':>14}")
print("-" * 46)
idle_used = kv_usage_pct()
print(f"{'idle':>28} {idle_used:>13.1f}%")
short_used = hold([SHORT] * 16)
print(f"{'16 x short prompts':>28} {short_used:>13.1f}%")
long_used = hold([LONG] * 16)
print(f"{'16 x long prompts':>28} {long_used:>13.1f}%")

# %% [markdown]
# **Read it.** Same number of requests, same GPU, same step time — and the long prompts
# use several times the KV. That is the whole of Module 08 · PagedAttention & KV Memory
# in one number: **the KV cache, not the model, is what limits concurrency.**
#
# Cross-check it against the model's shape.

# %%
from transformers import AutoTokenizer

KV_PER_TOKEN = 2 * 24 * 2 * 64 * 2      # layers x kv heads x head dim x 2 (K,V) x 2 bytes
tok = AutoTokenizer.from_pretrained(MODEL)
n_long = len(tok(LONG)["input_ids"])
n_short = len(tok(SHORT)["input_ids"])

print(f"short prompt: {n_short:>4} tokens = {n_short * KV_PER_TOKEN / 1e6:6.2f} MB "
      f"= {math.ceil(n_short / 16)} block(s) of 16 tokens")
print(f"long  prompt: {n_long:>4} tokens = {n_long * KV_PER_TOKEN / 1e6:6.2f} MB "
      f"= {math.ceil(n_long / 16)} blocks")
print(f"ratio        : {n_long / n_short:.1f}x")
if pool_tokens:
    print(f"\n16 long prompts want {16 * n_long:,} tokens = "
          f"{16 * n_long / pool_tokens * 100:.1f}% of the pool")

# %% [markdown]
# ## 6. Shut down

# %%
stop_server(proc)
print("server stopped")

# %% [markdown]
# ## Check yourself
#
# 1. Your measured ITL was larger than the number the arithmetic predicted. Where did the
#    extra time go, and which module is about removing it?
# 2. In the queue sweep, TTFT grew by roughly one ITL per extra request. Why does that
#    not mean the GPU got slower?
# 3. `TTFT = queue + prefill + one decode step`. Which of the three did you just make
#    bigger, and which would you attack first for a latency-hungry product?
# 4. The 16 long prompts used several times the KV of the 16 short ones. If you needed
#    twice the concurrency at that context length, what is the cheapest change?
# 5. The KV pool is sized **before** the first request arrives. What happens to a request
#    that needs more blocks than are left?