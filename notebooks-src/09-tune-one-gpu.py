# %% [markdown]
# # Lab 09.2 · Sweep the flags on one GPU
#
# **Tokens per Second** · Module 09
#
# You will:
# 1. Predict the KV cache pool on paper for three `gpu_memory_utilization` values.
# 2. Sweep `gpu_memory_utilization` and record the **actual** `GPU KV cache size` the
#    server prints. Compare it with your prediction.
# 3. Sweep `max_num_seqs` at a fixed workload and plot the two curves that matter:
#    per-user speed and total throughput.
# 4. Pick an operating point against a latency target you write down *first*.
#
# Runtime → Change runtime type → **T4 GPU**. Run the cells top to bottom.
#
# Each configuration **restarts the server**, and on a T4 that is ~40 s per start.
# Budget about ten minutes. If Colab disconnects you, reconnect and re-run from the
# top; the cells are independent.

# %% [markdown]
# ## 0. Install (restart the runtime afterwards)

# %%
!pip install -q vllm

# %% [markdown]
# ## 1. The predictions, written down first
#
# `Qwen/Qwen2.5-0.5B-Instruct` on a T4 (16 GB). Its KV cache needs 12 KB per token:
# 24 layers × 2 KV heads × 64 head dim × 2 (K and V) × 2 bytes = 12,288 B.

# %%
import json, math, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000
TOTAL = 16e9
PARAMS = 0.494e9
BYTES = 2
KV_PER_TOKEN = 2 * 24 * 2 * 64 * BYTES
MAX_TOKENS = 48
N_REQUESTS = 24
PROMPT = ("You are explaining GPU inference to a new engineer. "
          "In two or three sentences, explain why batching makes decoding faster.")

TOK = AutoTokenizer.from_pretrained(MODEL)

print(f"model        {MODEL}")
print(f"weights      {PARAMS * BYTES / 1e9:.2f} GB")
print(f"KV per token {KV_PER_TOKEN:,} B = {KV_PER_TOKEN / 1024:.0f} KB")
print(f"prompt       {len(TOK(PROMPT)['input_ids'])} tokens; "
      f"{N_REQUESTS} requests, <= {MAX_TOKENS} output tokens each")
print()
print("predicted pool and worst-case concurrency:")
for util in (0.60, 0.85, 0.90):
    pool = util * TOTAL - PARAMS * BYTES - 0.04 * TOTAL
    print(f"  util {util:.2f}  budget {util * TOTAL / 1e9:5.2f} GB  pool {pool / 1e9:6.2f} GB"
          f"  tokens {pool / KV_PER_TOKEN:10,.0f}  at 2048 ctx: {pool / KV_PER_TOKEN / 2048:8.0f} seqs")

# %% [markdown]
# ## 2. The harness
#
# Two helpers: one that starts a server with a given set of flags and waits for
# `/health`, and one that fires requests at it concurrently and reports the numbers
# Module 06 taught us to report.

# %%
def start_server(extra_args, log="vllm_sweep.log", timeout=900):
    """Start `vllm serve` with extra_args; return (process, log text)."""
    if os.path.exists(log):
        os.remove(log)
    cmd = ["vllm", "serve", MODEL, "--port", str(PORT), "--disable-log-requests",
           "--dtype", "float16", "--enforce-eager"] + list(extra_args)
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            return p, open(log).read()
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=2).status_code == 200:
                return p, open(log).read()
        except Exception:
            pass
        time.sleep(3)
    return p, open(log).read()


def stop_server(p):
    p.terminate()
    try:
        p.wait(timeout=60)
    except Exception:
        p.kill()
    time.sleep(5)          # let the port and the GPU memory come back


def one_request(prompt, max_tokens=MAX_TOKENS):
    """Send one non-streaming request; return (latency seconds, output tokens)."""
    t = time.perf_counter()
    try:
        r = requests.post(f"http://localhost:{PORT}/v1/completions", timeout=300, json={
            "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True,
        })
        r.raise_for_status()
        return time.perf_counter() - t, r.json()["usage"]["completion_tokens"]
    except Exception:
        return None, 0


def bench(concurrency, log_text=None, prompt=None, max_tokens=MAX_TOKENS):
    """Fire N_REQUESTS requests at `concurrency` at a time, and report honestly."""
    prompt = prompt or PROMPT
    lat, gen = [], 0
    t0 = time.perf_counter()
    for _ in range(math.ceil(N_REQUESTS / concurrency)):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for sec, n in ex.map(lambda _: one_request(prompt, max_tokens), range(concurrency)):
                if sec is not None:
                    lat.append(sec)
                    gen += n
    wall = time.perf_counter() - t0
    if not lat:
        return None
    s = sorted(lat)
    m = re.search(r"KV cache size: ([\d,]+) tokens", log_text) if log_text else None
    return {
        "concurrency": concurrency,
        "p50": s[len(s) // 2],
        "p95": s[min(len(s) - 1, int(0.95 * len(s)))],
        "out_tok_s": gen / wall,
        "req_s": len(lat) / wall,
        "kv_tokens": int(m.group(1).replace(",", "")) if m else None,
    }


def show(row):
    if row is None:
        print("  (every request failed — read the log)")
        return
    kv = f"{row['kv_tokens']:,}" if row["kv_tokens"] else "n/a"
    print(f"  conc {row['concurrency']:>4}   p50 {row['p50'] * 1000:7.0f} ms   "
          f"p95 {row['p95'] * 1000:7.0f} ms   output {row['out_tok_s']:8.1f} tok/s   "
          f"requests {row['req_s']:5.2f}/s   KV pool {kv} tokens")


print("harness ready")

# %% [markdown]
# ## 3. Sweep `gpu_memory_utilization`
#
# This flag does not change speed directly. It changes how big the KV cache pool is,
# which is what decides how many requests can be resident. Watch the printed pool.

# %%
for util in (0.60, 0.85, 0.90):
    print(f"\n=== gpu_memory_utilization {util}  (max_model_len 2048, max_num_seqs 32) ===")
    p, log_text = start_server([
        "--gpu-memory-utilization", str(util),
        "--max-model-len", "2048",
        "--max-num-seqs", "32",
    ])
    predicted = (util * TOTAL - PARAMS * BYTES - 0.04 * TOTAL) / KV_PER_TOKEN
    print(f"  predicted pool {predicted:,.0f} tokens")
    show(bench(8, log_text))
    stop_server(p)

# %% [markdown]
# **Read it.** Throughput barely moves between 0.60 and 0.90 at concurrency 8, because
# at that concurrency the *weights* dominate memory traffic, not the KV cache. The pool
# only matters when you have many requests, or long ones — which is section 5.
#
# Also read the pool line against your prediction. vLLM reserves more than our flat 4%
# (it leaves room for non-torch allocations and fragmentation), so the actual pool is
# smaller than the naive prediction. That is why the flag's default is conservative, and
# why `0.95` is not free money.

# %% [markdown]
# ## 4. Sweep `max_num_seqs`
#
# Same workload, one flag changing. This is the interesting sweep: per-user speed falls
# slowly, total throughput rises fast, and memory eventually ends the story.

# %%
print("starting the baseline server (util 0.90, max_model_len 2048) ...")
proc, LOG = start_server(["--gpu-memory-utilization", "0.90", "--max-model-len", "2048"])

sweep = []
for c in (1, 4, 8, 16, 32, 64):
    row = bench(c, LOG)
    show(row)
    sweep.append(row)

# %% [markdown]
# Now the chart: per-user speed and total throughput on one plot, concurrency on x.
# This is the picture from 09.3 — measured rather than modelled.

# %%
import matplotlib.pyplot as plt

ok = [r for r in sweep if r]
if ok:
    xs = [r["concurrency"] for r in ok]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(xs, [1000 / r["p50"] for r in ok], "o-", color="#83c167")
    ax.set_xlabel("concurrency (requests in flight)")
    ax.set_ylabel("tokens/s per user", color="#83c167")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    ax2 = ax.twinx()
    ax2.plot(xs, [r["out_tok_s"] for r in ok], "s--", color="#f2b544")
    ax2.set_ylabel("total output tokens/s", color="#f2b544")
    ax.grid(alpha=0.25)
    ax.set_title(f"{MODEL} on a T4 — concurrency sweep ({MAX_TOKENS} output tokens each)")
    fig.tight_layout()
    plt.show()

    base = 1000 / ok[0]["p50"]
    print(f"{'concurrency':>12} {'per user tok/s':>16} {'total tok/s':>14} {'vs conc 1':>11}")
    for r in ok:
        print(f"{r['concurrency']:>12} {1000 / r['p50']:>16.1f} {r['out_tok_s']:>14.1f} "
              f"{(1000 / r['p50']) / base:>10.2f}x")

# %% [markdown]
# **The sentence to write down.** Pick a per-user floor (say 30 tokens/s), find the
# largest concurrency that still clears it, and read off the total throughput there.
# That pair — concurrency and throughput — is your operating point, and it is the only
# thing worth putting in a design document.

# %%
FLOOR = 30.0
ok = [r for r in sweep if r]
fits = [r for r in ok if 1000 / r["p50"] >= FLOOR]
if fits:
    best = max(fits, key=lambda r: r["out_tok_s"])
    print(f"SLO: at least {FLOOR} tok/s per user")
    print(f"operating point: concurrency {best['concurrency']}, "
          f"{1000 / best['p50']:.0f} tok/s per user, {best['out_tok_s']:.0f} tok/s total")
else:
    print("No configuration in this sweep met the floor. Lower the floor, or lower max_model_len.")

# %% [markdown]
# ## 5. Sweep `max_num_batched_tokens`
#
# The last knob, and the one people forget. One engine iteration processes at most this
# many tokens; a prompt bigger than the budget is split across iterations.
#
# Our prompts above are short, so this barely matters — which is itself the lesson: the
# flag only matters when prompts are long. Measure it with a deliberately long prompt.

# %%
LONG = PROMPT * 8
long_tokens = len(TOK(LONG)["input_ids"])
print(f"long prompt is about {long_tokens} tokens")

for budget in (2048, 512):
    print(f"\n=== max-num-batched-tokens {budget} ===")
    p, log_text = start_server(["--gpu-memory-utilization", "0.90", "--max-model-len", "4096",
                                "--max-num-seqs", "16",
                                "--max-num-batched-tokens", str(budget)])
    show(bench(8, log_text, prompt=LONG))
    print(f"  at a {budget}-token budget this prompt needs "
          f"{math.ceil(long_tokens / budget)} or more iterations, chunked around decode steps")
    stop_server(p)

# %% [markdown]
# **Read it.** p95 goes up for the requests that were chunked, and the tail for everybody
# else gets smoother. That is the trade: you move latency from "everyone" to "the one
# request with the huge prompt". Module 10 · Prefill & Prefix Optimization is entirely
# about this knob.
#
# And notice the second effect: that long prompt also needs many more KV blocks, so at
# `max_model_len 4096` your worst-case concurrency is a quarter of what it is at 1,024.
# The memory lesson from 09.1, showing up again.

stop_server(proc)

# %% [markdown]
# ## Your turn
#
# 1. Add `gpu_memory_utilization 0.95`. Does it start? What happened to the pool, and why
#    is a 5% bigger pool not free?
# 2. Set `--max-num-seqs 512`. What fails first — memory, or your latency target?
# 3. Re-run the `max_num_seqs` sweep with `max_model_len 4096`. Which curve moves, and
#    why? (Hint: it is the *same* arithmetic as the KV-per-request table in 09.3.)
# 4. Put your operating point in **My notes** with all four numbers (p50 and p95 latency,
#    per-user speed, total throughput) and the four flags. Module 25 will price it.
# 5. Compare the printed `GPU KV cache size` with your prediction for each `util` and
#    write down the ratio. It is not 1.0, and understanding why is worth more than the
#    sweep itself.