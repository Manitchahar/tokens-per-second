# %% [markdown]
# # Project 17.P · Build an engine selection matrix
#
# **Tokens per Second** · Module 17
#
# You will:
# 1. Build the **derived** column of the matrix yourself, from weight and KV arithmetic.
# 2. Measure the **measured** column by actually running two engines on a T4.
# 3. Fill the **structural** column from documentation, and never mix the three kinds.
# 4. Apply hard constraints, then weighted criteria, and produce a decision you can defend.
#
# Runtime → Change runtime type → **T4 GPU**. Then run the cells top to bottom.
#
# The rule this notebook exists to teach: **most of a good selection matrix costs nothing
# to produce, because it is arithmetic.** Only the measured column needs a GPU, and only
# the measured column should ever be allowed to decide.

# %%
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
print("installed — you may need to RESTART the runtime before the next cell")

# %% [markdown]
# ## 1. The derived column: free, exact, and it eliminates candidates first
#
# Before you benchmark anything you can compute, for any (model, precision, GPU) triple:
#
# ```
#   weights        = params × bytes_per_weight
#   reserve        ≈ 0.04 × total_memory
#   kv_bytes_tok   = 2 × layers × kv_heads × head_dim × 2
#   kv_pool        = total_memory − weights − reserve
#   pool_tokens    = kv_pool ÷ kv_bytes_tok
#   max_batch      = pool_tokens ÷ context
# ```
#
# None of this needs a GPU, and none of it is a guess. It is the column that eliminates
# half your candidates before you have spun anything up.

# %%
GPUS = {
    "t4":  dict(mem=16e9, bw=320e9, flops=65e12,  price=0.35),
    "l4":  dict(mem=24e9, bw=300e9, flops=121e12, price=0.80),
    "a100":dict(mem=80e9, bw=2.0e12, flops=312e12, price=1.60),
    "h100":dict(mem=80e9, bw=3.35e12, flops=989e12, price=2.50),
}
MODELS = {
    "qwen05":  dict(name="Qwen2.5-0.5B",  params=0.49e9, layers=24, kv_heads=2,  head_dim=64),
    "llama3b": dict(name="Llama-3.2-3B",  params=3.2e9,  layers=28, kv_heads=8,  head_dim=128),
    "llama8b": dict(name="Llama-3.1-8B",  params=8.03e9, layers=32, kv_heads=8,  head_dim=128),
    "qwen32b": dict(name="Qwen2.5-32B",  params=32.8e9, layers=64, kv_heads=8,  head_dim=128),
    "llama70b":dict(name="Llama-3.1-70B",  params=70.6e9, layers=80, kv_heads=8,  head_dim=128),
}
BYTES = {2: "BF16", 1: "FP8/INT8", 0.5: "INT4"}

def derived(model_id, gpu_id, bpe=2, ctx=2048):
    m, g = MODELS[model_id], GPUS[gpu_id]
    weights = m["params"] * bpe
    reserve = 0.04 * g["mem"]
    kv_tok  = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * 2
    pool    = g["mem"] - weights - reserve
    tokens  = max(0, int(pool // kv_tok)) if kv_tok else 0
    return dict(
        weights_gb=weights/1e9,
        fits=bool(weights + reserve < g["mem"]),
        kv_pool_gb=pool/1e9,
        pool_tokens=tokens,
        max_batch=tokens // ctx if ctx else 0,
    )

print(f"{'model':<14}{'gpu':<6}{'prec':<8}{'weights':>9}{'fits':>6}{'KV pool':>10}{'tokens':>10}{'batch@2k':>10}")
print("-" * 74)
for mid in MODELS:
    for gid in ("t4", "l4", "a100", "h100"):
        for bpe in (2, 0.5):
            d = derived(mid, gid, bpe)
            print(f"{MODELS[mid]['name']:<14}{gid:<6}{BYTES[bpe]:<8}"
                  f"{d['weights_gb']:>7.1f}GB{'yes' if d['fits'] else 'NO':>6}"
                  f"{d['kv_pool_gb']:>8.1f}GB{d['pool_tokens']:>10,}{d['max_batch']:>10}")

# %% [markdown]
# ## 2. Read the table: this is your matrix's spine
#
# Three things to notice, and none of them required a GPU.
#
# **1. Qwen2.5-32B in BF16 does not fit on a T4 or an L4.** 65.6 GB of weights against
# 24 GB. Not "is slow" — *does not load*. In INT4 it is 16.4 GB and fits an L4 with room
# for roughly 50K tokens of context. This is the hard constraint that decides the edge
# case, and it is arithmetic.
#
# **2. Context length moves the batch cap by more than an order of magnitude.** For
# Llama-3.1-8B on an A100 at 2,048 tokens of context you get a large batch cap; at
# 16,384 the same card gives you a fraction of it. A matrix with no context column is
# not describing a system.
#
# **3. Llama-3.1-70B in BF16 fits exactly one A100 or H100 and leaves almost nothing for
# KV.** 141 GB of weights on an 80 GB card does not fit at all. This is Module 18's
# problem — tensor parallelism — but you should reach that conclusion *here*, from this
# table, before you benchmark anything.
#
# ### Your task
# Extend this function: add a `ctx` sweep, and add the **feasibility verdict** you would
# actually write into a matrix: `fits weights`, `fits with <n> streams at <ctx>`, or `OOM`.

# %%
def verdict(model_id, gpu_id, bpe=2, ctx=2048, want=16):
    d = derived(model_id, gpu_id, bpe, ctx)
    if not d["fits"]:
        return f"OOM: weights alone need {d['weights_gb']:.1f} GB"
    got = d["max_batch"]
    if got < 1:
        return f"OOM: 0 of {want} streams fit at {ctx:,} ctx"
    if got < want:
        return f"fits, but KV caps it at {got} of {want} streams"
    return f"OK: {got} streams at {ctx:,} ctx ({d['max_batch']*ctx/1e6:.1f}M tokens of KV)"

for mid, gid, bpe in [("qwen32b","l4",2), ("qwen32b","l4",0.5), ("llama70b","h100",2), ("llama8b","t4",2)]:
    print(f"{MODELS[mid]['name']:<14} {gid:<5} {BYTES[bpe]:<6} {verdict(mid,gid,bpe)}")

# %% [markdown]
# ## 3. The measured column: run two engines on the same harness
#
# This is the only part that needs the GPU, and it is the part people rush. The rules,
# from Module 17.1:
#
# - **Same workload** — same prompt file, same seed, same output length.
# - **Same arrival pattern** — fixed concurrency, and note it in the report.
# - **Same offered load** — report $\rho$, or at least the concurrency and the achieved
#   throughput so a reader can locate it.
# - **Quality parity** — temperature 0, pinned seed, and *check the tokens match*.
# - **Warm** — discard the first requests.
#
# We will run **vLLM** and **llama.cpp's `llama-cpp-python`**, because both install on a
# free T4 and they are the two extremes: the GPU-native general-purpose engine and the
# portable 4-bit one. SGLang and TensorRT-LLM are structurally similar to one of these
# two for this purpose, and their structural columns are what you will reason about.

# %%
!pip install -q llama-cpp-python
# Building from source on Colab can take several minutes. If the wheel is unavailable
# for your Python version, fall back to the structural column for llama.cpp and say so
# in your report — an honest gap beats a fabricated number.

# %%
import importlib.util
print("llama_cpp available:", importlib.util.find_spec("llama_cpp") is not None)

# %% [markdown]
# ## 4. The harness
#
# Write one collector, use it for every engine. The engine is the only thing that
# changes. If you find yourself writing a second collector, you have broken the contract.

# %%
import time, json, statistics, urllib.request

SERVED = "Qwen/Qwen2.5-0.5B-Instruct"
PARAMS = dict(temperature=0.0, top_p=1.0, seed=0, max_tokens=128, stream=True)

def collect(url, prompts, concurrency=4, warmup=2):
    """One collector, every engine. Returns raw per-request records."""
    import threading
    records, lock = [], threading.Lock()

    def one(prompt):
        body = dict(model=SERVED, prompt=prompt, **PARAMS)
        req = urllib.request.Request(
            f"{url}/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter(); first = None; n = 0
        with urllib.request.urlopen(req) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: "): continue
                if line == "data: [DONE]": break
                if first is None: first = time.perf_counter()
                n += 1
        t1 = time.perf_counter()
        if first is None or n == 0: return
        with lock:
            records.append(dict(ttft=first-t0, e2e=t1-t0, tokens=n, tpot=(t1-first)/max(1,n-1)))

    for _ in range(warmup):
        one(prompts[0])
    records.clear()
    q = list(prompts)
    threads = []
    for i in range(concurrency):
        p = q[i % len(q)]
        t = threading.Thread(target=one, args=(p,)); t.start(); threads.append(t)
        time.sleep(1.0 / max(1, concurrency))   # fixed arrival spacing, not "as fast as possible"
    for t in threads: t.join()
    return records

def summarize(records):
    if not records: return {}
    def pct(xs, p):
        xs = sorted(xs); return xs[min(len(xs)-1, int(p/100*len(xs)))]
    ttft = [r["ttft"] for r in records]
    return dict(
        n=len(records),
        ttft_p50=pct(ttft,50)*1000, ttft_p95=pct(ttft,95)*1000,
        tpot_p50=statistics.median([r["tpot"] for r in records])*1000,
        tok_per_s=sum(r["tokens"] for r in records)/max(1e-9, sum(r["e2e"] for r in records)),
        per_user=1/max(1e-9, statistics.median([r["tpot"] for r in records])),
    )

def parity_tokens(url, prompt, n=48):
    """Greedy decode, return the token ids so two engines can be compared exactly.

    Two engines serving the same weights at temperature 0 with a fixed seed must emit
    the same ids. If they do not, you are timing two different computations and the
    comparison is void — so run this BEFORE you read any timing.
    """
    body = dict(model=SERVED, prompt=prompt, **dict(PARAMS, stream=False))
    req = urllib.request.Request(
        f"{url}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0].get("logprobs", {}).get("tokens") or data["choices"][0]["text"]

PROMPTS = [f"Summarise in one sentence what an inference engine does. (variant {i})"
           for i in range(16)]

# %% [markdown]
# ## 5. Run it
#
# Start vLLM, run the harness, shut it down, then start llama.cpp and run the *same*
# harness against the same prompts. Record the model revision, the flags and the versions
# in your report as you go — that is three of the six contract items, filled in.

# %%
!vllm serve {SERVED} --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.85 > /tmp/vllm.log 2>&1 &

import time, urllib.error
for _ in range(90):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2); break
    except Exception:
        time.sleep(4)
else:
    raise SystemExit("vLLM did not come up — read /tmp/vllm.log")

vllm_res = summarize(collect("http://127.0.0.1:8000", PROMPTS, concurrency=4))
print("vLLM :", json.dumps(vllm_res, indent=2))
# quality parity: the tokens must be identical across engines at temperature 0
VLLM_TOKENS = parity_tokens("http://127.0.0.1:8000", PROMPTS[0])
print("parity baseline captured:", str(VLLM_TOKENS)[:120])

# %%
!pkill -f vllm || true
!pkill -f "VLLM" || true

# %% [markdown]
# ### Now llama.cpp
#
# `llama-cpp-python` serves the same OpenAI-compatible routes, so the same `collect()`
# function works unchanged. The point of this cell is not the number — it is that the
# **harness did not change**.

# %%
# Uncomment once the server above is running on its own port. Note the model file is a
# GGUF, so the `model` field is a path — the ONE thing that differs between the two arms,
# and it is a difference in weights, not in the harness.
#
#   from llama_cpp.server.app import create_app
#   import uvicorn
#   app = create_app(model="/content/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
#   uvicorn.run(app, host="127.0.0.1", port=8001)
#
# Then, in a separate cell:
#
#   cpp_res = summarize(collect("http://127.0.0.1:8001", PROMPTS, concurrency=4))
#   print("llama.cpp:", json.dumps(cpp_res, indent=2))
#
#   # PARITY FIRST. Compare before you compare timings.
#   cpp_tokens = parity_tokens("http://127.0.0.1:8001", PROMPTS[0])
#   print("parity:", cpp_tokens == VLLM_TOKENS)
#
# A False here is a finding, not an annoyance: it means the INT4 weights are not
# producing the FP16 token stream, so the two rows are different systems. Record that
# in the report rather than quietly reporting the faster number.
cpp_res = None   # filled in above

# %% [markdown]
# ## 6. Does the reported difference beat the noise?
#
# From Module 17.1, repeats needed to trust an effect $\delta$ against rig noise $c$:
#
# $$ k \approx (2c/\delta)^2 $$
#
# On a shared free Colab T4, expect $c$ in the range of 5–10%: it is a shared, virtualised
# machine and other tenants show up in `nvidia-smi`. **That is a fact about your rig, and
# it belongs in the report.** It is also why a 4% difference measured on a T4 is not a
# result — you would need $k \approx (2\times 0.08/0.04)^2 = 16$ repeats to claim it.

# %%
def repeats_needed(delta_pct, noise_pct):
    return max(1, round((2*noise_pct/delta_pct)**2))

print(f"{'claimed effect':>15}{'rig noise 5%':>16}{'rig noise 10%':>16}")
for d in (20, 10, 6, 4, 2):
    print(f"{d:>13}%{repeats_needed(d,5):>16}{repeats_needed(d,10):>16}")

# %% [markdown]
# ## 7. Assemble the matrix
#
# Now build the table this module is about. Rows are engines, columns are criteria, and
# **every column is labelled with its kind**. Fill the derived and structural columns from
# this notebook and the documentation; fill the measured column only from runs that met
# the contract.

# %%
MATRIX = [
    # engine,        measured tok/s, measured TTFT p95, derived weights, derived pool, structural notes
    ("vLLM",         vllm_res.get("tok_per_s", 0), vllm_res.get("ttft_p95", 0),
     "0.98 GB",  "~500K tok", "no build; needs CUDA; widest model coverage"),
    ("llama.cpp",    None, None,
     "0.35 GB (INT4)", "n/a — no CUDA context", "no build; runs on CPU/Metal; 4-bit native"),
    ("SGLang",       None, None,
     "0.98 GB",  "~500K tok", "no build; needs CUDA; radix prefix cache"),
    ("TensorRT-LLM", None, None,
     "0.98 GB",  "~500K tok", "BUILD per config; needs CUDA; lowest fixed cost"),
]

hdr = f"{'engine':<14}{'MEASURED tok/s':>15}{'MEASURED p95 TTFT':>19}{'DERIVED weights':>17}  STRUCTURAL"
print(hdr); print("-"*len(hdr))
for name, thr, ttft, w, pool, struct in MATRIX:
    m_thr = f"{thr:>15.1f}" if thr else f"{'not measured':>15}"
    m_tt  = f"{ttft:>17.0f}ms" if ttft else f"{'not measured':>19}"
    print(f"{name:<14}{m_thr}{m_tt}{w:>17}  {struct}")

# %% [markdown]
# ## 8. Hard constraints, then weights, then a decision
#
# The order matters more than the arithmetic.
#
# 1. **Hard constraints** — eliminate rows. No CUDA build available? No 4-bit path? The
#    model does not fit? These are yes/no and they run first, because a row that fails
#    one can never be the answer no matter how it scores elsewhere.
# 2. **Weighted criteria** — only among the survivors. Write the weights down *before*
#    you look at the scores; the weight is the argument.
# 3. **A sensitivity check** — if two candidates are within a few percent, say so and
#    give the tie-breaker (operational familiarity, build complexity) explicitly.

# %%
CRITERIA = {
    "fits our hardware": 3.0,   # hard constraint in practice, but scored here for illustration
    "p95 TTFT headroom": 2.5,
    "goodput per dollar": 3.0,
    "build/deploy simplicity": 1.0,
    "operational familiarity": 0.5,
}

SCORES = {   # 0-10, illustrative — replace with your own measured or derived values
    "vLLM":         dict(zip(CRITERIA, [10, 7, 6, 9, 9])),
    "llama.cpp":    dict(zip(CRITERIA, [10, 5, 8, 9, 4])),
    "SGLang":       dict(zip(CRITERIA, [8,  8, 6, 9, 5])),
    "TensorRT-LLM": dict(zip(CRITERIA, [5, 10, 6, 3, 4])),
}

print(f"{'engine':<14}" + "".join(f"{k[:11]:>13}" for k in CRITERIA) + f"{'weighted':>12}")
print("-" * 100)
ranked = []
for name, sc in SCORES.items():
    total = sum(CRITERIA[k] * sc[k] for k in CRITERIA)
    ranked.append((total, name))
    print(f"{name:<14}" + "".join(f"{sc[k]:>13.1f}" for k in CRITERIA) + f"{total:>12.1f}")
ranked.sort(reverse=True)
print()
print("ranking:", " > ".join(n for _, n in ranked))
gap = ranked[0][0] - ranked[1][0]
print(f"top-two gap: {gap:.2f} weighted points "
      f"({'decisive' if gap > 5 else 'CLOSE — state a tie-breaker explicitly'})")

# %% [markdown]
# ## 9. Your turn
#
# Extend this notebook into the deliverable. Required:
#
# 1. **Two or more GPUs** in the derived table, and a context sweep, with feasibility
#    verdicts written in words rather than colours.
# 2. **At least two engines measured** on the same harness, with the versions, flags and
#    revisions recorded, and a quality-parity check on the token streams.
# 3. **Interleaved repeats** (A, B, A, B) rather than all-A-then-all-B, and a stated
#    run-to-run spread $c$.
# 4. **A weighted decision** with weights written before the scores, plus a sensitivity
#    check on the top two.
# 5. **A `MATRIX.md`** where every column is headed `MEASURED`, `DERIVED` or `STRUCTURAL`.
#
# ## Check yourself
#
# - Could a reader tell, from any single cell, whether it is a measurement or a guess?
# - Does your weighted decision survive a 20% change in the most important weight?
# - Have you stated, in one sentence, the operating point and the engine you chose?
# - If your top two are close, have you named the tie-breaker instead of pretending
#   the numbers decided it?

# %% [markdown]
# ## Where this goes next
#
# Module 18 (Multi-GPU Parallelism) takes the row you *could not eliminate*: any model
# whose weights exceed one card. Your derived column already told you which models those
# are — now you find out what communication they cost.
