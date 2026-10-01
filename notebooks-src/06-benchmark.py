# %% [markdown]
# # Project 06 · Build your benchmark harness
#
# **Tokens per Second** · Module 06
#
# You are going to build a benchmark harness you can actually keep: one that measures TTFT, TPOT, ITL, throughput and percentiles on a **real** model, sweeps concurrency and context, and produces a report with an operating envelope.
#
# This notebook runs top to bottom on a **free Colab T4 (16 GB, FP16 — no BF16 tensor cores, no FP8)**. We use `Qwen2.5-0.5B-Instruct` so the weights, the KV cache and the batches all fit with room to sweep.
#
# What you will produce:
#
# 1. A **timing harness** that measures per-request latency correctly (stream, client-side clock, real tokenizer counts).
# 2. A **concurrency sweep** with medians and percentiles, not means.
# 3. A **context sweep**, showing how the safe region moves.
# 4. A **goodput** number against your own SLO.
# 5. An **operating envelope** written out as a table.
#
# Keep this file. In Module 09 you will run the same structure against vLLM and compare.

# %% [markdown]
# ## 0. Setup
#
# Install `transformers` if needed and check the GPU. The T4 has 16 GB and supports FP16 but **not** BF16, so we use FP16 throughout.

# %%
import json, math, time, random, statistics as stats

import torch

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

props = torch.cuda.get_device_properties(0)
print(f"GPU        {props.name}")
print(f"Memory     {props.total_memory/1e9:.1f} GB")
print(f"supports BF16 tensor cores: {torch.cuda.is_bf16_supported()}  (T4: False) -> use FP16")

DTYPE = torch.float16


def pctl(values, p):
    """Nearest-rank percentile. Safe for short lists (unlike sorted()[int(p*len)])."""
    if not values:
        return float('nan')
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]

# %% [markdown]
# ## 1. A fixed, seeded workload
#
# Module 06's second lesson: **the workload is half the measurement**. We build one here with a pinned seed and recorded length distributions, so two runs of this notebook measure the same thing.
#
# We use a mix of short and long prompts, and three output-length targets, so the sweep is not a single point.

# %%
def make_workload(n=24, seed=0):
    """A deterministic workload: prompts of varying length, and target output lengths."""
    rng = random.Random(seed)
    words = ("inference serving latency throughput kernel cache batching scheduler "
             "token memory bandwidth prefill decode quantize kernel roofline").split()
    items = []
    for i in range(n):
        # lengths spread from ~20 to ~400 tokens of prompt
        target_words = int(math.exp(rng.uniform(math.log(20), math.log(400))))
        prompt = " ".join(rng.choice(words) for _ in range(target_words)) + "\nSummarise the above."
        out_target = rng.choice([32, 64, 128])   # we will cap generation at this many tokens
        items.append({"prompt": prompt, "max_new_tokens": out_target})
    return items

WORKLOAD = make_workload(seed=0)
print(f"{len(WORKLOAD)} requests")
print(f"prompt words: min {min(len(w['prompt'].split()) for w in WORKLOAD)}, "
      f"max {max(len(w['prompt'].split()) for w in WORKLOAD)}, "
      f"median {int(stats.median(len(w['prompt'].split()) for w in WORKLOAD))}")

# %% [markdown]
# ## 2. Load the model
#
# `Qwen2.5-0.5B-Instruct` is small enough to sweep aggressively on a T4. We load once and reuse it — **model loading is not part of the measurement**, exactly like Module 22's warm-up lesson.

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="cuda").eval()
print("model loaded")

# The real token counts of our workload — this is what we report, never a character estimate.
prompt_lens = [len(tokenizer(w["prompt"]).input_ids) for w in WORKLOAD]
print(f"prompt TOKENS: min {min(prompt_lens)}, max {max(prompt_lens)}, median {int(stats.median(prompt_lens))}")

# %% [markdown]
# ## 3. The timing harness
#
# This is the heart of the project. Three rules, straight from the lessons:
#
# 1. **Stream the response.** Without streaming there is no TTFT.
# 2. **Clock on the client**, from before we start generating to the moment each token is available.
# 3. **Count tokens with the tokenizer**, never by splitting text.
#
# We generate with `model.generate(..., streamer=...)`, so we get a callback per token. That is our ITL.

# %%
class LatencyCollector:
    """Collects TTFT, per-token ITLs and end-to-end time for one streaming generation."""

    def __init__(self):
        self.stamps = []   # monotonic timestamp when each token became available

    def start(self, limit):
        self.t0 = time.perf_counter()
        self.stamps = []
        self.limit = limit

    def token(self):
        self.stamps.append(time.perf_counter())

    def finish(self):
        self.e2e = self.stamps[-1] - self.t0
        self.ttft = self.stamps[0] - self.t0
        itls = [self.stamps[i] - self.stamps[i-1] for i in range(1, len(self.stamps))]
        self.itls = itls
        self.tpot = (sum(itls) / len(itls)) if itls else None
        self.n_tokens = len(self.stamps)
        return self


class TokenStreamer:
    """Minimal streamer: records a client-side timestamp each time a token is produced."""

    def __init__(self, collector):
        self.collector = collector

    def put(self, input_ids):
        self.collector.token()

    def end(self):
        pass


from transformers import StoppingCriteria, StoppingCriteriaList


class NTokenStopper(StoppingCriteria):
    """Stops generation after exactly `n` new tokens.

    This is the supported way to end early: no exceptions thrown out of the
    generate loop, which keeps the timing we collected intact.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[-1] >= self.n


def run_one(prompt, max_new_tokens, temperature=0.0):
    """Generate one response with streaming timestamps. Returns a LatencyCollector."""
    coll = LatencyCollector()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[-1]

    # `n` counts the FULL sequence (prompt + generated), so subtract the prompt.
    stopper = StoppingCriteriaList([NTokenStopper(prompt_len + max_new_tokens)])
    streamer = TokenStreamer(coll)

    coll.start(max_new_tokens)
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            streamer=streamer,
            stopping_criteria=stopper,
            pad_token_id=tokenizer.pad_token_id,
        )
    if coll.stamps:
        return coll.finish()
    return None   # degenerate (empty) generation

print("harness ready")

# %% [markdown]
# ## 4. Warm up
#
# The first few generations pay kernel compilation, allocation and clock ramp. We **discard** them — we never average warm-up into the result. Warm until the timing is stable.

# %%
def warmup(n=5):
    """Run n throwaway generations; return the last TTFT so we can see it settle."""
    last = None
    for i in range(n):
        c = run_one(WORKLOAD[0]["prompt"], WORKLOAD[0]["max_new_tokens"])
        if c:
            last = c.ttft
        print(f"  warmup {i+1}: ttft {last*1000:.0f} ms" if last else "  warmup: degenerate")
    return last

print("warming up...")
warmup(5)
print("steady state reached — now measuring.")

# %% [markdown]
# ## 5. Measure one request and read every metric
#
# Before sweeping, run a single request and print all four latency numbers. Get comfortable reading them.

# %%
coll = run_one(WORKLOAD[3]["prompt"], WORKLOAD[3]["max_new_tokens"])
print(f"tokens generated : {coll.n_tokens}")
print(f"TTFT             : {coll.ttft*1000:.1f} ms")
print(f"TPOT (mean ITL)  : {coll.tpot*1000:.2f} ms   -> {1/coll.tpot:.1f} tok/s per user")
print(f"end-to-end       : {coll.e2e*1000:.1f} ms")
print(f"ITL p50 / p95    : {stats.median(coll.itls)*1000:.2f} / {pctl(coll.itls, 0.95)*1000:.2f} ms")
print(f"ITL min / max    : {min(coll.itls)*1000:.2f} / {max(coll.itls)*1000:.2f} ms")
print()
print("Note: TPOT is the MEAN gap; the p95 is the tail. If max >> p95, the mean is hiding a stutter.")

# %% [markdown]
# ## 6. Concurrency: does batching help, and where does it stop?
#
# The whole of Module 00 in one experiment. We generate N requests and watch how per-user speed and total throughput move together.
#
# Note we run these **sequentially**, so this measures latency under a growing load, not true parallel batching (that needs multiple streams sharing one batch — which is what a serving engine does, and what you will measure in Module 09). For a single-model harness, the honest thing is to report it as such.

# %%
def bench_concurrency(batch_sizes=(1, 2, 4, 8), repeats=3):
    """For each batch size, generate that many requests and report per-user and total throughput."""
    print(f"{'batch':>6} {'p50 TTFT':>10} {'p95 TTFT':>10} {'p50 TPOT':>10} {'tok/s total':>12} {'tok/s user':>11}")
    print("-" * 64)
    results = []
    for B in batch_sizes:
        run_ttfts, run_tpots, total_tokens, wall = [], [], 0, 0.0
        for rep in range(repeats):
            reqs = [WORKLOAD[(rep*B + i) % len(WORKLOAD)] for i in range(B)]
            t_start = time.perf_counter()
            for r in reqs:
                c = run_one(r["prompt"], r["max_new_tokens"])
                if c:
                    run_ttfts.append(c.ttft)
                    run_tpots.append(c.tpot)
                    total_tokens += c.n_tokens
            wall += time.perf_counter() - t_start
        p50_ttft = stats.median(run_ttfts)
        p95_ttft = pctl(run_ttfts, 0.95)
        p50_tpot = stats.median(run_tpots)
        thr_total = total_tokens / wall if wall > 0 else 0
        thr_user = total_tokens / (wall * B) if wall > 0 else 0
        results.append({"batch": B, "p50_ttft": p50_ttft, "p95_ttft": p95_ttft,
                        "p50_tpot": p50_tpot, "thr_total": thr_total, "thr_user": thr_user})
        print(f"{B:>6} {p50_ttft*1000:>9.0f}m {p95_ttft*1000:>9.0f}m {p50_tpot*1000:>9.2f}m "
              f"{thr_total:>12.1f} {thr_user:>11.1f}")
    return results

CONCURRENCY = bench_concurrency(batch_sizes=(1, 2, 4, 8), repeats=2)

# %% [markdown]
# **How to read that table.** As you serve more requests together, per-user speed (`tok/s user`) drops and total throughput (`tok/s total`) rises. That is the latency–throughput trade-off. The question this lesson adds is: *at what point does the SLO break?*

# %% [markdown]
# ## 7. Goodput against your own SLO
#
# Pick an SLO now — for this small interactive model, **TTFT under 1 second** is reasonable. Then compute goodput as the fraction of requests that met it, times throughput.
#
# This is the number you would put on a dashboard.

# %%
SLO_TTFT = 1.0  # seconds

def goodput_table(results, slo=SLO_TTFT):
    print(f"SLO: p95 TTFT < {slo*1000:.0f} ms")
    print(f"{'batch':>6} {'throughput':>12} {'p95 TTFT':>10} {'meets SLO':>11} {'goodput':>10}")
    print("-" * 52)
    out = []
    for r in results:
        meets = r["p95_ttft"] < slo
        good = r["thr_total"] if meets else 0.0
        out.append({"batch": r["batch"], "thr_total": r["thr_total"],
                    "p95_ttft": r["p95_ttft"], "meets": meets, "goodput": good})
        print(f"{r['batch']:>6} {r['thr_total']:>12.1f} {r['p95_ttft']*1000:>9.0f}m "
              f"{'yes' if meets else 'NO':>11} {good:>10.1f}")
    return out

GOODPUT = goodput_table(CONCURRENCY)
best = max(GOODPUT, key=lambda x: x["goodput"])
print(f"\nBest goodput at batch {best['batch']} -> {best['goodput']:.1f} tok/s")

# %% [markdown]
# ## 8. Context sweep: move the safe region
#
# Now the second axis. Longer prompts mean more prefill compute and a bigger KV cache. We build prompts of increasing length and watch TTFT grow — and we watch GPU memory fill.

# %%
def long_workload(length_words):
    words = ("inference serving latency throughput kernel cache batching "
             "scheduler token memory bandwidth").split()
    rng = random.Random(1234)
    return " ".join(rng.choice(words) for _ in range(length_words)) + "\nSummarise."

def bench_context(lengths=(50, 200, 800, 2000), repeats=3):
    print(f"{'words':>7} {'~tokens':>8} {'p50 TTFT':>10} {'p95 TTFT':>10} {'GPU mem GB':>11}")
    print("-" * 50)
    torch.cuda.reset_peak_memory_stats()
    out = []
    for L in lengths:
        prompt = long_workload(L)
        ttfts = []
        for _ in range(repeats):
            c = run_one(prompt, 64)
            if c:
                ttfts.append(c.ttft)
        approx_tokens = len(tokenizer(prompt).input_ids)
        mem = torch.cuda.max_memory_allocated() / 1e9
        p50 = stats.median(ttfts)
        p95 = pctl(ttfts, 0.95)
        out.append({"words": L, "approx_tokens": approx_tokens, "p50_ttft": p50, "p95_ttft": p95, "mem_gb": mem})
        print(f"{L:>7} {approx_tokens:>8} {p50*1000:>9.0f}m {p95*1000:>9.0f}m {mem:>10.2f}")
    return out

CONTEXT = bench_context()

# %% [markdown]
# **How to read that table.** TTFT grows with prompt length — this is prefill, and it is compute-bound. On a bigger model the growth is steeper. And peak memory climbs too, because the KV cache for a long prompt is large.

# %% [markdown]
# ## 9. Break it on purpose: find the memory edge
#
# The envelope's hard boundary is memory. Let's push toward it and watch what happens. We generate with a long context and a large `max_new_tokens`, growing the KV cache until the T4 (16 GB) complains.
#
# This is the "step past the edge on purpose" step. Run it once, observe the failure mode, then set a guardrail below it.

# %%
def find_memory_edge(steps=(256, 512, 1024, 2048), base_words=1000):
    print("pushing the KV cache until the 16 GB T4 refuses...")
    for out_len in steps:
        prompt = long_workload(base_words)
        try:
            torch.cuda.reset_peak_memory_stats()
            c = run_one(prompt, out_len)
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  max_new_tokens={out_len:>5}  OK   peak mem {mem:>5.2f} GB   "
                  f"p50 tpot {c.tpot*1000:.1f} ms")
        except torch.cuda.OutOfMemoryError:
            print(f"  max_new_tokens={out_len:>5}  OUT OF MEMORY  <-- this is your boundary")
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  max_new_tokens={out_len:>5}  OUT OF MEMORY  <-- this is your boundary")
                torch.cuda.empty_cache()
                break
            raise
        torch.cuda.empty_cache()

find_memory_edge()

# %% [markdown]
# Whatever happened above is your **memory edge** for this model on this GPU. Now write the guardrail: a `max_new_tokens` and a context limit comfortably below it, so that in a real service you fail fast with a clear error instead of crashing for everyone.
#
# For a 16 GB T4 running this 0.5B model, a sensible production guardrail is roughly: `max_model_len ≈ 4096` and `max_num_seqs` chosen so total KV stays under ~12 GB. You would enforce these in the engine config (Module 09) rather than hoping.

# %% [markdown]
# ## 10. The envelope table and the report
#
# We bring it all together. The deliverable is a small JSON report plus a printed summary you can paste into a design doc.

# %%
report = {
    "gpu": props.name,
    "gpu_memory_gb": round(props.total_memory / 1e9, 1),
    "dtype": "float16",
    "model": MODEL_ID,
    "workload": {"n": len(WORKLOAD), "seed": 0,
                 "median_prompt_tokens": int(stats.median(prompt_lens))},
    "slo": {"metric": "p95_ttft", "threshold_s": SLO_TTFT},
    "concurrency_sweep": CONCURRENCY,
    "goodput_sweep": GOODPUT,
    "context_sweep": CONTEXT,
    "best_goodput_batch": best["batch"],
    "best_goodput_tok_per_s": round(best["goodput"], 1),
}

with open("benchmark_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("=" * 60)
print("BENCHMARK REPORT")
print("=" * 60)
print(f"GPU / model     : {props.name} / {MODEL_ID} (FP16)")
print(f"Workload        : {len(WORKLOAD)} requests, median prompt {int(stats.median(prompt_lens))} tokens, seeded")
print(f"SLO             : p95 TTFT < {SLO_TTFT*1000:.0f} ms")
print(f"Best goodput    : batch {best['batch']} -> {best['goodput']:.1f} tok/s")
print(f"Per-user at best: {GOODPUT[0]['thr_total']:.1f} tok/s at batch 1")
print(f"Saved           : benchmark_report.json")
print("=" * 60)

# %% [markdown]
# ## 11. Reflect
#
# A few questions to answer in your **notes**:
#
# 1. From your concurrency sweep: at what batch did p95 TTFT first exceed your SLO? That is the edge.
# 2. From your context sweep: how did TTFT grow from 50 to 2000-word prompts — roughly linear, or steeper? Why?
# 3. Your best goodput was at some batch size. Was it the largest batch you tried, or smaller? What does that tell you about the right operating point?
# 4. Look at the `benchmark_report.json`. Could another engineer reproduce this run exactly from it? If not, what is missing?
#
# ## What to build next
#
# This harness measures a **single** model in a Python loop, which is the honest baseline. In Module 09 you will run the same measurement structure against **vLLM**, which batches requests across users for real — and you will see the concurrency sweep behave very differently, because the engine's continuous batching actually shares weight reads across in-flight requests the way this loop cannot.
#
# Keep `benchmark_report.json`. Comparing your loop's numbers to vLLM's on the same GPU and workload is the project at the end of Module 09.