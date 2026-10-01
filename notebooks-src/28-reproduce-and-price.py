# %% [markdown]
# # Lab 28 · Reproduce, measure the noise, and price a design
#
# **Tokens per Second** · Module 28
#
# You will:
# 1. Measure your rig's **run-to-run noise** on a real GPU, before measuring any treatment.
# 2. Apply the repeats-needed formula, k ≈ (2c/δ)², to decide what is defensible.
# 3. Run an interleaved A/B comparison of two engine settings and report a verdict.
# 4. Reproduce the **mechanism** (does a knob move memory traffic?) rather than a headline number.
# 5. Derive a capacity and cost budget from model shape alone, and compare it to what you measured.
# 6. Write the five-line verdict sheet for a real measurement.
#
# This is the lab behind *Reproducing a result* and the *Capstone*. Everything runs on a
# free Colab **T4** (16 GB, FP16, no BF16 tensor cores, no FP8) using small models.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.

# %%
import torch, time, statistics, math
assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
print(torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"memory: {props.total_memory/1e9:.1f} GB")

# %% [markdown]
# ## 1. A tiny benchmark we can repeat
#
# We will use one matrix multiply as our "treatment". It is fast, stable, and its
# variance behaves like a real inference kernel's: shared, virtualised, slightly noisy.

# %%
def time_matmul(m=4096, n=4096, k=4096, iters=50, warmup=10):
    """Return the median time of a [m,k]x[k,n] matmul, in milliseconds."""
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a @ b
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)

med, best = time_matmul()
print(f"median {med:.3f} ms   best {best:.3f} ms")

# %% [markdown]
# ## 2. Step one: measure the rig noise FIRST
#
# Before measuring any "improvement", measure how much the *same* measurement moves.
# Run the identical benchmark 8 times and look at the spread.
#
# This spread, written **c**, is the yardstick every claimed effect will be judged against.

# %%
runs = [time_matmul(iters=40)[0] for _ in range(8)]
for i, r in enumerate(runs):
    print(f"  run {i+1}: {r:.3f} ms")

mean = statistics.mean(runs)
spread = (max(runs) - min(runs)) / mean          # full range as a fraction of the mean
stdev = statistics.stdev(runs) / mean            # relative standard deviation
print()
print(f"mean      {mean:.3f} ms")
print(f"rig noise c (relative stdev) = {stdev*100:.2f}%")
print(f"full range                   = {spread*100:.2f}%")

# %% [markdown]
# On a free Colab T4 you will typically see **c between about 3% and 10%**. Write your
# number down — it is the yardstick for everything in this lab.
#
# On a quieter, dedicated machine c can be under 1%. That is exactly why you cannot quote
# someone else's noise as your own.

# %% [markdown]
# ## 3. Step two: repeats needed, k ≈ (2c/δ)²
#
# Given a claimed effect δ and your measured noise c, how many repeats do you need
# before the effect is defensible?

# %%
def repeats_needed(c, delta):
    """Repeats per arm needed to defend an effect of `delta` (both as fractions)."""
    return math.ceil((2 * c / delta) ** 2)

c = stdev
print(f"your rig noise c = {c*100:.2f}%\n")
print(f"{'claimed effect':>15} | {'repeats k':>10}")
print("-" * 30)
for delta in [0.02, 0.04, 0.06, 0.10, 0.20, 0.30]:
    k = repeats_needed(c, delta)
    bar = "***" * min(k, 10)
    print(f"{delta*100:>14.0f}% | {k:>10}  {bar}")

# %% [markdown]
# Read that table before you run anything else. It is the difference between an
# experiment and an anecdote: a 4% effect on this rig is **not** one run, it is a
# real experiment with real repeats.

# %% [markdown]
# ## 4. Step three: an interleaved A/B comparison
#
# Two arms:
#   **A** = plain matmul
#   **B** = the same matmul with TF32 enabled (a "treatment": cheaper matmul precision)
#
# The crucial part: **interleave** them (A, B, A, B, …) rather than running all of A
# then all of B. Interleaving makes any slow drift in the machine hit both arms
# equally, which shrinks your effective noise. This is the cheapest variance reduction
# there is, and it is what Module 17 called for.

# %%
torch.backends.cuda.matmul.allow_tf32 = False

def ab(iters=40):
    torch.backends.cuda.matmul.allow_tf32 = False
    a = time_matmul(iters=iters)[0]
    torch.backends.cuda.matmul.allow_tf32 = True
    b = time_matmul(iters=iters)[0]
    return a, b

pairs = [ab() for _ in range(8)]
A = [p[0] for p in pairs]
B = [p[1] for p in pairs]
delta_pct = (statistics.mean(A) - statistics.mean(B)) / statistics.mean(A) * 100

print(f"A (no TF32): {statistics.mean(A):.3f} ms   ±{statistics.stdev(A)/statistics.mean(A)*100:.2f}%")
print(f"B (TF32)    : {statistics.mean(B):.3f} ms   ±{statistics.stdev(B)/statistics.mean(B)*100:.2f}%")
print(f"observed effect δ = {delta_pct:.1f}%")

# %% [markdown]
# Now ask the real question — **is that effect bigger than the noise?**
# We have two honest ways to check it: a statistical one and a simple, very effective one.

# %%
# (a) Does the observed effect clear the noise yardstick?
def verdict(delta, c, n_pairs):
    need = repeats_needed(c, delta)
    if need <= n_pairs:
        return f"DEFENSIBLE (k needed {need}, you ran {n_pairs})"
    return f"INCONCLUSIVE (k needed {need}, you ran {n_pairs})"

# (b) The bulletproof check: do the arms' ranges overlap at all?
lo_A, hi_A = min(A), max(A)
lo_B, hi_B = min(B), max(B)
overlap = not (hi_B < lo_A or hi_A < lo_B)
print(f"A range: [{lo_A:.3f}, {hi_A:.3f}] ms")
print(f"B range: [{lo_B:.3f}, {hi_B:.3f}] ms")
print(f"ranges overlap: {overlap}")
print()
print("verdict:", verdict(abs(delta_pct)/100, c, len(pairs)))

# %% [markdown]
# **How to read this.**
# - If the ranges overlap → you have **reproduced** any effect of that size; a δ that
#   small is inside your rig noise, so "no clear effect".
# - If the ranges do **not** overlap → you have a **real, defensible** effect.
# - Either way, the honest sentence is: *"TF32 was X% faster over 8 interleaved repeats;
#   rig noise c = Y%; the effect is {bigger/smaller} than the noise, so I conclude Z."*
#
# That sentence — not a single fast-looking run — is what you put in your verdict sheet.

# %% [markdown]
# ## 5. Reproduce the mechanism, not the number
#
# If you don't have the paper's hardware (you don't — you have a T4), you can still test
# whether the *idea* is sound. A common claim is that **fewer bytes per number means less
# memory traffic and therefore faster decode**. Test the mechanism: read the same matrix
# in FP32, FP16, and INT8, and see whether time tracks bytes moved.

# %%
def bench_dtype(m=4096, n=4096, k=4096, iters=40, dtype=torch.float16):
    a = torch.randn(m, k, device="cuda").to(dtype)
    b = torch.randn(k, n, device="cuda").to(dtype)
    for _ in range(5): a @ b
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a @ b
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)

results = {}
for name, dt in [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]:
    results[name] = bench_dtype(dtype=dt)
    print(f"{name}: {results[name]:.3f} ms")

# %% [markdown]
# This is the **mechanism**, and it is the part that transfers. It does not matter that
# your T4's absolute numbers differ from the paper's A100 — what matters is that *time
# tracks bytes moved and precision*, which is the idea the paper's number rests on.

# %% [markdown]
# ## 6. Derive the capacity and cost budget (no benchmark needed)
#
# Now the capstone's derived budget, computed from model shape alone. These are exact
# formulas — not measurements — and they decide most of a design before you ever run it.

# %%
# Model shapes (params, layers, kv_heads, head_dim). Real public config values.
MODELS = {
    "Qwen2.5-0.5B":  dict(params=0.49e9, layers=24, kv_heads=2,  head_dim=64),
    "Llama-3.2-3B":  dict(params=3.2e9,  layers=28, kv_heads=8,  head_dim=128),
    "Qwen2.5-1.5B":  dict(params=1.5e9,  layers=28, kv_heads=2,  head_dim=128),
    "Llama-3.1-8B":  dict(params=8.03e9, layers=32, kv_heads=8,  head_dim=128),
}

def budget(name, w_bytes=2, kv_bytes=2, gpu_mem=16e9, ctx=2048):
    m = MODELS[name]
    weights = m["params"] * w_bytes
    reserve  = 0.04 * gpu_mem
    kv_tok   = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * kv_bytes
    per_seq  = ctx * kv_tok
    pool     = gpu_mem - weights - reserve
    max_batch = pool / per_seq if pool > 0 else 0
    print(f"{name} ({w_bytes*8}-bit) on {gpu_mem/1e9:.0f}GB, ctx={ctx}:")
    print(f"  weights      {weights/1e9:6.2f} GB")
    print(f"  reserve      {reserve/1e9:6.2f} GB")
    print(f"  KV/token     {kv_tok/1024:6.0f} KB")
    print(f"  KV/request   {per_seq/1e9:6.3f} GB")
    print(f"  max batch    {max_batch:6.1f} seqs" if max_batch else "  DOES NOT FIT")
    print()
    return max_batch

for name in MODELS:
    budget(name)

# %% [markdown]
# **This is where you see the design decisions fall out.** Llama-3.1-8B in BF16 needs
# 16.06 GB of weights — it will not even load on a 16 GB T4. Quantize it to 8-bit and it
# fits with room to spare. *Precision is a memory decision before it is a speed decision.*
#
# Now add context and watch concurrency collapse — the "context is a capacity column" point:

# %%
print("Llama-3.1-8B in BF16 on an 80GB A100, concurrency vs context:")
print(f"{'context':>10} | {'KV/request':>12} | {'max seqs/GPU':>14}")
print("-" * 44)
for ctx in [512, 1024, 2048, 4096, 8192]:
    m = MODELS["Llama-3.1-8B"]
    kv_tok = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * 2
    per_seq = ctx * kv_tok
    pool = 80e9 - 8.03e9 * 2 - 0.04 * 80e9
    print(f"{ctx:>10} | {per_seq/1e9:>11.2f} GB | {pool/per_seq:>14.0f}")

# %% [markdown]
# Doubling the context halves your concurrency. This is why a capacity plan that quotes a
# single "max batch" without naming the context it assumes is incomplete.

# %% [markdown]
# ## 7. Cost per million tokens
#
# The capstone's cost model. Cost depends on **goodput** (tokens served within the SLO),
# not raw throughput — a system that is fast but misses the SLO is worth nothing.

# %%
def cost_per_mtok(gpu_price_per_hr, goodput_tok_s):
    """Dollars per million good tokens, from GPU price and measured goodput."""
    tokens_per_hr = goodput_tok_s * 3600
    return (gpu_price_per_hr / tokens_per_hr) * 1e6

print("Cost per million good tokens (illustrative on-demand prices):")
for name, price in [("T4", 0.35), ("L4", 0.80), ("A100", 1.60), ("H100", 2.50)]:
    for goodput in [2000, 6000]:
        c = cost_per_mtok(price, goodput)
        print(f"  {name:>5} @ {goodput:>5} good tok/s: ${c:6.3f} / M tokens")

# %% [markdown]
# The lesson: **throughput is not cost.** The same GPU at a smaller goodput costs many
# times more per token. Optimizing for goodput-per-dollar (Module 25) is the capstone's
# real objective.

# %% [markdown]
# ## 8. Write your verdict sheet
#
# Fill this in for the A/B test you ran in §4, then for your capstone design. This is the
# artifact the lesson is really about.
#
# ```
# CLAIM       What single number, on what setup, am I testing?
# SETUP MATCH What hardware/workload, and what does the difference cost me?
# NOISE c     My measured run-to-run spread, and how I got it.
# RESULT      Mean ± spread, and how many interleaved repeats.
# VERDICT     Reproduced / did not reproduce / inconclusive — and why.
# ```
#
# For the A/B above, yours should look roughly like:
# *"TF32 was X% faster than FP16 on a 4096³ matmul, 8 interleaved repeats, T4 rig noise
# c ≈ Y%. The effect is {larger/smaller} than the noise, so I conclude {a real speedup /
# no clear effect}. On this rig a 4% effect would need about Z repeats."*

# %% [markdown]
# ## Check yourself
#
# 1. What was **your** rig noise `c`? Would it have been the same on a dedicated GPU?
# 2. For a claimed 6% effect, how many repeats did the formula say you needed? Did you run them?
# 3. Did you **interleave** your A/B arms? What happens to the noise if you don't?
# 4. For the TF32 test: did the ranges overlap? What is the honest verdict?
# 5. In §5, did time track bytes-moved (the mechanism)? Is that the same conclusion as a headline number?
# 6. From §6: at what context does Llama-3.1-8B stop fitting on a T4? Where does concurrency halve on an A100?
# 7. From §7: at a fixed price, how does cost per token change if goodput halves?
# 8. Could a colleague who has never seen your code reach the same verdict from your five lines?
#
# If the answer to any of these is "no", the experiment is not finished — go back and tighten it.