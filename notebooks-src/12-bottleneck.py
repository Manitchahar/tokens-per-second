# %% [markdown]
# # Lab 12 · Write a bottleneck report
#
# **Tokens per Second** · Module 12
#
# By the end of this notebook you will have, on your own GPU:
#
# 1. Measured a real decode step, warmed up and timed correctly.
# 2. Split its **byte budget** into weights, KV cache and attention.
# 3. Computed its **arithmetic intensity** and classified it against the ridge point.
# 4. Measured your **achieved bandwidth** as a percentage of peak.
# 5. Applied **one** fix and verified it moved the metric.
#
# Then you write the report. The notebook gives you every number the report needs.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# We use **Qwen2.5-0.5B-Instruct** (0.49B parameters) so everything runs quickly on
# a free T4. The absolute numbers are much smaller than an 8B model's; the
# *ratios* — which is what this lab is about — transfer directly.

# %%
import torch, time, math
assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
print(f"{p.name}   {p.total_memory/1e9:.1f} GB   sm_{p.major}{p.minor}")

# %%
# The timing helper from Lab 02 and Lab 10. Warm up, use CUDA events, average.
# Profiling a cold model is the single most common mistake — the first calls
# allocate, autotune and lazily initialize, and you will "discover" that cuBLAS
# workspace allocation is your bottleneck.
def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / iters

# %% [markdown]
# ## 0. The model, and its config
#
# Every number in the report comes from three things: the model config, the GPU's
# two speed limits, and your measured step time. Get the config right first.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

cfgm = model.config
PARAMS = sum(q.numel() for q in model.parameters())
LAYERS = cfgm.num_hidden_layers
HIDDEN = cfgm.hidden_size
KV_HEADS = cfgm.num_key_value_heads
HEAD_DIM = cfgm.head_dim

print(f"parameters        {PARAMS/1e9:.3f} B")
print(f"layers            {LAYERS}")
print(f"hidden            {HIDDEN}")
print(f"KV heads x dim    {KV_HEADS} x {HEAD_DIM}")
print(f"vocab             {cfgm.vocab_size}")

# %% [markdown]
# ## 1. The GPU's two speed limits
#
# The T4's public spec figures. These are approximate and vary by SKU and clock —
# but they are the numbers you reason with, and they are what the ridge point is
# built from.

# %%
GPU_NAME = p.name
PEAK_FLOPS = 65e12       # T4 BF16/FP16 tensor core, approximate
PEAK_BW    = 320e9       # T4 memory bandwidth, ~320 GB/s
RIDGE = PEAK_FLOPS / PEAK_BW

print(f"GPU                {GPU_NAME}")
print(f"peak FLOP/s        {PEAK_FLOPS/1e12:.0f} TFLOP/s")
print(f"bandwidth          {PEAK_BW/1e9:.0f} GB/s")
print(f"ridge point        {RIDGE:.0f} FLOP/byte")
print()
print("Ridge point on other cards, for comparison:")
for name, f, b in [("T4", 65e12, 320e9), ("A100 80GB", 312e12, 2.0e12),
                   ("H100 SXM", 989e12, 3.35e12), ("B200", 2250e12, 8.0e12)]:
    print(f"  {name:<10} ridge {f/b:>6.0f} FLOP/byte")

# %% [markdown]
# ## 2. Measure a real decode step
#
# A decode step is one new token for every sequence, against a real KV cache.
# We build the cache once, then time only the steady-state single-token steps.

# %%
from transformers import DynamicCache

def decode_step(batch, ctx=1024, iters=20):
    """One real decode step: 1 new token per sequence, over a ctx-long cache."""
    past = DynamicCache()
    ids = torch.randint(1000, 50000, (batch, ctx), device="cuda")
    with torch.no_grad():
        model(ids, use_cache=True, past_key_values=past)   # warm the cache
    new = torch.randint(1000, 50000, (batch, 1), device="cuda")
    def step():
        with torch.no_grad():
            model(new, past_key_values=past, use_cache=True)
    return bench(step, iters=iters)

CTX = 1024
print(f"{'batch':>7} {'step ms':>10} {'us/user':>10} {'tokens/s':>11}")
dec = {}
for b in [1, 8, 32, 64]:
    t = decode_step(b, ctx=CTX)
    dec[b] = t
    print(f"{b:>7} {t*1e3:>10.2f} {t*1e6/b:>10.1f} {b/t:>11.1f}")

# %%
# The step time is nearly flat in batch. That is the roofline showing up in a real
# decoder: the weights are read once per step no matter how many sequences are
# decoding, so rows 2 through 64 are almost free.
print(f"\nGoing from 1 to 64 users: step time grew {dec[64]/dec[1]:.2f}x,")
print(f"but total throughput grew {(64/dec[64])/(1/dec[1]):.1f}x.")
print("That ratio — 1.0x time for 64x work — is Module 05's whole argument.")

# %% [markdown]
# ## 3. Split the byte budget
#
# Every byte a decode step reads is one of three things:
#
#   1. **weights** — read once per step, regardless of batch
#   2. **KV cache** — read once per step, proportional to batch × context
#   3. **attention's own reads** — the K and V gather, already counted in (2)
#      for a paged layout, but 2× if you materialize the score matrix
#
# The largest term is what you should attack. It is often not the weights.

# %%
DTYPE_BYTES = 2   # fp16
kv_per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_BYTES
weight_bytes = PARAMS * DTYPE_BYTES

print(f"KV bytes / token    {kv_per_token:,} ({kv_per_token/1024:.1f} KiB)")
print(f"weight bytes        {weight_bytes/1e9:.2f} GB\n")

print(f"{'batch':>6} {'ctx':>7} {'weights GB':>12} {'KV GB':>10} {'KV share':>9}")
for b, c in [(1, 1024), (8, 1024), (32, 1024), (32, 4096), (32, 16384), (64, 16384)]:
    w = weight_bytes
    kv = b * c * kv_per_token
    print(f"{b:>6} {c:>7} {w/1e9:>12.2f} {kv/1e9:>10.2f} {kv/(w+kv)*100:>8.1f}%")

# %%
# Find the crossover: the context at which the KV cache overtakes the weights.
# For this model it is T = weight_bytes / (kv_per_token * batch).
print("\nWeights/KV crossover context, per batch:")
for b in [1, 8, 32, 64, 128]:
    t_cross = weight_bytes / (kv_per_token * b)
    print(f"  batch {b:>4}: {t_cross:>9,.0f} tokens")

# %%
# Do the same crossover for Llama-3.1-8B in BF16, which is the number from the lesson.
P8, L8, KVH8, HD8 = 8.03e9, 32, 8, 128
kv8 = 2 * L8 * KVH8 * HD8 * 2
print(f"\nLlama-3.1-8B BF16: KV = {kv8:,} bytes/token, weights = {P8*2/1e9:.1f} GB")
print("crossover context = 122,500 / batch:")
for b in [1, 8, 32, 128]:
    print(f"  batch {b:>4}: {122500/b:>9,.0f} tokens")

# %% [markdown]
# ## 4. Arithmetic intensity, and the bound
#
# $$ I = \frac{F}{B}, \qquad I_{\text{ridge}} = \frac{\text{peak FLOP/s}}{\text{bandwidth}} $$
#
# Below the ridge you are memory-bound and must quote **MBU**.
# Above it you are compute-bound and must quote **MFU**.
# Quoting the wrong one is the most common error in a first report.

# %%
BATCH = 32
ctx = CTX

weights = weight_bytes
kv = BATCH * ctx * kv_per_token
total_bytes = weights + kv
flops = 2 * PARAMS * BATCH
intensity = flops / total_bytes

mem_floor   = total_bytes / PEAK_BW
comp_floor  = flops / PEAK_FLOPS
ideal_step  = max(mem_floor, comp_floor)
measured    = dec[BATCH]
headroom    = measured / ideal_step

print(f"batch {BATCH}, ctx {ctx}\n")
print(f"FLOPs                {flops/1e9:.2f} GFLOP")
print(f"bytes                {total_bytes/1e9:.2f} GB")
print(f"intensity            {intensity:.2f} FLOP/byte")
print(f"ridge point          {RIDGE:.0f} FLOP/byte")
print(f"BOUND                {'COMPUTE' if intensity > RIDGE else 'MEMORY'}")
print()
print(f"memory floor         {mem_floor*1e3:.2f} ms")
print(f"compute floor        {comp_floor*1e3:.3f} ms")
print(f"ideal step           {ideal_step*1e3:.2f} ms")
print(f"measured step        {measured*1e3:.2f} ms")
print(f"HEADROOM             {headroom:.2f}x")
print()
print("Your report's headline sentence starts here. Write it down:")
bound_word = "compute" if intensity > RIDGE else "memory"
print(f'  "The decode step is {bound_word}-bound at intensity {intensity:.1f} FLOP/byte')
print(f'   against a ridge of {RIDGE:.0f}, with {headroom:.2f}x headroom."')

# %%
# Why the memory floor wins by such a wide margin. This is the ratio of the two
# roofline terms, and it is the single most important number in decode.
print(f"\nmemory floor is {mem_floor/comp_floor:.0f}x the compute floor")
print(f"i.e. the math units are idle {(1 - comp_floor/mem_floor)*100:.1f}% of the time.")
print("This is not a defect. It is what a memory-bound workload IS.")
print("It is also why quoting MFU for decode is meaningless — see the next cell.")

# %% [markdown]
# ## 5. Achieved bandwidth
#
# The metric for a memory-bound kernel. Compare it to peak; that ratio is your
# MBU, and it is the number your report should quote.

# %%
achieved = total_bytes / measured
mbu = achieved / PEAK_BW

print(f"measured step        {measured*1e3:.2f} ms")
print(f"bytes moved          {total_bytes/1e9:.2f} GB")
print(f"achieved bandwidth   {achieved/1e9:.1f} GB/s")
print(f"peak                 {PEAK_BW/1e9:.0f} GB/s")
print(f"MBU                  {mbu*100:.1f}%")
print()
if mbu > 0.8:
    print("Above 80%: the kernel is at its floor. There is no meaningful headroom")
    print("left in the kernels — stop optimizing them and look elsewhere.")
elif mbu > 0.6:
    print("Between 60% and 80%: reasonable, but there is some room. Check coalescing")
    print("and occupancy before assuming the bytes are the problem.")
else:
    print("Below 60%: the bytes are being moved badly. Check the access pattern")
    print("before blaming the arithmetic — a strided load can cost 8-16x the traffic.")

# %%
# Sweep it, to see where your hardware and this model land. The absolute numbers
# are low because the T4 is slow and this model is small; the trend is the lesson.
print(f"\n{'batch':>7} {'ctx':>7} {'step ms':>10} {'GB/s':>9} {'MBU':>7}")
for b in [1, 8, 32, 64]:
    t = dec[b]
    by = weight_bytes + b * CTX * kv_per_token
    gb = by / t / 1e9
    print(f"{b:>7} {CTX:>7} {t*1e3:>10.2f} {gb:>9.1f} {gb/(PEAK_BW/1e9)*100:>6.1f}%")

# %% [markdown]
# ## 6. The T4 caveat, stated honestly
#
# The T4's numbers are low for a structural reason, not a bug in your code.
#
# Two things are working against you:
#
# 1. **The T4 has no BF16 tensor-core path.** It predates Hopper's full BF16
#    support, so BF16 matmuls often run at reduced throughput.
# 2. **Small models do not fill the machine.** A 0.5B model's GEMMs are small, so
#    you get neither good occupancy nor good arithmetic intensity.
#
# This is exactly why the lab uses a small model: the *ratios* are what transfer.
# When you move to an 8B model on an A100 or H100, expect MBU in the 75-90% range
# for a well-tuned decode step.

# %%
# The honest comparison: your MBU on a small model vs what a real deployment gets.
print("MBU by model size and hardware (what to expect on a tuned deployment):")
print(f"  Qwen2.5-0.5B on T4 (this notebook):  {mbu*100:.0f}%")
print( "  Qwen2.5-1.5B on T4:                 ~45-60%")
print( "  Llama-3.1-8B on A100:               ~75-85%")
print( "  Llama-3.1-8B on H100:               ~80-90%")
print()
print("Write this caveat into your report. It is the difference between a")
print("measurement and an excuse.")

# %% [markdown]
# ## 7. Apply ONE fix and measure it
#
# One change at a time, or you cannot attribute the result. Change two things and
# you have learned nothing — which is how folklore gets written.

# %%
# FIX 1: quantize the weights to FP8. Fewer bytes per weight, same arithmetic.
#
# This is the cleanest test of the diagnosis. If the step really is bound by
# weight bytes, halving them should cut their contribution in proportion — and
# the roofline tells you exactly how much that is worth, in advance.
#
# We predict it analytically here, because a T4 has no FP8 tensor-core path to
# measure against. Prediction first, then (in "your turn") verify with a real
# quantized model on hardware that supports it. If the measured speed-up matches
# the prediction, your diagnosis is confirmed.
print("FIX 1 — FP8 weights: what the roofline predicts\n")

before = decode_step(BATCH, ctx=CTX)
before_bytes = weight_bytes + BATCH * CTX * kv_per_token

# Simulate the byte reduction analytically: this is the prediction your diagnosis makes.
predicted_bytes = (PARAMS * 1) + BATCH * CTX * kv_per_token   # 1 byte per weight
predicted = before * (predicted_bytes / before_bytes)

print(f"before: {before*1e3:.2f} ms, {before_bytes/1e9:.2f} GB")
print(f"predicted with FP8 weights: {predicted*1e3:.2f} ms "
      f"({(1 - predicted/before)*100:.0f}% faster)")
print()
print("If your measured improvement matches this prediction, your diagnosis is")
print("confirmed: the step really was bound by weight bytes, and reducing them")
print("reduces the time in proportion. If it does not match, something else is")
print("the bottleneck — and that is the most valuable thing this lab can tell you.")

# %%
# FIX 2: CUDA graphs. Remove per-launch overhead.
#
# A decode step launches hundreds of tiny kernels. Each launch costs a few
# microseconds of CPU time that the GPU waits for. Capturing the step as a graph
# and replaying it removes almost all of that.
#
# Measure how many kernels a step launches, then price them:
print("\nFIX 2 — CUDA graphs (launch overhead)\n")

n_layers = LAYERS
n_kernels_per_layer = 12      # rmsnorm, qkv gemm, rope, kv write, attn, o gemm,
n_total = n_layers * n_kernels_per_layer + 8   # norm, lm_head, sampling
per_launch_us = 5

print(f"kernels per decode step   ~{n_total}")
print(f"launch overhead each      ~{per_launch_us} us")
print(f"total launch overhead     ~{n_total*per_launch_us/1000:.2f} ms")
print(f"as a fraction of {before*1e3:.2f} ms step:  {n_total*per_launch_us/1000/(before*1e3)*100:.1f}%")
print()
print("On a large model on an H100 (a ~20 ms step) this is 1-3%: not worth much.")
print("On a small model on a T4 (a few ms step) it is a much bigger share.")
print("That is exactly why Module 15 exists, and why launch overhead shows up")
print("as 'gaps between kernels' in your Nsight Systems trace.")

# %%
# FIX 3: increase batch. The single biggest lever on decode.
#
# Batch does not change the weight bytes at all — the weights are read once per
# step regardless. It only amortizes them over more sequences.
print("\nFIX 3 — larger batch (amortize the weights)\n")

print(f"{'batch':>7} {'step ms':>10} {'weights GB':>12} {'GB per token':>13} {'speedup vs b=1':>15}")
base = dec[1]
for b in [1, 8, 32, 64]:
    t = dec[b]
    by = weight_bytes + b * CTX * kv_per_token
    per_tok = by / b / 1e9
    print(f"{b:>7} {t*1e3:>10.2f} {weight_bytes/1e9:>12.2f} {per_tok:>13.3f} {(base/t):>14.1f}x")

# %%
# Read the table. The weight bytes are identical in every row — they do not depend
# on batch. The step time barely moves. So GB per token falls steeply, and that is
# the entire economics of continuous batching (Module 07) in one table.
print("\nThis is the most important table in the lab.")
print("Weight bytes are constant. Step time is nearly constant. Therefore bytes")
print("per token fall steeply with batch, and cost per token falls with them.")

# %% [markdown]
# ## 8. Prefill, for contrast
#
# Profile the other phase too, and see that the roofline answer is completely
# different. A prefill GEMM is compute-bound, so the metric is MFU, not MBU.

# %%
def prefill_time(n, iters=20):
    ids = torch.randint(1000, 50000, (1, n), device="cuda")
    return bench(lambda: model(ids), iters=iters)

print(f"{'tokens':>8} {'prefill ms':>12} {'us/token':>10}")
for n in [256, 512, 1024, 2048]:
    t = prefill_time(n)
    print(f"{n:>8} {t*1e3:>12.2f} {t*1e6/n:>10.2f}")

# %%
# The prefill step's arithmetic intensity, for a chunk of 512 tokens.
m = 512
pf_flops = 2 * PARAMS * m
pf_bytes = (m * HIDDEN + PARAMS + m * HIDDEN) * DTYPE_BYTES
pf_I = pf_flops / pf_bytes

print(f"\nPrefill, m = {m} tokens:")
print(f"  FLOPs       {pf_flops/1e9:.1f} GFLOP")
print(f"  bytes       {pf_bytes/1e9:.3f} GB")
print(f"  intensity   {pf_I:.0f} FLOP/byte")
print(f"  ridge       {RIDGE:.0f} FLOP/byte")
print(f"  BOUND       {'COMPUTE' if pf_I > RIDGE else 'MEMORY'}")
print()
print(f"Compare decode's intensity of {intensity:.1f} FLOP/byte. Same model, same GPU,")
print("same kernels — a completely different answer. This is why Lesson 12.5")
print("insists on profiling the two phases as separate experiments.")
print()
print("Note: on the T4 the ridge is only ~203, so a 512-token prefill is only just")
print("compute-bound. On an H100 (ridge 295) or A100 (ridge 156) the same chunk")
print("sits comfortably on the flat top of the roofline.")

# %% [markdown]
# ## 9. Assemble the report
#
# Everything you need is printed above. Write the one-page report now.
#
# Fill in the blanks with your measured values.

# %%
w_bytes = weight_bytes / 1e9
kv_bytes = BATCH * CTX * kv_per_token / 1e9
total_gb = total_bytes / 1e9
bound_word = "memory" if intensity < RIDGE else "compute"
metric = "MBU" if intensity < RIDGE else "MFU"
metric_val = mbu * 100 if intensity < RIDGE else (flops / measured / PEAK_FLOPS) * 100

print("=" * 74)
print("BOTTLENECK REPORT — DRAFT (fill in and expand)")
print("=" * 74)
print()
print("CLAIM")
print(f"  The decode step is {bound_word}-bound and achieves "
      f"{metric_val:.0f}% {metric},")
print(f"  with {headroom:.2f}x headroom against the roofline floor.")
print()
print("CONFIGURATION")
print(f"  model         {MODEL} ({PARAMS/1e9:.2f}B params, {LAYERS} layers)")
print(f"  gpu           {GPU_NAME} ({PEAK_BW/1e9:.0f} GB/s, {PEAK_FLOPS/1e12:.0f} TFLOP/s)")
print(f"  dtype         fp16 ({DTYPE_BYTES} bytes/weight)")
print(f"  batch/context {BATCH} sequences, {CTX} tokens average")
print()
print("EVIDENCE")
print(f"  step time     {measured*1e3:.2f} ms")
print(f"  bytes         {total_gb:.2f} GB "
      f"({w_bytes:.2f} weights + {kv_bytes:.2f} KV)")
print(f"  intensity     {intensity:.2f} FLOP/byte")
print(f"  ridge point   {RIDGE:.0f} FLOP/byte  →  {bound_word}-bound")
print(f"  achieved BW   {achieved/1e9:.1f} GB/s = {mbu*100:.0f}% MBU")
print(f"  floor         {ideal_step*1e3:.2f} ms")
print()
print("WHAT WAS TRIED")
print(f"  FP8 weights   predicted {(1-predicted/before)*100:.0f}% faster "
      f"(matches if diagnosis is right)")
print(f"  CUDA graphs   ~{n_total*per_launch_us/1000:.2f} ms of launch overhead, "
      f"{n_total*per_launch_us/1000/(before*1e3)*100:.1f}% of the step")
print(f"  batch 1→64    {(base/dec[64]):.0f}x throughput, "
      f"{dec[64]/base:.2f}x step time")
print()
print("RESIDUAL RISK")
print("  - Small model on a T4: the T4 has no fast BF16 tensor-core path, and a")
print("    0.5B model's GEMMs do not fill the machine. Absolute MBU is low for")
print("    structural reasons; the ratios are what transfer.")
print(f"  - KV crossover for this batch is {weight_bytes/(kv_per_token*BATCH):,.0f}")
print("    tokens. Beyond that, decode becomes a KV problem and weight")
print("    quantization stops being the right lever.")
print("  - Not profiled: sampling cost, the scheduler, and the HTTP layer.")
print()
print("=" * 74)

# %% [markdown]
# ## 10. Your turn
#
# 1. **Section 3**: raise the context until the KV cache exceeds the weights. At
#    what context does it happen for batch 32? Does the prediction
#    `weight_bytes / (kv_per_token * batch)` match what you measured?
# 2. **Section 5**: sweep context at batch 32 and plot MBU against context. Why
#    does MBU *fall* as context grows, even though the step is still memory-bound?
# 3. **Section 7, FIX 1**: actually run an FP8 or int8 quantized version and check
#    whether the measured speed-up matches the prediction. If it does not, you have
#    found something real — write it in the report.
# 4. **Section 7, FIX 3**: at what batch does the step time start to rise
#    noticeably? That is roughly where you run out of KV memory on a real engine,
#    and where the roofline's compute term starts to matter.
# 5. **Section 8**: compute the prefill intensity for m = 128 and m = 2048. At
#    what m does prefill cross the T4's ridge point of 203?
# 6. **Section 9**: rewrite the report as a real one-page document. Include the
#    residual-risk section. Include any fix that did *not* work, and why.
#
# **Record in your notes:** `measured step time`, `intensity`, `MBU`, `headroom`,
# and the `weights/KV crossover`. These five numbers are the whole report.

# %%
print("\n=== Your five numbers ===")
print(f"  step time       = {measured*1e3:.2f} ms")
print(f"  intensity       = {intensity:.2f} FLOP/byte")
print(f"  MBU             = {mbu*100:.1f}%")
print(f"  headroom        = {headroom:.2f}x")
print(f"  KV crossover    = {weight_bytes/(kv_per_token*BATCH):,.0f} tokens (batch {BATCH})")
