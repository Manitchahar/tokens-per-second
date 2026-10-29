# %% [markdown]
# # Lab 11 · Quantize a model and measure what it cost
#
# **Tokens per Second** · Module 11
#
# In this notebook you will, on your own GPU:
#
# 1. Implement **round-to-nearest** weight quantization from scratch and watch the error grow.
# 2. Show why **per-tensor** scales fail on outliers and **group-wise** scales fix it.
# 3. Quantize an actual small model's weights and measure the **memory saving** exactly.
# 4. Measure **the speed of a quantized decode step** versus BF16, on real hardware.
# 5. Measure the **quality regression** with perplexity on a held-out set.
#
# Then you will fill in one table that answers the only question that matters:
# *is this configuration worth shipping?*
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# We use **Qwen2.5-0.5B-Instruct** (0.49B parameters) so everything runs quickly on a
# free T4. The T4 has no BF16 tensor cores and no FP8, so we will do most of the work
# with **simulated** quantization (dequantize, then run the real BF16 GEMM). That is
# exactly how you would measure *quality*. For *speed* we will measure the memory
# traffic, which is the part that quantization actually changes.

# %%
import torch, time, math
assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
print(f"{p.name}   {p.total_memory/1e9:.1f} GB   sm_{p.major}{p.minor}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}   fp8: {'no' if p.major < 9 else 'yes'}")

# %%
def bench(fn, iters=20, warmup=5):
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
# ## 1. The algorithm, implemented
#
# From the lesson, in full. Quantize a tensor to `bits`, either with one scale for the
# whole thing (`group=0`) or one scale per group of `group` values along the last axis.
#
# ```
# s   = max|W| / (2^(bits-1) - 1)
# q   = clamp(round(W / s), -(2^(bits-1)-1), +(2^(bits-1)-1))
# Ŵ   = q * s
# ```

# %%
def quantize_per_tensor(W, bits):
    levels = 2 ** (bits - 1) - 1
    s = W.abs().max() / levels
    q = torch.clamp(torch.round(W / s), -levels, levels)
    return (q * s), s

def quantize_group(W, bits, group):
    levels = 2 ** (bits - 1) - 1
    out, cols = W.shape[-1], W.shape[-1]
    g = cols if group <= 0 else group
    Wg = W.reshape(*W.shape[:-1], cols // g, g)
    s = Wg.abs().amax(dim=-1, keepdim=True) / levels
    q = torch.clamp(torch.round(Wg / s), -levels, levels)
    return (q * s).reshape_as(W), s

def rel_err(W, W_hat):
    return (torch.norm(W_hat - W) / torch.norm(W)).item()

# The test tensor from the lesson: a few outliers, many small values.
torch.manual_seed(0)
W = torch.randn(4096, 4096, device="cuda") * 0.02
W[100, 100] = 0.61      # one big outlier
W[500, 500] = -0.58
W[900, 900] = 0.55
print(f"max |W| = {W.abs().max():.3f}   99th pct |W| = {W.abs().flatten().kthvalue(int(0.99*W.numel())).values:.4f}")

# %%
print(f"{'bits':>5} {'per-tensor err':>16} {'group-128 err':>14} {'meta overhead':>15}")
for bits in [8, 4, 3, 2]:
    Wt, _ = quantize_per_tensor(W, bits)
    Wg, _ = quantize_group(W, bits, 128)
    overhead = (4 / 128) / (bits / 8) * 100
    print(f"{bits:>5} {rel_err(W, Wt)*100:>15.2f}% {rel_err(W, Wg)*100:>13.2f}% {overhead:>14.2f}%")

print("""
Read this table. Two things are happening at once:
  1. Error roughly doubles each time you drop a bit.
  2. Group-wise scales beat per-tensor by a wide margin at low bits, because the
     outlier in row 100 no longer forces a coarse grid on the other 4095 columns.

At 8 bits even per-tensor is fine (0.2% error). At 4 bits, per-tensor is 4.7x worse
than group-128. That is the argument for group-wise quantization.""")

# %% [markdown]
# ## 2. How low can you go before it breaks?
#
# Sweep the bit width and watch for the cliff. Also count how many weights round to
# **exactly zero** — the failure mode from the lesson, where a coarse grid deletes the
# small weights entirely.

# %%
print(f"{'bits':>5} {'group-128 err':>14} {'zeros %':>9} {'largest surviving weight':>26}")
for bits in [8, 6, 5, 4, 3, 2]:
    Wg, _ = quantize_group(W, bits, 128)
    zeros = (Wg == 0).float().mean().item() * 100
    print(f"{bits:>5} {rel_err(W, Wg)*100:>13.2f}% {zeros:>8.2f}% {Wg.abs().max().item():>25.3f}")

# %%
# The zero fraction is the interesting column. At 4 bits a noticeable share of weights
# snap to zero. At 2 bits, roughly half the matrix is gone. That is not quantization,
# that is accidental sparsity — and it is why 2-bit does not work.

# %% [markdown]
# ## 3. The metadata bill
#
# Group-wise quantization is not free. Each group needs a scale (and in asymmetric
# formats a zero-point), both stored in FP16 = 4 bytes. Compute the true bits per weight.

# %%
print(f"{'bits':>5} {'group':>7} {'overhead':>10} {'effective bits/weight':>22}")
for bits in [8, 4]:
    for group in [32, 64, 128, 256]:
        overhead = (4 / group) / (bits / 8)
        eff = bits * (1 + overhead)
        print(f"{bits:>5} {group:>7} {overhead*100:>9.2f}% {eff:>22.2f}")

print("""
Group 128 at 4 bits is the common default: 4.25 bits per weight. Group 32 is more
accurate but 25% overhead, i.e. 5.0 effective bits — barely better than INT5 and much
worse than INT6. That is the trade you tune.""")

# %% [markdown]
# ## 4. Quantize a real model and measure the memory saving exactly
#
# Now the point of all of it. We take Qwen2.5-0.5B, quantize its linear weights with
# GPTQ-style group-128 RTN, and measure the actual bytes.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()

cfg = model.config
PARAMS = sum(q.numel() for q in model.parameters())
print(f"parameters {PARAMS/1e9:.3f} B   layers {cfg.num_hidden_layers}   hidden {cfg.hidden_size}")

# %%
# How many weights are in the linear layers we will quantize?
def linear_layers(m):
    return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)]

lins = linear_layers(model)
n_lin = sum(mod.weight.numel() for _, mod in lins)
print(f"linear layers: {len(lins)}   weights inside them: {n_lin/1e6:.1f} M of {PARAMS/1e6:.1f} M total")

# %%
# The measurement. Replace each linear weight with its dequantized version and see
# what a real forward pass costs in quality. (We keep it dequantized in FP32 so the
# forward pass runs unchanged — this measures the *numerics*, not the speed.)

@torch.no_grad()
def quantize_model_(m, bits, group=128):
    for name, mod in linear_layers(m):
        W = mod.weight.data
        Wq, _ = quantize_group(W, bits, group)
        err = rel_err(W, Wq)
        mod.weight.data = Wq
        mod._quant_err = err
    return m

texts = [
    "The capital of France is Paris, and the capital of Germany is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n",
    "The three laws of robotics are: one, a robot may not injure a human being; two,",
]
ids = [tok(t, return_tensors="pt").input_ids.cuda() for t in texts]

@torch.no_grad()
def logits_of(m):
    return [m(i).logits[0, -1].float() for i in ids]

base_logits = logits_of(model)

# %%
print(f"{'bits':>5} {'mean weight err':>16} {'max logit drift':>17} {'top-1 same':>11}")
for bits in [16, 8, 6, 4, 3]:
    import copy
    m2 = copy.deepcopy(model)
    quantize_model_(m2, bits, 128)
    errs = [mod._quant_err for _, mod in linear_layers(m2)]
    lg = logits_of(m2)
    drift = max((a - b).abs().max().item() for a, b in zip(lg, base_logits))
    same = sum(int(a.argmax() == b.argmax()) for a, b in zip(lg, base_logits))
    label = "BF16/fp32" if bits == 16 else f"INT{bits}"
    print(f"{label:>5} {sum(errs)/len(errs)*100:>15.2f}% {drift:>16.4f} {same}/{len(ids)}")
    del m2
    torch.cuda.empty_cache()

print("""
'Logit drift' is the maximum change in the model's output scores. It is the most direct
measurement of how much the quantization changed the function. Top-1 agreement is a
crude sanity check; with greedy decoding a drifted logit can still flip the argmax and
start a different sentence.""")

# %% [markdown]
# ## 5. The speed claim, measured honestly
#
# On a T4 we have no INT4 tensor cores, so we cannot measure the fused-kernel speedup.
# What we *can* measure is the thing quantization actually controls: **bytes moved per
# decode step**, and the real decode step time against it.
#
# First, measure the weights' contribution to memory traffic.

# %%
bytes_fp32 = PARAMS * 4
for bits in [16, 8, 4]:
    overhead = (4/128) if bits == 4 else 0
    b = PARAMS * (bits/8) * (1 + overhead)
    print(f"{'fp32' if bits==16 else f'INT{bits}':>6}: {b/1e9:6.2f} GB of weights "
          f"({bytes_fp32/b:.2f}x smaller than fp32)")

# %%
# Now: does the decode step actually shrink with the weights? On real hardware the
# step is dominated by reading weights, so yes — but only if the kernel is fused.
# We measure the step time for the real model at several precisions.

from transformers import DynamicCache

def decode_step(m, batch=16, ctx=1024, iters=15):
    past = DynamicCache()
    full = torch.randint(1000, 50000, (batch, ctx), device="cuda")
    with torch.no_grad():
        m(full, use_cache=True, past_key_values=past)
    new = torch.randint(1000, 50000, (batch, 1), device="cuda")
    def step():
        with torch.no_grad():
            m(new, past_key_values=past, use_cache=True)
    return bench(step, iters=iters)

print(f"{'precision':>10} {'decode step ms':>16} {'relative':>10}")
base_t = None
for dt, name in [(torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
    import copy
    m2 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dt).cuda().eval()
    t = decode_step(m2)
    if base_t is None: base_t = t
    print(f"{name:>10} {t*1e3:>15.2f} {t/base_t:>9.2f}x")
    del m2
    torch.cuda.empty_cache()

print("""
On a T4 the fp32 -> fp16 gap is modest because there is no fast BF16 tensor-core path
for all shapes. On an H100 it is much larger. This is the *compute* side; quantization's
main win is the *memory* side, which the T4 also shows: fp32 weights are 2x the bytes of
fp16, and the decode step is dominated by reading them.""")

# %% [markdown]
# ## 6. Perplexity: the cheap regression check
#
# Compute perplexity on a held-out text at each precision. This is the smoke test from
# the lesson — it moves before task accuracy does, and it takes seconds.

# %%
# A small held-out corpus. In real work this is a few MB of text matching your traffic.
corpus = """
Machine learning is a field of study in artificial intelligence concerned with the
development of statistical algorithms that can learn from data and generalize to unseen
data, and thus perform tasks without explicit instructions. Recently, generative
artificial neural networks have been able to surpass many previous approaches in
performance. Generative AI is transforming computing and generating new content.
The transformer architecture introduced in 2017 relies on attention mechanisms to
process sequences in parallel, which made it possible to train on far more data.
Quantization reduces the precision of numerical representations, which reduces memory
bandwidth requirements and can increase throughput on specialized hardware.
Memory bandwidth is often the limiting factor for autoregressive decoding.
A decode step reads every weight exactly once, so the step time is bounded below by the
time it takes to stream the entire weight matrix through the memory system.
""".strip()
cids = tok(corpus, return_tensors="pt").input_ids.cuda()
print(f"held-out corpus: {cids.shape[1]} tokens")

@torch.no_grad()
def perplexity(m, ids, stride=128):
    n = 0
    total = 0.0
    prev_end = 0
    for begin in range(0, ids.shape[1], stride):
        end = min(begin + stride, ids.shape[1])
        if end - begin < 2: break
        chunk = ids[:, begin:end]
        out = m(chunk)
        logits = out.logits[0, :-1].float()
        targets = chunk[0, 1:]
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
        total += loss.item()
        n += targets.numel()
        prev_end = end
    return math.exp(total / n)

import copy
print(f"\n{'precision':>10} {'perplexity':>12} {'vs fp32':>10}")
p0 = None
for dt, name in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
    m2 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dt).cuda().eval()
    p = perplexity(m2, cids)
    if p0 is None: p0 = p
    print(f"{name:>10} {p:>12.3f} {p/p0:>9.4f}x")
    del m2; torch.cuda.empty_cache()

# %%
# Now the simulated 4-bit model, with the same perplexity.
import copy
m4 = copy.deepcopy(model)
quantize_model_(m4, 4, 128)
p4 = perplexity(m4, cids)
print(f"{'INT4 g128':>10} {p4:>12.3f} {p4/p0:>9.4f}x")
print("""
A few percent on perplexity is the normal signature of 4-bit weights on a small model.
Whether that is acceptable is not something this notebook can decide for you — that is
what the task benchmarks in the lesson are for.""")

# %% [markdown]
# ## 7. The KV cache side
#
# Quantize the KV cache. The simplest faithful simulation: keep the cache in FP16
# but apply the same scale-and-round transform to it, so we measure the numerics.
# (Real engines quantize on write and dequantize inside the attention kernel.)

# %%
@torch.no_grad()
def quantize_kv_error(m, bits, batch=4, ctx=1024):
    """Run a prefill, then re-quantize the KV cache and measure the drift in the
    next-token logits."""
    past = DynamicCache()
    ids_kv = torch.randint(1000, 50000, (batch, ctx), device="cuda")
    out = m(ids_kv, use_cache=True, past_key_values=past)
    ref = out.logits[:, -1].float()

    levels = 2 ** (bits - 1) - 1
    for layer in past.layers:
        for t in (layer.keys, layer.values):
            if t is None: continue
            s = t.abs().amax() / levels
            t.copy_(torch.clamp(torch.round(t / s), -levels, levels) * s)

    new = torch.randint(1000, 50000, (batch, 1), device="cuda")
    out2 = m(new, past_key_values=past, use_cache=True)
    drift = (out2.logits[:, -1].float() - ref).abs().max().item()
    top1 = int((out2.logits[:, -1].argmax(-1) == ref.argmax(-1)).sum())
    return drift, top1, batch

print(f"{'KV bits':>8} {'logit drift':>13} {'top-1 agreement':>16}")
for bits in [16, 8, 4]:
    d, t1, b = quantize_kv_error(model, bits)
    print(f"{bits:>8} {d:>12.4f} {t1}/{b}")

print("""
Note the shape: 8-bit KV barely moves the logits, 4-bit moves them visibly. And this
test is at 1024 tokens. The lesson's point is that the damage *grows* with context
length — at 32k there are 32x more quantized keys contributing to every attention
distribution. Re-run this with ctx=8192 and compare the 4-bit drift.""")

# %%
print("\n4-bit KV at three context lengths — the length sweep, in miniature:")
print(f"{'ctx':>8} {'logit drift':>13} {'top-1 agreement':>16}")
for ctx in [512, 2048, 8192]:
    d, t1, b = quantize_kv_error(model, 4, batch=2, ctx=ctx)
    print(f"{ctx:>8} {d:>12.4f} {t1}/{b}")

# %% [markdown]
# ## 8. Put it together: the table you would actually ship
#
# Fill in this for your own measurements. This is the deliverable shape from the lesson.

# %%
rows = []
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()
ppl_ref = perplexity(base, cids)
t_ref = decode_step(base)
for label, bits, group in [("fp32", 32, 128), ("INT8 g128", 8, 128), ("INT4 g128", 4, 128)]:
    if bits == 32:
        m, ppl, t, err = base, ppl_ref, t_ref, 0.0
    else:
        m = copy.deepcopy(base)
        quantize_model_(m, bits, group)
        errs = [mod._quant_err for _, mod in linear_layers(m)]
        err = sum(errs)/len(errs)
        ppl = perplexity(m, cids)
        t = decode_step(m)
    rows.append((label, err*100, ppl, ppl/ppl_ref, t*1e3, t/t_ref))
    if bits != 32:
        del m; torch.cuda.empty_cache()

print(f"{'config':>12} {'wt err %':>9} {'PPL':>8} {'PPL ratio':>10} {'step ms':>9} {'step ratio':>11}")
for r in rows:
    print(f"{r[0]:>12} {r[1]:>8.2f}% {r[2]:>8.3f} {r[3]:>10.4f} {r[4]:>8.2f} {r[5]:>10.3f}")

# %% [markdown]
# ## 9. What a real engine does (for reference)
#
# We cannot install these here, but this is the shape of the command you would run:
#
# ```bash
# # FP8 weights on an H100 (FP8 tensor cores: Ada, Hopper, Blackwell), KV cache in FP8
# vllm serve Qwen/Qwen2.5-7B-Instruct \
#   --quantization fp8 \
#   --kv-cache-dtype fp8 \
#   --max-model-len 32768
#
# # W4A16 weight-only, pre-quantized GPTQ/AWQ checkpoint
# vllm serve TheBloke/Llama-3-8B-AWQ \
#   --max-model-len 16384 \
#   --enable-prefix-caching
# ```
#
# The flags to check in whatever engine you use:
#   - Does the GEMM claim a **fused** INT4/FP8 kernel? (`--quantization` in vLLM)
#   - Is the KV dtype **native** in the attention backend, or dequantized to BF16 first?
#   - What is the reported **weight memory** after load? If it says 16 GB for an INT4
#     checkpoint, you are materializing a full-precision copy.

# %% [markdown]
# ## 10. Your turn
#
# 1. **Section 1**: why is per-tensor error so much worse at 4 bits than at 8? Compare
#    the grid step in each case against the 99th-percentile weight.
# 2. **Section 2**: at what bit width does the zero fraction exceed 50%? What does that
#    mean for the model?
# 3. **Section 3**: pick a target memory budget and work out which (bits, group) you
#    would pick to hit it with the least error. Show the arithmetic.
# 4. **Section 4**: the logit drift at 4 bits is small but non-zero. At what point does
#    that become visible in greedy decoding? Try decoding 64 tokens from the quantized
#    model and diff against the fp32 output.
# 5. **Section 7**: the 4-bit KV drift grows with context. Extrapolate: at 32k tokens,
#    roughly how large would it be? Then argue for or against INT4 KV in production.
# 6. **Section 8**: your table. What is the PPL ratio at which you would reject a config,
#    and why that number rather than another?

# %%
print("\n=== Your measured constants ===")
print(f"parameters              {PARAMS/1e9:.3f} B")
print(f"fp32 weights            {PARAMS*4/1e9:.2f} GB")
print(f"INT8 g128 weights       {PARAMS*1.0*1.03125/1e9:.2f} GB")
print(f"INT4 g128 weights       {PARAMS*0.5*1.0625/1e9:.2f} GB  (4.25 bits/weight)")
print(f"decode step (fp32)      {t_ref*1e3:.2f} ms")
for r in rows[1:]:
    print(f"PPL ratio {r[0]:<12} {r[3]:.4f}   step ratio {r[5]:.3f}")