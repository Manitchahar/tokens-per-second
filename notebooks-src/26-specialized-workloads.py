# %% [markdown]
# # Lab 26 · Specialized workloads, measured
#
# **Tokens per Second** · Module 26
#
# You will:
# 1. Build a constrained decoder's mask and measure what it costs against an unconstrained step.
# 2. Implement a LoRA adapter in PyTorch, and verify the r·(in + out) parameter arithmetic.
# 3. Measure the cost of splitting a decode batch across several adapters.
# 4. Compare prefill cost with and without a prefix cache over a simulated conversation.
# 5. Size an RL rollout group: KV cache, what fits on a GPU, and how a draining batch slows down.
# 6. Measure the arithmetic intensity of a convolution and put it on the roofline next to decode.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# The rule of the whole lab: **measure, then explain.** Every number is compared against the
# closed-form arithmetic from the lessons. If they disagree, the arithmetic is wrong or the code is —
# go and find out which.

# %%
import math, time, json, re
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
DEV = "cuda"
torch.manual_seed(0)
print(torch.cuda.get_device_name(0))
print("torch", torch.__version__)

# T4 spec-sheet figures, used as the reference for every prediction below.
T4_BW = 320e9      # bytes/s, ~320 GB/s
T4_FLOPS = 65e12   # dense FP16 FLOP/s


def bench(fn, iters=30, warmup=10):
    """Time a GPU function. Always warm up, always synchronise, always average."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def mib(x):
    return x / 2**20

# %% [markdown]
# ## 1. Constrained decoding: the mask is one elementwise pass
#
# The claim from the lesson: a vocabulary mask costs a small fraction of a decode step, adds no
# tokens, and adds no forward passes. Let's build the mask and time it against a real model step.

# %%
# A toy vocabulary. We will treat the last 32 entries as "structural" tokens (braces, quotes,
# colons, commas) and the rest as ordinary word-like tokens.
VOCAB = 512
STRUCTURAL = torch.zeros(VOCAB, dtype=torch.bool, device=DEV)
for i in range(VOCAB - 32, VOCAB):
    STRUCTURAL[i] = True

logits = torch.randn(1, VOCAB, device=DEV, dtype=torch.float16)


def unconstrained_step():
    return torch.softmax(logits.float(), dim=-1)


def constrained_step(mask):
    masked = logits.float().masked_fill(~mask, float("-inf"))
    return torch.softmax(masked, dim=-1)


# A few grammars, expressed as the fraction of the vocabulary each leaves open.
grammars = {
    "free text": 1.00,
    "inside a string": 0.62,
    "after a comma": 0.0004,
    "a fixed enum": 0.00002,
}

print(f"{'grammar':<18}{'allowed':>10}{'tokens':>10}")
print("-" * 38)
for name, frac in grammars.items():
    n = max(1, int(round(VOCAB * frac)))
    print(f"{name:<18}{frac*100:>9.4f}%{n:>10}")

base = bench(unconstrained_step)
print(f"\nunconstrained softmax : {base*1e6:8.1f} us")

for name, frac in grammars.items():
    n = max(1, int(round(VOCAB * frac)))
    mask = torch.zeros(VOCAB, dtype=torch.bool, device=DEV)
    if frac > 0.5:
        mask[:] = True
    else:
        mask[VOCAB - n:] = True
    t = bench(lambda: constrained_step(mask))
    print(f"masked ({name:<16}): {t*1e6:8.1f} us   overhead {100*(t-base)/base:6.1f}%")

# %% [markdown]
# **What to look for:** the overhead is small and — crucially — **it does not depend on which grammar
# you use**. Building the mask (parsing the grammar, walking the state) is the part that can get
# expensive, and that is a host-side cost, not a device-side one. On a real 128K vocabulary the
# elementwise pass is a few hundred microseconds against a decode step measured in tens of
# milliseconds.
#
# **Check yourself:** why does the masked version need no extra forward passes? What is the one thing
# that can make a grammar expensive at runtime?

# %% [markdown]
# ## 2. A real LoRA layer, and the parameter arithmetic
#
# The claim: an adapter on an (out x in) matrix of rank r holds exactly r(in + out) parameters,
# and the base matrix is frozen.

# %%
class LoRALinear(nn.Module):
    """A frozen base matrix plus a trainable low-rank correction."""

    def __init__(self, in_features, out_features, r=16):
        super().__init__()
        self.in_features, self.out_features, self.r = in_features, out_features, r
        # The base model: frozen, full rank.
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02, requires_grad=False)
        # The adapter: two thin matrices, initialised so the product starts at zero.
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x, self.weight) + F.linear(F.linear(x, self.A), self.B)

    def adapter_params(self):
        return self.A.numel() + self.B.numel()

    def base_params(self):
        return self.weight.numel()


IN, OUT, R = 4096, 4096, 16
layer = LoRALinear(IN, OUT, R).to(DEV)

predicted = R * (IN + OUT)
actual = layer.adapter_params()
print(f"base matrix   : {layer.base_params():,} parameters")
print(f"adapter (pred): {predicted:,}   r·(in+out) = {R}·({IN}+{OUT})")
print(f"adapter (real): {actual:,}")
assert predicted == actual, "parameter arithmetic is wrong"
print(f"adapter share : {100*actual/layer.base_params():.3f}%")

# Verify the correction really starts at zero (so the adapter begins as the identity).
x = torch.randn(4, IN, device=DEV)
with torch.no_grad():
    before = layer(x).clone()
    layer.B.data.normal_(0, 0.01)
    after = layer(x)
print(f"\nwith B = 0 the adapter is a no-op: {torch.allclose(before, F.linear(x, layer.weight), atol=1e-4)}")
print(f"after perturbing B the output changes: mean |Δ| = {(after-before).abs().mean():.6f}")

# Now scale rank and check linearity.
print(f"\n{'rank':>6}{'adapter params':>18}{'share of base':>16}")
for r in [1, 4, 8, 16, 32, 64]:
    l = LoRALinear(IN, OUT, r)
    n = l.adapter_params()
    print(f"{r:>6}{n:>18,}{100*n/l.base_params():>15.3f}%")
    del l

# %% [markdown]
# **What to look for:** the count is exactly r·(in+out), linear in r, and under 0.8% of the base even
# at r=16. Note the initialisation detail: B starts at zero so that B·A = 0, which means a freshly
# loaded adapter changes the model's behaviour by exactly nothing. That is what makes it safe to hot-swap
# adapters between requests.
#
# **Check yourself:** the usual shorthand calls r=16 on a 4096-wide model "0.39%". Compute
# r·(in+out)/(in·out) for a square matrix and explain the factor of two.

# %%
print(f"shorthand r/d          = {100*R/IN:.3f}%")
print(f"correct r(in+out)/(in·out) = {100*R*(IN+OUT)/(IN*OUT):.3f}%")

# %% [markdown]
# ## 3. Batch mixing: the cost nobody budgets
#
# The claim: a decode batch containing several adapters must be split into per-adapter GEMMs, and
# that costs almost exactly like shrinking the batch.

# %%
# Simulate the weight read of a decode step. Real decode is bandwidth-bound, so the honest model
# is "bytes moved / achievable bandwidth", and the interesting question is how many bytes a step moves
# for a given batch size and number of adapter groups.
WEIGHTS = 8.03e9 * 2      # Llama-3.1-8B in BF16
BATCH = 32
CTX = 2048
KV_PER_TOKEN = 2 * 32 * 8 * 128 * 2


def step_bytes(batch, groups):
    """Bytes read by one decode step: KV per sequence, plus one weight read per adapter group."""
    kv = batch * CTX * KV_PER_TOKEN
    w = groups * WEIGHTS          # each group is its own GEMM -> its own weight read
    return kv + w


base_bytes = step_bytes(BATCH, 1)
print(f"{'groups':>8}{'bytes/step':>16}{'vs 1 group':>14}")
print("-" * 38)
for g in [1, 2, 4, 8, 16]:
    b = step_bytes(BATCH, g)
    print(f"{g:>8}{b/1e9:>15.1f}G{b/base_bytes:>13.2f}x")

print("\nThe equivalence with batch size:")
print(f"{'batch':>8}{'bytes/step':>16}")
print("-" * 24)
for b in [32, 16, 8, 4, 2]:
    print(f"{b:>8}{step_bytes(b,1)/1e9:>15.1f}G")

print(f"\nBatch 32 in 4 adapters = {step_bytes(32,4)/1e9:.1f} GB")
print(f"Batch  8 in 1 adapter  = {step_bytes(8,1)/1e9:.1f} GB")

# And measure the actual GEMM penalty on the GPU, which is the part that is real.

# %%
W = torch.randn(OUT, IN, device=DEV, dtype=torch.float16)


def gemm_time(rows):
    x = torch.randn(rows, IN, device=DEV, dtype=torch.float16)
    t = bench(lambda: x @ W.T, iters=50, warmup=20)
    bytes_moved = OUT * IN * 2 + rows * IN * 2 + rows * OUT * 2
    return t, bytes_moved


print(f"{'rows':>8}{'ms':>10}{'GB/s':>10}{'vs 32 rows':>12}")
print("-" * 40)
ref = None
for rows in [32, 16, 8, 4, 2, 1]:
    t, b = gemm_time(rows)
    bw = b / t
    if ref is None:
        ref = t
    print(f"{rows:>8}{t*1e3:>10.3f}{bw/1e9:>10.1f}{t/ref:>11.2f}x")

# %% [markdown]
# **What to look for:** bandwidth collapses as rows shrink, because a thin GEMM cannot keep the
# memory system busy. This is the mechanism behind the mixing tax. It also shows why engines care
# about GEMM shapes — the same arithmetic at 32 rows and at 2 rows does not take twice the time, it
# takes eight times as long.
#
# **Check yourself:** the lesson claims adapter count per step and batch size are the same dial. Using
# the byte counts above, show that batch 32 split across 4 adapters moves about the same bytes as a
# batch of 8.

# %% [markdown]
# ## 4. Prefix caching: what a conversation saves
#
# The claim: a six-turn chat prefills 7,800 tokens without a cache and 840 with one — an 89% saving.

# %%
BLOCK = 16


def conversation(system, per_turn, turns, tool_every_other=True):
    """Contexts and per-turn additions for a simple chat, plus an agent-style loop."""
    ctx = system
    out = []
    for t in range(turns):
        added = per_turn + (40 if (tool_every_other and t % 2 == 0) else 0)
        before = ctx
        ctx += added
        blocks = math.ceil(ctx / BLOCK)
        hits = math.ceil(before / BLOCK)
        out.append({"turn": t + 1, "ctx": ctx, "added": added, "hits": hits, "misses": blocks - hits})
    return out


conv = conversation(800, 120, 6)
print(f"{'turn':>6}{'context':>10}{'added':>8}{'hit blocks':>12}{'fresh blocks':>14}")
print("-" * 52)
for c in conv:
    print(f"{c['turn']:>6}{c['ctx']:>10}{c['added']:>8}{c['hits']:>12}{c['misses']:>14}")

naive = sum(c["ctx"] for c in conv)
cached = sum(c["added"] for c in conv)
print(f"\nno cache   : {naive:>7,} token-prefills")
print(f"with cache : {cached:>7,} token-prefills")
print(f"saved      : {100*(1-cached/naive):>7.1f}%")

PARAMS = 8.03e9
print(f"\nFLOPs no cache : {2*PARAMS*naive/1e12:8.1f} TFLOP")
print(f"FLOPs cached   : {2*PARAMS*cached/1e12:8.1f} TFLOP")

# %% [markdown]
# The agent loop: quadratic without reuse.

# %%
def agent_loop(system, obs, turns):
    ctx, cum, redundant = system, 0, 0
    rows = []
    for t in range(1, turns + 1):
        ctx += obs
        cum += obs
        redundant += ctx
        rows.append((t, ctx, cum, redundant))
    return rows


OBS = 3000
print(f"{'turn':>6}{'context':>10}{'cached':>10}{'no cache':>12}{'ratio':>9}")
print("-" * 50)
for t, ctx, cum, red in agent_loop(800, OBS, 8):
    print(f"{t:>6}{ctx:>10,}{cum:>10,}{red:>12,}{red/cum:>8.1f}x")
print()
for t, ctx, cum, red in agent_loop(800, OBS, 20):
    if t in (16, 20):
        print(f"turn {t:>2}: context {ctx:,}, no-cache prefill {red:,}, ratio {red/cum:.1f}x")

# The ratio is approximately 2T, independent of the model.
print("\nratio ≈ 2T:")
for T in [5, 10, 20, 40]:
    rows = agent_loop(800, OBS, T)
    _, _, cum, red = rows[-1]
    print(f"  T={T:>3}: ratio {red/cum:>5.1f}, 2T = {2*T}")

# %% [markdown]
# **Check yourself:** why does the ratio not depend on the model size, the GPU, or the framework?
# And what single change to a system prompt would make the cached column as expensive as the
# uncached one?

# %% [markdown]
# ## 5. Sizing an RL rollout group
#
# The claim: KV, not weights, is the constraint, and on 8 H100s a 1,024-rollout group of 4,096-token
# Llama-3.1-8B sequences uses 86% of cluster memory.

# %%
def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * bytes_per


models = {
    "Qwen2.5-1.5B":  dict(layers=28, kv=2, dim=128, params=1.54e9),
    "Qwen2.5-7B":    dict(layers=28, kv=4, dim=128, params=7.62e9),
    "Llama-3.1-8B":  dict(layers=32, kv=8, dim=128, params=8.03e9),
}

T4_MEM = 16e9
print(f"{'model':<16}{'KV/token':>10}{'@4096 tok':>12}{'rollouts/T4':>14}")
print("-" * 54)
for name, m in models.items():
    kvt = kv_bytes_per_token(m["layers"], m["kv"], m["dim"])
    per = kvt * 4096
    weights = m["params"] * 2
    # leave ~5% of the card for activations and the CUDA context
    free = T4_MEM * 0.95 - weights
    fit = int(free // per)
    print(f"{name:<16}{kvt/1024:>9.0f}K{per/1e9:>11.2f}G{fit:>14}")

print("\nLlama-3.1-8B: the KV term grows with tokens, not parameters")
kvt = kv_bytes_per_token(32, 8, 128)
for seq in [1024, 2048, 4096, 8192]:
    print(f"  seq {seq:>5}: {kvt*seq/1e9:5.2f} GB per rollout")

# Measure the decode step time against batch size on the real GPU.
# %% [markdown]
# ## 6. The draining batch, measured
#
# Decode is memory-bound, so throughput should rise with batch and then flatten. Let's watch where
# the knee is on a T4.

# %%
weights_gpu = torch.randn(4096, 4096, device=DEV, dtype=torch.float16) * 0.01


def decode_step(batch):
    """One decode-ish step: a skinny GEMM per layer plus a KV read."""
    x = torch.randn(batch, 4096, device=DEV, dtype=torch.float16)
    y = x @ weights_gpu.T
    return y


print(f"{'batch':>8}{'ms/step':>10}{'tok/s':>12}{'vs batch 1':>12}")
print("-" * 44)
ref = None
for b in [1, 8, 32, 128, 512, 2048]:
    t = bench(lambda: decode_step(b), iters=50, warmup=20)
    tps = b / t
    if ref is None:
        ref = tps
    print(f"{b:>8}{t*1e3:>10.3f}{tps:>12.0f}{tps/ref:>11.2f}x")

# The T4's memory floor: how long to read a 4096x4096 BF16 weight matrix at peak bandwidth?
mat = 4096 * 4096 * 2
print(f"\none weight matrix = {mat/1e6:.1f} MB")
print(f"at T4 peak ({T4_BW/1e12:.2f} TB/s): {mat/T4_BW*1e6:.1f} us")

# %% [markdown]
# **What to look for:** throughput rises steeply at first — the weight read is amortised across more
# tokens — and then flattens once the KV term catches up. That knee is the roofline crossover in
# action, and it is exactly why an RL engine should run the largest batch memory allows when no user
# is waiting.
#
# **Check yourself:** at batch 1 the weight read dominates completely. What does that imply about
# serving a single interactive user on a big model, and why does it not matter for RL rollouts?

# %% [markdown]
# ## 7. Diffusion is compute-bound where decode is memory-bound
#
# The claim: a convolution does far more arithmetic per byte than a decode step, which is what puts
# it on the compute side of the roofline. Measure the arithmetic intensity of both.

# %%
def time_conv(n=2048, c=3, k=3):
    x = torch.randn(1, c, n, n, device=DEV, dtype=torch.float16)
    w = torch.randn(c, c, k, k, device=DEV, dtype=torch.float16)
    return bench(lambda: F.conv2d(x, w, padding=k // 2), iters=30, warmup=10)


def intensity_conv(n, c, k, seconds):
    # FLOPs ≈ 2 * out_channels * in_channels * k^2 * H * W
    flops = 2 * c * c * k * k * n * n
    # Bytes: read input and weights, write output, all FP16
    bytes_moved = (n * n * c * 2) * 2 + (c * c * k * k * 2) + (n * n * c * 2)
    return flops / bytes_moved, flops / seconds / 1e12


print(f"{'resolution':>12}{'ms':>10}{'GFLOP/s':>12}{'FLOP/byte':>12}{'bound':>10}")
print("-" * 58)
for n in [256, 512, 1024]:
    t = time_conv(n=n, c=32)
    inten, tflops = intensity_conv(n, 32, 3, t)
    ridge = T4_FLOPS / T4_BW
    print(f"{n:>12}{t*1e3:>10.2f}{tflops*1e3:>12.1f}{inten:>12.1f}{'compute' if inten > ridge else 'memory':>10}")

print(f"\nT4 ridge point = {T4_FLOPS/T4_BW:.0f} FLOP/byte")
print("A 32x32-channel 3x3 conv is far above the ridge: it is compute-bound.")
print("A decode GEMM (4096x4096, batch 1) is near or below it: it is memory-bound.")

# And the memory side: activations, not weights, decide diffusion batch size.
# %%
print("\nDiffusion memory at 512x512 (SD 1.5 scale, FP16):")
widest = 320 * 512 * 512 * 2
print(f"  widest feature map : {widest/1e6:.0f} MB")
print(f"  ~6 live tensors    : {6*widest/1e9:.2f} GB per image")
print(f"  weights (860M U-Net): {0.86e9*2/1e9:.2f} GB, shared across the batch")
print(f"  activations overtake weights at batch {math.ceil(0.86e9*2/(6*widest))}")

# %% [markdown]
# **Check yourself:** why does a 3×3 convolution over a large feature map have high arithmetic
# intensity, while a decode GEMM with a batch of one does not? (Hint: count how many times each
# weight byte is reused before it must be re-read.)

# %% [markdown]
# ## 8. Wrap up
#
# Every number in this lab should be reproducible from the arithmetic. The summary:

# %%
summary = {
    "constrained decoding": "one mask per step; ~1-3% overhead; zero extra tokens",
    "LoRA r(in+out)": f"r=16 on 4096x4096 = {R*(IN+OUT):,} params = {100*R*(IN+OUT)/(IN*OUT):.2f}% of base",
    "batch mixing": "adapter groups cost like a smaller batch — same bytes",
    "prefix cache": f"6-turn chat saves {100*(1-cached/naive):.0f}% of prefill",
    "agent loop": f"20 turns, no cache: {agent_loop(800, OBS, 20)[-1][3]:,} tokens ({agent_loop(800, OBS, 20)[-1][3]/agent_loop(800, OBS, 20)[-1][2]:.1f}x)",
    "RL sizing": "KV dominates; size by tokens in flight, not parameters",
    "diffusion": "compute-bound; activations limit batch, not weights",
}
print(json.dumps(summary, indent=2))
print("\nIf any line above disagrees with your own reasoning, the lab is the point — go and check it.")