# %% [markdown]
# # Lab 13.2 · Reductions and a fused softmax
#
# **Tokens per Second** · Module 13 · Writing Kernels with Triton
#
# You will:
# 1. Write block-wide reductions with `tl.sum` / `tl.max` and see the tree.
# 2. Implement the naive three-pass softmax and count its memory traffic.
# 3. Implement the fused one-pass softmax and prove it matches.
# 4. Implement the streaming (online) softmax for rows wider than one block.
# 5. Break `other=-inf` on purpose and watch probabilities stop summing to 1.
# 6. Compare all three against `torch.softmax` and against each other.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.

# %%
import torch, time, math
import triton
import triton.language as tl

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
print("Triton", triton.__version__, "|", torch.cuda.get_device_name(0))

def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# %% [markdown]
# ## 1. A reduction is a tree
#
# Start by confirming the shape of the operation: a block-wide reduction, no
# loop in your code, `log2(BLOCK)` steps in the hardware.

# %%
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + row, s)

M, N = 8, 1024
x = torch.randn(M, N, device="cuda", dtype=torch.float32)
out = torch.empty(M, device="cuda", dtype=torch.float32)
row_sum_kernel[(M,)](x, out, N, BLOCK=1024)
ref = x.sum(dim=1)
print("row sums match:", torch.allclose(out, ref, rtol=1e-4, atol=1e-3))
print("tree depth for BLOCK=1024: log2(1024) =", math.log2(1024), "steps")
print("vs a loop, which is", N, "serialized steps on one lane")

# %% [markdown]
# ## 2. Naive softmax — three passes
#
# Written the obvious way: this is what three separate kernels would do, and it
# is what PyTorch does if you write `row.max()` then `.exp()` then `.sum()`.

# %%
@triton.jit
def softmax_naive(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    l = tl.sum(e, axis=0)
    tl.store(out_ptr + row * n_cols + cols, e / l, mask=mask)

# %% [markdown]
# ## 3. Fused softmax — one pass
#
# Identical arithmetic, but structured so the row is read **once**. In practice
# you write this shape; the "naive" version above is what the *shape of the
# formula* forces on you if you are not careful.

# %%
@triton.jit
def softmax_fused(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # read ONCE. other=-inf keeps masked lanes neutral for tl.max.
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)          # reduction 1
    e = tl.exp(x - m)              # all values <= 0 here, so e <= 1
    l = tl.sum(e, axis=0)          # reduction 2
    tl.store(out_ptr + row * n_cols + cols, e / l, mask=mask)

# %%
def run_softmax(x, BLOCK=None):
    M, N = x.shape
    BLOCK = BLOCK or triton.next_power_of_2(N)
    out = torch.empty_like(x)
    softmax_fused[(M,)](x, out, N, BLOCK=BLOCK)
    return out

for N in [256, 1024, 4096]:
    x = torch.randn(64, N, device="cuda", dtype=torch.float32)
    out = run_softmax(x)
    ref = torch.softmax(x, dim=1)
    print(f"N={N:>5}  max abs err vs torch: {(out - ref).abs().max().item():.3e}  sums to: {out.sum(1).mean().item():.6f}")

# %% [markdown]
# ## 4. Break `other=-inf` on purpose
#
# The single most instructive bug in this module. Change it to 0.0 and see what
# happens — the answer is *not* an error, which is why it is worth finding.

# %%
@triton.jit
def softmax_bad(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)   # <-- 0.0, not -inf
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    l = tl.sum(e, axis=0)
    tl.store(out_ptr + row * n_cols + cols, e / l, mask=mask)

# a vocabulary-sized row so the tail of the block is mostly padding
NV = 40000
BLOCK = 65536
x = torch.randn(4, NV, device="cuda", dtype=torch.float32)
good = torch.empty_like(x); bad = torch.empty_like(x)
softmax_fused[(4,)](x, good, NV, BLOCK=BLOCK)
softmax_bad[(4,)](x, bad, NV, BLOCK=BLOCK)

print("padded lanes:", BLOCK - NV, "of", BLOCK)
print("fused  sums:", good.sum(1)[:4].tolist())
print("other=0 sums:", bad.sum(1)[:4].tolist())
print("fused  correct:", torch.allclose(good, torch.softmax(x, 1), atol=1e-5))
print("other=0 correct:", torch.allclose(bad, torch.softmax(x, 1), atol=1e-5))

# %% [markdown]
# **No error. No NaN. Just wrong.** The 25,536 padded lanes each contribute
# `exp(0 - m)` to the sum, so the real probabilities are all scaled *down* by the
# wrong normalizer. The output is a plausible-looking set of numbers that does
# not sum to 1 — and if you sample from it with top-p you silently change the
# model's behaviour.
#
# Rule: **masked lanes must be neutral for whatever reduction consumes them.**
# Neutral for a max is `-inf`. Neutral for a sum is `0`.

# %% [markdown]
# ## 5. Streaming (online) softmax for rows wider than a block
#
# When the row does not fit in one block, you cannot do `tl.max` over the whole
# thing. Stream it: carry a running max `m` and running sum `l`, rescaling `l`
# whenever a block raises the max. This is the kernel behind FlashAttention's
# inner loop, which Module 14 builds in full.

# %%
@triton.jit
def softmax_streaming(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * n_cols

    m = -float("inf")          # running max
    l = 0.0                    # running sum, measured against m

    # ── pass A: stream to find (m, l) ──
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        xb = tl.load(x_ptr + base + cols, mask=mask, other=-float("inf"))
        m_new = tl.maximum(m, tl.max(xb, axis=0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(xb - m_new), axis=0)
        m = m_new

    # ── pass B: normalise and write ──
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        xb = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        e = tl.exp(xb - m)
        tl.store(out_ptr + base + cols, e / l, mask=mask)

# %%
def run_streaming(x, BLOCK=1024):
    M, N = x.shape
    out = torch.empty_like(x)
    softmax_streaming[(M,)](x, out, N, BLOCK=BLOCK)
    return out

for N in [4096, 16384, 65536]:
    x = torch.randn(8, N, device="cuda", dtype=torch.float32)
    a = run_streaming(x, BLOCK=1024)
    b = run_softmax(x, BLOCK=triton.next_power_of_2(N)) if N <= 65536 else None
    ref = torch.softmax(x, 1)
    line = f"N={N:>6}  streaming err {(a - ref).abs().max().item():.3e}"
    if b is not None:
        line += f"   fused err {(b - ref).abs().max().item():.3e}"
    print(line)

# %% [markdown]
# **Note the two loops in the streaming kernel.** It reads the row twice — once to
# find $(m, l)$, once to write the normalised values — because it cannot hold the
# row in registers. That is *worse* than the single-block fused kernel, which is
# why the fused version is preferred whenever the row fits.
#
# The streaming version still beats the naive version, because it needs only
# **two** passes instead of three, and it is the only option when the row is
# wider than any block. FlashAttention's trick (Module 14) is to eliminate the
# second pass entirely by keeping the partial sums in registers and deferring the
# final normalisation — that is the "online" part of *online softmax* proper.

# %% [markdown]
# ## 6. Measure it
#
# Now the point of the whole lesson: count bytes, then confirm with a clock.

# %%
def traffic_naive(n_rows, n_cols, itemsize=4):
    return 4 * n_rows * n_cols * itemsize      # 3 reads + 1 write

def traffic_fused(n_rows, n_cols, itemsize=4):
    return 2 * n_rows * n_cols * itemsize      # 1 read + 1 write

for (R, C) in [(64, 4096), (64, 16384), (256, 4096)]:
    x = torch.randn(R, C, device="cuda", dtype=torch.float32)
    tn = traffic_naive(R, C); tf = traffic_fused(R, C)
    print(f"[{R} x {C}]  naive {tn/1024:8.1f} KB   fused {tf/1024:8.1f} KB   ratio {tn/tf:.1f}x")

# %%
# A real attention row: one query token, one head, 4096 context, FP16.
R, C = 1, 4096
x = torch.randn(R, C, device="cuda", dtype=torch.float16)
t_fused = bench(lambda: run_softmax(x), iters=200)
t_pt = bench(lambda: torch.softmax(x, dim=1), iters=200)
print(f"[1 x 4096] fp16  fused {t_fused*1e6:.1f} us   torch {t_pt*1e6:.1f} us")
print(f"fused moves {traffic_fused(R,C,2)} bytes -> {traffic_fused(R,C,2)/t_fused/1e9:.0f} GB/s")

# %% [markdown]
# At this size you are again measuring **launch overhead**, not bandwidth: a
# single row is 8 KB, which takes ~25 ns to move but ~5 µs to launch. The
# win from fusion at this shape comes from doing the work in fewer kernels, not
# from fewer bytes.
#
# That is why the interesting measurement is at a realistic *batch* of rows:

# %%
print()
print(f"{'shape':>18} {'fused us':>10} {'torch us':>10} {'GB/s':>8} {'% peak':>8}")
for R in [1, 64, 512, 4096]:
    x = torch.randn(R, 4096, device="cuda", dtype=torch.float16)
    tf = bench(lambda: run_softmax(x), iters=100)
    tp = bench(lambda: torch.softmax(x, dim=1), iters=100)
    gbs = traffic_fused(R, 4096, 2) / tf / 1e9
    print(f"{str(tuple(x.shape)):>18} {tf*1e6:>10.1f} {tp*1e6:>10.1f} {gbs:>8.0f} {gbs/320*100:>7.0f}%")

# %% [markdown]
# Two things to read here:
#
# 1. **GB/s rises with row count.** Small shapes are launch-bound; large shapes
#    approach the T4's ~320 GB/s. This is the pattern for every fused kernel.
# 2. **`% peak` is the number to optimise.** A memory-bound kernel at 80% of peak
#    has nothing left. At 20%, you have a problem to find.

# %% [markdown]
# ## 7. The reciprocal trick
#
# `e / l` is a division per element. Replace it with one reciprocal per row.

# %%
@triton.jit
def softmax_recip(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    inv = 1.0 / tl.sum(e, axis=0)     # ONE division for the whole row
    tl.store(out_ptr + row * n_cols + cols, e * inv, mask=mask)

# %%
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
o1 = torch.empty_like(x); o2 = torch.empty_like(x)
softmax_fused[(4096,)](x, o1, 4096, BLOCK=4096)
softmax_recip[(4096,)](x, o2, 4096, BLOCK=4096)
t_div = bench(lambda: softmax_fused[(4096,)](x, o1, 4096, BLOCK=4096))
t_rec = bench(lambda: softmax_recip[(4096,)](x, o2, 4096, BLOCK=4096))
print(f"divide:  {t_div*1e6:.1f} us")
print(f"recip:   {t_rec*1e6:.1f} us")
print(f"max diff: {(o1-o2).abs().max().item():.3e}")
print("on a bandwidth-bound kernel the difference is small — as predicted. Measure before you bother.")

# %% [markdown]
# ## 8. Numerical stability, measured
#
# The max subtraction is not optional. Feed the kernel scores it cannot survive.

# %%
for scale in [1.0, 50.0, 500.0]:
    x = torch.randn(64, 4096, device="cuda", dtype=torch.float16) * scale
    try:
        out = run_softmax(x, BLOCK=4096)
        ref = torch.softmax(x.float(), 1)
        err = (out.float() - ref).abs().max().item()
        print(f"scale {scale:>6}: max err {err:.3e}   sums to {out.float().sum(1).mean().item():.4f}")
    except Exception as e:
        print(f"scale {scale:>6}: {type(e).__name__}: {e}")

# %%
# Note that FP16 itself overflows above ~65504, so the naive *unstabilised*
# version dies well before the stabilised one. Here is that comparison directly:

# %%
@triton.jit
def softmax_unstable(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    e = tl.exp(x)                    # NO max subtraction
    l = tl.sum(e, axis=0)
    tl.store(out_ptr + row * n_cols + cols, e / l, mask=mask)

# %%
x = torch.randn(8, 1024, device="cuda", dtype=torch.float32) * 100
o_stable = torch.empty_like(x); o_un = torch.empty_like(x)
softmax_fused[(8,)](x, o_stable, 1024, BLOCK=1024)
softmax_unstable[(8,)](x, o_un, 1024, BLOCK=1024)
ref = torch.softmax(x, 1)
print("scores ~ N(0, 100):")
print("  stabilised max err:", (o_stable - ref).abs().max().item())
print("  UNSTABILISED      :", "all NaN" if torch.isnan(o_un).all() else f"max err {(o_un-ref).abs().max().item():.3e}")
print("  -> exp(100) = 2.7e43. In FP32 that is inf, so the sum is inf and every")
print("     probability becomes 0/inf = 0. The row is silently destroyed.")

# %% [markdown]
# ## 9. Predict, then run
#
# 1. Row length 4,096, batch of 1, versus batch of 4,096. Which is closer to
#    peak bandwidth, and why is the small one worse even though it moves 4,096×
#    fewer bytes?
# 2. You replace `other=-inf` with `other=0.0`. Does the kernel error, produce
#    NaN, or quietly produce probabilities that do not sum to 1? Which, and why?
# 3. For a 128,256-entry vocabulary, `BLOCK = 65536` cannot work. What is the
#    minimum structural change, and what does it cost in memory traffic versus
#    the fused version?
# 4. The streaming kernel reads the row twice. Can you write a one-pass version?
#    (Hint: what if you keep the exponentiated block in registers and normalise
#    at the very end, storing only after the final block? What is the obstacle?)
# 5. Why is `tl.max` needed at all if you divide by the sum anyway — the sum
#    would renormalise regardless?

# %% [markdown]
# ## 10. What this is worth
#
# Price it for a real model. One decode step of Llama-3.1-8B: 32 layers, 32 query
# heads each, softmax over a 4,096-token context in FP16.

# %%
layers, heads, ctx, itemsize = 32, 32, 4096, 2
rows = layers * heads
bytes_naive = 4 * rows * ctx * itemsize
bytes_fused = 2 * rows * ctx * itemsize
print(f"softmax rows per decode step: {layers} layers x {heads} heads = {rows}")
print(f"naive traffic : {bytes_naive/1e6:.1f} MB")
print(f"fused traffic : {bytes_fused/1e6:.1f} MB")
print(f"saved         : {(bytes_naive-bytes_fused)/1e6:.1f} MB per token")
for name, bw in [("T4", 320e9), ("A100", 2.0e12), ("H100", 3.35e12)]:
    saved = (bytes_naive - bytes_fused) / bw
    print(f"  on {name:>6}: {saved*1e6:7.0f} us saved per generated token")

# %% [markdown]
# On a T4 that is about a millisecond per token — from one extra line of code.
# And it is a *pure* win: no approximation, no quality change, no tradeoff. That
# is the rarest kind of optimization, and it is why fusion is where every
# inference engine starts.
#
# Next: `13-rmsnorm-rope.ipynb`, where we fuse a *reduction* with the operation
# that consumes it, and cut the launch count of a real transformer layer.

# %%
print("\nDone. Next: 13-rmsnorm-rope.ipynb")