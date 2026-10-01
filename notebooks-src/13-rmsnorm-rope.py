# %% [markdown]
# # Lab 13.3 · Fused RMSNorm & RoPE
#
# **Tokens per Second** · Module 13 · Writing Kernels with Triton
#
# You will:
# 1. Write RMSNorm as a reduction feeding an elementwise scale.
# 2. Write RoPE as a pairwise rotation and check it against a PyTorch reference.
# 3. Fuse them into one kernel and watch the launch count drop.
# 4. Confirm `n_spills` explains the cases where fusion does *not* help.
# 5. Measure at realistic batch sizes, where the arithmetic is actually the bottleneck.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.

# %%
import torch, time, math
import triton
import triton.language as tl

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
print("Triton", triton.__version__, "|", torch.cuda.get_device_name(0))

def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# %% [markdown]
# ## 1. RMSNorm
#
# `tl.sum(x*x, axis=0)` is the only reduction. Everything else is elementwise
# against the scalar it produces.

# %%
@triton.jit
def rmsnorm_kernel(x_ptr, out_ptr, w_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(out_ptr + row * n_cols + cols, x * rstd * w, mask=mask)


def rmsnorm(x, w, eps=1e-6):
    M, N = x.shape
    out = torch.empty_like(x)
    rmsnorm_kernel[(M,)](x, out, w, N, eps, BLOCK=triton.next_power_of_2(N))
    return out

torch.manual_seed(0)
M, N = 8, 4096
x = torch.randn(M, N, device="cuda", dtype=torch.float32)
w = torch.randn(N, device="cuda", dtype=torch.float32)

out = rmsnorm(x, w)
ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
print("RMSNorm max err:", (out - ref).abs().max().item())
print("output RMS (should be ~1):", out.pow(2).mean(-1).sqrt().mean().item())

# %% [markdown]
# **What `tl.sum(x * x, axis=0)` costs you:** one reduction per row, over `n_cols`
# elements. Everything after it is elementwise against one scalar held in a register.
# That single-scalar-then-broadcast is the signature shape of nearly every fused
# kernel in inference.

# %% [markdown]
# ## 2. RoPE
#
# Rotate each adjacent pair by `pos * theta_i`. We build the cos/sin tables on the
# CPU once — they are constant for the life of the model.

# %%
def make_rope_tables(head_dim, base=10000.0, max_pos=8192, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_pos, device=device).float()
    ang = pos[:, None] * inv_freq[None, :]
    return torch.cos(ang).contiguous(), torch.sin(ang).contiguous()

COS, SIN = make_rope_tables(128)
print("cos table:", tuple(COS.shape), "sin table:", tuple(SIN.shape))

# %%
@triton.jit
def rope_kernel(x_ptr, out_ptr, cos_ptr, sin_ptr, n_pairs, pos, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    half = tl.arange(0, BLOCK)
    mask = half < n_pairs

    lo = tl.load(x_ptr + row * 2 * n_pairs + 2 * half,     mask=mask, other=0.0)
    hi = tl.load(x_ptr + row * 2 * n_pairs + 2 * half + 1, mask=mask, other=0.0)

    c = tl.load(cos_ptr + pos * n_pairs + half, mask=mask, other=1.0)
    s = tl.load(sin_ptr + pos * n_pairs + half, mask=mask, other=0.0)

    tl.store(out_ptr + row * 2 * n_pairs + 2 * half,     lo * c - hi * s, mask=mask)
    tl.store(out_ptr + row * 2 * n_pairs + 2 * half + 1, lo * s + hi * c, mask=mask)


def rope(x, pos, n_pairs=64):
    M = x.shape[0]
    out = torch.empty_like(x)
    rope_kernel[(M,)](x, out, COS, SIN, n_pairs, pos, BLOCK=triton.next_power_of_2(n_pairs))
    return out

# %%
def rope_ref(x, pos, head_dim=128, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    ang = pos * inv_freq
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)

x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
for pos in [0, 1, 17, 4095]:
    a = rope(x, pos)
    b = rope_ref(x, pos)
    print(f"pos={pos:>5}  max err {(a-b).abs().max().item():.3e}  |x| preserved: {a.norm(dim=-1).mean().item():.4f} vs {x.norm(dim=-1).mean().item():.4f}")

# %% [markdown]
# The last column is the invariant that matters: **rotation preserves magnitude**.
# `|x|` is unchanged, which is why RoPE composes cleanly with RMSNorm instead of
# fighting it for control of the vector's scale.

# %% [markdown]
# ## 3. Fused: RMSNorm then RoPE, one kernel
#
# The intermediate never leaves registers.

# %%
@triton.jit
def fused_rmsnorm_rope(x_ptr, out_ptr, w_ptr, cos_ptr, sin_ptr, n_pairs, pos, eps,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    half = tl.arange(0, BLOCK)
    mask = half < n_pairs

    # ── RMSNorm: read the whole row, reduce, keep rstd in a register ──
    cols = tl.arange(0, BLOCK * 2)
    x = tl.load(x_ptr + row * BLOCK * 2 + cols, mask=cols < BLOCK * 2, other=0.0)
    w = tl.load(w_ptr + cols, mask=cols < BLOCK * 2, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / (BLOCK * 2) + eps)
    x = x * rstd * w

    # ── RoPE: consume x straight out of registers ──
    lo = tl.sum(tl.where((cols % (BLOCK * 2)) == 2 * half, x, 0.0), axis=0)   # even elements
    hi = tl.sum(tl.where((cols % (BLOCK * 2)) == 2 * half + 1, x, 0.0), axis=0)  # odd elements

    c = tl.load(cos_ptr + pos * n_pairs + half, mask=mask, other=1.0)
    s = tl.load(sin_ptr + pos * n_pairs + half, mask=mask, other=0.0)

    tl.store(out_ptr + row * BLOCK * 2 + 2 * half,     lo * c - hi * s, mask=mask)
    tl.store(out_ptr + row * BLOCK * 2 + 2 * half + 1, lo * s + hi * c, mask=mask)

# %%
D = 128   # head_dim: must be even
def fused(x, w, pos, eps=1e-6):
    M = x.shape[0]
    out = torch.empty_like(x)
    fused_rmsnorm_rope[(M,)](x, out, w, COS, SIN, D // 2, pos, eps, BLOCK=triton.next_power_of_2(D // 2))
    return out

x = torch.randn(4, D, device="cuda", dtype=torch.float32)
w = torch.randn(D, device="cuda", dtype=torch.float32)
a = fused(x, w, pos=17)
b = rope_ref(rmsnorm(x, w), pos=17)
print("fused vs unfused max err:", (a - b).abs().max().item())

# %% [markdown]
# ## 4. Count the launches
#
# This is the number the whole lesson is about. Count them with the profiler.

# %%
from torch.profiler import profile, ProfilerActivity

def count_launches(fn, label):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages() if e.device_time_total > 0]
    kernels = [n for n in names if "triton" in n.lower() or "elementwise" in n.lower()
               or "kernel" in n.lower() or "vectorized" in n.lower()]
    print(f"{label:>28}: {len(set(kernels)):>2} distinct kernels")
    for n in sorted(set(kernels))[:6]:
        print(f"      - {n[:70]}")
    return len(set(kernels))

B, D = 1, 128
xb = torch.randn(B, D, device="cuda", dtype=torch.float32)
wb = torch.randn(D, device="cuda", dtype=torch.float32)

n_unfused = count_launches(lambda: rope_ref(rmsnorm(xb, wb), 5), "PyTorch unfused (2 ops)")
n_fused = count_launches(lambda: fused(xb, wb, 5), "Triton fused (1 kernel)")

# %% [markdown]
# **This is the measurement to remember.** Two PyTorch ops became several CUDA
# kernels; the fused Triton kernel is one. At batch 1 the difference is almost
# entirely launch overhead, which is exactly the regime decode lives in.

# %% [markdown]
# ## 5. Time it
#
# Expect the fused version to win at batch 1 and to converge at large batch,
# because at large batch both are bandwidth-bound and move the same bytes.

# %%
B, D = 1, 128
print(f"{'batch':>7} {'unfused us':>12} {'fused us':>10} {'speedup':>8}  regime")
for B in [1, 4, 32, 256, 1024, 4096]:
    xb = torch.randn(B, D, device="cuda", dtype=torch.float16)
    wb = torch.randn(D, device="cuda", dtype=torch.float16)
    tu = bench(lambda: rope_ref(rmsnorm(xb.float(), wb.float()), 5), iters=100)
    tf = bench(lambda: fused(xb, wb, 5), iters=100)
    regime = "launch-bound" if B <= 32 else "bandwidth-bound"
    print(f"{B:>7} {tu*1e6:>12.1f} {tf*1e6:>10.1f} {tu/tf:>7.2f}x  {regime}")

# %% [markdown]
# **The crossover is the lesson.** At small batch the fused kernel wins big because
# it launches once instead of twice. At large batch the two converge because both
# are moving the same bytes and the memory system, not the launch count, is the
# limit.
#
# Decode is batch 1. That is why fusion matters most exactly where you care most
# about latency.

# %% [markdown]
# ## 6. When fusion does NOT help: register spills
#
# The honest failure case. Fuse too much and the block stops fitting in registers.

# %%
print(f"{'BLOCK':>8} {'regs':>6} {'spills':>8} {'time us':>10}")
for D in [128, 512, 2048, 8192, 32768]:
    try:
        xs = torch.randn(256, D, device="cuda", dtype=torch.float32)
        ws = torch.randn(D, device="cuda", dtype=torch.float32)
        ck = fused_rmsnorm_rope[(256,)](xs, torch.empty_like(xs), ws, COS, SIN,
                                       D // 2, 3, 1e-6, BLOCK=triton.next_power_of_2(D // 2))
        t = bench(lambda: fused_rmsnorm_rope[(256,)](xs, torch.empty_like(xs), ws, COS, SIN,
                                                     D // 2, 3, 1e-6,
                                                     BLOCK=triton.next_power_of_2(D // 2)), iters=50)
        print(f"{D:>8} {ck.n_regs:>6} {ck.n_spills:>8} {t*1e6:>10.1f}")
    except Exception as e:
        print(f"{D:>8}  {type(e).__name__}: {str(e)[:60]}")

# %% [markdown]
# Watch `n_spills`. When it turns non-zero, the block no longer fits in registers
# and the compiler is writing to slower local memory — adding back exactly the
# traffic fusion was meant to remove. A fused kernel that spills is **worse than
# the unfused version**, and the metadata tells you so before you benchmark.
#
# The fix is not to abandon fusion. It is to fuse *less* per kernel, or to process
# the row in chunks with a loop so the live set stays small.

# %% [markdown]
# ## 7. A chunked fused kernel for wide rows
#
# When the row is too wide to hold in registers, loop over it and keep only the
# running sum of squares live. Then do a second pass to normalize + rotate, using
# the stored `rstd`.

# %%
@triton.jit
def fused_rmsnorm_rope_chunked(x_ptr, out_ptr, w_ptr, cos_ptr, sin_ptr, n_pairs, pos, eps,
                               CHUNK: tl.constexpr):
    row = tl.program_id(0)
    base = row * 2 * n_pairs

    # pass 1: accumulate sum of squares only
    ss = 0.0
    for start in range(0, n_pairs, CHUNK):
        idx = start + tl.arange(0, CHUNK)
        mask = idx < n_pairs
        lo = tl.load(x_ptr + base + 2 * idx,     mask=mask, other=0.0)
        hi = tl.load(x_ptr + base + 2 * idx + 1, mask=mask, other=0.0)
        ss += tl.sum(lo * lo + hi * hi, axis=0)

    rstd = 1.0 / tl.sqrt(ss / (2 * n_pairs) + eps)

    # pass 2: normalize + rotate + store, one chunk live at a time
    for start in range(0, n_pairs, CHUNK):
        idx = start + tl.arange(0, CHUNK)
        mask = idx < n_pairs
        lo = tl.load(x_ptr + base + 2 * idx,     mask=mask, other=0.0) * rstd * tl.load(w_ptr + 2 * idx,     mask=mask, other=0.0)
        hi = tl.load(x_ptr + base + 2 * idx + 1, mask=mask, other=0.0) * rstd * tl.load(w_ptr + 2 * idx + 1, mask=mask, other=0.0)
        c = tl.load(cos_ptr + pos * n_pairs + idx, mask=mask, other=1.0)
        s = tl.load(sin_ptr + pos * n_pairs + idx, mask=mask, other=0.0)
        tl.store(out_ptr + base + 2 * idx,     lo * c - hi * s, mask=mask)
        tl.store(out_ptr + base + 2 * idx + 1, lo * s + hi * c, mask=mask)

# %%
for D in [2048, 8192]:
    n_pairs = D // 2
    xs = torch.randn(128, D, device="cuda", dtype=torch.float32)
    ws = torch.randn(D, device="cuda", dtype=torch.float32)
    o = torch.empty_like(xs)
    fused_rmsnorm_rope_chunked[(128,)](xs, o, ws, COS, SIN, n_pairs, 5, 1e-6,
                                       CHUNK=min(512, triton.next_power_of_2(n_pairs)))
    ref = rope_ref(rmsnorm(xs, ws), 5)
    print(f"D={D:>6} chunked fused max err: {(o - ref).abs().max().item():.3e}")

# %% [markdown]
# Two passes over the row, but only a `CHUNK` of it live at a time — so no spills,
# at any width. This is the general escape hatch and it is the same pattern as the
# streaming softmax from the previous notebook.

# %% [markdown]
# ## 8. What it is worth in a real decode step
#
# Llama-3.1-8B: 32 layers, hidden 4096, batch 1, FP16.

# %%
layers, hidden, itemsize = 32, 4096, 2
launches_unfused, launches_fused = 4, 1   # measured above, scaled per layer
per_token_us = (launches_unfused - launches_fused) * 5.0
print(f"launches saved per token : {layers * (launches_unfused - launches_fused)}")
print(f"time saved per token     : {per_token_us * layers:.0f} us  (at ~5 us/launch)")
print()
print("for comparison, a naive 8B decode step on an H100 was ~7 ms (Module 00).")
print(f"so this ONE fusion pattern is about {per_token_us * layers / 7000 * 100:.1f}% of that step.")

# %% [markdown]
# ## 9. Predict, then run
#
# 1. Count the kernel launches for `rmsnorm(x, w)` followed by `apply_rope(q, k, 0)`
#    on a `[1, 4096]` FP16 tensor. How many does your fused kernel need?
# 2. Measure at batch 1 and batch 256. Which is closer to peak bandwidth, and why is
#    the small one worse even though it does identical arithmetic?
# 3. Set `eps = 0` and feed an all-zero row. What happens, and where does it show?
# 4. If `n_spills` is non-zero on your fused kernel, does fusion still help? What is
#    the compiler telling you?

# %%
print("\nDone. Next: 13-benchmark.ipynb, where we learn to measure any of this honestly.")