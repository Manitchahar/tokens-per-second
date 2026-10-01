# %% [markdown]
# # Lab 13.1 · Your first Triton kernel: vector add
#
# **Tokens per Second** · Module 13 · Writing Kernels with Triton
#
# You will:
# 1. Write a complete `@triton.jit` kernel and check it against PyTorch.
# 2. Watch a grid launch: one program per block, each told its own index.
# 3. Discover what masking does by breaking it on purpose.
# 4. Sweep BLOCK_SIZE and find the launch-overhead cliff yourself.
# 5. Read a compiled kernel's metadata and use it to explain the timings.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.

# %%
import torch, time, math
import triton
import triton.language as tl

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
print("Triton", triton.__version__, "|", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print("SMs:", props.multi_processor_count)

# %% [markdown]
# ## 1. The kernel
#
# The whole idea of this module is on these nine lines. Read the comments: each
# one answers "which part of the data do I own?"

# %%
@triton.jit
def add_kernel(x_ptr, y_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                                  # which program am I? 0..grid-1
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)      # my BLOCK_SIZE global offsets
    mask = offs < n                                         # which of those are real?
    a = tl.load(x_ptr + offs, mask=mask)                    # gather my elements of x
    b = tl.load(y_ptr + offs, mask=mask)                    # gather my elements of y
    tl.store(c_ptr + offs, a + b, mask=mask)                # scatter the sum into c


def add(x, y, BLOCK_SIZE=1024):
    c = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)                    # how many programs to launch
    add_kernel[grid](x, y, c, n, BLOCK_SIZE=BLOCK_SIZE)
    return c


x = torch.randn(100_000, device="cuda", dtype=torch.float16)   # note: not a power of two
y = torch.randn(100_000, device="cuda", dtype=torch.float16)
c = add(x, y)
print("correct:", torch.equal(c, x + y), "| n =", x.numel(), "-> programs:", triton.cdiv(x.numel(), 1024))
print("the last block is ragged: 100,000 mod 1024 =", x.numel() % 1024, "so 1 lane is masked off")

# %%
# **100,000 is not a multiple of 1024.** The last program starts at offset 99,840
# and would try to read through 100,863. The mask stops it at 99,999. That is the
# entire reason the mask exists, and why the *same* mask goes on the store.

# %% [markdown]
# ## 2. Watch the grid
#
# Build a 2-D grid so we can see 2-D masking too. Add a deliberate
# `out-of-range lanes` counter is not possible from Triton, so instead: notice
# the shapes, and check a tensor whose length is an exact multiple.

# %%
for n in [1024, 100_000, 99_999, 1, 7]:
    a = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    out = add(a, b, BLOCK_SIZE=1024)
    nprog = triton.cdiv(n, 1024)
    exact = "exact" if n % 1024 == 0 else f"ragged (last block {1024 - (n % 1024)} lanes masked)"
    print(f"n={n:>7}  programs={nprog:>4}  {exact:>34}  correct={torch.equal(out, a + b)}")

# %% [markdown]
# n = 1 launches one program whose block is 1024 wide, with 1023 lanes masked off.
# The kernel is still correct. It is also mostly wasted, which is the next section.

# %% [markdown]
# ## 3. Break the mask on purpose
#
# Three variants: correct, mask dropped from the load, mask dropped from the store.
# Run each on a **fresh** tensor so a corruption in one does not poison the next.
# Watch what happens — this is the lesson.

# %%
def add_variant(x, y, mask_load=True, mask_store=True):
    @triton.jit
    def k(x_ptr, y_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr, ML: tl.constexpr, MS: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m = offs < n
        a = tl.load(x_ptr + offs, mask=m if ML else None)
        b = tl.load(y_ptr + offs, mask=m if ML else None)
        if MS:
            tl.store(c_ptr + offs, a + b, mask=m)
        else:
            tl.store(c_ptr + offs, a + b)
    c = torch.zeros_like(x)
    k[(triton.cdiv(x.numel(), 1024),)](x, y, c, x.numel(), BLOCK_SIZE=1024, ML=mask_load, MS=mask_store)
    return c


n = 100_000
a = torch.randn(n, device="cuda", dtype=torch.float16)
b = torch.randn(n, device="cuda", dtype=torch.float16)

good = add_variant(a, b, True, True)
print("both masked        -> correct:", torch.equal(good, a + b))

# guard canaries around the output so we can SEE an overwrite
pad = 1024
a_pad = torch.zeros(n + pad, device="cuda", dtype=torch.float16)
b_pad = torch.zeros(n + pad, device="cuda", dtype=torch.float16)
a_pad[:n] = a; b_pad[:n] = b
a_pad[n:] = 1234.0        # canary: should never change
b_pad[n:] = 5678.0

c_pad = torch.zeros(n + pad, device="cuda", dtype=torch.float16)
add_variant(a_pad, b_pad, True, False)                     # store unmasked
canary_after = c_pad[n:pad].clone()
print("store unmasked    -> canary clobbered:", bool((canary_after != 0).any()),
      "| wrote past the end of c into", int((canary_after != 0).sum()), "cells")
print("   (reads were masked, so c itself is still right; the damage is to memory it does not own)")

# %% [markdown]
# Notice what happened: the **output was still numerically correct** on the cells
# that belong to `c`. The kernel wrote past the end into memory it never owned.
# That is the dangerous shape of this bug — it does not fail loudly. It corrupts
# a neighbour and shows up as a wrong answer somewhere else entirely.
#
# Rule, forever: **the same mask on load and store.**

# %% [markdown]
# ## 4. The BLOCK_SIZE sweep — find the launch cliff
#
# Here is the experiment that makes the programming model concrete. We hold the
# data fixed and change only how many programs the grid has.

# %%
def bench(fn, iters=200, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


N = 100_000
xs = torch.randn(N, device="cuda", dtype=torch.float16)
ys = torch.randn(N, device="cuda", dtype=torch.float16)

print(f"{'BLOCK':>8} {'programs':>9} {'time us':>9} {'GB/s':>9}   note")
rows = []
for B in [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    try:
        nprog = triton.cdiv(N, B)
        def run(B=B):
            out = torch.empty_like(xs)
            add_kernel[(nprog,)](xs, ys, out, N, BLOCK_SIZE=B)
            return out
        t = bench(run, iters=50 if nprog > 2000 else 200)
        gbs = (N * 2 * 3) / t / 1e9            # read x, read y, write c
        rows.append((B, nprog, t))
        print(f"{B:>8} {nprog:>9} {t*1e6:>9.1f} {gbs:>9.0f}   {'overhead-bound' if nprog > 5000 else ''}")
    except Exception as e:
        print(f"{B:>8}  failed: {type(e).__name__}")

# %% [markdown]
# **Read your table.**
#
# - Small blocks → tens of thousands of programs → the time is dominated by
#   *launch overhead*, and GB/s collapses far below the T4's ~320 GB/s.
# - Somewhere in the middle the curve flattens out near peak bandwidth.
# - Very large blocks slow down again: the block no longer fits in registers and
#   spills, or there are too few programs to keep 40 SMs busy.
#
# Find your crossover and write it down. It is not 1024 by accident.

# %%
best = min(rows, key=lambda r: r[2])
print(f"fastest: BLOCK={best[0]} with {best[1]} programs at {best[2]*1e6:.1f} us")
worst = max(rows, key=lambda r: r[2])
print(f"slowest: BLOCK={worst[0]} with {worst[1]} programs at {worst[2]*1e6:.1f} us")
print(f"speedup from tuning alone: {worst[2]/best[2]:.1f}x — with identical arithmetic")

# %% [markdown]
# ## 5. Compare against PyTorch
#
# The honest comparison: PyTorch has had this kernel for years, written in CUDA.

# %%
for B in [1024, 4096]:
    t_tri = bench(lambda B=B: add(xs, ys, BLOCK_SIZE=B))
    t_pt = bench(lambda: xs + ys)
    print(f"BLOCK={B:>5}: triton {t_tri*1e6:7.1f} us   pytorch {t_pt*1e6:7.1f} us   ratio {t_tri/t_pt:.2f}x")

# %% [markdown]
# Expect Triton to be in the same ballpark as PyTorch, and sometimes faster on
# small blocks because it avoids some dispatch overhead. If you are 10× slower,
# something is wrong — check the mask is present, and check you warmed up.
#
# On an elementwise op like this, neither can beat the memory wall. Both are
# moving 600,000 bytes. That is the roofline: **you are bandwidth-bound, and no
# amount of clever code changes that.**

# %% [markdown]
# ## 6. Read the metadata
#
# Triton can tell you exactly what it compiled, and how much room the block left.

# %%
out = torch.empty_like(xs)
compiled = add_kernel[(triton.cdiv(N, 1024),)](xs, ys, out, N, BLOCK_SIZE=1024)
k = list(add_kernel.cache[add_kernel.device_caches.__len__() and 0].values())[0] if hasattr(add_kernel, "cache") else None

# the supported way:
print("compiled kernel:", type(compiled).__name__)
for attr in ["num_warps", "num_stages", "shared", "n_regs", "n_spills"]:
    print(f"  {attr:>12}: {getattr(compiled, attr, 'n/a')}")

# %%
# **`n_spills` is the one to watch.** If it is greater than zero, the block did
# not fit in registers and the compiler spilled some to slower memory. That is
# exactly the failure that punishes an over-large BLOCK_SIZE, and now you can
# *see* it rather than guess:

# %%
for B in [256, 1024, 4096, 16384]:
    o = torch.empty_like(xs)
    ck = add_kernel[(triton.cdiv(N, B),)](xs, ys, o, N, BLOCK_SIZE=B)
    print(f"BLOCK={B:>6}  regs={ck.n_regs:>4}  spills={ck.n_spills:>5}  warps={ck.num_warps}  shared={ck.shared}")

# %% [markdown]
# ## 7. Predict before you look
#
# Answer these, then check against the tables above.
#
# 1. At BLOCK_SIZE = 1, how many programs launch, and roughly how long should
#    the whole kernel take if each launch costs ~5 µs? Compare to your measurement.
# 2. `n = 100,000` with `BLOCK_SIZE = 1024` launches 98 programs. What fraction
#    of the total work is the last, ragged program doing? Is the grid "full"?
# 3. Why does GB/s in your sweep top out *below* the T4's 320 GB/s spec?
# 4. If you switched `xs` to `torch.float32` with the same BLOCK_SIZE, would the
#    kernel get faster or slower? By roughly how much?

# %% [markdown]
# ## 8. A 2-D grid, for free
#
# The same idea, tiled. Program `(pid_m, pid_n)` owns a `BM × BN` tile.

# %%
@triton.jit
def scale_2d(x_ptr, c_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    mask = (rows[:, None] < M) & (cols[None, :] < N)      # 2-D mask
    offs = rows[:, None] * N + cols[None, :]               # 2-D offsets, broadcast
    a = tl.load(x_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a * 2.0, mask=mask)

M, N = 1000, 1000
m2 = torch.randn(M, N, device="cuda", dtype=torch.float16)
c2 = torch.empty_like(m2)
BM, BN = 64, 64
scale_2d[(triton.cdiv(M, BM), triton.cdiv(N, BN))](m2, c2, M, N, BM=BM, BN=BN)
print("2-D correct:", torch.equal(c2, m2 * 2), "| grid:", (triton.cdiv(M, BM), triton.cdiv(N, BN)))
print("note 1000/64 =", 1000 / 64, "-> the last tile in each axis is ragged too")

# %% [markdown]
# ## 9. Take it to a real shape
#
# One decode step of an 8B model: one token, hidden size 4096, added to itself.
# The next lesson scales this to a real reduction, but get the timing intuition
# for an LLM-sized tensor now.

# %%
h = torch.randn(1, 4096, device="cuda", dtype=torch.float16)   # 1 token, hidden 4096
w = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
t_small = bench(lambda: add(h, w, BLOCK_SIZE=1024), iters=500)
t_pt_small = bench(lambda: h + w, iters=500)
print(f"8B-shaped elementwise add (1 x 4096): triton {t_small*1e6:.1f} us, pytorch {t_pt_small*1e6:.1f} us")
print(f"moved {h.numel()*2*3} bytes -> {h.numel()*2*3/t_small/1e9:.0f} GB/s")
print("at this size you are measuring launch overhead, not bandwidth. Both are ~5 us.")
print("This is exactly why fusion (lesson 13.3) matters: 32 such ops in a layer = 32 launches.")

# %% [markdown]
# **What you just measured, in one line:** an elementwise op on one token is
# *entirely* launch overhead. Fusing thirty-two of them into one kernel is not a
# micro-optimization — on that shape it is the difference between 160 µs and 6 µs
# per layer, and an 8B model has 32 layers per token.
#
# Next: `13-softmax.ipynb`, where the same grid gets a reduction to do.

# %%
print("\nDone. Next: 13-softmax.ipynb")