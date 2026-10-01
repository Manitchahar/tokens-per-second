# %% [markdown]
# # Lab 13.5 · Challenge: optimize a real inference kernel
#
# **Tokens per Second** · Module 13 · Writing Kernels with Triton
#
# This is the Module 13 capstone. You will take the **SiLU-gated MLP** — a real,
# memory-bound operation from every LLaMA-family decode step — and carry it from
# correct to faster than PyTorch, with a roofline argument and a located
# crossover batch.
#
# The notebook is scaffolding. **You are expected to extend it**, not just run it.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.

# %%
import torch, time, math, statistics
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
print("Triton", triton.__version__, "|", torch.cuda.get_device_name(0))

T4_BW = 320e9        # GB/s, approx peak bandwidth
T4_FL = 65e12        # FP16 FLOP/s
HIDDEN, INTER = 4096, 14336   # Llama-3.1-8B MLP dimensions

# %% [markdown]
# ## Step 0 — The roofline argument, before any code
#
# Do this on paper first. It predicts your result.
#
# | quantity | per element |
# |---|---|
# | FLOPs: SiLU ~3, multiply ~1 | **~4** |
# | Bytes: read gate (2), read up (2), write h (2) | **6** |
# | Arithmetic intensity | **0.67 FLOP/byte** |
#
# T4 ridge = 65e12 / 320e9 = **203 FLOP/byte**. We are ~300× below it.
# **Conclusion: memory-bound. Runtime is set by bytes; only fewer bytes or fewer
# launches can help.**
#
# Floor at batch 1: `1 * 14336 * 6 = 86,016 bytes` / 320e9 = **269 ns**. A launch
# is ~5,000 ns. So at batch 1 we are roughly **20× launch-bound**.
#
# Check that against your measurements at the end.

# %%
n_b1 = INTER
bytes_fused = n_b1 * 2 * 3
print(f"batch 1 tensor: {n_b1} elems, {bytes_fused/1024:.1f} KB moved when fused")
print(f"floor @ peak  : {bytes_fused/T4_BW*1e9:.0f} ns")
print(f"ridge         : {T4_FL/T4_BW:.0f} FLOP/byte, ours is {4/6:.2f} -> memory-bound")

# %% [markdown]
# ## Step 1 — The kernel, and correctness
#
# `silu(gate) * up` is three elementwise ops on a `[B, 14336]` tensor. Fused.

# %%
@triton.jit
def swiglu_kernel(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(gate_ptr + offs, mask=mask)
    u = tl.load(up_ptr + offs, mask=mask)
    act = g / (1.0 + tl.exp(-g))          # SiLU: x * sigmoid(x)
    tl.store(out_ptr + offs, act * u, mask=mask)

def fused_swiglu(gate, up):
    n = gate.numel()
    out = torch.empty_like(gate)
    BLOCK = 1024
    swiglu_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
    return out

# %%
print("dtype  shape              max abs err   max rel err")
for dt in [torch.float16, torch.float32]:
    for shape in [(1, INTER), (8, INTER), (32, INTER), (7, 14337)]:   # last is ragged
        g = torch.randn(*shape, device="cuda", dtype=dt)
        u = torch.randn(*shape, device="cuda", dtype=dt)
        a = fused_swiglu(g, u)
        b = F.silu(g) * u
        err = (a.float() - b.float()).abs().max().item()
        rel = ((a.float() - b.float()).abs() / b.float().abs().clamp(min=1e-6)).max().item()
        tag = " <- ragged" if shape[1] != INTER else ""
        print(f"{str(dt).split('.')[-1]:>7}  {str(shape):<18} {err:>12.3e} {rel:>12.3e}{tag}")

# %% [markdown]
# **Check the ragged case.** `7 x 14337` is not a multiple of `BLOCK=1024`, so the
# last program is partially masked. If that row is correct, your masking is right.
#
# **Numerical safety:** `exp(-g)` overflows to `inf` for very negative `g`, but
# then `g / inf = 0`, which is the *correct* limit of SiLU as g → -∞. So the
# overflow is harmless by construction. For large positive `g`, `exp(-g) → 0`, so
# `act → g`, also correct. This is why SiLU is safe without any clamping.

# %%
g = torch.tensor([-1000.0, -50.0, 0.0, 50.0, 1000.0], device="cuda", dtype=torch.float16)
print("SiLU at extremes:", F.silu(g).tolist())
print("(all correct: ->0 on the left, ->g on the right)")

# %% [markdown]
# ## Step 2 — Count the launches you removed
#
# Measure, do not estimate.

# %%
from torch.profiler import profile, ProfilerActivity

def count_kernels(fn, label, iters=20):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ks = {e.key for e in prof.key_averages() if e.device_time_total > 0 and e.key.count("/") >= 2}
    print(f"{label:>24}: {len(ks)} kernel(s)")
    for k in sorted(ks):
        print(f"      {k[:66]}")
    return len(ks)

g1 = torch.randn(1, INTER, device="cuda", dtype=torch.float16)
u1 = torch.randn(1, INTER, device="cuda", dtype=torch.float16)
o1 = torch.empty_like(g1)
print()
n_pt = count_kernels(lambda: F.silu(g1) * u1, "PyTorch silu*mul")
n_tri = count_kernels(lambda: fused_swiglu(g1, u1), "Triton fused")
print(f"\nlaunches removed: {n_pt - n_tri}")

# %%
# bytes, as computed in Step 0:
print(f"eager PyTorch bytes ~ {4 * n_b1 * 2:,}  (read+write for silu, then read gate+up, write h)")
print(f"fused bytes        =   {bytes_fused:,}")
print(f"saved per MLP      =   {4 * n_b1 * 2 - bytes_fused:,} bytes")
print(f"x 32 layers/token  =   {(4 * n_b1 * 2 - bytes_fused) * 32 / 1e6:.2f} MB per generated token")

# %% [markdown]
# ## Step 3 — Autotune
#
# **Note the `key`.** `n` changes on every call in a real serving workload (batch
# changes with continuous batching). Keying on `n` would re-tune constantly. We
# key on a *size bucket* instead.

# %%
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128},   num_warps=2),
        triton.Config({'BLOCK': 256},   num_warps=2),
        triton.Config({'BLOCK': 512},   num_warps=4),
        triton.Config({'BLOCK': 1024},  num_warps=4),
        triton.Config({'BLOCK': 2048},  num_warps=8),
        triton.Config({'BLOCK': 4096},  num_warps=8),
        triton.Config({'BLOCK': 8192},  num_warps=16),
    ],
    key=['bucket'],
)
@triton.jit
def swiglu_auto(gate_ptr, up_ptr, out_ptr, n, bucket, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(gate_ptr + offs, mask=mask)
    u = tl.load(up_ptr + offs, mask=mask)
    act = g / (1.0 + tl.exp(-g))
    tl.store(out_ptr + offs, act * u, mask=mask)

def fused_auto(gate, up):
    n = gate.numel()
    out = torch.empty_like(gate)
    bucket = 1 << max(1, n.bit_length())       # one tuning run per order of magnitude
    swiglu_auto[(triton.cdiv(n, 1024),)](gate, up, out, n, bucket)
    return out

# %%
print("tuning (first call, may take a few seconds)...")
t0 = time.perf_counter()
for B in [1, 8, 32]:
    fused_auto(torch.randn(B, INTER, device="cuda", dtype=torch.float16),
               torch.randn(B, INTER, device="cuda", dtype=torch.float16))
print(f"done in {time.perf_counter()-t0:.2f}s — paid ONCE per size bucket")

for key, cfg in list(swiglu_auto.cache.items()):
    print(f"  bucket={key}: {cfg}")

# %% [markdown]
# **Does the winner move with batch?** Record it. The answer is almost always
# yes: small batches prefer small blocks (more programs to fill the SMs), large
# batches prefer large blocks (fewer, fatter programs, less launch overhead).
# If it does *not* move, that itself is a finding worth explaining.

# %% [markdown]
# ## Step 4 — The honest benchmark
#
# Warmup, L2 flush, CUDA events, median. No shortcuts.

# %%
def bench_kernel(fn, warmup=50, reps=200, flush_mb=128):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    flush = torch.empty(int(flush_mb * 1e6 // 4), dtype=torch.int, device="cuda")
    times = []
    for _ in range(reps):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)
    times.sort()
    return times[len(times) // 2]

print(f"{'batch':>7} {'torch us':>10} {'triton us':>10} {'speedup':>8} {'GB/s':>8} {'% peak':>7}  regime")
rows = []
for B in [1, 8, 32, 256, 1024, 4096]:
    g = torch.randn(B, INTER, device="cuda", dtype=torch.float16)
    u = torch.randn(B, INTER, device="cuda", dtype=torch.float16)
    o = torch.empty_like(g)
    rep = 50 if B >= 1024 else 200
    t_pt = bench_kernel(lambda: F.silu(g) * u, reps=rep)
    t_tr = bench_kernel(lambda: fused_swiglu(g, u), reps=rep)
    by = g.numel() * 2 * 3
    gbs = by / (t_tr * 1e-6) / 1e9
    pct = gbs * 1e9 / T4_BW * 100
    regime = "launch-bound" if pct < 25 else "bandwidth-bound"
    rows.append((B, t_pt, t_tr, pct))
    print(f"{B:>7} {t_pt:>10.1f} {t_tr:>10.1f} {t_pt/t_tr:>7.2f}x {gbs:>8.0f} {pct:>6.0f}%  {regime}")

# %% [markdown]
# **This table is your result.** Fill `RESULTS.md` with it.
#
# Things to check yourself:
# - Does the speedup shrink as batch grows? It should — launch overhead is amortized away.
# - Does `% peak` climb toward 60-80%? It should — that is the practical ceiling for a
#   three-stream elementwise kernel on a T4.
# - Is your best `% peak` consistent with the roofline? If you claim 90% on a T4,
#   be suspicious.

# %% [markdown]
# ## Step 5 — Find the crossover
#
# The most valuable sentence in your write-up.

# %%
best = max(rows, key=lambda r: r[1] / r[2])
worst = min(rows, key=lambda r: r[1] / r[2])
print(f"best speedup : {best[1]/best[2]:.2f}x at batch {best[0]}  ({worst and ''}{best[3]:.0f}% of peak)")
print(f"worst speedup: {worst[1]/worst[2]:.2f}x at batch {worst[0]}")
print()
print("Locate the crossover and write the sentence:")
print('  "my kernel is __x__ faster at batch __ and breaks even at batch __,"')
print("   because below batch __ the launch count dominates, and above it both")
print("   versions are at the memory wall.\"")

# %% [markdown]
# ## Step 6 — Verify the roofline prediction
#
# Did your measurement confirm the paper argument from Step 0?

# %%
B = 1
g = torch.randn(B, INTER, device="cuda", dtype=torch.float16)
u = torch.randn(B, INTER, device="cuda", dtype=torch.float16)
t = bench_kernel(lambda: fused_swiglu(g, u), reps=500)
by = g.numel() * 2 * 3
floor_us = by / T4_BW * 1e6
print(f"measured   : {t*1e3:.0f} ns")
print(f"floor @peak: {floor_us*1e3:.0f} ns")
print(f"ratio      : {t*1e3/floor_us*1e3:.1f}x above the memory floor")
print()
print("If you are 10-20x above the floor at batch 1, you are launch-bound, not")
print("bandwidth-bound — exactly as Step 0 predicted. That is the whole story.")

# %% [markdown]
# ## Step 7 — Run the sweep twice
#
# The habit that separates a benchmark from a guess.

# %%
def sweep():
    out = {}
    for B in [1, 8, 32, 256]:
        g = torch.randn(B, INTER, device="cuda", dtype=torch.float16)
        u = torch.randn(B, INTER, device="cuda", dtype=torch.float16)
        out[B] = bench_kernel(lambda: fused_swiglu(g, u), reps=100)
    return out

r1 = sweep(); r2 = sweep()
print(f"{'batch':>7} {'run 1 us':>10} {'run 2 us':>10} {'spread':>8}")
spreads = []
for B in r1:
    d = abs(r1[B] - r2[B]) / max(r1[B], r2[B]) * 100
    spreads.append(d)
    print(f"{B:>7} {r1[B]:>10.1f} {r2[B]:>10.1f} {d:>7.1f}%")
print(f"\nworst spread: {max(spreads):.1f}%")
print("If the two runs disagree by more than a few percent, something is wrong with")
print("your harness (thermal drift, another tenant, no warmup). State the spread in")
print("RESULTS.md — it is the confidence interval on every number you report.")

# %% [markdown]
# ## Extensions — pick at least two
#
# 1. **Fuse the whole MLP**: SiLU-and-multiply *plus* the read for the down-projection.
# 2. **Eliminate the ragged tail**: size the grid so programs divide the SMs exactly.
# 3. **Vectorized loads**: 128-bit accesses per thread. Did GB/s move?
# 4. **Backward pass**: one graph instead of three. This is where Triton really pays.
# 5. **A different target**: fused RMSNorm + residual add (option B) or top-p sampling (C).

# %% [markdown]
# ## Check yourself
#
# 1. Does your kernel match PyTorch at every shape, including a ragged length and
#    both dtypes?
# 2. Can you state in one sentence **why** this is memory-bound, with arithmetic?
# 3. Did you *measure* the PyTorch launch count, or estimate it?
# 4. Does your autotune `key` survive batch changing every step? Why?
# 5. Have you run the sweep twice and confirmed it agrees with itself?
# 6. Is there a batch where your kernel **loses** to PyTorch? Do you know why?
# 7. Best achieved bandwidth as % of peak — and what does that tell you about
#    whether more tuning would help?

# %% [markdown]
# ## Where this goes next
#
# **Module 14 — Attention Optimization.** FlashAttention is this exact technique at
# its largest scale: a memory-bound operation made memory-efficient by fusing passes
# and keeping intermediates on-chip. You are ready for it.
#
# **Module 15 — Compilation & CUDA Graphs.** `torch.compile`'s Triton backend
# generates this class of kernel for you. Now you can read what it produces, and
# debug it when it is slow.

# %%
print("\nModule 13 complete. Keep your harness.")