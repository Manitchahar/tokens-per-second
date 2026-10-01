# %% [markdown]
# # Lab 13.4 · Benchmarking kernels against PyTorch
#
# **Tokens per Second** · Module 13 · Writing Kernels with Triton
#
# You will:
# 1. Watch four benchmark mistakes each inflate the result, one at a time.
# 2. Build an honest harness: warmup + L2 flush + CUDA events + median.
# 3. Convert timings into GB/s and TFLOP/s and judge them against the roofline.
# 4. Run `triton.autotune` and watch it search block sizes for you.
# 5. Compare your fused kernel against PyTorch at shapes that matter.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.

# %%
import torch, time, math, statistics
import triton
import triton.language as tl
from triton.testing import do_bench

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
print("Triton", triton.__version__, "|", torch.cuda.get_device_name(0))
print("T4 peak bandwidth ~320 GB/s, peak FP16 ~65 TFLOP/s")

# %% [markdown]
# ## 1. The kernel under test
#
# A fused RMSNorm: one reduction feeding one elementwise scale. Simple enough to
# be obviously correct, small enough to be entirely launch- and bandwidth-bound.

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

def run_rmsnorm(x, w, out, BLOCK=None):
    M, N = x.shape
    BLOCK = BLOCK or triton.next_power_of_2(N)
    rmsnorm_kernel[(M,)](x, out, w, N, 1e-6, BLOCK=BLOCK)

M, N = 256, 4096
x = torch.randn(M, N, device="cuda", dtype=torch.float16)
w = torch.randn(N, device="cuda", dtype=torch.float16)
out = torch.empty_like(x)

# %% [markdown]
# ## 2. Mistake 1: no warmup (you measure the compiler)
#
# The very first call JIT-compiles. Time it.

# %%
t0 = time.perf_counter()
run_rmsnorm(x, w, out)
torch.cuda.synchronize()
first = time.perf_counter() - t0
print(f"first call (compiles):  {first*1e3:8.1f} ms")

t0 = time.perf_counter()
run_rmsnorm(x, w, out)
torch.cuda.synchronize()
second = time.perf_counter() - t0
print(f"second call (cached):   {second*1e6:8.1f} us")
print(f"ratio: {first/second:,.0f}x — you were measuring PTX compilation, not the kernel")

# %% [markdown]
# ## 3. Mistake 2: no synchronize (you measure the CPU)
#
# CUDA is asynchronous. `run_rmsnorm` returns before the GPU has done anything.

# %%
# 20 launches with NO synchronize: the CPU just enqueues and races ahead
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    run_rmsnorm(x, w, out)
no_sync = (time.perf_counter() - t0) / 20
torch.cuda.synchronize()

# the same 20 launches, synchronized
t0 = time.perf_counter()
for _ in range(20):
    run_rmsnorm(x, w, out)
torch.cuda.synchronize()
with_sync = (time.perf_counter() - t0) / 20

print(f"20 launches, no sync:    {no_sync*1e6:8.1f} us each")
print(f"20 launches, synced:     {with_sync*1e6:8.1f} us each")
print(f"ratio: {with_sync/no_sync:.1f}x")
print()
print("no-sync measures how fast the CPU can enqueue. The GPU was still working")
print("long after Python returned — so the first number is not a kernel time at all.")

# %% [markdown]
# ## 4. Mistake 3: no L2 flush (you measure the cache)
#
# The T4 has a large L2. Re-running the same tensor serves it from cache.

# %%
nbytes = x.numel() * 2
print(f"tensor is {nbytes/1e6:.1f} MB")

def bench_no_flush(fn, iters=100):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)

def bench_flushed(fn, iters=100):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    cache = torch.empty(int(128e6 // 4), dtype=torch.int, device="cuda")
    ts = []
    for _ in range(iters):
        cache.zero_()                      # evict everything from L2
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)

t_nf = bench_no_flush(lambda: run_rmsnorm(x, w, out))
t_fl = bench_flushed(lambda: run_rmsnorm(x, w, out))
print(f"no flush:  {t_nf:7.1f} us -> {nbytes*3/t_nf/1e3:6.0f} GB/s reported")
print(f"flushed:   {t_fl:7.1f} us -> {nbytes*3/t_fl/1e3:6.0f} GB/s reported")
print(f"overstated by {t_nf/t_fl:.2f}x — a speedup that exists only because the data is cached")

# %% [markdown]
# ## 5. Mistake 4: one iteration (you measure noise)
#
# On a shared machine, a single sample is whatever jitter happened to land there.

# %%
samples = []
for _ in range(30):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); run_rmsnorm(x, w, out); e.record(); torch.cuda.synchronize()
    samples.append(s.elapsed_time(e) * 1e3)

samples.sort()
print(f"min {samples[0]:.1f} | median {samples[len(samples)//2]:.1f} | max {samples[-1]:.1f} us")
print(f"max/min = {samples[-1]/samples[0]:.1f}x — one sample could tell you anything in that range.")
print(f"Use the MEDIAN. Never report a single timing.")

# %% [markdown]
# ## 6. The honest harness
#
# All four fixes in one function. This is what you should write every time.

# %%
def bench_kernel(fn, warmup=25, reps=200, flush_mb=128):
    """Warm up, flush L2, time with CUDA events, return median microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    flush = torch.empty(int(flush_mb * 1e6 // 4), dtype=torch.int, device="cuda")
    times = []
    for _ in range(reps):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)   # elapsed_time is in ms -> us
    times.sort()
    return times[len(times) // 2]

t = bench_kernel(lambda: run_rmsnorm(x, w, out))
print(f"honest median: {t:.1f} us")

# %% [markdown]
# ### Or just use `do_bench`
#
# Triton's own implementation does all of this, correctly, and is faster than
# anything you will write by hand:

# %%
t_db = do_bench(lambda: run_rmsnorm(x, w, out), warmup=100, rep=500, return_mode="median")
print(f"do_bench (median): {t_db*1e3:.1f} us   [do_bench returns milliseconds]")
print(f"our harness:       {t:.1f} us")
print(f"agreement: {abs(t_db*1e3 - t)/t*100:.1f}% apart — same method, same answer")

# %% [markdown]
# ## 7. Convert to a rate, then judge against the roofline
#
# A time in microseconds compares to nothing. GB/s is comparable everywhere.

# %%
PEAK_BW = 320e9   # T4, approx
PEAK_FL = 65e12   # T4 FP16, approx

print(f"{'batch':>8} {'us':>9} {'GB/s':>9} {'% peak BW':>10} {'TFLOP/s':>9} {'verdict':>18}")
for B in [1, 8, 32, 128, 256, 1024, 4096, 16384]:
    xb = torch.randn(B, N, device="cuda", dtype=torch.float16)
    ob = torch.empty_like(xb)
    t = bench_kernel(lambda: run_rmsnorm(xb, w, ob), reps=100)
    by = (xb.numel() * 2) * 2 + N * 2        # read x, write out, read w
    fl = 4 * xb.numel()                      # square (1 mul), sum (1 add), scale (2)
    gbs = by / (t * 1e-6) / 1e9
    tfs = fl / (t * 1e-6) / 1e12
    pct_bw = gbs * 1e9 / PEAK_BW * 100
    verdict = "launch-bound" if pct_bw < 20 else ("tuning" if pct_bw < 70 else "near peak")
    print(f"{B:>8} {t:>9.1f} {gbs:>9.0f} {pct_bw:>9.0f}% {tfs:>9.2f} {verdict:>18}")

# %% [markdown]
# **Read this table carefully.** It is the whole lesson.
#
# - At batch 1, you get maybe 1% of peak bandwidth. The kernel is not slow because
#   it moves too little — it is slow because ~5 us of launch overhead dwarfs the
#   ~0.1 us of data movement. This is the **launch-bound** regime.
# - As batch grows, % of peak climbs toward 60-80%. Now you are **bandwidth-bound**
#   and the kernel is doing what a memory-bound kernel should.
# - TFLOP/s stays under 1 the entire time: the arithmetic is trivial compared to
#   the bytes. That is exactly right for RMSNorm, and it confirms you are on the
#   memory slope of the roofline, not the compute ceiling.
#
# **Never judge a kernel at one shape.** Sweep it.

# %% [markdown]
# ## 8. Autotuning: let Triton search
#
# `BLOCK_SIZE` is a constexpr — every value is a separate compilation. Let Triton
# benchmark them all once and keep the winner.

# %%
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def rmsnorm_autotuned(x_ptr, out_ptr, w_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(out_ptr + row * n_cols + cols, x * rstd * w, mask=mask)

# %%
def run_auto(xb, ob):
    rmsnorm_autotuned[(xb.shape[0],)](xb, ob, w, N, 1e-6)

print("first call triggers tuning (may take a few seconds)...")
t_tune = time.perf_counter()
run_auto(x, out)
torch.cuda.synchronize()
print(f"tuning took {time.perf_counter() - t_tune:.2f} s — paid ONCE, then cached")

best = rmsnorm_autotuned.cache
for key, cfg in list(best.items())[:3]:
    print(f"  n_cols={key[0]} -> {cfg}")
print("the winner is cached; later calls launch it directly with no search.")

# %%
print(f"{'batch':>8} {'fixed 1024':>12} {'autotuned':>11} {'gain':>7}")
for B in [1, 8, 32, 256, 1024, 4096]:
    xb = torch.randn(B, N, device="cuda", dtype=torch.float16)
    ob = torch.empty_like(xb)
    run_auto(xb, ob)   # ensure tuned
    t_fixed = bench_kernel(lambda: run_rmsnorm(xb, w, ob, BLOCK=1024), reps=100)
    t_auto = bench_kernel(lambda: run_auto(xb, ob), reps=100)
    print(f"{B:>8} {t_fixed:>11.1f}u {t_auto:>10.1f}u {t_fixed/t_auto:>6.2f}x")

# %% [markdown]
# The gain from autotuning is usually modest for a simple elementwise+reduction
# kernel (these are all bandwidth-bound once batch is large), but it can be large
# for tiled GEMMs where the interaction between `BLOCK_M`, `BLOCK_N`,
# `BLOCK_K` and `num_warps` is hard to reason about by hand.
#
# **The tuning cost is paid once per `key`.** Put only slow-varying shapes in
# `key`, or you re-tune on every call.

# %% [markdown]
# ## 9. The honest comparison against PyTorch
#
# PyTorch is not a strawman here — its RMSNorm kernel is already fused. The real
# question is whether *your* kernel beats it, and by how much, at shapes that matter.

# %%
print(f"{'batch':>8} {'triton us':>10} {'torch us':>10} {'speedup':>8}  note")
for B in [1, 8, 32, 128, 256, 1024, 4096]:
    xb = torch.randn(B, N, device="cuda", dtype=torch.float16)
    wb = w.half()
    ob = torch.empty_like(xb)

    def torch_rmsnorm():
        return xb * torch.rsqrt(xb.float().pow(2).mean(-1, keepdim=True) + 1e-6).half() * wb

    t_tri = bench_kernel(lambda: run_rmsnorm(xb, w, ob), reps=100)
    t_pt = bench_kernel(torch_rmsnorm, reps=100)
    note = "launch-bound" if B <= 32 else "bandwidth-bound"
    print(f"{B:>8} {t_tri:>10.1f} {t_pt:>10.1f} {t_pt/t_tri:>7.2f}x  {note}")

# %% [markdown]
# **The interesting result is the small-batch one.** At batch 1 the Triton kernel
# wins because it launches exactly once, while the PyTorch expression launches
# several elementwise/reduction kernels. At large batch they converge, because
# both are moving the same bytes through the same memory system — and no amount of
# cleverness beats a memory wall.
#
# That convergence is the honest outcome, and you should report it as such: "we are
# within 5% of cuBLAS-class bandwidth at batch >= 256, and 2x faster at batch 1
# where launch overhead dominates."

# %% [markdown]
# ## 10. Sanity-check the roofline arithmetic
#
# Compute the theoretical floor and confirm you are near it.

# %%
B = 4096
xb = torch.randn(B, N, device="cuda", dtype=torch.float16)
ob = torch.empty_like(xb)
t = bench_kernel(lambda: run_rmsnorm(xb, w, ob), reps=200)
by = (xb.numel() * 2) * 2 + N * 2

floor_us = by / PEAK_BW * 1e6          # time at 100% of peak
achieved = by / (t * 1e-6)
print(f"moved        : {by/1e6:.2f} MB")
print(f"measured     : {t:.1f} us")
print(f"floor @peak  : {floor_us:.1f} us")
print(f"achieved BW  : {achieved/1e9:.0f} GB/s = {achieved/PEAK_BW*100:.0f}% of peak")
print(f"overhead     : {t - floor_us:.1f} us above the memory floor")
print()
print("The gap above the floor is launch overhead + imperfect access patterns.")
print("On a bandwidth-bound kernel, that gap is all you can ever win.")

# %% [markdown]
# ## 11. Predict, then run
#
# 1. Time a Triton kernel 100× with `perf_counter` and no synchronize. What do you
#    actually measure, and what happens when you add a synchronize?
# 2. A tensor is 8 MB, benchmarked 100× with no flush. On a 50 MB L2, is it read
#    from HBM or cache? How do you prove it?
# 3. Convert a kernel moving 68 MB in 42 µs into GB/s, and as a % of H100 peak.
# 4. `key=['n_cols']` where `n_cols` changes every call: what happens?
# 5. Your kernel hits 82% of the roofline. Keep tuning, or stop?

# %% [markdown]
# ## 12. Take it away
#
# The habit to build: **never report a kernel timing you did not warm up, flush,
# synchronize, and repeat.** It takes four lines and it is the difference between
# a measurement and a guess.
#
# Next: the capstone. Take a real inference kernel — something with genuine room
# to improve — and carry it all the way from "correct" to "faster than PyTorch,
# and I can prove why".

# %%
print("\nDone. Next: the Module 13 challenge, 13-challenge-kernel.ipynb")