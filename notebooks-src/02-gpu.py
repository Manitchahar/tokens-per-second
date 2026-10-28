# %% [markdown]
# # Lab 02 · Measure your GPU's two speed limits
#
# **Tokens per Second** · Module 02
#
# By the end of this notebook you will have measured, on real hardware, the two numbers that decide how fast everything else in the course can possibly be:
#
# 1. **Bandwidth** — how many bytes per second this GPU can read.
# 2. **Compute** — how many FLOP per second this GPU can do.
#
# Then you will place your own operations on the roofline and see which limit each one hits. Section 6 is the one that matters: you will produce your own roofline plot.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# **Write every number you get into your notes.** You will be asked to compute with them in Module 05, and there is no substitute for having measured them yourself.

# %% [markdown]
# ## 0. What GPU am I actually running on?
#
# Say what you expect before you run it. If your guess is wrong, find out why.

# %%
import torch, time, math, subprocess
assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
print(f"Name                  {p.name}")
print(f"Multiprocessors (SMs)  {p.multi_processor_count}")
print(f"Total memory           {p.total_memory/1e9:.1f} GB")
print(f"Compute capability     sm_{p.major}{p.minor}")
print(f"torch                  {torch.__version__}")
print(f"cuBLAS on this GPU     {torch.backends.cuda.matmul.allow_tf32=}")

# %%
# The driver's own view, straight from the operating system.
print(subprocess.run(["nvidia-smi",
                      "--query-gpu=name,memory.total,driver_version,clocks.max.sm",
                      "--format=csv"],
                     capture_output=True, text=True).stdout)

# %% [markdown]
# ## 1. A timing helper that does not lie
#
# Module 01's lesson: CUDA launches are asynchronous, so you must synchronize. Here is the version using **CUDA events**, which are measured on the GPU itself and do not include host launch overhead. This is the helper you will reuse for the rest of the course.
#
# Three rules:
# 1. **Warm up** — the first call pays one-off costs (kernel compilation, memory allocation, cache fills).
# 2. **Synchronize** — or use events, which is better.
# 3. **Average** — and check that more iterations do not change the answer.

# %%
def bench(fn, iters=50, warmup=10):
    """Seconds per call, measured with CUDA events."""
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

# %%
# Prove the point. Time the same work with and without a synchronize.
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

t0 = time.perf_counter()
for _ in range(50):
    c = a @ b
no_sync = (time.perf_counter() - t0) / 50          # WRONG: times the launches

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    c = a @ b
torch.cuda.synchronize()
with_sync = (time.perf_counter() - t0) / 50         # RIGHT

flops = 2 * 4096 ** 3
print(f"without synchronize : {no_sync*1e6:8.1f} us   ->  {flops/no_sync/1e12:8.1f} TFLOP/s  (a lie)")
print(f"with synchronize    : {with_sync*1e6:8.1f} us   ->  {flops/with_sync/1e12:8.1f} TFLOP/s  (the truth)")
print(f"event timing        : {bench(lambda: a@b)*1e6:8.1f} us")
print(f"launch overhead alone: {no_sync*1e6:8.1f} us")

# %% [markdown]
# ## 2. Speed limit #1: memory bandwidth
#
# Bandwidth means: how many bytes can you pull from HBM in a second? The measurement is deliberately dumb. Move a big tensor and time it. Almost no arithmetic happens; it is a pure memory test.
#
# A copy reads N bytes and writes N bytes, so a 1 GB tensor moved with `copy_` moves 2 GB.

# %%
def measure_bandwidth(nbytes_each_way=512e6, iters=30):
    n = int(nbytes_each_way // 2)                    # fp16
    src = torch.empty(n, dtype=torch.float16, device="cuda").normal_()
    dst = torch.empty_like(src)
    t = bench(lambda: dst.copy_(src), iters=iters)
    moved = 2 * src.numel() * src.element_size()    # read + write
    return moved / t, t

bw, t = measure_bandwidth()
print(f"measured HBM bandwidth: {bw/1e9:.0f} GB/s   ({t*1e3:.2f} ms for 1 GB moved)")

# %%
# Is it stable? Does size matter? Bigger transfers amortize launch overhead better.
for mb in [1, 4, 16, 64, 256, 1024]:
    n = int(mb * 1e6 // 2)
    src = torch.empty(n, dtype=torch.float16, device="cuda").normal_()
    dst = torch.empty_like(src)
    t = bench(lambda: dst.copy_(src), iters=30)
    print(f"{mb:>5} MB moved   {2*src.numel()*2/t/1e9:>7.0f} GB/s")

# %%
# Pure read bandwidth: summing a tensor reads it and writes almost nothing.
x = torch.empty(int(512e6 // 4), dtype=torch.float32, device="cuda").normal_()
t = bench(lambda: torch.sum(x))
print(f"pure read (sum):      {x.numel()*4/t/1e9:.0f} GB/s")

# %%
# **Write it down.** `my_bw` is the single most useful number in this notebook.
my_bw = bw
print(f"\n>>> my_bw = {my_bw/1e9:.0f} GB/s")
print("T4 spec is about 320 GB/s. What fraction did you reach?")

# %% [markdown]
# ## 3. Speed limit #2: compute
#
# Now do the opposite: as much arithmetic as possible on as little memory as possible. A large square matrix multiply does this, because the weight is reused across every row.
#
# Use large sizes, because small matrix multiplies are launch-bound rather than compute-bound.

# %%
def measure_tflops(size=8192, iters=20):
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    t = bench(lambda: a @ b, iters=iters)
    fl = 2 * size ** 3
    return fl / t, t

tf, t = measure_tflops()
my_peak = tf
print(f"measured FP16 compute: {tf:.1f} TFLOP/s   ({t*1e3:.2f} ms for 2 x {8192**3*2/1e12:.1f} TFLOP)")

# %%
# Size matters: small matmuls never reach peak. Find the size where it saturates.
for size in [512, 1024, 2048, 4096, 8192, 16384]:
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    t = bench(lambda: a @ b, iters=20)
    fl = 2 * size ** 3
    by = (size*size + 2*size*size) * 2
    print(f"{size:>6}^3   {fl/t/1e12:>7.1f} TFLOP/s   intensity {fl/by:>8.1f} FLOP/byte   {t*1e6:>8.0f} us")

# %%
# Precision changes the compute ceiling, not the bytes.
print()
for dt, name in [(torch.float32, "FP32"), (torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
    a = torch.randn(8192, 8192, device="cuda", dtype=dt)
    b = torch.randn(8192, 8192, device="cuda", dtype=dt)
    t = bench(lambda: a @ b, iters=20)
    print(f"{name}  {2*8192**3/t/1e12:>7.1f} TFLOP/s   ({2*8192**3/t/my_bw*100:>5.1f}% of your measured bandwidth)")

# %%
print(f"\n>>> my_peak = {my_peak:.1f} TFLOP/s")

# %% [markdown]
# ## 4. The ridge point, from your own numbers
#
# ```
# ridge point (FLOP per byte) = peak compute / peak bandwidth
# ```
#
# Below the ridge you are memory-bound. Above it you are compute-bound. Compute yours and write it down next to the two numbers above it.

# %%
ridge = my_peak * 1e12 / my_bw
print(f"peak compute   {my_peak:8.1f} TFLOP/s")
print(f"peak bandwidth {my_bw/1e12:8.3f} TB/s")
print(f"ridge point    {ridge:8.0f} FLOP per byte")
print()
print("T4 reference: 65 TFLOP/s, 320 GB/s, ridge ~203 FLOP/byte.")
print("H100:        989 TFLOP/s, 3.35 TB/s, ridge ~295 FLOP/byte.")

# %% [markdown]
# ## 5. Is it memory-bound or compute-bound? Decide with arithmetic
#
# For a matrix multiply of `[m x 4096] @ [4096 x 4096]` in FP16:
#
# ```
# FLOPs = 2 * m * 4096 * 4096
# bytes = (4096*4096 + 2*m*4096) * 2
# time_floor = max(FLOPs / peak_flops, bytes / bandwidth)
# ```
#
# Run the sweep and compare the **measured** time with the **floor**. The ratio is your efficiency: how much of the roof you actually got.

# %%
K = 4096
W = torch.randn(K, K, device="cuda", dtype=torch.float16)
print(f"{'m':>6} {'measured us':>12} {'floor us':>10} {'eff %':>7} {'TFLOP/s':>9} {'GB/s':>8} {'intensity':>10}  bound")
for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    X = torch.randn(m, K, device="cuda", dtype=torch.float16)
    t = bench(lambda: X @ W, iters=100 if m <= 64 else 30)
    fl = 2 * m * K * K
    by = (K*K + 2*m*K) * 2
    floor = max(fl / my_peak, by / my_bw)
    bound = "compute" if fl/my_peak > by/my_bw else "memory"
    print(f"{m:>6} {t*1e6:>12.1f} {floor*1e6:>10.1f} {floor/t*100:>7.1f} {fl/t/1e12:>9.2f} {by/t/1e9:>8.0f} {fl/by:>10.1f}  {bound}")

# %% [markdown]
# **Read your own table before moving on.** You should see:
#
# - The measured time is nearly **flat** from m = 1 to m = 32. The GPU is not getting slower as you add work, because it is not doing more math: it is reading the same 33.5 MB of weights. This is why batching is nearly free for the first few dozen users.
# - Efficiency at m = 1 should be well above 50% (memory-bound work is easier to do well). Efficiency at large m will be higher still if you use FP16 and big matrices.
# - Intensity climbs almost linearly with m, because the weight bytes are constant.

# %% [markdown]
# ## 6. Your own roofline plot
#
# Plot achieved TFLOP/s against arithmetic intensity, both on log axes. Draw:
# 1. The **bandwidth floor**: `y = bandwidth * intensity`.
# 2. The **compute ceiling**: a horizontal line at `my_peak`.
# 3. A point per row of your sweep from section 5.
# 4. The **ridge point** where the two lines meet.
#
# Label the two ends. The whole right-hand half is "compute-bound" and the whole left-hand half is "memory-bound".

# %%
import matplotlib.pyplot as plt
import numpy as np

pts = []
for m in [1, 4, 16, 64, 256, 1024]:
    X = torch.randn(m, K, device="cuda", dtype=torch.float16)
    t = bench(lambda: X @ W, iters=50)
    fl = 2 * m * K * K
    by = (K*K + 2*m*K) * 2
    pts.append((fl/by, fl/t/1e12, m))
    del X

I = np.logspace(-1, 3.4, 200)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(I, my_bw * I / 1e12, color="#58c4dd", lw=2,
        label=f"bandwidth floor ({my_bw/1e9:.0f} GB/s)")
ax.axhline(my_peak, color="#f2b544", lw=2, ls="--", label=f"compute ceiling ({my_peak:.0f} TFLOP/s)")

xs = np.logspace(-1, 3.4, 200)
ax.fill_between(xs, my_peak, my_bw * xs / 1e12, color="#4a5262", alpha=0.18)
ax.plot([i for i, _, _ in pts], [f for _, f, _ in pts], "o", ms=9,
        color="#83c167", mec="black", label="measured")
for i, f, m in pts:
    ax.annotate(f"m={m}", (i, f), textcoords="offset points", xytext=(6, -12), fontsize=8)

ridge = my_peak * 1e12 / my_bw
ax.axvline(ridge, color="#7c8391", ls=":", lw=1)
ax.annotate(f"ridge {ridge:.0f} FLOP/byte", (ridge, my_peak), textcoords="offset points",
            xytext=(8, -16), fontsize=9, color="#7c8391")
ax.text(1.2, my_peak * 0.4, "memory-bound\n(adding FLOPs changes nothing)", fontsize=10, color="#58c4dd")
ax.text(ridge * 6, my_peak * 0.4, "compute-bound\n(more bytes would not help)", fontsize=10, color="#f2b544")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("arithmetic intensity  (FLOPs per byte)")
ax.set_ylabel("achieved  (TFLOP/s)")
ax.set_title(f"{p.name}: my roofline")
ax.grid(alpha=0.2, which="both")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Kernel launch overhead, measured
#
# Time N empty-ish kernel launches and divide. This is the floor under every Python-driven GPU program, and it is why fusing small operations matters.

# %%
tiny = torch.zeros(8, device="cuda")
for n in [1, 10, 100, 1000, 10000]:
    t = bench(lambda: [tiny.add_(1.0) for _ in range(n)], iters=5, warmup=2)
    print(f"{n:>6} tiny kernels  {t/n*1e6:>7.2f} us each   {t*1e3:>8.2f} ms total")

# %%
t = bench(lambda: torch.relu(W), iters=30)     # one kernel over 33.5 MB
byt = W.numel() * 2 * 2                          # read + write
print(f"\none fused ReLU over 33.5 MB: {t*1e6:.1f} us  ->  {byt/t/1e9:.0f} GB/s")
print("Compare with the 10,000-kernel row above. This is the whole argument for fusion.")

# %% [markdown]
# ## 8. The memory hierarchy, made visible
#
# The same read, at four levels of the hierarchy. Each one should be a different order of magnitude, and the L2 and L1 numbers will be far better than HBM.

# %%
# HBM: 512 MB tensor
big = torch.randn(256 * 1024 * 1024, dtype=torch.float32, device="cuda")
t = bench(lambda: torch.sum(big))
print(f"HBM read          {big.numel()*4/t/1e9:>8.0f} GB/s   {t*1e6:>9.0f} us")

# %%
# L2: a 8 MB tensor. Whether it stays resident depends on this GPU's L2, so
# read the absolute GB/s, not the ratio to the HBM row.
med = torch.randn(2 * 1024 * 1024, dtype=torch.float32, device="cuda")
t = bench(lambda: torch.sum(med))
print(f"small tensor read {med.numel()*4/t/1e9:>8.0f} GB/s   {t*1e6:>9.0f} us   ({med.numel()*4/1e6:.0f} MB)")

# %%
# L1 / shared: a single small read, entirely inside one SM's on-chip memory.
sm = torch.randn(32 * 1024, dtype=torch.float32, device="cuda")
t = bench(lambda: torch.sum(sm), iters=200)
print(f"single-SM read    {sm.numel()*4/t/1e9:>8.0f} GB/s   {t*1e6:>9.0f} us   ({sm.numel()*4/1e6:.2f} MB)")

# %%
l2 = getattr(p, "L2_cache_size", 0)
print(f"\nL2 cache size reported by the driver: {l2/1e6:.1f} MB" if l2 else "\nL2 size not exposed on this version")
print("The rows above should differ by roughly an order of magnitude each. That gap")
print("is why a kernel that reuses its data in shared memory is not a micro-optimisation.")

# %% [markdown]
# ## 9. Divergence and coalescing, measured
#
# Two experiments that show why the shape of your code matters more than the model.

# %%
# (a) Warp divergence: one data-dependent branch versus the same work without one.
n = 64 * 1024 * 1024
a = torch.randn(n, device="cuda")
b = torch.randn(n, device="cuda")

def with_branch():
    out = torch.empty_like(a)
    for _ in range(4):
        out = torch.where(a > 0, b, b * 2)      # branchless: select, do not branch
    return out

def with_arith():
    out = torch.empty_like(a)
    for _ in range(4):
        out = a * b + b                          # plain arithmetic
    return out

t1 = bench(with_branch, iters=20)
t2 = bench(with_arith, iters=20)
print(f"torch.where (branchless select): {t1*1e3:7.2f} ms  {(4*n*2*2)/t1/1e9:6.0f} GB/s")
print(f"plain arithmetic:                {t2*1e3:7.2f} ms  {(4*n*2*2)/t2/1e9:6.0f} GB/s")

# %%
# (b) Coalescing: a contiguous read versus a strided one, same number of elements.
idx = torch.arange(n, device="cuda")
t_contig = bench(lambda: a[idx])
t_stride = bench(lambda: a[::8])
n_contig, n_stride = a.numel(), a[::8].numel()
print(f"\ncontiguous a[idx]   {t_contig*1e3:7.2f} ms  {n_contig*4/t_contig/1e9:6.0f} GB/s effective")
print(f"strided    a[::8]   {t_stride*1e3:7.2f} ms  {n_stride*4/t_stride/1e9:6.0f} GB/s effective")
print("The strided read touches the same cache lines to deliver 8x fewer numbers.")

# %%
del big, med, sm, a, b
torch.cuda.empty_cache()

# %% [markdown]
# ## Your turn
#
# 1. **Section 2**: change `nbytes_each_way` to 8 GB (you will need a smaller `iters`). Does the bandwidth change? Why is 1 GB not always the best size to test?
# 2. **Section 3**: try `torch.float8_e4m3fn` if your GPU supports it (T4 does not; try on an L4 or H100). What happens to TFLOP/s and to the intensity?
# 3. **Section 5**: change `K` to 8192. Does the m = 1 time quadruple? It should: 4x the bytes. Confirm it, and explain why the FLOPs stay proportional to m.
# 4. **Section 6**: add a second curve for a memory-bound operation, such as `torch.relu(W)` or `torch.add(a, b)`, sitting near 1 FLOP/byte. It should land right on the bandwidth floor.
# 5. **Section 7**: rewrite the tiny-kernel loop as a single fused operation and show the time difference.
# 6. **Write in your notes:** `my_bw`, `my_peak`, `my_ridge`, and the m=1 GB/s from section 5. Module 05 assumes you have them.
