# %% [markdown]
# # Lab 01 · Tensors, dtypes & your first timing
#
# **Tokens per Second** · Module 01
#
# You will:
# 1. Build tensors and predict their shapes before printing them.
# 2. Measure how many bytes each dtype costs, and compute a model's size.
# 3. Time matrix multiplies and convert the time to TFLOP/s.
# 4. Discover, with your own measurements, that **matrix × vector** gets far fewer FLOP/s than **matrix × matrix**. This is the seed of everything in Module 02.
# 5. Write softmax yourself and check it against PyTorch.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.

# %%
import torch, time, math
assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
dev = "cuda"
print(torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. Shapes
#
# For each line, **say the shape out loud before you run it**. Then check.

# %%
x = torch.randn(3, 5, 4096)          # [B, T, D]
print("x          ", tuple(x.shape))
print("x[0]       ", tuple(x[0].shape))
print("x[0, 2]    ", tuple(x[0, 2].shape))
print("x[:, -1]   ", tuple(x[:, -1].shape))
heads = x.view(3, 5, 32, 128)        # split hidden into 32 heads of 128
print("heads      ", tuple(heads.shape))
print("transposed ", tuple(heads.transpose(1, 2).shape))
print("elements   ", x.numel(), "= 3*5*4096 =", 3 * 5 * 4096)

# %% [markdown]
# ## 2. Every number costs bytes
#
# `element_size()` tells you the bytes per number for a dtype.

# %%
for dt in [torch.float32, torch.float16, torch.bfloat16, torch.int8]:
    t = torch.zeros(1, dtype=dt)
    print(f"{str(dt):16s} {t.element_size()} bytes")

# Your turn: weights memory for a few models (billions x bytes = GB)
models = {"Qwen2.5-0.5B": 0.49e9, "Llama-3.1-8B": 8.03e9, "Llama-3.1-70B": 70.6e9}
for name, p in models.items():
    print(f"{name:14s}  BF16 {p*2/1e9:6.1f} GB   FP8 {p*1/1e9:6.1f} GB   INT4 {p*0.5/1e9:6.1f} GB")

# %% [markdown]
# **Precision in action.** 0.1 cannot be stored exactly. See how far off each format is:

# %%
for dt in [torch.float32, torch.float16, torch.bfloat16]:
    v = torch.tensor(0.1, dtype=dt).item()
    print(f"{str(dt):16s} stores 0.1 as {v:.10f}  (error {abs(v-0.1)/0.1*100:.4f}%)")
print("FP16 max:", torch.finfo(torch.float16).max, "  BF16 max:", torch.finfo(torch.bfloat16).max)
print("70000 in FP16 ->", torch.tensor(70000.0).half().item(), "(overflow!)")

# %% [markdown]
# ## 3. A timing helper you will reuse all course
#
# Rules: warm up first (the first call pays one-time costs), synchronize before and after, average many runs.

# %%
def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
dt_s = bench(lambda: a @ b)
flops = 2 * 4096 ** 3
print(f"[4096x4096] @ [4096x4096]: {dt_s*1e3:.2f} ms  ->  {flops/dt_s/1e12:.1f} TFLOP/s  (T4 FP16 peak ~65)")

# %% [markdown]
# ## 4. Matrix × vector vs matrix × matrix
#
# Now the key experiment. Keep the **weight** fixed at 4096 × 4096 (like one layer of a model) and change how many rows (tokens) we push through it: m = 1 (decode) up to m = 2048 (a long prefill).
#
# For each m we report:
# - **TFLOP/s**: how much math per second we achieved.
# - **GB/s**: how many bytes per second we moved (mostly the weight, 4096 × 4096 × 2 bytes = 33.5 MB).

# %%
W = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
print(f"{'m (tokens)':>10} {'time ms':>9} {'TFLOP/s':>9} {'GB/s':>8}  FLOPs per byte")
for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    X = torch.randn(m, 4096, device=dev, dtype=torch.float16)
    t = bench(lambda: X @ W)
    fl = 2 * m * 4096 * 4096
    by = (4096 * 4096 + 2 * m * 4096) * 2       # weight + input + output, 2 bytes each
    print(f"{m:>10} {t*1e3:>9.3f} {fl/t/1e12:>9.2f} {by/t/1e9:>8.0f}  {fl/by:>8.1f}")

# %% [markdown]
# **Read your table.**
#
# - At **m = 1** the time is almost the same as at m = 16 or 32: the GPU is busy *loading the weight*, not doing math. TFLOP/s is tiny, while GB/s sits near the T4's memory bandwidth (about 320 GB/s peak).
# - As m grows, the same weight bytes feed more math: TFLOP/s climbs toward the 65 peak and the time finally starts growing with m.
#
# You just measured, by yourself, that **decode (m = 1) is memory-bound and prefill (large m) is compute-bound**. The column "FLOPs per byte" is the **arithmetic intensity**. Module 02 turns this into the roofline model.

# %% [markdown]
# ## 5. Softmax by hand

# %%
def my_softmax(x):
    m = x.max(dim=-1, keepdim=True).values        # subtract the max: no overflow
    e = (x - m).exp()
    return e / e.sum(dim=-1, keepdim=True)

s = torch.tensor([[2.0, 1.0, 0.0], [100.0, 99.0, 98.0]])
print(my_softmax(s))
print(torch.softmax(s, dim=-1))
naive = s.exp() / s.exp().sum(dim=-1, keepdim=True)
print("naive (no max trick):", naive)       # the second row overflows in float32: e^100 is fine, but try .half()
print("naive in fp16:", s.half().exp() / s.half().exp().sum(dim=-1, keepdim=True))

# %% [markdown]
# Row 2 has the same *differences* as row 1, so the right answer is identical: [0.665, 0.245, 0.090]. In FP16 the naive version produces `nan` (inf / inf). The max trick fixes it.
#
# ## Your turn
# 1. In section 4, change the weight to 8192 × 8192. Does the m = 1 time double or quadruple? (It is 4× the bytes...)
# 2. Change the dtype in section 4 to `torch.float32`. What happens to TFLOP/s at large m? (T4's FP32 has no tensor cores.)
# 3. Write down your m = 1 GB/s and your best TFLOP/s in **My notes**. You will need them in Module 02.
