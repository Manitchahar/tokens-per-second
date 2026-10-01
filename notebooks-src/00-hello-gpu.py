# %% [markdown]
# # Lab 00 · Say hello to a real GPU
#
# **Tokens per Second** · Module 00
#
# In this notebook you will:
# 1. Check that you really have a GPU, and what kind.
# 2. Read `nvidia-smi`, the GPU's dashboard.
# 3. Put numbers on the GPU and multiply them.
# 4. Race the CPU against the GPU, and time it *correctly*.
#
# **Before you start:** in Colab, go to **Runtime → Change runtime type → T4 GPU → Save**.
# Then run each cell with **Shift + Enter**. Read the text between cells; it explains what just happened.

# %% [markdown]
# ## 1. Do we have a GPU?
#
# PyTorch is the library we will use to talk to the GPU. `torch.cuda.is_available()` asks: *is an NVIDIA GPU visible?*
# (CUDA is NVIDIA's software for running code on its GPUs. You will meet it properly in Module 02.)

# %%
import torch

print("PyTorch version:", torch.__version__)
print("GPU available:  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:       ", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory:      {props.total_memory / 1e9:.1f} GB")
    print(f"SM count:        {props.multi_processor_count}  (independent processing blocks, Module 02)")
else:
    print("No GPU. In Colab: Runtime -> Change runtime type -> T4 GPU, then run this cell again.")

# %% [markdown]
# You should see **Tesla T4**, about **15 GB** of memory and **40** SMs.
#
# ## 2. The GPU's dashboard: `nvidia-smi`
#
# `nvidia-smi` is the first command every inference engineer types on a new machine. The `!` runs a shell command from the notebook.
#
# Look for: the GPU name, **Memory-Usage** (used / total), **GPU-Util** (how busy the chip is right now), power draw, and temperature.

# %%
!nvidia-smi

# %% [markdown]
# Right now memory usage is near 0 and utilization is 0%: the GPU is idle. Let's give it work.
#
# ## 3. Numbers on the GPU
#
# A **tensor** is a grid of numbers. `torch.randn(4096, 4096)` makes a 4096 × 4096 grid of random numbers (about 16.8 million of them).
# By default it lives in the computer's normal memory (RAM), used by the CPU. `.to("cuda")` copies it into GPU memory.

# %%
a = torch.randn(4096, 4096)
print("a lives on:", a.device, "| dtype:", a.dtype, "| bytes:", a.numel() * a.element_size() / 1e6, "MB")

a_gpu = a.to("cuda")
print("a_gpu lives on:", a_gpu.device)
print(f"GPU memory in use by PyTorch: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

# %% [markdown]
# 16.8 million numbers × 4 bytes each (`float32`) = 67 MB. That matches. **Every number costs bytes**, and bytes are what inference engineers count all day.
#
# ## 4. CPU vs GPU: a matrix multiply race
#
# Multiplying two 4096 × 4096 matrices takes 2 × 4096³ ≈ **137 billion** floating-point operations (you will derive this formula in Module 01).
#
# ⚠️ **Timing GPUs has a trap.** When Python tells the GPU to do something, it does not wait for the GPU to finish: it just queues the work and moves on.
# To time GPU work honestly you must call `torch.cuda.synchronize()`, which means "wait until the GPU is actually done". Forgetting this is the #1 benchmarking bug.

# %%
import time

b = torch.randn(4096, 4096)
b_gpu = b.to("cuda")

# CPU
t0 = time.perf_counter()
c = a @ b
cpu_s = time.perf_counter() - t0

# GPU: warm up once (the first call pays one-time setup costs), then time properly
_ = a_gpu @ b_gpu
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()
gpu_s = (time.perf_counter() - t0) / 10

flops = 2 * 4096**3
print(f"CPU: {cpu_s * 1000:8.1f} ms  -> {flops / cpu_s / 1e12:6.2f} TFLOP/s")
print(f"GPU: {gpu_s * 1000:8.1f} ms  -> {flops / gpu_s / 1e12:6.2f} TFLOP/s")
print(f"GPU is {cpu_s / gpu_s:.0f}x faster")

# %% [markdown]
# ## 5. Half precision: fewer bytes, more speed
#
# LLMs are served in 16-bit numbers (`float16` or `bfloat16`), not 32-bit. Half the bytes per number, and the GPU's **tensor cores** (special matrix-multiply hardware) switch on.
# The T4 supports `float16` tensor cores (it is too old for `bfloat16` ones).

# %%
a16 = a_gpu.half()
b16 = b_gpu.half()
_ = a16 @ b16
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    c16 = a16 @ b16
torch.cuda.synchronize()
fp16_s = (time.perf_counter() - t0) / 10
print(f"FP32 on GPU: {gpu_s * 1000:6.2f} ms  ({flops / gpu_s / 1e12:5.1f} TFLOP/s)")
print(f"FP16 on GPU: {fp16_s * 1000:6.2f} ms  ({flops / fp16_s / 1e12:5.1f} TFLOP/s)")
print(f"FP16 speedup: {gpu_s / fp16_s:.1f}x   (T4 peak FP16 tensor-core speed is about 65 TFLOP/s)")

# %% [markdown]
# ## 6. Your turn
#
# 1. Run `!nvidia-smi` again. Memory usage is no longer zero: your tensors are sitting in GPU memory.
# 2. Change the size from 4096 to 1024 and rerun the race. Is the GPU's advantage bigger or smaller? (Hint: small jobs cannot keep thousands of cores busy. Module 02 explains why.)
# 3. Delete the `torch.cuda.synchronize()` lines in section 4 and rerun. The GPU suddenly looks impossibly fast. You have just seen the #1 benchmarking bug with your own eyes.
#
# **What to remember**
# - Numbers live in a specific memory (CPU RAM or GPU memory) and moving them costs time.
# - Every number costs bytes: `float32` = 4, `float16` = 2.
# - GPUs crush big matrix multiplies, especially in 16-bit on tensor cores.
# - Always `torch.cuda.synchronize()` before stopping a GPU timer.
