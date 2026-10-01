# %% [markdown]
# # Lab 15 · Compile & capture a decode loop
#
# **Tokens per Second** · Module 15
#
# By the end of this notebook you will have measured, on your own GPU, the two
# numbers that decide everything in this module:
#
# 1. **`launch_us`** — what one kernel launch costs *your* CPU. Lesson 15.1
#    assumed 6 µs; you are about to find the real number for this machine.
# 2. **`n_kernels`** — how many launches one decode step is made of.
#
# With those two you can compute the CPU bill of a decode step and decide, for
# this model on this GPU, whether you are **launch-bound** at all — and that
# answer is the whole reason this lab exists. Most people never check.
#
# Then you will do the two fixes for real: `torch.compile` the forward step, and
# capture it in a CUDA graph, watching the CPU bill collapse.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# We use **Qwen2.5-0.5B-Instruct** (0.49B parameters, 24 layers) so everything
# runs fast on a free T4. The *absolute* times are far smaller than for an 8B
# model; the *ratios* — which is all this module is about — transfer directly.
#
# **A note on the T4**: it has no BF16 tensor cores and no FP8. We keep the model
# in FP32 throughout, which is slower but perfectly adequate for measuring
# launch overhead, since launch cost is a property of the *CPU* and the driver,
# not of the arithmetic.

# %%
import torch, time, math
assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
print(f"{p.name}   {p.total_memory/1e9:.1f} GB   sm_{p.major}{p.minor}")
print(f"CUDA {torch.version.cuda}   torch {torch.__version__}")

# %%
# The timing helper. Two variants, and the distinction is the whole lesson:
#
# * `bench_gpu` — CUDA events around the work. This measures the **GPU**.
# * `bench_cpu` — `perf_counter` with the GPU *not* waited on. This measures
#   how long the CPU took to **say** what to do. If the two agree, you are
#   launch-bound. If the CPU number is much smaller, the GPU is the bottleneck
#   and none of this module's tricks will help you.

def bench_gpu(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / iters   # seconds

def bench_cpu(fn, iters=30, warmup=10):
    """CPU time to *issue* the work. Deliberately does not synchronise."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t_issue = (time.perf_counter() - t0) / iters  # no sync: the queue absorbs it
    torch.cuda.synchronize()
    return t_issue

# %% [markdown]
# ## 1. The model
#
# Qwen2.5-0.5B-Instruct: 0.49B parameters, **24 layers**, hidden size 896,
# 2 KV heads of head dim 64. Note the layer count early — it is what sets the
# kernel count, and therefore the launch bill.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()

cfg = model.config
PARAMS = sum(q.numel() for q in model.parameters())
print(f"params {PARAMS/1e9:.3f}B   layers {cfg.num_hidden_layers}   hidden {cfg.hidden_size}")
print(f"kv heads {cfg.num_key_value_heads} x head_dim {cfg.head_dim}   vocab {cfg.vocab_size}")

# %%
# One decode step: feed one token per sequence, with a cache, under
# `inference_mode` (which strips autograd bookkeeping from the dispatcher —
# Lesson 15.1's first free optimisation).

BATCH = 1
CTX = 1024

ids = torch.randint(0, cfg.vocab_size, (BATCH,), device="cuda")
cache = None

@torch.inference_mode()
def step():
    global cache
    out = model(ids.view(BATCH, 1), past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    return out.logits

# prime the cache so every later step is a true decode step (one token in)
step()
torch.cuda.synchronize()
print(f"cache ready · batch {BATCH} · context {CTX}")

# %% [markdown]
# ## 2. Part 1 — measure `launch_us` and `n_kernels`
#
# These are the two numbers the whole module turns on. Measure them directly.

# %%
from torch.profiler import profile, ProfilerActivity

def count_and_time(fn):
    """Kernel launches and GPU time for one call, via the PyTorch profiler."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
        fn()
        torch.cuda.synchronize()
    evts = [e for e in pr.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    launches = sum(e.count for e in evts if e.self_device_time_total > 0)
    gpu_us = sum(e.self_device_time_total for e in evts)
    return launches, gpu_us

n_kernels, gpu_us = count_and_time(step)
print(f"kernels per decode step : {n_kernels}")
print(f"GPU time per decode step: {gpu_us:.1f} us = {gpu_us/1000:.3f} ms")

# %%
# Now the CPU side. `t_issue` is the launch bill in seconds. Divide by the
# kernel count to get the cost of a single launch on this machine.

t_cpu = bench_cpu(step)
t_gpu = bench_gpu(step)
launch_us = (t_cpu / n_kernels) * 1e6

print(f"CPU time to ISSUE a step : {t_cpu*1e3:.3f} ms")
print(f"GPU time for a step      : {t_gpu*1e3:.3f} ms")
print(f"=> cost of one launch    : {launch_us:.2f} us")
print()
print(f"Lesson 15.1 assumed 6 us. Yours is {launch_us:.1f} us — "
      f"{'close to it' if 3 <= launch_us <= 10 else 'different, which is fine and exactly why you measure'}.")

# %%
# The verdict, in the form the lesson gives it: the step is the MAX of the two
# pipelines, not the sum.

cpu_ms, gpu_ms = t_cpu * 1e3, t_gpu * 1e3
step_ms = max(cpu_ms, gpu_ms)
ratio = cpu_ms / gpu_ms

print(f"step = max({cpu_ms:.3f} ms CPU, {gpu_ms:.3f} ms GPU) = {step_ms:.3f} ms")
print(f"CPU/GPU = {ratio:.2f}")
print()
if ratio > 1:
    print(f">>> LAUNCH-BOUND by {ratio:.1f}x. This model is paying {cpu_ms:.3f} ms")
    print(">>> of Python per token, and every one of the three levers in Module 15")
    print(">>> would help. Go on to Parts 2 and 3 — they will pay off here.")
else:
    print(f">>> GPU-BOUND. The CPU finishes issuing in {cpu_ms:.3f} ms while the GPU")
    print(f">>> needs {gpu_ms:.3f} ms, so the queue never empties. torch.compile and")
    print(">>> CUDA graphs will make this model *faster*, but barely — you are")
    print(">>> removing time that was never on the critical path.")
    print(">>> Still run Parts 2 and 3: the point is to see the ratio, not to")
    print(">>> be disappointed by it.")

# %%
# The derived prediction from Lesson 15.1, now with YOUR numbers:
#     t_cpu = N x c_launch
# Check that the measured t_cpu is consistent with the kernel count and launch cost.

predicted_ms = n_kernels * launch_us / 1e3
print(f"predicted t_cpu = {n_kernels} x {launch_us:.2f} us = {predicted_ms:.3f} ms")
print(f"measured  t_cpu = {cpu_ms:.3f} ms")
print(f"(these should agree to within a few percent; the gap is the sync and loop overhead)")

# %% [markdown]
# ## 3. Part 2 — `torch.compile` the forward step
#
# Compile the **forward step**, not `generate()`. The lesson explains why: the
# sampling loop and stopping criteria are full of `.item()` calls and data-
# dependent branching, which are exactly the things that cause graph breaks.
#
# First, make every graph break an error, so you know whether you have any.

# %%
print("compiling (this takes a minute or two on a T4, and it is a one-off cost)...")
t_compile_start = time.time()
compiled = torch.compile(model)
step()          # first call triggers tracing + Triton generation
torch.cuda.synchronize()
print(f"compile took {time.time() - t_compile_start:.1f} s")

# %%
nc, guc = count_and_time(compiled)
tc, tg = bench_cpu(compiled), bench_gpu(compiled)
lc = (tc / nc) * 1e6

print(f"               eager            compiled")
print(f"kernels/step  {n_kernels:>6d}            {nc:>6d}")
print(f"launch cost   {launch_us:>6.2f} us          {lc:>6.2f} us")
print(f"CPU issue     {cpu_ms:>6.3f} ms         {tc*1e3:>6.3f} ms")
print(f"GPU time      {gpu_ms:>6.3f} ms         {tg*1e3:>6.3f} ms")
print(f"step          {step_ms:>6.3f} ms         {max(tc,tg)*1e3:>6.3f} ms")
print()
print(f"speedup on the step: {step_ms / max(tc, tg):.2f}x")
print()
if tc < t_cpu * 0.95:
    print("The CPU bill fell — that is the interpreter leaving the loop.")
else:
    print("The CPU bill barely moved. Either the model was not launch-bound")
    print("(consistent with Part 1), or compilation hit graph breaks. Check below.")

# %%
# Did we get graph breaks? This is the tool from Lesson 15.2.
try:
    from torch._dynamo.explain import explain
    rep = explain(model)(ids.view(BATCH, 1), past_key_values=None, use_cache=True)
    print(f"graph_count        : {rep.graph_count}")
    print(f"graph_break_count  : {rep.graph_break_count}")
    print(f"op_count           : {rep.op_count}")
    reasons = rep.break_reasons or []
    if reasons:
        print("\nbreak reasons:")
        for r in reasons:
            print("  -", str(r)[:160])
except Exception as e:
    print("explain() unavailable in this torch build:", e)

# %% [markdown]
# ## 4. Part 3 — capture a CUDA graph
#
# This is the big lever: it removes the launch bill entirely, turning the whole
# step into **one API call**.
#
# The rules from Lesson 15.3, all three of which matter here:
#
# 1. **Warm up on a side stream first**, so cuBLAS workspaces and autotuning do
#    not get baked into the recording.
# 2. **Use a static input buffer.** The graph has addresses baked in; you copy
#    data into that buffer before each replay.
# 3. **The shapes are baked in too**, so this graph is only valid for
#    `BATCH = 1`.

# %%
@torch.inference_mode()
def step_from(ids_in):
    out = model(ids_in.view(BATCH, 1), past_key_values=None, use_cache=False)
    return out.logits

# rule 1: warm up on a side stream, three iterations
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        step_from(ids)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
print("warm-up done")

# rule 2: a static input buffer, allocated once
static_in = ids.clone()

# rule 3: capture. This executes nothing; it records the launches.
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        static_out = step_from(static_in)
    print("captured OK")
except Exception as e:
    print("capture failed:", type(e).__name__, e)
    raise

# %%
# Replay it and time it. Note what we measure: the replay is ONE call, so the
# CPU issue time should be microscopic. The GPU time should be *unchanged* —
# a graph never makes the GPU do less work.

for _ in range(10):
    g.replay()
torch.cuda.synchronize()

def replay_step():
    g.replay()

t_replay_cpu = bench_cpu(replay_step, iters=50, warmup=10)
t_replay_gpu = bench_gpu(replay_step, iters=50, warmup=10)

print(f"replay CPU issue : {t_replay_cpu*1e6:>8.1f} us   (was {cpu_ms*1e3:.0f} us eagerly)")
print(f"replay GPU time  : {t_replay_gpu*1e3:>8.3f} ms   (was {gpu_ms:.3f} ms eagerly)")
print()
print(f"step: eager {step_ms:.3f} ms -> graphed {t_replay_gpu*1e3:.3f} ms "
      f"= {step_ms / (t_replay_gpu*1e3):.2f}x")

# %%
# The correctness check that matters: does the graph actually compute the step?
# Compare it against the eager result for the same input.

with torch.inference_mode():
    ref = model(static_in.view(BATCH, 1), past_key_values=None, use_cache=False).logits
g.replay()
torch.cuda.synchronize()

match = torch.allclose(ref, static_out, atol=1e-3, rtol=1e-3)
print(f"logits match between eager and graph: {match}")
if not match:
    print(f"  max abs diff: {(ref - static_out).abs().max().item():.2e}")
    print("  (small differences are normal from non-deterministic reduction order)")

# %% [markdown]
# ## 5. What this all says about your GPU
#
# One table, three modes, and the verdict from Lesson 15.3: a graph only helps
# if the CPU was the longer pipeline.

# %%
print(f"{'mode':<10}{'CPU issue':>12}{'GPU time':>12}{'step':>12}")
print("-" * 46)
print(f"{'eager':<10}{cpu_ms:>9.3f} ms{gpu_ms:>9.3f} ms{step_ms:>9.3f} ms")
print(f"{'compiled':<10}{tc*1e3:>9.3f} ms{tg*1e3:>9.3f} ms{max(tc,tg)*1e3:>9.3f} ms")
print(f"{'graphed':<10}{t_replay_cpu*1e3:>9.3f} ms{t_replay_gpu*1e3:>9.3f} ms{t_replay_gpu*1e3:>9.3f} ms")
print()
best = min(step_ms, max(tc, tg), t_replay_gpu * 1e3)
print(f"best step: {best:.3f} ms")
print()
if step_ms / best > 1.15:
    print(f">>> Compilation and graphs together bought you {step_ms/best:.2f}x on this")
    print(">>> configuration. This model was launch-bound, as Qwen2.5-0.5B is")
    print(">>> on a T4, and the whole module applies to you directly.")
else:
    print(f">>> Only {step_ms/best:.2f}x. This configuration was never launch-bound,")
    print(">>> so removing CPU work barely moved the step. That is a real and")
    print(">>> useful result: you now know not to bother, and you know it from")
    print(">>> measurement rather than from a blog post.")

# %% [markdown]
# ## 6. Sweep the batch size
#
# The lesson's crossover, measured. The CPU bill should stay roughly flat as the
# batch grows (it depends on the kernel count, not on how many sequences there
# are), while the GPU bill grows with the KV cache you must read. Where they
# cross is the batch at which users stop being free.

# %%
print(f"{'batch':>7}{'CPU ms':>10}{'GPU ms':>10}{'bound':>10}{'tok/s':>10}")
print("-" * 47)

rows = []
for B in [1, 2, 4, 8, 16, 32]:
    bid = torch.randint(0, cfg.vocab_size, (B,), device="cuda")
    with torch.inference_mode():
        f = lambda: model(bid.view(B, 1), past_key_values=None, use_cache=False)
        c = bench_cpu(f, iters=20, warmup=5) * 1e3
        g_ = bench_gpu(f, iters=20, warmup=5) * 1e3
    st = max(c, g_)
    verdict = "CPU" if c > g_ else "GPU"
    rows.append((B, c, g_))
    print(f"{B:>7}{c:>9.3f}{g_:>9.3f}{verdict:>10}{B/(st/1e3):>10.1f}")

crossed = None
for i in range(1, len(rows)):
    if rows[i - 1][2] <= rows[i - 1][1] and rows[i][2] > rows[i][1]:
        crossed = rows[i][0]
        break
print()
if crossed:
    print(f">>> The crossover is between batch {rows[[r[0] for r in rows].index(crossed)-1]} and {crossed}.")
else:
    print(">>> No crossover in this range: the CPU term stays the longer one throughout.")

# %% [markdown]
# ## 7. Check yourself
#
# 1. Your `launch_us` — how does it compare to the 6 µs the lesson assumed? The
#    answer depends on your CPU and whether it is shared, which is exactly why
#    the lesson tells you to measure on the machine you deploy on.
# 2. Was your model launch-bound? If `CPU/GPU > 1` in Part 1, yes. If not, the
#    0.5B model on your T4 was fast enough that Python was not the problem.
# 3. Did `torch.compile` reduce the kernel count? If it did not, you probably hit
#    graph breaks — check the `explain()` output above.
# 4. Did the graph output match the eager output? It should, to within reduction
#    order noise. If it did not, your warm-up was insufficient.
# 5. From your batch sweep: where did the CPU and GPU bills cross?
#
# ## What to carry forward
#
# The habit, not the numbers: **measure both columns before you touch a knob.**
# `t_issue` (CPU) and `bench_gpu` (GPU) are two lines of code, and together they
# tell you whether any of this module's machinery can help you at all. Most
# configurations cannot, and knowing that early is worth more than the speedup
# you would have hoped for.
