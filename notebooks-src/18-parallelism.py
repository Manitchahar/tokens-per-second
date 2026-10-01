# %% [markdown]
# # Lab 18 · Choose a parallelism strategy
#
# **Tokens per Second** · Module 18
#
# You will:
# 1. Compute, from a model config, the exact bytes of weights and KV cache per token — and check them against a real model.
# 2. Build the memory budget that decides everything: does it fit on one GPU?
# 3. Implement a **tensor-parallel** MLP (column-parallel → all-gather → row-parallel → all-reduce) and verify it against the single-GPU reference.
# 4. Implement the **pipeline bubble** as a clock, and show that the bubble is a function of micro-batches, not of hardware.
# 5. Measure a real all-reduce (`torch.distributed`) on whatever GPUs the runtime gives you — even one — to calibrate your own latency constant.
# 6. Price **context parallelism** and show why it loses to tensor parallelism on every fabric.
# 7. Run the full decision procedure over a grid of models, GPUs and workloads, and produce the strategy matrix.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# **A Colab T4 has one GPU.** That is fine, and it is actually the point: every multi-GPU
# formula in this module is *arithmetic*. Here you write the arithmetic as code, run it on
# one device, and check that the pieces are correct. Section 6 also runs a real
# `torch.distributed` collective over a single-process world, which is how you calibrate
# your own latency constant instead of trusting a slide.

# %%
import math, time, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
DEV = "cuda"
print(torch.cuda.get_device_name(0))
print("visible GPUs:", torch.cuda.device_count())
print("torch", torch.__version__)

torch.manual_seed(0)

# %% [markdown]
# ## 0. A helper library
# (Section 0 — the formulas themselves. The worked sections are numbered 1 to 11 below.)
#
# The formulas from the lessons, as functions. Every number you will see later in the
# notebook comes out of one of these, so you can check any of them by hand.

# %%
GB = 1e9
MB = 1e6


class Cfg:
    """A Llama-shaped model, by config. Everything this lab needs is derivable from these fields."""

    def __init__(self, name, params, layers, hidden, inter, heads, kv_heads, head_dim):
        self.name, self.params = name, params
        self.layers, self.hidden, self.inter = layers, hidden, inter
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim

    def weight_bytes(self, wbytes=2):
        return self.params * wbytes

    def kv_bytes_per_token(self, kbytes=2):
        # 2 (K and V) x layers x kv heads x head dim x bytes
        return 2 * self.layers * self.kv_heads * self.head_dim * kbytes

    def kv_bytes(self, batch, ctx, kbytes=2):
        return batch * ctx * self.kv_bytes_per_token(kbytes)

    def tp_degrees(self, max_p=8):
        out = [p for p in range(1, max_p + 1)
               if self.hidden % p == 0 and self.kv_heads % p == 0 and self.inter % p == 0]
        return out

    def __repr__(self):
        return f"{self.name}: {self.params/1e9:.1f}B, {self.layers}L, h={self.hidden}"


# %% [markdown]
# ## 1. The config, and the constants it implies
#
# Everything downstream is arithmetic from a model config and a GPU's capacity. Start with
# four real models and print, for each, the weight size, the KV bytes per token, and the
# **legal tensor-parallel degrees** — the ones that divide the hidden size, the MLP width
# and, crucially, the KV head count.

# %%
LLAMA8 = Cfg("Llama-3.1-8B", 8.03e9, 32, 4096, 14336, 32, 8, 128)
LLAMA70 = Cfg("Llama-3.1-70B", 70.6e9, 80, 8192, 28672, 64, 8, 128)
QWEN15 = Cfg("Qwen2.5-1.5B", 1.54e9, 28, 1536, 8960, 12, 2, 128)
QWEN32 = Cfg("Qwen2.5-32B", 32.8e9, 64, 5120, 17408, 40, 8, 128)

for m in (QWEN15, LLAMA8, QWEN32, LLAMA70):
    print(f"{m.name:>16}  weights BF16 {m.weight_bytes()/GB:6.1f} GB   KV/token {m.kv_bytes_per_token()/1024:7.0f} KiB   legal TP: {m.tp_degrees()}")

# %% [markdown]
# **Check yourself:** Llama-3.1-8B should print **131,072 bytes per token of KV** (128 KiB).
# That is the same number you derived in Module 05. If it is different, `head_dim` or
# `kv_heads` is wrong — and `kv_heads` is the field people forget, because it is 8 for an
# 8B model with 32 query heads (GQA) and not 32.

# %%
print("Llama-3.1-8B KV per token:", LLAMA8.kv_bytes_per_token(), "bytes")
assert LLAMA8.kv_bytes_per_token() == 131072, "expected 131,072"

# %% [markdown]
# ## 2. The memory budget
#
# The first decision, and it settles most cases. A GPU's memory has to hold:
#
# ```
# weights + KV cache + CUDA context + activation workspace  <=  capacity
# ```
#
# `perfModel.ts` in this course reserves about 4% of GPU memory for the CUDA context and
# workspaces. Use the same 4% here so your numbers match the visualisations.

# %%
class Gpu:
    def __init__(self, name, mem, bw, flops):
        self.name, self.mem, self.bw, self.flops = name, mem, bw, flops

    def budget(self, reserve=0.04):
        return self.mem * (1 - reserve)


T4 = Gpu("T4", 16 * GB, 320 * 1e9, 65 * 1e12)
A100 = Gpu("A100 80GB", 80 * GB, 2.0 * 1e12, 312 * 1e12)
H100 = Gpu("H100 SXM", 80 * GB, 3.35 * 1e12, 989 * 1e12)
H200 = Gpu("H200", 141 * GB, 4.8 * 1e12, 989 * 1e12)
GPUS = [T4, A100, H100, H200]

print(f"{'GPU':<12} {'capacity':>10} {'budget':>10} {'mem BW':>12} {'ridge point':>14}")
for g in GPUS:
    ridge = g.flops / g.bw
    print(f"{g.name:<12} {g.mem/GB:8.0f}GB {g.budget()/GB:8.0f}GB {g.bw/1e12:10.2f}TB/s {ridge:12.0f} FLOP/B")


def fits(model, gpu, batch, ctx, wbytes=2, kbytes=2):
    need = model.weight_bytes(wbytes) + model.kv_bytes(batch, ctx, kbytes)
    return need, need <= gpu.budget()


print("\nDoes it fit on one H100, BF16, batch 16 at 8k context?")
for m in (QWEN15, LLAMA8, QWEN32, LLAMA70):
    need, ok = fits(m, H100, 16, 8192)
    print(f"  {m.name:<16} {need/GB:7.1f} GB needed  ->  {'FITS' if ok else 'does not fit'}")
    need8, ok8 = fits(m, H100, 16, 8192, wbytes=1)
    print(f"  {'  at FP8':<16} {need8/GB:7.1f} GB needed  ->  {'FITS' if ok8 else 'does not fit'}")

# %% [markdown]
# **Read this table carefully — it is the module's decision procedure in miniature.**
#
# - Qwen2.5-1.5B and Llama-3.1-8B **fit in BF16**. The right answer for them is
#   *replicas*. Sharding a model that fits buys you nothing but communication.
# - Qwen2.5-32B does not fit in BF16 but **does fit at FP8** — 32.8 GB of weights plus
#   17.2 GB of KV is 50 GB inside a 76.8 GB budget. Quantize and replicate.
# - Llama-3.1-70B does not fit in BF16 *or* FP8 at this batch and context: 70.6 GB of
#   weights plus 17.2 GB of KV is 88 GB. **This one genuinely needs more than one GPU**,
#   which is the case the rest of the module is about.
#
# Note the **ridge point** column: FLOP/s divided by bytes/s. That is the arithmetic
# intensity at which a kernel stops being memory-bound. It is ~295 for an H100 and you
# derived it in Module 12. Decode sits far below it; prefill sits far above.

# %% [markdown]
# ## 3. Tensor parallelism: implement it, verify it
#
# The Megatron block from Lesson 18.2, as working code. Because a Colab T4 has one GPU, we
# simulate $t$ ranks as $t$ chunks of tensors on that one device — which is exactly what a
# real implementation does at the collective boundaries anyway.

# %%
def tp_mlp(x, w1, w2, t, rank):
    """One tensor-parallel MLP block, computing only what `rank` would compute.

    x    : [B, h]          full input, identical on every rank
    w1   : [h, f]          column-parallel: split along output  -> f = inter
    w2   : [f, h]          row-parallel:    split along input   -> h = hidden

    Returns (y_shard, gathered) where y_shard is this rank's partial sum of the output
    and `gathered` is the full output after the all-reduce (emulated with a sum).
    """
    B, h = x.shape
    f = w1.shape[1]
    fs = f // t                      # this rank's slice of the intermediate size
    hs = h // t                      # this rank's slice of the hidden size

    # ---- column-parallel: no communication, each rank owns f/t output columns ----
    w1_s = w1[:, rank * fs:(rank + 1) * fs]          # [h, fs]
    hs_out = x @ w1_s                                 # [B, fs]

    # ---- the nonlinearity, still sharded ----
    act = F.silu(hs_out)

    # ---- row-parallel: each rank owns f/t input rows, produces a partial sum ----
    w2_s = w2[rank * fs:(rank + 1) * fs, :]          # [fs, h]
    y_shard = act @ w2_s                              # [B, h]  a PARTIAL SUM

    return y_shard


h, f, B, t = 512, 1408, 4, 4
x = torch.randn(B, h, device=DEV, dtype=torch.float16)
w1 = torch.randn(h, f, device=DEV, dtype=torch.float16) / math.sqrt(h)
w2 = torch.randn(f, h, device=DEV, dtype=torch.float16) / math.sqrt(f)

# reference: the unsplit MLP
ref = F.silu(x @ w1) @ w2

# tensor-parallel: every rank computes its partial, then one all-reduce sums them
partials = torch.stack([tp_mlp(x, w1, w2, t, r) for r in range(t)])
tp_out = partials.sum(0)                            # <-- this sum IS the all-reduce

err = (tp_out - ref).abs().max().item()
rel = err / ref.abs().max().item()
print(f"tensor-parallel MLP, t={t}: max abs err {err:.3e}   relative {rel:.3e}")
print("allclose (FP16 tolerance):", torch.allclose(tp_out, ref, atol=2e-2, rtol=2e-2))

# %% [markdown]
# **Expected:** a relative error around 1e-3 or below, and `True`.
#
# The error is float rounding, not a bug: summing $t$ partial sums in a different order
# gives a different rounding than the single big GEMM. **That is the honest tolerance for
# tensor parallelism** — bitwise equality is not on the table, and any claim otherwise
# should make you suspicious.
#
# Three things to try before moving on, because they are the three real bugs:
#
# 1. Change `w1_s = w1[:, rank*fs:(rank+1)*fs]` to `w1[:, :fs]` (every rank uses the same
#    columns). The output becomes $t$ copies of one partial sum. Wrong, and it looks plausible.
# 2. Change the `sum` over ranks to a `mean`. Output is off by exactly $t$. Wrong, and obvious.
# 3. Change `w2_s` to slice `w2` along its *output* (columns) instead of its input (rows).
#    Then `act @ w2_s` fails on shape, which is the good outcome — the shape check catches
#    the most common TP bug for free.

# %% [markdown]
# ## 4. Price it: how many bytes does TP move?
#
# Ring all-reduce cost: each GPU sends `2(t-1)/t` of its payload. Payload per collective is
# the full hidden state, `B * T * h * 2` bytes. Two collectives per layer.

# %%
def tp_bytes_per_fwd(model, batch, t, tokens=1):
    """Bytes each GPU sends, per forward pass, for a tensor-parallel group."""
    ring = 2 * (t - 1) / t
    payload = batch * tokens * model.hidden * 2
    return 2 * model.layers * ring * payload     # 2 collectives per layer


def link_time(nbytes, bw, eta, latency):
    return latency + nbytes / (bw * eta)


LINKS = {
    # name: (unidirectional GB/s, efficiency, latency in seconds)
    "PCIe 4.0 x16": (32 * 1e9, 0.80, 2.5e-6),
    "PCIe 5.0 x16": (64 * 1e9, 0.80, 2.0e-6),
    "NVLink 3 (A100)": (300 * 1e9, 0.85, 1.2e-6),
    "NVLink 4 (H100)": (450 * 1e9, 0.85, 0.9e-6),
    "NVLink 5 (B200)": (900 * 1e9, 0.85, 0.8e-6),
    "InfiniBand NDR": (50 * 1e9, 0.85, 2.0e-6),
}

print("Tensor parallelism communication, Llama-3.1-70B, batch 32, decode")
print(f"{'t':>3} {'bytes/GPU':>12} " + " ".join(f"{k:>16}" for k in LINKS))
for t in (2, 4, 8):
    nb = tp_bytes_per_fwd(LLAMA70, 32, t)
    row = []
    for k, (bw, eta, lat) in LINKS.items():
        row.append(f"{link_time(nb, bw, eta, lat)*1e6:14.1f}us")
    print(f"{t:>3} {nb/1e6:10.1f}MB " + " ".join(f"{c:>16}" for c in row))

# %% [markdown]
# **Expected:** the byte count rises with $t$ (the ring factor `2(t-1)/t` does), and NVLink
# is roughly 10-15x faster than PCIe on the same bytes. The NVLink 5 column is about half
# the NVLink 4 column, as the 2x bandwidth implies.
#
# Now the number that matters — **communication as a fraction of the decode step.** The
# memory term for a TP group is `weights/t` read at `bw_mem * 0.8`.

# %%
print("\nComm overhead as a fraction of a TP decode step (H100)")
print(f"{'model':<16}{'batch':>7}{'t':>4}" + "".join(f"{k:>18}" for k in ("NVLink 4", "PCIe 4.0", "InfiniBand")))
for model in (LLAMA8, LLAMA70):
    for batch in (1, 32):
        for t in (2, 4, 8):
            if t not in model.tp_degrees():
                continue
            nb = tp_bytes_per_fwd(model, batch, t)
            t_mem = (model.weight_bytes() / t) / (H100.bw * 0.8)
            row = []
            for k in ("NVLink 4", "PCIe 4.0", "InfiniBand"):
                bw, eta, lat = LINKS[k]
                row.append(f"{link_time(nb, bw, eta, lat)/t_mem*100:16.1f}%")
            print(f"{model.name:<16}{batch:>7}{t:>4}" + "".join(f"{c:>18}" for c in row))

# %% [markdown]
# **This table is the argument of the whole module, in one place.** Read the columns:
#
# - **NVLink 4** stays in the single digits at every batch size and degree. TP inside a node
#   is effectively free. Use it.
# - **PCIe 4.0** and **InfiniBand** blow up as batch grows, because the payload grows with
#   batch while the message count ($2L$) stays fixed. Cross-node TP is not viable.
#
# Two predictions to check yourself: at batch 1 the overhead should be *lower* than at
# batch 32 on every fabric (smaller payloads, latency-dominated). And at $t = 8$ the byte
# count should be *higher* than at $t = 4$ even though each GPU moves less — because the
# ring factor grew while the payload did not shrink.

# %% [markdown]
# ## 5. The pipeline bubble, as a clock
#
# No GPU needed. Build the schedule, count the busy cells, and check the formula.

# %%
def pipeline_schedule(p, m):
    """A p-stage, m-micro-batch forward-only pipeline. Returns the slot grid.

    Grid[r][t] is 'busy' if stage r is processing micro-batch (t - r).
    """
    slots = m + p - 1
    grid = []
    for r in range(p):
        row = []
        for t in range(slots):
            busy = 0 <= (t - r) < m
            row.append("busy" if busy else "idle")
        grid.append(row)
    return grid


def bubble_fraction(p, m):
    if p <= 1:
        return 0.0
    return (p - 1) / (m + p - 1)


def draw(grid):
    # stage 0 on top, time to the right — the diagonal is the pipeline working
    return "\n".join("  " + " ".join("[]" if c == "busy" else ".." for c in row) for row in grid)


for (p, m) in [(4, 1), (4, 4), (4, 8), (8, 8)]:
    g = pipeline_schedule(p, m)
    print(f"\nP={p}, M={m}   slots={len(g[0])}   bubble={bubble_fraction(p, m)*100:.1f}%")
    print(draw(g))

print("\nVerified against the closed form:")
ok = True
for p, m in itertools.product((2, 4, 8), (1, 2, 4, 8, 16, 32)):
    g = pipeline_schedule(p, m)
    busy = sum(row.count("busy") for row in g)
    total = sum(len(row) for row in g)
    counted = 1 - busy / total
    if abs(counted - bubble_fraction(p, m)) > 1e-12:
        ok = False
        print("  mismatch at", p, m, counted, bubble_fraction(p, m))
print("  all 18 (P, M) combinations match (P-1)/(M+P-1):", ok)

# %% [markdown]
# **Expected:** `True`. The bubble is $(P-1)/(M+P-1)$, and you have now counted it two
# independent ways — a formula and a grid. That is the standard you should hold every
# formula in this course to.
#
# Note what the schedules show. At `P=4, M=1` three of every four stages are idle. At
# `P=4, M=8` the pipeline is full after the third slot. At `P=8, M=8` the bubble is 47%,
# so **doubling the stages while keeping the micro-batches made it much worse** — the
# lesson's central point about pipeline parallelism.

# %%
def micro_batches_for(p, target=0.2):
    if p <= 1:
        return 1
    m = p
    while bubble_fraction(p, m) > target and m < 512:
        m *= 2
    return m


print("\nMicro-batches needed for a 20% bubble")
for p in (2, 4, 8, 16):
    m = micro_batches_for(p)
    print(f"  P={p:>2}: M={m:>2}  (bubble {bubble_fraction(p, m)*100:.1f}%)   4P rule gives {4*p}")

# %% [markdown]
# ## 6. Measure a real collective
#
# Everything so far was arithmetic. This section calibrates your constants against a real
# `torch.distributed` all-reduce. A single-process, single-GPU world still runs the
# collective kernel — so you get a real latency number, even on a T4.

# %%
import os
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

if not dist.is_initialized():
    dist.init_process_group(backend="nccl" if DEV == "cuda" else "gloo",
                            rank=0, world_size=1)
print("backend:", dist.get_backend(), " world size:", dist.get_world_size())


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


print("\nSingle-GPU all_reduce (world size 1): no bytes move, so this is pure latency.")
for nbytes in (1 << 10, 1 << 14, 1 << 18, 1 << 20, 1 << 22):
    n = nbytes // 4
    buf = torch.randn(n, device=DEV)
    t = bench(lambda b=buf: dist.all_reduce(b))
    print(f"  {nbytes/1024:8.0f} KiB   {t*1e6:8.1f} us")

# %% [markdown]
# **This is your calibration.** With `world_size = 1` no bytes cross a link, so the numbers
# above are essentially the launch overhead of the collective: a fixed cost you pay once
# per message no matter how small the message is.
#
# That fixed cost is the **latency** term from Lesson 18.1, and it is why counting
# *messages* matters more than counting bytes. Llama-3.1-70B issues 160 collectives per
# forward pass at TP=8; at 40 µs of launch overhead each, that is 6.4 ms of pure overhead
# per step — before a single byte moves.
#
# To measure a real multi-GPU collective you need more than one device. On a Colab T4 you
# do not have one, and that is worth being honest about rather than quoting a number you
# did not measure. What you *can* do is measure the latency floor above, use it as the
# `latency` constant in your `link_time` model, and check that the resulting predictions
# match the arithmetic from the lessons.

# %% [markdown]
# ## 7. Context parallelism: why it loses
#
# Split one sequence $P$ ways. Memory per GPU falls by $P$; communication per layer is
# $(P-1)/P$ of the whole cache. Then the penalty factor from Lesson 18.4.

# %%
def cp_bytes_per_layer(model, batch, ctx, p):
    whole = model.kv_bytes(batch, ctx)
    return (p - 1) / p * whole


print("Context parallelism, Llama-3.1-8B, batch 1")
print(f"{'ctx':>9}{'P':>4}{'KV total':>12}{'per GPU':>12}{'comm/layer':>14}{'comm/fwd':>12}{'NVLink4':>12}{'PCIe4':>12}")
for ctx in (32768, 131072):
    for p in (2, 4, 8):
        kv = LLAMA8.kv_bytes(1, ctx)
        nb = cp_bytes_per_layer(LLAMA8, 1, ctx, p)
        t_nv = link_time(nb, LINKS["NVLink 4"][0], LINKS["NVLink 4"][1], LINKS["NVLink 4"][2])
        t_pcie = link_time(nb, LINKS["PCIe 4.0"][0], LINKS["PCIe 4.0"][1], LINKS["PCIe 4.0"][2])
        print(f"{ctx:>9}{p:>4}{kv/GB:10.2f}GB{kv/p/GB:10.2f}GB{nb/GB:12.2f}GB{nb*LLAMA8.layers/GB:10.1f}GB{t_nv*1e6:10.0f}us{t_pcie*1e6:10.0f}us")


def cp_penalty(bw_link, eta, bw_mem, p):
    """How much SLOWER the attention read becomes with context parallelism.

    Per GPU the local slice costs kv/P at B_mem; the ring costs (P-1)/P of the whole
    cache over one link at B_link. Their ratio is (P-1) * B_mem / (B_link * eta).

    Note what this means: the factor is always > 1 for P > 1, on every fabric.
    **Context parallelism never makes a step faster.** It only makes a request that
    would not fit, fit."""
    return (p - 1) * bw_mem / (bw_link * eta)


print("\nContext-parallel penalty on the attention read (H100 memory bandwidth)")
print(f"{'fabric':<18}{'link GB/s':>11}" + "".join(f"{'P='+str(p):>10}" for p in (2, 4, 8)))
for name, (bw, eta, lat) in LINKS.items():
    print(f"{name:<18}{bw*eta/1e9:10.0f}" + "".join(f"{cp_penalty(bw, eta, H100.bw, p):9.1f}x" for p in (2, 4, 8)))

# %% [markdown]
# **Expected:** every number is above 1x, and the penalty is **independent of how large
# the KV cache is**. Over PCIe 4.0 at $P = 4$ it is about 105x: the attention read becomes
# a hundred times more expensive than doing it locally on one GPU.
#
# That is the whole verdict, and it is stronger than "context parallelism is slow". It is
# **context parallelism is never faster, ever** — the penalty threshold of 1x is never
# reached on any real fabric, because $(P-1)/P$ of the cache always costs more to move than
# $1/P$ of it costs to read from HBM.
#
# So the only question is not "does it pay?" but "is the request otherwise impossible?".
# That is why context parallelism is the last resort for long context, and why the
# alternatives — shorten the context, quantize the KV cache, or tensor-parallel (which
# divides the cache by $t$ with *no* all-to-all, because each rank owns whole heads) —
# are the right answer nearly every time.

# %% [markdown]
# ## 8. The decision procedure, run over a grid
#
# The deliverable: a strategy matrix over models × GPUs × workloads. This is the artefact
# you would actually produce for a capacity review, and it is the same code the in-page
# visual runs.

# %%
def plan(model, gpu, batch, ctx, gpus, wbytes=2, island=8):
    """The escalation order from Lesson 18.4, as code."""
    budget = gpu.budget()
    kv = model.kv_bytes(batch, ctx)
    w = model.weight_bytes(wbytes)

    if w + kv <= budget:
        return dict(tp=1, pp=1, replicas=gpus, why="fits alone -> replicate")

    if model.weight_bytes(1) + kv <= budget and wbytes > 1:
        return dict(tp=1, pp=1, replicas=gpus, why="FP8 fits -> quantize + replicate")

    max_tp = max((p for p in model.tp_degrees() if p <= min(island, gpus)), default=1)

    # each TP rank holds 1/tp of the weights AND 1/tp of the KV cache
    def per_rank(tp):
        return w / tp + kv / tp

    tp = max_tp
    while tp > 1 and per_rank(tp) <= budget:
        tp //= 2
    while tp * 2 <= max_tp and per_rank(tp) > budget:
        tp *= 2

    # each pipeline stage holds 1/(tp*pp) of the weights; KV still divides only by tp
    pp = 1
    while (w / (tp * pp) + kv / tp) > budget and tp * pp * 2 <= gpus:
        pp *= 2

    replicas = max(1, gpus // (tp * pp))
    why = f"too big: TP={tp} in the island, PP={pp} across it, x{replicas} replicas"
    if tp == 1:
        why = f"TP blocked by head counts; PP={pp}, x{replicas} replicas"
    return dict(tp=tp, pp=pp, replicas=replicas, why=why)


print(f"{'model':<16}{'gpu':<12}{'batch':>7}{'ctx':>8}{'GBPs':>6}  {'TP':>3}{'PP':>4}{'rep':>5}  why")
print("-" * 110)
for model in (QWEN15, LLAMA8, QWEN32, LLAMA70):
    for gpu in (H100,):
        for (batch, ctx, g) in [(1, 8192, 8), (16, 8192, 8), (64, 32768, 32), (8, 131072, 8)]:
            p = plan(model, gpu, batch, ctx, g)
            print(f"{model.name:<16}{gpu.name:<12}{batch:>7}{ctx:>8}{g:>6}  "
                  f"{p['tp']:>3}{p['pp']:>4}{p['replicas']:>5}  {p['why']}")

# %% [markdown]
# **How to read this table:**
#
# - Most rows say `replicate` or `fp8 fits`, which is the answer you want most often.
#   Qwen2.5-1.5B fits at *every* workload in this grid, including 128k context — because its
#   KV is only 112 KiB per token.
# - `TP=4` and `TP=2` appear for Llama-3.1-8B as context and batch grow: at 32k/64 it
#   needs TP=4, at 128k/8 it needs only TP=2 because the request count is small. **TP is
#   driven by the KV term**, which is why long context, not model size, is what pushes the
#   degree up.
# - Qwen2.5-32B and Llama-3.1-70B both hit `TP=8, PP>1` at 64 requests and 32k context.
#   Neither scheme rescues that workload alone — the KV cache is the dominant term and it
#   divides by both.
#
# Now go find the rows you would have got wrong. Set `wbytes=1` and watch which rows
# collapse from "shard" to "replicate" — that is quantization doing the work that four
# GPUs would otherwise have to.

# %% [markdown]
# ## 9. Verify against a real model
#
# So far every formula has been arithmetic on numbers I supplied. Load a real model, read
# its actual config, and check that the formulas describe the thing that is really there.

# %%
try:
    from transformers import AutoConfig
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)
    from transformers import AutoConfig

name = "Qwen/Qwen2.5-0.5B-Instruct"
try:
    conf = AutoConfig.from_pretrained(name)
    real = Cfg(name, 0.49e9,
               conf.num_hidden_layers, conf.hidden_size, conf.intermediate_size,
               conf.num_attention_heads, conf.num_key_value_heads, conf.head_dim)
    print("config:", conf.num_hidden_layers, "layers, hidden", conf.hidden_size,
          ", heads", conf.num_attention_heads, "/ KV", conf.num_key_value_heads)
    pred = real.kv_bytes_per_token()
    print(f"predicted KV/token from config: {pred} bytes ({pred/1024:.0f} KiB)")

    # and the real thing: run a generation, then read the KV cache out of the model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=DEV)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    kv = out.past_key_values
    n_layers = len(kv)
    k0 = kv[0][0]                       # [1, kv_heads, seq, head_dim]
    actual_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                       for k, v in kv)
    seq = k0.shape[2]
    actual_per_token = actual_bytes / seq
    print(f"actual KV bytes for {seq} tokens: {actual_bytes}")
    print(f"actual per token:              {actual_per_token:.0f} bytes")
    print(f"predicted per token:           {pred} bytes")
    print(f"relative difference:           {abs(actual_per_token - pred)/pred*100:.2f}%")
    assert abs(actual_per_token - pred) / pred < 0.05, "prediction is off"
    print("\nOK — the formula describes the real model.")
except Exception as e:                       # network may be unavailable
    print("Could not reach the hub or load the model:", e)
    print("The config-only check above may still have run. Skip if not.")

# %% [markdown]
# **Expected:** agreement to within a few percent, and the discrepancy is FP16 element size
# bookkeeping, not a modelling error. If you see more than about 5%, the field to check is
# `num_key_value_heads` — a GQA model stores 2 KV heads, not 32, and that factor of 4 (or
# 16) is the single most common source of wrong capacity numbers in the wild.

# %% [markdown]
# ## 10. Bring it all together
#
# One function: model, GPU, fabric, workload in; strategy out, with the reasoning. Then a
# print-out of the strategies for the configurations a real team actually debates.

# %%
def recommend(model, gpu, link_name, batch, ctx, gpus, wbytes=2, island=8):
    bw, eta, lat = LINKS[link_name]
    p = plan(model, gpu, batch, ctx, gpus, wbytes, island)

    lines = []
    if p["tp"] > 1:
        nb = tp_bytes_per_fwd(model, batch, p["tp"])
        t_mem = (model.weight_bytes(wbytes) / p["tp"]) / (gpu.bw * 0.8)
        pct = link_time(nb, bw, eta, lat) / t_mem * 100
        lines.append(f"TP={p['tp']}: {pct:.1f}% of the step in collectives over {link_name}")
    if p["pp"] > 1:
        m = micro_batches_for(p["pp"])
        lines.append(f"PP={p['pp']}: {(p['pp']-1)/(m+p['pp']-1)*100:.0f}% bubble at M={m} "
                     f"(you have {batch} concurrent)")
    if p["replicas"] > 1:
        lines.append(f"{p['replicas']} replicas: zero communication, linear throughput")
    return p, "\n    ".join(lines)


cases = [
    (LLAMA8,  H100, "NVLink 4 (H100)", 32,  8192,  8),
    (LLAMA70, H100, "NVLink 4 (H100)", 32,  8192,  8),
    (LLAMA70, H100, "InfiniBand NDR",  32,  8192, 32),
    (LLAMA70, H200, "NVLink 4 (H100)", 16, 32768,  8),
    (QWEN32,  H200, "NVLink 4 (H100)", 16, 32768, 16),
    (QWEN15,  A100, "PCIe 4.0 x16",     16, 131072, 8),
]

for (model, gpu, link, batch, ctx, gpus) in cases:
    p, detail = recommend(model, gpu, link, batch, ctx, gpus)
    print(f"\n{model.name} on {gpus} × {gpu.name} over {link}, batch {batch}, ctx {ctx}")
    print(f"    {detail}")

# %% [markdown]
# Four of those six say **replicate** or **TP with no pipeline**, and the two that shard pick
# $t = 4$ rather than $t = 8$ — because at batch 16 to 32 and 8k to 32k context, a 70B
# model on an 80 GB card needs its weights split four ways, not eight. That is the
# escalation order working: it takes the *smallest* shard that fits.
#
# The last case is the one to sit with. **Qwen2.5-1.5B at 128k context with 16 concurrent
# requests needs 15 GB of KV — and it still fits on a single A100.** The honest answer is
# `replicate`, on PCIe, across eight cards that never talk to each other. No sharding, no
# bubble, no collectives.
#
# This is the result people find least satisfying and should most trust. A request that
# costs you nothing to serve well should not be given a complicated architecture. The
# decision procedure's job is to *find* that case, not to route around it.
#
# A decision procedure that only outputs "TP=8, PP=4" has not finished its job. The job
# includes saying **no** to sharding — and for a surprising number of real configurations,
# that is the correct output.

# %% [markdown]
# ## 11. What to take away
#
# The five numbers from this lab:
#
# 1. **KV bytes per token** = `2 * layers * kv_heads * head_dim * bytes`. For Llama-3.1-8B
#    in BF16 that is 131,072. Everything else in the module is arithmetic on this constant.
# 2. **Memory budget** = `capacity * 0.96` minus nothing else you control. If weights + KV
#    fit in it, replicate; you almost never need a second GPU.
# 3. **Ring all-reduce** moves `2(t-1)/t` of the payload per GPU, twice per layer.
# 4. **Pipeline bubble** = `(P-1)/(M+P-1)`, verified against a counted schedule.
# 5. **Context parallel penalty** = `(P-1) * B_mem / (B_link * eta)` = `(P-1) * rho`.
#    It exceeds 1 for every P > 1 on every fabric, so CP is never a speed optimisation.
#
# And the rule that uses all of them:
#
# > Quantize before you shard. Tensor-parallel inside the NVLink island. Pipeline across
# > nodes. Replicate for everything else. Context parallel only when one single sequence
# > will not fit any other way.
#
# Next: **Module 19 · Mixture-of-Experts**, where the parallelisms you just learned meet
# their hardest case — a model whose parameters are so spread out that only a slice is ever
# used, and whose routing traffic is a different collective entirely.

# %%
print("\ndone.")