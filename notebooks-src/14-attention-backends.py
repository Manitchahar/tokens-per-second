# %% [markdown]
# # Lab 14 · Compare attention implementations
#
# **Tokens per Second** · Module 14
#
# You will:
# 1. Verify the online softmax against a two-pass reference, on the CPU, before optimising anything.
# 2. Watch eager attention's peak memory grow quadratically with context.
# 3. Write a fused FlashAttention kernel in Triton, **verify it**, then time it.
# 4. Sweep the tile size and explain the result from register pressure.
# 5. Confirm decode is bandwidth-bound, with intensity equal to the GQA group factor.
# 6. Implement split-K decode and find the batch size where it stops helping.
# 7. Write a paged attention kernel, verify it, and price its gather.
# 8. Check a real model's KV bytes per token against your hand calculation.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# Rule of the whole lab: **implement, verify, then time.** In that order.

# %%
import math, time
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
DEV = "cuda"
DT = torch.float16
print(torch.cuda.get_device_name(0))
print("torch", torch.__version__)

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
    print("triton", triton.__version__)
except ImportError:
    HAVE_TRITON = False
    print("Triton not available — sections 3 and 4 will be skipped.")

torch.manual_seed(0)


def bench(fn, iters=20, warmup=5):
    """Time a GPU function. Always warm up, always synchronise, always average."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# %% [markdown]
# ## 1. The online softmax, verified on the CPU
#
# Everything in this lab is a faster way of running one update. Get it right first, on the CPU, where
# mistakes are obvious and the answer is exact.
#
# The update, in words:
# - `m` is the largest score seen so far.
# - `l` is the sum of `exp(s - m)` over what has been seen.
# - `O` is the sum of `exp(s - m) * v` over what has been seen.
#
# When a new tile arrives with a larger maximum, rescale the old state to match. That is the whole trick.

# %%
def online_update(m, l, acc, s, v):
    """Fold one tile of scores s [Br, Bc] and values v [Bc, d] into the running state."""
    m_new = torch.maximum(m, s.max(dim=-1).values)   # new maximum
    corr = torch.exp(m - m_new)                      # how much to rescale the old state
    p = torch.exp(s - m_new.unsqueeze(-1))           # tile probabilities, in [0, 1]
    l = l * corr + p.sum(dim=-1)                     # running sum
    acc = acc * corr.unsqueeze(-1) + p @ v           # running output
    return m_new, l, acc


def attention_online(q, k, v, tile):
    """Exact attention, computed in tiles. q,k,v are [1, H, T, D]."""
    H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    m = torch.full((H, T), -float("inf"), dtype=torch.float32)   # NOTE: -inf, not 0
    l = torch.zeros(H, T, dtype=torch.float32)
    acc = torch.zeros(H, T, D, dtype=torch.float32)
    for j in range(0, T, tile):
        j2 = min(j + tile, T)
        s = (q[0, :, j:j2].float() @ k[0, :, j:j2].float().transpose(-1, -2)) * scale
        # causal mask inside the tile: query i (absolute j+i) may see key j2'
        rows = torch.arange(j, j2).unsqueeze(-1)
        cols = torch.arange(j, j2).unsqueeze(0)
        s = s.masked_fill(cols > rows, -float("inf"))
        m, l, acc = online_update(m, l, acc, s, v[0, :, j:j2].float())
    return (acc / l.unsqueeze(-1)).unsqueeze(0)


def attention_reference(q, k, v):
    """Plain two-pass attention. Slow, obvious, correct."""
    D = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(D)
    T = q.shape[-2]
    causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    s = s.masked_fill(causal, -float("inf"))
    return (torch.softmax(s, dim=-1) @ v.float())


for tile in (4, 8, 16, 32):
    q = torch.randn(1, 4, 64, 32)
    k = torch.randn(1, 4, 64, 32)
    v = torch.randn(1, 4, 64, 32)
    a = attention_online(q, k, v, tile)
    b = attention_reference(q, k, v)
    print(f"tile={tile:3d}  max abs diff = {(a - b).abs().max().item():.3e}   rows sum to 1: {torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)}")

# %% [markdown]
# **Expected:** differences at float32 rounding level (around 1e-6), for *every* tile size, and row sums of 1.
#
# Two things to try before moving on, because they are the two real bugs:
#
# 1. Change `m = torch.full((H, T), -float("inf"))` to `m = torch.zeros(H, T)` and re-run.
#    With small random scores it will *still pass*, because many of the maxima are positive.
#    Then try `q = torch.randn(...) - 3` so scores are mostly negative, and watch it break.
#    This is why "it passed the test" and "it is correct" are different statements.
#
# 2. Remove the causal mask. The result is still finite and still looks like attention — it just
#    lets every token see the future. Nothing crashes. This is the most dangerous bug in this lab.

# %%
# Demonstrate bug (1) concretely, so the failure is in front of you.
def online_buggy(q, k, v, tile):
    H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    m = torch.zeros(H, T, dtype=torch.float32)      # <-- the bug: should be -inf
    l = torch.zeros(H, T, dtype=torch.float32)
    acc = torch.zeros(H, T, D, dtype=torch.float32)
    for j in range(0, T, tile):
        j2 = min(j + tile, T)
        s = (q[0, :, j:j2].float() @ k[0, :, j:j2].float().transpose(-1, -2)) * scale
        rows = torch.arange(j, j2).unsqueeze(-1)
        cols = torch.arange(j, j2).unsqueeze(0)
        s = s.masked_fill(cols > rows, -float("inf"))
        m, l, acc = online_update(m, l, acc, s, v[0, :, j:j2].float())
    return (acc / l.unsqueeze(-1)).unsqueeze(0)


q = torch.randn(1, 4, 64, 32) - 3.0   # negative scores
print("negative scores, m=0   :", f"{(online_buggy(q,q,q,8) - attention_reference(q,q,q)).abs().max().item():.3e}")
print("negative scores, m=-inf:", f"{(attention_online(q,q,q,8) - attention_reference(q,q,q)).abs().max().item():.3e}")

# %% [markdown]
# ## 2. Eager attention's memory, growing quadratically
#
# Now watch the problem the module is about. We run plain `softmax(QKᵀ)V` at one layer's shape
# and track peak GPU memory as context grows.

# %%
# One layer of Llama-3.1-8B. We use a smaller head count so a T4 survives, but keep the
# T^2 structure — the shape is what matters here, not the model.
H, D = 8, 128

print(f"{'T':>7} {'peak MB':>10} {'ratio vs prev':>15} {'S+P MB (predicted)':>20}")
prev = None
for T in (512, 1024, 2048, 4096):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q = torch.randn(1, H, T, D, device=DEV, dtype=DT)
    k = torch.randn(1, H, T, D, device=DEV, dtype=DT)
    v = torch.randn(1, H, T, D, device=DEV, dtype=DT)
    base = torch.cuda.memory_allocated()

    s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    torch.cuda.synchronize()

    peak = (torch.cuda.max_memory_allocated() - base) / 1e6
    predicted = 2 * H * T * T * 2 / 1e6        # S and P, H heads, T x T, 2 bytes
    ratio = peak / prev if prev else float("nan")
    print(f"{T:>7} {peak:>10.1f} {ratio:>15.2f} {predicted:>20.1f}")
    prev = peak
    del q, k, v, s, p, o

# %% [markdown]
# **What to expect:** doubling `T` roughly **quadruples** peak memory. That is the $T^2$ from the
# score matrix — and the measured value tracks the closed form `2 · H · T² · 2 bytes` closely.
#
# Note the prediction column: you can compute this on paper, for any model, before running anything.
# If your measurement disagrees with the prediction by much, something other than attention is
# holding memory (usually the PyTorch caching allocator keeping earlier blocks).

# %%
# The closed form, for Llama-3.1-8B at one layer, 32 heads.
def eager_peak_bytes(T, heads=32, b=2):
    return 2 * heads * T * T * b      # S and P both resident


for T in (1024, 4096, 8192, 16384):
    print(f"T={T:>6}: eager peak for ONE layer = {eager_peak_bytes(T)/1e9:>7.2f} GB")

print("\nA T4 has 16 GB. Work out where eager attention becomes impossible for this layer shape.")

# %% [markdown]
# ## 3. A fused FlashAttention kernel in Triton
#
# The kernel from the lesson, as close to real as we can get. Structure to notice:
# - The query tile is loaded once and kept for the whole loop.
# - The score tile `qk` is **never stored**. It is computed, consumed, and replaced.
# - `m`, `l` and `acc` are FP32 even though Q, K, V are FP16.
# - The division by `l` happens once, after the loop.

# %%
if HAVE_TRITON:

    @triton.jit
    def flash_fwd(Q, K, V, O,
                  stride_qm, stride_kn, stride_vn, stride_om,
                  T, BR: tl.constexpr, BC: tl.constexpr, D: tl.constexpr,
                  SCALE: tl.constexpr):
        start_m = tl.program_id(0)
        offs_m = start_m * BR + tl.arange(0, BR)
        offs_d = tl.arange(0, D)
        offs_br = tl.arange(0, BR)
        offs_bc = tl.arange(0, BC)

        q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :])

        m_i = tl.full([BR], -float("inf"), tl.float32)     # -inf, always
        l_i = tl.zeros([BR], tl.float32)
        acc = tl.zeros([BR, D], tl.float32)                 # FP32 accumulator

        for start_n in range(0, T, BC):
            offs_n = start_n + offs_bc
            k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :])
            v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :])

            qk = tl.dot(q, tl.trans(k)) * SCALE
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))   # causal

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(O + offs_m[:, None] * stride_om + offs_d[None, :], acc)

    def flash_attention(q, k, v, BR=64, BC=64):
        """q,k,v: [B, H, T, D] contiguous. Returns the same shape."""
        B, H, T, D = q.shape
        o = torch.empty_like(q)
        grid = (triton.cdiv(T, BR), B * H)
        flash_fwd[grid](
            q, k, v, o,
            q.stride(-2), k.stride(-2), v.stride(-2), o.stride(-2),
            T, BR=BR, BC=BC, D=D, SCALE=1.0 / math.sqrt(D),
        )
        return o

    print("kernel written")

# %% [markdown]
# ### Verify it, *then* time it
#
# The order matters more than it looks like a formality. A speed number for an incorrect kernel
# is a number you will confidently report later.

# %%
if HAVE_TRITON:
    for T in (256, 512, 1024):
        B, Hh, Dd = 1, 4, 64
        q = torch.randn(B, Hh, T, Dd, device=DEV, dtype=DT)
        k = torch.randn(B, Hh, T, Dd, device=DEV, dtype=DT)
        v = torch.randn(B, Hh, T, Dd, device=DEV, dtype=DT)

        out = flash_attention(q, k, v)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        diff = (out.float() - ref.float()).abs()
        tol = 2e-2
        print(f"T={T:>5}  max abs diff = {diff.max().item():.4f}  "
              f"mean = {diff.mean().item():.5f}  "
              f"allclose(tol={tol}) = {torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol)}")

# %% [markdown]
# **Expected:** differences around 1e-3 to 1e-2 — that is FP16 rounding (three decimal digits), not
# algorithmic error — and `allclose` True at a sensible tolerance.
#
# If it disagrees badly, check in this order:
# 1. Is `SCALE` actually `1/sqrt(D)`? Missing it is the single most common mistake.
# 2. Is the causal mask `<=` the right way round?
# 3. Is `m_i` initialised to `-inf`?
# 4. Is `acc` FP32? Rounding there accumulates over tiles and looks like a quality regression.

# %%
if HAVE_TRITON:
    # Show what a missing scale actually looks like: plausible, not obviously broken.
    B, Hh, T, Dd = 1, 2, 512, 64
    q = torch.randn(B, Hh, T, Dd, device=DEV, dtype=DT)
    k = torch.randn(B, Hh, T, Dd, device=DEV, dtype=DT)
    v = torch.randn(B, Hh, T, Dd, device=DEV, dtype=DT)

    good = flash_attention(q, k, v).float()
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).float()
    print("relative error with scale   :", ((good - ref).norm() / ref.norm()).item())
    print("softmax(QK^T)V without scale:", (F.softmax(q @ k.transpose(-1, -2), dim=-1) @ v).float().sub(ref).norm().div(ref.norm()).item())

# %% [markdown]
# ## 4. Sweep the tile size, and time against eager
#
# Two questions: how much does fusion actually buy, and does the tile size matter?

# %%
if HAVE_TRITON:
    B, Hh, Dd, T = 1, 8, 64, 4096
    q = torch.randn(B, Hh, T, Dd, device=DEV, dtype=DT)
    k = torch.randn(B, Hh, T, Dd, device=DEV, dtype=DT)
    v = torch.randn(B, Hh, T, Dd, device=DEV, dtype=DT)

    def eager(q, k, v):
        s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        return torch.softmax(s, dim=-1) @ v

    t_eager = bench(lambda: eager(q, k, v))
    t_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    print(f"{'impl':>18} {'ms':>9}")
    print(f"{'eager':>18} {t_eager*1e3:>9.2f}")
    print(f"{'torch SDPA':>18} {t_sdpa*1e3:>9.2f}")

    print(f"\n{'BR':>5} {'BC':>5} {'ms':>9} {'vs eager':>10}")
    for BR, BC in ((32, 64), (64, 64), (128, 64), (64, 128), (128, 128)):
        try:
            t = bench(lambda: flash_attention(q, k, v, BR=BR, BC=BC), iters=10)
            print(f"{BR:>5} {BC:>5} {t*1e3:>9.2f} {t_eager/t:>9.1f}x")
        except Exception as e:
            print(f"{BR:>5} {BC:>5}   failed: {str(e)[:60]}")

# %% [markdown]
# **What to expect.** Your fused kernel beats eager by a wide margin at 4,096 tokens, and the best
# tile size is not the largest — `BR=128` needs a bigger `acc` register block (`128 x D`) and can
# lose occupancy. The precise winner depends on the GPU; on a T4, moderate tiles usually win.
#
# Notice also that `torch`'s own `scaled_dot_product_attention` is fast, because it dispatches to a
# fused kernel. That is the lesson's point in production form: **the fused algorithm is the
# default**, and your job is to know why rather than to hand-write it every time.

# %% [markdown]
# ## 5. Decode is bandwidth-bound — confirm it
#
# Now switch shape. Prefill has `T` query rows; decode has **one**. That is the whole difference.
#
# The prediction to test: decode's arithmetic intensity is the GQA group factor, a handful of FLOP
# per byte, so a decode-shaped call should show near-peak GB/s and near-zero TFLOP/s.

# %%
NQ, NKV, Dd, T = 32, 8, 128, 8192      # Llama-3.1-8B: 32 query heads, 8 KV heads

q = torch.randn(1, NQ, T, Dd, device=DEV, dtype=DT)
k = torch.randn(1, NKV, T, Dd, device=DEV, dtype=DT)
v = torch.randn(1, NKV, T, Dd, device=DEV, dtype=DT)

def report(name, fn, flops, bytes_moved, iters=20):
    t = bench(fn, iters=iters)
    gbs = bytes_moved / t / 1e9
    tfs = flops / t / 1e12
    print(f"{name:>22}: {t*1e3:>7.2f} ms   {gbs:>7.1f} GB/s   {tfs:>6.2f} TFLOP/s   "
          f"intensity {flops/bytes_moved:>6.1f} FLOP/B")

# Prefill-shaped: T query rows.
flops_pre = 2 * NQ * Dd * T * T * 2 / 2          # causal: about half
bytes_pre = 3 * (NQ + 2 * NKV) * T * Dd * 2
report("prefill (T x T)", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), flops_pre, bytes_pre)

# Decode-shaped: ONE query row against the whole cache.
qd = torch.randn(1, NQ, 1, Dd, device=DEV, dtype=DT)
flops_dec = 4 * NQ * Dd * T
bytes_dec = 2 * NKV * Dd * 2 * T
report("decode (1 x T)", lambda: F.scaled_dot_product_attention(qd, k, v, is_causal=False), flops_dec, bytes_dec)

print(f"\nGQA group factor = {NQ}/{NKV} = {NQ/NKV}")
print(f"Predicted decode intensity = {flops_dec/bytes_dec:.1f} FLOP/byte  (should equal the group factor)")
print(f"T4 peak bandwidth is about 320 GB/s. Decode should be near it.")

# %% [markdown]
# **Expected:** prefill shows high TFLOP/s (compute-bound at this length); decode shows GB/s near
# the T4's ~320 GB/s with TFLOP/s near zero, and an intensity equal to the GQA group factor.
#
# **If decode shows high TFLOP/s, you have a bug** — most likely the call is not actually
# decode-shaped. Check the query really is `[1, NQ, 1, D]` and that you are not secretly batching
# 8,192 query rows.
#
# Note what did *not* change between the two: the kernel. The same `scaled_dot_product_attention`
# serves both. The difference is the shape you hand it, which is why "which phase are we in"
# is a question your profiler should always answer before you optimise anything.

# %% [markdown]
# ## 6. Split-K decode, and its crossover
#
# One query row cannot fill an SM, so split the sequence, reduce each part to `(m, l, O)`, and
# merge with the associative rule. Then find the batch size where it stops helping — that number
# is the most useful thing in this section.

# %%
def decode_splitk(q, k, v, splits=4):
    """One query row against the cache, processed in `splits` parallel chunks."""
    NQ, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    T = k.shape[-2]
    chunk = (T + splits - 1) // splits

    ms, ls, os_ = [], [], []
    for j in range(0, T, chunk):
        kk = k[:, :, j:j+chunk].float()
        vv = v[:, :, j:j+chunk].float()
        s = (q.float() @ kk.transpose(-1, -2)) * scale          # [NQ, 1, chunk]
        m = s.max(dim=-1, keepdim=True).values
        p = torch.exp(s - m)
        ms.append(m)
        ls.append(p.sum(dim=-1))
        os_.append(p @ vv)

    # merge the partials — the associative online-softmax rule
    m_all = torch.cat(ms, dim=-1).max(dim=-1, keepdim=True).values
    w = torch.cat(ms, dim=-1).exp() - m_all                     # exp(m_j - m), [NQ,1,splits]
    l_all = (torch.cat(ls, dim=-1) * w).sum(dim=-1)
    o_all = (torch.cat(os_, dim=-2) * w.unsqueeze(-1)).sum(dim=-2)
    return o_all / l_all.unsqueeze(-1)


def decode_plain(q, k, v):
    NQ, _, D = q.shape
    scale = 1.0 / math.sqrt(D)
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    return torch.softmax(s, dim=-1) @ v.float()


T = 4096
q1 = torch.randn(1, NQ, 1, Dd, device=DEV, dtype=DT)
k1 = torch.randn(1, NKV, T, Dd, device=DEV, dtype=DT)
v1 = torch.randn(1, NKV, T, Dd, device=DEV, dtype=DT)

# correctness first
out_sk = decode_splitk(q1, k1, v1, splits=8)
out_pl = decode_plain(q1, k1, v1)
print("split-K vs plain, max abs diff:", f"{(out_sk - out_pl).abs().max().item():.3e}")
print("(float32, so this should be at rounding level)\n")

# now the sweep: does splitting help as the batch grows?
print(f"{'batch':>6} {'plain ms':>10} " + " ".join(f"{'sk'+str(n):>9}" for n in (2, 4, 8)))
for B in (1, 8, 32, 128):
    qb = torch.randn(B, NQ, 1, Dd, device=DEV, dtype=DT)
    kb = torch.randn(B, NKV, T, Dd, device=DEV, dtype=DT)
    vb = torch.randn(B, NKV, T, Dd, device=DEV, dtype=DT)
    t0 = bench(lambda: decode_plain(qb, kb, vb), iters=10)
    row = []
    for n in (2, 4, 8):
        t = bench(lambda: decode_splitk(qb, kb, vb, splits=n), iters=10)
        row.append(f"{t*1e3:>9.2f}")
    print(f"{B:>6} {t0*1e3:>10.2f} " + " ".join(row))
    del qb, kb, vb

# %% [markdown]
# **Expected:** the split matches plain to float32 rounding, and splitting helps most at batch 1,
# helping less as the batch grows and eventually not at all. At large batch the GPU already has
# enough parallel work from the batch itself, so splitting only adds the combine pass.
#
# In a real kernel the picture is cleaner than in PyTorch (where each "split" is still launched
# sequentially here), but the *crossover* is a real property of the GPU: it is the batch size at
# which the machine stops being under-occupied. Find it for your GPU and workload, and write it down.

# %% [markdown]
# ## 7. A paged attention kernel, verified and priced
#
# Real engines do not store KV contiguously (Module 08). A paged kernel walks a block table and
# gathers. Two things to establish: that it is **correct**, and what the gather **costs**.

# %%
BLOCK = 16
n_blocks = 512
n_seq, ctx = 4, 1024
n_kv, dh = 8, 128

k_cache = torch.randn(n_blocks, BLOCK, n_kv, dh, device=DEV, dtype=DT)
v_cache = torch.randn(n_blocks, BLOCK, n_kv, dh, device=DEV, dtype=DT)

# scatter one contiguous sequence into blocks, at scattered ids
seqs_k, seqs_v, tables, valid_slots = [], [], [], []
for i in range(n_seq):
    tk = torch.randn(ctx, n_kv, dh, device=DEV, dtype=DT)
    tv = torch.randn(ctx, n_kv, dh, device=DEV, dtype=DT)
    n_b = (ctx + BLOCK - 1) // BLOCK
    ids = torch.randperm(n_blocks, device=DEV)[:n_b].tolist()   # scattered, like a real pool
    for b, blk in enumerate(ids):
        sl = min(BLOCK, ctx - b * BLOCK)
        k_cache[blk, :sl] = tk[b*BLOCK:(b+1)*BLOCK]
        v_cache[blk, :sl] = tv[b*BLOCK:(b+1)*BLOCK]
    seqs_k.append(tk); seqs_v.append(tv); tables.append(ids); valid_slots.append(ctx)

# correctness: gather via the table, compare to the contiguous original
maxdiff = 0.0
for i in range(n_seq):
    n_b = len(tables[i])
    gk = torch.cat([k_cache[b, :min(BLOCK, ctx - j*BLOCK)] for j, b in enumerate(tables[i])], dim=0)
    gv = torch.cat([v_cache[b, :min(BLOCK, ctx - j*BLOCK)] for j, b in enumerate(tables[i])], dim=0)
    maxdiff = max(maxdiff, (gk - seqs_k[i]).abs().max().item())
    maxdiff = max(maxdiff, (gv - seqs_v[i]).abs().max().item())
print(f"gather round-trip max abs diff: {maxdiff:.2e}  (FP16 exact copy: 0 is expected)")

# cost of the gather
kv_bytes = 2 * n_kv * dh * 2 * ctx * n_seq
t_gather = bench(lambda: torch.cat([k_cache[b] for b in tables[0]], dim=0), iters=10)
print(f"\ngathered {kv_bytes/1e6:.2f} MB in {t_gather*1e6:.0f} µs -> {kv_bytes/t_gather/1e9:.0f} GB/s")
print("That is the price of paging in one number. As context grows, it approaches")
print("the full step time and the relative overhead falls toward ~1%.")

# %% [markdown]
# **Expected:** the round-trip difference is 0 (it is a copy, not arithmetic), and the gather
# bandwidth is a fraction of the card's peak because the reads are scattered.
#
# This is the honest version of "PagedAttention costs 1%": at short context the gather is
# latency-bound and the percentage looks worse; at long context it is bandwidth-bound and the
# percentage approaches the true figure. Both measurements are correct. Quote the one that matches
# your workload.

# %%
# the same gather, measured at several context lengths
print(f"{'ctx':>7} {'MB gathered':>12} {'GB/s':>9} {'as % of 320 GB/s':>18}")
for ctx2 in (256, 1024, 4096, 16384):
    n_b = (ctx2 + BLOCK - 1) // BLOCK
    if n_b > n_blocks:
        continue
    ids = torch.randperm(n_blocks, device=DEV)[:n_b].tolist()
    nb = 2 * n_kv * dh * 2 * ctx2
    t = bench(lambda: torch.cat([k_cache[b] for b in ids], dim=0), iters=10)
    print(f"{ctx2:>7} {nb/1e6:>12.2f} {nb/t/1e9:>9.0f} {nb/t/320e9*100:>17.1f}%")

# %% [markdown]
# ## 8. A real model: check KV bytes per token
#
# The constant behind every capacity decision in Module 05 and Module 08, verified against a
# model that actually exists.

# %%
try:
    from transformers import AutoConfig
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg = AutoConfig.from_pretrained(name)
    L, nkv, dh = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    kv_per_token = 2 * L * nkv * dh * 2
    print(f"{name}")
    print(f"  layers {L}, kv heads {nkv}, head_dim {dh}")
    print(f"  KV bytes per token = 2 x {L} x {nkv} x {dh} x 2 = {kv_per_token:,} bytes ({kv_per_token/1024:.0f} KiB)")
    print()
    print("Compare with Llama-3.1-8B: 32 x 8 x 128 x 2 x 2 = 131,072 bytes.")
    print("The difference is layers x kv heads x head_dim — everything else cancels.")
except Exception as e:
    print("transformers not available or model unreachable:", str(e)[:120])
    print("The formula is what matters: bytes/token = 2 x layers x kv_heads x head_dim x bytes_per_number")

# %% [markdown]
# ## 9. When it goes wrong
#
# Four ways a fused attention kernel produces wrong answers instead of crashing. Each of these
# passes a smoke test.
#
# **Forgotten scale.** Dropping `1/sqrt(D)`. The output is finite and plausible; quality is just
# quietly worse. Symptom: "the model got dumber after optimisation".
#
# **Missing causal mask.** In decode (one query row of length $T$) the mask looks like a no-op,
# because the query is the last token anyway. Remove it and nothing crashes — but in prefill and
# in training you have silently broken causality.
#
# **Running max initialised to 0.** Covered in section 1. Passes when scores are mostly positive,
# fails when they are not.
#
# **FP16 accumulator.** Covered in section 3. Error grows with context length and reads as a model
# quality regression rather than a kernel bug. Fix: `acc`, `m` and `l` in FP32, always.
#
# ## Your turn
# 1. In section 2, extend the sweep to `T = 8192` if the T4 survives. Does the measured peak still
#    track `2 x H x T² x 2` bytes? If it stops tracking, find out what else is holding memory.
# 2. In section 4, sweep `D` as well as `BR`. Why does head dimension change the register budget
#    but not the FLOP count per query row?
# 3. In section 5, compute the intensity of decode for a model with MQA (1 KV head instead of 8).
#    What changes, and what does it mean for concurrency?
# 4. Write down your GPU's split-K crossover batch size from section 6. It is a property of your
#    card, and it is worth knowing before you tune anything else.

# %%
print("done")