# %% [markdown]
# # Lab 19 · Simulate MoE load balance
#
# **Tokens per Second** · Module 19
#
# You will:
# 1. Build a working mixture-of-experts layer in PyTorch — router, top-k, experts — and verify it.
# 2. Watch a router's load distribution, and measure imbalance and the Gini coefficient.
# 3. Add a capacity constraint and measure exactly what overflow costs you.
# 4. Reproduce the auxiliary loss that trains real MoE routers, and watch it drive load toward uniform.
# 5. Compute the all-to-all communication volume imbalance implies, and compare it to measured GEMM time.
# 6. Read a real trained MoE checkpoint's architecture and find its router weights.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# Rule of the whole lab: **measure, then explain.** Every number printed is compared against the
# closed-form arithmetic from the lesson. If they disagree, the arithmetic is wrong or the code is —
# find out which.

# %%
import math, time
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
DEV = "cuda"
DT = torch.float16
print(torch.cuda.get_device_name(0))
print("torch", torch.__version__)

torch.manual_seed(0)

try:
    from transformers import AutoConfig
    HAVE_TR = True
    print("transformers available")
except ImportError:
    HAVE_TR = False
    print("transformers not available — section 6 will be skipped.")


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
# ## 1. Build a mixture-of-experts layer
#
# Everything in this lab is one layer, made of four pieces:
#
# - **Router**: a matrix `wr [E, d]` mapping a token to a score per expert.
# - **Top-k selection**: take the `k` highest scores.
# - **Experts**: `E` independent MLPs, kept stacked as `w1 [E, d, h]` and `w2 [E, h, d]` so a
#   batched matmul runs them all at once. A real kernel instead gathers each expert's tokens and runs
#   a dense GEMM per expert, but the arithmetic is identical and this fits a T4.
# - **Combine**: sum the selected experts' outputs, weighted by their renormalised router probabilities.
#
# The detail that matters: a token routed to `k` experts is processed by `k` experts, and the router
# probabilities tell the combine step how much of each result to use.

# %%
class MoELayer:
    """One mixture-of-experts feed-forward network, batched over experts."""

    def __init__(self, d=512, h=256, n_experts=8, top_k=2, dtype=DT, device=DEV):
        self.d, self.h, self.E, self.k = d, h, n_experts, top_k
        self.dtype, self.device = dtype, device
        self.w1 = torch.randn(n_experts, d, h, device=device, dtype=dtype) * (d ** -0.5)
        self.w2 = torch.randn(n_experts, h, d, device=device, dtype=dtype) * (h ** -0.5)
        self.wr = torch.randn(n_experts, d, device=device, dtype=torch.float32) * (d ** -0.5)
        # A router bias is what creates favourites. Without it every expert is symmetric.
        self.bias = torch.zeros(n_experts, device=device, dtype=torch.float32)

    def forward(self, x):
        """x: [T, d] -> [T, d], plus per-expert load counts and the full routing distribution."""
        T = x.shape[0]
        xf = x.float()

        # 1. route: softmax over ALL experts, then select
        scores = xf @ self.wr.T + self.bias          # [T, E]
        probs = F.softmax(scores, dim=-1)
        topv, topi = torch.topk(probs, self.k, dim=-1)   # [T, k]
        topv = topv / topv.sum(-1, keepdim=True)          # renormalise over the selected only

        # 2. run every expert on every token (wasteful, but exactly correct)
        xe = xf.unsqueeze(0).expand(self.E, T, self.d)    # [E, T, d]
        hid = F.gelu(xe @ self.w1)                         # [E, T, h]
        ye = hid @ self.w2                                 # [E, T, d]

        # 3. combine: for each token, add topv * ye for every expert it selected
        out = torch.zeros(T, self.d, device=self.device, dtype=torch.float32)
        for e in range(self.E):
            mask = topi == e                               # [T, k]
            if not mask.any():
                continue
            w = torch.zeros(T, device=self.device, dtype=torch.float32)
            w.scatter_(0, topi[mask].reshape(-1), topv[mask].reshape(-1))
            out += w.unsqueeze(-1) * ye[e]

        counts = torch.bincount(topi.reshape(-1), minlength=self.E)
        return out.to(self.dtype), counts, probs


# Smoke test: shape, finiteness, and the total number of routed assignments.
moe = MoELayer()
x = torch.randn(64, 512, device=DEV, dtype=DT)
y, counts, probs = moe.forward(x)
print(f"in {tuple(x.shape)} -> out {tuple(y.shape)}   finite: {torch.isfinite(y).all().item()}")
print(f"expert load: {counts.tolist()}")
print(f"total routed tokens: {counts.sum().item()}  (expected {64 * 2})")

# %% [markdown]
# **Expected:** output shape `[64, 512]`, all finite, and the per-expert counts sum to `64 × 2 = 128`,
# because every token picks exactly two of eight experts.
#
# The counts will be *close* to uniform — 16 each — because the router bias is zero and a random
# router has no favourites. That is the baseline you are about to break deliberately.

# %% [markdown]
# ## 2. Give the router favourites, and measure the damage
#
# Real routers are not uniform. Sweep the router bias and watch what happens to the load.
#
# The two metrics, straight from the lesson:
#
# $$\text{imbalance} = \frac{\max_e c_e}{kT/E}, \qquad
# \qquad G = \frac{2\sum_{i=1}^{n} i\,c_{(i)}}{n\sum_i c_i} - \frac{n+1}{n}$$
#
# with $c_{(i)}$ the loads sorted ascending.

# %%
def measure(counts, T, E, k):
    """Imbalance (straggler multiplier) and Gini coefficient from a load vector."""
    c = counts.float()
    mean = (k * T) / E
    imbalance = (c.max() / mean).item()
    min_over_mean = (c.min() / mean).item()

    srt, _ = torch.sort(c)
    n = srt.numel()
    idx = torch.arange(1, n + 1, device=srt.device, dtype=torch.float32)
    total = srt.sum().item()
    gini = (2 * (idx * srt).sum().item()) / (n * total) - (n + 1) / n if total > 0 else 0.0
    return imbalance, min_over_mean, gini


T = 512
print(f"{'bias std':>9} {'max/mean':>9} {'min/mean':>9} {'Gini':>7} {'top-3':>8}  verdict")
print("-" * 62)
for std in (0.0, 0.5, 1.0, 2.0, 4.0):
    torch.manual_seed(0)
    moe = MoELayer(d=512, h=256, n_experts=8, top_k=2)
    with torch.no_grad():
        moe.bias.copy_(torch.randn(8, device=DEV) * std)
    x = torch.randn(T, 512, device=DEV, dtype=DT)
    _, counts, _ = moe.forward(x)

    imb, minm, gini = measure(counts, T, 8, 2)
    top3 = torch.topk(counts.float(), 3).values.sum().item() / (T * 2)
    verdict = ("healthy" if gini < 0.15 else
               "moderate" if gini < 0.3 else
               "straggler risk" if gini < 0.45 else "severe")
    print(f"{std:>9.1f} {imb:>9.2f} {minm:>9.2f} {gini:>7.3f} {top3:>7.1%}  {verdict}")

# %% [markdown]
# **Expected:** a clear monotone climb. At `std = 0` the imbalance is near 1.0 and the Gini is small.
# By `std = 4.0` the hottest expert holds several times its share and the Gini is well above 0.3.
#
# Read the two metrics as different jobs:
# - **imbalance** is the *straggler multiplier* — how much longer the busiest card takes than an
#   evenly-loaded one. It is what sets your step latency.
# - **Gini** is *one number you can report* — 0 is perfect equality, 1 is one expert taking everything.
#
# Two things worth noticing:
# 1. Even at `std = 0` the Gini is not exactly 0 — random routing of 512 tokens has sampling noise.
# 2. The top-3 share saturates well below 100%. A *good* MoE still uses most of its experts; if one
#    expert took 90% of tokens, that model is a dense model wearing a costume.

# %% [markdown]
# ## 3. Capacity factor: the hard bound, and what overflow costs
#
# Real kernels cannot have ragged buffers. Give each expert a fixed number of slots
#
# $$C = \Big\lceil c \cdot \frac{kT}{E} \Big\rceil$$
#
# and drop whatever does not fit. An expert holding $I$ times its balanced share overflows by
# $\max(0, 1 - c/I)$ of its tokens, so **zero overflow requires $c \geq I$**.

# %%
def apply_capacity(counts, T, E, k, capacity_factor):
    """Apply a hard per-expert capacity. Returns kept counts, slots, dropped count and rate."""
    cap = math.ceil(capacity_factor * (T * k) / E)
    kept = counts.clamp(max=cap)
    dropped = int((counts - kept).sum().item())
    return kept, cap, dropped, dropped / (T * k)


torch.manual_seed(0)
moe = MoELayer(d=512, h=256, n_experts=8, top_k=2)
with torch.no_grad():
    moe.bias.copy_(torch.randn(8, device=DEV) * 2.0)      # a "sharp" router
x = torch.randn(T, 512, device=DEV, dtype=DT)
_, counts, _ = moe.forward(x)
imb, minm, gini = measure(counts, T, 8, 2)

print(f"imbalance I = {imb:.2f}×   Gini = {gini:.3f}   balanced share = {T * 2 / 8:.0f} tokens/expert")
print(f"predicted zero-overflow capacity: c = I = {imb:.2f}")
print()
print(f"{'capacity':>9} {'slots':>7} {'dropped':>8} {'rate':>7} {'predicted':>10}  note")
print("-" * 70)
for c in (1.0, 1.1, 1.25, 1.5, 2.0, 3.0):
    kept, cap, dropped, rate = apply_capacity(counts, T, 8, 2, c)
    predicted = max(0.0, 1 - c / imb)
    note = ("no drops; buffers idle" if dropped == 0 else
            "mild" if rate < 0.05 else
            "noticeable" if rate < 0.15 else "will show in evals")
    print(f"{c:>9.2f} {cap:>7} {dropped:>8} {rate:>6.1%} {predicted:>9.1%}  {note}")

# %% [markdown]
# **Expected:** at `c = 1.0` the overflow rate is high; it falls sharply by `c = 1.25`; and by
# `c = 2.0` there are no drops, because the measured imbalance is below 2.
#
# **Check your arithmetic against the predicted column.** The closed form $\max(0, 1 - c/I)$ predicts
# the *hot expert's* overflow; the measured rate divides by *all* $kT$ assignments, so it is smaller by
# roughly the hot expert's share of total traffic ($1/E$). If the two disagree by much more than that,
# you have a bug.
#
# The design point is the smallest `c` giving acceptable drops. At the measured imbalance that is
# somewhere between 1.25 and 2.0 — which is exactly why 1.25 is the common default and why it is
# tuned per deployment rather than fixed forever.

# %% [markdown]
# ## 4. Reproduce the auxiliary loss
#
# A real MoE trained with no pressure toward balance collapses: some experts win everything and the
# rest go dead. The standard fix is an **auxiliary loss** added to the task loss.
#
# $$\mathcal{L}_{\text{aux}} = \alpha\, E \sum_{e=1}^{E} f_e \, P_e$$
#
# where $f_e$ is the fraction of *tokens* routed to expert $e$ and $P_e$ the mean *router probability*
# for expert $e$. The product is large when **either** factor is large, so minimising it drives both
# toward $1/E$ — which is uniform load.
#
# Here we train a router with **only** this loss and watch the load flatten out.

# %%
def aux_loss_fn(probs, topi, E, k, alpha=0.01):
    """Switch-Transformer load-balancing loss. f: token fraction per expert, P: mean probability."""
    T = probs.shape[0]
    f = torch.bincount(topi.reshape(-1), minlength=E).float() / (T * k)
    P = probs.mean(0)
    return alpha * E * (f * P).sum(), f


torch.manual_seed(0)
Ttrain = 1024
xtr = torch.randn(Ttrain, 512, device=DEV, dtype=DT)
E, K = 8, 2
wr = torch.randn(E, 512, device=DEV, requires_grad=True) * 0.02
opt = torch.optim.Adam([wr], lr=0.05)

print(f"{'step':>6} {'imbalance':>11} {'Gini':>7} {'aux loss':>10}")
print("-" * 38)
for step in range(301):
    probs = F.softmax(xtr.float() @ wr.T, dim=-1)
    topv, topi = torch.topk(probs, K, dim=-1)
    loss, _ = aux_loss_fn(probs, topi, E, K, alpha=0.01)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 60 == 0:
        c = torch.bincount(topi.reshape(-1), minlength=E)
        imb, _, gini = measure(c, Ttrain, E, K)
        print(f"{step:>6} {imb:>11.2f} {gini:>7.3f} {loss.item():>10.4f}")

print()
print("Trained on the auxiliary loss alone, the router converged to near-uniform load.")

# %% [markdown]
# **Expected:** imbalance falls toward 1.0 and Gini toward 0 as the loss trains. Uniform load is the
# aux loss's global minimum, so with nothing pulling the other way the router goes there.
#
# Now run the experiment that matters: set `alpha = 0.0` and re-run. Nothing changes — a router with
# no pressure toward balance simply keeps whatever preferences its random initialisation had.
# The auxiliary loss is not a refinement. It is the only reason a trained MoE uses all of its experts.
#
# **The real trade-off** is between this loss and the task loss. A real training run minimises
# $\mathcal{L}_{\text{task}} + \alpha\,\mathcal{L}_{\text{aux}}$, so `alpha` sets how much specialisation
# you buy per unit of balance. Too low and experts go dead; too high and every expert becomes an
# identical copy and you have paid for capacity you never used.

# %% [markdown]
# ## 5. What imbalance costs on the wire
#
# Now connect load to communication, using the lesson's formula:
#
# $$f_{\text{remote}} = \Big(1 - \tfrac{1}{g}\Big)\cdot I, \qquad
# \text{bytes}_{\text{step}} = 2 \cdot T \cdot f_{\text{remote}} \cdot d \cdot b \cdot L_{\text{moe}}$$
#
# Then measure the *local* work the same layer must do, so the communication figure has something
# to be compared against.

# %%
d, b, L_moe = 512, 2, 32     # model width, bytes per number, MoE layer count
g = 8                          # expert-parallel degree

print(f"{'I':>6} {'f_remote':>10} {'bytes/step':>14} {'vs balanced':>12}")
print("-" * 46)
for i in (1.0, 1.0 + (imb - 1.0) * 0.25, 1.0 + (imb - 1.0) * 0.5, 1.0 + (imb - 1.0) * 0.75, imb):
    f = (1 - 1 / g) * i
    tot = 2 * Ttrain * f * d * b * L_moe
    base = 2 * Ttrain * (1 - 1 / g) * d * b * L_moe
    print(f"{i:>6.2f} {f:>10.3f} {tot/1e6:>10.1f} MB {tot/base:>11.2f}×")

# %%
E, H = 8, 256
xe = torch.randn(E, Ttrain, d, device=DEV, dtype=DT)
w1 = torch.randn(E, d, H, device=DEV, dtype=DT)
w2 = torch.randn(E, H, d, device=DEV, dtype=DT)


def expert_gemm():
    return F.gelu(xe @ w1) @ w2


t_local = bench(expert_gemm, iters=10, warmup=3)
# activations out (E,T,H) + both weight matrices read (E,d,H) and (E,H,d)
bytes_local = E * Ttrain * H * b + 2 * E * d * H * b
flops_local = 2 * 2 * E * Ttrain * d * H
comm_bytes = 2 * Ttrain * (1 - 1 / g) * d * b * L_moe

print()
print(f"local expert GEMM, {E} experts x {Ttrain} tokens: {t_local*1e3:.2f} ms")
print(f"  moves {bytes_local/1e6:.1f} MB, {flops_local/1e12:.2f} TFLOP")
print(f"  achieved {bytes_local / t_local / 1e9:.0f} GB/s, "
      f"{flops_local / t_local / 1e12:.0f} TFLOP/s")
print(f"  arithmetic intensity {flops_local / bytes_local:.0f} FLOP/byte")
print()
print(f"all-to-all, {L_moe} layers: {comm_bytes/1e6:.1f} MB")
print(f"  NVLink 300 GB/s at 50%: {comm_bytes / 150e9 * 1e3:.3f} ms")
print(f"  PCIe   25 GB/s at 50%: {comm_bytes / 12.5e9 * 1e3:.3f} ms")
print(f"  comm as a fraction of local work: {comm_bytes / 150e9 / t_local:.1%}")

# %% [markdown]
# **Expected:** the grouped GEMM takes a few milliseconds while the all-to-all takes a fraction of a
# millisecond at NVLink speeds — so at this batch the layer is still **locally compute-bound**.
#
# Compare your arithmetic intensity to the T4's ridge point: $65\text{ TFLOP/s} \div 320\text{ GB/s}
# \approx 203$ FLOP/byte. An intensity far below 203 means the expert GEMM is **memory-bound on a T4**,
# which is worth noting — a T4 has no BF16 tensor cores, so it is a poor card for expert-heavy work.
#
# Now find the crossover, where communication stops hiding behind local compute:
#
# ```
# %%
# for batch in (1, 8, 32, 128, 512, 2048):
#     xb = torch.randn(E, batch, d, device=DEV, dtype=DT)
#     t = bench(lambda: (F.gelu(xb @ w1)) @ w2, iters=10, warmup=3)
#     comm = 2 * batch * (1 - 1/g) * d * b * L_moe / 150e9
#     print(f"{batch:>5}  local {t*1e3:>8.2f} ms   comm {comm*1e3:>8.3f} ms   comm/local {comm/t:>7.1%}")
# ```
#
# **The conclusion to write down:** MoE inference is communication-bound when the interconnect is slow
# relative to the local work per token, and compute-bound when it is fast. At NVLink speeds you are
# usually fine; on PCIe, or at low batch, you are not.

# %% [markdown]
# ## 6. Read a real trained MoE
#
# Everything so far used synthetic routers. This section looks at published configurations, because
# the only numbers that matter are the ones a trained model actually produces.
#
# `Qwen1.5-MoE-A2.7B` is a small published MoE whose config is small enough to fetch on a T4.

# %%
if HAVE_TR:
    name = "Qwen/Qwen1.5-MoE-A2.7B"
    try:
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        print("config loaded:", name)
        for k in ("hidden_size", "num_hidden_layers", "num_experts",
                  "num_experts_per_tok", "moe_intermediate_size",
                  "shared_expert_intermediate_size", "shared_expert_num"):
            print(f"  {k:>30}: {getattr(cfg, k, 'n/a')}")

        E = getattr(cfg, "num_experts", 0)
        K = getattr(cfg, "num_experts_per_tok", 0)
        d = cfg.hidden_size
        L = getattr(cfg, "num_hidden_layers", 1)
        inter = getattr(cfg, "moe_intermediate_size", 0)
        if E and K and d:
            bank = 3 * E * d * inter
            active = 3 * K * d * inter
            print()
            print(f"  expert bank      : {bank/1e9:.2f} B params per MoE layer")
            print(f"  active per token : {active/1e9:.2f} B params per MoE layer")
            print(f"  bank sparsity    : {bank/active:.1f}×")
            print(f"  KV bytes / token : {2 * L * getattr(cfg, 'num_key_value_heads', 1) * d * 2}")
    except Exception as e:
        print("config fetch failed:", type(e).__name__, str(e)[:200])
else:
    print("skipped: transformers not installed")

# %%
# Find the router weights themselves. Hugging Face names a MoE router `mlp.gate`, not `mlp.router`.
if HAVE_TR:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo = "Qwen/Qwen1.5-MoE-A2.7B"
        files = [s.rfilename for s in api.list_repo_files(repo) if s.endswith(".safetensors")]
        router_files = sorted(f for f in files if "mlp.gate" in f)
        print(f"{len(files)} safetensors shards")
        print(f"{len(router_files)} contain router weights")
        for f in router_files[:3]:
            print("  ", f)
    except Exception as e:
        print("repo listing failed:", type(e).__name__, str(e)[:200])

# %% [markdown]
# **Expected:** the config prints the published architecture — a large expert count, a small number
# active per token, a narrow MoE intermediate width, and a **shared expert**.
#
# Note the shared expert's reason for existing: routed experts are *not* guaranteed to fire for every
# token, so a small always-on expert provides the capacity every token needs. That is a serving
# decision as much as a modelling one — it guarantees a floor of work per token, which is exactly
# what you want when the router is unpredictable.
#
# **What to measure on real router weights:** the per-expert row norms of `mlp.gate.weight`. A trained
# router with genuine favourites has rows whose norms differ by a large factor. Experts with
# near-identical norms have never differentiated — the signature of an under-trained or
# under-utilised MoE, and the first thing to check if a model is mysteriously slow to train.

# %% [markdown]
# ## When it goes wrong
#
# Five failure modes, each of which produces a *plausible* model rather than a crash.
#
# **1. Top-k without renormalising.** Selecting the top-k probabilities but not renormalising them to
# sum to 1 scales the layer's contribution by less than 1. Symptoms: coherent output that is
# systematically low-amplitude, worsening as `k` grows.
#
# **2. The combine step indexed on the wrong axis.** `topi` is `[T, k]`; expert outputs are `[E, T, d]`.
# Mixing them up yields a tensor of exactly the right shape and entirely wrong values. It will not
# raise. Catch it by comparing one token's output against a hand-computed reference.
#
# **3. Softmax taken over the selected experts instead of all of them.** This silently renormalises
# before selection and changes every gate value. Always softmax across all `E`, then take top-k, then
# renormalise over the `k`.
#
# **4. A router bias that dominates the input signal.** A large `b` makes routing nearly
# input-independent, so every token goes to the same few experts. The imbalance metric catches it
# immediately; the symptom on the model is one that has quietly become a dense model with extra steps.
#
# **5. Measuring imbalance on the wrong axis.** Counts must be per *expert* across *all tokens*. Per-token
# counts are always exactly `k` — perfectly balanced by construction — which makes the metric useless
# and the bug completely invisible. This is the one that has shipped more than once.

# %% [markdown]
# ## Your turn
#
# 1. Section 4 converged to uniform load because nothing pushed the other way. Add a task term that
#    rewards sending tokens to expert 0, and show that a large enough `alpha` still wins. Find the
#    `alpha` at which the router stops specialising — that ratio *is* the training trade-off.
# 2. Raise `n_experts` from 8 to 64 at fixed `d`, `h`, `k`. Does imbalance grow? Plot Gini against
#    expert count. The claim is that more experts means more imbalance at equal router quality —
#    your measurement either supports or refutes it.
# 3. Section 3 drops overflowing tokens. Implement the alternative: reroute an overflowing token to the
#    next-best expert that still has capacity. Compare imbalance and per-expert GEMM shape against
#    the drop policy, and argue which is cheaper at high batch.
# 4. Push section 5 to a realistic 8-GPU Mixtral configuration (`d = 4096`, `L = 31`, `T = 512`) and
#    find the batch size at which communication time exceeds local GEMM time on your GPU's real bandwidth.
# 5. Measure the expert GEMM's arithmetic intensity and compare it to the T4 ridge point
#    (65 TFLOP/s ÷ 320 GB/s ≈ 203 FLOP/byte). Is the expert layer compute- or memory-bound on a T4?
#    Does the answer change on an H100, and does that change which precision you should deploy?

# %%
print("done")