# %% [markdown]
# # Lab 16 · Map where speculation wins
# #
# **Tokens per Second** · Module 16
# #
# You will:
# 1. Implement speculative decoding from scratch and prove it is **exact** — same
#    distribution as plain sampling, not an approximation.
# 2. Measure the acceptance rate on real text with two different drafters.
# 3. Measure what a verify pass actually costs versus one decode step.
# 4. Measure the cost ratio rho on your GPU.
# 5. Check the formula: tokens-per-block, speedup, and the break-even rule
#    (the fraction of drafted tokens kept must beat rho).
# 6. Find where speculation loses, and confirm the model predicts it.
# #
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
# #
# Rule of the whole lab: **implement, verify, then time.** In that order.

# %%
import math
import time
import torch

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
DEV = "cuda"
DT = torch.float16
torch.manual_seed(0)

print(torch.cuda.get_device_name(0))
print("torch", torch.__version__)


def bench(fn, iters=30, warmup=10):
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
# ## 1. The exact algorithm
# #
# This is the whole technique in about thirty lines. Note how little is here:
# a proposed token, a target distribution, an acceptance test, and a correction.
# There is no "fast mode" and no quality knob — that is the point.

# %%
def speculative_step(target_p, draft_p, draft_tokens):
    """
    One speculative block, verified exactly.

    target_p, draft_p : [gamma+1, V] and [gamma, V] probability distributions.
    draft_tokens      : the gamma tokens the drafter actually emitted,
                        each sampled from the corresponding draft distribution.

    Returns (emitted_tokens, n_accepted, n_tested). n_tested counts the drafts the
    target actually examined, up to and including the first rejection. All gamma
    drafts were generated, and paid for, before this function ran.

    Three details that make this EXACT rather than approximate — get any of them
    wrong and the output distribution silently changes:
      1. the proposal x must be SAMPLED from q (which the drafter already did),
         never taken as q's argmax;
      2. an accepted token is emitted AS x — the q(x) in the acceptance test
         cancels the q(x) with which x was drawn, which is the whole proof;
      3. the bonus token is SAMPLED from p at the final position, not argmax,
         because plain sampling would have sampled it too.
    """
    gamma = len(draft_tokens)
    accepted = []

    for i, x in enumerate(draft_tokens):
        p, q = target_p[i], draft_p[i]
        # accept with probability min(1, p(x)/q(x))
        if float(p[x]) / max(float(q[x]), 1e-12) > torch.rand(1).item():
            accepted.append(x)
            continue
        # rejected: sample the replacement from normalize(max(0, p - q)),
        # i.e. from exactly the probability mass the draft failed to cover
        resid = (p - q).clamp(min=0)
        s = float(resid.sum())
        repl = int(target_p[i].argmax()) if s <= 0 else int(torch.multinomial(resid / s, 1).item())
        return accepted + [repl], len(accepted), i + 1

    # every draft accepted: the bonus token, sampled from p at the last position
    bonus = int(torch.multinomial(target_p[gamma], 1).item())
    return accepted + [bonus], len(accepted), gamma

# %% [markdown]
# ## 2. Verify it is exact
# #
# The claim is that this produces the same distribution as sampling from the
# target alone. The test: use a target that is a *mixture* of a peaked
# distribution and a uniform one, so the corrected distribution is easy to get
# wrong, then compare histograms over many trials.
# #
# If our implementation is right, the two histograms should agree to within
# sampling noise, even though the draft is often wrong.

# %%
V = 8
# target: peaked but with a real tail
base_p = torch.tensor([0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02], device=DEV, dtype=torch.float32)
# draft: deliberately miscalibrated — its best guess is the target's 4th choice
draft_q = torch.tensor([0.05, 0.10, 0.20, 0.30, 0.20, 0.08, 0.04, 0.03], device=DEV, dtype=torch.float32)

GAMMA = 3
N = 200_000

# The target scored every position of the block; the draft scored the gamma it proposed.
target_p = base_p.unsqueeze(0).repeat(GAMMA + 1, 1)
draft_p = draft_q.unsqueeze(0).repeat(GAMMA, 1)

# plain sampling from the target, for comparison
plain = torch.multinomial(base_p, N, replacement=True)

# speculative sampling. The drafter's proposals are themselves samples from q —
# that is what a real draft model produces.
counts = torch.zeros(V, device=DEV)
n_blocks = N // 20
n_emitted = 0
n_accepted = 0
n_tested = 0
for _ in range(n_blocks):
    draft_tokens = torch.multinomial(draft_p, 1, replacement=True).squeeze(-1)  # [gamma]
    toks, acc, tested = speculative_step(target_p, draft_p, draft_tokens)
    for tk in toks:
        counts[tk] += 1
    n_emitted += len(toks)
    n_accepted += acc
    n_tested += tested

print(f"{n_blocks} blocks produced {n_emitted} tokens ({n_emitted/n_blocks:.3f} per block)\n")
print("token   target     speculative   diff")
worst = 0.0
for v in range(V):
    a = float((plain == v).float().mean())
    b = float(counts[v]) / n_emitted
    worst = max(worst, abs(b - a))
    print(f"  {v}   {a:8.4f}   {b:10.4f}   {b - a:+.4f}")
print(f"\nlargest deviation: {worst:.4f}  (sampling noise at N=200k is ~{3/math.sqrt(N):.4f})")

# %% [markdown]
# **Expected:** every row's difference is within about 0.01 — sampling noise for
# 200,000 trials. If a row is off by a lot, the correction is wrong.
# This is the single most important check in the lab: everything else assumes it.
#
# If you see a *uniform* offset across all rows, the bug is almost always the
# normalisation: compare against the number of tokens actually emitted, not the
# number of trials.

# %%
# the analytic prediction, compared with what we just measured
a_rate = n_accepted / n_tested              # per-draft acceptance: accepted / drafts examined
kept = a_rate * (1 - a_rate**GAMMA) / (1 - a_rate)   # a + a^2 + ... + a^g
print("Acceptance and tokens per block")
print(f"  observed a              = {a_rate:.4f}")
print(f"  predicted tokens/block  = 1 + (a + a^2 + ... + a^g) = {1 + kept:.4f}")
print(f"  observed tokens/block   = {n_emitted/n_blocks:.4f}")
print(f"  predicted keep rate     = kept / g = {kept/GAMMA:.4f}")
print(f"  observed  keep rate     = {n_accepted/(n_blocks*GAMMA):.4f}")
print(f"  drafts paid per block   = {GAMMA}  (every draft, kept or not)")

# %% [markdown]
# ## 3. The formula, checked against the simulation
# #
# The lesson derived:
#
#     E[kept]          = a + a^2 + ... + a^g = a (1 - a^g) / (1 - a)
#     E[tokens/block]  = 1 + E[kept]
#     cost (steps)     = 1 + rho * g        (all g drafts are paid for)
#     speedup          = E[tokens] / cost
#     break-even       : keep rate E[kept]/g = rho   (a = rho only when g = 1)

# %%
def predict(a, rho, gamma):
    kept = a * (1 - a**gamma) / (1 - a) if a < 1 else gamma
    tokens = 1 + kept
    cost = 1 + rho * gamma
    return kept, tokens, cost, tokens / cost


print("gamma sweep at a=0.72, rho=0.125")
print(" g   E[kept]   tokens  cost   speedup")
for g in [1, 2, 3, 4, 6, 8, 12]:
    d, tk, c, sp = predict(0.72, 0.125, g)
    print(f"{g:2d}   {d:8.3f}  {tk:6.3f}  {c:5.3f}  {sp:6.3f}")

print("\nbreak-even: a at which speedup == 1 (idealised model)")
for rho in [0.02, 0.125, 0.3]:
    for g in [1, 4]:
        lo, hi = 0.0, 0.999
        for _ in range(60):
            mid = (lo + hi) / 2
            if predict(mid, rho, g)[3] < 1:
                lo = mid
            else:
                hi = mid
        print(f"  rho={rho:.3f} gamma={g} -> a* = {(lo+hi)/2:.4f}")
print("theory: a* = rho at gamma = 1; higher at gamma = 4, because every draft is")
print("paid for and later drafts are kept less often (expect ~0.074, 0.336, 0.574).")

# %% [markdown]
# ## 4. Measure it on a real model
# #
# Now the part that matters. Qwen2.5-0.5B-Instruct drafting Qwen2.5-1.5B-Instruct
# — both small enough to fit on a T4 with room for the KV cache.
# We run a real prompt, measure the acceptance rate on the real text, and time
# one decode step of each model.

# %%
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise SystemExit("pip install -q transformers")

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"

tok = AutoTokenizer.from_pretrained(TARGET)
target = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=DT, device_map=DEV).eval()
draft = AutoModelForCausalLM.from_pretrained(DRAFT, torch_dtype=DT, device_map=DEV).eval()

print("target params:", sum(p.numel() for p in target.parameters()) / 1e9, "B")
print("draft  params:", sum(p.numel() for p in draft.parameters()) / 1e9, "B")

# %% [markdown]
# ## 5. Acceptance rate on real text
# #
# The draft runs forward to produce gamma tokens; the target scores the same
# positions; we count how many drafted tokens matched the target's argmax.
# This is `a` — the number that decides everything.

# %%
PROMPT = (
    "Translate the following list into French, one item per line:\n"
    "1. apples\n2. bread\n3. cheese\n4. wine\n5. water\n"
)
ids = tok(PROMPT, return_tensors="pt").to(DEV)


@torch.no_grad()
def measure_acceptance(prompt_ids, gamma=4, n_tokens=120):
    accepted_total = 0
    tested_total = 0
    blocks = 0
    cur = prompt_ids.clone()

    for _ in range(n_tokens // gamma):
        ctx = cur

        # --- draft: generate gamma tokens autoregressively ---
        draft_ids = ctx.clone()
        for _ in range(gamma):
            out = draft(input_ids=draft_ids, use_cache=False)
            nxt = out.logits[0, -1].argmax().view(1, 1)
            draft_ids = torch.cat([draft_ids, nxt], dim=1)

        block = draft_ids[:, -gamma:]

        # --- target: score all gamma positions in ONE pass ---
        target_in = torch.cat([ctx, block], dim=1)
        logits = target(input_ids=target_in, use_cache=False).logits[0]

        # position j of the block is predicted by the logit at index len(ctx)-1+j
        base = ctx.shape[1] - 1
        ok = 0
        for j in range(gamma):
            if int(logits[base + j].argmax()) == int(block[0, j]):
                ok += 1
            else:
                break
        accepted_total += ok
        tested_total += ok + (1 if ok < gamma else 0)   # drafts the target examined
        blocks += 1
        cur = torch.cat([ctx, block[:, :ok]], dim=1)
        if cur.shape[1] > 400:
            break

    a = accepted_total / max(1, tested_total)          # per-draft acceptance
    keep = accepted_total / max(1, blocks * gamma)     # fraction of drafted tokens kept
    return a, keep, blocks


a_measured, keep_measured, nblocks = measure_acceptance(ids, gamma=4)
print(f"measured acceptance a = {a_measured:.3f} over {nblocks} blocks")
print(f"measured keep rate    = {keep_measured:.3f}  (accepted / all drafted, gamma = 4)")
print(f"(greedy drafting on a translate task — expect a fairly high rate)")

# %% [markdown]
# **Expected:** roughly 0.6–0.8 for this kind of templated, repetitive task.
# On free-form prose it will be lower. That gap is the whole lesson: acceptance
# is a property of the text, so measure it on *your* traffic.

# %% [markdown]
# ## 6. The cost ratio rho, measured
# #
# rho is how long one draft step takes relative to one target decode step.
# It is not just the parameter ratio — both models pay the same fixed per-step
# overhead, which is why a small model is not *as* cheap as its size suggests.

# %%
@torch.no_grad()
def time_decode(model, prompt_ids, n=40):
    cur = prompt_ids.clone()
    for _ in range(8):  # warm up
        out = model(input_ids=cur, use_cache=True)
        cur = torch.cat([cur, out.logits[0, -1].argmax().view(1, 1)], dim=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = model(input_ids=cur, use_cache=True)
        nxt = out.logits[0, -1].argmax().view(1, 1)
        cur = torch.cat([cur, nxt], dim=1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


t_target = time_decode(target, ids)
t_draft = time_decode(draft, ids)
rho_measured = t_draft / t_target

print(f"target decode step: {t_target*1e3:8.2f} ms")
print(f"draft  decode step: {t_draft*1e3:8.2f} ms")
print(f"measured rho = {rho_measured:.3f}")
print(f"parameter ratio = {sum(p.numel() for p in draft.parameters())/sum(p.numel() for p in target.parameters()):.3f}")
print("rho is LARGER than the parameter ratio, because the fixed per-step")
print("overhead (launches, scheduling, Python) is the same for both models.")

# %% [markdown]
# **Expected:** rho somewhere around 0.4–0.6 for this pair — much larger than the
# 0.33 parameter ratio. On a T4, which is slow at both matmul *and* launch, the
# fixed cost dominates. That is why a T4 is a poor place to run neural speculation
# and a good place to run n-gram or prompt lookup: rho near zero beats rho at 0.5.

# %% [markdown]
# ## 7. Is the verify pass really free?
# #
# The premise of the whole technique. Run the target on ONE position (a decode
# step) and on gamma+1 positions (a verify pass) and compare.

# %%
@torch.no_grad()
def time_pass(model, seq_len, batch_pos, n=30):
    x = torch.randint(0, 1000, (1, seq_len), device=DEV)
    for _ in range(10):
        model(input_ids=x, use_cache=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(input_ids=x, use_cache=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


prompt_len = ids.shape[1]
print("gamma  positions   time      vs 1 decode step")
t1 = time_pass(target, prompt_len, 1)
print(f"  1        1      {t1*1e3:7.2f} ms      1.000x   <- a decode step")
for g in [1, 2, 4, 8, 16]:
    t = time_pass(target, prompt_len + g + 1, g + 1)
    print(f" {g:2d}      {g+1:3d}     {t*1e3:7.2f} ms    {t/t1:.3f}x")

# %% [markdown]
# **Expected:** the ratio stays close to 1.0x for small gamma, because the step
# is dominated by reading the weights — the same bytes either way. It only starts
# to climb once the arithmetic becomes significant.
# This is the measurement the whole technique rests on. If this ratio were 4x,
# speculation would be pointless.

# %% [markdown]
# ## 8. Put it together: the predicted speedup
# %
# Now feed the *measured* a and rho into the formula and compare with what the
# formula said before you measured anything.

# %%
GAMMA = 4
kept, tokens, cost, speedup = predict(a_measured, rho_measured, GAMMA)
print(f"measured a   = {a_measured:.3f}")
print(f"measured rho = {rho_measured:.3f}")
print(f"gamma        = {GAMMA}")
print()
print(f"E[kept]         = {kept:.3f}   keep rate = {kept/GAMMA:.3f}")
print(f"E[tokens/block] = {tokens:.3f}")
print(f"cost (steps)    = {cost:.3f}")
print(f"speedup         = {speedup:.3f}x")
print()
if kept / GAMMA > rho_measured:
    print(f"keep rate > rho, so speculation WINS — expected {speedup:.2f}x on this task.")
else:
    print(f"keep rate < rho, so speculation LOSES ({speedup:.2f}x) at gamma = {GAMMA}.")
    best = max(range(1, 9), key=lambda g: predict(a_measured, rho_measured, g)[3])
    print(f"best gamma for these numbers: {best} -> {predict(a_measured, rho_measured, best)[3]:.2f}x")
    print("On a T4 with these two models this is the likely outcome — which is")
    print("itself the finding: measure rho before you trust any paper's speedup.")

# %% [markdown]
# ## 9. Where it loses
# #
# Three checks that each flip the sign. Run them and see which ones bite.

# %%
print("A. Keep rate below rho")
for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
    kp, tk, c, sp = predict(a, rho_measured, GAMMA)
    verdict = "win " if sp >= 1 else "LOSS"
    print(f"   a={a:.2f}  keep={kp/GAMMA:.3f}  rho={rho_measured:.3f}  speedup={sp:.3f}x  {verdict}")

print("\nB. Batch size kills the memory-bound premise")
# at batch B the weight read is amortised across the batch, so bytes per token fall as 1/B
cfg = target.config
kv_bytes = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2
weights_bytes = sum(p.numel() * p.element_size() for p in target.parameters())
for B in [1, 8, 32, 64]:
    per_token = (weights_bytes + prompt_len * kv_bytes) / B
    print(f"   batch {B:3d}: weight bytes per token = {weights_bytes/B/1e6:8.1f} MB  (kv {kv_bytes/1e3:.1f} KB/token)")

print("\nC. Short outputs never amortise")
print("   For a 3-token answer you pay the full draft cost and one verify pass,")
print("   and TTFT gets worse because nothing is emitted until the block is verified.")

# %% [markdown]
# ## 10. N-gram drafting, where rho is ~0
# #
# The escape hatch for small GPUs. No model runs at all, so rho is essentially
# zero and almost any acceptance rate wins.

# %%
class NgramDrafter:
    def __init__(self, n=3):
        self.n = n
        self.table = {}

    def update(self, ids):
        key = tuple(ids[-(self.n + 1):-1].tolist())
        self.table.setdefault(key, []).append(int(ids[-1]))

    def propose(self, ids):
        key = tuple(ids[-self.n:].tolist())
        cands = self.table.get(key)
        if not cands:
            return None
        # deterministic: pick the most recent continuation
        return cands[-1]


# Build an n-gram table from the model's OWN greedy output, then re-run:
# high agreement is expected because we are copying the model back to itself.
drafter = NgramDrafter(n=2)
with torch.no_grad():
    cur = ids.clone()
    gen = []
    for _ in range(80):
        out = target(input_ids=cur, use_cache=False)
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
        drafter.update(torch.tensor([nxt]))
        cur = torch.cat([cur, torch.tensor([[nxt]], device=DEV)], dim=1)

print(f"built n-gram table with {len(drafter.table)} entries from 80 tokens")

# replay and count agreement
hits = 0
total = 0
with torch.no_grad():
    cur = ids.clone()
    for _ in range(60):
        d = drafter.propose(cur[0].tolist())
        out = target(input_ids=cur, use_cache=False)
        t = int(out.logits[0, -1].argmax())
        total += 1
        if d is not None:
            hits += 1 if d == t else 0
        cur = torch.cat([cur, torch.tensor([[t]], device=DEV)], dim=1)

a_ngram = hits / total
print(f"n-gram acceptance = {a_ngram:.3f}")
print(f"n-gram rho        ~= 0.001 (a dict lookup)")
_, tk, c, sp = predict(a_ngram, 0.001, 4)
print(f"predicted speedup = {sp:.3f}x")
print()
print("Compare with the neural drafter's rho of", round(rho_measured, 3))
print("— the method with no model at all wins decisively on a T4.")

# %% [markdown]
# ## 11. Three ways to get this wrong
# #
# All three produce *plausible* output. None crashes. That is what makes them
# dangerous, and all three were real bugs during development of this notebook.

# %%
# (a) Using the draft's ARGMAX as the proposal instead of sampling from q.
# This looks reasonable — the drafter "suggests" a token — and it still accepts
# often, so tokens-per-block looks reasonable too. But the output distribution
# is no longer the target's.
def broken_argmax_step(p, q, gamma):
    accepted = []
    for i in range(gamma):
        x = int(q[i].argmax())                      # WRONG: not a sample from q
        if float(p[i][x]) / max(float(q[i][x]), 1e-12) > torch.rand(1).item():
            accepted.append(x)
        else:
            break
    return accepted + [int(p[gamma].argmax())], len(accepted)


cnt_bad = torch.zeros(V, device=DEV)
tot_bad = 0
for _ in range(n_blocks):
    toks, _ = broken_argmax_step(target_p, draft_p, GAMMA)
    for tk in toks:
        cnt_bad[tk] += 1
    tot_bad += len(toks)

print("same test, with argmax used as the proposal:")
print("token   target     broken       diff")
for v in range(V):
    a = float(base_p[v])
    b = float(cnt_bad[v]) / tot_bad
    print(f"  {v}   {a:8.4f}   {b:10.4f}   {b - a:+.4f}")
print("\nCompare section 2. The correct version matched to ~0.01; this one")
print("does not. Nothing threw an exception.")

# %%
# (b) Taking the bonus token as argmax instead of sampling it from p.
# This only shows up when the whole block is accepted, so it is rarer — and when
# the target distribution is flat it quietly flattens your output.
# (c) Forgetting that the accepted token is emitted AS the proposal x, rather
# than resampling. The q(x) that cancels is the one x was drawn with.

print("\nThe three fixes, in one place:")
print("  1. proposal x must be sampled from q          (not q.argmax())")
print("  2. accepted token is emitted AS x              (not resampled)")
print("  3. bonus token is sampled from p               (not p.argmax())")

# %% [markdown]
# ## What to take away
# %
# 1. Speculative decoding is **exact**. Section 2 showed the histogram matches
#    plain sampling; nothing was approximated. Section 11 showed how easily that
#    property is lost to a one-word change.
# 2. The whole decision is two numbers: **a** (acceptance, measured in §5) and
#    **rho** (draft cost, measured in §6). Win exactly when the fraction of
#    drafted tokens kept beats rho; at gamma = 1 that is simply a > rho.
# 3. The verify pass really is nearly free on a small GPU (§7) — that is what
#    makes the trick work, and it is measurable.
# 4. On a T4, **rho for a neural drafter is much higher than its parameter
#    ratio**, because fixed per-step overhead dominates a slow card. Measure it
#    rather than assuming.
# 5. When rho is large, the free methods (n-gram, prompt lookup) win — and on
#    real repetitive traffic they can have a high acceptance rate too.
# 6. Never normalise a histogram by the number of trials when the trial count
#    varies; normalise by the number of tokens actually emitted.

# %% [markdown]
# ## Going further
# %
# - Sweep `gamma` with the real models and find where the measured speedup peaks.
#   Does it match the formula's prediction of the peak?
# - Measure acceptance on your *own* prompts and watch it vary by task type.
# - Try the same experiment with `torch.compile` + CUDA graphs (Module 15) and see
#   whether a lower fixed overhead improves rho enough to matter.
# - If you have a bigger GPU, repeat with a genuinely large target (8B) and a 1B
#   draft, where rho is far lower and the win is much larger.