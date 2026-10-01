# %% [markdown]
# # Lab 03 · Read a model, trace one token
#
# **Tokens per Second** · Module 03
#
# You will:
# 1. Derive a model's parameter count from its config, **before** loading it, and check it.
# 2. Count real tokens and measure characters per token on four kinds of text.
# 3. **Hand-write one transformer block** and check it against the library.
# 4. Watch the KV cache grow, one row per layer per token.
# 5. Count module calls in a forward pass and reconcile them with the parameter count.
# 6. Time a decode step and compare it against your own memory-bound floor.
# 7. Measure the prefill-versus-generate ratio on your own GPU.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# **Precision note.** A T4 has FP16 tensor cores but **no BF16** and no FP8, so everything
# here is FP16. FP16 keeps about 3 decimal digits, so end-to-end differences below land
# around 1e-2. That is plenty to catch a real bug (a wrong RoPE axis gives errors of
# *order 1*), and the structural checks in step 3 are exact regardless of dtype.
# TinyLlama-1.1B is 2.2 GB in FP16 and 4.4 GB in FP32 — both fit a 16 GB T4 if you
# want to check at 1e-5 by changing `torch_dtype` to `torch.float32`.

# %%
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime → Change runtime type)."
DEV = "cuda"
print(torch.cuda.get_device_name(0), "|", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
DTYPE = torch.float16          # T4: FP16 tensor cores, no BF16
torch.manual_seed(0)

def banner(s):
    print("\n" + "=" * 76 + f"\n{s}\n" + "=" * 76)

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# %% [markdown]
# ## 1. Predict the parameter count from the config, then check it
#
# Do this **on paper first**. For a pre-norm Llama/Qwen2 block:
#
# $$P = L\left(\underbrace{d^2 + 2\,d\,n_{kv}d_h + d^2}_{\text{attention}} + \underbrace{3\,d\,m}_{\text{MLP}}\right) + \underbrace{(2L+1)d}_{\text{norm gains}} + \underbrace{qk\text{-norm gains}}_{\text{Qwen2 only}} + \underbrace{Vd}_{\text{embedding}} + \underbrace{Vd}_{\text{head, unless tied}}$$

# %%
from transformers import AutoConfig

NAMES = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-0.5B-Instruct"]

for name in NAMES:
    c = AutoConfig.from_pretrained(name)
    d, L = c.hidden_size, c.num_hidden_layers
    nq, nkv, dh = c.num_attention_heads, c.num_key_value_heads, c.head_dim
    m, V = c.intermediate_size, c.vocab_size
    tied = bool(getattr(c, "tie_word_embeddings", False))

    attn = d * d + 2 * d * nkv * dh + d * d          # q, k, v, o
    mlp = 3 * d * m                                    # gate, up, down
    norms = (2 * L + 1) * d                            # 2 per block + the final norm
    qk = 2 * L * nkv * dh if c.model_type in ("qwen2", "qwen3") else 0
    head = 0 if tied else V * d
    derived = L * (attn + mlp) + norms + qk + V * d + head

    banner(name)
    print(f"  model_type        {c.model_type}")
    print(f"  layers            {L}")
    print(f"  hidden d          {d}")
    print(f"  heads / kv heads  {nq} / {nkv}   (group of {nq // nkv})")
    print(f"  head dim d_h      {dh}    (d / n_heads = {d // nq})")
    print(f"  intermediate m    {m}    ({m / d:.2f} x d)")
    print(f"  vocab V           {V:,}")
    print(f"  tied embeddings   {tied}")
    print(f"  rope_theta        {getattr(c, 'rope_theta', None)}")
    print()
    print(f"  attention / layer {attn:>14,}    {attn/(attn+mlp)*100:5.1f}% of a block")
    print(f"  MLP / layer       {mlp:>14,}    {mlp/(attn+mlp)*100:5.1f}% of a block")
    print(f"  block total x{L}   {L*(attn+mlp):>14,}")
    print(f"  norms             {norms + qk:>14,}")
    print(f"  embed + head      {V*d + head:>14,}")
    print(f"  DERIVED TOTAL     {derived:>14,}  = {derived/1e9:.3f}B   BF16 {derived*2/1e9:.2f} GB")

# %% [markdown]
# ### Now check against the real weights
#
# If a derived count disagrees with `numel()` by more than 0.1%, find the matrix you forgot.
# The two usual reasons: Qwen2 adds RMSNorm gains on q and k, and some models tie the
# embedding table to the output head.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = NAMES[0]        # TinyLlama: plain Llama architecture, no extra q/k norms
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()

c = model.config
d, L, nkv, dh = c.hidden_size, c.num_hidden_layers, c.num_key_value_heads, c.head_dim
tied = bool(getattr(c, "tie_word_embeddings", False))

block_params = L * (d * d + 2 * d * nkv * dh + d * d + 3 * d * c.intermediate_size)
derived = block_params + (2 * L + 1) * d + c.vocab_size * d + (0 if tied else c.vocab_size * d)
actual = sum(p.numel() for p in model.parameters())

banner(f"{MODEL}: derived vs actual")
print(f"  derived from shapes   {derived:>14,}")
print(f"  sum(p.numel())        {actual:>14,}")
print(f"  difference            {actual - derived:>14,}   ({(actual/derived-1)*100:+.4f}%)")
assert abs(actual - derived) / derived < 1e-3, "parameter count mismatch — find the missing matrix"
print("  MATCH")

# %% [markdown]
# ## 2. Count real tokens
#
# Characters per token is the conversion factor between what a user types and what the
# GPU pays for. Measure it instead of assuming it.

# %%
TEXTS = {
    "English": "Explain the KV cache in two sentences.",
    "Code":    "def kv_bytes(n_layers, n_kv, d_head, ctx, b=2): return 2*n_layers*n_kv*d_head*ctx*b",
    "Numbers": "There were 1,284 requests at 3.75 ms each, 92.5% of them p99 < 450 ms.",
    "Chinese": "解释KV缓存需要两个句子。",
}
rows = []
for kind, text in TEXTS.items():
    ids_t = tok(text, add_special_tokens=False).input_ids
    n = len(ids_t)
    rows.append((kind, len(text), n, len(text) / n))
    print(f"  {kind:9s} chars {len(text):>4}   tokens {n:>4}   chars/token {len(text)/n:5.2f}")
    print(f"            {tok.convert_ids_to_tokens(ids_t)[:12]}")

banner("characters per token")
for kind, ch, n, cpt in rows:
    print(f"  {kind:9s} {cpt:5.2f}")

# %% [markdown]
# ### The KV cache that first prompt needs
#
# $$2 \times L \times n_{kv} \times d_h \times b$$

# %%
prompt_tokens = len(tok(TEXTS["English"], add_special_tokens=False).input_ids)
kv_per_tok = 2 * L * nkv * dh * 2
print(f"  prompt tokens              {prompt_tokens}")
print(f"  KV bytes per token         {kv_per_tok:,}  = {kv_per_tok/1024:.0f} KiB")
print(f"  KV bytes for the prompt    {prompt_tokens*kv_per_tok:,}  = {prompt_tokens*kv_per_tok/1024:.0f} KiB")
print(f"  weights, FP16              {actual*2/1e9:.2f} GB")
print(f"  KV equals ALL the weights at {actual*2/kv_per_tok:,.0f} tokens of context")

# %% [markdown]
# ## 3. Hand-write one transformer block
#
# The centre of the lab. Take the input to the final block from `output_hidden_states`,
# compute the whole block yourself, and compare with what the library produced.

# %%
x0 = torch.tensor([tok(TEXTS["English"], add_special_tokens=False).input_ids], device=DEV)

with torch.no_grad():
    hs = model(x0, output_hidden_states=True).hidden_states
block = model.model.layers[-1]
attn, mlp = block.self_attn, block.mlp
T = x0.shape[1]

banner(f"last block of {MODEL}: hand-written vs library")
with torch.no_grad():
    h = hs[-2].float()                       # input to the last block, in float32
    ref = hs[-1].float()                     # what the library produced
    dh_ = c.head_dim
    nq, nkv_ = c.num_attention_heads, c.num_key_value_heads
    n_rep = nq // nkv_
    eps1, eps2 = block.input_layernorm.eps, block.post_attention_layernorm.eps
    W = lambda lin: lin.weight.float()       # every weight, lifted to float32

    # 3a. RMSNorm, written out: divide by the RMS, multiply by the learned gain
    n1 = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps1) * W(block.input_layernorm)

    # 3b. Q, K, V
    q = F.linear(n1, W(attn.q_proj)).view(1, T, nq, dh_).transpose(1, 2)
    k = F.linear(n1, W(attn.k_proj)).view(1, T, nkv_, dh_).transpose(1, 2)
    v = F.linear(n1, W(attn.v_proj)).view(1, T, nkv_, dh_).transpose(1, 2)

    # 3c. RoPE by hand, using the config's rope_theta
    base = c.rope_theta
    inv = 1.0 / (base ** (torch.arange(0, dh_, 2, device=DEV).float() / dh_))
    ang = torch.arange(T, device=DEV).float()[:, None] * inv[None, :]     # [T, dh_/2]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]                # [1, 1, T, dh_/2]

    def rope(x):
        xp = x.reshape(*x.shape[:-1], -1, 2)
        xa, xb = xp[..., 0], xp[..., 1]
        return torch.stack([xa * cos - xb * sin, xa * sin + xb * cos], dim=-1).flatten(-2)

    q, k = rope(q), rope(k)                    # RoPE never touches V

    # 3d. GQA: each KV head serves n_rep query heads
    k_e, v_e = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)

    # 3e. scores -> causal mask -> softmax -> mix
    scores = (q @ k_e.transpose(-2, -1)) / math.sqrt(dh_)
    mask = torch.full((T, T), float("-inf"), device=DEV).triu(1)
    p = torch.softmax(scores + mask, dim=-1)
    ctxv = (p @ v_e).transpose(1, 2).reshape(1, T, nq * dh_)

    # 3f. output projection + residual
    h2 = h + F.linear(ctxv, W(attn.o_proj))

    # 3g. the MLP by hand (SwiGLU)
    n2 = torch.rsqrt(h2.pow(2).mean(-1, keepdim=True) + eps2) * W(block.post_attention_layernorm)
    out = h2 + F.linear(F.silu(F.linear(n2, W(mlp.gate_proj))) * F.linear(n2, W(mlp.up_proj)),
                        W(mlp.down_proj))

mb = torch.isinf(mask)
banner("structural checks (exact, whatever the dtype)")
print(f"  attention rows sum to 1     {p.sum(-1).min().item():.6f} .. {p.sum(-1).max().item():.6f}")
print(f"  masked cells exactly 0     {bool((p[mb] == 0).all())}   ({int(mb.sum())} cells above the diagonal)")
print(f"  RMS before RMSNorm          {h.pow(2).mean().sqrt().item():.3f}")
print(f"  RMS after  RMSNorm          {n1.pow(2).mean().sqrt().item():.3f}   (the learned gain is folded in)")
print(f"  mean attention share        {p.max(-1).values.mean().item():.3f}   (nq={nq}, nkv={nkv_}, group={n_rep})")

banner("end-to-end, hand-written vs library (FP16 model, float32 arithmetic)")
rel = ((out - ref).norm() / ref.norm()).item()
print(f"  output max |diff|   {(out - ref).abs().max().item():.3e}")
print(f"  output mean |diff|  {(out - ref).abs().mean().item():.3e}")
print(f"  relative error      {rel:.3e}")
print(f"\n  A relative error above ~1e-2 means something is wrong. Usual suspects, in order:")
print(f"    RoPE applied to V          RoPE on the wrong axis")
print(f"    mask applied after softmax causal direction reversed")
print(f"    forget to repeat_interleave the KV heads (GQA)")

# %% [markdown]
# ### The whole model, and where the next token comes from

# %%
with torch.no_grad():
    logits = model(x0).logits[0, -1].float()
probs = torch.softmax(logits, -1).clamp_min(1e-30)
ent = float(-(probs * probs.log2()).sum())
top = torch.topk(probs, 5)

banner(f"the whole model: {MODEL}")
print(f"  logits shape        {[1, T, c.vocab_size]}   ({c.vocab_size:,} vocabulary tokens)")
print(f"  logits bytes        {c.vocab_size * 2:,}  = {c.vocab_size * 2/1024:.0f} KiB in FP16")
print(f"  greedy next token   {tok.decode([int(logits.argmax())])!r}")
print(f"  top-5               {[(tok.decode([int(i)]), round(float(p), 4)) for i, p in zip(top.indices, top.values)]}")
print(f"  entropy             {ent:.3f} bits of {math.log2(c.vocab_size):.2f} possible")
print(f"  sampling cost       about {2*c.vocab_size/1e6:.2f} MFLOP — free next to {2*actual/1e9:.1f} GFLOP of weights")

# %% [markdown]
# ## 4. Watch the KV cache grow
#
# One row of `kv_heads × head_dim` per layer per token. Check the shape and the bytes.

# %%
def cache_tensors(past):
    """Every KV tensor in a cache object, across transformers versions."""
    if hasattr(past, "layers"):                       # >= 4.54
        return [t for lay in past.layers for t in (lay.keys, lay.values)]
    if hasattr(past, "key_cache"):                    # 4.4x - 4.53
        return list(past.key_cache) + list(past.value_cache)
    out = []
    for e in past:                                    # legacy tuple of tuples
        out += list(e) if isinstance(e, (tuple, list)) else [e]
    return out

def cache_bytes(past):
    return sum(t.numel() * t.element_size() for t in cache_tensors(past))

banner("cache shapes for one prompt")
with torch.no_grad():
    o = model(x0, use_cache=True)
past = o.past_key_values
ts = cache_tensors(past)
k0, v0 = ts[0], ts[1]
print(f"  layers in the cache     {len(ts)//2}")
print(f"  layer 0 key shape       {tuple(k0.shape)}   = [batch, kv_heads, seq, head_dim]")
print(f"  layer 0 value shape     {tuple(v0.shape)}")
print(f"  bytes for ONE layer     {(k0.numel()+v0.numel())*k0.element_size():,}")
print(f"  formula per token       {kv_per_tok:,}")
print(f"  MATCH                   {(k0.numel()+v0.numel())*k0.element_size()/prompt_tokens == kv_per_tok}")
print(f"  whole cache             {cache_bytes(past):,}  = {cache_bytes(past)/1024:.0f} KiB")
del o

banner("growth, one generated token at a time")
with torch.no_grad():
    cur = x0
    cache = None
    for step in range(5):
        o = model(cur, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        seq = cache_tensors(cache)[0].shape[2]
        by = cache_bytes(cache)
        print(f"  step {step}:  seq {seq:>3}   bytes {by:>12,}   =  {seq} x {kv_per_tok:,}")
        cur = o.logits[:, -1:].argmax(-1)
print(f"\n  The cache is the only structure in the model that grows during generation.")
print(f"  Weights do not grow. Activations are discarded after every step.")

# %% [markdown]
# ## 5. Count the work: module calls in one forward pass
#
# Hook every `nn.Linear` and count. There should be exactly 7 per block, plus the head.

# %%
counts = {}
mods = dict(model.named_modules())
handles = [m.register_forward_hook(
    (lambda nm: (lambda mod, i, o: counts.__setitem__(nm, counts.get(nm, 0) + 1)))(name))
    for name, m in mods.items() if isinstance(m, nn.Linear)]

with torch.no_grad():
    model(x0)
for h in handles:
    h.remove()

banner("linear layers called in one forward pass")
touched = 0
for name in sorted(counts):
    n_par = mods[name].weight.numel()
    touched += n_par * counts[name]
    print(f"  {name:50s} x{counts[name]:<3d} {n_par:>12,}")
print(f"  {'':50s}      {touched:>12,}  = {touched/1e9:.3f}B weights touched")

expect = block_params + (0 if tied else c.vocab_size * d)
print(f"\n  block parameters (derived in step 1)   {block_params:>12,}")
print(f"  plus the output head                  {(0 if tied else c.vocab_size*d):>12,}")
print(f"  equals the total touched: {touched == expect}")
print(f"  model total, for reference            {actual:>12,}")

# %% [markdown]
# ## 6. Decode against your own memory floor
#
# Measure this GPU's achievable bandwidth, then compute the floor from Module 02's
# formula: bytes / (bandwidth x 0.8). A healthy eager-PyTorch decode step lands within
# roughly 1.1-1.4x of it. Far above means launches and elementwise ops, not bandwidth.

# %%
src = torch.empty(int(2e8), dtype=torch.uint8, device=DEV)
dst = torch.empty_like(src)
t_copy = bench(lambda: dst.copy_(src), iters=20)
my_bw = 2 * src.numel() / t_copy
del src, dst
torch.cuda.empty_cache()
print(f"  measured copy bandwidth   {my_bw/1e9:.0f} GB/s   (T4 spec is about 320 GB/s)")

banner("decode step vs the memory floor")
print(f"  {'context':>8} {'measured ms':>12} {'floor ms':>10} {'ratio':>8} {'tok/s':>8}")
ratios = []
for ctx_len in [256, 1024, 4096, 8192]:
    ids_c = torch.randint(0, 30000, (1, ctx_len), device=DEV)
    with torch.no_grad():
        o = model(ids_c, use_cache=True)
        past_c = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1)
        del o
        t = bench(lambda: model(nxt, past_key_values=past_c, use_cache=True), iters=15)
    floor = (actual * 2 + ctx_len * kv_per_tok) / (my_bw * 0.8)
    ratios.append(t / floor)
    print(f"  {ctx_len:>8} {t*1000:>12.3f} {floor*1000:>10.3f} {t/floor:>7.2f}x {1/t:>8.1f}")
    del past_c
    torch.cuda.empty_cache()

banner("reading the table")
print(f"  ratio {min(ratios):.2f}x - {max(ratios):.2f}x over pure bandwidth")
print(f"  The floor rises with context because the KV cache grows. If your measured time")
print(f"  barely moves with context, something else dominates — check the ratio's slope.")
print(f"  A ratio of 3 or more means launches and unfused elementwise ops, not bandwidth:")
print(f"  that is Module 13 (kernels) and Module 15 (graphs).")

# %% [markdown]
# ## 7. Prefill versus generate
#
# The single most important efficiency fact in LLM serving, and it falls out of arithmetic
# you can do on paper. Measure it, then measure what batching does to it.

# %%
banner("prefill vs generate: the same 512-token answer")
ids_p = torch.randint(0, 30000, (1, 512), device=DEV)

with torch.no_grad():
    t_prefill = bench(lambda: model(ids_p, use_cache=False), iters=10, warmup=3)

    @torch.no_grad()
    def generate(prompt, n):
        cache, cur = None, prompt
        for _ in range(n):
            o = model(cur if cache is None else cur[:, -1:], past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            cur = o.logits[:, -1:].argmax(-1)
        return cache, cur

    cache1, nxt1 = generate(ids_p, 512)
    t_decode = bench(lambda: model(nxt1, past_key_values=cache1, use_cache=True), iters=15)

print(f"  prefill, 512 tokens once              {t_prefill*1000:>9.1f} ms")
print(f"  decode, 512 tokens one at a time      {t_decode*1000*512/1000:>9.0f} ms   ({t_decode*1000:.2f} ms x 512)")
print(f"  ratio                                 {t_decode*512/t_prefill:>9.0f}x")
print(f"\n  Same weights. Same arithmetic. Prefill reads every weight byte once and")
print(f"  reuses it 512 times; decode reads it once per token. That is the whole gap.")
del cache1
torch.cuda.empty_cache()

banner("now add batching: 8 identical sequences")
B = 8
ids_b = ids_p.repeat(B, 1)
with torch.no_grad():
    t_prefill_b = bench(lambda: model(ids_b, use_cache=False), iters=8, warmup=3)
    cache_b, nxt_b = generate(ids_b, 512)
    t_decode_b = bench(lambda: model(nxt_b, past_key_values=cache_b, use_cache=True), iters=12)

agg1, aggB = 1 / t_decode, B / t_decode_b
print(f"  {'':12} {'single':>14} {'batch of 8':>14} {'ratio':>9}")
print(f"  {'prefill':>12} {t_prefill*1000:>11.1f} ms {t_prefill_b*1000:>11.1f} ms {t_prefill_b/t_prefill:>8.2f}x")
print(f"  {'decode/user':>12} {t_decode*1000:>11.2f} ms {t_decode_b*1000:>11.2f} ms {t_decode_b/t_decode:>8.2f}x")
print(f"  {'aggregate':>12} {agg1:>11.1f} t/s {aggB:>11.1f} t/s {aggB/agg1:>8.2f}x")
print(f"\n  Per-user decode got {t_decode_b/t_decode:.2f}x slower. Aggregate throughput got")
print(f"  {aggB/agg1:.2f}x faster. That gap is the entire argument for continuous")
print(f"  batching (Module 07), and you have just measured it on real hardware.")

# %% [markdown]
# ## Write these down
#
# In **My notes**, from your own hardware:
#
# - `derived_params` for both models, and the exact difference from `numel()`
# - the four characters-per-token ratios
# - `kv_per_tok` for TinyLlama, and the context at which KV equals all the weights
# - the relative error of your hand-written block, and which structural checks passed
# - `my_bw`, your measured bandwidth
# - the decode ratio (measured / floor) at 256 and at 8,192 tokens
# - the prefill-vs-generate ratio, and the batch-8 aggregate speed-up
#
# Every memory and scheduling calculation from Module 05 onward uses these numbers.

# %%
print("\nDone. Write your numbers down before you close the notebook.")
