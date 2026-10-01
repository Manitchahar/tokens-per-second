# %% [markdown]
# # Lab 05 · The KV cache, measured
#
# **Tokens per Second** · Module 05
#
# Two questions, both answerable on the free Colab T4 in about ten minutes:
#
# 1. **Does the cache actually pay for itself?** Time generation with `use_cache=True`
#    against `use_cache=False`, and see the ratio grow with the prompt length.
# 2. **Will it fit?** Measure the cache's real bytes per token on this GPU, then work out
#    how many conversations this 16 GB card can hold at your context length.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# **Write every number into your notes.** Modules 08, 10, 11, 20 and 25 all do arithmetic
# with them, and none of them can do it for you.

# %% [markdown]
# ## 0. What am I running on?
#
# The T4 is 16 GB and about 320 GB/s. It has FP16 tensor cores but no BF16 ones, so
# everything here is `float16`. Say what you expect before you run it.

# %%
import math, time, gc, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
total_bytes, free_bytes = torch.cuda.mem_get_info()
print(f"GPU            {p.name}")
print(f"memory         {p.total_memory/1e9:.1f} GB nominal, {free_bytes/1e9:.2f} GB free right now")
print(f"torch          {torch.__version__}")
print(f"dtype          float16 (the T4 has no BF16 tensor cores)")

# %%
# A timing helper you can trust: warm up, use CUDA events, average.
def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / iters      # seconds per call

# %% [markdown]
# ## 1. The model, and the numbers that follow from its shape
#
# Qwen2.5-0.5B-Instruct is small enough for a T4 and real enough to be worth measuring.
# Three config fields decide the entire KV cache story: the number of layers, the number
# of **key/value** heads, and the head dimension.

# %%
NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float16).to("cuda").eval()

cfg = model.config
L, H, HKV, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
print(f"layers {L}   query heads {H}   kv heads {HKV}   head dim {D}   group {H // HKV}")

# %%
BYTES = 2                                    # float16
kv_per_token = 2 * L * HKV * D * BYTES       # K and V, every layer, one token
weights = sum(p.numel() for p in model.parameters()) * 2

print(f"\nKV bytes per token of context   {kv_per_token:>12,} B  = {kv_per_token/1024:.0f} KB")
print(f"weights in float16              {weights/1e9:>12.2f} GB")
for ctx in (512, 1024, 4096, 8192, 32768):
    print(f"  one conversation at {ctx:>6,} tokens: {ctx*kv_per_token/1e6:8.1f} MB")
print("\nNothing above is a guess. It is the shape of the model, multiplied out.")

# %% [markdown]
# ## 2. Does the cache pay for itself?
#
# The same prompt, the same number of new tokens, generated two ways. The uncached run
# re-runs the whole forward pass on the whole sequence at every step.
#
# **Expected:** the speed-up starts near 1 at the shortest prompt and climbs to roughly
# 15–20× by the longest one. At the shortest prompt the uncached run can even come out
# *ahead*: the cached version pays its per-step overhead 32 times over, while the uncached
# one gets a 128-token batched matrix multiply for free. That is not a bug — arithmetic is
# simply not the scarce resource at short context. The token-forward ratios printed below
# should sit near 30× and almost flat, because the prompt dwarfs the 32 output tokens.

# %%
PROMPTS = {n: ("The KV cache stores keys and values. " * n) for n in (16, 64, 256, 1024)}
NEW = 32

def generate(prompt, use_cache, new_tokens=NEW):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(
        **ids, max_new_tokens=new_tokens, do_sample=False,
        use_cache=use_cache, pad_token_id=tok.eos_token_id,
    )
    return out.shape[1] - ids.input_ids.shape[1]     # tokens actually produced

print(f"{'prompt tokens':>15} {'generated':>10} {'cached (ms)':>13} {'uncached (ms)':>14} {'speed-up':>9}")
for n, prompt in PROMPTS.items():
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    T = ids.input_ids.shape[1]
    got = generate(prompt, True)
    t_cached = bench(lambda: generate(prompt, True), iters=5, warmup=2)
    t_plain = bench(lambda: generate(prompt, False), iters=3, warmup=1)
    print(f"{T:>15,} {got:>10} {t_cached*1e3:>13.1f} {t_plain*1e3:>14.1f} {t_plain/t_cached:>8.1f}x")

# %%
# Why: count the token-forwards, exactly as in lesson 05.1.
def forwards_naive(p, n):  return n * p + n * (n - 1) / 2
def forwards_cached(p, n):  return p + n

print(f"\n{'prompt':>8} {'naive forwards':>16} {'cached forwards':>16} {'ratio':>8}")
for n, prompt in PROMPTS.items():
    T = tok(prompt, return_tensors="pt").input_ids.shape[1]
    a, c = forwards_naive(T, NEW), forwards_cached(T, NEW)
    print(f"{T:>8,} {a:>16,.0f} {c:>16,} {a/c:>7.0f}x")

print("\nThe speed-up column is smaller than the forwards ratio on purpose:")
print("recomputing turns a memory-bound step into a compute-bound one, and this T4")
print("has far more arithmetic than bandwidth. Compare your two numbers.")

# %% [markdown]
# ## 3. Prefill versus decode, on your own GPU
#
# Time one prefill pass over the prompt, then time a single decode step. The ratio should
# be in the hundreds, because prefill amortises one read of the weights over every prompt
# token while decode uses each weight exactly once.

# %%
ids = tok(PROMPTS[256], return_tensors="pt").input_ids.to("cuda")
T = ids.input_ids.shape[1]

with torch.no_grad():
    t_prefill = bench(lambda: model(**ids, use_cache=True), iters=20)

    # Time ONE decode step, with a fresh cache each time. Handing the same cache object
    # to the model in a loop would append a row every iteration and quietly change the
    # measurement, so the prefill sits outside the timed region.
    def decode_once(iters=20, warmup=3):
        for _ in range(warmup):
            o = model(input_ids=ids, use_cache=True)
            model(input_ids=o.logits[:, -1:].argmax(-1), past_key_values=o.past_key_values, use_cache=True)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            o = model(input_ids=ids, use_cache=True)
            nxt = o.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=nxt, past_key_values=o.past_key_values, use_cache=True)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sum(ts) / len(ts)

    t_decode = decode_once()

print(f"prompt {T} tokens")
print(f"prefill   {t_prefill*1e3:7.2f} ms   ->  {T/t_prefill:8,.0f} tokens/s")
print(f"decode    {t_decode*1e3:7.2f} ms   ->  {1/t_decode:8,.0f} tokens/s for one user")
print(f"\nper token: prefill {(t_prefill/T)*1e6:6.0f} us   decode {t_decode*1e3:6.2f} ms"
      f"   ->  decode is {t_decode/(t_prefill/T):,.0f}x more expensive per token")

# %%
# The decode step should be dominated by reading the weights. Check it.
w_gb = weights / 1e9
t_floor = w_gb * 1e9 / (320e9 * 0.8)      # 16 GB-ish at 80% of the T4's ~320 GB/s
print(f"\nweights              {w_gb:.2f} GB")
print(f"floor, weights only  {t_floor*1e3:6.2f} ms at 80% of peak bandwidth")
print(f"your decode step     {t_decode*1e3:6.2f} ms")
print(f"efficiency           {t_floor/t_decode*100:6.0f}% of the roofline")

# %% [markdown]
# ## 4. Watch the cache read grow with the context
#
# Build a cache once, then time a decode step at increasing lengths. If the cache is being
# read properly, the time should grow roughly **linearly**, and the slope should be
# `kv_per_token / bandwidth`.
#
# **Expected:** flat-ish below ~1K tokens (the weights dominate), then a steady climb.

# %%
def build_cache(n_tokens, dtype=torch.float16):
    """A cache of exactly the shape this model would use, filled with zeros."""
    return torch.zeros(L, 2, HKV, n_tokens, D, dtype=dtype, device="cuda")

def decode_step_at(n_tokens, iters=30):
    """Time one attention read against a cache of length n_tokens. The weights are constant."""
    cache = build_cache(n_tokens)
    # One query head per KEY/VALUE head: in the real model the query heads of a group are
    # all reading this same K, so the cost of the read is set by the cache, not by H.
    q = torch.randn(1, HKV, 1, D, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        def step():
            K = cache[0, 0].unsqueeze(0)                     # [1, HKV, n, D]
            scores = torch.einsum("bhd,bhsd->bhs", q, K)     # [1, HKV, 1, n]
            return torch.softmax(scores.float(), -1)
        t = bench(step, iters=iters, warmup=5)
    del cache
    torch.cuda.empty_cache()
    return t

print(f"{'context':>10} {'cache read':>12} {'measured':>11} {'expected':>11} {'eff':>7}")
for n in (256, 512, 1024, 2048, 4096, 8192, 16384):
    if n * kv_per_token > 3e9:
        break
    t = decode_step_at(n)
    read_gb = n * kv_per_token / 1e9
    exp = read_gb * 1e9 / (320e9 * 0.8)
    print(f"{n:>10,} {read_gb*1e3:>10.0f} MB {t*1e6:>9.0f} us {exp*1e6:>9.0f} us {exp/t*100:>6.0f}%")

# %% [markdown]
# ## 5. Measure the cache instead of trusting the formula
#
# Memory is the number that decides how many conversations you can serve, so measure it
# rather than deriving it. We do it two ways and check they agree.

# %%
def peak_delta(fn):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return peak - base

def run(use_cache):
    def f():
        with torch.no_grad():
            return model(**ids, use_cache=use_cache)
    return f

m_cached = peak_delta(run(True))
m_plain = peak_delta(run(False))
print(f"peak memory delta, use_cache=True   {m_cached/1e6:8.1f} MB")
print(f"peak memory delta, use_cache=False  {m_plain/1e6:8.1f} MB")
print(f"difference                          {(m_cached-m_plain)/1e6:8.1f} MB")
print(f"analytic cache size                 {T*kv_per_token/1e6:8.1f} MB  ({T} tokens)")

# %%
# The big term in both numbers is the logits tensor: prompt_length x vocab_size x 2 bytes.
# That is why long prompts run out of memory long before the cache does.
vocab = model.config.vocab_size
print(f"\nlogits tensor at this prompt: {T:,} x {vocab:,} x 2 B = {T*vocab*2/1e9:.2f} GB")
print(f"KV cache at this prompt:                  {T*kv_per_token/1e6:.1f} MB")
print("Rule of thumb: on a big-vocabulary model, the logits dominate prefill memory.")
print("Newer Transformers versions can skip all but the last few logits positions.")

# %%
# Now measure the cache tensor directly, so the number is exact.
with torch.no_grad():
    out = model(**ids, use_cache=True)
pkv = out.past_key_values

def cache_bytes(p):
    total = 0
    try:
        if isinstance(p, tuple):                       # legacy tuple of (k, v) per layer
            for k, v in p:
                total += k.numel() * k.element_size() + v.numel() * v.element_size()
        else:                                           # Cache objects
            for layer in p.layers:
                total += layer.keys.numel() * layer.keys.element_size()
                total += layer.values.numel() * layer.values.element_size()
    except Exception as exc:
        print("could not walk the cache object:", exc)
    return total

measured = cache_bytes(pkv)
print(f"\nmeasured cache                     {measured/1e6:8.1f} MB")
print(f"bytes per token, measured           {measured/max(T,1):>10,.0f} B")
print(f"bytes per token, from the config    {kv_per_token:>10,} B")
del out, pkv
torch.cuda.empty_cache()

# %% [markdown]
# ## 6. How many conversations fit on this T4?
#
# Free memory, minus the model, divided by the cache per conversation. This is the
# division from lesson 05.4, with numbers that came off your own hardware.

# %%
def budget(ctx, kv_dtype_bytes=BYTES, headroom=0.10, workspace_frac=0.04):
    per_token = 2 * L * HKV * D * kv_dtype_bytes
    free = (free_bytes - weights - workspace_frac * p.total_memory) * (1 - headroom)
    per_seq = ctx * per_token
    return {
        "per_token_KB":  per_token / 1024,
        "per_seq_MB":    per_seq / 1e6,
        "seqs":          int(max(0, free) // per_seq) if per_seq else 0,
        "tokens_total":  int(free // per_token),
    }

print(f"free at start {free_bytes/1e9:.1f} GB, weights {weights/1e9:.2f} GB")
print(f"{'context':>9} {'KB/token':>9} {'per seq':>10} {'conversations':>14} {'total tokens':>14}")
for ctx in (512, 2048, 8192, 32768):
    r = budget(ctx)
    print(f"{ctx:>9,} {r['per_token_KB']:>9.0f} {r['per_seq_MB']:>9.0f} MB {r['seqs']:>14,} {r['tokens_total']:>14,}")

# %%
# Prove it: allocate that many cache slots and see whether it survives.
def try_alloc(n_seqs, ctx, dtype=torch.float16):
    per = 2 * L * HKV * D * dtype.itemsize
    try:
        buf = torch.empty(n_seqs, L, 2, HKV, ctx, D, dtype=dtype, device="cuda")
        n = buf.numel() * buf.element_size()
        del buf
        torch.cuda.empty_cache()
        return True, n / 1e9
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, 0.0

for ctx in (1024, 8192):
    n = budget(ctx)["seqs"]
    ok, gb = try_alloc(max(0, n - 2), ctx)      # leave a small margin
    print(f"context {ctx:>6,}: predicted {n:,} conversations -> "
          f"allocating {max(0, n-2):,} used {gb:.1f} GB and {'SUCCEEDED' if ok else 'FAILED'}")

print("\nThe prediction and the allocation should agree to within a few percent.")
print("Any gap is the workspace reserve, which the prediction is guessing.")

# %% [markdown]
# ## 7. Halve it: the same cache in 8 bits
#
# The cache format is an independent choice from the weight format. Storing K and V in
# FP8 or INT8 halves the biggest consumer of memory, exactly. (We are measuring memory
# here, not quality — Module 11 is where you find out what it costs you in output quality.)

# %%
ctx = 8192
for name, dt in (("float16", torch.float16), ("int8", torch.int8)):
    per = 2 * L * HKV * D * dt.itemsize
    print(f"{name:<9} {per/1024:6.0f} KB/token   {ctx*per/1e6:8.0f} MB per conversation"
          f"   {budget(ctx, dt.itemsize)['seqs']:>7,} conversations")

# %%
# The same trade at model scale, with Llama-3.1-8B's shape (128 KB/token in BF16).
print("\nFor comparison, Llama-3.1-8B in BF16 is 128 KB/token, not "
      f"{kv_per_token/1024:.0f} KB — the difference is layers (32) and KV heads (8).")
print("A 16 GB T4 therefore has room for about 50 conversations of 8K context in FP8,")
print("if the weights fit at all, which in BF16 they do not.")

# %% [markdown]
# ## 8. Share a prefix
#
# Every request that begins with the same system prompt can reuse the rows for that
# prompt instead of writing them again. This is the whole idea behind prefix caching,
# and it is free to try by hand: just do not prefill the shared part twice.

# %%
SHARED = "You are a careful, concise assistant. " * 200      # a system prompt everyone sends
UNIQUE = "Now explain the KV cache. " * 20                  # the part that is per request

shared_n = len(tok(SHARED).input_ids)
unique_n = len(tok(UNIQUE).input_ids)

def naive(n_requests):
    """Each request stores its own copy of the shared prefix."""
    return n_requests * (shared_n + unique_n) * kv_per_token

def shared(n_requests):
    """The shared prefix is stored once; only the unique tail is per request."""
    return (shared_n + n_requests * unique_n) * kv_per_token

print(f"shared prompt {shared_n:,} tokens ({shared_n*kv_per_token/1e6:.0f} MB)"
      f"   per-request tail {unique_n} tokens\n")
print(f"{'requests':>9} {'no sharing':>14} {'with sharing':>14} {'saved':>12} {'x cheaper':>10}")
for n in (1, 10, 50, 200):
    a, b = naive(n), shared(n)
    print(f"{n:>9,} {a/1e9:>13.2f} G {b/1e9:>13.2f} G {(a-b)/1e9:>11.2f} G {a/b:>9.1f}x")

# %% [markdown]
# ## 9. Verify the cache is not lying to you
#
# The test from lesson 05.5, on the real model. Cached and uncached logits must agree to
# floating-point noise. If they do not, something is wrong with the cache — and the most
# common cause is a cache layout that disagrees with what attention expects.

# %%
with torch.no_grad():
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")

    a     = model(input_ids=ids, use_cache=True).logits[:, -1]       # cached prefill
    a_ref = model(input_ids=ids, use_cache=False).logits[:, -1]      # same five tokens, no cache

    nxt   = a.argmax(-1, keepdim=True)                               # a real, greedy token
    cache = model(input_ids=ids, use_cache=True).past_key_values
    b     = model(input_ids=nxt, past_key_values=cache, use_cache=True).logits[:, -1]   # cached step
    c     = model(input_ids=torch.cat([ids, nxt], 1), use_cache=False).logits[:, -1]    # all six, no cache

print("prefill,    cached vs uncached, max |diff|:", f"{(a - a_ref).abs().max().item():.3e}")
print("decode step, cached vs uncached, max |diff|:", f"{(b - c).abs().max().item():.3e}")
print("same predicted token:", torch.equal(b.argmax(-1), c.argmax(-1)),
      "->", repr(tok.decode(c.argmax(-1))))
print("\nThe prefill difference is exactly zero: a cache changes where things are stored,")
print("not what is computed. The decode difference is floating-point noise, because the")
print("attention sums are added in a different order.")
print("\nExpect something like 1e-3 in float16. If you see 1e-1 or more, the cache is wrong.")

# %% [markdown]
# ## Your turn
#
# 1. **Section 2**: replace `NEW = 32` with `NEW = 1` and `NEW = 256`. At what prompt
#    length does the uncached run stop being usable at all? Extrapolate from the forwards
#    ratio in the table.
# 2. **Section 3**: change the batch by replicating `ids` 8 times. How much does the decode
#    step grow, and why is it not 8×? (Hint: the weights are read once for all 8 rows.)
# 3. **Section 4**: fit a straight line to your measured times against cache bytes. What is
#    the slope, and what bandwidth does it imply? Compare with the T4's 320 GB/s.
# 4. **Section 6**: set `headroom=0.0` and `workspace_frac=0.0`. Where does the prediction
#    start to fail, and what error does the runtime give you?
# 5. **Section 7**: try `torch.float8_e4m3fn` for the cache. Does the T4 support it? What
#    happens to the size?
# 6. **Write in your notes:** `my_kv_bytes_per_token`, `my_weights_bytes`, the cached vs
#    uncached speed-up at a 1,024-token prompt, the decode step time at 8,192 tokens of
#    context, and the number of 4,096-token conversations this GPU holds.
#    Modules 08, 10, 11, 20 and 25 all start from those five numbers.

# %%
print("done. notes:")
print(f"  my_kv_bytes_per_token = {kv_per_token:,} B")
print(f"  my_weights_bytes      = {weights/1e9:.2f} GB")
print(f"  conversations @4K     = {budget(4096)['seqs']:,}")
