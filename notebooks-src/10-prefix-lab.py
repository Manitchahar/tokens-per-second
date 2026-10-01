# %% [markdown]
# # Lab 10 · Optimize a mixed long-prompt workload
#
# **Tokens per Second** · Module 10
#
# By the end of this notebook you will have measured, on your own GPU, the three
# numbers that decide everything in this module:
#
# 1. **t_prefill_per_token** — how long one more prompt token of prefill costs.
# 2. **t_decode_step** — how long one decode step costs for a batch of users.
# 3. **c_star** — the chunk size at which one chunk of prefill takes exactly as
#    long as one decode step. This is the knob from Lesson 10.1.
#
# Then you will compute the interference a long prompt inflicts on other users,
# and see it confirmed in a small engine loop you write yourself.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# We use **Qwen2.5-0.5B-Instruct** (0.49B parameters) so that everything runs
# quickly on a free T4. The *absolute* times are much smaller than for an 8B
# model; the *ratios* — which is all this module is about — transfer directly.

# %%
import torch, time, math
assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
print(f"{p.name}   {p.total_memory/1e9:.1f} GB   sm_{p.major}{p.minor}")

# %%
# The timing helper from Lab 02. Warm up, use CUDA events, average.
def bench(fn, iters=30, warmup=8):
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

# %% [markdown]
# ## 0. The model
#
# Qwen2.5-0.5B-Instruct: 0.49B parameters, 24 layers, hidden size 896,
# 2 KV heads of head dim 64. That KV shape is unusually narrow, which makes it a
# good model for this lab: decode is strongly memory-bound, exactly as with a
# larger model.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()

cfg = model.config
PARAMS = sum(q.numel() for q in model.parameters())
KV_PER_TOKEN = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 4  # fp32 here
print(f"parameters        {PARAMS/1e9:.3f} B")
print(f"layers            {cfg.num_hidden_layers}")
print(f"hidden            {cfg.hidden_size}")
print(f"KV heads x dim    {cfg.num_key_value_heads} x {cfg.head_dim}")
print(f"KV bytes / token  {KV_PER_TOKEN/1024:.1f} KiB  (fp32)")
print(f"KV per 1024 ctx   {KV_PER_TOKEN*1024/2**20:.1f} MiB")

# %% [markdown]
# ## 1. Prefill: how long does one more token cost?
#
# Prefill is a forward pass over many tokens at once — compute-bound, and
# **linear in token count**. We time `model(ids)` for a prompt of N tokens and
# look at the slope.

# %%
def prefill(n, iters=20):
    ids = torch.randint(1000, 50000, (1, n), device="cuda")
    t = bench(lambda: model(ids), iters=iters)
    return t

print(f"{'tokens':>8} {'prefill ms':>12} {'us/token':>10}  note")
ns = [128, 256, 512, 1024, 2048]
times = {}
for n in ns:
    t = prefill(n)
    times[n] = t
    print(f"{n:>8} {t*1e3:>12.2f} {t*1e6/n:>10.2f}  {'short prompt' if n <= 512 else 'long prompt'}")

# %%
# The slope is the number that matters. Fit it from the two largest points,
# where launch overhead is negligible.
t1k, t2k = times[1024], times[2048]
per_token = (t2k - t1k) / (2048 - 1024)
print(f"\nt_prefill_per_token = {per_token*1e6:.2f} us")
print(f"Linear fit check: prefill({n}) ~= {per_token*n*1e3:.2f} ms at n={n}")
print("Prefill time is linear in prompt length. That is the whole reason a long")
print("prompt hurts: it holds the GPU for a time proportional to its token count.")

# %%
# Compare dtypes: FP32 vs BF16 vs FP16. Fewer bits per weight means more math per byte.
print(f"\n{'dtype':>10} {'prefill 2048 ms':>18} {'us/token':>10}")
for dt, name in [(torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dt).cuda().eval()
    ids = torch.randint(1000, 50000, (1, 2048), device="cuda")
    t = bench(lambda: m(ids), iters=20)
    print(f"{name:>10} {t*1e3:>18.2f} {t*1e6/2048:>10.2f}")
    del m
    torch.cuda.empty_cache()
print("Read this table: on a T4 the gap between fp32 and fp16 is modest, because")
print("the T4 has no fast BF16/FP16 tensor-core path for everything. On an H100 it")
print("is much larger. Quantization (Module 11) is a bigger lever than this table shows.")

# %% [markdown]
# ## 2. Decode: how long does one step cost for N users?
#
# Decode is one token for the whole batch at once. It is memory-bound: every step
# reads all the weights and all the users' KV cache, no matter how many users
# there are. So the step time should be nearly **flat** in batch size — the first
# few dozen users are almost free.

# %%
# A clean, real decode measurement: build an actual cache, then time one
# forward pass of a single new token per sequence against it.
from transformers import DynamicCache

def decode_step(batch, ctx=1024, iters=20):
    """One real decode step: 1 new token per sequence, over a ctx-long cache."""
    past = DynamicCache()
    ids = torch.randint(1000, 50000, (batch, ctx), device="cuda")
    with torch.no_grad():
        model(ids, use_cache=True, past_key_values=past)  # warm the cache
    new = torch.randint(1000, 50000, (batch, 1), device="cuda")
    def step():
        with torch.no_grad():
            model(new, past_key_values=past, use_cache=True)
    return bench(step, iters=iters)

print(f"{'batch':>7} {'decode step ms':>16} {'us/user':>10} {'tokens/s total':>16}")
dec = {}
for b in [1, 8, 32, 64, 128]:
    t = decode_step(b)
    dec[b] = t
    print(f"{b:>7} {t*1e3:>16.2f} {t*1e6/b:>10.1f} {b/t:>16.1f}")

# %%
# The key observation: per-user time rises, but total throughput keeps climbing,
# and the step itself is nearly flat up to ~64 users. That flatness is the
# arithmetic-intensity argument from Lab 02 showing up in a real decoder.
gain = (64/dec[64]) / (1/dec[1])
print(f"\nGoing from 1 user to 64 users multiplies total throughput by {gain:.1f}x,")
print(f"while the step itself only grew {dec[64]/dec[1]:.2f}x.")

# %%
# And decode slows as context grows, because the KV cache it must read grows.
print(f"\n{'ctx':>7} {'decode(32) ms':>15} {'KV read MB':>12}")
for ctx in [256, 1024, 4096]:
    t = decode_step(32, ctx=ctx, iters=10)
    kv_bytes = 32 * ctx * KV_PER_TOKEN
    print(f"{ctx:>7} {t*1e3:>15.2f} {kv_bytes/2**20:>12.1f}")

# %% [markdown]
# ## 3. The crossover chunk size, measured
#
# Lesson 10.1 said: size the chunk so that one chunk of prefill takes as long as
# one decode step. That chunk size is
#
# ```
# c_star = t_decode_step / t_prefill_per_token
# ```
#
# Measure both sides and divide. This is the number you deploy.

# %%
t_decode = dec[32]           # a decode step for 32 users at 1024 context
c_star = t_decode / per_token
print(f"t_decode_step        {t_decode*1e3:8.2f} ms   (32 users, 1024 ctx)")
print(f"t_prefill_per_token  {per_token*1e6:8.2f} us")
print(f"c_star               {c_star:8.0f} tokens   <-- the knob")

# %%
# Verify: prefill a chunk of c_star tokens and confirm it takes ~one decode step.
c = int(round(c_star))
t_chunk = prefill(max(128, c), iters=20)
print(f"\nprefill {c} tokens: {t_chunk*1e3:.2f} ms   vs decode step {t_decode*1e3:.2f} ms")
print(f"ratio {t_chunk/t_decode:.2f}  (should be near 1.0)")

# %%
# Sweep it. Chunked prefill: total time for an 8192-token prompt, chunk by chunk.
# Each chunk is its own forward pass, so the overhead per chunk is visible.
print(f"\n{'chunk':>7} {'steps':>7} {'total ms':>10} {'us/1k tok':>11} {'note'}")
for ch in [128, 256, c_star, 512, 1024, 2048, 8192]:
    ch = int(ch)
    n_chunks = math.ceil(8192 / ch)
    # simulate: each chunk is a real prefill of that size, plus one decode step
    t_chunk_real = prefill(min(ch, 2048), iters=15)
    total = n_chunks * max(t_chunk_real, t_decode)
    note = "one shot (no chunking)" if ch >= 8192 else ("~1 decode step" if abs(ch-c_star) < c_star*0.4 else ("finer than needed" if ch < c_star else "coarser than needed"))
    print(f"{ch:>7} {n_chunks:>7} {total*1e3:>10.1f} {total/8.192:>11.1f}  {note}")

# %%
# Read the table. Total time barely changes from c_star to 8192 — chunking does
# not make prefill *slower* in any meaningful way. What it buys is that no single
# step is longer than a decode step. That is the whole trade: a tiny cost in
# total time, a huge reduction in the worst-case wait for other users.

# %% [markdown]
# ## 4. The interference, computed
#
# Now compute the damage an 8,192-token prompt does to everyone else, using the
# constants you measured. This is the table that should make you wince.

# %%
print(f"{'prompt':>8} {'prefill ms':>11} {'stall ms':>10} {'other users TPOT':>17} {'slowdown':>9}")
for n in [512, 2048, 8192, 32768]:
    tp = per_token * n
    stall = max(0, tp - t_decode)
    others_tpot = t_decode + stall
    slowdown = others_tpot / t_decode
    print(f"{n:>8} {tp*1e3:>11.1f} {stall*1e3:>10.1f} {others_tpot*1e3:>17.1f} {slowdown:>8.1f}x")

# %%
# With chunking at c_star, the stall is zero: every chunk fits inside a decode step.
print(f"\nWith chunking at c_star={c:.0f}: stall per step = "
      f"{max(0, per_token*c - t_decode)*1e3:.1f} ms  (down from "
      f"{max(0, per_token*8192 - t_decode)*1e3:.0f} ms unchunked)")

# %% [markdown]
# ## 5. Prefix caching: reuse the same prefix
#
# Run the *same* 512-token prefix twice. The second time, if you had a cache,
# you would prefill only the tokens after it. Here we can only demonstrate the
# principle by timing prefix-only vs full-prompt work — in a real engine the
# cached tokens are simply not recomputed.

# %%
SYS = 512   # a shared "system prompt" every request carries
uniq = 500  # the user's actual message

t_sys = prefill(SYS)
t_full = prefill(SYS + uniq)
print(f"prefill {SYS} token shared prefix:  {t_sys*1e3:7.2f} ms")
print(f"prefill {SYS+uniq} token full prompt: {t_full*1e3:7.2f} ms")
print(f"prefill only the {uniq} unique tokens: {prefill(uniq)*1e3:7.2f} ms")
print(f"\nCached-prefix saving: {(t_full - prefill(uniq))/t_full*100:.0f}% of prefill")
print("In a real engine the second request prefills only `uniq` tokens, because")
print("the first `SYS` tokens are already in the KV cache. Hit rate = SYS/(SYS+uniq)")

# %%
# The KV cache memory that prefix occupies. On a T4 with 16 GB and a small model,
# this is affordable; on an 80 GB H100 with an 8B model it is the real constraint.
print(f"\nKV cache for the {SYS}-token prefix: {KV_PER_TOKEN*SYS/2**20:.1f} MiB")
print(f"KV cache for a 32k context:          {KV_PER_TOKEN*32768/2**20:.1f} MiB")
print("At 32k context each single request holds a lot of KV. With many concurrent")
print("requests this is what limits how many you can serve at once.")

# %% [markdown]
# ## 6. A tiny engine loop
#
# Build the loop from the lesson — one engine step carries one chunk of prefill
# *and* one decode token for every active user, and the step lasts as long as the
# slower of the two. Then watch what a long prompt does to everyone else. This
# is the simulator from the lesson, rebuilt from your own measured constants.

# %%
def run(steps=400, chunk=None, long_prompt_at=None, long_tokens=8192, users=32):
    """A miniature engine. Returns per-step who was blocked."""
    if chunk is None:
        chunk = int(c_star)
    blocked = 0
    stall_ms = 0.0
    for step in range(steps):
        has_long = (long_prompt_at is not None and step == long_prompt_at)
        prefill_tokens = min(chunk, long_tokens) if has_long else 0
        t_pre = per_token * prefill_tokens
        t_dec = t_decode
        # one engine step = the slower of the two
        step_time = max(t_pre, t_dec)
        # if prefill exceeded a decode step, users waited the difference
        stall = max(0, t_pre - t_dec)
        if stall > 0:
            blocked += 1
            stall_ms += stall * 1e3
    return blocked, stall_ms

print(f"{'config':>26} {'steps blocked':>14} {'total stall ms':>15}")
blocked, stall = run(long_prompt_at=100, chunk=8192)
print(f"{'no chunking (8192 at once)':>26} {blocked:>14} {stall:>15.0f}")
blocked, stall = run(long_prompt_at=100, chunk=int(c_star))
print(f"{'chunked at c_star':>26} {blocked:>14} {stall:>15.0f}")
blocked, stall = run(long_prompt_at=100, chunk=256)
print(f"{'chunked at 256':>26} {blocked:>14} {stall:>15.0f}")

# %%
# Unchunked, one long prompt blocks a whole step (or several). Chunked at c_star,
# it blocks nothing: every chunk fits inside a decode step. This is the entire
# argument of Lesson 10.1, on your own GPU, with your own numbers.

# %% [markdown]
# ## 7. Compute what a big model would do
#
# Everything above used a 0.5B model on a T4. Here is the same calculation for
# the models and GPUs you will actually deploy, so you can size the chunk knob
# for real hardware. Prefill time scales linearly with parameter count; decode
# time scales with weights *and* KV bytes.

# %%
GPUS = {
    "T4":  dict(bw=320e9,  flops=65e12),
    "A100": dict(bw=2.0e12, flops=312e12),
    "H100": dict(bw=3.35e12, flops=989e12),
}
MODELS = {
    "0.5B": dict(p=0.49e9, kv=128*1024),   # measured above (fp32); use ~this for ratio
    "8B":   dict(p=8.03e9, kv=131072),
    "70B":  dict(p=70.6e9, kv=327680),
}

def estimate(model_name, gpu_name, batch, ctx):
    P = MODELS[model_name]["p"]; g = GPUS[gpu_name]
    kv_per_tok = MODELS[model_name]["kv"]
    # decode step: read weights (bf16) + KV, at ~80% of peak bandwidth
    bytes = P*2 + batch*ctx*kv_per_tok
    t_dec = bytes / (g["bw"]*0.8)
    # prefill per token: 2*P FLOP at ~55% MFU
    t_pre_tok = 2*P / (g["flops"]*0.55)
    c_star = t_dec / t_pre_tok
    return t_dec, t_pre_tok, c_star

print(f"{'model':>6} {'gpu':>6} {'batch':>6} {'ctx':>6} {'decode ms':>10} {'us/tok':>8} {'c_star':>8}")
for m in MODELS:
    for gname in GPUS:
        td, tpt, cs = estimate(m, gname, 32, 1024)
        print(f"{m:>6} {gname:>6} {32:>6} {1024:>6} {td*1e3:>10.2f} {tpt*1e6:>8.2f} {cs:>8.0f}")

# %%
# The trend: c_star grows with batch and context (decode step gets slower as more
# users and longer contexts must be read) and shrinks with model size (prefill
# per token gets more expensive). On an H100 with a big model the knob lands in
# the hundreds-to-low-thousands of tokens, which is why engines default to
# max_num_batched_tokens in that range.

# %% [markdown]
# ## 8. Your turn
#
# 1. **Section 1**: try prompt lengths 4096 and 8192. Does `per_token` stay constant? It should — prefill is linear. If it grows, you are becoming memory-bound on the KV write.
# 2. **Section 2**: decode_step with batch 256. Where does the flatness break? That batch size, times context, is roughly where you run out of KV memory on a real engine.
# 3. **Section 3**: sweep the chunk size and find where total time starts to rise noticeably. Below that, chunking is free.
# 4. **Section 4**: for the 32k prompt, compute how long one innocent user waits. Compare to their normal TPOT. Multiply by 31 other users to get "total user-seconds wasted" — that is the real cost of a cliff.
# 5. **Section 5**: what hit rate do you need to cut prefill time in half? (Hint: it is 50%, but say what that means in tokens for a 2k prompt.)
# 6. **Section 7**: find the model/gpu/batch combination where c_star is largest. That is where you must chunk most aggressively.
#
# **Write in your notes:** `t_prefill_per_token`, `t_decode_step`, and `c_star`. These
# are the three numbers that set every knob in Module 10 for your hardware.

# %%
print("\n=== Your measured constants ===")
print(f"t_prefill_per_token  = {per_token*1e6:.2f} us")
print(f"t_decode_step (b=32) = {t_decode*1e3:.2f} ms")
print(f"c_star               = {c_star:.0f} tokens")
print(f"prefill(8192)        = {per_token*8192*1e3:.1f} ms")
print(f"stall to 31 others   = {max(0, per_token*8192 - t_decode)*1e3:.0f} ms each")