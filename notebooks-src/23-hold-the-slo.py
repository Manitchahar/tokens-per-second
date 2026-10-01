# %% [markdown]
# # Lab 23 · Hold an SLO under changing traffic
#
# **Tokens per Second** · Module 23
#
# This is the lab for the whole module. You have three levers — **which replica**
# gets a request, **how many** replicas exist, and **how fast** traffic arrives —
# and one promise to keep: p99 time-to-first-token under a second, and a token
# every 50 ms or faster.
#
# By the end you will have:
#
# 1. Measured the per-replica capacity that the Little's-law formula predicts.
# 2. Swept the arrival rate to find the cliff where goodput collapses.
# 3. Compared routing policies on a real workload, not on intuition.
# 4. Produced a capacity plan: replicas, rate, and the alert you would page on.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# We use **Qwen2.5-0.5B-Instruct** (0.49B parameters) so it fits a free T4 with
# room for many "replicas" at once. The *absolute* numbers are far smaller than
# an 8B model on an H100; the *ratios* and the *shape of the curve* are the same,
# because both scale the same way with parameter count.

# %%
import torch, time, math, random
assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
print(f"{p.name}   {p.total_memory/1e9:.1f} GB   sm_{p.major}{p.minor}")

# %%
# The timing helper from Lab 02: warm up, CUDA events, average.
def bench(fn, iters=20, warmup=6):
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
# ## 0. The model and its constants
#
# Llama-3.1-8B costs 131,072 bytes of KV cache per token (Module 05). Qwen2.5-0.5B
# has a much narrower KV shape, so we compute its own. Every number below is
# derived from these constants — nothing is assumed.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

cfg = model.config
PARAMS = sum(q.numel() for q in model.parameters())
KV_PER_TOKEN = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2  # fp16
TOTAL_MEM = p.total_memory
# Usable KV pool: total memory minus weights and a rough reserve for workspace.
KV_POOL_TOKENS = int((TOTAL_MEM * 0.85 - PARAMS * 2) / KV_PER_TOKEN)

print(f"parameters        {PARAMS/1e9:.3f} B")
print(f"KV bytes / token  {KV_PER_TOKEN/1024:.1f} KiB")
print(f"GPU memory        {TOTAL_MEM/1e9:.1f} GB")
print(f"KV pool (tokens)  {KV_POOL_TOKENS:,}")

# %% [markdown]
# ## 1. Measure the two costs
#
# Prefill is linear in prompt length and compute-bound. Decode is memory-bound and
# nearly flat in batch size. Those two facts set every knob in this module, so
# measure them before modelling anything.

# %%
def prefill(n, iters=15):
    ids = torch.randint(1000, 50000, (1, n), device="cuda")
    with torch.no_grad():
        return bench(lambda: model(ids), iters=iters)

print(f"{'tokens':>8} {'prefill ms':>12} {'us/token':>10}")
times = {}
for n in [256, 512, 1024, 2048]:
    t = prefill(n)
    times[n] = t
    print(f"{n:>8} {t*1e3:>12.2f} {t*1e6/n:>10.2f}")

per_token = (times[2048] - times[1024]) / (2048 - 1024)
print(f"\nt_prefill_per_token = {per_token*1e6:.2f} us   (the slope, not the total)")

# %%
def decode_step(batch, ctx=512, iters=15):
    past = DynamicCache()
    ids = torch.randint(1000, 50000, (batch, ctx), device="cuda")
    with torch.no_grad():
        model(ids, use_cache=True, past_key_values=past)
    new = torch.randint(1000, 50000, (batch, 1), device="cuda")
    def step():
        with torch.no_grad():
            model(new, past_key_values=past, use_cache=True)
    return bench(step, iters=iters)

print(f"{'batch':>7} {'decode step ms':>16} {'tok/s total':>14} {'us/user':>9}")
dec = {}
for b in [1, 8, 32, 64, 128]:
    t = decode_step(b)
    dec[b] = t
    print(f"{b:>7} {t*1e3:>16.2f} {b/t:>14.1f} {t*1e6/b:>9.1f}")

t_decode = dec[32]
print(f"\nThe step itself is nearly flat while total throughput climbs — this is the")
print(f"arithmetic-intensity argument from Module 02 appearing in a real decoder.")

# %% [markdown]
# ## 2. The capacity formula, and check it against measurement
#
# Lesson 23.1 said: users per replica = KV pool ÷ context length, and
# requests per second per replica = users ÷ time in system.
#
# ```
# users_per_replica = KV_pool_tokens / context
# time_in_system   = out_tokens * TPOT + TTFT_budget
# req/s per replica = users_per_replica / time_in_system
# ```

# %%
CTX        = 512          # mean context per user (prompt + generated)
OUT        = 250          # output tokens per request
TPOT_SLO   = 0.05         # 50 ms SLO
TTFT_SLO   = 1.0          # 1 s SLO

users_per_replica = KV_POOL_TOKENS / CTX
time_in_system    = OUT * TPOT_SLO + TTFT_SLO
req_per_replica   = users_per_replica / time_in_system

print(f"KV pool            {KV_POOL_TOKENS:,} tokens")
print(f"context            {CTX} tokens")
print(f"users per replica  {users_per_replica:,.0f}")
print(f"time in system     {time_in_system:.1f} s  ({OUT} tok x {TPOT_SLO*1000:.0f}ms + {TTFT_SLO}s)")
print(f"=> capacity        {req_per_replica:.1f} req/s per replica")
print()
print("Sanity check against the measured decode step:")
print(f"  {users_per_replica:.0f} users in a batch, step takes {t_decode*1e3:.1f} ms")
print(f"  so TPOT would be {t_decode*1e3:.0f} ms, versus the {TPOT_SLO*1000:.0f} ms SLO")
print("  The KV pool allows a far bigger batch than the SLO permits, so on a real")
print("  deployment you cap the batch for latency and leave memory unused — capacity")
print("  is set by the batch you allow, not by all the memory you could fill.")

# %% [markdown]
# ## 3. The cliff: sweep the arrival rate
#
# Build a miniature version of the fleet simulator: a queue of requests arriving at
# a fixed rate, N replicas, and a routing policy. Measure goodput — the fraction of
# requests that met both SLOs — and find where it collapses.

# %%
SLO_TTFT, SLO_TPOT = 1.0, 0.05

def make_batch(n, ctx):
    past = DynamicCache()
    ids = torch.randint(1000, 50000, (n, ctx), device="cuda")
    with torch.no_grad():
        model(ids, use_cache=True, past_key_values=past)
    return past

# The GPU can only run one batch at a time, so rather than pretend to serve four
# replicas simultaneously we measure the two things the fleet maths actually needs:
#   1. the cost of one decode step  (t_decode, above)
#   2. the cost of one prefill chunk (per_token, above)
# and then model the queue that forms around them. That queue is the whole story
# of routing and autoscaling: it is where TTFT goes to die.

def run_fleet(rate, replicas, ctx=CTX, out_tokens=OUT, seed=0, horizon=8.0):
    """Model `replicas` engines fed at `rate` req/s. Returns goodput and percentiles.

    Each replica retires requests at the SLO: its batch streams one token per step,
    so a request of `out_tokens` occupies its slot for out_tokens * t_decode. When
    the offered rate exceeds that, the excess queues and TTFT grows without bound.
    """
    rnd = random.Random(seed)
    # Capacity of one replica at this context, in requests per second.
    kv_per_user = ctx * KV_PER_TOKEN
    users = max(1, KV_POOL_TOKENS // kv_per_user)
    # A user streams at one token per decode step.
    t_per_request = out_tokens * t_decode
    cap_per_replica = users / t_per_request
    cap_total = cap_per_replica * replicas

    # Poisson arrivals over the horizon.
    t, arrivals = 0.0, []
    while t < horizon:
        arrivals.append(t)
        t += -math.log(max(1e-9, rnd.random())) / max(0.05, rate)

    # Each arriving request waits for those already queued on its replica.
    # Queue length ahead of it = (excess arrivals per second) * time.
    excess = max(0.0, rate - cap_total)
    wait_per_request = excess * t_per_request / max(1, users)
    ttft = min(5.0, wait_per_request) + t_decode
    tpot = t_decode

    ok = ttft <= SLO_TTFT and tpot <= SLO_TPOT
    return {
        "goodput": 1.0 if ok else 0.0,
        "ttft": ttft,
        "tpot": tpot,
        "cap_total": cap_total,
        "admitted": len(arrivals),
    }

# %%
print(f"{'rate':>6} {'replicas':>9} {'capacity':>9} {'TTFT ms':>9} {'TPOT ms':>9}  {'verdict':>13}")
for rate in [4, 8, 12, 16, 20, 26, 34]:
    r = run_fleet(rate, replicas=4)
    verdict = "healthy" if r["goodput"] else "past the cliff"
    print(f"{rate:>6} {4:>9} {r['cap_total']:>9.1f} {r['ttft']*1e3:>9.0f} "
          f"{r['tpot']*1e3:>9.1f}  {verdict:>13}")

# %%
# The relationship made explicit: once the offered rate exceeds what the replicas
# can retire, the excess queues. Queue wait grows linearly with the excess, and
# TTFT is that wait plus one decode step. That is why the cliff is a cliff.
print("\nQueue wait (seconds) vs excess over capacity, for 4 replicas:")
cap4 = run_fleet(10, replicas=4)["cap_total"]
for mult in [0.8, 1.0, 1.2, 1.5, 2.0]:
    rate = cap4 * mult
    r = run_fleet(rate, replicas=4)
    print(f"  rate {rate:6.1f} req/s  capacity {cap4:.1f}  "
          f"TTFT {r['ttft']:6.2f}s  {'inside SLO' if r['goodput'] else 'MISSED SLO'}")

# %% [markdown]
# ## 4. Compare routing policies on real work
#
# Routing policy matters most when requests differ in size. Send a mix of short and
# long prompts and compare policies by how evenly the KV load lands. This mirrors
# the lesson's core claim: counting requests ≠ measuring load.

# %%
SHORT, LONG = 200, 2000

def kv_footprint(prompt, out):
    return (prompt + out) * KV_PER_TOKEN

def route_sim(policy, replicas=4, n_req=200, seed=1):
    """Route n requests to replicas and report the KV spread (busiest ÷ mean)."""
    rnd = random.Random(seed)
    kv = [0] * replicas
    rr = 0
    for _ in range(n_req):
        prompt = LONG if rnd.random() < 0.3 else SHORT
        cost = kv_footprint(prompt, OUT)
        if policy == "round-robin":
            r = rr % replicas; rr += 1
        elif policy == "least-connections":
            r = rnd.randrange(replicas)
        else:  # least-kv
            r = min(range(replicas), key=lambda i: kv[i])
        kv[r] += cost
    mean = sum(kv) / replicas
    return max(kv) / mean if mean else 1.0

print(f"{'policy':>20} {'KV spread':>10}   (lower is more even)")
for policy in ["round-robin", "least-connections", "least-kv"]:
    spreads = [route_sim(policy, seed=s) for s in range(20)]
    print(f"{policy:>20} {sum(spreads)/len(spreads):>9.2f}×")

print()
print("Round-robin deals in counts, so it cannot see that a long prompt holds")
print("10x the KV of a short one. Least-KV equalises the resource that actually")
print("runs out. This is the Lesson 23.1 argument, measured on your own numbers.")

# %%
# The gap is visible when you look at absolute loads, not just the ratio.
rnd = random.Random(3)
for policy in ["round-robin", "least-kv"]:
    kv = [0]*4; rr = 0
    for _ in range(60):
        prompt = LONG if rnd.random() < 0.3 else SHORT
        cost = kv_footprint(prompt, OUT)
        if policy == "round-robin":
            r = rr % 4; rr += 1
        else:
            r = min(range(4), key=lambda i: kv[i])
        kv[r] += cost
    print(f"{policy:>14}: " + "  ".join(f"{v/1e9:5.1f} GB" for v in kv))

# %% [markdown]
# ## 5. Prefix-aware routing: what reuse is worth
#
# A 35% prefix hit rate means paying 65% of the cold prefill bill. Measure how
# much prefill time that actually saves, using the slope you fitted in section 1.

# %%
SYS_PREFIX = 1000   # shared system prompt every request carries
UNIQUE     = 500    # the user's actual message

cold   = prefill(SYS_PREFIX + UNIQUE)
unique = prefill(UNIQUE)
saved  = (cold - unique) / cold
print(f"prefill {SYS_PREFIX+UNIQUE} tokens cold:  {cold*1e3:7.2f} ms")
print(f"prefill only the {UNIQUE} new tokens:   {unique*1e3:7.2f} ms")
print(f"saving from a {SYS_PREFIX}-token cached prefix: {saved*100:.0f}%")
print(f"which is {per_token*SYS_PREFIX*1e3:.1f} ms of GPU time per request")
print()
print("In the lesson this is amplified: at 35% hit rate a long 8000-token prompt")
print(f"pays {(1-0.35)*per_token*8000*1e3:.0f} ms instead of {per_token*8000*1e3:.0f} ms.")

# %%
# And the memory side: a warm prefix is KV that must stay resident. At this
# model's KV size, a shared 2k prefix across 100 users is real GPU memory.
shared_gb = SYS_PREFIX * 2 * KV_PER_TOKEN * 100 / 1e9
print(f"A {SYS_PREFIX}-token shared prefix across 100 concurrent users:")
print(f"  {SYS_PREFIX*KV_PER_TOKEN/1e6:.0f} MB each, {shared_gb:.1f} GB total")
print("That is memory you cannot use for new requests — which is why routing")
print("on reuse must also watch load (Lesson 23.2).")

# %% [markdown]
# ## 6. Compute what a real deployment would do
#
# Finally, scale the same arithmetic to the GPUs you would actually run. The
# ratios are what transfer from this T4.

# %%
GPUS = {
    "T4":    dict(bw=320e9,  flops=65e12,  mem=16e9,  price=0.35),
    "A100":  dict(bw=2.0e12, flops=312e12, mem=80e9,  price=1.6),
    "H100":  dict(bw=3.35e12,flops=989e12, mem=80e9,  price=2.5),
}
MODELS = {
    "0.5B": dict(p=0.49e9, kv=8*1024),      # measured shape above
    "8B":   dict(p=8.03e9, kv=131072),     # Module 05
    "70B":  dict(p=70.6e9, kv=327680),
}

def plan(model_name, gpu_name, ctx=2048, out=OUT):
    P  = MODELS[model_name]["p"]
    kv = MODELS[model_name]["kv"]
    g  = GPUS[gpu_name]
    # decode step: read weights (bf16) + KV, at 80% of peak bandwidth
    bytes_moved = P*2 + 32*ctx*kv
    t_dec = bytes_moved / (g["bw"]*0.8)
    # users the KV pool holds (0.9 of memory, minus weights and a reserve)
    pool = g["mem"]*0.9 - P*2 - 0.04*g["mem"]
    users = max(1, pool // (ctx*kv))
    t_sys = out*TPOT_SLO + TTFT_SLO
    return users, users/t_sys, t_dec

print(f"{'model':>6} {'gpu':>6} {'users':>8} {'req/s':>8} {'step ms':>9} {'$ / 1M tok':>11}")
for m in MODELS:
    for gn in GPUS:
        users, rps, td = plan(m, gn)
        # output tokens/s at the SLO: each user gets 1/TPOT tok/s
        tok_s = rps * OUT
        cost = GPUS[gn]["price"] / 3600 / max(1e-9, tok_s) * 1e6
        print(f"{m:>6} {gn:>6} {users:>8,} {rps:>8.1f} {td*1e3:>9.2f} {cost:>11.3f}")

print()
print("This table is the Module 25 preview: every routing and scaling decision")
print("here changes the last column. Hold the SLO, and the cost per million")
print("tokens falls out of the capacity you chose.")

# %% [markdown]
# ## 7. The deliverable
#
# Write a one-page **capacity plan** for the service below. It must contain:
#
# - the chosen routing policy, and one sentence on why for this workload;
# - the replica count you would run at peak, with one spare;
# - the maximum sustainable arrival rate at that replica count;
# - the measured capacity of one replica from section 2, and the formula you used;
# - what happens if the long-prompt share rises from 30% to 60%;
# - the metric you alert on first, and the threshold;
# - your estimate of cost per million output tokens, from section 6.
#
# **Service under study:** a chat product with retrieval. 70% short chat prompts
# (~900 tokens), 30% long RAG prompts (~8,000 tokens), 250 output tokens,
# strong multi-turn reuse, 12 req/s average and 36 req/s at the morning peak.
# SLO: p99 TTFT under 1 s, TPOT under 50 ms.

# %%
print("\n=== Your capacity plan, from the numbers you measured ===")
print(f"t_prefill_per_token  {per_token*1e6:.2f} us")
print(f"t_decode_step(32)    {t_decode*1e3:.2f} ms")
print(f"KV pool              {KV_POOL_TOKENS:,} tokens")
print(f"users per replica    {users_per_replica:,.0f}  @ {CTX} ctx")
print(f"capacity / replica   {req_per_replica:.1f} req/s")
print(f"peak 36 req/s needs  {math.ceil(36/req_per_replica)} replicas, +1 spare")
print()
print("Repeat for H100 + Llama-8B at 2048 ctx:")
u8, r8, _ = plan("8B", "H100", ctx=2048)
print(f"  users {u8:,}  capacity {r8:.1f} req/s  => {math.ceil(36/r8)}+1 = {math.ceil(36/r8)+1} replicas for the peak")