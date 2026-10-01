# %% [markdown]
# # Lab 20 · Long Context & Tiered KV
# #
# **Tokens per Second** · Module 20
# #
# You will:
# 1. Measure real KV bytes per token and check it against the formula.
# 2. Reproduce all three walls of long context with *measured* constants.
# 3. Measure the host-DRAM copy bandwidth your machine actually achieves.
# 4. Build a working CPU-offloaded KV cache and verify it for correctness.
# 5. Find the resident window where streaming stops being free.
# 6. Produce the deliverable: a maximum-context table per tier.
# #
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
# Expect about 20 minutes.
# %
# Everything here runs on a 16 GB T4 with **Qwen2.5-0.5B-Instruct** (0.49 B params).
# Its KV cache is 12,288 bytes per token — a tenth of Llama-3.1-8B's — which is
# exactly why it fits. Measure the *shapes and ratios* here; carry them to real
# hardware, because the T4 is small in every direction (16 GB, 320 GB/s).

# %%
import torch, time, math, gc
from transformers import AutoTokenizer, AutoModelForCausalLM

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first."
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()

props = model.config
P      = sum(p.numel() for p in model.parameters())
L      = props.num_hidden_layers
H_KV   = props.num_key_value_heads
D_HEAD = props.head_dim
DEVICE = next(model.parameters()).device
DTYPE  = torch.float16

print(f"device      {torch.cuda.get_device_name(0)}")
print(f"parameters  {P/1e9:.3f} B")
print(f"layers      {L}   kv_heads {H_KV}   head_dim {D_HEAD}")
print(f"weight mem  {P*2/1e9:.2f} GB")
print(f"free mem    {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# %% [markdown]
# ## 0 · The arithmetic you will check against
# %
# From Module 05, bytes of KV per token is `2 * L * H_kv * D_head * bytes_per_number`.
# Every capacity number in this module comes from that one line.

# %%
KV_PER_TOK = 2 * L * H_KV * D_HEAD * 2          # the formula
PEAK_BW    = 320e9                              # T4 spec sheet, GB/s
print(f"KV per token     {KV_PER_TOK:,} bytes")
print(f"KV at 4K         {4096*KV_PER_TOK/1e6:.0f} MB")
print(f"KV at 32K        {32768*KV_PER_TOK/1e9:.2f} GB")

def human_time(s):
    if s < 1e-6: return f"{s*1e9:.0f} ns"
    if s < 1e-3: return f"{s*1e6:.1f} us"
    return f"{s*1e3:.1f} ms"

# %% [markdown]
# ## Task 1 · Measure the KV cache, do not trust the formula
# %
# Walk the cache object the model returns and add up its tensors. Then compare
# with the memory manager's report.
# %
# **Expected: exactly 12,288 bytes per token.** The formula is a *count of numbers*,
# not an approximation, so any disagreement is a layout surprise — usually an extra
# dimension or a non-contiguous view.

# %%
@torch.inference_mode()
def build_cache(prompt_len):
    ids = torch.randint(0, 1000, (1, prompt_len), device=DEVICE)
    out = model(input_ids=ids, use_cache=True)
    return out.past_key_values, out.logits

def cache_bytes(cache):
    """Sum the real storage of a DynamicCache (or a tuple of (k,v) tensors)."""
    layers = cache.layers if hasattr(cache, "layers") else None
    if layers is not None:
        return sum(t.numel() * t.element_size()
                   for lyr in layers
                   for t in (lyr.keys, lyr.values))
    return sum(t.numel() * t.element_size() for pair in cache for t in pair)

def cache_ntokens(cache):
    layers = cache.layers if hasattr(cache, "layers") else None
    if layers is not None:
        return layers[0].keys.shape[-2]
    return cache[0][0].shape[-2]

rows = []
for n in (512, 2048, 8192):
    c, _ = build_cache(n)
    rows.append((n, cache_bytes(c), cache_bytes(c) / n))
    del c; gc.collect(); torch.cuda.empty_cache()

print(f"{'tokens':>8} {'measured':>14} {'bytes/token':>13} {'formula':>10} {'ratio':>7}")
for n, b, per in rows:
    print(f"{n:>8,} {b:>14,} {per:>13,.1f} {KV_PER_TOK:>10,} {per/KV_PER_TOK:>7.3f}")

# the memory manager's view, which includes padding and kernel scratch
torch.cuda.empty_cache()
base = torch.cuda.memory_allocated()
c, _ = build_cache(4096)
with_cache = torch.cuda.memory_allocated()
n_tok = cache_ntokens(c)
delta = with_cache - base
print(f"\nallocated for {n_tok:,} tokens: {delta/1e6:.1f} MB")
print(f"cache object alone:              {cache_bytes(c)/1e6:.1f} MB")
print(f"overhead beyond the cache:       {(delta-cache_bytes(c))/1e6:.1f} MB  <- alignment + scratch")
print("Record this delta: it is what a capacity plan actually needs.")
del c; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Task 2 · Confirm wall one: the memory cliff
# %
# Compute `N_max`, then **prove it** by allocating that many slots — and one more.
# Then find where the allocation actually fails. The gap is your workspace reserve.

# %%
torch.cuda.empty_cache()
free = torch.cuda.mem_get_info()[0]
weights = P * 2
workspace = 0.04 * torch.cuda.get_device_properties(0).total_memory
headroom  = 0.10
budget = (free - weights - workspace) * (1 - headroom)
print(f"free {free/1e9:.2f} GB - weights {weights/1e9:.2f} GB - ws {workspace/1e9:.2f} GB")
print(f"-> KV budget {budget/1e9:.2f} GB = {int(budget/KV_PER_TOK):,} context tokens\n")

print(f"{'context':>10} {'per conv':>12} {'predicted':>11} {'allocated':>10}")
def try_allocate(n_seq, ctx):
    """Try to allocate n_seq cache slots of ctx tokens each. Returns True/False."""
    try:
        n = L * n_seq * 2 * H_KV * ctx * D_HEAD
        t = torch.empty(n, dtype=DTYPE, device=DEVICE)
        del t
        return True
    except torch.cuda.OutOfMemoryError:
        return False

for ctx in (4096, 8192, 32768):
    per = ctx * KV_PER_TOK
    pred = int(budget // per)
    ok = try_allocate(pred, ctx) if pred > 0 else False
    print(f"{ctx:>10,} {per/1e6:>10.0f} MB {pred:>11,} {'yes' if ok else 'NO':>10}")
    gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Task 3 · Confirm wall three: the decode tax
# %
# Time one decode step against a cache of increasing length. **Expected:** roughly
# linear in cache size, flat while the 1 GB weight read dominates, climbing after.
# Fit a line: the slope is your achieved bandwidth, the intercept is the weight read.

# %%
def decode_step_time(ctx, batch=1, iters=20):
    """Time one decode step against a prebuilt cache of `ctx` tokens."""
    with torch.inference_mode():
        ids = torch.randint(0, 1000, (1, ctx), device=DEVICE)
        out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        nxt = torch.randint(0, 1000, (batch, 1), device=DEVICE)
        def step():
            o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            return o.logits
        for _ in range(5): step()                      # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): step()
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / iters
    del out, cache; gc.collect(); torch.cuda.empty_cache()
    return t

print(f"{'ctx':>8} {'KV bytes':>13} {'step ms':>9} {'tok/s':>8}")
data = []
for ctx in (256, 1024, 4096, 8192, 16384):
    if ctx * KV_PER_TOK > budget: continue
    t = decode_step_time(ctx)
    data.append((ctx, ctx * KV_PER_TOK, t))
    print(f"{ctx:>8,} {ctx*KV_PER_TOK/1e6:>10.0f} MB {t*1e3:>8.2f} {1/t:>8.1f}")
    gc.collect(); torch.cuda.empty_cache()

# fit: time = intercept + slope * bytes
n = len(data)
sx  = sum(d[1] for d in data); sy = sum(d[2] for d in data)
sxx = sum(d[1]**2 for d in data); sxy = sum(d[1]*d[2] for d in data)
slope = (n*sxy - sx*sy) / (n*sxx - sx**2)
intercept = (sy - slope*sx) / n
print(f"\nfit: step = {intercept*1e3:.2f} ms + bytes/{1/slope/1e9:.2f} GB/s")
print(f"achieved bandwidth  {1/slope/1e9:.2f} GB/s  (spec {PEAK_BW/1e9:.0f} GB/s)")
print(f"weight read alone   {intercept*1e3:.2f} ms")
wall = weights / slope
print(f"cache read overtakes the weights at ctx = {int(wall):,} tokens")
print("Compare with 2P/b_kv from the lesson — close, and any gap is your slope's error.")

# %% [markdown]
# ## Task 4 · Time prefill, and find your own wall two
# %
# **Expected:** close to linear, because `n* = P/(2*d*L) ≈ 11,400` tokens puts this
# model's quadratic wall inside the sweep. Fit `t` against `n` and against `n^2`.

# %%
def prefill_time(n, iters=3):
    with torch.inference_mode():
        ids = torch.randint(0, 1000, (1, n), device=DEVICE)
        for _ in range(2):
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / iters
    gc.collect(); torch.cuda.empty_cache()
    return t

print(f"{'tokens':>8} {'prefill ms':>12} {'us/token':>10}")
pf = []
for n in (512, 2048, 8192, 16384, 32768):
    t = prefill_time(n)
    pf.append((n, t))
    print(f"{n:>8,} {t*1e3:>11.1f} {t/n*1e6:>9.1f}")

def r2(xs, ys):
    nn = len(xs)
    mx, my = sum(xs)/nn, sum(ys)/nn
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
    return (sxy**2)/(sxx*syy)

ns    = [d[0] for d in pf]
ts    = [d[1] for d in pf]
print(f"\nR^2 linear in n:      {r2(ns, ts):.4f}")
print(f"R^2 linear in n^2:    {r2([x**2 for x in ns], ts):.4f}")
wall_compute = P / (2 * props.hidden_size * L)
print(f"predicted quadratic wall n* = {int(wall_compute):,} tokens")
print("Whichever fits better tells you which side of the wall this sweep is on.")

# %% [markdown]
# ## Task 5 · Measure your host tier
# %
# This is the most important measurement in the lab. Compare **pinned** against
# **pageable**. Pageable goes through a driver staging buffer and loses 3–10x.
# The pinned path can saturate; the pageable one cannot.

# %%
def measure_host_bw(size_mb, n_rounds=10, pin=True):
    """(GB/s, first-byte latency ms) for a host->device copy."""
    n = int(size_mb * 1e6 / 2)
    host = torch.empty(n, dtype=DTYPE).pin_memory() if pin else torch.empty(n, dtype=DTYPE)
    dev = torch.empty(n, dtype=DTYPE, device=DEVICE)
    for _ in range(3):
        dev.copy_(host, non_blocking=pin); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_rounds):
        dev.copy_(host, non_blocking=pin)
    torch.cuda.synchronize()
    bw = size_mb * 1e6 * n_rounds / (time.perf_counter() - t0)

    small = min(n, 2048)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(50):
        dev[:small].copy_(host[:small], non_blocking=pin)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t1) / 50 * 1e3
    del host, dev; gc.collect(); torch.cuda.empty_cache()
    return bw / 1e9, lat

print(f"{'size':>8} {'pinned GB/s':>13} {'pageable GB/s':>15} {'ratio':>7} {'lat ms':>8}")
results = {}
for mb in (4, 32, 128):
    bp, lp = measure_host_bw(mb, pin=True)
    bu, _  = measure_host_bw(mb, pin=False)
    results[mb] = bp
    print(f"{mb:>6} MB {bp:>13.2f} {bu:>15.2f} {bp/max(bu,1e-9):>7.1f} {lp:>8.3f}")

HOST_BW = results[32] * 1e9
print(f"\nuse HOST_BW = {HOST_BW/1e9:.2f} GB/s for everything below")

# %% [markdown]
# ## The crossover: is the host tier even worth it?
# %
# Recomputing one token of KV costs `2P` FLOPs at the prefill MFU. A decode step is
# memory-bound, so that arithmetic runs on tensor cores that are otherwise idle.
# Fetching wins above `b_kv / t_recompute`.

# %%
T_FLOPS = 65e12        # T4 BF16, approximate dense figure
MFU = 0.55
t_recompute = (2 * P) / (T_FLOPS * MFU)
XOVER = KV_PER_TOK / t_recompute
print(f"recompute one token   {human_time(t_recompute)}")
print(f"KV per token          {KV_PER_TOK:,} bytes")
print(f"break-even bandwidth  {XOVER/1e9:.2f} GB/s")
print(f"your pinned host tier {HOST_BW/1e9:.2f} GB/s -> "
      f"{'FETCH WINS' if HOST_BW > XOVER else 'RECOMPUTE WINS'} by {max(HOST_BW,XOVER)/min(HOST_BW,XOVER):.1f}x")
print("\nNote this crossover is a *bandwidth* question, not a latency one.")
print("And on this small model it is high, because recompute is cheap in FLOPs.")

# %% [markdown]
# ## Task 6 · Build the offloaded cache by hand
# %
# Keep the most recent `R` tokens in GPU memory, hold the rest in pinned host
# memory, and fetch the tail each step.
# %
# **Bugs that bite:** a pageable host tensor; forgetting `synchronize()` before
# timing; reusing the pinned buffer while the previous step still reads it.

# %%
class OffloadedCache:
    """Split KV cache: `resident` most-recent tokens on GPU, the rest in pinned host memory."""

    def __init__(self, layers, kv_heads, head_dim, max_len, resident, dtype=torch.float16):
        self.resident = min(resident, max_len)
        self.max_len = max_len
        n = layers * 2 * kv_heads * max_len * head_dim
        self.host = torch.empty(n, dtype=dtype).pin_memory()
        self.gpu = torch.empty((layers, 2, kv_heads, self.resident, head_dim),
                               dtype=dtype, device=DEVICE)
        self.shape = (layers, 2, kv_heads, max_len, head_dim)
        self.filled = 0

    def write(self, layer, k, v):
        """Append one token's k,v for `layer` at position self.filled."""
        t = k.shape[-2]                                   # new tokens
        off = self.filled
        if off >= self.max_len:
            raise RuntimeError("cache full")
        s = slice(off, off + t)
        self.host[layer, 0, :, s] = k[0].cpu()
        self.host[layer, 1, :, s] = v[0].cpu()
        if off + t > self.resident:                       # spill into the resident window
            rs = max(0, self.resident - off)
            self.gpu[layer, 0, :, :rs] = self.host[layer, 0, :, s.start:s.start+rs].to(DEVICE)
            self.gpu[layer, 1, :, :rs] = self.host[layer, 1, :, s.start:s.start+rs].to(DEVICE)
        self.filled = off + t

    def gather(self):
        """Return the full cache on GPU: resident window + streamed tail."""
        if self.filled <= self.resident:
            return self.gpu
        out = torch.empty((self.shape[0], self.shape[1], self.shape[2],
                           self.filled, self.shape[4]), dtype=DTYPE, device=DEVICE)
        res = self.resident
        out[:, :, :, :res] = self.gpu[:, :, :, :res]
        out[:, :, :, res:] = self.host[:self.shape[0], :, :, res:self.filled].to(DEVICE)
        return out

    def nbytes(self):
        return self.host.numel() * self.host.element_size()

# verify: with resident == full length, gathered must equal a plain resident build
@torch.inference_mode()
def verify_offload(n=1024):
    ids = torch.randint(0, 1000, (1, n), device=DEVICE)
    ref = model(input_ids=ids, use_cache=True)

    oc = OffloadedCache(L, H_KV, props.head_dim, n, resident=n)
    for li, lyr in enumerate(ref.past_key_values.layers):
        oc.write(li, lyr.keys, lyr.values)
    gathered = oc.gather()

    worst = 0.0
    for li, lyr in enumerate(ref.past_key_values.layers):
        worst = max(worst,
                    (gathered[li, 0] - lyr.keys[0]).abs().max().item(),
                    (gathered[li, 1] - lyr.values[0]).abs().max().item())
    print(f"gathered {tuple(gathered.shape)}")
    print(f"max abs difference vs resident cache: {worst:.3e}")
    print("Expected: exactly 0. A transfer changes where bytes live, not their values.")
    if worst == 0.0:
        print("PASS - offloaded cache is exact when nothing is actually spilled.")

verify_offload(1024)

# %% [markdown]
# ## Task 7 · Find the resident window
# %
# Sweep `R` and measure the gather time for a fixed long context. There is a knee:
# flat while the streamed tail is small, then rising in proportion to bytes fetched.
# Convert the knee into a **byte budget**, then into a maximum context.

# %%
def build_offloaded(n, resident):
    """Build an offloaded cache of n tokens with the given resident window, return (cache, gather_fn)."""
    oc = OffloadedCache(L, H_KV, props.head_dim, n, resident)
    with torch.inference_mode():
        ids = torch.randint(0, 1000, (1, n), device=DEVICE)
        out = model(input_ids=ids, use_cache=True)
        for li, lyr in enumerate(out.past_key_values.layers):
            oc.write(li, lyr.keys, lyr.values)
    del out; gc.collect()
    return oc

def gather_time(oc, iters=10):
    for _ in range(2): oc.gather()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): oc.gather()
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / iters
    del _ ; gc.collect(); torch.cuda.empty_cache()
    return t

CTX = 8192
print(f"streaming budget = host bandwidth x weight-read time")
print(f"                  = {HOST_BW/1e9:.2f} GB/s x {intercept*1e3:.2f} ms"
      f" = {HOST_BW*intercept/1e6:.0f} MB per step\n")

print(f"ctx {CTX:,}, resident sweep")
print(f"{'resident':>9} {'streamed':>10} {'stream MB':>11} {'gather ms':>11} {'ok?':>6}")
for r in (0, 1024, 2048, 4096, 8192):
    oc = build_offloaded(CTX, r)
    stream_mb = (CTX - min(r, CTX)) * KV_PER_TOK / 1e6
    t = gather_time(oc)
    budget_mb = HOST_BW * intercept / 1e6
    ok = stream_mb <= budget_mb
    print(f"{r:>9,} {CTX-min(r,CTX):>10,} {stream_mb:>10.0f}MB {t*1e3:>10.2f} "
          f"{'yes' if ok else 'OVER':>6}")
    del oc

budget_bytes = HOST_BW * intercept
max_stream_tokens = budget_bytes / KV_PER_TOK
print(f"\nmax streamable per step: {max_stream_tokens:,.0f} tokens ({budget_bytes/1e6:.0f} MB)")
print(f"with a resident window of R, max context = R + {max_stream_tokens:,.0f}")
for r in (2048, 4096, 8192):
    print(f"  R={r:>6,} -> max context {int(r + max_stream_tokens):>7,} tokens")

# %% [markdown]
# ## The streaming budget, and what it means for capacity
# %
# Streamed bytes per step must stay under `bandwidth x weight-read time` or the
# fetch shows up as latency. That is a **bandwidth** question, exactly like the
# recompute crossover in Task 5.

# %%
print(f"tier                 {'bandwidth':>11} {'per-step budget':>17} {'max ctx @R=4096':>17}")
tiers = [("GPU HBM", PEAK_BW * 0.8),
         ("Host DRAM (pinned, measured)", HOST_BW),
         ("NVMe (7 GB/s typical)", 7e9),
         ("Remote 10 GbE (1.25 GB/s)", 1.25e9)]
R = 4096
for name, bw in tiers:
    budget_b = bw * intercept
    max_ctx = R + budget_b / KV_PER_TOK
    print(f"{name:<26} {bw/1e9:>8.2f} GB/s {budget_b/1e6:>13.0f} MB {int(max_ctx):>15,}")

print("\nRead the table: the budget is set by bandwidth times the time the step")
print("was already spending on weights. A slower tier means a smaller maximum")
print("context for the same latency - or, equivalently, recompute instead.")

# %% [markdown]
# ## Task 8 · The deliverable
# %
# Build your capacity table. Fill in the measured constants from the cells above.
# Pick a concurrency target (32 here) and find the maximum context per tier.

# %%
CONCURRENCY = 32

def report(gpu_budget, tier_bw, tier_name, resident, weight_read):
    budget_b = tier_bw * weight_read
    max_ctx = resident + budget_b / KV_PER_TOK
    # capacity is still bounded by what the card itself can hold
    max_ctx_mem = gpu_budget / KV_PER_TOK
    ctx = int(min(max_ctx, max_ctx_mem))
    # per-user step time at that context, streamed tail + weight read
    stream_bytes = max(0, ctx - resident) * KV_PER_TOK
    t_step = weight_read + stream_bytes / tier_bw
    return {
        "tier": tier_name,
        "resident": resident,
        "max_ctx": ctx,
        "conversations": gpu_budget / (ctx * KV_PER_TOK),
        "per_user_tps": 1 / t_step,
        "stream_MB": stream_bytes / 1e6,
    }

# the card holds the full context even when streamed? No: streaming frees the card,
# so for a tiered cache the card holds only the resident windows.
def report_tiered(host_memory, tier_bw, tier_name, resident, weight_read):
    """With tiering, each conversation pins only `resident` tokens on the card."""
    budget_b = tier_bw * weight_read
    max_ctx = resident + budget_b / KV_PER_TOK
    conv = host_memory / (resident * KV_PER_TOK)
    stream_bytes = (max_ctx - resident) * KV_PER_TOK
    t_step = weight_read + stream_bytes / tier_bw
    return {
        "tier": tier_name,
        "resident": resident,
        "max_ctx": int(max_ctx),
        "conversations": int(conv),
        "per_user_tps": 1 / t_step,
        "stream_MB": stream_bytes / 1e6,
    }

HOST_MEMORY = 30e9   # how much host RAM you are willing to dedicate
R = 4096

print(f"concurrency target: {CONCURRENCY}\n")
print(f"{'tier':<28}{'R':>7}{'max ctx':>11}{'per-user tok/s':>16}{'stream MB':>11}")
rows = [
    report(gpu_budget=budget, tier_bw=PEAK_BW * 0.8, tier_name="resident only (HBM)",
           resident=CTX, weight_read=intercept),
    report_tiered(HOST_MEMORY, HOST_BW, "host DRAM (measured)", R, intercept),
    report_tiered(HOST_MEMORY, 7e9, "NVMe (7 GB/s)", R, intercept),
    report_tiered(HOST_MEMORY, 1.25e9, "remote 10 GbE", R, intercept),
]
for r in rows:
    print(f"{r['tier']:<28}{r['resident']:>7,}{r['max_ctx']:>11,}"
          f"{r['per_user_tps']:>16.1f}{r['stream_MB']:>10.0f}")

# %% [markdown]
# ## Refine: choose R by sweeping it, not by guessing
# %
# A bigger resident window means a bigger maximum context but fewer conversations,
# because each one pins more of the card. Find the knee.

# %%
print(f"{'R':>8}{'max ctx':>11}{'conversations':>15}{'per-user tok/s':>16}")
best = None
for r in (1024, 2048, 4096, 8192, 16384):
    rep = report_tiered(HOST_MEMORY, HOST_BW, "host DRAM", r, intercept)
    meets = rep["conversations"] >= CONCURRENCY
    print(f"{r:>8,}{rep['max_ctx']:>11,}{rep['conversations']:>15,}"
          f"{rep['per_user_tps']:>16.1f}{'  <- meets target' if meets else ''}")
    if meets and (best is None or rep["max_ctx"] > best[1]["max_ctx"]):
        best = (r, rep)

if best:
    r, rep = best
    print(f"\nbest R for {CONCURRENCY} conversations: {r:,}")
    print(f"  max context      {rep['max_ctx']:,} tokens")
    print(f"  per-user speed   {rep['per_user_tps']:.1f} tok/s")
    print(f"  streamed/step    {rep['stream_MB']:.0f} MB")
    print("\nThis is your deliverable: the number, the resident window that produced")
    print("it, the speed it costs, and the byte figure you would alert on.")
else:
    print(f"\nNo R meets {CONCURRENCY} conversations. That is a legitimate result:")
    print("report it, and say what you would change (smaller model, FP8 cache,")
    print("more cards, or a smaller context).")

# %% [markdown]
# ## Check yourself
# %
# 1. Your measured bytes per token: is it exactly 12,288? If not, what is the ratio?
# 2. Your KV budget in tokens, and how many 32K conversations fit?
# 3. Your pinned host bandwidth, your pageable one, and the ratio. What does the
#    crossover say — does the host tier win or lose for this model?
# 4. Your decode step time at 1,024 tokens and at 16,384. Where do they cross?
# 5. From the fit in Task 3: what fraction of the T4's 320 GB/s did you achieve?
# 6. With `R = T` the offloaded cache matched exactly. With `R < T` it does not.
#    Whose fault is that, and what would you have to change to get exact answers?

# %% [markdown]
# ## If you have a bigger GPU
# %
# On an 80 GB card with Llama-3.1-8B (131,072 bytes/token), check three predictions
# from the lesson:
# 1. The streaming budget is 330 MB per step (55 GB/s × 6.0 ms of weight read).
# 2. PCIe at 55 GB/s is far above the 4.4 GB/s recompute crossover.
# 3. A 32,768-token cache at batch 8 is 34 GB, which blows the budget by 100× —
#    so a resident window is mandatory, not an optimization.
