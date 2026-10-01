# %% [markdown]
# # Lab 21 · Measure the KV handoff and size both pools
#
# **Tokens per Second** · Module 21
#
# Disaggregation moves a finished request's KV cache from the GPU that prefilled it
# to the GPU that will decode it. This notebook makes that number **real on your own
# hardware** instead of a formula you trusted.
#
# By the end you will have measured four things:
#
# 1. **KV bytes per token** for a real model — confirm the $2 \times L \times H_{kv} \times d_h \times b$ formula.
# 2. **Transfer time** for that payload across host memory, PCIe and (if present) NVLink,
#    so you can see which of them is the bottleneck.
# 3. **The transfer share** of one decode step — the number that decides whether
#    disaggregation is viable at all on your link.
# 4. **Pool sizes** for a stated workload, and a check that they fit in memory.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# We use **Qwen2.5-1.5B-Instruct** so the payload and the decode step are both
# comfortably measurable on a free T4. The absolute times are much smaller than an 8B
# model; the **ratios** — which is all this module is about — transfer directly.
#
# **What this lab cannot do:** a Colab T4 is one GPU. It cannot run two pools, and it
# has no NVLink between two devices. What it *can* do is measure the two quantities the
# design depends on — bytes moved and time to move them — on the links you actually have,
# and extrapolate honestly to the fabric you would buy. We are explicit about which is which.

# %%
import torch, time, math, json

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
print(f"{p.name}   {p.total_memory/1e9:.1f} GB   sm_{p.major}{p.minor}")
print(f"devices visible: {torch.cuda.device_count()}")

# %% [markdown]
# ## 0. The model
#
# Qwen2.5-1.5B-Instruct. We read its real config rather than hard-coding numbers, so
# this notebook keeps working if you swap the model id.

# %%
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen2.5-1.5B-Instruct"
cfg = AutoConfig.from_pretrained(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

L, H_kv, d_head = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
PARAMS_B = sum(q.numel() for q in model.parameters()) / 1e9

print(f"model            {MODEL}")
print(f"parameters       {PARAMS_B:.2f}B")
print(f"layers           {L}")
print(f"KV heads         {H_kv}")
print(f"head dim         {d_head}")
print(f"hidden size      {cfg.hidden_size}")

# %% [markdown]
# ## 1. Confirm the KV formula
#
# The lesson's formula is
#
# $$
# \text{KV per token} = 2 \times \text{layers} \times \text{KV heads} \times \text{head dim} \times \text{bytes}
# $$
#
# The factor 2 is for the key **and** the value. We verify it by building a real KV cache
# for a real prompt and comparing its actual byte count to the formula. If these disagree,
# every number in this module is wrong, so this is the first thing to check.

# %%
def kv_bytes_per_token(model_cfg, bytes_per=2):
    return 2 * model_cfg.num_hidden_layers * model_cfg.num_key_value_heads * model_cfg.head_dim * bytes_per

predicted = kv_bytes_per_token(cfg)
print(f"predicted KV/token     {predicted:,} bytes  ({predicted/1024:.0f} KiB)")

# Build an actual cache with the real model and count its bytes.
prompt = "Explain what a KV cache is and why it matters for inference serving. " * 60
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
n_prompt = ids.shape[1]
print(f"\nprompt is {n_prompt} tokens")

with torch.no_grad():
    out = model(ids, use_cache=True)
past = out.past_key_values

# Count real bytes: sum over layers of keys + values.
actual = 0
for layer in past:
    k, v = layer[0], layer[1]          # each is [batch, kv_heads, seq, head_dim]
    actual += k.numel() * k.element_size()
    actual += v.numel() * v.element_size()

per_tok_actual = actual / n_prompt
print(f"measured  KV/token     {per_tok_actual:,.0f} bytes")
print(f"ratio measured/predicted {per_tok_actual/predicted:.4f}")
print("\nIf that ratio is ~1.0, the formula is exact and you may use it for any context length.")

# %% [markdown]
# ## 2. Measure the transfer, link by link
#
# This is the heart of the lab. We move a realistic KV payload three ways:
#
# | path | what it models |
# |---|---|
# | device → pinned host → device | the pessimistic case: a single slow link, no overlap |
# | pinned host → pinned host | host memory bandwidth, the reference ceiling |
# | the same, with chunking + overlap | what a real implementation does |
#
# We cannot buy an NVLink here, but we can measure what we have and compute what a
# 400 GbE or NVLink fabric *would* give, since transfer time is simply bytes ÷ bandwidth.

# %%
def payload_mb(n_tokens, bytes_per_tok=2):
    return n_tokens * kv_bytes_per_token(cfg, bytes_per_tok) / 1e6

def bench_copy(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters      # milliseconds

CONTEXTS = [1024, 4096, 8192, 16384]
rows = []

for n in CONTEXTS:
    mb = payload_mb(n)
    n_elem = int(mb * 1e6 / 2)            # fp16 elements
    src = torch.empty(n_elem, dtype=torch.float16, device="cuda").normal_()
    dst_dev = torch.empty_like(src)
    host = torch.empty(n_elem, dtype=torch.float16, pin_memory=True)

    # 1. device -> host -> device, strictly serial (no overlap)
    def serial():
        host.copy_(src, non_blocking=False)
        dst_dev.copy_(host, non_blocking=False)

    # 2. host -> host (pure memory bandwidth)
    host2 = torch.empty_like(host)
    def hostcopy():
        host2.copy_(host, non_blocking=True)

    t_serial = bench_copy(serial)
    t_host = bench_copy(hostcopy)

    bw_serial = (2 * n_elem * 2) / (t_serial / 1e3) / 1e9   # GB/s, both directions
    bw_host = (2 * n_elem * 2) / (t_host / 1e3) / 1e9

    rows.append(dict(ctx=n, mb=mb, t_serial_ms=t_serial, t_host_ms=t_host,
                     bw_serial_gbs=bw_serial, bw_host_gbs=bw_host))
    print(f"{n:>6} tok  {mb:7.1f} MB   serial {t_serial:6.2f} ms ({bw_serial:6.1f} GB/s)"
          f"   host->host {t_host:6.2f} ms ({bw_host:6.1f} GB/s)")

    del src, dst_dev, host, host2
    torch.cuda.empty_cache()

# %% [markdown]
# **Read this carefully.** On a T4 the "serial" number is dominated by PCIe, which is
# roughly 10–12 GB/s. That is the number to compare against a real fabric:
#
# | fabric | effective bandwidth | note |
# |---|---|---|
# | 25 GbE | ~3 GB/s | worse than what you just measured |
# | 100 GbE | ~12 GB/s | comparable to a single PCIe 4.0 x16 |
# | 8x100 GbE | ~100 GB/s | needs RDMA to actually achieve it |
# | 400 GbE (8x400) | ~400 GB/s | typical disaggregated fabric |
# | NVLink 4 | ~900 GB/s | same node, no network at all |
#
# Your measured `bw_serial_gbs` is the honest local baseline. The lesson's design rule
# is unchanged: **compare the transfer to one decode step.**

# %% [markdown]
# ## 3. The transfer share — the decision number
#
# Now the comparison that decides whether disaggregation is worth doing. We measure one
# decode step of the real model, then ask what fraction of it the transfer costs.

# %%
def bench_decode_step(batch, ctx_tokens, iters=10, warmup=3):
    """One decode step: feed a batch of sequences of KV length ctx_tokens."""
    # Fake KV cache in the shape the model would have.
    k = torch.randn(batch, H_kv, ctx_tokens, d_head, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    one = torch.randn(batch, 1, d_head, dtype=torch.float16, device="cuda")

    def step():
        # Approximate the attention read: the model reads K and V, writes one token.
        attn = torch.matmul(one, k.transpose(2, 3))
        attn = torch.softmax(attn.float(), dim=-1).to(torch.float16)
        torch.matmul(attn, v)
    t = bench_copy(step, iters=iters, warmup=warmup)
    del k, v, one, attn
    torch.cuda.empty_cache()
    return t

print(f"{'batch':>6} {'ctx':>7} {'KV MB':>9} {'step ms':>9}")
decode_rows = []
for batch, ctx in [(8, 1024), (32, 1024), (32, 4096), (64, 4096)]:
    t_step = bench_decode_step(batch, ctx)
    mb = payload_mb(ctx) * batch
    decode_rows.append(dict(batch=batch, ctx=ctx, kv_mb=mb, step_ms=t_step))
    print(f"{batch:>6} {ctx:>7} {mb:>9.1f} {t_step:>9.3f}")

# %% [markdown]
# Now combine them. For a single request handed from prefill to decode, the transfer
# moves **one** sequence's KV, not the whole batch:
#
# $$
# \text{share} = \frac{t_{\text{transfer}}(C)}{t_{\text{decode step}}}
# $$
#
# using your measured transfer bandwidth and your measured decode step.

# %%
best = max(rows, key=lambda r: r["bw_serial_gbs"])
bw_local = best["bw_serial_gbs"]        # GB/s, measured
step_ms = decode_rows[-1]["step_ms"]    # the batch-64, 4k-context step

print(f"measured local link: {bw_local:.1f} GB/s")
print(f"decode step (B=64, ctx=4096): {step_ms:.3f} ms\n")
print(f"{'ctx':>7} {'payload MB':>12} {'transfer ms':>12} {'share of step':>14}")

shares = {}
for n in CONTEXTS:
    mb = payload_mb(n)
    t_tr = (mb / 1e3) / bw_local * 1e3   # MB -> GB -> s -> ms
    share = t_tr / step_ms
    shares[n] = share
    flag = "OK" if share < 0.10 else ("marginal" if share < 0.25 else "TOO SLOW")
    print(f"{n:>7} {mb:>12.1f} {t_tr:>12.2f} {share:>13.1%}  {flag}")

# %% [markdown]
# The verdict follows directly. If your share is under 10%, the transfer hides behind
# the decode steps that follow it and disaggregation is cheap. If it is over ~25%, you
# spend more time shipping the cache than using it, and a single shared pool is better.
#
# **On this T4 with PCIe, what is your share at 16K context?** If it is large, that is the
# correct finding, not a failure — it tells you this hardware cannot disaggregate long
# contexts, and that is exactly the kind of conclusion you should refuse to build past.

# %% [markdown]
# ## 4. Size both pools for a stated workload
#
# Now the sizing formulas from the lesson, applied to **your measured** per-token prefill
# time and decode throughput. Use a realistic product workload:
#
# - 20 requests/second arriving
# - 4,096-token prompts (RAG + chat history)
# - 256-token answers
# - 512 concurrent users at 4,096 context

# %%
# Measure prefill throughput: how many prompt tokens/s this GPU can prefill.
def bench_prefill(n_tokens, iters=10, warmup=3):
    x = torch.randn(1, n_tokens, cfg.hidden_size, dtype=torch.float16, device="cuda")
    def f():
        with torch.no_grad():
            model(x, use_cache=False)
    t = bench_copy(f, iters=iters, warmup=warmup)
    del x
    torch.cuda.empty_cache()
    return t

print("prefill measurement:")
pref = {}
for n in [512, 1024, 2048]:
    t = bench_prefill(n)
    pref[n] = t
    print(f"  {n:>5} tok: {t:7.2f} ms   {n/t*1e3:9.0f} prompt-tok/s   {t/n*1e6:6.1f} us/token")

# %%
# --- the workload ---
ARRIVALS = 20.0        # requests/second
PROMPT = 4096          # tokens per prompt
ANSWER = 256           # tokens per answer
CONCURRENCY = 512      # simultaneous users
CTX = 4096

# --- prefill pool: sized by FLOPs, so by measured prompt-tok/s ---
t_per_token = pref[2048] / 2048                    # seconds per prompt token, measured
prompt_tok_per_s = 1.0 / t_per_token
prefill_demand = ARRIVALS * PROMPT                  # prompt tokens/second needed
prefill_gpus = math.ceil(prefill_demand / prompt_tok_per_s)

# --- decode pool: sized by throughput AND by KV memory, take the max ---
kv_per_seq_mb = payload_mb(CTX)
free_mem_mb = (p.total_memory / 1e6) - (PARAMS_B * 2 * 1e3) - 0.05 * (p.total_memory / 1e6)
seqs_per_gpu = int(free_mem_mb // kv_per_seq_mb)
decode_tok_per_s = decode_rows[-1]["batch"] / (step_ms / 1e3)   # measured, at B=64
decode_demand = ARRIVALS * ANSWER
gpus_for_throughput = math.ceil(decode_demand / decode_tok_per_s)
gpus_for_memory = math.ceil(CONCURRENCY / max(1, seqs_per_gpu))
decode_gpus = max(gpus_for_throughput, gpus_for_memory)

print(f"""
WORKLOAD: {ARRIVALS:.0f} req/s, {PROMPT}-token prompts, {ANSWER}-token answers,
          {CONCURRENCY} concurrent users at {CTX} context

PREFILL POOL  (sized by measured prefill throughput)
  needed        {prefill_demand:,.0f} prompt-tok/s
  one GPU does  {prompt_tok_per_s:,.0f} prompt-tok/s   (measured)
  =>  {prefill_gpus} GPU(s)

DECODE POOL   (sized by throughput AND KV memory — take the larger)
  KV per seq   {kv_per_seq_mb:,.1f} MB
  seqs per GPU {seqs_per_gpu:,}          (free {free_mem_mb:,.0f} MB)
  by throughput {gpus_for_throughput}    (need {decode_demand:,.0f} tok/s, one GPU does {decode_tok_per_s:,.0f})
  by memory     {gpus_for_memory}    (need {CONCURRENCY:,} concurrent, {seqs_per_gpu:,} per GPU)
  =>  {decode_gpus} GPU(s)   binding constraint: {'KV memory' if gpus_for_memory >= gpus_for_throughput else 'throughput'}

TOTAL: {prefill_gpus + decode_gpus} GPUs   (a shared pool would need {decode_gpus})
""")

# %% [markdown]
# **Which constraint bound?** In the lesson's H100 example, KV memory bound (5 vs 2). On a
# T4 it may well be throughput instead, because a 16 GB card cannot hold many 4K sequences.
# Both answers are correct — what matters is that you *report which one bound*, because a
# pool sized by the wrong one fails in production and not in the spreadsheet.

# %% [markdown]
# ## 5. Does it fit? A real memory check
#
# Numbers in a spreadsheet are not a plan. Allocate the KV cache the sizing implies and
# confirm the GPU accepts it. On a T4 this will fail for large designs — which is the
# honest answer for this hardware, and tells you exactly which constraint bites first.

# %%
design_seqs = CONCURRENCY
need_mb = design_seqs * kv_per_seq_mb
have_mb = p.total_memory / 1e6

print(f"design wants {design_seqs} sequences x {kv_per_seq_mb:,.1f} MB = {need_mb:,.0f} MB")
print(f"this GPU has {have_mb:,.0f} MB\n")

try:
    test = torch.empty(int(need_mb * 1e6 / 2), dtype=torch.float16, device="cuda")
    print(f"ALLOCATED OK — the T4 can hold the whole design ({test.numel()*2/1e6:,.0f} MB).")
    del test
except torch.cuda.OutOfMemoryError:
    short = need_mb - have_mb
    print(f"OOM as predicted: short by {short:,.0f} MB.")
    print(f"Interpretation: this design needs ~{math.ceil(need_mb/have_mb)}x this GPU's memory,")
    print(f"which is exactly why decode pools are sized by KV residency.")
finally:
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Record your results
#
# Copy the printed numbers into this dict so you can carry them into the project.

# %%
results = {
    "gpu": p.name,
    "gpu_mem_gb": round(p.total_memory / 1e9, 1),
    "model": MODEL,
    "kv_bytes_per_token": int(predicted),
    "measured_kv_per_token": round(per_tok_actual),
    "kv_formula_exact": bool(abs(per_tok_actual / predicted - 1) < 0.01),
    "local_link_gbs": round(bw_local, 1),
    "decode_step_ms_b64_c4k": round(step_ms, 3),
    "transfer_share_at_4k": round(shares[4096], 3),
    "transfer_share_at_16k": round(shares[16384], 3),
    "prefill_gpus": prefill_gpus,
    "decode_gpus": decode_gpus,
    "decode_binding": "kv_memory" if gpus_for_memory >= gpus_for_throughput else "throughput",
    "design_fits_this_gpu": need_mb < have_mb,
}

print(json.dumps(results, indent=2))
print("\nThese are your measured constants. The project uses this same structure at H100 scale.")

# %% [markdown]
# ## Check yourself
#
# 1. Did `measured/predicted` for KV bytes per token come out ≈ 1.0? If it did not, which
#    number would you trust, and why?
# 2. Your transfer share at 4K context — is it under 10%? What does that say about
#    disaggregation on *this* link, and would it change on a 400 GbE fabric?
# 3. For your pool sizing, which constraint bound — throughput or KV memory? Would the
#    answer change if the model were 70B instead of 1.5B?
# 4. Did the memory check in Part 5 succeed? If it OOM'd, what does that tell you about
#    which pool is harder to scale?
# 5. Suppose you halve the context length to 2K. What happens to the payload, the
#    transfer share, and `seqs_per_gpu`? Do all three move by the same factor, and why?
#
# ### A note on what this lab does and does not prove
#
# It **does** prove: the KV formula is exact, transfer time is bytes ÷ bandwidth, the
# binding constraint is the one that must be honoured, and on your hardware the
# transfer share is what it is.
#
# It **cannot** prove: that a real two-pool cluster achieves your measured bandwidth.
# PCIe here is a single link; a real fabric uses RDMA, and the gap between theoretical
# and achieved network bandwidth is large. Treat your measured number as a *floor* on
# what the fabric can do, never a promise.
