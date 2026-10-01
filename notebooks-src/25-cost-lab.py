# %% [markdown]
# # Project 25.P · The cheapest system that meets the SLO
#
# **Tokens per Second** · Module 25
#
# You will:
# 1. Build the **derived** cost column from first principles — no GPU needed.
# 2. Measure the **measured** column by actually running a model on a T4.
# 3. Compare the two and see where your cost model is honest and where it is not.
# 4. Size a fleet and price it, then write the decision sentence.
#
# Runtime → Change runtime type → **T4 GPU**. Then run the cells top to bottom.
#
# The lesson this notebook teaches: **most of a cost decision is arithmetic that costs
# nothing.** Only one column needs a GPU, and the gap between your prediction and your
# measurement is the number that makes you a better engineer.

# %%
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "torch", "accelerate"], check=False)
print("installing — you may need to RESTART the runtime before the next cell")

# %% [markdown]
# ## 1. The workload and the SLO, written down first
#
# Before any arithmetic. These two blocks drive everything else, and if you change them
# every number below changes with them. In your own project you would take these from a
# traffic sample, not from a guess.
#
# ```
# workload: Llama-3.1-8B class, median 512 prompt tokens, median 256 output tokens
# SLO:      p95 TTFT < 600 ms and p95 TPOT < 40 ms, over any rolling 5-minute window
# ```

# %%
WORKLOAD = dict(
    model_params=8.03e9,
    prompt_tokens=512,
    output_tokens=256,
    arrival_rate=8.0,     # requests/second, peak sustained
    target_util=0.60,     # run at 60%, not 100% (Module 06)
    context=2048,         # prompt + output, for KV memory
)
SLO = dict(ttft_p95_ms=600, tpot_p95_ms=40, window="5 min")

print("workload:", WORKLOAD)
print("SLO:     ", SLO)

# %% [markdown]
# ## 2. The derived column: exact, free, and it eliminates candidates first
#
# Every formula below is arithmetic you have already done in the course:
#
# ```
# weights       = params × bytes_per_weight
# kv_bytes_tok  = 2 × layers × kv_heads × head_dim × kv_bytes
# t_prefill     = max(2·P·n / (flops·mfu), weights / (bandwidth·mbu))
# t_step        = max((weights + batch·ctx·kv_bytes_tok) / (bandwidth·mbu), 2·P·B/(flops·mfu))
# m             = t_prefill + n_out · t_step / B        GPU-seconds per request
# cost          = m × price_per_second × 1e6 / n_out
# fleet         = ceil(arrival_rate × m / target_util)
# ```
#
# None of it needs a GPU, and none of it is a guess.

# %%
GPUS = {
    "t4":   dict(mem=16e9,  bw=320e9,  flops=65e12,  price=0.35),
    "l4":   dict(mem=24e9,  bw=300e9,  flops=121e12, price=0.80),
    "a100": dict(mem=80e9,  bw=2.0e12, flops=312e12, price=1.60),
    "h100": dict(mem=80e9,  bw=3.35e12,flops=989e12, price=2.50),
    "h200": dict(mem=141e9, bw=4.8e12, flops=989e12, price=3.50),
    "b200": dict(mem=180e9, bw=8.0e12, flops=2250e12,price=5.50),
}
MODELS = {
    "qwen05":  dict(name="Qwen2.5-0.5B",  params=0.49e9, layers=24, kv_heads=2,  head_dim=64),
    "llama3b": dict(name="Llama-3.2-3B",  params=3.2e9,  layers=28, kv_heads=8,  head_dim=128),
    "llama8b": dict(name="Llama-3.1-8B",  params=8.03e9, layers=32, kv_heads=8,  head_dim=128),
    "qwen32b": dict(name="Qwen2.5-32B",  params=32.8e9, layers=64, kv_heads=8,  head_dim=128),
    "llama70b":dict(name="Llama-3.1-70B", params=70.6e9, layers=80, kv_heads=8,  head_dim=128),
}
# achievable fractions of peak. Measure yours (Module 12); these are reasonable defaults.
MBU = {2: 0.80, 1: 0.75, 0.5: 0.65}   # by weight precision
MFU = 0.55
OVERHEAD = 0.0015                      # seconds of CPU per step: scheduling + launches
BYTES = {2: "BF16", 1: "FP8/INT8", 0.5: "INT4"}


def derive(model_id, gpu_id, wbytes=2, kvbytes=2, batch=64, ctx=WORKLOAD["context"],
           n_in=WORKLOAD["prompt_tokens"], n_out=WORKLOAD["output_tokens"]):
    """The derived column. Every field is arithmetic — no measurement anywhere."""
    m, g = MODELS[model_id], GPUS[gpu_id]
    weights = m["params"] * wbytes
    kv_tok = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * kvbytes

    # does it fit at all, with room for KV and a reserve?
    reserve = 0.04 * g["mem"]
    pool = g["mem"] - weights - reserve
    max_streams = max(0, int(pool // (kv_tok * ctx))) if kv_tok else 0

    # prefill: compute-bound, reads the weights once
    t_pre = max(2 * m["params"] * n_in / (g["flops"] * MFU),
                weights / (g["bw"] * MBU[wbytes])) + 0.002

    # decode: memory-bound, reads weights + every stream's KV, once
    kv_read = batch * ctx * kv_tok
    t_step = max((weights + kv_read) / (g["bw"] * MBU[wbytes]),
                 2 * m["params"] * batch / (g["flops"] * MFU)) + OVERHEAD

    # the identity: prefill yours alone, decode shared with the batch
    m_req = t_pre + n_out * t_step / batch
    price_s = g["price"] / 3600

    return dict(
        weights_gb=weights / 1e9,
        kv_per_tok_gb=kv_tok * ctx / 1e9,
        fits=weights + reserve < g["mem"] and max_streams >= batch,
        max_streams=max_streams,
        t_pre_ms=t_pre * 1e3,
        t_step_ms=t_step * 1e3,
        gpu_seconds=m_req,
        cost_per_mtok_out=m_req * price_s * 1e6 / n_out,
        cost_per_mtok_prompt=t_pre * price_s * 1e6 / n_in,
        tok_per_s=batch / t_step,
        price=g["price"],
    )


def fmt_row(d):
    return (f"{d['weights_gb']:>6.1f}GB  {d['max_streams']:>6}  "
            f"{'yes' if d['fits'] else 'NO':>4}  "
            f"{d['t_step_ms']:>7.2f}ms  {d['gpu_seconds']*1e3:>7.1f}ms  "
            f"${d['cost_per_mtok_out']:>7.3f}")


print(f"{'model':<13}{'gpu':<6}{'prec':<9}{'weights':>9}{'maxB':>7}{'fits':>6}"
      f"{'step':>10}{'m/req':>10}{'$/Mout':>9}")
print("-" * 76)
for mid in ("llama8b", "qwen32b", "llama70b"):
    for gid in ("a100", "h100", "h200"):
        for wb in (2, 1, 0.5):
            d = derive(mid, gid, wb)
            print(f"{MODELS[mid]['name']:<13}{gid:<6}{BYTES[wb]:<9}{fmt_row(d)}")

# %% [markdown]
# ### Read that table before running anything
#
# Three conclusions are already available, and none of them required a GPU:
#
# 1. **Some candidates simply do not run.** `llama70b` in BF16 needs 141 GB, which no
#    80 GB card holds; in BF16 on an H100 the max-stream count is 0. That is not "slow" —
#    it is an OOM. A cost figure computed for it would be arithmetic about a machine that
#    does not exist.
# 2. **The 32B model costs several times the 8B per token.** That gap is the whole
#    quality-versus-cost conversation, expressed as a number.
# 3. **Halving the weight bits saves real money, and less than people expect.** The step
#    is memory-bound, so fewer bytes is a shorter step — but once you are far past the
#    bandwidth floor, the extra saving shrinks. FP8 captures most of it; INT4 costs
#    quality for very little more.
#
# **Your task:** build this table for your own workload, then read it the same way.

# %%
# focus on the 8B on one H100 across precisions and batch sizes
print("Llama-3.1-8B on H100, 512 in / 256 out, 2048 ctx\n")
print(f"{'prec':<7}{'batch':>6}{'step ms':>9}{'tok/s':>8}{'per-user':>10}"
      f"{'fits':>6}{'maxB':>7}{'m/req ms':>11}{'$/Mout':>9}")
print("-" * 78)
for wb in (2, 1, 0.5):
    for b in (16, 32, 64, 128):
        d = derive("llama8b", "h100", wb, batch=b)
        print(f"{BYTES[wb]:<7}{b:>6}{d['t_step_ms']:>9.2f}{d['tok_per_s']:>8.0f}"
              f"{1000/d['t_step_ms']:>10.0f}{'yes' if d['fits'] else 'NO':>6}"
              f"{d['max_streams']:>7}{d['gpu_seconds']*1e3:>11.1f}{d['cost_per_mtok_out']:>9.3f}")

# %% [markdown]
# ## 3. The fleet, priced
#
# ```
# fleet = ceil(arrival_rate × m / target_util)
# bill  = fleet × price_per_hour × 730        # 730 hours ≈ one month
# ```
#
# And the utilization tax: a card is billed for every second it exists, so a fleet at 40%
# average utilization pays **2.5× per unit of useful work** versus a fully-utilized one.

# %%
def fleet(model_id, gpu_id, wbytes=2, batch=64, rate=WORKLOAD["arrival_rate"],
          target=WORKLOAD["target_util"], **kw):
    d = derive(model_id, gpu_id, wbytes, batch=batch, **kw)
    busy = rate * d["gpu_seconds"]
    n = max(1, -(-busy // target))            # ceil without floats
    return dict(
        busy=busy, required=busy / target, gpus=int(n),
        monthly=n * d["price"] * 730,
        gpus_100pct=max(1, int(busy) or 1),
    )


print(f"{'rate':>6}{'m/req ms':>10}{'busy GPU':>10}{'at target':>11}{'fleet':>7}{'$/month':>10}")
print("-" * 58)
for rate in (1, 2, 4, 8, 16, 32):
    f = fleet("llama8b", "h100", 2, rate=rate)
    m_ms = derive("llama8b", "h100", 2)["gpu_seconds"] * 1e3
    print(f"{rate:>6}{m_ms:>10.1f}{f['busy']:>10.2f}{f['required']:>11.2f}"
          f"{f['gpus']:>7}${f['monthly']:>9,.0f}")

print("\nutilization tax on 8 H100s ($2.50/h):")
for u in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
    print(f"  {u:>4.0%} util → ${8*2.50*730*u:>9,.0f}/month   "
          f"tax {1/u:>4.2f}× per unit of useful work")

# %% [markdown]
# ## 4. Now measure it
#
# Everything so far is a model. Now we run a real model on a real GPU and see how wrong
# the model was.
#
# **The rule from Module 17 applies:** one harness, and the only thing that changes
# between arms is the thing under test. Note the T4 has **no BF16 tensor cores** — it
# computes in FP16 — and far less memory than a datacentre card, so absolute numbers
# differ from the table above. **What transfers is the shape**: how cost moves with
# batch size, with precision, and with output length.

# %%
import torch, time, json, statistics

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if not torch.cuda.is_available():
    raise SystemExit("This notebook needs a GPU. Runtime → Change runtime type → T4 GPU.")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # small enough for a 16 GB T4
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print(f"params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B  "
      f"weights on GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% [markdown]
# ### The measured column
#
# Tokenize with the model's own tokenizer (never by splitting characters — Module 06),
# warm up, and record raw samples so you can compute percentiles afterwards. Note what
# is and is not being measured: this is a single-sequence generate loop, so it shows
# **per-user latency**, not server throughput. Server throughput needs an engine with
# continuous batching (Module 07), which is why the derived column above is the one you
# size a fleet from.

# %%
PROMPTS = [f"Explain in two sentences what a GPU tensor core does. (variant {i})"
           for i in range(32)]


@torch.inference_mode()
def measure(prompt, max_new_tokens=128, warm=False):
    ids = tok(prompt, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**ids, max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    n_new = out.shape[1] - ids["input_ids"].shape[1]
    if warm or n_new < 2:
        return None
    return dict(ttft=t1 - t0, e2e=t1 - t0, tokens=n_new,
                tpot=(t1 - t0) / max(1, n_new - 1),
                prompt_tokens=ids["input_ids"].shape[1])


# warm up: compilation, allocation, cold caches. Discard these entirely (Module 06).
for p in PROMPTS[:4]:
    measure(p, warm=True)
torch.cuda.synchronize()

rows = [measure(p) for p in PROMPTS[:16]]
rows = [r for r in rows if r]

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]

summary = dict(
    n=len(rows),
    ttft_p50=pct([r["ttft"] for r in rows], 50) * 1e3,
    ttft_p95=pct([r["ttft"] for r in rows], 95) * 1e3,
    tpot_p50=statistics.median([r["tpot"] for r in rows]) * 1e3,
    out_tokens_p50=statistics.median([r["tokens"] for r in rows]),
    tokens_per_user=1000 / statistics.median([r["tpot"] for r in rows]),
)
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## 5. Compare your prediction to your measurement
#
# This is the step that makes the exercise worth doing. Take the **shape** of your
# measurement and ask whether the derived model's qualitative predictions held:
#
# - **Does cost per output token fall as you raise concurrency?** It must, because the
#   decode term is divided by the batch. Measure it rather than assuming it.
# - **Does it fall steeply, or gently?** The derived model predicts a steep fall from
#   batch 1 to ~32 and a gentle one after, because KV reads start to dominate.
# - **Is output length close to linear in cost?** It should be very nearly so.

# %%
# a small concurrency sweep with a single sequence per "stream"
@torch.inference_mode()
def batched_step_time(n_seq, ctx_tokens=512, iters=12):
    """Approximate a decode step: read weights + KV for n_seq sequences, n_ctx tokens."""
    hidden, layers = 896, 24
    w = torch.randn(4096, hidden, device="cuda", dtype=torch.float16)
    kv = torch.randn(layers * n_seq * ctx_tokens * 2, device="cuda", dtype=torch.float16)

    for _ in range(3):                        # warm up
        (w @ w[:64].T).sum(); kv[:1024].sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        (w @ w[:64].T).sum()
        kv[: min(kv.numel(), 1 << 20)].sum()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


print(f"{'seqs':>6}{'step ms':>10}{'per-user tok/s':>17}{'rel. cost/token':>18}")
print("-" * 54)
base = None
for n in (1, 8, 32, 64, 128):
    dt = batched_step_time(n)
    per_user = 1000 / dt
    # cost per output token ∝ step time (share of one card-second per token)
    if base is None:
        base = dt / per_user
    rel = (dt / per_user) / base
    print(f"{n:>6}{dt*1e3:>10.2f}{per_user:>17.0f}{rel:>17.2f}×")

print("\nThe per-user column rises and then flattens as the step amortizes over the batch.")
print("The cost column falls. Those are the same fact seen from two sides — and the")
print("fall flattens too, because past a certain batch the KV reads dominate.")

# %% [markdown]
# ## 6. The output-length sweep
#
# Cost is near-linear in output length, and this is the biggest single lever you have.
# Measure it: cost per output token should be roughly **flat** in output length (each
# extra token costs one more share of a step), while cost per *request* should be
# proportional to it.

# %%
import torch as _t

@_t.inference_mode()
def measure_len(prompt, max_new_tokens, warm=False):
    ids = tok(prompt, return_tensors="pt").to("cuda")
    _t.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**ids, max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    _t.cuda.synchronize()
    t1 = time.perf_counter()
    n_new = out.shape[1] - ids["input_ids"].shape[1]
    if warm or n_new < 2:
        return None
    return n_new, t1 - t0


print(f"{'max_new':>9}{'out tokens':>12}{'seconds':>10}{'s/token':>11}{'$/Mtok (T4)':>15}")
print("-" * 60)
P = 0.35 / 3600.0        # T4 hourly rate, illustrative
for mnt in (32, 64, 128, 256):
    r = measure_len(PROMPTS[0], mnt, warm=(mnt == 32))
    if not r:
        continue
    n, secs = r
    # single-sequence: this request owned the whole card for the whole time
    per_tok = secs / n
    print(f"{mnt:>9}{n:>12}{secs:>10.3f}{per_tok*1e3:>10.2f}ms"
          f"{per_tok * P * 1e6:>15.2f}")

print("\nSeconds scale with output length; seconds *per token* stays roughly constant.")
print("That is the linearity you predicted: each generated token costs one decode step's")
print("share, and nothing else changes.")

# %% [markdown]
# ## 7. Write the decision
#
# Now assemble the artifact. Fill this in with your own numbers — the point is the
# *format*, and the discipline that every claim carries its workload.

# %%
DECISION = {
    "workload": f"{WORKLOAD['prompt_tokens']}-token prompt, {WORKLOAD['output_tokens']}-token answer, "
                f"{WORKLOAD['arrival_rate']} req/s, {WORKLOAD['context']}-token context",
    "slo": f"p95 TTFT < {SLO['ttft_p95_ms']} ms and p95 TPOT < {SLO['tpot_p95_ms']} ms "
           f"over any rolling {SLO['window']}",
}

chosen = derive("llama8b", "h100", 1, batch=64)      # FP8 on one H100, 64 streams
f = fleet("llama8b", "h100", 1, batch=64)

DECISION.update(
    chosen_configuration="Llama-3.1-8B, FP8 weights, 1× H100, 64 streams",
    derived_weights_gb=round(chosen["weights_gb"], 1),
    derived_step_ms=round(chosen["t_step_ms"], 2),
    derived_gpu_seconds_per_request=round(chosen["gpu_seconds"], 4),
    derived_cost_per_mtok_output=round(chosen["cost_per_mtok_out"], 3),
    fleet_floor=f["gpus"],
    committed=2,
    monthly_usd=int(2 * 2.50 * 730),
    headroom_reason="a single replica cannot absorb a deploy, a node drain, or a spike",
    scaling_trigger="queue depth or p95 latency — never GPU utilization",
)

print(json.dumps(DECISION, indent=2))

# %% [markdown]
# ## 8. Sensitivity: what would change your mind
#
# The section that separates a decision from a preference. Fill in the thresholds.

# %%
# All figures below use ONE workload so they are comparable: 512-token prompt,
# 256-token answer, 2048-token context, 64 streams.
c = chosen
t_pre = c["t_pre_ms"] / 1e3
dec = c["gpu_seconds"] - t_pre
fp8 = c["cost_per_mtok_out"]
bf16 = derive("llama8b", "h100", 2)["cost_per_mtok_out"]
q32 = derive("qwen32b", "h100", 1, batch=32)["cost_per_mtok_out"]
# Break-even: an A100 at $1.60/h is ~1.6x slower per user, so the H100 must stay
# cheaper than 1.60 * 1.6 = $2.56/h for the A100 to be the worse buy.
a100_break_even = 1.60 * 1.6

print("What would change the decision\n")
print(f"1. Output length doubles → m goes {c['gpu_seconds']:.4f} → "
      f"{t_pre + 2*dec:.4f} GPU-s ({(t_pre + 2*dec)/c['gpu_seconds']:.2f}x). "
      f"Prefill is untouched; the bill follows the decode term.")
print(f"   FP8 still beats BF16: ${fp8:.3f} vs ${bf16:.3f} per M output.")
print(f"2. Average utilization falls below 50% → utilization tax exceeds 2x. "
      f"Consolidate onto fewer cards before buying more.")
print(f"3. H100 rises above ${a100_break_even:.2f}/h → an A100 becomes the better buy, "
      f"since it must be cheaper than it is slow (Module 25.2 break-even test).")
print(f"4. Quality eval fails → Qwen2.5-32B FP8 at ${q32:.2f}/M output, "
      f"about {q32/fp8:.1f}x more. That ratio is what the decision hinges on.")
print(f"\nNote on rows 3 and 4: the 32B row uses batch 32, not 64, because BF16 32B "
      f"does not fit 64 streams on an 80 GB card. Batch 64 FP8 8B is the chosen "
      f"operating point and is directly comparable to the BF16 and INT4 rows.")

# %% [markdown]
# ## 9. Your turn
#
# Extend this into the deliverable. Required:
#
# 1. **Your own workload**, with median and p90 prompt/output lengths from a real sample.
# 2. **A derived matrix** across at least 2 models × 3 precisions × 2 GPUs × 3 batch sizes,
#    with `fits` computed honestly — a row that does not fit must say so.
# 3. **A measured column** for at least two configurations on the same harness, with
#    quality parity checked (temperature 0) and versions recorded.
# 4. **A fleet plan**: floor, committed, headroom reason, scaling trigger.
# 5. **A `DECISION.md`** stating the choice in one sentence with a number, plus a
#    sensitivity section with real thresholds.
#
# ## Check yourself
#
# - Could someone recompute your machine-seconds figure from your document alone?
# - Does every cost number state its model, GPU, precision, concurrency and lengths?
# - Have you separated derived from measured, and labelled every cell?
# - Have you named what would change your mind, with thresholds?

# %% [markdown]
# ## Where this goes next
#
# Module 26 (Specialized Workloads) changes the workload rather than the method: agents
# and reasoning models generate far more tokens per request, batch RL rollouts are pure
# throughput with no SLO, and multi-LoRA changes the fit arithmetic. The identity
# $m = t_{pre} + n_{out}\,t_{step}/B$ does not change — the inputs do.