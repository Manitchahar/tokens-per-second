# %% [markdown]
# # Project 09.3 · PyTorch vs vLLM, then optimize one GPU
#
# **Tokens per Second** · Module 09
#
# This is the project notebook. It gives you a working harness for the three halves of
# the project, and prints the matrix you are supposed to fill in. **Your job is to make
# the comparison fair, write the report, and explain the result.**
#
# You will:
# 1. Write the predictions first, from the arithmetic.
# 2. Measure a `transformers` generate loop, one prompt at a time.
# 3. Measure a **hand-rolled batch** — the honest middle of the road, and the control
#    that proves how much of vLLM's win is simply batching.
# 4. Measure `vllm.LLM` at several `max_num_seqs` values.
# 5. Sweep the tuning flags and pick an operating point against a latency target.
#
# Runtime → Change runtime type → **T4 GPU**. Run the cells top to bottom.
#
# Everything is small on purpose: `Qwen2.5-0.5B-Instruct` (0.98 GB in FP16) and a
# handful of prompts, so the whole notebook runs in about ten minutes on a free T4.

# %% [markdown]
# ## 0. Install (restart the runtime afterwards)

# %%
!pip install -q vllm

# %% [markdown]
# ## 1. The workload, fixed and shared by every engine
#
# One list of prompts, one output length, temperature 0 (greedy, so every engine is
# doing the same work). Anything that is not in this cell is not allowed to vary.

# %%
import gc, json, math, os, statistics, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.float16          # the T4 has no BF16 tensor cores
PROMPTS = [
    "In two sentences, explain what the KV cache is and why it exists.",
    "In two sentences, explain why GPU memory bandwidth limits decoding.",
    "In two sentences, explain what continuous batching does.",
    "In two sentences, explain what PagedAttention saves.",
    "In two sentences, explain what time to first token measures.",
    "In two sentences, explain what time per output token measures.",
    "In two sentences, explain why arithmetic intensity decides which limit you hit.",
    "In two sentences, explain why a decode step is memory-bound.",
    "In two sentences, explain what prefill computes.",
    "In two sentences, explain what a roofline diagram shows.",
    "In two sentences, explain what quantization trades away.",
    "In two sentences, explain what speculative decoding guesses.",
    "In two sentences, explain what tensor parallelism splits.",
    "In two sentences, explain what a CUDA graph replays.",
    "In two sentences, explain what goodput counts.",
    "In two sentences, explain what an SLO promises.",
]
MAX_NEW = 64
N_REQUESTS = len(PROMPTS)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

print(f"{N_REQUESTS} prompts, <= {MAX_NEW} new tokens each, greedy")
print("one prompt is", len(tok(PROMPTS[0])["input_ids"]), "tokens")

# %% [markdown]
# ## 2. The predictions, from the arithmetic
#
# Write these in `NOTES.md` **before** you measure. Keep them next to the results
# forever; being wrong in a documented way is how you learn where your mental model
# breaks.
#
# - weights = 0.494 B × 2 bytes ≈ **0.98 GB**
# - KV per token = 2 × 24 layers × 2 KV heads × 64 head dim × 2 bytes = **12,288 B**
# - decode step at concurrency 1 ≈ 0.98 GB ÷ (0.8 × 320 GB/s) + 1.5 ms ≈ **5.3 ms**
#   → about **190 tokens/s per user**
# - at concurrency 64 with ~700 tokens of context each, the KV read is
#   64 × 700 × 12,288 B ≈ 0.55 GB on top of the weights, so the step is roughly 1.6×
#   longer → about **120 tokens/s each**, ~**7,700 tokens/s total**
# - the pool: 0.9 × 16 GB − 0.98 GB − 0.64 GB ≈ 13.8 GB ÷ 12,288 B ≈ **1.1 M tokens**
#   → about **1,600 sequences at 700 tokens**, so memory is not the limit at these sizes

# %%
PARAMS, BYTES, TOTAL, UTIL = 0.494e9, 2, 16e9, 0.90
KV_PER_TOKEN = 2 * 24 * 2 * 64 * BYTES
t_bw = 0.98e9 / (0.8 * 320e9) + 0.0015
pool = UTIL * TOTAL - PARAMS * BYTES - 0.04 * TOTAL

print(f"predicted step at concurrency 1 : {t_bw*1000:.1f} ms  ->  {1/t_bw:.0f} tok/s")
print(f"predicted pool                  : {pool/1e9:.1f} GB -> {pool/KV_PER_TOKEN:,.0f} tokens")
print(f"predicted worst-case concurrency: {pool/KV_PER_TOKEN/700:,.0f} at 700 tokens each")
print("\n(Your numbers will differ. Write yours down first, then compare.)")

# %% [markdown]
# ## 3. The measurement, defined once
#
# One function reports the four numbers Module 06 insists on. Every engine goes through
# it, so the comparison is at least *consistent*, which is most of what fairness means.

# %%
def report(name, latencies, out_tokens, wall):
    s = sorted(latencies)
    p50 = s[len(s) // 2]
    p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    out = out_tokens / wall
    print(f"\n--- {name}")
    print(f"    wall clock      {wall:7.2f} s for {len(s)} requests")
    print(f"    per-request p50 {p50*1000:7.0f} ms   p95 {p95*1000:7.0f} ms")
    print(f"    output tokens   {out_tokens:5d}   -> {out:8.1f} tok/s total, "
          f"{1000/p50:6.1f} tok/s per user")
    return {"engine": name, "p50_ms": p50 * 1000, "p95_ms": p95 * 1000, "tok_s": out}


results = []

# %% [markdown]
# ## 4. Baseline: `transformers.generate`, one prompt at a time
#
# This is the floor. Each request waits for the previous one to finish all 64 tokens,
# because there is no batching and no continuous batching. Expect single-digit tokens
# per second of *total* throughput, and a per-request latency of roughly
# `64 × step`.

# %%
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to("cuda").eval()

ids = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
print("prompt lengths:", [len(i) for i in ids])


@torch.inference_mode()
def hf_one(prompt_ids):
    t = time.perf_counter()
    out = model.generate(prompt_ids.to("cuda"), max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return time.perf_counter() - t, out.shape[-1] - prompt_ids.shape[-1]


hf_one(ids[0])                      # warm-up: pay the one-time costs first
torch.cuda.synchronize()

lat, gen = [], 0
t0 = time.perf_counter()
for i in ids:
    sec, n = hf_one(i)
    lat.append(sec)
    gen += n
wall = time.perf_counter() - t0
results.append(report("PyTorch transformers, 1 at a time", lat, gen, wall))

# %% [markdown]
# **Read it.** Total throughput is `1 / per-request latency`, not the per-request speed:
# the GPU is idle between requests. That is exactly the cost continuous batching removes.

# %% [markdown]
# ## 5. The control: a hand-rolled batch
#
# This is the most important cell in the notebook. It uses the same PyTorch model, the
# same weights, the same greedy decoding — and puts a whole batch through the model in
# each step. If vLLM's advantage is mostly just batching, most of the gap closes here.
#
# **Why the prompts are truncated to a common length:** so that no padding is needed and
# the code stays obviously correct. That costs a few tokens on the longer prompts and is
# the same for every engine, so the comparison stays fair.

# %%
L = min(len(i) for i in ids)
batch = torch.stack([i[-L:] for i in ids]).to("cuda")
print("batch shape", tuple(batch.shape), "— all prompts truncated to the shortest length")


@torch.inference_mode()
def hf_batch_generate(input_ids, max_new, batch_size=None):
    """Greedy decode a batch, one forward pass per step, with a real KV cache."""
    B = input_ids.shape[0]
    batch_size = batch_size or B
    past = None
    cur = input_ids
    done = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
    steps = 0
    for _ in range(max_new):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        nxt = torch.where(done.unsqueeze(1), torch.zeros_like(nxt), nxt)
        done = done | (nxt.squeeze(1) == tok.eos_token_id)
        cur = nxt
        steps += 1
        if bool(done.all()):
            break
    return steps


torch.cuda.synchronize()
steps = hf_batch_generate(batch, 4)      # warm-up
print("warm-up ok, generated", steps, "steps")

# %% [markdown]
# Now measure the whole batch. `hf_batch_generate` needs `MAX_NEW` steps no matter how
# many rows it carries, instead of `MAX_NEW × rows` separate calls.

# %%
# Measure one batched step to get its cost, then estimate per-request latency honestly.
torch.cuda.synchronize()
t0 = time.perf_counter()
hf_batch_generate(batch, MAX_NEW)
torch.cuda.synchronize()
per_step_t = (time.perf_counter() - t0) / MAX_NEW
print(f"batched step (all {batch.shape[0]} rows) ≈ {per_step_t*1000:.1f} ms "
      f"-> {1000/per_step_t:.0f} tokens/s in total, "
      f"{1000/(per_step_t*batch.shape[0]):.0f} per user")

torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    n_steps = hf_batch_generate(batch, MAX_NEW)
torch.cuda.synchronize()
wall = time.perf_counter() - t0
lat = [n_steps * per_step_t] * batch.shape[0]
results.append(report(f"PyTorch, hand-rolled batch of {batch.shape[0]}", lat,
                      n_steps * batch.shape[0], wall))

# %% [markdown]
# **Read it, and compare the two tables.** The batched loop needs `MAX_NEW` steps no
# matter what, instead of `MAX_NEW × N_REQUESTS` steps. If total throughput has jumped
# by roughly the batch size, then **most of what people attribute to vLLM is just
# batching** — which is a real and useful finding, and you earned it by measuring rather
# than by reading a benchmark.
#
# What the hand-rolled batch does *not* do: it never lets a finished sequence leave
# early, so it wastes the tail of every batch. With a fixed `MAX_NEW` and greedy
# decoding that tail is small; with real traffic, where answers are different lengths, it
# is most of the waste. That is continuous batching, and it is what the vLLM rows show
# next.

# %% [markdown]
# ## 6. vLLM, at several `max_num_seqs`
#
# We use vLLM's in-process `LLM` class rather than the server, so all four settings share
# one notebook cell and one set of prompts. `--enforce-eager` (`enforce_eager=True`) keeps
# CUDA graphs off, which is the right call on a T4.
#
# Four engines in one process is a lot for 16 GB, so we free the PyTorch model first. If
# this cell ever runs out of memory, restart the runtime and run just one setting at a
# time — the comparison does not need them in the same process.

# %%
from vllm import LLM, SamplingParams

del model                     # the PyTorch baseline is done; give its memory back
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

for max_num_seqs in (1, 16, 64, 256):
    print(f"\n=== vllm.LLM  max_num_seqs={max_num_seqs} ===")
    llm = LLM(model=MODEL, dtype="float16", max_model_len=2048, gpu_memory_utilization=UTIL,
              max_num_seqs=max_num_seqs, enforce_eager=True, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
    outs = llm.generate(PROMPTS, sp)             # warm-up: compiles and allocates
    del outs

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    lat = [wall / len(PROMPTS)] * len(PROMPTS)
    results.append(report(f"vLLM, max_num_seqs={max_num_seqs}", lat, gen, wall))
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"    peak GPU memory {peak:.2f} GB")
    del llm
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 7. The matrix
#
# Fill this in and put it at the top of `REPORT.md`. Every column must travel together:
# a throughput number without its workload is not a result.

# %%
print(f"\n{'engine':<38} {'p50 ms':>9} {'p95 ms':>9} {'tok/s':>10}")
print("-" * 70)
for r in results:
    print(f"{r['engine']:<38} {r['p50_ms']:>9.0f} {r['p95_ms']:>9.0f} {r['tok_s']:>10.1f}")

with open("results_matrix.json", "w") as f:
    json.dump({"model": MODEL, "max_new_tokens": MAX_NEW, "n_requests": N_REQUESTS,
               "rows": results}, f, indent=2)
print("\nwrote results_matrix.json — that is your raw data. Never retype it.")

# %% [markdown]
# ## 8. Optimize the one GPU
#
# Write your latency target **now**, before you look at the numbers:
# *p95 per-request latency under 2 s, and at least 100 total output tokens/s.*
# Then sweep and find the setting that meets it with the most throughput.

# %%
TARGET_P95_MS = 2000
TARGET_TOK_S = 100

rows = [r for r in results if r["engine"].startswith("vLLM")]
meets = [r for r in rows if r["p95_ms"] <= TARGET_P95_MS and r["tok_s"] >= TARGET_TOK_S]
if meets:
    best = max(meets, key=lambda r: r["tok_s"])
    print(f"target: p95 <= {TARGET_P95_MS} ms and >= {TARGET_TOK_S} tok/s")
    print(f"operating point: {best['engine']}  "
          f"p95 {best['p95_ms']:.0f} ms  {best['tok_s']:.0f} tok/s")
else:
    print(f"nothing met the target (p95 <= {TARGET_P95_MS} ms, >= {TARGET_TOK_S} tok/s).")
    print("Either the workload is too small for a fair measurement, or you need a")
    print("longer answer and more prompts — raise MAX_NEW and PROMPTS and re-run.")

# %% [markdown]
# ## 9. Your turn: the report
#
# `REPORT.md`, four pages, in this order:
#
# 1. **Claim.** One sentence, with the workload inside it.
# 2. **Setup.** GPU, model, dtype, versions, the exact commands.
# 3. **Method.** Warm-up, what you timed, how, and what you did *not* measure.
# 4. **Results.** The matrix above, one plot, the operating point.
# 5. **Why.** Each difference tied to a mechanism from the course: paged KV, continuous
#    batching, the memory budget, and the fact that per-request speed is nearly
#    independent of batch size.
# 6. **Limits.** What would change on a bigger GPU, with a longer context, with
#    quantization (Module 11) or prefix caching (Module 10).
#
# Questions to answer before you call it done:
#
# - Your hand-rolled batch recovered a lot of vLLM's throughput. What exactly did it *not*
#   recover, and which component of the engine is responsible?
# - vLLM at `max_num_seqs=1` is not much faster than the naive loop. Why should that be
#   true, and what would it mean if it were not?
# - Only one of the two concurrency axes you measured shows up in the throughput column.
#   Which, and why?
# - Where did your p95 come from: the queue, the prefill, or the first decode step?
# - Name one thing this harness does not measure that a reviewer would ask about.
#
# **Rubric.** Method reproducible 20% · numbers honest 20% · explanation mechanistic 25% ·
# latency and throughput reported together 15% · operating point defended 10% · limits
# acknowledged 10%.