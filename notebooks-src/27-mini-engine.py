# %% [markdown]
# # Lab 27 · Build a mini inference engine
#
# **Tokens per Second** · Module 27
#
# Four lessons, four components, one working engine. By the end of this notebook you
# will have written your own **vLLM, small**: a model runner, a continuous-batching
# scheduler, a paged KV cache with prefix caching, and a streaming HTTP API with
# metrics — and you will have benchmarked it against the real thing.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# This is not a "follow along and it works" lab. The cells give you the load-bearing
# pieces; **you write the rest**. Every section ends with a task that has no answer
# in the output.
#
# **Write every number into your notes.** Your report at the end of the project is
# built from them.

# %% [markdown]
# ## 0. What am I running on?
#
# The T4 is 16 GB and about 320 GB/s. It has FP16 tensor cores but no BF16 ones, so
# everything here is `float16`. Predict the memory and bandwidth before you run it.

# %%
import math, time, gc, json, os, statistics
from collections import deque, defaultdict
import torch

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
print(f"GPU        {p.name}")
print(f"memory     {p.total_memory/1e9:.1f} GB")
print(f"SMs        {p.multi_processor_count}")
print(f"torch      {torch.__version__}")

DEVICE = "cuda"
DTYPE = torch.float16

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
# Qwen2.5-0.5B-Instruct fits comfortably on a T4 and is small enough that your engine's
# own overhead is visible — which is the point. Three config fields decide everything
# about the KV cache: layers, KV heads, head dimension.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=DTYPE).to(DEVICE).eval()
cfg = model.config

L, NKV, DH = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
params = sum(p.numel() for p in model.parameters())
WEIGHTS_BYTES = params * 2
KV_BYTES_PER_TOKEN = 2 * L * NKV * DH * 2

print(f"model         {NAME}")
print(f"params        {params/1e9:.3f} B")
print(f"layers        {L}   kv_heads {NKV}   head_dim {DH}   hidden {cfg.hidden_size}")
print(f"weights       {WEIGHTS_BYTES/1e9:.2f} GB in {DTYPE}")
print(f"KV per token  {KV_BYTES_PER_TOKEN:,} bytes = {KV_BYTES_PER_TOKEN/1024:.1f} KB")
print(f"KV at 4k ctx  {4096*KV_BYTES_PER_TOKEN/1e9:.3f} GB per user")

# %% [markdown]
# ## 2. Step 1 — the model runner
#
# The contract is one sentence: **a decode step takes one token per sequence and gives
# back one token per sequence.** The work is to keep the KV cache correct.
#
# Your tasks:
# 1. Fill in `KVCache.append` so a new K/V lands at `blocks[-1]`, slot `n % block_size`.
# 2. Make `execute` call the model and return one token per sequence.
# 3. **Verify**: your output must match `model.generate()` token for token.

# %%
BLOCK_SIZE = 16

class KVCache:
    """A flat pool of fixed-size blocks. One tensor; sequences are lists of block ids."""

    def __init__(self, num_blocks, block_size=BLOCK_SIZE):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.k = torch.zeros(L, num_blocks, block_size, NKV, DH,
                             dtype=DTYPE, device=DEVICE)
        self.v = torch.zeros_like(self.k)
        self.used = set()                     # block ids currently handed out

    def allocate(self, n_blocks):
        free = [b for b in range(self.num_blocks) if b not in self.used]
        if len(free) < n_blocks:
            return []                          # failure is normal: the scheduler preempts
        for b in free[:n_blocks]:
            self.used.add(b)
        return free[:n_blocks]

    def free(self, blocks):
        for b in blocks:
            self.used.discard(b)

    def append(self, layer, blocks, slot, k, v):
        """Write one token's K/V. TODO: fill me in."""
        raise NotImplementedError

    @property
    def blocks_used(self):
        return len(self.used)


# %%
# A request is just a list of token ids plus where its KV lives.
class Req:
    _next = 0
    def __init__(self, token_ids, max_new):
        Req._next += 1
        self.id = Req._next
        self.ids = list(token_ids)
        self.max_new = max_new
        self.generated = []
        self.blocks = []                      # physical block ids
        self.finished = False

    @property
    def tokens(self):
        return self.ids + self.generated

    @property
    def num_tokens(self):
        return len(self.tokens)

    def blocks_needed(self):
        return math.ceil(self.num_tokens / BLOCK_SIZE)


# %%
class ModelRunner:
    def __init__(self, model, kv, tokenizer):
        self.model = model
        self.kv = kv
        self.tok = tokenizer
        self.rng = torch.Generator(device=DEVICE)
        self.rng.manual_seed(0)               # seeded: an engine you cannot repeat is not an instrument

    def prefill(self, req):
        """Process the whole prompt, writing K/V for every token. TODO."""
        raise NotImplementedError

    def execute(self, reqs):
        """One decode step over many sequences. Returns one token id per sequence. TODO."""
        raise NotImplementedError

    def sample(self, logits, temperature=0.0):
        """logits: [B, vocab] -> [B] token ids."""
        if temperature <= 0:
            return logits.argmax(-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1, generator=self.rng).squeeze(-1)


# %% [markdown]
# ### Check 2a — does your runner agree with `transformers`?
#
# Fill in `prefill` and `execute`, then run the cell below. **If the tokens do not
# match exactly, everything downstream is measuring the wrong engine.** Debug it before
# you continue — the table in the project lesson tells you which bug you probably have.

# %%
NUM_BLOCKS = 300
kv = KVCache(NUM_BLOCKS)
runner = ModelRunner(model, kv, tok)

prompt = "Explain continuous batching in one sentence."
ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
req = Req(ids, max_new=24)

# --- your code runs here ---
try:
    runner.prefill(req)
    out = []
    for _ in range(req.max_new):
        t = runner.execute([req])[0]
        out.append(int(t))
        if t == tok.eos_token_id:
            break
    print("mine:      ", repr(tok.decode(out)))
except NotImplementedError:
    print("TODO: implement prefill and execute first")

with torch.no_grad():
    ref = model.generate(torch.tensor([ids]).to(DEVICE), max_new_tokens=24,
                         do_sample=False, use_cache=True)
ref_txt = tok.decode(ref[0][len(ids):], skip_special_tokens=True)
print("transformers:", repr(ref_txt))

# %% [markdown]
# ## 3. Step 2 — the continuous batching scheduler
#
# The scheduler holds two lists and moves requests between them. Everything else is
# bookkeeping, which is exactly why it can be pure Python.
#
# Your tasks:
# 1. `schedule()` returns the batch for this iteration and fills the waiting queue.
# 2. Finished sequences free their blocks **immediately** — this is the whole point.
# 3. Add a `max_num_seqs` cap and a `max_batched_tokens` budget.
# 4. **Measure**: run the same 8 requests through a static batcher and through yours,
#    and compare iterations to completion.

# %%
class Scheduler:
    def __init__(self, runner, max_num_seqs=16, max_batched_tokens=2048, block_size=BLOCK_SIZE):
        self.runner = runner
        self.waiting = deque()
        self.running = []
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.block_size = block_size
        self.preemptions = 0
        self.iterations = 0

    def add_request(self, req):
        self.waiting.append(req)

    def free_finished(self):
        """O(1) amortised per request, via the request's own block list. TODO."""
        raise NotImplementedError

    def admit(self, budget):
        """Move waiting -> running while there is room and memory. TODO.
        Allocate BEFORE admitting; return False if allocation failed."""
        raise NotImplementedError

    def schedule(self):
        """One engine iteration. Returns the batch to execute."""
        raise NotImplementedError


# %%
# The engine loop: schedule, run, sample, emit. This is the shape of a real engine.
def run_engine(sched, max_iterations=200, max_concurrency=8, verbose=False):
    """Drive the scheduler until every request finishes. Returns per-request ITLs."""
    itls = defaultdict(list)
    it = 0
    while (sched.running or sched.waiting) and it < max_iterations:
        batch = sched.schedule()
        if not batch:
            break
        t0 = time.perf_counter()
        tokens = sched.runner.execute(batch)
        dt = time.perf_counter() - t0
        it += 1
        sched.iterations = it
        for req, tokid in zip(batch, tokens):
            req.generated.append(int(tokid))
            itls[req.id].append(dt)
            if tokid == sched.runner.tok.eos_token_id or len(req.generated) >= req.max_new:
                req.finished = True
    return itls


# %% [markdown]
# ### Check 3a — continuous vs static batching
#
# Build a mixed-length workload: a short-chat majority and a long tail. Then run it both
# ways and record how many iterations each takes. This is the number that justifies
# the whole lesson.

# %%
PROMPTS = [
    ("Hi", 12), ("What is a GPU?", 40), ("Explain KV caching.", 64),
    ("Write a haiku about latency.", 24), ("What is a tensor core?", 48),
    ("Summarize PagedAttention.", 72), ("Hi again", 8), ("What is TTFT?", 56),
]

def make_reqs():
    Req._next = 0
    out = []
    for text, mx in PROMPTS:
        ids = tok(text, return_tensors="pt").input_ids[0].tolist()
        out.append(Req(ids, mx))
    return out

# static batching: fixed groups, each runs to its own longest member
def run_static(reqs, group=4):
    kv2 = KVCache(NUM_BLOCKS)
    r2 = ModelRunner(model, kv2, tok)
    iters = 0
    for i in range(0, len(reqs), group):
        batch = reqs[i:i + group]
        longest = max(r.max_new for r in batch)
        try:
            for r in batch:
                r2.prefill(r)
            for _ in range(longest):
                toks = r2.execute(batch)
                iters += 1
                for r, t in zip(batch, toks):
                    r.generated.append(int(t))
        except NotImplementedError:
            return None
    return iters

try:
    reqs = make_reqs()
    sched = Scheduler(runner)
    for r in reqs:
        sched.add_request(r)
    run_engine(sched)
    print(f"continuous: {sched.iterations} iterations")
except NotImplementedError:
    print("TODO: implement the scheduler first")

static_iters = run_static(make_reqs())
if static_iters is not None:
    print(f"static:     {static_iters} iterations")
    print(f"speedup:    {static_iters/max(1,sched.iterations):.2f}x fewer iterations")

# %% [markdown]
# ## 4. Step 3 — prefix caching
#
# A block whose hash chains to its parent's hash *is* its prefix. Two requests sharing a
# system prompt then share memory with no explicit matching.
#
# Your task: implement `PrefixCache` and show that the second request with a shared
# system prompt prefills **fewer tokens** than the first.

# %%
SHARED_SYSTEM = ("You are a precise technical assistant. Answer in at most two sentences. "
                 "Use exact terminology and show the arithmetic behind any claim.")

class PrefixCache:
    def __init__(self, block_size=BLOCK_SIZE):
        self.block_size = block_size
        self.blocks = {}          # hash -> physical block id
        self.refcount = defaultdict(int)

    def hash_block(self, token_ids, block_index):
        """TODO: hash this block's tokens together with its parent's hash."""
        raise NotImplementedError

    def match(self, token_ids):
        """How many leading tokens can be reused from cache? Returns tokens, not blocks."""
        raise NotImplementedError


# %%
# How much would it save? The arithmetic, before you write the code.
def prefix_savings(shared_tokens, n_requests, block_size=BLOCK_SIZE, kv_bytes=KV_BYTES_PER_TOKEN):
    blocks = shared_tokens // block_size
    saved_blocks = blocks * (n_requests - 1)
    return blocks, saved_blocks, saved_blocks * block_size * kv_bytes

b, sb, gb = prefix_savings(len(tok(SHARED_SYSTEM).input_ids), 8)
print(f"shared prompt   {len(tok(SHARED_SYSTEM).input_ids)} tokens -> {b} reusable blocks")
print(f"at 8 requests   {sb} blocks saved = {gb/1e6:.1f} MB of KV memory")
print(f"per second at 8 req/s: {gb*8/1e6:.0f} MB/s of allocation avoided")

# %% [markdown]
# ## 5. Step 4 — metrics
#
# Record five counters per iteration. They cost microseconds and you cannot retrofit
# them later.

# %%
class Metrics:
    def __init__(self):
        self.history = []

    def record(self, sched, tokens_out, dt):
        row = {
            "iteration": sched.iterations,
            "running": len(sched.running),
            "waiting": len(sched.waiting),
            "cache_usage": kv.blocks_used / kv.num_blocks,
            "preemptions": sched.preemptions,
            "tokens": tokens_out,
            "step_ms": dt * 1000,
        }
        self.history.append(row)
        return row

m = Metrics()
print("record(sched, tokens_out, dt) -> one row per iteration")

# %%
# Once your engine runs, sweep the batch cap and plot the two curves you care about.
print("TODO: sweep max_num_seqs in [1, 2, 4, 8, 16, 32] and record")
print("      throughput (tokens/s) and TPOT (ms) per setting")

# %% [markdown]
# ## 6. Compare against a real engine
#
# This is the point of the module. Install vLLM and measure the **same** workload through
# the **same** harness. Your engine will be slower — the goal is to find out by how much,
# and to attribute the gap to a specific component rather than to "Python".

# %%
# Uncomment once your own engine passes Check 2a.
# !pip install -q vllm

# %%
# The workload file both engines must read. Pin the seed and the token counts so the
# comparison is fair — count tokens with the tokenizer, never with len(string).
WORKLOAD = []
import random
random.seed(1234)
for text, mx in PROMPTS:
    ids = tok(text, return_tensors="pt").input_ids[0].tolist()
    WORKLOAD.append({"prompt_ids": ids, "max_new": mx})
print(f"{len(WORKLOAD)} prompts, prompt tokens: {[len(w['prompt_ids']) for w in WORKLOAD]}")
print(f"output tokens: {[w['max_new'] for w in WORKLOAD]}")
print("total output tokens:", sum(w['max_new'] for w in WORKLOAD))

with open("workload.json", "w") as f:
    json.dump({"seed": 1234, "requests": WORKLOAD}, f)

# %%
# The harness both engines run through. Streaming, client-side clock, raw ITLs stored.
def measure(engine_call, requests, concurrency=8):
    """engine_call(prompt_ids, max_new) -> iterator of token ids.
    Returns per-request TTFT, ITLs, token counts."""
    from concurrent.futures import ThreadPoolExecutor
    results = []

    def one(req):
        t0 = time.perf_counter()
        ttft, last, itls, n = None, t0, [], 0
        for tid in engine_call(req["prompt_ids"], req["max_new"]):
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            else:
                itls.append(now - last)
            last, n = now, n + 1
        return {"ttft_s": ttft, "itls_s": itls, "e2e_s": last - t0, "output_tokens": n}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(one, requests))
    return results

# %%
def summarise(results, label):
    ttfts = [r["ttft_s"] for r in results]
    tps = [statistics.mean(r["itls_s"]) for r in results if r["itls_s"]]
    total = sum(r["output_tokens"] for r in results)
    wall = max(r["ttft_s"] + r["e2e_s"] for r in results)
    print(f"\n{label}")
    print(f"  TTFT      p50 {statistics.median(ttfts)*1000:7.1f} ms   max {max(ttfts)*1000:7.1f} ms")
    print(f"  TPOT      p50 {statistics.median(tps)*1000:7.2f} ms" if tps else "  TPOT  n/a")
    print(f"  tokens    {total} in {wall:.2f} s = {total/wall:,.0f} tok/s")
    # the identity from the lesson: if this fails, your timing is wrong
    for r in results:
        if r["itls_s"]:
            err = abs(r["e2e_s"] - (r["ttft_s"] + (len(r["itls_s"])) * statistics.mean(r["itls_s"]))) / r["e2e_s"]
            if err > 0.05:
                print(f"  !! identity failed for a request: {err*100:.1f}% error")
    return {"ttft_p50": statistics.median(ttfts), "tpot_p50": statistics.median(tps) if tps else None,
            "throughput": total / wall, "tokens": total}

# %%
# Your engine, through the same harness:
def my_engine_call(prompt_ids, max_new):
    req = Req(list(prompt_ids), max_new)
    sched.add_request(req)
    while not req.finished and sched.iterations < 10000:
        sched.schedule()
        toks = sched.runner.execute([req] if req in sched.running else sched.running)
        for r, t in zip(sched.running, toks):
            r.generated.append(int(t))
            yield int(t)
            if t == tok.eos_token_id or len(r.generated) >= r.max_new:
                r.finished = True
    if req.finished:
        yield from []

# %%
# vLLM, through the same harness (in-process engine, no HTTP):
# from vllm import LLM, SamplingParams
# llm = LLM(model=NAME, dtype="float16", gpu_memory_utilization=0.75, max_model_len=2048)
# def vllm_call(prompt_ids, max_new):
#     sp = SamplingParams(temperature=0.0, max_tokens=max_new)
#     out = llm.generate([{"prompt_token_ids": prompt_ids}], sp, use_tqdm=False)
#     for o in out:
#         for tid in o.outputs[0].token_ids:
#             yield tid

# %% [markdown]
# ## 7. My notes
#
# The numbers your report is built from. Fill these in as you go.

NOTES = {
    "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
    "weights_gb": round(WEIGHTS_BYTES / 1e9, 2),
    "pool_blocks": NUM_BLOCKS,
    "static_iterations": static_iters,
    "continuous_iterations": getattr(sched, "iterations", None) if "sched" in dir() else None,
    "speedup": None,
    "prefix_blocks_saved": prefix_savings(len(tok(SHARED_SYSTEM).input_ids), 8)[1],
    "matches_transformers": None,     # True only if Check 2a passed exactly
}
print(json.dumps(NOTES, indent=2))

print("""
## What to report

1. Does your engine match `transformers.generate()` token for token? If not, which bug?
2. Static vs continuous: how many fewer iterations, and why exactly that many?
3. Your batch sweep: where does TPOT stop being flat, and does it match B_knee = weights / (ctx x KV/token)?
4. Prefix caching: what fraction of your system prompt was reused?
5. Your engine vs vLLM on the same workload: the throughput ratio, and **which component**
   you believe accounts for the gap (runner, scheduler, paged KV, or the stream layer)?
""")

# %% [markdown]
# ## Where this goes next
#
# **Module 28 · Research & Capstone.** You now have an instrument you built yourself and
# understand down to the block table. That is exactly the position from which reading
# PagedAttention's paper, or SGLang's radix cache, is a small step rather than a wall.