# %% [markdown]
# # Lab 04 · Run your first LLM
#
# **Tokens per Second** · Module 04
#
# This is the lab where everything in the first four modules becomes a thing you can touch. You will:
#
# 1. **Load** a real open-weights model onto the GPU and see it fit (or not) in memory.
# 2. **Generate** from it, and read the answer token by token.
# 3. **Watch** the GPU work: prefill versus decode, TTFT versus TPOT, memory climbing.
# 4. **Stream** tokens out one at a time.
#
# Everything runs on a free **Colab T4 (16 GB)** with a small model, so it works top to bottom without a credit card.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# **Write your measured numbers into your notes.** Module 05 starts from the KV-cache numbers you produce here.

# %% [markdown]
# ## 0. What GPU am I on, and what can it actually do?
#
# Say what you expect before you run it.

# %%
import subprocess, sys, importlib

assert importlib.util.find_spec("torch"), "Install PyTorch first if this cell fails."

try:
    import torch
    import transformers
except ImportError:
    !pip -q install -U torch transformers accelerate
    import torch
    import transformers

p = torch.cuda.get_device_properties(0)
print(f"GPU                 {p.name}")
print(f"Memory              {p.total_memory/1e9:.1f} GB")
print(f"Multiprocessors     {p.multi_processor_count}")
print(f"Compute capability  sm_{p.major}{p.minor}")
print(f"torch               {torch.__version__}")
print(f"transformers        {transformers.__version__}")
print()
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
     "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip())

# %% [markdown]
# > **A note on precision.** The T4 is a Turing GPU: it has FP16 tensor cores but **no native BF16**
# > tensor cores and no FP8. So on this machine we use `torch.float16`, not `bfloat16`. On an A100,
# > H100 or later you would use BF16 — same 2 bytes per weight, same memory arithmetic, but the
# > hardware accelerates it natively. The memory math in this course is identical either way.

# %%
DTYPE = torch.float16     # 2 bytes per weight, and this T4 can actually do it fast

# %% [markdown]
# ## 1. Choose a model — and check that it fits, before downloading
#
# Module 04's first lesson: **parameters × bytes per weight** is the memory floor. Predict first.
#
# We will use **Qwen2.5-0.5B-Instruct**: small, fast, chatty, and it leaves the T4 almost empty.

# %%
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PARAMS = 0.494e9           # public spec figure for Qwen2.5-0.5B
BYTES_PER_WEIGHT = 2

predicted_gb = PARAMS * BYTES_PER_WEIGHT / 1e9
free_gb = p.total_memory / 1e9
print(f"model                 {MODEL}")
print(f"parameters            {PARAMS/1e9:.2f}B")
print(f"predicted weights     {predicted_gb:.2f} GB  ({PARAMS/1e9:.2f}B x {BYTES_PER_WEIGHT} bytes)")
print(f"GPU memory            {free_gb:.1f} GB")
print(f"predicted headroom    {free_gb - predicted_gb:.1f} GB  ({100*(1-predicted_gb/free_gb):.0f}% spare)")
print()
print("Also downloading the KV cache later; at ~12 KB/token a 4,000-token")
print(f"conversation costs only {4000*12.288e3/1e9:.3f} GB. Plenty of room.")
print("(Section 6 derives the exact figure from the model config.)")

# %% [markdown]
# ## 2. Load it
#
# `dtype` sets the bytes per weight. `device_map="cuda"` puts it on the GPU as it loads.
#
# The first load downloads about 1 GB and takes a minute. Later runs use the Colab cache.

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL)

# `dtype` is the modern argument name; older transformers called it `torch_dtype`.
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cuda")
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, device_map="cuda")

print(f"loaded in {time.perf_counter()-t0:.1f}s")

model.eval()   # no dropout, no gradient bookkeeping: inference mode

# %%
actual_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
print(f"measured weights   {actual_gb:.2f} GB")
print(f"predicted          {predicted_gb:.2f} GB")
print(f"error              {100*(actual_gb-predicted_gb)/predicted_gb:+.1f}%")
print()
print(f"parameters         {sum(p.numel() for p in model.parameters())/1e9:.3f}B")
print(f"dtype              {next(model.parameters()).dtype}")
print(f"device             {next(model.parameters()).device}")
print(f"layers             {model.config.num_hidden_layers}")
print(f"hidden size        {model.config.hidden_size}")
print(f"query heads        {model.config.num_attention_heads}")
print(f"KV heads           {model.config.num_key_value_heads}   <- GQA: {model.config.num_attention_heads // model.config.num_key_value_heads}x smaller KV cache")
print(f"vocabulary         {model.config.vocab_size}")
print(f"torch GPU alloc    {torch.cuda.memory_allocated()/1e9:.2f} GB")

# %%
print("\nArchitecture straight from config:")
print(model.config)

# %% [markdown]
# ## 3. Tokenize: text becomes numbers
#
# The model never sees your letters. It sees integers that index the embedding table.

# %%
prompt = "What is a GPU?"
ids = tok(prompt, return_tensors="pt").to("cuda")

print(f"{prompt!r}")
print(f"{ids['input_ids'].shape[1]} tokens: {ids['input_ids'][0].tolist()}")
print()
for i, tid in enumerate(ids['input_ids'][0].tolist()):
    print(f"  {i:>2}  {tid:>7}  {tok.decode([tid])!r}")

# %%
print("The same tokenizer, on a longer text:")
long = "Serving a model over HTTP means one process, one GPU, and many requests at once."
print(f"{len(tok(long).input_ids)} tokens from {len(long)} characters "
      f"({len(long)/max(1,len(tok(long).input_ids)):.1f} characters per token)")

# %% [markdown]
# ## 4. Generate
#
# The chat template is not optional. An instruct model was trained on a specific
# conversation format, and that format lives on the tokenizer.

# %%
messages = [
    {"role": "system", "content": "You are terse. Answer in at most two sentences."},
    {"role": "user", "content": "What is a GPU?"},
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== what actually goes into the model ===")
print(text)
print("=== end ===\n")

inputs = tok(text, return_tensors="pt").to("cuda")
print(f"{inputs['input_ids'].shape[1]} tokens after templating "
      f"(vs {len(tok(prompt).input_ids)} for the raw question)")

# %%
out = model.generate(
    **inputs,
    max_new_tokens=96,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.eos_token_id,
)

prompt_len = inputs["input_ids"].shape[1]
new_ids = out[0][prompt_len:]          # only the tokens it generated

print(f"prompt:    {prompt_len} tokens")
print(f"generated: {len(new_ids)} tokens")
print()
print(tok.decode(new_ids, skip_special_tokens=True))

# %%
print("What greedy decoding produces (do_sample=False):")
greedy = model.generate(**inputs, max_new_tokens=96, do_sample=False,
                        pad_token_id=tok.eos_token_id)
print(tok.decode(greedy[0][prompt_len:], skip_special_tokens=True))

# %% [markdown]
# ## 5. Watch the GPU: TTFT and TPOT
#
# Two different numbers with two different causes. Module 04's whole point.

# %%
def bench(fn, iters=20, warmup=5):
    """Seconds per call, measured on the GPU clock (Module 02's helper)."""
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

print(f"weights are {actual_gb:.2f} GB and this GPU has {p.total_memory/1e9:.1f} GB")
print(f"a single decode step cannot be faster than "
      f"{actual_gb*1e9/320e9*1e3:.1f} ms (weights / 320 GB/s)\n")

# %%
def measure(prompt_text, max_new=32, iters=5):
    """Returns (ttft_ms, per_token_ms) using a streaming iterator."""
    from transformers import TextIteratorStreamer
    from threading import Thread

    txt = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                  tokenize=False, add_generation_prompt=True)
    inp = tok(txt, return_tensors="pt").to("cuda")

    # Warm up: the first call is 2-10x slower.
    model.generate(**inp, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)

    ttfts, tputs = [], []
    for _ in range(iters):
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        w = Thread(target=model.generate, kwargs={
            **inp, "max_new_tokens": max_new, "do_sample": False,
            "streamer": streamer, "pad_token_id": tok.eos_token_id})
        w.start()

        times = []
        for _ in streamer:
            times.append((time.perf_counter() - t0) * 1e3)
        w.join()

        if len(times) < 3:
            continue
        ttfts.append(times[0])
        tputs.append((times[-1] - times[0]) / (len(times) - 1))

    return sum(ttfts)/len(ttfts), sum(tputs)/len(tputs)

# %%
print(f"{'prompt tokens':>14} {'TTFT (ms)':>11} {'per token (ms)':>16} {'tokens/s':>10}")
rows = []
for n_words in [4, 16, 64, 256]:
    long_prompt = "Tell me about GPUs. " * n_words
    n_tok = len(tok(long_prompt).input_ids)
    ttft, tpot = measure(long_prompt, max_new=32)
    rows.append((n_tok, ttft, tpot))
    print(f"{n_tok:>14} {ttft:>11.1f} {tpot:>16.2f} {1000/tpot:>10.1f}")

# %%
base_n, base_ttft, base_tpot = rows[0]
print("\nRead this table before moving on:")
print("- **TTFT grows with prompt length.** That is prefill: more prompt tokens means more")
print("  compute before the first token can exist.")
print("- **Per-token time is nearly flat.** Every decode step reads the same ~1 GB of weights,")
print("  so a longer prompt barely changes it. That is why a long prompt is a TTFT problem")
print("  and a decode-speed problem is not.")

# %% [markdown]
# ## 6. The KV cache, growing
#
# Each generated token adds its key and value to the cache. Watch it happen.

# %%
def kv_bytes_per_token(model):
    c = model.config
    return 2 * c.num_hidden_layers * c.num_key_value_heads * c.head_dim * 2  # K+V, 2 bytes

kvpt = kv_bytes_per_token(model)
print(f"KV bytes per token   {kvpt:,} bytes = {kvpt/1024:.2f} KB")
print(f"  = 2 (K and V) x {model.config.num_hidden_layers} layers "
      f"x {model.config.num_key_value_heads} KV heads "
      f"x head_dim {model.config.head_dim} x 2 bytes")
print()
for ctx in [512, 2048, 8192, 32768, 131072]:
    gb = ctx * kvpt / 1e9
    print(f"  context {ctx:>7,} tokens  ->  KV {gb:>7.3f} GB per conversation")

# %%
# Watch the actual cache grow during one generation.
torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_allocated()

long_prompt = "Explain " * 200
inp = tok(tok.apply_chat_template([{"role": "user", "content": long_prompt}],
                                  tokenize=False, add_generation_prompt=True),
          return_tensors="pt").to("cuda")
plen = inp["input_ids"].shape[1]

streamer_out = model.generate(**inp, max_new_tokens=128, do_sample=False,
                              pad_token_id=tok.eos_token_id)
after = torch.cuda.memory_allocated()

print(f"prompt was            {plen} tokens")
print(f"generated             {streamer_out.shape[1] - plen} tokens")
print(f"memory before         {before/1e9:.3f} GB")
print(f"memory after          {after/1e9:.3f} GB")
print(f"growth                {(after-before)/1e6:.1f} MB")
print(f"predicted KV growth   {(plen + 128) * kvpt / 1e6:.1f} MB")
print()
print("The memory goes up and stays up. Nothing is freed until the request ends.")
print("That is exactly what Module 05 is about.")

# %% [markdown]
# ## 7. Streaming: the same work, felt differently

# %%
from transformers import TextIteratorStreamer
from threading import Thread

messages = [{"role": "user", "content": "Describe the ocean in three sentences."}]
txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(txt, return_tensors="pt").to("cuda")

streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
worker = Thread(target=model.generate, kwargs={
    **inputs,
    "max_new_tokens": 120,
    "do_sample": True, "temperature": 0.7, "streamer": streamer,
    "pad_token_id": tok.eos_token_id})
worker.start()

t0 = time.perf_counter()
times = []
print("streaming (each line is one token as it appeared):")
for piece in streamer:
    times.append((time.perf_counter() - t0) * 1e3)
    print(f"  [{times[-1]:7.1f} ms] {piece!r}", flush=True)
worker.join()
total = (time.perf_counter() - t0) * 1e3
n = len(times)

# %%
print(f"\n{n} tokens, {total:.0f} ms wall clock")
if n >= 2:
    print(f"time to first token   {times[0]:.1f} ms")
    print(f"average gap           {(times[-1]-times[0])/(n-1):.1f} ms per token")
print(f"throughput            {1000*n/total:.1f} tokens/s")
print()
print("Compare the felt experience above with section 4, where the whole answer")
print("appeared at once. Identical GPU work. The GPU does not care which you chose.")

# %% [markdown]
# ## 8. Sampling: watch a distribution, not just a token

# %%
inp = tok(tok.apply_chat_template([{"role": "user", "content": "The capital of France is"}],
                                  tokenize=False, add_generation_prompt=True),
          return_tensors="pt").to("cuda")

with torch.no_grad():
    logits = model(**inp).logits[0, -1].float()

probs = torch.softmax(logits, dim=-1)
top = torch.topk(probs, 8)

print(f"vocabulary size: {probs.shape[0]:,}")
print(f"\n{'token':>12} {'id':>8} {'probability':>12}")
for p, i in zip(top.values, top.indices):
    print(f"{tok.decode([i])!r:>12} {i.item():>8} {p.item()*100:>11.2f}%")

# %%
print("\nThe same logits at three temperatures:")
print(f"{'temperature':>12} {'top token p':>13} {'top-5 mass':>11} {'effective options':>18}")
for temp in [0.1, 0.7, 1.5]:
    q = torch.softmax(logits / temp, dim=-1)
    tq = torch.topk(q, 5)
    # Effective number of choices = exp(entropy): how many tokens are genuinely live.
    ent = -(q[q > 0] * q[q > 0].log()).sum()
    print(f"{temp:>12.1f} {tq.values[0].item()*100:>12.2f}% "
          f"{tq.values.sum().item()*100:>10.2f}% {ent.exp().item():>18.1f}")

# %%
print("\nGenerate the same prompt five times at temperature 1.2:")
txt = tok.apply_chat_template([{"role": "user", "content": "Name one colour."}],
                              tokenize=False, add_generation_prompt=True)
inp = tok(txt, return_tensors="pt").to("cuda")
for i in range(5):
    o = model.generate(**inp, max_new_tokens=8, do_sample=True, temperature=1.2,
                       top_p=1.0, pad_token_id=tok.eos_token_id)
    print(f"  {i+1}. {tok.decode(o[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)!r}")

# %% [markdown]
# ## 9. Measure the cost of one token, from first principles
#
# Module 02 said decode is bandwidth-bound. Prove it on your own numbers.

# %%
my_bw = 320e9                       # T4 spec, GB/s -> bytes/s
weight_bytes = actual_gb * 1e9
floor_ms = weight_bytes / my_bw * 1e3

_, measured_tpot = measure("Tell me about GPUs. " * 4, max_new=32)

print(f"weights               {actual_gb:.2f} GB = {weight_bytes/1e9:.2f} x 10^9 bytes")
print(f"bandwidth (T4 spec)   {my_bw/1e9:.0f} GB/s")
print(f"floor per token       {floor_ms:.2f} ms")
print(f"measured per token    {measured_tpot:.2f} ms")
print(f"efficiency            {100*floor_ms/measured_tpot:.0f}% of the bandwidth floor")
print()
print(f"overhead per step     {measured_tpot - floor_ms:.2f} ms")
print("That gap is Python, kernel launches and sampling. It is not the GPU being slow —")
print("and it is the first thing Modules 15 and 16 go after.")

# %%
# Confirm it is bandwidth, not FLOPs: a second copy of the model in FP32 must read
# twice the bytes and take about twice as long. (Built from the config, so the FP16
# model is untouched and later cells still work.)
from transformers import AutoConfig, AutoModelForCausalLM

print("\nSame architecture, same prompt, different dtype.")
print("Bytes double, so if decode is bandwidth-bound the time should follow.")
print(f"  {'dtype':>6} {'weights':>10} {'ms/token':>10} {'ratio':>7}")

fp32_model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, device_map="cuda").eval()

def measure_with(m, prompt_text, max_new=32, iters=3):
    from transformers import TextIteratorStreamer
    from threading import Thread
    txt = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                  tokenize=False, add_generation_prompt=True)
    inp = tok(txt, return_tensors="pt").to("cuda")
    m.generate(**inp, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    gaps = []
    for _ in range(iters):
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        w = Thread(target=m.generate, kwargs={**inp, "max_new_tokens": max_new,
                   "do_sample": False, "streamer": streamer,
                   "pad_token_id": tok.eos_token_id})
        w.start()
        t0 = time.perf_counter()
        times = [(time.perf_counter() - t0) * 1e3 for _ in streamer]
        w.join()
        if len(times) >= 3:
            gaps.append((times[-1] - times[0]) / (len(times) - 1))
    return sum(gaps) / len(gaps)

w16 = sum(q.numel() * q.element_size() for q in model.parameters()) / 1e9
w32 = sum(q.numel() * q.element_size() for q in fp32_model.parameters()) / 1e9

tpot16 = measure_with(model, "Tell me about GPUs. " * 4)
tpot32 = measure_with(fp32_model, "Tell me about GPUs. " * 4)
print(f"  {'FP16':>6} {w16:>9.2f}G {tpot16:>9.2f}ms {1.0:>6.2f}x")
print(f"  {'FP32':>6} {w32:>9.2f}G {tpot32:>9.2f}ms {tpot32/tpot16:>6.2f}x")
print(f"\n  bytes ratio {w32/w16:.2f}x, time ratio {tpot32/tpot16:.2f}x")
print("  Close to 1.00 would mean the extra time came from something other than bytes.")

del fp32_model
torch.cuda.empty_cache()

# %% [markdown]
# ## 10. Watch the GPU from outside
#
# In Colab, open a new tab, or run this cell to sample `nvidia-smi` while a
# generation is running.

# %%
import threading, re

stop = threading.Event()
samples = []

def sample_gpu():
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            u, m, pw = (x.strip() for x in r.stdout.strip().split(","))
            samples.append((int(u), float(m), float(pw)))
        except Exception:
            pass
        stop.wait(0.25)

t = threading.Thread(target=sample_gpu, daemon=True)
t.start()

inp = tok(tok.apply_chat_template([{"role": "user", "content": "Count slowly from one to twenty."}],
                                  tokenize=False, add_generation_prompt=True),
          return_tensors="pt").to("cuda")
model.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
stop.set(); t.join(timeout=2)

if samples:
    utils = [s[0] for s in samples]; mems = [s[1] for s in samples]; pws = [s[2] for s in samples]
    print(f"samples              {len(samples)}")
    print(f"utilization  min/avg/max   {min(utils)}/{sum(utils)/len(utils):.0f}/{max(utils)} %")
    print(f"memory (MB)  min/avg/max   {min(mems):.0f}/{sum(mems)/len(mems):.0f}/{max(mems):.0f}")
    print(f"power (W)    min/avg/max   {min(pws):.0f}/{sum(pws)/len(pws):.0f}/{max(pws):.0f}")
    print()
    print("Utilization is 'was any kernel running', not 'was it efficient'.")
    print("Averages of a boolean. Always pair it with tokens per second.")
else:
    print("Could not sample nvidia-smi. Open a second notebook tab and run `nvidia-smi -l 1`.")

# %% [markdown]
# ## 11. Write these down
#
# These are the numbers Module 05 assumes you have.

# %%
my_notes = {
    "model": MODEL,
    "weights_GB": round(actual_gb, 3),
    "predicted_GB": round(predicted_gb, 3),
    "kv_bytes_per_token": kvpt,
    "TTFT_ms_short_prompt": round(rows[0][1], 1),
    "TTFT_ms_long_prompt": round(rows[-1][1], 1),
    "per_token_ms": round(measured_tpot, 2),
    "tokens_per_second": round(1000 / measured_tpot, 1),
    "bandwidth_floor_ms": round(floor_ms, 2),
    "decode_efficiency_pct": round(100 * floor_ms / measured_tpot),
    "peak_memory_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
}

print("=== write these into My notes ===")
for k, v in my_notes.items():
    print(f"{k:>26}  {v}")
print("====================================\n")
print("Module 05 asks: why does generation recompute everything, and what does the")
print("KV cache save? The bytes-per-token figure above is the raw material.")

# %% [markdown]
# ## Your turn
#
# 1. **Section 1**: try `Qwen/Qwen2.5-1.5B-Instruct` and `HuggingFaceTB/SmolLM2-360M-Instruct`.
#    Write down the measured weights each time and check your prediction first. Does the
#    prediction error change, and why?
# 2. **Section 5**: set `max_new` to 8 and to 256. Why does per-token time barely change
#    while TTFT does not change at all?
# 3. **Section 6**: run with `max_new_tokens=512` and a 2,000-token prompt. How much KV
#    cache is allocated, and does it match `context × kv_bytes_per_token`?
# 4. **Section 8**: set `temperature=0.05` and regenerate. Then `temperature=2.0`. Where does
#    the text stop being English, and why is that not a bug?
# 5. **Section 9**: repeat the FP16-versus-FP32 comparison with a larger batch. Does the
#    ratio stay at 2x? Why might it not?
# 6. **Challenge**: load `meta-llama/Llama-3.1-8B` (gated — accept the licence on the Hub
#    first) in FP16 and try to load it on this T4. Read the out-of-memory error, then
#    explain it in one sentence using `parameters × bytes per weight`.
