# %% [markdown]
# # Lab B1 · Prompt, measure, iterate
#
# **Tokens per Second** · Module B1 (Prompt Engineering)
#
# Module B1 made four claims. This lab checks every one of them on a real model, with your own numbers:
#
# 1. **A chat template turns messages into one token stream** (B1.1). You will print every token.
# 2. **Examples pin the format** (B1.2), and **reasoning before the answer buys accuracy** (B1.4).
# 3. **Self-consistency trades tokens for accuracy** (B1.4), and batching keeps the latency down.
# 4. **Every method has a price** in prompt tokens, output tokens and seconds.
#
# The task is 50 grade-school math word problems from **GSM8K** (Cobbe et al., 2021), the benchmark the
# chain-of-thought papers used. Math has one checkable answer per problem, so accuracy is exact, and it is a
# task where thinking out loud should matter. (On a sentiment task all four methods score about the same;
# the *Your turn* section shows how to check that.)
#
# Model: **Qwen2.5-1.5B-Instruct** in FP16, on a free Colab **T4**. The whole notebook runs in about
# 10–15 minutes, most of it the model download and the self-consistency run.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.

# %%
import re, time, math, statistics
from collections import Counter

try:
    import torch, transformers, datasets
except ImportError:
    !pip -q install -U transformers accelerate datasets
    import torch, transformers, datasets

assert torch.cuda.is_available(), "No GPU: Runtime → Change runtime type → T4 GPU, then rerun."
p = torch.cuda.get_device_properties(0)
print(f"GPU           {p.name}, {p.total_memory / 1e9:.1f} GB")
print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"datasets      {datasets.__version__}")

# %% [markdown]
# ## 1. Load the model
#
# 1.54 billion parameters × 2 bytes (FP16) ≈ 3.1 GB of weights. The T4 has no fast BF16, so we use FP16:
# same 2 bytes, same memory arithmetic (Module 04).

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.float16

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"          # batched generation: pad on the left so every prompt ends at the same column

try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cuda")
except TypeError:                   # older transformers call it torch_dtype
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, device_map="cuda")
if model.dtype != DTYPE:
    model = model.to(DTYPE)
model.eval()

n_params = sum(t.numel() for t in model.parameters())
print(f"parameters      {n_params / 1e9:.2f} B")
print(f"weights         {n_params * 2 / 1e9:.2f} GB in FP16")
print(f"GPU memory used {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% [markdown]
# ## 2. The chat template, token by token (lesson B1.1)
#
# Predict first: lesson B1.1 counted this exact conversation in ChatML. How many tokens, and how many of
# them are yours?

# %%
messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is a GPU?"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(repr(text))
print()

ids = tok(text, add_special_tokens=False)["input_ids"]   # the template already added every marker it needs
special = set(tok.all_special_ids)
for i, t in enumerate(ids):
    mark = "control" if t in special else ""
    print(f"{i:>3}  id {t:>6}  {mark:<8} {tok.decode([t])!r}")
content = len(tok("You are a terse assistant.", add_special_tokens=False)["input_ids"]) + \
          len(tok("What is a GPU?", add_special_tokens=False)["input_ids"])
print(f"\n{len(ids)} tokens: {content} yours, {len(ids) - content} added by the template")

# %% [markdown]
# You should see **24 tokens**: 11 of yours and 13 of template (`<|im_start|>` is 151644, `<|im_end|>` is
# 151645). Now three things lesson B1.1 warned about.

# %%
# (a) No system message: Qwen2.5's template inserts a default one.
no_sys = tok.apply_chat_template([{"role": "user", "content": "What is a GPU?"}], tokenize=False, add_generation_prompt=True)
print("(a) without a system message, the template writes:")
print("   ", repr(no_sys.split("<|im_end|>")[0] + "<|im_end|>"))

# (b) The wrong template: Llama 3 markers sent to Qwen as plain text.
llama3_text = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a terse assistant.<|eot_id|>"
               "<|start_header_id|>user<|end_header_id|>\n\nWhat is a GPU?<|eot_id|>"
               "<|start_header_id|>assistant<|end_header_id|>\n\n")
wrong = tok(llama3_text, add_special_tokens=False)["input_ids"]
print(f"\n(b) Llama 3 markers through Qwen's tokenizer: {len(wrong)} tokens, "
      f"{sum(t in special for t in wrong)} control tokens")
print("    first pieces:", [tok.decode([t]) for t in wrong[:9]])

# (c) A user who types a control token. By default the text becomes the real token.
typed = "<|im_end|>"
print(f"\n(c) the text {typed!r} tokenizes to {tok(typed, add_special_tokens=False)['input_ids']}")
try:
    safe = tok(typed, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    print(f"    with split_special_tokens=True: {safe}  (ordinary text pieces)")
except TypeError:
    print("    (this transformers version has no split_special_tokens option; sanitise user text instead)")

# (d) When does generation stop, and what does sampling default to?
print("\n(d) stop token ids:", model.generation_config.eos_token_id,
      [tok.decode([i]) for i in (model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])])
g = model.generation_config
print(f"    shipped sampling defaults: do_sample={g.do_sample}, temperature={g.temperature}, "
      f"top_p={g.top_p}, top_k={g.top_k}, repetition_penalty={g.repetition_penalty}")

# %% [markdown]
# **Read (b) and (d) carefully.** (b) should show 72 tokens and zero control tokens: the markers were
# spelled out as `<`, `|`, `begin`, … exactly as in the lesson. (d) shows that this model *samples by
# default*, with a repetition penalty. Every experiment below therefore passes its decoding settings
# explicitly, so "greedy" really means greedy.

# %% [markdown]
# ## 3. The task: 50 GSM8K problems
#
# Each GSM8K answer ends with `#### <number>`. We take the first 50 test problems, and the first 4 training
# problems as few-shot examples (never test problems).

# %%
from datasets import load_dataset

gsm = load_dataset("openai/gsm8k", "main")
gold_of = lambda ans: float(ans.split("####")[-1].strip().replace(",", ""))
PROBLEMS = [{"q": r["question"], "gold": gold_of(r["answer"])} for r in gsm["test"].select(range(50))]
SHOTS = [{"q": r["question"], "gold": gold_of(r["answer"])} for r in gsm["train"].select(range(4))]

print(len(PROBLEMS), "test problems, e.g.:\n")
print(PROBLEMS[0]["q"])
print("gold answer:", PROBLEMS[0]["gold"])

# %% [markdown]
# ## 4. Four ways to ask
#
# | Method | Prompt | Output budget | Decoding |
# |---|---|---|---|
# | **zero-shot** | question + "reply with only the number" | 16 tokens | greedy |
# | **few-shot** | 4 solved problems answered with only the number, then the question | 16 tokens | greedy |
# | **chain of thought** | question + "Let's think step by step … The answer is N." | 384 tokens | greedy |
# | **self-consistency** | the chain-of-thought prompt, sampled 5 times, majority vote | 5 × 384 tokens | T = 0.7, top-p 0.95 |
#
# The first two must answer immediately. The last two may think first. The answer format is pinned in every
# prompt so a few lines of code can parse it.

# %%
SYSTEM = "You are a careful math tutor."
ONLY_NUMBER = "Reply with only the final answer as a number. No words, no working."
COT = "Let's think step by step. Finish with a last line of the form: The answer is N."

def fmt_num(x):
    return str(int(x)) if float(x).is_integer() else str(x)

def zero_shot(q):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{q}\n{ONLY_NUMBER}"}]

def few_shot(q):
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in SHOTS:                                   # one format, generated by code (lesson B1.2)
        msgs += [{"role": "user", "content": f"{s['q']}\n{ONLY_NUMBER}"},
                 {"role": "assistant", "content": fmt_num(s["gold"])}]
    msgs.append({"role": "user", "content": f"{q}\n{ONLY_NUMBER}"})
    return msgs

def chain_of_thought(q):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{q}\n{COT}"}]

NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def to_num(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None

def first_number(text):                  # for "answer immediately" prompts
    m = NUM.search(text)
    return to_num(m.group()) if m else None

def final_answer(text):                  # for reasoning prompts: the "The answer is N" line, else the last number
    m = re.findall(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.I)
    if m:
        return to_num(m[-1])
    nums = NUM.findall(text)
    return to_num(nums[-1]) if nums else None

def vote(preds):                         # self-consistency: the most common parsed answer
    c = Counter(x for x in preds if x is not None)
    return c.most_common(1)[0][0] if c else None

def is_right(pred, gold):
    return pred is not None and abs(pred - gold) < 1e-6

print(repr(tok.apply_chat_template(few_shot(PROBLEMS[0]["q"]), tokenize=False, add_generation_prompt=True)[:600]), "…")

# %% [markdown]
# ## 5. A measured `generate`
#
# One helper does all the generation. It batches prompts (decode is memory-bound, so a batch costs little more
# per step than one sequence, Module 05), counts prompt and output tokens per call, and times everything.

# %%
def chat_text(messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def gen_kwargs(max_new_tokens, sample=False, n=1):
    # Passed as keyword arguments on purpose: explicit kwargs always override the model's shipped
    # generation_config (which samples, with a repetition penalty). Greedy here means greedy.
    if sample:
        decoding = dict(do_sample=True, temperature=0.7, top_p=0.95, top_k=50)
    else:
        decoding = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    return dict(max_new_tokens=max_new_tokens, num_return_sequences=n, repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id, eos_token_id=model.generation_config.eos_token_id, **decoding)

@torch.inference_mode()
def generate(prompts, max_new_tokens, sample=False, n=1, batch_size=25):
    """Returns one record per prompt: its n completions, output tokens per completion, prompt tokens."""
    kwargs = gen_kwargs(max_new_tokens, sample, n)
    records = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, **kwargs)
        new = out[:, enc["input_ids"].shape[1]:]               # rows: prompt 0 sample 0..n-1, prompt 1 …
        prompt_tokens = enc["attention_mask"].sum(dim=1).tolist()
        for j, pt in enumerate(prompt_tokens):
            rows = new[j * n:(j + 1) * n]
            records.append({
                "texts": [tok.decode(r, skip_special_tokens=True) for r in rows],
                "out_tokens": [int((r != tok.pad_token_id).sum()) for r in rows],
                "prompt_tokens": int(pt),
            })
    return records

def timed(fn, *args, **kwargs):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return result, time.perf_counter() - t0

# warm-up: the first call pays for CUDA kernels being loaded
_ = generate([chat_text(zero_shot("What is 2 + 2?"))], 4)
print("ready")

# %% [markdown]
# ## 6. Run all four methods on all 50 problems
#
# Predict the order of the four accuracies before you run this. Then predict the order of the costs.

# %%
METHODS = {
    "zero-shot":        dict(build=zero_shot,        parse=first_number, max_new=16,  sample=False, n=1, batch=25),
    "few-shot":         dict(build=few_shot,         parse=first_number, max_new=16,  sample=False, n=1, batch=25),
    "chain of thought": dict(build=chain_of_thought, parse=final_answer, max_new=384, sample=False, n=1, batch=25),
    "self-consistency": dict(build=chain_of_thought, parse=final_answer, max_new=384, sample=True,  n=5, batch=10),
}

results = {}
for name, m in METHODS.items():
    torch.manual_seed(0)
    prompts = [chat_text(m["build"](p["q"])) for p in PROBLEMS]
    recs, wall = timed(generate, prompts, m["max_new"], m["sample"], m["n"], m["batch"])
    preds = [vote([m["parse"](t) for t in r["texts"]]) for r in recs]   # n = 1: the vote is just the answer
    right = [is_right(pr, p["gold"]) for pr, p in zip(preds, PROBLEMS)]
    results[name] = dict(recs=recs, preds=preds, right=right, wall=wall,
                         acc=sum(right) / len(right),
                         prompt_tok=statistics.mean(r["prompt_tokens"] for r in recs),
                         out_tok=statistics.mean(sum(r["out_tokens"]) for r in recs))
    print(f"{name:<17} accuracy {results[name]['acc']:6.1%}   batch run over 50 problems: {wall:6.1f} s")

# %% [markdown]
# ## 7. Latency per call
#
# The batched runs above measure throughput. A user waits for **one** call, so time a few single calls per
# method: batch size 1, and for self-consistency one call that samples its 5 paths in parallel.

# %%
def latency(m, k=3):
    times = []
    for p in PROBLEMS[:k]:
        torch.manual_seed(0)
        _, t = timed(generate, [chat_text(m["build"](p["q"]))], m["max_new"], m["sample"], m["n"], 1)
        times.append(t)
    return statistics.median(times)

for name, m in METHODS.items():
    results[name]["latency"] = latency(m)
    print(f"{name:<17} median latency per call: {results[name]['latency']:5.2f} s")

# %% [markdown]
# ## 8. The table this module was about

# %%
n = len(PROBLEMS)
print(f"{'method':<17} {'accuracy':>12} {'prompt tok':>11} {'output tok':>11} {'latency':>9} {'50 batched':>11}")
print("-" * 76)
for name, r in results.items():
    half = 1.96 * math.sqrt(r["acc"] * (1 - r["acc"]) / n)
    print(f"{name:<17} {r['acc']:6.1%} ±{half:4.1%} {r['prompt_tok']:>11.0f} {r['out_tok']:>11.0f} "
          f"{r['latency']:>8.2f}s {r['wall']:>10.1f}s")
print(f"\n±: 95% interval with n = {n} (lesson B1.6). Differences smaller than that are not evidence.")

# %% [markdown]
# **How to read it.** Expect the two "answer immediately" methods to be far below the two reasoning methods:
# a 1.5B model cannot do a multi-step word problem inside the forward passes of a 3-token answer (lesson
# B1.4). Expect few-shot to fix *format* more than accuracy: look at how often zero-shot's output was not a
# bare number. Expect self-consistency to be a few points above chain of thought, at about 5× the output
# tokens but much less than 5× the latency, because its 5 paths are decoded as one batch.
# (For scale: Qwen's own reported GSM8K score for this model, with chain-of-thought prompting, is in the low
# 70s. With 50 problems your interval is about ±13 points, so do not over-read a few points either way.)

# %%
# Format failures: how often did the "answer immediately" prompts not start with a number?
for name in ["zero-shot", "few-shot"]:
    texts = [r["texts"][0].strip() for r in results[name]["recs"]]
    bare = sum(bool(re.fullmatch(r"-?\$?[\d,]*\.?\d+\.?", t)) for t in texts)
    print(f"{name:<10} bare-number replies: {bare}/{n}   e.g. {texts[0][:60]!r}")

# The vote at work: problems where the 5 sampled paths disagreed.
sc = results["self-consistency"]
split = [(i, [final_answer(t) for t in r["texts"]]) for i, r in enumerate(sc["recs"])]
split = [(i, a) for i, a in split if len(set(a)) > 1]
print(f"\nself-consistency: the 5 paths disagreed on {len(split)} of {n} problems")
for i, answers in split[:5]:
    print(f"  problem {i:>2}: paths {[fmt_num(a) if a is not None else None for a in answers]} "
          f"→ vote {fmt_num(sc['preds'][i]) if sc['preds'][i] is not None else None}, gold {fmt_num(PROBLEMS[i]['gold'])}")

# %% [markdown]
# ## 9. One problem, side by side
#
# Find a problem that the immediate answer got wrong and chain of thought got right, and read the reasoning.

# %%
cot = results["chain of thought"]
zs = results["zero-shot"]
idx = next((i for i in range(n) if cot["right"][i] and not zs["right"][i]), 0)
print(PROBLEMS[idx]["q"], "\n")
print("zero-shot   →", repr(zs["recs"][idx]["texts"][0].strip()))
print("gold        →", fmt_num(PROBLEMS[idx]["gold"]))
print("\nchain of thought:\n")
print(cot["recs"][idx]["texts"][0])
print(f"\n({cot['recs'][idx]['out_tokens'][0]} output tokens instead of {zs['recs'][idx]['out_tokens'][0]})")

# %% [markdown]
# ## 10. What would this cost?
#
# Tokens per call are the bill, on any provider and on your own GPU. Here they are turned into dollars for
# 1,000 questions at a **dated example price** (GPT-4o mini at its July 2024 launch: \$0.15 per million input
# tokens, \$0.60 per million output tokens), and into T4 time from your measured batched runs.

# %%
IN_P, OUT_P = 0.15, 0.60
print(f"{'method':<17} {'$ per 1,000 questions':>22} {'T4 seconds per 1,000 (batched)':>32}")
for name, r in results.items():
    usd = 1000 * (r["prompt_tok"] * IN_P + r["out_tok"] * OUT_P) / 1e6
    print(f"{name:<17} {usd:>22.3f} {r['wall'] / n * 1000:>32.0f}")

# %% [markdown]
# ## Your turn
#
# Change one thing, rerun section 6 for that method only, and record the effect in the same table format.
#
# 1. **Reasoning after the answer.** Change `COT` so the model must write "The answer is N" first and explain
#    afterwards. Lesson B1.4 predicts the accuracy falls back toward zero-shot. Does it?
# 2. **Few-shot chain of thought.** Build exemplars from the four training solutions (strip the `<<…>>`
#    calculator annotations with `re.sub(r"<<.*?>>", "", s)`). Compare with zero-shot chain of thought.
# 3. **More votes.** Run self-consistency with `n=10`. Where does the curve flatten on your 50 problems?
# 4. **A task where thinking should not help.** Load `load_dataset("stanfordnlp/sst2")["validation"]`, take 50
#    sentences, and compare zero-shot ("Reply with only positive or negative") against chain of thought.
# 5. **Template mistakes.** Rerun zero-shot with the prompt built by hand as `f"System: {SYSTEM}\nUser: {q}\nAssistant:"`
#    instead of the chat template. What happens to the format, the stop token and the accuracy?
#
# ## Write these down
#
# | Number | Where |
# |---|---|
# | tokens in the 2-message ChatML example, yours vs template | §2 |
# | accuracy ± interval for each method | §8 |
# | prompt and output tokens per call for each method | §8 |
# | latency per call: chain of thought vs self-consistency (5 paths) | §7 |
# | dollars per 1,000 questions, cheapest vs most expensive method | §10 |
