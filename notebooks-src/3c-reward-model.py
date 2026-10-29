# %% [markdown]
# # Lab 3C · Train a tiny reward model
#
# **Tokens per Second** · Module 3C, From Autocomplete to Assistant
#
# You will build, in about 20 minutes on the free Colab T4, the second stage of the
# InstructGPT recipe:
#
# 1. **Look at real preference data**: Anthropic's HH-RLHF, pairs of assistant replies where
#    a person chose one.
# 2. **Train a reward model** with the Bradley-Terry loss, `-logsigmoid(r_chosen - r_rejected)`,
#    on a few thousand pairs.
# 3. **Measure it** with pairwise accuracy on held-out pairs, and compare it with a dumb
#    baseline ("the longer answer wins").
# 4. **Use it**: generate 4 answers with `Qwen2.5-0.5B-Instruct` and keep the one the reward
#    model likes best (best-of-N), timing what that costs.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
# The T4 has FP16 tensor cores but no BF16 ones, so training uses FP16 mixed precision.

# %% [markdown]
# ## 0. Setup
#
# `datasets` and `transformers` are preinstalled on Colab. The cell below only checks the GPU
# and fixes the random seeds so your numbers are close to the ones quoted in the lesson.

# %%
import math, random, time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."
print("GPU  ", torch.cuda.get_device_name(0))
print("torch", torch.__version__)

random.seed(0)
torch.manual_seed(0)

# %% [markdown]
# ## 1. Real preference data
#
# HH-RLHF (Bai et al., 2022) holds about 160,800 training and 8,552 test comparisons in total.
# We use its `helpful-base` subset, where people chose the more helpful of two replies.
#
# Each row has two full transcripts, `chosen` and `rejected`. They share the whole
# conversation and differ **only in the last assistant turn**. Print one and check.

# %%
ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
print(ds)

def split_last_turn(text):
    """Return (conversation so far, final assistant reply)."""
    i = text.rfind("\n\nAssistant:")
    return text[:i], text[i + len("\n\nAssistant:"):].strip()

ex = ds["train"][3]
prompt_c, reply_c = split_last_turn(ex["chosen"])
prompt_r, reply_r = split_last_turn(ex["rejected"])
print("same conversation before the last reply:", prompt_c == prompt_r)
print("\n--- conversation (last 500 chars) ---")
print(prompt_c[-500:])
print("\n--- CHOSEN reply ---\n", reply_c[:400])
print("\n--- REJECTED reply ---\n", reply_r[:400])

# %% [markdown]
# Read a few more (change the index). You will find pairs where you agree with the labeler,
# pairs where both replies are poor, and pairs where you would have chosen the other one.
# That disagreement is real: in the InstructGPT project labelers agreed with each other only
# about three times out of four. It caps how well **any** reward model can do on this data.

# %% [markdown]
# ## 2. A reward model is a transformer with a one-number head
#
# We use `distilroberta-base` (82M parameters): small enough to train in a couple of minutes.
# `AutoModelForSequenceClassification` with `num_labels=1` puts a single linear output on top
# of the encoder: exactly the "swap the vocabulary head for a scalar" from lesson 3C.3.
#
# Two details matter:
# - **Truncate from the left.** Conversations can be longer than 512 tokens, and the part
#   that differs is at the *end*. Default truncation cuts the end off and leaves the model
#   comparing two identical texts.
# - **Score chosen and rejected in one forward pass**, by stacking them into one batch.
#
# One difference from the decoder reward models in the lesson: RoBERTa is an *encoder*, with
# attention in both directions, so its head reads the first token (`<s>`), which already sees
# the whole text. A decoder reward model reads the **last** token instead, because under the
# causal mask only the last position has seen everything.

# %%
RM_NAME = "distilroberta-base"
MAX_LEN = 384

rm_tok = AutoTokenizer.from_pretrained(RM_NAME)
rm_tok.truncation_side = "left"
rm = AutoModelForSequenceClassification.from_pretrained(RM_NAME, num_labels=1).cuda()
print(f"reward model parameters: {sum(p.numel() for p in rm.parameters())/1e6:.1f} M")

def rm_scores(texts):
    enc = rm_tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        return rm(**enc).logits.squeeze(-1).float()

# %% [markdown]
# ## 3. The Bradley-Terry loss, checked by hand
#
# Before training, confirm the formula from the lesson: with scores 1.2 (chosen) and 0.4
# (rejected), the preference probability is σ(0.8) = 0.690 and the loss is 0.371 nats.

# %%
r_w, r_l = torch.tensor(1.2), torch.tensor(0.4)
print(f"P(chosen preferred) = sigma({(r_w - r_l).item():.1f}) = {torch.sigmoid(r_w - r_l).item():.3f}")
print(f"loss = -log sigma    = {-F.logsigmoid(r_w - r_l).item():.3f} nats")
print(f"wrong order, loss    = {-F.logsigmoid(r_l - r_w).item():.3f} nats")
print(f"shift both by +5     = {-F.logsigmoid((r_w + 5) - (r_l + 5)).item():.3f} nats  (only the gap matters)")

# %% [markdown]
# ## 4. Pairwise accuracy, before training
#
# Pairwise accuracy: the fraction of held-out pairs where the chosen reply gets the higher
# score. An untrained head is a coin flip, so expect about 50%.
#
# Alongside it, a baseline that needs no model at all: **the longer reply wins**. If a reward
# model cannot beat this, it has mostly learned length.

# %%
N_TRAIN, N_TEST = 4000, 1000
train = ds["train"].shuffle(seed=0).select(range(N_TRAIN))
test = ds["test"].shuffle(seed=0).select(range(N_TEST))

@torch.no_grad()
def pairwise_accuracy(data, batch=32):
    rm.eval()
    right, gaps = 0, []
    for i in range(0, len(data), batch):
        rows = data[i:i + batch]
        n = len(rows["chosen"])
        s = rm_scores(rows["chosen"] + rows["rejected"])
        gap = s[:n] - s[n:]
        right += (gap > 0).sum().item()
        gaps.append(gap.cpu())
    return right / len(data), torch.cat(gaps)

def length_baseline(data):
    wins = [len(split_last_turn(c)[1]) > len(split_last_turn(r)[1]) for c, r in zip(data["chosen"], data["rejected"])]
    return sum(wins) / len(wins)

acc0, _ = pairwise_accuracy(test)
print(f"untrained reward model   : {acc0:.1%}")
print(f"'longer reply wins'      : {length_baseline(test):.1%}")

# %% [markdown]
# ## 5. Train
#
# Two epochs over 4,000 pairs, batch of 16 pairs (32 sequences), AdamW at 2e-5 with a short
# warm-up. FP16 autocast with a gradient scaler, because the T4 has no BF16. Expect two to
# four minutes.
#
# Watch two numbers: the loss, which starts at ln 2 = 0.693 (a coin flip) and should fall,
# and the training accuracy on the current batch.

# %%
BATCH, EPOCHS, LR = 16, 2, 2e-5
steps_total = EPOCHS * math.ceil(N_TRAIN / BATCH)
opt = torch.optim.AdamW(rm.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps_total))
scaler = torch.amp.GradScaler("cuda")

step, t0 = 0, time.time()
for epoch in range(EPOCHS):
    order = list(range(N_TRAIN))
    random.shuffle(order)
    run_loss, run_acc = 0.0, 0.0
    for i in range(0, N_TRAIN, BATCH):
        rm.train()
        rows = train.select(order[i:i + BATCH])
        n = len(rows)
        s = rm_scores(rows["chosen"] + rows["rejected"])
        r_w, r_l = s[:n], s[n:]
        loss = -F.logsigmoid(r_w - r_l).mean()          # Bradley-Terry
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        run_loss += loss.item()
        run_acc += (r_w > r_l).float().mean().item()
        if step % 50 == 0:
            print(f"epoch {epoch}  step {step:4d}/{steps_total}  loss {run_loss/50:.3f}  "
                  f"batch acc {run_acc/50:.1%}  {time.time()-t0:5.0f} s")
            run_loss, run_acc = 0.0, 0.0
print(f"trained in {time.time()-t0:.0f} s")

# %% [markdown]
# ## 6. Measure it
#
# **Expected:** held-out pairwise accuracy clearly above 50%, roughly in the high 50s to
# mid 60s for this small model and data. If you see 50–52%, something is wrong: most often
# the truncation side, so the two texts the model sees are identical.
#
# Do not expect 90%. People disagree with each other on this data, so even large reward
# models stay well below that. Then look at **how confident** the model is: the spread of the
# score gaps.

# %%
acc1, gaps = pairwise_accuracy(test)
print(f"untrained               : {acc0:.1%}")
print(f"trained reward model    : {acc1:.1%}")
print(f"'longer reply wins'     : {length_baseline(test):.1%}")
p = torch.sigmoid(gaps)
print(f"\nmean P(chosen) the model assigns on held-out pairs: {p.mean():.3f}")
print(f"pairs where it is >90% sure: {(p > 0.9).float().mean():.1%}, "
      f"and right on {((gaps > 0) & (p > 0.9)).float().sum() / max(1, (p > 0.9).sum()):.1%} of those")

# %% [markdown]
# **Does it just like long answers?** Score every held-out reply on its own and correlate the
# score with the reply's length. A strong positive correlation is the length bias that
# lesson 3C.4 warned would be exploited by an optimiser.

# %%
@torch.no_grad()
def score_replies(data, batch=32):
    rm.eval()
    out, lens = [], []
    for i in range(0, len(data), batch):
        rows = data[i:i + batch]
        texts = rows["chosen"] + rows["rejected"]
        out.append(rm_scores(texts).cpu())
        lens += [len(split_last_turn(t)[1]) for t in texts]
    return torch.cat(out), torch.tensor(lens, dtype=torch.float)

sc, ln = score_replies(test.select(range(400)))
corr = torch.corrcoef(torch.stack([sc, ln]))[0, 1].item()
print(f"correlation between reward and reply length: {corr:+.2f}")

# %% [markdown]
# ## 7. Best-of-N with a real chat model
#
# Now use the reward model the way lesson 3C.3 described: generate N = 4 answers from
# `Qwen2.5-0.5B-Instruct`, score each, keep the highest.
#
# The reward model was trained on HH-RLHF's format, so we score answers written into that
# format: `"\n\nHuman: ...\n\nAssistant: ..."`.

# %%
GEN = "Qwen/Qwen2.5-0.5B-Instruct"
gtok = AutoTokenizer.from_pretrained(GEN)
gen = AutoModelForCausalLM.from_pretrained(GEN, torch_dtype=torch.float16).cuda().eval()

def chat_ids(prompt):
    text = gtok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return gtok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")

def generate(prompt, n=4, max_new=160, temperature=0.9, top_p=0.95):
    ids = chat_ids(prompt)
    out = gen.generate(ids, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new,
                       num_return_sequences=n, pad_token_id=gtok.eos_token_id)
    return [gtok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip() for o in out]

def hh_format(prompt, answer):
    return f"\n\nHuman: {prompt}\n\nAssistant: {answer}"

PROMPTS = [
    "How do I get my toddler to sleep through the night?",
    "What's a good way to start learning to play the guitar as an adult?",
    "Can you explain what inflation is in simple terms?",
    "I have a job interview tomorrow. How should I prepare tonight?",
    "How can I make my houseplants stop dying?",
    "What should I pack for a three-day hiking trip?",
    "My friend is upset with me and I don't know why. What should I do?",
    "How do I boil an egg so the yolk is still soft?",
]

torch.manual_seed(1)
results = []
for q in PROMPTS:
    answers = generate(q, n=4)
    with torch.no_grad():
        s = rm_scores([hh_format(q, a) for a in answers]).cpu()
    results.append((q, answers, s))

q, answers, s = results[0]
best, worst = s.argmax().item(), s.argmin().item()
print("PROMPT:", q)
print("scores:", [f"{v:+.2f}" for v in s.tolist()])
print(f"\n--- BEST (reward {s[best]:+.2f}) ---\n{answers[best]}")
print(f"\n--- WORST (reward {s[worst]:+.2f}) ---\n{answers[worst]}")

# %% [markdown]
# Read the best and worst for a few prompts (change `results[0]`). Do you agree with the
# reward model? Where you do not, can you see what it was rewarding instead?
#
# Now the numbers best-of-N is supposed to move: the reward of the picked answer against a
# randomly picked one, and the length of each.

# %%
pick_r = torch.stack([s.max() for _, _, s in results]).mean().item()
rand_r = torch.stack([s.mean() for _, _, s in results]).mean().item()
pick_len = sum(len(a[s.argmax()]) for _, a, s in results) / len(results)
rand_len = sum(sum(len(x) for x in a) / len(a) for _, a, s in results) / len(results)
print(f"mean reward, best of 4   : {pick_r:+.2f}")
print(f"mean reward, random pick : {rand_r:+.2f}")
print(f"mean length, best of 4   : {pick_len:.0f} chars")
print(f"mean length, random pick : {rand_len:.0f} chars")

# %% [markdown]
# The best-of-4 reward is higher **by construction**: you took a maximum. That is not evidence
# the answers are better. Only reading them (or an independent judge, Module B2) tells you
# that. If the picked answers are also much longer, your reward model's length bias is doing
# part of the choosing.

# %% [markdown]
# ## 8. What best-of-N costs
#
# Best-of-4 means four times the generated tokens. But on a GPU, decoding is limited by
# reading the weights (Module 02), and four sequences in one batch share each read.
# Time it: one answer, then four answers in one batched call, then the reward-model scoring.

# %%
def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps

q = PROMPTS[0]
ids = chat_ids(q)
fixed = dict(do_sample=True, temperature=0.9, max_new_tokens=128, min_new_tokens=128, pad_token_id=gtok.eos_token_id)
t1 = timed(lambda: gen.generate(ids, num_return_sequences=1, **fixed))
t4 = timed(lambda: gen.generate(ids, num_return_sequences=4, **fixed))
answers = generate(q, n=4)
ts = timed(lambda: rm_scores([hh_format(q, a) for a in answers]))
print(f"1 answer,  128 tokens  : {t1*1000:7.0f} ms")
print(f"4 answers, 128 tokens  : {t4*1000:7.0f} ms   ({t4/t1:.2f}x the time for 4x the tokens)")
print(f"score 4 with the RM    : {ts*1000:7.0f} ms")
print(f"best-of-4 overhead     : {(t4 + ts)/t1:.2f}x a single answer")

# %% [markdown]
# **Expected:** four answers take well under four times as long as one, often only 1.1–1.5×,
# because each decode step reads the 1 GB of weights once for all four rows. The reward model
# adds tens of milliseconds. Best-of-N is cheaper than it sounds at small N, and Part 3 of the
# course (batching) explains exactly why. At large N the batch stops being free, and the
# selection effect from section 7 grows.

# %% [markdown]
# ## Check yourself
#
# 1. Why must the reward model's tokenizer truncate from the **left** for this data?
# 2. Your trained model reaches, say, 62% pairwise accuracy. Name two reasons the ceiling on
#    this data is far below 100%.
# 3. The mean best-of-4 reward is higher than the random-pick reward. Why is that, on its own,
#    not evidence that best-of-4 answers are better?
# 4. Four answers took about 1.3× the time of one. Which resource limits a decode step, and
#    why does a batch of four share it?
#
# ## Your turn
#
# 1. **More data.** Set `N_TRAIN = 8000` (twice the time). How much does held-out accuracy
#    move? Compare with LIMA's lesson about quantity versus quality.
# 2. **Bigger N.** Run best-of-8 and best-of-16 on the same prompts. Does the picked answer's
#    length keep growing? That is reward hacking in miniature.
# 3. **Length control.** Subtract `0.002 * len(answer)` from each score before picking. Does
#    the choice change, and do you prefer the new picks?
# 4. **A decoder as the reward model (stretch).** Replace `RM_NAME` with
#    `"Qwen/Qwen2.5-0.5B-Instruct"` (set `rm_tok.pad_token = rm_tok.eos_token` and
#    `rm.config.pad_token_id = rm_tok.pad_token_id`), use `BATCH = 4` and `MAX_LEN = 512`.
#    It reads the score at the last non-padding token, exactly as lesson 3C.3 described. Is
#    it more accurate, and how much slower is it to train?
# 5. **Write in your notes:** your untrained, trained and length-baseline accuracies, the
#    length correlation, and the best-of-4 time overhead.

# %%
print("done. notes:")
print(f"  pairwise accuracy: untrained {acc0:.1%}, trained {acc1:.1%}, length baseline {length_baseline(test):.1%}")
print(f"  reward-length correlation: {corr:+.2f}")
print(f"  best-of-4 cost: {(t4 + ts)/t1:.2f}x one answer")
