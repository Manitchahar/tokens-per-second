# %% [markdown]
# # Lab 3A · Fill the blank vs finish the sentence
#
# **Tokens per Second** · Module 3A
#
# One machine, two jobs. You will put the two games of this module side by side on real
# models and check every number the lessons derived:
#
# 1. **BERT fills blanks** with `bert-base-uncased`, using both sides of the gap.
# 2. **The masking recipe**, measured: 15% chosen, then 80/10/10, and the MLM loss.
# 3. **GPT-2 finishes sentences** with `gpt2` (124M): per-token surprise, perplexity and bits per byte.
# 4. **An untrained model starts at ln V.** Check it.
# 5. **Swap the jobs.** BERT cannot continue a sentence well; GPT-2 cannot see the right-hand side.
# 6. **Supervision per pass, and why decoders can cache.**
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
# Everything runs in a few minutes; the two models together are about 1 GB to download.
#
# **Before each cell, write down what you expect.** The point is to be surprised in the right places.

# %% [markdown]
# ## 0. Setup
#
# Both models are small, so we keep them in `float32`: losses come out exact, and the T4
# has memory to spare. (A T4 has FP16 tensor cores but no BF16 ones; nothing here needs speed.)

# %%
import math, time, random
import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, GPT2Config, GPT2LMHeadModel, pipeline)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    print("No GPU found: everything still runs, just slower. Runtime → Change runtime type → T4 GPU.")
else:
    print(torch.cuda.get_device_name(0))
torch.manual_seed(0)
random.seed(0)

def banner(s):
    print("\n" + "=" * 76 + f"\n{s}\n" + "=" * 76)

# %% [markdown]
# ## 1. BERT fills the blank
#
# `bert-base-uncased` was pretrained on the masked-word game (lesson 3). Give it a sentence with a
# `[MASK]` and it returns its top guesses for the hidden token, **using both sides of the gap**.
#
# Expect: “paris” on top for the first sentence. For the two “bank” sentences, watch how the
# words *after* the blank change the guesses.

# %%
fill = pipeline("fill-mask", model="bert-base-uncased", device=0 if DEV == "cuda" else -1)

SENTENCES = [
    "The capital of France is [MASK].",
    "The [MASK] was steep and muddy, so we could not reach the water.",
    "The [MASK] was closed on Sunday, so I could not withdraw any money.",
    "She dropped the glass and it [MASK] on the floor.",
]
for s in SENTENCES:
    banner(s)
    for r in fill(s, top_k=5):
        print(f"  {r['token_str']:>12s}   p = {r['score']:.3f}")

# %% [markdown]
# ### Check: count BERT-Base's parameters from its shape
#
# Lesson 3 derived **109,482,240**. Rebuild that number from the config, then compare with the
# real weights. `AutoModel` loads the encoder and its small pooler, without the masked-LM head.

# %%
bert = AutoModel.from_pretrained("bert-base-uncased")
c = bert.config
V, d, L, m, P = c.vocab_size, c.hidden_size, c.num_hidden_layers, c.intermediate_size, c.max_position_embeddings

emb = V * d + P * d + c.type_vocab_size * d + 2 * d          # token, position, segment, LayerNorm
layer = 4 * (d * d + d) + (d * m + m) + (m * d + d) + 4 * d  # attention, MLP, two LayerNorms
pooler = d * d + d
derived = emb + L * layer + pooler
actual = sum(p.numel() for p in bert.parameters())

print(f"vocab {V:,}  hidden {d}  layers {L}  heads {c.num_attention_heads}  MLP {m}  positions {P}")
print(f"embeddings  {emb:>12,}   ({emb / actual:.1%} of the model)")
print(f"one layer   {layer:>12,}")
print(f"derived     {derived:>12,}")
print(f"actual      {actual:>12,}")
assert derived == actual == 109_482_240
print("MATCH")
del bert

# %% [markdown]
# ## 2. The masking recipe, measured
#
# Hugging Face's `DataCollatorForLanguageModeling` implements BERT's rule: choose 15% of tokens,
# then 80% → `[MASK]`, 10% → a random token, 10% unchanged. Rather than trust that, **measure it**
# on a batch, then compute the masked-LM loss only at the chosen positions.
#
# Expect: about 15.0% chosen; of those about 80 / 10 / 10. And roughly 0.15 × 128 ≈ 19 graded
# positions per 128-token sequence.

# %%
btok = AutoTokenizer.from_pretrained("bert-base-uncased")
mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(DEV).eval()

TEXT = (
    "The river had risen overnight, and by morning the path along the bank was under water. "
    "We walked up the hill instead, past the old mill and the bakery that opens at six, "
    "and stopped at the bridge to watch the boats. A man was repairing a fence while his dog "
    "slept in the sun. Nobody seemed to be in a hurry, and for once neither were we. "
) * 40
ids = btok(TEXT, add_special_tokens=False).input_ids
T = 126                                                    # + [CLS] and [SEP] = 128
seqs = [[btok.cls_token_id] + ids[i:i + T] + [btok.sep_token_id] for i in range(0, len(ids) - T, T)][:64]
print(f"{len(seqs)} sequences of {len(seqs[0])} tokens")

collator = DataCollatorForLanguageModeling(tokenizer=btok, mlm=True, mlm_probability=0.15)
batch = collator([{"input_ids": s} for s in seqs])
orig = torch.tensor(seqs)
chosen = batch["labels"] != -100
corrupted = batch["input_ids"]

n_real = (~torch.isin(orig, torch.tensor([btok.cls_token_id, btok.sep_token_id]))).sum().item()
n_chosen = chosen.sum().item()
to_mask = (chosen & (corrupted == btok.mask_token_id)).sum().item()
kept = (chosen & (corrupted == orig)).sum().item()
rand = n_chosen - to_mask - kept

banner("What the collator did")
print(f"real tokens          {n_real:,}")
print(f"chosen for loss      {n_chosen:,}   = {n_chosen / n_real:.1%}   (target 15%)")
print(f"  → [MASK]           {to_mask:,}   = {to_mask / n_chosen:.1%}  (target 80%)")
print(f"  → random token     {rand:,}   = {rand / n_chosen:.1%}  (target 10%)")
print(f"  → unchanged        {kept:,}   = {kept / n_chosen:.1%}  (target 10%)")
print(f"graded per sequence  {n_chosen / len(seqs):.1f}   (0.15 × 126 real tokens = {0.15 * 126:.1f})")

with torch.no_grad():
    out = mlm(input_ids=corrupted.to(DEV), labels=batch["labels"].to(DEV))
print(f"\nmasked-LM loss       {out.loss.item():.3f} nats  → perplexity {math.exp(out.loss.item()):.2f} on the hidden tokens")
print(f"uniform guess        {math.log(btok.vocab_size):.3f} nats  (ln {btok.vocab_size:,})")

# %% [markdown]
# ### Show one corrupted sequence
#
# How to read it: `[MASK](word)` is a masked position and the word it hid; `{x→word}` is a random
# replacement x for the real word; `<word>` was left unchanged but is still graded. The `[MASK]`s
# are the obvious exam questions; the other two are hidden among ordinary tokens.

# %%
row = 0
toks = btok.convert_ids_to_tokens(corrupted[row])
origs = btok.convert_ids_to_tokens(orig[row])
marked = []
for t, o, ch in zip(toks, origs, chosen[row]):
    if not ch:
        marked.append(t)
    elif t == "[MASK]":
        marked.append(f"[MASK]({o})")
    elif t == o:
        marked.append(f"<{t}>")             # unchanged but still predicted
    else:
        marked.append(f"{{{t}→{o}}}")       # random replacement → original
print(" ".join(marked[:70]), "…")

# %% [markdown]
# ## 3. GPT-2 finishes the sentence
#
# Now the other job. Load `gpt2` (the 124M model) and confirm lesson 4's count of **124,439,808**.

# %%
gtok = AutoTokenizer.from_pretrained("gpt2")
gpt = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
gc_ = gpt.config
derived = (gc_.vocab_size * gc_.n_embd + gc_.n_positions * gc_.n_embd
           + gc_.n_layer * (4 * gc_.n_embd ** 2 + 4 * gc_.n_embd + 8 * gc_.n_embd ** 2 + 5 * gc_.n_embd + 4 * gc_.n_embd)
           + 2 * gc_.n_embd)
actual = sum(p.numel() for p in gpt.parameters())
print(f"vocab {gc_.vocab_size:,}  context {gc_.n_positions}  layers {gc_.n_layer}  width {gc_.n_embd}  heads {gc_.n_head}")
print(f"derived {derived:,}   actual {actual:,}")
assert derived == actual == 124_439_808
print("MATCH  (the output head is tied to the token embeddings, so it adds nothing)")

prompt = "The river had risen overnight, and by morning"
x = gtok(prompt, return_tensors="pt").input_ids.to(DEV)
banner("greedy")
print(gtok.decode(gpt.generate(x, max_new_tokens=30, do_sample=False, pad_token_id=gtok.eos_token_id)[0]))
banner("sampled (temperature 0.8, top-p 0.9)")
torch.manual_seed(1)
print(gtok.decode(gpt.generate(x, max_new_tokens=30, do_sample=True, temperature=0.8, top_p=0.9,
                               pad_token_id=gtok.eos_token_id)[0]))

# %% [markdown]
# ### Per-token surprise, perplexity and bits per byte
#
# Lesson 2, on a real model. We put GPT-2's `<|endoftext|>` token in front so that even the first
# word gets predicted (from "start of document"). For every token we print the probability GPT-2
# gave it and its surprise, −ln p.
#
# Expect: function words like “the” and “of” cost under a nat; content words cost several; the
# mean lands somewhere around 3–5 nats for an ordinary sentence. The last line checks our
# arithmetic against Hugging Face's own `labels=` loss.

# %%
def score(text, show=False):
    ids = torch.tensor([[gtok.bos_token_id] + gtok(text).input_ids]).to(DEV)
    with torch.no_grad():
        logits = gpt(ids).logits
    logp = logits[0, :-1].log_softmax(-1)
    tgt = ids[0, 1:]
    nll = -logp.gather(1, tgt[:, None]).squeeze(1)          # surprise of each real token, nats
    if show:
        for t, l in zip(tgt.tolist(), nll.tolist()):
            print(f"  {gtok.decode([t])!r:>14}   p = {math.exp(-l):6.3f}   loss = {l:6.3f} nats  = {l / math.log(2):5.2f} bits")
    mean = nll.mean().item()
    bpb = nll.sum().item() / math.log(2) / len(text.encode("utf-8"))
    return mean, math.exp(mean), bpb, ids

s = "The cat sat on the mat because it was tired."
banner(s)
mean, ppl, bpb, ids = score(s, show=True)
print(f"\nmean loss {mean:.3f} nats/token   perplexity {ppl:.1f}   bits per byte {bpb:.3f}")
with torch.no_grad():
    hf = gpt(ids, labels=ids).loss.item()
print(f"Hugging Face's loss for the same ids: {hf:.3f}")
assert abs(hf - mean) < 1e-4

# %% [markdown]
# ### Perplexity as a lie detector
#
# Same model, four sentences. Before running, rank them by the perplexity you expect.

# %%
TESTS = {
    "natural":        "The river had risen overnight, so we walked up the hill instead.",
    "shuffled words": "hill overnight the instead, risen walked had up so river The we.",
    "true fact":      "The capital of France is Paris.",
    "false fact":     "The capital of France is Rome.",
}
for name, t in TESTS.items():
    mean, ppl, bpb, _ = score(t)
    print(f"{name:>15}   loss {mean:5.2f} nats   perplexity {ppl:9.1f}   bits/byte {bpb:5.2f}   {t}")

# %% [markdown]
# ## 4. An untrained model starts at ln V
#
# Lesson 2 claimed a model that knows nothing gives every token about 1/V, so its loss starts
# near ln 50,257 = 10.82 nats. Build a GPT-2 with **random** weights and check.

# %%
torch.manual_seed(0)
blank = GPT2LMHeadModel(GPT2Config()).to(DEV).eval()
ids = torch.tensor([[gtok.bos_token_id] + gtok(TESTS["natural"]).input_ids]).to(DEV)
with torch.no_grad():
    l0 = blank(ids, labels=ids).loss.item()
print(f"untrained loss {l0:.3f} nats   ln V = {math.log(gtok.vocab_size):.3f}   perplexity {math.exp(l0):,.0f}")
assert abs(l0 - math.log(gtok.vocab_size)) < 1.0
print("The trained GPT-2 got the same sentence down to the loss you printed above.")
del blank

# %% [markdown]
# ## 5. Swap the jobs
#
# ### 5a. GPT-2 cannot see the right-hand side
#
# “The ___ was steep and muddy, so we could not reach the water.” BERT used the words after the
# blank. GPT-2, predicting the token after “The”, sees **only “The”**. Compare its top guesses
# with BERT's from step 1.

# %%
ids = torch.tensor([[gtok.bos_token_id] + gtok("The").input_ids]).to(DEV)
with torch.no_grad():
    p = gpt(ids).logits[0, -1].softmax(-1)
top = p.topk(8)
print("GPT-2's next token after 'The':")
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {gtok.decode([i])!r:>12}  p = {v:.3f}")

# %% [markdown]
# The workaround is to **score whole sentences**: fill the blank with each candidate and ask GPT-2
# for the total log-probability of the complete sentence. Now the right-hand side counts, because
# it is part of what is being scored. The price: **one forward pass per candidate**, where BERT
# needed one pass for its whole vocabulary.

# %%
template = "The {} was steep and muddy, so we could not reach the water."
cands = ["bank", "road", "hill", "river", "money", "door"]
rows = []
for w in cands:
    t = template.format(w)
    mean, _, _, ids = score(t)
    rows.append((mean * (ids.shape[1] - 1), w))         # total nats for the whole sentence
for total, w in sorted(rows):
    print(f"  {w:>6}   total surprise {total:6.2f} nats")
print(f"\n{len(cands)} candidates = {len(cands)} forward passes. BERT ranked all {btok.vocab_size:,} in one.")

# %% [markdown]
# ### 5b. BERT cannot continue a sentence well
#
# Ask BERT to *write*: put six `[MASK]`s after a prompt and fill them left to right, one per pass,
# always taking the top guess. Then let GPT-2 continue the same prompt greedily.
#
# Expect: BERT's continuation is short, repetitive or ungrammatical, and it had to re-read the
# whole sequence on every step. Nothing about its earlier tokens could be reused, because every
# token attends to the new one (lesson 5).

# %%
def bert_write(prefix, n=6):
    body = btok(prefix, add_special_tokens=False).input_ids
    seq = [btok.cls_token_id] + body + [btok.mask_token_id] * n + [btok.convert_tokens_to_ids(".")] + [btok.sep_token_id]
    passes = forwards = 0
    for k in range(n):
        x = torch.tensor([seq]).to(DEV)
        with torch.no_grad():
            logits = mlm(input_ids=x).logits[0]
        pos = 1 + len(body) + k
        seq[pos] = logits[pos].argmax().item()
        passes += 1
        forwards += len(seq)                               # the whole sequence, every time
    return btok.decode(seq[1:-1]), passes, forwards

prefix = "After the storm, the villagers went down to the river and"
text, passes, forwards = bert_write(prefix)
banner("BERT, filling six masks one at a time")
print(text)
print(f"{passes} passes, {forwards} token-forwards")

banner("GPT-2, greedy")
x = gtok(prefix, return_tensors="pt").input_ids.to(DEV)
print(gtok.decode(gpt.generate(x, max_new_tokens=8, do_sample=False, pad_token_id=gtok.eos_token_id)[0]))

# %% [markdown]
# ## 6. Supervision per pass, and the cache
#
# ### 6a. Count graded positions on the same batch
#
# Lesson 5 claimed a causal objective grades every position and a masked one about 15%,
# a ratio near 6.7. Count it on the 64 sequences from step 2.

# %%
causal_targets = sum(len(s) - 1 for s in seqs)               # every next token
masked_targets = n_chosen                                    # only the chosen ones
print(f"causal graded positions  {causal_targets:,}")
print(f"masked graded positions  {masked_targets:,}")
print(f"ratio                    {causal_targets / masked_targets:.2f}   (1 / 0.15 = {1 / 0.15:.2f})")

# %% [markdown]
# ### 6b. The decoder's advantage, timed
#
# GPT-2 can keep the keys and values of earlier tokens because they never change; BERT cannot.
# Time GPT-2 generating 256 tokens with its KV cache on and off. Turning the cache off is what
# an encoder would be forced to do: re-read everything for every new token.
#
# Expect: the cached run is several times faster, and the gap grows with the number of tokens.

# %%
PROMPT = gtok("The river had risen overnight", return_tensors="pt").input_ids.to(DEV)

def gen_time(use_cache, n=256):
    gpt.generate(PROMPT, max_new_tokens=8, use_cache=use_cache, pad_token_id=gtok.eos_token_id)   # warm-up
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpt.generate(PROMPT, max_new_tokens=n, min_new_tokens=n, do_sample=False, use_cache=use_cache,
                 pad_token_id=gtok.eos_token_id)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0

N_NEW = 256
P_LEN = PROMPT.shape[1]
on, off = gen_time(True, N_NEW), gen_time(False, N_NEW)
print(f"with KV cache     {on:6.2f} s")
print(f"without KV cache  {off:6.2f} s   ({off / on:.1f}× slower)")
print(f"token-forwards: cached ≈ {P_LEN + N_NEW:,}, re-reading ≈ {N_NEW * P_LEN + N_NEW * (N_NEW + 1) // 2:,}")

# %% [markdown]
# ## Check yourself
#
# 1. In step 1, which words changed BERT's guess for the “bank” blank? Could GPT-2 have used them?
# 2. Your measured split in step 2: how close to 15% and 80/10/10? Why is it not exact?
# 3. Which sentence in step 3 had the highest perplexity, and why is “false fact” only a little
#    worse than “true fact” overall, even though one word is very surprising?
# 4. How far did training move the loss on the natural sentence, from the untrained model in
#    step 4 to the trained one?
# 5. In 5a, how many forward passes would scoring 1,000 candidate words take with GPT-2? With BERT?
# 6. In 6b, the measured slowdown is far smaller than the ratio of token-forwards. Why?
#    (Hint: Module 05 — at this size each step is dominated by fixed overheads, not arithmetic.)
#
# **Write in your notes:** the masked-LM loss, GPT-2's perplexity on your natural sentence, the
# untrained loss, and the cache speed-up. You will compare them in Module 3B.
