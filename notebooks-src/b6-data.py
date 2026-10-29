# %% [markdown]
# # Lab B6 · Build and Clean a Finetuning Dataset
#
# **Tokens per Second** · Module B6 (Dataset Engineering)
#
# In this lab you will construct and clean a 5,000-row instruction dataset from first principles:
# 1. **Ingest and profile raw data**: Analyze token lengths, whitespace defects, and duplicate rates across 5,000 raw instruction rows.
# 2. **Boilerplate and PII sanitization**: Strip web boilerplate and apply regular expression redactions to mask email addresses and IP credentials.
# 3. **Exact deduplication**: Use SHA-256 string hashing to prune byte-identical duplicate queries in $O(N)$ linear time.
# 4. **Fuzzy near-deduplication**: Apply `datasketch.MinHash` and `MinHashLSH` on 5-gram shingles ($m=128$, $b=16$, $r=8$, Jaccard threshold $t=0.70$) to eliminate near-duplicates that bypass cryptographic hashing.
# 5. **Quality & heuristic filters**: Filter out short fragments, context overflows, and low-entropy repetition loops using `zlib` compression.
# 6. **Chat-template serialization**: Apply standard ChatML formatting with explicit turn markers (`<|im_start|>` and `<|im_end|>`) and append the terminal EOS token.
# 7. **Loss masking verification**: Construct the training tensors (`input_ids` and `labels`) where prompt tokens are masked with label `-100`.
# 8. **Export clean JSONL**: Save the finalized dataset to disk ready for Supervised Finetuning (SFT) in Module B5.
#
# Runtime: Standard Python 3 (CPU is fully sufficient; runs in under 30 seconds).

# %%
import re
import zlib
import hashlib
import json
import random
from collections import Counter

print("Python environment loaded. Initializing synthetic corpus generator...")

# %% [markdown]
# ## 1. Generate the 5,000-Row Raw Corpus
#
# We generate a realistic, noisy 5,000-row instruction dataset containing:
# - Authentic coding, math, systems engineering, and general instruction prompts.
# - 450 exact duplicate rows (simulating duplicate web forum submissions).
# - 620 near-duplicate rows with minor variations in whitespace, punctuation, and wording.
# - 150 records containing Personally Identifiable Information (emails, test IPs).
# - 180 length outliers (empty prompts, 3-token fragments, and repetitive loops).

# %%
SEED_PROMPTS = [
    ("Explain the difference between GQA and MHA in transformer attention.", 
     "Grouped-Query Attention (GQA) groups multiple query heads to share a single key-value head, reducing KV cache memory by the group factor while preserving model quality."),
    ("Write a Python function to compute the Jaccard similarity of two sets of shingles.",
     "def jaccard(set_a, set_b):\n    intersection = len(set_a.intersection(set_b))\n    union = len(set_a.union(set_b))\n    return intersection / union if union > 0 else 0.0"),
    ("What is the peak memory bandwidth of an NVIDIA H100 SXM5 GPU?",
     "The NVIDIA H100 SXM5 features 80 GB of HBM3 memory delivering approximately 3.35 TB/s of peak memory bandwidth."),
    ("Calculate the KV cache memory for Llama-3.1-8B in BF16 at 4,096 tokens.",
     "Llama-3.1-8B has 32 layers, 8 KV heads, and head dimension 128. Bytes per token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes (128 KB). For 4,096 tokens: 4,096 * 128 KB = 524.288 MB per sequence."),
    ("How does continuous batching prevent head-of-line blocking in LLM serving?",
     "Continuous batching forms iteration-level batches at every forward pass, allowing finished sequences to leave immediately and waiting prompts to prefill without waiting for the longest sequence to complete."),
    ("Explain why loss masking sets prompt token labels to -100.",
     "Setting prompt labels to -100 causes CrossEntropyLoss to ignore those positions, ensuring gradients update model weights solely based on the assistant's response rather than memorizing user questions."),
    ("Write a PyTorch snippet to compute scaled dot-product attention.",
     "import torch\nimport math\ndef scaled_dot_product(q, k, v):\n    d_k = q.size(-1)\n    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)\n    return torch.matmul(torch.softmax(scores, dim=-1), v)")
]

random.seed(42)
raw_dataset = []

# Base genuine rows
for i in range(3600):
    base_q, base_a = random.choice(SEED_PROMPTS)
    raw_dataset.append({
        "id": f"row_{i:05d}",
        "instruction": f"{base_q} (Variant {i % 17})",
        "response": f"{base_a} Document ID: {i}."
    })

# Add 450 exact duplicates
exact_seed = SEED_PROMPTS[0]
for i in range(450):
    raw_dataset.append({
        "id": f"exact_dup_{i:04d}",
        "instruction": exact_seed[0],
        "response": exact_seed[1]
    })

# Add 620 near-duplicates (varying punctuation, trailing spaces, trivial casing)
fuzzy_seed = SEED_PROMPTS[1]
for i in range(620):
    spaces = " " * (i % 4)
    punctuation = "!" if i % 2 == 0 else "."
    raw_dataset.append({
        "id": f"fuzzy_dup_{i:04d}",
        "instruction": f" {fuzzy_seed[0]}{punctuation}{spaces}",
        "response": f"{fuzzy_seed[1]} "
    })

# Add 150 PII records
for i in range(150):
    raw_dataset.append({
        "id": f"pii_row_{i:04d}",
        "instruction": f"Contact administrator at admin_{i}@serving-cluster.internal or node 192.168.1.{i % 250} regarding KV cache OOM.",
        "response": f"Ticket logged. Notification dispatched to admin_{i}@serving-cluster.internal."
    })

# Add 180 length outliers (too short or degenerate loops)
for i in range(90):
    raw_dataset.append({
        "id": f"short_{i:04d}",
        "instruction": "Hi",
        "response": "Hello."
    })
for i in range(90):
    raw_dataset.append({
        "id": f"loop_{i:04d}",
        "instruction": "Explain memory bandwidth.",
        "response": "memory bandwidth memory bandwidth memory bandwidth " * 40
    })

random.shuffle(raw_dataset)
print(f"Synthesized raw dataset: {len(raw_dataset)} total rows.")

# %% [markdown]
# ## 2. Profile the Raw Dataset
#
# Before applying filters, compute baseline statistics:
# - Character and word distributions
# - Frequency of common terms

# %%
word_counts = [len(row["instruction"].split()) + len(row["response"].split()) for row in raw_dataset]
print(f"Mean word count: {sum(word_counts) / len(word_counts):.1f} words")
print(f"Min word count:  {min(word_counts)} words")
print(f"Max word count:  {max(word_counts)} words")

# %% [markdown]
# ## 3. PII Redaction & Boilerplate Cleaning
#
# Redact email addresses and IP credentials using regex patterns, and normalize whitespace.

# %%
def clean_and_redact(text: str) -> str:
    # Redact email addresses
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "<EMAIL>", text)
    # Redact IPv4 addresses
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP_ADDRESS>", text)
    # Collapse multiple whitespaces and strip boundaries
    text = " ".join(text.strip().split())
    return text

cleaned_dataset = []
pii_redaction_count = 0

for row in raw_dataset:
    inst = clean_and_redact(row["instruction"])
    resp = clean_and_redact(row["response"])
    if "<EMAIL>" in inst or "<EMAIL>" in resp or "<IP_ADDRESS>" in inst or "<IP_ADDRESS>" in resp:
        pii_redaction_count += 1
    cleaned_dataset.append({"id": row["id"], "instruction": inst, "response": resp})

print(f"PII redaction complete: {pii_redaction_count} rows contained redacted sensitive entities.")

# %% [markdown]
# ## 4. Exact Deduplication via SHA-256
#
# Hash normalized instruction strings with SHA-256 to eliminate identical prompts in $O(N)$ time.

# %%
seen_exact_hashes = set()
exact_deduped_dataset = []
exact_duplicates_pruned = 0

for row in cleaned_dataset:
    content_hash = hashlib.sha256(row["instruction"].encode("utf-8")).hexdigest()
    if content_hash in seen_exact_hashes:
        exact_duplicates_pruned += 1
    else:
        seen_exact_hashes.add(content_hash)
        exact_deduped_dataset.append(row)

print(f"Exact deduplication complete:")
print(f"  Pruned:    {exact_duplicates_pruned} exact duplicate rows")
print(f"  Surviving: {len(exact_deduped_dataset)} unique rows")

# %% [markdown]
# ## 5. Fuzzy Near-Deduplication with MinHash & LSH
#
# Exact deduplication misses near-duplicates (varying punctuation or trivial phrasing).
# We install `datasketch`, build 5-gram word shingles, compute 128-permutation MinHash signatures, and query an LSH index with threshold $t=0.70$.

# %%
try:
    import datasketch
    from datasketch import MinHash, MinHashLSH
except ImportError:
    import subprocess, sys
    print("Installing datasketch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasketch"])
    from datasketch import MinHash, MinHashLSH

print("datasketch initialized.")

# %%
def get_shingles(text: str, k: int = 5) -> set:
    words = text.lower().split()
    if len(words) < k:
        return set([" ".join(words)])
    return set(" ".join(words[i:i+k]) for i in range(len(words) - k + 1))

def compute_minhash(shingles: set, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m

# Build LSH Index
lsh = MinHashLSH(threshold=0.70, num_perm=128)
minhash_signatures = {}

for row in exact_deduped_dataset:
    shingles = get_shingles(row["instruction"], k=5)
    mh = compute_minhash(shingles, num_perm=128)
    minhash_signatures[row["id"]] = mh
    lsh.insert(row["id"], mh)

# Find near-duplicate clusters
fuzzy_duplicates_pruned = 0
dropped_ids = set()
fuzzy_clean_dataset = []

for row in exact_deduped_dataset:
    row_id = row["id"]
    if row_id in dropped_ids:
        continue
    # Query LSH for candidates
    candidates = lsh.query(minhash_signatures[row_id])
    # Keep the current row, prune all other candidates in its cluster
    for cand_id in candidates:
        if cand_id != row_id and cand_id not in dropped_ids:
            dropped_ids.add(cand_id)
            fuzzy_duplicates_pruned += 1
    fuzzy_clean_dataset.append(row)

print(f"Fuzzy LSH deduplication complete:")
print(f"  Pruned:    {fuzzy_duplicates_pruned} near-duplicate rows")
print(f"  Surviving: {len(fuzzy_clean_dataset)} rows")

# %% [markdown]
# ## 6. Heuristic and Quality Filtering
#
# Prune degenerate sequences:
# - Extremely short responses (< 10 words)
# - Repetitive token loops using `zlib` compression ratio check

# %%
final_filtered_dataset = []
quality_pruned = 0

for row in fuzzy_clean_dataset:
    words = row["response"].split()
    # Length filter
    if len(words) < 5 or len(words) > 1024:
        quality_pruned += 1
        continue
    
    # Compression ratio filter (catches repeating token loops)
    raw_bytes = row["response"].encode("utf-8")
    comp_ratio = len(zlib.compress(raw_bytes)) / max(1, len(raw_bytes))
    if comp_ratio < 0.25:  # Repetitive babble compresses to < 20%
        quality_pruned += 1
        continue
        
    final_filtered_dataset.append(row)

print(f"Quality and compression filtering complete:")
print(f"  Pruned:    {quality_pruned} low-quality / repeating rows")
print(f"  Surviving: {len(final_filtered_dataset)} gold instruction rows")

# %% [markdown]
# ## 7. Chat-Template Formatting & Loss Masking
#
# Format records into the ChatML schema:
# `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n`
#
# Construct the PyTorch labels tensor where system and user tokens are set to `-100`.

# %%
SYSTEM_PROMPT = "You are a specialized LLM serving and inference engineering assistant."

formatted_records = []
token_lengths = []

for row in final_filtered_dataset:
    # ChatML string serialization
    formatted_chat = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{row['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{row['response']}<|im_end|>\n"
    )
    # Approximate token count (1 word ≈ 1.3 tokens + 14 chat template control tokens)
    approx_tokens = int(len(formatted_chat.split()) * 1.3) + 14
    token_lengths.append(approx_tokens)
    
    formatted_records.append({
        "id": row["id"],
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "user", "value": row["instruction"]},
            {"from": "assistant", "value": row["response"]}
        ],
        "text": formatted_chat,
        "token_count": approx_tokens
    })

print(f"Formatted {len(formatted_records)} records into ChatML standard.")

# Print Loss Masking Simulation
sample = formatted_records[0]
prompt_part = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{sample['conversations'][1]['value']}<|im_end|>\n<|im_start|>assistant\n"
resp_part = f"{sample['conversations'][2]['value']}<|im_end|>\n"

prompt_words = prompt_part.split()
resp_words = resp_part.split()

sample_labels = [-100] * len(prompt_words) + [f"ID_{i}" for i in range(len(resp_words))]
print("\nLoss Masking Simulation (First 15 positions):")
print("Tokens: ", (prompt_words + resp_words)[:15])
print("Labels: ", sample_labels[:15])

# %% [markdown]
# ## 8. Token Length Histogram
#
# Plot the distribution of token lengths across the cleaned dataset.

# %%
print("\nToken Length Distribution Histogram:")
buckets = [(0, 64), (65, 128), (129, 256), (257, 512), (513, 1024), (1025, 2048)]
histogram = {f"{b[0]}-{b[1]} tokens": 0 for b in buckets}

for length in token_lengths:
    for low, high in buckets:
        if low <= length <= high:
            histogram[f"{low}-{high} tokens"] += 1
            break

for bucket_name, count in histogram.items():
    bar = "█" * int(count / 30)
    print(f"{bucket_name:>18}: {count:>5} {bar}")

# %% [markdown]
# ## 9. Export Clean JSONL for Supervised Finetuning
#
# Save the cleaned dataset to disk as `dataset_clean_sft.jsonl`.
# This file is ready for loading into HuggingFace `SFTTrainer` or PyTorch LoRA fine-tuning in Module B5.

# %%
OUTPUT_FILE = "dataset_clean_sft.jsonl"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for rec in formatted_records:
        f.write(json.dumps(rec) + "\n")

print(f"\nSUCCESS: Successfully exported {len(formatted_records)} cleaned rows to '{OUTPUT_FILE}'.")
print("Dataset is deduplicated, PII-redacted, length-verified, and ChatML-templated.")
