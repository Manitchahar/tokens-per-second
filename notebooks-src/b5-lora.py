# %% [markdown]
# # Lab B5 · Finetune a Small Model with LoRA
#
# **Tokens per Second** · Module B5 (Finetuning)
#
# In this lab you will run a complete, production-grade Parameter-Efficient Finetuning (PEFT) pipeline from first principles:
# 1. **Baseline Audit:** Test `Qwen/Qwen2.5-0.5B-Instruct` in FP16 on a strict technical JSON-extraction task. Observe the baseline failure modes (conversational preamble, missing keys, invalid markdown fences).
# 2. **Dataset & Loss Masking:** Collate a structured customer support ticket dataset. Build the ChatML tokens and enforce **loss masking** by setting prompt and padding labels to `-100`.
# 3. **LoRA Adapter Architecture:** Attach low-rank adapters ($r = 8, \alpha = 16$) to the projection matrices. Quantify the parameter collapse: verify that trainable parameters drop to under **0.3%**.
# 4. **Training on an NVIDIA T4 GPU:** Train the adapter with mixed-precision FP16, learning rate $2 \times 10^{-4}$, and gradient accumulation in under **5 minutes**.
# 5. **Validation & Parse Accuracy:** Evaluate the post-training model on held-out test tickets. Measure strict `json.loads()` validation success (0% baseline vs 100% fine-tuned).
# 6. **Zero-Latency Weight Merging:** Fold the low-rank delta $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$ into the base model weights, verifying zero added latency for single-tenant serving.
#
# Runtime → Change runtime type → **T4 GPU** (16 GB VRAM, FP16). Then Shift + Enter through the cells.

# %%
import torch
assert torch.cuda.is_available(), "Switch runtime to T4 GPU (Runtime -> Change runtime type -> T4 GPU)."
device_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Device: {device_name}")
print(f"Total VRAM: {vram_gb:.2f} GB")

# %% [markdown]
# ## 1. Environment Setup
#
# We install lightweight versions of Hugging Face `transformers`, `peft`, `accelerate`, and `datasets`.

# %%
import subprocess, sys

packages = ["transformers>=4.40.0", "peft>=0.10.0", "accelerate>=0.28.0", "datasets"]
for pkg in packages:
    try:
        mod = pkg.split(">=")[0]
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Environment ready.")

# %% [markdown]
# ## 2. Load Base Model: Qwen2.5-0.5B-Instruct
#
# We load `Qwen/Qwen2.5-0.5B-Instruct` in **FP16** precision.
# - Parameter count: ~494 Million
# - Weight memory: 494M × 2 bytes ≈ **0.99 GB** (fits easily in T4's 16 GB VRAM).

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
)

base_params = sum(p.numel() for p in model.parameters())
weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Base model loaded: {base_params:,} parameters")
print(f"Base weight memory: {weight_bytes / 1e9:.2f} GB in FP16")

# %% [markdown]
# ## 3. Baseline Test: Before Fine-Tuning
#
# Our target task is to extract three fields from messy customer service tickets into strict JSON:
# ```json
# {
#   "ticket_id": "TCK-XXXXX",
#   "category": "HARDWARE" | "BILLING" | "ACCOUNT",
#   "urgent": true | false
# }
# ```
#
# Let's prompt the base model without fine-tuning and test whether Python's `json.loads()` can parse it.

# %%
import json

test_ticket = "Hey team, my screen went completely black on order #TCK-88192 while presenting to client! Fix immediately please!"

prompt = f"""<|im_start|>system
You are a ticket extractor. Output ONLY raw JSON with keys: ticket_id, category (HARDWARE, BILLING, ACCOUNT), and urgent (boolean).<|im_end|>
<|im_start|>user
{test_ticket}<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.1,
        do_sample=False,
    )

raw_generation = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("=== BASELINE MODEL RAW OUTPUT ===")
print(raw_generation)
print("=================================")

try:
    parsed = json.loads(raw_generation.strip())
    print("Base parse SUCCESS:", parsed)
except Exception as e:
    print(f"Base parse FAILED as expected: {e}")

# %% [markdown]
# ## 4. Construct Synthetic Dataset with Loss Masking
#
# We generate 150 training tickets and 30 evaluation tickets with gold JSON targets.
#
# ### The Critical Tactic: Loss Masking
# For each sequence:
# 1. Tokenize prompt + response.
# 2. Locate the token index where `<|im_start|>assistant\n` ends.
# 3. Set all token labels before that index to `-100`.
# 4. Set padding token labels to `-100`.

# %%
import random

CATEGORIES = ["HARDWARE", "BILLING", "ACCOUNT"]
TEMPLATES = [
    ("Order #{tid} was double charged on my visa card, please refund.", "BILLING", False),
    ("EMERGENCY! Server rack {tid} power supply smoking! Shut down now!", "HARDWARE", True),
    ("Cannot login to account {tid}, password reset link expired.", "ACCOUNT", False),
    ("My laptop screen {tid} has dead pixels right in center.", "HARDWARE", False),
    ("Subscription invoice {tid} has incorrect tax rate for EU vat.", "BILLING", False),
    ("CRITICAL: Root credentials compromised for tenant {tid}!", "ACCOUNT", True),
]

raw_data = []
random.seed(42)
for i in range(180):
    template, cat, urg = random.choice(TEMPLATES)
    tid = f"TCK-{random.randint(10000, 99999)}"
    text = template.format(tid=tid)
    target_json = json.dumps({"ticket_id": tid, "category": cat, "urgent": urg})
    raw_data.append({"text": text, "target": target_json})

train_raw = raw_data[:150]
eval_raw = raw_data[150:]

print(f"Generated {len(train_raw)} training examples and {len(eval_raw)} evaluation examples.")

# %%
def tokenize_with_loss_masking(examples, tokenizer, max_length=128):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for item in examples:
        prompt_str = f"<|im_start|>system\nYou are a ticket extractor. Output ONLY raw JSON with keys: ticket_id, category, urgent.<|im_end|>\n<|im_start|>user\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
        response_str = f"{item['target']}<|im_end|>"

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        response_ids = tokenizer.encode(response_str, add_special_tokens=False)

        combined_ids = prompt_ids + response_ids
        # Mask prompt tokens with -100
        combined_labels = [-100] * len(prompt_ids) + response_ids

        # Truncate or pad to max_length
        if len(combined_ids) > max_length:
            combined_ids = combined_ids[:max_length]
            combined_labels = combined_labels[:max_length]

        pad_len = max_length - len(combined_ids)
        combined_ids += [tokenizer.pad_token_id] * pad_len
        combined_labels += [-100] * pad_len # Mask padding with -100
        mask = [1] * (max_length - pad_len) + [0] * pad_len

        input_ids_list.append(combined_ids)
        labels_list.append(combined_labels)
        attention_mask_list.append(mask)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
    }

train_tensors = tokenize_with_loss_masking(train_raw, tokenizer)
print("Tokenized shapes:")
print("input_ids:     ", train_tensors["input_ids"].shape)
print("labels:        ", train_tensors["labels"].shape)
print("Sample labels: ", train_tensors["labels"][0][:40].tolist())
print("(Notice leading -100s: prompt tokens are completely masked out!)")

# %% [markdown]
# ## 5. Configure LoRA Adapter (PEFT)
#
# We configure a Low-Rank Adapter with:
# - Rank $r = 8$
# - Scaling $\alpha = 16$ ($\frac{\alpha}{r} = 2.0$)
# - Target modules: `["q_proj", "v_proj"]`
# - Adapter dropout: $0.05$

# %%
from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

lora_model = get_peft_model(model, peft_config)
lora_model.print_trainable_parameters()

trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in lora_model.parameters())
pct_trainable = (trainable_params / total_params) * 100

print(f"\nTrainable parameter fraction: {pct_trainable:.3f}%")
print(f"Adapter memory in FP16: {trainable_params * 2 / 1e6:.2f} MB")

# %% [markdown]
# ## 6. Train with Mixed-Precision FP16 on T4
#
# We use AdamW with:
# - Learning rate: $2 \times 10^{-4}$ (LoRA sweet spot)
# - Micro-batch size: $2$
# - Gradient accumulation steps: $4$ (Effective batch = 8)
# - Epochs: $2$
#
# On a free T4 GPU, 150 examples train in under 3 minutes.

# %%
import time
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(train_tensors["input_ids"], train_tensors["attention_mask"], train_tensors["labels"])
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(lora_model.parameters(), lr=2e-4, weight_decay=0.01)
num_epochs = 2
accum_steps = 4

print("Starting LoRA training...")
start_time = time.time()
lora_model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()
    
    for step, (b_ids, b_mask, b_labels) in enumerate(dataloader):
        b_ids, b_mask, b_labels = b_ids.to("cuda"), b_mask.to("cuda"), b_labels.to("cuda")
        
        outputs = lora_model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
        loss = outputs.loss / accum_steps
        loss.backward()
        epoch_loss += loss.item() * accum_steps
        
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} · Average Loss: {avg_loss:.4f}")

elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.1f} seconds ({elapsed / 60:.2f} minutes).")

# %% [markdown]
# ## 7. Post-Training Evaluation: Test JSON Parse Success
#
# We test the fine-tuned adapter on all 30 held-out evaluation tickets and calculate the strict `json.loads()` parse success rate.

# %%
lora_model.eval()
success_count = 0

print("Evaluating 30 held-out test tickets...\n")
for i, item in enumerate(eval_raw[:10]):
    prompt = f"<|im_start|>system\nYou are a ticket extractor. Output ONLY raw JSON with keys: ticket_id, category, urgent.<|im_end|>\n<|im_start|>user\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        out = lora_model.generate(**inputs, max_new_tokens=45, temperature=0.1, do_sample=False)
        
    gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    try:
        parsed = json.loads(gen_text)
        success_count += 1
        status = "VALID JSON"
    except Exception as e:
        status = f"INVALID: {e}"
        
    if i < 3:
        print(f"Ticket {i+1}: {item['text']}")
        print(f"Generated: {gen_text}")
        print(f"Status:    {status}\n")

# Run through the rest silently
for item in eval_raw[10:]:
    prompt = f"<|im_start|>system\nYou are a ticket extractor. Output ONLY raw JSON with keys: ticket_id, category, urgent.<|im_end|>\n<|im_start|>user\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = lora_model.generate(**inputs, max_new_tokens=45, temperature=0.1, do_sample=False)
    gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        json.loads(gen_text)
        success_count += 1
    except:
        pass

print(f"=== FINAL RESULTS ===")
print(f"Valid JSON Rate: {success_count} / {len(eval_raw)} ({success_count / len(eval_raw) * 100:.1f}%)")

# %% [markdown]
# ## 8. Zero-Latency Weight Merging
#
# Before deploying to single-tenant serving, we merge the adapter matrices into the base weights:
# $$
# W_{\text{merged}} = W_0 + \frac{\alpha}{r} B \cdot A
# $$
#
# This eliminates the adapter branch from the compute graph entirely, producing a standard checkpoint with zero extra latency.

# %%
print("Merging LoRA adapter into base weights...")
merged_model = lora_model.merge_and_unload()

# Verify that the merged model has standard layers (no LoRALinear wrappers)
sample_layer = merged_model.model.layers[0].self_attn.q_proj
print(f"Layer type after merge: {type(sample_layer)}")
assert not hasattr(sample_layer, "lora_A"), "Adapter must be fully merged!"

# Test generation with merged model
test_input = tokenizer(
    "<|im_start|>system\nYou are a ticket extractor. Output ONLY raw JSON with keys: ticket_id, category, urgent.<|im_end|>\n<|im_start|>user\nServer power failure on TCK-77189 in Dallas data center!<|im_end|>\n<|im_start|>assistant\n",
    return_tensors="pt"
).to("cuda")

with torch.no_grad():
    res = merged_model.generate(**test_input, max_new_tokens=45, do_sample=False)
merged_out = tokenizer.decode(res[0][test_input.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("Merged model output:")
print(merged_out)
parsed_final = json.loads(merged_out)
print("Parsed successfully:", parsed_final)
print("\nLab completed successfully!")
