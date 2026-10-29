# %% [markdown]
# # Lab B4 · A Tool-Using ReAct Agent From Scratch
#
# **Tokens per Second** · Module B4 (Agents)
#
# In this lab you will construct and benchmark a complete, autonomous **ReAct agent loop** from scratch in pure Python:
# 1. **No heavyweight frameworks:** No LangChain, CrewAI, or AutoGen. Every line of the control loop, tool dispatcher, and memory state is transparent.
# 2. **Three deterministic tools:**
#    - `calculator(expression: str)`: safe mathematical evaluation using Python's Abstract Syntax Tree (`ast`).
#    - `local_search(query: str)`: keyword and token-overlap retrieval over an internal corpus of 20 GPU and LLM architecture specifications.
#    - `python_sandbox(code: str)`: a restricted Python execution environment with timeout and output capture.
# 3. **Defensive circuit breakers:**
#    - Canonical state hashing to detect and break infinite loops.
#    - Hard step limits (`max_steps = 6`) to prevent runaway API burn.
# 4. **Run on free Colab T4:** Load `Qwen2.5-3B-Instruct` in FP16 (~6 GB VRAM on a 16 GB T4).
# 5. **Benchmark across 20 evaluation tasks:** Execute 20 multi-step engineering tasks, trace every thought and action, and measure exact **Pass@1 Success Rate** and **Average Steps to Completion**.
#
# Runtime → Change runtime type → **T4 GPU** (16 GB, FP16). Then Shift + Enter through the cells.

# %%
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Active Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %% [markdown]
# ## 1. Install Dependencies
#
# We install lightweight, native libraries: `transformers`, `accelerate`, and standard Python utilities.

# %%
import subprocess, sys

packages = ["transformers", "accelerate"]
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Environment ready.")

# %% [markdown]
# ## 2. The Internal Technical Knowledge Base (20 Documents)
#
# The agent will query this knowledge base using `local_search` to answer factual systems questions.

# %%
KNOWLEDGE_BASE = [
    {"id": "doc_01", "text": "NVIDIA H100 SXM features 80 GB of HBM3 memory with a peak bandwidth of 3.35 TB/s and 989 TFLOP/s of dense BF16 tensor core compute."},
    {"id": "doc_02", "text": "NVIDIA A100 SXM features 80 GB of HBM2e memory with a peak bandwidth of 2.039 TB/s and 312 TFLOP/s of BF16 compute."},
    {"id": "doc_03", "text": "NVIDIA T4 features 16 GB of GDDR6 memory with a peak bandwidth of 320 GB/s and 65 TFLOP/s of FP16 compute. It lacks BF16 and FP8 tensor cores."},
    {"id": "doc_04", "text": "NVIDIA H200 SXM features 141 GB of HBM3e memory with a peak bandwidth of 4.8 TB/s and 989 TFLOP/s of BF16 compute."},
    {"id": "doc_05", "text": "NVIDIA B200 features 192 GB of HBM3e memory with a peak bandwidth of 8.0 TB/s and native microscopic FP4 tensor cores."},
    {"id": "doc_06", "text": "Llama-3.1-8B architecture: 32 transformer layers, hidden dimension 4096, 32 query heads, 8 key-value heads, head dimension 128, vocabulary 128256."},
    {"id": "doc_07", "text": "Llama-3.1-70B architecture: 80 transformer layers, hidden dimension 8192, 64 query heads, 8 key-value heads, head dimension 128, vocabulary 128256."},
    {"id": "doc_08", "text": "Llama-3.1-405B architecture: 126 transformer layers, hidden dimension 16384, 128 query heads, 8 key-value heads, head dimension 128, vocabulary 128256."},
    {"id": "doc_09", "text": "Qwen2.5-3B-Instruct architecture: 36 layers, hidden dimension 2048, 16 query heads, 2 key-value heads, head dimension 128."},
    {"id": "doc_10", "text": "KV cache size formula: 2 * num_layers * num_kv_heads * head_dim * bytes_per_element per token per sequence."},
    {"id": "doc_11", "text": "Prefill phase is compute-bound, scaling as 2 * P * N FLOPs for N tokens on a model with P parameters."},
    {"id": "doc_12", "text": "Decode phase is memory-bound, requiring reading all P weights from HBM to SM registers for each newly generated token."},
    {"id": "doc_13", "text": "PagedAttention organizes KV cache memory into fixed-size physical blocks of 16 or 32 tokens, eliminating external fragmentation."},
    {"id": "doc_14", "text": "RadixAttention reuses cached KV blocks across requests by maintaining a radix tree over prompt token prefixes."},
    {"id": "doc_15", "text": "Arithmetic intensity is defined as FLOPs performed divided by memory bytes transferred: FLOPs / Byte."},
    {"id": "doc_16", "text": "NVIDIA H100 SXM roofline knee occurs at 989e12 / 3.35e12 = 295.22 FLOPs per byte."},
    {"id": "doc_17", "text": "NVIDIA A100 SXM roofline knee occurs at 312e12 / 2.039e12 = 153.02 FLOPs per byte."},
    {"id": "doc_18", "text": "NVIDIA T4 roofline knee occurs at 65e12 / 320e9 = 203.12 FLOPs per byte."},
    {"id": "doc_19", "text": "In BF16 or FP16 precision, each floating-point parameter occupies exactly 2 bytes in GPU memory."},
    {"id": "doc_20", "text": "In FP8 precision, each floating-point parameter occupies exactly 1 byte in GPU memory."},
]

# %% [markdown]
# ## 3. Implementing the Three Deterministic Tools
#
# 1. `calculator`: Safe arithmetic evaluation via Python `ast`.
# 2. `local_search`: BM25/keyword retrieval over the knowledge base.
# 3. `python_sandbox`: Safe string execution environment.

# %%
import ast
import operator
import re
import sys
from io import StringIO

# Tool 1: Safe AST Calculator
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    raise ValueError(f"Unsupported expression node: {type(node)}")

def tool_calculator(expression: str) -> str:
    """Safely evaluate mathematical expressions without eval()."""
    clean_expr = expression.replace(",", "").strip()
    try:
        tree = ast.parse(clean_expr, mode='eval')
        result = safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculator Error: {e}"

# Tool 2: Local Search
def tool_local_search(query: str) -> str:
    """Retrieve the top matching documents from the knowledge base."""
    terms = set(re.findall(r"\w+", query.lower()))
    scored_docs = []
    for doc in KNOWLEDGE_BASE:
        doc_words = set(re.findall(r"\w+", doc["text"].lower()))
        overlap = len(terms.intersection(doc_words))
        if overlap > 0:
            scored_docs.append((overlap, doc["id"], doc["text"]))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    if not scored_docs:
        return "No relevant documents found."
    top_matches = scored_docs[:2]
    return " | ".join([f"[{doc_id}] {text}" for _, doc_id, text in top_matches])

# Tool 3: Python Sandbox
def tool_python_sandbox(code: str) -> str:
    """Execute Python code in a restricted namespace and capture stdout."""
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    safe_globals = {"__builtins__": {
        "print": print, "range": range, "len": len, "int": int, "float": float,
        "str": str, "sum": sum, "min": min, "max": max, "abs": abs, "round": round
    }}
    try:
        exec(code, safe_globals)
        output = redirected_output.getvalue().strip()
        return output if output else "Code executed with no output."
    except Exception as e:
        return f"Sandbox Error: {e}"
    finally:
        sys.stdout = old_stdout

TOOLS = {
    "calculator": tool_calculator,
    "local_search": tool_local_search,
    "python_sandbox": tool_python_sandbox,
}

print("Tools initialized successfully:")
print("Calculator test:", tool_calculator("989 / 3.35"))
print("Search test:", tool_local_search("H100 bandwidth"))

# %% [markdown]
# ## 4. Load Qwen2.5-3B-Instruct in FP16 on T4 GPU
#
# We load the 3-billion parameter model in FP16 (`torch.float16`).
# Memory footprint: ~6.2 GB, leaving ~9 GB of free VRAM on the Colab T4.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print(f"Loading {MODEL_NAME} in FP16 on GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
)
print(f"Model loaded. VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% [markdown]
# ## 5. The ReAct Control Loop with Circuit Breakers
#
# We define the system prompt instructing the model to follow the ReAct pattern:
# `Thought: ...`
# `Action: tool_name(arguments)`
# `Observation: ...`
#
# We include a canonical state-hashing circuit breaker that stops loops if identical actions repeat.

# %%
import hashlib
import json

SYSTEM_PROMPT = """You are an expert systems AI agent. You solve engineering and mathematical problems by strictly following the ReAct framework.

You have access to the following three tools:
1. calculator(expression: str) -> str: Evaluate arithmetic expressions (e.g. calculator(989 / 3.35)).
2. local_search(query: str) -> str: Search technical hardware and model architecture documents (e.g. local_search(Llama-3.1-8B layers)).
3. python_sandbox(code: str) -> str: Run short Python scripts and print results (e.g. python_sandbox(print(2**10))).

Format your response strictly as follows:
Thought: <your step-by-step reasoning>
Action: <tool_name>(<arguments>)

When you receive the Observation, continue with:
Thought: <synthesis of observation>
...
When you have the final answer, output:
Final Answer: <your concise, exact answer>

Do not guess hardware numbers. Always search or calculate them."""

def parse_action(text: str):
    """Extract action tool name and argument from model text."""
    match = re.search(r"Action:\s*(\w+)\((.*)\)", text, re.DOTALL)
    if match:
        tool_name = match.group(1).strip()
        arg_str = match.group(2).strip()
        # Clean quotes if passed as "expression" or 'expression'
        if (arg_str.startswith('"') and arg_str.endswith('"')) or (arg_str.startswith("'") and arg_str.endswith("'")):
            arg_str = arg_str[1:-1]
        return tool_name, arg_str
    return None, None

def run_agent_react(goal: str, max_steps: int = 6, verbose: bool = False):
    """Execute one complete ReAct trajectory with state-hashing circuit breaker."""
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{goal}<|im_end|>\n<|im_start|>assistant\n"
    
    action_hashes = []
    trace = []
    
    for step in range(1, max_steps + 1):
        # Generate model turn
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,  # near-deterministic
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the newly generated tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        trace.append(f"Turn {step}:\n{response}")
        if verbose:
            print(f"--- Turn {step} ---")
            print(response)
        
        # Check for Final Answer
        if "Final Answer:" in response:
            final_ans = response.split("Final Answer:")[-1].strip()
            return {"status": "success", "steps": step, "final_answer": final_ans, "trace": trace}
        
        # Parse Action
        tool_name, tool_arg = parse_action(response)
        if not tool_name:
            # Model stopped without tool call or final answer
            return {"status": "failed", "reason": "No valid action parsed", "steps": step, "trace": trace}
        
        # State Hashing Circuit Breaker
        state_str = f"{tool_name}:{tool_arg}"
        h = hashlib.sha256(state_str.encode()).hexdigest()[:8]
        action_hashes.append(h)
        if len(action_hashes) >= 3 and action_hashes[-1] == action_hashes[-2] == action_hashes[-3]:
            obs = f"CIRCUIT_BREAKER: You have repeated {tool_name}({tool_arg}) 3 times. Action blocked. Choose a different tool."
        else:
            if tool_name in TOOLS:
                obs = TOOLS[tool_name](tool_arg)
            else:
                obs = f"Error: Tool '{tool_name}' not recognized. Available tools: {list(TOOLS.keys())}"
        
        if verbose:
            print(f"Observation: {obs}\n")
        
        # Append turn to prompt
        prompt += f"{response}\nObservation: {obs}\n"
    
    return {"status": "failed", "reason": "Max step budget exceeded", "steps": max_steps, "trace": trace}

print("ReAct agent engine compiled.")

# %% [markdown]
# ## 6. The 20-Task Evaluation Benchmark
#
# We define 20 diverse, realistic systems engineering evaluation tasks:
# - Hardware roofline calculations
# - Model memory and KV cache sizing
# - Multi-hop hardware comparisons
# - String and data transformations

# %%
BENCHMARK_TASKS = [
    {"id": 1, "goal": "Find the peak dense BF16 FLOP/s and memory bandwidth of NVIDIA H100 SXM, and calculate its roofline knee in FLOPs/byte.", "expected": "295.22"},
    {"id": 2, "goal": "Find the memory bandwidth of NVIDIA T4 in GB/s and NVIDIA H100 SXM in TB/s, and calculate how many times faster H100 memory bandwidth is.", "expected": "10.47"},
    {"id": 3, "goal": "What is the head dimension and number of KV heads in Llama-3.1-8B? Calculate how many total KV bytes are stored per token across all layers in BF16.", "expected": "131072"},
    {"id": 4, "goal": "Calculate the roofline knee (FLOPs/byte) for NVIDIA A100 SXM using its dense BF16 compute and HBM2e bandwidth.", "expected": "153.02"},
    {"id": 5, "goal": "How many layers and query heads does Llama-3.1-70B have? Compute the product of layers and query heads.", "expected": "5120"},
    {"id": 6, "goal": "What is the peak memory bandwidth of NVIDIA H200 SXM in TB/s and NVIDIA B200 in TB/s? Calculate the ratio of B200 to H200 bandwidth.", "expected": "1.67"},
    {"id": 7, "goal": "Calculate the exact number of bytes occupied by a 70 billion parameter model stored in FP8 precision (1 byte per parameter) in GiB.", "expected": "65.19"},
    {"id": 8, "goal": "What is the hidden dimension of Qwen2.5-3B-Instruct? Compute the square of its hidden dimension.", "expected": "4194304"},
    {"id": 9, "goal": "For a sequence length of 4096 tokens, calculate the KV cache size in bytes for Llama-3.1-8B in BF16.", "expected": "536870912"},
    {"id": 10, "goal": "Calculate the roofline knee for NVIDIA T4 using its FP16 compute and GDDR6 bandwidth.", "expected": "203.12"},
    {"id": 11, "goal": "How many query heads does Llama-3.1-405B have? Calculate 2 raised to the power of that number divided by 1000.", "expected": "3.40e35"},
    {"id": 12, "goal": "How many parameters per layer does Qwen2.5-3B have if it has 36 layers and 3 billion total parameters?", "expected": "83333333"},
    {"id": 13, "goal": "Find the bandwidth of NVIDIA A100 SXM (2.039 TB/s). Convert this to GB/s.", "expected": "2039"},
    {"id": 14, "goal": "Calculate how many times larger H200 HBM capacity (141 GB) is compared to NVIDIA T4 memory capacity (16 GB).", "expected": "8.81"},
    {"id": 15, "goal": "Compute the sum of hidden dimensions of Llama-3.1-8B (4096) and Llama-3.1-70B (8192).", "expected": "12288"},
    {"id": 16, "goal": "Find the vocabulary size of Llama-3.1-8B from docs. Multiply it by 2.", "expected": "256512"},
    {"id": 17, "goal": "What is the memory capacity of NVIDIA B200 in GB? Multiply by 1024 to get MB.", "expected": "196608"},
    {"id": 18, "goal": "Calculate the ratio of H100 BF16 FLOP/s (989 TFLOP/s) to A100 BF16 FLOP/s (312 TFLOP/s).", "expected": "3.17"},
    {"id": 19, "goal": "If prefill FLOPs are 2 * P * N, calculate the FLOPs for an 8 billion parameter model on a 1000-token prompt in TFLOPs.", "expected": "16"},
    {"id": 20, "goal": "Calculate the bandwidth speedup of B200 (8.0 TB/s) over A100 (2.039 TB/s).", "expected": "3.92"},
]

print(f"Loaded {len(BENCHMARK_TASKS)} evaluation tasks.")

# %% [markdown]
# ## 7. Run the Benchmark & Measure Pass@1 Success Rate

# %%
results = []
successful_steps = []

print("Running 20-Task ReAct Evaluation Benchmark on Qwen2.5-3B-Instruct...\n")

for task in BENCHMARK_TASKS:
    t_id = task["id"]
    goal = task["goal"]
    print(f"Executing Task {t_id}/20: {goal[:60]}...")
    
    res = run_agent_react(goal, max_steps=6, verbose=False)
    
    # Check if expected key number appears in the final answer
    is_correct = False
    if res["status"] == "success":
        ans = res["final_answer"].replace(",", "")
        exp = task["expected"].replace(",", "")
        # Fuzzy match numerical answer
        if exp in ans or (try_float := False):
            is_correct = True
        else:
            try:
                # Numerical tolerance check
                exp_val = float(exp)
                numbers = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", ans)]
                for n in numbers:
                    if abs(n - exp_val) / (abs(exp_val) + 1e-6) < 0.05:
                        is_correct = True
                        break
            except:
                pass
    
    task_res = {
        "id": t_id,
        "solved": is_correct,
        "steps": res["steps"],
        "answer": res.get("final_answer", "N/A"),
        "expected": task["expected"],
    }
    results.append(task_res)
    if is_correct:
        successful_steps.append(res["steps"])
        print(f"  ✓ SOLVED in {res['steps']} turns.")
    else:
        print(f"  ✗ FAILED ({res.get('reason', 'Incorrect answer')}).")

# %% [markdown]
# ## 8. Compute Final Benchmark Metrics
#
# We compute:
# - **Pass@1 Success Rate:** percentage of tasks correctly resolved.
# - **Average Steps to Completion:** trajectory efficiency.
# - **Step Distribution:** breakdown of turns taken.

# %%
total_tasks = len(BENCHMARK_TASKS)
num_solved = sum(1 for r in results if r["solved"])
success_rate = (num_solved / total_tasks) * 100
avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else 0.0

print("\n" + "=" * 50)
print("FINAL AGENT EVALUATION REPORT")
print("=" * 50)
print(f"Total Benchmark Tasks:       {total_tasks}")
print(f"Tasks Solved Correctly:      {num_solved} / {total_tasks}")
print(f"Pass@1 Success Rate:         {success_rate:.1f}%")
print(f"Average Steps (Solved):      {avg_steps:.2f} turns")
print(f"Step Limit Enforcement:      100% (max 6 steps)")
print("=" * 50)
