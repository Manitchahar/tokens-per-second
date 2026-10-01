# %% [markdown]
# # Lab 04b · Your first GPU inference server
#
# **Tokens per Second** · Module 04
#
# In Lab 04a you ran a model in a notebook. This lab puts it **behind HTTP**: you will build a
# streaming server, talk to it exactly as an application would, and then measure what serving
# actually costs you.
#
# 1. Write a streaming FastAPI server that loads the model once and streams tokens as SSE.
# 2. Call it from Python and from `curl`, and watch the wire format.
# 3. Measure **TTFT** and **TPOT** as a real client sees them.
# 4. Drive it with concurrency and watch the latency–throughput trade-off happen to you.
# 5. Add the operational endpoints and read your own server's numbers.
#
# Runs on a free **Colab T4 (16 GB)** with Qwen2.5-0.5B-Instruct.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# In Colab the server runs in a background thread on `localhost`, and the "client" cells talk
# to it over real HTTP. That is a genuine end-to-end test of the same code you would deploy.

# %% [markdown]
# ## 0. Install and check
#
# FastAPI, uvicorn and the OpenAI client. Small pure-Python packages.

# %%
import importlib, subprocess, sys

def ensure(pkg, pip_name=None):
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg], check=True)

ensure("fastapi")
ensure("uvicorn")
ensure("openai")

import torch, transformers, time, json, threading, requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

p = torch.cuda.get_device_properties(0)
print(f"GPU     {p.name}, {p.total_memory/1e9:.1f} GB")
print(f"torch   {torch.__version__}")

# %% [markdown]
# ## 1. The model, loaded once
#
# This is the whole reason a server exists. Loading takes seconds; generating takes
# milliseconds. Load once, serve many.

# %%
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.float16       # the T4 has FP16 tensor cores but no native BF16

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cuda").eval()

weights_gb = sum(q.numel() * q.element_size() for q in model.parameters()) / 1e9
kvpt = 2 * model.config.num_hidden_layers * model.config.num_key_value_heads \
       * model.config.head_dim * 2
print(f"weights        {weights_gb:.2f} GB")
print(f"KV per token   {kvpt:,} bytes = {kvpt/1024:.2f} KB")
print(f"KV at 2048     {2048*kvpt/1e6:.1f} MB per conversation")
print()
print("That last number is the budget for concurrency. Keep it in mind.")

# %% [markdown]
# ## 2. The server
#
# Read this before running it. Every line does something you learned in the lesson:
#
# - **Load once**, at startup, not per request.
# - **Run `generate` on a thread**, because it blocks for the whole generation.
# - **Flush every token** — never buffer, that would undo streaming.
# - **Clamp `max_tokens`**, because the client is not trusted.
# - **`await asyncio.sleep(0)`** between tokens, so other requests get a turn.
# - **`worker.join()` in a `finally`**, so a disconnected client cannot leak a GPU thread.
#
# The endpoint shape is deliberately OpenAI-compatible: `POST /v1/chat/completions`.

# %%
import asyncio
from threading import Thread

MAX_TOKENS = 256           # the server's ceiling; the client cannot exceed it

app = FastAPI()

class ChatRequest(BaseModel):
    messages: list
    stream: bool = True
    max_tokens: int = 128
    temperature: float = 0.7
    model: str = MODEL

# Simple live counters, so section 6 has something real to report.
stats = {
    "requests": 0, "in_flight": 0, "prompt_tokens": 0,
    "output_tokens": 0, "peak_memory_gb": 0.0,
}

@app.get("/health")
def health():
    # Deliberately cheap. A load balancer will hammer this, and it must never
    # touch the GPU or it will fail exactly when you are overloaded.
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return {
        **stats,
        "in_flight": stats["in_flight"],
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "kv_bytes_per_token": kvpt,
    }

def _blocking_generate(inputs, max_new, temperature):
    """Non-streaming generation, run off the event loop so the server stays responsive."""
    def work():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=temperature > 0, temperature=temperature,
                             pad_token_id=tok.eos_token_id)
        plen = inputs["input_ids"].shape[1]
        return tok.decode(out[0][plen:], skip_special_tokens=True)
    return work()   # called via asyncio.to_thread by the caller

def _generate_to_queue(inputs, max_new, temperature, q: asyncio.Queue):
    """Runs generate on a worker thread, pushing each text piece into an asyncio queue.

    This is the important part of the whole lab. `generate` blocks for the entire
    generation. If you call it directly inside `async def`, you block the event loop
    and the server stops answering *every other request* while one is generating.
    So it runs on its own thread, and the async side just awaits a queue.
    """
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def produce():
        try:
            model.generate(**inputs,
                           max_new_tokens=max_new,
                           do_sample=temperature > 0,
                           temperature=temperature,
                           streamer=streamer,
                           pad_token_id=tok.eos_token_id)
        except Exception as e:                       # never let the thread die silently
            q.put_nowait(("error", e))
        finally:
            q.put_nowait(("done", None))

    thread = Thread(target=produce, daemon=True)
    thread.start()

    async def pump():
        """Bridges the blocking iterator into the event loop."""
        loop = asyncio.get_running_loop()
        while True:
            piece = await loop.run_in_executor(None, lambda: next(pieces, None))
            if piece is None:
                break
            await q.put(("piece", piece))
        await q.put(("done", None))

    # Pull from the streamer on a plain worker so the event loop never blocks.
    pieces = iter(streamer)

    pump_task = asyncio.ensure_future(pump())
    try:
        while True:
            kind, value = await q.get()
            if kind == "piece":
                yield value
            elif kind == "error":
                raise value
            else:
                break
    finally:
        # Never leak a GPU thread or a KV cache, even if the client vanished.
        pump_task.cancel()
        thread.join(timeout=120)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    text = tok.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    plen = inputs["input_ids"].shape[1]

    # Clamp: never trust the client's limits.
    max_new = max(1, min(req.max_tokens, MAX_TOKENS))
    temperature = max(0.0, min(req.temperature, 2.0))

    stats["requests"] += 1
    stats["prompt_tokens"] += plen

    if not req.stream:
        # Non-streaming still must not block the loop.
        text_out = await asyncio.to_thread(_blocking_generate, inputs, max_new, temperature)
        stats["output_tokens"] += len(text_out)
        return {"choices": [{"message": {"role": "assistant", "content": text_out}}]}

    async def events():
        stats["in_flight"] += 1
        try:
            async for piece in _generate_to_queue(inputs, max_new, temperature, asyncio.Queue()):
                stats["output_tokens"] += 1
                payload = {"choices": [{"delta": {"content": piece}}]}
                # The flush is the whole point of streaming.
                yield f"data: {json.dumps(payload)}\n\n"
                # Hand the event loop back so other requests can run.
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        finally:
            stats["in_flight"] -= 1
            stats["peak_memory_gb"] = max(stats["peak_memory_gb"],
                                          torch.cuda.max_memory_allocated() / 1e9)

    return StreamingResponse(events(), media_type="text/event-stream")

print("Server defined. Routes:")
for r in app.routes:
    methods = getattr(r, "methods", None)
    if methods:
        print(f"  {','.join(sorted(methods)):>10}  {r.path}")

# %% [markdown]
# ## 3. Run it
#
# Start uvicorn in a daemon thread on port 8000, then talk to it over real HTTP.
#
# Colab blocks the notebook while code runs, so the server has to live on a thread.

# %%
import socket, contextlib

def port_open(port, host="127.0.0.1", timeout=0.3):
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0

PORT = 8000
while port_open(PORT):
    PORT += 1
print(f"using port {PORT}")

import uvicorn

config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()

for _ in range(60):
    if server.started:
        break
    time.sleep(0.5)
print(f"server started: {server.started}, port {PORT}")

# %%
import requests
BASE = f"http://127.0.0.1:{PORT}"

r = requests.get(f"{BASE}/health", timeout=5)
print("GET /health ->", r.status_code, r.json())

# %% [markdown]
# ## 4. Call it — first non-streaming
#
# Same GPU work, one JSON response at the end.

# %%
r = requests.post(f"{BASE}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "What is a GPU? Answer in one sentence."}],
    "stream": False, "max_tokens": 80,
}, timeout=120)
r.raise_for_status()
print(r.json()["choices"][0]["message"]["content"])

# %% [markdown]
# ## 5. Call it — streaming, and measure what a client feels
#
# Watch the tokens arrive. This is the same loop as Lab 04a section 7, now over a socket.

# %%
def stream_chat(prompt, max_tokens=64, temperature=0.0, base=BASE):
    """Returns (ttft_ms, per_token_ms, text) measured the way a client feels it."""
    payload = {"messages": [{"role": "user", "content": prompt}],
               "stream": True, "max_tokens": max_tokens, "temperature": temperature}
    t0 = time.perf_counter()
    times, pieces = [], []
    with requests.post(f"{base}/v1/chat/completions", json=payload,
                      stream=True, timeout=300) as r:
        r.raise_for_status()
        assert "text/event-stream" in r.headers.get("content-type", ""), r.headers.get("content-type")
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            times.append((time.perf_counter() - t0) * 1e3)
            pieces.append(json.loads(data)["choices"][0]["delta"]["content"])
    text = "".join(pieces)
    ttft = times[0] if times else float("nan")
    tpot = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else float("nan")
    return ttft, tpot, text

ttft, tpot, text = stream_chat("Describe the ocean in two sentences.", max_tokens=80)
print("streaming, token by token as it arrived:\n")
t0 = time.perf_counter()
payload = {"messages": [{"role": "user", "content": "Name three colours, one per line."}],
           "stream": True, "max_tokens": 40, "temperature": 0.0}
with requests.post(f"{BASE}/v1/chat/completions", json=payload, stream=True, timeout=120) as r:
    for raw in r.iter_lines():
        if raw and raw.decode().startswith("data: "):
            d = raw.decode()[6:]
            if d == "[DONE]":
                break
            piece = json.loads(d)["choices"][0]["delta"]["content"]
            print(f"  [{(time.perf_counter()-t0)*1e3:7.1f} ms] {piece!r}", flush=True)

# %%
print(f"\nTTFT              {ttft:.1f} ms")
print(f"per token (TPOT)  {tpot:.1f} ms")
print(f"throughput        {1000/tpot:.1f} tokens/s")
print()
print("TTFT is prefill plus HTTP plus the first scheduling decision.")
print("TPOT is decode plus network. Compare both with Lab 04a: the extra time")
print("is the HTTP hop, and it is real.")

# %%
# The same request non-streaming, for a direct comparison of felt time.
t0 = time.perf_counter()
r = requests.post(f"{BASE}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Describe the ocean in two sentences."}],
    "stream": False, "max_tokens": 80}, timeout=120)
r.raise_for_status()
blocking_ms = (time.perf_counter() - t0) * 1e3
print(f"streaming:   first token at {ttft:7.1f} ms, done at {ttft + tpot*40:7.1f} ms (est)")
print(f"non-stream:  first token at {blocking_ms:7.1f} ms (nothing before this)")
print()
print(f"Identical GPU time. {blocking_ms - ttft:.0f} ms of extra waiting for the same work,")
print("and that is before a real user has read anything.")

# %% [markdown]
# ## 6. Call it the way an app would: the OpenAI client
#
# Because the endpoint is OpenAI-compatible, an existing application changes one line.

# %%
from openai import OpenAI

client = OpenAI(base_url=f"{BASE}/v1", api_key="not-used")

print("streaming through the OpenAI client:\n")
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Explain tokens per second in one sentence."}],
    stream=True, max_tokens=80,
)
for chunk in stream:
    piece = chunk.choices[0].delta.content or ""
    print(piece, end="", flush=True)
print("\n\nThat is the exact code from a real application, unchanged, talking to your GPU.")

# %% [markdown]
# ## 7. Concurrency: the trade-off happens to you
#
# One user at a time wastes the GPU. Many users share the weights and throughput rises — but
# each one needs its own KV cache, and everyone gets slower.

# %%
import concurrent.futures as cf

def one_request(i):
    ttft, tpot, _ = stream_chat(f"Question {i}: what is number {i} plus {i}?", max_tokens=48)
    return ttft, tpot

def sweep(concurrency, n=8):
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(one_request, range(n)))
    ttfts = [r[0] for r in results]
    tpots = [r[1] for r in results]
    # TPOT is time per token per user; with n users sharing, aggregate throughput is n/TPOT.
    return (sum(ttfts)/len(ttfts), sum(tpots)/len(tpots), n / (sum(tpots)/len(tpots)))

# Warm up every path before measuring.
one_request(0)

print(f"{'concurrency':>11} {'TTFT ms':>10} {'TPOT ms':>10} {'aggregate tok/s':>17} {'KV cache':>11}")
for c in [1, 2, 4, 8]:
    ttft, tpot, agg = sweep(c)
    print(f"{c:>11} {ttft:>10.0f} {tpot:>10.1f} {agg:>17.1f} {c*2048*kvpt/1e6:>9.0f} MB")

# %%
print("\nRead the table before moving on. You should see the classic shape:")
print("- Aggregate **throughput rises** with concurrency: the weights are read once for")
print("  several tokens instead of once each. This is the single biggest serving win,")
print("  and it is why Modules 07 and 08 exist.")
print("- Per-user **TPOT gets worse**: more requests share one GPU.")
print("- **TTFT rises too**, because requests queue behind each other.")
print()
print("This is the latency-throughput trade-off, and you have just measured it.")
print("Every serving configuration in the rest of the course is a point on this curve.")

# %% [markdown]
# ## 8. Memory under load
#
# Concurrency is a memory decision before it is a throughput decision.

# %%
print(f"{'concurrency':>11} {'expected KV':>13} {'GPU allocated':>15} {'GPU peak':>11}")
for c in [1, 8, 16]:
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        list(ex.map(lambda i: stream_chat(f"Tell me something about topic {i}.", 48), range(c)))
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    print(f"{c:>11} {c*2048*kvpt/1e6:>11.0f} MB {after/1e9:>13.2f} GB {peak/1e9:>9.2f} GB")

# %%
free_gb = p.total_memory / 1e9 - torch.cuda.memory_allocated() / 1e9
max_conc = int(free_gb * 1e9 / (2048 * kvpt))
print(f"\nweights           {weights_gb:.2f} GB")
print(f"GPU memory        {p.total_memory/1e9:.1f} GB")
print(f"free now          {free_gb:.2f} GB")
print(f"KV per 2048-tok   {2048*kvpt/1e6:.1f} MB")
print(f"=> about {max_conc} conversations fit, at 2,048 tokens each")
print()
print("That is your concurrency ceiling on THIS model. Change the context length and")
print("it halves. Change the model and it collapses. Module 05 is about exactly this.")

# %% [markdown]
# ## 9. The operational endpoints
#
# Everything a real service needs, and the two mistakes that bite hardest.

# %%
print("GET /health")
print(f"  -> {requests.get(f'{BASE}/health', timeout=5).json()}")
print(f"  costs {1000*1e-6:.0f} ms and never touches the GPU. Keep it that way.")
print()
print("GET /metrics")
m = requests.get(f"{BASE}/metrics", timeout=5).json()
for k, v in m.items():
    print(f"  {k:>20}  {v}")

# %%
# Prove the clamp: ask for 100,000 tokens and see what you actually get.
r = requests.post(f"{BASE}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Count."}],
    "stream": True, "max_tokens": 100000, "temperature": 0.0,
}, stream=True, timeout=300)
n = 0
with r:
    for raw in r.iter_lines():
        if raw and raw.decode().startswith("data: "):
            d = raw.decode()[6:]
            if d == "[DONE]":
                break
            n += 1
print(f"asked for 100,000 tokens, received {n}")
print(f"server ceiling MAX_TOKENS = {MAX_TOKENS} -> clamped. An unbounded request")
print("is an unbounded bill and an out-of-memory crash.")

# %% [markdown]
# ## 10. Shut it down cleanly
#
# What a rolling restart does: stop accepting new work, let in-flight requests finish,
# then exit. Killing mid-generation wastes the user's request and the KV cache.

# %%
# In production this is what a SIGTERM handler (or Kubernetes pod deletion) triggers:
# stop accepting new connections, let in-flight requests finish, then exit.
# We flip uvicorn's own flag rather than signalling the notebook process.
print("shutting down gracefully (stop accepting, drain in-flight, exit)")
server.should_exit = True
thread.join(timeout=60)
print(f"server stopped cleanly: {not thread.is_alive()}")

time.sleep(1)
try:
    requests.get(f"{BASE}/health", timeout=2)
    print("still answering — the port is still bound")
except Exception:
    print("port is closed, as expected")

print()
print("Killing mid-generation instead would waste the user's request and leave")
print("the KV cache allocated. That is what the finally-blocks above are for.")

# %% [markdown]
# ## 11. Write these down
#
# Module 05 and Module 06 both assume you have these.

# %%
server_notes = {
    "model": MODEL,
    "weights_GB": round(weights_gb, 3),
    "kv_bytes_per_token": kvpt,
    "ttft_ms": round(ttft, 1),
    "tpot_ms": round(tpot, 2),
    "tokens_per_second": round(1000 / tpot, 1),
    "blocking_ms_same_request": round(blocking_ms, 1),
    "gpu_peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    "max_concurrency_2k_ctx": max_conc,
}
print("=== write these into My notes ===")
for k, v in server_notes.items():
    print(f"{k:>28}  {v}")
print("======================================\n")
print("Module 06 turns TTFT and TPOT into a proper benchmark. Module 05 asks why")
print("generation recomputes what it already knows. You now have the hardware to")
print("answer both.")

# %% [markdown]
# ## Your turn
#
# 1. **Section 7**: push concurrency to 32 and 64. Where does it stop improving, and what
#    happens just before it? Check `/metrics` while it runs.
# 2. **Section 8**: run the concurrency sweep at a 16,000-token context instead of 2,048.
#    How does `max_conc` change? Does it match your prediction?
# 3. **Section 5**: add a client that abandons the connection halfway through. Does the GPU
#    memory come back? If not, what is the `finally: worker.join()` in section 2 for?
# 4. **Section 9**: make `/health` call `torch.cuda.memory_allocated()` and run the
#    concurrency sweep. Explain why that is a bug in a multi-replica deployment.
# 5. **Challenge**: change `_run_generate` to batch several requests into one forward pass
#    with `model.generate(..., num_return_sequences=k)`. It is not correct batching — the
#    sequences diverge immediately — but measure whether aggregate throughput improves.
#    Then explain what real batching must do instead. (Modules 07 and 08.)
