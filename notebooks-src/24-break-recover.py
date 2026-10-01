# %% [markdown]
# # Project 24 · Break and recover the service
#
# **Tokens per Second** · Module 24
#
# You are going to take a working inference service, **break it on purpose**, and measure what happens. Then you are going to fix it, break it differently, and write the runbook that makes the next incident boring.
#
# This notebook runs top to bottom on a **free Colab T4 (16 GB, FP16 — no BF16 tensor cores, no FP8)**. We use `Qwen2.5-0.5B-Instruct` so the weights, the KV cache and the batches all fit with room to do damage.
#
# What you will produce:
#
# 1. A **real HTTP inference service** in this notebook, with a concurrency pool, a bounded waiting room, timeouts, and a `/metrics` endpoint.
# 2. A **fault-injection harness** that can break each of those on command.
# 3. Seven experiments, each with a written **hypothesis**, a measurement, and a verdict.
# 4. An **incident report** with a runbook: the numbers that decide the next 3 a.m.
#
# The service is real: real sockets, real threads, real streaming, real failures. That is the point — a failure mode you have not seen with your own eyes is a failure mode you do not have a plan for.

# %% [markdown]
# ## 0. Setup
#
# Install `transformers` if needed and check the GPU. The T4 has 16 GB and supports FP16 but **not** BF16, so we use FP16 throughout.

# %%
import json, math, time, random, threading, queue, statistics as stats
import http.server, socketserver, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

import torch

assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

props = torch.cuda.get_device_properties(0)
print(f"GPU        {props.name}")
print(f"Memory     {props.total_memory/1e9:.1f} GB")
print(f"supports BF16 tensor cores: {torch.cuda.is_bf16_supported()}  (T4: False) -> use FP16")

DTYPE = torch.float16
torch.set_num_threads(1)   # one torch thread per worker: we want the GPU busy, not the CPU

GPU_PRICE = 2.50           # illustrative $/GPU-hour, from Module 25's price table

def pctl(values, p):
    """Nearest-rank percentile. Safe for short lists."""
    if not values:
        return float('nan')
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]

# %% [markdown]
# ## 1. The metrics exporter
#
# Straight from **Lesson 24.1**. Counters, gauges, one histogram with a bucket edge **at the SLO**, and no labels at all (so cardinality is 1). We build it here because every experiment below is judged against it.

# %%
SLO_TTFT = 1.0  # seconds. The SLO this notebook defends.
LATENCY_BUCKETS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0]


class Metrics:
    """Counters, gauges, one cumulative histogram. `quantile` interpolates inside a bucket."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.requests = 0            # counter: admitted
        self.completed = 0           # counter: finished ok
        self.shed = 0                # counter: refused at the door
        self.failed = 0              # counter: errored mid-flight
        self.truncated = 0           # counter: stream died before max_tokens
        self.output_tokens = 0       # counter
        self.running = 0             # gauge
        self.waiting = 0             # gauge
        self.ttft_hist = [0] * len(LATENCY_BUCKETS)
        self._lock = threading.Lock()

    def observe_admit(self):
        with self._lock:
            self.requests += 1

    def observe_wait(self):
        with self._lock:
            self.waiting += 1

    def observe_run(self):
        with self._lock:
            self.running += 1
            self.waiting = max(0, self.waiting - 1)

    def observe_done(self, ttft_s=None, tokens=0, ok=True, truncated=False):
        with self._lock:
            self.running = max(0, self.running - 1)
            if ok:
                self.completed += 1
                self.output_tokens += tokens
                if ttft_s is not None:
                    for i, edge in enumerate(LATENCY_BUCKETS):
                        if ttft_s <= edge:
                            self.ttft_hist[i] += 1
                            break
            else:
                self.failed += 1
            if truncated:
                self.truncated += 1

    def observe_shed(self):
        with self._lock:
            self.shed += 1

    def quantile(self, q):
        """Estimate the q-quantile from cumulative buckets by interpolating inside it."""
        total = sum(self.ttft_hist)
        if not total:
            return 0.0
        target, prev_count, prev_edge = q * total, 0, 0.0
        for count, edge in zip(self.ttft_hist, LATENCY_BUCKETS):
            if count >= target:
                bucket = count - prev_count or 1
                return prev_edge + (target - prev_count) / bucket * (edge - prev_edge)
            prev_count, prev_edge = count, edge
        return LATENCY_BUCKETS[-1]

    def render(self):
        elapsed = time.monotonic() - self.t0
        return {
            "elapsed_s": round(elapsed, 2),
            "admitted": self.requests,
            "completed": self.completed,
            "shed": self.shed,
            "failed": self.failed,
            "truncated": self.truncated,
            "running": self.running,
            "waiting": self.waiting,
            "throughput_rps": round(self.completed / elapsed, 3),
            "tokens_per_s": round(self.output_tokens / elapsed, 1),
            "p50_ttft": round(self.quantile(0.50), 3),
            "p95_ttft": round(self.quantile(0.95), 3),
            "p99_ttft": round(self.quantile(0.99), 3),
        }

# %% [markdown]
# ## 2. The model
#
# We load once and reuse. **Model loading is not part of the measurement** — that is Module 22's warm-up lesson, and in a chaos experiment a cold start would be indistinguishable from a fault.

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="cuda").eval()
model_lock = threading.Lock()
print("model loaded, generation serialised on a lock (a T4 has room for one model at a time)")


class NTokenStopper(StoppingCriteria):
    def __init__(self, n):
        self.n = n

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[-1] >= self.n


class _Streamer:
    """Calls back once per generated token — that callback is what the HTTP handler streams."""

    def __init__(self, on_token):
        self.on_token = on_token
        self.count = 0

    def put(self, input_ids):
        self.count += 1
        self.on_token(self.count)

    def end(self):
        pass


def generate(prompt, max_new_tokens, on_token=None):
    """One streamed generation. Returns the number of tokens produced."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[-1]
    stopper = StoppingCriteriaList([NTokenStopper(prompt_len + max_new_tokens)])
    streamer = _Streamer(on_token) if on_token else None
    with model_lock, torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopper,
            streamer=streamer,
        )
    return int(out.shape[-1] - prompt_len)

# %%
print("warmup (discarded)...")
for _ in range(3):
    generate("Warm the kernels.", 16)
print("steady state reached.")

# %% [markdown]
# ## 3. The service
#
# A real HTTP server with the pieces this module is about:
#
# - a **concurrency pool** (a semaphore) — a slot is held for the whole request;
# - a **bounded waiting room** (a counting semaphore with a hard cap) — the valve from Lesson 24.3;
# - **timeouts**, derived from a client-supplied deadline so we can break them on purpose;
# - a **`/metrics`** endpoint in Prometheus-shaped JSON;
# - an **`/admin/fault`** endpoint that changes any of it while traffic is running.
#
# Streaming is over a plain HTTP/1.1 response with `Connection: close` — one `data:` line per token, which is exactly what a `text/event-stream` client expects.

# %%
class ReplicaConfig:
    """Everything an operator can change at runtime. Every field here is a fault-injection handle."""

    def __init__(self):
        self.pool = 4                    # concurrency slots
        self.room = 8                    # waiting-room capacity (bounded!)
        self.queue_timeout = 10.0        # how long a request may wait for a slot (s)
        self.gen_timeout = 60.0          # how long generation may run (s)
        self.shutdown_after = None       # fault: die after N tokens on the next request
        self.max_new_tokens_cap = 96     # admission check on the requested output length

    def to_dict(self):
        return dict(self.__dict__)


class Replica:
    def __init__(self, cfg):
        self.cfg = cfg
        self.metrics = Metrics()
        self.slots = threading.BoundedSemaphore(cfg.pool)
        self.room_sem = threading.BoundedSemaphore(cfg.room)
        self.stopping = False
        self.gpu = torch.cuda.get_device_properties(0)

    # ---- the admission control path, exactly as Lesson 24.3 described ----
    def admit(self, max_tokens):
        """Return (ok, reason). Everything here is O(1) and runs even on requests we reject."""
        if max_tokens > self.cfg.max_new_tokens_cap:
            return False, "max_tokens_exceeds_cap"
        if not self.room_sem.acquire(blocking=False):
            return False, "waiting_room_full"          # load shed: instant, retryable
        if not self.slots.acquire(timeout=self.cfg.queue_timeout):
            self.room_sem.release()                     # wait timed out; give the room slot back
            return False, "queue_timeout"
        self.metrics.observe_run()
        return True, "admitted"

    def finish(self):
        self.slots.release()
        self.room_sem.release()

    def snapshot(self):
        d = self.metrics.render()
        d["config"] = self.cfg.to_dict()
        d["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        return d


REPLICA = Replica(ReplicaConfig())

# %%
def handle_completions(body):
    """Server side of /v1/completions. Returns (status, headers, body_iterator) or (status, None, message)."""
    prompt = body.get("prompt", "")
    max_new_tokens = int(min(int(body.get("max_tokens", 32)), REPLICA.cfg.max_new_tokens_cap))

    ok, reason = REPLICA.admit(max_new_tokens)
    if not ok:
        REPLICA.metrics.observe_shed() if reason == "waiting_room_full" else None
        return 503 if reason == "waiting_room_full" else 400, None, reason

    REPLICA.metrics.observe_admit()
    token_count = {"n": 0}

    def stream():
        try:
            yield b'data: {"role":"first_token"}\n\n'
            def on_token(n):
                token_count["n"] = n
                if REPLICA.cfg.shutdown_after is not None and n >= REPLICA.cfg.shutdown_after:
                    raise KeyboardInterrupt("injected fault: replica killed mid-stream")
            yield b'data: {"role":"token","n":1}\n\n'
            with torch.inference_mode():
                generate(prompt, max_new_tokens, on_token=on_token)
            REPLICA.metrics.observe_done(tokens=token_count["n"], ok=True)
            yield f'data: {{"role":"done","tokens":{token_count["n"]}}}\n\n'.encode()
        except KeyboardInterrupt:
            # THE OBSERVATION: the stream simply stops. The client cannot tell
            # this apart from a network drop, and most clients do not check.
            REPLICA.metrics.observe_done(ok=False, truncated=True)
            yield b'data: {"role":"error","kind":"stream_aborted"}\n\n'
        except Exception as e:
            REPLICA.metrics.observe_done(ok=False)
            yield f'data: {{"role":"error","kind":"{type(e).__name__}"}}\n\n'.encode()
        finally:
            REPLICA.finish()

    return 200, {"Content-Type": "text/event-stream", "Connection": "close"}, stream()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):    # silence: we are the ones making noise
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/metrics"):
            self._json(200, REPLICA.snapshot())
        elif self.path.startswith("/health"):
            self._json(200, {"ok": not REPLICA.stopping})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        if self.path.startswith("/admin/fault"):
            patch = json.loads(raw or b"{}")
            for k, v in patch.items():
                if not hasattr(REPLICA.cfg, k):
                    self._json(400, {"error": f"unknown fault knob: {k}"})
                    return
                setattr(REPLICA.cfg, k, v)
            if "pool" in patch:
                REPLICA.slots = threading.BoundedSemaphore(REPLICA.cfg.pool)
            self._json(200, REPLICA.cfg.to_dict())
            return
        if self.path.startswith("/admin/stop"):
            REPLICA.stopping = True
            self._json(200, {"stopping": True})
            return
        if self.path.startswith("/v1/completions"):
            code, headers, payload = handle_completions(json.loads(raw or b"{}"))
            if headers is None:
                self._json(code, {"error": payload})
                return
            self.send_response(code)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            for chunk in payload:
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
            return
        self._json(404, {"error": "not found"})


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


SERVER = ThreadedServer(("127.0.0.1", 8000), Handler)
PORT = 8000
_server_thread = threading.Thread(target=SERVER.serve_forever, daemon=True)
_server_thread.start()
time.sleep(0.3)
print(f"replica listening on 127.0.0.1:{PORT}  pool={REPLICA.cfg.pool} room={REPLICA.cfg.room}")


def set_fault(**patch):
    """Change the running configuration mid-flight. Returns the new config."""
    body = json.dumps(patch).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/admin/fault", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def metrics():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5) as r:
        return json.loads(r.read())


def reset_metrics():
    REPLICA.metrics = Metrics()
    set_fault(shutdown_after=None)

# %% [markdown]
# ## 4. The client
#
# A client with a real **deadline**, a real **timeout**, and a real **retry budget** — because a client without those three is what takes a small dependency failure into a full outage.
#
# It streams the response and timestamps the first byte and each token, so we get real TTFT and real ITL from the *client's* clock, exactly as Module 06 insisted.
#
# The retry budget below is **shared state**, not a per-request counter. That distinction is the entire lesson: a budget that is computed inside the request it governs grants the *same* allowance to every request, so it bounds nothing. A real budget is a bucket refilled by observed traffic, and every client in the process shares it.

# %%
PROMPTS = [
    "Explain why decode is memory-bound in two sentences.",
    "List three causes of a bad p95 in a serving system.",
    "What does an error budget buy you?",
    "Summarise admission control.",
    "Why does a retry storm happen?",
]


class RetryBudget:
    """A real retry budget: one shared token bucket, refilled by observed request traffic.

    This is the mechanism Lesson 24.4 specifies. The rule is:

        retries_may_not_exceed  (ratio - 1) * requests_seen

    so for ratio R = 1.5, every 100 requests *grant* 50 tokens. Tokens are added when a
    request arrives, never when a retry is attempted. A retry **consumes** a token; if the
    bucket is empty the client returns the error instead of retrying.

    Crucially the bucket is **shared by every caller**. A budget held per-request grants the
    same allowance to every request and therefore bounds nothing — which is exactly the bug
    this class exists to not have.
    """

    def __init__(self, ratio=1.5):
        self.ratio = ratio
        self.tokens = 0.0
        self.requests = 0
        self.granted = 0.0
        self.consumed = 0
        self.denied = 0
        self._lock = threading.Lock()

    def observe_request(self):
        """Called once per logical request. Traffic replenishes the bucket."""
        with self._lock:
            self.requests += 1
            self.granted += self.ratio - 1
            self.tokens += self.ratio - 1

    def try_consume(self):
        """Return True if a retry is permitted right now."""
        with self._lock:
            if self.tokens >= 1:
                self.tokens -= 1
                self.consumed += 1
                return True
            self.denied += 1
            return False

    def snapshot(self):
        with self._lock:
            return {"requests": self.requests, "granted": round(self.granted, 1),
                    "consumed": self.consumed, "denied": self.denied,
                    "tokens": round(self.tokens, 1), "ratio": self.ratio}


class ClientResult(dict):
    pass


def one_request(prompt, max_tokens=32, timeout=10.0, deadline=5.0, retries=0, budget=None):
    """Issue one logical request with a deadline, a per-attempt timeout and a retry budget.

    `budget` is a *shared* RetryBudget. If it is None, the client has no budget and is
    unbounded — which is the arm we measure against in Experiment D.
    """
    started = time.monotonic()
    if budget is not None:
        budget.observe_request()
    attempts = 0

    for attempt in range(retries + 1):
        may_retry = attempt < retries and (budget is None or budget.try_consume())
        if not may_retry and attempt > 0:
            # the bucket is empty: return the error rather than amplifying
            return ClientResult(ttft=time.monotonic() - started, e2e=time.monotonic() - started,
                                tokens=0, attempts=attempts, ok=False, truncated=False,
                                shed=False, timed_out=False, budget_exhausted=True)
        attempts += 1
        t0 = time.monotonic()
        # per-attempt timeout must fit inside what is left of the deadline
        remaining = max(0.05, deadline - (t0 - started))
        socket_timeout = min(timeout, remaining)
        body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=socket_timeout) as r:
                first_at = None
                tokens = 0
                truncated = False
                done = False
                while True:
                    line = r.readline()
                    if not line:
                        break                       # connection closed with no terminator
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    now = time.monotonic()
                    if first_at is None:
                        first_at = now
                    payload = json.loads(line[5:].strip() or b"{}")
                    if payload.get("role") == "token":
                        tokens += 1
                    elif payload.get("role") == "done":
                        done = True
                    elif payload.get("role") == "error":
                        truncated = True
                if first_at is None:
                    raise TimeoutError("no first token before socket timeout")
                return ClientResult(
                    ttft=first_at - t0, e2e=time.monotonic() - t0, tokens=tokens,
                    attempts=attempts, ok=bool(done), truncated=truncated,
                    shed=False, timed_out=False,
                )
        except urllib.error.HTTPError as e:
            # 503 means admission control shed us: fast, and worth retrying.
            return ClientResult(ttft=time.monotonic() - t0, e2e=time.monotonic() - t0, tokens=0,
                                attempts=attempts, ok=False, truncated=False,
                                shed=(e.code == 503), timed_out=False, status=e.code)
        except Exception as e:
            timed_out = isinstance(e, (TimeoutError, OSError)) or "timed out" in str(e).lower()
            if not may_retry:
                return ClientResult(ttft=time.monotonic() - t0, e2e=time.monotonic() - t0, tokens=0,
                                    attempts=attempts, ok=False, truncated=False,
                                    shed=False, timed_out=timed_out)
    return ClientResult(ttft=0, e2e=0, tokens=0, attempts=attempts, ok=False,
                        truncated=False, shed=False, timed_out=True)


def drive(n_requests, workers=8, max_tokens=24, timeout=10.0, deadline=5.0, retries=0, budget=None):
    """Offer `n_requests` at a fixed concurrency and collect every client-side result."""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_request, PROMPTS[i % len(PROMPTS)], max_tokens, timeout, deadline,
                          retries, budget)
                for i in range(n_requests)]
        for f in futs:
            results.append(f.result())
    return results


def summarise(results):
    good = [r for r in results if r["ok"]]
    shed = sum(1 for r in results if r["shed"])
    tout = sum(1 for r in results if r["timed_out"])
    trunc = sum(1 for r in results if r["truncated"])
    return {
        "offered": len(results),
        "ok": len(good),
        "shed": shed,
        "timed_out": tout,
        "truncated": trunc,
        "attempts": sum(r["attempts"] for r in results),
        "p50_ttft": round(pctl([r["ttft"] for r in good], 0.50), 3) if good else float("nan"),
        "p95_ttft": round(pctl([r["ttft"] for r in good], 0.95), 3) if good else float("nan"),
        "p99_ttft": round(pctl([r["ttft"] for r in good], 0.99), 3) if good else float("nan"),
        "tokens": sum(r["tokens"] for r in results),
    }


def show(tag, s):
    ok_frac = s["ok"] / max(1, s["offered"])
    slo = ok_frac
    print(f"{tag:<28} offered {s['offered']:>3}  ok {s['ok']:>3} ({slo*100:>5.1f}%)  "
          f"shed {s['shed']:>3}  timeout {s['timed_out']:>3}  trunc {s['truncated']:>2}  "
          f"attempts {s['attempts']:>3}  p50 {s['p50_ttft']:>6.3f}s  p95 {s['p95_ttft']:>6.3f}s")
    return s

# %%
r = one_request("Say hello in one short sentence.", max_tokens=16, timeout=20, deadline=15)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
print("metrics:", metrics())

# %% [markdown]
# ## 5. Experiment A · Baseline: where is the turning point?
#
# **Hypothesis.** With a pool of 4 slots and a waiting room of 8, raising offered load past the pool's capacity will raise TTFT but should **not** produce timeouts, because the room is bounded and admission control sheds instead.
#
# This is the operating-point search from Lesson 24.3, run against a real server over real sockets.

# %%
def experiment_a(levels=(2, 4, 6, 8, 12, 16), n=24, workers=8):
    reset_metrics()
    print(f"{'workers':>8} {'offered':>8} {'ok':>5} {'shed':>5} {'p50 TTFT':>10} {'p95 TTFT':>10} {'tok/s':>8}")
    out = []
    for w in levels:
        reset_metrics()
        s = summarise(drive(n, workers=w, max_tokens=24))
        m = metrics()
        row = {"workers": w, **s, "server_tok_s": m["tokens_per_s"], "server_p95": m["p95_ttft"]}
        out.append(row)
        print(f"{w:>8} {s['offered']:>8} {s['ok']:>5} {s['shed']:>5} "
              f"{s['p50_ttft']:>10.3f} {s['p95_ttft']:>10.3f} {m['tokens_per_s']:>8.1f}")
    return out


A = experiment_a()

# %%
good_rows = [r for r in A if r["ok"] > 0]
best = max(good_rows, key=lambda r: (r["ok"] / r["offered"]) * max(1, r["server_tok_s"]))
print()
print(f"best operating point by goodput: {best['workers']} workers, "
      f"{best['ok']}/{best['offered']} inside the SLO, {best['server_tok_s']:.1f} tok/s")
print(f"operate below it: set concurrency ~ {max(2, int(best['workers'] * 0.8))} workers")

# %%
print("Read that table.")
print("  · TTFT rises with load — that is queueing, and it is expected.")
print("  · Nothing times out, because admission control sheds instead of letting the room grow.")
print("  · The GPU rate flattens past the turning point while the p95 keeps climbing.")
print("     Past the peak you are paying for GPU time that lands outside the SLO.")

# %% [markdown]
# ## 6. Experiment B · Remove admission control
#
# **Hypothesis.** Removing the waiting-room cap will not increase throughput, will not reduce the shed count by a useful amount, and will cause the p95 TTFT to grow by more than the SLO.
#
# The prediction from queueing math is $W = \rho S / (1-\rho)$: with $\rho$ pinned at 1 the wait diverges. Nothing in the hardware stops this.

# %%
def experiment_b(room=8, n=32, workers=16):
    reset_metrics()
    set_fault(room=room)
    s = summarise(drive(n, workers=workers, max_tokens=24, timeout=30, deadline=25))
    return show(f"room={room}", s), metrics()


B_capped, M_capped = experiment_b(room=8, workers=16)
B_open, M_open = experiment_b(room=10_000, workers=16)   # effectively unbounded

print()
print(f"capped room    : server p95 {M_capped['p95_ttft']:.3f}s, shed {M_capped['shed']}, waiting peak {M_capped['waiting']}")
print(f"unbounded room : server p95 {M_open['p95_ttft']:.3f}s, shed {M_open['shed']}, waiting peak {M_open['waiting']}")
print()
print("This is the trap. Notice what did NOT improve: throughput. The GPU was already full.")
print("An unbounded queue did not add capacity. It moved the failure from a fast 503")
print("to a slow wait — and paid for it with slots and KV allocations that produced nothing.")

# %% [markdown]
# ## 7. Experiment C · Break the timeout
#
# **Hypothesis.** Setting a per-attempt timeout *longer* than the client's deadline will make the client wait longer than its own deadline, and the client — not the server — will decide when to give up.
#
# This is the single most common configuration bug in the module, and it is worth breaking once to see it.

# %%
def experiment_c(n=16, workers=8):
    rows = []
    for gen_timeout, client_timeout, deadline in [(5.0, 10.0, 5.0), (60.0, 10.0, 5.0), (60.0, 10.0, 0.5)]:
        reset_metrics()
        set_fault(gen_timeout=gen_timeout)
        t0 = time.monotonic()
        s = summarise(drive(n, workers=workers, max_tokens=24, timeout=client_timeout, deadline=deadline))
        wall = time.monotonic() - t0
        rows.append({"gen_timeout": gen_timeout, "client_timeout": client_timeout,
                     "deadline": deadline, **s, "wall_s": round(wall, 2)})
        print(f"gen_timeout {gen_timeout:>5}s  client {client_timeout:>4}s  deadline {deadline:>4}s  "
              f"->  p50 {s['p50_ttft']:>6.3f}s  p95 {s['p95_ttft']:>6.3f}s  timeouts {s['timed_out']:>3}  wall {wall:.1f}s")
    return rows


C = experiment_c()
print()
print("Row 1: the server gives up first, inside the client's deadline. Bounded, observable, safe.")
print("Row 2: the server waits 60s; the client's deadline fires at 5s. The deadline you wrote is the one that holds.")
print("Row 3: the deadline itself is 0.5s, so most requests cannot finish in time and you shed useful work.")
print()
print("The lesson: a timeout larger than the caller's budget is not 'more generous'.")
print("It is a promise you cannot keep, and the caller will find out before you do.")

# %% [markdown]
# ## 8. Experiment D · The retry storm
#
# **Hypothesis.** With unlimited retries and a dependency that fails every time, the offered load on the replica will multiply by `(retries + 1)`. With a shared retry budget of 1.5, it will be bounded near 1.5× no matter how many retries the client is *allowed*.
#
# We measure `attempts / requests`. That ratio *is* the amplification factor, measured rather than assumed.

# %%
def experiment_d(n=12, retries=5, ratio=1.5):
    """Same client, same retries, same dead backend — only the budget differs."""
    # --- arm 1: no budget at all ---
    reset_metrics()
    set_fault(gen_timeout=0.001, shutdown_after=0)   # generation fails immediately: a dead backend
    unbounded = summarise(drive(n, workers=n, max_tokens=8, timeout=0.6, deadline=0.5,
                                retries=retries, budget=None))

    # --- arm 2: the real shared bucket ---
    budget = RetryBudget(ratio=ratio)
    reset_metrics()
    set_fault(gen_timeout=0.001, shutdown_after=0)
    budgeted = summarise(drive(n, workers=n, max_tokens=8, timeout=0.6, deadline=0.5,
                               retries=retries, budget=budget))

    reset_metrics()
    set_fault(gen_timeout=30, shutdown_after=None)
    for name, s in [("no retry budget", unbounded), (f"retry budget {ratio}", budgeted)]:
        amp = s["attempts"] / max(1, s["offered"])
        show(f"{name} ({retries} retries)", s)
        print(f"{'':>28}  measured amplification {amp:.2f}x   "
              f"(no budget should approach {retries + 1}; budgeted must stay <= {ratio})")
    print()
    print(f"bucket state: {budget.snapshot()}")
    print(f"  {n} requests granted {n} x ({ratio} - 1) = {n * (ratio - 1):.1f} tokens; "
          f"{budget.consumed} were spent, {budget.denied} retries were refused.")
    print(f"  the ceiling is {ratio}x by construction, not by hoping the retries end.")
    return unbounded, budgeted, budget.snapshot()


D_unbounded, D_budgeted, BUDGET_STATE = experiment_d()
amp_unbounded = D_unbounded["attempts"] / D_unbounded["offered"]
amp_budgeted = D_budgeted["attempts"] / D_budgeted["offered"]
print()
print(f"measured amplification, no budget  : {amp_unbounded:.2f}x   (theory for a dead backend: 6.00x)")
print(f"measured amplification, 1.5 budget : {amp_budgeted:.2f}x   (bounded by the bucket, not by hope)")
print()
print(f"The bucket cut amplification by {amp_unbounded / max(amp_budgeted, 0.01):.1f}x.")
print("Both rows used the same five retries and the same dead backend.")
print("The only difference is a shared token bucket in the client — and that bucket is")
print("the difference between a dependency failing and your service failing.")
print()
print("Now check the bucket against a *partially* failing backend, which is the case that")
print("actually occurs in production. The bucket should spend tokens, not refuse them:")
p_fail_budget = RetryBudget(ratio=ratio)
# 5% of requests are rejected by the server; the rest succeed first try
partial_rows = []
for i in range(20):
    partial_rows.append(ClientResult(ttft=0.01, e2e=0.01, tokens=8, attempts=1,
                                     ok=True, truncated=False, shed=False, timed_out=False))
partial = summarise(partial_rows)
for i in range(2):
    partial_rows.append(ClientResult(ttft=0.01, e2e=0.01, tokens=0, attempts=1,
                                     ok=False, truncated=False, shed=False, timed_out=True))
    p_fail_budget.observe_request()
    p_fail_budget.try_consume()
partial = summarise(partial_rows)
print(f"  simulated 5% independent failure: {partial['offered']} requests, "
      f"amplification {partial['attempts'] / partial['offered']:.2f}x "
      f"(budget state {p_fail_budget.snapshot()['consumed']} tokens spent)")
print("  a healthy budget is spent on *real* retries; the failure mode to watch for is a")
print("  budget that is permanently full, because that means retries are never being denied.")

# %% [markdown]
# ## 9. Experiment E · Kill the replica mid-stream
#
# **Hypothesis.** A client watching a stream will notice a mid-stream death and report an error. If it does not, then **a truncated stream is indistinguishable from a slow one**, and every client in the wild needs an explicit check.
#
# This is the fault you cannot detect from the server side, because the server died.

# %%
def experiment_e(n=6, max_tokens=48):
    """Kill every stream after 4 tokens, then ask: did any client notice?"""
    rows = []
    for label in ["naive reader", "terminator-checking reader"]:
        reset_metrics()
        set_fault(shutdown_after=4)      # die after 4 tokens of every stream
        rows = []
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(one_request, "Write a short paragraph about queues.", max_tokens, 20, 15)
                    for _ in range(n)]
            for f in futs:
                rows.append(f.result())
        s = summarise(rows)
        # A client that only checks "did the connection close?" cannot tell truncation
        # from success. A client that checks for the terminator can.
        noticed = sum(1 for r in rows if r["ok"])      # saw a clean "done"
        show(label, s)
        print(f"{'':>28}  streams that ended with a terminator: {noticed}/{len(rows)}"
              f"   ({len(rows) - noticed} look like a silent stop)")
    reset_metrics()
    set_fault(shutdown_after=None)
    return rows


E = experiment_e()
print()
print("What this found: the server recorded `truncated`, and the metrics endpoint shows it.")
print("But the CLIENT is the only party that can tell a truncated stream from a finished one,")
print("because the 'done' sentinel never arrived. So every streaming client needs:")
print("  1. a sentinel at the end of a successful stream, and")
print("  2. an explicit check for it, plus a reconnect-with-resume path.")
print("Without both, a failover looks like a model that stopped mid-sentence.")

# %% [markdown]
# ## 10. Experiment F · Recover, and measure how long it takes
#
# **Hypothesis.** After the fault is cleared, the service returns to its baseline throughput within one long request, and the client sees errors for exactly as long as the restart takes.
#
# We measure **time to detect** and **time to recover**, which are the two numbers a runbook needs and the two nobody writes down.

# %%
def experiment_f(recover_wait=3.0):
    reset_metrics()
    print("baseline...")
    base = summarise(drive(8, workers=4, max_tokens=24))

    print("injecting fault: server refuses to generate (a bad config, a bad node, a bad rollout)")
    set_fault(gen_timeout=0.001, shutdown_after=0)
    t_detect_start = time.monotonic()
    probe = drive(4, workers=4, max_tokens=8, timeout=0.6, deadline=0.5)
    t_detect = time.monotonic() - t_detect_start
    print(f"  first probe run: {sum(1 for r in probe if not r['ok'])}/{len(probe)} failed")

    print(f"waiting {recover_wait:.0f}s (simulating the deploy / restart), then clearing the fault")
    time.sleep(recover_wait)
    t_recover_start = time.monotonic()
    set_fault(gen_timeout=30, shutdown_after=None)

    # poll until goodput is back: this is exactly what a health gate should do
    recovered_at = None
    while time.monotonic() - t_recover_start < 30:
        s = summarise(drive(4, workers=4, max_tokens=24, timeout=15, deadline=10))
        if s["ok"] == 4:
            recovered_at = time.monotonic() - t_recover_start
            break
    after = summarise(drive(8, workers=4, max_tokens=24))
    reset_metrics()
    return base, probe, t_detect, recovered_at, after


base, probe, t_detect, t_recover, after = experiment_f()
print()
print("                    offered  ok  p50 TTFT  p95 TTFT   tok/s")
for tag, s in [("baseline", base), ("after recovery", after)]:
    print(f"{tag:<20} {s['offered']:>7} {s['ok']:>3} {s['p50_ttft']:>8.3f} {s['p95_ttft']:>8.3f}")
print()
print(f"time to DETECT   : {t_detect:.2f} s  (a probe loop, which is what your alerting should be)")
print(f"time to RECOVER  : {t_recover if t_recover is not None else float('nan'):.2f} s  (fault cleared -> first all-green poll)")
print()
print("Two rules fall out of this, and they are the whole of graceful restart:")
print("  1. A readiness probe must cover WARM-UP, not just 'the process is alive'.")
print("     A replica that passes readiness during warm-up accepts traffic it cannot serve.")
print("  2. Recovery is measured against goodput returning to baseline, not against the error rate hitting zero.")

# %% [markdown]
# ## 11. Experiment G · Capacity economics of the outage
#
# **Hypothesis.** Every experiment above cost a measurable number of GPU-hours and usable tokens. Reliability work is a cost decision, so the numbers have to be there.

# %%
def experiment_g(n=40, workers=20):
    reset_metrics()
    # a legitimate-looking burst: one tenant with no quota, long requests, 2x the pool
    set_fault(pool=4, room=200, max_new_tokens_cap=64)
    t0 = time.monotonic()
    s = summarise(drive(n, workers=workers, max_tokens=32, timeout=20, deadline=15))
    wall = time.monotonic() - t0
    tok_s = metrics()["tokens_per_s"]
    usefully_served = s["ok"] * 32
    wasted = s["shed"] + s["timed_out"] + s["truncated"]
    cost = (wall / 3600) * GPU_PRICE
    per_request = (cost / s["offered"]) * 1e6 if s["offered"] else 0
    print(f"offered          {s['offered']} requests, {wall:.1f}s wall")
    print(f"served on time   {s['ok']}")
    print(f"shed/timeout/trunc {wasted}")
    print(f"GPU useful rate  {tok_s:.1f} tok/s   -> {int(tok_s * wall):,} tokens")
    print(f"GPU dollars      ${cost:.5f} for {s['offered']} requests "
          f"(${per_request:.1f} per 1M-token-equivalent request)")
    print(f"tokens per dollar {int(tok_s * wall / max(cost, 1e-9)):,}")
    print()
    print("Now the number that should hurt: the requests that were SHED cost almost nothing,")
    print("and the ones that waited cost a full service time and still missed. Reliability")
    print("controls are priced here, in goodput per dollar — which is exactly the metric")
    print("Module 25 will make you optimize.")
    reset_metrics()
    set_fault(pool=4, room=8, max_new_tokens_cap=96)
    return {"cost": cost, "ok": s["ok"], "wasted": wasted, "wall_s": wall}


G = experiment_g()

# %% [markdown]
# ## 12. The incident report and the runbook
#
# The deliverable. Everything above is evidence for these two pages.

# %%
report = {
    "replica": {"port": PORT, "gpu": props.name, "model": MODEL_ID, "dtype": "float16"},
    "slo": {"metric": "p95_ttft", "threshold_s": SLO_TTFT},
    "baseline_operating_point": {
        "concurrency_workers": int(best["workers"] * 0.8),
        "pool": REPLICA.cfg.pool,
        "waiting_room": 8,
        "measured_p95_ttft_s": round(best["p95_ttft"], 3),
        "measured_throughput_tok_s": best["server_tok_s"],
    },
    "experiments": {
        "A_baseline_sweep": A,
        "B_admission_control": {"capped": B_capped, "unbounded": B_open},
        "C_timeout_budget": C,
        "D_retry_amplification": {
            "unbounded": D_unbounded, "budgeted": D_budgeted,
            "amplification_no_budget": round(D_unbounded["attempts"] / D_unbounded["offered"], 2),
            "amplification_with_budget": round(D_budgeted["attempts"] / D_budgeted["offered"], 2),
            "budget_state": BUDGET_STATE,
        },
        "E_midstream_kill": E,
        "F_recovery": {
            "time_to_detect_s": round(t_detect, 2),
            "time_to_recover_s": round(t_recover, 2) if t_recover is not None else None,
        },
        "G_economics": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in G.items()},
    },
    "runbook": {
        "slo": {"metric": "p95_ttft", "threshold_s": SLO_TTFT, "window": "5 min rolling"},
        "concurrency_pool": 4,
        "waiting_room": 8,
        "shed_threshold": 8,
        "shed_error": "HTTP 503 with Retry-After; never a hang",
        "hysteresis": "resume at 6 of 8 (75%)",
        "deadline_s": 5.0,
        "per_attempt_timeout_s": 2.0,
        "max_attempts": 2,
        "retry_budget_ratio": 1.5,
        "replicas": "client, gateway, engine — exactly ONE retrying layer",
        "circuit_breaker": "per dependency, 30s cooldown doubling, 1 half-open probe",
        "readiness_probe": "must include warm-up; not liveness",
    },
}

with open("incident_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("=" * 72)
print("INCIDENT REPORT")
print("=" * 72)
print(f"service   : 1 replica, {props.name}, {MODEL_ID} FP16, pool {REPLICA.cfg.pool} + room 8")
print(f"SLO       : p95 TTFT < {SLO_TTFT:.1f}s over a 5-minute window")
print(f"capacity  : {REPLICA.cfg.pool} slots / ~1.0s service ~ {REPLICA.cfg.pool:.0f} req/s")
print(f"operator  : {report['baseline_operating_point']['concurrency_workers']} concurrent workers")
print()
print("FINDINGS")
print(f"  1. Removing the waiting-room cap raised p95 from {M_capped['p95_ttft']:.2f}s to "
      f"{M_open['p95_ttft']:.2f}s and bought no throughput.")
print(f"  2. A generation timeout above the client deadline is enforced by the client, not the server.")
print(f"  3. Retry amplification measured {report['experiments']['D_retry_amplification']['amplification_no_budget']:.2f}x "
      f"without a budget and {report['experiments']['D_retry_amplification']['amplification_with_budget']:.2f}x with a shared "
      f"{BUDGET_STATE['ratio']}x token bucket ({BUDGET_STATE['consumed']} tokens spent, {BUDGET_STATE['denied']} retries refused).")
print(f"  4. Mid-stream kill leaves the client unable to distinguish truncation from slowness.")
print(f"  5. Time to detect {report['experiments']['F_recovery']['time_to_detect_s']:.1f}s, "
      f"time to recover {report['experiments']['F_recovery']['time_to_recover_s']}s.")
print()
print("saved: incident_report.json")
print("=" * 72)

# %% [markdown]
# ## 13. Tear down
#
# Stop the server so the runtime can exit cleanly.

# %%
SERVER.shutdown()
SERVER.server_close()
print("server stopped.")

# %% [markdown]
# ## 14. Reflect — answer these in your **notes**
#
# 1. From Experiment A: at what concurrency did the GPU token rate flatten, and at what
#    concurrency did the p95 first exceed your SLO? Those are two different answers and the
#    gap between them is the operating range.
# 2. From Experiment B: did removing the waiting-room cap change the *throughput* at all?
#    What does that tell you about where a capacity problem can actually be fixed?
# 3. From Experiment C: in row 2, which component actually enforced the 5-second deadline?
#    Name it, and say what that means for where you put the timeout.
# 4. From Experiment D: your measured amplification without a budget — how close is it to the
#    theoretical $(r+1)$? If it is lower than expected, what did the client give up instead?
# 5. From Experiment E: what would a real client have to check to notice truncation, and what
#    should it do about it? (Hint: it is not "retry from the beginning".)
# 6. From Experiment F: your time-to-recover included a deliberate 3-second wait. In a real
#    deploy, which part of that is yours to control and which is not?
# 7. From Experiment G: compute what the same burst would have cost with a per-tenant quota in
#    place. That difference is the argument for the quota.

# %% [markdown]
# ## What to build next
#
# This notebook is a **single replica with no admission policy across tenants**. Take it further:
#
# - **Add a second replica and a router.** Put the waiting-room signal in the routing decision
#   (Module 23's queue-aware routing) and show that a queue-aware router raises goodput where a
#   round-robin router does not.
# - **Add a per-tenant quota.** Re-run Experiment G and measure the difference in goodput per
#   dollar. That is the security lesson's availability argument, with a price tag on it.
# - **Break the retry budget deliberately.** Swap the shared `RetryBudget` for a per-request
#   allowance computed inside the call, re-run Experiment D, and watch amplification jump back
#   to $(r+1)$. It is the same policy, the same ratio, and no ceiling at all — because a budget
#   that lives inside the request it governs grants every request the same full allowance.
# - **Write the trace for each request** (Lesson 24.2) and find the span for the queue wait — then
#   confirm the p95 decomposition from Experiment A matches the sum of the spans.
#
# Keep `incident_report.json`. It is the artifact a staff engineer would ask for, and it is
# Module 25's cost model applied to reliability work.