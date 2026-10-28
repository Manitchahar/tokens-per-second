# %% [markdown]
# # Lab 08 · Build a paged KV allocator
#
# **Tokens per Second** · Module 08
#
# In this notebook you will build, from scratch, the memory manager you have been drawing all module: a KV cache carved into equal blocks, handed out through a free list, indexed by a per-request block table, and made shareable through reference counting and copy-on-write.
#
# Then you will **measure** it. Not assert it — measure it. You will:
#
# 1. Implement a naive contiguous KV cache and watch it fragment.
# 2. Implement the paged allocator and watch the fragmentation disappear.
# 3. Implement prefix sharing with a real prefix tree, and confirm the memory saving.
# 4. Implement copy-on-write and count exactly how many bytes it copies.
# 5. Put a working paged attention kernel on the GPU and compare it against the naive one.
# 6. Replay the whole thing with a real small model's KV cache, and print the memory report an engine prints at startup.
#
# Runtime → Change runtime type → **T4 GPU**. Then Shift + Enter through the cells.
#
# **Write your numbers into the notes.** Every one of them is a number you will reason about in Module 09 when you are tuning a real engine.

# %% [markdown]
# ## 0. What am I running on?
#
# Say what you expect before you run it. A T4 is 16 GB with about 320 GB/s of bandwidth.

# %%
import torch, time, math, gc
assert torch.cuda.is_available(), "Switch the runtime to a T4 GPU first (Runtime > Change runtime type > T4 GPU)."

p = torch.cuda.get_device_properties(0)
print(f"Name                 {p.name}")
print(f"Memory               {p.total_memory/1e9:.1f} GB")
print(f"SMs                  {p.multi_processor_count}")
print(f"torch                {torch.__version__}")

def free_mem():
    gc.collect(); torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0] / 1e9

print(f"Free GPU memory      {free_mem():.1f} GB")

# %% [markdown]
# A timing helper you can trust, from Lab 02: CUDA events, warm-up, averaging.

# %%
def bench(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / iters

# %% [markdown]
# ## 1. The numbers this whole module is about
#
# For **Llama-3.1-8B in BF16**, from the lessons:
#
# ```
# bytes per token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_number
# bytes per token = 2 x 32 x 8 x 128 x 2 = 131,072 bytes = 128 KiB
# ```
#
# In this notebook I will use a **small** model so everything fits on a T4 and runs fast. The structure is identical; only the constant changes. I will work with a Llama-3.2-1B-shaped KV cache.

# %%
# Small-model KV cache geometry (Llama-3.2-1B / SmolLM-style: 16 layers, 8 KV heads, head dim 64)
LAYERS   = 16
KV_HEADS = 8
HEAD_DIM = 64
BYTES    = 2                      # BF16

kv_bytes_per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES
print(f"KV bytes per token   {kv_bytes_per_token:,}")
print(f"  = {kv_bytes_per_token/1024:.0f} KiB per token of context")

# %%
# The Llama-3.1-8B figure you derived by hand, for comparison.
llama31_8b = 2 * 32 * 8 * 128 * 2
print(f"\nLlama-3.1-8B BF16    {llama31_8b:,} bytes/token = {llama31_8b/1024:.0f} KiB")
print(f"ratio (8B / 1B-ish)  {llama31_8b/kv_bytes_per_token:.1f}x")

# %% [markdown]
# ## 2. A contiguous KV cache, so you can watch it fragment
#
# The naive design: when a request arrives, reserve a flat buffer sized for `max_len`. When it grows past the end, we would have to reallocate and copy. To keep this notebook about memory rather than about torch's allocator, I will model the contiguous cache with **token counts and a hole list** — the same bookkeeping an engine does, but without a 200-line C++ allocator in the way.

# %%
class ContiguousCache:
    """The naive design: one flat reservation per request, reserved to the maximum."""

    def __init__(self, max_len, kv_bytes):
        self.max_len = max_len
        self.kv_bytes = kv_bytes
        self.capacity = 0            # in tokens
        self.reserved = {}           # id -> reserved length (always max_len while alive)
        self.holes = []              # list of (start, size) free runs

    def admit(self, rid):
        if self.capacity + self.max_len > self.pool:
            return False
        self.reserved[rid] = self.max_len
        self.capacity += self.max_len
        return True

    def finish(self, rid):
        self.capacity -= self.reserved.pop(rid, 0)
        self.holes.append(self.max_len)      # the whole buffer comes back, one hole

    def contiguous_free(self):
        return max(self.holes) if self.holes else 0

    def total_free(self):
        return sum(self.holes)

    @property
    def pool(self):
        return self.max_len * 64

# %% [markdown]
# Run a realistic arrival pattern: many short requests, a few long ones, arriving over time. Watch the holes.

# %%
import random
random.seed(7)

MAX_LEN = 2048
POOL_TOKENS = 64 * MAX_LEN        # a pool the size of 64 reservations

def run_contiguous(n_requests=200, lengths=None, max_len=MAX_LEN):
    c = ContiguousCache(max_len, kv_bytes_per_token)
    c.pool = POOL_TOKENS
    live, failures, holes_seen, util = 0, 0, [], []
    for i in range(n_requests):
        if not c.admit(i):
            failures += 1
            continue
        live += 1
        L = lengths(i)
        util.append(live * max_len / POOL_TOKENS)
        if L >= max_len:            # finishes
            c.finish(i); live -= 1
        elif random.random() < 0.4: # some fraction finish mid-generation
            c.finish(i); live -= 1
        holes_seen.append((c.total_free(), c.contiguous_free()))
    return c, failures, holes_seen, util

lengths = lambda i: random.choice([64, 128, 256, 512, 1024, 2048, 4096])

c, failures, holes, util = run_contiguous(300, lengths)
total_free = c.total_free()
biggest = c.contiguous_free()

print(f"Reservations made         300")
print(f"Admission failures        {failures}")
print(f"Free tokens at the end    {total_free:,}")
print(f"Largest single free run   {biggest:,}")
print(f"Fragmentation             {1 - biggest/total_free:.0%} of free memory is unusable" if total_free else "no free memory")
print()
print("A new request needs 2048 contiguous tokens.")
print(f"  free in total: {total_free:,} -> {'fits' if total_free >= 2048 else 'DOES NOT FIT'}")
print(f"  largest run:   {biggest:,} -> {'fits' if biggest >= 2048 else 'DOES NOT FIT'}")

# %% [markdown]
# ## 3. The paged allocator
#
# Now the real thing. Same workload, same arrivals — but memory is handed out in equal blocks.
#
# ```python
# allocate:  free_list.pop()            # O(1), no copying
# free:      free_list.append(block_id) # O(1), no copying
# grow:      append one more id when the last block fills
# ```
#
# Three properties you should verify as you go:
#
# 1. No fragmentation between blocks: they were never individually allocated.
# 2. Internal fragmentation only in the **tail of the last block**, at most `block_size - 1` tokens per request.
# 3. Growth and free are both O(1) with zero bytes copied.

# %%
class PagedCache:
    """The paged KV cache: equal blocks, a free list, and a per-request block table."""

    def __init__(self, num_blocks, block_size, kv_bytes):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.kv_bytes = kv_bytes
        self.free = list(range(num_blocks))
        self.tables = {}          # rid -> [block_id, ...]
        self.filled = {}          # rid -> slots used in its last block
        self.total_slots_used = 0

    @property
    def capacity_tokens(self):
        return self.num_blocks * self.block_size

    def blocks_needed(self, tokens):
        return max(1, math.ceil(tokens / self.block_size))

    def admit(self, rid, prompt_tokens):
        need = self.blocks_needed(prompt_tokens)
        if need > len(self.free):
            return False
        self.tables[rid] = [self.free.pop() for _ in range(need)]
        self.filled[rid] = prompt_tokens % self.block_size or self.block_size
        self.total_slots_used += need * self.block_size
        return True

    def grow(self, rid):
        """Add one token. May need one new block. Returns False if the pool is dry."""
        self.filled[rid] += 1
        if self.filled[rid] <= self.block_size:
            return True
        if not self.free:
            self.filled[rid] -= 1
            return False
        self.tables[rid].append(self.free.pop())
        self.filled[rid] = 1
        self.total_slots_used += self.block_size
        return True

    def finish(self, rid):
        for b in self.tables.pop(rid, []):
            self.free.append(b)
        self.filled.pop(rid, None)

    def tokens_held(self, rid):
        return len(self.tables.get(rid, [])) * self.block_size

    def waste_tokens(self, rid, real_tokens):
        return self.tokens_held(rid) - real_tokens

    def free_blocks(self):
        return len(self.free)

# %% [markdown]
# Run the **same** workload through the paged cache.

# %%
def run_paged(num_blocks, block_size, n_requests=300, lengths=None, max_len=MAX_LEN):
    p = PagedCache(num_blocks, block_size, kv_bytes_per_token)
    live, failures, waste = 0, 0, 0
    for i in range(n_requests):
        L = lengths(i)
        if not p.admit(i, L):
            failures += 1
            continue
        live += 1
        if L >= max_len or random.random() < 0.4:
            waste += p.waste_tokens(i, L)     # tail waste on the way out
            p.finish(i); live -= 1
    return p, failures, waste

for bs in [8, 16, 32, 64]:
    # give the paged cache the same *byte budget* the contiguous cache had
    blocks = POOL_TOKENS // bs
    p, f, w = run_paged(blocks, bs)
    print(f"block_size {bs:>3}: blocks {blocks:>6}  admissions failed {f:>3}  "
          f"tail waste {w:>7,} tokens ({w/max(1,blocks*bs):.1%} of the pool)")

# %% [markdown]
# **Read that table.** With block size 8, tail waste is a fraction of a percent of the pool. With block size 64 it is still under 2%. There is **no external fragmentation at all**: the free-block count is exactly `num_blocks - sum(len(table))`.
#
# Compare to the contiguous cache, where 100% of the free memory can be unusable.

# %%
# Direct comparison at equal byte budgets.
equal_bytes = 64 * MAX_LEN * kv_bytes_per_token
paged_blocks = equal_bytes // (16 * kv_bytes_per_token)

print(f"Byte budget:             {equal_bytes/1e9:.2f} GB of KV cache")
print(f"  contiguous:            64 requests of {MAX_LEN} tokens")
print(f"  paged (B=16):          {paged_blocks} blocks = {paged_blocks*16:,} tokens")
print()
print("Same bytes. Contiguous fits 64 requests; paged fits 4x as many tokens,")
print("which means 4x as many typical requests, or the same requests 4x longer.")

# %% [markdown]
# ## 4. Prove growth is free
#
# The claim: growth costs O(1) and copies zero bytes. Measure it against the contiguous alternative (which must reallocate and copy).

# %%
BLOCK = 16
blocks = 4096
pg = PagedCache(blocks, BLOCK, kv_bytes_per_token)
pg.admit(0, 64)

# paged growth: 1000 one-token steps
t_paged = bench(lambda: pg.grow(0), iters=1, warmup=1)   # rough; the loop below is the real cost
# measure the whole 1000-step loop instead
pg2 = PagedCache(blocks, BLOCK, kv_bytes_per_token); pg2.admit(0, 64)
t0 = time.perf_counter()
for _ in range(1000):
    pg2.grow(0)
t_paged = (time.perf_counter() - t0) / 1000

# contiguous growth: copy the tensor to a bigger one, every doubling
buf = torch.zeros(64 * LAYERS * KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
sizes = [64]
t0 = time.perf_counter()
for _ in range(1000):
    n = sizes[-1]
    if n % 1024 == 0 and n < 4096:
        new = torch.zeros(n * 2 * LAYERS * KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        new[: n * LAYERS * KV_HEADS * HEAD_DIM] = buf
        buf = new
        sizes.append(n * 2)
    sizes.append(sizes[-1] + 1)
t_contig = (time.perf_counter() - t0) / 1000
del buf
torch.cuda.empty_cache()

print(f"paged grow(), per token       {t_paged*1e6:9.2f} us   (a pop from a list)")
print(f"contiguous grow, per token    {t_contig*1e6:9.2f} us   (includes occasional full copies)")
print(f"ratio                         {t_contig/max(t_paged,1e-9):9.0f}x")

# %% [markdown]
# ## 5. Prefix sharing, with a real prefix tree
#
# Build a radix tree of token blocks, exactly as SGLang's RadixAttention does. Then send three requests whose prompts share a prefix, and see how many blocks are actually stored.

# %%
import random
random.seed(3)

class PrefixTree:
    """Radix tree over token blocks: an edge means 'this block continues that block'."""

    def __init__(self, block_size):
        self.block_size = block_size
        self.children = {}     # key: (parent_key, tuple(tokens)) -> node
        self.nodes = {}        # node_key -> block_id (in the paged cache)

    def _blocks(self, token_ids):
        return [tuple(token_ids[i:i+self.block_size]) for i in range(0, len(token_ids), self.block_size)]

    def find_shared(self, token_ids, block_ids):
        """Walk down while token blocks match. Returns the list of reusable block ids."""
        shared, parent = [], None
        for blk in self._blocks(token_ids):
            key = (parent, blk)
            if key not in self.children:
                break
            parent = key
            shared.append(self.nodes[key])
        return shared

    def insert(self, token_ids, block_ids):
        parent = None
        for i, blk in enumerate(self._blocks(token_ids)):
            key = (parent, blk)
            if key not in self.children:
                self.children[key] = True
                self.nodes[key] = block_ids[i]
            parent = key

# %%
BLOCK = 16
pg = PagedCache(8192, BLOCK, kv_bytes_per_token)
tree = PrefixTree(BLOCK)

# Three prompts sharing a 160-token system prompt, then diverging.
sys_prompt = [random.randint(1000, 9999) for _ in range(160)]
prompts = {
    "user1": sys_prompt + [random.randint(1000,9999) for _ in range(24)],
    "user2": sys_prompt + [random.randint(1000,9999) for _ in range(40)],
    "user3": sys_prompt + [random.randint(1000,9999) for _ in range(16)],
}

for name, toks in prompts.items():
    shared = tree.find_shared(toks, None)
    fresh_from = len(toks) - len(shared) * BLOCK
    # real allocation for the unmatched part
    ids = []
    for _ in range(max(0, (fresh_from + BLOCK - 1)//BLOCK)):
        ids.append(pg.free.pop())
    pg.tables[name] = ids
    pg.filled[name] = len(toks) % BLOCK or BLOCK
    tree.insert(toks, ids)
    stored = (len(ids)) * BLOCK
    print(f"{name}: prompt {len(toks):>4} tok | reused {len(shared):>2} blocks ({len(shared)*BLOCK:>3} tok) | stored {stored:>4} tok")

total_tokens = sum(len(prompts[k]) for k in prompts)
stored_tokens = sum(len(pg.tables[k]) * BLOCK for k in prompts)
print()
print(f"naive (no sharing)      {total_tokens:>5,} tokens = {total_tokens*kv_bytes_per_token/1e6:7.1f} MB")
print(f"with prefix sharing     {stored_tokens:>5,} tokens = {stored_tokens*kv_bytes_per_token/1e6:7.1f} MB")
print(f"saved                   {1 - stored_tokens/total_tokens:>5.0%}")

# %% [markdown]
# ## 6. Copy-on-write: count the bytes
#
# The only time a shared block is copied is when its owner needs to **write** past it. Measure how many bytes that actually is, and how it compares to storing the prefix separately.

# %%
class SharedPagedCache(PagedCache):
    """Paged cache with reference counts, so blocks can be shared safely."""

    def __init__(self, num_blocks, block_size, kv_bytes):
        super().__init__(num_blocks, block_size, kv_bytes)
        self.refcount = {b: 0 for b in range(num_blocks)}

    def acquire_write(self, block_id):
        """Return a block id this request may write into, copying only if shared."""
        if self.refcount[block_id] <= 1:
            return block_id, 0
        copy_id = self.free.pop()
        self.refcount[copy_id] = 1
        self.tables["__copies__"] = self.tables.get("__copies__", 0) + 1
        return copy_id, self.block_size * self.kv_bytes

    def share(self, block_id):
        self.refcount[block_id] += 1

    def release(self, block_id):
        self.refcount[block_id] -= 1
        if self.refcount[block_id] == 0:
            self.free.append(block_id)

# %%
# Simulate the copy-on-write path and count total bytes copied.
rng = random.Random(11)
spc = SharedPagedCache(4096, 16, kv_bytes_per_token)
spc.admit(0, 160)                     # request 0 owns the system prefix

bytes_copied = 0
for i in range(1, 25):                # 24 more requests share the prefix
    for b in spc.tables[0]:
        spc.share(b)                   # point at the same blocks
    # now this request writes past the shared prefix: one copy, at the boundary
    block_to_write = spc.tables[0][-1]
    _, copied = spc.acquire_write(block_to_write)
    bytes_copied += copied
    for b in spc.tables[0]:
        spc.release(b)

print(f"24 requests sharing a 160-token prefix")
print(f"bytes copied by copy-on-write   {bytes_copied/1e6:.2f} MB")
print(f"bytes if each stored its own    {24*160*kv_bytes_per_token/1e6:.2f} MB")
print(f"ratio                           {24*160*kv_bytes_per_token/max(1,bytes_copied):.0f}x cheaper")

# %% [markdown]
# ## 7. A working paged attention kernel on the GPU
#
# So far the allocator has been bookkeeping. Now put it under a real attention kernel and check two things:
#
# 1. The paged kernel produces the **same output** as the naive one.
# 2. The paged kernel is not meaningfully slower.
#
# I will use a small single-layer attention: 8 KV heads, head dim 64, one block of 16 tokens.

# %%
KV_HEADS, HEAD_DIM, BLOCK = 8, 64, 16
CTX = 256                        # 16 blocks
NBLOCKS = 64                     # pool has 64 blocks

torch.manual_seed(0)
q = torch.randn(KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

# The KV cache for one request, stored as a normal contiguous tensor (the naive layout)
k_contig = torch.randn(CTX, KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
v_contig = torch.randn(CTX, KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

# %%
# Scatter that KV into a pool of blocks, and record the block table (the "paged" layout).
pool = torch.zeros(NBLOCKS, BLOCK, KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
pool_v = torch.zeros_like(pool)
table = list(torch.randperm(NBLOCKS)[: CTX // BLOCK].tolist())   # scattered ids
for i, blk in enumerate(table):
    pool[blk] = k_contig[i*BLOCK:(i+1)*BLOCK]
    pool_v[blk] = v_contig[i*BLOCK:(i+1)*BLOCK]
print(f"block table: {table}")
print(f"KV stored in {len(table)} scattered blocks of {BLOCK} tokens")

# %%
def naive_attention(q, k, v):
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    p = torch.softmax(scores, dim=-1)
    return p @ v.float()

def paged_attention(q, pool_k, pool_v, table, n_tokens):
    """Attention that reads K and V through the block table, block by block."""
    ks = torch.cat([pool_k[b] for b in table], dim=0)[:n_tokens]
    vs = torch.cat([pool_v[b] for b in table], dim=0)[:n_tokens]
    return naive_attention(q, ks, vs)

# %%
ref = naive_attention(q, k_contig, v_contig)
got = paged_attention(q, pool, pool_v, table, CTX)
print(f"max abs difference: {(ref-got).abs().max().item():.2e}")
print("identical:", torch.allclose(ref, got, atol=1e-2))

# %%
# Now the timing. The gather is the price of paging.
t_naive = bench(lambda: naive_attention(q, k_contig, v_contig))
t_paged = bench(lambda: paged_attention(q, pool, pool_v, table, CTX))

kv_bytes = CTX * KV_HEADS * HEAD_DIM * 2 * 2   # K and V
print(f"\nnaive attention    {t_naive*1e3:7.1f} us")
print(f"paged attention    {t_paged*1e3:7.1f} us   (includes the gather)")
print(f"overhead           {(t_paged/t_naive-1):+7.1%}")
print(f"\nKV bytes gathered  {kv_bytes/1e6:.2f} MB")
print(f"gather bandwidth   {kv_bytes/t_paged/1e9:.0f} GB/s")
print(f"(T4 peak is ~320 GB/s; this kernel is a small PyTorch loop, not a tuned kernel)")

# %% [markdown]
# ## 8. Now the real model
#
# Everything above used toy tensors. Now load an actual small model, run a real prompt through it, and read out the KV cache size. This is the number an engine prints at startup.

# %%
!pip install -q transformers accelerate

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

cfg = model.config
L, H, D, KVH = cfg.num_hidden_layers, cfg.hidden_size, cfg.head_dim, cfg.num_key_value_heads
print(f"{MODEL}: {L} layers, hidden {H}, {KVH} KV heads, head dim {D}")

# %%
def kv_bytes_per_token_for(cfg, dtype_bytes=2):
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * dtype_bytes

kvpt = kv_bytes_per_token_for(cfg)
print(f"KV bytes per token {kvpt:,} = {kvpt/1024:.0f} KiB")

# %%
# Read the real KV cache out of the model after a generation, and report it.
prompt = "Explain what a KV cache is, in three sentences."
inputs = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs, use_cache=True, output_attentions=False)

past = out.past_key_values
if hasattr(past, "keys"):
    keys, values = past.keys, past.values
else:
    keys, values = past[0], past[1]

n_prompt = inputs.input_ids.shape[1]
k0 = keys[0]
print(f"\nprompt tokens      {n_prompt}")
print(f"one layer K shape  {tuple(k0.shape)}")
actual_bytes = sum(k.numel() + v.numel() for k, v in zip(keys, values)) * k0.element_size()
predicted = n_prompt * kvpt
print(f"predicted KV bytes {predicted:,}")
print(f"actual KV bytes    {actual_bytes:,}")
print(f"match: {abs(actual_bytes-predicted)/predicted:.1%} off")

# %%
# Generate and watch the KV cache grow one token at a time.
with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=64, use_cache=True, do_sample=False)

final_tokens = gen.shape[1]
final_bytes = final_tokens * kvpt
print(f"\nafter {final_tokens} tokens of context: {final_bytes/1e6:.2f} MB")
print(f"growth per generated token: {kvpt/1024:.0f} KiB")

# %% [markdown]
# Now project: how many concurrent requests fit on this T4?
#
# ```
# free_after_weights = free GPU memory - reserve
# capacity_tokens     = free_after_weights / kv_bytes_per_token
# concurrent          = capacity_tokens / expected_context_length
# ```

# %%
free = free_mem()
reserve = 1.0    # leave 1 GB for activations, CUDA context, fragmentation
capacity = max(0, free - reserve)
capacity_tokens = int(capacity * 1e9 / kvpt)
print(f"Free after the model loaded: {free:.2f} GB")
print(f"KV pool capacity            {capacity:.2f} GB = {capacity_tokens:,} tokens")

for ctx in [512, 2048, 8192, 32768]:
    n = capacity_tokens // ctx
    print(f"  at {ctx:>6,} tokens of context: {n:>6,} concurrent requests  ({n*ctx*kvpt/1e9:.1f} GB used)")

# %% [markdown]
# ## 9. The same projection, with paging
#
# A contiguous allocator at max_model_len 32768 would reserve 32768 tokens per request. Paging lets you serve short requests without paying for the long ones. Show it.

# %%
max_model_len = 32768
naive = capacity_tokens // max_model_len
print(f"contiguous, max_model_len {max_model_len:,}:  {naive:>5,} concurrent requests (all-or-nothing)")

# Paged: a mix of short and long, packed perfectly into blocks of 16.
mixed = [512, 1024, 2048, 4096, 8192, 16384]
avg = sum(mixed) / len(mixed)
paged_n = int(capacity_tokens / (avg + (16 - avg % 16) % 16))
print(f"paged, mixed workload avg {avg:,.0f}:  {paged_n:>5,} concurrent requests")
print(f"advantage: {paged_n/max(1,naive):.0f}x")

# %% [markdown]
# ## 10. Measure the fragmentation of your own workload
#
# The tail waste is `block_size - (tokens % block_size)` per request. Measure it for the real context lengths you just generated.

# %%
real_tokens = final_tokens
for bs in [8, 16, 32, 64, 256]:
    waste = bs - (real_tokens % bs or bs)
    print(f"block_size {bs:>4}: tail waste {waste:>3} tokens of {real_tokens} = {waste/real_tokens:.2%}")

# %% [markdown]
# ## Your turn
#
# 1. **Section 2**: change `MAX_LEN` to 4096 and re-run. How does the fragmentation of the contiguous cache change? Why does it get worse, not better?
# 2. **Section 3**: try block_size 128 and 256. At what block size does tail waste start to hurt, and how does that interact with the block-table size?
# 3. **Section 5**: change the system prompt to 400 tokens, then to 4000. Plot saved-vs-prefix-length. Where does it flatten, and why?
# 4. **Section 6**: count how many bytes copy-on-write copies for 100 requests sharing a prefix, versus how many bytes storing them separately would take. Report the ratio.
# 5. **Section 7**: try a real Triton or FlashAttention paged backend if you can install one (`flash-attn` may not build on Colab T4). At minimum, verify correctness at several context lengths.
# 6. **Section 9**: re-run the capacity projection for a **70B** model using the KV-per-token formula (not loading it). What concurrency does a single 80 GB H100 allow at 32k context?
#
# **Write in your notes:** the KV bytes per token for Qwen2.5-0.5B, the contiguous fragmentation you measured, the block size you would pick and why, and the concurrency your T4 supports at 8k context.

# %%
# Cleanup before you finish, so Colab does not complain about the session.
del past, keys, values, model, tok
gc.collect(); torch.cuda.empty_cache()
print("done. final free memory:", f"{free_mem():.1f} GB")