import type { GlossaryEntry } from '../glossary';

// Terms this module introduces. Everything from the shared glossary (vllm,
// continuous-batching, paged-attention, kv-cache, prefill, decode, engine-core,
// model-runner, sampler, detokenizer, ttft, tpot, ...) is intentionally not repeated.

export const terms: Record<string, GlossaryEntry> = {
  'engine-core': {
    term: 'Engine core',
    def: 'The part of a serving engine that owns the GPU: the scheduler, the model runner, the KV cache manager and the sampler. It never parses HTTP. Everything that can be moved out of it should be, because Python on the critical path is latency.',
    lesson: 'mini-engine/read-vllm',
  },
  'model-runner': {
    term: 'Model runner',
    def: 'The component that actually runs the forward pass: it owns the weights and the KV cache tensors, builds the batch, launches the kernels, and writes the new KV entries back. Roughly 200 lines of real code, and the only part of the engine that touches the weights.',
    lesson: 'mini-engine/runner-and-kv',
  },
  'forward-pass': {
    term: 'Forward pass',
    def: 'One trip through every layer of the model. Called with a batch of sequences during decode (one token each) or with the whole prompt during prefill (many tokens for one sequence).',
    lesson: 'mini-engine/runner-and-kv',
  },
  'kv-allocator': {
    term: 'KV block allocator',
    def: 'The object that owns the KV memory: it hands out blocks of 16 tokens, tracks which blocks belong to which sequence, frees them when a request finishes, and decides what to evict or recompute when the pool runs dry.',
    lesson: 'mini-engine/paged-prefix-engine',
  },
  'block-table': {
    term: 'Block table',
    def: 'The per-request indirection array. Row i lists which physical KV blocks hold sequence i’s tokens, so attention can gather a scattered sequence into one logical row without any copying.',
    lesson: 'mini-engine/paged-prefix-engine',
  },
  'page-table': {
    term: 'Page table',
    def: 'The reverse index: which sequence owns each physical block. This is what makes freeing O(1) — you release a sequence\'s blocks by reading its row, not by searching the pool.',
    lesson: 'mini-engine/paged-prefix-engine',
  },
  'token-block': {
    term: 'Block size (tokens per block)',
    def: 'How many tokens of KV cache one physical block holds. vLLM\'s default is 16. Small blocks waste less on a short final chunk and share prefixes more finely; large blocks make the gather cheaper. Everything else about paged KV follows from this one number.',
    lesson: 'mini-engine/paged-prefix-engine',
  },
  'hash-prefix': {
    term: 'Prefix hashing',
    def: 'Hashing each block of tokens together with its parent\'s hash, so a block\'s identity encodes its entire prefix. Two requests that share a prefix automatically produce identical block hashes and can share memory with no explicit matching.',
    lesson: 'mini-engine/paged-prefix-engine',
  },
  'preemption': {
    term: 'Preemption',
    def: 'Evicting a running sequence when KV memory is needed, either by swapping it out to host memory or by recomputing its KV later from the tokens. The alternative to an OOM, at the cost of latency for whoever is preempted.',
    lesson: 'mini-engine/paged-prefix-engine',
  },
  'recompute': {
    term: 'Recompute preemption',
    def: 'Freeing a sequence\'s KV blocks and planning to rebuild them from its tokens if it is scheduled again. Trades memory for FLOPs, which is why it is often cheaper than swapping when the model is compute-rich and memory-poor.',
    lesson: 'mini-engine/paged-prefix-engine',
  },
  'scheduler': {
    term: 'Scheduler',
    def: 'The loop that decides, every step, which sequences are in the batch. It owns admission, eviction, preemption and the token budget. In a modern engine it is the single most important component for latency.',
    lesson: 'mini-engine/continuous-batching-engine',
  },
  'waiting-queue': {
    term: 'Waiting queue',
    def: 'Requests admitted by the API server that have not yet had their prefill computed. The scheduler\'s first job is to drain it; its length is most of what TTFT is made of.',
    lesson: 'mini-engine/continuous-batching-engine',
  },
  'running-batch': {
    term: 'Running batch',
    def: 'Sequences that have been prefilled and are now generating one token per step. Every one of them contributes exactly one token per forward pass, and its whole KV cache is read each step.',
    lesson: 'mini-engine/continuous-batching-engine',
  },
  'token-budget': {
    term: 'Token budget',
    def: 'The cap on how many tokens one engine iteration may process across all sequences (vLLM: `--max-num-batched-tokens`). It bounds how long a prefill can stall everyone\'s decode, and it is why chunked prefill exists.',
    lesson: 'mini-engine/continuous-batching-engine',
  },
  'admission-control-cap': {
    term: 'Max sequences (max_num_seqs)',
    def: 'The cap on how many sequences can be in the running batch at once. It bounds per-step latency; a larger cap buys throughput and costs every user\'s inter-token latency.',
    lesson: 'mini-engine/continuous-batching-engine',
  },
  'iteration': {
    term: 'Engine iteration',
    def: 'One full cycle of the scheduler: choose a batch, run one forward pass, sample, update state. Continuous batching means this cycle never has a fixed membership — it is re-formed every time.',
    lesson: 'mini-engine/continuous-batching-engine',
  },
  'detokenizer': {
    term: 'Detokenizer',
    def: 'The component that converts streamed token IDs back into text incrementally, handling the fact that a token is often part of a word, a multi-byte character or a byte fragment.',
    lesson: 'mini-engine/api-metrics-engine',
  },
  sampler: {
    term: 'Sampler',
    def: 'Turns one row of logits into one token id: temperature, top-k, top-p, and a seeded random draw. It runs once per sequence per step, so it is pure Python or a tiny kernel — and a seeded RNG is what makes an engine benchmarkable against itself.',
    lesson: 'mini-engine/runner-and-kv',
  },
  'stream-interval': {
    term: 'Stream interval',
    def: 'How often the engine pushes tokens out to HTTP clients. Tying it to engine iterations is natural and is what most engines do; decoupling it with a background queue is a small change that decouples the network from the GPU loop.',
    lesson: 'mini-engine/api-metrics-engine',
  },
  'scheduler-stats': {
    term: 'Scheduler stats',
    def: 'Per-iteration counters an engine should export every step: running and waiting counts, tokens in the batch, how many were preempted, and the KV cache usage percentage. They are the raw material of every dashboard.',
    lesson: 'mini-engine/api-metrics-engine',
  },
  'working-set': {
    term: 'Working set',
    def: 'The subset of weights that must be resident for a batch to run — one copy per replica, or more under tensor parallelism. Its size sets the memory floor no scheduling trick can go below.',
    lesson: 'mini-engine/runner-and-kv',
  },
  'matmul-batching': {
    term: 'Decode as a batched GEMV',
    def: 'Reading the model\'s per-token weights as one tall skinny matrix multiply, [tokens × d_model] · [d_model × N], so a whole step is a handful of big GEMMs instead of one tiny one per weight.',
    lesson: 'mini-engine/runner-and-kv',
  },
  'paradigm-gap': {
    term: 'Framework-to-engine gap',
    def: 'The distance between what a framework tutorial shows (one request, one batch, generate() in a loop) and what a serving engine does (many requests, one batch, a scheduler and a memory manager). Reading a real codebase is mostly learning what fills that gap.',
    lesson: 'mini-engine/read-vllm',
  },
  'engine-abstraction': {
    term: 'Engine abstraction',
    def: 'The small set of concepts a serving engine exposes — a sequence, a scheduler, a block pool, a sampler, a stream — that a scheduling algorithm can be written against without knowing anything about the model.',
    lesson: 'mini-engine/read-vllm',
  },
  'reference-implementation': {
    term: 'Reference implementation',
    def: 'A small, readable, correct version of a system built to be understood and extended rather than to be fast. Its job is to make the invariants visible; performance comes later.',
    lesson: 'mini-engine/read-vllm',
  },
  'event-loop': {
    term: 'Event loop (asyncio)',
    def: 'The single thread an async web server uses to juggle many connections by never blocking. A blocking call inside it (such as a GPU forward pass) freezes every connection at once, which is why the engine loop runs in its own thread or process.',
    lesson: 'mini-engine/api-metrics-engine',
  },
  'call-soon-threadsafe': {
    term: 'loop.call_soon_threadsafe',
    def: 'The asyncio method another thread uses to ask an event loop to run a callback, such as putting a token on an asyncio.Queue. asyncio objects are not thread-safe, so this is the correct bridge from a synchronous batching thread back to async request handlers.',
    lesson: 'mini-engine/api-metrics-engine',
  },
  'batching-window': {
    term: 'Batching window',
    def: 'How long a worker waits for more requests after the first one arrives before running a batch. At arrival rate λ a window of w seconds collects about 1 + λw requests; the wait itself lands entirely in time to first token.',
    lesson: 'mini-engine/api-metrics-engine',
  },
};
