import type { GlossaryEntry } from '../glossary';

// Terms Module 20 introduces. Keys already in src/data/glossary.ts and the
// per-module files (kv-cache, paged-attention, quantization, prefill-prefix,
// attention-optimization, vllm, ...) are not repeated here — only new keys.
export const terms: Record<string, GlossaryEntry> = {
  'lost-in-the-middle': {
    term: 'Lost in the middle',
    def: 'Retrieval accuracy against a long context is roughly U-shaped in position: the model uses the beginning and the end of its context well and the middle worst. Not a cache bug and not an engine bug — a property of attention and training. The cheapest fix is placement: put the important material at both ends.',
    lesson: 'long-context/long-context-scaling',
  },
  'context-cliff': {
    term: 'Context cliff',
    def: 'The capacity edge where one more token of context costs you a whole conversation. Conversations per card is (M − W − S)·0.9 ÷ (T·b_kv), so the count is a step function of T: 12 conversations at 32K, 3 at 128K, and then zero. Shipping a max-model-len means choosing a point on this cliff.',
    lesson: 'long-context/long-context-scaling',
  },
  'prefill-wall': {
    term: 'Prefill compute wall',
    def: 'The context length where prefill’s quadratic attention term overtakes its projection term: T = P/(2·d·L). About 30,600 tokens for Llama-3.1-8B, 53,900 for 70B, 11,400 for Qwen2.5-0.5B. Bigger models push it out; smaller models pull it in.',
    lesson: 'long-context/long-context-scaling',
  },
  'decode-bandwidth-wall': {
    term: 'Decode bandwidth wall',
    def: 'The context length where the per-sequence KV read overtakes the weight read in a decode step: T = P·b ÷ b_kv, about 122,500 tokens for Llama-3.1-8B in BF16. A bandwidth crossover, not a memory one, and the most dangerous of the three walls because it fails silently and taxes every generated token.',
    lesson: 'long-context/long-context-scaling',
  },
  'attention-mass': {
    term: 'Attention mass',
    def: 'How much of a token’s attention weight falls on nearby tokens versus the whole context. Strongly concentrated near the end of the context in practice, which is why a resident window of the most recent few thousand tokens covers most of what attention reads — and why streaming only the tail works.',
    lesson: 'long-context/tiered-kv',
  },
  'derived-data': {
    term: 'Derived data',
    def: 'Anything that can be regenerated from its source instead of stored. The KV cache is the one piece of a transformer that qualifies: every byte comes from prefill, so it can be recomputed. That property is what allows it to live on a cheap tier, and what makes recompute a legitimate substitute for a fetch.',
    lesson: 'long-context/tiered-kv',
  },
  'tiered-kv': {
    term: 'Tiered KV',
    def: 'Keeping the KV cache across several storage tiers — GPU HBM, host DRAM, NVMe, a remote node — and fetching from the appropriate one per step, instead of requiring the whole cache to be resident on the GPU. Buys capacity rather than speed.',
    lesson: 'long-context/tiered-kv',
  },
  'kv-offload': {
    term: 'KV offload',
    def: 'Residing part or all of the KV cache in host memory and gathering it to the GPU on demand. vLLM exposes this through its kv transfer config and cpu_offload_gb; SGLang through its hierarchical cache. The gain is memory, the price is PCIe bandwidth and a synchronization hazard.',
    lesson: 'long-context/tiered-kv',
  },
  'recompute-crossover': {
    term: 'Fetch-vs-recompute crossover',
    def: 'The storage bandwidth at which fetching a KV block costs the same as recomputing it: b_kv ÷ t_recompute, about 4.4 GB/s for Llama-3.1-8B on an H100 and 1.3 GB/s for 70B. Below it, recompute — the tensor cores are idle during a memory-bound decode step. Bigger models tolerate slower storage.',
    lesson: 'long-context/tiered-kv',
  },
  'resident-window': {
    term: 'Resident window',
    def: 'The most recent R tokens of KV kept permanently in GPU memory while the rest streams from a slower tier. Sized from the latency knee: R plus the bytes receivable in one step. The split that makes tiered KV fast enough to use.',
    lesson: 'long-context/tiered-kv',
  },
  'streaming-budget': {
    term: 'Streaming budget',
    def: 'The bytes a tier can deliver in one decode step without adding latency: tier bandwidth × the time the step already spends on weights. 330 MB per step over PCIe on an H100. Beyond it the fetch appears as latency, so it is the number every offload design is sized against.',
    lesson: 'long-context/tiered-kv',
  },
  'prefetch-depth': {
    term: 'Prefetch depth',
    def: 'How many steps ahead a tiered cache fetches blocks it will need. Depth 1 hides a fetch behind one step; deeper hides more of the latency variance. A miss at depth 0 is a visible stall, which is why every real implementation prefetches at least one step further than it thinks it needs.',
    lesson: 'long-context/tiered-kv',
  },
  'double-buffering': {
    term: 'Double buffering',
    def: 'Using two host or device buffers alternately so the next step’s writes never overwrite the buffer the current step is still reading. The fix for the most dangerous silent failure in offloaded KV — a transfer/compute race that yields wrong numbers with no crash.',
    lesson: 'long-context/tiered-kv',
  },
  'tier-backpressure': {
    term: 'Tier backpressure',
    def: 'Bounding how many sequences may be concurrently served from a slow tier. A remote cache with seconds of latency is fine for throughput and catastrophic for memory: every blocked request pins its GPU blocks for the whole round trip, so a traffic spike becomes an outage without admission control.',
    lesson: 'long-context/tiered-kv',
  },
  'hicache': {
    term: 'Hierarchical cache',
    def: 'SGLang’s multi-level KV cache — GPU, host, and optionally a third backing tier — with an LFU-style eviction policy that protects actively used layers. At the time of writing the central name for a cache with more than one level.',
    lesson: 'long-context/tiered-kv',
  },
  'lmcache': {
    term: 'LMCache',
    def: 'An open-source KV cache layer that runs beside the model server and stores KV blocks in host RAM, local disk or a remote store, handing them back when a request can reuse them. Blocks are keyed by their tokens, so reuse crosses requests and, with a shared store, crosses engine instances — which is what separates it from a prefix cache that dies with its process.',
    lesson: 'long-context/tiered-kv',
  },
  'kv-connector': {
    term: 'KV connector',
    def: 'The plug-in interface a serving engine uses to hand KV blocks to an external store and ask for them back. In vLLM it is named in a kv-transfer-config JSON at startup together with a kv_role — store only, load only, or both. The connector is the boundary between the engine’s paged pool and whatever tier is behind it.',
    lesson: 'long-context/tiered-kv',
  },
  'achieved-bandwidth': {
    term: 'Achieved bandwidth',
    def: 'The bytes per second your code actually moves, which is what a capacity plan needs — not the tier’s spec sheet. Measure it in your own process, with pinned host memory and a proper warmup, because pageable copies and cold first touches routinely cost 3–10×.',
    lesson: 'long-context/lab-long-context',
  },
  'streamed-tokens-per-step': {
    term: 'Streamed tokens per step',
    def: 'The number of context tokens a tiered cache must fetch for every decode step — the converted form of the streaming budget, and the figure to alert on in production. It rises silently with context length and is invisible in every latency metric until it is too late.',
    lesson: 'long-context/lab-long-context',
  },
};
