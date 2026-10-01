import type { GlossaryEntry } from '../glossary';

// Terms Module 05 introduces. Keys already in src/data/glossary.ts (kv-cache, prefill,
// decode, autoregressive, oom, context-length, gqa, rope, batch, ...) are not repeated here.
export const terms: Record<string, GlossaryEntry> = {
  'token-forward': {
    term: 'Token-forward',
    def: 'One token pushed through the whole model once: 2 × parameters FLOPs. 16.1 GFLOP for Llama-3.1-8B. The natural unit of work for counting inference cost.',
    lesson: 'kv-cache/recompute-waste',
  },
  'naive-recompute': {
    term: 'Naive recompute',
    def: 'Generating without a KV cache, by re-running the whole forward pass over the whole sequence at every step. Does n·p + n(n−1)/2 token-forwards where p + n would do, so the work is quadratic in the output length.',
    lesson: 'kv-cache/recompute-waste',
  },
  'waste-currency': {
    term: 'Waste only costs what it is made of',
    def: 'Why 99% wasted FLOPs can be only 2× wasted time: a memory-bound step has idle tensor cores, so extra arithmetic is nearly free. Optimizations are only worth what they save in the scarce resource.',
    lesson: 'kv-cache/recompute-waste',
  },
  'kv-bytes-per-token': {
    term: 'KV bytes per token',
    def: 'The memory one token of context costs in a KV cache: 2 × layers × key/value heads × head dimension × bytes per number. 128 KB for Llama-3.1-8B in BF16, 12 KB for Qwen2.5-0.5B.',
    lesson: 'kv-cache/kv-cache',
  },
  'kv-write-read-asymmetry': {
    term: 'Write O(1), read O(t)',
    def: 'A decode step writes exactly one cache row (constant cost) and reads every row that exists (cost grows with context). This asymmetry is why decode attention kernels look nothing like prefill attention kernels.',
    lesson: 'kv-cache/kv-cache',
  },
  'rope-caching': {
    term: 'Rotating before caching',
    def: 'Keys must be rotated by their rotary position embedding before being written to the cache. The rotation is the only place position enters, and it needs the vector to still be attached to its input, so caching the unrotated projection forces recomputation.',
    lesson: 'kv-cache/kv-cache',
  },
  'position-baked-keys': {
    term: 'Position-baked keys',
    def: 'A cached key is only valid at the position it was created for. Cache rows cannot be reordered, re-positioned or inserted into, which constrains every paged or shared-cache design.',
    lesson: 'kv-cache/kv-cache',
  },
  'decode-step': {
    term: 'Decode step',
    def: 'One forward pass producing one token from one sequence. Reads all the weights plus the whole KV cache, so it is memory-bound: about 1 FLOP per byte.',
    lesson: 'kv-cache/prefill-vs-decode',
  },
  'prefill-decode-crossover': {
    term: 'Prefill/decode crossover',
    def: 'The context length where prefill attention costs as much FLOPs as all the weights: P / (2 · hidden · layers). About 30,600 tokens for Llama-3.1-8B, 54,000 for 70B. Past it, prefill is dominated by quadratic arithmetic.',
    lesson: 'kv-cache/prefill-vs-decode',
  },
  'context-tokens-per-gpu': {
    term: 'Context tokens per GPU',
    def: '(GPU memory − weights − workspace) × 0.9 ÷ KV bytes per token. The batch-independent capacity of a card: every conversation-count question is this number divided by a context length. 417,000 for Llama-3.1-8B on an 80 GB card.',
    lesson: 'kv-cache/memory-math',
  },
  'memory-limited-batch': {
    term: 'Memory-limited batch size',
    def: 'The largest batch that fits at a given context length. It is the practical reason a decode step slows down as you add users, because every extra conversation adds cache bytes to every step.',
    lesson: 'kv-cache/memory-math',
  },
  'workspace': {
    term: 'Workspace / activation reserve',
    def: 'GPU memory set aside for the CUDA context, activations, the logits tensor and kernel scratch. Budget about 4% of the card plus 10% headroom, and measure it rather than trusting the guess.',
    lesson: 'kv-cache/memory-math',
  },
  'shared-prefix': {
    term: 'Shared prefix',
    def: 'Rows of the KV cache that several requests can share, such as a common system prompt. Storing a 2,000-token prefix once instead of ten times saves 256 MB per request on Llama-3.1-8B. The basis of prefix caching (Module 10).',
    lesson: 'kv-cache/memory-math',
  },
  'preallocated-cache': {
    term: 'Preallocated cache',
    def: 'A KV cache tensor allocated once at max length and filled one row at a time. Its weakness: most of it is empty — reserving 8,192 rows and using 1,024 wastes 87% of it per request, which is why paged attention exists.',
    lesson: 'kv-cache/build-kv-cache',
  },
  'cat-trap': {
    term: 'The cat trap',
    def: 'Appending to a KV cache with torch.cat or torch.stack. It returns the right tensor but copies the entire history every step: 1 + 2 + … + t rows, so 524,800 rows for a 1,024-token request. Makes the memory traffic quadratic and cancels the benefit of caching.',
    lesson: 'kv-cache/build-kv-cache',
  },
  'per-row-length': {
    term: 'Per-row length',
    def: 'In a batched KV cache, how many rows are valid in each batch row, rather than one shared cursor. Sequences in a batch reach different lengths, so attention must mask above each row’s own length — which is why batch memory scales with the longest sequence, not the average.',
    lesson: 'kv-cache/build-kv-cache',
  },
  'cache-equivalence-test': {
    term: 'Cache equivalence test',
    def: 'Run the same tokens with and without the cache and compare the logits. They should agree to floating-point noise (~10⁻⁶ in FP32), and the difference is summation order, not logic. Run it at two context lengths: short catches logic errors, long catches layout errors.',
    lesson: 'kv-cache/build-kv-cache',
  },
  'logits-tensor': {
    term: 'Logits tensor',
    def: 'The scores the model produces for every position against the whole vocabulary: prompt_length × vocab_size × bytes. Often larger than the KV cache and the real reason a long prompt runs out of memory.',
    lesson: 'kv-cache/lab-kv-cache',
  },
};
