import type { GlossaryEntry } from '../glossary';

// New vocabulary introduced in Module 14. Terms the learner already met in
// earlier modules (attention, softmax, kv-cache, gqa, sram, bandwidth, ...)
// live in src/data/glossary.ts.
export const terms: Record<string, GlossaryEntry> = {
  'score-matrix': {
    term: 'Score matrix',
    def: 'The T x T tensor attention builds between queries and keys, where each entry is one dot product divided by the square root of the head dimension. It is the tensor naive attention writes to memory and fused attention never creates.',
    lesson: 'attention-optimization/online-softmax',
  },
  'intermediate-tensor': {
    term: 'Intermediate tensor',
    def: 'A value computed by one operation and consumed by the next, which is materialised in memory rather than kept on chip. In attention, the score and probability matrices are the expensive example: quadratic in context length and touched four times.',
    lesson: 'attention-optimization/online-softmax',
  },
  'online-softmax': {
    term: 'Online softmax',
    def: 'A row-wise softmax computed in one streaming pass by carrying three small vectors — a running maximum, a running sum, and a running output — instead of needing the whole row. Exact, not approximate, because the merge rule is associative.',
    lesson: 'attention-optimization/online-softmax',
  },
  'sufficient-statistic': {
    term: 'Sufficient statistic',
    def: 'A compact summary that loses no information needed later: however the data arrived, the pair (maximum, shifted sum) fully determines the softmax denominator. This is what turns a global reduction into a streaming one.',
    lesson: 'attention-optimization/online-softmax',
  },
  'io-aware-fusion': {
    term: 'IO-aware fusion',
    def: 'Restructuring computation so intermediates stay in fast on-chip memory instead of making the round trip to main memory. It attacks four problems: redundant traffic, materialised intermediates, low occupancy, and kernel launch overhead.',
    lesson: 'attention-optimization/online-softmax',
  },
  'register-pressure': {
    term: 'Register pressure',
    def: 'How many on-chip registers a kernel’s live values need. Too many and fewer blocks fit on each multiprocessor, so occupancy falls and latency hiding suffers. It is why fused attention tile sizes are tuned per GPU.',
    lesson: 'attention-optimization/online-softmax',
  },
  flashattention: {
    term: 'FlashAttention',
    def: 'An exact attention algorithm that tiles the queries and keys, computes each score tile in on-chip SRAM and folds it into a running softmax, never writing the T x T intermediate. Memory becomes O(T) instead of O(T²).',
    lesson: 'attention-optimization/flash-attention',
  },
  'flash-attention-fwd': {
    term: 'Forward pass',
    def: 'Computing the attention output from queries, keys and values. In inference this is all you need; the backward pass is only used for training and gradient-based methods.',
    lesson: 'attention-optimization/flash-attention',
  },
  'flash-attention-bwd': {
    term: 'Backward pass',
    def: 'Computing gradients through attention. FlashAttention does not store the score matrix for it; it recomputes each tile from the saved queries, keys, values and the two small state vectors.',
    lesson: 'attention-optimization/flash-attention',
  },
  'logsumexp': {
    term: 'Logsumexp',
    def: 'The quantity log(sum of exponentials) — what a safe softmax really computes. Saved per query row, it is the only extra state a FlashAttention backward pass needs beyond Q, K and V.',
    lesson: 'attention-optimization/flash-attention',
  },
  'arithmetic-intensity-fused': {
    term: 'Fused arithmetic intensity',
    def: 'For a fused attention kernel, FLOPs per byte of real traffic. Bytes are O(T) while FLOPs are O(T²), so intensity rises with context length — the opposite of the eager kernel, whose extra O(T²) traffic drags it down.',
    lesson: 'attention-optimization/flash-attention',
  },
  'causal-skip': {
    term: 'Causal skip',
    def: 'Skipping score-matrix tiles that lie above the causal diagonal, because no query can attend to a later key. It removes about half the FLOPs for free: a correctness constraint that also pays a performance dividend.',
    lesson: 'attention-optimization/flash-attention',
  },
  'flash-decoding': {
    term: 'FlashDecoding',
    def: 'Decode attention that splits the key sequence into chunks processed in parallel on separate multiprocessors, each producing a partial maximum, sum and output. The partials are merged exactly by the online-softmax rule. Also called split-K.',
    lesson: 'attention-optimization/decode-attention',
  },
  'split-k': {
    term: 'Split-K',
    def: 'Running one logical reduction as several independent partial reductions over slices of the input, then combining them. Used in decode attention because a single query row cannot fill the GPU.',
    lesson: 'attention-optimization/decode-attention',
  },
  'gqa-group-factor': {
    term: 'GQA group factor',
    def: 'The ratio of query heads to key-value heads. It is not only how much the KV cache shrinks — it is also decode attention’s arithmetic intensity in FLOP per byte, which is why a smaller KV cache and a more bandwidth-starved kernel are the same fact.',
    lesson: 'attention-optimization/decode-attention',
  },
  'decode-shape': {
    term: 'Decode shape',
    def: 'The attention call shape used while generating: one query row against the entire key cache. It is a matrix-vector product, which is the least efficient shape on a GPU and cannot fill a tensor core.',
    lesson: 'attention-optimization/decode-attention',
  },
  'attention-backend': {
    term: 'Attention backend',
    def: 'The choice of attention kernel an engine will run — eager, FlashAttention, FlashDecoding, paged, or a vendor build. No single backend wins everywhere, because the right one depends on batch size, context length and phase.',
    lesson: 'attention-optimization/decode-attention',
  },
  'no-gather-backend': {
    term: 'No-gather backend',
    def: 'A paged attention kernel that indexes the block table directly instead of first copying scattered blocks into a contiguous buffer, removing the gather copy that paged attention would otherwise pay.',
    lesson: 'attention-optimization/decode-attention',
  },
  'bandwidth-bound-decode': {
    term: 'Bandwidth-bound decode',
    def: 'Decode attention whose time is set by how fast the KV cache can be read, not by arithmetic. Its intensity is fixed at the GQA group factor, so it stays bandwidth-bound at every context length and no kernel removes that.',
    lesson: 'attention-optimization/decode-attention',
  },
};