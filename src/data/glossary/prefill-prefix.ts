// Module 10 · Prefill & Prefix Optimization — new terms.
import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  'chunked-prefill': {
    term: 'Chunked prefill',
    def: 'Splitting a long prompt’s prefill into pieces small enough to fit in one engine step, so tokens keep streaming to other users between chunks. The FLOPs are unchanged; only their timing moves.',
    lesson: 'prefill-prefix/chunked-prefill',
  },
  'head-of-line-blocking': {
    term: 'Head-of-line blocking',
    def: 'One slow item (here, a long prefill) delaying everything behind it. On a serving GPU it shows up as a TPOT regression for every other user, not just a slow TTFT for the unlucky one.',
    lesson: 'prefill-prefix/chunked-prefill',
  },
  'interference': {
    term: 'Long-prompt interference',
    def: 'The slowdown a long prompt inflicts on other requests sharing the GPU, because an engine step is either prefill or decode and never both at once.',
    lesson: 'prefill-prefix/chunked-prefill',
  },
  'prefix-caching': {
    term: 'Prefix caching',
    def: 'Reusing the KV cache of tokens already seen at the start of a prompt, so only the new suffix is computed. Exact rather than approximate: the same token IDs at the same positions produce identical keys and values.',
    lesson: 'prefill-prefix/prefix-caching',
  },
  'cache-key': {
    term: 'Cache key',
    def: 'What a prefix cache hashes to decide whether it can reuse stored KV. Must include token IDs, positions, model identity and adapter identity — an incomplete key serves wrong answers silently instead of crashing.',
    lesson: 'prefill-prefix/prefix-caching',
  },
  'hit-rate': {
    term: 'Hit rate',
    def: 'The fraction of prompt tokens a prefix cache could reuse. Prefill cost becomes (1 − hit rate) of its cold cost, so hit rate is the single number describing the win.',
    lesson: 'prefill-prefix/prefix-caching',
  },
  'radix-attention': {
    term: 'RadixAttention',
    def: 'A KV cache organised as a radix tree over token blocks, one tree per model per GPU, so every shared prefix is stored once and a new request walks down to find its own history.',
    lesson: 'prefill-prefix/radix-attention',
  },
  'block-hash-chaining': {
    term: 'Block hash chaining',
    def: 'Hashing each KV block together with its parent’s hash (hash(block ‖ parent)), so a block’s identity includes its whole prefix. This is what makes a prefix-cache hit exact rather than approximate.',
    lesson: 'prefill-prefix/radix-attention',
  },
  eviction: {
    term: 'Eviction policy',
    def: 'How a prefix cache decides what to drop when memory runs short. For a radix tree the safe rule is structural: evict only leaves whose whole subtree is cold, oldest first — a shared node with living children is never a candidate.',
    lesson: 'prefill-prefix/radix-attention',
  },
  'cold-block': {
    term: 'Cold subtree',
    def: 'A branch of the prefix tree that no running request points into. Only cold subtrees are safe to evict; evicting a node with live descendants would strand every request beneath it.',
    lesson: 'prefill-prefix/radix-attention',
  },
  'cache-friendly-app': {
    term: 'Cache-friendly application',
    def: 'A prompt shaped for prefix reuse: long byte-identical stable prefix first, short variable tail last, retrieved documents sorted deterministically, and no per-request timestamp or ID in front.',
    lesson: 'prefill-prefix/radix-attention',
  },
};