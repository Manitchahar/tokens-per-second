import type { GlossaryEntry } from '../glossary';

// New vocabulary introduced in Module 08. Terms the learner already met in
// earlier modules (kv-cache, batch, oom, paged-attention, ...) live in src/data/glossary.ts.
export const terms: Record<string, GlossaryEntry> = {
  'max-model-len': {
    term: 'Max model length',
    def: 'The longest context the engine is configured to accept, in tokens. With contiguous KV memory every running sequence must reserve a buffer this big up front, whether it needs it or not.',
    lesson: 'paged-attention/fragmentation',
  },
  fragmentation: {
    term: 'Fragmentation',
    def: 'Memory you allocated but cannot use. The KV cache suffers from it in two ways: internal (a partly filled block) and external (free memory scattered into holes that no single new sequence fits into).',
    lesson: 'paged-attention/fragmentation',
  },
  'internal-fragmentation': {
    term: 'Internal fragmentation',
    def: 'Space wasted inside an allocation you already own, because the allocation is bigger than what you used. The tail of the last KV block is the everyday case.',
    lesson: 'paged-attention/fragmentation',
  },
  'external-fragmentation': {
    term: 'External fragmentation',
    def: 'Free memory that exists in total but is split into scattered holes, so one contiguous request no longer fits in any of them.',
    lesson: 'paged-attention/fragmentation',
  },
  'kv-block': {
    term: 'KV block',
    def: 'The unit of KV cache allocation: a fixed-size chunk holding the keys and values of a fixed number of tokens (commonly 16) for every layer. A sequence owns a list of blocks, not one buffer.',
    lesson: 'paged-attention/blocks-and-tables',
  },
  'block-size': {
    term: 'Block size',
    def: 'How many tokens one KV block holds. Bigger blocks waste less on the partially filled tail, but make the gather kernel coarser.',
    lesson: 'paged-attention/blocks-and-tables',
  },
  'token-slot': {
    term: 'Token slot',
    def: 'One token-sized row inside a block: the K and V vectors for a single token in a single layer. A block is just a fixed number of stacked slots.',
    lesson: 'paged-attention/blocks-and-tables',
  },
  'block-table': {
    term: 'Block table',
    def: 'The per-request list of block ids that says where its KV cache actually lives, plus how many slots of the last block are valid. One small lookup turns the indirection into a normal tensor read.',
    lesson: 'paged-attention/blocks-and-tables',
  },
  'kv-pool': {
    term: 'KV cache pool',
    def: 'The one big pre-allocated tensor holding every block for every request. Sizing it once, at startup, is what removes per-request allocation from the hot loop.',
    lesson: 'paged-attention/blocks-and-tables',
  },
  'free-list': {
    term: 'Free list',
    def: 'The list of block ids in the pool that belong to nobody right now. Allocating a token means popping one; freeing a request means pushing its blocks back.',
    lesson: 'paged-attention/blocks-and-tables',
  },
  'paged-kernel': {
    term: 'Paged attention kernel',
    def: 'An attention kernel that reads K and V through the block table instead of assuming one contiguous buffer. Every block has the same shape, so the inner loop stays a fast fixed-shape attention.',
    lesson: 'paged-attention/blocks-and-tables',
  },
  gather: {
    term: 'Gather',
    def: 'Reading scattered addresses and writing them into one compact buffer, so the following math sees a normal contiguous tensor. Paged attention always pays for exactly one gather.',
    lesson: 'paged-attention/blocks-and-tables',
  },
  'prefix-sharing': {
    term: 'Prefix sharing',
    def: 'Two requests whose prompts start identically can point at the same physical KV blocks instead of storing them twice. Block-level sharing is free; a block table makes it exact.',
    lesson: 'paged-attention/sharing-cow',
  },
  'copy-on-write': {
    term: 'Copy-on-write',
    def: 'The rule that makes sharing safe: while a block has more than one owner, writing into it is illegal, so the writer takes a private copy first. Shared blocks are read-only by construction.',
    lesson: 'paged-attention/sharing-cow',
  },
  refcount: {
    term: 'Reference count',
    def: 'How many sequences currently point at a block. A block is free to reuse only when its count reaches zero, which is exactly the moment a block can be recycled with no copying at all.',
    lesson: 'paged-attention/sharing-cow',
  },
  'refcount-walk': {
    term: 'Prefix walk',
    def: 'Matching a new prompt against the blocks already cached by walking down their shared trie token by token. The part of the prompt that matches is reused; the first mismatch is where prefill starts.',
    lesson: 'paged-attention/sharing-cow',
  },
  preemption: {
    term: 'Preemption',
    def: 'Taking resources away from a request that is already running, because the pool is full and someone else must go. In a paged engine it is rare, which is the point.',
    lesson: 'paged-attention/kv-pressure',
  },
  'kv-swap': {
    term: 'Swap out',
    def: 'Preemption by moving a request’s KV blocks to host memory and back later. It costs PCIe bandwidth and adds a latency spike, but keeps the work already done.',
    lesson: 'paged-attention/kv-pressure',
  },
  recompute: {
    term: 'Recompute',
    def: 'Preemption by throwing a request’s KV blocks away and rebuilding them by running prefill over its prompt again. No host memory and no PCIe traffic, but it pays the FLOPs a second time.',
    lesson: 'paged-attention/kv-pressure',
  },
  'admission-control': {
    term: 'Admission control',
    def: 'Refusing or queueing new work before it can wedge the system. The cheapest eviction policy: a request that never started has wasted nothing.',
    lesson: 'paged-attention/kv-pressure',
  },
};