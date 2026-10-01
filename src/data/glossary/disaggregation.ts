// Module 21 · Prefill/Decode Disaggregation — new terms.
import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  disaggregation: {
    term: 'Disaggregation',
    def: 'Running prefill and decode on separate sets of GPUs instead of the same one, so each phase gets hardware specialised for the bottleneck it is actually bound by. Prefill wants FLOPs; decode wants memory bandwidth. The price is a KV-cache transfer between the pools.',
    lesson: 'disaggregation/why-disaggregate',
  },
  'prefill-pool': {
    term: 'Prefill pool',
    def: 'The set of GPUs dedicated to prefill, sized by FLOP/s because prefill is compute-bound. Prefill instances are nearly stateless: they read a prompt, write its KV cache, ship it, and are immediately free.',
    lesson: 'disaggregation/kv-transfer-sizing',
  },
  'decode-pool': {
    term: 'Decode pool',
    def: 'The set of GPUs dedicated to decoding, sized by two constraints at once — the output tokens per second it can deliver, and the KV memory it can hold for hundreds of concurrent sequences. Whichever binds first sets the count.',
    lesson: 'disaggregation/kv-transfer-sizing',
  },
  'kv-transfer': {
    term: 'KV transfer (handoff)',
    def: 'Moving a finished request’s KV cache from the prefill instance that computed it to the decode instance that will stream from it. Everything prefill produced crosses the wire: the KV cache, the position offset and the sampling state. The weights never move — each instance loads its own.',
    lesson: 'disaggregation/why-disaggregate',
  },
  'transfer-share': {
    term: 'Transfer share',
    def: 'The cost of one KV transfer measured as a fraction of one decode step. Keep it under about 10%: the transfer is paid once per request while decode pays its step once per generated token, so a small share is invisible in steady state.',
    lesson: 'disaggregation/kv-transfer-sizing',
  },
  'ridge-point': {
    term: 'Ridge point',
    def: 'The arithmetic intensity at which a chip stops being memory-bound and becomes compute-bound: peak FLOP/s divided by bandwidth. About 295 FLOP per byte on an H100. Prefill sits above it, decode far below.',
    lesson: 'disaggregation/why-disaggregate',
  },
  'bin-packing': {
    term: 'Bin-packing (routing)',
    def: 'Routing requests to decode instances by whether the request fits and how tightly it packs, rather than by load average. A request needs a slot with room for its own context, so an “emptiest” instance can still be unable to accept it.',
    lesson: 'disaggregation/kv-transfer-sizing',
  },
  'specialisation': {
    term: 'Pool specialisation',
    def: 'The reason low utilisation on a specialised pool is not waste: a GPU is only worth having when it is doing the work it is good at. 27% utilisation doing memory-bound decode beats 48% utilisation doing someone else’s compute-bound prefill.',
    lesson: 'disaggregation/kv-transfer-sizing',
  },
  'binding-constraint': {
    term: 'Binding constraint',
    def: 'When a resource is sized by two limits, the binding one is the larger. Decode sizing typically has two — throughput and KV residency — and whichever demands more instances is the one that must be honoured. Sizing by the other ships an OOM.',
    lesson: 'disaggregation/kv-transfer-sizing',
  },
  'prefill-duty': {
    term: 'Prefill duty cycle',
    def: 'The share of a shared GPU’s time spent prefilling other users’ prompts instead of decoding. On a fleet sized only for decode, a large duty cycle inflates every streaming user’s TPOT by 1/(1 − duty).',
    lesson: 'disaggregation/kv-transfer-sizing',
  },
};
