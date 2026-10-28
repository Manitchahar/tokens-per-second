// Module 26 · Specialized Workloads — the arithmetic, shared by the module's visuals.
//
// Every number a lesson or a visual shows comes from a function in this file, so the
// prose, the cheat sheet and the pictures can never drift apart. Model shapes come from
// published configuration files / model cards; hardware figures are public spec-sheet
// numbers and vary a little by SKU and clock, so treat them as approximate.
//
// Nothing here reads the clock, the network or Math.random(): visuals must render
// identically on the server and in the browser.

import { GPUS, type Gpu } from '../lib/perfModel';

export const gpu = (id: string): Gpu => GPUS.find((g) => g.id === id) ?? GPUS[3];

// ───────────────────────── shared formatting ─────────────────────────

/** 8.03e9 -> "8.0B", 131072 -> "131K". Used everywhere a parameter count appears. */
export function B(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e9) return `${(n / 1e9).toFixed(a / 1e9 >= 10 ? 0 : 1)}B`;
  if (a >= 1e6) return `${(n / 1e6).toFixed(a / 1e6 >= 100 ? 0 : 1)}M`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(a / 1e3 >= 100 ? 0 : 1)}K`;
  return `${Math.round(n)}`;
}

// ═════════════════════════ 1 · structured output ═════════════════════════

/**
 * A constrained-decoding step. The engine holds the model distribution and, alongside
 * it, a *mask* over the vocabulary that is computed on the fly from a JSON grammar.
 * The mask has one bit per token: may this token be emitted here without breaking the
 * grammar? Masking is elementwise and free next to the rest of the step, so the whole
 * question is how much of the vocabulary survives at each state of the grammar.
 */
export interface GrammarState {
  /** Tokens the grammar permits right now, as a fraction of the vocabulary. */
  allowedFrac: number;
  /** What the grammar expects next — the label shown on the visual. */
  expect: string;
  /** Human sentence describing the state. */
  note: string;
  /** Whether a wrong-but-valid-token branch still exists here. */
  branchy: boolean;
}

/**
 * The vocab-allowed fraction at each point of parsing `{"city": "Paris", "n": 3}` with a
 * JSON grammar. The two orders matter:
 *
 *  - Structural characters (`{ " :` ,) are a handful of exact token IDs, so the allowed
 *    set is tiny — typically under 0.1% of the vocabulary.
 *  - Inside a string, the allowed set is *every* token that continues a legal string:
 *    all the letters, the digits, the escapes, the closing quote. That is most of the
 *    vocabulary, because most tokens are ordinary words.
 *
 * So a JSON grammar alternates between a near-pinned state and a nearly-unconstrained
 * one, and the fraction only drops hard right after a `"` that ends a key or a value.
 */
export const GRAMMAR_TRACE: GrammarState[] = [
  { allowedFrac: 1.0, expect: 'anything', note: 'Unconstrained. Every token in the vocabulary is allowed.', branchy: true },
  { allowedFrac: 0.0004, expect: '{', note: 'Object start. Only the opening brace survives.', branchy: false },
  { allowedFrac: 0.0009, expect: '"city"', note: 'A key. Only string fragments can begin a key.', branchy: true },
  { allowedFrac: 0.62, expect: 'value text', note: 'Inside the string: nearly anything legal is allowed.', branchy: true },
  { allowedFrac: 0.0021, expect: '" ,', note: 'Closing quote, then comma or brace. Two states, both narrow.', branchy: false },
  { allowedFrac: 0.0009, expect: '"n"', note: 'The next key. Narrow again.', branchy: true },
  { allowedFrac: 0.51, expect: 'digits', note: 'A number field: digits, minus, exponent, closing brace.', branchy: true },
  { allowedFrac: 0.0006, expect: '}', note: 'Object end. Pinned to a single token.', branchy: false },
  { allowedFrac: 1.0, expect: 'anything', note: 'Done. The mask is released and sampling is free again.', branchy: true },
];

/** Grammars worth comparing, by fraction of the vocabulary they leave open mid-string. */
export const GRAMMARS = [
  { id: 'free', label: 'Free text', mid: 1.0, narrow: 1.0, blurb: 'no constraint at all' },
  { id: 'json', label: 'JSON', mid: 0.62, narrow: 0.0004, blurb: 'alternates: pinned, then wide' },
  { id: 'sql', label: 'SQL', mid: 0.44, narrow: 0.006, blurb: 'wide in text, narrow on keywords' },
  { id: 'enum', label: 'Fixed enum', mid: 0.00002, narrow: 0.00002, blurb: 'pinned everywhere: k tokens' },
  { id: 'regex', label: 'Regex', mid: 0.31, narrow: 0.002, blurb: 'depends entirely on the pattern' },
] as const;

export type GrammarId = (typeof GRAMMARS)[number]['id'];

/**
 * A tool call must be emitted as valid JSON, and the *arguments* are the part a model
 * actually gets wrong. This is the shape of a function call, with the fields a schema
 * pins down. It is the object the visual walks through.
 */
export const TOOL_CALL = {
  name: 'get_weather',
  schema: [
    { field: 'city', type: 'string', example: '"Paris"', required: true },
    { field: 'unit', type: 'enum(c,f)', example: '"c"', required: true },
    { field: 'days', type: 'int 1-7', example: '3', required: false },
  ],
} as const;

/**
 * Tokens spent on a tool call, split by what produced them.
 *
 * A tool call is *not* free text: the model emits a name and an argument object. The
 * boilerplate (the function tag, the field names, the braces and quotes) is forced by
 * the grammar and is a fixed cost per call; only the argument *values* are real choice.
 * That ratio is what decides whether schema pinning is worth its rigidness.
 */
export interface ToolCallCost {
  /** Tokens for the fixed scaffolding: function name, keys, punctuation. */
  boilerplate: number;
  /** Tokens for the argument values themselves. */
  payload: number;
  total: number;
  /** Payload as a fraction of the call. */
  payloadFrac: number;
}

/** Token counts for the tool call above, with a growing number of arguments. */
export function toolCallCost(nArgs: number): ToolCallCost {
  const boilerplate = 8 + nArgs * 5; // function tag, one key + quotes + colon + comma each
  const payload = nArgs * 7; // a short string or integer value
  return {
    boilerplate,
    payload,
    total: boilerplate + payload,
    payloadFrac: payload / (boilerplate + payload),
  };
}

// ═════════════════════════ 2 · multi-LoRA ═════════════════════════

/** A dense model shape, used as the base for every LoRA calculation. */
export interface DenseShape {
  id: string;
  name: string;
  layers: number;
  hidden: number;
  /** Feed-forward intermediate width. */
  intermediate: number;
  kvHeads: number;
  headDim: number;
  params: number;
}

/**
 * Llama-3.1-8B is the workhorse of this course and the natural base for an adapter
 * demo. The shape below is the published configuration; the parameter total is
 * derived from it in the lesson and lands within a few percent of the published 8.03B.
 */
export const BASE_8B: DenseShape = {
  id: 'llama8b',
  name: 'Llama-3.1-8B',
  layers: 32,
  hidden: 4096,
  intermediate: 14336,
  kvHeads: 8,
  headDim: 128,
  params: 8.03e9,
};

/** A 1B-scale base, for "one adapter for every tenant" stories. */
export const BASE_1B: DenseShape = {
  id: 'qwen05',
  name: 'Qwen2.5-0.5B',
  layers: 24,
  hidden: 896,
  intermediate: 4864,
  kvHeads: 2,
  headDim: 64,
  params: 0.49e9,
};

/**
 * Parameters in one LoRA pair on one matrix of shape (out, in) with rank r:
 *
 *     A: r x in     B: out x r      ->  r (in + out)
 *
 * The base matrix it modifies has in*out parameters. So the adapter's share of that
 * matrix is r(in+out)/(in*out), and the usual shorthand r/d compares against the
 * *hidden* width rather than both dimensions — which is why rank is quoted in units of
 * "hidden width" and why r=16 on a 4096-wide model is 0.39%.
 */
export const loraParams = (out: number, inp: number, r: number): number => r * (out + inp);

/**
 * Every matrix a LoRA adapter targets, in a transformer block. Attention's four
 * projections are out x in with both sides = hidden (times heads for q/o); the
 * feed-forward pair is intermediate x hidden and hidden x intermediate.
 */
export interface LoraBreakdown {
  /** Total adapter parameters when every listed matrix is adapted. */
  total: number;
  /** Share of the base model's parameters. */
  pctOfBase: number;
  /** Bytes of the adapter at 2 bytes per weight. */
  bytes: number;
  perKind: { kind: string; params: number; /** How many matrices of this kind are adapted. */ count: number }[];
}

export function loraBreakdown(m: DenseShape, r: number, targets: ('attn' | 'ffn')[]): LoraBreakdown {
  const d = m.hidden;
  const kinds: LoraBreakdown['perKind'] = [];
  let total = 0;

  if (targets.includes('attn')) {
    // Four projections per block: q and o at d x d, k and v at kvD x d (GQA).
    const kvD = m.kvHeads * m.headDim;
    const params =
      m.layers * (loraParams(d, d, r) * 2 + loraParams(kvD, d, r) * 2);
    kinds.push({ kind: 'attention q/k/v/o', params, count: 4 });
    total += params;
  }
  if (targets.includes('ffn')) {
    const params = m.layers * (loraParams(m.intermediate, d, r) + loraParams(d, m.intermediate, r));
    kinds.push({ kind: 'feed-forward up/down', params, count: 2 });
    total += params;
  }
  return {
    total,
    pctOfBase: total / m.params,
    bytes: total * 2,
    perKind: kinds,
  };
}

/** The ranks worth comparing, as a fraction of hidden width. */
export const RANKS = [1, 4, 8, 16, 32, 64] as const;

/**
 * Serving N adapters over one base: the memory you must hold is the base *plus every
 * adapter you can serve at once*, because you cannot know which tenant's adapter the
 * next request needs. This is the whole cost of the technique.
 */
export interface AdapterFleet {
  baseBytes: number;
  /** Bytes of all N adapters resident together. */
  adaptersBytes: number;
  totalBytes: number;
  /** Fraction of memory that is base model — how much of the card is the foundation. */
  baseFrac: number;
  /** How many of these fleets fit on one GPU. */
  fitsOnCard: number;
  /** The GPU's memory as a fraction used. */
  utilFrac: number;
}

export function adapterFleet(m: DenseShape, r: number, n: number, wBytes: number, cardId = 'a100'): AdapterFleet {
  const b = loraBreakdown(m, r, ['attn', 'ffn']);
  const baseBytes = m.params * wBytes;
  const adaptersBytes = b.total * wBytes * n;
  const totalBytes = baseBytes + adaptersBytes;
  const card = gpu(cardId);
  return {
    baseBytes,
    adaptersBytes,
    totalBytes,
    baseFrac: baseBytes / totalBytes,
    fitsOnCard: Math.floor(card.mem / totalBytes),
    utilFrac: totalBytes / card.mem,
  };
}

/**
 * Batch composition decides whether multi-LoRA is cheap or ruinous.
 *
 * If every sequence in a batch wants the same adapter, this is an ordinary batch. If
 * the batch is a mix of tenants, the engine must either (a) group by adapter and pay
 * more, smaller batches, or (b) gather the rows of each adapter into its own GEMM.
 * Option (b) is what production engines do: `groups`, one small GEMM per adapter.
 *
 * Read `groups` as "how many distinct adapters are in flight".
 */
export interface BatchMix {
  /** Distinct adapters in the batch. */
  groups: number;
  /** Cost multiplier on the decode step from splitting the batch this way. */
  slowdown: number;
  /** Tokens per individual GEMM at batch 32. */
  perGroupTokens: number;
  /** Whether the batch is still worth running merged. */
  worthMerging: boolean;
}

export function batchMix(groups: number, batch = 32): BatchMix {
  // Splitting a batch of B into g groups makes each GEMM B/g rows. Small-row GEMMs
  // lose bandwidth utilisation, so the penalty is super-linear in g but sub-linear
  // in the total: it saturates once each group has only a few rows.
  const rows = batch / Math.max(1, groups);
  const slowdown = 1 + (groups - 1) * 0.06 + Math.max(0, 16 - rows) * 0.004;
  return {
    groups,
    slowdown,
    perGroupTokens: rows,
    worthMerging: slowdown < 2,
  };
}

// ═════════════════════════ 3 · multimodal ═════════════════════════

/**
 * A vision-language model turns an image into *tokens* before the language model sees
 * anything. Two encoders do the work:
 *
 *   - A vision transformer crops the image into patches and projects each into the
 *     language model's hidden width. One patch becomes one or more text-side tokens.
 *   - Those tokens are concatenated with the text tokens and prefill runs over all
 *     of them exactly as it would over text alone.
 *
 * So "multimodal inference" is, mechanically, prefill with a longer sequence plus an
 * extra model in front. Everything that made long prompts expensive (Module 10) now
 * applies to images.
 */
export interface VisionShape {
  id: string;
  name: string;
  /** Pixels along one side each crop is resized to. */
  imageSize: number;
  /** Patch size the vision tower cuts a crop into. */
  patch: number;
  /** Vision transformer width. */
  visionHidden: number;
  /** Vision transformer depth. */
  visionLayers: number;
  /** How many text tokens one patch becomes (pooling factor). */
  tokensPerPatch: number;
  /** How many 336px crops this model looks at for one image (anyres). */
  crops: number;
  /** Language model width. */
  hidden: number;
  /** Language model KV heads (for the decode-side KV math). */
  kvHeads: number;
  /** Language model head dim. */
  headDim: number;
  /** Language model layers. */
  layers: number;
  /** Vision tower parameters, for the FLOP comparison. */
  visionParams: number;
  /** Text parameters. */
  textParams: number;
}

/**
 * Three shapes worth comparing.
 *
 * LLaVA-1.5 uses one 336px crop of a CLIP ViT-L/14 tower: 24x24 = 576 patches, one
 * text token each, so 576 image tokens. That is the number the model card quotes.
 *
 * LLaVA-1.6 ("anyres") looks at four 336px crops instead of one and pools each 2x2,
 * so four crops of 576 patches become 4 x 576/4 = 576 tokens — the same budget, but
 * covering four times the area, at four times the vision-tower compute.
 */
export const VISION_MODELS: VisionShape[] = [
  {
    id: 'llava15',
    name: 'LLaVA-1.5-7B',
    imageSize: 336,
    patch: 14,
    visionHidden: 1024,
    visionLayers: 24,
    tokensPerPatch: 1,
    crops: 1,
    hidden: 4096,
    kvHeads: 8,
    headDim: 128,
    layers: 32,
    visionParams: 0.3e9,
    textParams: 6.7e9,
  },
  {
    id: 'llava16',
    name: 'LLaVA-1.6-7B',
    imageSize: 336,
    patch: 14,
    visionHidden: 1024,
    visionLayers: 24,
    tokensPerPatch: 0.25,
    crops: 4,
    hidden: 4096,
    kvHeads: 8,
    headDim: 128,
    layers: 32,
    visionParams: 0.3e9,
    textParams: 6.7e9,
  },
  {
    id: 'llava13b',
    name: 'LLaVA-1.5-13B',
    imageSize: 336,
    patch: 14,
    visionHidden: 1024,
    visionLayers: 24,
    tokensPerPatch: 1,
    crops: 1,
    hidden: 5120,
    kvHeads: 8,
    headDim: 128,
    layers: 40,
    visionParams: 0.3e9,
    textParams: 13.4e9,
  },
];

export const visionById = (id: string): VisionShape => VISION_MODELS.find((v) => v.id === id)!;

/** Patches along one side of one crop. */
export const patchesPerSide = (v: VisionShape): number => Math.round(v.imageSize / v.patch);

/** Patches the vision tower processes for one image, across all its crops. */
export const patchesPerImage = (v: VisionShape): number => patchesPerSide(v) ** 2 * v.crops;

/**
 * Text-side tokens one image costs.
 *
 * `crops x patchesPerSide^2` patches, each becoming `tokensPerPatch` text tokens. For
 * LLaVA-1.5 that is 1 x 576 x 1 = 576 tokens; for LLaVA-1.6, 4 x 576 x 0.25 = 576.
 * Both land on the same budget by design — the anyres version buys four times the
 * image area for the same number of tokens.
 *
 * This is the number that decides everything else: a 576-token image is a 576-token
 * prompt as far as the language model's prefill is concerned.
 */
export const imageTokens = (v: VisionShape): number =>
  Math.round(patchesPerSide(v) ** 2 * v.crops * v.tokensPerPatch);

/** Bytes the vision tower alone reads, at 2 bytes per weight. */
export const visionWeightBytes = (v: VisionShape): number => v.visionParams * 2;

/**
 * The projector: a small MLP mapping visionHidden to the text model's hidden width.
 * Two linear layers is the common choice, and it is worth knowing it is *small* —
 * the vision tower holds the parameters, the projector holds almost nothing.
 */
export function projectorParams(v: VisionShape, hidden: number): number {
  const mid = v.visionHidden * 2;
  return v.visionHidden * mid + mid * hidden;
}

/**
 * Prefill for a multimodal request: the vision tower runs once per image over its
 * patches, then the language model prefills text tokens + image tokens together.
 *
 * Note the shape of the vision cost. It is *not* proportional to the language model's
 * context: it depends only on the image, and it is a GEMM over 576 patches — large
 * enough to be efficient, small enough to be fast. One image costs about one extra
 * prefill pass over a few hundred tokens.
 */
export interface VisionPrefill {
  /** Patches the vision tower processes. */
  patches: number;
  /** FLOPs of the vision tower, 2 per multiply-accumulate. */
  visionFlops: number;
  /** ms for the vision tower alone at an assumed 40% of peak. */
  visionMs: number;
  /** FLOPs of the language model over prompt + image tokens. */
  textFlops: number;
  /** ms for the language model prefill at 55% of peak. */
  textMs: number;
  /** Image tokens as a fraction of the whole prefill sequence. */
  imageFrac: number;
  totalMs: number;
}

export function visionPrefill(v: VisionShape, g: Gpu, promptTokens: number, nImages = 1): VisionPrefill {
  const patches = patchesPerImage(v) * nImages;
  const perPatchFlops = 2 * v.visionParams; // ~ one parameter-weight per output, per token
  const visionFlops = patches * perPatchFlops;
  const visionMs = (visionFlops / (g.flops * 0.4)) * 1e3;

  const imgTok = imageTokens(v) * nImages;
  const seq = promptTokens + imgTok;
  const textFlops = 2 * (v.textParams + projectorParams(v, v.hidden)) * seq;
  const textMs = (textFlops / (g.flops * 0.55)) * 1e3;

  return {
    patches,
    visionFlops,
    visionMs,
    textFlops,
    textMs,
    imageFrac: imgTok / seq,
    totalMs: visionMs + textMs,
  };
}

/**
 * Vision encoder batching is limited by *images*, not sequences.
 *
 * With batch 64 of text-only requests, a decode step is one GEMM of 64 rows. With 64
 * requests each carrying an image, the vision tower must run for every image in the
 * batch that step — and the tower is a separate model that does not share the
 * language model's batching. This is why image-heavy traffic needs its own capacity
 * plan: you are sizing two different bottlenecks.
 */
export interface VisionBatch {
  /** Requests per step. */
  batch: number;
  /** Images per request (multi-image chat is normal now). */
  imagesPerReq: number;
  /** Vision tower invocations per step. */
  towerCalls: number;
  /** Vision tower work per step, in "image-equivalents". */
  work: number;
  /** Decode step time in ms, decode only. */
  decodeMs: number;
  /** Vision time in ms. */
  visionMs: number;
  /** Share of the step the vision tower takes. */
  visionShare: number;
}

export function visionBatch(v: VisionShape, g: Gpu, batch: number, imagesPerReq: number): VisionBatch {
  const n = patchesPerSide(v);
  const work = batch * imagesPerReq;
  const towerFlops = work * n * n * v.crops * 2 * v.visionParams;
  const visionMs = (towerFlops / (g.flops * 0.4)) * 1e3;
  // Decode of the text model for the same batch: weights + KV, memory-bound.
  const kvPerToken = 2 * v.layers * v.kvHeads * v.headDim * 2;
  const bytes = v.textParams * 2 + batch * 512 * kvPerToken;
  const decodeMs = (bytes / (g.bw * 0.8)) * 1e3;
  return {
    batch,
    imagesPerReq,
    towerCalls: work,
    work,
    decodeMs,
    visionMs,
    visionShare: visionMs / (visionMs + decodeMs),
  };
}

// ═════════════════════════ 4 · agents & reasoning ═════════════════════════

/**
 * A multi-turn conversation is not a new workload. It is the *same* decode loop with a
 * growing prefix, and the prefix is almost entirely shared with the previous turn.
 *
 * That shared part is the cache. Module 10 introduced prefix caching: hash blocks of
 * the prompt, and reuse the KV for blocks you have already computed. A conversation of
 * k turns is one prefix that grows by one block at a time, so caching turns turns
 * N prefill passes into N block reads.
 */
export interface ConversationTurn {
  /** Turn index, 0-based. */
  turn: number;
  /** Tokens of context before this turn's prompt. */
  ctxBefore: number;
  /** Tokens this turn adds. */
  added: number;
  /** Tokens of context after this turn. */
  ctxAfter: number;
  /** Blocks the prefix cache already holds for this turn. */
  cacheHits: number;
  /** Blocks that must be computed fresh. */
  cacheMisses: number;
  /** Total blocks in the prefix. */
  blocks: number;
}

export const BLOCK = 16; // vLLM's default block size, in tokens

export function conversation(systemPrompt: number, perTurn: number, turns: number): ConversationTurn[] {
  const out: ConversationTurn[] = [];
  let ctx = systemPrompt;
  for (let t = 0; t < turns; t++) {
    const added = perTurn + (t % 2 === 0 ? 40 : 0); // user turn, then a longer tool result
    const before = ctx;
    ctx += added;
    const blocks = Math.ceil(ctx / BLOCK);
    // The cache holds every fully-written block from the previous turn.
    const hits = Math.ceil(before / BLOCK);
    out.push({ turn: t, ctxBefore: before, added, ctxAfter: ctx, cacheHits: hits, cacheMisses: Math.max(0, blocks - hits), blocks });
  }
  return out;
}

/**
 * Prefill work with and without a prefix cache, in FLOPs. The cache does not make
 * attention cheaper about *new* tokens, but it removes the entire prefill pass over
 * the prefix.
 */
export interface PrefillSaving {
  /** FLOPs if every turn prefills its whole context. */
  naiveFlops: number;
  /** FLOPs with a perfect prefix cache: only new tokens are prefilled. */
  cachedFlops: number;
  /** Fraction of prefill work the cache removes. */
  saved: number;
  /** Number of full-context prefills avoided. */
  avoided: number;
}

export function prefillSaving(turns: ConversationTurn[], params: number, flopsPerToken = 2): PrefillSaving {
  let naive = 0;
  let cached = 0;
  for (const t of turns) {
    naive += t.ctxAfter * params * flopsPerToken;
    cached += t.added * params * flopsPerToken;
  }
  return {
    naiveFlops: naive,
    cachedFlops: cached,
    saved: 1 - cached / naive,
    avoided: turns.length,
  };
}

/**
 * Reasoning models do two things that change serving:
 *
 *  1. They emit far more tokens per answer. A short answer is 200 tokens; a hard
 *     reasoning trace can be 5,000 and still not finished.
 *  2. The trace must be *streamed or hidden*. Showing a user 4,000 tokens of
 *     deliberation before the answer is a UX decision, not a serving one — but the
 *     serving consequence is identical either way: the context keeps growing.
 *
 * The compounding problem is that the trace joins the context. Turn 2 of a reasoning
 * conversation carries turn 1's 4,000 tokens of scratch work, so decode slows as the
 * conversation goes on — the KV cache grows linearly, and TPOT is proportional to it.
 */
export interface ReasoningProfile {
  /** Tokens the model emits per question. */
  trace: number;
  /** KV bytes per token for the model in use. */
  kvPerToken: number;
  /** Bytes per token at the end of the trace — the slowest it ever gets. */
  stepMs: number;
  /** Tokens per second *at the end* of the trace. */
  tps: number;
  /** Seconds to emit the whole trace, integrated. */
  totalS: number;
}

/**
 * The slowdown over a long trace, at a realistic batch size.
 *
 * Two effects are in play and they pull in opposite directions, which is why the
 * naive numbers people quote are wrong:
 *
 *  1. **KV grows.** At step t the KV cache holds t·kv bytes, so longer traces read
 *     more — this is the effect that hurts.
 *  2. **The batch amortises the weights.** A step reads the weight set once for the
 *     whole batch, so per-token weight cost is w/B. Only the KV term is per-token.
 *
 * The second effect dominates whenever the batch is large enough, which it is in any
 * real serving deployment. So the right model is per-token bytes = w/B + t·kv, and
 * the batch size is not a detail — it is the parameter that decides whether a long
 * trace is 20 seconds or two.
 *
 *   bytes(t) = w/B + t·kv
 *   total    = (wT/B + kv·T²/2) / BW
 *
 * Note what this means: at batch 1 a single sequence reads the entire model for one
 * token, which is the least efficient use of a GPU imaginable — and the reason
 * nobody serves a reasoning model at batch 1.
 */
export function reasoning(profile: {
  trace: number;
  kvPerToken: number;
  /** Bytes of weights read per step (the whole set, once). */
  wBytes: number;
  g: Gpu;
  /** Sequences decoding together. */
  batch?: number;
  mbu?: number;
}): ReasoningProfile {
  const g = profile.g;
  const T = profile.trace;
  const B = Math.max(1, profile.batch ?? 32);
  const w = profile.wBytes / B; // amortised across the batch
  const kv = profile.kvPerToken;
  const mbu = profile.mbu ?? 0.85;
  const totalMs = ((w * T + (kv * T * T) / 2) / (g.bw * mbu)) * 1e3;
  // Speed at the end of the trace — what the final tokens actually experience.
  const bytesEnd = w + T * kv;
  const stepMs = (bytesEnd / (g.bw * mbu)) * 1e3;
  return {
    trace: T,
    kvPerToken: kv,
    stepMs,
    tps: 1e3 / stepMs,
    totalS: totalMs / 1e3,
  };
}

/** Trace lengths worth comparing, as multiples of an ordinary answer. */
export const TRACE_LENGTHS = [
  { id: 'short', label: 'Direct answer', tokens: 200 },
  { id: 'medium', label: 'Brief reasoning', tokens: 1_200 },
  { id: 'long', label: 'Hard problem', tokens: 5_000 },
  { id: 'verylong', label: 'Very hard', tokens: 16_000 },
] as const;

/**
 * An agent is a loop, and the loop's cost is dominated by *context accumulation*.
 * Each turn appends the observation (tool output, screen state, file contents) to the
 * context. Nothing is ever removed, so the context grows linearly in turns while the
 * model's per-token cost grows with it.
 *
 * `obsTokens` is the size of one tool result. 3,000 tokens is a realistic search or
 * file read.
 */
export interface AgentLoop {
  turn: number;
  /** Total context tokens at this turn. */
  ctx: number;
  /** Tokens of new context this turn had to prefill (the cached prefix is free). */
  newTokens: number;
  /** Cumulative tokens prefilled across the whole loop. */
  cumPrefillTokens: number;
  /** How many times this loop has prefilled this same context, if there is no cache. */
  redundantPrefillTokens: number;
}

/**
 * Trace the first `turns` steps of an agent loop, each appending one observation.
 *
 * `redundantPrefillTokens` is the number you would prefill *without* a prefix cache:
 * without reuse, turn t must prefill the whole context again. With a cache it only
 * prefills the new observation. The gap between the two grows with the square of the
 * turn count, which is the real reason agent serving leans on prefix caching.
 */
export function agentLoop(systemPrompt: number, obsTokens: number, turns: number): AgentLoop[] {
  const out: AgentLoop[] = [];
  let ctx = systemPrompt;
  let cum = 0;
  let redundant = 0;
  for (let t = 1; t <= turns; t++) {
    ctx += obsTokens;
    cum += obsTokens;
    redundant += ctx;
    out.push({
      turn: t,
      ctx,
      newTokens: obsTokens,
      cumPrefillTokens: cum,
      redundantPrefillTokens: redundant,
    });
  }
  return out;
}

// ═════════════════════════ 5 · batch inference & RL rollouts ═════════════════════════

/**
 * RL fine-tuning needs thousands of *complete generations* from one frozen policy,
 * per optimisation step. Nothing else in this course has that shape.
 *
 * The consequences are structural, and they all fall out of one number — how many
 * rollouts are in flight at once:
 *
 *  1. **Every rollout is a whole conversation**, so the KV cache is enormous. With
 *     4096 rollouts of 4,096 tokens you are holding 16.8M tokens of KV cache.
 *  2. **The generation is ragged.** Rollouts finish at different steps, so the batch
 *     shrinks continuously — the opposite of a serving workload, which refills.
 *  3. **Nobody is waiting.** There is no user to feel latency, so throughput is the
 *     only objective, and the optimal operating point is a batch far past anything
 *     you would serve to humans.
 */
export interface RolloutConfig {
  modelId: string;
  params: number;
  layers: number;
  hidden: number;
  kvHeads: number;
  headDim: number;
  /** Rollouts generated per optimisation step (a whole batch). */
  groupSize: number;
  /** Tokens per rollout: prompt + generated. */
  seqLen: number;
  /** Fraction of the sequence that is generated. */
  genFrac: number;
  /** GPUs available. */
  gpus: number;
  gpuId: string;
  wBytes: number;
}

/**
 * Two published dense models, the usual RL fine-tuning workhorses.
 *
 * The KV-head count is the whole story here. Qwen2.5-1.5B uses 2 KV heads of dim 128,
 * so its KV cache is 28 KB per token; Llama-3.1-8B uses 8, so 128 KB per token — 4.5x
 * more for a model only 5x bigger, and 131 KB/token is what makes long rollouts
 * expensive. Rollout clusters are sized by *tokens*, not by parameters.
 */
export const ROLLOUT_MODELS: RolloutConfig[] = [
  {
    modelId: 'qwen2.5-1.5b',
    params: 1.54e9,
    layers: 28,
    hidden: 1536,
    kvHeads: 2,
    headDim: 128,
    groupSize: 512,
    seqLen: 4096,
    genFrac: 0.75,
    gpus: 8,
    gpuId: 'h100',
    wBytes: 2,
  },
  {
    modelId: 'qwen2.5-7b',
    params: 7.62e9,
    layers: 28,
    hidden: 3584,
    kvHeads: 4,
    headDim: 128,
    groupSize: 1024,
    seqLen: 4096,
    genFrac: 0.75,
    gpus: 8,
    gpuId: 'h100',
    wBytes: 2,
  },
  {
    modelId: 'llama3.1-8b',
    params: 8.03e9,
    layers: 32,
    hidden: 4096,
    kvHeads: 8,
    headDim: 128,
    groupSize: 1024,
    seqLen: 4096,
    genFrac: 0.75,
    gpus: 8,
    gpuId: 'h100',
    wBytes: 2,
  },
];

/** Look up a rollout config by its array id (e.g. 'qwen2.5-7b'). */
export const rolloutById = (id: string): RolloutConfig => ROLLOUT_MODELS.find((m) => m.id === id) ?? ROLLOUT_MODELS[0];

/** KV bytes per token for a rollout config. Same formula as everywhere else. */
export const rolloutKvPerToken = (m: RolloutConfig, bytes = 2): number =>
  2 * m.layers * m.kvHeads * m.headDim * bytes;

export interface RolloutCost {
  /** KV bytes for one rollout at full length. */
  kvPerRollout: number;
  /** KV bytes for the whole group — what must be resident. */
  totalKvBytes: number;
  /** Plus weights. */
  totalBytes: number;
  /** Rollouts that fit on one GPU after weights. */
  fitPerGpu: number;
  /** Rollouts that fit across the cluster. */
  fitTotal: number;
  /** GPUs needed for the whole group. */
  gpusNeeded: number;
  fits: boolean;
  /** Decode step ms at the resident batch on one GPU. */
  stepMs: number;
  /** Generated tokens per second per GPU. */
  tpsPerGpu: number;
  /** Generated tokens per second for the cluster. */
  tpsCluster: number;
  /** Seconds to generate the whole group. */
  wallS: number;
  /** Prefill FLOPs for the whole group. */
  prefillFlops: number;
}

/** The full cost of one RL rollout group, sized to a specific cluster. */
export function rolloutCost(c: RolloutConfig): RolloutCost {
  const g = gpu(c.gpuId);
  const kvT = rolloutKvPerToken(c);
  const kvPerRollout = c.seqLen * kvT;
  const totalKvBytes = kvPerRollout * c.groupSize;
  const weights = c.params * c.wBytes;
  // A rough reserve for activations and the CUDA context, as in perfModel.
  const reserve = 0.05 * g.mem;

  const fitPerGpu = Math.max(0, Math.floor((g.mem - weights - reserve) / kvPerRollout));
  const fitTotal = fitPerGpu * c.gpus;
  const gpusNeeded = Math.ceil(c.groupSize / Math.max(1, fitPerGpu));

  // Decode at the resident batch: memory-bound, so bigger batch is nearly free.
  const batch = Math.max(1, Math.min(fitPerGpu, c.groupSize));
  const bytes = weights + batch * kvPerRollout * (1 - c.genFrac);
  const stepMs = (bytes / (g.bw * 0.8)) * 1e3;
  const tpsPerGpu = 1e3 / stepMs;
  const tpsCluster = tpsPerGpu * c.gpus;

  const genTokens = c.groupSize * c.seqLen * c.genFrac;
  const prefillFlops = 2 * c.params * c.groupSize * c.seqLen * (1 - c.genFrac);

  return {
    kvPerRollout,
    totalKvBytes,
    totalBytes: weights * c.gpus + totalKvBytes,
    fitPerGpu,
    fitTotal,
    gpusNeeded,
    fits: gpusNeeded <= c.gpus,
    stepMs,
    tpsPerGpu,
    tpsCluster,
    wallS: genTokens / tpsCluster,
    prefillFlops,
  };
}

/**
 * The ragged-batch problem, quantified.
 *
 * A rollout group finishes at different times. If completion times are spread over a
 * window, the *average* number of resident sequences over the generation is roughly
 * half the group, not the whole group — but the peak is the whole group, and peak is
 * what must fit in memory. Meanwhile the arithmetic intensity *rises* as the batch
 * shrinks, so the tail of the generation runs at a fraction of peak throughput.
 *
 * `spread` is the fraction of the generation window over which rollouts complete
 * (0 = all finish together, 1 = completions spread uniformly).
 */
export interface Raggedness {
  /** Average resident batch over the window. */
  avgBatch: number;
  /** Peak resident batch. */
  peakBatch: number;
  /** Bytes read per token at the average batch. */
  bytesPerToken: number;
  /** Same at the peak. */
  bytesPerTokenPeak: number;
  /** Arithmetic intensity at the average batch, FLOP per byte. */
  intensity: number;
  /** Roofline intensity where memory stops being the limit. */
  ridgeIntensity: number;
  /** How far the average batch is below the ridge — 1.0 means still memory-bound. */
  ridgeGap: number;
}

export function raggedness(c: RolloutConfig, spread = 0.8): Raggedness {
  const g = gpu(c.gpuId);
  const kvT = rolloutKvPerToken(c);
  const avg = Math.max(1, Math.round(c.groupSize * (1 - spread / 2)));
  const peak = c.groupSize;
  const wBytesPerTok = c.params * c.wBytes;
  const bytesAt = (b: number) => wBytesPerTok + b * c.seqLen * kvT;
  const intensity = (b: number) => (2 * c.params) / bytesAt(b);
  // The roofline ridge: FLOP/s per byte/s.
  const ridge = g.flops / g.bw;
  const ia = intensity(avg);
  return {
    avgBatch: avg,
    peakBatch: peak,
    bytesPerToken: bytesAt(avg),
    bytesPerTokenPeak: bytesAt(peak),
    intensity: ia,
    ridgeIntensity: ridge,
    ridgeGap: ia / ridge,
  };
}

/**
 * Generation length is the RL engineer's main dial and people get it wrong by 10x.
 *
 * Too short and the rollouts are truncated before the model can finish a thought, so
 * the reward signal is about partial work. Too long and a fraction of rollouts never
 * terminate, so the group never completes and the optimisation step hangs.
 *
 * The fix is a hard cap plus an explicit "no EOS" penalty, and — the part that is
 * usually missed — the cap must be *well* inside the model's context, because a
 * rollout that hits the cap has no room for the reward computation or a retry.
 */
export function rolloutHeadroom(c: RolloutConfig, capFrac = 0.75) {
  const cap = Math.floor(c.seqLen * capFrac);
  const prompt = Math.floor(c.seqLen * (1 - c.genFrac));
  return {
    cap,
    prompt,
    genBudget: cap - prompt,
    /** Tokens of slack left for the reward model or a retry. */
    slack: c.seqLen - cap,
    /** Generation stops this close to the model's context limit. */
    stopsAt: (cap / c.seqLen) * 100,
  };
}

// ═════════════════════════ 6 · diffusion ═════════════════════════

/**
 * Diffusion inverts noise into an image in two halves, and the halves have completely
 * different characters.
 *
 *   - **Denoising** runs the U-Net (or DiT) *many* times — 20 to 50 steps is typical.
 *     Each step processes the whole image at once. That is a big, dense, compute-bound
 *     GEMM workload: exactly like prefill, not like decode.
 *   - **VAE decoding** runs *once*, turning the final latent into pixels. It is one
 *     huge convolution over high-resolution feature maps and it is frequently the
 *     single most memory-hungry kernel in the stack.
 *
 * Stable Diffusion 1.5 at 512x512 is the reference: a 860M-parameter U-Net operating
 * on 4-channel 64x64 latents (a factor-8 downsample of the image), then a decoder that
 * turns that into 3x512x512 pixels.
 */
export interface DiffusionModel {
  id: string;
  name: string;
  /** U-Net / DiT parameters. */
  unetParams: number;
  /** Text encoder parameters (CLIP). */
  textParams: number;
  /** VAE decoder parameters. */
  vaeParams: number;
  /** Latent channels. */
  latentChannels: number;
  /** Spatial downsample from image to latent. */
  downsample: number;
  /** Default step count. */
  steps: number;
  /**
   * Published throughput on an A100 at batch 1, in seconds per image, with
   * classifier-free guidance on. Model-card and community benchmark figures; treat as
   * approximate, since they depend on the sampler and the attention backend. The
   * point is the *ratio* between models, not the absolute number.
   */
  refSeconds: number;
  /** Approximate peak VRAM to run it at 512px, from model cards. */
  vramGB: number;
}

export const DIFFUSION_MODELS: DiffusionModel[] = [
  {
    id: 'sd15',
    name: 'Stable Diffusion 1.5',
    unetParams: 0.86e9,
    textParams: 0.12e9,
    vaeParams: 0.05e9,
    latentChannels: 4,
    downsample: 8,
    steps: 20,
    refSeconds: 1.2,
    vramGB: 4.0,
  },
  {
    id: 'sd21',
    name: 'SD 2.1',
    unetParams: 0.97e9,
    textParams: 0.34e9,
    vaeParams: 0.05e9,
    latentChannels: 4,
    downsample: 8,
    steps: 25,
    refSeconds: 1.8,
    vramGB: 5.0,
  },
  {
    id: 'sdxl',
    name: 'SDXL base',
    unetParams: 2.6e9,
    textParams: 0.82e9,
    vaeParams: 0.08e9,
    latentChannels: 4,
    downsample: 8,
    steps: 30,
    refSeconds: 4.5,
    vramGB: 8.0,
  },
  {
    // Distilled few-step models: the same U-Net, trained to be usable in 1-4 steps.
    // This is the single biggest latency lever in the ecosystem, and it is a
    // training-time decision, not a serving one.
    id: 'sdxl-turbo',
    name: 'SDXL-Turbo',
    unetParams: 2.6e9,
    textParams: 0.82e9,
    vaeParams: 0.08e9,
    latentChannels: 4,
    downsample: 8,
    steps: 4,
    refSeconds: 0.8,
    vramGB: 8.0,
  },
];

export const diffusionById = (id: string): DiffusionModel => DIFFUSION_MODELS.find((m) => m.id === id)!;

/**
 * The memory picture. Diffusion has no KV cache — and it is still the workload where
 * activations, not weights, decide whether you fit.
 *
 * The dominant term is the U-Net's widest feature map. At 512x512 the highest-
 * resolution blocks carry 320 channels, so one tensor is 320 x 64 x 64 x 2 B =
 * 2.62 MB, and a handful of them live at once for the backward-free inference pass.
 * Per image that is on the order of a gigabyte — *more than the weights themselves*.
 */
export interface DiffusionBatch {
  /** Image height and width in pixels. */
  size: number;
  /** Images processed per forward. */
  batch: number;
  /** Latent side length: size / downsample. */
  latentSide: number;
  /** Latent values per image: channels x side x side. */
  latentTokens: number;
  /** One widest-layer feature map, in bytes. */
  widestTensor: number;
  /** Activation bytes per image across the live tensors. */
  actBytesPerImage: number;
  /** Activation bytes for the whole batch. */
  actBytes: number;
  /** Weight bytes for U-Net + VAE + text encoder. */
  weightBytes: number;
  /** Total bytes needed. */
  totalBytes: number;
  /** Largest batch that fits on this card. */
  maxBatch: number;
  fits: boolean;
}

export function diffusionBatch(m: DiffusionModel, g: Gpu, size = 512, batch = 4): DiffusionBatch {
  const latentSide = size / m.downsample;
  const latentTokens = m.latentChannels * latentSide * latentSide;
  // 320 channels at full resolution, FP16.
  const widestTensor = 320 * latentSide * latentSide * 2;
  // ~6 live tensors at the widest stage is a reasonable inference-pass footprint.
  const actBytesPerImage = widestTensor * 6;
  const weightBytes = (m.unetParams + m.vaeParams + m.textParams) * 2;
  const totalBytes = weightBytes + actBytesPerImage * batch;
  return {
    size,
    batch,
    latentSide,
    latentTokens,
    widestTensor,
    actBytesPerImage,
    actBytes: actBytesPerImage * batch,
    weightBytes,
    totalBytes,
    maxBatch: Math.max(1, Math.floor((g.mem - weightBytes) / actBytesPerImage)),
    fits: totalBytes <= g.mem,
  };
}

/**
 * Where one denoising step sits relative to the roofline.
 *
 * Diffusion has no KV cache and no autoregressive loop — there is nothing to reuse,
 * so every one of the 20 to 50 steps re-reads every weight. What it *does* have is a
 * large activation footprint, and that is what puts it firmly on the compute side.
 *
 * Rather than assert a FLOP count (which depends on details of the U-Net that vary a
 * lot between checkpoints), we compute the honest, directly measurable quantity:
 *
 *   bandwidth utilisation = bytes per step / (step time x peak bandwidth)
 *
 * If a step moved its bytes at full bandwidth it would be memory-bound. At roughly 2%
 * of peak it is nowhere near — so the step is compute-bound, and the compute units are
 * the constraint. That is the whole contrast with LLM decode, which sits at 60-80%
 * of peak bandwidth and is therefore bound by it.
 */
export interface DiffusionStep {
  /** Bytes one denoising step touches: every weight plus the live activations. */
  bytesPerStep: number;
  /** Seconds per step (batch 1, guidance on, calibrated to published A100 timings). */
  stepS: number;
  /** ms per step. */
  stepMs: number;
  /** Seconds for the whole sampling loop at batch 1, including the VAE. */
  loopS: number;
  /** Seconds for the one-off VAE decode. */
  vaeS: number;
  /** Fraction of peak bandwidth the step actually achieves. */
  bandwidthUtil: number;
  /** Seconds to move the same bytes at 100% of peak bandwidth. */
  memFloorS: number;
  /** The GPU's ridge point: FLOP/s per byte/s. */
  ridge: number;
  bound: 'compute' | 'memory';
  /** Images per second per GPU at this batch. */
  ipsPerGpu: number;
  /** Seconds per image, whole loop. */
  secPerImage: number;
}

/**
 * Calibrate a model against its published A100 time, then re-scale for GPU and
 * resolution.
 *
 * Resolution is the one thing that changes the FLOPs: a U-Net's work scales with pixel
 * count, so 1024x1024 costs about 4x a 512x512 image. Batch is different — while the
 * step is compute-bound, adding images to it is nearly free until the activations stop
 * fitting, which is what diffusionBatch.maxBatch reports.
 */
export function diffusionStep(m: DiffusionModel, g: Gpu, size = 512, batch = 4): DiffusionStep {
  const ref = diffusionStepRef(m);

  // The published reference is an A100, so rescale by the ratio of peak FLOP/s.
  const gpuFactor = g.flops / 312e12;
  const pxFactor = (size * size) / (512 * 512);
  const stepS = ref.stepS * pxFactor / gpuFactor;
  const loopS = stepS * m.steps + ref.vaeS;

  // Bytes touched by a step: every weight (it is re-read each time) plus the widest
  // live activation tensors, which are what actually fills the card.
  const bytesPerStep = m.unetParams * 2 + 320 * (size / m.downsample) * (size / m.downsample) * 2 * 6;
  // What the same bytes would cost at 100% of peak bandwidth.
  const memFloorS = bytesPerStep / g.bw;
  const bandwidthUtil = memFloorS / stepS;

  return {
    bytesPerStep,
    stepS,
    stepMs: stepS * 1e3,
    loopS,
    vaeS: ref.vaeS,
    bandwidthUtil,
    memFloorS,
    ridge: g.flops / g.bw,
    bound: bandwidthUtil > 0.5 ? 'memory' : 'compute',
    // Batch is nearly free while compute-bound, so images per second scales with it
    // right up until the activations stop fitting (see diffusionBatch.maxBatch).
    ipsPerGpu: batch / loopS,
    secPerImage: loopS / batch,
  };
}

/**
 * The reference timing: published A100 seconds per image, with classifier-free
 * guidance already on, split into the per-step cost and the one-off VAE decode.
 */
export function diffusionStepRef(m: DiffusionModel) {
  const total = m.refSeconds;
  // The VAE decode is a small, fixed slice of wall-clock — call it 8%.
  const vaeS = total * 0.08;
  return { totalS: total, vaeS, stepS: (total - vaeS) / m.steps };
}

/** CFG: classifier-free guidance runs the U-Net twice per step, batched together. */
export function cfgCost(m: DiffusionModel, g: Gpu, size = 512) {
  const one = diffusionStep(m, g, size, 1);
  return {
    /** Two forward passes instead of one: the step time doubles. */
    multiplier: 2,
    overheadFrac: 1,
    baselineMs: one.stepMs * m.steps,
    guidedMs: one.stepMs * m.steps * 2,
    vaeMs: one.vaeS * 1e3,
    steps: m.steps,
    note: 'Published reference times already include guidance.',
  };
}

/**
 * Steps against latency. Step count is *linear* in latency — the simplest and largest
 * lever there is — but the quality curve flattens: going from 50 steps to 30 changes
 * the image less than going from 30 to 4, which is exactly why distilled few-step
 * models are worth the retraining.
 */
export function stepQualityTradeoff(m: DiffusionModel, g: Gpu, size = 512) {
  const ref = diffusionStepRef(m);
  return [1, 2, 4, 8, 15, 20, 30, 50].map((steps) => {
    const s = steps * ref.stepS + ref.vaeS;
    return { steps, s, ms: s * 1e3, ips: 1 / s, saved: 1 - steps / m.steps };
  });
}

/**
 * Diffusion has no prefix cache, so caching means caching *within* the loop: later
 * denoising steps see a latent that changes very little, so the U-Net's early-layer
 * features barely move and can be reused. These are the published retain fractions.
 */
export const CACHE_MODES = [
  { id: 'none', label: 'No cache', retain: 1.0, note: 'every step is a full forward pass' },
  { id: 'cumulative', label: 'Cumulative', retain: 0.35, note: 'reuse a growing prefix of layers' },
  { id: 'residual', label: 'Residual', retain: 0.15, note: 'reuse the shallow feature deltas' },
] as const;