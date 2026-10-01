// MoE arithmetic, shared by the module's visuals.
//
// Model shapes come from published configuration files / model cards, and each is
// labelled with the parameter figures the published descriptions quote. Where the
// figures disagree with a model card it is because counts here are approximate
// (tied vs untied embeddings, exact intermediate widths) — always say "approximate".

import { GPUS, type Gpu } from '../lib/perfModel';

export interface MoeModel {
  id: string;
  name: string;
  /** Transformer blocks. */
  layers: number;
  /** Model width (the size of the residual stream). */
  hidden: number;
  /** FFN intermediate width in a *dense* block — the reference the MoE replaces. */
  denseIntermediate: number;
  /** FFN intermediate width *inside one expert*. */
  expertIntermediate: number;
  /** Routed experts per MoE layer. */
  experts: number;
  /** Experts activated per token. */
  topK: number;
  /** Always-on experts present on every token, every layer (Mixtral has none). */
  shared: number;
  /** Index of the first MoE block; layers before it are dense. 0 = all blocks are MoE. */
  moeFirstLayer: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  vocab: number;
  /** Total parameters the published description quotes, for checking the arithmetic. */
  quotedTotal: number;
  /** Active parameters the published description quotes. */
  quotedActive: number;
}

export const MOE_MODELS: MoeModel[] = [
  {
    id: 'mixtral',
    name: 'Mixtral 8x7B',
    layers: 32,
    hidden: 4096,
    denseIntermediate: 14336,
    expertIntermediate: 14336,
    experts: 8,
    topK: 2,
    shared: 0,
    moeFirstLayer: 1,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    vocab: 32000,
    quotedTotal: 46.7e9,
    quotedActive: 12.9e9,
  },
  {
    id: 'deepseek16',
    name: 'DeepSeek-MoE-16B',
    layers: 27,
    hidden: 2048,
    denseIntermediate: 5632,
    expertIntermediate: 1408,
    experts: 64,
    topK: 6,
    shared: 2,
    moeFirstLayer: 1,
    heads: 32,
    kvHeads: 32,
    headDim: 128,
    vocab: 102400,
    quotedTotal: 16.4e9,
    quotedActive: 2.8e9,
  },
  {
    id: 'qwen30',
    name: 'Qwen3-30B-A3B',
    layers: 48,
    hidden: 2048,
    denseIntermediate: 6144,
    expertIntermediate: 768,
    experts: 128,
    topK: 8,
    shared: 1,
    moeFirstLayer: 0,
    heads: 32,
    kvHeads: 4,
    headDim: 128,
    vocab: 151936,
    quotedTotal: 30.5e9,
    quotedActive: 3.3e9,
  },
];

export const moeById = (id: string): MoeModel => MOE_MODELS.find((m) => m.id === id)!;

/** How many blocks carry routed experts. */
export const moeLayerCount = (m: MoeModel): number => m.layers - m.moeFirstLayer;

/** Parameters in one MoE block's bank of routed experts: three matrices per expert. */
export const expertBankParams = (m: MoeModel): number => 3 * m.experts * m.hidden * m.expertIntermediate;

/** Parameters in one block's always-on shared experts. */
export const sharedBankParams = (m: MoeModel): number => (m.shared ? 3 * m.shared * m.hidden * m.expertIntermediate : 0);

/** The router itself: one score vector per expert over the hidden width. */
export const routerParams = (m: MoeModel): number => (m.experts + m.shared + 1) * m.hidden;

/** Attention across the whole model, plus the embedding table. */
export function denseParams(m: MoeModel): number {
  const attn = m.layers * (2 * m.hidden * m.heads * m.headDim + 2 * m.hidden * m.kvHeads * m.headDim);
  return attn + m.vocab * m.hidden;
}

/**
 * Total parameters — everything that must live in GPU memory, and everything a
 * dense trainer would have to touch every step.
 *
 * The key structural fact: an MoE block's feed-forward network *is* the expert
 * bank. There is no separate dense FFN alongside it, so the dense width only
 * appears in the few layers that were left dense.
 */
export function totalParams(m: MoeModel): number {
  const nMoe = moeLayerCount(m);
  const densePerBlock = 3 * m.hidden * m.denseIntermediate;
  const perMoe = expertBankParams(m) + sharedBankParams(m) + routerParams(m);
  return denseParams(m) + nMoe * perMoe + (m.layers - nMoe) * densePerBlock;
}

/**
 * Active parameters per token: what one token actually reads.
 * Only the top-k routed experts count; shared experts count in full.
 */
export function activeParams(m: MoeModel): number {
  const nMoe = moeLayerCount(m);
  const densePerBlock = 3 * m.hidden * m.denseIntermediate;
  const perMoe = 3 * m.topK * m.hidden * m.expertIntermediate + sharedBankParams(m) + routerParams(m);
  return denseParams(m) + nMoe * perMoe + (m.layers - nMoe) * densePerBlock;
}

/** The headline ratio: total parameters divided by active parameters. */
export const sparsity = (m: MoeModel): number => totalParams(m) / activeParams(m);

/** The dense model with the same depth, width and vocabulary, for comparison. */
export function denseTwin(m: MoeModel): { name: string; params: number } {
  const densePerBlock = 3 * m.hidden * m.denseIntermediate;
  return { name: `${m.name} built dense`, params: denseParams(m) + m.layers * densePerBlock };
}

// ───────────────────────── decode step ─────────────────────────

export interface StepOpts {
  gpu: string;
  /** Bytes per weight: 2 = BF16, 1 = FP8/INT8, 0.5 = 4-bit. */
  wBytes?: number;
  /** Tokens of context per sequence. */
  ctx?: number;
  /** Sequences decoded together (continuous batching). */
  batch?: number;
  mbu?: number;
  mfu?: number;
}

export interface MoeCost {
  activeBytes: number;
  weightBytes: number;
  stepMs: number;
  bound: 'memory' | 'compute';
  /** GPUs the whole weight set needs at this precision. */
  fitsOn: number;
  /** Fraction of the weight read a token never touches. */
  skipFrac: number;
}

export const gpu = (id: string): Gpu => GPUS.find((g) => g.id === id) ?? GPUS[3];

/** Bytes of KV cache per token for a model — the same 2 L Hkv D formula as before. */
export const kvBytesPerToken = (m: MoeModel): number => 2 * m.layers * m.kvHeads * m.headDim * 2;

/**
 * One decode step. Same structure as the decode-step model from numbers-and-math,
 * with the MoE's *active* parameter count doing the reading. That single
 * substitution is the entire latency story of mixture-of-experts.
 */
export function moeStep(m: MoeModel, o: StepOpts): MoeCost {
  const g = gpu(o.gpu);
  const wBytes = o.wBytes ?? 2;
  const ctx = o.ctx ?? 4096;
  const batch = o.batch ?? 32;
  const mbu = o.mbu ?? 0.8;
  const mfu = o.mfu ?? 0.55;

  const total = totalParams(m);
  const active = activeParams(m);
  const weightBytes = total * wBytes;
  const activeBytes = active * wBytes + batch * ctx * kvBytesPerToken(m);
  const flops = 2 * active * batch;
  const tMem = activeBytes / (g.bw * mbu);
  const tComp = flops / (g.flops * mfu);
  return {
    activeBytes,
    weightBytes,
    stepMs: Math.max(tMem, tComp) * 1e3,
    bound: tMem > tComp ? 'memory' : 'compute',
    fitsOn: Math.ceil(weightBytes / g.mem),
    skipFrac: 1 - active / total,
  };
}

// ───────────────────────── routing simulation ─────────────────────────

/** Deterministic pseudo-random in [0,1). No Math.random, so server and client agree. */
export function rand(seed: number): number {
  const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

/** How peaked the router is. 0 = uniform (no specialisation), higher = sharper preference. */
export const CONCENTRATION = [
  { id: 'flat', label: 'Uniform', value: 0.0, blurb: 'every expert identical — no specialisation to exploit' },
  { id: 'mild', label: 'Mild', value: 0.55, blurb: 'a weak preference, as in a lightly trained MoE' },
  { id: 'sharp', label: 'Sharp', value: 1.3, blurb: 'the specialisation a trained MoE usually shows' },
  { id: 'very', label: 'Very sharp', value: 2.4, blurb: 'a few experts swallow nearly everything' },
] as const;

export type ConcentrationId = (typeof CONCENTRATION)[number]['id'];

/**
 * Route `tokens` tokens to `topK` distinct experts each, with a Zipf-like expert
 * preference. Deterministic in (tokens, experts, topK, concentration, seed).
 */
export function route(tokens: number, experts: number, topK: number, concentration: number, seed = 0): number[] {
  const w: number[] = [];
  let norm = 0;
  for (let i = 0; i < experts; i++) {
    const v = Math.pow(i + 1, -concentration);
    w.push(v);
    norm += v;
  }
  const cum: number[] = [];
  let acc = 0;
  for (const v of w) { acc += v / norm; cum.push(acc); }

  const counts = new Array(experts).fill(0);
  for (let t = 0; t < tokens; t++) {
    const chosen = new Set<number>();
    for (let k = 0; k < topK; k++) {
      const target = rand(seed * 131 + t * 17 + k * 7 + 3);
      // walk the cumulative table, skipping experts already taken this token
      let pick = -1;
      for (let i = 0; i < experts; i++) {
        if (!chosen.has(i) && cum[i] >= target) { pick = i; break; }
      }
      if (pick === -1) {
        for (let i = experts - 1; i >= 0; i--) if (!chosen.has(i)) { pick = i; break; }
      }
      if (pick >= 0) { chosen.add(pick); counts[pick]++; }
    }
  }
  return counts;
}

export interface Routing {
  counts: number[];
  fracs: number[];
  maxFrac: number;
  minFrac: number;
  /** maxFrac divided by the ideal share. 1.0 is perfectly balanced. */
  imbalance: number;
  /** Tokens a hard per-expert capacity of `capTokens` would drop. */
  dropped: number;
  hottest: number;
  /** Index of the coldest expert. */
  coldest: number;
}

/** Capacity factor: hard slots per expert, as a multiple of the perfectly balanced share. */
export function capacityTokens(tokens: number, experts: number, topK: number, capacityFactor: number): number {
  return Math.ceil(capacityFactor * ((topK * tokens) / experts));
}

export function summarise(counts: number[], tokens: number, experts: number, topK: number, capacityFactor: number): Routing {
  const fracs = counts.map((c) => c / tokens);
  const ideal = topK / experts;
  const cap = capacityTokens(tokens, experts, topK, capacityFactor);
  let dropped = 0;
  for (const c of counts) if (c > cap) dropped += c - cap;
  const hot = counts.indexOf(Math.max(...counts));
  const cold = counts.indexOf(Math.min(...counts));
  return {
    counts,
    fracs,
    maxFrac: Math.max(...fracs),
    minFrac: Math.min(...fracs),
    imbalance: Math.max(...fracs) / ideal,
    dropped,
    hottest: hot,
    coldest: cold,
  };
}

// ───────────────────────── expert-parallel communication ─────────────────────────

export interface NetConfig {
  /** Cards in the expert-parallel group. */
  gpus: number;
  /** Per-link bandwidth, bytes/s. Approximate figures for the named interconnect. */
  linkBw: number;
  label: string;
  short: string;
}

export const NETS: NetConfig[] = [
  { gpus: 2, linkBw: 25e9, label: 'PCIe 4.0 x16, ~25 GB/s per link', short: 'PCIe 4.0' },
  { gpus: 4, linkBw: 25e9, label: 'PCIe 4.0 x16, ~25 GB/s per link', short: 'PCIe 4.0' },
  { gpus: 8, linkBw: 300e9, label: 'NVLink 3 on 8x H100, ~300 GB/s bidirectional', short: 'NVLink 3' },
  { gpus: 16, linkBw: 450e9, label: 'NVLink 3 on 16x H100, ~450 GB/s bidirectional', short: 'NVLink 3' },
];

export interface CommCost {
  /** Bytes handed to experts that live on another card. */
  dispatched: number;
  /** Bytes those experts hand back. */
  returned: number;
  /** Both directions, all MoE layers. */
  totalBytes: number;
  /** Time for the dispatch half alone. */
  dispatchMs: number;
  /** Share of expert activations that cross a card boundary. */
  fractionRemote: number;
}

/**
 * All-to-all cost for expert parallelism.
 *
 * With `g` cards and perfectly balanced routing, each card keeps 1/g of its
 * tokens' expert activations local and sends the rest. Imbalance makes it worse:
 * when one expert is hot, the tokens aimed at it pile onto whichever card owns it,
 * so a larger share of the batch has to travel. `imbalance` is the same
 * max/ideal load ratio the routing simulator reports.
 */
export function allToAll(m: MoeModel, net: NetConfig, tokens: number, imbalance = 1): CommCost {
  const g = net.gpus;
  const fracRemote = Math.min(1, (1 - 1 / g) * imbalance);
  const perToken = m.hidden * 2; // one BF16 activation vector in, one out
  const dispatched = tokens * fracRemote * perToken;
  const totalBytes = 2 * dispatched * moeLayerCount(m);
  // An all-to-all is a ring, not a memcpy: roughly half the achievable link rate.
  const eff = net.linkBw * 0.55;
  return {
    dispatched,
    returned: dispatched,
    totalBytes,
    dispatchMs: (dispatched / eff) * 1e3,
    fractionRemote: fracRemote,
  };
}

/** 46.7B etc. */
export function B(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n / 1e9 >= 10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n / 1e6 >= 100 ? 0 : 1)}M`;
  return `${Math.round(n / 1e3)}K`;
}