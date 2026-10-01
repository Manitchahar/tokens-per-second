// Everything Module 03 derives about a real model from its published config.
// Layer dimensions, parameter counts, FLOPs and bytes — nothing invented here.
// Source of truth for GPU and model numbers is ../lib/perfModel; this file only adds
// the transformer-specific fields (MLP width, norm placement, RoPE base, weight tying)
// and the arithmetic that turns a config into a bill.

import { MODELS, modelById, kvBytesPerToken, type Model } from '../lib/perfModel';

export type ModelId = 'qwen05' | 'llama3b' | 'llama8b' | 'qwen32b' | 'llama70b';

export interface Arch {
  /** MLP inner width (Llama/Qwen: gate_proj and up_proj output this many). */
  inter: number;
  /** RoPE base (θ). Llama-3.1 raised it from 10,000 to 500,000. */
  ropeBase: number;
  /** Qwen2 adds an RMSNorm on q and k before RoPE; Llama does not. */
  qkNorm: boolean;
  /** Whether the output projection reuses the embedding table. */
  tiedEmbed: boolean;
  /** Short public spec-sheet note shown in the visuals. */
  note: string;
}

/** Only the fields perfModel does not already carry. */
export const ARCH: Record<ModelId, Arch> = {
  qwen05: { inter: 4864, ropeBase: 1e6, qkNorm: true, tiedEmbed: true, note: 'small, 7 query heads per KV head' },
  llama3b: { inter: 8192, ropeBase: 500000, qkNorm: false, tiedEmbed: true, note: 'tied embeddings' },
  llama8b: { inter: 14336, ropeBase: 500000, qkNorm: false, tiedEmbed: false, note: 'the workhorse of this course' },
  qwen32b: { inter: 27648, ropeBase: 1e6, qkNorm: true, tiedEmbed: false, note: '40 query heads, 8 KV heads' },
  llama70b: { inter: 28672, ropeBase: 500000, qkNorm: false, tiedEmbed: false, note: '80 layers, 8192 wide' },
};

export const modelIds: ModelId[] = ['qwen05', 'llama3b', 'llama8b', 'qwen32b', 'llama70b'];

export function arch(id: ModelId) {
  return { ...modelById(id), ...ARCH[id] } as Model & Arch;
}

/** Bytes per stored weight. 2 = BF16/FP16, the default for inference. */
export const W_BYTES = 2;

/* ─────────────── parameter counts, derived from shapes ─────────────── */

/** Query, key, value and output projections in one layer. */
export const attnParams = (m: Model & Arch) =>
  m.hidden * m.hidden + // q_proj:  d x d
  2 * m.hidden * (m.kvHeads * m.headDim) + // k_proj, v_proj:  d x (n_kv d_h)
  m.hidden * m.hidden; // o_proj:  d x d

/** gate_proj, up_proj, down_proj. */
export const mlpParams = (m: Model & Arch) => 3 * m.hidden * m.inter;

/** One RMSNorm gain per norm; Llama and Qwen2 use two per block plus one final. */
export const normParams = (m: Model & Arch) => 3 * m.hidden + (m.qkNorm ? 2 * m.kvHeads * m.headDim : 0);

export const perLayerParams = (m: Model & Arch) => attnParams(m) + mlpParams(m) + normParams(m);

export const embedParams = (m: Model & Arch) => m.vocab * m.hidden; // the lookup table
export const headParams = (m: Model & Arch) => (m.tiedEmbed ? 0 : m.vocab * m.hidden); // lm_head

export const derivedParams = (m: Model & Arch) =>
  m.layers * perLayerParams(m) + embedParams(m) + headParams(m);

/** Bytes of parameters. What a decode step must read, every step. */
export const weightBytes = (m: Model & Arch, bytes = W_BYTES) => m.params * bytes;

/* ─────────────── FLOPs, per generated token ─────────────── */

/** One multiply plus one add per weight. */
export const linearFlopsPerToken = (m: Model & Arch) => 2 * (m.layers * perLayerParams(m) + headParams(m));

/**
 * Attention FLOPs for one query token against a context of T cached tokens.
 * Q·Kᵀ is 2·T·d_model, and (attn)·V is the same again: 4·T·d_model per layer.
 * Independent of how many heads there are — heads split the work, they do not add to it.
 */
export const attnFlopsPerToken = (m: Model & Arch, ctx: number) => 4 * ctx * m.hidden * m.layers;

/**
 * Attention FLOPs for processing a whole prompt of T tokens at once (prefill).
 * Only the lower triangle exists under a causal mask, so it is half of the T×T square.
 */
export const prefillAttnFlops = (m: Model & Arch, tokens: number) => 2 * tokens * tokens * m.hidden * m.layers;

/**
 * Context length at which attention costs as many FLOPs as every weight matrix
 * combined: 2T²·d·L = 2·T·P_layer  ⟹  T = P_layer / (d · L).
 */
export const attnCrossoverTokens = (m: Model & Arch) => m.layers * perLayerParams(m) / (m.hidden * m.layers);

/* ─────────────── attention heads ─────────────── */

/** How many query heads share one KV head. 1 = MHA (plain), qh = MQA. */
export const groupSize = (m: Model & Arch) => m.heads / m.kvHeads;

/** Query heads (dimension d_model split into `heads` slices of `headDim`). */
export const qHeads = (m: Model & Arch) => m.heads;
/** KV heads: only `kvHeads` slices of `headDim`, so the cache is heads/kvHeads times smaller. */
export const kvHeads = (m: Model & Arch) => m.kvHeads;

/* ─────────────── KV cache ─────────────── */

export { kvBytesPerToken };

/** KV cache for one sequence of T tokens, in bytes. */
export const kvBytes = (m: Model & Arch, ctx: number, bytes = W_BYTES) => ctx * kvBytesPerToken(m, bytes);

/** The same cache as a fraction of the weights, and of a GPU's memory. */
export const kvShare = (m: Model & Arch, ctx: number) => kvBytes(m, ctx) / weightBytes(m);

/* ─────────────── the residual stream ─────────────── */

/** How many separate vectors are summed into one token's stream by the whole model. */
export const streamWrites = (m: Model & Arch) => 1 + 2 * m.layers; // embedding + 2 per block

/** Bytes of the stream read and written per sublayer visit. */
export const streamBytes = (m: Model & Arch) => m.hidden * W_BYTES;

export { MODELS };
export type { Model };
