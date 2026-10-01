// The arithmetic of Module 05, in one place, so every visual in the module quotes the same numbers.
// Everything here is derived in the prose: KV cache bytes, attention FLOPs, and the waste of naive generation.

import type { Model } from '../lib/perfModel';
import { decodeStep, maxBatch, prefillTime } from '../lib/perfModel';

/** Bytes of KV cache one token of context costs, for one sequence, at `b` bytes per number. */
export const kvPerToken = (m: Model, b = 2) => 2 * m.layers * m.kvHeads * m.headDim * b;

/** The same, for a model with no grouping: one key/value head per query head. */
export const kvPerTokenMHA = (m: Model, b = 2) => 2 * m.layers * m.heads * m.headDim * b;

/** How many query heads share one key/value head. */
export const group = (m: Model) => m.heads / m.kvHeads;

/** Bytes of KV cache for one whole conversation of `ctx` tokens. */
export const kvPerSeq = (m: Model, ctx: number, b = 2) => ctx * kvPerToken(m, b);

/** Bytes of weights at `b` bytes per parameter. */
export const weightBytes = (m: Model, b = 2) => m.params * b;

/**
 * Attention FLOPs for a sequence of `t` tokens, summed over layers.
 * Per layer: scores are [H, t, t] = 2·H·t²·D, and the weighted sum is the same again.
 * Grouping (GQA) shrinks the KV *cache* but not the attention math, because the number of
 * query heads is unchanged.
 */
export const attnFlops = (m: Model, t: number) => 4 * t * t * m.hidden * m.layers;

/** The linear part: every weight touched once per token, 2 FLOPs per multiply-add. */
export const linearFlops = (m: Model, t: number) => 2 * m.params * t;

/** Context length at which quadratic attention costs as much FLOPs as the linear part. */
export const attnCrossover = (m: Model) => m.params / (2 * m.hidden * m.layers);

/** FLOPs of one forward pass over a whole sequence, attention included. */
export const forwardFlops = (m: Model, t: number) => linearFlops(m, t) + attnFlops(m, t);

/**
 * Token-forwards done by a naive generator: at step i it pushes the whole sequence
 * (p + i tokens) through the model again.
 */
export const naiveForwards = (p: number, n: number) => n * p + (n * (n - 1)) / 2;

/** Token-forwards done with a KV cache: the prompt once, then one token per step. */
export const cachedForwards = (p: number, n: number) => p + n;

/** Bytes of KV read by one decode step at context `ctx`, for `batch` sequences. */
export const kvRead = (m: Model, batch: number, ctx: number, b = 2) => batch * kvPerSeq(m, ctx, b);

/** Context length at which reading the KV cache costs more than reading the weights. */
export const kvBeatsWeights = (m: Model, wB = 2, kvB = 2) => weightBytes(m, wB) / kvPerToken(m, kvB);

/**
 * How many context tokens a GPU can hold in total, once the weights and a workspace
 * reserve are paid for. This is the single most useful capacity number in serving.
 */
export function contextTokens(
  m: Model,
  gpuMem: number,
  wB = 2,
  kvB = 2,
  reserveFrac = 0.1,
) {
  const free = gpuMem * (1 - reserveFrac) - weightBytes(m, wB);
  return Math.max(0, Math.floor(free / kvPerToken(m, kvB)));
}

/** Largest batch that fits at this context, using the shared perf model. */
export const batchThatFits = (m: Model, gpuMem: number, bw: number, flops: number, ctx: number, wB = 2, kvB = 2) =>
  maxBatch({ model: m, gpu: { id: '', name: '', mem: gpuMem, bw, flops, price: 0 }, batch: 1, ctx, wBytes: wB, kvBytes: kvB });

/** Re-exported so a visual needs one import. */
export { decodeStep, prefillTime };

/** "128 KB" / "1.5 GB" style byte strings that match the prose exactly. */
export function kb(bytes: number) {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : bytes >= 1e6 ? `${(bytes / 1e6).toFixed(0)} MB` : `${Math.round(bytes / 1e3)} KB`;
}
