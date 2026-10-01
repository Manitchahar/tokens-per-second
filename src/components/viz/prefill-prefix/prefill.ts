// Module 10 · Prefill & Prefix Optimization — shared math.
//
// Everything here is derived in the module's prose. Nothing is magic: each
// function is the cheapest version of the formula, using only numbers the
// learner already owns (Module 05's KV math, Module 02's roofline).
import { modelById, gpuById, kvBytesPerToken, type Model, type Gpu } from '../lib/perfModel';

const llama8b = modelById('llama8b');
const llama70b = modelById('llama70b');

/** Bytes of KV cache one sequence occupies at `ctx` tokens, in BF16. */
export const seqKvBytes = (ctx: number, model: Model = llama8b) => ctx * kvBytesPerToken(model, 2);

/** A chat prompt wrapped in a chat template, in tokens. Realistic, measured, approximate. */
export const chatTokens = (text: number, turns = 1) => 16 + text + 73 * turns;

/**
 * Prefill time for `n` prompt tokens, in seconds.
 *
 *   compute term: 2 * params * n FLOPs / (peak FLOP/s * 0.55)
 *   memory term:  params * 2 bytes / (bandwidth * 0.8)   — the weights, read once
 *
 * Prefill is compute-bound from a few hundred tokens upward, so for any prompt
 * worth measuring the compute term is the one you get.
 */
export function prefillSec(n: number, model: Model = llama8b, gpu: Gpu = gpuById('h100')) {
  const flops = 2 * model.params * n;
  const wBytes = model.params * 2;
  const tCompute = flops / (gpu.flops * 0.55);
  const tMem = wBytes / (gpu.bw * 0.8);
  return { t: Math.max(tCompute, tMem), tCompute, tMem, flops };
}

/** Decode seconds for one step, from the roofline terms Module 02 gave us. */
export function decodeSec(batch: number, ctx: number, model: Model = llama8b, gpu: Gpu = gpuById('h100')) {
  const weights = model.params * 2;
  const kv = batch * seqKvBytes(ctx, model);
  const tMem = (weights + kv) / (gpu.bw * 0.8);
  const tCompute = (2 * model.params * batch) / (gpu.flops * 0.55);
  return { t: Math.max(tMem, tCompute) + 0.0015, tMem, tCompute };
}

/** Per-user tokens per second during the decode phase. */
export const decodeRate = (batch: number, ctx: number, model?: Model, gpu?: Gpu) => 1 / decodeSec(batch, ctx, model, gpu).t;

/** How many users a GPU can hold at this context length, from KV memory alone. */
export function maxUsers(ctx: number, gpu: Gpu = gpuById('h100'), model: Model = llama8b, reserveFrac = 0.04) {
  const free = gpu.mem - model.params * 2 - reserveFrac * gpu.mem;
  return Math.floor(free / seqKvBytes(ctx, model));
}

/** Bytes a naive full-attention implementation must materialize for T tokens, one sequence. */
export const attentionMatrixBytes = (t: number, bytes = 2) => t * t * bytes;

export { llama8b, llama70b };