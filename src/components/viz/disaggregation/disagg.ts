// Module 21 · Prefill/Decode Disaggregation — shared math.
//
// Every number here is derived in the module's prose from things the learner
// already owns: Module 02's roofline, Module 05's two-phase picture, Module 10's
// chunked prefill. Nothing is magic.
import { modelById, gpuById, kvBytesPerToken, type Model, type Gpu } from '../lib/perfModel';

export const m8 = modelById('llama8b');
export const m70 = modelById('llama70b');

/** Assumed fractions of the hardware ceilings, matching the rest of the course. */
export const MBU = 0.8;
export const MFU = 0.55;
/** Fixed per-step cost: kernel launches, scheduling, sampling. */
export const OVERHEAD = 0.0015;

/** Seconds of GPU time to prefill one prompt token. Compute-bound for any prompt worth measuring. */
export const prefillPerToken = (model: Model = m8, gpu: Gpu = gpuById('h100')) =>
  (2 * model.params) / (gpu.flops * MFU);

/** Seconds to prefill a whole prompt of `n` tokens. */
export const prefillSec = (n: number, model: Model = m8, gpu: Gpu = gpuById('h100')) =>
  prefillPerToken(model, gpu) * n;

/** Bytes of KV one sequence holds at `ctx` tokens, BF16. */
export const seqKvBytes = (ctx: number, model: Model = m8) => ctx * kvBytesPerToken(model, 2);

/** Bytes of KV that must cross the wire when a prefill instance hands off to a decode instance. */
export const handoffBytes = (ctx: number, model: Model = m8) => ctx * kvBytesPerToken(model, 2);

/**
 * One decode step. The whole point of the module is the gap between the two terms:
 * tMem saturates the bandwidth, tCompute is the FLOPs a big batch could do.
 */
export function decodeStep(batch: number, ctx: number, model: Model = m8, gpu: Gpu = gpuById('h100')) {
  const weights = model.params * 2;
  const kv = batch * seqKvBytes(ctx, model);
  const tWeights = weights / (gpu.bw * MBU);
  const tKv = kv / (gpu.bw * MBU);
  const tMem = tWeights + tKv;
  const tCompute = (2 * model.params * batch) / (gpu.flops * MFU);
  const t = Math.max(tMem, tCompute) + OVERHEAD;
  const activations = 0.04 * gpu.mem;
  const memUsed = weights + kv + activations;
  return {
    t, tWeights, tKv, tMem, tCompute, weights, kv,
    perUser: 1 / t,          // tok/s for one user in this batch
    total: batch / t,         // tok/s for the whole fleet slot
    memUsed,
    fits: memUsed <= gpu.mem,
    /** How much of the decode step the bandwidth-limited term accounts for. ~1 means fully memory-bound. */
    memShare: tMem / t,
    /** How much of the decode step the FLOP-limited term accounts for. */
    computeShare: tCompute / t,
  };
}

/** Largest batch that fits in HBM at this context length: weights + KV + workspace. */
export function maxBatch(ctx: number, model: Model = m8, gpu: Gpu = gpuById('h100')) {
  const free = gpu.mem - model.params * 2 - 0.04 * gpu.mem;
  return Math.max(0, Math.floor(free / seqKvBytes(ctx, model)));
}

/** Seconds to move one request's KV across a link of `bw` bytes/s. */
export const transferSec = (ctx: number, bw: number, model: Model = m8) => handoffBytes(ctx, model) / bw;

/** Link speeds we compare, in bytes/s. Public approximate figures. */
export const LINKS: { id: string; label: string; bw: number }[] = [
  { id: 'nvlink', label: 'NVLink 4 (900 GB/s)', bw: 900e9 },
  { id: '400g', label: '8×400 GbE (400 GB/s)', bw: 400e9 },
  { id: '100g', label: '100 GbE (100 GB/s)', bw: 100e9 },
  { id: '25g', label: '25 GbE (25 GB/s)', bw: 25e9 },
];

/** The arithmetic ridge point: intensity at which a job stops being memory-bound. */
export const ridgePoint = (gpu: Gpu = gpuById('h100')) => gpu.flops / gpu.bw;

/**
 * Arithmetic intensity of a prefill step: FLOPs done per byte moved.
 * Weights are read once for the whole batch, so more rows means more reuse.
 */
export const prefillIntensity = (rows: number, model: Model = m8) =>
  (2 * model.params * rows) / (model.params * 2 + rows * kvBytesPerToken(model, 2));

/** The same, for one decode token: rows == batch, so bigger batches buy more reuse. */
export const decodeIntensity = (batch: number, ctx: number, model: Model = m8) =>
  (2 * model.params * batch) / (model.params * 2 + batch * ctx * kvBytesPerToken(model, 2));

/** Prefill-instance count for an arrival rate: GPU-seconds of work per second, rounded up. */
export function prefillPool(
  arrivalsPerSec: number,
  promptTokens: number,
  model: Model = m8,
  gpu: Gpu = gpuById('h100'),
) {
  const gpuSeconds = arrivalsPerSec * promptTokens * prefillPerToken(model, gpu);
  return { gpuSeconds, count: Math.max(1, Math.ceil(gpuSeconds)) };
}

/** Decode-instance count: whichever of throughput and KV memory binds first. */
export function decodePool(
  arrivalsPerSec: number,
  outTokens: number,
  targetConcurrency: number,
  ctx: number,
  model: Model = m8,
  gpu: Gpu = gpuById('h100'),
) {
  const slots = maxBatch(ctx, model, gpu);
  const perGpu = slots / decodeStep(slots, ctx, model, gpu).t;
  const demand = arrivalsPerSec * outTokens;
  const byThroughput = Math.ceil(demand / perGpu);
  const byMemory = Math.ceil(targetConcurrency / slots);
  return { slots, perGpu, demand, byThroughput, byMemory, count: Math.max(1, Math.max(byThroughput, byMemory)) };
}
