// Module 25 · Inference Economics — shared cost math.
//
// Everything here is a rearrangement of arithmetic the learner has already done:
// the decode step from Modules 02/05 (perfModel.decodeStep), prefill from Module 05
// (perfModel.prefillTime), and weight/KV byte counts from Module 01. Nothing in this
// file invents a benchmark. The whole module rests on one identity:
//
//   GPU-seconds per request = t_prefill + n_out x t_step / B
//
// which is Module 06's service time, read as a quantity of *machine time* rather than
// latency. Multiply it by a price per second and you have the price of a request.
//
// Prices are illustrative on-demand cloud rates (perfModel.GPUS), not quotes.

import {
  decodeStep,
  prefillTime,
  maxBatch,
  gpuById,
  modelById,
  kvBytesPerToken,
  GPUS,
  type Gpu,
  type Model,
} from '../lib/perfModel';

/** An operating point: which model on which card, at how much concurrency, in how many bits. */
export interface Op {
  modelId: string;
  gpuId: string;
  batch: number; // concurrent decode sequences
  ctx: number; // average tokens of context per sequence (prompt + generated so far)
  wBytes: number; // bytes per weight: 2 = BF16/FP16, 1 = FP8/INT8, 0.5 = INT4
  kvBytes?: number;
  mbu?: number;
  mfu?: number;
  overhead?: number;
}

/**
 * Achievable bandwidth fraction by weight precision. A 2-byte kernel streams weights
 * almost as fast as the memory can deliver them; 4-bit and 8-bit GEMV kernels reach a
 * lower fraction of peak because they are bound by unpacking and by smaller transfers.
 * These are illustrative — Module 12's job is to measure yours.
 */
export const MBU_FOR_BITS: Record<number, number> = { 2: 0.8, 1: 0.75, 0.5: 0.65 };

export const mbuFor = (op: Op) => op.mbu ?? MBU_FOR_BITS[op.wBytes] ?? 0.8;

/**
 * The context length a request is costed at. A sequence's KV cache grows with prompt
 * plus output, so the floor is the prompt — but a 512-token prompt does not mean the
 * service only ever holds 512 tokens of context, so the floor is the stated working
 * context instead: 2,048 tokens, unless the prompt is longer than that.
 *
 * Every table in this module is computed at this context, which is why the visuals and
 * the prose agree to the digit. Cost at a shorter context would be a fairer *measurement*
 * and a dishonest *model*, because it under-counts the KV every real request carries.
 */
export const WORKING_CTX = 2048;
export const ctxFor = (nIn: number) => Math.max(nIn, WORKING_CTX);

/** One decode step at this operating point. */
export function stepAt(op: Op) {
  return decodeStep({
    model: modelById(op.modelId),
    gpu: gpuById(op.gpuId),
    batch: op.batch,
    ctx: op.ctx,
    wBytes: op.wBytes,
    kvBytes: op.kvBytes ?? 2,
    mbu: mbuFor(op),
    mfu: op.mfu,
    overhead: op.overhead,
  });
}

/** Prefill of `tokens` prompt tokens: a compute-bound pass that reads the weights once. */
export function prefillAt(op: Op, tokens: number) {
  return { t: prefillTime({
    model: modelById(op.modelId),
    gpu: gpuById(op.gpuId),
    tokens,
    wBytes: op.wBytes,
    mbu: mbuFor(op),
    mfu: op.mfu,
  }) };
}

/**
 * The identity the module is built on. Prefill belongs to one request alone; a decode
 * step is shared by every sequence in the batch, so a request's share of it is
 * t_step / B. Sum them and you have how much machine time the request consumed.
 *
 * `prefillFrac` is the share of that machine time which is FLOP-priced, and
 * `decodeFrac` the share which is bandwidth-priced. The ratio between them is decided
 * by the workload's shape — prompt length against output length — and it is the single
 * biggest reason two products with the same model have different bills.
 */
export function serviceTime(op: Op, nOut: number, nIn = 512) {
  const tPre = prefillAt(op, nIn).t;
  const tStep = stepAt(op).t;
  const decodeShare = (nOut * tStep) / op.batch;
  const gpuSeconds = tPre + decodeShare;
  return {
    tPre,
    tStep,
    decodeShare,
    gpuSeconds,
    prefillFrac: tPre / gpuSeconds,
    decodeFrac: decodeShare / gpuSeconds,
  };
}

/** Dollars per second of one GPU. The bill is wall-clock, not busy. */
export const perSecond = (gpuId: string) => gpuById(gpuId).price / 3600;

/** Dollars per hour of one GPU. */
export const perHour = (gpuId: string) => gpuById(gpuId).price;

/** Dollars per million *output* tokens — the number vendors quote and the one that hurts. */
export function costPerMTokOut(op: Op, nOut: number, nIn = 512) {
  return (serviceTime(op, nOut, nIn).gpuSeconds * perSecond(op.gpuId) * 1e6) / nOut;
}

/** Dollars per million tokens counting prompt and output together. Always far lower. */
export function blendedPerMTok(op: Op, nIn: number, nOut: number) {
  return (serviceTime(op, nOut, nIn).gpuSeconds * perSecond(op.gpuId) * 1e6) / (nIn + nOut);
}

/** Dollars per million *prompt* tokens. Prefill is compute-bound and therefore cheap. */
export function promptPerMTok(op: Op, nIn = 512) {
  return (prefillAt(op, nIn).t * perSecond(op.gpuId) * 1e6) / nIn;
}

/** How many output tokens the output tokens of a request cost, versus its prompt tokens. */
export const outputPremium = (op: Op, nIn = 512, nOut = 256) =>
  costPerMTokOut(op, nOut, nIn) / promptPerMTok(op, nIn);

/**
 * The utilization tax. A card rented by the hour is billed for every second it exists,
 * busy or not, so a service that is only busy part of the time divides its goodput —
 * and multiplies its cost — by the fraction of the hour it was actually working.
 */
export const utilizationTax = (avgUtil: number) => 1 / avgUtil;

/** Largest number of streams at `ctx` that fit beside the weights on this card. */
export function maxStreams(op: Omit<Op, 'batch'>) {
  return maxBatch({
    model: modelById(op.modelId),
    gpu: gpuById(op.gpuId),
    ctx: op.ctx,
    wBytes: op.wBytes,
    kvBytes: op.kvBytes ?? 2,
    mbu: mbuFor(op),
    mfu: op.mfu,
    overhead: op.overhead,
  });
}

/* ──────────────────────── when the model does not fit on one card ──────────────────────── */

/**
 * Tensor parallelism: shard the weights across k cards. Wall-clock barely changes —
 * the cards work together on one forward pass — but the *bill* multiplies by k, and
 * the all-reduces make each step slightly slower still.
 */
export function tensorParallel(op: Op) {
  const gpu = gpuById(op.gpuId);
  const model = modelById(op.modelId);
  const weights = model.params * op.wBytes;
  const reserve = 0.04 * gpu.mem;
  const usable = gpu.mem * 0.92;
  const k = Math.max(1, Math.ceil((weights + reserve) / usable));
  // One all-reduce of the hidden state per layer group: tens of microseconds on a
  // connected machine, and it grows with k. ~15% per extra rank is a reasonable stand-in.
  const stepPenalty = 1 + 0.15 * (k - 1);
  return { cards: k, stepPenalty, fitsOnOne: k === 1 };
}

/** Dollars per million output tokens once tensor parallelism is paid for. */
export function costPerMTokTP(op: Op, nOut: number, nIn = 512) {
  const tp = tensorParallel(op);
  if (tp.cards === 1) return costPerMTokOut(op, nOut, nIn);
  // k cards each billed, each step slowed by the collectives: cost per token ~ k x penalty.
  return costPerMTokOut(op, nOut, nIn) * tp.cards * tp.stepPenalty;
}

/* ───────────────────────────── the three regimes ───────────────────────────── */

/**
 * Which of the three limits sets the price, for a given operating point.
 *
 *   memory — the weights do not fit beside the KV on one card, so more cards are needed.
 *             The bill is multiplied and each step is slowed by the collectives.
 *   decode  — the step is dominated by bytes. What you pay is set by price per TB/s.
 *   prefill — the request is dominated by prompt tokens. What you pay is set by
 *             price per PFLOP/s, and batching buys you almost nothing.
 *
 * The crossover is worth computing: batch a request until the decode phase dominates
 * and it becomes bandwidth-priced; give it a long prompt and a short answer and it
 * stays FLOP-priced no matter how large you make the batch.
 */
export function regimeOf(op: Op, nIn = 512, nOut = 256) {
  const tp = tensorParallel(op);
  if (tp.cards > 1) return 'memory' as const;
  const st = serviceTime(op, nOut, nIn);
  return st.prefillFrac > 0.5 ? ('prefill' as const) : ('decode' as const);
}

/**
 * The batch at which a request stops being FLOP-priced and becomes bandwidth-priced.
 * Solve n_out x t_step / B = t_prefill for B. Below it, this shape of request is
 * dominated by its prompt; above it, by its answer.
 */
export function crossoverBatch(op: Op, nIn = 512, nOut = 256) {
  const { tPre, tStep } = serviceTime(op, nOut, nIn);
  return (nOut * tStep) / tPre;
}

/**
 * Rank a card by how cheap it makes a token, *without* the batch and bits the reader has
 * already set. These two scalars are the entire GPU-generation story:
 *
 *   price / bandwidth  — the decode ranking  (dollars per TB/s)
 *   price / FLOPs     — the prefill ranking (dollars per PFLOP/s)
 */
export function gpuScalars(g: Gpu) {
  return {
    id: g.id,
    name: g.name,
    price: g.price,
    mem: g.mem,
    bw: g.bw,
    flops: g.flops,
    perBandwidth: g.price / (g.bw / 1e12), // dollars per (TB/s)
    perFlops: g.price / (g.flops / 1e15), // dollars per (PFLOP/s)
    pricePerGbMem: g.price / (g.mem / 1e9), // dollars per GB of memory
  };
}

export const GPU_SCALARS = GPUS.map(gpuScalars);

/* ───────────────────────────── demand and fleet sizing ───────────────────────────── */

export interface Demand {
  reqPerSec: number; // peak offered rate
  nIn: number; // prompt tokens
  nOut: number; // output tokens
}

/**
 * Steady-state fleet size. Every request consumes `gpuSeconds` of machine time, so
 * `reqPerSec x gpuSeconds` is the number of GPUs the fleet must be busy at. Dividing
 * by the utilization target leaves room for the queue to breathe.
 */
export function fleetSize(op: Op, d: Demand, rho = 0.6) {
  const st = serviceTime(op, d.nOut, d.nIn).gpuSeconds;
  const busyGpus = d.reqPerSec * st;
  return { serviceSeconds: st, busyGpus, atTarget: busyGpus / rho, gpus: Math.ceil(busyGpus / rho) };
}

/** Dollars per month for a fleet, at a given average utilization. 730 hours is a month. */
export function monthlyCost(gpus: number, gpuId: string, avgUtil = 1) {
  return gpus * perHour(gpuId) * 730 * avgUtil;
}

/**
 * Mixed fleets lose capacity to bin packing: a fleet with several card sizes cannot be
 * packed as tightly as a homogeneous one, because leftover space on a big card is not
 * a small card. `packingEff` is the fraction of purchased capacity that is usable.
 * Between 0.85 and 1.0 is the honest range for a two- or three-size fleet.
 */
export const packingEfficiency = (packingEff: number) => packingEff;

/**
 * Break-even speed of a cheaper card. If card B is `k` times slower per token than
 * card A but costs `c` times less, B wins when k x c is below the goodput-per-dollar
 * ratio you would get from A — and wins outright when c > k.
 *
 * The useful form of this, used constantly in the project: **a cheaper card wins iff it
 * is proportionally slower than it is cheaper.** If it is 2.5x cheaper and only 1.8x
 * slower, it wins on every token.
 */
export const pricePerSpeedRatio = (priceRatio: number, speedRatio: number) => priceRatio / speedRatio;

export { gpuById, modelById, kvBytesPerToken, GPUS };
export type { Gpu, Model };