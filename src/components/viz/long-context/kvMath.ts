// Module 20 · Long Context & Tiered KV — shared math.
//
// Every number here is derived in the module's prose from Module 05's KV
// arithmetic (2 * L * H_kv * D * b bytes per token) and Module 02's roofline
// (time = bytes / (bandwidth * fraction) or FLOPs / (peak * MFU)).
// Nothing is measured, tuned or invented: these are the same formulas the
// learner already owns, arranged for long context.
import { modelById, gpuById, kvBytesPerToken, type Model, type Gpu } from '../lib/perfModel';

export const llama8b = modelById('llama8b');
export const qwen05b = modelById('qwen05');
export const qwen32b = modelById('qwen32b');
export const llama70b = modelById('llama70b');
export const h100 = gpuById('h100');
export const a100 = gpuById('a100');
export const h200 = gpuById('h200');
export const t4 = gpuById('t4');

/** Bytes of KV cache one sequence occupies at `ctx` tokens, at `bytes` per number. */
export const seqKvBytes = (ctx: number, model: Model = llama8b, bytes = 2) =>
  ctx * kvBytesPerToken(model, bytes);

/** Fraction of peak bandwidth a real decode step reaches (Module 02's MBU). */
export const MBU = 0.8;
/** Fraction of peak FLOP/s a real prefill reaches (Module 02's MFU). */
export const MFU = 0.55;

// ───────────────────────── the walls of long context ─────────────────────────

/** KV memory the card can still spend on conversations, in bytes. */
export function kvBudget(model: Model = llama8b, gpu: Gpu = h100) {
  const weights = model.params * 2;
  const workspace = 0.04 * gpu.mem;
  return (gpu.mem - weights - workspace) * 0.9;
}

/** Module 05's context-tokens-per-GPU: the whole card in tokens. */
export const contextTokensPerGpu = (model: Model = llama8b, gpu: Gpu = h100) =>
  kvBudget(model, gpu) / kvBytesPerToken(model, 2);

/** Where a decode step stops being weight-bound: T * b_kv = P * b. */
export const bandwidthWall = (model: Model = llama8b, bytes = 2) =>
  (model.params * bytes) / kvBytesPerToken(model, bytes);

/** Where prefill attention overtakes the projections: P / (2 * d * L). */
export const computeWall = (model: Model = llama8b) =>
  model.params / (2 * model.hidden * model.layers);

/**
 * One decode step, split into the weight read (shared by the whole batch) and
 * the KV read (paid per sequence). Seconds.
 */
export function decodeTerms(ctx: number, model: Model = llama8b, gpu: Gpu = h100) {
  const tWeights = (model.params * 2) / (gpu.bw * MBU);
  const tKv = seqKvBytes(ctx, model) / (gpu.bw * MBU);
  return { tWeights, tKv, t: tWeights + tKv + 0.0015 };
}

/**
 * Prefill of `tokens`, including the quadratic attention term Module 05 gave us:
 *   FLOPs = 2 * P * n  +  4 * n^2 * d * L
 */
export function prefillTTFT(tokens: number, model: Model = llama8b, gpu: Gpu = h100, mfu = MFU) {
  const flopsProj = 2 * model.params * tokens;
  const flopsAttn = 4 * tokens * tokens * model.hidden * model.layers;
  return { t: (flopsProj + flopsAttn) / (gpu.flops * mfu), flopsProj, flopsAttn };
}

// ───────────────────────── the tier ladder ─────────────────────────

/**
 * A storage tier, described the only two ways that matter for a KV read:
 * how many bytes a second it can deliver, and how long a single access takes.
 * Bandwidth figures are approximate public spec-sheet numbers.
 */
export interface Tier {
  id: string;
  name: string;
  short: string;
  /** bytes per second, sustained sequential */
  bw: number;
  /** nanoseconds to first byte */
  latencyNs: number;
  /** typical capacity of one machine */
  capacity: string;
}

export const TIERS: Tier[] = [
  { id: 'hbm', name: 'GPU memory (H100 SXM)', short: 'HBM', bw: 3.35e12, latencyNs: 350, capacity: '80–141 GB' },
  { id: 'dram', name: 'Host DRAM, PCIe 5.0 x16', short: 'CPU DRAM', bw: 55e9, latencyNs: 90, capacity: '256 GB–2 TB' },
  { id: 'nvme', name: 'Local NVMe, PCIe 4.0 x4', short: 'NVMe', bw: 7e9, latencyNs: 80e3, capacity: '2–30 TB' },
  { id: 'remote', name: 'Remote node, 10 GbE', short: 'Remote', bw: 1.25e9, latencyNs: 200e3, capacity: 'node-sized' },
];

export const tierById = (id: string): Tier => TIERS.find((t) => t.id === id)!;

/** Seconds to move `bytes` from a tier, at line rate. */
export const fetchSec = (bytes: number, tier: Tier) => bytes / tier.bw;

/** Seconds to fetch one token's KV from a tier. */
export const fetchPerToken = (model: Model, tier: Tier, bytes = 2) =>
  kvBytesPerToken(model, bytes) / tier.bw;

/** Seconds to recompute one token's KV on a GPU: 2P FLOP at the prefill MFU. */
export const recomputePerToken = (model: Model, gpu: Gpu = h100, mfu = MFU) =>
  (2 * model.params) / (gpu.flops * mfu);

/**
 * The bandwidth at which fetching and recomputing cost the same time.
 * Above it, fetch wins (a tier is genuinely fast); below it, recompute wins
 * (the FLOPs are cheaper than the bytes, and a decode step leaves the
 * tensor cores idle — Module 05's "waste only costs what it is made of").
 */
export const crossoverBW = (model: Model, gpu: Gpu = h100, mfu = MFU) =>
  kvBytesPerToken(model, 2) / recomputePerToken(model, gpu, mfu);

/** Bytes a decode step must move when the cache is split into resident + streamed. */
export const stepBytes = (ctx: number, resident: number, model: Model = llama8b, bytes = 2) =>
  seqKvBytes(ctx, model, bytes);

/** Short, honest label for a duration that may span eight orders of magnitude. */
export function humanTime(s: number): string {
  if (s < 1e-6) return `${(s * 1e9).toFixed(0)} ns`;
  if (s < 1e-3) return `${(s * 1e6).toFixed(s * 1e6 < 10 ? 1 : 0)} µs`;
  if (s < 1) return `${(s * 1e3).toFixed(s * 1e3 < 10 ? 1 : 0)} ms`;
  return `${s.toFixed(s < 10 ? 2 : 0)} s`;
}
