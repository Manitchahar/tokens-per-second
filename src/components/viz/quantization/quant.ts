// Module 11 · Quantization — shared math.
//
// Every function here is the cheapest possible version of a formula the module's
// prose derives. Nothing is invented: the bytes come from Module 01's "how a number
// lives in memory", the roofline terms from Module 02, and the weights/KV shapes
// from Module 05's memory math.

import { gpuById, modelById, kvBytesPerToken, type Model, type Gpu } from '../lib/perfModel';

const llama8b = modelById('llama8b');
const llama70b = modelById('llama70b');
const h100 = gpuById('h100');
const t4 = gpuById('t4');

/* ───────────────────────────── bits & bytes ───────────────────────────── */

/** Bytes one weight occupies at a given bit width. The single most important number in the module. */
export const bytesPerWeight = (bits: number) => bits / 8;

/** Bytes one activation occupies at a given bit width. */
export const bytesPerAct = bytesPerWeight;

/**
 * Bytes a whole model occupies on the GPU at a given weight bit width.
 * Metadata (scales, zero-points) is the real cost of low-bit weights and is
 * not free — we add it explicitly with `metaFrac`.
 */
export const weightBytes = (model: Model, bits: number, metaFrac = 0.0) =>
  model.params * bytesPerWeight(bits) * (1 + metaFrac);

/**
 * Scale-and-zero-point overhead, as a fraction of the packed weights.
 * For group-wise INT4 with a group of g elements stored in FP16, each group costs
 * 4 extra bytes (2 for scale, 2 for zero), so metaFrac = 4/g / 0.5.
 * At g = 128 that is 6.25%; at g = 64, 12.5%; at g = 32, 25%.
 */
export function metaFracInt(bits: number, group: number) {
  if (bits >= 16) return 0; // FP16/BF16 need no scale
  return (4 / group) / bytesPerWeight(bits);
}

/** The KV cache for one sequence at `ctx` tokens, with quantized KV. */
export const kvBytes = (ctx: number, kvBits: number, model: Model = llama8b) =>
  ctx * kvBytesPerToken(model, kvBits / 8);

/** Total GPU memory a serving config needs: weights + KV for `batch` users + a workspace reserve. */
export function serveMemory(p: {
  model?: Model;
  wBits?: number;
  kvBits?: number;
  batch: number;
  ctx: number;
  metaFrac?: number;
  gpu?: Gpu;
}) {
  const model = p.model ?? llama8b;
  const wBits = p.wBits ?? 16;
  const kvBits = p.kvBits ?? 16;
  const gpu = p.gpu ?? h100;
  const meta = p.metaFrac ?? metaFracInt(wBits, 128);
  const weights = weightBytes(model, wBits, meta);
  const perSeq = kvBytes(p.ctx, kvBits, model);
  const kv = p.batch * perSeq;
  const reserve = 0.05 * gpu.mem; // activations, workspace, CUDA context
  return { weights, perSeq, kv, reserve, total: weights + kv + reserve, gpu };
}

/* ───────────────────────────── roofline terms ───────────────────────────── */

/**
 * One decode step, in seconds, with `wBits`/`aBits` weights and activations.
 *
 *   bytes read = weights + batch × ctx × kvBytesPerToken
 *   t_memory   = bytes / (bandwidth × MBU)
 *   t_compute  = 2 × params × batch / (tensor-core FLOP/s at these bits × MFU)
 *
 * The tensor cores are faster in lower precision, which is the whole reason a
 * W4 model can be *compute*-faster as well as smaller.
 */
export function decodeStepQuant(p: {
  model?: Model;
  gpu?: Gpu;
  batch: number;
  ctx: number;
  wBits?: number;
  kvBits?: number;
  metaFrac?: number;
  overhead?: number;
  mbu?: number;
  mfu?: number;
}) {
  const model = p.model ?? llama8b;
  const gpu = p.gpu ?? h100;
  const wBits = p.wBits ?? 16;
  const kvBits = p.kvBits ?? 16;
  const mbu = p.mbu ?? 0.8;
  const overhead = p.overhead ?? 0.0015;
  const meta = p.metaFrac ?? metaFracInt(wBits, 128);

  const weights = weightBytes(model, wBits, meta);
  const kvPerSeq = kvBytes(p.ctx, kvBits, model);
  const kvRead = p.batch * kvPerSeq;

  // Achieved tensor-core throughput. A GPU's quoted peak is for the widest common
  // format; 4-bit integer math is not generically faster than INT8 on every chip, so
  // we only credit FP8 and BF16-class formats with the 2× tensor-core rate.
  const mfu = p.mfu ?? (wBits <= 8 ? 0.5 : 0.55);
  const tWeights = weights / (gpu.bw * mbu);
  const tKv = kvRead / (gpu.bw * mbu);
  const tMem = tWeights + tKv;
  const tCompute = (2 * model.params * p.batch) / (gpu.flops * mfu);

  const t = Math.max(tMem, tCompute) + overhead;
  return {
    t, tWeights, tKv, tMem, tCompute, overhead,
    weights, kvPerSeq, kvRead,
    perUser: 1 / t,
    total: p.batch / t,
    bound: tCompute > tMem ? ('compute' as const) : ('memory' as const),
  };
}

/** Prefill time for `n` tokens with `wBits` weights. Prefill is compute-bound, so bits barely matter — except for the weight read. */
export function prefillQuant(p: { model?: Model; gpu?: Gpu; tokens: number; wBits?: number; metaFrac?: number; mfu?: number; mbu?: number }) {
  const model = p.model ?? llama8b;
  const gpu = p.gpu ?? h100;
  const wBits = p.wBits ?? 16;
  const meta = p.metaFrac ?? metaFracInt(wBits, 128);
  const mbu = p.mbu ?? 0.8;
  const mfu = p.mfu ?? (wBits <= 8 ? 0.5 : 0.55);
  const tCompute = (2 * model.params * p.tokens) / (gpu.flops * mfu);
  const tMem = weightBytes(model, wBits, meta) / (gpu.bw * mbu);
  return { t: Math.max(tCompute, tMem) + 0.002, tCompute, tMem };
}

/* ───────────────────────── round-to-nearest, in JS ───────────────────────── */

/**
 * Quantize one row to `bits` bits with a symmetric range and round-half-to-even,
 * the same rule every real quantizer uses. Returns the codes and the scale.
 * Used by the in-browser labs so the error you see is the error the math predicts.
 */
export function quantizeRow(vals: number[], bits: number) {
  const levels = 1 << bits; // 16 for INT4
  const maxAbs = Math.max(...vals.map(Math.abs));
  const scale = maxAbs / (levels / 2 - 1); // symmetric: [-8, 7] for INT4
  const clampMax = levels / 2 - 1;
  const codes = vals.map((v) => {
    const r = Math.round(v / scale);
    return Math.max(-clampMax, Math.min(clampMax, r));
  });
  const deq = codes.map((c) => c * scale);
  return { codes, deq, scale, maxAbs };
}

/** Relative error of a dequantized tensor, ‖ΔW‖ / ‖W‖ — the number quality papers report. */
export function relErr(orig: number[], deq: number[]) {
  let num = 0, den = 0;
  for (let i = 0; i < orig.length; i++) {
    const d = deq[i] - orig[i];
    num += d * d;
    den += orig[i] * orig[i];
  }
  return Math.sqrt(num / den);
}

/* ─────────────────────── named configurations, used everywhere ─────────────────────── */

export interface Scheme {
  id: string;
  label: string;
  wBits: number;
  kvBits: number;
  note: string;
}

/** The schemes this module compares. wBits = weight precision, kvBits = KV cache precision. */
export const SCHEMES: Scheme[] = [
  { id: 'bf16', label: 'BF16', wBits: 16, kvBits: 16, note: 'The default. Full precision, nothing saved.' },
  { id: 'fp8', label: 'FP8', wBits: 8, kvBits: 16, note: 'E4M3 weights. 2× smaller, native on Ada, Hopper, Blackwell.' },
  { id: 'w8a8', label: 'W8A8', wBits: 8, kvBits: 8, note: 'INT8 weights and activations. The safest big win.' },
  { id: 'w4a16', label: 'W4A16', wBits: 4, kvBits: 16, note: '4-bit weights, 16-bit compute. The standard 4-bit recipe.' },
  { id: 'w4kv8', label: 'W4 + KV8', wBits: 4, kvBits: 8, note: '4-bit weights and a halved KV cache. Memory-bound serving wins here.' },
  { id: 'w4kv4', label: 'W4 + KV4', wBits: 4, kvBits: 4, note: 'Everything in 4 bits. Maximum density, maximum risk.' },
];

/* ───────────────────── error accumulation through a model ───────────────────── */

/**
 * The compounding rule that makes or breaks a quantized model.
 *
 * One weight perturbation of relative size ε arriving at the output of a linear
 * layer produces a relative output perturbation of exactly ε — in the worst case,
 * and averaged over the layer it is smaller. Quantization errors across layers are
 * roughly independent, so after L layers:
 *
 *   relative error at the output ≈ ε × sqrt(L)
 *
 * Not ε × L: that is the difference between 32 layers being fine and hopeless.
 * This is the whole reason naive 4-bit quantization of a deep model degrades but
 * does not explode.
 */
export function stackError(eps: number, layers: number) {
  return eps * Math.sqrt(layers);
}

/** The arithmetic-intensity of a decode GEMM at a given batch size and bit width. */
export function intensity(batch: number, model: Model = llama8b, wBits = 16) {
  return (2 * model.params * batch) / weightBytes(model, wBits);
}

export { llama8b, llama70b, h100, t4 };