// Module 15 · Compilation & CUDA Graphs — shared math.
//
// Every number here is arithmetic on numbers the learner already owns
// (Module 02's roofline and speed limits, Module 05's KV bytes, the public
// spec-sheet figures in lib/perfModel). Nothing is a measured benchmark.
// The CPU-side constants are the one place we have to *assume*, and the lab
// notebook measures them, so they live in named constants the reader can check.

import { kvBytesPerToken, type Gpu, type Model } from '../lib/perfModel';

export type Mode = 'eager' | 'compiled' | 'graphed';

export const MODES: Mode[] = ['eager', 'compiled', 'graphed'];

export const MODE_LABEL: Record<Mode, string> = {
  eager: 'Eager (PyTorch default)',
  compiled: 'torch.compile',
  graphed: 'torch.compile + CUDA graph',
};

export const MODE_SHORT: Record<Mode, string> = {
  eager: 'eager',
  compiled: 'compiled',
  graphed: 'CUDA graph',
};

// ── The CPU side: what it costs to *say* something to the GPU ──────────────

/**
 * Microseconds of CPU time per kernel launch, measured with the PyTorch profiler
 * in Lab 15. Typical range on a modern server core: 4–10 µs. This is Python
 * attribute lookup, the dispatcher, the caching allocator, and cudaLaunchKernel.
 */
export const LAUNCH_US = 6;

/** After torch.compile the Python interpreter is out of the loop: ~2 µs per kernel. */
export const LAUNCH_US_COMPILED = 2;

/** Replaying a captured graph is one cudaGraphLaunch, whatever is inside it: ~8 µs. */
export const LAUNCH_US_REPLAY = 8;

/** One host/device rendezvous per decode step, to read the sampled token ids back. */
export const SYNC_MS = 0.2;

// ── How many kernels a decode step is made of ──────────────────────────────
// Counted from Llama's block: 7 GEMMs, 1 attention call, and the elementwise
// chain (2 RMSNorms, 2 RoPEs, SiLU, the gate/up product, 2 residual adds).
// The head adds a final norm, the LM projection, and the sampling chain.

export const ELEMENTWISE_EAGER = 9; // 2 RMSNorm + 2 RoPE + SiLU + mul + 2 residual adds
export const ELEMENTWISE_COMPILED = 5; // inductor folds the SiLU+mul and the add+norm
export const GEMM_PER_LAYER = 7;
export const ATTN_PER_LAYER = 1;
export const HEAD_EAGER = 12; // final norm, lm_head, argmax, sort, gather, divide, ...
export const HEAD_COMPILED = 4; // the whole sampling chain becomes one or two kernels

/** Kernel count for one decode step, by mode. A captured graph is *one* launch. */
export function kernelsPerStep(model: Model, mode: Mode): number {
  if (mode === 'graphed') return 1;
  const ew = mode === 'eager' ? ELEMENTWISE_EAGER : ELEMENTWISE_COMPILED;
  const head = mode === 'eager' ? HEAD_EAGER : HEAD_COMPILED;
  return model.layers * (ew + GEMM_PER_LAYER + ATTN_PER_LAYER) + head;
}

/** The same count, split by what the kernel does, so the visual can colour them. */
export function kernelBreakdown(model: Model, mode: Mode) {
  if (mode === 'graphed') return { gemm: 0, attn: 0, elementwise: 0, head: 0 };
  const ew = mode === 'eager' ? ELEMENTWISE_EAGER : ELEMENTWISE_COMPILED;
  const head = mode === 'eager' ? HEAD_EAGER : HEAD_COMPILED;
  return {
    gemm: model.layers * GEMM_PER_LAYER,
    attn: model.layers * ATTN_PER_LAYER,
    elementwise: model.layers * ew,
    head,
  };
}

/** CPU milliseconds spent issuing one decode step. */
export function cpuMs(model: Model, mode: Mode, launchUs = LAUNCH_US): number {
  if (mode === 'graphed') return LAUNCH_US_REPLAY / 1000;
  const per = mode === 'eager' ? launchUs : Math.min(launchUs, LAUNCH_US_COMPILED);
  return (kernelsPerStep(model, mode) * per) / 1000;
}

// ── The GPU side: the roofline, unchanged ──────────────────────────────────

export interface StepInput {
  model: Model;
  gpu: Gpu;
  batch: number;
  ctx: number;
  bytesPerWeight?: number; // 2 = BF16, 1 = FP8/INT8
  mbu?: number; // achievable fraction of peak bandwidth
  mfu?: number; // achievable fraction of peak FLOP/s
}

/** The bytes one decode step must read: all the weights, plus the batch's KV cache. */
export function stepBytes(p: StepInput) {
  const b = p.bytesPerWeight ?? 2;
  const weights = p.model.params * b;
  const kvPerSeq = p.ctx * kvBytesPerToken(p.model, b);
  const kv = p.batch * kvPerSeq;
  return { weights, kv, kvPerSeq, total: weights + kv };
}

/** GPU time for one decode step, from the two speed limits. Decode is memory-bound. */
export function gpuMs(p: StepInput) {
  const { weights, kv } = stepBytes(p);
  const mbu = p.mbu ?? 0.8;
  const mfu = p.mfu ?? 0.55;
  const tMem = (weights + kv) / (p.gpu.bw * mbu);
  const tCompute = (2 * p.model.params * p.batch) / (p.gpu.flops * mfu);
  return { tMem, tCompute, t: Math.max(tMem, tCompute) };
}

/** Which side of the step is the longer one. */
export function bound(p: StepInput, mode: Mode, launchUs = LAUNCH_US) {
  const g = gpuMs(p);
  const c = cpuMs(p.model, mode, launchUs);
  return { ...g, cpu: c, launchBound: c > g.t };
}

/** One decode step, end to end: the two pipelines run concurrently, so the step is the longer of them. */
export function stepMs(p: StepInput, mode: Mode, launchUs = LAUNCH_US, syncMs = SYNC_MS) {
  const b = bound(p, mode, launchUs);
  const t = Math.max(b.cpu, b.t) + syncMs;
  return {
    ...b,
    sync: syncMs,
    t,
    tps: p.batch / t, // aggregate output tokens per second
    tpot: t, // every user waits one step per token
  };
}

// ── The crossover: how many users before the GPU becomes the bottleneck ────

/**
 * Batch at which the CPU's launch work exactly equals the GPU's memory work.
 *
 *   t_cpu = t_gpu(B)  =>  B* = (t_cpu * bw * mbu - weights) / (ctx * kvPerToken)
 *
 * Returns 0 when the GPU is already the bottleneck at batch 1.
 */
export function crossoverBatch(p: Omit<StepInput, 'batch'>, mode: Mode, launchUs = LAUNCH_US): number {
  const b = p.bytesPerWeight ?? 2;
  const mbu = p.mbu ?? 0.8;
  const c = cpuMs(p.model, mode, launchUs) / 1000;
  const headroom = c * p.gpu.bw * mbu - p.model.params * b;
  if (headroom <= 0) return 0;
  const perSeq = p.ctx * kvBytesPerToken(p.model, b);
  return headroom / perSeq;
}

/** True when the step is limited by Python and the dispatcher rather than by the GPU. */
export function isLaunchBound(p: StepInput, mode: Mode, launchUs = LAUNCH_US) {
  return cpuMs(p.model, mode, launchUs) > gpuMs(p).t;
}

// ── One decoder layer's elementwise chain, in bytes per token ──────────────
// Llama-3.1-8B shapes: hidden 4096, 8 KV heads of 128, FFN intermediate 14336.
// One "unit" is the hidden-size row: 4096 numbers in BF16 = 8,192 bytes.

const U = 8192;
const H = 1; // hidden row
const K = 0.25; // one KV head row: 128 numbers
const F = 14336 / 4096; // one FFN row

export interface Op {
  name: string;
  units: number; // HBM traffic, in hidden-rows of 8,192 bytes
  note: string;
}

/** The nine eager kernels that surround the seven GEMMs of one decoder layer. */
export const CHAIN_EAGER: Op[] = [
  { name: 'RMSNorm', units: 2 * H, note: 'read x, write normalised' },
  { name: 'split heads', units: 3 * (H + K), note: 'Q, K and V copied into a contiguous tensor' },
  { name: 'RoPE on Q', units: 2 * H, note: 'rotate Q in place' },
  { name: 'RoPE on K', units: 2 * K, note: 'rotate K in place' },
  { name: 'residual add', units: 3 * H, note: 'read x, read attn, write h' },
  { name: 'RMSNorm', units: 2 * H, note: 'read h, write normalised' },
  { name: 'SiLU(gate)', units: 2 * F, note: 'read gate, write activated' },
  { name: '× up', units: 3 * F, note: 'read both, write the product' },
  { name: 'residual add', units: 3 * H, note: 'read h, read mlp, write h' },
];

/** The five kernels inductor leaves behind, and the bytes they still move. */
export const CHAIN_FUSED: Op[] = [
  { name: 'RMSNorm', units: 2 * H, note: 'a prologue: it feeds a GEMM' },
  { name: 'RoPE on Q,K', units: 2 * (H + K), note: 'the split and both rotations in one kernel' },
  { name: 'add + RMSNorm', units: 4 * H, note: 'the row fits in SRAM, so it is re-read from there' },
  { name: 'SiLU(gate)×up', units: 3 * F, note: 'one kernel instead of two' },
  { name: 'residual add', units: 3 * H, note: 'left alone between the two GEMMs' },
];

export const chainUnits = (ops: Op[]) => ops.reduce((a, o) => a + o.units, 0);

/** HBM bytes for that chain, for `tokens` tokens through a whole model. */
export function chainBytes(tokens: number, layers: number, ops: Op[]) {
  return chainUnits(ops) * U * tokens * layers;
}

/** The chain a given mode leaves behind. A captured graph still runs fused kernels. */
export const chainFor = (mode: Mode): Op[] => (mode === 'eager' ? CHAIN_EAGER : CHAIN_FUSED);

/**
 * The three-way comparison the lesson turns on: a whole 512-token chunk, the
 * elementwise chain's HBM traffic, the number of elementwise launches, and the
 * CPU time for the entire decode step.
 */
export function chainStats(model: Model, tokens: number, mode: Mode, breakLayer = -1) {
  const ops = chainFor(mode);
  const layers = breakLayer < 0 ? model.layers : breakLayer;
  const eagerLayers = breakLayer < 0 ? 0 : model.layers - breakLayer;
  const units = chainUnits(ops) * layers + chainUnits(CHAIN_EAGER) * eagerLayers;
  const ewKernels = (mode === 'eager' ? ELEMENTWISE_EAGER : ELEMENTWISE_COMPILED) * layers + ELEMENTWISE_EAGER * eagerLayers;
  const head = mode === 'eager' ? HEAD_EAGER : HEAD_COMPILED;
  const kernels = layers * ((mode === 'eager' ? ELEMENTWISE_EAGER : ELEMENTWISE_COMPILED) + GEMM_PER_LAYER + ATTN_PER_LAYER)
    + eagerLayers * (ELEMENTWISE_EAGER + GEMM_PER_LAYER + ATTN_PER_LAYER) + head;
  return { units, ewKernels, kernels, bytes: units * U * tokens, cpuMs: (kernels * (mode === 'eager' ? LAUNCH_US : LAUNCH_US_COMPILED)) / 1000 };
}

// ── CUDA graph accounting ──────────────────────────────────────────────────

/**
 * The price of a graph: you cannot change an address inside it, so inputs and
 * outputs must live in buffers allocated once. Copying in costs a `batch × hidden`
 * write; copying out costs whatever you choose to read.
 */
export function copyInBytes(batch: number, hidden: number, bytes = 2) {
  return batch * hidden * bytes;
}

/** Logits for the whole vocabulary are huge. One sampled token id per sequence is 4 bytes. */
export function copyOutBytes(batch: number, vocab: number, bytes = 2) {
  return batch * vocab * bytes;
}

/** What vLLM does: sample inside the graph, so the graph's output is 4 bytes per sequence. */
export const SAMPLED_ID_BYTES = 4;

// ── Formatting shared by the module's visuals ──────────────────────────────

export const ms = (v: number) => (v < 10 ? `${v.toFixed(2)} ms` : `${v.toFixed(1)} ms`);

export const us = (v: number) => (v < 10 ? `${v.toFixed(1)} µs` : `${Math.round(v)} µs`);

export const short = (m: Model) => m.name.replace('Llama-', '').replace('Qwen2.5-', 'Qwen');
