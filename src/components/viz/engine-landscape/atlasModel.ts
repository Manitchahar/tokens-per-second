// The cost model behind the Module 17.2 visual.
//
// IMPORTANT: this is a model of the *shape* of the trade-off, not a benchmark of any real
// engine. Every number comes from arithmetic the learner has already done: weight bytes,
// KV bytes per token, peak bandwidth and peak FLOP/s from perfModel, plus a fixed
// per-step cost per engine. The three things that genuinely differ between engines —
// per-step overhead, achievable bandwidth fraction, and whether a 4-bit path exists —
// are the only free parameters. It never claims who is faster.

import { kvBytesPerToken, modelById, gpuById, type Model, type Gpu } from '../lib/perfModel';
import { ENGINES, type EngineSpec } from './fairModel';

export interface Workload {
  modelId: string;
  gpuId: string;
  batch: number;   // sequences in this decode step
  ctx: number;     // tokens of context per sequence
  bytes: number;   // 2 = BF16, 0.5 = 4-bit
}

export const WORKLOADS: { id: string; label: string; w: Workload; blurb: string }[] = [
  {
    id: 'latency', label: 'latency',
    blurb: 'Llama-3.1-8B on one H100, BF16, 2,048 tokens of context, one stream. Fixed per-step cost is most of the bill here.',
    w: { modelId: 'llama8b', gpuId: 'h100', batch: 1, ctx: 2048, bytes: 2 },
  },
  {
    id: 'server', label: 'chat server',
    blurb: 'The same model and GPU, 16 concurrent streams. Still interactive, but the machine is no longer idle between steps.',
    w: { modelId: 'llama8b', gpuId: 'h100', batch: 16, ctx: 2048, bytes: 2 },
  },
  {
    id: 'bulk', label: 'bulk batch',
    blurb: 'The same model and GPU, 64 concurrent streams — an offline job. Almost the whole step is now bytes moved.',
    w: { modelId: 'llama8b', gpuId: 'h100', batch: 64, ctx: 2048, bytes: 2 },
  },
  {
    id: 'edge', label: '4-bit edge',
    blurb: 'Qwen2.5-32B on a 24 GB L4 in 4-bit. Weights are 16.4 GB, so the question stops being speed and becomes whether it loads at all.',
    w: { modelId: 'qwen32b', gpuId: 'l4', batch: 8, ctx: 1024, bytes: 0.5 },
  },
];

export interface Breakdown {
  engine: EngineSpec;
  batch: number;
  weightGB: number;
  kvReadGB: number;
  tMem: number;     // seconds: weights + KV across every stream
  tFlop: number;    // seconds: the math
  tGpu: number;     // max of the two — the GPU is busy for this long
  tCpu: number;     // seconds: scheduling + kernel launches, GPU idle
  tTotal: number;
  idleFrac: number;
  bound: 'memory' | 'compute';
  tokensPerUser: number;
  totalTokens: number;
  poolTokens: number;  // KV tokens that fit beside the weights
  fitsWeights: boolean;
}

/** One engine, one workload, one decode step. Every field is arithmetic the learner can redo. */
export function breakdown(spec: EngineSpec, w: Workload): Breakdown {
  const model: Model = modelById(w.modelId);
  const gpu: Gpu = gpuById(w.gpuId);
  // The 4-bit path is the one structural difference: it changes the weight bytes
  // themselves, and therefore whether a model loads at all.
  const bytes = spec.q4 ? Math.min(w.bytes, 0.5) : w.bytes;
  const weightBytes = model.params * bytes;
  const kvTok = kvBytesPerToken(model, 2);
  const kvReadBytes = w.batch * w.ctx * kvTok;

  const tMem = (weightBytes + kvReadBytes) / (gpu.bw * spec.mbu);
  const tFlop = (2 * model.params * w.batch) / (gpu.flops * spec.mfu);
  const tGpu = Math.max(tMem, tFlop);
  const tCpu = spec.overheadMs / 1000;
  const tTotal = tGpu + tCpu;

  // What is left for KV after the weights and a ~4% reserve for context, graphs and workspace.
  const reserve = 0.04 * gpu.mem;
  const poolTokens = Math.max(0, Math.floor((gpu.mem - weightBytes - reserve) / kvTok));

  return {
    engine: spec,
    batch: w.batch,
    weightGB: weightBytes / 1e9,
    kvReadGB: kvReadBytes / 1e9,
    tMem, tFlop, tGpu, tCpu, tTotal,
    idleFrac: tCpu / tTotal,
    bound: tFlop > tMem ? 'compute' : 'memory',
    tokensPerUser: 1 / tTotal,
    totalTokens: w.batch / tTotal,
    poolTokens,
    fitsWeights: weightBytes + reserve < gpu.mem,
  };
}

export const engineRows = (w: Workload) => ENGINES.map((e) => breakdown(e, w));

/**
 * The three kinds of column a comparison may have, kept apart on purpose. Mixing a
 * measurement with a guess in one cell is how a spreadsheet invents a ranking.
 */
export interface Column {
  id: string;
  label: string;
  kind: 'measured' | 'derived' | 'structural';
  value: (b: Breakdown) => string;
  note: string;
}

export const COLUMNS: Column[] = [
  {
    id: 'thr', label: 'total tok/s', kind: 'measured',
    value: (b) => Math.round(b.totalTokens).toLocaleString('en-US'),
    note: 'output tokens per second across every stream in the step',
  },
  {
    id: 'per', label: 'per user', kind: 'measured',
    value: (b) => `${b.tokensPerUser.toFixed(0)} tok/s`,
    note: 'one stream, one step — the interactive number',
  },
  {
    id: 'idle', label: 'GPU idle', kind: 'derived',
    value: (b) => `${(b.idleFrac * 100).toFixed(0)}%`,
    note: 'share of the step spent waiting on the CPU, not doing math',
  },
  {
    id: 'mem', label: 'weights', kind: 'derived',
    value: (b) => `${b.weightGB.toFixed(1)} GB`,
    note: 'params x bytes — exact arithmetic, no benchmark required',
  },
  {
    id: 'pool', label: 'KV pool', kind: 'derived',
    value: (b) => `${(b.poolTokens / 1000).toFixed(0)}K tok`,
    note: 'context that fits in memory beside those weights',
  },
  {
    id: 'build', label: 'build step', kind: 'structural',
    value: (b) => (b.engine.id === 'trtllm' ? 'per config' : 'none'),
    note: 'from the documentation, at the time of writing',
  },
  {
    id: 'vendor', label: 'needs NVIDIA', kind: 'structural',
    value: (b) => (b.engine.id === 'trtllm' ? 'yes' : 'no'),
    note: 'at the time of writing: vLLM also runs on AMD ROCm, Google TPU, x86 CPU and Intel Gaudi; SGLang on AMD ROCm and CPU; llama.cpp on CPU, Apple Metal and Vulkan',
  },
];

export const KIND_COLOR: Record<Column['kind'], string> = {
  measured: 'var(--token)',
  derived: 'var(--memory)',
  structural: 'var(--compute)',
};
