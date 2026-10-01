// A first-principles cost model for the Module 17 fair-comparison visual.
//
// IMPORTANT: this is a MODEL of the *shape* of the trade-off, not a benchmark of any
// real engine. Every number it produces comes from quantities the learner derives in the
// course — weight bytes, KV bytes, achievable bandwidth, fixed per-step cost. The engine
// rows differ only in three honest, structural ways: how much fixed cost each decode step
// adds (Python scheduling and kernel dispatch versus a compiled plan), what fraction of
// peak bandwidth its kernels reach, and whether it has a 4-bit path that fits a model the
// others cannot load at all. It never claims who is faster.

import { kvBytesPerToken, modelById, gpuById } from '../lib/perfModel';

export type ProfileId = 'chat' | 'longctx' | 'batch' | 'edge';

export interface Profile {
  id: ProfileId;
  label: string;
  blurb: string;
  modelId: string;
  gpuId: string;
  prompt: number;      // prompt tokens per request
  ctx: number;         // steady-state context: prompt + generated so far
  shared: number;      // leading prompt tokens byte-identical across requests (prefix cache hit)
  wantBatch: number;   // the concurrency this traffic asks the engine for
  out: number;         // generated tokens per request
  sloTTFTms: number;   // the promise the product makes to a user
  sloTPOTms: number;
}

export interface EngineSpec {
  id: string;
  name: string;
  overheadMs: number;  // fixed cost per decode step: scheduling + kernel launch
  mbu: number;         // achievable fraction of peak memory bandwidth
  mfu: number;         // achievable fraction of peak FLOP/s
  q4: boolean;         // has a weight-only 4-bit path
  trade: string;
}

export interface Row {
  engine: EngineSpec;
  weightGB: number;
  fits: boolean;
  why: string;
  batch: number;
  tStep: number;
  throughput: number;
  perUser: number;
  ttftMs: number;
  meets: boolean;
}

export const ENGINES: EngineSpec[] = [
  { id: 'vllm', name: 'vLLM', overheadMs: 1.5, mbu: 0.78, mfu: 0.55, q4: false,
    trade: 'paged blocks + continuous batching, scheduled in Python' },
  { id: 'sglang', name: 'SGLang', overheadMs: 1.6, mbu: 0.8, mfu: 0.55, q4: false,
    trade: 'prefix cache is a radix tree the kernel walks directly' },
  { id: 'trtllm', name: 'TensorRT-LLM', overheadMs: 0.6, mbu: 0.82, mfu: 0.65, q4: false,
    trade: 'an execution plan built once, replayed with few launches' },
  { id: 'llamacpp', name: 'llama.cpp', overheadMs: 4.0, mbu: 0.6, mfu: 0.3, q4: true,
    trade: '4-bit weights and no CUDA requirement, on any machine' },
];

export const PROFILES: Profile[] = [
  {
    id: 'chat', label: 'chat',
    blurb: 'Llama-3.1-8B on one H100. Short prompts, 512 tokens of shared system prompt, 256 tokens out.',
    modelId: 'llama8b', gpuId: 'h100', prompt: 512, ctx: 2048, shared: 512, wantBatch: 16,
    out: 256, sloTTFTms: 800, sloTPOTms: 50,
  },
  {
    id: 'longctx', label: 'long ctx',
    blurb: 'Llama-3.1-8B on one H100. 24k-token prompts, nothing shared, 16 concurrent streams.',
    modelId: 'llama8b', gpuId: 'h100', prompt: 24576, ctx: 28672, shared: 0, wantBatch: 16,
    out: 256, sloTTFTms: 6000, sloTPOTms: 80,
  },
  {
    id: 'batch', label: 'batch',
    blurb: 'Llama-3.1-8B on one H100. Nobody is waiting, so the SLO is deliberately loose.',
    modelId: 'llama8b', gpuId: 'h100', prompt: 2048, ctx: 4096, shared: 0, wantBatch: 128,
    out: 512, sloTTFTms: 30000, sloTPOTms: 400,
  },
  {
    id: 'edge', label: 'edge',
    blurb: 'Qwen2.5-32B on a single 24 GB L4. Nobody is waiting; the question is whether it fits at all.',
    modelId: 'qwen32b', gpuId: 'l4', prompt: 512, ctx: 1024, shared: 0, wantBatch: 8,
    out: 128, sloTTFTms: 5000, sloTPOTms: 400,
  },
];

export const profileById = (id: ProfileId) => PROFILES.find((p) => p.id === id)!;

/** One engine, one traffic profile, one offered load. Every field is arithmetic you can redo. */
export function evaluate(spec: EngineSpec, p: Profile, load: number): Row {
  const model = modelById(p.modelId);
  const gpu = gpuById(p.gpuId);
  const bytesPerWeight = spec.q4 && p.id === 'edge' ? 0.5 : 2;
  const weightBytes = model.params * bytesPerWeight;
  const reserve = 0.04 * gpu.mem;
  const kvTok = kvBytesPerToken(model, 2);
  // A shared prefix is stored once (refcounted blocks) and read once per step, not once per sequence.
  const perSeq = (p.ctx - p.shared) * kvTok;
  const free = gpu.mem - weightBytes - reserve;
  const base: Row = {
    engine: spec, weightGB: weightBytes / 1e9, fits: false, why: '', batch: 0,
    tStep: 0, throughput: 0, perUser: 0, ttftMs: Infinity, meets: false,
  };

  if (weightBytes + reserve >= gpu.mem) {
    base.why = `weights alone need ${(weightBytes / 1e9).toFixed(1)} GB`;
    return base;
  }
  const batch = Math.max(0, Math.min(p.wantBatch, Math.floor(free / Math.max(1, perSeq))));
  if (batch < 1) {
    base.why = `KV cache: ${batch} of ${p.wantBatch} streams fit`;
    return base;
  }

  const kvRead = p.shared * kvTok + batch * perSeq;
  const tMem = (weightBytes + kvRead) / (gpu.bw * spec.mbu);
  const tFlop = (2 * model.params * batch) / (gpu.flops * spec.mfu);
  const tStep = Math.max(tMem, tFlop) + spec.overheadMs / 1000;
  const tPrefill = Math.max((2 * model.params * p.prompt) / (gpu.flops * spec.mfu), weightBytes / (gpu.bw * spec.mbu)) + 0.002;

// Little's law for a pool of B equal servers: queue wait grows as rho/(1-rho) x S/B.
  const serviceS = tPrefill + p.out * tStep;
  const rho = Math.min(Math.max(load, 0), 0.95);
  const waitS = (rho / (1 - rho)) * (serviceS / batch);
  const ttftMs = (waitS + tPrefill) * 1000;

  return {
    engine: spec, weightGB: weightBytes / 1e9, fits: true,
    why: batch < p.wantBatch ? `KV pool caps it at ${batch} streams` : '',
    batch, tStep, throughput: batch / tStep, perUser: 1 / tStep, ttftMs,
    meets: ttftMs <= p.sloTTFTms && tStep * 1000 <= p.sloTPOTms,
  };
}