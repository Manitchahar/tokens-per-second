// Shared math for Module 18 · Multi-GPU Parallelism.
//
// Every number here is derived in the lessons, from the same perfModel constants the rest of
// the course uses. Bandwidths are public spec-sheet figures (approximate, and quoted per
// *direction* where the vendor quotes bidirectional) — nothing is invented.

import { GPUS, type Gpu } from '../lib/perfModel';

export type { Gpu };
export const gpuById = (id: string) => GPUS.find((g) => g.id === id)!;

/** An interconnect. `bw` is unidirectional peak; `latency` is a small-message round trip. */
export interface Link {
  id: string;
  name: string;
  bw: number; // bytes/s, one direction
  latency: number; // seconds
  eff: number; // fraction of peak a real collective achieves
  scope: string;
  note: string;
}

export const LINKS: Link[] = [
  { id: 'pcie4', name: 'PCIe 4.0 x16', bw: 32e9, latency: 2.5e-6, eff: 0.8, scope: 'host ↔ GPU', note: 'the default path for any card that is not NVLink-attached' },
  { id: 'pcie5', name: 'PCIe 5.0 x16', bw: 64e9, latency: 2.0e-6, eff: 0.8, scope: 'host ↔ GPU', note: 'H100 and later host links' },
  { id: 'nvlink3', name: 'NVLink 3 (A100 SXM)', bw: 300e9, latency: 1.2e-6, eff: 0.85, scope: 'inside one 8-GPU node', note: '600 GB/s quoted bidirectional' },
  { id: 'nvlink4', name: 'NVLink 4 (H100 SXM)', bw: 450e9, latency: 0.9e-6, eff: 0.85, scope: 'inside one 8-GPU node', note: '900 GB/s quoted bidirectional' },
  { id: 'nvlink5', name: 'NVLink 5 (B200)', bw: 900e9, latency: 0.8e-6, eff: 0.85, scope: 'inside one 8-GPU node', note: '1.8 TB/s quoted bidirectional' },
  { id: 'ib', name: 'InfiniBand NDR', bw: 50e9, latency: 2.0e-6, eff: 0.85, scope: 'between nodes', note: '400 Gb/s per port; the only sane way to cross a rack' },
];

export const linkById = (id: string) => LINKS.find((l) => l.id === id)!;

// ── models, with the extra fields this module needs ────────────────────────────

export interface BigModel {
  id: string;
  name: string;
  params: number;
  layers: number;
  hidden: number;
  inter: number; // MLP intermediate size
  heads: number;
  kvHeads: number;
  headDim: number;
}

export const BMODELS: BigModel[] = [
  { id: 'qwen15b', name: 'Qwen2.5-1.5B', params: 1.54e9, layers: 28, hidden: 1536, inter: 8960, heads: 12, kvHeads: 2, headDim: 128 },
  { id: 'llama8b', name: 'Llama-3.1-8B', params: 8.03e9, layers: 32, hidden: 4096, inter: 14336, heads: 32, kvHeads: 8, headDim: 128 },
  { id: 'qwen32b', name: 'Qwen2.5-32B', params: 32.8e9, layers: 64, hidden: 5120, inter: 17408, heads: 40, kvHeads: 8, headDim: 128 },
  { id: 'llama70b', name: 'Llama-3.1-70B', params: 70.6e9, layers: 80, hidden: 8192, inter: 28672, heads: 64, kvHeads: 8, headDim: 128 },
];

export const bmodelById = (id: string) => BMODELS.find((m) => m.id === id)!;

// ── sizes ─────────────────────────────────────────────────────────────────────

/** Bytes of KV cache per token, BF16: 2 (K and V) × layers × kv heads × head dim × 2. */
export const kvBytesPerToken = (m: BigModel) => 2 * m.layers * m.kvHeads * m.headDim * 2;

/** Bytes of weights at `bytes` per weight (2 = BF16, 1 = FP8/INT8, 0.5 = 4-bit). */
export const weightBytes = (m: BigModel, bytes = 2) => m.params * bytes;

/** What one MLP layer's weights cost, so the split across P GPUs is exact. */
export const mlpBytesPerLayer = (m: BigModel, bytes = 2) => 3 * m.hidden * m.inter * bytes;

/** What one attention layer's weights cost. */
export const attnBytesPerLayer = (m: BigModel, bytes = 2) => m.hidden * m.hidden * bytes + 2 * m.hidden * (m.kvHeads * m.headDim) * bytes;

// ── collectives ───────────────────────────────────────────────────────────────

/** Ring algorithms move (P-1)/P of the payload in each of two phases. */
export const ringFactor = (P: number) => (P <= 1 ? 0 : (2 * (P - 1)) / P);
/** All-gather / reduce-scatter move (P-1)/P of the payload in one phase. */
export const gatherFactor = (P: number) => (P <= 1 ? 0 : (P - 1) / P);

export const allReduceBytes = (payload: number, P: number) => ringFactor(P) * payload;
export const allGatherBytes = (payload: number, P: number) => gatherFactor(P) * payload;

/** Time to move `bytes` over a link, including the fixed cost of getting started. */
export const linkTime = (bytes: number, link: Link) => link.latency + bytes / (link.bw * link.eff);

// ── tensor parallelism ────────────────────────────────────────────────────────

/** Degrees of TP a model actually supports: P must divide heads, KV heads and the MLP width. */
export const tpDegrees = (m: BigModel, maxP = 8): number[] => {
  const out: number[] = [];
  for (let P = 1; P <= maxP; P *= 2) {
    if (m.heads % P === 0 && m.kvHeads % P === 0 && m.inter % P === 0) out.push(P);
  }
  return out;
};

/** The KV head count is usually what caps TP, and it is easy to forget. */
export const tpCapReason = (m: BigModel) =>
  m.kvHeads < m.heads
    ? `${m.kvHeads} KV heads cap TP at ${m.kvHeads} — GQA limits the split, not the weights`
    : `${m.heads} query heads, TP up to ${m.heads}`;

/** Size of one row-parallel all-reduce payload: the full hidden state, [batch, tokens, hidden]. */
export const tpMessageBytes = (m: BigModel, batch: number, tokens = 1) => batch * tokens * m.hidden * 2;

/** Two all-reduces per layer, each moving 2(P-1)/P of its payload: bytes per GPU per step. */
export const tpCommBytes = (m: BigModel, batch: number, P: number, tokens = 1) =>
  2 * m.layers * ringFactor(P) * tpMessageBytes(m, batch, tokens);

export interface StepBreakdown {
  tMem: number; // weights, split across P GPUs
  tCompute: number;
  tComm: number;
  t: number;
  memBound: boolean;
  oneGpu: number;
  speedup: number;
  efficiency: number;
  bytesPerGpu: number;
}

export function tpDecodeStep(o: {
  model: BigModel;
  gpu: Gpu;
  P: number;
  link: Link;
  batch: number;
  wBytes?: number;
  mbu?: number;
  mfu?: number;
  overhead?: number;
}): StepBreakdown {
  const { model, gpu, P, link, batch } = o;
  const wBytes = o.wBytes ?? 2;
  const mbu = o.mbu ?? 0.8;
  const mfu = o.mfu ?? 0.55;
  const overhead = (o.overhead ?? 0.0015) * P; // more stages to launch
  const weights = weightBytes(model, wBytes);
  const bytesPerGpu = weights / P;
  const tMem = bytesPerGpu / (gpu.bw * mbu);
  const tCompute = (2 * model.params * batch) / (P * gpu.flops * mfu);
  const tComm = linkTime(tpCommBytes(model, batch, P), link);
  const t = Math.max(tMem, tCompute) + tComm + overhead;
  const oneGpu = Math.max(weights / (gpu.bw * mbu), (2 * model.params * batch) / (gpu.flops * mfu)) + 0.0015;
  return {
    tMem,
    tCompute,
    tComm,
    t,
    bytesPerGpu,
    memBound: tMem >= tCompute,
    oneGpu,
    speedup: oneGpu / t,
    efficiency: oneGpu / (P * t),
  };
}

export function tpPrefillTime(o: {
  model: BigModel;
  gpu: Gpu;
  P: number;
  link: Link;
  tokens: number;
  wBytes?: number;
  mbu?: number;
  mfu?: number;
}): { tCompute: number; tComm: number; overheadFrac: number; messageBytes: number } {
  const { model, gpu, P, link, tokens } = o;
  const mfu = o.mfu ?? 0.55;
  const mbu = o.mbu ?? 0.8;
  const tCompute = (2 * model.params * tokens) / (P * gpu.flops * mfu);
  const messageBytes = tpMessageBytes(model, 1, tokens);
  const tComm = linkTime(2 * model.layers * ringFactor(P) * messageBytes, link);
  const tMem = weightBytes(model, o.wBytes ?? 2) / (P * gpu.bw * mbu);
  const total = Math.max(tCompute, tMem) + tComm;
  return { tCompute, tComm, overheadFrac: tComm / total, messageBytes };
}

// ── pipeline parallelism ──────────────────────────────────────────────────────

/** GPipe bubble: P-1 idle slots in a schedule of P + M - 1. Inference has no backward pass. */
export const bubbleFraction = (P: number, M: number) => (P <= 1 || M <= 0 ? 0 : (P - 1) / (M + P - 1));

/** 1F1B fills the bubble with backward work, so its idle fraction is the forward-only one. */
export const bubbleFraction1F1B = (P: number, M: number) => bubbleFraction(P, Math.max(1, M - P + 1));

/** Bytes crossing one pipeline boundary for one micro-batch. */
export const ppBoundaryBytes = (m: BigModel, batch: number, tokens: number) => batch * tokens * m.hidden * 2;

export const ppBoundaryTime = (m: BigModel, batch: number, tokens: number, link: Link) =>
  linkTime(ppBoundaryBytes(m, batch, tokens), link);

/** Smallest micro-batch count that keeps the forward bubble at or below `target`. */
export const microBatchesFor = (P: number, target = 0.2) => {
  if (P <= 1) return 1;
  let M = P;
  while (bubbleFraction(P, M) > target && M < 512) M *= 2;
  return M;
};

// ── context parallelism ───────────────────────────────────────────────────────

/**
 * Bytes each GPU pushes through the ring to see the whole KV cache: P-1 rotations of a
 * 1/P slice. Independent of how the sequence is split, only of P and the cache size.
 */
export const cpCommBytes = (m: BigModel, batch: number, ctx: number, P: number) =>
  gatherFactor(P) * batch * ctx * kvBytesPerToken(m);

export interface CpVerdict {
  kvTotal: number;
  localBytes: number;
  commBytes: number;
  tLocal: number;
  tComm: number;
  tSingle: number;
  slowdown: number;
}

export function contextParallelStep(o: {
  model: BigModel;
  gpu: Gpu;
  P: number;
  link: Link;
  batch: number;
  ctx: number;
}): CpVerdict {
  const { model, gpu, P, link, batch, ctx } = o;
  const kvTotal = batch * ctx * kvBytesPerToken(model);
  const localBytes = kvTotal / P;
  const commBytes = cpCommBytes(model, batch, ctx, P);
  const tLocal = localBytes / (gpu.bw * 0.8);
  const tComm = linkTime(commBytes, link);
  const tSingle = kvTotal / (gpu.bw * 0.8);
  return { kvTotal, localBytes, commBytes, tLocal, tComm, tSingle, slowdown: (tLocal + tComm) / tSingle };
}

/**
 * The exact CP penalty factor: how much slower the attention read becomes.
 *
 * Per GPU the local slice costs kv/P / B_mem; the ring costs (P-1)/P of the whole
 * cache over one link at B_link. Their ratio is (P-1) * B_mem / (B_link * eff)
 * — independent of cache size, and it is always > 1. Context parallelism therefore
 * never makes a step faster; it only makes a request that would not fit, fit.
 */
export const cpPenalty = (gpu: Gpu, link: Link, P: number) =>
  P <= 1 ? 1 : ((P - 1) * gpu.bw) / (link.bw * link.eff);

// ── the decision rule ─────────────────────────────────────────────────────────

export interface Plan {
  fitsAlone: boolean;
  fitsQuantized: boolean;
  tp: number;
  pp: number;
  replicas: number;
  reason: string;
}

/**
 * The rule from the last lesson, as arithmetic: fit on one GPU; otherwise quantize; then
 * tensor-parallel inside the NVLink island; then pipeline across it; replicate for throughput.
 */
export function planFor(o: {
  model: BigModel;
  gpu: Gpu;
  link: Link;
  gpus: number;
  batch: number;
  ctx: number;
  wBytes?: number;
  overheadPerGpu?: number;
  island?: number;
}): Plan {
  const { model, gpu, gpus, batch, ctx, wBytes = 2 } = o;
  const island = o.island ?? 8;
  const overheadPerGpu = o.overheadPerGpu ?? 0.04;
  const budget = gpu.mem * (1 - 0.08) - overheadPerGpu * gpu.mem;
  const kv = batch * ctx * kvBytesPerToken(model);

  const fitsAlone = weightBytes(model, wBytes) + kv <= budget;
  const fitsQuantized = weightBytes(model, 1) + kv <= budget;
  if (fitsAlone) {
    return { fitsAlone, fitsQuantized, tp: 1, pp: 1, replicas: Math.max(1, gpus), reason: 'It fits on one GPU. Replicate; do not shard the math.' };
  }
  if (fitsQuantized && wBytes > 1) {
    return { fitsAlone, fitsQuantized, tp: 1, pp: 1, replicas: Math.max(1, gpus), reason: 'FP8 fits where BF16 does not. Quantizing is cheaper than buying a GPU.' };
  }

  const maxTP = Math.max(...tpDegrees(model).filter((p) => p <= Math.min(island, gpus)));

  // Memory on each TP rank: 1/TP of the weights, 1/TP of the KV (the head dimension splits).
  const perRank = (t: number) => weightBytes(model, wBytes) / t + kv / t;
  let tp = maxTP;
  while (tp > 1 && perRank(tp) <= budget) tp /= 2;
  while (tp * 2 <= maxTP && perRank(tp) > budget) tp *= 2;

  // Pipeline stages hold a contiguous slice of layers. Weights split 1/(TP·PP);
  // the KV cache splits 1/TP because a stage needs every head of its layers.
  const perStage = (t: number, p: number) => weightBytes(model, wBytes) / (t * p) + kv / t;
  let pp = 1;
  while (perStage(tp, pp) > budget && tp * pp * 2 <= gpus) pp *= 2;

  const replicas = Math.max(1, Math.floor(gpus / (tp * pp)));
  return {
    fitsAlone,
    fitsQuantized,
    tp,
    pp,
    replicas,
    reason:
      tp > 1
        ? `Too big for one card: TP=${tp} inside the NVLink island, then PP=${pp} across it, ${replicas} replica${replicas > 1 ? 's' : ''}.`
        : 'This model cannot be tensor-parallelised past 1; the split is pipeline-only.',
  };
}

export const gpuPeakTps = (model: BigModel, gpu: Gpu, batch: number, ctx: number, P = 1) => {
  const kv = batch * ctx * kvBytesPerToken(model);
  const tMem = weightBytes(model) / (P * gpu.bw * 0.8) + kv / (gpu.bw * 0.8);
  const tFlop = (2 * model.params * batch) / (P * gpu.flops * 0.55);
  return (P * batch) / Math.max(tMem, tFlop);
};