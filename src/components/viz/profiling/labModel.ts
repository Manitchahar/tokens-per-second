// Module 12 · Profiling & GPU Performance — the lab's model.
//
// A deliberately simplified engine: one GPU, one model, and a handful of knobs.
// It is NOT a simulator of vLLM. It exists to make the roofline and the profiling
// funnel *legible*: you can turn one thing at a time and watch one number move.

export interface Config {
  batch: number;        // sequences decoding together
  ctx: number;          // average context length, tokens
  wBytes: number;       // bytes per weight: 2 = BF16, 1 = FP8, 0.5 = 4-bit
  kvBytes: number;      // bytes per KV number
  fused: boolean;       // elementwise ops fused into one kernel
  coalesced: boolean;   // all loads perfectly coalesced
  cudaGraph: boolean;   // the decode loop is captured as a CUDA graph
  attnFlash: boolean;   // attention uses the memory-efficient kernel
}

export const DEFAULT_CONFIG: Config = {
  batch: 32,
  ctx: 4096,
  wBytes: 2,
  kvBytes: 2,
  fused: false,
  coalesced: true,
  cudaGraph: false,
  attnFlash: true,
};

export const PARAMS = 8.03e9;
export const LAYERS = 32;
export const HIDDEN = 4096;
export const INTER = 14336;      // Llama-3.1-8B MLP inner width
export const HEADS = 32;
export const KV_HEADS = 8;
export const HEAD_DIM = 128;
export const VOCAB = 128256;

/** Bytes of KV cache per token, in BF16. */
export const KV_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2; // 131,072

/** Roofline constants for an H100, so the numbers line up with the lessons. */
export const H100 = { bw: 3.35e12, flops: 989e12 };

/** Achievable fractions a well-tuned kernel reaches. */
export const MBU = 0.82;   // memory bandwidth utilization, good kernel
export const MFU = 0.60;   // model FLOP utilization, good GEMM

export interface Breakdown {
  weightBytes: number;
  kvBytes: number;
  attnBytes: number;
  totalBytes: number;
  flops: number;
  intensity: number;
  bound: 'compute' | 'memory';
  memoryFloor: number;
  computeFloor: number;
  idealStep: number;
  // what the launch configuration actually costs
  gemmTime: number;
  attnTime: number;
  elementwiseTime: number;
  normTime: number;
  launchOverhead: number;
  stepTime: number;
  achievedBps: number;
  mbu: number;
  mfu: number;
  headroom: number;
  tokensPerSecTotal: number;
  tokensPerSecPerUser: number;
  nKernels: number;
}

/** Pure matrix multiply the decode step performs, per layer. */
const GEMM_FLOPS_PER_LAYER = 2 * PARAMS * 1.0; // one token per sequence, scaled by batch below

export function analyze(c: Config): Breakdown {
  const batch = Math.max(1, c.batch);
  const ctx = Math.max(16, c.ctx);

  // ── the byte budget ──────────────────────────────────────────────
  const weightBytes = PARAMS * c.wBytes;
  const kvBytes = batch * ctx * KV_PER_TOKEN * (c.kvBytes / 2);

  // A real step splits its KV traffic in two: the GEMMs stream the K/V projections
  // when they write them, and the attention kernel gathers K and V for every context
  // token. Reading `kvBytes` once is the honest floor; `kvShare` is the part the
  // GEMMs pay for and `attnShare` the part attention pays for.
  const kvShare = kvBytes * 0.55;
  const attnShare = kvBytes * 0.45;

  // A naive attention kernel also materializes the T×T score matrix, which costs
  // real extra traffic; FlashAttention does not.
  const scoreMatrixBytes = batch * ctx * ctx * (c.kvBytes / 2);
  const attnBytes = attnShare + (c.attnFlash ? 0 : scoreMatrixBytes);

  // ── the FLOP count ───────────────────────────────────────────────
  const flops = 2 * PARAMS * batch;

  // The byte budget as the hardware sees it: weights once, KV once, plus whatever
  // extra the attention implementation wastes.
  const totalBytes = weightBytes + kvBytes + (c.attnFlash ? 0 : scoreMatrixBytes);
  const intensity = flops / totalBytes;
  const bound: 'compute' | 'memory' = intensity > H100.flops / H100.bw ? 'compute' : 'memory';

  const memoryFloor = totalBytes / H100.bw;
  const computeFloor = flops / H100.flops;
  const idealStep = Math.max(memoryFloor, computeFloor);

  // ── what the step actually costs, broken into the bars a trace shows ──
  // Each bar's time is its own bytes at a realistic efficiency for that access
  // pattern, so the bars sum to the step and the step's achieved bandwidth is a
  // real measurement rather than a derived wish.
  const coalescePenalty = c.coalesced ? 1 : 1.35; // a strided weight read
  const gemmBytes = (weightBytes + kvShare) * coalescePenalty;
  const gemmTime = gemmBytes / (H100.bw * MBU);

  // Attention reads its share of the KV at a lower efficiency because a gather is
  // not a pure stream, and a non-flash kernel additionally writes and re-reads
  // the score matrix.
  const attnTraffic = attnShare + (c.attnFlash ? 0 : scoreMatrixBytes * 2);
  const attnTime = attnTraffic / (H100.bw * (c.attnFlash ? 0.7 : 0.5));

  // Elementwise ops (SiLU, the multiply, the residual adds) stream the activation.
  const actElems = batch * INTER;
  const actBytes = actElems * 2;
  const elementwiseTime = (c.fused ? actBytes * 1.0 : actBytes * 2.6) / (H100.bw * MBU);

  const normBytes = batch * HIDDEN * 2 * LAYERS * 2;
  const normTime = normBytes / (H100.bw * MBU);

  // Kernel launch overhead: ~5 µs each, removed by CUDA graphs.
  const nKernels = (c.fused ? 7 : 13) * LAYERS + 8;
  const launchOverhead = c.cudaGraph ? nKernels * 0.0000008 : nKernels * 0.000005;

  const stepTime = gemmTime + attnTime + elementwiseTime + normTime + launchOverhead;

  // Achieved bandwidth counts the bytes the hardware had to move. A step can never
  // be faster than its roofline floor, so the reported MBU is capped at 100% and
  // the headroom is floored at 1.0 — the model is a teaching aid, and a number that
  // says "114% of peak" would teach the wrong lesson.
  const achievedBps = totalBytes / stepTime;
  const mbu = Math.min(1, achievedBps / H100.bw);
  const mfu = Math.min(1, flops / stepTime / H100.flops);

  return {
    weightBytes,
    kvBytes,
    attnBytes,
    totalBytes,
    flops,
    intensity,
    bound,
    memoryFloor,
    computeFloor,
    idealStep,
    gemmTime,
    attnTime,
    elementwiseTime,
    normTime,
    launchOverhead,
    stepTime,
    achievedBps,
    mbu,
    mfu,
    headroom: Math.max(1, stepTime / idealStep),
    tokensPerSecTotal: batch / stepTime,
    tokensPerSecPerUser: 1 / stepTime,
    nKernels,
  };
}

/** The grade a report would get, from the two numbers the rubric scores. */
export function verdict(b: Breakdown): { ok: boolean; msg: string; kind: 'token' | 'error' | 'compute' } {
  const eff = b.bound === 'memory' ? b.mbu : b.mfu;
  if (b.headroom > 1.6) {
    return { ok: false, kind: 'error', msg: `${b.headroom.toFixed(2)}× headroom — a big win is still on the table` };
  }
  if (eff < 0.55) {
    return { ok: false, kind: 'error', msg: `${(eff * 100).toFixed(0)}% ${b.bound === 'memory' ? 'MBU' : 'MFU'} — below 55%, the kernel is the problem` };
  }
  if (b.headroom < 1.15 && eff > 0.8) {
    return { ok: true, kind: 'token', msg: `at the floor: ${(eff * 100).toFixed(0)}% of the speed of light` };
  }
  return { ok: true, kind: 'compute', msg: `efficient: ${(eff * 100).toFixed(0)}% ${b.bound === 'memory' ? 'MBU' : 'MFU'}, ${b.headroom.toFixed(2)}× headroom` };
}

/** The knobs, in the order a bottleneck report should discuss them. */
export const KNOBS: { key: keyof Config; label: string; detail: string; kind: 'compute' | 'memory' | 'token' | 'comm' }[] = [
  { key: 'batch', label: 'Batch', detail: 'sequences decoding together', kind: 'compute' },
  { key: 'ctx', label: 'Context', detail: 'average tokens per sequence', kind: 'memory' },
  { key: 'wBytes', label: 'Weight bytes', detail: 'BF16 → FP8 → 4-bit', kind: 'memory' },
  { key: 'kvBytes', label: 'KV bytes', detail: 'BF16 → FP8 KV cache', kind: 'memory' },
  { key: 'attnFlash', label: 'Flash attention', detail: 'memory-efficient attention', kind: 'compute' },
  { key: 'fused', label: 'Fused elementwise', detail: 'one kernel instead of three', kind: 'compute' },
  { key: 'coalesced', label: 'Coalesced loads', detail: 'off = 35% wasted sectors', kind: 'memory' },
  { key: 'cudaGraph', label: 'CUDA graphs', detail: 'remove per-launch overhead', kind: 'comm' },
];
