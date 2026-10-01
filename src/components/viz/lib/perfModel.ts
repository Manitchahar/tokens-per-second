// A first-principles model of LLM inference speed, memory and cost.
// Every number here is something the learner derives in the course; nothing is magic.
// Spec-sheet numbers are approximate public figures; prices are rough on-demand cloud rates (illustrative).

export interface Gpu { id: string; name: string; mem: number; bw: number; flops: number; price: number }
export interface Model { id: string; name: string; params: number; layers: number; hidden: number; heads: number; kvHeads: number; headDim: number; vocab: number }

export const GPUS: Gpu[] = [
  { id: 't4', name: 'T4 (free Colab)', mem: 16e9, bw: 320e9, flops: 65e12, price: 0.35 },
  { id: 'l4', name: 'L4', mem: 24e9, bw: 300e9, flops: 121e12, price: 0.8 },
  { id: 'a100', name: 'A100 80GB', mem: 80e9, bw: 2.0e12, flops: 312e12, price: 1.6 },
  { id: 'h100', name: 'H100 SXM', mem: 80e9, bw: 3.35e12, flops: 989e12, price: 2.5 },
  { id: 'h200', name: 'H200', mem: 141e9, bw: 4.8e12, flops: 989e12, price: 3.5 },
  { id: 'b200', name: 'B200', mem: 180e9, bw: 8.0e12, flops: 2250e12, price: 5.5 },
];

export const MODELS: Model[] = [
  { id: 'qwen05', name: 'Qwen2.5-0.5B', params: 0.49e9, layers: 24, hidden: 896, heads: 14, kvHeads: 2, headDim: 64, vocab: 151936 },
  { id: 'llama3b', name: 'Llama-3.2-3B', params: 3.2e9, layers: 28, hidden: 3072, heads: 24, kvHeads: 8, headDim: 128, vocab: 128256 },
  { id: 'llama8b', name: 'Llama-3.1-8B', params: 8.03e9, layers: 32, hidden: 4096, heads: 32, kvHeads: 8, headDim: 128, vocab: 128256 },
  { id: 'qwen32b', name: 'Qwen2.5-32B', params: 32.8e9, layers: 64, hidden: 5120, heads: 40, kvHeads: 8, headDim: 128, vocab: 152064 },
  { id: 'llama70b', name: 'Llama-3.1-70B', params: 70.6e9, layers: 80, hidden: 8192, heads: 64, kvHeads: 8, headDim: 128, vocab: 128256 },
];

export const gpuById = (id: string) => GPUS.find((g) => g.id === id)!;
export const modelById = (id: string) => MODELS.find((m) => m.id === id)!;

/** Bytes of KV cache per token: 2 (K and V) × layers × kv heads × head dim × bytes per number. */
export const kvBytesPerToken = (m: Model, bytes = 2) => 2 * m.layers * m.kvHeads * m.headDim * bytes;

export interface StepInput {
  model: Model;
  gpu: Gpu;
  batch: number;
  ctx: number; // average tokens of context per sequence (prompt + generated so far)
  wBytes?: number; // bytes per weight (2 = BF16, 1 = FP8/INT8, 0.5 = 4-bit)
  kvBytes?: number;
  overhead?: number; // fixed seconds per step (kernel launches, scheduling, sampling)
  mbu?: number; // achievable fraction of peak bandwidth
  mfu?: number; // achievable fraction of peak FLOP/s
}

export function decodeStep(p: StepInput) {
  const { model, gpu, batch, ctx } = p;
  const wBytes = p.wBytes ?? 2;
  const kvB = p.kvBytes ?? 2;
  const overhead = p.overhead ?? 0.0015;
  const mbu = p.mbu ?? 0.8;
  const mfu = p.mfu ?? 0.55;
  const weights = model.params * wBytes;
  const kvPerSeq = ctx * kvBytesPerToken(model, kvB);
  const kvRead = batch * kvPerSeq;
  const tWeights = weights / (gpu.bw * mbu);
  const tKv = kvRead / (gpu.bw * mbu);
  const tMem = tWeights + tKv;
  const tCompute = (2 * model.params * batch) / (gpu.flops * mfu);
  const t = Math.max(tMem, tCompute) + overhead;
  const activations = 0.04 * gpu.mem; // workspace, activations, CUDA context: a rough reserve
  const memUsed = weights + kvRead + activations;
  return {
    t,
    tWeights,
    tKv,
    tMem,
    tCompute,
    overhead,
    perUser: 1 / t,
    total: batch / t,
    weights,
    kvPerSeq,
    kvTotal: kvRead,
    activations,
    memUsed,
    fits: memUsed <= gpu.mem,
    bound: tCompute > tMem ? ('compute' as const) : ('memory' as const),
    costPerMTok: (gpu.price / 3600 / (batch / t)) * 1e6,
  };
}

/** Largest batch that fits in memory at this context length. */
export function maxBatch(p: Omit<StepInput, 'batch'>) {
  const r = decodeStep({ ...p, batch: 1 });
  const free = p.gpu.mem - r.weights - r.activations;
  return Math.max(0, Math.floor(free / r.kvPerSeq));
}

/** Prefill time for a prompt of n tokens (compute-bound once n is more than a few dozen). */
export function prefillTime(p: { model: Model; gpu: Gpu; tokens: number; wBytes?: number; mfu?: number; mbu?: number }) {
  const wBytes = p.wBytes ?? 2;
  const tCompute = (2 * p.model.params * p.tokens) / (p.gpu.flops * (p.mfu ?? 0.55));
  const tMem = (p.model.params * wBytes) / (p.gpu.bw * (p.mbu ?? 0.8));
  return Math.max(tCompute, tMem) + 0.002;
}
