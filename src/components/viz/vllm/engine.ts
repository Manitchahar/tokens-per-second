// Shared numbers for the Module 09 visuals.
//
// Everything here is derived with the same arithmetic the learner does in the
// lessons: KV bytes per token from the model's shape, and a KV cache pool that
// is "whatever is left of the memory budget after the weights".
//
// vLLM's block size is 16 tokens per block in the V1 engine at the time of
// writing; it is a flag (`--block-size`), so treat it as the default, not a law.

import { kvBytesPerToken, type Gpu, type Model } from '../lib/perfModel';

/** Tokens of context held by one KV cache block (vLLM V1 default). */
export const KV_BLOCK_TOKENS = 16;

export interface LaunchPlan {
  /** Total bytes vLLM is allowed to touch: util × GPU memory. */
  budget: number;
  /** Weights on the GPU. */
  weights: number;
  /** CUDA context, activations and the profiling run's peak — the reserve. */
  reserve: number;
  /** What is left for the KV cache. */
  pool: number;
  /** How many tokens of context the pool can hold at once. */
  poolTokens: number;
  /** How many 16-token blocks that is. */
  blocks: number;
  /** True when the weights plus the reserve fit inside the budget. */
  fits: boolean;
}

/** Reproduce vLLM's startup budget on paper: budget → weights + reserve + KV pool. */
export function launchPlan(model: Model, gpu: Gpu, util = 0.9, wBytes = 2): LaunchPlan {
  const budget = gpu.mem * util;
  const weights = model.params * wBytes;
  const reserve = 0.04 * gpu.mem;
  const pool = Math.max(0, budget - weights - reserve);
  const per = kvBytesPerToken(model, wBytes);
  return {
    budget,
    weights,
    reserve,
    pool,
    poolTokens: per > 0 ? Math.floor(pool / per) : 0,
    blocks: per > 0 ? Math.floor(pool / (per * KV_BLOCK_TOKENS)) : 0,
    fits: pool > 0,
  };
}

/** How many sequences of `ctx` tokens each hold their own KV cache simultaneously. */
export const concurrencyFor = (poolTokens: number, ctx: number) => (ctx > 0 ? Math.floor(poolTokens / ctx) : 0);

/** The streamed answer used by the visuals: a short, real-looking completion. */
export const ANSWER_TOKENS = [
  'Paged', 'Attention', ' lets', ' an', ' engine', ' keep', ' many', ' conversations', ' in', ' memory',
  ' by', ' giving', ' each', ' one', ' fixed', '-', 'size', ' blocks', ' instead', ' of', ' a', ' fresh', ' contiguous', ' buffer',
  '.', ' Try', ' it', ':', ' ask', ' me', ' anything',
];

/** Log lines vLLM prints at startup (paraphrased from the usual v1 log). */
export const BOOT_LOG = [
  'INFO  Loading model weights from safetensors ...',
  'INFO  Loading safetensors checkpoint shards: 100%',
  'INFO  Capturing CUDA graphs (decode) ...',
  'INFO  GPU KV cache size: 389,120 tokens',
  'INFO  Maximum concurrency for 32,768 tokens per request: 11.8x',
];