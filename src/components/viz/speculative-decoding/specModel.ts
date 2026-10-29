// The arithmetic of speculative decoding, derived once so every visual and every
// lesson in this module quote the same numbers.
//
// TWO quantities must never be confused, because both are called "alpha" in the
// wild:
//
//   a    — ACCEPTANCE RATE: how often a drafted token is one the target model
//          would also have produced. Purely about text. Decides how many tokens
//          leave per block.
//   rho — DRAFT COST RATIO: the price of one drafted token, in units of one
//          target decode step. Purely about hardware. Decides what drafting costs.
//
// The load-bearing hardware fact: a decode step is memory-bound, so its cost is
// set by reading the weights, and that does not depend on how many positions the
// step processes. Running the target on gamma+1 positions therefore costs almost
// the same wall-clock as running it on one — that is why a block of drafts can be
// checked for nearly free. "Nearly": the block does gamma times the FLOPs, and at
// some point that stops being negligible. The best gamma comes mostly from the
// drafting side, though: every draft costs rho, and later drafts are kept less often.

import { GPUS } from '../lib/perfModel';

export const SPEC_GPUS = ['t4', 'a100', 'h100'] as const;
export type SpecGpuId = (typeof SPEC_GPUS)[number];
export const specGpu = (id: SpecGpuId) => GPUS.find((g) => g.id === id)!;

export interface Machine {
  bw: number; // bytes per second
  flops: number; // FLOP per second
  weights: number; // bytes of target weights
  kvBytesPerToken: number;
  ctx: number; // average context length in tokens
  params: number; // target parameter count (for FLOPs)
  mbu?: number;
  mfu?: number;
  overhead?: number; // fixed seconds per forward pass: launches + scheduler
}

/** Cost of a forward pass over `n` positions, in seconds. max(memory, compute) + overhead. */
export function forwardPass(m: Machine, n: number) {
  const mbu = m.mbu ?? 0.8;
  const mfu = m.mfu ?? 0.55;
  const overhead = m.overhead ?? 0.0015; // kernel launches + scheduler (Module 15)
  const kvRead = m.ctx * m.kvBytesPerToken;
  const kvWrite = n * m.kvBytesPerToken;
  const memory = (m.weights + kvRead + kvWrite) / (m.bw * mbu);
  const compute = (2 * m.params * n) / (m.flops * mfu);
  return { memory, compute, overhead, total: Math.max(memory, compute) + overhead, bound: compute > memory ? ('compute' as const) : ('memory' as const) };
}

/** One ordinary decode step: the baseline everything is compared against. */
export const decodeStep = (m: Machine) => forwardPass(m, 1).total;

/** One block's verification pass: gamma+1 positions, one forward pass. */
export const verifyPass = (m: Machine, gamma: number) => forwardPass(m, gamma + 1);

/** One drafted token, on the cheap model. rho is that price in decode-step units. */
export const draftStep = (rho: number, tDecode: number) => rho * tDecode;

export interface BlockResult {
  gamma: number;
  /** Drafted tokens per block: always gamma. The draft model proposes all of them before the target verifies any. */
  draftSteps: number;
  /** Expected drafts the target accepts: a + a^2 + ... + a^gamma. */
  accepted: number;
  /** Keep rate: the fraction of drafted tokens that survive, accepted / gamma. */
  keepRate: number;
  /** Tokens that leave the block: the accepted run plus one corrected-or-bonus token. */
  tokens: number;
  /** Target forward passes per delivered token, 1 / tokens. */
  rounds: number;
  /** Cost in seconds (or decode-step units when tDecode = 1). */
  cost: number;
  perToken: number;
  speedup: number;
  /** Drafted tokens thrown away: gamma minus the accepted ones. */
  wasted: number;
  /** Probability every draft in the block is accepted. */
  allAccepted: number;
}

/** Expected accepted drafts in a block of gamma: sum of a^i for i = 1..gamma = a (1 - a^gamma) / (1 - a). */
export function expectedAccepted(a: number, gamma: number) {
  const g = Math.max(1, Math.round(gamma));
  return Math.abs(1 - a) < 1e-9 ? g : (a * (1 - Math.pow(a, g))) / (1 - a);
}

/**
 * One block of standard speculative decoding (Leviathan et al. 2023; Chen et al. 2023).
 *
 * The draft model proposes all gamma tokens, one after another, so all gamma draft
 * steps are paid for. The target then verifies the gamma drafts in ONE forward pass
 * over gamma+1 positions, keeps the prefix up to the first rejection, and adds one
 * more token: a corrected token at the rejected position, or a bonus token if all
 * gamma were accepted. With each draft accepted independently with probability a:
 *
 *   accepted drafts  = a + a^2 + ... + a^gamma = a (1 - a^gamma) / (1 - a)
 *   tokens out       = 1 + accepted          = (1 - a^(gamma+1)) / (1 - a)
 *   cost             = verify + gamma * rho  (drafts you do not keep are still paid)
 */
export function block(a: number, rho: number, gamma: number, tVerify = 1, tDecode = 1): BlockResult {
  const g = Math.max(1, Math.round(gamma));
  const accepted = expectedAccepted(a, g);
  const tokens = 1 + accepted;
  // tVerify and tDecode are SECONDS. The baseline is one plain decode step per token,
  // so the speedup is a true ratio: we emit `tokens` tokens for `cost` seconds.
  const cost = tVerify + rho * tDecode * g;
  return {
    gamma: g,
    draftSteps: g,
    accepted,
    keepRate: accepted / g,
    tokens,
    rounds: 1 / tokens,
    cost,
    perToken: cost / tokens,
    speedup: tokens / (cost / tDecode),
    wasted: g - accepted,
    allAccepted: Math.pow(a, g),
  };
}

/**
 * Break-even acceptance rate: the a at which the block exactly pays for itself.
 *
 * When verification costs exactly one decode step, speedup = (1 + K) / (1 + gamma*rho)
 * with K the expected accepted drafts, so speedup > 1 exactly when K > gamma*rho, i.e.
 * when the KEEP RATE K / gamma beats rho. Each drafted token costs rho of a step and
 * each kept one saves a whole step. For gamma = 1 the keep rate is a, so the rule is
 * a > rho; for longer blocks later drafts are kept less often, so the break-even
 * acceptance rises with gamma. Real verification costs slightly more than a decode
 * step (see verifyPass), which nudges it a little higher still.
 *
 * Both times are in SECONDS.
 */
export function breakEvenAcceptance(rho: number, gamma: number, tVerify: number, tDecode: number) {
  let lo = 0, hi = 0.999999;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (block(mid, rho, gamma, tVerify, tDecode).speedup < 1) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

/** The idealised break-even (verification = one decode step): the a whose keep rate equals rho. */
export const breakEvenIdeal = (rho: number, gamma: number) => breakEvenAcceptance(rho, gamma, 1, 1);

/** Scan gamma and return the best one. It is finite even with free verification, because every draft costs rho and later drafts are kept less often. */
export function bestGamma(a: number, rho: number, machine: Machine, maxGamma = 32) {
  const tDecode = decodeStep(machine);
  let best: BlockResult | null = null;
  for (let g = 1; g <= maxGamma; g++) {
    const r = block(a, rho, g, verifyPass(machine, g).total, tDecode);
    if (!best || r.speedup > best.speedup) best = r;
  }
  return best!;
}

/** Speedup for a whole (a, rho, gamma) configuration, in absolute seconds. */
export function specSpeedup(a: number, rho: number, gamma: number, machine: Machine) {
  const tDecode = decodeStep(machine);
  return block(a, rho, gamma, verifyPass(machine, gamma).total, tDecode);
}

/**
 * The machine every visual defaults to: Llama-3.1-8B in BF16, 2,048-token
 * average context. KV bytes/token = 2 x 32 layers x 8 KV heads x 128 x 2 = 131,072.
 */
export const LLAMA8B: Machine = {
  bw: 3.35e12,
  flops: 989e12,
  weights: 8.03e9 * 2,
  kvBytesPerToken: 2 * 32 * 8 * 128 * 2,
  ctx: 2048,
  params: 8.03e9,
};

/** The same model on a free Colab T4, and on an A100. */
export const T4_8B: Machine = { ...LLAMA8B, bw: 320e9, flops: 65e12 };
export const A100_8B: Machine = { ...LLAMA8B, bw: 2.0e12, flops: 312e12 };

/** The published draft-model family, as cost-ratio / acceptance pairs. */
export const METHODS: { id: string; label: string; blurb: string; rho: number; a: number }[] = [
  { id: 'lookup', label: 'Prompt lookup', blurb: 'no model at all', rho: 0.005, a: 0.30 },
  { id: 'ngram', label: 'N-gram', blurb: 'a table, not a network', rho: 0.01, a: 0.35 },
  { id: 'medusa', label: 'Medusa', blurb: 'extra heads on the target', rho: 0.10, a: 0.60 },
  { id: 'draft', label: 'Draft model', blurb: '1B drafting 8B', rho: 0.125, a: 0.72 },
  { id: 'eagle', label: 'EAGLE', blurb: 'predicts the next hidden state', rho: 0.15, a: 0.78 },
  { id: 'mtp', label: 'MTP', blurb: 'the next block, in parallel', rho: 0.20, a: 0.80 },
];

export const MACHINES: Record<string, Machine> = { t4: T4_8B, a100: A100_8B, h100: LLAMA8B };