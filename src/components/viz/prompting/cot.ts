// Toy model of chain of thought, self-consistency and a verifier.
// ILLUSTRATIVE: a question set where each question has its own chance q that one sampled
// reasoning path ends in the right answer. Wrong answers split 50/30/20 across three distractors.
// Everything below is exact probability on that toy set; the latency comes from the perf model.
import { decodeStep, modelById, gpuById } from '../lib/perfModel';

/** Difficulty bins: per-path correct probability q and the share of questions at that q. */
export const BINS: { q: number; w: number }[] = [
  { q: 0.02, w: 0.08 }, { q: 0.1, w: 0.08 }, { q: 0.25, w: 0.1 }, { q: 0.4, w: 0.12 },
  { q: 0.55, w: 0.14 }, { q: 0.7, w: 0.16 }, { q: 0.85, w: 0.16 }, { q: 0.95, w: 0.16 },
];
const WRONG_SPLIT = [0.5, 0.3, 0.2];
export const DIRECT_ACC = 0.2; // answering immediately, no reasoning (illustrative)
export const VERIFIER_SKILL = 0.7; // chance the verifier recognises a correct path when one exists

const LF: number[] = [0];
for (let i = 1; i <= 64; i++) LF[i] = LF[i - 1] + Math.log(i);

/** P(majority vote is correct) for one question, N paths; ties split evenly. */
function voteCorrect(q: number, n: number) {
  const p = [q, ...WRONG_SPLIT.map((s) => s * (1 - q))];
  const lp = p.map((x) => Math.log(Math.max(x, 1e-300)));
  let total = 0;
  for (let k0 = 0; k0 <= n; k0++) {
    for (let k1 = 0; k1 <= n - k0; k1++) {
      for (let k2 = 0; k2 <= n - k0 - k1; k2++) {
        const k3 = n - k0 - k1 - k2;
        const mx = Math.max(k1, k2, k3);
        if (k0 < mx) continue;
        const ties = 1 + (k1 === k0 ? 1 : 0) + (k2 === k0 ? 1 : 0) + (k3 === k0 ? 1 : 0);
        const logP = LF[n] - LF[k0] - LF[k1] - LF[k2] - LF[k3] + k0 * lp[0] + k1 * lp[1] + k2 * lp[2] + k3 * lp[3];
        total += Math.exp(logP) / ties;
      }
    }
  }
  return total;
}

const voteCache = new Map<number, number>();
/** Dataset accuracy of self-consistency with N sampled paths. */
export function selfConsistency(n: number) {
  if (!voteCache.has(n)) voteCache.set(n, BINS.reduce((a, b) => a + b.w * voteCorrect(b.q, n), 0));
  return voteCache.get(n)!;
}
/** Fraction of questions where at least one of N paths is right (a perfect checker's ceiling). */
export const passAtN = (n: number) => BINS.reduce((a, b) => a + b.w * (1 - Math.pow(1 - b.q, n)), 0);
/** Best-of-N with an imperfect verifier: with probability v it finds a correct path if one exists, else it picks at random. */
export const verifierAcc = (n: number) => BINS.reduce((a, b) => a + b.w * (VERIFIER_SKILL * (1 - Math.pow(1 - b.q, n)) + (1 - VERIFIER_SKILL) * b.q), 0);
export const singleCoT = () => selfConsistency(1);

// ── the one question shown in the visual ──
export const QUESTION = 'A shop sells 3 pens for $2. How much do 12 pens cost?';
export const ANSWERS = [
  { a: '$8', p: 0.55, right: true },
  { a: '$6', p: 0.17 },
  { a: '$24', p: 0.13 },
  { a: '$16', p: 0.09 },
  { a: '$4', p: 0.06 },
];
const rand = (i: number) => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };
export interface Path { answer: string; right: boolean; tokens: number; score: number }
/** Deterministic sampled paths, the same on server and client. Path 0 is the worked example. */
export const PATHS: Path[] = Array.from({ length: 32 }, (_, i) => {
  const u = i === 0 ? 0.1 : rand(i + 7);
  let acc = 0;
  let pick = ANSWERS[ANSWERS.length - 1];
  for (const c of ANSWERS) { acc += c.p; if (u < acc) { pick = c; break; } }
  const right = !!pick.right;
  const tokens = Math.round(170 + rand(i + 101) * 150);
  // a verifier's score: usually high for right paths, sometimes fooled
  const score = i === 12 ? 0.97 : right ? 0.62 + rand(i + 211) * 0.33 : 0.08 + rand(i + 307) * 0.55; // path 12 fools the verifier
  return { answer: pick.a, right, tokens, score };
});
export function tally(n: number) {
  const m = new Map<string, number>();
  PATHS.slice(0, n).forEach((p) => m.set(p.answer, (m.get(p.answer) ?? 0) + 1));
  return ANSWERS.map((c) => ({ a: c.a, right: !!c.right, n: m.get(c.a) ?? 0 }));
}

// ── cost ──
export const COT_TOKENS = 250; // average reasoning + answer tokens per path
export const DIRECT_TOKENS = 5;
export const REASONING_TOKENS = 2000; // a reasoning model's typical trace on such a problem (illustrative)
const M = modelById('llama8b');
const G = gpuById('h100');
/** Seconds per decode step with `batch` paths decoded together (memory-bound: nearly flat in batch). */
export const stepS = (batch: number) => decodeStep({ model: M, gpu: G, batch, ctx: 400 }).t;
export const latencyParallel = (n: number, tok = COT_TOKENS) => tok * stepS(n);
export const latencySequential = (n: number, tok = COT_TOKENS) => n * tok * stepS(1);
