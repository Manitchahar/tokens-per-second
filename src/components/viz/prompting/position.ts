// Toy model of "lost in the middle": accuracy of answering from k retrieved documents as a function of
// where the one relevant document sits. ILLUSTRATIVE: the shape follows Liu et al. (2023)
// (U-shaped, primacy stronger than recency, the middle can fall below closed-book), not their data.

export type K = 10 | 20 | 30;
const PARAMS: Record<K, { floor: number; primacy: number; recency: number }> = {
  10: { floor: 0.6, primacy: 0.17, recency: 0.08 },
  20: { floor: 0.52, primacy: 0.24, recency: 0.11 },
  30: { floor: 0.48, primacy: 0.27, recency: 0.12 },
};
export const CLOSED_BOOK = 0.56; // answering with no documents at all
export const ORACLE = 0.88; // given only the relevant document
export const TOKENS_PER_DOC = 130;
export const QUESTION_TOKENS = 40;
export const INSTRUCTION_TOKENS = 60;

/** Accuracy (illustrative) with the relevant document at 1-based position `pos` of `k`. */
export function acc(pos: number, k: K) {
  const p = PARAMS[k];
  const x = k === 1 ? 0 : (pos - 1) / (k - 1);
  return p.floor + p.primacy * Math.exp(-x / 0.15) + p.recency * Math.exp(-(1 - x) / 0.12);
}
export const promptTokens = (k: K) => INSTRUCTION_TOKENS + k * TOKENS_PER_DOC + QUESTION_TOKENS;

/** Where the retriever ranks the relevant document (rank 1 = top): an illustrative distribution. */
export const RANK_P = (rank: number, k: K) => {
  const head = [0.5, 0.2, 0.1];
  if (rank <= 3) return head[rank - 1];
  return 0.2 / (k - 3);
};
/** Slot (1-based) a rank lands in under each ordering. */
export const ORDERINGS = {
  random: { label: 'random order', slot: (_r: number, _k: K) => 0 },
  ranked: { label: 'most relevant first', slot: (r: number) => r },
  sandwich: { label: 'best at both ends', slot: (r: number, k: K) => (r % 2 === 1 ? (r + 1) / 2 : k - r / 2 + 1) },
} as const;
export type OrderId = keyof typeof ORDERINGS;
export function expectedAcc(order: OrderId, k: K) {
  if (order === 'random') {
    let s = 0;
    for (let pos = 1; pos <= k; pos++) s += acc(pos, k);
    return s / k;
  }
  let s = 0;
  for (let r = 1; r <= k; r++) s += RANK_P(r, k) * acc(ORDERINGS[order].slot(r, k), k);
  return s;
}

// Needle-in-a-haystack toy heatmap: recall by context length and needle depth.
export const NIAH_LENGTHS = ['4K', '8K', '16K', '32K', '64K', '128K'];
export const NIAH_DEPTHS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
export function niah(li: number, depthPct: number) {
  const strain = Math.max(0, li - 3) / 2.2; // trouble starts past 32K in this toy
  const mid = Math.exp(-Math.pow((depthPct - 40) / 22, 2));
  return Math.max(0, Math.min(1, 1 - strain * 0.75 * mid));
}
