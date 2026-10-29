// Toy model of few-shot prompting for a 4-label ticket classifier.
// The accuracy curves are ILLUSTRATIVE: their shapes follow published findings
// (GPT-3's rising few-shot curve, a high zero-shot start after instruction tuning,
// format mattering more than label correctness, order sensitivity), not a measurement.
// The token and dollar arithmetic is exact.

export type ModelKind = 'base' | 'instruct';
export type Variant = 'clean' | 'format' | 'labels' | 'order';

export const CHANCE = 0.25; // four labels
const sat = (k: number, tau: number) => 1 - Math.exp(-k / tau);

/** Mean accuracy (illustrative) with k examples. */
export function accuracy(model: ModelKind, k: number, v: Variant = 'clean') {
  const clean = model === 'base' ? CHANCE + 0.55 * sat(k, 3) : 0.8 + 0.12 * sat(k, 1.5);
  const hurt = {
    clean: 0,
    format: model === 'base' ? 0.15 : 0.06, // examples disagree on how the answer looks
    labels: model === 'base' ? 0.05 : 0.02, // labels shuffled at random: barely matters
    order: model === 'base' ? 0.08 : 0.03, // same label on the last few examples
  }[v];
  return clean - hurt * sat(k, 2);
}

/** Half-width of the spread across example orderings (illustrative). */
export function spread(model: ModelKind, k: number, v: Variant) {
  if (k === 0) return 0;
  const wide = v === 'order';
  return (model === 'base' ? (wide ? 0.15 : 0.05) : wide ? 0.04 : 0.012) * Math.min(1, k / 2);
}

export const TOK = { instruction: 150, perExample: 200, query: 100, output: 5 };
export const tokensPerCall = (k: number) => TOK.instruction + k * TOK.perExample + TOK.query;
export const CALLS_PER_DAY = 100_000;

/** Dated example list prices for input tokens, USD per million. Prices change; these are history. */
export const PRICES = [
  { id: 'mini', usdPerM: 0.15, label: '$0.15 / M', note: 'GPT-4o mini launch price, Jul 2024' },
  { id: 'big', usdPerM: 2.5, label: '$2.50 / M', note: 'GPT-4o input price from Aug 2024' },
] as const;
export type PriceId = (typeof PRICES)[number]['id'];

/** Extra input tokens per day that k examples add, and what they cost. */
export function exampleCost(k: number, usdPerM: number, calls = CALLS_PER_DAY) {
  const extraPerCall = k * TOK.perExample;
  const perDay = extraPerCall * calls;
  const usdPerDay = (perDay / 1e6) * usdPerM;
  return { extraPerCall, perDay, usdPerDay, usdPerYear: usdPerDay * 365 };
}

/** Prefill time for n extra tokens on Llama-3.1-8B, H100 at 55% of 989 TFLOP/s. */
export const prefillMs = (n: number) => ((2 * 8.03e9 * n) / (0.55 * 989e12)) * 1e3;
