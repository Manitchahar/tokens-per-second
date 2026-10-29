// A prompt's life as versioned code: offline eval scores per version, the noise on those scores,
// a release gate, and the bill for automatic optimization. Scores are ILLUSTRATIVE;
// the confidence intervals and call counts are exact arithmetic.

export interface Version { id: string; change: string; score: number; tokens: number; live?: number }
export const VERSIONS: Version[] = [
  { id: 'v1', change: 'first draft', score: 0.71, tokens: 420 },
  { id: 'v2', change: 'explicit label list', score: 0.78, tokens: 480 },
  { id: 'v3', change: '+ one output example', score: 0.84, tokens: 610 },
  { id: 'v4', change: 'friendlier tone', score: 0.85, tokens: 640 },
  { id: 'v5', change: '+ refund rule', score: 0.87, tokens: 700 },
  { id: 'v6', change: 'shortened for cost', score: 0.8, tokens: 520 },
  { id: 'v7', change: 'moved to a new model', score: 0.86, tokens: 700, live: -0.06 },
];

/** Half-width of a 95% confidence interval for an accuracy p measured on n examples. */
export const ci95 = (p: number, n: number) => 1.96 * Math.sqrt((p * (1 - p)) / n);

/** Release gate: a candidate ships only if it is not worse than production by more than `margin`. */
export const passesGate = (cand: number, prod: number, margin = 0.01) => cand >= prod - margin;

// Automatic optimization (DSPy-style random search over bootstrapped demonstrations).
export const OPT = {
  bootstrap: 100, // training inputs run through a teacher program to collect demonstrations
  dev: 200, // dev examples each candidate is scored on
  callsPerRun: 3, // LM calls one run of the program makes (e.g. retrieve-query, reason, answer)
  inTok: 1500,
  outTok: 200,
};
export function compileCost(candidates: number) {
  const calls = OPT.bootstrap * OPT.callsPerRun + candidates * OPT.dev * OPT.callsPerRun;
  return { calls, inTok: calls * OPT.inTok, outTok: calls * OPT.outTok };
}
/** Dated example prices, USD per million tokens (input, output). */
export const PRICE_SETS = [
  { id: 'small', label: '$0.15 / $0.60', note: 'GPT-4o mini launch prices, Jul 2024', inP: 0.15, outP: 0.6 },
  { id: 'large', label: '$2.50 / $10', note: 'GPT-4o prices from Aug 2024', inP: 2.5, outP: 10 },
] as const;
export type PriceSetId = (typeof PRICE_SETS)[number]['id'];
export const dollars = (inTok: number, outTok: number, inP: number, outP: number) => (inTok / 1e6) * inP + (outTok / 1e6) * outP;

const rand = (i: number) => { const x = Math.sin(i * 91.7 + 13.1) * 43758.5453; return x - Math.floor(x); };
/** Candidate programs an optimizer tries: (prompt tokens, dev score). Deterministic. */
export const CANDIDATES = Array.from({ length: 32 }, (_, i) => {
  const demos = 1 + Math.floor(rand(i) * 6); // 1–6 bootstrapped demonstrations of ~250 tokens
  const tokens = 700 + demos * 250;
  const score = Math.min(0.93, 0.8 + demos * 0.012 + (rand(i + 50) - 0.5) * 0.08);
  return { tokens, score, demos };
});
export const bestOf = (k: number) => CANDIDATES.slice(0, k).reduce((b, c) => (c.score > b.score ? c : b), CANDIDATES[0]);
