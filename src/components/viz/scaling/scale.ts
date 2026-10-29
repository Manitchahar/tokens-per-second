// The arithmetic of Module 3B, in one place. Every constant here is a published number
// the lessons quote and derive; nothing is tuned to make a picture look nice.
//
//   training compute   C ≈ 6 N D                         (forward 2N + backward 4N per token)
//   GPU time           C / (peak FLOP/s × MFU)
//   Kaplan et al. 2020 L(N) = (Nc/N)^αN, L(D) = (Dc/D)^αD, L(C) = (Cc/C)^αC
//   Hoffmann et al. 2022 (Chinchilla) L(N, D) = E + A/N^α + B/D^β
import { GPUS, type Gpu } from '../lib/perfModel';

export const SECONDS_PER_HOUR = 3600;
/** One petaflop/s-day, the compute unit Kaplan et al. and the GPT-3 paper use. */
export const PF_DAY = 1e15 * 86400; // 8.64e19 FLOPs

/** Training FLOPs: 2N forward + 4N backward per token, times D tokens. */
export const trainFlops = (N: number, D: number) => 6 * N * D;
/** Serving FLOPs: a forward pass only, 2N per token. */
export const serveFlops = (N: number, tokens: number) => 2 * N * tokens;

/** GPUs a training run is usually priced on (perfModel spec numbers and illustrative rates). */
export const TRAIN_GPUS: Gpu[] = GPUS.filter((g) => ['a100', 'h100', 'h200', 'b200'].includes(g.id));
export const gpu = (id: string) => GPUS.find((g) => g.id === id)!;

export const gpuSeconds = (C: number, g: Gpu, mfu: number) => C / (g.flops * mfu);
export const gpuHours = (C: number, g: Gpu, mfu: number) => gpuSeconds(C, g, mfu) / SECONDS_PER_HOUR;
export const flopsFromHours = (hours: number, g: Gpu, mfu: number) => hours * SECONDS_PER_HOUR * g.flops * mfu;

/* ───────────── Kaplan et al. 2020, Section 1.2 (non-embedding parameters, nats per token) */
export const KAPLAN = {
  aN: 0.076, Nc: 8.8e13, // parameters
  aD: 0.095, Dc: 5.4e13, // tokens
  aC: 0.050, Cc: 3.1e8, // compute-efficient frontier, in PF-days
};
export const kaplanN = (N: number) => Math.pow(KAPLAN.Nc / N, KAPLAN.aN);
export const kaplanD = (D: number) => Math.pow(KAPLAN.Dc / D, KAPLAN.aD);
/** C in FLOPs. */
export const kaplanC = (C: number) => Math.pow(KAPLAN.Cc / (C / PF_DAY), KAPLAN.aC);

/* ───────────── Chinchilla parametric fit (Hoffmann et al. 2022, Approach 3) */
export interface Fit { id: string; name: string; E: number; A: number; B: number; alpha: number; beta: number }
/** The constants as printed in the paper. */
export const HOFFMANN: Fit = { id: 'hoffmann', name: 'Hoffmann et al. 2022', E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28 };
/** Besiroglu et al. 2024 refit the same experiments and got constants that agree with Approaches 1 and 2. */
export const EPOCH: Fit = { id: 'epoch', name: 'Besiroglu et al. 2024 refit', E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658 };
export const FITS = [HOFFMANN, EPOCH];

export const lossND = (f: Fit, N: number, D: number) => f.E + f.A / Math.pow(N, f.alpha) + f.B / Math.pow(D, f.beta);
/** Capacity term and data term separately (the two walls of the IsoFLOP valley). */
export const capTerm = (f: Fit, N: number) => f.A / Math.pow(N, f.alpha);
export const dataTerm = (f: Fit, D: number) => f.B / Math.pow(D, f.beta);
/** Loss on the IsoFLOP curve: fix C, choose N, D = C / 6N. */
export const lossAtC = (f: Fit, C: number, N: number) => lossND(f, N, C / (6 * N));

/** Exact minimiser of the fit on an IsoFLOP curve: N_opt = G (C/6)^(β/(α+β)). */
export function nOptFit(f: Fit, C: number) {
  const G = Math.pow((f.alpha * f.A) / (f.beta * f.B), 1 / (f.alpha + f.beta));
  return G * Math.pow(C / 6, f.beta / (f.alpha + f.beta));
}
/** The rule of thumb: D = 20 N and C = 6 N D, so C = 120 N² and N = √(C/120). */
export const nOptRule = (C: number) => Math.sqrt(C / 120);

/** Tokens needed for model N to reach loss L under a fit, or Infinity if N is too small to ever get there. */
export function tokensForLoss(f: Fit, N: number, L: number) {
  const room = L - f.E - capTerm(f, N);
  if (room <= 0) return Infinity;
  return Math.pow(f.B / room, 1 / f.beta);
}

/* ───────────── famous training runs: parameters and training tokens */
export interface Run { id: string; name: string; N: number; D: number; year: number }
export const RUNS: Run[] = [
  { id: 'gpt3', name: 'GPT-3 175B', N: 175e9, D: 300e9, year: 2020 },
  { id: 'gopher', name: 'Gopher 280B', N: 280e9, D: 300e9, year: 2021 },
  { id: 'chinchilla', name: 'Chinchilla 70B', N: 70e9, D: 1.4e12, year: 2022 },
  { id: 'llama1-7b', name: 'LLaMA-1 7B', N: 7e9, D: 1e12, year: 2023 },
  { id: 'llama3-8b', name: 'Llama-3.1-8B', N: 8e9, D: 15e12, year: 2024 },
  { id: 'llama3-405b', name: 'Llama-3.1-405B', N: 405e9, D: 15.6e12, year: 2024 },
];
export const run = (id: string) => RUNS.find((r) => r.id === id)!;

/* ───────────── GPT-3's eight sizes (Brown et al. 2020, Table 2.1) */
export const GPT3_SIZES = [
  { name: 'Small', N: 125e6, layers: 12, d: 768 },
  { name: 'Medium', N: 350e6, layers: 24, d: 1024 },
  { name: 'Large', N: 760e6, layers: 24, d: 1536 },
  { name: 'XL', N: 1.3e9, layers: 24, d: 2048 },
  { name: '2.7B', N: 2.7e9, layers: 32, d: 2560 },
  { name: '6.7B', N: 6.7e9, layers: 32, d: 4096 },
  { name: '13B', N: 13e9, layers: 40, d: 5140 },
  { name: '175B', N: 175e9, layers: 96, d: 12288 },
];

/* ───────────── formatting */
const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
/** 3.15e23 → "3.15 × 10²³" */
export function sci(n: number, digits = 2) {
  if (!isFinite(n) || n <= 0) return '—';
  const e = Math.floor(Math.log10(n));
  const m = n / Math.pow(10, e);
  const ms = m.toFixed(digits);
  // 9.999 rounds to 10.00: carry into the exponent
  if (Number(ms) >= 10) return sci(Math.pow(10, e + 1), digits);
  return `${ms} × 10${String(e).split('').map((c) => SUP[c]).join('')}`;
}
/** Parameters / tokens: 175e9 → "175B", 1.4e12 → "1.4T". */
export function count(n: number) {
  if (!isFinite(n)) return '∞';
  const u: [number, string][] = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
  for (const [v, s] of u) {
    if (n >= v) {
      const x = n / v;
      return `${x >= 100 ? x.toFixed(0) : x >= 10 ? String(parseFloat(x.toFixed(1))) : String(parseFloat(x.toFixed(2)))}${s}`;
    }
  }
  return n.toFixed(0);
}
/** GPU-hours: 221183 → "221k" */
export function hours(h: number) {
  if (!isFinite(h)) return '∞';
  if (h >= 1e6) return `${(h / 1e6).toFixed(h >= 1e7 ? 0 : 1)}M`;
  if (h >= 1e3) return `${(h / 1e3).toFixed(h >= 1e4 ? 0 : 1)}k`;
  return h.toFixed(0);
}
/** Dollars: 552958 → "$553k" */
export function usd(x: number) {
  if (!isFinite(x)) return '∞';
  if (x >= 1e9) return `$${(x / 1e9).toFixed(x >= 1e10 ? 0 : 1)}B`;
  if (x >= 1e6) return `$${(x / 1e6).toFixed(x >= 1e7 ? 0 : 1)}M`;
  if (x >= 1e3) return `$${(x / 1e3).toFixed(0)}k`;
  return `$${x.toFixed(0)}`;
}
/** Wall-clock days for a run spread over `n` GPUs. */
export const daysOn = (gpuHrs: number, n: number) => gpuHrs / n / 24;

/** Log-scale mapping helper: value → pixel. */
export const logScale = (lo: number, hi: number, p0: number, p1: number) => (v: number) =>
  p0 + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (p1 - p0);
