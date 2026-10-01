// Module 23 · Routing, Autoscaling & Orchestration — the shared fleet model.
//
// One place where every number the module uses is derived, so a lesson, a visual
// and the lab can never disagree. Everything is a pure function of its inputs and
// a seeded PRNG: the same config always produces the same run, which is what makes
// the visuals reproducible and hydration-safe (no Date.now, no Math.random).
//
// The physics comes from perfModel.ts: a decode step reads the weights once plus
// every running sequence's KV cache, and prefill costs FLOPs per prompt token.

import { GPUS, MODELS, decodeStep, prefillTime, type Gpu, type Model } from '../lib/perfModel';

export { GPUS, MODELS, decodeStep, prefillTime };
export type { Gpu, Model };

export const MODEL: Model = MODELS.find((m) => m.id === 'llama8b')!;
export const GPU: Gpu = GPUS.find((g) => g.id === 'h100')!;

/** KV bytes per token for our model in BF16 — the "128 KiB" of Module 05. */
export const KV_PER_TOKEN = 2 * MODEL.layers * MODEL.kvHeads * MODEL.headDim * 2;

/** Usable KV tokens on one replica: the pool minus weights, CUDA context and workspace. */
export function kvTokensFor(gpu: Gpu = GPU, model: Model = MODEL, util = 0.9): number {
  const pool = gpu.mem * util;
  const weights = model.params * 2;
  const reserve = 0.04 * gpu.mem;
  return Math.max(1, Math.floor((pool - weights - reserve) / KV_PER_TOKEN));
}

/* ───────────────────────── the deployment ───────────────────────── */

export type Policy =
  | 'single'
  | 'round-robin'
  | 'least-count'
  | 'least-tokens'
  | 'p2c'
  | 'session'
  | 'prefix';

export const POLICIES: { value: Policy; label: string; blurb: string }[] = [
  { value: 'single', label: 'One replica', blurb: 'No routing at all — everything to replica 0.' },
  { value: 'round-robin', label: 'Round robin', blurb: 'Next in line. Ignores every bit of state.' },
  { value: 'least-count', label: 'Least queued', blurb: 'Fewest queued requests: the classic least-connections balancer.' },
  { value: 'least-tokens', label: 'Least KV tokens', blurb: 'Fewest KV tokens held — the LLM-aware version of least-queued.' },
  { value: 'p2c', label: 'Power of two', blurb: 'Sample two replicas, send the request to the better one.' },
  { value: 'session', label: 'Session affinity', blurb: 'Pin a conversation to one replica so its prefix stays warm.' },
  { value: 'prefix', label: 'Cache-aware', blurb: 'Send to whichever replica already holds the longest prefix.' },
];

export interface Config {
  replicas: number;
  rate: number;
  longShare: number;
  shortPrompt: number;
  longPrompt: number;
  outTokens: number;
  policy: Policy;
  locality: number;
  kvPerReplica: number;
  /**
   * For `prefix`: how hard cache-aware routing trades a warm prefix against an even
   * load. The score is `hit_tokens - prefixWeight × kv_held`, so at 0 the router is
   * pure affinity (it piles every conversation onto whichever replica owns it) and as
   * the weight grows it behaves more and more like least-loaded. This dial is the
   * whole trade-off in Lesson 23.2.
   */
  prefixWeight: number;
}

export const DEFAULT: Config = {
  replicas: 4,
  rate: 14,
  longShare: 0.3,
  shortPrompt: 900,
  longPrompt: 8000,
  outTokens: 250,
  policy: 'least-tokens',
  locality: 0.6,
  kvPerReplica: 0,
  prefixWeight: 8,
};

/* ───────────────────────── SLO ───────────────────────── */

/** The interactive SLO this module holds. Same one Module 10's lab uses. */
export const SLO = { ttft: 1.0, tpot: 0.05 };

export function percentile(sorted: number[], q: number): number {
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
  return sorted[i];
}

/* ───────────────────────── deterministic randomness ───────────────────────── */

/** mulberry32: small, fast, and identical on the server and in the browser. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ───────────────────────── per-request record ───────────────────────── */

export interface Request {
  id: number;
  conv: number;
  prompt: number;
  out: number;
  /** Prompt tokens the chosen replica already had cached (0 unless cache-aware). */
  reused: number;
  /** Where it was routed. */
  replica: number;
  /** Seconds spent queued before it joined the running batch. */
  wait: number;
  ttft: number;
  tpot: number;
  ok: boolean;
}

export interface Result {
  reqs: Request[];
  p50ttft: number;
  p99ttft: number;
  meanTpot: number;
  /** Output tokens per second across the whole fleet. */
  tput: number;
  /** Fraction of completed requests that met both SLOs. */
  goodput: number;
  /** Fraction of prompt tokens served from an already-warm prefix. */
  hitRate: number;
  /** Peak KV load of each replica, in tokens — the profile an autoscaler must provision for. */
  kvPeaks: number[];
  /** Peak queue depth of each replica. */
  qPeaks: number[];
  /** KV-load spread: busiest replica ÷ mean replica. 1.0 is perfect. */
  imbalance: number;
  /** Peak KV usage on the busiest replica, as a fraction of its pool. */
  kvPeak: number;
  /** Mean queue wait across all requests. */
  meanWait: number;
  /** Wall-clock length of the run, seconds. */
  horizon: number;
  /** Engine steps executed across the fleet. */
  steps: number;
  /** Fraction of wall-clock time the fleet's GPUs were actually working. */
  util: number;
}

/* ───────────────────────── the simulation ───────────────────────── */

interface Waiting {
  req: Request;
  arrT: number;
}

interface Live {
  req: Request;
  arrT: number;
  /** Prompt tokens still to prefill. */
  preLeft: number;
  /** Output tokens still to generate. */
  outLeft: number;
  /** Sum of step times spent generating, and how many tokens that bought. */
  genTime: number;
  genTokens: number;
  ttft: number;
}

interface Arrival {
  t: number;
  conv: number;
  long: boolean;
}

/**
 * A discrete-event fleet simulation. Each replica runs the engine loop from Module 07:
 * every step it emits one decode token for each running request, plus one chunk of
 * prefill for each request still filling its prompt. Prefill is chunked (Module 10) so
 * a long prompt cannot monopolise a GPU.
 *
 * Deliberately small but not a toy: routing, KV pressure, prefix reuse and queueing all
 * interact, which is exactly what this module is about.
 */
export function simulate(
  cfg: Config,
  opts: { seed?: number; chunk?: number; maxTime?: number } = {},
): Result {
  const rand = rng(opts.seed ?? 12345);
  const chunk = opts.chunk ?? 512;
  const maxTime = opts.maxTime ?? 30;
  const kvCap = cfg.kvPerReplica > 0 ? cfg.kvPerReplica : kvTokensFor();

  const n = Math.max(1, Math.min(24, Math.floor(cfg.replicas)));
  /** Per replica: the queue, the running batch, and the KV tokens each holds. */
  const queues: Waiting[][] = Array.from({ length: n }, () => []);
  const live: Live[][] = Array.from({ length: n }, () => []);
  const kv: number[] = new Array(n).fill(0);
  /** Per replica: conversation id -> prompt tokens whose KV it holds. */
  const prefix: Map<number, number>[] = Array.from({ length: n }, () => new Map());

  /* arrivals: Poisson process, with `locality` deciding whether a request
     continues the previous conversation (a multi-turn user or an agent loop) */
  const arrivals: Arrival[] = [];
  let t = 0;
  let conv = 0;
  let lastConv = 0;
  while (t < maxTime) {
    if (lastConv > 0 && rand() < cfg.locality) arrivals.push({ t, conv: lastConv, long: rand() < cfg.longShare });
    else { conv += 1; lastConv = conv; arrivals.push({ t, conv, long: rand() < cfg.longShare }); }
    t += -Math.log(Math.max(1e-9, rand())) / Math.max(0.05, cfg.rate);
  }

  let rr = 0;
  let time = 0;
  let next = 0;
  let steps = 0;
  let busy = 0;
  /** Peak KV tokens each replica ever held — the number autoscaling must provision for. */
  const kvPeak = new Array(n).fill(0);
  /** Peak queue depth each replica ever saw. */
  const qPeak = new Array(n).fill(0);

  const pick = (c: number): number => {
    switch (cfg.policy) {
      case 'single': return 0;
      case 'round-robin': return rr++ % n;
      case 'least-count':
        return argMin(n, (i) => queues[i].length + live[i].length);
      case 'least-tokens':
        return argMin(n, (i) => kv[i]);
      case 'p2c': {
        const a = Math.floor(rand() * n);
        let b = Math.floor(rand() * n);
        if (b === a) b = (b + 1) % n;
        return queues[a].length + live[a].length <= queues[b].length + live[b].length ? a : b;
      }
      case 'session': {
        for (let i = 0; i < n; i++) if (prefix[i].has(c)) return i;
        return rr++ % n;
      }
      case 'prefix': {
        // The production trade-off: reward a warm prefix, penalise a loaded replica.
        // Both terms are normalised to a 0–1 scale, because they are otherwise in
        // incomparable units (hit is tokens, load is a fraction of the KV pool):
        //
        //     score = hit_frac  -  alpha × load_frac
        //
        // alpha = 0 is pure affinity (pile every conversation onto its owner); a large
        // alpha ignores the cache and becomes least-loaded. This is the shape production
        // cache-aware routers use — SGLang's router and the vLLM production stack both
        // score replicas on a combination of prefix affinity and running load.
        let best = 0;
        let bestScore = -Infinity;
        for (let i = 0; i < n; i++) {
          const hit = (prefix[i].get(c) ?? 0) / Math.max(1, cfg.longPrompt);
          const load = kv[i] / kvCap;
          const score = hit - cfg.prefixWeight * load;
          if (score > bestScore) { bestScore = score; best = i; }
        }
        return best;
      }
    }
  };

  const done: Request[] = [];

  while (next < arrivals.length || live.some((l) => l.length) || queues.some((q) => q.length)) {
    // ── admit arrivals that have already happened ──
    while (next < arrivals.length && arrivals[next].t <= time) {
      const a = arrivals[next++];
      const prompt = a.long ? cfg.longPrompt : cfg.shortPrompt;
      const r = pick(a.conv);
      const warm = (cfg.policy === 'prefix' || cfg.policy === 'session') ? prefix[r].get(a.conv) ?? 0 : 0;
      const req: Request = {
        id: done.length + next, conv: a.conv, prompt, out: cfg.outTokens,
        reused: Math.min(warm, prompt), replica: r, wait: 0, ttft: 0, tpot: 0, ok: false,
      };
      queues[r].push({ req, arrT: a.t });
    }

    // ── promote queued requests into the running batch where KV allows ──
    for (let i = 0; i < n; i++) {
      while (queues[i].length) {
        const w = queues[i][0];
        const cost = w.req.prompt + w.req.out;
        if (kv[i] + cost > kvCap && live[i].length > 0) break;   // no room and busy: wait
        if (kv[i] + cost > kvCap && live[i].length === 0) {
          // Idle but over budget: the request cannot fit at all. Evict the oldest
          // finished contribution so it can start — Module 08's preemption, crudely.
          kv[i] = 0;
          prefix[i].clear();
        }
        queues[i].shift();
        const waited = Math.max(0, time - w.arrT);
        w.req.wait = waited;
        kv[i] += cost;
        prefix[i].set(w.req.conv, Math.max(prefix[i].get(w.req.conv) ?? 0, w.req.prompt));
        live[i].push({
          req: w.req, arrT: w.arrT, preLeft: Math.max(0, w.req.prompt - w.req.reused),
          outLeft: w.req.out, genTime: 0, genTokens: 0, ttft: 0,
        });
      }
    }

    // ── one engine step, per replica, in parallel; wall-clock advances by the slowest ──
    let stepTime = 0;
    for (let i = 0; i < n; i++) {
      const batch = live[i];
      if (!batch.length) continue;
      const decoders = batch.filter((f) => f.preLeft <= 0);
      const prefills = batch.filter((f) => f.preLeft > 0);
      const ctx = Math.max(
        32,
        Math.floor(batch.reduce((s, f) => s + (f.req.prompt - f.preLeft) + (f.req.out - f.outLeft), 0) / batch.length),
      );
      const dec = decodeStep({ model: MODEL, gpu: GPU, batch: Math.max(1, decoders.length), ctx, mbu: 0.8 });
      let pre = 0;
      for (const f of prefills) pre += prefillTime({ model: MODEL, gpu: GPU, tokens: Math.min(chunk, f.preLeft) });
      const dt = Math.max(dec.t, pre);
      stepTime = Math.max(stepTime, dt);
      busy += dt;

      for (const f of batch) {
        if (f.preLeft > 0) {
          f.preLeft -= Math.min(chunk, f.preLeft);
          if (f.preLeft <= 0) f.ttft = time - f.arrT;   // first token lands this step
          continue;
        }
        if (f.outLeft > 0) {
          f.outLeft -= 1;
          f.genTime += dt;
          f.genTokens += 1;
          if (f.outLeft <= 0) {
            f.req.ttft = f.ttft;
            f.req.tpot = f.genTokens > 0 ? f.genTime / f.genTokens : 0;
            f.req.ok = f.req.ttft <= SLO.ttft && f.req.tpot <= SLO.tpot;
            done.push(f.req);
            kv[i] -= f.req.prompt + f.req.out;
          }
        }
      }
      // drop the finished, keep the rest
      live[i] = batch.filter((f) => f.outLeft > 0 || f.preLeft > 0);
    }

    // Sample peaks after every step: end-of-run load is nearly zero and hides everything.
    for (let i = 0; i < n; i++) {
      if (kv[i] > kvPeak[i]) kvPeak[i] = kv[i];
      if (queues[i].length > qPeak[i]) qPeak[i] = queues[i].length;
    }

    if (stepTime <= 0) break;
    steps += 1;
    time += stepTime;
    if (time > maxTime * 3 + 5) break;
  }

  /* ── aggregate ── */
  const sorted = done.map((r) => r.ttft).sort((a, b) => a - b);
  const meanPeak = kvPeak.reduce((s, v) => s + v, 0) / n;
  const totalOut = done.reduce((s, r) => s + r.out, 0);
  const totalPrompt = done.reduce((s, r) => s + r.prompt, 0);
  const totalReused = done.reduce((s, r) => s + r.reused, 0);

  return {
    reqs: done,
    p50ttft: percentile(sorted, 0.5),
    p99ttft: percentile(sorted, 0.99),
    meanTpot: done.length ? done.reduce((s, r) => s + r.tpot, 0) / done.length : 0,
    tput: time > 0 ? totalOut / time : 0,
    goodput: done.length ? done.filter((r) => r.ok).length / done.length : 0,
    hitRate: totalPrompt > 0 ? totalReused / totalPrompt : 0,
    kvPeaks: kvPeak,
    qPeaks: qPeak,
    imbalance: meanPeak > 0 ? Math.max(...kvPeak) / meanPeak : 1,
    kvPeak: kvCap > 0 ? Math.max(...kvPeak) / kvCap : 0,
    meanWait: done.length ? done.reduce((s, r) => s + r.wait, 0) / done.length : 0,
    horizon: time,
    steps,
    util: time > 0 ? Math.min(1, busy / (time * n)) : 0,
  };
}

function argMin(n: number, score: (i: number) => number): number {
  let best = 0;
  let bv = score(0);
  for (let i = 1; i < n; i++) {
    const v = score(i);
    if (v < bv) { bv = v; best = i; }
  }
  return best;
}

/* ───────────────────────── scaling helpers ───────────────────────── */

/**
 * Peak sustainable output tokens per second per replica, from perfModel.
 * Decode is memory-bound, so this is the fleet's capacity ceiling: below the point
 * where the batch starts reading a comparable amount of KV to weights, throughput
 * rises almost linearly with the batch.
 */
export function capacityPerReplica(gpu: Gpu = GPU, model: Model = MODEL, ctx = 1024, maxBatch = 64): number {
  let best = 0;
  for (let b = 1; b <= maxBatch; b++) {
    const r = decodeStep({ model, gpu, batch: b, ctx, mbu: 0.8 });
    if (r.total > best) best = r.total;
  }
  return best;
}

/**
 * How many replicas do I need for this offered rate?
 *
 * Two limits bind at once, and taking the larger is the whole job:
 *
 *   **Latency.** One replica holds `usersPerReplica(ctx)` sequences before its KV
 *   pool is full. To give each request a TPOT of `tpot` seconds/token, the decode
 *   step must be about `tpot` long — that fixes the batch. A request occupies a
 *   slot for its whole life, `prompt + out` tokens of generation apart, so one
 *   replica retires `out / tpot` requests per second at steady state.
 *
 *   **Throughput.** The batch cannot exceed what the memory bus allows, which
 *   `capacityPerReplica` gives directly in output tokens per second.
 *
 * A request is not memory: it spends TTFT waiting and prefilling too. So the safe
 * form divides by the *time in the system*, not just the generation time.
 */
export function replicasForRate(
  cfg: Config,
  opts: { gpu?: Gpu; model?: Model; ctx?: number; util?: number } = {},
): { perReplica: number; needed: number } {
  const gpu = opts.gpu ?? GPU;
  const model = opts.model ?? MODEL;
  const ctx = opts.ctx ?? Math.round(cfg.shortPrompt + cfg.outTokens);
  const users = usersPerReplica(ctx, gpu, model, opts.util ?? 0.9);

  // Latency limit: each request holds a KV slot for its whole life in the system.
  const timeInSystem = SLO.tpot * cfg.outTokens + SLO.ttft; // generation + queue/prefill budget
  const perReplica = users / timeInSystem;

  return { perReplica, needed: Math.max(1, Math.ceil(cfg.rate / perReplica)) };
}

/** Users a single replica can hold concurrently at a given context length. */
export function usersPerReplica(ctx: number, gpu: Gpu = GPU, model: Model = MODEL, util = 0.9): number {
  return Math.max(1, Math.floor(kvTokensFor(gpu, model, util) / Math.max(1, ctx)));
}