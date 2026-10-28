// Module 24 · Observability, Reliability & Security — shared math and models.
//
// Everything here is either arithmetic the learner already owns (Modules 05, 06, 08, 09)
// or a deterministic model they can check by hand. No Math.random, no Date.now: every
// "measurement" is produced by a seeded generator so the server render and the browser
// render are byte-identical.

// ───────────────────────────── deterministic randomness

/** A tiny linear congruential generator. Same seed → same stream, on every machine. */
export function lcg(seed: number) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** A smooth-ish deterministic series for the dashboard sparklines. */
export function series(n: number, base: number, amp: number, driftPerStep = 0, seed = 7, cycles = 1.4): number[] {
  const r = lcg(seed);
  const out: number[] = [];
  let v = base;
  for (let i = 0; i < n; i++) {
    const t = i / Math.max(1, n - 1);
    v += Math.sin(t * Math.PI * 2 * cycles) * amp * 0.05 + (r() - 0.5) * amp * 0.07 + driftPerStep;
    v = Math.max(0, v);
    out.push(v);
  }
  return out;
}

// ───────────────────────────── latency histograms

/** A cumulative histogram over bucket edges, as a monitoring system stores it. */
export interface Histogram { edges: number[]; counts: number[] }

/** Build a cumulative histogram from raw samples: counts[i] = how many are <= edges[i]. */
export function histogram(samples: number[], edges: number[]): Histogram {
  const counts = new Array(edges.length).fill(0);
  for (const v of samples) {
    for (let i = 0; i < edges.length; i++) {
      if (v <= edges[i]) { counts[i]++; }
    }
  }
  return { edges, counts };
}

/**
 * Estimate a quantile from a cumulative histogram by linear interpolation inside
 * the bucket that contains it. This is exactly what a monitoring system does, and
 * why a quantile read off a histogram is approximate.
 */
export function quantileFromHistogram(h: Histogram, q: number, total?: number): number {
  const { edges, counts } = h;
  const n = total ?? counts[counts.length - 1] ?? 0;
  if (!n) return 0;
  const target = q * n;
  let prevCount = 0;
  let prevEdge = 0;
  for (let i = 0; i < counts.length; i++) {
    if (counts[i] >= target) {
      const bucket = counts[i] - prevCount || 1;
      const into = (target - prevCount) / bucket;
      return prevEdge + into * (edges[i] - prevEdge);
    }
    prevCount = counts[i];
    prevEdge = edges[i];
  }
  return edges[edges.length - 1];
}

/**
 * Cardinality: how many distinct time series one label value creates.
 * `fanout` is the number of label values that combine (model × tenant × replica × status).
 */
export const seriesCount = (perLabel: Record<string, number>) =>
  Object.values(perLabel).reduce((a, b) => a * b, 1);

// ───────────────────────────── the overload model

export interface MixEntry { frac: number; dur: number; name: string }

/** A realistic chat/completion mixture: mostly short, a long tail that ties up a slot. */
export const REQUEST_MIX: MixEntry[] = [
  { frac: 0.62, dur: 0.8, name: 'short chat' },
  { frac: 0.3, dur: 2.5, name: 'tool call' },
  { frac: 0.08, dur: 8, name: 'long summary' },
];

export const meanService = (mix = REQUEST_MIX) => mix.reduce((a, m) => a + m.frac * m.dur, 0);

export interface OverloadResult {
  offered: number;
  admitted: number;
  shed: number;
  completed: number;
  throughput: number;
  waits: number[];
  p50: number;
  p95: number;
  p99: number;
  occupancy: number; // mean fraction of the concurrency pool in use
}

const DT = 0.05; // simulation tick, seconds
const HORIZON = 200; // seconds of simulated traffic
const MAX_TICKS = Math.ceil(HORIZON / DT) + 400;

/**
 * A discrete-event queue simulation with a hard concurrency cap and a small waiting room.
 *
 *   - arrivals are Poisson at `rate` per second (drawn from a seeded stream)
 *   - a request occupies one concurrency slot for its duration while it is generating
 *   - a free slot admits immediately; a full slot queues in a `bufferSize` waiting room
 *   - a full waiting room sheds: instant, retryable failure (the load-shed path)
 *
 * Unbounded queues are this model with bufferSize → Infinity: wait time grows without limit.
 */
export function simulateOverload(p: {
  rate: number;
  pool: number;
  bufferSize: number;
  seed?: number;
  mix?: MixEntry[];
}): OverloadResult {
  const rate = Math.max(0, p.rate);
  const pool = Math.max(1, Math.floor(p.pool));
  const bufCap = p.bufferSize >= 1e6 ? 1e9 : Math.max(0, Math.floor(p.bufferSize));
  const mix = p.mix ?? REQUEST_MIX;
  const rnd = lcg(p.seed ?? 12345);

  const done = new Array(MAX_TICKS).fill(0); // completions per tick
  const waiting: { start: number; dur: number }[] = [];
  const waits: number[] = [];
  let busy = 0;
  let admitted = 0;
  let shed = 0;
  let completed = 0;
  let occSum = 0;
  const ticks = Math.ceil(HORIZON / DT);
  const lambdaDt = rate * DT;

  for (let t = 0; t < ticks; t++) {
    // retire whatever finished on this tick
    if (done[t]) { busy -= done[t]; completed += done[t]; }
    // drain the waiting room into free slots
    while (busy < pool && waiting.length > 0) {
      const j = waiting.shift()!;
      busy++;
      done[t + Math.max(1, Math.ceil(j.dur / DT))] += 1;
      waits.push(t * DT - j.start);
    }
    // Poisson arrivals in this tick (Knuth, for small lambda*dt)
    let n = 0;
    if (lambdaDt > 0) {
      const L = Math.exp(-lambdaDt);
      let k = 0;
      let p = 1;
      do { k++; p *= rnd(); } while (p > L);
      n = k - 1;
    }
    for (let i = 0; i < n; i++) {
      const u = rnd();
      let acc = 0;
      let m = mix[mix.length - 1];
      for (const e of mix) { acc += e.frac; if (u <= acc) { m = e; break; } }
      if (busy < pool) {
        busy++;
        admitted++;
        done[t + Math.max(1, Math.ceil(m.dur / DT))] += 1;
      } else if (waiting.length < bufCap) {
        waiting.push({ start: t * DT, dur: m.dur });
        admitted++;
      } else {
        shed++;
      }
    }
    occSum += busy / pool;
  }

  waits.sort((a, b) => a - b);
  const pct = (q: number) => (waits.length ? waits[Math.min(waits.length - 1, Math.floor(q * (waits.length - 1)))] : 0);
  const offered = admitted + shed;
  return {
    offered,
    admitted,
    shed,
    completed,
    throughput: completed / HORIZON,
    waits,
    p50: pct(0.5),
    p95: pct(0.95),
    p99: pct(0.99),
    occupancy: occSum / ticks,
  };
}

/** Goodput against a queueing-wait SLO: completed requests whose wait fitted the budget. */
export function goodputOf(r: OverloadResult, sloWait: number) {
  const ok = r.waits.filter((w) => w <= sloWait).length;
  return { good: ok / HORIZON, fraction: r.waits.length ? ok / r.waits.length : 0 };
}

export const HORIZON_S = HORIZON;

// ───────────────────────────── retry amplification

/**
 * Expected number of attempts for one logical request when each attempt fails
 * independently with probability p and the client retries up to `retries` times.
 */
export function attemptFactor(p: number, retries: number) {
  if (p <= 0) return 1;
  if (p >= 1) return retries + 1;
  let s = 0;
  let pk = 1;
  for (let i = 0; i <= retries; i++) { s += pk; pk *= p; }
  return s;
}

/**
 * The correlated-failure case, which is the one that actually melts a service:
 * the downstream is down, so every attempt fails with probability ~1.
 */
export const correlatedFactor = (retries: number) => retries + 1;

/** Amplification through a chain of `depth` retrying layers. */
export const layeredFactor = (retries: number, depth: number) => Math.pow(retries + 1, depth);

/**
 * Retry budget: the standard OpenAI-style rule that a client may spend at most
 * `ratio` × requests on retries, so amplification is bounded by construction.
 * Retries permitted = requests × (ratio − 1).
 */
export const retryBudget = (requests: number, ratio: number) => Math.max(0, Math.floor(requests * (ratio - 1)));

// ───────────────────────────── the timeout budget

/**
 * Split a total request budget across N sequential hops, leaving a safety margin.
 * Each hop's timeout must be strictly less than the remaining parent budget or the
 * parent can never fail cleanly before its own deadline.
 */
export function timeoutSplit(totalS: number, hops: number, margin = 0.8) {
  const per = (totalS * margin) / hops;
  return { per, total: per * hops, margin: totalS - per * hops };
}

// ───────────────────────────── tracing

export interface Span {
  name: string;
  service: string;
  start: number;  // ms from request arrival
  dur: number;    // ms
  kind: 'queue' | 'compute' | 'memory' | 'comm' | 'token' | 'idle';
  parent?: number;
  note?: string;
}

/** One request through a disaggregated stack: edge, router, prefill pool, KV transfer, decode pool. */
export const TRACE_SPANS: Span[] = [
  { name: 'edge: TLS + auth', service: 'edge', start: 0, dur: 26, kind: 'comm' },
  { name: 'rate limit + admission', service: 'edge', start: 26, dur: 4, kind: 'idle' },
  { name: 'tokenize', service: 'edge', start: 30, dur: 9, kind: 'memory', parent: 0 },
  { name: 'router: prefix lookup', service: 'router', start: 39, dur: 11, kind: 'memory' },
  { name: 'queue wait', service: 'router', start: 50, dur: 58, kind: 'idle', note: 'scheduler queue' },
  { name: 'kv transfer P→D', service: 'engine', start: 108, dur: 21, kind: 'comm', note: 'NCCL / RDMA' },
  { name: 'prefill 512 tok', service: 'prefill', start: 108, dur: 44, kind: 'compute', note: 'compute-bound' },
  { name: 'detokenize + first token', service: 'edge', start: 129, dur: 6, kind: 'token' },
  { name: 'decode step 1', service: 'decode', start: 135, dur: 11, kind: 'memory' },
  { name: 'decode steps 2..n', service: 'decode', start: 146, dur: 180, kind: 'memory', note: '180 tok × 1.0 ms' },
  { name: 'stream out', service: 'edge', start: 326, dur: 12, kind: 'token' },
];

export const CRITICAL_PATH: { start: number; dur: number; label: string }[] = [
  { start: 0, dur: 50, label: 'edge + router' },
  { start: 50, dur: 58, label: 'queue wait' },
  { start: 108, dur: 44, label: 'prefill' },
  { start: 152, dur: 174, label: 'decode' },
  { start: 326, dur: 12, label: 'egress' },
];

// ───────────────────────────── the request path, as a trust diagram

export interface Hop {
  id: string;
  name: string;
  attackerInput: string;
  threat: string;
  control: string;
  severity: 1 | 2 | 3; // 3 = loses money or data fastest
}

export const ATTACK_SURFACE: Hop[] = [
  {
    id: 'client', name: 'client → edge',
    attackerInput: 'prompt bytes, max_tokens, headers, request rate',
    threat: 'unauthenticated spend; a single client can drain a GPU',
    control: 'API key or mTLS at the edge, quota per key',
    severity: 3,
  },
  {
    id: 'edge', name: 'edge / gateway',
    attackerInput: 'chosen headers, claimed token counts, retry storms',
    threat: 'amplification: one client call becomes 64 backend calls',
    control: 'admission control, retry budget, header allow-list',
    severity: 3,
  },
  {
    id: 'router', name: 'router / cache',
    attackerInput: 'prompt prefix, tenant id, timing of requests',
    threat: 'prefix-cache poisoning across tenants; cache-key confusion',
    control: 'per-tenant cache namespace, cache key includes tenant',
    severity: 2,
  },
  {
    id: 'engine', name: 'engine / scheduler',
    attackerInput: 'concurrency, context length, cancellation timing',
    threat: 'resource exhaustion: one 128k request starves everyone',
    control: 'per-tenant concurrency cap, max_model_len, KV budget',
    severity: 3,
  },
  {
    id: 'kv', name: 'KV cache',
    attackerInput: 'nothing directly — but slots outlive requests',
    threat: 'a reused or mis-scoped slot leaks another tenant\'s tokens',
    control: 'allocate per request, free on completion, zero on reuse',
    severity: 3,
  },
  {
    id: 'model', name: 'model weights',
    attackerInput: 'many long, high-entropy outputs',
    threat: 'the weights themselves are the asset: sampling them is exfiltration',
    control: 'rate limit, cap tokens per request, watermark, keep weights off the wire',
    severity: 2,
  },
  {
    id: 'logs', name: 'logs & traces',
    attackerInput: 'nothing — but everything lands here',
    threat: 'prompts, completions and API keys written to disk in plaintext',
    control: 'redact by default, encrypt at rest, short retention',
    severity: 2,
  },
];