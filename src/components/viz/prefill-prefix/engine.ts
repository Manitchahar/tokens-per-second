/**
 * Module 10 lab · a mixed-workload engine simulator.
 *
 * A deliberately small model of the loop from Lessons 10.1 and 10.2: requests
 * arrive, each one prefills (optionally chunked, optionally from a prefix cache)
 * and then decodes one token per engine step. Everything the simulator decides
 * comes from the same two formulas the lessons derive — prefill time is
 * proportional to tokens, decode time is the memory floor over weights and KV.
 *
 * Arrivals are deterministic (no RNG) so a run is reproducible: the same knobs
 * always produce the same table.
 */
import { modelById, gpuById } from '../lib/perfModel';

export const model = modelById('llama8b');
export const gpu = gpuById('h100');

const MFU = 0.55; // achieved fraction of peak FLOP/s in prefill
const MBU = 0.8; // achieved fraction of peak bandwidth
const OVERHEAD = 0.0015; // scheduler + launch cost of one engine step, seconds

export const WEIGHT_BYTES = model.params * 2;
export const KV_PER_TOKEN = 2 * model.layers * model.kvHeads * model.headDim * 2;

export const prefillTime = (tokens: number) =>
  Math.max((2 * model.params * tokens) / (gpu.flops * MFU), WEIGHT_BYTES / (gpu.bw * MBU));

export const decodeTime = (batch: number, ctx: number) =>
  Math.max((WEIGHT_BYTES + batch * ctx * KV_PER_TOKEN) / (gpu.bw * MBU), (2 * model.params * batch) / (gpu.flops * MFU)) +
  OVERHEAD;

export interface Req {
  id: number;
  arrive: number;
  prefix: number; // tokens shared with earlier requests
  unique: number; // tokens nobody else has
  out: number; // tokens to generate
  ttft: number;
  e2e: number;
  tpot: number;
  ok: boolean;
}

export interface Config {
  /** arrival rate, requests per second */
  rate: number;
  /** fraction of requests that are long (RAG-style) prompts */
  longShare: number;
  /** chunked prefill on/off */
  chunked: boolean;
  /** chunk size in tokens */
  chunk: number;
  /** prefix caching on/off */
  cache: boolean;
  /** how many requests to simulate */
  count: number;
}

export const DEFAULT: Config = { rate: 3, longShare: 0.35, chunked: true, chunk: 512, cache: true, count: 48 };

const shortPrompt = [180, 260, 340, 420];
const longPrompt = [3000, 4500, 6000, 8000, 11000];
const outs = [120, 180, 240, 320];

/** Build a deterministic request stream. */
export function makeRequests(c: Config): Req[] {
  const out: Req[] = [];
  let t = 0.35;
  for (let i = 0; i < c.count; i++) {
    // deterministic pattern that mixes short and long without a random source
    const isLong = ((i * 7) % 20) / 20 < c.longShare;
    const k = i % (isLong ? longPrompt.length : shortPrompt.length);
    const total = isLong ? longPrompt[k] : shortPrompt[k];
    const prefix = isLong ? 900 : 260; // a system prompt every request carries
    out.push({
      id: i,
      arrive: t,
      prefix: c.cache ? prefix : 0,
      unique: total - (c.cache ? prefix : 0),
      out: outs[i % outs.length],
      ttft: NaN,
      e2e: NaN,
      tpot: NaN,
      ok: false,
    });
    t += 1 / c.rate;
  }
  return out;
}

export interface Result {
  reqs: Req[];
  p50ttft: number;
  p99ttft: number;
  meanTpot: number;
  tput: number; // output tokens per second, all users
  goodput: number; // fraction meeting the SLO
  horizon: number;
  steps: number;
  maxQueue: number;
}

const pct = (xs: number[], p: number) => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.floor((p / 100) * s.length));
  return s[i];
};

/** SLO used for goodput, matching a typical interactive chat target. */
export const SLO = { ttft: 1.0, tpot: 0.05 };

/**
 * Run the engine loop.
 *
 * One engine step carries at most one chunk of prefill and exactly one decode
 * token per active sequence. The step lasts as long as the slower of the two,
 * which is precisely the fact Lesson 10.1 is about.
 */
export function simulate(c: Config): Result {
  const reqs = makeRequests(c);
  const remaining = new Map<number, number>(reqs.map((r) => [r.id, r.unique]));
  const context = new Map<number, number>(reqs.map((r) => [r.id, r.prefix + r.unique]));
  const generated = new Map<number, number>(reqs.map((r) => [r.id, 0]));

  let t = 0;
  let steps = 0;
  let maxQueue = 0;
  const finished: Req[] = [];

  while (finished.length < reqs.length && t < 120 && steps < 200000) {
    steps++;
    const finishedIds = new Set(finished.map((r) => r.id));
    const arrived = reqs.filter((r) => r.arrive <= t && !finishedIds.has(r.id) && generated.get(r.id)! < r.out);
    if (!arrived.length) {
      const next = reqs.filter((r) => r.arrive > t).map((r) => r.arrive);
      if (!next.length) break; // everything has been served
      t = Math.min(...next);
      continue;
    }

    const prefilling = arrived.filter((r) => remaining.get(r.id)! > 0);
    const decoders = arrived.filter((r) => remaining.get(r.id)! === 0);
    maxQueue = Math.max(maxQueue, arrived.length);

    // prefill work for this step
    let preTok = 0;
    let target = -1;
    if (prefilling.length) {
      target = prefilling[0].id;
      preTok = c.chunked ? Math.min(c.chunk, remaining.get(target)!) : remaining.get(target)!;
    }
    const tPrefill = preTok > 0 ? prefillTime(preTok) : 0;

    // decode work for this step: one token per active sequence
    const avgCtx = decoders.length ? decoders.reduce((s, r) => s + context.get(r.id)!, 0) / decoders.length : 0;
    const tDecode = decoders.length ? decodeTime(decoders.length, avgCtx) : 0;

    const step = Math.max(tPrefill, tDecode) + OVERHEAD;

    // apply the step
    for (const r of decoders) {
      const g = generated.get(r.id)! + 1;
      generated.set(r.id, g);
      if (g === 1) r.ttft = t + step;
      context.set(r.id, context.get(r.id)! + 1);
      if (g >= r.out) {
        r.e2e = t + step;
        r.tpot = (r.e2e - r.ttft) / (r.out - 1);
        r.ok = r.ttft <= SLO.ttft && r.tpot <= SLO.tpot;
        finished.push(r);
      }
    }
    if (target >= 0) {
      remaining.set(target, remaining.get(target)! - preTok);
      // last chunk done: this sequence can decode from the next step on
    }

    t += step;
  }

  const done = finished.filter((r) => Number.isFinite(r.ttft));
  const horizon = Math.max(...done.map((r) => r.e2e), 0.01);
  return {
    reqs: done,
    p50ttft: pct(done.map((r) => r.ttft), 50),
    p99ttft: pct(done.map((r) => r.ttft), 99),
    meanTpot: done.reduce((s, r) => s + r.tpot, 0) / (done.length || 1),
    tput: done.reduce((s, r) => s + r.out, 0) / horizon,
    goodput: done.filter((r) => r.ok).length / (done.length || 1),
    horizon,
    steps,
    maxQueue,
  };
}