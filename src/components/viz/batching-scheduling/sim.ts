// A tiny model of an LLM serving engine's scheduler, shared by every visual in this
// module and by the in-browser lab simulator. It is deliberately plain: the point is
// that the numbers on the page all come from one place, and that the rules are few.
//
// One `simulate()` call is one engine stepping through a workload. Each iteration:
//   1. free the slots of anything that finished, was cancelled or was preempted
//   2. admit waiting requests until the batch is full (or the policy says stop)
//   3. run one step: every running sequence produces exactly one token
//   4. recycle finished slots on the very next iteration  <- this is continuous batching

export interface ReqSpec {
  id: number;
  /** Engine step at which the request arrives. 0 = part of the initial wave. */
  arrive: number;
  /** Output tokens wanted. The request is "done" after this many decode steps. */
  want: number;
  /** Lower is served earlier. Only used by the priority policy. */
  priority?: number;
}

export type Policy = 'fifo' | 'priority';

export interface SimOptions {
  reqs: ReqSpec[];
  /** max_num_seqs: how many sequences may be in the running batch. */
  slots: number;
  policy?: Policy;
  /** request id -> engine step at which the client disconnects */
  cancel?: Record<number, number>;
  /** request id -> engine step at which the engine is asked to preempt it */
  preempt?: Record<number, number>;
  /** How many sequences the KV budget allows. Fewer than `slots` forces preemption. */
  kvSlots?: number;
  /** Stop after this many steps (safety valve; the default is far above any real workload). */
  maxSteps?: number;
}

export interface StepLog {
  running: number[];
  waiting: number[];
  /** slots that occupied the GPU this step without producing a wanted token */
  wasted: number;
}

export interface SimResult {
  /** engine steps until every request finished, was cancelled, or could never run */
  steps: number;
  useful: number;
  wastedSlots: number;
  slotsUsed: number;
  batchSize: number[];
  queueDepth: number[];
  log: StepLog[];
  /** first engine step each request was admitted; -1 if it never ran */
  admitted: number[];
  finished: Record<number, number>;
  cancelled: number[];
  preempted: number[];
  /** tokens re-generated because a preempted sequence had to rebuild its KV cache */
  recomputeTokens: number;
}

type Row = ReqSpec & {
  /** tokens already delivered to the client (survives preemption) */
  emitted: number;
  /** tokens currently in the KV cache; a preemption throws this away */
  ctx: number;
  state: 'queued' | 'running' | 'done' | 'gone';
};

const LIMIT = 4000;

/** Continuous batching (or plain FIFO / priority admission) over a list of requests. */
export function simulate(o: SimOptions): SimResult {
  const policy: Policy = o.policy ?? 'fifo';
  const slots = Math.max(1, Math.round(o.slots));
  const kvSlots = Math.max(1, Math.round(o.kvSlots ?? slots));
  const cancel = o.cancel ?? {};
  const preempt = o.preempt ?? {};
  const cap = o.maxSteps ?? LIMIT;

  const rows: Row[] = o.reqs.map((r) => ({ ...r, emitted: 0, ctx: 0, state: 'queued' }));
  const running: Row[] = [];
  const finished: Record<number, number> = {};
  const cancelled: number[] = [];
  const preempted: number[] = [];
  const admitted = rows.map(() => -1);
  const batchSize: number[] = [];
  const queueDepth: number[] = [];
  const log: StepLog[] = [];

  let steps = 0;
  let useful = 0;
  let wastedSlots = 0;
  let slotsUsed = 0;
  /** ids admitted with an empty KV cache this step: they must recompute their prefix */
  let recompute: number[] = [];
  let recomputeTokens = 0;

  const ready = () => rows.filter((r) => r.state === 'queued' && r.arrive <= steps);
  const rank = (a: Row, b: Row) =>
    policy === 'priority'
      ? (a.priority ?? 0) - (b.priority ?? 0) || a.arrive - b.arrive || a.id - b.id
      : a.arrive - b.arrive || a.id - b.id;
  const ordered = () => ready().sort(rank);

  for (let guard = 0; guard < cap; guard++) {
    // 1 & 2. recycle free slots, then admit
    const queue = ordered();
    while (running.length < slots && queue.length) {
      const r = queue.shift()!;
      r.state = 'running';
      // A preempted sequence comes back with an empty KV cache and pays to recompute
      // the prefix it had already emitted. That recompute is what preemption costs.
      r.ctx = 0;
      admitted[r.id] = steps;
      recompute.push(r.id);
      running.push(r);
    }
    if (!running.length) {
      if (!ready().length) break;
      steps += 1; // nothing runnable yet: burn an idle step, exactly like an engine
      continue;
    }

    // 3. run the step
    const ids = running.map((r) => r.id);
    let wasted = 0;
    let recomputeWork = 0;
    for (const r of running) {
      if (cancel[r.id] === steps) {
        // the client hung up: its KV blocks are freed immediately
        r.state = 'gone';
        r.ctx = 0;
        cancelled.push(r.id);
        wasted += 1;
        continue;
      }
      if (preempt[r.id] === steps) {
        // asked to yield its seat: the KV cache is dropped and it goes to the back
        r.state = 'queued';
        r.arrive = steps + 1;
        r.ctx = 0;
        admitted[r.id] = -1;
        preempted.push(r.id);
        wasted += 1;
        continue;
      }
      if (recompute.includes(r.id)) {
        // Rebuild the prefix this sequence already streamed to the client. This is
        // what preemption costs: paid work that produces no new token.
        if (r.ctx < r.emitted) {
          r.ctx += 1;
          recomputeWork += 1;
          continue;
        }
        recompute = recompute.filter((id) => id !== r.id);
      }
      r.ctx += 1;
      r.emitted += 1;
      useful += 1;
      if (r.emitted >= r.want) {
        r.state = 'done';
        finished[r.id] = steps;
      }
    }
    recompute = [];
    for (let i = running.length - 1; i >= 0; i -= 1) if (running[i].state !== 'running') running.splice(i, 1);

    // 4. KV pressure: the batch outgrew the memory budget, so the newest
    // (shortest-so-far, cheapest-to-rebuild) sequence yields its seat
    while (running.length > Math.min(slots, kvSlots)) {
      const victim = running.pop()!;
      victim.state = 'queued';
      victim.arrive = steps + 1;
      victim.ctx = 0;
      admitted[victim.id] = -1;
      preempted.push(victim.id);
      wasted += 1;
    }

    slotsUsed += ids.length;
    wastedSlots += wasted + recomputeWork;
    recomputeTokens += recomputeWork;
    batchSize.push(ids.length);
    queueDepth.push(ready().length);
    log.push({ running: ids, waiting: ready().map((r) => r.id), wasted: wasted + recomputeWork });
    steps += 1;
  }
  return { steps, useful, wastedSlots, slotsUsed, batchSize, queueDepth, log, admitted, finished, cancelled, preempted, recomputeTokens };
}

export interface StaticRow {
  id: number;
  /** engine steps at which the request was admitted to its batch */
  start: number;
  /** engine steps at which its batch finished (the slowest member decides) */
  end: number;
  want: number;
}

export interface StaticResult {
  steps: number;
  groups: ReqSpec[][];
  rows: StaticRow[];
  useful: number;
  slotsUsed: number;
  wastedSlots: number;
}

/**
 * Static batching: sort by arrival, cut into fixed groups of `slots`, and run each
 * group for as many steps as its slowest member needs. Short requests keep a slot
 * (and a row of the batch) for the rest of the group's life.
 */
export function simulateStatic(reqs: ReqSpec[], slots: number): StaticResult {
  const order = [...reqs].sort((a, b) => a.arrive - b.arrive || a.id - b.id);
  const groups: ReqSpec[][] = [];
  let cur: ReqSpec[] = [];
  for (const r of order) {
    cur.push(r);
    if (cur.length === Math.max(1, slots)) {
      groups.push(cur);
      cur = [];
    }
  }
  if (cur.length) groups.push(cur);

  const rows: StaticRow[] = [];
  let steps = 0;
  let slotsUsed = 0;
  for (const g of groups) {
    const span = g.reduce((m, r) => Math.max(m, r.want), 0);
    for (const r of g) rows.push({ id: r.id, start: steps, end: steps + span, want: r.want });
    slotsUsed += g.length * span;
    steps += span;
  }
  const useful = reqs.reduce((s, r) => s + r.want, 0);
  rows.sort((a, b) => a.id - b.id);
  return { steps, groups, rows, useful, slotsUsed, wastedSlots: slotsUsed - useful };
}

/** Percentiles by nearest-rank. Used for TTFT / TPOT readouts. */
export function pct(values: number[], p: number): number {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1));
  return s[i];
}

/** Llama-3.1-8B on one H100, in BF16 — the standing example for this module. */
export const KV_BYTES_PER_TOKEN_8B = 2 * 32 * 8 * 128 * 2; // 131,072 B = 128 KiB
export const WEIGHTS_BYTES_8B = 8.03e9 * 2; // 16.06 GB