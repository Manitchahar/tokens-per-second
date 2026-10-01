import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Ruler, Fade, Dim, fmt } from '../lib/svg';
import { modelById, gpuById, decodeStep } from '../lib/perfModel';

/**
 * Continuous batching as a timeline. Static and continuous share one canvas so the
 * idle tail of a fixed batch is something you look at, not something you are told.
 */

const model = modelById('llama8b');
const gpu = gpuById('h100');

const STEPS = [
  'Static batching: four at a time',
  'The idle tail',
  'Continuous batching: slots, not groups',
  'Re-formed at every step',
  'The token budget is a lever',
  'Chunked prefill',
  'What the queue is telling you',
];
const CAPTIONS = [
  'Static batching takes four requests and runs them to completion together. The GPU is busy — and every user waits for the slowest one in the group.',
  'A short answer finishes early and then sits idle for the rest of the group, still holding its KV blocks, while three long answers continue. Hatched means a slot that is occupied and producing nothing.',
  'Continuous batching makes the batch a set of slots rather than a group. Whoever finishes leaves at their next step, and a waiting request takes the slot immediately.',
  'The same workload, but every step boundary is a fresh decision. Short requests are cheap because they leave early, and nothing is ever held back by a slow neighbour.',
  'Cap the tokens per iteration and you cap how much prefill can stall everyone’s decode. This is the lever that stops one long prompt from becoming everybody’s latency spike.',
  'Chunked prefill: a long prompt is prefilled a chunk at a time, interleaved with decode, instead of in one enormous step that blocks the whole batch.',
  'Slots filling and emptying means the scheduler is doing its job. A bar pinned at the top with a rising queue means it cannot keep up, and latency is about to explode.',
];

const PRESET: Record<number, { maxSeqs: number; budget: number }> = {
  0: { maxSeqs: 4, budget: 2048 },
  1: { maxSeqs: 4, budget: 2048 },
  2: { maxSeqs: 4, budget: 2048 },
  3: { maxSeqs: 8, budget: 2048 },
  4: { maxSeqs: 8, budget: 512 },
  5: { maxSeqs: 8, budget: 512 },
  6: { maxSeqs: 8, budget: 2048 },
};

/** A synthetic workload: a short-chat majority and a long tail. */
const REQS = [
  { id: 'a', len: 6, kind: 'short' as const },
  { id: 'b', len: 11, kind: 'short' as const },
  { id: 'e', len: 9, kind: 'short' as const },
  { id: 'g', len: 7, kind: 'short' as const },
  { id: 'c', len: 40, kind: 'long' as const },
  { id: 'd', len: 44, kind: 'long' as const },
  { id: 'f', len: 48, kind: 'long' as const },
  { id: 'h', len: 46, kind: 'long' as const },
];
const GROUP_A = ['a', 'b', 'e', 'g'];
const GROUP_B = ['c', 'd', 'f', 'h'];

const X0 = 118;
const X1 = 624;
const MAX_STEPS = 52;
const SW = (X1 - X0) / MAX_STEPS;
const LANE_Y = 96;
const LANE_H = 17;
const LANE_GAP = 22;

type Lane = { id: string; from: number; to: number; kind: 'short' | 'long' };

/** Static: two fixed groups of four, each running to its own longest member. */
function staticSchedule(): { lanes: Lane[]; boundaries: number[]; span: number } {
  const lanes: Lane[] = [];
  const boundaries: number[] = [];
  let cur = 0;
  for (const group of [GROUP_A, GROUP_B]) {
    const longest = Math.max(...group.map((id) => REQS.find((r) => r.id === id)!.len));
    for (const id of group) {
      const r = REQS.find((x) => x.id === id)!;
      lanes.push({ id, from: cur, to: cur + r.len, kind: r.kind });
    }
    boundaries.push(cur);
    cur += longest;
  }
  boundaries.push(cur);
  return { lanes, boundaries, span: Math.min(MAX_STEPS, cur) };
}

/** Continuous: refill free slots from the waiting queue at every step boundary. */
function continuousSchedule(maxSeqs: number): { lanes: Lane[]; sizes: number[]; span: number } {
  const lanes: Lane[] = [];
  const sizes: number[] = [];
  const queue = [...REQS];
  let t = 0;
  const live: { len: number; left: number }[] = [];
  while (t < MAX_STEPS && (queue.length || live.length)) {
    while (live.length < maxSeqs && queue.length) {
      const r = queue.shift()!;
      lanes.push({ id: r.id, from: t, to: Math.min(MAX_STEPS, t + r.len), kind: r.kind });
      live.push({ len: r.len, left: r.len });
    }
    if (!live.length) break;
    sizes.push(live.length);
    for (const s of live) s.left -= 1;
    for (let i = live.length - 1; i >= 0; i--) if (live[i].left <= 0) live.splice(i, 1);
    t += 1;
  }
  return { lanes, sizes, span: t };
}

export default function ContinuousBatching() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const p0 = PRESET[s.step];
  const [maxSeqs, setMaxSeqs] = useState(p0.maxSeqs);
  const [budget, setBudget] = useState(p0.budget);
  useEffect(() => { setMaxSeqs(p0.maxSeqs); setBudget(p0.budget); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const stat = useMemo(() => staticSchedule(), []);
  const cont = useMemo(() => continuousSchedule(maxSeqs), [maxSeqs]);

  const k = s.step;
  const continuous = k >= 2;
  const lanes = continuous ? cont.lanes : stat.lanes;
  const span = Math.min(MAX_STEPS, continuous ? cont.span : stat.span);
  const slots = continuous ? maxSeqs : 4;

  // occupancy: slot-steps actually producing a token, over slot-steps available
  const busy = useMemo(() => {
    if (continuous) {
      let filled = 0;
      for (let t = 0; t < cont.span; t++) {
        filled += cont.lanes.filter((l) => l.from <= t && t < l.to).length;
      }
      return filled / (cont.span * maxSeqs);
    }
    let filled = 0;
    for (let t = 0; t < stat.span; t++) {
      filled += stat.lanes.filter((l) => l.from <= t && t < l.to).length;
    }
    return filled / (stat.span * 4);
  }, [continuous, cont, stat, maxSeqs]);

  const r = decodeStep({ model, gpu, batch: maxSeqs, ctx: 1024 });
  const budgetW = X1 - X0;
  const decodeFrac = Math.max(0.08, Math.min(0.9, (budget - maxSeqs) / budget));

  return (
    <Stage
      rootRef={ref}
      title="Continuous batching: the batch is re-formed at every step"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'compute', ['idle', 'idle slot'], ['error', 'prefill stall']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Iterations to finish', value: `${span}`, kind: 'text', note: continuous ? `static needed ${stat.span}` : 'static batching' },
          { label: 'Slots producing', value: `${(busy * 100).toFixed(0)}%`, kind: 'token', note: `${((1 - busy) * 100).toFixed(0)}% of the batch idle` },
          { label: 'Step time', value: fmt.ms(r.t), kind: 'compute', note: `batch ${maxSeqs}, 1k context` },
          { label: 'Throughput', value: `${fmt.int(r.total)} tok/s`, kind: 'token', note: 'all users together' },
        ]} />
      }
      controls={
        <>
          <Slider label="Max sequences in the batch" value={maxSeqs} min={2} max={16} onChange={setMaxSeqs} format={(v) => `${v}`} kind="token" />
          <Slider label="Token budget per iteration" value={budget} min={128} max={4096} log onChange={setBudget} format={(v) => `${fmt.int(v)} tokens`} kind="compute" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A timeline of eight requests across engine iterations. Under static batching they run as two fixed groups of four and short requests sit idle at the end of their group. Under continuous batching each request occupies a slot only for its own duration and waiting requests join as slots free up. A bar below shows how many slots are producing tokens at each step.">
        <Label x={X0} y={20} tone={3} size={10}>engine iterations →</Label>
        <Ruler x={X0} y={78} w={X1 - X0} ticks={[{ at: 0, label: '0' }, { at: 0.25 }, { at: 0.5, label: `${MAX_STEPS / 2}` }, { at: 0.75 }, { at: 1, label: `${MAX_STEPS}` }]} />

        {/* static batch boundaries */}
        <Fade show={!continuous}>
          {stat.boundaries.map((b, i) => (
            <line key={i} x1={X0 + b * SW} x2={X0 + b * SW} y1={82} y2={LANE_Y + 8 * LANE_GAP - 6} stroke="var(--text-3)" strokeDasharray="3 4" />
          ))}
          <Label x={X0 + (stat.boundaries[0] + stat.boundaries[1]) / 2 * SW} y={LANE_Y + 8 * LANE_GAP + 6} anchor="middle" tone={3} size={9}>group 1 ends here → group 2 starts</Label>
        </Fade>
        <Fade show={continuous}>
          <Label x={(X0 + X1) / 2} y={LANE_Y + 8 * LANE_GAP + 6} anchor="middle" tone={3} size={9}>every boundary: finish → free → admit</Label>
        </Fade>

        {/* request lanes */}
        {lanes.map((l) => {
          const i = REQS.findIndex((r) => r.id === l.id);
          const w = Math.max(3, (l.to - l.from) * SW);
          return (
            <g key={l.id}>
              <text x={X0 - 9} y={LANE_Y + i * LANE_GAP + 12} textAnchor="end" className="v-label" style={{ fontSize: 10, fill: 'var(--text-3)' }}>{l.id}</text>
              <rect x={X0 + l.from * SW} y={LANE_Y + i * LANE_GAP} width={w} height={LANE_H} rx={2.5}
                fill={`var(--${l.kind === 'short' ? 'token' : 'compute'})`} className="v-fade" />
              <Label x={X0 + l.from * SW + w / 2} y={LANE_Y + i * LANE_GAP + 12} anchor="middle" size={9}
                style={{ fill: '#0e1116' }}>{`${l.to - l.from}`}</Label>
            </g>
          );
        })}

        {/* the idle tail, static only */}
        <Fade show={!continuous}>
          {stat.lanes.map((l) => {
            const i = REQS.findIndex((r) => r.id === l.id);
            const groupEnd = GROUP_A.includes(l.id) ? stat.boundaries[1] : stat.boundaries[2];
            const idle = groupEnd - l.to;
            if (idle <= 0) return null;
            return (
              <g key={`idle-${l.id}`}>
                <rect x={X0 + l.to * SW} y={LANE_Y + i * LANE_GAP} width={idle * SW} height={LANE_H} rx={2.5} fill="url(#tps-hatch)" stroke="var(--idle)" />
                <Label x={X0 + (l.to + groupEnd) / 2 * SW} y={LANE_Y + i * LANE_GAP + 12} anchor="middle" size={9} tone={3}>{`${idle}`}</Label>
              </g>
            );
          })}
          <Label x={X1} y={LANE_Y - 8} anchor="end" size={9.5} tone={3}>hatched = finished, still occupying a slot</Label>
        </Fade>

        {/* occupancy strip */}
        <Label x={X0} y={300} tone={2} size={10.5} mono={false}>{`slots producing tokens, step by step (${slots} slots)`}</Label>
        <rect x={X0} y={306} width={X1 - X0} height={16} rx={2} fill="none" stroke="var(--rule)" />
        {Array.from({ length: span }, (_, t) => {
          const busyNow = lanes.filter((l) => l.from <= t && t < l.to).length;
          const fw = Math.max(1, (1 / span) * (X1 - X0) - 0.6);
          const fh = (busyNow / slots) * 16;
          return (
            <g key={t}>
              <rect x={X0 + (t / span) * (X1 - X0)} y={306} width={fw} height={16} fill="url(#tps-hatch)" opacity={0.5} />
              {busyNow > 0 && <rect x={X0 + (t / span) * (X1 - X0)} y={306 + 16 - fh} width={fw} height={fh} fill="var(--token)" opacity={0.9} />}
            </g>
          );
        })}
        <Label x={X1} y={336} anchor="end" tone={3} size={9.5}>{`${(busy * 100).toFixed(0)}% of slot-steps produce a token`}</Label>

        {/* token budget */}
        <Fade show={k >= 4}>
          <Label x={X0} y={358} tone={2} size={10.5} mono={false}>{`one iteration, ${fmt.int(budget)} tokens of budget`}</Label>
          <rect x={X0} y={362} width={budgetW * decodeFrac} height={12} rx={2} fill="var(--token)" />
          <rect x={X0 + budgetW * decodeFrac} y={362} width={budgetW * (1 - decodeFrac)} height={12} rx={2} fill="var(--compute-dim)" stroke="var(--compute)" />
          <Label x={X0 + budgetW * decodeFrac / 2} y={371} anchor="middle" size={9} style={{ fill: '#0e1116' }}>{`decode ${maxSeqs}`}</Label>
          <Label x={X0 + budgetW * (1 + decodeFrac) / 2} y={371} anchor="middle" size={9} tone={2}>prefill chunk — everyone waits</Label>
        </Fade>
      </svg>
    </Stage>
  );
}