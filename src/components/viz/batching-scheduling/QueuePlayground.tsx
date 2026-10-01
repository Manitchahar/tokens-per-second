import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade, Ruler, fmt } from '../lib/svg';
import { simulate, type ReqSpec } from './sim';

// The queue is where fairness, cancellation and preemption live. Four policies over one
// workload: first-come-first-served, priority, cancellation, and preemption under memory
// pressure. Each answers the same question differently: who gets the next free seat?

const STEPS = [
  'The queue is a policy',
  'FIFO: fair and blind',
  'FIFO is not always fair',
  'Priority: who is more valuable',
  'Priority starves',
  'The head-of-line problem',
  'Cancellation: the seat comes back',
  'Preemption: eviction',
  'Preemption costs recompute',
  'Choosing a policy',
];
const CAPTIONS = [
  'A scheduler is two decisions: who is next, and what happens when someone leaves. Both are policy, and both are where fairness lives.',
  'First-come-first-served is the default. Requests are admitted in arrival order and no one can be jumped. It is easy to reason about and hard to beat on fairness.',
  'But FIFO is blind to size. A 300-token request and four 20-token requests are all equally important to the queue, and a long request holds its seat for a long time.',
  'Priority fixes the value judgement: higher-priority requests are admitted first, so a paid tier can jump a free one.',
  'And that is exactly the danger. Under a constant flood of high-priority traffic, low-priority requests can wait forever. Starvation is the price of priority.',
  'The head-of-line problem is the milder cousin: four short requests finish, but the seat is not free until the long request beside them is done. FIFO does not fix it either.',
  'Cancellation is the easiest win in production. When a client disconnects, the engine frees its seat and its KV blocks immediately — no waiting for the batch.',
  'Preemption is eviction under memory pressure. When a new request needs more KV than is free, the engine evicts the newest sequence, the one cheapest to rebuild.',
  'The price of preemption is recompute. An evicted sequence lost its KV cache and must replay the tokens it already sent. That is a real, paid cost.',
  'Every policy here trades one group’s latency for another’s. Choosing is knowing who you are willing to make wait.',
];

const X0 = 104;
const X1 = 622;

type Policy = 'fifo' | 'priority' | 'cancel' | 'preempt';

// A workload built so every policy tells a different story:
//   R0, R2  two 200-token jobs that take both seats for 200 steps
//   R1      a LOW-priority 20-token request that arrives at step 5, while full
//   R3..R11 high-priority 20-token requests arriving from step 50 onward
// FIFO admits R1 in arrival order; priority jumps it to the back; cancelling R0 or R2
// frees a seat a full 200 steps early; preemption evicts and pays to replay.
const WORK: ReqSpec[] = [
  { id: 0, arrive: 0, want: 200, priority: 0 },
  { id: 1, arrive: 5, want: 20, priority: 9 }, // low priority
  { id: 2, arrive: 0, want: 200, priority: 0 },
  ...Array.from({ length: 9 }, (_, i) => ({ id: 3 + i, arrive: 50 + i * 10, want: 20, priority: 0 })),
];

export default function QueuePlayground() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [policy, setPolicy] = useState<Policy>('fifo');
  const [slots, setSlots] = useState(2);

  const runs = useMemo(
    () => ({
      fifo: simulate({ reqs: WORK, slots }),
      priority: simulate({ reqs: WORK, slots, policy: 'priority' }),
      cancel: simulate({ reqs: WORK, slots, cancel: { 0: 60 } }),
      preempt: simulate({ reqs: WORK, slots, kvSlots: Math.max(1, slots - 1) }),
    }),
    [slots],
  );
  const r = runs[policy];
  const scale = Math.max(...Object.values(runs).map((x) => x.steps), 1);

  return (
    <Stage
      rootRef={ref}
      title="Who gets the next seat? FIFO, priority, cancellation and preemption"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', ['error', 'seat blocked / evicted'], ['idle', 'waiting'], 'compute']}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'All done after', value: `${r.steps} steps`, kind: 'compute', note: `${slots} seats, policy ${policy}` },
            {
              label: 'R1 (low priority) waits',
              value: `${r.admitted[1] - 5} steps`,
              kind: r.admitted[1] - 5 > 250 ? 'error' : 'token',
              note: 'arrived at step 5',
            },
            {
              label: 'Tokens produced',
              value: `${fmt.int(r.useful)}`,
              kind: 'token',
              note: r.useful < 600 ? 'of 600 wanted' : 'all of them',
            },
            {
              label: 'Recompute paid',
              value: `${fmt.int(r.recomputeTokens)}`,
              kind: r.recomputeTokens > 0 ? 'error' : 'text',
              note: r.recomputeTokens > 0 ? 'lost to preemption' : 'nothing evicted',
            },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            label="Policy"
            value={policy}
            options={[
              { value: 'fifo', label: 'FIFO' },
              { value: 'priority', label: 'Priority' },
              { value: 'cancel', label: 'Cancel' },
              { value: 'preempt', label: 'Preempt' },
            ]}
            onChange={setPolicy}
          />
          <Slider label="Seats" value={slots} min={2} max={8} onChange={setSlots} format={(v) => String(v)} kind="compute" />
        </>
      }
    >
      <svg
        viewBox="0 0 640 400"
        role="img"
        aria-label="A timeline of twelve requests under four scheduling policies. FIFO admits in arrival order, priority admits high-priority requests first and starves the low-priority one, cancellation frees a seat immediately when a client disconnects, and preemption evicts a sequence to make room and pays to recompute it."
      >
        <Label x={X0} y={16} size={11} kind="text" bold>
          Twelve requests, two seats, four ways to choose who goes next
        </Label>

        {/* timeline rows */}
        {WORK.map((req, i) => {
          const y = 44 + i * 10;
          const start = r.admitted[req.id];
          const end = r.finished[req.id] ?? start + req.want;
          const isLow = (req.priority ?? 0) > 0;
          const cancelled = r.cancelled.includes(req.id);
          const wasPreempted = r.preempted.includes(req.id);
          const ax = X0 + (req.arrive / scale) * (X1 - X0);
          const x = X0 + (Math.max(0, start) / scale) * (X1 - X0);
          const w = Math.max(2, ((end - Math.max(0, start)) / scale) * (X1 - X0));
          return (
            <g key={req.id}>
              <text x={X0 - 10} y={y + 6} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>
                {`R${req.id}`}
              </text>
              {/* hatched: time spent waiting for a seat */}
              <rect x={ax} y={y} width={Math.max(0, x - ax)} height={8} rx={2} fill="url(#tps-hatch)" stroke="var(--idle)" strokeWidth={0.7} />
              <Span
                x={x}
                y={y}
                w={w}
                h={8}
                kind={cancelled ? 'error' : isLow && s.step >= 4 ? 'compute' : 'token'}
                opacity={0.92}
              />
              {cancelled && <Label x={x + w + 4} y={y + 6} size={8} kind="error">cancelled</Label>}
              {wasPreempted && s.step >= 7 && !cancelled && <Label x={x + w + 4} y={y + 6} size={8} kind="error">evicted → replay</Label>}
              {s.step >= 3 && isLow && <circle cx={X0 - 22} cy={y + 4} r={2.5} fill="var(--error)" />}
            </g>
          );
        })}

        <Ruler
          x={X0}
          y={170}
          w={X1 - X0}
          ticks={[{ at: 0, label: '0' }, { at: 0.5 }, { at: 1, label: `${scale} steps` }]}
          label="engine steps"
        />

        {/* when the last wave arrives */}
        <g>
          <line
            x1={X0 + (50 / scale) * (X1 - X0)}
            x2={X0 + (50 / scale) * (X1 - X0)}
            y1={38}
            y2={166}
            stroke="var(--error)"
            strokeDasharray="3 3"
            strokeWidth={1}
          />
          <Label x={X0 + (50 / scale) * (X1 - X0) + 4} y={36} size={8.5} kind="error">
            high-priority flood arrives
          </Label>
        </g>

        {/* comparison table */}
        <Fade show={s.step >= 1}>
          <g>
            <Label x={X0} y={196} size={10} kind="text">The same workload under every policy</Label>
            {(['fifo', 'priority', 'cancel', 'preempt'] as Policy[]).map((p, i) => {
              const rr = runs[p];
              const y = 208 + i * 26;
              const isThis = p === policy;
              const r1Wait = rr.admitted[1] - 5;
              return (
                <g key={p}>
                  <text x={X0} y={y + 11} className="v-label" style={{ fontSize: 10, fontWeight: isThis ? 700 : 400, fill: isThis ? 'var(--text)' : 'var(--text-3)' }}>
                    {p === 'cancel' ? 'cancellation' : p === 'preempt' ? 'preemption' : p}
                  </text>
                  <rect x={X0 + 84} y={y} width={(X1 - X0 - 84) * 0.62} height={16} rx={2} fill="none" stroke="var(--rule-2)" />
                  <Span
                    x={X0 + 84}
                    y={y}
                    w={((X1 - X0 - 84) * 0.62 * rr.steps) / scale}
                    h={16}
                    kind={p === 'preempt' ? 'error' : 'compute'}
                    opacity={isThis ? 1 : 0.45}
                  />
                  <Label
                    x={X0 + 84 + (X1 - X0 - 84) * 0.62 * (rr.steps / scale) + 5}
                    y={y + 12}
                    size={9}
                    tone={isThis ? 1 : 3}
                  >{`${rr.steps} steps`}</Label>
                  <Label x={X1} y={y + 12} anchor="end" size={9} kind={p === 'priority' && r1Wait > 250 ? 'error' : 'text'} tone={3}>
                    {`R1 waits ${r1Wait}${rr.recomputeTokens ? ` · ${rr.recomputeTokens} recomputed` : ''}`}
                  </Label>
                </g>
              );
            })}
          </g>
        </Fade>

        {/* the point of each policy */}
        <Fade show={s.step >= 9}>
          <g>
            <rect x={X0} y={318} width={X1 - X0} height={70} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
            <text x={X0 + 12} y={335} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text)' }}>Fairness, in one line: FIFO shares pain, priority spends it.</text>
            <text x={X0 + 12} y={351} className="v-label t3" style={{ fontSize: 10 }}>Cancellation is free money — it gives back a seat nobody is using.</text>
            <text x={X0 + 12} y={367} className="v-label t3" style={{ fontSize: 10 }}>Preemption is the emergency valve: it saves the batch, and the evicted user pays.</text>
            <text x={X0 + 12} y={383} className="v-label t3" style={{ fontSize: 10 }}>Pick the policy that matches who you promised to serve first.</text>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}