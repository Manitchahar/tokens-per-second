import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Span, Label, Fade, Ruler, fmt } from '../lib/svg';
import { simulate, simulateStatic, type ReqSpec } from './sim';

// Continuous batching: the batch is re-formed every engine step. The same 12-request
// workload is run twice on one shared time axis — once as static batches, once with a
// scheduler that refills every freed seat on the very next step — so the difference
// between "the batch" and "the seats" becomes obvious.

const STEPS = [
  'Static: the tail is dead weight',
  'The batch is a promise',
  'Admission: fill the seats',
  'A short request finishes',
  'Its seat is now free',
  'Refill: the next one in',
  'Step time is the same',
  'Continuous wins, and why',
  'The one-line rule',
];
const CAPTIONS = [
  'Static batching pays for every seat until the slowest member of each group ends. On this workload roughly 38% of what the GPU paid for produced nothing.',
  'A static batch is a promise: the group was committed together, so nobody leaves and nobody joins until the last one is done.',
  'Continuous batching starts the same way. At engine step 1 the scheduler fills as many seats as it can from the queue.',
  'By step 20 several short requests are done. Their output is complete, but their seat is still occupied, still reading KV, still producing nothing.',
  'The rule that fixes it: a seat belongs to the scheduler, not to the request. The moment a request leaves, its seat is free.',
  'So on the very next step the scheduler refills it from the queue. No barrier, no waiting for a group to drain.',
  'Every step the engine reads the same weights and the same KV bytes. The step time barely changed; only how much comes out of it did.',
  'The wave finished 14% sooner in wall-clock steps, and every single seat produced a token at every single step. Two different wastes, one arrow between them.',
  'That is the whole idea: form the batch fresh every step, and fill freed seats immediately.',
];

// A fixed, ragged workload: 12 requests arriving over time, with different lengths.
const LENGTHS = [44, 176, 62, 152, 98, 38, 140, 84, 120, 68, 90, 110];
const ARRIVALS = [0, 0, 5, 9, 14, 18, 22, 27, 33, 40, 46, 52];

const X0 = 96;
const X1 = 626;
const N = LENGTHS.length;

type Mode = 'static' | 'continuous';

export default function ContinuousBatching() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [slots, setSlots] = useState(6);
  const [tick, setTick] = useState(6);

  const reqs = useMemo<ReqSpec[]>(
    () => LENGTHS.map((want, i) => ({ id: i, arrive: ARRIVALS[i], want })),
    [],
  );
  const st = useMemo(() => simulateStatic(reqs, slots), [reqs, slots]);
  const cb = useMemo(() => simulate({ reqs, slots }), [reqs, slots]);

  const scale = Math.max(st.steps, cb.steps, 1);
  const t = useTweens({ scale }, 600);
  const px = (step: number) => (step / t.scale) * (X1 - X0);

  const stWaste = st.slotsUsed ? st.wastedSlots / st.slotsUsed : 0;
  const cbWaste = cb.slotsUsed ? cb.wastedSlots / cb.slotsUsed : 0;
  const showStatic = s.step <= 1;
  const saved = Math.max(0, Math.round((1 - cb.steps / Math.max(1, st.steps)) * 100));

  // Continuous rows: each request from admission to its last produced token.
  const cbRows = useMemo(
    () =>
      reqs.map((r) => {
        const start = cb.admitted[r.id];
        const end = cb.finished[r.id];
        if (start < 0 || end === undefined) return { id: r.id, start: 0, end: 0, want: r.want };
        return { id: r.id, start, end: end + 1, want: r.want };
      }),
    [reqs, cb],
  );
  const rows = showStatic ? st.rows : cbRows;

  // Seat occupancy per step of the continuous run (capped at the first 40 steps).
  const busy = cb.batchSize.slice(0, 40);
  const avgBatch = cb.batchSize.reduce((a, b) => a + b, 0) / Math.max(1, cb.batchSize.length);
  const fullSteps = cb.batchSize.filter((b) => b === slots).length;

  // Static: token portion of each row ends when that request finished.
  const stSpan = Math.max(1, Math.max(...st.rows.map((r) => r.end - r.start)));

  return (
    <Stage
      rootRef={ref}
      title="The same 12 requests, run twice: static batching vs continuous batching"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', ['error', 'seat paid for, token wasted'], ['idle', 'waiting in the queue']]}
      {...s}
      readout={
        <Readout
          items={[
            {
              label: 'Wall-clock steps',
              value: `${showStatic ? st.steps : cb.steps}`,
              kind: 'compute',
              note: showStatic ? `static batches of ${slots}` : `continuous, ${slots} seats`,
            },
            {
              label: 'Seats wasted',
              value: `${Math.round((showStatic ? stWaste : cbWaste) * 100)}%`,
              kind: 'error',
              note: showStatic ? 'held past the token' : 'every seat produced every step',
            },
            {
              label: showStatic ? 'Continuous would save' : 'Average batch',
              value: showStatic ? `${saved}%` : `${avgBatch.toFixed(1)} / ${slots}`,
              kind: 'token',
              note: showStatic ? 'same tokens, fewer steps' : `${fullSteps} of ${cb.batchSize.length} steps ran a full batch`,
            },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            label="View"
            value={showStatic ? 'static' : 'continuous'}
            options={[
              { value: 'static', label: 'Static' },
              { value: 'continuous', label: 'Continuous' },
            ]}
            onChange={(v) => (v === 'static' ? s.go(1) : s.go(4))}
          />
          <Slider
            label="Seats in the engine"
            value={slots}
            min={2}
            max={N}
            onChange={setSlots}
            format={(v) => String(v)}
            kind="compute"
          />
          <Slider
            label="Engine step cursor"
            value={tick}
            min={1}
            max={Math.max(st.steps, cb.steps)}
            onChange={setTick}
            format={(v) => `step ${v}`}
            kind="memory"
          />
        </>
      }
    >
      <svg
        viewBox="0 0 640 392"
        role="img"
        aria-label="A timeline of twelve requests arriving over time. Static batching holds every seat until the slowest member of each group finishes, so short requests waste their seats. Continuous batching frees each seat the moment its request ends and refills it from the queue on the next step."
      >
        <Label x={X0} y={16} size={11} kind={showStatic ? 'error' : 'token'} bold>
          {showStatic ? 'Static batching · the group is committed together' : 'Continuous batching · the group is re-formed every step'}
        </Label>

        {/* waiting queue */}
        <text x={X0 - 10} y={40} textAnchor="end" className="v-track-label">
          Queue
        </text>
        {reqs.map((r, i) => {
          const queued = showStatic ? i >= slots : cb.admitted[r.id] >= tick;
          return (
            <g key={r.id}>
              <rect
                x={X0 + i * 12}
                y={30}
                width={10}
                height={12}
                rx={2}
                fill={queued ? 'url(#tps-hatch)' : 'none'}
                stroke={queued ? 'var(--idle)' : 'var(--rule-2)'}
                strokeWidth={1}
              >
                <title>{`U${i + 1} · ${r.want} tokens`}</title>
              </rect>
            </g>
          );
        })}
        <Label x={X0 + N * 12 + 8} y={40} size={9.5} tone={3}>
          {`${N} requests`}
        </Label>

        {/* the Gantt */}
        {rows.map((r, i) => {
          const row = r as { start: number; end: number; want: number };
          const y = 58 + i * 12;
          const x = X0 + px(row.start);
          const w = Math.max(1.5, px(row.end) - px(row.start));
          const tokenFrac = showStatic ? row.want / stSpan : 1;
          const tokenW = Math.max(1, tokenFrac * w);
          return (
            <g key={i}>
              <text x={X0 - 10} y={y + 7} textAnchor="end" className="v-label t3" style={{ fontSize: 9 }}>
                {`U${i + 1}`}
              </text>
              <Span x={x} y={y} w={w} h={9} kind="token" opacity={0.9} />
              {showStatic && w > tokenW + 0.6 && (
                <Span x={x + tokenW} y={y} w={w - tokenW} h={9} kind="error" opacity={0.8} />
              )}
            </g>
          );
        })}

        {/* step cursor for the continuous view */}
        {!showStatic && (
          <g>
            <line
              x1={X0 + px(tick)}
              x2={X0 + px(tick)}
              y1={52}
              y2={52 + N * 12 + 4}
              stroke="var(--text-3)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <Label x={X0 + px(tick)} y={48} anchor="middle" size={9.5} tone={3}>
              {`step ${tick}`}
            </Label>
          </g>
        )}

        {/* ruler */}
        <Ruler
          x={X0}
          y={212}
          w={X1 - X0}
          ticks={[{ at: 0, label: '0' }, { at: 0.5 }, { at: 1, label: `${Math.round(scale)} steps` }]}
          label="engine steps"
        />

        {/* seat occupancy, continuous only */}
        <Fade show={s.step >= 5 && !showStatic}>
          <text x={X0 - 10} y={240} textAnchor="end" className="v-track-label">
            Seats
          </text>
          <rect x={X0} y={228} width={X1 - X0} height={40} rx={2} fill="none" stroke="var(--rule-2)" />
          {busy.map((b, i) => {
            const w = (X1 - X0) / busy.length;
            const h = Math.max(2, (b / slots) * 40);
            return (
              <rect
                key={i}
                x={X0 + i * w}
                y={228 + 40 - h}
                width={Math.max(1, w - 1)}
                height={h}
                rx={1}
                fill="var(--compute)"
                opacity={0.6}
              />
            );
          })}
          <Label x={X0} y={282} size={9.5} tone={3}>
            {`seats in use, steps 1–${busy.length} of ${cb.steps} · average ${avgBatch.toFixed(1)} of ${slots}`}
          </Label>
        </Fade>

        {/* waste accounting */}
        <Fade show={s.step >= 3}>
          <text x={X0 - 10} y={312} textAnchor="end" className="v-track-label">
            Paid
          </text>
          <rect x={X0} y={302} width={X1 - X0} height={12} rx={2} fill="none" stroke="var(--rule-2)" />
          <Span
            x={X0}
            y={302}
            w={(X1 - X0) * (1 - (showStatic ? stWaste : cbWaste))}
            h={12}
            kind="token"
            label={`${fmt.int(cb.useful)} tokens`}
          />
          <Span
            x={X0 + (X1 - X0) * (1 - (showStatic ? stWaste : cbWaste))}
            y={302}
            w={(X1 - X0) * (showStatic ? stWaste : cbWaste)}
            h={12}
            kind="error"
            label={`${Math.round((showStatic ? stWaste : cbWaste) * 100)}% wasted`}
            labelTone="light"
          />
          <Label x={X1} y={326} anchor="end" size={9.5} tone={3}>
            {showStatic ? 'seats held for the slowest member of each group' : 'every seat produces a token at every step'}
          </Label>
        </Fade>

        {/* the one-line rule */}
        <Fade show={s.step >= 7}>
          <g>
            <rect x={X0} y={342} width={X1 - X0} height={40} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
            <text x={X0 + 14} y={359} className="v-label" style={{ fontSize: 12, fill: 'var(--token)' }}>
              the seat belongs to the scheduler, not to the request
            </text>
            <text x={X0 + 14} y={374} className="v-label t3" style={{ fontSize: 10.5 }}>
              when a request leaves, its seat is free and the queue fills it on the very next step
            </text>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}