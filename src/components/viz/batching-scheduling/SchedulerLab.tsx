import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade, Ruler, fmt } from '../lib/svg';
import { simulate, simulateStatic, type ReqSpec, pct } from './sim';

// The in-browser scheduler lab. You build a workload, choose a policy, and watch the
// engine step through it. Everything the page claims about scheduling is computed here,
// by the same simulate() the other visuals use.

const STEPS = [
  'The workload',
  'Watch one step',
  'Run it',
  'Read the result',
  'Break it: too few seats',
  'Break it: too many seats',
  'Fairness under pressure',
  'Your turn',
];
const CAPTIONS = [
  'A workload is a list of requests: when each arrives, how many tokens it wants, and optionally its priority. Change the sliders and the timeline redraws.',
  'Each cell is one engine step. Green = a sequence that produced a token somebody wanted. Red = a seat paid for that produced nothing. Hatched = waiting.',
  'Press run and the engine steps through the whole workload. Finished requests leave, waiting ones join, on the very next step.',
  'Four numbers decide whether a configuration is any good: steps to finish, tokens produced, average wait (TTFT) and the worst wait (p99).',
  'Two seats for twelve requests is starvation by arithmetic: most users wait nearly the whole run. The GPU is busy the entire time and almost nobody is served.',
  'Eighty seats for twelve requests is the opposite waste: the batch is never full, so you pay for a GPU that is mostly idle. More seats is not better.',
  'Raise the arrival rate and the picture changes: with a deep queue the engine fills its seats and the average wait becomes the thing worth optimising.',
  'Now it is yours. Find the seat count that finishes this workload fastest while keeping the worst wait under a number you choose.',
];

const X0 = 100;
const X1 = 624;

type Policy = 'fifo' | 'priority';

export default function SchedulerLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [n, setN] = useState(12);
  const [avg, setAvg] = useState(120);
  const [spacing, setSpacing] = useState(20); // steps between arrivals
  const [slots, setSlots] = useState(6);
  const [policy, setPolicy] = useState<Policy>('fifo');
  const [ragged, setRagged] = useState(1); // 0 = uniform, 1 = maximally ragged
  const [showStatic, setShowStatic] = useState(false);

  const reqs = useMemo<ReqSpec[]>(() => {
    // deterministic ragged lengths around avg, plus a stable priority for the last request
    const out: ReqSpec[] = [];
    for (let i = 0; i < n; i++) {
      const f = (i * 0.6180339887) % 1; // golden-ratio spread: stable, no Math.random
      const factor = 1 - ragged * 0.75 + ragged * 1.5 * f;
      const want = Math.max(8, Math.round(avg * factor));
      out.push({ id: i, arrive: i * spacing, want, priority: i === n - 1 ? 9 : 0 });
    }
    return out;
  }, [n, avg, spacing, ragged]);

  const run = useMemo(() => simulate({ reqs, slots, policy }), [reqs, slots, policy]);
  const stat = useMemo(() => simulateStatic(reqs, slots), [reqs, slots]);
  const r = showStatic ? stat : run;

  const scale = Math.max(run.steps, stat.steps, 1);

  const waits = reqs.map((q) => (run.admitted[q.id] >= 0 ? run.admitted[q.id] - q.arrive : -1));
  const realWaits = waits.filter((w) => w >= 0);
  const avgWait = realWaits.length ? realWaits.reduce((a, b) => a + b, 0) / realWaits.length : 0;
  const p99 = pct(realWaits, 99);
  const fill = run.batchSize.length ? run.batchSize.reduce((a, b) => a + b, 0) / run.batchSize.length / slots : 0;
  const wanted = reqs.reduce((a, b) => a + b.want, 0);
  const staticWaste = stat.slotsUsed ? stat.wastedSlots / stat.slotsUsed : 0;

  return (
    <Stage
      rootRef={ref}
      title="Scheduler lab: build a workload and run it"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', ['error', 'seat paid for, no token'], ['idle', 'waiting'], 'compute']}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'Engine steps to finish', value: fmt.int(showStatic ? stat.steps : run.steps), kind: 'compute', note: showStatic ? 'static batching' : 'continuous batching' },
            { label: 'Tokens produced', value: fmt.int(showStatic ? stat.useful : run.useful), kind: 'token', note: `of ${fmt.int(wanted)} wanted` },
            {
              label: 'Average wait (TTFT)',
              value: `${Math.round(avgWait)} steps`,
              kind: 'token',
              note: `${realWaits.length} of ${n} served`,
            },
            { label: 'Worst wait (p99)', value: `${fmt.int(p99)} steps`, kind: p99 > avgWait * 2 ? 'error' : 'token', note: 'the user who complains' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            label="Engine"
            value={showStatic ? 'static' : 'continuous'}
            options={[
              { value: 'continuous', label: 'Continuous' },
              { value: 'static', label: 'Static' },
            ]}
            onChange={(v) => setShowStatic(v === 'static')}
          />
          <Segmented
            label="Policy"
            value={policy}
            options={[
              { value: 'fifo', label: 'FIFO' },
              { value: 'priority', label: 'Priority' },
            ]}
            onChange={setPolicy}
          />
          <Slider label="Requests" value={n} min={4} max={24} onChange={setN} format={(v) => String(v)} kind="compute" />
          <Slider label="Average length" value={avg} min={20} max={400} step={10} onChange={setAvg} format={(v) => `${v} tok`} kind="memory" />
          <Slider label="Arrivals every" value={spacing} min={1} max={60} onChange={setSpacing} format={(v) => `${v} steps`} kind="idle" />
          <Slider label="Seats" value={slots} min={1} max={24} onChange={setSlots} format={(v) => String(v)} kind="token" />
          <Slider label="Raggedness" value={ragged} min={0} max={100} step={5} onChange={setRagged} format={(v) => (v === 0 ? 'uniform' : `${v}%`)} kind="error" />
        </>
      }
    >
      <svg
        viewBox="0 0 640 392"
        role="img"
        aria-label="An interactive scheduler simulator. Each row is a request over time: hatched while it waits for a seat, green while it produces tokens, red when a seat is paid for but produces nothing. Controls set the number of requests, their length, arrival rate, seat count and policy."
      >
        <Label x={X0} y={15} size={11} kind="text" bold>
          {`${n} requests · ${slots} seats · ${showStatic ? 'static batching' : `continuous, ${policy}`}`}
        </Label>

        {/* the timeline */}
        {reqs.slice(0, 16).map((q, i) => {
          const y = 34 + i * 12;
          const start = showStatic ? (stat.rows[i]?.start ?? 0) : run.admitted[q.id];
          const end = showStatic ? (stat.rows[i]?.end ?? 0) : run.finished[q.id] ?? start + q.want;
          if (start < 0) return null;
          const ax = X0 + (q.arrive / scale) * (X1 - X0);
          const x = X0 + (start / scale) * (X1 - X0);
          const w = Math.max(2, ((end - start) / scale) * (X1 - X0));
          return (
            <g key={q.id}>
              <text x={X0 - 8} y={y + 7} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>{`R${q.id}`}</text>
              <rect x={ax} y={y} width={Math.max(0, x - ax)} height={8} rx={2} fill="url(#tps-hatch)" stroke="var(--idle)" strokeWidth={0.7} />
              <Span x={x} y={y} w={w} h={8} kind={q.priority === 9 && policy === 'priority' ? 'compute' : 'token'} opacity={0.92} />
              {showStatic && (
                <Span
                  x={x + (q.want / Math.max(...stat.rows.map((z) => z.end - z.start))) * w}
                  y={y}
                  w={w - (q.want / Math.max(...stat.rows.map((z) => z.end - z.start))) * w}
                  h={8}
                  kind="error"
                  opacity={0.8}
                />
              )}
            </g>
          );
        })}

        <Ruler
          x={X0}
          y={232}
          w={X1 - X0}
          ticks={[{ at: 0, label: '0' }, { at: 0.5 }, { at: 1, label: `${scale} steps` }]}
          label="engine steps"
        />

        {/* per-step seat occupancy: the fill-waste meter */}
        <Fade show={s.step >= 1}>
          <text x={X0 - 8} y={256} textAnchor="end" className="v-track-label">Seats</text>
          <rect x={X0} y={244} width={X1 - X0} height={34} rx={2} fill="none" stroke="var(--rule-2)" />
          {run.batchSize.slice(0, 64).map((b, i) => {
            const w = (X1 - X0) / 64;
            const h = Math.max(1.5, (b / slots) * 34);
            return <rect key={i} x={X0 + i * w} y={244 + 34 - h} width={Math.max(1, w - 0.5)} height={h} fill="var(--compute)" opacity={0.6} />;
          })}
          <Label x={X1} y={292} anchor="end" size={9.5} tone={3}>
            {`seats in use · average ${(fill * 100).toFixed(0)}% full${fill < 0.5 ? ' — the batch is mostly empty' : fill > 0.95 ? ' — the batch is saturated' : ''}`}
          </Label>
        </Fade>

        {/* verdict */}
        <Fade show={s.step >= 3}>
          <g>
            <rect x={X0} y={306} width={X1 - X0} height={74} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
            <text x={X0 + 12} y={324} className="v-label" style={{ fontSize: 11, fill: 'var(--text)' }}>
              {showStatic
                ? `Static: ${stat.steps} steps, ${(staticWaste * 100).toFixed(0)}% of the slots paid for produced nothing.`
                : `Continuous: ${run.steps} steps, ${run.useful} tokens, worst wait ${p99} steps.`}
            </text>
            <text x={X0 + 12} y={341} className="v-label" style={{ fontSize: 10.5, fill: 'var(--token)' }}>
              {fill < 0.5
                ? 'Under half the seats are busy. Add seats? No — this is a starved engine; it wants traffic, not capacity.'
                : fill > 0.95 && p99 > 40
                ? 'The batch is full and people are waiting. The step time is your constraint, not the seat count.'
                : 'A balanced batch: seats are busy and nobody waits too long.'}
            </text>
            <text x={X0 + 12} y={358} className="v-label t3" style={{ fontSize: 10 }}>
              Try: set seats to 2 and watch the worst wait. Then set seats to 24 and watch the fill.
            </text>
            <text x={X0 + 12} y={374} className="v-label t3" style={{ fontSize: 10 }}>
              Every one of these numbers comes from the loop in 07.2 — retire, schedule, step.
            </text>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}