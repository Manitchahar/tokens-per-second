import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Fade, Move, fmt } from '../lib/svg';

/**
 * One replica's whole life: starting, warming, serving, draining, gone.
 * The point of the visual is the drain — how many requests are still in flight
 * when the terminate deadline lands, and which ones get lost.
 */

type Phase = 'starting' | 'ready' | 'draining' | 'gone';

const STEPS = [
  'A replica is created',
  'Readiness is false',
  'Warm-up finishes, readiness flips',
  'Serving: new work arrives',
  'SIGTERM arrives',
  'Readiness flips false, drain begins',
  'The drain deadline lands',
  'Capacity over the whole life',
];

const CAPTIONS = [
  'The platform creates a replica. Until it passes readiness, the load balancer sends it nothing.',
  'Startup and warm-up: imports, weights, graph capture, autotune, warm traffic. The probe must be failing for all of this, or a request lands on a replica that cannot answer.',
  'Readiness flips only when the replica is honest. After this single instant, everything else is steady state.',
  'Steady state: the replica takes new work. This is the only phase that produces user value.',
  'The platform sends SIGTERM — a deploy, a scale-down, a node drain. From this instant the replica has a deadline, and the deadline does not negotiate.',
  'The replica stops passing readiness and keeps finishing what it has. New requests go to the fleet; in-flight ones stay here.',
  'When the deadline expires the process is killed. Anything still generating at that moment is truncated — the user sees half an answer.',
  'Add up the life: most of the wall clock is startup and drain, but only the middle band is serving. Everything else is capacity you are paying for and not using.',
];

const W = 640;

// Wall-clock geometry
const TX = 40;
const TW = 560;
const TY = 84;
const TH = 40;

// Request rows below the band
const ROW_Y = 182;
const ROW_H = 18;
const ROWS = 6;

export default function ReplicaLifecycle() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [coldStart, setColdStart] = useState(240);
  const [drainBudget, setDrainBudget] = useState(60);
  const [outLen, setOutLen] = useState(400);
  const [stepMs, setStepMs] = useState(30);

  // A single generation takes outLen * stepMs milliseconds of decode.
  const genMs = outLen * stepMs;
  // Drain must outlast the longest request that was already running.
  const needed = genMs / 1000;
  const drainFits = drainBudget >= needed;

  // Timeline scale
  const total = coldStart + drainBudget;
  const px = (t: number) => (t / total) * TW;

  // Requests: when each started (t seconds from creation) and how long it runs.
  // Five arrive during serving; the sixth arrives right at SIGTERM.
  const arrivals = [0.15, 0.3, 0.42, 0.5, 0.62, 1.0].map((f) => f * coldStart);
  const requests = arrivals.map((t0, i) => ({
    i,
    t0,
    t1: i === arrivals.length - 1 ? t0 + genMs / 1000 : Math.min(total, t0 + genMs / 1000),
  }));

  const truncated = requests.filter((q) => q.t1 > coldStart + drainBudget);

  // phase boundaries in seconds
  const sigterm = coldStart;
  const dead = coldStart + drainBudget;

  const ready = s.step >= 2;
  const serving = s.step >= 3 && s.step <= 4;
  const draining = s.step >= 5 && s.step <= 6;
  const gone = s.step >= 7;

  const lostCount = truncated.length;

  return (
    <Stage
      rootRef={ref}
      title="One replica: starting, warm, serving, draining, gone"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['idle', 'starting / warming'],
        ['token', 'serving'],
        ['comm', 'draining'],
        ['error', 'request truncated'],
        ['text', 'not taking new work'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Startup', value: `${coldStart} s`, kind: 'idle', note: 'paid every scale-up' },
          { label: 'Drain budget', value: `${drainBudget} s`, kind: drainFits ? 'comm' : 'error', note: drainFits ? 'fits the longest request' : `${needed.toFixed(0)} s needed` },
          { label: 'Longest request', value: `${(genMs / 1000).toFixed(0)} s`, kind: 'token', note: `${outLen} tokens at ${stepMs} ms/tok` },
          { label: 'Truncated', value: `${lostCount} of ${requests.length}`, kind: lostCount ? 'error' : 'token', note: lostCount ? 'half an answer each' : 'nothing lost' },
        ]} />
      }
      controls={
        <>
          <Slider label="Startup (cold start)" value={coldStart} min={30} max={600} step={10} onChange={setColdStart} format={(v) => `${v} s`} kind="idle" />
          <Slider label="Drain timeout" value={drainBudget} min={5} max={300} step={5} onChange={setDrainBudget} format={(v) => `${v} s`} kind="comm" />
          <Slider label="Output length" value={outLen} min={50} max={2000} step={10} onChange={setOutLen} format={(v) => `${v} tok`} kind="token" />
          <Slider label="Time per token" value={stepMs} min={5} max={80} onChange={setStepMs} format={(v) => `${v} ms`} kind="compute" />
          <Segmented
            label="Termination signal"
            value="sigterm"
            options={[{ value: 'sigterm', label: 'SIGTERM with a deadline' }]}
            onChange={() => {}}
          />
        </>
      }
    >
      <svg viewBox={`0 0 ${W} 352`} role="img" aria-label="A timeline of one serving replica: a startup band, a serving band, and a drain band, with request rows below showing six requests arriving and generating, and any request still generating when the drain deadline lands marked as truncated in red.">
        <Label x={TX} y={30} tone={2} size={11} mono={false} bold>{'Wall clock of one replica — only the middle band produces value'}</Label>
        <Label x={TX} y={46} tone={3} size={9}>{'the drain timeout must outlast the longest request already running, or every deploy truncates the longest generations'}</Label>

        {/* axis */}
        <line x1={TX} x2={TX + TW} y1={TY - 14} y2={TY - 14} className="v-rule" />
        <Label x={TX} y={TY - 20} tone={3} size={9.5}>{'created'}</Label>
        <Label x={TX + TW} y={TY - 20} anchor="end" tone={3} size={9.5}>{'process gone'}</Label>

        {/* bands */}
        <Fade show={s.step >= 0}>
          <Span x={TX} y={TY} w={px(coldStart)} h={TH} kind="idle" label={`startup ${coldStart}s`} />
        </Fade>
        <Fade show={ready}>
          <Span x={TX + px(coldStart)} y={TY} w={px(drainBudget)} h={TH} kind={draining ? 'comm' : 'token'} label={draining ? `drain ${drainBudget}s` : `serving ${drainBudget}s`} />
        </Fade>
        <Fade show={gone}>
          <Label x={TX + px(total)} y={TY + 26} anchor="end" tone={1} size={10} bold kind="error">{`exit — ${truncated.length} truncated`}</Label>
        </Fade>

        {/* markers */}
        <Fade show={ready}>
          <g>
            <Arrow x1={TX + px(coldStart)} y1={TY - 12} x2={TX + px(coldStart)} y2={TY - 2} kind="token" />
            <Label x={TX + px(coldStart) + 5} y={TY - 16} tone={1} size={9.5} bold kind="token">{`ready`}</Label>
          </g>
        </Fade>
        <Fade show={s.step >= 4}>
          <g>
            <line x1={TX + px(sigterm)} x2={TX + px(sigterm)} y1={TY + TH} y2={ROW_Y + ROWS * ROW_H + 16} stroke="var(--error)" strokeWidth={1.2} strokeDasharray="4 4" />
            <Label x={TX + px(sigterm) - 5} y={TY + TH + 14} anchor="end" tone={1} size={9.5} bold kind="error">{`SIGTERM`}</Label>
          </g>
        </Fade>
        <Fade show={draining}>
          <g>
            <line x1={TX + px(dead)} x2={TX + px(dead)} y1={TY} y2={ROW_Y + ROWS * ROW_H + 16} stroke="var(--error)" strokeWidth={1.6} />
            <Label x={TX + px(dead) + 5} y={TY - 16} tone={1} size={9.5} bold kind="error">{`kill at +${drainBudget}s`}</Label>
          </g>
        </Fade>

        {/* requests */}
        <Label x={TX} y={ROW_Y - 12} tone={3} size={9.5}>{'requests on this replica'}</Label>
        {requests.map((q, i) => {
          const x0 = TX + px(q.t0);
          const x1 = TX + px(q.t1);
          const y = ROW_Y + i * ROW_H;
          const cut = q.t1 > dead + 1e-9;
          const shown = s.step >= (q.i === 5 ? 4 : 3);
          const kind = cut ? 'error' : q.t0 >= sigterm ? 'comm' : 'token';
          return (
            <g key={q.i} style={{ opacity: shown ? 1 : 0.12 }}>
              <Label x={TX - 8} y={y + 14} anchor="end" tone={3} size={9}>{`r${i + 1}`}</Label>
              <Move x={x0} y={y}>
                <Span x={0} y={0} w={Math.max(2, x1 - x0)} h={14} kind={kind} />
              </Move>
              {cut && shown && (
                <Label x={Math.min(x1, TX + TW) - 4} y={y + 11} anchor="end" tone={1} size={9} bold kind="error">{`truncated`}</Label>
              )}
            </g>
          );
        })}

        {/* the verdict */}
        <Fade show={s.step >= 6}>
          <g>
            <line x1={TX} x2={TX + TW} y1={ROW_Y + ROWS * ROW_H + 18} y2={ROW_Y + ROWS * ROW_H + 18} className="v-rule faint" />
            <Label x={TX} y={ROW_Y + ROWS * ROW_H + 34} tone={1} size={10} bold kind={lostCount ? 'error' : 'token'}>
              {lostCount
                ? `${lostCount} request${lostCount > 1 ? 's' : ''} still generating at the deadline — the user sees half an answer, with no error.`
                : 'Every request finished inside the drain budget. Nothing truncated.'}
            </Label>
            <Label x={TX} y={ROW_Y + ROWS * ROW_H + 50} tone={2} size={9.5}>
              {lostCount
                ? `Drain timeout must be ≥ ${needed.toFixed(0)} s (longest request), or stop accepting long generations.`
                : `Drain ${drainBudget} s ≥ longest request ${needed.toFixed(0)} s. This deploy is safe for these workloads.`}
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}