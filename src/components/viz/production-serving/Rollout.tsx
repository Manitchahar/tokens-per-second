import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Fade, fmt } from '../lib/svg';

/**
 * Rollout strategies over the same fleet of N replicas. Each strategy is drawn
 * as a lane of replica cells over time, so you can see exactly when capacity
 * dips, when it doubles, and how long the old version keeps serving.
 */

type Strategy = 'rolling' | 'canary' | 'shadow' | 'bluegreen';

const STEPS = [
  'One fleet, four ways to replace it',
  'Rolling update',
  'Rolling: capacity holds',
  'Canary: a slice of traffic',
  'Canary: the evidence',
  'Shadow: run it, discard it',
  'Shadow: what it costs',
  'Blue/green: switch it all',
  'Choose on the arithmetic',
];

const CAPTIONS = [
  'The same eight replicas, four strategies. Capacity, exposure to a bad version, and rollback time are the three things they trade.',
  'Rolling update: bring up a few new replicas, wait for readiness, retire the same number of old ones, repeat. Capacity stays roughly flat the whole way.',
  'With maxSurge 2 and maxUnavailable 1 the fleet never drops below the desired count. Those two numbers are the entire safety model of a rolling update.',
  'Canary: a small, real slice of production traffic goes to the new version. Small enough that a regression costs little, large enough to measure.',
  'The canary only earns its keep if you decide against a named metric. A canary you never fail is a canary you never read.',
  'Shadow traffic mirrors live requests to the new version and throws the response away. Real inputs, real load, zero user exposure.',
  'The cost: you serve every request twice for the duration. On a GPU-bound fleet, doubling served work doubles your bill for the observation window.',
  'Blue/green runs two full fleets and moves traffic with a switch. Rollback is instantaneous — and you have been paying for double capacity the whole time.',
  'The three real criteria: how long rollback must take, whether you can afford double capacity, and how much traffic you need to detect a regression.',
];

const W = 640;

// Fleet geometry
const FX = 96;
const FW = 470;
const FY = 128;
const FH = 34;
const N = 8;

// Strategy cells: for each of the N replica slots, when it is old, new, or absent.
function cells(strategy: Strategy, t: number): ('old' | 'new' | 'idle')[] {
  const out: ('old' | 'new' | 'idle')[] = [];
  if (strategy === 'bluegreen') {
    // 0.0-0.2 nothing changes, 0.2-0.4 warming the green fleet, 0.4+ all green.
    if (t < 0.4) return Array(N).fill('old');
    return Array(N).fill('new');
  }
  if (strategy === 'shadow') {
    // Old fleet serves all the way; the new fleet exists in parallel (shadowed).
    if (t < 0.35) return Array(N).fill('old');
    if (t < 0.75) return Array.from({ length: N }, (_, i) => (i < 3 ? 'new' : 'old'));
    return Array(N).fill('new');
  }
  if (strategy === 'canary') {
    // Two replicas get the new version early and soak before the rest follow.
    if (t < 0.2) return Array(N).fill('old');
    if (t < 0.55) return Array.from({ length: N }, (_, i) => (i < 2 ? 'new' : 'old'));
    return Array.from({ length: N }, (_, i) => (i < 6 ? 'new' : 'old'));
  }
  // rolling: wave by wave, 3 at a time
  if (t < 0.22) return Array(N).fill('old');
  if (t < 0.55) return Array.from({ length: N }, (_, i) => (i < 3 ? 'new' : 'old'));
  if (t < 0.85) return Array.from({ length: N }, (_, i) => (i < 6 ? 'new' : 'old'));
  return Array(N).fill('new');
}

const STRATEGIES: { id: Strategy; name: string; blurb: string; rollback: string }[] = [
  { id: 'rolling', name: 'Rolling', blurb: 'Replace a few at a time. Capacity flat, no double cost.', rollback: 'minutes' },
  { id: 'canary', name: 'Canary', blurb: 'A small real slice first, decide on a named metric.', rollback: 'seconds' },
  { id: 'shadow', name: 'Shadow', blurb: 'Mirror traffic, discard output. Evidence at full cost.', rollback: 'instant' },
  { id: 'bluegreen', name: 'Blue/green', blurb: 'Two full fleets, one switch. Instant, always double.', rollback: 'instant' },
];

export default function Rollout() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [strategy, setStrategy] = useState<Strategy>('rolling');
  const [n, setN] = useState(8);
  const [shards, setShards] = useState(3);
  const [canaryPct, setCanaryPct] = useState(5);

  const cellW = FW / n;
  const cellX = (i: number) => FX + i * cellW;
  const c = cells(strategy, 0.5);

  // Stage the chosen strategy; other strategies preview only once revealed.
  const reveal: Record<Strategy, number> = { rolling: 1, canary: 3, shadow: 5, bluegreen: 7 };
  const shown = (id: Strategy) => s.step >= reveal[id];

  const surge = strategy === 'rolling' ? shards : strategy === 'bluegreen' ? n : strategy === 'shadow' ? Math.ceil(n * 0.4) : Math.max(1, Math.round((n * canaryPct) / 100));

  return (
    <Stage
      rootRef={ref}
      title="Four ways to replace a fleet without dropping it"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['text', 'old version, serving'],
        ['token', 'new version, serving'],
        ['comm', 'warming up'],
        ['idle', 'not yet up'],
        ['error', 'capacity below desired'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Fleet size', value: `${n} replicas`, kind: 'text', note: 'each with its own GPUs' },
          { label: 'Peak cost', value: strategy === 'bluegreen' || strategy === 'shadow' ? `${n + surge} replicas` : `${n + surge} replicas`, kind: strategy === 'bluegreen' ? 'error' : 'comm', note: strategy === 'bluegreen' ? '2× forever' : strategy === 'shadow' ? '2× during shadow' : 'transient' },
          { label: 'User exposure', value: strategy === 'shadow' ? 'none' : strategy === 'canary' ? `${canaryPct}% of traffic` : 'all, gradually', kind: strategy === 'shadow' ? 'token' : 'comm' },
          { label: 'Rollback', value: STRATEGIES.find((x) => x.id === strategy)!.rollback, kind: 'token', note: 'time to restore old' },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Strategy"
            value={strategy}
            options={STRATEGIES.map((x) => ({ value: x.id, label: x.name }))}
            onChange={setStrategy}
          />
          <Slider label="Replicas" value={n} min={4} max={16} onChange={setN} format={(v) => `${v}`} kind="text" />
          <Slider label="Rolling shard size" value={shards} min={1} max={8} onChange={setShards} format={(v) => `${v} at a time`} kind="comm" />
          <Slider label="Canary slice" value={canaryPct} min={1} max={50} onChange={setCanaryPct} format={(v) => `${v}%`} kind="token" />
        </>
      }
    >
      <svg viewBox={`0 0 ${W} 300`} role="img" aria-label="A grid of replica cells coloured by version across four rollout strategies — rolling, canary, shadow and blue/green — showing when old and new replicas coexist and when capacity doubles.">
        {/* chosen strategy fleet */}
        <Label x={FX} y={78} tone={1} size={11.5} bold>{`${STRATEGIES.find((x) => x.id === strategy)!.name} — ${STRATEGIES.find((x) => x.id === strategy)!.blurb}`}</Label>

        {/* desired-capacity line */}
        <g>
          {Array.from({ length: n }, (_, i) => (
            <g key={i}>
              <rect
                x={cellX(i) + 2}
                y={FY}
                width={cellW - 4}
                height={FH}
                rx={2}
                className="v-cell"
                fill={c[i] === 'old' ? 'var(--ground-3)' : c[i] === 'new' ? 'var(--token-dim)' : 'none'}
                stroke={c[i] === 'new' ? 'var(--token)' : 'var(--rule-2)'}
              />
              {c[i] === 'new' && <Label x={cellX(i) + cellW / 2} y={FY + FH / 2 + 4} anchor="middle" tone={1} size={9.5} bold kind="token">{'new'}</Label>}
              {c[i] === 'old' && <Label x={cellX(i) + cellW / 2} y={FY + FH / 2 + 4} anchor="middle" tone={2} size={9.5}>{'old'}</Label>}
            </g>
          ))}
        </g>

        <Label x={FX - 8} y={FY + FH / 2 + 4} anchor="end" tone={3} size={9.5}>{'fleet'}</Label>

        {/* the other three, as smaller previews */}
        <Label x={FX} y={208} tone={3} size={9.5}>{'the other strategies at a glance (same fleet size)'}</Label>
        {STRATEGIES.filter((x) => x.id !== strategy).map((x, r) => {
          const cc = cells(x.id, 0.5);
          const y = 220 + r * 22;
          return (
            <g key={x.id} style={{ opacity: shown(x.id) ? 0.35 : 0.12 }}>
              <Label x={FX + 78} y={y + 10} anchor="end" tone={3} size={9}>{x.name}</Label>
              {Array.from({ length: Math.min(n, 12) }, (_, i) => (
                <rect
                  key={i}
                  x={FX + 86 + i * ((FW - 100) / Math.min(n, 12))}
                  y={y}
                  width={(FW - 100) / Math.min(n, 12) - 3}
                  height={13}
                  rx={1.5}
                  fill={cc[i] === 'new' ? 'var(--token-dim)' : cc[i] === 'old' ? 'var(--ground-3)' : 'none'}
                  stroke={cc[i] === 'new' ? 'var(--token)' : 'var(--rule-2)'}
                />
              ))}
            </g>
          );
        })}

        {/* exposure caption */}
        <Fade show={s.step >= 2}>
          <g>
            <Label x={FX} y={96} tone={2} size={10}>
              {strategy === 'bluegreen' || strategy === 'shadow'
                ? `peak: ${n + surge} replicas running at once — plan your budget for it`
                : `peak: ${n + surge} replicas briefly, capacity never dips below ${n}`}
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}