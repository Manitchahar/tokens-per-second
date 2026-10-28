import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Span, Arrow, fmt } from '../lib/svg';
import { attemptFactor, correlatedFactor, layeredFactor, retryBudget, timeoutSplit } from './orMetrics';

// Failure is arithmetic before it is an incident. This visual is the whole lesson:
// how long to wait, how many times to retry, and what amplification costs you.

const STEPS = [
  'A request with a deadline',
  'Timeouts are the only bound',
  'Splitting a budget across hops',
  'Retries, and what they cost',
  'The retry storm',
  'Independent vs correlated failure',
  'The retry budget',
  'Four layers of retries',
  'What to write in the runbook',
];
const CAPTIONS = [
  'A user with a 2-second budget. Everything downstream inherits the deadline, and every layer that can fail needs a timeout shorter than what is left of it.',
  'Without a timeout, a hung downstream does not fail fast — it occupies a slot forever. A timeout is not about speed; it is about making failure observable and bounded.',
  'Five hops sharing a 2-second budget with a 20% safety margin gives 320 ms each. If any hop is allowed the full 2 s, the request can never fail before its own deadline.',
  'A retry is free if the failure is rare and independent. It is not free in latency, in load, or in the slot it holds while waiting.',
  'When a downstream dies, its failures are correlated — near 100%. Every client then retries at once, multiplying the offered load on a system that is already down.',
  'Independent 5% failures with 2 retries cost 1.05× load. The same 2 retries against a dead backend cost 3× load. The retry policy is only safe under the first assumption.',
  'The standard fix: a client may spend at most a small multiple of its request count on retries. 100 requests with a ratio of 1.5 permits 50 retries — amplification is bounded by construction.',
  'Retries at four layers do not add; they multiply. Three retries at each of four layers is 4⁴ = 256× offered load. This is the classic cause of a self-inflicted outage.',
  'Five numbers, in a runbook, that decide the incident: the total deadline, the per-hop timeout, the max attempts, the retry budget ratio, and whether failures are independent.',
];

const BUDGET = 2.0;
const HOPS = 5;
const MARGIN = 0.8;

// layered chain: edge → gateway → engine → backend
const CHAIN = [
  { name: 'client', kind: 'text' as const, depth: 0 },
  { name: 'edge gateway', kind: 'comm' as const, depth: 1 },
  { name: 'router', kind: 'memory' as const, depth: 1 },
  { name: 'engine', kind: 'compute' as const, depth: 1 },
  { name: 'model backend', kind: 'memory' as const, depth: 1 },
];

const X0 = 150;
const CHAIN_W = 76;

export default function FailureBudget() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [pFail, setPFail] = useState(0.05);
  const [retries, setRetries] = useState(2);

  const split = timeoutSplit(BUDGET, HOPS, MARGIN);
  const indep = attemptFactor(pFail, retries);
  const corr = correlatedFactor(retries);
  const layered = layeredFactor(retries, 4);

  const budgetOk = retryBudget(100, 1.5);

  const X1 = X0 + 4 * (CHAIN_W + 18);
  const toX = (i: number) => X0 + i * (CHAIN_W + 18);

  const ampFrac = Math.min(1, (corr - 1) / 7);
  const stormRows = useMemo(() => {
    const rows = [];
    let load = 100;
    for (let i = 0; i <= retries; i++) {
      rows.push({ attempt: i + 1, load });
      load = load * corr;
      if (load > 5000) break;
    }
    return rows;
  }, [retries]);

  const showSplit = s.step >= 1;
  const showRetries = s.step >= 3;
  const showStorm = s.step >= 4;
  const showCorr = s.step >= 5;
  const showBudget = s.step >= 6;
  const showLayered = s.step >= 7;

  return (
    <Stage
      rootRef={ref}
      title="Deadlines, timeouts and the arithmetic of trying again"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'comm', ['token', 'within budget'], ['error', 'over budget'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Deadline', value: `${BUDGET.toFixed(2)} s`, kind: 'compute', note: 'from the user, not from you' },
          { label: 'Per-hop timeout', value: `${(split.per * 1000).toFixed(0)} ms`, kind: 'memory', note: `${HOPS} hops, ${(MARGIN * 100).toFixed(0)}% margin` },
          { label: 'Load factor (independent)', value: `${indep.toFixed(2)}×`, kind: indep < 1.5 ? 'token' : 'error', note: `p=${(pFail * 100).toFixed(0)}%, ${retries} retries` },
          { label: 'Load factor (correlated)', value: `${corr.toFixed(0)}×`, kind: 'error', note: 'backend down: every attempt fails' },
        ]} />
      }
      controls={
        <>
          <Slider label="Independent failure probability per attempt" value={Math.round(pFail * 100)} min={1} max={100} step={1} onChange={(v) => setPFail(v / 100)} format={(v) => `${Math.round(v)}%`} kind="memory" />
          <Slider label="Maximum retries" value={retries} min={0} max={5} step={1} onChange={(v) => setRetries(Math.round(v))} format={(v) => `${Math.round(v)}`} kind="error" />
        </>
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A chain of four dependency hops sharing a two-second request budget, with a per-hop timeout, then a retry amplification comparison between independent and correlated failures, and a layered retry multiplication diagram.">
        {/* ── the deadline ── */}
        <Label x={16} y={18} tone={1} size={11} bold mono={false}>one user request, one deadline</Label>
        <Label x={624} y={18} anchor="end" size={9.5} kind="token" bold>{`${BUDGET.toFixed(1)} s end-to-end`}</Label>
        <line x1={X0} x2={624} y1={34} y2={34} className="v-rule" />
        <line x1={X0} x2={624} y1={38} y2={38} stroke="var(--token)" strokeWidth={2} />
        <Label x={X0} y={50} size={9} kind="token">0 ms</Label>
        <Label x={624} y={50} anchor="end" size={9} kind="error" bold>{`${BUDGET.toFixed(1)} s — the deadline`}</Label>

        {/* ── the hop chain ── */}
        <Fade show={showSplit}>
          {CHAIN.map((h, i) => (
            <g key={h.name}>
              {i > 0 && <Arrow x1={toX(i - 1) + CHAIN_W + 2} y1={96} x2={toX(i) - 4} y2={96} kind={h.kind} muted={h.depth === 0} />}
              <rect x={toX(i)} y={76} width={CHAIN_W} height={40} rx={3}
                fill="var(--ground-3)" stroke={h.depth ? 'var(--rule-2)' : 'var(--token)'} strokeWidth={1} className="v-cell" />
              <Label x={toX(i) + CHAIN_W / 2} y={94} anchor="middle" size={9}>{h.name}</Label>
              <Label x={toX(i) + CHAIN_W / 2} y={107} anchor="middle" size={8} tone={3}>{h.depth ? 'timeout' : 'no timeout'}</Label>
            </g>
          ))}
          <Label x={X0} y={64} size={9.5} tone={2} bold mono={false}>Each hop gets its slice of what is left</Label>
          <Label x={624} y={64} anchor="end" size={9.5} kind="memory" bold>{`${(split.per * 1000).toFixed(0)} ms per hop · ${((1 - MARGIN) * 100).toFixed(0)}% margin kept`}</Label>
          <Label x={X0} y={134} size={9} tone={3}>
            Give any hop the full 2 s and the request cannot fail before its own deadline — the deadline becomes decorative.
          </Label>
        </Fade>

        {/* ── retry amplification ── */}
        <line x1={16} y1={148} x2={624} y2={148} className="v-rule faint" />
        <Fade show={showRetries}>
          <Label x={16} y={168} tone={2} size={10.5} bold mono={false}>What {retries} retries cost you in offered load</Label>

          <Label x={16} y={192} size={9} tone={3}>independent</Label>
          <rect x={96} y={182} width={460} height={14} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={96} y={182} w={Math.min(460, 460 * (indep / 8))} h={14} kind={indep < 1.5 ? 'token' : 'error'} />
          <Label x={566} y={193} size={9.5} kind={indep < 1.5 ? 'token' : 'error'} bold>{`${indep.toFixed(2)}×`}</Label>

          {showCorr && (
            <>
              <Label x={16} y={216} size={9} tone={3}>correlated</Label>
              <rect x={96} y={206} width={460} height={14} rx={2} fill="none" stroke="var(--rule)" />
              <Span x={96} y={206} w={460 * ampFrac} h={14} kind="error" />
              <Label x={566} y={217} size={9.5} kind="error" bold>{`${corr.toFixed(0)}×`}</Label>
              <Label x={16} y={238} size={9} tone={3} kind="error">
                {`Same retry count, same code path — ${(corr / indep).toFixed(0)}× more load, because every attempt fails when the backend is down.`}
              </Label>
            </>
          )}
        </Fade>

        {/* ── the storm ── */}
        <Fade show={showStorm}>
          <Label x={16} y={262} tone={2} size={10.5} bold mono={false}>The storm, from one client sending 100 requests</Label>
          {stormRows.map((row, i) => (
            <g key={row.attempt}>
              <Label x={16} y={280 + i * 18} size={9} tone={3}>{`attempt ${row.attempt}`}</Label>
              <rect x={96} y={271 + i * 18} width={280} height={13} rx={2} fill="none" stroke="var(--rule)" />
              <Span x={96} y={271 + i * 18} w={Math.min(280, 280 * Math.log10(1 + row.load) / Math.log10(1 + 5000))} h={13} kind={i === 0 ? 'compute' : 'error'} />
              <Label x={388} y={281 + i * 18} size={9.5} kind={i === 0 ? 'compute' : 'error'}>{fmt.int(row.load)} calls in flight</Label>
            </g>
          ))}
        </Fade>

        {/* ── the budget ── */}
        <Fade show={showBudget}>
          <Label x={430} y={262} tone={2} size={10.5} bold mono={false}>The retry budget</Label>
          <Label x={430} y={280} size={9} tone={3}>{`ratio 1.5 on 100 requests`}</Label>
          <rect x={430} y={288} width={180} height={14} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={430} y={288} w={180 * (budgetOk / 100)} h={14} kind="token" />
          <Label x={430} y={318} size={9.5} kind="token" bold>{`${budgetOk} retries permitted`}</Label>
          <Label x={430} y={332} size={9} tone={3}>Amplification is bounded by construction,</Label>
          <Label x={430} y={344} size={9} tone={3}>and 50 of the 100 can still succeed.</Label>
        </Fade>

        {/* ── layered retries ── */}
        <Fade show={showLayered}>
          <Label x={16} y={362} tone={2} size={10.5} bold mono={false}>Retries at every layer multiply</Label>
          <Label x={230} y={362} size={9.5} tone={2} mono={false}>4 layers × {retries + 1} attempts</Label>
          <Arrow x1={330} y1={358} x2={356} y2={358} kind="error" />
          <Label x={366} y={362} size={11} kind="error" bold>{`${layered.toFixed(0)}× offered load — your own client is the outage`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}