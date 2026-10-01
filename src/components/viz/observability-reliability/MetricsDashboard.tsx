import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Span, fmt } from '../lib/svg';
import { series, seriesCount } from './orMetrics';

/** The bucket edges a TTFT histogram should use when the SLO is 600 ms.
 *  The SLO threshold is deliberately one of them, so "missed the SLO" is a count. */
const TTFT_BUCKETS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.2, 2, 3, 5, 10];

// The point of this visual: a service emits millions of events, a dashboard shows five
// numbers, and every one of those five numbers costs storage proportional to how many
// distinct label values can appear. Observability is mostly a budgeting exercise.

const STEPS = [
  'One request emits a handful of numbers',
  'Three kinds of metric, three costs',
  'Panel 1 — rate and goodput',
  'Panel 2 — latency as a histogram',
  'Panel 3 — pressure: who is waiting',
  'Panel 4 — is the engine keeping up',
  'The bill: cardinality',
  'Burn rate is what pages you',
];
const CAPTIONS = [
  'A finished request produces prompt_tokens, output_tokens, TTFT, TPOT and an end-to-end time. That is it. Everything else in the service is bookkeeping.',
  'A counter only ever goes up and you read a rate off it. A gauge goes up and down and you read it directly. A histogram accumulates counts in fixed buckets, and you read quantiles out of it later.',
  'Tokens per second, and goodput next to it. Throughput alone is a vanity number: it rises after the SLO has already broken. The pair is the honest summary from Module 06.',
  'Never store a mean latency. Store a histogram of buckets and ask for the 95th percentile at query time. A mean of 240 ms and a p95 of 1,400 ms describe different populations.',
  'Running and waiting requests are the leading indicator. Latency is the lagging one: by the time p95 moves, you have already been slow for a minute. Watch queue depth first.',
  'Decode step time, KV blocks in use and preemptions. These tell you whether the engine is healthy even when every user-facing number still looks fine.',
  'Each histogram bucket and each label value is a separate stored time series. Eleven buckets times models times replicas times tenants times outcomes is where your monitoring bill comes from.',
  'Turn the SLO into an error budget: how much failure you are allowed, and how fast you are spending it. Burn rate above 1 is the only thing that should page a human.',
];

const EVENTS = [
  { k: 'prompt_tokens', v: '1,204', kind: 'text' as const },
  { k: 'output_tokens', v: '180', kind: 'token' as const },
  { k: 'ttft_seconds', v: '0.412', kind: 'memory' as const },
  { k: 'tpot_seconds', v: '0.0231', kind: 'memory' as const },
  { k: 'slo_met', v: 'true', kind: 'token' as const },
];

const TOKENS = series(40, 3200, 260, 6, 11, 1.6);
const WAITING = series(40, 4, 30, 0.4, 23, 2.2);
const KVB = series(40, 0.55, 0.5, 0.004, 31, 1.1);
const STEPM = series(40, 11.4, 3.6, 0.01, 47, 2.6);

const HIST = [420, 610, 880, 1020, 1160, 980, 640, 350, 180, 70];

const CARDS = [
  { id: 'rate', x: 16, y: 54, title: '1 · RATE', sub: 'rate(vllm:output_tokens_total[1m])' },
  { id: 'lat', x: 328, y: 54, title: '2 · LATENCY', sub: 'histogram_quantile(0.95, …)' },
  { id: 'press', x: 16, y: 146, title: '3 · PRESSURE', sub: 'running / waiting / kv_blocks' },
  { id: 'engine', x: 328, y: 146, title: '4 · ENGINE', sub: 'step_ms / preemptions' },
];

const CW = 296;
const CH = 84;

const LB = { y: 250 };

function Spark({
  data, x, y, w, h, kind, max, ghost,
}: { data: number[]; x: number; y: number; w: number; h: number; kind: 'compute' | 'memory' | 'comm' | 'token'; max?: number; ghost?: boolean }) {
  const m = max ?? Math.max(...data);
  const pts = data.map((v, i) => `${(x + (i / (data.length - 1)) * w).toFixed(1)},${(y + h - (v / m) * h).toFixed(1)}`);
  const area = `M${x},${y + h} L${pts.join(' L')} L${x + w},${y + h} Z`;
  return (
    <g className="v-fade">
      {ghost ? (
        <path d={`M${pts.join(' L')}`} fill="none" stroke="var(--rule-2)" strokeWidth={1.4} strokeDasharray="3 3" />
      ) : (
        <>
          <path d={area} fill={kind === 'token' ? 'var(--token-dim)' : kind === 'memory' ? 'var(--memory-dim)' : kind === 'comm' ? 'var(--comm-dim)' : 'var(--compute-dim)'} />
          <path d={`M${pts.join(' L')}`} fill="none" stroke={`var(--${kind})`} strokeWidth={1.5} />
        </>
      )}
    </g>
  );
}

export default function MetricsDashboard() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [tenants, setTenants] = useState(10);
  useEffect(() => { setTenants(10); }, [s.step]);

  const labels = { model: 2, replica: 8, tenant: Math.max(1, Math.round(tenants)), outcome: 2 };
  const fanout = seriesCount(labels);
  const perMetric = TTFT_BUCKETS.length * fanout;
  const barFrac = Math.max(0.02, Math.min(1, (Math.log10(perMetric) - 1.7) / (Math.log10(2e5) - 1.7)));
  const burn = 2.4;

  const on = (id: string) => CARDS.findIndex((c) => c.id === id);

  return (
    <Stage
      rootRef={ref}
      title="Five numbers on a wall, and the storage bill behind them"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', 'comm', ['error', 'over the budget']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Output rate', value: `${fmt.si(TOKENS[TOKENS.length - 1])} tok/s`, kind: 'token', note: 'counter → rate()' },
          { label: 'p95 TTFT', value: '1.40 s', kind: 'error', note: 'read from buckets, SLO 600 ms' },
          { label: 'Waiting requests', value: fmt.int(Math.round(WAITING[WAITING.length - 1])), kind: Math.round(WAITING[WAITING.length - 1]) > 20 ? 'error' : 'idle', note: 'leading indicator' },
          { label: 'Series per metric', value: fmt.int(perMetric), kind: perMetric > 20000 ? 'error' : 'memory', note: 'buckets × label values' },
        ]} />
      }
      controls={
        <Slider label="Distinct tenants producing traffic" value={tenants} min={1} max={500} step={1} onChange={(v) => setTenants(Math.round(v))} format={(v) => `${Math.round(v)}`} kind="error" />
      }
    >
      <svg viewBox="0 0 640 430" role="img" aria-label="A dashboard: a strip of per-request counters at the top, four metric panels in a two by two grid showing rate, latency histogram, queue pressure and engine health, and below them a cardinality ledger showing how the number of stored time series grows with the number of tenants.">
        {/* ── the events one request emits ── */}
        <Label x={16} y={16} tone={2} size={10.5} bold mono={false}>One finished request, after all the machinery of Modules 04–23 has run</Label>
        {EVENTS.map((e, i) => (
          <g key={e.k}>
            <rect x={16 + i * 122} y={24} width={116} height={22} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x={22 + i * 122} y={38} size={9} kind={e.kind === 'text' ? undefined : e.kind} bold>{e.v}</Label>
            <Label x={22 + i * 122} y={52} size={8} tone={3} mono={false}>{e.k}</Label>
          </g>
        ))}

        {/* ── the four panels ── */}
        {CARDS.map((c, i) => (
          <g key={c.id}>
            <rect x={c.x} y={c.y} width={CW} height={CH} rx={4}
              fill="var(--ground-3)" stroke={s.step > i ? 'var(--rule-2)' : 'var(--rule)'} className="v-cell" />
            <Label x={c.x + 12} y={c.y + 17} size={9.5} bold
              style={{ fill: s.step > i ? 'var(--text)' : 'var(--text-3)' }}>{c.title}</Label>
            <Label x={c.x + 12} y={c.y + CH - 8} size={8} tone={3}>{s.step > i ? c.sub : '— not yet —'}</Label>
          </g>
        ))}

        {/* panel 1 — rate */}
        <Fade show={s.step >= 2}>
          <Spark data={TOKENS} x={28} y={72} w={230} h={44} kind="token" ghost={s.step < 2} />
          <Label x={288} y={92} anchor="end" size={9.5} kind="token" bold>{fmt.si(TOKENS[TOKENS.length - 1])}</Label>
          <Label x={288} y={105} anchor="end" size={8.5} tone={3}>tok/s now</Label>
        </Fade>

        {/* panel 2 — histogram */}
        <Fade show={s.step >= 3}>
          {HIST.map((n, i) => {
            const h = (n / 1300) * 40;
            return <Span key={i} x={340 + i * 14} y={118 - h} w={11} h={h} kind={i >= 7 ? 'error' : 'memory'} dim={i >= 7} />;
          })}
          <line x1={340 + 7.4 * 14} x2={340 + 7.4 * 14} y1={66} y2={120} stroke="var(--error)" strokeWidth={1.4} strokeDasharray="3 3" />
          <Label x={340 + 7.4 * 14} y={64} anchor="middle" size={8.5} kind="error" bold>p95</Label>
          <Label x={340} y={130} size={8} tone={3}>0.1 s</Label>
          <Label x={478} y={130} size={8} tone={3} anchor="end">5 s</Label>
        </Fade>

        {/* panel 3 — pressure */}
        <Fade show={s.step >= 4}>
          <Label x={28} y={168} size={8.5} tone={3}>running</Label>
          <Spark data={KVB} x={72} y={160} w={186} h={18} kind="comm" ghost={s.step < 4} />
          <Label x={28} y={192} size={8.5} tone={3}>waiting</Label>
          <Spark data={WAITING} x={72} y={184} w={186} h={18} kind="compute" ghost={s.step < 4} />
          <Label x={288} y={170} anchor="end" size={9.5} kind="compute" bold>{fmt.int(Math.round(WAITING[WAITING.length - 1]))}</Label>
          <Label x={288} y={196} anchor="end" size={8.5} tone={3}>in the queue</Label>
        </Fade>

        {/* panel 4 — engine */}
        <Fade show={s.step >= 5}>
          <Spark data={STEPM} x={340} y={164} w={186} h={26} kind="memory" ghost={s.step < 5} />
          <Label x={340 + 186} y={172} anchor="end" size={9.5} kind="memory" bold>{STEPM[STEPM.length - 1].toFixed(1)} ms</Label>
          <Label x={340} y={202} size={8.5} tone={3}>decode step · red bars = preemption</Label>
          <Spark data={KVB.map((v) => Math.max(0, v - 0.7) * 22)} x={340} y={206} w={186} h={10} kind="error" ghost={s.step < 5} />
        </Fade>

        {/* ── the cardinality ledger ── */}
        <line x1={16} y1={LB.y} x2={624} y2={LB.y} className="v-rule faint" />
        <Fade show={s.step >= 6}>
          <Label x={16} y={LB.y + 20} tone={2} size={10.5} bold mono={false}>What that costs to store</Label>
          <Label x={624} y={LB.y + 20} anchor="end" size={9} tone={3}>one histogram metric = buckets × every label value</Label>

          <Label x={16} y={LB.y + 42} size={9.5} tone={2}>{`${TTFT_BUCKETS.length} buckets × ${labels.model} models × ${labels.replica} replicas × ${labels.tenant} tenants × ${labels.outcome} outcomes`}</Label>
          <rect x={16} y={LB.y + 50} width={520} height={16} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={16} y={LB.y + 50} width={520 * barFrac} height={16} rx={2}
            fill={perMetric > 20000 ? 'var(--error)' : 'var(--memory)'} className="v-cell" />
          <Label x={544} y={LB.y + 62} size={10} kind={perMetric > 20000 ? 'error' : 'memory'} bold>{`${fmt.int(perMetric)} series`}</Label>
          <Label x={16} y={LB.y + 84} size={9.5} kind={perMetric > 20000 ? 'error' : undefined}>
            {perMetric > 20000
              ? 'Too many. Drop a label, coarsen the buckets, or sample.'
              : 'Still a budget. Every label value you add multiplies every bucket.'}
          </Label>
          <Label x={16} y={LB.y + 100} size={9} tone={3}>The fix is never “add a label”. It is “which one question does this answer?”</Label>
        </Fade>

        {/* ── burn rate ── */}
        <Fade show={s.step >= 7}>
          <Label x={16} y={LB.y + 128} tone={2} size={10.5} bold mono={false}>The error budget, and the burn rate</Label>
          <Label x={624} y={LB.y + 128} anchor="end" size={9} tone={3}>SLO p95 TTFT &lt; 600 ms, 30-day window → budget = 0.5% of requests may fail</Label>
          <rect x={16} y={LB.y + 136} width={520} height={16} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={16} y={LB.y + 136} width={520 * (burn / 3)} height={16} rx={2} fill="var(--error)" className="v-cell" />
          <line x1={16 + (520 / 3) * 1} x2={16 + (520 / 3) * 1} y1={LB.y + 132} y2={LB.y + 156} stroke="var(--text-2)" strokeWidth={1.4} />
          <Label x={16} y={LB.y + 170} size={9.5} kind="error" bold>{`burn rate ${burn.toFixed(1)}× — the budget for a month is gone in ${(30 / burn).toFixed(0)} days`}</Label>
          <Label x={16} y={LB.y + 186} size={9} tone={3}>Alert on burn rate over a long window and on the symptom over a short one. Never on a single percentile.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}