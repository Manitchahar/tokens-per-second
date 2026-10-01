import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Arrow } from '../lib/svg';
import { TRACE_SPANS, CRITICAL_PATH, type Span } from './orMetrics';

// A trace is a set of spans that nest. The waterfall you read on screen is a
// projection of that tree onto time — and the moment you notice is the moment you
// realize one of those spans is unattributed.

const STEPS = [
  'One request, eleven spans',
  'The waterfall',
  'Spans nest; the picture flattens them',
  'The critical path',
  'Read it top down',
  'Queue wait is usually the answer',
  'The gap that has no span',
  'Sampling, and what you lose',
];
const CAPTIONS = [
  'Every interesting moment in a request gets a span: a name, a service, a start and a duration. These eleven spans are everything that happened to one streamed chat completion.',
  'Plotted as horizontal bars on a shared time axis, the spans form a waterfall. Width is duration, and every bar sits exactly where in the request it happened.',
  'A span may have a parent. Detokenize belongs to the edge request; the KV transfer belongs to the engine. The waterfall shows you the timing but not the tree — so keep the parent link, and query it.',
  'The critical path is the chain of spans with no gap: edge, queue, prefill, decode, egress. Its total is the latency. Anything off the path is interesting, and anything ON it is the latency.',
  'Read from the top: the edge work is 50 ms and is constant, so it bounds your floor no matter how fast the GPU is. Then the queue. Then prefill, which is compute-bound.',
  'Here 58 ms of 338 ms is spent waiting for a scheduler slot. That is 17% of the latency, and it is not a kernel problem — it disappears the moment you add capacity or a waiting-request cap.',
  'The grey bar at the end has no span covering it. Either something uninstrumented took 340 ms, or telemetry itself failed. Both are true outages from the user’s point of view.',
  'At high volume you cannot store every trace. Head sampling decides at the door and keeps an unbiased sample — which means it often throws away exactly the slow request you needed. Tail sampling decides after the outcome is known.',
];

const SERVICES = [
  { name: 'edge', x: 24, w: 96, kind: 'comm' as const },
  { name: 'router', x: 128, w: 96, kind: 'memory' as const },
  { name: 'prefill', x: 232, w: 96, kind: 'compute' as const },
  { name: 'decode', x: 336, w: 104, kind: 'memory' as const },
  { name: 'engine', x: 448, w: 96, kind: 'comm' as const },
];

const X0 = 132;
const X1 = 620;
const TOTAL = 678; // ms of wall clock including the unattributed gap
const sx = (ms: number) => X0 + (ms / TOTAL) * (X1 - X0);

const ORDER = ['edge', 'router', 'prefill', 'decode', 'engine'];

const ROW_H = 22;
const LANE_Y = 118;

const KIND_COLOR: Record<Span['kind'], string> = {
  queue: 'var(--idle)',
  compute: 'var(--compute)',
  memory: 'var(--memory)',
  comm: 'var(--comm)',
  token: 'var(--token)',
};

export default function TraceWaterfall() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [budget, setBudget] = useState(600);

  const rows = ORDER.flatMap((svc) => TRACE_SPANS.filter((sp) => sp.service === svc).map((sp) => ({ svc, sp })));
  const total = 338;
  const overBudget = total > budget;

  const showAll = s.step >= 1;
  const showPath = s.step >= 3;
  const showQueue = s.step >= 5;
  const showGap = s.step >= 6;
  const showSample = s.step >= 7;

  return (
    <Stage
      rootRef={ref}
      title="One request, eleven spans, and 338 ms of wall clock"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['comm', 'network'], 'token', ['idle', 'waiting']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Wall clock', value: `${total} ms`, kind: 'compute', note: 'sum on the critical path' },
          { label: 'Queue wait', value: '58 ms', kind: 'idle', note: `${((58 / total) * 100).toFixed(0)}% of the latency` },
          { label: 'Decode', value: '191 ms', kind: 'memory', note: '181 tokens × ~1.06 ms' },
          { label: 'Against budget', value: overBudget ? `+${total - budget} ms over` : `${budget - total} ms spare`, kind: overBudget ? 'error' : 'token', note: `budget ${budget} ms` },
        ]} />
      }
      controls={
        <Slider label="End-to-end latency budget" value={budget} min={200} max={1200} step={10} onChange={setBudget} format={(v) => `${Math.round(v)} ms`} kind="token" />
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A distributed trace waterfall: eleven spans for one request across five services on a shared time axis, with the critical path highlighted, a long idle queue-wait bar, and a grey unattributed gap at the end.">
        <Label x={16} y={16} tone={1} size={11} bold mono={false}>request 8f3a · stream=true · 1,204 prompt tokens · 180 output tokens</Label>
        <Label x={624} y={16} anchor="end" size={9.5} tone={3}>trace_id · 8f3a2c11 · 11 spans · 5 services</Label>

        {/* ── time axis ── */}
        <line x1={X0} x2={X1} y1={LANE_Y - 16} y2={LANE_Y - 16} className="v-rule" />
        {[0, 100, 200, 300, 400, 500, 600].map((t) => (
          <Label key={t} x={sx(t)} y={LANE_Y - 21} anchor="middle" size={8.5} tone={3}>{t}</Label>
        ))}
        <Label x={X1} y={LANE_Y - 32} anchor="end" size={8.5} tone={3}>milliseconds since the request arrived</Label>

        {/* ── service lanes ── */}
        {SERVICES.map((sv, i) => (
          <g key={sv.name}>
            <rect x={sv.x} y={LANE_Y + i * (ROW_H + 4) - 3} width={sv.w} height={ROW_H} rx={3}
              fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={0.7} />
            <Label x={sv.x + sv.w / 2} y={LANE_Y + i * (ROW_H + 4) + 11} anchor="middle" size={9.5}
              style={{ fill: 'var(--text-2)' }}>{sv.name}</Label>
          </g>
        ))}

        {/* ── spans ── */}
        <Fade show={showAll}>
          {rows.map(({ svc, sp }) => {
            const i = ORDER.indexOf(svc);
            const y = LANE_Y + i * (ROW_H + 4);
            const w = Math.max(2, sx(sp.start + sp.dur) - sx(sp.start));
            const onPath = CRITICAL_PATH.some((c) => Math.abs(c.start - sp.start) < 1 && Math.abs(c.dur - sp.dur) < 1);
            return (
              <g key={`${svc}-${sp.name}`} className="v-fade">
                <rect x={sx(sp.start)} y={y} width={w} height={ROW_H - 6} rx={2}
                  fill={KIND_COLOR[sp.kind]}
                  stroke={onPath && showPath ? 'var(--text)' : KIND_COLOR[sp.kind]}
                  strokeWidth={onPath && showPath ? 1.4 : 0.7}
                  opacity={onPath && showPath ? 1 : 0.85} />
                {w > 34 && (
                  <text x={sx(sp.start) + w / 2} y={y + 11} textAnchor="middle" className="v-label"
                    style={{ fontSize: 8.5, fill: sp.kind === 'queue' ? 'var(--text)' : '#0e1116', fontWeight: 600 }}>
                    {sp.name}
                  </text>
                )}
              </g>
            );
          })}
        </Fade>

        {/* ── critical path ── */}
        <Fade show={showPath}>
          {CRITICAL_PATH.map((c, i) => (
            <g key={i}>
              <rect x={sx(c.start)} y={LANE_Y + 5 * (ROW_H + 4) - 12} width={sx(c.start + c.dur) - sx(c.start)} height={9} rx={2}
                fill="var(--text)" opacity={0.85} />
            </g>
          ))}
          <Label x={X0} y={LANE_Y + 5 * (ROW_H + 4) + 12} size={9.5} tone={1} bold>critical path</Label>
          <Label x={sx(338)} y={LANE_Y + 5 * (ROW_H + 4) + 12} anchor="end" size={9.5} tone={1} bold>{`${total} ms`}</Label>
        </Fade>

        {/* ── the budget line ── */}
        <g className="v-fade">
          <line x1={X0} x2={X1} y1={LANE_Y + 5 * (ROW_H + 4) + 26} y2={LANE_Y + 5 * (ROW_H + 4) + 26}
            stroke={overBudget ? 'var(--error)' : 'var(--token)'} strokeWidth={1} strokeDasharray="4 3" />
          <Label x={X0} y={LANE_Y + 5 * (ROW_H + 4) + 40} size={9.5} kind={overBudget ? 'error' : 'token'} bold>
            {overBudget ? `budget ${budget} ms — this request is ${total - budget} ms over` : `budget ${budget} ms — ${budget - total} ms of headroom left`}
          </Label>
        </g>

        {/* ── the queue wait, called out ── */}
        <Fade show={showQueue}>
          <Arrow x1={sx(50)} y1={248} x2={sx(58)} y2={268} kind="idle" />
          <Label x={sx(60)} y={272} size={9.5} kind="compute" bold>58 ms of queueing</Label>
          <Label x={sx(60)} y={286} size={9} tone={3}>not a kernel. Fix it with capacity or a cap, not with a better GEMM.</Label>
        </Fade>

        {/* ── the unattributed gap ── */}
        <Fade show={showGap}>
          <rect x={sx(326)} y={296} width={sx(666) - sx(326)} height={20} rx={2}
            fill="url(#tps-hatch)" stroke="var(--error)" strokeDasharray="3 3" />
          <Label x={sx(330)} y={331} size={9.5} kind="error" bold>340 ms with no span covering it</Label>
          <Label x={sx(330)} y={345} size={9} tone={3}>Either something is uninstrumented, or your exporter died. Same thing to the user.</Label>
        </Fade>

        {/* ── sampling ── */}
        <Fade show={showSample}>
          <Label x={16} y={374} tone={2} size={10.5} bold mono={false}>Sampling: 1% of traces kept</Label>
          <Label x={200} y={374} size={9.5} tone={3}>tail-based keeps every slow request</Label>
          <Arrow x1={392} y1={370} x2={424} y2={370} kind="token" />
          <Label x={432} y={374} size={9.5} kind="token" bold>your p99 story survives; head sampling may lose it</Label>
        </Fade>
      </svg>
    </Stage>
  );
}