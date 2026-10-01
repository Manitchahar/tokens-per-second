import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, Span, fmt } from '../lib/svg';
import { PctMark } from './helpers';

// One latency distribution, summarised four different ways. The point of the
// lesson is that the four summaries describe four different populations.

const STEPS = [
  'The distribution',
  'The mean',
  'The median',
  'The tail',
  'Same data, four answers',
  'The headroom rule',
  'Choosing a percentile',
];
const CAPTIONS = [
  'Every dot is one request. Most are fast; a few are very slow. This is the shape of every LLM service.',
  'The mean adds them up and divides. At 512 ms it sits nearly twice the median, because a handful of slow requests count as much as all the fast ones.',
  'The median is the halfway point: 280 ms. This is where most users actually live, and it is the number to quote as "normal".',
  'p99 is the value 99% of requests beat. This is the unlucky 1% — often ten to a hundred times the median.',
  'Same data, four answers: mean 512 ms, p50 280 ms, p90 1,007 ms, p99 2,806 ms. All four are true, and they describe different populations.',
  'To hit a p99 target you must provision for the p99 case — here 10× the typical request. That is where the cost of the tail comes from.',
  'Report p50 for the typical user and p95/p99 for the tail. A bare mean for latency describes nobody.',
];

// A plausible TTFT distribution in ms: a tight cluster plus a slow tail.
const SAMPLES = [
  180, 205, 190, 240, 220, 265, 195, 230, 250, 210, 285, 275, 200, 260, 240, 225,
  310, 340, 290, 355, 380, 410, 470, 520, 610, 740, 980, 1250, 1840, 3200,
];

const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const pct = (a: number[], p: number) => {
  const s = [...a].sort((x, y) => x - y);
  const i = (s.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (i - lo);
};

const H0 = 56;   // histogram top
const HB = 250;  // histogram bottom
const H1 = 610;  // histogram right
const V0 = 96;   // dot plot left
const V1 = 610;  // dot plot right
const MAX = 3400;

const hx = (v: number) => H0 + (v / MAX) * (H1 - H0);
const vx = (v: number) => V0 + (v / MAX) * (V1 - V0);
// histogram bars: log-ish buckets so the tail is visible at all
const BUCKETS = [0, 200, 250, 300, 400, 600, 1000, 1800, 3400];

export default function TailLatency() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [target, setTarget] = useState(99);

  const mu = mean(SAMPLES);
  const med = pct(SAMPLES, 0.5);
  const p90 = pct(SAMPLES, 0.9);
  const p99 = pct(SAMPLES, 0.99);
  const max = Math.max(...SAMPLES);

  const counts = BUCKETS.slice(0, -1).map((lo, i) => ({
    lo,
    hi: BUCKETS[i + 1],
    n: SAMPLES.filter((v) => v >= lo && (i === BUCKETS.length - 2 ? v <= BUCKETS[i + 1] : v < BUCKETS[i + 1])).length,
  }));
  const maxN = Math.max(...counts.map((c) => c.n));

  const tMean = useTween(mu, 700);
  const tMed = useTween(med, 700);
  const tP99 = useTween(p99, 700);

  // what fraction of the slowest tail must the slowest percentile be provisioned for
  const prov = pct(SAMPLES, target / 100);

  const showMean = s.step >= 1;
  const showMed = s.step >= 2;
  const showTail = s.step >= 3;
  const showCompare = s.step >= 4;
  const showHead = s.step >= 5;
  const showPick = s.step >= 6;

  return (
    <Stage
      rootRef={ref}
      title="One latency distribution, four summaries — and they disagree"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'memory', 'error', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Mean', value: `${fmt.int(mu)} ms`, kind: 'compute', note: 'nobody experiences this' },
          { label: 'Median (p50)', value: `${fmt.int(med)} ms`, kind: 'token', note: 'the typical user' },
          { label: 'p99', value: `${fmt.int(p99)} ms`, kind: 'error', note: `${(p99 / med).toFixed(1)}× the median` },
          { label: `Provision for p${target}`, value: `${fmt.int(prov)} ms`, kind: 'memory', note: 'sizing the slow case' },
        ]} />
      }
      controls={
        <Slider label="Tail percentile to provision for" value={target} min={50} max={99} onChange={setTarget} format={(v) => `p${v}`} kind="error" />
      }
    >
      <svg viewBox="0 0 640 360" role="img" aria-label="A histogram of request latencies with a long right tail, above a dot plot where markers show the mean, median, p90 and p99 positions, and a bar comparing the four values.">
        {/* ── histogram ── */}
        <Label x={H0} y={H0 - 14} tone={2} size={10.5} mono={false} bold>30 requests, one dot each — the distribution</Label>
        {counts.map((c, i) => {
          const w = hx(c.hi) - hx(c.lo);
          const h = (c.n / maxN) * (HB - H0 - 16);
          return (
            <g key={i}>
              <Span x={hx(c.lo)} y={HB - h} w={Math.max(2, w - 2)} h={h} kind="memory" dim={i > 4} label={c.n > 0 && w > 14 ? String(c.n) : undefined} labelTone="light" />
            </g>
          );
        })}
        <line x1={H0} x2={H1} y1={HB} y2={HB} className="v-rule" />
        {[0, 500, 1000, 2000, 3000].map((v) => (
          <Label key={v} x={hx(v)} y={HB + 13} anchor="middle" tone={3} size={9}>{v === 0 ? '0' : `${v / 1000}k`}</Label>
        ))}
        <Label x={(H0 + H1) / 2} y={HB + 26} anchor="middle" tone={3} size={9.5}>TTFT in milliseconds</Label>

        {/* ── dot plot with the four summaries ── */}
        <Label x={V0} y={V0 - 8} tone={2} size={10.5} mono={false} bold>Same data, different questions</Label>
        <line x1={V0} x2={V1} y1={V0 + 30} y2={V0 + 30} className="v-rule" />

        {SAMPLES.map((v, i) => (
          <circle key={i} cx={vx(v)} cy={V0 + 30} r={3.4} fill="var(--memory)" opacity={0.55} />
        ))}

        {/* mean */}
        {showMean && (
          <Fade show={showMean}>
            <line x1={vx(tMean)} x2={vx(tMean)} y1={V0 + 14} y2={V0 + 46} stroke="var(--compute)" strokeWidth={1.6} />
            <Label x={vx(tMean)} y={V0 + 10} anchor="middle" size={9.5} kind="compute" bold>{`mean ${fmt.int(mu)}`}</Label>
          </Fade>
        )}
        {/* median */}
        {showMed && (
          <Fade show={showMed}>
            <line x1={vx(tMed)} x2={vx(tMed)} y1={V0 + 14} y2={V0 + 46} stroke="var(--token)" strokeWidth={1.6} />
            <Label x={vx(tMed)} y={V0 + 58} anchor="middle" size={9.5} kind="token" bold>{`p50 ${fmt.int(med)}`}</Label>
          </Fade>
        )}
        {/* tail */}
        {showTail && (
          <Fade show={showTail}>
            <PctMark x={vx(p90)} y={V0 + 20} h={30} label={`p90 ${fmt.int(p90)}`} color="var(--idle)" />
            <PctMark x={vx(p99)} y={V0 + 20} h={30} label={`p99 ${fmt.int(p99)}`} color="var(--error)" strong />
            <line x1={vx(med)} x2={vx(p99)} y1={V0 + 78} y2={V0 + 78} stroke="var(--error)" strokeWidth={1} />
            <line x1={vx(med)} x2={vx(med)} y1={V0 + 74} y2={V0 + 82} stroke="var(--error)" strokeWidth={1} />
            <line x1={vx(p99)} x2={vx(p99)} y1={V0 + 74} y2={V0 + 82} stroke="var(--error)" strokeWidth={1} />
            <Label x={(vx(med) + vx(p99)) / 2} y={V0 + 92} anchor="middle" size={9.5} kind="error" bold>{`${(p99 / med).toFixed(1)}× apart — nobody lives at the mean`}</Label>
          </Fade>
        )}

        {/* ── the four bars ── */}
        {showCompare && (
          <Fade show={showCompare}>
            <Label x={V0} y={V0 + 130} tone={2} size={10.5} mono={false} bold>Which number answers your question?</Label>
            {([
              { k: 'mean', v: mu, kind: 'compute' as const, who: 'nobody' },
              { k: 'p50', v: med, kind: 'token' as const, who: 'most users' },
              { k: 'p90', v: p90, kind: 'idle' as const, who: 'tail users' },
              { k: 'p99', v: p99, kind: 'error' as const, who: 'the 1%' },
            ]).map((r, i) => {
              const y = V0 + 142 + i * 22;
              const w = (Math.log10(r.v + 1) / Math.log10(MAX + 1)) * 380;
              return (
                <g key={r.k}>
                  <text x={V0} y={y + 10} className="v-label t2" style={{ fontSize: 10 }}>{r.k}</text>
                  <rect x={V0 + 30} y={y} width={380} height={14} rx={2} fill="none" stroke="var(--rule)" />
                  <Span x={V0 + 30} y={y} w={w} h={14} kind={r.kind} />
                  <text x={V0 + 30 + w + 6} y={y + 10} className="v-label t1" style={{ fontSize: 9.5 }}>{`${fmt.int(r.v)} ms · ${r.who}`}</text>
                </g>
              );
            })}
          </Fade>
        )}

        {/* ── headroom ── */}
        {showHead && (
          <Fade show={showHead}>
            <text x={V0} y={V0 + 236} className="v-label t2" style={{ fontSize: 10.5, fontWeight: 600 }}>
              {`To keep p${target} under a target, size the system for ${fmt.int(prov)} ms — ${(prov / med).toFixed(1)}× the typical request.`}
            </text>
            <text x={V0} y={V0 + 250} className="v-label t3" style={{ fontSize: 9.5 }}>
              The tail is not free: it sets your capacity, your queue depth and your bill.
            </text>
          </Fade>
        )}

        {showPick && (
          <Fade show={showPick}>
            <text x={V0} y={V0 + 278} className="v-label t2" style={{ fontSize: 10.5, fontWeight: 600 }}>
              Latency SLOs are written in percentiles. Never write "average TTFT under 1 s".
            </text>
          </Fade>
        )}
      </svg>
    </Stage>
  );
}