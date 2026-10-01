import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade } from '../lib/svg';
import { PctMark } from './helpers';

// Reproducibility, drawn as a scatter of repeated measurements plus the
// warm-up curve. No RNG: the samples are deterministic so a reader who moves
// the slider twice sees the same picture, which is the whole point of the lesson.

const STEPS = [
  'One run tells you nothing',
  'Repeat it five times',
  'The spread is real',
  'Warm-up changes the answer',
  'Averaging is not enough',
  'Report a distribution',
  'What to fix first',
];
const CAPTIONS = [
  'A single timing is a draw from a distribution you have not looked at. It can be off by 40% in either direction.',
  'Five repeats of the identical workload. Same code, same prompt, same GPU.',
  'The runs do not agree. This spread is not noise to be averaged away — it is the honest behaviour of the system.',
  'Early runs are slow: kernels compile, caches fill, the allocator grows. The warm-up curve has a shape.',
  'One outlier at the far right drags the mean up while barely moving the median.',
  'The trustworthy summary is a distribution with a median and a tail, not a single number.',
  'If repeats disagree this much, the harness is not ready to compare anything. Fix the harness first.',
];

const N = 12;
const WARMUP = [1.0, 0.74, 0.55, 0.46, 0.42, 0.40, 0.39, 0.385, 0.38, 0.379, 0.378, 0.377];
// measured ms/token at batch 1 after warm-up, for one configuration
const baseSamples = [6.9, 7.4, 7.1, 8.9, 7.2, 7.0, 7.5, 11.8, 7.3, 7.1, 7.4, 7.2];

const CX0 = 78;
const CX1 = 300;
const CY1 = 66;
const CY0 = 236;
const Y0 = 12; // 4 ms
const Y1 = 14; // 16 ms
const cy = (v: number) => CY0 - ((Math.min(Y1, Math.max(Y0, v)) - Y0) / (Y1 - Y0)) * (CY0 - CY1);

const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const pctl = (a: number[], p: number) => {
  const s = [...a].sort((x, y) => x - y);
  const i = (s.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (i - lo);
};

export default function RunVariance() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [jitter, setJitter] = useState(1);
  const [warmupOn, setWarmupOn] = useState(true);

  const samples = baseSamples.map((v, i) => {
    // a deterministic wobble so the slider visibly changes the spread
    const w = jitter === 1 ? 0 : Math.sin(i * 2.3) * (jitter - 1) * 0.55;
    return Math.max(3, v + w);
  });

  const mu = mean(samples);
  const med = median(samples);
  const p95 = pctl(samples, 0.95);
  const lo = Math.min(...samples);
  const hi = Math.max(...samples);
  const cv = ((mu - med) / med) * 100; // how far the mean sits above the median

  const warmPts = WARMUP.map((f, i) => `${i ? 'L' : 'M'}${(CX0 + 40 + i * 14).toFixed(1)},${cy(f * med).toFixed(1)}`).join(' ');
  const warmEnd = WARMUP[WARMUP.length - 1] * med;
  const warm = useTween(warmEnd, 700);

  const showRepeats = s.step >= 1;
  const showSpread = s.step >= 2;
  const showWarmup = s.step >= 3;
  const showMean = s.step >= 4;
  const showDist = s.step >= 5;
  const showFix = s.step >= 6;

  return (
    <Stage
      rootRef={ref}
      title="Five runs are not a measurement until they agree"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'compute', 'error', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Median', value: `${med.toFixed(2)} ms`, kind: 'token', note: 'use this one' },
          { label: 'Mean', value: `${mu.toFixed(2)} ms`, kind: cv > 4 ? 'error' : 'text', note: `${cv > 0 ? '+' : ''}${cv.toFixed(1)}% above median` },
          { label: 'Slowest repeat', value: `${hi.toFixed(1)} ms`, kind: 'error', note: `${((hi / med - 1) * 100).toFixed(0)}% slower than median` },
          { label: 'Warm-up cost', value: `${(WARMUP[0] * med - warm).toFixed(1)} ms`, kind: 'compute', note: 'run 1 vs run 12' },
        ]} />
      }
      controls={
        <>
          <Slider label="Run-to-run spread" value={jitter} min={0} max={3} step={0.1} onChange={setJitter} format={(v) => `${(v * 20).toFixed(0)}% jitter`} kind="error" />
          <Slider label="Warm-up iterations (0 to 12)" value={warmupOn ? 12 : 0} min={0} max={12} step={1} onChange={(v) => setWarmupOn(v >= 8)} format={(v) => (v === 0 ? 'none' : `${v} warm`)} kind="compute" />
        </>
      }
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="A scatter plot of twelve repeated timings with an outlier, plus a warm-up curve falling from the first iteration to a steady value, and markers for the mean, median and slowest repeat.">
        {/* axis */}
        <line x1={CX0} x2={CX1} y1={CY0} y2={CY0} className="v-rule" />
        <line x1={CX0} x2={CX0} y1={CY1 - 10} y2={CY0} className="v-rule" />
        {[4, 8, 12, 16].map((v) => (
          <g key={v}>
            <line x1={CX0 - 4} x2={CX0} y1={cy(v)} y2={cy(v)} className="v-rule" />
            <text x={CX0 - 8} y={cy(v) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 9 }}>{v}</text>
          </g>
        ))}
        <text x={CX0 - 52} y={(CY0 + CY1) / 2} transform={`rotate(-90 ${CX0 - 52} ${(CY0 + CY1) / 2})`} textAnchor="middle" className="v-label t3" style={{ fontSize: 9.5 }}>ms per token</text>

        {/* 12 repeated runs */}
        {showRepeats && samples.map((v, i) => (
          <g key={i} className="v-fade" style={{ animationDelay: `${i * 40}ms` }}>
            <circle cx={CX0 + 22 + i * 16} cy={cy(v)} r={4} fill={v > med * 1.35 ? 'var(--error)' : 'var(--token)'} stroke="var(--ground-2)" strokeWidth={1.2} />
            <text x={CX0 + 22 + i * 16} y={CY0 + 13} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{i + 1}</text>
          </g>
        ))}
        {showRepeats && <Label x={CX0 + 22} y={CY0 + 28} tone={3} size={9.5}>repeat number</Label>}

        {/* the outlier, called out */}
        {showSpread && hi > med * 1.3 && (
          <Fade show={showSpread}>
            <Arrow2 x={CX0 + 22 + samples.indexOf(hi) * 16} y={cy(hi)} label="one bad run" />
          </Fade>
        )}

        {/* summary markers */}
        {showMean && (
          <Fade show={showMean}>
            <line x1={CX0} x2={CX1} y1={cy(mu)} y2={cy(mu)} stroke="var(--compute)" strokeWidth={1.4} strokeDasharray="5 3" />
            <Label x={CX0 + 4} y={cy(mu) - 6} tone={2} size={9.5} kind="compute">{`mean ${mu.toFixed(2)}`}</Label>
          </Fade>
        )}
        {showDist && (
          <Fade show={showDist}>
            <PctMark x={CX1 + 34} y={cy(med)} h={CY0 - cy(med)} label="median" color="var(--token)" strong />
            <PctMark x={CX1 + 68} y={cy(hi)} h={CY0 - cy(hi)} label="slowest" color="var(--error)" strong />
          </Fade>
        )}

        {/* warm-up curve */}
        {showWarmup && (
          <Fade show={showWarmup}>
            <path d={warmPts} fill="none" stroke="var(--compute)" strokeWidth={1.6} />
            {WARMUP.map((f, i) => (
              <circle key={i} cx={CX0 + 40 + i * 14} cy={cy(f * med)} r={i === 11 ? 3 : 1.8} fill="var(--compute)" />
            ))}
            <Label x={CX0 + 36} y={CY1 - 14} tone={2} size={10} kind="compute" mono={false}>Warm-up: iteration vs steady state</Label>
            <text x={CX0 + 40} y={cy(WARMUP[0] * med) - 8} className="v-label" style={{ fill: 'var(--compute)', fontSize: 9 }}>
              first run: {(WARMUP[0] * med).toFixed(1)} ms
            </text>
            <text x={CX0 + 186} y={cy(warm) + 16} className="v-label" style={{ fill: 'var(--token)', fontSize: 9 }}>
              after warm-up: {warm.toFixed(2)} ms
            </text>
          </Fade>
        )}

        {/* the fix list */}
        {showFix && (
          <Fade show={showFix}>
            <Label x={CX1 + 100} y={CY1 - 14} tone={2} size={10} mono={false} bold>Before comparing anything</Label>
            {[
              '1. warm up until flat',
              '2. repeat at least 5×',
              '3. report the median + p95',
              '4. pin seed, model, versions',
            ].map((t, i) => (
              <text key={t} x={CX1 + 100} y={CY1 + 6 + i * 15} className="v-label t2" style={{ fontSize: 9.5 }}>{t}</text>
            ))}
          </Fade>
        )}
      </svg>
    </Stage>
  );
}

// a short leader line with a label, used to point at a single outlier
function Arrow2({ x, y, label }: { x: number; y: number; label: string }) {
  return (
    <g>
      <line x1={x} x2={x + 20} y1={y} y2={y - 22} stroke="var(--error)" strokeWidth={1} />
      <text x={x + 23} y={y - 25} className="v-label" style={{ fill: 'var(--error)', fontSize: 9.5, fontWeight: 600 }}>{label}</text>
    </g>
  );
}