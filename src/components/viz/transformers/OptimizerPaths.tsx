import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade } from '../lib/svg';

// Two weights (p, q) and a long, narrow valley: L = ½(p² + 50·q²). The minimum is at (0, 0).
// The valley is 50× steeper across (q) than along (p): the shape that makes plain gradient descent zig-zag.
const CURV = [1, 50];
const START: [number, number] = [-5, 1.5];
const N = 40;
const lossAt = (p: number, q: number) => 0.5 * (CURV[0] * p * p + CURV[1] * q * q);

type Mode = 'gd' | 'sgd' | 'adam';

/** Deterministic normal noise (a seeded LCG + Box–Muller), so server and browser draw the same path. */
function noiseSeq(n: number, seed = 7) {
  let s = seed >>> 0;
  const u = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s + 0.5) / 4294967296; };
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const r = Math.sqrt(-2 * Math.log(u()));
    const t = 2 * Math.PI * u();
    out.push([r * Math.cos(t), r * Math.sin(t)]);
  }
  return out;
}
const NOISE = noiseSeq(N);
const NOISE_1 = 4; // per-example gradient noise (std) at batch size 1

function run(mode: Mode, batch: number) {
  const pts: [number, number][] = [START];
  let [p, q] = START;
  let m = [0, 0], v = [0, 0];
  const sd = NOISE_1 / Math.sqrt(batch);
  for (let k = 0; k < N; k++) {
    let g = [CURV[0] * p, CURV[1] * q];
    if (mode !== 'gd') g = [g[0] + sd * NOISE[k][0], g[1] + sd * NOISE[k][1]];
    if (mode === 'adam') {
      const lr = 0.2, b1 = 0.9, b2 = 0.999, eps = 1e-8;
      m = [b1 * m[0] + (1 - b1) * g[0], b1 * m[1] + (1 - b1) * g[1]];
      v = [b2 * v[0] + (1 - b2) * g[0] ** 2, b2 * v[1] + (1 - b2) * g[1] ** 2];
      const c1 = 1 - b1 ** (k + 1), c2 = 1 - b2 ** (k + 1);
      p -= (lr * (m[0] / c1)) / (Math.sqrt(v[0] / c2) + eps);
      q -= (lr * (m[1] / c1)) / (Math.sqrt(v[1] / c2) + eps);
    } else {
      const lr = 0.035; // just under the 2/50 = 0.04 limit set by the steep direction
      p -= lr * g[0];
      q -= lr * g[1];
    }
    pts.push([p, q]);
  }
  return pts;
}

const STEPS = ['Two weights, one surface', 'Full-batch gradient descent', 'Mini-batches: cheap but noisy', 'Bigger batches, less noise', 'Adam: a step size per weight'];
const CAPTIONS = [
  'With two weights the loss is a surface. Each ring is a contour of equal loss; the bottom is the centre. This valley is 50× steeper across than along; real loss surfaces have directions that differ far more.',
  'Gradient descent on the whole dataset. The steep direction caps the learning rate (above 2/50 = 0.04 it diverges), so η = 0.035. The gradient points mostly at the walls: the path zig-zags and creeps along the floor, and after 40 steps the loss is still 0.72.',
  'A mini-batch of 8 examples gives a noisy estimate of the true gradient for 1/8 the work of 64. The path wobbles, but it still goes downhill on average. This is stochastic gradient descent (SGD).',
  'Noise in the gradient shrinks like 1 / √(batch size): 64 examples give 8× less noise than one. Drag the batch size. Bigger batches cost more per step but take straighter steps.',
  'Adam divides each weight’s step by a running average of that weight’s gradient size, so both directions move at a similar pace. No zig-zag: the loss is about 0.08 after the same 40 steps. With a fixed step size it then hovers around the bottom, which is why real training also shrinks the learning rate over time.',
];
const PRESET: Record<number, { mode: Mode; batch: number }> = {
  0: { mode: 'gd', batch: 64 }, 1: { mode: 'gd', batch: 64 }, 2: { mode: 'sgd', batch: 8 }, 3: { mode: 'sgd', batch: 64 }, 4: { mode: 'adam', batch: 8 },
};

// geometry: isotropic map from (p, q) to the left panel
const S = 52;
const OX0 = 330, OY0 = 175; // where (0, 0) sits
const sx = (p: number) => OX0 + p * S;
const sy = (q: number) => OY0 - q * S;
const LEVELS = [0.25, 1, 3, 8, 18, 35, 60];
// right panel: loss per step, log scale
const CX0 = 438, CX1 = 618, CY0 = 300, CY1 = 72;
const cx = (k: number) => CX0 + (k / N) * (CX1 - CX0);
const cy = (l: number) => CY0 - ((Math.log10(Math.max(l, 1e-3)) + 3) / 5) * (CY0 - CY1);

export default function OptimizerPaths() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [mode, setMode] = useState<Mode>('gd');
  const [batch, setBatch] = useState(64);
  useEffect(() => { setMode(PRESET[st].mode); setBatch(PRESET[st].batch); }, [st]);

  const pts = useMemo(() => run(mode, batch), [mode, batch]);
  const losses = pts.map(([p, q]) => lossAt(p, q));
  const show = st >= 1;
  const color = mode === 'adam' ? 'var(--token)' : 'var(--compute)';
  const clipP = (p: number) => Math.max(-5.9, Math.min(5.6, p));
  const clipQ = (q: number) => Math.max(-2.7, Math.min(2.7, q));
  const line = pts.map(([p, q], i) => `${i ? 'L' : 'M'}${sx(clipP(p)).toFixed(1)},${sy(clipQ(q)).toFixed(1)}`).join(' ');
  const curve = losses.map((l, i) => `${i ? 'L' : 'M'}${cx(i).toFixed(1)},${cy(l).toFixed(1)}`).join(' ');
  const final = losses[losses.length - 1];
  const examples = mode === 'gd' ? 'all of them' : `${batch * N}`;

  return (
    <Stage
      rootRef={ref}
      title="Gradient descent, SGD and Adam on the same valley"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['text', 'loss contours'], ['compute', 'GD / SGD path'], ['token', 'Adam path']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'optimizer', value: mode === 'gd' ? 'full-batch GD' : mode === 'sgd' ? `SGD, batch ${batch}` : `Adam, batch ${batch}`, kind: mode === 'adam' ? 'token' : 'compute' },
          { label: 'loss after 40 steps', value: final < 0.01 ? final.toExponential(1) : final.toFixed(3), kind: 'error', note: `started at ${losses[0].toFixed(1)}` },
          { label: 'examples read', value: examples, kind: 'memory', note: '40 steps' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Optimizer" value={mode} options={[{ value: 'gd' as Mode, label: 'Full-batch GD' }, { value: 'sgd' as Mode, label: 'Mini-batch SGD' }, { value: 'adam' as Mode, label: 'Adam' }]} onChange={setMode} />
          {mode !== 'gd' && <Slider label="Batch size" value={batch} min={1} max={256} log onChange={setBatch} format={(v) => `${v}`} kind="memory" />}
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="Left: elliptical loss contours of a narrow valley with an optimizer's path from the start to the centre. Right: the loss after each step on a log scale.">
        <defs>
          <clipPath id="optpaths-clip"><rect x={20} y={40} width={392} height={280} /></clipPath>
        </defs>
        <Label x={24} y={30} tone={1} size={10.5} bold>two weights: the loss surface from above</Label>
        <g clipPath="url(#optpaths-clip)">
          {LEVELS.map((c) => (
            <ellipse key={c} cx={sx(0)} cy={sy(0)} rx={Math.sqrt((2 * c) / CURV[0]) * S} ry={Math.sqrt((2 * c) / CURV[1]) * S}
              fill="none" stroke="var(--rule-2)" strokeWidth={1} />
          ))}
          <line x1={20} x2={412} y1={sy(0)} y2={sy(0)} className="v-rule faint" />
          <line x1={sx(0)} x2={sx(0)} y1={40} y2={320} className="v-rule faint" />
          <Fade show={show}>
            <path d={line} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" style={{ transition: 'stroke .4s' }} />
            {pts.map(([p, q], i) => (
              <circle key={i} cx={sx(clipP(p))} cy={sy(clipQ(q))} r={i === pts.length - 1 ? 4.5 : 1.9} fill={color} />
            ))}
          </Fade>
        </g>
        <circle cx={sx(START[0])} cy={sy(START[1])} r={5} fill="var(--text)" stroke="var(--ground-2)" strokeWidth={2} />
        <Label x={sx(START[0]) + 8} y={sy(START[1]) - 8} tone={2} size={9.5}>start</Label>
        <circle cx={sx(0)} cy={sy(0)} r={3} fill="var(--text)" />
        <Label x={sx(0) + 6} y={sy(0) + 14} tone={3} size={9}>minimum</Label>
        <Label x={24} y={334} tone={3} size={9.5}>weight p (gentle direction) →</Label>
        <Label x={412} y={334} anchor="end" tone={3} size={9.5}>steep direction: weight q ↕</Label>

        {/* loss per step */}
        <Label x={CX0} y={30} tone={1} size={10.5} bold>loss after each step</Label>
        <line x1={CX0} y1={CY0} x2={CX1} y2={CY0} className="v-rule" />
        <line x1={CX0} y1={CY0} x2={CX0} y2={CY1 - 6} className="v-rule" />
        {[0.001, 0.1, 10].map((v) => <Label key={v} x={CX0 - 5} y={cy(v) + 3} anchor="end" tone={3} size={9}>{v >= 1 ? String(v) : v}</Label>)}
        {[0, 20, 40].map((k) => <Label key={k} x={cx(k)} y={CY0 + 14} anchor="middle" tone={3} size={9}>{k}</Label>)}
        <Label x={(CX0 + CX1) / 2} y={CY0 + 30} anchor="middle" tone={3} size={9.5}>step</Label>
        <Fade show={show}>
          <path d={curve} fill="none" stroke="var(--error)" strokeWidth={1.4} />
        </Fade>
        <Fade show={!show}>
          <Label x={(CX0 + CX1) / 2} y={190} anchor="middle" tone={3} size={9.5}>(no steps yet)</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
