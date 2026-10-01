import { useRef, useState } from 'react';
import { Stage, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Label, Fade } from '../lib/svg';

const KEYS = [
  { name: 'cat', v: [2.2, 1.2] },
  { name: 'sat', v: [0.5, 2.4] },
  { name: 'the', v: [-2.0, 0.8] },
  { name: 'mat', v: [1.6, -1.6] },
];
const STEPS = ['A vector is an arrow', 'The dot product', 'Alignment = similarity', 'Exponentiate', 'Normalize: softmax', 'Temperature', 'Staying numerically safe'];
const CAPTIONS = [
  'A vector of 2 numbers is an arrow on a plane. (Real models use 128 numbers per head, but the geometry is the same.)',
  'The dot product multiplies matching coordinates and adds them. For q = (3, 1) and k = (2, 2): 3×2 + 1×2 = 8.',
  'Arrows pointing the same way give a large positive dot product, perpendicular ones about zero, opposite ones negative.',
  'Raise e to each score. Everything becomes positive, and bigger scores get much bigger.',
  'Divide by the total so the numbers sum to 1. That is softmax: scores in, probabilities out.',
  'Divide scores by a temperature first. Low temperature sharpens the winner; high temperature flattens toward uniform.',
  'Real kernels subtract the largest score before exponentiating. Same answer, but e^x never overflows.',
];

const CX = 150, CY = 165, SC = 42;
export default function DotSoftmax() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [angle, setAngle] = useState(25);
  const [temp, setTemp] = useState(1);
  const T = st >= 5 ? temp : 1;

  const q = st <= 1 ? [3, 1] : [2.6 * Math.cos((angle * Math.PI) / 180), 2.6 * Math.sin((angle * Math.PI) / 180)];
  const keys = st === 1 ? [{ name: 'k', v: [2, 2] }] : KEYS;
  const scores = keys.map((k) => q[0] * k.v[0] + q[1] * k.v[1]);
  const scaled = scores.map((x) => x / T);
  const mx = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - mx));
  const rawExps = scaled.map((x) => Math.exp(x));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);
  const best = probs.indexOf(Math.max(...probs));

  // bar values depending on stage
  const mode = st <= 2 ? 'score' : st === 3 ? 'exp' : 'prob';
  const vals = mode === 'score' ? scores : mode === 'exp' ? rawExps : probs;
  const maxAbs = mode === 'score' ? 12 : mode === 'exp' ? Math.max(...rawExps) : 1;
  const tw = useTweens(Object.fromEntries(vals.map((v, i) => [`b${i}`, v / maxAbs])) as Record<string, number>, 600);

  const pt = (v: number[]) => [CX + v[0] * SC, CY - v[1] * SC];
  const [qx, qy] = pt(q);
  const BX = 360, BW = 250;
  const zeroX = mode === 'score' ? BX + BW * 0.4 : BX;
  const span = mode === 'score' ? BW * 0.6 : BW;

  return (
    <Stage
      rootRef={ref}
      title="Dot products and softmax"
      steps={STEPS}
      captions={CAPTIONS}
      {...s}
      controls={st >= 2 ? (
        <>
          <Slider label="Query direction" value={angle} min={-180} max={180} onChange={setAngle} format={(v) => `${v}°`} />
          {st >= 5 && <Slider label="Temperature" value={temp} min={0.1} max={3} step={0.05} onChange={setTemp} format={(v) => v.toFixed(2)} kind="compute" />}
        </>
      ) : undefined}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="A query arrow and key arrows on a plane, their dot products as bars, then softmax probabilities.">
        {/* plane */}
        <line x1={CX - 130} x2={CX + 130} y1={CY} y2={CY} className="v-rule faint" />
        <line x1={CX} x2={CX} y1={CY - 130} y2={CY + 130} className="v-rule faint" />
        {keys.map((k, i) => {
          const [x, y] = pt(k.v);
          const hl = st >= 4 && i === best;
          return (
            <g key={k.name}>
              <line x1={CX} y1={CY} x2={x} y2={y} stroke="var(--memory)" strokeWidth={hl ? 2.4 : 1.6} markerEnd="url(#tps-arrow)" style={{ opacity: st === 0 ? 0 : 1, transition: 'opacity .5s' }} />
              {st > 0 && <Label x={x + (k.v[0] >= 0 ? 6 : -6)} y={y + (k.v[1] >= 0 ? -6 : 14)} anchor={k.v[0] >= 0 ? 'start' : 'end'} size={11} kind="memory" bold={hl}>{k.name}</Label>}
            </g>
          );
        })}
        <line x1={CX} y1={CY} x2={qx} y2={qy} stroke="var(--text)" strokeWidth={2.2} markerEnd="url(#tps-arrow)" />
        <Label x={qx + 6} y={qy - 6} tone={1} size={11} bold>q</Label>
        <Fade show={st <= 1}>
          <Label x={CX - 120} y={30} tone={2} size={10.5}>{st === 0 ? 'q = (3, 1)' : 'q = (3, 1),  k = (2, 2)'}</Label>
        </Fade>

        {/* bars */}
        <Fade show={st >= 1}>
          <g>
            <Label x={BX} y={30} tone={2} size={10.5}>{mode === 'score' ? 'dot product  q · k' : mode === 'exp' ? 'e^(score)' : st >= 5 ? `softmax(score / ${T.toFixed(2)})` : 'softmax = e^score / sum'}</Label>
            {mode === 'score' && <line x1={zeroX} x2={zeroX} y1={46} y2={46 + keys.length * 44} className="v-rule" />}
            {keys.map((k, i) => {
              const f = tw[`b${i}`] ?? 0;
              const w = Math.abs(f) * span;
              const x = f >= 0 ? zeroX : zeroX - w;
              const y = 56 + i * 44;
              const hl = mode === 'prob' && i === best;
              return (
                <g key={k.name}>
                  <text x={BX - 10} y={y + 12} textAnchor="end" className="v-label" style={{ fill: 'var(--memory)', fontSize: 11 }}>{k.name}</text>
                  <rect x={x} y={y} width={Math.max(1, w)} height={16} rx={2}
                    fill={hl ? 'var(--text)' : f < 0 ? 'none' : 'rgb(232 230 225 / 0.35)'} stroke={f < 0 ? 'var(--text-3)' : hl ? 'var(--text)' : 'var(--text-2)'} strokeDasharray={f < 0 ? '3 3' : undefined} />
                  <Label x={f >= 0 ? x + w + 6 : x - 6} y={y + 12} anchor={f >= 0 ? 'start' : 'end'} tone={1} size={10.5}>
                    {mode === 'score' ? vals[i].toFixed(1) : mode === 'exp' ? (vals[i] > 9999 ? vals[i].toExponential(1) : vals[i].toFixed(1)) : `${(vals[i] * 100).toFixed(1)}%`}
                  </Label>
                </g>
              );
            })}
            <Fade show={mode === 'prob'}>
              <Label x={BX} y={56 + keys.length * 44 + 8} tone={3} size={10}>{`sum = ${(probs.reduce((a, b) => a + b, 0) * 100).toFixed(0)}%`}</Label>
            </Fade>
          </g>
        </Fade>

        <Fade show={st === 6}>
          <g>
            <Label x={BX} y={262} tone={2} size={10.5}>{`max score = ${mx.toFixed(1)}: compute e^(score − max)`}</Label>
            <Label x={BX} y={282} tone={3} size={10}>{'FP16 overflows past e^11 (≈ 65,504). Subtracting the max keeps'}</Label>
            <Label x={BX} y={298} tone={3} size={10}>{'every exponent ≤ 0, so every e^x is between 0 and 1.'}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
