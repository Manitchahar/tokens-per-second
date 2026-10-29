import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep, prefersReducedMotion } from '../lib/useStep';
import { Label, Fade } from '../lib/svg';

// A one-weight model, ŷ = w·x, fitted to three examples that follow y = 2x.
// Loss = mean squared error = (1/3)·Σ(w·xᵢ − yᵢ)² = (14/3)·(w − 2)², so the slope is (28/3)·(w − 2).
const XS = [1, 2, 3];
const YS = [2, 4, 6];
const W0 = 0.5;
const loss = (w: number) => XS.reduce((s, x, i) => s + (w * x - YS[i]) ** 2, 0) / XS.length;
const slope = (w: number) => (2 / XS.length) * XS.reduce((s, x, i) => s + x * (w * x - YS[i]), 0);

/** Learning rate that the course quotes for each step of the story. */
const PRESET: Record<number, number> = { 4: 0.05, 5: 0.05, 6: 0.25, 7: 0.005 };
const ITERS = 8;

/** Steps of gradient descent until the loss drops below 0.01, or null if it never does. */
function stepsToConverge(eta: number) {
  let w = W0;
  for (let k = 0; k < 2000; k++) {
    if (loss(w) < 0.01) return k;
    w -= eta * slope(w);
    if (!Number.isFinite(w) || Math.abs(w) > 1e6) return null;
  }
  return null;
}

const STEPS = [
  'A model that is wrong',
  'Loss: one number for how wrong',
  'Every weight has a loss',
  'The slope points uphill',
  'Take one step downhill',
  'Repeat until the bottom',
  'Too big a step: divergence',
  'Too small a step: crawling',
];
const CAPTIONS = [
  'Three examples that follow y = 2x, and a one-weight model ŷ = w·x that starts at w = 0.5. Its predictions (0.5, 1.0, 1.5) all fall short. The red bars are the errors.',
  'Square each error and average them: (2.25 + 9 + 20.25) / 3 = 10.5. That single number is the loss. Lower is better, zero is perfect.',
  'Try every w and plot its loss: a bowl with its bottom at w = 2. Drag the weight yourself and watch the line and the ball move together. Finding the bottom by hand is fine for one weight and hopeless for eight billion.',
  'The slope of the bowl at the ball is the gradient, dL/dw. At w = 0.5 it is −14: the loss falls as w grows. The gradient points uphill, so we move the other way.',
  'Update: w ← w − η · slope. With a learning rate η = 0.05 the step is −0.05 × (−14) = +0.7, so w goes from 0.5 to 1.2 and the loss drops from 10.5 to 2.99.',
  'Do it again from the new spot. The slope shrinks as the bowl flattens, so the steps shrink too: after 6 steps the loss is below 0.01. Drag η to feel how the step size changes the path.',
  'At η = 0.25 every step overshoots the bottom by more than it started: w goes 0.5 → 4.0 → −0.67 → 5.56. The loss grows by 1.78× per step. Training has diverged.',
  'At η = 0.005 nothing goes wrong, but after 8 steps the loss is still 4.9. It needs 73 steps to get under 0.01, against 6 at η = 0.05. Too small wastes compute; too big blows up.',
];

// left panel: the data and the fitted line
const DX0 = 46, DX1 = 236, DY0 = 266, DY1 = 70;
const dx = (x: number) => DX0 + (x / 3.5) * (DX1 - DX0);
const dy = (y: number) => DY0 - (y / 8) * (DY0 - DY1);
// right panel: the loss curve over w
const WMIN = -1.5, WMAX = 5.5, LMAX = 60;
const LX0 = 300, LX1 = 612, LY0 = 266, LY1 = 70;
const lx = (w: number) => LX0 + ((w - WMIN) / (WMAX - WMIN)) * (LX1 - LX0);
const ly = (l: number) => LY0 - (Math.min(l, LMAX) / LMAX) * (LY0 - LY1);
const clampW = (w: number) => Math.max(WMIN, Math.min(WMAX, w));

export default function LossDescent() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [wHand, setWHand] = useState(W0);
  const [eta, setEta] = useState(0.05);
  const [shown, setShown] = useState(ITERS);

  useEffect(() => { if (PRESET[st] !== undefined) setEta(PRESET[st]); }, [st]);

  const K = st === 4 ? 1 : ITERS;
  const path = useMemo(() => {
    const out = [W0];
    for (let k = 0; k < K; k++) out.push(out[k] - eta * slope(out[k]));
    return out;
  }, [eta, K]);

  // reveal the iterates one at a time, calmly
  useEffect(() => {
    if (st < 4) return;
    if (prefersReducedMotion()) { setShown(K); return; }
    setShown(0);
    let k = 0;
    const t = window.setInterval(() => { k += 1; setShown(k); if (k >= K) window.clearInterval(t); }, 620);
    return () => window.clearInterval(t);
  }, [st, eta, K]);

  const descending = st >= 4;
  const idx = Math.min(shown, path.length - 1);
  const w = descending ? path[idx] : st >= 2 ? wHand : W0;
  const L = loss(w);
  const g = slope(w);
  const diverging = descending && Math.abs(1 - eta * (2 / 3) * 14) >= 1;
  const conv = stepsToConverge(eta);
  const offChart = w < WMIN || w > WMAX || L > LMAX;

  const curve = useMemo(() => {
    const pts: string[] = [];
    for (let i = 0; i <= 80; i++) {
      const ww = WMIN + ((WMAX - WMIN) * i) / 80;
      pts.push(`${i ? 'L' : 'M'}${lx(ww).toFixed(1)},${ly(loss(ww)).toFixed(1)}`);
    }
    return pts.join(' ');
  }, []);

  const errs = XS.map((x, i) => w * x - YS[i]);
  const bx = lx(clampW(w));
  const by = ly(L);
  const half = 0.7;
  const tanColor = 'var(--compute)';

  return (
    <Stage
      rootRef={ref}
      title="Gradient descent: roll the weight downhill on the loss"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['text', 'data / model'], ['error', 'error & loss'], ['compute', 'gradient & step']]}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'weight w', value: Math.abs(w) > 999 ? w.toExponential(1) : w.toFixed(3), kind: 'text', note: descending ? `after ${idx} step${idx === 1 ? '' : 's'}` : 'the one knob' },
            { label: 'loss L(w)', value: L > 9999 ? L.toExponential(1) : L.toFixed(L < 0.1 ? 4 : 2), kind: 'error', note: 'mean squared error' },
            ...(st >= 3 ? [{ label: 'slope dL/dw', value: Math.abs(g) > 999 ? g.toExponential(1) : g.toFixed(2), kind: 'compute' as const, note: g < 0 ? 'downhill is to the right' : g > 0 ? 'downhill is to the left' : 'at the bottom' }] : []),
            ...(descending ? [{ label: 'steps to L < 0.01', value: conv === null ? 'never' : String(conv), kind: (conv === null ? 'error' : 'text') as 'error' | 'text', note: `η = ${eta.toFixed(3)}` }] : []),
          ]}
        />
      }
      controls={
        st === 2 || st === 3 ? (
          <Slider label="Weight w (tune it by hand)" value={wHand} min={-1} max={5} step={0.05} onChange={setWHand} format={(v) => v.toFixed(2)} />
        ) : descending ? (
          <Slider label="Learning rate η" value={eta} min={0.001} max={0.25} step={0.001} onChange={setEta} format={(v) => v.toFixed(3)} kind="compute" />
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 360" role="img" aria-label="Left: three data points on the line y = 2x and a model line y = w times x with red error bars. Right: the loss as a function of w, a bowl with its bottom at w = 2, with a ball that steps downhill by the learning rate times the slope.">
        <defs>
          <clipPath id="lossdescent-data"><rect x={DX0} y={DY1 - 6} width={DX1 - DX0 + 4} height={DY0 - DY1 + 6} /></clipPath>
        </defs>

        {/* ── left: the data and the model ── */}
        <Label x={DX0} y={40} tone={1} size={10.5} bold>the data and the model ŷ = w·x</Label>
        <line x1={DX0} y1={DY0} x2={DX1} y2={DY0} className="v-rule" />
        <line x1={DX0} y1={DY0} x2={DX0} y2={DY1 - 6} className="v-rule" />
        {[1, 2, 3].map((x) => <Label key={x} x={dx(x)} y={DY0 + 14} anchor="middle" tone={3} size={9}>{x}</Label>)}
        {[2, 4, 6, 8].map((y) => <Label key={y} x={DX0 - 6} y={dy(y) + 3} anchor="end" tone={3} size={9}>{y}</Label>)}
        <Label x={(DX0 + DX1) / 2} y={DY0 + 30} anchor="middle" tone={3} size={9.5}>input x</Label>
        <g clipPath="url(#lossdescent-data)">
          <line x1={dx(0)} y1={dy(0)} x2={dx(3.5)} y2={dy(3.5 * w)} stroke="var(--text-2)" strokeWidth={1.6} style={{ transition: 'all .55s var(--ease-out)' }} />
          {XS.map((x, i) => (
            <line key={i} x1={dx(x)} x2={dx(x)} y1={dy(YS[i])} y2={dy(w * x)} stroke="var(--error)" strokeWidth={2.4} style={{ transition: 'all .55s var(--ease-out)' }} />
          ))}
          {XS.map((x) => (
            <circle key={x} cx={dx(x)} cy={dy(w * x)} r={3.5} fill="var(--ground-2)" stroke="var(--text-2)" strokeWidth={1.2} style={{ transition: 'all .55s var(--ease-out)' }} />
          ))}
        </g>
        {XS.map((x, i) => <circle key={x} cx={dx(x)} cy={dy(YS[i])} r={4.5} fill="var(--text)" />)}
        <Label x={DX1 - 4} y={DY1 + 6} anchor="end" tone={3} size={9}>{`line: w = ${Math.abs(w) > 99 ? w.toExponential(1) : w.toFixed(2)}`}</Label>

        <Fade show={st >= 1}>
          <Label x={DX0} y={318} tone={2} size={9.5}>{`errors   ${errs.map((e) => (Math.abs(e) > 99 ? e.toExponential(0) : e.toFixed(2))).join('  ')}`}</Label>
          <Label x={DX0} y={332} tone={2} size={9.5}>{`squared  ${errs.map((e) => (e * e > 999 ? (e * e).toExponential(0) : (e * e).toFixed(2))).join('  ')}`}</Label>
          <Label x={DX0} y={348} size={10} kind="error" bold>{`mean = loss = ${L > 9999 ? L.toExponential(1) : L.toFixed(2)}`}</Label>
        </Fade>

        {/* ── right: the loss over every possible w ── */}
        <Fade show={st >= 2}>
          <Label x={LX0} y={40} tone={1} size={10.5} bold>the loss for every possible w</Label>
          <line x1={LX0} y1={LY0} x2={LX1} y2={LY0} className="v-rule" />
          <line x1={lx(0)} y1={LY0} x2={lx(0)} y2={LY1 - 6} className="v-rule faint" />
          {[-1, 0, 1, 2, 3, 4, 5].map((v) => <Label key={v} x={lx(v)} y={LY0 + 14} anchor="middle" tone={3} size={9}>{v}</Label>)}
          {[20, 40, 60].map((v) => <Label key={v} x={lx(0) - 5} y={ly(v) + 3} anchor="end" tone={3} size={9}>{v}</Label>)}
          <Label x={(LX0 + LX1) / 2} y={LY0 + 30} anchor="middle" tone={3} size={9.5}>weight w</Label>
          <path d={curve} fill="none" stroke="var(--text-3)" strokeWidth={1.6} />
          <line x1={lx(2)} x2={lx(2)} y1={ly(0)} y2={ly(44)} stroke="var(--token)" strokeWidth={1} strokeDasharray="2 4" opacity={0.7} />
          <circle cx={lx(2)} cy={ly(0)} r={3} fill="var(--token)" />
          <Label x={lx(2)} y={ly(46)} anchor="middle" size={9} kind="token">bottom: w = 2, loss 0</Label>
        </Fade>

        {/* the slope at the ball */}
        <Fade show={st >= 3 && !offChart}>
          <line x1={lx(clampW(w - half))} y1={ly(L - g * half)} x2={lx(clampW(w + half))} y2={ly(L + g * half)}
            stroke={tanColor} strokeWidth={1.6} strokeDasharray="5 3" style={{ transition: 'all .55s var(--ease-out)' }} />
        </Fade>
        <Fade show={st === 3}>
          <Label x={bx + (g < 0 ? 14 : -14)} y={by - 12} anchor={g < 0 ? 'start' : 'end'} size={10} kind="compute" bold>{`slope ${g.toFixed(1)}`}</Label>
          {Math.abs(g) > 0.05 && (
            <line x1={bx} y1={by + 16} x2={bx + (g < 0 ? 30 : -30)} y2={by + 16} stroke={tanColor} strokeWidth={1.6} markerEnd="url(#tps-arrow)" />
          )}
          <Label x={bx + (g < 0 ? 36 : -36)} y={by + 19} anchor={g < 0 ? 'start' : 'end'} tone={3} size={9}>{Math.abs(g) > 0.05 ? 'downhill' : 'flat: done'}</Label>
        </Fade>

        {/* the trajectory */}
        <Fade show={descending}>
          {path.slice(0, idx + 1).map((pw, k) => {
            const inside = pw >= WMIN && pw <= WMAX && loss(pw) <= LMAX;
            const next = path[k + 1];
            const showSeg = k < idx && next !== undefined;
            return (
              <g key={k}>
                {inside && <circle cx={lx(pw)} cy={ly(loss(pw))} r={2.6} fill={diverging ? 'var(--error)' : 'var(--compute)'} opacity={0.75} />}
                {showSeg && (
                  <line x1={lx(clampW(pw))} y1={ly(loss(pw))} x2={lx(clampW(next))} y2={ly(loss(next))}
                    stroke={diverging ? 'var(--error)' : 'var(--compute)'} strokeWidth={1} strokeDasharray="3 3" opacity={0.7} />
                )}
              </g>
            );
          })}
          <Label x={LX1} y={58} anchor="end" size={9.5} kind={diverging ? 'error' : 'compute'}>
            {`w ← w − η·slope   (η = ${eta.toFixed(3)})`}
          </Label>
        </Fade>

        {/* the ball */}
        <Fade show={st >= 2}>
          <g className="v-move" style={{ transform: `translate(${bx}px, ${by}px)` }}>
            <circle r={7} fill={offChart ? 'var(--error)' : 'var(--text)'} stroke="var(--ground-2)" strokeWidth={2.5} />
          </g>
          {offChart && (
            <Label x={(LX0 + LX1) / 2} y={LY1 + 24} anchor="middle" size={9.5} kind="error" bold>
              {`off the chart: loss ${L > 9999 ? L.toExponential(1) : L.toFixed(0)}`}
            </Label>
          )}
        </Fade>
      </svg>
    </Stage>
  );
}
