import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade } from '../lib/svg';

// The smallest network with two layers: z = w1·x, a = ReLU(z), ŷ = w2·a, loss L = (ŷ − y)².
// Default numbers are the lesson's worked example: x = 1.5, w1 = 0.8, w2 = 0.5, target y = 2.
const W1 = 0.8;
const W2 = 0.5;
const Y = 2;
const ETA = 0.05;

function forward(x: number, w1: number, w2: number) {
  const z = w1 * x;
  const a = Math.max(0, z);
  const yh = w2 * a;
  const L = (yh - Y) ** 2;
  return { z, a, yh, L };
}
function backward(x: number, w1: number, w2: number) {
  const f = forward(x, w1, w2);
  const dyh = 2 * (f.yh - Y);
  const dw2 = dyh * f.a;
  const da = dyh * w2;
  const relu = f.z > 0 ? 1 : 0;
  const dz = da * relu;
  const dw1 = dz * x;
  return { ...f, dyh, dw2, da, relu, dz, dw1 };
}

const STEPS = [
  'Forward: compute left to right',
  'Score it: the loss',
  'Backward starts at the end',
  'Through the last weight',
  'Through the nonlinearity',
  'Back to the first weight',
  'Update both weights at once',
];
const CAPTIONS = [
  'Forward pass: x = 1.5 times w1 = 0.8 gives z = 1.2. ReLU leaves a positive number alone, so a = 1.2. Times w2 = 0.5 gives the prediction ŷ = 0.6.',
  'The target is 2, so the loss is (0.6 − 2)² = 1.96. Every value from the forward pass is kept: the backward pass needs them.',
  'Begin where the loss is. How fast does L change if ŷ moves? dL/dŷ = 2(ŷ − y) = −2.8. Negative: raising ŷ lowers the loss.',
  'ŷ = w2·a, so nudging w2 moves ŷ by a = 1.2 per unit: dL/dw2 = −2.8 × 1.2 = −3.36. Nudging a moves ŷ by w2 = 0.5: dL/da = −2.8 × 0.5 = −1.4, which we pass further back.',
  'ReLU’s slope is 1 when z is positive and 0 when it is negative. Here z = 1.2, so the gradient passes straight through: dL/dz = −1.4. Drag x below zero to see a ReLU block the gradient completely.',
  'z = w1·x, so dL/dw1 = dL/dz × x = −1.4 × 1.5 = −2.1. That is the whole chain rule: multiply the local slopes along the path from the loss back to the weight.',
  'Step both weights against their gradients with η = 0.05: w1 → 0.905, w2 → 0.668. Run forward again: ŷ = 0.907 and the loss falls from 1.96 to 1.20. One backward pass gave the direction for every weight at once.',
];

// geometry
const ROW = 118; // forward row (centre line)
const NODE_W = 52, OP_W = 56, H = 30;
const NX = { x: 36, z: 176, a: 312, yh: 448, L: 594 };
const OX = { w1: 106, relu: 244, w2: 380, loss: 521 };
const GROW = 196; // gradient row

function fmt2(v: number) { return (Math.abs(v) < 0.005 ? 0 : v).toFixed(2); }

export default function BackpropViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [x, setX] = useState(1.5);

  const b = backward(x, W1, W2);
  const nw1 = W1 - ETA * b.dw1;
  const nw2 = W2 - ETA * b.dw2;
  const after = forward(x, nw1, nw2);
  const updated = st >= 6;
  const cur = updated ? after : b;
  const w1 = updated ? nw1 : W1;
  const w2 = updated ? nw2 : W2;

  const node = (cx: number, label: string, value: string, show: boolean, kind: 'text' | 'error' = 'text') => (
    <Fade show={show}>
      <rect x={cx - NODE_W / 2} y={ROW - H / 2} width={NODE_W} height={H} rx={4}
        fill={kind === 'error' ? 'var(--error-dim)' : 'var(--ground-3)'} stroke={kind === 'error' ? 'var(--error)' : 'var(--rule-2)'} />
      <text x={cx} y={ROW + 4} textAnchor="middle" className="v-label t1" style={{ fontSize: 11.5, fontWeight: 700 }}>{value}</text>
      <Label x={cx} y={ROW - H / 2 - 6} anchor="middle" tone={3} size={9.5}>{label}</Label>
    </Fade>
  );
  const op = (cx: number, label: string) => (
    <g>
      <rect x={cx - OP_W / 2} y={ROW - 12} width={OP_W} height={24} rx={12} fill="none" stroke="var(--rule-2)" strokeDasharray="3 3" />
      <text x={cx} y={ROW + 4} textAnchor="middle" className="v-label t2" style={{ fontSize: 10 }}>{label}</text>
    </g>
  );
  const grad = (cx: number, label: string, value: number, show: boolean, w = 64) => (
    <Fade show={show}>
      <rect x={cx - w / 2} y={GROW - 13} width={w} height={26} rx={3} fill="var(--compute)" stroke="var(--compute)" />
      <text x={cx} y={GROW + 4.5} textAnchor="middle" className="v-label" style={{ fontSize: 11, fontWeight: 700, fill: '#0e1116' }}>{fmt2(value)}</text>
      <Label x={cx} y={GROW + 28} anchor="middle" size={9.5} kind="compute">{label}</Label>
    </Fade>
  );

  const chain = [
    '',
    `L = (ŷ − y)² = (${fmt2(b.yh)} − 2)² = ${b.L.toFixed(2)}`,
    `dL/dŷ = 2(ŷ − y) = 2 × (${fmt2(b.yh)} − 2) = ${fmt2(b.dyh)}`,
    `dL/dw2 = dL/dŷ × a = ${fmt2(b.dyh)} × ${fmt2(b.a)} = ${fmt2(b.dw2)};   dL/da = dL/dŷ × w2 = ${fmt2(b.da)}`,
    `dL/dz = dL/da × ReLU′(z) = ${fmt2(b.da)} × ${b.relu} = ${fmt2(b.dz)}`,
    `dL/dw1 = dL/dŷ · w2 · ReLU′(z) · x = ${fmt2(b.dyh)} × ${fmt2(W2)} × ${b.relu} × ${x.toFixed(2)} = ${fmt2(b.dw1)}`,
    `w1 ← ${W1} − 0.05 × (${fmt2(b.dw1)}) = ${nw1.toFixed(3)};   w2 ← ${W2} − 0.05 × (${fmt2(b.dw2)}) = ${nw2.toFixed(3)}`,
  ];

  return (
    <Stage
      rootRef={ref}
      title="Backpropagation: the chain rule, run backwards"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['text', 'forward values'], ['memory', 'weights'], ['compute', 'gradients'], ['error', 'loss']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'loss', value: cur.L.toFixed(3), kind: 'error', note: updated ? `was ${b.L.toFixed(3)}` : `target y = ${Y}` },
          ...(st >= 3 ? [{ label: 'dL/dw2', value: fmt2(b.dw2), kind: 'compute' as const, note: 'gradient of weight 2' }] : []),
          ...(st >= 5 ? [{ label: 'dL/dw1', value: fmt2(b.dw1), kind: 'compute' as const, note: b.relu ? 'gradient of weight 1' : 'blocked by ReLU' }] : []),
        ]} />
      }
      controls={<Slider label="Input x" value={x} min={-2} max={3} step={0.1} onChange={setX} format={(v) => v.toFixed(1)} />}
    >
      <svg viewBox="0 0 640 300" role="img" aria-label="A two-layer network drawn left to right: input x, multiply by weight w1, ReLU, multiply by weight w2, prediction, squared-error loss. Forward values fill in left to right, then gradients fill in right to left underneath, each the product of the local slopes along the way.">
        {/* weights, read from memory */}
        {([['w1', OX.w1, w1], ['w2', OX.w2, w2]] as const).map(([name, cx, v]) => (
          <g key={name}>
            <rect x={cx - 37} y={34} width={74} height={26} rx={3} fill="var(--memory)" stroke="var(--memory)" />
            <text x={cx} y={51} textAnchor="middle" className="v-label" style={{ fontSize: 11, fontWeight: 700, fill: '#0e1116' }}>{`${name} = ${v.toFixed(updated ? 3 : 1)}`}</text>
            <line x1={cx} y1={60} x2={cx} y2={ROW - 13} stroke="var(--memory)" strokeWidth={1.2} markerEnd="url(#tps-arrow)" />
          </g>
        ))}
        <Label x={OX.loss} y={51} anchor="middle" tone={2} size={10}>{`target y = ${Y}`}</Label>
        <line x1={OX.loss} y1={58} x2={OX.loss} y2={ROW - 13} stroke="var(--text-3)" strokeWidth={1} markerEnd="url(#tps-arrow)" />

        {/* forward chain */}
        {op(OX.w1, '× w1')}
        {op(OX.relu, 'ReLU')}
        {op(OX.w2, '× w2')}
        {op(OX.loss, '(ŷ − y)²')}
        <Arrow x1={NX.x + NODE_W / 2} y1={ROW} x2={OX.w1 - OP_W / 2 - 2} y2={ROW} muted />
        <Arrow x1={OX.w1 + OP_W / 2} y1={ROW} x2={NX.z - NODE_W / 2 - 2} y2={ROW} muted />
        <Arrow x1={NX.z + NODE_W / 2} y1={ROW} x2={OX.relu - OP_W / 2 - 2} y2={ROW} muted />
        <Arrow x1={OX.relu + OP_W / 2} y1={ROW} x2={NX.a - NODE_W / 2 - 2} y2={ROW} muted />
        <Arrow x1={NX.a + NODE_W / 2} y1={ROW} x2={OX.w2 - OP_W / 2 - 2} y2={ROW} muted />
        <Arrow x1={OX.w2 + OP_W / 2} y1={ROW} x2={NX.yh - NODE_W / 2 - 2} y2={ROW} muted />
        <Arrow x1={NX.yh + NODE_W / 2} y1={ROW} x2={OX.loss - OP_W / 2 - 2} y2={ROW} muted />
        <Arrow x1={OX.loss + OP_W / 2} y1={ROW} x2={NX.L - NODE_W / 2 - 2} y2={ROW} muted />

        {node(NX.x, 'input x', x.toFixed(2), true)}
        {node(NX.z, 'z', fmt2(cur.z), true)}
        {node(NX.a, 'a', fmt2(cur.a), true)}
        {node(NX.yh, 'prediction ŷ', fmt2(cur.yh), true)}
        {node(NX.L, 'loss L', cur.L.toFixed(2), st >= 1, 'error')}

        {/* gradients, right to left */}
        <Fade show={st >= 2}>
          <Label x={20} y={156} tone={3} size={9.5}>backward: how much the loss moves per unit nudge</Label>
        </Fade>
        {grad(NX.yh, 'dL/dŷ', b.dyh, st >= 2)}
        {grad(NX.a, 'dL/da', b.da, st >= 3)}
        {grad(NX.z, 'dL/dz', b.dz, st >= 4)}
        <Fade show={st >= 3}>
          <Arrow x1={NX.yh - 34} y1={GROW} x2={NX.a + 36} y2={GROW} kind="compute" />
        </Fade>
        <Fade show={st >= 4}>
          <Arrow x1={NX.a - 34} y1={GROW} x2={NX.z + 36} y2={GROW} kind="compute" />
          <Label x={OX.relu} y={GROW - 18} anchor="middle" size={9} kind="compute">{`× ReLU′ = ${b.relu}`}</Label>
        </Fade>
        <Fade show={st >= 3}>
          <Label x={OX.w2} y={GROW - 18} anchor="middle" size={9} kind="compute">× w2</Label>
        </Fade>

        {/* weight gradients, dropped next to the weights they belong to */}
        <Fade show={st >= 3}>
          <rect x={OX.w2 + 41} y={34} width={56} height={26} rx={3} fill="var(--compute)" stroke="var(--compute)" />
          <text x={OX.w2 + 69} y={51} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fontWeight: 700, fill: '#0e1116' }}>{fmt2(b.dw2)}</text>
          <Label x={OX.w2 + 69} y={26} anchor="middle" size={9} kind="compute">dL/dw2</Label>
        </Fade>
        <Fade show={st >= 5}>
          <rect x={OX.w1 + 41} y={34} width={56} height={26} rx={3} fill={b.relu ? 'var(--compute)' : 'none'} stroke="var(--compute)" strokeDasharray={b.relu ? undefined : '3 3'} />
          <text x={OX.w1 + 69} y={51} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fontWeight: 700, fill: b.relu ? '#0e1116' : 'var(--compute)' }}>{fmt2(b.dw1)}</text>
          <Label x={OX.w1 + 69} y={26} anchor="middle" size={9} kind="compute">dL/dw1</Label>
        </Fade>

        {/* the arithmetic for this step */}
        <line x1={20} x2={620} y1={244} y2={244} className="v-rule faint" />
        <Label x={20} y={266} tone={1} size={10.5}>{chain[st] || `forward: z = ${W1} × ${x.toFixed(2)} = ${fmt2(b.z)},  a = ReLU(z) = ${fmt2(b.a)},  ŷ = ${W2} × ${fmt2(b.a)} = ${fmt2(b.yh)}`}</Label>
        <Fade show={updated}>
          <Label x={20} y={286} size={10} kind="error">{`forward again: ŷ = ${fmt2(after.yh)}, loss ${b.L.toFixed(2)} → ${after.L.toFixed(2)}`}</Label>
        </Fade>
        <Fade show={st >= 4 && st < 6 && b.relu === 0}>
          <Label x={620} y={286} anchor="end" size={10} kind="error" bold>z ≤ 0: ReLU passes no gradient, w1 cannot learn from this example</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
