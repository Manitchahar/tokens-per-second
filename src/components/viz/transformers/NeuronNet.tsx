import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, Matrix, fmt } from '../lib/svg';
import { arch, mlpParams, attnParams, embedParams, headParams, normParams, weightBytes } from './modelShapes';

// One neuron of a real layer, in numbers you can check by hand.
// x is one token's 4-number toy vector; w is one row of the weight matrix.
const X = [0.4, -1.2, 2.0, 0.7];
const W = [1.1, 0.4, -0.8, 0.9];
const B = 0.25;

type ActId = 'relu' | 'gelu' | 'silu';
const ACTS: Record<ActId, { label: string; note: string; f: (z: number) => number }> = {
  relu: { label: 'ReLU', note: 'max(0, z)', f: (z) => Math.max(0, z) },
  gelu: { label: 'GELU', note: 'a smooth ReLU', f: (z) => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z * z * z))) },
  silu: { label: 'SiLU', note: 'z · σ(z)', f: (z) => z / (1 + Math.exp(-z)) },
};

// Llama-3.1-8B, straight from its published config.
const M = arch('llama8b');
const TOTAL = M.params;
const PARTS = [
  { name: 'MLP · 3 matrices × 32 layers', v: M.layers * mlpParams(M), kind: 'compute' as const },
  { name: 'Attention q,k,v,o · 32 layers', v: M.layers * attnParams(M), kind: 'compute' as const },
  { name: 'Embedding table', v: embedParams(M), kind: 'memory' as const },
  { name: 'Output head (lm_head)', v: headParams(M), kind: 'memory' as const },
  { name: 'RMSNorm gains', v: M.layers * normParams(M) + M.hidden, kind: 'text' as const },
];

const STEPS = [
  'One number in',
  'Multiply by a weight',
  'Add them all up',
  'Add the bias',
  'Squash it',
  'A layer is a stack of them',
  'Where 8.03 billion weights live',
  'Training moves them. Inference reads them.',
];
const CAPTIONS = [
  'A neuron takes a vector in and gives one number out. Here a token arrives as four numbers: 0.4, −1.2, 2.0, 0.7.',
  'Each incoming number is multiplied by its own weight. Those weights were learned once, during training. From now on they are just numbers sitting in memory.',
  'A dot product: multiply matching coordinates and add. 0.44 − 0.48 − 1.60 + 0.63 = −1.01.',
  'Then add a learned bias, one number per neuron. z = −0.76.',
  'Squash z with a nonlinearity. Without one, stacking 32 layers would collapse into a single linear map. Llama and Qwen2 use SiLU.',
  'A layer is a whole row of these at once. q_proj is 4,096 × 4,096: 4,096 neurons, each with 4,096 weights. 16.8 million numbers.',
  'Do that arithmetic for a whole model and you get where the weights are. Two thirds of Llama-3.1-8B sit in MLP matrices, not in attention.',
  'At inference nothing is learned. Every one of those numbers is read from HBM on every decode step: 16.1 GB, and that single fact sets the speed of the whole model.',
];

const COL = [34, 80, 126, 172];
const BOXW = 42;
const BOXH = 24;
const MAT = { x: 34, y: 244, cell: 10, gap: 2, n: 8, row: 3 };
const BARX = 214;
const BARW = 406;

export default function NeuronNet() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [x3, setX3] = useState(X[2]);
  const [act, setAct] = useState<ActId>('silu');

  const x = [X[0], X[1], x3, X[3]];
  const prods = x.map((v, i) => v * W[i]);
  const dot = prods.reduce((a, b) => a + b, 0);
  const z = dot + B;
  const a = ACTS[act];
  const out = a.f(z);
  const actOn = st >= 4;
  const matOn = st >= 5;
  const barOn = st >= 6;
  const mlpShare = (M.layers * mlpParams(M)) / TOTAL;

  let acc = 0;
  const segs = PARTS.map((p) => {
    const w = (p.v / TOTAL) * BARW;
    const x0 = BARX + acc;
    acc += w;
    return { ...p, x0, w, pct: p.v / TOTAL };
  });
  const swatch = (kind: 'compute' | 'memory' | 'text', x: number, y: number) => (
    <rect x={x} y={y} width={11} height={8} rx={1.5}
      fill={kind === 'text' ? 'var(--rule)' : kind === 'compute' ? 'var(--compute)' : 'var(--memory)'} />
  );

  return (
    <Stage
      rootRef={ref}
      title="A neuron is a dot product, a bias and a squash"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory']}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'dot product', value: dot.toFixed(2), kind: 'compute', note: `Σ xᵢ·wᵢ, then + bias ${B.toFixed(2)}` },
            { label: 'neuron output', value: actOn ? out.toFixed(3) : 'waiting', kind: 'token', note: actOn ? a.label : 'not computed yet' },
            ...(barOn
              ? [
                  { label: 'share in MLPs', value: `${(mlpShare * 100).toFixed(1)}%`, kind: 'compute' as const, note: '5.637B of 8.03B' },
                  { label: 'weights in BF16', value: fmt.gb(weightBytes(M)), kind: 'memory' as const, note: 'read every decode step' },
                ]
              : matOn
                ? [{ label: 'q_proj, one layer', value: '16.8M weights', kind: 'compute' as const, note: '4,096 neurons × 4,096' }]
                : []),
          ]}
        />
      }
      controls={
        <>
          <Slider label="One input value (x₃)" value={x3} min={-2.5} max={2.5} step={0.1} onChange={setX3} format={(v) => v.toFixed(1)} />
          {actOn && (
            <Segmented
              label="Nonlinearity"
              value={act}
              options={[{ value: 'relu' as ActId, label: 'ReLU' }, { value: 'gelu' as ActId, label: 'GELU' }, { value: 'silu' as ActId, label: 'SiLU' }]}
              onChange={setAct}
            />
          )}
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="One neuron multiplying four inputs by four weights, summing them, adding a bias and applying a nonlinearity; below, a weight matrix grid and a stacked bar showing that about 70 percent of Llama-3.1-8B's weights live in MLP matrices.">
        {/* the two panels that arrive later, outlined from the start */}
        <rect x={24} y={214} width={168} height={154} rx={4} fill="none" stroke="var(--rule)" strokeWidth={1} strokeDasharray="4 4" className="v-fade" />
        <rect x={204} y={214} width={416} height={154} rx={4} fill="none" stroke="var(--rule)" strokeWidth={1} strokeDasharray="4 4" className="v-fade" />
        <Label x={34} y={234} tone={3} size={10}>a whole layer</Label>
        <Label x={214} y={234} tone={3} size={10}>the whole model</Label>

        {/* ── one neuron ── */}
        {COL.map((cx, i) => (
          <g key={i}>
            <rect x={cx} y={38} width={BOXW} height={BOXH} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1} />
            <text x={cx + BOXW / 2} y={54} textAnchor="middle" className="v-label t1" style={{ fontSize: 11.5 }}>{x[i].toFixed(1)}</text>
            <Fade show={st >= 1}>
              <text x={cx + BOXW / 2} y={70} textAnchor="middle" className="v-label t3" style={{ fontSize: 10 }}>×</text>
              <rect x={cx} y={78} width={BOXW} height={BOXH} rx={3} fill="var(--memory)" stroke="var(--memory)" strokeWidth={1} />
              <text x={cx + BOXW / 2} y={94} textAnchor="middle" className="v-label" style={{ fontSize: 11.5, fontWeight: 600, fill: '#0e1116' }}>{W[i].toFixed(1)}</text>
              <text x={cx + BOXW / 2} y={118} textAnchor="middle" className="v-label t3" style={{ fontSize: 10 }}>=</text>
              <rect x={cx} y={126} width={BOXW} height={BOXH} rx={3}
                fill={prods[i] < 0 ? 'none' : 'var(--compute)'} stroke="var(--compute)" strokeWidth={1}
                strokeDasharray={prods[i] < 0 ? '3 3' : undefined} />
              <text x={cx + BOXW / 2} y={142} textAnchor="middle" className="v-label"
                style={{ fontSize: 11, fill: prods[i] < 0 ? 'var(--text-3)' : 'var(--text)' }}>{prods[i].toFixed(2)}</text>
              <text x={cx + BOXW / 2} y={166} textAnchor="middle" className="v-label t3" style={{ fontSize: 9.5 }}>{`x${i + 1}`}</text>
            </Fade>
          </g>
        ))}

        <Fade show={st >= 2}>
          <Arrow x1={220} y1={138} x2={242} y2={114} kind="compute" />
          <rect x={244} y={90} width={58} height={44} rx={3} fill="var(--compute)" stroke="var(--compute)" strokeWidth={1} />
          <text x={273} y={107} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fontWeight: 600, fill: '#0e1116' }}>Σ</text>
          <text x={273} y={124} textAnchor="middle" className="v-label" style={{ fontSize: 12, fontWeight: 700, fill: '#0e1116' }}>{dot.toFixed(2)}</text>
          <Label x={273} y={152} anchor="middle" tone={3} size={9}>dot product</Label>
        </Fade>

        <Fade show={st >= 3}>
          <Arrow x1={304} y1={112} x2={318} y2={112} kind="text" />
          <rect x={322} y={90} width={74} height={44} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1} />
          <text x={359} y={107} textAnchor="middle" className="v-label t3" style={{ fontSize: 9.5 }}>{`+ bias ${B.toFixed(2)}`}</text>
          <text x={359} y={124} textAnchor="middle" className="v-label t1" style={{ fontSize: 12, fontWeight: 700 }}>{`z = ${z.toFixed(2)}`}</text>
          <Label x={359} y={152} anchor="middle" tone={3} size={9}>pre-activation</Label>
        </Fade>

        <Fade show={actOn}>
          <Arrow x1={398} y1={112} x2={412} y2={112} kind="compute" />
          <rect x={416} y={90} width={80} height={44} rx={3} fill="var(--compute)" stroke="var(--compute)" strokeWidth={1} />
          <text x={456} y={107} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fontWeight: 600, fill: '#0e1116' }}>{a.label}</text>
          <text x={456} y={124} textAnchor="middle" className="v-label" style={{ fontSize: 12, fontWeight: 700, fill: '#0e1116' }}>{out.toFixed(3)}</text>
          <Arrow x1={498} y1={112} x2={512} y2={112} kind="token" />
          <circle cx={532} cy={112} r={18} fill="var(--token)" stroke="var(--token)" strokeWidth={1.4} />
          <Label x={554} y={108} tone={2} size={10}>one neuron</Label>
          <Label x={554} y={122} tone={3} size={9}>{a.note}</Label>
        </Fade>

        <Fade show={actOn}>
          <Label x={34} y={194} tone={2} size={10.5}>no squash: stacking 32 layers would be one big linear map</Label>
        </Fade>

        {/* ── a whole layer ── */}
        <Fade show={matOn}>
          <Label x={34} y={234} tone={1} size={10.5} bold>q_proj [4,096 × 4,096]</Label>
          <Matrix x={MAT.x} y={MAT.y} rows={MAT.n} cols={MAT.n} cell={MAT.cell} gap={MAT.gap}
            fill={(r) => (r === MAT.row ? 'compute' : 'dim:memory')} />
          <rect x={MAT.x - 2} y={MAT.y + MAT.row * (MAT.cell + MAT.gap) - 2}
            width={MAT.n * (MAT.cell + MAT.gap) - MAT.gap + 4} height={MAT.cell + 4} rx={2}
            fill="none" stroke="var(--token)" strokeWidth={1.4} />
          <Label x={136} y={290} tone={3} size={9}>row 3</Label>
          <Label x={136} y={302} tone={3} size={9}>= our</Label>
          <Label x={136} y={314} tone={3} size={9}>neuron</Label>
          <Label x={34} y={356} tone={3} size={9.5}>{'4,096 rows · 16.8M weights'}</Label>
        </Fade>

        {/* ── where all the weights are ── */}
        <Fade show={barOn}>
          <Label x={214} y={234} tone={1} size={10.5} bold>Llama-3.1-8B · 8.03B weights</Label>
          {segs.map((sg) => (
            <rect key={sg.name} x={sg.x0} y={242} width={Math.max(0.6, sg.w)} height={28} rx={1.5}
              fill={sg.kind === 'text' ? 'var(--rule)' : sg.kind === 'compute' ? 'var(--compute)' : 'var(--memory)'}
              stroke={sg.kind === 'text' ? 'var(--rule-2)' : sg.kind === 'compute' ? 'var(--compute)' : 'var(--memory)'}
              strokeWidth={0.8} className="v-cell" />
          ))}
          <text x={BARX + segs[0].w / 2} y={260} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: '#0e1116' }}>
            {`${(segs[0].pct * 100).toFixed(0)}%`}
          </text>
          {PARTS.map((p, i) => (
            <g key={p.name}>
              {swatch(p.kind, 214, 282 + i * 14)}
              <text x={232} y={290 + i * 14} className="v-label t2" style={{ fontSize: 9.5 }}>{p.name}</text>
              <text x={612} y={290 + i * 14} textAnchor="end" className="v-label t1" style={{ fontSize: 9.5 }}>
                {`${(p.v / 1e9).toFixed(3)}B · ${((p.v / TOTAL) * 100).toFixed(1)}%`}
              </text>
            </g>
          ))}
          <Fade show={st >= 7}>
            <Label x={612} y={358} anchor="end" tone={1} size={10} kind="memory" bold>
              {`read all ${fmt.gb(weightBytes(M))} of it, every step`}
            </Label>
          </Fade>
        </Fade>
      </svg>
    </Stage>
  );
}
