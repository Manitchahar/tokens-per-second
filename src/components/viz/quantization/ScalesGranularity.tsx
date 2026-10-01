import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Arrow, Dim, Fade, Matrix } from '../lib/svg';
import { quantizeRow, relErr, metaFracInt } from './quant';

const STEPS = [
  'You cannot store -0.42 as 4 bits',
  'Scale and zero-point',
  'Quantize, then dequantize',
  'One scale for the whole matrix',
  'The outliers ruin the grid',
  'One scale per group',
  'One scale per channel',
  'The metadata you pay for it',
];
const CAPTIONS = [
  'A weight like -0.42 has as many digits as you write down. Four bits can name only sixteen values. Something has to choose which sixteen — and that choice is all quantization is.',
  'The fix is a scale and a zero-point. Divide by the scale to move the real range onto the small integer grid, add a zero-point to center it, round to the nearest integer. The integers are what you store.',
  'To use it, multiply back: v̂ = (q − z) × s. If s and z are stored, this step is exact and free. The information was lost at the rounding, not here — so round carefully and store s and z well.',
  'The cheapest assignment is one scale for the whole matrix: a single pair of numbers per layer. It is elegant, and it is what naive implementations do. Now watch what it costs.',
  'This matrix has a few large weights near 0.6 and many near 0.01. One scale must span 0.6, so the grid step is coarse, and every small weight rounds to zero. The errors are all in the small weights — which are most of the matrix.',
  'Split the matrix into groups and give each group its own scale. An outlier now only ruins its own group. This is what every good low-bit quantizer does, and the group size is the dial.',
  'Split by output channel instead, and each row gets its own scale. In an LLM, channels genuinely have different magnitudes, so this costs almost no metadata and removes most of the error.',
  'Each group costs one scale and one zero-point in FP16 — four bytes. At group size 128 that is 6.25% overhead; at 32 it is 25%. Metadata is the hidden tax on low-bit weights, and it is why group size cannot go to 1.',
];

/* A small, deterministic weight block that shows the outlier problem well. */
const W = [
  [0.03, -0.02, 0.05, 0.01, 0.04, -0.03, 0.02, 0.06],
  [-0.11, 0.08, -0.14, 0.09, -0.07, 0.12, -0.10, 0.05],
  [0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03, 0.01],
  [-0.61, 0.55, -0.58, 0.52, -0.60, 0.57, -0.59, 0.53],
  [0.04, -0.03, 0.06, -0.02, 0.03, -0.05, 0.04, -0.03],
  [0.09, -0.07, 0.11, -0.08, 0.06, -0.10, 0.08, -0.06],
  [0.02, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03],
  [0.28, -0.26, 0.31, -0.25, 0.27, -0.29, 0.30, -0.24],
];
const FLAT = W.flat();
const ROWS = W.length;
const COLS = W[0].length;
/** The widest magnitude in the matrix: the outlier that decides a per-tensor scale. */
const MAX_ABS = Math.max(...FLAT.map(Math.abs));
/** A typical "ordinary" weight, for scale: the many near-zero values. */
const SMALL = 0.02;

const MODES = [
  { id: 'tensor', label: 'Per tensor' },
  { id: 'group', label: 'Per group' },
  { id: 'channel', label: 'Per channel' },
] as const;
type Mode = (typeof MODES)[number]['id'];

const modeLabel = (m: Mode, group: number) =>
  m === 'tensor' ? 'one scale for the matrix' : m === 'group' ? `one scale per ${group}` : 'one scale per row (channel)';

/* ── the single-number demo ── */
const DEMO_X = 26;
const DEMO_W = 256;
const LINE_Y = 96;
/* ── the matrix pair ── */
const MX = 46;
const MY = 216;
const CELL = 15;
const GAP = 3;
const MW = COLS * (CELL + GAP) - GAP;

/** Dequantize the whole matrix under a mode, with real per-group scales, and report its error. */
function quantizeMatrix(mode: Mode, bits: number, group: number) {
  const half = (1 << bits) / 2 - 1;
  const snap = (v: number, maxAbs: number) => Math.max(-half, Math.min(half, Math.round(v / (maxAbs / half)))) * (maxAbs / half);
  const deq: number[] = [];
  if (mode === 'tensor') {
    for (const v of FLAT) deq.push(snap(v, MAX_ABS));
  } else if (mode === 'channel') {
    for (let r = 0; r < ROWS; r++) {
      const m = Math.max(...W[r].map(Math.abs)) || 1e-9;
      for (let c = 0; c < COLS; c++) deq.push(snap(W[r][c], m));
    }
  } else {
    const g = Math.max(1, Math.min(group, COLS));
    for (let r = 0; r < ROWS; r++) {
      for (let c0 = 0; c0 < COLS; c0 += g) {
        const end = Math.min(COLS, c0 + g);
        const m = Math.max(...W[r].slice(c0, end).map(Math.abs)) || 1e-9;
        for (let c = c0; c < end; c++) deq.push(snap(W[r][c], m));
      }
    }
  }
  return { deq, err: relErr(FLAT, deq) };
}

export default function ScalesGranularity() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [bits, setBits] = useState(4);
  const [mode, setMode] = useState<Mode>('group');
  const [group, setGroup] = useState(4);

  /* Steps 0–4 show per-tensor regardless of the controls; from step 5 the learner picks. */
  const activeMode: Mode = st < 5 ? 'tensor' : mode;
  const activeGroup = st < 5 ? COLS : group;
  const { deq, err } = useMemo(() => quantizeMatrix(activeMode, bits, activeGroup), [activeMode, bits, activeGroup]);
  const meta = metaFracInt(bits, activeGroup);
  const errKind = err > 0.15 ? ('error' as const) : err > 0.05 ? ('compute' as const) : ('token' as const);

  /* The single weight snapping onto its grid. */
  const v = -0.42;
  const snapped = quantizeRow([v], bits).deq[0];
  const levels = (1 << bits) / 2 - 1;
  const gridStep = (MAX_ABS / levels) * 2; // spacing implied by the per-tensor scale

  const cellFill = (r: number, c: number) => {
    const i = r * COLS + c;
    const trueV = W[r][c];
    const e = Math.abs(deq[i] - trueV);
    const rel = Math.abs(trueV) > 0.02 ? e / Math.abs(trueV) : e / SMALL;
    return rel > 0.4 ? ('error' as const) : rel > 0.12 ? ('dim:error' as const) : null;
  };

  return (
    <Stage
      rootRef={ref}
      title="Quantizing a weight matrix: scales, zero-points, and the grid"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'error', 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Matrix error', value: `${(err * 100).toFixed(1)}%`, kind: errKind, note: `relative ‖ΔW‖/‖W‖ at ${bits} bits, ${modeLabel(activeMode, activeGroup)}` },
          { label: 'Metadata overhead', value: `${(meta * 100).toFixed(1)}%`, kind: meta > 0.1 ? 'error' : 'text', note: `scales at group ${activeGroup}` },
          { label: 'Effective bits', value: `${((bits / 8) * (1 + meta) * 8).toFixed(2)} bits`, kind: 'memory', note: 'data plus scales, per weight' },
          { label: 'Grid spacing', value: gridStep.toFixed(3), kind: 'text', note: 'worst-case error is half of this' },
        ]} />
      }
      controls={
        <>
          <Slider label="Bits per weight" value={bits} min={2} max={8} step={1} onChange={setBits} format={(v) => `${v} bits · ${1 << v} values`} kind="compute" />
          {st >= 5 && <Slider label="Group size — weights sharing one scale" value={group} min={1} max={COLS} step={1} onChange={setGroup} format={(v) => `${v} per row`} kind="memory" />}
          {st >= 6 && <Segmented label="How scales are assigned" value={mode} options={MODES.map((m) => ({ value: m.id, label: m.label }))} onChange={setMode} />}
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A weight matrix quantized at a chosen bit width. Above, one weight value snapping onto the integer grid, with the scale and zero-point equations. Below, the original matrix beside its dequantized version with per-cell error shown in red, and the metadata overhead of the chosen group size.">
        {/* ── the single number on its grid ── */}
        <Fade show={st <= 3}>
          <Label x={DEMO_X} y={30} tone={2} size={11} bold>{`One weight, ${bits} bits`}</Label>
          <Label x={DEMO_X} y={48} size={10} tone={3}>{`true ${v.toFixed(3)}   stored ${snapped.toFixed(3)}   grid step ${(MAX_ABS / levels).toFixed(3)}`}</Label>
          <line x1={DEMO_X} x2={DEMO_X + DEMO_W} y1={LINE_Y} y2={LINE_Y} className="v-rule" />
          {Array.from({ length: 9 }, (_, i) => {
            const gx = DEMO_X + 12 + i * ((DEMO_W - 24) / 8);
            return (
              <g key={i}>
                <line x1={gx} x2={gx} y1={LINE_Y - 4} y2={LINE_Y + 4} className="v-rule faint" />
                <Label x={gx} y={LINE_Y + 16} anchor="middle" tone={3} size={8}>{((i - 4) * (MAX_ABS / levels)).toFixed(2)}</Label>
              </g>
            );
          })}
          <circle cx={DEMO_X + 12 + (4 - v / (MAX_ABS / levels)) * ((DEMO_W - 24) / 8)} cy={LINE_Y} r={3.5} fill="var(--error)" />
          <Label x={DEMO_X + 12 + (4 - v / (MAX_ABS / levels)) * ((DEMO_W - 24) / 8)} y={LINE_Y - 10} anchor="middle" size={9} kind="error">{`v=${v}`}</Label>
          <circle cx={DEMO_X + 12 + (4 - snapped / (MAX_ABS / levels)) * ((DEMO_W - 24) / 8)} cy={LINE_Y} r={4} fill="none" stroke="var(--compute)" strokeWidth={1.8} />
          <Label x={DEMO_X + 12 + (4 - snapped / (MAX_ABS / levels)) * ((DEMO_W - 24) / 8)} y={LINE_Y + 30} anchor="middle" size={9} kind="compute">{`v̂=${snapped.toFixed(2)}`}</Label>
          <Dim x1={DEMO_X + 12 + (4 - v / (MAX_ABS / levels)) * ((DEMO_W - 24) / 8)} x2={DEMO_X + 12 + (4 - snapped / (MAX_ABS / levels)) * ((DEMO_W - 24) / 8)} y={LINE_Y + 46} kind="error" label={`${Math.abs(v - snapped).toFixed(3)}`} />
        </Fade>

        {/* ── the two equations ── */}
        <Fade show={st >= 1 && st <= 3}>
          <g>
            <Label x={330} y={30} tone={2} size={11} bold>The two equations</Label>
            <Label x={330} y={62} size={11.5} tone={1} math>{'q = clamp(round(v/s) + z, 0, 2ᵇ⁻¹)'}</Label>
            <Label x={330} y={88} size={11.5} tone={1} math>{'v̂ = s · (q − z)'}</Label>
            <Label x={330} y={118} size={9.5} tone={3}>s = scale · z = zero-point · v̂ = dequantized</Label>
            <Label x={330} y={136} size={9.5} tone={3}>Only the integer q is stored, plus s and z.</Label>
          </g>
        </Fade>

        {/* ── per-tensor: the outlier problem, called out ── */}
        <Fade show={st === 4}>
          <g>
            <Dim x1={MX + 3 * (CELL + GAP)} x2={MX + 3 * (CELL + GAP) + CELL} y={MY - 10} y2={MY + ROWS * (CELL + GAP) - GAP + 10} vertical kind="compute" label={`outlier ${MAX_ABS.toFixed(2)}`} />
            <Label x={MX + 3 * (CELL + GAP) + CELL / 2} y={MY - 14} anchor="middle" size={9} kind="compute">{`the ${MAX_ABS.toFixed(2)} row sets the scale`}</Label>
          </g>
        </Fade>

        {/* ── the matrix pair ── */}
        <Label x={MX} y={MY - 12} tone={2} size={11} bold>{`Same matrix, ${modeLabel(activeMode, activeGroup)}`}</Label>
        <Matrix
          x={MX}
          y={MY}
          rows={ROWS}
          cols={COLS}
          cell={CELL}
          gap={GAP}
          label="original"
          fill={(r, c) => (Math.abs(W[r][c]) > MAX_ABS * 0.7 ? 'compute' : Math.abs(W[r][c]) < SMALL ? null : 'dim:text')}
        />
        <Arrow x1={MX + MW + 8} x2={MX + MW + 24} y1={MY + (ROWS * (CELL + GAP)) / 2} y2={MY + (ROWS * (CELL + GAP)) / 2} kind="compute" />
        <Matrix
          x={MX + MW + 32}
          y={MY}
          rows={ROWS}
          cols={COLS}
          cell={CELL}
          gap={GAP}
          label="dequantized"
          fill={cellFill}
        />
        {/* group boundaries, when we are grouping */}
        {activeMode === 'group' && activeGroup < COLS &&
          Array.from({ length: Math.floor((COLS - 1) / activeGroup) }, (_, i) => (
            <line key={i} x1={MX + MW + 32 + (i + 1) * activeGroup * (CELL + GAP) - GAP / 2} x2={MX + MW + 32 + (i + 1) * activeGroup * (CELL + GAP) - GAP / 2} y1={MY - 5} y2={MY + ROWS * (CELL + GAP) - GAP + 5} stroke="var(--memory)" strokeWidth={1.1} opacity={0.85} />
          ))}
        {/* per-channel boundaries */}
        {activeMode === 'channel' &&
          Array.from({ length: ROWS - 1 }, (_, i) => (
            <line key={i} x1={MX + MW + 32 - 5} x2={MX + MW + 32 + MW + 5} y1={MY + (i + 1) * (CELL + GAP) - GAP / 2} y2={MY + (i + 1) * (CELL + GAP) - GAP / 2} stroke="var(--memory)" strokeWidth={1.1} opacity={0.85} />
          ))}

        <Label x={MX} y={MY + ROWS * (CELL + GAP) + 20} tone={3} size={9.5}>
          {`Solid red: more than 40% of the value lost. Faded red: more than 12%. Error concentrates in the small weights, which are most of the matrix.`}
        </Label>

        {/* ── the metadata bill ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={MX} y={MY + ROWS * (CELL + GAP) + 46} tone={2} size={11} bold>{`The metadata bill, at group size ${activeGroup}`}</Label>
            <Span x={MX} y={MY + ROWS * (CELL + GAP) + 56} w={MW} h={14} kind="compute" label={`data ${bits} bits`} />
            <Span x={MX + MW + 6} y={MY + ROWS * (CELL + GAP) + 56} w={Math.max(6, MW * meta * 8)} h={14} kind="memory" label={`scales ${(meta * 100).toFixed(1)}%`} />
            <Label x={MX + MW + 6 + Math.max(6, MW * meta * 8) + 10} y={MY + ROWS * (CELL + GAP) + 67} size={10} kind="memory">
              {`${(((bits / 8) * (1 + meta)) * 8).toFixed(2)} bits per weight, all in`}
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}