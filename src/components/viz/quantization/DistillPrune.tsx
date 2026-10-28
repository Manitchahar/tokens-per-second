import { useRef } from 'react';
import type { CSSProperties } from 'react';
import { Stage, Readout } from '../lib/Stage';
import type { Kind } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Chip } from '../lib/svg';
import { h100, llama8b, llama70b } from './quant';

/**
 * The other two levers, next to quantization.
 *
 * One grid of weight squares is put through four different treatments, then the
 * teacher/student picture for distillation, then the bill: bytes and tokens/s for
 * the three options the lesson compares. Every number is weight bytes, and every
 * token rate is bytes ÷ achieved bandwidth, so the learner can check the arithmetic.
 */

const STEPS = [
  'A model is a grid of weights',
  'Quantize: fewer bits per weight',
  'Prune: zero individual weights',
  '2:4: a pattern the hardware knows',
  'Structured: delete whole rows',
  'Distill: copy the distribution',
  'The bill, three ways',
];

const CAPTIONS = [
  'Every square is a block of weights. At inference all of them are read on every decode step, so the grid is the latency.',
  'Quantization keeps the grid and makes each square smaller: 16 bits becomes 4, and the model is a quarter of the bytes for the same number of parameters.',
  'Unstructured pruning sets individual weights to zero. The grid keeps its shape, the bytes are unchanged, and a dense matmul cannot skip the zeros — so nothing gets faster.',
  'Two zeros in every group of four is the one sparse pattern NVIDIA’s sparse tensor cores can skip. That is a compute win, which is why it shows up in prefill more than in decode.',
  'Structured pruning deletes whole rows, heads or layers. The matrix is genuinely smaller, so an ordinary dense kernel runs faster on any hardware — at a bigger quality cost per weight removed.',
  'Distillation does not compress the old model at all: it trains a new, smaller one on the teacher’s probability distribution. The shape of that distribution carries more information than the single correct answer.',
  'Three ways to serve the same capability: a 70B teacher that does not fit on one card, the same model at 4 bits, and an 8B student. Bytes of weights decide both the fit and the tokens per second.',
];

/* ── the shared arithmetic ── */

/** Achieved bandwidth on an H100: 3.35 TB/s peak × 0.8 MBU. */
const BW = h100.bw * 0.8;
const GB = 1e9;

const W16_8B = llama8b.params * 2;
const W4_8B = llama8b.params * 0.5 * 1.0625; // group-128 scales
const W16_70B = llama70b.params * 2;
const W4_70B = llama70b.params * 0.5 * 1.0625;
const W_STRUCT = W16_8B * (6 / 8); // 2 of 8 rows deleted

/** Tokens per second, one stream: bytes per step ÷ achieved bandwidth. */
const tps = (bytes: number) => BW / bytes;
/** One decode step in milliseconds, weights only. */
const stepMs = (bytes: number) => (bytes / BW) * 1e3;

const gb = (b: number) => `${(b / GB).toFixed(b / GB < 10 ? 1 : 0)} GB`;

/* ── grid geometry ── */

const COLS = 12;
const ROWS = 8;
const SLOT = 14;
const GAP = 3;
const GX = 40;
const GY = 66;
const GRID_H = ROWS * (SLOT + GAP) - GAP;

/** Deterministic, scattered 50% mask. No Math.random, so server and client agree. */
function hash(i: number) {
  let x = Math.imul(i + 1, 2654435761) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 2246822519) >>> 0;
  x ^= x >>> 13;
  return x;
}
const zeroUnstructured = (r: number, c: number) => hash(r * COLS + c) % 2 === 0;
const zero24 = (_r: number, c: number) => c % 4 === 1 || c % 4 === 3;

const ALL_ROWS = Array.from({ length: ROWS }, (_, i) => i);
/** Structured pruning: rows 1 and 4 deleted, the survivors close up. */
const KEPT_ROWS = [0, 2, 3, 5, 6, 7];

const T_ROWS = Array.from({ length: 8 }, (_, i) => i);

/* ── the per-step notes beside the grid ── */

const NOTES: {
  title: string;
  lines: string[];
  big: string;
  bigKind: Kind;
  sub: string;
  foot: string;
}[] = [
  {
    title: 'Every square is a block of weights',
    lines: ['Eight billion of them, read in full', 'on every token of the answer.'],
    big: gb(W16_8B),
    bigKind: 'memory',
    sub: '8.03B weights at 2 bytes each',
    foot: '16 bits per weight. Decode reads every square, every token, however many users are waiting.',
  },
  {
    title: 'Quantize: fewer bits per weight',
    lines: ['The grid does not change. Each number', 'is stored on a coarser grid.'],
    big: gb(W4_8B),
    bigKind: 'memory',
    sub: '4.25 effective bits per weight',
    foot: 'Same parameter count, a quarter of the bytes. This is everything Module 11 has done so far.',
  },
  {
    title: 'Prune: zero individual weights',
    lines: ['Half the weights are set to zero.', 'The grid, the bytes and the matmul', 'are exactly as large as before.'],
    big: gb(W16_8B),
    bigKind: 'idle',
    sub: 'unchanged: zeros are still stored',
    foot: 'A dense kernel cannot skip a zero, so unstructured pruning buys quality risk and no speed.',
  },
  {
    title: '2:4: a pattern hardware knows',
    lines: ['Two zeros in every four weights.', 'Ampere and newer sparse tensor', 'cores can skip them.'],
    big: 'up to 2×',
    bigKind: 'compute',
    sub: 'peak math, on compute-bound work',
    foot: 'The zeros are still there for a dense kernel. Only 2:4-aware hardware and kernels get the speedup.',
  },
  {
    title: 'Structured: delete whole rows',
    lines: ['Whole rows, heads or layers removed.', 'The matrix is genuinely smaller, so', 'any dense kernel runs faster.'],
    big: gb(W_STRUCT),
    bigKind: 'memory',
    sub: '6 of 8 rows left, any hardware',
    foot: 'Two rows deleted: a quarter of the weights gone. Real speedup everywhere, and the biggest quality risk.',
  },
];

const READOUT: { label: string; value: string; kind: Kind; note: string }[][] = [
  [
    { label: 'Weights', value: gb(W16_8B), kind: 'memory', note: '8.03B × 2 bytes' },
    { label: 'One decode step', value: `${stepMs(W16_8B).toFixed(1)} ms`, kind: 'compute', note: `${gb(W16_8B)} ÷ 2.68 TB/s` },
    { label: 'Single stream', value: `${tps(W16_8B).toFixed(0)} tok/s`, kind: 'token', note: 'one user, no batching' },
  ],
  [
    { label: 'Weights', value: gb(W4_8B), kind: 'memory', note: '4 bits + group-128 scales' },
    { label: 'One decode step', value: `${stepMs(W4_8B).toFixed(1)} ms`, kind: 'compute', note: 'the same bytes → time rule' },
    { label: 'Single stream', value: `${tps(W4_8B).toFixed(0)} tok/s`, kind: 'token', note: '3.8× the BF16 rate' },
  ],
  [
    { label: 'Weights stored', value: gb(W16_8B), kind: 'error', note: '47 of 96 squares zero, none skipped' },
    { label: 'One decode step', value: `${stepMs(W16_8B).toFixed(1)} ms`, kind: 'compute', note: 'identical to BF16' },
    { label: 'Single stream', value: `${tps(W16_8B).toFixed(0)} tok/s`, kind: 'token', note: 'no gain without sparse kernels' },
  ],
  [
    { label: 'Weights stored', value: gb(W16_8B), kind: 'memory', note: 'dense layout keeps the zeros' },
    { label: 'Prefill math', value: 'up to 2×', kind: 'compute', note: '2:4 sparse tensor cores' },
    { label: 'Decode', value: `${tps(W16_8B).toFixed(0)} tok/s`, kind: 'token', note: 'unchanged unless stored compressed' },
  ],
  [
    { label: 'Weights', value: gb(W_STRUCT), kind: 'memory', note: 'two of eight rows deleted' },
    { label: 'One decode step', value: `${stepMs(W_STRUCT).toFixed(1)} ms`, kind: 'compute', note: 'dense kernel, any GPU' },
    { label: 'Single stream', value: `${tps(W_STRUCT).toFixed(0)} tok/s`, kind: 'token', note: '1.33× from the same bytes rule' },
  ],
  [
    { label: 'Teacher, 70B BF16', value: gb(W16_70B), kind: 'error', note: 'does not fit one 80 GB card' },
    { label: 'Student, 8B BF16', value: gb(W16_8B), kind: 'memory', note: 'fits with room for the KV cache' },
    { label: 'Student, single stream', value: `${tps(W16_8B).toFixed(0)} tok/s`, kind: 'token', note: 'same bytes rule as the teacher' },
    { label: 'Soft-label KL', value: '0.01 nats', kind: 'text', note: 'what the student actually minimises' },
  ],
  [
    { label: '70B BF16 teacher', value: '141 GB', kind: 'error', note: 'two GPUs, or not at all' },
    { label: '70B INT4', value: '37.5 GB · 71 tok/s', kind: 'memory', note: 'one GPU, 4.25 effective bits' },
    { label: '8B BF16 student', value: '16.1 GB · 167 tok/s', kind: 'token', note: 'fastest, and a different model' },
  ],
];

/* ── the cell grid ── */

function Grid({
  x,
  y,
  rowKeys,
  cols,
  slot,
  size,
  gap,
  fillOf,
}: {
  x: number;
  y: number;
  rowKeys: number[];
  cols: number;
  slot: number;
  size: number;
  gap: number;
  fillOf: (r: number, c: number) => { fill: string; stroke: string };
}) {
  const cells = [];
  for (let ri = 0; ri < rowKeys.length; ri++) {
    for (let c = 0; c < cols; c++) {
      const f = fillOf(rowKeys[ri], c);
      const cx = x + c * (slot + gap) + (slot - size) / 2;
      const cy = y + ri * (slot + gap) + (slot - size) / 2;
      cells.push(
        <rect
          key={`${rowKeys[ri]}-${c}`}
          x={cx}
          y={cy}
          width={size}
          height={size}
          rx={1.5}
          fill={f.fill}
          stroke={f.stroke}
          strokeWidth={0.9}
          className="v-cell"
          style={{ transition: 'x .55s var(--ease-out), y .55s var(--ease-out), width .55s var(--ease-out), height .55s var(--ease-out)' } as CSSProperties}
        />,
      );
    }
  }
  return <g>{cells}</g>;
}

const LIVE = { fill: 'var(--memory)', stroke: 'var(--memory)' };
const ZERO = { fill: 'url(#tps-hatch)', stroke: 'var(--idle)' };

/* ── distillation bars ── */

const TOKENS = ['Paris', 'Lyon', 'Marseille'];
const TEACHER_T1 = [0.844, 0.114, 0.042];
const TEACHER_T2 = [0.629, 0.231, 0.140];
const STUDENT_T2 = [0.556, 0.276, 0.168];

const BAR_BASE = 218;
const BAR_MAX = 104; // pixels for probability 1.0

/* ── the bill ── */

const BILL = [
  { label: '70B FP16 teacher', bytes: W16_70B, tps: tps(W16_70B), fits: false },
  { label: '70B INT4', bytes: W4_70B, tps: tps(W4_70B), fits: true },
  { label: '8B BF16 student', bytes: W16_8B, tps: tps(W16_8B), fits: true },
];
const BILL_SCALE = 160 * GB; // full width of the byte axis
const BYTES_X = 160;
const BYTES_W = 270;
const TPS_X = 500;
const TPS_W = 110;
const TPS_SCALE = 180;

export default function DistillPrune() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const size = st === 1 ? 7 : SLOT;
  const rowKeys = st >= 4 && st <= 4 ? KEPT_ROWS : ALL_ROWS;
  const cells = (r: number, c: number) => {
    if (st >= 2 && st <= 2 && zeroUnstructured(r, c)) return ZERO;
    if (st >= 3 && st <= 3 && zero24(r, c)) return ZERO;
    return LIVE;
  };

  const note = NOTES[Math.min(st, NOTES.length - 1)];
  const readout = READOUT[Math.min(st, READOUT.length - 1)];

  return (
    <Stage
      rootRef={ref}
      title="Four ways to shrink a model, and what each one does to bytes per token"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', ['idle', 'pruned to zero'], 'compute', 'token', 'error']}
      {...s}
      readout={<Readout items={readout} />}
    >
      <svg
        viewBox="0 0 640 384"
        role="img"
        aria-label="A grid of weight squares put through four treatments: quantized so each square shrinks to a quarter of its area, unstructured pruning that hatches random squares in place, a regular 2:4 pattern of hatched squares, and structured pruning that removes two whole rows and makes the grid shorter. Then a teacher grid feeding a small student grid beside two sets of probability bars, and finally a bar chart of weight gigabytes and single-stream tokens per second for a 70B teacher, a 70B INT4 model and an 8B student."
      >
        {/* ── steps 0-4: the grid ── */}
        <Fade show={st <= 4}>
          <Label x={24} y={44} tone={2} size={11} bold>
            One square stands for a block of weights
          </Label>
          <Label x={616} y={44} anchor="end" tone={3} size={9.5}>
            Llama-3.1-8B · one H100 · single stream
          </Label>

          <Grid x={GX} y={GY} rowKeys={rowKeys} cols={COLS} slot={SLOT} size={size} gap={GAP} fillOf={cells} />

          <Label x={24} y={GY + GRID_H + 26} tone={3} size={9.5}>
            {note.foot}
          </Label>
          <Label x={24} y={GY + GRID_H + 42} tone={3} size={9}>
            {'Grey hatched = a weight set to zero. Amber = compute. Blue = memory traffic.'}
          </Label>

          {/* the note beside the grid */}
          <Label x={296} y={70} tone={1} size={11.5} bold mono={false}>
            {note.title}
          </Label>
          {note.lines.map((line, i) => (
            <Label key={i} x={296} y={92 + i * 16} tone={2} size={10} mono={false}>
              {line}
            </Label>
          ))}
          <Label x={296} y={178} size={24} bold kind={note.bigKind}>
            {note.big}
          </Label>
          <Label x={296} y={196} tone={3} size={9.5}>
            {note.sub}
          </Label>

          <Fade show={st === 4}>
            <Chip x={296} y={210} w={176} text="dense kernel: really smaller" kind="token" size={9.5} h={20} />
          </Fade>
          <Fade show={st === 2}>
            <Chip x={296} y={210} w={176} text="dense kernel: same bytes" kind="error" size={9.5} h={20} />
          </Fade>
          <Fade show={st === 3}>
            <Chip x={296} y={210} w={196} text="sparse tensor core: 2× math" kind="compute" size={9.5} h={20} />
          </Fade>
        </Fade>

        {/* ── step 5: distillation ── */}
        <Fade show={st === 5}>
          <Label x={24} y={44} tone={2} size={11} bold>
            Distillation: train a small model to copy a big one
          </Label>
          <Label x={616} y={44} anchor="end" tone={3} size={9.5}>
            teacher 70B · student 8B · shapes are illustrative
          </Label>

          <Label x={34} y={60} tone={3} size={9.5}>
            teacher · 70B · 141 GB
          </Label>
          <Grid
            x={34}
            y={64}
            rowKeys={T_ROWS}
            cols={12}
            slot={8}
            size={6.5}
            gap={1}
            fillOf={() => LIVE}
          />

          <path d="M52,146 L52,168" stroke="var(--token)" strokeWidth={1.4} markerEnd="url(#tps-arrow)" />
          <Label x={62} y={162} size={9} tone={3}>
            outputs, or logits
          </Label>

          <Grid
            x={34}
            y={178}
            rowKeys={[0, 1, 2, 3, 4]}
            cols={8}
            slot={11}
            size={9}
            gap={2}
            fillOf={() => LIVE}
          />
          <Label x={146} y={204} size={9.5} kind="memory">
            student · 8B · 16.1 GB
          </Label>
          <Label x={146} y={220} tone={3} size={9}>
            fits on one card, and
          </Label>
          <Label x={146} y={234} tone={3} size={9}>
            reads 8.8× fewer bytes
          </Label>

          {/* probability bars */}
          <Label x={252} y={62} tone={2} size={10} bold mono={false}>
            Next token after “The capital of France is”
          </Label>
          <Label x={252} y={78} tone={3} size={8.5}>
            filled = teacher · outline = student · both at T = 2
          </Label>
          <line x1={252} x2={616} y1={BAR_BASE} y2={BAR_BASE} className="v-rule" />
          <Label x={244} y={BAR_BASE + 3} anchor="end" tone={3} size={8.5}>
            0
          </Label>

          {TOKENS.map((tok, i) => {
            const cx = 322 + i * 96;
            const hT = TEACHER_T2[i] * BAR_MAX;
            const hS = STUDENT_T2[i] * BAR_MAX;
            return (
              <g key={tok}>
                <rect x={cx - 18} y={BAR_BASE - hT} width={16} height={hT} rx={1.5} fill="var(--token)" opacity={0.92} />
                <rect x={cx + 2} y={BAR_BASE - hS} width={16} height={Math.max(2, hS)} rx={1.5} fill="none" stroke="var(--token)" strokeWidth={1.4} />
                <Label x={cx - 10} y={BAR_BASE - hT - 5} anchor="middle" size={8.5} tone={2}>
                  {TEACHER_T2[i].toFixed(2)}
                </Label>
                <Label x={cx} y={BAR_BASE + 14} anchor="middle" size={9.5} tone={3}>
                  {tok}
                </Label>
              </g>
            );
          })}

          <Label x={24} y={258} tone={3} size={9.5}>
            {'At T = 1 the teacher says 0.84 / 0.11 / 0.04. At T = 2 it says 0.63 / 0.23 / 0.14.'}
          </Label>
          <Label x={24} y={274} tone={3} size={9.5}>
            {'A hard label only says “Paris”. The soft label ranks Lyon above Marseille, with the size of the gap.'}
          </Label>
          <Label x={24} y={290} tone={3} size={9.5}>
            {'Fitting that shape is what transfers knowledge — not the single correct token.'}
          </Label>
        </Fade>

        {/* ── step 6: the bill ── */}
        <Fade show={st === 6}>
          <Label x={24} y={44} tone={2} size={11} bold>
            The bill: bytes of weights, and the tokens those bytes buy
          </Label>
          <Label x={616} y={44} anchor="end" tone={3} size={9.5}>
            one H100-class GPU · 2.68 TB/s achieved · single stream
          </Label>

          <Label x={BYTES_X} y={196} tone={3} size={9}>
            weights on the GPU (GB)
          </Label>
          <Label x={TPS_X} y={196} tone={3} size={9}>
            decode tokens/s
          </Label>

          {/* the 80 GB wall */}
          <line
            x1={BYTES_X + (80 / 160) * BYTES_W}
            x2={BYTES_X + (80 / 160) * BYTES_W}
            y1={200}
            y2={294}
            stroke="var(--error)"
            strokeWidth={1.2}
            strokeDasharray="4 4"
            opacity={0.8}
          />
          <Label x={BYTES_X + (80 / 160) * BYTES_W} y={306} anchor="middle" kind="error" size={9}>
            one 80 GB card
          </Label>

          {BILL.map((b, i) => {
            const y = 206 + i * 34;
            const wpx = (b.bytes / BILL_SCALE) * BYTES_W;
            const wall = (80 / 160) * BYTES_W;
            const tpx = (b.tps / TPS_SCALE) * TPS_W;
            return (
              <g key={b.label}>
                <Label x={BYTES_X - 10} y={y + 11} anchor="end" tone={1} size={10}>
                  {b.label}
                </Label>
                <rect x={BYTES_X} y={y} width={Math.min(wpx, wall)} height={16} rx={2} fill="var(--memory)" opacity={0.92} />
                {!b.fits && (
                  <rect
                    x={BYTES_X + wall}
                    y={y}
                    width={wpx - wall}
                    height={16}
                    rx={2}
                    fill="url(#tps-hatch-red)"
                    stroke="var(--error)"
                    strokeWidth={1.1}
                  />
                )}
                <Label x={BYTES_X + BYTES_W + 6} y={y + 11} size={9.5} kind={b.fits ? 'text' : 'error'}>
                  {`${(b.bytes / GB).toFixed(b.bytes / GB < 100 ? 1 : 0)} GB`}
                </Label>
                <rect x={TPS_X} y={y} width={Math.max(2, tpx)} height={16} rx={2} fill="var(--token)" opacity={0.9} />
                <Label x={TPS_X + TPS_W + 6} y={y + 11} size={9.5} kind="token">
                  {`${b.tps.toFixed(0)}`}
                </Label>
              </g>
            );
          })}

          <Label x={24} y={326} tone={3} size={9}>
            {'Weights: 70.6e9 × 2 = 141 GB · 70.6e9 × 0.5 × 1.0625 = 37.5 GB · 8.03e9 × 2 = 16.1 GB'}
          </Label>
          <Label x={24} y={342} tone={3} size={9}>
            {'Tokens/s ≈ 2.68 TB/s ÷ weight bytes: 19 (if the teacher fit), 71, 167 — one token in flight.'}
          </Label>
          <Label x={24} y={358} tone={3} size={9}>
            {'The student wins on both axes. It is also the only one of the three that is a different model.'}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
