import { useRef } from 'react';
import { Stage, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Move, Matrix, Arrow, Dim, fmt } from '../lib/svg';

// The point: a Triton kernel is a grid of independent programs. You never write
// a thread, a block or a loop — you write "one program handles this slice", and
// the grid tells it which slice. That inversion is what makes the model learnable.

const N = 128; // elements in the output
const STEPS = [
  'One program, not one thread',
  'The grid is launched once',
  'Every program asks its index',
  'A block of elements per program',
  'The mask for the ragged edge',
  'Too small a block starves the GPU',
  'One launch, one output',
];
const CAPTIONS = [
  'A CUDA kernel asks you to write a thread and decide its own index. Triton asks you to write one program and let it be handed its index.',
  'Adding two vectors of N elements launches a grid: many copies of your function, run at the same time across the SMs.',
  'Inside, program_id(0) says "I am program number p of the grid". That single number decides which slice this program owns.',
  'Each program handles BLOCK_SIZE elements as a vector built by tl.arange, and computes offsets = p × BLOCK_SIZE + arange(BLOCK_SIZE).',
  'A grid rarely divides the length evenly. A mask loads the in-range elements and drops the rest, so one kernel serves any N.',
  'N = 128 with BLOCK_SIZE = 1 launches 128 tiny programs and spends all its time on launch overhead. With BLOCK_SIZE = 128 one program does it.',
  'Load, compute, store — once. There is no loop and no thread index arithmetic, because the grid already covered the work.',
];

const ROWS = 4;
const COLS = 8;
const CELL = 34;
const GAP = 4;
const GX = 236;
const GY = 74;
const PW = COLS * (CELL + GAP) - GAP;

// one "program" tile in the grid diagram
const prog = (i: number) => ({
  c: i % COLS,
  r: Math.floor(i / COLS),
});

const BLOCK = 32; // elements per program at the "good" setting
const GOOD = N / BLOCK; // 4 programs
const BAD = 128; // programs when BLOCK_SIZE = 1

export default function LaunchGrid() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3600 });
  const st = s.step;

  // how many programs are drawn, and which one is highlighted
  const shown = st === 5 ? BAD : st >= 6 ? GOOD : st >= 3 ? GOOD : 4;
  const shownProgs = Math.min(shown, ROWS * COLS);
  const active = st >= 2 ? 1 : -1;
  const small = st === 5;

  const nPerProg = small ? 1 : BLOCK;
  const nProgs = small ? BAD : GOOD;
  const waves = nProgs / 40; // T4 has 40 SMs

  return (
    <Stage
      rootRef={ref}
      title="A Triton kernel is a grid of programs, not a pile of threads"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'idle', 'error', 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Elements', value: fmt.int(N), kind: 'memory' },
          { label: 'BLOCK_SIZE', value: fmt.int(nPerProg), kind: 'compute', note: small ? 'starves the GPU' : 'one program, one slice' },
          { label: 'Programs', value: fmt.int(nProgs), kind: st === 5 ? 'error' : 'compute', note: 'the grid size' },
          { label: 'Waves on a T4', value: waves.toFixed(2), kind: st === 5 ? 'error' : 'token', note: '40 SMs' },
        ]} />
      }
    >
      <svg viewBox="0 0 640 360" role="img" aria-label="A vector of 128 elements feeding a grid of Triton programs. Each program takes a contiguous slice using its program index, masked at the ragged edge, and writes one slice of the output.">
        {/* ── input vector ── */}
        <Label x={GX / 2 + 20} y={34} anchor="middle" tone={2} size={10.5}>x  [128 elements]</Label>
        <Matrix
          x={40}
          y={44}
          rows={2}
          cols={16}
          cell={11}
          gap={2}
          fill={(r, c) => (st >= 4 && c >= (r === 0 ? 16 - (BLOCK % 16) : 0) ? 'dim:text' : 'dim:memory')}
        />
        <Label x={40 + (16 * 13 - 2) / 2} y={86} anchor="middle" tone={3} size={9.5}>128 numbers, 256 bytes</Label>

        {/* ── the grid ── */}
        <Label x={GX + PW / 2} y={GY - 12} anchor="middle" tone={2} size={10.5}>
          {st === 0 ? 'your @triton.jit function, launched N times' : `grid = ${nProgs} programs`}
        </Label>
        {Array.from({ length: shownProgs }, (_, i) => {
          const { c, r } = prog(i);
          const isActive = i === active;
          const on = st >= 1;
          return (
            <g key={i}>
              <rect
                x={GX + c * (CELL + GAP)}
                y={GY + r * (CELL + GAP)}
                width={CELL}
                height={CELL}
                rx={3}
                fill={isActive ? 'var(--compute)' : on ? 'var(--memory-dim)' : 'transparent'}
                stroke={isActive ? 'var(--compute)' : on ? 'var(--memory)' : 'var(--rule-2)'}
                strokeWidth={isActive ? 1.8 : 1}
                strokeDasharray={on ? undefined : '3 3'}
                className="v-cell"
              />
              <text
                x={GX + c * (CELL + GAP) + CELL / 2}
                y={GY + r * (CELL + GAP) + CELL / 2 + 3.4}
                textAnchor="middle"
                className="v-label"
                style={{ fontSize: 9, fill: isActive ? '#0e1116' : 'var(--text-3)', fontWeight: 700 }}
              >
                {on ? (small ? '1' : `p${i}`) : ''}
              </text>
            </g>
          );
        })}
        {st === 5 && (
          <text x={GX + PW / 2} y={GY + ROWS * (CELL + GAP) + 16} anchor="middle" className="v-label" style={{ fill: 'var(--error)', fontSize: 10, fontWeight: 600 }}>
            128 programs for 128 elements — all launch overhead, no work
          </text>
        )}

        {/* ── program body ── */}
        <Fade show={st >= 2}>
          <g>
            <rect x={366} y={GY + 26} width={238} height={126} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x={378} y={GY + 44} tone={1} size={10.5} bold>program p = program_id(0)</Label>
            <text x={378} y={GY + 64} className="v-label" style={{ fontSize: 10, fill: 'var(--memory)' }}>
              offs = p × {BLOCK} + tl.arange(0, {BLOCK})
            </text>
            <text x={378} y={GY + 82} className="v-label" style={{ fontSize: 10, fill: 'var(--memory)' }}>
              mask = offs &lt; {N}
            </text>
            <text x={378} y={GY + 100} className="v-label" style={{ fontSize: 10, fill: 'var(--compute)' }}>
              a = tl.load(x + offs, mask)
            </text>
            <text x={378} y={GY + 116} className="v-label" style={{ fontSize: 10, fill: 'var(--compute)' }}>
              tl.store(c + offs, a + b, mask)
            </text>
            <Label x={378} y={GY + 140} tone={3} size={9.5}>No loop. No thread id. The grid did it.</Label>
          </g>
        </Fade>

        {/* ── arrows ── */}
        <Fade show={st >= 1}>
          <Arrow x1={216} y1={70} x2={GX - 8} y2={GY + 30} kind="memory" />
          <Arrow x1={GX + PW + 6} y2={GY + 30} x2={358} y2={GY + 60} kind="compute" curve={-18} />
        </Fade>

        {/* ── output vector ── */}
        <Label x={GX / 2 + 20} y={GY + ROWS * (CELL + GAP) + 40} anchor="middle" tone={2} size={10.5}>c = x + y</Label>
        <Matrix x={40} y={GY + ROWS * (CELL + GAP) + 50} rows={2} cols={16} cell={11} gap={2} fill={() => 'dim:compute'} />

        {/* ── offset arithmetic, step 3 ── */}
        <Fade show={st === 3 || st === 4}>
          <g>
            <Dim x1={GX + CELL / 2} x2={GX + PW + 4} y={GY - 22} label={`block ${BLOCK} elements`} kind="memory" />
            <Label x={GX} y={GY - 40} tone={1} size={10.5}>{'offs = p × 32 + [0, 1, 2, … 31]'}</Label>
            <Label x={GX} y={GY + ROWS * (CELL + GAP) + 16} tone={2} size={10}>{`p = 0 → offs 0..31   p = 1 → offs 32..63`}</Label>
            <Label x={GX} y={GY + ROWS * (CELL + GAP) + 30} tone={2} size={10}>{'p = 2 → offs 64..95   p = 3 → offs 96..127'}</Label>
          </g>
        </Fade>

        {/* ── masking, step 4 ── */}
        <Fade show={st === 4}>
          <g>
            <Label x={366} y={GY + 174} tone={2} size={10}>{'mask = offs < 128'}</Label>
            <text x={366} y={GY + 192} className="v-label" style={{ fontSize: 10, fill: 'var(--error)' }}>
              offs = 128…159 → dropped, never loaded
            </text>
            <Label x={366} y={GY + 208} tone={3} size={9.5}>128 is not a multiple of 32. The mask absorbs that.</Label>
          </g>
        </Fade>

        {/* ── block-size trade-off, step 5 ── */}
        <Fade show={st === 5}>
          <g>
            <Label x={366} y={GY + 176} tone={2} size={10.5}>{'BLOCK_SIZE = 1'}</Label>
            <text x={366} y={GY + 192} className="v-label" style={{ fontSize: 10, fill: 'var(--error)' }}>
              128 launches × ~5 µs ≈ 640 µs of pure overhead
            </text>
            <Label x={366} y={GY + 210} tone={2} size={10.5}>{'The 128 bytes take ~0.4 µs to move.'}</Label>
          </g>
        </Fade>

        {/* ── the good configuration, step 6 ── */}
        <Fade show={st === 6}>
          <g>
            <Label x={40} y={GY + ROWS * (CELL + GAP) + 80} tone={1} size={11} kind="token" bold>
              {'4 programs, one launch, ~4 µs total'}
            </Label>
            <Label x={40} y={GY + ROWS * (CELL + GAP) + 98} tone={3} size={9.5}>
              Same arithmetic, same bytes, 160× less time. The win was in the shape of the work, not the math.
            </Label>
          </g>
        </Fade>

        {/* active-program marker */}
        {st >= 2 && st <= 4 && (
          <Move x={0} y={-22}>
            <Label x={GX + CELL / 2} y={GY + ROWS * (CELL + GAP) + 76} anchor="middle" tone={3} size={9.5}>
              {'← this program owns these 32 elements'}
            </Label>
          </Move>
        )}
      </svg>
    </Stage>
  );
}