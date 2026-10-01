import { useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, Matrix, Move, Arrow, fmt } from '../lib/svg';

// The point: a reduction is not a loop over elements — it is one block-wide
// collapse. And softmax can be done in a single pass if you carry a running max
// and a running sum that you rescale as you go. That single pass is the fusion.

const STEPS = [
  'A row of scores',
  'Softmax, as you know it',
  'The max has to come first',
  'The cost of three passes',
  'A reduction in one step',
  'Carry the running max',
  'One pass, one read',
  'The fused kernel',
];
const CAPTIONS = [
  'A row of attention scores — the input to softmax. In attention this is [1 × context_length]; the next token is chosen from these numbers.',
  'Softmax is: subtract the max, exponentiate, divide by the sum. It turns arbitrary scores into probabilities that sum to 1.',
  'You must know the row maximum before you can exponentiate safely, otherwise exp() of a large score overflows to infinity.',
  'Naively that is three passes over memory: one for the max, one to exponentiate and sum, one to divide. Three reads and one write.',
  'A reduction collapses a whole block into one value with a tree of pairwise operations — log2(n) steps instead of n.',
  'The trick: carry a running max m and a running sum l as you stream blocks. When a new block raises the max, rescale the old sum — and you never re-read anything.',
  'One pass. One read of the row, one write of the result. The row never leaves on-chip memory.',
  'That is the fused kernel: read once, compute everything, write once. Traffic drops by ~3× and the kernel becomes purely bandwidth-bound.',
];

// a toy row of 8 scores
const SCORES = [1.2, 3.4, 0.5, 2.8, 4.1, 1.9, 3.3, 0.8];
const NCOLS = SCORES.length;
const CELL = 40;
const GAP = 6;
const RW = NCOLS * (CELL + GAP) - GAP;

type Pass = 'naive' | 'online';
const PRESET: Record<number, Pass> = { 0: 'naive', 1: 'naive', 2: 'naive', 3: 'naive', 4: 'naive', 5: 'online', 6: 'online', 7: 'online' };

export default function SoftmaxBlocks() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3600 });
  const st = s.step;
  const [pass, setPass] = useState<Pass>(PRESET[0]);

  // compute the maths for the toy row
  const m = Math.max(...SCORES);
  const exps = SCORES.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sum);

  const x0 = 90;
  const y0 = 58;
  const rw = useTween(1, 600); // reveal of the row

  const showNaive = pass === 'naive';
  const onLine = st >= 5;
  const isFused = st >= 7;

  return (
    <Stage
      rootRef={ref}
      title="Softmax: three passes over memory, or one"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', 'token', 'error', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Row max', value: m.toFixed(2), kind: 'compute' },
          { label: 'Sum of exps', value: sum.toFixed(3), kind: 'compute' },
          { label: 'Passes over memory', value: isFused ? '1' : '3', kind: isFused ? 'token' : 'error' },
          { label: 'Reads of the row', value: isFused ? '1' : '3', kind: isFused ? 'token' : 'error' },
        ]} />
      }
      controls={
        <Segmented
          label="Strategy"
          value={pass}
          onChange={setPass}
          options={[
            { value: 'naive' as Pass, label: 'Three-pass' },
            { value: 'online' as Pass, label: 'Online (one-pass)' },
          ]}
        />
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A row of attention scores going through softmax. The three-pass version reads the row three times: once for the max, once to exponentiate and sum, once to divide. The online version carries a running max and sum so it reads the row once.">
        {/* the row of scores */}
        <Label x={x0 + RW / 2} y={y0 - 12} anchor="middle" tone={2} size={10.5}>scores [1 × {NCOLS}]</Label>
        <g className="v-move" style={{ transform: `scale(${rw}, 1)`, transformOrigin: `${x0}px ${y0}px` }}>
          <Matrix
            x={x0}
            y={y0}
            rows={1}
            cols={NCOLS}
            cell={CELL}
            gap={GAP}
            fill={() => 'dim:memory'}
            text={(r, c) => SCORES[c].toFixed(1)}
          />
        </g>

        {/* pass 1: max */}
        <Fade show={showNaive && st >= 2}>
          <g>
            <Label x={x0} y={y0 + CELL + 30} tone={1} size={11} kind="memory">{'Pass 1 — read the row for the max:  max = 4.1'}</Label>
            <Dim2 x1={x0} x2={x0 + RW} y={y0 + CELL + 14} />
            <Label x={x0 + RW + 10} y={y0 + CELL + 18} tone={2} size={10}>max = 4.1</Label>
          </g>
        </Fade>

        {/* pass 2: exp + sum */}
        <Fade show={showNaive && st >= 3}>
          <g>
            <Label x={x0} y={y0 + CELL + 56} tone={1} size={11} kind="memory">{'Pass 2 — exponentiate and sum:  sum = 12.87'}</Label>
            <Matrix x={x0} y={y0 + CELL + 70} rows={1} cols={NCOLS} cell={CELL} gap={GAP}
              fill={() => 'dim:compute'} text={(r, c) => exps[c].toFixed(2)} />
            <Label x={x0 + RW + 10} y={y0 + CELL + 96} tone={2} size={10}>sum = 12.87</Label>
          </g>
        </Fade>

        {/* pass 3: divide */}
        <Fade show={showNaive && st >= 4}>
          <g>
            <Label x={x0} y={y0 + CELL + 128} tone={1} size={11} kind="memory">{'Pass 3 — divide by the sum to get probabilities'}</Label>
            <Matrix x={x0} y={y0 + CELL + 142} rows={1} cols={NCOLS} cell={CELL} gap={GAP}
              fill={() => 'token'} text={(r, c) => probs[c].toFixed(2)} />
            <Label x={x0 + RW + 10} y={y0 + CELL + 168} tone={2} size={10}>sums to 1.00</Label>
          </g>
        </Fade>

        {/* online version */}
        <Fade show={!showNaive}>
          <g>
            <Label x={x0} y={y0 + CELL + 34} tone={1} size={11} kind="token">{'One pass — read each element exactly once'}</Label>
            {/* running state */}
            <rect x={x0} y={y0 + CELL + 48} width={RW} height={40} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x={x0 + 10} y={y0 + CELL + 66} tone={3} size={10}>running state, carried in registers:</Label>
            <text x={x0 + 10} y={y0 + CELL + 82} className="v-label" style={{ fontSize: 10.5, fill: 'var(--compute)' }}>
              m (running max) &nbsp; l (running sum, rescaled)
            </text>
            {/* blocks streaming through */}
            {[0, 1, 2].map((b) => {
              const bx = x0 + b * (RW / 3);
              return (
                <Fade key={b} show={onLine && st >= 5 + (b === 0 ? 0 : 0)}>
                  <g>
                    <rect x={bx + 4} y={y0 + CELL + 100} width={RW / 3 - 8} height={22} rx={3}
                      fill="var(--memory-dim)" stroke="var(--memory)" />
                    <text x={bx + RW / 6} y={y0 + CELL + 115} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>
                      {`block ${b}`}
                    </text>
                    <Arrow x1={bx + RW / 6} y1={y0 + CELL + 96} x2={bx + RW / 6} y2={y0 + CELL + 100} kind="memory" />
                  </g>
                </Fade>
              );
            })}
            <Label x={x0} y={y0 + CELL + 142} tone={2} size={10.5}>{'When a block raises the max, the old sum is rescaled:'}</Label>
            <text x={x0} y={y0 + CELL + 160} className="v-label" style={{ fontSize: 10.5, fill: 'var(--compute)' }}>
              l = l · exp(m_old − m_new),&nbsp; m = max(m_old, block_max)
            </text>
            {/* result */}
            <Matrix x={x0} y={y0 + CELL + 176} rows={1} cols={NCOLS} cell={CELL} gap={GAP}
              fill={() => 'token'} text={(r, c) => probs[c].toFixed(2)} label="same probabilities, one read" />
          </g>
        </Fade>

        {/* traffic summary */}
        <Fade show={st >= 3}>
          <g>
            <rect x={x0} y={y0 + CELL + 200} width={RW} height={54} rx={4}
              fill="var(--ground-2)" stroke={isFused ? 'var(--token)' : 'var(--error)'} strokeDasharray={isFused ? undefined : '4 3'} />
            <Label x={x0 + 10} y={y0 + CELL + 220} tone={2} size={10.5}>memory traffic for one row of 8 (FP16):</Label>
            <Label x={x0 + 10} y={y0 + CELL + 240} tone={1} size={11} kind={isFused ? 'token' : 'error'}>
              {isFused ? '1 read + 1 write = 32 bytes' : '3 reads + 1 write = 64 bytes — 2× the traffic'}
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}

function Dim2({ x1, x2, y }: { x1: number; x2: number; y: number }) {
  return <line x1={x1} x2={x2} y1={y} y2={y} stroke="var(--memory)" strokeWidth={1} strokeDasharray="3 3" />;
}