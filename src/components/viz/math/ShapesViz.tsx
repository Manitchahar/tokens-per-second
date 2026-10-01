import { useRef, type ReactNode } from 'react';
import { Stage } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Matrix } from '../lib/svg';

const STEPS = ['A scalar', 'A vector', 'A matrix', 'A batch: 3-D tensor', 'Weights are matrices', 'Indexing', 'Reshape into heads'];
const CAPTIONS = [
  'A single number is a scalar. Shape: [] (no dimensions).',
  'A list of numbers is a vector. Each token inside an LLM is a vector of 4,096 numbers. Shape: [4096].',
  'Stack the vectors of a 5-token sentence and you get a matrix. Shape: [5, 4096], read as "5 rows of 4,096".',
  'Stack 3 sentences (a batch) and you get a 3-D tensor. Shape: [3, 5, 4096] = batch × tokens × hidden size.',
  'The model’s weights are matrices too: a [4096, 4096] projection holds 16.8 million numbers (33.5 MB in BF16).',
  'Indexing picks out pieces. x[0, 2] is the vector of token 2 in sentence 0, shape [4096].',
  'Reshape reinterprets the same numbers. [5, 4096] → [5, 32, 128]: each token’s vector split into 32 attention heads of 128.',
];

const D = 12; // cells drawn per vector (stands for 4096)
const T = 5;
const B = 3;
const C = 20; // cell pitch
const CELL = 17;
const X0 = 170;
const Y0 = 110;

export default function ShapesViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const pos = (b: number, t: number, d: number): { x: number; y: number; show: boolean; o: number } => {
    if (st === 0) return { x: 300, y: 150, show: b === 0 && t === 0 && d === 0, o: 1 };
    if (st === 1) return { x: X0 + d * C, y: 150, show: b === 0 && t === 0, o: 1 };
    let x = X0 + d * C;
    let y = Y0 + t * C;
    if (st === 3) { x += b * 18; y -= b * 18; return { x, y, show: true, o: b === 0 ? 1 : 0.55 - b * 0.12 }; }
    if (st === 4) { x = 40 + d * 15; y = Y0 + t * 15; return { x, y, show: b === 0, o: 0.5 }; }
    if (st === 6) { const h = Math.floor(d / 3); x = X0 - 20 + h * (3 * C + 16) + (d % 3) * C; }
    return { x, y, show: b === 0, o: st === 5 && t !== 2 ? 0.25 : 1 };
  };

  const cells: ReactNode[] = [];
  for (let b = B - 1; b >= 0; b--) {
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < D; d++) {
        const p = pos(b, t, d);
        const big = st === 0;
        const size = st === 4 ? 13 : big ? 34 : CELL;
        const hl = st === 5 && t === 2 && b === 0;
        const headTone = st === 6 ? Math.floor(d / 3) % 2 === 0 : false;
        cells.push(
          <g key={`${b}-${t}-${d}`} className="v-move" style={{ transform: `translate(${p.x}px, ${p.y}px)`, opacity: p.show ? p.o : 0 }}>
            <rect width={size} height={size} rx={1.5}
              fill={hl ? 'var(--text)' : b > 0 ? 'var(--ground-3)' : headTone ? 'rgb(232 230 225 / 0.28)' : 'rgb(232 230 225 / 0.14)'}
              stroke={hl ? 'var(--text)' : 'var(--rule-2)'} className="v-cell" />
            {big && <text x={17} y={22} textAnchor="middle" className="v-label t1" style={{ fontSize: 11 }}>3.14</text>}
          </g>,
        );
      }
    }
  }

  const shape = ['[ ]', '[4096]', '[5, 4096]', '[3, 5, 4096]', 'x: [5, 4096]    W: [4096, 4096]', 'x[0, 2] → [4096]', '[5, 4096] → [5, 32, 128]'][st];

  return (
    <Stage rootRef={ref} title="Tensors and their shapes" steps={STEPS} captions={CAPTIONS} {...s}>
      <svg viewBox="0 0 640 300" role="img" aria-label="Numbers arranged as a scalar, vector, matrix, 3-D tensor and weight matrix, with their shapes.">
        <text x={20} y={34} className="v-label t1" style={{ fontSize: 15 }}>shape {shape}</text>
        {cells}
        {/* dimension labels */}
        <Fade show={st >= 1 && st <= 3 || st === 5}>
          <g>
            <line x1={X0} x2={X0 + D * C - 3} y1={Y0 + T * C + 14 - (st === 1 ? -40 + T * C - C : 0)} y2={Y0 + T * C + 14 - (st === 1 ? -40 + T * C - C : 0)} stroke="var(--text-3)" />
            <Label x={X0 + (D * C) / 2} y={Y0 + T * C + 30 - (st === 1 ? -40 + T * C - C : 0)} anchor="middle" tone={2} size={10.5}>hidden size: 4,096 numbers (12 drawn)</Label>
          </g>
        </Fade>
        <Fade show={st >= 2 && st <= 3 || st === 5}>
          <g>
            <line x1={X0 - 12} x2={X0 - 12} y1={Y0} y2={Y0 + T * C - 3} stroke="var(--text-3)" />
            <Label x={X0 - 18} y={Y0 + (T * C) / 2 + 4} anchor="end" tone={2} size={10.5}>5 tokens</Label>
          </g>
        </Fade>
        <Fade show={st === 3}>
          <Label x={X0 + D * C + 50} y={Y0 - 30} tone={2} size={10.5}>3 sequences</Label>
        </Fade>
        <Fade show={st === 5}>
          <g>
            <Label x={X0 + D * C + 16} y={Y0 + 2 * C + 13} tone={1} size={11}>{'← x[0, 2]'}</Label>
            <Label x={X0 + D * C + 16} y={Y0 + 2 * C + 28} tone={3} size={9.5}>row 2 of sequence 0</Label>
          </g>
        </Fade>
        <Fade show={st === 6}>
          <g>
            {[0, 1, 2, 3].map((h) => (
              <Label key={h} x={X0 - 20 + h * (3 * C + 16) + 29} y={Y0 - 10} anchor="middle" tone={2} size={10}>{h === 3 ? 'head 31' : `head ${h}`}</Label>
            ))}
            <Label x={X0 - 20} y={Y0 + T * C + 26} tone={3} size={10}>each head gets 128 numbers of every token: same data, new shape</Label>
          </g>
        </Fade>

        {/* weight matrix */}
        <Fade show={st === 4}>
          <g>
            <Label x={40} y={Y0 - 14} tone={2} size={10.5}>activations x</Label>
            <Label x={40} y={Y0 + T * 15 + 16} tone={3} size={10}>[5, 4096]</Label>
            <Matrix x={330} y={70} rows={12} cols={12} cell={13} gap={2} fill={() => 'dim:text'} />
            <Label x={330 + 90} y={60} anchor="middle" tone={2} size={10.5}>weight matrix W</Label>
            <Label x={330 + 90} y={270} anchor="middle" tone={3} size={10}>{'[4096, 4096] = 16.8 M numbers = 33.5 MB in BF16'}</Label>
            <Label x={260} y={112} anchor="middle" tone={1} math size={22}>{'×'}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
