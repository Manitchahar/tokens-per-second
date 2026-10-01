import { useEffect, useRef, useState } from 'react';
import { Stage } from '../lib/Stage';
import { useStep, prefersReducedMotion } from '../lib/useStep';
import { Label, Fade, Matrix, Span } from '../lib/svg';

const A = [[1, 2, 0, 1], [3, 1, 2, 0], [0, 1, 1, 2]];
const Bm = [[2, 1, 0], [1, 0, 3], [0, 2, 1], [1, 1, 2]];
const Cm = A.map((r) => Bm[0].map((_, j) => r.reduce((s, a, k) => s + a * Bm[k][j], 0)));
const m = 3, k = 4, n = 3;

const STEPS = ['The setup', 'One output cell', 'Count the work', 'Every cell', 'The FLOP formula', 'Decode: matrix × vector', 'Prefill: matrix × matrix'];
const CAPTIONS = [
  'A is 3 tokens × 4 features (activations). B is a 4 × 3 weight matrix. Their product C is 3 × 3.',
  'Each output cell is a dot product: walk along a row of A and down a column of B, multiply pairs, add them up.',
  'One cell took k = 4 multiplies and 4 adds: 2k floating-point operations (FLOPs).',
  'Repeat for all m × n cells. Every row of A meets every column of B.',
  'Total work = m × n cells × 2k FLOPs = 2mkn. For a real layer this is billions.',
  'Decode feeds one token (m = 1). Each weight is loaded from memory and used once: about 1 FLOP per byte.',
  'Prefill feeds hundreds of tokens at once. Each weight loaded is reused for every token: hundreds of FLOPs per byte.',
];

const CS = 26;
const G = 4;
export default function MatmulViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3400 });
  const st = s.step;
  const [filled, setFilled] = useState(0);

  useEffect(() => {
    if (st < 3) { setFilled(st >= 1 ? 1 : 0); return; }
    if (st > 3 || prefersReducedMotion()) { setFilled(m * n); return; }
    setFilled(1);
    let i = 1;
    const t = window.setInterval(() => { i++; setFilled(i); if (i >= m * n) window.clearInterval(t); }, 380);
    return () => window.clearInterval(t);
  }, [st]);

  const cur = Math.max(0, Math.min(m * n - 1, filled - 1));
  const ci = Math.floor(cur / n);
  const cj = cur % n;
  const showCells = st <= 4;
  const hlRow = st >= 1 && st <= 3 ? ci : -1;
  const hlCol = st >= 1 && st <= 3 ? cj : -1;

  const cx = 250, cy = 150;
  const ax = 40, ay = cy + 2;

  const flops = filled * 2 * k;

  return (
    <Stage rootRef={ref} title="Matrix multiply, one cell at a time" steps={STEPS} captions={CAPTIONS} legend={['compute', 'memory']} {...s}>
      <svg viewBox="0 0 640 320" role="img" aria-label="Matrix multiplication C = A times B shown cell by cell, with a running count of floating-point operations.">
        <Fade show={showCells}>
          <g>
            <Label x={ax + (k * (CS + G)) / 2} y={ay - 10} anchor="middle" tone={2} size={10.5}>A [3 × 4]</Label>
            <Matrix x={ax} y={ay} rows={m} cols={k} cell={CS} gap={G}
              fill={(r) => (r === hlRow ? 'dim:compute' : 'dim:text')}
              text={(r, c) => String(A[r][c])} />
            <Label x={cx + (n * (CS + G)) / 2} y={cy - k * (CS + G) - 20} anchor="middle" tone={2} size={10.5}>B [4 × 3]</Label>
            {/* B sits above C so columns line up */}
            <Matrix x={cx} y={cy - k * (CS + G) - 12} rows={k} cols={n} cell={CS} gap={G}
              fill={(_, c) => (c === hlCol ? 'dim:memory' : 'dim:text')}
              text={(r, c) => String(Bm[r][c])} />
            <Label x={cx + (n * (CS + G)) / 2} y={cy + m * (CS + G) + 18} anchor="middle" tone={2} size={10.5}>C = A × B [3 × 3]</Label>
            <Matrix x={cx} y={cy + 2} rows={m} cols={n} cell={CS} gap={G}
              fill={(r, c) => (r * n + c < filled ? (r * n + c === cur && st <= 3 && st >= 1 ? 'token' : 'dim:token') : null)}
              text={(r, c) => (r * n + c < filled ? String(Cm[r][c]) : null)} />
          </g>
        </Fade>

        {/* dot product expression */}
        <Fade show={st >= 1 && st <= 3}>
          <g>
            <Label x={400} y={150} tone={3} size={10}>{`C[${ci}, ${cj}] = row ${ci} of A · column ${cj} of B`}</Label>
            <text x={400} y={174} className="v-label t1" style={{ fontSize: 12.5 }}>
              {A[ci].map((a, kk) => `${a}×${Bm[kk][cj]}`).join(' + ')} = <tspan style={{ fill: 'var(--token)', fontWeight: 700 }}>{Cm[ci][cj]}</tspan>
            </text>
          </g>
        </Fade>
        <Fade show={st >= 2 && st <= 4}>
          <g>
            <Label x={400} y={214} tone={2} size={10.5}>{`per cell: ${k} multiplies + ${k} adds = 2k = ${2 * k} FLOPs`}</Label>
            <Label x={400} y={240} tone={1} size={12} kind="compute">{`FLOPs so far: ${flops} of ${2 * m * n * k}`}</Label>
          </g>
        </Fade>
        <Fade show={st === 4}>
          <g>
            <Label x={400} y={100} tone={1} math size={20}>{'FLOPs = 2 · m · k · n'}</Label>
            <Label x={400} y={124} tone={3} size={10}>{`here: 2 · 3 · 4 · 3 = ${2 * m * k * n}`}</Label>
            <Label x={400} y={280} tone={2} size={10.5}>{'real layer: 2 · 1000 · 4096 · 4096'}</Label>
            <Label x={400} y={298} tone={1} size={11} kind="compute">{'= 33.6 billion FLOPs'}</Label>
          </g>
        </Fade>

        {/* decode vs prefill */}
        <Fade show={st >= 5}>
          <g>
            <Label x={40} y={34} tone={2} size={10.5}>{'weight W [4096 × 4096]: 33.5 MB to load from memory'}</Label>
            <Matrix x={300} y={52} rows={10} cols={10} cell={17} gap={2} fill={() => 'dim:memory'} />
            <Label x={300 + 95} y={262} anchor="middle" tone={3} size={10}>loaded once per step</Label>
            {/* input rows */}
            {Array.from({ length: st === 5 ? 1 : 10 }, (_, r) => (
              <Span key={r} x={60} y={52 + r * 19} w={190} h={15} kind="compute" dim={r > 0} />
            ))}
            <Label x={155} y={st === 5 ? 90 : 262} anchor="middle" tone={3} size={10}>{st === 5 ? 'x: 1 token [1 × 4096]' : 'x: 512 tokens [512 × 4096] (10 drawn)'}</Label>
            <Label x={270} y={112} anchor="middle" tone={1} math size={20}>{'×'}</Label>
            <g>
              <Label x={40} y={296} tone={2} size={11}>{st === 5 ? 'work: 2 × 1 × 4096 × 4096 = 33.5 M FLOPs' : 'work: 2 × 512 × 4096 × 4096 = 17.2 G FLOPs'}</Label>
              <Label x={626} y={296} anchor="end" tone={1} size={12} kind={st === 5 ? 'memory' : 'compute'}>{st === 5 ? '≈ 1 FLOP per byte loaded' : '≈ 512 FLOPs per byte loaded'}</Label>
            </g>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
