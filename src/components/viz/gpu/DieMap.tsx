import { useRef } from 'react';
import { Stage, Readout } from '../lib/Stage';
import { useStep, prefersReducedMotion } from '../lib/useStep';
import { useClock } from '../lib/tween';
import { Label, Arrow, Fade, fmt } from '../lib/svg';

const STEPS = [
  'The package',
  'HBM: the memory',
  'L2: the shared cache',
  'The multiprocessors',
  'Inside one SM',
  'The latency ladder',
  'So reuse is everything',
];
const CAPTIONS = [
  'An H100 is a single silicon die packed with 132 multiprocessors, large caches, and five stacks of HBM memory on the same package.',
  'HBM is the model’s home: 80 GB of it, read at 3.35 TB/s. It is off-die, so every access crosses a wide memory bus, and it is slow to answer.',
  'The L2 cache sits on the die and is shared by everything on it. A 50 MB slice catches most of the traffic the multiprocessors generate, so most requests never reach HBM.',
  'The 132 multiprocessors are the workers. Each has its own small caches, its own registers, and its own tensor cores for matrix multiply.',
  'One multiprocessor: four warp schedulers pick instructions, a 256 KB register file holds the numbers being worked on, four tensor cores do the arithmetic, and 256 KB of L1 and shared memory is its own fast scratchpad.',
  'The closer memory is to the math, the faster it answers: about 700 ns for HBM, 200 ns for L2, 30 ns for L1, and effectively instant for a register.',
  'A weight read once from HBM is expensive. The same weight read a hundred times out of L1 is nearly free. Every optimization in this course is a way of creating reuse.',
];

/** HBM → L2 → L1 → register. Rough public figures for an H100, for shape, not simulation. */
const LADDER = [
  { name: 'HBM', ns: 700, kind: 'memory' as const, size: '80 GB, off the die' },
  { name: 'L2 cache', ns: 200, kind: 'memory' as const, size: '50 MB, shared' },
  { name: 'L1 / shared', ns: 30, kind: 'compute' as const, size: '256 KB per SM' },
  { name: 'Register', ns: 1, kind: 'compute' as const, size: '256 KB per SM' },
];

const DIE = { x: 168, y: 36, w: 304, h: 192 };
const COLS = 11;
const SM_W = 23.8;
const SM_H = 7;
const smX = (c: number) => DIE.x + 6 + c * (SM_W + 3);
const smRowY = (block: 0 | 1, r: number) => (block === 0 ? 50 : 146) + r * 10;

const ZOOM = { x: 56, y: 258, w: 528, h: 104 };
const COL_X = [68, 198, 328, 458];
const COL_W = 118;

const BAR_X = 132;
const BAR_MAX = 236;

export default function DieMap() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const clock = useClock(s.playing, 0.45);
  const p = st === 5 && s.playing && !prefersReducedMotion() ? clock % 1 : st === 5 ? 0.35 : 0;

  const hbm = (x: number, y: number, w: number, h: number) => (
    <g className="v-fade" style={{ opacity: st >= 1 ? 1 : 0.28 }}>
      <rect x={x} y={y} width={w} height={h} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1.2} />
      {Array.from({ length: Math.max(2, Math.round(h / 16)) }, (_, i) => (
        <line key={i} x1={x + 6} x2={x + w - 6} y1={y + 11 + i * 16} y2={y + 11 + i * 16} stroke="var(--memory)" strokeWidth={0.7} opacity={0.5} />
      ))}
    </g>
  );

  const l2On = st >= 2;
  const zoomOn = st >= 4;
  const ladderOn = st >= 5;
  const activeLadder = ladderOn ? Math.min(3, Math.floor(p * 4)) : -1;
  const l2Y = 108;

  return (
    <Stage
      rootRef={ref}
      title="Inside an H100, and how far a number has to travel"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['comm', 'fast interconnect']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Memory on the package', value: '80 GB HBM3e', kind: 'memory', note: '3.35 TB/s over 5,120 bits' },
          { label: 'Multiprocessors', value: '132', kind: 'compute', note: '4 tensor cores each' },
          { label: 'L2 cache', value: '50 MB', kind: 'memory', note: 'shared, on the die' },
          { label: 'Tensor cores, total', value: '528', kind: 'compute', note: 'the matmul specialists' },
        ]} />
      }
    >
      <svg viewBox="0 0 640 468" role="img" aria-label="A schematic GPU package: HBM stacks around a die containing 132 multiprocessors and a shared L2 cache, with a zoom of one multiprocessor and a chart of memory latency by level.">
        {/* NVLink fingers */}
        <g className="v-fade" style={{ opacity: 0.7 }}>
          {Array.from({ length: 6 }, (_, i) => (
            <rect key={i} x={16} y={42 + i * 30} width={22} height={19} rx={1.5} fill="var(--comm-dim)" stroke="var(--comm)" strokeWidth={0.9} />
          ))}
          <Label x={27} y={224} anchor="middle" tone={3} size={9}>NVLink</Label>
        </g>

        {/* HBM stacks */}
        {hbm(48, 40, 104, 78)}
        {hbm(48, 130, 104, 78)}
        {hbm(488, 40, 104, 54)}
        {hbm(488, 100, 104, 54)}
        {hbm(488, 160, 104, 54)}

        {/* memory bus */}
        <g className="v-fade" style={{ opacity: st >= 1 ? 1 : 0.15 }}>
          {[79, 169, 67, 127, 187].map((y, i) => (
            <line key={i} x1={i < 2 ? 152 : 488} x2={i < 2 ? 168 : 472} y1={y} y2={y} stroke="var(--memory)" strokeWidth={1.2} />
          ))}
        </g>
        <Label x={320} y={240} anchor="middle" tone={2} size={9.5} kind="memory">
          HBM · 80 GB · 3.35 TB/s, over a 5,120-bit bus
        </Label>

        {/* the die */}
        <rect x={DIE.x} y={DIE.y} width={DIE.w} height={DIE.h} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1.2} />

        {/* 132 SMs: 11 x 6 above the L2, 11 x 6 below */}
        <g className="v-fade" style={{ opacity: st >= 3 ? 1 : 0.18 }}>
          {Array.from({ length: 132 }, (_, i) => {
            const top = i < 66;
            const idx = top ? i : i - 66;
            return (
              <rect key={i} x={smX(idx % COLS)} y={smRowY(top ? 0 : 1, Math.floor(idx / COLS))} width={SM_W} height={SM_H} rx={1.5}
                fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={0.7} className="v-cell" />
            );
          })}
          <Label x={DIE.x + 6} y={46} tone={2} size={9} kind="compute">66 SMs</Label>
          <Label x={DIE.x + 6} y={141} tone={2} size={9} kind="compute">66 more SMs</Label>
        </g>

        {/* L2 */}
        <g className="v-fade" style={{ opacity: l2On ? 1 : 0.2 }}>
          <rect x={DIE.x + 4} y={l2Y} width={DIE.w - 8} height={17} rx={2}
            fill={l2On ? 'var(--memory)' : 'var(--memory-dim)'} stroke="var(--memory)" strokeWidth={1} className="v-cell" />
          <text x={DIE.x + DIE.w / 2} y={l2Y + 12} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: l2On ? '#0e1116' : 'var(--text)' }}>
            L2 cache · 50 MB, shared by all 132 SMs
          </text>
        </g>

        {/* ── zoom into one SM */}
        <Fade show={zoomOn}>
          <line x1={56} y1={250} x2={584} y2={250} className="v-rule faint" />
          <rect x={ZOOM.x} y={ZOOM.y} width={ZOOM.w} height={ZOOM.h} rx={4} fill="var(--ground-3)" stroke="var(--compute)" strokeWidth={1.2} />
          <Label x={ZOOM.x + 12} y={ZOOM.y + 16} tone={1} size={10.5} kind="compute" bold>one streaming multiprocessor</Label>
          {COL_X.map((x, i) => (
            <g key={`b${i}`}>
              <rect x={x} y={278} width={COL_W} height={12} rx={2} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={0.9} />
              <text x={x + COL_W / 2} y={287} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>warp scheduler</text>
              <Arrow x1={x + COL_W / 2} y1={290} x2={x + COL_W / 2} y2={295} kind="compute" width={1} />
              <rect x={x} y={296} width={COL_W} height={30} rx={2} fill="var(--compute)" stroke="var(--compute)" strokeWidth={1} className="v-cell" />
              <text x={x + COL_W / 2} y={315} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: '#0e1116' }}>tensor core</text>
              <Arrow x1={x + COL_W / 2} y1={326} x2={x + COL_W / 2} y2={331} kind="compute" width={1} />
            </g>
          ))}
          <rect x={68} y={332} width={248} height={18} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.9} className="v-cell" />
          <text x={192} y={344} textAnchor="middle" className="v-label" style={{ fontSize: 9 }}>register file · 256 KB</text>
          <rect x={324} y={332} width={248} height={18} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.9} className="v-cell" />
          <text x={448} y={344} textAnchor="middle" className="v-label" style={{ fontSize: 9 }}>L1 / shared memory · 256 KB</text>
          <Label x={320} y={378} anchor="middle" tone={2} size={9.5} mono={false}>
            Every number a kernel uses is fetched into a register first, and a register is the only place a tensor core reads from.
          </Label>
        </Fade>

        {/* ── the latency ladder */}
        <Fade show={ladderOn}>
          <line x1={56} y1={390} x2={584} y2={390} className="v-rule faint" />
          <Label x={56} y={408} tone={1} size={10.5} bold mono={false}>How long a number takes to arrive, by where it lives</Label>
          {LADDER.map((r, i) => {
            const y = 420 + i * 12;
            const w = ((Math.log10(r.ns) + 0.5) / 3.5) * BAR_MAX;
            const on = activeLadder === i;
            return (
              <g key={r.name} className="v-fade" style={{ opacity: activeLadder > i ? 0.35 : 1 }}>
                <text x={122} y={y + 8} textAnchor="end" className="v-label" style={{ fontSize: 9.5, fill: on ? 'var(--text)' : 'var(--text-2)' }}>{r.name}</text>
                <rect x={BAR_X} y={y} width={w} height={9} rx={2} fill={r.kind === 'memory' ? 'var(--memory)' : 'var(--compute)'} className="v-cell" />
                <text x={BAR_X + BAR_MAX + 10} y={y + 8} className="v-label t2" style={{ fontSize: 9.5 }}>
                  {`~${r.ns} ns · ${r.size}`}
                </text>
                <text x={584} y={y + 8} textAnchor="end" className="v-label t3" style={{ fontSize: 9 }}>
                  {i === 0 ? '1×' : `${fmt.int(700 / r.ns)}× faster`}
                </text>
              </g>
            );
          })}
        </Fade>
      </svg>
    </Stage>
  );
}
