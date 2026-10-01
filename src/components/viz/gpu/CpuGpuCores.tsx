import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep, prefersReducedMotion } from '../lib/useStep';
import { useClock } from '../lib/tween';
import { Label, Fade, fmt } from '../lib/svg';

const STEPS = [
  'One instruction',
  'The problem: a stall',
  'Every core stalls together',
  'Small cores, many of them',
  'Warps hide the wait',
  'The cost of many warps',
  'What it means for inference',
];
const CAPTIONS = [
  'A core can run one stream of instructions at a time. Ask it for a number from memory and it cannot proceed until that number arrives.',
  'Almost every instruction in a neural network needs a value from memory. So the core spends most of its life waiting.',
  'All 8 cores were issued the same instruction at the same time, so they all wait at the same time. The machine sits idle.',
  'A GPU replaces a few large cores with many small ones: 132 multiprocessors on an H100, 40 on a free T4.',
  'Each multiprocessor runs several independent warps. While one warp waits for memory, another does arithmetic, so the SM keeps working.',
  'Warps are not free: each needs its own registers and instruction slot. You buy latency hiding with memory and scheduling.',
  'LLM inference is thousands of independent matrix multiplies. That is exactly the shape of work a GPU is built for.',
];

/** One warp: a long wait for memory, a short burst of arithmetic, then repeat. */
const CYCLE = 0.17;
const DUTY = 0.176; // fraction of a cycle a single warp spends doing math

const CPU_CORES = 8;
const SMS = 132;
const COLS = 12;
const CELL_W = 37;
const CELL_H = 13;
const GAP_X = 4;
const GAP_Y = 3.4;

const X0 = 128;
const GRID_W = COLS * (CELL_W + GAP_X) - GAP_X;
const GRID_Y = 140;
const ROWS = Math.ceil(SMS / COLS);
const GRID_H = ROWS * (CELL_H + GAP_Y) - GAP_Y;
const INSET_Y = GRID_Y + GRID_H + 52;

// A warp's phase is its own: real warps stall at uncorrelated moments, and only
// independence makes P(at least one computing) = 1 - (1 - duty)^N true.
// `phase` is where the warp sits at t = 0, in units of cycles; it does the first
// DUTY of every cycle and then waits.
const frac = (x: number) => x - Math.floor(x);
const warpPhase = (sm: number, w: number) => frac(Math.sin(sm * 127.1 + w * 311.7 + 0.5) * 43758.5453);

/** Is this warp doing math at time t? */
const warpBusy = (sm: number, w: number, t: number) => frac(t / CYCLE + warpPhase(sm, w)) < DUTY;

/** Fraction of a unit's warps that are doing math at time t. */
function busyFraction(warps: number, sm: number, t: number) {
  let n = 0;
  for (let w = 0; w < warps; w++) if (warpBusy(sm, w, t)) n++;
  return n / warps;
}

const Y = (r: number) => GRID_Y + r * (CELL_H + GAP_Y);

export default function CpuGpuCores() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [warps, setWarps] = useState(8);
  const clock = useClock(s.playing, 0.08);
  const t = s.playing && !prefersReducedMotion() ? clock : s.step * 0.045;

  const cpuBusy = frac(t / CYCLE) < DUTY;
  const gpuUtil = 1 - Math.pow(1 - DUTY, warps);
  const slot = GRID_W / CPU_CORES;

  const cells = [];
  for (let i = 0; i < SMS; i++) {
    const f = busyFraction(warps, i, t);
    const busy = f > 0;
    const cx = X0 + (i % COLS) * (CELL_W + GAP_X);
    const cy = Y(Math.floor(i / COLS));
    cells.push(
      <g key={i}>
        <rect x={cx} y={cy} width={CELL_W} height={CELL_H} rx={1.5} fill="url(#tps-hatch)" stroke="var(--idle)" strokeWidth={0.8} />
        {busy && <rect x={cx} y={cy} width={CELL_W} height={CELL_H} rx={1.5} fill="var(--compute)" stroke="var(--compute)" strokeWidth={0.8} className="v-cell" />}
      </g>,
    );
  }

  return (
    <Stage
      rootRef={ref}
      title="Why a GPU is thousands of small cores that almost never wait"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', ['idle', 'stalled, waiting for memory']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'CPU computing', value: `${(DUTY * 100).toFixed(0)}%`, kind: cpuBusy ? 'compute' : 'idle', note: 'all 8 stall together' },
          { label: 'GPU computing', value: `${(gpuUtil * 100).toFixed(0)}%`, kind: 'compute', note: `of 132 SMs, ${warps} warps each` },
          { label: 'Against that CPU', value: `${((gpuUtil * SMS) / (DUTY * CPU_CORES)).toFixed(0)}×`, kind: 'token', note: 'arithmetic actually in flight' },
          { label: 'Warps bought', value: fmt.int(SMS * warps), kind: 'memory', note: 'registers and slots spent' },
        ]} />
      }
      controls={
        <Slider label="Warps in flight per multiprocessor" value={warps} min={1} max={16} step={1} onChange={setWarps} format={(v) => String(v)} kind="compute" />
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="Eight CPU cores that stall in lockstep, compared with 132 GPU multiprocessors each running many warps, so the GPU is almost never idle.">
        {/* CPU */}
        <Label x={X0} y={26} tone={1} size={11} bold mono={false}>CPU · 8 big cores, few things in flight</Label>
        <Label x={X0 + GRID_W} y={26} anchor="end" tone={3} size={9.5}>one instruction stream per core</Label>
        {Array.from({ length: CPU_CORES }, (_, i) => {
          const x = X0 + i * slot;
          return (
            <g key={i}>
              <rect x={x + 2} y={38} width={slot - 8} height={44} rx={3}
                fill={cpuBusy ? 'var(--compute)' : 'url(#tps-hatch)'}
                stroke={cpuBusy ? 'var(--compute)' : 'var(--idle)'} strokeWidth={1} className="v-cell" />
              <text x={x + slot / 2} y={64} textAnchor="middle" className="v-label bold" style={{ fontSize: 10.5, fill: cpuBusy ? '#0e1116' : 'var(--text-3)' }}>
                {cpuBusy ? 'math' : 'wait'}
              </text>
            </g>
          );
        })}
        <Label x={X0} y={98} tone={3} size={9.5}>at this instant: {cpuBusy ? 'all 8 computing' : '0 of 8 computing'}</Label>

        <line x1={X0} y1={112} x2={X0 + GRID_W} y2={112} className="v-rule faint" />

        {/* GPU */}
        <Label x={X0} y={132} tone={1} size={11} bold mono={false}>GPU · 132 multiprocessors</Label>
        <Label x={X0 + GRID_W} y={132} anchor="end" tone={3} size={9.5}>amber = at least one warp computing</Label>
        {cells}
        <Label x={X0} y={GRID_Y + GRID_H + 15} tone={3} size={9.5}>
          {SMS} SMs, {fmt.int(SMS * warps)} warps resident · a block stays dark only while every one of its warps is stalled
        </Label>

        {/* zoom into one SM */}
        <Fade show={s.step >= 4}>
          <line x1={X0} y1={INSET_Y - 22} x2={X0 + GRID_W} y2={INSET_Y - 22} className="v-rule faint" />
          <Label x={X0} y={INSET_Y - 6} tone={2} size={10} mono={false}>Zoom into one SM: {warps} warp{warps > 1 ? 's' : ''}, each at its own point in the cycle</Label>
          {Array.from({ length: 16 }, (_, w) => {
            const wslot = GRID_W / 16;
            const active = w < warps;
            const b = active && warpBusy(0, w, t);
            return (
              <g key={w} style={{ opacity: active ? 1 : 0.22 }}>
                <rect x={X0 + w * wslot + 1} y={INSET_Y} width={wslot - 4} height={12} rx={1.5}
                  fill={b ? 'var(--compute)' : active ? 'url(#tps-hatch)' : 'none'}
                  stroke={active ? (b ? 'var(--compute)' : 'var(--idle)') : 'var(--rule-2)'} strokeWidth={0.8} className="v-cell" />
                <text x={X0 + w * wslot + wslot / 2} y={INSET_Y + 26} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>w{w}</text>
              </g>
            );
          })}
          {s.step >= 5 && (
            <Label x={X0} y={INSET_Y + 42} tone={3} size={9.5}>
              {warps <= 4 ? 'too few warps and the stalls show through — drag the slider up to hide them' : 'the extra warps are what buy the hiding, and they cost registers and instruction slots'}
            </Label>
          )}
        </Fade>
      </svg>
    </Stage>
  );
}
