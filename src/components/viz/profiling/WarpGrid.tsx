import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Arrow, Dim, Fade, Chip, fmt } from '../lib/svg';
import { GPUS } from '../lib/perfModel';
import { gridFor, waves, waveEfficiency } from './profiling';

const STEPS = [
  'One kernel launch',
  'Threads are grouped into warps',
  'Warps are grouped into blocks',
  'Blocks are the unit the grid scheduler sees',
  'Not every block is resident at once',
  'Occupancy: how full the machine is',
  'Too many blocks, and the tail bites',
  'What to actually do with this',
];
const CAPTIONS = [
  'When you call a CUDA kernel you choose how many threads to launch. The GPU takes that grid and chops it into warps.',
  'A warp is 32 threads that issue the same instruction together. All of its lanes are busy or all idle: there is no partial warp.',
  'Threads are bundled into blocks of 32, 64, 128, 256 or 1024. Each block runs on one SM and cannot straddle two.',
  'The grid is the whole launch: hundreds of thousands of blocks, each independent. They are scheduled onto the SMs as slots free up.',
  'Each SM can host only so many resident blocks, set by registers and shared memory. The rest wait in line.',
  'Occupancy is resident warps divided by the SM maximum. It is not a goal in itself: it is how much stall-hiding you bought.',
  'One round of blocks is a wave. If the grid is 1.05 waves, the last wave uses 5% of the machine and 95% of the time.',
  'Pick block sizes that divide your work cleanly, keep occupancy high enough to hide latency, and be suspicious of grids just over a wave boundary.',
];

const PRESET: Record<number, { elems: number; tpb: number; blocksPerSm: number }> = {
  0: { elems: 1_000_000, tpb: 256, blocksPerSm: 8 },
  1: { elems: 1_000_000, tpb: 256, blocksPerSm: 8 },
  2: { elems: 1_000_000, tpb: 256, blocksPerSm: 8 },
  3: { elems: 1_000_000, tpb: 256, blocksPerSm: 8 },
  4: { elems: 1_000_000, tpb: 256, blocksPerSm: 8 },
  5: { elems: 1_000_000, tpb: 256, blocksPerSm: 8 },
  6: { elems: 1_000_000, tpb: 256, blocksPerSm: 8 },
  7: { elems: 1_000_000, tpb: 256, blocksPerSm: 8 },
};

const TPBS = [32, 64, 128, 256, 512, 1024];
const COLS = 8;
const CELL_W = 40;
const CELL_H = 15;
const GAP = 5;
const GRID_W = COLS * (CELL_W + GAP) - GAP;

export default function WarpGrid() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const p = PRESET[s.step];
  const [elems, setElems] = useState(p.elems);
  const [tpb, setTpb] = useState(p.tpb);
  const [bps, setBps] = useState(p.blocksPerSm);
  useEffect(() => {
    setElems(PRESET[s.step].elems); setTpb(PRESET[s.step].tpb); setBps(PRESET[s.step].blocksPerSm);
  }, [s.step]);

  const g = gridFor(elems, tpb);
  const shown = Math.min(g.blocks, 48);
  const shownWaves = Math.min(Math.ceil(waves(g.blocks, bps)), 4);
  const wEff = waveEfficiency(g.blocks, bps);
  const warps = tpb / 32;
  const residentWarps = bps * warps;
  const maxResidentWarps = 64;
  const occ = Math.min(1, residentWarps / maxResidentWarps);

  const X0 = (640 - GRID_W) / 2;
  const gy = (r: number) => 176 + r * (CELL_H + GAP);

  return (
    <Stage
      rootRef={ref}
      title="From one launch to 132 multiprocessors: threads, warps, blocks, waves"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', ['memory', 'resident on an SM'], ['idle', 'waiting for a slot'], 'error']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Elements to process', value: fmt.int(elems), kind: 'text', note: 'one thread per element' },
          { label: 'Blocks in the grid', value: fmt.int(g.blocks), kind: 'compute', note: `${tpb} threads each` },
          { label: 'Waves needed', value: shownWaves >= 4 ? '4+' : fmt.si(shownWaves), kind: 'compute', note: `${bps} blocks resident at once` },
          { label: 'Occupancy', value: `${(occ * 100).toFixed(0)}%`, kind: occ > 0.5 ? 'token' : 'error', note: `${fmt.int(residentWarps)} of 64 warp slots used` },
        ]} />
      }
      controls={
        <>
          <Slider label="Elements (batch × tokens × hidden) — e.g. 32 × 4096" value={elems} min={1024} max={16_777_216} log onChange={setElems} format={(v) => fmt.int(v)} kind="compute" />
          <Slider label="Threads per block" value={tpb} min={32} max={1024} step={32} onChange={setTpb} format={(v) => `${v}`} kind="memory" />
          <Slider label="Blocks resident per SM (register & shared-memory limited)" value={bps} min={1} max={16} step={1} onChange={setBps} format={(v) => `${v}`} kind="memory" />
        </>
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A CUDA grid of blocks being scheduled onto 132 multiprocessors in waves, showing thread blocks made of warps, resident and waiting blocks, occupancy, and the cost of a partial final wave.">
        {/* ── step 0: the raw thread count ── */}
        <Fade show={s.step === 0}>
          <Label x={320} y={40} anchor="middle" tone={1} size={12} bold mono={false}>
            {`You ask for ${fmt.int(elems)} threads. One per element.`}
          </Label>
          <Label x={320} y={60} anchor="middle" tone={3} size={10}>
            The hardware has to decide who gets which core. That decision has three layers.
          </Label>
          {Array.from({ length: 3 }, (_, i) => (
            <Chip key={i} x={208 + i * 80} y={88} kind="text" text={['threads', 'warps', 'blocks'][i]} w={68} />
          ))}
          <Arrow x1={276} y1={99} x2={284} y2={99} kind="text" muted />
          <Arrow x1={356} y1={99} x2={364} y2={99} kind="text" muted />
        </Fade>

        {/* ── the block grid, always the stage's centre of gravity ── */}
        <Label x={X0} y={150} tone={2} size={11} bold mono={false}>
          {s.step === 0 ? 'the grid the launch asked for' : 'a piece of the grid: each cell is one thread block'}
        </Label>
        <Label x={X0 + GRID_W} y={150} anchor="end" tone={3} size={9.5}>
          {`${fmt.int(g.blocks)} blocks total · ${tpb} threads each · ${warps} warps each`}
        </Label>

        {Array.from({ length: shown }, (_, i) => {
          const inFirstWave = i < bps;
          const full = inFirstWave;
          return (
            <g key={i} className="v-fade">
              <rect
                x={X0 + (i % COLS) * (CELL_W + GAP)}
                y={gy(Math.floor(i / COLS))}
                width={CELL_W}
                height={CELL_H}
                rx={2}
                fill={full ? 'var(--memory)' : 'url(#tps-hatch)'}
                stroke={full ? 'var(--memory)' : 'var(--idle)'}
                strokeWidth={0.9}
              />
              <Label x={X0 + (i % COLS) * (CELL_W + GAP) + CELL_W / 2} y={gy(Math.floor(i / COLS)) + CELL_H / 2 + 3.4}
                anchor="middle" size={8} tone={full ? 1 : 3}>{`w${i % 8}`}</Label>
            </g>
          );
        })}

        {/* one block unpacked into warps */}
        <Fade show={s.step >= 1}>
          <Label x={X0} y={gy(Math.ceil(shown / COLS)) + 6} tone={3} size={9}>
            {s.step === 1 ? 'one block, seen close up: its threads are 32-lane warps' : ''}
          </Label>
        </Fade>

        {/* ── the SMs ── */}
        <Fade show={s.step >= 3}>
          <line x1={40} y1={286} x2={600} y2={286} className="v-rule faint" />
          <Label x={40} y={306} tone={2} size={10.5} bold mono={false}>
            {`132 SMs × ${bps} blocks resident = ${fmt.int(bps * 132)} blocks running at once`}
          </Label>
          <Label x={600} y={306} anchor="end" tone={3} size={9.5}>
            {`remaining ${fmt.int(Math.max(0, g.blocks - bps * 132))} blocks are in the queue`}
          </Label>
          {Array.from({ length: 24 }, (_, i) => (
            <rect key={i} x={40 + i * 23.5} y={316} width={19} height={12} rx={2}
              fill={i % 3 === 0 ? 'var(--memory)' : 'var(--memory-dim)'} stroke="var(--memory)" strokeWidth={0.7} className="v-cell" />
          ))}
          <Label x={40} y={340} tone={3} size={9}>
            {`24 SMs drawn out of 132. Each running up to ${bps} blocks of ${warps} warps.`}
          </Label>
        </Fade>

        {/* ── occupancy bar ── */}
        <Fade show={s.step >= 4}>
          <Label x={40} y={364} tone={2} size={10} bold mono={false}>Occupancy</Label>
          <rect x={112} y={356} width={372} height={11} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={112} y={356} width={372 * occ} height={11} rx={2}
            fill={occ > 0.5 ? 'var(--token)' : 'var(--error)'} stroke={occ > 0.5 ? 'var(--token)' : 'var(--error)'} strokeWidth={1}
            style={{ transition: 'width .6s var(--ease-out)' }} />
          <Label x={112 + 372 + 8} y={365} size={9.5} kind={occ > 0.5 ? 'token' : 'error'}>
            {`${(occ * 100).toFixed(0)}% · ${fmt.int(residentWarps)}/${maxResidentWarps} slots`}
          </Label>
        </Fade>

        {/* ── the tail: wave quantization ── */}
        <Fade show={s.step >= 6}>
          <Label x={40} y={382} tone={2} size={10} bold mono={false}>Wave eff.</Label>
          <Label x={112} y={382} tone={1} size={9.5} kind={wEff > 0.85 ? 'token' : 'error'}>
            {shownWaves >= 4 ? '4+ waves: tail amortised, fine'
              : `${waves(g.blocks, bps).toFixed(2)} waves, last one ${(wEff * 100).toFixed(0)}% full`}
          </Label>
          <Dim x1={430} x2={600} y={382} label={wEff > 0.85 ? 'small tail' : 'quantization loss'} kind={wEff > 0.85 ? 'token' : 'error'} />
        </Fade>
      </svg>
    </Stage>
  );
}
