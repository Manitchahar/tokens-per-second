import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Arrow, Fade, fmt } from '../lib/svg';
import { GPUS } from '../lib/perfModel';

const STEPS = [
  'Two roofs, not one',
  'The ridge point decides',
  'Two efficiencies, not one',
  'Put your kernel on the plot',
  'How far below the roof are you?',
  'The three fixes, and which is available',
  'MFU or MBU: which number to quote',
  'The roofline as a decision procedure',
];
const CAPTIONS = [
  'The same point on this graph means different things on different GPUs, because the slope and the ceiling are set by the hardware.',
  'Left of the ridge the memory bandwidth is the wall. Right of it the arithmetic units are the wall. Nothing you write changes which wall you hit.',
  'MFU is how much of the compute ceiling you used. MBU is how much of the bandwidth you used. A memory-bound kernel has a low MFU and that is fine — quote MBU instead.',
  'Compute the FLOPs and the bytes for the kernel as actually executed, including wasted sectors, and you have a point on the graph.',
  'The vertical gap between your point and the roof is your entire optimization budget. A 40% gap means a real 1.6× is available to a better kernel.',
  'Three moves exist: raise intensity (batch, tile), lower bytes (quantize, fuse), or improve efficiency (coalescing, occupancy). Only the first two change which wall you hit.',
  'Quote the efficiency that matches your bound. MFU under 50% on a memory-bound kernel is not a defect; MBU under 60% on a memory-bound kernel is.',
  'Compute intensity, find the ridge, place the point, measure the gap. Four steps, and they answer most "why is this slow" questions.',
];

const PRESET: Record<number, { gpuId: string; m: number }> = {
  0: { gpuId: 'h100', m: 512 },
  1: { gpuId: 'h100', m: 512 },
  2: { gpuId: 'h100', m: 512 },
  3: { gpuId: 'h100', m: 512 },
  4: { gpuId: 'h100', m: 512 },
  5: { gpuId: 'h100', m: 512 },
  6: { gpuId: 'h100', m: 512 },
  7: { gpuId: 'h100', m: 512 },
};

const K = 4096;
const N = 4096;
const BYTES = 2;

/** [m × K]·[K × N] in BF16: 2·m·K·N FLOPs, and (mK + KN + mN)·2 bytes. */
export const gemmIntensity = (m: number) => (2 * m * K * N) / ((m * K + K * N + m * N) * BYTES);

const X0 = 92;
const X1 = 606;
const Y0 = 300;
const Y1 = 46;
const XMIN = -1;
const XMAX = 3.5;
const YMIN = -1;
const YMAX = 3.48;
const X_FAR = 3162;

const px = (I: number) => X0 + ((Math.log10(I) - XMIN) / (XMAX - XMIN)) * (X1 - X0);
const py = (F: number) => Y0 - ((Math.log10(Math.max(F, 1e-3)) - YMIN) / (YMAX - YMIN)) * (Y0 - Y1);

export default function RooflineLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [gpuId, setGpuId] = useState(PRESET[s.step].gpuId);
  const [m, setM] = useState(PRESET[s.step].m);
  useEffect(() => {
    setGpuId(PRESET[s.step].gpuId); setM(PRESET[s.step].m);
  }, [s.step]);

  const gpu = GPUS.find((g) => g.id === gpuId)!;
  const bwT = gpu.bw / 1e12;
  const peakT = gpu.flops / 1e12;
  const ridge = gpu.flops / gpu.bw;

  const I = gemmIntensity(m);
  const flops = 2 * m * K * N;
  const bytes = (m * K + K * N + m * N) * BYTES;

  // What a well-tuned kernel actually reaches on each bound.
  const MBU = 0.85;
  const MFU = 0.65;

  // The roof at this intensity: the bandwidth slope up to the ridge, the flat
  // compute ceiling past it. This is the theoretical maximum rate.
  const roofHere = Math.min(bwT * I, peakT);
  const bound: 'memory' | 'compute' = I * gpu.bw > gpu.flops ? 'compute' : 'memory';

  // What a *good* kernel actually reaches, which is below the roof by exactly the
  // gap the lesson is about: 85% of the bound it is limited by.
  const achieved = bound === 'memory' ? roofHere * MBU : roofHere * MFU;

  // gap: what a perfect kernel could do vs what this one does
  const tRoof = Math.max(flops / gpu.flops, bytes / gpu.bw);
  const tReal = Math.max(flops / (gpu.flops * MFU), bytes / (gpu.bw * MBU));
  const speedupLeft = tReal / tRoof;

  // The efficiency that matches the bound: only that one is a fair thing to quote.
  const quotedEff = bound === 'compute' ? MFU : MBU;

  const ax = useTween(I, 600);
  const ay = useTween(achieved, 600);

  const roof = useMemo(() => {
    const pts: string[] = [];
    const from = 0.1;
    for (let i = 0; i <= 24; i++) {
      const Ii = from * Math.pow(10, (Math.log10(ridge / from) * i) / 24);
      pts.push(`${i ? 'L' : 'M'}${px(Ii).toFixed(1)},${py(bwT * Ii).toFixed(1)}`);
    }
    pts.push(`L${px(X_FAR).toFixed(1)},${py(peakT).toFixed(1)}`);
    return pts.join(' ');
  }, [ridge, bwT, peakT]);

  const ceilingY = py(peakT);
  const ridgeX = px(ridge);

  return (
    <Stage
      rootRef={ref}
      title="The roofline: place your kernel, then measure the gap to the roof"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['idle', 'unreachable'], ['token', 'your kernel'], ['error', 'headroom']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Arithmetic intensity', value: `${I < 10 ? I.toFixed(2) : fmt.int(I)} FLOP/byte`, kind: 'text', note: `m = ${fmt.int(m)} rows` },
          { label: 'Limited by', value: bound === 'compute' ? 'compute' : 'memory', kind: bound === 'compute' ? 'compute' : 'memory', note: bound === 'compute' ? 'the math units' : 'the memory system' },
          { label: 'Quote this', value: `${(quotedEff * 100).toFixed(0)}% ${bound === 'compute' ? 'MFU' : 'MBU'}`, kind: bound === 'compute' ? 'compute' : 'memory', note: 'the efficiency that matches the bound' },
          { label: 'Available headroom', value: `${speedupLeft.toFixed(2)}×`, kind: speedupLeft > 1.3 ? 'error' : 'token', note: 'to a perfect kernel' },
        ]} />
      }
      controls={
        <>
          <Slider label="Rows per matrix multiply (m)" value={m} min={1} max={4096} log onChange={setM} format={(v) => fmt.int(v)} kind="compute" />
          <Segmented
            label="GPU"
            value={gpuId}
            options={GPUS.filter((g) => ['t4', 'a100', 'h100', 'b200'].includes(g.id)).map((g) => ({ value: g.id, label: g.id.toUpperCase() }))}
            onChange={setGpuId}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A roofline plot of achievable performance against arithmetic intensity for a selectable GPU, with the bandwidth slope, compute ceiling, ridge point, a moving marker for a matrix multiply, and a vertical arrow showing the headroom between it and the roof.">
        <line x1={X0} y1={Y0} x2={X1} y2={Y0} className="v-rule" />
        <line x1={X0} y1={Y0} x2={X0} y2={Y1 - 14} className="v-rule" />
        {[0.1, 1, 10, 100, 1000].map((v) => (
          <Label key={v} x={px(v)} y={Y0 + 14} anchor="middle" tone={3} size={9}>{v < 1 ? '0.1' : v}</Label>
        ))}
        {[0.1, 1, 10, 100, 1000].map((v) => (
          <Label key={`y${v}`} x={X0 - 6} y={py(v) + 3} anchor="end" tone={3} size={9}>{v < 1 ? '0.1' : v}</Label>
        ))}
        <Label x={(X0 + X1) / 2} y={Y0 + 30} anchor="middle" tone={2} size={10.5} mono={false}>arithmetic intensity — FLOPs per byte</Label>
        <text x={22} y={(Y0 + Y1) / 2} transform={`rotate(-90 22 ${(Y0 + Y1) / 2})`} textAnchor="middle" className="v-label t2" style={{ fontSize: 10.5 }}>achieved TFLOP/s</text>

        <path d={`${roof} L${X1},${Y0} L${px(0.1)},${Y0} Z`} fill="rgb(74 82 98 / 0.13)" className="v-fade" style={{ opacity: s.step >= 1 ? 1 : 0 }} />
        <path d={roof} fill="none" stroke="var(--text-2)" strokeWidth={1.6} className="v-fade" style={{ opacity: s.step >= 1 ? 1 : 0 }} />

        <Fade show={s.step >= 1}>
          <line x1={X0} x2={X1} y1={ceilingY} y2={ceilingY} stroke="var(--compute)" strokeWidth={1.4} strokeDasharray="6 4" />
          <Label x={X0 + 4} y={ceilingY - 7} size={10} kind="compute">{`${gpu.name}: ${fmt.si(peakT)} TFLOP/s`}</Label>
        </Fade>
        <Fade show={s.step >= 1}>
          <Label x={px(0.115)} y={py(bwT * 0.115) - 11} className="v-label" style={{ fontSize: 9.5, fill: 'var(--memory)' }}>
            bandwidth: {bwT} TB/s
          </Label>
        </Fade>
        <Fade show={s.step >= 1}>
          <line x1={ridgeX} x2={ridgeX} y1={Y1 - 8} y2={Y0} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="3 4" />
          <circle cx={ridgeX} cy={ceilingY} r={4} fill="var(--text)" />
          <Label x={ridgeX} y={Y1 - 14} anchor="middle" tone={1} size={10}>{`ridge: ${fmt.int(ridge)} FLOP/byte`}</Label>
        </Fade>

        {/* the moving point */}
        <g>
          <line x1={px(ax)} x2={px(ax)} y1={py(ay)} y2={Y0} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="2 3" className="v-fade" />
          <g className="v-move" style={{ transform: `translate(${px(ax)}px, ${py(ay)}px)` }}>
            <circle r={6} fill={bound === 'compute' ? 'var(--compute)' : 'var(--memory)'} stroke="var(--ground-2)" strokeWidth={2.5} />
          </g>
          <Label x={px(ax)} y={py(ay) - 12} anchor="middle" size={9.5} kind={bound === 'compute' ? 'compute' : 'memory'}>
            {`m = ${fmt.int(m)} · ${bound}-bound`}
          </Label>

          {/* headroom */}
          <Fade show={s.step >= 4}>
            <line x1={px(ax) + 10} x2={px(ax) + 10} y1={py(roofHere)} y2={py(ay)} stroke="var(--error)" strokeWidth={1.6} />
            <Arrow x1={px(ax) + 10} x2={px(ax) + 10} y1={py(ay) + 2} y2={py(roofHere) - 4} kind="error" width={1.4} />
            <Label x={px(ax) + 18} y={(py(roofHere) + py(ay)) / 2} size={9.5} kind="error">
              {speedupLeft > 1.15 ? `${speedupLeft.toFixed(2)}× available` : 'near the roof'}
            </Label>
          </Fade>
        </g>

        {/* the two efficiencies */}
        <Fade show={s.step >= 2}>
          <Label x={X0} y={Y0 - 16} size={10} kind={bound === 'compute' ? 'compute' : 'memory'}>
            {bound === 'compute'
              ? `MFU ${(MFU * 100).toFixed(0)}% — the number that matters here`
              : `MBU ${(MBU * 100).toFixed(0)}% — the number that matters here`}
          </Label>
          <Label x={X0} y={Y0 - 4} tone={3} size={9.5}>
            {bound === 'compute'
              ? 'memory utilization is low, and that is expected: the bytes kept pace.'
              : 'MFU is low, and that is expected: each byte only feeds ~1 FLOP.'}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
