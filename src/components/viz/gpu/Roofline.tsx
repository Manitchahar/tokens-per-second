import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, fmt } from '../lib/svg';
import { GPUS } from '../lib/perfModel';

// A 4096 × 4096 weight in BF16: the shape of one of Llama-3.1-8B's matrices.
const K = 4096;
const N = 4096;
const BYTES = 2;

/** FLOPs per byte moved for [m × K] · [K × N], both operands in and out. */
export const gemmIntensity = (m: number) => (2 * m * K * N) / ((m * K + K * N + m * N) * BYTES);
export const gemmFlops = (m: number) => 2 * m * K * N;
export const gemmBytes = (m: number) => (m * K + K * N + m * N) * BYTES;

/** Llama-3.1-8B attention at context T, in BF16: FLOPs per byte of KV read. */
const KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * BYTES;
const attnIntensity = (t: number) => (4 * t * t * K) / (t * KV_BYTES_PER_TOKEN);

const STEPS = [
  'Two numbers',
  'The roofline',
  'The ridge point',
  'Prefill sits on the right',
  'Decode slides left',
  'The wasted machine',
  'The whole course in one picture',
];
const CAPTIONS = [
  'Every GPU has two headline numbers: how fast it can do arithmetic, and how fast it can read memory. Plot both on one graph.',
  'A job can never go faster than its bytes allow. Read x bytes and do y FLOPs, and the bandwidth line is a hard floor you cannot pass.',
  'Past that point there are enough FLOPs to keep every math unit busy, and only the compute ceiling applies. That is the flat top.',
  'Prefill: one big matrix multiply with hundreds of rows. Each weight is reused for every row, so there are hundreds of FLOPs per byte. Compute-bound.',
  'Decode: the same weight, but one row at a time. Every byte fetched feeds one multiply-add. Deep on the left, hard against the bandwidth floor. Memory-bound.',
  'The vertical gap between a point and the roof is performance thrown away. Closing it is the entire job of kernel work, batching, quantization and fusion.',
  'Almost every idea in this course is one of two moves: more FLOPs per byte, or fewer bytes. There is no third move.',
];

const PRESET: Record<number, number> = { 0: 1, 1: 1, 2: 1, 3: 512, 4: 1, 5: 512, 6: 8 };

const X0 = 78;
const X1 = 604;
const Y0 = 322;
const Y1 = 52;
const XMIN = -1; // 0.1 FLOP/byte
const XMAX = 3.5;
const YMIN = -1; // 0.1 TFLOP/s
const YMAX = 3.48;
const X_FAR = 3162; // right edge, in FLOP/byte

const px = (I: number) => X0 + ((Math.log10(I) - XMIN) / (XMAX - XMIN)) * (X1 - X0);
const py = (F: number) => Y0 - ((Math.log10(Math.max(F, 1e-3)) - YMIN) / (YMAX - YMIN)) * (Y0 - Y1);

const REFS = [
  { name: 'activation (SiLU, add)', I: 0.3, kind: 'memory' as const },
  { name: 'write the new KV', I: 0.5, kind: 'memory' as const },
  { name: 'decode GEMM', I: gemmIntensity(1), kind: 'memory' as const },
  { name: 'attention, 1k ctx', I: attnIntensity(1024), kind: 'memory' as const },
  { name: 'prefill GEMM', I: gemmIntensity(512), kind: 'compute' as const },
];

export default function Roofline() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [gpuId, setGpuId] = useState('h100');
  const [m, setM] = useState(1);
  useEffect(() => { setM(PRESET[s.step]); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const gpu = GPUS.find((g) => g.id === gpuId)!;
  const bwTflops = gpu.bw / 1e12;
  const peakTflops = gpu.flops / 1e12;
  const ridge = gpu.flops / gpu.bw;

  const I = gemmIntensity(m);
  const flops = gemmFlops(m);
  const bytes = gemmBytes(m);
  const tRoof = Math.max(flops / gpu.flops, bytes / gpu.bw);
  const tReal = Math.max(flops / (gpu.flops * 0.55), bytes / (gpu.bw * 0.8));
  const eff = tRoof / tReal;
  const bound = flops / gpu.flops > bytes / gpu.bw ? 'compute' : 'memory';
  const showPoint = s.step >= 3;

  // where you end up, versus where the roof lets you
  const achieved = flops / tReal / 1e12;                       // TFLOP/s
  const roofHere = Math.min(bwTflops * I, peakTflops);         // TFLOP/s
  const wasted = Math.max(0, (roofHere - achieved) / roofHere);

  const ax = useTween(I, 600);
  const roof = useMemo(() => {
    const pts: string[] = [];
    const from = 0.1;
    for (let i = 0; i <= 24; i++) {
      const Ii = from * Math.pow(10, (Math.log10(ridge / from) * i) / 24);
      pts.push(`${i ? 'L' : 'M'}${px(Ii).toFixed(1)},${py(bwTflops * Ii).toFixed(1)}`);
    }
    pts.push(`L${px(X_FAR).toFixed(1)},${py(peakTflops).toFixed(1)}`);
    return pts.join(' ');
  }, [ridge, bwTflops, peakTflops]);
  const ceilingY = py(peakTflops);
  const ridgeX = px(ridge);

  return (
    <Stage
      rootRef={ref}
      title="The roofline: FLOPs per byte decides which limit you hit"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['idle', 'unreachable']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Arithmetic intensity', value: `${I < 10 ? I.toFixed(2) : fmt.int(I)} FLOP/byte`, kind: 'text', note: `${fmt.int(m)} row${m === 1 ? '' : 's'} at a time` },
          { label: 'Limited by', value: bound === 'compute' ? 'compute' : 'memory', kind: bound === 'compute' ? 'compute' : 'memory', note: bound === 'compute' ? 'the math units' : 'the memory system' },
          { label: 'Best possible', value: `${(tRoof * 1e6).toFixed(0)} µs`, kind: 'idle', note: 'at 100% of peak' },
          { label: 'Realistic', value: `${(tReal * 1e6).toFixed(0)} µs`, kind: 'text', note: `${(eff * 100).toFixed(0)}% of the roof` },
        ]} />
      }
      controls={
        <>
          <Slider
            label="Rows per matrix multiply (m) — 1 is decode, hundreds is prefill"
            value={m}
            min={1}
            max={4096}
            log
            onChange={setM}
            format={(v) => `${fmt.int(v)}`}
            kind="compute"
          />
          <Segmented
            label="GPU"
            value={gpuId}
            options={GPUS.filter((g) => ['t4', 'a100', 'h100', 'b200'].includes(g.id)).map((g) => ({ value: g.id, label: g.id.toUpperCase() }))}
            onChange={setGpuId}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A roofline plot: performance in TFLOP per second against arithmetic intensity in FLOPs per byte, with a bandwidth slope, a compute ceiling, and a moving marker for a matrix multiply at a chosen number of rows.">
        {/* axes */}
        <line x1={X0} y1={Y0} x2={X1} y2={Y0} className="v-rule" />
        <line x1={X0} y1={Y0} x2={X0} y2={Y1 - 14} className="v-rule" />
        {[0.1, 1, 10, 100, 1000].map((v) => (
          <Label key={v} x={px(v)} y={Y0 + 14} anchor="middle" tone={3} size={9}>{v < 1 ? '0.1' : v}</Label>
        ))}
        {[0.1, 1, 10, 100, 1000].map((v) => (
          <Label key={`y${v}`} x={X0 - 6} y={py(v) + 3} anchor="end" tone={3} size={9}>{v < 1 ? '0.1' : v}</Label>
        ))}
        <Label x={(X0 + X1) / 2} y={Y0 + 30} anchor="middle" tone={2} size={10.5} mono={false}>arithmetic intensity — FLOPs done per byte read</Label>
        <text x={20} y={(Y0 + Y1) / 2} transform={`rotate(-90 20 ${(Y0 + Y1) / 2})`} textAnchor="middle" className="v-label t2" style={{ fontSize: 10.5 }}>achieved TFLOP/s</text>

        {/* unreachable region under the roof */}
        <path d={`${roof} L${X1},${Y0} L${px(0.1)},${Y0} Z`} fill="rgb(74 82 98 / 0.13)" className="v-fade" style={{ opacity: s.step >= 1 ? 1 : 0 }} />
        <path d={roof} fill="none" stroke="var(--text-2)" strokeWidth={1.6} className="v-fade" style={{ opacity: s.step >= 1 ? 1 : 0 }} />

        {/* compute ceiling */}
        <Fade show={s.step >= 1}>
          <line x1={X0} x2={X1} y1={ceilingY} y2={ceilingY} stroke="var(--compute)" strokeWidth={1.4} strokeDasharray="6 4" />
          <Label x={X0 + 4} y={ceilingY - 7} size={10} kind="compute">{`${gpu.name}: ${fmt.si(peakTflops)} TFLOP/s peak`}</Label>
        </Fade>
        <Fade show={s.step >= 1}>
          <text x={px(0.115)} y={py(bwTflops * 0.115) - 11} className="v-label" style={{ fontSize: 9.5, fill: 'var(--memory)' }}>
            bandwidth: {bwTflops} TB/s of bytes per second
          </text>
        </Fade>

        {/* ridge point */}
        <Fade show={s.step >= 2}>
          <line x1={ridgeX} x2={ridgeX} y1={Y1 - 8} y2={Y0} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="3 4" />
          <circle cx={ridgeX} cy={ceilingY} r={4} fill="var(--text)" />
          <Label x={ridgeX} y={Y1 - 14} anchor="middle" tone={1} size={10}>{`ridge: ${fmt.int(ridge)} FLOP/byte`}</Label>
        </Fade>

        {/* reference operations */}
        <Fade show={s.step >= 3}>
          {REFS.map((r) => {
            const y = py(Math.min(bwTflops * r.I, peakTflops));
            const left = r.I > ridge; // keep the crowded right-hand side clear of the marker
            return (
              <g key={r.name} className="v-fade">
                <circle cx={px(r.I)} cy={y} r={3} fill="none" stroke={r.kind === 'compute' ? 'var(--compute)' : 'var(--memory)'} strokeWidth={1.2} />
                <Label x={px(r.I) + (left ? -7 : 7)} y={y + 3.5} anchor={left ? 'end' : 'start'} tone={3} size={9}>{r.name}</Label>
              </g>
            );
          })}
        </Fade>

        {/* the moving point */}
        {showPoint && (
          <g>
            <line x1={px(ax)} x2={px(ax)} y1={py(achieved)} y2={Y0} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="2 3" className="v-fade" />
            <g className="v-move" style={{ transform: `translate(${px(ax)}px, ${py(achieved)}px)` }}>
              <circle r={6} fill={bound === 'compute' ? 'var(--compute)' : 'var(--memory)'} stroke="var(--ground-2)" strokeWidth={2.5} />
            </g>
            {s.step >= 5 && (
              <g>
                <line x1={px(ax)} x2={px(ax)} y1={py(roofHere)} y2={py(achieved)} stroke="var(--error)" strokeWidth={1.4} />
                <line x1={px(ax) - 4} x2={px(ax) + 4} y1={py(roofHere)} y2={py(roofHere)} stroke="var(--error)" strokeWidth={1.4} />
                <Label x={px(ax) + 8} y={(py(roofHere) + py(achieved)) / 2 + 3.5} size={9.5} kind="error">{`${(wasted * 100).toFixed(0)}% of the machine wasted`}</Label>
              </g>
            )}
            {s.step >= 3 && s.step < 5 && (
              <Label x={px(ax)} y={py(achieved) - 12} anchor="middle" size={9.5} kind={bound === 'compute' ? 'compute' : 'memory'}>
                {`m = ${fmt.int(m)} · ${bound}-bound`}
              </Label>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}

