import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Arrow, Dim, Fade, Span, Ruler, fmt } from '../lib/svg';
import { decodePhase } from './profiling';

const STEPS = [
  'Profile the same model twice',
  'Prefill: dominated by GEMM',
  'Prefill: the number to quote',
  'Decode: a different profile entirely',
  'The SOL percentage, read properly',
  'The line between them is context',
  'What to fix in each phase',
  'Verify the fix the same way',
];
const CAPTIONS = [
  'Same weights, same kernels, two completely different profiles. If your optimization only helps one of these, you have found out which problem you were solving.',
  'Prefill is thousands of rows against each weight matrix, so it is compute-bound and the profile is dominated by large cuBLAS GEMM bars.',
  'A prefill GEMM at 60% MFU is healthy. Below 50% on a large square GEMM means cuBLAS chose badly for the shape and a different library or a Triton kernel may beat it.',
  'Decode is one row per sequence. The profile is still mostly GEMM, but now every one of those bars is bound by streaming 16 GB of weights from HBM.',
  'The Speed of Light percentage is the metric. Memory throughput is the one that matters in decode, and 76% is a well-tuned number, not a mediocre one.',
  'As context grows, the KV cache becomes the dominant byte stream and the attention bar overtakes the GEMMs. The profile changes shape again.',
  'Prefill wants better GEMMs and fusion of the elementwise ops. Decode wants fewer bytes — quantization, paged layout, larger batch — and almost never a better GEMM.',
  'Re-profile with the identical workload and report MBU or MFU, not tokens per second. If the number did not move, the fix did not work.',
];

const PRESET: Record<number, { ctx: number; batch: number }> = {
  0: { ctx: 2048, batch: 32 },
  1: { ctx: 2048, batch: 32 },
  2: { ctx: 2048, batch: 32 },
  3: { ctx: 2048, batch: 32 },
  4: { ctx: 2048, batch: 32 },
  5: { ctx: 2048, batch: 32 },
  6: { ctx: 2048, batch: 32 },
  7: { ctx: 2048, batch: 32 },
};

const X0 = 92;
const X1 = 612;
const W = X1 - X0;

/** One transformer block's decode step, split into the bars a trace actually shows. */
const DECODE_BARS = [
  { name: 'qkv proj', frac: 0.17, kind: 'compute' as const },
  { name: 'rope + kv write', frac: 0.05, kind: 'memory' as const },
  { name: 'attention', frac: 0.13, kind: 'compute' as const },
  { name: 'o proj', frac: 0.10, kind: 'compute' as const },
  { name: 'norm', frac: 0.02, kind: 'memory' as const },
  { name: 'gate_up', frac: 0.27, kind: 'compute' as const },
  { name: 'silu·mul', frac: 0.06, kind: 'memory' as const },
  { name: 'down proj', frac: 0.17, kind: 'compute' as const },
  { name: 'norm + head', frac: 0.03, kind: 'memory' as const },
];

const PREFILL_BARS = [
  { name: 'qkv proj', frac: 0.17, kind: 'compute' as const },
  { name: 'rope + kv write', frac: 0.03, kind: 'memory' as const },
  { name: 'attention', frac: 0.14, kind: 'compute' as const },
  { name: 'o proj', frac: 0.10, kind: 'compute' as const },
  { name: 'norm', frac: 0.01, kind: 'memory' as const },
  { name: 'gate_up', frac: 0.27, kind: 'compute' as const },
  { name: 'silu·mul', frac: 0.02, kind: 'memory' as const },
  { name: 'down proj', frac: 0.17, kind: 'compute' as const },
  { name: 'norm + head', frac: 0.02, kind: 'memory' as const },
];

export default function PhaseTrace() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const p = PRESET[s.step];
  const [ctx, setCtx] = useState(p.ctx);
  const [batch, setBatch] = useState(p.batch);
  useEffect(() => {
    setCtx(PRESET[s.step].ctx); setBatch(PRESET[s.step].batch);
  }, [s.step]);

  const d = decodePhase({ batch, ctx });
  const bytes = d.weightBytes + d.kvBytes;
  const mbu = Math.min(1, bytes / d.gpu.bw / d.t);
  const kvShare = d.kvBytes / bytes;

  // Prefill: 512-token chunk, the default chunk budget on an H100.
  const CHUNK = 512;
  const pfFlops = 2 * 8.03e9 * CHUNK;
  const pfMfu = 0.6;
  const pfTime = pfFlops / (d.gpu.flops * pfMfu);

  const decodeT = useTween(d.t, 500);
  const prefillT = useTween(pfTime, 500);

  const rows: { y: number; label: string; bars: typeof DECODE_BARS }[] = [
    { y: 120, label: 'prefill · 512-token chunk', bars: PREFILL_BARS },
    { y: 190, label: 'decode · 1 token, 32 sequences', bars: DECODE_BARS },
  ];

  return (
    <Stage
      rootRef={ref}
      title="The same model, profiled twice: what changes between prefill and decode"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['token', 'the metric that matters'], ['error', 'headroom']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Decode step', value: `${(decodeT * 1e3).toFixed(1)} ms`, kind: 'memory', note: `batch ${batch}, ctx ${fmt.int(ctx)}` },
          { label: 'Decode MBU', value: `${(mbu * 100).toFixed(0)}%`, kind: mbu > 0.8 ? 'token' : 'error', note: mbu > 0.8 ? 'at the floor, stop' : 'headroom left' },
          { label: 'KV share of traffic', value: `${(kvShare * 100).toFixed(0)}%`, kind: 'memory', note: `${(d.kvBytes / 1e9).toFixed(1)} GB of ${(bytes / 1e9).toFixed(1)} GB` },
          { label: 'Prefill chunk', value: `${(prefillT * 1e3).toFixed(0)} ms`, kind: 'compute', note: `512 tokens at ${(pfMfu * 100).toFixed(0)}% MFU` },
        ]} />
      }
      controls={
        <>
          <Slider label="Context length (tokens)" value={ctx} min={128} max={32768} log onChange={setCtx} format={(v) => fmt.int(v)} kind="memory" />
          <Slider label="Batch (sequences decoding)" value={batch} min={1} max={256} log onChange={setBatch} format={(v) => fmt.int(v)} kind="compute" />
          <Segmented
            label="Phase"
            value={s.step < 4 ? 'prefill' : 'decode'}
            options={[{ value: 'prefill', label: 'Prefill' }, { value: 'decode', label: 'Decode' }]}
            onChange={(v) => { if (v === 'prefill') setBatch(1); }}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="Two profiler timelines stacked: a prefill step dominated by large compute-bound GEMM bars, and a decode step with the same kernels but a much smaller elementwise share, plus a byte breakdown showing weights against KV cache as context grows.">
        <Label x={16} y={30} tone={1} size={11} bold mono={false}>One transformer block, two profiles</Label>
        <Label x={624} y={30} anchor="end" tone={3} size={9.5}>bar widths are proportional to time</Label>

        {rows.map((r) => {
          let x = X0;
          return (
            <g key={r.label}>
              <Label x={16} y={r.y + 18} tone={2} size={10} bold mono={false}>{r.label}</Label>
              <Label x={16} y={r.y + 32} tone={3} size={8.5} mono={false}>
                {r.label.startsWith('prefill') ? 'compute-bound · quote MFU' : 'memory-bound · quote MBU'}
              </Label>
              {r.bars.map((b) => {
                const w = b.frac * W;
                const el = (
                  <Span key={b.name} x={x} y={r.y} w={w} h={26} kind={b.kind} label={b.name} />
                );
                x += w;
                return el;
              })}
              <Ruler x={X0} y={r.y + 30} w={W} ticks={[{ at: 0 }, { at: 0.5 }, { at: 1 }]} label={r.label.startsWith('prefill') ? `${(prefillT * 1e3).toFixed(0)} ms` : `${(decodeT * 1e3).toFixed(1)} ms`} />
            </g>
          );
        })}

        {/* the byte breakdown — the real difference between the two phases */}
        <Fade show={s.step >= 3}>
          <line x1={16} y1={262} x2={624} y2={262} className="v-rule faint" />
          <Label x={16} y={284} tone={2} size={11} bold mono={false}>What a decode step has to read</Label>
          <Label x={624} y={284} anchor="end" tone={3} size={9.5}>{`total ${(bytes / 1e9).toFixed(1)} GB`}</Label>
          <rect x={16} y={296} width={520 * (1 - kvShare)} height={18} rx={2} fill="var(--compute)" className="v-cell" />
          <Label x={22} y={309} size={9} tone={1} bold>{`weights ${(d.weightBytes / 1e9).toFixed(1)} GB`}</Label>
          <rect x={16 + 520 * (1 - kvShare)} y={296} width={520 * kvShare} height={18} rx={2} fill="var(--memory)" className="v-cell" />
          <Label x={22 + 520 * (1 - kvShare)} y={309} size={9} tone={1} bold>{`KV ${(d.kvBytes / 1e9).toFixed(1)} GB`}</Label>
          <Label x={16} y={334} tone={3} size={9.5}>
            {kvShare > 0.5
              ? `KV is now ${(kvShare * 100).toFixed(0)}% of the traffic. This is a KV problem, not a weights problem.`
              : `Weights still dominate at this context. Reduce them: quantize (Module 11).`}
          </Label>
        </Fade>

        {/* the SOL reading */}
        <Fade show={s.step >= 4}>
          <Dim x1={160} x2={340} y={356} label={`MBU ${(mbu * 100).toFixed(0)}%`} kind={mbu > 0.8 ? 'token' : 'error'} />
          <Label x={360} y={360} tone={3} size={9.5} mono={false}>
            {mbu > 0.8 ? 'at the floor — no kernel work left here' : 'below 80% — Nsight Compute will name the stall'}
          </Label>
        </Fade>

        {/* what to fix */}
        <Fade show={s.step >= 6}>
          <Arrow x1={30} y1={382} x2={70} y2={382} kind="compute" />
          <Label x={78} y={386} size={9.5} kind="compute" bold>prefill</Label>
          <Label x={126} y={386} tone={3} size={9} mono={false}>bigger GEMMs, fuse the elementwise ops</Label>
          <Arrow x1={330} y1={382} x2={370} y2={382} kind="memory" />
          <Label x={378} y={386} size={9.5} kind="memory" bold>decode</Label>
          <Label x={426} y={386} tone={3} size={9} mono={false}>fewer bytes, bigger batch — never a better GEMM</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
