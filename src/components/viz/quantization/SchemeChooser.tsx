import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, Chip, fmt } from '../lib/svg';
import { llama8b, h100, decodeStepQuant, weightBytes, metaFracInt, SCHEMES } from './quant';

const WORKLOADS = [
  { id: 'fit', label: 'Fit on one GPU' },
  { id: 'throughput', label: 'Max throughput' },
  { id: 'latency', label: 'Latency per user' },
  { id: 'longctx', label: 'Long context' },
] as const;
type Workload = (typeof WORKLOADS)[number]['id'];

const STEPS = [
  'What the letters mean',
  'W4A16: weights only',
  'W8A8: both sides',
  'FP8: a float, not an integer',
  'The KV cache has its own budget',
  'Choosing by workload, not by fashion',
];
const CAPTIONS = [
  'A scheme is two numbers. W is the bit width of the weights, A the bit width of the activations. A8 only pays off when there is a real integer tensor-core GEMM to run it on.',
  'W4A16 shrinks only the weights and leaves activations in BF16. It is the standard 4-bit recipe: a quarter of the weight memory, and a proportionally shorter decode step because that step is dominated by reading weights.',
  'W8A8 quantizes the activations too, and that is what actually reaches the INT8 tensor cores. It is the fastest safe win on hardware with good INT8 support, but activations have outliers and need per-token dynamic scales.',
  'FP8 is a floating-point format with its own exponent, so it needs no group scale and copes with outliers the way BF16 does. On GPUs with FP8 tensor cores (Ada Lovelace, Hopper, Blackwell) it runs at full tensor-core rate, which makes it the high-quality option rather than the cheap one.',
  'The KV cache is a separate decision with a separate budget. Halving it to 8 bits halves the per-step read and often doubles concurrency, and it costs less quality than shrinking the weights further.',
  'There is no best scheme — only the one matching your bottleneck. Watch the recommendation move as you change batch, context, and what you say you are optimizing for.',
];

/** Which scheme each step spotlight. -1 means "no single highlight". */
const FOCUS = [-1, 3, 2, 1, -1, -1];

const ROW_Y = 46;
const ROW_H = 27;
const BAR_X = 112;
const BW = 286;

export default function SchemeChooser() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [batch, setBatch] = useState(32);
  const [ctx, setCtx] = useState(4096);
  const [workload, setWorkload] = useState<Workload>('throughput');

  const rows = useMemo(
    () =>
      SCHEMES.map((sc) => {
        const meta = metaFracInt(sc.wBits, 128);
        const w = weightBytes(llama8b, sc.wBits, meta);
        const d = decodeStepQuant({ batch, ctx, wBits: sc.wBits, kvBits: sc.kvBits, metaFrac: meta });
        // share of the step that is pure memory wait, with overhead removed
        const memShare = d.t > 0 ? (d.t - d.overhead) / d.t : 1;
        return { ...sc, meta, w, t: d.t, throughput: d.total, memShare, memBound: d.bound };
      }),
    [batch, ctx],
  );

  /* How much each scheme suits the stated goal. The lesson explains the reasoning. */
  const fit = (r: (typeof rows)[number]) => {
    if (workload === 'fit') return r.w < 5e9 ? 1 : r.w < 9e9 ? 0.7 : 0.2;
    if (workload === 'throughput') return r.throughput > rows[0].throughput * 2 ? 1 : r.throughput > rows[0].throughput * 1.3 ? 0.75 : 0.5;
    if (workload === 'latency') return r.t < rows[0].t * 0.6 ? 1 : r.t < rows[0].t ? 0.7 : 0.5;
    return r.kvBits <= 4 ? 1 : r.kvBits <= 8 ? 0.85 : 0.3; // long context
  };

  const recommended = useMemo(() => {
    const scored = rows.map((r) => ({ r, v: r.throughput * (0.3 + 0.7 * fit(r)) }));
    return scored.sort((a, b) => b.v - a.v)[0].r;
  }, [rows, workload]);

  const baseT = rows[0].t;
  const liveIdx = st === 5 ? -2 : FOCUS[st];

  return (
    <Stage
      rootRef={ref}
      title="Choosing a scheme: W4A16, W8A8, FP8, and the KV budget"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', 'token', 'error', 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Recommended', value: recommended.label, kind: 'token', note: `for ${WORKLOADS.find((w) => w.id === workload)!.label.toLowerCase()}` },
          { label: 'Weights on GPU', value: fmt.gb(recommended.w), kind: 'memory', note: `${((recommended.w / llama8b.params) * 8).toFixed(2)} bits/weight incl. scales` },
          { label: 'One decode step', value: `${(recommended.t * 1e3).toFixed(2)} ms`, kind: 'compute', note: `${baseT / recommended.t > 1.15 ? (baseT / recommended.t).toFixed(2) + '× faster than BF16' : 'about the same as BF16'}` },
          { label: 'Throughput', value: `${recommended.throughput.toFixed(0)} tok/s`, kind: 'token', note: `${recommended.memBound === 'memory' ? 'memory-bound: bytes are the limit' : 'compute-bound: math is the limit'}` },
        ]} />
      }
      controls={
        <>
          <Slider label="Concurrent users (batch)" value={batch} min={1} max={256} log onChange={setBatch} format={(v) => `${fmt.int(v)} users`} kind="token" />
          <Slider label="Context length per user" value={ctx} min={512} max={32768} log onChange={setCtx} format={(v) => `${fmt.int(v)} tokens`} kind="memory" />
          {st >= 5 && <Segmented label="What are you optimizing for?" value={workload} options={WORKLOADS.map((w) => ({ value: w.id, label: w.label }))} onChange={setWorkload} />}
        </>
      }
    >
      <svg viewBox="0 0 640 384" role="img" aria-label="Six quantization schemes for Llama-3.1-8B compared on weight memory and decode step time as paired bars, plus a second panel showing what share of each decode step is spent waiting on memory, and a recommendation that responds to batch size, context length and workload goal.">
        <Label x={0} y={22} tone={2} size={11} bold>Same 8B model, same GPU, six schemes</Label>
        <Label x={640} y={22} anchor="end" tone={3} size={9.5}>{`Llama-3.1-8B · H100 · ${batch} users · ${fmt.int(ctx)} tokens of context each`}</Label>

        {/* ── paired bars: weight memory above, decode step below ── */}
        {rows.map((r, i) => {
          const y = ROW_Y + i * ROW_H;
          const live = liveIdx === -2 || i === liveIdx || r.id === recommended.id;
          const spotlight = liveIdx === -2 ? r.id === recommended.id : i === liveIdx;
          return (
            <g key={r.id}>
              <Label x={104} y={y + 9} anchor="end" tone={live ? 1 : 3} size={10.5} bold={spotlight}>{r.label}</Label>
              <Label x={104} y={y + 20} anchor="end" tone={3} size={8.5}>{`W${r.wBits} · KV${r.kvBits}`}</Label>
              <rect x={BAR_X} y={y} width={BW} height={9} rx={2} fill="none" stroke="var(--rule)" />
              <Span x={BAR_X} y={y} w={(r.w / 20e9) * BW} h={9} kind="memory" opacity={live ? 0.95 : 0.45} />
              <rect x={BAR_X} y={y + 12} width={BW} height={9} rx={2} fill="none" stroke="var(--rule)" />
              <Span x={BAR_X} y={y + 12} w={Math.min(1, r.t / baseT) * BW} h={9} kind="compute" opacity={live ? 0.95 : 0.45} />
              <Label x={BAR_X + BW + 6} y={y + 8} tone={live ? 1 : 3} size={9}>{`${(r.w / 1e9).toFixed(1)} GB`}</Label>
              <Label x={BAR_X + BW + 6} y={y + 20} tone={live ? 1 : 3} size={9}>{`${(r.t * 1e3).toFixed(2)} ms`}</Label>
              {st === 5 && r.id === recommended.id && <Chip x={586} y={y + 3} text="pick" kind="token" size={9.5} h={16} />}
            </g>
          );
        })}
        <Fade show={st >= 1}>
          <Label x={112} y={228} tone={3} size={9.5}>upper bar: weight memory (20 GB scale) · lower bar: decode step (BF16 = full width)</Label>
        </Fade>

        {/* ── what each scheme is actually bound by ── */}
        <Fade show={st >= 1}>
          <g>
            <Label x={0} y={256} tone={2} size={11} bold>What each scheme is waiting on</Label>
            {rows.map((r, i) => {
              const y = 268 + i * 17;
              return (
                <g key={r.id}>
                  <Label x={104} y={y + 9} anchor="end" tone={3} size={9}>{r.label}</Label>
                  <rect x={BAR_X} y={y} width={BW} height={12} rx={2} fill="none" stroke="var(--rule)" />
                  <Span x={BAR_X} y={y} w={Math.max(4, BW * r.memShare)} h={12} kind="memory" opacity={0.8} label={r.memBound === 'memory' ? 'waiting on bytes' : 'doing math'} />
                </g>
              );
            })}
            <Label x={112} y={268 + rows.length * 17 + 12} tone={3} size={9}>blue = share of the step spent moving bytes. Decode stays memory-bound across almost this whole range, which is why bytes are what you buy.</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}