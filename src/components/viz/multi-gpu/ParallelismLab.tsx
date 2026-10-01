import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';
import {
  BMODELS, bmodelById, gpuById, linkById,
  kvBytesPerToken, weightBytes, tpDegrees,
  tpDecodeStep, tpPrefillTime, bubbleFraction, microBatchesFor,
  planFor, ringFactor,
} from './parallel';

const STEPS = [
  'Set the model and the card',
  'Does it fit?',
  'If not, does it fit quantized?',
  'Then how many GPUs',
  'Inside the node: tensor',
  'Across nodes: pipeline',
  'Fill the pipeline',
  'Everything left: replicas',
];

const CAPTIONS = [
  'Pick the model and the GPU. Everything downstream is arithmetic from these two, plus the batch and context you actually serve.',
  'First question, and it decides most cases: do the weights plus the KV cache fit in memory with room to spare? If they do, you want replicas, not sharding.',
  'If not, does it fit at FP8 or 4 bits? Quantization costs no communication at all, so it is almost always cheaper than buying a second GPU.',
  'Still too big. Now count GPUs — but inside the island first, because tensor parallelism cannot leave a fast fabric.',
  'Tensor parallelism divides weights and KV by t, at the price of 2 collectives per layer. Inside an NVLink node that is 0.1-2% overhead, so it is effectively free.',
  'Pipeline parallelism moves one activation per boundary instead of two collectives per layer — 40x less data — which is why it can cross InfiniBand where TP cannot.',
  'Pipelining is not free: the bubble is (P-1)/(M+P-1). With 16 concurrent requests you cannot fill a 32-stage pipeline, and that is the whole trade.',
  'Whatever GPUs are left over become replicas. Zero communication, linear throughput, and the actual way serving systems scale.',
];

const GPUS = [
  { value: 'h100', label: 'H100 80GB' },
  { value: 'h200', label: 'H200 141GB' },
  { value: 'a100', label: 'A100 80GB' },
];
const LINKS = [
  { value: 'nvlink4', label: 'NVLink 4' },
  { value: 'ib', label: 'InfiniBand' },
  { value: 'pcie4', label: 'PCIe 4.0' },
];

export default function ParallelismLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama70b');
  const [gpuId, setGpuId] = useState('h100');
  const [linkId, setLinkId] = useState('nvlink4');
  const [gpus, setGpus] = useState(8);
  const [batch, setBatch] = useState(16);
  const [ctx, setCtx] = useState(8192);
  const [wBytes, setWBytes] = useState(2);
  const [island, setIsland] = useState(8);

  const model = bmodelById(modelId);
  const gpu = gpuById(gpuId);
  const link = linkById(linkId);

  const kvTok = kvBytesPerToken(model);
  const kv = batch * ctx * kvTok;
  const weights = weightBytes(model, wBytes);
  const overhead = 0.04 * gpu.mem;
  const budget = gpu.mem * 0.92 - overhead;

  const plan = planFor({ model, gpu, link, gpus, batch, ctx, wBytes, island });
  const degs = tpDegrees(model, 8);
  const dec = tpDecodeStep({ model, gpu, P: Math.max(1, plan.tp), link, batch });
  const pre = tpPrefillTime({ model, gpu, P: Math.max(1, plan.tp), link, tokens: batch });
  const M = microBatchesFor(plan.pp || 1, 0.2);
  const bubble = bubbleFraction(plan.pp || 1, M);

  const memT = useTween(Math.min(1, weights / budget), 400);
  const commT = useTween(plan.tp > 1 ? dec.tComm / dec.t : 0, 400);

  // memory bar
  const seg = [
    { label: 'weights', v: weights, k: 'compute' as const },
    { label: 'KV cache', v: kv, k: 'memory' as const },
    { label: 'overhead', v: overhead, k: 'idle' as const },
  ].filter((x) => x.v > 0);
  const used = seg.reduce((a, x) => a + x.v, 0);

  return (
    <Stage
      rootRef={ref}
      title="Lab: choose a parallelism strategy"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'weights'],
        ['memory', 'KV cache'],
        ['comm', 'collectives'],
        ['token', 'fits'],
        ['error', 'does not fit'],
        'text',
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Verdict', value: plan.reason.length > 44 ? plan.reason.slice(0, 44) + '…' : plan.reason, kind: plan.fitsAlone || plan.fitsQuantized ? 'token' : 'comm', note: plan.fitsAlone ? 'replicate' : plan.fitsQuantized ? 'quantize + replicate' : 'shard' },
          { label: 'Plan', value: `TP=${plan.tp} · PP=${plan.pp} · ×${plan.replicas}`, kind: 'text', note: `on ${gpus} GPUs` },
          { label: 'Comm overhead', value: `${(commT * 100).toFixed(1)}%`, kind: commT > 0.2 ? 'error' : 'token', note: plan.tp > 1 ? 'of a decode step' : 'no collectives' },
          { label: 'Pipeline bubble', value: `${(bubble * 100).toFixed(0)}%`, kind: bubble > 0.2 ? 'error' : 'token', note: `M=${M} micro-batches` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={BMODELS.map((m) => ({ value: m.id, label: m.name.replace('Llama-3.1-', 'L3.1-').replace('Qwen2.5-', 'Q2.5-') }))} onChange={setModelId} />
          <Segmented label="GPU" value={gpuId} options={GPUS} onChange={setGpuId} />
          <Segmented label="Interconnect" value={linkId} options={LINKS} onChange={setLinkId} />
          <Segmented label="Weight precision" value={wBytes} options={[{ value: 2, label: 'BF16' }, { value: 1, label: 'FP8' }, { value: 0.5, label: 'INT4' }]} onChange={(v) => setWBytes(v)} />
          <Slider label="GPUs available" kind="compute" value={gpus} min={1} max={64} onChange={setGpus} format={(v) => `${v}`} />
          <Slider label="Island size (fast fabric)" kind="comm" value={island} min={2} max={8} onChange={setIsland} format={(v) => `${v} GPUs`} />
          <Slider label="Batch" kind="token" value={batch} min={1} max={256} log onChange={setBatch} format={(v) => `${fmt.int(v)}`} />
          <Slider label="Context (tokens)" kind="memory" value={ctx} min={1024} max={131072} log onChange={setCtx} format={(v) => `${fmt.int(v)}`} />
        </>
      }
    >
      <svg viewBox="0 0 640 384" role="img" aria-label="A decision panel for multi-GPU serving. A stacked bar shows weights, KV cache and overhead against GPU memory capacity, in red if it does not fit. Below, four strategy rows report what each would do: replicate, quantize, tensor-parallel and pipeline-parallel, with the recommended plan and its communication overhead.">
        {/* ── the memory bar ── */}
        <Label x={0} y={22} tone={2} size={11} bold>{`${model.name} on ${gpu.name}`}</Label>
        <Label x={604} y={22} anchor="end" tone={3} size={9.5}>{`${fmt.int(batch)} × ${fmt.int(ctx)} ctx`}</Label>

        <rect x={0} y={30} width={604} height={26} rx={2} fill="none" stroke="var(--rule)" />
        {(() => {
          const total = Math.max(budget, used) * 1.02;
          const W = 604;
          let x = 0;
          const over = used > budget;
          return (
            <>
              {seg.map((g) => {
                const w = (g.v / total) * W;
                const el = (
                  <g key={g.label}>
                    <rect x={x} y={30} width={Math.max(0, w)} height={26} fill={g.k === 'idle' ? 'url(#tps-hatch)' : `var(--${g.k})`} opacity={0.9} stroke={`var(--${g.k})`} strokeWidth={0.8} />
                    {w > 44 && <text x={x + w / 2} y={47} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fontWeight: 600, fill: g.k === 'idle' ? 'var(--text-2)' : '#0e1116' }}>{g.label}</text>}
                  </g>
                );
                x += w;
                return el;
              })}
              {over && (
                <>
                  <rect x={(budget / total) * W} y={30} width={Math.max(0, W - (budget / total) * W)} height={26} fill="url(#tps-hatch-red)" opacity={0.45} stroke="var(--error)" strokeWidth={1} strokeDasharray="3 3" />
                  <Label x={(budget / total) * W - 6} y={47} anchor="end" size={9.5} kind="error">{`${fmt.gb(budget)} limit`}</Label>
                </>
              )}
            </>
          );
        })()}
        <Label x={0} y={72} size={9.5} kind={used > budget ? 'error' : 'token'}>
          {used > budget
            ? `${fmt.gb(used)} needed vs ${fmt.gb(budget)} available — does not fit on one GPU`
            : `${fmt.gb(used)} needed, ${fmt.gb(budget)} available — fits, with ${fmt.gb(budget - used)} spare`}
        </Label>
        <Label x={604} y={72} anchor="end" size={9} tone={3}>{`${fmt.bytes(kvTok)} KV/token`}</Label>

        {/* ── the four strategies ── */}
        <Label x={0} y={98} tone={2} size={11} bold>What each strategy would do</Label>

        {(() => {
          const rows = [
            {
              name: 'Replicate',
              sub: `${gpus} × ${model.name}`,
              ok: plan.fitsAlone,
              note: plan.fitsAlone ? 'fits: zero communication, linear throughput' : `needs ${fmt.gb(used)} per GPU`,
              k: 'token' as const,
            },
            {
              name: 'Quantize',
              sub: `FP8: ${fmt.gb(weightBytes(model, 1))} weights`,
              ok: plan.fitsQuantized,
              note: plan.fitsQuantized ? 'fits at FP8 — no communication cost' : wBytes === 2 ? 'FP8 alone is not enough here' : 'already quantized',
              k: 'memory' as const,
            },
            {
              name: 'Tensor parallel',
              sub: `t=${plan.tp} of ${degs.join('/')} legal`,
              ok: plan.tp > 1,
              note: plan.tp > 1 ? `${((dec.tComm / dec.t) * 100).toFixed(1)}% comm · ${fmt.gb(weights / plan.tp + kv / plan.tp)} per GPU` : 'not needed, or the model cannot split this far',
              k: 'compute' as const,
            },
            {
              name: 'Pipeline parallel',
              sub: `P=${plan.pp} stages`,
              ok: plan.pp > 1,
              note: plan.pp > 1 ? `${(bubble * 100).toFixed(0)}% bubble at M=${M} · ${fmt.gb((plan.pp - 1) * batch * model.hidden * 2)} moved` : 'not needed inside the island',
              k: 'comm' as const,
            },
          ];
          return (
            <g>
              {rows.map((r, i) => {
                const y = 106 + i * 34;
                const on = i === (plan.fitsAlone ? 0 : plan.fitsQuantized ? 1 : plan.tp > 1 ? 2 : 3);
                return (
                  <g key={r.name}>
                    <rect x={0} y={y} width={604} height={30} rx={2}
                      fill={on ? 'var(--ground-3)' : 'var(--ground-2)'}
                      stroke={on ? `var(--${r.k})` : 'var(--rule)'} strokeWidth={on ? 1.4 : 0.8} />
                    <Label x={10} y={y + 19} size={10} kind={r.k} bold>{r.name}</Label>
                    <Label x={110} y={y + 19} size={9.5} tone={3}>{r.sub}</Label>
                    <Label x={594} y={y + 19} anchor="end" size={9} kind={r.ok ? 'token' : 'error'}>
                      {r.ok ? (on ? 'recommended' : 'works') : 'no'}
                    </Label>
                    <Label x={404} y={y + 19} anchor="end" size={8.5} tone={3}>{r.note}</Label>
                  </g>
                );
              })}
            </g>
          );
        })()}

        {/* ── the verdict ── */}
        <Fade show={s.step >= 7}>
          <Dim x1={0} x2={604} y={250} kind="token" label={`TP=${plan.tp} · PP=${plan.pp} · ${plan.replicas} replica${plan.replicas > 1 ? 's' : ''}`} />
        </Fade>

        <Label x={0} y={276} tone={2} size={11} bold>Decode step at this plan</Label>
        {(() => {
          const t1 = tpDecodeStep({ model, gpu, P: 1, link, batch });
          const tP = tpDecodeStep({ model, gpu, P: Math.max(1, plan.tp), link, batch });
          const rows = [
            { n: `1 GPU`, mem: t1.tMem, fl: t1.tCompute, cm: 0, tot: t1.t, k: 'text' as const },
            { n: `TP=${plan.tp} (÷${plan.tp || 1} stages)`, mem: tP.tMem, fl: tP.tCompute, cm: tP.tComm, tot: tP.t, k: 'comm' as const },
          ];
          const maxTot = Math.max(...rows.map((r) => r.tot));
          return (
            <g>
              {rows.map((r, i) => {
                const y = 286 + i * 30;
                const W = 300;
                const x = 84;
                const segs = [
                  { v: r.mem, k: 'memory' as const, n: '' },
                  { v: r.fl, k: 'compute' as const, n: '' },
                  { v: r.cm, k: 'comm' as const, n: '' },
                ];
                let cx = x;
                return (
                  <g key={r.n}>
                    <Label x={x - 10} y={y + 10} anchor="end" size={9} mono={false} tone={3}>{r.n}</Label>
                    <rect x={x} y={y} width={W} height={16} rx={2} fill="none" stroke="var(--rule)" />
                    {segs.map((sg, j) => {
                      const w = (sg.v / maxTot) * W;
                      const el = sg.v > 0 ? (
                        <rect key={j} x={cx} y={y} width={Math.max(0, w)} height={16} fill={`var(--${sg.k})`} opacity={0.85} />
                      ) : null;
                      cx += w;
                      return el;
                    })}
                    <Label x={x + W + 8} y={y + 11} size={9.5} tone={2}>{`${(r.tot * 1000).toFixed(1)} ms`}</Label>
                    <Label x={x + W + 8} y={y + 22} size={8.5} kind="comm">{r.cm > 0 ? `${((r.cm / r.tot) * 100).toFixed(0)}% comm` : 'no collectives'}</Label>
                  </g>
                );
              })}
            </g>
          );
        })()}

        <Label x={604} y={378} anchor="end" size={9} tone={3}>{`prefill at TP=${plan.tp}: ${((pre.overheadFrac) * 100).toFixed(0)}% comm`}</Label>
        <Label x={0} y={378} size={9} tone={3}>{`${plan.tp > 1 ? `${fmt.bytes(ringFactor(plan.tp) * batch * model.hidden * 2 * model.layers * 2)} of collectives per decode step` : 'no TP: nothing crosses the wire'}`}</Label>
      </svg>
    </Stage>
  );
}