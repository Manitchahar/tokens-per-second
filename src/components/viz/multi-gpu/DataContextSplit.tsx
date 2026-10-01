import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';
import { bmodelById, gpuById, linkById, contextParallelStep, cpPenalty, kvBytesPerToken } from './parallel';

const STEPS = [
  'Replicas: no communication at all',
  'Data parallelism needs a sync',
  'Context parallelism: split the sequence',
  'The ring exchange',
  'What it actually saves',
  'When it does not pay',
  'The penalty question',
  'The four-way decision',
];

const CAPTIONS = [
  'Replica parallelism: three complete copies of the model, three different requests, zero communication. Each copy is a full GPU or a full model-parallel group. This is the cheapest throughput there is.',
  'True data parallelism — different requests on each GPU — is nearly free at inference because the weights are read-only. It only costs a sync when the requests must share state, such as one request spread across GPUs.',
  'Context parallelism attacks the other half of memory. One request with a 128k context needs 16 GB of KV on Llama-3.1-8B. Split the sequence across four GPUs and each holds 4 GB.',
  'But attention is global: token 40,000 must attend to tokens 0 and 20,000, which live on other GPUs. So the K and V have to rotate around the ring — P-1 rotations of a 1/P slice.',
  'Per GPU the memory falls by exactly P. So does the local attention read. What does not fall is the communication: nearly the whole cache crosses the wire, every layer.',
  'Over PCIe that all-to-all of KV is catastrophic. Over NVLink it is survivable at moderate context and ruinous at long context. The price scales with cache size, not with model size.',
  'The break-even test is one comparison: is the aggregate link faster than the memory bandwidth you gave up? The KV cache cancels out of the ratio, leaving (P-1)·rho — above 1 for every P > 1. Over NVLink at P=4 that is 26x; over PCIe, 393x.',
  'Four tools for four problems: replicas for throughput, quantization for memory, tensor parallelism to fit, pipeline parallelism to cross nodes, context parallelism only when one sequence alone will not fit.',
];

const MODELS = [
  { value: 'llama8b', label: 'L3.1-8B' },
  { value: 'qwen15b', label: 'Q2.5-1.5B' },
  { value: 'llama70b', label: 'L3.1-70B' },
];
const LINKS = [
  { value: 'nvlink4', label: 'NVLink 4' },
  { value: 'pcie4', label: 'PCIe 4.0' },
  { value: 'ib', label: 'InfiniBand' },
];

const SLICES = 4;
const SW = 116;
const SGAP = 20;
const SX0 = 30;

export default function DataContextSplit() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [mode, setMode] = useState<'replica' | 'data' | 'context'>('replica');
  const [modelId, setModelId] = useState('llama8b');
  const [linkId, setLinkId] = useState('nvlink4');
  const [P, setP] = useState(4);
  const [ctx, setCtx] = useState(32768);
  const [batch, setBatch] = useState(1);

  const model = bmodelById(modelId);
  const gpu = gpuById('h100');
  const link = linkById(linkId);

  const kvTok = kvBytesPerToken(model);
  const kvTotal = batch * ctx * kvTok;
  const cp = contextParallelStep({ model, gpu, P, link, batch, ctx });
  const slow = useTween(cp.slowdown, 500);
  const penalty = cpPenalty(gpu, link, P);

  // weights fit?
  const wBytes = model.params * 2;

  return (
    <Stage
      rootRef={ref}
      title="Replicas, data parallelism and context parallelism"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'the GPU’s own work'],
        ['memory', 'KV it holds'],
        ['comm', 'what crosses the wire'],
        ['token', 'finished requests'],
        ['error', 'wasted'],
        'text',
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Mode', value: mode === 'replica' ? 'Replicas' : mode === 'data' ? 'Data parallel' : `Context parallel (P=${P})`, kind: 'text', note: `${model.name}, ${fmt.int(ctx)} ctx` },
          { label: 'KV total', value: fmt.gb(kvTotal), kind: 'memory', note: `${fmt.bytes(kvTok)} per token` },
          { label: 'KV per GPU', value: fmt.gb(kvTotal / (mode === 'context' ? P : 1)), kind: 'memory', note: mode === 'context' ? `divided by ${P}` : 'unchanged' },
          { label: 'Comm / forward', value: mode === 'context' ? fmt.gb(cp.commBytes * model.layers) : '0 B', kind: mode === 'context' ? 'comm' : 'token', note: mode === 'context' ? `slowdown ${slow.toFixed(2)}x` : 'no collectives' },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Strategy"
            value={mode}
            options={[{ value: 'replica', label: 'Replicas' }, { value: 'data', label: 'Data parallel' }, { value: 'context', label: 'Context parallel' }]}
            onChange={setMode}
          />
          <Segmented label="Model" value={modelId} options={MODELS} onChange={setModelId} />
          <Segmented label="Interconnect" value={linkId} options={LINKS} onChange={setLinkId} />
          <Slider label="GPUs in the group" kind="compute" value={P} min={2} max={8} onChange={setP} format={(v) => `${v}`} />
          <Slider label="Context length (tokens)" kind="memory" value={ctx} min={1024} max={131072} log onChange={setCtx} format={(v) => `${fmt.int(v)}`} />
          <Slider label="Batch" kind="token" value={batch} min={1} max={64} log onChange={setBatch} format={(v) => `${fmt.int(v)}`} />
        </>
      }
    >
      <svg viewBox="0 0 640 384" role="img" aria-label="Four GPUs side by side. In replica mode each holds the full model and serves a different request with no communication. In data parallel mode each still holds the full model. In context parallel mode the sequence is split into four slices of KV, each GPU holds one slice, and violet arrows show K and V rotating around the ring between GPUs.">
        {/* ── mode: replicas / data ── */}
        <Fade show={mode !== 'context'}>
          <Label x={SX0} y={22} tone={2} size={11} bold>{mode === 'replica' ? `${SLICES} replicas, ${SLICES} × ${model.name}` : `${SLICES} GPUs, ${SLICES} × ${model.name}`}</Label>
          <Label x={600} y={22} anchor="end" tone={3} size={9.5}>{`${fmt.gb(wBytes)} of weights each`}</Label>

          {Array.from({ length: SLICES }, (_, i) => {
            const x = SX0 + i * (SW + SGAP);
            const hasReq = mode === 'replica' || true;
            return (
              <g key={i}>
                <rect x={x} y={34} width={SW} height={86} rx={3} fill="var(--ground-2)" stroke="var(--rule-2)" />
                <Label x={x + SW / 2} y={48} anchor="middle" size={9.5} tone={3}>{`GPU ${i}`}</Label>
                <rect x={x + 10} y={54} width={SW - 20} height={20} rx={2} fill="var(--memory)" opacity={0.8} />
                <Label x={x + SW / 2} y={68} anchor="middle" size={9}>{'full weights'}</Label>
                <rect x={x + 10} y={78} width={SW - 20} height={14} rx={2} fill="var(--compute)" opacity={0.8} />
                <Label x={x + SW / 2} y={88} anchor="middle" size={8.5}>{'full KV'}</Label>
                {hasReq && <Label x={x + SW / 2} y={108} anchor="middle" size={9} kind="token">{`req ${i + 1}`}</Label>}
              </g>
            );
          })}

          <Fade show={s.step <= 1}>
            <Label x={SX0} y={140} size={10} kind="token" bold>0 bytes cross between them.</Label>
            <Label x={SX0} y={158} size={9.5} tone={3}>{`Throughput scales linearly. One request’s latency does not improve at all.`}</Label>
          </Fade>

          <Fade show={s.step >= 1}>
            <Label x={SX0} y={140} size={10} kind="comm" bold>Weights are read-only, so no sync is needed — until</Label>
            <Label x={SX0} y={158} size={9.5} tone={3}>one request spans several GPUs: then it is not data parallelism,</Label>
            <Label x={SX0} y={172} size={9.5} tone={3}>it is context parallelism, and it needs the all-to-all below.</Label>
          </Fade>

          {/* cost line */}
          <Label x={SX0} y={210} tone={2} size={11} bold>What it buys, what it costs</Label>
          <Span x={SX0} y={220} w={190} h={16} kind="token" opacity={0.85} label="throughput ×4" labelTone="light" />
          <Span x={SX0 + 200} y={220} w={180} h={16} kind="compute" opacity={0.85} label="memory ×1 per GPU" labelTone="light" />
          <Label x={SX0} y={254} size={10} kind="token">{`4 × ${fmt.gb(wBytes)} = ${fmt.gb(wBytes * SLICES)} of total GPU memory`}</Label>
          <Label x={SX0} y={272} size={10} kind="error">{`but 4 × ${fmt.gb(wBytes)} of total cost — four copies of the same weights`}</Label>
        </Fade>

        {/* ── mode: context parallel ── */}
        <Fade show={mode === 'context'}>
          <Label x={SX0} y={22} tone={2} size={11} bold>{`One request, ${fmt.int(ctx)} tokens, split ${P} ways`}</Label>
          <Label x={600} y={22} anchor="end" tone={3} size={9.5}>{`${fmt.gb(kvTotal)} of KV total`}</Label>

          {Array.from({ length: P }, (_, i) => {
            const rowW = SLICES * SW + (SLICES - 1) * SGAP;
            const gap = 8;
            const w = (rowW - (P - 1) * gap) / P;
            const x = SX0 + i * (w + gap);
            return (
              <g key={i}>
                <rect x={x} y={34} width={w} height={78} rx={3} fill="var(--ground-2)" stroke="var(--rule-2)" />
                <Label x={x + w / 2} y={48} anchor="middle" size={w > 40 ? 9 : 7} tone={3}>{`GPU ${i}`}</Label>
                <rect x={x + 5} y={54} width={Math.max(0, w - 10)} height={22} rx={2} fill="var(--memory)" opacity={0.85} />
                <Label x={x + w / 2} y={68} anchor="middle" size={w > 70 ? 8 : 6}>{'full weights'}</Label>
                <rect x={x + 5} y={80} width={Math.max(0, w - 10)} height={16} rx={2} fill="var(--compute)" opacity={0.85} />
                <Label x={x + w / 2} y={91} anchor="middle" size={w > 60 ? 7.5 : 6}>{`KV ${i}`}</Label>
                <Label x={x + w / 2} y={107} anchor="middle" size={w > 60 ? 7.5 : 6} kind="memory">{`1/${P}`}</Label>
              </g>
            );
          })}

          {/* the sequence bar */}
          <Label x={SX0} y={132} tone={3} size={9.5}>the context</Label>
          <rect x={SX0} y={138} width={SLICES * SW + (SLICES - 1) * SGAP} height={16} rx={2} fill="none" stroke="var(--rule)" />
          {Array.from({ length: P }, (_, i) => {
            const totalW = SLICES * SW + (SLICES - 1) * SGAP;
            return (
              <rect key={i} x={SX0 + (totalW / P) * i} y={138} width={totalW / P - 2} height={16}
                fill="var(--memory)" opacity={0.85} stroke="var(--memory)" rx={1} />
            );
          })}
          <Label x={SX0 + (SLICES * SW + (SLICES - 1) * SGAP) / 2} y={170} anchor="middle" size={9} tone={3}>{`each GPU holds ${fmt.gb(kvTotal / P)} instead of ${fmt.gb(kvTotal)}`}</Label>

          {/* the ring */}
          {s.step >= 3 && (
            <g>
              {(() => {
                const rowW = SLICES * SW + (SLICES - 1) * SGAP;
                const gap = 8;
                const w = (rowW - (P - 1) * gap) / P;
                const cxs = Array.from({ length: P }, (_, i) => SX0 + i * (w + gap) + w / 2);
                return (
                  <>
                    {Array.from({ length: P - 1 }, (_, i) => (
                      <Arrow key={i} x1={cxs[i] + 4} y1={188} x2={cxs[i + 1] - 6} y2={188} kind="comm" curve={9} width={1.2} />
                    ))}
                    <Arrow x1={cxs[P - 1] + 4} y1={196} x2={cxs[0] - 6} y2={196} kind="comm" curve={-9} width={1.2} />
                  </>
                );
              })()}
              <Label x={SX0} y={210} size={9.5} kind="comm">{`${P - 1} rotations per layer — ${fmt.gb(cp.commBytes)} per layer, ${fmt.gb(cp.commBytes * model.layers)} per forward pass`}</Label>
            </g>
          )}

          {/* the verdict */}
          <Label x={SX0} y={240} tone={2} size={11} bold>Over {link.name}</Label>
          <Label x={SX0} y={258} size={9.5} tone={3}>{`local read ${(cp.tLocal * 1e6).toFixed(0)} µs  +  comm ${(cp.tComm * 1e6).toFixed(0)} µs`}</Label>
          <Span x={SX0} y={266} w={Math.max(2, Math.min(400, slow * 100))} h={14} kind={slow > 1.5 ? 'error' : 'comm'} opacity={0.9} />
          <Label x={SX0 + Math.max(2, Math.min(400, slow * 100)) + 8} y={277} size={10} kind={slow > 1.5 ? 'error' : 'token'}>{`${slow.toFixed(2)}x slower than one GPU`}</Label>
          <Fade show={s.step >= 6}>
            <Label x={SX0} y={298} size={10} kind="error" bold>{`penalty = (P-1)·rho = ${penalty.toFixed(1)}x`}</Label>
            <Label x={SX0} y={312} size={9} tone={3}>{`independent of model and context: the KV cache cancels out of the ratio.`}</Label>
          </Fade>
          <Label x={SX0} y={330} size={9} tone={3}>{`one GPU would need ${fmt.gb(kvTotal)}; on ${P} it needs ${fmt.gb(kvTotal / P)}`}</Label>
        </Fade>

        {/* ── the decision table ── */}
        <Fade show={s.step >= 7}>
          <Label x={SX0} y={348} tone={2} size={11} bold>Four tools, four problems</Label>
          {(['replica → throughput', 'quantize → memory', 'TP → fit in a node', 'PP → cross nodes'] as const).map((label, i) => (
            <Label key={label} x={SX0 + (i % 2) * 290} y={366 + Math.floor(i / 2) * 15} size={9} kind={i === 0 ? 'token' : i === 1 ? 'memory' : i === 2 ? 'compute' : 'comm'}>{label}</Label>
          ))}
        </Fade>
      </svg>
    </Stage>
  );
}