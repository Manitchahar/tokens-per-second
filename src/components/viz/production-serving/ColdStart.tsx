import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Dim, fmt } from '../lib/svg';
import { GPUS, MODELS, gpuById, modelById } from '../lib/perfModel';

/**
 * The startup path of a replica, drawn to scale against a readiness budget.
 * Each step of the timeline is a real phase, sized by a model of what actually
 * costs time: process launch, weight bytes over disk/PCIe/HBM, graph capture,
 * and the warm-up passes needed to reach steady-state decode speed.
 */

type Source = 'disk' | 'pcie' | 'host' | 'none';

interface Phase {
  id: string;
  name: string;
  /** what makes it slow */
  cause: string;
  /** bytes that must move, if any */
  bytes: number;
  /** where those bytes come from */
  source: Source;
  /** fixed cost: interpreter start, imports, allocator, config (seconds) */
  fixed: number;
  /** cost that depends on batch: graph capture, shape-specialized compile (seconds) */
  perBatch: number;
  /** cost that depends on context length: caches, allocator growth (seconds) */
  perCtx: number;
  /** warm-up decode steps needed before this replica is honest */
  warmup: number;
}

// Speeds are deliberately rough and generous: real startup is I/O- and
// compile-bound, and exact figures depend on the filesystem, the node and the
// engine version. Treat them as order-of-magnitude.
const SRC: Record<Source, { name: string; bps: number; kind: 'comm' | 'memory' | 'compute' }> = {
  disk: { name: 'disk (cold cache)', bps: 1.2e9, kind: 'comm' }, // object-store / local NVMe, order of magnitude
  pcie: { name: 'PCIe (page cache miss)', bps: 24e9, kind: 'comm' },
  host: { name: 'host page cache', bps: 8e9, kind: 'memory' },
  none: { name: 'nothing moves', bps: 1, kind: 'text' },
};

const PHASES: Phase[] = [
  {
    id: 'process',
    name: 'Process launch & imports',
    cause: 'Interpreter start, then every library the engine imports. Python import time is the surprise here: it is not the model, it is the dependency tree.',
    bytes: 0, source: 'none',
    fixed: 18, perBatch: 0, perCtx: 0, warmup: 0,
  },
  {
    id: 'weights',
    name: 'Read & convert weights',
    cause: 'Pull the checkpoint bytes and turn them into the tensors the engine wants. This is the phase that scales with model size and disk speed.',
    bytes: 1, source: 'disk', // overwritten with the model's real weight bytes
    fixed: 4, perBatch: 0, perCtx: 0, warmup: 0,
  },
  {
    id: 'h2d',
    name: 'Copy weights to the GPU',
    cause: 'Move every weight tensor into HBM. Measured once per process; per replica, every start.',
    bytes: 1, source: 'pcie',
    fixed: 1, perBatch: 0, perCtx: 0, warmup: 0,
  },
  {
    id: 'capture',
    name: 'Capture CUDA graphs',
    cause: 'Record one decode iteration per batch shape and replay it as a single launch. Cost grows with the number of shapes captured, so capture only the shapes you serve.',
    bytes: 0, source: 'none',
    fixed: 2, perBatch: 0.35, perCtx: 0, warmup: 0,
  },
  {
    id: 'profiles',
    name: 'Select kernels / autotune',
    cause: 'Measure candidate kernels for your shapes and keep the winner. This is a one-time measurement cost that later pays for itself on every request.',
    bytes: 0, source: 'none',
    fixed: 6, perBatch: 0, perCtx: 0, warmup: 0,
  },
  {
    id: 'warmup',
    name: 'Warm-up requests',
    cause: 'Run synthetic traffic until decode reaches steady state: caches populated, allocator grown, no lazy paths left. Until it finishes, latency numbers are fiction.',
    bytes: 0, source: 'none',
    fixed: 0, perBatch: 0, perCtx: 0, warmup: 24,
  },
];

const STEPS = [
  'The replica exists',
  'Launch & imports',
  'Read the checkpoint',
  'Copy weights into HBM',
  'Capture CUDA graphs',
  'Autotune the kernels',
  'Warm-up is still not enough',
  'Everything against the budget',
];

const CAPTIONS = [
  'A replica is created. The load balancer knows nothing about it yet, and it cannot answer a request for some time. That gap is the cold start.',
  'Interpreter startup plus the import tree. It has nothing to do with model size, which is exactly why it is surprising on a small model with a huge dependency list.',
  'The checkpoint bytes arrive. Sized as weights ÷ disk throughput — the single phase that grows with the model.',
  'Every weight tensor is copied into HBM. Now the bytes move over PCIe rather than disk, which is why a warm page cache only helps the previous phase.',
  'One decode iteration per batch shape is recorded as a graph. Each captured shape costs real time, so this scales with how many shapes you allow.',
  'Candidate kernels are measured for your shapes. Cheap the first time, but you pay it on every single replica start.',
  'Synthetic traffic runs until the replica is honest. A replica that skipped this is fast for the first request and slow for the second hundred.',
  'The whole path against your readiness budget. Anything over the line is not a slow start, it is an outage every time you scale.',
];

const W = 640;

// A steady-state decode step for this model on this GPU, for the warm-up math.
function stepSeconds(modelId: string, gpuId: string, batch: number, ctx: number) {
  const model = modelById(modelId);
  const gpu = gpuById(gpuId);
  const weights = model.params * 2;
  const kv = batch * ctx * (2 * model.layers * model.kvHeads * model.headDim * 2);
  const tMem = (weights + kv) / (gpu.bw * 0.8);
  const tCompute = (2 * model.params * batch) / (gpu.flops * 0.55);
  return Math.max(tMem, tCompute) + 0.0015;
}

export default function ColdStart() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [model, setModel] = useState('llama8b');
  const [gpu, setGpu] = useState('h100');
  const [batch, setBatch] = useState(32);
  const [ctx, setCtx] = useState(2048);
  const [cached, setCached] = useState(false);
  const [budget, setBudget] = useState(300);

  const m = modelById(model);
  const g = gpuById(gpu);
  const weightBytes = m.params * 2;

  // A sensible number of captured shapes: one per power-of-two-ish bucket you
  // actually admit, not one per possible batch.
  const shapes = [1, 2, 4, 8, 16, 32, 64, 128].filter((b) => b <= batch * 2).length;

  const seconds = PHASES.map((p) => {
    if (p.id === 'weights') {
      // Cold: the bytes come off disk. Warm: the OS already has them in RAM.
      return { t: p.fixed + weightBytes / (cached ? SRC.host.bps : SRC.disk.bps), bytes: weightBytes, src: cached ? 'host' : 'disk' };
    }
    if (p.id === 'h2d') {
      // Weights may already be resident (mmap'ed straight into HBM) or must be copied.
      return { t: p.fixed + (weightBytes / SRC.pcie.bps), bytes: weightBytes, src: 'pcie' };
    }
    if (p.id === 'capture') return { t: p.fixed + shapes * p.perBatch, bytes: 0, src: 'none' };
    if (p.id === 'warmup') return { t: p.warmup * stepSeconds(model, gpu, batch, ctx), bytes: 0, src: 'none' };
    return { t: p.fixed, bytes: 0, src: 'none' };
  });

  const total = seconds.reduce((a, x) => a + x.t, 0);
  const over = total > budget;

  // timeline geometry
  const tx = 40;
  const tw = 560;
  const scaleMax = Math.max(total * 1.1, budget * 1.25);
  const px = (t: number) => (t / scaleMax) * tw;

  // stacked cumulative
  let acc = 0;
  const cum = seconds.map((x) => {
    const start = acc;
    acc += x.t;
    return { start, end: acc, t: x.t };
  });

  const ty = 128;
  const th = 34;
  const barY = ty;

  const visiblePhases = Math.min(s.step + 1, PHASES.length);

  return (
    <Stage
      rootRef={ref}
      title="The startup path, to scale, against a readiness budget"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['comm', 'bytes over a link'],
        ['memory', 'host / HBM work'],
        ['compute', 'capture & autotune'],
        ['idle', 'warm-up (not serving)'],
        ['error', 'over budget'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Weights', value: fmt.bytes(weightBytes), kind: 'memory', note: `${m.name} in BF16` },
          { label: 'Total startup', value: `${total.toFixed(1)} s`, kind: over ? 'error' : 'token', note: over ? `${(total / budget).toFixed(1)}× your budget` : 'inside budget' },
          { label: 'Replicas/min at start', value: fmt.int(Math.max(0, Math.floor((60 / total) * 1))), kind: 'token', note: 'one GPU each' },
          { label: 'Dominant phase', value: PHASES[cum.reduce((bi, c, i) => (c.t > seconds[bi].t ? i : bi), 0)].name, kind: 'comm' },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Model"
            value={model}
            options={MODELS.map((mm) => ({ value: mm.id, label: mm.name }))}
            onChange={setModel}
          />
          <Segmented
            label="GPU"
            value={gpu}
            options={GPUS.map((gg) => ({ value: gg.id, label: gg.name }))}
            onChange={setGpu}
          />
          <Slider label="Batch shapes served" value={shapes} min={1} max={8} onChange={(v) => setBatch(v * 8)} format={(v) => `${v} shapes`} kind="compute" />
          <Slider label="Warm context" value={ctx} min={512} max={32768} log onChange={setCtx} format={(v) => `${fmt.int(v)} tok`} kind="memory" />
          <Slider label="Readiness budget" value={budget} min={30} max={900} step={10} onChange={setBudget} format={(v) => `${v} s`} kind="token" />
          <Segmented
            label="Checkpoint bytes are"
            value={cached ? 'warm' : 'cold'}
            options={[{ value: 'cold', label: 'cold (first read)' }, { value: 'warm', label: 'in page cache' }]}
            onChange={(v) => setCached(v === 'warm')}
          />
        </>
      }
    >
      <svg viewBox={`0 0 ${W} 340`} role="img" aria-label="A stacked timeline of everything a serving replica does between existing and being ready: process launch, checkpoint read, copy into GPU memory, CUDA graph capture, kernel autotune and warm-up requests, drawn to scale against a readiness budget, with the phases that push it over marked in red.">
        {/* budget marker */}
        <Label x={tx} y={44} tone={2} size={11} mono={false} bold>{'What happens between "replica exists" and "replica answers requests"'}</Label>

        <Line y={70} x1={tx} x2={tx + tw} />
        <Label x={tx} y={84} tone={3} size={9.5}>{'0'}</Label>

        {/* budget line */}
        {budget < scaleMax && (
          <g>
            <line x1={tx + px(budget)} x2={tx + px(budget)} y1={60} y2={barY + th + 26} stroke="var(--error)" strokeWidth={1.4} strokeDasharray="4 4" />
            <Label x={tx + px(budget) - 6} y={56} anchor="end" tone={1} size={10} bold kind="error">{`readiness budget = ${budget} s`}</Label>
            <Label x={tx + px(budget) + 6} y={barY + th + 24} tone={1} size={9.5} kind="error">{`over budget`}</Label>
          </g>
        )}
        <Label x={tx + tw} y={84} anchor="end" tone={3} size={9.5}>{`${scaleMax.toFixed(0)} s`}</Label>

        {/* stacked bar */}
        {cum.map((c, i) => {
          const shown = i < visiblePhases;
          const p = PHASES[i];
          const kind = p.id === 'warmup' ? 'idle' : p.source === 'none' ? (p.id === 'profiles' ? 'compute' : 'memory') : SRC[p.source as Source].kind;
          const w = px(c.t);
          return (
            <g key={p.id} style={{ opacity: shown ? 1 : 0.18 }}>
              <Span
                x={tx + px(c.start)}
                y={barY}
                w={w}
                h={th}
                kind={i === visiblePhases - 1 && over ? 'error' : kind}
                label={w > 46 ? `${p.name.split(' ')[0]} ${c.t.toFixed(p.id === 'warmup' ? 0 : 1)}s` : undefined}
              />
            </g>
          );
        })}

        {/* per-phase rows */}
        {PHASES.map((p, i) => {
          const y = 214 + i * 19;
          const shown = i < visiblePhases;
          const kind = p.id === 'warmup' ? 'idle' : p.source === 'none' ? (p.id === 'profiles' ? 'compute' : 'memory') : SRC[p.source as Source].kind;
          return (
            <g key={p.id} style={{ opacity: shown ? 1 : 0.25 }}>
              <Label x={tx} y={y + 4} tone={shown ? 1 : 3} size={10} bold={shown && i === visiblePhases - 1}>{p.name}</Label>
              <Label x={tx + tw} y={y + 4} anchor="end" tone={2} size={10}>
                {`${seconds[i].t.toFixed(p.id === 'warmup' ? 1 : 1)} s`}
                {seconds[i].bytes > 0 && <tspan className="v-label" style={{ fill: 'var(--text-3)' }}>{`  (${fmt.bytes(seconds[i].bytes)} over ${SRC[seconds[i].src as Source].name})`}</tspan>}
              </Label>
              <line x1={tx} x2={tx + tw} y1={y + 9} y2={y + 9} className="v-rule faint" />
            </g>
          );
        })}

        <Dim x1={tx} x2={tx + px(total)} y={332} label={`total startup ${total.toFixed(1)} s`} kind={over ? 'error' : 'token'} />
      </svg>
    </Stage>
  );
}

function Line({ x1, x2, y }: { x1: number; x2: number; y: number }) {
  return <line x1={x1} x2={x2} y1={y} y2={y} className="v-rule" />;
}