import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider, type Kind } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Fade, fmt } from '../lib/svg';
import { modelById, gpuById, MODELS } from '../lib/perfModel';
import {
  LAUNCH_US,
  LAUNCH_US_REPLAY,
  cpuMs,
  gpuMs,
  kernelsPerStep,
  kernelBreakdown,
  stepMs,
  copyInBytes,
  SAMPLED_ID_BYTES,
  ms,
  us,
  short,
} from './model';

// Lesson 15.3 — recording a decode step once and replaying it forever.
// The top band is the win: the CPU's whole contribution collapses to one
// call. The middle band is the machinery: static buffers on both ends and a
// node chain in between, with the addresses baked in at capture time.
// The bottom band is the honest comparison — 556 ticks versus one.

const STEPS = [
  'Eager: 556 launches, one at a time',
  'Capture mode: record, do not execute',
  'One graph, addresses baked in',
  'Replay: a single API call',
  'What replay costs, and does not',
  'The shape rule: one graph per batch',
  'The pool rule: graphs need their own KV',
];

const CAPTIONS = [
  'The CPU issues every kernel of a decode step separately, one driver call after another. Each call is a few microseconds of Python and the dispatcher, and the total is the launch bill from the first lesson.',
  'Run the step once more inside a capture region and the launches stop executing and start being written down. Nothing is computed; a tape of kernels and arguments is built on the device.',
  'What you get is a single graph object: an ordered chain of kernel nodes with their pointers already resolved. That is why it is fast to replay and why it cannot be pointed at new addresses later.',
  'Replaying issues the whole chain with one API call. The CPU does the same work as launching a single tiny kernel, and the GPU still runs all 556 kernels in exactly the same order.',
  'Replay removes the CPU bill and leaves the GPU bill completely alone. If the GPU was already the longer pipeline, a graph buys you almost nothing — this is the same test as the last two lessons.',
  'Because the shapes are baked in, a graph captured at batch 64 is wrong at batch 33. You either capture a graph per batch size, or run the big graph on padding sequences and throw the extra results away.',
  'A graph writes to fixed addresses, so it cannot use a KV pool that is free to move blocks around. Engines keep a second, non-preemptible pool for the graphs, which is memory you cannot reclaim under pressure.',
];

type Out = 'logits' | 'sampled';

interface Vars {
  modelName: string;
  kernE: number;
  kernG: number;
  cpuE: number;
  cpuG: number;
  gpuT: number;
  stE: number;
  stG: number;
  speed: number;
  batch: number;
  bytesIn: string;
  bytesOutFull: string;
  bytesOutSampled: string;
  launchUs: number;
  replayUs: number;
}

const SCRIPT: ((v: Vars) => { head: string; lines: string[]; verdict: string; kind: Kind })[] = [
  () => ({
    head: 'What the CPU is asked to do, one kernel at a time',
    lines: [
      'Every operator in the step becomes its own driver call. Nothing is batched, and the CPU',
      'walks the list in Python. The step is finished when the list is finished being described.',
    ],
    verdict: 'This is the bill from Lesson 15.1, unchanged: N kernels × the cost of one launch.',
    kind: 'error',
  }),
  () => ({
    head: 'Capture mode: the launches are written down, not run',
    lines: [
      'Run the step once inside a capture region and the driver stops executing and starts',
      'recording. Each kernel becomes an entry in a tape: which function, which pointers,',
      'what shape. It runs no arithmetic at all, which is why capture is nearly free.',
    ],
    verdict: 'You get a recording. The model has not computed anything yet.',
    kind: 'text',
  }),
  () => ({
    head: 'One graph object, with the addresses already resolved',
    lines: [
      'The tape is a chain of kernel nodes, ordered, with every pointer turned into a device',
      'address at capture time. Replaying it means handing that chain to the GPU as one unit.',
      'This is also the constraint: the graph has no idea where your data is now.',
    ],
    verdict: 'Fast because the CPU is already done. Rigid because the addresses are fixed.',
    kind: 'compute',
  }),
  () => ({
    head: 'Replay: hundreds of kernels, one API call',
    lines: [
      'One call launches the whole chain. The GPU runs every kernel in the recorded order, with',
      'the same data dependencies, and the CPU is free again immediately. The work is not',
      'smaller — it is only described once.',
    ],
    verdict: 'The CPU cost of a step stops depending on how many kernels the model has.',
    kind: 'token',
  }),
  (v) => ({
    head: 'What replay removes, and what it cannot touch',
    lines: [
      `Replay takes the CPU side from ${us(v.cpuE * 1000)} to ${us(v.cpuG * 1000)}, whatever that is,`,
      `because there is now one call rather than ${fmt.int(v.kernE)}. The GPU side is ${ms(v.gpuT)} either`,
      'way: same kernels, same bytes, same arithmetic. A graph is a CPU optimisation only.',
    ],
    verdict: `${v.speed.toFixed(2)}× on this configuration — and the number to check is whether the CPU was the bottleneck.`,
    kind: 'memory',
  }),
  (v) => ({
    head: 'Shapes are baked in, so a graph is per-batch-size',
    lines: [
      'A graph captured at batch 64 assumes 64 rows in every buffer. Run it at batch 33 and it',
      'reads past the end of what you allocated. You have two honest options: capture a graph',
      'per bucket size, or run the big graph on padding rows and discard the extra results.',
    ],
    verdict: `Padding is why the CPU bill is a step function, not a smooth curve at batch ${fmt.int(v.batch)}.`,
    kind: 'comm',
  }),
  () => ({
    head: 'Graphs cannot use a movable KV pool',
    lines: [
      'The graph reads and writes fixed addresses. A KV pool that preemption is free to compact',
      'or relocate would hand the graph different memory than it captured against. So graph-',
      'captured memory has to sit in a pool nobody is allowed to move.',
    ],
    verdict: 'That second pool is memory your cache manager cannot reclaim under pressure.',
    kind: 'error',
  }),
];

const IN_X = 8;
const IN_W = 88;
const G_X = 116;
const G_W = 368;
const OUT_X = 504;
const OUT_W = 128;
const BOX_Y = 106;
const BOX_H = 32;
const MID_Y = BOX_Y + BOX_H / 2;

const COMB_X = 132;
const COMB_PITCH = 0.9;

const tickColor = (k: Kind) =>
  k === 'compute' ? 'var(--compute)'
    : k === 'memory' ? 'var(--memory)'
      : k === 'comm' ? 'var(--comm)'
        : 'var(--text-3)';

/** The kernel sequence as coloured kinds, so the comb shows what runs, not just how many. */
function combKinds(modelId: string, mode: 'eager' | 'compiled'): Kind[] {
  const b = kernelBreakdown(modelById(modelId), mode);
  const out: Kind[] = [];
  for (let l = 0; l < modelById(modelId).layers; l++) {
    for (let i = 0; i < b.gemm / modelById(modelId).layers; i++) out.push('compute');
    for (let i = 0; i < b.attn / modelById(modelId).layers; i++) out.push('comm');
    for (let i = 0; i < b.elementwise / modelById(modelId).layers; i++) out.push('memory');
  }
  for (let i = 0; i < b.head; i++) out.push('text');
  return out;
}

export default function GraphReplay() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [launchUs, setLaunchUs] = useState(LAUNCH_US);
  const [out, setOut] = useState<Out>('sampled');
  const [gpuId, setGpuId] = useState('h100');
  const [batch, setBatch] = useState(1);

  const model = modelById(modelId);
  const gpu = gpuById(gpuId);

  const cpuE = cpuMs(model, 'eager', launchUs);
  const cpuG = LAUNCH_US_REPLAY / 1000;
  const g = gpuMs({ model, gpu, batch, ctx: 1024 });
  const stE = stepMs({ model, gpu, batch, ctx: 1024 }, 'eager', launchUs);
  const stG = stepMs({ model, gpu, batch, ctx: 1024 }, 'graphed', launchUs);
  const speed = stE.t / stG.t;

  const bytesIn = copyInBytes(batch, model.hidden);
  const bytesOutFull = batch * model.vocab * 2;
  const bytesOutSampled = batch * SAMPLED_ID_BYTES;

  const v: Vars = {
    modelName: short(model),
    kernE: kernelsPerStep(model, 'eager'),
    kernG: kernelsPerStep(model, 'compiled'),
    cpuE,
    cpuG,
    gpuT: g.t,
    stE: stE.t,
    stG: stG.t,
    speed,
    batch,
    bytesIn: fmt.bytes(bytesIn),
    bytesOutFull: fmt.bytes(bytesOutFull),
    bytesOutSampled: fmt.bytes(bytesOutSampled),
    launchUs,
    replayUs: LAUNCH_US_REPLAY,
  };

  const script = SCRIPT[s.step](v);

  const kinds = useMemo(() => combKinds(modelId, 'eager'), [modelId]);
  const compiledKinds = useMemo(() => combKinds(modelId, 'compiled'), [modelId]);

  // shared scale for the two CPU bars, so the collapse is honest
  const scaleMax = Math.max(cpuE, cpuG);
  const bx = 170;
  const bw = 440;
  const barW = (t: number) => Math.max(4, (t / scaleMax) * bw);

  const captured = s.step >= 2;
  const recording = s.step === 1;
  const copyOut = out === 'sampled' ? v.bytesOutSampled : v.bytesOutFull;

  return (
    <Stage
      rootRef={ref}
      title="Capturing a decode step, and replaying it once per token"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'elementwise kernel'],
        ['compute', 'matrix multiply'],
        ['comm', 'attention'],
        ['token', 'the one replay launch'],
        ['error', 'what the graph cannot do'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'CPU: eager', value: us(cpuE * 1000), kind: 'error', note: `${fmt.int(v.kernE)} launches at ${launchUs} µs` },
          { label: 'CPU: replayed', value: us(cpuG * 1000), kind: 'token', note: 'one call, any number of kernels' },
          { label: 'GPU: unchanged', value: ms(g.t), kind: 'memory', note: 'a graph never moves fewer bytes' },
          { label: 'Step: eager → graph', value: `${ms(stE.t)} → ${ms(stG.t)}`, kind: speed > 1.2 ? 'token' : 'text', note: `${speed.toFixed(2)}×, and ${(cpuE > g.t ? 'the CPU was' : 'the GPU was')} the bottleneck` },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Model"
            value={modelId}
            options={MODELS.filter((m) => ['qwen05', 'llama3b', 'llama8b'].includes(m.id)).map((m) => ({ value: m.id, label: short(m) }))}
            onChange={setModelId}
          />
          <Segmented
            label="Graph output"
            value={out}
            options={[{ value: 'logits', label: 'full logits' }, { value: 'sampled', label: 'sampled id only' }]}
            onChange={setOut}
          />
          <Slider label="Concurrent users (batch)" value={batch} min={1} max={256} log onChange={setBatch} format={(x) => fmt.int(x)} kind="token" />
          <Slider label="CPU cost of one launch" value={launchUs} min={1} max={30} step={0.5} onChange={setLaunchUs} format={(x) => `${x} µs`} kind="idle" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A CUDA graph for a decode step: a bar comparing the CPU time to issue a step eagerly against the CPU time to replay a captured graph, a diagram of a static input buffer feeding a chain of kernel nodes that writes to a static output buffer, and a comb of 556 kernel launches compared against a single replay launch.">
        <Label x={8} y={18} tone={2} size={10.5} mono={false} bold>
          {`${v.modelName} · ${model.layers} layers · batch ${fmt.int(batch)} · ${gpu.name} · ${launchUs} µs per launch`}
        </Label>

        {/* ── band A: the CPU bill, before and after ─────────────────────── */}
        <Label x={8} y={34} tone={2} size={9.5} mono={false}>CPU time to issue one decode step</Label>
        <Label x={bx - 8} y={53} anchor="end" size={9.5} tone={3}>eager</Label>
        <rect x={bx} y={42} width={bw} height={15} rx={2} fill="none" stroke="var(--rule)" />
        <rect x={bx} y={42} width={barW(cpuE)} height={15} rx={2} fill="var(--error)" opacity={0.72} />
        <Label x={bx + barW(cpuE) + 7} y={53} size={9.5} kind="error" bold>{`${us(cpuE * 1000)}`}</Label>

        <Label x={bx - 8} y={72} anchor="end" size={9.5} tone={3}>graph</Label>
        <rect x={bx} y={61} width={bw} height={15} rx={2} fill="none" stroke="var(--rule)" />
        <rect x={bx} y={61} width={barW(cpuG)} height={15} rx={2} fill="var(--token)" opacity={0.85} />
        <Label x={bx + barW(cpuG) + 7} y={72} size={9.5} kind="token" bold>{`${us(cpuG * 1000)} — one call`}</Label>

        {/* ── band B: the machinery ──────────────────────────────────────── */}
        <Span x={IN_X} y={BOX_Y} w={IN_W} h={BOX_H} kind="memory" label="static in" labelTone="dark" />
        <Arrow x1={IN_X + IN_W + 3} y1={MID_Y} x2={G_X - 4} y2={MID_Y} kind="memory" muted width={1.2} />

        <rect
          x={G_X} y={BOX_Y - 5} w={G_W} h={BOX_H + 10} rx={4}
          fill={captured ? 'var(--compute-dim)' : 'none'}
          stroke={captured ? 'var(--compute)' : 'var(--rule-2)'}
          strokeWidth={1.4} strokeDasharray={captured ? undefined : '5 4'}
        />
        {captured ? (
          <>
            {kinds.slice(0, 46).map((k, i) => (
              <rect key={i} x={G_X + 8 + i * 7.6} y={BOX_Y + 4} width={4.6} height={BOX_H - 8} rx={1} fill={tickColor(k)} opacity={0.85} />
            ))}
            <Label x={G_X + G_W / 2} y={BOX_Y + BOX_H / 2 + 4} anchor="middle" size={10.5} tone={1} bold>
              {`captured graph · ${fmt.int(v.kernE)} nodes, in order`}
            </Label>
          </>
        ) : (
          <Label x={G_X + G_W / 2} y={MID_Y + 4} anchor="middle" size={10.5} tone={3}>
            {recording ? 'recording the launches — executing nothing' : 'not recorded yet'}
          </Label>
        )}

        <Arrow x1={G_X + G_W + 4} y1={MID_Y} x2={OUT_X - 4} y2={MID_Y} kind="memory" muted width={1.2} />
        <Span x={OUT_X} y={BOX_Y} w={OUT_W} h={BOX_H} kind={out === 'sampled' ? 'token' : 'error'} label={out === 'sampled' ? 'sampled ids' : 'full logits'} labelTone="dark" />

        <Label x={8} y={BOX_Y + BOX_H + 17} size={9.5} tone={3}>
          {`copy in ${v.bytesIn} per step  ·  copy out ${copyOut}  ·  ${out === 'sampled'
            ? 'sampling inside the graph keeps the output to one id per sequence'
            : 'copying every logit out costs more than the weights you just read'}`}
        </Label>
        {recording && (
          <Label x={G_X + 4} y={BOX_Y - 10} size={9.5} kind="error" bold>
            {'capture: the kernels are written down, not run'}
          </Label>
        )}
        {s.step === 3 && (
          <Label x={G_X + 4} y={BOX_Y - 10} size={9.5} kind="token" bold>
            {'replay: cudaGraphLaunch, once per token'}
          </Label>
        )}

        {/* ── band C: the prose ──────────────────────────────────────────── */}
        <Label x={8} y={176} tone={2} size={10.5} mono={false} bold>{script.head}</Label>
        {script.lines.map((ln, i) => (
          <Label key={i} x={8} y={198 + i * 18} tone={3} size={9.5}>{ln}</Label>
        ))}
        <Label x={8} y={252} size={10.5} kind={script.kind} mono={false} bold>{script.verdict}</Label>

        {s.step >= 5 && (
          <g>
            <Label x={8} y={278} size={9.5} tone={3}>
              {s.step === 5
                ? 'Graphs captured per bucket: 1, 4, 16, 64, 256. A batch of 33 replays the 64 graph and throws 31 rows away.'
                : 'Two pools, not one. Graph-captured KV must not be movable, so it cannot be preempted or compacted.'}
            </Label>
            {[1, 4, 16, 64, 256].map((b, i) => (
              <Fade key={b} show={s.step === 5}>
                <rect x={8 + i * 62} y={286} width={54} height={14} rx={2} fill="var(--comm-dim)" stroke="var(--comm)" strokeWidth={1} />
                <Label x={8 + i * 62 + 27} y={296.5} anchor="middle" size={8.5} kind="comm">{`b=${b}`}</Label>
              </Fade>
            ))}
            {s.step === 6 && (
              <>
                <rect x={8} y={286} width={150} height={16} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} />
                <Label x={83} y={297.5} anchor="middle" size={9} kind="memory">movable KV pool</Label>
                <rect x={166} y={286} width={180} height={16} rx={2} fill="var(--error-dim)" stroke="var(--error)" strokeWidth={1} />
                <Label x={256} y={297.5} anchor="middle" size={9} kind="error">graph pool · fixed addresses</Label>
              </>
            )}
          </g>
        )}

        {/* ── band D: what the CPU actually issues ───────────────────────── */}
        <Label x={8} y={318} tone={2} size={10.5} mono={false} bold>One tick per thing the CPU hands the GPU</Label>
        <Label x={8} y={328} size={9} tone={2}>{`eager · ${fmt.int(kinds.length)}`}</Label>
        <Label x={8} y={344} size={9} tone={2}>{`compiled · ${fmt.int(compiledKinds.length)}`}</Label>
        <Label x={8} y={360} size={9} kind="token" bold>replay</Label>
        {[0, 1].map((row) => (
          <g key={row} className="v-fade" style={{ opacity: row === 0 ? 1 : 0.4 }}>
            {(row === 0 ? kinds : compiledKinds).map((k, i) => (
              <rect key={i} x={COMB_X + i * COMB_PITCH} y={320 + row * 16} width={COMB_PITCH * 0.68} height={11} fill={tickColor(k)} />
            ))}
          </g>
        ))}
        <rect x={COMB_X} y={352} width={COMB_PITCH * 0.68} height={11} rx={1} fill="var(--token)" opacity={0.9} />
        <Label x={COMB_X + 70} y={361} size={9.5} kind="token" bold>{`1 call, ${fmt.int(kinds.length)} kernels`}</Label>
        <Label x={632} y={369} anchor="end" size={9} tone={3}>{'the GPU still runs every one of them'}</Label>
      </svg>
    </Stage>
  );
}
