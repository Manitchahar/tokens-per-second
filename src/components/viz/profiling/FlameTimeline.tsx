import { useEffect, useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, Span, fmt } from '../lib/svg';
import { Bracket } from '../benchmarking/helpers';

const STEPS = [
  'A step is not one bar',
  'Zoom in: the CUDA runtime below your code',
  'Where the gaps come from',
  'The three tools, and what each one answers',
  'Nsight Systems: the timeline',
  'Nsight Compute: inside one kernel',
  'The 80% rule, and what to do about the rest',
  'The profiling loop, end to end',
];
const CAPTIONS = [
  'One decode step, drawn as a profiler would: a dense row of GPU work with a thin row of CPU work underneath. The GPU row is what you are trying to explain.',
  'Zooming into any single GEMM bar reveals a dozen tiny bars beneath it: the memset that zeroes the output, the allocator, and the launches the runtime does on your behalf.',
  'On a decode step the GPU row is nearly solid — that is the goal. The interesting question is not "is the GPU busy" but "which bar is long, and why is it longer than the bandwidth floor".',
  'Nsight Systems answers "when and what". Nsight Compute answers "how well" for one kernel. The PyTorch profiler answers "where did my time go" for free, before either of them.',
  'Systems gives you a timeline of every kernel, copy, synchronization and gap, over seconds of real workload. This is where you find launch bubbles, CPU bottlenecks and NCCL stalls.',
  'Compute replays a single kernel many times and reports its memory throughput, compute throughput, achieved occupancy, warp stall reasons and the source line. It rewrites your loop for speed.',
  'If the top kernel is already above 80% of the speed of light, stop optimizing it and go to the next one. Below 80%, Compute will name the reason: coalescing, occupancy, or the memory chart.',
  'Profile, find the longest kernel, ask Compute why it is slow, fix one thing, re-profile. Never change two things at once, or you will not know which one worked.',
];

interface Bar { name: string; kind: 'compute' | 'memory' | 'token' | 'comm' | 'idle' | 'error' | 'text'; x: number; w: number; detail: string }

const PRESET: Record<number, { tool: 'torch' | 'systems' | 'compute'; focus: string }> = {
  0: { tool: 'systems', focus: 'step' },
  1: { tool: 'systems', focus: 'launch' },
  2: { tool: 'systems', focus: 'step' },
  3: { tool: 'systems', focus: 'all' },
  4: { tool: 'systems', focus: 'systems' },
  5: { tool: 'compute', focus: 'compute' },
  6: { tool: 'compute', focus: 'compute' },
  7: { tool: 'torch', focus: 'all' },
};

const X0 = 96;
const X1 = 616;
const SPAN_W = X1 - X0;
const TOTAL = 20.0; // ms in the drawn window

const GPU_TOP: Bar[] = [
  { name: 'reshape', kind: 'memory', x: 0.0, w: 0.7, detail: 'no-op view' },
  { name: 'qkv GEMM', kind: 'compute', x: 0.7, w: 3.1, detail: '4128→12288, m=32' },
  { name: 'rope', kind: 'memory', x: 3.8, w: 0.5, detail: 'fused' },
  { name: 'write K,V', kind: 'memory', x: 4.3, w: 0.6, detail: 'into paged blocks' },
  { name: 'attention', kind: 'compute', x: 4.9, w: 2.4, detail: 'flash, ctx 2048' },
  { name: 'o_proj GEMM', kind: 'compute', x: 7.3, w: 1.9, detail: '4096→4096' },
  { name: 'RMSNorm', kind: 'memory', x: 9.2, w: 0.4, detail: 'fused' },
  { name: 'gate_up GEMM', kind: 'compute', x: 9.6, w: 4.6, detail: '4096→28672' },
  { name: 'SiLU·up', kind: 'memory', x: 14.2, w: 0.9, detail: 'elementwise' },
  { name: 'down GEMM', kind: 'compute', x: 15.1, w: 3.2, detail: '28672→4096' },
  { name: 'RMSNorm', kind: 'memory', x: 18.3, w: 0.4, detail: 'fused' },
  { name: 'lm_head', kind: 'compute', x: 18.7, w: 1.0, detail: '4096→128256' },
];

const CPU_TOP: Bar[] = [
  { name: 'schedule', kind: 'idle', x: 0.0, w: 0.4, detail: '' },
  { name: 'launch×24', kind: 'text', x: 0.4, w: 1.9, detail: 'python + cuda' },
  { name: 'wait', kind: 'idle', x: 2.3, w: 16.6, detail: 'GPU is the bottleneck' },
  { name: 'sample', kind: 'token', x: 18.9, w: 0.8, detail: 'argmax' },
];

const LAUNCH: Bar[] = [
  { name: 'memset out', kind: 'memory', x: 0.0, w: 0.02, detail: 'zero the output' },
  { name: 'gemm', kind: 'compute', x: 0.02, w: 0.14, detail: 'the kernel you meant' },
  { name: 'epilogue', kind: 'compute', x: 0.16, w: 0.03, detail: 'bias + split' },
  { name: 'record', kind: 'text', x: 0.19, w: 0.01, detail: 'cudaEventRecord' },
];

export default function FlameTimeline() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const p = PRESET[s.step];
  const [tool, setTool] = useState(p.tool);
  useEffect(() => { setTool(PRESET[s.step].tool); }, [s.step]);

  const px = (ms: number) => X0 + (ms / TOTAL) * SPAN_W;
  const zoom = s.step === 1;

  // When zoomed, the same 20 ms window shows the interior of a single kernel launch.
  const ZOOM_SCALE = 12;
  const zx = (ms: number) => X0 + ((ms / TOTAL) * SPAN_W * ZOOM_SCALE);

  const gpuTotal = GPU_TOP.reduce((a, b) => a + b.w, 0);
  const gpuBusy = gpuTotal / TOTAL;
  const gaps = TOTAL - gpuTotal;

  return (
    <Stage
      rootRef={ref}
      title="A decode step as a profiler draws it — and the three tools that explain it"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', ['idle', 'gap / waiting'], ['text', 'runtime overhead']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'GPU busy', value: `${(gpuBusy * 100).toFixed(0)}%`, kind: gpuBusy > 0.9 ? 'token' : 'error', note: `${(TOTAL - gpuTotal).toFixed(1)} ms of gaps` },
          { label: 'Longest kernel', value: 'gate_up GEMM', kind: 'compute', note: '4.6 ms · 23% of the step' },
          { label: 'Kernels per step', value: '~24', kind: 'text', note: 'for one 28-layer block' },
          { label: 'Looking with', value: tool === 'torch' ? 'PyTorch profiler' : tool === 'systems' ? 'Nsight Systems' : 'Nsight Compute', kind: 'text', note: tool === 'compute' ? 'one kernel, in detail' : 'the whole timeline' },
        ]} />
      }
      controls={
        <Segmented
          label="Which tool's view is shown"
          value={tool}
          options={[
            { value: 'torch', label: 'PyTorch profiler' },
            { value: 'systems', label: 'Nsight Systems' },
            { value: 'compute', label: 'Nsight Compute' },
          ]}
          onChange={setTool}
        />
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A profiler timeline of one decode step: a dense row of GPU kernel bars above a CPU launch row with a long idle gap, and a zoom showing the small runtime launches hidden inside a single kernel bar.">
        {/* ── GPU row ── */}
        <Label x={16} y={40} tone={1} size={11} bold mono={false}>{s.step === 1 ? 'Zooming into one bar' : 'GPU · one decode step, 20 ms'}</Label>
        <Label x={624} y={40} anchor="end" tone={3} size={9.5}>{`${(gpuBusy * 100).toFixed(0)}% busy`}</Label>

        {zoom ? (
          <g>
            {LAUNCH.map((b) => (
              <Span key={b.name} x={zx(b.x)} y={52} w={(b.w / TOTAL) * SPAN_W * ZOOM_SCALE} h={26} kind={b.kind} label={b.name} />
            ))}
            <Label x={16} y={92} tone={3} size={9}>the kernel you wrote is a small part of what its own bar contains</Label>
          </g>
        ) : (
          <g>
            {GPU_TOP.map((b) => (
              <Span key={b.name} x={px(b.x)} y={52} w={(b.w / TOTAL) * SPAN_W} h={26} kind={b.kind} label={b.name} />
            ))}
            <rect x={px(gpuTotal)} y={52} width={(gaps / TOTAL) * SPAN_W} height={26} rx={2} fill="url(#tps-hatch)" stroke="var(--idle)" strokeWidth={1} />
            <Label x={px(TOTAL / 2)} y={98} anchor="middle" tone={3} size={9.5}>
              24 layers × this pattern, repeated
            </Label>
          </g>
        )}

        {/* ── CPU row ── */}
        <Label x={16} y={126} tone={2} size={11} bold mono={false}>CPU</Label>
        {CPU_TOP.map((b) => (
          <Span key={b.name} x={px(b.x)} y={116} w={(b.w / TOTAL) * SPAN_W} h={18} kind={b.kind} label={b.name} />
        ))}
        <Label x={px(10.6)} y={152} anchor="middle" size={9} kind="idle">idle — the GPU is the bottleneck</Label>

        <Bracket x1={X0} x2={px(gpuTotal)} y={168} label={`${gpuTotal.toFixed(1)} ms of real kernels`} color="var(--compute)" />

        {/* ── step 2: where the gaps come from ── */}
        <Fade show={s.step >= 2}>
          <line x1={16} y1={186} x2={624} y2={186} className="v-rule faint" />
          <Label x={16} y={208} tone={2} size={11} bold mono={false}>Read it this way</Label>
          <Label x={16} y={228} tone={3} size={9.5}>GPU row nearly solid → the step is GPU-bound. Good. Now ask which bar is long.</Label>
          <Label x={16} y={244} tone={3} size={9.5}>GPU row full of holes → you are launch-bound. Fix that (Module 15), not the kernels.</Label>
          <Label x={16} y={260} tone={3} size={9.5}>CPU row busy while GPU idles → you are CPU-bound: Python, sampling, or a slow host.</Label>
        </Fade>

        {/* ── step 3: the three tools ── */}
        <Fade show={s.step >= 3}>
          <line x1={16} y1={276} x2={624} y2={276} className="v-rule faint" />
          <Label x={16} y={298} tone={2} size={11} bold mono={false}>Three tools, three questions</Label>
          {[
            ['PyTorch profiler', 'where did my time go?', 'compute', 'free, no install, gives a per-op table'],
            ['Nsight Systems', 'when did what run, and what were the gaps?', 'memory', 'whole-timeline, CPU + GPU + NCCL'],
            ['Nsight Compute', 'why is this one kernel slow?', 'token', 'one kernel: throughput, stalls, source'],
          ].map(([name, q, kind, note], i) => (
            <g key={name}>
              <rect x={16} y={306 + i * 26} width={5} height={20} rx={1.5} fill={`var(--${kind})`} />
              <Label x={30} y={320 + i * 26} size={10} kind={kind as 'compute'} bold>{name}</Label>
              <Label x={150} y={320 + i * 26} tone={2} size={9.5} mono={false}>{q}</Label>
              <Label x={470} y={320 + i * 26} tone={3} size={9} mono={false}>{note}</Label>
            </g>
          ))}
        </Fade>

        {/* ── the 80% rule ── */}
        <Fade show={s.step >= 6}>
          <Arrow x1={40} y1={392} x2={120} y2={392} kind="error" />
          <Label x={128} y={396} size={10} kind="error" bold>the 80% rule</Label>
          <Label x={226} y={396} tone={3} size={9.5} mono={false}>if the top kernel is already at &gt;80% of the speed of light, stop. Optimize the next one.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
