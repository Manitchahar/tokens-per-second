import { useEffect, useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Chip, Fade, fmt } from '../lib/svg';
import { WORKLOADS, engineRows, KIND_COLOR, type Workload } from './atlasModel';
import { ENGINES } from './fairModel';

// Seven steps: what an engine is, the cost of a step, then one step per engine,
// then the axis that decides it, and finally the matrix with its three column kinds.
const STEPS = [
  'One step, four engines',
  'What an engine actually does',
  'vLLM: the general-purpose one',
  'SGLang: prefix-heavy traffic',
  'TensorRT-LLM: pay for the build',
  'llama.cpp: the 4-bit answer',
  'Which column decides it',
];

const CAPTIONS = [
  'Four engines, one model, one GPU, one decode step. Each bar is that step split into the time the GPU spends moving bytes and doing math, and the time it sits idle waiting for the CPU to launch the next kernel. The shapes are the argument of this lesson: the engines differ in overhead, in how close their kernels get to peak, and in whether a 4-bit path exists — and almost nothing else.',
  'An engine is not the model and not the kernels. It is the machinery that decides when each of them runs: a scheduler on the CPU picks which sequences go into the next forward pass, a KV manager allocates and frees blocks, and a launcher issues a few hundred kernels. Everything in the idle slice of the bar above is that machinery, and it is the part each engine designs differently.',
  'vLLM is the general-purpose engine and the one most serving stacks are built on. Paged KV blocks and continuous batching (Modules 08 and 07) do the memory work; the per-step cost is Python-side scheduling plus a few hundred kernel launches, which is the idle slice. Its strength is breadth: nearly every model, a large ecosystem, and the most documentation.',
  'SGLang came out of the same lineage and its distinguishing idea is RadixAttention: the prefix cache is a radix tree that the attention kernel walks directly, so a shared system prompt or multi-turn history is read once without a gather copy. Its fixed cost is similar, but on prefix-heavy traffic it does less work per step — which is a workload effect, not a faster-kernel effect.',
  'TensorRT-LLM runs a build phase ahead of serving. It traces the model, fuses operators, autotunes kernels for one exact configuration of model, precision, GPU and parallel layout, and emits an execution plan. The plan replays with very few launches, so the idle slice nearly vanishes — at the cost of a build you repeat whenever any of those inputs change, and a plan that is invalid on any other GPU.',
  'llama.cpp is the outlier: portable, no CUDA requirement, and built around 4-bit weights. It reaches a smaller fraction of peak bandwidth and pays a larger fixed cost per step than the GPU-native engines, and on an NVIDIA card serving an unquantized model it will usually lose. Where it wins is the case nothing else can serve at all — a 32B model in 16.4 GB on a 24 GB card, or a laptop with no GPU at all.',
  'So which one? Not by looking at the throughput column, which is the noisiest and least transferable number on the screen. Start from the columns you can derive: weights, KV pool, whether the model fits. Then apply the hard constraints. Only among the survivors do the measured columns decide, and even then the winner depends on the workload you selected above.',
];

/** Per-step preset of the batch size, so each step makes its point without the reader hunting. */
const PRESET: Record<number, string> = {
  0: 'latency', 1: 'latency', 2: 'server', 3: 'server', 4: 'bulk', 5: 'edge', 6: 'server',
};

export default function EngineAtlas() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [wlId, setWlId] = useState(PRESET[0]);
  const [ctx, setCtx] = useState(2048);

  useEffect(() => setWlId(PRESET[s.step] ?? 'server'), [s.step]);

  const wl = WORKLOADS.find((w) => w.id === wlId)!;
  const work: Workload = { ...wl.w, ctx };
  const rows = engineRows(work);
  const slowest = Math.max(...rows.map((r) => r.tTotal));

  // Each step nominates the engine it is talking about; the last step shows all four.
  const focus = s.step >= 2 && s.step <= 5 ? ENGINES[s.step - 2].id : null;

  return (
    <Stage
      rootRef={ref}
      title="Four engines, one decode step: where the time goes"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'memory traffic'],
        ['compute', 'compute'],
        ['idle', 'CPU: scheduling and launches'],
        ['error', 'does not fit'],
        ['text', 'per-user tokens/s'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Workload', value: wl.label, kind: 'text', note: `${work.batch} stream${work.batch === 1 ? '' : 's'} · ${fmt.int(ctx)} ctx` },
          { label: 'Bottleneck', value: rows[0].bound, kind: rows[0].bound === 'compute' ? 'compute' : 'memory', note: 'shared by all four here' },
          {
            label: 'Leads at this workload',
            value: rows.reduce((a, b) => (b.tTotal < a.tTotal ? b : a)).engine.name,
            kind: 'token',
            note: `${(slowest / rows.reduce((a, b) => (b.tTotal < a.tTotal ? b : a)).tTotal).toFixed(2)}x the slowest`,
          },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Workload"
            value={wlId}
            options={WORKLOADS.map((w) => ({ value: w.id, label: w.label }))}
            onChange={setWlId}
          />
          <Slider
            label="Context per stream (moves the KV slice)"
            kind="memory"
            min={512}
            max={16384}
            step={512}
            value={ctx}
            onChange={setCtx}
            format={(v) => `${fmt.int(v)} tokens`}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="Four inference engines compared on one decode step. Each row is a horizontal bar split into memory traffic, compute, and an idle slice where the CPU schedules and launches kernels while the GPU waits. A workload selector changes the batch size and model, and a context slider grows the KV slice.">
        <Label x={40} y={20} tone={2} size={11} bold>One decode step, split into where the time goes</Label>
        <Label x={614} y={20} anchor="end" tone={3} size={9}>{focus ? `showing ${ENGINES.find((e) => e.id === focus)!.name}` : 'all four'}</Label>

        {rows.map((r, i) => {
          const y = 40 + i * 46;
          const W = 372;
          const gpuW = (r.tGpu / slowest) * W;
          const memW = (r.tMem / slowest) * W;
          const cpuW = (r.tCpu / slowest) * W;
          const dim = focus !== null && focus !== r.engine.id;
          return (
            <g key={r.engine.id} style={{ opacity: dim ? 0.32 : 1 }} className="v-fade">
              <Label x={40} y={y + 12} size={10.5} bold tone={1}>{r.engine.name}</Label>
              <Label x={40} y={y + 26} size={8.5} tone={3}>{r.engine.trade}</Label>

              {/* the step */}
              <rect x={166} y={y} width={W} height={22} rx={2} fill="none" stroke="var(--rule)" />
              {!r.fitsWeights ? (
                <rect x={166} y={y} width={W} height={22} rx={2} fill="url(#tps-hatch)" stroke="var(--error)" />
              ) : (
                <>
                  <rect x={166} y={y} width={Math.max(1, memW)} height={22} rx={2}
                    fill="var(--memory)" fillOpacity={0.85} className="v-cell" />
                  {/* compute only shows when it is the longer of the two, i.e. genuinely binding */}
                  {r.tFlop > r.tMem && (
                    <rect x={166} y={y} width={Math.max(1, (r.tFlop / slowest) * W)} height={22} rx={2}
                      fill="var(--compute)" fillOpacity={0.85} className="v-cell" />
                  )}
                  <rect x={166 + memW} y={y} width={Math.max(1, cpuW)} height={22} rx={2}
                    fill="url(#tps-hatch)" stroke="var(--idle)" className="v-cell" />
                </>
              )}

              <Label x={166 + W + 10} y={y + 15} size={10.5} tone={1}>
                {r.fitsWeights ? `${(r.tTotal * 1000).toFixed(1)} ms` : '—'}
              </Label>
              <Label x={614} y={y + 15} size={9} tone={3} anchor="end">
                {r.fitsWeights ? `${r.tokensPerUser.toFixed(0)}/user` : 'no fit'}
              </Label>
            </g>
          );
        })}

        {/* axis labels */}
        <Label x={166} y={232} size={9} tone={3}>0 ms</Label>
        <Label x={538} y={232} anchor="end" size={9} tone={3}>{(slowest * 1000).toFixed(1)} ms — the slowest engine's whole step</Label>
        <line x1={166} y1={226} x2={538} y2={226} className="v-rule faint" />

        {/* the one-line reading */}
        <Fade show={s.step >= 1}>
          <line x1={40} y1={244} x2={614} y2={244} className="v-rule faint" />
          <Label x={40} y={262} tone={2} size={10.5} bold>The two slices, and which one each engine is fighting</Label>
          <Label x={40} y={280} size={9.5} kind="memory">memory — weights plus every stream's KV, read once per step</Label>
          <Label x={40} y={296} size={9.5} kind="idle">idle — CPU picks the batch and launches the kernels; the GPU waits</Label>
          <Label x={40} y={312} size={9.5} kind="compute">compute — only drawn when the math is the longer of the two</Label>
        </Fade>

        <Fade show={s.step < 1}>
          <Label x={40} y={266} tone={3} size={10} mono={false}>
            Every bar is one forward pass. The right-hand end is the slowest engine's whole step, so the
            bars are directly comparable — but nothing here is a benchmark, only the shape of the cost.
          </Label>
        </Fade>

        {/* the matrix, last step */}
        <Fade show={s.step >= 6}>
          <Label x={40} y={336} tone={2} size={10.5} bold>Read the columns in this order — the cheap certain ones first</Label>
          <Label x={40} y={354} size={9.5} kind="memory">derived: weights 16.1 GB, KV pool, does it fit</Label>
          <Label x={40} y={368} size={9.5} kind="compute">structural: build step, vendor lock, model coverage</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
