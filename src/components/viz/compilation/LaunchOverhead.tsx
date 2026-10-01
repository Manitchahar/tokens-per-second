import { useEffect, useRef, useState } from 'react';
import { Stage, Readout, Slider, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Fade, fmt } from '../lib/svg';
import { GPUS, MODELS, modelById, gpuById } from '../lib/perfModel';
import { LAUNCH_US, cpuMs, gpuMs, kernelsPerStep, stepBytes, stepMs, crossoverBatch, ms, short } from './model';

// Lesson 15.1 — where the time in a decode step actually goes.
// One decode step is two pipelines running at once: the CPU says what to do,
// the GPU does it. This visual is the whole lesson — the two lanes, the
// starvation, and the batch size at which the two sides cross.

const STEPS = [
  'Eager mode is not one thing',
  'The cost is per launch',
  'The GPU has its own cost',
  'Two lanes, running together',
  'The queue runs dry',
  'The step is the longer one',
  'Where the two sides cross',
  'The toll is depth, not size',
];

const CAPTIONS = [
  'One decode step of a 32-layer model is not one operation. PyTorch runs it one operator at a time, and every operator becomes its own kernel launch.',
  'Each launch costs the CPU a few microseconds before the GPU hears about it: a Python call, the dispatcher, the caching allocator, then cudaLaunchKernel.',
  'The GPU cost is a different thing entirely: bytes divided by bandwidth. At batch 1 that is almost all weights and almost no KV cache.',
  'The two pipelines overlap. The GPU starts the first kernel while the CPU is still describing the tenth — as long as the CPU stays ahead.',
  'When the CPU falls behind, the queue drains and the GPU spends the rest of the step on an empty queue. That hatched part is pure waste, and it repeats every token.',
  'The step is max(CPU, GPU), not CPU + GPU. Whichever side is longer is the one you have to make faster.',
  'Raise the batch and the GPU term grows with the KV cache while the CPU term does not move at all. They cross exactly once.',
  'Kernel count follows the number of layers. Bytes follow the number of parameters. Small deep models are launch-bound; large ones are not.',
];

const X0 = 96;
const X1 = 616;
const CPU_Y = 30;
const GPU_Y = 70;
const LANE_H = 17;

const PRESET: Record<number, { model: string; batch: number }> = {
  0: { model: 'llama8b', batch: 1 },
  1: { model: 'llama8b', batch: 1 },
  2: { model: 'llama8b', batch: 1 },
  3: { model: 'llama8b', batch: 1 },
  4: { model: 'llama3b', batch: 1 },
  5: { model: 'llama3b', batch: 1 },
  6: { model: 'llama3b', batch: 1 },
  7: { model: 'qwen05', batch: 1 },
};

const ROWS = ['qwen05', 'llama3b', 'llama8b', 'llama70b'];
const CTX = 1024;

export default function LaunchOverhead() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const preset = PRESET[s.step];
  const [modelId, setModelId] = useState(preset.model);
  const [batch, setBatch] = useState(preset.batch);
  const [gpuId, setGpuId] = useState('h100');
  const [launchUs, setLaunchUs] = useState(LAUNCH_US);
  useEffect(() => { setModelId(preset.model); setBatch(preset.batch); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const model = modelById(modelId);
  const gpu = gpuById(gpuId);

  const kern = kernelsPerStep(model, 'eager');
  const cpu = cpuMs(model, 'eager', launchUs);
  const bytes = stepBytes({ model, gpu, batch, ctx: CTX });
  const g = gpuMs({ model, gpu, batch, ctx: CTX });
  const st = stepMs({ model, gpu, batch, ctx: CTX }, 'eager', launchUs);
  const launchBound = cpu > g.t;
  const xover = crossoverBatch({ model, gpu, ctx: CTX }, 'eager', launchUs);

  // one shared time scale, so the two lanes are honestly comparable
  const scale = Math.max(cpu, g.t) * 1.14;
  const w = (t: number) => (t / scale) * (X1 - X0);
  const cpuW = useTween(w(cpu), 550);
  const gpuW = useTween(w(g.t), 550);

  // ── batch sweep: log x from 1 to 512 users, log y from 0.1 ms to 100 ms ──
  const BY0 = 268;
  const BY1 = 322;
  const bpx = (b: number) => X0 + (Math.log10(b) / Math.log10(512)) * (X1 - X0);
  const bpy = (v: number) => BY1 - ((Math.log10(Math.max(v, 0.1)) + 1) / 3) * (BY1 - BY0);
  const curve = Array.from({ length: 65 }, (_, i) => {
    const b = Math.pow(2, (i / 64) * 9);
    return `${i ? 'L' : 'M'}${bpx(b).toFixed(1)},${bpy(Math.max(cpu, gpuMs({ model, gpu, batch: b, ctx: CTX }).t)).toFixed(1)}`;
  }).join(' ');

  return (
    <Stage
      rootRef={ref}
      title="One decode step: two pipelines, and whichever is longer wins"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'GPU working'],
        ['idle', 'GPU waiting for the CPU'],
        ['text', 'CPU issuing launches'],
        ['error', 'time lost to launching'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'CPU time', value: ms(cpu), kind: launchBound ? 'error' : 'text', note: `${fmt.int(kern)} launches × ${launchUs} µs` },
          { label: 'GPU time', value: ms(g.t), kind: 'memory', note: `${(bytes.total / 1e9).toFixed(2)} GB at ${(gpu.bw * 0.8 / 1e12).toFixed(2)} TB/s` },
          { label: 'Step time', value: ms(st.t), kind: launchBound ? 'error' : 'memory', note: `${st.tps.toFixed(1)} tok/s at batch ${batch}` },
          { label: 'Bottleneck', value: launchBound ? 'the CPU' : 'the GPU', kind: launchBound ? 'error' : 'memory', note: launchBound ? `${(cpu / g.t).toFixed(1)}× of the step is launching` : `${(g.t / cpu).toFixed(1)}× headroom on the CPU` },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Model"
            value={modelId}
            options={MODELS.filter((m) => ROWS.includes(m.id)).map((m) => ({ value: m.id, label: short(m) }))}
            onChange={setModelId}
          />
          <Segmented
            label="GPU"
            value={gpuId}
            options={GPUS.filter((x) => ['t4', 'a100', 'h100', 'h200'].includes(x.id)).map((x) => ({ value: x.id, label: x.id.toUpperCase() }))}
            onChange={setGpuId}
          />
          <Slider label="Concurrent users (batch)" value={batch} min={1} max={512} log onChange={setBatch} format={(v) => `${fmt.int(v)}`} kind="token" />
          <Slider label="CPU cost of one launch" value={launchUs} min={1} max={30} step={0.5} onChange={setLaunchUs} format={(v) => `${v} µs`} kind="idle" />
        </>
      }
    >
      <svg viewBox="0 0 640 356" role="img" aria-label="A decode step drawn as two overlapping pipelines: a CPU lane where a few microseconds per kernel launch are spent describing the step, and a GPU lane where milliseconds are spent reading weights, with the starved part of the GPU hatched. Below, a log-log curve of step time against batch size showing where the two sides cross, and in the last step a table comparing four models.">
        {/* ── the two lanes ─────────────────────────────────────────────── */}
        <Label x={8} y={20} tone={2} size={10.5} mono={false} bold>
          {`${short(model)} · BF16 · ${gpu.name} · batch ${fmt.int(batch)} · ${CTX.toLocaleString('en-US')} tokens of context`}
        </Label>

        <Label x={X0 - 8} y={CPU_Y + 12} anchor="end" tone={2} size={9.5} mono={false}>CPU</Label>
        <Label x={X0 - 8} y={GPU_Y + 12} anchor="end" tone={2} size={9.5} mono={false}>GPU</Label>

        <rect x={X0} y={CPU_Y} width={X1 - X0} height={LANE_H} rx={2} fill="none" stroke="var(--rule-2)" />
        {s.step >= 1 && (
          <>
            <rect x={X0} y={CPU_Y} width={Math.max(1, cpuW)} height={LANE_H} rx={2} fill="var(--text-3)" opacity={0.6} />
            {cpuW > 100 && <Label x={X0 + cpuW / 2} y={CPU_Y + 12} anchor="middle" size={9.5} tone={1} bold>{`${fmt.int(kern)} launches`}</Label>}
          </>
        )}

        <rect x={X0} y={GPU_Y} width={X1 - X0} height={LANE_H} rx={2} fill="none" stroke="var(--rule-2)" />
        {s.step >= 2 && (
          <>
            <rect x={X0} y={GPU_Y} width={Math.max(1, gpuW)} height={LANE_H} rx={2} fill="var(--memory)" opacity={0.75} />
            {gpuW > 130 && <Label x={X0 + gpuW / 2} y={GPU_Y + 12} anchor="middle" size={9.5} tone={1} bold>{`${(bytes.total / 1e9).toFixed(2)} GB @ ${(gpu.bw * 0.8 / 1e12).toFixed(2)} TB/s`}</Label>}
          </>
        )}
        {s.step >= 4 && launchBound && cpuW > gpuW + 2 && (
          <g>
            <rect x={X0 + gpuW} y={GPU_Y} width={cpuW - gpuW} height={LANE_H} fill="url(#tps-hatch)" stroke="var(--idle)" strokeWidth={1} />
            <Label x={X0 + (cpuW + gpuW) / 2} y={GPU_Y - 5} anchor="middle" size={9.5} kind="error">{`starved for ${ms(cpu - g.t)}`}</Label>
          </g>
        )}

        {s.step >= 5 && (
          <g>
            <line x1={X0} x2={X0 + Math.max(cpuW, gpuW)} y1={GPU_Y + LANE_H + 13} y2={GPU_Y + LANE_H + 13} stroke="var(--text-2)" strokeWidth={1} />
            <line x1={X0} x2={X0} y1={GPU_Y + LANE_H + 9} y2={GPU_Y + LANE_H + 17} stroke="var(--text-2)" strokeWidth={1} />
            <line x1={X0 + Math.max(cpuW, gpuW)} x2={X0 + Math.max(cpuW, gpuW)} y1={GPU_Y + LANE_H + 9} y2={GPU_Y + LANE_H + 17} stroke="var(--text-2)" strokeWidth={1} />
            <Label x={X0 + Math.max(cpuW, gpuW) / 2} y={GPU_Y + LANE_H + 28} anchor="middle" size={10} tone={1} bold>
              {`one step = ${ms(st.t)} = max(${ms(cpu)} CPU, ${ms(g.t)} GPU) + 0.2 ms`}
            </Label>
          </g>
        )}

        {/* ── step 0: the operator list eager mode actually walks ───────── */}
        {s.step === 0 && (
          <g>
            <Label x={8} y={136} tone={2} size={10.5} mono={false} bold>What the CPU is asked to do, one operator at a time</Label>
            {['nrm', 'Q', 'K', 'V', 'rope', 'attn', 'O', '+res', 'mlp'].map((op, i) => (
              <Span key={op} x={8 + i * 68} y={146} w={62} h={20} kind={i % 3 === 0 ? 'compute' : i % 3 === 1 ? 'memory' : 'comm'} label={op} labelTone="dark" />
            ))}
            <Label x={X1} y={160} anchor="end" size={9.5} tone={3}>× {model.layers} layers, then the head</Label>
            <Label x={8} y={188} tone={3} size={9.5}>Seven matrix multiplies (amber) and eight small elementwise kernels (blue, violet) per layer, each with its own launch.</Label>
            <Label x={8} y={206} tone={3} size={9.5}>{`Times ${model.layers} layers, plus the head and the sampling chain: ${fmt.int(kern)} launches to produce one token.`}</Label>
            <Label x={8} y={224} tone={3} size={9.5}>Move the sliders. The CPU lane and the GPU lane are the whole cost of a token.</Label>
          </g>
        )}

        {/* ── step 1: what one launch costs ─────────────────────────────── */}
        {s.step === 1 && (
          <g>
            <Label x={8} y={136} tone={2} size={10.5} mono={false} bold>Zoom in on the first {launchUs} µs of the CPU lane</Label>
            {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
              <rect key={i} x={8 + i * 26} y={148} width={4} height={22} rx={1} fill="var(--text-3)" opacity={0.7} />
            ))}
            <Label x={8} y={186} tone={3} size={9.5}>{'Python call → dispatcher → caching allocator → cudaLaunchKernel. Four things, none of them the math.'}</Label>
            <Label x={8} y={204} tone={3} size={9.5}>{`${fmt.int(kern)} × ${launchUs} µs = ${ms(cpu)} of CPU, before the GPU has done a single byte of work.`}</Label>
            <Label x={8} y={222} tone={3} size={9.5}>None of this is the GPU's fault, and none of it is the model's fault. It is the price of being written one operator at a time.</Label>
          </g>
        )}

        {/* ── step 2: what the GPU side costs ───────────────────────────── */}
        {s.step === 2 && (
          <g>
            <Label x={8} y={136} tone={2} size={10.5} mono={false} bold>The GPU side is a completely different bill</Label>
            <Label x={8} y={160} tone={3} size={9.5}>bytes ÷ bandwidth</Label>
            <rect x={132} y={150} width={470} height={16} rx={2} fill="none" stroke="var(--rule-2)" />
            <rect x={132} y={150} width={(bytes.weights / bytes.total) * 470} height={16} rx={2} fill="var(--memory)" opacity={0.75} />
            <Label x={140} y={162} size={9.5} tone={1} bold>{`weights ${(bytes.weights / 1e9).toFixed(2)} GB`}</Label>
            <Label x={598} y={182} anchor="end" size={9} kind="memory">{`KV ${(bytes.kv / 1e6).toFixed(1)} MB (${((bytes.kv / bytes.total) * 100).toFixed(1)}%)`}</Label>
            <Label x={8} y={206} tone={3} size={9.5}>{`At batch ${fmt.int(batch)} almost all of it is weights, so the GPU time barely moves as you add users — until the KV cache grows.`}</Label>
            <Label x={8} y={224} tone={3} size={9.5}>{`${(bytes.total / 1e9).toFixed(2)} GB ÷ ${(gpu.bw * 0.8 / 1e12).toFixed(2)} TB/s = ${ms(g.t)}. One number, and it is Module 02's roofline floor.`}</Label>
          </g>
        )}

        {/* ── step 3: overlap ───────────────────────────────────────────── */}
        {s.step === 3 && (
          <g>
            <Label x={8} y={136} tone={2} size={10.5} mono={false} bold>The two lanes do not add up — they overlap</Label>
            <Label x={8} y={160} tone={3} size={9.5}>{'kernel 1 → 2 → 3 runs on the GPU while the CPU is still describing kernels 4 → 5 → 6.'}</Label>
            <Label x={8} y={180} tone={3} size={9.5}>The GPU sees a queue, not a script. As long as the queue is never empty, the CPU is free.</Label>
            <Label x={8} y={206} tone={1} size={10} mono={false} bold>{`CPU ${ms(cpu)}   ·   GPU ${ms(g.t)}   ·   step ${ms(st.t)}`}</Label>
            <Label x={8} y={224} tone={3} size={9.5}>{`Because ${ms(cpu)} ${launchBound ? '>' : '<'} ${ms(g.t)}, the step is ${ms(st.t)} — and ${ms(cpu - g.t)} of it is the CPU working alone.`}</Label>
          </g>
        )}

        {/* ── step 4: starvation ────────────────────────────────────────── */}
        {s.step === 4 && (
          <g>
            <Label x={8} y={136} tone={2} size={10.5} mono={false} bold>What an empty queue looks like</Label>
            {launchBound ? (
              <>
                <Label x={8} y={162} tone={3} size={9.5}>{`The GPU finishes its last kernel and waits ${ms(cpu - g.t)} for the CPU to describe the next one. Every step. Every token.`}</Label>
                <Label x={8} y={186} kind="error" size={11} mono={false} bold>{`${((1 - g.t / cpu) * 100).toFixed(0)}% of the machine is idle while this request waits.`}</Label>
                <Label x={8} y={210} tone={3} size={9.5}>This is not a tail-latency problem you can average away. It is a constant tax on every single token.</Label>
              </>
            ) : (
              <>
                <Label x={8} y={162} tone={3} size={9.5}>{`The CPU is ${ms(cpu)} and the GPU is ${ms(g.t)}, so the queue never empties: the GPU is busy ${((g.t / Math.max(cpu, g.t)) * 100).toFixed(0)}% of the step.`}</Label>
                <Label x={8} y={186} kind="memory" size={11} mono={false} bold>Nothing is starved here. That is a configuration worth protecting.</Label>
                <Label x={8} y={210} tone={3} size={9.5}>But the CPU still has to finish ${fmt.int(kern)} launches per step, which is the headroom you would spend on a busier machine.</Label>
              </>
            )}
          </g>
        )}

        {/* ── step 5: the verdict ───────────────────────────────────────── */}
        {s.step === 5 && (
          <g>
            <Label x={8} y={140} tone={2} size={10.5} mono={false} bold>So which side is longer?</Label>
            <Label x={8} y={168} size={12} mono={false} bold kind={launchBound ? 'error' : 'memory'}>
              {launchBound ? `The CPU is the bottleneck: ${ms(cpu)} of a ${ms(st.t)} step.` : `The GPU is the bottleneck: ${ms(g.t)} of a ${ms(st.t)} step.`}
            </Label>
            <Label x={8} y={196} tone={3} size={9.5}>{`Either way the step is ${ms(st.t)}, and the only two ways to shorten it are fewer, cheaper launches or less data.`}</Label>
            <Label x={8} y={216} tone={3} size={9.5}>One decode step produces one token. Multiply by the output length and you have the number the user feels.</Label>
          </g>
        )}

        {/* ── step 6 & 7: the sweep, and the model comparison ───────────── */}
        {s.step >= 6 && (
          <g>
            <Label x={8} y={136} tone={2} size={10.5} mono={false} bold>
              {s.step === 6 ? 'Add users: the CPU line is flat, the GPU line grows' : 'One H100, batch 1, 1,024 tokens of context — which side is longer'}
            </Label>

            {s.step === 6 && (
              <g>
                {[0.1, 1, 10, 100].map((v) => (
                  <g key={v}>
                    <line x1={X0} x2={X1} y1={bpy(v)} y2={bpy(v)} stroke="var(--rule)" strokeWidth={1} />
                    <Label x={X0 - 6} y={bpy(v) + 3.2} anchor="end" tone={3} size={9}>{v < 1 ? '0.1' : v}</Label>
                  </g>
                ))}
                <Label x={X0 - 6} y={bpy(0.1) - 5} anchor="end" tone={3} size={8.5}>ms</Label>
                {[1, 4, 16, 64, 256, 512].map((b) => (
                  <Label key={`b${b}`} x={bpx(b)} y={BY1 + 13} anchor="middle" tone={3} size={9}>{b}</Label>
                ))}
                <Label x={X1} y={BY1 + 26} anchor="end" tone={3} size={9}>concurrent users →</Label>

                {xover > 0 && xover < 512 && (
                  <path d={`M${bpx(1)},${bpy(cpu)} L${bpx(xover)},${bpy(cpu)} L${bpx(xover)},${bpy(gpuMs({ model, gpu, batch: xover, ctx: CTX }).t)} Z`} fill="var(--error)" opacity={0.15} />
                )}
                <path d={`M${bpx(1)},${bpy(cpu)} L${bpx(512)},${bpy(cpu)}`} fill="none" stroke="var(--text-2)" strokeWidth={1.4} strokeDasharray="5 4" />
                <Label x={bpx(2)} y={bpy(cpu) - 6} size={9.5} tone={2}>{`CPU: flat at ${ms(cpu)}`}</Label>
                <path d={curve} fill="none" stroke="var(--memory)" strokeWidth={1.8} />
                {xover > 0 && xover < 512 && (
                  <>
                    <circle cx={bpx(xover)} cy={bpy(cpu)} r={4.5} fill="var(--error)" stroke="var(--ground-2)" strokeWidth={2} />
                    <Label x={bpx(xover)} y={bpy(cpu) + 17} anchor="middle" size={9.5} kind="error">{`${fmt.int(xover)} users`}</Label>
                  </>
                )}
                {xover === 0 && <Label x={bpx(2)} y={bpy(gpuMs({ model, gpu, batch: 1, ctx: CTX }).t) - 7} size={9.5} kind="memory">already GPU-bound at one user</Label>}
                <line x1={bpx(batch)} x2={bpx(batch)} y1={BY0} y2={BY1} stroke="var(--text)" strokeWidth={1} strokeDasharray="2 3" />
                <circle cx={bpx(batch)} cy={bpy(st.t)} r={3.5} fill="var(--text)" />
                <Label x={8} y={234} tone={3} size={9.5}>
                  {`Below the crossing the step time is the CPU's, and every user you add is free. Above it, every user costs KV bandwidth.`}
                </Label>
              </g>
            )}

            {s.step === 7 && (
              <g>
                {ROWS.map((id, i) => {
                  const m = modelById(id);
                  const c = cpuMs(m, 'eager', launchUs);
                  const gg = gpuMs({ model: m, gpu, batch: 1, ctx: CTX }).t;
                  const lb = c > gg;
                  const bx = 396;
                  const bw = 216;
                  const frac = c / (c + gg);
                  return (
                    <g key={id}>
                      <Label x={8} y={166 + i * 26} size={10} tone={1}>{short(m)}</Label>
                      <Label x={116} y={166 + i * 26} size={9} tone={3}>{`${m.layers} layers`}</Label>
                      <Label x={186} y={166 + i * 26} size={9} tone={2}>{`${fmt.int(kernelsPerStep(m, 'eager'))} kernels`}</Label>
                      <Label x={268} y={166 + i * 26} size={9} tone={1} bold>{`${ms(c)} CPU`}</Label>
                      <Label x={268} y={178 + i * 26} size={9} kind="memory">{`${ms(gg)} GPU`}</Label>
                      <rect x={bx} y={157 + i * 26} width={bw} height={16} rx={2} fill="none" stroke="var(--rule)" />
                      <rect x={bx} y={157 + i * 26} width={bw * frac} height={16} rx={2} fill="var(--text-3)" opacity={0.65} />
                      <rect x={bx + bw * frac} y={157 + i * 26} width={bw * (1 - frac)} height={16} rx={2} fill="var(--memory)" opacity={0.7} />
                      <Label x={X1} y={169 + i * 26} anchor="end" size={9.5} kind={lb ? 'error' : 'memory'} bold={lb}>
                        {lb ? `launch-bound ×${(c / gg).toFixed(1)}` : `GPU-bound ×${(gg / c).toFixed(1)}`}
                      </Label>
                    </g>
                  );
                })}
                <Label x={8} y={272} tone={3} size={9.5}>{'grey = CPU launch time, blue = GPU memory time, each row split so the longer half shows. The CPU number tracks layers; the GPU number tracks parameters.'}</Label>
              </g>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
