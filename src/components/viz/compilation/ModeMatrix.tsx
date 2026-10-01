import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, fmt } from '../lib/svg';
import { GPUS, modelById, gpuById } from '../lib/perfModel';
import { MODES, MODE_LABEL, MODE_SHORT, cpuMs, gpuMs, kernelsPerStep, stepMs, ms, short } from './model';

// A single-question instrument, no steps: for this model, on this GPU, at this
// batch, which of the three execution modes is actually buying anything?
// The step time is always max(CPU, GPU), so the table is the whole argument.

const STEPS = ['Which mode does this configuration need?'];
const CAPTIONS = [
  'Three ways to run the same decode step. The step time is the max of its CPU column and its GPU column, so a row only changes the answer when it changes the longer of the two.',
];

const MODEL_IDS = ['qwen05', 'llama3b', 'llama8b'];
const CTX = 1024;
const BAR_X = 74;
const BAR_W = 392;

export default function ModeMatrix() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('qwen05');
  const [gpuId, setGpuId] = useState('h100');
  const [batch, setBatch] = useState(1);

  const model = modelById(modelId);
  const gpu = gpuById(gpuId);
  const input = { model, gpu, batch, ctx: CTX };
  const g = gpuMs(input);
  const rows = MODES.map((m) => {
    const r = stepMs(input, m);
    return { mode: m, kern: kernelsPerStep(model, m), cpu: cpuMs(model, m), step: r.t, tps: r.tps };
  });
  const base = rows[0].step;
  const best = rows.reduce((a, b) => (b.step < a.step ? b : a));
  const gain = base / best.step;
  const launchBound = best.mode !== 'eager' && cpuMs(model, best.mode) > g.t;
  const scale = Math.max(...rows.map((r) => Math.max(r.cpu, g.t))) * 1.05;
  const w = (t: number) => (t / scale) * BAR_W;

  return (
    <Stage
      rootRef={ref}
      title="Which of the three execution modes does this configuration need?"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['error', 'eager, launch-bound'], ['compute', 'torch.compile'], ['memory', 'CUDA graph'], ['text', 'the head and sampling chain']]}
      hideTransport
      {...s}
      readout={
        <Readout items={[
          { label: 'Configuration', value: `${short(model)} · ${gpu.name}`, kind: 'text', note: `batch ${fmt.int(batch)}, ${CTX} ctx, BF16` },
          { label: 'GPU term', value: ms(g.t), kind: 'memory', note: 'bytes over bandwidth — the same in every row' },
          { label: 'Best step', value: ms(best.step), kind: 'token', note: MODE_LABEL[best.mode] },
          { label: 'Gain over eager', value: `${gain.toFixed(2)}×`, kind: gain > 1.3 ? 'token' : 'idle', note: `${best.tps.toFixed(1)} tok/s at batch ${batch}` },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Model"
            value={modelId}
            options={MODEL_IDS.map((id) => ({ value: id, label: short(modelById(id)) }))}
            onChange={setModelId}
          />
          <Segmented
            label="GPU"
            value={gpuId}
            options={GPUS.filter((x) => ['t4', 'a100', 'h100', 'h200'].includes(x.id)).map((x) => ({ value: x.id, label: x.id.toUpperCase() }))}
            onChange={setGpuId}
          />
          <Slider label="Concurrent users (batch)" value={batch} min={1} max={512} log onChange={setBatch} format={(v) => `${fmt.int(v)}`} kind="token" />
        </>
      }
    >
      <svg viewBox="0 0 640 310" role="img" aria-label="A table comparing one decode step run eagerly, compiled and replayed from a CUDA graph, with the launch count, CPU time, GPU time, step time and speed-up for each, plus paired bars showing the CPU and GPU terms on one time scale.">
        <Label x={8} y={20} tone={2} size={10.5} mono={false} bold>
          {`${short(model)} · ${gpu.name} · batch ${fmt.int(batch)} · ${CTX} tokens of context · BF16 · 6 µs per launch`}
        </Label>

        {[['launches', 188], ['CPU term', 288], ['GPU term', 392], ['step', 492], ['vs eager', 600]].map(([t, x]) => (
          <Label key={t} x={Number(x)} y={44} anchor="middle" size={9} tone={3}>{t}</Label>
        ))}

        {rows.map((r, i) => {
          const y = 54 + i * 34;
          const kind = r.mode === 'eager' ? 'error' : r.mode === 'compiled' ? 'compute' : 'memory';
          const bestRow = r.mode === best.mode;
          return (
            <g key={r.mode}>
              <rect x={8} y={y} width={624} height={28} rx={3} fill={bestRow ? 'var(--ground-3)' : 'transparent'} stroke={bestRow ? 'var(--rule-2)' : 'none'} />
              <Label x={18} y={y + 18} size={10} tone={1} bold={bestRow} kind={kind}>{MODE_LABEL[r.mode]}</Label>
              <Label x={188} y={y + 18} anchor="middle" size={9.5} tone={2}>{fmt.int(r.kern)}</Label>
              <Label x={288} y={y + 18} anchor="middle" size={9.5} tone={r.cpu > g.t ? 1 : 2} bold={r.cpu > g.t}>{ms(r.cpu)}</Label>
              <Label x={392} y={y + 18} anchor="middle" size={9.5} kind="memory">{ms(g.t)}</Label>
              <Label x={492} y={y + 18} anchor="middle" size={10} tone={1} bold>{ms(r.step)}</Label>
              <Label x={600} y={y + 18} anchor="middle" size={9.5} kind={bestRow && gain > 1.3 ? 'token' : 'text'} bold={bestRow}>
                {r.mode === 'eager' ? '—' : `${(base / r.step).toFixed(2)}×`}
              </Label>
            </g>
          );
        })}

        <Label x={8} y={178} tone={2} size={10.5} mono={false} bold>The two terms, one time scale · the step is whichever bar is longer</Label>
        {rows.map((r, i) => {
          const y = 190 + i * 30;
          return (
            <g key={`b${r.mode}`}>
              <Label x={8} y={y + 12} size={9} tone={3}>{MODE_SHORT[r.mode]}</Label>
              <rect x={BAR_X} y={y} width={BAR_W} height={7} rx={1.5} fill="none" stroke="var(--rule)" />
              <rect x={BAR_X} y={y} width={Math.max(1, w(r.cpu))} height={7} rx={1.5} fill="var(--text-3)" opacity={0.75} />
              <rect x={BAR_X} y={y + 9} width={BAR_W} height={7} rx={1.5} fill="none" stroke="var(--rule)" />
              <rect x={BAR_X} y={y + 9} width={Math.max(1, w(g.t))} height={7} rx={1.5} fill="var(--memory)" opacity={0.8} />
              <Label x={BAR_X + Math.max(w(r.cpu), w(g.t)) + 8} y={y + 13} size={9.5} tone={1} bold>{ms(r.step)}</Label>
            </g>
          );
        })}
        <Label x={BAR_X} y={286} size={9} tone={3}>grey = CPU issuing launches · blue = GPU reading bytes</Label>
        <Label x={8} y={302} size={10} tone={1} mono={false} bold kind={launchBound ? 'token' : 'idle'}>
          {launchBound
            ? `Launch-bound: ${MODE_LABEL[best.mode]} gives ${gain.toFixed(2)}×, and the GPU was idle most of the time.`
            : `GPU-bound: the CPU term never mattered, so compilation changes almost nothing. Add users, or a smaller model.`}
        </Label>
      </svg>
    </Stage>
  );
}
