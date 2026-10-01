import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, fmt } from '../lib/svg';
import { GPUS, MODELS, gpuById, modelById } from '../lib/perfModel';
import { kvPerToken, weightBytes, attnFlops, linearFlops } from './kvMath';

const MFU = 0.55, MBU = 0.8, OVERHEAD = 0.0015;

const STEPS = [
  'A request arrives',
  'Prefill: the whole prompt at once',
  'Prefill is compute-bound',
  'The first token comes out',
  'Decode: one token at a time',
  'Decode is memory-bound',
  'One model, two workloads',
];
const CAPTIONS = [
  'A request arrives carrying a prompt. Everything the GPU does for it happens in two phases, and they want opposite things from the hardware.',
  'Prefill pushes every prompt token through the model in a single batched pass, filling the KV cache and producing the first token.',
  'Because hundreds of tokens share one read of the weights, prefill runs at hundreds of FLOPs per byte: compute-bound, with the tensor cores busy.',
  'That first token is the end of prefill. From here on the request is on its own, one token per step, with no more parallelism to find.',
  'Each decode step handles exactly one token, so every weight is used once. There is nothing to amortise and no arithmetic to hide behind.',
  'A decode step moves all 16 GB of weights to do about 16 GFLOP: roughly one FLOP per byte, hard against the bandwidth floor.',
  'Same weights, same kernels, same GPU. Opposite bottlenecks, opposite optimizations, opposite costs. This is the distinction the whole course is built on.',
];

const PF_X = 104, PF_W = 104;
const DC_X = 232, DC_W = 94, DC_N = 4;
const AX0 = 40, AX1 = 600, AX_Y = 282;
const CHART_MAX = 600;
const lx = (I: number) => AX0 + ((Math.log10(Math.max(I, 0.3)) + 0.523) / 4) * (AX1 - AX0);

export default function TwoPhases() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [gpuId, setGpuId] = useState('h100');
  const [tp, setTp] = useState(512);
  const [n, setN] = useState(128);

  useEffect(() => {
    if (s.step <= 1 || s.step === 3) setTp(512);
  }, [s.step]);

  const model = modelById(modelId);
  const gpu = gpuById(gpuId);
  const ridge = gpu.flops / gpu.bw;
  const td = tp + n / 2;

  const pfFlops = linearFlops(model, tp) + attnFlops(model, tp);
  const pfBytes = weightBytes(model, 2) + tp * model.hidden * 2 + tp * kvPerToken(model, 2);
  const pfCompute = pfFlops / (gpu.flops * MFU);
  const pfMem = pfBytes / (gpu.bw * MBU);
  const pfTime = Math.max(pfCompute, pfMem) + 0.002;
  const pfI = pfFlops / pfBytes;

  const dcFlops = 2 * model.params + 4 * model.hidden * td * model.layers;
  const dcBytes = weightBytes(model, 2) + td * kvPerToken(model, 2);
  const dcCompute = dcFlops / (gpu.flops * MFU);
  const dcMem = dcBytes / (gpu.bw * MBU);
  const dcTime = Math.max(dcCompute, dcMem) + OVERHEAD;
  const dcI = dcFlops / dcBytes;
  const mathFrac = dcCompute / dcMem;
  const pfIc = Math.min(pfI, CHART_MAX);

  return (
    <Stage
      rootRef={ref}
      title="Prefill and decode: the same model, two opposite workloads"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['text', 'not started yet']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Time to first token', value: fmt.ms(pfTime), kind: 'compute', note: 'all of prefill' },
          { label: 'Time per token after', value: fmt.ms(dcTime), kind: 'memory', note: 'one decode step' },
          { label: 'Prefill intensity', value: `${fmt.int(pfI)} FLOP/byte`, kind: pfI > ridge ? 'compute' : 'memory', note: pfI > ridge ? 'compute-bound' : 'memory-bound, it is too short' },
          { label: 'Decode intensity', value: `${dcI.toFixed(1)} FLOP/byte`, kind: 'memory', note: 'memory-bound at any context' },
        ]} />
      }
      controls={
        <>
          <Slider label="Prompt length (prefill)" value={tp} min={8} max={32768} log onChange={setTp} format={(v) => `${fmt.int(v)} tokens`} kind="compute" />
          <Slider label="Tokens generated (decode steps)" value={n} min={8} max={1024} log onChange={setN} format={(v) => `${fmt.int(v)}`} kind="memory" />
          <Segmented label="Model" value={modelId} options={MODELS.filter((m) => ['qwen05', 'llama8b', 'llama70b'].includes(m.id)).map((m) => ({ value: m.id, label: m.name.replace('Llama-3.1-', '').replace('Qwen2.5-', '') }))} onChange={setModelId} />
          <Segmented label="GPU" value={gpuId} options={GPUS.filter((g) => ['t4', 'a100', 'h100'].includes(g.id)).map((g) => ({ value: g.id, label: g.id.toUpperCase() }))} onChange={setGpuId} />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A timeline of one request: a short prefill pass followed by many memory-bound decode steps, a side-by-side comparison of the two phases, and where each lands on the roofline.">
        <Label x={24} y={16} tone={3} size={10}>{`${model.name} on one ${gpu.name} · ${fmt.int(tp)}-token prompt, ${fmt.int(n)} tokens out`}</Label>

        <text x={PF_X - 10} y={36} textAnchor="end" className="v-track-label">Prefill</text>
        <rect x={PF_X} y={24} width={PF_W} height={20} rx={2} fill="none" stroke="var(--rule-2)" strokeDasharray="3 3" />
        <Fade show={s.step >= 1}>
          <Span x={PF_X} y={24} w={PF_W} h={20} kind={s.step >= 2 ? 'compute' : 'text'} label={`${fmt.int(tp)} tokens`} />
        </Fade>

        <text x={PF_X - 10} y={70} textAnchor="end" className="v-track-label">Decode</text>
        <rect x={DC_X} y={52} width={DC_W * DC_N} height={22} rx={2} fill="none" stroke="var(--rule-2)" strokeDasharray="3 3" />
        <Fade show={s.step >= 4}>
          {Array.from({ length: DC_N }, (_, i) => {
            const x = DC_X + i * DC_W;
            return (
              <g key={i}>
                <Span x={x} y={52} w={DC_W - 3} h={10} kind="memory" dim={s.step < 5} label={i === 0 ? '1 token' : undefined} labelTone="light" />
                <rect x={x} y={64} width={Math.min(DC_W - 3, Math.max(2, (DC_W - 3) * mathFrac))} height={10} rx={1} fill="var(--compute)" />
              </g>
            );
          })}
        </Fade>
        <line x1={PF_X + PF_W + 6} y1={28} x2={PF_X + PF_W + 12} y2={34} stroke="var(--text-3)" strokeWidth={1.2} />
        <line x1={PF_X + PF_W + 11} y1={28} x2={PF_X + PF_W + 17} y2={34} stroke="var(--text-3)" strokeWidth={1.2} />

        <Label x={PF_X} y={90} tone={1} size={10} kind="compute" bold>{fmt.ms(pfTime)}</Label>
        <Label x={DC_X} y={90} tone={1} size={10} kind="memory" bold>{`${fmt.ms(dcTime)} each`}</Label>
        <Label x={DC_X + DC_W * DC_N} y={90} anchor="end" tone={3} size={9.5}>{`… and ${fmt.int(n)} of them`}</Label>
        <Label x={24} y={106} tone={2} size={10.5} mono={false}>
          {`Time to first token ${fmt.ms(pfTime)} — about ${(pfTime / dcTime).toFixed(1)} decode steps. The remaining ${fmt.int(n)} tokens cost ${fmt.ms(dcTime * n)}.`}
        </Label>

        {[0, 1].map((half) => {
          const x = half === 0 ? 24 : 322;
          const flops = half === 0 ? pfFlops : dcFlops;
          const bytes = half === 0 ? pfBytes : dcBytes;
          const inten = half === 0 ? pfI : dcI;
          const bound = inten > ridge;
          return (
            <g key={half}>
              <rect x={x} y={116} width={294} height={116} rx={4} fill="var(--ground-3)" stroke={bound ? 'var(--compute)' : 'var(--memory)'} strokeWidth={1.2} />
              <Label x={x + 12} y={134} size={11} kind={bound ? 'compute' : 'memory'} bold mono={false}>{half === 0 ? 'Prefill' : 'Decode'}</Label>
              {[
                ['tokens per step', half === 0 ? fmt.int(tp) : '1'],
                ['FLOPs per step', flops >= 1e12 ? `${(flops / 1e12).toFixed(2)} TFLOP` : `${(flops / 1e9).toFixed(1)} GFLOP`],
                ['bytes per step', `${(bytes / 1e9).toFixed(1)} GB`],
                ['arithmetic intensity', `${inten < 10 ? inten.toFixed(1) : fmt.int(inten)} FLOP/byte`],
                ['limited by', bound ? 'compute' : 'memory'],
              ].map(([k, v], i) => (
                <g key={k}>
                  <Label x={x + 12} y={152 + i * 16} tone={2} size={9.5} mono={false}>{k}</Label>
                  <Label x={x + 282} y={152 + i * 16} anchor="end" tone={1} size={9.5}>{v}</Label>
                </g>
              ))}
            </g>
          );
        })}

        <line x1={AX0} x2={AX1} y1={AX_Y} y2={AX_Y} className="v-rule" />
        <Label x={24} y={AX_Y - 34} tone={2} size={10.5} mono={false} bold>Arithmetic intensity, log scale</Label>
        <line x1={lx(ridge)} x2={lx(ridge)} y1={AX_Y - 6} y2={AX_Y} stroke="var(--text-3)" strokeDasharray="3 3" />
        <Label x={lx(ridge) - 6} y={AX_Y - 10} anchor="end" size={9} tone={3}>{`${gpu.id.toUpperCase()} ridge ${fmt.int(ridge)}`}</Label>
        {[1, 10, 100, 1000].map((v) => (
          <g key={v}>
            <line x1={lx(v)} x2={lx(v)} y1={AX_Y} y2={AX_Y + 4} className="v-rule" />
            <Label x={lx(v)} y={AX_Y + 16} anchor="middle" tone={3} size={9}>{v}</Label>
          </g>
        ))}
        <Fade show={s.step >= 2}>
          <circle cx={lx(pfIc)} cy={AX_Y} r={5} fill="var(--compute)" />
          <Label x={lx(pfIc)} y={AX_Y - 10} anchor="middle" size={9.5} kind="compute" bold>{`prefill: ${pfI > CHART_MAX ? '>' : ''}${fmt.int(pfI)}`}</Label>
        </Fade>
        <Fade show={s.step >= 5}>
          <circle cx={lx(dcI)} cy={AX_Y} r={5} fill="var(--memory)" />
          <Label x={lx(dcI)} y={AX_Y - 10} anchor="middle" size={9.5} kind="memory" bold>{`decode: ${dcI.toFixed(1)}`}</Label>
        </Fade>
        <Label x={24} y={AX_Y + 34} tone={3} size={10} mono={false}>Left of the ridge the GPU is waiting for bytes; right of it, waiting for the math.</Label>
        <Label x={24} y={AX_Y + 50} tone={2} size={10} mono={false}>
          {`One prefill token costs ${((pfTime / tp) * 1e6).toFixed(0)} µs; one decode token costs ${(dcTime * 1e3).toFixed(1)} ms — ${Math.round(dcTime / (pfTime / tp))}× more.`}
        </Label>
        <Label x={24} y={AX_Y + 66} tone={3} size={10} mono={false}>
          {`In one decode step only ${(dcCompute * 1e6).toFixed(0)} µs (${(mathFrac * 100).toFixed(1)}%) is arithmetic. The rest is memory.`}
        </Label>
      </svg>
    </Stage>
  );
}
