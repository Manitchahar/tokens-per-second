import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Fade, Dim, fmt } from '../lib/svg';
import { MODELS, gpuById, modelById, decodeStep, prefillTime } from '../lib/perfModel';
import { naiveForwards, cachedForwards } from './kvMath';

const gpu = gpuById('h100');
const MFU = 0.55;

const STEPS = [
  { label: 'One token at a time', t: 0 },
  { label: 'Only one token is new', t: 1 },
  { label: 'Naive: derive it all again', t: 1 },
  { label: 'The waste compounds', t: 63 },
  { label: 'Count the work', t: 127 },
  { label: 'Quadratic, not linear', t: 127 },
];
const CAPTIONS = [
  'An autoregressive model produces one token at a time. Each new token is appended to the sequence and fed back in.',
  'The important observation: the earlier tokens have not changed. Only the last one is new.',
  'A naive implementation re-runs the whole forward pass on the whole sequence every single step.',
  'By the last step it has pushed 639 tokens through the model to produce one, and 128 steps have repeated almost identical work.',
  'Counting token-forwards: 73,664 of them, against 640 that were actually needed. That is 99.1% waste.',
  'With a cache, the old tokens are read back from memory instead of recomputed, and the work becomes linear in the output length.',
];

const X0 = 100;
const X1 = 622;
const BW = X1 - X0;
const CX0 = 100, CX1 = 622, CY0 = 296, CY1 = 166;

export default function RecomputeWaste() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [p, setP] = useState(512);
  const [n, setN] = useState(128);
  const [t, setT] = useState(0);

  useEffect(() => {
    setT(STEPS[s.step].t);
    setN((v) => Math.max(v, STEPS[s.step].t + 1));
  }, [s.step]);

  const model = modelById(modelId);
  const total = p + n;
  const px = (tokens: number) => (tokens / total) * BW;

  const naive = naiveForwards(p, n);
  const cached = cachedForwards(p, n);
  const waste = 1 - cached / naive;
  const flopsPerForward = 2 * model.params;

  const naiveSeconds = (naive * flopsPerForward) / (gpu.flops * MFU);
  const cachedSeconds =
    prefillTime({ model, gpu, tokens: p }) +
    n * decodeStep({ model, gpu, batch: 1, ctx: p + n / 2 }).t;

  const barW = useTween(px(p + t), 600);
  const promptW = useTween(px(p), 600);

  const curve = useMemo(() => {
    const naivePts: string[] = [];
    const cachedPts: string[] = [];
    for (let i = 0; i <= 48; i++) {
      const ti = Math.round((i / 48) * n);
      const x = CX0 + (ti / n) * (CX1 - CX0);
      naivePts.push(`${i ? 'L' : 'M'}${x.toFixed(1)},${(CY0 - (naiveForwards(p, ti) / naive) * (CY0 - CY1)).toFixed(1)}`);
      cachedPts.push(`${i ? 'L' : 'M'}${x.toFixed(1)},${(CY0 - ((p + ti) / naive) * (CY0 - CY1)).toFixed(1)}`);
    }
    return { naive: naivePts.join(' '), cached: cachedPts.join(' ') };
  }, [p, n, naive]);

  const yCached = CY0 - ((p + n) / naive) * (CY0 - CY1);

  return (
    <Stage
      rootRef={ref}
      title="What the GPU actually does for every generated token"
      steps={STEPS.map((x) => x.label)}
      captions={CAPTIONS}
      legend={['error', 'memory', 'token', 'compute']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Work done, recomputing', value: `${fmt.int(naive)} tf`, kind: 'error', note: 'token-forwards × 2 × parameters' },
          { label: 'Work actually needed', value: `${fmt.int(cached)} tf`, kind: 'token', note: 'the prompt once, then one token a step' },
          { label: 'Share wasted', value: `${(waste * 100).toFixed(1)}%`, kind: 'error', note: `${(naive / cached).toFixed(0)}× more arithmetic than necessary` },
          { label: 'Wall clock, one H100', value: `${fmt.ms(naiveSeconds)} → ${fmt.ms(cachedSeconds)}`, kind: 'compute', note: `${(naiveSeconds / cachedSeconds).toFixed(1)}× — the wasted math runs on an idle machine` },
        ]} />
      }
      controls={
        <>
          <Slider label="Prompt length" value={p} min={64} max={4096} log onChange={setP} format={(v) => `${fmt.int(v)} tokens`} kind="memory" />
          <Slider label="Tokens to generate" value={n} min={8} max={1024} log onChange={setN} format={(v) => `${fmt.int(v)} tokens`} kind="token" />
          <Segmented
            label="Model"
            value={modelId}
            options={MODELS.filter((m) => ['llama8b', 'qwen32b', 'llama70b'].includes(m.id)).map((m) => ({ value: m.id, label: m.name.replace('Llama-3.1-', '').replace('Qwen2.5-', '') }))}
            onChange={setModelId}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="The sequence so far, one naive generation step that reprocesses every token, and a chart of cumulative token-forwards comparing recomputation with caching.">
        <Label x={X0} y={16} tone={3} size={10}>{`the sequence so far: ${fmt.int(p)} prompt + ${fmt.int(t)} generated`}</Label>
        <Label x={X1} y={16} anchor="end" size={10} kind="token" bold>one new token</Label>
        <line x1={X1} x2={X1} y1={19} y2={24} stroke="var(--token)" strokeWidth={1.2} />

        <Span x={X0} y={24} w={promptW} h={16} kind="error" dim label={`${fmt.int(p)} prompt`} labelTone="light" />
        <Span x={X0 + promptW} y={24} w={Math.max(0, BW - promptW)} h={16} kind="error" dim />
        <rect x={X1 - 3} y={24} width={3} height={16} fill="var(--token)" />
        <Fade show={s.step >= 1}>
          <Label x={X0} y={56} tone={2} size={10.5} mono={false}>
            {`The ${fmt.int(p + t - 1)} tokens before it have already been through the model, unchanged.`}
          </Label>
        </Fade>

        <Label x={X0} y={82} tone={3} size={10}>
          {s.step >= 5 ? 'one step with a cache: read the old keys and values back' : 'one naive step: push the whole sequence through again'}
        </Label>
        {s.step >= 5 ? (
          <g>
            <Span x={X0} y={88} w={Math.max(0, barW - 3)} h={16} kind="memory" dim label="read K and V" labelTone="light" />
            <rect x={X0 + Math.max(0, barW - 3)} y={88} width={3} height={16} fill="var(--token)" />
          </g>
        ) : (
          <g>
            <rect x={X0} y={88} width={BW} height={16} rx={2} fill="none" stroke="var(--rule-2)" strokeDasharray="3 3" />
            <Span x={X0} y={88} w={barW} h={16} kind="error" label={`recompute ${fmt.int(p + t)} tokens`} />
          </g>
        )}
        <Dim x1={X0} x2={X0 + barW} y={122} kind={s.step >= 5 ? 'memory' : 'error'} label={s.step >= 5 ? '1 new token' : 'recomputed'} />
        <Fade show={s.step >= 3}>
          <Label x={X0} y={142} tone={3} size={10} kind={s.step >= 5 ? 'memory' : 'error'}>
            {s.step >= 5
              ? `same answer, ${(naiveForwards(p, t + 1) / (p + t + 1)).toFixed(0)}× less arithmetic`
              : `and the next step does ${fmt.int(p + t + 1)} of them`}
          </Label>
        </Fade>

        <Label x={CX0} y={CY1 - 10} tone={2} size={10.5} mono={false} bold>Cumulative work, in token-forwards</Label>
        <line x1={CX0} x2={CX1} y1={CY0} y2={CY0} className="v-rule" />
        <line x1={CX0} x2={CX0} y1={CY1 - 4} y2={CY0} className="v-rule" />
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={CX0 - 4} x2={CX0} y1={CY0 - f * (CY0 - CY1)} y2={CY0 - f * (CY0 - CY1)} className="v-rule" />
            <Label x={CX0 - 8} y={CY0 - f * (CY0 - CY1) + 3.5} anchor="end" tone={3} size={9}>{f === 0 ? '0' : fmt.si(naive * f)}</Label>
          </g>
        ))}
        {[0, 0.5, 1].map((f) => (
          <Label key={`x${f}`} x={CX0 + f * (CX1 - CX0)} y={CY0 + 15} anchor="middle" tone={3} size={9}>{fmt.int(n * f)}</Label>
        ))}
        <Label x={CX1} y={CY0 + 32} anchor="end" tone={3} size={10} mono={false}>tokens generated →</Label>
        <path d={curve.naive} fill="none" stroke="var(--error)" strokeWidth={2} />
        <Fade show={s.step >= 5}>
          <path d={curve.cached} fill="none" stroke="var(--memory)" strokeWidth={2} />
          <circle cx={CX1} cy={yCached} r={3.5} fill="var(--memory)" />
        </Fade>
        <Fade show={s.step >= 4}>
          <Label x={CX1 - 4} y={CY1 + 12} anchor="end" size={9.5} kind="error">{`recompute: ${fmt.int(naive)}`}</Label>
        </Fade>
        <Fade show={s.step >= 5}>
          <Label x={CX1 - 4} y={yCached - 8} anchor="end" size={9.5} kind="memory">{`cache: ${fmt.int(cached)}`}</Label>
        </Fade>
        <Label x={CX0} y={CY0 + 52} tone={2} size={10.5} mono={false}>
          {`${model.name}: ${(2 * model.params * total / 1e12).toFixed(1)} TFLOP to prefill, then ${(flopsPerForward / 1e9).toFixed(1)} GFLOP per decode step.`}
        </Label>
      </svg>
    </Stage>
  );
}
