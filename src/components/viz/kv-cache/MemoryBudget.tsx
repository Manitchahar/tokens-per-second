import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Fade, fmt } from '../lib/svg';
import { GPUS, MODELS, gpuById, modelById } from '../lib/perfModel';
import { kvPerToken, weightBytes, group } from './kvMath';

const STEPS = [
  'Everything must fit',
  'The three consumers',
  'Weights: fixed',
  'KV: 128 KB a token',
  'One conversation',
  'How many conversations',
  'The four levers',
];
const CAPTIONS = [
  'A GPU has a fixed amount of memory and a request must live entirely inside it. Out of memory is not a slow path; it is a crash.',
  'Three things want that memory: the model’s weights, the KV caches of every conversation in flight, and a workspace for activations and scratch.',
  'The weights are a constant, paid once per GPU. No amount of cleverness makes an 8B model in BF16 smaller than 16 GB.',
  'The KV cache is the only consumer that scales with your traffic: charged per user, per token, and never freed until the request ends.',
  'At 4,096 tokens of context, one Llama-3.1-8B conversation occupies 512 MB — a thirty-second of the model, for a single user.',
  'So the question is not whether the model fits but how many conversations fit in what is left. The answer is a division, and it is worth doing in your head.',
  'Four levers change that division: fewer KV heads, a smaller cache format, shorter contexts, or a smaller model. Everything else is scheduling.',
];

const BX = 24, BR = 624, BW = BR - BX, BY = 30, BH = 22;
const HEAD = 216, ROW = 20;
const gb = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${Math.round(b / 1e6)} MB`);

const LEVERS = (g: number): [string, string, string][] => [
  ['Fewer KV heads', `${g} query heads already share one key/value head.`, 'More sharing shrinks the cache again.'],
  ['A smaller cache format', 'FP8 or INT8 KV halves the cache bytes,', 'exactly, with no change to the weights.'],
  ['Shorter contexts', 'Linear and free: cap the window, or', 'summarize and drop the middle of history.'],
  ['A smaller model, or more GPUs', 'Quantize the weights, or shard the', 'model across cards. The other half of the bar.'],
];

export default function MemoryBudget() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [gpuId, setGpuId] = useState('h100');
  const [ctx, setCtx] = useState(4096);
  const [batch, setBatch] = useState(64);

  useEffect(() => {
    if (s.step === 2 || s.step === 3) setBatch(1);
    if (s.step >= 4) setCtx(4096);
  }, [s.step]);

  const model = modelById(modelId);
  const gpu = gpuById(gpuId);
  const w = weightBytes(model, 2);
  const per = kvPerToken(model, 2);
  const perSeq = ctx * per;
  const kv = batch * perSeq;
  const ws = 0.04 * gpu.mem;
  const avail = Math.max(0, (gpu.mem - w - ws) * 0.9);
  const maxSeq = Math.floor(avail / perSeq);
  const tokAll = Math.floor(avail / per);
  const fits = w + kv + ws <= gpu.mem;

  const fw = useTween(BW * Math.min(1, w / gpu.mem), 600);
  const fk = useTween(BW * Math.min(1, kv / gpu.mem), 600);
  const fr = useTween(BW * Math.min(1, ws / gpu.mem), 600);
  const ctxT = useTween(ctx, 600);
  const x1 = Math.min(BR, BX + fw);
  const x2 = Math.min(BR, x1 + fk);

  return (
    <Stage
      rootRef={ref}
      title="Memory math: weights, KV cache, and how many conversations fit"
      steps={STEPS.map((x) => x.label)}
      captions={CAPTIONS}
      legend={[['memory', 'weights'], ['memory', 'KV cache'], 'idle', ['error', 'out of memory']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Weights', value: `${(w / 1e9).toFixed(1)} GB`, kind: 'memory', note: 'fixed for the life of the GPU' },
          { label: 'KV cache now', value: gb(kv), kind: 'memory', note: `${fmt.int(batch)} × ${fmt.int(ctx)} × ${fmt.int(per / 1024)} KB` },
          { label: 'Conversations that fit', value: fmt.int(maxSeq), kind: maxSeq > 0 ? 'token' : 'error', note: `${fmt.int(ctx)} tokens each, 10% headroom` },
          { label: 'Context tokens per GPU', value: tokAll > 0 ? fmt.si(tokAll) : '0', kind: 'token', note: 'the number that predicts everything' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={MODELS.filter((m) => ['qwen05', 'llama8b', 'qwen32b', 'llama70b'].includes(m.id)).map((m) => ({ value: m.id, label: m.name.replace('Llama-3.1-', '').replace('Qwen2.5-', '') }))} onChange={setModelId} />
          <Segmented label="GPU" value={gpuId} options={GPUS.filter((g) => ['t4', 'a100', 'h100', 'h200'].includes(g.id)).map((g) => ({ value: g.id, label: g.id.toUpperCase() }))} onChange={setGpuId} />
          <Slider label="Context per conversation" value={ctx} min={256} max={131072} log onChange={setCtx} format={(v) => `${fmt.int(v)} tokens`} kind="memory" />
          <Slider label="Conversations in flight" value={batch} min={1} max={512} log onChange={setBatch} format={(v) => `${fmt.int(v)}`} kind="token" />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A GPU memory budget: weights, KV caches and workspace competing for one bar, the arithmetic behind each number, and either a five-model comparison or the four levers that change the answer.">
        <Label x={BX} y={20} tone={2} size={10.5} mono={false} bold>{`${gpu.name} · ${(gpu.mem / 1e9).toFixed(0)} GB of memory`}</Label>
        <Label x={BR} y={20} anchor="end" size={10} kind={fits ? 'token' : 'error'} bold>{fits ? 'fits' : 'OUT OF MEMORY'}</Label>

        <rect x={BX} y={BY} width={BW} height={BH} rx={2} fill="none" stroke="var(--rule-2)" />
        {s.step >= 1 && <Span x={BX} y={BY} w={x1 - BX} h={BH} kind="memory" label={`weights ${(w / 1e9).toFixed(1)} GB`} />}
        {s.step >= 3 && <Span x={x1} y={BY} w={x2 - x1} h={BH} kind="memory" dim label={`KV ${gb(kv)}`} labelTone="light" />}
        {s.step >= 1 && <Span x={x2} y={BY} w={Math.max(0, Math.min(BR, x2 + fr) - x2)} h={BH} kind="idle" label="workspace" labelTone="light" />}
        <Fade show={!fits}>
          <rect x={BX} y={BY - 3} width={BW} height={BH + 6} rx={3} fill="url(#tps-hatch-red)" stroke="var(--error)" />
          <text x={320} y={BY + 15} textAnchor="middle" className="v-label bold" style={{ fill: 'var(--error)', fontSize: 12 }}>
            {`needs ${((w + kv + ws) / 1e9).toFixed(0)} GB, the card has ${(gpu.mem / 1e9).toFixed(0)}`}
          </text>
        </Fade>
        <Label x={BX} y={BY + BH + 16} tone={3} size={9.5}>
          {s.step >= 1 ? `free after weights and workspace: ${(avail / 1e9).toFixed(1)} GB` : 'weights · KV cache · workspace · free'}
        </Label>
        <Label x={BR} y={BY + BH + 16} anchor="end" size={9.5} kind="token">{`= ${tokAll > 0 ? fmt.si(tokAll) : '0'} tokens of context in all`}</Label>

        <line x1={BX} x2={BR} y1={92} y2={92} className="v-rule faint" />
        <Label x={BX} y={108} tone={2} size={10.5} mono={false} bold>The arithmetic, every time</Label>
        {[
          ['weights', `params × 2 bytes`, `${(w / 1e9).toFixed(2)} GB`],
          ['KV / token', `2 × ${model.layers} × ${model.kvHeads} × ${model.headDim} × 2`, `${fmt.int(per / 1024)} KB`],
          ['1 request', `${fmt.int(ctx)} tokens × KV/token`, gb(perSeq)],
          ['in flight', `${fmt.int(batch)} requests`, gb(kv)],
        ].map(([k, f, v], i) => (
          <g key={k}>
            <Label x={BX} y={126 + i * 18} tone={2} size={10}>{k}</Label>
            <Label x={110} y={126 + i * 18} tone={3} size={9.5}>{f}</Label>
            <Label x={BR} y={126 + i * 18} anchor="end" tone={1} size={10} kind="memory">{v}</Label>
          </g>
        ))}

        {s.step === 5 ? (
          <g>
            <line x1={BX} x2={BR} y1={194} y2={194} className="v-rule faint" />
            <Label x={BX} y={204} tone={2} size={10.5} mono={false} bold>{`Same arithmetic, five models, on the ${gpu.id.toUpperCase()} at ${fmt.int(ctx)} tokens each`}</Label>
            <Label x={210} y={HEAD} anchor="middle" tone={3} size={9}>KB / token</Label>
            <Label x={350} y={HEAD} anchor="middle" tone={3} size={9}>per request</Label>
            <Label x={540} y={HEAD} anchor="middle" tone={3} size={9}>requests that fit</Label>
            {MODELS.map((m, i) => {
              const b = kvPerToken(m, 2);
              const room = Math.max(0, (gpu.mem - weightBytes(m, 2) - 0.04 * gpu.mem) * 0.9);
              const n = Math.floor(room / (ctx * b));
              return (
                <g key={m.id}>
                  <Label x={BX} y={HEAD + 18 + i * ROW} size={10} mono={false} tone={m.id === modelId ? 1 : 2}>{m.name}</Label>
                  <Label x={210} y={HEAD + 18 + i * ROW} anchor="middle" size={9.5} tone={2}>{fmt.int(b / 1024)}</Label>
                  <Label x={350} y={HEAD + 18 + i * ROW} anchor="middle" size={9.5} tone={2}>{gb(ctx * b)}</Label>
                  <Label x={540} y={HEAD + 18 + i * ROW} anchor="middle" size={9.5} kind={n > 0 ? 'token' : 'error'} bold={m.id === modelId}>{n > 0 ? fmt.int(n) : 'no room'}</Label>
                </g>
              );
            })}
          </g>
        ) : (
          <Fade show={s.step >= 6}>
            <g>
              <line x1={BX} x2={BR} y1={194} y2={194} className="v-rule faint" />
              {LEVERS(group(model)).map(([t, l1, l2], i) => {
                const x = BX + (i % 2) * 306;
                const y = 202 + Math.floor(i / 2) * 66;
                return (
                  <g key={t}>
                    <rect x={x} y={y} width={294} height={58} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1} />
                    <Label x={x + 12} y={y + 20} size={10.5} kind="memory" bold mono={false}>{t}</Label>
                    <Label x={x + 12} y={y + 38} tone={3} size={9} mono={false}>{l1}</Label>
                    <Label x={x + 12} y={y + 52} tone={3} size={9} mono={false}>{l2}</Label>
                  </g>
                );
              })}
            </g>
          </Fade>
        )}

        <Label x={BX} y={350} tone={2} size={10.5} mono={false}>
          {`${model.name}: ${(w / 1e9).toFixed(1)} GB of weights, ${fmt.int(per / 1024)} KB per token, ${tokAll > 0 ? fmt.si(tokAll) : 'no'} tokens of context per GPU.`}
        </Label>
        <Label x={BR} y={350} anchor="end" size={10} mono={false} tone={3}>{`${fmt.int(maxSeq)} seqs of ${fmt.int(ctxT)} tokens`}</Label>
      </svg>
    </Stage>
  );
}
