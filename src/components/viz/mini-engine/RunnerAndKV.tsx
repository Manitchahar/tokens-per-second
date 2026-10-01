import { useEffect, useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Span, Label, Arrow, Dim, Fade, fmt } from '../lib/svg';
import { modelById, gpuById, decodeStep, kvBytesPerToken } from '../lib/perfModel';

/**
 * One decode step, in and out of a runner you wrote yourself.
 * The point of every step: the weights are read once no matter the batch size, and
 * the only thing that grows with context is the KV read.
 */

const STEPS = [
  'Inputs: one token each',
  'Write K and V',
  'Attention reads the whole cache',
  'The MLP is most of the weights',
  'LM head and sample',
  'One step, N tokens out',
  'The only thing that matters',
];
const CAPTIONS = [
  'A decode step takes B sequences and gives each one token in. That is the whole interface: [B] ids in, [B] ids out.',
  'The new token produces a new key and a new value. Append them to that sequence’s cache and the cache is correct — no recomputation, ever.',
  'Attention reads the entire cache for every sequence, not just the new token. This is the part of the step that grows with context.',
  'Most of the model’s bytes are in the feed-forward blocks. Reading them once per step is the floor on decode latency.',
  'The last hidden state goes through the LM head into logits over the vocabulary, and the sampler turns logits into one token.',
  'Run that once and you have generated B tokens. The cost of the step barely moved as B grew — that is the whole of batching.',
  'Change the batch size or the context and watch the step time. It is flat in batch, and it grows linearly in context. Everything else is noise.',
];

const PRESET: Record<number, { batch: number; ctx: number }> = {
  0: { batch: 1, ctx: 1024 },
  1: { batch: 1, ctx: 1024 },
  2: { batch: 1, ctx: 1024 },
  3: { batch: 1, ctx: 1024 },
  4: { batch: 1, ctx: 1024 },
  5: { batch: 48, ctx: 1024 },
  6: { batch: 48, ctx: 1024 },
};

const X0 = 116;
const X1 = 616;
const TS = 0.025; // full width = 25 ms

export default function RunnerAndKV() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [gpuId, setGpuId] = useState('h100');
  const p0 = PRESET[s.step];
  const [batch, setBatch] = useState(p0.batch);
  const [ctx, setCtx] = useState(p0.ctx);
  useEffect(() => { setBatch(p0.batch); setCtx(p0.ctx); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const model = modelById(modelId);
  const gpu = gpuById(gpuId);
  const r = decodeStep({ model, gpu, batch, ctx });

  const px = (sec: number) => Math.min(X1 - X0, (sec / TS) * (X1 - X0));
  const g = useTween(r.t, 500);
  const gw = useTween(px(r.tWeights), 500);
  const gk = useTween(px(r.tKv), 500);
  const gc = useTween(px(r.tCompute), 500);

  const kvPer = kvBytesPerToken(model, 2);
  const shown = Math.min(batch, 40);
  const batchOn = s.step >= 5;

  // layer diagram
  const LX = 40;
  const LY = 150;
  const layerH = 7;
  const layerW = 66;

  return (
    <Stage
      rootRef={ref}
      title="One decode step through a runner you wrote"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Step time', value: fmt.ms(r.t), kind: 'text', note: `${fmt.int(r.perUser)} tok/s per user` },
          { label: 'Tokens out', value: `${fmt.int(batch)}`, kind: 'token', note: 'one per sequence, per step' },
          { label: 'Weight read', value: fmt.gb(r.weights), kind: 'memory', note: `always ${fmt.int(1000 * r.tWeights / r.t)}% independent of batch` },
          { label: 'KV read', value: fmt.gb(r.kvTotal), kind: 'memory', note: `grows with batch × context` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={[{ value: 'qwen05', label: '0.5B' }, { value: 'llama3b', label: '3B' }, { value: 'llama8b', label: '8B' }]} onChange={setModelId} />
          <Segmented label="GPU" value={gpuId} options={[{ value: 't4', label: 'T4' }, { value: 'a100', label: 'A100' }, { value: 'h100', label: 'H100' }]} onChange={setGpuId} />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A decode step through a hand-written model runner: a timeline of weight reads, KV cache reads and compute, a stack of transformer layers with a growing KV cache beside them, and a row of produced tokens.">
        {/* ── the step, as a timeline */}
        <Label x={X0} y={16} tone={3} size={10}>one decode step, scaled so 25 ms fills the width</Label>
        <text x={X0 - 10} y={42} textAnchor="end" className="v-track-label">Memory</text>
        <Span x={X0} y={32} w={gw} h={15} kind="memory" label="read weights" />
        <Fade show={s.step >= 2}>
          <Span x={X0 + gw + 1} y={32} w={gk} h={15} kind="memory" dim label="read KV" labelTone="light" />
        </Fade>
        <text x={X0 - 10} y={68} textAnchor="end" className="v-track-label">Math</text>
        <Span x={X0} y={58} w={Math.max(2, gc)} h={15} kind="compute" />
        <Fade show={s.step >= 5}>
          <Span x={X0 + Math.max(2, gc) + 1} y={58} w={Math.max(0, gk + gw - gc - 1)} h={15} kind="idle" />
        </Fade>
        <line x1={X0 + g} x2={X0 + g} y1={26} y2={80} stroke="var(--text)" />
        <Label x={X0 + g + 6} y={54} tone={1} size={10.5}>{fmt.ms(r.t)}</Label>
        <Label x={X1} y={16} anchor="end" tone={3} size={9.5}>{`${fmt.int(r.t * 1000)} ms · ${fmt.int(batch)} tokens`}</Label>

        {/* ── tokens out */}
        <text x={X0 - 10} y={104} textAnchor="end" className="v-track-label">Tokens</text>
        <Fade show={batchOn}>
          {Array.from({ length: shown }, (_, i) => (
            <rect key={i} x={X0 + i * 10} y={94} width={7.5} height={12} rx={1.5} fill="var(--token)" />
          ))}
          {batch > shown && <Label x={X0 + shown * 10 + 6} y={104} size={10.5} kind="token">+{fmt.int(batch - shown)} more</Label>}
          <Label x={X0} y={120} tone={3} size={9.5}>{`${fmt.int(batch)} sequences × 1 token = ${fmt.int(batch)} tokens from one step`}</Label>
        </Fade>
        <Fade show={!batchOn}>
          <Label x={X0} y={104} tone={3} size={10} mono={false}>one sequence, one token — the tutorial’s shape</Label>
        </Fade>

        {/* ── the model, one layer repeated */}
        <text x={X0 - 10} y={148} textAnchor="end" className="v-track-label">Model</text>
        <Label x={LX} y={LY - 8} tone={2} size={10.5} mono={false}>the runner walks these {model.layers} times</Label>
        {[0, 1, 2].map((i) => (
          <g key={i}>
            <rect x={LX + i * (layerW + 8)} y={LY} width={layerW} height={layerH * 3 + 6} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <rect x={LX + i * (layerW + 8) + 5} y={LY + 5} width={layerW - 10} height={layerH} rx={2} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={0.8} />
            <text x={LX + i * (layerW + 8) + layerW / 2} y={LY + 12} textAnchor="middle" className="v-label" style={{ fontSize: 7.5, fill: 'var(--text-3)' }}>attn</text>
            <rect x={LX + i * (layerW + 8) + 5} y={LY + 5 + layerH + 3} width={layerW - 10} height={layerH * 2} rx={2} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={0.8} />
            <text x={LX + i * (layerW + 8) + layerW / 2} y={LY + 12 + layerH + 3 + 7} textAnchor="middle" className="v-label" style={{ fontSize: 7.5, fill: 'var(--text-3)' }}>mlp · most weights</text>
          </g>
        ))}
        <Fade show={s.step >= 3}>
          <Dim x1={LX} x2={LX + 2 * (layerW + 8) + layerW} y={LY + 34} label="read once, whole model" kind="compute" />
        </Fade>

        {/* ── the KV cache, growing */}
        <text x={X0 - 10} y={228} textAnchor="end" className="v-track-label">KV cache</text>
        {(() => {
          const kx = 316, ky = 204, kw = 300;
          return (
            <>
              <Label x={kx} y={ky - 6} tone={2} size={10}>{`${fmt.int(ctx)} tokens × ${fmt.bytes(kvPer)} = ${fmt.bytes(ctx * kvPer)} per sequence`}</Label>
              <rect x={kx} y={ky} width={kw} height={20} rx={2} fill="none" stroke="var(--rule)" />
              <Fade show={s.step >= 1}>
                <Span x={kx + 1} y={ky + 1} w={kw - 2} h={18} kind="memory" dim opacity={0.9} />
                <rect x={kx + kw - 1 - Math.max(6, 200 / Math.max(ctx, 1))} y={ky + 1} width={Math.max(6, 200 / Math.max(ctx, 1))} height={18} rx={2} fill="var(--memory)" stroke="var(--memory)" />
                <Label x={kx + kw - 1 - Math.max(6, 200 / Math.max(ctx, 1)) - 4} y={ky + 14} anchor="end" size={9} kind="memory">new</Label>
              </Fade>
              <Fade show={s.step >= 2}>
                <Arrow x1={kx + kw / 2} y1={ky + 26} x2={kx + kw / 2} y2={ky + 42} kind="memory" />
                <Label x={kx + kw / 2 + 8} y={ky + 38} tone={2} size={10}>attention reads all of it, every step</Label>
              </Fade>
            </>
          );
        })()}

        {/* ── memory budget */}
        <text x={X0 - 10} y={304} textAnchor="end" className="v-track-label">HBM</text>
        <rect x={X0} y={292} width={X1 - X0} height={18} rx={2} fill="none" stroke="var(--rule)" />
        <Span x={X0} y={292} w={(X1 - X0) * (r.weights / gpu.mem)} h={18} kind="memory" label={`weights ${fmt.gb(r.weights)}`} />
        <Span x={X0 + (X1 - X0) * (r.weights / gpu.mem)} y={292} w={(X1 - X0) * (r.kvTotal / gpu.mem)} h={18} kind="memory" dim label={`KV ${fmt.gb(r.kvTotal)}`} labelTone="light" />
        <Fade show={!r.fits}>
          <rect x={X0} y={289} width={X1 - X0} height={24} rx={3} fill="url(#tps-hatch-red)" stroke="var(--error)" />
          <text x={(X0 + X1) / 2} y={305} textAnchor="middle" className="v-label bold" style={{ fill: 'var(--error)', fontSize: 11 }}>
            OOM: {(r.memUsed / 1e9).toFixed(0)} GB needed, {gpu.mem / 1e9} GB here
          </text>
        </Fade>
        <Label x={X0} y={324} tone={3} size={9.5}>{`weights ${fmt.gb(r.weights)} + ${fmt.int(batch)} × ${fmt.gb(ctx * kvPer)} KV = ${fmt.gb(r.memUsed)} of ${fmt.gb(gpu.mem)}`}</Label>

        <Fade show={s.step >= 6}>
          <Label x={X0} y={352} size={10.5} kind="token" bold>Step time is flat in batch. It grows linearly in context. Batch buys tokens per second; context buys nothing.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}