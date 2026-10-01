import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Fade, fmt } from '../lib/svg';
import { MODELS, modelById } from '../lib/perfModel';
import { kvPerToken, weightBytes, group } from './kvMath';

const STEPS = [
  { label: 'What attention needs', ctx: 0 },
  { label: 'Keys and values, per layer', ctx: 0 },
  { label: 'Store them: the cache', ctx: 512 },
  { label: 'One step writes one row', ctx: 512 },
  { label: 'One step reads them all back', ctx: 512 },
  { label: 'The bytes it costs', ctx: 512 },
  { label: 'It grows as you write', ctx: 65536 },
];
const CAPTIONS = [
  'To predict the next token the model needs a key and a value for every token it has ever seen, including the one it just produced.',
  'Each transformer layer keeps its own: 8 key heads of 128 numbers and 8 value heads of 128 numbers, for Llama-3.1-8B.',
  'Those 4,096 numbers per layer per token are written into a buffer instead of being thrown away. That buffer is the KV cache.',
  'A decode step computes keys and values for exactly one new token, so it writes exactly one row, at the current length.',
  'Attention then reads every row that exists — the whole cache — to compute that one token’s output.',
  'Weights are fixed at 16.1 GB per step. The cache adds to that, and it starts small: 512 tokens is only 64 MB.',
  'At 64K tokens the cache is 8 GB, over a third of everything the step reads. Past about 122K tokens it outweighs the model itself.',
];

const ROWS = 9;
const CELL = 26;
const GAP = 3;
const KX = 54, VX = 310;
const RX0 = KX - 8;
const GY = 92, RH = 17;
const BW = 520, BX = 96;
const BY = GY + ROWS * RH + 30;

const mb = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${Math.round(b / 1e6)} MB`);

export default function KVCache() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [kvB, setKvB] = useState(2);
  const [ctx, setCtx] = useState(512);

  useEffect(() => { setCtx(STEPS[s.step].ctx); }, [s.step]);

  const model = modelById(modelId);
  const per = kvPerToken(model, kvB);
  const w = weightBytes(model, 2);
  const kvTotal = ctx * per;
  const share = kvTotal / (w + kvTotal);
  const rows = Math.min(ROWS, ctx);
  const cols = Math.min(8, model.kvHeads);
  const gridW = cols * (CELL + GAP) - GAP;

  const barW = useTween(BW * (1 - share), 600);
  const ctxT = useTween(ctx, 600);

  const showShape = s.step >= 1;
  const showFill = s.step >= 2;
  const showWrite = s.step >= 3;
  const showRead = s.step >= 4;
  const showBytes = s.step >= 5;
  const newest = rows - 1;

  return (
    <Stage
      rootRef={ref}
      title="The KV cache: one row written, all rows read, every step"
      steps={STEPS.map((x) => x.label)}
      captions={CAPTIONS}
      legend={['memory', 'token']}
      {...s}
      readout={
        <Readout items={[
          { label: 'KV bytes per token', value: `${fmt.int(per / 1024)} KB`, kind: 'memory', note: `${model.layers} layers × ${model.kvHeads} KV heads × ${model.headDim} dims × 2 × ${kvB} B` },
          { label: 'Cache at this context', value: mb(kvTotal), kind: 'memory', note: `${fmt.int(ctxT)} tokens of context` },
          { label: 'Weights read per step', value: `${(w / 1e9).toFixed(1)} GB`, kind: 'memory', note: 'fixed, whatever the context' },
          { label: 'Cache share of traffic', value: `${(share * 100).toFixed(1)}%`, kind: share > 0.5 ? 'error' : 'memory', note: share > 0.5 ? 'the cache is now the bigger read' : 'still dwarfed by the weights' },
        ]} />
      }
      controls={
        <>
          <Slider label="Context length (prompt + generated)" value={ctx} min={128} max={131072} log onChange={setCtx} format={(v) => `${fmt.int(v)} tokens`} kind="memory" />
          <Segmented
            label="Model"
            value={modelId}
            options={MODELS.filter((m) => ['qwen05', 'llama8b', 'llama70b'].includes(m.id)).map((m) => ({ value: m.id, label: m.name.replace('Llama-3.1-', '').replace('Qwen2.5-', '') }))}
            onChange={setModelId}
          />
          <Segmented label="KV cache format" value={kvB} options={[{ value: 2, label: 'BF16 · 2 B' }, { value: 1, label: 'FP8 · 1 B' }]} onChange={setKvB} />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A new token arriving, the keys and values it needs already stored in the cache, one new row being written, and the bytes the cache adds to each decode step.">
        <Label x={24} y={16} tone={3} size={10}>{`${model.name}, one decode step`}</Label>
        {Array.from({ length: 9 }, (_, i) => (
          <rect key={i} x={24 + i * 26} y={24} width={20} height={16} rx={2} className="v-cell"
            fill={ctx > i ? 'var(--memory-dim)' : 'transparent'} stroke="var(--rule-2)" strokeWidth={0.8} strokeDasharray={ctx > i ? undefined : '2 3'} />
        ))}
        <rect x={24 + 9 * 26} y={24} width={22} height={16} rx={2} fill="var(--token)" />
        <Label x={624} y={36} anchor="end" size={10} kind="token" bold>one new token: its own K and V are computed now</Label>

        <Label x={24} y={80} tone={2} size={10.5} mono={false} bold>one layer</Label>
        <Label x={KX + gridW / 2} y={80} anchor="middle" tone={3} size={9.5}>{`K · ${model.kvHeads} heads × ${model.headDim} dims`}</Label>
        <Label x={VX + gridW / 2} y={80} anchor="middle" tone={3} size={9.5}>{`V · ${model.kvHeads} heads × ${model.headDim} dims`}</Label>

        {[0, 1].map((half) => {
          const gx = (half === 0 ? KX : VX) + (cols * (CELL + GAP) - GAP - gridW) / 2;
          return (
            <g key={half}>
              {Array.from({ length: cols }, (_, c) => (
                <rect key={`e${c}`} x={gx + c * (CELL + GAP)} y={GY} width={CELL} height={ROWS * RH - GAP} rx={2}
                  fill="none" stroke="var(--rule-2)" strokeWidth={0.8} strokeDasharray="3 3" />
              ))}
              {Array.from({ length: ROWS }, (_, r) => {
                const filled = r < rows;
                const isNew = filled && r === newest && showWrite;
                return (
                  <rect key={r} x={gx} y={GY + r * RH} width={gridW} height={RH - GAP} rx={1.5} className="v-cell"
                    fill={filled && showFill ? (isNew ? 'var(--token)' : showRead ? 'var(--memory)' : 'var(--memory-dim)') : 'transparent'}
                    stroke={filled && showFill ? (isNew ? 'var(--token)' : 'var(--memory)') : 'var(--rule-2)'}
                    strokeWidth={isNew ? 1.8 : 0.8} strokeDasharray={filled && showFill ? undefined : '2 3'} />
                );
              })}
              {showWrite && rows > 0 && (
                <rect x={gx - 5} y={GY + newest * RH - 1} width={gridW + 10} height={RH - GAP + 2} rx={2}
                  fill="none" stroke="var(--token)" strokeWidth={1.4} className="v-fade" />
              )}
            </g>
          );
        })}

        <Label x={RX0} y={GY + 9} anchor="end" tone={3} size={9}>{showFill ? 'row 0' : 'rows'}</Label>
        {showFill && rows > 0 && (
          <Label x={RX0} y={GY + newest * RH + 9} anchor="end" size={9} kind={showWrite ? 'token' : 'memory'} bold>{`row ${newest}`}</Label>
        )}
        {showShape && (
          <Label x={24} y={GY + ROWS * RH + 12} tone={2} size={9.5}>{`one row = one token = ${fmt.int(per / model.layers)} bytes in this layer`}</Label>
        )}
        {ctx > ROWS && (
          <Label x={VX + gridW / 2} y={GY + ROWS * RH + 12} anchor="middle" tone={3} size={9}>{`… ${fmt.int(ctx)} rows in all`}</Label>
        )}
        {showWrite && rows > 0 && (
          <Label x={624} y={GY + newest * RH + 9} anchor="end" size={9.5} kind="token" bold>write row t</Label>
        )}
        {showRead && (
          <Label x={624} y={GY + 22} anchor="end" size={9.5} kind="memory" bold>read all of it</Label>
        )}
        {showRead && (
          <Label x={624} y={GY + 35} anchor="end" tone={3} size={9}>{mb(ctx * per)}</Label>
        )}

        <Fade show={showBytes}>
          <line x1={24} x2={624} y1={BY} y2={BY} className="v-rule faint" />
          <Label x={24} y={BY + 20} tone={2} size={10.5} mono={false} bold>Bytes one step must read from HBM</Label>
          <text x={BX - 10} y={BY + 42} textAnchor="end" className="v-track-label">HBM</text>
          <Span x={BX} y={BY + 30} w={barW} h={16} kind="memory" label={`weights ${(w / 1e9).toFixed(1)} GB`} />
          <Span x={BX + barW + 1} y={BY + 30} w={Math.max(0, BW - barW - 1)} h={16} kind="memory" dim label={`cache ${mb(kvTotal)}`} labelTone="light" />
          <Label x={24} y={BY + 64} tone={2} size={10.5} mono={false}>
            {`Every token adds ${fmt.int(per / 1024)} KB: at ${fmt.int(ctxT)} tokens the cache is ${(share * 100).toFixed(0)}% of what a step reads.`}
          </Label>
          <Label x={24} y={BY + 80} tone={3} size={10} mono={false}>
            {`GQA: ${group(model)} query heads share one key/value head, so the cache is ${group(model)}× smaller than full attention.`}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
