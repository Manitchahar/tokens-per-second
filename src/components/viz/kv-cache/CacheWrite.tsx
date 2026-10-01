import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Fade, fmt } from '../lib/svg';
import { MODELS, modelById } from '../lib/perfModel';
import { kvPerToken } from './kvMath';

const STEPS = [
  { label: 'One buffer, allocated once', t: 0 },
  { label: 'Prefill fills the first rows', t: 128 },
  { label: 'Decode writes row t', t: 128 },
  { label: 'Write is O(1), read is O(t)', t: 512 },
  { label: 'The trap: torch.cat', t: 1024 },
  { label: 'Is it actually correct?', t: 1024 },
  { label: 'And the part that still hurts', t: 4096 },
];
const CAPTIONS = [
  'The whole cache is one flat tensor per layer, shaped [key heads, max length, head dim] for K and again for V. Allocate it once, when the request starts, and never move it.',
  'Prefill writes the prompt’s rows in a single copy — 128 here, all contiguous. After it, the buffer has a valid prefix and a tail of zeros.',
  'Every decode step computes K and V for exactly one new token and writes one row, at the current length. The buffer never moves, so nothing is ever copied twice.',
  'Writing is one row. Reading is every row that exists. The write is O(1); the read is O(t) and grows for the whole life of the request.',
  'Appending with torch.cat looks identical and is quadratic: it allocates a fresh tensor and copies the entire history on every step.',
  'Compare against the same generator with the cache switched off. Same tokens, and the logits differ only by the order in which floating-point sums were added.',
  'A static buffer is mostly empty. At 8,192 rows with 1,024 used, 87% of the memory you reserved is zeros — and that waste is exactly why paged attention exists.',
];

const ROWS = 16, RX = 60, RW = 186, RY = 56, RH = 15;
const PX0 = 282, PW = 342;
const gb = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${Math.round(b / 1e6)} MB`);

export default function CacheWrite() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [t, setT] = useState(128);
  const [maxLen, setMaxLen] = useState(8192);

  useEffect(() => { setT(STEPS[s.step].t); }, [s.step]);

  const model = modelById(modelId);
  const per = kvPerToken(model, 2);
  const drawn = Math.min(ROWS, t);
  const cursor = drawn - 1;
  const readBytes = t * per;
  const catRows = (t * (t + 1)) / 2;
  const fullness = Math.min(1, t / maxLen);
  const tweenT = useTween(t, 600);
  const tweenFill = useTween(fullness, 600);

  return (
    <Stage
      rootRef={ref}
      title="Build a KV cache: one buffer, one cursor, one row per token"
      steps={STEPS.map((x) => x.label)}
      captions={CAPTIONS}
      legend={['memory', 'token', 'error', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Bytes per token', value: `${fmt.int(per / 1024)} KB`, kind: 'memory', note: `${model.layers} layers × ${model.kvHeads} KV heads × ${model.headDim} dims × 2 × 2 B` },
          { label: 'Written per step', value: `${fmt.int(per / 1024)} KB`, kind: 'token', note: 'one row, always' },
          { label: 'Read per step', value: gb(readBytes), kind: 'memory', note: `${fmt.int(t)} rows, growing for ever` },
          { label: 'Buffer allocated', value: gb(maxLen * per), kind: fullness < 0.5 ? 'idle' : 'memory', note: `${maxLen} rows, ${(fullness * 100).toFixed(0)}% full` },
        ]} />
      }
      controls={
        <>
          <Slider label="Context length so far (prompt + generated)" value={t} min={16} max={8192} log onChange={setT} format={(v) => `${fmt.int(v)} rows`} kind="memory" />
          <Segmented label="Rows preallocated (MAXLEN)" value={maxLen} options={[{ value: 1024, label: '1,024' }, { value: 4096, label: '4,096' }, { value: 8192, label: '8,192' }]} onChange={setMaxLen} />
          <Segmented label="Model" value={modelId} options={MODELS.filter((m) => ['qwen05', 'llama8b', 'llama70b'].includes(m.id)).map((m) => ({ value: m.id, label: m.name.replace('Llama-3.1-', '').replace('Qwen2.5-', '') }))} onChange={setModelId} />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A preallocated KV buffer drawn as a column of token rows with a write cursor, beside its shapes, the bytes written and read each step, the cost of torch.cat, and how much of the buffer is empty.">
        <Label x={24} y={20} tone={3} size={10}>{`${model.name} · one layer: ${model.kvHeads} heads × ${maxLen} rows × ${model.headDim} dims`}</Label>
        <Label x={624} y={20} anchor="end" size={10} tone={3}>{`whole cache = ${gb(maxLen * per)}`}</Label>

        {Array.from({ length: ROWS }, (_, i) => {
          const filled = i < drawn;
          const isCursor = filled && i === cursor && s.step >= 2;
          return (
            <g key={i}>
              <rect x={RX} y={RY + i * RH} width={RW} height={RH - 2} rx={1.5} className="v-cell"
                fill={filled ? (isCursor ? 'var(--token)' : 'var(--memory-dim)') : 'transparent'}
                stroke={filled ? (isCursor ? 'var(--token)' : 'var(--memory)') : 'var(--rule-2)'}
                strokeWidth={isCursor ? 1.8 : 0.8} strokeDasharray={filled ? undefined : '2 3'} />
              {filled && <text x={RX + 8} y={RY + i * RH + 9} className="v-label" style={{ fontSize: 8, fill: isCursor ? '#0e1116' : 'var(--text-3)' }}>{`K,V  row ${i}`}</text>}
            </g>
          );
        })}
        {s.step >= 2 && drawn > 0 && (
          <g>
            <line x1={RX} x2={RX + RW + 6} y1={RY + cursor * RH + 6.5} y2={RY + cursor * RH + 6.5} stroke="var(--token)" strokeWidth={1.4} />
            <Label x={RX + RW + 6} y={44} anchor="end" size={9.5} kind="token" bold>{`write row t = ${fmt.int(tweenT)}`}</Label>
          </g>
        )}
        <Label x={RX - 8} y={RY + 9} anchor="end" tone={3} size={9}>{drawn > 0 ? 'row 0' : 'rows'}</Label>
        {drawn > 0 && (
          <Label x={RX - 8} y={RY + cursor * RH + 9} anchor="end" size={9} kind={s.step >= 2 ? 'token' : 'memory'} bold>{`row ${cursor}`}</Label>
        )}
        <Label x={RX - 8} y={RY + ROWS * RH} anchor="end" tone={3} size={9}>MAXLEN</Label>
        <Label x={RX} y={RY + ROWS * RH + 18} tone={3} size={9.5}>{`showing ${ROWS} of ${maxLen} rows`}</Label>
        <Fade show={s.step >= 6}>
          <Label x={RX} y={RY + ROWS * RH + 38} size={10} kind="error" bold mono={false}>{`${((1 - fullness) * 100).toFixed(0)}% of the buffer is zeros`}</Label>
        </Fade>

        <line x1={PX0 - 14} x2={PX0 - 14} y1={44} y2={352} className="v-rule faint" />

        <Label x={PX0} y={60} tone={2} size={9.5}>{`K[ℓ]  [${model.kvHeads}, MAXLEN, ${model.headDim}]  BF16, zeroed once`}</Label>
        <Label x={PX0} y={76} tone={2} size={9.5}>{`V[ℓ]  the same shape, one flat allocation`}</Label>
        <Label x={PX0} y={92} tone={3} size={9.5}>{`× ${model.layers} layers = ${gb(maxLen * per)} at MAXLEN ${maxLen}`}</Label>

        <Fade show={s.step >= 2}>
          <g>
            <line x1={PX0} x2={624} y1={106} y2={106} className="v-rule faint" />
            <Label x={PX0} y={124} size={10} kind="token" bold mono={false}>What one decode step does</Label>
            <Label x={PX0} y={142} tone={2} size={9.5}>{'1.  x → W_K, W_V      (one token)'}</Label>
            <Label x={PX0} y={158} tone={2} size={9.5}>{'2.  RoPE the key, then write at index t'}</Label>
            <Label x={PX0} y={174} tone={2} size={9.5}>{'3.  attend over rows 0 … t'}</Label>
          </g>
        </Fade>

        <Fade show={s.step >= 3}>
          {s.step === 4 ? (
            <g>
              <line x1={PX0} x2={624} y1={196} y2={196} className="v-rule faint" />
              <rect x={PX0} y={204} width={PW} height={62} rx={4} fill="url(#tps-hatch-red)" stroke="var(--error)" />
              <Label x={PX0 + 10} y={222} size={10.5} kind="error" bold mono={false}>torch.cat on every step</Label>
              <Label x={PX0 + 10} y={238} tone={2} size={9.5}>{`copies 1 + 2 + … + t rows = ${fmt.si(catRows)} rows`}</Label>
              <Label x={PX0 + 10} y={256} size={9.5} kind="error" bold>{`${fmt.int(catRows / Math.max(1, t))}× more traffic than it needed`}</Label>
              <Label x={PX0} y={286} tone={3} size={9.5}>The result is the same tensor. The copy is not.</Label>
            </g>
          ) : s.step === 6 ? (
            <g>
              <line x1={PX0} x2={624} y1={196} y2={196} className="v-rule faint" />
              <Label x={PX0} y={212} tone={2} size={9.5} mono={false}>How much of the buffer you reserved is actually used</Label>
              <Span x={PX0} y={220} w={PW} h={16} kind="memory" dim ghost label={`${fmt.int(t)} rows used`} labelTone="light" />
              <Span x={PX0} y={220} w={PW * tweenFill} h={16} kind="memory" />
              <Label x={PX0} y={254} tone={2} size={9.5}>{`${fmt.int(maxLen - t)} rows of reserved-but-empty memory`}</Label>
              <Label x={PX0} y={272} tone={3} size={9.5}>{`= ${gb((maxLen - t) * per)} wasted for one request`}</Label>
              <Label x={PX0} y={290} size={9.5} kind="error" bold>Paged attention (Module 08) exists to fix this.</Label>
            </g>
          ) : (
            <g>
              <line x1={PX0} x2={624} y1={196} y2={196} className="v-rule faint" />
              <Label x={PX0} y={206} tone={2} size={9.5}>write this step</Label>
              <Span x={PX0} y={210} w={Math.max(3, PW / Math.max(1, t))} h={12} kind="token" />
              <Label x={PX0} y={240} tone={2} size={9.5}>{`1 row = ${fmt.int(per / 1024)} KB, and it never changes`}</Label>
              <Label x={PX0} y={258} tone={2} size={9.5}>read this step</Label>
              <Span x={PX0} y={262} w={PW} h={12} kind="memory" label={`${fmt.int(t)} rows = ${gb(readBytes)}`} />
              <Label x={PX0} y={290} size={9.5} kind="memory" bold>{`${fmt.int(t)}× more read than write, and rising`}</Label>
            </g>
          )}
        </Fade>

        <Fade show={s.step === 5}>
          <g>
            <line x1={PX0} x2={624} y1={302} y2={302} className="v-rule faint" />
            <Label x={PX0} y={318} size={9.5} tone={2}>{`Same tokens as use_cache=False. The logits`}</Label>
            <Label x={PX0} y={334} size={9.5} tone={2}>{`differ only by the order in which floating-point`}</Label>
            <Label x={PX0} y={350} size={9.5} tone={2}>{`sums were added — not by a logic error.`}</Label>
          </g>
        </Fade>

        <Fade show={s.step >= 6}>
          <g>
            <line x1={PX0} x2={624} y1={302} y2={302} className="v-rule faint" />
            <Label x={PX0} y={318} size={9.5} tone={2}>{`Allocate once. Write at t. Read 0 … t.`}</Label>
            <Label x={PX0} y={334} size={9.5} tone={2}>{`Store keys after RoPE or positions are lost.`}</Label>
            <Label x={PX0} y={350} size={9.5} tone={2}>{`Never cat: the copy is the whole cost.`}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
