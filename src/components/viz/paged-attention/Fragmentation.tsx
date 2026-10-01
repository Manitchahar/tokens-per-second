import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';

// Llama-3.1-8B, BF16: 2 (K and V) x 32 layers x 8 KV heads x 128 dim x 2 bytes
const KV_PER_TOKEN = 131072; // bytes per token of context, all layers

// Four requests, fixed for the whole visual.
// `prompt` is what prefill fills; `gen` is what the request actually goes on to generate.
interface Seq { name: string; prompt: number; gen: number }

const SEQS: Seq[] = [
  { name: 'A', prompt: 40, gen: 40 },
  { name: 'B', prompt: 20, gen: 40 },
  { name: 'C', prompt: 56, gen: 128 },
  { name: 'D', prompt: 12, gen: 20 },
];

const SUM_PROMPT = SEQS.reduce((a, q) => a + q.prompt, 0); // 128
const SUM_FINAL = SEQS.reduce((a, q) => a + q.prompt + q.gen, 0); // 356

// The new request in step 6: a long one that wants a lot of headroom.
const NEW_TOKENS = 528;

const MB = (tokens: number) => ((tokens * KV_PER_TOKEN) / 1e6).toFixed(1);

const STEPS = [
  'Four requests arrive',
  'Reserve prompt + max_new_tokens',
  'At arrival, the gap is worst',
  'They decode, the gap closes',
  'What is wasted at the end',
  'Holes, not blocks',
  'A long request cannot start',
  'The rule',
];

const CAPTIONS = [
  'Four users send requests at once. Their prompts are 40, 20, 56 and 12 tokens. None of them knows how long its own answer will be.',
  'A contiguous allocator must reserve one flat buffer per request, sized for prompt plus the maximum the client could ask for. That maximum is a guess, and it is the whole problem.',
  'Immediately after admission the pool holds 640 reserved tokens and only 128 of them hold data. Four fifths of the memory in flight is empty.',
  'Each request then decodes, and the gap closes as real tokens fill it. The gap never closes completely, because no engine can guess a length in advance.',
  'At the end, 640 tokens are still reserved and only 356 hold useful data: 284 tokens, about 37 MB, of a GPU that is serving nobody.',
  'When requests finish, their buffers come back whole and differently sized: 168, 148 and 140 tokens. That is 22.0, 19.4 and 18.4 MB in three holes.',
  'A new request reserving 528 tokens needs 69.2 MB contiguously. The pool has 59.8 MB free and no hole that big. Out of memory, with memory to spare.',
  'Never reserve what you have not been asked for. Reserve in small fixed units and grow on demand — which is exactly what the next lesson builds.',
];

const W = 640;
const BAR_X = 92;
const BAR_W = 476;
const MAXV = 720; // tokens at full bar width

const px = (tokens: number) => (tokens / MAXV) * BAR_W;

export default function Fragmentation() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  // max_new_tokens: the ceiling each request reserves headroom for.
  const [maxNew, setMaxNew] = useState(128);

  const reserved = SEQS.map((q) => q.prompt + maxNew);
  const reservedSum = reserved.reduce((a, b) => a + b, 0);
  const wasteAtArrival = reservedSum - SUM_PROMPT;
  const wasteAtEnd = reservedSum - SUM_FINAL;

  const g = useTweens(
    {
      rA: px(reserved[0]), rB: px(reserved[1]), rC: px(reserved[2]), rD: px(reserved[3]),
      uA: px(SEQS[0].prompt), uB: px(SEQS[1].prompt), uC: px(SEQS[2].prompt), uD: px(SEQS[3].prompt),
    },
    600,
  );

  const rows = useMemo(
    () => SEQS.map((q, i) => ({ ...q, y: 46 + i * 30, res: reserved[i] })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [maxNew],
  );

  const res = [g.rA, g.rB, g.rC, g.rD];
  const used = [g.uA, g.uB, g.uC, g.uD];

  // the arena after A, B and D finish. C (the long one) is still decoding.
  const cRes = reserved[2];
  const cUsed = SEQS[2].prompt + SEQS[2].gen;
  const holes = [reserved[0], reserved[1], reserved[3]];
  const holeSum = holes.reduce((a, b) => a + b, 0);
  const bigHole = Math.max(...holes);

  return (
    <Stage
      rootRef={ref}
      title="Contiguous KV memory: reserving is not the same as using"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'KV cache holding data'],
        ['error', 'reserved but empty'],
        ['idle', 'free, in the wrong shape'],
        ['text', 'still to be generated'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Reserved, 4 requests', value: `${fmt.int(reservedSum)} tok`, kind: 'memory', note: `${MB(reservedSum)} MB held for four users` },
          { label: 'Useful at the end', value: `${fmt.int(SUM_FINAL)} tok`, kind: 'token', note: `${MB(SUM_FINAL)} MB actually holding data` },
          { label: 'Wasted', value: `${((wasteAtEnd / reservedSum) * 100).toFixed(0)}%`, kind: 'error', note: `${MB(wasteAtEnd)} MB serving nobody` },
          { label: 'Free, unusable', value: `${MB(holeSum)} MB`, kind: 'idle', note: `largest single hole ${MB(bigHole)} MB` },
        ]} />
      }
      controls={
        <Slider
          label="max_new_tokens — the headroom each request reserves"
          value={maxNew}
          min={32}
          max={256}
          step={16}
          onChange={setMaxNew}
          format={(v) => `${v} tokens`}
          kind="memory"
        />
      }
    >
      <svg viewBox={`0 0 ${W} 372`} role="img" aria-label="Four requests each reserving one flat KV buffer sized for its prompt plus a maximum new-token allowance, with the reserved-but-empty part in red, then the same pool after three requests finish leaving three differently sized holes that cannot serve one long contiguous request.">
        <Label x={BAR_X} y={22} tone={2} size={10.5} mono={false} bold>One KV pool · {maxNew} extra tokens of headroom reserved per request</Label>

        {rows.map((r, i) => (
          <g key={r.name}>
            <Label x={BAR_X - 12} y={r.y + 13} anchor="end" tone={1} size={11} bold>{r.name}</Label>
            {/* the whole reservation */}
            <Span x={BAR_X} y={r.y} w={res[i]} h={18} kind="error" dim labelTone="light" />
            {/* the prompt, filled during prefill */}
            <Span x={BAR_X} y={r.y} w={Math.min(used[i], res[i])} h={18} kind="memory" label={`${r.prompt}`} />
            {/* what it will actually generate, revealed later */}
            <Fade show={s.step >= 3}>
              <Span
                x={BAR_X + Math.min(used[i], res[i])}
                y={r.y}
                w={Math.max(0, Math.min(res[i] - used[i], px(r.gen)))}
                h={18}
                kind="token"
              />
            </Fade>
            <Label x={BAR_X + res[i] + 6} y={r.y + 13} tone={3} size={9}>{`${MB(r.res)} MB`}</Label>
          </g>
        ))}

        <Dim x1={BAR_X} x2={BAR_X + px(SUM_PROMPT)} y={178} label={`${fmt.int(SUM_PROMPT)} tok = ${MB(SUM_PROMPT)} MB in use`} kind="memory" />
        <Dim x1={BAR_X + px(SUM_PROMPT)} x2={BAR_X + px(reservedSum)} y={178} label={`${fmt.int(wasteAtArrival)} tok empty = ${MB(wasteAtArrival)} MB`} kind="error" />

        {/* step 3/4: the same comparison after decoding */}
        <Fade show={s.step >= 4}>
          <Dim x1={BAR_X + px(SUM_FINAL)} x2={BAR_X + px(reservedSum)} y={200} label={`${fmt.int(wasteAtEnd)} tok = ${MB(wasteAtEnd)} MB wasted forever`} kind="error" />
          <Label x={BAR_X} y={222} tone={1} size={10.5} bold kind="error">
            {`${((wasteAtEnd / reservedSum) * 100).toFixed(0)}% of the KV cache is serving nobody`}
          </Label>
        </Fade>

        {/* the arena after three requests finish */}
        <Fade show={s.step >= 5}>
          <Label x={BAR_X} y={248} tone={2} size={10.5} mono={false} bold>Later: A, B and D finish and hand their buffers back</Label>
          <rect x={BAR_X} y={256} width={px(reservedSum)} height={20} rx={2} fill="none" stroke="var(--rule-2)" />
          <Span x={BAR_X} y={256} w={px(cRes)} h={20} kind="memory" label="C still decoding" />
          {(() => {
            let x = px(cRes);
            return holes.map((h, i) => {
              const el = (
                <g key={i}>
                  <Span x={BAR_X + x} y={256} w={px(h)} h={20} kind="idle" />
                  <Label x={BAR_X + x + px(h) / 2} y={270} anchor="middle" tone={3} size={9}>{`${MB(h)}`}</Label>
                </g>
              );
              x += px(h);
              return el;
            });
          })()}
          <Label x={BAR_X + px(cRes)} y={290} tone={3} size={9}>{`free in total ${MB(holeSum)} MB · largest hole ${MB(bigHole)} MB`}</Label>
        </Fade>

        {/* a new request that cannot start */}
        <Fade show={s.step >= 6}>
          <rect x={BAR_X} y={302} width={px(NEW_TOKENS)} height={20} rx={2} fill="none" stroke="var(--error)" strokeDasharray="4 3" />
          <Label x={BAR_X + 4} y={316} tone={3} size={9.5}>{`a new request reserving ${NEW_TOKENS} tokens = ${MB(NEW_TOKENS)} MB, contiguous`}</Label>
          <Arrow x1={BAR_X + px(NEW_TOKENS) + 4} y1={312} x2={BAR_X + px(NEW_TOKENS) - 6} y2={312} kind="error" />
          <Label x={BAR_X + px(NEW_TOKENS) + 10} y={316} tone={1} size={10} bold kind="error">{"no hole fits"}</Label>
          <Label x={BAR_X} y={342} tone={1} size={11} bold kind="error">
            {`OUT OF MEMORY — ${MB(holeSum)} MB free, but nothing contiguous that big`}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}