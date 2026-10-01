import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout, type Kind } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Span, Label, Arrow, Fade, fmt } from '../lib/svg';
import { decodeStep, prefillTime, modelById, gpuById, kvBytesPerToken } from '../lib/perfModel';
import { KV_BLOCK_TOKENS } from './engine';

const model = modelById('llama8b');
const gpu = gpuById('h100');
const WAIT = 0.06; // seconds spent in the queue: a fixed, plausible arrival delay

const STAGES = ['HTTP API', 'Tokenizer', 'Scheduler', 'Model runner', 'Sampler', 'Detokenizer'];
const ACTIVE = [0, 1, 2, 3, 4, 5, 5];

const PAYLOAD = [
  'in:  {"model": "…", "messages": [...], "max_tokens": 256, "stream": true}',
  'in:  [8831, 1743, 2851, 4190, 6082, …]   the same words, as integers',
  'in:  waiting [R2, R3, R4]   →   running [R1]',
  'in:  every prompt token in one forward pass → one KV block per 16',
  'in:  logits[128256] → softmax → sample → "Paged"',
  'in:  +1 token per step: " Attention", " lets", " an", …',
  'out: the blocks are freed → the next request takes them in one step',
];

const STEPS = [
  'A POST arrives',
  'Tokenize',
  'The scheduler admits it',
  'Prefill',
  'The first token',
  'Decode, step after step',
  'Finish, free, refill',
];
const CAPTIONS = [
  'The API server takes the HTTP body and validates it: model name, messages, sampling parameters. No GPU work yet; the request joins the waiting queue.',
  'The tokenizer turns your text into integer token IDs. The engine never sees words — from here on it is integers in and integers out.',
  'The scheduler decides who runs. A request stays in the running batch until it finishes, so admission is the moment its latency clock really starts.',
  'Prefill: the whole prompt through the model in one pass. This is the compute-bound phase, and it writes one KV block for every 16 tokens of context.',
  'The last hidden state goes through the LM head, softmax and the sampler. One token exists. Everything the user was waiting for just happened: that was the TTFT.',
  'Every step after that is the memory-bound phase: read all the weights, read this sequence’s KV cache, produce one token, write it. Then repeat, identically.',
  'When a request hits its stop condition its blocks return to the pool, and in the very next step a waiting request takes them. Nothing waits for the batch to drain.',
];

const L = 40;
const R = 610;

interface Lane {
  name: string;
  t0: number | null;
  t1: number | null;
  kind: Kind;
  label: string;
  dim?: boolean;
}

export default function RequestInsideEngine() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [prompt, setPrompt] = useState(256);
  const [out, setOut] = useState(64);
  useEffect(() => { setPrompt(s.step >= 3 ? 512 : 256); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const itl = decodeStep({ model, gpu, batch: 1, ctx: prompt }).t;
  const pre = prefillTime({ model, gpu, tokens: prompt });
  const ttft = WAIT + pre + itl;
  const total = pre + out * itl;

  const preBlocks = Math.ceil(prompt / KV_BLOCK_TOKENS);
  const allBlocks = Math.ceil((prompt + out) / KV_BLOCK_TOKENS);
  const shownBlocks = Math.min(allBlocks, 24);
  const kvBytes = (prompt + out) * kvBytesPerToken(model);
  const blocksNow = s.step >= 6 ? 0 : s.step >= 5 ? allBlocks : s.step >= 3 ? preBlocks : 0;

  const tw = useTweens({ wait: (WAIT / ttft) * 250, pre: (pre / ttft) * 250, itl: (itl / ttft) * 250 });

  // decode ticks at their true spacing, packed until the panel is full
  const nominal = 272 / Math.max(1, out - 1);
  const tw2 = Math.max(1.4, Math.min(5, nominal * 0.62));
  const visible = Math.max(2, Math.min(out, Math.floor(272 / Math.max(1.4, nominal)) + 1));
  const tickOn = (i: number) => (s.step >= 5 ? i < visible : s.step === 4 && i === 0);

  const span = total * 2.4;
  const bx = (t: number) => L + Math.min(1, t / span) * (R - L);
  const lanes = useMemo<Lane[]>(
    () => [
      { name: 'R1', t0: 0, t1: total, kind: 'token', label: 'this request' },
      { name: 'R2', t0: 0, t1: total * 1.55, kind: 'token', dim: true, label: 'a longer answer' },
      { name: 'R3', t0: s.step >= 6 ? total : null, t1: s.step >= 6 ? total * 1.5 : null, kind: 'compute', label: 'was waiting' },
      { name: 'R4', t0: null, t1: null, kind: 'idle', label: 'still waiting' },
    ],
    [total, s.step],
  );

  const step = Math.min(s.step, PAYLOAD.length - 1);

  return (
    <Stage
      rootRef={ref}
      title="One request inside the engine core"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', ['idle', 'queueing']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Prompt', value: `${fmt.int(prompt)} tokens`, kind: 'text' },
          { label: 'Prefill', value: `${(pre * 1000).toFixed(1)} ms`, kind: 'compute', note: 'compute-bound' },
          { label: 'Time to first token', value: `${(ttft * 1000).toFixed(0)} ms`, kind: 'idle', note: `${(1 / itl).toFixed(0)} tok/s after that` },
          { label: 'KV held at the end', value: `${fmt.int(allBlocks)} blocks`, kind: 'memory', note: `${(kvBytes / 1e6).toFixed(0)} MB of the pool` },
        ]} />
      }
      controls={
        <>
          <Slider label="Prompt length" value={prompt} min={64} max={8192} log onChange={setPrompt} format={(v) => `${fmt.int(v)} tokens`} kind="compute" />
          <Slider label="Answer length" value={out} min={1} max={512} log onChange={setOut} format={(v) => `${fmt.int(v)} tokens`} kind="token" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A request moving through the vLLM pipeline — HTTP API, tokenizer, scheduler, model runner, sampler, detokenizer — with a time-to-first-token bar, a stream of decode ticks, the KV blocks the request holds, and the lanes of the running batch.">
        {/* ── the pipeline ── */}
        {STAGES.map((name, i) => {
          const x = 16 + i * 103.2;
          const on = i === ACTIVE[step];
          return (
            <g key={name}>
              <rect x={x} y={22} width={92} height={30} rx={3}
                fill={on ? 'var(--compute-dim)' : 'transparent'}
                stroke={on ? 'var(--compute)' : 'var(--rule-2)'} />
              <Label x={x + 46} y={41} anchor="middle" size={9.5} bold={on} kind={on ? 'compute' : undefined} tone={on ? 1 : 3}>{name}</Label>
              {i < STAGES.length - 1 && <Arrow x1={x + 94} y1={37} x2={x + 101} y2={37} muted />}
            </g>
          );
        })}
        <Label x={16} y={68} tone={1} size={9.5}>{PAYLOAD[step]}</Label>

        <line x1={16} x2={624} y1={84} y2={84} className="v-rule faint" />

        {/* ── 1. time to first token ── */}
        <Label x={16} y={102} tone={2} size={10.5} mono={false} bold>1. Time to first token</Label>
        <text x={L - 12} y={134} textAnchor="end" className="v-track-label">ttft</text>
        <rect x={L} y={120} width={250} height={22} rx={2} fill="none" stroke="var(--rule)" />
        <Span x={L} y={120} w={tw.wait} h={22} kind="idle" label="queue" labelTone="light" />
        <Span x={L + tw.wait} y={120} w={tw.pre} h={22} kind="compute" />
        <Span x={L + tw.wait + tw.pre} y={120} w={tw.itl} h={22} kind="token" />
        <Label x={L + tw.wait / 2} y={157} anchor="middle" tone={3} size={9}>{`${(WAIT * 1000).toFixed(0)} ms in the queue`}</Label>
        <Label x={L + tw.wait + tw.pre / 2} y={173} anchor="middle" size={9} kind="compute">{`${(pre * 1000).toFixed(0)} ms prefill`}</Label>
        <Label x={L + 258} y={134} tone={1} size={10}>{`${(ttft * 1000).toFixed(0)} ms`}</Label>
        <Label x={L} y={192} tone={3} size={9.5}>{`GPU busy ${(pre * 1000).toFixed(0)} ms · queue ${(WAIT * 1000).toFixed(0)} ms`}</Label>

        {/* ── 2. decode, one token per step ── */}
        <Label x={334} y={102} tone={2} size={10.5} mono={false} bold>2. Then one token per step</Label>
        <Fade show={s.step >= 4}>
          {Array.from({ length: visible }, (_, i) => {
            const x = 340 + i * nominal;
            return x <= 614 ? (
              <rect key={i} x={x} y={118} width={tw2} height={20} rx={1.5} fill="var(--token)" className="v-fade" style={{ opacity: tickOn(i) ? 1 : 0 }} />
            ) : null;
          })}
          {nominal >= 12 ? (
            <g>
              <line x1={340} x2={340 + nominal} y1={150} y2={150} stroke="var(--text-2)" strokeWidth={1} />
              <line x1={340} x2={340} y1={146} y2={154} stroke="var(--text-2)" strokeWidth={1} />
              <line x1={340 + nominal} x2={340 + nominal} y1={146} y2={154} stroke="var(--text-2)" strokeWidth={1} />
              <Label x={340 + nominal / 2} y={166} anchor="middle" tone={2} size={9}>{`${(itl * 1000).toFixed(1)} ms`}</Label>
            </g>
          ) : null}
          <Label x={612} y={134} anchor="end" tone={3} size={9}>{`${fmt.int(out)} tokens = ${(out * itl * 1000).toFixed(0)} ms of streaming`}</Label>
          <Label x={340} y={192} tone={3} size={9.5}>{`${(1 / itl).toFixed(0)} tokens/s per user, memory-bound`}</Label>
        </Fade>

        <line x1={16} x2={624} y1={210} y2={210} className="v-rule faint" />

        {/* ── 3. the KV blocks it holds ── */}
        <Label x={16} y={228} tone={2} size={10.5} mono={false} bold>3. The KV blocks it holds</Label>
        {Array.from({ length: shownBlocks }, (_, i) => {
          const on = i < blocksNow;
          return (
            <rect key={i} x={L + i * 19} y={238} width={16} height={18} rx={2}
              fill={on ? 'var(--memory)' : 'none'} stroke={on ? 'var(--memory)' : 'var(--rule-2)'} strokeDasharray={on ? undefined : '3 3'} />
          );
        })}
        {allBlocks > shownBlocks && <Label x={L + shownBlocks * 19 + 6} y={251} tone={3} size={9.5}>{`+${fmt.int(allBlocks - shownBlocks)}`}</Label>}
        <Label x={R} y={274} anchor="end" tone={2} size={9.5}>
          {`${fmt.int(allBlocks)} blocks × ${KV_BLOCK_TOKENS} tokens = ${fmt.int(prompt + out)} tokens × 128 KB = ${(kvBytes / 1e6).toFixed(1)} MB`}
        </Label>

        <line x1={16} x2={624} y1={282} y2={282} className="v-rule faint" />

        {/* ── 4. the running batch ── */}
        <Label x={16} y={300} tone={2} size={10.5} mono={false} bold>4. The running batch: one weight read serves all of it</Label>
        {lanes.map((ln, i) => {
          const y = 312 + i * 17;
          const live = ln.t0 !== null && ln.t1 !== null;
          return (
            <g key={ln.name}>
              <Label x={L - 12} y={y + 8} anchor="end" tone={3} size={9}>{ln.name}</Label>
              <line x1={L} x2={R} y1={y + 11} y2={y + 11} className="v-rule faint" />
              <Fade show={s.step >= 2}>
                {live ? (
                  <Span x={bx(ln.t0 as number)} y={y + 2} w={Math.max(2, bx(ln.t1 as number) - bx(ln.t0 as number))} h={11} kind={ln.kind} dim={ln.dim} label={ln.label} />
                ) : (
                  <Span x={L} y={y + 2} w={R - L} h={11} kind="idle" label={ln.label} labelTone="light" ghost />
                )}
              </Fade>
            </g>
          );
        })}
      </svg>
    </Stage>
  );
}