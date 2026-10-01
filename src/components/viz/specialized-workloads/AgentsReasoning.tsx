import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Chip, Fade, Bar, fmt } from '../lib/svg';
import {
  conversation, prefillSaving, reasoning, agentLoop, TRACE_LENGTHS, gpu, BLOCK,
} from './workloads';

const STEPS = [
  'One turn is a normal request',
  'Turn two repeats turn one',
  'The prefix cache turns repeats into reads',
  'How much prefill a conversation avoids',
  'Reasoning models emit far more tokens',
  'A long trace slows itself down',
  'The agent loop: context that never shrinks',
  'What to do about it',
];
const CAPTIONS = [
  'Start with the case you already know. A chat turn is a request: prefill the prompt, decode a reply, stream it out. Nothing about multi-turn conversation changes the engine — at first.',
  'Now the second turn. The model is shown the whole conversation again, including the reply it just gave. That first turn is byte-identical to the first request. We computed it already.',
  'Prefix caching hashes blocks of the prompt and reuses their KV. A conversation is one prefix that grows by a block at a time, so every turn after the first is mostly a cache hit. This is Module 10’s idea, applied to the one workload that repeats itself perfectly.',
  'Worked example: a 6-turn conversation with a system prompt and tool results. Without a cache, every turn prefills its entire context — quadratic in the number of turns. With one, only the new tokens are computed.',
  'Reasoning models change the token count, not the machinery. A hard question can produce thousands of tokens of deliberation before the answer. Each one is a decode step, and each one extends the context.',
  'Because the trace joins the context and decode is memory-bound, a longer trace means a slower trace. Tokens per second falls as the conversation goes on — the model gets slower as it thinks more.',
  'An agent is this loop with a tool call in the middle. Every observation appends to the context and nothing is ever removed. Context grows linearly, prefill work without a cache grows quadratically.',
  'The three things you actually control: cap the observation size, cache the prefix, and truncate the oldest turns when the context is full. Everything else is the model’s business.',
];

export default function AgentsReasoning() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [traceId, setTraceId] = useState<string>('long');
  const [obs, setObs] = useState(3000);
  const [system, setSystem] = useState(800);
  const [turns, setTurns] = useState(6);
  const [batchSize, setBatchSize] = useState(32);

  const conv = useMemo(() => conversation(system, 120, turns), [system, turns]);
  const save = useMemo(() => prefillSaving(conv, 8e9), [conv]);
  const t = TRACE_LENGTHS.find((x) => x.id === traceId)!;
  const rp = useMemo(() => reasoning({ trace: t.tokens, kvPerToken: 131072, wBytes: 16.06e9, g: gpu('h100'), batch: batchSize }), [t, batchSize]);
  const loop = useMemo(() => agentLoop(system, obs, turns), [system, obs, turns]);

  const maxCtx = Math.max(...conv.map((c) => c.ctxAfter));
  const cacheFrac = save.saved;

  return (
    <Stage
      rootRef={ref}
      title="Agents & reasoning: context that only grows"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'KV cache held'],
        ['token', 'new tokens'],
        ['compute', 'prefill work'],
        ['error', 'recomputed from scratch'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Context after turn 6', value: fmt.int(conv[conv.length - 1].ctxAfter), kind: 'memory', note: `${BLOCK}-token blocks` },
          { label: 'Prefill work saved', value: `${(cacheFrac * 100).toFixed(0)}%`, kind: 'token', note: 'by the prefix cache' },
          { label: 'Reasoning trace', value: fmt.int(t.tokens), kind: 'compute', note: `batch ${batchSize}: ${rp.tps.toFixed(0)} tok/s, ${rp.totalS.toFixed(1)}s` },
          { label: 'Agent context, turn 6', value: fmt.int(loop[loop.length - 1].ctx), kind: 'error', note: `${fmt.int(loop[loop.length - 1].redundantPrefillTokens)} prefill tokens without a cache` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Reasoning trace" value={traceId} options={TRACE_LENGTHS.map((x) => ({ value: x.id, label: x.label }))} onChange={setTraceId} />
          <Slider label="System prompt" kind="memory" min={100} max={4000} step={100} value={system} onChange={setSystem} format={(v) => `${fmt.int(v)} tok`} />
          <Slider label="Tool observation" kind="error" min={200} max={12000} step={200} value={obs} onChange={setObs} format={(v) => `${fmt.int(v)} tok`} />
          <Slider label="Turns" kind="token" min={2} max={20} step={1} value={turns} onChange={setTurns} format={(v) => `${v} turns`} />
          <Slider label="Decode batch" kind="compute" min={1} max={64} step={1} value={batchSize} onChange={setBatchSize} format={(v) => `batch ${v}`} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="A growing multi-turn context shown as stacked blocks, with cached prefix blocks in blue and newly computed blocks in green, alongside the prefill work a prefix cache avoids and how a reasoning trace slows itself down.">
        {/* ── panel A: the growing conversation ── */}
        <Label x={12} y={20} tone={2} size={11} bold>The conversation, turn by turn</Label>
        {conv.map((c, i) => {
          const y = 30 + i * 26;
          const scale = maxCtx / 480;
          const fullW = c.ctxAfter * scale;
          const hitW = c.cacheHits * BLOCK * scale;
          return (
            <g key={i}>
              <Label x={12} y={y + 10} tone={3} size={8.5}>t{c.turn + 1}</Label>
              <rect x={40} y={y} width={fullW} height={14} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} className="v-cell" />
              {c.cacheHits > 0 && (
                <rect x={40} y={y} width={hitW} height={14} rx={2} fill="var(--memory)" opacity={0.75} className="v-cell" />
              )}
              <rect x={40 + hitW} y={y} width={Math.max(2, c.cacheMisses * BLOCK * scale)} height={14} rx={2} fill="var(--token)" className="v-cell" />
              <Label x={44 + fullW} y={y + 10} tone={2} size={8.5}>{fmt.int(c.ctxAfter)}</Label>
            </g>
          );
        })}
        {st >= 2 && (
          <g>
            <Label x={40} y={30 + conv.length * 26 + 4} tone={3} size={8.5} kind="memory">blue = cache hit, green = newly prefilled</Label>
          </g>
        )}

        {/* ── panel B: the saving ── */}
        <Fade show={st >= 2}>
          <g>
            <Label x={360} y={20} tone={2} size={11} bold>Prefill work over {turns} turns</Label>
            <Label x={360} y={40} tone={3} size={9.5} mono={false}>without a cache</Label>
            <Bar x={360} y={46} w={240} h={13} frac={1} kind="error" />
            <Label x={360} y={76} tone={1} size={9.5}>{`${fmt.si(save.naiveFlops / 1e12)} TFLOP`}</Label>
            <Label x={360} y={96} tone={3} size={9.5} mono={false}>with a prefix cache</Label>
            <Bar x={360} y={102} w={Math.max(3, (save.cachedFlops / save.naiveFlops) * 240)} h={13} frac={save.cachedFlops / save.naiveFlops} kind="token" />
            <Label x={360} y={132} tone={1} size={9.5}>{`${fmt.si(save.cachedFlops / 1e12)} TFLOP`}</Label>
            <Label x={360} y={152} size={10} kind="token" bold>{`${(cacheFrac * 100).toFixed(0)}% of the prefill disappears`}</Label>
          </g>
        </Fade>

        {/* ── panel C: reasoning trace ── */}
        <Fade show={st >= 4}>
          <g>
            <Label x={360} y={180} tone={2} size={11} bold>The cost of thinking out loud</Label>
            {TRACE_LENGTHS.map((x, i) => {
              const on = x.id === traceId;
              const y = 192 + i * 22;
              return (
                <g key={x.id}>
                  <Label x={360} y={y + 10} tone={on ? 1 : 3} size={9} mono={false}>{x.label}</Label>
                  <Bar x={440} y={y + 2} w={160 * (x.tokens / 16000)} h={11} frac={x.tokens / 16000} kind={on ? 'compute' : 'memory'} ghost={!on} />
                  <Label x={608} y={y + 11} anchor="end" tone={on ? 1 : 3} size={9}>{fmt.int(x.tokens)}</Label>
                </g>
              );
            })}
          </g>
        </Fade>

        {/* ── panel D: trace slows itself ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={360} y={290} tone={2} size={9.5} mono={false}>{`${t.label} at batch ${batchSize}: ${rp.tps.toFixed(0)} tok/s, ${rp.totalS.toFixed(1)}s to finish`}</Label>
            <Label x={360} y={306} size={9.5} kind={batchSize <= 4 ? 'error' : 'token'} mono={false}>{`${(rp.stepMs).toFixed(2)} ms/token — weights read once per step, so batch ${batchSize} pays 1/${batchSize}.`}</Label>
          </g>
        </Fade>

        {/* ── panel E: the agent loop ── */}
        <Fade show={st >= 6}>
          <g>
            <Label x={360} y={186} tone={2} size={11} bold>Agent loop: nothing is ever removed</Label>
            {loop.slice(0, 6).map((l, i) => {
              const y = 196 + i * 20;
              const scale = 200 / (system + obs * 6);
              return (
                <g key={l.turn}>
                  <Label x={360} y={y + 10} tone={3} size={8.5}>turn {l.turn}</Label>
                  <Bar x={402} y={y + 2} w={Math.max(3, l.ctx * scale)} h={11} frac={l.ctx / (system + obs * 6)} kind="memory" />
                  <Label x={608} y={y + 11} anchor="end" tone={2} size={8.5}>{fmt.int(l.ctx)}</Label>
                </g>
              );
            })}
          </g>
        </Fade>

        <Fade show={st >= 7}>
          <g>
            <Chip x={12} y={286} text="cap observations" kind="compute" h={18} w={104} size={9} />
            <Chip x={124} y={286} text="cache the prefix" kind="memory" h={18} w={88} size={9} />
            <Chip x={220} y={286} text="truncate oldest turns" kind="error" h={18} w={116} size={9} />
            <Label x={12} y={322} tone={3} size={9.5}>The three levers an inference engineer actually has here.</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}