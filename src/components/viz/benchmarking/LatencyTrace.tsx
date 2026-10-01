import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Span, Label, Fade, fmt } from '../lib/svg';
import { Bracket, msec } from './helpers';

// One request's whole life, on one time axis. Every number here is derived in the
// lesson: 16 GB of weights on an H100 ≈ 6 ms of memory time per decode step,
// plus ~1.5 ms of fixed overhead.

const STEPS = [
  'The request arrives',
  'Wait in the queue',
  'Prefill: read it all at once',
  'First token — TTFT',
  'Decode, one token per step',
  'The inter-token gaps',
  'End to end',
  'Only two knobs matter',
];
const CAPTIONS = [
  'A request arrives at t = 0. Everything that happens to it happens on one timeline.',
  'It may sit in a queue first. Queueing delay is invisible in kernel profiles and it is often the largest term.',
  'Prefill processes the whole prompt in one parallel pass: compute-bound, and its cost grows with prompt length.',
  'TTFT is queueing plus prefill plus one decode step. This is the wait a user calls "the model is slow".',
  'After that, every remaining token costs one decode step, dominated by reading 16 GB of weights.',
  'ITL is each individual gap. If the gaps are uneven, the average is not what the user feels.',
  'Total time = TTFT + (n−1) × TPOT. Change the prompt length or n and you change which term dominates.',
  'Prompt length moves TTFT. n moves the tail. Those are the two inputs you control.',
];

// Presets: promptTokens, outputTokens, queueMs
const PRESET: Record<number, { prompt: number; out: number; queue: number }> = {
  0: { prompt: 20, out: 1, queue: 0 },
  1: { prompt: 20, out: 1, queue: 240 },
  2: { prompt: 900, out: 1, queue: 240 },
  3: { prompt: 900, out: 1, queue: 240 },
  4: { prompt: 900, out: 200, queue: 240 },
  5: { prompt: 900, out: 200, queue: 240 },
  6: { prompt: 900, out: 200, queue: 240 },
  7: { prompt: 900, out: 200, queue: 240 },
};

const X0 = 104;
const X1 = 622;
const BASE_Y = 186;

// Model of one decode step for Llama-3.1-8B in BF16 on an H100, batch 1.
const STEP_MS = 7.5; // 16 GB / 3.35 TB/s at ~80% MBU, + 1.5 ms overhead
const PREFLIP_MS_PER_1K = 34; // prefill: ~1.8 TFLOP per 1k tokens on an H100, ~55% MFU

export default function LatencyTrace() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const preset = PRESET[s.step];
  const [prompt, setPrompt] = useState(preset.prompt);
  const [out, setOut] = useState(preset.out);
  const [queue, setQueue] = useState(preset.queue);
  useEffect(() => { setPrompt(preset.prompt); setOut(preset.out); setQueue(preset.queue); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const prefillMs = Math.max(8, (prompt / 1000) * PREFLIP_MS_PER_1K);
  const decodeSteps = Math.max(0, out - 1);
  const ttft = queue + prefillMs + STEP_MS;
  const tpot = STEP_MS;
  const total = ttft + decodeSteps * STEP_MS;

  // full scale is a little more than the whole request so nothing is clipped
  const scaleMax = Math.max(total * 1.12, 60);
  const px = (ms: number) => (ms / scaleMax) * (X1 - X0);

  const tQueue = useTween(queue, 600);
  const tPre = useTween(prefillMs, 600);
  const tDecode = useTween(decodeSteps * STEP_MS, 600);

  const showQueue = s.step >= 1;
  const showPrefill = s.step >= 2;
  const showTtft = s.step >= 3;
  const showStream = s.step >= 4;
  const showItl = s.step >= 5;
  const showTotal = s.step >= 6;
  const showKnobs = s.step >= 7;

  const q = px(tQueue);
  const p = px(tPre);
  const d = px(tDecode);
  const xFirst = X0 + q + p + px(STEP_MS); // first token lands here
  const xLast = xFirst + d;

  // the per-token gaps, drawn as tick marks on the decode span
  const gapStep = Math.max(1, Math.ceil(out / 40));

  return (
    <Stage
      rootRef={ref}
      title="One request on one timeline: where the waiting actually happens"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', 'token', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'TTFT', value: msec(ttft), kind: 'idle', note: 'queue + prefill + 1 step' },
          { label: 'TPOT', value: `${STEP_MS.toFixed(1)} ms`, kind: 'token', note: `${(1000 / STEP_MS).toFixed(0)} tok/s per user` },
          { label: 'End to end', value: msec(total), kind: 'text', note: `${out} token${out === 1 ? '' : 's'}` },
          { label: 'Queue share', value: `${((queue / total) * 100).toFixed(0)}%`, kind: queue / total > 0.3 ? 'error' : 'idle', note: 'of the wait' },
        ]} />
      }
      controls={
        <>
          <Slider label="Prompt length (tokens)" value={prompt} min={8} max={4096} log onChange={setPrompt} format={(v) => `${fmt.int(v)} tok`} kind="compute" />
          <Slider label="Tokens generated" value={out} min={1} max={512} log onChange={setOut} format={(v) => `${fmt.int(v)}`} kind="token" />
          <Slider label="Queueing delay" value={queue} min={0} max={2000} onChange={setQueue} format={(v) => msec(v)} kind="idle" />
        </>
      }
    >
      <svg viewBox="0 0 640 348" role="img" aria-label="A timeline of one request: a queueing wait, then a prefill pass, then one decode step per generated token, with brackets marking TTFT, TPOT and end-to-end time.">
        {/* time axis */}
        <line x1={X0} x2={X1} y1={BASE_Y + 46} y2={BASE_Y + 46} className="v-rule" />
        <Label x={X0} y={BASE_Y + 62} tone={3} size={9.5}>0 ms</Label>
        <Label x={X1} y={BASE_Y + 62} anchor="end" tone={3} size={9.5}>{msec(scaleMax)}</Label>

        {/* the request itself */}
        <Label x={X0 - 10} y={BASE_Y + 10} anchor="end" tone={2} size={10.5} mono={false}>request</Label>

        {/* queue */}
        {showQueue && (
          <>
            <Span x={X0} y={BASE_Y} w={q} h={18} kind="idle" />
            {q > 40 && <text x={X0 + q / 2} y={BASE_Y + 12.8} textAnchor="middle" className="v-label" style={{ fontSize: 10 }}>waiting</text>}
            <Bracket x1={X0} x2={X0 + q} y={BASE_Y - 12} label={`queue ${msec(queue)}`} color="var(--text-3)" side="left" />
          </>
        )}

        {/* prefill */}
        {showPrefill && (
          <>
            <Span x={X0 + q} y={BASE_Y} w={p} h={18} kind="compute" label={p > 46 ? 'prefill' : undefined} />
            <Bracket x1={X0 + q} x2={X0 + q + p} y={BASE_Y + 32} label={`prefill ${msec(prefillMs)}`} color="var(--compute)" side="right" />
          </>
        )}

        {/* the first decode step: prefill does not emit a token until a decode step runs */}
        {showTtft && (
          <Fade show={s.step >= 3}>
            <Span x={X0 + q + p} y={BASE_Y} w={Math.max(2, px(STEP_MS))} h={18} kind="memory" />
            <line x1={xFirst} x2={xFirst} y1={BASE_Y - 24} y2={BASE_Y + 26} stroke="var(--text)" strokeWidth={1.2} />
            <Label x={xFirst} y={BASE_Y - 30} anchor="middle" tone={1} size={10} bold>first token</Label>
          </Fade>
        )}

        {/* the rest of the decode */}
        {showStream && d > 0 && (
          <Fade show={s.step >= 4}>
            <Span x={xFirst} y={BASE_Y} w={d} h={18} kind="memory" label={d > 46 ? `decode × ${fmt.int(out - 1)}` : undefined} />
          </Fade>
        )}

        {/* individual inter-token gaps */}
        {showItl && d > 6 && (
          <Fade show={s.step >= 5}>
            {Array.from({ length: Math.min(40, out) }, (_, i) => (
              <line
                key={i}
                x1={xFirst + (i * gapStep * STEP_MS) / scaleMax * (X1 - X0)}
                x2={xFirst + (i * gapStep * STEP_MS) / scaleMax * (X1 - X0)}
                y1={BASE_Y + 22}
                y2={BASE_Y + 30}
                stroke="var(--token)"
                strokeWidth={1}
              />
            ))}
            <Bracket x1={xFirst} x2={xFirst + Math.min(d, (STEP_MS / scaleMax) * (X1 - X0))} y={BASE_Y + 74} label={`ITL ${STEP_MS.toFixed(1)} ms`} color="var(--token)" side="left" />
            <Label x={X0} y={BASE_Y + 78} tone={3} size={9.5}>every gap, measured separately</Label>
          </Fade>
        )}

        {/* TTFT */}
        {showTtft && (
          <Fade show={s.step >= 3}>
            <Bracket x1={X0} x2={xFirst} y={BASE_Y - 52} label={`TTFT = ${msec(ttft)}`} color="var(--idle)" side="left" weight={700} />
            <Label x={X0} y={BASE_Y - 58} tone={2} size={10} bold>time to first token</Label>
          </Fade>
        )}

        {/* end to end */}
        {showTotal && (
          <Fade show={s.step >= 6}>
            <line x1={XLast} x2={XLast} y1={BASE_Y - 24} y2={BASE_Y + 34} stroke="var(--text)" strokeWidth={1.2} />
            <Bracket x1={X0} x2={XLast} y={BASE_Y + 100} label={`end to end = ${msec(total)}`} color="var(--text-2)" side="right" weight={700} />
            <Label x={X0} y={BASE_Y + 94} tone={2} size={10} bold>the whole response</Label>
          </Fade>
        )}

        {/* the decomposition */}
        {showTotal && (
          <Fade show={s.step >= 6}>
            <Label x={X0} y={BASE_Y + 128} tone={2} size={10.5} mono={false} bold>Where the {msec(total)} goes</Label>
            <Span x={X0} y={BASE_Y + 136} w={Math.max(1, px(queue))} h={14} kind="idle" />
            <Span x={X0 + px(queue)} y={BASE_Y + 136} w={Math.max(1, px(prefillMs))} h={14} kind="compute" />
            <Span x={X0 + px(queue) + px(prefillMs)} y={BASE_Y + 136} w={Math.max(1, px(decodeSteps * STEP_MS + STEP_MS))} h={14} kind="memory" />
            <Label x={X0} y={BASE_Y + 166} tone={3} size={9.5}>
              {`queue ${((queue / total) * 100).toFixed(0)}%  ·  prefill ${((prefillMs / total) * 100).toFixed(0)}%  ·  decode ${(((decodeSteps * STEP_MS + STEP_MS) / total) * 100).toFixed(0)}%`}
            </Label>
          </Fade>
        )}
      </svg>
    </Stage>
  );
}