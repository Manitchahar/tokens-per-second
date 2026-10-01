import { useRef } from 'react';
import { Stage } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween, easeInOutCubic } from '../lib/tween';
import { RequestTrace, T, TTFT, END, tokenTime } from './RequestTrace';

const STEPS = [
  { label: 'Press Enter', t: 0 },
  { label: 'Across the network', t: T.net },
  { label: 'Text becomes token IDs', t: T.tokenizeEnd, ids: true },
  { label: 'Wait in the queue', t: T.queueEnd, ghosts: true },
  { label: 'Prefill the prompt', t: T.prefillEnd },
  { label: 'First token arrives', t: TTFT },
  { label: 'Decode, token by token', t: tokenTime(9) },
  { label: 'Done', t: END + 20, brackets: true },
];

const CAPTIONS = [
  'You type a question and press Enter. The clock starts now.',
  'Your text travels over the internet to the provider’s API server (tens of milliseconds).',
  'The server splits the text into tokens and turns each into an integer ID. The model only ever sees numbers.',
  'The GPU is busy with other people’s requests, so yours waits briefly in a queue.',
  'Prefill: the GPU processes all 8 prompt tokens in one parallel pass, fills the KV cache, and produces the first output token.',
  'The first token travels back and appears on your screen. That delay is the time to first token (TTFT).',
  'Decode: one GPU step per token. Each step reads the whole model from memory to produce just one token, so it is memory-bound (blue).',
  'The request ends. Two numbers describe how it felt: TTFT (how long until it started) and TPOT (how fast it streamed).',
];

export default function RequestJourney() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3200 });
  const cur = STEPS[s.step];
  // slower when the step covers a lot of time, so the trace visibly "draws"
  const t = useTween(cur.t, 1400, easeInOutCubic);
  return (
    <Stage
      rootRef={ref}
      title="One request, drawn as a trace"
      steps={STEPS.map((x) => x.label)}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', 'comm', 'idle']}
      {...s}
    >
      <RequestTrace t={t} showIds={!!cur.ids} showQueueGhosts={!!cur.ghosts || s.step > 3} showBrackets={!!cur.brackets} />
    </Stage>
  );
}
