import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, fmt } from '../lib/svg';
import {
  serviceTime,
  stepAt,
  costPerMTokOut,
  blendedPerMTok,
  perSecond,
  tensorParallel,
  regimeOf,
  gpuById,
  ctxFor,
  type Op,
} from './econ';

// The one identity: what a request costs is the machine time it consumed, times the
// price of a second of that machine. Seven steps build that sentence up, and each one
// adds exactly one term to the bill.

const STEPS = [
  'A GPU rents by the second',
  'Prefill is one request’s own work',
  'A decode step is shared work',
  'The identity: machine seconds',
  'Multiply by price, and you have a bill',
  'The bill has two halves',
  'The shape of your traffic decides the split',
  'When the model will not fit',
];
const CAPTIONS = [
  'A rented GPU is charged for every second it exists — busy or idle. So the first thing to know is not how fast it is, but what a second costs. At \$2.50 per hour, one second is \$0.00069.',
  'Prefill is the one phase that belongs to a single request alone: the prompt goes through the model in one parallel, compute-bound pass. Nobody shares it, so its cost is entirely yours.',
  'Decode is different. Each step reads all the weights and all the KV, but it produces one token for every sequence in the batch. So one request\'s share of a step is the step time divided by the batch.',
  'Add them and you have the machine time one request consumed: t_prefill + n_out × t_step / B. This is service time, but read as a quantity of machine rather than latency.',
  'Multiply by the price of a second and divide by the tokens. That is dollars per million tokens — the whole subject of this lesson in one line.',
  'The two halves are priced by two different limits. Prefill is compute-bound, so it costs what you pay per unit of FLOP/s. Decode is memory-bound, so it costs what you pay per unit of bandwidth. They are not the same price.',
  'Which half dominates is set by your traffic, not by your hardware. A long prompt and a short answer stays compute-priced. A one-line prompt and a long answer becomes bandwidth-priced. Slide to see it move.',
  'Sometimes neither regime applies, because the model does not fit on one card at all. Then the bill is multiplied by the number of cards and every step is slowed by the communication between them.',
];

const PRESET: Record<number, { batch: number; nIn: number; nOut: number }> = {
  0: { batch: 64, nIn: 512, nOut: 256 },
  1: { batch: 64, nIn: 512, nOut: 256 },
  2: { batch: 64, nIn: 512, nOut: 256 },
  3: { batch: 64, nIn: 512, nOut: 256 },
  4: { batch: 64, nIn: 512, nOut: 256 },
  5: { batch: 64, nIn: 512, nOut: 256 },
  6: { batch: 64, nIn: 2048, nOut: 64 },
  7: { batch: 64, nIn: 512, nOut: 256 },
};

export default function CostPerToken() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [batch, setBatch] = useState(PRESET[0].batch);
  const [nIn, setNIn] = useState(PRESET[0].nIn);
  const [nOut, setNOut] = useState(PRESET[0].nOut);
  const [modelId, setModelId] = useState('llama8b');
  const [gpuId, setGpuId] = useState('h100');
  const [wBytes, setWBytes] = useState(2);

  useEffect(() => {
    const p = PRESET[s.step];
    if (p) { setBatch(p.batch); setNIn(p.nIn); setNOut(p.nOut); }
    if (s.step === 7) setModelId('llama70b');
    else if (s.step < 7) setModelId('llama8b');
  }, [s.step]);

  const op: Op = { modelId, gpuId, batch, ctx: ctxFor(nIn), wBytes };
  const st = serviceTime(op, nOut, nIn);
  const dk = stepAt(op);
  const tp = tensorParallel(op);
  const cOut = costPerMTokOut(op, nOut, nIn) * (tp.cards > 1 ? tp.cards * tp.stepPenalty : 1);
  const cBlend = blendedPerMTok(op, nIn, nOut);
  const regime = regimeOf(op, nIn, nOut);
  const pSec = perSecond(gpuId);

  // the dollar bar: this request's machine time, laid out as prefill and decode
  const TOT = st.gpuSeconds;
  const BAR_X = 96, BAR_W = 468, BAR_Y = 176, BAR_H = 30;
  const preW = (st.tPre / TOT) * BAR_W;
  const decW = (st.decodeShare / TOT) * BAR_W;

  const showPre = s.step >= 1;
  const showDec = s.step >= 2;
  const showBoth = s.step >= 3;
  const showMoney = s.step >= 4;
  const showHalves = s.step >= 5;
  const showShape = s.step >= 6;
  const showTP = s.step >= 7;

  return (
    <Stage
      rootRef={ref}
      title="What one request costs, term by term"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'prefill — priced per FLOP/s'],
        ['memory', 'decode — priced per bandwidth'],
        ['text', 'machine seconds'],
        ['error', 'does not fit on one card'],
        'idle',
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Machine time', value: `${(TOT * 1000).toFixed(1)} ms`, kind: 'memory', note: `per request, at batch ${batch}` },
          { label: 'Prefill share', value: `${(st.prefillFrac * 100).toFixed(0)}%`, kind: 'compute', note: regime === 'prefill' ? 'the dominant half' : 'the smaller half' },
          { label: 'Decode share', value: `${(st.decodeFrac * 100).toFixed(0)}%`, kind: 'memory', note: regime === 'decode' ? 'the dominant half' : 'the smaller half' },
          { label: 'Cost / 1M output tokens', value: `$${cOut.toFixed(2)}`, kind: showMoney ? 'token' : 'text', note: `blended $${cBlend.toFixed(2)}` },
        ]} />
      }
      controls={
        <>
          <Segmented label="GPU" value={gpuId} options={[{ value: 't4', label: 'T4' }, { value: 'a100', label: 'A100' }, { value: 'h100', label: 'H100' }, { value: 'h200', label: 'H200' }, { value: 'b200', label: 'B200' }]} onChange={setGpuId} />
          <Segmented label="Weights" value={wBytes} options={[{ value: 2, label: 'BF16' }, { value: 1, label: 'FP8' }, { value: 0.5, label: 'INT4' }]} onChange={setWBytes} />
          <Slider label="Concurrency (batch)" kind="memory" value={batch} min={1} max={256} log onChange={setBatch} format={(v) => `${fmt.int(v)} streams`} />
          <Slider label="Prompt tokens" kind="compute" value={nIn} min={64} max={8192} log onChange={setNIn} format={(v) => `${fmt.int(v)}`} />
          <Slider label="Output tokens" kind="token" value={nOut} min={8} max={2048} log onChange={setNOut} format={(v) => `${fmt.int(v)}`} />
        </>
      }
    >
      <svg viewBox="0 0 640 356" role="img" aria-label="A horizontal bar of the machine time one request consumes, split into a prefill portion in amber and a decode portion in blue, with the arithmetic of each shown underneath and the resulting dollars per million tokens on the right.">
        <Label x={16} y={20} tone={2} size={11} bold>One request, and the machine time it takes</Label>
        <Label x={624} y={20} anchor="end" tone={3} size={9}>
          {`${showTP ? 'Llama-3.1-70B' : 'Llama-3.1-8B'} · ${gpuById(gpuId).name} · ${wBytes === 2 ? 'BF16' : wBytes === 1 ? 'FP8' : 'INT4'}`}
        </Label>

        {/* the memory edge: a number that assumes a config that does not run is not a cost */}
        {s.step >= 3 && !dk.fits && (
          <g>
            <rect x={0} y={0} width={640} height={356} fill="url(#tps-hatch-red)" opacity={0.5} />
            <rect x={140} y={128} width={360} height={20} rx={3} fill="var(--ground-2)" stroke="var(--error)" />
            <Label x={320} y={142} anchor="middle" size={10.5} kind="error" bold>
              {`does not fit: ${fmt.gb(dk.memUsed)} of weights + KV on an 80 GB card`}
            </Label>
            <Label x={320} y={164} anchor="middle" size={9} tone={3}>
              lower the batch, shorten the context, use fewer bits, or buy another card
            </Label>
          </g>
        )}

        {/* ── the term-by-term stack ── */}
        <Label x={16} y={62} tone={3} size={9.5}>term 1 · prefill, yours alone</Label>
        <Label x={16} y={92} tone={3} size={9.5}>term 2 · decode, your share</Label>
        <Label x={16} y={122} tone={3} size={9.5} bold>total machine time</Label>

        <Fade show={showPre}>
          <rect x={BAR_X} y={50} width={BAR_W} height={16} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={BAR_X} y={50} w={BAR_W * 0.35} h={16} kind="compute" />
          <Label x={BAR_X + BAR_W * 0.35 / 2} y={62} anchor="middle" size={9.5} style={{ fill: '#0e1116' }} bold>{`${(st.tPre * 1000).toFixed(1)} ms`}</Label>
          <Label x={BAR_X + BAR_W * 0.35 + 8} y={62} tone={2} size={9.5} mono={false}>{`t_prefill = ${fmt.int(nIn)} tokens of prompt, not shared`}</Label>
        </Fade>

        <Fade show={showDec}>
          <rect x={BAR_X} y={80} width={BAR_W} height={16} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={BAR_X} y={80} w={BAR_W * 0.35} h={16} kind="memory" />
          <Label x={BAR_X + BAR_W * 0.35 / 2} y={92} anchor="middle" size={9.5} style={{ fill: '#0e1116' }} bold>{`${(st.decodeShare * 1000).toFixed(1)} ms`}</Label>
          <Label x={BAR_X + BAR_W * 0.35 + 8} y={92} tone={2} size={9.5} mono={false}>{`n_out × t_step / B = ${fmt.int(nOut)} × ${(st.tStep * 1000).toFixed(1)} / ${batch} ms`}</Label>
        </Fade>

        <Fade show={showBoth}>
          <rect x={BAR_X} y={110} width={BAR_W} height={16} rx={2} fill="none" stroke="var(--rule-2)" />
          <Span x={BAR_X} y={110} w={BAR_W} h={16} kind="text" dim />
          <Label x={BAR_X + 6} y={122} size={9.5} kind="text" bold>{`${(TOT * 1000).toFixed(1)} ms of machine time per request`}</Label>
        </Fade>

        {/* ── the equation, revealed in pieces ── */}
        <line x1={16} x2={624} y1={142} y2={142} className="v-rule faint" />

        <Fade show={showBoth}>
          <Label x={16} y={162} tone={2} size={10.5} bold>the identity</Label>
          <Label x={96} y={162} size={10.5} mono={false}>
            GPU-seconds per request = t_prefill + n_out × t_step / B
          </Label>
        </Fade>

        {/* ── the money bar ── */}
        <Fade show={showMoney}>
          <g>
            <Label x={16} y={190} tone={2} size={10.5} bold>and what that costs</Label>
            <rect x={BAR_X} y={BAR_Y} width={BAR_W} height={BAR_H} rx={2} fill="none" stroke="var(--rule)" />
            <Span x={BAR_X} y={BAR_Y} w={preW} h={BAR_H} kind="compute" label="prefill" />
            <Span x={BAR_X + preW} y={BAR_Y} w={decW} h={BAR_H} kind="memory" label="decode" />
            <Label x={624} y={BAR_Y + 12} anchor="end" size={9.5} tone={1}>{`$${(TOT * pSec).toFixed(6)}`}</Label>
            <Label x={624} y={BAR_Y + 26} anchor="end" size={9} tone={3}>for this request</Label>
          </g>
        </Fade>

        {/* ── the two halves, priced differently ── */}
        <Fade show={showHalves}>
          <g>
            <Label x={16} y={234} tone={2} size={10.5} bold>two halves, two prices</Label>
            <rect x={96} y={222} width={10} height={10} rx={2} fill="var(--compute)" />
            <Label x={114} y={231} size={9.5} mono={false}>prefill is compute-bound → you pay per unit of FLOP/s</Label>
            <rect x={96} y={250} width={10} height={10} rx={2} fill="var(--memory)" />
            <Label x={114} y={259} size={9.5} mono={false}>decode is memory-bound → you pay per unit of bandwidth</Label>
            <rect x={96} y={278} width={10} height={10} rx={2} fill="var(--token)" />
            <Label x={114} y={287} size={9.5} mono={false}>
              {`this request is ${regime === 'prefill' ? 'prefill' : 'decode'}-priced (${(Math.max(st.prefillFrac, st.decodeFrac) * 100).toFixed(0)}% of its time)`}
            </Label>
          </g>
        </Fade>

        {/* ── the two headline numbers ── */}
        <Fade show={showMoney}>
          <g>
            <Label x={16} y={318} size={9.5} tone={3}>output tokens only</Label>
            <Label x={146} y={318} size={12} kind="token" bold>{`$${cOut.toFixed(2)}`}</Label>
            <Label x={210} y={318} size={9.5} tone={3}>per 1M</Label>
            <Label x={300} y={318} size={9.5} tone={3}>prompt + output</Label>
            <Label x={416} y={318} size={12} kind="text" bold>{`$${cBlend.toFixed(2)}`}</Label>
            <Label x={480} y={318} size={9.5} tone={3}>per 1M</Label>
            <Label x={16} y={336} size={9} tone={3}>
              The two differ because an output token is priced on the step it shares, while a prompt token is priced on its own.
            </Label>
          </g>
        </Fade>

        {/* ── the tensor-parallel case ── */}
        <Fade show={showTP}>
          <g>
            <rect x={0} y={38} width={640} height={282} rx={3} fill="var(--ground-2)" opacity={0.94} />
            <Label x={16} y={62} size={11} kind="error" bold>
              {`Llama-3.1-70B in ${wBytes === 2 ? 'BF16' : wBytes === 1 ? 'FP8' : 'INT4'} needs ${fmt.gb(70.6e9 * wBytes)} of weights`}
            </Label>
            <Label x={16} y={84} size={9.5} mono={false}>
              {`A ${fmt.int(gpuById(gpuId).mem / 1e9)} GB card, less a 4% reserve, cannot hold that — so the model is sharded across ${tp.cards} card${tp.cards === 1 ? '' : 's'}.`}
            </Label>
            <Label x={16} y={106} size={9.5} kind="memory" mono={false}>{tp.cards > 1 ? `the bill is now multiplied by ${tp.cards}` : 'on this card it fits in one — the bill is not multiplied'}</Label>
            {tp.cards > 1 && (
              <Label x={16} y={124} size={9.5} kind="comm" mono={false}>{`and each step is ${((tp.stepPenalty - 1) * 100).toFixed(0)}% slower for the all-reduces between shards`}</Label>
            )}
            <Label x={16} y={152} size={10.5} bold>The cheapest system for a 70B model is not the fastest one.</Label>
            <Label x={16} y={172} size={9.5} mono={false}>Compare a 4-bit copy on one card against BF16 split across several, and count every card you buy.</Label>
            <Label x={16} y={196} size={9.5} kind="token" mono={false}>
              {`INT4: ${fmt.gb(70.6e9 * 0.5)} of weights — fits an 80 GB card, one card, no communication tax.`}
            </Label>
            <Label x={16} y={216} size={9.5} kind="token" mono={false}>
              {`BF16: ${fmt.gb(70.6e9 * 2)} — two cards, double the bill, and a slower step on top.`}
            </Label>
            <Label x={16} y={244} size={9.5} tone={1} bold>Memory, not speed, decided this one.</Label>
            <Label x={16} y={264} size={9} tone={3}>
              When a model does not fit, the first question is never how fast the card is. It is how many cards you have to buy.
            </Label>
            <Label x={16} y={286} size={9} tone={3}>
              Choose a cheaper card above and watch the card count fall — that is the decision this step is about.
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}