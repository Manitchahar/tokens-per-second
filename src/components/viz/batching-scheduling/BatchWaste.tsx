import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Span, Label, Fade, Ruler, fmt } from '../lib/svg';
import { decodeStep, gpuById, modelById } from '../lib/perfModel';

// Llama-3.1-8B in BF16 on one H100. Every number on this page comes from perfModel,
// the same model the earlier modules used, so the trace looks like the ones you have
// already learned to read.

const model = modelById('llama8b');
const gpu = gpuById('h100');
const PROMPT = 200; // prompt tokens every request in the wave carries

const STEPS = [
  'One user at a time',
  'The wait is the cost',
  'Static batching',
  'The step barely moved',
  'Everyone finishes together',
  'Counting the waste',
  'Padding makes it worse',
  'The fix, previewed',
];
const CAPTIONS = [
  'The naive server answers one user at a time. The GPU spends its whole life reading the same 16 GB of weights for a single token.',
  'Eight requests arrived together, so seven of them are queued. Nobody is waiting for anything except each other.',
  'Static batching: collect them into one batch and run them together. The weights are read once for all eight.',
  'Going from one request per step to eight barely changes the step time, because the weight read is the same read. This is why batching works at all.',
  'But the batch cannot end until its slowest member ends. Short answers keep a slot for the rest of the run and produce nothing.',
  'Add it up: over a third of the slots the GPU paid for produced no token at all. That is the whole inefficiency, in one number.',
  'It is also structural: a dense batch is a rectangle, so every short answer is padded out to the longest one and the rows are masked.',
  'The fix is to re-form the batch every step, so a finished request is replaced on the very next step. Module 07.2 builds that.',
];

const X0 = 92;
const X1 = 628;
const MAXN = 16;

const frac = (x: number) => x - Math.floor(x);
/** A deterministic, pleasantly ragged spread of answer lengths around `avg`. */
const spread = (i: number, avg: number) => Math.max(8, Math.round(avg * (0.3 + 1.4 * frac(i * 0.6180339887))));

export default function BatchWaste() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [n, setN] = useState(8);
  const [avg, setAvg] = useState(300);

  const lens = useMemo(() => Array.from({ length: n }, (_, i) => spread(i, avg)), [n, avg]);
  const ctx = PROMPT + avg / 2;
  const t1 = decodeStep({ model, gpu, batch: 1, ctx }).t;
  const tN = decodeStep({ model, gpu, batch: n, ctx }).t;

  const sum = lens.reduce((a, b) => a + b, 0);
  const maxL = Math.max(...lens);
  const seqT = sum * t1;
  const staticT = maxL * tN;
  const perfectT = Math.ceil(sum / n) * tN; // what perfectly packed continuous batching would cost
  const scale = Math.max(seqT, staticT);

  // where each request starts in each policy
  const seqStart: number[] = [];
  let acc = 0;
  for (const L of lens) {
    seqStart.push(acc);
    acc += L * t1;
  }

  const g = useTweens({
    seq: seqT / scale,
    stat: staticT / scale,
    perf: perfectT / scale,
    t1: t1 / (t1 * 1.6),
    tN: tN / (t1 * 1.6),
  });

  const wasted = n * maxL - sum;
  const wasteFrac = wasted / (n * maxL);
  const idleFrac = 1 - staticT / seqT;
  const seqMode = s.step <= 1;
  const px = (sec: number) => Math.min(X1 - X0, (sec / scale) * (X1 - X0));

  return (
    <Stage
      rootRef={ref}
      title="Llama-3.1-8B on one H100: one-at-a-time vs static batching"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', ['idle', 'queued / waiting'], ['error', 'slot paid for, token wasted']]}
      {...s}
      readout={
        <Readout items={[
          { label: seqMode ? 'Wave finished (one at a time)' : 'Wave finished (static batch)', value: fmt.ms(seqMode ? seqT : staticT), kind: 'compute', note: `${sum} tokens wanted in ${n} requests` },
          { label: 'Decode step', value: `${(seqMode ? t1 : tN).toFixed(1)} ms`, kind: 'memory', note: `batch ${seqMode ? 1 : n}` },
          { label: 'Throughput', value: `${fmt.int((seqMode ? sum / seqT : sum / staticT))} tok/s`, kind: 'token', note: 'across all requests' },
          { label: seqMode ? 'GPU idle waiting' : 'Slots wasted', value: seqMode ? `${(idleFrac * 100).toFixed(0)}%` : `${(wasteFrac * 100).toFixed(0)}%`, kind: 'error', note: seqMode ? 'of the wave, nobody runs' : 'of the run produced nothing' },
        ]} />
      }
      controls={
        <>
          <Slider label="Requests in the wave" value={n} min={2} max={MAXN} onChange={setN} format={(v) => String(v)} kind="token" />
          <Slider label="Average answer length" value={avg} min={60} max={1000} step={20} onChange={setAvg} format={(v) => `${v} tokens`} kind="memory" />
        </>
      }
    >
      <svg viewBox="0 0 640 358" role="img" aria-label="A timeline of a wave of requests served two ways: one at a time, or all together as one static batch, with the step time of each policy and the fraction of batch slots wasted.">
        <defs>
          <clipPath id="bw-gantt"><rect x={X0} y={44} width={X1 - X0} height={196} /></clipPath>
        </defs>

        {/* mode banner */}
        <Label x={X0} y={16} size={11} kind={seqMode ? 'idle' : 'token'} bold>
          {seqMode ? 'Policy A · serve one request at a time' : 'Policy B · one static batch, all at once'}
        </Label>
        <Label x={X1} y={16} anchor="end" tone={3} size={9.5}>{`wave of ${n} requests arrives at t = 0`}</Label>

        {/* waiting queue */}
        <text x={X0 - 10} y={36} textAnchor="end" className="v-track-label">Queue</text>
        {Array.from({ length: n }, (_, i) => (
          <rect key={i} x={X0 + i * 11} y={26} width={9} height={11} rx={1.5}
            fill={seqMode && i === 0 ? 'var(--token)' : 'url(#tps-hatch)'} stroke={seqMode && i === 0 ? 'var(--token)' : 'var(--idle)'} strokeWidth={1} />
        ))}

        {/* the Gantt */}
        <g clipPath="url(#bw-gantt)">
          {lens.map((L, i) => {
            const y = 46 + i * 12;
            const seqX = X0 + (seqStart[i] / scale) * (X1 - X0);
            const seqW = Math.max(1.5, (L * t1 / scale) * (X1 - X0));
            const statW = Math.max(1.5, (staticT / scale) * (X1 - X0));
            const perfW = Math.max(1.5, (perfectT / scale) * (X1 - X0));
            const tailW = statW - (L / maxL) * statW;
            return (
              <g key={i}>
                <text x={X0 - 10} y={y + 7} textAnchor="end" className="v-label t3" style={{ fontSize: 9 }}>{`U${i + 1}`}</text>
                {seqMode ? (
                  <Span x={seqX} y={y} w={seqW} h={9} kind="token" opacity={0.9} />
                ) : (
                  <>
                    <Span x={X0} y={y} w={statW} h={9} kind="token" />
                    {s.step >= 4 && tailW > 0.8 && (
                      <Span x={X0 + (L / maxL) * statW} y={y} w={tailW} h={9} kind="error" opacity={0.75} />
                    )}
                  </>
                )}
                {s.step >= 6 && !seqMode && (
                  <rect x={X0} y={y} width={statW} height={9} rx={2} fill="none" stroke="var(--text-3)" strokeDasharray="2 3" />
                )}
              </g>
            );
          })}
          {s.step >= 7 && !seqMode && (
            <g>
              <rect x={X0} y={44} width={(perfectT / scale) * (X1 - X0)} height={196} fill="none" stroke="var(--token)" strokeWidth={1.4} strokeDasharray="5 4" />
              <Label x={X0 + (perfectT / scale) * (X1 - X0) + 6} y={54} size={10} kind="token">same work, perfectly packed</Label>
            </g>
          )}
        </g>
        {s.step === 6 && !seqMode && (
          <text x={(X0 + X1) / 2} y={232} textAnchor="middle" className="v-label" style={{ fill: 'var(--text-3)', fontSize: 10 }}>
            one dense batch: [n × maxLen] — the dashed outline is the padding
          </text>
        )}

        {/* shared time ruler */}
        <Ruler x={X0} y={254} w={X1 - X0}
          ticks={[{ at: 0, label: '0' }, { at: 0.25 }, { at: 0.5 }, { at: 0.75 }, { at: 1, label: fmt.ms(scale) }]}
          label={seqMode ? 'one-at-a-time total' : 'static-batch total'} />

        {/* step time: the thing batching does not make worse */}
        <Fade show={s.step >= 3}>
          <text x={X0 - 10} y={284} textAnchor="end" className="v-track-label">Step</text>
          <rect x={X0} y={274} width={(X1 - X0) * g.t1} height={11} rx={2} fill="none" stroke="var(--rule-2)" />
          <Span x={X0} y={274} w={(X1 - X0) * g.t1} h={11} kind="memory" label={`batch 1 · ${(t1 * 1000).toFixed(1)} ms`} />
          <rect x={X0} y={290} width={(X1 - X0) * g.tN} height={11} rx={2} fill="none" stroke="var(--rule-2)" />
          <Span x={X0} y={290} w={(X1 - X0) * g.tN} h={11} kind="memory" label={`batch ${n} · ${(tN * 1000).toFixed(1)} ms`} />
          <Label x={X1} y={312} anchor="end" size={10} kind="token">{`${n}× the tokens for ${(tN / t1).toFixed(2)}× the time`}</Label>
        </Fade>

        {/* slot accounting */}
        <Fade show={s.step >= 5 && !seqMode}>
          <text x={X0 - 10} y={336} textAnchor="end" className="v-track-label">Slots</text>
          <rect x={X0} y={326} width={X1 - X0} height={12} rx={2} fill="none" stroke="var(--rule-2)" />
          <Span x={X0} y={326} w={((X1 - X0) * (1 - wasteFrac))} h={12} kind="token" label={`${sum} useful tokens`} />
          <Span x={X0 + (X1 - X0) * (1 - wasteFrac)} y={326} w={(X1 - X0) * wasteFrac} h={12} kind="error" label={`${wasted} wasted`} labelTone="light" />
          <Label x={X1} y={350} anchor="end" tone={3} size={9.5}>{`${n} slots × ${maxL} steps = ${n * maxL} slot-steps`}</Label>
        </Fade>
        <Fade show={s.step < 3 && seqMode}>
          <Label x={X1} y={330} anchor="end" tone={3} size={10}>{`${(idleFrac * 100).toFixed(0)}% of this wall-clock time has exactly one request running`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}