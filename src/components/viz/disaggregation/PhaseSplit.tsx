import { useRef, useState } from 'react';
import { Stage, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade, Move, Track, fmt } from '../lib/svg';
import {
  m8, decodeStep, maxBatch, prefillPerToken, prefillIntensity, decodeIntensity, ridgePoint,
} from './disagg';
import { gpuById } from '../lib/perfModel';

const STEPS = [
  'One GPU, one request',
  'A batch forms',
  'Decode is memory-bound',
  'Prefill is compute-bound',
  'The ridge point',
  'Two jobs, one bottleneck',
  'The repair',
  'What it costs',
];
const CAPTIONS = [
  'A single user sends a prompt. The engine prefills it, then decodes one token at a time. One request keeps the memory units busy and leaves the math units almost entirely idle.',
  'Requests pile up and the engine batches them, as in Module 07. Batching helps decode a lot, because the weights are read once for the whole batch instead of once per user.',
  'Even at batch 64, a decode step is 93% a memory-bandwidth story. The tensor cores are asked for 9% of the work they could do in the time the step takes.',
  'Prefill flips it. Thousands of rows reuse every weight many times, so the step becomes pure arithmetic and the bandwidth goes idle instead.',
  'Every chip has a ridge point where FLOPs and bytes balance: H100 about 295 FLOP per byte. Above the ridge a job is compute-bound, below it memory-bound.',
  'Here is the whole problem in one picture. Prefill sits far above the ridge, decode far below it, and both are scheduled onto the same processor at different times.',
  'If the two phases want opposite bottlenecks, stop giving them the same bottleneck. Put prefill on a pool sized for FLOPs, decode on a pool sized for bandwidth and KV memory.',
  'The repair is not free. You now ship the KV cache across a network between the pools, and you run two schedulers instead of one. Both costs are payable because both are bounded and known.',
];

// Layout
const X0 = 96;
const X1 = 616;
const RIDGE_Y = 92; // the roofline ridge line
const TOP = 30;
const BOT = 196;

export default function PhaseSplit() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [ctx, setCtx] = useState(4096);
  const [batch, setBatch] = useState(64);
  const gpu = gpuById('h100');
  const ridge = ridgePoint(gpu);
  const dec = decodeStep(batch, ctx);
  const maxB = maxBatch(ctx);

  // Intensity axis: 0.1 FLOP/byte .. 4000, log scale.
  const LO = 0.1;
  const HI = 4000;
  const iy = (v: number) => BOT - (Math.log10(Math.max(LO, v)) - Math.log10(LO)) / (Math.log10(HI) - Math.log10(LO)) * (BOT - TOP);
  const ix = (b: number) => X0 + Math.log10(Math.min(maxB, b)) / Math.log10(maxB) * (X1 - X0);

  const pfInt = prefillIntensity(512);
  const dcInt = decodeIntensity(batch, ctx);

  return (
    <Stage
      rootRef={ref}
      title="Llama-3.1-8B on an H100: the two phases sit on opposite sides of the ridge"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', ['error', 'wasted capacity'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Decode step', value: `${(dec.t * 1e3).toFixed(1)} ms`, kind: 'memory', note: `at batch ${batch}, ${fmt.int(ctx)} ctx` },
          { label: 'Decode arithmetic intensity', value: `${dcInt.toFixed(1)} FLOP/B`, kind: 'memory', note: `ridge is ${ridge.toFixed(0)} — ${(dec.memShare * 100).toFixed(0)}% memory-bound` },
          { label: 'Prefill intensity', value: `${pfInt.toFixed(0)} FLOP/B`, kind: 'compute', note: 'at 512 rows — above the ridge' },
          { label: 'Tensor cores idle in decode', value: `${((1 - dec.computeShare) * 100).toFixed(0)}%`, kind: 'error', note: 'the opportunity this module takes' },
        ]} />
      }
      controls={
        <>
          <Slider
            label="Decode batch size — how many users share one weight read"
            value={batch}
            min={1}
            max={maxB}
            onChange={setBatch}
            format={(v) => `${fmt.int(v)}${v >= maxB ? '  (KV full)' : ''}`}
            kind="memory"
          />
          <Slider
            label="Context length per user"
            value={ctx}
            min={512}
            max={32768}
            log
            onChange={setCtx}
            format={(v) => `${fmt.int(v)} tokens`}
            kind="text"
          />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A roofline chart for Llama-3.1-8B on an H100. Arithmetic intensity runs from 0.1 to 4000 FLOP per byte. A memory-bound prefill marker sits high on the compute ceiling near 500 FLOP per byte, while decode markers sit far to the right on the low, flat bandwidth slope, showing decode is memory-bound even at large batch.">
        {/* axes */}
        <line x1={X0} x2={X1} y1={BOT} y2={BOT} className="v-rule" />
        <line x1={X0} x2={X0} y1={TOP} y2={BOT} className="v-rule" />
        <Label x={X0 - 6} y={BOT + 16} anchor="end" tone={3} size={9}>0.1</Label>
        <Label x={X0 - 6} y={iy(1) + 3} anchor="end" tone={3} size={9}>1</Label>
        <Label x={X0 - 6} y={iy(100) + 3} anchor="end" tone={3} size={9}>100</Label>
        <Label x={X0 - 6} y={TOP + 8} anchor="end" tone={3} size={9}>4k</Label>
        <Label x={(X0 + X1) / 2} y={BOT + 30} anchor="middle" tone={3} size={9.5} mono={false}>
          Arithmetic intensity (FLOP per byte moved) →
        </Label>
        <Label x={X0 - 6} y={TOP - 6} anchor="end" tone={3} size={9}>FLOP/s</Label>

        {/* the roofline: memory slope then compute ceiling */}
        <path
          d={`M${X0},${iy(ridge)} L${X1},${iy(ridge)}`}
          stroke="var(--memory)" strokeWidth={2} fill="none"
        />
        <path
          d={`M${X0},${BOT} L${X0},${iy(ridge)}`}
          stroke="var(--memory)" strokeWidth={2} fill="none"
        />
        <path
          d={`M${ix(1)},${BOT} L${ix(maxB)},${iy(decodeIntensity(maxB, ctx))}`}
          stroke="var(--memory)" strokeWidth={2} fill="none" opacity={0.5}
        />

        {/* ridge label */}
        <Fade show={s.step >= 4}>
          <line x1={X0} x2={X1} y1={iy(ridge)} y2={iy(ridge)} stroke="var(--text-3)" strokeDasharray="4 4" />
          <Label x={X1} y={iy(ridge) - 6} anchor="end" size={9.5} tone={2}>
            {`ridge ≈ ${ridge.toFixed(0)} FLOP/byte`}
          </Label>
          <Label x={X0 + 8} y={BOT - 8} size={9.5} kind="memory">bandwidth-bound</Label>
          <Label x={X0 + 8} y={iy(ridge) - 8} size={9.5} kind="compute">compute-bound</Label>
        </Fade>

        {/* prefill marker */}
        <Fade show={s.step >= 1}>
          <Move x={ix(1) + 6} y={iy(pfInt)}>
            <circle r={6} fill="var(--compute)" stroke="var(--ground-2)" strokeWidth={2} />
            <Label x={14} y={-6} size={10.5} kind="compute" bold>prefill (512 rows)</Label>
            <Label x={14} y={7} tone={3} size={9}>{`${pfInt.toFixed(0)} FLOP/byte`}</Label>
          </Move>
        </Fade>

        {/* decode marker — moves with batch */}
        <Fade show={s.step >= 2}>
          <Move x={ix(batch)} y={iy(dcInt)}>
            <circle r={6} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={2} />
            <Label x={0} y={-14} anchor="middle" size={10.5} kind="memory" bold>{`decode, batch ${batch}`}</Label>
            <Label x={0} y={20} anchor="middle" tone={3} size={9}>{`${dcInt.toFixed(1)} FLOP/byte`}</Label>
          </Move>
        </Fade>

        {/* the waste: the gap under the ceiling that decode leaves */}
        <Fade show={s.step >= 2}>
          <rect
            x={ix(batch) - 4} y={iy(dcInt)} width={Math.max(8, ix(maxB) - ix(batch))} height={Math.max(0, iy(ridge) - iy(dcInt))}
            fill="url(#tps-hatch-red)" opacity={0.25}
          />
          <Label x={ix(batch) + 10} y={(iy(dcInt) + iy(ridge)) / 2} size={9.5} kind="error">
            {`${((1 - dec.computeShare) * 100).toFixed(0)}% of the math units unused`}
          </Label>
        </Fade>

        {/* step 6: the two pools */}
        <Fade show={s.step >= 6}>
          <g>
            <rect x={X0} y={250} width={240} height={96} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x={X0 + 12} y={270} size={11} kind="compute" bold>Prefill pool</Label>
            <Label x={X0 + 12} y={287} tone={3} size={9.5}>sized by FLOP/s</Label>
            <Label x={X0 + 12} y={303} tone={3} size={9.5}>{`${(1 / prefillPerToken()).toFixed(0)} prompt-tok/s per GPU`}</Label>
            <Label x={X0 + 12} y={319} tone={3} size={9.5}>short bursts, many GPUs</Label>
            <Label x={X0 + 12} y={335} tone={3} size={9.5}>almost no KV cache held</Label>

            <rect x={X0 + 260} y={250} width={240} height={96} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x={X0 + 272} y={270} size={11} kind="memory" bold>Decode pool</Label>
            <Label x={X0 + 272} y={287} tone={3} size={9.5}>sized by bandwidth &amp; KV memory</Label>
            <Label x={X0 + 272} y={303} tone={3} size={9.5}>{`${dec.total.toFixed(0)} tok/s per GPU`}</Label>
            <Label x={X0 + 272} y={319} tone={3} size={9.5}>long streams, steady load</Label>
            <Label x={X0 + 272} y={335} tone={3} size={9.5}>hundreds of KV caches held</Label>
          </g>
        </Fade>

        {/* step 7: what it costs */}
        <Fade show={s.step >= 7}>
          <g>
            <rect x={X0} y={250} width={500} height={96} rx={5} fill="none" stroke="var(--error)" strokeDasharray="4 4" />
            <Label x={X0 + 12} y={270} size={11} kind="error" bold>Paid for</Label>
            <Label x={X0 + 12} y={290} tone={2} size={10} mono={false}>
              1 · shipping {(fmt.int(ctx * 131072 / 1e6))} MB of KV per request across a link
            </Label>
            <Label x={X0 + 12} y={308} tone={2} size={10} mono={false}>
              2 · two schedulers, two failure domains, one harder failure mode
            </Label>
            <Label x={X0 + 12} y={326} tone={2} size={10} mono={false}>
              3 · the transfer joins TTFT, so it is paid by the user who is already waiting
            </Label>
          </g>
        </Fade>

        {/* step 0: the lonely single request */}
        <Fade show={s.step === 0}>
          <g>
            <Track x={X0} y={230} w={X1 - X0} h={22} label="1 req">
              <Span x={X0} y={230} w={(prefillPerToken() * 512 * 1e3)} h={22} kind="compute" opacity={0.9} />
              <Label x={X0 + 6} y={245} size={10} kind="compute" labelTone="dark">
                {`prefill 512 tok = ${(prefillPerToken() * 512 * 1e3).toFixed(0)} ms`}
              </Label>
            </Track>
            <Label x={X0} y={272} tone={2} size={10.5} mono={false}>
              One request: the bandwidth term is pure weight-reading, so the step is memory-bound.
            </Label>
            <Label x={X0} y={290} tone={3} size={10} mono={false}>
              {`bandwidth term ${(decodeStep(1, ctx).tWeights * 1e3).toFixed(1)} ms of a ${(decodeStep(1, ctx).t * 1e3).toFixed(1)} ms step — arithmetic intensity ${decodeIntensity(1, ctx).toFixed(2)} FLOP/byte`}
            </Label>
            <Label x={X0} y={310} tone={3} size={10} mono={false}>
              The math units are there, doing almost nothing. Every extra concurrent request is free capacity.
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
