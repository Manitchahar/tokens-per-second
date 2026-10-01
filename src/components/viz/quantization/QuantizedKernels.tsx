import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Span, Label, Arrow, Fade, fmt } from '../lib/svg';
import { llama8b, h100, decodeStepQuant, kvBytes, weightBytes, metaFracInt } from './quant';

/** The serving shape this visual reasons about. */
const BATCH = 32;
const CTX = 4096;
/** Share of GPU memory we allow to weights + KV, leaving room for activations. */
const BUDGET = 0.8;

const STEPS = [
  'Where the dequantization has to happen',
  'The unfused path, and why it is slower',
  'The fused path: dequantize inside the GEMM',
  'The KV cache is a separate pipeline',
  'How far can KV go before quality breaks',
  'What this buys a decode step',
];
const CAPTIONS = [
  'A quantized GEMM has to turn integers back into floats somewhere. Where it happens decides whether you saved anything. There are only two places that work.',
  'Unfused: unpack the 4-bit weights into a BF16 tensor in HBM, then call an ordinary matmul. You paid to write and read the BF16 copy again. Disk saved; speed gone, and often negative.',
  'Fused: the kernel loads a packed tile into shared memory, widens it to BF16 in registers, and feeds the tensor cores without a BF16 copy ever reaching HBM. This is the only version worth shipping.',
  'The KV cache is a different pipeline. It is written once per token and read once per step, with a different access pattern and a different error lifetime — so it takes different settings.',
  'Eight bits is safe. Four bits needs per-token or per-head scales and starts to cost real quality, especially at long context where the error accumulates over every cached token.',
  'Now add it up. Weights and KV are independent levers in the same byte count, and this is what a config that pushes both actually buys.',
];

/** Users that fit, from weights + KV memory against the budget. */
const usersFit = (wBits: number, kvBits: number) => {
  const free = h100.mem * BUDGET - weightBytes(llama8b, wBits, metaFracInt(wBits, 128));
  return Math.max(0, Math.floor(free / kvBytes(CTX, kvBits)));
};

export default function QuantizedKernels() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [wBits, setWBits] = useState(4);
  const [kvBits, setKvBits] = useState(16);

  const meta = metaFracInt(wBits, 128);
  const step = useMemo(() => decodeStepQuant({ batch: BATCH, ctx: CTX, wBits, kvBits, metaFrac: meta }), [wBits, kvBits, meta]);
  const bf16 = useMemo(() => decodeStepQuant({ batch: BATCH, ctx: CTX, wBits: 16, kvBits: 16 }), []);
  const bytesTotal = step.weights + step.kvRead;

  /* Byte-movement timelines, all in ms at ~80% of achievable bandwidth. */
  const BW = h100.bw * 0.8;
  const msFor = (bytes: number) => (bytes / BW) * 1e3;
  const bf16Ms = msFor(bf16.weights + bf16.kvRead);
  const fusedMs = msFor(bytesTotal);
  const unfusedMs = msFor(bytesTotal + step.weights); // the BF16 copy written and read back

  const X0 = 116;
  const GRID = 424;
  const span = Math.max(unfusedMs, bf16Ms) * 1.06;
  const px = (ms: number) => (ms / span) * GRID;
  const tFused = useTween(fusedMs, 600);
  const tUnfused = useTween(unfusedMs, 600);

  const users = usersFit(wBits, kvBits);
  const usersBf16 = usersFit(16, 16);
  const speedup = bf16.t / step.t;
  const KV_SCALE = (bf16.weights + bf16.kvRead) / span / (BW / 1e3);

  return (
    <Stage
      rootRef={ref}
      title="Quantized kernels & KV cache: where the dequantization must happen"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', 'token', 'error', 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Decode step', value: `${(step.t * 1e3).toFixed(2)} ms`, kind: 'compute', note: `W${wBits} + KV${kvBits} · ${BATCH} users @ ${fmt.int(CTX)} ctx` },
          { label: 'Versus BF16', value: `${speedup.toFixed(2)}×`, kind: speedup > 1.3 ? 'token' : 'text', note: 'both byte terms shrank' },
          { label: 'Bytes per step', value: fmt.gb(bytesTotal), kind: 'memory', note: `${(step.weights / 1e9).toFixed(1)} GB weights + ${(step.kvRead / 1e9).toFixed(1)} GB KV` },
          { label: 'Users on one H100', value: `${users}`, kind: users > usersBf16 * 2 ? 'token' : 'memory', note: `${kvBits}-bit KV · was ${usersBf16} at BF16` },
        ]} />
      }
      controls={
        <>
          <Slider label="Weight bits" value={wBits} min={4} max={16} step={2} onChange={setWBits} format={(v) => `W${v}${v === 4 ? ' (group 128)' : ''}`} kind="compute" />
          <Slider label="KV cache bits" value={kvBits} min={4} max={16} step={2} onChange={setKvBits} format={(v) => `KV${v}`} kind="memory" />
        </>
      }
    >
      <svg viewBox="0 0 640 388" role="img" aria-label="Three timelines comparing byte movement for one decode step: the BF16 baseline, a fused 4-bit GEMM that dequantizes inside the kernel, and an unfused path that writes and re-reads a BF16 copy. Below, the KV cache read size and the number of users that fit on an H100 at the chosen precision.">
        {/* ── the GEMM timelines ── */}
        <Fade show={st <= 3}>
          <Label x={0} y={20} tone={2} size={11} bold>Bytes moved per decode step</Label>
          <Label x={640} y={20} anchor="end" tone={3} size={9.5}>{`${h100.name} · ${(BW / 1e12).toFixed(2)} TB/s achieved · time is bytes ÷ bandwidth`}</Label>

          <Label x={108} y={46} anchor="end" tone={3} size={9.5}>BF16</Label>
          <rect x={X0} y={36} width={GRID} height={15} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={X0} y={36} w={px(bf16Ms)} h={15} kind="memory" opacity={0.5} label={`${bf16Ms.toFixed(1)} ms`} />

          <Label x={108} y={72} anchor="end" tone={3} size={9.5}>fused W{wBits}</Label>
          <rect x={X0} y={62} width={GRID} height={15} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={X0} y={62} w={px(tFused)} h={15} kind="token" label={`${tFused.toFixed(1)} ms`} />

          <Label x={108} y={98} anchor="end" tone={3} size={9.5}>unfused</Label>
          <rect x={X0} y={88} width={GRID} height={15} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={X0} y={88} w={px(tUnfused)} h={15} kind="error" label={`${tUnfused.toFixed(1)} ms`} />
          <Fade show={st >= 1}>
            <Label x={X0 + px(tUnfused) + 6} y={99} size={9.5} kind="error">{`+${(tUnfused - tFused).toFixed(1)} ms of pure copy`}</Label>
          </Fade>
        </Fade>

        {/* ── inside the fused kernel ── */}
        <Fade show={st === 2}>
          <g>
            <Label x={0} y={130} tone={2} size={11} bold>Inside the fused kernel</Label>
            {['load packed tile', 'widen to BF16', 'tensor-core matmul'].map((name, i) => (
              <g key={name}>
                <rect x={116 + i * 172} y={142} width={150} height={26} rx={3} fill={i === 2 ? 'var(--token)' : 'var(--compute)'} opacity={0.92} />
                <Label x={191 + i * 172} y={159} anchor="middle" size={10} style={{ fill: '#0e1116' }}>{name}</Label>
                {i < 2 && <Arrow x1={270 + i * 172} x2={286 + i * 172} y1={155} y2={155} kind="memory" />}
              </g>
            ))}
            <Label x={116} y={186} tone={3} size={9.5}>The widened values live in registers only. Nothing writes a BF16 copy back to HBM — that single fact is the whole speedup.</Label>
          </g>
        </Fade>
        <Fade show={st === 1}>
          <Label x={0} y={132} tone={2} size={11} bold>Why unfused is worse than BF16</Label>
          <Label x={0} y={152} tone={3} size={10}>You read 4-bit weights, write a BF16 copy, read that copy, compute, then throw it away.</Label>
          <Label x={0} y={170} tone={3} size={10}>The copy is the same size as the original model. Disk saved, bandwidth burned.</Label>
        </Fade>

        {/* ── the KV pipeline ── */}
        <Fade show={st >= 3}>
          <g>
            <Label x={0} y={216} tone={2} size={11} bold>{`The KV cache is a separate pipeline: ${kvBits}-bit, ${BATCH} users at ${fmt.int(CTX)} tokens`}</Label>

            <Label x={108} y={242} anchor="end" tone={3} size={9.5}>read per step</Label>
            <rect x={X0} y={232} width={GRID} height={15} rx={2} fill="none" stroke="var(--rule)" />
            <Span x={X0} y={232} w={Math.max(4, step.kvRead * KV_SCALE)} h={15} kind="memory" label={`${(step.kvRead / 1e9).toFixed(1)} GB`} />
            <Label x={X0 + Math.max(4, step.kvRead * KV_SCALE) + 6} y={244} size={9} kind="memory">{kvBits < 16 ? `was ${(bf16.kvRead / 1e9).toFixed(1)} GB at BF16` : 'halve this and the step halves its other half'}</Label>

            <Label x={108} y={272} anchor="end" tone={3} size={9.5}>users on one H100</Label>
            <rect x={X0} y={262} width={GRID} height={15} rx={2} fill="none" stroke="var(--rule)" />
            <Span x={X0} y={262} w={Math.min(1, users / 400) * GRID} h={15} kind={users > usersBf16 * 2 ? 'token' : 'memory'} label={`${users} concurrent users`} />
            <Label x={X0 + Math.min(1, users / 400) * GRID + 6} y={274} size={9} tone={3}>{`was ${usersBf16} at BF16`}</Label>

            <Label x={108} y={302} anchor="end" tone={3} size={9.5}>quality cost</Label>
            <rect x={X0} y={292} width={GRID} height={15} rx={2} fill="none" stroke="var(--rule)" />
            <Span
              x={X0}
              y={292}
              w={kvBits <= 4 ? GRID : kvBits <= 8 ? GRID * 0.45 : GRID * 0.12}
              h={15}
              kind={kvBits <= 4 ? 'error' : 'compute'}
              label={kvBits <= 4 ? 'real cost at long context' : kvBits <= 8 ? 'mild; usually undetectable' : 'negligible'}
            />
          </g>
        </Fade>

        {/* ── the sum ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={0} y={338} tone={2} size={11} bold>Both levers, one byte count</Label>
            <Label x={0} y={358} tone={1} size={10}>
              {`W${wBits} + KV${kvBits}: ${(step.t * 1e3).toFixed(2)} ms per step versus ${(bf16.t * 1e3).toFixed(2)} ms for BF16 — a ${speedup.toFixed(2)}× step`}
            </Label>
            <Label x={0} y={374} tone={3} size={10}>
              {`and ${users} concurrent users instead of ${usersBf16}, from the same 80 GB card.`}
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}