import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';
import {
  BMODELS, bmodelById, gpuById, linkById,
  tpDecodeStep, tpPrefillTime, tpDegrees, tpMessageBytes, ringFactor,
  type BigModel, type Link,
} from './parallel';

const STEPS = [
  'One layer, four GPUs',
  'Column-parallel: split the output',
  'Row-parallel: split the input',
  'One all-gather, one all-reduce',
  'The all-reduce is the price',
  'Attention splits the same way',
  'Memory divides by t',
  'Time does not',
];

const CAPTIONS = [
  'A transformer layer is a big block: attention, then an MLP that expands to 3.5x the hidden size and projects back. Tensor parallelism cuts those two big matrices into pieces and gives each GPU one piece.',
  'The first MLP matrix, W1, is split along its output dimension. Each GPU computes a slice of the expanded activations and no communication is needed — the shards already are the answer.',
  'The second matrix, W2, is split along its input dimension. Each GPU multiplies its slice of activations by its slice of W2 and produces a partial sum of the final hidden state.',
  'Partial sums have to be added up. So the pair is: an all-gather going in (to rebuild the full activations) and an all-reduce coming out (to sum the partials). This is the Megatron block.',
  'The all-reduce moves 2(P-1)/P of the full hidden state per layer per direction. At t=4 that is 1.5x the hidden state, twice per layer. Everything else about this block is free.',
  'Attention splits identically: Q, K and V are column-parallel (each GPU gets whole heads), the attention output projection is row-parallel. One difference — the KV cache divides by t too.',
  'Weights fall to 1/t and the KV cache falls to 1/t, because each GPU holds whole KV heads. For Llama-3.1-70B on an 80 GB card, t=4 is the difference between fitting and not fitting.',
  'Memory time falls as 1/t. Compute time falls as 1/t. Communication time does not fall at all. So speedup is sublinear, and the curve flattens exactly where the wire starts to hurt.',
];

const GPUS = [
  { id: 'h100', label: 'H100 SXM' },
  { id: 'a100', label: 'A100 80GB' },
];
const LINKS = [
  { id: 'nvlink4', label: 'NVLink 4' },
  { id: 'pcie4', label: 'PCIe 4.0' },
  { id: 'ib', label: 'InfiniBand' },
];

const CARD_GAP = 12;
const CARD_X0 = 24;
const CARD_AREA = 592; // px available for the whole row of cards

/** Card width so that t cards always fit the row, whatever t is. */
const cardW = (t: number) => Math.max(46, (CARD_AREA - (Math.max(1, t) - 1) * CARD_GAP) / Math.max(1, t));
const rowW = (t: number) => cardW(t) * Math.max(1, t) + (Math.max(1, t) - 1) * CARD_GAP;

export default function TensorParallelBlock() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama70b');
  const [gpuId, setGpuId] = useState('h100');
  const [linkId, setLinkId] = useState('nvlink4');
  const [t, setT] = useState(4);
  const [batch, setBatch] = useState(32);
  const [phase, setPhase] = useState<'decode' | 'prefill'>('decode');

  const model: BigModel = bmodelById(modelId);
  const gpu = gpuById(gpuId);
  const link: Link = linkById(linkId);
  const degs = tpDegrees(model, 8);
  const tEff = Math.max(1, Math.min(t, Math.max(...degs)));

  const dec = tpDecodeStep({ model, gpu, P: tEff, link, batch });
  const pre = tpPrefillTime({ model, gpu, P: tEff, link, tokens: batch });
  const msgBytes = tpMessageBytes(model, batch);
  const commBytesPerPass = ringFactor(tEff) * msgBytes;

  const speedup = phase === 'decode' ? dec.speedup : (2 * model.params * batch) / (gpu.flops * 0.55) / pre.tCompute;
  const speedupT = useTween(speedup, 500);

  // chart: speedup for every legal t
  const chart = degs.map((p) => {
    if (phase === 'decode') return { p, v: tpDecodeStep({ model, gpu, P: p, link, batch }).speedup };
    const r = tpPrefillTime({ model, gpu, P: p, link, tokens: batch });
    return { p, v: (2 * model.params * batch) / (gpu.flops * 0.55) / r.tCompute };
  });
  const maxSpeed = Math.max(...chart.map((c) => Math.max(c.v, c.p)), 2);

  const wBytes = model.params * 2;
  const kvFull = batch * 4096 * (2 * model.layers * model.kvHeads * model.headDim * 2);
  const perGpu = wBytes / tEff + kvFull / tEff;

  const commPct = phase === 'decode' ? dec.tComm / dec.t : pre.overheadFrac;

  return (
    <Stage
      rootRef={ref}
      title="Tensor parallelism: the Megatron block, and what t=4 actually buys"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'this GPU’s shard'],
        ['comm', 'the collective'],
        ['memory', 'KV cache'],
        ['error', 'idle / waste'],
        'text',
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Model', value: model.name, kind: 'text', note: `${(model.params / 1e9).toFixed(1)}B, hidden ${fmt.int(model.hidden)}` },
          { label: 'TP degree', value: `${tEff}`, kind: 'compute', note: `legal: ${degs.join(', ')}` },
          { label: 'Weights per GPU', value: fmt.gb(wBytes / tEff), kind: 'memory', note: `from ${fmt.gb(wBytes)}` },
          { label: 'Speedup', value: `${speedupT.toFixed(2)}x`, kind: speedupT / tEff > 0.85 ? 'token' : 'error', note: `${((speedupT / tEff) * 100).toFixed(0)}% of ideal` },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Model"
            value={modelId}
            options={BMODELS.map((m) => ({ value: m.id, label: m.name.replace('Llama-3.1-', 'L3.1-').replace('Qwen2.5-', 'Q2.5-') }))}
            onChange={(v) => { setModelId(v); setT(tpDegrees(bmodelById(v), 8).reduce((a, b) => Math.abs(b - 4) < Math.abs(a - 4) ? b : a, 1)); }}
          />
          <Segmented label="GPU" value={gpuId} options={GPUS} onChange={setGpuId} />
          <Segmented label="Interconnect" value={linkId} options={LINKS} onChange={setLinkId} />
          <Slider label={`TP degree t (max ${Math.max(...degs)})`} kind="compute" value={t} min={1} max={8} onChange={setT} format={(v) => `${Math.min(v, Math.max(...degs))}`} />
          <Slider label="Batch (decode) / tokens (prefill)" kind="memory" value={batch} min={1} max={512} log onChange={setBatch} format={(v) => `${fmt.int(v)}`} />
          <Segmented label="Phase" value={phase} options={[{ value: 'decode', label: 'Decode' }, { value: 'prefill', label: 'Prefill' }]} onChange={setPhase} />
        </>
      }
    >
      <svg viewBox="0 0 640 384" role="img" aria-label="A transformer MLP block split across a group of GPUs: the first matrix split by output columns with no communication, the second split by input rows producing partial sums, one all-gather going in and one all-reduce coming out. Below, a chart of speedup against tensor-parallel degree compared to the ideal line, and a breakdown of memory, compute and communication time.">
        {/* ── the block ───────────────────────────────────────── */}
        <Label x={24} y={22} tone={2} size={11} bold>{`One MLP block on ${tEff} GPU${tEff > 1 ? 's' : ''}`}</Label>
        <Label x={616} y={22} anchor="end" tone={3} size={9.5}>{`hidden ${fmt.int(model.hidden)} → ${fmt.int(model.inter)} → ${fmt.int(model.hidden)}`}</Label>

        {(() => {
          const n = Math.max(1, tEff);
          const CW = cardW(n);
          const xInW = n > 1 ? 86 : 132;      // shrink the input box when cards are wide
          return (
            <>
              <Fade show={s.step >= 1}>
                <Span x={24} y={58} w={xInW} h={22} kind="text" label="X  [B,T,h]" ghost={s.step >= 3} labelTone="light" />
                <Arrow x1={24 + xInW + 4} y1={69} x2={CARD_X0 - 8} y2={69} kind={s.step >= 3 ? 'comm' : 'text'} muted={s.step < 3} />
              </Fade>

              {Array.from({ length: n }, (_, i) => {
                const x = CARD_X0 + i * (CW + CARD_GAP);
                const pad = Math.min(8, CW * 0.14);
                return (
                  <g key={i}>
                    <rect x={x} y={34} width={CW} height={66} rx={3} fill="var(--ground-2)" stroke="var(--rule-2)" />
                    <Label x={x + CW / 2} y={48} anchor="middle" size={CW > 70 ? 9.5 : 8} tone={3}>{n === 1 ? 'GPU 0' : `GPU ${i}`}</Label>
                    <rect x={x + pad} y={54} width={CW - 2 * pad} height={13} rx={2} fill="var(--compute)" opacity={0.85} />
                    <Label x={x + CW / 2} y={64} anchor="middle" size={CW > 80 ? 8.5 : 7}>{n === 1 ? 'W1 (full)' : `W1 cols ${i}`}</Label>
                    <rect x={x + pad} y={70} width={CW - 2 * pad} height={11} rx={2} fill="var(--text)" opacity={0.09} />
                    <Label x={x + CW / 2} y={79} anchor="middle" size={CW > 60 ? 8 : 7} tone={3}>{n === 1 ? 'act' : `act ${i}`}</Label>
                    <rect x={x + pad} y={84} width={CW - 2 * pad} height={13} rx={2} fill="var(--compute)" opacity={n > 1 ? 0.85 : 0.4} />
                    <Label x={x + CW / 2} y={94} anchor="middle" size={CW > 80 ? 8.5 : 7}>{n === 1 ? 'W2 (full)' : `W2 rows ${i}`}</Label>
                  </g>
                );
              })}

              {n > 1 && (
                <Fade show={s.step >= 4}>
                  {Array.from({ length: n }, (_, i) => {
                    const x = CARD_X0 + i * (CW + CARD_GAP);
                    return <Arrow key={i} x1={x + CW / 2} y1={101} x2={x + CW / 2} y2={113} kind="comm" />;
                  })}
                  <Dim x1={CARD_X0} x2={CARD_X0 + rowW(n)} y={117} label="all-reduce: sum the partials" kind="comm" />
                </Fade>
              )}
              {n > 1 && s.step >= 7 && (
                <Label x={616} y={113} anchor="end" size={9.5} kind="error">{`${((1 - speedupT / n) * 100).toFixed(0)}% of the ideal speedup lost`}</Label>
              )}
              {s.step < 4 && (
                <Label x={CARD_X0} y={116} size={9.5} tone={3}>column-parallel W1 (no comm) → activation → row-parallel W2</Label>
              )}
              {s.step >= 3 && (
                <Dim x1={CARD_X0} x2={CARD_X0 + rowW(n)} y={133} label="all-gather: rebuild the full X" kind="comm" />
              )}
            </>
          );
        })()}

        {/* ── the speedup chart ───────────────────────────────── */}
        <Label x={24} y={164} tone={2} size={11} bold>Speedup against TP degree</Label>
        <Label x={616} y={164} anchor="end" tone={3} size={9}>{`${phase}, batch ${fmt.int(batch)}, ${link.name}`}</Label>

        {(() => {
          const X0 = 60;
          const W = 400;
          const H = 118;
          const y0 = 180;
          const bw = W / chart.length;
          return (
            <g>
              {/* ideal line */}
              <line x1={X0} y1={y0 + H} x2={X0 + W} y2={y0 + H - (H / maxSpeed) * maxSpeed} stroke="var(--text-3)" strokeDasharray="4 4" strokeWidth={1} />
              <Label x={X0 + W + 4} y={y0 + 2} size={8.5} tone={3}>ideal</Label>
              <Label x={X0 - 10} y={y0 + 4} anchor="end" size={8.5} tone={3}>{maxSpeed.toFixed(0)}x</Label>
              <Label x={X0 - 10} y={y0 + H + 4} anchor="end" size={8.5} tone={3}>0</Label>
              {chart.map((c, i) => {
                const h = (c.v / maxSpeed) * H;
                const sel = c.p === tEff;
                return (
                  <g key={c.p}>
                    <rect x={X0 + i * bw + 6} y={y0 + H - h} width={bw - 12} height={h} rx={2}
                      fill={sel ? 'var(--token)' : 'var(--compute)'} opacity={sel ? 0.95 : 0.4} stroke={sel ? 'var(--token)' : 'var(--compute)'} />
                    <Label x={X0 + i * bw + bw / 2} y={y0 + H - h - 4} anchor="middle" size={8.5} kind={sel ? 'token' : 'text'}>{c.v.toFixed(2)}x</Label>
                    <Label x={X0 + i * bw + bw / 2} y={y0 + H + 12} anchor="middle" size={9} tone={3}>t={c.p}</Label>
                  </g>
                );
              })}
            </g>
          );
        })()}

        {/* ── the three terms ─────────────────────────────────── */}
        <Label x={24} y={332} tone={2} size={11} bold>{`Where the time goes at t=${tEff}`}</Label>
        {(() => {
          const terms = phase === 'decode'
            ? [
                { k: 'memory' as const, n: 'weights (memory)', v: dec.tMem },
                { k: 'compute' as const, n: 'compute', v: dec.tCompute },
                { k: 'comm' as const, n: 'collectives', v: dec.tComm },
              ]
            : [
                { k: 'compute' as const, n: 'compute', v: pre.tCompute },
                { k: 'comm' as const, n: 'collectives', v: pre.tComm },
              ];
          const tot = terms.reduce((a, x) => a + x.v, 0);
          const X = 168;
          const W = 300;
          return (
            <g>
              {terms.map((x, i) => (
                <g key={x.n}>
                  <Label x={160} y={346 + i * 14} anchor="end" size={9} mono={false} tone={3}>{x.n}</Label>
                  <rect x={X} y={337 + i * 14} width={W} height={10} rx={2} fill="none" stroke="var(--rule)" />
                  <Span x={X} y={337 + i * 14} w={(x.v / tot) * W} h={10} kind={x.k} />
                  <Label x={X + (x.v / tot) * W + 6} y={345 + i * 14} size={9} kind={x.k}>{`${((x.v / tot) * 100).toFixed(0)}%`}</Label>
                </g>
              ))}
            </g>
          );
        })()}
        <Label x={616} y={372} anchor="end" size={9.5} kind={commPct > 0.2 ? 'error' : 'token'}>
          {`${fmt.bytes(commBytesPerPass * 2 * model.layers)} of comm per forward pass`}
        </Label>
        <Label x={24} y={372} size={9} tone={3}>{`${fmt.gb(perGpu)} of weights+KV per GPU`}</Label>
      </svg>
    </Stage>
  );
}