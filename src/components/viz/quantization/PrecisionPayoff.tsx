import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade, Chip, fmt } from '../lib/svg';
import { llama8b, h100, decodeStepQuant, weightBytes, kvBytes, metaFracInt, quantizeRow, relErr, SCHEMES } from './quant';

/** 32 users, each holding 4,096 tokens of context: the serving shape this module uses throughout. */
const BATCH = 32;
const CTX = 4096;

const STEPS = [
  'Bytes decide what fits',
  'Bytes decide the decode step',
  'The KV cache joins the bill',
  'Precision is telling values apart',
  'Fewer bytes moves the roofline',
  'And there is a price',
];
const CAPTIONS = [
  'A weight is a number in memory, so bytes per weight is bytes per model. Halving the bits halves the card you need — before a single token is produced.',
  'Decode reads the entire weight matrix once per step, so one step costs about (bytes in the model) divided by (memory bandwidth). Fewer bits, shorter step.',
  'But a decode step also reads every user’s KV cache. At 32 users and 4k context the KV cache is already larger than the weights — so quantizing weights alone has less headroom than you expect.',
  'Bits per weight is a memory decision. How many distinct values the format can tell apart is a different question. Four bits is sixteen of them, evenly spaced.',
  'Quantization does not only shrink memory: it moves the workload along the roofline. Fewer bytes per FLOP keeps decode memory-bound to a much larger batch, which is where a serving engine lives.',
  'Every bit removed buys bytes and milliseconds, and also buys error. The whole job of this module is spending that error deliberately instead of by accident.',
];

const FORMATS = [
  { bits: 16, label: 'BF16' },
  { bits: 8, label: 'INT8' },
  { bits: 4, label: 'INT4' },
  { bits: 2, label: 'INT2' },
];

/** Which SCHEMES row each format's readout describes. */
const SCHEME_FOR = [0, 2, 3, 5];

/** A deterministic slice of a real weight row: the shape matters, the values are illustrative. */
const ROW = [-0.42, 0.11, -0.03, 0.65, -0.87, 0.24, 0.05, -0.19, 0.38, -0.71, 0.16, 0.02, -0.55, 0.29, -0.08, 0.47];

/* ── bar chart geometry ── */
const X0 = 108;
const BAR_W = 380;
const PX_GB = BAR_W / 60; // 60 GB full scale
const ROW0 = 38;
const ROW_H = 21;

/* ── roofline mini-chart geometry ── */
const RX0 = 52, RX1 = 600, RY1 = 306, RY0 = 358;
const rx = (b: number) => RX0 + (Math.log2(b) / 8) * (RX1 - RX0); // batch 1 → 256, log axis
const ry = (f: number) => RY0 - Math.min(1, f) * (RY0 - RY1);

/** Machine balance of the H100 in FLOP per byte: the ridge where a memory-bound job becomes compute-bound. */
const BALANCE = (h100.flops * 0.55) / (h100.bw * 0.8);

/** Achieved fraction of peak FLOP/s at batch b, for a job using `bytesPerParam` bytes per weight. */
const achieved = (b: number, bytesPerParam: number) => Math.min((2 * b) / bytesPerParam / BALANCE, 1);

/** The batch at which decode stops waiting on bytes. Depends only on bytes per parameter. */
const crossBatch = (bytesPerParam: number) => (BALANCE * bytesPerParam) / 2;

export default function PrecisionPayoff() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [fi, setFi] = useState(1);
  const bits = FORMATS[fi].bits;

  /* One row per scheme: weight bytes (including scales), KV bytes, decode step. */
  const rows = useMemo(
    () =>
      SCHEMES.map((sc) => {
        const meta = metaFracInt(sc.wBits, 128);
        const w = weightBytes(llama8b, sc.wBits, meta);
        const kv = BATCH * kvBytes(CTX, sc.kvBits);
        const step = decodeStepQuant({ batch: BATCH, ctx: CTX, wBits: sc.wBits, kvBits: sc.kvBits, metaFrac: meta });
        return { ...sc, meta, w, kv, t: step.t, bytesPerParam: w / llama8b.params };
      }),
    [],
  );

  const cur = rows[SCHEME_FOR[fi]];
  const speedup = rows[0].t / cur.t;
  const q = useMemo(() => quantizeRow(ROW, bits), [bits]);
  const err = useMemo(() => relErr(ROW, q.deq), [q]);
  const liveRow = rows.findIndex((r) => r.id === cur.id);

  /* reconstruction plot geometry */
  const MID = 240, K = 30, CX0 = 24, CELL = 37.4;
  const yOf = (v: number) => MID - v * K;

  return (
    <Stage
      rootRef={ref}
      title="Llama-3.1-8B on one H100: what 16 bits buys you, and what 4 bits does not"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', ['memory', 'weights'], ['memory', 'KV cache'], 'error', 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Weights on the GPU', value: fmt.gb(cur.w), kind: 'memory', note: `+${(cur.meta * 100).toFixed(1)}% scales and zero-points` },
          { label: 'One decode step', value: `${(cur.t * 1e3).toFixed(1)} ms`, kind: 'compute', note: `${BATCH} users · ${fmt.int(CTX)} tokens of context each` },
          { label: 'Versus BF16', value: `${speedup.toFixed(2)}×`, kind: speedup > 1.15 ? 'token' : 'text', note: 'decode is memory-bound, so bytes become time' },
          { label: `Weight error at ${bits} bits`, value: `${(err * 100).toFixed(1)}%`, kind: err > 0.2 ? 'error' : err > 0.04 ? 'compute' : 'text', note: 'relative: ‖ΔW‖ / ‖W‖' },
        ]} />
      }
      controls={
        st >= 3 ? (
          <Segmented
            label="Format used for the reconstruction and the readout"
            value={fi}
            options={FORMATS.map((f, i) => ({ value: i, label: `${f.label} · ${f.bits} bits` }))}
            onChange={setFi}
          />
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Six bar rows comparing GPU memory and decode step time for six precision schemes of Llama-3.1-8B, with weights and KV cache as separate segments. Below, a row of weight values reconstructed at a chosen bit width with the reconstruction error hatched in red. Below that, a roofline chart where lower precision keeps decode memory-bound to a larger batch.">
        <Label x={0} y={22} tone={2} size={11} bold>What one decode step has to read</Label>
        <Label x={640} y={22} anchor="end" tone={3} size={9.5}>{`Llama-3.1-8B · H100 · ${BATCH} users · ${fmt.int(CTX)} tokens of context each`}</Label>

        {/* ── one row per scheme ── */}
        {rows.map((r, i) => {
          const wpx = (r.w / 1e9) * PX_GB;
          const kvpx = (r.kv / 1e9) * PX_GB;
          const y = ROW0 + i * ROW_H;
          const live = i === liveRow;
          return (
            <g key={r.id}>
              <Label x={100} y={y + 11} anchor="end" tone={live ? 1 : 3} size={10.5} bold={live}>{r.label}</Label>
              <rect x={X0} y={y} width={BAR_W} height={14} rx={2} fill="none" stroke="var(--rule)" />
              <Span x={X0} y={y} w={wpx} h={14} kind="memory" opacity={live ? 0.95 : 0.5} />
              {st >= 2 && (
                <rect x={X0 + wpx} y={y} width={kvpx} height={14} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} opacity={live ? 1 : 0.5} />
              )}
              <Label x={X0 + wpx + kvpx + 8} y={y + 11} tone={live ? 1 : 3} size={10}>
                {st >= 2 ? `${((r.w + r.kv) / 1e9).toFixed(1)} GB · ${(r.t * 1e3).toFixed(1)} ms` : `${(r.w / 1e9).toFixed(1)} GB`}
              </Label>
            </g>
          );
        })}

        {/* which blue segment is which */}
        <Fade show={st >= 2}>
          <g>
            <rect x={X0} y={166} width={11} height={9} rx={1.5} fill="var(--memory)" />
            <Label x={X0 + 16} y={175} tone={3} size={9.5}>weights</Label>
            <rect x={X0 + 72} y={166} width={11} height={9} rx={1.5} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} />
            <Label x={X0 + 88} y={175} tone={3} size={9.5}>KV cache</Label>
            <Label x={X0 + 244} y={175} tone={3} size={9.5}>{`${(rows[0].kv / 1e9).toFixed(1)} GB at BF16 — more than the weights themselves`}</Label>
          </g>
        </Fade>

        {/* ── what the format can actually store ── */}
        <Fade show={st >= 3}>
          <Label x={0} y={198} tone={2} size={11} bold>{`One row of weights, reconstructed at ${bits} bits`}</Label>
          <Label x={640} y={198} anchor="end" size={9.5} kind={err > 0.2 ? 'error' : 'compute'}>
            {`${(err * 100).toFixed(1)}% relative error · ${1 << bits} representable values`}
          </Label>
          <g>
            <line x1={CX0 - 8} x2={618} y1={MID} y2={MID} className="v-rule faint" />
            {ROW.map((v, i) => {
              const cx = CX0 + i * CELL;
              const gap = Math.abs(q.deq[i] - v) * K;
              return (
                <g key={i}>
                  <line x1={cx} x2={cx} y1={208} y2={264} className="v-rule faint" />
                  {gap > 0.6 && (
                    <rect x={cx - 6} y={Math.min(yOf(v), yOf(q.deq[i]))} width={12} height={gap} fill="url(#tps-hatch-red)" opacity={0.9} />
                  )}
                  {/* the true weight */}
                  <line x1={cx - 7} x2={cx + 7} y1={yOf(v)} y2={yOf(v)} stroke="var(--text)" strokeWidth={1.5} />
                  {/* what the format stores, from zero */}
                  <rect x={cx - 4} y={Math.min(MID, yOf(q.deq[i]))} width={8} height={Math.max(1.2, Math.abs(yOf(q.deq[i]) - MID))} rx={1} fill="var(--compute)" stroke="var(--compute)" />
                  <Label x={cx} y={278} anchor="middle" tone={3} size={8}>{v.toFixed(2)}</Label>
                </g>
              );
            })}
          </g>
          <Label x={CX0 - 8} y={264} tone={3} size={9}>0</Label>
          <Label x={CX0 - 8} y={214} tone={3} size={9}>+1</Label>
          <Label x={618} y={292} anchor="end" tone={3} size={9.5}>
            {bits >= 8
              ? 'White ticks are the true weights; amber bars are what the format can represent.'
              : `${bits} bits leaves ${1 << bits} evenly spaced values, so nearby weights collapse onto the same one.`}
          </Label>
        </Fade>

        {/* ── roofline: memory-bound or compute-bound ── */}
        <Fade show={st >= 4}>
          <Label x={0} y={RY1 - 8} tone={2} size={11} bold>Fewer bytes per weight keeps decode memory-bound to a larger batch</Label>
          <line x1={RX0} x2={RX1} y1={RY0} y2={RY0} className="v-rule" />
          <line x1={RX0} x2={RX0} y1={RY1} y2={RY0} className="v-rule" />
          <Curve bytesPerParam={2.031} color="var(--memory)" />
          <Curve bytesPerParam={1.047} color="var(--compute)" />
          <Curve bytesPerParam={0.531} color="var(--token)" />
          {[1, 4, 16, 64, 256].map((b) => (
            <Label key={b} x={rx(b)} y={RY0 + 12} anchor="middle" tone={3} size={9}>{b}</Label>
          ))}
          <Label x={RX1} y={RY0 + 25} anchor="end" tone={3} size={9}>batch size (users) →</Label>
          <Label x={RX0 + 6} y={RY1 + 12} size={9} tone={3}>memory-bound</Label>
          <Label x={RX1 - 6} y={RY1 + 12} anchor="end" size={9} tone={3}>compute-bound</Label>
          <Mark bytesPerParam={2.031} label="BF16 · 216" kind="memory" />
          <Mark bytesPerParam={1.047} label="INT8 · 108" kind="compute" />
          <Mark bytesPerParam={0.531} label="INT4 · 54" kind="token" />
        </Fade>
      </svg>
    </Stage>
  );
}

/** One roofline curve: achieved FLOP/s as a fraction of peak, against batch size. */
function Curve({ bytesPerParam, color }: { bytesPerParam: number; color: string }) {
  const pts: string[] = [];
  for (let b = 1; b <= 256; b = b < 4 ? b + 1 : Math.round(b * 1.2)) {
    pts.push(`${pts.length ? 'L' : 'M'}${rx(b).toFixed(1)},${ry(achieved(b, bytesPerParam)).toFixed(1)}`);
  }
  return <path d={pts.join(' ')} fill="none" stroke={color} strokeWidth={1.8} opacity={0.9} />;
}

/** Where a curve reaches the compute ceiling: the batch beyond which more users do not raise FLOP/s. */
function Mark({ bytesPerParam, label, kind }: { bytesPerParam: number; label: string; kind: 'memory' | 'compute' | 'token' }) {
  const bx = rx(crossBatch(bytesPerParam));
  return (
    <g>
      <line x1={bx} x2={bx} y1={RY1 + 2} y2={RY0} stroke={`var(--${kind})`} strokeWidth={1.1} strokeDasharray="3 3" opacity={0.75} />
      <Chip x={bx - 24} y={RY0 + 18} text={label} kind={kind} size={9} h={15} />
    </g>
  );
}