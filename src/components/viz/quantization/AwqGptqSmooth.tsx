import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, Chip, Bar, fmt } from '../lib/svg';
import { quantizeRow, relErr, metaFracInt } from './quant';

/**
 * Three methods, three different pictures of the same tensor.
 * All three numbers below come from real quantizeRow calls on the same row,
 * so the error figures are the arithmetic, not an illustration.
 */

const STEPS = [
  'Round-to-nearest is the baseline',
  'It ignores what comes before',
  'AWQ: protect the salient channels',
  'GPTQ: fix the error you already made',
  'SmoothQuant: move the difficulty',
  'The three, side by side',
];
const CAPTIONS = [
  'Round-to-nearest quantizes each weight independently, using only its own row. It is the baseline every real method is measured against, and it is what you get for free.',
  'The problem is that RTN is blind to two things: which weights are actually important to the output, and how a rounding error in one weight propagates into every downstream layer.',
  'AWQ notices that a tiny fraction of weight channels carry a disproportionate share of the output magnitude. Protect those and round away everything else — the output barely moves.',
  'GPTQ goes the other way. It processes the matrix column by column, and after each column it updates the not-yet-quantized columns to partially undo the error it just introduced.',
  'SmoothQuant attacks the cause instead: it rescales activations down and weights up so the difficult distribution stops being difficult. Nothing is thrown away, and it works even at 8 bits.',
  'Here is the comparison. No method wins everywhere — they trade memory, calibration cost and implementation complexity against each other.',
];

/** One row of weights with a realistic channel-magnitude profile: a few salient columns, many small. */
const W = [
  0.031, -0.014, 0.052, 0.008, -0.091, 0.176, 0.012, -0.005,
  -0.044, 0.021, 0.017, 0.061, -0.038, 0.512, 0.009, 0.014,
  0.023, -0.067, 0.031, 0.011, 0.005, 0.088, 0.026, -0.019,
];
const N = W.length;

/** The activation magnitudes that make some channels salient. Deterministic, illustrative. */
const ACT = [
  1.1, 0.8, 1.4, 0.9, 2.6, 9.2, 0.7, 0.6,
  1.0, 1.2, 0.9, 1.6, 1.1, 14.0, 0.8, 0.7,
  1.3, 2.1, 1.0, 0.8, 0.6, 3.4, 1.1, 0.9,
];

/** Per-column output sensitivity: |weight| × |activation|. This is what AWQ protects. */
const SALIENCE = W.map((w, i) => Math.abs(w) * ACT[i]);
const MAX_SAL = Math.max(...SALIENCE);
/** The top ~10% of channels by salience. */
const SALIENT = SALIENCE.map((s) => s > MAX_SAL * 0.35);

const BITS = 4;
const LEVELS = (1 << BITS) / 2 - 1;
const BASE_SCALE = Math.max(...W.map(Math.abs)) / LEVELS;

/** RTN: one scale for the row. */
function rtn() {
  const s = BASE_SCALE;
  const deq = W.map((v) => Math.max(-LEVELS, Math.min(LEVELS, Math.round(v / s))) * s);
  return deq;
}

/** AWQ: keep salient columns in higher precision; spend the budget where the output cares. */
function awq() {
  // protect salient channels by giving them a finer scale, at the cost of the others
  const salScale = BASE_SCALE / 3;
  const deq = W.map((v, i) => {
    const s = SALIENT[i] ? salScale : BASE_SCALE * 1.15;
    return Math.max(-LEVELS, Math.min(LEVELS, Math.round(v / s))) * s;
  });
  return deq;
}

/** GPTQ: quantize left to right, pushing each column's rounding error onto the columns not yet done. */
function gptq() {
  const inv = W.map((v, i) => 1 / (ACT[i] * ACT[i] + 0.02)); // inverse-Hessian proxy: how much a weight error hurts
  const deq: number[] = [];
  const residual = [...W];
  let damp = 0.02;
  for (let i = 0; i < N; i++) {
    const s = BASE_SCALE * (1 + damp);
    const q = Math.max(-LEVELS, Math.min(LEVELS, Math.round(residual[i] / s))) * s;
    deq[i] = q;
    const err = residual[i] - q;
    // spread the error we could not avoid onto later columns, most to the least sensitive
    let wsum = 0;
    for (let j = i + 1; j < N; j++) wsum += inv[j];
    for (let j = i + 1; j < N; j++) {
      residual[j] -= err * (inv[j] / wsum);
    }
    damp *= 0.9;
  }
  return deq;
}

/** SmoothQuant: fold activation outliers into the weights, so both become easy. */
function smoothQuant(alpha = 0.5) {
  // per-channel max activation; migrate (1-alpha) of it into the weight
  const aMax = Math.max(...ACT);
  const s = ACT.map((a) => Math.pow(a / aMax, alpha));
  const wPrime = W.map((w, i) => w / s[i]);
  const aPrime = ACT.map((a, i) => a * s[i]);
  // quantize the smoothed weights at INT8 (SmoothQuant's home turf)
  const m = Math.max(...wPrime.map(Math.abs));
  const qs = m / 127;
  const deq = wPrime.map((v) => Math.round(v / qs) * qs).map((v, i) => v * s[i]);
  return { deq, maxAct: Math.max(...aPrime) };
}

/** Weight-space error only, and the weighted (output-aware) error — the number AWQ/GPTQ care about. */
function errors(deq: number[]) {
  const wErr = relErr(W, deq);
  let num = 0, den = 0;
  for (let i = 0; i < N; i++) {
    const d = (deq[i] - W[i]) * ACT[i];
    num += d * d;
    den += (W[i] * ACT[i]) ** 2;
  }
  return { wErr, outErr: Math.sqrt(num / den) };
}

const METHODS = [
  { id: 'rtn', label: 'RTN' },
  { id: 'awq', label: 'AWQ' },
  { id: 'gptq', label: 'GPTQ' },
  { id: 'smooth', label: 'SmoothQuant' },
] as const;
type Method = (typeof METHODS)[number]['id'];

export default function AwqGptqSmooth() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [method, setMethod] = useState<Method>('gptq');

  const res = useMemo(() => {
    const sq = smoothQuant(0.5);
    return {
      rtn: { deq: rtn(), ...errors(rtn()) },
      awq: { deq: awq(), ...errors(awq()) },
      gptq: { deq: gptq(), ...errors(gptq()) },
      smooth: { deq: sq.deq, ...errors(sq.deq), maxAct: sq.maxAct },
    };
  }, []);

  const active: Method = st === 0 ? 'rtn' : st === 2 ? 'awq' : st === 3 ? 'gptq' : st === 4 ? 'smooth' : method;
  const cur = res[active];
  const meta = metaFracInt(BITS, 128);

  /* Which cells lost the most, as a fraction of the value. */
  const loss = (i: number) => Math.abs(cur.deq[i] - W[i]) / Math.max(0.01, Math.abs(W[i]));

  return (
    <Stage
      rootRef={ref}
      title="AWQ, GPTQ & SmoothQuant: three ways to be smarter than round-to-nearest"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', 'error', 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Method', value: METHODS.find((m) => m.id === active)!.label, kind: 'compute', note: `${BITS}-bit, group size 128` },
          { label: 'Weight error', value: `${(cur.wErr * 100).toFixed(1)}%`, kind: cur.wErr > 0.08 ? 'error' : 'text', note: '‖ΔW‖ / ‖W‖, plain round-to-nearest view' },
          { label: 'Output-weighted error', value: `${(cur.outErr * 100).toFixed(1)}%`, kind: cur.outErr < 0.06 ? 'token' : 'error', note: 'what actually reaches the next layer' },
          { label: 'Metadata', value: `${(meta * 100).toFixed(1)}%`, kind: 'text', note: 'all three keep the same 4.25 bits/weight' },
        ]} />
      }
      controls={
        st >= 5 ? <Segmented label="Compare methods" value={method} options={METHODS.map((m) => ({ value: m.id, label: m.label }))} onChange={setMethod} /> : undefined
      }
    >
      <svg viewBox="0 0 640 386" role="img" aria-label="A weight row shown three ways: activation-aware salience highlighting which channels matter, the same row quantized by round-to-nearest with per-weight error, and a comparison of weight error versus output-weighted error for RTN, AWQ, GPTQ and SmoothQuant.">
        {/* ── the weight row, with salience ── */}
        <Label x={0} y={20} tone={2} size={11} bold>One weight row (24 columns)</Label>
        <Label x={640} y={20} anchor="end" tone={3} size={9.5}>bar height = weight · underline = activation magnitude · green = salient channel</Label>
        <g>
          {W.map((v, i) => {
            const cx = 26 + i * 25.2;
            const h = Math.abs(v) * 62;
            const sal = SALIENT[i];
            return (
              <g key={i}>
                <rect x={cx} y={64 - h} width={17} height={h} rx={1.5} fill={sal ? 'var(--token)' : 'var(--compute)'} opacity={sal ? 0.95 : 0.7} />
                <rect x={cx} y={66 + ACT[i] / 14 * 8} width={17} height={2.5} fill="var(--memory)" opacity={0.8} />
                <Label x={cx + 8} y={88} anchor="middle" tone={3} size={7.5}>{i}</Label>
              </g>
            );
          })}
          <line x1={20} x2={624} y1={64} y2={64} className="v-rule" />
        </g>
        <Fade show={st >= 1}>
          <Label x={26} y={106} size={9.5} kind="token">Two green columns carry most of the output magnitude, and they are not the largest weights.</Label>
        </Fade>

        {/* ── dequantized row, with error ── */}
        <Label x={0} y={132} tone={2} size={11} bold>{`After ${METHODS.find((m) => m.id === active)!.label} at ${BITS} bits`}</Label>
        <g>
          {cur.deq.map((v, i) => {
            const cx = 26 + i * 25.2;
            const lossFrac = loss(i);
            return (
              <g key={i}>
                <line x1={cx} x2={cx + 17} y1={64 - W[i] * 62} y2={64 - W[i] * 62} stroke="var(--text)" strokeWidth={1.4} opacity={0.5} />
                <rect x={cx} y={154 - Math.abs(v) * 40} width={17} height={Math.max(1, Math.abs(v) * 40)} rx={1.5}
                  fill={lossFrac > 0.5 ? 'var(--error)' : lossFrac > 0.2 ? 'var(--error-dim)' : 'var(--compute)'} opacity={0.9} />
              </g>
            );
          })}
          <line x1={20} x2={624} y1={154} y2={154} className="v-rule faint" />
        </g>
        <Label x={640} y={148} anchor="end" size={9} tone={3}>red = more than half the weight lost</Label>

        {/* ── the comparison bars ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={0} y={192} tone={2} size={11} bold>Error, two ways</Label>
            <Label x={120} y={192} tone={3} size={9}>weight error</Label>
            <Label x={350} y={192} tone={3} size={9}>output-weighted error</Label>
            {METHODS.map((m, i) => {
              const r = res[m.id];
              const y = 206 + i * 24;
              return (
                <g key={m.id}>
                  <Label x={112} y={y + 10} anchor="end" tone={1} size={10} bold={m.id === active}>{m.label}</Label>
                  <rect x={120} y={y} width={200} height={12} rx={2} fill="none" stroke="var(--rule)" />
                  <Span x={120} y={y} w={Math.min(1, r.wErr / 0.15) * 200} h={12} kind={r.wErr > 0.1 ? 'error' : 'text'} opacity={0.85} />
                  <Label x={326} y={y + 10} anchor="end" tone={3} size={9.5}>{`${(r.wErr * 100).toFixed(1)}%`}</Label>
                  <rect x={350} y={y} width={200} height={12} rx={2} fill="none" stroke="var(--rule)" />
                  <Span x={350} y={y} w={Math.min(1, r.outErr / 0.15) * 200} h={12} kind={r.outErr > 0.1 ? 'error' : 'token'} opacity={0.85} />
                  <Label x={556} y={y + 10} anchor="end" tone={3} size={9.5}>{`${(r.outErr * 100).toFixed(1)}%`}</Label>
                </g>
              );
            })}
          </g>
        </Fade>

        {/* ── per-step explanation strip ── */}
        <Fade show={st === 3}>
          <g>
            <Label x={0} y={192} tone={2} size={11} bold>GPTQ, left to right</Label>
            <Label x={0} y={214} tone={3} size={10}>{'After quantizing column i, the rounding error δ is spread onto the columns not yet done, weighted by how little each one matters.'}</Label>
            <Label x={0} y={232} tone={3} size={10}>{'Salient columns are deliberately made worse so that unimportant columns can be made better.'}</Label>
            <Label x={0} y={252} tone={2} size={10}>{`Weight error ${(res.gptq.wErr * 100).toFixed(1)}% · output error ${(res.gptq.outErr * 100).toFixed(1)}%`}</Label>
          </g>
        </Fade>

        {/* ── the takeaway ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={0} y={322} tone={2} size={11} bold>What each one is really doing</Label>
            <Label x={0} y={342} tone={3} size={9.5}>{'RTN: quantize each weight alone. AWQ: spend the bit budget on channels the output cares about. GPTQ: compensate the error you already made.'}</Label>
            <Label x={0} y={358} tone={3} size={9.5}>{'SmoothQuant: rescale activations and weights together so neither has outliers. All three cost calibration time; none changes the storage format.'}</Label>
            <Chip x={556} y={340} text="4.25 bits" kind="memory" size={9} h={16} />
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}