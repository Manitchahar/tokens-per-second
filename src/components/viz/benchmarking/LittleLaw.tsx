import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Label, Fade, fmt } from '../lib/svg';
import { msec } from './helpers';
import { decodeStep, gpuById, modelById } from '../lib/perfModel';

const model = modelById('llama8b');
const gpu = gpuById('h100');

// Offered load is measured in requests per second. Each request has a fixed
// service demand (~2.5 s of GPU work at batch 1), so throughput = B / t_step.

const STEPS = [
  'One user at a time',
  'Offered load grows',
  'Throughput rises, slowly',
  'The GPU fills up',
  'Saturation',
  'Beyond the edge',
  'Little’s law',
  'Capacity from three numbers',
];
const CAPTIONS = [
  'One request in flight. The GPU is mostly idle and per-user speed is at its best.',
  'Add arrivals. Each one joins the same step, so the weights are read once for all of them.',
  'Throughput climbs far faster than the single-user baseline. This is why batching is the cheapest optimization there is.',
  'At high batch sizes the KV cache adds memory traffic, and the step starts growing with batch again.',
  'At the knee, the GPU is saturated: adding arrivals no longer adds throughput, only queue.',
  'Past the knee you are in overload. Offered load exceeds capacity, the queue grows without bound, and latency explodes.',
  'L = λ × W: the number in the system equals arrival rate times time in the system. Queue length is the triangle’s height.',
  'You need three numbers to size a system: per-request service time, the GPU you will use, and the latency you promise.',
];

const PRESET: Record<number, { batch: number; ctx: number; load: number }> = {
  0: { batch: 1, ctx: 1024, load: 10 },
  1: { batch: 1, ctx: 1024, load: 30 },
  2: { batch: 24, ctx: 1024, load: 50 },
  3: { batch: 96, ctx: 1024, load: 65 },
  4: { batch: 256, ctx: 1024, load: 80 },
  5: { batch: 512, ctx: 1024, load: 99 },
  6: { batch: 96, ctx: 1024, load: 80 },
  7: { batch: 96, ctx: 1024, load: 80 },
};

// capacity curve: max sustainable throughput (tokens/s) vs batch, stops at OOM
const buildCurve = () => {
  const pts: { b: number; thr: number; fits: boolean }[] = [];
  for (let b = 1; b <= 1024; b = b <= 8 ? b + 1 : Math.ceil(b * 1.15)) {
    const r = decodeStep({ model, gpu, batch: b, ctx: 1024 });
    pts.push({ b, thr: r.total, fits: r.fits });
  }
  return pts;
};
const CURVE = buildCurve();

const X0 = 92;
const X1 = 616;
const Y0 = 300;
const Y1 = 108;
const MAXB = 1024;
const MAXT = 12000;
const bx = (b: number) => X0 + (Math.log2(Math.max(1, b)) / Math.log2(MAXB)) * (X1 - X0);
const by = (t: number) => Y0 - Math.min(1, t / MAXT) * (Y0 - Y1);

export default function LittleLaw() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const preset = PRESET[s.step];
  const [batch, setBatch] = useState(preset.batch);
  const [ctx, setCtx] = useState(preset.ctx);
  const [offeredPct, setOfferedPct] = useState(preset.load);
  useEffect(() => { setBatch(preset.batch); setCtx(preset.ctx); setOfferedPct(preset.load); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const r = decodeStep({ model, gpu, batch, ctx });

  // Each request costs a fixed amount of GPU time: one prefill of `promptTokens`
  // plus `outTokens` decode steps. Capacity in requests/s is the reciprocal.
  const promptTokens = Math.round(ctx);
  const outTokens = 40;
  const prefillSec = Math.max(0.008, (promptTokens / 1000) * 0.034);
  const serviceSec = prefillSec + outTokens * r.t; // one request's GPU time at this batch
  const capacity = 1 / serviceSec; // req/s this configuration can retire
  const arrival = Math.min(capacity * 1.05, (offeredPct / 100) * capacity);

  // M/M/1 queueing: wait -> infinity as utilisation -> 1. This is the mechanism
  // behind "overload is a cliff, not a slope".
  const util = Math.min(1, arrival / capacity);
  const wait = util >= 0.995 ? Infinity : (serviceSec * util) / (2 * (1 - util));
  const L = arrival * (serviceSec + wait); // Little's law: in system = rate x time

  const path = CURVE.filter((p) => p.fits).map((p, i) => `${i ? 'L' : 'M'}${bx(p.b).toFixed(1)},${by(p.thr).toFixed(1)}`).join(' ');
  const oomAt = CURVE.find((p) => !p.fits);

  const g = useTweens({
    x: bx(batch),
    y: by(r.total),
    lw: Math.min(300, Math.log10(Math.max(10, Math.min(L, 1e6))) * 60),
    lh: Math.min(140, Math.log10(Math.max(1, Math.min(L, 1e6))) * 42),
  });

  const knee = CURVE.find((p) => p.thr > MAXT * 0.86) ?? CURVE[CURVE.length - 1];
  const showCurve = s.step >= 1;
  const showKnee = s.step >= 3;
  const showSat = s.step >= 4;
  const showOver = s.step >= 5;
  const showLaw = s.step >= 6;
  const showCap = s.step >= 7;

  return (
    <Stage
      rootRef={ref}
      title="One H100, Llama-3.1-8B: throughput against concurrency, and the queue it creates"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'compute', 'memory', 'idle', ['error', 'overload / out of memory']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Per-user speed', value: `${fmt.int(r.perUser)} tok/s`, kind: 'token', note: `TPOT ${r.t.toFixed(1)} ms` },
          { label: 'Throughput', value: `${fmt.int(r.total)} tok/s`, kind: 'text', note: `batch ${batch}` },
          { label: 'Capacity', value: `${capacity.toFixed(2)} req/s`, kind: 'compute', note: `${promptTokens}-tok prompt, 40 out` },
          { label: 'In the system', value: L > 9999 ? 'unbounded' : `${fmt.int(L)} req`, kind: util > 0.8 ? 'error' : 'idle', note: `λ×W at ${(util * 100).toFixed(0)}% load` },
        ]} />
      }
      controls={
        <>
          <Slider label="Requests in the batch (concurrency)" value={batch} min={1} max={1024} log onChange={setBatch} format={(v) => `${fmt.int(v)}`} kind="token" />
          <Slider label="Average context per request" value={ctx} min={128} max={8192} log onChange={setCtx} format={(v) => `${fmt.int(v)} tok`} kind="memory" />
          <Slider label="Offered load (% of this GPU's capacity)" value={offeredPct} min={1} max={100} onChange={setOfferedPct} format={(v) => `${v}%`} kind="idle" />
        </>
      }
    >
      <svg viewBox="0 0 640 366" role="img" aria-label="A chart of throughput in tokens per second against batch size on a log axis, rising steeply then flattening near saturation, with an out-of-memory marker and a Little's law triangle showing requests in the system.">
        {/* axes */}
        <line x1={X0} x2={X1} y1={Y0} y2={Y0} className="v-rule" />
        <line x1={X0} x2={X0} y1={Y1 - 20} y2={Y0} className="v-rule" />
        {[1, 8, 64, 512].map((b) => (
          <Label key={b} x={bx(b)} y={Y0 + 14} anchor="middle" tone={3} size={9}>{b}</Label>
        ))}
        <Label x={(X0 + X1) / 2} y={Y0 + 30} anchor="middle" tone={2} size={10.5} mono={false}>requests in flight (batch size) — log scale →</Label>
        {[0, 4000, 8000, 12000].map((t) => (
          <Label key={t} x={X0 - 6} y={by(t) + 3} anchor="end" tone={3} size={9}>{t ? `${t / 1000}k` : '0'}</Label>
        ))}
        <text x={22} y={(Y0 + Y1) / 2} transform={`rotate(-90 22 ${(Y0 + Y1) / 2})`} textAnchor="middle" className="v-label t2" style={{ fontSize: 10.5 }}>throughput (tokens/s)</text>

        {/* capacity curve */}
        {showCurve && (
          <Fade show={showCurve}>
            <path d={path} fill="none" stroke="var(--text-2)" strokeWidth={1.8} />
          </Fade>
        )}

        {/* saturation band */}
        {showKnee && (
          <Fade show={showKnee}>
            <rect x={bx(knee.b)} y={Y1 - 20} width={X1 - bx(knee.b)} height={Y0 - Y1 + 20} fill="rgb(74 82 98 / 0.12)" />
            <Label x={bx(knee.b) + 6} y={Y1 - 26} size={9.5} kind="compute">saturated: batch no longer helps</Label>
          </Fade>
        )}

        {/* OOM marker */}
        {oomAt && (
          <Fade show={showSat}>
            <line x1={bx(oomAt.b)} x2={bx(oomAt.b)} y1={Y1 - 20} y2={Y0} stroke="var(--error)" strokeDasharray="3 3" strokeWidth={1.2} />
            <text x={bx(oomAt.b) - 5} y={Y1 - 26} textAnchor="end" className="v-label" style={{ fill: 'var(--error)', fontSize: 9.5, fontWeight: 600 }}>OOM at batch {oomAt.b}</text>
          </Fade>
        )}

        {/* the moving point */}
        <g>
          <line x1={g.x} x2={g.x} y1={g.y} y2={Y0} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="2 3" />
          <g className="v-move" style={{ transform: `translate(${g.x}px, ${g.y}px)` }}>
            <circle r={5.5} fill="var(--token)" stroke="var(--ground-2)" strokeWidth={2} />
          </g>
          <text x={g.x} y={g.y - 12} textAnchor="middle" className="v-label t1" style={{ fontSize: 10, fontWeight: 600 }}>
            {`${fmt.int(r.total)} tok/s`}
          </text>
        </g>

        {/* overload: arrivals keep coming, nothing more is produced */}
        {showOver && (
          <Fade show={showOver}>
            <line x1={X0} x2={X1} y1={Y1 - 44} y2={Y1 - 44} stroke="var(--error)" strokeWidth={1.2} strokeDasharray="5 4" />
            <text x={X1} y={Y1 - 50} textAnchor="end" className="v-label" style={{ fill: 'var(--error)', fontSize: 10, fontWeight: 600 }}>
              offered load exceeds capacity → the queue grows forever
            </text>
          </Fade>
        )}

        {/* Little's law triangle */}
        {showLaw && (
          <Fade show={showLaw}>
            <rect x={X0 + 18} y={Y0 - g.lh} width={g.lw} height={g.lh} fill="rgb(74 82 98 / 0.16)" stroke="var(--text-3)" strokeWidth={1} strokeDasharray="4 3" />
            <text x={X0 + 26} y={Y0 - 10} className="v-label t2" style={{ fontSize: 10, fontWeight: 600 }}>L = λ × W</text>
            <text x={X0 + 26} y={Y0 - 24} className="v-label t3" style={{ fontSize: 9.5 }}>
              {`L = ${L > 9999 ? 'unbounded' : `${fmt.int(L)} in the system`}`}
            </text>
            <text x={X0 + 26} y={Y0 - g.lh - 6} className="v-label t3" style={{ fontSize: 9.5 }}>
              {`λ = ${arrival.toFixed(2)} req/s · W = ${wait === Infinity ? '∞' : msec(wait * 1000)}`}
            </text>
          </Fade>
        )}

        {/* capacity formula */}
        {showCap && (
          <Fade show={showCap}>
            <text x={X0} y={Y0 + 52} className="v-label t2" style={{ fontSize: 10.5 }}>
              {`capacity = 1 / service time = 1 / ${(serviceSec * 1000).toFixed(0)} ms = ${capacity.toFixed(2)} req/s`}
            </text>
            <text x={X0} y={Y0 + 68} className="v-label t3" style={{ fontSize: 9.5 }}>
              {`service time = ${(prefillSec * 1000).toFixed(0)} ms prefill + ${outTokens} × ${(r.t * 1000).toFixed(1)} ms decode`}
            </text>
          </Fade>
        )}
      </svg>
    </Stage>
  );
}