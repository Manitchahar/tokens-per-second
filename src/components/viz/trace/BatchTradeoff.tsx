import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Span, Label, Fade, fmt } from '../lib/svg';
import { decodeStep, gpuById, modelById } from '../lib/perfModel';

const model = modelById('llama8b');
const gpu = gpuById('h100');

const STEPS = [
  { label: 'One user', batch: 1, ctx: 1024 },
  { label: 'Latency', batch: 1, ctx: 1024 },
  { label: 'Add users: throughput', batch: 16, ctx: 1024 },
  { label: 'Many users', batch: 96, ctx: 1024 },
  { label: 'Memory runs out', batch: 256, ctx: 4096 },
  { label: 'Cost', batch: 96, ctx: 1024 },
];
const CAPTIONS = [
  'One user, one decode step: the GPU reads all 16 GB of weights to produce a single token.',
  'Latency is how long one user waits per token: one decode step (about 7.5 ms here), or ~130 tokens per second.',
  'Serve 16 users in the same step. The weights are read once for all of them, so the step barely gets slower, but 16 tokens come out.',
  'At 96 users the step is slower (reading everyone’s KV cache adds up), yet total throughput is huge.',
  'Every user’s KV cache lives in GPU memory. With long contexts and many users, it no longer fits: out of memory.',
  'Cost per million tokens is the GPU’s hourly price divided by tokens produced per hour. Batching makes it dramatically cheaper.',
];

const W = 640;
export default function BatchTradeoff() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const preset = STEPS[s.step];
  const [batch, setBatch] = useState(preset.batch);
  const [ctx, setCtx] = useState(preset.ctx);
  useEffect(() => { setBatch(preset.batch); setCtx(preset.ctx); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const r = decodeStep({ model, gpu, batch, ctx });
  const curve = useMemo(() => {
    const pts: { b: number; x: number; y: number; fits: boolean }[] = [];
    for (let b = 1; b <= 512; b = Math.ceil(b * 1.25)) {
      const q = decodeStep({ model, gpu, batch: b, ctx });
      pts.push({ b, x: q.total, y: q.perUser, fits: q.fits });
    }
    return pts;
  }, [ctx]);

  // trace geometry: one decode step, scaled so 25 ms fills the width
  const X0 = 96, X1 = 626, TS = 0.025;
  const px = (sec: number) => Math.min(X1 - X0, (sec / TS) * (X1 - X0));
  const g = useTweens({ w: px(r.tWeights), k: px(r.tKv), c: px(r.tCompute), o: px(r.overhead), t: px(r.t), used: Math.min(1.15, r.memUsed / gpu.mem), wt: r.weights / gpu.mem, kv: r.kvTotal / gpu.mem });

  // chart geometry
  const CX0 = 96, CX1 = 626, CY0 = 330, CY1 = 232;
  const maxX = 32000, maxY = 160;
  const cx = (v: number) => CX0 + Math.min(1, v / maxX) * (CX1 - CX0);
  const cy = (v: number) => CY0 - Math.min(1, v / maxY) * (CY0 - CY1);
  const path = curve.filter((p) => p.fits).map((p, i) => `${i ? 'L' : 'M'}${cx(p.x)},${cy(p.y)}`).join(' ');
  const shownTokens = Math.min(batch, 48);

  return (
    <Stage
      rootRef={ref}
      title="Llama-3.1-8B on one H100: batch size vs everything"
      steps={STEPS.map((x) => x.label)}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', ['error', 'out of memory'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Latency per token', value: fmt.ms(r.t), kind: 'text', note: `${fmt.int(r.perUser)} tok/s per user` },
          { label: 'Throughput', value: `${fmt.int(r.total)} tok/s`, kind: 'token', note: 'all users together' },
          { label: 'GPU memory', value: `${(r.memUsed / 1e9).toFixed(0)} / 80 GB`, kind: r.fits ? 'memory' : 'error', note: r.fits ? 'fits' : 'does not fit' },
          { label: 'Cost', value: r.fits ? `$${r.costPerMTok < 1 ? r.costPerMTok.toFixed(3) : r.costPerMTok.toFixed(2)}` : '—', kind: 'compute', note: 'per million tokens at $2.50/hr' },
        ]} />
      }
      controls={
        <>
          <Slider label="Users in the batch" value={batch} min={1} max={512} log onChange={setBatch} format={(v) => String(v)} kind="token" />
          <Slider label="Average context per user" value={ctx} min={128} max={8192} log onChange={setCtx} format={(v) => `${fmt.int(v)} tokens`} kind="memory" />
        </>
      }
    >
      <svg viewBox={`0 0 ${W} 372`} role="img" aria-label="One decode step broken into weight reads, KV cache reads and compute, the GPU memory in use, and the throughput versus per-user speed curve.">
        {/* one decode step as a trace */}
        <Label x={X0} y={16} tone={3} size={10}>one decode step</Label>
        <text x={X0 - 10} y={40} textAnchor="end" className="v-track-label">HBM</text>
        <Span x={X0} y={30} w={g.w} h={14} kind="memory" label="read weights" />
        <Span x={X0 + g.w + 1} y={30} w={g.k} h={14} kind="memory" dim label="read KV" labelTone="light" />
        <text x={X0 - 10} y={64} textAnchor="end" className="v-track-label">Math</text>
        <Span x={X0} y={54} w={Math.max(1.5, g.c)} h={14} kind="compute" />
        {g.c < g.w + g.k && <Span x={X0 + Math.max(1.5, g.c) + 1} y={54} w={g.w + g.k - g.c - 1} h={14} kind="idle" />}
        <Span x={X0 + Math.max(g.w + g.k, g.c) + 1} y={42} w={g.o} h={14} kind="idle" style={{ opacity: 0.8 }} />
        <line x1={X0 + g.t} x2={X0 + g.t} y1={24} y2={76} stroke="var(--text)" />
        <Label x={X0 + g.t + 6} y={52} tone={1} size={10.5}>{fmt.ms(r.t)}</Label>
        {/* tokens produced this step */}
        <text x={X0 - 10} y={96} textAnchor="end" className="v-track-label">Tokens</text>
        {Array.from({ length: shownTokens }, (_, i) => (
          <rect key={i} x={X0 + i * 9.5} y={88} width={7} height={11} rx={1.5} fill="var(--token)" />
        ))}
        {batch > shownTokens && <Label x={X0 + shownTokens * 9.5 + 6} y={97} tone={2} size={10.5} kind="token">+{batch - shownTokens} more</Label>}

        {/* GPU memory */}
        <text x={X0 - 10} y={148} textAnchor="end" className="v-track-label">GPU memory</text>
        <rect x={X0} y={136} width={X1 - X0} height={18} rx={2} fill="none" stroke="var(--rule-2)" />
        <Span x={X0} y={136} w={(X1 - X0) * g.wt} h={18} kind="memory" label="weights 16 GB" />
        <Span x={X0 + (X1 - X0) * g.wt} y={136} w={Math.min((X1 - X0) * (1 - g.wt), (X1 - X0) * g.kv)} h={18} kind="memory" dim label={`KV cache ${(r.kvTotal / 1e9).toFixed(1)} GB`} labelTone="light" />
        <Fade show={!r.fits}>
          <rect x={X0} y={133} width={X1 - X0} height={24} rx={3} fill="url(#tps-hatch-red)" stroke="var(--error)" />
          <text x={(X0 + X1) / 2} y={149} textAnchor="middle" className="v-label bold" style={{ fill: 'var(--error)', fontSize: 12 }}>
            OUT OF MEMORY: needs {(r.memUsed / 1e9).toFixed(0)} GB, the GPU has 80
          </text>
        </Fade>
        <Label x={X1} y={170} anchor="end" tone={3} size={9.5}>{`KV per user = ${fmt.int(ctx)} tokens × 128 KB = ${(r.kvPerSeq / 1e9).toFixed(2)} GB`}</Label>

        {/* trade-off curve */}
        <Label x={CX0} y={CY1 - 30} tone={2} size={10.5} mono={false} bold>The trade-off every serving system makes</Label>
        <line x1={CX0} x2={CX1} y1={CY0} y2={CY0} className="v-rule" />
        <line x1={CX0} x2={CX0} y1={CY1 - 12} y2={CY0} className="v-rule" />
        <Label x={CX1} y={CY0 + 30} anchor="end" tone={3} size={9.5}>total throughput (tokens/s, all users) →</Label>
        <text x={CX0 - 8} y={CY1 - 4} textAnchor="end" className="v-label t3" style={{ fontSize: 9.5 }}>speed per user</text>
        <text x={CX0 - 8} y={CY1 + 8} textAnchor="end" className="v-label t3" style={{ fontSize: 9.5 }}>(tok/s)</text>
        {[0, 8000, 16000, 24000, 32000].map((v) => <Label key={v} x={cx(v)} y={CY0 + 14} anchor="middle" tone={3} size={9}>{v ? `${v / 1000}k` : '0'}</Label>)}
        {[0, 50, 100, 150].map((v) => <Label key={v} x={CX0 - 6} y={cy(v) + 3} anchor="end" tone={3} size={9}>{v}</Label>)}
        <path d={path} fill="none" stroke="var(--text-2)" strokeWidth={1.5} />
        {curve.filter((p) => !p.fits).slice(0, 1).map((p) => (
          <g key="oom"><line x1={cx(p.x)} x2={cx(p.x)} y1={CY1 - 12} y2={CY0} stroke="var(--error)" strokeDasharray="3 3" /><Label x={cx(p.x) - 5} y={CY1 - 4} anchor="end" size={9} kind="error">memory limit</Label></g>
        ))}
        {r.fits && (
          <g className="v-move" style={{ transform: `translate(${cx(r.total)}px, ${cy(r.perUser)}px)` }}>
            <circle r={5.5} fill="var(--token)" stroke="var(--ground-2)" strokeWidth={2} />
            <text x={8} y={-8} className="v-label t1" style={{ fontSize: 10 }}>batch {batch}</text>
          </g>
        )}
      </svg>
    </Stage>
  );
}
