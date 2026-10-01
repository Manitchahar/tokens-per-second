import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, fmt } from '../lib/svg';
import { simulate, DEFAULT, kvTokensFor, SLO, POLICIES, type Config, type Policy } from './fleet';

const STEPS = [
  'One replica',
  'Round robin',
  'Least queued',
  'Least KV tokens',
  'Power of two',
  'Session affinity',
  'Cache-aware',
  'Pick a policy',
];
const CAPTIONS = [
  'Everything goes to one GPU. It is the whole fleet: every queue, every wait, every tail. This is what a single instance looks like under real traffic, and it is why you scale out at all.',
  'Round robin hands out requests in turn. It is stateless and perfectly fair by count, but it knows nothing about how big each request is — so it will cheerfully send an 8,000-token prompt to a replica that already holds a full KV pool.',
  'Least-queued sends each request to the replica with the fewest waiting. Better, because it reacts to load. But it counts requests, not tokens, so a stream of short requests still hides a stream of long ones.',
  'Least-KV-tokens counts the thing that actually fills the GPU: KV tokens held. It equalises the real pressure, and it is the default most LLM routers converge on.',
  'Power of two samples two replicas and sends to the better one. With many replicas, this balances almost as well as checking every one, at O(1) cost — the reason it is the default in modern L7 balancers.',
  'Session affinity pins a conversation to one replica, so its prefix stays warm. Fewer cold prefixes, but the load it sends is no longer even.',
  'Cache-aware routing scores each replica on how much of this request it already holds. That is the whole lesson: routing on prefix match, not on connection count.',
  'No policy wins everywhere. Round robin is free and fair; least-KV is the robust default; affinity and cache-aware buy real hit rates and pay in balance. Pick from your workload, and measure.',
];

const PRESET: Record<number, Partial<Config>> = {
  0: { policy: 'single', replicas: 1, rate: 14 },
  1: { policy: 'round-robin', replicas: 4, rate: 14 },
  2: { policy: 'least-count', replicas: 4, rate: 14 },
  3: { policy: 'least-tokens', replicas: 4, rate: 14 },
  4: { policy: 'p2c', replicas: 4, rate: 14 },
  5: { policy: 'session', replicas: 4, rate: 14, locality: 0.6 },
  6: { policy: 'prefix', replicas: 4, rate: 14, locality: 0.6 },
  7: { policy: 'least-tokens', replicas: 4, rate: 14 },
};

const KV_CAP = kvTokensFor();

/** Greedy word wrap for SVG <text>, which has no automatic wrapping. */
function wrap(text: string, width: number): string[] {
  const words = text.split(' ');
  const lines: string[] = [];
  let line = '';
  for (const w of words) {
    if (line && line.length + 1 + w.length > width) { lines.push(line); line = w; }
    else line = line ? `${line} ${w}` : w;
  }
  if (line) lines.push(line);
  return lines.slice(0, 2);
}

export default function LoadBalance() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [cfg, setCfg] = useState<Config>({ ...DEFAULT, ...PRESET[0] });
  useEffect(() => setCfg((c) => ({ ...DEFAULT, ...PRESET[s.step], ...c, policy: PRESET[s.step].policy!, replicas: PRESET[s.step].replicas! })), [s.step]);
  const set = (patch: Partial<Config>) => setCfg((c) => ({ ...c, ...patch }));

  const res = useMemo(() => simulate(cfg), [cfg]);
  const n = Math.max(1, Math.min(8, cfg.replicas));
  const peaks = useMemo(() => {
    const p = res.kvPeaks.slice(0, n);
    while (p.length < n) p.push(0);
    return p;
  }, [res, n]);
  const maxPeak = Math.max(KV_CAP * 0.25, ...peaks);
  const animated = useTween(1, 500);

  const pol = POLICIES.find((p) => p.value === cfg.policy)!;
  const goodTone = res.goodput > 0.9 ? 'token' : res.goodput > 0.6 ? 'compute' : 'error';

  // Bars: one per replica, height = peak KV as a fraction of the pool.
  const barX = 40;
  const barW = Math.min(72, (560 - (n - 1) * 12) / n);
  const gap = n > 1 ? (560 - n * barW) / (n - 1) : 0;
  const baseY = 250;
  const maxH = 150;

  return (
    <Stage
      rootRef={ref}
      title="Four replicas, seven routing policies"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', ['token', 'met the SLO'], ['error', 'missed the SLO'], ['idle', 'free KV pool'], ['comm', 'the front door']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Goodput', value: `${(res.goodput * 100).toFixed(0)}%`, kind: goodTone, note: `TTFT < ${SLO.ttft}s and TPOT < ${SLO.tpot * 1000}ms` },
          { label: 'p99 TTFT', value: `${(res.p99ttft * 1000).toFixed(0)} ms`, kind: res.p99ttft > SLO.ttft ? 'error' : 'token' },
          { label: 'Mean TPOT', value: `${(res.meanTpot * 1000).toFixed(1)} ms`, kind: 'memory', note: `${(1000 / Math.max(0.001, res.meanTpot)).toFixed(0)} tok/s per user` },
          { label: 'KV spread', value: `${res.imbalance.toFixed(2)}×`, kind: res.imbalance < 1.15 ? 'token' : 'compute', note: 'busiest ÷ mean replica' },
          { label: 'Prefix hit', value: `${(res.hitRate * 100).toFixed(0)}%`, kind: 'token' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Routing policy" value={cfg.policy} options={POLICIES.map((p) => ({ value: p.value as Policy, label: p.label }))} onChange={(v) => set({ policy: v })} />
          <Slider label="Replicas" value={cfg.replicas} min={1} max={8} step={1} onChange={(v) => set({ replicas: v })} format={(v) => String(v)} kind="comm" />
          <Slider label="Arrival rate" value={cfg.rate} min={2} max={40} step={1} onChange={(v) => set({ rate: v })} format={(v) => `${v} req/s`} kind="token" />
          <Slider label="Traffic correlation (repeat conversations)" value={cfg.locality} min={0} max={0.9} step={0.05} onChange={(v) => set({ locality: v })} format={(v) => `${(v * 100).toFixed(0)}%`} kind="compute" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A fleet of inference replicas behind a load balancer. Each replica is a bar whose fill is its peak KV cache usage; colour of the bar top marks whether that replica's requests met the service level objective. As the routing policy changes from round robin to least-KV-tokens the bars even out; cache-aware routing trades evenness for prefix hits.">
        {/* the front door */}
        <rect x={248} y={30} width={144} height={26} rx={4} fill="none" stroke="var(--comm)" strokeWidth={1.4} />
        <text x={320} y={47} textAnchor="middle" className="v-label" style={{ fontSize: 11, fill: 'var(--comm)', fontWeight: 600 }}>front door · {pol.label}</text>
        <Label x={640} y={47} anchor="end" tone={3} size={9.5}>{`${cfg.rate.toFixed(0)} req/s in`}</Label>

        {/* fan-out to replicas */}
        {Array.from({ length: n }, (_, i) => {
          const cx = barX + i * (barW + gap) + barW / 2;
          return <line key={i} x1={320} y1={56} x2={cx} y2={baseY} stroke="var(--comm)" strokeWidth={1} opacity={0.35} />;
        })}

        {/* KV bars per replica */}
        {Array.from({ length: n }, (_, i) => {
          const frac = Math.min(1, peaks[i] / maxPeak);
          const h = frac * maxH * animated;
          const over = peaks[i] / KV_CAP > 0.85;
          const x = barX + i * (barW + gap);
          return (
            <g key={i}>
              {/* pool outline */}
              <rect x={x} y={baseY - maxH} width={barW} height={maxH} rx={3} fill="none" stroke="var(--rule)" />
              {/* peak fill */}
              <rect x={x} y={baseY - h} width={barW} height={h} rx={3} fill={over ? 'var(--error-dim)' : 'var(--memory-dim)'} stroke={over ? 'var(--error)' : 'var(--memory)'} strokeWidth={1.2} style={{ transition: 'y .5s var(--ease-out), height .5s var(--ease-out)' }} />
              {/* peak marker line */}
              <line x1={x} x2={x + barW} y1={baseY - h} y2={baseY - h} stroke={over ? 'var(--error)' : 'var(--memory)'} strokeWidth={2} />
              <text x={x + barW / 2} y={baseY + 16} textAnchor="middle" className="v-label" style={{ fontSize: 10, fill: 'var(--text)' }}>r{i}</text>
              <text x={x + barW / 2} y={baseY + 28} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{`${Math.round((peaks[i] / KV_CAP) * 100)}% KV`}</text>
              <text x={x + barW / 2} y={baseY + 40} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{fmt.si(peaks[i])} tok</text>
            </g>
          );
        })}

        {/* the KV pool ceiling, annotated once */}
        <Fade show={s.step >= 2}>
          <line x1={barX - 6} x2={barX + n * barW + (n - 1) * gap + 6} y1={baseY - maxH} y2={baseY - maxH} stroke="var(--error)" strokeDasharray="4 4" strokeWidth={1} />
          <Label x={barX + n * barW + (n - 1) * gap + 10} y={baseY - maxH + 4} size={9} kind="error">KV pool ceiling</Label>
        </Fade>

        {/* explanation band */}
        <g>
          <rect x={0} y={306} width={640} height={62} rx={4} fill="var(--ground-3)" opacity={0.5} />
          <Label x={14} y={324} tone={2} size={10.5} bold>{pol.label}</Label>
          {wrap(pol.blurb, 96).map((line, i) => (
            <text key={i} x={14} y={340 + i * 13} className="v-label sans" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>{line}</text>
          ))}
        </g>

        {/* imbalance callout */}
        <Fade show={s.step >= 3}>
          <Label x={14} y={70} size={10} kind={res.imbalance < 1.15 ? 'token' : 'compute'}>
            {`KV spread ${res.imbalance.toFixed(2)}× — ${res.imbalance < 1.15 ? 'evenly loaded' : 'one replica is carrying the fleet'}`}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}