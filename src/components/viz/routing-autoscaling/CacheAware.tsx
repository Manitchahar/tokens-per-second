import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade } from '../lib/svg';
import { simulate, DEFAULT, kvTokensFor, SLO, POLICIES, type Config } from './fleet';

const STEPS = [
  'A conversation returns',
  'Reuse is worth a lot',
  'Affinity pins it',
  'Affinity hot-spots',
  'Score both terms',
  'Turn the dial',
  'When to use which',
];
const CAPTIONS = [
  'A user comes back to the same conversation. Their next request repeats the whole transcript, so any replica that already holds that prefix can skip its prefill.',
  'Reuse is not a rounding error. At a 35% hit rate you pay 65% of the cold prefill bill, and you skip the GPU work entirely — this is a throughput win, not just a latency one.',
  'Session affinity pins the conversation to the replica that already owns it. The hit rate jumps, and mean TPOT falls with it, because shorter prompts prefill faster.',
  'But affinity has no load term in its score, so a popular conversation always returns to the same GPU. The spread gets worse, and when that replica fills, the wait is much worse than the average suggests.',
  'Cache-aware routing scores every replica on both: how much of this prefix it holds, minus how full it is. Both terms normalised to 0–1, because a token count and a memory fraction are otherwise incomparable.',
  'The load weight α slides between pure affinity (0) and pure least-loaded (very large). Sweep it and watch hit rate fall as balance improves — this is the trade-off you are actually tuning.',
  'High repetition plus long shared prefixes: cache-aware wins big. Independent prompts: it degenerates to least-loaded, which is fine. The dial only pays when reuse is real.',
];

const PRESET: Record<number, Partial<Config>> = {
  0: { policy: 'session', prefixWeight: 0, locality: 0.9, rate: 14 },
  1: { policy: 'prefix', prefixWeight: 8, locality: 0.9, rate: 14 },
  2: { policy: 'session', prefixWeight: 0, locality: 0.9, rate: 14 },
  3: { policy: 'prefix', prefixWeight: 0, locality: 0.9, rate: 14 },
  4: { policy: 'prefix', prefixWeight: 8, locality: 0.9, rate: 14 },
  5: { policy: 'prefix', prefixWeight: 8, locality: 0.9, rate: 14 },
  6: { policy: 'prefix', prefixWeight: 32, locality: 0.9, rate: 14 },
};

const KV_CAP = kvTokensFor();
const N = 4;

export default function CacheAware() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [cfg, setCfg] = useState<Config>({ ...DEFAULT, ...PRESET[0] });
  useEffect(() => setCfg((c) => ({ ...DEFAULT, ...PRESET[s.step], ...c, policy: PRESET[s.step].policy! })), [s.step]);
  const set = (patch: Partial<Config>) => setCfg((c) => ({ ...c, ...patch }));

  const res = useMemo(() => simulate(cfg), [cfg]);
  // A reference run at the same settings but with no cache reuse at all, for contrast.
  const cold = useMemo(() => simulate({ ...cfg, policy: 'least-tokens', locality: 0 }), [cfg.locality, cfg.rate, cfg.longPrompt, cfg.shortPrompt, cfg.outTokens, cfg.replicas]);

  const peaks = useMemo(() => {
    const p = res.kvPeaks.slice(0, N);
    while (p.length < N) p.push(0);
    return p;
  }, [res]);
  const maxPeak = Math.max(KV_CAP * 0.2, ...peaks);
  const hit = res.hitRate;
  const barX = 46;
  const barW = 78;
  const gap = 28;
  const baseY = 236;
  const maxH = 140;

  const score = (i: number) => {
    const load = peaks[i] / KV_CAP;
    // Illustrate the scoring for one chosen conversation: assume replica i holds a
    // prefix worth (N-i)/N of the prompt, so later replicas are progressively colder.
    const assumed = ((N - i) / N) * 0.9;
    return assumed - cfg.prefixWeight * load;
  };

  return (
    <Stage
      rootRef={ref}
      title="Cache-aware routing: reuse versus balance"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', ['token', 'warm prefix'], ['error', 'hot replica'], ['comm', 'score'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Prefix hit rate', value: `${(hit * 100).toFixed(0)}%`, kind: hit > 0.3 ? 'token' : 'error', note: `prefill cost × ${(1 - hit).toFixed(2)}` },
          { label: 'Mean TPOT', value: `${(res.meanTpot * 1000).toFixed(1)} ms`, kind: 'memory', note: `no reuse: ${(cold.meanTpot * 1000).toFixed(1)} ms` },
          { label: 'KV spread', value: `${res.imbalance.toFixed(2)}×`, kind: res.imbalance < 1.2 ? 'token' : 'error', note: 'busiest ÷ mean' },
          { label: 'p99 TTFT', value: `${(res.p99ttft * 1000).toFixed(0)} ms`, kind: res.p99ttft > SLO.ttft ? 'error' : 'token' },
          { label: 'Goodput', value: `${(res.goodput * 100).toFixed(0)}%`, kind: res.goodput > 0.9 ? 'token' : 'compute' },
        ]} />
      }
      controls={
        <>
          <Slider label="α · how much load beats cache (0 = pure affinity)" value={cfg.prefixWeight} min={0} max={256} step={1} log={false} onChange={(v) => set({ prefixWeight: v })} format={(v) => v.toFixed(0)} kind="comm" />
          <Slider label="Traffic correlation (repeat conversations)" value={cfg.locality} min={0} max={0.95} step={0.05} onChange={(v) => set({ locality: v })} format={(v) => `${(v * 100).toFixed(0)}%`} kind="compute" />
          <Slider label="Long-prompt share" value={cfg.longShare} min={0} max={0.8} step={0.05} onChange={(v) => set({ longShare: v })} format={(v) => `${(v * 100).toFixed(0)}%`} kind="compute" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Four inference replicas shown as KV-usage bars. As the routing policy switches from session affinity to cache-aware scoring, a conversation is routed to the replica that holds most of its prefix, trading prefix hit rate against even load. The load weight alpha slides the balance between the two.">
        <Label x={0} y={20} tone={2} size={11} bold>Where each replica stands, and what the router sees</Label>
        <Label x={640} y={20} anchor="end" tone={3} size={9.5}>{`policy: ${POLICIES.find((p) => p.value === cfg.policy)?.label} · α=${cfg.prefixWeight.toFixed(0)}`}</Label>

        {/* per-replica bars */}
        {Array.from({ length: N }, (_, i) => {
          const x = barX + i * (barW + gap);
          const frac = Math.min(1, peaks[i] / maxPeak);
          const h = frac * maxH;
          const hottest = peaks[i] === Math.max(...peaks);
          return (
            <g key={i}>
              <rect x={x} y={baseY - maxH} width={barW} height={maxH} rx={3} fill="none" stroke="var(--rule)" />
              <rect x={x} y={baseY - h} width={barW} height={h} rx={3}
                fill={hottest && res.imbalance > 1.4 ? 'var(--error-dim)' : 'var(--memory-dim)'}
                stroke={hottest && res.imbalance > 1.4 ? 'var(--error)' : 'var(--memory)'} strokeWidth={1.2}
                style={{ transition: 'y .5s var(--ease-out), height .5s var(--ease-out)' }} />
              <text x={x + barW / 2} y={baseY + 15} textAnchor="middle" className="v-label" style={{ fontSize: 10 }}>r{i}</text>
              <text x={x + barW / 2} y={baseY + 27} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{`${Math.round((peaks[i] / KV_CAP) * 100)}% KV`}</text>
            </g>
          );
        })}

        {/* the scoring: which replica would this request pick, and why */}
        <Fade show={s.step >= 4}>
          <line x1={0} x2={640} y1={272} y2={272} className="v-rule faint" />
          <Label x={0} y={290} tone={2} size={10.5} bold>Score for one conversation: prefix held − α × KV used</Label>
          {Array.from({ length: N }, (_, i) => {
            const x = barX + i * (barW + gap);
            const v = score(i);
            const chosen = v === Math.max(...Array.from({ length: N }, (_, j) => score(j)));
            return (
              <g key={`sc${i}`}>
                <text x={x + barW / 2} y={308} textAnchor="middle" className="v-label t3" style={{ fontSize: 9 }}>r{i}</text>
                <text x={x + barW / 2} y={322} textAnchor="middle" className="v-label"
                  style={{ fontSize: 10, fill: chosen ? 'var(--token)' : 'var(--text-3)', fontWeight: chosen ? 700 : 400 }}>
                  {v.toFixed(2)}
                </text>
                {chosen && <text x={x + barW / 2} y={336} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: 'var(--token)' }}>picked</text>}
              </g>
            );
          })}
        </Fade>

        {/* the trade-off band */}
        <Fade show={s.step >= 5}>
          <g>
            <rect x={0} y={344} width={640} height={28} rx={4} fill="var(--ground-3)" opacity={0.5} />
            <Label x={12} y={362} size={10} kind="token">{`hit ${(hit * 100).toFixed(0)}%`}</Label>
            <Label x={120} y={362} size={10} kind="error">{`spread ${res.imbalance.toFixed(2)}×`}</Label>
            <text x={220} y={362} className="v-label sans" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>
              {res.imbalance > 2.2 ? 'Affinity is winning too hard — one replica is carrying the fleet.' : hit < 0.1 ? 'No reuse left — the router has effectively become least-loaded.' : 'A balance: most reuse captured, load still roughly even.'}
            </text>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}