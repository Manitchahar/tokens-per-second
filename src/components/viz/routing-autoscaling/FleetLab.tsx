import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label } from '../lib/svg';
import { simulate, DEFAULT, replicasForRate, SLO, POLICIES, type Config, type Policy } from './fleet';

const STEPS = [
  'The promise',
  'One replica alone',
  'Add capacity',
  'The rate that fits',
  'Traffic shifts',
  'A burst hits',
  'Choose the config',
];
const CAPTIONS = [
  'The service promises: p99 time-to-first-token under a second, and tokens every 50 milliseconds or faster. A request that misses either is not a request served — it is a refund. The grid below is every replica count against every arrival rate, coloured by the fraction of requests that met the promise.',
  'One replica holds to about sixteen requests per second and then degrades. Watch the colour change along that row: green to amber to red, and the boundary is sharper than you expect from a smooth curve.',
  'Every extra replica moves the boundary right. This is the capacity curve you measured in the autoscaling lesson, and it is the single most important chart in capacity planning — it tells you what to buy before the traffic arrives.',
  'The interesting number per row is the last green cell. That is the maximum sustainable rate at that replica count, and it is a capacity figure you can put in a contract.',
  'Now change the traffic mix. Raising the long-prompt share moves every boundary left, because a long prompt occupies the same KV slot as a short one for far more tokens. A fleet sized for chat will not serve a RAG workload.',
  'A burst is different from a sustained load. You can briefly exceed the sustainable rate and keep goodput acceptable, because the queue drains afterwards — but only if the burst is short relative to the drain time.',
  'Now choose: a policy, a replica count with headroom, and the rate you will deploy at. Then defend it. That triple — policy, replicas, rate — is the deliverable.',
];

const RATES = [6, 10, 14, 18, 22, 26, 30, 36, 44];
const REPS = [1, 2, 3, 4, 6, 8];

const PRESET: Record<number, Partial<Config>> = {
  0: { rate: 18, replicas: 4 },
  1: { rate: 18, replicas: 1 },
  2: { rate: 22, replicas: 4 },
  3: { rate: 22, replicas: 4 },
  4: { rate: 22, replicas: 4, longShare: 0.6 },
  5: { rate: 40, replicas: 6 },
  6: { rate: 20, replicas: 5 },
};

/** Goodput at every (replicas, rate) cell — the capacity grid. */
function grid(cfg: Config): number[][] {
  return REPS.map((n) => RATES.map((r) => simulate({ ...cfg, replicas: n, rate: r }).goodput));
}

function band(g: number): 'token' | 'compute' | 'error' {
  return g >= 0.95 ? 'token' : g >= 0.8 ? 'compute' : 'error';
}

export default function FleetLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [cfg, setCfg] = useState<Config>({ ...DEFAULT, ...PRESET[0] });
  useEffect(() => setCfg((c) => ({ ...DEFAULT, ...PRESET[s.step], ...c, rate: PRESET[s.step].rate!, replicas: PRESET[s.step].replicas! })), [s.step]);
  const set = (patch: Partial<Config>) => setCfg((c) => ({ ...c, ...patch }));

  const g = useMemo(() => grid(cfg), [cfg.replicas, cfg.longShare, cfg.shortPrompt, cfg.longPrompt, cfg.outTokens, cfg.policy, cfg.locality, cfg.prefixWeight]);
  const res = useMemo(() => simulate(cfg), [cfg]);

  // Best sustainable rate for each replica count: the last cell still ≥ 90% goodput.
  const cap = useMemo(() => {
    return REPS.map((n, i) => {
      const row = g[i];
      let last = -1;
      row.forEach((v, j) => { if (v >= 0.9) last = j; });
      return last >= 0 ? RATES[last] : 0;
    });
  }, [g]);

  const X = 78, Y = 52, cw = 56, ch = 34;
  const gridW = RATES.length * cw;
  const gridH = REPS.length * ch;

  // Highlight the chosen operating point.
  const rIdx = REPS.indexOf(cfg.replicas);
  const cIdx = RATES.reduce((best, r, i) => (Math.abs(r - cfg.rate) < Math.abs(RATES[best] - cfg.rate) ? i : best), 0);
  const need = replicasForRate(cfg);

  return (
    <Stage
      rootRef={ref}
      title="The capacity grid: replicas × arrival rate"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', ['compute', 'degraded'], ['error', 'past the SLO'], ['comm', 'sustainable capacity'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Goodput', value: `${(res.goodput * 100).toFixed(0)}%`, kind: band(res.goodput), note: `at ${cfg.replicas} replicas, ${cfg.rate} req/s` },
          { label: 'Sustainable here', value: cap[Math.max(0, rIdx)] > 0 ? `${cap[Math.max(0, rIdx)]} req/s` : '—', kind: 'comm', note: `last ≥90% cell in this row` },
          { label: 'p99 TTFT', value: `${(res.p99ttft * 1000).toFixed(0)} ms`, kind: res.p99ttft > SLO.ttft ? 'error' : 'token' },
          { label: 'Mean TPOT', value: `${(res.meanTpot * 1000).toFixed(1)} ms`, kind: 'memory' },
          { label: 'KV spread', value: `${res.imbalance.toFixed(2)}×`, kind: res.imbalance < 1.2 ? 'token' : 'compute' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Routing policy" value={cfg.policy} options={POLICIES.map((p) => ({ value: p.value as Policy, label: p.label }))} onChange={(v) => set({ policy: v })} />
          <Slider label="Arrival rate" value={cfg.rate} min={2} max={48} step={1} onChange={(v) => set({ rate: v })} format={(v) => `${v} req/s`} kind="token" />
          <Slider label="Replicas" value={cfg.replicas} min={1} max={8} step={1} onChange={(v) => set({ replicas: v })} format={(v) => String(v)} kind="comm" />
          <Slider label="Long-prompt share" value={cfg.longShare} min={0} max={0.8} step={0.05} onChange={(v) => set({ longShare: v })} format={(v) => `${(v * 100).toFixed(0)}%`} kind="compute" />
          <Slider label="Traffic correlation" value={cfg.locality} min={0} max={0.9} step={0.05} onChange={(v) => set({ locality: v })} format={(v) => `${(v * 100).toFixed(0)}%`} kind="compute" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A grid of goodput values with replica count down the rows and arrival rate across the columns. Cells are green where most requests meet the service level objective, amber where the service degrades and red where it fails. A marker shows the current operating point and a line traces the sustainable capacity for each replica count.">
        <Label x={0} y={22} tone={2} size={11} bold>Every cell is a full simulated run · colour = goodput</Label>
        <Label x={640} y={22} anchor="end" tone={3} size={9.5}>{`policy: ${POLICIES.find((p) => p.value === cfg.policy)?.label}`}</Label>

        {/* column headers */}
        {RATES.map((r, j) => (
          <Label key={r} x={X + j * cw + cw / 2} y={Y - 8} anchor="middle" tone={3} size={9}>{r}</Label>
        ))}
        <Label x={X + gridW / 2} y={Y - 22} anchor="middle" tone={2} size={9.5}>arrival rate (req/s)</Label>

        {/* the grid */}
        {REPS.map((n, i) => (
          <g key={n}>
            <Label x={X - 10} y={Y + i * ch + ch / 2 + 4} anchor="end" tone={2} size={10}>{n}</Label>
            {RATES.map((r, j) => {
              const v = g[i][j];
              const k = band(v);
              const on = i === rIdx && j === cIdx;
              return (
                <g key={r}>
                  <rect x={X + j * cw} y={Y + i * ch} width={cw - 3} height={ch - 3} rx={2}
                    fill={`var(--${k}-dim)`} stroke={on ? 'var(--text)' : 'transparent'} strokeWidth={1.6}
                    style={{ transition: 'fill .4s var(--ease-out)' }} />
                  <text x={X + j * cw + (cw - 3) / 2} y={Y + i * ch + ch / 2 + 1} textAnchor="middle" className="v-label"
                    style={{ fontSize: 9, fill: k === 'error' ? 'var(--error)' : 'var(--text-3)' }}>
                    {`${(v * 100).toFixed(0)}`}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
        <text x={X - 34} y={Y + gridH / 2} transform={`rotate(-90 ${X - 34} ${Y + gridH / 2})`} textAnchor="middle" className="v-label t3" style={{ fontSize: 9 }}>replicas</text>

        {/* sustainable capacity line: the last green cell in each row */}
        <path
          d={REPS.map((n, i) => {
            const c = cap[i] > 0 ? RATES.indexOf(cap[i]) : -1;
            return c < 0 ? '' : `${i === 0 || cap[i - 1] > 0 ? 'L' : 'M'}${(X + c * cw + (cw - 3) / 2).toFixed(1)},${(Y + i * ch + ch / 2).toFixed(1)}`;
          }).join(' ')}
          fill="none" stroke="var(--comm)" strokeWidth={1.8} strokeDasharray="4 3" opacity={0.9}
        />
        <Label x={X + 4} y={Y + gridH + 14} size={9.5} kind="comm">— — sustainable capacity (last cell at 90% goodput)</Label>

        {/* operating point */}
        {rIdx >= 0 && (
          <g>
            <rect x={X + cIdx * cw - 2} y={Y + rIdx * ch - 2} width={cw + 1} height={ch + 1} rx={3} fill="none" stroke="var(--text)" strokeWidth={2} />
            <Label x={X + cIdx * cw + cw / 2} y={Y + rIdx * ch - 6} anchor="middle" size={9.5} bold>you are here</Label>
          </g>
        )}

        {/* the deliverable */}
        <g>
          <rect x={0} y={306} width={640} height={64} rx={4} fill="var(--ground-3)" opacity={0.5} />
          <Label x={12} y={322} size={10.5} bold tone={2}>The configuration you would ship</Label>
          <Label x={12} y={338} size={10} kind="comm">{`replicas ${cfg.replicas}`}</Label>
          <Label x={130} y={338} size={10} kind="token">{`rate ≤ ${cap[Math.max(0, rIdx)] || 0} req/s`}</Label>
          <Label x={300} y={338} size={10} kind="memory">{`formula says ${need.needed}`}</Label>
          <text x={12} y={356} className="v-label sans" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>
            {res.goodput >= 0.95 ? 'Holding the SLO with margin. Deploy this.' : res.goodput >= 0.8 ? 'Degraded. Add capacity or lower the offered rate before you ship.' : 'Failing the SLO. This is past the cliff — adding a little more rate will not help.'}
          </text>
        </g>
      </svg>
    </Stage>
  );
}