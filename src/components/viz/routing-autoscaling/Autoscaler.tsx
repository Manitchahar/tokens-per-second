import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade } from '../lib/svg';
import { simulate, DEFAULT, replicasForRate, SLO, type Config } from './fleet';

const STEPS = [
  'CPU is the wrong signal',
  'The goodput cliff',
  'Queue depth is the signal',
  'Scale out ahead of time',
  'Look at the whole window',
  'Scale in carefully',
  'The three controls',
];
const CAPTIONS = [
  'A classic autoscaler watches CPU and memory. On an inference replica, CPU is pinned near 100% by the scheduler no matter whether one request or two hundred are in flight — the GPU is the bottleneck, and the CPU is just the thing that feeds it.',
  'So scale on something else. The number that tracks real overload is goodput: as the arrival rate rises, latency stays flat and then falls off a cliff, while throughput has been flat all along. Below the cliff you are fine; above it, tokens keep coming but they are useless.',
  'Queue depth predicts that cliff before it arrives. Depth grows long before latency does, because latency is queueing delay and queueing delay is what queue depth measures. It is a leading indicator, and leading is what you want.',
  'Adding a replica is not instant. It has to be scheduled, the image pulled, weights loaded and warmed — tens of seconds to minutes. So the scaler must act on predicted load, not current load.',
  'A single signal can be fooled by burstiness. A spike of very long prompts raises load without raising request rate; a spike of short ones does the reverse. Watch several windows and the shape of the distribution, not one gauge.',
  'Scaling in is riskier than scaling out. Removing a replica that still holds in-flight KV cache interrupts those requests, so you must drain first and only then take it away.',
  'Three numbers govern autoscaling: the target (a queue depth or an SLO-derived concurrency), the window (how long you average over), and the bounds. Get the bounds wrong and you either thrash or run out.',
];

const PRESET: Record<number, Partial<Config>> = {
  0: { rate: 14, replicas: 4 },
  1: { rate: 14, replicas: 4 },
  2: { rate: 26, replicas: 4 },
  3: { rate: 26, replicas: 4 },
  4: { rate: 30, replicas: 4 },
  5: { rate: 14, replicas: 4 },
  6: { rate: 14, replicas: 4 },
};

const SWEEP = [4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 40];
const MAXP = 40;

export default function Autoscaler() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [cfg, setCfg] = useState<Config>({ ...DEFAULT, ...PRESET[0] });
  useEffect(() => setCfg((c) => ({ ...DEFAULT, ...PRESET[s.step], ...c, rate: PRESET[s.step].rate! })), [s.step]);
  const set = (patch: Partial<Config>) => setCfg((c) => ({ ...c, ...patch }));

  // Sweep the whole curve once at the current settings, so the chart always shows the cliff.
  const curve = useMemo(
    () => SWEEP.map((r) => simulate({ ...cfg, rate: r })),
    [cfg.replicas, cfg.longShare, cfg.shortPrompt, cfg.longPrompt, cfg.outTokens, cfg.policy, cfg.locality, cfg.prefixWeight],
  );
  const res = useMemo(() => simulate(cfg), [cfg]);
  const need = useMemo(() => replicasForRate(cfg), [cfg]);

  const X = 56, Y = 46, W = 556, H = 200;
  const x = (r: number) => X + (r / MAXP) * W;
  const yGood = (g: number) => Y + H - g * H;
  const yTput = (t: number) => Y + H - (t / 2500) * H;
  const yQ = (q: number) => Y + H - Math.min(1, q / 60) * H;

  const maxTput = Math.max(...curve.map((c) => c.tput));
  const poly = (vals: number[], f: (v: number) => number) => vals.map((v, i) => `${i ? 'L' : 'M'}${x(SWEEP[i]).toFixed(1)},${f(v).toFixed(1)}`).join(' ');
  const maxQueue = Math.max(...curve.map((c) => c.meanWait), 1);

  return (
    <Stage
      rootRef={ref}
      title="Autoscaling: the cliff you must not fall off"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', ['compute', 'goodput'], ['error', 'past the cliff'], ['comm', 'throughput'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Goodput', value: `${(res.goodput * 100).toFixed(0)}%`, kind: res.goodput > 0.9 ? 'token' : 'error', note: `p99 TTFT ${(res.p99ttft * 1000).toFixed(0)} ms` },
          { label: 'Mean queue wait', value: `${(res.meanWait * 1000).toFixed(0)} ms`, kind: res.meanWait > SLO.ttft ? 'error' : 'compute', note: 'leading indicator' },
          { label: 'Throughput', value: `${res.tput.toFixed(0)} tok/s`, kind: 'comm', note: 'flat past the cliff' },
          { label: 'Replicas for this rate', value: `${need.needed}`, kind: 'comm', note: `${need.perReplica.toFixed(1)} req/s each` },
        ]} />
      }
      controls={
        <>
          <Slider label="Arrival rate" value={cfg.rate} min={2} max={MAXP} step={1} onChange={(v) => set({ rate: v })} format={(v) => `${v} req/s`} kind="token" />
          <Slider label="Replicas currently running" value={cfg.replicas} min={1} max={8} step={1} onChange={(v) => set({ replicas: v })} format={(v) => String(v)} kind="comm" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A chart of goodput, throughput and mean queue wait against arrival rate for a fixed number of replicas. Goodput stays near 100 percent then falls off a cliff, while throughput continues rising slowly. A vertical marker shows the current operating rate and a dashed line shows the replica count needed.">
        <Label x={0} y={22} tone={2} size={11} bold>Goodput falls off a cliff; throughput does not</Label>
        <Label x={640} y={22} anchor="end" tone={3} size={9.5}>{`${cfg.replicas} replica${cfg.replicas === 1 ? '' : 's'}`}</Label>

        {/* axes */}
        <line x1={X} x2={X} y1={Y} y2={Y + H} className="v-rule" />
        <line x1={X} x2={X + W} y1={Y + H} y2={Y + H} className="v-rule" />
        {[0, 0.5, 1].map((v) => (
          <Label key={v} x={X - 6} y={Y + H - v * H + 4} anchor="end" tone={3} size={9}>{`${(v * 100).toFixed(0)}%`}</Label>
        ))}
        {[0, 10, 20, 30, 40].map((v) => (
          <Label key={v} x={x(v)} y={Y + H + 16} anchor="middle" tone={3} size={9}>{v}</Label>
        ))}
        <text x={X + W / 2} y={Y + H + 34} textAnchor="middle" className="v-label t3" style={{ fontSize: 9 }}>arrival rate (req/s)</text>
        <text x={X - 30} y={Y + H / 2} transform={`rotate(-90 ${X - 30} ${Y + H / 2})`} textAnchor="middle" className="v-label t3" style={{ fontSize: 9 }}>goodput</text>

        {/* throughput (behind), its own scale */}
        <path d={poly(curve.map((c) => c.tput), yTput)} fill="none" stroke="var(--comm)" strokeWidth={1.6} strokeDasharray="4 3" opacity={0.8} />
        <Label x={X + W - 4} y={yTput(maxTput) - 5} anchor="end" size={9} kind="comm">throughput (right scale)</Label>

        {/* queue wait */}
        <Fade show={s.step >= 2}>
          <path d={poly(curve.map((c) => c.meanWait * 1000), (v) => yQ(v / maxQueue))} fill="none" stroke="var(--compute)" strokeWidth={1.3} opacity={0.7} />
          <Label x={X + 6} y={Y + 14} size={9} kind="compute">queue wait rises first</Label>
        </Fade>

        {/* goodput, coloured past the cliff */}
        {curve.map((c, i) => {
          const bad = c.goodput < 0.9;
          return (
            <g key={i}>
              <circle cx={x(SWEEP[i])} cy={yGood(c.goodput)} r={3.4}
                fill={bad ? 'var(--error)' : 'var(--token)'} stroke="var(--ground-2)" strokeWidth={1} />
              {(i === 0 || SWEEP[i] % 8 === 0) && <text x={x(SWEEP[i])} y={yGood(c.goodput) - 8} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{`${(c.goodput * 100).toFixed(0)}`}</text>}
            </g>
          );
        })}

        {/* the SLO threshold */}
        <line x1={X} x2={X + W} y1={yGood(0.9)} y2={yGood(0.9)} stroke="var(--token)" strokeDasharray="5 3" strokeWidth={1} opacity={0.7} />
        <Label x={X + 4} y={yGood(0.9) - 5} size={9} kind="token">90% goodput = the ceiling</Label>

        {/* current operating point */}
        <line x1={x(cfg.rate)} x2={x(cfg.rate)} y1={Y} y2={Y + H} stroke="var(--text)" strokeWidth={1.2} />
        <circle cx={x(cfg.rate)} cy={yGood(res.goodput)} r={5} fill="none" stroke="var(--text)" strokeWidth={1.6} />
        <Label x={x(cfg.rate)} y={Y - 6} anchor="middle" size={9.5} bold>you are here</Label>

        {/* replicas needed */}
        <Fade show={s.step >= 3}>
          <line x1={x(SWEEP[SWEEP.length - 1])} x2={X + W} y1={Y + H + 6} y2={Y + H + 6} className="v-rule faint" />
          <Label x={X} y={Y + H + 6} size={9.5} kind="comm">{`replicas needed at 40 req/s: ${replicasNeededAt(cfg, 40)}`}</Label>
        </Fade>

        {/* signal panel */}
        <g>
          <rect x={0} y={316} width={640} height={56} rx={4} fill="var(--ground-3)" opacity={0.5} />
          <Label x={12} y={332} size={10.5} bold tone={2}>{PANEL[s.step].title}</Label>
          <text x={12} y={348} className="v-label sans" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>{PANEL[s.step].body}</text>
        </g>
      </svg>
    </Stage>
  );
}

const PANEL: { title: string; body: string }[] = [
  { title: 'Signal: CPU', body: 'Pinned near 100% at any concurrency. Useless as a trigger.' },
  { title: 'Signal: goodput', body: 'The thing you actually promised. Reactive, but unambiguous.' },
  { title: 'Signal: queue depth', body: 'Rises before latency does. A leading indicator of the cliff.' },
  { title: 'Action: scale out', body: 'Act on predicted load, because a replica takes a minute to be useful.' },
  { title: 'Signal: the whole window', body: 'Rate alone is fooled by long prompts. Watch the distribution too.' },
  { title: 'Action: scale in', body: 'Drain first. Removing a replica mid-request interrupts its KV cache.' },
  { title: 'Controls: target, window, bounds', body: 'Get the bounds wrong and you either thrash or run out of capacity.' },
];

function replicasNeededAt(cfg: Config, rate: number): number {
  return replicasForRate({ ...cfg, rate }).needed;
}