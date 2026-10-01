import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Dim, Fade, fmt } from '../lib/svg';
import { simulate, DEFAULT, SLO, prefillTime, KV_PER_TOKEN, type Config } from './engine';

const STEPS = [
  'A mixed workload',
  'Turn off everything',
  'Chunked prefill only',
  'Add the prefix cache',
  'Push the rate',
  'Pick the config',
];
const CAPTIONS = [
  'A queue of short chat prompts and long RAG prompts sharing one GPU. This is what most production traffic actually looks like, and the two kinds want opposite things from the scheduler.',
  'Unoptimized, one long prompt takes the GPU in a single step and everyone behind it waits. Watch p99 time-to-first-token: the long prompts dominate it entirely.',
  'Chunking cuts every prefill into pieces that fit inside one engine step. The median barely moves, but the tail stops being a cliff — this is a fairness fix, not a speed fix.',
  'The prefix cache removes the shared system prompt and retrieved context from the bill entirely. Now the long prompts cost only their unique tokens, and throughput moves as well as latency.',
  'Raise the arrival rate and watch the trade-off: throughput per user falls, total throughput rises, and goodput — the fraction meeting the SLO — is what falls off first.',
  'No single config wins. Chunk small enough for fairness, cache as much as the prefix tree holds, and accept the throughput ceiling that physics sets.',
];

const PRESET: Record<number, Partial<Config>> = {
  0: { rate: 3, longShare: 0.35, chunked: true, chunk: 512, cache: true },
  1: { rate: 3, longShare: 0.35, chunked: false, chunk: 8192, cache: false },
  2: { rate: 3, longShare: 0.35, chunked: true, chunk: 512, cache: false },
  3: { rate: 3, longShare: 0.35, chunked: true, chunk: 512, cache: true },
  4: { rate: 8, longShare: 0.35, chunked: true, chunk: 512, cache: true },
  5: { rate: 3, longShare: 0.35, chunked: true, chunk: 512, cache: true },
};

export default function MixedWorkload() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [cfg, setCfg] = useState<Config>({ ...DEFAULT, ...PRESET[0] });
  const set = (patch: Partial<Config>) => {
    setCfg({ ...DEFAULT, ...PRESET[s.step], ...cfg, ...patch });
  };

  const res = useMemo(() => simulate(cfg), [cfg]);
  const base = useMemo(() => simulate({ ...DEFAULT, ...PRESET[1] }), []);

  // histogram of TTFT, one bar per request, coloured by whether it met the SLO
  const maxTtft = Math.max(0.6, ...res.reqs.map((r) => r.ttft));
  const barW = Math.max(2, 570 / DEFAULT.count - 3);

  const tp50 = res.p50ttft * 1000;
  const tp99 = res.p99ttft * 1000;
  const base50 = base.p50ttft * 1000;
  const base99 = base.p99ttft * 1000;

  return (
    <Stage
      rootRef={ref}
      title="Mixed long/short workload: what each knob is worth"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', ['memory', 'cached KV'], 'token', ['error', 'missed the SLO'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'p50 TTFT', value: `${tp50.toFixed(0)} ms`, kind: 'text', note: `unoptimized: ${base50.toFixed(0)} ms` },
          { label: 'p99 TTFT', value: `${tp99.toFixed(0)} ms`, kind: tp99 > 1000 ? 'error' : 'token', note: `unoptimized: ${base99.toFixed(0)} ms` },
          { label: 'Mean TPOT', value: `${(res.meanTpot * 1000).toFixed(1)} ms`, kind: 'memory', note: `${(1000 / res.meanTpot).toFixed(0)} tok/s per user` },
          { label: 'Goodput', value: `${(res.goodput * 100).toFixed(0)}%`, kind: res.goodput > 0.8 ? 'token' : 'error', note: `TTFT &lt; 1 s and TPOT &lt; ${SLO.tpot * 1000} ms` },
          { label: 'Throughput', value: `${res.tput.toFixed(0)} tok/s`, kind: 'compute', note: `all users, ${res.reqs.length} requests` },
        ]} />
      }
      controls={
        <>
          <Slider label="Arrival rate (requests per second)" value={cfg.rate} min={0.5} max={20} step={0.5} onChange={(v) => set({ rate: v })} format={(v) => `${v} req/s`} kind="token" />
          <Slider label="Share of requests that are long (RAG-style prompts)" value={cfg.longShare} min={0} max={1} step={0.05} onChange={(v) => set({ longShare: v })} format={(v) => `${(v * 100).toFixed(0)}%`} kind="compute" />
          <Segmented label="Prefix cache" value={cfg.cache ? 'on' : 'off'} options={[{ value: 'off', label: 'Off' }, { value: 'on', label: 'On' }]} onChange={(v) => set({ cache: v === 'on' })} />
          <Segmented label="Chunked prefill" value={cfg.chunked ? 'on' : 'off'} options={[{ value: 'off', label: 'Off' }, { value: 'on', label: 'On' }]} onChange={(v) => set({ chunked: v === 'on' })} />
          {cfg.chunked && (
            <Slider label="Chunk size (tokens)" value={cfg.chunk} min={128} max={4096} log onChange={(v) => set({ chunk: v })} format={(v) => `${fmt.int(v)}`} kind="compute" />
          )}
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A bar chart of time-to-first-token for every request in a mixed workload, coloured by whether it met the SLO, with a marker at the p50 and p99 lines and the unoptimized baseline shown for comparison.">
        {/* axis */}
        <Label x={0} y={20} tone={2} size={11} bold>Time to first token, request by request</Label>
        <Label x={600} y={20} anchor="end" tone={3} size={9.5}>{`${res.reqs.length} requests · horizon ${res.horizon.toFixed(1)} s · ${res.steps} engine steps`}</Label>
        <line x1={0} y1={312} x2={600} y2={312} className="v-rule" />
        {[0, 0.5, 1, 1.5, 2].map((v) => (
          <Label key={v} x={-4} y={312 - (v / maxTtft) * 260 + 3} anchor="end" tone={3} size={9}>{v.toFixed(1)}</Label>
        ))}
        <text x={-16} y={190} transform="rotate(-90 -16 190)" textAnchor="middle" className="v-label t3" style={{ fontSize: 9 }}>seconds</text>

        {/* the SLO line */}
        <line x1={0} x2={600} y1={312 - (SLO.ttft / maxTtft) * 260} y2={312 - (SLO.ttft / maxTtft) * 260} stroke="var(--token)" strokeWidth={1.3} strokeDasharray="5 3" />
        <Label x={600} y={312 - (SLO.ttft / maxTtft) * 260 - 5} anchor="end" size={9.5} kind="token">SLO: 1 s</Label>

        {/* the p99 of the unoptimized baseline, for comparison */}
        <Fade show={s.step >= 1 && base99 > tp99}>
          <line x1={0} x2={600} y1={312 - (base99 / maxTtft) * 260} y2={312 - (base99 / maxTtft) * 260} stroke="var(--error)" strokeWidth={1.2} strokeDasharray="2 4" />
          <Label x={4} y={312 - (base99 / maxTtft) * 260 - 5} size={9.5} kind="error">{`unoptimized p99: ${base99.toFixed(0)} ms`}</Label>
        </Fade>

        {/* one bar per request */}
        {res.reqs.map((r, i) => {
          const h = Math.max(2, (r.ttft / maxTtft) * 260);
          return (
            <g key={r.id}>
              <rect x={i * (600 / DEFAULT.count)} y={312 - h} width={barW} height={h} rx={1.5}
                fill={r.ok ? 'var(--token)' : 'var(--error)'} opacity={r.ok ? 0.8 : 0.75} />
              {i % 6 === 0 && <Label x={i * (600 / DEFAULT.count) + barW / 2} y={326} anchor="middle" tone={3} size={8.5}>{String(i)}</Label>}
            </g>
          );
        })}
        <Label x={600} y={340} anchor="end" tone={3} size={9}>requests, in arrival order →</Label>

        {/* median marker */}
        <g>
          <line x1={0} x2={600} y1={312 - (res.p50ttft / maxTtft) * 260} y2={312 - (res.p50ttft / maxTtft) * 260} stroke="var(--compute)" strokeWidth={1.2} />
          <Label x={2} y={312 - (res.p50ttft / maxTtft) * 260 + 12} size={9.5} kind="compute">{`p50 ${tp50.toFixed(0)} ms`}</Label>
        </g>

        {/* what one long prompt costs on its own */}
        <Label x={0} y={356} tone={3} size={9.5}>
          {`One 8,000-token prompt, alone: ${(prefillTime(8000) * 1000).toFixed(0)} ms of GPU. Its KV cache alone is ${fmt.bytes(8000 * KV_PER_TOKEN)}.`}
        </Label>
        <Label x={600} y={356} anchor="end" size={9.5} kind={res.goodput > 0.8 ? 'token' : 'error'}>
          {`${(res.goodput * 100).toFixed(0)}% of requests met the SLO`}
        </Label>
        <Fade show={s.step >= 4}>
          <Dim x1={0} x2={600} y={368} label={`${(base.p99ttft / Math.max(1e-9, res.p99ttft)).toFixed(1)}× p99 reduction vs unoptimized`} kind="token" />
        </Fade>
      </svg>
    </Stage>
  );
}