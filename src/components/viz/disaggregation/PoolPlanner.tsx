import { useRef, useState } from 'react';
import { Stage, Readout, Slider, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade, Bar, fmt } from '../lib/svg';
import { prefillPool, decodePool, decodeStep, handoffBytes, prefillPerToken } from './disagg';
import { gpuById } from '../lib/perfModel';

const STEPS = [
  'The four inputs',
  'Prefill pool',
  'Decode pool',
  'The shared-fleet trap',
  'The verdict',
  'Sweep the workload',
  'Where it flips',
  'Write the recommendation',
];
const CAPTIONS = [
  'Four numbers decide the whole design: arrivals per second, prompt tokens, answer tokens, and target concurrency. Everything else is a formula from this course applied to those four.',
  'Prefill is compute-bound, so its pool is sized by FLOPs: GPU-seconds of work per second, rounded up. It is the cheapest half to size because prefill instances hold almost nothing.',
  'Decode is bound twice — by throughput and by KV memory. Whichever demands more instances is the binding constraint, and it is the one you must honour.',
  'The trap is sizing a shared fleet by decode alone. The prefill work still has to happen, and it lands on those same GPUs, inflating every streaming user’s step time.',
  'Here is the comparison the project asks you to produce: the same fleet, split and un-split, with TPOT, capacity and utilisation for each.',
  'Now sweep the workload. A design that works at 20 requests per second can collapse at 40, because prefill demand grows faster than decode slots do.',
  'Find the crossover — the load at which the prefill pool and the decode pool reach their full utilisation at the same time. That point is where the design is balanced, and just past it one pool starts idling.',
  'Finish with a recommendation in one sentence: how many of each, for which workload, against which SLO, and what would make you change your mind.',
];

const GPU = gpuById('h100');
const CTXS = [2048, 4096, 8192, 16384];

export default function PoolPlanner() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [arrivals, setArrivals] = useState(20);
  const [prompt, setPrompt] = useState(4096);
  const [answer, setAnswer] = useState(256);
  const [conc, setConc] = useState(512);
  const [ctx, setCtx] = useState(4096);

  const pp = prefillPool(arrivals, prompt);
  const dp = decodePool(arrivals, answer, conc, ctx);
  const dutyOnShared = pp.gpuSeconds / dp.count;
  const tStep = decodeStep(Math.min(dp.slots, 128), ctx).t;
  const sharedTpot = tStep / (1 - Math.min(0.95, dutyOnShared));
  const splitTpot = tStep;
  const sharedThr = dp.count * dp.perGpu * (1 - Math.min(0.95, dutyOnShared));
  const splitThr = dp.count * dp.perGpu;

  // Sweep curve: total GPUs vs arrival rate.
  const SX0 = 92, SX1 = 612, SY0 = 356, SY1 = 214;
  const maxGpu = 28;
  const sx = (a: number) => SX0 + (Math.log10(Math.max(1, a)) / Math.log10(80)) * (SX1 - SX0);
  const sy = (n: number) => SY0 - (n / maxGpu) * (SY0 - SY1);
  const sweep = [2, 5, 10, 20, 40, 60, 80].map((a) => {
    const p2 = prefillPool(a, prompt);
    const d2 = decodePool(a, answer, conc, ctx);
    return { a, pre: p2.count, dec: d2.count, total: p2.count + d2.count };
  });
  const curTotal = pp.count + dp.count;

  return (
    <Stage
      rootRef={ref}
      title="Designing a disaggregated fleet: Llama-3.1-8B on H100s"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', ['error', 'shared-fleet penalty'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Prefill pool', value: `${pp.count} GPU${pp.count > 1 ? 's' : ''}`, kind: 'compute', note: `${pp.gpuSeconds.toFixed(2)} GPU-s per second` },
          { label: 'Decode pool', value: `${dp.count} GPU${dp.count > 1 ? 's' : ''}`, kind: 'memory', note: `bound by ${dp.byMemory >= dp.byThroughput ? 'KV memory' : 'throughput'}` },
          { label: 'TPOT', value: `${(splitTpot * 1e3).toFixed(1)} ms`, kind: 'token', note: `shared fleet: ${(sharedTpot * 1e3).toFixed(1)} ms` },
          { label: 'KV per request', value: fmt.bytes(handoffBytes(ctx)), kind: 'comm', note: `${fmt.int(ctx)} tokens` },
        ]} />
      }
      controls={
        <>
          <Slider label="Arrivals (req/s)" value={arrivals} min={1} max={80} log onChange={setArrivals}
            format={(v) => `${v.toFixed(1)} req/s`} kind="token" />
          <Slider label="Prompt length (tokens)" value={prompt} min={512} max={32768} log onChange={setPrompt}
            format={(v) => `${fmt.int(v)}`} kind="compute" />
          <Slider label="Answer length (tokens)" value={answer} min={32} max={2048} log onChange={setAnswer}
            format={(v) => `${fmt.int(v)}`} kind="token" />
          <Slider label="Target concurrency" value={conc} min={64} max={4096} log onChange={setConc}
            format={(v) => `${fmt.int(v)} users`} kind="memory" />
          <Segmented label="Context length" value={ctx} options={CTXS.map((c) => ({ value: c, label: `${c / 1024}k` }))} onChange={setCtx} />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="An interactive planner for a disaggregated serving fleet. Controls set arrival rate, prompt length, answer length, concurrency and context. The readout reports prefill pool size, decode pool size, TPOT compared against a shared fleet, and KV payload. A sweep chart shows total GPU count against arrival rate, split into prefill and decode components.">
        {/* steps 1-3: the two pools */}
        <Fade show={s.step <= 2}>
          <Label x={SX0} y={SY1 - 18} tone={2} size={11} mono={false} bold>Two pools, two different formulas</Label>

          <rect x={SX0} y={SY1 - 4} width={242} height={126} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <Label x={SX0 + 12} y={SY1 + 14} size={11} kind="compute" bold>Prefill pool</Label>
          <Label x={SX0 + 12} y={SY1 + 32} tone={3} size={9.5} mono={false}>bound by FLOPs</Label>
          <Label x={SX0 + 12} y={SY1 + 52} tone={3} size={9.5} mono={false}>{`${arrivals.toFixed(1)} × ${fmt.int(prompt)} × ${(prefillPerToken() * 1e6).toFixed(1)}µs`}</Label>
          <Label x={SX0 + 12} y={SY1 + 70} tone={3} size={9.5} mono={false}>{`= ${pp.gpuSeconds.toFixed(2)} GPU-s per second`}</Label>
          <Label x={SX0 + 12} y={SY1 + 92} size={14} kind="compute" bold>{`${pp.count} GPU${pp.count > 1 ? 's' : ''}`}</Label>
          <Label x={SX0 + 12} y={SY1 + 110} tone={3} size={9} mono={false}>holds ~no KV</Label>

          <rect x={SX0 + 256} y={SY1 - 4} width={264} height={126} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <Label x={SX0 + 268} y={SY1 + 14} size={11} kind="memory" bold>Decode pool</Label>
          <Label x={SX0 + 268} y={SY1 + 32} tone={3} size={9.5} mono={false}>bound by throughput AND KV memory</Label>
          <Bar x={SX0 + 268} y={SY1 + 42} w={150} h={9} frac={dp.byThroughput / Math.max(dp.byThroughput, dp.byMemory, 1)} kind="token" />
          <Label x={SX0 + 424} y={SY1 + 50} tone={3} size={9}>{`tput ${dp.byThroughput}`}</Label>
          <Bar x={SX0 + 268} y={SY1 + 58} w={150} h={9} frac={dp.byMemory / Math.max(dp.byThroughput, dp.byMemory, 1)} kind="memory" />
          <Label x={SX0 + 424} y={SY1 + 66} tone={3} size={9}>{`mem ${dp.byMemory}`}</Label>
          <Label x={SX0 + 268} y={SY1 + 86} size={14} kind="memory" bold>{`${dp.count} GPU${dp.count > 1 ? 's' : ''}`}</Label>
          <Label x={SX0 + 268} y={SY1 + 104} tone={3} size={9} mono={false}>{`${dp.slots} seqs each, ${dp.perGpu.toFixed(0)} tok/s`}</Label>
          <Label x={SX0 + 268} y={SY1 + 116} tone={3} size={8.5} mono={false}>
            {`binding: ${dp.byMemory >= dp.byThroughput ? 'KV memory' : 'throughput'}`}
          </Label>
        </Fade>

        {/* step 4-5: the comparison */}
        <Fade show={s.step >= 3 && s.step <= 4}>
          <Label x={SX0} y={SY1 - 18} tone={2} size={11} mono={false} bold>Same fleet, two arrangements</Label>

          <rect x={SX0} y={SY1 - 4} width={520} height={132} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" />

          {/* shared */}
          <Label x={SX0 + 14} y={SY1 + 18} size={10.5} kind="error" bold>{`Shared: ${dp.count} GPUs`}</Label>
          <Label x={SX0 + 14} y={SY1 + 36} tone={3} size={9.5} mono={false}>
            {`prefill duty ${(Math.min(0.99, dutyOnShared) * 100).toFixed(0)}% — it lands on the decode GPUs anyway`}
          </Label>
          <Bar x={SX0 + 14} y={SY1 + 44} w={230} h={10} frac={Math.min(1, sharedTpot / (splitTpot * 2.4))} kind="error" label="TPOT" label="" />
          <Label x={SX0 + 250} y={SY1 + 53} size={10} kind="error">{`${(sharedTpot * 1e3).toFixed(1)} ms`}</Label>
          <Label x={SX0 + 14} y={SY1 + 72} tone={3} size={9.5} mono={false}>{`decode capacity ${sharedThr.toFixed(0)} tok/s vs ${arrivals * answer} needed`}</Label>

          {/* split */}
          <Label x={SX0 + 14} y={SY1 + 96} size={10.5} kind="token" bold>{`Split: ${pp.count} + ${dp.count} = ${curTotal} GPUs`}</Label>
          <Bar x={SX0 + 14} y={SY1 + 102} w={230} h={10} frac={splitTpot / (splitTpot * 2.4)} kind="token" label="" />
          <Label x={SX0 + 250} y={SY1 + 111} size={10} kind="token">{`${(splitTpot * 1e3).toFixed(1)} ms`}</Label>
          <Label x={SX0 + 14} y={SY1 + 124} tone={3} size={9.5} mono={false}>
            {`decode capacity ${splitThr.toFixed(0)} tok/s — ${(splitThr / Math.max(1, arrivals * answer)).toFixed(1)}× headroom`}
          </Label>
        </Fade>

        {/* step 6-8: the sweep */}
        <Fade show={s.step >= 5}>
          <Label x={SX0} y={SY1 - 18} tone={2} size={11} mono={false} bold>How the fleet grows with load</Label>
          <line x1={SX0} x2={SX1} y1={SY0} y2={SY0} className="v-rule" />
          <line x1={SX0} x2={SX0} y1={SY1 - 4} y2={SY0} className="v-rule" />
          {[0, 8, 16, 24].map((v) => (
            <Label key={v} x={SX0 - 6} y={sy(v) + 3} anchor="end" tone={3} size={9}>{v}</Label>
          ))}
          <Label x={SX0 - 6} y={SY1 - 18} anchor="end" tone={3} size={9}>GPUs</Label>
          {[2, 10, 40, 80].map((v) => (
            <Label key={v} x={sx(v)} y={SY0 + 13} anchor="middle" tone={3} size={9}>{v}</Label>
          ))}
          <Label x={SX1} y={SY0 + 27} anchor="end" tone={3} size={9}>arrivals (req/s) →</Label>

          {/* stacked: decode then prefill */}
          {sweep.map((p, i) => {
            const x0 = sx(p.a), x1 = sx(sweep[i + 1]?.a ?? p.a * 1.35);
            return (
              <g key={p.a}>
                <rect x={x0} y={sy(p.dec)} width={Math.max(1, x1 - x0)} height={Math.max(1, SY0 - sy(p.dec))} fill="var(--memory)" opacity={0.75} />
                <rect x={x0} y={sy(p.total)} width={Math.max(1, x1 - x0)} height={Math.max(1, sy(p.dec) - sy(p.total))} fill="var(--compute)" opacity={0.8} />
              </g>
            );
          })}
          {/* the current design marker */}
          <line x1={sx(arrivals)} x2={sx(arrivals)} y1={sy(curTotal)} y2={SY0} stroke="var(--text-3)" strokeDasharray="2 3" />
          <circle cx={sx(arrivals)} cy={sy(curTotal)} r={5} fill="var(--token)" stroke="var(--ground-2)" strokeWidth={2} />
          <Label x={sx(arrivals)} y={sy(curTotal) - 10} anchor="middle" size={9.5} kind="token">{`${curTotal} GPUs`}</Label>

          {/* crossover: where the pools are balanced */}
          {(() => {
            const util = (a: number) => {
              const p2 = prefillPool(a, prompt);
              const d2 = decodePool(a, answer, conc, ctx);
              return p2.gpuSeconds / Math.max(1, p2.count) - (a * answer) / Math.max(1, d2.perGpu * d2.count);
            };
            let cross = 0;
            for (let a = 1; a < 80; a += 1) { if (util(a) <= 0 && util(a - 1) > 0) { cross = a; break; } }
            if (cross > 0 && s.step >= 6) {
              return (
                <g>
                  <line x1={sx(cross)} x2={sx(cross)} y1={SY1 - 4} y2={SY0} stroke="var(--token)" strokeWidth={1.2} strokeDasharray="4 3" />
                  <Label x={sx(cross) + 5} y={SY1 + 4} size={9.5} kind="token">{`balanced ≈ ${cross} req/s`}</Label>
                </g>
              );
            }
            return null;
          })()}

          <g>
            <rect x={SX1 - 116} y={SY1 - 2} width={10} height={8} fill="var(--memory)" opacity={0.75} />
            <Label x={SX1 - 102} y={SY1 + 6} size={9} tone={3}>decode</Label>
            <rect x={SX1 - 62} y={SY1 - 2} width={10} height={8} fill="var(--compute)" opacity={0.8} />
            <Label x={SX1 - 48} y={SY1 + 6} size={9} tone={3}>prefill</Label>
          </g>

          <Label x={SX0} y={SY0 + 46} tone={3} size={9.5}>
            {s.step >= 7
              ? 'Write the recommendation: how many of each, for which workload, against which SLO.'
              : 'Prefill demand grows with prompt length; decode slots grow with context. One curve rarely fits both.'}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
