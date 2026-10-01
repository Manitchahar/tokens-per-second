import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Span, fmt } from '../lib/svg';
import { simulateOverload, goodputOf, meanService, HORIZON_S, REQUEST_MIX } from './orMetrics';

// The overload lesson in one screen: three levers (offered load, concurrency pool,
// waiting room) and the five numbers they move. The one non-monotonic curve is goodput.

const STEPS = [
  'A single server, drawn honestly',
  'The levers: load, pool, waiting room',
  'Below capacity, nothing queues',
  'Above capacity, a queue appears',
  'The waiting room is the valve',
  'What happens when you remove the cap',
  'Shedding early beats shedding late',
  'Goodput turns over',
  'Pick the operating point',
];
const CAPTIONS = [
  'One server with a concurrency pool of 40 slots. Each request occupies a slot for its whole duration — from prefill through the last token — not just while it needs the GPU.',
  'Three sliders: offered load in requests per second, the number of concurrent slots, and how many requests may wait before the door is closed.',
  'At 8 req/s against 40 slots, occupancy is around 25%. Almost every request starts immediately. Queue wait is effectively zero.',
  'Past 21 req/s — which is where 40 slots ÷ 1.89 s puts capacity — the pool is full. Requests start waiting, and because service times vary (some are 8-second summaries, some are 0.8-second chats) the queue fills unevenly and the tail appears.',
  'The waiting room is a buffer, not a solution. A buffer of 200 holds the burst and flattens the p95 — but it cannot make the server faster, and it converts rejection into waiting.',
  'Unbounded queues are this panel with the buffer removed. Requests are all admitted, nobody is ever rejected, and latency grows without limit while throughput stays flat.',
  'Load shedding means refusing new work immediately and retryably. The shed requests cost you nothing; the queued ones cost you everything and still miss.',
  'Goodput is throughput multiplied by the fraction of requests whose wait fit the budget. It peaks well below saturation, exactly as Module 06 showed.',
  'The operating point is where goodput peaks, with headroom below it. Running the pool at 95% occupancy buys you nothing and costs you the tail.',
];

const SLO_WAIT = 1.0; // seconds

const MIX = REQUEST_MIX;

export default function OverloadSim() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  // The three sliders are the reader's; the per-step presets say what to look at
  // while scrolling. A manual edit wins until the step changes.
  const [manual, setManual] = useState<{ rate?: number; pool?: number; bufferSize?: number }>({});
  const [stepLoad, setStepLoad] = useState(8);
  const [stepPool, setStepPool] = useState(40);
  const [stepRoom, setStepRoom] = useState(200);

  useEffect(() => {
    setStepLoad(s.step >= 4 ? 22 : 8);
    setStepPool(s.step >= 2 ? 24 : 40);
    setStepRoom(s.step >= 5 ? 200 : 8);
    setManual({});
  }, [s.step]);

  const rate = manual.rate ?? stepLoad;
  const pool = manual.pool ?? stepPool;
  const bufferSize = manual.bufferSize ?? stepRoom;

  const unbounded = s.step >= 5;
  const effBuffer = unbounded ? 1e9 : bufferSize;

  const r = useMemo(
    () => simulateOverload({ rate, pool, bufferSize: effBuffer, seed: 4242, mix: MIX }),
    [rate, pool, effBuffer],
  );
  const g = useMemo(() => goodputOf(r, SLO_WAIT), [r]);

  const serviceS = meanService(MIX);
  const mu = pool / serviceS; // theoretical capacity, req/s

  // a sweep of goodput against offered load, for the final panels
  const sweep = useMemo(() => {
    const out: { rate: number; good: number; thr: number; shed: number }[] = [];
    for (let x = 1; x <= 40; x++) {
      const res = simulateOverload({ rate: x, pool, bufferSize: effBuffer, seed: 4242, mix: MIX });
      const gg = goodputOf(res, SLO_WAIT);
      out.push({ rate: x, good: gg.good, thr: res.throughput, shed: res.shed / Math.max(1, res.offered) });
    }
    return out;
  }, [pool, effBuffer]);

  const best = sweep.reduce((a, b) => (b.good > a.good ? b : a));
  const maxGood = Math.max(...sweep.map((x) => x.good)) || 1;

  const X0 = 60;
  const X1 = 612;
  const Y0 = 268;
  const Y1 = 96;
  const gx = (v: number) => X0 + (v / 40) * (X1 - X0);
  const gy = (v: number) => Y0 - (v / (maxGood * 1.15)) * (Y0 - Y1);

  const occupancy = Math.min(1.2, r.occupancy);
  const offeredShed = r.offered > 0 ? r.shed / r.offered : 0;
  const overSlo = 1 - g.fraction;

  const showPool = s.step >= 1;
  const showQueue = s.step >= 3;
  const showRoom = s.step >= 4;
  const showShed = s.step >= 6;
  const showSweep = s.step >= 7;

  // slot grid: 40 cells in 2 rows
  const occupied = Math.min(40, Math.round(occupancy * 40));
  const slots = Array.from({ length: 40 }, (_, i) => i);

  return (
    <Stage
      rootRef={ref}
      title="One server, three levers, and the point where the work stops being useful"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', ['idle', 'waiting'], 'token', ['error', 'shed / too slow'], ['memory', 'goodput curve']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Offered load', value: `${fmt.f1(rate)} req/s`, kind: 'compute', note: `capacity ≈ ${fmt.f1(mu)} req/s` },
          { label: 'p95 queue wait', value: `${(r.p95 * 1000).toFixed(0)} ms`, kind: r.p95 > SLO_WAIT ? 'error' : 'token', note: `budget ${(SLO_WAIT * 1000).toFixed(0)} ms` },
          { label: 'Shed', value: `${(offeredShed * 100).toFixed(1)}%`, kind: offeredShed > 0.01 ? 'error' : 'token', note: `${fmt.int(r.shed)} of ${fmt.int(r.offered)}` },
          { label: 'Goodput', value: `${fmt.f2(g.good)} req/s`, kind: g.good > 0 ? 'token' : 'error', note: `${(g.fraction * 100).toFixed(0)}% inside the budget` },
        ]} />
      }
      controls={
        <>
          <Slider label="Offered load (requests per second)" value={rate} min={1} max={40} step={0.5} onChange={(v) => setManual((m) => ({ ...m, rate: v }))} format={(v) => `${v.toFixed(1)} req/s`} kind="compute" />
          <Slider label="Concurrency pool (slots)" value={pool} min={4} max={96} step={1} onChange={(v) => setManual((m) => ({ ...m, pool: Math.round(v) }))} format={(v) => `${Math.round(v)} slots`} kind="token" />
          <Slider label="Waiting room (queued requests)" value={bufferSize} min={0} max={600} step={5} onChange={(v) => setManual((m) => ({ ...m, bufferSize: Math.round(v) }))} format={(v) => (Math.round(v) >= 600 ? 'unbounded' : `${Math.round(v)}`)} kind="idle" />
          <Segmented
            label="Queue policy"
            value={unbounded ? 'none' : 'bounded'}
            options={[{ value: 'bounded', label: 'Bounded room' }, { value: 'none', label: 'No cap at all' }]}
            onChange={(v) => { if (v === 'none') setManual((m) => ({ ...m, bufferSize: 600 })); }}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A simulation of a single inference server: a grid of concurrency slots that fill up, a waiting room that grows, requests being shed at the door, and a goodput curve against offered load that peaks below saturation and then falls.">
        <Label x={16} y={16} tone={1} size={11} bold mono={false}>
          {`one replica · pool ${pool} slots · mean service ${serviceS.toFixed(2)} s · horizon ${HORIZON_S} s`}
        </Label>
        <Label x={624} y={16} anchor="end" size={9.5} tone={3}>{`capacity ≈ ${mu.toFixed(1)} req/s · offered ${rate.toFixed(1)} req/s`}</Label>

        {/* ── the concurrency pool ── */}
        {showPool && (
          <Fade show={showPool}>
            <Label x={16} y={40} tone={2} size={10} bold mono={false}>Concurrency pool — a slot is held for the whole request</Label>
            {slots.map((i) => {
              const row = Math.floor(i / 20);
              const col = i % 20;
              const filled = i < occupied;
              return (
                <rect key={i} x={16 + col * 14} y={48 + row * 16} width={11} height={12} rx={1.5}
                  fill={filled ? 'var(--compute)' : 'none'}
                  stroke={filled ? 'var(--compute)' : 'var(--rule-2)'} className="v-cell" />
              );
            })}
            <Label x={300} y={72} size={9} kind="compute" bold>{`${occupied} running`}</Label>
          </Fade>
        )}

        {/* ── the waiting room ── */}
        {showQueue && (
          <Fade show={showQueue}>
            <Label x={16} y={98} tone={2} size={10} bold mono={false}>Waiting room</Label>
            <rect x={110} y={88} width={420} height={16} rx={2} fill="none" stroke="var(--rule-2)" />
            {unbounded ? (
              <rect x={110} y={88} width={420} height={16} rx={2} fill="url(#tps-hatch-red)" stroke="var(--error)" strokeDasharray="3 3" />
            ) : (
              <>
                <rect x={110} y={88} width={420} height={16} rx={2} fill="var(--idle)" stroke="var(--idle)" className="v-cell" />
                <rect x={110 + 420 * 0.72} y={88} width={420 * 0.28} height={16} rx={2} fill="none" stroke="var(--error)" strokeDasharray="2 2" />
              </>
            )}
            <Label x={540} y={100} size={9} kind={r.p95 > SLO_WAIT ? 'error' : 'idle'} bold>
              {unbounded ? 'unbounded — nothing is ever refused' : `${fmt.int(r.waits.length)} admitted late`}
            </Label>
            <Label x={110} y={120} size={9} tone={3}>
              {`p50 wait ${(r.p50 * 1000).toFixed(0)} ms · p95 ${(r.p95 * 1000).toFixed(0)} ms · p99 ${(r.p99 * 1000).toFixed(0)} ms`}
            </Label>
          </Fade>
        )}

        {/* ── the door ── */}
        {showShed && (
          <Fade show={showShed}>
            <Label x={16} y={142} tone={2} size={10} bold mono={false}>At the door</Label>
            <Span x={110} y={132} w={420 * Math.min(1, 1 - offeredShed)} h={16} kind="token" />
            {offeredShed > 0.001 && <Span x={110 + 420 * (1 - offeredShed)} y={132} w={420 * offeredShed} h={16} kind="error" />}
            <Label x={540} y={144} size={9} kind={offeredShed > 0.01 ? 'error' : 'token'} bold>
              {`${(offeredShed * 100).toFixed(1)}% shed instantly`}
            </Label>
            <Label x={110} y={164} size={9} tone={3}>A shed request costs one admission check. A queued request costs a whole service time.</Label>
          </Fade>
        )}

        {/* ── the sweep ── */}
        <Fade show={showSweep}>
          <line x1={X0} x2={X1} y1={Y0} y2={Y0} className="v-rule" />
          <line x1={X0} x2={X0} y1={Y1 - 16} y2={Y0} className="v-rule" />
          {[10, 20, 30, 40].map((v) => (
            <Label key={v} x={gx(v)} y={Y0 + 14} anchor="middle" size={9} tone={3}>{v}</Label>
          ))}
          <Label x={(X0 + X1) / 2} y={Y0 + 28} anchor="middle" size={9.5} tone={2} mono={false}>offered load, requests per second</Label>

          <path d={sweep.map((p, i) => `${i ? 'L' : 'M'}${gx(p.rate).toFixed(1)},${gy(p.thr).toFixed(1)}`).join(' ')}
            fill="none" stroke="var(--compute)" strokeWidth={1.6} />
          <path d={sweep.map((p, i) => `${i ? 'L' : 'M'}${gx(p.rate).toFixed(1)},${gy(p.good).toFixed(1)}`).join(' ')}
            fill="none" stroke="var(--memory)" strokeWidth={2.2} />

          <Label x={X1 - 4} y={gy(best.thr) - 8} anchor="end" size={9.5} kind="compute">throughput</Label>
          <Label x={X1 - 4} y={gy(sweep[10].good) + 20} anchor="end" size={9.5} kind="memory" bold>goodput</Label>

          {rate <= 40 && (
            <g>
              <line x1={gx(rate)} x2={gx(rate)} y1={Y1 - 16} y2={Y0} stroke="var(--text-3)" strokeDasharray="2 3" />
              <circle cx={gx(rate)} cy={gy(r.throughput)} r={4} fill="var(--compute)" stroke="var(--ground-2)" strokeWidth={1.6} />
              <circle cx={gx(rate)} cy={gy(g.good)} r={4} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={1.6} />
            </g>
          )}

          <line x1={gx(best.rate)} x2={gx(best.rate)} y1={Y1 - 16} y2={Y0} stroke="var(--token)" strokeDasharray="3 4" />
          <circle cx={gx(best.rate)} cy={gy(best.good)} r={5} fill="var(--token)" stroke="var(--ground-2)" strokeWidth={2} />
          <Label x={gx(best.rate) + 7} y={gy(best.good) - 6} size={9.5} kind="token" bold>{`peak goodput at ${best.rate} req/s`}</Label>

          <Label x={X0} y={Y0 + 52} size={9.5} tone={2}>
            {`${(overSlo * 100).toFixed(0)}% of admitted requests are waiting too long to be useful at this load.`}
          </Label>
          <Label x={X0} y={Y0 + 68} size={9} tone={3}>
            Throughput keeps rising to the right. Goodput turns over and comes back down. That gap is your money.
          </Label>
          <Label x={X0} y={Y0 + 90} size={10} style={{ fill: 'var(--text)' }} bold>
            {`Operate at ${(best.rate * 0.85).toFixed(1)} req/s, not at ${rate.toFixed(1)}. Below the peak, with headroom for bursts.`}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}