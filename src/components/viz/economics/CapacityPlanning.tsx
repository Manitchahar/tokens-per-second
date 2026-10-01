import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, fmt } from '../lib/svg';
import { serviceTime, stepAt, fleetSize, monthlyCost, perHour, gpuById, ctxFor, type Op } from './econ';

// Capacity planning is a division problem: how much machine time does one request
// consume, how many requests do I promise to accept, and what happens to the answer
// when the arrival rate is twice what you sized for.

const STEPS = [
  'Size the machine, not the peak',
  'One request, in GPU-seconds',
  'Arrival rate is the other input',
  'Busy GPUs: the product of the two',
  'Divide by your target utilization',
  'Headroom is not optional',
  'The same fleet, five arrival rates',
  'What you actually run',
];
const CAPTIONS = [
  'Peak is the wrong number. A fleet sized for peak sits idle for most of the day and still falls over at 6pm, because peak is when the queue is deepest and there is no slack anywhere in the system.',
  'Start from the one quantity you can actually control: how much machine time a single request consumes. Prefill, plus the request\'s share of each decode step. For an 8B model on an H100 with 64 streams, about 73 milliseconds.',
  'Multiply by the arrival rate and you get the number of GPUs that must be busy simultaneously to retire that traffic. This is pure arithmetic — it does not care about your GPU utilization graphs.',
  'At 12 requests per second each consuming 73 ms of machine time, 0.87 GPUs must be busy. That is not a fleet size, it is a floor.',
  'Now divide by your target utilization. Running at 60% instead of 100% is not waste: it is the queue having somewhere to go when traffic is uneven. Here it costs one whole extra card — and it is what keeps the tail flat.',
  'Headroom is what you hold for the burst, the deploy, the traffic spike you did not predict. Without it every incident is a capacity problem, and every capacity problem is an incident.',
  'The last three rows are the answer to the question leadership actually asks: what does the bill look like at two, six, twelve and thirty requests per second — and where does it break. At thirty you need four cards to stay under 60%, so the question becomes how fast new replicas appear.',
  'What you run is a floor plus headroom, with autoscaling as the thing that closes the gap between them. Everything else is a guess with a spreadsheet attached.',
];

const PRESET: Record<number, number> = { 0: 12, 1: 12, 2: 12, 3: 12, 4: 12, 5: 12, 6: 12, 7: 12 };

const RATES = [2, 6, 12, 30];

export default function CapacityPlanning() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [rps, setRps] = useState(PRESET[0]);
  const [nIn, setNIn] = useState(512);
  const [nOut, setNOut] = useState(256);
  const [target, setTarget] = useState(0.6);

  useEffect(() => setRps(PRESET[s.step]), [s.step]);

  const op: Op = { modelId: 'llama8b', gpuId: 'h100', batch: 64, ctx: ctxFor(nIn), wBytes: 2 };
  const st = serviceTime(op, nOut, nIn);
  const f = fleetSize(op, { reqPerSec: rps, nIn, nOut }, target);
  const bill = monthlyCost(f.gpus, 'h100');
  const h100 = gpuById('h100');

  const BUS_X = 168, BUS_W = 380;
  const showTime = s.step >= 1;
  const showRate = s.step >= 2;
  const showBusy = s.step >= 3;
  const showTarget = s.step >= 4;
  const showHead = s.step >= 5;
  const showTable = s.step >= 6;
  const showRun = s.step >= 7;

  return (
    <Stage
      rootRef={ref}
      title="Capacity planning: the division you do before you buy anything"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'prefill time'],
        ['memory', 'decode time'],
        ['token', 'work arriving'],
        ['error', 'over capacity'],
        'idle',
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Machine time / request', value: `${(st.gpuSeconds * 1000).toFixed(1)} ms`, kind: 'memory', note: `${(st.prefillFrac * 100).toFixed(0)}% prefill` },
          { label: 'Arrival rate', value: `${rps} req/s`, kind: 'token', note: `${fmt.int(nIn)} in / ${fmt.int(nOut)} out` },
          { label: 'GPUs must be busy', value: f.busyGpus.toFixed(2), kind: 'memory', note: 'a floor, not a fleet size' },
          { label: 'Fleet at target', value: `${f.gpus} GPU${f.gpus === 1 ? '' : 's'}`, kind: f.busyGpus > f.gpus * target ? 'error' : 'token', note: `$${fmt.int(bill)}/month at ${(target * 100).toFixed(0)}% target` },
        ]} />
      }
      controls={
        <>
          <Slider label="Arrival rate" kind="token" value={rps} min={1} max={60} onChange={setRps} format={(v) => `${v} req/s`} />
          <Slider label="Target utilization" kind="memory" value={target} min={0.3} max={0.95} step={0.05} onChange={setTarget} format={(v) => `${(v * 100).toFixed(0)}%`} />
          <Slider label="Prompt tokens" kind="compute" value={nIn} min={64} max={8192} log onChange={setNIn} format={(v) => `${fmt.int(v)}`} />
          <Slider label="Output tokens" kind="token" value={nOut} min={8} max={1024} log onChange={setNOut} format={(v) => `${fmt.int(v)}`} />
        </>
      }
    >
      <svg viewBox="0 0 640 360" role="img" aria-label="A capacity planning calculation: machine time per request, multiplied by arrival rate, divided by target utilization, giving a fleet size and a monthly bill, with a table of the answer at four different arrival rates.">
        <Label x={16} y={20} tone={2} size={11} bold>From one request to a fleet size, in three operations</Label>
        <Label x={624} y={20} anchor="end" tone={3} size={9}>Llama-3.1-8B · H100 · 64 streams</Label>

        {/* ── op 1: machine time per request ── */}
        <Label x={16} y={48} tone={3} size={9.5}>1 · machine time per request</Label>
        <g>
          <rect x={BUS_X} y={38} width={BUS_W} height={18} rx={2} fill="none" stroke="var(--rule)" />
          {showTime ? (
            <>
              <Span x={BUS_X} y={38} w={(st.tPre / st.gpuSeconds) * BUS_W} h={18} kind="compute" />
              <Span x={BUS_X + (st.tPre / st.gpuSeconds) * BUS_W} y={38} w={(st.decodeShare / st.gpuSeconds) * BUS_W} h={18} kind="memory" />
            </>
          ) : (
            <rect x={BUS_X} y={38} width={BUS_W} height={18} rx={2} fill="url(#tps-hatch)" stroke="var(--idle)" />
          )}
          <Label x={624} y={52} anchor="end" size={10.5} tone={1} bold>{`${(st.gpuSeconds * 1000).toFixed(1)} ms`}</Label>
        </g>

        {/* ── op 2: multiply by arrival rate ── */}
        <Fade show={showRate}>
          <g>
            <Label x={16} y={84} tone={3} size={9.5}>2 × arrival rate</Label>
            <Label x={116} y={84} size={10.5} tone={2}>{`${rps} req/s`}</Label>
            <Label x={188} y={84} size={10.5} tone={3}>=</Label>
            <Label x={208} y={84} size={11} kind="memory" bold>{`${f.busyGpus.toFixed(2)} GPUs busy`}</Label>
            <Label x={340} y={84} size={9} tone={3}>— this is the floor, not the answer</Label>
          </g>
        </Fade>

        {/* ── op 3: divide by target ── */}
        <Fade show={showTarget}>
          <g>
            <Label x={16} y={110} tone={3} size={9.5}>3 ÷ target utilization</Label>
            <Label x={116} y={110} size={10.5} tone={2}>{`${(target * 100).toFixed(0)}%`}</Label>
            <Label x={148} y={110} size={10.5} tone={3}>=</Label>
            <Label x={168} y={110} size={11} kind="token" bold>{`${f.atTarget.toFixed(2)} → ${f.gpus} GPU${f.gpus === 1 ? '' : 's'}`}</Label>
            <Label x={330} y={110} size={9} tone={3}>— round up, and hold the extra</Label>
          </g>
        </Fade>

        <line x1={16} x2={624} y1={126} y2={126} className="v-rule faint" />

        {/* ── headroom ── */}
        <Fade show={showHead}>
          <g>
            <Label x={16} y={148} tone={2} size={10.5} bold>Headroom is the difference between a plan and a hope</Label>
            <Label x={16} y={166} size={9.5} tone={3}>
              {`Running at 100% would size this at ${Math.max(1, Math.ceil(f.busyGpus))} GPU and buy you a queue that grows without bound.`}
            </Label>
            <Label x={16} y={182} size={9.5} kind="error" mono={false}>
              {`Running at ${(target * 100).toFixed(0)}% costs ${(1 / target).toFixed(2)}× the arithmetic: ${f.gpus} GPUs, $${fmt.int(bill)}/month.`}
            </Label>
          </g>
        </Fade>

        {/* ── the four rates ── */}
        <Fade show={showTable}>
          <g>
            <Label x={16} y={212} tone={2} size={10.5} bold>The same fleet at four arrival rates</Label>
            <Label x={16} y={230} size={9} tone={3}>rate</Label>
            <Label x={80} y={230} size={9} tone={3}>busy GPUs</Label>
            <Label x={180} y={230} size={9} tone={3}>at target</Label>
            <Label x={276} y={230} size={9} tone={3}>fleet</Label>
            <Label x={340} y={230} size={9} tone={3}>/month</Label>
            <Label x={430} y={230} size={9} tone={3}>verdict</Label>
            <line x1={16} x2={624} y1={234} y2={234} className="v-rule faint" />
            {RATES.map((r, i) => {
              const x = fleetSize(op, { reqPerSec: r, nIn, nOut }, target);
              const over = r === rps && x.busyGpus > x.gpus * target;
              return (
                <g key={r}>
                  <rect x={12} y={240 + i * 17} width={612} height={16} rx={2}
                    fill={r === rps ? 'var(--token)' : 'transparent'} fillOpacity={r === rps ? 0.1 : 0} />
                  <Label x={16} y={252 + i * 17} size={9.5} tone={r === rps ? 1 : 2} bold={r === rps}>{`${r} req/s`}</Label>
                  <Label x={80} y={252 + i * 17} size={9.5} tone={2}>{x.busyGpus.toFixed(2)}</Label>
                  <Label x={180} y={252 + i * 17} size={9.5} tone={2}>{x.atTarget.toFixed(2)}</Label>
                  <Label x={276} y={252 + i * 17} size={9.5} kind={over ? 'error' : 'token'} bold>{`${x.gpus}`}</Label>
                  <Label x={340} y={252 + i * 17} size={9.5} tone={2}>{`$${fmt.int(monthlyCost(x.gpus, 'h100'))}`}</Label>
                  <Label x={430} y={252 + i * 17} size={9} tone={3}>
                    {r === 2 ? 'one card, mostly idle' : r === 6 ? 'comfortable' : r === 12 ? 'the design point' : 'four cards, autoscale territory'}
                  </Label>
                </g>
              );
            })}
          </g>
        </Fade>

        {/* ── what you actually run ── */}
        <Fade show={showRun}>
          <g>
            <rect x={0} y={300} width={640} height={60} fill="var(--ground-2)" opacity={0.95} />
            <Label x={16} y={322} size={10.5} tone={1} bold>Run: floor + headroom, with autoscaling closing the gap between them</Label>
            <Label x={16} y={340} size={9} tone={3}>
              {`A minimum of ${Math.max(1, Math.ceil(f.busyGpus / target))} to hold the floor, autoscaling on queue depth or p95 latency — never on GPU utilization.`}
            </Label>
            <Label x={16} y={354} size={9} tone={3}>
              Re-run this whenever the model, the engine, the context length or the traffic mix changes. Every one of them moves the machine time.
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}