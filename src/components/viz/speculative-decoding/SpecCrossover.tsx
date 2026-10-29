import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Chip, Fade, Dim, Bar } from '../lib/svg';
import { block, decodeStep, breakEvenAcceptance, MACHINES, type Machine } from './specModel';

// When does speculation pay? Four dials decide it: how predictable the text is
// (acceptance), how much a batch already amortises the weights (batching), how
// much of the step is launch overhead (compile), and how big the draft model is.

type Regimes = 'interactive' | 'throughput' | 'longform' | 'reasoning';

const STEPS = [
  'Speculation is a latency technique',
  'Batching already fixed the wall',
  'The step overhead changes everything',
  'Small models are the best case',
  'Predictable text is free money',
  'Where it loses outright',
  'The decision, end to end',
  'What to measure',
];

const CAPTIONS = [
  'Speculation attacks the per-token cost of a decode step, which is what a single user feels. That makes it a latency tool. If you are optimising tokens per second across many concurrent users, you are solving a different problem and may already have solved it.',
  'At large batch, the weights are read once for the whole batch, so the memory-bound wall that speculation exploits is much weaker. The remaining bottleneck is compute — and there, running gamma positions at once saves nothing at all.',
  'A decode step also carries fixed overhead: kernel launches, scheduling, Python. At small batch on a fast GPU that overhead can be a sizeable fraction of the step, and speculation replaces one step with gamma+1 steps, so it multiplies that overhead rather than hiding it.',
  'The trick relies on the verify pass being free relative to a decode step, and that is most true when the weights dominate. Small models have cheap matmuls relative to their weight read, so verification stays memory-bound and speculation keeps paying as gamma grows.',
  'Acceptance rate is entirely a property of the text. Code, extraction, templated output and agent loops repeat; open-ended prose does not. The same engine can be a big win on one request type and a loss on another.',
  'Some cases are outright losses: a large batch where compute dominates, text with very low acceptance, a draft model large enough that rho approaches the fraction of drafts kept, and short outputs where the block never gets the chance to pay off.',
  'Walk the four dials together. Speculation wins when acceptance is high, batch is small, per-step overhead is large, and the draft model is much cheaper than the target.',
  'Measure a and rho on your own traffic before enabling anything. Those two numbers, plus your batch size, determine the outcome; everything else is a detail of which method delivered them.',
];

const REGIMES: { id: Regimes; label: string; a: number; batch: number; note: string }[] = [
  { id: 'interactive', label: 'Interactive chat', a: 0.72, batch: 1, note: 'latency-bound: the best case for speculation' },
  { id: 'throughput', label: 'High-concurrency serving', a: 0.72, batch: 64, note: 'compute-bound: the weights are already amortised' },
  { id: 'longform', label: 'Summarise a long document', a: 0.55, batch: 1, note: 'repetitive, but less so than code' },
  { id: 'reasoning', label: 'Long-form reasoning', a: 0.4, batch: 1, note: 'every token is a new thought' },
];

export default function SpecCrossover() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [a, setA] = useState(0.72);
  const [batch, setBatch] = useState(1);
  const [mKey, setMKey] = useState<keyof typeof MACHINES>('h100');
  const [gamma, setGamma] = useState(4);
  const [overheadMs, setOverheadMs] = useState(1.5); // ms of fixed per-step cost

  const machine: Machine = MACHINES[mKey];
  const tDecode = decodeStep(machine);

  // Batch amortises the weight read: with B sequences the bytes per token fall by B.
  // So the "free verification" assumption weakens as B grows, and eventually the
  // block becomes compute-bound, which speculation cannot help with.
  const perTokenBytes = (n: number) => (machine.weights + machine.ctx * machine.kvBytesPerToken + n * machine.kvBytesPerToken) / batch;
  const passTime = (n: number) => {
    const mbu = 0.8, mfu = 0.55, oh = overheadMs / 1000;
    const memory = perTokenBytes(n) / (machine.bw * mbu);
    const compute = (2 * machine.params * n * batch) / (machine.flops * mfu);
    return Math.max(memory, compute) + oh;
  };
  const tStep = passTime(1);
  const tVer = passTime(gamma + 1);
  const r = block(a, 0.125, gamma, tVer, tStep);
  const be = breakEvenAcceptance(0.125, gamma, tVer, tStep);
  const sp = useTween(r.speedup, 450);

  const overheadShare = overheadMs / 1000 / tStep;
  const bound = passTime(1);

  const CX = 44;
  const BAR_W = 300;

  return (
    <Stage
      rootRef={ref}
      title="When speculation wins, and when it does not"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['token', 'speculation helps'],
        ['error', 'speculation hurts'],
        ['compute', 'cost'],
        ['memory', 'memory-bound work'],
        ['idle', 'overhead'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Acceptance a', value: `${(a * 100).toFixed(0)}%`, kind: 'token', note: `break-even ${(be * 100).toFixed(0)}% at ρ = 0.125, γ = ${gamma}` },
          { label: 'Batch', value: `${batch}`, kind: 'compute', note: bound > 1 ? 'step is compute-bound' : 'step is memory-bound' },
          { label: 'Overhead share', value: `${(overheadShare * 100).toFixed(0)}%`, kind: overheadShare > 0.2 ? 'error' : 'idle', note: `${overheadMs.toFixed(1)} ms of ${(tStep * 1e3).toFixed(2)} ms` },
          { label: 'Speedup', value: `${sp.toFixed(2)}x`, kind: sp >= 1 ? 'token' : 'error', note: sp >= 1.05 ? 'worth enabling' : sp >= 0.95 ? 'a wash' : 'turn it off' },
        ]} />
      }
      controls={
        <>
          <Slider label="Acceptance rate a" kind="token" min={0} max={0.95} step={0.01} value={a} onChange={setA} format={(v) => `${(v * 100).toFixed(0)}%`} />
          <Slider label="Batch size" kind="compute" min={1} max={128} log value={batch} onChange={setBatch} format={(v) => `${v} seq`} />
          <Slider label="Per-step overhead" kind="idle" min={0} max={6} step={0.1} value={overheadMs} onChange={setOverheadMs} format={(v) => `${v.toFixed(1)} ms`} />
          <Segmented label="GPU" value={mKey} options={[{ value: 't4', label: 'T4' }, { value: 'a100', label: 'A100' }, { value: 'h100', label: 'H100' }]} onChange={setMKey} />
        </>
      }
    >
      <svg viewBox="0 0 640 396" role="img" aria-label="A diagram showing when speculative decoding helps. A marker on a spectrum of workloads indicates the current speedup, and three panels show the acceptance rate against break-even, the batch size against whether the step is memory or compute bound, and the fixed per-step overhead share of the step.">
        {/* ── the verdict ──────────────────────────────────── */}
        <Label x={CX} y={24} tone={2} size={11} mono={false} bold>Right now, for this configuration</Label>
        <Fade show={s.step >= 0}>
          <Span x={CX} y={34} w={BAR_W} h={26} kind="idle" dim label="1.0x — no effect" labelTone="light" />
          <Span x={CX + BAR_W * 0.62} y={34} w={BAR_W * 0.38 * Math.min(1, Math.max(0, (sp - 1) / 1.5))} h={26} kind="token" />
          <Label x={CX + BAR_W + 12} y={52} size={13} kind={sp >= 1 ? 'token' : 'error'} bold>{`${sp.toFixed(2)}x`}</Label>
          <Label x={CX} y={76} tone={3} size={9.5}>
            {sp >= 1.15 ? 'clearly worth enabling' : sp >= 1 ? 'a real but modest win' : sp >= 0.95 ? 'a wash — not worth the complexity' : 'a loss — turn it off'}
          </Label>
        </Fade>

        {/* ── workload presets ─────────────────────────────── */}
        <Fade show={s.step >= 1}>
          <Label x={CX} y={100} tone={2} size={11} mono={false} bold>Four workloads, four answers</Label>
          {REGIMES.map((g, i) => {
            const x = CX + i * 148;
            const rr = block(g.a, 0.125, gamma, passTime(gamma + 1), passTime(1));
            const win = rr.speedup >= 1;
            return (
              <g key={g.id}>
                <rect x={x} y={110} width={138} height={54} rx={3} fill="var(--ground-2)" stroke="var(--rule-2)" />
                <Label x={x + 8} y={126} tone={1} size={9.5} bold>{g.label}</Label>
                <Label x={x + 8} y={140} tone={3} size={8.5}>{`a ≈ ${(g.a * 100).toFixed(0)}%, batch ${g.batch}`}</Label>
                <Label x={x + 8} y={156} size={10} kind={win ? 'token' : 'error'} bold>{`${rr.speedup.toFixed(2)}x`}</Label>
                <Label x={x + 130} y={156} anchor="end" size={8} tone={3}>{win ? 'win' : 'loss'}</Label>
              </g>
            );
          })}
          <Label x={CX} y={180} tone={3} size={9}>{`Same model, same GPU. Only a and batch changed — the outcome moved from ${block(REGIMES[0].a, 0.125, gamma, passTime(gamma + 1), passTime(1)).speedup.toFixed(1)}x to ${block(REGIMES[3].a, 0.125, gamma, passTime(gamma + 1), passTime(1)).speedup.toFixed(2)}x.`}</Label>
        </Fade>

        {/* ── panel: acceptance ───────────────────────────── */}
        <Fade show={s.step >= 4}>
          <rect x={CX} y={196} width={188} height={104} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={CX + 10} y={212} tone={2} size={10} mono={false} bold>1. Acceptance</Label>
          <Bar x={CX + 10} y={222} w={168} h={12} frac={Math.min(1, a / 0.95)} kind="token" value={`${(a * 100).toFixed(0)}%`} />
          <Dim x1={CX + 10 + (Math.min(be, 0.95) / 0.95) * 168} x2={CX + 10 + (Math.min(be, 0.95) / 0.95) * 168} y={250} label="" kind="error" />
          <line x1={CX + 10 + (Math.min(be, 0.95) / 0.95) * 168} y1={218} x2={CX + 10 + (Math.min(be, 0.95) / 0.95) * 168} y2={240} stroke="var(--error)" strokeWidth={1.5} strokeDasharray="3 3" />
          <Label x={CX + 10 + (Math.min(be, 0.95) / 0.95) * 168} y={254} anchor="middle" size={8} kind="error">{`break-even ${(be * 100).toFixed(0)}%`}</Label>
          <Label x={CX + 10} y={272} tone={3} size={8.5}>{a > be ? 'Above the line: drafts pay for themselves.' : 'Below the line: a net loss.'}</Label>
          <Label x={CX + 10} y={286} tone={3} size={8.5}>Repetitive text → high. Novel text → low.</Label>
        </Fade>

        {/* ── panel: batching ─────────────────────────────── */}
        <Fade show={s.step >= 2}>
          <rect x={CX + 202} y={196} width={188} height={104} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={CX + 212} y={212} tone={2} size={10} mono={false} bold>2. Batch size</Label>
          <Label x={CX + 212} y={228} tone={3} size={8.5}>weights read once for the batch:</Label>
          <Bar x={CX + 212} y={234} w={168} h={12} frac={Math.min(1, 1 / batch)} kind={batch === 1 ? 'memory' : 'compute'} value={`${batch}×`} />
          <Label x={CX + 212} y={264} size={9} kind={batch === 1 ? 'memory' : batch > 8 ? 'compute' : 'text'} bold>
            {batch === 1 ? 'memory-bound — the best case' : batch > 8 ? 'compute-bound — speculation saves nothing' : 'the wall is thinning'}
          </Label>
          <Label x={CX + 212} y={280} tone={3} size={8.5}>At 1 seq, verification really is</Label>
          <Label x={CX + 212} y={291} tone={3} size={8.5}>nearly free. At 64 seq it is not.</Label>
        </Fade>

        {/* ── panel: overhead ─────────────────────────────── */}
        <Fade show={s.step >= 3}>
          <rect x={CX + 404} y={196} width={188} height={104} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={CX + 414} y={212} tone={2} size={10} mono={false} bold>3. Step overhead</Label>
          <Bar x={CX + 414} y={222} w={168} h={12} frac={Math.min(1, overheadShare)} kind="idle" value={`${(overheadShare * 100).toFixed(0)}%`} />
          <Label x={CX + 414} y={252} tone={3} size={8.5}>{`${overheadMs.toFixed(1)} ms of a ${(tStep * 1e3).toFixed(2)} ms step`}</Label>
          <Label x={CX + 414} y={268} size={9} kind={overheadShare > 0.25 ? 'error' : 'text'} bold>
            {overheadShare > 0.25 ? 'speculation multiplies this' : 'small enough to hide'}
          </Label>
          <Label x={CX + 414} y={284} tone={3} size={8.5}>{`γ+1 = ${gamma + 1} passes, not 1`}</Label>
        </Fade>

        {/* ── verdict line ────────────────────────────────── */}
        <Fade show={s.step >= 6}>
          <Label x={CX} y={326} tone={2} size={11} mono={false} bold>The decision, in one line</Label>
          <rect x={CX} y={336} width={592} height={26} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <Label x={CX + 10} y={353} size={10} kind={sp >= 1 ? 'token' : 'error'} bold>
            {sp >= 1
              ? `enable: a = ${(a * 100).toFixed(0)}% keeps ${(r.keepRate * 100).toFixed(0)}% of drafts, beating ρ = 0.125; batch ${batch} keeps the step memory-bound, and overhead is only ${(overheadShare * 100).toFixed(0)}%`
              : `do not enable: at batch ${batch} the step is ${batch > 8 ? 'compute-bound' : 'nearly so'}, so the free-verification premise has failed`}
          </Label>
          <Label x={CX} y={378} tone={3} size={9}>Measure a on your traffic and ρ on your card. Those two decide it; batch size decides how much it is worth.</Label>
          <Label x={CX} y={391} size={9} tone={3}>Short outputs are the remaining caveat: the first block pays off, but there may be no second block to amortise it.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}