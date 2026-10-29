import { useRef, useMemo, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Chip, Fade, Dim, Bar, fmt } from '../lib/svg';
import { block, decodeStep, verifyPass, breakEvenAcceptance, bestGamma, METHODS, MACHINES, type Machine } from './specModel';

// The lab's simulator. Everything here is a prediction from first principles —
// the notebook measures the same quantities for real, so you can check the model
// against hardware. That is the point of the exercise.

type Preset = 'chat' | 'code' | 'summarise' | 'creative' | 'batch64';

const PRESETS: Record<Preset, { label: string; a: number; batch: number; note: string }> = {
  chat: { label: 'Interactive chat', a: 0.72, batch: 1, note: 'predictable phrasing, low batch' },
  code: { label: 'Code completion', a: 0.8, batch: 1, note: 'heavy repetition and syntax' },
  summarise: { label: 'Summarise a document', a: 0.62, batch: 1, note: 'draws on the prompt' },
  creative: { label: 'Creative writing', a: 0.42, batch: 1, note: 'every token is a new choice' },
  batch64: { label: 'Serving at batch 64', a: 0.72, batch: 64, note: 'compute-bound: the wall is gone' },
};

const STEPS = [
  'Pick a workload',
  'The acceptance rate is the whole game',
  'The break-even line',
  'Draft cost decides the method',
  'gamma: find the flat part',
  'Batch kills the benefit',
  'Read your own verdict',
  'Now go measure it',
];

const CAPTIONS = [
  'Five workloads, same model and card. Only two things change: how predictable the text is, and how many sequences are in the batch. Watch what that does to the verdict.',
  'Acceptance is the single strongest lever. Code accepts around 0.8 because syntax repeats; creative writing sits near 0.4 because it does not. The speedup moves several-fold across that range.',
  'The break-even line is a = rho, and for a 1B drafting an 8B that is 12.5%. Nothing below it is worth doing, whatever else is going on.',
  'The method only changes rho, and therefore where your workload sits relative to that line. Prompt lookup sits almost on top of it; a big draft model sits well to the right.',
  'Speedup rises with gamma and then flattens as the token count approaches 1/(1-a). Past the flat part, more drafts buy nothing — and cost memory and scheduling.',
  'Raise the batch and the verdict inverts. The weights are read once for everyone, the step is compute-bound, and the free-verification premise no longer holds.',
  'Your verdict line combines all of it: acceptance, method, batch and block size. Before you trust it, measure the two numbers it rests on.',
  'The notebook measures acceptance rate, draft cost and the verify-versus-decode time on a real T4. Compare its numbers with these predictions and you will know exactly how much to trust a model like this one.',
];

export default function SpecLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [preset, setPreset] = useState<Preset>('chat');
  const [a, setA] = useState(PRESETS.chat.a);
  const [batch, setBatch] = useState(PRESETS.chat.batch);
  const [method, setMethod] = useState('draft');
  const [gamma, setGamma] = useState(4);
  const [mKey, setMKey] = useState<keyof typeof MACHINES>('h100');

  const m = METHODS.find((x) => x.id === method)!;
  const machine: Machine = MACHINES[mKey];
  const tDecodeBase = decodeStep(machine);

  // batching amortises the weight read across the batch
  const pass = (n: number) => {
    const oh = 0.0015;
    const bytes = (machine.weights + machine.ctx * machine.kvBytesPerToken + n * machine.kvBytesPerToken) / batch;
    const memory = bytes / (machine.bw * 0.8);
    const compute = (2 * machine.params * n * batch) / (machine.flops * 0.55);
    return { total: Math.max(memory, compute) + oh, bound: compute > memory ? ('compute' as const) : ('memory' as const) };
  };
  const tStep = pass(1);
  const r = block(a, m.rho, gamma, pass(gamma + 1).total, tStep);
  const sp = useTween(r.speedup, 450);

  const be = breakEvenAcceptance(m.rho, gamma, pass(gamma + 1).total, tStep);
  const bg = useMemo(() => bestGamma(a, m.rho, machine, 24), [a, m.rho, machine]);

  // speedup across every method at the current a, for the comparison strip
  const rows = useMemo(
    () => METHODS.map((x) => ({ ...x, sp: block(a, x.rho, gamma, pass(gamma + 1).total, tStep).speedup })),
    [a, gamma, machine, batch],
  );
  const maxSp = Math.max(1.6, ...rows.map((x) => x.sp));

  const apply = (p: Preset) => {
    setPreset(p);
    setA(PRESETS[p].a);
    setBatch(PRESETS[p].batch);
  };

  const CX = 44;

  return (
    <Stage
      rootRef={ref}
      title="Map where speculation wins"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['token', 'faster than plain decode'],
        ['error', 'slower — leave it off'],
        ['memory', 'break-even, a = rho'],
        ['compute', 'draft cost'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Workload', value: PRESETS[preset].label, kind: 'text', note: PRESETS[preset].note },
          { label: 'Acceptance a', value: `${(a * 100).toFixed(0)}%`, kind: 'token', note: `break-even ${(be * 100).toFixed(1)}%` },
          { label: 'Tokens per block', value: r.tokens.toFixed(2), kind: 'token', note: `ceiling ${(1 / (1 - a)).toFixed(2)}` },
          { label: `Speedup at γ=${gamma}`, value: `${sp.toFixed(2)}x`, kind: sp >= 1 ? 'token' : 'error', note: `best γ ≈ ${bg.gamma} gives ${bg.speedup.toFixed(2)}x` },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Workload"
            value={preset}
            options={(Object.keys(PRESETS) as Preset[]).map((k) => ({ value: k, label: PRESETS[k].label.split(' ')[0] }))}
            onChange={apply}
          />
          <Segmented
            label="Method"
            value={method}
            options={METHODS.map((x) => ({ value: x.id, label: x.label.split(' ')[0] }))}
            onChange={setMethod}
          />
          <Slider label="Acceptance rate a" kind="token" min={0} max={0.95} step={0.01} value={a} onChange={setA} format={(v) => `${(v * 100).toFixed(0)}%`} />
          <Slider label="Batch size" kind="compute" min={1} max={128} log value={batch} onChange={setBatch} format={(v) => `${v} seq`} />
          <Slider label="Drafts per block γ" kind="memory" min={1} max={12} value={gamma} onChange={setGamma} />
          <Segmented label="GPU" value={mKey} options={[{ value: 't4', label: 'T4' }, { value: 'a100', label: 'A100' }, { value: 'h100', label: 'H100' }]} onChange={setMKey} />
        </>
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A simulator for speculative decoding. It shows the predicted speedup for five workloads, plots the current acceptance rate against the break-even line, compares every drafting method on a bar chart, and reports the best block size.">
        {/* ── the verdict ──────────────────────────────────── */}
        <Label x={CX} y={22} tone={2} size={11} mono={false} bold>Your verdict</Label>
        <Fade show={s.step >= 0}>
          <rect x={CX} y={32} width={592} height={30} rx={3} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Span x={CX + 8} y={40} w={300} h={14} kind="idle" dim label="1.0x — plain decode" labelTone="light" />
          <Span
            x={CX + 8 + 300 * 0.62}
            y={40}
            w={Math.max(2, Math.min(1, Math.abs(sp - 1) / 1.6)) * 300 * 0.38}
            h={14}
            kind={sp >= 1 ? 'token' : 'error'}
          />
          <Label x={CX + 316} y={51} size={11} kind={sp >= 1 ? 'token' : 'error'} bold>{`${sp.toFixed(2)}x`}</Label>
          <Label x={CX + 372} y={51} tone={2} size={9.5}>
            {sp >= 1.15 ? 'enable it' : sp >= 1 ? 'marginal' : sp >= 0.95 ? 'not worth it' : 'a loss — disable'}
          </Label>
        </Fade>

        {/* ── acceptance vs break-even ─────────────────────── */}
        <Fade show={s.step >= 1}>
          <Label x={CX} y={86} tone={2} size={11} mono={false} bold>Acceptance against the break-even line</Label>
          <Bar x={CX} y={96} w={300} h={16} frac={Math.min(1, a / 0.95)} kind={a >= be ? 'token' : 'error'} label="" />
          <line x1={CX + (be / 0.95) * 300} y1={92} x2={CX + (be / 0.95) * 300} y2={118} stroke="var(--memory)" strokeWidth={2} />
          <Label x={CX + (be / 0.95) * 300} y={130} anchor="middle" size={8.5} kind="memory">{`break-even ${(be * 100).toFixed(1)}%`}</Label>
          <Label x={CX + 300} y={130} anchor="end" tone={3} size={8.5}>100%</Label>
          <Label x={CX + 316} y={106} size={9.5} kind="token" bold>{`a = ${(a * 100).toFixed(0)}%`}</Label>
          <Label x={CX + 316} y={120} tone={3} size={8.5}>{a >= be ? 'above the line: profitable' : 'below the line: a net loss'}</Label>
          <Label x={CX + 316} y={132} tone={3} size={8.5}>{`${PRESETS[preset].note}`}</Label>
        </Fade>

        {/* ── every method, at this workload ──────────────── */}
        <Label x={CX} y={156} tone={2} size={11} mono={false} bold>Every method at a = {(a * 100).toFixed(0)}%</Label>
        {rows.map((x, i) => {
          const y = 168 + i * 22;
          const on = x.id === method;
          const frac = Math.min(1, x.sp / maxSp);
          return (
            <g key={x.id}>
              <Label x={CX} y={y + 10} size={9.5} tone={on ? 1 : 3} bold={on}>{x.label}</Label>
              <rect x={CX + 108} y={y} width={150} height={14} rx={2} fill="none" stroke="var(--rule)" />
              <rect x={CX + 108} y={y} width={Math.max(2, frac * 150)} height={14} rx={2}
                fill={x.sp >= 1 ? (on ? 'var(--token)' : 'var(--token-dim)') : (on ? 'var(--error)' : 'var(--error-dim)')}
                stroke={x.sp >= 1 ? 'var(--token)' : 'var(--error)'} className="v-cell" />
              <line x1={CX + 108 + (1 / maxSp) * 150} y1={y - 2} x2={CX + 108 + (1 / maxSp) * 150} y2={y + 16} stroke="var(--memory)" strokeWidth={1.5} strokeDasharray="2 2" />
              <Label x={CX + 266} y={y + 11} size={9.5} kind={on ? (x.sp >= 1 ? 'token' : 'error') : 'text'}>{`${x.sp.toFixed(2)}x`}</Label>
            </g>
          );
        })}
        <Label x={CX + 108} y={168 + rows.length * 22 + 6} tone={3} size={8.5}>dashed line = 1.0x, the point of no return</Label>

        {/* ── gamma curve + best gamma ────────────────────── */}
        <Fade show={s.step >= 4}>
          <rect x={CX + 300} y={164} width={292} height={118} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={CX + 312} y={182} tone={2} size={10} mono={false} bold>Block size</Label>
          {Array.from({ length: 12 }, (_, i) => {
            const g = i + 1;
            const rr = block(a, m.rho, g, pass(g + 1).total, tStep);
            const h = Math.max(2, (rr.speedup / Math.max(maxSp, 1.6)) * 64);
            return (
              <g key={g}>
                <rect x={CX + 314 + i * 22} y={214 - h} width={16} height={h} rx={1.5}
                  fill={g === gamma ? 'var(--compute)' : 'var(--text-3)'} fillOpacity={g === gamma ? 1 : 0.55}
                  stroke={g === gamma ? 'var(--compute)' : 'var(--rule-2)'} strokeWidth={0.8} className="v-cell" />
                <Label x={CX + 322 + i * 22} y={226} anchor="middle" size={7.5} tone={3}>{g}</Label>
              </g>
            );
          })}
          <Label x={CX + 312} y={200} tone={3} size={8.5}>{`tokens per block ${r.tokens.toFixed(2)}`}</Label>
          <Label x={CX + 312} y={240} size={9} kind="compute" bold>{`best γ ≈ ${bg.gamma} → ${bg.speedup.toFixed(2)}x`}</Label>
          <Label x={CX + 312} y={254} tone={3} size={8.5}>{`ceiling 1/(1−a) = ${(1 / (1 - a)).toFixed(2)} tokens`}</Label>
          <Label x={CX + 312} y={266} tone={3} size={8.5}>{`drafts paid: ${r.draftSteps.toFixed(2)}, wasted ${r.wasted.toFixed(2)}`}</Label>
        </Fade>

        {/* ── the machine ──────────────────────────────────── */}
        <Fade show={s.step >= 5}>
          <Label x={CX} y={300} tone={2} size={11} mono={false} bold>This machine</Label>
          <Dim x1={CX} x2={CX + 592} y={312} label={`Llama-3.1-8B · batch ${batch} · step ${(tStep * 1e3).toFixed(2)} ms`} kind="text" />
          <Label x={CX} y={332} tone={3} size={9}>{`weights ${fmt.gb(machine.weights)} · KV ${fmt.gb(machine.ctx * machine.kvBytesPerToken)} · step is ${pass(1).bound}-bound`}</Label>
          <Label x={CX} y={346} size={9} kind={pass(1).bound === 'memory' ? 'token' : 'error'}>
            {pass(1).bound === 'memory'
              ? 'Memory-bound: verification really is nearly free. Speculation applies.'
              : 'Compute-bound: batching already amortised the weights. Speculation saves nothing.'}
          </Label>
        </Fade>

        <Fade show={s.step >= 7}>
          <Label x={CX} y={372} tone={2} size={10} bold>These are predictions from bandwidth, FLOP/s and your two measured numbers.</Label>
          <Label x={CX} y={388} tone={3} size={9}>The notebook measures the same three quantities on a real T4 — check them against this.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}