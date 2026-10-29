import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Chip, Fade, Bar } from '../lib/svg';
import { block, decodeStep, verifyPass, breakEvenAcceptance, MACHINES, type Machine } from './specModel';

// The derivation, one term at a time: how many tokens a block emits, what it
// costs, and therefore what the speedup is. Every number here comes from
// specModel.ts, the same functions the lessons quote.

const STEPS = [
  'Each draft is a bet',
  'Tokens per block',
  'Every draft is paid for',
  'Put cost against tokens',
  'The speedup',
  'Break-even: keep rate = rho',
  'gamma has an optimum',
  'Two numbers, then gamma',
];

const CAPTIONS = [
  'The draft model proposes all gamma tokens, then the target checks them in one pass and keeps the prefix up to the first rejection. Draft i survives only if it and every draft before it are accepted: probability a^i. The bars fall geometrically, and the expected number kept is a + a² + … + a^gamma.',
  'Out of the block come the kept drafts plus one more token from the target itself: the corrected token at the first rejection, or a bonus token if every draft was kept. So a block always emits at least one token, and on average (1 − a^(gamma+1)) / (1 − a).',
  'Drafting is sequential and happens before verification, so all gamma drafts are paid for, rho of a target step each, whether they are kept or thrown away. The drafts after the first rejection are pure waste, and there are more of them the longer the block.',
  'So a block costs one verification pass plus gamma times rho. The first term is nearly one decode step while decode is memory-bound; the second grows linearly with gamma, while the tokens it buys grow less and less.',
  'Speedup is tokens divided by cost, against the honest baseline of one plain decode step per token. Move the sliders: acceptance pushes it up, draft cost pulls it down.',
  'Speculation pays exactly when the fraction of drafted tokens you keep beats rho: each draft costs rho of a step and each kept one saves a whole step. With one draft per block the keep rate is a, so the rule is a > rho. With longer blocks the keep rate falls, so the break-even acceptance rises with gamma.',
  'Raise gamma and tokens per block flattens toward 1 / (1 − a) while drafting cost keeps rising by rho per draft, so the speedup peaks and turns over. Add draft gamma+1 only while its chance of being kept, a^(gamma+1), beats rho times the current speedup.',
  'Measure a on your traffic and rho on your hardware. The keep rate tells you whether to speculate at all; the optimum gamma (usually 3 to 6) tells you how far.',
];

export default function SpecMath() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [a, setA] = useState(0.72);
  const [rho, setRho] = useState(0.125);
  const [gamma, setGamma] = useState(4);
  const [mKey, setMKey] = useState<keyof typeof MACHINES>('h100');

  const machine: Machine = MACHINES[mKey];
  const tDecode = decodeStep(machine);
  const r = block(a, rho, gamma, verifyPass(machine, gamma).total, tDecode);

  const tk = useTween(r.tokens, 500);
  const ck = useTween(r.cost / tDecode, 500);
  const sp = useTween(r.speedup, 500);

  const ideal = 1 / (1 - a);
  const CEIL = 12; // tokens per block, for the bar scale
  const maxSp = 3.5;
  const be = breakEvenAcceptance(rho, gamma, verifyPass(machine, gamma).total, tDecode);
  const keep = r.keepRate;

  // the curve of speedup vs gamma, drawn as a sparkline
  const N = 24;
  const curve = Array.from({ length: N }, (_, i) => {
    const g = i + 1;
    return block(a, rho, g, verifyPass(machine, g).total, tDecode).speedup;
  });
  const bestIdx = curve.indexOf(Math.max(...curve));
  const CX0 = 44, CY0 = 252, CW = 250, CH = 66;
  const px = (i: number) => CX0 + (i / (N - 1)) * CW;
  const py = (v: number) => CY0 + CH - Math.min(v / maxSp, 1) * CH;
  const line = curve.map((v, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ');

  return (
    <Stage
      rootRef={ref}
      title="The speedup, derived"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['token', 'tokens delivered'],
        ['compute', 'cost, in decode steps'],
        ['error', 'rejected drafts'],
        ['memory', 'the verify pass'],
        ['text', 'nothing'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Acceptance a', value: `${(a * 100).toFixed(0)}%`, kind: 'token', note: 'per drafted token' },
          { label: 'Keep rate', value: `${(keep * 100).toFixed(0)}%`, kind: 'token', note: `${r.accepted.toFixed(2)} of ${gamma} drafts kept` },
          { label: 'Draft cost rho', value: rho.toFixed(3), kind: 'error', note: 'per drafted token' },
          { label: 'Speedup', value: `${sp.toFixed(2)}x`, kind: sp >= 1 ? 'token' : 'error', note: `vs one decode step per token` },
        ]} />
      }
      controls={
        <>
          <Slider label="Acceptance rate a" kind="token" min={0} max={0.95} step={0.01} value={a} onChange={setA} format={(v) => `${(v * 100).toFixed(0)}%`} />
          <Slider label="Draft cost rho" kind="error" min={0} max={0.6} step={0.005} value={rho} onChange={setRho} format={(v) => v.toFixed(3)} />
          <Slider label="Drafts per block gamma" kind="compute" min={1} max={16} value={gamma} onChange={setGamma} />
          <Segmented label="GPU" value={mKey} options={[{ value: 't4', label: 'T4' }, { value: 'a100', label: 'A100' }, { value: 'h100', label: 'H100' }]} onChange={setMKey} />
        </>
      }
    >
      <svg viewBox="0 0 640 396" role="img" aria-label="A derivation of speculative decoding speedup. Bars show the falling chance that each drafted token is kept, the tokens delivered per block, and the cost of verification plus every drafted token. A curve plots speedup against the number of drafts per block, peaking and then turning over.">
        {/* ── term 1: kept drafts ─────────────────────────── */}
        <Label x={44} y={22} tone={2} size={11} mono={false} bold>1. How many drafts survive?</Label>
        <Fade show={s.step >= 0}>
          <Label x={44} y={42} size={10.5}>Draft i survives with chance aⁱ:</Label>
          <text x={44} y={62} className="v-label math" style={{ fontSize: 13, fill: 'var(--text)' }}>E[kept] = a + a² + … + a^γ</text>
          {Array.from({ length: Math.min(gamma, 10) }, (_, i) => {
            const p = Math.pow(a, i + 1); // probability draft i+1 survives
            return (
              <g key={i}>
                <rect x={44 + i * 26} y={74} width={22} height={22} rx={2} fill="none" stroke="var(--error)" strokeDasharray="2 2" opacity={0.6} />
                <Span x={44 + i * 26} y={96 - 22 * p} w={22} h={22 * p} kind="token" dim={i > 0} />
              </g>
            );
          })}
          <Label x={44} y={124} tone={3} size={9.5}>{`expected ${r.accepted.toFixed(2)} of ${gamma} kept · all ${gamma} paid for`}</Label>
        </Fade>

        {/* ── term 2: tokens ───────────────────────────────── */}
        <Label x={330} y={22} tone={2} size={11} mono={false} bold>2. What comes out?</Label>
        <Fade show={s.step >= 1}>
          <Label x={330} y={42} size={10.5}>Kept drafts, plus one token from the target:</Label>
          <text x={330} y={62} className="v-label math" style={{ fontSize: 13, fill: 'var(--text)' }}>E[tokens] = 1 + E[kept]</text>
          <Chip x={330} y={74} text={`${tk.toFixed(2)}`} kind="token" w={54} size={11} />
          <Label x={392} y={90} size={9.5} kind="token">tokens per block</Label>
          <Bar x={330} y={102} w={180} h={12} frac={Math.min(1, tk / CEIL)} kind="token" label="" value={`ceiling ${ideal.toFixed(1)}`} />
          <Label x={330} y={128} tone={3} size={9.5}>{`= (1 − a^(γ+1)) / (1 − a), ceiling 1/(1−a)`}</Label>
        </Fade>

        {/* ── term 3: cost ─────────────────────────────────── */}
        <Fade show={s.step >= 2}>
          <Label x={44} y={152} tone={2} size={11} mono={false} bold>3. What does it cost?</Label>
          <text x={44} y={172} className="v-label math" style={{ fontSize: 12.5, fill: 'var(--text)' }}>cost = 1 verify pass + γ · ρ</text>
          <Span x={44} y={182} w={92} h={16} kind="memory" dim label="verify pass" labelTone="light" />
          <Label x={142} y={195} tone={3} size={9}>{`${(verifyPass(machine, gamma).total * 1e3).toFixed(2)} ms ≈ one decode step`}</Label>
          <Label x={330} y={182} tone={3} size={9.5}>{`all ${gamma} drafts paid: ${gamma} × ρ = ${(gamma * rho).toFixed(3)} steps`}</Label>
          <Span x={330} y={190} w={Math.max(2, Math.min(280, gamma * rho * 180))} h={12} kind="error" dim />
          <Label x={330} y={216} tone={3} size={9.5}>{`total ${ck.toFixed(2)} decode steps · ${r.wasted.toFixed(2)} drafts wasted`}</Label>
        </Fade>

        {/* ── term 4: speedup ──────────────────────────────── */}
        <Fade show={s.step >= 3}>
          <Label x={44} y={236} tone={2} size={11} mono={false} bold>4. So the speedup is</Label>
          <text x={196} y={236} className="v-label math" style={{ fontSize: 12.5, fill: 'var(--text)' }}>E[tokens] / cost</text>
          <Label x={300} y={236} size={11} kind={sp >= 1 ? 'token' : 'error'} bold>{`= ${sp.toFixed(2)}x`}</Label>
          <Label x={440} y={236} tone={3} size={9}>{sp >= 1 ? 'faster than plain decode' : 'SLOWER — do not speculate here'}</Label>
        </Fade>

        {/* ── the gamma curve ──────────────────────────────── */}
        <Fade show={s.step >= 4}>
          <rect x={CX0 - 8} y={CY0 - 10} width={CW + 34} height={CH + 30} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={CX0} y={CY0 + 4} tone={2} size={10} mono={false} bold>Speedup vs gamma</Label>
          <line x1={CX0} y1={py(1)} x2={CX0 + CW} y2={py(1)} className="v-rule" strokeDasharray="3 3" />
          <Label x={CX0 + CW + 4} y={py(1) + 3} size={8.5} tone={3}>1.0</Label>
          <path d={line} fill="none" stroke={sp >= 1 ? 'var(--token)' : 'var(--error)'} strokeWidth={1.8} className="v-fade" />
          <circle cx={px(Math.min(gamma, N) - 1)} cy={py(curve[Math.min(gamma, N) - 1])} r={3.4} fill="var(--compute)" stroke="var(--compute)" />
          <Fade show={s.step >= 6}>
            <circle cx={px(bestIdx)} cy={py(curve[bestIdx])} r={5} fill="none" stroke="var(--text)" strokeWidth={1.2} />
            <Label x={px(bestIdx) + 8} y={py(curve[bestIdx]) - 6} size={8.5} tone={1}>{`best γ = ${bestIdx + 1}`}</Label>
          </Fade>
          <Label x={CX0} y={CY0 + CH + 14} tone={3} size={9}>gamma = 1</Label>
          <Label x={CX0 + CW} y={CY0 + CH + 14} anchor="end" tone={3} size={9}>gamma = {N}</Label>
          <Label x={CX0 + CW / 2} y={CY0 + CH + 14} anchor="middle" size={9} kind="compute">{`you are at ${gamma}`}</Label>
        </Fade>

        {/* ── break-even ───────────────────────────────────── */}
        <Fade show={s.step >= 5}>
          <rect x={CX0 + CW + 34} y={CY0 - 10} width={228} height={CH + 30} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={CX0 + CW + 44} y={CY0 + 4} tone={2} size={10} mono={false} bold>The decision rule</Label>
          <text x={CX0 + CW + 44} y={CY0 + 22} className="v-label math" style={{ fontSize: 12, fill: 'var(--text)' }}>keep rate &gt; ρ  ⇒  you win</text>
          <Arrow x1={CX0 + CW + 44} y1={CY0 + 34} x2={CX0 + CW + 44 + Math.max(2, Math.min(180, keep * 180))} y2={CY0 + 34}
            kind={keep > rho ? 'token' : 'error'} width={2.4} />
          <line x1={CX0 + CW + 44 + Math.min(180, rho * 180)} x2={CX0 + CW + 44 + Math.min(180, rho * 180)} y1={CY0 + 27} y2={CY0 + 42} stroke="var(--error)" strokeWidth={2} />
          <Label x={CX0 + CW + 44 + Math.min(180, rho * 180) + 5} y={CY0 + 47} size={9} kind="error">ρ</Label>
          <Label x={CX0 + CW + 44} y={CY0 + 66} size={9.5} kind={keep > rho ? 'token' : 'error'} bold>
            {`keep rate ${(keep * 100).toFixed(0)}% ${keep > rho ? '>' : '<'} ρ = ${(rho * 100).toFixed(1)}%`}
          </Label>
          <Label x={CX0 + CW + 44} y={CY0 + 80} tone={3} size={9}>{`break-even a at γ = ${gamma}: ${(be * 100).toFixed(1)}% (γ = 1: a > ρ)`}</Label>
        </Fade>

        {/* ── the optimum ──────────────────────────────────── */}
        <Fade show={s.step >= 6}>
          <Label x={44} y={358} tone={2} size={11} mono={false} bold>Why gamma turns over</Label>
          <Label x={44} y={374} size={9.5} tone={3}>Each extra draft costs ρ for sure and is kept with chance a^(γ+1).</Label>
          <Label x={44} y={387} size={9.5} tone={3}>{`At γ = ${gamma}: the next one is kept ${(Math.pow(a, gamma + 1) * 100).toFixed(0)}% of the time, against ρ × speedup = ${(rho * sp * 100).toFixed(0)}%.`}</Label>
        </Fade>
        <Fade show={s.step >= 7}>
          <Label x={620} y={358} anchor="end" size={10} kind={sp >= 1 ? 'token' : 'error'} bold>{`a ${(a * 100).toFixed(0)}% · ρ ${rho.toFixed(3)} · γ ${gamma} → ${sp.toFixed(2)}x (best γ ${bestIdx + 1}: ${curve[bestIdx].toFixed(2)}x)`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}