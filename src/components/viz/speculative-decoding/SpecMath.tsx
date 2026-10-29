import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Chip, Fade, Dim, Bar, fmt } from '../lib/svg';
import { block, decodeStep, verifyPass, LLAMA8B, MACHINES, type Machine } from './specModel';

// The derivation, one term at a time: how many tokens a block emits, what it
// costs, and therefore what the speedup is. Every number here comes from
// specModel.ts, the same functions the lessons quote.

const STEPS = [
  'The accepted run is geometric',
  'Tokens per block',
  'The draft cost is not free',
  'Put cost against tokens',
  'The speedup',
  'The break-even is a = rho',
  'gamma has an optimum',
  'The two numbers that matter',
];

const CAPTIONS = [
  'Drafting stops at the first token the target rejects, so the number of drafts you pay for is not gamma — it is however many were accepted plus one. If each draft is accepted with probability a, that count is geometric with mean (1 - a^gamma) / (1 - a).',
  'Out of the block come the accepted drafts plus one more: the bonus token from the final position, which the draft never predicted and the target therefore gets right for free. Multiply by the acceptance rate and you have the expected tokens per block.',
  'Drafting is not free, because it is sequential. Each drafted token costs rho of a target decode step, and you pay for every one you attempt — including the one that gets rejected and thrown away.',
  'So the cost of a block is one verification pass plus rho times the expected number of draft steps. Both terms are now written down, and both are measurable on your hardware.',
  'Speedup is tokens divided by cost, measured against the honest baseline: one plain decode step for one token. Move the sliders. This is the number an engine actually reports.',
  'Set the acceptance rate equal to the cost ratio and the speedup is exactly 1. Below it you are slower than not speculating at all. This single inequality is the whole decision rule, and gamma drops out of it entirely.',
  'Raise gamma and tokens per block climbs toward 1 / (1 - a), but the FLOPs of the verification pass grow too. Past some point the verification stops being free, and speedup turns over. There is an optimum, not a monotonic improvement.',
  'Everything reduces to two numbers: a, how often the draft is right, and rho, what a drafted token costs. Measure both on your model and your traffic, and the configuration follows.',
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

  // the curve of speedup vs gamma, drawn as a sparkline
  const N = 24;
  const curve = Array.from({ length: N }, (_, i) => {
    const g = i + 1;
    return block(a, rho, g, verifyPass(machine, g).total, tDecode).speedup;
  });
  const CX0 = 44, CY0 = 236, CW = 250, CH = 74;
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
          { label: 'Acceptance a', value: `${(a * 100).toFixed(0)}%`, kind: 'token', note: 'draft matches target' },
          { label: 'Draft cost rho', value: rho.toFixed(3), kind: 'error', note: 'per drafted token' },
          { label: 'Tokens per block', value: `${tk.toFixed(2)}`, kind: 'token', note: `ceiling ${ideal.toFixed(2)}` },
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
      <svg viewBox="0 0 640 396" role="img" aria-label="A derivation of speculative decoding speedup. Bars show the tokens delivered per block and the cost measured in decode steps. A formula is built up term by term, and a curve plots speedup against the number of drafted tokens, turning over at a large gamma.">
        {/* ── term 1: expected draft steps ─────────────────── */}
        <Label x={44} y={22} tone={2} size={11} mono={false} bold>1. How many drafts do you pay for?</Label>
        <Fade show={s.step >= 0}>
          <Label x={44} y={42} size={10.5}>Drafting stops at the first rejection, so the count is geometric:</Label>
          <text x={44} y={62} className="v-label math" style={{ fontSize: 13, fill: 'var(--text)' }}>E[drafts] = (1 − a^γ) / (1 − a)</text>
          {Array.from({ length: Math.min(gamma, 10) }, (_, i) => {
            const p = Math.pow(a, i); // probability draft i is reached at all
            return (
              <Span
                key={i}
                x={44 + i * 26}
                y={96 - 22 * p}
                w={22}
                h={22 * p}
                kind={i === 0 ? 'token' : 'token'}
                dim={i > 0}
              />
            );
          })}
          <Label x={44} y={124} tone={3} size={9.5}>{`expected ${r.draftSteps.toFixed(2)} of ${gamma} drafted — you never pay for the ones after the first rejection`}</Label>
        </Fade>

        {/* ── term 2: tokens ───────────────────────────────── */}
        <Label x={330} y={22} tone={2} size={11} mono={false} bold>2. What comes out?</Label>
        <Fade show={s.step >= 1}>
          <Label x={330} y={42} size={10.5}>Accepted drafts, plus one free bonus token:</Label>
          <text x={330} y={62} className="v-label math" style={{ fontSize: 13, fill: 'var(--text)' }}>E[tokens] = 1 + a · E[drafts]</text>
          <Chip x={330} y={74} text={`${tk.toFixed(2)}`} kind="token" w={54} size={11} />
          <Label x={392} y={90} size={9.5} kind="token">tokens per block</Label>
          <Bar x={330} y={102} w={180} h={12} frac={Math.min(1, tk / CEIL)} kind="token" label="" value={`ceiling ${ideal.toFixed(1)}`} />
          <Label x={330} y={128} tone={3} size={9.5}>{`rises with gamma toward 1/(1−a) = ${ideal.toFixed(2)}, then stops`}</Label>
        </Fade>

        {/* ── term 3: cost ─────────────────────────────────── */}
        <Fade show={s.step >= 2}>
          <Label x={44} y={152} tone={2} size={11} mono={false} bold>3. What does it cost?</Label>
          <text x={44} y={172} className="v-label math" style={{ fontSize: 12.5, fill: 'var(--text)' }}>cost = 1 verify pass + ρ · E[drafts]</text>
          <Span x={44} y={182} w={92} h={16} kind="memory" dim label="verify pass" labelTone="light" />
          <Label x={142} y={195} tone={3} size={9}>{`${(verifyPass(machine, gamma).total * 1e3).toFixed(2)} ms ≈ one decode step`}</Label>
          <Label x={330} y={182} tone={3} size={9.5}>{`drafting is sequential: ${r.draftSteps.toFixed(2)} steps × ρ=${rho.toFixed(3)}`}</Label>
          <Span x={330} y={190} w={Math.max(2, (r.cost / tDecode - 1) * 180)} h={12} kind="error" dim />
          <Label x={330} y={216} tone={3} size={9.5}>{`total ${ck.toFixed(2)} decode steps per block`}</Label>
        </Fade>

        {/* ── term 4: speedup ──────────────────────────────── */}
        <Fade show={s.step >= 3}>
          <Label x={44} y={228} tone={2} size={11} mono={false} bold>4. So the speedup is</Label>
          <text x={196} y={228} className="v-label math" style={{ fontSize: 12.5, fill: 'var(--text)' }}>E[tokens] / cost</text>
          <Label x={300} y={228} size={11} kind={sp >= 1 ? 'token' : 'error'} bold>{`= ${sp.toFixed(2)}x`}</Label>
          <Label x={440} y={228} tone={3} size={9}>{sp >= 1 ? 'faster than plain decode' : 'SLOWER — do not speculate here'}</Label>
        </Fade>

        {/* ── the gamma curve ──────────────────────────────── */}
        <Fade show={s.step >= 4}>
          <rect x={CX0 - 8} y={CY0 - 16} width={CW + 34} height={CH + 34} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={CX0} y={CY0 - 3} tone={2} size={10} mono={false} bold>Speedup vs gamma</Label>
          <line x1={CX0} y1={py(1)} x2={CX0 + CW} y2={py(1)} className="v-rule" strokeDasharray="3 3" />
          <Label x={CX0 + CW + 4} y={py(1) + 3} size={8.5} tone={3}>1.0</Label>
          <path d={line} fill="none" stroke={sp >= 1 ? 'var(--token)' : 'var(--error)'} strokeWidth={1.8} className="v-fade" />
          <circle cx={px(Math.min(gamma, N) - 1)} cy={py(curve[Math.min(gamma, N) - 1])} r={3.4} fill="var(--compute)" stroke="var(--compute)" />
          <Label x={CX0} y={CY0 + CH + 12} tone={3} size={9}>gamma = 1</Label>
          <Label x={CX0 + CW} y={CY0 + CH + 12} anchor="end" tone={3} size={9}>gamma = {N}</Label>
          <Label x={CX0 + CW / 2} y={CY0 + CH + 12} anchor="middle" size={9} kind="compute">{`you are at ${gamma}`}</Label>
        </Fade>

        {/* ── break-even ───────────────────────────────────── */}
        <Fade show={s.step >= 5}>
          <rect x={CX0 + CW + 34} y={CY0 - 16} width={228} height={CH + 34} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={CX0 + CW + 44} y={CY0 - 3} tone={2} size={10} mono={false} bold>The whole decision rule</Label>
          <text x={CX0 + CW + 44} y={CY0 + 16} className="v-label math" style={{ fontSize: 12, fill: 'var(--text)' }}>a &gt; ρ  ⇒  you win</text>
          <Label x={CX0 + CW + 44} y={CY0 + 32} tone={3} size={9}>{`you: a = ${(a * 100).toFixed(0)}%, ρ = ${rho.toFixed(3)}`}</Label>
          <Arrow x1={CX0 + CW + 44} y1={CY0 + 40} x2={CX0 + CW + 44 + Math.min(180, a * 180)} y2={CY0 + 40}
            kind={a > rho ? 'token' : 'error'} width={2.4} />
          <Dim x1={CX0 + CW + 44} x2={CX0 + CW + 44 + rho * 180} y={CY0 + 54} label={`ρ = ${rho.toFixed(2)}`} kind="error" />
          <Label x={CX0 + CW + 44 + Math.min(180, a * 180)} y={CY0 + 40} size={9} kind="token" anchor="middle">a</Label>
          <Label x={CX0 + CW + 44} y={CY0 + 72} size={9.5} kind={a > rho ? 'token' : 'error'} bold>
            {a > rho ? `margin ${((a / rho - 1) * 100).toFixed(0)}% above break-even` : `short by ${((1 - a / rho) * 100).toFixed(0)}%`}
          </Label>
          <Label x={CX0 + CW + 44} y={CY0 + 88} tone={3} size={9}>Note gamma does not appear here.</Label>
        </Fade>

        {/* ── the optimum ──────────────────────────────────── */}
        <Fade show={s.step >= 6}>
          <Label x={44} y={344} tone={2} size={11} mono={false} bold>Why gamma turns over</Label>
          <Label x={44} y={362} size={9.5} tone={3}>More drafts → more tokens per block, so far so good.</Label>
          <Label x={44} y={375} size={9.5} tone={3}>But the verify pass does γ× the FLOPs of a decode step. Past the knee,</Label>
          <Label x={330} y={362} size={9.5} kind="compute">compute catches up with memory and the pass stops being free.</Label>
          <Label x={330} y={375} size={9.5} tone={3}>The wasted draft, at a = {(a * 100).toFixed(0)}%: {r.wasted.toFixed(2)} of {gamma}.</Label>
        </Fade>

        <Fade show={s.step >= 7}>
          <Label x={44} y={390} size={10} kind="token" bold>{`a = {(a * 100).toFixed(0)}% · ρ = ${rho.toFixed(3)} · γ = ${gamma} → ${sp.toFixed(2)}x`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}