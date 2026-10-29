import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Chip, Fade, Dim, Bar, fmt } from '../lib/svg';
import { block, decodeStep, verifyPass, LLAMA8B, METHODS, type Machine } from './specModel';

// Every drafting method is a point on one plane: acceptance rate a on one axis,
// draft cost rho on the other. The diagonal a = rho is the break-even line.
// Everything about choosing a method follows from where a method sits.

const STEPS = [
  'Everything is a point on a plane',
  'N-gram and prompt lookup',
  'The small draft model',
  'Medusa: heads on the target',
  'EAGLE: predict the hidden state',
  'MTP: decode a block in parallel',
  'The frontier is the frontier',
  'Choosing a method',
];

const CAPTIONS = [
  'Strip away the implementations and every drafting method answers the same two questions: how often is it right, and what does one drafted token cost? Plot those and the whole design space collapses to a plane with one meaningful diagonal on it.',
  'The cheapest methods use no network at all. An n-gram table memorises what followed similar contexts before; prompt lookup finds a matching span in the prompt or earlier output and copies from it. Both are nearly free, which means they win on almost anything — and lose the moment the text stops repeating.',
  'A small model drafting a large one is the general case. It is always right often enough because it is genuinely the same model, just smaller. Its cost is a real forward pass, so it sits at a fixed fraction of the target weight read. This is the workhorse configuration.',
  'Medusa keeps the target model and adds a few small linear heads that jump straight from the final hidden state to the next few tokens, bypassing most of the network. It is cheap and has no second model, but it must be trained, and it sees less context than a full forward pass.',
  'EAGLE changes what is predicted. Instead of predicting the next token, it predicts the next hidden state — the input to the next layer — and reuses the target model as the verifier for a second, parallel step. That extra context is why its acceptance rate is the highest of the cheap methods.',
  'MTP trains the model to predict several future blocks at once, so drafting becomes a single forward pass over a widened head rather than a sequence of steps. The per-token cost drops sharply, at the price of a training objective the base model did not have.',
  'The methods trade acceptance against cost along a real frontier. Nothing here is free: a method that is more accurate is paying for it with compute, and a method that is nearly free is nearly right only when the text is predictable.',
  'Choose by where your traffic sits, not by which method sounds best. Highly repetitive text favours the free methods; open-ended generation needs the accurate ones, and only has the budget for them when the target model is large enough to pay for it.',
];

export default function SpecMethods() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [sel, setSel] = useState('draft');
  const [gamma, setGamma] = useState(4);
  const [predict, setPredict] = useState(0.5); // how predictable is your text? sets the method's a

  const m = METHODS.find((x) => x.id === sel)!;
  const machine: Machine = LLAMA8B;
  const tDecode = decodeStep(machine);

  // the method's acceptance is its published value scaled by how predictable the text is
  const a = Math.min(0.97, m.a * (0.55 + predict * 0.85));
  const r = block(a, m.rho, gamma, verifyPass(machine, gamma).total, tDecode);
  const sp = useTween(r.speedup, 450);

  // the plane: a on x, rho on y
  const PX = 96, PY = 56, PW = 300, PH = 190;
  const AMIN = 0, AMAX = 0.95;
  const RMIN = 0, RMAX = 0.32;
  const px = (av: number) => PX + (av / AMAX) * PW;
  const py = (rv: number) => PY + PH - (rv / RMAX) * PH;

  return (
    <Stage
      rootRef={ref}
      title="Every drafting method is a point on one plane"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['token', 'methods that win (a > rho)'],
        ['error', 'methods that lose'],
        ['text', 'the chosen method'],
        ['memory', 'break-even line a = rho'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Method', value: m.label, kind: 'text', note: m.blurb },
          { label: 'Acceptance a', value: `${(a * 100).toFixed(0)}%`, kind: 'token', note: `base ${(m.a * 100).toFixed(0)}%` },
          { label: 'Cost rho', value: m.rho.toFixed(3), kind: 'error', note: 'per drafted token' },
          { label: `Speedup at gamma=${gamma}`, value: `${sp.toFixed(2)}x`, kind: sp >= 1 ? 'token' : 'error', note: sp >= 1 ? 'faster' : 'slower' },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Method"
            value={sel}
            options={METHODS.map((x) => ({ value: x.id, label: x.label.split(' ')[0] }))}
            onChange={setSel}
          />
          <Slider label="How predictable is your text?" kind="token" min={0} max={1} step={0.02} value={predict} onChange={setPredict} format={(v) => `${(v * 100).toFixed(0)}% predictable`} />
          <Slider label="Drafts per block gamma" kind="compute" min={1} max={8} value={gamma} onChange={setGamma} />
        </>
      }
    >
      <svg viewBox="0 0 640 392" role="img" aria-label="A scatter plot with acceptance rate on the horizontal axis and draft cost ratio on the vertical axis, showing six speculative decoding methods as labelled points and a diagonal break-even line where acceptance equals cost. Points below and to the right of the line are faster than plain decoding.">
        {/* ── axes ─────────────────────────────────────────── */}
        <Label x={PX} y={34} tone={2} size={11} mono={false} bold>The plane every method lives on</Label>
        <Label x={PX + PW} y={PY - 8} anchor="end" size={9.5} tone={2}>acceptance rate a →</Label>
        <Label x={PX - 10} y={PY + 4} anchor="end" size={9.5} tone={2}>cost rho →</Label>
        <Label x={PX - 10} y={PY + 17} anchor="end" size={8.5} tone={3}>(down = cheap)</Label>

        {/* grid */}
        {[0.2, 0.4, 0.6, 0.8].map((g) => (
          <g key={g}>
            <line x1={px(g)} y1={PY} x2={px(g)} y2={PY + PH} className="v-rule faint" />
            <Label x={px(g)} y={PY + PH + 14} anchor="middle" tone={3} size={8.5}>{g}</Label>
          </g>
        ))}
        {[0.1, 0.2, 0.3].map((g) => (
          <Label key={g} x={PX - 8} y={py(g) + 3} anchor="end" tone={3} size={8.5}>{g}</Label>
        ))}

        {/* ── the break-even diagonal a = rho ───────────────── */}
        <line x1={px(0)} y1={py(0)} x2={px(RMAX)} y2={py(RMAX)} stroke="var(--memory)" strokeWidth={2} />
        <Label x={px(0.175)} y={py(0.175) + 4} size={9} kind="memory" bold>a = ρ</Label>
        <Label x={px(0.055)} y={py(0.055) - 8} size={8.5} kind="error" anchor="middle">lose</Label>
        <Label x={px(0.28)} y={py(0.28) + 14} size={8.5} kind="token" anchor="middle">win</Label>

        {/* ── the methods ──────────────────────────────────── */}
        {METHODS.map((x) => {
          const win = x.a > x.rho;
          const on = x.id === sel;
          // prompt-lookup and n-gram sit almost on top of each other on this
          // plane, so the second label is lifted and given a leader line
          const lift = x.id === 'ngram' ? 22 : 0;
          return (
            <g key={x.id} className="v-fade" style={{ opacity: on ? 1 : 0.75 }}>
              <circle cx={px(x.a)} cy={py(x.rho)} r={on ? 7 : 5} fill={on ? 'var(--text)' : win ? 'var(--token)' : 'var(--error)'}
                stroke={on ? 'var(--text)' : win ? 'var(--token)' : 'var(--error)'} fillOpacity={on ? 0.95 : 0.6} />
              {lift > 0 && (
                <line x1={px(x.a)} y1={py(x.rho) - 6} x2={px(x.a)} y2={py(x.rho) - lift + 3}
                  stroke="var(--rule-2)" strokeWidth={0.8} />
              )}
              <Label x={px(x.a)} y={py(x.rho) - (on ? 14 : 11) - lift} anchor="middle" size={on ? 10 : 8.5} tone={on ? 1 : 2} bold={on}>
                {x.label}
              </Label>
            </g>
          );
        })}

        <Dim x1={PX} x2={PX + PW} y={PY + PH + 30} label="acceptance rate" kind="text" />

        {/* ── the chosen method, unpacked ──────────────────── */}
        <Fade show={s.step >= 1}>
          <rect x={PX + PW + 24} y={PY - 8} width={196} height={PH + 24} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={PX + PW + 34} y={PY + 6} tone={1} size={11} mono={false} bold>{m.label}</Label>
          <Label x={PX + PW + 34} y={PY + 20} tone={3} size={9}>{m.blurb}</Label>

          <Label x={PX + PW + 34} y={PY + 40} size={9}>acceptance</Label>
          <Bar x={PX + PW + 34} y={PY + 46} w={172} h={11} frac={a / AMAX} kind="token" value={`${(a * 100).toFixed(0)}%`} />
          <Label x={PX + PW + 34} y={PY + 76} size={9}>draft cost</Label>
          <Bar x={PX + PW + 34} y={PY + 82} w={172} h={11} frac={m.rho / RMAX} kind="error" value={m.rho.toFixed(3)} />

          <Label x={PX + PW + 34} y={PY + 110} size={10} kind={sp >= 1 ? 'token' : 'error'} bold>
            {sp >= 1 ? `${sp.toFixed(2)}x faster` : `${((1 - sp) * 100).toFixed(0)}% slower`}
          </Label>
          <Label x={PX + PW + 34} y={PY + 124} tone={3} size={9}>{`${r.tokens.toFixed(2)} tokens per block`}</Label>
          <Label x={PX + PW + 34} y={PY + 137} tone={3} size={9}>{`${r.draftSteps.toFixed(2)} draft steps, ${r.wasted.toFixed(2)} wasted`}</Label>
          <Label x={PX + PW + 34} y={PY + 157} size={9} kind={a > m.rho ? 'token' : 'error'}>
            {a > m.rho ? `a > ρ: win by ${((a / m.rho - 1) * 100).toFixed(0)}%` : 'a < ρ: not worth it'}
          </Label>
          <Label x={PX + PW + 34} y={PY + 174} tone={3} size={8.5}>{`on your traffic, a ≈ ${(a * 100).toFixed(0)}%`}</Label>
        </Fade>

        {/* ── the ordering story ────────────────────────────── */}
        <Fade show={s.step >= 2}>
          <Label x={PX} y={PY + PH + 52} tone={2} size={11} mono={false} bold>How the methods order up</Label>
          {[
            ['No network at all', 'lookup, n-gram', 0.005, 0.35],
            ['A second, smaller network', '1B drafting 8B', 0.125, 0.72],
            ['Extra heads on the target', 'Medusa', 0.1, 0.6],
            ['Predict the hidden state', 'EAGLE', 0.15, 0.78],
            ['Train for blocks', 'MTP', 0.2, 0.8],
          ].map((row, i) => {
            const [what, who, cost, acc] = row as [string, string, number, number];
            return (
              <g key={who}>
                <Label x={PX + 12 + i * 116} y={PY + PH + 72} tone={2} size={9}>{what}</Label>
                <Label x={PX + 12 + i * 116} y={PY + PH + 85} tone={1} size={9} bold>{who}</Label>
                <Label x={PX + 12 + i * 116} y={PY + PH + 98} tone={3} size={8.5}>{`ρ ${cost.toFixed(3)} · a ${(acc * 100).toFixed(0)}%`}</Label>
              </g>
            );
          })}
        </Fade>

        <Fade show={s.step >= 7}>
          <Label x={PX} y={PY + PH + 118} size={9.5} tone={2}>Every method trades cost against accuracy. Pick the one whose corner matches your traffic.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}