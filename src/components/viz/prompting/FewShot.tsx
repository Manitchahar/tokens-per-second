import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, fmt } from '../lib/svg';
import {
  accuracy, spread, tokensPerCall, exampleCost, prefillMs, PRICES, TOK, CHANCE, CALLS_PER_DAY,
  type ModelKind, type Variant, type PriceId,
} from './fewshot';

const STEPS = [
  'The task, and the zero-shot answer',
  'One example pins the format',
  'GPT-3: a base model learns from the prompt',
  'An instruction-tuned model starts high',
  'Format matters more than correct labels',
  'Order matters: the recency trap',
  'Every example is paid on every call',
];
const CAPTIONS = [
  'Sort support tickets into BILLING, BUG, ACCOUNT or OTHER. An instruction-tuned model gets the idea with no examples, but answers in a sentence your parser cannot read.',
  'Add one labelled example and the answer snaps to the format: one word, from the list. One example taught the shape of the answer, not the task.',
  'A base model with no examples is near chance. Each example lifts it: in-context learning, the GPT-3 discovery from Module 3B. No weight changes.',
  'An instruction-tuned model already starts high. Examples now buy a few points, mostly in the first two or three, then the curve goes flat.',
  'Examples whose format disagrees cost real accuracy. Examples with shuffled, wrong labels cost far less. The examples mostly teach format, label space and input style.',
  'End with several examples of one label and the model leans toward that label. Different orders of the same examples give a wide spread. Balance and interleave them.',
  'Every example is sent, and billed, on every call. 5 examples × 200 tokens is 1,000 extra input tokens per call, 100 million a day at 100,000 calls.',
];

const DEFAULTS: { k: number; model: ModelKind; v: Variant }[] = [
  { k: 0, model: 'instruct', v: 'clean' },
  { k: 1, model: 'instruct', v: 'clean' },
  { k: 4, model: 'base', v: 'clean' },
  { k: 4, model: 'instruct', v: 'clean' },
  { k: 4, model: 'instruct', v: 'format' },
  { k: 4, model: 'instruct', v: 'order' },
  { k: 5, model: 'instruct', v: 'clean' },
];

const KMAX = 16;
const BAR_X = 16;
const BAR_W = 608;
const PX_PER_TOK = BAR_W / tokensPerCall(KMAX);
// chart frame
const CX0 = 64;
const CX1 = 572;
const CY0 = 92; // accuracy 1.0
const CY1 = 274; // accuracy 0.2
const xOf = (k: number) => CX0 + (k / KMAX) * (CX1 - CX0);
const yOf = (a: number) => CY1 - ((a - 0.2) / 0.8) * (CY1 - CY0);
const yTok = (t: number) => CY1 - (t / 3600) * (CY1 - CY0);

function curve(fn: (k: number) => number) {
  const pts: string[] = [];
  for (let i = 0; i <= 64; i++) { const k = (i / 64) * KMAX; pts.push(`${xOf(k).toFixed(1)},${yOf(fn(k)).toFixed(1)}`); }
  return pts.join(' ');
}
function band(fn: (k: number) => number, w: (k: number) => number) {
  const up: string[] = [];
  const dn: string[] = [];
  for (let i = 0; i <= 64; i++) {
    const k = (i / 64) * KMAX;
    up.push(`${xOf(k).toFixed(1)},${yOf(Math.min(1, fn(k) + w(k))).toFixed(1)}`);
    dn.unshift(`${xOf(k).toFixed(1)},${yOf(Math.max(0.2, fn(k) - w(k))).toFixed(1)}`);
  }
  return [...up, ...dn].join(' ');
}

const VARIANTS: { value: Variant; label: string }[] = [
  { value: 'clean', label: 'clean' },
  { value: 'format', label: 'mixed formats' },
  { value: 'labels', label: 'random labels' },
  { value: 'order', label: 'one label last' },
];

export default function FewShot() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [k, setK] = useState(0);
  const [model, setModel] = useState<ModelKind>('instruct');
  const [v, setV] = useState<Variant>('clean');
  const [price, setPrice] = useState<PriceId>('mini');

  useEffect(() => {
    const d = DEFAULTS[st];
    setK(d.k); setModel(d.model); setV(d.v);
  }, [st]);

  const acc = accuracy(model, k, v);
  const sp = spread(model, k, v);
  const tokens = tokensPerCall(k);
  const P = PRICES.find((p) => p.id === price)!;
  const cost = exampleCost(k, P.usdPerM);

  // the prompt, drawn to scale
  const blocks: { x: number; w: number; kind: 'instr' | 'ex' | 'query' | 'out'; label: string }[] = [];
  let x = BAR_X;
  const push = (n: number, kind: 'instr' | 'ex' | 'query' | 'out', label: string) => { const w = n * PX_PER_TOK; blocks.push({ x, w, kind, label }); x += w; };
  push(TOK.instruction, 'instr', 'instr');
  for (let i = 0; i < k; i++) push(TOK.perExample, 'ex', `ex ${i + 1}`);
  push(TOK.query, 'query', 'ticket');
  push(Math.max(TOK.output, 30), 'out', '');

  const showChart = st >= 2;
  const showInstruct = st !== 2;
  const fnSel = (kk: number) => accuracy(model, kk, v);

  const notes: Record<number, [string, string]> = {
    0: ['Zero-shot: the instruction alone. The model understood the task but not the shape of the answer.', 'A label buried in a sentence fails the parser, so for your program this counts as wrong.'],
    1: ['One-shot: a single example. Same task, same model, and now the answer is one word from the list.', 'The example taught what an answer looks like. That is most of what examples do for an instruct model.'],
    2: ['A base model with zero examples sits near chance (25% with four labels). Examples lift it a lot.', 'Learning from the prompt, with frozen weights: in-context learning, the GPT-3 result (Module 3B).'],
    3: ['Instruction tuning (Module 3C) moved the whole curve up. Zero-shot already works.', 'Most of the remaining gain arrives in the first two or three examples; after that it is flat.'],
    4: ['Inconsistent formats hurt; wrong labels barely do. Try both with the toggle below.', 'The examples mostly teach the label space, the input style and the format, not the input → label mapping.'],
    5: ['A run of one label at the end tilts answers toward it (recency bias). Band = spread over orders.', 'Balance the labels and interleave them; then the order stops mattering much.'],
    6: [`${k} examples × ${TOK.perExample} tokens = ${fmt.int(k * TOK.perExample)} extra input tokens per call, ≈ ${prefillMs(k * TOK.perExample).toFixed(0)} ms of 8B prefill on an H100.`, `× ${fmt.int(CALLS_PER_DAY)} calls/day = ${k ? fmt.si(cost.perDay) : '0'} tokens/day; at ${P.label.replace(' / M', '')} per million that is $${cost.usdPerDay.toFixed(2)} a day.`],
  };
  const [l1, l2] = notes[st];

  return (
    <Stage
      rootRef={ref}
      title="How many examples? Accuracy versus tokens"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['memory', 'prompt tokens'], ['token', 'answer'], ['text', 'accuracy (illustrative)'], ['error', 'unusable answer']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'examples', value: `${k}`, kind: 'text', note: k === 0 ? 'zero-shot' : k === 1 ? 'one-shot' : 'few-shot' },
          { label: 'accuracy', value: `${(acc * 100).toFixed(0)}%${sp > 0.005 ? ` ± ${(sp * 100).toFixed(0)}` : ''}`, kind: 'text', note: `illustrative · ${model === 'base' ? 'base model' : 'instruction-tuned'}` },
          { label: 'input tokens per call', value: fmt.int(tokens), kind: 'memory', note: `150 + ${k} × 200 + 100` },
          { label: 'examples cost', value: `$${cost.usdPerDay.toFixed(2)}/day`, kind: 'memory', note: `${fmt.int(CALLS_PER_DAY)}/day at ${P.label} (${P.note})` },
        ]} />
      }
      controls={
        <>
          <Slider label="Examples in the prompt (k)" value={k} min={0} max={KMAX} onChange={setK} format={(n) => `k = ${n}`} kind="memory" />
          {st >= 2 && st <= 5 && (
            <Segmented label="Model" value={model} options={[{ value: 'base' as const, label: 'base' }, { value: 'instruct' as const, label: 'instruction-tuned' }]} onChange={setModel} />
          )}
          {st >= 4 && st <= 5 && <Segmented label="The examples are…" value={v} options={VARIANTS} onChange={setV} />}
          {st === 6 && (
            <Segmented label="Dated example input price" value={price} options={PRICES.map((p) => ({ value: p.id, label: p.label }))} onChange={setPrice} />
          )}
        </>
      }
    >
      <svg viewBox="0 0 640 360" role="img" aria-label="A prompt drawn to scale grows by 200 tokens per example, while a chart shows illustrative accuracy rising quickly for the first few examples and then flattening, for a base model and an instruction-tuned model.">
        {/* ── the prompt, to scale ── */}
        <Label x={16} y={16} tone={2} size={10} bold>one call’s prompt, drawn to scale</Label>
        <Label x={624} y={16} anchor="end" tone={3} size={9}>{`${fmt.int(tokens)} input tokens · ${TOK.perExample} per example`}</Label>
        {blocks.map((b, i) => (
          <g key={`${b.kind}-${i}`}>
            <rect x={b.x} y={24} width={Math.max(1, b.w - 1)} height={22} rx={2}
              fill={b.kind === 'out' ? 'var(--token)' : b.kind === 'ex' ? 'var(--memory)' : 'var(--memory-dim)'}
              stroke={b.kind === 'out' ? 'var(--token)' : 'var(--memory)'} strokeWidth={0.8}
              style={{ transition: 'x .5s var(--ease-out), width .5s var(--ease-out)' }} />
            {b.label && b.w > 30 && (
              <text x={b.x + b.w / 2} y={38.5} textAnchor="middle" className="v-label" style={{ fontSize: 8, fontWeight: 600, fill: b.kind === 'ex' ? '#0e1116' : 'var(--text)' }}>{b.label}</text>
            )}
          </g>
        ))}
        <Label x={16} y={62} tone={3} size={9}>{k === 0 ? 'no examples: instruction + ticket' : `the ${k} example${k > 1 ? 's are' : ' is'} resent with every single ticket`}</Label>

        {/* ── steps 1–2: the literal prompt ── */}
        <Fade show={!showChart}>
          <g>
            <rect x={16} y={76} width={608} height={st === 0 ? 120 : 144} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" className="v-fade" />
            <text x={30} y={98} className="v-label t3" style={{ fontSize: 9 }}>instruction</text>
            <text x={130} y={98} className="v-label t1" style={{ fontSize: 9.5 }}>Classify the ticket as BILLING, BUG, ACCOUNT or OTHER.</text>
            <Fade show={st === 1}>
              <text x={30} y={122} className="v-label t3" style={{ fontSize: 9 }}>example</text>
              <text x={130} y={122} className="v-label" style={{ fontSize: 9.5, fill: 'var(--memory)' }}>{'Ticket: "The app crashes when I upload a photo."'}</text>
              <text x={130} y={138} className="v-label" style={{ fontSize: 9.5, fill: 'var(--memory)' }}>Label: BUG</text>
            </Fade>
            <text x={30} y={st === 0 ? 122 : 162} className="v-label t3 v-fade" style={{ fontSize: 9 }}>ticket</text>
            <text x={130} y={st === 0 ? 122 : 162} className="v-label t1" style={{ fontSize: 9.5 }}>{`Ticket: "I was charged twice for my plan this month."${st === 1 ? '' : ''}`}</text>
            {st === 1 && <text x={130} y={178} className="v-label t1" style={{ fontSize: 9.5 }}>Label:</text>}
            <text x={30} y={st === 0 ? 150 : 204} className="v-label t3" style={{ fontSize: 9 }}>reply</text>
            {st === 0 ? (
              <>
                <text x={130} y={150} className="v-label" style={{ fontSize: 9.5, fill: 'var(--error)' }}>This ticket appears to concern a billing problem, since the</text>
                <text x={130} y={164} className="v-label" style={{ fontSize: 9.5, fill: 'var(--error)' }}>customer reports a duplicate charge. Suggested category: Billing.</text>
                <text x={130} y={186} className="v-label t3" style={{ fontSize: 9 }}>right idea, wrong shape: parse("…") ≠ "BILLING"</text>
              </>
            ) : (
              <>
                <rect x={128} y={192} width={60} height={18} rx={3} fill="var(--token)" />
                <text x={158} y={205} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fontWeight: 700, fill: '#0e1116' }}>BILLING</text>
                <text x={200} y={205} className="v-label t3" style={{ fontSize: 9 }}>one token-cheap word, straight into your parser</text>
              </>
            )}
          </g>
        </Fade>

        {/* ── steps 3+: the chart ── */}
        <Fade show={showChart}>
          <g>
            {[0.2, 0.4, 0.6, 0.8, 1].map((a) => (
              <g key={a}>
                <line x1={CX0} x2={CX1} y1={yOf(a)} y2={yOf(a)} className="v-rule faint" />
                <text x={CX0 - 6} y={yOf(a) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>{`${(a * 100).toFixed(0)}%`}</text>
              </g>
            ))}
            {[0, 4, 8, 12, 16].map((kk) => (
              <text key={kk} x={xOf(kk)} y={CY1 + 14} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{kk}</text>
            ))}
            <text x={(CX0 + CX1) / 2} y={CY1 + 28} textAnchor="middle" className="v-label t3" style={{ fontSize: 9 }}>examples in the prompt, k</text>
            <text x={CX0 - 6} y={CY0 - 10} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>accuracy</text>
            {/* right axis: tokens per call */}
            {[0, 1200, 2400, 3600].map((t) => (
              <text key={t} x={CX1 + 6} y={yTok(t) + 3} className="v-label" style={{ fontSize: 8.5, fill: 'var(--memory)' }}>{fmt.int(t)}</text>
            ))}
            <text x={CX1 + 6} y={CY0 - 10} className="v-label" style={{ fontSize: 8.5, fill: 'var(--memory)' }}>tokens/call</text>
            <line x1={xOf(0)} y1={yTok(tokensPerCall(0))} x2={xOf(KMAX)} y2={yTok(tokensPerCall(KMAX))} stroke="var(--memory)" strokeWidth={1.4} strokeDasharray="5 4" />
            {/* chance line */}
            <line x1={CX0} x2={CX1} y1={yOf(CHANCE)} y2={yOf(CHANCE)} stroke="var(--text-3)" strokeDasharray="2 3" />
            <text x={CX1 - 4} y={yOf(CHANCE) - 4} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>chance (4 labels)</text>

            {/* spread across orderings */}
            {sp > 0 && <polygon points={band(fnSel, (kk) => spread(model, kk, v))} fill="rgb(232 230 225 / 0.10)" stroke="none" className="v-fade" />}
            {/* base and instruct clean curves */}
            <polyline points={curve((kk) => accuracy('base', kk))} fill="none" stroke="var(--text-2)" strokeWidth={model === 'base' ? 2 : 1.2} strokeDasharray={model === 'base' ? undefined : '4 3'} className="v-fade" style={{ opacity: model === 'base' ? 1 : 0.55 }} />
            <text x={xOf(KMAX) + 0} y={yOf(accuracy('base', KMAX)) + 14} textAnchor="end" className="v-label t2" style={{ fontSize: 8.5 }}>base</text>
            <g className="v-fade" style={{ opacity: showInstruct ? 1 : 0 }}>
              <polyline points={curve((kk) => accuracy('instruct', kk))} fill="none" stroke="var(--text)" strokeWidth={model === 'instruct' ? 2 : 1.2} strokeDasharray={model === 'instruct' ? undefined : '4 3'} style={{ opacity: model === 'instruct' ? 1 : 0.55 }} />
              <text x={xOf(KMAX)} y={yOf(accuracy('instruct', KMAX)) - 6} textAnchor="end" className="v-label t1" style={{ fontSize: 8.5 }}>instruction-tuned</text>
            </g>
            {/* the hurt variant */}
            {v !== 'clean' && (
              <polyline points={curve(fnSel)} fill="none" stroke="var(--error)" strokeWidth={1.8} className="v-fade" />
            )}
            {/* current k marker */}
            <line x1={xOf(k)} x2={xOf(k)} y1={CY0} y2={CY1} stroke="var(--text-3)" strokeDasharray="2 3" style={{ transition: 'x1 .4s, x2 .4s' }} />
            <circle cx={xOf(k)} cy={yOf(acc)} r={4.5} fill={v === 'clean' ? 'var(--text)' : 'var(--error)'} stroke="var(--ground-2)" strokeWidth={1.5} style={{ transition: 'cx .4s var(--ease-out), cy .4s var(--ease-out)' }} />
            <circle cx={xOf(k)} cy={yTok(tokens)} r={4} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={1.5} style={{ transition: 'cx .4s var(--ease-out), cy .4s var(--ease-out)' }} />
          </g>
        </Fade>

        {/* ── notes ── */}
        <line x1={16} x2={624} y1={314} y2={314} className="v-rule faint" />
        <Label x={16} y={332} tone={1} size={9.5}>{l1}</Label>
        <Label x={16} y={350} tone={2} size={9}>{l2}</Label>
      </svg>
    </Stage>
  );
}
