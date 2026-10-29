import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Move, Arrow, fmt } from '../lib/svg';
import {
  INTENTS, MONO_SECTIONS, MONO_SYSTEM, MONO_ACC, MONO_MEAN, SUB_ACC, T, tokens, timeline, ttft, total, TPOT_MS,
  type Intent, type Seg,
} from './chain';

const STEPS = [
  'The monolith: one prompt for everything',
  'Every fix touches every case',
  'Split it: classify the intent first',
  'Route to a small, focused prompt',
  'Accuracy multiplies along a chain',
  'Tokens: pay only for what this ticket needs',
  'Latency: one more round trip',
];
const CAPTIONS = [
  'One 3,000-token system prompt handles billing, bugs, account recovery and refunds. A billing ticket needs about 600 of those tokens, and pays for all 3,000.',
  'Add a rule to fix refunds and the refunds score goes up while billing and account recovery quietly go down. In one big prompt every instruction competes with every other.',
  'Split the job. A small router prompt reads the ticket and outputs one label, a few tokens long. Its whole task is classification, which is easy to test.',
  'The label picks one focused prompt that contains only the policy this ticket needs. Each sub-prompt is short enough to read, test and fix on its own.',
  'A chain is only right if every step is right: accuracy multiplies. 97% routing × 94% answering = 91.2%. Drag the router accuracy down and watch the advantage vanish below 87%.',
  'Per request, the chain sends 1,400 input tokens instead of 3,150: the router prompt and one sub-prompt, never the policies the ticket does not need.',
  'The price is a second call: another network round trip and another prefill before the answer starts. About 21 ms more time to first token here, 77 ms if the monolith is cached.',
];

const DEF_INTENT: Intent = 'billing';
const SUB_Y: Record<Intent, number> = { billing: 30, bug: 72, account: 114, other: 156 };
const MONO_X = 82;
const MONO_W = 140;
const MONO_Y = 30;
const MONO_H = 158;
const ROUTER = { x: 404, y: 90, w: 76, h: 36 };
const SUB_X = 512;
const SUB_W = 76;
const SUB_H = 32;

function SegTrack({ segs, x0, y, scale, maxMs }: { segs: Seg[]; x0: number; y: number; scale: number; maxMs: number }) {
  let t = 0;
  return (
    <g>
      {segs.map((sg, i) => {
        const start = t;
        t += sg.ms;
        const vis = Math.max(0, Math.min(sg.ms, maxMs - start));
        if (vis <= 0) return null;
        const x = x0 + start * scale;
        const w = vis * scale;
        const fill = sg.kind === 'idle' ? 'url(#tps-hatch)' : sg.kind === 'compute' ? 'var(--compute)' : 'var(--token)';
        const stroke = sg.kind === 'idle' ? 'var(--idle)' : sg.kind === 'compute' ? 'var(--compute)' : 'var(--token)';
        return (
          <g key={i}>
            <rect x={x} y={y} width={Math.max(1.5, w - 0.5)} height={16} rx={1.5} fill={fill} stroke={stroke} strokeWidth={0.8} className="v-cell" />
            {sg.kind === 'compute' && <text x={x + w / 2} y={y - 3} textAnchor="middle" className="v-label" style={{ fontSize: 7.5, fill: 'var(--compute)' }}>{`${sg.ms.toFixed(0)}`}</text>}
            {sg.kind === 'token' && sg.label === 'label' && <text x={x + w / 2} y={y - 3} textAnchor="middle" className="v-label" style={{ fontSize: 7.5, fill: 'var(--token)' }}>{`${sg.ms.toFixed(0)}`}</text>}
          </g>
        );
      })}
    </g>
  );
}

export default function PromptChain() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [intent, setIntent] = useState<Intent>(DEF_INTENT);
  const [v2, setV2] = useState(false);
  const [router, setRouter] = useState(0.97);
  const [cached, setCached] = useState(false);

  useEffect(() => {
    if (st === 1) setV2(true);
    if (st === 4) setRouter(0.97);
    if (st === 6) setCached(false);
  }, [st]);

  const chainOn = st >= 2;
  const routed = st >= 3;
  const chainAcc = router * SUB_ACC;
  const breakEven = MONO_MEAN / SUB_ACC;

  const monoAcc: Record<Intent, number> = v2 && st === 1
    ? { billing: MONO_ACC.billing - 0.03, bug: MONO_ACC.bug, account: MONO_ACC.account - 0.02, other: MONO_ACC.other + 0.06 }
    : MONO_ACC;

  const segMono = timeline('mono', cached);
  const segChain = timeline('chain', cached);

  // a ticket dot travelling through the chain
  const dot = !chainOn ? { x: 352, y: 108 } : !routed ? { x: ROUTER.x + ROUTER.w / 2, y: ROUTER.y + ROUTER.h / 2 } : st >= 4 ? { x: 612, y: SUB_Y[intent] + SUB_H / 2 } : { x: SUB_X + SUB_W / 2, y: SUB_Y[intent] + SUB_H / 2 };

  // monolith section geometry
  let yAcc = MONO_Y;
  const sections = MONO_SECTIONS.map((sec) => {
    const h = (sec.tokens / MONO_SYSTEM) * MONO_H;
    const r = { ...sec, y: yAcc, h };
    yAcc += h;
    return r;
  });

  const notes: [string, string][] = [
    ['A billing ticket uses ~600 of the 3,000 system tokens and pays for all of them, on every call.', 'Long single prompts are also hard to reason about: which rule caused this answer?'],
    ['v2 adds a refund rule: refunds improve, billing and account recovery regress. The mean barely moves.', 'In one prompt every instruction competes for the model’s attention. Fixes become whack-a-mole.'],
    ['Step 1 of the chain: a 400-token router prompt. Its only output is a label, 3 tokens.', 'Classification is the easiest thing to evaluate: you can measure the router alone (Module B2).'],
    ['Step 2: the label selects a 700-token prompt holding only this intent’s policy.', 'Each sub-prompt can be written, tested and improved without touching the others.'],
    [`p(chain) = p(router) × p(answer) = ${(router * 100).toFixed(1)}% × ${(SUB_ACC * 100).toFixed(0)}% = ${(chainAcc * 100).toFixed(1)}%   vs   monolith ${(MONO_MEAN * 100).toFixed(0)}%`, `Break-even router accuracy = ${(MONO_MEAN * 100).toFixed(0)}% ÷ ${(SUB_ACC * 100).toFixed(0)}% = ${(breakEven * 100).toFixed(1)}%. Misroutes get the wrong policy: add an OTHER route.`],
    [`Input tokens per ticket: ${fmt.int(tokens.monoIn)} → ${fmt.int(tokens.chainIn)} (${(100 - (tokens.chainIn / tokens.monoIn) * 100).toFixed(0)}% fewer). Output: ${tokens.monoOut} → ${tokens.chainOut}.`, 'The router’s label is the only extra output. Keep it a label, never a sentence: each token is ~7.5 ms.'],
    [`Time to first token: ${ttft(segMono).toFixed(0)} ms → ${ttft(segChain).toFixed(0)} ms. Total: ${fmt.int(total(segMono))} → ${fmt.int(total(segChain))} ms.`, `Assumes 50 ms of network per call. The 250 answer tokens (${fmt.int(T.answer * TPOT_MS)} ms) dominate either way.`],
  ];
  const [l1, l2] = notes[st];

  const TL_X = 120;
  const TL_MAX = 300; // ms shown
  const TL_SCALE = (600 - TL_X) / TL_MAX;

  return (
    <Stage
      rootRef={ref}
      title="One big prompt, or a chain of small ones"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['memory', 'prompt tokens'], ['token', 'generated'], ['compute', 'prefill'], ['idle', 'network / waiting'], ['error', 'regression']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'input tokens / ticket', value: `${fmt.int(tokens.monoIn)} → ${chainOn ? fmt.int(tokens.chainIn) : '?'}`, kind: 'memory', note: 'monolith → chain' },
          { label: 'accuracy (illustrative)', value: `${(MONO_MEAN * 100).toFixed(0)}% → ${chainOn ? `${(chainAcc * 100).toFixed(1)}%` : '?'}`, kind: 'text', note: `router ${(router * 100).toFixed(1)}% × answer ${(SUB_ACC * 100).toFixed(0)}%` },
          { label: 'time to first token', value: `${ttft(segMono).toFixed(0)} → ${chainOn ? ttft(segChain).toFixed(0) : '?'} ms`, kind: 'compute', note: cached ? 'static prompts cached' : 'no prefix cache · 8B on H100' },
        ]} />
      }
      controls={
        <>
          {st <= 3 && (
            <Segmented label="Incoming ticket" value={intent} options={INTENTS.map((i) => ({ value: i, label: i }))} onChange={setIntent} />
          )}
          {st === 1 && (
            <Segmented label="Monolith version" value={v2 ? 'v2' : 'v1'} options={[{ value: 'v1', label: 'v1' }, { value: 'v2', label: 'v2: + refund rule' }]} onChange={(x) => setV2(x === 'v2')} />
          )}
          {st >= 4 && (
            <Slider label="Router accuracy" value={router} min={0.8} max={0.995} step={0.005} onChange={setRouter} format={(x) => `${(x * 100).toFixed(1)}%`} />
          )}
          {st === 6 && (
            <Segmented label="Static prompts in the prefix cache" value={cached ? 'on' : 'off'} options={[{ value: 'off', label: 'off' }, { value: 'on', label: 'on' }]} onChange={(x) => setCached(x === 'on')} />
          )}
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Left: one 3,000-token prompt with six sections, of which a ticket needs one. Right: a small router prompt sends the ticket to one of four focused prompts. Below, panels compare accuracy, tokens and time to first token.">
        {/* ── left: the monolith ── */}
        <Label x={16} y={16} tone={2} size={10} bold>one prompt</Label>
        <Label x={290} y={16} anchor="end" tone={3} size={9}>{`${fmt.int(MONO_SYSTEM)} system tokens`}</Label>
        <rect x={16} y={98} width={46} height={20} rx={3} fill="var(--memory)" />
        <text x={39} y={111.5} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fontWeight: 700, fill: '#0e1116' }}>{intent}</text>
        <Arrow x1={64} y1={108} x2={MONO_X - 3} y2={108} />
        {sections.map((sec) => {
          const rel = sec.intent === intent;
          const always = !sec.intent;
          return (
            <g key={sec.label}>
              <rect x={MONO_X} y={sec.y} width={MONO_W} height={sec.h - 1.5} rx={2}
                fill={rel ? 'var(--memory)' : always ? 'var(--memory-dim)' : 'transparent'}
                stroke="var(--memory)" strokeWidth={0.8} strokeDasharray={rel || always ? undefined : '3 3'} className="v-cell" />
              <text x={MONO_X + 6} y={sec.y + sec.h / 2 + 3} className="v-label" style={{ fontSize: 8.5, fill: rel ? '#0e1116' : always ? 'var(--text)' : 'var(--text-3)', fontWeight: rel ? 700 : 400 }}>{sec.label}</text>
              <text x={MONO_X + MONO_W - 5} y={sec.y + sec.h / 2 + 3} textAnchor="end" className="v-label" style={{ fontSize: 8, fill: rel ? '#0e1116' : 'var(--text-3)' }}>{sec.tokens}</text>
            </g>
          );
        })}
        <Arrow x1={MONO_X + MONO_W + 3} y1={108} x2={MONO_X + MONO_W + 20} y2={108} />
        <rect x={MONO_X + MONO_W + 23} y={98} width={44} height={20} rx={3} fill="var(--token)" />
        <text x={MONO_X + MONO_W + 45} y={111.5} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fontWeight: 700, fill: '#0e1116' }}>answer</text>

        <line x1={306} x2={306} y1={6} y2={196} className="v-rule faint" />

        {/* ── right: the chain ── */}
        <g className="v-fade" style={{ opacity: chainOn ? 1 : 0.18 }}>
          <Label x={322} y={16} tone={2} size={10} bold>a chain</Label>
          <Label x={624} y={16} anchor="end" tone={3} size={9}>{`${T.router} router + ${T.sub} per focused prompt`}</Label>
          <rect x={322} y={98} width={52} height={20} rx={3} fill="var(--memory)" />
          <text x={348} y={111.5} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fontWeight: 700, fill: '#0e1116' }}>{intent}</text>
          <Arrow x1={376} y1={108} x2={ROUTER.x - 4} y2={108} />
          <rect x={ROUTER.x} y={ROUTER.y} width={ROUTER.w} height={ROUTER.h} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} />
          <text x={ROUTER.x + ROUTER.w / 2} y={ROUTER.y + 15} textAnchor="middle" className="v-label t1" style={{ fontSize: 9, fontWeight: 700 }}>router</text>
          <text x={ROUTER.x + ROUTER.w / 2} y={ROUTER.y + 28} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{`${T.router} tokens`}</text>
          {INTENTS.map((it) => {
            const on = routed && it === intent;
            return (
              <g key={it}>
                <path d={`M${ROUTER.x + ROUTER.w + 2},${ROUTER.y + ROUTER.h / 2} C${ROUTER.x + ROUTER.w + 18},${ROUTER.y + ROUTER.h / 2} ${SUB_X - 18},${SUB_Y[it] + SUB_H / 2} ${SUB_X - 3},${SUB_Y[it] + SUB_H / 2}`}
                  fill="none" stroke={on ? 'var(--token)' : 'var(--rule-2)'} strokeWidth={on ? 1.6 : 1} strokeDasharray={on ? undefined : '3 3'} className="v-cell" />
                <rect x={SUB_X} y={SUB_Y[it]} width={SUB_W} height={SUB_H} rx={3}
                  fill={on ? 'var(--memory)' : 'transparent'} stroke="var(--memory)" strokeWidth={0.9} strokeDasharray={on ? undefined : '3 3'} className="v-cell" />
                <text x={SUB_X + SUB_W / 2} y={SUB_Y[it] + 13} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fontWeight: 700, fill: on ? '#0e1116' : 'var(--text-2)' }}>{it}</text>
                <text x={SUB_X + SUB_W / 2} y={SUB_Y[it] + 25} textAnchor="middle" className="v-label" style={{ fontSize: 7.5, fill: on ? '#0e1116' : 'var(--text-3)' }}>{`${T.sub} tokens`}</text>
              </g>
            );
          })}
          <Fade show={st >= 2}>
            <rect x={ROUTER.x + 8} y={ROUTER.y + ROUTER.h + 8} width={60} height={16} rx={3} fill="var(--token)" />
            <text x={ROUTER.x + 38} y={ROUTER.y + ROUTER.h + 19.5} textAnchor="middle" className="v-label" style={{ fontSize: 8, fontWeight: 700, fill: '#0e1116' }}>{intent.toUpperCase()}</text>
            <text x={ROUTER.x + 38} y={ROUTER.y + ROUTER.h + 36} textAnchor="middle" className="v-label t3" style={{ fontSize: 7.5 }}>3-token label</text>
          </Fade>
          <Fade show={routed}>
            <Arrow x1={SUB_X + SUB_W + 3} y1={SUB_Y[intent] + SUB_H / 2} x2={SUB_X + SUB_W + 14} y2={SUB_Y[intent] + SUB_H / 2} kind="token" />
          </Fade>
          <Move x={dot.x} y={dot.y} opacity={chainOn ? 1 : 0}>
            <circle r={4} fill="var(--text)" stroke="var(--ground-2)" strokeWidth={1.5} />
          </Move>
        </g>

        <line x1={16} x2={624} y1={204} y2={204} className="v-rule faint" />

        {/* ── middle panel ── */}
        <Fade show={st === 0}>
          <Label x={16} y={224} tone={2} size={9.5}>{`what a ${intent} ticket needs from the system prompt`}</Label>
          {sections.map((sec, i) => {
            const x0 = 16 + (MONO_SECTIONS.slice(0, i).reduce((a, q) => a + q.tokens, 0) / MONO_SYSTEM) * 608;
            const w = (sec.tokens / MONO_SYSTEM) * 608;
            const need = sec.intent === intent || !sec.intent;
            return <rect key={sec.label} x={x0} y={234} width={w - 1.5} height={18} rx={2} fill={need ? (sec.intent ? 'var(--memory)' : 'var(--memory-dim)') : 'var(--error-dim)'} stroke={need ? 'var(--memory)' : 'var(--error)'} strokeWidth={0.8} className="v-cell" />;
          })}
          <Label x={16} y={272} tone={3} size={9}>{`red: ${fmt.int(MONO_SYSTEM - 750 - (MONO_SECTIONS.find((q) => q.intent === intent)?.tokens ?? 0))} tokens of other intents’ policy, paid for and unused on this ticket`}</Label>
        </Fade>

        <Fade show={st === 1}>
          <Label x={16} y={224} tone={2} size={9.5}>monolith accuracy by ticket type (illustrative)</Label>
          {INTENTS.map((it, i) => {
            const a = monoAcc[it];
            const d = a - MONO_ACC[it];
            const y = 234 + i * 15;
            return (
              <g key={it}>
                <text x={86} y={y + 9} textAnchor="end" className="v-label t2" style={{ fontSize: 8.5 }}>{it}</text>
                <rect x={92} y={y} width={a * 420} height={11} rx={1.5} fill="var(--text-2)" style={{ transition: 'width .5s var(--ease-out)' }} />
                {d < 0 && <rect x={92 + a * 420} y={y} width={-d * 420} height={11} fill="var(--error-dim)" stroke="var(--error)" strokeWidth={0.8} />}
                <text x={92 + Math.max(a, MONO_ACC[it]) * 420 + 6} y={y + 9} className="v-label" style={{ fontSize: 8.5, fill: d < 0 ? 'var(--error)' : d > 0 ? 'var(--text)' : 'var(--text-2)' }}>
                  {`${(a * 100).toFixed(0)}%${d ? ` (${d > 0 ? '+' : ''}${(d * 100).toFixed(0)})` : ''}`}
                </text>
              </g>
            );
          })}
        </Fade>

        <Fade show={st === 2 || st === 3}>
          <Label x={16} y={224} tone={2} size={9.5}>{st === 2 ? 'router prompt (sketch)' : `focused prompt for ${intent} (sketch)`}</Label>
          <rect x={16} y={232} width={608} height={60} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
          {(st === 2
            ? ['Classify the support ticket into exactly one label:', 'BILLING, BUG, ACCOUNT, OTHER. Definitions and 8 balanced examples follow.', 'Reply with the label only.']
            : [`You are a support engineer handling ${intent} tickets for our product.`, `Policy: (only the ${MONO_SECTIONS.find((q) => q.intent === intent)?.label ?? ''} section, ~600 tokens)`, 'Answer in under 120 words, then a JSON line {"escalate": true|false}.']
          ).map((line, i) => (
            <text key={i} x={28} y={250 + i * 15} className="v-label t1" style={{ fontSize: 9 }}>{line}</text>
          ))}
        </Fade>

        <Fade show={st === 4}>
          {[
            { label: 'monolith', val: MONO_MEAN, kind: 'text' },
            { label: 'chain', val: chainAcc, kind: chainAcc < MONO_MEAN ? 'error' : 'text' },
          ].map((b, i) => (
            <g key={b.label}>
              <text x={86} y={232 + i * 22} textAnchor="end" className="v-label t2" style={{ fontSize: 9 }}>{b.label}</text>
              <rect x={92} y={222 + i * 22} width={b.val * 460} height={14} rx={2} fill={b.kind === 'error' ? 'var(--error)' : 'var(--text-2)'} style={{ transition: 'width .4s var(--ease-out), fill .3s' }} />
              <text x={92 + b.val * 460 + 6} y={233 + i * 22} className="v-label t1" style={{ fontSize: 9 }}>{`${(b.val * 100).toFixed(1)}%`}</text>
            </g>
          ))}
          <line x1={92 + MONO_MEAN * 460} x2={92 + MONO_MEAN * 460} y1={216} y2={262} stroke="var(--text-3)" strokeDasharray="2 3" />
          <Label x={92} y={282} tone={3} size={9}>{`a misrouted ticket is answered with the wrong policy: ${((1 - router) * 100).toFixed(1)}% of tickets here`}</Label>
        </Fade>

        <Fade show={st === 5}>
          {[
            { label: 'monolith', parts: [{ n: MONO_SYSTEM, l: 'system 3,000', dim: true }, { n: T.ticket, l: 'ticket' }] },
            { label: 'chain', parts: [{ n: T.router, l: 'router 400', dim: true }, { n: T.ticket, l: 'ticket' }, { n: T.sub, l: 'focused 700', dim: true }, { n: T.ticket, l: 'ticket' }] },
          ].map((row, i) => {
            let x = 92;
            return (
              <g key={row.label}>
                <text x={86} y={236 + i * 30} textAnchor="end" className="v-label t2" style={{ fontSize: 9 }}>{row.label}</text>
                {row.parts.map((p, j) => {
                  const w = (p.n / tokens.monoIn) * 470;
                  const r = <g key={j}><rect x={x} y={224 + i * 30} width={w - 1.5} height={18} rx={2} fill={p.dim ? 'var(--memory-dim)' : 'var(--memory)'} stroke="var(--memory)" strokeWidth={0.8} />{w > 46 && <text x={x + w / 2} y={236.5 + i * 30} textAnchor="middle" className="v-label t1" style={{ fontSize: 8 }}>{p.l}</text>}</g>;
                  x += w;
                  return r;
                })}
                <text x={x + 6} y={236.5 + i * 30} className="v-label t1" style={{ fontSize: 9 }}>{fmt.int(i === 0 ? tokens.monoIn : tokens.chainIn)}</text>
              </g>
            );
          })}
        </Fade>

        <Fade show={st === 6}>
          <text x={TL_X - 8} y={234} textAnchor="end" className="v-label t2" style={{ fontSize: 9 }}>monolith</text>
          <SegTrack segs={segMono} x0={TL_X} y={222} scale={TL_SCALE} maxMs={TL_MAX} />
          <text x={TL_X - 8} y={268} textAnchor="end" className="v-label t2" style={{ fontSize: 9 }}>chain</text>
          <SegTrack segs={segChain} x0={TL_X} y={256} scale={TL_SCALE} maxMs={TL_MAX} />
          {[segMono, segChain].map((sg, i) => {
            const x = TL_X + ttft(sg) * TL_SCALE;
            return (
              <g key={i}>
                <line x1={x} x2={x} y1={218 + i * 34} y2={242 + i * 34} stroke="var(--text)" strokeWidth={1.4} />
                <text x={x + 5} y={234 + i * 34} className="v-label" style={{ fontSize: 8.5, fontWeight: 700, fill: '#0e1116' }}>{`first answer token at ${ttft(sg).toFixed(0)} ms`}</text>
              </g>
            );
          })}
          {[0, 100, 200, 300].map((t) => (
            <text key={t} x={TL_X + t * TL_SCALE} y={292} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{`${t} ms`}</text>
          ))}
        </Fade>

        {/* ── notes ── */}
        <line x1={16} x2={624} y1={308} y2={308} className="v-rule faint" />
        <Label x={16} y={328} tone={1} size={9.5}>{l1}</Label>
        <Label x={16} y={348} tone={2} size={9}>{l2}</Label>
      </svg>
    </Stage>
  );
}
