import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, fmt } from '../lib/svg';
import {
  VERSIONS, ci95, passesGate, compileCost, PRICE_SETS, dollars, CANDIDATES, bestOf, OPT,
  type PriceSetId,
} from './promptops';

const STEPS = [
  'A prompt with no history',
  'Every change is a version',
  'Every version gets a score',
  'Is v4 really better? Count the noise',
  'A regression test gates the release',
  'Ship, watch, roll back',
  'Let an optimizer search',
  'The bill for searching',
];
const CAPTIONS = [
  'The prompt lives in an f-string inside the app. Someone edits it on Friday, the old text is gone, and on Monday nobody can say what changed or whether it was better.',
  'Move the prompt into its own file under version control. Now every edit is a commit with a message: seven versions, each one recoverable.',
  'Run each version on the same labelled eval set. Now every commit carries a number, and the history reads as a curve, not a story.',
  'A score on 200 examples is ±5 points at 95% confidence. v3 → v4 is one point: noise. Drag the eval-set size and watch the intervals shrink.',
  'In CI, a candidate must not score worse than production by more than a small margin. v6, shortened to save tokens, falls 7 points and is blocked before any user sees it.',
  'v7 passes offline and ships. Live, its escalation rate rises. Because production points at a version, rolling back to v5 is one line.',
  'An optimizer such as DSPy writes candidate prompts for you, here by bootstrapping few-shot demonstrations, and scores each on the dev set. More candidates, a better best.',
  'Each candidate is scored by running the whole program on every dev example. 16 candidates × 200 examples × 3 calls is 9,600 calls, plus the bootstrap runs. And the winner is longer.',
];

const DEF_N = [200, 200, 200, 200, 200, 200, 200, 200];
const VX = (i: number) => 64 + i * 86;
const BASE = 262; // score axis bottom (0.6)
const TOP = 112; // 0.95
const yS = (v: number) => BASE - ((v - 0.6) / 0.35) * (BASE - TOP);
// scatter frame for the optimizer
const SX0 = 80;
const SX1 = 560;
const xT = (t: number) => SX0 + ((t - 400) / 2000) * (SX1 - SX0);
const yD = (v: number) => 262 - ((v - 0.68) / 0.27) * 150;

const CHANGE_LINES: string[][] = [
  ['first draft'], ['explicit', 'label list'], ['+ one output', 'example'], ['friendlier', 'tone'],
  ['+ refund', 'rule'], ['shortened', 'for cost'], ['moved to a', 'new model'],
];

export default function PromptTimeline() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [n, setN] = useState(200);
  const [cands, setCands] = useState(16);
  const [price, setPrice] = useState<PriceSetId>('small');

  useEffect(() => { setN(DEF_N[st]); setCands(16); }, [st]);

  // production pointer through the story
  const gate = VERSIONS.map((v, i) => {
    if (i === 0) return { pass: true, prodBefore: 0 };
    let prod = VERSIONS[0].score;
    for (let j = 1; j < i; j++) if (passesGate(VERSIONS[j].score, prod)) prod = VERSIONS[j].score;
    return { pass: passesGate(v.score, prod), prodBefore: prod };
  });
  const prodIdx = st <= 3 ? 6 : st === 4 ? 6 : 4; // after the gate v7 ships; after the rollback v5 is live
  const prodV = VERSIONS[prodIdx];
  const best = bestOf(cands);
  const cost = compileCost(cands);
  const P = PRICE_SETS.find((p) => p.id === price)!;
  const usd = Math.round(dollars(cost.inTok, cost.outTok, P.inP, P.outP) * 100 + 1e-6) / 100;
  const evalTok = n * (prodV.tokens + 150 + 5);

  const notes: [string, string][] = [
    ['prompt = f"You are a support assistant... {ticket}" lives in three places in the codebase.', 'No history, no tests, no owner: every edit is a silent change to production behaviour.'],
    ['prompts/support_router.md + a header: id, model, temperature, owner. Seven commits, each with a reason.', 'The prompt is pinned to a model: the same text on another model is a different program.'],
    [`Same ${n} labelled tickets for every version. v1 ${(VERSIONS[0].score * 100).toFixed(0)}% → v5 ${(VERSIONS[4].score * 100).toFixed(0)}% (illustrative).`, 'Log the version with every request too, so any production answer traces back to its exact prompt.'],
    [`95% interval at n = ${fmt.int(n)}: ±${(ci95(0.85, n) * 100).toFixed(1)} points around 85%. 1.96 × √(p(1 − p) / n).`, 'Compare versions on the same examples (paired) and only trust differences larger than the noise.'],
    [`Gate: candidate ≥ production − 1 point. v6: ${(VERSIONS[5].score * 100).toFixed(0)}% vs ${(gate[5].prodBefore * 100).toFixed(0)}% → blocked.`, 'The cheapest bug is the one CI catches. Module B2 builds the eval pipeline this gate runs.'],
    ['Offline 86%, but live escalations rose 6 points. Production → v5 again, in one config change.', 'Offline evals are a sample of yesterday’s traffic. Watch live metrics after every release.'],
    [`${cands} candidates: best dev score ${(best.score * 100).toFixed(1)}% vs hand-written v5 ${(VERSIONS[4].score * 100).toFixed(0)}%, but ${fmt.int(best.tokens)} prompt tokens vs 700.`, 'The best of many noisy scores is biased upward. Confirm the winner on a held-out test set.'],
    [`${fmt.int(cost.calls)} calls, ${fmt.si(cost.inTok)} input + ${fmt.si(cost.outTok)} output tokens: $${usd.toFixed(2)} per compile at ${P.label} per M.`, `Then every production call carries ${fmt.int(best.tokens - 700)} more prompt tokens. Count both bills.`],
  ];
  const [l1, l2] = notes[st];

  return (
    <Stage
      rootRef={ref}
      title="A prompt’s history: versions, scores, gates"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['text', 'eval score (illustrative)'], ['token', 'in production'], ['error', 'blocked / regressed'], ['memory', 'prompt tokens'], ['compute', 'optimizer calls']]}
      {...s}
      readout={
        <Readout items={st >= 6 ? [
          { label: 'candidates tried', value: `${cands}`, kind: 'compute', note: `each run on ${OPT.dev} dev examples` },
          { label: 'best dev score', value: `${(best.score * 100).toFixed(1)}%`, kind: 'text', note: `±${(ci95(best.score, OPT.dev) * 100).toFixed(1)} at n = ${OPT.dev}` },
          { label: 'calls to compile', value: fmt.int(cost.calls), kind: 'compute', note: `${OPT.bootstrap} × 3 + ${cands} × ${OPT.dev} × 3` },
          { label: 'cost per compile', value: `$${usd.toFixed(2)}`, kind: 'compute', note: P.note },
        ] : [
          { label: 'in production', value: st === 0 ? '?' : prodV.id, kind: 'token', note: st === 0 ? 'nobody knows which text' : prodV.change },
          { label: 'offline score', value: st < 2 ? '—' : `${(prodV.score * 100).toFixed(0)}% ± ${(ci95(prodV.score, n) * 100).toFixed(1)}`, kind: 'text', note: `95% interval, n = ${fmt.int(n)}` },
          { label: 'tokens per eval run', value: st < 2 ? '—' : fmt.si(evalTok), kind: 'memory', note: `${fmt.int(n)} × (prompt + ticket + label)` },
        ]} />
      }
      controls={
        st >= 3 && st <= 5 ? (
          <Slider label="Eval set size (n)" value={n} min={50} max={2000} step={50} onChange={setN} format={(v) => `n = ${fmt.int(v)}`} />
        ) : st >= 6 ? (
          <>
            <Slider label="Candidate prompts the optimizer tries" value={cands} min={4} max={32} onChange={setCands} format={(v) => `${v}`} kind="compute" />
            {st === 7 && <Segmented label="Dated example price (input / output per M)" value={price} options={PRICE_SETS.map((p) => ({ value: p.id, label: p.label }))} onChange={setPrice} />}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A timeline of seven prompt versions with an eval score and confidence interval for each, a release gate that blocks a regression, a rollback, and a scatter of candidate prompts found by an automatic optimizer with their token cost.">
        {/* ── step 1: the f-string ── */}
        <Fade show={st === 0}>
          <rect x={16} y={18} width={608} height={92} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <text x={30} y={40} className="v-label t3" style={{ fontSize: 9 }}># app.py, handlers.py, batch_job.py</text>
          <text x={30} y={60} className="v-label t1" style={{ fontSize: 9.5 }}>{'prompt = f"You are a support assistant. Be friendly but brief.'}</text>
          <text x={30} y={76} className="v-label t1" style={{ fontSize: 9.5 }}>{'Classify the ticket as BILLING, BUG, ACCOUNT or OTHER... {ticket}"'}</text>
          <text x={30} y={98} className="v-label" style={{ fontSize: 9, fill: 'var(--error)' }}>edited Friday 17:40 · previous text: gone · score before or after: unknown</text>
        </Fade>

        {/* ── the version timeline ── */}
        <g className="v-fade" style={{ opacity: st === 0 ? 0.12 : st >= 6 ? 0.35 : 1 }}>
          <line x1={VX(0)} x2={VX(6)} y1={30} y2={30} stroke="var(--rule-2)" strokeWidth={1.5} />
          {VERSIONS.map((v, i) => {
            const blocked = st >= 4 && !gate[i].pass;
            const live = i === prodIdx && st >= 1;
            return (
              <g key={v.id}>
                <circle cx={VX(i)} cy={30} r={6} fill={blocked ? 'var(--error)' : live ? 'var(--token)' : 'var(--ground-3)'} stroke={blocked ? 'var(--error)' : live ? 'var(--token)' : 'var(--text-2)'} strokeWidth={1.4} className="v-cell" />
                <text x={VX(i)} y={18} textAnchor="middle" className="v-label t1" style={{ fontSize: 9, fontWeight: 700 }}>{v.id}</text>
                {CHANGE_LINES[i].map((ln, j) => (
                  <text key={j} x={VX(i)} y={50 + j * 11} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{ln}</text>
                ))}
                {live && <text x={VX(i)} y={82} textAnchor="middle" className="v-label" style={{ fontSize: 8, fontWeight: 700, fill: 'var(--token)' }}>▲ production</text>}
                {blocked && <text x={VX(i)} y={82} textAnchor="middle" className="v-label" style={{ fontSize: 8, fontWeight: 700, fill: 'var(--error)' }}>✕ blocked</text>}
              </g>
            );
          })}
        </g>

        {/* ── scores per version ── */}
        <Fade show={st >= 2 && st <= 5}>
          {[0.6, 0.7, 0.8, 0.9].map((v) => (
            <g key={v}>
              <line x1={32} x2={620} y1={yS(v)} y2={yS(v)} className="v-rule faint" />
              <text x={28} y={yS(v) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>{`${v * 100}%`}</text>
            </g>
          ))}
          {VERSIONS.map((v, i) => {
            const h = yS(0.6) - yS(v.score);
            const e = ci95(v.score, n);
            const blocked = st >= 4 && !gate[i].pass;
            const live = i === prodIdx;
            return (
              <g key={v.id}>
                <rect x={VX(i) - 16} y={yS(v.score)} width={32} height={h} rx={2} fill={blocked ? 'var(--error)' : live ? 'var(--token)' : 'var(--text-2)'} opacity={blocked || live ? 0.9 : 0.55} className="v-cell" />
                <line x1={VX(i)} x2={VX(i)} y1={yS(Math.min(0.95, v.score + e))} y2={yS(Math.max(0.6, v.score - e))} stroke="var(--text)" strokeWidth={1.2} style={{ transition: 'all .4s var(--ease-out)' }} />
                <line x1={VX(i) - 6} x2={VX(i) + 6} y1={yS(Math.min(0.95, v.score + e))} y2={yS(Math.min(0.95, v.score + e))} stroke="var(--text)" strokeWidth={1.2} style={{ transition: 'all .4s var(--ease-out)' }} />
                <line x1={VX(i) - 6} x2={VX(i) + 6} y1={yS(Math.max(0.6, v.score - e))} y2={yS(Math.max(0.6, v.score - e))} stroke="var(--text)" strokeWidth={1.2} style={{ transition: 'all .4s var(--ease-out)' }} />
                <text x={VX(i) + 20} y={yS(v.score) + 4} className="v-label t1" style={{ fontSize: 8.5 }}>{`${(v.score * 100).toFixed(0)}`}</text>
                {st === 5 && v.live !== undefined && (
                  <g>
                    <rect x={VX(i) - 16} y={yS(v.score)} width={32} height={(yS(0.6) - yS(0.6 + Math.abs(v.live)))} fill="var(--error)" opacity={0.9} />
                    <text x={VX(i)} y={yS(v.score) - 10} textAnchor="middle" className="v-label" style={{ fontSize: 8, fill: 'var(--error)', fontWeight: 700 }}>live −6</text>
                  </g>
                )}
              </g>
            );
          })}
          <Fade show={st >= 4}>
            <line x1={VX(5) - 30} x2={VX(5) + 30} y1={yS(gate[5].prodBefore - 0.01)} y2={yS(gate[5].prodBefore - 0.01)} stroke="var(--error)" strokeDasharray="3 2" />
            <text x={VX(5) + 34} y={yS(gate[5].prodBefore - 0.01) + 3} className="v-label" style={{ fontSize: 7.5, fill: 'var(--error)' }}>gate</text>
          </Fade>
          <Fade show={st === 5}>
            <path d={`M${VX(6)},${BASE + 14} C${VX(6) - 40},${BASE + 30} ${VX(4) + 40},${BASE + 30} ${VX(4)},${BASE + 14}`} fill="none" stroke="var(--token)" strokeWidth={1.4} markerEnd="url(#tps-arrow)" />
            <text x={(VX(4) + VX(6)) / 2} y={BASE + 40} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: 'var(--token)' }}>rollback: production → v5</text>
          </Fade>
        </Fade>

        {/* ── optimizer search ── */}
        <Fade show={st >= 6}>
          <Label x={SX0} y={104} tone={2} size={9}>dev score against prompt length, one dot per candidate</Label>
          {[0.7, 0.8, 0.9].map((v) => (
            <g key={v}>
              <line x1={SX0} x2={SX1} y1={yD(v)} y2={yD(v)} className="v-rule faint" />
              <text x={SX0 - 6} y={yD(v) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>{`${v * 100}%`}</text>
            </g>
          ))}
          {[400, 1200, 2000, 2400].map((t) => (
            <text key={t} x={xT(t)} y={276} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{fmt.int(t)}</text>
          ))}
          <text x={SX1} y={290} textAnchor="end" className="v-label" style={{ fontSize: 8, fill: 'var(--memory)' }}>prompt tokens per call →</text>
          {CANDIDATES.map((c, i) => (
            <circle key={i} cx={xT(c.tokens) + ((i % 5) - 2) * 4} cy={yD(c.score)} r={3.4} fill={i < cands ? (c === best ? 'var(--compute)' : 'var(--compute-dim)') : 'none'} stroke={i < cands ? 'var(--compute)' : 'var(--rule)'} strokeWidth={0.8} className="v-cell" />
          ))}
          {VERSIONS.slice(0, 5).map((v) => (
            <circle key={v.id} cx={xT(v.tokens)} cy={yD(v.score)} r={3.6} fill="var(--text)" />
          ))}
          <text x={xT(VERSIONS[4].tokens) + 8} y={yD(VERSIONS[4].score) + 3} className="v-label t1" style={{ fontSize: 8 }}>hand-written v5</text>
          <text x={xT(best.tokens) - 8} y={yD(best.score) - 6} textAnchor="end" className="v-label" style={{ fontSize: 8, fill: 'var(--compute)', fontWeight: 700 }}>{`best of ${cands}: ${(best.score * 100).toFixed(1)}%`}</text>
          <Fade show={st === 7}>
            <rect x={SX1 + 10} y={120} width={58} height={140} rx={3} fill="none" stroke="var(--rule-2)" />
            <text x={SX1 + 39} y={136} textAnchor="middle" className="v-label t2" style={{ fontSize: 8 }}>compile</text>
            <rect x={SX1 + 24} y={256 - Math.min(110, (cost.calls / 19500) * 110)} width={30} height={Math.min(110, (cost.calls / 19500) * 110)} fill="var(--compute)" style={{ transition: 'all .4s var(--ease-out)' }} />
            <text x={SX1 + 39} y={250 - Math.min(110, (cost.calls / 19500) * 110)} textAnchor="middle" className="v-label t1" style={{ fontSize: 8 }}>{fmt.si(cost.calls)}</text>
            <text x={SX1 + 39} y={272} textAnchor="middle" className="v-label t3" style={{ fontSize: 7.5 }}>calls</text>
          </Fade>
        </Fade>

        {/* ── notes ── */}
        <line x1={16} x2={624} y1={312} y2={312} className="v-rule faint" />
        <Label x={16} y={332} tone={1} size={9.5}>{l1}</Label>
        <Label x={16} y={352} tone={2} size={9}>{l2}</Label>
      </svg>
    </Stage>
  );
}
