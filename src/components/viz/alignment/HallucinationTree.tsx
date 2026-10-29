import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade } from '../lib/svg';

// One question, two branches. 10,403 = 101 × 103, so the right answer is "No".
// Conditional probabilities are illustrative; the path products are computed from them.
const P_YES = 0.55;
const P_NO = 0.45;
const YES_CHAIN: [string, number][] = [['it is prime', 0.92], ['not divisible by 2,3,5,7', 0.84], ['no factors.', 0.88]];
const NO_CHAIN: [string, number][] = [['10,403 =', 0.71], ['101 × 103', 0.66], ['.', 0.97]];
const prod = (first: number, chain: [string, number][], upto: number) => chain.slice(0, upto).reduce((a, [, p]) => a * p, first);

const STEPS = [
  'A question the model is unsure about',
  'The first token is a commitment',
  'Everything after conditions on it',
  'The branch it did not take',
  'Fluent is not the same as true',
  'Asked separately, it knows',
  'Guessing beats abstaining, if you score 0/1',
  'What actually helps',
];
const CAPTIONS = [
  'Is 10,403 prime? It is not: 101 × 103. But the model has to say Yes or No first, before any working, and it is close to a coin flip: 55% Yes, 45% No.',
  'Sampling picks “Yes”. One token, chosen with 55% probability, and the answer is already wrong. Nothing after it can take it back.',
  'Every later token is predicted given “Yes”. The likeliest continuations are fluent justifications of primality, each step 84–92% likely. The error snowballs.',
  'Had it sampled “No”, the same model would have written the factorisation. Both branches are fluent; only one is true. The model had the knowledge all along.',
  'Pretraining rewards plausible continuations, and a confident justification after “Yes” is very plausible text. Nothing in the loss checks it against the world.',
  'Ask the same model, in a fresh context, whether 10,403 is divisible by 101, and it says yes. Models often recognise their own snowballed claims as wrong when asked separately.',
  'Under 0/1 grading, a guess with any chance of being right beats “I don’t know”. Penalise wrong answers by t/(1−t) and answering pays only above confidence t.',
  'Retrieval gives the model the facts; training to abstain makes “I don’t know” a valid answer; verification catches errors; sampling settings change how branches are picked.',
];

const ROOT = { x: 26, y: 150 };
const FIRST_X = 96;
const GAP = 40;
const chipW = (t: string) => Math.max(30, t.length * 6.1 + 14);
/** x of each chip on a branch: the first-token chip, then the chain */
const chainX = (first: string, chain: [string, number][]) => {
  const xs = [FIRST_X];
  let x = FIRST_X + chipW(first) + GAP;
  for (const [t] of chain) { xs.push(x); x += chipW(t) + GAP; }
  return xs;
};
const YES_X = chainX('Yes', YES_CHAIN);
const NO_X = chainX('No', NO_CHAIN);

export default function HallucinationTree() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [conf, setConf] = useState(0.3);
  const [thr, setThr] = useState(0.75);
  useEffect(() => { setConf(0.3); setThr(0.75); }, [st]);

  const showTree = st <= 5;
  const yesShown = st >= 2 ? YES_CHAIN.length : 0;
  const noShown = st >= 3 ? NO_CHAIN.length : 0;
  const sampled = st >= 1;
  const showScore = st === 6;
  const showFix = st === 7;

  const pen = thr / (1 - thr);
  const eBinary = conf;
  const ePen = conf - (1 - conf) * pen;

  const yesY = 92;
  const noY = 212;
  const chip = (x: number, y: number, text: string, tone: 'wrong' | 'right' | 'plain', dim = false) => {
    const w = chipW(text);
    const col = tone === 'wrong' ? 'var(--error)' : tone === 'right' ? 'var(--token)' : 'var(--rule-2)';
    return (
      <g style={{ opacity: dim ? 0.35 : 1, transition: 'opacity .5s' }}>
        <rect x={x} y={y - 12} width={w} height={24} rx={3} fill={tone === 'plain' ? 'var(--ground-3)' : tone === 'wrong' ? 'var(--error-dim)' : 'var(--token-dim)'} stroke={col} />
        <text x={x + w / 2} y={y + 4} textAnchor="middle" className="v-label" style={{ fontSize: 10, fill: 'var(--text)', fontWeight: 600 }}>{text}</text>
      </g>
    );
  };

  return (
    <Stage
      rootRef={ref}
      title="How one early token commits a whole answer"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['error', 'false claim'], ['token', 'true claim'], ['compute', 'probability']]}
      {...s}
      readout={
        showTree ? (
          <Readout items={[
            { label: 'P(first token = Yes)', value: `${(P_YES * 100).toFixed(0)}%`, kind: 'error', note: 'the wrong answer, by a little' },
            { label: 'P(this wrong path)', value: st >= 2 ? prod(P_YES, YES_CHAIN, 3).toFixed(3) : '—', kind: 'error', note: '0.55 × 0.92 × 0.84 × 0.88' },
            { label: 'P(this right path)', value: st >= 3 ? prod(P_NO, NO_CHAIN, 3).toFixed(3) : '—', kind: 'token', note: '0.45 × 0.71 × 0.66 × 0.97' },
          ]} />
        ) : showScore ? (
          <Readout items={[
            { label: 'answer, 0/1 grading', value: `+${eBinary.toFixed(2)}`, kind: 'token', note: 'expected score vs 0 for “I don’t know”' },
            { label: `answer, penalty ${pen.toFixed(1)}`, value: `${ePen >= 0 ? '+' : ''}${ePen.toFixed(2)}`, kind: ePen >= 0 ? 'token' : 'error', note: `p − (1 − p) · t/(1 − t)` },
            { label: 'best choice', value: ePen > 0 ? 'answer' : 'abstain', kind: ePen > 0 ? 'token' : 'text', note: `answer only if p > t = ${thr.toFixed(2)}` },
          ]} />
        ) : undefined
      }
      controls={
        showScore ? (
          <>
            <Slider label="Model’s chance of being right, p" value={conf} min={0.05} max={0.99} step={0.01} onChange={setConf} format={(v) => v.toFixed(2)} kind="token" />
            <Slider label="Confidence target t (wrong answers cost t/(1−t))" value={thr} min={0.05} max={0.95} step={0.05} onChange={setThr} format={(v) => v.toFixed(2)} kind="error" />
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="A probability tree for the question whether 10,403 is prime. The first token Yes, chosen with 55% probability, leads to a fluent but false justification; the No branch leads to the factorisation 101 times 103. Later panels show that the model recognises the error when asked separately, how 0/1 grading rewards guessing over abstaining, and four mitigations.">
        {/* ── the tree ── */}
        <Fade show={showTree}>
          <g>
            <Label x={24} y={20} tone={2} size={10} mono={false}>{'User: Is 10,403 a prime number? Answer, then explain.'}</Label>
            <circle cx={ROOT.x + 6} cy={ROOT.y} r={6} fill="var(--text-2)" />
            {/* first-token edges */}
            {[{ y: yesY, p: P_YES, t: 'Yes', tone: 'wrong' as const }, { y: noY, p: P_NO, t: 'No', tone: 'right' as const }].map((b) => {
              const dim = sampled && b.t === 'No' && st < 3;
              return (
                <g key={b.t}>
                  <path d={`M${ROOT.x + 12},${ROOT.y} C${ROOT.x + 40},${ROOT.y} ${FIRST_X - 30},${b.y} ${FIRST_X - 2},${b.y}`} fill="none"
                    stroke={sampled && b.t === 'Yes' ? 'var(--error)' : 'var(--rule-2)'} strokeWidth={1 + b.p * 5} style={{ opacity: dim ? 0.35 : 1, transition: 'stroke .4s, opacity .4s' }} />
                  <text x={ROOT.x + 36} y={b.y < ROOT.y ? (ROOT.y + b.y) / 2 - 6 : (ROOT.y + b.y) / 2 + 14} className="v-label" style={{ fontSize: 9.5, fill: 'var(--compute)' }}>{b.p.toFixed(2)}</text>
                  {chip(FIRST_X, b.y, b.t, st >= 4 || (sampled && b.t === 'Yes') || st >= 3 ? b.tone : 'plain', dim)}
                </g>
              );
            })}
            {sampled && <Label x={FIRST_X - 4} y={yesY - 22} size={9} kind="error">sampled</Label>}

            {/* chains */}
            {[{ chain: YES_CHAIN, xs: YES_X, first: 'Yes', y: yesY, shown: yesShown, tone: 'wrong' as const }, { chain: NO_CHAIN, xs: NO_X, first: 'No', y: noY, shown: noShown, tone: 'right' as const }].map((br) => (
              <g key={br.y}>
                {br.chain.map(([t, p], i) => {
                  const on = i < br.shown;
                  const x0 = br.xs[i] + chipW(i === 0 ? br.first : br.chain[i - 1][0]) + 2;
                  const x1 = br.xs[i + 1] - 2;
                  return (
                    <g key={t} style={{ opacity: on ? 1 : 0, transition: `opacity .5s ${i * 0.35}s` }}>
                      <line x1={x0} x2={x1} y1={br.y} y2={br.y} stroke={br.tone === 'wrong' ? 'var(--error)' : 'var(--token)'} strokeWidth={1 + p * 3} />
                      <text x={(x0 + x1) / 2} y={br.y - 16} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--compute)' }}>{p.toFixed(2)}</text>
                      {chip(br.xs[i + 1], br.y, t, br.tone)}
                    </g>
                  );
                })}
              </g>
            ))}
            <Fade show={st >= 2}>
              <Label x={YES_X[1]} y={yesY + 32} size={9} kind="error">{`path probability ${prod(P_YES, YES_CHAIN, 3).toFixed(3)}: fluent, confident, false`}</Label>
            </Fade>
            <Fade show={st >= 3}>
              <Label x={NO_X[1]} y={noY + 32} size={9} kind="token">{`path probability ${prod(P_NO, NO_CHAIN, 3).toFixed(3)}: the same model, the true answer`}</Label>
            </Fade>

            {/* separately */}
            <Fade show={st === 5}>
              <g>
                <rect x={330} y={128} width={286} height={52} rx={4} fill="var(--ground-3)" stroke="var(--token)" />
                <Label x={342} y={146} tone={2} size={9}>{'new context: Is 10,403 divisible by 101?'}</Label>
                <Label x={342} y={166} size={10} kind="token">{'Yes: 101 × 103 = 10,403.'}</Label>
              </g>
            </Fade>

            <Label x={24} y={274} tone={1} size={10} bold>
              {st === 0 ? 'the answer token comes first, before any reasoning'
                : st === 1 ? 'one draw, 55% likely, and the answer is already wrong'
                : st === 2 ? 'each later token is likely given the earlier ones: the error snowballs'
                : st === 3 ? 'both branches are fluent; the model holds the knowledge for the true one'
                : st === 4 ? 'the training objective scores plausibility, not truth'
                : 'it can spot the error it just made, when it is not committed to it'}
            </Label>
            <Label x={24} y={292} tone={2} size={9}>
              {st === 0 ? 'The format asks for Yes or No up front. Whatever comes out first, the explanation must then fit it.'
                : st === 1 ? 'Greedy decoding would pick “Yes” every time here: 0.55 beats 0.45. Lower temperature makes it worse, not better.'
                : st === 2 ? 'Given “Yes”, a justification of primality is the most plausible text. The model is consistent, not correct.'
                : st === 3 ? 'Right path 0.205, wrong path 0.374. An early coin flip, not ignorance, decided the outcome.'
                : st === 4 ? 'Cross-entropy rewards text that looks like the training data. Confident explanations look like the training data.'
                : 'Zhang et al. (2023) found ChatGPT and GPT-4 recognised 67% and 87% of their own snowballed false claims.'}
            </Label>
            <Label x={24} y={308} tone={3} size={9}>
              {st <= 2 ? 'Probabilities are illustrative; the products are exact.' : st === 5 ? 'Which is why “reason first, answer last” and separate verification passes help.' : 'This is a hallucination: fluent, confident text that is not true.'}
            </Label>
          </g>
        </Fade>

        {/* ── scoring ── */}
        <Fade show={showScore}>
          <g>
            <Label x={24} y={22} tone={1} size={10} bold>{'expected score of answering, against 0 for “I don’t know”'}</Label>
            {(() => {
              const X0 = 70; const W = 300; const Y0 = 210; const H = 160;
              const xs = (p: number) => X0 + p * W;
              const ys = (v: number) => Y0 - ((Math.max(-3, Math.min(1, v)) + 3) / 4) * H;
              const lineP = (f: (p: number) => number) => {
                let d = '';
                for (let i = 0; i <= 60; i++) { const p = 0.01 + 0.98 * i / 60; d += `${i ? 'L' : 'M'}${xs(p).toFixed(1)},${ys(f(p)).toFixed(1)} `; }
                return d;
              };
              return (
                <g>
                  <line x1={X0} x2={X0 + W} y1={ys(0)} y2={ys(0)} className="v-rule" />
                  <line x1={X0} x2={X0} y1={ys(1)} y2={ys(-3)} className="v-rule" />
                  {[1, 0, -1, -2, -3].map((v) => <text key={v} x={X0 - 6} y={ys(v) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>{v}</text>)}
                  {[0, 0.5, 1].map((p) => <text key={p} x={xs(p)} y={Y0 + 14} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{p}</text>)}
                  <text x={X0 + W} y={Y0 + 26} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>chance of being right, p</text>
                  <path d={lineP((p) => p)} fill="none" stroke="var(--token)" strokeWidth={1.6} />
                  <path d={lineP((p) => p - (1 - p) * pen)} fill="none" stroke="var(--error)" strokeWidth={1.6} />
                  <text x={xs(1) - 2} y={ys(1) - 6} textAnchor="end" className="v-label" style={{ fontSize: 9, fill: 'var(--token)' }}>0/1 grading: p</text>
                  <text x={xs(0.5)} y={ys(-2.7)} className="v-label" style={{ fontSize: 9, fill: 'var(--error)' }}>{`with penalty t/(1−t) = ${pen.toFixed(2)}`}</text>
                  <line x1={xs(thr)} x2={xs(thr)} y1={ys(1)} y2={ys(-3)} stroke="var(--text-3)" strokeDasharray="3 3" />
                  <text x={xs(thr) + 4} y={ys(-0.4)} className="v-label t3" style={{ fontSize: 8.5 }}>{`t = ${thr.toFixed(2)}`}</text>
                  <circle cx={xs(conf)} cy={ys(conf)} r={4} fill="var(--token)" />
                  <circle cx={xs(conf)} cy={ys(conf - (1 - conf) * pen)} r={4} fill="var(--error)" />
                </g>
              );
            })()}
            <Label x={400} y={60} tone={1} size={9.5} bold>under 0/1 grading</Label>
            <Label x={400} y={76} tone={2} size={9}>{`answer: ${conf.toFixed(2)} expected`}</Label>
            <Label x={400} y={90} tone={2} size={9}>{'abstain: 0. Guessing always wins.'}</Label>
            <Label x={400} y={120} tone={1} size={9.5} bold>{`with a confidence target t = ${thr.toFixed(2)}`}</Label>
            <Label x={400} y={136} tone={2} size={9}>{`right +1, wrong −${pen.toFixed(2)}, abstain 0`}</Label>
            <Label x={400} y={150} tone={2} size={9}>{`answer: ${ePen >= 0 ? '+' : ''}${ePen.toFixed(2)} expected`}</Label>
            <Label x={400} y={164} tone={2} size={9}>{`→ ${ePen > 0 ? 'answer' : 'say “I don’t know”'}`}</Label>
            <Label x={24} y={262} tone={1} size={10} bold>benchmarks and graders that score 0/1 train and select for guessing</Label>
            <Label x={24} y={280} tone={2} size={9}>{'Kalai et al. (2025): state the target in the prompt and score it, so a wrong answer costs t/(1−t).'}</Label>
            <Label x={24} y={296} tone={2} size={9}>{'Then answering pays exactly when p > t, and an honest “I don’t know” is the best move below it.'}</Label>
            <Label x={24} y={312} tone={3} size={9}>{'At p = t the two options tie: p − (1 − p)·t/(1 − t) = t − t = 0.'}</Label>
          </g>
        </Fade>

        {/* ── mitigations ── */}
        <Fade show={showFix}>
          <g>
            {[
              { t: 'Retrieval', d: ['put the facts in the prompt', '(Module B3: RAG)'], k: 'var(--memory)' },
              { t: 'Abstention', d: ['train and score “I don’t know”', 'as a valid answer'], k: 'var(--token)' },
              { t: 'Verification', d: ['check claims in a separate', 'pass, with tools or sources'], k: 'var(--compute)' },
              { t: 'Sampling', d: ['reason before answering;', 'disagreement = uncertainty'], k: 'var(--text-2)' },
            ].map((m, i) => {
              const x = 24 + (i % 2) * 300;
              const y = 30 + Math.floor(i / 2) * 104;
              return (
                <g key={m.t}>
                  <rect x={x} y={y} width={290} height={90} rx={5} fill="var(--ground-3)" stroke={m.k} />
                  <text x={x + 14} y={y + 26} className="v-label" style={{ fontSize: 11, fill: 'var(--text)', fontWeight: 700 }}>{m.t}</text>
                  {m.d.map((l, j) => <text key={j} x={x + 14} y={y + 50 + j * 16} className="v-label t2" style={{ fontSize: 9.5 }}>{l}</text>)}
                </g>
              );
            })}
            <Label x={24} y={262} tone={1} size={10} bold>none of these makes a model know more; each makes it guess less, or get caught</Label>
            <Label x={24} y={280} tone={2} size={9}>{'Lower temperature only changes which branch is picked. Here greedy picks the wrong one every time.'}</Label>
            <Label x={24} y={296} tone={3} size={9}>{'The cheapest fix is often the prompt: ask for the working first and the verdict last.'}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
