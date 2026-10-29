import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, fmt } from '../lib/svg';
import {
  QUESTION, PATHS, tally, selfConsistency, passAtN, verifierAcc, DIRECT_ACC,
  COT_TOKENS, DIRECT_TOKENS, REASONING_TOKENS, latencyParallel, latencySequential, stepS,
} from './cot';

const STEPS = [
  'Answer immediately, and miss',
  'Show the work: few-shot chain of thought',
  'Or just ask: let’s think step by step',
  'Why it works: tokens are computation',
  'Sample many reasoning paths',
  'Majority vote: self-consistency',
  'A verifier instead of a vote',
  'The bill: tokens and time',
];
const CAPTIONS = [
  'Ask for the answer straight away and the model must commit to it in its first tokens. It writes $6: twelve divided by two. One forward pass per answer token is all the thinking it gets.',
  'Show one worked example whose answer spells out the steps, and the model imitates the format: it writes the intermediate results, then the answer. $8.',
  'No example needed: append "Let’s think step by step" and a large enough model writes its own reasoning. Zero-shot chain of thought.',
  'Every generated token is a full forward pass that can read every earlier token. A 40-token chain gives the model 40 passes of computation, each building on the last, before it commits.',
  'Sample with temperature above zero and each run reasons differently. Most paths reach $8; some slip. A single sample is a coin toss weighted by the model’s skill.',
  'Take the most common final answer across N paths. Wrong paths scatter across many answers; right ones agree. Accuracy climbs, then flattens.',
  'Or score each path with a verifier and keep the best. It can beat the vote, but search hard enough and you find a wrong path the verifier likes: path 13 here.',
  'Every path is paid in output tokens. Decoded in parallel, N paths take about as long as one, because decode is memory-bound. One after another, N times as long.',
];

const DEFAULT_N = [1, 1, 1, 1, 5, 5, 5, 5];
const SHOW = 12;
const LOGMAX = 5; // log2(32)
const CX0 = 64;
const CX1 = 560;
const CY0 = 206;
const CY1 = 296;
const xN = (n: number) => CX0 + (Math.log2(n) / LOGMAX) * (CX1 - CX0);
const yA = (a: number) => CY1 - a * (CY1 - CY0);
const yTok = (t: number) => CY1 - (t / 8000) * (CY1 - CY0);
const NS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32];

function poly(fn: (n: number) => number, y: (v: number) => number) {
  return NS.map((n) => `${xN(n).toFixed(1)},${y(fn(n)).toFixed(1)}`).join(' ');
}

export default function SelfConsistency() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [n, setN] = useState(1);
  const [pick, setPick] = useState<'vote' | 'verifier'>('vote');

  useEffect(() => {
    setN(DEFAULT_N[st]);
    setPick(st === 6 ? 'verifier' : 'vote');
  }, [st]);

  const curves = useMemo(() => ({
    sc: poly(selfConsistency, yA),
    pass: poly(passAtN, yA),
    ver: poly(verifierAcc, yA),
    tok: poly((k) => k * COT_TOKENS, yTok),
  }), []);

  const showPaths = st >= 4;
  const t = tally(n);
  const maxVotes = Math.max(...t.map((x) => x.n));
  const winners = t.filter((x) => x.n === maxVotes && x.n > 0);
  const voteWinner = winners.length === 1 ? winners[0] : null;
  const shown = PATHS.slice(0, n);
  const best = shown.reduce((b, p, i) => (p.score > shown[b].score ? i : b), 0);
  const verPick = shown[best];
  const useVer = st >= 5 && pick === 'verifier';
  const acc = st <= 0 ? DIRECT_ACC : useVer ? verifierAcc(n) : selfConsistency(n);
  const outTok = st === 0 ? DIRECT_TOKENS : n * COT_TOKENS;
  const latPar = st === 0 ? DIRECT_TOKENS * stepS(1) : latencyParallel(n);
  const latSeq = st === 0 ? latPar : latencySequential(n);

  const notes: [string, string][] = [
    ['Direct answer: 2 output tokens, decided after one forward pass each. The model says $6.', 'On a set of such problems this toy model answers 20% correctly (illustrative).'],
    ['The example taught a format with the steps written out. The model copies it: 4 groups × $2 = $8.', 'Wei et al. (2022): eight such exemplars took a 540B model from about 18% to 57% on GSM8K.'],
    ['Kojima et al. (2022): one sentence, no examples. 17.7% → 78.7% on MultiArith for an InstructGPT model.', 'A second prompt then extracts the final answer from the reasoning.'],
    ['Direct: 2 passes deep. Chain of thought: 40 passes deep, each pass reading every step before it.', 'Only content helps: in ablations, filler tokens of the same length did not (Wei et al.).'],
    [`${n} path${n > 1 ? 's' : ''} at temperature 0.7: ${shown.filter((p) => p.right).length} reach $8, ${shown.filter((p) => !p.right).length} slip. Drag N.`, 'One sample has a 56% chance of being right on the toy set: chain of thought, but still a gamble.'],
    [`Vote with N = ${n}: ${voteWinner ? `${voteWinner.a} wins ${voteWinner.n} of ${n}` : 'a tie, broken at random'}. Toy-set accuracy ${(selfConsistency(n) * 100).toFixed(1)}%.`, 'Self-consistency (Wang et al., 2022). Gains come fast up to about 10 paths, then flatten.'],
    [`Verifier picks path ${best + 1} (${verPick.answer}, score ${verPick.score.toFixed(2)}). Toy-set accuracy ${(verifierAcc(n) * 100).toFixed(1)}%.`, 'Cobbe et al. (2021): best-of-100 with a 6B verifier slightly beat a finetuned 175B model on GSM8K.'],
    [`N = ${n}: ${fmt.int(n * COT_TOKENS)} output tokens. In parallel ${latencyParallel(n).toFixed(2)} s; one after another ${latencySequential(n).toFixed(1)} s.`, `A reasoning model writing ${fmt.int(REASONING_TOKENS)} thinking tokens takes ${(REASONING_TOKENS * stepS(1)).toFixed(0)} s at the same 7.5 ms per token.`],
  ];
  const [l1, l2] = notes[st];

  const textLines: { t: string; tone: 'dim' | 'q' | 'ok' | 'bad' | 'note' }[][] = [
    [
      { t: `Q: ${QUESTION}`, tone: 'q' },
      { t: 'A: $6', tone: 'bad' },
    ],
    [
      { t: 'Q: Tom has 5 apples and buys 2 bags of 4 apples. How many apples does he have?', tone: 'dim' },
      { t: 'A: 2 bags × 4 = 8 apples. 5 + 8 = 13. The answer is 13.', tone: 'dim' },
      { t: `Q: ${QUESTION}`, tone: 'q' },
      { t: 'A: 12 pens is 12 ÷ 3 = 4 groups of 3 pens. Each group costs $2,', tone: 'ok' },
      { t: '   so 4 × $2 = $8. The answer is $8.', tone: 'ok' },
    ],
    [
      { t: `Q: ${QUESTION}`, tone: 'q' },
      { t: 'A: Let’s think step by step.', tone: 'q' },
      { t: '12 pens make 12 ÷ 3 = 4 sets of 3 pens. Each set costs $2,', tone: 'ok' },
      { t: 'so the total is 4 × $2 = $8.', tone: 'ok' },
      { t: 'Therefore, the answer is $8.', tone: 'ok' },
    ],
  ];

  return (
    <Stage
      rootRef={ref}
      title="Thinking out loud, many times over"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['token', 'generated reasoning'], ['error', 'wrong answer'], ['compute', 'forward passes'], ['text', 'accuracy (toy model)']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'paths', value: st < 4 ? '1' : `${n}`, kind: 'text', note: st === 0 ? 'direct answer' : st < 4 ? 'one chain of thought' : useVer ? 'best by verifier' : 'majority vote' },
          { label: 'accuracy', value: `${(acc * 100).toFixed(1)}%`, kind: 'text', note: 'illustrative toy question set' },
          { label: 'output tokens / question', value: fmt.int(st >= 1 && st < 4 ? COT_TOKENS : outTok), kind: 'token', note: st === 0 ? 'just the answer' : `${COT_TOKENS} per path` },
          { label: 'latency (8B, H100)', value: `${(st >= 1 && st < 4 ? latencyParallel(1) : latPar).toFixed(2)} s`, kind: 'compute', note: n > 1 && st >= 4 ? `parallel · ${latSeq.toFixed(1)} s one by one` : `${(stepS(1) * 1e3).toFixed(1)} ms per token` },
        ]} />
      }
      controls={
        st >= 4 ? (
          <>
            <Slider label="Sampled paths (N)" value={n} min={1} max={32} onChange={setN} format={(v) => `N = ${v}`} kind="token" />
            {st >= 5 && st <= 6 && (
              <Segmented label="Choose the answer by" value={pick} options={[{ value: 'vote' as const, label: 'majority vote' }, { value: 'verifier' as const, label: 'verifier score' }]} onChange={setPick} />
            )}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A word problem answered directly (wrong), then with a written chain of thought (right), then by sampling many reasoning paths whose final answers are tallied by majority vote or scored by a verifier, with toy accuracy and token-cost curves against the number of paths.">
        {/* ── steps 1–3: the prompt and the answer, as text ── */}
        <Fade show={st <= 2}>
          <rect x={16} y={14} width={608} height={st === 0 ? 70 : 118} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" className="v-fade" />
          {textLines[Math.min(st, 2)].map((ln, i) => (
            <text key={`${st}-${i}`} x={30} y={36 + i * 19} className="v-label" style={{
              fontSize: 9.5,
              fill: ln.tone === 'bad' ? 'var(--error)' : ln.tone === 'ok' ? 'var(--token)' : ln.tone === 'dim' ? 'var(--text-3)' : 'var(--text)',
              fontWeight: ln.tone === 'bad' || ln.tone === 'ok' ? 600 : 400,
            }}>{ln.t}</text>
          ))}
          {st === 1 && <text x={612} y={124} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>grey: the exemplar you wrote · green: the model’s reply</text>}
        </Fade>

        {/* ── step 4: tokens are computation ── */}
        <Fade show={st === 3}>
          <Label x={16} y={22} tone={2} size={10} bold>serial computation before the answer is committed</Label>
          {[{ label: 'direct', cols: 2, y: 36 }, { label: 'chain of thought', cols: 40, y: 96 }].map((row) => (
            <g key={row.label}>
              <text x={16} y={row.y + 26} className="v-label t2" style={{ fontSize: 9 }}>{row.label}</text>
              {Array.from({ length: row.cols }, (_, c) => (
                <g key={c}>
                  {Array.from({ length: 8 }, (_, r) => (
                    <rect key={r} x={128 + c * 12} y={row.y + r * 6} width={10} height={5} rx={1} fill="var(--compute)" opacity={0.35 + 0.65 * (r / 7)} />
                  ))}
                  <rect x={128 + c * 12} y={row.y + 50} width={10} height={6} rx={1} fill={row.cols === 2 ? 'var(--error)' : 'var(--token)'} />
                </g>
              ))}
              <text x={128 + row.cols * 12 + 6} y={row.y + 26} className="v-label t1" style={{ fontSize: 9 }}>{row.cols === 2 ? '2 passes × 32 layers' : ''}</text>
            </g>
          ))}
          <text x={612} y={168} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>each column: one generated token = one full forward pass (8 of 32 layers drawn)</text>
          <text x={612} y={182} textAnchor="end" className="v-label t1" style={{ fontSize: 9 }}>40 passes × 32 layers, each one able to read every step before it</text>
        </Fade>

        {/* ── steps 5–8: sampled paths ── */}
        <Fade show={showPaths}>
          <Label x={16} y={16} tone={2} size={9.5}>{QUESTION}</Label>
          {PATHS.slice(0, SHOW).map((p, i) => {
            const on = i < n;
            const y = 26 + i * 13.2;
            const w = (p.tokens / 330) * 300;
            const chosen = st >= 5 && on && (useVer ? i === best : voteWinner?.a === p.answer);
            return (
              <g key={i} className="v-fade" style={{ opacity: on ? 1 : 0.12 }}>
                <text x={58} y={y + 8.5} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>{`path ${i + 1}`}</text>
                <rect x={64} y={y} width={w} height={9.5} rx={1.5} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={0.6} />
                <rect x={64 + w + 4} y={y - 0.5} width={32} height={10.5} rx={2} fill={p.right ? 'var(--token)' : 'var(--error)'} stroke={chosen ? 'var(--text)' : 'none'} strokeWidth={1.4} />
                <text x={64 + w + 20} y={y + 7.8} textAnchor="middle" className="v-label" style={{ fontSize: 8, fontWeight: 700, fill: '#0e1116' }}>{p.answer}</text>
                {useVer && on && (
                  <text x={64 + w + 42} y={y + 8} className="v-label" style={{ fontSize: 7.5, fill: i === best ? 'var(--text)' : 'var(--text-3)' }}>{p.score.toFixed(2)}</text>
                )}
              </g>
            );
          })}
          {n > SHOW && <text x={64} y={26 + SHOW * 13.2 + 8} className="v-label t3" style={{ fontSize: 8.5 }}>{`+ ${n - SHOW} more paths in the tally`}</text>}

          {/* tally */}
          <Fade show={st >= 5 && !useVer}>
            <Label x={470} y={30} tone={2} size={9} bold>{`votes over ${n} path${n > 1 ? 's' : ''}`}</Label>
            {t.map((x, i) => {
              const y = 40 + i * 22;
              const win = voteWinner?.a === x.a;
              return (
                <g key={x.a}>
                  <text x={500} y={y + 11} textAnchor="end" className="v-label" style={{ fontSize: 9, fill: x.right ? 'var(--token)' : 'var(--error)' }}>{x.a}</text>
                  <rect x={506} y={y + 2} width={(x.n / Math.max(n, 8)) * 96} height={12} rx={1.5} fill={x.right ? 'var(--token)' : 'var(--error)'} stroke={win ? 'var(--text)' : 'none'} strokeWidth={1.3} style={{ transition: 'width .4s var(--ease-out)' }} />
                  <text x={510 + (x.n / Math.max(n, 8)) * 96 + 2} y={y + 11.5} className="v-label t1" style={{ fontSize: 8.5 }}>{x.n}</text>
                </g>
              );
            })}
          </Fade>
          <Fade show={useVer}>
            <Label x={470} y={30} tone={2} size={9} bold>verifier keeps the top score</Label>
            <text x={470} y={52} className="v-label t1" style={{ fontSize: 9 }}>{`path ${best + 1}: ${verPick.answer}`}</text>
            <text x={470} y={68} className="v-label" style={{ fontSize: 9, fill: verPick.right ? 'var(--token)' : 'var(--error)' }}>{verPick.right ? 'right' : 'wrong, but scored highest'}</text>
            <text x={470} y={92} className="v-label t3" style={{ fontSize: 8.5 }}>try N ≥ 13</text>
          </Fade>
        </Fade>

        {/* ── chart: accuracy vs N ── */}
        <Fade show={st >= 4 && st <= 6}>
          <line x1={16} x2={624} y1={192} y2={192} className="v-rule faint" />
          {[0, 0.5, 1].map((a) => (
            <g key={a}>
              <line x1={CX0} x2={CX1} y1={yA(a)} y2={yA(a)} className="v-rule faint" />
              <text x={CX0 - 6} y={yA(a) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>{`${a * 100}%`}</text>
            </g>
          ))}
          {[1, 2, 4, 8, 16, 32].map((k) => (
            <text key={k} x={xN(k)} y={CY1 + 12} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{k}</text>
          ))}
          <text x={CX1} y={CY1 + 12} dx={18} className="v-label t3" style={{ fontSize: 8 }}>N</text>
          <line x1={CX0} x2={CX1} y1={yA(DIRECT_ACC)} y2={yA(DIRECT_ACC)} stroke="var(--error)" strokeDasharray="3 3" />
          <text x={CX1 + 4} y={yA(DIRECT_ACC) + 3} className="v-label" style={{ fontSize: 8, fill: 'var(--error)' }}>direct</text>
          <polyline points={curves.tok} fill="none" stroke="var(--token)" strokeWidth={1.2} strokeDasharray="5 3" />
          <text x={xN(16) - 6} y={yTok(16 * COT_TOKENS) - 4} textAnchor="end" className="v-label" style={{ fontSize: 8, fill: 'var(--token)' }}>output tokens, 250 × N</text>
          <polyline points={curves.pass} fill="none" stroke="var(--text-3)" strokeWidth={1.1} strokeDasharray="2 3" />
          <text x={xN(3)} y={yA(passAtN(3)) - 5} className="v-label t3" style={{ fontSize: 8 }}>at least one path right</text>
          <polyline points={curves.sc} fill="none" stroke="var(--text)" strokeWidth={useVer ? 1.2 : 2} style={{ opacity: useVer ? 0.6 : 1 }} />
          <text x={CX1 + 4} y={yA(selfConsistency(32)) + 8} className="v-label t1" style={{ fontSize: 8 }}>vote</text>
          <g className="v-fade" style={{ opacity: st >= 6 || useVer ? 1 : 0 }}>
            <polyline points={curves.ver} fill="none" stroke="var(--text-2)" strokeWidth={useVer ? 2 : 1.2} strokeDasharray="7 3" />
            <text x={CX1 + 4} y={yA(verifierAcc(32)) + 3} className="v-label t2" style={{ fontSize: 8 }}>verifier</text>
          </g>
          <line x1={xN(n)} x2={xN(n)} y1={CY0 - 4} y2={CY1} stroke="var(--text-3)" strokeDasharray="2 3" />
          <circle cx={xN(n)} cy={yA(acc)} r={4} fill="var(--text)" stroke="var(--ground-2)" strokeWidth={1.5} style={{ transition: 'cx .4s var(--ease-out), cy .4s var(--ease-out)' }} />
        </Fade>

        {/* ── steps 1–4 bottom: what it costs per answer ── */}
        <Fade show={st <= 3}>
          <line x1={16} x2={624} y1={192} y2={192} className="v-rule faint" />
          <Label x={16} y={214} tone={2} size={9.5}>output tokens per answer, and decode time at 7.5 ms per token (8B on an H100)</Label>
          {[
            { label: 'direct', tok: DIRECT_TOKENS, kind: 'error' },
            { label: 'chain of thought', tok: COT_TOKENS, kind: 'token' },
          ].map((b, i) => (
            <g key={b.label}>
              <text x={130} y={240 + i * 26} textAnchor="end" className="v-label t2" style={{ fontSize: 9 }}>{b.label}</text>
              <rect x={136} y={230 + i * 26} width={Math.max(2, (b.tok / 260) * 400)} height={14} rx={2} fill={b.kind === 'error' ? 'var(--error)' : 'var(--token)'} opacity={st === 0 && i === 1 ? 0.25 : 1} />
              <text x={136 + Math.max(2, (b.tok / 260) * 400) + 6} y={241 + i * 26} className="v-label t1" style={{ fontSize: 9 }}>{`${b.tok} tokens · ${(b.tok * stepS(1) * 1e3).toFixed(0)} ms`}</text>
            </g>
          ))}
          <text x={136} y={292} className="v-label t3" style={{ fontSize: 8.5 }}>50× the output tokens: the price of thinking out loud is paid in decode steps, the slow kind</text>
        </Fade>

        {/* ── step 8: the bill ── */}
        <Fade show={st === 7}>
          <line x1={16} x2={624} y1={192} y2={192} className="v-rule faint" />
          <Label x={16} y={212} tone={2} size={9.5}>output tokens and latency per question (8B on an H100, batch decoding for parallel paths)</Label>
          {[
            { label: 'direct', tok: DIRECT_TOKENS, s: DIRECT_TOKENS * stepS(1) },
            { label: 'chain of thought', tok: COT_TOKENS, s: latencyParallel(1) },
            { label: `self-consistency ×${n}`, tok: n * COT_TOKENS, s: latencyParallel(n) },
            { label: 'reasoning model', tok: REASONING_TOKENS, s: REASONING_TOKENS * stepS(1) },
          ].map((b, i) => {
            const w = Math.max(2, (b.tok / 8000) * 300);
            return (
              <g key={b.label}>
                <text x={136} y={234 + i * 19} textAnchor="end" className="v-label t2" style={{ fontSize: 9 }}>{b.label}</text>
                <rect x={142} y={225 + i * 19} width={w} height={12} rx={2} fill="var(--token)" style={{ transition: 'width .4s var(--ease-out)' }} />
                <text x={142 + w + 6} y={234 + i * 19} className="v-label t1" style={{ fontSize: 8.5 }}>{`${fmt.int(b.tok)} tok · ${b.s < 1 ? `${(b.s * 1e3).toFixed(0)} ms` : `${b.s.toFixed(1)} s`}`}</text>
              </g>
            );
          })}
        </Fade>

        {/* ── notes ── */}
        <line x1={16} x2={624} y1={312} y2={312} className="v-rule faint" />
        <Label x={16} y={332} tone={1} size={9.5}>{l1}</Label>
        <Label x={16} y={352} tone={2} size={9}>{l2}</Label>
      </svg>
    </Stage>
  );
}
