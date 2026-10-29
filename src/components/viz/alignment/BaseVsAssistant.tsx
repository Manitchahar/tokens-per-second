import { useEffect, useRef, useState } from 'react';
import { Stage, Segmented, Slider, Readout } from '../lib/Stage';
import { useStep, prefersReducedMotion } from '../lib/useStep';
import { Label, Fade } from '../lib/svg';

// The same question, sent three ways. Probabilities are illustrative, shaped like what a small
// base model (Qwen2.5-0.5B) and its instruct sibling actually produce; the lesson shows how to
// print the real ones.
type Scen = 'plain' | 'qa' | 'chat';
type Lead = 'answer' | 'ramble';
interface Piece { t: string; k: 'answer' | 'ramble' | 'stop' | 'meta' }

const SCEN: Record<Scen, {
  name: string; model: string; prompt: [string, boolean][][]; bars: [string, number, Lead][]; other: number;
  cont: Piece[][]; tokensToStop: number | null; usefulTokens: number;
}> = {
  plain: {
    name: 'base · plain', model: 'base model · pretraining only',
    prompt: [[['What is the capital of France?', false]]],
    bars: [['↵', 0.40, 'ramble'], [' The', 0.12, 'answer'], [' What', 0.12, 'ramble'], [' Paris', 0.10, 'answer'], [' A', 0.06, 'ramble'], [' It', 0.04, 'answer']],
    other: 0.16,
    cont: [
      [{ t: '↵What is the capital of Germany?', k: 'ramble' }],
      [{ t: '↵What is the capital of Italy?', k: 'ramble' }],
      [{ t: '↵What is the capital of Spain?', k: 'ramble' }],
      [{ t: '↵What is the capital of Portugal?', k: 'ramble' }],
      [{ t: '… and on, until max_new_tokens', k: 'meta' }],
    ],
    tokensToStop: null, usefulTokens: 0,
  },
  qa: {
    name: 'base · Q/A', model: 'base model · pretraining only',
    prompt: [[['Q: What is the capital of France?', false]], [['A:', false]]],
    bars: [[' Paris', 0.71, 'answer'], [' The', 0.17, 'answer'], [' France', 0.03, 'answer'], ['↵', 0.02, 'ramble'], [' It', 0.02, 'answer'], [' (', 0.01, 'ramble']],
    other: 0.04,
    cont: [
      [{ t: ' Paris', k: 'answer' }],
      [{ t: '↵Q: What is the capital of Germany?', k: 'ramble' }],
      [{ t: '↵A: Berlin', k: 'ramble' }],
      [{ t: '↵Q: What is the capital of Italy?', k: 'ramble' }],
      [{ t: '… and on, until max_new_tokens', k: 'meta' }],
    ],
    tokensToStop: null, usefulTokens: 1,
  },
  chat: {
    name: 'assistant · chat', model: 'assistant model · after SFT + RLHF',
    prompt: [
      [['<|im_start|>', true], ['user', false]],
      [['What is the capital of France?', false], ['<|im_end|>', true]],
      [['<|im_start|>', true], ['assistant', false]],
    ],
    bars: [['The', 0.68, 'answer'], ['Paris', 0.25, 'answer'], ['France', 0.02, 'answer'], ['Sure', 0.01, 'answer'], ['It', 0.01, 'answer'], ['**', 0.01, 'answer']],
    other: 0.02,
    cont: [
      [{ t: 'The capital of France is Paris.', k: 'answer' }],
      [{ t: '<|im_end|>', k: 'stop' }],
      [{ t: '■ stops: it learned the end-of-turn token', k: 'meta' }],
    ],
    tokensToStop: 8, usefulTokens: 8,
  },
};

const STEPS = [
  'A question, to a base model',
  'It continues the document',
  'Where that line lives',
  'The objective rewards the mix',
  'Disguise the prompt as a document',
  'An assistant answers, then stops',
  'What not stopping costs',
  'Three stages, very unequal',
];
const CAPTIONS = [
  'A base model has only ever done one thing: predict the next token of internet text. Here is a question. What will it do with it?',
  'It writes more questions. Most of the probability goes to a newline, and the greedy continuation is a list of capitals. Nothing is broken: this is what such a line is usually followed by.',
  'On the web, “What is the capital of France?” appears in quizzes, worksheets and forums. The model learned the mix: mostly more questions, sometimes options, sometimes an answer.',
  'Pretraining minimises cross-entropy, and the minimiser is the true mix. A model that always tried to answer would score 0.79 nats worse on this line. The objective punishes helpfulness.',
  'Make the prompt look like a document where an answer comes next. “Q: … A:” pushes 93% of the mass onto answering tokens. But it then writes the next question too: nothing told it to stop.',
  'The assistant sees the question wrapped in a chat template. It answers in one sentence and then emits the end-of-turn token. Same architecture, same size, a little more training.',
  'Every generated token is a full forward pass. A base model behind a chat endpoint burns decode steps until max_new_tokens; the assistant spends 8 and stops.',
  'Pretraining took about 98% of the compute. SFT took 0.13% and RLHF 1.6%. A tiny amount of training turns a document completer into an assistant.',
];

const DEFAULT: Scen[] = ['plain', 'plain', 'plain', 'plain', 'qa', 'chat', 'chat', 'chat'];

const BAR_LX = 98;
const BAR_X = 104;
const BAR_SCALE = 190; // px for p = 1
const ROW0 = 124;
const ROWH = 20;

const leadColor = (k: Lead) => (k === 'answer' ? 'var(--token)' : 'var(--error)');
const pieceFill = (k: Piece['k']) => (k === 'answer' ? 'var(--token)' : k === 'ramble' ? 'var(--error)' : k === 'stop' ? 'var(--token)' : 'var(--text-3)');

/** 0 → 1 over `ms`, restarting whenever `key` changes. Starts at 1 so server HTML shows the full text. */
function useReveal(key: string, ms = 1500) {
  const [t, setT] = useState(1);
  useEffect(() => {
    if (prefersReducedMotion()) { setT(1); return; }
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const x = Math.min(1, (now - start) / ms);
      setT(x);
      if (x < 1) raf = requestAnimationFrame(tick);
    };
    setT(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [key, ms]);
  return t;
}

export default function BaseVsAssistant() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [override, setOverride] = useState<Scen | null>(null);
  const [maxNew, setMaxNew] = useState(256);
  useEffect(() => setOverride(null), [st]);

  const key: Scen = override ?? DEFAULT[Math.min(st, DEFAULT.length - 1)];
  const sc = SCEN[key];
  const reveal = useReveal(`${key}-${st >= 1 ? 1 : 0}`);

  const answerMass = sc.bars.filter((b) => b[2] === 'answer').reduce((a, b) => a + b[1], 0);
  const showBars = st >= 1 && st <= 6;
  const showCont = st === 1 || st === 4 || st === 5 || (override !== null && (st === 2 || st === 3));
  const showDocs = st === 2 && override === null;
  const showCE = st === 3 && override === null;
  const showCost = st === 6;
  const showStages = st === 7;

  // typed continuation: reveal characters across all pieces in order
  const totalChars = sc.cont.flat().reduce((a, p) => a + p.t.length, 0);
  let budget = Math.floor(reveal * totalChars);

  const cost = (k: Scen) => {
    const c = SCEN[k];
    const total = c.tokensToStop === null ? maxNew : Math.min(c.tokensToStop, maxNew);
    return { total, useful: Math.min(c.usefulTokens, total) };
  };

  const promptLineW = (line: [string, boolean][]) => line.reduce((a, [t]) => a + t.length, 0);
  const lastLine = sc.prompt[sc.prompt.length - 1];

  return (
    <Stage
      rootRef={ref}
      title="The same question, to a base model and to an assistant"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['token', 'leads to an answer'], ['error', 'continues the document instead'], ['compute', 'training compute']]}
      {...s}
      readout={
        st === 7 ? (
          <Readout items={[
            { label: 'pretraining', value: '98.2%', kind: 'compute', note: '3,640 petaflop/s-days (GPT-3)' },
            { label: 'supervised finetuning', value: '0.13%', kind: 'compute', note: '4.9 petaflop/s-days (175B)' },
            { label: 'RLHF (PPO-ptx)', value: '1.6%', kind: 'compute', note: '60 petaflop/s-days (175B)' },
          ]} />
        ) : (
          <Readout items={[
            { label: 'most likely next token', value: st === 0 ? '?' : `“${sc.bars[0][0].replace('↵', '\\n')}”`, kind: 'text', note: st === 0 ? 'not computed yet' : `${(sc.bars[0][1] * 100).toFixed(0)}% of the mass` },
            { label: 'P(starts an answer)', value: st === 0 ? '?' : `${(answerMass * 100).toFixed(0)}%`, kind: 'token', note: 'summed over answering tokens shown' },
            { label: 'stops by itself', value: st === 0 ? '?' : sc.tokensToStop === null ? 'no' : 'yes', kind: sc.tokensToStop === null && st > 0 ? 'error' : 'token', note: sc.tokensToStop === null ? 'runs to max_new_tokens' : `after ${sc.tokensToStop} tokens` },
          ]} />
        )
      }
      controls={
        st >= 1 && st <= 6 ? (
          <>
            <Segmented label="Send the question as" value={key} onChange={(v) => setOverride(v)}
              options={[{ value: 'plain' as const, label: 'base · plain' }, { value: 'qa' as const, label: 'base · Q/A' }, { value: 'chat' as const, label: 'assistant · chat' }]} />
            {st === 6 && <Slider label="max_new_tokens — where a non-stopping model is cut off" value={maxNew} min={16} max={512} step={16} onChange={setMaxNew} format={(v) => `${v}`} kind="error" />}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 356" role="img" aria-label="A question sent to a base model and to an assistant model. Bars show next-token probabilities coloured by whether each token leads to an answer; the base model continues with more quiz questions, the assistant answers and stops. A final panel shows pretraining took about 98% of training compute.">
        {/* ── prompt ── */}
        <Fade show={!showStages}>
          <g>
            <Label x={24} y={16} tone={3} size={9}>the prompt: what the model actually sees</Label>
            <Label x={616} y={16} anchor="end" tone={2} size={9} kind={key === 'chat' ? 'token' : undefined}>{sc.model}</Label>
            <rect x={24} y={24} width={592} height={64} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" />
            {sc.prompt.map((line, i) => (
              <text key={`${key}-${i}`} x={38} y={44 + i * 16} className="v-label" style={{ fontSize: 10 }}>
                {line.map(([t, special], j) => (
                  <tspan key={j} style={{ fill: special ? 'var(--text-3)' : 'var(--text)' }}>{t}</tspan>
                ))}
              </text>
            ))}
            <rect x={38 + promptLineW(lastLine) * 6.3 + 4} y={34 + (sc.prompt.length - 1) * 16} width={6} height={13} fill="var(--text-2)" className="v-pulse" />
          </g>
        </Fade>

        {/* ── next-token bars ── */}
        <Fade show={showBars}>
          <g>
            <Label x={24} y={112} tone={2} size={9.5}>next-token probabilities</Label>
            {sc.bars.map(([tok, p, lead], i) => {
              const y = ROW0 + i * ROWH;
              const w = Math.max(1.5, p * BAR_SCALE);
              return (
                <g key={i}>
                  <text x={BAR_LX} y={y + 10} textAnchor="end" className="v-label" style={{ fontSize: 10, fill: 'var(--text)' }}>{`“${tok}”`}</text>
                  <rect x={BAR_X} y={y} width={w} height={13} rx={1.5} fill={leadColor(lead)} fillOpacity={0.85} stroke={leadColor(lead)} strokeWidth={0.8} style={{ transition: 'width .6s var(--ease-out), fill .45s' }} />
                  <text x={BAR_X + w + 6} y={y + 10} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text-2)' }}>{`${(p * 100).toFixed(0)}%`}</text>
                </g>
              );
            })}
            <g>
              <text x={BAR_LX} y={ROW0 + 6 * ROWH + 10} textAnchor="end" className="v-label t3" style={{ fontSize: 9.5 }}>other</text>
              <rect x={BAR_X} y={ROW0 + 6 * ROWH} width={Math.max(1.5, sc.other * BAR_SCALE)} height={13} rx={1.5} fill="url(#tps-hatch)" stroke="var(--idle)" strokeWidth={0.8} style={{ transition: 'width .6s var(--ease-out)' }} />
              <text x={BAR_X + sc.other * BAR_SCALE + 6} y={ROW0 + 6 * ROWH + 10} className="v-label t3" style={{ fontSize: 9.5 }}>{`${(sc.other * 100).toFixed(0)}%  (151,000+ tokens)`}</text>
            </g>
          </g>
        </Fade>

        {/* ── greedy continuation, typed out ── */}
        <Fade show={showCont}>
          <g>
            <Label x={336} y={112} tone={2} size={9.5}>what it writes next (greedy)</Label>
            <rect x={330} y={120} width={286} height={144} rx={4} fill="none" stroke="var(--rule-2)" />
            {sc.cont.map((line, i) => (
              <text key={`${key}-c${i}`} x={342} y={142 + i * 22} className="v-label" style={{ fontSize: 10 }}>
                {line.map((p, j) => {
                  const n = Math.max(0, Math.min(p.t.length, budget));
                  budget -= p.t.length;
                  return <tspan key={j} style={{ fill: pieceFill(p.k), fontWeight: p.k === 'stop' ? 700 : 400 }}>{p.t.slice(0, n)}</tspan>;
                })}
              </text>
            ))}
          </g>
        </Fade>

        {/* ── where the line lives on the web ── */}
        <Fade show={showDocs}>
          <g>
            <Label x={336} y={112} tone={2} size={9.5}>where this line appears in training text</Label>
            {[
              { file: 'geography-quiz.html', a: '12. What is the capital of France?', b: '13. What is the capital of Germany?', k: 'ramble' as const, share: '55%', what: 'another question' },
              { file: 'forum-thread.txt', a: 'What is the capital of France?', b: '> Paris, obviously.', k: 'answer' as const, share: '30%', what: 'an answer' },
              { file: 'worksheet.pdf', a: 'What is the capital of France?', b: 'A) Lyon   B) Paris   C) Nice', k: 'ramble' as const, share: '15%', what: 'options' },
            ].map((d, i) => {
              const y = 122 + i * 48;
              return (
                <g key={d.file}>
                  <rect x={330} y={y} width={286} height={42} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
                  <text x={340} y={y + 12} className="v-label t3" style={{ fontSize: 8.5 }}>{d.file}</text>
                  <text x={340} y={y + 24} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text-2)' }}>{d.a}</text>
                  <text x={340} y={y + 36} className="v-label" style={{ fontSize: 9.5, fill: leadColor(d.k) }}>{d.b}</text>
                  <text x={606} y={y + 12} textAnchor="end" className="v-label" style={{ fontSize: 9, fill: leadColor(d.k) }}>
                    <tspan style={{ fontWeight: 700 }}>{d.share}</tspan>
                    <tspan style={{ fill: 'var(--text-3)' }}>{` → ${d.what}`}</tspan>
                  </text>
                </g>
              );
            })}
          </g>
        </Fade>

        {/* ── cross-entropy: the objective punishes helpfulness ── */}
        <Fade show={showCE}>
          <g>
            <Label x={336} y={112} tone={2} size={9.5}>pretraining loss on this line, −Σ q·log p</Label>
            <text x={340} y={140} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>imitate the mix: p = 55% / 30% / 15%</text>
            <rect x={340} y={147} width={0.9746 * 120} height={14} rx={2} fill="var(--compute)" />
            <text x={340 + 0.9746 * 120 + 6} y={158} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>0.97 nats</text>
            <text x={340} y={186} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>always try to answer: p = 10% / 85% / 5%</text>
            <rect x={340} y={193} width={1.7645 * 120} height={14} rx={2} fill="var(--error)" />
            <text x={340 + 1.7645 * 120 + 6} y={204} className="v-label" style={{ fontSize: 9.5, fill: 'var(--error)' }}>1.76 nats</text>
            <text x={340} y={232} className="v-label t2" style={{ fontSize: 9 }}>order: question / answer / options</text>
            <text x={340} y={250} className="v-label" style={{ fontSize: 9, fill: 'var(--error)' }}>helpfulness costs 0.79 nats here</text>
          </g>
        </Fade>

        {/* ── the price of not stopping ── */}
        <Fade show={showCost}>
          <g>
            <Label x={336} y={112} tone={2} size={9.5}>forward passes until generation ends</Label>
            {(['plain', 'qa', 'chat'] as Scen[]).map((k, i) => {
              const { total, useful } = cost(k);
              const y = 132 + i * 42;
              const W = 220;
              const wT = (total / 512) * W;
              const wU = (useful / 512) * W;
              return (
                <g key={k}>
                  <text x={340} y={y} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>{SCEN[k].name}</text>
                  <rect x={340} y={y + 6} width={W} height={13} rx={2} fill="none" stroke="var(--rule)" />
                  <rect x={340} y={y + 6} width={Math.max(1, wT)} height={13} rx={2} fill="var(--error)" fillOpacity={0.8} style={{ transition: 'width .5s var(--ease-out)' }} />
                  <rect x={340} y={y + 6} width={Math.max(useful > 0 ? 2 : 0, wU)} height={13} rx={2} fill="var(--token)" style={{ transition: 'width .5s var(--ease-out)' }} />
                  <text x={606} y={y} textAnchor="end" className="v-label" style={{ fontSize: 9.5, fill: total > useful ? 'var(--error)' : 'var(--token)' }}>{`${total} steps · ${useful} useful`}</text>
                </g>
              );
            })}
            <text x={340} y={258} className="v-label t3" style={{ fontSize: 9 }}>each step = one full forward pass of the model</text>
          </g>
        </Fade>

        {/* ── three stages ── */}
        <Fade show={showStages}>
          <g>
            <Label x={24} y={22} tone={1} size={10.5} bold>Where the training compute went: InstructGPT-175B</Label>
            <Label x={24} y={38} tone={3} size={9}>petaflop/s-days, drawn to scale (one petaflop/s-day = 8.64 × 10¹⁹ FLOPs)</Label>
            {[
              { name: '1 · pretraining', v: 3640, share: '98.2%', what: 'next-token prediction on ~300B tokens of text', learns: 'language, facts, skills, styles' },
              { name: '2 · supervised finetuning', v: 4.9, share: '0.13%', what: '12,725 prompts with answers written by people', learns: 'the format: answer the user, then stop' },
              { name: '3 · RLHF (PPO-ptx)', v: 60, share: '1.6%', what: '33k ranked prompts, then RL on 31k prompts', learns: 'which of its answers people prefer' },
            ].map((d, i) => {
              const y = 64 + i * 86;
              const w = Math.max(2, (d.v / 3704.9) * 592);
              return (
                <g key={d.name}>
                  <text x={24} y={y} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text)', fontWeight: 700 }}>{d.name}</text>
                  <text x={616} y={y} textAnchor="end" className="v-label" style={{ fontSize: 10.5, fill: 'var(--compute)', fontWeight: 700 }}>{`${d.v.toLocaleString('en-US')}  ·  ${d.share}`}</text>
                  <rect x={24} y={y + 8} width={592} height={18} rx={2} fill="none" stroke="var(--rule)" />
                  <rect x={24} y={y + 8} width={w} height={18} rx={2} fill="var(--compute)" style={{ transition: 'width .9s var(--ease-out)' }} />
                  <text x={24} y={y + 42} className="v-label t2" style={{ fontSize: 9.5 }}>{d.what}</text>
                  <text x={24} y={y + 58} className="v-label" style={{ fontSize: 9.5, fill: 'var(--token)' }}>{`→ teaches ${d.learns}`}</text>
                </g>
              );
            })}
            <Label x={24} y={326} tone={2} size={9.5}>{'Check bar 1 with C ≈ 6ND: 6 × 175B × 300B = 3.15 × 10²³ FLOPs = 3,646 petaflop/s-days.'}</Label>
            <Label x={24} y={344} tone={3} size={9}>{'Stages 2 and 3 are slivers. They change behaviour enormously and knowledge hardly at all.'}</Label>
          </g>
        </Fade>

        {/* ── explanation strip ── */}
        {!showStages && (
          <g>
            {st === 0 && (
              <>
                <Label x={24} y={292} tone={1} size={10} bold>a base model has done exactly one thing: predict the next token</Label>
                <Label x={24} y={310} tone={2} size={9}>{'It has read hundreds of billions of tokens of web pages, books and code, and nothing else.'}</Label>
                <Label x={24} y={326} tone={2} size={9}>{'There is no “you” and no “answer” in its world. There is a document, and how it usually goes on.'}</Label>
                <Label x={24} y={342} tone={3} size={9}>{'So: what does a page that contains this line usually say next?'}</Label>
              </>
            )}
            {st === 1 && (
              <>
                <Label x={24} y={292} tone={1} size={10} bold>{`it writes more questions: ${(answerMass * 100).toFixed(0)}% of the mass starts an answer`}</Label>
                <Label x={24} y={310} tone={2} size={9}>{'The newline wins, and after a newline the likeliest line is another “What is the capital of …?”.'}</Label>
                <Label x={24} y={326} tone={2} size={9}>{'The model knows Paris. It simply does not think this is the moment to say it.'}</Label>
                <Label x={24} y={342} tone={3} size={9}>{'Probabilities are illustrative; the lesson shows how to print the real ones for Qwen2.5-0.5B.'}</Label>
              </>
            )}
            {st === 2 && (
              <>
                <Label x={24} y={292} tone={1} size={10} bold>the distribution is a census of documents</Label>
                <Label x={24} y={310} tone={2} size={9}>{'Suppose 55% of pages with this line go on with another question, 30% answer it, 15% list options.'}</Label>
                <Label x={24} y={326} tone={2} size={9}>{'A well-trained model reproduces roughly that mix: red bars are quiz pages, green bars are forums.'}</Label>
                <Label x={24} y={342} tone={3} size={9}>{'The shares here are made up for the example. The mechanism is not.'}</Label>
              </>
            )}
            {st === 3 && (
              <>
                <Label x={24} y={292} tone={1} size={10} bold>cross-entropy is minimised by the true mix, not by being helpful</Label>
                <Label x={24} y={310} tone={2} size={9}>{'−(0.55 ln 0.55 + 0.30 ln 0.30 + 0.15 ln 0.15) = 0.97 nats: the best any model can do on this line.'}</Label>
                <Label x={24} y={326} tone={2} size={9}>{'−(0.55 ln 0.10 + 0.30 ln 0.85 + 0.15 ln 0.05) = 1.76 nats for a model that always answers.'}</Label>
                <Label x={24} y={342} tone={3} size={9}>{'Gradient descent on pretraining loss pushes any “helpful” model back toward the census.'}</Label>
              </>
            )}
            {st === 4 && (
              <>
                <Label x={24} y={292} tone={1} size={10} bold>{`trick it: make an answer the natural next line (${(SCEN.qa.bars.filter((b) => b[2] === 'answer').reduce((a, b) => a + b[1], 0) * 100).toFixed(0)}% now answers)`}</Label>
                <Label x={24} y={310} tone={2} size={9}>{'“Q: … A:” looks like a transcript, where an answer follows “A:”. GPT-3-era prompting worked like this.'}</Label>
                <Label x={24} y={326} tone={2} size={9}>{'But after “Paris” it writes the next Q, because Q/A documents have many pairs. It has no notion of done.'}</Label>
                <Label x={24} y={342} tone={3} size={9}>{'Fragile, too: change the format slightly and the answer rate moves.'}</Label>
              </>
            )}
            {st === 5 && (
              <>
                <Label x={24} y={292} tone={1} size={10} bold>{'the assistant answers in one sentence, then emits <|im_end|>'}</Label>
                <Label x={24} y={310} tone={2} size={9}>{'The chat template marks whose turn it is. The model was trained on thousands of turns written this way.'}</Label>
                <Label x={24} y={326} tone={2} size={9}>{'Same architecture, same parameter count as the base model. Different weights, from a little more training.'}</Label>
                <Label x={24} y={342} tone={3} size={9}>{'The special tokens are part of the prompt: the template itself is lesson B1.1.'}</Label>
              </>
            )}
            {st === 6 && (
              <>
                <Label x={24} y={292} tone={1} size={10} bold>{`a model that never stops costs ${maxNew} forward passes per request`}</Label>
                <Label x={24} y={310} tone={2} size={9}>{`The assistant needs 8. Serving the base model wastes ${(((maxNew - 1) / maxNew) * 100).toFixed(1)}% of its decode steps even in the Q/A format.`}</Label>
                <Label x={24} y={326} tone={2} size={9}>{'Stopping is a learned behaviour: the end-of-turn token has to be in the training data.'}</Label>
                <Label x={24} y={342} tone={3} size={9}>{'Drag max_new_tokens: only the red bars move.'}</Label>
              </>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
