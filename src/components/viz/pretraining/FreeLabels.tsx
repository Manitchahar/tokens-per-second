import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Arrow, Fade, fmt } from '../lib/svg';

// One ordinary sentence. Every token after the first is the answer to
// "what comes next?" for the tokens before it, so the sentence labels itself.
const WORDS = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.'];
const N = WORDS.length;
const GAP = 6;
const CW = WORDS.map((w) => Math.max(26, w.length * 7 + 14));
const ROW_W = CW.reduce((a, b) => a + b, 0) + GAP * (N - 1);
const X0 = (640 - ROW_W) / 2;
const CX = CW.map((_, i) => X0 + CW.slice(0, i).reduce((a, b) => a + b, 0) + GAP * i);
const ROW_Y = 34;
const CH = 22;

const LABEL_SECONDS = 10; // assumed seconds per hand-written label (generous)
const YEAR_S = 365.25 * 24 * 3600;

// Log axis for the corpus step: 10^4 .. 10^14 examples.
const AX0 = 70;
const AX1 = 610;
const LOG_MIN = 4;
const LOG_MAX = 14;
const ax = (n: number) => AX0 + ((Math.log10(n) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * (AX1 - AX0);
const MARKS: { n: number; name: string; note: string; human: boolean; up: boolean }[] = [
  { n: 1e5, name: 'SQuAD 1.1', note: '~100K questions', human: true, up: true },
  { n: 1.28e6, name: 'ImageNet', note: '1.28M images', human: true, up: false },
  { n: 3.3e9, name: 'BERT’s text', note: '3.3B words', human: false, up: true },
  { n: 1.5e13, name: 'Llama 3', note: '15T+ tokens', human: false, up: false },
];

const TICK: Record<number, string> = { 4: '10K', 5: '100K', 6: '1M', 7: '10M', 8: '100M', 9: '1B', 10: '10B', 11: '100B', 12: '1T', 13: '10T', 14: '100T' };

const TASKS = ['classify', 'answer', 'summarise', 'translate', 'chat'];

const STEPS = [
  'Labels cost people',
  'Raw text has no labels… or does it?',
  'Hide the next token',
  'Slide: every position is an example',
  'One pass trains every position',
  'Or hide a token in the middle',
  'A corpus is a pile of examples',
  'Pretrain once, adapt many times',
];
const CAPTIONS = [
  'Supervised learning needs a person to write every answer. At an assumed 10 seconds per label, a million labels is 2,778 hours of work — and a new task starts from zero.',
  'An ordinary sentence. Nobody labelled it. But every token in it is the answer to a question the sentence itself can ask: given the tokens before me, what comes next?',
  'Cover a token and keep the ones before it. The covered token is the label, and it was written by whoever wrote the sentence. Drag the window and the question changes.',
  'Slide the window one token at a time and every position emits a (context → target) pair. Eleven tokens, ten free training examples, no annotator.',
  'With the causal mask from the attention lesson, row t only ever sees tokens up to t. So one forward pass over the sentence makes all ten predictions at once, and none of them can peek.',
  'A different way to hide: cover a token in the middle and predict it from both sides. The answer is still free. This is BERT’s game, and the subject of lesson 3.',
  'Scale it up. A corpus of N tokens holds about N next-token examples. Drag the slider: a trillion examples by hand would take hundreds of thousands of person-years.',
  'Learn once from raw text, at great expense. Then adapt the same weights to many tasks cheaply, by a little finetuning or just a prompt. That reusable model is a foundation model.',
];

type ChipState = 'plain' | 'context' | 'target' | 'hidden' | 'mask' | 'faint';

function SentenceRow({ state }: { state: (i: number) => ChipState }) {
  return (
    <g>
      {WORDS.map((w, i) => {
        const s = state(i);
        const fill = s === 'target' ? 'var(--token)' : s === 'context' ? 'var(--ground-3)' : s === 'mask' ? 'var(--token-dim)' : 'var(--ground-2)';
        const stroke = s === 'target' || s === 'mask' ? 'var(--token)' : s === 'context' ? 'var(--text-2)' : 'var(--rule-2)';
        const txt = s === 'mask' ? '[MASK]' : s === 'hidden' ? '?' : w;
        const color = s === 'target' ? '#0e1116' : s === 'faint' ? 'var(--text-3)' : s === 'hidden' ? 'var(--text-3)' : 'var(--text)';
        const extra = s === 'mask' ? 18 : 0;
        return (
          <g key={i} className="v-fade" style={{ opacity: s === 'faint' ? 0.45 : 1 }}>
            <rect x={CX[i] - extra / 2} y={ROW_Y} width={CW[i] + extra} height={CH} rx={3} fill={fill} stroke={stroke}
              strokeDasharray={s === 'hidden' || s === 'mask' ? '3 3' : undefined} className="v-cell" />
            <text x={CX[i] + CW[i] / 2} y={ROW_Y + 15} textAnchor="middle" className="v-label"
              style={{ fontSize: s === 'mask' ? 10 : 11, fontWeight: 600, fill: color }}>{txt}</text>
          </g>
        );
      })}
    </g>
  );
}

export default function FreeLabels() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [pos, setPos] = useState(3); // index of the target token (1..N-1)
  const [logN, setLogN] = useState(12); // corpus size, log10 tokens

  const t = Math.max(1, Math.min(N - 1, pos));
  const winW = useTween(CX[t - 1] + CW[t - 1] - CX[0] + 10, 500);
  const tgtX = useTween(CX[t] + CW[t] / 2, 500);
  const corpus = Math.pow(10, logN);
  const handYears = (corpus * LABEL_SECONDS) / YEAR_S;

  const rowState = (i: number): ChipState => {
    if (st === 0) return 'faint';
    if (st === 1) return 'plain';
    if (st === 2 || st === 3) return i < t ? 'context' : i === t ? (st === 2 ? 'hidden' : 'target') : 'faint';
    if (st === 4) return i <= t - 1 ? 'context' : i === t ? 'target' : 'faint';
    if (st === 5) return i === 2 ? 'mask' : 'context';
    return 'faint';
  };

  const readoutItems =
    st === 0
      ? [
          { label: '1 label', value: `${LABEL_SECONDS} s`, kind: 'error' as const, note: 'assumed, by a person' },
          { label: '1M labels', value: `${fmt.int((1e6 * LABEL_SECONDS) / 3600)} h`, kind: 'error' as const, note: '≈ 1.4 work-years' },
          { label: 'new task', value: 'start again', kind: 'error' as const, note: 'labels do not transfer' },
        ]
      : st <= 4
        ? [
            { label: 'tokens in sentence', value: String(N), kind: 'text' as const },
            { label: 'free examples', value: String(N - 1), kind: 'token' as const, note: 'one per position after the first' },
            { label: 'current pair', value: st >= 2 ? `${t} → 1` : '—', kind: 'text' as const, note: st >= 2 ? `${t} tokens of context, 1 target` : 'nothing hidden yet' },
            { label: 'labelling cost', value: '0 s', kind: 'token' as const, note: 'the author wrote the answers' },
          ]
        : st === 5
          ? [
              { label: 'hidden', value: '1 of 11', kind: 'token' as const, note: 'BERT hides ~15%' },
              { label: 'sees', value: 'both sides', kind: 'text' as const, note: '2 tokens left, 8 right' },
              { label: 'labelling cost', value: '0 s', kind: 'token' as const },
            ]
          : st === 6
            ? [
                { label: 'corpus', value: `${fmt.si(corpus)} tokens`, kind: 'token' as const },
                { label: 'next-token examples', value: `≈ ${fmt.si(corpus)}`, kind: 'token' as const, note: 'one per token' },
                { label: 'by hand at 10 s each', value: handYears >= 1 ? `${fmt.si(handYears)} years` : `${fmt.int(handYears * 365.25 * 24)} h`, kind: 'error' as const, note: 'one person, no sleep' },
                { label: 'vs SQuAD', value: `${fmt.si(corpus / 1e5)}×`, kind: 'text' as const, note: 'its ~100K questions' },
              ]
            : [
                { label: 'pretraining', value: 'once', kind: 'compute' as const, note: 'raw text, huge compute' },
                { label: 'adaptation', value: 'many times', kind: 'token' as const, note: 'small data or just a prompt' },
              ];

  return (
    <Stage
      rootRef={ref}
      title="Raw text labels itself"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['token', 'target (free label)'], ['text', 'context'], ['error', 'human cost'], ['idle', 'cannot see']]}
      {...s}
      readout={<Readout items={readoutItems} />}
      controls={
        st >= 2 && st <= 4 ? (
          <Slider label="Which token is hidden (the target)" value={t} min={1} max={N - 1} step={1} onChange={setPos}
            format={(v) => `#${v + 1} “${WORDS[v]}”`} kind="token" />
        ) : st === 6 ? (
          <Slider label="Corpus size (tokens)" value={logN} min={4} max={13.2} step={0.1} onChange={setLogN}
            format={(v) => fmt.si(Math.pow(10, v))} kind="token" />
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="A sentence of eleven tokens turns into ten (context, target) training pairs by hiding each next token in turn; the causal mask lets one forward pass make all ten predictions; a slider scales the idea to a corpus of trillions of tokens; one pretrained model then adapts to many tasks.">
        {/* ── the sentence row, present from step 1 to 5 ── */}
        <Fade show={st >= 1 && st <= 5}>
          <SentenceRow state={rowState} />
          {st === 1 && WORDS.slice(0, -1).map((_, i) => (
            <Arrow key={i} x1={CX[i] + CW[i] / 2} y1={ROW_Y + CH + 4} x2={CX[i + 1] + CW[i + 1] / 2 - 4} y2={ROW_Y + CH + 6} curve={-10} muted width={1} />
          ))}
        </Fade>

        {/* ── step 0: supervised labelling ── */}
        <Fade show={st === 0}>
          <Label x={40} y={30} tone={1} size={11} bold mono={false}>Supervised learning: a person writes every answer</Label>
          {[
            ['“Loved every minute of it.”', 'positive'],
            ['“The battery died in an hour.”', 'negative'],
            ['“Where is the Eiffel Tower?”', 'Paris'],
          ].map(([inp, lab], i) => (
            <g key={i}>
              <rect x={40} y={52 + i * 40} width={250} height={26} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x={52} y={69 + i * 40} tone={1} size={10.5} mono={false}>{inp}</Label>
              <Arrow x1={296} y1={65 + i * 40} x2={350} y2={65 + i * 40} kind="error" />
              <rect x={358} y={52 + i * 40} width={92} height={26} rx={3} fill="var(--error-dim)" stroke="var(--error)" />
              <Label x={404} y={69 + i * 40} anchor="middle" tone={1} size={10.5}>{lab}</Label>
              <Label x={462} y={69 + i * 40} tone={3} size={9.5}>written by a person</Label>
            </g>
          ))}
          <Label x={40} y={196} tone={2} size={10.5} mono={false}>Every arrow is someone’s time. Assume 10 seconds per label:</Label>
          <Label x={40} y={218} tone={1} size={10.5}>{'1,000,000 labels × 10 s = 10,000,000 s ≈ 2,778 hours'}</Label>
          <Label x={40} y={238} tone={2} size={10.5}>{'1,000,000,000,000 labels × 10 s ≈ 317,000 person-years'}</Label>
          <Label x={40} y={270} tone={3} size={10} mono={false}>And a new task means a new dataset: sentiment labels say nothing about translation.</Label>
          <Label x={40} y={290} tone={3} size={10} mono={false}>An LLM learns from trillions of tokens. Hand-labelling at that scale is not expensive; it is impossible.</Label>
        </Fade>

        {/* ── step 1: the question each token answers ── */}
        <Fade show={st === 1}>
          <Label x={320} y={110} anchor="middle" tone={1} size={11.5} bold mono={false}>No labels anywhere. Or are there?</Label>
          <Label x={320} y={136} anchor="middle" tone={2} size={10.5} mono={false}>Each arrow asks: given everything to my left, what comes next?</Label>
          <Label x={320} y={156} anchor="middle" tone={2} size={10.5} mono={false}>And the sentence already contains the answer to every one of those questions.</Label>
          <Label x={320} y={196} anchor="middle" tone={3} size={10} mono={false}>The labels are not missing. They are the text itself, shifted by one position.</Label>
        </Fade>

        {/* ── step 2: one pair ── */}
        <Fade show={st === 2}>
          <rect x={CX[0] - 5} y={ROW_Y - 5} width={winW} height={CH + 10} rx={5} fill="none" stroke="var(--text-2)" strokeDasharray="4 3" />
          <Label x={CX[0]} y={ROW_Y - 12} tone={2} size={10}>context: the input the model sees</Label>
          <Label x={tgtX} y={ROW_Y + CH + 48} anchor="middle" kind="token" size={10}>target</Label>
          <Arrow x1={tgtX} y1={ROW_Y + CH + 36} x2={tgtX} y2={ROW_Y + CH + 6} kind="token" />
          <rect x={150} y={128} width={340} height={70} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <Label x={320} y={152} anchor="middle" tone={1} size={11}>{`“${WORDS.slice(0, t).join(' ')}” → ?`}</Label>
          <Label x={320} y={176} anchor="middle" kind="token" size={11} bold>{`answer: “${WORDS[t]}”`}</Label>
          <Label x={320} y={226} anchor="middle" tone={2} size={10.5} mono={false}>One training example: an input, and the correct answer for it.</Label>
          <Label x={320} y={246} anchor="middle" tone={3} size={10} mono={false}>Nobody wrote the label for us. Whoever wrote the sentence did.</Label>
        </Fade>

        {/* ── step 3: every pair ── */}
        <Fade show={st === 3}>
          {WORDS.slice(1).map((w, k) => {
            const i = k + 1;
            const hot = i === t;
            const y = 82 + k * 21;
            return (
              <g key={k} className="v-fade" style={{ opacity: hot ? 1 : 0.62 }}>
                <Label x={382} y={y + 4} anchor="end" tone={hot ? 1 : 2} size={9.5} bold={hot}>{WORDS.slice(0, i).join(' ')}</Label>
                <Arrow x1={388} y1={y} x2={412} y2={y} kind={hot ? 'token' : undefined} muted={!hot} width={1} />
                <rect x={418} y={y - 8} width={Math.max(26, w.length * 7 + 14)} height={16} rx={2} fill={hot ? 'var(--token)' : 'var(--token-dim)'} stroke="var(--token)" className="v-cell" />
                <text x={418 + Math.max(26, w.length * 7 + 14) / 2} y={y + 3.5} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fontWeight: 600, fill: hot ? '#0e1116' : 'var(--text)' }}>{w}</text>
              </g>
            );
          })}
          <Label x={520} y={92} tone={2} size={10} mono={false}>10 pairs</Label>
          <Label x={520} y={108} tone={3} size={9.5} mono={false}>from 11 tokens,</Label>
          <Label x={520} y={124} tone={3} size={9.5} mono={false}>0 annotators</Label>
        </Fade>

        {/* ── step 4: one pass, every position ── */}
        <Fade show={st === 4}>
          {(() => {
            const cell = 11;
            const g = 2;
            const mx = 150;
            const my = 84;
            const cells = [];
            for (let r = 0; r < N; r++) {
              for (let c = 0; c < N; c++) {
                const seen = c <= r;
                const hot = r === t - 1;
                cells.push(
                  <rect key={`${r}-${c}`} x={mx + c * (cell + g)} y={my + r * (cell + g)} width={cell} height={cell} rx={1.5}
                    fill={seen ? (hot ? 'var(--compute)' : 'var(--compute-dim)') : 'url(#tps-hatch)'}
                    stroke={seen ? 'var(--compute)' : 'var(--idle)'} strokeWidth={0.7} className="v-cell" />,
                );
              }
            }
            return (
              <g>
                <Label x={mx + (N * (cell + g)) / 2} y={my - 8} anchor="middle" tone={3} size={9}>can see →</Label>
                {cells}
                {WORDS.map((w, r) => (
                  <g key={r}>
                    <Label x={mx - 8} y={my + r * (cell + g) + 9} anchor="end" tone={r === t - 1 ? 1 : 3} size={9} bold={r === t - 1}>{w}</Label>
                    <Label x={mx + N * (cell + g) + 8} y={my + r * (cell + g) + 9} tone={3} size={9}>→</Label>
                    <Label x={mx + N * (cell + g) + 22} y={my + r * (cell + g) + 9} kind={r < N - 1 ? 'token' : undefined} tone={3} size={9} bold={r === t - 1}>
                      {r < N - 1 ? WORDS[r + 1] : '(next sentence)'}
                    </Label>
                  </g>
                ))}
              </g>
            );
          })()}
          <Label x={420} y={100} tone={1} size={10.5} bold mono={false}>one forward pass</Label>
          <Label x={420} y={120} tone={2} size={9.5} mono={false}>Row t is the model’s output at</Label>
          <Label x={420} y={134} tone={2} size={9.5} mono={false}>position t: its guess for t + 1.</Label>
          <Label x={420} y={156} tone={2} size={9.5} mono={false}>The causal mask hides every</Label>
          <Label x={420} y={170} tone={2} size={9.5} mono={false}>later token, so no row can peek</Label>
          <Label x={420} y={184} tone={2} size={9.5} mono={false}>at its own answer.</Label>
          <Label x={420} y={210} kind="token" size={10} bold mono={false}>11 tokens in →</Label>
          <Label x={420} y={226} kind="token" size={10} bold mono={false}>10 graded guesses out</Label>
          <Label x={420} y={252} tone={3} size={9} mono={false}>(the last row’s answer lives in</Label>
          <Label x={420} y={265} tone={3} size={9} mono={false}>the next chunk of text)</Label>
        </Fade>

        {/* ── step 5: masked token ── */}
        <Fade show={st === 5}>
          <Arrow x1={CX[1] + CW[1] / 2} y1={ROW_Y + CH + 4} x2={300} y2={128} curve={-12} kind="token" width={1.1} />
          <Arrow x1={CX[0] + CW[0] / 2} y1={ROW_Y + CH + 4} x2={292} y2={128} curve={-16} kind="token" width={1.1} />
          {[3, 4, 5, 6].map((i) => (
            <Arrow key={i} x1={CX[i] + CW[i] / 2} y1={ROW_Y + CH + 4} x2={340} y2={128} curve={10} kind="token" width={1.1} opacity={0.85} />
          ))}
          <rect x={250} y={130} width={140} height={34} rx={4} fill="var(--token-dim)" stroke="var(--token)" />
          <Label x={320} y={151} anchor="middle" tone={1} size={11}>{'[MASK] → sat'}</Label>
          <Label x={320} y={196} anchor="middle" tone={2} size={10.5} mono={false}>Hide a token in the middle, predict it from both sides.</Label>
          <Label x={320} y={216} anchor="middle" tone={2} size={10.5} mono={false}>The answer is still free: it was in the text before we covered it.</Label>
          <Label x={320} y={246} anchor="middle" tone={3} size={10} mono={false}>Only the hidden tokens are examples, so each pass learns from fewer of them.</Label>
          <Label x={320} y={264} anchor="middle" tone={3} size={10} mono={false}>That trade-off is lesson 3 (BERT) and lesson 5 (why decoders won).</Label>
        </Fade>

        {/* ── step 6: corpus scale ── */}
        <Fade show={st === 6}>
          <Label x={AX0} y={40} tone={1} size={11} bold mono={false}>How many training examples? (log scale: each tick is 10× the last)</Label>
          <line x1={AX0} x2={AX1} y1={200} y2={200} className="v-rule" />
          {Array.from({ length: LOG_MAX - LOG_MIN + 1 }, (_, k) => LOG_MIN + k).map((e) => (
            <g key={e}>
              <line x1={ax(Math.pow(10, e))} x2={ax(Math.pow(10, e))} y1={200} y2={205} className="v-rule" />
              <Label x={ax(Math.pow(10, e))} y={218} anchor="middle" tone={3} size={8.5}>{TICK[e]}</Label>
            </g>
          ))}
          <rect x={AX0} y={150} width={Math.max(0, ax(corpus) - AX0)} height={18} rx={2} fill="var(--token)" style={{ transition: 'width .5s var(--ease-out)' }} />
          <Label x={ax(corpus) > 330 ? ax(corpus) - 8 : ax(corpus) + 8} y={163} anchor={ax(corpus) > 330 ? 'end' : 'start'} tone={1} size={10} bold
            style={{ fill: ax(corpus) > 330 ? '#0e1116' : undefined }}>{`your corpus: ${fmt.si(corpus)} examples`}</Label>
          {MARKS.map((m) => (
            <g key={m.name}>
              <line x1={ax(m.n)} x2={ax(m.n)} y1={m.up ? 176 : 200} y2={m.up ? 200 : 236} stroke={m.human ? 'var(--error)' : 'var(--token)'} strokeWidth={1.2} />
              <circle cx={ax(m.n)} cy={200} r={3.5} fill={m.human ? 'var(--error)' : 'var(--token)'} />
              <Label x={ax(m.n)} y={m.up ? 132 - 0 : 252} anchor="middle" tone={1} size={9.5} bold>{m.name}</Label>
              <Label x={ax(m.n)} y={m.up ? 145 : 265} anchor="middle" tone={3} size={9}>{m.note}</Label>
            </g>
          ))}
          <Label x={AX0} y={300} tone={2} size={10} mono={false}>Red: every label written by a person. Green: the labels came with the text.</Label>
          <Label x={AX0} y={318} tone={3} size={9.5} mono={false}>A corpus of N tokens is about N next-token examples, whatever the window length.</Label>
        </Fade>

        {/* ── step 7: foundation model ── */}
        <Fade show={st === 7}>
          <rect x={30} y={110} width={140} height={70} rx={4} fill="var(--token-dim)" stroke="var(--token)" />
          <Label x={100} y={136} anchor="middle" tone={1} size={10.5} bold mono={false}>raw text</Label>
          <Label x={100} y={153} anchor="middle" tone={2} size={9.5} mono={false}>trillions of tokens</Label>
          <Label x={100} y={168} anchor="middle" tone={3} size={9} mono={false}>no human labels</Label>
          <Arrow x1={174} y1={145} x2={226} y2={145} kind="compute" width={2} />
          <Label x={200} y={132} anchor="middle" kind="compute" size={9}>pretrain</Label>
          <rect x={232} y={98} width={172} height={94} rx={5} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1.3} />
          <Label x={318} y={128} anchor="middle" tone={1} size={11} bold mono={false}>one pretrained model</Label>
          <Label x={318} y={148} anchor="middle" tone={2} size={9.5} mono={false}>the weights you download</Label>
          <Label x={318} y={168} anchor="middle" kind="compute" size={9.5} mono={false}>trained once, at great cost</Label>
          {TASKS.map((task, i) => {
            const y = 70 + i * 40;
            return (
              <g key={task}>
                <Arrow x1={408} y1={145} x2={482} y2={y + 11} kind="token" width={1.2} curve={i < 2 ? -6 : i > 2 ? 6 : 0} />
                <rect x={486} y={y} width={100} height={22} rx={3} fill="var(--ground-3)" stroke="var(--token)" />
                <Label x={536} y={y + 15} anchor="middle" tone={1} size={10.5} mono={false}>{task}</Label>
              </g>
            );
          })}
          <Label x={440} y={290} anchor="middle" kind="token" size={10} mono={false}>adapt cheaply: a little labelled data, or just a prompt</Label>
          <Label x={318} y={228} anchor="middle" tone={1} size={10.5} bold mono={false}>a “foundation model”</Label>
          <Label x={318} y={246} anchor="middle" tone={3} size={9.5} mono={false}>everything else is built on it</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
