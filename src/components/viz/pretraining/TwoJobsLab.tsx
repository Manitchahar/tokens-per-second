import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, fmt } from '../lib/svg';

// What each model is allowed to look at, for every experiment in the lab notebook.
// No model outputs are shown here: the notebook produces those on your GPU.
const WORDS = ['The', 'bank', 'was', 'steep', 'and', 'muddy', ',', 'so', 'we', 'could', 'not', 'reach', 'it', '.'];
const N = WORDS.length;
const CW = 40;
const GAP = 3;
const X0 = (640 - (N * CW + (N - 1) * GAP)) / 2;
const cx = (i: number) => X0 + i * (CW + GAP);

const STEPS = [
  'BERT fills a blank from both sides',
  'GPT-2 predicts from the left only',
  'The blank near the start',
  'Scoring whole sentences instead',
  'Asking BERT to write',
  'Graded positions per pass',
];
const CAPTIONS = [
  'Lab step 1: BERT reads every token around the [MASK]. Drag the blank along the sentence: the evidence always comes from both sides.',
  'Lab step 3: GPT-2 predicts each token from the tokens before it. Drag the position: everything to the right is invisible.',
  'Lab step 5a: for “The ___ was steep and muddy”, GPT-2 sees only “The”. Every word that tells you it is a river bank is hidden.',
  'GPT-2’s workaround: write out each candidate sentence in full and score it. The right-hand side now counts, but it costs one pass per candidate.',
  'Lab step 5b: BERT fills six masks at the end, one per pass. Each pass re-reads the whole sequence, because every token sees every other.',
  'Lab step 6a: on the same text, GPT-2 is graded at every position and BERT at about 15% of them.',
];

type Vis = 'context' | 'target' | 'hidden' | 'plain' | 'faint';

export default function TwoJobsLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [pos, setPos] = useState(3);
  const [cands, setCands] = useState(6);

  const k = st === 2 || st === 3 ? 1 : Math.max(1, Math.min(N - 2, pos));
  const vis = (i: number): Vis => {
    if (st === 0) return i === k ? 'target' : 'context';
    if (st === 1 || st === 2) return i < k ? 'context' : i === k ? 'target' : 'hidden';
    if (st === 3) return 'context';
    return 'plain';
  };
  const bertGraded = [3, 9, 12]; // ~15% of 14 positions, for the last step

  const writeLen = 10 + 6 + 3; // [CLS] + prompt + masks + . + [SEP], for the BERT-writes step
  const bertWriteForwards = 6 * writeLen;

  const readout =
    st === 0
      ? [
          { label: 'BERT sees', value: `${N - 1} tokens`, kind: 'compute' as const, note: `${k} left, ${N - 1 - k} right` },
          { label: 'passes', value: '1', kind: 'token' as const, note: 'ranks all 30,522 tokens at once' },
        ]
      : st === 1 || st === 2
        ? [
            { label: 'GPT-2 sees', value: `${k} token${k === 1 ? '' : 's'}`, kind: 'compute' as const, note: `${N - 1 - k} to the right are hidden` },
            { label: 'predicting', value: `“${WORDS[k]}”`, kind: 'token' as const },
          ]
        : st === 3
          ? [
              { label: 'candidates', value: String(cands), kind: 'text' as const },
              { label: 'GPT-2 passes', value: String(cands), kind: 'error' as const, note: 'one full sentence each' },
              { label: 'BERT passes', value: '1', kind: 'token' as const, note: 'one [MASK], whole vocabulary' },
            ]
          : st === 4
            ? [
                { label: 'BERT passes', value: '6', kind: 'error' as const, note: 'one per new token' },
                { label: 'token-forwards', value: fmt.int(bertWriteForwards), kind: 'error' as const, note: `6 × ${writeLen} tokens` },
                { label: 'reusable work', value: 'none', kind: 'error' as const, note: 'every row changes each time' },
              ]
            : [
                { label: 'GPT-2 graded', value: `${N - 1} of ${N - 1}`, kind: 'token' as const, note: 'every next token' },
                { label: 'BERT graded', value: `≈ ${(0.15 * N).toFixed(1)}`, kind: 'token' as const, note: '15% of positions' },
                { label: 'ratio', value: '≈ 6.7×', kind: 'text' as const },
              ];

  return (
    <Stage
      rootRef={ref}
      title="What each model can see, experiment by experiment"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['compute', 'visible context'], ['token', 'predicted'], ['idle', 'hidden'], ['error', 'repeated work']]}
      {...s}
      readout={<Readout items={readout} />}
      controls={st === 0 || st === 1 ? (
        <Slider label="Position being predicted" value={k} min={1} max={N - 2} step={1} onChange={setPos} format={(v) => `#${v + 1} “${WORDS[v]}”`} kind="token" />
      ) : st === 3 ? (
        <Slider label="Candidate words to score" value={cands} min={1} max={1000} log onChange={setCands} format={fmt.int} kind="text" />
      ) : undefined}
    >
      <svg viewBox="0 0 640 300" role="img" aria-label="A sentence with one token being predicted: BERT can see both sides, GPT-2 only the left; GPT-2 can only use the right side by scoring whole candidate sentences, one pass each; BERT writing six tokens needs six full passes; and GPT-2 is graded at every position while BERT is graded at about 15%.">
        {/* sentence row for steps 0-3 */}
        <Fade show={st <= 3}>
          <Label x={X0} y={40} tone={1} size={10.5} bold mono={false}>
            {st === 0 ? 'BERT (bert-base-uncased)' : st === 3 ? 'GPT-2, scoring one complete candidate sentence' : 'GPT-2 (gpt2, 124M)'}
          </Label>
          {WORDS.map((w, i) => {
            const v = vis(i);
            const fill = v === 'target' ? (st === 0 ? 'var(--token-dim)' : 'var(--token)') : v === 'context' ? 'var(--compute-dim)' : v === 'hidden' ? 'url(#tps-hatch)' : 'var(--ground-3)';
            const stroke = v === 'target' ? 'var(--token)' : v === 'context' ? 'var(--compute)' : v === 'hidden' ? 'var(--idle)' : 'var(--rule-2)';
            const txt = v === 'target' ? (st === 0 ? '[MASK]' : '?') : st === 3 && i === 1 ? 'bank' : w;
            return (
              <g key={i}>
                <rect x={cx(i)} y={56} width={CW} height={26} rx={3} fill={fill} stroke={stroke} strokeDasharray={v === 'target' ? '3 3' : undefined} className="v-cell" />
                <Label x={cx(i) + CW / 2} y={73} anchor="middle" tone={v === 'hidden' ? 3 : 1} size={txt.length > 5 ? 8 : 9.5}
                  style={{ fill: v === 'target' && st !== 0 ? '#0e1116' : undefined }}>{txt}</Label>
              </g>
            );
          })}
          {(st === 0 || st === 1 || st === 2) && WORDS.map((_, i) => {
            const see = st === 0 ? i !== k : i < k;
            if (!see) return null;
            return <Arrow key={i} x1={cx(i) + CW / 2} y1={86} x2={cx(k) + CW / 2 + (i < k ? -4 : 4)} y2={128} curve={i < k ? 8 : -8} kind="compute" width={0.9} opacity={0.7} />;
          })}
        </Fade>

        <Fade show={st === 0}>
          <rect x={220} y={132} width={200} height={30} rx={4} fill="var(--token-dim)" stroke="var(--token)" />
          <Label x={320} y={151} anchor="middle" tone={1} size={10}>{`guess “${WORDS[k]}” from both sides`}</Label>
          <Label x={320} y={200} anchor="middle" tone={2} size={10} mono={false}>In the notebook: fill-mask on four sentences, top 5 each. For the two “bank” sentences,</Label>
          <Label x={320} y={216} anchor="middle" tone={2} size={10} mono={false}>only the words after the blank differ, so any change in BERT’s guesses came from the right.</Label>
        </Fade>

        <Fade show={st === 1 || st === 2}>
          <rect x={220} y={132} width={200} height={30} rx={4} fill="var(--token)" stroke="var(--token)" />
          <Label x={320} y={151} anchor="middle" size={10} style={{ fill: '#0e1116' }}>{`guess token #${k + 1} from the left`}</Label>
        </Fade>
        <Fade show={st === 1}>
          <Label x={320} y={200} anchor="middle" tone={2} size={10} mono={false}>In the notebook: per-token loss of a sentence. Each row of that table is one of these</Label>
          <Label x={320} y={216} anchor="middle" tone={2} size={10} mono={false}>predictions; every one sees only its left, and all of them come from a single pass.</Label>
        </Fade>
        <Fade show={st === 2}>
          <Label x={320} y={200} anchor="middle" tone={2} size={10} mono={false}>{'“steep”, “muddy”, “reach it”: all hatched. GPT-2’s guesses after “The” can only be'}</Label>
          <Label x={320} y={216} anchor="middle" tone={2} size={10} mono={false}>the generic ways English sentences start. Compare them with BERT’s in step 1.</Label>
        </Fade>

        <Fade show={st === 3}>
          {Array.from({ length: Math.min(6, cands) }, (_, i) => (
            <rect key={i} x={X0 + i * 6} y={100 + i * 8} width={N * (CW + GAP) - GAP} height={14} rx={2}
              fill="var(--error-dim)" stroke="var(--error)" opacity={1 - i * 0.12} />
          ))}
          <Label x={X0} y={182} tone={2} size={10} mono={false}>{`${fmt.int(cands)} candidate${cands === 1 ? '' : 's'} = ${fmt.int(cands)} full forward pass${cands === 1 ? '' : 'es'} over ${N} tokens = ${fmt.int(cands * N)} token-forwards`}</Label>
          <Label x={X0} y={206} tone={2} size={10} mono={false}>Now “steep and muddy” is part of what gets scored, so the right side finally counts.</Label>
          <Label x={X0} y={222} tone={2} size={10} mono={false}>{`BERT does the same job in one pass of ${N} tokens, ranking its whole vocabulary at once.`}</Label>
        </Fade>

        {/* step 4: BERT writes */}
        <Fade show={st === 4}>
          <Label x={40} y={34} tone={1} size={10.5} bold mono={false}>BERT asked to continue: one [MASK] filled per pass, left to right</Label>
          {Array.from({ length: 6 }, (_, pass) => (
            <g key={pass}>
              <Label x={64} y={66 + pass * 32} anchor="end" tone={3} size={9}>{`pass ${pass + 1}`}</Label>
              {Array.from({ length: writeLen }, (_, j) => {
                const isPrompt = j >= 1 && j <= 10;
                const m = j - 11;
                const filled = m >= 0 && m < pass;
                const now = m === pass;
                const special = j === 0 || j === writeLen - 1 || j === writeLen - 2;
                const fill = now ? 'var(--token)' : filled ? 'var(--token-dim)' : m >= 0 && m < 6 ? 'var(--ground-2)' : 'var(--error-dim)';
                const stroke = now || filled ? 'var(--token)' : m >= 0 && m < 6 ? 'var(--rule-2)' : 'var(--error)';
                return <rect key={j} x={72 + j * 28} y={54 + pass * 32} width={25} height={18} rx={2} fill={special ? 'var(--ground-3)' : fill} stroke={special ? 'var(--rule-2)' : stroke}
                  strokeDasharray={m >= pass + 1 && m < 6 ? '3 3' : undefined} opacity={isPrompt || m >= 0 || special ? 1 : 0.5} />;
              })}
            </g>
          ))}
          <Label x={72} y={260} tone={2} size={10} mono={false}>Red: the prompt, re-read on every pass. Green: tokens written so far. Dashed: masks still empty.</Label>
          <Label x={72} y={278} tone={3} size={9.5} mono={false}>A decoder would read the prompt once and then process one new token per step (lesson 5).</Label>
        </Fade>

        {/* step 5: graded positions */}
        <Fade show={st === 5}>
          {(['GPT-2: every next token', 'BERT: the chosen 15%'] as const).map((name, r) => (
            <g key={name}>
              <Label x={X0} y={54 + r * 100} tone={1} size={10.5} bold mono={false}>{name}</Label>
              {WORDS.map((w, i) => {
                const graded = r === 0 ? i >= 1 : bertGraded.includes(i);
                return (
                  <g key={i}>
                    <rect x={cx(i)} y={64 + r * 100} width={CW} height={26} rx={3} fill={graded ? 'var(--token)' : 'var(--ground-3)'} stroke={graded ? 'var(--token)' : 'var(--rule-2)'} />
                    <Label x={cx(i) + CW / 2} y={81 + r * 100} anchor="middle" tone={1} size={w.length > 5 ? 8 : 9.5} style={{ fill: graded ? '#0e1116' : undefined }}>{w}</Label>
                  </g>
                );
              })}
            </g>
          ))}
          <Label x={X0} y={252} tone={2} size={10} mono={false}>The notebook counts both on the same 64 sequences and prints the ratio. Expect about 6.7.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
