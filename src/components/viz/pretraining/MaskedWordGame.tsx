import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, fmt } from '../lib/svg';

// Steps 1-4: a sentence whose first content word needs its right-hand side.
const SENT = ['the', 'bank', 'was', 'steep', 'and', 'muddy'];
const NS = SENT.length;
const MX = 150;
const MY = 104;
const CELL = 24;
const GAP = 3;
const PITCH = CELL + GAP;

// Steps 5-6: one BERT training sequence, WordPiece-style and lower-cased.
const SEQ = ['[CLS]', 'the', 'river', 'bank', 'was', 'steep', ',', 'so', 'we', 'walked', 'along', 'the', 'path', 'to', 'the', 'old', 'wooden', 'bridge', '.', '[SEP]'];
const NQ = SEQ.length;
const REAL = SEQ.map((t, i) => (t.startsWith('[') ? -1 : i)).filter((i) => i >= 0); // 18 maskable positions
const QW = 58;
const QX = (i: number) => 22 + (i % 10) * (QW + 4);
const QY = (i: number) => 52 + Math.floor(i / 10) * 32;

/** Deterministic PRNG so the server and browser draw the same "random" mask. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function pick(seed: number, k: number) {
  const r = mulberry32(seed * 7919 + 13);
  const pool = [...REAL];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, k).sort((a, b) => a - b);
}

const SEG = ['[CLS]', 'the', 'bank', 'was', 'steep', '[SEP]', 'we', 'climbed', 'down', '[SEP]'];

const STEPS = [
  'Left-to-right sees only half',
  'Meaning lives on both sides',
  'Just remove the mask?',
  'Hide the answer in the input',
  'Pick 15% of the tokens',
  '80% [MASK], 10% random, 10% unchanged',
  'Two sentences: [CLS] A [SEP] B [SEP]',
  'One body, a small head per task',
];
const CAPTIONS = [
  'A left-to-right model, like GPT-1, builds the meaning of each token from the tokens before it. Row “bank” can only see “the bank”.',
  'Is it a river bank or a money bank? Only “steep” and “muddy” say, and they are on the right, behind the causal mask.',
  'Drop the mask and every row sees everything. But now the row that must predict the next token can simply look at it. The loss falls to zero and the model learns to copy.',
  'BERT’s fix: take the answer out of the input. Replace “steep” with [MASK], let every token see both sides, and predict what was hidden. Nothing to copy.',
  'In each training sequence BERT picks 15% of the real tokens at random to predict. Here that is 3 of 18. Only those positions are graded. Drag the rate or re-roll the draw.',
  'Of the chosen tokens, 80% become [MASK], 10% become a random token and 10% stay as they were. The model never knows which tokens to trust, so it must understand all of them.',
  'Each input is two segments, A and B, wrapped in [CLS] and [SEP]. Half the time B really follows A. The [CLS] output is trained to tell: next-sentence prediction, later found unnecessary.',
  'Pretraining builds one 110M-parameter encoder. For each task you add a tiny head on top, about 1,500 new weights, and finetune everything for a few epochs.',
];

export default function MaskedWordGame() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [rate, setRate] = useState(15);
  const [seed, setSeed] = useState(1);

  const k = Math.max(1, Math.round((rate / 100) * REAL.length));
  const chosen = pick(seed, k);
  const chosenSet = new Set(chosen);
  // Fates for the illustration: first chosen → [MASK], second → random, third → unchanged, rest → [MASK].
  const fate = (rank: number) => (rank === 1 ? 'random' : rank === 2 ? 'keep' : 'mask');
  const per512 = (rate / 100) * 512;

  const rowHot = 1; // the row for "bank"
  const matrixMode = st === 0 || st === 1 ? 'causal' : 'full';

  const readout =
    st <= 3
      ? [
          { label: 'attention', value: matrixMode === 'causal' ? 'left only' : 'both sides', kind: 'compute' as const, note: matrixMode === 'causal' ? `${(NS * (NS + 1)) / 2} of ${NS * NS} pairs` : `all ${NS * NS} pairs` },
          { label: '“bank” can see', value: matrixMode === 'causal' ? '2 tokens' : '6 tokens', kind: 'text' as const },
          { label: 'what is learned', value: st === 2 ? 'copying' : st === 3 ? 'meaning' : '—', kind: (st === 2 ? 'error' : 'token') as 'error' | 'token', note: st === 2 ? 'the answer is visible' : st === 3 ? 'the answer is hidden' : '' },
        ]
      : st <= 5
        ? [
            { label: 'mask rate', value: `${rate}%`, kind: 'token' as const, note: `${k} of ${REAL.length} tokens here` },
            { label: 'predictions per 512 tokens', value: per512.toFixed(1), kind: 'token' as const, note: `${(100 - rate)}% of positions teach nothing` },
            { label: '[MASK] / random / same', value: `${(per512 * 0.8).toFixed(1)} / ${(per512 * 0.1).toFixed(1)} / ${(per512 * 0.1).toFixed(1)}`, kind: 'text' as const, note: 'per 512-token sequence' },
          ]
        : st === 6
          ? [
              { label: 'input = sum of', value: 'token + segment + position', kind: 'text' as const },
              { label: 'NSP labels', value: '50% IsNext', kind: 'token' as const, note: '50% NotNext (random B)' },
            ]
          : [
              { label: 'BERT-Base body', value: fmt.int(109482240), kind: 'compute' as const, note: '12 layers, 768 wide' },
              { label: 'sentiment head', value: fmt.int(768 * 2 + 2), kind: 'token' as const, note: '768 × 2 + 2 biases' },
              { label: 'QA span head', value: fmt.int(2 * 768), kind: 'token' as const, note: 'start and end vectors' },
            ];

  const drawMatrix = () => {
    const cells = [];
    for (let r = 0; r < NS; r++) {
      for (let c = 0; c < NS; c++) {
        const allowed = matrixMode === 'full' || c <= r;
        const hotRow = (st === 0 || st === 1) && r === rowHot;
        const cheat = st === 2 && c === r + 1;
        const fill = cheat ? 'var(--error)' : !allowed ? 'url(#tps-hatch)' : hotRow ? 'var(--compute)' : 'var(--compute-dim)';
        const stroke = cheat ? 'var(--error)' : !allowed ? 'var(--idle)' : 'var(--compute)';
        cells.push(
          <rect key={`${r}-${c}`} x={MX + c * PITCH} y={MY + r * PITCH} width={CELL} height={CELL} rx={2}
            fill={fill} stroke={stroke} strokeWidth={0.8} className="v-cell" />,
        );
      }
    }
    return cells;
  };
  const colWord = (c: number) => (st === 3 && c === 3 ? '[MASK]' : SENT[c]);

  return (
    <Stage
      rootRef={ref}
      title="BERT: the masked-word game"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['compute', 'attention allowed'], ['idle', 'masked out'], ['token', 'predicted'], ['error', 'leak']]}
      {...s}
      readout={<Readout items={readout} />}
      controls={st === 4 || st === 5 ? (
        <>
          <Slider label="Share of tokens chosen for prediction" value={rate} min={5} max={50} step={5} onChange={setRate} format={(v) => `${v}%`} kind="token" />
          <Segmented label="Random draw" value={seed} onChange={setSeed} options={[{ value: 1, label: 'draw 1' }, { value: 2, label: 'draw 2' }, { value: 3, label: 'draw 3' }]} />
        </>
      ) : undefined}
    >
      <svg viewBox="0 0 640 350" role="img" aria-label="Attention-mask matrices for a left-to-right model and for BERT; why removing the causal mask lets a next-word model copy its answer; BERT's masking of 15% of tokens with the 80/10/10 rule; the [CLS] A [SEP] B [SEP] input with token, segment and position embeddings; and small task heads on a shared encoder.">
        {/* ── steps 0-3: the attention matrix ── */}
        <Fade show={st <= 3}>
          {SENT.map((w, i) => {
            const isMask = st === 3 && i === 3;
            const right = st === 1 && i > rowHot;
            return (
              <g key={i}>
                <rect x={40 + i * 92} y={24} width={84} height={24} rx={3}
                  fill={isMask ? 'var(--token-dim)' : right ? 'var(--ground-3)' : 'var(--ground-3)'}
                  stroke={isMask ? 'var(--token)' : right ? 'var(--text)' : 'var(--rule-2)'} strokeDasharray={isMask ? '3 3' : undefined} className="v-cell" />
                <Label x={82 + i * 92} y={40} anchor="middle" tone={1} size={11} bold={right || (st <= 1 && i === rowHot)}>{isMask ? '[MASK]' : w}</Label>
              </g>
            );
          })}
          {SENT.map((_, c) => (
            <Label key={c} x={MX + c * PITCH + CELL / 2} y={MY - 8} anchor="middle" tone={3} size={8.5}
              style={{ fill: st === 3 && c === 3 ? 'var(--token)' : undefined }}>{colWord(c).slice(0, 6)}</Label>
          ))}
          {SENT.map((w, r) => (
            <Label key={r} x={MX - 8} y={MY + r * PITCH + 16} anchor="end" tone={(st <= 1 && r === rowHot) ? 1 : 3} size={9.5}
              bold={st <= 1 && r === rowHot} style={{ fill: st === 3 && r === 3 ? 'var(--token)' : undefined }}>{colWord(r)}</Label>
          ))}
          {drawMatrix()}
          <Label x={MX + (NS * PITCH) / 2} y={MY + NS * PITCH + 16} anchor="middle" tone={3} size={9}>row = token being built · column = token it may read</Label>
        </Fade>

        <Fade show={st === 0}>
          <Label x={340} y={118} tone={1} size={11} bold mono={false}>Causal mask (GPT-1, 2018)</Label>
          <Label x={340} y={140} tone={2} size={10} mono={false}>Each row reads only itself and the rows above.</Label>
          <Label x={340} y={156} tone={2} size={10} mono={false}>That is what makes next-word training honest</Label>
          <Label x={340} y={172} tone={2} size={10} mono={false}>(lesson 1): no token can see its own answer.</Label>
          <Label x={340} y={204} tone={1} size={10} mono={false}>The cost: the vector built for “bank”</Label>
          <Label x={340} y={220} tone={1} size={10} mono={false}>is made from “the bank” and nothing else.</Label>
        </Fade>

        <Fade show={st === 1}>
          <Label x={340} y={118} tone={1} size={11} bold mono={false}>Which bank?</Label>
          <Label x={340} y={140} tone={2} size={10} mono={false}>“the bank was steep and muddy” → a river bank</Label>
          <Label x={340} y={156} tone={2} size={10} mono={false}>“the bank was closed on sunday” → a money bank</Label>
          <Label x={340} y={186} tone={2} size={10} mono={false}>Everything that decides it is to the right,</Label>
          <Label x={340} y={202} tone={2} size={10} mono={false}>in the hatched cells “bank” is not allowed to read.</Label>
          <Label x={340} y={232} tone={3} size={9.5} mono={false}>Classifying, tagging and answering questions</Label>
          <Label x={340} y={247} tone={3} size={9.5} mono={false}>all want every token to know its whole context.</Label>
        </Fade>

        <Fade show={st === 2}>
          <Label x={340} y={118} tone={1} size={11} bold mono={false}>Bidirectional, but still guessing the next word</Label>
          <Label x={340} y={140} tone={2} size={10} mono={false}>Row “bank” must predict “was”… and the red</Label>
          <Label x={340} y={156} tone={2} size={10} mono={false}>cell lets it read “was” directly. So can every row.</Label>
          <Label x={340} y={186} kind="error" size={10.5} bold mono={false}>The model learns to copy one position over.</Label>
          <Label x={340} y={204} tone={2} size={10} mono={false}>Loss drops to almost zero; nothing about</Label>
          <Label x={340} y={220} tone={2} size={10} mono={false}>language is learned. (With several layers the</Label>
          <Label x={340} y={236} tone={2} size={10} mono={false}>answer leaks even through indirect paths.)</Label>
        </Fade>

        <Fade show={st === 3}>
          <Label x={340} y={118} tone={1} size={11} bold mono={false}>Hide the answer, not the context</Label>
          <Label x={340} y={140} tone={2} size={10} mono={false}>“steep” is replaced by [MASK] in the input.</Label>
          <Label x={340} y={156} tone={2} size={10} mono={false}>Every token now reads both sides freely, and</Label>
          <Label x={340} y={172} tone={2} size={10} mono={false}>the [MASK] row predicts the hidden word:</Label>
          <rect x={340} y={186} width={200} height={28} rx={3} fill="var(--token-dim)" stroke="var(--token)" />
          <Label x={440} y={204} anchor="middle" tone={1} size={11}>{'[MASK] → steep'}</Label>
          <Label x={340} y={236} tone={3} size={9.5} mono={false}>A fill-in-the-blank test, which reading researchers</Label>
          <Label x={340} y={251} tone={3} size={9.5} mono={false}>call a cloze test. BERT calls it a masked LM.</Label>
        </Fade>

        {/* ── steps 4-5: the masking procedure ── */}
        <Fade show={st === 4 || st === 5}>
          <Label x={22} y={36} tone={1} size={11} bold mono={false}>One training sequence (special tokens are never chosen)</Label>
          {SEQ.map((t, i) => {
            const isChosen = chosenSet.has(i);
            const rank = chosen.indexOf(i);
            const f = st === 5 && isChosen ? fate(rank) : null;
            const shown = f === 'mask' ? '[MASK]' : f === 'random' ? 'piano' : t;
            const special = t.startsWith('[');
            return (
              <g key={i}>
                <rect x={QX(i)} y={QY(i)} width={QW} height={24} rx={3}
                  fill={isChosen ? (f === 'random' ? 'var(--error-dim)' : 'var(--token-dim)') : special ? 'var(--ground-2)' : 'var(--ground-3)'}
                  stroke={isChosen ? (f === 'random' ? 'var(--error)' : 'var(--token)') : 'var(--rule-2)'}
                  strokeDasharray={isChosen && f !== 'keep' ? '3 3' : undefined} className="v-cell" />
                <Label x={QX(i) + QW / 2} y={QY(i) + 16} anchor="middle" tone={special ? 3 : 1} size={shown.length > 6 ? 9 : 10.5}>{shown}</Label>
              </g>
            );
          })}
        </Fade>

        <Fade show={st === 4}>
          {chosen.map((i) => (
            <circle key={i} cx={QX(i) + QW / 2} cy={QY(i) - 5} r={2.5} fill="var(--token)" />
          ))}
          <Label x={22} y={176} tone={1} size={10.5} mono={false}>{`Chosen at random: ${chosen.map((i) => `“${SEQ[i]}”`).join(', ')}.`}</Label>
          <Label x={22} y={196} tone={2} size={10} mono={false}>{`The loss is computed only at these ${k} positions. The other ${REAL.length - k} tokens are context, not exam questions.`}</Label>
          <Label x={22} y={226} tone={2} size={10} mono={false}>Why 15%? Too few and each pass teaches almost nothing. Too many and too little context</Label>
          <Label x={22} y={242} tone={2} size={10} mono={false}>is left to guess from. The BERT authors picked 15%; later work found higher rates can also work.</Label>
          <Label x={22} y={272} tone={3} size={9.5}>{`at ${rate}%: ${(rate / 100).toFixed(2)} × 512 = ${per512.toFixed(1)} graded positions per 512-token sequence`}</Label>
        </Fade>

        <Fade show={st === 5}>
          <Label x={22} y={168} tone={1} size={10.5} bold mono={false}>What happens to each chosen token (this draw shows one of each)</Label>
          {[
            { k: 'token', w: 0.8, name: '80% → [MASK]', why: 'the normal case: predict the hidden word' },
            { k: 'error', w: 0.1, name: '10% → random token', why: 'e.g. “piano”: the input may be lying' },
            { k: 'text', w: 0.1, name: '10% → unchanged', why: 'still predicted: the input may be honest' },
          ].map((row, i) => {
            const x0 = 22 + [0, 0.8, 0.9][i] * 596;
            return (
              <g key={row.name}>
                <rect x={x0} y={180} width={row.w * 596 - 3} height={20} rx={2}
                  fill={row.k === 'token' ? 'var(--token)' : row.k === 'error' ? 'var(--error)' : 'var(--ground-3)'}
                  stroke={row.k === 'token' ? 'var(--token)' : row.k === 'error' ? 'var(--error)' : 'var(--text-2)'} />
                <Label x={32} y={226 + i * 18} tone={1} size={10}>{row.name}</Label>
                <Label x={200} y={226 + i * 18} tone={2} size={10} mono={false}>{row.why}</Label>
              </g>
            );
          })}
          <Label x={22} y={296} tone={2} size={10} mono={false}>[MASK] never appears when the model is used later. If it were the only signal, the model could</Label>
          <Label x={22} y={312} tone={2} size={10} mono={false}>learn to care only about [MASK] positions. Mixing in random and unchanged tokens means it can never</Label>
          <Label x={22} y={328} tone={2} size={10} mono={false}>tell which inputs are real, so it has to keep a good representation of every token.</Label>
        </Fade>

        {/* ── step 6: input format and NSP ── */}
        <Fade show={st === 6}>
          {SEG.map((t, i) => {
            const x = 70 + i * 56;
            const segB = i >= 6;
            return (
              <g key={i}>
                <rect x={x} y={34} width={52} height={22} rx={3} fill="var(--ground-3)" stroke={t.startsWith('[') ? 'var(--text-2)' : 'var(--rule-2)'} />
                <Label x={x + 26} y={49} anchor="middle" tone={1} size={t.length > 6 ? 8.5 : 10}>{t}</Label>
                <Label x={x + 26} y={78} anchor="middle" tone={2} size={9.5}>{`E_${t.startsWith('[') ? t.slice(1, 4) : t.slice(0, 5)}`}</Label>
                <Label x={x + 26} y={98} anchor="middle" tone={segB ? 1 : 2} size={9.5} bold={segB}>{segB ? 'E_B' : 'E_A'}</Label>
                <Label x={x + 26} y={118} anchor="middle" tone={3} size={9.5}>{`E_${i}`}</Label>
              </g>
            );
          })}
          <Label x={62} y={78} anchor="end" tone={3} size={9}>token</Label>
          <Label x={62} y={98} anchor="end" tone={3} size={9}>segment</Label>
          <Label x={62} y={118} anchor="end" tone={3} size={9}>position</Label>
          <line x1={70} x2={626} y1={126} y2={126} className="v-rule faint" />
          <Label x={62} y={142} anchor="end" tone={1} size={9}>sum</Label>
          <Label x={70} y={142} tone={2} size={9.5} mono={false}>each column is added together into one 768-number vector, then the 12 layers run</Label>
          <rect x={70} y={162} width={160} height={30} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" />
          <Label x={150} y={181} anchor="middle" tone={1} size={10}>[CLS] output</Label>
          <Arrow x1={232} y1={177} x2={282} y2={177} kind="token" />
          <rect x={288} y={162} width={200} height={30} rx={3} fill="var(--token-dim)" stroke="var(--token)" />
          <Label x={388} y={181} anchor="middle" tone={1} size={10}>IsNext or NotNext?</Label>
          <Label x={70} y={222} tone={2} size={10} mono={false}>Next-sentence prediction: half the time B is the sentence that really followed A, half the</Label>
          <Label x={70} y={238} tone={2} size={10} mono={false}>time a random sentence from elsewhere. The idea was to teach relations between sentences.</Label>
          <Label x={70} y={268} tone={1} size={10} bold mono={false}>RoBERTa (2019) trained without it and matched or slightly beat BERT downstream:</Label>
          <Label x={70} y={284} tone={1} size={10} mono={false}>the masked-word game was doing the real work.</Label>
        </Fade>

        {/* ── step 7: heads ── */}
        <Fade show={st === 7}>
          <rect x={40} y={60} width={220} height={210} rx={5} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1.2} />
          {Array.from({ length: 12 }, (_, i) => (
            <rect key={i} x={56} y={76 + i * 15} width={188} height={11} rx={2} fill="var(--compute)" opacity={0.35 + i * 0.04} />
          ))}
          <Label x={150} y={52} anchor="middle" tone={1} size={10.5} bold mono={false}>pretrained BERT-Base</Label>
          <Label x={150} y={290} anchor="middle" tone={2} size={9.5} mono={false}>12 layers · 768 wide · 110M weights</Label>
          {[
            { y: 74, name: 'sentiment', head: '[CLS] → 2 classes', n: '1,538 weights' },
            { y: 146, name: 'question answering', head: 'each token → start? end?', n: '1,536 weights' },
            { y: 218, name: 'named entities', head: 'each token → a tag', n: '768 × tags' },
          ].map((h) => (
            <g key={h.name}>
              <Arrow x1={264} y1={h.y + 20} x2={334} y2={h.y + 20} kind="token" />
              <rect x={340} y={h.y} width={250} height={42} rx={4} fill="var(--token-dim)" stroke="var(--token)" />
              <Label x={352} y={h.y + 17} tone={1} size={10.5} bold mono={false}>{h.name}</Label>
              <Label x={352} y={h.y + 33} tone={2} size={9.5}>{h.head}</Label>
              <Label x={582} y={h.y + 17} anchor="end" kind="token" size={9}>{h.n}</Label>
            </g>
          ))}
          <Label x={340} y={292} tone={3} size={9.5} mono={false}>Finetune all weights for 2–4 epochs. The paper: at most</Label>
          <Label x={340} y={307} tone={3} size={9.5} mono={false}>1 hour on one Cloud TPU per task, versus 4 days to pretrain.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
