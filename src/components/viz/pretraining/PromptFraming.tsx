import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Label, Arrow, Fade, fmt } from '../lib/svg';

// The four GPT-2 sizes. Parameter counts are the released checkpoints'
// (the paper's table printed 117M, 345M, 762M and 1542M).
const SIZES = [
  { id: 0, name: '124M', paper: '117M', L: 12, d: 768, h: 12, params: 124439808, lambadaAcc: 45.99, lambadaPpl: 35.13, wt103: 37.5 },
  { id: 1, name: '355M', paper: '345M', L: 24, d: 1024, h: 16, params: 354823168, lambadaAcc: 55.48, lambadaPpl: 15.6, wt103: 26.37 },
  { id: 2, name: '774M', paper: '762M', L: 36, d: 1280, h: 20, params: 774030080, lambadaAcc: 60.12, lambadaPpl: 10.87, wt103: 22.05 },
  { id: 3, name: '1.5B', paper: '1542M', L: 48, d: 1600, h: 25, params: 1557611200, lambadaAcc: 63.24, lambadaPpl: 8.63, wt103: 17.48 },
];

type Task = 'summarise' | 'translate' | 'answer' | 'continue';
const TASKS: Record<Task, { prompt: string[]; out: string; note: string }> = {
  summarise: {
    prompt: ['The city council met on Tuesday and voted', '7–2 to turn the old rail yard into a park…', 'TL;DR:'],
    out: ' The rail yard will become a park.',
    note: 'the paper appended “TL;DR:” after news articles',
  },
  translate: {
    prompt: ['good morning = bonjour', 'thank you = merci', 'see you tomorrow ='],
    out: ' à demain',
    note: 'the paper primed with example pairs “english = french”',
  },
  answer: {
    prompt: ['Q: What is the capital of Japan?', 'A:'],
    out: ' Tokyo',
    note: 'question-answer pairs, then a new question',
  },
  continue: {
    prompt: ['Once upon a time, a small robot', 'woke up in an empty workshop and'],
    out: ' looked around for its maker.',
    note: 'no framing at all: just keep writing',
  },
};

const STEPS = [
  'BERT’s bill: a head and a dataset per task',
  'The web already writes tasks down',
  'WebText: links people upvoted',
  'Byte-level BPE: nothing is unknown',
  'One decoder, four sizes',
  'Phrase the task as text',
  'Bigger is smoothly better',
  'Released in stages',
];
const CAPTIONS = [
  'With BERT, every task needs a labelled dataset, a new head and a finetuning run. And none of them can write a sentence.',
  'Web text is full of tasks written out in plain language: translations, questions with their answers, summaries after “TL;DR:”. A model that predicts web text well must learn to do them.',
  'WebText: every outbound link from Reddit with at least 3 karma, a cheap signal that a person found the page worth reading. About 45M links became 8M documents and 40 GB of text.',
  'GPT-2 runs BPE on raw bytes: 256 byte tokens, 50,000 merges and one end-of-text token make 50,257. Any string at all can be encoded, so there is no unknown token.',
  'One decoder-only transformer in four sizes, from 12 layers and 768 wide to 48 layers and 1,600 wide. LayerNorm moves to the input of each sub-block: pre-norm.',
  'No heads, no finetuning. The prompt decides the job. The same weights summarise, translate, answer and continue, depending only on how the task is written.',
  'Zero-shot, the 1.5B model set state-of-the-art results on 7 of 8 language-modelling benchmarks, and every metric improved steadily with size. The largest still underfit WebText.',
  'Citing misuse concerns, OpenAI released the smallest model in February 2019 and the full 1.5B model only in November, after publishing each larger size in turn.',
];

// results chart geometry
const GX0 = 70;
const GX1 = 290;
const lx = (p: number) => GX0 + ((Math.log10(p) - Math.log10(1e8)) / (Math.log10(2e9) - Math.log10(1e8))) * (GX1 - GX0);

export default function PromptFraming() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [size, setSize] = useState(3);
  const [task, setTask] = useState<Task>('summarise');
  const m = SIZES[size];

  // stack geometry tweens smoothly between sizes
  const g = useTweens({ L: m.L, d: m.d }, 600);
  const stackW = 60 + (g.d / 1600) * 140;
  const layerH = 200 / 48;

  const kvPerToken = 2 * m.L * m.d * 2; // K and V, fp16
  const readout =
    st === 4 || st === 6
      ? [
          { label: 'parameters', value: fmt.si(m.params), kind: 'compute' as const, note: `paper’s table: ${m.paper}` },
          { label: 'layers × width', value: `${m.L} × ${fmt.int(m.d)}`, kind: 'text' as const, note: `${m.h} heads of 64` },
          { label: 'weights in FP16', value: fmt.bytes(m.params * 2), kind: 'memory' as const },
          { label: 'KV cache per token', value: `${fmt.int(kvPerToken / 1024)} KiB`, kind: 'memory' as const, note: `2 × ${m.L} × ${fmt.int(m.d)} × 2 bytes` },
        ]
      : st === 2
        ? [
            { label: 'links (≥ 3 karma)', value: '45M', kind: 'text' as const },
            { label: 'documents kept', value: '~8M', kind: 'token' as const, note: 'deduplicated, cleaned, Wikipedia removed' },
            { label: 'text', value: '40 GB', kind: 'token' as const },
          ]
        : st === 3
          ? [
              { label: 'byte tokens', value: '256', kind: 'text' as const },
              { label: 'merges', value: '50,000', kind: 'text' as const },
              { label: '<|endoftext|>', value: '1', kind: 'text' as const },
              { label: 'vocabulary', value: '50,257', kind: 'token' as const, note: 'no unknown token possible' },
            ]
          : st === 5
            ? [
                { label: 'task', value: task, kind: 'token' as const, note: TASKS[task].note },
                { label: 'new weights', value: '0', kind: 'token' as const, note: 'no head, no finetuning' },
                { label: 'labelled examples', value: '0', kind: 'token' as const, note: 'zero-shot' },
              ]
            : st === 7
              ? [
                  { label: 'Feb 2019', value: '124M', kind: 'text' as const, note: 'with the paper' },
                  { label: 'May 2019', value: '355M', kind: 'text' as const },
                  { label: 'Aug 2019', value: '774M', kind: 'text' as const },
                  { label: 'Nov 2019', value: '1.5B', kind: 'token' as const, note: 'the full model' },
                ]
              : [
                  { label: 'per task (BERT)', value: 'data + head + run', kind: 'error' as const },
                  { label: 'per task (GPT-2)', value: 'a prompt', kind: 'token' as const },
                ];

  return (
    <Stage
      rootRef={ref}
      title="GPT-2: one decoder, many jobs"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['compute', 'model'], ['token', 'generated / free'], ['error', 'per-task cost'], ['memory', 'memory']]}
      {...s}
      readout={<Readout items={readout} />}
      controls={st === 4 || st === 6 ? (
        <Segmented label="Model size" value={size} onChange={setSize} options={SIZES.map((z) => ({ value: z.id, label: z.name }))} />
      ) : st === 5 ? (
        <Segmented label="How the task is written" value={task} onChange={setTask}
          options={(Object.keys(TASKS) as Task[]).map((t) => ({ value: t, label: t }))} />
      ) : undefined}
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="GPT-2: BERT needs a head and labelled data per task; web text already contains tasks; the WebText dataset of 8 million documents; byte-level BPE with 50,257 tokens; four model sizes with pre-norm blocks; the same model summarising, translating, answering or continuing depending on the prompt; zero-shot results that improve with size; and the staged release during 2019.">
        {/* ── step 0: BERT's per-task bill ── */}
        <Fade show={st === 0}>
          <rect x={40} y={90} width={150} height={120} rx={5} fill="var(--compute-dim)" stroke="var(--compute)" />
          <Label x={115} y={146} anchor="middle" tone={1} size={11} bold mono={false}>pretrained BERT</Label>
          <Label x={115} y={164} anchor="middle" tone={3} size={9.5} mono={false}>reads, cannot write</Label>
          {['sentiment', 'entailment', 'question answering', 'named entities'].map((t, i) => (
            <g key={t}>
              <Arrow x1={194} y1={150} x2={258} y2={58 + i * 62} kind="error" curve={i < 2 ? -6 : 6} />
              <rect x={262} y={44 + i * 62} width={130} height={30} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x={327} y={63 + i * 62} anchor="middle" tone={1} size={10} mono={false}>{t}</Label>
              <rect x={402} y={44 + i * 62} width={200} height={30} rx={3} fill="var(--error-dim)" stroke="var(--error)" />
              <Label x={502} y={57 + i * 62} anchor="middle" tone={1} size={9.5} mono={false}>labelled dataset + new head</Label>
              <Label x={502} y={69 + i * 62} anchor="middle" tone={2} size={9} mono={false}>+ a finetuning run</Label>
            </g>
          ))}
          <Label x={40} y={306} tone={2} size={10} mono={false}>A new task means new labels and a new copy of the model. A task with no labelled data is out of reach.</Label>
        </Fade>

        {/* ── step 1: tasks inside web text ── */}
        <Fade show={st === 1}>
          <Label x={40} y={34} tone={3} size={9.5} mono={false}>illustrative snippets of the kind of text found on the web</Label>
          {[
            { y: 50, src: 'a language-learning forum', lines: ['“Je suis fatigué” just means “I am tired”, it’s', 'what you say after a long day.'], task: 'translation' },
            { y: 130, src: 'a quiz page', lines: ['Q: What is the capital of Japan?', 'A: Tokyo.'], task: 'question answering' },
            { y: 210, src: 'a long blog post, at the end', lines: ['…so after three weeks I finally fixed it.', 'TL;DR: the update broke my wifi driver.'], task: 'summarisation' },
          ].map((c) => (
            <g key={c.task}>
              <rect x={40} y={c.y} width={420} height={62} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x={52} y={c.y + 16} tone={3} size={9} mono={false}>{c.src}</Label>
              <Label x={52} y={c.y + 34} tone={1} size={10.5} mono={false}>{c.lines[0]}</Label>
              <Label x={52} y={c.y + 51} tone={1} size={10.5} mono={false}>{c.lines[1]}</Label>
              <Arrow x1={464} y1={c.y + 31} x2={494} y2={c.y + 31} kind="token" />
              <rect x={498} y={c.y + 18} width={116} height={26} rx={3} fill="var(--token-dim)" stroke="var(--token)" />
              <Label x={556} y={c.y + 35} anchor="middle" tone={1} size={9.5} mono={false}>{c.task}</Label>
            </g>
          ))}
          <Label x={40} y={300} tone={2} size={10} mono={false}>Predicting the next token of such text well means doing the task. Nobody labelled it as one.</Label>
        </Fade>

        {/* ── step 2: WebText funnel ── */}
        <Fade show={st === 2}>
          {[
            { w: 560, label: 'every outbound link posted on Reddit', sub: 'pages that people shared' },
            { w: 420, label: 'links with at least 3 karma: ~45M', sub: 'a human vote that the page was worth reading' },
            { w: 290, label: 'text extracted, deduplicated, cleaned', sub: 'all Wikipedia pages removed (test-set overlap)' },
            { w: 190, label: '~8M documents · 40 GB', sub: 'WebText' },
          ].map((r, i) => (
            <g key={i}>
              <rect x={320 - r.w / 2} y={44 + i * 62} width={r.w} height={40} rx={4}
                fill={i === 3 ? 'var(--token-dim)' : 'var(--ground-3)'} stroke={i === 3 ? 'var(--token)' : 'var(--rule-2)'} />
              <Label x={320} y={61 + i * 62} anchor="middle" tone={1} size={10.5} bold={i === 3} mono={false}>{r.label}</Label>
              <Label x={320} y={76 + i * 62} anchor="middle" tone={3} size={9} mono={false}>{r.sub}</Label>
              {i < 3 && <Arrow x1={320} y1={86 + i * 62} x2={320} y2={104 + i * 62} muted />}
            </g>
          ))}
          <Label x={320} y={310} anchor="middle" tone={2} size={10} mono={false}>Diverse on purpose: news, how-tos, fiction, forums, code. Curated cheaply, by votes.</Label>
        </Fade>

        {/* ── step 3: byte-level BPE ── */}
        <Fade show={st === 3}>
          <Label x={40} y={40} tone={1} size={11} bold mono={false}>Any text is a sequence of bytes (UTF-8)</Label>
          {[['n', '6E'], ['a', '61'], ['ï', 'C3'], ['', 'AF'], ['v', '76'], ['e', '65'], ['·', '20'], ['🙂', 'F0'], ['', '9F'], ['', '99'], ['', '82']].map(([ch, hx], i) => (
            <g key={i}>
              <rect x={40 + i * 50} y={56} width={44} height={40} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x={62 + i * 50} y={73} anchor="middle" tone={1} size={11}>{ch}</Label>
              <Label x={62 + i * 50} y={89} anchor="middle" tone={3} size={9}>{hx}</Label>
            </g>
          ))}
          <Label x={40} y={116} tone={3} size={9.5} mono={false}>“naïve 🙂” is 11 bytes: ï takes two, the emoji takes four. Every one of the 256 possible bytes is a token.</Label>
          <rect x={40} y={140} width={150} height={44} rx={4} fill="var(--ground-3)" stroke="var(--text-2)" />
          <Label x={115} y={160} anchor="middle" tone={1} size={12} bold>256</Label>
          <Label x={115} y={176} anchor="middle" tone={3} size={9} mono={false}>byte tokens</Label>
          <Label x={204} y={167} anchor="middle" tone={1} size={14}>+</Label>
          <rect x={218} y={140} width={150} height={44} rx={4} fill="var(--ground-3)" stroke="var(--text-2)" />
          <Label x={293} y={160} anchor="middle" tone={1} size={12} bold>50,000</Label>
          <Label x={293} y={176} anchor="middle" tone={3} size={9} mono={false}>learned merges</Label>
          <Label x={382} y={167} anchor="middle" tone={1} size={14}>+</Label>
          <rect x={396} y={140} width={80} height={44} rx={4} fill="var(--ground-3)" stroke="var(--text-2)" />
          <Label x={436} y={160} anchor="middle" tone={1} size={12} bold>1</Label>
          <Label x={436} y={176} anchor="middle" tone={3} size={9} mono={false}>end of text</Label>
          <Label x={490} y={167} anchor="middle" tone={1} size={14}>=</Label>
          <rect x={504} y={140} width={100} height={44} rx={4} fill="var(--token-dim)" stroke="var(--token)" />
          <Label x={554} y={167} anchor="middle" tone={1} size={13} bold>50,257</Label>
          <Label x={40} y={222} tone={2} size={10} mono={false}>Common words become single tokens; rare words, misspellings, code and emoji fall back to</Label>
          <Label x={40} y={238} tone={2} size={10} mono={false}>shorter pieces, and in the worst case to single bytes. BERT’s WordPiece has an [UNK] token for</Label>
          <Label x={40} y={254} tone={2} size={10} mono={false}>what it cannot spell; GPT-2 can encode any string at all. That is what lets it read the whole web.</Label>
          <Label x={40} y={284} tone={3} size={9.5} mono={false}>(GPT-2 also stops merges from crossing character types, so “dog.” “dog!” “dog?” do not each get a token.)</Label>
        </Fade>

        {/* ── step 4: four sizes + pre-norm block ── */}
        <Fade show={st === 4}>
          <Label x={150} y={30} anchor="middle" tone={1} size={10.5} bold mono={false}>{`GPT-2 ${m.name}: ${m.L} layers × ${m.d}`}</Label>
          {Array.from({ length: 48 }, (_, i) => (
            <rect key={i} x={150 - stackW / 2} y={250 - (i + 1) * layerH} width={stackW} height={layerH - 1}
              fill="var(--compute)" opacity={i < g.L ? 0.85 : 0} rx={0.8} />
          ))}
          <line x1={40} x2={260} y1={252} y2={252} className="v-rule" />
          <Label x={150} y={270} anchor="middle" tone={3} size={9.5} mono={false}>height = layers · width = hidden size</Label>
          <Label x={150} y={290} anchor="middle" tone={2} size={9.5}>{`context 1,024 · vocab 50,257`}</Label>
          {/* pre-norm block */}
          <Label x={470} y={30} anchor="middle" tone={1} size={10.5} bold mono={false}>one block, pre-norm</Label>
          <line x1={470} x2={470} y1={290} y2={44} stroke="var(--text-2)" strokeWidth={2} />
          <Label x={478} y={292} tone={3} size={9}>residual stream in</Label>
          <Label x={478} y={52} tone={3} size={9}>out</Label>
          {[{ y: 220, n: 'attention' }, { y: 120, n: 'MLP' }].map((b) => (
            <g key={b.n}>
              <Arrow x1={470} y1={b.y + 34} x2={396} y2={b.y + 34} muted />
              <rect x={330} y={b.y + 22} width={64} height={24} rx={3} fill="var(--ground-3)" stroke="var(--text-2)" />
              <Label x={362} y={b.y + 38} anchor="middle" tone={1} size={10}>LayerNorm</Label>
              <Arrow x1={362} y1={b.y + 20} x2={362} y2={b.y - 2} muted />
              <rect x={322} y={b.y - 26} width={80} height={24} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" />
              <Label x={362} y={b.y - 10} anchor="middle" tone={1} size={10}>{b.n}</Label>
              <Arrow x1={404} y1={b.y - 14} x2={464} y2={b.y - 14} muted />
              <circle cx={470} cy={b.y - 14} r={7} fill="var(--ground-2)" stroke="var(--text-2)" />
              <Label x={470} y={b.y - 10} anchor="middle" tone={1} size={11}>+</Label>
            </g>
          ))}
          <Label x={520} y={180} tone={2} size={9.5} mono={false}>Normalise the copy</Label>
          <Label x={520} y={194} tone={2} size={9.5} mono={false}>going in, not the</Label>
          <Label x={520} y={208} tone={2} size={9.5} mono={false}>stream itself: the</Label>
          <Label x={520} y={222} tone={2} size={9.5} mono={false}>path from input to</Label>
          <Label x={520} y={236} tone={2} size={9.5} mono={false}>output stays clean.</Label>
        </Fade>

        {/* ── step 5: prompt framing ── */}
        <Fade show={st === 5}>
          <Label x={40} y={36} tone={3} size={9.5} mono={false}>prompt (what you write)</Label>
          <rect x={40} y={46} width={380} height={96} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" />
          {TASKS[task].prompt.map((line, i) => (
            <Label key={`${task}-${i}`} x={54} y={70 + i * 22} tone={1} size={10.5}>{line}</Label>
          ))}
          <Arrow x1={230} y1={146} x2={230} y2={178} kind="compute" />
          <rect x={150} y={182} width={160} height={40} rx={5} fill="var(--compute-dim)" stroke="var(--compute)" />
          <Label x={230} y={200} anchor="middle" tone={1} size={10.5} bold mono={false}>the same GPT-2</Label>
          <Label x={230} y={214} anchor="middle" tone={3} size={9} mono={false}>no head, no finetuning</Label>
          <Arrow x1={230} y1={226} x2={230} y2={256} kind="token" />
          <rect x={40} y={260} width={380} height={30} rx={4} fill="var(--token-dim)" stroke="var(--token)" />
          <Label x={54} y={279} tone={1} size={10.5}>{TASKS[task].out}</Label>
          <Label x={40} y={306} tone={3} size={9} mono={false}>outputs are illustrative, not sampled from GPT-2</Label>
          <Label x={450} y={70} tone={1} size={10.5} bold mono={false}>The prompt is the task</Label>
          <Label x={450} y={92} tone={2} size={9.5} mono={false}>A model of web text has</Label>
          <Label x={450} y={106} tone={2} size={9.5} mono={false}>seen “TL;DR:” followed by</Label>
          <Label x={450} y={120} tone={2} size={9.5} mono={false}>summaries thousands of</Label>
          <Label x={450} y={134} tone={2} size={9.5} mono={false}>times. Writing the same</Label>
          <Label x={450} y={148} tone={2} size={9.5} mono={false}>pattern asks for one.</Label>
          <Label x={450} y={180} tone={2} size={9.5} mono={false}>{TASKS[task].note.split(' ').slice(0, 4).join(' ')}</Label>
          <Label x={450} y={194} tone={2} size={9.5} mono={false}>{TASKS[task].note.split(' ').slice(4).join(' ')}</Label>
        </Fade>

        {/* ── step 6: results vs size ── */}
        <Fade show={st === 6}>
          {[
            { x0: 0, title: 'LAMBADA accuracy (%) ↑', key: 'lambadaAcc' as const, lo: 40, hi: 70, prev: 59.23, prevLabel: 'previous best 59.23' },
            { x0: 320, title: 'WikiText-103 perplexity ↓', key: 'wt103' as const, lo: 10, hi: 45, prev: null, prevLabel: '' },
          ].map((pnl) => {
            const yy = (v: number) => 250 - ((v - pnl.lo) / (pnl.hi - pnl.lo)) * 180;
            const pts = SIZES.map((z) => `${pnl.x0 + lx(z.params)},${yy(z[pnl.key])}`).join(' ');
            return (
              <g key={pnl.title}>
                <Label x={pnl.x0 + GX0} y={40} tone={1} size={10.5} bold mono={false}>{pnl.title}</Label>
                <line x1={pnl.x0 + GX0} x2={pnl.x0 + GX1} y1={250} y2={250} className="v-rule" />
                <line x1={pnl.x0 + GX0} x2={pnl.x0 + GX0} y1={250} y2={64} className="v-rule" />
                {[pnl.lo, (pnl.lo + pnl.hi) / 2, pnl.hi].map((v) => (
                  <Label key={v} x={pnl.x0 + GX0 - 6} y={yy(v) + 3} anchor="end" tone={3} size={8.5}>{v.toFixed(0)}</Label>
                ))}
                {pnl.prev !== null && (
                  <g>
                    <line x1={pnl.x0 + GX0} x2={pnl.x0 + GX1} y1={yy(pnl.prev)} y2={yy(pnl.prev)} stroke="var(--text-3)" strokeDasharray="4 4" />
                    <Label x={pnl.x0 + GX0 + 4} y={yy(pnl.prev) - 5} tone={3} size={8.5}>{pnl.prevLabel}</Label>
                  </g>
                )}
                <polyline points={pts} fill="none" stroke="var(--token)" strokeWidth={1.8} />
                {SIZES.map((z) => (
                  <g key={z.id}>
                    <circle cx={pnl.x0 + lx(z.params)} cy={yy(z[pnl.key])} r={z.id === size ? 6 : 3.5}
                      fill={z.id === size ? 'var(--token)' : 'var(--ground-2)'} stroke="var(--token)" strokeWidth={1.5} style={{ transition: 'r .3s' }} />
                    <Label x={pnl.x0 + lx(z.params)} y={yy(z[pnl.key]) + (pnl.key === 'wt103' ? -10 : 18)} anchor="middle" tone={z.id === size ? 1 : 3} size={9} bold={z.id === size}>
                      {z[pnl.key].toFixed(pnl.key === 'wt103' ? 1 : 1)}
                    </Label>
                    <Label x={pnl.x0 + lx(z.params)} y={266} anchor="middle" tone={3} size={8.5}>{z.name}</Label>
                  </g>
                ))}
                <Label x={pnl.x0 + (GX0 + GX1) / 2} y={284} anchor="middle" tone={3} size={9}>parameters (log scale)</Label>
              </g>
            );
          })}
          <Label x={320} y={312} anchor="middle" tone={2} size={10} mono={false}>Zero-shot: no training on either benchmark. Each size is better than the last, with no sign of stopping.</Label>
        </Fade>

        {/* ── step 7: staged release ── */}
        <Fade show={st === 7}>
          <line x1={60} x2={590} y1={160} y2={160} className="v-rule" />
          {[
            { x: 80, m: 'Feb 2019', s: '124M', n: 'paper + smallest model' },
            { x: 240, m: 'May 2019', s: '355M', n: '' },
            { x: 400, m: 'Aug 2019', s: '774M', n: 'with a report on' },
            { x: 560, m: 'Nov 2019', s: '1.5B', n: 'the full model' },
          ].map((e, i) => (
            <g key={e.m}>
              <circle cx={e.x} cy={160} r={7} fill={i === 3 ? 'var(--token)' : 'var(--ground-3)'} stroke={i === 3 ? 'var(--token)' : 'var(--text-2)'} strokeWidth={1.5} />
              <Label x={e.x} y={138} anchor="middle" tone={1} size={11} bold>{e.s}</Label>
              <Label x={e.x} y={188} anchor="middle" tone={2} size={10}>{e.m}</Label>
              {e.n && <Label x={e.x} y={206} anchor="middle" tone={3} size={9} mono={false}>{e.n}</Label>}
              {i === 2 && <Label x={e.x} y={220} anchor="middle" tone={3} size={9} mono={false}>release strategies</Label>}
            </g>
          ))}
          <Label x={60} y={60} tone={1} size={11} bold mono={false}>“Too dangerous to release”?</Label>
          <Label x={60} y={80} tone={2} size={10} mono={false}>OpenAI worried the text was good enough for spam and fake news at scale, so it released</Label>
          <Label x={60} y={96} tone={2} size={10} mono={false}>the weights in stages while it watched for misuse. It was the first big public argument</Label>
          <Label x={60} y={112} tone={2} size={10} mono={false}>about whether to publish a model’s weights at all; Module 04 picks up the open-weights story.</Label>
          <Label x={60} y={268} tone={3} size={9.5} mono={false}>Today the 124M model is the “hello world” of language models: 250 MB in FP16, and it runs on a laptop.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
