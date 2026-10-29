import { useEffect, useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Move } from '../lib/svg';

// One demonstration, tokenized with a ChatML-style template (the format Qwen2.5 uses).
// Per-token losses are illustrative but shaped like the real thing: template tokens are almost
// free to predict, the user's words are expensive, the response sits in between.
type Role = 'tmpl' | 'user' | 'resp';
const TOKENS: { t: string; role: Role; loss: number | null }[] = [
  { t: '<|im_start|>', role: 'tmpl', loss: null },
  { t: 'user', role: 'tmpl', loss: 0.02 },
  { t: '↵', role: 'tmpl', loss: 0.01 },
  { t: 'Name', role: 'user', loss: 6.1 },
  { t: ' a', role: 'user', loss: 2.3 },
  { t: ' prime', role: 'user', loss: 5.4 },
  { t: ' above', role: 'user', loss: 3.9 },
  { t: ' 10', role: 'user', loss: 2.8 },
  { t: '.', role: 'user', loss: 1.2 },
  { t: '<|im_end|>', role: 'tmpl', loss: 0.9 },
  { t: '↵', role: 'tmpl', loss: 0.01 },
  { t: '<|im_start|>', role: 'tmpl', loss: 0.01 },
  { t: 'assistant', role: 'tmpl', loss: 0.05 },
  { t: '↵', role: 'tmpl', loss: 0.01 },
  { t: '11', role: 'resp', loss: 1.6 },
  { t: ' is', role: 'resp', loss: 0.4 },
  { t: ' prime', role: 'resp', loss: 0.3 },
  { t: '.', role: 'resp', loss: 0.2 },
  { t: '<|im_end|>', role: 'resp', loss: 0.7 },
];
const ROW_SPLIT = 11; // tokens 0..10 on the first row (the user turn), the rest on the second

const STEPS = [
  'A demonstration',
  'One sequence, with a template',
  'Next-token loss on every token',
  'Most of it is the user’s words',
  'Mask the prompt',
  'The stop is a target too',
  'A batch: only amber cells learn',
  'How many demonstrations?',
];
const CAPTIONS = [
  'Supervised finetuning starts from pairs: a prompt, and the response a skilled person would want the model to give. This is a demonstration.',
  'The chat template turns the pair into one token sequence, with special tokens marking whose turn it is. To the model it is just another document.',
  'Train it exactly like pretraining: at every position, the loss is −log p of the true next token. Averaged over all 18 predicted tokens: 1.44 nats.',
  'But 88% of that loss comes from guessing what the user typed. Minimise it and the model gets better at writing questions, which is not the job.',
  'So mask the prompt: set its labels to −100 and the loss ignores those positions. Only the 5 response tokens are learned, mean 0.64 nats.',
  'The response ends with the end-of-turn token, and it is a target like any other. This is where the model learns to stop.',
  'In a padded batch, prompt and padding positions are computed but not learned. Here only 36 of 120 positions carry a loss.',
  'InstructGPT used 12,725 demonstrations, Llama 2-Chat 27,540, and LIMA just 1,000 carefully chosen ones. A sliver next to pretraining.',
];

// a padded batch of four demonstrations: [prompt length, response length]
const BATCH: [number, number][] = [[14, 16], [20, 4], [8, 10], [22, 6]];
const BATCH_COLS = 30;

const CHIP_H = 20;
const CHAR_W = 5.6;
const chipW = (t: string) => Math.max(18, t.length * CHAR_W + 10);

function layout() {
  const pos: { x: number; y: number; w: number }[] = [];
  let x = 24;
  TOKENS.forEach((tk, i) => {
    if (i === ROW_SPLIT) x = 24;
    const y = i < ROW_SPLIT ? 44 : 158;
    const w = chipW(tk.t);
    pos.push({ x, y, w });
    x += w + 4;
  });
  return pos;
}
const POS = layout();
const MAX_LOSS = 6.1;
const BAR_MAX = 34;

export default function LossMask() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [override, setOverride] = useState<'all' | 'resp' | null>(null);
  useEffect(() => setOverride(null), [st]);

  const mode: 'all' | 'resp' = override ?? (st >= 4 ? 'resp' : 'all');
  const inLoss = (i: number) => TOKENS[i].loss !== null && (mode === 'all' || TOKENS[i].role === 'resp');
  const counted = TOKENS.map((_, i) => i).filter(inLoss);
  const sum = counted.reduce((a, i) => a + (TOKENS[i].loss ?? 0), 0);
  const promptSum = counted.filter((i) => TOKENS[i].role !== 'resp').reduce((a, i) => a + (TOKENS[i].loss ?? 0), 0);
  const mean = counted.length ? sum / counted.length : 0;

  const showPair = st === 0;
  const showTokens = st >= 1 && st <= 5;
  const showBars = st >= 2 && st <= 5;
  const showBatch = st === 6;
  const showSizes = st === 7;

  const batchCells = BATCH_COLS * BATCH.length;
  const respCells = BATCH.reduce((a, [, r]) => a + r, 0);
  const promptCells = BATCH.reduce((a, [p]) => a + p, 0);

  return (
    <Stage
      rootRef={ref}
      title="Supervised finetuning: which tokens the loss sees"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['token', 'response tokens'], ['compute', 'counted in the loss'], ['idle', 'masked (label −100)']]}
      {...s}
      readout={
        st >= 2 && st <= 5 ? (
          <Readout items={[
            { label: 'tokens in the loss', value: `${counted.length}`, kind: 'compute', note: `of ${TOKENS.length - 1} predicted positions` },
            { label: 'mean loss', value: `${mean.toFixed(2)} nats`, kind: 'compute', note: `${sum.toFixed(2)} ÷ ${counted.length}` },
            { label: 'share from the prompt', value: `${sum > 0 ? ((promptSum / sum) * 100).toFixed(0) : 0}%`, kind: promptSum > 0 ? 'error' : 'token', note: 'of the summed loss' },
          ]} />
        ) : st === 6 ? (
          <Readout items={[
            { label: 'positions computed', value: `${batchCells}`, kind: 'text', note: '4 sequences × 30, padded' },
            { label: 'positions learned', value: `${respCells}`, kind: 'compute', note: `${((respCells / batchCells) * 100).toFixed(0)}% of the batch` },
            { label: 'prompt + padding', value: `${batchCells - respCells}`, kind: 'idle', note: 'computed, never a target' },
          ]} />
        ) : undefined
      }
      controls={
        st >= 2 && st <= 5 ? (
          <Segmented label="Compute the loss on" value={mode} onChange={(v) => setOverride(v)}
            options={[{ value: 'all' as const, label: 'every token' }, { value: 'resp' as const, label: 'response only' }]} />
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 352" role="img" aria-label="A prompt and response tokenized with a chat template. Bars under each token show its next-token loss; masking turns the prompt tokens grey so only the response, including the end-of-turn token, is learned. Later panels show a padded batch and the sizes of real SFT datasets.">
        {/* ── the human-written pair ── */}
        <Fade show={showPair}>
          <g>
            <Label x={24} y={30} tone={2} size={9.5}>prompt (from a user, or written by a labeler)</Label>
            <rect x={24} y={38} width={592} height={44} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x={40} y={65} tone={1} size={12} mono={false}>Name a prime above 10.</Label>
            <Label x={24} y={116} tone={2} size={9.5}>ideal response (written by a person who knows how to answer)</Label>
            <rect x={24} y={124} width={592} height={44} rx={4} fill="var(--token-dim)" stroke="var(--token)" />
            <Label x={40} y={151} size={12} mono={false} kind="token">11 is prime.</Label>
            <Label x={24} y={206} tone={1} size={10} bold>a demonstration: show, don’t tell</Label>
            <Label x={24} y={224} tone={2} size={9}>{'No rules, no reward, no explanation of what “helpful” means. Just thousands of examples of it.'}</Label>
            <Label x={24} y={240} tone={2} size={9}>{'InstructGPT hired about 40 contractors to write these for 12,725 prompts.'}</Label>
          </g>
        </Fade>

        {/* ── token rows ── */}
        <Fade show={showTokens}>
          <g>
            <Label x={24} y={30} tone={3} size={9}>the user turn</Label>
            <Label x={24} y={146} tone={3} size={9}>the assistant turn</Label>
            {TOKENS.map((tk, i) => {
              const p = POS[i];
              const masked = showBars && !inLoss(i);
              const fill = masked ? 'url(#tps-hatch)' : tk.role === 'resp' ? 'var(--token)' : 'var(--ground-3)';
              const stroke = masked ? 'var(--idle)' : tk.role === 'resp' ? 'var(--token)' : 'var(--rule-2)';
              const txt = tk.role === 'resp' && !masked ? '#0e1116' : tk.role === 'tmpl' ? 'var(--text-3)' : 'var(--text)';
              const h = tk.loss === null ? 0 : (tk.loss / MAX_LOSS) * BAR_MAX;
              const barY = p.y + CHIP_H + 5;
              const isStop = i === TOKENS.length - 1;
              return (
                <g key={i}>
                  <rect x={p.x} y={p.y} width={p.w} height={CHIP_H} rx={3} fill={fill} stroke={stroke} className="v-cell" />
                  <text x={p.x + p.w / 2} y={p.y + 13.5} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fill: txt, fontWeight: 600 }}>{tk.t}</text>
                  {/* loss bar, hanging below the chip */}
                  <g style={{ opacity: showBars && tk.loss !== null ? 1 : 0, transition: 'opacity .5s' }}>
                    <rect x={p.x + p.w / 2 - 5} y={barY} width={10} height={Math.max(1, h)} rx={1}
                      fill={inLoss(i) ? 'var(--compute)' : 'url(#tps-hatch)'} stroke={inLoss(i) ? 'var(--compute)' : 'var(--idle)'} strokeWidth={0.8} className="v-cell" />
                    <text x={p.x + p.w / 2} y={barY + Math.max(1, h) + 10} textAnchor="middle" className="v-label" style={{ fontSize: 8, fill: inLoss(i) ? 'var(--text-2)' : 'var(--text-3)' }}>
                      {tk.loss !== null ? (tk.loss >= 0.1 ? tk.loss.toFixed(1) : '0') : ''}
                    </text>
                  </g>
                  {isStop && (
                    <Fade show={st === 5}>
                      <rect x={p.x - 4} y={p.y - 4} width={p.w + 8} height={CHIP_H + 8} rx={4} fill="none" stroke="var(--token)" strokeWidth={1.6} className="v-pulse" />
                    </Fade>
                  )}
                </g>
              );
            })}
            {/* ruler between the two turns */}
            <line x1={24} x2={616} y1={133} y2={133} className="v-rule faint" />
            <Fade show={st === 5}>
              <g>
                <Label x={POS[TOKENS.length - 1].x + POS[TOKENS.length - 1].w + 10} y={POS[TOKENS.length - 1].y + 14} size={9.5} kind="token" bold>← the model learns to stop here</Label>
              </g>
            </Fade>
          </g>
        </Fade>

        {/* ── explanation strip for token steps ── */}
        {showTokens && (
          <g>
            {st === 1 && (
              <>
                <Label x={24} y={258} tone={1} size={10} bold>the template turns the pair into one document</Label>
                <Label x={24} y={276} tone={2} size={9}>{'<|im_start|> and <|im_end|> are special tokens: single vocabulary entries that mark turns.'}</Label>
                <Label x={24} y={292} tone={2} size={9}>{'Qwen2.5 uses this ChatML format; Llama 3 uses <|start_header_id|> … <|eot_id|>.'}</Label>
                <Label x={24} y={308} tone={3} size={9}>{'The exact template is part of the model: use a different one at inference and quality drops.'}</Label>
              </>
            )}
            {(st === 2 || st === 3 || st === 4) && (
              <>
                <Label x={24} y={258} tone={1} size={10} bold>
                  {mode === 'all' ? 'loss on every token: mean = 25.91 ÷ 18 = 1.44 nats' : 'loss on the response only: mean = 3.20 ÷ 5 = 0.64 nats'}
                </Label>
                <Label x={24} y={276} tone={2} size={9}>
                  {mode === 'all'
                    ? 'Prompt tokens sum to 22.71 nats, response tokens to 3.20. The prompt is 88% of what is being minimised.'
                    : 'Masked positions still go through the forward pass (the response attends to them), but no gradient asks the'}
                </Label>
                <Label x={24} y={292} tone={2} size={9}>
                  {mode === 'all'
                    ? 'Guessing “prime” after “Name a” costs 5.4 nats. Learning that teaches the model to write user questions.'
                    : 'model to predict them. In code: labels = input_ids.clone(); labels[:prompt_len] = −100.'}
                </Label>
                <Label x={24} y={308} tone={3} size={9}>
                  {mode === 'all'
                    ? 'Template tokens are nearly free (0.01–0.05): after “<|im_start|>” comes a role name, always.'
                    : 'PyTorch’s cross-entropy skips every position whose label is −100. That is the whole trick.'}
                </Label>
              </>
            )}
            {st === 5 && (
              <>
                <Label x={24} y={258} tone={1} size={10} bold>{'<|im_end|> is in the response, so it is a target'}</Label>
                <Label x={24} y={276} tone={2} size={9}>{'Every demonstration ends with it, so after a complete answer the model learns to put its probability there.'}</Label>
                <Label x={24} y={292} tone={2} size={9}>{'Forget to include it (or mask it) and the finetuned model rambles on, exactly like the base model did.'}</Label>
                <Label x={24} y={308} tone={3} size={9}>{'At inference the engine stops when it samples this token: that is the stop condition you configure.'}</Label>
              </>
            )}
          </g>
        )}

        {/* ── a padded batch ── */}
        <Fade show={showBatch}>
          <g>
            <Label x={24} y={26} tone={1} size={10} bold>a batch of four demonstrations, padded to 30 positions</Label>
            {BATCH.map(([pl, rl], r) => {
              const y = 44 + r * 34;
              return (
                <g key={r}>
                  <Label x={24} y={y + 13} tone={3} size={9}>{`seq ${r + 1}`}</Label>
                  {Array.from({ length: BATCH_COLS }, (_, c) => {
                    const kind = c < pl ? 'prompt' : c < pl + rl ? 'resp' : 'pad';
                    return (
                      <rect key={c} x={72 + c * 16} y={y} width={14} height={18} rx={2}
                        fill={kind === 'resp' ? 'var(--compute)' : kind === 'prompt' ? 'url(#tps-hatch)' : 'none'}
                        stroke={kind === 'resp' ? 'var(--compute)' : kind === 'prompt' ? 'var(--idle)' : 'var(--rule)'}
                        strokeDasharray={kind === 'pad' ? '2 2' : undefined} />
                    );
                  })}
                  <Label x={616} y={y + 13} anchor="end" tone={2} size={9}>{`${rl} learned`}</Label>
                </g>
              );
            })}
            <Label x={72} y={196} tone={3} size={9}>hatched = prompt (masked) · amber = response (in the loss) · dashed = padding (masked)</Label>
            <Label x={24} y={232} tone={1} size={10} bold>{`${respCells} of ${batchCells} positions carry a loss: ${((respCells / batchCells) * 100).toFixed(0)}%`}</Label>
            <Label x={24} y={250} tone={2} size={9}>{`Prompt: ${promptCells} positions. Padding: ${batchCells - promptCells - respCells}. All 120 go through the forward pass, because the response`}</Label>
            <Label x={24} y={266} tone={2} size={9}>{'tokens attend to the prompt. Only the amber cells contribute a gradient.'}</Label>
            <Label x={24} y={282} tone={2} size={9}>{'Packing sequences end to end removes the padding; masking stays, because the prompt is not the job.'}</Label>
            <Label x={24} y={298} tone={3} size={9}>{'The mean is taken over amber cells only. Dividing by 120 would quietly shrink the learning rate.'}</Label>
          </g>
        </Fade>

        {/* ── how many demonstrations ── */}
        <Fade show={showSizes}>
          <g>
            <Label x={24} y={26} tone={1} size={10} bold>how many demonstrations real models were finetuned on</Label>
            {[
              { name: 'LIMA (65B LLaMA)', n: 1000, note: 'curated by hand for quality and variety' },
              { name: 'InstructGPT (GPT-3)', n: 12725, note: 'prompts with labeler-written answers' },
              { name: 'Llama 2-Chat', n: 27540, note: 'annotations collected for Meta, after which they stopped' },
            ].map((d, i) => {
              const y = 52 + i * 62;
              const w = Math.max(3, (d.n / 27540) * 440);
              return (
                <Move key={d.name} x={0} y={0}>
                  <text x={24} y={y} className="v-label" style={{ fontSize: 10, fill: 'var(--text)', fontWeight: 700 }}>{d.name}</text>
                  <rect x={24} y={y + 8} width={w} height={16} rx={2} fill="var(--token)" style={{ transition: 'width .8s var(--ease-out)' }} />
                  <text x={24 + w + 8} y={y + 20} className="v-label" style={{ fontSize: 10, fill: 'var(--token)' }}>{d.n.toLocaleString('en-US')}</text>
                  <text x={24} y={y + 40} className="v-label t3" style={{ fontSize: 9 }}>{d.note}</text>
                </Move>
              );
            })}
            <Label x={24} y={258} tone={1} size={10} bold>versus roughly 300,000,000,000 pretraining tokens for GPT-3</Label>
            <Label x={24} y={276} tone={2} size={9}>{'12,725 demonstrations of a few hundred tokens each is a few million tokens: about 0.001% as much text.'}</Label>
            <Label x={24} y={292} tone={2} size={9}>{'LIMA’s claim: the knowledge is already in the base model. SFT teaches the format to answer in.'}</Label>
            <Label x={24} y={308} tone={3} size={9}>{'In LIMA’s ablations, more examples from one source helped far less than better or more varied ones.'}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
