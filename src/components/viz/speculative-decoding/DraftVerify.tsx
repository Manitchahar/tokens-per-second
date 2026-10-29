import { useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Chip, Fade, Dim, fmt } from '../lib/svg';
import { decodeStep, verifyPass, LLAMA8B, block } from './specModel';

// One block of speculative decoding, from the cheap proposal to the tokens that
// actually leave. gamma drafts + 1 target position are checked in ONE forward
// pass, which is why the whole block costs about what one decode step costs.

const STEPS = [
  'Decode is one token per step',
  'A small model guesses ahead',
  'The guesses are guesses',
  'The target checks all of them at once',
  'Accept the prefix, cut at the first rejection',
  'The last position is a bonus token',
  'One pass, many tokens',
  'Why the check is nearly free',
];

const CAPTIONS = [
  'Ordinary decode is strictly serial: token n+1 cannot be computed until token n has been sampled and fed back. Each step is a full read of the weights to produce exactly one token. That is the wall this module is about.',
  'So let something cheaper take a guess. A small draft model — or a table, or a few extra heads — proposes gamma tokens ahead. These are the target model not being asked, so they carry no guarantee of any kind.',
  'The draft tokens are usually right and sometimes badly wrong. Here the run of guesses agrees with the target for two tokens and then breaks. Nothing is lost by being wrong — they were never going to be output.',
  'The target model now runs ONCE over the whole block: gamma drafts plus the position after them. One forward pass, because every position is independent given the context to its left. The weights are read once.',
  'Walk the block left to right and keep the tokens that match the target\'s own prediction. The accepted prefix is the longest one; everything from the first disagreement onward is discarded, including the draft that disagreed.',
  'The last position is different: it is a position the draft never claimed to have predicted. Whatever the target samples there is correct by construction, and it is free. Every block gets at least one token.',
  'One pass over gamma+1 positions has produced several tokens. Compare the work: the alternative was several serial passes, each reading the weights again. The weights are the cost, and we read them once.',
  'Decode is memory-bound, so its time is set by weight traffic — and weight traffic does not depend on how many positions the step handles. That is the entire trick. The block does gamma times the FLOPs, which is the only thing that ever makes it stop being free.',
];

// A worked example: the target would have produced this run of tokens.
const TARGET = ['The', ' capital', ' of', ' France', ' is', ' Paris', '.', ' It'];
const DRAFT = ['The', ' capital', ' of', ' France', ' is', ' Berlin', '.', ' Much'];

export default function DraftVerify() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [gamma, setGamma] = useState(4);

  const tDecode = decodeStep(LLAMA8B);
  const verify = verifyPass(LLAMA8B, gamma);
  const verifyT = useTween(verify.total, 600);
  const decodeT = useTween(tDecode, 600);

  // this example accepts 2 drafts then breaks: tokens out = 2 accepted + 1 bonus
  const accepted = 2;
  const G = Math.min(gamma, DRAFT.length);
  const speedup = block(0.72, 0.125, G, verify.total, tDecode).speedup;

  const CX = 34;
  const CHIP_W = 62;
  const GAP = 6;
  const rowW = CHIP_W + GAP;

  return (
    <Stage
      rootRef={ref}
      title="One speculative block: draft, then verify"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['token', 'accepted, goes to the user'],
        ['compute', 'the target model checking'],
        ['error', 'rejected, discarded'],
        ['idle', 'drafted, not yet checked'],
        ['memory', 'weights read from HBM'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'One decode step', value: `${(decodeT * 1e3).toFixed(2)} ms`, kind: 'memory', note: '1 token, Llama-3.1-8B on an H100' },
          { label: `Verify ${G + 1} positions`, value: `${(verifyT * 1e3).toFixed(2)} ms`, kind: 'compute', note: `${(verify.total / tDecode).toFixed(2)}x a decode step` },
          { label: 'Tokens from the block', value: `${accepted + 1}`, kind: 'token', note: `${accepted} accepted + 1 bonus` },
          { label: 'vs. decoding them', value: `${((accepted + 1) * tDecode * 1e3).toFixed(2)} ms`, kind: 'error', note: 'serial passes, weights re-read' },
        ]} />
      }
      controls={
        <Segmented
          label="Drafts per block (gamma)"
          value={gamma}
          options={[{ value: 2, label: '2' }, { value: 4, label: '4' }, { value: 6, label: '6' }, { value: 8, label: '8' }]}
          onChange={setGamma}
        />
      }
    >
      <svg viewBox="0 0 640 392" role="img" aria-label="A speculative decoding block. A draft model proposes several tokens, the target model checks them all in one forward pass, the accepted prefix is kept and everything from the first disagreement is discarded, and the final position contributes a free bonus token. A comparison at the bottom shows that one verification pass costs about the same as a single decode step.">
        {/* ── the target's own answer ───────────────────────── */}
        <Label x={CX} y={24} tone={2} size={11} mono={false} bold>What the target model would have said</Label>
        <Fade show={s.step >= 0}>
          {TARGET.slice(0, G + 1).map((t, i) => (
            <Chip key={i} x={CX + i * rowW} y={34} text={t} kind={i === 0 ? 'text' : 'token'} w={CHIP_W} size={9.5} opacity={i <= 1 ? 1 : 0.55} />
          ))}
          <Label x={CX} y={72} tone={3} size={9}>one token at a time — each needs the one before it</Label>
        </Fade>

        {/* ── the draft ─────────────────────────────────────── */}
        <Label x={CX} y={100} tone={2} size={11} mono={false} bold>A small model guesses</Label>
        <Fade show={s.step >= 1}>
          {DRAFT.slice(0, G).map((t, i) => {
            const right = i <= accepted - 1;
            const broken = i === accepted;
            const checked = s.step >= 3;
            const kind = !checked ? 'idle' : right ? 'token' : broken || i > accepted ? 'error' : 'token';
            return (
              <Chip key={i} x={CX + i * rowW} y={110} text={t} kind={kind} w={CHIP_W} size={9.5}
                dim={!checked && !broken} opacity={checked ? (i > accepted ? 0.75 : 1) : 0.9} />
            );
          })}
          <Label x={CX} y={148} tone={3} size={9}>drafted gamma tokens, then stop — no guarantee, and no cost to be wrong</Label>
        </Fade>

        {/* the mismatch callout */}
        <Fade show={s.step >= 2}>
          <Arrow x1={CX + accepted * rowW + CHIP_W / 2} y1={132} x2={CX + accepted * rowW + CHIP_W / 2} y2={170} kind="error" />
          <Label x={CX + accepted * rowW + CHIP_W / 2 + 6} y={182} size={9.5} kind="error" bold>first disagreement</Label>
          <Label x={CX + accepted * rowW + CHIP_W / 2 + 6} y={193} tone={3} size={9}>everything from here on is discarded</Label>
        </Fade>

        {/* ── one pass over the whole block ─────────────────── */}
        <Fade show={s.step >= 3}>
          <rect x={CX - 6} y={102} width={(G + 1) * rowW - GAP + 12} height={46} rx={4}
            fill="none" stroke="var(--compute)" strokeWidth={1.6} />
          <Label x={CX - 6} y={98} size={9.5} kind="compute" bold>{`one forward pass over ${G + 1} positions`}</Label>
        </Fade>

        {/* ── what comes out ────────────────────────────────── */}
        <Label x={CX} y={222} tone={2} size={11} mono={false} bold>What leaves the block</Label>
        <Fade show={s.step >= 4}>
          {Array.from({ length: accepted }, (_, i) => (
            <Chip key={`a${i}`} x={CX + i * rowW} y={232} text={DRAFT[i]} kind="token" w={CHIP_W} size={9.5} />
          ))}
          <Chip x={CX + accepted * rowW} y={232} text={TARGET[accepted]} kind="compute" w={CHIP_W} size={9.5} />
          <Label x={CX + (accepted + 1) * rowW + 10} y={240} size={9.5} kind="compute" bold>bonus token</Label>
          <Label x={CX + (accepted + 1) * rowW + 10} y={252} tone={3} size={9}>never predicted by</Label>
          <Label x={CX + (accepted + 1) * rowW + 10} y={263} tone={3} size={9}>the draft, so free</Label>
          <Dim x1={CX} x2={CX + (accepted + 1) * rowW - GAP} y={280} label={`${accepted + 1} tokens`} kind="token" />
        </Fade>

        {/* ── the cost comparison ───────────────────────────── */}
        <Fade show={s.step >= 5}>
          <Label x={CX + (accepted + 1) * rowW + 10} y={277} tone={3} size={9}>{`at ${accepted + 1} tokens per block`}</Label>
        </Fade>

        <Fade show={s.step >= 6}>
          <Label x={CX} y={306} tone={2} size={11} mono={false} bold>The same tokens, two ways</Label>

          <Label x={CX} y={326} tone={3} size={9.5}>decode them one at a time</Label>
          {Array.from({ length: accepted + 1 }, (_, i) => (
            <Span key={i} x={CX + 200 + i * 62} y={317} w={56} h={14} kind="memory" />
          ))}
          <Label x={CX + 200 + (accepted + 1) * 62 + 4} y={328} size={9.5} kind="memory">{`${((accepted + 1) * tDecode * 1e3).toFixed(2)} ms`}</Label>

          <Label x={CX} y={348} tone={3} size={9.5}>one block</Label>
          <Span x={CX + 200} y={339} w={86} h={14} kind="compute" />
          <Label x={CX + 292} y={350} size={9.5} kind="compute">{`${(verify.total * 1e3).toFixed(2)} ms`}</Label>

          <Label x={CX + 340} y={326} tone={2} size={10} bold>{`${((accepted + 1) * tDecode / verify.total).toFixed(1)}x faster`}</Label>
          <Label x={CX + 340} y={340} tone={3} size={9}>{`weights read once, not ${accepted + 1} times`}</Label>
          <Label x={CX + 340} y={352} tone={3} size={9}>{`a=0.72 at gamma=${G} gives ${speedup.toFixed(2)}x in expectation`}</Label>
        </Fade>

        <Fade show={s.step >= 7}>
          <Label x={CX} y={376} size={10} kind="memory" bold>
            {`one pass over ${G + 1} positions = ${(verify.total / tDecode).toFixed(2)}x a decode step — decode is memory-bound, and weight traffic does not care how many positions it processes`}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}