import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Move, Chip, Arrow, fmt } from '../lib/svg';
import { modelById, gpuById, decodeStep, kvBytesPerToken } from '../lib/perfModel';

// The generate loop, slowed down until you can see every part: tokenize once, run one
// forward pass over the prompt (prefill), then repeat a one-token step (decode) until a stop token.

const STEPS = [
  'The loop you will rebuild',
  'Your text becomes tokens',
  'Prefill: the prompt at once',
  'The first token appears',
  'Decode: one token, fed back',
  'Feed the token back in',
  'Repeat until a stop token',
  'Stream it as you go',
  'What streaming buys the user',
];
const CAPTIONS = [
  'Everything in the rest of this course is an optimisation of this one loop: run the model, take one token, put it back, repeat until it says stop.',
  'A string becomes a list of integers via the tokenizer. "Hello world" is usually two tokens; the numbers index the embedding table you met in Module 03.',
  'The whole prompt goes through the network in one parallel pass. Every token attends to every token before it, and the KV cache is filled. This is prefill, and it is compute-bound.',
  'The last position of the prefill emits logits, and one token is sampled from them. This is the first token a user sees — after the entire prompt, in one shot.',
  'Now only the new token enters the network. It reads the cached keys and values instead of recomputing them. This is decode, and it is memory-bound: one token, all the weights.',
  'The sampled token is decoded back to text, appended to the sequence, and fed in as the input for the next step. The loop closes on itself — this is why it is called autoregressive.',
  'A stop condition ends the loop: an end-of-sequence token, a maximum length, or your own stopping string. Without one, a model can keep generating forever.',
  'Instead of collecting all the tokens and returning once, you emit each one the instant it is produced. Same total work, completely different felt speed.',
  'Time to first token is paid once at the start; after that each token arrives at the decode rate. Streaming makes perceived speed depend on TTFT, not on total answer length.',
];

const PROMPT = 'Write one sentence about the sea.';
const PROMPT_TOKENS = [4080, 3672, 1063, 2213, 1109, 421, 4996, 13];
const ANSWER = ['The', ' sea', ' is', ' vast', ' and', ' deep', '.'];
const EOS = '⟨end⟩';

// A plausible token sequence for the prompt above, as the tokenizer would emit it.
const TOK_TEXT = PROMPT.split(' ');

// Qwen2.5-0.5B, the model this module actually runs. Numbers come from the shared model.
const M = modelById('qwen05');
const G = gpuById('t4');

export default function GenerateLoop() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [maxNew, setMaxNew] = useState(8);

  // A decode step's real cost on a T4, so the readout is honest, not decorative.
  const r0 = decodeStep({ model: M, gpu: G, batch: 1, ctx: 8 });
  const stepMs = r0.t * 1e3;
  const perTok = kvBytesPerToken(M);

  // How many decode steps are on screen at once.
  const nShown = Math.min(maxNew, ANSWER.length + 1);

  return (
    <Stage
      rootRef={ref}
      title="The generate loop: prefill once, then decode one token at a time"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['text', 'prompt tokens'],
        'compute',
        'memory',
        ['comm', 'streamed to user'],
        'token',
        'idle',
      ]}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'Prefill', value: `${(PROMPT_TOKENS.length * 0.9).toFixed(1)} ms`, kind: 'compute', note: `${PROMPT_TOKENS.length} tokens, all at once` },
            { label: 'Decode, one step', value: `${stepMs.toFixed(1)} ms`, kind: 'memory', note: '1 token in, all the weights read' },
            { label: 'KV cache', value: `${fmt.bytes(perTok)}/tok`, kind: 'memory', note: 'grows every step' },
            { label: 'Whole answer', value: `${(PROMPT_TOKENS.length * 0.9 + stepMs * nShown).toFixed(0)} ms`, kind: 'token', note: `${nShown} tokens, streamed` },
          ]}
        />
      }
      controls={
        <Segmented
          label="max_new_tokens — the safety cap on generated length"
          value={maxNew}
          options={[4, 8, 16, 64].map((v) => ({ value: v, label: String(v) }))}
          onChange={setMaxNew}
        />
      }
    >
      <svg viewBox="0 0 640 420" role="img" aria-label="A diagram of the generate loop: a prompt is tokenized into chips, processed in one prefill pass to produce a first token, then a repeated decode step feeds each new token back in until a stop token, with each produced token optionally streamed to the user as it is generated.">
        {/* ── the prompt row */}
        <Label x={56} y={30} tone={1} size={11} bold mono={false}>Your prompt</Label>
        <text x={168} y={30} className="v-label" style={{ fontSize: 11, fill: 'var(--text-2)' }}>{`"${PROMPT}"`}</text>

        <Move x={0} y={0}>
          {TOK_TEXT.map((w, i) => (
            <Chip key={i} x={56 + i * 60} y={42} text={w} kind="text" size={10} w={54} h={24} />
          ))}
        </Move>
        <Label x={56 + TOK_TEXT.length * 60 + 4} y={58} tone={3} size={9}>
          {`${TOK_TEXT.length} tokens`}
        </Label>
        <Arrow x1={56} y1={78} x2={56} y2={92} kind="text" width={1.2} />
        <Label x={50} y={90} anchor="end" tone={3} size={8.5}>ids</Label>

        {/* ── the model box */}
        <rect x={56} y={96} width={290} height={92} rx={5}
          fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1.2} className="v-fade" />
        <Label x={70} y={116} tone={1} size={10.5} bold mono={false}>the transformer</Label>
        <Label x={70} y={131} tone={3} size={9}>{`${M.name} · ${M.layers} layers · hidden ${M.hidden}`}</Label>

        {/* prefill pass */}
        <Fade show={st === 2 || st === 3}>
          <rect x={68} y={140} width={266} height={20} rx={3} fill="var(--compute)" stroke="var(--compute)" className="v-cell" />
          <text x={201} y={154} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: '#0e1116' }}>
            prefill: all {TOK_TEXT.length} prompt tokens, one parallel pass
          </text>
        </Fade>
        {/* decode pass */}
        <Fade show={st >= 4}>
          <rect x={68} y={162} width={266} height={20} rx={3} fill="var(--memory)" stroke="var(--memory)" className="v-cell" />
          <text x={201} y={176} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: '#0e1116' }}>
            decode: 1 new token, reads the KV cache
          </text>
        </Fade>
        <Fade show={st < 2}>
          <rect x={68} y={140} width={266} height={42} rx={3} fill="none" stroke="var(--rule-2)" strokeDasharray="3 3" />
          <text x={201} y={165} textAnchor="middle" className="v-label t3" style={{ fontSize: 10 }}>32 layers run here</text>
        </Fade>

        {/* the KV cache growing beside the model */}
        <Label x={470} y={116} tone={1} size={10.5} bold mono={false}>KV cache</Label>
        <Fade show={st >= 2}>
          <rect x={470} y={126} width={12} height={20} rx={2} fill="var(--memory)" className="v-cell" />
          {Array.from({ length: Math.min(TOK_TEXT.length, nShown + 3) }, (_, i) => (
            <rect key={i} x={470 + (i + 1) * 14} y={126} width={12} height={20} rx={2}
              fill={i < TOK_TEXT.length ? 'var(--memory)' : 'var(--memory-dim)'} className="v-cell" />
          ))}
          <Label x={476} y={162} anchor="middle" tone={3} size={8.5}>prefill fills it</Label>
        </Fade>
        <Fade show={st >= 4}>
          <Arrow x1={470 + (Math.min(TOK_TEXT.length, nShown + 3) + 1) * 14} y1={136} x2={490} y2={136} kind="memory" width={1.2} />
          <Label x={490} y={158} anchor="middle" tone={3} size={8.5}>+1 per step</Label>
        </Fade>
        <Fade show={st < 2}>
          <rect x={470} y={126} width={110} height={20} rx={2} fill="none" stroke="var(--rule-2)" strokeDasharray="3 3" />
        </Fade>

        {/* ── the generated tokens coming out */}
        <Label x={56} y={218} tone={1} size={11} bold mono={false}>Tokens produced</Label>

        <Move x={0} y={0}>
          {ANSWER.slice(0, Math.min(nShown, ANSWER.length)).map((w, i) => {
            const on = st >= 3;
            const streamed = st >= 7 && i <= Math.min(nShown, ANSWER.length) - 1;
            return (
              <g key={i} style={{ opacity: on ? 1 : 0.12 }} className="v-fade">
                <rect x={56 + i * 52} y={228} width={48} height={24} rx={3}
                  fill="var(--token)" stroke="var(--token)" strokeWidth={1} className="v-cell" />
                <text x={56 + i * 52 + 24} y={244} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: '#0e1116' }}>{w}</text>
                {/* streaming arrow */}
                {streamed && (
                  <g className="v-fade">
                    <Arrow x1={80 + i * 52} y1={256} x2={80 + i * 52} y2={288} kind="comm" width={1.2} />
                  </g>
                )}
              </g>
            );
          })}
          {/* the stop token */}
          {st >= 6 && nShown > ANSWER.length && (
            <g className="v-fade">
              <rect x={56 + ANSWER.length * 52} y={228} width={48} height={24} rx={3}
                fill="none" stroke="var(--error)" strokeWidth={1.4} strokeDasharray="3 3" />
              <text x={56 + ANSWER.length * 52 + 24} y={244} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: 'var(--error)' }}>{EOS}</text>
            </g>
          )}
        </Move>

        {/* ── the feedback arrow: token back into the model */}
        <Fade show={st === 5}>
          <path d="M 80 268 L 80 300 L 201 300 L 201 192"
            fill="none" stroke="var(--compute)" strokeWidth={1.4} strokeDasharray="4 3" markerEnd="url(#tps-arrow)" />
          <Label x={214} y={304} tone={2} size={9.5}>appended to the input, then fed back in</Label>
        </Fade>

        {/* ── streaming to the user */}
        <Fade show={st >= 7}>
          <rect x={56} y={308} width={300} height={30} rx={4}
            fill="var(--comm-dim)" stroke="var(--comm)" strokeWidth={1.2} className="v-cell" />
          <text x={206} y={327} textAnchor="middle" className="v-label bold" style={{ fontSize: 10.5, fill: 'var(--comm)' }}>
            streamed to the user as each token appears
          </text>
          <Arrow x1={356} y1={323} x2={420} y2={323} kind="comm" width={1.5} />
          <text x={432} y={327} className="v-label" style={{ fontSize: 11, fill: 'var(--text)' }}>
            {ANSWER.slice(0, Math.min(nShown, ANSWER.length)).join('')}
          </text>
        </Fade>

        {/* ── TTFT / TPOT ruler */}
        <Fade show={st === 8}>
          <line x1={56} y1={372} x2={584} y2={372} className="v-rule" />
          <line x1={56} x2={120} y1={368} y2={376} stroke="var(--compute)" strokeWidth={2.5} />
          <line x1={120} x2={584} y1={368} y2={376} stroke="var(--token)" strokeWidth={2.5} />
          <Label x={88} y={390} anchor="middle" size={9.5} kind="compute">TTFT</Label>
          <Label x={120} y={390} anchor="start" size={9.5} kind="token" style={{ marginLeft: 6 }}>
            {`one token every ${stepMs.toFixed(1)} ms (TPOT)`}
          </Label>
          <Label x={584} y={406} anchor="end" tone={3} size={9}>
            Felt speed ≈ time to first token + n × time per token
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
