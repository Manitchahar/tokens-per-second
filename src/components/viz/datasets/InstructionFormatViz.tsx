import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, COLOR, DIM } from '../lib/svg';

const STEPS = [
  'Tokens: special ID vs text',
  'Anatomy of a chat turn',
  'Loss masking: target = -100',
  'The whitespace trap',
  'The missing EOS trap',
  'Exact inference alignment',
];

const CAPTIONS = [
  'Special markers like <|im_start|> must map to a single reserved token ID (e.g. 151644). If tokenized as raw ASCII characters, attention fails to recognize the boundary.',
  'A standard instruction format packs three sections into a continuous sequence: system prompt, user query, and assistant response.',
  'During training, the loss is masked on system and user tokens (label = -100). Gradients backpropagate ONLY through the assistant tokens.',
  'A single trailing space ("assistant\\n" vs "assistant \\n") changes the tokenization completely, breaking prefix caching and confusing generation.',
  'If the assistant response does not end with the explicit end-of-turn token (<|im_end|>), the model will babble indefinitely at inference time.',
  'The training prefix must match the inference prompt down to the exact byte. Chat templates automate this consistency.',
];

interface TokenBlock {
  text: string;
  id: number;
  role: 'system' | 'user' | 'assistant' | 'special';
  masked: boolean;
}

export default function InstructionFormatViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [template, setTemplate] = useState<'chatml' | 'llama3'>('chatml');

  // Token sequence definition
  const chatmlTokens: TokenBlock[] = [
    { text: '<|im_start|>', id: 151644, role: 'special', masked: true },
    { text: 'system\\n', id: 8948, role: 'system', masked: true },
    { text: 'You are helpful', id: 2610, role: 'system', masked: true },
    { text: '<|im_end|>\\n', id: 151645, role: 'special', masked: true },
    { text: '<|im_start|>', id: 151644, role: 'special', masked: true },
    { text: 'user\\n', id: 872, role: 'user', masked: true },
    { text: 'What is 12*12?', id: 18420, role: 'user', masked: true },
    { text: '<|im_end|>\\n', id: 151645, role: 'special', masked: true },
    { text: '<|im_start|>', id: 151644, role: 'special', masked: true },
    { text: 'assistant\\n', id: 77091, role: 'assistant', masked: true },
    { text: '144', id: 18924, role: 'assistant', masked: false },
    { text: '<|im_end|>', id: 151645, role: 'special', masked: false },
  ];

  const totalTokens = chatmlTokens.length;
  const trainedTokens = chatmlTokens.filter((t) => !t.masked).length;
  const maskedTokens = totalTokens - trainedTokens;

  return (
    <Stage
      rootRef={ref}
      title="Instruction Formatting & Loss Masking"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'idle', 'error', 'memory']}
      readout={
        <Readout
          items={[
            {
              label: 'Total Sequence',
              value: `${totalTokens} tokens`,
              kind: 'memory',
              note: 'Prompt + generation in one forward pass',
            },
            {
              label: 'Loss-Active Tokens',
              value: `${s.step >= 2 ? trainedTokens : totalTokens} tokens`,
              kind: 'token',
              note: s.step >= 2 ? 'Target != -100 (Backprop active)' : 'Unmasked (Naive baseline)',
            },
            {
              label: 'Loss Masked (-100)',
              value: `${s.step >= 2 ? maskedTokens : 0} tokens`,
              kind: 'idle',
              note: s.step >= 2 ? 'Zero gradient on prompt' : 'No masking: prompt memorized',
            },
            {
              label: 'Template Standard',
              value: template === 'chatml' ? 'ChatML (Qwen)' : 'Llama-3 Header Format',
              kind: 'memory',
              note: 'Must match inference template exactly',
            },
          ]}
        />
      }
      controls={
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <Segmented
            value={template}
            onChange={(v) => setTemplate(v as 'chatml' | 'llama3')}
            options={[
              { value: 'chatml', label: 'ChatML Standard' },
              { value: 'llama3', label: 'Llama-3 Standard' },
            ]}
          />
        </div>
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Visualizing chat template formatting, special tokens, and loss masking">
        {/* Step 0: Special Token ID vs Character Token Bug */}
        {s.step === 0 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              The Special Token Trap: Single ID vs ASCII Decomposition
            </Label>

            {/* Correct Case */}
            <rect x={40} y={65} width={260} height={120} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={55} y={88} anchor="start" size={12} bold style={{ fill: COLOR.token }}>
              Correct: Special Token Configured
            </Label>
            <Label x={55} y={108} anchor="start" size={10} tone={2}>
              Text: {'"<|im_start|>"'}
            </Label>
            <Span x={55} y={120} w={120} h={30} kind="token" label="ID: 151644" />
            <Label x={55} y={168} anchor="start" size={9} tone={3}>
              Mapped directly to atomic vocabulary index
            </Label>

            {/* Buggy Case */}
            <rect x={340} y={65} width={260} height={120} rx={4} fill="#141820" stroke={COLOR.error} />
            <Label x={355} y={88} anchor="start" size={12} bold style={{ fill: COLOR.error }}>
              Bug: Treated as Plain Text String
            </Label>
            <Label x={355} y={108} anchor="start" size={10} tone={2}>
              Text: {'"<|im_start|>"'} splits into 6 pieces:
            </Label>
            <g>
              <Span x={355} y={120} w={22} h={30} kind="error" label="<" />
              <Span x={380} y={120} w={22} h={30} kind="error" label="|" />
              <Span x={405} y={120} w={28} h={30} kind="error" label="im" />
              <Span x={436} y={120} w={15} h={30} kind="error" label="_" />
              <Span x={454} y={120} w={40} h={30} kind="error" label="start" />
              <Span x={497} y={120} w={35} h={30} kind="error" label="|>" />
            </g>
            <Label x={355} y={168} anchor="start" size={9} tone={2} style={{ fill: COLOR.error }}>
              Model treats marker as conversational text!
            </Label>

            <rect x={40} y={205} width={560} height={50} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
            <Label x={320} y={226} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
              Always pass special_tokens_map or use tokenizer.apply_chat_template()
            </Label>
            <Label x={320} y={242} anchor="middle" size={9} tone={2}>
              Never use string.format() or naive regex string joins without tokenization verification.
            </Label>
          </g>
        )}

        {/* Steps 1, 2, 3: Full Token Stream & Loss Masking */}
        {(s.step === 1 || s.step === 2) && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              {s.step === 1 ? 'Token Stream: System, User, and Assistant Turns' : 'Loss Masking: Zero Gradient on Prompts (-100)'}
            </Label>

            {/* Sequence Tokens Layout */}
            {chatmlTokens.map((tok, i) => {
              const x = 35 + (i % 6) * 95;
              const row = Math.floor(i / 6);
              const y = 65 + row * 90;

              const isTargetActive = s.step === 2 && !tok.masked;
              const isTargetMasked = s.step === 2 && tok.masked;

              const fillKind = s.step === 1 ? (tok.role === 'special' ? 'memory' : tok.role === 'assistant' ? 'token' : 'compute') : (isTargetActive ? 'token' : 'idle');

              return (
                <g key={i}>
                  <rect
                    x={x}
                    y={y}
                    width={88}
                    height={50}
                    rx={3}
                    fill={fillKind === 'idle' ? 'url(#tps-hatch)' : fillKind === 'token' ? DIM.token : DIM[fillKind]}
                    stroke={fillKind === 'idle' ? 'var(--rule-2)' : COLOR[fillKind]}
                    strokeWidth={1.5}
                  />
                  <Label x={x + 44} y={y + 18} anchor="middle" size={9.5} bold tone={1}>
                    {tok.text}
                  </Label>
                  <Label x={x + 44} y={y + 32} anchor="middle" size={8.5} tone={3}>
                    ID: {tok.id}
                  </Label>
                  <Label x={x + 44} y={y + 44} anchor="middle" size={7.5} tone={2}>
                    {tok.role}
                  </Label>

                  {/* Step 2: Loss target label */}
                  {s.step === 2 && (
                    <g>
                      <rect
                        x={x + 10}
                        y={y + 55}
                        width={68}
                        height={16}
                        rx={2}
                        fill={isTargetActive ? COLOR.token : '#1a202c'}
                        stroke="var(--rule-1)"
                      />
                      <Label
                        x={x + 44}
                        y={y + 66}
                        anchor="middle"
                        size={8}
                        bold
                        style={{ fill: isTargetActive ? '#0e1116' : 'var(--text-3)' }}
                      >
                        {isTargetActive ? 'Target: Loss ON' : 'Label: -100'}
                      </Label>
                    </g>
                  )}
                </g>
              );
            })}

            {s.step === 2 && (
              <g>
                <rect x={40} y={260} width={560} height={40} rx={4} fill="#141820" stroke={COLOR.token} />
                <Label x={320} y={278} anchor="middle" size={10} bold style={{ fill: COLOR.token }}>
                  PyTorch CrossEntropyLoss ignores index -100: Only response tokens update weights!
                </Label>
                <Label x={320} y={292} anchor="middle" size={9} tone={2}>
                  Unmasked training forces the model to memorize user questions, degrading general reasoning.
                </Label>
              </g>
            )}
          </g>
        )}

        {/* Step 3: The Whitespace Trap */}
        {s.step === 3 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              The Whitespace Trap: Trailing Spaces Alter Tokenization
            </Label>

            {/* Version A: No trailing space */}
            <rect x={40} y={70} width={260} height={140} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={55} y={92} anchor="start" size={11} bold style={{ fill: COLOR.token }}>
              Correct: No trailing space
            </Label>
            <Label x={55} y={112} anchor="start" size={10} mono tone={1}>
              {'prompt = "...<|im_start|>assistant\\n"'}
            </Label>
            <Label x={55} y={135} anchor="start" size={9} tone={2}>
              Token ID: [151644, 77091]
            </Label>
            <Label x={55} y={155} anchor="start" size={9} tone={2}>
              First decoded token: {'" 144"'} (leading space byte)
            </Label>
            <rect x={55} y={175} width={230} height={22} rx={2} fill={DIM.token} />
            <Label x={170} y={189} anchor="middle" size={9} bold style={{ fill: COLOR.token }}>
              Matches generation prompt exactly
            </Label>

            {/* Version B: Accidental space */}
            <rect x={340} y={70} width={260} height={140} rx={4} fill="#141820" stroke={COLOR.error} />
            <Label x={355} y={92} anchor="start" size={11} bold style={{ fill: COLOR.error }}>
              Trap: Trailing space inserted
            </Label>
            <Label x={355} y={112} anchor="start" size={10} mono tone={1}>
              {'prompt = "...<|im_start|>assistant \\n"'}
            </Label>
            <Label x={355} y={135} anchor="start" size={9} tone={2} style={{ fill: COLOR.error }}>
              Token ID: [151644, 8291, 220] (different token IDs!)
            </Label>
            <Label x={355} y={155} anchor="start" size={9} tone={2}>
              First decoded token: {'"144"'} (no space byte)
            </Label>
            <rect x={355} y={175} width={230} height={22} rx={2} fill={DIM.error} />
            <Label x={470} y={189} anchor="middle" size={9} bold style={{ fill: COLOR.error }}>
              Prefix cache misses, hallucinated output
            </Label>

            <rect x={40} y={230} width={560} height={40} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
            <Label x={320} y={254} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
              Byte-level determinism: BPE tokenizers treat `"word "` and `" word"` as different integers.
            </Label>
          </g>
        )}

        {/* Step 4: The Missing EOS Trap */}
        {s.step === 4 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              The Missing EOS Trap: Endless Babbling at Inference
            </Label>

            <rect x={40} y={70} width={560} height={140} rx={4} fill="#141820" stroke={COLOR.error} />
            <Label x={60} y={95} anchor="start" size={11} bold style={{ fill: COLOR.error }}>
              Training without End-Of-Turn Token:
            </Label>

            <g>
              <Span x={60} y={115} w={100} h={30} kind="token" label="Prompt" />
              <Span x={165} y={115} w={150} h={30} kind="token" label="Assistant Response" />
              <Span x={320} y={115} w={100} h={30} kind="error" label="NO EOS TOKEN" />
            </g>

            <Label x={60} y={170} anchor="start" size={10} tone={2}>
              What happens during vLLM / inference serving:
            </Label>
            <Label x={60} y={188} anchor="start" size={9} tone={3}>
              The model never emits the stop token (e.g. &lt;|im_end|&gt; or &lt;|eot_id|&gt;).
            </Label>
            <Label x={60} y={202} anchor="start" size={9} style={{ fill: COLOR.error }}>
              It continues decoding useless hallucinations until hitting max_tokens (e.g. 2,048), wasting 95% of GPU VRAM & FLOPs.
            </Label>

            <rect x={40} y={230} width={560} height={45} rx={4} fill={DIM.token} stroke={COLOR.token} />
            <Label x={320} y={250} anchor="middle" size={10} bold style={{ fill: COLOR.token }}>
              Rule: Always append tokenizer.eos_token to the target during dataset pre-processing!
            </Label>
            <Label x={320} y={264} anchor="middle" size={9} tone={2}>
              Loss must be active on that EOS token so the model learns exactly when to cease generation.
            </Label>
          </g>
        )}

        {/* Step 5: Inference Alignment */}
        {s.step === 5 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              Exact Inference Alignment: add_generation_prompt=True
            </Label>

            <rect x={40} y={70} width={560} height={150} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={60} y={95} anchor="start" size={11} bold tone={1}>
              The Training Prefix vs The Inference Prompt:
            </Label>

            {/* Visual comparison */}
            <Label x={60} y={120} anchor="start" size={9} tone={2}>
              Training sequence prefix:
            </Label>
            <g>
              <Span x={220} y={110} w={120} h={20} kind="memory" label="System Turn" />
              <Span x={345} y={110} w={120} h={20} kind="memory" label="User Turn" />
              <Span x={470} y={110} w={110} h={20} kind="token" label="<|im_start|>assistant\n" />
            </g>

            <Label x={60} y={150} anchor="start" size={9} tone={2}>
              Inference prompt (add_generation_prompt=True):
            </Label>
            <g>
              <Span x={220} y={140} w={120} h={20} kind="memory" label="System Turn" />
              <Span x={345} y={140} w={120} h={20} kind="memory" label="User Turn" />
              <Span x={470} y={140} w={110} h={20} kind="token" label="<|im_start|>assistant\n" />
            </g>

            <Label x={320} y={195} anchor="middle" size={10} bold style={{ fill: COLOR.token }}>
              Identical token sequence = 100% Prefix Cache Hit (Module 10) & Zero Mismatch Drift!
            </Label>

            <rect x={40} y={235} width={560} height={40} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
            <Label x={320} y={258} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
              Use the exact same tokenizer.apply_chat_template() function in training and serving.
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
