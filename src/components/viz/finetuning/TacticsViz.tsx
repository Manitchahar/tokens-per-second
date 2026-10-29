import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade } from '../lib/svg';

const STEPS = [
  'Loss Masking: Target only the response',
  'The failure mode: What unmasked loss causes',
  'Effective Batch: Micro-batch × Accumulation × GPUs',
  'Learning Rate: Why LoRA needs 1e-4 (not 1e-5)',
  'Epochs 1 to 3: The forgetting boundary',
  'Reading Loss Curves: Healthy vs Overfitting',
];

const CAPTIONS = [
  'In Supervised Finetuning, user prompt tokens must be excluded from loss calculation. By setting their labels to -100, PyTorch cross-entropy computes gradients exclusively on response tokens.',
  'If you compute loss over prompt tokens, the model wastes parameter updates learning to predict user inputs. It degrades reasoning and starts echoing prompt templates back to users.',
  'Effective batch size = micro-batch × gradient accumulation steps × GPUs. Accumulating gradients over multiple micro-steps simulates a large batch without overflowing GPU VRAM.',
  'Full-parameter fine-tuning uses learning rates of 1e-5 to 2e-5. LoRA requires ~10× higher (1e-4 to 2e-4) because updates are constrained to a low-rank subspace and must take larger steps.',
  'Most instruction datasets achieve optimal alignment in 1 to 3 epochs. Training for 5+ epochs causes catastrophic forgetting, repetitive generations, and benchmark degradation.',
  'Monitor validation loss closely. A healthy run shows train and validation loss declining together. If validation loss begins climbing while training loss plunges, stop training immediately.',
];

const CURVES = [
  { name: 'Healthy Run', desc: 'Train & Val drop together. Generalization preserved.', finalVal: 1.24 },
  { name: 'Overfitting', desc: 'Train drops to 0.1, Val climbs to 2.8. Memorizing data.', finalVal: 2.85 },
  { name: 'LR Too High (Divergence)', desc: 'Loss oscillates violently and explodes to NaN.', finalVal: 9.99 },
];

export default function TacticsViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [curveIdx, setCurveIdx] = useState(0);

  // Micro-batch arithmetic: micro=2, accum=8, gpus=4 => 64
  const micro = 2;
  const accum = 8;
  const gpus = 4;
  const effectiveBatch = micro * accum * gpus;

  return (
    <Stage
      rootRef={ref}
      title="Finetuning Tactics: Loss Masking, Batch Dynamics & Curves"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['token', 'active loss tokens (labels != -100)'],
        ['idle', 'masked tokens (labels == -100)'],
        ['compute', 'training loss trajectory'],
        ['error', 'validation loss / OOM spike'],
      ]}
      controls={
        st === 5 ? (
          <Segmented
            options={CURVES.map((c) => c.name)}
            value={CURVES[curveIdx].name}
            onChange={(val) => {
              const idx = CURVES.findIndex((c) => c.name === val);
              if (idx !== -1) setCurveIdx(idx);
            }}
          />
        ) : undefined
      }
      readout={
        st === 2 ? (
          <Readout
            items={[
              { label: 'Micro-Batch', value: `${micro}` },
              { label: 'Accumulation Steps', value: `${accum}` },
              { label: 'GPUs', value: `${gpus}` },
              { label: 'Effective Batch', value: `${effectiveBatch} seqs` },
            ]}
          />
        ) : st === 5 ? (
          <Readout
            items={[
              { label: 'Curve Profile', value: CURVES[curveIdx].name },
              { label: 'Final Val Loss', value: CURVES[curveIdx].finalVal.toFixed(2) },
              { label: 'Diagnosis', value: CURVES[curveIdx].desc },
            ]}
          />
        ) : (
          <Readout
            items={[
              { label: 'Tactic Focus', value: STEPS[st] },
              { label: 'Core Metric', value: st === 0 || st === 1 ? 'Prompt Labels = -100' : st === 3 ? 'LR = 2e-4 (LoRA)' : 'Epochs = 1 to 3' },
            ]}
          />
        )
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Visual explanation of loss masking, effective batch size, and loss curve diagnostics">
        {/* UPPER PANEL: Loss Masking Sequence Visualizer (Steps 0, 1) */}
        <g className="token-masking-display">
          <Label x={32} y={24} tone={1} bold size={11.5} mono={false}>
            Token Stream & Loss Masking Structure
          </Label>

          {/* Tokens row */}
          {/* Prompt Tokens (Masked in Step 0, unmasked in Step 1) */}
          <Span
            x={32}
            y={35}
            w={65}
            h={28}
            kind={st === 1 ? 'error' : 'idle'}
            label="<|im_start|>"
          />
          <Span
            x={100}
            y={35}
            w={60}
            h={28}
            kind={st === 1 ? 'error' : 'idle'}
            label="user"
          />
          <Span
            x={163}
            y={35}
            w={120}
            h={28}
            kind={st === 1 ? 'error' : 'idle'}
            label="Extract JSON schema"
          />
          <Span
            x={286}
            y={35}
            w={60}
            h={28}
            kind={st === 1 ? 'error' : 'idle'}
            label="<|im_end|>"
          />

          {/* Assistant Response Tokens (Active loss) */}
          <Span
            x={349}
            y={35}
            w={55}
            h={28}
            kind="token"
            label="assistant"
          />
          <Span
            x={407}
            y={35}
            w={105}
            h={28}
            kind="token"
            label={'{\\"status\\":\\"ok\\"}'}
          />
          <Span
            x={515}
            y={35}
            w={45}
            h={28}
            kind="token"
            label="<|im_end|>"
          />

          {/* Padding Token (Masked) */}
          <Span
            x={563}
            y={35}
            w={45}
            h={28}
            kind="idle"
            label="<pad>"
          />

          {/* Labels row under tokens */}
          <Label x={160} y={78} tone={st === 1 ? 1 : 3} size={9} mono anchor="middle">
            {st === 1 ? 'Label != -100 (WRONG: gradients computed!)' : 'Prompt Tokens: Label = -100 (Gradient = 0)'}
          </Label>
          <Label x={450} y={78} kind="token" bold size={9} mono anchor="middle">
            Response Tokens: Active Cross-Entropy Loss
          </Label>
          <Label x={585} y={78} tone={3} size={8.5} mono anchor="middle">
            -100
          </Label>
        </g>

        {/* MIDDLE / LOWER PANEL: Dynamic based on Step */}
        {/* Steps 0 to 4: Tactical Explanations & Diagrams */}
        {st < 5 && (
          <g className="tactical-diagrams">
            <line x1={32} y1={98} x2={608} y2={98} stroke="var(--rule-2)" strokeWidth={1} />

            {/* Step 2: Effective Batch Diagram */}
            {st === 2 && (
              <g>
                <Label x={32} y={120} tone={1} bold size={12} mono={false}>
                  Simulating Big Batches with Gradient Accumulation
                </Label>
                {/* 4 GPUs */}
                {[0, 1, 2, 3].map((gpuIdx) => (
                  <g key={`gpu-${gpuIdx}`}>
                    <rect
                      x={32 + gpuIdx * 146}
                      y={135}
                      width={136}
                      height={50}
                      rx={4}
                      fill="rgba(255, 255, 255, 0.03)"
                      stroke="var(--rule-2)"
                    />
                    <Label x={42 + gpuIdx * 146} y={152} tone={2} bold size={10} mono>
                      GPU {gpuIdx}: micro_batch = 2
                    </Label>
                    <Label x={42 + gpuIdx * 146} y={170} tone={3} size={8.5} mono>
                      8 accum steps = 16 seqs
                    </Label>
                  </g>
                ))}
                <Label x={320} y={202} kind="token" bold size={11} mono anchor="middle">
                  Total Effective Batch = 4 GPUs × (2 micro × 8 accum) = 64 Sequences
                </Label>
              </g>
            )}

            {/* Step 3: Learning Rate Comparison */}
            {st === 3 && (
              <g>
                <Label x={32} y={120} tone={1} bold size={12} mono={false}>
                  The Learning Rate Gap: Full SFT vs LoRA
                </Label>
                {/* Full SFT bar */}
                <rect x={32} y={135} width={280} height={55} rx={4} fill="rgba(247, 144, 9, 0.05)" stroke="var(--rule-2)" />
                <Label x={44} y={154} tone={1} bold size={11} mono={false}>
                  Full-Parameter SFT: 1e-5 to 2e-5
                </Label>
                <Label x={44} y={172} tone={3} size={9.5} mono>
                  All 8B weights update. High LR causes catastrophic collapse.
                </Label>

                {/* LoRA bar */}
                <rect x={328} y={135} width={280} height={55} rx={4} fill="rgba(18, 183, 106, 0.05)" stroke="var(--token)" />
                <Label x={340} y={154} kind="token" bold size={11} mono={false}>
                  LoRA / QLoRA: 1e-4 to 2e-4 (~10× Higher)
                </Label>
                <Label x={340} y={172} tone={2} size={9.5} mono>
                  Constrained low-rank subspace requires larger step sizes.
                </Label>
              </g>
            )}

            {/* Step 4: Epochs & The LIMA Rule */}
            {st === 4 && (
              <g>
                <Label x={32} y={120} tone={1} bold size={12} mono={false}>
                  Epoch Boundaries: Quality Over Quantity (The LIMA Principle)
                </Label>
                <rect x={32} y={135} width={180} height={55} rx={4} fill="rgba(18, 183, 106, 0.08)" stroke="var(--token)" />
                <Label x={44} y={155} kind="token" bold size={11} mono>
                  Epoch 1 · Sweet Spot
                </Label>
                <Label x={44} y={172} tone={2} size={8.5} mono>
                  Learns format; zero forgetting.
                </Label>

                <rect x={228} y={135} width={180} height={55} rx={4} fill="rgba(247, 144, 9, 0.08)" stroke="var(--rule-2)" />
                <Label x={240} y={155} tone={1} bold size={11} mono>
                  Epoch 2–3 · Maximum
                </Label>
                <Label x={240} y={172} tone={2} size={8.5} mono>
                  Refines subtle edge cases.
                </Label>

                <rect x={424} y={135} width={184} height={55} rx={4} fill="rgba(240, 68, 56, 0.08)" stroke="var(--error)" />
                <Label x={436} y={155} kind="error" bold size={11} mono>
                  Epoch 5+ · Danger Zone
                </Label>
                <Label x={436} y={172} tone={3} size={8.5} mono>
                  Memorization & catastrophic collapse.
                </Label>
              </g>
            )}

            {/* Step 0 & 1 Explanations */}
            {(st === 0 || st === 1) && (
              <g>
                <Label x={32} y={120} tone={1} bold size={12} mono={false}>
                  {st === 0 ? 'Why Loss Masking Is Mandatory in SFT' : 'The Catastrophic Result of Unmasked Loss'}
                </Label>
                <Label x={32} y={142} tone={2} size={10.5} mono={false}>
                  {st === 0
                    ? 'Cross-entropy should evaluate: "Given the prompt x, how accurately does the model predict the next response token y_t?"'
                    : 'When you compute loss on the user prompt, the model is penalized for not predicting the user’s exact wording!'}
                </Label>
                <Label x={32} y={160} tone={2} size={10.5} mono={false}>
                  {st === 0
                    ? 'PyTorch loss function: nn.CrossEntropyLoss(ignore_index=-100). Any label set to -100 produces 0 loss and 0 gradients.'
                    : 'Symptoms of unmasked training: the model starts generating its own prompts, mimics question tags, and repeats user instructions.'}
                </Label>
                <Label x={32} y={178} tone={3} size={10} mono={false}>
                  In your dataset collation script, always locate the start of the assistant header and mask all prior token IDs to -100.
                </Label>
              </g>
            )}
          </g>
        )}

        {/* STEP 5: Interactive Loss Curve Visualizer */}
        {st === 5 && (
          <g className="loss-curve-visualizer">
            <line x1={40} y1={210} x2={600} y2={210} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={40} y1={110} x2={40} y2={210} stroke="var(--rule-2)" strokeWidth={1} />

            <Label x={40} y={105} tone={3} size={8.5} mono>
              Loss
            </Label>
            <Label x={600} y={222} tone={3} size={8.5} mono anchor="end">
              Training Steps →
            </Label>

            {/* Curve 0: Healthy */}
            {curveIdx === 0 && (
              <g>
                {/* Train loss curve (Compute: Amber) */}
                <path
                  d="M 40 190 Q 120 180 200 150 T 360 135 T 580 128"
                  fill="none"
                  stroke="var(--compute)"
                  strokeWidth={2}
                />
                {/* Val loss curve (Token: Green) */}
                <path
                  d="M 40 195 Q 120 185 200 158 T 360 145 T 580 138"
                  fill="none"
                  stroke="var(--token)"
                  strokeWidth={2}
                  strokeDasharray="4 2"
                />
                <Label x={585} y={126} kind="compute" size={9} mono>
                  Train: 0.95
                </Label>
                <Label x={585} y={142} kind="token" size={9} mono>
                  Val: 1.24
                </Label>
              </g>
            )}

            {/* Curve 1: Overfitting */}
            {curveIdx === 1 && (
              <g>
                {/* Train loss plunges */}
                <path
                  d="M 40 190 Q 120 180 200 130 T 360 115 T 580 105"
                  fill="none"
                  stroke="var(--compute)"
                  strokeWidth={2}
                />
                {/* Val loss bottoms out then climbs */}
                <path
                  d="M 40 195 Q 120 185 200 155 Q 320 150 420 175 T 580 205"
                  fill="none"
                  stroke="var(--error)"
                  strokeWidth={2}
                  strokeDasharray="4 2"
                />
                <Label x={585} y={105} kind="compute" size={9} mono>
                  Train: 0.12
                </Label>
                <Label x={585} y={205} kind="error" size={9} mono>
                  Val: 2.85 (Overfit!)
                </Label>
              </g>
            )}

            {/* Curve 2: Divergence / Exploding */}
            {curveIdx === 2 && (
              <g>
                <path
                  d="M 40 190 L 100 160 L 160 185 L 220 130 L 280 195 L 340 115 L 400 208"
                  fill="none"
                  stroke="var(--error)"
                  strokeWidth={2}
                />
                <Label x={420} y={208} kind="error" bold size={11} mono>
                  NaN / Exploding Gradient!
                </Label>
              </g>
            )}
          </g>
        )}

        {/* Lower Detail Card */}
        <rect
          x={32}
          y={232}
          width={576}
          height={72}
          rx={6}
          fill="rgba(0, 0, 0, 0.25)"
          stroke="var(--rule-2)"
          strokeWidth={1}
        />

        <Fade show={st === 0}>
          <Label x={46} y={252} tone={1} bold size={11} mono={false}>
            Golden Rule 1 · Loss Masking (labels = -100)
          </Label>
          <Label x={46} y={268} tone={2} size={10} mono={false}>
            Only response tokens should have valid target IDs. All prompt tokens and padding must be set to -100.
          </Label>
          <Label x={46} y={284} tone={3} size={9.5} mono={false}>
            PyTorch cross-entropy automatically skips index -100 during loss calculation and backward pass.
          </Label>
        </Fade>

        <Fade show={st === 1}>
          <Label x={46} y={252} kind="error" bold size={11} mono={false}>
            Warning · Unmasked Training Destroys Model Behavior
          </Label>
          <Label x={46} y={268} tone={1} size={10} mono={false}>
            If you train on prompts, the model treats user questions as ground truth to memorize.
          </Label>
          <Label x={46} y={284} tone={3} size={9.5} mono={false}>
            The model will lose prompt-adherence, echo system headers, and hallucinate conversation turns.
          </Label>
        </Fade>

        <Fade show={st === 2}>
          <Label x={46} y={252} tone={1} bold size={11} mono={false}>
            Golden Rule 2 · Effective Batch Size Architecture
          </Label>
          <Label x={46} y={268} tone={2} size={10} mono={false}>
            Formula: Effective Batch = micro_batch × gradient_accumulation_steps × num_gpus.
          </Label>
          <Label x={46} y={284} tone={3} size={9.5} mono={false}>
            For stable SFT on 8B models, target an effective batch size between 32 and 128 sequences.
          </Label>
        </Fade>

        <Fade show={st === 3}>
          <Label x={46} y={252} tone={1} bold size={11} mono={false}>
            Golden Rule 3 · Learning Rate: 1e-4 for LoRA, 2e-5 for Full SFT
          </Label>
          <Label x={46} y={268} tone={2} size={10} mono={false}>
            LoRA parameters begin at 0 and update only 0.3% of weights, demanding higher learning rates.
          </Label>
          <Label x={46} y={284} tone={3} size={9.5} mono={false}>
            Always pair with a linear warmup (3% of total steps) and a cosine learning rate decay scheduler.
          </Label>
        </Fade>

        <Fade show={st === 4}>
          <Label x={46} y={252} tone={1} bold size={11} mono={false}>
            Golden Rule 4 · Epoch Budgeting: 1 to 3 Epochs Only
          </Label>
          <Label x={46} y={268} tone={2} size={10} mono={false}>
            LIMA proved that 1,000 carefully curated demonstrations in 1–2 epochs achieve peak assistant quality.
          </Label>
          <Label x={46} y={284} tone={3} size={9.5} mono={false}>
            More epochs do not teach more intelligence; they cause catastrophic forgetting of pre-trained reasoning.
          </Label>
        </Fade>

        <Fade show={st === 5}>
          <Label x={46} y={252} tone={1} bold size={11} mono={false}>
            Golden Rule 5 · Reading Loss Curves in Real Time
          </Label>
          <Label x={46} y={268} tone={2} size={10} mono={false}>
            Always evaluate on a held-out validation set every 50–100 steps. Select scenarios above to inspect!
          </Label>
          <Label x={46} y={284} tone={3} size={9.5} mono={false}>
            If validation loss diverges upward while train loss falls, checkpoint checkpointing has passed its peak.
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
