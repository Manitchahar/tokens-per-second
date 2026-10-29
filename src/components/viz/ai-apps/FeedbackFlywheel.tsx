import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, Arrow, COLOR, fmt } from '../lib/svg';

const STEPS = [
  'The static model trap',
  'Explicit signals: Thumbs up / down',
  'Implicit signals: Copies & regenerations',
  'Synthesizing (x, y_w, y_l) preference pairs',
  'The continuous DPO alignment flywheel',
  'The danger of sycophantic feedback loops',
  'Designing high-integrity feedback UIs',
];

const CAPTIONS = [
  'Inference runs against static weights. Without capturing production interactions, model errors never get resolved, creating a stagnant system that cannot adapt.',
  'Explicit thumbs-up/down ratings are direct, but suffer from low engagement: only 1% to 3% of sessions click, creating strong selection bias toward extreme outrage or delight.',
  'Implicit signals capture 40% to 60% of sessions without user friction: copy-to-clipboard indicates success; immediate regenerations and prompt edits signal failure.',
  'When a user clicks "Regenerate" or edits the text, the system captures a preference pair: the initial output is rejected (y_l), and the edit/regeneration is chosen (y_w).',
  'Cleaned preference pairs feed directly into Direct Preference Optimization (DPO, Module 3C) and dynamic few-shot retrieval banks, closing the data flywheel.',
  'If a model is trained uncritically on user clicks, it learns sycophancy: validating user misconceptions and telling users what they want to hear rather than the truth.',
  'A production feedback interface pairs frictionless inline editing with rigorous validation filters to discard toxic noise before pairs enter the fine-tuning queue.',
];

export default function FeedbackFlywheel() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [implicitShare, setImplicitShare] = useState(55);
  const [explicitShare, setExplicitShare] = useState(2);

  // Volume arithmetic:
  // 100,000 sessions/day
  const dailySessions = 100000;
  const explicitEvents = Math.round(dailySessions * (explicitShare / 100));
  const implicitEvents = Math.round(dailySessions * (implicitShare / 100));
  const dpoPairsDaily = Math.round(implicitEvents * 0.12); // ~12% yield clean pairs

  return (
    <Stage
      rootRef={ref}
      title="The user feedback data flywheel and preference pair synthesis"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['comm', 'user interaction'],
        ['memory', 'implicit telemetry'],
        ['compute', 'DPO training loop'],
        ['token', 'chosen completion (y_w)'],
        ['error', 'rejected completion (y_l)'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Explicit volume',
              value: `${explicitEvents.toLocaleString()}/day`,
              kind: 'comm',
              note: `${explicitShare}% engagement rate`,
            },
            {
              label: 'Implicit volume',
              value: `${implicitEvents.toLocaleString()}/day`,
              kind: 'token',
              note: `${implicitShare}% behavioral signals`,
            },
            {
              label: 'Clean DPO pairs',
              value: `${dpoPairsDaily.toLocaleString()}/day`,
              kind: 'compute',
              note: 'derived from edits & retries',
            },
            {
              label: 'Flywheel status',
              value: s.step >= 4 ? 'Active Loop' : 'Open Loop',
              kind: s.step >= 4 ? 'token' : 'error',
              note: s.step >= 4 ? 'continuous improvement' : 'static weights',
            },
          ]}
        />
      }
      controls={
        s.step >= 2 ? (
          <>
            <Slider
              label="Implicit signal capture rate"
              value={implicitShare}
              min={20}
              max={80}
              step={5}
              onChange={setImplicitShare}
              format={(v) => `${v}%`}
              kind="token"
            />
            {s.step === 1 && (
              <Slider
                label="Explicit click rate"
                value={explicitShare}
                min={1}
                max={10}
                step={1}
                onChange={setExplicitShare}
                format={(v) => `${v}%`}
                kind="comm"
              />
            )}
          </>
        ) : undefined
      }
      {...s}
    >
      <svg
        viewBox="0 0 640 330"
        role="img"
        aria-label="Feedback flywheel showing explicit thumbs up/down, implicit copy and regeneration telemetry, preference pair extraction, and DPO alignment training."
      >
        {/* Step 0: The Static Model Trap */}
        {s.step === 0 && (
          <g>
            <rect x={40} y={50} width={220} height={100} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
            <Label x={150} y={74} anchor="middle" tone={1} size={11} bold>Production Inference</Label>
            <Label x={55} y={96} anchor="start" tone={2} size={9.5}>• Model generates text</Label>
            <Label x={55} y={114} anchor="start" tone={2} size={9.5}>• Weights stay completely frozen</Label>
            <Label x={55} y={132} anchor="start" tone={3} size={9}>• No feedback captured</Label>

            <Arrow x1={260} y1={100} x2={380} y2={100} kind="error" width={2} />

            <rect x={380} y={50} width={220} height={100} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.2} />
            <Label x={490} y={74} anchor="middle" tone={1} size={11} bold kind="error">The Open Loop Defect</Label>
            <Label x={395} y={96} anchor="start" tone={2} size={9.5}>• When the model makes an error,</Label>
            <Label x={395} y={114} anchor="start" tone={2} size={9.5}>it repeats it tomorrow.</Label>
            <Label x={395} y={132} anchor="start" tone={1} size={9.5} bold kind="error">Zero compounding learning.</Label>

            <rect x={80} y={190} width={480} height={70} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
            <Label x={320} y={215} anchor="middle" tone={1} size={11} bold>The Core Dilemma</Label>
            <Label x={320} y={235} anchor="middle" tone={2} size={9.5}>Inference creates data, but without a capture harness, all user interactions vanish forever.</Label>
          </g>
        )}

        {/* Steps >= 1: Feedback Pipeline & Flywheel */}
        {s.step >= 1 && (
          <g>
            {/* User Interface Box */}
            <rect x={20} y={90} width={130} height={130} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1.2} />
            <Label x={85} y={112} anchor="middle" tone={1} size={11} bold>Chat Interface</Label>

            {/* Thumbs up/down buttons */}
            <rect x={32} y={125} width={48} height={24} rx={3} fill="var(--comm-dim)" stroke={COLOR.comm} strokeWidth={1} />
            <Label x={56} y={141} anchor="middle" tone={1} size={10} bold kind="comm">👍</Label>

            <rect x={88} y={125} width={48} height={24} rx={3} fill="var(--comm-dim)" stroke={COLOR.comm} strokeWidth={1} />
            <Label x={112} y={141} anchor="middle" tone={1} size={10} bold kind="comm">👎</Label>
            <Label x={85} y={162} anchor="middle" tone={3} size={8.5}>{`Explicit: ~${explicitShare}%`}</Label>

            {/* Implicit signals */}
            {s.step >= 2 && (
              <g>
                <rect x={32} y={172} width={104} height={38} rx={3} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1} />
                <Label x={84} y={187} anchor="middle" tone={1} size={8.5} bold kind="memory">Implicit Telemetry</Label>
                <Label x={84} y={200} anchor="middle" tone={2} size={7.5}>{`Copy, Retry, Edit (${implicitShare}%)`}</Label>
              </g>
            )}

            {/* Arrow to Event Collector */}
            <Arrow x1={150} y1={155} x2={190} y2={155} kind="comm" width={1.8} />

            {/* Preference Pair Synthesizer (Step 3+) */}
            {s.step >= 3 ? (
              <g>
                <rect x={190} y={80} width={190} height={150} rx={4} fill="var(--bg-card)" stroke={COLOR.token} strokeWidth={1.5} />
                <Label x={285} y={102} anchor="middle" tone={1} size={10.5} bold kind="token">Preference Pair Miner</Label>

                {/* Prompt block */}
                <rect x={202} y={112} width={166} height={22} rx={2} fill="var(--comm-dim)" stroke={COLOR.comm} strokeWidth={1} />
                <Label x={285} y={127} anchor="middle" tone={1} size={8.5} bold kind="comm">x: User Prompt</Label>

                {/* Chosen block y_w */}
                <rect x={202} y={140} width={166} height={34} rx={2} fill="var(--token-dim)" stroke={COLOR.token} strokeWidth={1} />
                <Label x={285} y={154} anchor="middle" tone={1} size={8.5} bold kind="token">y_w (Chosen Response)</Label>
                <Label x={285} y={167} anchor="middle" tone={2} size={7.5}>Regenerated answer or user edit</Label>

                {/* Rejected block y_l */}
                <rect x={202} y={180} width={166} height={34} rx={2} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1} />
                <Label x={285} y={194} anchor="middle" tone={1} size={8.5} bold kind="error">y_l (Rejected Response)</Label>
                <Label x={285} y={207} anchor="middle" tone={2} size={7.5}>Initial output discarded by user</Label>
              </g>
            ) : (
              <g>
                <rect x={190} y={110} width={150} height={90} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
                <Label x={265} y={140} anchor="middle" tone={1} size={10} bold>Raw Telemetry Store</Label>
                <Label x={265} y={160} anchor="middle" tone={2} size={8.5}>Clicks & Timestamps</Label>
              </g>
            )}

            {/* DPO Training Loop (Step 4+) */}
            {s.step >= 4 ? (
              <g>
                <Arrow x1={380} y1={155} x2={430} y2={155} kind="compute" width={2} />

                <rect x={430} y={80} width={180} height={150} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.5} />
                <Label x={520} y={102} anchor="middle" tone={1} size={10.5} bold kind="compute">Continuous DPO Engine</Label>
                <Label x={445} y={124} anchor="start" tone={2} size={8.5}>• Loss: L_DPO(π_θ; π_ref)</Label>
                <Label x={445} y={142} anchor="start" tone={2} size={8.5}>• Increases log-odds of y_w</Label>
                <Label x={445} y={160} anchor="start" tone={2} size={8.5}>• Decreases log-odds of y_l</Label>
                <Label x={445} y={178} anchor="start" tone={1} size={8.5} bold kind="compute">• LoRA checkpoint exported</Label>
                <Label x={445} y={196} anchor="start" tone={3} size={8}>Evaluated on Golden Test Set</Label>

                {/* Return arrow closing the flywheel */}
                <path d="M 520 80 L 520 30 L 85 30 L 85 90" fill="none" stroke={COLOR.token} strokeWidth={1.5} strokeDasharray="3 3" />
                <Label x={300} y={22} anchor="middle" tone={1} size={9} kind="token">Updated Model Weights / Dynamic Few-Shot Banks</Label>
              </g>
            ) : (
              s.step >= 3 && <Arrow x1={380} y1={155} x2={480} y2={155} kind="token" />
            )}

            {/* Step 5: Sycophancy Warning */}
            {s.step === 5 && (
              <g>
                <rect x={80} y={245} width={480} height={60} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.2} />
                <Label x={320} y={268} anchor="middle" tone={1} size={10.5} bold kind="error">The Sycophancy Hazard</Label>
                <Label x={320} y={286} anchor="middle" tone={2} size={9}>When trained purely on user clicks, models learn to flatter biases and agree with errors instead of telling the truth.</Label>
              </g>
            )}

            {/* Step 6: Safe UI Design */}
            {s.step === 6 && (
              <g>
                <rect x={80} y={245} width={480} height={60} rx={4} fill="var(--token-dim)" stroke={COLOR.token} strokeWidth={1.2} />
                <Label x={320} y={268} anchor="middle" tone={1} size={10.5} bold kind="token">High-Integrity Feedback Interface</Label>
                <Label x={320} y={286} anchor="middle" tone={2} size={9}>Inline markdown edits, anti-sycophancy golden sets, and automated toxicity filters protect training queue hygiene.</Label>
              </g>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
