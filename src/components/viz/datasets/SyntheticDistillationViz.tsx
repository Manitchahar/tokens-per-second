import { useRef, useState, useEffect } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, COLOR, DIM, fmt } from '../lib/svg';

const STEPS = [
  'The human annotation wall',
  'Self-Instruct: from 175 seeds',
  'Diversity pruning via ROUGE',
  'Alpaca: 52K pairs under $500',
  'Reverse instructions (Backtranslation)',
  'Teacher-to-student distillation',
];

const CAPTIONS = [
  'Human annotation costs $1 to $50 per complex reasoning sample and takes months. Synthetic generation creates high-volume instruction data in hours.',
  'Self-Instruct bootstraps from 175 human seed tasks. An LLM samples 8 existing tasks in-context to propose a completely new instruction.',
  'To prevent mode collapse, new instructions are scored against the existing pool using ROUGE-L. Any prompt with similarity > 0.7 is discarded.',
  'Stanford Alpaca produced 52,000 instruction-response pairs from text-davinci-003 for under $500 in API credits.',
  'Reverse instruction takes pristine reference text (docs, code, textbooks) and prompts an LLM to generate the query it answers, ensuring factual validity.',
  'A large frontier teacher (70B/405B) generates reasoning traces to train a compact 8B student model that runs fast and cheap on edge GPUs.',
];

export default function SyntheticDistillationViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [datasetSize, setDatasetSize] = useState(52000);
  const [pipelineMode, setPipelineMode] = useState<'self-instruct' | 'reverse' | 'distill'>('self-instruct');

  useEffect(() => {
    if (s.step === 0) setDatasetSize(1000);
    else if (s.step === 1) setDatasetSize(5000);
    else if (s.step === 2) setDatasetSize(12000);
    else if (s.step === 3) setDatasetSize(52000);
    else if (s.step === 4) {
      setDatasetSize(20000);
      setPipelineMode('reverse');
    } else if (s.step === 5) {
      setDatasetSize(50000);
      setPipelineMode('distill');
    }
  }, [s.step]);

  const animSize = useTween(datasetSize, 400);

  // Cost comparison: API synthetic cost vs human annotation cost
  // Typical synthetic cost: ~1,000 tokens total per pair (prompt + output). At $0.50/M tokens -> $0.0005/pair.
  // In 2023 Davinci-003 was $0.02/1k tokens -> ~$500 for 52k pairs (~$0.01/pair).
  const synthCost = Math.round(datasetSize * 0.0096);
  const humanCost = Math.round(datasetSize * 2.5); // $2.50 per validated human pair

  return (
    <Stage
      rootRef={ref}
      title="Synthetic Generation & Model Distillation"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'token', 'memory', 'error']}
      readout={
        <Readout
          items={[
            {
              label: 'Synthetic Dataset Size',
              value: `${fmt.int(Math.round(animSize))} pairs`,
              kind: 'token',
              note: `${Math.round(animSize * 0.95)} passed quality filtering`,
            },
            {
              label: 'Synthetic API Cost',
              value: `$${synthCost}`,
              kind: 'compute',
              note: 'Frontier LLM token generation API',
            },
            {
              label: 'Human Annotation Equivalent',
              value: `$${fmt.int(humanCost)}`,
              kind: 'error',
              note: `~$2.50/row: ${(humanCost / Math.max(1, synthCost)).toFixed(0)}× more expensive`,
            },
            {
              label: 'Diversity Rejection Rate',
              value: '18.4%',
              kind: 'idle',
              note: 'Pruned by ROUGE-L > 0.7 overlap',
            },
          ]}
        />
      }
      controls={
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', width: '100%', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <Slider
              label="Generated Pairs"
              value={datasetSize}
              min={1000}
              max={100000}
              step={1000}
              onChange={setDatasetSize}
            />
          </div>
          <Segmented
            value={pipelineMode}
            onChange={(v) => setPipelineMode(v as 'self-instruct' | 'reverse' | 'distill')}
            options={[
              { value: 'self-instruct', label: 'Self-Instruct' },
              { value: 'reverse', label: 'Reverse Prompting' },
              { value: 'distill', label: 'Distillation' },
            ]}
          />
        </div>
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Visualizing synthetic data generation pipelines, Self-Instruct, and distillation">
        {/* Step 0: Human Wall Comparison */}
        {s.step === 0 && (
          <g>
            <Label x={320} y={40} anchor="middle" size={13} bold tone={1}>
              The Economics of Data: Human Annotation vs Synthetic Generation
            </Label>

            {/* Human Bar */}
            <rect x={60} y={80} width={520} height={70} rx={4} fill="#141820" stroke={COLOR.error} />
            <Label x={80} y={105} anchor="start" size={12} bold style={{ fill: COLOR.error }}>
              Human Annotation (Crowdsourced / Contractors)
            </Label>
            <Label x={80} y={125} anchor="start" size={10} tone={2}>
              Cost: $1.00 – $50.00 per verified reasoning prompt | Speed: ~50 pairs / day / person
            </Label>
            <Label x={80} y={140} anchor="start" size={9} tone={3}>
              Vulnerable to label fatigue, subjective bias, and strict vendor NDA bottlenecks
            </Label>

            {/* Synthetic Bar */}
            <rect x={60} y={170} width={520} height={70} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={80} y={195} anchor="start" size={12} bold style={{ fill: COLOR.token }}>
              Synthetic Generation (Frontier LLM API / Local Cluster)
            </Label>
            <Label x={80} y={215} anchor="start" size={10} tone={2}>
              Cost: $0.005 – $0.02 per pair | Speed: 100,000+ pairs / hour across batch queues
            </Label>
            <Label x={80} y={230} anchor="start" size={9} tone={3}>
              Deterministic schemas, programmatic verification, and scalable domain expansion
            </Label>
          </g>
        )}

        {/* Step 1 & 2: Self-Instruct Pipeline & ROUGE Filter */}
        {(s.step === 1 || s.step === 2) && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              {s.step === 1 ? 'The Self-Instruct Loop (Wang et al., 2022)' : 'Diversity Guard: ROUGE-L Overlap Pruning'}
            </Label>

            {/* Pipeline Stage 1: Seed Tasks */}
            <rect x={40} y={75} width={100} height={90} rx={4} fill={DIM.token} stroke={COLOR.token} />
            <Label x={90} y={98} anchor="middle" size={10} bold style={{ fill: COLOR.token }}>
              Seed Pool
            </Label>
            <Label x={90} y={118} anchor="middle" size={16} bold tone={1}>
              175
            </Label>
            <Label x={90} y={140} anchor="middle" size={8} tone={2}>
              Human gold tasks
            </Label>

            {/* Arrow */}
            <line x1={140} y1={120} x2={175} y2={120} stroke="var(--rule-2)" strokeWidth={1.5} markerEnd="url(#arrow)" />

            {/* Pipeline Stage 2: Prompt Sampler */}
            <rect x={180} y={75} width={120} height={90} rx={4} fill="#141820" stroke="var(--rule-2)" />
            <Label x={240} y={98} anchor="middle" size={10} bold tone={1}>
              Few-shot Sample
            </Label>
            <Label x={240} y={118} anchor="middle" size={9} tone={2}>
              6 Seed + 2 Gen
            </Label>
            <Label x={240} y={140} anchor="middle" size={8} tone={3}>
              Random 8-task prompt
            </Label>

            {/* Arrow */}
            <line x1={300} y1={120} x2={335} y2={120} stroke="var(--rule-2)" strokeWidth={1.5} />

            {/* Pipeline Stage 3: LLM Proposer */}
            <rect x={340} y={75} width={120} height={90} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
            <Label x={400} y={98} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
              Teacher LLM
            </Label>
            <Label x={400} y={118} anchor="middle" size={9} tone={1}>
              New Instruction
            </Label>
            <Label x={400} y={140} anchor="middle" size={8} tone={2}>
              + Input / Output
            </Label>

            {/* Arrow */}
            <line x1={460} y1={120} x2={495} y2={120} stroke="var(--rule-2)" strokeWidth={1.5} />

            {/* Pipeline Stage 4: Filter */}
            <rect x={500} y={75} width={105} height={90} rx={4} fill={s.step === 2 ? DIM.error : '#141820'} stroke={s.step === 2 ? COLOR.error : 'var(--rule-2)'} />
            <Label x={552} y={98} anchor="middle" size={10} bold style={{ fill: s.step === 2 ? COLOR.error : 'var(--text)' }}>
              ROUGE-L Check
            </Label>
            <Label x={552} y={118} anchor="middle" size={9} tone={2}>
              Overlap &gt; 0.7?
            </Label>
            <Label x={552} y={140} anchor="middle" size={8} tone={3}>
              {s.step === 2 ? 'Discard Redundant' : 'Accept Novel'}
            </Label>

            {/* Step 2 Callout */}
            {s.step === 2 && (
              <g>
                <rect x={40} y={195} width={560} height={75} rx={4} fill="#141820" stroke={COLOR.error} />
                <Label x={60} y={220} anchor="start" size={10} bold style={{ fill: COLOR.error }}>
                  Why ROUGE-L &lt; 0.7 is Enforced:
                </Label>
                <Label x={60} y={238} anchor="start" size={9} tone={2}>
                  Without strict similarity filtering, LLMs repeatedly generate slight rewordings:
                </Label>
                <Label x={60} y={252} anchor="start" size={9} mono tone={3}>
                  {"'Write a poem about dogs' -> 'Author a verse regarding puppies' (ROUGE-L = 0.82) -> REJECTED"}
                </Label>
              </g>
            )}
          </g>
        )}

        {/* Step 3: Alpaca Scale */}
        {s.step === 3 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              Stanford Alpaca (Taori et al., March 2023)
            </Label>

            <rect x={50} y={70} width={540} height={140} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={70} y={98} anchor="start" size={12} bold style={{ fill: COLOR.token }}>
              52,000 Instruction-Following Pairs Generated for $500
            </Label>
            <Label x={70} y={122} anchor="start" size={10} tone={2}>
              Teacher Engine: OpenAI text-davinci-003 | Student Model: LLaMA-7B
            </Label>
            <Label x={70} y={142} anchor="start" size={9} tone={3}>
              Proved that a small open-weights model could learn fluent conversational instruction following in 3 hours of SFT.
            </Label>

            <g>
              <Span x={70} y={160} w={150} h={25} kind="compute" label="175 Human Seeds" />
              <Span x={230} y={160} w={200} h={25} kind="token" label="52,000 Synthetic Pairs" />
              <Span x={440} y={160} w={130} h={25} kind="memory" label="3 Epochs on 8x A100" />
            </g>

            <rect x={50} y={230} width={540} height={50} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
            <Label x={320} y={252} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
              The Synthesis Breakthrough: SFT is Bound by Strategy, Not Million-Dollar Budgets
            </Label>
            <Label x={320} y={268} anchor="middle" size={9} tone={2}>
              Demonstrated that frontier models can act as infinite teaching engines for smaller models.
            </Label>
          </g>
        )}

        {/* Step 4: Reverse Instructions (Backtranslation) */}
        {s.step === 4 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              Reverse Instruction: Grounding Answers in Real Documents
            </Label>

            <rect x={40} y={70} width={260} height={150} rx={4} fill="#141820" stroke={COLOR.memory} />
            <Label x={55} y={95} anchor="start" size={11} bold style={{ fill: COLOR.memory }}>
              1. Authentic Source Grounding
            </Label>
            <Label x={55} y={115} anchor="start" size={9} tone={2}>
              Take real textbook, Python library, or RFC:
            </Label>
            <rect x={55} y={125} width={230} height={45} rx={2} fill={DIM.memory} />
            <Label x={65} y={142} anchor="start" size={8.5} mono tone={1}>
              def attend(q, k, v): ...
            </Label>
            <Label x={65} y={156} anchor="start" size={8.5} mono tone={1}>
              return softmax(q @ k.T) @ v
            </Label>
            <Label x={55} y={195} anchor="start" size={8} tone={3}>
              Target answer is guaranteed factually sound!
            </Label>

            <line x1={300} y1={145} x2={340} y2={145} stroke="var(--rule-2)" strokeWidth={1.5} />

            <rect x={340} y={70} width={260} height={150} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={355} y={95} anchor="start" size={11} bold style={{ fill: COLOR.token }}>
              2. Backtranslate the Prompt
            </Label>
            <Label x={355} y={115} anchor="start" size={9} tone={2}>
              Ask LLM: What user question produces this?
            </Label>
            <rect x={355} y={125} width={230} height={45} rx={2} fill={DIM.token} />
            <Label x={365} y={142} anchor="start" size={8.5} tone={1}>
              &quot;Implement naive scaled dot product&quot;
            </Label>
            <Label x={365} y={156} anchor="start" size={8.5} tone={1}>
              &quot;attention in pure PyTorch.&quot;
            </Label>
            <Label x={355} y={195} anchor="start" size={8} tone={3}>
              Synthetic prompt + Real answer = Zero hallucination
            </Label>
          </g>
        )}

        {/* Step 5: Distillation to Edge Student */}
        {s.step === 5 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              Teacher-to-Student Distillation Architecture
            </Label>

            {/* Teacher */}
            <rect x={60} y={70} width={220} height={120} rx={4} fill="#141820" stroke={COLOR.compute} />
            <Label x={170} y={95} anchor="middle" size={12} bold style={{ fill: COLOR.compute }}>
              Frontier Teacher
            </Label>
            <Label x={170} y={115} anchor="middle" size={10} tone={2}>
              Llama-3.1-405B / Qwen2.5-72B
            </Label>
            <Label x={170} y={135} anchor="middle" size={9} tone={3}>
              Slow, expensive, deep multi-step reasoning
            </Label>
            <Span x={80} y={148} w={180} h={22} kind="compute" label="Generates Chain-of-Thought" />

            {/* Transfer arrow */}
            <g>
              <line x1={280} y1={130} x2={355} y2={130} stroke={COLOR.token} strokeWidth={2} />
              <polygon points="355,126 365,130 355,134" fill={COLOR.token} />
              <Label x={320} y={120} anchor="middle" size={8.5} bold style={{ fill: COLOR.token }}>
                SFT Pairs
              </Label>
            </g>

            {/* Student */}
            <rect x={365} y={70} width={220} height={120} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={475} y={95} anchor="middle" size={12} bold style={{ fill: COLOR.token }}>
              Compact Student
            </Label>
            <Label x={475} y={115} anchor="middle" size={10} tone={2}>
              Llama-3.1-8B / Qwen2.5-1.5B
            </Label>
            <Label x={475} y={135} anchor="middle" size={9} tone={3}>
              Fast, fits on 1 GPU, 80+ tokens/s serving
            </Label>
            <Span x={385} y={148} w={180} h={22} kind="token" label="Matches Teacher Behaviors" />

            <rect x={60} y={215} width={525} height={60} rx={4} fill={DIM.memory} stroke={COLOR.memory} />
            <Label x={320} y={236} anchor="middle" size={10} bold style={{ fill: COLOR.memory }}>
              Inference Speedup: The 8B student runs 12× faster with 90% lower serving costs!
            </Label>
            <Label x={320} y={255} anchor="middle" size={9} tone={2}>
              Terms of service check: Verify teacher provider allows model training (e.g. Apache 2.0 open weights).
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
