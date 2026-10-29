import { useRef, useState } from 'react';
import { Stage, Slider, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, COLOR } from '../lib/svg';

const STEPS = [
  'Component-level decomposition: isolate every failure',
  'Rubrics with anchor examples',
  'Slicing and Simpson’s paradox: the aggregation trap',
  'Sample size and standard error: SE = sqrt(p(1-p)/n)',
  'Worked arithmetic: n=100 vs n=1000 sample bounds',
  'The evaluation loop: offline golden sets to online telemetry',
];

const CAPTIONS = [
  'Never evaluate an AI application as a monolithic black box. Decompose the pipeline into discrete stages (query rewrite, retrieval, generation, guardrails) to pinpoint bugs.',
  'Vague rubrics produce drifting scores. Anchor rubrics provide explicit positive and negative boundary examples for each numerical score on the scale.',
  'Simpson’s paradox: Model B beats Model A overall (77% vs 58%), yet Model A is strictly superior on every individual slice (Easy: 90% vs 85%, Hard: 50% vs 45%). Always slice your data.',
  'Sample size controls statistical resolution. Standard error is SE = sqrt(p(1-p)/n). A small test set of 100 samples cannot distinguish a 3% quality regression from random noise.',
  'At p=0.8: n=100 gives SE=4.0% (95% CI width ~15.7%); n=1000 gives SE=1.26% (95% CI width ~4.9%). Detecting a 3x smaller delta requires roughly 9x to 10x more samples.',
  'Production evaluation is a continuous loop: offline golden regression suites gate PR merges, while online shadow evaluation and implicit telemetry monitor live drift.',
];

export default function EvalPipelineVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [sampleN, setSampleN] = useState(100);
  const [activeSlice, setActiveSlice] = useState<'easy' | 'hard' | 'overall'>('overall');

  // Standard error at p=0.8
  const p = 0.8;
  const se = Math.sqrt((p * (1 - p)) / sampleN);
  const margin95 = 1.96 * se;

  return (
    <Stage
      rootRef={ref}
      title="Designing Your Evaluation Pipeline: Slicing, Rubrics, and Sample Size"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'error', 'memory', 'compute']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Sample size (n)', value: `${sampleN}`, kind: 'text' },
          { label: 'Standard Error (SE)', value: `${(se * 100).toFixed(2)}%`, kind: se > 0.03 ? 'error' : 'token', note: `at baseline p = 80%` },
          { label: '95% Margin of Error', value: `±${(margin95 * 100).toFixed(1)}%`, kind: 'memory', note: `CI width: ${(margin95 * 200).toFixed(1)}%` },
          { label: 'Resolution', value: sampleN >= 900 ? 'High (±1.5%)' : sampleN >= 300 ? 'Moderate (±3%)' : 'Coarse (±8%)', kind: 'compute' },
        ]} />
      }
      controls={
        s.step >= 2 ? (
          <>
            {s.step === 2 && (
              <Segmented
                label="Inspect evaluation slice"
                value={activeSlice}
                options={[
                  { value: 'overall', label: 'Overall Aggregate' },
                  { value: 'easy', label: 'Easy Slice (Short Queries)' },
                  { value: 'hard', label: 'Hard Slice (Multi-Step Logic)' },
                ]}
                onChange={(val) => setActiveSlice(val as any)}
              />
            )}
            {s.step >= 3 && (
              <Slider
                label="Evaluation dataset sample size (n)"
                value={sampleN}
                min={50}
                max={1500}
                onChange={setSampleN}
                format={(v) => `${v} samples`}
                kind="token"
              />
            )}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Visual explanation of component-level pipeline decomposition, Simpson's paradox in slicing, and standard error sample size arithmetic.">
        {/* Step 0: Component-level decomposition */}
        {s.step === 0 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Component-Level vs End-to-End Evaluation</Label>

            {/* Stage 1 */}
            <rect x={24} y={60} width={130} height={140} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={89} y={82} tone={1} size={10.5} bold anchor="middle">1. Query Rewrite</Label>
            <text x={34} y={106} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Keyword extraction</text>
            <text x={34} y={124} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Query expansion</text>
            <Label x={89} y={160} tone={3} size={8.5} anchor="middle">Metric:</Label>
            <Label x={89} y={176} kind="memory" size={9} bold anchor="middle">Retrieval recall boost</Label>

            <Arrow x1={158} y1={130} x2={178} y2={130} />

            {/* Stage 2 */}
            <rect x={182} y={60} width={130} height={140} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={247} y={82} tone={1} size={10.5} bold anchor="middle">2. Vector Search</Label>
            <text x={192} y={106} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Embedding index</text>
            <text x={192} y={124} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Top-k chunk recall</text>
            <Label x={247} y={160} tone={3} size={8.5} anchor="middle">Metric:</Label>
            <Label x={247} y={176} kind="compute" size={9} bold anchor="middle">NDCG@5 / Recall@k</Label>

            <Arrow x1={316} y1={130} x2={336} y2={130} />

            {/* Stage 3 */}
            <rect x={340} y={60} width={130} height={140} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={405} y={82} tone={1} size={10.5} bold anchor="middle">3. Generation</Label>
            <text x={350} y={106} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Synthesis from context</text>
            <text x={350} y={124} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Hallucination check</text>
            <Label x={405} y={160} tone={3} size={8.5} anchor="middle">Metric:</Label>
            <Label x={405} y={176} kind="token" size={9} bold anchor="middle">Faithfulness score</Label>

            <Arrow x1={474} y1={130} x2={494} y2={130} />

            {/* Stage 4 */}
            <rect x={498} y={60} width={118} height={140} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={557} y={82} tone={1} size={10.5} bold anchor="middle">4. Guardrails</Label>
            <text x={508} y={106} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• PII redactor</text>
            <text x={508} y={124} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Safety filter</text>
            <Label x={557} y={160} tone={3} size={8.5} anchor="middle">Metric:</Label>
            <Label x={557} y={176} kind="error" size={9} bold anchor="middle">False rejection rate</Label>

            {/* Bottom takeaway */}
            <rect x={24} y={220} width={592} height={60} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={242} tone={1} size={11} bold>The Localization Rule:</Label>
            <text x={40} y={262} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">
              If your end-to-end answer is wrong, isolated metrics reveal whether the retriever failed to find the document or the LLM hallucinated despite having it.
            </text>
          </g>
        )}

        {/* Step 1: Rubrics with anchor examples */}
        {s.step === 1 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Rubrics with Explicit Anchor Examples (5-Point Scale)</Label>

            <rect x={24} y={54} width={592} height={220} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />

            {/* Anchor 5 */}
            <rect x={36} y={68} width={568} height={36} rx={3} fill="var(--token-dim)" stroke={COLOR.token} />
            <Label x={48} y={86} kind="token" size={10} bold>Score 5 (Exemplary):</Label>
            <text x={165} y={86} fill="var(--text)" fontSize={9.5} fontFamily="var(--font-sans)">
              100% grounded in retrieved text, concise, no conversational filler. "The server port is 8080."
            </text>

            {/* Anchor 3 */}
            <rect x={36} y={112} width={568} height={36} rx={3} fill="var(--compute-dim)" stroke={COLOR.compute} />
            <Label x={48} y={130} kind="compute" size={10} bold>Score 3 (Flawed):</Label>
            <text x={165} y={130} fill="var(--text)" fontSize={9.5} fontFamily="var(--font-sans)">
              Factually correct answer, but introduces 150 words of unsolicited advice and speculative claims.
            </text>

            {/* Anchor 1 */}
            <rect x={36} y={156} width={568} height={36} rx={3} fill="var(--error-dim)" stroke={COLOR.error} />
            <Label x={48} y={174} kind="error" size={10} bold>Score 1 (Failure):</Label>
            <text x={165} y={174} fill="var(--text)" fontSize={9.5} fontFamily="var(--font-sans)">
              Hallucinated port number ("Port 443"), contradicts context, or leaks sensitive system prompt instructions.
            </text>

            <Label x={40} y={224} tone={2} size={10}>
              Key Insight: Defining concrete anchor sentences for each score eliminates judge calibration drift across prompt versions.
            </Label>
          </g>
        )}

        {/* Step 2: Slicing & Simpson's Paradox */}
        {s.step === 2 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Simpson’s Paradox in Slicing: When Aggregates Deceive</Label>

            {/* Table */}
            <rect x={24} y={54} width={592} height={170} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />

            {/* Header row */}
            <Label x={40} y={78} tone={1} size={10.5} bold>Query Slice</Label>
            <Label x={220} y={78} kind="token" size={10.5} bold>Model 1 (Production Baseline)</Label>
            <Label x={430} y={78} kind="compute" size={10.5} bold>Model 2 (Candidate Prompt)</Label>

            <line x1={40} y1={88} x2={600} y2={88} stroke="var(--rule-2)" strokeWidth={1} />

            {/* Row 1: Easy Slice */}
            <Label x={40} y={112} tone={activeSlice === 'easy' ? 1 : 2} size={10} bold={activeSlice === 'easy'}>
              Slice A: Easy Short Queries
            </Label>
            <Label x={220} y={112} kind="token" size={10} bold>18 / 20 = 90.0% (Superior)</Label>
            <Label x={430} y={112} tone={3} size={10}>68 / 80 = 85.0%</Label>

            {/* Row 2: Hard Slice */}
            <Label x={40} y={142} tone={activeSlice === 'hard' ? 1 : 2} size={10} bold={activeSlice === 'hard'}>
              Slice B: Hard Multi-Step
            </Label>
            <Label x={220} y={142} kind="token" size={10} bold>40 / 80 = 50.0% (Superior)</Label>
            <Label x={430} y={142} tone={3} size={10}>9 / 20 = 45.0%</Label>

            <line x1={40} y1={154} x2={600} y2={154} stroke="var(--rule-2)" strokeWidth={1} />

            {/* Overall Aggregate Row */}
            <Label x={40} y={178} tone={1} size={10.5} bold>Overall Aggregate Score:</Label>
            <Label x={220} y={178} kind="error" size={11} bold>58 / 100 = 58.0%</Label>
            <Label x={430} y={178} kind="compute" size={11} bold>77 / 100 = 77.0% (Apparent Winner)</Label>

            {/* Explanation banner */}
            <rect x={24} y={236} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={256} kind="error" size={10} bold>The Paradox:</Label>
            <text x={120} y={256} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">
              Model 2 appears to win by +19%, but it loses on EVERY slice! Its overall score is higher only because 80% of its test cases were easy.
            </text>
          </g>
        )}

        {/* Step 3 & 4: Standard error and sample size */}
        {(s.step === 3 || s.step === 4) && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Sample Size and Standard Error: SE = sqrt(p(1 - p) / n)</Label>

            {/* Left box: Formula & math */}
            <rect x={24} y={54} width={340} height={160} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={36} y={76} tone={1} size={11} bold>The Binomial Standard Error:</Label>
            <text x={36} y={104} fill="var(--text)" fontSize={12} fontFamily="var(--font-mono)">
              SE = sqrt[ p(1 - p) / n ]
            </text>
            <text x={36} y={130} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">
              At baseline accuracy p = 0.80 (80%):
            </text>
            <text x={36} y={150} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-mono)">
              p(1 - p) = 0.80 * 0.20 = 0.160
            </text>
            <text x={36} y={174} fill={COLOR.token} fontSize={10.5} fontFamily="var(--font-mono)">
              • n = 100  → SE = sqrt(0.0016) = 0.0400 (4.0%)
            </text>
            <text x={36} y={194} fill={COLOR.token} fontSize={10.5} fontFamily="var(--font-mono)">
              • n = 1000 → SE = sqrt(0.00016) = 0.0126 (1.26%)
            </text>

            {/* Right box: The 10x rule */}
            <rect x={380} y={54} width={236} height={160} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={396} y={76} tone={1} size={11} bold>The Inverse Square Rule:</Label>
            <text x={396} y={102} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">To cut uncertainty in half (2x):</text>
            <text x={396} y={118} fill={COLOR.compute} fontSize={9.5} fontFamily="var(--font-mono)">Needs 2^2 = 4x samples.</text>

            <text x={396} y={146} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">To detect a 3x smaller delta:</text>
            <text x={396} y={162} fill={COLOR.token} fontSize={9.5} fontFamily="var(--font-mono)">Needs 3^2 = 9x to 10x samples.</text>

            <Span x={396} y={184} w={204} h={20} kind="token" label="Inverse Square Law" />

            {/* Bottom dynamic readout */}
            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={1} size={10.5} bold>
              Current n = {sampleN}: Standard Error = {(se * 100).toFixed(2)}%. A measured improvement of {(margin95 * 100).toFixed(1)}% is within random noise!
            </Label>
          </g>
        )}

        {/* Step 5: Offline vs Online loop */}
        {s.step === 5 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Continuous Evaluation: Offline Golden Gating to Online Telemetry</Label>

            {/* Left Box: Offline */}
            <rect x={24} y={54} width={285} height={170} rx={4} fill="var(--surface-2)" stroke={COLOR.memory} />
            <Label x={36} y={76} tone={1} size={11} bold>1. Offline Pre-Merge Evaluation</Label>
            <text x={36} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• 200-500 curated golden edge cases</text>
            <text x={36} y={120} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Automated CI/CD regression gating</text>
            <text x={36} y={140} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• High statistical rigor (pass@k, rubrics)</text>
            <Span x={36} y={185} w={180} h={20} kind="memory" label="Gating Pull Requests" />

            {/* Right Box: Online */}
            <rect x={331} y={54} width={285} height={170} rx={4} fill="var(--surface-2)" stroke={COLOR.token} />
            <Label x={343} y={76} tone={1} size={11} bold>2. Online Production Telemetry</Label>
            <text x={343} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• A/B testing canary splits (e.g. 10% traffic)</text>
            <text x={343} y={120} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Implicit signals: copy-to-clipboard, retries</text>
            <text x={343} y={140} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Real production distribution drift alerts</text>
            <Span x={343} y={185} w={180} h={20} kind="token" label="Detecting Live Drift" />

            {/* Bottom rule */}
            <rect x={24} y={236} width={592} height={44} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={320} y={262} anchor="middle" tone={1} size={10.5} bold>
              Offline evals prove your system did not regress; online telemetry proves users actually prefer it.
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
