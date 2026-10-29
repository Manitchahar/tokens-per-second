import { useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, COLOR } from '../lib/svg';

const STEPS = [
  'Open-ended outputs: infinite valid answers',
  'Exact string match collapses to 0%',
  'Goodhart’s Law and benchmark saturation',
  'Evaluation-Driven Development (EDD)',
  'The four evaluation criteria buckets',
  'The trade-off frontier: quality vs cost vs speed',
];

const CAPTIONS = [
  'Unlike classical classification where an image is a dog or a cat, an open-ended prompt has millions of valid phrasings, lengths, and valid interpretations.',
  'If you test open-ended generation with naive exact match or simple n-gram overlap, valid, brilliant outputs score 0% simply for using synonyms or different sentence order.',
  'When a public benchmark becomes the target, models overfit to its format and distribution. The score hits 95%+, but production failure rates barely budge.',
  'Evaluation-Driven Development (EDD) fixes this: define measurable golden test cases, rubrics, and acceptance thresholds before prompt engineering or fine-tuning.',
  'A complete evaluation spans four distinct buckets: Domain Capability, Generation Quality, Instruction Following, and Cost & Latency.',
  'Production systems never optimize one bucket in isolation. Pushing generation quality to maximum often increases latency and inference cost exponentially.',
];

export default function EvalCriteriaViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [selectedBucket, setSelectedBucket] = useState<'domain' | 'quality' | 'instruction' | 'ops'>('domain');
  const [qualitySlider, setQualitySlider] = useState(80);

  // Derived calculations for step 6 trade-off frontier
  const latencyMs = Math.round(15 + Math.pow(qualitySlider / 100, 3) * 65);
  const costPerMillion = (0.2 + Math.pow(qualitySlider / 100, 2.5) * 4.8).toFixed(2);
  const passRate = Math.min(99, Math.round(50 + (qualitySlider / 100) * 48));

  return (
    <Stage
      rootRef={ref}
      title="Why evaluating LLMs is hard: from open-ended outputs to the 4 criteria"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', 'error']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Active step', value: `${s.step + 1} of ${STEPS.length}`, kind: 'text' },
          { label: 'Domain accuracy', value: `${passRate}%`, kind: 'token', note: 'estimated test pass rate' },
          { label: 'TTFT / Step Latency', value: `${latencyMs} ms`, kind: latencyMs > 50 ? 'compute' : 'memory', note: 'per generated token' },
          { label: 'Serving cost', value: `$${costPerMillion} / 1M`, kind: Number(costPerMillion) > 2 ? 'error' : 'token', note: 'blended input/output' },
        ]} />
      }
      controls={
        s.step >= 4 ? (
          <>
            <Segmented
              label="Inspect criteria bucket"
              value={selectedBucket}
              options={[
                { value: 'domain', label: 'Domain capability' },
                { value: 'quality', label: 'Generation quality' },
                { value: 'instruction', label: 'Instruction following' },
                { value: 'ops', label: 'Cost & latency' },
              ]}
              onChange={(val) => setSelectedBucket(val as any)}
            />
            {s.step === 5 && (
              <Slider
                label="Target quality & reasoning depth"
                value={qualitySlider}
                min={20}
                max={100}
                onChange={setQualitySlider}
                format={(v) => `${v}% target`}
                kind="token"
              />
            )}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Visual explanation of why open-ended language model evaluation is difficult, benchmark saturation, and the four criteria buckets.">
        {/* Step 0: Open-ended outputs */}
        {s.step === 0 && (
          <g>
            <Label x={24} y={32} tone={1} size={13} bold>The Prompt: "Explain gradient descent in one sentence"</Label>
            <rect x={24} y={48} width={592} height={36} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <text x={36} y={71} fill="var(--text)" fontSize={12} fontFamily="var(--font-sans)">
              Goal: optimize parameters by moving opposite the gradient of the loss function.
            </text>

            <Label x={24} y={116} tone={2} size={11} bold>Candidate Model Outputs (All 100% Factually Correct):</Label>

            {/* Candidate A */}
            <rect x={24} y={130} width={592} height={36} rx={4} fill="var(--token-dim)" stroke={COLOR.token} strokeWidth={1} />
            <text x={36} y={148} fill={COLOR.token} fontSize={10} fontWeight={600} fontFamily="var(--font-mono)">Candidate 1 (Concise)</text>
            <text x={36} y={160} fill="var(--text)" fontSize={11} fontFamily="var(--font-sans)">
              Gradient descent iteratively adjusts parameters in the direction of steepest loss reduction.
            </text>

            {/* Candidate B */}
            <rect x={24} y={178} width={592} height={36} rx={4} fill="var(--token-dim)" stroke={COLOR.token} strokeWidth={1} />
            <text x={36} y={196} fill={COLOR.token} fontSize={10} fontWeight={600} fontFamily="var(--font-mono)">Candidate 2 (Intuitive)</text>
            <text x={36} y={208} fill="var(--text)" fontSize={11} fontFamily="var(--font-sans)">
              It is like walking downhill in thick fog by taking small steps along the steepest downward slope.
            </text>

            {/* Candidate C */}
            <rect x={24} y={226} width={592} height={36} rx={4} fill="var(--token-dim)" stroke={COLOR.token} strokeWidth={1} />
            <text x={36} y={244} fill={COLOR.token} fontSize={10} fontWeight={600} fontFamily="var(--font-mono)">Candidate 3 (Formal Math)</text>
            <text x={36} y={256} fill="var(--text)" fontSize={11} fontFamily="var(--font-sans)">
              Weights update according to theta := theta - alpha * nabla L(theta) until convergence.
            </text>

            <Label x={320} y={296} anchor="middle" tone={3} size={11}>
              Notice: identical semantics, zero string overlap. No regex or reference match can capture all three.
            </Label>
          </g>
        )}

        {/* Step 1: Naive string match fails */}
        {s.step === 1 && (
          <g>
            <Label x={24} y={32} tone={1} size={13} bold>Naive String Matching against Reference Ground Truth</Label>
            
            <rect x={24} y={50} width={280} height={100} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={36} y={72} tone={2} size={11} bold>Reference Answer in Dataset:</Label>
            <text x={36} y={96} fill="var(--text)" fontSize={11} fontFamily="var(--font-sans)">
              "Steps downhill toward minimal loss"
            </text>
            <text x={36} y={116} fill="var(--text-3)" fontSize={10} fontFamily="var(--font-mono)">
              Exact tokens: [Steps, downhill, ...]
            </text>

            <rect x={336} y={50} width={280} height={100} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={348} y={72} tone={2} size={11} bold>Model Prediction:</Label>
            <text x={348} y={96} fill="var(--text)" fontSize={11} fontFamily="var(--font-sans)">
              "Walks downwards along the steepest slope"
            </text>
            <text x={348} y={116} fill={COLOR.token} fontSize={10} fontFamily="var(--font-mono)">
              Human verdict: Perfect analogy (100%)
            </text>

            {/* Scorecard */}
            <rect x={24} y={170} width={592} height={120} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={196} tone={1} size={12} bold>Evaluation Metric Comparison:</Label>

            <Label x={40} y={224} tone={2} size={11} mono>Exact String Match:</Label>
            <Span x={190} y={212} w={140} h={18} kind="error" label="0% Match (False Neg)" />

            <Label x={40} y={250} tone={2} size={11} mono>ROUGE-1 (Unigram):</Label>
            <Span x={190} y={238} w={140} h={18} kind="error" label="0.00 Overlap" />

            <Label x={40} y={276} tone={2} size={11} mono>Semantic Judge / Human:</Label>
            <Span x={190} y={264} w={140} h={18} kind="token" label="100% Pass (True Pos)" />

            <Arrow x1={350} y1={221} x2={430} y2={221} kind="error" />
            <Label x={440} y={225} tone={3} size={10} kind="error">Brittleness rejects valid code & answers</Label>
          </g>
        )}

        {/* Step 2: Benchmark Saturation & Goodhart's law */}
        {s.step === 2 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Goodhart’s Law: "When a metric becomes the target, it ceases to be good"</Label>

            {/* Saturation Chart */}
            <rect x={24} y={50} width={360} height={230} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={74} tone={2} size={11} bold>Benchmark Trajectory vs Production Incidents</Label>

            {/* Axes */}
            <line x1={60} y1={240} x2={360} y2={240} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={60} y1={90} x2={60} y2={240} stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={40} y={95} tone={3} size={9} anchor="end">100%</Label>
            <Label x={40} y={165} tone={3} size={9} anchor="end">50%</Label>
            <Label x={40} y={240} tone={3} size={9} anchor="end">0%</Label>
            <Label x={60} y={256} tone={3} size={9}>2022</Label>
            <Label x={200} y={256} tone={3} size={9}>2024</Label>
            <Label x={340} y={256} tone={3} size={9}>2026</Label>

            {/* Saturated benchmark curve */}
            <path d="M 60 210 Q 150 180 200 120 T 360 98" fill="none" stroke={COLOR.token} strokeWidth={2.5} />
            <Label x={280} y={92} kind="token" size={10} bold>Public Benchmark (98%)</Label>

            {/* Real world failure rate */}
            <path d="M 60 170 Q 160 160 240 155 T 360 150" fill="none" stroke={COLOR.error} strokeWidth={2} strokeDasharray="4 3" />
            <Label x={240} y={170} kind="error" size={10} bold>Production Failure Rate (~25%)</Label>

            {/* Explanation box */}
            <rect x={400} y={50} width={216} height={230} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={412} y={74} tone={1} size={11} bold>Why Benchmarks Lie:</Label>
            <text x={412} y={100} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">
              1. Format Overfitting:
            </text>
            <text x={412} y={116} fill="var(--text-3)" fontSize={9.5} fontFamily="var(--font-sans)">
              Models learn multiple-choice trickery rather than true reasoning.
            </text>

            <text x={412} y={145} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">
              2. Distribution Drift:
            </text>
            <text x={412} y={161} fill="var(--text-3)" fontSize={9.5} fontFamily="var(--font-sans)">
              Synthetic homework problems do not match messy customer telemetry.
            </text>

            <text x={412} y={190} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">
              3. Data Contamination:
            </text>
            <text x={412} y={206} fill="var(--text-3)" fontSize={9.5} fontFamily="var(--font-sans)">
              Test sets leaked directly into the multi-trillion token pretraining pool.
            </text>

            <Span x={412} y={235} w={192} h={26} kind="error" label="Saturated = Unusable" />
          </g>
        )}

        {/* Step 3: Evaluation-Driven Development (EDD) */}
        {s.step === 3 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Evaluation-Driven Development: The Production Lifecycle</Label>

            {/* Flow boxes */}
            <rect x={24} y={70} width={120} height={120} rx={4} fill="var(--surface-2)" stroke={COLOR.memory} strokeWidth={1.5} />
            <Label x={84} y={95} tone={1} size={11} bold anchor="middle">1. Define Eval</Label>
            <text x={84} y={120} textAnchor="middle" fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">
              Golden dataset (50-200 curated examples)
            </text>
            <text x={84} y={150} textAnchor="middle" fill={COLOR.memory} fontSize={9.5} fontFamily="var(--font-mono)">
              Explicit rubrics
            </text>

            <Arrow x1={148} y1={130} x2={176} y2={130} kind="memory" />

            <rect x={180} y={70} width={120} height={120} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={240} y={95} tone={1} size={11} bold anchor="middle">2. Baseline Model</Label>
            <text x={240} y={120} textAnchor="middle" fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">
              Prompting, zero-shot, few-shot
            </text>
            <text x={240} y={150} textAnchor="middle" fill="var(--text-3)" fontSize={9.5} fontFamily="var(--font-mono)">
              Measure initial pass rate
            </text>

            <Arrow x1={304} y1={130} x2={332} y2={130} />

            <rect x={336} y={70} width={120} height={120} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={396} y={95} tone={1} size={11} bold anchor="middle">3. Iterate & Optimize</Label>
            <text x={396} y={120} textAnchor="middle" fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">
              Refine prompts, RAG, tool schemas
            </text>
            <text x={396} y={150} textAnchor="middle" fill={COLOR.compute} fontSize={9.5} fontFamily="var(--font-mono)">
              Catch regressions
            </text>

            <Arrow x1={460} y1={130} x2={488} y2={130} kind="token" />

            <rect x={492} y={70} width={124} height={120} rx={4} fill="var(--surface-2)" stroke={COLOR.token} strokeWidth={1.5} />
            <Label x={554} y={95} tone={1} size={11} bold anchor="middle">4. Gate Deployment</Label>
            <text x={554} y={120} textAnchor="middle" fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">
              CI/CD test suites
            </text>
            <text x={554} y={150} textAnchor="middle" fill={COLOR.token} fontSize={9.5} fontFamily="var(--font-mono)">
              {'Pass > 95% threshold'}
            </text>

            {/* Bottom takeaway */}
            <rect x={24} y={220} width={592} height={60} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={244} tone={1} size={11} bold>The Core Rule of AI Engineering:</Label>
            <text x={40} y={264} fill="var(--text-2)" fontSize={10.5} fontFamily="var(--font-sans)">
              "Never write or tweak a prompt without an automated harness to prove you did not break five other edge cases."
            </text>
          </g>
        )}

        {/* Step 4 & 5: The 4 Criteria Buckets */}
        {(s.step === 4 || s.step === 5) && (
          <g>
            <Label x={24} y={26} tone={1} size={13} bold>The Four Buckets of Evaluation Criteria</Label>

            {/* 4 Quadrants/Columns */}
            {/* 1. Domain Capability */}
            <g
              transform="translate(24, 46)"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelectedBucket('domain')}
            >
              <rect
                x={0}
                y={0}
                width={140}
                height={220}
                rx={4}
                fill={selectedBucket === 'domain' ? 'var(--memory-dim)' : 'var(--surface-2)'}
                stroke={selectedBucket === 'domain' ? COLOR.memory : 'var(--rule-2)'}
                strokeWidth={selectedBucket === 'domain' ? 2 : 1}
              />
              <Label x={70} y={24} tone={1} size={10.5} bold anchor="middle">1. Domain Capability</Label>
              <text x={12} y={50} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Subject expertise</text>
              <text x={12} y={70} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Mathematical reasoning</text>
              <text x={12} y={90} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Code execution tests</text>
              <text x={12} y={110} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Information extraction</text>
              <Label x={70} y={160} tone={3} size={9} anchor="middle">How it is scored:</Label>
              <Label x={70} y={178} kind="memory" size={9.5} bold anchor="middle">Unit tests / pass@k</Label>
            </g>

            {/* 2. Generation Quality */}
            <g
              transform="translate(174, 46)"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelectedBucket('quality')}
            >
              <rect
                x={0}
                y={0}
                width={140}
                height={220}
                rx={4}
                fill={selectedBucket === 'quality' ? 'var(--token-dim)' : 'var(--surface-2)'}
                stroke={selectedBucket === 'quality' ? COLOR.token : 'var(--rule-2)'}
                strokeWidth={selectedBucket === 'quality' ? 2 : 1}
              />
              <Label x={70} y={24} tone={1} size={10.5} bold anchor="middle">2. Generation Quality</Label>
              <text x={12} y={50} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Factual consistency</text>
              <text x={12} y={70} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Hallucination rate</text>
              <text x={12} y={90} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Safety & toxicity</text>
              <text x={12} y={110} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Tone & clarity</text>
              <Label x={70} y={160} tone={3} size={9} anchor="middle">How it is scored:</Label>
              <Label x={70} y={178} kind="token" size={9.5} bold anchor="middle">LLM Judge / Rubrics</Label>
            </g>

            {/* 3. Instruction Following */}
            <g
              transform="translate(324, 46)"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelectedBucket('instruction')}
            >
              <rect
                x={0}
                y={0}
                width={140}
                height={220}
                rx={4}
                fill={selectedBucket === 'instruction' ? 'var(--compute-dim)' : 'var(--surface-2)'}
                stroke={selectedBucket === 'instruction' ? COLOR.compute : 'var(--rule-2)'}
                strokeWidth={selectedBucket === 'instruction' ? 2 : 1}
              />
              <Label x={70} y={24} tone={1} size={10.5} bold anchor="middle">3. Instruction Follow</Label>
              <text x={12} y={50} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• JSON / XML schema</text>
              <text x={12} y={70} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Length constraints</text>
              <text x={12} y={90} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Negative constraints</text>
              <text x={12} y={110} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Tool call argument spec</text>
              <Label x={70} y={160} tone={3} size={9} anchor="middle">How it is scored:</Label>
              <Label x={70} y={178} kind="compute" size={9.5} bold anchor="middle">Deterministic Schema</Label>
            </g>

            {/* 4. Cost & Latency */}
            <g
              transform="translate(474, 46)"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelectedBucket('ops')}
            >
              <rect
                x={0}
                y={0}
                width={142}
                height={220}
                rx={4}
                fill={selectedBucket === 'ops' ? 'var(--error-dim)' : 'var(--surface-2)'}
                stroke={selectedBucket === 'ops' ? COLOR.error : 'var(--rule-2)'}
                strokeWidth={selectedBucket === 'ops' ? 2 : 1}
              />
              <Label x={71} y={24} tone={1} size={10.5} bold anchor="middle">4. Cost & Latency</Label>
              <text x={12} y={50} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• TTFT (Time to 1st tok)</text>
              <text x={12} y={70} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• TPOT (Decode speed)</text>
              <text x={12} y={90} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• $ per 1M tokens</text>
              <text x={12} y={110} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• GPU Memory footprint</text>
              <Label x={71} y={160} tone={3} size={9} anchor="middle">How it is scored:</Label>
              <Label x={71} y={178} kind="error" size={9.5} bold anchor="middle">Telemetry / Counters</Label>
            </g>

            {/* Footer notice */}
            <Label x={320} y={290} anchor="middle" tone={3} size={10.5}>
              Click any bucket to inspect. A model scoring high on capability but failing schema is un-shippable.
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
