import { useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, COLOR } from '../lib/svg';

const STEPS = [
  'The canonical public benchmarks: MMLU, GSM8K, HumanEval',
  'Benchmark saturation: the ceiling effect',
  'Correlation between benchmarks: redundant signals',
  'Contamination mechanism 1: 8-gram and 13-gram overlap',
  'Contamination mechanism 2: perplexity drop on canonical questions',
  'The coarse screening rule: filter, never select',
];

const CAPTIONS = [
  'Three benchmarks dominated early LLM evaluation: MMLU (57 academic subjects, MCQ), GSM8K (multi-step grade school arithmetic), and HumanEval (164 Python challenges).',
  'Over four years, frontier model scores rose from ~40% to over 95%. When benchmarks saturate near 100%, their dynamic range collapses, hiding subtle differences.',
  'MMLU, GSM8K, and ARC exhibit high Pearson correlation (>0.85). Evaluating an engine across twenty generic benchmarks wastes compute without adding new signal.',
  'Data contamination occurs when pretraining web crawls ingest test questions. We detect contamination by scanning training corpora for 8-gram and 13-gram exact matches.',
  'Memorization test: a contaminated model displays unnaturally low perplexity (PPL ~1.2) on canonical benchmark text, but perplexity spikes to 8.4 on slightly rephrased variants.',
  'Use public benchmarks strictly as a coarse screening stage to discard underpowered models. Never make final production architectural choices based on public leaderboards.',
];

export default function BenchmarkContaminationVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [testVariant, setTestVariant] = useState<'canonical' | 'perturbed'>('canonical');
  const [ngramLen, setNgramLen] = useState(8);

  // Derived calculations for perplexity test
  const pplValue = testVariant === 'canonical' ? 1.35 : 7.82;
  const isContaminated = testVariant === 'canonical';

  return (
    <Stage
      rootRef={ref}
      title="Public Benchmarks: Saturation, Contamination, and Coarse Screening"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'error', 'compute', 'memory']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Selected view', value: s.step <= 1 ? 'Saturation' : s.step <= 3 ? 'N-gram Overlap' : 'Perplexity Test', kind: 'text' },
          { label: 'Evaluation question', value: testVariant === 'canonical' ? 'Original GSM8K Question' : 'Perturbed (Variables Renamed)', kind: 'memory' },
          { label: 'Model Perplexity (PPL)', value: `${pplValue}`, kind: pplValue < 2.0 ? 'error' : 'token', note: pplValue < 2.0 ? 'memorized test leakage' : 'normal generalization' },
          { label: 'N-gram scan window', value: `${ngramLen}-grams`, kind: 'compute', note: 'Common Crawl deduplication standard' },
        ]} />
      }
      controls={
        s.step >= 3 ? (
          <>
            {s.step === 3 && (
              <Slider
                label="N-gram collision threshold"
                value={ngramLen}
                min={6}
                max={15}
                onChange={setNgramLen}
                format={(v) => `${v}-grams`}
                kind="compute"
              />
            )}
            {s.step >= 4 && (
              <Segmented
                label="Benchmark question formulation"
                value={testVariant}
                options={[
                  { value: 'canonical', label: 'Canonical Question (Web Verbatim)' },
                  { value: 'perturbed', label: 'Perturbed Variant (Renamed Vars)' },
                ]}
                onChange={(val) => setTestVariant(val as any)}
              />
            )}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Visual explanation of public benchmark saturation, n-gram contamination detection, perplexity drop memorization tests, and coarse screening funnels.">
        {/* Step 0: The Big Three Benchmarks */}
        {s.step === 0 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>The Canonical Benchmark Triumvirate</Label>

            {/* Benchmark 1: MMLU */}
            <rect x={24} y={54} width={185} height={210} rx={4} fill="var(--surface-2)" stroke={COLOR.memory} strokeWidth={1.5} />
            <Label x={116} y={76} tone={1} size={11} bold anchor="middle">1. MMLU (2020)</Label>
            <text x={36} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• 57 academic subjects</text>
            <text x={36} y={120} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Multiple-choice (A, B, C, D)</text>
            <text x={36} y={140} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Tests world knowledge breadth</text>
            <text x={36} y={160} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• 5-shot prompt standard</text>
            <Span x={36} y={230} w={161} h={20} kind="memory" label="Knowledge Breadth" />

            {/* Benchmark 2: GSM8K */}
            <rect x={227} y={54} width={185} height={210} rx={4} fill="var(--surface-2)" stroke={COLOR.compute} strokeWidth={1.5} />
            <Label x={319} y={76} tone={1} size={11} bold anchor="middle">2. GSM8K (2021)</Label>
            <text x={239} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• 8,500 grade school word math</text>
            <text x={239} y={120} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Multi-step arithmetic reasoning</text>
            <text x={239} y={140} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Chain-of-thought (CoT)</text>
            <text x={239} y={160} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Exact numerical answer check</text>
            <Span x={239} y={230} w={161} h={20} kind="compute" label="Multi-Step Reasoning" />

            {/* Benchmark 3: HumanEval */}
            <rect x={431} y={54} width={185} height={210} rx={4} fill="var(--surface-2)" stroke={COLOR.token} strokeWidth={1.5} />
            <Label x={523} y={76} tone={1} size={11} bold anchor="middle">3. HumanEval (2021)</Label>
            <text x={443} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• 164 hand-written Python puzzles</text>
            <text x={443} y={120} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Unit test functional correctness</text>
            <text x={443} y={140} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• pass@1 and pass@k metrics</text>
            <text x={443} y={160} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Eliminates regex fragility</text>
            <Span x={443} y={230} w={161} h={20} kind="token" label="Code Logic & Execution" />
          </g>
        )}

        {/* Step 1: Benchmark Saturation */}
        {s.step === 1 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Benchmark Saturation: The Ceiling Collapse (2020 - 2026)</Label>

            {/* Coordinate chart */}
            <rect x={24} y={50} width={420} height={220} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <line x1={60} y1={230} x2={420} y2={230} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={60} y1={80} x2={60} y2={230} stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={45} y={85} tone={3} size={9} anchor="end">100%</Label>
            <Label x={45} y={155} tone={3} size={9} anchor="end">50%</Label>
            <Label x={45} y={230} tone={3} size={9} anchor="end">0%</Label>

            <Label x={60} y={246} tone={3} size={8.5}>2021</Label>
            <Label x={180} y={246} tone={3} size={8.5}>2023</Label>
            <Label x={300} y={246} tone={3} size={8.5}>2025</Label>
            <Label x={410} y={246} tone={3} size={8.5}>2026</Label>

            {/* MMLU saturation curve */}
            <path d="M 60 190 Q 180 140 280 95 T 410 88" fill="none" stroke={COLOR.memory} strokeWidth={2.5} />
            <Label x={330} y={82} kind="memory" size={9.5} bold>MMLU (92%)</Label>

            {/* GSM8K saturation curve */}
            <path d="M 60 215 Q 160 180 260 98 T 410 86" fill="none" stroke={COLOR.compute} strokeWidth={2.5} />
            <Label x={330} y={112} kind="compute" size={9.5} bold>GSM8K (96%)</Label>

            {/* Ceiling zone */}
            <rect x={60} y={80} width={360} height={25} fill="var(--error-dim)" opacity={0.4} />
            <Label x={240} y={96} kind="error" size={9} anchor="middle" bold>SATURATION CEILING (No dynamic range left)</Label>

            {/* Right card */}
            <rect x={460} y={50} width={156} height={220} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={472} y={74} tone={1} size={11} bold>Consequences:</Label>
            <text x={472} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Score compression:</text>
            <text x={472} y={116} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">All top 10 models score between 92% and 96%.</text>
            <text x={472} y={145} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Noise domination:</text>
            <text x={472} y={161} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">A single ambiguous question flips rank order.</text>
            <text x={472} y={190} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Shift to Hard Evals:</text>
            <text x={472} y={206} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">Moved to MMLU-Pro & SWE-bench.</text>
          </g>
        )}

        {/* Step 2: Correlation */}
        {s.step === 2 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>High Correlation Across General Benchmarks</Label>

            <rect x={24} y={54} width={592} height={160} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={78} tone={1} size={11.5} bold>Empirical Correlation Matrix (Pearson r across 50 models):</Label>

            {/* Matrix rows */}
            <Label x={40} y={108} tone={2} size={10} mono>MMLU vs GSM8K:</Label>
            <Span x={220} y={96} w={240} h={18} kind="compute" label="r = 0.89 (Extremely High)" />

            <Label x={40} y={138} tone={2} size={10} mono>MMLU vs ARC-Challenge:</Label>
            <Span x={220} y={126} w={255} h={18} kind="compute" label="r = 0.93 (Nearly Redundant)" />

            <Label x={40} y={168} tone={2} size={10} mono>MMLU vs Proprietary JSON Eval:</Label>
            <Span x={220} y={156} w={70} h={18} kind="error" label="r = 0.22" />

            {/* Takeaway */}
            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={2} size={10.5}>
              Diminishing returns: adding 10 more academic benchmarks tells you nothing about whether a model will extract your hospital invoices correctly.
            </Label>
          </g>
        )}

        {/* Step 3: Contamination Mode 1 (N-grams) */}
        {s.step === 3 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Contamination Detection 1: N-Gram Overlap Scanning</Label>

            <rect x={24} y={54} width={592} height={150} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={76} tone={1} size={11} bold>Pretraining Web Dump Token Stream (Common Crawl):</Label>
            <text x={40} y={100} fill="var(--text-3)" fontSize={9.5} fontFamily="var(--font-mono)">
              "...and on page 42 we find the solution to the riddle: Janet has 16 eggs. She bakes 3 cakes..."
            </text>

            <Label x={40} y={130} tone={1} size={11} bold>GSM8K Test Set Question #142:</Label>
            <text x={40} y={150} fill="var(--text)" fontSize={9.5} fontFamily="var(--font-mono)">
              "Janet has 16 eggs. She bakes 3 cakes using 4 eggs each. How many eggs are left?"
            </text>

            {/* Match Span */}
            <Span x={40} y={170} w={380} h={22} kind="error" label={`Exact Collision: ${ngramLen}-gram Match Detected!`} />

            <rect x={24} y={220} width={592} height={60} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={242} tone={1} size={11} bold>The 13-Gram Rule:</Label>
            <text x={40} y={262} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">
              In modern deduplication (GPT-4 / LLaMA tech reports), any continuous 13-token match between pretraining data and test sets flags immediate contamination.
            </text>
          </g>
        )}

        {/* Step 4: Contamination Mode 2 (Perplexity Drop) */}
        {s.step === 4 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Contamination Detection 2: The Perplexity Perturbation Test</Label>

            <rect x={24} y={54} width={285} height={160} rx={4} fill="var(--surface-2)" stroke={testVariant === 'canonical' ? COLOR.error : 'var(--rule-2)'} />
            <Label x={36} y={76} tone={1} size={11} bold>Canonical Benchmark Text:</Label>
            <text x={36} y={98} fill="var(--text)" fontSize={9} fontFamily="var(--font-mono)">
              "A car travels from Town A to Town B at 60 mph..."
            </text>
            <text x={36} y={130} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">
              Identical to web crawl test split.
            </text>
            <Label x={36} y={160} kind="error" size={10} bold>Perplexity: 1.35 (Unnaturally low loss)</Label>
            <Span x={36} y={176} w={180} h={20} kind="error" label="Rote Memorization Detected" />

            <rect x={331} y={54} width={285} height={160} rx={4} fill="var(--surface-2)" stroke={testVariant === 'perturbed' ? COLOR.token : 'var(--rule-2)'} />
            <Label x={343} y={76} tone={1} size={11} bold>Perturbed Variant (Same Logic):</Label>
            <text x={343} y={98} fill="var(--text)" fontSize={9} fontFamily="var(--font-mono)">
              "A truck drives from City X to City Y at 60 mph..."
            </text>
            <text x={343} y={130} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">
              Names and places swapped, math identical.
            </text>
            <Label x={343} y={160} kind="token" size={10} bold>Perplexity: 7.82 (Normal Language Loss)</Label>
            <Span x={343} y={176} w={180} h={20} kind="token" label="True Reasoning Capability" />

            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={2} size={10.5}>
              {'If model accuracy drops by > 15% when entity names are swapped, the model memorized the test set rather than reasoning.'}
            </Label>
          </g>
        )}

        {/* Step 5: Coarse Screening Rule */}
        {s.step === 5 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>The Production Funnel: Coarse Screening vs Selection</Label>

            {/* Funnel Stage 1 */}
            <rect x={24} y={54} width={592} height={50} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={74} tone={1} size={11} bold>Stage 1: Public Benchmark Screening (Coarse Filter)</Label>
            <text x={40} y={92} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">
              {'Rule: Discard models scoring < 70% on MMLU / HumanEval. (Eliminates models lacking baseline literacy).'}
            </text>

            {/* Funnel Stage 2 */}
            <rect x={64} y={114} width={512} height={50} rx={4} fill="var(--surface-2)" stroke={COLOR.compute} />
            <Label x={80} y={134} kind="compute" size={11} bold>Stage 2: Your Proprietary Golden Dataset (Custom Eval)</Label>
            <text x={80} y={152} fill="var(--text)" fontSize={9.5} fontFamily="var(--font-sans)">
              Test on 100-300 internal prompts: domain schemas, business edge cases, exact tone, safety constraints.
            </text>

            {/* Funnel Stage 3 */}
            <rect x={104} y={174} width={432} height={50} rx={4} fill="var(--surface-2)" stroke={COLOR.token} />
            <Label x={120} y={194} kind="token" size={11} bold>Stage 3: Latency & Cost SLO Feasibility</Label>
            <text x={120} y={212} fill="var(--text)" fontSize={9.5} fontFamily="var(--font-sans)">
              Measure TTFT, TPOT, memory footprint, and serving economics on real target hardware.
            </text>

            {/* Bottom rule */}
            <rect x={24} y={236} width={592} height={44} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={320} y={262} anchor="middle" tone={1} size={10.5} bold>
              Public benchmarks qualify candidates for the interview; your custom eval hires them.
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
