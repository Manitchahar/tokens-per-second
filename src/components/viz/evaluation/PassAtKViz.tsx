import { useRef, useState } from 'react';
import { Stage, Slider, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, COLOR } from '../lib/svg';

const STEPS = [
  'Functional correctness: does the code run?',
  'The high variance of naive sampling',
  'The unbiased hypergeometric estimator',
  'Worked arithmetic: n=10, c=3',
  'Interactive pass@k sensitivity curve',
  'The spectrum of exact evaluation metrics',
];

const CAPTIONS = [
  'For coding and mathematical tasks, we evaluate functional correctness: execute the generated code in a sandbox against unit tests instead of matching text.',
  'Generating k samples and checking if at least one passes has huge variance. If k=1, you miss that the model could solve it in 3 out of 10 tries.',
  'Chen et al. (2021) derived the unbiased pass@k estimator: generate n samples (n >= k), count c passing tests, and compute the exact combinatorial expectation.',
  'For n=10 and c=3: pass@1 = 0.30 (30%), while pass@5 = 1 - (7 choose 5)/(10 choose 5) = 1 - 21/252 = 0.917 (91.7%).',
  'Explore how pass@k scales with temperature and sample budget. As k grows, the chance of finding at least one correct solution climbs rapidly.',
  'From Exact Match to BLEU, ROUGE, and BERTScore: each metric trades off strictness against semantic flexibility.',
];

function comb(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  let c = 1;
  for (let i = 1; i <= k; i++) {
    c = (c * (n - (k - i))) / i;
  }
  return c;
}

function calcPassAtK(n: number, c: number, k: number): number {
  if (n - c < k) return 1.0;
  return 1.0 - comb(n - c, k) / comb(n, k);
}

export default function PassAtKViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [nVal, setNVal] = useState(10);
  const [cVal, setCVal] = useState(3);
  const [selectedK, setSelectedK] = useState(5);
  const [activeMetric, setActiveMetric] = useState<'em' | 'passk' | 'ngram' | 'bert'>('passk');

  const p1 = calcPassAtK(nVal, cVal, 1);
  const pk = calcPassAtK(nVal, cVal, Math.min(selectedK, nVal));

  return (
    <Stage
      rootRef={ref}
      title="Exact Evaluation: pass@k, functional correctness, and similarity"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'error', 'memory', 'compute']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Samples (n)', value: `${nVal}`, kind: 'text' },
          { label: 'Passing (c)', value: `${cVal}`, kind: 'token', note: `${Math.round((cVal / nVal) * 100)}% raw success` },
          { label: 'pass@1', value: `${(p1 * 100).toFixed(1)}%`, kind: 'memory', note: 'single-sample expectation' },
          { label: `pass@${selectedK}`, value: `${(pk * 100).toFixed(1)}%`, kind: pk > 0.8 ? 'token' : 'compute', note: `prob at least 1 of ${selectedK} passes` },
        ]} />
      }
      controls={
        s.step >= 3 ? (
          <>
            <Slider
              label="Correct samples (c)"
              value={cVal}
              min={0}
              max={nVal}
              onChange={(v) => setCVal(Math.min(v, nVal))}
              format={(v) => `${v} of ${nVal} pass`}
              kind="token"
            />
            {s.step === 4 && (
              <Segmented
                label="Target k for pass@k"
                value={selectedK}
                options={[
                  { value: 1, label: 'k=1' },
                  { value: 2, label: 'k=2' },
                  { value: 5, label: 'k=5' },
                  { value: 10, label: 'k=10' },
                ]}
                onChange={(val) => setSelectedK(Number(val))}
              />
            )}
            {s.step === 5 && (
              <Segmented
                label="Metric comparison"
                value={activeMetric}
                options={[
                  { value: 'em', label: 'Exact Match' },
                  { value: 'passk', label: 'pass@k' },
                  { value: 'ngram', label: 'ROUGE/BLEU' },
                  { value: 'bert', label: 'BERTScore' },
                ]}
                onChange={(val) => setActiveMetric(val as any)}
              />
            )}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Visual explanation of pass@k unbiased estimator, combinatorial arithmetic, and exact evaluation metric comparison.">
        {/* Step 0: Functional correctness */}
        {s.step === 0 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Unit Testing vs Text Matching for Code & Math</Label>
            
            {/* Code Box */}
            <rect x={24} y={50} width={340} height={150} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={36} y={72} tone={2} size={11} bold>Prompt & Generated Function:</Label>
            <text x={36} y={96} fill="var(--text)" fontSize={10.5} fontFamily="var(--font-mono)">
              {'def reverse_words(s: str) -> str:'}
            </text>
            <text x={56} y={116} fill={COLOR.token} fontSize={10.5} fontFamily="var(--font-mono)">
              return " ".join(s.split()[::-1])
            </text>
            <text x={36} y={150} fill="var(--text-3)" fontSize={9.5} fontFamily="var(--font-sans)">
              Reference had: " ".join(reversed(s.split()))
            </text>
            <Label x={36} y={174} tone={3} size={10}>Text match says: MISMATCH (0%). But is the code correct?</Label>

            {/* Test Suite Box */}
            <rect x={380} y={50} width={236} height={150} rx={4} fill="var(--surface-1)" stroke={COLOR.token} strokeWidth={1.5} />
            <Label x={396} y={72} tone={1} size={11} bold>Sandboxed Test Runner:</Label>
            <text x={396} y={96} fill={COLOR.token} fontSize={10} fontFamily="var(--font-mono)">✓ assert f("hello world") == "world hello"</text>
            <text x={396} y={120} fill={COLOR.token} fontSize={10} fontFamily="var(--font-mono)">✓ assert f("a b c") == "c b a"</text>
            <text x={396} y={144} fill={COLOR.token} fontSize={10} fontFamily="var(--font-mono)">✓ assert f("") == ""</text>
            <text x={396} y={168} fill={COLOR.token} fontSize={10} fontFamily="var(--font-mono)">✓ assert f("  hi  ") == "hi"</text>

            <rect x={24} y={220} width={592} height={60} rx={4} fill="var(--token-dim)" stroke={COLOR.token} />
            <Label x={40} y={244} tone={1} size={11} bold kind="token">Verdict: 100% Functional Correctness</Label>
            <text x={40} y={264} fill="var(--text)" fontSize={10.5} fontFamily="var(--font-sans)">
              Code and reasoning tasks must be executed, never string-matched. If all unit tests pass, the output is correct.
            </text>
          </g>
        )}

        {/* Step 1: Naive pass@k variance */}
        {s.step === 1 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>The Flaw in Naive pass@k: High Variance</Label>

            <rect x={24} y={50} width={592} height={100} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={72} tone={2} size={11} bold>Naive Method: Generate k=1 sample per problem</Label>
            <text x={40} y={94} fill="var(--text-2)" fontSize={10.5} fontFamily="var(--font-sans)">
              Run 1: Sample fails test suite → Score = 0%
            </text>
            <text x={40} y={114} fill="var(--text-2)" fontSize={10.5} fontFamily="var(--font-sans)">
              Run 2: Sample happens to pass → Score = 100%
            </text>
            <Label x={40} y={136} kind="error" size={10} bold>
              Huge variance! High temperature causes run-to-run scores to swing wildly.
            </Label>

            <rect x={24} y={170} width={592} height={100} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={192} tone={2} size={11} bold>Naive Multi-Sample: Generate k samples, check if ANY pass</Label>
            <text x={40} y={214} fill="var(--text-2)" fontSize={10.5} fontFamily="var(--font-sans)">
              Problem: If you want pass@10, generating 10 samples for each problem and taking the binary max is noisy.
            </text>
            <text x={40} y={234} fill="var(--text-2)" fontSize={10.5} fontFamily="var(--font-sans)">
              To evaluate 164 problems, you waste huge sample budgets without calculating the true expected probability.
            </text>
            <Span x={40} y={246} w={500} h={18} kind="error" label="High Variance & Inefficient Sampling" />
          </g>
        )}

        {/* Step 2: Unbiased Estimator Formula */}
        {s.step === 2 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>The Chen et al. (2021) Unbiased pass@k Estimator</Label>

            <rect x={24} y={50} width={592} height={110} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={74} tone={1} size={12} bold>The Hypergeometric Derivation:</Label>
            <text x={40} y={100} fill="var(--text)" fontSize={12} fontFamily="var(--font-mono)">
              pass@k = E [ 1 - C(n - c, k) / C(n, k) ]
            </text>
            <text x={40} y={124} fill="var(--text-2)" fontSize={10.5} fontFamily="var(--font-sans)">
              Where: n = total generated samples (e.g. 10 or 100), c = number of samples that pass unit tests.
            </text>
            <text x={40} y={142} fill="var(--text-3)" fontSize={10} fontFamily="var(--font-sans)">
              C(n-c, k) is the number of ways to pick k samples that ALL fail. Subtracting from 1 gives at least one pass.
            </text>

            {/* Grid of n samples */}
            <rect x={24} y={180} width={592} height={100} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={202} tone={2} size={11} bold>Sampling Pool (n=10 total, c=3 passing, n-c=7 failing):</Label>

            {/* 10 sample chips */}
            {Array.from({ length: 10 }, (_, i) => {
              const passes = i < 3;
              return (
                <g key={i} transform={`translate(${40 + i * 56}, 218)`}>
                  <rect
                    width={48}
                    height={36}
                    rx={4}
                    fill={passes ? 'var(--token-dim)' : 'var(--error-dim)'}
                    stroke={passes ? COLOR.token : COLOR.error}
                    strokeWidth={1}
                  />
                  <Label x={24} y={18} tone={1} size={9} anchor="middle" bold>
                    #{i + 1}
                  </Label>
                  <Label x={24} y={29} kind={passes ? 'token' : 'error'} size={8.5} anchor="middle">
                    {passes ? 'PASS' : 'FAIL'}
                  </Label>
                </g>
              );
            })}
          </g>
        )}

        {/* Step 3: Worked Arithmetic */}
        {s.step === 3 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Worked Numeric Example: n = 10 samples, c = 3 correct</Label>

            {/* Box 1: k=1 */}
            <rect x={24} y={54} width={285} height={160} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={36} y={76} tone={1} size={11.5} bold>Case 1: Single sample (k = 1)</Label>
            <text x={36} y={102} fill="var(--text)" fontSize={11} fontFamily="var(--font-mono)">
              C(10 - 3, 1) = C(7, 1) = 7
            </text>
            <text x={36} y={124} fill="var(--text)" fontSize={11} fontFamily="var(--font-mono)">
              C(10, 1) = 10
            </text>
            <text x={36} y={150} fill="var(--text)" fontSize={11.5} fontFamily="var(--font-mono)">
              pass@1 = 1 - 7/10 = 0.30
            </text>
            <Span x={36} y={166} w={120} h={20} kind="memory" label="30.0% Success" />

            {/* Box 2: k=5 */}
            <rect x={331} y={54} width={285} height={160} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={343} y={76} tone={1} size={11.5} bold>Case 2: Five attempts (k = 5)</Label>
            <text x={343} y={102} fill="var(--text)" fontSize={11} fontFamily="var(--font-mono)">
              C(7, 5) = 21
            </text>
            <text x={343} y={124} fill="var(--text)" fontSize={11} fontFamily="var(--font-mono)">
              C(10, 5) = 252
            </text>
            <text x={343} y={150} fill="var(--text)" fontSize={11.5} fontFamily="var(--font-mono)">
              pass@5 = 1 - 21/252 = 0.9167
            </text>
            <Span x={343} y={166} w={130} h={20} kind="token" label="91.7% Success" />

            {/* Bottom summary */}
            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={2} size={10.5}>
              Notice how the probability leaps from 30.0% to 91.7%: with 5 independent chances, the probability of failing all 5 is only 21/252 = 8.3%.
            </Label>
          </g>
        )}

        {/* Step 4: Interactive Sensitivity Curve */}
        {s.step === 4 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>pass@k Curve (n={nVal}, c={cVal})</Label>

            {/* Coordinate chart */}
            <rect x={24} y={50} width={420} height={220} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            
            {/* Grid lines */}
            <line x1={60} y1={230} x2={420} y2={230} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={60} y1={80} x2={60} y2={230} stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={45} y={85} tone={3} size={9} anchor="end">1.0</Label>
            <Label x={45} y={155} tone={3} size={9} anchor="end">0.5</Label>
            <Label x={45} y={230} tone={3} size={9} anchor="end">0.0</Label>

            {/* Curve points */}
            {Array.from({ length: Math.min(10, nVal) }, (_, idx) => {
              const k = idx + 1;
              const prob = calcPassAtK(nVal, cVal, k);
              const xPos = 60 + (k / 10) * 340;
              const yPos = 230 - prob * 150;
              return (
                <g key={k}>
                  <circle cx={xPos} cy={yPos} r={4} fill={k === selectedK ? COLOR.token : COLOR.compute} />
                  <Label x={xPos} y={246} tone={3} size={9} anchor="middle">k={k}</Label>
                  {k === selectedK && (
                    <Label x={xPos} y={yPos - 10} kind="token" size={10} bold anchor="middle">
                      {(prob * 100).toFixed(1)}%
                    </Label>
                  )}
                </g>
              );
            })}

            {/* Readout side-panel */}
            <rect x={460} y={50} width={156} height={220} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={472} y={74} tone={1} size={11} bold>Curve Insights:</Label>
            <text x={472} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">{'• Temperature > 0:'}</text>
            <text x={472} y={116} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">Generates diverse code attempts.</text>
            <text x={472} y={145} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Cost scaling:</text>
            <text x={472} y={161} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">pass@k costs k forward passes.</text>
            <text x={472} y={190} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Verification:</text>
            <text x={472} y={206} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">Needs an oracle/test runner to filter.</text>
          </g>
        )}

        {/* Step 5: Exact Metric Spectrum */}
        {s.step === 5 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Spectrum of Exact & Similarity Evaluation Metrics</Label>

            {/* Table / Cards */}
            <rect x={24} y={54} width={592} height={220} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />

            {/* Column 1: EM */}
            <g transform="translate(36, 68)">
              <rect width={130} height={180} rx={4} fill={activeMetric === 'em' ? 'var(--error-dim)' : 'var(--surface-1)'} stroke={activeMetric === 'em' ? COLOR.error : 'var(--rule-2)'} strokeWidth={activeMetric === 'em' ? 1.5 : 1} />
              <Label x={65} y={24} tone={1} size={10} bold anchor="middle">Exact Match (EM)</Label>
              <text x={10} y={50} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• String equality</text>
              <text x={10} y={70} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Normalization needed</text>
              <text x={10} y={90} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• 0 tolerance for synonyms</text>
              <Label x={65} y={135} tone={3} size={8.5} anchor="middle">Best for:</Label>
              <Label x={65} y={150} kind="error" size={9} bold anchor="middle">Regex / IDs / MCQ</Label>
            </g>

            {/* Column 2: pass@k */}
            <g transform="translate(180, 68)">
              <rect width={130} height={180} rx={4} fill={activeMetric === 'passk' ? 'var(--token-dim)' : 'var(--surface-1)'} stroke={activeMetric === 'passk' ? COLOR.token : 'var(--rule-2)'} strokeWidth={activeMetric === 'passk' ? 1.5 : 1} />
              <Label x={65} y={24} tone={1} size={10} bold anchor="middle">pass@k (Execution)</Label>
              <text x={10} y={50} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Sandboxed unit tests</text>
              <text x={10} y={70} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Unbiased estimator</text>
              <text x={10} y={90} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Ignores syntax style</text>
              <Label x={65} y={135} tone={3} size={8.5} anchor="middle">Best for:</Label>
              <Label x={65} y={150} kind="token" size={9} bold anchor="middle">Code / Math logic</Label>
            </g>

            {/* Column 3: ROUGE / BLEU */}
            <g transform="translate(324, 68)">
              <rect width={130} height={180} rx={4} fill={activeMetric === 'ngram' ? 'var(--compute-dim)' : 'var(--surface-1)'} stroke={activeMetric === 'ngram' ? COLOR.compute : 'var(--rule-2)'} strokeWidth={activeMetric === 'ngram' ? 1.5 : 1} />
              <Label x={65} y={24} tone={1} size={10} bold anchor="middle">ROUGE / BLEU</Label>
              <text x={10} y={50} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• N-gram precision/recall</text>
              <text x={10} y={70} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Brevity penalty</text>
              <text x={10} y={90} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Fails on negation</text>
              <Label x={65} y={135} tone={3} size={8.5} anchor="middle">Best for:</Label>
              <Label x={65} y={150} kind="compute" size={9} bold anchor="middle">Translation / Summaries</Label>
            </g>

            {/* Column 4: BERTScore */}
            <g transform="translate(468, 68)">
              <rect width={130} height={180} rx={4} fill={activeMetric === 'bert' ? 'var(--memory-dim)' : 'var(--surface-1)'} stroke={activeMetric === 'bert' ? COLOR.memory : 'var(--rule-2)'} strokeWidth={activeMetric === 'bert' ? 1.5 : 1} />
              <Label x={65} y={24} tone={1} size={10} bold anchor="middle">BERTScore</Label>
              <text x={10} y={50} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Contextual embedding</text>
              <text x={10} y={70} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Greedy cosine matching</text>
              <text x={10} y={90} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Semantic tolerance</text>
              <Label x={65} y={135} tone={3} size={8.5} anchor="middle">Best for:</Label>
              <Label x={65} y={150} kind="memory" size={9} bold anchor="middle">Paraphrase / QA</Label>
            </g>
          </g>
        )}
      </svg>
    </Stage>
  );
}
