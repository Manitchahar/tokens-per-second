import { useRef, useState, useEffect } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, COLOR, DIM, fmt } from '../lib/svg';

const STEPS = [
  'Repeated data hurts: memorization',
  'Exact hashing: SHA-256 limits',
  'Shingling: text into k-grams',
  'MinHash: permutation signatures',
  'LSH banding: sub-quadratic search',
  'The complete cleaning pipeline',
];

const CAPTIONS = [
  'Hernandez et al. (2022) showed that repeated data causes models to memorize tokens exponentially faster, wasting compute and harming generalization.',
  'Exact SHA-256 hashing finds byte-identical duplicates in O(N) time, but fails if a single space, comma, or timestamp differs between two texts.',
  'Shingling splits documents into overlapping sets of k-grams (e.g. 5-word shingles). The Jaccard similarity measures their relative set intersection.',
  'MinHash compresses large sets of shingles into m integer hashes. The probability that two MinHash values match equals their exact Jaccard similarity.',
  'Locality-Sensitive Hashing (LSH) groups m hashes into b bands of r rows. The S-curve enables finding all candidate duplicates without comparing all N² pairs.',
  'A production pipeline layers exact hashing, MinHash LSH, PII redaction, and boilerplate stripping to deliver a pristine corpus.',
];

export default function DedupCleaningViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [sim, setSim] = useState(0.8);
  const [bands, setBands] = useState(16);
  const rows = 8; // m = 128 total hashes = 16 bands * 8 rows

  useEffect(() => {
    if (s.step === 0) setSim(0.9);
    else if (s.step === 1) setSim(0.95);
    else if (s.step === 2) setSim(0.75);
    else if (s.step === 3) setSim(0.8);
    else if (s.step === 4) setSim(0.8);
    else if (s.step === 5) setSim(0.85);
  }, [s.step]);

  const animSim = useTween(sim, 300);

  // LSH probability math: P = 1 - (1 - s^r)^b
  const probCandidate = 1 - Math.pow(1 - Math.pow(animSim, rows), bands);
  const threshold = Math.pow(1 / bands, 1 / rows);

  return (
    <Stage
      rootRef={ref}
      title="Deduplication at Scale: MinHash & LSH Banding"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'compute', 'memory', 'error']}
      readout={
        <Readout
          items={[
            {
              label: 'Jaccard Similarity (s)',
              value: animSim.toFixed(2),
              kind: animSim >= threshold ? 'token' : 'idle',
              note: `${(animSim * 100).toFixed(0)}% shingle overlap between docs`,
            },
            {
              label: 'Candidate Match Prob',
              value: `${(probCandidate * 100).toFixed(1)}%`,
              kind: probCandidate > 0.5 ? 'token' : 'idle',
              note: `P = 1 - (1 - s^${rows})^${bands}`,
            },
            {
              label: 'LSH Threshold (t)',
              value: threshold.toFixed(2),
              kind: 'memory',
              note: `t ≈ (1 / ${bands})^(1 / ${rows})`,
            },
            {
              label: 'Comparison Complexity',
              value: s.step === 4 ? 'O(N) via Buckets' : 'O(N²) All-Pairs',
              kind: s.step === 4 ? 'token' : 'error',
              note: s.step === 4 ? '10M docs: feasible in minutes' : '10M docs: 10¹⁴ checks (impossible)',
            },
          ]}
        />
      }
      controls={
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', width: '100%', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <Slider
              label="Document Similarity (s)"
              value={sim}
              min={0.0}
              max={1.0}
              step={0.05}
              onChange={setSim}
            />
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <Slider
              label="LSH Bands (b)"
              value={bands}
              min={4}
              max={32}
              step={4}
              onChange={setBands}
            />
          </div>
        </div>
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Visualizing deduplication, MinHash, and Locality-Sensitive Hashing">
        {/* Step 0: Hernandez Memorization Scaling Law */}
        {s.step === 0 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              The Memorization Danger: Repeated Data vs Generalization
            </Label>

            <rect x={50} y={65} width={540} height={130} rx={4} fill="#141820" stroke={COLOR.error} />
            <Label x={70} y={92} anchor="start" size={12} bold style={{ fill: COLOR.error }}>
              Hernandez et al. (2022) Finding:
            </Label>
            <Label x={70} y={115} anchor="start" size={10} tone={2}>
              Memorization scales super-linearly with data repetition.
            </Label>
            <Label x={70} y={135} anchor="start" size={9.5} tone={3}>
              An instruction repeated 10× in the training set is memorized up to 100× more strongly,
            </Label>
            <Label x={70} y={152} anchor="start" size={9.5} tone={3}>
              causing the model to parrot training inputs verbatim rather than generalizing.
            </Label>

            <g>
              <Span x={70} y={165} w={150} h={20} kind="token" label="Unique: Good generalization" />
              <Span x={230} y={165} w={170} h={20} kind="compute" label="10× Duplicates: Overfit" />
              <Span x={410} y={165} w={160} h={20} kind="error" label="100× Duplicates: Memorized" />
            </g>

            <rect x={50} y={215} width={540} height={45} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
            <Label x={320} y={235} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
              Lee et al. (2021): Deduplicating Web Crawls Removes Up to 50% Redundant Tokens
            </Label>
            <Label x={320} y={249} anchor="middle" size={9} tone={2}>
              Training on deduplicated corpora improves perplexity and slashes total training FLOPs.
            </Label>
          </g>
        )}

        {/* Step 1: Exact Hashing vs Near-Duplicate Trap */}
        {s.step === 1 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              Exact Hashing (SHA-256) vs Near-Duplicate Failure
            </Label>

            {/* Doc A */}
            <rect x={40} y={65} width={260} height={110} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={55} y={88} anchor="start" size={11} bold style={{ fill: COLOR.token }}>
              Document A: Original
            </Label>
            <Label x={55} y={108} anchor="start" size={9} mono tone={2}>
              &quot;FlashAttention reduces HBM reads.&quot;
            </Label>
            <Label x={55} y={135} anchor="start" size={9} tone={3}>
              SHA-256: e8b3f1...a429
            </Label>

            {/* Doc B */}
            <rect x={340} y={65} width={260} height={110} rx={4} fill="#141820" stroke={COLOR.error} />
            <Label x={355} y={88} anchor="start" size={11} bold style={{ fill: COLOR.error }}>
              Document B: 1-Character Alteration
            </Label>
            <Label x={355} y={108} anchor="start" size={9} mono tone={2}>
              &quot;FlashAttention reduces HBM reads!&quot;
            </Label>
            <Label x={355} y={135} anchor="start" size={9} tone={3}>
              SHA-256: 3c94d0...f11b (Completely Different!)
            </Label>

            <rect x={40} y={195} width={560} height={60} rx={4} fill="#141820" stroke={COLOR.error} />
            <Label x={60} y={218} anchor="start" size={10} bold style={{ fill: COLOR.error }}>
              Exact Hash Collision Check Result: False Negative!
            </Label>
            <Label x={60} y={236} anchor="start" size={9} tone={2}>
              Cryptographic hashes exhibit the avalanche effect: changing 1 bit scrambles 50% of hash bits.
            </Label>
            <Label x={60} y={248} anchor="start" size={9} tone={3}>
              Near-duplicates bypass exact SHA-256 filters entirely. We need fuzzy Jaccard similarity.
            </Label>
          </g>
        )}

        {/* Step 2: Shingling */}
        {s.step === 2 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              Shingling: Converting Text into Sets of k-grams
            </Label>

            <rect x={40} y={65} width={560} height={120} rx={4} fill="#141820" stroke={COLOR.memory} />
            <Label x={60} y={90} anchor="start" size={11} bold style={{ fill: COLOR.memory }}>
              Document: &quot;FlashAttention reduces memory traffic on GPUs&quot;
            </Label>
            <Label x={60} y={110} anchor="start" size={10} tone={2}>
              Word 3-grams (Shingles of length k = 3):
            </Label>

            <g>
              <Span x={60} y={125} w={150} h={25} kind="memory" label="['FlashAttention', 'reduces', 'memory']" />
              <Span x={220} y={125} w={150} h={25} kind="memory" label="['reduces', 'memory', 'traffic']" />
              <Span x={380} y={125} w={140} h={25} kind="memory" label="['memory', 'traffic', 'on']" />
            </g>

            <rect x={40} y={205} width={560} height={50} rx={4} fill={DIM.token} stroke={COLOR.token} />
            <Label x={320} y={226} anchor="middle" size={10} bold style={{ fill: COLOR.token }}>
              Jaccard Similarity: J(A, B) = |A ∩ B| / |A ∪ B|
            </Label>
            <Label x={320} y={242} anchor="middle" size={9} tone={2}>
              Documents with small typos share 80%+ of their shingles. But computing intersections for N docs is O(N²).
            </Label>
          </g>
        )}

        {/* Step 3: MinHash Signatures */}
        {s.step === 3 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              MinHash: Probabilistic Jaccard Estimation in O(m)
            </Label>

            <rect x={50} y={65} width={260} height={130} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={65} y={88} anchor="start" size={11} bold style={{ fill: COLOR.token }}>
              Doc A: MinHash Signature
            </Label>
            <Label x={65} y={108} anchor="start" size={9} tone={2}>
              m = 128 hash permutations
            </Label>
            <Span x={65} y={120} w={230} h={22} kind="token" label="[h₁(A), h₂(A), ..., h₁₂₈(A)]" />
            <Label x={65} y={165} anchor="start" size={9} tone={3}>
              hᵢ(A) = min_{'{x∈A}'} hash_i(x)
            </Label>

            <rect x={330} y={65} width={260} height={130} rx={4} fill="#141820" stroke={COLOR.token} />
            <Label x={345} y={88} anchor="start" size={11} bold style={{ fill: COLOR.token }}>
              Doc B: MinHash Signature
            </Label>
            <Label x={345} y={108} anchor="start" size={9} tone={2}>
              m = 128 hash permutations
            </Label>
            <Span x={345} y={120} w={230} h={22} kind="token" label="[h₁(B), h₂(B), ..., h₁₂₈(B)]" />
            <Label x={345} y={165} anchor="start" size={9} tone={3}>
              hᵢ(B) = min_{'{x∈B}'} hash_i(x)
            </Label>

            <rect x={50} y={215} width={540} height={50} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
            <Label x={320} y={236} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
              Fundamental Theorem: P(min h(A) == min h(B)) = Jaccard(A, B)
            </Label>
            <Label x={320} y={252} anchor="middle" size={9} tone={2}>
              The fraction of identical signature slots directly approximates true document similarity.
            </Label>
          </g>
        )}

        {/* Step 4: LSH Banding & The S-Curve */}
        {s.step === 4 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              Locality-Sensitive Hashing: The S-Curve Threshold
            </Label>

            {/* S-curve chart */}
            <line x1={80} y1={230} x2={560} y2={230} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={80} y1={60} x2={80} y2={230} stroke="var(--rule-2)" strokeWidth={1} />

            <Label x={70} y={65} anchor="end" size={9} tone={3}>
              1.0
            </Label>
            <Label x={70} y={145} anchor="end" size={9} tone={3}>
              0.5
            </Label>
            <Label x={70} y={230} anchor="end" size={9} tone={3}>
              0.0
            </Label>

            <Label x={80} y={245} anchor="start" size={9} tone={3}>
              0.0
            </Label>
            <Label x={320} y={245} anchor="middle" size={9} tone={2}>
              Jaccard Similarity s
            </Label>
            <Label x={560} y={245} anchor="end" size={9} tone={3}>
              1.0
            </Label>

            {/* S-curve plotting */}
            {(() => {
              const pts: [number, number][] = [];
              for (let sVal = 0; sVal <= 1.0; sVal += 0.02) {
                const pVal = 1 - Math.pow(1 - Math.pow(sVal, rows), bands);
                const x = 80 + sVal * 480;
                const y = 230 - pVal * 170;
                pts.push([x, y]);
              }
              const d = pts.reduce((acc, [x, y], idx) => (idx === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`), '');
              return <path d={d} fill="none" stroke={COLOR.token} strokeWidth={2.5} />;
            })()}

            {/* Current point */}
            {(() => {
              const curX = 80 + animSim * 480;
              const curY = 230 - probCandidate * 170;
              return (
                <g>
                  <circle cx={curX} cy={curY} r={6} fill={COLOR.token} />
                  <line x1={curX} y1={curY} x2={curX} y2={230} stroke="var(--rule-2)" strokeDasharray="2 2" />
                  <line x1={80} y1={curY} x2={curX} y2={curY} stroke="var(--rule-2)" strokeDasharray="2 2" />
                  <rect x={curX - 45} y={curY - 24} width={90} height={18} rx={2} fill="#141820" stroke="var(--rule-2)" />
                  <Label x={curX} y={curY - 12} anchor="middle" size={9} bold tone={1}>
                    {(probCandidate * 100).toFixed(0)}% Candidate
                  </Label>
                </g>
              );
            })()}

            {/* Threshold Line */}
            {(() => {
              const tx = 80 + threshold * 480;
              return (
                <g>
                  <line x1={tx} y1={60} x2={tx} y2={230} stroke={COLOR.compute} strokeWidth={1.5} strokeDasharray="3 3" />
                  <Label x={tx} y={55} anchor="middle" size={9} bold style={{ fill: COLOR.compute }}>
                    Threshold t ≈ {threshold.toFixed(2)}
                  </Label>
                </g>
              );
            })()}
          </g>
        )}

        {/* Step 5: Full Cleaning Pipeline */}
        {s.step === 5 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              The Production Dataset Cleaning Pipeline
            </Label>

            {/* Stage 1 */}
            <rect x={50} y={65} width={120} height={90} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
            <Label x={110} y={88} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
              1. Boilerplate & PII
            </Label>
            <Label x={110} y={108} anchor="middle" size={8.5} tone={2}>
              Strip HTML/headers
            </Label>
            <Label x={110} y={122} anchor="middle" size={8.5} tone={2}>
              Regex redact emails/IPs
            </Label>
            <Label x={110} y={138} anchor="middle" size={8} tone={3}>
              Remove cookie junk
            </Label>

            <line x1={170} y1={110} x2={195} y2={110} stroke="var(--rule-2)" strokeWidth={1.5} />

            {/* Stage 2 */}
            <rect x={195} y={65} width={120} height={90} rx={4} fill={DIM.token} stroke={COLOR.token} />
            <Label x={255} y={88} anchor="middle" size={10} bold style={{ fill: COLOR.token }}>
              2. Exact SHA-256
            </Label>
            <Label x={255} y={108} anchor="middle" size={8.5} tone={2}>
              Normalize whitespace
            </Label>
            <Label x={255} y={122} anchor="middle" size={8.5} tone={2}>
              Hash table lookup
            </Label>
            <Label x={255} y={138} anchor="middle" size={8} tone={3}>
              Drop identical texts
            </Label>

            <line x1={315} y1={110} x2={340} y2={110} stroke="var(--rule-2)" strokeWidth={1.5} />

            {/* Stage 3 */}
            <rect x={340} y={65} width={120} height={90} rx={4} fill={DIM.memory} stroke={COLOR.memory} />
            <Label x={400} y={88} anchor="middle" size={10} bold style={{ fill: COLOR.memory }}>
              3. MinHash + LSH
            </Label>
            <Label x={400} y={108} anchor="middle" size={8.5} tone={2}>
              5-gram shingles
            </Label>
            <Label x={400} y={122} anchor="middle" size={8.5} tone={2}>
              m = 128, b = 16
            </Label>
            <Label x={400} y={138} anchor="middle" size={8} tone={3}>
              Drop near-duplicates
            </Label>

            <line x1={460} y1={110} x2={485} y2={110} stroke="var(--rule-2)" strokeWidth={1.5} />

            {/* Stage 4 */}
            <rect x={485} y={65} width={110} height={90} rx={4} fill="#141820" stroke="var(--rule-2)" />
            <Label x={540} y={88} anchor="middle" size={10} bold tone={1}>
              4. Clean Corpus
            </Label>
            <Label x={540} y={108} anchor="middle" size={8.5} tone={2}>
              Quality verified
            </Label>
            <Label x={540} y={122} anchor="middle" size={8.5} tone={2}>
              No test leaks
            </Label>
            <Label x={540} y={138} anchor="middle" size={8} tone={3}>
              Ready for SFT
            </Label>

            <rect x={50} y={185} width={545} height={60} rx={4} fill="#141820" stroke="var(--rule-2)" />
            <Label x={320} y={208} anchor="middle" size={10} bold tone={1}>
              Result: 30%–50% Data Reduction, Better Downstream Generalization & 2× Faster Training
            </Label>
            <Label x={320} y={226} anchor="middle" size={9} tone={2}>
              Deduplication prevents the model from memorizing repetitive web boilerplate and test benchmarks.
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
