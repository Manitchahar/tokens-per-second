import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow } from '../lib/svg';

const STEPS = [
  '1. The Exact Match Blindspot',
  '2. Inverted Index Lookup',
  '3. BM25 Frequency Saturation (k1)',
  '4. Length Normalization (b)',
  '5. The Score Scale Mismatch',
  '6. Reciprocal Rank Fusion (RRF)',
  '7. The Hybrid Advantage',
];

const CAPTIONS = [
  'Dense embeddings fail on exact IDs, part numbers, and rare entities: "ERR_404_OOM" is blurred into general "system errors".',
  'An inverted index maps every distinct token to a posting list of document IDs, enabling instantaneous O(1) keyword lookups.',
  'BM25 replaces linear term frequency with a saturating curve controlled by k1 (typically 1.2), preventing keyword-stuffing exploits.',
  'Parameter b (typically 0.75) penalizes verbose documents: a 10,000-word manual mentioning a word once is penalized over a concise 50-word memo.',
  'BM25 produces unbounded scores (e.g. 24.8), while cosine similarity lies in [-1, 1]. Direct score addition fails due to scale mismatch.',
  'Reciprocal Rank Fusion (RRF) sums 1 / (60 + rank) across sparse and dense lists. It relies only on relative rank order, never raw scores.',
  'Hybrid search combines the semantic reasoning of dense embeddings with the verbatim precision of BM25, outperforming either alone.',
];

export default function BM25Hybrid() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [k1, setK1] = useState(1.2);
  const [b, setB] = useState(0.75);
  const [docLen, setDocLen] = useState(250); // words
  const avgdl = 250;

  // BM25 term saturation curve calculation: f * (k1 + 1) / (f + k1 * (1 - b + b * (|D| / avgdl)))
  const lenNorm = 1 - b + b * (docLen / avgdl);
  const computeBM25Term = (tf: number) => (tf * (k1 + 1)) / (tf + k1 * lenNorm);

  // Worked RRF values
  // Doc A: BM25 rank 1, Dense rank 3
  const rrfA = 1 / (60 + 1) + 1 / (60 + 3);
  // Doc B: BM25 rank 4, Dense rank 1
  const rrfB = 1 / (60 + 4) + 1 / (60 + 1);
  // Doc C: BM25 rank 2, Dense rank 5
  const rrfC = 1 / (60 + 2) + 1 / (60 + 5);

  return (
    <Stage
      rootRef={ref}
      title="Sparse Keyword Search, BM25 & Reciprocal Rank Fusion"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'memory', 'compute', 'text']}
      {...s}
      readout={
        <Readout items={[
          {
            label: 'k1 (Saturation)',
            value: k1.toFixed(2),
            kind: 'compute',
            note: 'Cap on term frequency gain (as tf -> inf, factor -> k1 + 1)',
          },
          {
            label: 'b (Length penalty)',
            value: b.toFixed(2),
            kind: 'token',
            note: `${docLen} tokens vs ${avgdl} avg length (ratio: ${(docLen / avgdl).toFixed(2)})`,
          },
          {
            label: 'RRF Top Score (Doc A)',
            value: rrfA.toFixed(5),
            kind: 'memory',
            note: `1/(60+1) + 1/(60+3) = ${(1 / 61).toFixed(4)} + ${(1 / 63).toFixed(4)}`,
          },
          {
            label: 'RRF Doc B Score',
            value: rrfB.toFixed(5),
            kind: 'memory',
            note: `1/(60+4) + 1/(60+1) = ${(1 / 64).toFixed(4)} + ${(1 / 61).toFixed(4)}`,
          },
        ]} />
      }
      controls={
        <>
          <Slider
            label="BM25 k1 (Term Saturation)"
            value={k1}
            min={0.2}
            max={3.0}
            step={0.1}
            onChange={setK1}
            format={(v) => `k1 = ${v.toFixed(1)}`}
            kind="compute"
          />
          <Slider
            label="BM25 b (Length Normalization)"
            value={b}
            min={0.0}
            max={1.0}
            step={0.05}
            onChange={setB}
            format={(v) => `b = ${v.toFixed(2)}`}
            kind="token"
          />
          <Segmented
            label="Document Length (|D|)"
            value={docLen}
            options={[
              { value: 100, label: 'Short (100w)' },
              { value: 250, label: 'Average (250w)' },
              { value: 600, label: 'Long (600w)' },
            ]}
            onChange={setDocLen}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="BM25 saturation curve vs linear TF, inverted index posting lists, and Reciprocal Rank Fusion ranking comparison.">
        {/* Left Side: BM25 Term Saturation Curve */}
        <g transform="translate(30, 25)">
          <rect x={0} y={0} width={270} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={16} y={22} tone={1} size={11} bold>BM25 Term Saturation Curve</Label>
          <Label x={16} y={38} tone={3} size={9}>Score contribution vs Term Frequency (TF)</Label>

          {/* Plot axes */}
          <g transform="translate(40, 50)">
            <line x1={0} y1={140} x2={205} y2={140} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={0} y1={0} x2={0} y2={140} stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={100} y={158} anchor="middle" tone={3} size={8.5}>Term Frequency (tf in document)</Label>
            <Label x={-10} y={10} anchor="end" tone={3} size={8.5}>Score</Label>

            {/* Asymptotic ceiling k1 + 1 */}
            {(() => {
              const ceilY = 140 - (140 / 3.5) * (k1 + 1);
              return (
                <g>
                  <line x1={0} y1={ceilY} x2={200} y2={ceilY} stroke="var(--rule-2)" strokeWidth={0.8} strokeDasharray="3 3" />
                  <Label x={202} y={ceilY + 3} tone={3} size={8}>Ceiling: {(k1 + 1).toFixed(1)}</Label>
                </g>
              );
            })()}

            {/* Linear TF curve (raw unconstrained count) */}
            <line x1={0} y1={140} x2={160} y2={20} stroke="var(--error)" strokeWidth={1} strokeDasharray="2 3" opacity={0.6} />
            <Label x={165} y={24} tone={3} size={7.5} kind="error">Naive TF (exploitable)</Label>

            {/* BM25 Curve */}
            {(() => {
              const points: string[] = [];
              for (let tf = 0; tf <= 15; tf += 0.5) {
                const px = (tf / 15) * 190;
                const score = computeBM25Term(tf);
                // Scale so score=3.5 reaches top
                const py = 140 - (score / 3.5) * 130;
                points.push(`${px.toFixed(1)},${py.toFixed(1)}`);
              }
              return (
                <polyline
                  points={points.join(' ')}
                  fill="none"
                  stroke="var(--compute)"
                  strokeWidth={2}
                />
              );
            })()}

            {/* Marker at tf = 3 */}
            {(() => {
              const tf = 3;
              const px = (tf / 15) * 190;
              const score = computeBM25Term(tf);
              const py = 140 - (score / 3.5) * 130;
              return (
                <g>
                  <circle cx={px} cy={py} r={3.5} fill="var(--compute)" />
                  <Label x={px + 6} y={py - 6} tone={1} size={8.5} bold kind="compute">
                    tf=3 → {score.toFixed(2)}
                  </Label>
                </g>
              );
            })()}
          </g>
        </g>

        {/* Right Side: Reciprocal Rank Fusion Matrix */}
        <g transform="translate(320, 25)">
          <rect x={0} y={0} width={290} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={16} y={22} tone={1} size={11} bold>Reciprocal Rank Fusion (RRF)</Label>
          <Label x={16} y={38} tone={3} size={9}>Score(d) = Σ 1 / (60 + rank)</Label>

          {/* Table Header */}
          <g transform="translate(16, 52)">
            <rect x={0} y={0} width={258} height={22} rx={3} fill="var(--ground-3)" />
            <Label x={10} y={15} tone={3} size={8.5} bold>Doc</Label>
            <Label x={55} y={15} tone={3} size={8.5} bold>BM25 Rank</Label>
            <Label x={125} y={15} tone={3} size={8.5} bold>Dense Rank</Label>
            <Label x={195} y={15} tone={3} size={8.5} bold>RRF Score</Label>

            {/* Row 1: Doc A (Keyword champion) */}
            <g transform="translate(0, 26)">
              <rect x={0} y={0} width={258} height={32} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} />
              <Label x={10} y={20} tone={1} size={9.5} bold kind="memory">Doc A</Label>
              <Label x={55} y={20} tone={1} size={9}>#1 (1/61)</Label>
              <Label x={125} y={20} tone={2} size={9}>#3 (1/63)</Label>
              <Label x={195} y={20} tone={1} size={9.5} bold kind="token">{rrfA.toFixed(4)}</Label>
            </g>

            {/* Row 2: Doc B (Dense champion) */}
            <g transform="translate(0, 62)">
              <rect x={0} y={0} width={258} height={32} rx={2} fill="var(--ground-1)" stroke="var(--rule-2)" strokeWidth={0.6} />
              <Label x={10} y={20} tone={1} size={9.5} bold>Doc B</Label>
              <Label x={55} y={20} tone={2} size={9}>#4 (1/64)</Label>
              <Label x={125} y={20} tone={1} size={9}>#1 (1/61)</Label>
              <Label x={195} y={20} tone={2} size={9.5} bold>{rrfB.toFixed(4)}</Label>
            </g>

            {/* Row 3: Doc C (Consistently moderate) */}
            <g transform="translate(0, 98)">
              <rect x={0} y={0} width={258} height={32} rx={2} fill="var(--ground-1)" stroke="var(--rule-2)" strokeWidth={0.6} />
              <Label x={10} y={20} tone={1} size={9.5} bold>Doc C</Label>
              <Label x={55} y={20} tone={2} size={9}>#2 (1/62)</Label>
              <Label x={125} y={20} tone={2} size={9}>#5 (1/65)</Label>
              <Label x={195} y={20} tone={3} size={9.5}>{rrfC.toFixed(4)}</Label>
            </g>
          </g>

          {/* RRF Formula Callout */}
          <g transform="translate(16, 195)">
            <rect x={0} y={0} width={258} height={28} rx={3} fill="var(--ground-3)" />
            <Label x={129} y={18} anchor="middle" tone={2} size={9} mono>
              RRF = 1/(60 + r_sparse) + 1/(60 + r_dense)
            </Label>
          </g>
        </g>

        {/* Bottom Takeaway */}
        <g transform="translate(30, 275)">
          <rect x={0} y={0} width={580} height={45} rx={5} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={15} y={18} tone={1} size={10} bold>The Hybrid Principle:</Label>
          <Label x={15} y={32} tone={2} size={9.5}>
            Never normalize and add BM25 scores to cosine similarities. Fuse their <tspan fill="var(--token)" fontWeight="600">ranks</tspan> using RRF to eliminate score calibration bugs permanently.
          </Label>
        </g>
      </svg>
    </Stage>
  );
}
