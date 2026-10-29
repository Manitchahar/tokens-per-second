import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, fmt } from '../lib/svg';

const STEPS = [
  '1. Text to Vector',
  '2. L2 Normalization',
  '3. Cosine Similarity & Angle',
  '4. Linear Scan: N × d MACs',
  '5. Memory Footprint',
  '6. Memory Bandwidth Bottleneck',
  '7. The Need for ANN',
];

const CAPTIONS = [
  'An encoder model turns variable-length text into a single fixed-size floating-point vector via mean pooling.',
  'Vectors are normalized to unit length (L2 norm = 1), mapping every document and query onto the surface of a unit hypersphere.',
  'Cosine similarity measures the angle theta between vectors: cos(theta) = q · d. Orthogonal is 0, identical direction is 1.',
  'Exact k-NN performs a brute-force dot product of the query against every document vector: N × d multiply-accumulate operations.',
  'Memory scales linearly: 1M vectors of dimension 768 in FP32 require exactly 3.07 GB of RAM just to store raw floats.',
  'Brute force scan has arithmetic intensity of 0.5 FLOP/byte. It is entirely memory-bandwidth bound, taking 60+ ms on CPU.',
  'At 10M+ documents, linear scan breaks latency SLOs. We need approximate nearest neighbor (ANN) indexes to search sub-linearly.',
];

export default function EmbeddingSearch() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [nDocs, setNDocs] = useState(1000000); // 1M default
  const [dim, setDim] = useState(768);
  const [bytesPerFloat, setBytesPerFloat] = useState(4); // FP32

  const macs = nDocs * dim;
  const flops = macs * 2;
  const memoryBytes = nDocs * dim * bytesPerFloat;
  const memGB = memoryBytes / 1e9;

  // Bandwidth time: DDR5 CPU ~50 GB/s, T4 ~320 GB/s, H100 ~3350 GB/s
  const cpuTimeMs = (memGB / 50) * 1000;
  const t4TimeMs = (memGB / 320) * 1000;

  // Interactive visual angle
  const angleDeg = s.step <= 1 ? 0 : s.step === 2 ? 35 : 48;
  const rad = (angleDeg * Math.PI) / 180;
  const cosSim = Math.cos(rad);

  return (
    <Stage
      rootRef={ref}
      title="Embeddings & Exact Vector Search: Geometry, Memory and MACs"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'memory', 'compute']}
      {...s}
      readout={
        <Readout items={[
          {
            label: 'Vector Dimensions',
            value: `${dim} dims`,
            kind: 'token',
            note: `${dim * bytesPerFloat} bytes per vector (${bytesPerFloat === 4 ? 'FP32' : 'FP16'})`,
          },
          {
            label: 'Memory Footprint',
            value: `${memGB.toFixed(2)} GB`,
            kind: 'memory',
            note: `${fmt.int(nDocs)} docs × ${dim} dims × ${bytesPerFloat} B`,
          },
          {
            label: 'Exact Scan FLOPs',
            value: `${(flops / 1e9).toFixed(2)} GFLOPs`,
            kind: 'compute',
            note: `${(macs / 1e6).toFixed(0)}M MACs (${fmt.int(nDocs)} × ${dim})`,
          },
          {
            label: 'Linear Scan Time',
            value: `${cpuTimeMs.toFixed(1)} ms (CPU)`,
            kind: cpuTimeMs > 20 ? 'error' : 'compute',
            note: `${t4TimeMs.toFixed(1)} ms on T4 GPU (320 GB/s)`,
          },
        ]} />
      }
      controls={
        <>
          <Slider
            label="Document Count (N)"
            value={nDocs}
            min={100000}
            max={10000000}
            log
            onChange={setNDocs}
            format={(v) => `${fmt.int(v)} docs`}
            kind="memory"
          />
          <Segmented
            label="Model Dimension (d)"
            value={dim}
            options={[
              { value: 384, label: '384 · MiniLM' },
              { value: 768, label: '768 · BGE-base' },
              { value: 1536, label: '1536 · OpenAI' },
            ]}
            onChange={setDim}
          />
          <Segmented
            label="Precision"
            value={bytesPerFloat}
            options={[
              { value: 4, label: 'FP32 (4 B)' },
              { value: 2, label: 'FP16 (2 B)' },
            ]}
            onChange={setBytesPerFloat}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Vector geometry showing query and document vectors on a unit circle, cosine similarity angle, and brute-force linear scan memory scaling.">
        {/* Left Side: Geometry of Vector Space */}
        <g transform="translate(40, 30)">
          {/* Unit Circle Background */}
          <circle cx={110} cy={110} r={95} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} strokeDasharray="3 3" />
          <line x1={5} y1={110} x2={215} y2={110} stroke="var(--rule-2)" strokeWidth={0.8} />
          <line x1={110} y1={5} x2={110} y2={215} stroke="var(--rule-2)" strokeWidth={0.8} />
          <Label x={218} y={114} tone={3} size={8.5}>+x</Label>
          <Label x={110} y={15} anchor="middle" tone={3} size={8.5}>+y</Label>

          {/* Query Vector q (fixed along angle 20 deg) */}
          {(() => {
            const qAng = (15 * Math.PI) / 180;
            const qx = 110 + 95 * Math.cos(qAng);
            const qy = 110 - 95 * Math.sin(qAng);
            return (
              <g>
                <Arrow x1={110} y1={110} x2={qx} y2={qy} kind="token" width={2.2} />
                <Label x={qx + 6} y={qy + 4} tone={1} size={11} bold kind="token">q (Query)</Label>
              </g>
            );
          })()}

          {/* Doc Vector d1 (similar, angle 15 + angleDeg) */}
          {(() => {
            const dAng = ((15 + angleDeg) * Math.PI) / 180;
            const dx = 110 + 95 * Math.cos(dAng);
            const dy = 110 - 95 * Math.sin(dAng);
            return (
              <g>
                <Arrow x1={110} y1={110} x2={dx} y2={dy} kind="memory" width={2} />
                <Label x={dx + 6} y={dy} tone={1} size={10.5} bold kind="memory">d1 (Match)</Label>
              </g>
            );
          })()}

          {/* Doc Vector d2 (dissimilar, orthogonal-ish) */}
          {(() => {
            const d2Ang = (115 * Math.PI) / 180;
            const d2x = 110 + 95 * Math.cos(d2Ang);
            const d2y = 110 - 95 * Math.sin(d2Ang);
            return (
              <g>
                <Arrow x1={110} y1={110} x2={d2x} y2={d2y} kind="compute" width={1.4} dashed />
                <Label x={d2x - 8} y={d2y - 4} anchor="end" tone={3} size={9.5}>d2 (Irrelevant)</Label>
              </g>
            );
          })()}

          {/* Angle arc between q and d1 */}
          <path
            d="M 145 101 A 38 38 0 0 0 137 81"
            fill="none"
            stroke="var(--token)"
            strokeWidth={1.2}
          />
          <Label x={148} y={90} tone={2} size={10} bold kind="token">θ = {angleDeg}°</Label>
          <Label x={110} y={230} anchor="middle" tone={1} size={11} bold>
            cos(θ) = {cosSim.toFixed(3)}
          </Label>
          <Label x={110} y={244} anchor="middle" tone={3} size={9}>
            q · d on unit sphere
          </Label>
        </g>

        {/* Right Side: The Linear Scan Pipeline & Arithmetic */}
        <g transform="translate(290, 25)">
          <rect x={0} y={0} width={320} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={16} y={24} tone={1} size={12} bold>Brute Force Matrix-Vector Scan</Label>
          <Label x={16} y={40} tone={3} size={9.5}>Exact k-NN multiplies Query [1 × d] by Corpus [d × N]</Label>

          {/* Vector Matrix Graphic */}
          <g transform="translate(20, 55)">
            {/* Query vector q */}
            <rect x={0} y={0} width={65} height={18} rx={2} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={1} />
            <Label x={32} y={13} anchor="middle" tone={1} size={9} kind="token" bold>q [1 × {dim}]</Label>

            <Label x={75} y={13} tone={2} size={12}>×</Label>

            {/* Document Matrix D */}
            <rect x={90} y={-10} width={75} height={60} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} />
            <Label x={127} y={16} anchor="middle" tone={1} size={9} kind="memory" bold>D [{dim} × N]</Label>
            <Label x={127} y={32} anchor="middle" tone={3} size={8}>N = {fmt.int(nDocs)}</Label>

            <Label x={175} y={13} tone={2} size={12}>=</Label>

            {/* Scores Vector S */}
            <rect x={190} y={0} width={75} height={18} rx={2} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1} />
            <Label x={227} y={13} anchor="middle" tone={1} size={9} kind="compute" bold>Scores [1 × N]</Label>
          </g>

          {/* Hardware Bottleneck Breakdown */}
          <g transform="translate(16, 130)">
            <line x1={0} y1={0} x2={288} y2={0} stroke="var(--rule-2)" strokeWidth={0.8} />
            
            <Label x={0} y={20} tone={2} size={10} bold>Memory Required:</Label>
            <Label x={288} y={20} anchor="end" tone={1} size={10} bold kind="memory">
              {memGB.toFixed(2)} GB ({bytesPerFloat === 4 ? 'FP32' : 'FP16'})
            </Label>

            <Label x={0} y={38} tone={2} size={10} bold>Arithmetic Intensity:</Label>
            <Label x={288} y={38} anchor="end" tone={1} size={10} bold kind="compute">
              {(2 / bytesPerFloat).toFixed(2)} FLOP / Byte
            </Label>

            <Label x={0} y={56} tone={2} size={10} bold>Scan Time (CPU @ 50 GB/s):</Label>
            <Label x={288} y={56} anchor="end" tone={1} size={10} bold kind={cpuTimeMs > 20 ? 'error' : 'token'}>
              {cpuTimeMs.toFixed(1)} ms
            </Label>

            <Label x={0} y={74} tone={2} size={10} bold>Scan Time (T4 @ 320 GB/s):</Label>
            <Label x={288} y={74} anchor="end" tone={1} size={10} bold kind="token">
              {t4TimeMs.toFixed(1)} ms
            </Label>
          </g>
        </g>

        {/* Bottom Takeaway */}
        <g transform="translate(40, 275)">
          <rect x={0} y={0} width={570} height={45} rx={5} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={15} y={18} tone={1} size={10} bold>The Scaling Wall:</Label>
          <Label x={15} y={32} tone={2} size={9.5}>
            At 1M vectors, linear scan requires {memGB.toFixed(1)} GB of memory traffic. For sub-millisecond retrieval, we must prune 99% of candidates using ANN indexes.
          </Label>
        </g>
      </svg>
    </Stage>
  );
}
