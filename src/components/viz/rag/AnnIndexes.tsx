import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow } from '../lib/svg';

const STEPS = [
  '1. The Linear Scan Wall',
  '2. IVF: Voronoi Partitioning',
  '3. IVF Tuning: The nprobe Knob',
  '4. HNSW: Multi-Layer Graph',
  '5. HNSW: Greedy Highway Descent',
  '6. Product Quantization (PQ)',
  '7. The Pareto Frontier',
];

const CAPTIONS = [
  'Exact brute-force search compares the query against every vector. At 1M+ vectors, memory bandwidth limits query throughput.',
  'Inverted File Index (IVF) clusters space into C centroids (Voronoi cells). Each vector is assigned to its nearest centroid list.',
  'At query time, the engine visits only the top nprobe centroids. Small nprobe is fast; larger nprobe catches edge vectors.',
  'HNSW organizes vectors into a hierarchy of proximity graphs inspired by skip-lists, with long-range edges at top layers.',
  'Search begins at the sparse top layer, taking long highway jumps toward the query before dropping down to dense local layers.',
  'Product Quantization (PQ) slices 768-d vectors into 96 sub-vectors of 8 dims, compressing 3,072 bytes down to 96 bytes (32x).',
  'The engineering trade-off: Flat is 100% accurate but slow; IVF balances memory and speed; HNSW delivers sub-millisecond queries.',
];

export default function AnnIndexes() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [nprobe, setNprobe] = useState(3);
  const [indexType, setIndexType] = useState<'flat' | 'ivf' | 'hnsw' | 'pq'>('hnsw');

  // Sync indexType with step if in step-through
  const activeType = s.step <= 0 ? 'flat' : s.step <= 2 ? 'ivf' : s.step <= 4 ? 'hnsw' : s.step === 5 ? 'pq' : indexType;

  // Recall & latency model approximations for 1M vectors
  // Flat: 100% recall, 60 ms
  // IVF (C=1024): recall scales with nprobe / 1024
  const ivfRecall = Math.min(99.5, 70 + (nprobe / 32) * 29);
  const ivfLatency = 0.4 + (nprobe / 32) * 4.5;

  // HNSW: ~98% recall, 1.2 ms
  // PQ: ~88% recall, 0.8 ms, 32x compression
  const recallVal = activeType === 'flat' ? 100 : activeType === 'ivf' ? ivfRecall : activeType === 'hnsw' ? 98.4 : 88.5;
  const latencyVal = activeType === 'flat' ? 61.4 : activeType === 'ivf' ? ivfLatency : activeType === 'hnsw' ? 1.2 : 0.8;
  const memoryGB = activeType === 'pq' ? 0.096 : 3.07;

  return (
    <Stage
      rootRef={ref}
      title="ANN Indexing: IVF Voronoi Cells, HNSW Graphs & Product Quantization"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'memory', 'compute', 'idle']}
      {...s}
      readout={
        <Readout items={[
          {
            label: 'Index Structure',
            value: activeType.toUpperCase(),
            kind: 'token',
            note: activeType === 'flat' ? 'Exact brute-force linear scan' : activeType === 'ivf' ? `IVF (nprobe=${nprobe})` : activeType === 'hnsw' ? 'HNSW (Hierarchical Graph)' : 'Product Quantization (PQ)',
          },
          {
            label: 'Recall@10',
            value: `${recallVal.toFixed(1)}%`,
            kind: recallVal > 95 ? 'token' : recallVal > 85 ? 'compute' : 'error',
            note: 'Fraction of true nearest neighbors found',
          },
          {
            label: 'Query Latency (1M)',
            value: `${latencyVal.toFixed(1)} ms`,
            kind: latencyVal < 2 ? 'token' : latencyVal < 10 ? 'compute' : 'error',
            note: 'Search time per query',
          },
          {
            label: 'Memory Footprint',
            value: `${memoryGB.toFixed(2)} GB`,
            kind: 'memory',
            note: activeType === 'pq' ? '32× compression (96 bytes/vec)' : 'FP32 uncompressed (3,072 B/vec)',
          },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Index Paradigm"
            value={activeType}
            options={[
              { value: 'flat', label: '1. Flat (Exact)' },
              { value: 'ivf', label: '2. IVF (Centroids)' },
              { value: 'hnsw', label: '3. HNSW (Graph)' },
              { value: 'pq', label: '4. PQ (Quantized)' },
            ]}
            onChange={(v) => setIndexType(v as 'flat' | 'ivf' | 'hnsw' | 'pq')}
          />
          {activeType === 'ivf' && (
            <Slider
              label="IVF nprobe (Centroids to search)"
              value={nprobe}
              min={1}
              max={32}
              step={1}
              onChange={setNprobe}
              format={(v) => `nprobe = ${v} / 1024 cells`}
              kind="compute"
            />
          )}
        </>
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Visual breakdown of IVF Voronoi cell partitioning with nprobe search and HNSW multi-layer hierarchical graph walk.">
        {/* PARADIGM 1 & 2: IVF VORONOI CLUSTERS */}
        {(activeType === 'flat' || activeType === 'ivf') && (
          <g transform="translate(30, 25)">
            <rect x={0} y={0} width={280} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={16} y={22} tone={1} size={11} bold>
              {activeType === 'flat' ? 'Brute Force Linear Scan' : `IVF Voronoi Cells (nprobe = ${nprobe})`}
            </Label>
            <Label x={16} y={38} tone={3} size={9}>
              {activeType === 'flat' ? 'Checks all 1,000,000 vectors' : 'Clusters 1,000,000 vectors into Voronoi cells'}
            </Label>

            {/* Grid of 6 Voronoi Cells */}
            <g transform="translate(20, 50)">
              {/* Cell 1 (top-left) */}
              <rect x={0} y={0} width={75} height={70} rx={4}
                fill={activeType === 'ivf' && nprobe >= 1 ? 'var(--memory-dim)' : 'var(--ground-3)'}
                stroke={activeType === 'ivf' && nprobe >= 1 ? 'var(--memory)' : 'var(--rule-2)'}
                strokeWidth={activeType === 'ivf' && nprobe >= 1 ? 1.4 : 0.8} />
              <circle cx={37} cy={35} r={4} fill="var(--memory)" />
              <Label x={37} y={52} anchor="middle" tone={3} size={7.5}>Centroid 1</Label>

              {/* Cell 2 (top-mid) - Contains Query! */}
              <rect x={82} y={0} width={75} height={70} rx={4}
                fill={activeType === 'ivf' ? 'var(--token-dim)' : 'var(--ground-3)'}
                stroke="var(--token)"
                strokeWidth={1.4} />
              <circle cx={119} cy={35} r={4} fill="var(--memory)" />
              {/* Query point q */}
              <circle cx={128} cy={28} r={5} fill="var(--token)" />
              <Label x={136} y={26} tone={1} size={8.5} bold kind="token">q</Label>
              <Label x={119} y={52} anchor="middle" tone={2} size={7.5}>Nearest Cell</Label>

              {/* Cell 3 (top-right) */}
              <rect x={164} y={0} width={75} height={70} rx={4}
                fill={activeType === 'ivf' && nprobe >= 2 ? 'var(--memory-dim)' : 'var(--ground-3)'}
                stroke={activeType === 'ivf' && nprobe >= 2 ? 'var(--memory)' : 'var(--rule-2)'}
                strokeWidth={activeType === 'ivf' && nprobe >= 2 ? 1.4 : 0.8} />
              <circle cx={201} cy={35} r={4} fill="var(--memory)" />
              <Label x={201} y={52} anchor="middle" tone={3} size={7.5}>Centroid 3</Label>

              {/* Cell 4 (bot-left) */}
              <rect x={0} y={76} width={75} height={70} rx={4}
                fill={activeType === 'ivf' && nprobe >= 5 ? 'var(--memory-dim)' : 'var(--ground-3)'}
                stroke={activeType === 'ivf' && nprobe >= 5 ? 'var(--memory)' : 'var(--rule-2)'}
                strokeWidth={0.8} />
              <circle cx={37} cy={111} r={4} fill="var(--memory)" />

              {/* Cell 5 (bot-mid) */}
              <rect x={82} y={76} width={75} height={70} rx={4}
                fill={activeType === 'ivf' && nprobe >= 3 ? 'var(--memory-dim)' : 'var(--ground-3)'}
                stroke={activeType === 'ivf' && nprobe >= 3 ? 'var(--memory)' : 'var(--rule-2)'}
                strokeWidth={activeType === 'ivf' && nprobe >= 3 ? 1.4 : 0.8} />
              <circle cx={119} cy={111} r={4} fill="var(--memory)" />
              <Label x={119} y={128} anchor="middle" tone={3} size={7.5}>Centroid 5</Label>

              {/* Cell 6 (bot-right) */}
              <rect x={164} y={76} width={75} height={70} rx={4}
                fill={activeType === 'ivf' && nprobe >= 4 ? 'var(--memory-dim)' : 'var(--ground-3)'}
                stroke={activeType === 'ivf' && nprobe >= 4 ? 'var(--memory)' : 'var(--rule-2)'}
                strokeWidth={0.8} />
              <circle cx={201} cy={111} r={4} fill="var(--memory)" />
            </g>
          </g>
        )}

        {/* PARADIGM 3: HNSW MULTI-LAYER GRAPH */}
        {activeType === 'hnsw' && (
          <g transform="translate(30, 25)">
            <rect x={0} y={0} width={280} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={16} y={22} tone={1} size={11} bold>HNSW Multi-Layer Skip Graph</Label>
            <Label x={16} y={38} tone={3} size={9}>Hierarchical Navigable Small World</Label>

            {/* Layer 2: Express Highway */}
            <g transform="translate(20, 52)">
              <rect x={0} y={0} width={240} height={40} rx={3} fill="var(--ground-3)" />
              <Label x={10} y={15} tone={3} size={8} bold>Layer 2 (Express Highway)</Label>
              <circle cx={40} cy={24} r={4} fill="var(--token)" />
              <Label x={40} y={36} anchor="middle" tone={1} size={7} kind="token">Entry</Label>
              <line x1={44} y1={24} x2={196} y2={24} stroke="var(--token)" strokeWidth={1.4} />
              <circle cx={200} cy={24} r={4} fill="var(--memory)" />
            </g>

            {/* Down arrow to Layer 1 */}
            <Arrow x1={220} y1={92} x2={220} y2={108} kind="compute" width={1.2} />

            {/* Layer 1: Regional Roads */}
            <g transform="translate(20, 110)">
              <rect x={0} y={0} width={240} height={42} rx={3} fill="var(--ground-3)" />
              <Label x={10} y={15} tone={3} size={8} bold>Layer 1 (Regional)</Label>
              <circle cx={40} cy={25} r={3.5} fill="var(--memory)" />
              <circle cx={120} cy={25} r={3.5} fill="var(--memory)" />
              <circle cx={200} cy={25} r={3.5} fill="var(--token)" />
              <line x1={43} y1={25} x2={117} y2={25} stroke="var(--rule-2)" strokeWidth={1} />
              <line x1={123} y1={25} x2={197} y2={25} stroke="var(--token)" strokeWidth={1.2} />
            </g>

            {/* Down arrow to Layer 0 */}
            <Arrow x1={140} y1={152} x2={140} y2={168} kind="compute" width={1.2} />

            {/* Layer 0: All Vectors (Dense local) */}
            <g transform="translate(20, 170)">
              <rect x={0} y={0} width={240} height={45} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} />
              <Label x={10} y={14} tone={1} size={8} bold kind="memory">Layer 0 (All 1M Vectors)</Label>
              <circle cx={30} cy={28} r={3} fill="var(--memory)" />
              <circle cx={70} cy={28} r={3} fill="var(--memory)" />
              <circle cx={110} cy={28} r={3} fill="var(--memory)" />
              <circle cx={140} cy={28} r={4} fill="var(--token)" />
              <circle cx={180} cy={28} r={3} fill="var(--memory)" />
              <circle cx={220} cy={28} r={3} fill="var(--memory)" />
              <Label x={140} y={40} anchor="middle" tone={1} size={7.5} bold kind="token">Nearest Match</Label>
            </g>
          </g>
        )}

        {/* PARADIGM 4: PRODUCT QUANTIZATION (PQ) */}
        {activeType === 'pq' && (
          <g transform="translate(30, 25)">
            <rect x={0} y={0} width={280} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={16} y={22} tone={1} size={11} bold>Product Quantization (PQ-96)</Label>
            <Label x={16} y={38} tone={3} size={9}>Compresses 3,072 bytes → 96 bytes (32×)</Label>

            <g transform="translate(15, 60)">
              {/* Original 768 float vector */}
              <rect x={0} y={0} width={250} height={22} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} />
              <Label x={125} y={15} anchor="middle" tone={1} size={8.5} kind="memory" bold>
                Original Vector: 768 floats (3,072 B in FP32)
              </Label>

              <Arrow x1={125} y1={25} x2={125} y2={45} kind="compute" width={1.2} />

              {/* Sliced into 96 sub-vectors */}
              <g transform="translate(0, 48)">
                <Label x={0} y={14} tone={2} size={8.5}>Slice into 96 sub-vectors of 8 dimensions:</Label>
                <g transform="translate(0, 20)">
                  <rect x={0} y={0} width={55} height={20} rx={2} fill="var(--ground-3)" stroke="var(--rule-2)" />
                  <Label x={27} y={13} anchor="middle" tone={3} size={7.5}>Sub 1 (8d)</Label>
                  <rect x={60} y={0} width={55} height={20} rx={2} fill="var(--ground-3)" stroke="var(--rule-2)" />
                  <Label x={87} y={13} anchor="middle" tone={3} size={7.5}>Sub 2 (8d)</Label>
                  <Label x={130} y={13} tone={3} size={8}>...</Label>
                  <rect x={150} y={0} width={60} height={20} rx={2} fill="var(--ground-3)" stroke="var(--rule-2)" />
                  <Label x={180} y={13} anchor="middle" tone={3} size={7.5}>Sub 96 (8d)</Label>
                </g>
              </g>

              <Arrow x1={125} y1={95} x2={125} y2={115} kind="compute" width={1.2} />

              {/* Quantized to 1 byte per sub-vector */}
              <g transform="translate(0, 118)">
                <rect x={0} y={0} width={250} height={26} rx={2} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={1} />
                <Label x={125} y={17} anchor="middle" tone={1} size={8.5} kind="token" bold>
                  Quantized: 96 Bytes (1 byte per sub-vector)
                </Label>
              </g>
            </g>
          </g>
        )}

        {/* Right Side: The Pareto Trade-Off Landscape */}
        <g transform="translate(325, 25)">
          <rect x={0} y={0} width={285} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={16} y={22} tone={1} size={11} bold>Recall vs Latency Pareto Frontier</Label>
          <Label x={16} y={38} tone={3} size={9}>1,000,000 vectors (768 dimensions)</Label>

          {/* Chart Axes */}
          <g transform="translate(45, 55)">
            <line x1={0} y1={130} x2={220} y2={130} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={0} y1={0} x2={0} y2={130} stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={110} y={148} anchor="middle" tone={3} size={8.5}>Query Latency (ms) → Lower is better</Label>
            <Label x={-10} y={10} anchor="end" tone={3} size={8}>Recall (%)</Label>

            {/* Y Ticks */}
            <Label x={-5} y={15} anchor="end" tone={3} size={7}>100%</Label>
            <Label x={-5} y={65} anchor="end" tone={3} size={7}>90%</Label>
            <Label x={-5} y={115} anchor="end" tone={3} size={7}>80%</Label>

            {/* Plot Flat Point (top right) */}
            <circle cx={200} cy={15} r={5} fill="var(--compute)" />
            <Label x={195} y={28} anchor="end" tone={3} size={8}>Flat (Exact, 61ms)</Label>

            {/* Plot IVF Curve (nprobe 1 to 32) */}
            <path d="M 25 110 Q 55 50 110 25" fill="none" stroke="var(--memory)" strokeWidth={1.4} strokeDasharray="3 3" />
            <circle cx={65} cy={45} r={5} fill="var(--memory)" />
            <Label x={72} y={42} tone={2} size={8} kind="memory">IVF (nprobe={nprobe})</Label>

            {/* Plot HNSW Point (top left sweet spot) */}
            <circle cx={25} cy={20} r={6} fill="var(--token)" stroke="var(--text)" strokeWidth={1.2} />
            <Label x={35} y={22} tone={1} size={8.5} bold kind="token">HNSW (1.2ms, 98%)</Label>

            {/* Plot PQ Point (fastest, lower recall) */}
            <circle cx={15} cy={75} r={5} fill="var(--idle)" />
            <Label x={25} y={80} tone={3} size={8}>IVF+PQ (0.8ms, 88%)</Label>
          </g>
        </g>

        {/* Bottom Takeaway */}
        <g transform="translate(30, 275)">
          <rect x={0} y={0} width={580} height={45} rx={5} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={15} y={18} tone={1} size={10} bold>Production Rule of Thumb:</Label>
          <Label x={15} y={32} tone={2} size={9.5}>
            Use <tspan fill="var(--token)" fontWeight="600">HNSW</tspan> for strict sub-2ms latency when RAM allows (4 GB / 1M docs). Use <tspan fill="var(--idle)" fontWeight="600">IVF-PQ</tspan> when scaling to 10M+ vectors on limited memory.
          </Label>
        </g>
      </svg>
    </Stage>
  );
}
