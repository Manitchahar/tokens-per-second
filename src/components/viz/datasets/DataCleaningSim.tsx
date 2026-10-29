import { useState } from 'react';
import { Span, Label, COLOR, DIM, fmt } from '../lib/svg';

export default function DataCleaningSim() {
  const [exactDedup, setExactDedup] = useState(true);
  const [fuzzyDedup, setFuzzyDedup] = useState(true);
  const [piiFilter, setPiiFilter] = useState(true);
  const [lengthFilter, setLengthFilter] = useState(true);
  const [chatFormat, setChatFormat] = useState(true);

  const rawCount = 5000;
  const exactDuplicates = 450;
  const fuzzyDuplicates = 620;
  const lengthOutliers = 180;
  const piiInstances = 145;

  let currentCount = rawCount;
  if (exactDedup) currentCount -= exactDuplicates;
  if (fuzzyDedup) currentCount -= fuzzyDuplicates;
  if (lengthFilter) currentCount -= lengthOutliers;

  const totalRemoved = rawCount - currentCount;
  const retentionPct = ((currentCount / rawCount) * 100).toFixed(1);

  // Token distribution buckets for surviving rows
  const buckets = [
    { range: '0–64', count: lengthFilter ? 40 : 120, avgLen: 48 },
    { range: '65–128', count: 850, avgLen: 96 },
    { range: '129–256', count: 1420, avgLen: 190 },
    { range: '257–512', count: 980, avgLen: 380 },
    { range: '513–1024', count: 380, avgLen: 720 },
    { range: '1025–2048', count: 80, avgLen: 1450 },
  ];

  return (
    <div style={{ background: '#0e1116', border: '1px solid var(--rule-2)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h3 style={{ margin: 0, fontSize: 16, color: 'var(--text)' }}>Interactive Dataset Cleaning Simulator</h3>
          <p style={{ margin: '4px 0 0', fontSize: 12, color: 'var(--text-3)' }}>
            Simulate the 5-stage cleaning pipeline on 5,000 raw instruction pairs.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <label style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', color: 'var(--text-2)' }}>
            <input type="checkbox" checked={exactDedup} onChange={(e) => setExactDedup(e.target.checked)} />
            Exact SHA-256
          </label>
          <label style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', color: 'var(--text-2)' }}>
            <input type="checkbox" checked={fuzzyDedup} onChange={(e) => setFuzzyDedup(e.target.checked)} />
            MinHash LSH (0.7)
          </label>
          <label style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', color: 'var(--text-2)' }}>
            <input type="checkbox" checked={lengthFilter} onChange={(e) => setLengthFilter(e.target.checked)} />
            Length Filters
          </label>
          <label style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', color: 'var(--text-2)' }}>
            <input type="checkbox" checked={piiFilter} onChange={(e) => setPiiFilter(e.target.checked)} />
            PII Redaction
          </label>
          <label style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', color: 'var(--text-2)' }}>
            <input type="checkbox" checked={chatFormat} onChange={(e) => setChatFormat(e.target.checked)} />
            ChatML Template
          </label>
        </div>
      </div>

      {/* Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 12, marginBottom: 20 }}>
        <div style={{ background: '#141820', padding: 10, borderRadius: 4, border: '1px solid var(--rule-1)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-3)' }}>RAW INPUT</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text)' }}>5,000 rows</div>
        </div>
        <div style={{ background: '#141820', padding: 10, borderRadius: 4, border: '1px solid var(--rule-1)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-3)' }}>PRUNED ROWS</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: COLOR.error }}>-{fmt.int(totalRemoved)} rows</div>
        </div>
        <div style={{ background: '#141820', padding: 10, borderRadius: 4, border: '1px solid var(--rule-1)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-3)' }}>SURVIVING SET</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: COLOR.token }}>{fmt.int(currentCount)} rows</div>
        </div>
        <div style={{ background: '#141820', padding: 10, borderRadius: 4, border: '1px solid var(--rule-1)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-3)' }}>RETENTION RATE</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: COLOR.compute }}>{retentionPct}%</div>
        </div>
        <div style={{ background: '#141820', padding: 10, borderRadius: 4, border: '1px solid var(--rule-1)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-3)' }}>PII REDACTED</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: COLOR.memory }}>
            {piiFilter ? `${piiInstances} instances` : '0 (Raw PII)'}
          </div>
        </div>
      </div>

      {/* Token Distribution Histogram SVG */}
      <svg viewBox="0 0 640 220" role="img" aria-label="Visualizing dataset token length histogram">
        <rect x={0} y={0} width={640} height={220} fill="#141820" rx={4} />
        <line x1={50} y1={170} x2={600} y2={170} stroke="var(--rule-2)" strokeWidth={1} />

        <Label x={50} y={25} anchor="start" size={11} bold tone={1}>
          Token Length Distribution (Chat-Templated Sequences)
        </Label>
        <Label x={600} y={25} anchor="end" size={9} tone={3}>
          {chatFormat ? 'Includes ChatML header tokens + EOS' : 'Raw untemplated text'}
        </Label>

        {/* Histogram Bars */}
        {buckets.map((b, i) => {
          const x = 70 + i * 85;
          const maxCount = 1500;
          const barH = (b.count / maxCount) * 120;
          const y = 170 - barH;

          return (
            <g key={b.range}>
              <rect
                x={x}
                y={y}
                width={55}
                height={barH}
                rx={2}
                fill={DIM.token}
                stroke={COLOR.token}
                strokeWidth={1}
              />
              <Label x={x + 27} y={y - 6} anchor="middle" size={9} bold tone={1}>
                {b.count}
              </Label>
              <Label x={x + 27} y={185} anchor="middle" size={8.5} tone={2}>
                {b.range}
              </Label>
              <Label x={x + 27} y={198} anchor="middle" size={7.5} tone={3}>
                tokens
              </Label>
            </g>
          );
        })}

        {/* Pipeline Summary Badge */}
        <g>
          <rect x={70} y={50} width={500} height={24} rx={3} fill="#0e1116" stroke="var(--rule-1)" />
          <Label x={80} y={66} anchor="start" size={8.5} tone={2}>
            Format: {chatFormat ? '<|im_start|>role...<|im_end|>' : 'Raw Plaintext'}
          </Label>
          <Label x={280} y={66} anchor="start" size={8.5} tone={2}>
            Loss Masking: {chatFormat ? 'Active (Target = -100 on prompt)' : 'None'}
          </Label>
          <Label x={560} y={66} anchor="end" size={8.5} bold style={{ fill: COLOR.token }}>
            Ready for SFT: {currentCount > 0 ? 'YES' : 'NO'}
          </Label>
        </g>
      </svg>
    </div>
  );
}
