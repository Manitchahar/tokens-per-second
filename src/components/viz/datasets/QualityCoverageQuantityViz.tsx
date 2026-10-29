import { useRef, useState, useEffect } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, COLOR, DIM, fmt } from '../lib/svg';

const STEPS = [
  'Quantity alone fails',
  'LIMA: 1,000 curated gems',
  'Logarithmic scaling curves',
  'Domain mix: math & code',
  'Teaching behaviors: CoT & tools',
  'The pilot playbook: 50–100 rows',
];

const CAPTIONS = [
  'Training on 50,000 noisy, unvetted instructions yields hallucinations and poor formatting. Raw token volume cannot substitute for clarity.',
  'LIMA showed that just 1,000 hand-crafted examples on a strong pretrained base (LLaMA-65B) matched or beat models trained on 50× more data.',
  'Performance scales with log(N). After roughly 1,000 high-quality samples, returns flatten unless fresh task coverage is introduced.',
  'Including 20–25% code and math improves step-by-step reasoning on general English tasks, not just programming.',
  'A few hundred step-by-step Chain-of-Thought (CoT) and tool-calling examples teach the model structured behavioral execution.',
  'Always begin with 50 to 100 diverse, pristine examples to measure feasibility before scaling budget and pipeline complexity.',
];

export default function QualityCoverageQuantityViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [samples, setSamples] = useState(1000);
  const [qualityMode, setQualityMode] = useState<'noisy' | 'curated'>('curated');

  // Sync step changes to reasonable defaults
  useEffect(() => {
    if (s.step === 0) {
      setSamples(50000);
      setQualityMode('noisy');
    } else if (s.step === 1) {
      setSamples(1000);
      setQualityMode('curated');
    } else if (s.step === 2) {
      setSamples(2500);
      setQualityMode('curated');
    } else if (s.step === 3) {
      setSamples(1000);
      setQualityMode('curated');
    } else if (s.step === 4) {
      setSamples(1200);
      setQualityMode('curated');
    } else if (s.step === 5) {
      setSamples(100);
      setQualityMode('curated');
    }
  }, [s.step]);

  const animSamples = useTween(samples, 400);

  // Model win rate approximation based on sample size and quality
  const qualityFactor = qualityMode === 'curated' ? 1.0 : 0.45;
  const winRate = Math.min(
    92,
    Math.round(20 + qualityFactor * 22 * Math.log10(Math.max(10, animSamples)))
  );

  const estCost =
    qualityMode === 'curated'
      ? Math.round(samples * 1.5) // $1.50 per manual curated sample
      : Math.round(samples * 0.04); // $0.04 per noisy scraped/raw sample

  // Domain mix bars (step 3 & 4)
  const domains = [
    { name: 'Coding & Syntax', pct: 25, kind: 'compute' as const },
    { name: 'Math & Logic', pct: 20, kind: 'comm' as const },
    { name: 'General Reasoning', pct: 35, kind: 'token' as const },
    { name: 'Multi-turn Dialog', pct: 10, kind: 'memory' as const },
    { name: 'Safety & Refusal', pct: 10, kind: 'error' as const },
  ];

  return (
    <Stage
      rootRef={ref}
      title="Data Triad: Quality, Coverage & Quantity"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'compute', 'memory', 'error']}
      readout={
        <Readout
          items={[
            {
              label: 'Dataset Size (N)',
              value: `${fmt.int(Math.round(animSamples))} rows`,
              kind: 'token',
              note: qualityMode === 'curated' ? 'Curated / high-touch' : 'Noisy / web-scraped',
            },
            {
              label: 'Model Win Rate',
              value: `${winRate}%`,
              kind: winRate > 70 ? 'token' : 'error',
              note: 'vs base unaligned model on human evaluation',
            },
            {
              label: 'Estimated Cost',
              value: `$${estCost}`,
              kind: 'memory',
              note: qualityMode === 'curated' ? '~$1.50 per curated row' : '~$0.04 per scraped row',
            },
          ]}
        />
      }
      controls={
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', width: '100%', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <Slider
              label="Examples (N)"
              value={samples}
              min={50}
              max={50000}
              step={50}
              onChange={setSamples}
            />
          </div>
          <Segmented
            value={qualityMode}
            onChange={(v) => setQualityMode(v as 'noisy' | 'curated')}
            options={[
              { value: 'curated', label: 'Curated (LIMA-style)' },
              { value: 'noisy', label: 'Noisy Web Data' },
            ]}
          />
        </div>
      }
      {...s}
    >
      <svg viewBox="0 0 640 300" role="img" aria-label="Visualizing dataset quality, coverage, and quantity trade-offs">
        {/* Background gridlines */}
        <line x1={50} y1={250} x2={590} y2={250} stroke="var(--rule-2)" strokeWidth={1} />
        <line x1={50} y1={50} x2={50} y2={250} stroke="var(--rule-2)" strokeWidth={1} />

        {s.step !== 3 && s.step !== 4 && (
          <g>
            {/* Axis Labels */}
            <Label x={50} y={42} anchor="start" size={10} tone={2}>
              Model Win Rate (%)
            </Label>
            <Label x={590} y={268} anchor="end" size={10} tone={2}>
              Dataset Size N (log scale: 50 → 50,000)
            </Label>

            {/* Y ticks */}
            {[20, 40, 60, 80].map((val) => {
              const y = 250 - (val / 100) * 200;
              return (
                <g key={val}>
                  <line x1={46} y1={y} x2={590} y2={y} stroke="var(--rule-1)" strokeDasharray="3 3" />
                  <Label x={42} y={y + 3} anchor="end" size={9} tone={3}>
                    {val}%
                  </Label>
                </g>
              );
            })}

            {/* X ticks */}
            {[100, 1000, 10000, 50000].map((val) => {
              const minLog = Math.log10(50);
              const maxLog = Math.log10(50000);
              const x = 50 + ((Math.log10(val) - minLog) / (maxLog - minLog)) * 540;
              return (
                <g key={val}>
                  <line x1={x} y1={250} x2={x} y2={254} stroke="var(--rule-2)" />
                  <Label x={x} y={264} anchor="middle" size={9} tone={3}>
                    {val >= 1000 ? `${val / 1000}k` : val}
                  </Label>
                </g>
              );
            })}

            {/* Benchmark Curve 1: Curated (Green) */}
            <path
              d="M 50 178 Q 120 120, 240 85 T 590 62"
              fill="none"
              stroke={COLOR.token}
              strokeWidth={qualityMode === 'curated' ? 2.5 : 1.2}
              opacity={qualityMode === 'curated' ? 1 : 0.4}
            />
            <Label x={540} y={55} anchor="end" size={10} tone={1} style={{ fill: COLOR.token }}>
              High-Quality Curated (LIMA)
            </Label>

            {/* Benchmark Curve 2: Noisy (Red) */}
            <path
              d="M 50 220 Q 180 200, 360 170 T 590 148"
              fill="none"
              stroke={COLOR.error}
              strokeWidth={qualityMode === 'noisy' ? 2.5 : 1.2}
              opacity={qualityMode === 'noisy' ? 1 : 0.4}
            />
            <Label x={540} y={165} anchor="end" size={10} tone={2} style={{ fill: COLOR.error }}>
              Noisy / Unfiltered Crawl
            </Label>

            {/* Current Operating Point Marker */}
            {(() => {
              const minLog = Math.log10(50);
              const maxLog = Math.log10(50000);
              const ptX = 50 + ((Math.log10(Math.max(50, animSamples)) - minLog) / (maxLog - minLog)) * 540;
              const ptY = 250 - (winRate / 100) * 200;
              return (
                <g>
                  <circle cx={ptX} cy={ptY} r={6} fill={qualityMode === 'curated' ? COLOR.token : COLOR.error} />
                  <circle cx={ptX} cy={ptY} r={10} fill="none" stroke="white" strokeWidth={1.5} opacity={0.6} />
                  <line x1={ptX} y1={ptY} x2={ptX} y2={250} stroke="var(--rule-2)" strokeDasharray="2 2" />
                  <line x1={50} y1={ptY} x2={ptX} y2={ptY} stroke="var(--rule-2)" strokeDasharray="2 2" />
                  <rect x={ptX - 40} y={ptY - 24} width={80} height={18} rx={3} fill="#141820" stroke="var(--rule-2)" />
                  <Label x={ptX} y={ptY - 12} anchor="middle" size={10} bold tone={1}>
                    {winRate}% win rate
                  </Label>
                </g>
              );
            })()}

            {/* LIMA highlight box at step 1 */}
            {s.step === 1 && (
              <g>
                <rect x={220} y={68} width={90} height={40} rx={4} fill={DIM.token} stroke={COLOR.token} />
                <Label x={265} y={84} anchor="middle" size={10} bold style={{ fill: COLOR.token }}>
                  LIMA Spot
                </Label>
                <Label x={265} y={98} anchor="middle" size={9} tone={2}>
                  1,000 samples
                </Label>
              </g>
            )}

            {/* Pilot study highlight at step 5 */}
            {s.step === 5 && (
              <g>
                <rect x={70} y={150} width={95} height={42} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
                <Label x={117} y={166} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
                  Pilot Study
                </Label>
                <Label x={117} y={180} anchor="middle" size={9} tone={2}>
                  50–100 samples
                </Label>
              </g>
            )}
          </g>
        )}

        {/* Steps 3 & 4: Domain Composition & Behavioral Data View */}
        {(s.step === 3 || s.step === 4) && (
          <g>
            <Label x={320} y={40} anchor="middle" size={13} bold tone={1}>
              {s.step === 3 ? 'Balanced Domain Mix: Why Math & Code Matter' : 'Teaching Behaviors: CoT & Tool Invocations'}
            </Label>

            {/* Domain breakdown bars */}
            {domains.map((d, idx) => {
              const barY = 70 + idx * 36;
              const barWidth = d.pct * 4;
              return (
                <g key={d.name}>
                  <Label x={155} y={barY + 14} anchor="end" size={11} tone={2}>
                    {d.name}
                  </Label>
                  <Span
                    x={165}
                    y={barY}
                    w={barWidth}
                    h={20}
                    kind={d.kind}
                    label={`${d.pct}%`}
                  />
                  <Label x={175 + barWidth} y={barY + 14} anchor="start" size={10} tone={3}>
                    {idx === 0 && 'Builds syntax discipline & structured logic'}
                    {idx === 1 && 'Enforces step-by-step state tracking'}
                    {idx === 2 && 'Instruction following & knowledge retrieval'}
                    {idx === 3 && 'Context persistence across conversational turns'}
                    {idx === 4 && 'Refusing unsafe or impossible instructions'}
                  </Label>
                </g>
              );
            })}

            {s.step === 4 && (
              <g>
                <rect x={70} y={250} width={500} height={35} rx={4} fill="#141820" stroke={COLOR.compute} />
                <Label x={320} y={272} anchor="middle" size={10} bold style={{ fill: COLOR.compute }}>
                  Behaviors transfer across domains: A model trained on 300 CoT traces reasons better on law and medicine.
                </Label>
              </g>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
