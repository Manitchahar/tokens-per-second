import { useRef, useState } from 'react';
import { Stage, Slider, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, COLOR } from '../lib/svg';

const STEPS = [
  'The four-stage model selection funnel',
  'Stage 1: Hard constraints filter first',
  'Stage 2: Public benchmark shortlisting',
  'Stage 3: Your proprietary golden evaluation',
  'Build vs Buy: architectural trade-off matrix',
  'The Build vs Buy break-even economics',
];

const CAPTIONS = [
  'Model selection is not a subjective choice: it is a structured four-stage funnel that eliminates unqualified models before spending engineering time on fine-tuning or evaluation.',
  'Hard constraints filter first: commercial licenses, HIPAA/GDPR privacy, maximum latency budget, and context window eliminate 80% of candidate models in minutes.',
  'Public benchmarks serve only to shortlist: discard models that fail basic literacy or reasoning thresholds (e.g. MMLU < 75%), leaving 3 to 4 viable candidates.',
  'Your custom golden dataset makes the final call: score shortlisted candidates against your own prompt rubrics, custom schemas, and representative slices.',
  'Build vs Buy trade-offs: Managed APIs offer instant setup and zero GPU maintenance; self-hosted open weights offer data sovereignty, custom logits/LoRA, and zero drift.',
  'Break-even economics: Fixed GPU leases have high base cost but low marginal token cost. At high request volumes, self-hosting beats commercial APIs by 5x to 10x.',
];

export default function ModelSelectionFunnelViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [monthlyTokensMillion, setMonthlyTokensMillion] = useState(500);
  const [modelTier, setModelTier] = useState<'8b' | '70b'>('8b');

  // Economics arithmetic:
  // 8B tier:
  // Leased GPU (1x L4 or RTX 4090 / A10G): ~$0.80/hr = ~$576/month
  // Commercial API 8B: $0.30 per 1M tokens
  // 70B tier:
  // Leased GPU (4x A100 or 2x H100): ~$6.00/hr = ~$4,320/month
  // Commercial API 70B: $1.80 per 1M tokens

  const gpuMonthlyCost = modelTier === '8b' ? 576 : 4320;
  const apiRatePerMillion = modelTier === '8b' ? 0.30 : 1.80;
  const apiMonthlyCost = Math.round(monthlyTokensMillion * apiRatePerMillion);

  const breakEvenMillions = Math.round(gpuMonthlyCost / apiRatePerMillion);
  const selfHostCheaper = monthlyTokensMillion >= breakEvenMillions;
  const costDelta = Math.abs(apiMonthlyCost - gpuMonthlyCost);

  return (
    <Stage
      rootRef={ref}
      title="Model Selection: The Funnel and Build vs Buy Economics"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'error', 'memory', 'compute']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Volume', value: `${monthlyTokensMillion}M tok/mo`, kind: 'text' },
          { label: 'Managed API Cost', value: `$${apiMonthlyCost.toLocaleString()}/mo`, kind: selfHostCheaper ? 'compute' : 'token', note: `$${apiRatePerMillion.toFixed(2)} / 1M` },
          { label: 'Self-Hosted GPU', value: `$${gpuMonthlyCost.toLocaleString()}/mo`, kind: selfHostCheaper ? 'token' : 'error', note: 'fixed dedicated instance lease' },
          { label: 'Optimal Architecture', value: selfHostCheaper ? `Self-Host (Save $${costDelta.toLocaleString()})` : `API (Save $${costDelta.toLocaleString()})`, kind: selfHostCheaper ? 'token' : 'memory' },
        ]} />
      }
      controls={
        s.step >= 4 ? (
          <>
            <Segmented
              label="Model weight class"
              value={modelTier}
              options={[
                { value: '8b', label: '8B Tier (Small/Fast)' },
                { value: '70b', label: '70B Tier (Frontier Reasoning)' },
              ]}
              onChange={(val) => setModelTier(val as any)}
            />
            {s.step === 5 && (
              <Slider
                label="Monthly generation volume (Millions of Tokens)"
                value={monthlyTokensMillion}
                min={50}
                max={4000}
                onChange={setMonthlyTokensMillion}
                format={(v) => `${v}M tokens`}
                kind="token"
              />
            )}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Visual representation of the model selection funnel, criteria filtering, and build versus buy economic break-even curve.">
        {/* Step 0: Funnel overview */}
        {s.step === 0 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>The 4-Stage Model Selection Funnel</Label>

            {/* Funnel 1 */}
            <path d="M 40 54 L 600 54 L 540 104 L 100 104 Z" fill="var(--surface-2)" stroke={COLOR.error} strokeWidth={1.5} />
            <Label x={320} y={78} kind="error" size={11} bold anchor="middle">Stage 1: Hard Constraints Gate (100+ Models → 8)</Label>
            <text x={320} y={94} textAnchor="middle" fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">
              Commercial License • Data Residency • Latency SLO • Context Window
            </text>

            {/* Funnel 2 */}
            <path d="M 100 108 L 540 108 L 480 158 L 160 158 Z" fill="var(--surface-2)" stroke={COLOR.compute} strokeWidth={1.5} />
            <Label x={320} y={132} kind="compute" size={11} bold anchor="middle">Stage 2: Public Benchmark Screen (8 Models → 3-4)</Label>
            <text x={320} y={148} textAnchor="middle" fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">
              Coarse capability filter: MMLU, GSM8K, HumanEval
            </text>

            {/* Funnel 3 */}
            <path d="M 160 162 L 480 162 L 420 212 L 220 212 Z" fill="var(--surface-2)" stroke={COLOR.memory} strokeWidth={1.5} />
            <Label x={320} y={186} kind="memory" size={11} bold anchor="middle">Stage 3: Custom Application Eval (3-4 Models → 1-2)</Label>
            <text x={320} y={202} textAnchor="middle" fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">
              Proprietary golden dataset • Task rubrics • Segmented slices
            </text>

            {/* Funnel 4 */}
            <path d="M 220 216 L 420 216 L 370 266 L 270 266 Z" fill="var(--surface-2)" stroke={COLOR.token} strokeWidth={1.5} />
            <Label x={320} y={240} kind="token" size={11} bold anchor="middle">Stage 4: Canary Deployment</Label>
            <text x={320} y={256} textAnchor="middle" fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">
              Shadow traffic • Production monitoring • User telemetry
            </text>
          </g>
        )}

        {/* Step 1: Hard Constraints */}
        {s.step === 1 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Stage 1: Hard Constraints (Non-Negotiable Filters)</Label>

            {/* 4 Cards */}
            <rect x={24} y={54} width={138} height={160} rx={4} fill="var(--surface-2)" stroke={COLOR.error} />
            <Label x={93} y={76} kind="error" size={10} bold anchor="middle">1. License & IP</Label>
            <text x={34} y={100} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Commercial use allowed?</text>
            <text x={34} y={118} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Apache 2.0 / MIT</text>
            <text x={34} y={136} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Non-compete clauses?</text>
            <text x={34} y={154} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Training data terms</text>
            <Label x={93} y={196} tone={3} size={8} anchor="middle">Pass/Fail Boolean</Label>

            <rect x={175} y={54} width={138} height={160} rx={4} fill="var(--surface-2)" stroke={COLOR.error} />
            <Label x={244} y={76} kind="error" size={10} bold anchor="middle">2. Data Privacy</Label>
            <text x={185} y={100} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• HIPAA / GDPR rules</text>
            <text x={185} y={118} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Zero data retention</text>
            <text x={185} y={136} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• On-prem / VPC required?</text>
            <text x={185} y={154} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Audit log custody</text>
            <Label x={244} y={196} tone={3} size={8} anchor="middle">Pass/Fail Boolean</Label>

            <rect x={326} y={54} width={138} height={160} rx={4} fill="var(--surface-2)" stroke={COLOR.error} />
            <Label x={395} y={76} kind="error" size={10} bold anchor="middle">3. Latency SLO</Label>
            <text x={336} y={100} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• TTFT &lt; 400ms target</text>
            <text x={336} y={118} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• TPOT &gt; 50 tok/s</text>
            <text x={336} y={136} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• P99 tail SLA</text>
            <text x={336} y={154} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Streaming support</text>
            <Label x={395} y={196} tone={3} size={8} anchor="middle">Numeric Threshold</Label>

            <rect x={477} y={54} width={138} height={160} rx={4} fill="var(--surface-2)" stroke={COLOR.error} />
            <Label x={546} y={76} kind="error" size={10} bold anchor="middle">4. Context Length</Label>
            <text x={487} y={100} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• 32K vs 128K window</text>
            <text x={487} y={118} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Document retrieval size</text>
            <text x={487} y={136} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• KV memory overhead</text>
            <text x={487} y={154} fill="var(--text-2)" fontSize={8.5} fontFamily="var(--font-sans)">• Long-context recall</text>
            <Label x={546} y={196} tone={3} size={8} anchor="middle">Numeric Threshold</Label>

            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={2} size={10.5}>
              Never evaluate a model on your custom dataset until it clears all four hard constraints. It saves weeks of wasted benchmarking.
            </Label>
          </g>
        )}

        {/* Step 2: Stage 2 Benchmark screening */}
        {s.step === 2 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Stage 2: Public Benchmark Screening (Coarse Filter)</Label>

            <rect x={24} y={54} width={592} height={160} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={78} tone={1} size={11} bold>Shortlist Gate Matrix (Example Evaluation Candidates):</Label>

            {/* Model A */}
            <Label x={40} y={108} tone={2} size={10} mono>Candidate 1 (Frontier 70B):</Label>
            <Span x={220} y={96} w={200} h={18} kind="token" label="MMLU: 86% | HumanEval: 78%" />
            <Span x={430} y={96} w={80} h={18} kind="token" label="PASS" />

            {/* Model B */}
            <Label x={40} y={138} tone={2} size={10} mono>Candidate 2 (Efficient 8B):</Label>
            <Span x={220} y={126} w={180} h={18} kind="token" label="MMLU: 79% | HumanEval: 68%" />
            <Span x={430} y={126} w={80} h={18} kind="token" label="PASS" />

            {/* Model C */}
            <Label x={40} y={168} tone={2} size={10} mono>Candidate 3 (Legacy 7B):</Label>
            <Span x={220} y={156} w={120} h={18} kind="error" label="MMLU: 58% | HumanEval: 34%" />
            <Span x={430} y={156} w={80} h={18} kind="error" label="DISCARD" />

            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={2} size={10.5}>
              Stage 2 discards Candidate 3 without running costly internal evaluations. Only Candidates 1 and 2 proceed to Stage 3.
            </Label>
          </g>
        )}

        {/* Step 3: Custom Domain Evaluation */}
        {s.step === 3 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Stage 3: Proprietary Golden Evaluation (The Decisive Gate)</Label>

            <rect x={24} y={54} width={592} height={160} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={78} tone={1} size={11} bold>Custom Scorecard over Your Company's 200 Golden Test Cases:</Label>

            {/* Row 1 */}
            <Label x={40} y={108} tone={2} size={10} mono>JSON Schema Compliance:</Label>
            <Span x={220} y={96} w={140} h={18} kind="token" label="Cand 1: 99.5%" />
            <Span x={380} y={96} w={140} h={18} kind="token" label="Cand 2: 98.0%" />

            {/* Row 2 */}
            <Label x={40} y={138} tone={2} size={10} mono>Domain Extraction Accuracy:</Label>
            <Span x={220} y={126} w={140} h={18} kind="token" label="Cand 1: 91.0%" />
            <Span x={380} y={126} w={120} h={18} kind="compute" label="Cand 2: 86.5%" />

            {/* Row 3 */}
            <Label x={40} y={168} tone={2} size={10} mono>P95 TTFT Latency:</Label>
            <Span x={220} y={156} w={100} h={18} kind="error" label="Cand 1: 650 ms" />
            <Span x={380} y={156} w={140} h={18} kind="token" label="Cand 2: 120 ms" />

            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={2} size={10.5}>
              Notice: Candidate 1 has +4.5% domain accuracy, but Candidate 2 is 5x faster and 8x cheaper. This leads directly to the Build vs Buy decision.
            </Label>
          </g>
        )}

        {/* Step 4: Build vs Buy Matrix */}
        {s.step === 4 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Build vs Buy: Architectural Trade-Off Matrix</Label>

            <rect x={24} y={54} width={592} height={170} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            
            {/* Headers */}
            <Label x={40} y={76} tone={1} size={10.5} bold>Architectural Dimension</Label>
            <Label x={220} y={76} kind="memory" size={10.5} bold>Commercial Cloud API (Buy)</Label>
            <Label x={420} y={76} kind="token" size={10.5} bold>Self-Hosted Open Weights (Build)</Label>
            <line x1={40} y1={86} x2={600} y2={86} stroke="var(--rule-2)" strokeWidth={1} />

            {/* Dimensions */}
            <Label x={40} y={106} tone={2} size={9.5}>Data Privacy & Sovereignty</Label>
            <Label x={220} y={106} tone={3} size={9.5}>3rd-party trust (ZDR promise)</Label>
            <Label x={420} y={106} kind="token" size={9.5} bold>100% On-Prem / VPC Custody</Label>

            <Label x={40} y={128} tone={2} size={9.5}>Latency Guarantees</Label>
            <Label x={220} y={128} kind="error" size={9.5}>Multi-tenant queue spikes</Label>
            <Label x={420} y={128} kind="token" size={9.5} bold>Dedicated GPU (Guaranteed SLO)</Label>

            <Label x={40} y={150} tone={2} size={9.5}>Engine Control (Logits/LoRA)</Label>
            <Label x={220} y={150} tone={3} size={9.5}>Black-box JSON endpoint</Label>
            <Label x={420} y={150} kind="token" size={9.5} bold>Custom CUDA, LoRA, Logits</Label>

            <Label x={40} y={172} tone={2} size={9.5}>Model Deprecation & Drift</Label>
            <Label x={220} y={172} kind="error" size={9.5}>Silently updated or retired</Label>
            <Label x={420} y={172} kind="token" size={9.5} bold>Pinned, immutable Safetensors</Label>

            <Label x={40} y={194} tone={2} size={9.5}>Setup & Ops Complexity</Label>
            <Label x={220} y={194} kind="memory" size={9.5} bold>Zero setup (API key)</Label>
            <Label x={420} y={194} kind="compute" size={9.5}>Requires GPU DevOps (vLLM)</Label>

            <rect x={24} y={236} width={592} height={44} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={320} y={262} anchor="middle" tone={1} size={10.5}>
              Rule: Start on managed APIs for rapid iteration. Migrate to self-hosted open weights when volume explodes or privacy is non-negotiable.
            </Label>
          </g>
        )}

        {/* Step 5: Break-Even Economics */}
        {s.step === 5 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>The Break-Even Economics: Fixed GPU Lease vs Variable Token Price</Label>

            {/* Coordinate chart */}
            <rect x={24} y={50} width={420} height={220} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <line x1={60} y1={230} x2={420} y2={230} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={60} y1={80} x2={60} y2={230} stroke="var(--rule-2)" strokeWidth={1} />

            <Label x={45} y={85} tone={3} size={9} anchor="end">High</Label>
            <Label x={45} y={230} tone={3} size={9} anchor="end">$0</Label>
            <Label x={60} y={246} tone={3} size={8.5}>0M</Label>
            <Label x={235} y={246} tone={3} size={8.5} anchor="middle">{breakEvenMillions}M (Break-even)</Label>
            <Label x={410} y={246} tone={3} size={8.5} anchor="end">4000M</Label>

            {/* Flat GPU lease line */}
            <line x1={60} y1={170} x2={420} y2={170} stroke={COLOR.token} strokeWidth={2} />
            <Label x={350} y={162} kind="token" size={9.5} bold>Self-Host GPU (${gpuMonthlyCost}/mo)</Label>

            {/* Sloping API cost line */}
            <line x1={60} y1={230} x2={420} y2={70} stroke={COLOR.compute} strokeWidth={2} strokeDasharray="4 3" />
            <Label x={340} y={64} kind="compute" size={9.5} bold>Managed API (Variable)</Label>

            {/* Break-even point circle */}
            <circle cx={235} cy={170} r={5} fill={COLOR.error} stroke="#fff" strokeWidth={1.5} />
            <Label x={235} y={192} kind="error" size={9.5} bold anchor="middle">Cross-over Point</Label>

            {/* Right side stats card */}
            <rect x={460} y={50} width={156} height={220} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={472} y={74} tone={1} size={11} bold>Tier: {modelTier.toUpperCase()}</Label>
            <text x={472} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Break-even Volume:</text>
            <text x={472} y={118} fill={COLOR.token} fontSize={11} fontFamily="var(--font-mono)">{breakEvenMillions}M tok/mo</text>

            <text x={472} y={146} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Current Volume:</text>
            <text x={472} y={164} fill="var(--text)" fontSize={10.5} fontFamily="var(--font-mono)">{monthlyTokensMillion}M tok/mo</text>

            <text x={472} y={192} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Recommendation:</text>
            <Label x={472} y={212} kind={selfHostCheaper ? 'token' : 'compute'} size={10} bold>
              {selfHostCheaper ? 'Deploy Self-Hosted' : 'Use Managed API'}
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
