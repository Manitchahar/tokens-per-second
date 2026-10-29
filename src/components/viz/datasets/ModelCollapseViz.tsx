import { useRef, useState, useEffect } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, COLOR, DIM, fmt } from '../lib/svg';

const STEPS = [
  'The 3-tier verification funnel',
  'Gen 0: Authentic human data',
  'Gen 1: Minor tail erosion',
  'Gen 3: Early collapse & mode loss',
  'Gen 5: Late collapse into gibberish',
  'The cure: Human anchor & tests',
];

const CAPTIONS = [
  'Raw synthetic data must pass three filters: fast heuristic rules, deterministic code/math execution, and LLM-as-a-judge quality rubrics.',
  'Generation 0 represents genuine human text: high semantic variance with wide, rich tails containing rare vocabulary and specialized domain knowledge.',
  'When Gen 1 trains on samples from Gen 0, low-probability tail events are under-sampled. Rare topics begin to disappear from the learned distribution.',
  'By Gen 3, early collapse sets in: variance shrinks dramatically, the distribution centers on common modes, and specialized edge-case capabilities are lost.',
  'By Gen 5, recursive synthetic training causes complete collapse: the model output degenerates into repetitive, low-perplexity nonsense.',
  'The cure is twofold: anchor every training run with 20–40% real human text, and enforce strict rejection sampling with functional code execution.',
];

export default function ModelCollapseViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [generation, setGeneration] = useState(0);
  const [hasAnchor, setHasAnchor] = useState(false);

  useEffect(() => {
    if (s.step === 0) {
      setGeneration(0);
      setHasAnchor(false);
    } else if (s.step === 1) {
      setGeneration(0);
      setHasAnchor(false);
    } else if (s.step === 2) {
      setGeneration(1);
      setHasAnchor(false);
    } else if (s.step === 3) {
      setGeneration(3);
      setHasAnchor(false);
    } else if (s.step === 4) {
      setGeneration(5);
      setHasAnchor(false);
    } else if (s.step === 5) {
      setGeneration(5);
      setHasAnchor(true);
    }
  }, [s.step]);

  const animGen = useTween(generation, 400);

  // Variance and tail metrics based on generation and human anchor
  // Shumailov formula: variance ~ sigma_0^2 * (1 - 1/M)^k
  const effectiveGen = hasAnchor ? Math.min(1.2, animGen * 0.25) : animGen;
  const variance = Math.max(0.08, 1.0 - effectiveGen * 0.18);
  const tailCoverage = Math.max(4, Math.round(100 * Math.pow(variance, 1.8)));

  // Bell curve points calculation
  const getCurvePath = () => {
    const points: [number, number][] = [];
    const mean = 320;
    const std = 110 * Math.sqrt(variance);
    const peakHeight = (1 / Math.sqrt(variance)) * 110;

    for (let x = 80; x <= 560; x += 10) {
      const z = (x - mean) / std;
      const y = 250 - peakHeight * Math.exp(-0.5 * z * z);
      points.push([x, Math.max(50, y)]);
    }

    return points.reduce((acc, [x, y], idx) => {
      return idx === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
    }, '');
  };

  return (
    <Stage
      rootRef={ref}
      title="Verification Funnel & The Mechanics of Model Collapse"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'compute', 'error', 'memory']}
      readout={
        <Readout
          items={[
            {
              label: 'Recursive Generation',
              value: `Gen ${Math.round(animGen)}`,
              kind: animGen > 2 && !hasAnchor ? 'error' : 'token',
              note: hasAnchor ? 'Anchored with 30% real data' : '100% recursive synthetic data',
            },
            {
              label: 'Distribution Variance',
              value: `${(variance * 100).toFixed(0)}%`,
              kind: variance > 0.5 ? 'token' : 'error',
              note: variance < 0.4 ? 'Mode collapse: tails eliminated' : 'Healthy semantic diversity',
            },
            {
              label: 'Tail Knowledge Coverage',
              value: `${tailCoverage}%`,
              kind: tailCoverage > 50 ? 'token' : 'error',
              note: 'Rare facts, specialized jargon, edge cases',
            },
            {
              label: 'Model State',
              value:
                s.step === 0
                  ? 'Verification Pipeline'
                  : hasAnchor
                  ? 'Stable Distribution'
                  : generation === 0
                  ? 'Pristine Base'
                  : generation <= 2
                  ? 'Mild Degradation'
                  : generation <= 4
                  ? 'Severe Mode Collapse'
                  : 'Degenerate Gibberish',
              kind: hasAnchor || generation <= 1 ? 'token' : 'error',
              note: hasAnchor ? 'Protected by human ground truth' : 'Compounding synthetic noise',
            },
          ]}
        />
      }
      controls={
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', width: '100%', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <Slider
              label="Generation (k)"
              value={generation}
              min={0}
              max={5}
              step={1}
              onChange={setGeneration}
            />
          </div>
          <Segmented
            value={hasAnchor ? 'anchored' : 'unanchored'}
            onChange={(v) => setHasAnchor(v === 'anchored')}
            options={[
              { value: 'unanchored', label: '100% Recursive Synthetic' },
              { value: 'anchored', label: 'Anchor: 30% Real Data' },
            ]}
          />
        </div>
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Visualizing model collapse and the verification funnel">
        {/* Step 0: Verification Funnel */}
        {s.step === 0 && (
          <g>
            <Label x={320} y={35} anchor="middle" size={13} bold tone={1}>
              The Three-Tier Verification Funnel
            </Label>

            {/* Level 1: Heuristics */}
            <rect x={70} y={65} width={500} height={45} rx={4} fill={DIM.compute} stroke={COLOR.compute} />
            <Label x={85} y={85} anchor="start" size={11} bold style={{ fill: COLOR.compute }}>
              Tier 1: Heuristic Filters (Fastest, O(1))
            </Label>
            <Label x={85} y={100} anchor="start" size={9} tone={2}>
              Regex syntax, token length bounds [10..2048], FastText language ID, zlib compression ratio check
            </Label>
            <Label x={550} y={92} anchor="end" size={10} bold tone={3}>
              Prunes 25%
            </Label>

            {/* Level 2: Functional Checks */}
            <rect x={110} y={125} width={420} height={45} rx={4} fill={DIM.token} stroke={COLOR.token} />
            <Label x={125} y={145} anchor="start" size={11} bold style={{ fill: COLOR.token }}>
              Tier 2: Functional Verification (Ground Truth)
            </Label>
            <Label x={125} y={160} anchor="start" size={9} tone={2}>
              Execute unit tests in sandbox (pytest), AST parsing, JSON schema validation, SymPy math solvers
            </Label>
            <Label x={510} y={152} anchor="end" size={10} bold tone={3}>
              Prunes 40%
            </Label>

            {/* Level 3: AI Judge */}
            <rect x={150} y={185} width={340} height={45} rx={4} fill={DIM.memory} stroke={COLOR.memory} />
            <Label x={165} y={205} anchor="start" size={11} bold style={{ fill: COLOR.memory }}>
              Tier 3: LLM-as-a-Judge (Rubric Scoring)
            </Label>
            <Label x={165} y={220} anchor="start" size={9} tone={2}>
              Frontier model rates adherence, factual consistency, and tone against strict criteria
            </Label>
            <Label x={470} y={212} anchor="end" size={10} bold tone={3}>
              Prunes 15%
            </Label>

            {/* Output */}
            <rect x={210} y={245} width={220} height={35} rx={3} fill="#141820" stroke="var(--rule-2)" />
            <Label x={320} y={267} anchor="middle" size={10} bold tone={1}>
              Clean Gold Training Set (20% of raw)
            </Label>
          </g>
        )}

        {/* Steps 1 - 5: Distribution Collapse Visualizer */}
        {s.step > 0 && (
          <g>
            {/* Axis */}
            <line x1={80} y1={250} x2={560} y2={250} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={320} y1={40} x2={320} y2={250} stroke="var(--rule-1)" strokeDasharray="3 3" />

            <Label x={320} y={32} anchor="middle" size={12} bold tone={1}>
              Semantic Distribution Shape (Generation {generation})
            </Label>
            <Label x={80} y={268} anchor="start" size={9} tone={3}>
              Rare Tails (Edge cases, deep jargon)
            </Label>
            <Label x={320} y={268} anchor="middle" size={9} tone={2}>
              Common Modes (Standard conversation)
            </Label>
            <Label x={560} y={268} anchor="end" size={9} tone={3}>
              Rare Tails (Complex logic, math)
            </Label>

            {/* Reference Original Distribution (Ghost curve) */}
            <path
              d="M 80 248 Q 200 240, 260 170 T 320 140 T 380 170 T 560 248"
              fill="none"
              stroke="var(--rule-2)"
              strokeWidth={1.5}
              strokeDasharray="4 4"
            />
            <Label x={480} y={190} anchor="start" size={9} tone={3}>
              Gen 0 Ground Truth
            </Label>

            {/* Collapsed/Active Curve */}
            <path
              d={getCurvePath()}
              fill="none"
              stroke={hasAnchor || generation <= 1 ? COLOR.token : generation <= 3 ? COLOR.compute : COLOR.error}
              strokeWidth={2.5}
            />

            {/* Tail warning markers */}
            {!hasAnchor && generation >= 3 && (
              <g>
                {/* Left tail erased */}
                <rect x={80} y={200} width={90} height={40} rx={3} fill={DIM.error} stroke={COLOR.error} />
                <Label x={125} y={218} anchor="middle" size={9} bold style={{ fill: COLOR.error }}>
                  Tails Erased
                </Label>
                <Label x={125} y={232} anchor="middle" size={8} tone={2}>
                  Zero probability
                </Label>

                {/* Right tail erased */}
                <rect x={470} y={200} width={90} height={40} rx={3} fill={DIM.error} stroke={COLOR.error} />
                <Label x={515} y={218} anchor="middle" size={9} bold style={{ fill: COLOR.error }}>
                  Logic Lost
                </Label>
                <Label x={515} y={232} anchor="middle" size={8} tone={2}>
                  Zero probability
                </Label>
              </g>
            )}

            {/* Step 4: Gibberish text sample */}
            {s.step === 4 && !hasAnchor && (
              <g>
                <rect x={160} y={70} width={320} height={50} rx={4} fill="#141820" stroke={COLOR.error} />
                <Label x={320} y={90} anchor="middle" size={9} bold style={{ fill: COLOR.error }}>
                  Degenerate Text Emission (Late Collapse):
                </Label>
                <Label x={320} y={106} anchor="middle" size={8.5} mono tone={2}>
                  &quot;The tower of the church of the church of the...&quot;
                </Label>
              </g>
            )}

            {/* Step 5: Cure explanation */}
            {s.step === 5 && hasAnchor && (
              <g>
                <rect x={140} y={60} width={360} height={55} rx={4} fill={DIM.token} stroke={COLOR.token} />
                <Label x={320} y={80} anchor="middle" size={10} bold style={{ fill: COLOR.token }}>
                  Human Anchor Preserves Distribution Variance!
                </Label>
                <Label x={320} y={96} anchor="middle" size={8.5} tone={2}>
                  30% real text injects rare tail tokens; execution checks reject false proofs.
                </Label>
              </g>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
