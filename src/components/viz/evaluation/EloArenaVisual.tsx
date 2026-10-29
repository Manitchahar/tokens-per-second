import { useRef, useState } from 'react';
import { Stage, Slider, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, COLOR } from '../lib/svg';

const STEPS = [
  'Blind pairwise battles in Chatbot Arena',
  'The Elo expected outcome curve',
  'Worked arithmetic: 1200 vs 1100 rating',
  'Rating updates: R’ = R + K(S - E)',
  'Bradley-Terry: the statistical foundation',
  'The Arena dilemma: public chats vs enterprise apps',
];

const CAPTIONS = [
  'In crowdsourced arenas, human users submit arbitrary prompts. Two anonymous models generate answers side-by-side. The user votes on the better answer.',
  'The Elo system calculates expected win probability: E_A = 1 / (1 + 10^((R_B - R_A)/400)). A 400-point advantage implies a 91% expected win rate.',
  'When Model A (1200) faces Model B (1100), the 100-point spread gives Model A an expected win probability of 64.0%.',
  'If Model A wins (S=1) with K=32: Model A gains +32*(1 - 0.64) = +11.5 points, while Model B loses -11.5 points. Ratings are zero-sum.',
  'Under the Bradley-Terry model, P(A > B) = exp(r_A) / (exp(r_A) + exp(r_B)). Maximum likelihood estimation fits continuous latent skill parameters.',
  'What wins crowdsourced chat votes (humor, markdown styling, polite verbosity) is often inversely correlated with production needs (strict schema, low latency).',
];

export default function EloArenaVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [ratingA, setRatingA] = useState(1200);
  const [ratingB, setRatingB] = useState(1100);
  const [outcome, setOutcome] = useState<'A' | 'B' | 'Tie'>('A');

  // Elo arithmetic
  const delta = ratingB - ratingA;
  const expProbA = 1 / (1 + Math.pow(10, delta / 400));
  const expProbB = 1 - expProbA;
  const K = 32;

  const scoreA = outcome === 'A' ? 1.0 : outcome === 'B' ? 0.0 : 0.5;
  const scoreB = 1.0 - scoreA;

  const deltaA = Math.round(K * (scoreA - expProbA) * 10) / 10;
  const deltaB = Math.round(K * (scoreB - expProbB) * 10) / 10;

  const newRatingA = Math.round(ratingA + deltaA);
  const newRatingB = Math.round(ratingB + deltaB);

  return (
    <Stage
      rootRef={ref}
      title="Comparative Evaluation: Elo Ratings and Chatbot Arena"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'error', 'memory', 'compute']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Model A Rating', value: `${ratingA}`, kind: 'token', note: `Expected Win: ${(expProbA * 100).toFixed(1)}%` },
          { label: 'Model B Rating', value: `${ratingB}`, kind: 'memory', note: `Expected Win: ${(expProbB * 100).toFixed(1)}%` },
          { label: 'Outcome', value: outcome === 'A' ? 'Model A Won' : outcome === 'B' ? 'Model B Won' : 'Tie', kind: 'text' },
          { label: 'Rating Shift', value: `${deltaA >= 0 ? '+' : ''}${deltaA} pts`, kind: deltaA >= 0 ? 'token' : 'error', note: `New: ${newRatingA} vs ${newRatingB}` },
        ]} />
      }
      controls={
        s.step >= 1 ? (
          <>
            <Slider
              label="Model A Rating (R_A)"
              value={ratingA}
              min={900}
              max={1500}
              onChange={setRatingA}
              format={(v) => `${v} Elo`}
              kind="token"
            />
            <Slider
              label="Model B Rating (R_B)"
              value={ratingB}
              min={900}
              max={1500}
              onChange={setRatingB}
              format={(v) => `${v} Elo`}
              kind="memory"
            />
            {s.step >= 3 && (
              <Segmented
                label="Match outcome"
                value={outcome}
                options={[
                  { value: 'A', label: 'Model A Wins (S=1)' },
                  { value: 'Tie', label: 'Tie (S=0.5)' },
                  { value: 'B', label: 'Model B Wins (S=0)' },
                ]}
                onChange={(val) => setOutcome(val as any)}
              />
            )}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Visual explanation of Elo ratings, expected score curves, rating updates, and Chatbot Arena comparative evaluation.">
        {/* Step 0: Arena setup */}
        {s.step === 0 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Crowdsourced Blind Evaluation: LMSYS Chatbot Arena</Label>

            {/* Prompt Box */}
            <rect x={24} y={50} width={592} height={40} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <text x={36} y={74} fill="var(--text)" fontSize={11} fontFamily="var(--font-sans)">
              User Prompt: "Write a Python script to verify if a graph contains a negative cycle."
            </text>

            {/* Side by side outputs */}
            <rect x={24} y={100} width={285} height={120} rx={4} fill="var(--surface-1)" stroke={COLOR.token} strokeWidth={1.5} />
            <Label x={36} y={122} kind="token" size={10.5} bold>Model A (Hidden: Llama-3.1-70B)</Label>
            <text x={36} y={144} fill="var(--text)" fontSize={9.5} fontFamily="var(--font-sans)">
              Provides concise Bellman-Ford algorithm with docstrings and time complexity analysis O(V * E).
            </text>
            <Label x={36} y={200} tone={3} size={9}>Anonymous to human user</Label>

            <rect x={331} y={100} width={285} height={120} rx={4} fill="var(--surface-1)" stroke={COLOR.memory} strokeWidth={1.5} />
            <Label x={343} y={122} kind="memory" size={10.5} bold>Model B (Hidden: Qwen2.5-32B)</Label>
            <text x={343} y={144} fill="var(--text)" fontSize={9.5} fontFamily="var(--font-sans)">
              Provides Floyd-Warshall approach with detailed comments and interactive example execution.
            </text>
            <Label x={343} y={200} tone={3} size={9}>Anonymous to human user</Label>

            {/* User action */}
            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={1} size={11} bold>Human Vote:</Label>
            <text x={120} y={252} fill="var(--text-2)" fontSize={10.5} fontFamily="var(--font-sans)">
              User clicks "Model A is better" → Model names revealed → Elo rating updated in database.
            </text>
          </g>
        )}

        {/* Step 1: Elo Expected Score Curve */}
        {s.step === 1 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>The Elo Logistic Win Probability Curve</Label>

            {/* Chart Area */}
            <rect x={24} y={50} width={420} height={220} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            
            {/* Axes */}
            <line x1={50} y1={160} x2={420} y2={160} stroke="var(--rule-2)" strokeWidth={1} strokeDasharray="3 3" />
            <line x1={235} y1={70} x2={235} y2={240} stroke="var(--rule-2)" strokeWidth={1} strokeDasharray="3 3" />
            <line x1={50} y1={240} x2={420} y2={240} stroke="var(--rule-2)" strokeWidth={1} />
            <line x1={50} y1={70} x2={50} y2={240} stroke="var(--rule-2)" strokeWidth={1} />

            <Label x={42} y={75} tone={3} size={9} anchor="end">100%</Label>
            <Label x={42} y={160} tone={3} size={9} anchor="end">50%</Label>
            <Label x={42} y={240} tone={3} size={9} anchor="end">0%</Label>

            <Label x={50} y={255} tone={3} size={8.5}>-400 pts</Label>
            <Label x={235} y={255} tone={3} size={8.5} anchor="middle">0 (Even)</Label>
            <Label x={420} y={255} tone={3} size={8.5} anchor="end">+400 pts</Label>

            {/* S-curve path */}
            <path
              d="M 50 225 Q 160 215 235 160 T 420 85"
              fill="none"
              stroke={COLOR.token}
              strokeWidth={2.5}
            />

            {/* Current point */}
            {(() => {
              const ratingDiff = ratingA - ratingB;
              const cx = 235 + (ratingDiff / 400) * 185;
              const cy = 240 - expProbA * 170;
              return (
                <g>
                  <circle cx={Math.max(50, Math.min(420, cx))} cy={cy} r={5} fill={COLOR.compute} stroke="#fff" strokeWidth={1} />
                  <Label x={Math.max(60, Math.min(400, cx))} y={cy - 10} kind="compute" size={10} bold anchor="middle">
                    {`${(expProbA * 100).toFixed(1)}% (Diff: ${ratingA - ratingB > 0 ? '+' : ''}${ratingA - ratingB})`}
                  </Label>
                </g>
              );
            })()}

            {/* Right formula card */}
            <rect x={460} y={50} width={156} height={220} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={472} y={74} tone={1} size={11} bold>The Equation:</Label>
            <text x={472} y={100} fill="var(--text)" fontSize={10} fontFamily="var(--font-mono)">E_A = 1 /</text>
            <text x={472} y={116} fill="var(--text)" fontSize={10} fontFamily="var(--font-mono)">[1 + 10^(Δ/400)]</text>
            <text x={472} y={145} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-sans)">Δ = R_B - R_A</text>
            <text x={472} y={170} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Δ = 0 → E = 50%</text>
            <text x={472} y={190} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Δ = -200 → E = 76%</text>
            <text x={472} y={210} fill="var(--text-2)" fontSize={9} fontFamily="var(--font-sans)">• Δ = -400 → E = 91%</text>
          </g>
        )}

        {/* Step 2: Worked example */}
        {s.step === 2 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Worked Numeric Example: R_A = 1200, R_B = 1100</Label>

            <rect x={24} y={54} width={592} height={210} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={78} tone={1} size={11.5} bold>Step 1: Compute Rating Differential</Label>
            <text x={40} y={100} fill="var(--text)" fontSize={11} fontFamily="var(--font-mono)">
              R_B - R_A = 1100 - 1200 = -100
            </text>

            <Label x={40} y={128} tone={1} size={11.5} bold>Step 2: Exponentiate the Scaled Difference</Label>
            <text x={40} y={150} fill="var(--text)" fontSize={11} fontFamily="var(--font-mono)">
              10^(-100 / 400) = 10^(-0.25) ≈ 0.56234
            </text>

            <Label x={40} y={178} tone={1} size={11.5} bold>Step 3: Solve Expected Probability</Label>
            <text x={40} y={200} fill="var(--text)" fontSize={11} fontFamily="var(--font-mono)">
              E_A = 1 / (1 + 0.56234) = 1 / 1.56234 ≈ 0.6401 (64.0%)
            </text>

            <Span x={40} y={222} w={200} h={24} kind="token" label="Model A Expected Win: 64.0%" />
            <Span x={250} y={222} w={200} h={24} kind="memory" label="Model B Expected Win: 36.0%" />
          </g>
        )}

        {/* Step 3: Rating update */}
        {s.step === 3 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>Rating Update Dynamics (K = 32)</Label>

            {/* Model A update card */}
            <rect x={24} y={54} width={285} height={160} rx={4} fill="var(--surface-2)" stroke={COLOR.token} />
            <Label x={36} y={76} tone={1} size={11.5} bold>Model A Update</Label>
            <text x={36} y={100} fill="var(--text)" fontSize={10.5} fontFamily="var(--font-mono)">R'_A = R_A + K * (S_A - E_A)</text>
            <text x={36} y={122} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">Current: {ratingA} | E_A: {(expProbA * 100).toFixed(1)}%</text>
            <text x={36} y={142} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">Actual Score S_A: {scoreA}</text>
            <text x={36} y={166} fill={deltaA >= 0 ? COLOR.token : COLOR.error} fontSize={11} fontFamily="var(--font-mono)">
              Shift: {K} * ({scoreA} - {expProbA.toFixed(2)}) = {deltaA >= 0 ? '+' : ''}{deltaA}
            </text>
            <Span x={36} y={180} w={160} h={22} kind="token" label={`New Elo: ${newRatingA}`} />

            {/* Model B update card */}
            <rect x={331} y={54} width={285} height={160} rx={4} fill="var(--surface-2)" stroke={COLOR.memory} />
            <Label x={343} y={76} tone={1} size={11.5} bold>Model B Update</Label>
            <text x={343} y={100} fill="var(--text)" fontSize={10.5} fontFamily="var(--font-mono)">R'_B = R_B + K * (S_B - E_B)</text>
            <text x={343} y={122} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">Current: {ratingB} | E_B: {(expProbB * 100).toFixed(1)}%</text>
            <text x={343} y={142} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">Actual Score S_B: {scoreB}</text>
            <text x={343} y={166} fill={deltaB >= 0 ? COLOR.token : COLOR.error} fontSize={11} fontFamily="var(--font-mono)">
              Shift: {K} * ({scoreB} - {expProbB.toFixed(2)}) = {deltaB >= 0 ? '+' : ''}{deltaB}
            </text>
            <Span x={343} y={180} w={160} h={22} kind="memory" label={`New Elo: ${newRatingB}`} />

            {/* Zero-sum takeaway */}
            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={2} size={10.5}>
              Zero-sum invariant: ΔR_A + ΔR_B = 0. Points won by one model are exactly deducted from the loser.
            </Label>
          </g>
        )}

        {/* Step 4: Bradley-Terry */}
        {s.step === 4 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>The Bradley-Terry Model: Links to Alignment (Module 3C)</Label>

            <rect x={24} y={54} width={592} height={150} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={40} y={78} tone={1} size={11.5} bold>The Bradley-Terry Formulation:</Label>
            <text x={40} y={104} fill="var(--text)" fontSize={12} fontFamily="var(--font-mono)">
              {"P(A > B) = exp(r_A) / [exp(r_A) + exp(r_B)] = 1 / [1 + exp(-(r_A - r_B))]"}
            </text>
            <text x={40} y={132} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">
              In Module 3C (RLHF & DPO), reward models are trained directly on Bradley-Terry pairwise preference loss:
            </text>
            <text x={40} y={154} fill={COLOR.token} fontSize={10} fontFamily="var(--font-mono)">
              L_BT = - log( sigma( r_theta(x, y_w) - r_theta(x, y_l) ) )
            </text>
            <text x={40} y={178} fill="var(--text-3)" fontSize={9.5} fontFamily="var(--font-sans)">
              Elo is simply the logistical re-scaling of Bradley-Terry with base 10 and scale factor 400.
            </text>

            <rect x={24} y={218} width={592} height={60} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={240} tone={1} size={11} bold>Leaderboard Estimation:</Label>
            <text x={40} y={258} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">
              Chatbot Arena fits Bradley-Terry coefficients using Logistic Regression over millions of battles, computing 95% bootstrap confidence intervals.
            </text>
          </g>
        )}

        {/* Step 5: The Arena Dilemma */}
        {s.step === 5 && (
          <g>
            <Label x={24} y={30} tone={1} size={13} bold>The Arena Dilemma: Why Public Elo Fails Enterprise AI</Label>

            <rect x={24} y={54} width={285} height={160} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
            <Label x={36} y={76} kind="token" size={11} bold>What Wins Chatbot Arena:</Label>
            <text x={36} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Conversational charm & sycophancy</text>
            <text x={36} y={120} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Elaborate bold/italic markdown</text>
            <text x={36} y={140} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Long, thorough answers (verbosity)</text>
            <text x={36} y={160} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Creative writing & poetry</text>
            <Span x={36} y={180} w={180} h={20} kind="token" label="Crowd Appeal Optimization" />

            <rect x={331} y={54} width={285} height={160} rx={4} fill="var(--surface-2)" stroke={COLOR.error} />
            <Label x={343} y={76} kind="error" size={11} bold>What Production Apps Actually Need:</Label>
            <text x={343} y={100} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Strict JSON / Pydantic schema validity</text>
            <text x={343} y={120} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Sub-200ms latency (TTFT & TPOT)</text>
            <text x={343} y={140} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Zero hallucination on corporate docs</text>
            <text x={343} y={160} fill="var(--text-2)" fontSize={9.5} fontFamily="var(--font-sans)">• Predictable cost at volume ($ / 1M)</text>
            <Span x={343} y={180} w={180} h={20} kind="error" label="Engineering Reliability" />

            <rect x={24} y={230} width={592} height={50} rx={4} fill="var(--surface-1)" stroke="var(--rule-2)" />
            <Label x={40} y={252} tone={2} size={10.5}>
              Never select your enterprise foundation model based solely on Chatbot Arena Elo. Build your domain arena.
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
