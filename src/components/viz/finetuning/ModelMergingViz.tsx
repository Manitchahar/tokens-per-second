import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade } from '../lib/svg';

const STEPS = [
  'The weight space paradox: Linear mode connectivity',
  'Task Arithmetic: The task vector τ = θ_ft - θ_base',
  'Linear Merging & Model Soups',
  'SLERP: Spherical interpolation on the hypersphere',
  'TIES: Trimming noise & electing sign consensus',
  'DARE: Random Bernoulli pruning & rescaling',
];

const CAPTIONS = [
  'Merging randomly initialized networks produces noise. But models fine-tuned from the same shared base model share a convex loss basin — there are no energy barriers between them.',
  'A task vector represents the pure behavioral delta: τ = θ_finetuned - θ_base. Task vectors can be added to combine capabilities (Code + Math) or subtracted to negate behaviors.',
  'Model Soups simply average model weights: θ_merged = 0.5·θ_A + 0.5·θ_B. Simple and effective for similar checkpoints, but causes norm collapse and interference on distinct tasks.',
  'In high dimensions, linear interpolation cuts through the interior of the hypersphere, reducing weight magnitude. SLERP travels along the spherical geodesic, preserving parameter scale.',
  'TIES (Trimming, Electing Signs, Merging) resolves directional interference: it prunes 80% smallest deltas, calculates sign consensus per parameter, and averages only aligned weights.',
  'DARE (Drop And REscale) randomly zeroes 90% of delta parameters with probability p and scales the remaining weights by 1/(1-p). It eliminates cross-task interference with zero tuning.',
];

const METHODS = [
  { name: 'Linear', formula: 'θ_merged = (θ_A + θ_B) / 2', desc: 'Simple element-wise average. Suffers norm shrinking in high dimensions.' },
  { name: 'SLERP', formula: 'θ_slerp = (sin((1-t)Ω)/sinΩ)θ_A + (sin(tΩ)/sinΩ)θ_B', desc: 'Spherical geodesic interpolation. Preserves weight norms and angles.' },
  { name: 'TIES', formula: 'Trim(top 20%) → Elect Sign → Disjoint Merge', desc: 'Removes parameter interference and sign conflicts across multiple tasks.' },
  { name: 'DARE', formula: 'Mask(Bernoulli p=0.9) × 1/(1-p) + θ_base', desc: 'Extreme 90% sparsification + rescaling. Outstanding multi-task retention.' },
];

export default function ModelMergingViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [activeMethod, setActiveMethod] = useState(0);

  // Geometry: Base at (160, 180), Model A (Coding) at (260, 60), Model B (Math) at (460, 160)
  const basePos = { x: 120, y: 190 };
  const aPos = { x: 280, y: 55 };
  const bPos = { x: 490, y: 160 };

  // Midpoints
  const linearMid = { x: (aPos.x + bPos.x) / 2, y: (aPos.y + bPos.y) / 2 }; // (385, 107.5)
  // SLERP arc point: bowed outward along circle from base
  const slerpPos = { x: 410, y: 85 };
  const tiesPos = { x: 395, y: 95 };
  const darePos = { x: 400, y: 90 };

  const currentMethod = METHODS[activeMethod];

  return (
    <Stage
      rootRef={ref}
      title="Model Merging & Task Arithmetic in Parameter Space"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'base foundation model θ_base'],
        ['compute', 'expert models (Code / Math)'],
        ['token', 'merged model checkpoint θ_merged'],
        ['idle', 'task vectors (τ_A, τ_B)'],
      ]}
      controls={
        st >= 2 ? (
          <Segmented
            options={METHODS.map((m) => m.name)}
            value={METHODS[activeMethod].name}
            onChange={(val) => {
              const idx = METHODS.findIndex((m) => m.name === val);
              if (idx !== -1) setActiveMethod(idx);
            }}
          />
        ) : undefined
      }
      readout={
        st >= 2 ? (
          <Readout
            items={[
              { label: 'Algorithm', value: currentMethod.name },
              { label: 'Equation', value: currentMethod.formula },
              { label: 'Mechanic', value: currentMethod.desc },
            ]}
          />
        ) : (
          <Readout
            items={[
              { label: 'Phase', value: STEPS[st] },
              { label: 'Principle', value: st === 0 ? 'Linear Mode Connectivity' : 'Task Vector: τ = θ_ft - θ_base' },
            ]}
          />
        )
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Geometric illustration of task vector arithmetic, spherical linear interpolation, TIES, and DARE model merging">
        {/* Background Convex Basin / Hypersphere Arc */}
        <path
          d="M 60 210 Q 250 250 560 200"
          fill="none"
          stroke="var(--rule-2)"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <Label x={540} y={225} tone={3} size={9} mono anchor="end">
          Shared Loss Basin
        </Label>

        {/* Task Vector Vectors from Base */}
        {/* Vector 1: Base -> Model A (Coding) */}
        <line
          x1={basePos.x}
          y1={basePos.y}
          x2={aPos.x}
          y2={aPos.y}
          stroke="var(--rule-2)"
          strokeWidth={1.5}
          strokeDasharray={st === 0 ? '3 3' : undefined}
        />
        <Label x={(basePos.x + aPos.x) / 2 - 25} y={(basePos.y + aPos.y) / 2 - 10} tone={2} size={10} mono>
          τ_code = θ_A - θ_0
        </Label>

        {/* Vector 2: Base -> Model B (Math) */}
        <line
          x1={basePos.x}
          y1={basePos.y}
          x2={bPos.x}
          y2={bPos.y}
          stroke="var(--rule-2)"
          strokeWidth={1.5}
          strokeDasharray={st === 0 ? '3 3' : undefined}
        />
        <Label x={(basePos.x + bPos.x) / 2 + 10} y={(basePos.y + bPos.y) / 2 + 20} tone={2} size={10} mono>
          τ_math = θ_B - θ_0
        </Label>

        {/* Interpolation paths between A and B */}
        {/* Straight linear chord */}
        <line
          x1={aPos.x}
          y1={aPos.y}
          x2={bPos.x}
          y2={bPos.y}
          stroke={activeMethod === 0 && st >= 2 ? 'var(--token)' : 'var(--rule-2)'}
          strokeWidth={activeMethod === 0 && st >= 2 ? 2 : 1}
          strokeDasharray={activeMethod === 0 ? undefined : '2 2'}
        />

        {/* Curved Spherical Arc (SLERP) */}
        <path
          d={`M ${aPos.x} ${aPos.y} Q 405 60 ${bPos.x} ${bPos.y}`}
          fill="none"
          stroke={activeMethod === 1 && st >= 2 ? 'var(--token)' : 'rgba(255, 255, 255, 0.1)'}
          strokeWidth={activeMethod === 1 && st >= 2 ? 2 : 1}
        />

        {/* NODE 1: Base Foundation Model θ_0 */}
        <g className="base-node">
          <circle cx={basePos.x} cy={basePos.y} r={18} fill="rgba(46, 144, 250, 0.15)" stroke="var(--memory)" strokeWidth={2} />
          <Label x={basePos.x} y={basePos.y + 4} tone={1} bold size={11} mono anchor="middle">
            θ_0
          </Label>
          <Label x={basePos.x} y={basePos.y + 32} tone={1} size={10} mono anchor="middle">
            Base Model
          </Label>
          <Label x={basePos.x} y={basePos.y + 44} tone={3} size={8.5} mono anchor="middle">
            Llama-3.1-8B
          </Label>
        </g>

        {/* NODE 2: Finetuned Model A (Coding Specialist) */}
        <g className="model-a-node">
          <circle cx={aPos.x} cy={aPos.y} r={16} fill="rgba(247, 144, 9, 0.15)" stroke="var(--compute)" strokeWidth={1.5} />
          <Label x={aPos.x} y={aPos.y + 4} tone={1} bold size={10.5} mono anchor="middle">
            θ_A
          </Label>
          <Label x={aPos.x} y={aPos.y - 24} tone={1} bold size={10.5} mono anchor="middle">
            Model A (Code)
          </Label>
          <Label x={aPos.x} y={aPos.y - 12} tone={3} size={8.5} mono anchor="middle">
            Python/Rust SFT
          </Label>
        </g>

        {/* NODE 3: Finetuned Model B (Math Specialist) */}
        <g className="model-b-node">
          <circle cx={bPos.x} cy={bPos.y} r={16} fill="rgba(247, 144, 9, 0.15)" stroke="var(--compute)" strokeWidth={1.5} />
          <Label x={bPos.x} y={bPos.y + 4} tone={1} bold size={10.5} mono anchor="middle">
            θ_B
          </Label>
          <Label x={bPos.x} y={bPos.y + 30} tone={1} bold size={10.5} mono anchor="middle">
            Model B (Math)
          </Label>
          <Label x={bPos.x} y={bPos.y + 42} tone={3} size={8.5} mono anchor="middle">
            GSM8K / Proof SFT
          </Label>
        </g>

        {/* NODE 4: Merged Model Result */}
        {st >= 2 && (
          <g className="merged-node">
            {(() => {
              const pt =
                activeMethod === 0 ? linearMid : activeMethod === 1 ? slerpPos : activeMethod === 2 ? tiesPos : darePos;
              return (
                <g>
                  {/* Vector from Base to Merged */}
                  <line x1={basePos.x} y1={basePos.y} x2={pt.x} y2={pt.y} stroke="var(--token)" strokeWidth={1.5} strokeDasharray="3 3" />
                  <circle cx={pt.x} cy={pt.y} r={17} fill="rgba(18, 183, 106, 0.25)" stroke="var(--token)" strokeWidth={2} />
                  <Label x={pt.x} y={pt.y + 4} tone={1} bold size={11} mono anchor="middle">
                    θ_merge
                  </Label>
                  <Label x={pt.x + 24} y={pt.y - 10} kind="token" bold size={10} mono>
                    {activeMethod === 0
                      ? 'Linear Average'
                      : activeMethod === 1
                      ? 'SLERP Geodesic'
                      : activeMethod === 2
                      ? 'TIES Consensus'
                      : 'DARE Rescaled'}
                  </Label>
                  <Label x={pt.x + 24} y={pt.y + 5} tone={3} size={8.5} mono>
                    Code + Math (Zero extra GPU compute!)
                  </Label>
                </g>
              );
            })()}
          </g>
        )}

        {/* Lower Detail Card */}
        <rect
          x={32}
          y={236}
          width={576}
          height={68}
          rx={6}
          fill="rgba(0, 0, 0, 0.25)"
          stroke="var(--rule-2)"
          strokeWidth={1}
        />

        <Fade show={st === 0}>
          <Label x={46} y={254} tone={1} bold size={11.5} mono={false}>
            Linear Mode Connectivity: Why Model Merging Is Possible
          </Label>
          <Label x={46} y={270} tone={2} size={10} mono={false}>
            Independent pretraining creates disconnected local minima. But models fine-tuned from the SAME base
          </Label>
          <Label x={46} y={286} tone={3} size={9.5} mono={false}>
            stay inside the same convex loss basin. Linear paths between them contain zero loss barriers.
          </Label>
        </Fade>

        <Fade show={st === 1}>
          <Label x={46} y={254} tone={1} bold size={11.5} mono={false}>
            Task Arithmetic: Vectors in Parameter Space
          </Label>
          <Label x={46} y={270} tone={2} size={10} mono={false}>
            Task vector τ = θ_finetuned - θ_base isolates the exact directional update for a skill.
          </Label>
          <Label x={46} y={286} tone={3} size={9.5} mono={false}>
            Capabilities can be added: θ_new = θ_0 + τ_code + τ_math; or unlearned: θ_clean = θ_0 - τ_toxic.
          </Label>
        </Fade>

        <Fade show={st === 2}>
          <Label x={46} y={254} tone={1} bold size={11.5} mono={false}>
            Linear Merging & Model Soups (Wortsman et al., 2022)
          </Label>
          <Label x={46} y={270} tone={2} size={10} mono={false}>
            Simple weighted average: θ_merged = λ_A·θ_A + (1-λ_A)·θ_B.
          </Label>
          <Label x={46} y={286} tone={3} size={9.5} mono={false}>
            Problem: Cutting straight through the hypersphere shrinks vector magnitude, dampening confident predictions.
          </Label>
        </Fade>

        <Fade show={st === 3}>
          <Label x={46} y={254} tone={1} bold size={11.5} mono={false}>
            SLERP: Spherical Linear Interpolation
          </Label>
          <Label x={46} y={270} tone={2} size={10} mono={false}>
            Interpolates along the constant-radius spherical geodesic using angles: Ω = arccos(θ_A · θ_B / (|θ_A||θ_B|)).
          </Label>
          <Label x={46} y={286} tone={3} size={9.5} mono={false}>
            Preserves the original parameter norm and activation sharpness. Default standard in MergeKit for 2 models.
          </Label>
        </Fade>

        <Fade show={st === 4}>
          <Label x={46} y={254} tone={1} bold size={11.5} mono={false}>
            TIES-Merging: Trimming Noise & Electing Sign Consensus (Yadav et al., 2023)
          </Label>
          <Label x={46} y={270} tone={2} size={10} mono={false}>
            1. Trim bottom 80% parameter deltas. 2. Elect sign consensus (+1 or -1) per parameter. 3. Average aligned weights.
          </Label>
          <Label x={46} y={286} tone={3} size={9.5} mono={false}>
            Resolves destructive parameter interference when combining 3 to 10 diverse expert models simultaneously.
          </Label>
        </Fade>

        <Fade show={st === 5}>
          <Label x={46} y={254} tone={1} bold size={11.5} mono={false}>
            DARE: Drop And REscale (Yu et al., 2023)
          </Label>
          <Label x={46} y={270} tone={2} size={10} mono={false}>
            Randomly drops up to 90% of delta weights using a Bernoulli mask and scales remaining weights by 1/(1-p).
          </Label>
          <Label x={46} y={286} tone={3} size={9.5} mono={false}>
            Proves task vectors are extremely sparse and robust. Merges multiple 7B models on a CPU laptop in minutes.
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
