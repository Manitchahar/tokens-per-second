import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade } from '../lib/svg';

const STEPS = [
  'The rank bottleneck intuition',
  'The low-rank factorisation: ΔW = B × A',
  'Initialization: B = 0 ensures identity start',
  'The scaling factor: α / r',
  'Parameter efficiency: 0.39% trainable weights',
  'Zero-latency deployment: Folding ΔW into W0',
];

const CAPTIONS = [
  'Full finetuning updates all d × k weights in W. But neural network task updates have a very low "intrinsic dimension" — they live in a tiny low-rank subspace.',
  'Instead of optimizing a dense 4096 × 4096 matrix, factorize ΔW into two thin matrices: A (r × 4096) and B (4096 × r), where rank r is typically 4 to 16.',
  'Matrix A is initialized from a Gaussian distribution; matrix B is initialized to exactly zero. Therefore ΔW = 0 at step 0, preserving the base model’s exact behavior.',
  'The adapter output is scaled by α/r, a constant scaling hyperparameter. Setting α = 16 or 2r stabilizes optimization when experimenting with different ranks.',
  'At d = 4096 and r = 8, W has 16,777,216 parameters while A + B have only 65,536 parameters. That is 256× fewer weights (0.39%) to store and compute gradients for.',
  'For single-tenant inference, compute W_merged = W0 + (α/r)·BA once at startup. The serving engine executes a standard linear layer with zero added latency.',
];

const RANKS = [2, 4, 8, 16, 64];

export default function LoraViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [rank, setRank] = useState(8);
  const [isMerged, setIsMerged] = useState(false);

  const d = 4096;
  const k = 4096;
  const denseParams = d * k; // 16,777,216
  const loraParams = (k * rank) + (rank * d); // 2 * 4096 * r
  const pct = (loraParams / denseParams) * 100;
  const reduction = denseParams / loraParams;

  const effectiveMerged = st === 5 || isMerged;

  return (
    <Stage
      rootRef={ref}
      title="LoRA: Low-Rank Adaptation (ΔW = B × A)"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'frozen base weight W0'],
        ['compute', 'down-projection A (r × k)'],
        ['token', 'up-projection B (d × r)'],
        ['idle', 'input / output activations'],
      ]}
      controls={
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <Segmented
            options={RANKS.map((r) => `r = ${r}`)}
            value={`r = ${rank}`}
            onChange={(val) => {
              const rVal = parseInt(val.replace('r = ', ''), 10);
              if (!isNaN(rVal)) setRank(rVal);
            }}
          />
          {st >= 4 && (
            <button
              type="button"
              className="stage-btn"
              onClick={() => setIsMerged(!isMerged)}
              style={{
                padding: '4px 10px',
                fontSize: 11,
                fontFamily: 'var(--mono)',
                background: effectiveMerged ? 'var(--token)' : 'rgba(255,255,255,0.06)',
                color: effectiveMerged ? '#000' : 'var(--text-1)',
                border: '1px solid var(--rule-2)',
                borderRadius: 4,
                cursor: 'pointer',
              }}
            >
              {effectiveMerged ? 'Merged: W = W0 + BA' : 'Separate: W0 + BA'}
            </button>
          )}
        </div>
      }
      readout={
        <Readout
          items={[
            { label: 'Rank r', value: `${rank}` },
            { label: 'Trainable Params', value: `${loraParams.toLocaleString()} (${pct.toFixed(2)}%)` },
            { label: 'Base Params', value: `${(denseParams / 1e6).toFixed(2)} M` },
            { label: 'Parameter Savings', value: `${reduction.toFixed(0)}× fewer` },
          ]}
        />
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Architecture diagram of Low-Rank Adaptation with base weights and parallel adapter branch">
        {/* Input Vector x */}
        <g className="input-stream">
          <Label x={40} y={150} tone={1} size={11} mono anchor="middle">
            Input x
          </Label>
          <Label x={40} y={166} tone={3} size={9} mono anchor="middle">
            [1, 4096]
          </Label>
          <circle cx={40} cy={135} r={4} fill="var(--rule-2)" />
          {/* Main branch line */}
          <line x1={44} y1={135} x2={120} y2={135} stroke="var(--rule-2)" strokeWidth={1.5} />
          {/* Adapter split line */}
          {!effectiveMerged && (
            <path
              d="M 80 135 L 80 230 L 160 230"
              fill="none"
              stroke="var(--rule-2)"
              strokeWidth={1.5}
              strokeDasharray={st === 0 ? '3 3' : undefined}
            />
          )}
        </g>

        {/* Path A: Frozen Base Weight W0 */}
        <g className="base-weight">
          <rect
            x={120}
            y={80}
            width={110}
            height={110}
            rx={4}
            fill={effectiveMerged ? 'rgba(46, 144, 250, 0.25)' : 'rgba(46, 144, 250, 0.12)'}
            stroke="var(--memory)"
            strokeWidth={1.5}
          />
          <Label x={175} y={128} tone={1} bold size={13} mono anchor="middle">
            {effectiveMerged ? 'W_merged' : 'W_0'}
          </Label>
          <Label x={175} y={146} tone={2} size={10} mono anchor="middle">
            {effectiveMerged ? 'W0 + (α/r)BA' : 'FROZEN'}
          </Label>
          <Label x={175} y={162} tone={3} size={9} mono anchor="middle">
            4096 × 4096
          </Label>
          <line x1={230} y1={135} x2={430} y2={135} stroke="var(--rule-2)" strokeWidth={1.5} />
        </g>

        {/* Path B: Low-Rank Adapter Branch (A and B) */}
        {!effectiveMerged && (
          <g className="adapter-branch" style={{ opacity: st === 0 ? 0.25 : 1, transition: 'opacity 0.3s ease' }}>
            {/* Matrix A: Down-projection [r, 4096] */}
            {/* Drawn as a wide, very thin horizontal rectangle */}
            <rect
              x={160}
              y={212}
              width={90}
              height={Math.max(12, rank * 1.5)}
              rx={2}
              fill="rgba(247, 144, 9, 0.2)"
              stroke="var(--compute)"
              strokeWidth={1.5}
            />
            <Label x={205} y={225} tone={1} bold size={11} mono anchor="middle">
              A
            </Label>
            <Label x={205} y={248} tone={3} size={8.5} mono anchor="middle">
              [{rank} × 4096]
            </Label>
            <Label x={205} y={260} tone={2} size={8} mono anchor="middle">
              {st >= 2 ? 'Gaussian N(0, σ²)' : 'Trainable'}
            </Label>

            {/* Intermediate activation vector h_mid = x·A */}
            <line x1={250} y1={220} x2={300} y2={220} stroke="var(--rule-2)" strokeWidth={1.5} />
            <Label x={275} y={212} tone={3} size={8.5} mono anchor="middle">
              r = {rank}
            </Label>

            {/* Matrix B: Up-projection [4096, r] */}
            {/* Drawn as a tall, narrow vertical rectangle */}
            <rect
              x={300}
              y={180}
              width={Math.max(12, rank * 1.5)}
              height={80}
              rx={2}
              fill="rgba(18, 183, 106, 0.2)"
              stroke="var(--token)"
              strokeWidth={1.5}
            />
            <Label x={300 + Math.max(6, (rank * 1.5) / 2)} y={224} tone={1} bold size={11} mono anchor="middle">
              B
            </Label>
            <Label x={300 + Math.max(6, (rank * 1.5) / 2)} y={276} tone={3} size={8.5} mono anchor="middle">
              [4096 × {rank}]
            </Label>
            <Label x={300 + Math.max(6, (rank * 1.5) / 2)} y={288} tone={2} size={8} mono anchor="middle">
              {st >= 2 ? 'Init: Exactly 0' : 'Trainable'}
            </Label>

            {/* Connection from B to Scale & Add Node */}
            <path
              d={`M ${300 + Math.max(12, rank * 1.5)} 220 L 370 220 L 370 170`}
              fill="none"
              stroke="var(--rule-2)"
              strokeWidth={1.5}
            />

            {/* Scaling multiplier node (α / r) */}
            <circle cx={370} cy={170} r={14} fill="rgba(255, 255, 255, 0.04)" stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={370} y={173} tone={1} size={8.5} mono anchor="middle">
              × {st >= 3 ? 'α/r' : '1.0'}
            </Label>
            <line x1={370} y1={156} x2={430} y2={140} stroke="var(--rule-2)" strokeWidth={1.5} />
          </g>
        )}

        {/* Summation Node (+) */}
        <g className="sum-node">
          <circle cx={445} cy={135} r={16} fill="rgba(255, 255, 255, 0.05)" stroke="var(--text-1)" strokeWidth={1.5} />
          <Label x={445} y={140} tone={1} bold size={16} mono anchor="middle">
            {effectiveMerged ? '=' : '+'}
          </Label>
          <line x1={461} y1={135} x2={530} y2={135} stroke="var(--rule-2)" strokeWidth={1.5} />
        </g>

        {/* Output Vector h */}
        <g className="output-stream">
          <circle cx={534} cy={135} r={4} fill="var(--token)" />
          <Label x={570} y={132} tone={1} bold size={11} mono anchor="middle">
            Output h
          </Label>
          <Label x={570} y={148} tone={3} size={9} mono anchor="middle">
            {effectiveMerged ? 'h = x·W_merged' : 'h = x·W0 + x·BA(α/r)'}
          </Label>
          <Label x={570} y={164} tone={2} size={8.5} mono anchor="middle">
            [1, 4096]
          </Label>
        </g>

        {/* Upper Mathematical Formula Display */}
        <rect
          x={40}
          y={15}
          width={560}
          height={48}
          rx={4}
          fill="rgba(0, 0, 0, 0.25)"
          stroke="var(--rule-2)"
          strokeWidth={1}
        />

        <Label x={55} y={34} tone={1} bold size={11} mono={false}>
          Forward Equation:
        </Label>
        <Label x={160} y={34} tone={2} size={11} mono>
          {effectiveMerged
            ? 'h = x · (W_0 + (α/r) · B · A) = x · W_merged'
            : 'h = x · W_0 + \\frac{α}{r} (x · A) · B'}
        </Label>
        <Label x={55} y={52} tone={3} size={9.5} mono={false}>
          {effectiveMerged
            ? 'Zero added latency! The low-rank matrices are folded directly into the base weights prior to inference.'
            : 'Base weights W0 stay strictly frozen. Gradients and optimizer states exist only for A and B.'}
        </Label>
      </svg>
    </Stage>
  );
}
