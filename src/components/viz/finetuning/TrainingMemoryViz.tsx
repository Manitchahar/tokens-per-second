import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade } from '../lib/svg';

const STEPS = [
  'The 16-byte parameter stack',
  'Weights: 2 bytes (BF16 / FP16)',
  'Gradients: 2 bytes (BF16 / FP16)',
  'Optimizer States: 12 bytes (AdamW)',
  'Activations: The context length explosion',
  'Gradient Checkpointing: The 33% compute trade',
];

const CAPTIONS = [
  'Inference needs only the weights (2 bytes/param). Mixed-precision AdamW training demands 16 bytes per parameter for static states alone, before adding a single activation.',
  'Weights live in BF16 or FP16: 2 bytes per parameter. For an 8B model, the model weights take 16 GB — exactly what fits into an entry-level inference card.',
  'During backpropagation, every trainable weight accumulates an incoming gradient: another 2 bytes per parameter in BF16. Total static memory reaches 4 B/param (32 GB for 8B).',
  'Mixed-precision AdamW stores an FP32 master weight (4 B), first momentum m (4 B), and second momentum v (4 B). That is 12 B/param for the optimizer alone! Total: 16 B/param (128 GB for 8B).',
  'Activations store intermediate forward tensors for the backward pass. Without checkpointing, they scale with batch size, sequence length, and layer depth, easily surpassing 40+ GB.',
  'Gradient checkpointing discards intermediate activations and recomputes them on the backward pass. You save 70–80% activation memory at the cost of 33% more compute.',
];

const MODELS = [
  { name: '1.5B (Qwen2.5)', params: 1.54, layers: 28, hidden: 1536 },
  { name: '8B (Llama-3.1)', params: 8.03, layers: 32, hidden: 4096 },
  { name: '70B (Llama-3.1)', params: 70.6, layers: 80, hidden: 8192 },
];

export default function TrainingMemoryViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [modelIdx, setModelIdx] = useState(1); // default 8B
  const [useCheckpointing, setUseCheckpointing] = useState(false);

  const m = MODELS[modelIdx];
  const p = m.params; // in billions

  // Static memory calculation in GB
  const wMem = p * 2; // Weights (BF16)
  const gMem = p * 2; // Gradients (BF16)
  const masterMem = p * 4; // FP32 master
  const mMem = p * 4; // FP32 momentum m
  const vMem = p * 4; // FP32 momentum v
  const optMem = masterMem + mMem + vMem; // 12 B
  const staticTotal = wMem + gMem + optMem; // 16 B

  // Activations rough estimate for batch=2, seq=2048
  // Standard: ~ 2 * 2048 * hidden * layers * 34 bytes (without ckpt)
  // Checkpointed: ~ 2 * 2048 * hidden * layers * 2 bytes
  const rawActGB = (2 * 2048 * m.hidden * m.layers * 34) / 1e9;
  const ckptActGB = (2 * 2048 * m.hidden * m.layers * 4.5) / 1e9;
  const actMem = (st === 5 ? useCheckpointing : st >= 4) ? (st === 5 && !useCheckpointing ? rawActGB : ckptActGB) : (st >= 4 ? rawActGB : 0);

  const totalVRAM =
    st === 0
      ? staticTotal
      : st === 1
      ? wMem
      : st === 2
      ? wMem + gMem
      : st === 3
      ? staticTotal
      : staticTotal + actMem;

  return (
    <Stage
      rootRef={ref}
      title="Training Memory Anatomy: 16 Bytes per Parameter"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'weights & master (FP32/BF16)'],
        ['compute', 'gradients (BF16)'],
        ['token', 'AdamW momentum (m & v)'],
        ['idle', 'activations (intermediate tensors)'],
      ]}
      controls={
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <Segmented
            options={MODELS.map((mod) => mod.name)}
            value={MODELS[modelIdx].name}
            onChange={(val) => {
              const idx = MODELS.findIndex((mod) => mod.name === val);
              if (idx !== -1) setModelIdx(idx);
            }}
          />
          {st === 5 && (
            <button
              type="button"
              className="stage-btn"
              onClick={() => setUseCheckpointing(!useCheckpointing)}
              style={{
                padding: '4px 10px',
                fontSize: 11,
                fontFamily: 'var(--mono)',
                background: useCheckpointing ? 'var(--token)' : 'rgba(255,255,255,0.06)',
                color: useCheckpointing ? '#000' : 'var(--text-1)',
                border: '1px solid var(--rule-2)',
                borderRadius: 4,
                cursor: 'pointer',
              }}
            >
              {useCheckpointing ? 'Checkpointing: ON (-78% Act)' : 'Checkpointing: OFF (Full Act)'}
            </button>
          )}
        </div>
      }
      readout={
        <Readout
          items={[
            { label: 'Model', value: `${p.toFixed(1)}B params` },
            { label: 'Inference Weights', value: `${wMem.toFixed(1)} GB (2 B/p)` },
            { label: 'Static Training', value: `${staticTotal.toFixed(1)} GB (16 B/p)` },
            { label: 'Total VRAM', value: `${totalVRAM.toFixed(1)} GB` },
          ]}
        />
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Breakdown of training memory by parameter state and activations">
        {/* Background Grid & Axes */}
        <line x1={32} y1={230} x2={608} y2={230} stroke="var(--rule-2)" strokeWidth={1} />

        {/* 16 Bytes / Param Visual Stack (Width scaled to 16 bytes = 576 px => 36 px / byte) */}
        <g className="byte-stack">
          <Label x={32} y={25} tone={1} bold size={12} mono={false}>
            Single Parameter Memory Stack (16 Bytes Total)
          </Label>

          {/* 1. Weight (2 B) */}
          <Span
            x={32}
            y={35}
            w={72}
            h={36}
            kind="memory"
            label="Weight: 2 B"
            dim={st !== 0 && st !== 1}
          />

          {/* 2. Gradient (2 B) */}
          <Span
            x={104}
            y={35}
            w={72}
            h={36}
            kind="compute"
            label="Grad: 2 B"
            dim={st !== 0 && st !== 2}
          />

          {/* 3. FP32 Master Weight (4 B) */}
          <Span
            x={176}
            y={35}
            w={144}
            h={36}
            kind="memory"
            label="FP32 Master: 4 B"
            dim={st !== 0 && st !== 3}
          />

          {/* 4. Adam m momentum (4 B) */}
          <Span
            x={320}
            y={35}
            w={144}
            h={36}
            kind="token"
            label="Adam m (1st): 4 B"
            dim={st !== 0 && st !== 3}
          />

          {/* 5. Adam v variance (4 B) */}
          <Span
            x={464}
            y={35}
            w={144}
            h={36}
            kind="token"
            label="Adam v (2nd): 4 B"
            dim={st !== 0 && st !== 3}
          />

          {/* Rulers / Sub-brackets */}
          <Label x={32 + 36} y={85} tone={3} anchor="middle" size={9.5} mono>
            BF16 W (2 B)
          </Label>
          <Label x={104 + 36} y={85} tone={3} anchor="middle" size={9.5} mono>
            BF16 G (2 B)
          </Label>
          <Label x={176 + 72} y={85} tone={3} anchor="middle" size={9.5} mono>
            FP32 Master (4 B)
          </Label>
          <Label x={320 + 72} y={85} tone={3} anchor="middle" size={9.5} mono>
            1st Momentum (4 B)
          </Label>
          <Label x={464 + 72} y={85} tone={3} anchor="middle" size={9.5} mono>
            2nd Variance (4 B)
          </Label>
        </g>

        {/* Global Memory Footprint Bar for chosen Model */}
        <g className="total-memory-bars">
          <Label x={32} y={115} tone={1} bold size={12} mono={false}>
            {m.name} Total Memory Footprint (GB)
          </Label>

          {/* Visual Bar Container (Scale max 200 GB => 576 px = ~2.88 px / GB) */}
          {(() => {
            const scale = 576 / (m.params > 50 ? 1400 : 180);
            const wW = wMem * scale;
            const wG = gMem * scale;
            const wOpt = optMem * scale;
            const wAct = actMem * scale;

            // 80 GB A100 marker
            const a100X = 32 + 80 * scale;

            return (
              <g>
                {/* 80 GB A100 Line */}
                {m.params < 50 && (
                  <g>
                    <line x1={a100X} y1={125} x2={a100X} y2={220} stroke="var(--error)" strokeDasharray="3 3" strokeWidth={1} />
                    <Label x={a100X} y={120} kind="error" size={9} mono anchor="middle">
                      80 GB A100 VRAM Limit
                    </Label>
                  </g>
                )}

                {/* Weights */}
                <Span
                  x={32}
                  y={130}
                  w={wW}
                  h={28}
                  kind="memory"
                  label={wMem >= 10 ? `${wMem.toFixed(0)} GB` : ''}
                />

                {/* Grads */}
                {(st === 0 || st >= 2) && (
                  <Span
                    x={32 + wW}
                    y={130}
                    w={wG}
                    h={28}
                    kind="compute"
                    label={gMem >= 10 ? `${gMem.toFixed(0)} GB` : ''}
                  />
                )}

                {/* Opt States */}
                {(st === 0 || st >= 3) && (
                  <Span
                    x={32 + wW + wG}
                    y={130}
                    w={wOpt}
                    h={28}
                    kind="token"
                    label={optMem >= 20 ? `Opt: ${optMem.toFixed(0)} GB` : ''}
                  />
                )}

                {/* Activations */}
                {st >= 4 && (
                  <Span
                    x={32 + wW + wG + wOpt}
                    y={130}
                    w={wAct}
                    h={28}
                    kind="idle"
                    label={wAct >= 15 ? `Act: ${actMem.toFixed(1)} GB` : ''}
                  />
                )}
              </g>
            );
          })()}
        </g>

        {/* Dynamic Detail Text / Annotations */}
        <rect
          x={32}
          y={170}
          width={576}
          height={88}
          rx={6}
          fill="rgba(0, 0, 0, 0.25)"
          stroke="var(--rule-2)"
          strokeWidth={1}
        />

        <Fade show={st === 0}>
          <Label x={46} y={192} tone={1} bold size={11.5} mono={false}>
            The 16× Multiplier: Why Training Breaks Single-GPU Budgets
          </Label>
          <Label x={46} y={210} tone={2} size={10.5} mono={false}>
            - For Llama-3.1-8B, weights alone take 16.1 GB. But training requires 128.5 GB for static parameters!
          </Label>
          <Label x={46} y={226} tone={2} size={10.5} mono={false}>
            - You cannot train an 8B model on a single 80 GB A100 GPU without distributed sharding (ZeRO / FSDP) or LoRA.
          </Label>
          <Label x={46} y={242} tone={3} size={10} mono={false}>
            Static memory is completely independent of batch size: it is locked in VRAM before the first forward pass.
          </Label>
        </Fade>

        <Fade show={st === 1}>
          <Label x={46} y={192} tone={1} bold size={11.5} mono={false}>
            1. Model Weights (2 bytes per parameter)
          </Label>
          <Label x={46} y={210} tone={2} size={10.5} mono={false}>
            - Stored in 16-bit precision (BF16 or FP16). Formula: M_weights = 2 × P bytes.
          </Label>
          <Label x={46} y={226} tone={2} size={10.5} mono={false}>
            - Read during forward pass and read again during backward pass to compute input activations.
          </Label>
        </Fade>

        <Fade show={st === 2}>
          <Label x={46} y={192} tone={1} bold size={11.5} mono={false}>
            2. Gradients (2 bytes per parameter)
          </Label>
          <Label x={46} y={210} tone={2} size={10.5} mono={false}>
            - Computed via backpropagation: dL/dW. Stored in BF16/FP16 (2 × P bytes).
          </Label>
          <Label x={46} y={226} tone={2} size={10.5} mono={false}>
            - In LoRA, gradients are computed ONLY for adapter matrices (&lt;1%), dropping gradient VRAM to near zero.
          </Label>
        </Fade>

        <Fade show={st === 3}>
          <Label x={46} y={192} tone={1} bold size={11.5} mono={false}>
            3. Optimizer States: The 12-Byte Elephant in the Room
          </Label>
          <Label x={46} y={210} tone={2} size={10.5} mono={false}>
            - FP32 Master Weights (4 B): prevents tiny gradient updates from underflowing into zero in BF16.
          </Label>
          <Label x={46} y={226} tone={2} size={10.5} mono={false}>
            - First Momentum m_t (4 B) + Second Momentum v_t (4 B): tracks running gradient statistics.
          </Label>
          <Label x={46} y={242} tone={3} size={10} mono={false}>
            Total optimizer memory = 12 bytes/param. That is 75% of all static parameter memory in training!
          </Label>
        </Fade>

        <Fade show={st === 4}>
          <Label x={46} y={192} tone={1} bold size={11.5} mono={false}>
            4. Activations: The Dynamic Context Scaler
          </Label>
          <Label x={46} y={210} tone={2} size={10.5} mono={false}>
            - Forward tensors saved for backprop: attention matrices, MLP expansions, LayerNorm inputs.
          </Label>
          <Label x={46} y={226} tone={2} size={10.5} mono={false}>
            - Formula: scales linearly with sequence length s, batch size b, and layer count L.
          </Label>
          <Label x={46} y={242} tone={3} size={10} mono={false}>
            At s = 4096, activations alone will OOM even an 80 GB GPU unless gradient checkpointing is enabled.
          </Label>
        </Fade>

        <Fade show={st === 5}>
          <Label x={46} y={192} tone={1} bold size={11.5} mono={false}>
            5. Gradient Checkpointing: Trading Compute for Survival
          </Label>
          <Label x={46} y={210} tone={2} size={10.5} mono={false}>
            - Stash only block boundary activations. Recompute internal MLP/Attention tensors during backward pass.
          </Label>
          <Label x={46} y={226} tone={2} size={10.5} mono={false}>
            - Compute cost: Normal pass = 2 fwd + 4 bwd = 6 FLOPs. With ckpt: 2 + 2 + 4 = 8 FLOPs (+33.3%).
          </Label>
          <Label x={46} y={242} tone={3} size={10} mono={false}>
            Memory reward: Activations drop from ~35 GB down to ~7 GB. Toggle the button above to observe!
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
