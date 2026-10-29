import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade } from '../lib/svg';

const STEPS = [
  'The 4-bit barrier',
  'NF4: Information-theoretic quantile grid',
  'On-the-fly dequantization: 4-bit to BF16',
  'Double Quantization: Squeezing the scales',
  'Paged Optimizers: Taming memory spikes',
  'The milestone: 65B on a single 48 GB GPU',
];

const CAPTIONS = [
  'In standard LoRA, base weights still live in 16-bit (16 GB for 8B; 130 GB for 65B). A single GPU cannot hold a 65B base model, even if adapter gradients are tiny.',
  'Neural network weights follow a normal distribution. NF4 (4-bit NormalFloat) sets 16 bin boundaries so each bin contains exactly 1/16th (6.25%) of the Gaussian probability mass.',
  'Weights are stored on the GPU in 4-bit NF4. During the forward pass, a fused CUDA kernel reads 4-bit integers and dequantizes them to BF16 in SRAM just before matrix multiplication.',
  'Block quantization with block size 64 requires an FP32 scale every 64 weights (0.5 bits/param). Double Quantization (DQ) quantizes these scales to FP8, slashing scale overhead to 0.127 bits/param.',
  'Gradient checkpointing and activation spikes during long sequences cause transient VRAM spikes. Paged optimizers allocate AdamW states in CUDA Unified Memory, paging out to CPU RAM instead of crashing.',
  'Combining NF4, Double Quantization, and Paged Optimizers compresses an 8B model to ~4.1 GB and a 65B model to ~33 GB. A 65B model trains comfortably on one 48 GB workstation GPU.',
];

const MODELS = [
  { name: '8B Model', fp16Weights: 16.0, qloraWeights: 4.13, loraVram: 22.0, qloraTotal: 6.8, singleGpu: 'T4 / RTX 4060 (16 GB)' },
  { name: '65B Model', fp16Weights: 130.0, qloraWeights: 33.5, loraVram: 160.0, qloraTotal: 41.5, singleGpu: 'RTX 6000 / A6000 (48 GB)' },
];

export default function QloraViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [modelIdx, setModelIdx] = useState(0);
  const m = MODELS[modelIdx];

  // NF4 16 quantiles from Dettmers et al. (2023)
  const nf4Bins = [
    -1.0, -0.6961928, -0.5250731, -0.3949175,
    -0.2844414, -0.1845100, -0.0910500, 0.0,
    0.0795802, 0.1609302, 0.2461123, 0.3379152,
    0.4407098, 0.5626170, 0.7229568, 1.0,
  ];

  return (
    <Stage
      rootRef={ref}
      title="QLoRA: 4-bit NF4, Double Quantization & Paged Optimizers"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'NF4 4-bit base weights'],
        ['compute', 'dequantization to BF16 (SRAM)'],
        ['token', 'LoRA adapter (trainable BF16)'],
        ['error', 'OOM danger zone (without QLoRA)'],
      ]}
      controls={
        <Segmented
          options={MODELS.map((mod) => mod.name)}
          value={MODELS[modelIdx].name}
          onChange={(val) => {
            const idx = MODELS.findIndex((mod) => mod.name === val);
            if (idx !== -1) setModelIdx(idx);
          }}
        />
      }
      readout={
        <Readout
          items={[
            { label: 'Base Model', value: m.name },
            { label: 'FP16 Weights', value: `${m.fp16Weights} GB` },
            { label: '4-bit NF4 Weights', value: `${m.qloraWeights} GB` },
            { label: 'Hardware Required', value: m.singleGpu },
          ]}
        />
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Visual explanation of QLoRA NF4 quantization grid, double quantization, and paged memory architecture">
        {/* Step 1: NF4 Quantile Curve vs Standard Grid */}
        <g className="nf4-distribution">
          <Label x={32} y={24} tone={1} bold size={11.5} mono={false}>
            {st === 1 ? 'NF4: Quantile Grid for Normally Distributed Weights N(0, σ²)' : 'QLoRA Architecture & Quantization Grid'}
          </Label>

          {/* Normal Distribution Bell Curve Illustration */}
          <path
            d="M 40 100 Q 120 100 160 85 Q 190 70 210 35 Q 220 15 225 12 Q 230 15 240 35 Q 260 70 290 85 Q 330 100 410 100"
            fill="none"
            stroke="var(--rule-2)"
            strokeWidth={1.5}
          />

          {/* 16 Quantile Tick Lines across the bell curve */}
          {nf4Bins.map((val, idx) => {
            // Map [-1, 1] to x in [50, 400]
            const x = 225 + val * 175;
            const isZero = idx === 7;
            return (
              <line
                key={`bin-${val.toFixed(3)}`}
                x1={x}
                y1={isZero ? 25 : 45}
                x2={x}
                y2={100}
                stroke={isZero ? 'var(--token)' : st === 1 ? 'var(--compute)' : 'var(--rule-2)'}
                strokeWidth={isZero ? 1.5 : 1}
                strokeDasharray={st === 1 ? undefined : '2 2'}
              />
            );
          })}

          <Label x={225} y={112} tone={st === 1 ? 1 : 3} size={9} mono anchor="middle">
            0.0 (Exact Zero)
          </Label>
          <Label x={50} y={112} tone={3} size={8.5} mono anchor="start">
            -1.0
          </Label>
          <Label x={400} y={112} tone={3} size={8.5} mono anchor="end">
            +1.0
          </Label>
          <Label x={425} y={65} tone={2} size={9.5} mono={false}>
            16 bins, each exactly
          </Label>
          <Label x={425} y={80} kind="token" bold size={11} mono>
            6.25% of total weights
          </Label>
        </g>

        {/* 3 Pillars of QLoRA Diagram */}
        <g className="qlora-three-pillars">
          {/* Pillar 1: NF4 Storage */}
          <g>
            <rect
              x={32}
              y={130}
              width={180}
              height={70}
              rx={4}
              fill="rgba(46, 144, 250, 0.08)"
              stroke={st === 2 ? 'var(--compute)' : 'var(--memory)'}
              strokeWidth={st === 2 ? 1.5 : 1}
            />
            <Label x={44} y={150} tone={1} bold size={11} mono={false}>
              1. NF4 Storage (4-bit)
            </Label>
            <Label x={44} y={166} tone={2} size={9} mono>
              Storage: 0.5 bytes / param
            </Label>
            <Label x={44} y={180} tone={3} size={8.5} mono>
              Dequantize in SRAM to BF16
            </Label>
          </g>

          {/* Pillar 2: Double Quantization */}
          <g>
            <rect
              x={228}
              y={130}
              width={180}
              height={70}
              rx={4}
              fill="rgba(247, 144, 9, 0.08)"
              stroke={st === 3 ? 'var(--compute)' : 'var(--rule-2)'}
              strokeWidth={st === 3 ? 1.5 : 1}
            />
            <Label x={240} y={150} tone={1} bold size={11} mono={false}>
              2. Double Quant (DQ)
            </Label>
            <Label x={240} y={166} tone={2} size={9} mono>
              Scales: 0.5 b → 0.127 b/p
            </Label>
            <Label x={240} y={180} tone={3} size={8.5} mono>
              Saves ~3 GB on a 65B model
            </Label>
          </g>

          {/* Pillar 3: Paged Optimizers */}
          <g>
            <rect
              x={424}
              y={130}
              width={184}
              height={70}
              rx={4}
              fill="rgba(18, 183, 106, 0.08)"
              stroke={st === 4 ? 'var(--token)' : 'var(--rule-2)'}
              strokeWidth={st === 4 ? 1.5 : 1}
            />
            <Label x={436} y={150} tone={1} bold size={11} mono={false}>
              3. Paged Optimizers
            </Label>
            <Label x={436} y={166} tone={2} size={9} mono>
              CUDA Unified Memory
            </Label>
            <Label x={436} y={180} tone={3} size={8.5} mono>
              Spill AdamW spikes to CPU RAM
            </Label>
          </g>
        </g>

        {/* Lower Detail Card & Memory Footprint Bar */}
        <rect
          x={32}
          y={212}
          width={576}
          height={92}
          rx={6}
          fill="rgba(0, 0, 0, 0.25)"
          stroke="var(--rule-2)"
          strokeWidth={1}
        />

        <Fade show={st === 0}>
          <Label x={46} y={232} tone={1} bold size={11.5} mono={false}>
            The Hardware Wall: Why LoRA Alone Is Not Enough for Single GPUs
          </Label>
          <Label x={46} y={250} tone={2} size={10.5} mono={false}>
            - Standard LoRA freezes base weights, but leaves them in 16-bit precision.
          </Label>
          <Label x={46} y={268} tone={2} size={10.5} mono={false}>
            - For LLaMA-65B, 16-bit weights occupy 130 GB — impossible to load on any single 80 GB or 48 GB GPU.
          </Label>
          <Label x={46} y={286} tone={3} size={10} mono={false}>
            QLoRA solves this by compressing the frozen base weights to 4-bit while training 16-bit LoRA adapters.
          </Label>
        </Fade>

        <Fade show={st === 1}>
          <Label x={46} y={232} tone={1} bold size={11.5} mono={false}>
            NormalFloat4 (NF4): Why Uniform FP4 Quantization Fails
          </Label>
          <Label x={46} y={250} tone={2} size={10.5} mono={false}>
            - Pretrained weights follow a bell curve: 95% of weights cluster within [-2σ, +2σ].
          </Label>
          <Label x={46} y={268} tone={2} size={10.5} mono={false}>
            - Standard integer or FP4 quantizers space bins uniformly, allocating empty bins to the tails.
          </Label>
          <Label x={46} y={286} tone={3} size={10} mono={false}>
            NF4 sets quantile thresholds so every one of the 16 bins carries equal information (6.25% of weights).
          </Label>
        </Fade>

        <Fade show={st === 2}>
          <Label x={46} y={232} tone={1} bold size={11.5} mono={false}>
            Fused Dequantization: Zero Loss of Compute Precision
          </Label>
          <Label x={46} y={250} tone={2} size={10.5} mono={false}>
            - Base weights are NEVER stored in 16-bit in VRAM. They remain 4-bit in High Bandwidth Memory.
          </Label>
          <Label x={46} y={268} tone={2} size={10.5} mono={false}>
            - A fused kernel loads 4-bit nibbles directly into on-chip SRAM, dequantizes to BF16, and performs the GEMM.
          </Label>
          <Label x={46} y={286} tone={3} size={10} mono={false}>
            Math executes in full BF16 tensor core precision, preventing quantization drift during forward/backward.
          </Label>
        </Fade>

        <Fade show={st === 3}>
          <Label x={46} y={232} tone={1} bold size={11.5} mono={false}>
            Double Quantization (DQ): Eliminating the Quantization Scale Tax
          </Label>
          <Label x={46} y={250} tone={2} size={10.5} mono={false}>
            - Standard block-wise quantization (block size 64) requires a 32-bit scale constant every 64 parameters.
          </Label>
          <Label x={46} y={268} tone={2} size={10.5} mono={false}>
            - Scale memory overhead = 32 / 64 = 0.5 bits per parameter (a 12.5% memory penalty on 4-bit weights!).
          </Label>
          <Label x={46} y={286} tone={3} size={10} mono={false}>
            DQ quantizes the 32-bit scales into 8-bit FP with block size 256, dropping scale tax to 0.127 bits/param.
          </Label>
        </Fade>

        <Fade show={st === 4}>
          <Label x={46} y={232} tone={1} bold size={11.5} mono={false}>
            Paged Optimizers: Eliminating Transient OOM Crashes
          </Label>
          <Label x={46} y={250} tone={2} size={10.5} mono={false}>
            - Training runs frequently crash from sudden memory spikes during long-prompt batches or checkpointing.
          </Label>
          <Label x={46} y={268} tone={2} size={10.5} mono={false}>
            - Paged AdamW relies on CUDA unified virtual memory to page inactive optimizer states to host RAM over PCIe.
          </Label>
          <Label x={46} y={286} tone={3} size={10} mono={false}>
            The GPU survives transient activation peaks without throwing CUDA out-of-memory errors.
          </Label>
        </Fade>

        <Fade show={st === 5}>
          <Label x={46} y={232} tone={1} bold size={11.5} mono={false}>
            The Breakthrough: Democratizing LLM Post-Training
          </Label>
          <Label x={46} y={250} tone={2} size={10.5} mono={false}>
            - For an 8B model: weights take ~4.13 GB; full training fits in &lt; 7 GB on a single 16 GB T4 or RTX 4060.
          </Label>
          <Label x={46} y={268} tone={2} size={10.5} mono={false}>
            - For a 65B model: weights take ~33.5 GB; full training fits in ~41.5 GB on a single 48 GB RTX 6000 Ada.
          </Label>
          <Label x={46} y={286} tone={3} size={10} mono={false}>
            Dettmers et al. proved 4-bit QLoRA matched 16-bit full finetuning across 26 academic and chatbot benchmarks.
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
