import { useRef, useState } from 'react';
import { Stage, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Label, Fade, Span, Ruler, Arrow, Chip, fmt } from '../lib/svg';

const STEPS = [
  'The SLO is the constraint',
  'Model sizes vs the memory budget',
  'Where the KV cache goes',
  'How many replicas, and why',
  'Quantize: the memory-for-quality trade',
  'Hold the SLO, price the system',
];
const CAPTIONS = [
  'Start from the promise to the user, not from the hardware. Every other decision is downstream of the SLO.',
  'Weights are the fixed cost: they decide which GPUs the model can live on at all.',
  'The KV cache is the variable cost: it grows with every user and every token, and it usually sets concurrency.',
  'Replicas come from capacity planning with headroom, not from dividing traffic by a single-GPU number.',
  'Quantization buys memory and bandwidth at the cost of quality. It is a decision with a measured price tag.',
  'The finished system is a priced, defended design that meets the SLO and knows what would break it.',
];

const W = 640;
const X0 = 110;
const X1 = 622;

export default function CapstoneDesign() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  // The design knobs the learner drags while the story walks through the phases.
  const [modelB, setModelB] = useState(8); // Llama-3.1-8B or 70B
  const [ctx, setCtx] = useState(2048);
  const [quant, setQuant] = useState(16); // weight bytes

  // Llama-3.1 shapes (from perfModel): 8B: 32 layers, 8 kv heads, 128 dim. 70B: 80 layers, 8 kv heads, 128 dim.
  const params = modelB === 8 ? 8.03e9 : 70.6e9;
  const layers = modelB === 8 ? 32 : 80;
  const kvPerTok = 2 * layers * 8 * 128 * 2; // bytes, BF16 KV
  const weights = params * quant;
  const gpuMem = 80e9; // one A100/H100
  const reserve = 0.04 * gpuMem;
  const pool = gpuMem - weights - reserve;
  const perSeq = ctx * kvPerTok;
  const seqsPerGpu = Math.max(0, Math.floor(pool / perSeq));
  const fitsWeights = weights + reserve <= gpuMem;

  // Cost: token throughput scales with concurrency but is capped; keep it simple and honest.
  const throughputPerGpu = Math.min(seqsPerGpu * 22, 6500); // tok/s, roughly
  const gpuPrice = modelB === 8 ? 1.6 : 2.5; // $/hr A100 vs H100-class
  const costPerMTok = (gpuPrice / 3600 / Math.max(1, throughputPerGpu)) * 1e6;

  const g = useTweens({ w: weights / gpuMem, kv: Math.min(1, seqsPerGpu * perSeq / gpuMem), seq: Math.min(1, seqsPerGpu / 256) });

  return (
    <Stage
      rootRef={ref}
      title="Capstone · designing a system against the SLO and the memory budget"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'KV cache', 'compute', 'token', ['error', 'does not fit'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Weights', value: fmt.bytes(weights), kind: fitsWeights ? 'memory' : 'error', note: `${modelB}B at ${quant / 8}-bit` },
          { label: 'Concurrency / GPU', value: seqsPerGpu > 0 ? `${fmt.int(seqsPerGpu)} seqs` : 'OOM', kind: seqsPerGpu > 0 ? 'token' : 'error', note: `${fmt.int(ctx)} tok ctx each` },
          { label: 'Throughput / GPU', value: `${fmt.int(throughputPerGpu)} tok/s`, kind: 'compute', note: 'modeled' },
          { label: 'Cost / M tokens', value: `$${costPerMTok.toFixed(costPerMTok < 1 ? 3 : 2)}`, kind: 'compute', note: `$${gpuPrice.toFixed(2)}/hr class` },
        ]} />
      }
      controls={
        <>
          <Slider label="Model size" value={modelB} min={8} max={70} step={62} onChange={(v) => setModelB(v > 8 ? 70 : 8)} format={(v) => (v > 8 ? 'Llama-3.1-70B' : 'Llama-3.1-8B')} kind="memory" />
          <Slider label="Context per request" value={ctx} min={512} max={8192} log onChange={setCtx} format={(v) => `${fmt.int(v)} tokens`} kind="token" />
          <Slider label="Weight precision" value={quant} min={8} max={16} step={8} onChange={setQuant} format={(v) => (v === 8 ? 'INT8' : 'BF16')} kind="compute" />
        </>
      }
    >
      <svg viewBox={`0 0 ${W} 380`} role="img" aria-label="A GPU memory budget bar showing weights and KV cache, alongside the concurrency and cost the design supports, updating as model size, context and precision change.">
        {/* Phase 1: the SLO */}
        <Fade show={s.step === 0}>
          <rect x={20} y={6} width={600} height={34} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" />
          <text x={30} y={22} className="v-label bold" style={{ fontSize: 11, fill: 'var(--compute)' }}>SLO first, always:</text>
          <text x={30} y={35} className="v-label sans" style={{ fontSize: 10.5, fill: 'var(--text)' }}>p95 TTFT under 1 s, at least 30 tokens/s per user, at your real arrival rate.</text>
        </Fade>

        {/* Phase 2/3: memory budget bar */}
        <Label x={X0} y={s.step === 0 ? 78 : 20} tone={2} size={11} mono={false} bold>One GPU&apos;s memory budget (80 GB)</Label>
        <text x={X0 - 10} y={(s.step === 0 ? 78 : 20) + 44} textAnchor="end" className="v-track-label">80 GB</text>
        <rect x={X0} y={(s.step === 0 ? 78 : 20) + 26} width={X1 - X0} height={22} rx={2} fill="none" stroke="var(--rule-2)" />
        <Span x={X0} y={(s.step === 0 ? 78 : 20) + 26} w={(X1 - X0) * g.w} h={22} kind="memory" label={`weights ${(weights / 1e9).toFixed(1)} GB`} />
        <Span x={X0 + (X1 - X0) * g.w} y={(s.step === 0 ? 78 : 20) + 26} w={(X1 - X0) * g.kv} h={22} kind="memory" dim label={`KV ${((seqsPerGpu * perSeq) / 1e9).toFixed(1)} GB`} labelTone="light" />
        <Span x={X0 + (X1 - X0) * (g.w + g.kv)} y={(s.step === 0 ? 78 : 20) + 26} w={(X1 - X0) * 0.06} h={22} kind="idle" />

        {/* does it fit */}
        <Fade show={!fitsWeights || seqsPerGpu === 0}>
          <rect x={X0} y={(s.step === 0 ? 78 : 20) + 22} width={X1 - X0} height={30} rx={3} fill="url(#tps-hatch-red)" stroke="var(--error)" />
          <text x={(X0 + X1) / 2} y={(s.step === 0 ? 78 : 20) + 42} textAnchor="middle" className="v-label bold" style={{ fill: 'var(--error)', fontSize: 11.5 }}>
            {fitsWeights ? `Room left, but not for a whole sequence: OOM at ${fmt.int(ctx)} ctx` : `Does not fit: weights ${(weights / 1e9).toFixed(0)} GB > 80 GB`}
          </text>
        </Fade>
        <Label x={X1} y={(s.step === 0 ? 78 : 20) + 62} anchor="end" tone={3} size={9.5}>
          {`KV per token = 2×${layers}×8×128×2 = ${(kvPerTok / 1024).toFixed(0)} KB · per request = ${fmt.bytes(perSeq)}`}
        </Label>

        {/* Phase 4: replicas / capacity */}
        <Fade show={s.step >= 3}>
          <Label x={X0} y={200} tone={2} size={11} mono={false} bold>Capacity, with headroom</Label>
          {/* flow of requests through replicas */}
          {Array.from({ length: 4 }, (_, i) => (
            <Chip key={i} x={X0 + i * 60} y={212} text={`replica ${i + 1}`} kind="compute" w={54} size={9.5} />
          ))}
          <Arrow x1={X0 + 250} y1={223} x2={X0 + 300} y2={223} kind="comm" />
          <g className="v-move" style={{ transform: `translate(${X0 + 306}px, 0)` }}>
            <Label x={0} y={220} size={10.5} kind="token" bold>{`${seqsPerGpu} seqs each`}</Label>
            <Label x={0} y={234} tone={3} size={9.5}>{`replicas = ceil(demand / (seqs × 0.7))`}</Label>
          </g>
        </Fade>

        {/* Phase 5: quantization trade */}
        <Fade show={s.step === 4}>
          <Label x={X0} y={200} tone={2} size={11} mono={false} bold>The precision trade, priced</Label>
          <Ruler x={X0} y={252} w={260} ticks={[{ at: 0, label: 'BF16' }, { at: 0.5, label: 'INT8' }, { at: 1, label: 'INT4' }]} />
          <text x={X0} y={280} className="v-label sans" style={{ fontSize: 10, fill: 'var(--text-2)' }}>memory & bandwidth bought</text>
          <Ruler x={X0} y={316} w={260} ticks={[{ at: 0, label: 'none' }, { at: 1, label: 'quality loss' }]} />
          <text x={X0} y={344} className="v-label sans" style={{ fontSize: 10, fill: 'var(--text-2)' }}>accuracy paid for it</text>
          <g className="v-move" style={{ transform: `translate(${X0 + 260 * (quant === 8 ? 0.5 : 0)}px, 0)` }}>
            <Label x={6} y={244} size={10} kind="memory" bold>{quant === 8 ? 'you are here: INT8' : 'you are here: BF16'}</Label>
          </g>
        </Fade>

        {/* Phase 6: the defended system */}
        <Fade show={s.step === 5}>
          <rect x={20} y={170} width={600} height={196} rx={4} fill="var(--ground-2)" stroke="var(--token)" />
          <text x={34} y={192} className="v-label bold" style={{ fontSize: 12.5, fill: 'var(--token)' }}>The design, in one defensible sentence</text>
          <text x={34} y={214} className="v-label sans" style={{ fontSize: 11, fill: 'var(--text)' }}>
            {`Serve ${modelB}B at ${quant / 8}-bit on A100-class GPUs; ${fmt.int(ctx)}-token context;`}
          </text>
          <text x={34} y={230} className="v-label sans" style={{ fontSize: 11, fill: 'var(--text)' }}>
            {`${seqsPerGpu} concurrent per GPU, sized for headroom, at ~$${costPerMTok.toFixed(costPerMTok < 1 ? 3 : 2)}/M tokens,`}
          </text>
          <text x={34} y={246} className="v-label sans" style={{ fontSize: 11, fill: 'var(--text)' }}>
            meeting TTFT &amp; per-user SLO — and here is the measurement that would change our mind.
          </text>
          <text x={34} y={276} className="v-label t3" style={{ fontSize: 10 }}>Every number above is derived from memory and cost arithmetic first, then measured (Module 6).</text>
          <text x={34} y={292} className="v-label t3" style={{ fontSize: 10 }}>Weights decide which GPUs are possible; the KV cache decides how many users fit (Modules 5 &amp; 8).</text>
          <text x={34} y={308} className="v-label t3" style={{ fontSize: 10 }}>The SLO is a constraint you maximize throughput under — never a target to trade away (Module 25).</text>
        </Fade>
      </svg>
    </Stage>
  );
}