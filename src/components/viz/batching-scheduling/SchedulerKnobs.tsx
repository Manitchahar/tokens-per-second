import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade, fmt } from '../lib/svg';
import { decodeStep, gpuById, modelById } from '../lib/perfModel';

// The scheduler's three numbers. Each caps a different thing: how many sequences may run
// (max_num_seqs), how much HBM the KV cache may claim (gpu_memory_utilization), and how
// many tokens one step may process (max_num_batched_tokens). Tuning is the search over them.

const model = modelById('llama8b');
const gpu = gpuById('h100');

const STEPS = [
  'The three numbers',
  'Knob 1: seats',
  'Knob 2: the memory budget',
  'Divide the budget',
  'Knob 3: tokens per step',
  'A prefill is a stall',
  'Chunked prefill',
  'Setting all three',
  'Diagnose, then turn one knob',
];
const CAPTIONS = [
  'A scheduler is three numbers. Seats caps how many sequences may decode at once. The memory budget caps how many bytes the KV cache may take. The token budget caps how many tokens one step may process.',
  'max_num_seqs is the knob you reach for first, and it is not free: every seat you add makes the KV read bigger and therefore the step slower.',
  'gpu_memory_utilization is the fraction of HBM the engine may claim. The rest has to stay free for activations, the CUDA context and workspace.',
  'Free bytes divided by KV bytes per token is your real seat limit. At 90% on an H100 that is about 196 conversations of 2,048 tokens — and 49 at 8,192 tokens.',
  'max_num_batched_tokens caps the tokens processed in one step. It bounds a prefill so one enormous prompt cannot stall everyone.',
  'A 2,048-token prefill takes about 62 ms. With 64 users decoding at 14 ms per step, that is four and a half steps during which nobody gets a token.',
  'Chunked prefill splits the prompt into budget-sized chunks and interleaves them with decode steps, so the stall is paid in small slices.',
  'The three numbers together: seats must fit the budget, and the token budget is the chunk size that decides how big a stall you tolerate.',
  'Diagnose first: average batch near the seat limit with a deep queue means the step is too long. Average batch low with an empty queue means you are starved.',
];

const X0 = 104;
const X1 = 622;
const CTX = 2048;
const KV_PER_TOKEN = 131072; // 2 × 32 layers × 8 KV heads × 128 × 2 B
const KV_PER_SEQ = (CTX * KV_PER_TOKEN) / 1e9; // 0.268 GB
const WEIGHTS = (model.params * 2) / 1e9; // 16.06 GB
const ACT = 0.04; // perfModel's activation/workspace reserve, as a fraction of HBM

export default function SchedulerKnobs() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [seats, setSeats] = useState(32);
  const [util, setUtil] = useState(0.9);
  const [maxTok, setMaxTok] = useState(2048);

  useEffect(() => {
    if (s.step === 1) setSeats(32);
    if (s.step === 3) setUtil(0.9);
    if (s.step === 4) setSeats(196);
    if (s.step >= 5) { setUtil(0.9); setSeats(64); setMaxTok(2048); }
  }, [s.step]);

  const budgetGB = gpu.mem * util / 1e9 - WEIGHTS - gpu.mem * ACT / 1e9;
  const seatsFit = Math.max(0, Math.floor((budgetGB * 1e9) / KV_PER_SEQ));
  const fits = seats <= seatsFit;

  const step = decodeStep({ model, gpu, batch: Math.max(1, seats), ctx: CTX });

  // how a step's time splits, across a range of batch sizes
  const maxB = 256;
  const rows = useMemo(() => {
    const out: { b: number; tw: number; tk: number; tc: number; oh: number }[] = [];
    for (const b of [1, 8, 32, 64, 128, 196, 256]) {
      const q = decodeStep({ model, gpu, batch: b, ctx: CTX });
      out.push({ b, tw: q.tWeights, tk: q.tKv, tc: q.tCompute, oh: q.overhead });
    }
    return out;
  }, []);
  const maxT = Math.max(...rows.map((r) => r.tw + r.tk + r.tc + r.oh)) * 1000;

  // prefill stall: how many decode steps does one prefill cost?
  const prefillT = Math.max((2 * model.params * CTX) / (gpu.flops * 0.55), (model.params * 2) / (gpu.bw * 0.8)) + 0.002;
  const stalls = prefillT / step.t;
  const chunks = Math.ceil(CTX / maxTok);

  const bx = (b: number) => X0 + ((b - 1) / (maxB - 1)) * (X1 - X0);
  const bh = (ms: number) => (ms / maxT) * 66;

  return (
    <Stage
      rootRef={ref}
      title="Llama-3.1-8B on one H100: the scheduler’s three numbers"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', 'token', ['error', 'over budget']]}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'max_num_seqs', value: String(seats), kind: 'compute', note: `up to ${seats} sequences` },
            { label: 'Seats that fit', value: String(seatsFit), kind: fits ? 'token' : 'error', note: `${(util * 100).toFixed(0)}% of 80 GB at ${CTX} ctx` },
            { label: 'Decode step', value: `${(step.t * 1000).toFixed(1)} ms`, kind: 'memory', note: `${fmt.int(step.total)} tok/s` },
            { label: 'max_num_batched_tokens', value: fmt.int(maxTok), kind: 'token', note: `a ${CTX}-token prefill → ${chunks} chunk${chunks > 1 ? 's' : ''}` },
          ]}
        />
      }
      controls={
        <>
          <Slider label="max_num_seqs · seats" value={seats} min={1} max={maxB} log onChange={setSeats} format={(v) => String(v)} kind="compute" />
          <Slider label="gpu_memory_utilization" value={util} min={0.5} max={0.95} step={0.01} onChange={setUtil} format={(v) => `${(v * 100).toFixed(0)}%`} kind="memory" />
          <Slider label="max_num_batched_tokens" value={maxTok} min={256} max={8192} step={256} onChange={setMaxTok} format={(v) => fmt.int(v)} kind="token" />
        </>
      }
    >
      <svg
        viewBox="0 0 640 420"
        role="img"
        aria-label="Three scheduler limits for Llama-3.1-8B on an H100: a seat count, a KV memory budget, and a per-step token budget. Shows how the step time splits into weight read, KV read and compute as the batch grows."
      >
        <Label x={X0} y={15} size={11} kind="text" bold>
          A scheduler is three numbers
        </Label>

        {/* HBM budget */}
        <text x={X0 - 10} y={38} textAnchor="end" className="v-track-label">HBM</text>
        <rect x={X0} y={28} width={X1 - X0} height={18} rx={2} fill="none" stroke="var(--rule-2)" />
        <Span x={X0} y={28} w={((X1 - X0) * WEIGHTS) / 80} h={18} kind="compute" />
        <Span x={X0 + ((X1 - X0) * WEIGHTS) / 80} y={28} w={(X1 - X0) * Math.max(0, budgetGB) / 80} h={18} kind="memory" label={`KV budget ${budgetGB.toFixed(1)} GB`} />
        <rect x={X0 + ((X1 - X0) * (WEIGHTS + budgetGB)) / 80} y={28} w={(X1 - X0) * Math.max(0, 80 - WEIGHTS - budgetGB - gpu.mem * ACT / 1e9) / 80} height={18} rx={2} fill="url(#tps-hatch)" stroke="var(--idle)" strokeWidth={1} />
        <Label x={X0 + ((X1 - X0) * WEIGHTS) / 80 / 2} y={41} anchor="middle" size={9}>weights 16.1 GB</Label>
        <Label x={X1} y={58} anchor="end" size={9.5} tone={3}>
          {`hatched: ${(100 - util * 100).toFixed(0)}% left alone + 3.2 GB activations/workspace`}
        </Label>

        {/* seats vs seats that fit */}
        <Fade show={s.step >= 1}>
          <text x={X0 - 10} y={82} textAnchor="end" className="v-track-label">Seats</text>
          <rect x={X0} y={72} width={X1 - X0} height={16} rx={2} fill="none" stroke="var(--rule-2)" />
          <Span x={X0} y={72} w={(X1 - X0) * (seats / maxB)} h={16} kind="compute" label={`${seats} seats asked`} />
          <line x1={X0 + (X1 - X0) * (seatsFit / maxB)} x2={X0 + (X1 - X0) * (seatsFit / maxB)} y1={68} y2={92} stroke={fits ? 'var(--token)' : 'var(--error)'} strokeWidth={2} />
          <Label x={X0 + (X1 - X0) * (seatsFit / maxB)} y={64} anchor="middle" size={9.5} kind={fits ? 'token' : 'error'}>
            {`${seatsFit} fit`}
          </Label>
          <Label x={X1} y={100} anchor="end" size={9.5} tone={3}>
            {fits
              ? `${seatsFit} seats × ${CTX} tokens × ${(KV_PER_TOKEN / 1024).toFixed(0)} KiB = ${(seatsFit * KV_PER_SEQ).toFixed(0)} GB`
              : `${seats} > ${seatsFit}: the engine runs out of memory`}
          </Label>
        </Fade>

        {/* step-time split vs batch */}
        <Fade show={s.step >= 1}>
          <g>
            <text x={X0 - 10} y={126} textAnchor="end" className="v-track-label">Step</text>
            <Label x={X0} y={120} size={10}>Where a decode step’s milliseconds go, as seats grow</Label>
            <line x1={X0} x2={X1} y1={196} y2={196} className="v-rule" />
            {rows.map((r) => {
              const x = bx(r.b);
              const w = 34;
              let y = 196;
              const seg = (ms: number, kind: 'memory' | 'compute') => {
                const h = bh(ms * 1000);
                const el = <rect key={kind} x={x - w / 2} y={y - h} width={w} height={Math.max(1, h)} fill={`var(--${kind})`} opacity={kind === 'compute' ? 0.55 : 0.9} />;
                y -= h;
                return el;
              };
              return (
                <g key={r.b}>
                  {seg(r.tw * 1000, 'memory')}
                  {seg(r.tk * 1000, 'memory')}
                  {seg(r.tc * 1000, 'compute')}
                  <text x={x} y={208} textAnchor="middle" className="v-label t3" style={{ fontSize: 9 }}>{r.b}</text>
                </g>
              );
            })}
            <Label x={X1} y={226} anchor="end" size={9.5} kind="memory">blue: weights (flat) then KV (grows)</Label>
            <Label x={X0} y={226} size={9.5} kind="compute">amber: math</Label>
            <Fade show={s.step >= 2}>
              <Label x={bx(196) + 24} y={158} size={9.5} kind="error">at 196 seats the KV read</Label>
              <Label x={bx(196) + 24} y={170} size={9.5} kind="error">is 3× the weight read</Label>
            </Fade>
          </g>
        </Fade>

        {/* token budget */}
        <Fade show={s.step >= 4}>
          <text x={X0 - 10} y={258} textAnchor="end" className="v-track-label">Tokens</text>
          <Label x={X0} y={252} size={10}>max_num_batched_tokens per step</Label>
          <rect x={X0} y={262} width={X1 - X0} height={16} rx={2} fill="none" stroke="var(--rule-2)" />
          <Span x={X0} y={262} w={(X1 - X0) * (maxTok / 8192)} h={16} kind="token" label={fmt.int(maxTok)} />
          <rect x={X0 + (X1 - X0) * (maxTok / 8192)} y={262} w={(X1 - X0) * (1 - maxTok / 8192)} height={16} rx={2} fill="none" stroke="var(--rule-2)" strokeDasharray="3 3" />
          <Label x={X1} y={290} anchor="end" size={9.5} tone={3}>{`a ${CTX}-token prompt → ${chunks} chunk${chunks > 1 ? 's' : ''} of ${fmt.int(maxTok)} tokens`}</Label>
        </Fade>

        {/* prefill stall */}
        <Fade show={s.step >= 5}>
          <g>
            <Label x={X0} y={318} size={10}>What one prefill does to the people already decoding</Label>
            {[0, 1, 2, 3, 4, 5].map((i) => {
              const blocked = i < Math.min(6, stalls);
              return (
                <g key={i}>
                  <rect x={X0 + i * 88} y={330} width={80} height={18} rx={2}
                    fill={blocked ? 'var(--error)' : 'none'} stroke={blocked ? 'var(--error)' : 'var(--rule-2)'} opacity={blocked ? 0.75 : 1} strokeDasharray={blocked ? undefined : '3 3'} />
                  <text x={X0 + i * 88 + 40} y={343} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: blocked ? '#0e1116' : 'var(--text-3)' }}>
                    {blocked ? 'stalled' : `step ${i + 1}`}
                  </text>
                </g>
              );
            })}
            <Label x={X1} y={366} anchor="end" size={9.5} kind="error">
              {`one ${CTX}-token prefill = ${(prefillT * 1000).toFixed(0)} ms = ${stalls.toFixed(1)} decode steps`}
            </Label>
          </g>
        </Fade>

        {/* diagnosis */}
        <Fade show={s.step >= 8}>
          <g>
            <rect x={X0} y={380} width={X1 - X0} height={34} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
            <text x={X0 + 12} y={393} className="v-label" style={{ fontSize: 10, fill: 'var(--token)' }}>
              {fits ? `avg batch below ${seats} seats · queue 0 → starved: raise max_num_seqs or shrink KV` : `seats capped at ${seatsFit} by memory · long contexts starve the queue`}
            </text>
            <text x={X0 + 12} y={407} className="v-label" style={{ fontSize: 10, fill: 'var(--error)' }}>
              avg batch = seat limit · queue deep → step too long: cut max_num_batched_tokens or chunk the prefill
            </text>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}