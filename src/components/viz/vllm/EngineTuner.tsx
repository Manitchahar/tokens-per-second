import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Span, Label, Fade, fmt } from '../lib/svg';
import { decodeStep, modelById, gpuById } from '../lib/perfModel';
import { launchPlan } from './engine';

const STEPS = [
  'Read the log first',
  'max_model_len: promise less',
  'gpu_memory_utilization: leave room',
  'max_num_seqs: the concurrency cap',
  'max_num_batched_tokens: share a step',
  'Find the operating point',
];
const CAPTIONS = [
  'Before touching a flag, read what the server already told you: the size of the weights, the KV cache size in tokens, and the concurrency that implies. Every flag below moves one of those numbers.',
  'max_model_len is a promise about the longest request you accept. Lowering it shrinks the worst-case KV footprint of every request, which multiplies how many of them fit at once.',
  'gpu_memory_utilization decides the size of the whole budget. Push it up and the KV pool grows, until it collides with the CUDA context and you run out of memory at startup instead.',
  'max_num_seqs caps how many sequences the scheduler runs in one iteration. It is the direct lever on per-step latency, and the red line is where the KV pool runs dry at this context length.',
  'max_num_batched_tokens is how many tokens one iteration may process. A prompt bigger than this is split across iterations, which is what stops one long prompt from stalling everybody else’s decode.',
  'Put the cap where your latency target and your memory agree. Left of the SLO line you are wasting throughput; right of it you are breaking the promise you made to users.',
];

const model = modelById('llama8b');

/** The assumed workload: 2,048 tokens of context per user, mid-conversation. */
const CTX = 2048;
/** "Each user should feel at least this fast" — the kind of SLO Module 06 defined. */
const SLO_SPEED = 30;
/** A long prompt, for the chunking panel. */
const LONG_PROMPT = 4096;

const L = 96;
const R = 620;
const PLOT_X1 = 548;
const PLOT_Y0 = 176;
const PLOT_Y1 = 280;
const MAX_BATCH = 512;

const cx = (b: number) => L + (Math.log10(b) / Math.log10(MAX_BATCH)) * (PLOT_X1 - L);
const cySpeed = (v: number) => PLOT_Y1 - Math.min(1, v / 150) * (PLOT_Y1 - PLOT_Y0);
const cyTotal = (v: number) => PLOT_Y1 - Math.min(1, v / 14000) * (PLOT_Y1 - PLOT_Y0);

export default function EngineTuner() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [util, setUtil] = useState(0.9);
  const [maxLen, setMaxLen] = useState(8192);
  const [maxSeqs, setMaxSeqs] = useState(64);
  const [budget, setBudget] = useState(2048);

  const gpu = gpuById('h100');
  const plan = launchPlan(model, gpu, util);
  const kvPerToken = plan.poolTokens > 0 ? plan.pool / plan.poolTokens : 0;

  const curve = useMemo(() => {
    const pts: { b: number; speed: number; total: number; fits: boolean }[] = [];
    for (let b = 1; b <= MAX_BATCH; b = Math.ceil(b * 1.15)) {
      const r = decodeStep({ model, gpu, batch: b, ctx: CTX });
      pts.push({ b, speed: r.perUser, total: r.total, fits: r.fits });
    }
    return pts;
  }, [model, gpu]);

  const maxFit = (curve.find((p) => !p.fits)?.b ?? MAX_BATCH + 1) - 1;
  const worstCase = Math.floor(plan.poolTokens / Math.max(1, maxLen));
  const cap = Math.max(1, Math.min(maxSeqs, maxFit));
  const here = decodeStep({ model, gpu, batch: cap, ctx: CTX });
  const sloBatch = [...curve].reverse().find((p) => p.speed >= SLO_SPEED)?.b ?? 1;

  const iters = Math.ceil(LONG_PROMPT / budget);
  const drawn = Math.min(iters, 17);
  const slot = (PLOT_X1 + 40 - L) / Math.max(1, drawn * 2 - 1);

  const barW = (v: number) => (v / gpu.mem) * (R - L);
  const budgetW = useTween(barW(plan.budget), 500);
  const weightsW = useTween(barW(plan.weights), 500);
  const poolW = useTween(barW(plan.pool), 500);
  const reserveW = useTween(barW(plan.reserve), 500);
  const markerX = useTween(cx(cap), 500);
  const kvUse = plan.pool > 0 ? (cap * CTX * kvPerToken) / plan.pool : 0;

  return (
    <Stage
      rootRef={ref}
      title="Llama-3.1-8B on one H100: four flags, four consequences"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', ['idle', 'waiting'], ['error', 'out of memory']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'KV cache pool', value: plan.pool > 0 ? `${(plan.pool / 1e9).toFixed(1)} GB` : '—', kind: plan.pool > 0 ? 'memory' : 'error', note: `${fmt.si(plan.poolTokens)} tokens` },
          { label: 'Worst-case concurrency', value: fmt.int(worstCase), kind: 'memory', note: `pool ÷ max_model_len (${fmt.int(maxLen)})` },
          { label: 'Per-user at the cap', value: fmt.int(here.perUser), kind: 'token', note: 'tokens/s, memory-bound' },
          { label: 'Throughput at the cap', value: `${fmt.int(here.total)} tok/s`, kind: 'compute', note: `${fmt.int(cap)} running · KV ${(kvUse * 100).toFixed(0)}%` },
        ]} />
      }
      controls={
        <>
          <Slider label="gpu_memory_utilization" value={util} min={0.2} max={0.98} step={0.01} onChange={setUtil} format={(v) => v.toFixed(2)} kind="memory" />
          <Slider label="max_model_len" value={maxLen} min={1024} max={32768} log onChange={setMaxLen} format={(v) => fmt.int(v)} kind="memory" />
          <Slider label="max_num_seqs" value={maxSeqs} min={1} max={512} log onChange={setMaxSeqs} format={(v) => fmt.int(v)} kind="token" />
          <Slider label="max_num_batched_tokens" value={budget} min={256} max={8192} log onChange={setBudget} format={(v) => fmt.int(v)} kind="compute" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A tuning panel: a memory budget bar splitting an H100 into weights, a KV cache pool and a reserve; a chart where per-user speed falls and total throughput rises against batch size, with the KV memory limit and an SLO line marked; and a panel showing how a 4,096-token prompt is split across engine iterations.">
        {/* ── 1. the memory budget ── */}
        <Label x={16} y={20} tone={2} size={10.5} mono={false} bold>1. The memory budget</Label>
        <rect x={L} y={30} width={R - L} height={24} rx={2} fill="none" stroke="var(--rule)" />
        <Span x={L} y={30} w={weightsW} h={24} kind="memory" label="weights" />
        <Span x={L + weightsW} y={30} w={Math.max(0, poolW)} h={24} kind="memory" dim label="KV pool" labelTone="light" />
        <Span x={L + weightsW + Math.max(0, poolW)} y={30} w={Math.max(0, Math.min(reserveW, R - L - weightsW - Math.max(0, poolW)))} h={24} kind="idle" />
        {[0, 20, 40, 60, 80].map((v) => (
          <Label key={v} x={L + (v / 80) * (R - L)} y={68} anchor="middle" tone={3} size={9}>{v}</Label>
        ))}
        <Label x={L} y={86} tone={2} size={9.5}>{`weights ${(plan.weights / 1e9).toFixed(1)} GB · KV pool ${(plan.pool / 1e9).toFixed(1)} GB · left over ${((gpu.mem - plan.budget) / 1e9).toFixed(1)} GB`}</Label>
        <Fade show={s.step >= 1}>
          <Label x={L} y={102} tone={1} size={9.5}>{`worst case per request ${fmt.int(maxLen)} × ${(kvPerToken / 1024).toFixed(0)} KB = ${(maxLen * kvPerToken / 1e9).toFixed(2)} GB → ${fmt.int(worstCase)} at once`}</Label>
        </Fade>
        <Fade show={s.step >= 2 && plan.fits}>
          <line x1={L + budgetW} x2={L + budgetW} y1={26} y2={58} stroke="var(--text-2)" strokeWidth={1.2} />
          <Label x={Math.min(R - 46, L + budgetW + 5)} y={24} tone={3} size={9}>{`${util.toFixed(2)} × 80 GB = ${(plan.budget / 1e9).toFixed(0)} GB`}</Label>
        </Fade>
        <Fade show={!plan.fits}>
          <rect x={L} y={26} width={R - L} height={32} rx={3} fill="url(#tps-hatch-red)" stroke="var(--error)" />
          <Label x={(L + R) / 2} y={46} anchor="middle" size={11} kind="error" bold>{`${(plan.weights / 1e9).toFixed(0)} GB of weights do not fit in a ${(plan.budget / 1e9).toFixed(0)} GB budget`}</Label>
        </Fade>

        <line x1={16} x2={624} y1={116} y2={116} className="v-rule faint" />

        {/* ── 2. the two curves ── */}
        <Label x={16} y={134} tone={2} size={10.5} mono={false} bold>2. Throughput up, per-user speed down (2,048 tokens of context)</Label>
        <Fade show={s.step >= 5}>
          <Label x={16} y={148} size={9.5} kind={cap <= sloBatch ? 'token' : 'error'}>
            {cap <= sloBatch
              ? `max_num_seqs ${fmt.int(maxSeqs)} holds the SLO: ${fmt.int(here.perUser)} tok/s each, ${fmt.int(here.total)} tok/s total.`
              : `max_num_seqs ${fmt.int(maxSeqs)} breaks the ${SLO_SPEED} tok/s SLO — the last safe cap is ${fmt.int(sloBatch)}.`}
          </Label>
        </Fade>
        <Label x={L} y={162} tone={3} size={9}>per-user tok/s</Label>
        <Label x={PLOT_X1} y={162} anchor="end" tone={3} size={9}>total tok/s</Label>
        <line x1={L} x2={PLOT_X1} y1={PLOT_Y1} y2={PLOT_Y1} className="v-rule" />
        <line x1={L} x2={L} y1={PLOT_Y0} y2={PLOT_Y1} className="v-rule" />
        {[150, 100, 50, 0].map((v) => (
          <Label key={v} x={L - 6} y={cySpeed(v) + 3} anchor="end" tone={3} size={9}>{v}</Label>
        ))}
        {[10, 5, 0].map((v) => (
          <Label key={`t${v}`} x={PLOT_X1 + 6} y={cyTotal(v * 1000) + 3} tone={3} size={9}>{v ? `${v}k` : '0'}</Label>
        ))}
        {[1, 8, 64, 512].map((v) => (
          <Label key={v} x={cx(v)} y={PLOT_Y1 + 13} anchor="middle" tone={3} size={9}>{v}</Label>
        ))}
        <Label x={(L + PLOT_X1) / 2} y={PLOT_Y1 + 26} anchor="middle" tone={3} size={9}>sequences running in one engine step</Label>
        <Fade show={s.step >= 3}>
          <path d={curve.map((p, i) => `${i ? 'L' : 'M'}${cx(p.b).toFixed(1)},${cySpeed(p.speed).toFixed(1)}`).join(' ')} fill="none" stroke="var(--token)" strokeWidth={1.8} />
          <path d={curve.map((p, i) => `${i ? 'L' : 'M'}${cx(p.b).toFixed(1)},${cyTotal(p.total).toFixed(1)}`).join(' ')} fill="none" stroke="var(--compute)" strokeWidth={1.8} />
          <Label x={cx(3)} y={cySpeed(curve[0].speed) - 10} size={9.5} kind="token">per-user speed</Label>
          <Label x={cx(maxFit) - 6} y={cyTotal(decodeStep({ model, gpu, batch: maxFit, ctx: CTX }).total) - 10} anchor="end" size={9.5} kind="compute">total throughput</Label>
          <line x1={L} x2={PLOT_X1} y1={cySpeed(SLO_SPEED)} y2={cySpeed(SLO_SPEED)} stroke="var(--error)" strokeWidth={1} strokeDasharray="4 4" />
          <Label x={L + 4} y={cySpeed(SLO_SPEED) - 5} size={9} kind="error">{`your SLO: ${SLO_SPEED} tok/s per user`}</Label>
          <line x1={cx(maxFit)} x2={cx(maxFit)} y1={PLOT_Y0 - 4} y2={PLOT_Y1} stroke="var(--error)" strokeWidth={1.4} strokeDasharray="3 3" />
          <Label x={cx(maxFit) - 5} y={PLOT_Y0 + 10} anchor="end" size={9} kind="error">{`no room past ${fmt.int(maxFit)}`}</Label>
          <line x1={markerX} x2={markerX} y1={PLOT_Y0 - 4} y2={PLOT_Y1} stroke="var(--memory)" strokeWidth={2} />
          <Label x={markerX - 6} y={PLOT_Y1 - 7} anchor="end" size={9.5} kind="memory" bold>{`max_num_seqs ${fmt.int(maxSeqs)}`}</Label>
          <Fade show={s.step >= 5}>
            <circle cx={markerX} cy={cySpeed(here.perUser)} r={4.5} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={2} />
          </Fade>
        </Fade>

        <line x1={16} x2={624} y1={318} y2={318} className="v-rule faint" />

        {/* ── 3. one iteration's token budget ── */}
        <Label x={16} y={336} tone={2} size={10.5} mono={false} bold>3. A {fmt.int(LONG_PROMPT)}-token prompt vs max_num_batched_tokens</Label>
        <Fade show={s.step >= 4}>
          <text x={L - 12} y={357} textAnchor="end" className="v-track-label">step</text>
          {Array.from({ length: drawn }, (_, i) => (
            <g key={i}>
              <rect x={L + i * slot * 2} y={347} width={Math.max(2, slot)} height={12} rx={2} fill="var(--compute)" />
              {i < drawn - 1 && <rect x={L + i * slot * 2 + slot * 1.3} y={351} width={Math.max(1.5, slot * 0.45)} height={4} rx={1} fill="var(--token)" />}
            </g>
          ))}
          {iters > drawn && <Label x={L + drawn * slot * 2 + 4} y={357} tone={3} size={9}>{`+${iters - drawn}`}</Label>}
          <Label x={L} y={374} tone={3} size={9.5}>{`${fmt.int(LONG_PROMPT)} ÷ ${fmt.int(budget)} = ${iters} iteration${iters > 1 ? 's' : ''} · amber is this prompt, green are decode steps in between`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}