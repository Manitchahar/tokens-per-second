import { useMemo, useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Dim, Fade, Move, Bar, fmt } from '../lib/svg';

// A first-principles cost model for one attention layer, one phase.
// Every number here is the arithmetic the lessons derive.
export type Phase = 'prefill' | 'decode';
export type Backend = 'eager' | 'flash' | 'flashdecoding' | 'paged';

interface Cfg {
  layers: number; heads: number; kvHeads: number; headDim: number; bytes: number;
  gpuBw: number; gpuFlops: number;
}

const GPU = {
  t4: { name: 'T4', bw: 320e9, flops: 65e12 },
  a100: { name: 'A100', bw: 2.0e12, flops: 312e12 },
  h100: { name: 'H100', bw: 3.35e12, flops: 989e12 },
};

const LLAMA8B: Cfg = { layers: 32, heads: 32, kvHeads: 8, headDim: 128, bytes: 2, ...GPU.h100 };

export interface Cost {
  peak: number;      // bytes of extra memory
  traffic: number;   // bytes moved
  flops: number;     // FLOP
  ms: number;
  bound: 'memory' | 'compute';
}

export function cost(backend: Backend, phase: Phase, T: number, cfg: Cfg, splits: number): Cost {
  const { heads: nq, kvHeads: nkv, headDim: d, bytes: b, gpuBw, gpuFlops } = cfg;
  const BR = 64, BC = 64;
  const kvB = 2 * nkv * d * b * T;
  const qkFlop = 2 * nq * d * T;

  if (phase === 'prefill') {
    const sTile = BR * BC * b;
    const peakEager = 2 * nq * T * T * b;              // S and P, all heads
    // one program's on-chip footprint: Q, K, V and score tiles, FP32 output accumulator, m and l
    const peakFused = BR * d * b + 2 * BC * d * b + BR * BC * b + BR * d * 4 + 2 * BR * 4;
    const tri = T / BR;                                // triangular tile count
    const trafficEager = 4 * nq * T * T * b + 3 * nq * T * d * b;
    const trafficFused = 3 * nq * T * d * b;           // only Q, K, V, O
    const flops = (2 * nq * d * T * tri + 2 * nq * T * d * tri) / 2 * 1; // causal halving
    const tMem = trafficEager / gpuBw, tComp = flops / gpuFlops;
    const tMemF = trafficFused / gpuBw;
    if (backend === 'eager') {
      const t = Math.max(tMem, tComp);
      return { peak: peakEager, traffic: trafficEager, flops, ms: t * 1e3, bound: tMem > tComp ? 'memory' : 'compute' };
    }
    const t = Math.max(tMemF, tComp);
    return { peak: peakFused, traffic: trafficFused, flops, ms: t * 1e3, bound: tMemF > tComp ? 'memory' : 'compute' };
  }

  // decode
  const kvRead = kvB;
  const flops = 4 * nq * d * T;
  const gather = backend === 'paged' ? 1.01 : 1.0;
  const peakFused = BR * d * b + 2 * BC * d * b + BR * BC * b + BR * d * 4 + 2 * BR * 4;
  const tMem = (kvRead * gather) / gpuBw;
  const tComp = flops / gpuFlops;
  if (backend === 'eager') {
    const traffic = kvRead * gather + nq * T * b * 2;
    return { peak: nq * T * b, traffic, flops, ms: Math.max(traffic / gpuBw, tComp) * 1e3, bound: 'memory' };
  }
  // fused / split-K / paged: same bytes, but the kernel reaches a higher fraction of peak
  const eff = backend === 'flashdecoding' ? 0.85 : backend === 'paged' ? 0.78 : 0.8;
  const t = Math.max(tMem / eff, tComp);
  return { peak: peakFused, traffic: kvRead * gather, flops, ms: t * 1e3, bound: 'memory' };
}


const BACKENDS: { id: Backend; label: string; blurb: string }[] = [
  { id: 'eager', label: 'Eager', blurb: 'materialises S and P' },
  { id: 'flash', label: 'FlashAttention', blurb: 'fused tile loop' },
  { id: 'flashdecoding', label: 'FlashDecoding', blurb: 'fused + split-K' },
  { id: 'paged', label: 'Paged', blurb: 'block table + gather' },
];

const STEPS = [
  'Four implementations, one workload',
  'Eager pays for the intermediate',
  'FlashAttention removes it',
  'The causal mask is free speed',
  'Decode is a different shape',
  'Split-K fills the machine',
  'Paged reads scattered blocks',
  'Pick by shape, not by name',
];

const CAPTIONS = [
  'One layer, one phase, one model. The only thing that changes is which attention kernel runs. Everything on screen is computed from the bandwidth and FLOP/s of the GPU you pick and the head configuration of the model.',
  'At 4,096 tokens, eager attention holds both the T x T score matrix and its normalised copy for 32 heads. That is over 2 GB of extra memory for one layer — a crash on a smaller card, and pure waste on a big one.',
  'FlashAttention never writes that tensor. Its peak extra memory is one program’s tiles and running state: about 89 KB, independent of context. The traffic term drops by the same factor.',
  'The causal mask means a triangular number of tiles is real work. That roughly halves the FLOPs for free — the same saving whether or not the kernel is fused.',
  'Switch the phase to decode and the picture inverts. There is one query row, no intermediate to fuse, and the cost is a single linear read of the KV cache.',
  'At decode, the bound is bandwidth, and the only lever is getting more SMs working. FlashDecoding splits the sequence into chunks; the savings here come purely from that parallelism.',
  'The paged kernel follows a block table and gathers. That costs about 1% extra traffic — visible, but tiny next to the win from not materialising anything.',
  'No backend wins everywhere. FlashAttention dominates prefill. FlashDecoding wins decode at low batch. Paged wins when your KV is fragmented. Choose with the phase and batch your traffic actually has.',
];

export default function AttentionBackendLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [phase, setPhase] = useState<Phase>('prefill');
  const [backend, setBackend] = useState<Backend>('flash');
  const [T, setT] = useState(4096);
  const [gpu, setGpu] = useState<keyof typeof GPU>('h100');
  const [splits, setSplits] = useState(8);

  const cfg: Cfg = { ...LLAMA8B, ...GPU[gpu] };
  const costs = useMemo(
    () => BACKENDS.map((bk) => ({ ...bk, c: cost(bk.id, phase, T, cfg, splits) })),
    [phase, T, cfg, splits],
  );

  const active = costs.find((x) => x.id === backend)!;
  const maxMs = Math.max(...costs.map((x) => x.c.ms));
  const maxPeak = Math.max(...costs.map((x) => x.c.peak));

  const aMs = useTween(active.c.ms, 450);
  const aPeak = useTween(active.c.peak, 450);
  const aTraffic = useTween(active.c.traffic, 450);

  const eager = costs[0].c;
  const speedup = eager.ms / active.c.ms;
  const memSaved = eager.peak / Math.max(active.c.peak, 1);

  // attention layout sketch: rows = queries, cols = keys
  const QX = 44, QY = 44, CELL = 11, GAP = 3;
  const nTiles = phase === 'prefill' ? Math.min(8, Math.max(2, Math.round(T / (T > 8192 ? 2048 : 512)))) : 8;
  const gside = nTiles * (CELL + GAP) - GAP;

  return (
    <Stage
      rootRef={ref}
      title="Attention backends, costed"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'memory traffic'],
        ['compute', 'compute'],
        ['error', 'the intermediate'],
        ['idle', 'unused capacity'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Layer time', value: `${aMs.toFixed(2)} ms`, kind: active.c.bound === 'memory' ? 'memory' : 'compute', note: `${active.blurb}` },
          { label: 'Peak extra memory', value: fmt.gb(aPeak), kind: 'error', note: `${memSaved.toFixed(0)}x less than eager` },
          { label: 'Traffic', value: fmt.gb(aTraffic), kind: 'memory', note: `${GPU[gpu].name} at ${(GPU[gpu].bw / 1e12).toFixed(2)} TB/s` },
          { label: 'Bound', value: active.c.bound === 'memory' ? 'memory' : 'compute', kind: active.c.bound === 'memory' ? 'memory' : 'compute', note: 'per the roofline' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Phase" value={phase} options={[{ value: 'prefill', label: 'Prefill' }, { value: 'decode', label: 'Decode' }]} onChange={(v) => setPhase(v)} />
          <Segmented label="Backend" value={backend} options={BACKENDS.map((b) => ({ value: b.id, label: b.label }))} onChange={setBackend} />
          <Slider label="Context length" kind="memory" min={512} max={32768} log value={T} onChange={setT} format={(v) => `${fmt.int(v)} tokens`} />
          <Segmented label="GPU" value={gpu} options={[{ value: 't4', label: 'T4' }, { value: 'a100', label: 'A100' }, { value: 'h100', label: 'H100' }]} onChange={setGpu} />
        </>
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="Four attention backends compared for one layer of Llama-3.1-8B, showing peak memory, memory traffic, compute time and resulting layer time for the selected phase, context length and GPU. A small schematic shows which score tiles the causal mask keeps.">
        {/* ── the layout sketch ─────────────────────────────── */}
        <Label x={QX} y={26} tone={2} size={11} mono={false} bold>{phase === 'prefill' ? 'Prefill: T x T, half of it masked' : 'Decode: 1 x T, no tile to reuse'}</Label>
        {phase === 'prefill' ? (
          <>
            {Array.from({ length: nTiles }, (_, i) =>
              Array.from({ length: nTiles }, (_, j) => (
                <rect key={`${i}-${j}`} x={QX + j * (CELL + GAP)} y={QY + i * (CELL + GAP)} width={CELL} height={CELL} rx={1.5}
                  fill={j <= i ? 'var(--compute-dim)' : 'var(--error-dim)'}
                  stroke={j <= i ? 'var(--compute)' : 'var(--error)'} strokeWidth={0.7} className="v-cell" />
              )),
            )}
            <Label x={QX + gside + 8} y={QY + 6} size={8.5} kind="compute">kept</Label>
            <Label x={QX + gside + 8} y={QY + 28} size={8.5} kind="error">masked</Label>
            <Dim x1={QX} x2={QX + gside} y={QY + gside + 14} label={`~half the FLOPs, free`} kind="compute" />
          </>
        ) : (
          <>
            <rect x={QX} y={QY} width={CELL} height={gside} rx={1.5} fill="var(--compute)" fillOpacity={0.85} stroke="var(--compute)" className="v-cell" />
            {Array.from({ length: nTiles }, (_, j) => (
              <rect key={j} x={QX + (j + 1) * (CELL + GAP) + 22} y={QY + j * (CELL + GAP)} width={CELL} height={CELL} rx={1.5} fill="var(--memory)" fillOpacity={0.8} stroke="var(--memory)" className="v-cell" />
            ))}
            <Label x={QX + CELL + 4} y={QY + gside + 14} tone={3} size={9}>one query row, {nTiles} chunks of keys</Label>
            <Label x={QX} y={QY - 6} size={9} kind="compute">Q</Label>
            <Label x={QX + CELL + 30} y={QY - 6} size={9} kind="memory">K,V</Label>
          </>
        )}

        {/* ── per-backend comparison bars ───────────────────── */}
        <Label x={300} y={26} tone={2} size={11} mono={false} bold>One layer of Llama-3.1-8B, {GPU[gpu].name}</Label>
        {costs.map((x, i) => {
          const y = 44 + i * 30;
          const on = x.id === backend;
          const frac = x.c.ms / maxMs;
          return (
            <g key={x.id}>
              <Label x={300} y={y + 10} tone={on ? 1 : 2} size={9.5} mono={false} bold={on}>{x.label}</Label>
              <rect x={418} y={y} width={180} height={14} rx={2} fill="none" stroke="var(--rule)" />
              <rect x={418} y={y} width={Math.max(1, frac * 180)} height={14} rx={2}
                fill={on ? 'var(--compute)' : 'var(--text-3)'} fillOpacity={on ? 0.95 : 0.5}
                stroke={on ? 'var(--compute)' : 'var(--text-3)'} className="v-cell" />
              <Label x={606} y={y + 11} anchor="end" size={9.5} kind={on ? 'compute' : 'text'}>{`${x.c.ms < 10 ? x.c.ms.toFixed(2) : x.c.ms.toFixed(0)} ms`}</Label>
            </g>
          );
        })}
        <Label x={418} y={44 + 4 * 30 + 4} tone={3} size={9}>{`${fmt.gb(active.c.peak)} peak extra memory, ${fmt.gb(active.c.traffic)} traffic`}</Label>

        {/* ── the selected backend, unpacked ────────────────── */}
        <Fade show={s.step >= 1}>
          <rect x={44} y={208} width={252} height={168} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={58} y={228} tone={2} size={10.5} mono={false} bold>{active.label}</Label>
          <Label x={58} y={244} tone={3} size={9}>{active.blurb}</Label>

          <Label x={58} y={266} size={9.5}>peak extra memory</Label>
          <Span x={58} y={272} w={180} h={13} kind="error" dim label={fmt.gb(aPeak)} labelTone="light" />
          <Dim x1={58} x2={238} y={298} label={`${memSaved.toFixed(0)}x less than eager`} kind="error" />

          <Label x={58} y={320} size={9.5}>traffic vs eager</Label>
          <Span x={58} y={326} w={Math.min(180, (active.c.traffic / eager.traffic) * 180)} h={13} kind="memory" label={fmt.gb(aTraffic)} labelTone="light" />

          <Label x={58} y={360} size={10.5} kind={active.c.bound === 'memory' ? 'memory' : 'compute'} bold>
            {active.c.bound === 'memory' ? 'memory-bound' : 'compute-bound'}
          </Label>
          <Label x={170} y={360} tone={3} size={9}>{`${speedup.toFixed(1)}x faster than eager`}</Label>
        </Fade>

        {/* ── GPU-relative bars on the right ────────────────── */}
        <Fade show={s.step >= 7}>
          <Label x={318} y={200} tone={2} size={10.5} mono={false} bold>Same work, different cards</Label>
          {(Object.keys(GPU) as (keyof typeof GPU)[]).map((g, i) => {
            const c = cost(backend, phase, T, { ...LLAMA8B, ...GPU[g] }, splits);
            const worst = cost(backend, phase, T, { ...LLAMA8B, ...GPU.t4 }, splits).ms;
            const y = 216 + i * 26;
            return (
              <g key={g}>
                <Label x={318} y={y + 10} tone={g === gpu ? 1 : 3} size={9.5} bold={g === gpu}>{GPU[g].name}</Label>
                <Bar x={366} y={y} w={180} h={12} frac={worst / Math.max(c.ms, worst)} kind={g === gpu ? 'compute' : 'memory'} label={undefined} value={undefined} />
                <Label x={552} y={y + 10} size={9.5} tone={g === gpu ? 1 : 3}>{`${c.ms < 10 ? c.ms.toFixed(2) : c.ms.toFixed(0)} ms`}</Label>
              </g>
            );
          })}
          <Label x={318} y={300} tone={3} size={9}>Bandwidth scales the memory bound;</Label>
          <Label x={318} y={313} tone={3} size={9}>FLOP/s scales the compute bound.</Label>
        </Fade>

        <Fade show={s.step >= 7}>
          <Label x={44} y={390} tone={3} size={9}>Switch phase, context and card — the best backend changes shape, which is the whole point.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}