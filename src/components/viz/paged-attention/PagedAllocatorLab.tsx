import { useMemo, useRef, useState } from 'react';
import { Stage, Readout, Slider, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';
import { KV_BYTES_PER_TOKEN, blocksNeeded } from './model';

// ── The allocator you are about to build, running live ──────────────────────

interface Seq { id: number; prompt: number; length: number; served: number; blocks: number[]; alive: boolean }

export interface SimState {
  seqs: Seq[];
  pool: number; // number of blocks
  blockSize: number;
  free: number;
  maxLen: number;
  preempted: number;
  recomputed: number;
  swaps: number;
  queued: number;
}

const KV = KV_BYTES_PER_TOKEN;

/** Contiguous: each live sequence holds a reservation of `maxLen` tokens. */
export function contiguousUsed(state: SimState) {
  return state.seqs.filter((s) => s.alive).length * state.maxLen;
}

/** Paged: each live sequence holds ceil(tokens / blockSize) blocks. */
export function pagedUsed(state: SimState) {
  return state.seqs.filter((s) => s.alive).reduce((a, s) => a + blocksNeeded(s.prompt + s.served, state.blockSize) * state.blockSize, 0);
}

/** Grow one step; admit if free, else apply the policy. Returns a new state. */
export function stepSim(state: SimState): SimState {
  const need = (s: Seq) => blocksNeeded(s.prompt + s.served, state.blockSize);
  let free = state.free;
  const seqs = state.seqs.map((s) => ({ ...s }));
  let preempted = state.preempted, recomputed = state.recomputed, swaps = state.swaps, queued = state.queued;

  for (const s of seqs) {
    if (!s.alive) continue;
    if (s.served >= s.length) { // finish: return its blocks
      free += s.blocks.length;
      s.blocks = [];
      s.alive = false;
      continue;
    }
    s.served += 1;
    while (need(s) > s.blocks.length) {
      if (free > 0) { free -= 1; s.blocks.push(1); }
      else {
        // preempt the newest *other* live sequence
        const victim = seqs.filter((v) => v.alive && v.id !== s.id).sort((a, b) => b.served - a.served)[0];
        if (!victim) { queued += 1; break; }
        free += victim.blocks.length;
        victim.blocks = [];
        victim.alive = false;
        victim.served = 0;
        victim.blocks = [];
        preempted += 1; recomputed += 1;
        free -= 1; s.blocks.push(1); // room for the writer
      }
    }
  }
  return { ...state, seqs, free, preempted, recomputed, swaps, queued };
}

export function initSim(pool: number, blockSize: number, maxLen: number, n: number, lengths: number[]): SimState {
  const seqs: Seq[] = Array.from({ length: n }, (_, i) => ({
    id: i, prompt: 32, length: lengths[i] ?? 96, served: 0, blocks: [], alive: true,
  }));
  // give the first ones a head start so the pool has partial use
  for (const s of seqs) s.blocks = Array.from({ length: blocksNeeded(32, blockSize) }, () => 1);
  const used = seqs.reduce((a, s) => a + s.blocks.length, 0);
  return { seqs, pool, blockSize, free: Math.max(0, pool - used), maxLen, preempted: 0, recomputed: 0, swaps: 0, queued: 0 };
}

export const DEFAULT_LENGTHS = [96, 64, 140, 48, 180, 72, 110, 56];

const STEPS = [
  'Start: eight requests, one pool',
  'Grow them one token at a time',
  'Watch the two allocators diverge',
  'Try a smaller max_len',
  'Try a bigger block',
  'Your turn',
];

const CAPTIONS = [
  'Eight requests share a pool of 64 blocks. Every one of them will keep growing. Below, two allocators serve the same workload side by side.',
  'Press play, or step, and watch the two used-memory bars move at very different speeds as the requests generate tokens.',
  'The red bar is a contiguous allocator: each request reserved max_len tokens the moment it arrived. The blue bar is paged: it holds only whole blocks that are actually in use.',
  'Halve max_len and the red bar halves too. That is the whole benefit of the setting — and long requests would now be rejected.',
  'Raise the block size and the blue bar grows slightly, because each request wastes the empty tail of its last block.',
  'Now it is your turn: set the sliders, run it, and answer the questions on the lab page.',
];

const W = 640;
const X0 = 92;
const XW = 470;

export default function PagedAllocatorLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [pool, setPool] = useState(64);
  const [blockSize, setBlockSize] = useState(16);
  const [maxLen, setMaxLen] = useState(256);
  const [tokens, setTokens] = useState(1);

  const capacity = pool * blockSize; // tokens the pool can hold if perfectly packed
  const scale = Math.max(maxLen * DEFAULT_LENGTHS.length, capacity * 1.2);

  const state = useMemo(() => {
    let st = initSim(pool, blockSize, maxLen, DEFAULT_LENGTHS.length, DEFAULT_LENGTHS);
    for (let i = 0; i < tokens; i++) st = stepSim(st);
    return st;
  }, [pool, blockSize, maxLen, tokens]);

  const paged = pagedUsed(state);
  const contig = contiguousUsed(state);
  const live = state.seqs.filter((q) => q.alive).length;
  const width = (t: number) => Math.min(XW, (t / scale) * XW);

  const perSeq = state.seqs.map((q) => {
    const used = blocksNeeded(q.prompt + q.served, blockSize) * blockSize;
    return { ...q, used, tail: used - (q.prompt + q.served) };
  });

  return (
    <Stage
      rootRef={ref}
      title="Paged allocator vs contiguous allocator, on the same workload"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['memory', 'paged: in use'], ['error', 'contiguous: reserved'], ['idle', 'block tail waste'], ['comm', 'copied on evict']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Paged KV in use', value: `${fmt.int(paged)} tok`, kind: 'memory', note: `${(paged * KV / 1e6).toFixed(0)} MB, ${live} requests alive` },
          { label: 'Contiguous reserved', value: `${fmt.int(contig)} tok`, kind: 'error', note: `${(contig * KV / 1e6).toFixed(0)} MB at max_len ${maxLen}` },
          { label: 'Advantage', value: `${(contig / Math.max(1, paged)).toFixed(2)}×`, kind: 'text', note: contig > paged ? 'more memory reserved' : 'contiguous has won' },
          { label: 'Preemptions so far', value: `${state.preempted}`, kind: 'comm', note: `${fmt.int(tokens)} decode steps run` },
        ]} />
      }
      controls={
        <>
          <Slider label="Decode steps run" value={tokens} min={0} max={400} step={1} onChange={setTokens} format={(v) => `${v}`} kind="token" />
          <Slider label="Pool size (blocks)" value={pool} min={16} max={160} step={8} onChange={setPool} format={(v) => `${v} blk · ${fmt.int(v * blockSize)} tok`} kind="memory" />
          <Slider label="Block size (tokens)" value={blockSize} min={4} max={64} step={4} onChange={setBlockSize} format={(v) => `${v}`} kind="memory" />
          <Slider label="max_len (contiguous reservation)" value={maxLen} min={64} max={512} step={64} onChange={setMaxLen} format={(v) => `${v} tok`} kind="error" />
        </>
      }
    >
      <svg viewBox={`0 0 ${W} 340`} role="img" aria-label="Two memory bars showing the same eight decoding requests: a contiguous allocator reserving the maximum length per request in red, and a paged allocator holding only whole blocks actually in use in blue, with a block grid below showing per-request blocks and their empty tails.">
        {/* live requests still decoding */}
        <Label x={X0} y={22} tone={2} size={10.5} mono={false} bold>{`${live} of ${DEFAULT_LENGTHS.length} requests alive after ${fmt.int(tokens)} decode steps`}</Label>

        {/* contiguous bar */}
        <Label x={X0} y={46} tone={2} size={10.5} mono={false}>Contiguous · reserved</Label>
        <rect x={X0} y={52} width={XW} height={18} rx={2} fill="none" stroke="var(--rule-2)" />
        <Span x={X0} y={52} w={width(contig)} h={18} kind="error" dim label={`${fmt.int(contig)} tokens`} labelTone="light" />

        {/* paged bar */}
        <Label x={X0} y={86} tone={2} size={10.5} mono={false}>Paged · in use</Label>
        <rect x={X0} y={92} width={XW} height={18} rx={2} fill="none" stroke="var(--rule-2)" />
        <Span x={X0} y={92} w={width(paged)} h={18} kind="memory" label={`${fmt.int(paged)} tokens`} />

        {/* the gap, labelled */}
        <Dim x1={X0 + width(paged)} x2={X0 + width(contig)} y={126} label={`${((contig / Math.max(1, paged) - 1) * 100).toFixed(0)}% overhead`} kind="error" />
        <Label x={X0} y={146} tone={3} size={9.5}>{`pool capacity ${fmt.int(capacity)} tokens · ${fmt.int(state.free)} blocks free · ${fmt.int(pool - state.free)}/${pool} in use`}</Label>

        {/* per-request block view */}
        <Label x={X0} y={170} tone={2} size={10.5} mono={false} bold>Each request, as blocks (blue = used, hatched = empty tail)</Label>
        {perSeq.map((q, i) => {
          const y = 180 + i * 19;
          const real = q.prompt + q.served;
          const blocks = blocksNeeded(real, blockSize);
          const w = (blocks * blockSize / scale) * XW;
          const usedW = (real / scale) * XW;
          return (
            <g key={q.id}>
              <Label x={X0 - 10} y={y + 9} anchor="end" tone={q.alive ? 2 : 3} size={9}>{`${q.id + 1}`}</Label>
              <rect x={X0} y={y} width={Math.max(2, w)} height={12} rx={2} fill="none" stroke="var(--rule-2)" />
              {q.alive ? (
                <>
                  <rect x={X0} y={y} width={Math.min(w, usedW)} height={12} rx={2} fill="var(--memory)" />
                  {q.tail > 0 && <rect x={X0 + usedW} y={y} width={Math.max(0, w - usedW)} height={12} fill="url(#tps-hatch)" stroke="var(--error)" strokeWidth={0.8} />}
                </>
              ) : (
                <text x={X0 + 4} y={y + 9.5} className="v-label" style={{ fontSize: 8.5, fill: 'var(--text-3)' }}>done</text>
              )}
            </g>
          );
        })}

        <Fade show={s.step >= 4}>
          <Label x={X0 + 200} y={170} tone={2} size={9.5}>{"bigger blocks → a fatter tail on every request"}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}