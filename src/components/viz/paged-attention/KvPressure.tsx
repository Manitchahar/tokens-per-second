import { useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';

const KV_PER_TOKEN = 131072;
const POOL_GB = 0.25; // the slice of GPU memory the pool is allowed: 256 MB

// Four requests with prompt + generated tokens, and a pool that fills up as they decode.
interface R { id: string; prompt: number; gen: number }

const REQS: R[] = [
  { id: 'A', prompt: 40, gen: 60 },   // short
  { id: 'B', prompt: 900, gen: 400 }, // long, holds a shared prefix too (handled below)
  { id: 'C', prompt: 40, gen: 260 },
  { id: 'D', prompt: 40, gen: 520 },
];

const CAPACITY_TOKENS = Math.floor((POOL_GB * 1e9) / KV_PER_TOKEN); // ~1,907 tokens

const STEPS = [
  'The pool fills up',
  'A new request needs a block',
  'Policy 1: admission control',
  'Policy 2: preempt and recompute',
  'Policy 3: preempt and swap',
  'Recompute, step by step',
  'Swap, step by step',
  'Choosing',
];

const CAPTIONS = [
  'Four requests are decoding. Their KV caches grow one block at a time until the pool is full.',
  'Request E arrives and needs one more block. There is none. This is the moment every KV memory policy exists for.',
  'The cheapest policy: do not admit E yet. It waits. Nothing already running is disturbed, and nothing is wasted.',
  'Recompute: evict a victim, drop its KV blocks, and rebuild them later by running prefill over its prompt again. FLOPs are paid twice.',
  'Swap: evict a victim, copy its KV blocks to host memory, and bring them back later. PCIe bytes are paid instead of FLOPs.',
  'Recompute costs a full prefill of the victim’s context, paid fresh at the moment of restore.',
  'Swap costs a write and a read of the same KV bytes over PCIe — and for this victim that is far cheaper than the prefill.',
  'Both costs scale with victim length, so length cancels. What decides the winner is prefill throughput versus link bandwidth — and the visible cost is a p99 tail.',
];

const W = 640;

export default function KvPressure() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [policy, setPolicy] = useState<'admit' | 'recompute' | 'swap'>('recompute');

  const barX = 96;
  const barW = 470;
  const capW = barW; // pool is exactly full at 100%

  const totalTokens = REQS.reduce((a, r) => a + r.prompt + r.gen, 0);
  const usedFrac = Math.min(1, totalTokens / CAPACITY_TOKENS);
  const fits = totalTokens <= CAPACITY_TOKENS;

  const rowY = (i: number) => 46 + i * 26;
  const kinds = ['memory', 'token', 'compute', 'comm'] as const;

  return (
    <Stage
      rootRef={ref}
      title="The pool is full: what an engine does about it"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'KV cache in the pool'],
        ['token', 'KV cache in host memory'],
        ['compute', 'recomputed on restore'],
        ['comm', 'moved over PCIe'],
        ['idle', 'queued, not admitted'],
        ['error', 'pool full'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Pool capacity', value: `${(POOL_GB * 1000).toFixed(0)} MB`, kind: 'memory', note: `${fmt.int(CAPACITY_TOKENS)} tokens of KV` },
          { label: 'Demanded now', value: `${fmt.int(totalTokens)} tokens`, kind: fits ? 'memory' : 'error', note: fits ? 'fits' : `${((totalTokens / CAPACITY_TOKENS) * 100).toFixed(0)}% of the pool` },
          { label: 'Policy cost', value: policy === 'admit' ? '0' : policy === 'swap' ? 'PCIe bytes' : 're-prefill FLOPs', kind: policy === 'recompute' ? 'compute' : policy === 'swap' ? 'comm' : 'text', note: policy === 'admit' ? 'E waits' : 'paid on restore' },
        ]} />
      }
      controls={
        <Segmented
          label="When the pool is full, prefer"
          value={policy}
          options={[{ value: 'admit', label: 'queue (admission control)' }, { value: 'recompute', label: 'recompute' }, { value: 'swap', label: 'swap to host' }]}
          onChange={setPolicy}
        />
      }
    >
      <svg viewBox={`0 0 ${W} 372`} role="img" aria-label="A full KV pool with four decoding requests and one blocked request, then the three policies an engine can apply: queue it, recompute it later, or swap it to host memory and back.">
        {/* the pool */}
        <Label x={barX} y={26} tone={2} size={10.5} mono={false} bold>{`KV pool = ${(POOL_GB * 1000).toFixed(0)} MB = ${fmt.int(CAPACITY_TOKENS)} tokens`}</Label>
        <rect x={barX} y={34} width={capW} height={12} rx={2} fill="none" stroke="var(--rule-2)" />
        <Span x={barX} y={34} w={capW * usedFrac} h={12} kind={fits ? 'memory' : 'error'} label="" />
        {!fits && <Label x={barX + capW / 2} y={43} anchor="middle" tone={1} size={10} bold kind="error">{"FULL — demand exceeds capacity"}</Label>}

        {/* per-request rows */}
        {REQS.map((r, i) => {
          const y = rowY(i);
          const ctx = r.prompt + r.gen;
          const w = (ctx / CAPACITY_TOKENS) * capW;
          const colour = kinds[i];
          return (
            <g key={r.id}>
              <Label x={barX - 12} y={y + 14} anchor="end" tone={1} size={11} bold kind={colour}>{r.id}</Label>
              <Span x={barX} y={y + 2} w={w} h={18} kind={colour} label={`${ctx} tok`} />
              <Label x={barX + w + 6} y={y + 15} tone={3} size={9}>{`${(ctx * KV_PER_TOKEN / 1e9).toFixed(2)} GB`}</Label>
            </g>
          );
        })}

        {/* blocked request E */}
        <g>
          <Label x={barX - 12} y={rowY(4) + 14} anchor="end" tone={1} size={11} bold>E</Label>
          <rect x={barX} y={rowY(4) + 2} width={(60 / CAPACITY_TOKENS) * capW} height={18} rx={2} fill="none" stroke="var(--error)" strokeDasharray="3 3" />
          <Label x={barX + (60 / CAPACITY_TOKENS) * capW + 6} y={rowY(4) + 15} tone={1} size={9.5} kind="error">{"needs 60 tokens; the pool is full"}</Label>
        </g>

        <Dim x1={barX} x2={barX + capW} y={rowY(4) + 34} label={`${(POOL_GB * 1000).toFixed(0)} MB`} kind={fits ? 'memory' : 'error'} />

        {/* three policies */}
        <Label x={96} y={212} tone={2} size={10.5} mono={false} bold>Three policies, in increasing order of invasiveness</Label>

        {/* admission */}
        <g>
          <rect x={96} y={222} width={172} height={62} rx={3} fill="var(--ground-3)" stroke={policy === 'admit' ? 'var(--idle)' : 'var(--rule-2)'} strokeWidth={policy === 'admit' ? 1.8 : 1} />
          <Label x={106} y={238} tone={1} size={10} bold>1 · Queue (admission control)</Label>
          <text x={106} y={252} className="v-label sans" style={{ fontSize: 9, fill: 'var(--text-2)' }}>{"E stays outside the pool."}</text>
          <text x={106} y={264} className="v-label sans" style={{ fontSize: 9, fill: 'var(--text-2)' }}>{"Nobody is disturbed. TTFT for E grows."}</text>
          <text x={106} y={276} className="v-label" style={{ fontSize: 9, fill: 'var(--idle)' }}>{"cost: E's wait"}</text>
        </g>

        {/* recompute */}
        <g>
          <rect x={280} y={222} width={172} height={62} rx={3} fill="var(--ground-3)" stroke={policy === 'recompute' ? 'var(--compute)' : 'var(--rule-2)'} strokeWidth={policy === 'recompute' ? 1.8 : 1} />
          <Label x={290} y={238} tone={1} size={10} bold>2 · Recompute</Label>
          <text x={290} y={252} className="v-label sans" style={{ fontSize: 9, fill: 'var(--text-2)' }}>{"Evict a victim. Throw its KV away."}</text>
          <text x={290} y={264} className="v-label sans" style={{ fontSize: 9, fill: 'var(--text-2)' }}>{"Rebuild later with a full prefill."}</text>
          <text x={290} y={276} className="v-label" style={{ fontSize: 9, fill: 'var(--compute)' }}>{"cost: FLOPs, no PCIe"}</text>
        </g>

        {/* swap */}
        <g>
          <rect x={464} y={222} width={150} height={62} rx={3} fill="var(--ground-3)" stroke={policy === 'swap' ? 'var(--comm)' : 'var(--rule-2)'} strokeWidth={policy === 'swap' ? 1.8 : 1} />
          <Label x={474} y={238} tone={1} size={10} bold>3 · Swap</Label>
          <text x={474} y={252} className="v-label sans" style={{ fontSize: 9, fill: 'var(--text-2)' }}>{"Evict, copy KV to host."}</text>
          <text x={474} y={264} className="v-label sans" style={{ fontSize: 9, fill: 'var(--text-2)' }}>{"Restore on eviction back."}</text>
          <text x={474} y={276} className="v-label" style={{ fontSize: 9, fill: 'var(--comm)' }}>{"cost: PCIe"}</text>
        </g>

        {/* what the chosen policy does to the victim */}
        <Fade show={s.step >= 2}>
          <Arrow x1={96} y1={296} x2={96} y2={306} kind={policy === 'admit' ? 'idle' : policy === 'swap' ? 'comm' : 'compute'} />
          {policy === 'admit' && (
            <g>
              <Label x={96} y={320} tone={1} size={10.5} bold kind="idle">{"E waits its turn. Throughput unchanged; its TTFT is the cost."}</Label>
            </g>
          )}
          {policy === 'recompute' && (
            <g>
              <Label x={96} y={320} tone={2} size={10.5} bold>{"Restore a victim with "}</Label>
              <Label x={210} y={320} tone={1} size={10.5} bold kind="compute">{"prefill(prompt + so far)"}</Label>
              <Label x={96} y={336} tone={3} size={9.5}>{"e.g. 900-token victim: 900 tokens of prefill ≈ 36 GFLOP × 900 ≈ a full prefill's worth of compute, paid again."}</Label>
            </g>
          )}
          {policy === 'swap' && (
            <g>
              <Label x={96} y={320} tone={2} size={10.5} bold>{"Move victim KV: write + read "}</Label>
              <Label x={300} y={320} tone={1} size={10.5} bold kind="comm">{"2 × ctx × 128 KiB"}</Label>
              <Label x={96} y={336} tone={3} size={9.5}>{"e.g. 900-token victim: 236 MB over PCIe ≈ 20 ms at 24 GB/s each way. Cheaper than recompute for long contexts."}</Label>
            </g>
          )}
        </Fade>
      </svg>
    </Stage>
  );
}