import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Arrow, Fade, Dim, fmt } from '../lib/svg';

/**
 * Paged KV: physical blocks, the block table that maps sequences onto them, and
 * prefix sharing by block hash. One pool, two allocation policies, one cache.
 */

const STEPS = [
  'One pool of blocks',
  'Contiguous: the fragmentation tax',
  'Blocks: at most 15 wasted',
  'The block table',
  'Attention gathers through it',
  'Prefix hashing',
  'A shared prefix',
  'Eviction under pressure',
];
const CAPTIONS = [
  'The KV cache is one big tensor: [num_blocks, block_size, kv_heads, head_dim]. The first axis is the whole point — a sequence’s tokens do not have to be adjacent in memory.',
  'If each sequence had to own one contiguous region, every request would round its length up and waste the remainder. Eight mixed-length requests waste most of the pool.',
  'Fixed-size blocks of 16 tokens: a sequence of any length is a whole number of blocks plus at most 15 unused slots, in the last block only.',
  'The block table is the indirection. Row i lists the physical blocks holding sequence i. Nothing else in the engine needs to know where memory is.',
  'Because tokens are scattered, attention gathers through the block table: load each block, concatenate logically, and run attention as if it were contiguous.',
  'Hash each block together with its parent’s hash and a block’s identity encodes its whole prefix. Sharing becomes a dictionary lookup instead of a search.',
  'Two requests with the same 1,000-token system prompt produce the same 62 block hashes, so the second one computes 1,000 tokens and allocates 63 blocks instead of 125.',
  'When the pool is full, evict. Under prefix caching, evicting a shared block hurts everyone still using it, so a reference count decides.',
];

const BLOCK = 16;
const POOL_BLOCKS = 40;          // drawn
const COLS = 20;

/** Eight requests with mixed lengths, as Module 08's fragmentation example. */
const REQS = [
  { id: 'a', len: 37 },
  { id: 'b', len: 41 },
  { id: 'c', len: 58 },
  { id: 'd', len: 33 },
  { id: 'e', len: 72 },
  { id: 'f', len: 26 },
  { id: 'g', len: 49 },
  { id: 'h', len: 44 },
];

const PRESET: Record<number, { mode: 'paged' | 'contig'; share: number }> = {
  0: { mode: 'paged', share: 0 },
  1: { mode: 'contig', share: 0 },
  2: { mode: 'paged', share: 0 },
  3: { mode: 'paged', share: 0 },
  4: { mode: 'paged', share: 0 },
  5: { mode: 'paged', share: 250 },
  6: { mode: 'paged', share: 250 },
  7: { mode: 'paged', share: 100 },
};

const PX = 96;
const PY = 150;
const PW = 520;
const CW = PW / COLS;
const CH = 22;
const ROWS = Math.ceil(POOL_BLOCKS / COLS);

type Alloc = { id: string; from: number; count: number; sharedWith: boolean };

/** Paged: a whole number of blocks per request. With a shared prefix, every request's
 *  row begins at block 0 and only its *private* blocks are allocated from the pool. */
function pagedAlloc(sharedBlocks: number): Alloc[] {
  let cur = sharedBlocks;
  return REQS.map((r) => {
    const total = Math.ceil(r.len / BLOCK);
    const n = Math.max(1, total - sharedBlocks);
    const a: Alloc = { id: r.id, from: sharedBlocks > 0 ? 0 : cur, count: n, sharedWith: sharedBlocks > 0 };
    cur += n;
    return a;
  });
}

/** Contiguous: one region per request, rounded up to a whole block, laid end to end.
 *  This is the "no paging" baseline the block pool exists to beat. */
function contigAlloc(sharedBlocks: number): Alloc[] {
  let cur = sharedBlocks;
  return REQS.map((r) => {
    const total = Math.ceil(r.len / BLOCK);
    const a: Alloc = { id: r.id, from: sharedBlocks > 0 ? 0 : cur, count: Math.max(1, total - sharedBlocks), sharedWith: sharedBlocks > 0 };
    cur += Math.max(1, total - sharedBlocks);
    return a;
  });
}

export default function PagedKV() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const p0 = PRESET[s.step];
  const [mode, setMode] = useState<'paged' | 'contig'>(p0.mode);
  const [sharedPrefix, setSharedPrefix] = useState(p0.share);
  useEffect(() => { setMode(p0.mode); setSharedPrefix(p0.share); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const k = s.step;
  const contiguous = mode === 'contig';

  // prefix sharing: `share` tokens common to every request, rounded to whole blocks
  const sharedBlocks = Math.floor(sharedPrefix / BLOCK);
  const allocs = useMemo(
    () => (contiguous ? contigAlloc(sharedBlocks) : pagedAlloc(sharedBlocks)),
    [contiguous, sharedBlocks],
  );

  // physical blocks actually consumed: shared blocks counted once, private blocks once each
  const privateBlocks = allocs.reduce((n, a) => n + a.count, 0);
  const ownedBlocks = sharedBlocks ? sharedBlocks + privateBlocks : privateBlocks;

  const tokensUsed = REQS.reduce((n, r) => n + r.len, 0);
  const waste = 1 - tokensUsed / (ownedBlocks * BLOCK);

  // who owns physical block b: shared blocks (0..sharedBlocks-1) belong to everyone
  const blockOwner = (b: number): { id: string; shared: boolean } | null => {
    if (sharedBlocks > 0 && b < sharedBlocks) return { id: 'shared', shared: true };
    for (const a of allocs) {
      const lo = sharedBlocks > 0 ? sharedBlocks : a.from;
      if (b >= lo && b < lo + a.count) return { id: a.id, shared: false };
    }
    return null;
  };

  return (
    <Stage
      rootRef={ref}
      title="Paged KV: one pool, two allocation policies"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', ['idle', 'free block'], ['error', 'evicted']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Blocks in use', value: `${ownedBlocks}`, kind: 'memory', note: `of ${POOL_BLOCKS} in the pool` },
          { label: 'Tokens stored', value: fmt.int(tokensUsed), kind: 'token', note: `wastes ${(waste * 100).toFixed(0)}% of the blocks` },
          { label: 'Requests fit', value: `${allocs.length}`, kind: 'text', note: contiguous ? 'never more than this' : 'limited by blocks, not by gaps' },
          { label: 'Shared prefix', value: sharedBlocks ? `${sharedPrefix} tokens = ${sharedBlocks} blocks` : 'none', kind: 'compute', note: sharedBlocks ? `saves ${(sharedBlocks * (allocs.length - 1))} blocks` : 'nothing to reuse' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Allocation" value={mode} options={[{ value: 'paged', label: 'Paged (16-token blocks)' }, { value: 'contig', label: 'Contiguous regions' }]} onChange={setMode} />
          <Slider label="Shared prefix on every request" value={sharedPrefix} min={0} max={512} step={16} onChange={setSharedPrefix} format={(v) => v ? `${v} tokens` : 'none'} kind="compute" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A pool of fixed-size KV cache blocks drawn as a grid, with a block table beneath mapping eight sequences onto the blocks they own. Comparing contiguous regions with paged allocation shows the wasted tail, and highlighting shared blocks shows prefix caching.">
        {/* the pool */}
        <Label x={PX} y={PY - 22} tone={2} size={10.5}>{`KV pool: ${POOL_BLOCKS} blocks × ${BLOCK} tokens = ${POOL_BLOCKS * BLOCK} tokens`}</Label>
        {Array.from({ length: POOL_BLOCKS }, (_, b) => {
          const o = blockOwner(b);
          const cx = PX + (b % COLS) * CW;
          const cy = PY + Math.floor(b / COLS) * CH;
          const fill = !o ? 'url(#tps-hatch)' : o.shared ? 'var(--compute)' : 'var(--memory)';
          const stroke = !o ? 'var(--idle)' : o.shared ? 'var(--compute)' : 'var(--memory)';
          return (
            <g key={b}>
              <rect x={cx} y={cy} width={CW - 2} height={CH - 2} rx={2} fill={fill} stroke={stroke} strokeWidth={0.9} className="v-fade" />
              {o && (
                <text x={cx + (CW - 2) / 2} y={cy + 12} textAnchor="middle" className="v-label"
                  style={{ fontSize: 9, fontWeight: 600, fill: '#0e1116' }}>
                  {o.shared ? '◆' : o.id}
                </text>
              )}
            </g>
          );
        })}
        <Label x={PX + PW} y={PY - 8} anchor="end" tone={3} size={9}>{contiguous ? 'one contiguous region per request' : 'one block = 16 tokens, wherever it lives'}</Label>

        {/* block table */}
        <Label x={PX} y={PY + ROWS * CH + 26} tone={2} size={10.5}>block table — sequence → physical blocks</Label>
        {allocs.map((a, i) => {
          const col = i % 4;
          const rowI = Math.floor(i / 4);
          const x = PX + 14 + col * 124;
          const y = PY + ROWS * CH + 42 + rowI * 24;
          const blocks = Array.from({ length: Math.min(a.count, 11) }, (_, j) => (sharedBlocks > 0 ? j : a.from) + j);
          return (
            <g key={a.id}>
              <text x={x - 8} y={y + 12} textAnchor="end" className="v-label" style={{ fontSize: 10, fill: 'var(--text-3)' }}>{a.id}</text>
              {blocks.map((b, j) => (
                <rect key={j} x={x + j * 7} y={y + 3} width={5.4} height={11} rx={1}
                  fill={sharedBlocks > 0 && j < sharedBlocks ? 'var(--compute)' : 'var(--memory-dim)'}
                  stroke={sharedBlocks > 0 && j < sharedBlocks ? 'var(--compute)' : 'var(--memory)'} strokeWidth={0.7} />
              ))}
              {a.count > 11 && <Label x={x + 11 * 7 + 4} y={y + 12} tone={3} size={8}>{`+${a.count - 11}`}</Label>}
            </g>
          );
        })}

        <Fade show={k >= 3}>
          <Label x={PX} y={PY + ROWS * CH + 98} tone={2} size={10} mono={false}>{`◆ = one shared prefix block, referenced by all ${allocs.length} requests`}</Label>
        </Fade>

        {/* gathering one sequence */}
        <Fade show={k >= 4}>
          <Label x={PX} y={PY + ROWS * CH + 120} tone={2} size={10.5}>gathering sequence c through its block table</Label>
          {sharedBlocks > 0 && <Span x={PX} y={PY + ROWS * CH + 128} w={CW * 1.4} h={16} kind="compute" label="◆◆" />}
          <Span x={PX + (sharedBlocks > 0 ? CW * 1.6 : 0)} y={PY + ROWS * CH + 128} w={CW * 2.4} h={16} kind="memory" dim label="private blocks" labelTone="light" />
          <Arrow x1={PX + CW * 4.2} y1={PY + ROWS * CH + 136} x2={PX + CW * 6.4} y2={PY + ROWS * CH + 136} kind="memory" />
          <Label x={PX + CW * 6.6} y={PY + ROWS * CH + 140} size={9.5}>attention sees one contiguous [58 × heads × dim] row</Label>
        </Fade>

        {/* the arithmetic */}
        <Fade show={k >= 2}>
          <Dim x1={PX} x2={PX + PW} y={PY + ROWS * CH + 160} label={`${ownedBlocks} blocks × ${BLOCK} = ${ownedBlocks * BLOCK} token slots`} kind="memory" />
          <Label x={PX} y={PY + ROWS * CH + 180} size={10.5} kind={waste < 0.1 ? 'token' : 'error'}>
            {`${fmt.int(tokensUsed)} real tokens in ${ownedBlocks * BLOCK} slots — ${(waste * 100).toFixed(1)}% wasted`}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}