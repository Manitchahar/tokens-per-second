import { useMemo, useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Dim, Fade, Move, fmt } from '../lib/svg';

const POOL_BLOCKS = 32; // blocks pre-allocated at startup
const BLOCK_SIZE = 16; // tokens per block

// How many blocks each request owns at each step. Growing = +1, never a move.
const OWNERS: { step: number; blocks: Record<string, number> }[] = [
  { step: 0, blocks: {} },
  { step: 1, blocks: {} },
  { step: 2, blocks: { A: 7, B: 3, C: 2 } },
  { step: 3, blocks: { A: 8, B: 3, C: 2 } },
  { step: 4, blocks: { A: 8, B: 3, C: 2 } },
  { step: 5, blocks: { A: 8, B: 3, C: 2 } },
  { step: 6, blocks: { A: 8, B: 3, C: 2 } },
  { step: 7, blocks: { A: 8, B: 3, C: 2 } },
];

const STEPS = [
  'One pool, allocated once',
  'A block is 16 tokens',
  'Requests own lists of blocks',
  'Growing pops one block',
  'The block table is the index',
  'Attention follows the table',
  'One gather, once per step',
  'Waste is now one block',
];

const CAPTIONS = [
  'At startup the engine allocates the whole KV cache once, as 32 equal blocks. Nothing is allocated per request, ever again.',
  'Each block holds the keys and values of 16 tokens, for every layer and every KV head. Same size, always.',
  'A request does not own a region. It owns a list of block ids. Its table says: logical block 0 lives at physical block 19, and so on.',
  'When request A needs one more token and crosses a block boundary, the engine pops a free block id and appends it. No bytes move.',
  'The block table is the only indirection. One small integer vector per request turns it into an ordinary gather.',
  'Attention never needs a contiguous buffer: the kernel walks the table, block by block, and every block has the same fixed shape.',
  'Because the shapes are uniform, the inner loop is a fast fixed-shape attention. The only extra work is one gather of the KV into SRAM.',
  'With blocks of 16 tokens, the only waste left is the empty tail of the last block of each sequence — at most 15 tokens each.',
];

const W = 640;
// pool grid
const GX = 318;
const GY = 44;
const CW = 34;
const CH = 22;
const CGAP = 5;
const COLS = 8;
const cellX = (i: number) => GX + (i % COLS) * (CW + CGAP);
const cellY = (i: number) => GY + Math.floor(i / COLS) * (CH + CGAP);

// block table rows on the left
const TX = 42;
const ROW_Y: Record<string, number> = { A: 52, B: 86, C: 120 };

const COLORS: Record<string, string> = { A: 'var(--memory)', B: 'var(--token)', C: 'var(--compute)' };

export default function BlocksAndTables() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [bs, setBs] = useState(BLOCK_SIZE);
  const [focus, setFocus] = useState<string>('A');

  const owners = OWNERS[Math.min(s.step, OWNERS.length - 1)].blocks;

  // Assign blocks in arrival order so the tables read like real ones.
  const tables = useMemo(() => {
    const taken: number[] = [];
    const out: Record<string, number[]> = {};
    for (const name of ['A', 'B', 'C']) {
      const n = owners[name] ?? 0;
      const ids: number[] = [];
      for (let i = 0; i < n; i++) {
        // spread A across the pool, as a real allocator would
        const id = (i * 5 + (name === 'A' ? 0 : name === 'B' ? 3 : 6)) % POOL_BLOCKS;
        if (!taken.includes(id)) { ids.push(id); taken.push(id); }
        else {
          const spare = Array.from({ length: POOL_BLOCKS }, (_, k) => k).find((k) => !taken.includes(k));
          if (spare !== undefined) { ids.push(spare); taken.push(spare); }
        }
      }
      out[name] = ids;
    }
    return out;
  }, [owners]);

  const ownerOf = useMemo(() => {
    const m = new Map<number, string>();
    for (const name of ['A', 'B', 'C']) for (const id of tables[name]) m.set(id, name);
    return m;
  }, [tables]);

  const focusBlocks = tables[focus] ?? [];
  const focusTokens = focusBlocks.length * bs;
  const usedBlocks = focusBlocks.length;
  const freeBlocks = POOL_BLOCKS - usedBlocks;

  // the pool, scaled by block size so the slider is visible in the drawing
  const showSlots = s.step >= 1;
  const anim = useTween(freeBlocks, 500);

  const gx = 42;
  const gw = 540;
  const slotW = gw / (POOL_BLOCKS * bs); // one token's worth of width in the gather strip

  return (
    <Stage
      rootRef={ref}
      title="A KV pool of equal blocks, indexed by a block table"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'request A blocks'],
        ['token', 'request B blocks'],
        ['compute', 'request C blocks'],
        ['idle', 'free'],
        ['error', 'wasted tail'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Pool', value: `${POOL_BLOCKS} blocks`, kind: 'memory', note: `allocated once at startup` },
          { label: 'Block size', value: `${bs} tokens`, kind: 'text', note: `${(bs * 131072 / 1e6).toFixed(1)} MB per block` },
          { label: `Request ${focus}`, value: `${usedBlocks} blocks`, kind: focus === 'A' ? 'memory' : focus === 'B' ? 'token' : 'compute', note: `${fmt.int(focusTokens)} tokens of context` },
          { label: 'Free blocks', value: `${Math.round(anim)}`, kind: 'idle', note: 'popped one at a time' },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Block size"
            value={bs}
            options={[{ value: 8, label: '8 tokens' }, { value: 16, label: '16 tokens' }, { value: 32, label: '32 tokens' }]}
            onChange={(v) => setBs(v)}
          />
          <Segmented
            label="Look at request"
            value={focus}
            options={[{ value: 'A', label: 'A' }, { value: 'B', label: 'B' }, { value: 'C', label: 'C' }]}
            onChange={setFocus}
          />
        </>
      }
    >
      <svg viewBox={`0 0 ${W} 372`} role="img" aria-label="A pre-allocated pool of 32 equal KV blocks with three requests pointing at scattered block ids through their block tables, and a gather strip that copies those blocks into a contiguous buffer for attention.">
        {/* ── pool ───────────────────────────────────────────── */}
        <Label x={GX} y={28} tone={2} size={10.5} mono={false} bold>KV pool · {POOL_BLOCKS} blocks of {bs} tokens = {fmt.int(POOL_BLOCKS * bs)} tokens total</Label>
        {Array.from({ length: POOL_BLOCKS }, (_, i) => {
          const owner = ownerOf.get(i);
          const shown = s.step === 0 ? false : !!owner;
          const inFocus = owner === focus && s.step >= 4;
          return (
            <g key={i}>
              <rect
                x={cellX(i)}
                y={cellY(i)}
                width={CW}
                height={CH}
                rx={2}
                fill={shown && owner ? COLORS[owner] : 'none'}
                fillOpacity={shown && owner ? (inFocus ? 0.95 : 0.32) : 1}
                stroke={shown && owner ? COLORS[owner] : 'var(--rule-2)'}
                strokeWidth={inFocus ? 1.6 : 1}
                strokeDasharray={shown && owner ? undefined : '3 3'}
                className="v-cell"
              />
              {showSlots && shown && owner && (
                <text x={cellX(i) + CW / 2} y={cellY(i) + CH / 2 + 3.2} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: inFocus ? '#0e1116' : 'var(--text-2)' }}>
                  {i}
                </text>
              )}
            </g>
          );
        })}
        <Label x={GX} y={GY + 4 * (CH + CGAP) + 12} tone={3} size={9.5}>physical block id</Label>

        {/* ── block tables ───────────────────────────────────── */}
        <Label x={TX} y={28} tone={2} size={10.5} mono={false} bold>Block tables</Label>
        {['A', 'B', 'C'].map((name) => {
          const y = ROW_Y[name];
          const ids = tables[name];
          return (
            <g key={name}>
              <Label x={TX} y={y + 12} tone={1} size={11} bold kind={name === 'A' ? 'memory' : name === 'B' ? 'token' : 'compute'}>{`req ${name}`}</Label>
              <Arrow x1={TX + 34} y1={y + 9} x2={TX + 48} y2={y + 9} kind={name === 'A' ? 'memory' : name === 'B' ? 'token' : 'compute'} muted={s.step < 2} />
              {s.step >= 2 &&
                ids.map((id, i) => {
                  const cw = 21;
                  const x = TX + 50 + i * (cw + 3);
                  return (
                    <g key={id}>
                      <rect x={x} y={y} width={cw} height={18} rx={2} fill={COLORS[name]} fillOpacity={focus === name && s.step >= 4 ? 0.9 : 0.3} stroke={COLORS[name]} strokeWidth={focus === name && s.step >= 4 ? 1.5 : 1} />
                      <text x={x + cw / 2} y={y + 12.6} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fill: focus === name && s.step >= 4 ? '#0e1116' : 'var(--text-2)' }}>{id}</text>
                    </g>
                  );
                })}
              {s.step >= 2 && (
                <Label x={TX + 52 + ids.length * 24} y={y + 12} tone={3} size={9}>{`${ids.length} blk · ${ids.length * bs} tok`}</Label>
              )}
            </g>
          );
        })}
        <Fade show={s.step >= 3}>
          <text x={TX} y={162} className="v-label" style={{ fontSize: 9.5, fill: 'var(--token)' }}>A crossed a boundary:</text>
          <text x={TX} y={175} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text-2)' }}>append one id. 0 bytes copied.</text>
        </Fade>

        {/* ── gather ─────────────────────────────────────────── */}
        <Fade show={s.step >= 5}>
          <Label x={gx} y={212} tone={2} size={10.5} mono={false} bold>{`gather: block table → one contiguous buffer in SRAM, request ${focus}`}</Label>
          {focusBlocks.map((id, i) => {
            const x = gx + i * bs * slotW;
            return (
              <g key={`f${id}`}>
                <Arrow
                  x1={cellX(id) + CW / 2}
                  y1={cellY(id) + CH + 2}
                  x2={x + (bs * slotW) / 2}
                  y2={238}
                  kind={focus === 'A' ? 'memory' : focus === 'B' ? 'token' : 'compute'}
                  dashed
                  opacity={0.5}
                />
                <Span x={x} y={240} w={bs * slotW - 1} h={22} kind={focus === 'A' ? 'memory' : focus === 'B' ? 'token' : 'compute'} labelTone="dark" />
              </g>
            );
          })}
          <Dim x1={gx} x2={gx + usedBlocks * bs * slotW} y={274} label={`${usedBlocks} × ${bs} = ${fmt.int(focusTokens)} tokens, contiguous`} kind={focus === 'A' ? 'memory' : focus === 'B' ? 'token' : 'compute'} />
          <Span x={gx + usedBlocks * bs * slotW} y={240} w={Math.min(18, Math.max(0, gw - usedBlocks * bs * slotW))} h={22} kind="error" dim />
          <Label x={gx + usedBlocks * bs * slotW + 6} y={254} tone={3} size={9}>idle</Label>
          <Move x={0} y={0}>
            <Span x={gx} y={278} w={Math.min(gw, usedBlocks * bs * slotW)} h={20} kind="compute" label="attention reads this" />
          </Move>
        </Fade>

        {/* ── waste ──────────────────────────────────────────── */}
        <Fade show={s.step >= 6}>
          <Label x={gx} y={322} tone={2} size={10.5} mono={false} bold>Worst case, all three requests at a token boundary − 1</Label>
          <Span x={gx} y={330} w={gw} h={16} kind="error" dim label="up to 15 empty tokens per request" labelTone="light" />
          <Label x={gx + gw + 6} y={341} anchor="start" tone={3} size={9}>{`≤ ${((bs - 1) / bs * 100).toFixed(0)}%`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}