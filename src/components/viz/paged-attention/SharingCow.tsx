import { useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';

const KV_PER_TOKEN = 131072;

// Block size in tokens; blocks 0..7 exist. Three requests share a prefix.
const BLOCK_SIZE = 16;

// Prompt A: 48 tokens = blocks 0,1,2. Request B shares blocks 0,1 then diverges into 5,6.
// Request C shares block 0 only, then diverges into 7.
const OWNERS: Record<number, string[]> = {
  0: ['A', 'B', 'C'],
  1: ['A', 'B'],
  2: ['A'],
  3: [], // free
  4: [], // free
  5: ['B'],
  6: ['B'],
  7: ['C'],
};

const TABLES: Record<string, number[]> = {
  A: [0, 1, 2],
  B: [0, 1, 5, 6],
  C: [0, 7],
};

const PROMPTS: Record<string, string> = {
  A: 'You are a helpful assistant for Acme Corp.',
  B: 'You are a helpful assistant for Acme Corp. Also answer in French.',
  C: 'You are a helpful assistant for',
};

const STEPS = [
  'Request A alone',
  'Request B arrives with the same first 32 tokens',
  'Share the matching blocks',
  'B writes into block 5, not block 2',
  'A third request, one block in common',
  'Reference counts',
  'End of prompt: copy the shared tail',
  'The saving, measured',
];

const CAPTIONS = [
  'Request A prefills 48 tokens. It owns blocks 0, 1 and 2 outright, and each has a reference count of 1.',
  'Request B has the same first 32 tokens. Storing them again would mean two identical copies of 4 MB.',
  'The engine walks B’s prompt token by token against what is already cached, and matches 32 tokens: blocks 0 and 1 are pointed at by both requests.',
  'B’s prompt then diverges. It gets fresh blocks 5 and 6, and prefill only runs on the unmatched remainder.',
  'Request C shares just the first block. Sharing is per block, so a partial match is still a partial match — nothing has to line up.',
  'Each block now knows how many requests point at it. Block 0 has 3, block 1 has 2, block 5 has 1.',
  'When a shared block is not the last one, whoever needs to write past it copies it first. This is the copy-on-write rule.',
  'A, B and C together hold 9 block references but only 6 distinct blocks: three blocks saved, 6 MB, for zero copying of the shared part.',
];

const W = 640;
const PX0 = 30;
const PW = 250;
const CELL = 30;
const ROW_GAP = 8;
const ROW_Y = { A: 108, B: 152, C: 196 };

export default function SharingCow() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [who, setWho] = useState<string>('B');

  const tables = TABLES[who];
  const refs = tables.map((id) => OWNERS[id].length);
  const distinct = new Set(tables).size;
  const saved = (tables.length - distinct) * BLOCK_SIZE * KV_PER_TOKEN;
  const totalBytes = tables.length * BLOCK_SIZE * KV_PER_TOKEN;

  const focusIds = new Set(tables);
  const showRefs = s.step >= 5;
  const showCow = s.step >= 6;

  return (
    <Stage
      rootRef={ref}
      title="Sharing KV blocks: prefixes and copy-on-write"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'owned by request A'],
        ['token', 'owned by request B'],
        ['compute', 'owned by request C'],
        ['comm', 'shared by two or more'],
        ['error', 'copy before writing'],
        ['idle', 'free'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: `Request ${who}`, value: `${distinct} blocks`, kind: 'memory', note: `${fmt.int(distinct * BLOCK_SIZE)} tokens of context` },
          { label: 'Shared with others', value: `${tables.length - distinct} blocks`, kind: 'comm', note: `${(saved / 1e6).toFixed(1)} MB not stored twice` },
          { label: 'Bytes for this request', value: `${(totalBytes / 1e6).toFixed(1)} MB`, kind: 'text', note: `${fmt.int(tables.length)} block slots, ${distinct} distinct` },
        ]} />
      }
      controls={
        <Segmented
          label="Request"
          value={who}
          options={[{ value: 'A', label: 'A' }, { value: 'B', label: 'B (longest shared prefix)' }, { value: 'C', label: 'C' }]}
          onChange={setWho}
        />
      }
    >
      <svg viewBox={`0 0 ${W} 336`} role="img" aria-label="Three requests whose prompts share a common prefix, pointing at shared and private KV blocks, with a reference count on each block and a copy-on-write step where a request must take a private copy before writing.">
        {/* prompts */}
        <Label x={PX0} y={26} tone={2} size={10.5} mono={false} bold>Prompts arriving</Label>
        {(['A', 'B', 'C'] as const).map((k, i) => (
          <g key={k}>
            <rect x={PX0} y={36 + i * 22} width={220} height={18} rx={2} fill="none" stroke={k === 'A' ? 'var(--memory)' : k === 'B' ? 'var(--token)' : 'var(--compute)'} strokeWidth={1} />
            <text x={PX0 + 6} y={49 + i * 22} className="v-label sans" style={{ fontSize: 9, fill: 'var(--text-2)' }}>
              {PROMPTS[k].slice(0, 42)}{PROMPTS[k].length > 42 ? '…' : ''}
            </text>
          </g>
        ))}
        <Label x={PX0} y={106} tone={3} size={9}>the first 16 tokens are identical</Label>

        {/* block tables */}
        {(['A', 'B', 'C'] as const).map((k) => {
          const y = ROW_Y[k];
          const ids = TABLES[k];
          const colour = k === 'A' ? 'var(--memory)' : k === 'B' ? 'var(--token)' : 'var(--compute)';
          return (
            <g key={k}>
              <Label x={PX0} y={y + 14} tone={1} size={11} bold kind={k === 'A' ? 'memory' : k === 'B' ? 'token' : 'compute'}>{`req ${k}`}</Label>
              <Arrow x1={PX0 + 34} y1={y + 11} x2={PX0 + 48} y2={y + 11} kind={k === 'A' ? 'memory' : k === 'B' ? 'token' : 'compute'} muted={s.step < 1} />
              {ids.map((id, i) => {
                const x = PX0 + 50 + i * (CELL + 5);
                const n = OWNERS[id].length;
                const shared = n > 1;
                const on = focusIds.has(id) && who === k;
                return (
                  <g key={`${k}${id}`}>
                    <rect x={x} y={y} width={CELL} height={24} rx={2} fill={shared && s.step >= 2 ? 'var(--comm)' : colour} fillOpacity={on ? 0.9 : 0.28} stroke={shared && s.step >= 2 ? 'var(--comm)' : colour} strokeWidth={on ? 1.6 : 1} />
                    <text x={x + CELL / 2} y={y + 15} textAnchor="middle" className="v-label" style={{ fontSize: 10, fill: on ? '#0e1116' : 'var(--text-2)' }}>{id}</text>
                    {showRefs && (
                      <text x={x + CELL / 2} y={y + 36} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: shared ? 'var(--comm)' : 'var(--text-3)' }}>
                        ×{n}
                      </text>
                    )}
                  </g>
                );
              })}
              <Label x={PX0 + 52 + ids.length * (CELL + 5)} y={y + 15} tone={3} size={9}>{`${ids.length} blk`}</Label>
            </g>
          );
        })}

        {/* what sharing saved */}
        <Fade show={s.step >= 7}>
          <Label x={PX0} y={248} tone={2} size={10.5} mono={false} bold>All three together</Label>
          <Span x={PX0} y={256} w={9 * (CELL + 5)} h={16} kind="memory" dim label="" />
          <Span x={PX0} y={256} w={6 * (CELL + 5)} h={16} kind="comm" label="6 distinct blocks" labelTone="dark" />
          <Dim x1={PX0} x2={PX0 + 9 * (CELL + 5)} y={284} label="9 block references" kind="memory" />
          <Label x={PX0} y={300} tone={1} size={10.5} bold kind="comm">{"3 blocks saved = 6.3 MB, for zero copying"}</Label>
        </Fade>

        {/* copy-on-write, on the right */}
        <g>
          <Label x={400} y={26} tone={2} size={10.5} mono={false} bold>The one rule that makes it safe</Label>
          <rect x={400} y={36} width={214} height={46} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <text x={410} y={54} className="v-label sans" style={{ fontSize: 10, fill: 'var(--text)' }}>{"if refcount(block) > 1:"}</text>
          <text x={416} y={70} className="v-label" style={{ fontSize: 10, fill: 'var(--error)' }}>{"    copy it, then write"}</text>
          <Label x={400} y={96} tone={3} size={9.5}>Shared blocks are read-only by construction.</Label>

          <Fade show={s.step >= 6}>
            <Label x={400} y={122} tone={2} size={10.5} mono={false} bold>Block 1 is shared by A and B</Label>
            <Span x={400} y={130} w={100} h={26} kind="comm" label="block 1" />
            <Label x={504} y={147} tone={1} size={10} bold kind="comm">refcount 2</Label>
            <Arrow x1={504} y1={158} x2={504} y2={180} kind="error" />
            <Label x={504} y={194} tone={3} size={9.5}>B needs to write token 33</Label>
            <Arrow x1={504} y1={200} x2={504} y2={216} kind="error" />
            <Span x={400} y={222} w={100} h={26} kind="error" dim label="block 1′ (copy)" labelTone="light" />
            <Arrow x1={504} y1={252} x2={504} y2={266} kind="error" />
            <Label x={504} y={280} tone={1} size={10} bold kind="text">B writes to 1′</Label>
            <Label x={400} y={280} tone={2} size={10}>A still reads block 1</Label>
            <Label x={400} y={294} tone={3} size={9.5}>64 KiB copied, once, only on the divergence block.</Label>
          </Fade>
        </g>
      </svg>
    </Stage>
  );
}