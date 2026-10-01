import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, Move, fmt } from '../lib/svg';

const KV_TOK = 128 * 1024; // bytes of KV per token, Llama-3.1-8B BF16

const STEPS = [
  'A flat cache',
  'A radix tree',
  'Sharing is free',
  'It is a trie of token blocks',
  'Evicting a leaf',
  'Evicting a shared ancestor',
  'Design the application',
];
const CAPTIONS = [
  'A flat cache stores completed sequences and matches a new prompt against whole entries. The conversation you are typing shares 2,000 tokens with the one you just sent, and gets nothing back.',
  'A radix tree keys the cache on every shared prefix instead. One tree per model per GPU, holding the blocks of every request, so a prompt finds its own history wherever it was made.',
  'Two turns of the same conversation hang off the same nodes. The 2,000 shared tokens are stored once and reused by both — the copy-on-write from Module 08, now automatic.',
  'Every edge is one KV block, so a hit is exact and lossless. The same token IDs at the same positions produce identical keys and values, so reuse is not an approximation.',
  'When memory runs short the cache has to drop something. The safe candidate is a leaf: a node no other request needs any more. On this tree, that is the small cold branch.',
  'Evicting an internal node would strand every request below it. RadixAttention only drops nodes whose whole subtree is cold, oldest first — so the shared root and any live branch are never at risk.',
  'Because the cache follows the tree, the shape of your application decides the hit rate. Put the stable prefix first, keep the variable part last, and share one system prompt across every request.',
];

interface Node {
  id: number;
  parent: number | null;
  depth: number; // tokens of context this node represents
  tokens: number; // tokens stored in this node and above
  x: number;
  y: number;
  lastUsed: number; // lower = longer ago
  live: boolean; // a running request still points at it
}

const TOTAL = 2000 + 300 + 500 + 120 + 700 + 250 + 80;

/**
 * Three conversations sharing one 2,000-token system prompt, each branching as
 * the user keeps talking, plus a small unrelated conversation from another product.
 */
function buildTree(): Node[] {
  const ROOT = 0;
  const u1 = 1, u2 = 2, u3 = 3, a2 = 4, b2 = 5, other = 6;
  return [
    { id: ROOT, parent: null, depth: 0, tokens: 2000, x: 70, y: 44, lastUsed: 2, live: true },
    { id: u1, parent: ROOT, depth: 2000, tokens: 300, x: 220, y: 44, lastUsed: 5, live: true },
    { id: u2, parent: ROOT, depth: 2000, tokens: 500, x: 220, y: 122, lastUsed: 3, live: true },
    { id: u3, parent: ROOT, depth: 2000, tokens: 120, x: 220, y: 200, lastUsed: 4, live: false },
    { id: a2, parent: u1, depth: 2300, tokens: 700, x: 380, y: 122, lastUsed: 6, live: true },
    { id: b2, parent: u2, depth: 2500, tokens: 250, x: 380, y: 200, lastUsed: 1, live: false },
    { id: other, parent: null, depth: 0, tokens: 80, x: 70, y: 278, lastUsed: 7, live: true },
  ];
}

/**
 * Evict the least-recently-used nodes whose entire subtree is cold.
 *
 * A node with a surviving child is not a leaf, and a node with a live request
 * below it is not cold. Neither is ever a candidate — that is the rule that
 * keeps eviction from destroying work other requests depend on.
 */
function evict(tree: Node[], cap: number) {
  const alive = new Set(tree.map((n) => n.id));
  const childrenOf = (id: number) => tree.filter((n) => n.parent === id && alive.has(n.id));
  const isLive = (id: number): boolean => {
    const self = tree.find((n) => n.id === id)!;
    return self.live || childrenOf(id).some((c) => isLive(c.id));
  };
  const dropped: number[] = [];
  let used = TOTAL;
  let guard = 0;
  while (used > cap && guard++ < 64) {
    const candidates = tree
      .filter((n) => alive.has(n.id))
      .filter((n) => !isLive(n.id))
      .filter((n) => childrenOf(n.id).length === 0)
      .sort((a, b) => a.lastUsed - b.lastUsed);
    if (!candidates.length) break;
    const pick = candidates[0];
    alive.delete(pick.id);
    dropped.push(pick.id);
    used -= pick.tokens;
  }
  return { evicted: new Set(dropped), used };
}

const PRESET: Record<number, number> = { 0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0.5, 6: 0.5 };

export default function RadixTree() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [budget, setBudget] = useState(PRESET[s.step]);
  useEffect(() => setBudget(PRESET[s.step]), [s.step]);

  const tree = useMemo(buildTree, []);
  const { evicted, used } = useMemo(() => evict(tree, TOTAL * budget), [tree, budget]);

  const byId = useMemo(() => new Map(tree.map((n) => [n.id, n])), [tree]);
  const fracUsed = useTween(used / TOTAL, 500);

  // A new request that shares the root's system prompt and user A's first turn.
  const canHitRoot = !evicted.has(0);
  const canHitBranch = !evicted.has(1);
  const reusable = (canHitRoot ? 2000 : 0) + (canHitBranch ? 300 : 0);

  return (
    <Stage
      rootRef={ref}
      title="RadixAttention: the KV cache as a tree of shared prefixes"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', ['token', 'blocks a live request needs'], ['error', 'evicted'], ['idle', 'free space']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Cache budget', value: `${(budget * 100).toFixed(0)}%`, kind: 'memory', note: `${fmt.int(TOTAL * budget)} of ${fmt.int(TOTAL)} tokens` },
          { label: 'In use', value: `${fmt.int(used)} tokens`, kind: evicted.size ? 'error' : 'memory', note: fmt.bytes(used * KV_TOK) },
          { label: 'Evicted', value: `${evicted.size} node${evicted.size === 1 ? '' : 's'}`, kind: evicted.size ? 'error' : 'text', note: evicted.size ? 'cold leaves, oldest first' : 'nothing dropped' },
          { label: 'Next request reuses', value: `${fmt.int(reusable)} tokens`, kind: 'token', note: `${((reusable / 2300) * 100).toFixed(0)}% of its prompt` },
        ]} />
      }
      controls={
        <Slider
          label="Cache budget (share of GPU memory given to the prefix tree)"
          value={budget}
          min={0.2}
          max={1}
          step={0.02}
          onChange={setBudget}
          format={(v) => `${(v * 100).toFixed(0)}%`}
          kind="memory"
        />
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A radix tree of cached KV blocks. A shared 2,000-token system prompt sits at the root, with several conversation branches hanging off it. Under a tight memory budget the coldest leaf nodes are evicted, shown in red hatching, while the shared root and the live branches are kept.">
        <Label x={20} y={22} tone={2} size={11} bold>Cached KV, as a tree of shared token prefixes</Label>
        <Label x={620} y={22} anchor="end" tone={3} size={9.5}>one node = one KV block of tokens</Label>

        {/* edges */}
        {tree.map((n) => {
          if (n.parent === null) return null;
          const p = byId.get(n.parent)!;
          const dead = evicted.has(n.id) || evicted.has(n.parent);
          return (
            <line
              key={`e${n.id}`}
              x1={p.x + 6} y1={p.y + 6} x2={n.x} y2={n.y + 6}
              stroke={dead ? 'var(--error)' : 'var(--rule-2)'}
              strokeWidth={dead ? 1 : 1.6}
              strokeDasharray={dead ? '3 3' : undefined}
              opacity={dead ? 0.4 : 1}
            />
          );
        })}

        {/* nodes */}
        {tree.map((n) => {
          const dead = evicted.has(n.id);
          const root = n.parent === null && n.tokens === 2000;
          const fill = dead ? 'url(#tps-hatch-red)' : root ? 'var(--memory)' : n.live ? 'var(--token)' : 'var(--memory)';
          const r = root ? 10 : 8;
          return (
            <g key={n.id}>
              <circle cx={n.x + 6} cy={n.y + 6} r={r} fill={fill}
                stroke={dead ? 'var(--error)' : root ? 'var(--memory)' : n.live ? 'var(--token)' : 'var(--memory)'}
                strokeWidth={root ? 2 : 1.4} opacity={dead ? 0.55 : 1} />
              <text x={n.x + 6} y={n.y + 10} textAnchor="middle" className="v-label"
                style={{ fontSize: root ? 8.5 : 8, fontWeight: 700, fill: dead ? 'var(--error)' : root || n.live ? '#0e1116' : 'var(--text)' }}>
                {fmt.int(n.tokens)}
              </text>
              {!dead && (
                <text x={n.x + 6} y={n.y - 8} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>
                  {root ? 'system prompt' : n.live ? 'live branch' : 'cold leaf'}
                </text>
              )}
              {dead && (
                <text x={n.x + 6} y={n.y - 8} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: 'var(--error)' }}>
                  evicted
                </text>
              )}
            </g>
          );
        })}

        {/* a new request walking down to its shared prefix */}
        <Fade show={s.step >= 2}>
          <Move x={byId.get(1)!.x + 6} y={byId.get(1)!.y + 6}>
            <circle r={26} fill="none" stroke="var(--token)" strokeWidth={1.2} opacity={0.5} />
          </Move>
          <path d={`M${byId.get(1)!.x + 6},${byId.get(1)!.y + 6} L${byId.get(0)!.x + 6},${byId.get(0)!.y + 6}`}
            stroke="var(--token)" strokeWidth={2.5} fill="none" opacity={0.55} />
          <Label x={byId.get(0)!.x + 22} y={byId.get(0)!.y - 2} size={9.5} kind="token">
            a new request walks down and reuses 2,300 tokens
          </Label>
        </Fade>

        {/* what the eviction rule guarantees */}
        <Fade show={s.step >= 4}>
          <Label x={20} y={324} tone={2} size={10.5} bold>What is allowed to go</Label>
          <Label x={20} y={341} size={9.5} kind="token">live — a running request reads it every step</Label>
          <Label x={20} y={357} size={9.5} kind="memory">internal — a child still hangs off it; dropping it would strand that child</Label>
          <Label x={20} y={373} size={9.5} kind="error">evicted — cold leaf, nobody needs it, oldest first</Label>
        </Fade>

        {/* usage bar */}
        <g>
          <Label x={480} y={356} tone={3} size={9}>KV cache in use</Label>
          <rect x={480} y={362} width={140} height={12} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={480} y={362} width={140 * fracUsed} height={12} rx={2} fill="var(--memory)" opacity={0.8} />
          <rect x={480 + 140 * fracUsed} y={362} width={140 * (1 - fracUsed)} height={12} rx={2} fill="url(#tps-hatch)" opacity={0.4} />
        </g>
      </svg>
    </Stage>
  );
}