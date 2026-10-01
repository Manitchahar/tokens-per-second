import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Span, Label, Arrow, Dim, Fade, fmt } from '../lib/svg';
import { llama8b, prefillSec, seqKvBytes } from './prefill';

const KV_TOK = 128 * 1024; // bytes of KV per token, Llama-3.1-8B BF16

const STEPS = [
  'Two users, same system prompt',
  'The waste in plain numbers',
  'The first request pays',
  'The second one does not',
  'Hit rate is the whole story',
  'Block granularity',
  'Never cache the suffix',
];
const CAPTIONS = [
  'Two users of the same chat product. Their prompts differ, but both begin with the identical 2,000-token system prompt plus tools.',
  'Without reuse, the second request recomputes 2,000 tokens of prefill that were already computed a moment ago. Same weights, same input, same answer — thrown away.',
  'The first request must compute all 2,500 tokens. Nothing to reuse: nobody has asked before.',
  'The second request matches 2,000 of them. Those tokens are already in the KV cache, so it computes only the 500 new ones. Prefill drops from 59 ms to 15 ms.',
  'Define the hit rate as cached prefix tokens divided by total prompt tokens. When most traffic shares a prefix, most prefill work disappears.',
  'Caching works at the granularity of the KV blocks from Module 08. A partial block cannot be reused, which is why block size trades memory waste for hit rate.',
  'The generated suffix is the one part you must never cache: it is the output of this request, it changes every time, and reusing it would be incorrect as well as wasteful.',
];

const PRESET: Record<number, { ctx: number; cached: number; mode: 'exact' | 'turns' | 'rag' }> = {
  0: { ctx: 2000, cached: 0, mode: 'turns' },
  1: { ctx: 2000, cached: 0, mode: 'turns' },
  2: { ctx: 2500, cached: 0, mode: 'turns' },
  3: { ctx: 2500, cached: 2000, mode: 'turns' },
  4: { ctx: 2500, cached: 2000, mode: 'turns' },
  5: { ctx: 2500, cached: 2000, mode: 'turns' },
  6: { ctx: 2500, cached: 2000, mode: 'turns' },
};

const MAXW = 534; // px available for the token strip
const pxPerTok = MAXW / 2600;
const W = MAXW + 60;

export default function PrefixCache() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [ctx, setCtx] = useState(PRESET[s.step].ctx);
  const [cached, setCached] = useState(PRESET[s.step].cached);
  const [mode, setMode] = useState(PRESET[s.step].mode);
  useEffect(() => {
    const p = PRESET[s.step];
    setCtx(p.ctx); setCached(p.cached); setMode(p.mode);
  }, [s.step]);

  const hitRate = ctx > 0 ? cached / ctx : 0;
  const total = prefillSec(ctx, llama8b).t;
  const cold = prefillSec(Math.max(0, ctx - cached), llama8b).t;
  const t = useTween(cached, 600);
  const saved = total - cold;
  const frac = ctx / 2600;

  // block view: how many 16-token blocks, and how many of them are fully reusable
  const BLOCK = 16;
  const shownBlocks = 26;

  const cachedMs = useTween(total * 1000, 500);
  const coldMs = useTween(cold * 1000, 500);

  const PRESET_HIT: Record<string, number> = {
    turns: 0.8, // most of a chat prompt is the conversation so far
    rag: 0.4, // a retrieved context block, plus some boilerplate
    exact: 0.99, // the same request repeated
  };
  const trafficHit = PRESET_HIT[mode];

  return (
    <Stage
      rootRef={ref}
      title="Two identical system prompts: what a prefix cache saves"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', ['memory', 'reused KV cache'], 'token', ['error', 'recomputed'], 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Prompt', value: `${fmt.int(ctx)} tokens`, kind: 'text', note: `${(ctx * KV_TOK / 2**20).toFixed(2)} MiB of KV` },
          { label: 'Hit rate', value: `${(hitRate * 100).toFixed(0)}%`, kind: hitRate > 0.5 ? 'token' : 'text', note: `${fmt.int(cached)} of ${fmt.int(ctx)} tokens reused` },
          { label: 'Prefill, cold', value: `${(cachedMs).toFixed(0)} ms`, kind: 'error', note: 'if nothing were cached' },
          { label: 'Prefill, cached', value: `${coldMs.toFixed(0)} ms`, kind: 'compute', note: saved > 0.001 ? `${(saved * 1000).toFixed(0)} ms saved` : 'nothing to reuse' },
        ]} />
      }
      controls={
        <>
          <Slider label="Prompt length (tokens)" value={ctx} min={16} max={4096} log onChange={(v) => { setCtx(v); setCached(Math.min(cached, v)); }} format={(v) => `${fmt.int(v)}`} kind="memory" />
          <Slider label="Tokens already cached from a previous request" value={cached} min={0} max={4096} log onChange={(v) => setCached(Math.min(v, ctx))} format={(v) => `${fmt.int(v)}  (${ctx ? ((v / ctx) * 100).toFixed(0) : 0}%)`} kind="token" />
          <Segmented
            label="What a real workload shares"
            value={mode}
            options={[{ value: 'turns', label: 'Multi-turn chat' }, { value: 'rag', label: 'RAG' }, { value: 'exact', label: 'Identical request' }]}
            onChange={(v) => {
              setMode(v);
              if (v === 'turns') { setCtx(2500); setCached(2000); }
              if (v === 'rag') { setCtx(3000); setCached(1200); }
              if (v === 'exact') { setCtx(2500); setCached(2496); }
            }}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 384" role="img" aria-label="A prompt split into a cached prefix and a newly computed suffix, with the prefill cost of each part shown as a bar, plus a block-level view of the same prompt showing which blocks are reusable.">
        {/* ── the token strip ── */}
        <Label x={0} y={26} tone={2} size={11} bold>The prompt, token by token</Label>
        <Label x={578} y={26} anchor="end" tone={3} size={9.5}>{`${fmt.int(ctx)} tokens · ${(ctx * KV_TOK / 2**20).toFixed(1)} MiB of KV`}</Label>

        <g>
          {/* cached prefix */}
          <rect x={0} y={36} width={Math.min(MAXW, t * pxPerTok)} height={26} rx={2} fill="var(--memory)" opacity={0.85} />
          {hitRate > 0.02 && (
            <text x={Math.min(MAXW, t * pxPerTok) / 2} y={53} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fontWeight: 600, fill: '#0e1116' }}>
              {`${fmt.int(cached)} tokens cached`}
            </text>
          )}
          {/* the hit/miss boundary */}
          {hitRate > 0.02 && (
            <g>
              <line x1={t * pxPerTok} x2={t * pxPerTok} y1={30} y2={68} stroke="var(--text)" strokeWidth={1.5} />
              <Label x={t * pxPerTok} y={80} anchor="middle" size={9.5} kind="memory">{`${fmt.int(cached)}`}</Label>
            </g>
          )}
          {/* the fresh suffix */}
          <rect x={t * pxPerTok + 1} y={36} width={Math.max(0, Math.min(MAXW, frac * MAXW) - t * pxPerTok)} height={26} rx={2} fill="var(--compute)" opacity={0.9} />
          {ctx - cached > 20 && (
            <text x={(t * pxPerTok + frac * MAXW) / 2} y={53} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fontWeight: 600, fill: '#0e1116' }}>
              {`${fmt.int(ctx - cached)} to compute`}
            </text>
          )}
          <rect x={0} y={36} width={MAXW} height={26} rx={2} fill="none" stroke="var(--rule-2)" />
        </g>

        <Fade show={s.step >= 1}>
          <Arrow x1={frac * MAXW + 6} x2={frac * MAXW + 40} y1={72} y2={72} kind="error" dashed />
          <Label x={frac * MAXW + 44} y={76} size={10} kind="error">{`${fmt.int(cached)} wasted without a cache`}</Label>
        </Fade>

        {/* ── cost breakdown ── */}
        <Label x={0} y={116} tone={2} size={11} bold>Prefill cost</Label>
        <Label x={0} y={140} tone={3} size={9.5}>cold</Label>
        <rect x={40} y={132} width={MAXW} height={14} rx={2} fill="none" stroke="var(--rule)" />
        <Span x={40} y={132} w={Math.max(1, MAXW * (ctx / 2600))} h={14} kind="error" opacity={0.55} />
        <Label x={44 + MAXW * (ctx / 2600) + 6} y={143} size={10} kind="error">{`${(total * 1000).toFixed(0)} ms`}</Label>

        <Label x={0} y={172} tone={3} size={9.5}>cached</Label>
        <rect x={40} y={164} width={MAXW} height={14} rx={2} fill="none" stroke="var(--rule)" />
        <Span x={40} y={164} w={Math.max(1, MAXW * ((ctx - cached) / 2600))} h={14} kind="memory" />
        <Label x={44 + MAXW * ((ctx - cached) / 2600) + 6} y={175} size={10} kind="memory">{`${(cold * 1000).toFixed(0)} ms`}</Label>

        <Fade show={saved > 0.001}>
          <Dim x1={40 + MAXW * ((ctx - cached) / 2600)} x2={40 + MAXW * (ctx / 2600)} y={194} kind="token"
            label={`${((saved / total) * 100).toFixed(0)}% of prefill removed`} />
        </Fade>

        {/* ── block-level view ── */}
        <Fade show={s.step >= 4}>
          <Label x={0} y={232} tone={2} size={11} bold>Across a whole workload</Label>
          <rect x={0} y={242} width={MAXW} height={16} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={0} y={242} w={MAXW * trafficHit} h={16} kind="token" opacity={0.85} />
          <Label x={MAXW * trafficHit + 8} y={254} size={10} kind="token">{`${(trafficHit * 100).toFixed(0)}% hit rate`}</Label>
          <Label x={0} y={278} size={10} tone={3}>{`A ${fmt.int(ctx)}-token prompt at this hit rate costs ${(prefillSec(ctx * (1 - trafficHit), llama8b).t * 1000).toFixed(0)} ms of prefill instead of ${(total * 1000).toFixed(0)} ms.`}</Label>
          <Label x={0} y={296} size={9.5} tone={3}>{`Multi-turn chat is the easy case: the conversation above you is already a shared prefix.`}</Label>
        </Fade>

        <Fade show={s.step >= 5}>
          <Label x={0} y={334} tone={2} size={11} bold>What the cache can actually reuse (16-token KV blocks)</Label>
          <g>
            {Array.from({ length: shownBlocks }, (_, i) => {
              const from = i * 100;
              const isCached = from < cached - (cached % 100);
              const partial = !isCached && from < cached;
              return (
                <rect
                  key={i}
                  x={i * 20.6}
                  y={344}
                  width={18}
                  height={16}
                  rx={2}
                  fill={isCached ? 'var(--memory)' : partial ? 'url(#tps-hatch-red)' : from < ctx ? 'var(--compute)' : 'transparent'}
                  stroke={isCached ? 'var(--memory)' : from < ctx ? 'var(--compute)' : 'var(--rule-2)'}
                  opacity={from < ctx ? 0.9 : 0.5}
                  strokeDasharray={from < ctx ? undefined : '3 3'}
                />
              );
            })}
          </g>
          <Label x={0} y={372} size={9.5} tone={3}>
            Reusable (memory) · must compute (compute) · a partial block at the boundary is wasted, because only whole 16-token KV blocks can be shared.
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}