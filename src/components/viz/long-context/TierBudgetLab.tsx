import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Span, Label, fmt } from '../lib/svg';
import { llama8b, h100, kvBudget, seqKvBytes, humanTime } from './kvMath';

// Llama-3.1-8B, BF16, one H100. The configuration the whole module has used.
const BUDGET = kvBudget(llama8b, h100);        // ~54.7 GB of KV the card can hold
const KV_TOK = seqKvBytes(1, llama8b);          // 131,072 B per token
const WEIGHTS = llama8b.params * 2;             // 16.06 GB
const T_WEIGHTS = WEIGHTS / (h100.bw * 0.8);    // 6.0 ms of the decode step

// Tier ladder, same numbers as the lesson.
const TIERS = [
  { id: 'dram', name: 'Host DRAM (PCIe 5.0)', bw: 55e9 },
  { id: 'nvme', name: 'Local NVMe', bw: 7e9 },
  { id: 'remote', name: 'Remote node (10 GbE)', bw: 1.25e9 },
] as const;
type TierId = (typeof TIERS)[number]['id'];
const bwOf = (id: TierId) => TIERS.find((t) => t.id === id)!.bw;

// The streaming budget: bytes you can move per step with zero added latency.
const BUDGET_PCIE = bwOf('dram') * T_WEIGHTS;  // ~330 MB

const RECOMPUTE = (2 * llama8b.params) / (h100.flops * 0.55); // 29.5 us per token
const XOVER_BW = KV_TOK / RECOMPUTE;                          // ~4.4 GB/s

// ── geometry
const PX0 = 96;
const PX1 = 616;
const BYTES_MIN = 2e8;   // 200 MB
const BYTES_MAX = 4e10;  // 40 GB
const bx = (b: number) =>
  PX0 + (Math.log10(Math.min(BYTES_MAX, Math.max(BYTES_MIN, b))) - Math.log10(BYTES_MIN)) /
    (Math.log10(BYTES_MAX) - Math.log10(BYTES_MIN)) * (PX1 - PX0);

const CTX_MIN = 1024;
const CTX_MAX = 131072;
const lx = (c: number) => (Math.log2(Math.min(CTX_MAX, Math.max(CTX_MIN, c))) - Math.log2(CTX_MIN)) / (Math.log2(CTX_MAX) - Math.log2(CTX_MIN));
const cx = (c: number) => PX0 + lx(c) * (PX1 - PX0);

const CTX_TICKS = [1024, 4096, 16384, 65536, 131072];
const ctxLabel = (v: number) => (v >= 1024 ? `${v / 1024}K` : `${v}`);

// ── panels
const A_Y = 52;
const A_H = 40;
const C_Y = 128;
const C_BOT = 262;
const SPEED_MIN = 0.05;
const SPEED_MAX = 200;
const sy = (v: number) =>
  C_BOT - (Math.log10(Math.min(SPEED_MAX, Math.max(SPEED_MIN, v))) - Math.log10(SPEED_MIN)) /
    (Math.log10(SPEED_MAX) - Math.log10(SPEED_MIN)) * (C_BOT - C_Y);

const E_Y = 306;
const E_H = 34;

const STEPS = [
  'One card, three tiers',
  'What stays resident',
  'What gets streamed',
  'The streaming budget',
  'What it costs you',
  'Sweep the context',
  'Write down the answer',
];
const CAPTIONS = [
  'The H100 running Llama-3.1-8B. Weights sit on the GPU and never move, and the card has 54.7 GB left for KV after the workspace and headroom — about 417,000 tokens in total, whether it is one conversation or two hundred.',
  'Everything in the resident window lives in GPU memory permanently. It costs nothing extra per step beyond the read you were already paying, and in practice the last few thousand tokens carry most of the attention mass, so this is where you put your money.',
  'Everything between the shared prefix and the resident window is streamed from a cheaper tier, fresh every step. The useful thing about this region is that it is big, cold and identical across similar requests — so it is also the part worth deduplicating.',
  'Streaming is free only up to a budget: tier bandwidth times the time the step was already spending on weights. Over PCIe that is 330 MB per step. Past that line the fetch starts showing up as latency, and this is the number to design against.',
  'Now the cost. Pulling bytes from a slow tier every step drags on the critical path, and per-user speed falls off a cliff. Speed is what your users see; capacity is what your finance team sees.',
  'Sweep the context and the tension becomes obvious: the resident window that keeps you fast grows exactly as fast as the thing you cannot afford. Everything to the right of the budget line is a problem you have to solve with capacity, not with tuning.',
  'The deliverable of this lab is a number, not a feeling: the maximum context per conversation at your concurrency target, for each tier. Everything else follows from where the budget line sits.',
];

const PRESET: Record<number, { ctx: number; resident: number; tier: TierId; focus: number }> = {
  0: { ctx: 8192, resident: 2048, tier: 'dram', focus: -1 },
  1: { ctx: 8192, resident: 2048, tier: 'dram', focus: 0 },
  2: { ctx: 8192, resident: 2048, tier: 'dram', focus: 1 },
  3: { ctx: 8192, resident: 2048, tier: 'dram', focus: 2 },
  4: { ctx: 8192, resident: 2048, tier: 'dram', focus: 3 },
  5: { ctx: 8192, resident: 2048, tier: 'dram', focus: 4 },
  6: { ctx: 8192, resident: 2048, tier: 'dram', focus: -1 },
};

// Where the card's KV budget runs out, in tokens.
const TOK_BUDGET = BUDGET / KV_TOK; // ~417,000

/** Per-decode-step model: resident read from HBM, streamed read from the tier. */
function step(ctx: number, resident: number, tier: TierId) {
  const resBytes = Math.min(resident, ctx) * KV_TOK;
  const streamBytes = Math.max(0, ctx - resident) * KV_TOK;
  const tRes = resBytes / (h100.bw * 0.8);
  const tStream = streamBytes / bwOf(tier);
  const tW = T_WEIGHTS;
  // A step is max(weights, cache) when everything is local; a streamed tail adds its own term.
  const tStep = Math.max(tW, tRes) + tStream;
  return { resBytes, streamBytes, tRes, tStream, tW, tStep };
}

/** How many conversations of this length fit on the card. */
const conversations = (ctx: number) => Math.floor(TOK_BUDGET / ctx);

export default function TierBudgetLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const p0 = PRESET[s.step];
  const [ctx, setCtx] = useState(p0.ctx);
  const [resident, setResident] = useState(p0.resident);
  const [tier, setTier] = useState<TierId>(p0.tier);
  const [focus, setFocus] = useState(p0.focus);
  useEffect(() => {
    setCtx(PRESET[s.step].ctx);
    setResident(PRESET[s.step].resident);
    setTier(PRESET[s.step].tier);
    setFocus(PRESET[s.step].focus);
  }, [s.step]);

  const st = step(ctx, resident, tier);
  const conv = conversations(ctx);
  const overBudget = st.streamBytes > BUDGET_PCIE;
  const exceedsCard = conv < 1;
  const speed = 1 / st.tStep;

  // Speed, and where each tier lands on the fetch-vs-recompute scale.
  const tierSpeeds = useMemo(
    () => TIERS.map((t) => ({ id: t.id, name: t.name, wins: t.bw > XOVER_BW })),
    [],
  );

  const mx = useTween(cx(ctx), 300);
  const sSpeed = 1 / st.tStep;

  const dim = (panel: number) => (focus < 0 || focus === panel ? 1 : 0.3);

  const readout = (
    <Readout items={[
      { label: 'GPU-resident KV', value: fmt.bytes(st.resBytes), kind: 'memory', note: `${fmt.int(resident)} tokens` },
      {
        label: 'Streamed per step',
        value: fmt.bytes(st.streamBytes),
        kind: overBudget ? 'error' : 'memory',
        note: overBudget ? 'over the per-step budget' : `${humanTime(st.tStream)} to fetch`,
      },
      {
        label: 'Conversations on the card',
        value: exceedsCard ? '0' : String(conv),
        kind: conv < 8 ? 'error' : 'text',
        note: `needs ${fmt.gb(ctx * KV_TOK)} each`,
      },
      {
        label: 'Decode step',
        value: `${(st.tStep * 1000).toFixed(1)} ms`,
        kind: overBudget ? 'error' : 'memory',
        note: `${speed.toFixed(speed < 10 ? 1 : 0)} tok/s per user`,
      },
    ]} />
  );

  return (
    <Stage
      rootRef={ref}
      title="Lab: how much context fits once the cache is tiered"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', ['comm', 'streamed'], ['error', 'over budget'], 'text']}
      {...s}
      readout={readout}
      controls={
        <>
          <Slider label="Context per conversation (tokens)" value={ctx} min={CTX_MIN} max={CTX_MAX} log
            onChange={setCtx} format={(v) => `${fmt.int(v)} tokens`} kind="memory" />
          <Slider label="Resident window in GPU memory (tokens)" value={resident} min={256} max={Math.max(256, ctx)} log
            onChange={setResident} format={(v) => `${fmt.int(v)} tokens`} kind="memory" />
          <Segmented label="Tier for the cold tail" value={tier}
            options={TIERS.map((t) => ({ value: t.id, label: t.name.split(' (')[0] }))}
            onChange={(v) => setTier(v as TierId)} />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Three stacked panels against context length. The top panel shows the H100's 54.7 GB KV budget as a bar, with a green resident window inside it and a violet streamed tail, turning red hatched once the streamed portion exceeds the 330 MB per-step budget. The middle panel shows per-user decode speed falling on a log scale with a line per storage tier, with the NVMe line dropping below the remote line because it is faster. The bottom panel shows the maximum context achievable per conversation against concurrency, with a falling frontier as the streamed tail grows.">
        {/* ═════ panel A · the card and the split ═════ */}
        <g style={{ opacity: dim(0) }}>
          <Label x={20} y={20} tone={2} size={10.5} mono={false} bold>The card, and how the cache is split</Label>
          <Label x={620} y={20} anchor="end" tone={3} size={9.5}>{`${fmt.int(TOK_BUDGET)} tokens of KV total`}</Label>

          <rect x={PX0} y={A_Y} width={PX1 - PX0} height={A_H} rx={2} fill="none" stroke="var(--rule)" />
          {/* the resident/streamed split of one conversation, drawn as bytes against the card's budget */}
          {(() => {
            const full = BUDGET;
            const wRes = Math.min(st.resBytes, full) / full * (PX1 - PX0);
            const wStr = Math.min(st.streamBytes, Math.max(0, full - st.resBytes)) / full * (PX1 - PX0);
            return (
              <>
                <Span x={PX0} y={A_Y} w={wRes} h={A_H} kind="memory" opacity={0.9} />
                <Span x={PX0 + wRes} y={A_Y} w={wStr} h={A_H} kind="comm" opacity={0.9} />
                {wRes > 46 && <Label x={PX0 + wRes / 2} y={A_Y + A_H / 2 + 3.5} anchor="middle" size={9} kind="memory">{`resident ${fmt.int(resident)}`}</Label>}
                <Label x={PX0} y={A_Y - 8} size={9} kind="comm">{`streamed: ${fmt.int(Math.max(0, ctx - resident))} tokens → ${fmt.bytes(st.streamBytes)}`}</Label>
              </>
            );
          })()}
          <Line3 x={mx} y1={A_Y + A_H + 4} y2={E_Y + E_H + 4} />
          <Label x={PX1} y={A_Y + A_H + 28} anchor="end" size={9} kind={overBudget ? 'error' : 'memory'}>
            {overBudget
              ? `streamed ${fmt.bytes(st.streamBytes)} is over the ${fmt.bytes(BUDGET_PCIE)} per-step budget`
              : `streamed ${fmt.bytes(st.streamBytes)} within the ${fmt.bytes(BUDGET_PCIE)} budget`}
          </Label>
        </g>

        {/* ═════ panel B · speed ═════ */}
        <g style={{ opacity: dim(1) }}>
          <Label x={20} y={112} tone={2} size={10.5} mono={false} bold>What it costs: per-user speed</Label>
          <Label x={620} y={112} anchor="end" tone={3} size={9.5}>tokens/s, log scale</Label>

          <line x1={PX0} x2={PX1} y1={C_BOT} y2={C_BOT} className="v-rule" />
          <line x1={PX0} x2={PX0} y1={C_Y - 4} y2={C_BOT} className="v-rule" />
          {[1, 10, 100].map((v) => (
            <Label key={v} x={PX0 - 6} y={sy(v) + 3} anchor="end" tone={3} size={9}>{v}</Label>
          ))}
          <Label x={PX0 - 6} y={sy(SPEED_MIN) + 3} anchor="end" tone={3} size={9}>0.05</Label>

          {tierSpeeds.map((ts) => {
            const on = ts.id === tier;
            const sw = 1 / step(ctx, Math.min(resident, ctx), ts.id).tStep;
            return (
              <g key={ts.id}>
                <SpeedLine resident={resident} tier={ts.id} />
                <Label x={PX1} y={sy(sw) - 5 - tierSpeeds.filter((z) => z.id !== ts.id && 1 / step(ctx, Math.min(resident, ctx), z.id).tStep > sw).length * 11}
                  anchor="end" size={9} kind={on ? (ts.wins ? 'memory' : 'error') : 'text'}
                  style={{ opacity: on ? 1 : 0.6 }}>
                  {ts.name.split(' (')[0]} {sw < 10 ? sw.toFixed(1) : sw.toFixed(0)}
                </Label>
              </g>
            );
          })}

          <circle cx={mx} cy={sy(speed)} r={4.5} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={2} />
          <Label x={Math.min(mx, PX1 - 34)} y={sy(speed) + 16} anchor={mx > PX1 - 70 ? 'end' : 'start'} size={9.5} kind="memory">
            {speed < 10 ? speed.toFixed(1) : speed.toFixed(0)} tok/s
          </Label>
        </g>

        {/* ═════ panel C · the frontier ═════ */}
        <g style={{ opacity: dim(2) }}>
          <Label x={20} y={272} tone={2} size={10.5} mono={false} bold>The frontier: context vs concurrency</Label>
          <Label x={620} y={272} anchor="end" tone={3} size={9.5}>conversations that fit on this card</Label>

          <rect x={PX0} y={E_Y} width={PX1 - PX0} height={E_H} rx={2} fill="none" stroke="var(--rule)" />
          {(() => {
            const w = Math.min(1, conv / 200) * (PX1 - PX0);
            return (
              <>
                <Span x={PX0} y={E_Y} w={w} h={E_H} kind="memory" opacity={0.85} />
                {w > 60 && (
                  <Label x={PX0 + w / 2} y={E_Y + E_H / 2 + 3.5} anchor="middle" size={9.5} kind="memory">
                    {exceedsCard ? 'none fit' : `${conv} conversations`}
                  </Label>
                )}
              </>
            );
          })()}
          <Line3 x={mx} y1={E_Y - 6} y2={E_Y + E_H + 6} />
          <Label x={PX1} y={E_Y - 10} anchor="end" tone={3} size={9}>{`${fmt.int(ctx)} tokens each`}</Label>
          <Label x={PX1} y={E_Y + E_H + 16} anchor="end" size={9} kind={conv < 8 ? 'error' : 'text'}>
            {exceedsCard
              ? `${fmt.gb(ctx * KV_TOK)} needed — past this card entirely`
              : `${fmt.gb(ctx * KV_TOK)} each · ${fmt.gb(BUDGET)} available`}
          </Label>
        </g>

        {/* ═════ shared x-axis ═════ */}
        {CTX_TICKS.map((v) => (
          <Label key={v} x={cx(v)} y={C_BOT + 14} anchor="middle" tone={3} size={9}>{ctxLabel(v)}</Label>
        ))}
        <Label x={PX1} y={C_BOT + 26} anchor="end" tone={3} size={9}>context per conversation (tokens) →</Label>
      </svg>
    </Stage>
  );
}

/** One tier's speed curve over context, with its own resident split. */
function SpeedLine({ resident, tier }: { resident: number; tier: TierId }) {
  const pts: string[] = [];
  for (let i = 0; i <= 60; i++) {
    const c = CTX_MIN * Math.pow(CTX_MAX / CTX_MIN, i / 60);
    const r = Math.min(resident, c);
    const s = 1 / step(c, r, tier).tStep;
    pts.push(`${pts.length ? 'L' : 'M'}${cx(c).toFixed(1)},${sy(s).toFixed(1)}`);
  }
  const on = tier === 'dram';
  return <path d={pts.join(' ')} fill="none" stroke={on ? 'var(--memory)' : 'var(--text-3)'} strokeWidth={on ? 2 : 1.4} opacity={on ? 1 : 0.55} />;
}

/** The vertical guide that tracks the context slider across panels. */
function Line3({ x, y1, y2 }: { x: number; y1: number; y2: number }) {
  return <line x1={x} x2={x} y1={y1} y2={y2} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="2 3" />;
}
