import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Span, Label, Fade, fmt } from '../lib/svg';
import {
  llama8b, h100, kvBudget, seqKvBytes, prefillTTFT, decodeTerms, bandwidthWall, computeWall,
} from './kvMath';

const BATCH = 32; // conversations in flight, the thread running through the course
const WORKSPACE = 0.04 * h100.mem;
const WEIGHTS = llama8b.params * 2;
const CARD = h100.mem;
const BUDGET = kvBudget(llama8b, h100);

// The two walls, in tokens. Both are derived, not tuned.
const WALL_COMPUTE = computeWall(llama8b);   // prefill attention overtakes the projections
const WALL_BW = bandwidthWall(llama8b);      // the KV read overtakes the weight read

const STEPS = [
  'A service that fits',
  'The cache grows linearly',
  'Wall one: memory',
  'Wall two: prefill arithmetic',
  'Wall three: decode bandwidth',
  'All three, together',
  'Three different problems',
];
const CAPTIONS = [
  'One H100 serving 32 conversations at 4,096 tokens each. Weights, workspace and KV all fit with room to spare. Every panel on this canvas is healthy, and this is the picture you have been carrying since Module 05.',
  'Slide the context out to 32,768 tokens and nothing has broken — but every number is now eight times worse. Nothing about the model changed. The same arithmetic, evaluated at a bigger input, is the whole problem.',
  'At 128K the 32 caches alone want 549 GB. That is not a slowdown; it is a refusal. Memory is the first wall, it arrives first, and it is the only one of the three that is a hard crash rather than a slow answer.',
  'Prefill is quadratic. Attention FLOPs grow as the square of the length, so past about 30,600 tokens the attention matrix costs more than every projection in the model combined. This is Wall two, and it is a compute wall.',
  'Decode is linear but relentless: every step reads the whole cache again. Past about 122,500 tokens the cache read costs more than reading all the weights. This is Wall three, and it is a bandwidth wall.',
  'Back to 32K, where nothing has crashed yet. All three effects are already present and all three get worse together, which is why long context is hard in a way that a single slow component is not.',
  'The walls are different in kind. Memory is binary and static, prefill is one expensive event, and decode is a permanent tax on every token for the rest of the conversation. Fixing one does nothing for the others.',
];

// Context length and which panel each step is about.
const PRESET: Record<number, { ctx: number; focus: number }> = {
  0: { ctx: 4096, focus: -1 },
  1: { ctx: 32768, focus: 0 },
  2: { ctx: 131072, focus: 0 },
  3: { ctx: 131072, focus: 1 },
  4: { ctx: 131072, focus: 2 },
  5: { ctx: 32768, focus: -1 },
  6: { ctx: 131072, focus: -1 },
};

const X0 = 112;
const X1 = 620;
const CTX_MIN = 512;
const CTX_MAX = 131072;
const lx = (ctx: number) => (Math.log2(Math.min(CTX_MAX, Math.max(CTX_MIN, ctx))) - Math.log2(CTX_MIN)) / (Math.log2(CTX_MAX) - Math.log2(CTX_MIN));
const cx = (ctx: number) => X0 + lx(ctx) * (X1 - X0);

const TICKS = [1024, 4096, 16384, 65536, 131072];
const tickLabel = (v: number) => (v >= 1024 ? `${v / 1024}K` : `${v}`);

// Panel B: prefill TTFT, log scale from 10 ms to 100 s.
const B_TOP = 138;
const B_BOT = 236;
const T_MIN = 1e-2;
const T_MAX = 1e2;
const ty = (s: number) => B_BOT - ((Math.log10(Math.min(T_MAX, Math.max(T_MIN, s))) - Math.log10(T_MIN)) / (Math.log10(T_MAX) - Math.log10(T_MIN))) * (B_BOT - B_TOP);

// Panel C: KV read per sequence, 0 to 8 ms.
const C_TOP = 288;
const C_BOT = 352;
const KV_MAX = 8e-3;
const ky = (s: number) => C_BOT - Math.min(1, s / KV_MAX) * (C_BOT - C_TOP);

const CXTICKS = [1e-2, 1e-1, 1, 10, 100];

function curve(fn: (c: number) => number, map: (v: number) => number) {
  const pts: string[] = [];
  for (let i = 0; i <= 120; i++) {
    const ctx = CTX_MIN * Math.pow(CTX_MAX / CTX_MIN, i / 120);
    pts.push(`${pts.length ? 'L' : 'M'}${cx(ctx).toFixed(1)},${map(fn(ctx)).toFixed(1)}`);
  }
  return pts.join(' ');
}

export default function ContextWalls() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [ctx, setCtx] = useState(PRESET[s.step].ctx);
  const [focus, setFocus] = useState(PRESET[s.step].focus);
  useEffect(() => { setCtx(PRESET[s.step].ctx); setFocus(PRESET[s.step].focus); }, [s.step]);

  const kv1 = seqKvBytes(ctx, llama8b);
  const kvBatch = kv1 * BATCH;
  const used = WORKSPACE + WEIGHTS + kvBatch;
  const fits = used <= CARD;
  const convs = Math.floor(BUDGET / kv1);
  const ttft = prefillTTFT(ctx, llama8b, h100).t;
  const step1 = decodeTerms(ctx, llama8b, h100);
  const step32 = decodeTerms(ctx, llama8b, h100).tWeights + (step1.tKv * BATCH);

  // Panel A geometry: the card as a stack of memory consumers.
  const GBX = (X1 - X0) / (CARD / 1e9);
  const wsW = WORKSPACE / 1e9 * GBX;
  const wW = WEIGHTS / 1e9 * GBX;
  const kvW = useTween(Math.min(kvBatch / 1e9 * GBX, (CARD / 1e9) * GBX), 550);
  const overW = useTween(Math.min(Math.max(0, kvBatch - BUDGET) / 1e9 * GBX, (X1 - X0) * 0.5), 550);

  const mx = useTween(cx(ctx), 350);
  const my = useTween(ty(ttft), 500);
  const kmy = useTween(ky(step1.tKv), 400);

  const prefillPath = useMemo(() => curve((c) => prefillTTFT(c, llama8b, h100).t, ty), []);
  const kvPath = useMemo(() => curve((c) => decodeTerms(c, llama8b, h100).tKv, ky), []);

  const dim = (panel: number) => (focus < 0 || focus === panel ? 1 : 0.32);

  const readout = (
    <Readout items={[
      { label: 'KV cache, one conversation', value: fmt.bytes(kv1), kind: 'memory', note: `${fmt.int(ctx)} tokens` },
      {
        label: `${BATCH} conversations fit?`,
        value: fits ? 'yes' : `no — ${fmt.bytes(kvBatch - BUDGET)} over`,
        kind: fits ? 'text' : 'error',
        note: fits ? `${convs} would fit` : `${convs} max at this length`,
      },
      { label: 'Prefill (TTFT)', value: ttft < 1 ? `${(ttft * 1000).toFixed(0)} ms` : `${ttft.toFixed(ttft < 10 ? 2 : 0)} s`, kind: ttft > 1 ? 'compute' : 'text', note: `${fmt.int(ctx)} tokens, one H100` },
      { label: 'Decode step, batch 32', value: `${(step32 * 1000).toFixed(1)} ms`, kind: 'memory', note: `${(1 / step32).toFixed(1)} tok/s per user` },
    ]} />
  );

  return (
    <Stage
      rootRef={ref}
      title="Llama-3.1-8B on one H100: what changes as the context grows"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['error', 'over the card'], 'text']}
      {...s}
      readout={readout}
      controls={
        <Slider
          label="Context length per conversation (tokens)"
          value={ctx}
          min={CTX_MIN}
          max={CTX_MAX}
          log
          onChange={(v) => setCtx(v)}
          format={(v) => `${fmt.int(v)} tokens`}
          kind="memory"
        />
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Three stacked panels against context length from 512 to 131,072 tokens. The top panel is a bar of an H100's 80 GB split into workspace, weights and the KV cache of 32 conversations, which overflows past about 13,000 tokens. The middle panel is a curve of prefill time rising from 30 milliseconds to about 18 seconds, crossing the projection cost at 30,600 tokens. The bottom panel is a curve of KV read time per sequence per decode step, rising to 6 milliseconds and crossing the 6 millisecond weight read at 122,500 tokens.">
        {/* ═════ panel A · memory ═════ */}
        <g style={{ opacity: dim(0) }}>
          <Label x={X0} y={26} tone={2} size={10.5} mono={false} bold>Wall one · does it fit?</Label>
          <Label x={X1} y={26} anchor="end" tone={3} size={9.5}>{`32 conversations · ${fmt.bytes(kvBatch)} of KV`}</Label>

          <rect x={X0} y={38} width={X1 - X0} height={26} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={X0} y={38} w={wsW} h={26} kind="text" opacity={0.5} />
          <Span x={X0 + wsW} y={38} w={wW} h={26} kind="compute" opacity={0.85} />
          <Span x={X0 + wsW + wW} y={38} w={kvW} h={26} kind="memory" />
          {!fits && <rect x={X1 - overW} y={38} width={overW} height={26} rx={2} fill="url(#tps-hatch-red)" opacity={0.85} />}
          {/* where the KV budget runs out */}
          <line x1={X0 + wsW + wW + (BUDGET / 1e9) * GBX} x2={X0 + wsW + wW + (BUDGET / 1e9) * GBX} y1={34} y2={70}
            stroke="var(--error)" strokeWidth={1.2} strokeDasharray="3 3" />
          <Label x={X0 + wsW + wW + (BUDGET / 1e9) * GBX + 4} y={80} size={9} kind="error">KV budget ends here</Label>

          <Label x={X0 + 2} y={53} size={9} kind="text" mono={false}>workspace</Label>
          <Label x={X0 + wsW + wW / 2} y={53} anchor="middle" size={9.5} kind="compute">weights 16.1 GB</Label>
          <Label x={X1} y={93} anchor="end" size={9.5} kind={fits ? 'text' : 'error'}>
            {fits
              ? `${fmt.gb(used)} of 80 GB used · ${convs} conversations at this length`
              : `${fmt.gb(used)} needed · ${fmt.gb(kvBatch - BUDGET)} past the budget`}
          </Label>
          <Line2 x={mx} y1={34} y2={96} />
        </g>

        {/* ═════ panel B · prefill compute ═════ */}
        <g style={{ opacity: dim(1) }}>
          <Label x={X0} y={124} tone={2} size={10.5} mono={false} bold>Wall two · how long the prompt takes</Label>
          <Label x={X1} y={124} anchor="end" tone={3} size={9.5}>prefill, one H100 at 55% MFU</Label>

          <line x1={X0} x2={X1} y1={B_BOT} y2={B_BOT} className="v-rule" />
          <line x1={X0} x2={X0} y1={B_TOP - 6} y2={B_BOT} className="v-rule" />
          {CXTICKS.map((v) => (
            <Label key={v} x={X0 - 6} y={ty(v) + 3} anchor="end" tone={3} size={9}>
              {v < 1 ? `${v * 1000} ms` : `${v} s`}
            </Label>
          ))}

          <path d={prefillPath} fill="none" stroke="var(--compute)" strokeWidth={1.8} />
          <line x1={cx(WALL_COMPUTE)} x2={cx(WALL_COMPUTE)} y1={B_TOP - 4} y2={B_BOT} stroke="var(--compute)" strokeWidth={1} strokeDasharray="4 3" opacity={0.7} />
          <Label x={cx(WALL_COMPUTE) + 4} y={B_TOP + 2} size={9} kind="compute">{`attention = projections at ${fmt.int(WALL_COMPUTE)}`}</Label>

          <circle cx={mx} cy={my} r={4.5} fill="var(--compute)" stroke="var(--ground-2)" strokeWidth={2} />
          <Label x={Math.min(mx, X1 - 46)} y={my - 9} anchor={mx > X1 - 90 ? 'end' : 'start'} size={9.5} kind="compute">
            {ttft < 1 ? `${(ttft * 1000).toFixed(0)} ms` : `${ttft.toFixed(1)} s`}
          </Label>
          <Line2 x={mx} y1={B_TOP - 6} y2={B_BOT} />
        </g>

        {/* ═════ panel C · decode bandwidth ═════ */}
        <g style={{ opacity: dim(2) }}>
          <Label x={X0} y={274} tone={2} size={10.5} mono={false} bold>Wall three · what each token costs</Label>
          <Label x={X1} y={274} anchor="end" tone={3} size={9.5}>KV read per sequence, one decode step</Label>

          <line x1={X0} x2={X1} y1={C_BOT} y2={C_BOT} className="v-rule" />
          <line x1={X0} x2={X0} y1={C_TOP - 4} y2={C_BOT} className="v-rule" />
          {[0, 2, 4, 6, 8].map((v) => (
            <Label key={v} x={X0 - 6} y={ky(v / 1000) + 3} anchor="end" tone={3} size={9}>{`${v} ms`}</Label>
          ))}

          {/* the weight read: the constant every step also pays */}
          <line x1={X0} x2={X1} y1={ky(decodeTerms(0, llama8b, h100).tWeights)} y2={ky(decodeTerms(0, llama8b, h100).tWeights)}
            stroke="var(--compute)" strokeWidth={1.3} strokeDasharray="5 3" />
          <Label x={X0 + 4} y={ky(decodeTerms(0, llama8b, h100).tWeights) - 5} size={9} kind="compute">
            reading all 16.1 GB of weights = 6.0 ms
          </Label>

          <path d={kvPath} fill="none" stroke="var(--memory)" strokeWidth={1.8} />
          <line x1={cx(WALL_BW)} x2={cx(WALL_BW)} y1={C_TOP - 4} y2={C_BOT} stroke="var(--memory)" strokeWidth={1} strokeDasharray="4 3" opacity={0.7} />
          <Label x={cx(WALL_BW) - 4} y={C_TOP + 2} anchor="end" size={9} kind="memory">{`equal at ${fmt.int(WALL_BW)}`}</Label>

          <circle cx={mx} cy={kmy} r={4.5} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={2} />
          <Label x={Math.min(mx, X1 - 40)} y={kmy - 9} anchor={mx > X1 - 80 ? 'end' : 'start'} size={9.5} kind="memory">{(step1.tKv * 1000).toFixed(2)} ms</Label>
          <Line2 x={mx} y1={C_TOP - 4} y2={C_BOT} />
        </g>

        {/* ═════ shared x-axis ═════ */}
        <line x1={X0} x2={X1} y1={C_BOT} y2={C_BOT} className="v-rule faint" />
        {TICKS.map((v) => (
          <Label key={v} x={cx(v)} y={C_BOT + 14} anchor="middle" tone={3} size={9}>{tickLabel(v)}</Label>
        ))}
        <Label x={X1} y={C_BOT + 28} anchor="end" tone={3} size={9}>context length (tokens) →</Label>

        <Fade show={s.step === 6}>
          <rect x={X0 - 8} y={16} width={X1 - X0 + 16} height={C_BOT + 34} rx={4} fill="none" stroke="var(--rule-2)" strokeDasharray="4 4" />
        </Fade>
      </svg>
    </Stage>
  );
}

/** The vertical guide that tracks the slider across all three panels. */
function Line2({ x, y1, y2 }: { x: number; y1: number; y2: number }) {
  return <line x1={x} x2={x} y1={y1} y2={y2} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="2 3" />;
}
