import { useRef } from 'react';
import { Stage, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade } from '../lib/svg';

// The 2017 Transformer block next to the LLaMA block, one change at a time.
// LLaMA-7B numbers: d = 4096, 32 layers, 32 heads (full multi-head attention), FFN width 11,008, vocab 32,000.
const D = 4096;
const FFN_2017 = 4 * D; // 16,384 with two matrices
const FFN_LLAMA = 256 * Math.ceil(Math.floor((2 * 4 * D) / 3) / 256); // ⅔·4d = 10,922 → rounded up to a multiple of 256 = 11,008
const P_2017 = 2 * D * FFN_2017;
const P_LLAMA = 3 * D * FFN_LLAMA;

type Focus = 'none' | 'norm-place' | 'norm-kind' | 'ffn' | 'pos' | 'budget' | 'all';
const STEPS = [
  'The 2017 block',
  'Change 1: normalise first (pre-norm)',
  'Change 2: LayerNorm becomes RMSNorm',
  'Change 3: ReLU becomes SwiGLU',
  'Change 4: positions become rotations (RoPE)',
  'Same parameter budget',
  'LLaMA-7B, in numbers',
];
const FOCUS: Focus[] = ['none', 'norm-place', 'norm-kind', 'ffn', 'pos', 'budget', 'all'];
const CAPTIONS = [
  'The original Transformer block: sinusoidal positions are added to the embeddings once, at the bottom. Each sublayer (attention, then a ReLU feed-forward network 4× wider than the model) is followed by "add the residual, then LayerNorm".',
  'LLaMA moves the norm in front of each sublayer and leaves the residual path untouched: x + Attn(Norm(x)). Gradients flow straight down the residual line, so deep stacks train stably without delicate warm-up tricks.',
  'LayerNorm subtracts the mean, divides by the standard deviation, then applies a gain and a bias. RMSNorm only divides by the root-mean-square and applies a gain. Cheaper, and it works as well.',
  'The ReLU feed-forward network (two matrices, 4d wide) becomes SwiGLU: a SiLU-gated product of two branches, then a down-projection. Three matrices, so the width shrinks to ⅔ · 4d to keep the parameter count: 10,923, rounded up to 11,008 for LLaMA-7B.',
  'No position vector is added at the bottom. Instead, every layer rotates its query and key vectors by an angle proportional to the token’s position (RoPE), so attention scores depend on relative distance. Values are never rotated.',
  'Two matrices of 4096 × 16,384 hold 134.2M weights; three of 4096 × 11,008 hold 135.3M. The ⅔ rule keeps the feed-forward budget the same, so the gain comes from the gating, not from extra size.',
  'LLaMA-7B: 32 blocks like the one on the right, d = 4096, 32 attention heads (plain multi-head attention, no GQA yet), FFN 11,008, a 32,000-token vocabulary and a 2,048-token context: 6.7 billion parameters.',
];

const LX = 40, RX = 400, BW = 200; // column x and box width
const H = 26;

function Box({ x, y, label, kind, hl, ghost, sub }: { x: number; y: number; label: string; kind: 'compute' | 'text'; hl?: boolean; ghost?: boolean; sub?: string }) {
  const fill = ghost ? 'none' : kind === 'compute' ? 'var(--compute-dim)' : 'var(--ground-3)';
  const stroke = hl ? 'var(--text)' : kind === 'compute' ? 'var(--compute)' : 'var(--rule-2)';
  return (
    <g className="v-fade" style={{ opacity: ghost ? 0.35 : 1 }}>
      <rect x={x} y={y} width={BW} height={H} rx={3} fill={fill} stroke={stroke} strokeWidth={hl ? 2 : 1} strokeDasharray={ghost ? '3 3' : undefined} style={{ transition: 'stroke .4s, stroke-width .4s' }} />
      <text x={x + BW / 2} y={y + (sub ? 11 : 17)} textAnchor="middle" className="v-label t1" style={{ fontSize: 10.5, fontWeight: hl ? 700 : 600 }}>{label}</text>
      {sub && <text x={x + BW / 2} y={y + 22} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{sub}</text>}
    </g>
  );
}

function AddDot({ cx, cy, hl }: { cx: number; cy: number; hl?: boolean }) {
  return (
    <g>
      <circle cx={cx} cy={cy} r={8} fill="var(--ground-2)" stroke={hl ? 'var(--text)' : 'var(--text-2)'} strokeWidth={hl ? 2 : 1.2} />
      <text x={cx} y={cy + 4} textAnchor="middle" className="v-label t1" style={{ fontSize: 12 }}>+</text>
    </g>
  );
}

export default function ArchDiff() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const f = FOCUS[st];
  const right = st >= 1;
  const hlNorm = f === 'norm-place' || f === 'norm-kind';
  const hlFfn = f === 'ffn' || f === 'budget';
  const hlPos = f === 'pos';
  const rmsLabel = st >= 2 ? 'RMSNorm' : 'LayerNorm';
  const ffnLabel = st >= 3 ? 'SwiGLU feed-forward' : 'ReLU feed-forward';
  const ffnSub = st >= 3 ? `3 matrices · width ${FFN_LLAMA.toLocaleString('en-US')}` : `2 matrices · width ${FFN_2017.toLocaleString('en-US')}`;
  const attnSub = st >= 4 ? 'RoPE rotates Q and K here' : 'no position information here';

  // residual spine x positions
  const LS = LX - 14;
  const RS = RX - 14;

  return (
    <Stage
      rootRef={ref}
      title="From the 2017 Transformer to LLaMA: four changes"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['compute', 'matrix multiplies'], ['text', 'norms, adds, positions']]}
      {...s}
      readout={
        <Readout items={[
          { label: '2017 FFN, one layer', value: `${(P_2017 / 1e6).toFixed(1)}M`, kind: 'compute', note: `2 × 4096 × ${FFN_2017.toLocaleString('en-US')}` },
          { label: 'LLaMA-7B FFN, one layer', value: st >= 3 ? `${(P_LLAMA / 1e6).toFixed(1)}M` : '—', kind: 'compute', note: st >= 3 ? `3 × 4096 × ${FFN_LLAMA.toLocaleString('en-US')}` : 'after change 3' },
          { label: 'norms per layer', value: st >= 1 ? '2 before + 0 after' : '0 before + 2 after', kind: 'text', note: st >= 2 ? 'RMSNorm: gain only' : 'LayerNorm: gain + bias' },
        ]} />
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Two transformer blocks side by side. Left: the 2017 Transformer with sinusoidal positions added at the input, attention and a ReLU feed-forward network each followed by add and LayerNorm. Right: the LLaMA block with RMSNorm before each sublayer, rotary position embeddings applied to queries and keys inside attention, and a SwiGLU feed-forward network.">
        <Label x={LX + BW / 2} y={22} anchor="middle" tone={1} size={11} bold>2017 Transformer</Label>
        <Label x={RX + BW / 2} y={22} anchor="middle" tone={1} size={11} bold>LLaMA (2023)</Label>

        {/* ── left: post-norm block ── */}
        <line x1={LS} x2={LS} y1={300} y2={70} stroke="var(--text-3)" strokeWidth={1.2} />
        <Box x={LX} y={316} label="token embedding" kind="text" />
        <AddDot cx={LX + BW / 2} cy={296} hl={hlPos} />
        <Label x={LX + BW / 2 + 14} y={300} tone={hlPos ? 1 : 2} size={9.5} bold={hlPos}>sinusoidal positions</Label>
        <Box x={LX} y={248} label="multi-head attention" sub="no position information here" kind="compute" />
        <Box x={LX} y={208} label="add, then LayerNorm" kind="text" hl={hlNorm} />
        <Box x={LX} y={156} label="ReLU feed-forward" sub={`2 matrices · width ${FFN_2017.toLocaleString('en-US')}`} kind="compute" hl={hlFfn} />
        <Box x={LX} y={116} label="add, then LayerNorm" kind="text" hl={hlNorm} />
        <Label x={LX + BW / 2} y={92} anchor="middle" tone={3} size={9.5}>× N blocks, then the output head</Label>
        <Arrow x1={LX + BW / 2} y1={316} x2={LX + BW / 2} y2={306} muted />
        <Arrow x1={LX + BW / 2} y1={288} x2={LX + BW / 2} y2={276} muted />
        <Arrow x1={LX + BW / 2} y1={248} x2={LX + BW / 2} y2={236} muted />
        <Arrow x1={LX + BW / 2} y1={208} x2={LX + BW / 2} y2={184} muted />
        <Arrow x1={LX + BW / 2} y1={156} x2={LX + BW / 2} y2={144} muted />
        <Label x={LS - 4} y={232} anchor="end" tone={3} size={8.5}>skip</Label>

        {/* ── right: pre-norm block ── */}
        <g className="v-fade" style={{ opacity: right ? 1 : 0.18 }}>
          <line x1={RS} x2={RS} y1={316} y2={70} stroke="var(--text-2)" strokeWidth={1.6} />
          <Label x={RS - 4} y={84} anchor="end" tone={2} size={8.5}>residual</Label>
          <Label x={RS - 4} y={95} anchor="end" tone={2} size={8.5}>stream</Label>
          <Box x={RX} y={316} label="token embedding" kind="text" />
          <Fade show={st >= 4}>
            <Label x={RX + BW / 2} y={311} anchor="middle" tone={hlPos ? 1 : 3} size={9} bold={hlPos}>nothing added at the input</Label>
          </Fade>
          <Box x={RX} y={268} label={rmsLabel} kind="text" hl={hlNorm} />
          <Box x={RX} y={226} label="multi-head attention" sub={attnSub} kind="compute" hl={hlPos} />
          <AddDot cx={RS} cy={212} />
          <Box x={RX} y={170} label={rmsLabel} kind="text" hl={hlNorm} />
          <Box x={RX} y={128} label={ffnLabel} sub={ffnSub} kind="compute" hl={hlFfn} />
          <AddDot cx={RS} cy={112} />
          <Label x={RX + BW / 2} y={92} anchor="middle" tone={3} size={9.5}>× N blocks, final RMSNorm, output head</Label>
          {/* branches off the stream and back into it */}
          <line x1={RS} x2={RX} y1={294} y2={294} stroke="var(--text-3)" strokeWidth={1} />
          <line x1={RX + 20} x2={RX + 20} y1={294} y2={268 + H} stroke="var(--text-3)" strokeWidth={1} />
          <Arrow x1={RX + BW / 2} y1={268} x2={RX + BW / 2} y2={254} muted />
          <path d={`M${RX + BW / 2},226 L${RX + BW / 2},212 L${RS + 10},212`} fill="none" stroke="var(--text-3)" strokeWidth={1} markerEnd="url(#tps-arrow)" />
          <line x1={RS} x2={RX} y1={202} y2={202} stroke="var(--text-3)" strokeWidth={1} />
          <line x1={RX + 20} x2={RX + 20} y1={202} y2={170 + H} stroke="var(--text-3)" strokeWidth={1} />
          <Arrow x1={RX + BW / 2} y1={170} x2={RX + BW / 2} y2={156} muted />
          <path d={`M${RX + BW / 2},128 L${RX + BW / 2},112 L${RS + 10},112`} fill="none" stroke="var(--text-3)" strokeWidth={1} markerEnd="url(#tps-arrow)" />
        </g>

        {/* ── the change being described ── */}
        <Fade show={f === 'norm-place'}>
          <Label x={320} y={196} anchor="middle" tone={1} size={10} bold>norm moves</Label>
          <Label x={320} y={210} anchor="middle" tone={1} size={10} bold>before the</Label>
          <Label x={320} y={224} anchor="middle" tone={1} size={10} bold>sublayer</Label>
          <Arrow x1={286} y1={232} x2={354} y2={262} />
        </Fade>
        <Fade show={f === 'norm-kind'}>
          <Label x={320} y={188} anchor="middle" tone={2} size={9}>(x − mean) / std</Label>
          <Label x={320} y={200} anchor="middle" tone={2} size={9}>· gain + bias</Label>
          <Label x={320} y={218} anchor="middle" tone={1} size={10} bold>↓</Label>
          <Label x={320} y={234} anchor="middle" tone={1} size={9}>x / rms(x)</Label>
          <Label x={320} y={246} anchor="middle" tone={1} size={9}>· gain</Label>
        </Fade>
        <Fade show={f === 'ffn'}>
          <Label x={320} y={150} anchor="middle" tone={2} size={9}>ReLU(xW₁)W₂</Label>
          <Label x={320} y={166} anchor="middle" tone={1} size={10} bold>↓</Label>
          <Label x={320} y={182} anchor="middle" tone={1} size={9}>(SiLU(xW) ⊙ xV)W₂</Label>
          <Label x={320} y={198} anchor="middle" tone={3} size={9}>⅔ · 4d → 11,008</Label>
        </Fade>
        <Fade show={f === 'pos'}>
          <Label x={320} y={326} anchor="middle" tone={2} size={9}>added once</Label>
          <Label x={320} y={338} anchor="middle" tone={2} size={9}>at the bottom</Label>
          <Arrow x1={300} y1={300} x2={392} y2={242} />
          <Label x={320} y={204} anchor="middle" tone={1} size={9}>rotate Q, K</Label>
          <Label x={320} y={216} anchor="middle" tone={1} size={9}>in every layer</Label>
        </Fade>
        <Fade show={f === 'budget'}>
          <Label x={320} y={148} anchor="middle" size={10} kind="compute" bold>{`${(P_2017 / 1e6).toFixed(1)}M`}</Label>
          <Label x={320} y={162} anchor="middle" tone={2} size={9}>vs</Label>
          <Label x={320} y={176} anchor="middle" size={10} kind="compute" bold>{`${(P_LLAMA / 1e6).toFixed(1)}M`}</Label>
          <Label x={320} y={192} anchor="middle" tone={3} size={9}>per layer</Label>
        </Fade>
        <Fade show={f === 'all'}>
          <Label x={320} y={124} anchor="middle" tone={1} size={10} bold>LLaMA-7B</Label>
          {['d = 4,096', '32 layers', '32 heads (MHA)', 'FFN 11,008', 'vocab 32,000', 'context 2,048', '6.7B params'].map((t, i) => (
            <Label key={t} x={320} y={140 + i * 15} anchor="middle" tone={2} size={9}>{t}</Label>
          ))}
        </Fade>

        <line x1={20} x2={620} y1={352} y2={352} className="v-rule faint" />
        <Label x={20} y={370} tone={2} size={9.5}>
          {st === 0 ? 'post-norm · LayerNorm · ReLU · sinusoidal positions'
            : `pre-norm${st >= 2 ? ' · RMSNorm' : ''}${st >= 3 ? ' · SwiGLU' : ''}${st >= 4 ? ' · RoPE' : ''}   (LLaMA also drops all bias terms)`}
        </Label>
      </svg>
    </Stage>
  );
}
