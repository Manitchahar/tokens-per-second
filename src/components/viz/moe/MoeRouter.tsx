import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Chip, Arrow, Dim, Fade, Bar, fmt } from '../lib/svg';
import {
  MOE_MODELS, moeById, totalParams, activeParams, sparsity, denseTwin, moeStep,
  route, summarise, capacityTokens, CONCENTRATION, type ConcentrationId, B,
} from './moe';

const STEPS = [
  'A dense FFN is the thing to beat',
  'Replace it with a bank of experts',
  'The router picks k of them',
  'Top-1: the simplest form',
  'Top-2 buys quality for 2× work',
  'Specialisation is the whole point',
  'The experts are not equal',
  'Sparsity is the number that matters',
  'What it costs at decode time',
];
const CAPTIONS = [
  'Every transformer block ends in a feed-forward network: one wide matrix up, one down, a nonlinearity between. It is usually two thirds of the parameters, and every single token reads all of them.',
  'An MoE deletes that one network and puts a bank of identical experts in its place — same shape, each a third of the width, so hundreds fit where one used to be. Alongside them sits the router, a matrix of shape (experts × hidden) that turns a token into scores.',
  'The router is one matrix-vector product, the cheapest operation in the block. Its scores are softmaxed and the top k are selected. Nothing downstream in the layer knows which experts were chosen.',
  'Switch Transformer, the paper that made the idea mainstream, uses 2048 experts and picks exactly one per token. One expert per token means one expert computation per token: FLOPs stay where a dense block had them.',
  'Mixtral picks two of eight. Per-token expert work doubles, but each token gets two experts and the quality holds at a total parameter count top-1 routing could not reach. This is the standard production shape.',
  'Trained experts specialise. Some take syntax, some take numbers, some take rare tokens. That is why the experts need not be identical — and also why they stop being interchangeable, which is the trap in the next step.',
  'Specialisation gives the router favourites. A few experts receive far more than their share of tokens, so the card that owns one of them does far more work than its neighbours. Load imbalance is the tax you pay for specialisation.',
  'The number to quote is sparsity: total parameters divided by active parameters. Mixtral is about 3.6×, Qwen3-30B-A3B about 9×. That ratio is what makes a 46.7B model decode as fast as a 13B one.',
  'Put it in the decode-step model. Active bytes replace total bytes and the step collapses — while the weight memory barely moves, because every expert still has to be resident.',
];

export default function MoeRouter() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [modelId, setModelId] = useState<string>('mixtral');
  const [concId, setConcId] = useState<ConcentrationId>('sharp');
  const [tokens, setTokens] = useState(64);

  const m = moeById(modelId);
  const conc = CONCENTRATION.find((c) => c.id === concId)!.value;

  const routing = useMemo(
    () => summarise(route(tokens, m.experts, m.topK, conc, 11), tokens, m.experts, m.topK, 1.25),
    [tokens, m, conc],
  );

  const total = totalParams(m);
  const active = activeParams(m);
  const sp = sparsity(m);
  const twin = denseTwin(m);
  const step = moeStep(m, { gpu: 'h100', batch: 32, ctx: 4096 });
  const cap = capacityTokens(tokens, m.experts, m.topK, 1.25);

  const ranked = routing.counts.map((c, i) => ({ c, i })).sort((a, b) => b.c - a.c);
  const hot = ranked.slice(0, 2).map((r) => r.i);
  const cold = ranked[ranked.length - 1].i;
  const maxCount = Math.max(1, ...routing.counts);
  const showGrid = m.experts <= 16;

  const COLS = Math.min(8, m.experts);
  const ROWS = Math.ceil(m.experts / COLS);
  const CELL = 26, GAP = 4;
  const GW = COLS * (CELL + GAP) - GAP;
  const GH = ROWS * (CELL + GAP) - GAP;
  const GX = 12, GY = 108;

  return (
    <Stage
      rootRef={ref}
      title="Mixture-of-Experts: which experts does this token use?"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'experts this token uses'],
        ['memory', 'experts skipped'],
        ['token', 'the token'],
        ['error', 'hot / overloaded expert'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Total parameters', value: B(total), kind: 'text', note: 'all of it must be resident' },
          { label: 'Active per token', value: B(active), kind: 'compute', note: `${m.name}: ${m.topK} of ${m.experts} experts` },
          { label: 'Sparsity', value: `${sp.toFixed(1)}×`, kind: 'token', note: 'total ÷ active' },
          { label: 'Hot expert', value: `${routing.maxFrac.toFixed(2)}× ideal`, kind: routing.imbalance > 2.2 ? 'error' : 'token', note: `balanced would be ${(m.topK / m.experts).toFixed(2)}×` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={MOE_MODELS.map((x) => ({ value: x.id, label: x.name.split(' ')[0] }))} onChange={setModelId} />
          <Segmented label="Router sharpness" value={concId} options={CONCENTRATION.map((c) => ({ value: c.id, label: c.label }))} onChange={setConcId} />
          <Slider label="Tokens in the batch" kind="token" min={8} max={256} step={8} value={tokens} onChange={setTokens} format={(v) => `${v} tokens`} />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A dense feed-forward network shown next to a bank of mixture-of-experts with a router selecting the top k experts per token. Below, the load each of the experts received from a batch of tokens, shaded from cold to hot, and the ratio of active to total parameters.">
        {/* ── panel A: the block's feed-forward path ── */}
        <Fade show={st <= 1}>
          <g>
            <Label x={12} y={20} tone={2} size={11} bold>A transformer block ends in a feed-forward network</Label>
            <Chip x={12} y={34} text="residual" kind="text" h={26} w={60} size={10} />
            <Arrow x1={76} y1={47} x2={106} y2={47} kind="text" />
            <Chip x={110} y={34} text={`up: d → ${m.denseIntermediate}`} kind="compute" h={26} w={104} size={10} />
            <Arrow x1={218} y1={47} x2={248} y2={47} kind="text" />
            <Chip x={252} y={34} text="act" kind="text" h={26} w={40} size={10} />
            <Arrow x1={296} y1={47} x2={326} y2={47} kind="text" />
            <Chip x={330} y={34} text={`down: ${m.denseIntermediate} → d`} kind="compute" h={26} w={110} size={10} />
            <Label x={452} y={44} tone={2} size={9.5} mono={false}>One network.</Label>
            <Label x={452} y={58} tone={3} size={9} mono={false}>Every token reads all of it.</Label>
          </g>
        </Fade>

        <Fade show={st >= 1}>
          <g>
            <Label x={12} y={20} tone={2} size={11} bold>Delete the one network; put a bank of them in its place</Label>
            <Chip x={12} y={34} text="token x" kind="token" h={26} w={54} size={10} />
            <Arrow x1={70} y1={47} x2={100} y2={47} kind="compute" />
            <Chip x={104} y={34} text={`router ${m.experts}×${m.hidden}`} kind="compute" h={26} w={108} size={10} />
            <Arrow x1={216} y1={47} x2={246} y2={47} kind="token" />
            <Chip x={250} y={34} text={`top ${m.topK} of ${m.experts}`} kind="token" h={26} w={86} size={10} />
            <Arrow x1={340} y1={47} x2={372} y2={47} kind="compute" />
            <Chip x={376} y={34} text={`${m.experts} experts × ${m.expertIntermediate}`} kind="compute" h={26} w={160} size={10} />
            <Label x={12} y={78} tone={3} size={9.5}>The router is one matrix-vector product — the cheapest operation in the block.</Label>
          </g>
        </Fade>

        {/* ── panel A lower: per-step explanation ── */}
        <Fade show={st === 2}>
          <g>
            <Label x={12} y={96} tone={2} size={11} bold>Router scores, and which ones get selected</Label>
            {routing.counts.map((_, i) => {
              const bw = 470 / m.experts;
              const x = 12 + i * bw;
              const on = i < m.topK;
              const hgt = 16 + ((i * 29 + 7) % m.experts) / m.experts * 40;
              return (
                <g key={i}>
                  <rect x={x + 1.5} y={116 - hgt} width={Math.max(3, bw - 3)} height={hgt} rx={2}
                    fill={on ? 'var(--compute)' : 'var(--memory-dim)'} stroke={on ? 'var(--compute)' : 'var(--memory)'} strokeWidth={1} className="v-cell" />
                  <Label x={x + bw / 2} y={130} anchor="middle" tone={on ? 2 : 3} size={8}>{on ? `E${i}` : i}</Label>
                </g>
              );
            })}
            <line x1={12} x2={482} y1={116} y2={116} className="v-rule" />
            <Label x={12} y={150} tone={3} size={9.5}>Amber = in the top {m.topK}. Everything else is skipped entirely for this token.</Label>
            <Label x={12} y={166} size={9.5} kind="token">Cost of routing: {fmt.int(m.experts * m.hidden)} parameters read, once.</Label>
          </g>
        </Fade>

        <Fade show={st === 3 || st === 4}>
          <g>
            <Label x={12} y={96} tone={2} size={11} bold>{st === 3 ? 'Top-1 routing (Switch Transformer)' : `Top-${m.topK} routing (Mixtral 8x7B)`}</Label>
            <Label x={12} y={120} tone={2} size={9.5} mono={false}>{st === 3
              ? '2048 experts, one selected per token.'
              : 'Eight experts, two selected per token.'}</Label>
            <Label x={12} y={140} size={9.5} kind="compute" mono={false}>Expert FLOPs per token</Label>
            <Span x={180} y={130} w={st === 3 ? 90 : 180} h={16} kind="compute" label={`${st === 3 ? 1 : m.topK}× dense`} labelTone="dark" />
            <Label x={12} y={166} size={9.5} kind="memory" mono={false}>Parameters available</Label>
            <Span x={180} y={156} w={250} h={16} kind="memory" label={`${st === 3 ? 2048 : m.experts}× dense`} labelTone="dark" />
            <Label x={12} y={192} tone={3} size={9.5}>{st === 3
              ? 'More parameters, identical per-token work. Quality rises; memory and FLOPs do not.'
              : 'Still far more parameters than a dense block, at twice the expert work. This is the production default.'}</Label>
          </g>
        </Fade>

        <Fade show={st === 5}>
          <g>
            <Label x={12} y={96} tone={2} size={11} bold>Trained experts specialise — and that is why it works</Label>
            {[
              { t: 'syntax & grammar', e: 0 },
              { t: 'numbers', e: 2 },
              { t: 'rare tokens', e: 4 },
              { t: 'quoted text', e: 6 },
            ].map((x) => (
              <g key={x.t}>
                <Chip x={12} y={110 + x.e * 24} text="token" kind="token" h={18} w={44} size={9} />
                <Arrow x1={60} y1={119 + x.e * 24} x2={92} y2={119 + x.e * 24} kind="compute" curve={x.e % 2 === 0 ? -8 : 8} />
                <Chip x={96} y={110 + x.e * 24} text={`expert ${x.e}`} kind="compute" h={18} w={62} size={9} />
                <Label x={168} y={123 + x.e * 24} tone={2} size={10} mono={false}>{x.t}</Label>
              </g>
            ))}
            <Label x={320} y={126} tone={3} size={9.5} mono={false}>Different tokens, different experts.</Label>
            <Label x={320} y={142} tone={3} size={9.5} mono={false}>No single expert has to model</Label>
            <Label x={320} y={158} tone={3} size={9.5} mono={false}>the whole distribution.</Label>
            <Label x={12} y={216} tone={2} size={9.5}>This is the whole mechanism — and the source of the next problem.</Label>
          </g>
        </Fade>

        {/* ── panel B: where the tokens went ── */}
        <Fade show={st >= 6}>
          <g>
            <Label x={GX} y={96} tone={2} size={11} bold>{`Where ${tokens} tokens went`}</Label>
            {showGrid ? (
              <>
                {routing.counts.map((c, i) => {
                  const cx = GX + (i % COLS) * (CELL + GAP);
                  const cy = GY + Math.floor(i / COLS) * (CELL + GAP);
                  const isHot = hot.includes(i);
                  const over = c > cap;
                  return (
                    <g key={i}>
                      <rect x={cx} y={cy} width={CELL} height={CELL} rx={2}
                        fill={isHot ? 'var(--error)' : i === cold ? 'var(--memory-dim)' : 'var(--memory)'}
                        stroke={over ? 'var(--error)' : 'var(--memory)'} strokeWidth={over ? 1.8 : 1}
                        strokeDasharray={i === cold ? '3 3' : undefined}
                        opacity={isHot ? 0.95 : 0.72} className="v-cell" />
                      <text x={cx + CELL / 2} y={cy + CELL / 2 + 3.6} textAnchor="middle" className="v-label"
                        style={{ fontSize: 10, fontWeight: 600, fill: isHot ? '#0e1116' : 'var(--text)' }}>{c}</text>
                    </g>
                  );
                })}
                <Label x={GX + GW + 10} y={GY + 10} size={8.5} kind="error">hot</Label>
                <Label x={GX + GW + 10} y={GY + 22} size={8.5} kind="memory">warm</Label>
                <Label x={GX + GW + 10} y={GY + 34} size={8.5} kind="memory">cold</Label>
                <Label x={GX + GW + 10} y={GY + 50} size={8.5} tone={3}>capacity</Label>
                <Label x={GX + GW + 10} y={GY + 62} size={8.5} tone={1}>{cap} tokens</Label>
                <Dim x1={GX} x2={GX + GW} y={GY + GH + 14} label={`${routing.imbalance.toFixed(1)}× imbalance`} kind={routing.imbalance > 2.2 ? 'error' : 'text'} />
              </>
            ) : (
              <>
                {routing.counts.map((c, i) => {
                  const bw = 470 / m.experts;
                  const x = GX + i * bw;
                  const hgt = Math.max(2, (c / maxCount) * 84);
                  const isHot = hot.includes(i);
                  return (
                    <rect key={i} x={x + 0.5} y={GY + 96 - hgt} width={Math.max(1, bw - 1)} height={hgt}
                      fill={isHot ? 'var(--error)' : 'var(--memory)'} opacity={isHot ? 0.95 : 0.6} className="v-cell" />
                  );
                })}
                <line x1={GX} x2={GX + 470} y1={GY + 96} y2={GY + 96} className="v-rule" />
                <Label x={GX} y={GY + 112} tone={3} size={9}>{`${m.experts} experts · tallest = ${maxCount} tokens · red are the router's favourites`}</Label>
                <Dim x1={GX} x2={GX + 470} y={GY + 124} label={`${routing.imbalance.toFixed(1)}× imbalance`} kind={routing.imbalance > 2.2 ? 'error' : 'text'} />
              </>
            )}
            <Label x={12} y={252} tone={3} size={9.5}>Balance would put {((m.topK / m.experts) * 100).toFixed(0)}% of the batch on every expert. The router does not do that.</Label>
          </g>
        </Fade>

        {/* ── panel C: sparsity ── */}
        <Fade show={st >= 7}>
          <g>
            <Label x={12} y={278} tone={2} size={11} bold>Sparsity: total ÷ active</Label>
            <Bar x={12} y={288} w={260} h={16} frac={active / total} kind="compute" />
            <Label x={280} y={301} tone={1} size={10.5}>{`${B(active)} of ${B(total)}`}</Label>
            <Label x={12} y={324} tone={3} size={9.5}>{`${sp.toFixed(1)}× — one token reads ${B(active)} parameters, not ${B(total)}.`}</Label>
            <Label x={12} y={340} tone={3} size={9.5}>{`A dense model of the same depth would be ${B(twin.params)}. Slower, bigger, no better.`}</Label>
          </g>
        </Fade>

        {/* ── panel D: what it costs per decode step ── */}
        <Fade show={st >= 8}>
          <g>
            <Label x={392} y={278} tone={2} size={11} bold>Decode step</Label>
            <Label x={392} y={298} tone={2} size={9.5} mono={false}>batch 32 · 4K ctx · H100</Label>
            <Label x={392} y={318} size={10} kind="token" bold>{`≈ ${step.stepMs.toFixed(1)} ms`}</Label>
            <Label x={392} y={334} tone={3} size={9}>{`${fmt.bytes(step.activeBytes)} read`}</Label>
            <Label x={392} y={348} tone={3} size={9}>{`${(100 * step.skipFrac).toFixed(0)}% of weights skipped`}</Label>
            <Label x={392} y={364} tone={2} size={9} kind="error">{`weights still span ${step.fitsOn} GPUs`}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}