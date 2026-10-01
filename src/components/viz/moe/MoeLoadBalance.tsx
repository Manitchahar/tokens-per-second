import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Chip, Dim, Fade, Bar, Track, fmt } from '../lib/svg';
import {
  MOE_MODELS, moeById, route, summarise, capacityTokens,
  CONCENTRATION, type ConcentrationId, totalParams, activeParams, allToAll, NETS,
} from './moe';

const STEPS = [
  'Pick a model and a batch',
  'Route every token',
  'Read the load, not the mean',
  'Turn it into a Gini coefficient',
  'Add the capacity constraint',
  'Then pay for it on the wire',
];
const CAPTIONS = [
  'The simulator: choose a published MoE, choose a batch size, and route it. Every number below is the arithmetic, computed as you drag — nothing here is a stored benchmark.',
  'Each token picks its top-k experts. The router has preferences, and the preference curve is the "sharpness" control — flat means no specialisation, sharp means a few experts dominate.',
  'The bars are each expert’s share of the batch. The dashed line is the perfectly balanced share, k/E. Read the tallest bar against that line, not against the average.',
  'The Gini coefficient summarises the whole picture as one number in 0 to 1: 0 is perfect equality, 1 is one expert taking everything. It is the number to report, because a mean load tells you nothing.',
  'Real kernels cannot have ragged buffers. Give each expert a capacity of c × (k·T/E) slots. Everything beyond is dropped or overflowed, and the overflow rate is the quality cost of a rectangular kernel.',
  'Finally, connect it to the previous lesson: imbalance multiplies the fraction of activations that cross a card, which is all-to-all traffic in every MoE layer, forever.',
];

const GINI_STEPS = [
  'Every load, sorted',
  'The area between the curve and the diagonal',
];

export default function MoeLoadBalance() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [modelId, setModelId] = useState<string>('mixtral');
  const [concId, setConcId] = useState<ConcentrationId>('sharp');
  const [tokens, setTokens] = useState(256);
  const [capF, setCapF] = useState(1.25);

  const m = moeById(modelId);
  const conc = CONCENTRATION.find((c) => c.id === concId)!.value;

  const r = useMemo(
    () => summarise(route(tokens, m.experts, m.topK, conc, 7), tokens, m.experts, m.topK, capF),
    [tokens, m, conc, capF],
  );
  const cap = capacityTokens(tokens, m.experts, m.topK, capF);
  const ideal = m.topK / m.experts;

  // Gini of the expert load distribution.
  const gini = useMemo(() => {
    const a = [...r.counts].sort((x, y) => x - y);
    const n = a.length;
    const tot = a.reduce((p, c) => p + c, 0);
    if (tot === 0) return 0;
    let cum = 0;
    for (let i = 0; i < n; i++) cum += (i + 1) * a[i];
    return (2 * cum) / (n * tot) - (n + 1) / n;
  }, [r.counts]);

  const comm = useMemo(() => allToAll(m, NETS[2], tokens, r.imbalance), [m, tokens, r.imbalance]);
  const ranked = r.counts.map((c, i) => ({ c, i })).sort((a, b) => b.c - a.c);
  const top3 = ranked.slice(0, 3);
  const top3Share = top3.reduce((p, x) => p + x.c, 0) / (tokens * m.topK);

  const maxC = Math.max(1, ...r.counts);
  const nBars = Math.min(64, m.experts);
  const bw = 560 / nBars;

  return (
    <Stage
      rootRef={ref}
      title="MoE load balance simulator"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'expert load'],
        ['error', 'over capacity'],
        ['compute', 'balanced share'],
        ['comm', 'all-to-all volume'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Imbalance', value: `${r.imbalance.toFixed(2)}×`, kind: r.imbalance > 1.6 ? 'error' : 'token', note: `hottest ÷ balanced share (${ideal.toFixed(2)}×)` },
          { label: 'Gini', value: gini.toFixed(3), kind: gini > 0.35 ? 'error' : 'token', note: '0 = perfect equality, 1 = one expert takes all' },
          { label: 'Overflow', value: `${r.dropped} tokens`, kind: r.dropped > 0 ? 'error' : 'token', note: `capacity ${capF.toFixed(2)}× → ${cap} slots per expert` },
          { label: 'Top-3 experts hold', value: `${(top3Share * 100).toFixed(0)}%`, kind: 'memory', note: 'of all expert assignments' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={MOE_MODELS.map((x) => ({ value: x.id, label: x.name.split(' ')[0] }))} onChange={setModelId} />
          <Segmented label="Router sharpness" value={concId} options={CONCENTRATION.map((c) => ({ value: c.id, label: c.label }))} onChange={setConcId} />
          <Slider label="Batch size (tokens)" kind="token" min={32} max={1024} step={32} value={tokens} onChange={setTokens} format={(v) => `${v} tokens`} />
          <Slider label="Capacity factor" kind={capF < 1.2 ? 'error' : 'text'} min={1} max={3} step={0.05} value={capF} onChange={setCapF} format={(v) => `${v.toFixed(2)}×`} />
        </>
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A bar chart of how many tokens each expert in a mixture-of-experts model received from a simulated batch, with a dashed line showing the perfectly balanced share, plus a Lorenz curve summarising the inequality as a Gini coefficient and a capacity bar showing overflow.">
        {/* ── 1: the load bars ── */}
        <Fade show={st >= 1}>
          <g>
            <Label x={12} y={20} tone={2} size={11} bold>{`${m.name} · ${tokens} tokens · top-${m.topK} of ${m.experts} experts`}</Label>
            <Label x={628} y={20} anchor="end" tone={3} size={9}>{conc.toFixed(2)} concentration</Label>
            {r.counts.slice(0, nBars).map((c, i) => {
              const h = (c / maxC) * 108;
              const over = c > cap;
              const isHot = ranked.slice(0, 2).some((x) => x.i === i);
              return (
                <g key={i}>
                  <rect x={12 + i * bw} y={140 - h} width={Math.max(1.5, bw - 2)} height={Math.max(1, h)} rx={1.5}
                    fill={over ? 'var(--error)' : isHot ? 'var(--compute)' : 'var(--memory)'}
                    fillOpacity={over ? 0.95 : isHot ? 0.9 : 0.65} className="v-cell" />
                </g>
              );
            })}
            <line x1={12} x2={572} y1={140} y2={140} className="v-rule" />
            <line x1={12} x2={572} y1={140 - (ideal * tokens / maxC) * 108} y2={140 - (ideal * tokens / maxC) * 108}
              stroke="var(--compute)" strokeWidth={1.4} strokeDasharray="5 4" />
            <Label x={572} y={140 - (ideal * tokens / maxC) * 108 - 6} anchor="end" size={9} kind="compute">
              balanced share {ideal.toFixed(2)}×
            </Label>
            <Label x={12} y={156} tone={3} size={9}>Tokens received per expert. Red = over capacity.</Label>
          </g>
        </Fade>

        {/* ── 2: the read-out ── */}
        <Fade show={st >= 2}>
          <g>
            <Label x={12} y={182} tone={2} size={11} bold>Do not read the mean</Label>
            <Label x={12} y={202} tone={3} size={9.5} mono={false}>Mean load is {(tokens * m.topK / m.experts).toFixed(0)} tokens. Hottest expert holds {r.counts[r.hottest]} — {r.imbalance.toFixed(1)}× the mean.</Label>
            <Label x={12} y={218} tone={3} size={9.5} mono={false}>Coldest expert holds {r.counts[r.coldest]}. In a deployment this card idles while its neighbour stalls.</Label>
            {top3.map((x, i) => (
              <Chip key={x.i} x={12 + i * 110} y={230} text={`expert ${x.i}: ${x.c} tokens`} kind="compute" h={20} w={102} size={9} />
            ))}
            <Label x={12} y={274} tone={2} size={10}>Top-3 experts hold {(top3Share * 100).toFixed(0)}% of all assignments.</Label>
          </g>
        </Fade>

        {/* ── 3: the Lorenz curve and Gini ── */}
        <Fade show={st >= 3}>
          <g>
            <Label x={12} y={182} tone={2} size={11} bold>Gini coefficient: one number for all of it</Label>
            {(() => {
              const sorted = [...r.counts].sort((a, b) => a - b);
              const tot = sorted.reduce((p, c) => p + c, 0) || 1;
              const pts = sorted.map((c, i) => {
                const x = 320 + ((i + 1) / sorted.length) * 260;
                const y = 330 - (sorted.slice(0, i + 1).reduce((p, v) => p + v, 0) / tot) * 110;
                return `${x},${y}`;
              }).join(' ');
              return (
                <g>
                  <line x1={320} x2={580} y1={330} y2={330} className="v-rule" />
                  <line x1={320} x2={320} y1={330} y2={220} className="v-rule" />
                  <line x1={320} x2={580} y1={330} y2={220} stroke="var(--rule-2)" strokeWidth={1} strokeDasharray="3 3" />
                  <polyline points={`320,330 ${pts}`} fill="none" stroke="var(--memory)" strokeWidth={2} />
                  <Label x={320} y={344} size={8.5} tone={3}>0% of experts</Label>
                  <Label x={580} y={344} anchor="end" size={8.5} tone={3}>100%</Label>
                  <Label x={322} y={216} size={8.5} tone={3}>100% of tokens</Label>
                  <Label x={450} y={200} anchor="middle" size={11} kind="compute" bold>{`G = ${gini.toFixed(3)}`}</Label>
                </g>
              );
            })()}
            <Label x={12} y={200} tone={3} size={9.5} mono={false}>The curve bows below the diagonal</Label>
            <Label x={12} y={214} tone={3} size={9.5} mono={false}>by the area between them. That area,</Label>
            <Label x={12} y={228} tone={3} size={9.5} mono={false}>normalised, is the Gini coefficient.</Label>
            <Label x={12} y={250} size={9.5} kind={gini > 0.35 ? 'error' : 'token'} mono={false} bold>
              {gini > 0.5 ? 'Severely imbalanced — expect stragglers.' : gini > 0.3 ? 'Imbalanced — capacity factor must exceed 1.3.' : gini > 0.15 ? 'Moderate — the default 1.25 capacity holds.' : 'Near-uniform — little straggler risk.'}
            </Label>
            <Label x={12} y={272} tone={3} size={9}>Below 0.10 is suspiciously uniform: real routers</Label>
            <Label x={12} y={286} tone={3} size={9}>specialise, and specialists are uneven.</Label>
          </g>
        </Fade>

        {/* ── 4: capacity ── */}
        <Fade show={st >= 4}>
          <g>
            <Label x={12} y={182} tone={2} size={11} bold>Capacity: the rectangular-kernel constraint</Label>
            <Label x={12} y={202} tone={3} size={9.5} mono={false}>Each expert gets {cap} slots. Red bars exceed it.</Label>
            <Track x={12} y={214} w={560} h={26} label="">
              <rect x={12} y={214} width={560} height={26} rx={3} fill="none" stroke="var(--rule)" />
              <rect x={12} y={214} width={Math.max(2, Math.min(1, cap / Math.max(maxC, cap)) * 560)} height={26} rx={3}
                fill="var(--memory)" fillOpacity={0.35} className="v-cell" />
              <line x1={12 + (Math.min(1, cap / Math.max(maxC, cap)) * 560)} x2={12 + (Math.min(1, cap / Math.max(maxC, cap)) * 560)} y1={210} y2={244}
                stroke="var(--compute)" strokeWidth={1.6} />
            </Track>
            <Label x={12 + (Math.min(1, cap / Math.max(maxC, cap)) * 560)} y={208} anchor="middle" size={9} kind="compute">capacity {cap}</Label>
            <Label x={12} y={262} size={10} kind={r.dropped > 0 ? 'error' : 'token'} mono={false} bold>
              {r.dropped > 0 ? `${r.dropped} tokens overflow — ${((r.dropped / (tokens * m.topK)) * 100).toFixed(1)}% of assignments` : 'No overflow at this capacity factor'}
            </Label>
            <Label x={12} y={280} tone={3} size={9.5} mono={false}>Raising the capacity factor to {Math.max(capF, r.imbalance).toFixed(2)} would stop every drop,</Label>
            <Label x={12} y={294} tone={3} size={9.5} mono={false}>at the cost of provisioning buffers that sit empty most of the time.</Label>
            <Label x={12} y={318} tone={2} size={9.5}>C = ceil(capacity × tokens × k / experts)</Label>
            <Label x={12} y={336} tone={3} size={9}>The default 1.25 buys headroom for a moderate router.</Label>
            <Label x={12} y={352} tone={3} size={9}>Below 1.0 you drop tokens on purpose, and quality pays.</Label>
          </g>
        </Fade>

        {/* ── 5: the wire cost ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={12} y={182} tone={2} size={11} bold>What imbalance costs on the wire</Label>
            <Label x={12} y={202} tone={3} size={9.5} mono={false}>8 GPUs, NVLink, this batch. All-to-all volume, all MoE blocks, both directions:</Label>
            <Bar x={12} y={214} w={300} h={16} frac={Math.min(1, comm.totalBytes / 4e11)} kind="comm" />
            <Label x={320} y={227} size={10} kind="comm" bold>{fmt.bytes(comm.totalBytes)}</Label>
            <Label x={12} y={252} size={9.5} mono={false}>Fraction of activations crossing a card</Label>
            <Label x={12} y={270} size={10.5} kind="comm" bold>{`${(comm.fractionRemote * 100).toFixed(0)}%`}</Label>
            <Label x={12} y={288} tone={3} size={9}>(balanced would be {((1 - 1 / 8) * 100).toFixed(0)}%)</Label>
            <Label x={12} y={312} tone={2} size={9.5} mono={false}>The {m.name} takes {fmt.bytes(comm.totalBytes)} across the fabric per step at this routing.</Label>
            <Label x={12} y={330} tone={2} size={9.5} mono={false}>Below the all-to-all, {r.imbalance.toFixed(1)}× of the compute is on one card while {((1 - r.maxFrac / ideal) * 100).toFixed(0)}% of expert capacity idles.</Label>
            <Label x={12} y={350} size={9.5} kind="error" mono={false}>In a synchronous step, that card sets everyone's latency.</Label>
          </g>
        </Fade>

        {/* ── 0: the setup ── */}
        <Fade show={st === 0}>
          <g>
            <Label x={12} y={20} tone={2} size={11} bold>What this simulator does</Label>
            <Label x={12} y={44} tone={2} size={9.5} mono={false}>1. Take a published MoE configuration.</Label>
            <Label x={12} y={62} tone={2} size={9.5} mono={false}>2. Route a batch of tokens through it, with a</Label>
            <Label x={12} y={78} tone={2} size={9.5} mono={false}>{`   controllable preference curve (top-${m.topK} of ${m.experts}).`}</Label>
            <Label x={12} y={100} tone={2} size={9.5} mono={false}>3. Report the load distribution, the imbalance,</Label>
            <Label x={12} y={116} tone={2} size={9.5} mono={false}>   the Gini coefficient, and the overflow rate.</Label>
            <Label x={12} y={138} tone={2} size={9.5} mono={false}>4. Convert imbalance into all-to-all bytes.</Label>
            <Dim x1={12} x2={300} y={170} label={`${m.name}: ${totalParams(m) / 1e9 >= 10 ? (totalParams(m) / 1e9).toFixed(0) : (totalParams(m) / 1e9).toFixed(1)}B total, ${(activeParams(m) / 1e9).toFixed(1)}B active`} />
            <Label x={12} y={196} tone={3} size={9}>Drag the sliders. Every number updates live.</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}