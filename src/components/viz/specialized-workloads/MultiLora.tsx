import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Chip, Arrow, Fade, Dim, Bar, fmt } from '../lib/svg';
import {
  BASE_8B, BASE_1B, loraBreakdown, loraParams, adapterFleet, batchMix, B,
  type DenseShape,
} from './workloads';

const STEPS = [
  'One adapter is a low-rank correction',
  'The parameter arithmetic: r × (in + out)',
  'How small is small',
  'Adapting everything vs adapting a little',
  'What it costs to serve many tenants',
  'Adapters share the base — so the base is the asset',
  'A batch of mixed tenants',
  'Choosing a rank',
];
const CAPTIONS = [
  'A LoRA adapter is not a new model. It is a pair of thin matrices added to one existing weight matrix, trained while the base stays frozen — so one specialist adapter costs a fraction of a percent of the base it modifies.',
  'For a matrix of shape (out, in) with rank r the adapter holds r×in plus out×r parameters: r×(in+out). The base matrix it corrects holds in×out. The gap between those two numbers is the entire economics of fine-tuning.',
  'Rank is quoted against hidden width. On a 4096-wide model, r=16 is about 0.4% of a projection. That is why an adapter is 100–200 MB when the base is 16 GB.',
  'You can attach adapters to all projections or only some. Attention-only is the cheapest and usually enough for style and format; adding the feed-forward pair roughly doubles it and buys more capacity.',
  'Here is the bill. To serve many tenants from one base, every adapter you can serve at once must be resident — you cannot know which tenant the next request belongs to. Memory grows linearly in tenant count.',
  'The base is loaded once and amortised across every adapter and every request. That is the entire argument for multi-LoRA: one set of weights, many behaviours, and the marginal cost of a tenant is a few megabytes.',
  'A decode batch mixes tenants. The engine gathers the rows belonging to each adapter into its own GEMM. More adapters in flight means smaller GEMMs, and small GEMMs waste bandwidth — this is the real per-tenant cost.',
  'Pick the rank the task needs, not the largest you can afford. Style and format changes saturate at low rank; genuinely new capability needs more, and you will see it in evaluation, not in throughput.',
];

export default function MultiLora() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [modelId, setModelId] = useState<string>('llama8b');
  const [rank, setRank] = useState(16);
  const [nAdapters, setNAdapters] = useState(16);
  const [targets, setTargets] = useState<'attn' | 'all'>('all');
  const [groups, setGroups] = useState(4);

  const m: DenseShape = modelId === 'llama8b' ? BASE_8B : BASE_1B;
  const tList = targets === 'attn' ? (['attn'] as const) : (['attn', 'ffn'] as const);

  const bd = useMemo(() => loraBreakdown(m, rank, [...tList]), [m, rank, targets]);
  const fleet = useMemo(() => adapterFleet(m, rank, nAdapters, 2), [m, rank, nAdapters]);
  const mix = useMemo(() => batchMix(groups), [groups]);

  // One projection's shape, for the arithmetic panel.
  const d = m.hidden;
  const onePair = loraParams(d, d, rank);
  const baseMat = d * d;
  const H = 300;
  const baseH = 90;
  const adH = Math.max(3, (onePair / baseMat) * baseH);

  return (
    <Stage
      rootRef={ref}
      title="Multi-LoRA: one base, many adapters"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'the frozen base model'],
        ['compute', 'the adapters'],
        ['token', 'rows gathered per adapter'],
        ['error', 'over the card’s memory'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Adapter params', value: B(bd.total), kind: 'compute', note: `rank ${rank} · ${targets === 'attn' ? 'attention only' : 'attention + FFN'}` },
          { label: 'Share of base', value: `${(bd.pctOfBase * 100).toFixed(2)}%`, kind: 'token', note: `base is ${B(m.params)}` },
          { label: `${nAdapters} adapters`, value: fmt.bytes(fleet.adaptersBytes), kind: 'compute', note: `base ${fmt.bytes(fleet.baseBytes)}` },
          { label: 'Base share of memory', value: `${(fleet.baseFrac * 100).toFixed(1)}%`, kind: fleet.utilFrac > 1 ? 'error' : 'memory', note: fleet.fitsOnCard > 0 ? `${fleet.fitsOnCard} fleets fit on an A100` : 'does not fit on an A100' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Base model" value={modelId} options={[{ value: 'llama8b', label: 'Llama-3.1-8B' }, { value: 'qwen05', label: 'Qwen2.5-0.5B' }]} onChange={setModelId} />
          <Segmented label="Adapted matrices" value={targets} options={[{ value: 'attn', label: 'Attention only' }, { value: 'all', label: 'Attention + FFN' }]} onChange={setTargets} />
          <Slider label="Rank r" kind="compute" min={1} max={64} step={1} value={rank} onChange={setRank} format={(v) => `r = ${v}`} />
          <Slider label="Adapters served at once" kind="memory" min={1} max={64} step={1} value={nAdapters} onChange={setNAdapters} format={(v) => `${v} tenants`} />
          <Slider label="Adapters mixed in one batch" kind="token" min={1} max={16} step={1} value={groups} onChange={setGroups} format={(v) => `${v} groups`} />
        </>
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="A frozen base model with thin low-rank adapters added to its projections, a comparison of base versus adapter memory, and the total memory needed to serve many tenants on one GPU.">
        {/* ── panel A: the low-rank shape ── */}
        <Label x={12} y={20} tone={2} size={11} bold>Adding a low-rank correction to one projection</Label>

        <Fade show={st >= 0}>
          <g>
            <Chip x={12} y={30} text={`W: ${d}×${d}`} kind="memory" h={22} w={84} size={9.5} />
            <Arrow x1={100} y1={41} x2={128} y2={41} kind="memory" />
            <Label x={132} y={38} tone={3} size={9}>frozen, shared by all tenants</Label>
            <Label x={132} y={52} tone={2} size={9.5}>{B(baseMat)} params</Label>
          </g>
        </Fade>

        <Fade show={st >= 1}>
          <g>
            {/* A: r x in */}
            <rect x={12} y={62} width={adH * 2.2} height={70} rx={2} fill="var(--compute)" stroke="var(--compute)" strokeWidth={1} className="v-cell" />
            <Label x={12} y={146} size={9} kind="compute">A: r×{d}</Label>
            <Arrow x1={12 + adH * 2.2 + 6} y1={97} x2={92} y2={97} kind="compute" />
            {/* x */}
            <Label x={100} y={100} size={9} kind="token">x</Label>
            <Arrow x1={108} y1={97} x2={128} y2={97} kind="compute" />
            {/* B: out x r */}
            <rect x={128} y={62} width={70} height={adH * 2.2} rx={2} fill="var(--compute)" stroke="var(--compute)" strokeWidth={1} className="v-cell" />
            <Label x={128} y={146} size={9} kind="compute">B: {d}×r</Label>
            <Arrow x1={204} y1={97} x2={232} y2={97} kind="compute" />
            <Label x={236} y={100} size={9} kind="token">+ W x → out</Label>
            <Label x={12} y={162} size={9.5} kind="compute">{`adapter = r(in+out) = ${rank}×(${d}+${d}) = ${fmt.int(onePair)} params`}</Label>
          </g>
        </Fade>

        {/* ── panel B: base vs adapter size ── */}
        <Fade show={st >= 2}>
          <g>
            <Label x={320} y={20} tone={2} size={11} bold>How small is small</Label>
            <Label x={320} y={40} tone={3} size={9.5} mono={false}>base weight matrix</Label>
            <Span x={320} y={46} w={220} h={baseH} kind="memory" label={`${B(baseMat)}`} labelTone="dark" />
            <Label x={320} y={158} tone={3} size={9.5} mono={false}>the r×(in+out) adapter</Label>
            <Span x={320} y={164} w={Math.max(4, (onePair / baseMat) * 220)} h={14} kind="compute" />
            <Label x={320 + 220 + 8} y={175} tone={1} size={9.5}>{`${((onePair / baseMat) * 100).toFixed(2)}%`}</Label>
            <Label x={320} y={196} tone={3} size={9.5} mono={false}>{`Across all matrices: ${B(bd.total)} params = ${fmt.bytes(bd.bytes)}.`}</Label>
          </g>
        </Fade>

        {/* ── panel C: per-kind breakdown ── */}
        <Fade show={st >= 3}>
          <g>
            <Label x={12} y={186} tone={2} size={11} bold>What the adapter covers</Label>
            {bd.perKind.map((k, i) => (
              <g key={k.kind}>
                <Label x={12} y={204 + i * 18} tone={2} size={9.5} mono={false}>{k.kind}</Label>
                <Bar x={200} y={196 + i * 18} w={110} h={12} frac={k.params / bd.total} kind="compute" />
                <Label x={316} y={206 + i * 18} tone={1} size={9.5}>{B(k.params)}</Label>
              </g>
            ))}
            <Label x={320} y={218} tone={3} size={9.5}>{`${(bd.pctOfBase * 100).toFixed(2)}% of the base model`}</Label>
          </g>
        </Fade>

        {/* ── panel D: the fleet ── */}
        <Fade show={st >= 4}>
          <g>
            <Label x={12} y={244} tone={2} size={11} bold>{`Serving ${nAdapters} tenants on one A100 (80 GB)`}</Label>
            <Label x={12} y={264} tone={3} size={9.5} mono={false}>base model, once</Label>
            <Bar x={190} y={256} w={300} h={12} frac={fleet.baseBytes / (80e9)} kind="memory" />
            <Label x={496} y={266} tone={2} size={9.5}>{fmt.bytes(fleet.baseBytes)}</Label>
            <Label x={12} y={282} tone={3} size={9.5} mono={false}>{`${nAdapters} adapters`}</Label>
            <Bar x={190} y={274} w={300} h={12} frac={fleet.adaptersBytes / (80e9)} kind="compute" />
            <Label x={496} y={284} tone={2} size={9.5}>{fmt.bytes(fleet.adaptersBytes)}</Label>
            <Label x={12} y={298} size={9.5} kind={fleet.utilFrac > 1 ? 'error' : 'token'}>{`total ${fmt.bytes(fleet.totalBytes)} — ${(fleet.utilFrac * 100).toFixed(1)}% of the card`}</Label>
          </g>
        </Fade>

        {/* ── panel E: mixed batch ── */}
        <Fade show={st >= 6}>
          <g>
            <Label x={12} y={318} tone={2} size={9.5}>{`Batch of 32, ${groups} adapter${groups > 1 ? 's' : ''} in it → ${mix.perGroupTokens.toFixed(1)} rows per GEMM, ${mix.slowdown.toFixed(2)}× the step`}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}