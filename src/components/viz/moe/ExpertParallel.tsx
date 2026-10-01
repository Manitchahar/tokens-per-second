import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Dim, Fade, Ruler, fmt } from '../lib/svg';
import {
  MOE_MODELS, moeById, totalParams, moeLayerCount, allToAll, NETS, gpu,
  type NetConfig,
} from './moe';

const STEPS = [
  'The memory wall arrives first',
  'Shard experts across GPUs',
  'Then you need an all-to-all',
  'What the all-to-all actually moves',
  'Imbalance makes the wire busier',
  'Capacity: the hard bound',
  'Hybrid parallelism: EP × TP',
  'Pick the layout, then pay',
];
const CAPTIONS = [
  'Mixtral in BF16 is 90.6 GB of weights and an H100 has 80. The model does not fit. This is not an unusual problem — it is the normal problem, and it is why you learned pipeline and tensor parallelism in Module 18.',
  'Expert parallelism shards the expert bank the same way pipeline parallelism shards layers: give each GPU a slice of the experts. Every parameter is now resident somewhere, and no GPU holds the whole model.',
  'But the tokens are not where the experts are. A token whose expert lives on GPU 3 is sitting on GPU 0. Something has to move it there — and that something is an all-to-all, a collective where every rank sends a different slice to every other rank.',
  'The payload per token is small: one hidden-sized activation vector in, one back. Multiply by the tokens, by the fraction that must leave, and by the number of MoE blocks, and you have the comm cost.',
  'Balance is the assumption hidden inside that formula. When one expert is hot, the tokens aimed at it pile onto whichever GPU owns it, so more of the batch has to travel and the busiest link gets busier.',
  'The standard defence is a capacity factor: give each expert a fixed number of slots, and drop or overflow what does not fit. It converts a ragged problem into a rectangular one, at the cost of dropping tokens.',
  'Expert parallelism alone does not shard a single expert, so a wide expert still has to fit on one GPU. In practice you multiply it by tensor parallelism — EP on the outer, TP inside — and both must divide the number of GPUs.',
  'The layout is a decision, not a default. Pick the expert count, the split, and the interconnect together, then measure whether the wire is your bottleneck or your GEMMs are.',
];

export default function ExpertParallel() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [modelId, setModelId] = useState<string>('mixtral');
  const [netIdx, setNetIdx] = useState(2);
  const [tokens, setTokens] = useState(256);
  const [imbalance, setImbalance] = useState(1);
  const [wBytes, setWBytes] = useState(2);

  const m = moeById(modelId);
  const net: NetConfig = NETS[netIdx];
  const g = gpu('h100');

  const total = totalParams(m);
  const weightBytes = total * wBytes;

  const comm = useMemo(
    () => allToAll(m, net, tokens, imbalance),
    [m, net, tokens, imbalance],
  );

  const perGpuBytes = weightBytes / net.gpus;
  const fitsOne = weightBytes <= g.mem;
  const expertsPerGpu = m.experts / net.gpus;
  const tpNeeded = Math.max(1, Math.ceil(m.expertIntermediate / 8192));
  const totalGpus = net.gpus * tpNeeded;

  const expertsPerGpuFrac = Math.min(1, expertsPerGpu);
  const nShown = Math.min(8, Math.ceil(m.experts / net.gpus));

  return (
    <Stage
      rootRef={ref}
      title="Expert parallelism: where the experts live, and what it costs to reach them"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'weights resident on a GPU'],
        ['compute', 'expert doing work'],
        ['comm', 'all-to-all traffic'],
        ['error', 'over capacity / dropped'],
        ['idle', 'unused memory'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Weights total', value: fmt.bytes(weightBytes), kind: 'memory', note: `${m.name} at ${wBytes === 2 ? 'BF16' : wBytes === 1 ? 'FP8' : '4-bit'}` },
          { label: 'Per GPU', value: fmt.bytes(perGpuBytes), kind: fitsOne ? 'memory' : 'error', note: `${net.gpus}× H100` },
          { label: 'All-to-all', value: fmt.bytes(comm.totalBytes), kind: 'comm', note: `both directions, all ${moeLayerCount(m)} MoE blocks` },
          { label: 'Dispatch time', value: `${comm.dispatchMs.toFixed(2)} ms`, kind: 'comm', note: `per MoE block · ${net.short}` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={MOE_MODELS.map((x) => ({ value: x.id, label: x.name.split(' ')[0] }))} onChange={setModelId} />
          <Segmented label="Interconnect" value={netIdx} options={NETS.map((n, i) => ({ value: i, label: `${n.gpus}× ${n.short}` }))} onChange={setNetIdx} />
          <Segmented label="Precision" value={wBytes} options={[{ value: 2, label: 'BF16' }, { value: 1, label: 'FP8' }, { value: 0.5, label: '4-bit' }]} onChange={setWBytes} />
          <Slider label="Tokens per step" kind="token" min={32} max={2048} step={32} value={tokens} onChange={setTokens} format={(v) => `${fmt.int(v)} tokens`} />
          <Slider label="Routing imbalance" kind={imbalance > 1.8 ? 'error' : 'text'} min={1} max={3} step={0.05} value={imbalance} onChange={setImbalance} format={(v) => `${v.toFixed(2)}× ideal`} />
        </>
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A mixture-of-experts model sharded across a group of GPUs, with the expert each rank owns, the tokens that must travel to reach a remote expert over an all-to-all collective, and the resulting communication volume per layer.">
        {/* ── 0: the memory wall ── */}
        <Fade show={st === 0}>
          <g>
            <Label x={12} y={22} tone={2} size={11} bold>First problem: it does not fit</Label>
            <Label x={12} y={48} size={10} mono={false} kind="memory">{`${m.name} weights at ${wBytes === 2 ? 'BF16' : wBytes === 1 ? 'FP8' : '4-bit'}: ${fmt.bytes(weightBytes)}`}</Label>
            <Label x={12} y={68} size={10} mono={false} kind="error">{`One H100 holds ${fmt.bytes(g.mem)}. ${fitsOne ? 'It fits.' : `${(weightBytes / g.mem).toFixed(2)}× too big.`}`}</Label>
            <rect x={12} y={84} width={560} height={30} rx={3} fill="none" stroke="var(--rule)" />
            <rect x={12} y={84} width={Math.min(560, (g.mem / weightBytes) * 560)} height={30} rx={3} fill="var(--memory)" fillOpacity={0.8} stroke="var(--memory)" />
            <Label x={578} y={104} anchor="end" size={9} tone={1}>{fitsOne ? 'fits' : `${(weightBytes / g.mem).toFixed(2)}× over`}</Label>
            <Label x={12} y={134} tone={3} size={9.5}>The blue bar is one H100's 80 GB. The model is the whole outline.</Label>
          </g>
        </Fade>

        {/* ── 1+: the sharded layout ── */}
        <Fade show={st >= 1}>
          <g>
            <Label x={12} y={22} tone={2} size={11} bold>{`${net.gpus} GPUs, experts sharded across them`}</Label>
            {Array.from({ length: net.gpus }, (_, r) => {
              const gw = Math.min(130, 560 / net.gpus);
              const x = 12 + r * gw;
              const shown = Math.min(nShown, Math.ceil(m.experts / net.gpus));
              const localShare = expertsPerGpuFrac;
              return (
                <g key={r}>
                  <rect x={x} y={36} width={gw - 8} height={128} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
                  <Label x={x + (gw - 8) / 2} y={50} anchor="middle" size={9.5} tone={1}>GPU {r}</Label>
                  <Label x={x + (gw - 8) / 2} y={63} anchor="middle" tone={3} size={8}>{fmt.bytes(perGpuBytes)}</Label>
                  {Array.from({ length: shown }, (_, e) => {
                    const cw = (gw - 8 - 8) / shown;
                    // the first expert on each card is hot when imbalance > 1
                    const hot = imbalance > 1.2 && e === 0;
                    const busy = hot ? Math.min(1, localShare * imbalance) : localShare;
                    return (
                      <g key={e}>
                        <rect x={x + 4 + e * cw} y={70} width={Math.max(2, cw - 2)} height={18} rx={1.5}
                          fill={hot ? 'var(--error)' : 'var(--memory)'} fillOpacity={hot ? 0.95 : 0.55} className="v-cell" />
                        <rect x={x + 4 + e * cw} y={92} width={Math.max(2, (cw - 2) * busy)} height={10} rx={1.5}
                          fill={hot ? 'var(--error)' : 'var(--compute)'} fillOpacity={0.85} className="v-cell" />
                      </g>
                    );
                  })}
                  <Label x={x + (gw - 8) / 2} y={122} anchor="middle" tone={3} size={8}>{`${(m.experts / net.gpus).toFixed(expertsPerGpu % 1 ? 1 : 0)} experts`}</Label>
                  <Label x={x + (gw - 8) / 2} y={136} anchor="middle" tone={3} size={8}>{`${(localShare * 100).toFixed(0)}% of work`}</Label>
                  {st >= 2 && r > 0 && (
                    <Arrow x1={x - 2} y1={100} x2={x - 8} y2={100} kind="comm" curve={-14} />
                  )}
                  {st >= 2 && r === 0 && (
                    <Arrow x1={x + gw - 10} y1={100} x2={x + gw - 6} y2={100} kind="comm" curve={14} />
                  )}
                </g>
              );
            })}
            <Label x={12} y={182} tone={3} size={9}>
              {net.gpus > 8 ? `Few experts per card — each expert must still fit on one GPU.` : `Bars above experts: work done. Red = a hot expert when routing is imbalanced.`}
            </Label>
          </g>
        </Fade>

        {/* ── 2+: the all-to-all ── */}
        <Fade show={st >= 2}>
          <g>
            <Label x={12} y={208} tone={2} size={11} bold>The all-to-all</Label>
            <Label x={12} y={228} tone={3} size={9.5} mono={false}>{`Every rank sends a different slice of its tokens to every other rank, once, in one collective.`}</Label>
            <Label x={12} y={252} size={10} mono={false} kind="comm">Dispatch (activations to remote experts)</Label>
            <Span x={250} y={242} w={Math.max(6, Math.min(310, (comm.dispatched / 8e9) * 310))} h={16} kind="comm" label={fmt.bytes(comm.dispatched)} labelTone="dark" />
            <Label x={12} y={278} size={10} mono={false} kind="comm">Return (expert outputs back)</Label>
            <Span x={250} y={268} w={Math.max(6, Math.min(310, (comm.returned / 8e9) * 310))} h={16} kind="comm" label={fmt.bytes(comm.returned)} labelTone="dark" />
            <Dim x1={250} x2={560} y={298} label={`${(comm.fractionRemote * 100).toFixed(0)}% of expert activations cross a card`} kind="comm" />
            <Label x={12} y={316} tone={3} size={9}>Per MoE block. Both directions, all {moeLayerCount(m)} blocks: {fmt.bytes(comm.totalBytes)}.</Label>
          </g>
        </Fade>

        {/* ── 4: imbalance ── */}
        <Fade show={st === 4}>
          <g>
            <Label x={12} y={208} tone={2} size={11} bold>Imbalance loads the wire unevenly</Label>
            <Label x={12} y={228} tone={3} size={9.5} mono={false}>Drag the imbalance control. A hot expert means more tokens travelling,</Label>
            <Label x={12} y={242} tone={3} size={9.5} mono={false}>and the busiest link becomes the latency, not the average link.</Label>
            <Label x={12} y={270} size={10} mono={false} kind="comm">All-to-all volume</Label>
            <Span x={250} y={260} w={Math.max(6, Math.min(310, (comm.totalBytes / 3e11) * 310))} h={16} kind="comm" label={fmt.bytes(comm.totalBytes)} labelTone="dark" />
            <Label x={12} y={300} tone={3} size={9}>At {imbalance.toFixed(2)}× the busiest expert, {fmt.bytes(comm.totalBytes)} cross the fabric per step.</Label>
            <Label x={12} y={318} size={9.5} kind="error" bold>{`The card owning the hot expert does ${imbalance.toFixed(1)}× the compute of an idle card.`}</Label>
          </g>
        </Fade>

        {/* ── 5: capacity ── */}
        <Fade show={st === 5}>
          <g>
            <Label x={12} y={208} tone={2} size={11} bold>Capacity factor: the hard bound</Label>
            <Label x={12} y={228} tone={3} size={9.5} mono={false}>Give each expert C slots and refuse the rest. Rectangular buffers,</Label>
            <Label x={12} y={242} tone={3} size={9.5} mono={false}>predictable memory — at the cost of dropping tokens.</Label>
            {[{ c: 1.0, k: 'text' as const }, { c: 1.25, k: 'token' as const }, { c: 2.0, k: 'error' as const }].map((r, i) => {
              const y = 262 + i * 30;
              const drop = Math.max(0, imbalance - r.c);
              return (
                <g key={r.c}>
                  <Label x={12} y={y + 10} tone={2} size={9.5}>C = {r.c.toFixed(2)}</Label>
                  <rect x={70} y={y} width={300} height={16} rx={2} fill="none" stroke="var(--rule)" />
                  <rect x={70} y={y} width={Math.max(2, (Math.min(r.c, imbalance) / 3) * 300)} height={16} rx={2}
                    fill={drop > 0 ? 'var(--error)' : 'var(--memory)'} fillOpacity={0.85} className="v-cell" />
                  <Label x={382} y={y + 12} size={9.5} tone={drop > 0 ? 1 : 3}>
                    {drop > 0 ? `${((drop / imbalance) * 100).toFixed(0)}% of hot-expert tokens dropped` : 'no drops at this imbalance'}
                  </Label>
                </g>
              );
            })}
            <Label x={12} y={368} tone={3} size={9}>C must be raised as imbalance rises, or the overflow rate climbs with it.</Label>
          </g>
        </Fade>

        {/* ── 6: hybrid ── */}
        <Fade show={st === 6}>
          <g>
            <Label x={12} y={208} tone={2} size={11} bold>Expert parallelism × tensor parallelism</Label>
            <Label x={12} y={228} tone={3} size={9.5} mono={false}>EP shards the bank; TP shards each expert. Both must divide the GPUs.</Label>
            <g>
              <rect x={12} y={244} width={560} height={92} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
              <Label x={24} y={262} size={10} mono={false} bold>Layout</Label>
              <Label x={24} y={282} size={9.5} mono={false} tone={2}>{`EP ${net.gpus} × TP ${tpNeeded} = ${totalGpus} GPUs`}</Label>
              <Label x={24} y={300} size={9.5} mono={false} tone={2}>{`${m.experts} experts ÷ ${net.gpus} = ${(m.experts / net.gpus).toFixed(1)} per card`}</Label>
              <Label x={24} y={318} size={9.5} mono={false} tone={2}>{`expert width ${fmt.int(m.expertIntermediate)} ÷ ${tpNeeded} = ${fmt.int(m.expertIntermediate / tpNeeded)} per shard`}</Label>
              <Label x={330} y={262} size={10} mono={false} bold>Constraint</Label>
              <Label x={330} y={282} size={9} mono={false} tone={2}>{`EP must divide ${m.experts}`}</Label>
              <Label x={330} y={298} size={9} mono={false} tone={2}>{`needs ${fmt.int(m.expertIntermediate / tpNeeded)} ≤ 8,192`}</Label>
              <Label x={330} y={316} size={9} mono={false} tone={2}>{`${totalGpus} GPUs × 80 GB = ${fmt.bytes(totalGpus * g.mem)}`}</Label>
            </g>
            <Label x={12} y={360} tone={3} size={9}>Shard an expert too wide and it will not fit; that, not bandwidth, is the usual EP limit.</Label>
          </g>
        </Fade>

        {/* ── 7: the summary ── */}
        <Fade show={st >= 7}>
          <g>
            <Label x={12} y={208} tone={2} size={11} bold>What you are actually choosing</Label>
            <Ruler x={12} y={252} w={560} ticks={[{ at: 0, label: 'PCIe' }, { at: 0.5, label: 'NVLink' }, { at: 1, label: 'NVSwitch' }]} label="interconnect bandwidth" />
            <Label x={12} y={282} tone={3} size={9.5} mono={false}>Slow fabric (PCIe): keep experts on one card, accept TP inside each.</Label>
            <Label x={12} y={298} tone={3} size={9.5} mono={false}>Fast fabric (NVLink/NVSwitch): shard the bank widely, EP degree up, all-to-all cheap.</Label>
            <Label x={12} y={320} size={10} mono={false} kind="comm" bold>{`Here: ${comm.dispatchMs.toFixed(2)} ms dispatch, ${fmt.bytes(comm.totalBytes)} per step`}</Label>
            <Label x={12} y={338} tone={3} size={9}>{`Weights ${fmt.bytes(weightBytes)} over ${net.gpus} cards · ${fmt.bytes(perGpuBytes)} each · ${tpNeeded}-way TP`}</Label>
            <Label x={12} y={358} size={9.5} kind="compute" bold>Measure whether the wire or the GEMMs is the bottleneck before optimising either.</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}