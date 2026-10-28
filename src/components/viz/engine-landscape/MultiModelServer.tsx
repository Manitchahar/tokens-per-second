import { useRef, useState } from 'react';
import { Stage, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Arrow, Ruler } from '../lib/svg';

// Six steps: the dedicated fleet, the packed pool, the three cache tiers, the
// price of a cold load, LRU eviction, and finally the router that ties it together.
const STEPS = [
  'Twenty models, twenty GPUs',
  'Pack them into one pool',
  'Three tiers: GPU, RAM, disk',
  'The bill for a cold model',
  'Evict the least recently used',
  'Route each name to a replica',
];

const CAPTIONS = [
  'Twenty small models, each with its own GPU. Every card is loaded with one model all the time, and almost every card is idle almost all the time: about three percent compute, a few percent of memory, and the rest is hatched waiting. The bill is paid for twenty cards to do the work of less than one.',
  'The same twenty models packed into two GPUs. Models are small enough to share a card, so seven fit on each and the rest wait on CPU RAM or disk. The work is unchanged, the latency of a resident model is unchanged, and the fleet is now two cards instead of twenty.',
  'A packed server has three places a model can be: resident in GPU memory, staged in CPU RAM, or stored on disk. Only the first can answer immediately. A warm swap copies weights from CPU RAM over PCIe; a cold load reads them from disk first. The tiers trade cost per byte against time to first token.',
  'A cache hit costs nothing. A warm swap of a 14 GB model over PCIe at about 25 GB/s costs roughly half a second. A cold load adds the disk read in front of it: at an approximate 1 GB/s that is fourteen seconds before the first token. This is the number that makes routing, not packing, the hard part of multi-model serving.',
  'The GPU pool is full, so a cold model must evict something. LRU throws out the model that has gone longest without a request: here M03 first, then M08, then M10 and onward, until 14 GB is free. The cache is a heuristic, not an oracle: a model that is about to get busy again can be evicted just before it is needed.',
  'The router is the front door. It holds a map from model name to the hosts that already have that model, plus a replica count per model. A request for the hot model M07 goes to either of its two replicas; a request for M18 goes to the one host that can load it, and pays the cold load there. Without this map, packing is a way to turn every request into a coin flip.',
];

interface M {
  id: string;
  gb: number;
  hot?: boolean;
}

// Twenty small models: embeddings, rerankers, classifiers and small fine-tunes.
// Sizes are deliberately concrete so the memory arithmetic on screen adds up.
const MODELS: M[] = [
  { id: 'M01', gb: 0.8 }, { id: 'M02', gb: 1.6 }, { id: 'M03', gb: 1.0 },
  { id: 'M04', gb: 1.3 }, { id: 'M05', gb: 3.1 }, { id: 'M06', gb: 4.7 },
  { id: 'M07', gb: 2.4, hot: true }, { id: 'M08', gb: 1.8 }, { id: 'M09', gb: 2.9 },
  { id: 'M10', gb: 2.1 }, { id: 'M11', gb: 3.9 }, { id: 'M12', gb: 5.2 },
  { id: 'M13', gb: 2.3 }, { id: 'M14', gb: 3.4 }, { id: 'M15', gb: 6.8 },
  { id: 'M16', gb: 1.6 }, { id: 'M17', gb: 4.4 }, { id: 'M18', gb: 14.0 },
  { id: 'M19', gb: 2.6 }, { id: 'M20', gb: 3.6 },
];

const sum = (ms: M[]) => ms.reduce((t, m) => t + m.gb, 0);

const GPU_A = MODELS.slice(0, 7);                       // 14.9 GB
const GPU_B = [...MODELS.slice(7, 13), MODELS[6]];      // 20.6 GB, includes M07's second replica
const STAGED = [MODELS[13], MODELS[14], MODELS[16], MODELS[19]]; // M14, M15, M17, M20
const DEEP = [MODELS[15], MODELS[17], MODELS[18]];      // M16, M18, M19

const TOTAL_GB = sum(MODELS);
const GPU_GB = sum(GPU_A) + sum(GPU_B);
const STAGED_GB = sum(STAGED);

// LRU victims, oldest first. Enough are listed to make the ordering visible.
const LRU = ['M03', 'M08', 'M10', 'M01', 'M13', 'M02', 'M04'];

const PCIE_GBPS = 25; // approximate PCIe Gen4 x16 copy bandwidth
const COLD_GB = 14;    // one 7B model in FP16, the cold-load example

function ModelChip({
  x, y, w, h, m, tier, evict,
}: {
  x: number; y: number; w: number; h: number; m: M;
  tier: 'gpu' | 'cpu' | 'disk'; evict?: boolean;
}) {
  const fill = evict ? 'var(--error-dim)' : tier === 'gpu' ? 'var(--memory)' : tier === 'cpu' ? 'var(--memory-dim)' : 'url(#tps-hatch)';
  const stroke = evict ? 'var(--error)' : tier === 'disk' ? 'var(--idle)' : 'var(--memory)';
  const ink = evict ? 'var(--text)' : tier === 'gpu' ? '#0e1116' : 'var(--text)';
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={3} fill={fill} stroke={stroke} strokeWidth={1} className="v-cell" />
      <text x={x + 7} y={y + h / 2 + 3.5} className="v-label sans bold" style={{ fontSize: 10, fill: ink }}>{m.id}</text>
      <text x={x + w - 7} y={y + h / 2 + 3.5} textAnchor="end" className="v-label" style={{ fontSize: 8.5, fill: tier === 'gpu' && !evict ? '#0e1116' : 'var(--text-2)' }}>
        {m.gb.toFixed(1)}
      </text>
      {m.hot && !evict && <circle cx={x + w - 7} cy={y + 6} r={2.2} fill="var(--compute)" />}
    </g>
  );
}

export default function MultiModelServer() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [diskGBps, setDiskGBps] = useState(1);

  const tDisk = COLD_GB / diskGBps;
  const tPcie = COLD_GB / PCIE_GBPS;
  const tCold = tDisk + tPcie;

  const scaleMax = 20; // seconds shown by the cold-load ruler
  const barX = 150;
  const barW = 380;
  const secW = (sec: number) => (sec / scaleMax) * barW;

  return (
    <Stage
      rootRef={ref}
      title="Multi-model serving: twenty models, one pool of GPUs"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'GPU / RAM memory'],
        ['compute', 'compute or a hot model'],
        ['comm', 'PCIe / network'],
        ['idle', 'idle or stored'],
        ['error', 'evicted / miss'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Models', value: '20', kind: 'text', note: '13 resident, 7 on demand' },
          { label: 'GPU memory', value: `${GPU_GB.toFixed(1)} / 43.2 GB`, kind: 'memory', note: 'two 24 GB L4s, 10% reserve' },
          { label: 'Fleet cost', value: '$1.60/h', kind: 'token', note: 'vs $16/h dedicated' },
          { label: 'Cold load, 7B', value: `${tCold.toFixed(1)} s`, kind: 'error', note: `${COLD_GB} GB at ${diskGBps.toFixed(1)} GB/s` },
        ]} />
      }
      controls={
        <Slider
          label="Disk read bandwidth (drives the cold-load bar)"
          kind="memory"
          min={1}
          max={4}
          step={0.5}
          value={diskGBps}
          onChange={setDiskGBps}
          format={(v) => `${v.toFixed(1)} GB/s`}
        />
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Twenty small models served on twenty mostly idle GPUs, then packed into two GPUs. Three cache tiers appear: GPU memory, CPU RAM and disk. A cold load is priced on a time ruler, LRU eviction is shown on the resident models, and finally a router maps each model name to the GPU replicas that hold it.">
        <Label x={40} y={22} tone={2} size={11} bold>
          {[
            'Before: one model per GPU, all loaded all the time',
            'After: one pool, two GPUs, models load on demand',
            'Three tiers: resident, staged, stored',
            'The bill for a cold model',
            'The pool is full: LRU decides who leaves',
            'The front door: model name to host',
          ][s.step]}
        </Label>
        <Label x={614} y={22} anchor="end" tone={3} size={9}>
          {s.step === 0 ? '20 GPUs · ≈ $16/h' : s.step === 1 ? '2 GPUs · ≈ $1.60/h' : 'one pool · 20 models'}
        </Label>

        {/* ── step 0: twenty dedicated GPUs, almost all idle ───────────── */}
        <Fade show={s.step === 0}>
          {MODELS.map((m, i) => {
            const c = i % 5;
            const r = Math.floor(i / 5);
            const x = 70 + c * 102;
            const y = 50 + r * 58;
            const bw = 78;
            return (
              <g key={m.id}>
                <rect x={x} y={y} width={94} height={46} rx={3} fill="none" stroke="var(--rule)" className="v-cell" />
                <Label x={x + 8} y={y + 16} size={9.5} bold tone={1}>{m.id}</Label>
                <Label x={x + 86} y={y + 16} anchor="end" size={7.5} tone={3}>idle</Label>
                {/* one bar: memory, then a sliver of compute, then idle hatch */}
                <rect x={x + 8} y={y + 25} width={bw} height={8} rx={2} fill="url(#tps-hatch)" stroke="var(--idle)" />
                <rect x={x + 8} y={y + 25} width={bw * 0.12} height={8} rx={2} fill="var(--memory)" fillOpacity={0.85} />
                <rect x={x + 8 + bw * 0.12} y={y + 25} width={Math.max(1, bw * 0.03)} height={8} rx={1} fill="var(--compute)" />
                <Label x={x + 8} y={y + 42} size={7.5} tone={3}>1 model resident</Label>
              </g>
            );
          })}
          <Label x={40} y={296} size={10.5} bold tone={2}>Twenty cards, twenty weights files, twenty idle stories</Label>
          <Label x={40} y={312} size={9.5} tone={3}>Each GPU is about 12% full of weights and about 3% busy on compute.</Label>
          <Label x={40} y={326} size={9.5} tone={3}>The hatched area is capacity you pay for and do not use.</Label>
          <Label x={40} y={344} size={9.5} kind="error">Cost ≈ 20 × $0.80/h = $16/h, for aggregate compute under one GPU’s worth.</Label>
        </Fade>

        {/* ── step 1: packed onto two GPUs ─────────────────────────────── */}
        <Fade show={s.step === 1}>
          {[GPU_A, GPU_B].map((arr, gi) => {
            const x = 40 + gi * 290;
            const used = sum(arr);
            const name = gi === 0 ? 'GPU A' : 'GPU B';
            return (
              <g key={name}>
                <rect x={x} y={46} width={270} height={206} rx={4} fill="none" stroke="var(--rule-2)" className="v-cell" />
                <Label x={x + 10} y={66} size={10.5} bold tone={1}>{name} · {used.toFixed(1)} / 24 GB</Label>
                <Label x={x + 260} y={66} anchor="end" size={8.5} tone={3}>{arr.length} instances</Label>
                {arr.map((m, i) => {
                  const c = i % 2;
                  const r = Math.floor(i / 2);
                  return (
                    <ModelChip key={`${name}-${m.id}`} x={x + 10 + c * 126} y={80 + r * 38} w={120} h={32} m={m} tier="gpu" />
                  );
                })}
                <rect x={x + 10} y={236} width={250} height={6} rx={3} fill="url(#tps-hatch)" stroke="var(--idle)" />
                <rect x={x + 10} y={236} width={250 * (used / 24)} height={6} rx={3} fill="var(--memory)" fillOpacity={0.85} />
              </g>
            );
          })}
          <Label x={40} y={286} size={10.5} bold tone={2}>Same models, same traffic, one tenth of the fleet</Label>
          <Label x={40} y={302} size={9.5} tone={3}>Thirteen models are resident; M07 has a second replica on GPU B because it is hot.</Label>
          <Label x={40} y={316} size={9.5} tone={3}>The other seven are on CPU RAM or disk.</Label>
          <Label x={40} y={332} size={9.5} kind="memory">Weights 35.5 GB of a 43.2 GB usable pool.</Label>
          <Label x={40} y={346} size={9.5} kind="memory">Compute is the easy part; model memory is what limits the packing.</Label>
          <Label x={40} y={362} size={9.5} kind="token">Cost ≈ 2 × $0.80/h = $1.60/h. A resident request is served by the same kernels as before.</Label>
        </Fade>

        {/* ── steps 2-4: the cache tiers ───────────────────────────────── */}
        <Fade show={s.step >= 2 && s.step <= 4}>
          {/* GPU band */}
          <Label x={40} y={48} size={10} bold tone={2}>GPU VRAM — resident, answer now</Label>
          <rect x={40} y={54} width={470} height={92} rx={4} fill="var(--memory-dim)" stroke="var(--memory)" strokeOpacity={0.5} />
          {[...GPU_A, ...GPU_B].map((m, i) => {
            const c = i % 7;
            const r = Math.floor(i / 7);
            return (
              <ModelChip key={`gpu-${i}`} x={54 + c * 64} y={64 + r * 38} w={58} h={30} m={m} tier="gpu" />
            );
          })}
          <Label x={496} y={62} anchor="end" size={8.5} tone={3}>{GPU_GB.toFixed(1)} GB used</Label>

          {/* CPU band */}
          <Label x={40} y={162} size={10} bold tone={2}>CPU RAM — staged, one PCIe copy away</Label>
          <rect x={40} y={168} width={470} height={52} rx={4} fill="var(--ground-3)" stroke="var(--memory)" strokeOpacity={0.35} strokeDasharray="4 3" />
          {STAGED.map((m, i) => (
            <ModelChip key={`cpu-${m.id}`} x={54 + i * 116} y={180} w={108} h={28} m={m} tier="cpu" />
          ))}
          <Label x={496} y={198} anchor="end" size={8.5} tone={3}>{STAGED_GB.toFixed(1)} GB</Label>

          {/* disk band */}
          <Label x={40} y={234} size={10} bold tone={2}>Disk / object store — every model file, the source of truth</Label>
          <rect x={40} y={240} width={470} height={40} rx={4} fill="none" stroke="var(--rule-2)" strokeDasharray="4 3" />
          {MODELS.map((m, i) => (
            <g key={`disk-${m.id}`}>
              <rect x={54 + i * 20} y={252} width={15} height={16} rx={2}
                fill={DEEP.includes(m) ? 'url(#tps-hatch)' : 'var(--memory-dim)'}
                stroke={DEEP.includes(m) ? 'var(--idle)' : 'var(--memory)'} strokeWidth={0.8} />
              <text x={61.5 + i * 20} y={263.5} textAnchor="middle" className="v-label" style={{ fontSize: 6.5, fill: 'var(--text-2)' }}>{m.id.slice(1)}</text>
            </g>
          ))}
          <Label x={496} y={264} anchor="end" size={8.5} tone={3}>{TOTAL_GB.toFixed(1)} GB total</Label>

          {/* the latency panel, constant across these steps */}
          <line x1={520} y1={54} x2={520} y2={280} className="v-rule faint" />
          <Label x={530} y={48} size={10} bold tone={2}>A request waits</Label>
          <Label x={530} y={70} size={9} tone={3}>cache hit</Label>
          <Label x={530} y={86} size={11} kind="token">≈ 0 s</Label>
          <Label x={530} y={108} size={9} tone={3}>warm swap</Label>
          <Label x={530} y={124} size={11} kind="memory">{tPcie.toFixed(2)} s</Label>
          <Label x={530} y={146} size={9} tone={3}>cold load</Label>
          <Label x={530} y={162} size={11} kind="error">{tCold.toFixed(1)} s</Label>
          <Label x={530} y={182} size={8} tone={3}>7B FP16 = 14 GB;</Label>
          <Label x={530} y={194} size={8} tone={3}>PCIe ≈ 25 GB/s</Label>
          <Label x={530} y={214} size={8.5} tone={2}>Tiers trade</Label>
          <Label x={530} y={226} size={8.5} tone={2}>cost per GB</Label>
          <Label x={530} y={238} size={8.5} tone={2}>against time.</Label>
        </Fade>

        {/* ── step 2: what the tiers mean ──────────────────────────────── */}
        <Fade show={s.step === 2}>
          <Label x={40} y={300} size={10} bold tone={2}>Where a model sits decides what a request pays</Label>
          <Label x={40} y={318} size={9.5} kind="memory">resident — weights in GPU memory; the request starts now</Label>
          <Label x={40} y={336} size={9.5} kind="comm">staged — weights in CPU RAM; copy over PCIe before the first token</Label>
          <Label x={40} y={354} size={9.5} kind="idle">stored — weights on disk; read into RAM, then copy over PCIe</Label>
        </Fade>

        {/* ── step 3: price the three cases ────────────────────────────── */}
        <Fade show={s.step === 3}>
          <Label x={40} y={284} size={10} bold tone={2}>Time to first token, by where the model is</Label>
          {/* hit */}
          <Label x={40} y={302} size={9.5} tone={2}>cache hit</Label>
          <rect x={barX} y={290} width={2} height={16} fill="var(--token)" />
          <Label x={barX + 10} y={302} size={9} tone={3}>already resident</Label>
          <Label x={614} y={302} anchor="end" size={10} kind="token">0 s</Label>
          {/* warm swap */}
          <Label x={40} y={326} size={9.5} tone={2}>warm swap</Label>
          <rect x={barX} y={314} width={Math.max(2, secW(tPcie))} height={16} rx={2} fill="var(--comm)" />
          <Label x={614} y={326} anchor="end" size={10} kind="comm">{tPcie.toFixed(2)} s</Label>
          {/* cold load */}
          <Label x={40} y={350} size={9.5} tone={2}>cold load</Label>
          <rect x={barX} y={338} width={Math.max(2, secW(tDisk))} height={16} rx={2} fill="var(--memory)" fillOpacity={0.85} />
          <rect x={barX + secW(tDisk)} y={338} width={Math.max(2, secW(tPcie))} height={16} rx={2} fill="var(--comm)" />
          <Label x={614} y={350} anchor="end" size={10} kind="error">{tCold.toFixed(1)} s</Label>
          <Ruler
            x={barX}
            y={360}
            w={barW}
            ticks={[0, 5, 10, 15, 20].map((v) => ({ at: v / scaleMax, label: `${v} s` }))}
          />
        </Fade>

        {/* ── step 4: LRU eviction ─────────────────────────────────────── */}
        <Fade show={s.step === 4}>
          <Label x={40} y={288} size={10} bold tone={2}>LRU queue — oldest request first</Label>
          {LRU.map((id, i) => (
            <g key={`lru-${id}`}>
              <rect x={40 + i * 72} y={298} width={64} height={26} rx={3}
                fill={i === 0 ? 'var(--error-dim)' : 'var(--memory-dim)'}
                stroke={i === 0 ? 'var(--error)' : 'var(--memory)'} />
              <text x={72 + i * 72} y={315} textAnchor="middle" className="v-label sans bold" style={{ fontSize: 10, fill: 'var(--text)' }}>{id}</text>
            </g>
          ))}
          <Arrow x1={72} y1={324} x2={72} y2={338} kind="error" />
          <Label x={40} y={352} size={8.5} kind="error">M03 evicted to disk</Label>
          <Label x={132} y={352} size={9} tone={3}>M08 and M10 are next in line.</Label>
          <Label x={40} y={368} size={9.5} tone={2}>M18 needs 14 GB. The cache frees it in LRU order: M03 first, then M08, M10, …</Label>
        </Fade>

        {/* ── step 5: the router ───────────────────────────────────────── */}
        <Fade show={s.step === 5}>
          {/* request into the router */}
          <rect x={40} y={34} width={76} height={46} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" />
          <Label x={78} y={54} anchor="middle" size={8} tone={3}>request</Label>
          <Label x={78} y={70} anchor="middle" size={11} bold kind="compute">M07</Label>
          <Arrow x1={116} y1={57} x2={132} y2={57} kind="compute" />
          {/* router */}
          <rect x={132} y={34} width={482} height={46} rx={3} fill="var(--ground-3)" stroke="var(--comm)" />
          <Label x={144} y={54} size={10} bold kind="comm">ROUTER</Label>
          <Label x={144} y={70} size={8.5} tone={3}>model name → hosts, plus a replica count per model</Label>
          <Label x={604} y={54} anchor="end" size={8.5} tone={2}>M07 ×2</Label>
          <Label x={604} y={70} anchor="end" size={8.5} tone={2}>M04 ×1 · M18 ×0</Label>
          <Arrow x1={230} y1={80} x2={180} y2={102} kind="comm" />
          <Arrow x1={500} y1={80} x2={470} y2={102} kind="comm" />
          {/* two pools */}
          {[
            { list: GPU_A, x: 40, name: 'GPU A' },
            { list: GPU_B, x: 330, name: 'GPU B' },
          ].map(({ list, x, name }) => {
            return (
              <g key={name}>
                <rect x={x} y={104} width={270} height={142} rx={4} fill="none" stroke="var(--rule-2)" />
                <Label x={x + 10} y={122} size={10} bold tone={1}>{name} · {sum(list).toFixed(1)} / 24 GB</Label>
                {list.map((m, i) => {
                  const c = i % 3;
                  const r = Math.floor(i / 3);
                  return (
                    <ModelChip key={`${name}-${m.id}`} x={x + 10 + c * 84} y={132 + r * 32} w={78} h={26} m={m} tier="gpu" />
                  );
                })}
                <Label x={x + 10} y={236} size={8.5} tone={3}>{name === 'GPU A' ? 'serves M01–M07' : 'serves M08–M13 + a hot M07 replica'}</Label>
              </g>
            );
          })}
          {/* route table */}
          <Label x={40} y={270} size={10} bold tone={2}>The map is the source of truth</Label>
          <Label x={40} y={288} size={9.5} kind="compute">M07 (hot) → GPU A + GPU B · 2 replicas · route to either</Label>
          <Label x={40} y={306} size={9.5} kind="memory">M04 → GPU A · 1 replica · already resident, so serve now</Label>
          <Label x={40} y={324} size={9.5} kind="error">M18 (cold) → GPU B · no replica resident · pay ≈ 14.6 s first</Label>
          <Label x={40} y={338} size={9} tone={3}>A request routed to a host without the model is not an error; it is a load.</Label>
          <Label x={40} y={352} size={9} tone={3}>The router’s job is to know that in advance.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
