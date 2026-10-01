import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Arrow, Dim, Fade, Span, Bar, fmt } from '../lib/svg';
import { analyze, verdict, DEFAULT_CONFIG, H100, type Config } from './labModel';

// An eight-step diagnostic funnel: the report you write follows this order exactly.
// The stock transport is hidden because this funnel is navigated by clicking a
// stage of the diagnosis, the way you actually work — read the next row, not a scroll.
const STEPS = [
  'Profile the step',
  'Split the byte budget',
  'Classify the bound',
  'Compare against the floor',
  'Read the Speed of Light',
  'Change one knob',
  'Re-profile and compare',
  'Write the verdict',
];
const CAPTIONS = [
  'One decode step, bar by bar, on an H100 running Llama-3.1-8B. Click any bar to select it — that is how you actually work.',
  'Every byte is one of three: weights, the KV cache, or the attention reads. The largest one is what you should attack.',
  'Compute the intensity and compare it to the ridge point. Below the ridge you are memory-bound and MFU is meaningless here.',
  'The GEMM bars dominate by duration, but duration is not the right sort. Sort by headroom: time over floor.',
  'Speed of Light: the higher of compute % and memory %. Above 80% means stop. Below 60% and the profile names your fix.',
  'One knob at a time. Change two and you cannot attribute the result to either.',
  'Re-run the profile and compare the metric. If MBU or MFU did not move, the change did nothing for this phase.',
  'State the bound, the metric, and the number. Then say what you would try next — or that you would stop.',
];

export default function BottleneckLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { loop: false });
  const [cfg, setCfg] = useState<Config>(DEFAULT_CONFIG);

  const b = analyze(cfg);
  const v = verdict(b);

  const total = b.stepTime;
  const bars = [
    { name: 'GEMM (qkv·o·gate_up·down)', t: b.gemmTime, kind: 'compute' as const },
    { name: 'attention', t: b.attnTime, kind: 'compute' as const },
    { name: 'elementwise (SiLU·mul·add)', t: b.elementwiseTime, kind: 'memory' as const },
    { name: 'RMSNorm', t: b.normTime, kind: 'memory' as const },
    { name: 'launch overhead', t: b.launchOverhead, kind: 'comm' as const },
  ];
  const barT = useTween(total, 400);
  const eff = b.bound === 'memory' ? b.mbu : b.mfu;

  // The byte budget, the thing the report has to split.
  const kvShare = b.kvBytes / b.totalBytes;
  const wShare = b.weightBytes / b.totalBytes;
  const aShare = b.attnBytes / b.totalBytes;

  const set = (k: keyof Config, val: Config[keyof Config]) => setCfg((c) => ({ ...c, [k]: val }));

  return (
    <Stage
      rootRef={ref}
      title="The diagnostic funnel: from a slow decode step to a written bottleneck report"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['token', 'efficient'], ['comm', 'launch overhead'], ['error', 'headroom']]}
      hideTransport
      {...s}
      readout={
        <Readout items={[
          { label: 'Decode step', value: `${(barT * 1e3).toFixed(1)} ms`, kind: 'memory', note: `batch ${cfg.batch}, ctx ${fmt.int(cfg.ctx)}` },
          { label: 'Bound', value: b.bound, kind: b.bound === 'compute' ? 'compute' : 'memory', note: `intensity ${b.intensity.toFixed(1)} FLOP/byte` },
          { label: 'Quote', value: `${(eff * 100).toFixed(0)}% ${b.bound === 'memory' ? 'MBU' : 'MFU'}`, kind: eff > 0.8 ? 'token' : eff > 0.55 ? 'compute' : 'error', note: v.msg },
          { label: 'Headroom', value: `${b.headroom.toFixed(2)}×`, kind: b.headroom > 1.4 ? 'error' : 'token', note: `floor ${(b.idealStep * 1e3).toFixed(1)} ms` },
        ]} />
      }
      controls={
        <>
          <Slider label="Batch (sequences)" value={cfg.batch} min={1} max={256} log onChange={(x) => set('batch', x)} format={(x) => fmt.int(x)} kind="compute" />
          <Slider label="Context (tokens)" value={cfg.ctx} min={128} max={32768} log onChange={(x) => set('ctx', x)} format={(x) => fmt.int(x)} kind="memory" />
          <Segmented
            label="Weight precision"
            value={String(cfg.wBytes)}
            options={[{ value: '2', label: 'BF16' }, { value: '1', label: 'FP8' }, { value: '0.5', label: '4-bit' }]}
            onChange={(x) => set('wBytes', Number(x))}
          />
          <Segmented
            label="Knobs"
            value={[
              cfg.attnFlash && 'attnFlash', cfg.fused && 'fused',
              cfg.coalesced && 'coalesced', cfg.cudaGraph && 'cudaGraph',
            ].filter(Boolean).join(',') || 'none'}
            options={[
              { value: 'attnFlash', label: 'Flash attn' },
              { value: 'fused', label: 'Fuse elemwise' },
              { value: 'coalesced', label: 'Coalesced' },
              { value: 'cudaGraph', label: 'CUDA graphs' },
              { value: 'none', label: 'All off' },
            ]}
            onChange={(x) => {
              const on = x === 'none' ? [] : x.split(',');
              setCfg((c) => ({ ...c, attnFlash: on.includes('attnFlash'), fused: on.includes('fused'), coalesced: on.includes('coalesced'), cudaGraph: on.includes('cudaGraph') }));
            }}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 420" role="img" aria-label="A decode step broken into kernel bars, with a byte-budget split between weights, KV cache and attention, the ridge-point classification, the Speed of Light reading and the remaining headroom — with sliders that change the configuration.">
        {/* ── step header: the funnel you are on ── */}
        <Label x={16} y={26} tone={1} size={11} bold mono={false}>{STEPS[s.step]}</Label>
        <Label x={624} y={26} anchor="end" size={9.5} tone={3}>
          Llama-3.1-8B · H100 · one decode step
        </Label>

        {/* the funnel rail: click a stage of the diagnosis to move through it */}
        <g>
          {STEPS.map((label, i) => {
            const w = 608 / STEPS.length;
            const x = 16 + i * w;
            const done = i < s.step;
            const now = i === s.step;
            return (
              <g
                key={label}
                style={{ cursor: 'pointer' }}
                onClick={() => s.go(i)}
                role="button"
                aria-label={`Step ${i + 1}: ${label}`}
              >
                <rect
                  x={x + 1} y={32} width={w - 2} height={4} rx={2}
                  fill={now ? 'var(--text)' : done ? 'var(--text-3)' : 'var(--rule-2)'}
                />
                <Label x={x + w / 2} y={30} anchor="middle" size={7.5} tone={now ? 1 : 3}>{`${i + 1}`}</Label>
              </g>
            );
          })}
        </g>

        {/* ── the kernel bars ── */}
        <Label x={16} y={52} tone={2} size={10} bold mono={false}>Where the time goes</Label>
        <Label x={624} y={52} anchor="end" tone={3} size={9}>{`${b.nKernels} kernels`}</Label>
        {(() => {
          let x = 16;
          const W = 608;
          return bars.map((bar) => {
            const w = (bar.t / total) * W;
            const el = (
              <g key={bar.name}>
                <Span x={x} y={60} w={Math.max(0, w)} h={24} kind={bar.kind} label={bar.name} />
                {w > 40 && (
                  <Label x={x + w / 2} y={92} anchor="middle" size={8.5} tone={3}>
                    {(bar.t * 1e3).toFixed(bar.t * 1e3 < 1 ? 2 : 1)} ms
                  </Label>
                )}
              </g>
            );
            x += w;
            return el;
          });
        })()}
        <Dim x1={16} x2={624} y={108} label={`${(total * 1e3).toFixed(1)} ms total`} kind="text" />

        {/* ── the funnel: one panel per step, revealed as you advance ── */}
        <Fade show={s.step >= 1}>
          <Label x={16} y={136} tone={2} size={10} bold mono={false}>The byte budget</Label>
          <Label x={624} y={136} anchor="end" tone={3} size={9}>{`${(b.totalBytes / 1e9).toFixed(1)} GB per step`}</Label>
          <Bar x={112} y={128} w={428} frac={wShare} kind="compute" value={`${(b.weightBytes / 1e9).toFixed(1)} GB`} label="weights" />
          <Bar x={112} y={148} w={428} frac={kvShare} kind="memory" value={`${(b.kvBytes / 1e9).toFixed(1)} GB`} label="KV cache" />
          <Bar x={112} y={168} w={428} frac={aShare} kind="memory" value={`${(b.attnBytes / 1e9).toFixed(1)} GB`} label="attention" />
        </Fade>

        <Fade show={s.step >= 2}>
          <Label x={16} y={204} tone={2} size={10} bold mono={false}>Against the ridge</Label>
          <Label x={70} y={204} size={9.5} tone={2}>
            intensity {b.intensity.toFixed(1)} vs ridge {(H100.flops / H100.bw).toFixed(0)} FLOP/byte →
          </Label>
          <Label x={430} y={204} size={10} bold kind={b.bound === 'compute' ? 'compute' : 'memory'}>
            {b.bound === 'compute' ? 'COMPUTE-BOUND' : 'MEMORY-BOUND'}
          </Label>
          {/* a tiny intensity bar against the ridge */}
          <rect x={70} y={212} width={420} height={8} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={70} y={212} width={420 * Math.min(1, b.intensity / 300)} height={8} rx={2} fill="var(--compute)" className="v-cell" />
          <line x1={70 + 420 * Math.min(1, (H100.flops / H100.bw) / 300)} x2={70 + 420 * Math.min(1, (H100.flops / H100.bw) / 300)} y1={208} y2={224} stroke="var(--error)" strokeWidth={1.4} />
          <Label x={70 + 420 * Math.min(1, (H100.flops / H100.bw) / 300) + 4} y={232} size={8.5} kind="error">ridge</Label>
        </Fade>

        {/* ── Speed of Light ── */}
        <Fade show={s.step >= 4}>
          <Label x={16} y={256} tone={2} size={10} bold mono={false}>Speed of Light</Label>
          <Label x={70} y={256} size={9.5} tone={3}>memory</Label>
          <rect x={120} y={248} width={300} height={10} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={120} y={248} width={300 * Math.min(1, b.mbu)} height={10} rx={2} fill="var(--memory)" className="v-cell" />
          <Label x={428} y={257} size={9.5} kind="memory">{(b.mbu * 100).toFixed(0)}% MBU</Label>
          <Label x={70} y={276} size={9.5} tone={3}>compute</Label>
          <rect x={120} y={268} width={300} height={10} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={120} y={268} width={300 * Math.min(1, b.mfu)} height={10} rx={2} fill="var(--compute)" className="v-cell" />
          <Label x={428} y={277} size={9.5} kind="compute">{(b.mfu * 100).toFixed(0)}% MFU</Label>
          {/* the one that matters */}
          <rect x={120} y={286} width={300} height={2} fill="none" />
          <Arrow x1={480} y1={253} x2={480} y2={273} kind={b.bound === 'compute' ? 'compute' : 'memory'} />
          <Label x={490} y={266} size={9.5} bold kind={eff > 0.8 ? 'token' : eff > 0.55 ? 'compute' : 'error'}>
            {eff > 0.8 ? 'near the roof' : eff > 0.55 ? 'some room' : 'fix this kernel'}
          </Label>
        </Fade>

        {/* ── headroom vs the floor ── */}
        <Fade show={s.step >= 3}>
          <Label x={16} y={316} tone={2} size={10} bold mono={false}>Headroom</Label>
          <Label x={16} y={336} tone={3} size={9}>floor</Label>
          <rect x={70} y={328} width={420} height={12} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={70} y={328} width={420 * Math.min(1, b.idealStep / total)} height={12} rx={2} fill="var(--idle)" className="v-cell" />
          <Label x={70 + 420 * Math.min(1, b.idealStep / total) + 6} y={338} size={9.5} tone={3}>{`${(b.idealStep * 1e3).toFixed(1)} ms`}</Label>
          <Label x={16} y={354} tone={3} size={9}>you</Label>
          <rect x={70} y={346} width={420} height={12} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={70} y={346} width={420} height={12} rx={2} fill="var(--error)" opacity={0.5} className="v-cell" />
          <Label x={70 + 426} y={356} size={9.5} kind="error">{`${(total * 1e3).toFixed(1)} ms = ${b.headroom.toFixed(2)}× the floor`}</Label>
        </Fade>

        {/* ── the verdict banner ── */}
        <Fade show={s.step >= 7}>
          <rect x={16} y={374} width={608} height={40} rx={3}
            fill={v.ok ? 'var(--token-dim)' : 'var(--error-dim)'}
            stroke={v.ok ? 'var(--token)' : 'var(--error)'} strokeWidth={1} />
          <Label x={30} y={390} size={10.5} bold kind={v.ok ? 'token' : 'error'}>
            {`${cfg.batch} batch · ${fmt.int(cfg.ctx)} ctx → ${b.bound}-bound · quote ${(eff * 100).toFixed(0)}% ${b.bound === 'memory' ? 'MBU' : 'MFU'} · ${b.headroom.toFixed(2)}× headroom`}
          </Label>
          <Label x={30} y={406} size={9.5} kind={v.ok ? 'token' : 'error'}>{v.msg}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
