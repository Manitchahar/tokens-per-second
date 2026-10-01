import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Chip, Arrow, Fade, Bar, fmt } from '../lib/svg';
import {
  DIFFUSION_MODELS, diffusionById, diffusionBatch, diffusionStep, stepQualityTradeoff, CACHE_MODES, gpu,
} from './workloads';

const STEPS = [
  'Diffusion is not autoregressive',
  'The latent: why 512×512 becomes 64×64',
  'The denoising loop runs 20–50 times',
  'Where it sits on the roofline',
  'The contrast with LLM decode',
  'Guidance doubles the batch',
  'VAE decode: one huge kernel at the end',
  'Fewer steps, smaller model, cached features',
];
const CAPTIONS = [
  'An LLM generates one token at a time and keeps a KV cache so it never recomputes. Diffusion has neither. It starts with pure noise and repeatedly refines the whole image, running the network over every pixel every time. There is no cache, because there is nothing to reuse.',
  'The U-Net never sees pixels. It sees a latent: an image downsampled by 8 and given 4 channels. At 512×512 that is 64×64×4 — 1,024 numbers instead of 786,432. Everything downstream happens at that size, and only the final decode returns to full resolution.',
  'Each denoising step takes the current latent plus a timestep and predicts less noise. Twenty to fifty of those steps, each a full forward pass of the U-Net. This is the same shape as prefill — a large dense compute job — repeated N times.',
  'Put that on the roofline from Module 12. A U-Net step does dense convolutions over hundreds of feature maps, so arithmetic intensity is tens of FLOP per byte. It is firmly compute-bound, in a region LLM decode never reaches.',
  'That single fact explains most of the engineering differences. Diffusion throughput scales with FLOP/s, so a faster compute card wins. LLM decode scales with bandwidth, so a wider memory bus wins. Same GPU, opposite conclusions.',
  'Classifier-free guidance runs the U-Net twice per step — once on the prompt, once on an empty prompt — and combines them. That means batch 2 internally, which doubles the step time. It is not a refinement; it is the standard recipe.',
  'The last step decodes the latent back to pixels through the VAE. One enormous convolution stack at full resolution, run once. It is easy to overlook because it happens once, and it can dominate the memory profile.',
  'Three ways to go faster, in order of how much they change the image: fewer steps (a different picture), a smaller or distilled model (a different model), or caching features across steps (nearly the same picture, 20–40% less compute).',
];

export default function Diffusion() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [mid, setMid] = useState<string>('sd15');
  const [size, setSize] = useState(512);
  const [batch, setBatch] = useState(4);
  const [gId, setGId] = useState<string>('a100');
  const [cacheId, setCacheId] = useState<string>('residual');

  const m = diffusionById(mid);
  const g = gpu(gId);
  const db = useMemo(() => diffusionBatch(m, g, size, batch), [m, g, size, batch]);
  const ds = useMemo(() => diffusionStep(m, g, size, batch), [m, g, size, batch]);
  const trade = useMemo(() => stepQualityTradeoff(m, g, size), [m, g, size]);
  const cache = CACHE_MODES.find((c) => c.id === cacheId)!;

  const latentSide = size / m.downsample;
  // A coarse latent grid, drawn small.
  const GRID = 8;
  const cell = 4;

  return (
    <Stage
      rootRef={ref}
      title="Diffusion inference: many dense steps, no cache"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'the U-Net / DiT'],
        ['memory', 'activations and the VAE'],
        ['token', 'the latent being denoised'],
        ['comm', 'text encoder'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Latent', value: `${latentSide}×${latentSide}×${m.latentChannels}`, kind: 'token', note: `${fmt.int(db.latentTokens)} values per image` },
          { label: 'Per image', value: `${ds.loopS.toFixed(2)} s`, kind: 'compute', note: `${m.steps} steps × ${(ds.stepMs).toFixed(0)} ms` },
          { label: 'Bandwidth used', value: `${(ds.bandwidthUtil * 100).toFixed(1)}%`, kind: ds.bound === 'compute' ? 'token' : 'memory', note: `${ds.bound}-bound` },
          { label: 'Throughput', value: `${ds.ipsPerGpu.toFixed(2)} img/s`, kind: 'memory', note: `batch ${batch} at ${size}px` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={mid} options={DIFFUSION_MODELS.map((x) => ({ value: x.id, label: x.name.replace('Stable Diffusion ', 'SD ').replace(' base', '') }))} onChange={setMid} />
          <Segmented label="GPU" value={gId} options={[{ value: 'a100', label: 'A100' }, { value: 'h100', label: 'H100' }, { value: 'l4', label: 'L4' }, { value: 't4', label: 'T4' }]} onChange={setGId} />
          <Slider label="Resolution" kind="memory" min={256} max={1024} step={64} value={size} onChange={setSize} format={(v) => `${v}px`} />
          <Slider label="Images per batch" kind="token" min={1} max={16} step={1} value={batch} onChange={setBatch} format={(v) => `${v} images`} />
          <Segmented label="Feature cache" value={cacheId} options={CACHE_MODES.map((c) => ({ value: c.id, label: c.label }))} onChange={setCacheId} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="A diffusion latent being denoised over many steps, its position on the roofline compared with LLM decode, and the memory split between weights and activations.">
        {/* ── panel A: noise to image ── */}
        <Label x={12} y={20} tone={2} size={11} bold>{st === 0 ? 'Noise in, image out — no tokens, no cache' : 'The denoising loop'}</Label>
        {/* latent grid */}
        <g transform="translate(12,32)">
          <rect x={0} y={0} width={GRID * cell} height={GRID * cell} rx={2} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={1} />
          {Array.from({ length: GRID }).map((_, r) =>
            Array.from({ length: GRID }).map((_, c) => (
              <rect key={`${r}-${c}`} x={c * cell + 0.4} y={r * cell + 0.4} width={cell - 0.8} height={cell - 0.8}
                fill="var(--token)" opacity={0.3 + 0.6 * ((r * 5 + c * 11 + st * 3) % 7) / 7} className="v-cell" />
            )),
          )}
          <Label x={GRID * cell / 2} y={GRID * cell + 15} anchor="middle" tone={3} size={8.5}>{`${latentSide}×${latentSide}×${m.latentChannels}`}</Label>
        </g>
        <Arrow x1={12 + GRID * cell + 6} y1={32 + (GRID * cell) / 2} x2={78} y2={32 + (GRID * cell) / 2} kind="compute" />
        <Chip x={80} y={32 + (GRID * cell) / 2 - 11} text="U-Net" kind="compute" h={22} w={44} size={9.5} />
        <Arrow x1={128} y1={32 + (GRID * cell) / 2} x2={158} y2={32 + (GRID * cell) / 2} kind="compute" />
        <Chip x={160} y={32 + (GRID * cell) / 2 - 11} text="× steps" kind="compute" h={22} w={44} size={9.5} />
        <Arrow x1={208} y1={32 + (GRID * cell) / 2} x2={238} y2={32 + (GRID * cell) / 2} kind="memory" />
        <Chip x={240} y={32 + (GRID * cell) / 2 - 11} text="VAE" kind="memory" h={22} w={36} size={9.5} />
        <Arrow x1={280} y1={32 + (GRID * cell) / 2} x2={310} y2={32 + (GRID * cell) / 2} kind="memory" />
        <rect x={312} y={32} width={GRID * cell} height={GRID * cell} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} />
        <Label x={312 + GRID * cell / 2} y={32 + GRID * cell + 15} anchor="middle" tone={3} size={8.5}>{size}×{size}</Label>

        <Fade show={st >= 2}>
          <g>
            <Label x={392} y={32} tone={2} size={9.5} mono={false}>The loop, step by step</Label>
            {trade.slice(0, 6).map((r, i) => (
              <g key={r.steps}>
                <Bar x={392} y={36 + i * 20} w={Math.min(150, (r.s / (trade[trade.length - 1].s || 1)) * 150)} h={11}
                  frac={r.s / (trade[trade.length - 1].s || 1)} kind={r.steps === m.steps ? 'compute' : 'memory'} ghost={r.steps !== m.steps} />
                <Label x={628} y={45 + i * 20} anchor="end" tone={r.steps === m.steps ? 1 : 3} size={8.5}>{`${r.steps} steps · ${r.s.toFixed(1)}s`}</Label>
              </g>
            ))}
          </g>
        </Fade>

        {/* ── panel B: the roofline ── */}
        <Fade show={st >= 3}>
          <g>
            <Label x={12} y={106} tone={2} size={11} bold>How much of the memory system does a step use?</Label>
            <Label x={12} y={126} tone={3} size={9.5} mono={false}>time to move the step's bytes at full bandwidth</Label>
            <Bar x={230} y={118} w={220} h={13} frac={0.02} kind="memory" />
            <Label x={456} y={129} tone={2} size={9.5}>{`${(ds.memFloorS * 1e3).toFixed(1)} ms`}</Label>
            <Label x={12} y={146} tone={3} size={9.5} mono={false}>the actual step</Label>
            <Bar x={230} y={138} w={220} h={13} frac={Math.min(1, 0.02 / Math.max(0.02, ds.bandwidthUtil))} kind="compute" />
            <Label x={456} y={149} tone={2} size={9.5}>{`${ds.stepMs.toFixed(0)} ms`}</Label>
            <Label x={12} y={166} size={9.5} kind={ds.bound === 'compute' ? 'token' : 'memory'}>{`${(ds.bandwidthUtil * 100).toFixed(1)}% of peak bandwidth — the step is ${ds.bound}-bound.`}</Label>
          </g>
        </Fade>

        {/* ── panel C: the contrast ── */}
        <Fade show={st >= 4}>
          <g>
            <Label x={12} y={190} tone={2} size={11} bold>Same GPU, opposite conclusions</Label>
            <Label x={12} y={208} tone={3} size={9.5} mono={false}>LLM decode → 60-80% of bandwidth → buy bandwidth</Label>
            <Label x={12} y={226} tone={3} size={9.5} mono={false}>diffusion step → {`${(ds.bandwidthUtil * 100).toFixed(0)}%`} of bandwidth → buy FLOP/s</Label>
            <Label x={12} y={246} size={9.5} kind="compute">{`${m.steps} steps × ${ds.stepMs.toFixed(0)} ms = ${(ds.stepMs * m.steps).toFixed(0)} ms, plus ${(ds.vaeS * 1e3).toFixed(0)} ms of VAE.`}</Label>
          </g>
        </Fade>

        {/* ── panel D: guidance ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={12} y={268} tone={2} size={11} bold>Classifier-free guidance doubles the batch</Label>
            <Label x={12} y={288} tone={3} size={9.5} mono={false}>one forward pass</Label>
            <Bar x={200} y={280} w={180} h={12} frac={0.5} kind="memory" />
            <Label x={388} y={290} tone={2} size={9.5}>{`${(ds.stepMs).toFixed(0)} ms`}</Label>
            <Label x={12} y={306} tone={3} size={9.5} mono={false}>two: prompt + empty prompt</Label>
            <Bar x={200} y={298} w={180} h={12} frac={1} kind="compute" />
            <Label x={388} y={308} tone={2} size={9.5}>{`${(ds.stepMs * 2).toFixed(0)} ms`}</Label>
            <Label x={420} y={290} size={9.5} kind="error">+100%</Label>
            <Label x={420} y={308} tone={3} size={9}>standard practice</Label>
          </g>
        </Fade>

        {/* ── panel E: memory ── */}
        <Fade show={st >= 6}>
          <g>
            <Label x={440} y={322} tone={3} size={9} mono={false}>{`activations ${fmt.bytes(db.actBytes)} vs weights ${fmt.bytes(db.weightBytes)}`}</Label>
            <Label x={440} y={336} size={9} kind={db.fits ? 'text' : 'error'} mono={false}>{db.fits ? `fits; max batch ${db.maxBatch}` : `OOM at batch ${batch}`}</Label>
          </g>
        </Fade>

        <Fade show={st >= 7}>
          <g>
            <Chip x={12} y={320} text="fewer steps" kind="compute" h={18} w={82} size={9} />
            <Chip x={100} y={320} text="smaller model" kind="compute" h={18} w={84} size={9} />
            <Chip x={190} y={320} text={cache.label.toLowerCase()} kind="token" h={18} w={96} size={9} />
            <Label x={294} y={333} tone={3} size={9.5}>{`${(cache.retain * 100).toFixed(0)}% of step compute retained`}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}