import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Chip, Arrow, Fade, Dim, Bar, fmt } from '../lib/svg';
import {
  VISION_MODELS, visionById, patchesPerSide, imageTokens, visionPrefill, visionBatch, projectorParams, gpu, B,
} from './workloads';

const STEPS = [
  'An image arrives as pixels, not tokens',
  'The vision tower cuts it into patches',
  'The projector makes patches into tokens',
  'Now it is just a long prompt',
  'Where the FLOPs actually go',
  'A bigger image costs more tokens, not more compute per token',
  'Batching images is a different bottleneck',
  'Sizing an image-heavy service',
];
const CAPTIONS = [
  'A chat message with an image in it reaches the serving engine as pixels. Nothing about it is a token yet. The engine has to turn an image into a sequence before any language model can read it.',
  'A vision transformer slices the image into fixed-size patches and runs a transformer over them. At 336 pixels with 14-pixel patches, an image is 24×24 = 576 patches — each patch is a little bag of pixels, and each becomes a vector.',
  'A small projector maps each patch vector into the language model’s hidden width. This is where image features turn into something that lives alongside text embeddings. It is two linear layers, and it is tiny compared with either model.',
  'Now prefill runs over text tokens plus image tokens, exactly as it would over text alone. This is the whole insight: multimodal inference is long-prompt prefill with an extra model bolted on the front.',
  'The vision tower is compute-bound and small; the language model prefill dominates the time. The projector is negligible. Knowing which part is expensive is what tells you where to optimise.',
  'Doubling the image resolution does not double the FLOPs per token — it quadruples the patches, which the projector pools back down. Resolution mostly buys you tokens, and tokens buy you prefill time and KV cache.',
  'With images in the batch, the vision tower runs once per image and cannot be batched away. That is a second, independent bottleneck sitting in front of the language model, and it needs its own capacity plan.',
  'Two levers matter: tokens per image (resolution and pooling) and images per second. Both scale linearly with batch, but the tower’s memory footprint is what will run you out of VRAM first.',
];

export default function Multimodal() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [mid, setMid] = useState<string>('llava15');
  const [batch, setBatch] = useState(8);
  const [imgs, setImgs] = useState(1);
  const [gId, setGId] = useState<string>('h100');

  const m = visionById(mid);
  const g = gpu(gId);
  const side = patchesPerSide(m);
  const imgTok = imageTokens(m);
  const pf = useMemo(() => visionPrefill(m, g, 200, imgs), [m, g, imgs]);
  const vb = useMemo(() => visionBatch(m, g, batch, imgs), [m, g, batch, imgs]);
  const proj = projectorParams(m, m.hidden);

  // A patch grid, drawn small. 24x24 is too many cells to draw; show a coarse sample.
  const GRID = 12;
  const cell = 3.6;

  return (
    <Stage
      rootRef={ref}
      title="Multimodal: an image is a long prompt"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'image patches'],
        ['compute', 'the two models’ math'],
        ['token', 'tokens fed to the language model'],
        ['comm', 'projector'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Image tokens', value: fmt.int(imgTok), kind: 'token', note: `${side}×${side} patches at ${m.imageSize}px` },
          { label: 'Text tokens', value: fmt.int(200 * imgs), kind: 'text', note: 'the question, per image' },
          { label: 'Vision tower', value: `${pf.visionMs.toFixed(1)} ms`, kind: 'compute', note: `${fmt.int(pf.patches)} patches` },
          { label: 'Prefill total', value: `${pf.totalMs.toFixed(0)} ms`, kind: 'memory', note: `image is ${(pf.imageFrac * 100).toFixed(0)}% of the sequence` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={mid} options={VISION_MODELS.map((x) => ({ value: x.id, label: x.name.replace('LLaVA-', '') }))} onChange={setMid} />
          <Segmented label="GPU" value={gId} options={[{ value: 'a100', label: 'A100' }, { value: 'h100', label: 'H100' }, { value: 'l4', label: 'L4' }]} onChange={setGId} />
          <Slider label="Requests per step" kind="token" min={1} max={64} step={1} value={batch} onChange={setBatch} format={(v) => `${v} requests`} />
          <Slider label="Images per request" kind="memory" min={1} max={4} step={1} value={imgs} onChange={setImgs} format={(v) => `${v} image${v > 1 ? 's' : ''}`} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="An image sliced into patches, passed through a vision tower and a small projector into a sequence of tokens, with the time split between the vision tower and the language model prefill.">
        {/* ── panel A: the image and its patches ── */}
        <Label x={12} y={20} tone={2} size={11} bold>{st === 0 ? 'The request: pixels' : 'The image, sliced into patches'}</Label>
        {/* Image frame + patch grid */}
        <g transform={`translate(12,28)`}>
          <rect x={0} y={0} width={GRID * cell} height={GRID * cell} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} />
          {Array.from({ length: GRID }).map((_, r) =>
            Array.from({ length: GRID }).map((_, c) => (
              <rect key={`${r}-${c}`} x={c * cell + 0.4} y={r * cell + 0.4} width={cell - 0.8} height={cell - 0.8}
                fill="var(--memory)" opacity={0.35 + 0.5 * ((r * 7 + c * 3) % 5) / 5} className="v-cell" />
            )),
          )}
          <Label x={GRID * cell / 2} y={GRID * cell + 16} anchor="middle" tone={3} size={8.5}>{m.imageSize}×{m.imageSize}px</Label>
        </g>
        {/* arrow to patches */}
        <Arrow x1={12 + GRID * cell + 6} y1={28 + (GRID * cell) / 2} x2={80} y2={28 + (GRID * cell) / 2} kind="memory" />
        {/* vision tower */}
        <Fade show={st >= 1}>
          <g>
            <Chip x={82} y={28 + (GRID * cell) / 2 - 11} text="ViT" kind="compute" h={22} w={34} size={9.5} />
            <Label x={82} y={28 + (GRID * cell) / 2 + 26} tone={3} size={8.5}>{side}×{side}={side * side} patches</Label>
          </g>
        </Fade>
        <Fade show={st >= 2}>
          <g>
            <Arrow x1={120} y1={28 + (GRID * cell) / 2} x2={150} y2={28 + (GRID * cell) / 2} kind="comm" />
            <Chip x={152} y={28 + (GRID * cell) / 2 - 11} text="proj" kind="comm" h={22} w={40} size={9.5} />
          </g>
        </Fade>
        {/* the token strip */}
        <Fade show={st >= 3}>
          <g>
            <Arrow x1={196} y1={28 + (GRID * cell) / 2} x2={226} y2={28 + (GRID * cell) / 2} kind="token" />
            {/* a run of image tokens */}
            {Array.from({ length: 22 }).map((_, i) => (
              <rect key={i} x={228 + i * 5.4} y={28 + (GRID * cell) / 2 - 8} width={4} height={16} rx={1}
                fill="var(--token)" opacity={0.85} className="v-cell" />
            ))}
            <Label x={228} y={28 + (GRID * cell) / 2 + 26} tone={3} size={8.5}>{fmt.int(imgTok)} image tokens</Label>
          </g>
        </Fade>

        {/* ── panel B: the sequence ── */}
        <Fade show={st >= 3}>
          <g>
            <Label x={12} y={112} tone={2} size={11} bold>The prefill sequence</Label>
            <Label x={12} y={130} tone={3} size={9.5} mono={false}>text</Label>
            <Span x={12} y={136} w={Math.max(20, (200 * imgs / (200 * imgs + imgTok * imgs)) * 300)} h={16} kind="text" label={`${200 * imgs} text`} labelTone="light" />
            <Label x={200} y={130} tone={3} size={9.5} mono={false}>image</Label>
            <Span x={200} y={136} w={(imgTok * imgs / (200 * imgs + imgTok * imgs)) * 300} h={16} kind="token" label={`${imgTok * imgs} image`} labelTone="dark" />
            <Dim x1={12} x2={488} y={164} label={`${fmt.int(200 * imgs + imgTok * imgs)} tokens prefilled`} kind="token" />
            <Label x={12} y={182} tone={3} size={9.5}>{`Image tokens are ${(pf.imageFrac * 100).toFixed(0)}% of the sequence.`}</Label>
          </g>
        </Fade>

        {/* ── panel C: where the time goes ── */}
        <Fade show={st >= 4}>
          <g>
            <Label x={12} y={210} tone={2} size={11} bold>Where the prefill time goes ({g.name})</Label>
            <Label x={12} y={230} tone={3} size={9.5} mono={false}>vision tower ({fmt.int(pf.patches)} patches)</Label>
            <Bar x={220} y={222} w={260} h={13} frac={pf.visionMs / pf.totalMs} kind="compute" />
            <Label x={486} y={233} tone={1} size={9.5}>{`${pf.visionMs.toFixed(1)} ms`}</Label>
            <Label x={12} y={250} tone={3} size={9.5} mono={false}>language model prefill</Label>
            <Bar x={220} y={242} w={260} h={13} frac={pf.textMs / pf.totalMs} kind="token" />
            <Label x={486} y={253} tone={1} size={9.5}>{`${pf.textMs.toFixed(0)} ms`}</Label>
            <Label x={12} y={272} tone={3} size={9.5} mono={false}>projector (once, negligible)</Label>
            <Bar x={220} y={264} w={260} h={13} frac={0.01} kind="comm" />
            <Label x={486} y={275} tone={3} size={9.5}>{B(proj)} params</Label>
          </g>
        </Fade>

        {/* ── panel D: the batch ── */}
        <Fade show={st >= 6}>
          <g>
            <Label x={12} y={296} tone={2} size={9.5}>{`Batch ${batch} × ${imgs} image${imgs > 1 ? 's' : ''}: ${fmt.int(vb.towerCalls)} vision passes this step, ${vb.visionMs.toFixed(0)} ms tower vs ${vb.decodeMs.toFixed(1)} ms decode`}</Label>
            <Label x={12} y={312} size={9.5} kind={vb.visionShare > 0.5 ? 'error' : 'token'}>{`The vision tower is ${(vb.visionShare * 100).toFixed(0)}% of the step — a second bottleneck.`}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}