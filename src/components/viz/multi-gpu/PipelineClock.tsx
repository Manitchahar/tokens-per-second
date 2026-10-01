import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';
import { bmodelById, gpuById, linkById, bubbleFraction, ppBoundaryBytes, ppBoundaryTime } from './parallel';

const STEPS = [
  'Four stages, one request',
  'The warm-up, one stage at a time',
  'The bubble',
  'Fill it with micro-batches',
  'Why more stages is worse',
  'The one activation per boundary',
  'Communication is not the problem',
  'Prefill and decode both pay it',
];

const CAPTIONS = [
  'Four stages, four GPUs, ten transformer layers each. A request enters stage 0 and walks the pipeline one stage at a time. Nothing is split inside a layer, so no GPU does half a matrix multiply.',
  'Stage 0 works for one unit of time. Then stage 1 works while stage 0 waits for the next unit. This ramp-up is structural: a stage cannot start until the previous one has produced something.',
  'The ramp-up at the front and the drain at the back are the bubble. With P stages and M units of work the busy fraction is M/(M+P-1); the rest is idle.',
  'Put four requests on the pipeline instead of one. While stage 0 handles request 2, stage 1 handles request 1. The pipeline fills, and the bubble shrinks toward nothing as M grows.',
  'The bubble fraction is (P-1)/(M+P-1). Doubling M halves it. Doubling P makes it worse. This is why you add micro-batches and not stages.',
  'Each stage sends exactly one activation tensor to the next: B × T × hidden × 2 bytes. For Llama-3.1-70B at batch 32 that is 512 KB — and it happens P-1 times, not L times.',
  'Over NVLink this is free. Over InfiniBand it is about 11 µs, and at batch 32 decode that is 2.5% of the step. Communication is affordable. The bubble is the bill.',
  'Pipelining splits the batch into micro-batches but does not change the total work — a pipeline at batch 32 does exactly the work of a single GPU at batch 32. This is inference, so there is no backward pass to fill the bubble.',
];

const P = 4;

export default function PipelineClock() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [micro, setMicro] = useState(2);
  const [modelId, setModelId] = useState('llama70b');
  const [linkId, setLinkId] = useState('ib');
  const [batch, setBatch] = useState(32);
  const [layers, setLayers] = useState(80);

  const model = bmodelById(modelId);
  const gpu = gpuById('h100');
  const link = linkById(linkId);

  const bubble = bubbleFraction(P, micro);
  const bubbleT = useTween(bubble, 450);

  // total slots in the schedule
  const slots = micro + P - 1;
  const CELL = 30;
  const X0 = 68;
  const Y0 = 76;

  // per-stage compute time, relative units (1 unit = one stage's work on one micro-batch)
  const stageBytes = ppBoundaryBytes(model, batch / micro, 1);
  const tComm = ppBoundaryTime(model, batch / micro, 1, link).valueOf();
  const tStage = (weightBytesTotal(model, layers) / P) / (gpu.bw * 0.8);
  const commFrac = tComm / tStage;

  const gridW = X0 + slots * CELL + 46;

  return (
    <Stage
      rootRef={ref}
      title="Pipeline parallelism: a clock with a hole in it"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'a stage working'],
        ['comm', 'the activation crossing'],
        ['idle', 'the bubble'],
        ['token', 'a finished request'],
        'text',
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Stages', value: `${P}`, kind: 'text', note: `${layers} layers → ${Math.floor(layers / P)} each` },
          { label: 'Micro-batches', value: `${micro}`, kind: 'compute', note: `schedule of ${slots} slots` },
          { label: 'Bubble', value: `${(bubbleT * 100).toFixed(0)}%`, kind: bubbleT > 0.2 ? 'error' : 'token', note: `(P-1)/(M+P-1)` },
          { label: 'Boundary transfer', value: `${(tComm * 1e6).toFixed(1)} µs`, kind: 'comm', note: `${fmt.bytes(stageBytes)} over ${link.name}` },
        ]} />
      }
      controls={
        <>
          <Slider label="Micro-batches M" kind="compute" value={micro} min={1} max={16} onChange={setMicro} format={(v) => `${v}`} />
          <Slider label="Model layers" kind="memory" value={layers} min={8} max={128} onChange={setLayers} format={(v) => `${v}`} />
          <Slider label="Total batch" kind="memory" value={batch} min={1} max={512} log onChange={setBatch} format={(v) => `${fmt.int(v)}`} />
          <Segmented label="Model" value={modelId} options={[{ value: 'llama8b', label: 'L3.1-8B' }, { value: 'llama70b', label: 'L3.1-70B' }]} onChange={setModelId} />
          <Segmented label="Boundary link" value={linkId} options={[{ value: 'nvlink4', label: 'NVLink 4' }, { value: 'ib', label: 'InfiniBand' }]} onChange={setLinkId} />
        </>
      }
    >
      <svg viewBox="0 0 640 384" role="img" aria-label="A schedule grid with four pipeline stages as rows and time as columns. Work cells run diagonally down the grid, with hatched idle cells at the top-left forming the pipeline bubble and at the bottom-right forming the drain. A bar underneath shows the bubble fraction shrinking as micro-batches are added.">
        <Label x={0} y={22} tone={2} size={11} bold>{`${P} stages · ${micro} micro-batch${micro > 1 ? 'es' : ''} · ${slots} time slots`}</Label>
        <Label x={604} y={22} anchor="end" tone={3} size={9.5}>{`${(bubble * 100).toFixed(0)}% idle`}</Label>

        {/* stage labels */}
        {Array.from({ length: P }, (_, i) => (
          <Label key={i} x={X0 - 12} y={Y0 + i * 22 + 14} anchor="end" size={9.5} tone={3}>{`stage ${i}`}</Label>
        ))}

        {/* the grid */}
        {Array.from({ length: micro }, (_, mb) =>
          Array.from({ length: P }, (_, st) => {
            const slot = mb + st;
            if (slot >= slots) return null;
            const x = X0 + slot * CELL;
            const y = Y0 + st * 22;
            return (
              <g key={`${mb}-${st}`}>
                <rect x={x} y={y} width={CELL - 3} height={18} rx={2} fill="var(--compute)" opacity={0.9} className="v-cell" />
                <Label x={x + (CELL - 3) / 2} y={y + 12} anchor="middle" size={8} >{`m${mb}`}</Label>
                {st < P - 1 && <Arrow x1={x + (CELL - 3) / 2} y1={y + 19} x2={x + (CELL - 3) / 2 + CELL - 3} y2={y + 22} kind="comm" width={1} />}
              </g>
            );
          }),
        )}

        {/* the bubble: idle cells */}
        {Array.from({ length: P }, (_, st) =>
          Array.from({ length: slots }, (_, slot) => {
            const busy = slot >= st && slot < st + micro;
            if (busy) return null;
            const x = X0 + slot * CELL;
            const y = Y0 + st * 22;
            return (
              <rect key={`i-${st}-${slot}`} x={x} y={y} width={CELL - 3} height={18} rx={2}
                fill="url(#tps-hatch)" stroke="var(--idle)" strokeWidth={0.8} opacity={0.7} />
            );
          }),
        )}

        <Fade show={s.step >= 2}>
          <Label x={X0 - 12} y={Y0 - 6} anchor="end" size={9} kind="error">{`${Math.round(bubble * P * 10) / 10} stage-slots`}</Label>
          <Arrow x1={X0 + 6} y1={Y0 + 4} x2={X0 + 6 + Math.max(0, (micro - 1) * CELL)} y2={Y0 + 4} kind="error" dashed />
          <Label x={X0 + Math.max(0, (micro - 1) * CELL) + 12} y={Y0 + 7} size={9} kind="error">{`the bubble: ${(bubble * 100).toFixed(0)}%`}</Label>
        </Fade>

        {/* the bubble bar */}
        <Label x={0} y={188} tone={2} size={11} bold>Idle fraction, (P-1)/(M+P-1)</Label>
        {(() => {
          const W = 420;
          const rows = [1, 2, 4, 6, 8, 12, 16];
          return (
            <g>
              <rect x={40} y={196} width={W} height={112} rx={2} fill="none" stroke="var(--rule)" />
              {rows.map((m, i) => {
                const b = bubbleFraction(P, m);
                const y = 202 + i * 15;
                const sel = m === micro;
                return (
                  <g key={m}>
                    <Label x={34} y={y + 8} anchor="end" size={8.5} tone={3}>{`M=${m}`}</Label>
                    <rect x={40} y={y} width={W} height={11} rx={2} fill="none" stroke="var(--rule-2)" strokeWidth={0.6} />
                    <Span x={40} y={y} w={W * (1 - b)} h={11} kind="compute" opacity={sel ? 0.95 : 0.35} />
                    <Span x={40 + W * (1 - b)} y={y} w={W * b} h={11} kind="idle" opacity={sel ? 0.9 : 0.4} />
                    <Label x={40 + W + 8} y={y + 8.5} size={9} kind={b > 0.2 ? 'error' : 'token'}>{`${(b * 100).toFixed(0)}% idle`}</Label>
                  </g>
                );
              })}
            </g>
          );
        })()}

        <Fade show={s.step >= 5}>
          <Label x={40} y={330} tone={2} size={11} bold>What crosses each boundary</Label>
          <Span x={40} y={338} w={190} h={14} kind="comm" opacity={0.85} />
          <Label x={238} y={349} size={10} kind="comm">{`${fmt.bytes(ppBoundaryBytes(model, batch / micro, 1))} per boundary, P-1 = ${P - 1} of them`}</Label>
          <Label x={40} y={368} size={9} tone={3}>{`over ${link.name}: ${(tComm * 1e6).toFixed(1)} µs — ${(commFrac * 100).toFixed(1)}% of a stage's own compute`}</Label>
        </Fade>
        <Fade show={s.step < 5}>
          <Label x={40} y={340} size={9.5} tone={3}>{`Total work is unchanged: ${P} stages × ${micro} micro-batches = ${P * micro} stage-slots.`}</Label>
          <Label x={40} y={356} size={9.5} tone={3}>Pipelining changes when the work happens, never how much of it there is.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}

// weights bytes for a given layer count, so the slider is honest
function weightBytesTotal(model: ReturnType<typeof bmodelById>, layers: number) {
  const per = (model.params / model.layers) * 2;
  return per * layers;
}