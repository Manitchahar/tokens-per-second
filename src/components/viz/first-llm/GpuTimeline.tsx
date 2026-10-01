import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Move, Track, Ruler, fmt } from '../lib/svg';
import { modelById, gpuById, decodeStep, prefillTime, kvBytesPerToken } from '../lib/perfModel';

// What the GPU actually does while `generate` runs: a burst of compute for the prompt,
// then a long train of tiny memory-bound steps with Python-sized gaps between them.

const STEPS = [
  'One request on the timeline',
  'Prefill: a burst of compute',
  'The KV cache starts growing',
  'Decode: small steps, one token',
  'The gap between tokens',
  'Utilization is not the whole story',
  'Memory climbs, it does not fall',
  'Time to first token',
  'Read your own GPU',
];
const CAPTIONS = [
  'One generate() call drawn to scale on a real axis. It is not a smooth ribbon of work — it is a fast burst, then a train of small steps, then nothing at all.',
  'Prefill processes the whole prompt in one pass. Because it does hundreds of rows of work at once, the GPU is genuinely busy: this is the compute-bound phase.',
  'Prefill fills the KV cache. From here on every step reads it, and it never shrinks during the request — it only grows by one token per step.',
  'Each decode step produces one token. The work per step is tiny, so each step is short and memory-bound. Drag the prompt length to see prefill scale with it while decode steps do not.',
  'Between steps the GPU is often idle, waiting for Python: sampling, detokenizing, and the host-side loop. Those hatched gaps are real time your user is waiting.',
  'GPU utilization is a percentage of time any kernel was running. High utilization with low throughput still means poor efficiency — always pair it with tokens per second.',
  'Watch the memory trace: it steps up once per generated token and never comes back down until the request ends. That stair-step is the KV cache, and it is why long answers crash small GPUs.',
  'Time to first token is the wait before token 1 appears: queueing plus prefill. Everything after that is the decode rate. The two have different causes and different fixes.',
  'The habit: run nvidia-smi in a second terminal while your model generates. Seeing utilization fall to zero between tokens is how you learn where the bottleneck is.',
];

const M = modelById('qwen05');
const G = gpuById('t4');

const TL = { x: 66, w: 500, y: 150, h: 54 };
const memY = 254;
const memH = 16;

export default function GpuTimeline() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [prompt, setPrompt] = useState(128);
  const [maxNew, setMaxNew] = useState(24);

  const pf = prefillTime({ model: M, gpu: G, tokens: prompt });
  const step = decodeStep({ model: M, gpu: G, batch: 1, ctx: prompt + maxNew / 2 });
  const stepT = step.t;
  const overhead = step.overhead;
  const busy = stepT - overhead;

  const total = pf + stepT * maxNew;
  const msToPx = TL.w / Math.max(total, pf + stepT * maxNew);

  const pfW = pf * msToPx;
  const stepW = Math.max(1.5, stepT * msToPx);

  // The memory stair-step: weights, then +kv per token.
  const wFrac = step.weights / G.mem;
  const kvPx = (step.kvPerSeq / G.mem) * TL.w * msToPx * stepT;

  const ttftMs = pf * 1e3;
  const perTokMs = stepT * 1e3;
  const util = (busy / stepT) * 100;

  return (
    <Stage
      rootRef={ref}
      title="What the GPU does during one generate() call"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['idle', 'host-side gap'], 'token', ['error', 'would not fit']]}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'Time to first token', value: `${ttftMs.toFixed(0)} ms`, kind: 'compute', note: `prefill of ${fmt.int(prompt)} tokens` },
            { label: 'Time per output token', value: `${perTokMs.toFixed(1)} ms`, kind: 'memory', note: `≈ ${(1000 / perTokMs).toFixed(0)} tokens/s` },
            { label: 'GPU busy during a step', value: `${util.toFixed(0)}%`, kind: util > 70 ? 'token' : 'error', note: util > 70 ? 'close to the limit' : 'mostly waiting on Python' },
            { label: 'Memory, end of request', value: fmt.gb(step.weights + maxNew * kvBytesPerToken(M)), kind: 'memory', note: `of ${fmt.gb(G.mem)} on a T4` },
          ]}
        />
      }
      controls={
        <>
          <Slider label="Prompt length" value={prompt} min={8} max={2048} log onChange={setPrompt} format={(v) => `${fmt.int(v)} tokens`} kind="compute" />
          <Slider label="Tokens generated" value={maxNew} min={4} max={512} log onChange={setMaxNew} format={(v) => `${fmt.int(v)}`} kind="token" />
        </>
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A timeline of one text generation request: a short compute-heavy prefill burst, followed by many small memory-bound decode steps separated by idle gaps, with a memory trace below that steps upward once per generated token.">
        {/* ── the prompt header */}
        <Label x={56} y={28} tone={1} size={11} bold mono={false}>One request</Label>
        <text x={126} y={28} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>
          {`${fmt.int(prompt)} prompt tokens  →  ${fmt.int(maxNew)} generated tokens`}
        </text>

        {/* ── time ruler */}
        <Ruler
          x={TL.x}
          y={TL.y - 24}
          w={TL.w}
          label={st >= 1 ? `${fmt.ms(total)} total` : undefined}
          ticks={[
            { at: 0, label: '0' },
            { at: 0.25, label: fmt.ms(total * 0.25) },
            { at: 0.5, label: fmt.ms(total * 0.5) },
            { at: 0.75, label: fmt.ms(total * 0.75) },
            { at: 1, label: fmt.ms(total) },
          ]}
        />

        {/* ── prefill ─ */}
        <g className="v-fade" style={{ opacity: st >= 1 ? 1 : 0.2 }}>
          <rect x={TL.x} y={TL.y} width={Math.max(2, pfW)} height={TL.h} rx={2}
            fill="var(--compute)" stroke="var(--compute)" strokeWidth={1}
            style={{ transition: 'width .6s cubic-bezier(.16,1,.3,1)' }} className="v-cell" />
          {pfW > 74 && (
            <text x={TL.x + pfW / 2} y={TL.y + 22} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: '#0e1116' }}>
              prefill
            </text>
          )}
          {pfW > 74 && (
            <text x={TL.x + pfW / 2} y={TL.y + 37} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: '#0e1116' }}>
              {fmt.ms(pf)} · compute-bound
            </text>
          )}
          {st >= 7 && (
            <>
              <line x1={TL.x} x2={TL.x + pfW} y1={TL.y - 8} y2={TL.y - 8} stroke="var(--compute)" strokeWidth={2.5} />
              <text x={TL.x + pfW / 2} y={TL.y - 12} textAnchor="middle" className="v-label bold" style={{ fontSize: 9, fill: 'var(--compute)' }}>
                {`TTFT ${ttftMs.toFixed(0)} ms`}
              </text>
            </>
          )}
        </g>

        {/* ── decode steps ─ */}
        {st >= 3 &&
          Array.from({ length: Math.min(maxNew, 60) }, (_, i) => {
            const x = TL.x + pfW + i * stepW;
            if (x > TL.x + TL.w) return null;
            const bW = Math.max(1, stepW * (busy / stepT));
            return (
              <g key={i} className="v-fade">
                {/* the host-side gap: the part of the step that is not the GPU */}
                {st >= 4 && stepW > 6 && (
                  <rect x={x} y={TL.y} width={Math.max(0.6, stepW - bW)} height={TL.h}
                    fill="url(#tps-hatch)" opacity={0.85} />
                )}
                <rect x={x} y={TL.y} width={bW} height={TL.h}
                  fill="var(--memory)" stroke="var(--memory)" strokeWidth={0.6} className="v-cell" />
              </g>
            );
          })}

        {st >= 3 && (
          <Label x={TL.x + pfW + Math.min(maxNew, 60) * stepW / 2} y={TL.y + TL.h + 18} anchor="middle" size={9.5} kind="memory">
            {`${fmt.int(maxNew)} decode steps · ${perTokMs.toFixed(1)} ms each`}
          </Label>
        )}

        <line x1={TL.x} x2={TL.x + TL.w} y1={TL.y + TL.h + 4} y2={TL.y + TL.h + 4} className="v-rule" />
        <Track x={TL.x} y={TL.y} w={TL.w} h={TL.h} label="GPU" />

        {/* ── utilization read-out for the decode phase */}
        {st >= 5 && (
          <g className="v-fade">
            <rect x={TL.x + pfW} y={TL.y + TL.h + 26} width={Math.max(1, Math.min(maxNew, 60) * stepW)} height={5} rx={2}
              fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={0.8} />
            <rect x={TL.x + pfW} y={TL.y + TL.h + 26} width={Math.max(1, (Math.min(maxNew, 60) * stepW * util) / 100)} height={5} rx={2}
              fill={util > 70 ? 'var(--token)' : 'var(--error)'} className="v-cell" />
            <text x={TL.x + pfW + Math.min(maxNew, 60) * stepW + 8} y={TL.y + TL.h + 32} className="v-label"
              style={{ fontSize: 9.5, fill: util > 70 ? 'var(--token)' : 'var(--error)' }}>
              {`${util.toFixed(0)}% busy`}
            </text>
          </g>
        )}

        {/* ── memory trace ─ */}
        <line x1={56} y1={memY - 22} x2={584} y2={memY - 22} className="v-rule faint" />
        <Label x={56} y={memY - 8} tone={1} size={10.5} bold mono={false}>GPU memory</Label>
        <Label x={584} y={memY - 8} anchor="end" tone={3} size={9}>{fmt.gb(G.mem)} on a T4</Label>

        {/* capacity */}
        <rect x={TL.x} y={memY} width={TL.w} height={memH} rx={2} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1} />
        {/* weights, constant */}
        <rect x={TL.x} y={memY} width={Math.max(1, wFrac * TL.w)} height={memH} fill="var(--memory)" className="v-cell" />
        <text x={TL.x + 6} y={memY + 12} className="v-label bold" style={{ fontSize: 9.5, fill: '#0e1116' }}>
          {`weights ${fmt.gb(step.weights)}`}
        </text>

        {/* the KV stair-step: memory climbs with every generated token, over the decode stretch */}
        {st >= 2 && (() => {
          const n = Math.min(maxNew, 40);
          const genW = TL.w - pfW;
          const topY = memY - Math.min(n * kvPx, memH - 4);
          const pts: string[] = [];
          for (let i = 0; i <= n; i++) {
            const x = TL.x + pfW + (i / n) * genW;
            const y = memY - Math.min(i * kvPx, memH - 4);
            pts.push(`${i === 0 ? 'L' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
          }
          return (
            <g className="v-fade">
              <path d={`M${TL.x + pfW},${memY} ${pts.join(' ')} L${(TL.x + pfW + genW).toFixed(1)},${memY} Z`}
                fill="var(--memory-dim)" stroke="none" />
              <path d={pts.join(' ')} fill="none" stroke="var(--memory)" strokeWidth={1.4} />
              <Label x={TL.x + pfW + genW - 4} y={topY - 6} anchor="end" size={9} kind="memory">
                {`+${fmt.bytes(kvBytesPerToken(M))} per token`}
              </Label>
              <Label x={TL.x + pfW + 6} y={memY + 28} size={9} kind="memory">KV cache</Label>
            </g>
          );
        })()}

        {st >= 6 && (
          <Label x={TL.x} y={memY + memH + 30} tone={2} size={9.5} mono={false}>
            {`After ${fmt.int(maxNew)} tokens: ${fmt.gb(step.weights + maxNew * kvBytesPerToken(M))} of ${fmt.gb(G.mem)}. Nothing is freed until the request ends.`}
          </Label>
        )}
      </svg>
    </Stage>
  );
}
