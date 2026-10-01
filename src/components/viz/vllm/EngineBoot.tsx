import { useEffect, useRef, useState } from 'react';
import { Stage, Segmented, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Span, Label, Arrow, Fade, fmt } from '../lib/svg';
import { modelById, gpuById } from '../lib/perfModel';
import { launchPlan, KV_BLOCK_TOKENS, ANSWER_TOKENS } from './engine';

const MODELS = [
  { id: 'qwen05', label: 'Qwen2.5-0.5B' },
  { id: 'llama8b', label: 'Llama-3.1-8B' },
  { id: 'llama70b', label: 'Llama-3.1-70B' },
];

const STEPS = [
  'One command, one process',
  'Read the weights',
  'Warm up and capture the graph',
  'Buy a KV cache pool',
  'Ready: /health answers',
  'One request, streamed',
  'Many requests, one GPU',
];
const CAPTIONS = [
  'pip install vllm, then vllm serve <model>. That is the whole install. The process starts and imports PyTorch and CUDA; the GPU is still empty.',
  'The engine core reads the model’s safetensors files into GPU memory. This is disk-bound work, and it is why an 8B model takes tens of seconds to appear.',
  'vLLM runs the model once on a dummy batch to measure its peak activation memory, then captures the decode step as a CUDA graph. Only then does it allocate for real.',
  'Whatever is left of the memory budget becomes the KV cache pool, carved into equal 16-token blocks. That pool, not the model, is the server’s real capacity.',
  'The server answers GET /health. Only now is it worth sending real work; before this point the port is up but the engine cannot run.',
  'POST /v1/chat/completions goes in. The prompt is prefilled, and the answer comes back over a stream as server-sent events, one chunk per token.',
  'The next requests do not queue behind the first. The scheduler keeps them all running in the same iteration, so one read of the weights serves every one of them.',
];

const L = 96;
const R = 620;
const W = R - L;
const READY = 0.84; // where the process can finally serve

const BOOT_SPANS = [
  { from: 0.0, to: 0.34, label: 'imports + CUDA init', kind: 'idle' as const },
  { from: 0.34, to: 0.62, label: 'read weights', kind: 'memory' as const },
  { from: 0.62, to: READY, label: 'CUDA graphs', kind: 'compute' as const },
];

/** Startup time grows with the size of the checkpoint. Approximate, and mostly disk. */
const bootSeconds = (params: number) => 8 + (params / 1e9) * 3.1;

export default function EngineBoot() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [util, setUtil] = useState(0.9);
  const [shown, setShown] = useState(0);
  const [clients, setClients] = useState(1);

  useEffect(() => {
    setShown(s.step >= 6 ? 24 : s.step >= 5 ? 14 : 0);
    setClients(s.step >= 6 ? 9 : 1);
  }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const model = modelById(modelId);
  const gpu = gpuById('h100');
  const plan = launchPlan(model, gpu, util);
  const boot = bootSeconds(model.params);
  const px = (v: number) => (v / gpu.mem) * W;

  const t = useTweens({ w: px(plan.weights), p: px(plan.pool), r: px(plan.reserve), b: px(plan.budget) });

  const l1 = ANSWER_TOKENS.slice(0, Math.min(shown, 12)).join(' ');
  const l2 = ANSWER_TOKENS.slice(12, Math.min(shown, 24)).join(' ');
  const running = Math.min(3, clients);

  return (
    <Stage
      rootRef={ref}
      title="From vllm serve to a streamed answer"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', ['idle', 'starting / waiting'], ['error', 'out of memory']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Weights on the GPU', value: plan.weights > 1e9 ? `${(plan.weights / 1e9).toFixed(1)} GB` : `${(plan.weights / 1e6).toFixed(0)} MB`, kind: 'memory' },
          { label: 'KV cache pool', value: `${(plan.pool / 1e9).toFixed(1)} GB`, kind: 'memory', note: `${fmt.si(plan.poolTokens)} tokens of context` },
          { label: `Blocks of ${KV_BLOCK_TOKENS} tokens`, value: fmt.int(plan.blocks), kind: 'memory', note: 'carved once, at startup' },
          { label: 'Startup', value: `${Math.round(boot)} s`, kind: 'compute', note: 'mostly reading the checkpoint' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={MODELS} onChange={setModelId} />
          <Slider label="gpu_memory_utilization" value={util} min={0.5} max={0.99} step={0.01} onChange={setUtil} format={(v) => v.toFixed(2)} kind="memory" />
        </>
      }
    >
      <svg viewBox="0 0 640 344" role="img" aria-label="A timeline of starting a vLLM server: the process warms up, weights and a KV cache pool fill the GPU memory bar, the health check passes, then one request streams tokens back while nine requests share the same GPU steps.">
        <Label x={16} y={14} tone={2} size={10}>{`$ vllm serve ${model.name} --dtype float16 --max-model-len 8192`}</Label>

        {/* ── the startup timeline ── */}
        <text x={L - 10} y={46} textAnchor="end" className="v-track-label">startup</text>
        <line x1={L} x2={R} y1={36} y2={36} className="v-rule faint" />
        {s.step >= 1 ? (
          BOOT_SPANS.map((sp, i) => (
            <g key={sp.label} className="v-fade" style={{ opacity: i <= s.step - 1 ? 1 : 0 }}>
              <Span x={L + sp.from * W} y={30} w={(sp.to - sp.from) * W} h={16} kind={sp.kind} label={sp.label} labelTone={sp.kind === 'idle' ? 'light' : undefined} />
              <Label x={L + ((sp.from + sp.to) / 2) * W} y={60} anchor="middle" tone={3} size={9}>{`${Math.round((sp.to - sp.from) * boot)} s`}</Label>
            </g>
          ))
        ) : (
          <Span x={L} y={30} w={W} h={16} kind="idle" label="starting the process" labelTone="light" />
        )}
        <Fade show={s.step >= 4}>
          <line x1={L + READY * W} x2={L + READY * W} y1={24} y2={52} stroke="var(--token)" strokeWidth={1.6} />
          <Label x={L + READY * W + 6} y={40} size={9.5} kind="token" bold>{`ready ≈ ${Math.round(boot)} s`}</Label>
        </Fade>

        {/* ── what the memory budget became ── */}
        <text x={L - 10} y={100} textAnchor="end" className="v-track-label">gpu memory</text>
        <rect x={L} y={88} width={W} height={22} rx={2} fill="none" stroke="var(--rule)" />
        <Fade show={s.step >= 3}>
          <Span x={L} y={88} w={t.w} h={22} kind="memory" label="weights" />
          <Span x={L + t.w} y={88} w={Math.max(0, t.p)} h={22} kind="memory" dim label="KV pool" labelTone="light" />
          <Span x={L + t.w + Math.max(0, t.p)} y={88} w={Math.max(0, Math.min(t.r, W - t.w - Math.max(0, t.p)))} h={22} kind="idle" />
          <line x1={L + t.b} x2={L + t.b} y1={84} y2={114} stroke="var(--text-2)" strokeWidth={1.2} />
          <Label x={L + Math.min(W - 60, t.b + 5)} y={82} tone={3} size={9}>{`budget ${util.toFixed(2)} × ${(gpu.mem / 1e9).toFixed(0)} GB`}</Label>
        </Fade>
        {[0, 20, 40, 60, 80].map((v) => (
          <Label key={v} x={L + (v / 80) * W} y={124} anchor="middle" tone={3} size={9}>{v}</Label>
        ))}
        <Fade show={s.step >= 3 && !plan.fits}>
          <rect x={L} y={85} width={W} height={28} rx={3} fill="url(#tps-hatch-red)" stroke="var(--error)" />
          <Label x={L + W / 2} y={103} anchor="middle" size={11} kind="error" bold>
            {`does not fit: ${(plan.weights / 1e9).toFixed(0)} GB of weights, ${(plan.budget / 1e9).toFixed(0)} GB of budget`}
          </Label>
        </Fade>
        <Fade show={s.step >= 3 && plan.fits}>
          <Label x={L} y={140} tone={2} size={10}>
            {`${(plan.weights / 1e9).toFixed(1)} GB of weights + ${(plan.pool / 1e9).toFixed(1)} GB of KV pool = ${fmt.si(plan.poolTokens)} tokens of context`}
          </Label>
        </Fade>

        {/* ── the first real request ── */}
        <Fade show={s.step < 5 && plan.fits}>
          <rect x={L} y={178} width={W} height={138} rx={3} fill="none" stroke="var(--rule-2)" strokeDasharray="4 4" />
          <Label x={L + W / 2} y={240} anchor="middle" tone={3} size={10.5}>up, and idle: the pool is full of empty blocks, waiting for a request</Label>
        </Fade>
        <Fade show={s.step >= 5 && !plan.fits}>
          <rect x={L} y={178} width={W} height={138} rx={3} fill="url(#tps-hatch-red)" stroke="var(--error)" />
          <Label x={L + W / 2} y={240} anchor="middle" size={10.5} kind="error">no request can be served: smaller model, more GPUs, or quantize</Label>
        </Fade>
        <Fade show={s.step >= 5 && plan.fits}>
          <Label x={16} y={168} tone={2} size={10.5} mono={false} bold>Now it takes work</Label>
          <rect x={L} y={178} width={92} height={20} rx={3} fill="none" stroke="var(--token)" strokeDasharray="3 3" />
          <Label x={L + 46} y={192} anchor="middle" size={9.5} kind="token">GET /health</Label>
          <Arrow x1={L + 96} y1={188} x2={L + 114} y2={188} kind="token" />
          <Label x={L + 119} y={192} size={10} kind="token">200</Label>
          <rect x={L + 146} y={178} width={188} height={20} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <Label x={L + 154} y={192} size={9.5}>POST /v1/chat/completions</Label>
          <Arrow x1={L + 338} y1={188} x2={L + 360} y2={188} kind="compute" />
          <rect x={L + 364} y={172} width={88} height={32} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" />
          <Label x={L + 408} y={186} anchor="middle" size={10} kind="compute" bold>vLLM</Label>
          <Label x={L + 408} y={198} anchor="middle" tone={3} size={8.5}>engine core</Label>
          <Arrow x1={L + 456} y1={188} x2={L + 474} y2={188} kind="token" />
          <Label x={L + 480} y={192} size={9.5} kind="token">SSE stream</Label>
        </Fade>

        {/* ── the answer, arriving one token at a time ── */}
        <Fade show={s.step >= 5 && plan.fits}>
          <rect x={L} y={210} width={W} height={52} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
          {l1 ? (
            <text x={L + 10} y={229} className="v-label" style={{ fontSize: 10.5, fill: 'var(--token)' }}>
              {l1}
              {shown <= 12 ? <tspan className="v-pulse" fill="var(--token)">▍</tspan> : null}
            </text>
          ) : null}
          {l2 ? (
            <text x={L + 10} y={247} className="v-label" style={{ fontSize: 10.5, fill: 'var(--token)' }}>
              {l2}
              <tspan className="v-pulse" fill="var(--token)">▍</tspan>
            </text>
          ) : null}
          <Label x={L} y={278} tone={3} size={9.5}>{`${shown} tokens streamed, one \`data:\` line each`}</Label>
        </Fade>

        {/* ── and now, many requests ── */}
        <Fade show={s.step >= 6 && plan.fits}>
          <text x={L - 10} y={300} textAnchor="end" className="v-track-label">requests</text>
          {Array.from({ length: Math.max(1, clients) }, (_, i) => (
            <rect key={i} x={L + i * 13} y={292} width={9} height={12} rx={1.5}
              fill={i < running ? 'var(--token)' : 'url(#tps-hatch)'}
              stroke={i < running ? 'var(--token)' : 'var(--idle)'} />
          ))}
          <Label x={L + Math.max(1, clients) * 13 + 6} y={302} size={9.5} kind="token">{`${clients} in flight: one weight read serves all of them`}</Label>
          <Label x={L} y={324} tone={3} size={9}>{`GET /metrics → vllm:num_requests_running ${running}   vllm:num_requests_waiting ${Math.max(1, clients) - running}`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}