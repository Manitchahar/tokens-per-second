import { useEffect, useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Arrow, Fade, Dim, Chip, fmt } from '../lib/svg';
import { modelById, gpuById, decodeStep } from '../lib/perfModel';

/**
 * The token's path out of the engine and the three clocks that judge it.
 * Left: the SSE stream and where TTFT is actually spent. Right: the counters an
 * engine must export, since a metric nobody collected cannot be alerted on.
 */

const model = modelById('llama8b');
const gpu = gpuById('h100');

const STEPS = [
  'Three clocks, not one',
  'Where TTFT goes',
  'The stream, token by token',
  'SSE framing',
  'The detokenizer trap',
  'Decoupling the network',
  'Counters every iteration',
  'The four that matter',
];
const CAPTIONS = [
  'A request is judged by three numbers: time to first token, time per output token, and end-to-end. They are measured on a streamed response from a client-side clock, or you are measuring the wrong thing.',
  'TTFT is queue wait plus prefill plus one decode step. On a loaded server the queue term dominates, and it is invisible to a kernel profiler because no kernel is running while a request waits.',
  'The stream is one chunk per token. It matters that the first chunk leaves as soon as the first token exists, not when generation finishes — a non-streaming response has TTFT equal to end-to-end latency.',
  'Server-sent events: one `data:` line per chunk, terminated by `data: [DONE]`. The connection stays open for the whole generation.',
  'A token is often part of a word, part of a UTF-8 character, or a piece of a multi-byte sequence. Streaming token ids directly to a client produces mojibake unless a detokenizer buffers partial sequences.',
  'Pushing tokens straight to the socket couples the network to the GPU loop. A background queue absorbs a slow client so a 200 ms socket write never lands in a 7 ms step.',
  'Five counters per iteration cost almost nothing and become the raw material of every dashboard: running, waiting, tokens, preemptions, cache usage.',
  'Running, waiting, cache usage and preemptions. Waiting count is the leading indicator: throughput still looks healthy while it climbs, and then latency diverges.',
];

const PRESET: Record<number, { batch: number; mode: 'direct' | 'queued' }> = {
  0: { batch: 32, mode: 'direct' },
  1: { batch: 32, mode: 'direct' },
  2: { batch: 32, mode: 'direct' },
  3: { batch: 32, mode: 'direct' },
  4: { batch: 32, mode: 'direct' },
  5: { batch: 32, mode: 'queued' },
  6: { batch: 32, mode: 'queued' },
  7: { batch: 32, mode: 'queued' },
};

const X0 = 92;
const TL_X1 = 330;         // timeline right edge
const QW = 238;            // panel width

export default function ApiMetrics() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const p0 = PRESET[s.step];
  const [batch, setBatch] = useState(p0.batch);
  const [mode, setMode] = useState<'direct' | 'queued'>(p0.mode);
  useEffect(() => { setBatch(p0.batch); setMode(p0.mode); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const k = s.step;
  const r = decodeStep({ model, gpu, batch, ctx: 2048 });

  // TTFT decomposition. Queue wait scales with batch; prefill from perfModel.
  const prefillMs = (2 * model.params * 512) / (gpu.flops * 0.55) * 1000;
  const queueMs = 120 + batch * 9;
  const stepMs = r.t * 1000;
  const ttft = queueMs + prefillMs + stepMs;
  const TLW = TL_X1 - X0;

  // the metrics panel
  const METRICS = [
    { name: 'vllm:num_requests_running', value: `${batch}`, kind: 'compute' as const, note: 'the live batch' },
    { name: 'vllm:num_requests_waiting', value: `${Math.max(0, Math.round(batch * 0.42))}`, kind: 'error' as const, note: 'queue depth → TTFT' },
    { name: 'vllm:gpu_cache_usage_perc', value: `${(0.31 + batch * 0.021).toFixed(2)}`, kind: 'memory' as const, note: '> 0.95 preempts' },
    { name: 'vllm:preemption_count', value: batch > 60 ? '4,182' : '0', kind: 'error' as const, note: batch > 60 ? 'over capacity' : 'healthy' },
    { name: 'vllm:iteration_tokens_total', value: fmt.int(batch * 214), kind: 'token' as const, note: '÷ wall time = throughput' },
  ];

  // the stream: chunks per token
  const CHUNKS = 7;

  return (
    <Stage
      rootRef={ref}
      title="Streaming out, and measuring it honestly"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'memory', 'compute', 'comm', 'idle', 'error']}
      {...s}
      readout={
        <Readout items={[
          { label: 'TTFT', value: `${ttft.toFixed(0)} ms`, kind: 'text', note: `${((queueMs / ttft) * 100).toFixed(0)}% queueing` },
          { label: 'TPOT', value: `${stepMs.toFixed(1)} ms`, kind: 'compute', note: '≈ one iteration' },
          { label: 'Throughput', value: `${fmt.int(r.total)} tok/s`, kind: 'token', note: `batch ${batch}` },
          { label: 'Stream path', value: mode === 'direct' ? 'GPU → socket' : 'GPU → queue → socket', kind: 'comm', note: mode === 'direct' ? 'network on the critical path' : 'network decoupled' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Batch size" value={String(batch)} options={[8, 32, 64, 128].map((b) => ({ value: String(b), label: String(b) }))} onChange={(v) => setBatch(Number(v))} />
          <Segmented label="Stream path" value={mode} options={[{ value: 'direct', label: 'Direct to socket' }, { value: 'queued', label: 'Background queue' }]} onChange={setMode} />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A timeline decomposing time to first token into queue wait, prefill and one decode step, a row of streamed server-sent event chunks, and a panel listing the five counters a serving engine exports every iteration.">
        {/* ── TTFT decomposition */}
        <Label x={X0} y={18} tone={2} size={11} bold>time to first token, decomposed</Label>
        <text x={X0 - 10} y={44} textAnchor="end" className="v-track-label">TTFT</text>
        <Span x={X0} y={34} w={(TLW * queueMs) / ttft} h={18} kind="idle" label="queue" labelTone="light" />
        <Span x={X0 + (TLW * queueMs) / ttft} y={34} w={(TLW * prefillMs) / ttft} h={18} kind="compute" label="prefill" />
        <Span x={X0 + (TLW * (queueMs + prefillMs)) / ttft} y={34} w={(TLW * stepMs) / ttft} h={18} kind="memory" label="1 step" />
        <Label x={X0 + (TLW * queueMs) / ttft / 2} y={24} anchor="middle" tone={3} size={9}>{`${queueMs.toFixed(0)} ms`}</Label>
        <Label x={X0 + (TLW * (queueMs + prefillMs)) / ttft} y={24} anchor="middle" tone={3} size={9}>{`${prefillMs.toFixed(0)} ms`}</Label>
        <Dim x1={X0} x2={X0 + TLW} y={66} label={`TTFT = ${ttft.toFixed(0)} ms`} kind="text" />
        <Label x={X0} y={82} tone={3} size={9.5}>{`queue ${((queueMs / ttft) * 100).toFixed(0)}% · prefill ${((prefillMs / ttft) * 100).toFixed(0)}% · step ${((stepMs / ttft) * 100).toFixed(0)}%`}</Label>

        {/* ── the stream */}
        <Label x={X0} y={116} tone={2} size={11} bold>what the client receives</Label>
        {mode === 'direct' ? (
          <>
            <Label x={X0} y={136} tone={3} size={9.5}>GPU iteration → socket write, on the critical path</Label>
            <Span x={X0} y={144} w={TLW * 0.82} h={16} kind="comm" dim label="iteration 1 … n, each writing straight to the socket" labelTone="light" />
            <Fade show={k >= 5}>
              <Span x={X0 + TLW * 0.82} y={144} w={TLW * 0.18} h={16} kind="error" label="slow client" />
            </Fade>
          </>
        ) : (
          <>
            <Label x={X0} y={136} tone={3} size={9.5}>GPU iteration → bounded queue → socket, off the critical path</Label>
            <Span x={X0} y={144} w={TLW * 0.62} h={16} kind="token" label="iterations: produce and enqueue" />
            <Arrow x1={X0 + TLW * 0.64} y1={152} x2={X0 + TLW * 0.72} y2={152} kind="comm" />
            <Span x={X0 + TLW * 0.74} y={144} w={TLW * 0.26} h={16} kind="comm" dim label="writer thread" labelTone="light" />
            <Fade show={k >= 5}>
              <Label x={X0 + TLW * 0.74} y={176} size={9.5} kind="token">a 200 ms socket write no longer lands inside a 7 ms step</Label>
            </Fade>
          </>
        )}

        {/* SSE chunks */}
        <Label x={X0} y={196} tone={3} size={9.5}>SSE: one `data:` line per token</Label>
        <Fade show={k >= 3}>
          {Array.from({ length: CHUNKS }, (_, i) => {
            const cw = 30;
            const x = X0 + i * (cw + 4);
            return (
              <g key={i}>
                <rect x={x} y={204} width={cw} height={20} rx={2} fill="var(--ground-3)" stroke="var(--comm)" strokeWidth={0.9} />
                <text x={x + cw / 2} y={217} textAnchor="middle" className="v-label" style={{ fontSize: 8, fill: 'var(--text-2)' }}>data: …</text>
              </g>
            );
          })}
          <Label x={X0 + CHUNKS * 34 + 6} y={217} tone={3} size={9}>… then `data: [DONE]`</Label>
        </Fade>

        {/* detokenizer */}
        <Fade show={k >= 4}>
          <Label x={X0} y={248} tone={2} size={10.5}>detokenizer: ids are not characters</Label>
          <Chip x={X0} y={256} text="thr" kind="comm" w={34} h={18} size={9} />
          <Chip x={X0 + 40} y={256} text="ough" kind="comm" w={40} h={18} size={9} />
          <Label x={X0 + 86} y={269} tone={3} size={9.5}>two tokens, one word — and “café” spans three, one byte at a time</Label>
        </Fade>

        {/* ── the metrics panel */}
        <rect x={366} y={16} width={QW} height={190} rx={5} fill="var(--ground-2)" stroke="var(--rule)" />
        <Label x={378} y={34} tone={2} size={10.5} bold>exported every iteration</Label>
        {METRICS.map((m, i) => {
          const y = 52 + i * 30;
          return (
            <g key={m.name}>
              <rect x={378} y={y} width={3.5} height={22} rx={1} fill={`var(--${m.kind})`} />
              <text x={388} y={y + 10} className="v-label" style={{ fontSize: 8.2, fill: 'var(--text-2)' }}>{m.name}</text>
              <text x={388} y={y + 21} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text-3)' }}>{m.note}</text>
              <text x={366 + QW - 12} y={y + 15} textAnchor="end" className="v-label" style={{ fontSize: 12, fontWeight: 600, fill: `var(--${m.kind})` }}>{m.value}</text>
            </g>
          );
        })}
        <Fade show={k >= 7}>
          <Label x={378} y={224} size={9.5} kind="error" bold>waiting count is the leading indicator</Label>
          <Label x={378} y={238} size={9.5} tone={2} mono={false}>throughput looks fine while it climbs —</Label>
          <Label x={378} y={250} size={9.5} tone={2} mono={false}>then p99 diverges without warning.</Label>
        </Fade>

        {/* ── the identity */}
        <Fade show={k >= 0}>
          <Label x={X0} y={302} tone={2} size={10.5} bold>the identity your harness must satisfy</Label>
          <g>
            <Chip x={X0} y={312} text="TTFT" kind="memory" w={44} h={20} />
            <Label x={X0 + 48} y={326} tone={3} size={10}>+</Label>
            <Chip x={X0 + 60} y={312} text="TPOT" kind="compute" w={48} h={20} />
            <Label x={X0 + 112} y={326} tone={3} size={10}>×</Label>
            <Chip x={X0 + 124} y={312} text="n − 1" kind="text" w={42} h={20} />
            <Label x={X0 + 172} y={326} tone={3} size={11}>=</Label>
            <Chip x={X0 + 188} y={312} text="end-to-end" kind="token" w={84} h={20} />
          </g>
          <Label x={X0} y={352} tone={3} size={9.5}>If this fails in your harness, you have a timing bug and no other number is trustworthy.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}