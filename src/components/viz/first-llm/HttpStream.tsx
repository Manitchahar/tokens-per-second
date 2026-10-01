import { useRef, useState } from 'react';
import { Stage, Segmented, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Move, Arrow, fmt } from '../lib/svg';
import { modelById, gpuById, decodeStep } from '../lib/perfModel';

// A request is an HTTP message. A streaming answer is thousands of them, on one connection.
// The transport decisions here are latency decisions, and they show up in TTFT.

const STEPS = [
  'One HTTP request',
  'JSON goes in',
  'The server owns the model',
  'One answer, or many',
  'Server-sent events',
  'The stream, token by token',
  'What the client sees',
  'Concurrency is not free',
  'Health, limits, shutdown',
];
const CAPTIONS = [
  'Serving a model means answering an HTTP POST with generated text. The request is JSON; the model never sees HTTP, it sees tokens.',
  'The client sends a model, a list of messages and generation parameters. Everything the model needs and everything you are allowed to ask for travels in this one object.',
  'The server holds the weights resident on the GPU and keeps them there between requests. That is the whole reason an engine exists: loading is expensive, so you load once.',
  'You can return the finished answer in one response, or stream it. Both use the same endpoint. The difference is whether the client waits for everything.',
  'Streaming uses server-sent events: one long-lived HTTP response with a text/event-stream content type, and a line of data per token.',
  'Each token is written and flushed immediately. The client receives it in the same order it was produced, without waiting for the next.',
  'The first token arrives after time to first token; the rest arrive at the decode rate. A streaming client renders text as fast as a chat product does.',
  'A server can hold many requests at once, but each one still needs KV cache. Concurrency is a memory decision before it is a throughput decision.',
  'Three operational endpoints every real server has: a cheap health check, a request size limit, and a graceful shutdown that stops accepting work.',
];

const M = modelById('qwen05');
const G = gpuById('t4');

// A plausible SSE body, the shape every client library expects.
const SSE_LINES = [
  'data: {"choices":[{"delta":{"content":"The"},"index":0}]}',
  'data: {"choices":[{"delta":{"content":" sea"},"index":0}]}',
  'data: {"choices":[{"delta":{"content":" is"},"index":0}]}',
  'data: {"choices":[{"delta":{"content":" vast"},"index":0}]}',
  'data: {"choices":[{"delta":{"content":"."},"index":0}]}',
  'data: [DONE]',
];

const TOKS = ['The', ' sea', ' is', ' vast', '.'];

export default function HttpStream() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [conc, setConc] = useState(1);
  const r = decodeStep({ model: M, gpu: G, batch: Math.max(1, conc), ctx: 2048 });
  const tps = r.total;

  return (
    <Stage
      rootRef={ref}
      title="One HTTP request, one model, one stream of tokens"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['comm', 'token', 'memory', 'text']}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'Requests in flight', value: String(conc), kind: 'comm', note: 'each one needs its own KV cache' },
            { label: 'Throughput', value: `${tps.toFixed(1)} tok/s`, kind: 'token', note: 'aggregate, all requests' },
            { label: 'Per user', value: `${r.perUser.toFixed(1)} tok/s`, kind: 'memory', note: 'TPOT is the reciprocal' },
            { label: 'KV cache in use', value: fmt.gb(r.kvTotal), kind: 'memory', note: `${conc} × 2,048 tokens` },
          ]}
        />
      }
      controls={
        <Slider label="Requests in flight at once" value={conc} min={1} max={64} log onChange={setConc} format={(v) => `${fmt.int(v)} concurrent`} kind="comm" />
      }
    >
      <svg viewBox="0 0 640 420" role="img" aria-label="A diagram of a streaming HTTP request: JSON with model and messages posted to a server holding a model on the GPU, which returns server-sent events, one JSON line per generated token, arriving at the client at the decode rate.">
        {/* ── the client and the server ── */}
        <rect x={56} y={70} width={112} height={132} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1.2} />
        <Label x={112} y={90} anchor="middle" tone={1} size={10.5} bold mono={false}>client</Label>
        <Label x={112} y={106} anchor="middle" tone={3} size={8.5}>browser · app</Label>
        <g className="v-fade" style={{ opacity: st >= 6 ? 1 : 0.25 }}>
          <rect x={68} y={118} width={88} height={64} rx={3} fill="var(--ground-2)" stroke="var(--token)" strokeWidth={1.2} className="v-cell" />
          <text x={112} y={132} textAnchor="middle" className="v-label bold" style={{ fontSize: 9, fill: 'var(--token)' }}>renders</text>
          <text x={112} y={148} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--text-2)' }}>The sea is</text>
          <text x={112} y={162} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--text-2)' }}>vast.</text>
          <text x={112} y={176} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>caret blinking</text>
        </g>
        <Label x={112} y={196} anchor="middle" tone={3} size={8.5}>HTTP client</Label>

        {/* the server */}
        <rect x={300} y={52} width={284} height={168} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1.2} />
        <Label x={316} y={72} tone={1} size={10.5} bold mono={false}>server</Label>
        <Label x={568} y={72} anchor="end" tone={3} size={8.5}>one process, one GPU</Label>

        {/* the model resident on the GPU */}
        <g className="v-fade" style={{ opacity: st >= 2 ? 1 : 0.25 }}>
          <rect x={316} y={84} width={124} height={54} rx={4} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1.2} className="v-cell" />
          <text x={378} y={102} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: 'var(--memory)' }}>model</text>
          <text x={378} y={116} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--text-2)' }}>{M.name}</text>
          <text x={378} y={130} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{`${fmt.gb(M.params * 2)} resident, loaded once`}</text>
        </g>

        {/* the KV cache, per request */}
        <g className="v-fade" style={{ opacity: st >= 7 ? 1 : 0.25 }}>
          <rect x={452} y={84} width={116} height={54} rx={4} fill="var(--ground-2)" stroke="var(--memory)" strokeWidth={1.2} strokeDasharray="4 3" className="v-cell" />
          <text x={510} y={102} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: 'var(--memory)' }}>KV cache</text>
          <text x={510} y={116} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--text-2)' }}>{conc} request{conc === 1 ? '' : 's'}</text>
          <text x={510} y={130} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{fmt.gb(r.kvTotal)}</text>
        </g>

        {/* the generate loop, running */}
        <g className="v-fade" style={{ opacity: st >= 2 ? 1 : 0.2 }}>
          <rect x={316} y={150} width={252} height={54} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <text x={330} y={166} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text-2)' }}>generate loop</text>
          <text x={330} y={180} className="v-label t3" style={{ fontSize: 8.5 }}>prefill → decode one token → stream it → repeat</text>
          <text x={330} y={194} className="v-label t3" style={{ fontSize: 8.5 }}>{`${r.perUser.toFixed(0)} steps/s per user · ${(r.t * 1e3).toFixed(1)} ms/token`}</text>
        </g>

        {/* ── the request going up ── */}
        <Fade show={st >= 1}>
          <path d="M 168 100 L 296 100" fill="none" stroke="var(--comm)" strokeWidth={1.6} markerEnd="url(#tps-arrow)" />
          <rect x={186} y={78} width={104} height={15} rx={2} fill="var(--ground)" />
          <text x={238} y={89} textAnchor="middle" className="v-label bold" style={{ fontSize: 8.5, fill: 'var(--comm)' }}>POST /v1/chat</text>
        </Fade>

        {/* the JSON body */}
        <Fade show={st === 1}>
          <rect x={56} y={232} width={252} height={92} rx={4} fill="var(--ground-2)" stroke="var(--comm)" strokeWidth={1.2} />
          <text x={68} y={248} className="v-label bold" style={{ fontSize: 9, fill: 'var(--comm)' }}>request body</text>
          <text x={68} y={264} className="v-label" style={{ fontSize: 8.5, fill: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{'{'}</text>
          <text x={76} y={277} className="v-label" style={{ fontSize: 8.5, fill: 'var(--text-2)' }}>{'"model": "Qwen2.5-0.5B",'}</text>
          <text x={76} y={290} className="v-label" style={{ fontSize: 8.5, fill: 'var(--text-2)' }}>{'"messages": [...],'}</text>
          <text x={76} y={303} className="v-label" style={{ fontSize: 8.5, fill: 'var(--text-2)' }}>{'"stream": true,'}</text>
          <text x={76} y={316} className="v-label" style={{ fontSize: 8.5, fill: 'var(--text-2)' }}>{'"max_tokens": 256'}</text>
        </Fade>

        {/* ── the response coming down: SSE ── */}
        <Fade show={st >= 4}>
          <path d="M 568 118 L 578 118 L 578 96 L 172 96" fill="none" stroke="var(--comm)" strokeWidth={1.6} markerEnd="url(#tps-arrow)" />
        </Fade>
        <Fade show={st >= 4}>
          <rect x={368} y={342} width={216} height={14} rx={2} fill="var(--ground)" />
          <text x={476} y={352} textAnchor="middle" className="v-label bold" style={{ fontSize: 8.5, fill: 'var(--comm)' }}>
            Content-Type: text/event-stream
          </text>
        </Fade>

        {/* the SSE body */}
        <Fade show={st >= 5}>
          <rect x={330} y={232} width={254} height={132} rx={4} fill="var(--ground-2)" stroke="var(--token)" strokeWidth={1.2} />
          <text x={342} y={248} className="v-label bold" style={{ fontSize: 9, fill: 'var(--token)' }}>the response, as it is written</text>
          {SSE_LINES.map((l, i) => (
            <text key={i} x={342} y={264 + i * 17} className="v-label" style={{ fontSize: 7.5, fill: i === SSE_LINES.length - 1 ? 'var(--text-3)' : 'var(--text-2)' }}>
              {l.length > 40 ? `${l.slice(0, 38)}…` : l}
            </text>
          ))}
        </Fade>

        {/* one line per token, arriving */}
        <Fade show={st >= 5}>
          <Label x={56} y={352} tone={1} size={10} bold mono={false}>One line per token</Label>
          {TOKS.map((t, i) => (
            <g key={i} className="v-fade">
              <rect x={56 + i * 50} y={360} width={46} height={20} rx={3}
                fill={i === TOKS.length - 1 ? 'var(--token-dim)' : 'var(--token-dim)'} stroke="var(--token)" strokeWidth={0.9} className="v-cell" />
              <text x={79 + i * 50} y={374} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--text)' }}>{t}</text>
            </g>
          ))}
          <text x={56 + TOKS.length * 50 + 4} y={374} className="v-label t3" style={{ fontSize: 8.5 }}>then [DONE]</text>
        </Fade>

        {/* ── operational endpoints, last step ── */}
        <Fade show={st === 8}>
          <line x1={56} y1={396} x2={584} y2={396} className="v-rule faint" />
          <Label x={56} y={412} tone={2} size={9.5} mono={false}>
            /health answers in microseconds and never touches the GPU · max_tokens bounds the cost of every request · SIGTERM stops new work and drains in-flight requests
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
