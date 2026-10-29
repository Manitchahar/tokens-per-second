import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, Arrow, COLOR, fmt } from '../lib/svg';

const STEPS = [
  'Traditional metrics miss AI failures',
  'Distributed OpenTelemetry trace spans',
  'Granular token & cost telemetry',
  'Sampled LLM-as-a-judge scoring',
  'Semantic embedding drift detection',
  'Privacy scrubbing & log redaction',
  'The production telemetry dashboard',
];

const CAPTIONS = [
  'HTTP 200 success codes and CPU utilization gauges show healthy green status even when the model is generating hallucinated facts or leaking confidential instructions.',
  'OpenTelemetry distributed tracing captures the end-to-end hierarchy: parent request span, input guardrails, retrieval, model prefill, streaming decode, and output validation.',
  'Per-span telemetry records exact prompt tokens, completion tokens, TTFT, TPOT, and micro-dollar costs, attributing expenses to individual tenant IDs in real time.',
  'Logging 100% of text is cheap, but evaluating 100% with frontier models is cost-prohibitive. A 5% async sample is dispatched to an LLM judge to score faithfulness and relevance.',
  'Tracking average query embedding centroids over time detects distribution drift (PSI > 0.25), alerting engineers when user behaviors shift or novel topics emerge.',
  'Before traces are persisted to long-term telemetry lakes, an asynchronous privacy scrubber redacts residual PII and separates operational metadata from prompt text.',
  'A production observability dashboard synthesizes metrics, traces, judge scores, and drift alerts, enabling proactive alerting before users report degradation.',
];

export default function AppObservabilityTrace() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [sampleRate, setSampleRate] = useState(5);
  const [driftScore, setDriftScore] = useState(0.18);

  // Arithmetic for evaluation cost & sample volume:
  // 100,000 requests/day
  // Sample rate %
  const dailyQueries = 100000;
  const evalQueries = Math.round(dailyQueries * (sampleRate / 100));
  // Eval cost: $0.005 per eval query
  const evalCostDaily = evalQueries * 0.005;

  const driftStatus = driftScore > 0.25 ? 'Significant Drift' : driftScore > 0.1 ? 'Moderate Shift' : 'Stable';

  return (
    <Stage
      rootRef={ref}
      title="End-to-end AI application observability and distributed tracing"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['comm', 'distributed spans'],
        ['memory', 'token & cost telemetry'],
        ['compute', 'sampled LLM judge'],
        ['token', 'verified quality'],
        ['error', 'drift / anomaly alert'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Sample rate',
              value: `${sampleRate}%`,
              kind: 'compute',
              note: `${evalQueries.toLocaleString()} judge evals/day`,
            },
            {
              label: 'Judge cost/day',
              value: `$${evalCostDaily.toFixed(2)}`,
              kind: evalCostDaily < 50 ? 'token' : 'error',
              note: `$${(evalCostDaily * 30).toFixed(0)} / month`,
            },
            {
              label: 'Drift index (PSI)',
              value: driftScore.toFixed(2),
              kind: driftScore > 0.25 ? 'error' : driftScore > 0.1 ? 'compute' : 'token',
              note: driftStatus,
            },
            {
              label: 'Telemetry delay',
              value: '0 ms',
              kind: 'token',
              note: 'fully asynchronous buffer',
            },
          ]}
        />
      }
      controls={
        s.step >= 3 ? (
          <>
            <Slider
              label="Judge sample rate"
              value={sampleRate}
              min={1}
              max={20}
              step={1}
              onChange={setSampleRate}
              format={(v) => `${v}%`}
              kind="compute"
            />
            {s.step >= 4 && (
              <Slider
                label="Embedding drift (PSI)"
                value={Math.round(driftScore * 100)}
                min={5}
                max={45}
                step={5}
                onChange={(v) => setDriftScore(v / 100)}
                format={(v) => `${(v / 100).toFixed(2)}`}
                kind={driftScore > 0.25 ? 'error' : 'memory'}
              />
            )}
          </>
        ) : undefined
      }
      {...s}
    >
      <svg
        viewBox="0 0 640 330"
        role="img"
        aria-label="Observability trace diagram showing OpenTelemetry spans, token telemetry, LLM-as-a-judge sampling pipeline, and embedding drift detection."
      >
        {/* Step 0: Traditional APM Blindness */}
        {s.step === 0 && (
          <g>
            <rect x={30} y={40} width={260} height={120} rx={4} fill="var(--bg-card)" stroke={COLOR.token} strokeWidth={1.5} />
            <Label x={160} y={64} anchor="middle" tone={1} size={11} bold kind="token">Traditional APM (Datadog / NewRelic)</Label>
            <Label x={45} y={88} anchor="start" tone={2} size={9.5}>• HTTP Status: 200 OK (Green)</Label>
            <Label x={45} y={106} anchor="start" tone={2} size={9.5}>• CPU Usage: 14% (Normal)</Label>
            <Label x={45} y={124} anchor="start" tone={2} size={9.5}>• Latency: 420 ms (Within SLA)</Label>
            <Label x={45} y={142} anchor="start" tone={1} size={9.5} bold kind="token">Verdict: SYSTEM HEALTHY</Label>

            <rect x={330} y={40} width={280} height={120} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.5} />
            <Label x={470} y={64} anchor="middle" tone={1} size={11} bold kind="error">What the User Actually Saw</Label>
            <Label x={345} y={88} anchor="start" tone={2} size={9}>Prompt: "What is the return window for laptops?"</Label>
            <Label x={345} y={106} anchor="start" tone={1} size={9} bold kind="error">Response: "Laptops can be returned in 999 years."</Label>
            <Label x={345} y={126} anchor="start" tone={2} size={8.5}>• Critical hallucination on company policy</Label>
            <Label x={345} y={142} anchor="start" tone={1} size={9.5} bold kind="error">Silent Failure: APM recorded HTTP 200</Label>

            <rect x={80} y={200} width={480} height={60} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1} />
            <Label x={320} y={224} anchor="middle" tone={1} size={11} bold kind="error">The Semantic Blind Spot of Traditional Ops</Label>
            <Label x={320} y={244} anchor="middle" tone={2} size={9.5}>AI applications do not fail with HTTP 500 crashes; they fail semantically with confident falsehoods.</Label>
          </g>
        )}

        {/* Steps >= 1: OpenTelemetry Trace Waterfall */}
        {s.step >= 1 && (
          <g>
            {/* Trace Timeline header */}
            <Label x={30} y={28} anchor="start" tone={1} size={11} bold>Trace Waterfall (query_id: 8f2a1b9c)</Label>
            <Label x={540} y={28} anchor="end" tone={3} size={9}>Duration: 412 ms</Label>

            {/* Parent Span */}
            <rect x={30} y={40} width={580} height={22} rx={2} fill="var(--comm-dim)" stroke={COLOR.comm} strokeWidth={1} />
            <Label x={40} y={55} anchor="start" tone={1} size={9.5} bold kind="comm">POST /v1/chat/completions (root)</Label>
            <Label x={600} y={55} anchor="end" tone={2} size={8.5}>412 ms</Label>

            {/* Child Span: Guardrails In */}
            <rect x={40} y={68} width={45} height={18} rx={2} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1} />
            <Label x={92} y={81} anchor="start" tone={2} size={8.5}>input_guardrails (24 ms)</Label>

            {/* Child Span: Cache Check */}
            <rect x={88} y={92} width={15} height={18} rx={2} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1} />
            <Label x={110} y={105} anchor="start" tone={2} size={8.5}>cache_lookup: miss (3 ms)</Label>

            {/* Child Span: RAG Retrieval */}
            <rect x={106} y={116} width={75} height={18} rx={2} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1} />
            <Label x={188} y={129} anchor="start" tone={2} size={8.5}>rag_retrieval (48 ms · 4 chunks)</Label>

            {/* Child Span: Model Prefill & Stream */}
            <rect x={184} y={140} width={380} height={24} rx={2} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.2} />
            <Label x={192} y={156} anchor="start" tone={1} size={9.5} bold kind="compute">llm_generation: llama-3.1-70b (312 ms)</Label>
            <Label x={554} y={156} anchor="end" tone={2} size={8.5}>182 tok</Label>

            {/* TTFT Marker inside LLM span */}
            <line x1={264} y1={140} x2={264} y2={164} stroke="var(--token)" strokeWidth={1.8} strokeDasharray="2 2" />
            <Label x={266} y={151} anchor="start" tone={1} size={8} bold kind="token">TTFT: 80ms</Label>

            {/* Child Span: Output Guardrails */}
            <rect x={568} y={170} width={38} height={18} rx={2} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1} />
            <Label x={560} y={198} anchor="end" tone={2} size={8.5}>output_verification (18 ms)</Label>

            {/* Step 2: Telemetry Ledger */}
            {s.step === 2 && (
              <g>
                <rect x={30} y={210} width={580} height={90} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
                <Label x={50} y={232} anchor="start" tone={1} size={11} bold kind="memory">Per-Span Metadata & Cost Accounting</Label>
                <Label x={50} y={252} anchor="start" tone={2} size={9.5}>• Prompt tokens: 1,840 | Completion tokens: 182 | Total: 2,022 tokens</Label>
                <Label x={50} y={270} anchor="start" tone={2} size={9.5}>• TTFT: 80 ms | TPOT: 1.27 ms/token | Provider: us-east-1 vLLM Cluster</Label>
                <Label x={50} y={288} anchor="start" tone={1} size={9.5} bold kind="token">• Incurred Cost: $0.00825 (billed to tenant: "enterprise-sales")</Label>
              </g>
            )}

            {/* Step 3: LLM Judge Sampling */}
            {s.step === 3 && (
              <g>
                <rect x={30} y={210} width={580} height={95} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.2} />
                <Label x={50} y={232} anchor="start" tone={1} size={11} bold kind="compute">Asynchronous LLM-as-a-Judge Pipeline</Label>
                <Label x={50} y={252} anchor="start" tone={2} size={9.5}>
                  {`• Evaluates ${sampleRate}% sample (${evalQueries} req/day) via background worker pool.`}
                </Label>
                <Label x={50} y={270} anchor="start" tone={2} size={9.5}>• Faithfulness Score: 0.94 / 1.00 (grounded in retrieved context)</Label>
                <Label x={50} y={288} anchor="start" tone={2} size={9.5}>• Answer Relevance: 0.96 / 1.00 | Safety Compliance: Pass (1.00)</Label>
              </g>
            )}

            {/* Step 4: Semantic Drift Detection */}
            {s.step === 4 && (
              <g>
                <rect x={30} y={210} width={580} height={95} rx={4} fill={driftScore > 0.25 ? 'var(--error-dim)' : 'var(--bg-card)'} stroke={driftScore > 0.25 ? COLOR.error : 'var(--rule)'} strokeWidth={1.2} />
                <Label x={50} y={232} anchor="start" tone={1} size={11} bold kind={driftScore > 0.25 ? 'error' : 'token'}>
                  {`Semantic Embedding Drift Detection: PSI = ${driftScore.toFixed(2)} (${driftStatus})`}
                </Label>
                <Label x={50} y={252} anchor="start" tone={2} size={9.5}>• Comparing 7-day rolling query embedding centroids against baseline launch corpus.</Label>
                <Label x={50} y={270} anchor="start" tone={2} size={9.5}>
                  {driftScore > 0.25
                    ? '• WARNING: Substantial shift detected in user vocabulary. Retrieval Recall dropping.'
                    : '• Traffic distribution matches reference baseline within normal variance.'}
                </Label>
                <Label x={50} y={288} anchor="start" tone={3} size={8.5}>Wasserstein distance & Population Stability Index calculated over d=384 embeddings</Label>
              </g>
            )}

            {/* Step 5: Privacy Scrubbing */}
            {s.step === 5 && (
              <g>
                <rect x={30} y={210} width={580} height={95} rx={4} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1.2} />
                <Label x={50} y={232} anchor="start" tone={1} size={11} bold kind="memory">Asynchronous Telemetry Anonymization & Cold Storage</Label>
                <Label x={50} y={252} anchor="start" tone={2} size={9.5}>• Residual PII scrubbed from raw traces before S3 / ClickHouse persistence.</Label>
                <Label x={50} y={270} anchor="start" tone={2} size={9.5}>• Separation of concerns: operational metrics kept indefinitely; raw prompt text expires in 14 days.</Label>
                <Label x={50} y={288} anchor="start" tone={1} size={9.5} bold kind="token">• Fully compliant with GDPR Article 17 and HIPAA data governance.</Label>
              </g>
            )}

            {/* Step 6: Live Production Dashboard */}
            {s.step === 6 && (
              <g>
                <rect x={30} y={210} width={580} height={95} rx={4} fill="var(--bg-card)" stroke={COLOR.token} strokeWidth={1.5} />
                <Label x={50} y={232} anchor="start" tone={1} size={11} bold kind="token">Production Health Summary</Label>
                <Label x={50} y={252} anchor="start" tone={2} size={9.5}>• P95 TTFT: 142 ms | P95 TPOT: 1.4 ms/token | Error Rate: 0.02%</Label>
                <Label x={50} y={270} anchor="start" tone={2} size={9.5}>• Daily Token Volume: 42.8M tokens | Daily Spend: $184.20</Label>
                <Label x={50} y={288} anchor="start" tone={1} size={9.5} bold kind="token">• Composite Quality Score: 96.2% | Automated Alerting: 0 Active Incidents</Label>
              </g>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
