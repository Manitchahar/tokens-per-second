import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, Arrow, COLOR, fmt } from '../lib/svg';

const STEPS = [
  'The bare model call',
  'Grounding with context',
  'Input guardrails',
  'Gateway and caching',
  'Model routing',
  'Output validation',
  'Observability and feedback',
];

const CAPTIONS = [
  'A bare model call sends raw user text straight to the LLM. It is simple to prototype, but ungrounded, unmonitored, expensive, and completely unprotected.',
  'Retrieval-Augmented Generation (RAG) and tool calls inject internal facts into the prompt, eliminating knowledge cutoff errors at the expense of longer prefill.',
  'Input guardrails intercept prompts to scrub PII with reversible tokens and block prompt injections before they consume expensive model compute.',
  'An API gateway enforces rate limits while an exact cache answers repeated queries in 2 ms with zero GPU cycles and zero inference cost.',
  'An intent router inspects query complexity, sending 70% of simple tasks to a cheap 8B model and saving the expensive 70B model for multi-step reasoning.',
  'Output guardrails validate JSON schemas, verify factual grounding against context, and substitute de-masked PII safely before delivering to the user.',
  'Asynchronous telemetry logs traces, tokens, TTFT, and user feedback into a data flywheel without adding latency to the user’s critical path.',
];

export default function ArchitectureStack() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [cacheHit, setCacheHit] = useState(25);
  const [smallShare, setSmallShare] = useState(70);

  // Arithmetic for latency and cost:
  // Base costs per 1k requests:
  // Large model alone: $9.00 / 1k req ($0.009/req)
  // Small model: $0.42 / 1k req ($0.00042/req)
  // Guardrails infra: $0.01 / 1k req
  // Cache hit: $0.00
  const effectiveCacheHit = s.step >= 3 ? cacheHit / 100 : 0;
  const effectiveSmallShare = s.step >= 4 ? smallShare / 100 : 0;

  const costLarge = 9.0;
  const costSmall = 0.42;
  const missCost = effectiveSmallShare * costSmall + (1 - effectiveSmallShare) * costLarge;
  const totalCost = (1 - effectiveCacheHit) * missCost + (s.step >= 2 ? 0.01 : 0);

  // Latency breakdown in milliseconds:
  const tGateway = s.step >= 3 ? 4 : 0;
  const tCache = s.step >= 3 ? 3 : 0;
  const tGuardIn = s.step >= 2 ? 22 : 0;
  const tRetrieval = s.step >= 1 ? 45 : 0;
  const tLlmSmall = 140;
  const tLlmLarge = 580;
  const tLlmAvg = s.step >= 4
    ? effectiveSmallShare * tLlmSmall + (1 - effectiveSmallShare) * tLlmLarge
    : tLlmLarge;
  const tGuardOut = s.step >= 5 ? 18 : 0;

  // Cache hit bypasses guardrails & LLM
  const avgLatency = effectiveCacheHit * (tGateway + tCache) +
    (1 - effectiveCacheHit) * (tGateway + tCache + tGuardIn + tRetrieval + tLlmAvg + tGuardOut);

  const safetyCoverage = s.step < 2 ? 0 : s.step < 5 ? 60 : 100;

  return (
    <Stage
      rootRef={ref}
      title="Evolution of a production AI application stack"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['comm', 'gateway & routing'],
        ['memory', 'cache & context'],
        ['error', 'guardrails'],
        ['compute', 'model inference'],
        ['token', 'telemetry & output'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Avg latency',
              value: `${Math.round(avgLatency)} ms`,
              kind: avgLatency < 300 ? 'token' : 'compute',
              note: effectiveCacheHit > 0 ? `${cacheHit}% cached in 7 ms` : 'full model pass',
            },
            {
              label: 'Cost / 1k req',
              value: `$${totalCost.toFixed(2)}`,
              kind: totalCost < 3 ? 'token' : 'error',
              note: s.step >= 4 ? `${smallShare}% to 8B model` : '100% to frontier model',
            },
            {
              label: 'Safety coverage',
              value: `${safetyCoverage}%`,
              kind: safetyCoverage === 100 ? 'token' : safetyCoverage > 0 ? 'compute' : 'error',
              note: s.step >= 5 ? 'input & output guarded' : s.step >= 2 ? 'input only' : 'unprotected',
            },
            {
              label: 'Cache savings',
              value: `${Math.round(effectiveCacheHit * 100)}%`,
              kind: 'memory',
              note: 'zero GPU cycles on hit',
            },
          ]}
        />
      }
      controls={
        s.step >= 3 ? (
          <>
            <Slider
              label="Cache hit rate"
              value={cacheHit}
              min={0}
              max={50}
              step={5}
              onChange={setCacheHit}
              format={(v) => `${v}%`}
              kind="memory"
            />
            {s.step >= 4 && (
              <Slider
                label="Small model share"
                value={smallShare}
                min={0}
                max={100}
                step={5}
                onChange={setSmallShare}
                format={(v) => `${v}%`}
                kind="compute"
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
        aria-label="Interactive architecture diagram showing layers of an AI application: Gateway, Cache, Guardrails, Retrieval, Router, LLMs, and Observability."
      >
        {/* User Ingress */}
        <rect x={16} y={135} width={64} height={42} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
        <Label x={48} y={154} anchor="middle" tone={1} size={11} bold>Client</Label>
        <Label x={48} y={168} anchor="middle" tone={3} size={9.5}>Prompt</Label>

        {/* Step 0: Bare model flow */}
        {s.step === 0 && (
          <>
            <Arrow x1={80} y1={156} x2={260} y2={156} kind="compute" width={2} />
            <rect x={260} y={120} width={130} height={72} rx={6} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.5} />
            <Label x={325} y={150} anchor="middle" tone={1} size={13} bold kind="compute">Frontier LLM</Label>
            <Label x={325} y={168} anchor="middle" tone={2} size={10}>Raw Matrix Mult</Label>
            <Label x={325} y={182} anchor="middle" tone={3} size={9}>$9.00 / 1k req · 580 ms</Label>

            <Arrow x1={390} y1={156} x2={550} y2={156} kind="token" width={2} />
            <rect x={550} y={135} width={74} height={42} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
            <Label x={587} y={154} anchor="middle" tone={1} size={11} bold>Output</Label>
            <Label x={587} y={168} anchor="middle" tone={3} size={9.5}>Unverified</Label>

            {/* Warning callout banner inside SVG */}
            <rect x={120} y={240} width={400} height={46} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1} />
            <Label x={320} y={260} anchor="middle" tone={1} size={11} bold kind="error">Vulnerabilities in Bare Call</Label>
            <Label x={320} y={276} anchor="middle" tone={2} size={9.5}>PII leaks, prompt injections, no caching, 100% frontier model bill</Label>
          </>
        )}

        {/* Evolving Architecture: Step >= 1 */}
        {s.step >= 1 && (
          <>
            {/* Gateway & Ingress (Step 3+) */}
            {s.step >= 3 ? (
              <g>
                <Arrow x1={80} y1={156} x2={106} y2={156} kind="comm" />
                <rect x={106} y={115} width={70} height={82} rx={4} fill="var(--comm-dim)" stroke={COLOR.comm} strokeWidth={1.2} />
                <Label x={141} y={135} anchor="middle" tone={1} size={10} bold kind="comm">Gateway</Label>
                <Label x={141} y={150} anchor="middle" tone={2} size={8.5}>Auth & Rate</Label>
                <Label x={141} y={164} anchor="middle" tone={3} size={8}>Limit (4 ms)</Label>

                {/* Exact Cache branch */}
                <Arrow x1={141} y1={115} x2={141} y2={64} kind="memory" />
                <rect x={100} y={24} width={82} height={40} rx={4} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1.2} />
                <Label x={141} y={40} anchor="middle" tone={1} size={9.5} bold kind="memory">Exact Cache</Label>
                <Label x={141} y={54} anchor="middle" tone={2} size={8.5}>{`${cacheHit}% hit (2 ms)`}</Label>

                {/* Cache hit short-circuit line to output */}
                <path d="M 182 44 L 560 44 L 560 135" fill="none" stroke={COLOR.token} strokeWidth={1.4} strokeDasharray="3 3" />
                <Label x={370} y={38} anchor="middle" tone={1} size={9} kind="token">Cache hit: 0 ms GPU / $0 cost</Label>

                <Arrow x1={176} y1={156} x2={202} y2={156} kind="comm" />
              </g>
            ) : (
              <Arrow x1={80} y1={156} x2={202} y2={156} kind="comm" />
            )}

            {/* Input Guardrails (Step 2+) */}
            {s.step >= 2 ? (
              <g>
                <rect x={202} y={115} width={76} height={82} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.2} />
                <Label x={240} y={134} anchor="middle" tone={1} size={9.5} bold kind="error">Input Rails</Label>
                <Label x={240} y={149} anchor="middle" tone={2} size={8.5}>PII Masking</Label>
                <Label x={240} y={163} anchor="middle" tone={2} size={8.5}>Injection Fltr</Label>
                <Label x={240} y={178} anchor="middle" tone={3} size={8}>+22 ms</Label>
                <Arrow x1={278} y1={156} x2={304} y2={156} kind="memory" />
              </g>
            ) : (
              <Arrow x1={140} y1={156} x2={304} y2={156} kind="memory" />
            )}

            {/* RAG & Context layer (Step 1+) */}
            <g>
              <rect x={304} y={115} width={74} height={82} rx={4} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1.2} />
              <Label x={341} y={134} anchor="middle" tone={1} size={9.5} bold kind="memory">Context</Label>
              <Label x={341} y={149} anchor="middle" tone={2} size={8.5}>RAG Search</Label>
              <Label x={341} y={163} anchor="middle" tone={2} size={8.5}>Tools / DB</Label>
              <Label x={341} y={178} anchor="middle" tone={3} size={8}>+45 ms</Label>
              <Arrow x1={378} y1={156} x2={404} y2={156} kind="compute" />
            </g>

            {/* Router & Models (Step 4+) */}
            {s.step >= 4 ? (
              <g>
                {/* Router box */}
                <rect x={404} y={115} width={48} height={82} rx={4} fill="var(--comm-dim)" stroke={COLOR.comm} strokeWidth={1.2} />
                <Label x={428} y={145} anchor="middle" tone={1} size={9.5} bold kind="comm">Router</Label>
                <Label x={428} y={162} anchor="middle" tone={3} size={8}>Intent</Label>

                {/* Upper path: Small model */}
                <Arrow x1={452} y1={136} x2={476} y2={100} kind="compute" />
                <rect x={476} y={76} width={82} height={46} rx={4} fill="var(--compute-dim)" stroke={COLOR.token} strokeWidth={1.2} />
                <Label x={517} y={93} anchor="middle" tone={1} size={9.5} bold kind="token">Small 8B Tier</Label>
                <Label x={517} y={108} anchor="middle" tone={2} size={8.5}>{`${smallShare}% · $0.42/1k`}</Label>

                {/* Lower path: Large model */}
                <Arrow x1={452} y1={174} x2={476} y2={210} kind="compute" />
                <rect x={476} y={190} width={82} height={46} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.2} />
                <Label x={517} y={207} anchor="middle" tone={1} size={9.5} bold kind="compute">Large 70B Tier</Label>
                <Label x={517} y={222} anchor="middle" tone={2} size={8.5}>{`${100 - smallShare}% · $9.00/1k`}</Label>

                {/* Converge to Output Guardrails */}
                <Arrow x1={558} y1={99} x2={574} y2={136} kind="token" />
                <Arrow x1={558} y1={213} x2={574} y2={174} kind="token" />
              </g>
            ) : (
              <g>
                <rect x={404} y={115} width={90} height={82} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.5} />
                <Label x={449} y={145} anchor="middle" tone={1} size={11} bold kind="compute">Frontier LLM</Label>
                <Label x={449} y={162} anchor="middle" tone={2} size={9}>Full Context</Label>
                <Label x={449} y={178} anchor="middle" tone={3} size={8}>580 ms · $9.00/1k</Label>
                <Arrow x1={494} y1={156} x2={574} y2={156} kind="token" />
              </g>
            )}

            {/* Output Guardrails (Step 5+) */}
            {s.step >= 5 ? (
              <g>
                <rect x={574} y={115} width={56} height={82} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.2} />
                <Label x={602} y={135} anchor="middle" tone={1} size={9} bold kind="error">Output</Label>
                <Label x={602} y={149} anchor="middle" tone={2} size={8}>Schema</Label>
                <Label x={602} y={163} anchor="middle" tone={2} size={8}>De-mask</Label>
                <Label x={602} y={178} anchor="middle" tone={3} size={7.5}>+18 ms</Label>
              </g>
            ) : (
              <g>
                <rect x={574} y={135} width={56} height={42} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
                <Label x={602} y={154} anchor="middle" tone={1} size={10} bold>Output</Label>
                <Label x={602} y={168} anchor="middle" tone={3} size={8.5}>Raw</Label>
              </g>
            )}

            {/* Observability Tap (Step 6) */}
            {s.step >= 6 && (
              <g>
                <path d="M 141 197 L 141 290 L 602 290 L 602 197" fill="none" stroke={COLOR.comm} strokeWidth={1.2} strokeDasharray="4 3" opacity={0.7} />
                <rect x={220} y={272} width={280} height={36} rx={4} fill="var(--bg-card)" stroke={COLOR.comm} strokeWidth={1.2} />
                <Label x={360} y={288} anchor="middle" tone={1} size={10} bold kind="comm">Async Observability & Feedback Flywheel</Label>
                <Label x={360} y={302} anchor="middle" tone={2} size={8.5}>OpenTelemetry spans · TTFT / TPOT tracking · Cost ledger · Thumbs +/-</Label>
              </g>
            )}
          </>
        )}
      </svg>
    </Stage>
  );
}
