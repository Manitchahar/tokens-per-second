import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, Arrow, COLOR, fmt } from '../lib/svg';

const STEPS = [
  'The unprotected pipeline',
  'Input: Reversible PII masking',
  'Input: Topic & injection screen',
  'Model generation with surrogates',
  'Output: Format & schema checks',
  'Output: Factuality & de-masking',
  'The streaming dilemma: chunk buffers',
];

const CAPTIONS = [
  'A naive pipeline sends raw user text into the model and streams raw tokens back. PII leaks upstream, injections execute unchecked, and malformed outputs break downstream clients.',
  'Reversible PII masking replaces sensitive entities with surrogate tokens like [SSN_1] using a request-scoped map, ensuring raw credentials never touch the model.',
  'A fast classification screen checks for jailbreak signatures and prohibited topics in 18 ms, short-circuiting toxic or malicious queries before GPU cycles are burned.',
  'The model reasons over the masked prompt. Because surrogates preserve syntactic role and entity relationships, output quality remains high without exposing private data.',
  'Output rails validate that responses adhere to JSON schema or grammatical rules. A schema parse failure triggers an immediate structured retry or safe fallback.',
  'Factuality checkers verify claims against retrieved context chunks, and the de-masking engine substitutes original values back into the final user-facing response.',
  'Streaming requires chunk buffering: rather than holding the entire response, tokens buffer to punctuation boundaries (15–25 tokens), balancing safety checks with low TTFT.',
];

export default function GuardrailsFlow() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [bufferSize, setBufferSize] = useState(20);
  const [tpot, setTpot] = useState(25); // ms per token

  // Latency arithmetic:
  const tPiiMask = s.step >= 1 ? 12 : 0;
  const tInjection = s.step >= 2 ? 16 : 0;
  const tInputTotal = tPiiMask + tInjection;

  const tLlmPrefill = 120;
  const tLlmDecodeTotal = 150 * (tpot / 1000) * 1000; // 150 tokens
  const tOutputSchema = s.step >= 4 ? 14 : 0;
  const tFactuality = s.step >= 5 ? 22 : 0;
  const tOutputTotal = tOutputSchema + tFactuality;

  // In streaming step 6: buffer delay = bufferSize * tpot
  const streamBufferDelay = s.step >= 6 ? bufferSize * tpot : 0;
  const perceivedTtft = tInputTotal + tLlmPrefill + streamBufferDelay;
  const totalLatency = tInputTotal + tLlmPrefill + tLlmDecodeTotal + tOutputTotal;

  return (
    <Stage
      rootRef={ref}
      title="Input and output guardrails with streaming buffers"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['error', 'guardrails (input/output)'],
        ['memory', 'PII map & context'],
        ['compute', 'model prefill & decode'],
        ['token', 'verified tokens'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Input rail latency',
              value: `${tInputTotal} ms`,
              kind: tInputTotal > 0 ? 'error' : 'idle',
              note: s.step >= 2 ? 'PII mask (12ms) + filter (16ms)' : 'none',
            },
            {
              label: 'Perceived TTFT',
              value: `${Math.round(perceivedTtft)} ms`,
              kind: perceivedTtft < 250 ? 'token' : 'compute',
              note: s.step >= 6 ? `includes ${bufferSize}-token buffer` : 'prefill + input rails',
            },
            {
              label: 'Output rail latency',
              value: `${tOutputTotal} ms`,
              kind: tOutputTotal > 0 ? 'error' : 'idle',
              note: s.step >= 5 ? 'schema (14ms) + fact check (22ms)' : 'none',
            },
            {
              label: 'End-to-end latency',
              value: `${Math.round(totalLatency)} ms`,
              kind: 'compute',
              note: '150 tokens generated',
            },
          ]}
        />
      }
      controls={
        s.step >= 6 ? (
          <>
            <Slider
              label="Streaming buffer size"
              value={bufferSize}
              min={5}
              max={50}
              step={5}
              onChange={setBufferSize}
              format={(v) => `${v} tokens`}
              kind="memory"
            />
            <Slider
              label="TPOT (ms / token)"
              value={tpot}
              min={10}
              max={60}
              step={5}
              onChange={setTpot}
              format={(v) => `${v} ms`}
              kind="compute"
            />
          </>
        ) : undefined
      }
      {...s}
    >
      <svg
        viewBox="0 0 640 330"
        role="img"
        aria-label="Guardrails architecture showing input PII masking, safety classifier, LLM inference, output schema validation, factuality check, and streaming chunk buffer."
      >
        {/* Step 0: Unprotected */}
        {s.step === 0 && (
          <g>
            <rect x={24} y={135} width={90} height={50} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
            <Label x={69} y={157} anchor="middle" tone={1} size={11} bold>Raw Input</Label>
            <Label x={69} y={173} anchor="middle" tone={3} size={9}>SSN, Names, Injection</Label>

            <Arrow x1={114} y1={160} x2={250} y2={160} kind="error" width={2} />

            <rect x={250} y={120} width={140} height={80} rx={6} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.5} />
            <Label x={320} y={152} anchor="middle" tone={1} size={13} bold kind="compute">Model Call</Label>
            <Label x={320} y={170} anchor="middle" tone={2} size={10}>No input/output checks</Label>
            <Label x={320} y={185} anchor="middle" tone={3} size={8.5}>Raw exposure</Label>

            <Arrow x1={390} y1={160} x2={520} y2={160} kind="error" width={2} />

            <rect x={520} y={135} width={96} height={50} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.2} />
            <Label x={568} y={157} anchor="middle" tone={1} size={11} bold kind="error">Raw Stream</Label>
            <Label x={568} y={173} anchor="middle" tone={2} size={8.5}>Hallucinations & leaks</Label>

            <rect x={100} y={245} width={440} height={44} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1} />
            <Label x={320} y={264} anchor="middle" tone={1} size={11} bold kind="error">Risks of Unguarded Model Access</Label>
            <Label x={320} y={279} anchor="middle" tone={2} size={9.5}>Data exfiltration, PII compliance fines, broken JSON schema, and prompt injection exploits</Label>
          </g>
        )}

        {/* Steps >= 1: Guardrails Pipeline */}
        {s.step >= 1 && (
          <g>
            {/* Input Side */}
            <rect x={16} y={135} width={70} height={50} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
            <Label x={51} y={157} anchor="middle" tone={1} size={10.5} bold>Client</Label>
            <Label x={51} y={172} anchor="middle" tone={3} size={8.5}>Prompt</Label>

            <Arrow x1={86} y1={160} x2={106} y2={160} kind="error" />

            {/* PII Masking Box */}
            <rect x={106} y={115} width={76} height={90} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.2} />
            <Label x={144} y={135} anchor="middle" tone={1} size={9.5} bold kind="error">PII Mask</Label>
            <Label x={144} y={150} anchor="middle" tone={2} size={8.5}>Regex & NER</Label>
            <Label x={144} y={165} anchor="middle" tone={2} size={8}>SSN → [SSN_1]</Label>
            <Label x={144} y={180} anchor="middle" tone={3} size={8}>+12 ms</Label>

            {/* PII Dictionary Map (Memory) */}
            <Arrow x1={144} y1={115} x2={144} y2={58} kind="memory" />
            <rect x={90} y={16} width={108} height={42} rx={4} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1.2} />
            <Label x={144} y={32} anchor="middle" tone={1} size={9} bold kind="memory">Request PII Map</Label>
            <Label x={144} y={47} anchor="middle" tone={2} size={8}>{`{"[SSN_1]": "000-12-..."}`}</Label>

            <Arrow x1={182} y1={160} x2={202} y2={160} kind="error" />

            {/* Topic & Injection Screen (Step >= 2) */}
            {s.step >= 2 ? (
              <g>
                <rect x={202} y={115} width={76} height={90} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.2} />
                <Label x={240} y={135} anchor="middle" tone={1} size={9.5} bold kind="error">Safety Filter</Label>
                <Label x={240} y={150} anchor="middle" tone={2} size={8.5}>Injection Screen</Label>
                <Label x={240} y={165} anchor="middle" tone={2} size={8}>Topic Policy</Label>
                <Label x={240} y={180} anchor="middle" tone={3} size={8}>+16 ms</Label>
                <Arrow x1={278} y1={160} x2={298} y2={160} kind="compute" />
              </g>
            ) : (
              <Arrow x1={182} y1={160} x2={298} y2={160} kind="compute" />
            )}

            {/* Model Generation Box */}
            <rect x={298} y={115} width={90} height={90} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.5} />
            <Label x={343} y={138} anchor="middle" tone={1} size={11} bold kind="compute">Model Run</Label>
            <Label x={343} y={154} anchor="middle" tone={2} size={9}>Prefill: 120 ms</Label>
            <Label x={343} y={169} anchor="middle" tone={2} size={8.5}>Masked Tokens</Label>
            <Label x={343} y={184} anchor="middle" tone={3} size={8}>{`150 tok @ ${tpot}ms`}</Label>

            {/* Output Side */}
            {s.step >= 4 ? (
              <g>
                <Arrow x1={388} y1={160} x2={410} y2={160} kind="error" />
                <rect x={410} y={115} width={78} height={90} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.2} />
                <Label x={449} y={135} anchor="middle" tone={1} size={9.5} bold kind="error">Schema Rail</Label>
                <Label x={449} y={150} anchor="middle" tone={2} size={8.5}>JSON Parser</Label>
                <Label x={449} y={165} anchor="middle" tone={2} size={8}>Pydantic Validate</Label>
                <Label x={449} y={180} anchor="middle" tone={3} size={8}>+14 ms</Label>
              </g>
            ) : (
              <Arrow x1={388} y1={160} x2={508} y2={160} kind="token" />
            )}

            {/* Factuality & De-masking (Step >= 5) */}
            {s.step >= 5 ? (
              <g>
                <Arrow x1={488} y1={160} x2={510} y2={160} kind="error" />
                <rect x={510} y={115} width={80} height={90} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1.2} />
                <Label x={550} y={135} anchor="middle" tone={1} size={9.5} bold kind="error">Verify & De-mask</Label>
                <Label x={550} y={150} anchor="middle" tone={2} size={8.5}>Factuality Check</Label>
                <Label x={550} y={165} anchor="middle" tone={2} size={8}>Map [SSN_1] back</Label>
                <Label x={550} y={180} anchor="middle" tone={3} size={8}>+22 ms</Label>

                {/* De-mask connection from top map */}
                <path d="M 198 37 L 550 37 L 550 115" fill="none" stroke={COLOR.memory} strokeWidth={1.2} strokeDasharray="3 3" />
                <Label x={374} y={30} anchor="middle" tone={2} size={8.5} kind="memory">Re-inject original PII securely</Label>

                <Arrow x1={590} y1={160} x2={608} y2={160} kind="token" />
              </g>
            ) : (
              s.step >= 4 && <Arrow x1={488} y1={160} x2={608} y2={160} kind="token" />
            )}

            {/* Final Client Output */}
            <rect x={608} y={135} width={28} height={50} rx={3} fill="var(--bg-card)" stroke={COLOR.token} strokeWidth={1} />
            <Label x={622} y={164} anchor="middle" tone={1} size={9} bold kind="token">OK</Label>

            {/* Step 6: Streaming Buffer Display */}
            {s.step >= 6 && (
              <g>
                <rect x={80} y={235} width={480} height={66} rx={4} fill="var(--bg-card)" stroke={COLOR.token} strokeWidth={1.2} />
                <Label x={320} y={254} anchor="middle" tone={1} size={11} bold kind="token">
                  {`The Streaming Buffer Dilemma: Sliding Window of ${bufferSize} Tokens`}
                </Label>
                <Label x={320} y={271} anchor="middle" tone={2} size={9.5}>
                  {`Token buffering adds ${bufferSize} × ${tpot} ms = ${bufferSize * tpot} ms delay before the first verified sentence reaches the user.`}
                </Label>
                <Label x={320} y={287} anchor="middle" tone={3} size={8.5}>
                  Balances interactive TTFT with sentence-level safety checks before emission.
                </Label>
              </g>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
