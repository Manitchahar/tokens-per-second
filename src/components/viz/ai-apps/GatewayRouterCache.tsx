import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, Arrow, COLOR, fmt } from '../lib/svg';

const STEPS = [
  'The unrouted API dilemma',
  'Unified model gateway & auth',
  'Tier 1: Exact hash cache (Redis)',
  'Tier 2: Semantic vector cache & threshold',
  'Intent routing: Small vs Large',
  'Provider prompt prefix caching',
  'Blended pricing & failover circuit',
];

const CAPTIONS = [
  'Direct client-to-vendor connections create operational chaos: hardcoded keys, fragmented billing, uncoordinated rate limits, and total downtime when a single provider suffers an outage.',
  'A model gateway provides a single ingress point: tenant API keys, per-user token quotas, rate limiting, and automated failover across models and geographic regions.',
  'An exact cache hashes normalized prompts using SHA-256 into Redis. Identical queries return in 2 ms with zero GPU usage and zero token expense.',
  'A semantic cache checks query embedding similarity. At threshold τ ≥ 0.92, paraphrased queries hit reliably; below 0.88, false hits risk answering the wrong question.',
  'The intent router evaluates semantic complexity: 70% of routine queries route to a fast $0.15/M model, reserving the $3.00/M frontier model for multi-step reasoning.',
  'For prompts reaching upstream engines, static system prompts and few-shot prefixes hit provider KV caches (Module 10), slashing prefill latency and pricing by up to 80%.',
  'Blended routing reshapes unit economics: combining a 25% cache hit rate with 70% small-tier routing cuts total inference costs by more than 75% while boosting uptime to 99.99%.',
];

export default function GatewayRouterCache() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [cacheHitRate, setCacheHitRate] = useState(25);
  const [smallShare, setSmallShare] = useState(70);
  const [threshold, setThreshold] = useState(0.92);

  // Blended cost math (per 1M tokens blended):
  // Large model: $3.00/M in, $15.00/M out (avg $4.50/M tokens)
  // Small model: $0.15/M in, $0.60/M out (avg $0.20/M tokens)
  // Prefix discount: 50% on cached prompt tokens
  const cLarge = 4.50;
  const cSmall = 0.20;

  const hitFrac = s.step >= 2 ? cacheHitRate / 100 : 0;
  const sShare = s.step >= 4 ? smallShare / 100 : 0;

  const missCost = sShare * cSmall + (1 - sShare) * cLarge;
  const blendedPricePerMillion = (1 - hitFrac) * missCost;

  // Latency calculation (ms):
  const tExact = 3;
  const tSemantic = 18;
  const tSmall = 150;
  const tLarge = 620;
  const avgMissLatency = sShare * tSmall + (1 - sShare) * tLarge;
  const avgLatency = s.step >= 2
    ? hitFrac * tExact + (1 - hitFrac) * (avgMissLatency + (s.step >= 3 ? tSemantic : 0))
    : tLarge;

  const falseHitRisk = threshold < 0.88 ? 'High' : threshold < 0.91 ? 'Moderate' : 'Low';

  return (
    <Stage
      rootRef={ref}
      title="Gateway, multi-tier caching, and intent-based routing"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['comm', 'gateway & quotas'],
        ['memory', 'exact & semantic cache'],
        ['compute', 'model inference tier'],
        ['token', 'routed response'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Blended price / 1M',
              value: `$${blendedPricePerMillion.toFixed(2)}`,
              kind: blendedPricePerMillion < 1.5 ? 'token' : 'error',
              note: s.step >= 4 ? `${smallShare}% small, ${cacheHitRate}% cache` : 'unrouted baseline ($4.50)',
            },
            {
              label: 'Avg latency',
              value: `${Math.round(avgLatency)} ms`,
              kind: avgLatency < 250 ? 'token' : 'compute',
              note: hitFrac > 0 ? `${cacheHitRate}% in 3 ms` : 'full frontier model',
            },
            {
              label: 'Semantic threshold',
              value: `τ = ${threshold.toFixed(2)}`,
              kind: threshold >= 0.91 ? 'token' : 'error',
              note: `False hit risk: ${falseHitRisk}`,
            },
            {
              label: 'Cache efficiency',
              value: `${Math.round(hitFrac * 100)}%`,
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
              value={cacheHitRate}
              min={0}
              max={50}
              step={5}
              onChange={setCacheHitRate}
              format={(v) => `${v}%`}
              kind="memory"
            />
            {s.step >= 3 && (
              <Slider
                label="Semantic threshold (τ)"
                value={Math.round(threshold * 100)}
                min={80}
                max={98}
                step={2}
                onChange={(v) => setThreshold(v / 100)}
                format={(v) => `${(v / 100).toFixed(2)}`}
                kind="memory"
              />
            )}
            {s.step >= 4 && (
              <Slider
                label="Small tier share"
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
        aria-label="Model gateway diagram showing authentication, Redis exact cache, semantic vector cache, intent router, small model tier, large model tier, and fallback circuit."
      >
        {/* Step 0: Unrouted chaos */}
        {s.step === 0 && (
          <g>
            <rect x={20} y={40} width={100} height={40} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
            <Label x={70} y={64} anchor="middle" tone={1} size={10} bold>App Client A</Label>

            <rect x={20} y={130} width={100} height={40} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
            <Label x={70} y={154} anchor="middle" tone={1} size={10} bold>App Client B</Label>

            <rect x={20} y={220} width={100} height={40} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
            <Label x={70} y={244} anchor="middle" tone={1} size={10} bold>App Client C</Label>

            <Arrow x1={120} y1={60} x2={480} y2={60} kind="compute" width={1.5} />
            <Arrow x1={120} y1={150} x2={480} y2={150} kind="compute" width={1.5} />
            <Arrow x1={120} y1={240} x2={480} y2={240} kind="compute" width={1.5} />

            <rect x={480} y={40} width={140} height={50} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.2} />
            <Label x={550} y={64} anchor="middle" tone={1} size={10.5} bold kind="compute">Provider 1 (OpenAI)</Label>
            <Label x={550} y={78} anchor="middle" tone={3} size={8.5}>Separate Key · Rate Limit Risk</Label>

            <rect x={480} y={130} width={140} height={50} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.2} />
            <Label x={550} y={154} anchor="middle" tone={1} size={10.5} bold kind="compute">Provider 2 (Anthropic)</Label>
            <Label x={550} y={168} anchor="middle" tone={3} size={8.5}>Separate Key · No Failover</Label>

            <rect x={480} y={220} width={140} height={50} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.2} />
            <Label x={550} y={244} anchor="middle" tone={1} size={10.5} bold kind="compute">Self-Hosted vLLM</Label>
            <Label x={550} y={258} anchor="middle" tone={3} size={8.5}>No Cache · Full GPU Load</Label>

            <rect x={160} y={95} width={280} height={50} rx={4} fill="var(--error-dim)" stroke={COLOR.error} strokeWidth={1} />
            <Label x={300} y={116} anchor="middle" tone={1} size={10} bold kind="error">Direct Connection Failure Modes</Label>
            <Label x={300} y={132} anchor="middle" tone={2} size={8.5}>Fragmented billing, no caching, 429 cascading outages</Label>
          </g>
        )}

        {/* Step >= 1: Unified Architecture */}
        {s.step >= 1 && (
          <g>
            {/* Ingress Client */}
            <rect x={16} y={135} width={64} height={50} rx={4} fill="var(--bg-card)" stroke="var(--rule)" strokeWidth={1} />
            <Label x={48} y={157} anchor="middle" tone={1} size={10.5} bold>Clients</Label>
            <Label x={48} y={173} anchor="middle" tone={3} size={8.5}>Single API</Label>

            <Arrow x1={80} y1={160} x2={104} y2={160} kind="comm" width={2} />

            {/* Model Gateway */}
            <rect x={104} y={105} width={76} height={110} rx={4} fill="var(--comm-dim)" stroke={COLOR.comm} strokeWidth={1.5} />
            <Label x={142} y={125} anchor="middle" tone={1} size={10} bold kind="comm">Gateway</Label>
            <Label x={142} y={141} anchor="middle" tone={2} size={8.5}>Auth & Keys</Label>
            <Label x={142} y={156} anchor="middle" tone={2} size={8.5}>Token Bucket</Label>
            <Label x={142} y={171} anchor="middle" tone={2} size={8.5}>Cost Ledger</Label>
            <Label x={142} y={188} anchor="middle" tone={3} size={8}>Failover Circuit</Label>

            {/* Tier 1: Exact Cache (Step >= 2) */}
            {s.step >= 2 ? (
              <g>
                <Arrow x1={142} y1={105} x2={142} y2={58} kind="memory" width={1.5} />
                <rect x={96} y={18} width={92} height={40} rx={4} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1.2} />
                <Label x={142} y={34} anchor="middle" tone={1} size={9} bold kind="memory">Exact Cache</Label>
                <Label x={142} y={48} anchor="middle" tone={2} size={8}>SHA-256 (3 ms)</Label>

                {/* Direct cache hit bypass */}
                <path d="M 188 38 L 575 38 L 575 135" fill="none" stroke={COLOR.token} strokeWidth={1.2} strokeDasharray="3 3" />
                <Label x={380} y={32} anchor="middle" tone={1} size={8.5} kind="token">Hit: 0 ms GPU / $0 token cost</Label>

                <Arrow x1={180} y1={160} x2={208} y2={160} kind="memory" />
              </g>
            ) : (
              <Arrow x1={180} y1={160} x2={310} y2={160} kind="compute" />
            )}

            {/* Tier 2: Semantic Cache (Step >= 3) */}
            {s.step >= 3 ? (
              <g>
                <rect x={208} y={105} width={82} height={110} rx={4} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1.2} />
                <Label x={249} y={125} anchor="middle" tone={1} size={9.5} bold kind="memory">Semantic Cache</Label>
                <Label x={249} y={141} anchor="middle" tone={2} size={8.5}>Vector Embed</Label>
                <Label x={249} y={158} anchor="middle" tone={1} size={8.5} bold kind={threshold >= 0.91 ? 'token' : 'error'}>
                  {`τ ≥ ${threshold.toFixed(2)}`}
                </Label>
                <Label x={249} y={174} anchor="middle" tone={3} size={8}>+18 ms lookup</Label>
                <Label x={249} y={190} anchor="middle" tone={2} size={7.5}>{`Risk: ${falseHitRisk}`}</Label>

                <Arrow x1={290} y1={160} x2={318} y2={160} kind="comm" />
              </g>
            ) : (
              s.step >= 2 && <Arrow x1={180} y1={160} x2={318} y2={160} kind="comm" />
            )}

            {/* Intent Router (Step >= 4) */}
            {s.step >= 4 ? (
              <g>
                <rect x={318} y={105} width={74} height={110} rx={4} fill="var(--comm-dim)" stroke={COLOR.comm} strokeWidth={1.2} />
                <Label x={355} y={128} anchor="middle" tone={1} size={10} bold kind="comm">Router</Label>
                <Label x={355} y={146} anchor="middle" tone={2} size={8.5}>Intent / Complexity</Label>
                <Label x={355} y={164} anchor="middle" tone={2} size={8.5}>Classifier</Label>
                <Label x={355} y={185} anchor="middle" tone={3} size={8}>5 ms delay</Label>

                {/* Upper path: Small model */}
                <Arrow x1={392} y1={135} x2={424} y2={95} kind="compute" width={1.8} />
                <rect x={424} y={70} width={120} height={50} rx={4} fill="var(--compute-dim)" stroke={COLOR.token} strokeWidth={1.5} />
                <Label x={484} y={90} anchor="middle" tone={1} size={10} bold kind="token">Small 8B Tier</Label>
                <Label x={484} y={106} anchor="middle" tone={2} size={8.5}>{`${smallShare}% traffic · $0.20/M · 150 ms`}</Label>

                {/* Lower path: Large model */}
                <Arrow x1={392} y1={185} x2={424} y2={225} kind="compute" width={1.8} />
                <rect x={424} y={200} width={120} height={50} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.5} />
                <Label x={484} y={220} anchor="middle" tone={1} size={10} bold kind="compute">Frontier 70B Tier</Label>
                <Label x={484} y={236} anchor="middle" tone={2} size={8.5}>{`${100 - smallShare}% traffic · $4.50/M · 620 ms`}</Label>

                {/* Converge to client */}
                <Arrow x1={544} y1={95} x2={575} y2={145} kind="token" />
                <Arrow x1={544} y1={225} x2={575} y2={175} kind="token" />
              </g>
            ) : (
              <g>
                <rect x={318} y={115} width={120} height={90} rx={4} fill="var(--compute-dim)" stroke={COLOR.compute} strokeWidth={1.5} />
                <Label x={378} y={145} anchor="middle" tone={1} size={11} bold kind="compute">Frontier LLM</Label>
                <Label x={378} y={165} anchor="middle" tone={2} size={9}>100% of misses</Label>
                <Label x={378} y={182} anchor="middle" tone={3} size={8}>$4.50/M avg · 620 ms</Label>
                <Arrow x1={438} y1={160} x2={575} y2={160} kind="token" />
              </g>
            )}

            {/* Ingress Client Response */}
            <rect x={575} y={135} width={48} height={50} rx={4} fill="var(--bg-card)" stroke={COLOR.token} strokeWidth={1.2} />
            <Label x={599} y={157} anchor="middle" tone={1} size={10} bold kind="token">Client</Label>
            <Label x={599} y={173} anchor="middle" tone={2} size={8}>Reply</Label>

            {/* Provider Prompt Caching Callout (Step 5) */}
            {s.step === 5 && (
              <g>
                <rect x={140} y={260} width={400} height={56} rx={4} fill="var(--memory-dim)" stroke={COLOR.memory} strokeWidth={1.2} />
                <Label x={340} y={280} anchor="middle" tone={1} size={10.5} bold kind="memory">Provider Prompt Caching (Module 10)</Label>
                <Label x={340} y={298} anchor="middle" tone={2} size={9}>Static system prompt tokens hit provider KV caches, cutting prefill cost by up to 50–80%.</Label>
              </g>
            )}

            {/* Blended Cost & SLA Banner (Step 6) */}
            {s.step === 6 && (
              <g>
                <rect x={100} y={260} width={480} height={56} rx={4} fill="var(--bg-card)" stroke={COLOR.token} strokeWidth={1.2} />
                <Label x={340} y={280} anchor="middle" tone={1} size={10.5} bold kind="token">
                  {`Blended System Economics: $${blendedPricePerMillion.toFixed(2)} / 1M tokens`}
                </Label>
                <Label x={340} y={298} anchor="middle" tone={2} size={9}>
                  {`75% cost reduction vs naive $4.50 baseline · P50 latency drops from 620 ms to ${Math.round(avgLatency)} ms.`}
                </Label>
              </g>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
