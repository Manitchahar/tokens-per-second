// Module 24 · Observability, Reliability & Security — new terms.
import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  counter: {
    term: 'Counter',
    def: 'A metric that only ever increases: requests served, tokens generated, retries issued. You never read it directly — you read a rate over a window, which makes it immune to process restarts and resets.',
    lesson: 'observability-reliability/metrics-dashboards',
  },
  gauge: {
    term: 'Gauge',
    def: 'A metric that goes up and down: queue depth, running requests, KV blocks in use. Read the value directly, and read it as a range because the instantaneous number is noise.',
    lesson: 'observability-reliability/metrics-dashboards',
  },
  histogram: {
    term: 'Histogram (metrics)',
    def: 'A metric that accumulates counts into fixed buckets. The only metric type that can answer "what fraction were slower than X", and therefore the only source of quantiles. Stored as cumulative counts so buckets can be pooled across replicas before a percentile is taken.',
    lesson: 'observability-reliability/metrics-dashboards',
  },
  cardinality: {
    term: 'Cardinality',
    def: 'The number of distinct time series a metric stores: buckets × the product of every label value. Multiplicative, never sublinear, so a new label multiplies the bill. 12 buckets across 2 models, 8 replicas and 250 tenants is 48,000 series for one metric.',
    lesson: 'observability-reliability/metrics-dashboards',
  },
  'error-budget': {
    term: 'Error budget',
    def: 'The amount of failure an SLO permits: for p95 TTFT < 600 ms, 5% of requests may miss it. Spending it faster than it refills is what an incident is.',
    lesson: 'observability-reliability/metrics-dashboards',
  },
  'burn-rate': {
    term: 'Burn rate',
    def: 'How fast you are spending the error budget: actual fraction missing ÷ allowed fraction. Burn 1 is on budget; burn 2.4 exhausts a 30-day budget in 13 days; burn 14.4 exhausts it in 2 hours. The standard page condition is burn over a short window AND a long one.',
    lesson: 'observability-reliability/metrics-dashboards',
  },
  span: {
    term: 'Span',
    def: 'One timed operation inside a trace: a name, a service, a start, a duration, and a parent. Spans nest, and that tree is the structure a waterfall view throws away.',
    lesson: 'observability-reliability/tracing',
  },
  'critical-path': {
    term: 'Critical path',
    def: 'The chain of spans with no gaps between them. Its total is the request latency; work that overlapped it happened in parallel and cost nothing to fix first. Compute it before optimizing anything.',
    lesson: 'observability-reliability/tracing',
  },
  'trace-id': {
    term: 'Trace ID',
    def: 'The correlation identifier minted at the edge and propagated through every hop, including engine-internal events and streaming callbacks. Without one, a user can only report a symptom; with one, you can find the exact request.',
    lesson: 'observability-reliability/tracing',
  },
  'tail-sampling': {
    term: 'Tail sampling',
    def: 'Deciding which traces to keep after the outcome is known, so slow and failed requests are kept at a higher rate than fast ones. The only sampling policy that preserves your p99 stories; head sampling throws away exactly the case you built traces for.',
    lesson: 'observability-reliability/tracing',
  },
  'admission-control': {
    term: 'Admission control',
    def: 'The O(1) checks a request must pass before taking a concurrency slot: identity and quota, request shape (max_model_len, max_tokens), whether a slot is free and the waiting room is not full, and priority. All of it runs even on the requests you reject.',
    lesson: 'observability-reliability/overload',
  },
  'load-shedding': {
    term: 'Load shedding',
    def: 'Deliberately refusing work that would otherwise degrade the service for everyone. The rule: shed when serving one request would make more requests miss the SLO than it would make succeed. Needs hysteresis or it oscillates.',
    lesson: 'observability-reliability/overload',
  },
  backpressure: {
    term: 'Backpressure',
    def: 'Making overload visible to whoever can relieve it, before you have to refuse work. Inside a replica that is the admission check failing; across replicas it is queue-aware routing sending fewer requests to a saturated one.',
    lesson: 'observability-reliability/overload',
  },
  'waiting-room': {
    term: 'Waiting room (bounded queue)',
    def: 'A hard cap on how many requests may wait for a concurrency slot. Absorbs bursts and flattens p95, but size it from a burst budget — excess requests × tolerable wait — and accept that a room big enough for a large burst also permits multi-minute waits.',
    lesson: 'observability-reliability/overload',
  },
  'retry-storm': {
    term: 'Retry storm',
    def: 'Correlated failure turning retries into an amplifier: when a backend dies, every attempt fails, so a client with 2 retries triples its offered load at the worst possible moment. Bounded with a retry budget or a circuit breaker.',
    lesson: 'observability-reliability/failures',
  },
  'retry-amplification': {
    term: 'Retry amplification',
    def: 'Expected attempts per logical request: 1 + p + p² + … + p^r. 1.05× at 5% independent failure, 1.24× at 20%, but (r + 1) against a dead backend — the same code, 57× more load.',
    lesson: 'observability-reliability/failures',
  },
  'retry-budget': {
    term: 'Retry budget',
    def: 'A client-side cap of N × (R − 1) retries over a window, so amplification is bounded by construction instead of by hope. When the budget is exhausted, return the error. Usually R ≤ 2.',
    lesson: 'observability-reliability/failures',
  },
  'circuit-breaker': {
    term: 'Circuit breaker',
    def: 'A three-state guard in front of a dependency: closed (counting failures), open (failing immediately without calling, for a cooldown), half-open (one probe call decides). Turns a correlated-failure storm into approximately zero load on a dead dependency.',
    lesson: 'observability-reliability/failures',
  },
  'graceful-degradation': {
    term: 'Graceful degradation',
    def: 'Serving a slightly worse answer instead of refusing: cap max_tokens, drop to FP8, shorten context, fall back to a smaller model. A dial with several settings, where refusal is the last one rather than the only one.',
    lesson: 'observability-reliability/failures',
  },
  'chaos-engineering': {
    term: 'Chaos testing',
    def: 'Injecting faults deliberately, one at a time and with a written hypothesis, to find out how the system actually fails before an incident does. Measures time-to-detect and time-to-recover as numbers.',
    lesson: 'observability-reliability/failures',
  },
  'thundering-herd': {
    term: 'Thundering herd',
    def: 'Every client that failed during an outage retrying at the same instant the dependency recovers, all arriving together and re-outaging it. Prevented with jittered retries and a breaker that closes gradually.',
    lesson: 'observability-reliability/failures',
  },
  'threat-model': {
    term: 'Threat model',
    def: 'A short document naming four things: the assets worth stealing, the attackers and what they can do, the trust boundaries where untrusted input becomes trusted, and the residual risk you are not fixing. Each answer should be one sentence with a number in it.',
    lesson: 'observability-reliability/security',
  },
  'weight-extraction': {
    term: 'Weight extraction',
    def: 'Reconstructing a model’s weights by querying it many times with carefully chosen inputs. Established to be possible, which is why rate limits and output caps are confidentiality controls and not only availability ones.',
    lesson: 'observability-reliability/security',
  },
  'kv-isolation': {
    term: 'KV cache isolation',
    def: 'Keeping one tenant’s KV blocks unreachable from another’s: namespace the block table by tenant, zero a block on free where it held sensitive data, and cap each tenant’s share of the pool. Without it, a public prompt can hit another tenant’s cached prefix.',
    lesson: 'observability-reliability/security',
  },
  runbook: {
    term: 'Runbook',
    def: 'The document that survives the engineer: SLO, admission limits with their derivations, timeouts and which single layer retries, the recovery gate, and the blast-radius limits — each a number with a justification, written before the incident.',
    lesson: 'observability-reliability/failures',
  },
};