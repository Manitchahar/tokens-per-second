// Module 23 · Routing, Autoscaling & Orchestration — new terms.
import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  'load-balancer': {
    term: 'Load balancer',
    def: 'The component in front of a fleet that decides which replica serves each request. For a web server it can count connections; for LLM inference it must score replicas on the resource that actually runs out — KV cache tokens.',
    lesson: 'routing-autoscaling/load-balancing',
  },
  'round-robin': {
    term: 'Round robin',
    def: 'Dealing requests to replicas in strict rotation. Perfectly fair by count and completely stateless, but blind: an 8,000-token prompt and a 40-token greeting are treated identically.',
    lesson: 'routing-autoscaling/load-balancing',
  },
  'least-connections': {
    term: 'Least connections (least-queued)',
    def: 'Routing each request to the replica with the fewest waiting requests. Better than round robin because it reacts to load, but it counts requests rather than memory, so long prompts still hide behind short ones.',
    lesson: 'routing-autoscaling/load-balancing',
  },
  'least-kv': {
    term: 'Least-KV-tokens routing',
    def: 'Scoring each replica on the KV cache tokens it currently holds and sending the request to the least loaded. The robust default for LLM fleets, because KV is the resource that actually runs out.',
    lesson: 'routing-autoscaling/load-balancing',
  },
  'power-of-two-choices': {
    term: 'Power of two choices',
    def: 'Sampling two replicas at random and sending the request to the better one. Balances nearly as well as scoring every replica but costs two comparisons instead of N, which is why modern L7 balancers default to it.',
    lesson: 'routing-autoscaling/load-balancing',
  },
  'session-affinity': {
    term: 'Session affinity',
    def: 'Pinning a conversation to one replica so its prefix stays warm. Raises the cache hit rate but has no load term in its score, so a popular conversation hot-spots a single GPU.',
    lesson: 'routing-autoscaling/load-balancing',
  },
  'kv-spread': {
    term: 'KV spread (imbalance)',
    def: 'Busiest replica’s KV load divided by the mean replica’s. 1.0 is perfectly even; 4.0 means one GPU is carrying the fleet. The early-warning metric that catches hot replicas before the tail does.',
    lesson: 'routing-autoscaling/load-balancing',
  },
  'cache-aware-routing': {
    term: 'Cache-aware routing',
    def: 'Routing on how much of a request’s prefix each replica already holds, minus how full that replica is. Combines the reuse of affinity with the balance of load balancing through a tunable weight.',
    lesson: 'routing-autoscaling/cache-aware-routing',
  },
  'prefix-affinity': {
    term: 'Prefix affinity',
    def: 'The tendency of a request to be served by the replica that already holds its prompt prefix. The reason a per-replica prefix cache is worth anything at all: a warm cache on the wrong GPU is unreachable.',
    lesson: 'routing-autoscaling/cache-aware-routing',
  },
  'hot-replica': {
    term: 'Hot replica',
    def: 'A replica carrying far more than its share of load, usually caused by affinity without a load penalty. Invisible in a fleet average while producing the worst latency in the system for the users who matter most.',
    lesson: 'routing-autoscaling/cache-aware-routing',
  },
  'load-weight': {
    term: 'Load weight (α)',
    def: 'The unitless knob in a cache-aware routing score that trades a warm prefix against an even load. Zero is pure affinity (hot-spots); large values ignore the cache and become least-loaded. Both score terms must be normalised to 0–1 for it to mean anything.',
    lesson: 'routing-autoscaling/cache-aware-routing',
  },
  'prefix-hint': {
    term: 'Cache hint',
    def: 'The router’s belief about which prefixes a replica holds, obtained by asking the engine, deriving it from routing history, or bloom-filtering block hashes. A hint is itself a cache: it goes stale on restart and eviction, so routing must fall back to least-loaded when it is missing.',
    lesson: 'routing-autoscaling/cache-aware-routing',
  },
  'capacity': {
    term: 'Capacity per replica',
    def: 'The requests per second one replica retires while holding its SLO, equal to its users divided by their time in the system. Derived from memory and timing, so it can be computed before anything is deployed.',
    lesson: 'routing-autoscaling/autoscaling',
  },
  'time-in-system': {
    term: 'Time in system',
    def: 'How long a request occupies its replica: generation time (output tokens × TPOT) plus queueing and prefill. Little’s law run forward converts this and the replica’s user capacity into requests per second.',
    lesson: 'routing-autoscaling/load-balancing',
  },
  'goodput-cliff': {
    term: 'The goodput cliff',
    def: 'The sharp drop in the fraction of requests meeting their SLO once a fleet saturates, while throughput stays roughly flat. Past saturation extra arrivals do not add progress, they only add queue wait.',
    lesson: 'routing-autoscaling/autoscaling',
  },
  'autoscaling-signal': {
    term: 'Autoscaling signal',
    def: 'The metric an autoscaler watches. CPU is useless for inference because it is pinned at every concurrency; queue depth works because it leads latency, since latency is mostly queueing delay.',
    lesson: 'routing-autoscaling/autoscaling',
  },
  'slo-derived-concurrency': {
    term: 'SLO-derived concurrency',
    def: 'Scaling to hold the fleet-wide count of concurrent users near a target chosen so each replica stays inside its SLO. Self-correcting: if latency degrades, the admissible concurrency drops and the scaler sees it is over target.',
    lesson: 'routing-autoscaling/autoscaling',
  },
  'autoscaling-window': {
    term: 'Autoscaling window',
    def: 'The period over which the scaling signal is averaged. It must be several times the step time and longer than a replica’s start-up, or the scaler reacts to noise it caused itself.',
    lesson: 'routing-autoscaling/autoscaling',
  },
  'scaling-oscillation': {
    term: 'Scaling oscillation',
    def: 'A scaler that adds replicas, sees the metric recover, removes them, and sees it worsen again. Fixed by a longer window and asymmetric rates — scale out fast, scale in slowly — plus minimum and maximum bounds.',
    lesson: 'routing-autoscaling/autoscaling',
  },
  'scale-to-zero': {
    term: 'Scale to zero',
    def: 'Removing every replica when traffic stops, so an idle service costs nothing. It makes the next request pay a full cold start (Module 22), which is why most production fleets keep a small minimum replica count.',
    lesson: 'routing-autoscaling/autoscaling',
  },
  'draining': {
    term: 'Draining a replica',
    def: 'Stopping new requests reaching a replica, letting its in-flight ones finish, and only then removing it. Necessary because a replica holds real users’ KV cache; killing it interrupts their conversations.',
    lesson: 'routing-autoscaling/autoscaling',
  },
  'kserve': {
    term: 'KServe',
    def: 'A Kubernetes custom resource and controller that turns a model declaration into a deployment, autoscaler, metrics adapter and ingress. Trades control for far less code to own.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'ray-serve': {
    term: 'Ray Serve',
    def: 'A serving framework where a request is the primitive and a deployment is a Python class. Ray owns placement, batching, routing and scale-to-zero, giving more control at the cost of more code to maintain.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'llm-d': {
    term: 'llm-d',
    def: 'A scheduler-aware control plane for LLM inference on Kubernetes. Its premise is that placement — cache locality, prefix affinity, prefill/decode balance — is a scheduling problem generic Kubernetes cannot express, so it belongs in its own layer. Young at the time of writing.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'inference-replica': {
    term: 'Inference replica',
    def: 'One serving process holding a model’s weights and its share of KV cache, usually pinned to one GPU. It breaks four Kubernetes pod assumptions: small state, fast start, cheap replacement, and one process per GPU.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'startup-probe': {
    term: 'Startup probe',
    def: 'A Kubernetes probe that tolerates a long start without counting it as a hang. Essential for inference pods, which take minutes to load weights; using a liveness probe instead produces a restart loop that never serves a request.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'termination-grace': {
    term: 'Termination grace period',
    def: 'How long Kubernetes waits after asking a pod to stop before killing it. An inference pod needs a long one (minutes) so it can drain in-flight requests whose KV cache lives in its memory.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'admission-control': {
    term: 'Admission control',
    def: 'Deciding which requests to accept, queue or reject when a replica runs out of KV memory. Whether a fleet sheds load or simply grows an unbounded queue is a deliberate choice, and the queue is where p99 goes to die.',
    lesson: 'routing-autoscaling/lab-hold-the-slo',
  },
  'exact-match-cache': {
    term: 'Exact-match response cache',
    def: 'A cache that stores a finished answer under a hash of the normalised request — model and version, template, system prompt, tools, sampling settings, locale and tenant. A hit returns the stored response with no prefill and no decode, but any volatile field in the key turns every request into a miss.',
    lesson: 'routing-autoscaling/semantic-caching',
  },
  'semantic-cache': {
    term: 'Semantic cache',
    def: 'A cache that stores a question embedding and its answer, then returns that answer when a new question is close enough in embedding space. It can match different wordings, and it saves the whole LLM call — but an approximate match can return a confident answer to a different question.',
    lesson: 'routing-autoscaling/semantic-caching',
  },
  'false-hit': {
    term: 'False hit',
    def: 'A semantic-cache hit whose stored answer is wrong for the new request. It costs the wrong-answer penalty rather than saving a call, which is why a cache is tuned for precision among hits, not for the highest possible hit rate.',
    lesson: 'routing-autoscaling/semantic-caching',
  },
  'similarity-threshold': {
    term: 'Similarity threshold',
    def: 'The minimum cosine similarity at which a stored question is considered a match. Lowering it widens the match region and raises the hit rate, but also raises the false-hit rate; the right value comes from a labelled evaluation set and the economics of a wrong answer, not from a universal constant.',
    lesson: 'routing-autoscaling/semantic-caching',
  },
  'pii-masking': {
    term: 'PII masking',
    def: 'Replacing personally identifiable information — names, emails, phone numbers, card and account numbers — with stable placeholders before text is embedded, cached or logged. A named-entity model plus regular expressions is the usual tool; the raw value must never reach the vector index or the stored answer.',
    lesson: 'routing-autoscaling/semantic-caching',
  },
  'cache-scoping': {
    term: 'Cache scoping',
    def: 'Putting every dimension that changes the answer into the cache key or the search filter: tenant, locale, model and model version, prompt template, embedding model, tool permissions and index version. Tenant isolation must happen inside the vector search, not as a cleanup afterwards.',
    lesson: 'routing-autoscaling/semantic-caching',
  },
  'semantic-routing': {
    term: 'Semantic routing',
    def: 'Using an embedding or a small classifier to score a request and choose which model or mode serves it — a small model for easy queries, a large model or reasoning mode for hard ones. It sits above replica-level routing: this decision is which model, not which copy of one model.',
    lesson: 'routing-autoscaling/semantic-caching',
  },
  'shadow-mode': {
    term: 'Shadow mode',
    def: 'Running a new cache or router on real traffic while still serving the normal answer, logging what the new component would have done. It exposes would-be hits and would-be false hits with no user impact, and it is the cheapest way to earn the right to turn the component on.',
    lesson: 'routing-autoscaling/semantic-caching',
  },
  'api-gateway': {
    term: 'API gateway',
    def: 'The public front door of a shared inference service. It authenticates the caller, finds their tier, checks their quota and rate limit, validates the request size and writes the audit record, then admits or rejects the request before any GPU work. It decides whether a request may run; the router behind it decides where.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'token-bucket': {
    term: 'Token bucket',
    def: 'A rate limiter that holds up to b tokens and refills at r tokens per second; each request removes its cost when admitted and is rejected if the bucket is short. It allows a burst of b and a long-run rate of r, so the most a tenant can spend in T seconds is b + rT. A 60,000 tokens/min plan with a 20,000 burst allows 80,000 in the worst minute.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'token-quota': {
    term: 'Token quota',
    def: 'A tenant limit counted in tokens (prompt plus output) rather than requests, because a request can be 40 tokens or 40,000 and the GPU spends tokens. The gateway usually reserves prompt plus max_tokens at admission and refunds the unused output when the request finishes.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'audit-record': {
    term: 'Audit record',
    def: 'One log entry per request: tenant, model and version, prompt and output token counts, the admission decision and the latency. Billing, quota enforcement and incident review all read it, so the tokens are counted once and feed all three. It holds counts and IDs, not raw prompts.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'model-multiplexing': {
    term: 'Model multiplexing',
    def: 'Serving many models from fewer replicas: each replica keeps a few models loaded, requests carry a model id, and the router prefers a replica that already holds the requested model. Ray Serve provides it with @serve.multiplexed(max_num_models_per_replica=…) and serve.get_multiplexed_model_id(); each replica evicts its least recently used model.',
    lesson: 'routing-autoscaling/orchestration',
  },
  'model-selection-router': {
    term: 'Model-selection router',
    def: 'The function that turns the model a client asked for into the one that will run: resolve aliases, enforce the tenant’s allow-list, and apply weighted canary splits (hashed on the user so a conversation stays on one version).',
    lesson: 'routing-autoscaling/orchestration',
  },
  'lru-eviction': {
    term: 'LRU eviction',
    def: 'Least recently used: when a cache is full, remove the entry that was used longest ago. For a model cache, an OrderedDict does it in two calls: move_to_end on every hit, popitem(last=False) to evict, and evict before loading so the newcomer never has to coexist with its victim.',
    lesson: 'routing-autoscaling/orchestration',
  },
};
