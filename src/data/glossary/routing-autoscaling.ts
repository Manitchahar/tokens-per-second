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
};