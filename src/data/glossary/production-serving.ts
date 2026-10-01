import type { GlossaryEntry } from '../glossary';

// New vocabulary introduced in Module 22. Terms the learner already met in
// earlier modules (batch, ttft, slo, goodput, p99, prefill, decode, kv-cache,
// paged-attention, cuda-graph, quantization, ...) live in src/data/glossary.ts.
export const terms: Record<string, GlossaryEntry> = {
  'cold-start': {
    term: 'Cold start',
    def: 'The interval between a replica existing and that replica being able to serve a request well. It covers process launch, weight loading, graph capture and warm-up — and it is paid again on every scale-up and every deploy.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'startup-budget': {
    term: 'Startup budget',
    def: 'The time you are willing to let a replica take before it must be serving. Every step of the startup path (imports, compile, weights, capture, warm-up) has to fit inside it or be moved off the request path.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'image-layer': {
    term: 'Image layer',
    def: 'One immutable snapshot in a container image. Layers are cached and shared, so a 15 GB weights layer is downloaded once per machine and every later pull of that layer is free.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'container-image': {
    term: 'Container image',
    def: 'A filesystem snapshot plus a launch command, built once and run anywhere. It makes "the code that ran on my machine" a single artifact you can pin by digest.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'weight-conversion': {
    term: 'Weight conversion',
    def: 'Translating a training checkpoint (usually BF16 or FP32, sometimes sharded and in an exotic layout) into the exact tensor layout, dtype and quantization scheme an engine wants. Done offline, it is minutes of CPU time; done online, it is minutes of your cold start.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'safetensors': {
    term: 'safetensors',
    def: 'A simple, memory-mappable checkpoint format: a JSON header of tensor name, dtype, shape and byte offset, then raw tensor bytes. No executable pickle, so loading cannot run arbitrary code.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'cuda-graph-capture': {
    term: 'CUDA graph capture',
    def: 'Recording one full decode iteration as a single replayable CUDA graph, then launching it with one call instead of hundreds. It removes per-step launch overhead, which is why it belongs in warm-up and not on the first request.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'lazy-loading': {
    term: 'Lazy loading',
    def: 'Deferring work until first use — an import, a module, a compiled kernel, an expert weight — so startup does not pay for things a given replica may never need. Trades a first-request hiccup for a shorter startup.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'page-cache': {
    term: 'Page cache',
    def: 'The OS cache of file bytes you have read recently. A restarted process on the same machine reads a 15 GB checkpoint from RAM instead of disk, so identical restarts are far faster than genuinely new machines.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'mmap': {
    term: 'Memory mapping (mmap)',
    def: 'Pointing the process at a file and letting the OS page it in on demand, rather than reading it all into a heap allocation. Loads are near-instant because the bytes arrive while the first forward pass needs them.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  'weight-offload': {
    term: 'Weight offload',
    def: 'Keeping the least-used part of a model (usually experts in a mixture-of-experts) in host memory and streaming it to the GPU on demand. It trades bandwidth for capacity, and it is exactly where a cold start can stall.',
    lesson: 'production-serving/packaging-cold-starts',
  },
  warmup: {
    term: 'Warm-up',
    def: 'Running synthetic requests at startup until the engine matches its steady-state performance — graphs captured, caches populated, kernels selected. Until warm-up finishes the replica is a liability, so it must not receive real traffic.',
    lesson: 'production-serving/service-lifecycle',
  },
  'warmup-requests': {
    term: 'Warm-up requests',
    def: 'The throwaway prompts a server sends itself at startup. They are chosen to fill the cache shapes the real traffic will use — longest context, largest batch — because warming the wrong shape leaves the first real request slow.',
    lesson: 'production-serving/service-lifecycle',
  },
  'readiness-probe': {
    term: 'Readiness probe',
    def: 'A check that answers "should this replica receive traffic right now?". It fails during startup and warm-up and after a GPU fault, which is what keeps the load balancer from sending requests to a replica that cannot answer.',
    lesson: 'production-serving/service-lifecycle',
  },
  'liveness-probe': {
    term: 'Liveness probe',
    def: 'A check that answers "is this process alive and not hopelessly stuck?". Unlike readiness it must not flap: a liveness failure restarts the replica, so a probe sensitive to a brief GPU hiccup turns a small problem into an outage.',
    lesson: 'production-serving/service-lifecycle',
  },
  'startup-probe': {
    term: 'Startup probe',
    def: 'A liveness check that is allowed a long budget and is only armed once per container lifetime. It exists so a slow start is never mistaken for a hung process.',
    lesson: 'production-serving/service-lifecycle',
  },
  'graceful-shutdown': {
    term: 'Graceful shutdown',
    def: 'Draining a replica before killing it: stop accepting new requests, finish the in-flight ones, then exit. On SIGTERM the server receives a deadline, and exceeding it means those in-flight requests are lost.',
    lesson: 'production-serving/service-lifecycle',
  },
  draining: {
    term: 'Draining',
    def: 'The state a replica is in after it stops passing its readiness probe but keeps serving what it already has. A drain is bounded by a deadline, and requests still running at the deadline are lost — which is why drain timeout and longest request length are one decision.',
    lesson: 'production-serving/service-lifecycle',
  },
  'drain-timeout': {
    term: 'Drain timeout',
    def: 'How long a terminating replica keeps serving before it is killed. It must exceed the longest expected request or every deploy truncates the longest generations.',
    lesson: 'production-serving/service-lifecycle',
  },
  'rolling-update': {
    term: 'Rolling update',
    def: 'Replacing replicas a few at a time: add new ones, wait for them to be ready, remove old ones, repeat. It keeps capacity roughly flat and never has a moment with no healthy replica.',
    lesson: 'production-serving/safe-releases',
  },
  'max-surge': {
    term: 'Max surge',
    def: 'How many replicas above the desired count a rollout may create at once. Surge costs money and memory during the deploy; not surging is what makes a deploy slow and downtime-prone.',
    lesson: 'production-serving/safe-releases',
  },
  'max-unavailable': {
    term: 'Max unavailable',
    def: 'How many replicas below the desired count a rollout may go. One is the safe answer; two is a bet that the rest of the fleet can hold the load.',
    lesson: 'production-serving/safe-releases',
  },
  'canary': {
    term: 'Canary',
    def: 'Sending a small slice of real traffic to the new version and comparing it against the old on the metrics you already watch. Small enough that a regression costs little, big enough to measure.',
    lesson: 'production-serving/safe-releases',
  },
  'shadow-traffic': {
    term: 'Shadow traffic',
    def: 'Mirroring live requests to the new version and discarding its response, so it is exercised by real input distributions without any user seeing it. It buys correctness and latency signal at full cost of serving it.',
    lesson: 'production-serving/safe-releases',
  },
  'blue-green': {
    term: 'Blue/green',
    def: 'Two identical full fleets and a switch between them. Rollback is instant and total, at the price of running double capacity all the time.',
    lesson: 'production-serving/safe-releases',
  },
  'rollback': {
    term: 'Rollback',
    def: 'Returning to the previous version. It is only real if the previous weights are still on disk and the code is still runnable — which is why rollback is a property of the artifact, not a reaction.',
    lesson: 'production-serving/safe-releases',
  },
  'version-skew': {
    term: 'Version skew',
    def: 'Replicas serving different application versions at the same time, which happens for the whole duration of every rollout. Whether it is safe depends on whether your protocol assumes one version.',
    lesson: 'production-serving/safe-releases',
  },
  'kill-switch': {
    term: 'Kill switch',
    def: 'A control that disables a new behaviour without a deploy. If a fix requires shipping new code, you have to ship before you can stop the bleeding.',
    lesson: 'production-serving/safe-releases',
  },
  'ablation': {
    term: 'Ablation',
    def: 'Turning one thing off while holding everything else fixed, to measure that thing’s contribution. In serving it answers questions like "how much of this latency is the speculative draft, and what would we get without it?".',
    lesson: 'production-serving/safe-releases',
  },
  'model-registry': {
    term: 'Model registry',
    def: 'A versioned store of checkpoints and their metadata. Serving is only reproducible if the weights, the tokenizer, the quantization config and the code are all pinned to the same version string.',
    lesson: 'production-serving/project-productionize',
  },
  'service-level-indicator': {
    term: 'Service level indicator (SLI)',
    def: 'The measured quantity behind an SLO — p95 TTFT, goodput, error rate. An SLO is a threshold on an SLI; you need both before you can talk about meeting one.',
    lesson: 'production-serving/project-productionize',
  },
  'error-budget': {
    term: 'Error budget',
    def: 'How much SLO violation you are allowed. Spending it deliberately on a risky release is the mechanism that makes shipping fast releases and holding a stable service the same behaviour rather than a contradiction.',
    lesson: 'production-serving/project-productionize',
  },
};