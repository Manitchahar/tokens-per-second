import type { GlossaryEntry } from '../glossary';

// Terms this module introduces that are not in the shared glossary.
// Latency metrics already defined there (ttft, tpot, itl, slo, goodput, p99,
// throughput, concurrency) are intentionally not repeated.

export const terms: Record<string, GlossaryEntry> = {
  'service-time': {
    term: 'Service time',
    def: 'How much GPU time one request actually consumes: its prefill plus one decode step per output token. Capacity is 1 ÷ service time, so capacity depends on your workload, not only on the GPU.',
    lesson: 'benchmarking/throughput-concurrency',
  },
  'littles-law': {
    term: 'Little’s law',
    def: 'L = λ × W: the mean number of requests in a system equals the arrival rate times the mean time each spends there. Exact for any stable queue, and it turns a latency target into a concurrency requirement.',
    lesson: 'benchmarking/throughput-concurrency',
  },
  'offered-load': {
    term: 'Offered load',
    def: 'The arrival rate of requests, expressed as a fraction of what the system can retire. Below 100% the queue drains; above it the queue grows without bound and latency diverges.',
    lesson: 'benchmarking/throughput-concurrency',
  },
  utilization: {
    term: 'Utilization (ρ)',
    def: 'Arrivals divided by service rate: how busy the system is. Run at 50–70%. Near 100% the M/M/1 mean queue wait is ρ/(1−ρ) × service time, so latency explodes faster than throughput improves.',
    lesson: 'benchmarking/throughput-concurrency',
  },
  'queueing-delay': {
    term: 'Queueing delay',
    def: 'Time a request spends waiting for the GPU before its first step. Often the largest term in TTFT, and invisible to a kernel profiler because nothing is running on the GPU while it waits.',
    lesson: 'benchmarking/latency-metrics',
  },
  'end-to-end': {
    term: 'End-to-end latency',
    def: 'Total time from sending a request to receiving its last token. Exactly TTFT + (n−1) × TPOT, so it is not an independent measurement but the sum of the other two plus the output-length term.',
    lesson: 'benchmarking/latency-metrics',
  },
  'skewed-distribution': {
    term: 'Right-skewed distribution',
    def: 'A distribution with most values clustered low and a long tail to the right — the normal shape of latency. The mean sits above the median and describes no actual request.',
    lesson: 'benchmarking/percentiles',
  },
  'latency-slo': {
    term: 'Latency SLO',
    def: 'A performance promise naming four things: metric, percentile, threshold and window — e.g. "p95 TTFT under 600 ms over any rolling 5 minutes". Missing any one makes it unenforceable.',
    lesson: 'benchmarking/slos-goodput',
  },
  'goodput-per-dollar': {
    term: 'Goodput per dollar',
    def: 'Goodput in tokens per hour divided by GPU cost per hour. The only metric that compares two configurations honestly: two systems can differ by 20% in throughput and 5× in the value they deliver per dollar.',
    lesson: 'benchmarking/slos-goodput',
  },
  'operating-envelope': {
    term: 'Operating envelope',
    def: 'The set of (concurrency, context) conditions where a configuration both fits in memory and meets its SLO. It belongs to the whole stack — model, precision, GPU, engine version — and must be re-swept when any of them change.',
    lesson: 'benchmarking/break-one-gpu',
  },
  'safe-region': {
    term: 'Safe region',
    def: 'The cells of a performance sweep that fit in GPU memory and meet the latency objective. Everything you ship runs inside it; the operating point is where goodput peaks within it.',
    lesson: 'benchmarking/break-one-gpu',
  },
  'sweep': {
    term: 'Parameter sweep',
    def: 'Measuring a grid of configurations instead of a single point. Each cell is warmed up and repeated, then summarised as a median with its tail. The output is a surface and a boundary, not a number.',
    lesson: 'benchmarking/break-one-gpu',
  },
  'load-shedding': {
    term: 'Load shedding',
    def: 'Deliberately refusing work when the system is overloaded, so that accepted requests stay inside the SLO. Protects goodput for everyone already in flight; a fast error beats a slow success.',
    lesson: 'benchmarking/slos-goodput',
  },
  'warm-up': {
    term: 'Warm-up',
    def: 'Iterations run and discarded before measuring. The first calls pay kernel compilation, memory allocation, cold caches and GPU clock ramp — costs that do not exist in steady state and do not belong in your numbers.',
    lesson: 'benchmarking/workloads-reproducibility',
  },
  'benchmark-harness': {
    term: 'Benchmark harness',
    def: 'A reusable program that measures a system under a pinned, seeded workload and emits a machine-readable report. Four parts: instrument (correct streaming timing), workload, sweep, and verdict (SLO, goodput, operating point).',
    lesson: 'benchmarking/project-benchmark-harness',
  },
  'run-to-run-variance': {
    term: 'Run-to-run variance',
    def: 'How much repeated measurements of the identical workload disagree. If your own repeats differ by more than a few percent, no comparison made with that harness means anything — fix the harness before drawing conclusions.',
    lesson: 'benchmarking/workloads-reproducibility',
  },
  'admission-control': {
    term: 'Admission control',
    def: 'Capping how many requests may be in flight (concurrency) or queued, so that latency stays inside the SLO under load instead of degrading without bound. The main guardrail at the latency edge of the envelope.',
    lesson: 'benchmarking/break-one-gpu',
  },
};