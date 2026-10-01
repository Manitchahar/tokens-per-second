import type { GlossaryEntry } from '../glossary';

// Terms this module introduces that are not in the shared glossary. Latency metrics,
// SLOs, goodput and the operating envelope are already defined in glossary.ts and in
// data/glossary/benchmarking.ts, and are intentionally not repeated.

export const terms: Record<string, GlossaryEntry> = {
  // ── Module 25.1 · Cost per million tokens, from scratch
  'gpu-seconds-per-request': {
    term: 'GPU-seconds per request',
    def: 'How much machine time one request actually consumed: t_prefill + n_out × t_step / B. Prefill belongs to the request alone; a decode step is shared across the batch, so the batch is divided out. It is service time read as a quantity of GPU rather than of latency, and multiplied by the price of a second it becomes the cost of the request.',
    lesson: 'economics/cost-per-token',
  },
  'shared-decode-step': {
    term: 'Shared decode step',
    def: 'A decode step that serves every sequence in the batch at once — it reads all the weights and all the KV once and emits one token per sequence. Because it is shared, one request owns only 1/B of it, which is why raising concurrency makes every output token cheaper rather than merely faster.',
    lesson: 'economics/cost-per-token',
  },
  'output-premium': {
    term: 'Output premium',
    def: 'The ratio of dollars per million output tokens to dollars per million prompt tokens. Typically several times, because an output token is priced on the memory-bound decode step while a prompt token is priced on the compute-bound prefill. It is not a constant: it tracks your prompt-to-answer ratio, so the same model can show 8× or 40× depending on the workload.',
    lesson: 'economics/cost-per-token',
  },
  'blended-token-price': {
    term: 'Blended token price',
    def: 'Dollars per million tokens counting prompt and output together. It lets you total a bill while hiding the prompt-to-answer ratio, which is the thing that actually decides what to optimize — so two products with completely different cost structures can report the same blended price.',
    lesson: 'economics/cost-per-token',
  },
  'decode-priced': {
    term: 'Decode-priced',
    def: 'A request whose machine time is dominated by its answer. What you pay is set by price per unit of memory bandwidth, and batching helps enormously because it amortizes the weights and the KV across more sequences.',
    lesson: 'economics/cost-per-token',
  },
  'prefill-priced': {
    term: 'Prefill-priced',
    def: 'A request whose machine time is dominated by its prompt — long prompts with short answers. What you pay is set by price per unit of FLOP/s, and batching barely helps, because prefill is a parallel pass over one request’s own tokens and adding other requests cannot make it cheaper.',
    lesson: 'economics/cost-per-token',
  },
  'memory-priced': {
    term: 'Memory-priced',
    def: 'What applies when the weights do not fit beside their KV on one card. The bill is multiplied by the number of cards and every step is slowed by the communication between shards, so the cost question becomes "how many cards must I buy" rather than "how fast is this card".',
    lesson: 'economics/cost-per-token',
  },

  // ── Module 25.2 · Goodput per dollar & choosing GPUs
  'price-per-bandwidth': {
    term: 'Dollars per unit of bandwidth',
    def: 'Hourly price divided by memory bandwidth, in dollars per (TB/s). It is the scalar that ranks hardware for decode-bound serving, because a decode step is memory-bound. From an A100 to a B200 this falls only about 14% while raw bandwidth rises fourfold — memory got much better and much more expensive at nearly the same rate.',
    lesson: 'economics/goodput-per-dollar',
  },
  'price-per-flops': {
    term: 'Dollars per unit of FLOP/s',
    def: 'Hourly price divided by arithmetic throughput, in dollars per (PFLOP/s). It ranks hardware for prefill-bound serving, because prefill is compute-bound. From an A100 to a B200 this roughly halves — arithmetic improved much faster relative to its price than bandwidth did, so new hardware helps prompt-heavy products far more than it helps answer-heavy ones.',
    lesson: 'economics/goodput-per-dollar',
  },
  'break-even-test': {
    term: 'Break-even test',
    def: 'The rule for comparing two cards in your head: a cheaper card wins if it is proportionally slower than it is cheaper. Compute both factors per user rather than per batch, because batching makes every card look similar and hides the difference entirely. A card 1.6× slower at 1.56× cheaper loses; the same comparison flips once memory forces you to a different card count.',
    lesson: 'economics/goodput-per-dollar',
  },
  'value-spread': {
    term: 'Value spread',
    def: 'The ratio between the best and worst candidates on goodput per dollar, compared against the ratio on raw throughput. A large throughput spread with a small value spread means hardware generations buy speed much faster than they buy value — at which point the cost argument is exhausted and the decision should be made on fit, availability and operational risk.',
    lesson: 'economics/goodput-per-dollar',
  },
  'precision-first': {
    term: 'Precision first, then card',
    def: 'The ordering rule for hardware decisions: choose weight precision before choosing the GPU. Precision determines how many bytes the weights take, which determines how many cards you must buy, and the card count is a multiplier on the entire bill. An INT4 70B on one card can beat a BF16 70B on two by a factor of five, and no amount of card shopping reverses that.',
    lesson: 'economics/goodput-per-dollar',
  },

  // ── Module 25.3 · Capacity planning & headroom
  'busy-gpus': {
    term: 'Busy GPUs',
    def: 'Arrival rate times machine seconds per request: the number of GPUs that must be working simultaneously to retire that traffic. It is a floor, not a fleet size, because it assumes 100% utilization — which is a service that has already failed rather than one you can run.',
    lesson: 'economics/capacity-planning',
  },
  'committed-headroom': {
    term: 'Committed headroom',
    def: 'Capacity that is bought and running before the spike that needs it arrives. Distinct from an autoscaling target: scaling on GPU utilization gives you a floor but never a ceiling in time, because new cards arrive after the queue has already formed. Headroom is what absorbs a deploy, a node drain and a traffic spike.',
    lesson: 'economics/capacity-planning',
  },
  'utilization-tax': {
    term: 'Utilization tax',
    def: 'The multiplier by which average under-utilization raises your cost per unit of useful work: 1 ÷ average utilization. A fleet running at 40% average utilization pays 2.5× per unit of work versus a fully-utilized one. The response is to consolidate onto fewer cards, not to buy more.',
    lesson: 'economics/capacity-planning',
  },
  'step-function-bill': {
    term: 'Step-function bill',
    def: 'The fact that your monthly cost is a step function of fleet size, not a smooth function of utilization. Utilization moves you within a step; only card count moves you between steps. This is why "what utilization should we run at" is the wrong question and "how many cards have we committed to" is the right one.',
    lesson: 'economics/capacity-planning',
  },
  'capacity-plan': {
    term: 'Capacity plan',
    def: 'A living document stating the workload (with length distributions, not averages), the SLO, the derived machine-seconds per request, the busy-GPU floor, the committed fleet with its headroom argument, and the scaling trigger. It must be re-run whenever the model, precision, engine, context length or traffic mix changes — especially when traffic crosses the last step you planned.',
    lesson: 'economics/capacity-planning',
  },

  // ── Module 25.P · Project: the cheapest system that meets the SLO
  'cheapest-system-that-meets-slo': {
    term: 'Cheapest system that meets the SLO',
    def: 'The actual deliverable of inference cost engineering: the lowest-cost configuration that still honours the latency promise. It is found by eliminating candidates on hard constraints (does it fit, does it meet the SLO) before scoring anything on value, and it is usually not the fastest system and sometimes not the smallest model.',
    lesson: 'economics/project-cheapest-system',
  },
  'decision-document': {
    term: 'Decision document',
    def: 'The written artifact stating what the cheapest system that meets the SLO is, in one sentence with a number attached, plus the provenance for every claim and a sensitivity section naming the thresholds at which the decision would reverse. A document without thresholds for reversal is a preference rather than a decision.',
    lesson: 'economics/project-cheapest-system',
  },
};