import type { GlossaryEntry } from '../glossary';

// Terms introduced in Module 07 · Batching & Scheduling.
// `batch` and `continuous-batching` already exist in src/data/glossary.ts (taught here),
// so they are deliberately not repeated.

export const terms: Record<string, GlossaryEntry> = {
  'static-batching': {
    term: 'Static batching',
    def: 'The old way of serving: collect a group of requests into one batch, run the whole batch until the slowest request finishes, then start the next batch. Every request in the group waits for the slowest one.',
    lesson: 'batching-scheduling/naive-and-static-batching',
  },
  scheduler: {
    term: 'Scheduler',
    def: 'The component in a serving engine that decides, for each step, which requests run together and which wait. It is the only part of the engine deciding who gets the GPU right now.',
    lesson: 'batching-scheduling/continuous-batching',
  },
  'engine-step': {
    term: 'Engine step (iteration)',
    def: 'One full pass of the model over the current batch: prefill of any admitted chunks plus one decode token for every running sequence. Engines also call it an iteration or a scheduler tick.',
    lesson: 'batching-scheduling/continuous-batching',
  },
  'running-batch': {
    term: 'Running batch',
    def: 'The set of sequences that are being decoded right now. Its size is usually called the batch size, and it is what sets the step time.',
    lesson: 'batching-scheduling/continuous-batching',
  },
  'waiting-queue': {
    term: 'Waiting queue',
    def: 'Requests that have arrived and been tokenized but are not in the running batch. Their KV cache does not exist yet; they cost almost no GPU memory, only waiting time.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  'queueing-delay': {
    term: 'Queueing delay',
    def: 'Time a request spends in the waiting queue before its first token is computed. It adds straight to TTFT and is invisible in a single-user test.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  'batch-slot': {
    term: 'Batch slot',
    def: 'One place in the running batch. max_num_seqs caps how many there are, and each slot that a finished request vacates is reusable by the next request on the very next step.',
    lesson: 'batching-scheduling/continuous-batching',
  },
  padding: {
    term: 'Padding',
    def: 'Filling a batch out to a common shape by adding positions that produce nothing. Padded slots still cost math, and in a dense implementation they can still cost KV memory.',
    lesson: 'batching-scheduling/naive-and-static-batching',
  },
  'ragged-batch': {
    term: 'Ragged batch',
    def: 'A batch whose sequences have different lengths, so rows finish at different steps. Real traffic is almost always ragged, which is exactly why static batching wastes so much.',
    lesson: 'batching-scheduling/naive-and-static-batching',
  },
  'max-num-batched-tokens': {
    term: 'max_num_batched_tokens',
    def: 'The engine setting that caps how many prompt tokens plus decode tokens one engine step may process. A decode-only step uses one token per sequence, so this mainly limits prefill.',
    lesson: 'batching-scheduling/scheduler-knobs',
  },
  'max-num-seqs': {
    term: 'max_num_seqs',
    def: 'The engine setting that caps how many sequences may be in the running batch at once. It is the main control on KV-cache memory and on per-user speed.',
    lesson: 'batching-scheduling/scheduler-knobs',
  },
  'max-model-len': {
    term: 'max_model_len',
    def: 'The engine setting that sets the maximum context length (prompt plus generated tokens) it will accept for one request. Setting it high reserves KV memory for the worst case.',
    lesson: 'batching-scheduling/scheduler-knobs',
  },
  'gpu-memory-utilization': {
    term: 'gpu_memory_utilization',
    def: 'The fraction of the GPU’s memory an engine is allowed to fill at startup. Whatever is left after weights and workspace becomes the KV-cache budget.',
    lesson: 'batching-scheduling/scheduler-knobs',
  },
  'kv-budget': {
    term: 'KV budget',
    def: 'The slice of GPU memory an engine reserves for KV cache. It is the real limit on how many sequences can run at once, and it is set at startup by measuring free memory.',
    lesson: 'batching-scheduling/scheduler-knobs',
  },
  'chunked-prefill': {
    term: 'Chunked prefill',
    def: 'Splitting a long prompt into pieces that each fit inside one step’s token budget, so a single huge prompt cannot monopolize the GPU. Interleaves the chunks with decode steps so the stall is paid in small slices.',
    lesson: 'batching-scheduling/scheduler-knobs',
  },
  fairness: {
    term: 'Fairness',
    def: 'How the scheduler shares the GPU between users. It decides whether one user’s long answer can delay everybody else, and whether anyone can be starved indefinitely.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  'head-of-line-blocking': {
    term: 'Head-of-line blocking',
    def: 'A short request stuck behind a long one because the scheduler serves strictly in arrival order. It is the classic fairness problem, and it is why engines support priority queues.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  'priority-queue': {
    term: 'Priority queue',
    def: 'A waiting queue ordered by priority rather than arrival time, so premium or interactive traffic is admitted first. Fairness is then a policy decision, not a side effect.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  cancellation: {
    term: 'Cancellation',
    def: 'Dropping an in-flight request when the client disconnects or the user presses stop. Freeing the slot immediately is pure win: the tokens of a cancelled request are worth nothing.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  preemption: {
    term: 'Preemption',
    def: 'Taking a running request out of the batch to free memory for others, then putting it back later by recomputing its prefix or swapping its KV cache out to host memory. The cost is paid as extra prefill.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  'admission-control': {
    term: 'Admission control',
    def: 'Refusing new requests when the queue is already too deep. Turning a request away with a clear error beats accepting it and waiting 30 seconds for a first token.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  backpressure: {
    term: 'Backpressure',
    def: 'Letting a full queue push back on the callers upstream, so the system slows down instead of absorbing unbounded work. Implemented with queue limits, timeouts and 429-style rejections.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  'load-shedding': {
    term: 'Load shedding',
    def: 'Deliberately dropping some requests to protect the latency of the rest. It is the correct behaviour at overload, and it needs a policy: what to drop, and who decides.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  'wasted-slot': {
    term: 'Wasted slot',
    def: 'A batch slot that occupies the GPU without producing a token anyone wants: a finished request kept in the batch, or a padded position. The single most useful waste number to log.',
    lesson: 'batching-scheduling/naive-and-static-batching',
  },

  // ── the two wastes, and the metric that tells them apart ──
  'tail-waste': {
    term: 'Tail waste',
    def: 'Batch slots held by requests that already finished, so the GPU pays for them but they produce no token. Continuous batching drives this to zero by construction.',
    lesson: 'batching-scheduling/continuous-batching',
  },
  'fill-waste': {
    term: 'Fill waste',
    def: 'Batch slots that sit empty because no request was waiting. Not caused by any scheduling policy — caused by having less traffic than capacity. Equals 1 − average batch / seats.',
    lesson: 'batching-scheduling/continuous-batching',
  },
  'average-batch': {
    term: 'Average batch size',
    def: 'The mean number of sequences in a step. The most useful health metric in a serving system: at the seat limit means saturated, well below it means starved.',
    lesson: 'batching-scheduling/continuous-batching',
  },
  'batch-wait': {
    term: 'Batch wait',
    def: 'The delay a request pays while waiting to be admitted. It dominates TTFT, is invisible in single-user testing, and can only be fixed by the queue, never by a faster GPU.',
    lesson: 'batching-scheduling/continuous-batching',
  },

  // ── the two engine failure modes ──
  'starved-engine': {
    term: 'Starved engine',
    def: 'An engine whose average batch sits well below its seat limit because too few requests are queued. It is paying for a mostly idle GPU; adding seats makes it worse, not better.',
    lesson: 'batching-scheduling/scheduler-knobs',
  },
  'saturated-engine': {
    term: 'Saturated engine',
    def: 'An engine running at its seat limit with requests waiting. Throughput is capped by step time, so shortening the longest step — not adding seats — is what helps.',
    lesson: 'batching-scheduling/scheduler-knobs',
  },
  'prefill-stall': {
    term: 'Prefill stall',
    def: 'The pause every decoding user feels when one long prompt is prefilled into their step. A 2,048-token prefill on an H100 is about 63 ms, which is several decode steps.',
    lesson: 'batching-scheduling/scheduler-knobs',
  },

  // ── fairness and eviction ──
  starvation: {
    term: 'Starvation',
    def: 'A request admitted so rarely it effectively never runs. The characteristic failure of a priority queue under sustained high-priority load; cured by aging.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  aging: {
    term: 'Aging',
    def: 'Raising a waiting request’s scheduling priority the longer it has waited, so it eventually outranks newer arrivals. Makes starvation impossible and bounds the worst-case wait.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  'recompute-preemption': {
    term: 'Recompute preemption',
    def: 'Preemption in which the evicted sequence’s KV cache is dropped and rebuilt on re-admission. Cost is proportional to how many tokens it had already streamed, which is why the newest sequence is evicted.',
    lesson: 'batching-scheduling/queues-preemption',
  },
  'kv-pressure': {
    term: 'KV pressure',
    def: 'The running batch’s KV caches no longer fitting the memory budget, forcing the scheduler to evict or stall. Raise the memory fraction first; preempt only as the fallback.',
    lesson: 'batching-scheduling/queues-preemption',
  },
};