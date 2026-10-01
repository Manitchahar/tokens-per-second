import type { GlossaryEntry } from '../glossary';

// New vocabulary introduced in Module 17. Terms the learner already met in earlier
// modules (vllm, ttft, tpot, goodput, oom, quantization, ...) live in
// src/data/glossary.ts or in an earlier module's glossary file.
export const terms: Record<string, GlossaryEntry> = {
  // ── Module 17.1 · How to compare engines fairly
  'harness-contract': {
    term: 'Harness contract',
    def: 'The list of things you promise to hold identical across every engine in a comparison: the workload, the arrival pattern, the load point, the output quality bar, the tuning each engine is allowed, and the versions. If an item is not on the contract, two numbers are not comparable and neither is the conclusion drawn from them.',
    lesson: 'engine-landscape/fair-comparison',
  },
  confound: {
    term: 'Confound',
    def: 'Anything besides the thing you are testing that differs between two arms of an experiment. A confound is not noise: it is a real cause you did not intend, and it moves the result by an amount you then attribute to the wrong cause.',
    lesson: 'engine-landscape/fair-comparison',
  },
  'closed-loop-load': {
    term: 'Closed-loop load generator',
    def: 'A load test that keeps exactly N requests in flight and issues the next one only when one finishes. Simple and repeatable, but it hides queueing: the offered load falls automatically when the system slows down, so a slow engine is quietly given easier conditions.',
    lesson: 'engine-landscape/fair-comparison',
  },
  'open-loop-load': {
    term: 'Open-loop load generator',
    def: 'A load test that issues requests on a fixed schedule regardless of how the server is keeping up. Offered load is therefore a property of the test, not of the server, so a slow engine is caught by a growing queue instead of being throttled into looking fine.',
    lesson: 'engine-landscape/fair-comparison',
  },
  'offered-load': {
    term: 'Offered load',
    def: 'The arrival rate you present to the system, expressed as a fraction rho of what it could serve with every server permanently busy. At rho below about 1 a queue never forms; above it the queue grows without bound, which is why you compare engines at a stated rho rather than at "as much as possible".',
    lesson: 'engine-landscape/fair-comparison',
  },
  'run-to-run-spread': {
    term: 'Run-to-run spread',
    def: 'How far apart two runs of the identical experiment land. Nothing is a result until the effect you are claiming is larger than this spread; measuring it is the only way to know whether a 6% difference between two engines is a fact or a coincidence.',
    lesson: 'engine-landscape/fair-comparison',
  },
  'quality-parity': {
    term: 'Quality parity',
    def: 'The requirement that every engine in a comparison produce the same output tokens for the same input. Held by fixing the sampling parameters (greedy, temperature 0, fixed seed) and by checking the token streams match, because a faster engine that quietly changed the numerics is not faster, it is different.',
    lesson: 'engine-landscape/fair-comparison',
  },
  'warm-state': {
    term: 'Warm state',
    def: 'The condition of a server after compilation, allocation, graph capture and clock ramp-up have finished. Steady-state numbers come from the warm state; cold-start numbers are a separate, real and different measurement, and mixing the two is a classic way to make an engine look worse than it is.',
    lesson: 'engine-landscape/fair-comparison',
  },
  'version-pin': {
    term: 'Version pin',
    def: 'Recording the exact version of the engine, its runtime and every library, and holding them fixed while you compare. Engine behaviour moves fast: defaults, default backends and even default dtypes change between releases, so an unpinned comparison measures two different dates.',
    lesson: 'engine-landscape/fair-comparison',
  },

  // ── Module 17.2 · vLLM, SGLang, TensorRT-LLM & llama.cpp
  'inference-engine': {
    term: 'Inference engine',
    def: 'The program that owns a model and serves it to many requests: it schedules, manages the KV cache, launches the kernels and speaks HTTP. An engine is not the model and not the kernels; it is the machinery that decides when each of them runs.',
    lesson: 'engine-landscape/engines-compared',
  },
  'scheduler-stage': {
    term: 'Scheduler stage',
    def: 'The part of an engine that decides what goes into the next forward pass: which running sequences continue, which waiting ones join, how many tokens each may produce, and who is preempted when memory runs out. It is pure control logic, runs on the CPU, and is where most of an engine\'s design personality lives.',
    lesson: 'engine-landscape/engines-compared',
  },
  sglang: {
    term: 'SGLang',
    def: 'An open-source serving engine, from the lineage of vLLM, whose distinguishing idea is RadixAttention: the prefix cache is a radix tree over token sequences that the attention kernel walks directly, so shared prefixes are reused without a gather copy. Strongest on prefix-heavy traffic such as multi-turn chat and agents.',
    lesson: 'engine-landscape/engines-compared',
  },
  radixattention: {
    term: 'RadixAttention',
    def: 'Attention that indexes a prefix tree in the kernel instead of copying blocks through a block table first. It is what makes prefix sharing free in SGLang: a shared system prompt or multi-turn history is read once, and the kernel follows the tree rather than gathering scattered blocks.',
    lesson: 'engine-landscape/engines-compared',
  },
  'tensorrt-llm': {
    term: 'TensorRT-LLM',
    def: 'NVIDIA\'s inference engine. It runs a build phase ahead of serving that traces the model, fuses operators, selects and autotunes kernels for one specific (model, precision, GPU, parallel configuration), and emits an execution plan. The plan then replays with very little per-step overhead, at the cost of a build you must repeat whenever any of those inputs change.',
    lesson: 'engine-landscape/engines-compared',
  },
  'execution-plan': {
    term: 'Execution plan',
    def: 'The compiled artefact an ahead-of-time engine produces: the full list of fused kernels, their shapes, their chosen algorithms and their launch order for one exact configuration. Loading it is fast and running it is lean; producing it can take minutes to hours, and it is invalid the moment the model, precision, GPU or parallel layout changes.',
    lesson: 'engine-landscape/engines-compared',
  },
  'ahead-of-time-compile': {
    term: 'Ahead-of-time compile',
    def: 'Doing the expensive kernel selection, fusion and autotuning before serving starts rather than during the first request. It buys a lean, predictable steady state and a much lower fixed cost per step, and it charges you for that with build time and a rebuild whenever anything it depends on changes.',
    lesson: 'engine-landscape/engines-compared',
  },
  'llama-cpp': {
    term: 'llama.cpp',
    def: 'A small, portable inference engine written in C/C++ that runs the same transformer on CPUs, Apple Silicon, older GPUs and tiny accelerators. It is the default answer when there is no NVIDIA GPU, when memory is the binding constraint, and when a 4-bit quantized model has to fit on hardware that cannot hold it in 16 bits.',
    lesson: 'engine-landscape/engines-compared',
  },
  ggml: {
    term: 'ggml / GGUF',
    def: 'ggml is the small tensor library llama.cpp is built on; GGUF is the container format it reads, which stores weights together with their quantization scales and the model\'s metadata in one file. A GGUF file is a whole model in one portable artifact, which is why this stack is so easy to move between machines.',
    lesson: 'engine-landscape/engines-compared',
  },
  'weight-only-quant': {
    term: 'Weight-only quantization',
    def: 'Storing the weights in fewer bits while computing activations in full precision, so the bandwidth saved on the weights per decode step is the whole win. It is the strongest lever when the model must fit in memory at all, which is exactly llama.cpp\'s situation on a laptop or an edge device.',
    lesson: 'engine-landscape/engines-compared',
  },
  'openai-compatible-api': {
    term: 'OpenAI-compatible API',
    def: 'Serving the same request and response shapes as a hosted model provider, so client code and SDKs can be pointed at a local engine by changing a base URL. It is the reason engines are substitutable at all: without it you rewrite your client, and with it you can A/B two engines behind one interface.',
    lesson: 'engine-landscape/engines-compared',
  },

  // ── Module 17.P · Project: build an engine selection matrix
  'selection-matrix': {
    term: 'Selection matrix',
    def: 'The decision table that turns "which engine?" into arithmetic: rows are candidate engines, columns are criteria, some columns are hard constraints that eliminate a row outright, and the rest are weighted scores. It is only trustworthy when every number in a scored column came from the same harness on your traffic.',
    lesson: 'engine-landscape/project-engine-matrix',
  },
  'hard-constraint': {
    term: 'Hard constraint',
    def: 'A criterion a candidate either satisfies or does not, with no trading off. It runs before any scoring, because a row that fails one can never be the answer no matter how well it scores elsewhere.',
    lesson: 'engine-landscape/project-engine-matrix',
  },
  'weighted-criteria': {
    term: 'Weighted criteria',
    def: 'Scored criteria combined as sum(weight x score) with weights that come from your traffic, not from your taste. The weight is the argument: it is where you state that interactive latency matters more than peak batch throughput, and it must be written down before the scores are collected.',
    lesson: 'engine-landscape/project-engine-matrix',
  },
  'goodput-per-dollar': {
    term: 'Goodput per dollar',
    def: 'Tokens per second that met your SLO, divided by what you paid per second to produce them. It is the single number that decides between two engines that both work, because it is the only one that prices the latency you promised rather than the speed you achieved.',
    lesson: 'engine-landscape/project-engine-matrix',
  },
};