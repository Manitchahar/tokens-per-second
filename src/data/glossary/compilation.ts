import type { GlossaryEntry } from '../glossary';

// New vocabulary introduced in Module 15 · Compilation & CUDA Graphs.
// Terms the learner already met earlier (kernel, cuda, kv-cache, prefill,
// decode, paged-attention, quantization, ...) live in src/data/glossary.ts.

export const terms: Record<string, GlossaryEntry> = {
  // ── Lesson 15.1 · eager mode & launch overhead
  'eager-mode': {
    term: 'Eager mode',
    def: 'PyTorch’s default execution model. Every operation you write is dispatched immediately, one at a time, so each one becomes its own GPU kernel launch. The opposite of lazy mode.',
    lesson: 'compilation/launch-overhead',
  },
  'lazy-mode': {
    term: 'Lazy mode',
    def: 'Execution model in which operations are recorded into a tape as you write them and only turned into real work later, in larger pieces. What torch.compile and JAX give you by default.',
    lesson: 'compilation/launch-overhead',
  },
  'kernel-launch': {
    term: 'Kernel launch',
    def: 'The act of telling the GPU to run one function over some pointers. A forward pass is hundreds of them.',
    lesson: 'compilation/launch-overhead',
  },
  'launch-overhead': {
    term: 'Launch overhead',
    def: 'The CPU work that happens before a kernel starts: a Python call, the dispatcher, the caching allocator, then cudaLaunchKernel. A few microseconds each, and it is paid for every launch whether or not the GPU needs it.',
    lesson: 'compilation/launch-overhead',
  },
  'launch-bound': {
    term: 'Launch-bound',
    def: 'A configuration where the CPU cannot issue work fast enough to keep the GPU busy. The step time is set by Python and the dispatcher, not by memory bandwidth, and it is a constant tax on every generated token.',
    lesson: 'compilation/launch-overhead',
  },
  'crossover-batch': {
    term: 'Crossover batch',
    def: 'The concurrency at which the GPU’s per-step work exactly equals the CPU’s per-step launch work. Below it, extra users are almost free; above it, every user costs KV bandwidth.',
    lesson: 'compilation/launch-overhead',
  },
  'host-sync': {
    term: 'Host sync',
    def: 'The one moment in a step when the CPU waits for the GPU: reading the sampled token ids back so it can decide what to do next. About 0.2 ms per decode step, and the only place the two pipelines are not overlapped.',
    lesson: 'compilation/launch-overhead',
  },

  // ── Lesson 15.2 · torch.compile, fusion & graph breaks
  'torch-compile': {
    term: 'torch.compile',
    def: 'PyTorch’s compiler front door. It traces your Python into a graph, checks guards on the inputs, and hands the graph to Inductor, which generates fused kernels — mostly Triton — and calls them from C++ instead of from Python.',
    lesson: 'compilation/torch-compile',
  },
  dynamo: {
    term: 'Dynamo',
    def: 'The tracing half of torch.compile. It runs your Python with a tensor proxy that records what you do, until it hits something it cannot record — that is a graph break.',
    lesson: 'compilation/torch-compile',
  },
  inductor: {
    term: 'Inductor',
    def: 'The code-generating half of torch.compile. It takes a traced graph and emits Triton kernels, choosing fusion groups, tiling and autotuning configurations for the hardware in front of it.',
    lesson: 'compilation/torch-compile',
  },
  guard: {
    term: 'Guard',
    def: 'The check Dynamo runs before reusing a compiled graph: are the dtypes, shapes, devices and static properties the same as last time? A failed guard means a cache miss, and a miss means recompiling.',
    lesson: 'compilation/torch-compile',
  },
  'recompilation': {
    term: 'Recompilation',
    def: 'What happens when a guard fails: Dynamo traces again and Inductor generates and compiles new kernels. It costs seconds. A handful of recompiles is normal; thousands is a bug in your code.',
    lesson: 'compilation/torch-compile',
  },
  'graph-break': {
    term: 'Graph break',
    def: 'A point where Dynamo cannot keep tracing — a .item() call, a data-dependent branch, printing a tensor, an unsupported op. The graph is split there, that op runs in eager mode, and everything after it starts a new graph. Graph breaks are how a compiled model silently becomes an eager one.',
    lesson: 'compilation/torch-compile',
  },
  'operator-fusion': {
    term: 'Operator fusion',
    def: 'Combining several elementwise operations into one kernel, so the intermediate never reaches GPU memory. Saves launches (the CPU bill) and HBM round trips (the GPU bill) at once.',
    lesson: 'compilation/torch-compile',
  },
  'epilogue-fusion': {
    term: 'Epilogue fusion',
    def: 'Fusing the pointwise work that follows a matrix multiply into the matrix multiply’s own kernel, so the result of the GEMM is never written to HBM and read back. This is where most of a compiled model’s real byte savings come from.',
    lesson: 'compilation/torch-compile',
  },
  'dynamic-shape': {
    term: 'Dynamic shape',
    def: 'A tensor dimension whose value changes between calls, such as a batch size or a sequence length. Left automatic, every new value can be a new guard failure and a recompile.',
    lesson: 'compilation/torch-compile',
  },
  'mark-dynamic': {
    term: 'mark_dynamic',
    def: 'A torch.compile hint that tells Dynamo a dimension will vary, so it generates a kernel that is correct for a range of sizes instead of one specialised to an exact shape.',
    lesson: 'compilation/torch-compile',
  },
  'reduce-overhead': {
    term: 'mode="reduce-overhead"',
    def: 'A torch.compile mode that wraps the compiled region in CUDA graphs and replays them, so the CPU issues one launch per step instead of hundreds. The same technique as the next lesson, requested with a flag.',
    lesson: 'compilation/torch-compile',
  },
  triton: {
    term: 'Triton',
    def: 'The Python-embedded GPU kernel language taught in Module 13. It is the language Inductor emits: that is why most of a compiled model’s fused kernels are readable Triton rather than hand-tuned CUDA.',
    lesson: 'compilation/torch-compile',
  },
  'max-autotune': {
    term: 'mode="max-autotune"',
    def: 'A torch.compile mode that benchmarks candidate Triton matmul configurations for your exact shapes and keeps the winner. Slower to compile, and often the fastest to run for GEMM-heavy prefill.',
    lesson: 'compilation/torch-compile',
  },

  // ── Lesson 15.3 · CUDA graphs
  'cuda-graph': {
    term: 'CUDA graph',
    def: 'A recorded list of kernel launches, with their arguments, stored on the device. Replaying it re-issues every kernel in order with no CPU in between, so a step of 500 launches costs one API call.',
    lesson: 'compilation/cuda-graphs',
  },
  'graph-capture': {
    term: 'Graph capture',
    def: 'Running the step once in capture mode, where launches are recorded into a graph instead of being executed. Capture executes nothing, so it is fast — but it must be done again for every distinct shape.',
    lesson: 'compilation/cuda-graphs',
  },
  'graph-replay': {
    term: 'Graph replay',
    def: 'Launching a captured graph. One call, hundreds of kernels, correct order, and the CPU is free for the rest of the step. This is the whole point.',
    lesson: 'compilation/cuda-graphs',
  },
  'static-input': {
    term: 'Static input buffer',
    def: 'A GPU tensor allocated once, at a fixed address, that a captured graph reads from. You copy new data into it before each replay, because the graph has the address baked in and cannot be told a different one.',
    lesson: 'compilation/cuda-graphs',
  },
  'static-output': {
    term: 'Static output buffer',
    def: 'The other end of the same rule: a graph always writes to the same address, so results come out wherever that buffer is. Keeping the output small — one sampled token id per sequence instead of the full logits — is what makes replay cheap.',
    lesson: 'compilation/cuda-graphs',
  },
  'make-graphed-callables': {
    term: 'make_graphed_callables',
    def: 'A torch.cuda.graphs helper that captures a whole module or function for you, including the static input and output buffers, instead of making you manage capture mode yourself.',
    lesson: 'compilation/cuda-graphs',
  },
  'sampled-id': {
    term: 'Sampled token id',
    def: 'The one integer per sequence that a decode step ultimately produces. Keeping it — rather than the full logit vector — is why an engine puts its sampler inside the CUDA graph and keeps the graph output tiny.',
    lesson: 'compilation/cuda-graphs',
  },
  'cudagraph-padding': {
    term: 'Graph padding',
    def: 'Replaying a graph captured for 64 sequences when you only have 33: run it anyway on 31 padding sequences and throw the extra results away. Engine-sized graphs need a graph per batch size, or this trick.',
    lesson: 'compilation/cuda-graphs',
  },
  'cudagraph-pool': {
    term: 'Graph memory pool',
    def: 'The separate KV memory that graphs must use, because a graph writes to fixed addresses and therefore cannot use a pool that preemption is free to move blocks around in. Engines keep a second pool for this reason.',
    lesson: 'compilation/cuda-graphs',
  },
};
