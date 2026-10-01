// Module 13 · Writing Kernels with Triton — new terms.
import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  triton: {
    term: 'Triton',
    def: 'An open-source language and compiler for writing GPU kernels in Python. You write a Python function decorated with @triton.jit, and Triton compiles it to machine code for the specific GPU, caching the result per shape and dtype.',
    lesson: 'triton/triton-model',
  },
  jit: {
    term: 'Just-in-time (JIT) compilation',
    def: 'Compiling at the moment the function is first called, rather than ahead of time. A Triton kernel is translated the first time it sees a given set of argument types and dtypes; later calls with the same signature reuse the compiled binary.',
    lesson: 'triton/triton-model',
  },
  'program-id': {
    term: 'program_id',
    def: 'In Triton, the index of the calling program in the launch grid. It is how a kernel learns which slice of the data it owns, so every program can compute its own offsets instead of every program touching everything.',
    lesson: 'triton/triton-model',
  },
  'block-size': {
    term: 'Block size',
    def: 'How many elements one Triton program handles per step, chosen by the programmer with next_power_of_2. Too small wastes the GPU on launch and occupancy overhead; too large spills registers to slower memory and stops fitting.',
    lesson: 'triton/triton-model',
  },
  masking: {
    term: 'Masking',
    def: 'Restricting a block of work with a comparison mask so out-of-range elements are loaded and stored safely but skipped. It is how one kernel handles any length without a separate code path per shape.',
    lesson: 'triton/triton-model',
  },
  reduction: {
    term: 'Reduction',
    def: 'Collapsing many elements into one value, such as a sum, max or mean along an axis. On a GPU this is not a loop: one thread updates a private accumulator, then a tree combines the partial results, so the work takes log₂(n) steps.',
    lesson: 'triton/reductions-softmax',
  },
  'online-softmax': {
    term: 'Online softmax',
    def: 'Computing softmax over blocks with one pass: keep a running max and a running sum that are rescaled as each new block arrives, so the row never has to be visited twice and the global max is never needed first.',
    lesson: 'triton/reductions-softmax',
  },
  'numerical-stability': {
    term: 'Numerical stability',
    def: 'Keeping a computation’s answer close to the true one when intermediate values get extreme. Subtracting the row max before exponentiating is what stops softmax overflowing, because every exponentiated value is then at most 1.',
    lesson: 'triton/reductions-softmax',
  },
  fusion: {
    term: 'Kernel fusion',
    def: 'Running several operations in one kernel so intermediates stay in on-chip memory instead of making a round trip to HBM. Fusion trades registers and shared memory for bandwidth, which is why it wins on memory-bound work.',
    lesson: 'triton/fused-rmsnorm-rope',
  },
  'kernel-launch': {
    term: 'Kernel launch overhead',
    def: 'The fixed CPU-side cost of getting a kernel running on the GPU, roughly 5–10 µs. Small kernels that each take only a few microseconds are dominated by it, which is the main reason fusion exists.',
    lesson: 'triton/fused-rmsnorm-rope',
  },
  'roofline-gap': {
    term: 'Achieved bandwidth',
    def: 'Bytes moved per second divided by the GPU’s peak bandwidth, the memory-side analogue of MFU. A kernel at 70% is doing well; a fused kernel at 80% has little left to give.',
    lesson: 'triton/benchmark-kernels',
  },
  'l2-cache': {
    term: 'L2 cache',
    def: 'A large on-chip cache (tens of MB) sitting between HBM and the SMs. Repeatedly benchmarking a small tensor serves it from L2 instead of HBM, which inflates the result and is why benchmarks flush the cache between runs.',
    lesson: 'triton/benchmark-kernels',
  },
  autotuning: {
    term: 'Autotuning',
    def: 'Letting Triton benchmark a set of candidate configurations once at first call and keep the fastest for that shape, dtype and GPU. triton.autotune sweeps BLOCK_SIZE and num_warps automatically.',
    lesson: 'triton/benchmark-kernels',
  },
  'grid-metadata': {
    term: 'Kernel metadata',
    def: 'The numbers that describe a compiled kernel — its block size, number of warps, shared-memory use and register count. Triton prints them, and they are the first thing to look at when a kernel is slower than expected.',
    lesson: 'triton/triton-model',
  },
  'l2-flush': {
    term: 'Cache flushing',
    def: 'Evicting a tensor from cache before a timed run so the kernel must read from HBM, as it would in a real model. Standard practice: write a buffer larger than L2, then run the benchmark.',
    lesson: 'triton/benchmark-kernels',
  },
};