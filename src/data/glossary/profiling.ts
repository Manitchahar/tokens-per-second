// Module 12 · Profiling & GPU Performance — new terms.
import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  warp: {
    term: 'Warp',
    def: 'The 32 threads a GPU hardware scheduler issues instructions to, all in lockstep on one instruction. The unit that actually executes, not the thread. If the lanes in a warp disagree on a branch, the warp runs both branches.',
    lesson: 'profiling/execution-model',
  },
  'thread-block': {
    term: 'Thread block',
    def: 'A group of warps (32–1024 threads) that runs on one SM and no other. Because a block owns its SM privately, it is the scope over which shared memory is addressable — which is why tiling lives inside a block.',
    lesson: 'profiling/execution-model',
  },
  grid: {
    term: 'Grid',
    def: 'The complete set of blocks in one kernel launch. Blocks are independent, with no ordering guarantee, and the GPU scheduler places them onto SM slots as they free up.',
    lesson: 'profiling/execution-model',
  },
  occupancy: {
    term: 'Occupancy',
    def: 'Resident warps divided by the maximum an SM can hold. It exists to hide memory latency (~700 ns for HBM) by keeping more warps in flight. It is capped by registers per thread, and it is a means, not a goal.',
    lesson: 'profiling/execution-model',
  },
  'register-pressure': {
    term: 'Register pressure',
    def: 'When a kernel needs more registers per thread than an SM can supply for the occupancy you want. The fix is fewer live values per thread; the disease is spilling to local memory, which is HBM and doubles the damage.',
    lesson: 'profiling/execution-model',
  },
  'warp-divergence': {
    term: 'Warp divergence',
    def: 'Threads within one warp taking different branches of an if/else. The hardware executes every branch with the unused lanes masked, so N branches cost N instructions. Fix it with torch.where or arithmetic, never with if.',
    lesson: 'profiling/execution-model',
  },
  'wave-quantization': {
    term: 'Wave quantization',
    def: 'The loss from a grid that does not divide evenly into full rounds of blocks. A grid needing 1.05 waves runs 4 waves, the last nearly empty, so efficiency is W/⌈W⌉ — under 1.15× and you are wasting real time.',
    lesson: 'profiling/execution-model',
  },
  sector: {
    term: 'Sector',
    def: 'The 32-byte unit in which HBM actually transfers data. A warp load fetches every sector that any of its 32 lanes touched, which is why a strided access moves far more bytes than it asked for.',
    lesson: 'profiling/memory-hierarchy',
  },
  coalescing: {
    term: 'Coalescing',
    def: 'When the 32 lanes of a warp read consecutive addresses, so their loads land in few 128-byte cache lines. Coalesced BF16 accesses move 64 bytes in two sectors; a stride of 8 moves 512 bytes for the same data.',
    lesson: 'profiling/memory-hierarchy',
  },
  'bank-conflict': {
    term: 'Bank conflict',
    def: 'Two or more lanes of a warp reading the same shared-memory bank in one cycle. The access serializes into extra cycles. Reading a column of a [32][32] tile is the classic 32-way conflict; padding the row stride by one fixes it.',
    lesson: 'profiling/memory-hierarchy',
  },
  'achieved-bandwidth': {
    term: 'Achieved bandwidth',
    def: 'Bytes actually transferred divided by elapsed time. Compare it to the machine\'s peak: under ~60% on a memory-bound kernel means sectors are being wasted, not that the arithmetic is slow.',
    lesson: 'profiling/memory-hierarchy',
  },
  mfu: {
    term: 'MFU (model FLOP utilization)',
    def: 'Achieved FLOP/s divided by peak FLOP/s. Only meaningful for compute-bound work: a memory-bound decode kernel has a tiny MFU by construction. Also inflated by tile padding at small batch.',
    lesson: 'profiling/roofline',
  },
  mbu: {
    term: 'MBU (memory bandwidth utilization)',
    def: 'Achieved bytes/s divided by peak bandwidth. The metric to quote for a memory-bound kernel such as decode, where 75–90% means it is at its floor. The higher of MFU and MBU is the kernel\'s bound.',
    lesson: 'profiling/roofline',
  },
  headroom: {
    term: 'Headroom',
    def: 'Measured kernel time divided by the roofline floor: the speed-up a perfect kernel would give on identical hardware. Below ~1.15× there is nothing to win, and the honest move is to stop optimizing that kernel.',
    lesson: 'profiling/roofline',
  },
  'ridge-point': {
    term: 'Ridge point',
    def: 'Compute ceiling divided by bandwidth: the arithmetic intensity at which the two limits balance. ~295 FLOP/byte on H100, ~156 on A100, ~203 on T4. Below it a kernel is memory-bound; above it, compute-bound.',
    lesson: 'profiling/roofline',
  },
  'speed-of-light': {
    term: 'Speed of Light (SOL)',
    def: 'The Nsight Compute section showing achieved compute % and memory % side by side. The higher of the two is the bound; above 80% there is nothing left to win, below 60% the profile names the fix.',
    lesson: 'profiling/profiling-tools',
  },
  'nsight-systems': {
    term: 'Nsight Systems',
    def: 'NVIDIA\'s whole-process timeline profiler. Shows every kernel, memcpy, synchronization and NCCL call against the CPU thread, which is where you find launch gaps, CPU bottlenecks and communication stalls.',
    lesson: 'profiling/profiling-tools',
  },
  'nsight-compute': {
    term: 'Nsight Compute',
    def: 'NVIDIA\'s single-kernel profiler. Replays one kernel many times while varying cache state, then reports Speed of Light, warp stall reasons, register counts and per-source-line attribution.',
    lesson: 'profiling/profiling-tools',
  },
  'chrome-trace': {
    term: 'Chrome trace',
    def: 'The JSON timeline the PyTorch profiler exports, openable in Perfetto, chrome://tracing or VS Code. Unlike the per-op table it shows concurrency, which is how you tell GPU-bound from launch-bound from CPU-bound.',
    lesson: 'profiling/profiling-tools',
  },
  'launch-bound': {
    term: 'Launch-bound',
    def: 'When gaps between kernels dominate the timeline: the GPU is starved by per-launch overhead rather than by the kernels themselves. The fix is torch.compile and CUDA graphs (Module 15), not better kernels.',
    lesson: 'profiling/profiling-tools',
  },
  'warp-stall': {
    term: 'Warp stall',
    def: 'A warp that cannot issue its next instruction. The reason matters: Stall Long Scoreboard means waiting on memory, Stall MIO Throttle means too many shared-memory instructions queued, Stall Wait means an instruction-cache miss.',
    lesson: 'profiling/profiling-tools',
  },
  'kv-crossover': {
    term: 'Weights/KV crossover',
    def: 'The context length at which a decode step reads more KV cache than weights. For Llama-3.1-8B in BF16 it is 122,500 ÷ batch tokens: 3,830 at batch 32, 957 at batch 128. Below it quantize weights, above it quantize KV.',
    lesson: 'profiling/profile-prefill-decode',
  },
  'bottleneck-report': {
    term: 'Bottleneck report',
    def: 'The one-page write-up that makes a profiling result checkable: a claim with a number, the configuration, the evidence with its source, what was tried including what failed, and a residual-risk section.',
    lesson: 'profiling/lab-bottleneck-report',
  },
};
