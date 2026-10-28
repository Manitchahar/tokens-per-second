import type { GlossaryEntry } from '../glossary';

// New vocabulary introduced in Module 02 · The GPU from First Principles.
// Terms the learner already met earlier (gpu, hbm, sram, roofline, ridge-point,
// npu, ...) live in src/data/glossary.ts or in another module's file.

export const terms: Record<string, GlossaryEntry> = {
  'memory-wall': {
    term: 'Memory wall',
    def: 'The growing gap between how fast chips can do arithmetic and how fast memory can feed them. Compute has improved much faster than bandwidth, so arithmetic units spend more of their life waiting for bytes. Decode sits on the wrong side of this wall.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  asic: {
    term: 'ASIC',
    def: 'Application-specific integrated circuit: a chip designed for one job rather than for general-purpose programming. An inference ASIC can spend its silicon on matrix engines and memory instead of graphics, many dtypes and general control flow, but it is awkward at work outside its design point.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  tpu: {
    term: 'TPU (tensor processing unit)',
    def: 'Google’s accelerator for matrix-heavy machine learning. It pairs a systolic array with a compiler (XLA) and a dedicated chip-to-chip interconnect, and is reached through JAX or PyTorch/XLA rather than CUDA.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  'systolic-array': {
    term: 'Systolic array',
    def: 'A grid of multiply-accumulate units in which one operand stays put and the other pulses through, each unit passing data to its neighbour. The regular flow gives large reuse without a big cache, at the cost of being happiest with dense, regular matrix shapes.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  'tpu-pod': {
    term: 'TPU pod',
    def: 'A group of TPUs joined by a dedicated high-speed interconnect so they behave like one larger machine. The pod is how a model too big for one chip is spread across many without going through an ordinary datacenter network.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  lpu: {
    term: 'LPU (language processing unit)',
    def: 'Groq’s inference accelerator, built around a large pool of on-chip SRAM rather than HBM. It is designed for very low, predictable latency, and it needs many chips because each one holds only about 230 MB of weights.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  'wafer-scale': {
    term: 'Wafer-scale engine',
    def: 'A chip built from a whole silicon wafer instead of a die cut from it. Cerebras’s WSE-3 puts many cores and tens of gigabytes of SRAM on one wafer, removing chip-to-chip hops inside the wafer but still leaving a hard capacity limit.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  rocm: {
    term: 'ROCm',
    def: 'AMD’s software stack for GPU computing, the counterpart to CUDA. PyTorch ships a ROCm build and many serving engines support it, but coverage of the newest kernels and quantisation formats tends to lag the CUDA ecosystem at the time of writing.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  xla: {
    term: 'XLA',
    def: 'A compiler that turns a machine-learning graph into optimised code for a target accelerator, chiefly Google TPUs. It fuses operations, chooses memory layouts and schedules the matrix engine; JAX is the Python front end most TPU work uses.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  'neuron-sdk': {
    term: 'Neuron SDK',
    def: 'AWS’s compiler and runtime for Trainium and Inferentia. Models are compiled into Neuron graphs, so the supported architecture list and the maturity of each kernel are the practical limits on what runs fast.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
  trainium: {
    term: 'AWS Trainium / Inferentia',
    def: 'AWS’s custom inference and training ASICs, reached through the Neuron SDK. Their pitch is cost per token inside AWS rather than a headline peak-FLOPs number, and the trade is that model and kernel support arrive on AWS’s schedule.',
    lesson: 'gpu-first-principles/other-accelerators',
  },
};
