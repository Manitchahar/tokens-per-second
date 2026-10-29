import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  'catastrophic-forgetting': {
    term: 'Catastrophic forgetting',
    def: 'The tendency of a neural network to abruptly overwrite previously learned representations (such as general reasoning, coding, or safety alignment) when fine-tuned on a narrow domain dataset.',
    lesson: 'finetuning/when-to-finetune',
  },
  activation: {
    term: 'Activation memory',
    def: 'Intermediate forward-pass tensors (attention scores, LayerNorm inputs, MLP projections) that must be preserved in GPU VRAM until consumed during backpropagation.',
    lesson: 'finetuning/training-memory',
  },
  qlora: {
    term: 'QLoRA',
    def: 'Efficient fine-tuning combining 4-bit NormalFloat (NF4) base weight storage, Double Quantization of scale constants, and Paged Optimizers to train models on a single GPU without quality degradation.',
    lesson: 'finetuning/qlora',
  },
  nf4: {
    term: 'NF4 (NormalFloat4)',
    def: 'An information-theoretically optimal 4-bit data type for zero-mean Gaussian distributed weights, partitioning the bell curve into 16 bins that each contain exactly 6.25% of the parameter probability mass.',
    lesson: 'finetuning/qlora',
  },
  'double-quantization': {
    term: 'Double Quantization (DQ)',
    def: 'Quantizing the FP32 block-quantization scale constants into 8-bit FP with block size 256, slashing scale memory overhead from 0.500 down to 0.127 bits per parameter.',
    lesson: 'finetuning/qlora',
  },
  'paged-optimizer': {
    term: 'Paged Optimizer',
    def: 'An optimizer implementation using CUDA Unified Virtual Memory to automatically page inactive AdamW states to host CPU RAM across PCIe during transient activation memory spikes.',
    lesson: 'finetuning/qlora',
  },
  'effective-batch-size': {
    term: 'Effective batch size',
    def: 'The total number of sequences processed per optimizer parameter update step, computed as micro-batch size × gradient accumulation steps × number of GPU workers.',
    lesson: 'finetuning/finetuning-tactics',
  },
  'task-vector': {
    term: 'Task vector (τ)',
    def: 'The parameter displacement vector defined as fine-tuned weights minus pre-trained base weights: τ = θ_finetuned − θ_base. Isolates the directional update responsible for a specific capability.',
    lesson: 'finetuning/model-merging',
  },
  'linear-mode-connectivity': {
    term: 'Linear Mode Connectivity (LMC)',
    def: 'The empirical observation that models fine-tuned from the same shared pre-trained initialization reside in the same convex loss basin, with zero loss barrier along the linear path between them.',
    lesson: 'finetuning/model-merging',
  },
  'model-soup': {
    term: 'Model Soup',
    def: 'A simple parameter-averaging technique that combines multiple checkpoints fine-tuned from the same base model under different hyperparameters to improve generalization and cancel out noise.',
    lesson: 'finetuning/model-merging',
  },
  slerp: {
    term: 'SLERP (Spherical Linear Interpolation)',
    def: 'A geometric interpolation method that rotates along the constant-radius spherical geodesic between two weight vectors, preventing high-dimensional parameter norm collapse.',
    lesson: 'finetuning/model-merging',
  },
  'ties-merging': {
    term: 'TIES-Merging',
    def: 'A multi-model merging algorithm that resolves directional interference in three steps: Trimming low-magnitude deltas, Electing sign consensus per parameter, and Disjoint merging aligned weights.',
    lesson: 'finetuning/model-merging',
  },
  dare: {
    term: 'DARE (Drop And REscale)',
    def: 'A model merging technique that randomly zeroes out up to 90% of task vector deltas using a Bernoulli mask and rescales surviving weights by 1/(1−p) to eliminate cross-task interference.',
    lesson: 'finetuning/model-merging',
  },
  'model-merging': {
    term: 'Model Merging',
    def: 'The practice of combining multiple fine-tuned models into a single checkpoint with zero additional training data, zero backpropagation, and zero GPU compute.',
    lesson: 'finetuning/model-merging',
  },
};
