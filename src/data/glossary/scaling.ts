import type { GlossaryEntry } from '../glossary';

// Terms Module 3B introduces. FLOPs, MFU, perplexity, open weights and decoder-only models
// are defined elsewhere (glossary.ts, profiling.ts, pretraining) and are not repeated here.

export const terms: Record<string, GlossaryEntry> = {
  // ── 3B.1 · Counting training compute
  'training-compute': { term: 'Training compute (C ≈ 6ND)', def: 'The total FLOPs needed to train a model: about 6 × parameters (N) × training tokens (D). The forward pass costs 2N per token and the backward pass 4N. GPT-3: 6 × 175B × 300B ≈ 3.15 × 10²³ FLOPs.', lesson: 'scaling/training-compute' },
  'backward-pass': { term: 'Backward pass', def: 'The half of a training step that works out how much every weight contributed to the loss. For each weight matrix it does two matrix multiplies the size of the forward one: one sends the error to the layer below, one gives each weight its gradient. So it costs about twice the forward pass.', lesson: 'scaling/training-compute' },
  'gpu-hour': { term: 'GPU-hour', def: 'One GPU busy for one hour: the unit training runs and cloud bills are priced in. GPU-hours = training FLOPs ÷ (peak FLOP/s × MFU) ÷ 3,600.', lesson: 'scaling/training-compute' },
  'activation-checkpointing': { term: 'Activation checkpointing', def: 'Saving memory in training by discarding intermediate activations in the forward pass and recomputing them during the backward pass. The GPU does about 8N FLOPs per token instead of 6N, but MFU still counts only the 6N the model needs.', lesson: 'scaling/training-compute' },

  // ── 3B.2 · GPT-3: learning from the prompt
  'in-context-learning': { term: 'In-context learning', def: 'A model picking up a task from instructions and examples in its prompt, with no change to its weights. The examples are read in one forward pass and forgotten afterwards. It gets much stronger as models get bigger.', lesson: 'scaling/gpt3-in-context' },
  'zero-shot': { term: 'Zero-shot', def: 'Prompting with only a description of the task and the input, no solved examples. One-shot adds one example.', lesson: 'scaling/gpt3-in-context' },
  'few-shot': { term: 'Few-shot', def: 'Prompting with K solved examples (typically 10–100 for GPT-3) before the real input. Every example costs prompt tokens, so prefill work, on every query.', lesson: 'scaling/gpt3-in-context' },
};
