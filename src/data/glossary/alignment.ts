import type { GlossaryEntry } from '../glossary';

// Terms Module 3C introduces. Keys already defined elsewhere (kl-divergence, eos, chat-template,
// instruction-tuning, rollout, greedy, ...) are reused, not repeated.
export const terms: Record<string, GlossaryEntry> = {
  'base-model': {
    term: 'Base model',
    def: 'A model after pretraining and nothing else. It continues documents: given a question it predicts how pages containing that question usually go on, which is often more questions. It knows a lot and answers nobody.',
    lesson: 'alignment/base-vs-assistant',
  },
  'post-training': {
    term: 'Post-training',
    def: 'Everything done to a model after pretraining: supervised finetuning, preference tuning, safety training. It used under 2% of InstructGPT’s training compute and changes behaviour far more than knowledge.',
    lesson: 'alignment/base-vs-assistant',
  },
  demonstration: {
    term: 'Demonstration',
    def: 'One supervised finetuning example: a prompt paired with the response a skilled person would want the model to give. InstructGPT’s SFT set held 12,725 of them, LIMA’s just 1,000.',
    lesson: 'alignment/sft',
  },
  sft: {
    term: 'Supervised finetuning (SFT)',
    def: 'Continuing to train a pretrained model with the same next-token loss, but on demonstrations formatted with a chat template, usually with the loss computed only on the response tokens. Teaches the format of answering and where to stop.',
    lesson: 'alignment/sft',
  },
  'loss-masking': {
    term: 'Loss masking',
    def: 'Excluding some positions from the training loss by setting their labels to −100, so cross-entropy skips them. In SFT the prompt (and padding) are masked so only response tokens are learned; the masked tokens still go through the forward pass.',
    lesson: 'alignment/sft',
  },
  'superficial-alignment': {
    term: 'Superficial alignment hypothesis',
    def: 'The LIMA paper’s claim that a model’s knowledge and abilities come almost entirely from pretraining, and that alignment mostly teaches which format and style to use with users. Hence 1,000 excellent examples can go a long way.',
    lesson: 'alignment/sft',
  },
};
