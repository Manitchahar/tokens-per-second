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
};
