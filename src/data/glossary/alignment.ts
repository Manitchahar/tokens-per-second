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
  'preference-data': {
    term: 'Preference data',
    def: 'Records of which of two (or K) model answers a person, or a judge model, prefers for the same prompt: (prompt, chosen, rejected). Cheaper and more consistent to collect than written ideal answers.',
    lesson: 'alignment/reward-models',
  },
  'reward-model': {
    term: 'Reward model',
    def: 'A transformer whose vocabulary head is replaced by a single output, trained on preference data to score how much people would like an answer to a prompt. Only differences between its scores for the same prompt carry meaning.',
    lesson: 'alignment/reward-models',
  },
  'bradley-terry': {
    term: 'Bradley-Terry model',
    def: 'P(A preferred to B) = σ(r_A − r_B): scores are log-odds, and their difference sets the odds of winning. Its negative log-likelihood, −log σ(r_w − r_l), is the standard reward-model loss.',
    lesson: 'alignment/reward-models',
  },
  'pairwise-accuracy': {
    term: 'Pairwise accuracy',
    def: 'The fraction of held-out comparisons where a reward model scores the chosen answer above the rejected one. 50% is chance; the practical ceiling is human agreement, roughly 73–77% in InstructGPT’s data.',
    lesson: 'alignment/reward-models',
  },
  'best-of-n': {
    term: 'Best-of-N',
    def: 'Generate N answers, score each with a reward model, return the highest. Costs N generations plus N scoring passes; the larger N, the more it also selects for the reward model’s mistakes.',
    lesson: 'alignment/reward-models',
  },
};
