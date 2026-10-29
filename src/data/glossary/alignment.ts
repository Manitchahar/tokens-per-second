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
  rlhf: {
    term: 'RLHF',
    def: 'Reinforcement learning from human feedback: the policy samples answers, a reward model trained on human comparisons scores them, and an RL algorithm (classically PPO) makes high-scoring answers likelier, with a KL penalty to the SFT model.',
    lesson: 'alignment/rlhf-instructgpt',
  },
  policy: {
    term: 'Policy',
    def: 'Reinforcement-learning name for the model being trained: given the prompt and the tokens so far, it chooses the next token. In RLHF the policy starts as the SFT model.',
    lesson: 'alignment/rlhf-instructgpt',
  },
  'kl-penalty': {
    term: 'KL penalty (KL leash)',
    def: 'β × log(π_RL(y|x) / π_SFT(y|x)), subtracted from the reward during RLHF. Summed over tokens, it grows as the policy drifts from the SFT model, keeping it where the reward model’s scores can be trusted.',
    lesson: 'alignment/rlhf-instructgpt',
  },
  'reward-hacking': {
    term: 'Reward hacking',
    def: 'When optimisation raises a learned reward without raising real quality, by exploiting the reward model’s mistakes: padding, flattery, a tone it over-rewards. Its score keeps rising while human judgement peaks and falls.',
    lesson: 'alignment/rlhf-instructgpt',
  },
  ppo: {
    term: 'PPO (proximal policy optimisation)',
    def: 'The RL algorithm InstructGPT used: raise the probability of sampled answers in proportion to their advantage (reward minus a value-model baseline), with the probability ratio clipped to [1 − ε, 1 + ε] so each update stays small.',
    lesson: 'alignment/rlhf-instructgpt',
  },
  'alignment-tax': {
    term: 'Alignment tax',
    def: 'Performance lost on skills the reward model never checks (InstructGPT: SQuAD, DROP, HellaSwag, translation) when a model is tuned to human preferences. PPO-ptx reduced it by mixing pretraining gradients into each RL update.',
    lesson: 'alignment/rlhf-instructgpt',
  },
  dpo: {
    term: 'DPO (Direct Preference Optimization)',
    def: 'Preference tuning with one supervised-style loss, −log σ(β[log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)]), derived from the closed-form optimum of KL-regularised RLHF. No reward model, no value model, no sampling loop.',
    lesson: 'alignment/dpo',
  },
  'implicit-reward': {
    term: 'Implicit reward',
    def: 'β · log(π(y|x) / π_ref(y|x)): the reward a policy implicitly assigns an answer, by how much more likely it makes it than the reference model does. DPO trains this quantity with the Bradley-Terry loss.',
    lesson: 'alignment/dpo',
  },
};
