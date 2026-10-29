import type { GlossaryEntry } from '../glossary';

// New terms introduced in Module 16 · Speculative Decoding.
// Existing keys live in ../glossary.ts — do not duplicate those keys here.
export const terms: Record<string, GlossaryEntry> = {
  'speculative-decoding': {
    term: 'Speculative decoding',
    def: 'Producing several tokens per forward pass by letting a cheap model guess them and the real model check them all at once. The guesses that match are kept; the rest are thrown away. The output is exactly what plain sampling would have produced — it is not an approximation.',
    lesson: 'speculative-decoding/draft-and-verify',
  },
  'speculative-sampling': {
    term: 'Speculative sampling',
    def: 'The exact form of speculative decoding, where correctness is provable: accept a drafted token with probability min(1, p(x)/q(x)) and on rejection sample from the corrected distribution normalize(max(0, p - q)). The two branches sum back to p exactly.',
    lesson: 'speculative-decoding/draft-and-verify',
  },
  'draft-model': {
    term: 'Draft model',
    def: 'A smaller model that proposes tokens for a larger target model, one sequential step at a time. Llama-3.2-1B drafting Llama-3.1-8B is the standard pairing. It needs no training, only a compatible smaller checkpoint.',
    lesson: 'speculative-decoding/spec-methods',
  },
  'speculative-verify': {
    term: 'Verification pass',
    def: 'The single forward pass in which the target model scores every position of a draft block at once. Because a decode step is memory-bound and weight traffic does not depend on the number of positions, checking γ+1 positions costs about what one costs — which is the whole trick.',
    lesson: 'speculative-decoding/draft-and-verify',
  },
  'acceptance-rate': {
    term: 'Acceptance rate (a)',
    def: 'How often a drafted token is one the target model would also have produced. A property of your text and your model pair, not of the hardware. Speculation is faster exactly when this exceeds the draft cost ratio.',
    lesson: 'speculative-decoding/spec-math',
  },
  'draft-cost-ratio': {
    term: 'Draft cost ratio (ρ)',
    def: 'The time to draft one token divided by the time of one target decode step. A property of your hardware and model sizes. Larger than the parameter ratio, because both models pay the same fixed per-step overhead.',
    lesson: 'speculative-decoding/spec-math',
  },
  'bonus-token': {
    term: 'Bonus token',
    def: 'The extra token the target emits when every draft in a block is accepted: it comes from position γ+1, which the draft never predicted, scored in the same verification pass. When a draft is rejected the target emits a corrected token at that position instead. Either way every block emits at least one token.',
    lesson: 'speculative-decoding/draft-and-verify',
  },
  'rejection-sampling-correction': {
    term: 'Rejection correction',
    def: 'On a rejected draft, sampling from normalize(max(0, p - q)) instead of from p. It keeps exactly the probability mass the drafter failed to cover, which is what makes speculative decoding output-equivalent to plain sampling.',
    lesson: 'speculative-decoding/spec-math',
  },
  'tokens-per-block': {
    term: 'Tokens per block',
    def: '1 + a·(1 - a^γ)/(1 - a) = (1 - a^(γ+1))/(1 - a): the expected tokens a speculative block emits, the accepted drafts plus one token from the target. It rises with γ and saturates at 1/(1 - a), a ceiling set by how predictable the text is rather than by the block size.',
    lesson: 'speculative-decoding/spec-math',
  },
  'break-even-acceptance': {
    term: 'Break-even acceptance rate',
    def: 'The acceptance rate at which speculation exactly pays for itself. With verification costing about one decode step, that is where the keep rate (the fraction of the γ drafted tokens that are accepted) equals the draft cost ratio ρ. At γ = 1 it is simply a = ρ; it rises with γ, to about 0.34 at γ = 4 when ρ = 0.125.',
    lesson: 'speculative-decoding/spec-math',
  },
  'lookahead-decoding': {
    term: 'Lookahead decoding',
    def: 'A drafting method that speculates several steps ahead from a position the target already computed, using Jacobi-style iterations to refine the guesses in parallel instead of sequentially. It needs no second model and no training.',
    lesson: 'speculative-decoding/spec-methods',
  },
  'prompt-lookup-decoding': {
    term: 'Prompt lookup decoding',
    def: 'Drafting by copying from the longest matching span of the prompt or earlier output. No model runs at all, so its cost ratio is near zero, which makes it very strong on extractive, templated and code-editing traffic.',
    lesson: 'speculative-decoding/spec-methods',
  },
  medusa: {
    term: 'Medusa',
    def: 'A drafting method that adds a few trained linear heads to the target model, predicting future tokens straight from the final hidden state. It avoids holding a second model in memory but must be trained, and predicts from less context.',
    lesson: 'speculative-decoding/spec-methods',
  },
  eagle: {
    term: 'EAGLE',
    def: 'A drafting method that predicts the next hidden state rather than the next token, then lets the target model verify a block built from those predicted states in parallel. The extra information gives it the highest acceptance of the cheap methods.',
    lesson: 'speculative-decoding/spec-methods',
  },
  'multi-token-prediction': {
    term: 'Multi-token prediction (MTP)',
    def: 'Training a model with heads that emit several future tokens from one hidden state, so drafting is a single forward pass instead of γ sequential steps. Drafting cost stops scaling with γ, at the price of a changed training objective.',
    lesson: 'speculative-decoding/spec-methods',
  },
  'speculation-saves-weights': {
    term: 'Batching and speculation are substitutes',
    def: 'Both techniques amortise the same weight read. The more you have already batched, the less speculation has left to give — and at large compute-bound batch it gives nothing at all.',
    lesson: 'speculative-decoding/when-spec-wins',
  },
};