// Module 03 · From Neurons to Transformers
// Terms introduced while building the transformer from neurons up.
// Plain words first; the precise definition follows.

import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  'transformer-block': {
    term: 'Transformer block',
    def: 'One layer of a transformer: RMSNorm, attention, residual add, RMSNorm, MLP, residual add. It takes a hidden vector and returns a vector of the same size. Llama-3.1-8B has 32 of them stacked.',
    lesson: 'transformers/mlp-norm-residual',
  },
  'forward-pass': {
    term: 'Forward pass',
    def: 'One complete trip through the model: embed, every block, the output head, sample. Nothing is stored or learned. "One forward pass per generated token" is the unit of work in decode.',
    lesson: 'transformers/forward-pass',
  },
  'hidden-state': {
    term: 'Hidden state',
    def: 'The vector that represents a token as it moves through the model — 4,096 numbers for Llama-3.1-8B. Attention and the MLP both read it and write their result back into it.',
    lesson: 'transformers/embeddings-positions',
  },
  'output-head': {
    term: 'Output head (lm_head)',
    def: 'The final matrix that turns the last hidden state into one score per vocabulary token: [hidden × vocab], so 4096 × 128,256 for Llama-3.1-8B. 525 million parameters, fully read on every decode step.',
    lesson: 'transformers/forward-pass',
  },
  'tied-embeddings': {
    term: 'Tied embeddings',
    def: 'When the embedding table and the output head are the same weights, halving the parameter count of both. Qwen2.5-0.5B and Llama-3.2-3B tie; Llama-3.1-8B does not.',
    lesson: 'transformers/neural-networks',
  },

  'bpe': {
    term: 'Byte-pair encoding (BPE)',
    def: 'How tokenizers are built: start from bytes, repeatedly merge the most frequent adjacent pair, and stop at a fixed vocabulary size. Every merge is one learned rule.',
    lesson: 'transformers/tokens',
  },
  'byte-fallback': {
    term: 'Byte fallback',
    def: 'Every tokenizer keeps one token for each of the 256 possible byte values, so any input can be spelled out one byte at a time. That is why real vocabulary sizes are multiples of 256 and why no tokenizer needs an "unknown" token.',
    lesson: 'transformers/tokens',
  },
  'special-tokens': {
    term: 'Special tokens',
    def: 'Reserved vocabulary entries such as <|im_start|> or <|endoftext|> that the chat template injects to mark roles and boundaries. They are real tokens: they consume context and cost KV cache.',
    lesson: 'transformers/tokens',
  },

  'attention-head': {
    term: 'Attention head',
    def: 'One independent slice of the attention operation. Llama-3.1-8B runs 32 of them per layer, in parallel, each 128 numbers wide. Each produces its own average of the value vectors.',
    lesson: 'transformers/multi-head-gqa',
  },
  'kv-head': {
    term: 'Key/value head',
    def: 'A head that produces keys and values rather than queries. Modern models have fewer KV heads than query heads — 8 against 32 in Llama-3.1-8B — which is what shrinks the KV cache.',
    lesson: 'transformers/multi-head-gqa',
  },
  'head-dim': {
    term: 'head_dim',
    def: 'The width of one attention head: hidden size divided by the number of heads. It stays at 128 across most modern models, so growing the model adds heads rather than widening each one.',
    lesson: 'transformers/multi-head-gqa',
  },
  mha: {
    term: 'Multi-head attention (MHA)',
    def: 'Every query head owns its own key and value head, so n_kv = n_q. Used by GPT-2, GPT-3 and Llama-2. The largest KV cache of the three options.',
    lesson: 'transformers/multi-head-gqa',
  },
  mqa: {
    term: 'Multi-query attention (MQA)',
    def: 'All query heads share one key and value head. The KV cache shrinks by the query-head count — up to 32× — with a small quality cost. Used by Grok-1 and some Llama-2-70B releases.',
    lesson: 'transformers/multi-head-gqa',
  },
  'mla': {
    term: 'Multi-head latent attention (MLA)',
    def: 'Introduced with DeepSeek-V2 and used in DeepSeek-V3. Instead of caching full K and V, each layer caches one small compressed latent per token — 512 numbers for DeepSeek-V3 — plus a 64-number RoPE key, and rebuilds per-head K and V with learned up-projections.',
    lesson: 'transformers/multi-head-gqa',
  },
  'weight-absorption': {
    term: 'Weight absorption (MLA)',
    def: 'Folding MLA’s K and V up-projections into the query and output projections when the weights are loaded, so the decode kernel attends in the small latent space and never writes reconstructed K or V to HBM. It makes MLA’s extra compute a one-time cost.',
    lesson: 'transformers/multi-head-gqa',
  },
  'sliding-window-attention': {
    term: 'Sliding-window attention',
    def: 'Each token attends only to the most recent W tokens, so a local layer’s KV cache is capped at W entries however long the context is. Hybrid models interleave local layers with a few global layers that see the whole context.',
    lesson: 'transformers/multi-head-gqa',
  },
  'attention-scale': {
    term: 'The 1/√d in attention',
    def: 'Scores are divided by the square root of the head width before the softmax. Dot products over d terms get large, and a softmax over large scores goes one-hot, killing gradients. It is a fixed softmax temperature: 11.31 at d = 128.',
    lesson: 'transformers/attention',
  },
  'rope-base': {
    term: 'RoPE base (θ)',
    def: 'The number that sets RoPE’s frequency spread: 10,000 for Llama-2, 500,000 for Llama-3.1, 1,000,000 for Qwen2.5. A small base aliases — positions 16 apart can score higher than positions 8 apart — and caps the usable context.',
    lesson: 'transformers/embeddings-positions',
  },
  'qk-norm': {
    term: 'QK-norm',
    def: 'A small RMSNorm applied to every query head and key head before RoPE. Qwen2 models use it; Llama does not. Costs 2 numbers per head and one elementwise kernel.',
    lesson: 'transformers/embeddings-positions',
  },

  silu: {
    term: 'SiLU (swish)',
    def: 'The sigmoid times the input: SiLU(z) = z · σ(z), where σ(z) = 1/(1+e^−z). Keeps a little negative signal instead of clipping it to zero. Used in Llama, Qwen2 and Mistral.',
    lesson: 'transformers/neural-networks',
  },
  gelu: {
    term: 'GELU',
    def: 'A smooth approximation to ReLU built from tanh. Used by GPT-2 and GPT-3 and many small models. SiLU is the modern replacement and is nearly identical in shape.',
    lesson: 'transformers/neural-networks',
  },
  swiglu: {
    term: 'SwiGLU',
    def: 'The gated MLP used by Llama and Qwen2: down( SiLU(gate(x)) ⊙ up(x) ). Two branches mean one is a learned gate on the other. It needs three matrices instead of two, so the break-even width is 8/3 of the hidden size.',
    lesson: 'transformers/mlp-norm-residual',
  },
  'pre-norm': {
    term: 'Pre-norm',
    def: 'Normalising the copy that goes into a sublayer and adding the result back, leaving the residual stream itself untouched. Leaves an unbroken identity path, which is why every current model uses it. Costs 2L + 1 norms.',
    lesson: 'transformers/mlp-norm-residual',
  },

  logit: {
    term: 'Logit',
    def: 'One raw score the output head gives a vocabulary token, before any softmax. Unbounded and signless; only differences between logits matter, because softmax is shift-invariant.',
    lesson: 'transformers/sampling',
  },
  sampling: {
    term: 'Sampling',
    def: 'Turning 128,256 logits into one token: softmax, filter by temperature / top-k / top-p, renormalise, and draw from the cumulative sum. About 0.26 MFLOP — free next to the 15 GFLOP of weights in the same step.',
    lesson: 'transformers/sampling',
  },
  'top-k': {
    term: 'Top-k sampling',
    def: 'Keep only the k most probable tokens and zero the rest. k = 1 is greedy decoding. A fixed k is a poor fit: too loose on an easy question, too tight on a hard one.',
    lesson: 'transformers/sampling',
  },
  'min-p': {
    term: 'Min-p sampling',
    def: 'Keep every token with probability at least ε × p_max. One threshold and no sort, unlike top-p. On a peaked distribution it keeps 1–2 tokens, on a flat one dozens.',
    lesson: 'transformers/sampling',
  },
  'greedy-decoding': {
    term: 'Greedy decoding',
    def: 'Always take the highest-scoring token, never sample. Deterministic and reproducible, but prone to repetition loops and bland text. Equivalent to top-k with k = 1.',
    lesson: 'transformers/sampling',
  },
  seed: {
    term: 'Seed',
    def: 'The number that makes a sampling run reproducible: same prompt, same seed, same tokens. Its existence is why sampling and determinism are not in conflict.',
    lesson: 'transformers/sampling',
  },
  entropy: {
    term: 'Entropy',
    def: 'The number of bits of choice left in a distribution: 0 bits means one outcome is certain, log₂(V) bits means no preference at all. Measured in bits, it is far more meaningful than a temperature value.',
    lesson: 'transformers/sampling',
  },
  word2vec: {
    term: 'Word2Vec',
    def: 'The 2013 technique that mapped words to dense vectors in continuous space based on local context. Words with similar meanings landed close together, but each word had one fixed vector regardless of context.',
    lesson: 'transformers/neural-networks',
  },
  'recurrent-neural-network': {
    term: 'Recurrent neural network (RNN)',
    def: 'A sequence model that updates a hidden state step by step: token t strictly requires the state from token t−1. This created a sequential bottleneck that prevented parallel processing on GPUs.',
    lesson: 'transformers/neural-networks',
  },
  'decoder-only': {
    term: 'Decoder-only transformer',
    def: 'A transformer architecture using only causal self-attention layers to predict the next token given prior tokens. It forms the foundation of modern autoregressive LLMs like GPT, Llama, and Qwen.',
    lesson: 'transformers/neural-networks',
  },
  'instruction-tuning': {
    term: 'Instruction tuning',
    def: 'Fine-tuning a base pre-trained language model on demonstrations of questions and answers so it follows instructions and converses rather than simply completing text.',
    lesson: 'transformers/neural-networks',
  },
  loss: {
    term: 'Loss',
    def: 'One number that says how wrong the model is on some data: zero is perfect, bigger is worse. Mean squared error for predicting numbers; cross-entropy, −ln(probability given to the right token), for language models. Training exists to make it smaller.',
    lesson: 'transformers/how-models-learn',
  },
  gradient: {
    term: 'Gradient',
    def: 'The slope of the loss with respect to every weight at once: one number per weight saying how fast the loss rises if that weight is nudged up. It has as many entries as the model has parameters, and it points uphill.',
    lesson: 'transformers/how-models-learn',
  },
  'gradient-descent': {
    term: 'Gradient descent',
    def: 'The training rule w ← w − η·gradient: step every weight a little in the direction that lowers the loss, recompute the slope, repeat. A ball rolling downhill on the loss surface.',
    lesson: 'transformers/how-models-learn',
  },
  'learning-rate': {
    term: 'Learning rate (η)',
    def: 'How big a step gradient descent takes. Too large and every step overshoots the bottom until the loss blows up (divergence); too small and training crawls. LLaMA-7B peaked at 3 × 10⁻⁴ with warm-up and cosine decay.',
    lesson: 'transformers/how-models-learn',
  },
  backpropagation: {
    term: 'Backpropagation',
    def: 'Computing the gradient of the loss for every weight in one backward sweep, by the chain rule: start at the loss and multiply local slopes layer by layer back to the input. Costs about twice a forward pass.',
    lesson: 'transformers/how-models-learn',
  },
  'mini-batch': {
    term: 'Mini-batch / SGD',
    def: 'Estimating the gradient from a small random sample of the data instead of all of it. Stochastic gradient descent (SGD) takes one cheap, noisy step per mini-batch; the noise shrinks like 1/√(batch size). LLaMA used batches of 4M tokens.',
    lesson: 'transformers/how-models-learn',
  },
  adam: {
    term: 'Adam / AdamW',
    def: 'The optimizer almost every LLM is trained with. It keeps a running average of each weight’s gradient (momentum) and of its square, and divides one by the square root of the other, giving every weight its own step size. AdamW adds decoupled weight decay. Two extra numbers per weight.',
    lesson: 'transformers/how-models-learn',
  },
  'training-step': {
    term: 'Training step',
    def: 'One turn of the training loop: forward pass on a mini-batch, compute the loss, backward pass for the gradients, optimizer update. About 3× the compute of a forward pass, which is where the 6 in C ≈ 6ND comes from.',
    lesson: 'transformers/how-models-learn',
  },
};
