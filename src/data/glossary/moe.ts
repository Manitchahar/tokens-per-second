import type { GlossaryEntry } from '../glossary';

// New vocabulary introduced in Module 19. Terms the learner already met in
// earlier modules (mlp, kv-cache, decode, paged-attention, ...) live in
// src/data/glossary.ts or the per-module glossary files.
//
// `pipeline-parallelism` and `tensor-parallelism` are defined here rather than in
// module 18 because this module references them directly and Module 18 is written
// by another author; if that author also defines them, their file simply wins.
export const terms: Record<string, GlossaryEntry> = {
  'mixture-of-experts': {
    term: 'Mixture-of-Experts (MoE)',
    def: 'A feed-forward layer replaced by a bank of smaller independent "experts" plus a router. Each token is processed by only a few of them, so the model can hold far more parameters than it reads per token. Attention is unchanged; only the block’s feed-forward network is swapped.',
    lesson: 'moe/moe-routing',
  },
  expert: {
    term: 'Expert',
    def: 'One independent feed-forward network inside a mixture-of-experts layer. With E experts of intermediate width 4d/E, the whole bank has exactly as many parameters as the single dense network of width 4d it replaced — but a token only reads the few it needs.',
    lesson: 'moe/moe-routing',
  },
  router: {
    term: 'Router',
    def: 'A single matrix of shape (E+1) x hidden inside a mixture-of-experts layer that turns a token into one score per expert. It costs E·hidden parameters — negligible against the bank — which is why designs with thousands of experts are affordable.',
    lesson: 'moe/moe-routing',
  },
  'top-k-routing': {
    term: 'Top-k routing',
    def: 'The selection step: softmax the router scores across all experts, keep the k highest, and renormalise those k probabilities to sum to 1. k is the model’s main quality-versus-work dial — k=1 keeps dense FLOPs, k=2 is the common production default.',
    lesson: 'moe/moe-routing',
  },
  sparsity: {
    term: 'Sparsity',
    def: 'Total parameters divided by active parameters. It is the headline number for a mixture-of-experts model: about 3.6× for Mixtral 8x7B, 5.4× for DeepSeek-MoE-16B, 9.3× for Qwen3-30B-A3B. Higher sparsity means more parameters than the compute can reach, not a slower model.',
    lesson: 'moe/moe-routing',
  },
  'active-params': {
    term: 'Active parameters',
    def: 'The parameters one token actually reads: all of attention and embeddings, the top-k routed experts, and every shared expert in full. Substituting active for total parameters in the decode-step model is the whole latency story of MoE.',
    lesson: 'moe/moe-routing',
  },
  'shared-expert': {
    term: 'Shared expert',
    def: 'An always-on expert processed by every token in a layer, alongside the routed ones. It exists because routed experts are not guaranteed to fire for every token, so the shared expert provides a floor of work per token and keeps the layer from collapsing under an unpredictable router.',
    lesson: 'moe/moe-routing',
  },
  'load-imbalance': {
    term: 'Load imbalance',
    def: 'The situation where experts do not receive equal shares of tokens. Because trained experts specialise, routers develop favourites, and this is a consequence rather than a defect. Measured as the imbalance ratio: hottest expert divided by the balanced share.',
    lesson: 'moe/moe-routing',
  },
  straggler: {
    term: 'Straggler',
    def: 'The rank in a synchronous collective that takes longest, and therefore sets the whole step’s duration. In an MoE, a hot expert makes its card a straggler, so one expert holding twice its share roughly doubles the step time rather than adding a quarter of it.',
    lesson: 'moe/expert-parallelism',
  },
  'expert-parallelism': {
    term: 'Expert parallelism (EP)',
    def: 'Sharding a mixture-of-experts bank across GPUs by giving each card E/g of the experts. Every parameter becomes resident somewhere. The GPU count must divide the expert count, which is why published models use large power-of-two expert counts.',
    lesson: 'moe/expert-parallelism',
  },
  'all-to-all': {
    term: 'All-to-all',
    def: 'A collective in which every rank sends a different slice of its data to every other rank, once. Expert parallelism needs one per mixture-of-experts layer, because a token’s expert is usually on a different card than the token.',
    lesson: 'moe/expert-parallelism',
  },
  'all-to-all-volume': {
    term: 'All-to-all volume',
    def: 'Bytes crossing the fabric per layer: T tokens × f_remote × d × b for the dispatch, doubled for the return, times the number of MoE layers. Under balanced routing f_remote = 1 − 1/g; routing imbalance multiplies it by the load ratio.',
    lesson: 'moe/expert-parallelism',
  },
  'hybrid-parallelism': {
    term: 'Hybrid parallelism',
    def: 'Combining expert parallelism with tensor parallelism, EP outside and TP inside each expert. Required because EP cannot shard a single expert: narrow experts favour a wide EP degree, wide experts force TP. Both degrees must divide the GPU count.',
    lesson: 'moe/expert-parallelism',
  },
  'capacity-factor': {
    term: 'Capacity factor',
    def: 'The multiple of the perfectly-balanced share of slots provisioned per expert, C = ceil(c · kT/E). Tokens beyond it are dropped or overflowed. It converts a ragged problem into a rectangular one; zero overflow needs c at least as large as the imbalance.',
    lesson: 'moe/expert-parallelism',
  },
  'capacity-drop': {
    term: 'Capacity drop',
    def: 'What happens when a token is routed to an expert with no free slot: the assignment is either discarded or sent to the next-best expert. It is the quality cost of a rectangular kernel, and at capacity factor c with imbalance I it affects max(0, 1 − c/I) of the hot expert’s tokens.',
    lesson: 'moe/expert-parallelism',
  },
  'auxiliary-loss': {
    term: 'Auxiliary loss',
    def: 'A load-balancing term added to the training objective, α·E·Σ fₑ·Pₑ, where fₑ is the fraction of tokens routed to expert e and Pₑ its mean router probability. Without it a mixture-of-experts collapses onto a few experts and the rest go dead.',
    lesson: 'moe/moe-routing',
  },
  gini: {
    term: 'Gini coefficient',
    def: 'The area between a sorted load distribution’s cumulative curve and the uniform diagonal, normalised. One number in 0 to 1 summarising expert load: 0 is perfect equality, 1 is one expert taking everything. Above 0.3 expect stragglers; below 0.10 suspect the router never specialised.',
    lesson: 'moe/moe-routing',
  },
  'remote-fraction': {
    term: 'Remote fraction',
    def: 'The share of a rank’s expert activations that must cross a device boundary: (1 − 1/g) · I under balanced routing and imbalance I. It is the whole of the all-to-all cost, because each activation is only one hidden-sized vector.',
    lesson: 'moe/expert-parallelism',
  },
  'hot-expert': {
    term: 'Hot expert',
    def: 'An expert receiving substantially more than its balanced share of tokens, usually because the router has specialised around it. It is simultaneously a compute straggler, a communication hotspot and a buffer-overflow risk — the three costs of specialisation arriving together.',
    lesson: 'moe/expert-parallelism',
  },
  'switch-transformer': {
    term: 'Switch Transformer',
    def: 'The paper that made mixture-of-experts mainstream, using 2048 experts with top-1 routing and an auxiliary load-balancing loss. Its defining property is that per-token expert FLOPs are unchanged from the dense block it replaced.',
    lesson: 'moe/moe-routing',
  },
  'pipeline-parallelism': {
    term: 'Pipeline parallelism',
    def: 'Splitting a model by depth so each GPU holds a contiguous range of layers, and neighbouring GPUs pass activations forward. It reduces per-GPU memory but introduces pipeline bubbles unless enough independent work is batched to fill every stage.',
    lesson: 'moe/expert-parallelism',
  },
  'tensor-parallelism': {
    term: 'Tensor parallelism',
    def: 'Splitting an individual matrix across GPUs so each holds a slice of every weight, combined by an all-reduce after each layer. It shards what expert parallelism cannot — a single expert — but pays a collective per layer, so its degree is limited by interconnect bandwidth.',
    lesson: 'moe/expert-parallelism',
  },
};