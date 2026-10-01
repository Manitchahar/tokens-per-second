import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  pcie: {
    term: 'PCIe',
    def: 'The peripheral bus that connects a GPU to the host (and through the host bridge, to other GPUs). About 32 GB/s on a 4.0 x16 slot, 64 GB/s on 5.0 x16. Any card that is not NVLink-attached talks to its neighbours this way.',
    lesson: 'multi-gpu/interconnects-collectives',
  },
  nvlink: {
    term: 'NVLink',
    def: 'A high-speed chip-to-chip link that bypasses PCIe. Inside an 8-GPU SXM node every GPU has a full mesh of them: 450 GB/s each way on H100, 900 GB/s on B200. This is what makes tensor parallelism practical.',
    lesson: 'multi-gpu/interconnects-collectives',
  },
  'infini-band': {
    term: 'InfiniBand (IB)',
    def: 'The standard high-speed fabric between machines — about 50 GB/s per port for NDR. Roughly a third of NVLink, which is why cross-node parallelism moves whole layers rather than individual vectors.',
    lesson: 'multi-gpu/interconnects-collectives',
  },
  'all-reduce': {
    term: 'All-reduce',
    def: 'A collective that sums a tensor across all GPUs and leaves the identical result on every one. Built as reduce-scatter followed by all-gather, so a ring implementation moves 2(P-1)/P of the payload per GPU. The workhorse of tensor parallelism.',
    lesson: 'multi-gpu/interconnects-collectives',
  },
  'all-gather': {
    term: 'All-gather',
    def: 'A collective where each GPU holds a different shard of a tensor and afterwards holds the whole of it. Moves (P-1)/P of the payload per GPU in a ring implementation.',
    lesson: 'multi-gpu/interconnects-collectives',
  },
  'reduce-scatter': {
    term: 'Reduce-scatter',
    def: 'A collective that sums a tensor across GPUs and then leaves each GPU holding only one shard of the result. It is the first half of an all-reduce, and costs (P-1)/P of the payload per GPU.',
    lesson: 'multi-gpu/interconnects-collectives',
  },
  'all-to-all': {
    term: 'All-to-all',
    def: 'A collective where every GPU sends a distinct slice of its data to every other GPU, so each ends up holding a differently-assembled result. A permutation rather than a reduction. This is how MoE tokens reach their experts (Module 19).',
    lesson: 'multi-gpu/interconnects-collectives',
  },
  broadcast: {
    term: 'Broadcast',
    def: 'A collective that copies one GPU’s value to every other GPU. Used to initialise replicated state and to hand a new tensor to the whole group.',
    lesson: 'multi-gpu/interconnects-collectives',
  },
  'collective-efficiency': {
    term: 'Collective efficiency',
    def: 'The fraction of a link’s quoted peak bandwidth a real collective actually achieves, because ring and tree algorithms do not perfectly saturate the wire. Assume 0.7–0.9 on a good fabric; anything lower is worth investigating.',
    lesson: 'multi-gpu/interconnects-collectives',
  },
  'tensor-parallelism': {
    term: 'Tensor parallelism (TP)',
    def: 'Splitting the matrices inside a single layer across GPUs so each one computes a slice of the same token’s forward pass. Needs two all-reduces per layer, so it only pays inside a fast interconnect island (the NVLink node).',
    lesson: 'multi-gpu/tensor-parallelism',
  },
  'row-parallel': {
    term: 'Row-parallel linear',
    def: 'The second half of a Megatron-style MLP: the output projection split along its input dimension. Each GPU computes a partial sum, and one all-reduce turns those partials into the real output. Follows a column-parallel all-gather, so the pair costs one all-reduce.',
    lesson: 'multi-gpu/tensor-parallelism',
  },
  'column-parallel': {
    term: 'Column-parallel linear',
    def: 'A linear layer split along its output dimension: each GPU computes a slice of the output columns. No communication is needed on the way out, because the shards are the result.',
    lesson: 'multi-gpu/tensor-parallelism',
  },
  'tp-degree': {
    term: 'TP degree (t)',
    def: 'How many GPUs one layer is split across. It must divide the number of attention heads, the number of KV heads, and the MLP intermediate width — so GQA’s small KV head count is often what caps it.',
    lesson: 'multi-gpu/tensor-parallelism',
  },
  'pipeline-parallelism': {
    term: 'Pipeline parallelism (PP)',
    def: 'Giving each GPU a contiguous group of transformer layers and passing activations from one to the next. Only one activation tensor crosses each boundary, so communication is 10–100x cheaper than tensor parallelism and it can cross the node boundary.',
    lesson: 'multi-gpu/pipeline-parallelism',
  },
  bubble: {
    term: 'Pipeline bubble',
    def: 'The fraction of a pipeline’s time wasted with most stages idle. In inference (forward only) it is (P-1)/(M+P-1) for P stages and M micro-batches; the cure is more micro-batches, not more stages.',
    lesson: 'multi-gpu/pipeline-parallelism',
  },
  'pipeline-bubble-fraction': {
    term: 'Bubble fraction',
    def: 'The share of pipeline wall-clock time where fewer than all stages are working: (P-1)/(M+P-1). At P=4 and M=4 it is 3/7 ≈ 43% — a fourth of your GPUs doing nothing. Inference has no backward pass to fill it with.',
    lesson: 'multi-gpu/pipeline-parallelism',
  },
  'pipeline-microbatch': {
    term: 'Pipeline micro-batch',
    def: 'A small slice of the request batch handed to one stage at a time, so several can be in flight and keep the pipeline full. More micro-batches shrink the bubble; each one costs an extra activation transfer.',
    lesson: 'multi-gpu/pipeline-parallelism',
  },
  'pipeline-parallel-batch': {
    term: 'Pipeline-parallel batch (num_stages)',
    def: 'How many contiguous groups of layers the model is cut into — how many GPUs each request walks through. Compare with the batch size B: P stages against B tokens in flight is what the bubble fraction is made of.',
    lesson: 'multi-gpu/pipeline-parallelism',
  },
  'mega-batch': {
    term: 'Mega-batch',
    def: 'The whole set of micro-batches a pipeline processes together. Pipelining splits the request batch into micro-batches but never changes the total work: mega-batch size is the same as the batch size you would have used on one GPU.',
    lesson: 'multi-gpu/pipeline-parallelism',
  },
  'replica-parallelism': {
    term: 'Replica parallelism',
    def: 'Running several complete copies of the model and splitting incoming traffic between them. No communication at all — so it is the fastest way to add throughput and the only parallelism that helps a single request’s latency not at all.',
    lesson: 'multi-gpu/data-context-parallelism',
  },
  'data-parallelism': {
    term: 'Data parallelism (DP)',
    def: 'Different GPUs running different requests from the same weights. Free at inference (weights are read-only), so it only appears where something must be shared: attention heads within a group, or a KV cache that lives on one GPU.',
    lesson: 'multi-gpu/data-context-parallelism',
  },
  'sequence-parallelism': {
    term: 'Sequence parallelism',
    def: 'Splitting the token dimension of the activations (and of the LayerNorm / dropout regions) across a tensor-parallel group, so those memory-bound operations are not repeated on every rank. Only a small, cheap all-gather or all-reduce.',
    lesson: 'multi-gpu/data-context-parallelism',
  },
  'context-parallelism': {
    term: 'Context parallelism (CP)',
    def: 'Splitting one long sequence across GPUs so each holds a slice of the KV cache, then exchanging K/V through the fabric so every GPU can attend over the whole context. Cuts per-GPU memory by P — and pays an all-to-all of nearly the whole cache.',
    lesson: 'multi-gpu/data-context-parallelism',
  },
  'expert-parallelism': {
    term: 'Expert parallelism (EP)',
    def: 'Spreading a model’s experts across GPUs and routing tokens to the GPU that owns each expert with an all-to-all. The multi-GPU sibling of MoE, coming in Module 19.',
    lesson: 'multi-gpu/data-context-parallelism',
  },
  'island': {
    term: 'The island',
    def: 'The largest group of GPUs reachable over one fast fabric (an NVLink node, usually 8 GPUs). Fine-grained sharding is confined to the island; GPUs in different islands get separate replicas. Decide this before choosing any parallel strategy.',
    lesson: 'multi-gpu/data-context-parallelism',
  },
  'communication-overhead': {
    term: 'Communication overhead fraction',
    def: 'The share of a step’s wall-clock time spent in collectives rather than math. Above roughly 20% you are losing more than half your parallel efficiency, and adding GPUs makes things worse rather than better.',
    lesson: 'multi-gpu/tensor-parallelism',
  },
  'collective-cost': {
    term: 'Message cost',
    def: 'latency + bytes / (bandwidth x efficiency). The first term is paid per message no matter how small; the second scales with payload. Count the messages, not just the bytes.',
    lesson: 'multi-gpu/interconnects-collectives',
  },
};