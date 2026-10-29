// The single source of truth for the course structure.
// Parts group modules; modules hold lessons in teaching order.
// Every lesson here has an MDX file at src/content/lessons/<module.slug>/<lesson.slug>.mdx

export type LessonKind = 'lesson' | 'lab' | 'project';

export interface LessonMeta {
  slug: string;
  title: string;
  kind: LessonKind;
  minutes: number;
}

export interface ModuleMeta {
  num: string; // "05"
  slug: string;
  title: string;
  tagline: string;
  part: number;
  track: 'core' | 'runtime' | 'scale' | 'research' | 'build';
  lessons: LessonMeta[];
}

export interface PartMeta {
  num: number;
  label?: string; // shown instead of num when set ("B")
  title: string;
  question: string; // the question this part answers
}

// Part 1 is the trunk everyone takes. After it the course forks: Part B builds products on top of
// an LLM, Parts 2-8 make serving fast. The parts array order is the display order.
export const parts: PartMeta[] = [
  { num: 1, title: 'Foundations', question: 'How does an LLM work, how did we get here, and how do I run one?' },
  { num: 9, label: 'B', title: 'Build with it', question: 'How do I build reliable products on top of an LLM?' },
  { num: 2, title: 'How inference works', question: 'What actually happens when a model produces a token?' },
  { num: 3, title: 'Serving engines', question: 'How do we serve many people at once from one GPU?' },
  { num: 4, title: 'Single-GPU optimization', question: 'How do we squeeze more out of the GPU we have?' },
  { num: 5, title: 'Runtime & kernels', question: 'What is happening inside the kernels, and how do we make them faster?' },
  { num: 6, title: 'Scaling out', question: 'What changes when one GPU is not enough?' },
  { num: 7, title: 'Production', question: 'How do we run this for real users, reliably and cheaply?' },
  { num: 8, title: 'Mastery', question: 'How do I build, extend and push the frontier myself?' },
];

const L = (slug: string, title: string, minutes = 15, kind: LessonKind = 'lesson'): LessonMeta => ({ slug, title, minutes, kind });
const LAB = (slug: string, title: string, minutes = 45): LessonMeta => ({ slug, title, minutes, kind: 'lab' });
const PROJECT = (slug: string, title: string, minutes = 90): LessonMeta => ({ slug, title, minutes, kind: 'project' });

export const modules: ModuleMeta[] = [
  // ───────────── Part 1 · Foundations
  {
    num: '00', slug: 'big-picture', title: 'The Big Picture', part: 1, track: 'core',
    tagline: 'Follow one message from your keyboard to a GPU and back.',
    lessons: [
      L('press-enter', 'What happens when you press Enter', 15),
      L('the-stack', 'What an inference engineer actually does', 12),
      L('four-numbers', 'The four numbers you will live by', 15),
      LAB('lab-hello-gpu', 'Lab: Say hello to a real GPU', 30),
    ],
  },
  {
    num: '01', slug: 'numbers-and-math', title: 'Numbers & Math from Scratch', part: 1, track: 'core',
    tagline: 'Bytes, shapes, matrix multiplies and softmax: the only math inference needs.',
    lessons: [
      L('numbers-in-memory', 'How a number lives in memory', 20),
      L('vectors-matrices-shapes', 'Vectors, matrices & shapes', 18),
      L('matmul-and-flops', 'Matrix multiply, step by step, and counting FLOPs', 20),
      L('dot-product-softmax', 'Dot products & softmax', 18),
      LAB('lab-tensors', 'Lab: Tensors, dtypes & your first timing', 40),
    ],
  },
  {
    num: '02', slug: 'gpu-first-principles', title: 'The GPU from First Principles', part: 1, track: 'core',
    tagline: 'Why GPUs, what is inside one, and the two speed limits that rule everything.',
    lessons: [
      L('cpu-vs-gpu', 'CPU vs GPU: a few geniuses or ten thousand workers', 15),
      L('inside-a-gpu', 'Inside a GPU: SMs, tensor cores & memory', 20),
      L('two-speed-limits', 'The two speed limits: compute and bandwidth', 22),
      L('software-stack', 'The software stack: drivers, CUDA, PyTorch & remote machines', 18),
      L('other-accelerators', 'Beyond NVIDIA: TPUs, Trainium, Groq & other accelerators', 18),
      LAB('lab-measure-gpu', 'Lab: Measure your GPU’s speed limits', 45),
    ],
  },
  {
    num: '03', slug: 'transformers', title: 'From Neurons to Transformers', part: 1, track: 'core',
    tagline: 'Build the transformer in your head, one block at a time.',
    lessons: [
      L('neural-networks', 'Neurons, layers & weights', 18),
      L('how-models-learn', 'How a model learns: loss & gradient descent', 22),
      L('tokens', 'Tokens & tokenizers', 15),
      L('embeddings-positions', 'Embeddings & positions', 16),
      L('attention', 'Attention from scratch: Q, K and V', 25),
      L('multi-head-gqa', 'Multi-head attention, MQA & GQA', 18),
      L('mlp-norm-residual', 'The MLP, RMSNorm & the residual stream', 18),
      L('forward-pass', 'The whole forward pass: one token’s journey', 22),
      L('sampling', 'Logits to tokens: sampling', 18),
      LAB('lab-trace-a-token', 'Visual lab: Trace one token end to end', 45),
    ],
  },
  {
    num: '3A', slug: 'pretraining', title: 'One Machine, Two Jobs', part: 1, track: 'core',
    tagline: 'The same transformer, two training games: fill in the blank, or guess the next word.',
    lessons: [
      L('self-supervision', 'Free labels: learning from raw text', 18),
      L('loss-perplexity', 'Cross-entropy, perplexity & bits per byte', 20),
      L('bert', 'Reading: BERT and the masked-word game', 22),
      L('gpt2', 'Writing: GPT-2 and the next-word game', 22),
      L('encoder-vs-decoder', 'Why decoders won', 16),
      LAB('lab-two-jobs', 'Lab: Fill the blank vs finish the sentence', 45),
    ],
  },
  {
    num: '3B', slug: 'scaling', title: 'What Scale Buys', part: 1, track: 'core',
    tagline: 'Make the model bigger and new abilities appear. But how big, on how much data?',
    lessons: [
      L('training-compute', 'Counting training compute: C ≈ 6ND', 20),
      L('gpt3-in-context', 'GPT-3: learning from the prompt', 22),
      L('scaling-laws', 'Scaling laws: loss falls as a power law', 20),
      L('chinchilla', 'Chinchilla: the compute-optimal model', 22),
      L('beyond-chinchilla', 'Over-training for cheap inference & the limits of scale', 18),
      LAB('lab-compute-budget', 'Lab: Spend a compute budget', 40),
    ],
  },
  {
    num: '3C', slug: 'alignment', title: 'From Autocomplete to Assistant', part: 1, track: 'core',
    tagline: 'A base model only continues text. Here is how it learns to answer you.',
    lessons: [
      L('base-vs-assistant', 'Why a base model does not answer you', 16),
      L('sft', 'Supervised finetuning: teaching by example', 18),
      L('reward-models', 'Preference data & reward models', 20),
      L('rlhf-instructgpt', 'RLHF: the InstructGPT recipe', 24),
      L('dpo', 'DPO: preference tuning without RL', 18),
      L('hallucination', 'Why models hallucinate', 18),
      LAB('lab-preferences', 'Lab: Train a tiny reward model', 45),
    ],
  },
  {
    num: '04', slug: 'first-llm', title: 'Run Your First LLM', part: 1, track: 'core',
    tagline: 'Download a real model, make it talk, watch the GPU, put it behind HTTP.',
    lessons: [
      L('open-weights-llama', 'Open weights: what LLaMA changed', 18),
      L('choose-a-model', 'Choosing an open model', 15),
      L('load-and-generate', 'Load, tokenize, generate & stream', 22),
      L('watch-the-gpu', 'Watching the GPU work', 15),
      L('serve-over-http', 'Serving a model over HTTP', 20),
      LAB('lab-first-server', 'Lab: Your first GPU inference server', 60),
    ],
  },

  // ───────────── Part B · Build with it
  {
    num: 'B1', slug: 'prompting', title: 'Prompt Engineering', part: 9, track: 'build',
    tagline: 'Steer a model without touching its weights, and measure whether it worked.',
    lessons: [
      L('chat-templates', 'Chat templates & system prompts', 16),
      L('in-context', 'Zero-shot, few-shot & in-context learning in practice', 18),
      L('writing-prompts', 'Writing prompts that work: instructions, context & decomposition', 20),
      L('chain-of-thought', 'Chain of thought & test-time compute', 20),
      L('context-position', 'Long prompts: lost in the middle', 16),
      L('prompt-ops', 'Versioning, testing & optimizing prompts', 16),
      LAB('lab-prompting', 'Lab: Prompt, measure, iterate', 40),
    ],
  },
  {
    num: 'B2', slug: 'evaluation', title: 'Evaluating AI Systems', part: 9, track: 'build',
    tagline: 'If you cannot measure it, you cannot ship it. How to score open-ended output.',
    lessons: [
      L('why-eval-is-hard', 'Why evaluating LLMs is hard', 16),
      L('exact-evaluation', 'Exact evaluation: pass@k, exact match & similarity', 20),
      L('ai-as-judge', 'AI as a judge, and its biases', 20),
      L('comparative-eval', 'Comparative evaluation: Elo & arenas', 18),
      L('benchmarks-contamination', 'Public benchmarks, saturation & contamination', 18),
      L('eval-pipeline', 'Designing your evaluation pipeline', 22),
      L('model-selection', 'Choosing a model: the selection funnel', 18),
      LAB('lab-eval', 'Lab: Build an eval harness with an AI judge', 50),
    ],
  },
  {
    num: 'B3', slug: 'rag', title: 'Retrieval & RAG', part: 9, track: 'build',
    tagline: 'Give the model the facts it was never trained on, at the moment it needs them.',
    lessons: [
      L('why-rag', 'Why retrieval: finetuning for form, RAG for facts', 16),
      L('embeddings-search', 'Embeddings & vector search', 20),
      L('bm25-hybrid', 'Keyword search, BM25 & hybrid retrieval', 18),
      L('ann-indexes', 'Approximate nearest neighbours: HNSW & IVF', 20),
      L('chunking-reranking', 'Chunking, query rewriting & reranking', 20),
      L('evaluating-retrieval', 'Evaluating retrieval: precision, recall & NDCG', 18),
      LAB('lab-rag', 'Lab: Build and measure a RAG pipeline', 60),
    ],
  },
  {
    num: 'B4', slug: 'agents', title: 'Agents', part: 9, track: 'build',
    tagline: 'From a model that answers to a model that acts: tools, plans and memory.',
    lessons: [
      L('what-is-an-agent', 'What makes an agent', 16),
      L('tools', 'Tools: read, compute & write actions', 18),
      L('planning', 'Planning: ReAct, reflection & plan-then-execute', 22),
      L('memory', 'Agent memory', 18),
      L('agent-failures', 'Agent failure modes & evaluation', 18),
      LAB('lab-agent', 'Lab: A tool-using agent from scratch', 60),
    ],
  },
  {
    num: 'B5', slug: 'finetuning', title: 'Finetuning', part: 9, track: 'build',
    tagline: 'Change the weights, cheaply: when it is worth it, and how LoRA makes it fit.',
    lessons: [
      L('when-to-finetune', 'When to finetune (and when not to)', 16),
      L('training-memory', 'Training memory: weights, gradients & optimizer states', 20),
      L('lora', 'LoRA: low-rank adaptation', 24),
      L('qlora', 'QLoRA & finetuning on one GPU', 18),
      L('finetuning-tactics', 'Finetuning tactics: learning rate, loss masking & epochs', 18),
      L('model-merging', 'Model merging & task arithmetic', 16),
      LAB('lab-lora', 'Lab: Finetune a small model with LoRA', 60),
    ],
  },
  {
    num: 'B6', slug: 'datasets', title: 'Dataset Engineering', part: 9, track: 'build',
    tagline: 'The model is only as good as its data: curate, synthesize, verify, clean.',
    lessons: [
      L('quality-coverage-quantity', 'Quality, coverage & quantity', 18),
      L('instruction-data', 'Formatting instruction data', 16),
      L('synthetic-data', 'Synthetic data & distillation', 20),
      L('verifying-data', 'Verifying synthetic data & model collapse', 18),
      L('dedup-cleaning', 'Deduplication & cleaning at scale', 18),
      LAB('lab-data', 'Lab: Build and clean a finetuning set', 45),
    ],
  },
  {
    num: 'B7', slug: 'ai-apps', title: 'Production AI Applications', part: 9, track: 'build',
    tagline: 'Wrap the model in a system: guardrails, routing, caching, observability and feedback.',
    lessons: [
      L('app-architecture', 'From one model call to a full AI system', 18),
      L('guardrails', 'Guardrails: input & output', 18),
      L('prompt-injection', 'Prompt injection & jailbreaks', 20),
      L('gateway-routing', 'Model gateways, routers & caches', 18),
      L('app-observability', 'Observability for AI apps', 18),
      L('user-feedback', 'User feedback & the data flywheel', 18),
      PROJECT('project-ai-app', 'Project: Ship an AI app end to end', 120),
    ],
  },

  // ───────────── Part 2 · How inference works
  {
    num: '05', slug: 'kv-cache', title: 'Prefill, Decode & the KV Cache', part: 2, track: 'core',
    tagline: 'The single most important idea in LLM inference.',
    lessons: [
      L('recompute-waste', 'Why naive generation wastes almost everything', 16),
      L('kv-cache', 'The KV cache: remember, don’t recompute', 20),
      L('prefill-vs-decode', 'Prefill vs decode: two different workloads', 22),
      L('memory-math', 'Memory math: weights, KV cache & concurrency', 25),
      L('build-kv-cache', 'Build a KV cache yourself', 25),
      LAB('lab-kv-cache', 'Lab: Cached vs uncached & will it fit?', 60),
    ],
  },
  {
    num: '06', slug: 'benchmarking', title: 'Measure Like a Pro', part: 2, track: 'core',
    tagline: 'If you cannot measure it correctly, you cannot optimize it.',
    lessons: [
      L('latency-metrics', 'TTFT, TPOT, ITL & end-to-end latency', 18),
      L('throughput-concurrency', 'Throughput, concurrency & Little’s law', 20),
      L('workloads-reproducibility', 'Workloads, warm-up & reproducible runs', 16),
      L('percentiles', 'Percentiles: why averages lie', 16),
      L('slos-goodput', 'SLOs & goodput', 15),
      L('break-one-gpu', 'Break one GPU: sweeps & the operating envelope', 22),
      PROJECT('project-benchmark-harness', 'Project: Build your benchmark harness', 90),
    ],
  },

  // ───────────── Part 3 · Serving engines
  {
    num: '07', slug: 'batching-scheduling', title: 'Batching & Scheduling', part: 3, track: 'core',
    tagline: 'How one GPU serves hundreds of conversations at once.',
    lessons: [
      L('naive-and-static-batching', 'Why naive serving breaks & static batching', 18),
      L('continuous-batching', 'Continuous batching', 22),
      L('scheduler-knobs', 'The scheduler’s knobs: token budgets & limits', 18),
      L('queues-preemption', 'Queues, fairness, cancellation & preemption', 18),
      LAB('lab-scheduler-sim', 'Lab: Build a scheduler simulator', 60),
    ],
  },
  {
    num: '08', slug: 'paged-attention', title: 'PagedAttention & KV Memory', part: 3, track: 'core',
    tagline: 'Borrowing an idea from operating systems to fit 3× more requests.',
    lessons: [
      L('fragmentation', 'Why contiguous KV memory wastes space', 16),
      L('blocks-and-tables', 'Blocks, block tables & PagedAttention', 22),
      L('sharing-cow', 'Sharing blocks: prefixes & copy-on-write', 18),
      L('kv-pressure', 'Running out: preemption, swap & recompute', 16),
      LAB('lab-paged-allocator', 'Visual lab: Build a paged KV allocator', 50),
    ],
  },
  {
    num: '09', slug: 'vllm', title: 'A Real Engine: vLLM', part: 3, track: 'core',
    tagline: 'Everything you learned, running inside a production engine.',
    lessons: [
      L('start-vllm', 'Start vLLM & talk to it', 18),
      L('anatomy-of-a-request', 'Anatomy of a request inside vLLM', 22),
      L('tune-one-gpu', 'Metrics & tuning one GPU', 20),
      L('optimization-plan', 'An optimization plan, end to end', 22),
      PROJECT('project-pytorch-vs-vllm', 'Project: PyTorch vs vLLM, then optimize one GPU', 90),
    ],
  },

  // ───────────── Part 4 · Single-GPU optimization
  {
    num: '10', slug: 'prefill-prefix', title: 'Prefill & Prefix Optimization', part: 4, track: 'core',
    tagline: 'Stop long prompts from blocking everyone, and never compute the same prefix twice.',
    lessons: [
      L('chunked-prefill', 'Long-prompt interference & chunked prefill', 20),
      L('prefix-caching', 'Prefix caching', 18),
      L('radix-attention', 'RadixAttention, eviction & cache-friendly apps', 20),
      L('rag-cag', 'RAG, CAG & the serving bill', 20),
      LAB('lab-mixed-workload', 'Lab: Optimize a mixed long-prompt workload', 60),
    ],
  },
  {
    num: '11', slug: 'quantization', title: 'Quantization', part: 4, track: 'core',
    tagline: 'Fewer bits, more speed. Know exactly what you trade away.',
    lessons: [
      L('why-precision', 'What precision buys you', 16),
      L('how-quantization-works', 'How quantization works: scales & granularity', 22),
      L('quant-schemes', 'W4A16, W8A8, FP8: choosing a scheme', 18),
      L('awq-gptq-smoothquant', 'AWQ, GPTQ & SmoothQuant', 22),
      L('quantized-kernels-kv', 'Quantized GEMMs & KV cache quantization', 18),
      L('measure-quality', 'Measuring quality regression', 16),
      L('distillation-pruning', 'Beyond quantization: distillation & pruning', 18),
      PROJECT('project-pareto', 'Project: The quality/speed/memory frontier', 90),
    ],
  },
  {
    num: '12', slug: 'profiling', title: 'Profiling & GPU Performance', part: 4, track: 'core',
    tagline: 'Stop guessing. Find the real bottleneck.',
    lessons: [
      L('execution-model', 'How a GPU runs your code: threads, warps & blocks', 22),
      L('memory-hierarchy', 'The memory hierarchy & coalescing', 18),
      L('roofline', 'The roofline model, in depth', 22),
      L('profiling-tools', 'A profiling workflow: PyTorch profiler, Nsight Systems & Compute', 22),
      L('profile-prefill-decode', 'Profiling prefill vs decode', 18),
      LAB('lab-bottleneck-report', 'Lab: Write a bottleneck report', 60),
    ],
  },

  // ───────────── Part 5 · Runtime & kernels
  {
    num: '13', slug: 'triton', title: 'Writing Kernels with Triton', part: 5, track: 'runtime',
    tagline: 'Write your own GPU kernels in Python.',
    lessons: [
      L('triton-model', 'The Triton programming model: vector add', 22),
      L('reductions-softmax', 'Reductions & a fused softmax', 22),
      L('fused-rmsnorm-rope', 'Fused RMSNorm & RoPE', 22),
      L('benchmark-kernels', 'Benchmarking kernels against PyTorch', 16),
      PROJECT('challenge-kernel', 'Challenge: Optimize a real inference kernel', 90),
    ],
  },
  {
    num: '14', slug: 'attention-optimization', title: 'Attention Optimization', part: 5, track: 'runtime',
    tagline: 'FlashAttention, built up from a single observation.',
    lessons: [
      L('online-softmax', 'Attention’s memory problem & online softmax', 22),
      L('flash-attention', 'FlashAttention, step by step', 25),
      L('decode-attention', 'Decode attention & attention backends', 20),
      LAB('lab-attention-backends', 'Lab: Compare attention implementations', 60),
    ],
  },
  {
    num: '15', slug: 'compilation', title: 'Compilation & CUDA Graphs', part: 5, track: 'runtime',
    tagline: 'When the GPU waits on Python, and how to stop it.',
    lessons: [
      L('launch-overhead', 'Eager mode & launch overhead', 16),
      L('torch-compile', 'torch.compile, fusion & graph breaks', 20),
      L('cuda-graphs', 'CUDA graphs: capture & replay', 18),
      LAB('lab-compile-decode', 'Lab: Compile & capture a decode loop', 50),
    ],
  },
  {
    num: '16', slug: 'speculative-decoding', title: 'Speculative Decoding', part: 5, track: 'runtime',
    tagline: 'Guess several tokens cheaply, verify them all at once.',
    lessons: [
      L('draft-and-verify', 'Draft & verify: the core trick', 20),
      L('spec-math', 'The math: acceptance, speedup & exactness', 22),
      L('spec-methods', 'N-gram, Medusa, EAGLE & MTP', 20),
      L('when-spec-wins', 'When speculation wins, and when it loses', 16),
      LAB('lab-speculative', 'Lab: Map where speculation wins', 60),
    ],
  },
  {
    num: '17', slug: 'engine-landscape', title: 'The Engine Landscape', part: 5, track: 'runtime',
    tagline: 'vLLM, SGLang, TensorRT-LLM, llama.cpp: how to compare them fairly.',
    lessons: [
      L('fair-comparison', 'How to compare engines fairly', 16),
      L('engines-compared', 'vLLM, SGLang, TensorRT-LLM & llama.cpp', 22),
      L('multi-model-triton', 'Multi-model serving & Triton Inference Server', 22),
      PROJECT('project-engine-matrix', 'Project: Build an engine selection matrix', 60),
    ],
  },

  // ───────────── Part 6 · Scaling out
  {
    num: '18', slug: 'multi-gpu', title: 'Multi-GPU Parallelism', part: 6, track: 'scale',
    tagline: 'Split a model across GPUs without drowning in communication.',
    lessons: [
      L('interconnects-collectives', 'Interconnects & collectives', 20),
      L('tensor-parallelism', 'Tensor parallelism', 22),
      L('pipeline-parallelism', 'Pipeline parallelism', 18),
      L('data-context-parallelism', 'Data parallelism & context parallelism', 18),
      LAB('lab-parallelism', 'Lab: Choose a parallelism strategy', 50),
    ],
  },
  {
    num: '19', slug: 'moe', title: 'Mixture-of-Experts Inference', part: 6, track: 'scale',
    tagline: 'Huge models that only use a slice of themselves per token.',
    lessons: [
      L('moe-routing', 'How MoE works: routing tokens to experts', 20),
      L('expert-parallelism', 'Expert parallelism, all-to-all & hot experts', 22),
      LAB('lab-moe', 'Lab: Simulate MoE load balance', 45),
    ],
  },
  {
    num: '20', slug: 'long-context', title: 'Long Context & Tiered KV', part: 6, track: 'scale',
    tagline: 'When the KV cache is bigger than the GPU.',
    lessons: [
      L('long-context-scaling', 'What breaks at long context', 18),
      L('tiered-kv', 'Tiered KV: CPU, SSD & remote caches', 20),
      LAB('lab-long-context', 'Lab: Push context beyond GPU memory', 45),
    ],
  },
  {
    num: '21', slug: 'disaggregation', title: 'Prefill/Decode Disaggregation', part: 6, track: 'scale',
    tagline: 'Give each phase its own GPUs.',
    lessons: [
      L('why-disaggregate', 'Why separate prefill and decode', 18),
      L('kv-transfer-sizing', 'KV transfer, pool sizing & routing', 20),
      PROJECT('project-disaggregation', 'Project: Model a disaggregated system', 60),
    ],
  },

  // ───────────── Part 7 · Production
  {
    num: '22', slug: 'production-serving', title: 'Production Serving', part: 7, track: 'core',
    tagline: 'Package it, start it fast, ship new versions safely.',
    lessons: [
      L('packaging-cold-starts', 'Packaging, weights & cold starts', 18),
      L('service-lifecycle', 'Warm-up, health checks & graceful shutdown', 16),
      L('safe-releases', 'Versioning, canaries, shadow traffic & rollbacks', 16),
      PROJECT('project-productionize', 'Project: Productionize your server', 90),
    ],
  },
  {
    num: '23', slug: 'routing-autoscaling', title: 'Routing, Autoscaling & Orchestration', part: 7, track: 'core',
    tagline: 'Many replicas, one smart front door.',
    lessons: [
      L('load-balancing', 'Load balancing & queue-aware routing', 18),
      L('cache-aware-routing', 'Prefix- & cache-aware routing', 16),
      L('semantic-caching', 'Semantic caching & semantic routing', 18),
      L('autoscaling', 'Autoscaling on the right signals', 18),
      L('orchestration', 'Kubernetes, KServe, Ray Serve & llm-d', 18),
      LAB('lab-hold-the-slo', 'Lab: Hold an SLO under changing traffic', 50),
    ],
  },
  {
    num: '24', slug: 'observability-reliability', title: 'Observability, Reliability & Security', part: 7, track: 'core',
    tagline: 'See everything, survive overload, recover from failure.',
    lessons: [
      L('metrics-dashboards', 'Metrics & dashboards that matter', 18),
      L('tracing', 'Tracing a request across the system', 14),
      L('overload', 'Admission control, backpressure & load shedding', 18),
      L('failures', 'Timeouts, retries, recovery & chaos testing', 18),
      L('security', 'Securing an inference service', 16),
      PROJECT('project-break-recover', 'Project: Break and recover the service', 60),
    ],
  },
  {
    num: '25', slug: 'economics', title: 'Inference Economics', part: 7, track: 'core',
    tagline: 'Every design decision is a cost decision.',
    lessons: [
      L('cost-per-token', 'Cost per million tokens, from scratch', 20),
      L('goodput-per-dollar', 'Goodput per dollar & choosing GPUs', 18),
      L('capacity-planning', 'Capacity planning & headroom', 18),
      L('build-vs-buy', 'Managed APIs, cloud platforms & build vs buy', 20),
      PROJECT('project-cheapest-system', 'Project: The cheapest system that meets the SLO', 60),
    ],
  },

  // ───────────── Part 8 · Mastery
  {
    num: '26', slug: 'specialized-workloads', title: 'Specialized Workloads', part: 8, track: 'research',
    tagline: 'Structured output, LoRA, vision, agents, reasoning, RL and diffusion.',
    lessons: [
      L('structured-output', 'Structured output & tool calling', 18),
      L('multi-lora', 'Multi-LoRA serving', 16),
      L('multimodal', 'Vision-language & multimodal inference', 16),
      L('edge-inference', 'On-device & edge inference', 18),
      L('agents-reasoning', 'Agents, multi-turn & reasoning models', 18),
      L('batch-rl-rollouts', 'Batch inference & RL rollouts', 16),
      L('diffusion', 'Diffusion inference', 18),
      LAB('lab-specialized', 'Lab: Specialized workloads, measured', 50),
    ],
  },
  {
    num: '27', slug: 'mini-engine', title: 'Build a Mini Inference Engine', part: 8, track: 'research',
    tagline: 'Put everything together: write your own vLLM, small.',
    lessons: [
      L('read-vllm', 'Reading a real engine’s codebase', 20),
      L('runner-and-kv', 'Step 1: Model runner & KV cache', 30),
      L('continuous-batching-engine', 'Step 2: Continuous batching scheduler', 30),
      L('paged-prefix-engine', 'Step 3: Paged KV & prefix caching', 30),
      L('api-metrics-engine', 'Step 4: Streaming API & metrics', 25),
      PROJECT('project-mini-engine', 'Project: Benchmark your engine against vLLM', 120),
    ],
  },
  {
    num: '28', slug: 'research-capstone', title: 'Research & Capstone', part: 8, track: 'research',
    tagline: 'Read papers, reproduce results, design a complete system, get hired.',
    lessons: [
      L('read-papers', 'How to read an inference systems paper', 18),
      L('reproduce-research', 'Reproducing a result', 20),
      PROJECT('capstone', 'Capstone: Design, build & defend a system', 240),
      L('interview-prep', 'Interview prep & where to go next', 25),
    ],
  },
];

export interface FlatLesson extends LessonMeta {
  module: ModuleMeta;
  index: number; // global index across the course
  href: string;
  code: string; // "05.2"
}

export const flatLessons: FlatLesson[] = (() => {
  const out: FlatLesson[] = [];
  let i = 0;
  for (const m of modules) {
    m.lessons.forEach((l, j) => {
      out.push({
        ...l,
        module: m,
        index: i++,
        href: `/learn/${m.slug}/${l.slug}/`,
        code: l.kind === 'lesson' ? `${m.num}.${j + 1}` : `${m.num}.${l.kind === 'lab' ? 'L' : 'P'}`,
      });
    });
  }
  return out;
})();

export const totalMinutes = flatLessons.reduce((s, l) => s + l.minutes, 0);

export function findLesson(moduleSlug: string, lessonSlug: string) {
  return flatLessons.find((l) => l.module.slug === moduleSlug && l.slug === lessonSlug);
}

export function lessonKey(l: { module: { slug: string }; slug: string }) {
  return `${l.module.slug}/${l.slug}`;
}
