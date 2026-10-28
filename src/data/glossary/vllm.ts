import type { GlossaryEntry } from '../glossary';

// Module 09 · A Real Engine: vLLM
// Terms that only show up once you start reading a real serving engine's code
// and its command-line flags. Everything here is taught in this module.
export const terms: Record<string, GlossaryEntry> = {
  'openai-compatible-api': {
    term: 'OpenAI-compatible API',
    def: 'An HTTP interface with the same paths and JSON bodies as OpenAI’s, so existing SDKs and clients can point at a local server instead. vLLM serves /v1/chat/completions, /v1/completions and /v1/models.',
    lesson: 'vllm/start-vllm',
  },
  'api-server': {
    term: 'API server',
    def: 'The HTTP front end of vLLM: it checks the request, runs the tokenizer, hands the work to the engine core, and streams the answer back. It does almost no math itself.',
    lesson: 'vllm/start-vllm',
  },
  'engine-core': {
    term: 'Engine core',
    def: 'The process that owns the GPU inside vLLM: the scheduler, the KV-cache manager and the model runner. The API server hands it requests; it decides what runs each iteration.',
    lesson: 'vllm/anatomy-of-a-request',
  },
  'model-runner': {
    term: 'Model runner',
    def: 'The part of the engine core that actually launches GPU kernels for the batched tokens it was given, runs the sampler, and writes the new keys and values into the KV cache.',
    lesson: 'vllm/anatomy-of-a-request',
  },
  'safetensors': {
    term: 'Safetensors',
    def: 'A simple tensor file format (weights plus their names, no pickled code). vLLM reads model weights from `.safetensors` files in the Hugging Face cache.',
    lesson: 'vllm/start-vllm',
  },
  'kv-cache-pool': {
    term: 'KV cache pool',
    def: 'All the GPU memory vLLM sets aside for the KV cache, carved into equal blocks at startup before any request arrives. Its size is the real capacity of the server.',
    lesson: 'vllm/start-vllm',
  },
  'kv-block': {
    term: 'KV block',
    def: 'One fixed-size chunk of the KV cache pool: 16 tokens of keys and values, for every layer, in one contiguous allocation. A request holds however many blocks its context needs.',
    lesson: 'vllm/anatomy-of-a-request',
  },
  'detokenizer': {
    term: 'Detokenizer',
    def: 'The part of the engine that turns token IDs back into text as they are produced, so a streamed answer can be printed before the request has finished.',
    lesson: 'vllm/anatomy-of-a-request',
  },
  'sampling-params': {
    term: 'SamplingParams',
    def: 'The per-request knobs vLLM turns a JSON body into: temperature, top_p, max_tokens, stop sequences, n, logprobs, and whether to stream. Different requests in the same batch can have different ones.',
    lesson: 'vllm/start-vllm',
  },
  'gpu-memory-utilization': {
    term: 'gpu_memory_utilization',
    def: 'The fraction of the GPU’s total memory vLLM is allowed to use (0.9 by default at the time of writing). Weights come out of it first; whatever is left becomes the KV cache pool.',
    lesson: 'vllm/start-vllm',
  },
  'max-model-len': {
    term: 'max_model_len',
    def: 'The longest prompt-plus-output vLLM will accept, in tokens. It bounds a single request’s worst-case KV usage and must not exceed the model’s trained context length.',
    lesson: 'vllm/tune-one-gpu',
  },
  'max-num-seqs': {
    term: 'max_num_seqs',
    def: 'The most sequences the scheduler may keep running in one engine iteration. A hard cap on concurrency per GPU: it bounds both KV usage and per-step latency.',
    lesson: 'vllm/tune-one-gpu',
  },
  'max-num-batched-tokens': {
    term: 'max_num_batched_tokens',
    def: 'How many tokens one engine iteration may process in total (prefill plus decode). Long prompts get split into chunks of this size instead of monopolising a step.',
    lesson: 'vllm/tune-one-gpu',
  },
  'enforce-eager': {
    term: 'enforce_eager',
    def: 'vLLM’s "no CUDA graphs" mode. It skips capturing and replaying the decode step, which makes small or old GPUs easier to run, at the cost of some per-step launch overhead.',
    lesson: 'vllm/tune-one-gpu',
  },
  'engine-metrics': {
    term: '/metrics endpoint',
    def: 'The Prometheus-format counters vLLM serves on its own port: requests waiting and running, KV-cache usage, token throughput and the per-step timing histograms. The cheapest way to see what the engine is really doing.',
    lesson: 'vllm/tune-one-gpu',
  },
  'operating-point': {
    term: 'Operating point',
    def: 'The specific setting of a server — batch cap, context length, memory budget — that sits inside your latency SLO while giving the most goodput. Found by sweeping, not by guessing.',
    lesson: 'vllm/tune-one-gpu',
  },
  'optimization-plan': {
    term: 'Optimization plan',
    def: 'A repeatable 7-step engineering loop for LLM serving: inspect the hardware, define the workload, set SLO metrics, deploy a baseline, benchmark, change one knob at a time, and record every experiment.',
    lesson: 'vllm/optimization-plan',
  },
  'traffic-generator': {
    term: 'Benchmark traffic generator',
    def: 'A harness or CLI tool (such as vLLM’s bench serve) that replays a fixed prompt and output length distribution at controlled arrival rates to measure server performance under realistic concurrency.',
    lesson: 'vllm/optimization-plan',
  },
  'keep-or-revert': {
    term: 'Keep-or-revert rule',
    def: 'The discipline of benchmarking a single isolated knob change against a baseline, keeping it only if goodput improves without violating SLOs, and immediately reverting otherwise.',
    lesson: 'vllm/optimization-plan',
  },
  'offline-inference': {
    term: 'Offline inference (batch processing)',
    def: 'Running model inference directly in a Python script using the LLM class without an HTTP server or network layer. Built for high-throughput batch workloads like dataset translation and document summarization.',
    lesson: 'vllm/start-vllm',
  },
  'llm-class': {
    term: 'vLLM LLM class',
    def: 'The Python API in vLLM for running offline batch generation. It runs the scheduler, memory manager, and model runner directly in your process without exposing HTTP endpoints.',
    lesson: 'vllm/start-vllm',
  },
  'worker-process': {
    term: 'Worker process',
    def: 'An independent process spawned by the serving engine to manage a single GPU. In multi-GPU setups using tensor parallelism, one worker runs per GPU to execute shards of the model.',
    lesson: 'vllm/anatomy-of-a-request',
  },
  'memory-profiling': {
    term: 'Memory profiling',
    def: 'The engine initialization step where a dummy forward pass runs to measure peak activation memory before allocating the KV cache pool, ensuring the server will not run out of memory under load.',
    lesson: 'vllm/anatomy-of-a-request',
  },
};
