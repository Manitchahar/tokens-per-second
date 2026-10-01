import type { GlossaryEntry } from '../glossary';

// Terms introduced in Module 04 · Run Your First LLM.
// Terms already defined in src/data/glossary.ts (inference, tokenizer, kv-cache, sse, oom, ...)
// are NOT repeated here; this file only adds what the learner meets for the first time here.
export const terms: Record<string, GlossaryEntry> = {
  'open-weights': {
    term: 'Open weights',
    def: 'A model whose trained weights are published for anyone to download and run. The code may be open source or not; what matters for inference is that the numbers in the checkpoint are yours to fetch.',
    lesson: 'first-llm/choose-a-model',
  },
  checkpoint: {
    term: 'Checkpoint',
    def: 'The file on disk holding a trained model’s weights. For an 8B model in BF16 it is roughly 16 GB — about 6,000 photographs, or 0.004% of the data an H100 reads in one second.',
    lesson: 'first-llm/choose-a-model',
  },
  'weights-memory': {
    term: 'Weights memory',
    def: 'The bytes a model occupies just to exist: parameters × bytes per weight. In BF16, 8.03B parameters × 2 bytes = 16.1 GB. This is a floor you cannot optimize away, only shrink by quantizing.',
    lesson: 'first-llm/choose-a-model',
  },
  'headroom': {
    term: 'Headroom',
    def: 'The memory left over after weights and KV cache are placed. A model using 16.1 GB of a 16 GB GPU has none, and one long conversation will crash it. Budget roughly 10–20% before you serve anyone.',
    lesson: 'first-llm/choose-a-model',
  },
  'head-count': {
    term: 'Parameter count',
    def: 'How many weights a model has, written as 0.5B, 8B, 70B. It predicts quality, weights memory, and — because decode reads all of them per token — your speed. All three from one number.',
    lesson: 'first-llm/choose-a-model',
  },
  'architecture-provenance': {
    term: 'Base vs instruct model',
    def: 'A base model continues your text; an instruct (chat) model was fine-tuned on conversations and follows instructions. Same architecture, same weight count, very different behaviour — and only the instruct one answers questions.',
    lesson: 'first-llm/choose-a-model',
  },
  'chat-template': {
    term: 'Chat template',
    def: 'The exact text format a chat model was trained on, such as special tokens around each turn. Using the wrong one makes a model ramble or ignore you, and the template is not in the weights — it is a function on the tokenizer.',
    lesson: 'first-llm/load-and-generate',
  },
  'special-tokens': {
    term: 'Special tokens',
    def: 'Reserved entries in the vocabulary that mean control rather than text: <|begin_of_text|>, <|eot_id|>, <|im_start|>. They mark the boundaries of turns and are how the model knows to stop.',
    lesson: 'first-llm/load-and-generate',
  },
  eos: {
    term: 'EOS token',
    def: 'The end-of-sequence token. When the model emits it, generation is finished. Every API call takes a stop token list, because getting it wrong means paying to generate thousands of tokens of garbage.',
    lesson: 'first-llm/load-and-generate',
  },
  'max-new-tokens': {
    term: 'max_new_tokens',
    def: 'The hard cap on how many tokens generation may produce. It is the single most effective protection against a run that never stops: the worst case for a 500-token answer is bounded the moment you set it.',
    lesson: 'first-llm/load-and-generate',
  },
  greedy: {
    term: 'Greedy decoding',
    def: 'Always picking the highest-probability next token. Deterministic and fast, but it can loop forever, because the same context can keep producing the same favourite token.',
    lesson: 'first-llm/load-and-generate',
  },
  'generate-loop': {
    term: 'The generate loop',
    def: 'The cycle you will reimplement for the rest of the course: run a forward pass, sample one token, append it to the input, repeat until a stop condition. Everything else is an optimisation of this loop.',
    lesson: 'first-llm/load-and-generate',
  },
  'streaming-token': {
    term: 'Streaming',
    def: 'Sending each token to the client the moment it exists, instead of waiting for the whole answer. Perceived speed is dominated by time to first token, which streaming makes almost independent of answer length.',
    lesson: 'first-llm/load-and-generate',
  },
  'ttft-mental': {
    term: 'Prefill before first token',
    def: 'Before any word appears, the whole prompt must be processed. That delay is time to first token, and it is the number users actually perceive when they say "the app feels slow".',
    lesson: 'first-llm/watch-the-gpu',
  },
  'nvitop': {
    term: 'Live GPU monitor',
    def: 'A second terminal running nvidia-smi -l 1, so you watch utilization, memory and power while your model generates. Seeing the GPU go idle during Python is how you learn where the bottleneck is.',
    lesson: 'first-llm/watch-the-gpu',
  },
  'cuda-event-timing': {
    term: 'CUDA event timing',
    def: 'Timing with start.record() / end.record() on the device clock. Immune to the asynchrony trap from Module 02 and to network jitter on a remote GPU, which is why every serious benchmark uses it.',
    lesson: 'first-llm/watch-the-gpu',
  },
  'warmup-run': {
    term: 'Warm-up run',
    def: 'The first generate() call pays one-off costs: kernel compilation, lazy allocation, weight paging from disk. Timings taken before a warm-up are 2–10× pessimistic and every conclusion drawn from them is wrong.',
    lesson: 'first-llm/watch-the-gpu',
  },
  'openai-compatible': {
    term: 'OpenAI-compatible API',
    def: 'An endpoint that accepts the same JSON as OpenAI’s /v1/chat/completions. Nearly every open engine offers one, so you can point an existing app at a local GPU server by changing a base URL.',
    lesson: 'first-llm/serve-over-http',
  },
  'health-endpoint': {
    term: 'Health endpoint',
    def: 'A cheap route like /health that loads no model and does no work. Load balancers poll it to decide where to send traffic, which is why it must stay fast even when the GPU is completely saturated.',
    lesson: 'first-llm/serve-over-http',
  },
  'backpressure-http': {
    term: 'Backpressure',
    def: 'When a slow client cannot keep up, the server either buffers unboundedly (memory grows until it OOMs) or tells the producer to stop. Bounding the buffer is the whole job.',
    lesson: 'first-llm/serve-over-http',
  },
  'port-forward': {
    term: 'Port forwarding',
    def: 'Tunneling a remote machine’s port to your laptop so localhost:8000 reaches the GPU host. Remember that a 40 ms round trip is roughly five decode steps before you blame the model.',
    lesson: 'first-llm/serve-over-http',
  },
  'single-slot-server': {
    term: 'Single-slot server',
    def: 'A server that runs one request at a time. Correct and terribly slow, because the GPU spends most of each step waiting. It is the right baseline to measure every optimisation against.',
    lesson: 'first-llm/lab-first-server',
  },
  idle: {
    term: 'Idle time',
    def: 'Time when the GPU is running no useful kernel — waiting on Python, sampling, or the host. Shown as hatched grey in this course’s diagrams. It is why GPU utilization and efficiency are different numbers.',
    lesson: 'first-llm/watch-the-gpu',
  },
};
