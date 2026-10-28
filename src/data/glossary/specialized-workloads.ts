import type { GlossaryEntry } from '../glossary';

// Vocabulary introduced in Module 26. Terms the learner already met earlier
// (kv-cache, prefix-cache, prefill, decode, batch, roofline, quantization, ...)
// live in src/data/glossary.ts or earlier per-module files.
//
// One entry per genuinely new concept. Where a term already has a common name in
// the field, the definition says what it actually is mechanically — this course
// teaches the mechanism and the name follows from it.

export const terms: Record<string, GlossaryEntry> = {
  // ── 1 · structured output ──
  'constrained-decoding': {
    term: 'Constrained decoding',
    def: 'Generating with the sampler restricted to tokens that keep a grammar valid. The engine builds a mask over the vocabulary from a JSON Schema, regex or grammar, and zeroes everything else in the logits before the softmax. It adds no tokens and no forward passes — only one mask evaluation per step — so it buys correctness, not speed.',
    lesson: 'specialized-workloads/structured-output',
  },
  'allowed-set': {
    term: 'Allowed set (allowed fraction)',
    def: 'The tokens a constrained decoder permits at the current grammar state, as a fraction of the vocabulary. For a JSON grammar this swings enormously between states: about 0.04% of a 128K vocabulary right after a comma, and about 62% inside a string. A narrow state is effectively a forced token.',
    lesson: 'specialized-workloads/structured-output',
  },
  'tool-call': {
    term: 'Tool call',
    def: 'A structured request from the model to run a function, usually emitted as JSON with a name and an argument object. Over 40% of its tokens are grammar-forced scaffolding (function name, keys, punctuation) rather than information, so tool schemas are a prefill and streaming cost as well as a correctness one.',
    lesson: 'specialized-workloads/structured-output',
  },
  'schema-constraint': {
    term: 'Schema constraint',
    def: 'A restriction expressed in a JSON Schema that a decoder enforces structurally: required fields, types, enums, array lengths. It is a type system over syntax — it guarantees the shape of the output but knows nothing about whether a referenced record exists or whether the caller may use it.',
    lesson: 'specialized-workloads/structured-output',
  },
  // ── 2 · multi-LoRA ──
  lora: {
    term: 'LoRA (low-rank adaptation)',
    def: 'A way to specialise a model by training a thin pair of matrices added to its existing weights, leaving the base frozen. For an (out x in) matrix of rank r the adapter holds r(in + out) parameters instead of in*out — on Llama-3.1-8B at r=16 that is 32.5M parameters, 0.41% of the model.',
    lesson: 'specialized-workloads/multi-lora',
  },
  adapter: {
    term: 'Adapter',
    def: 'A small trained overlay on a frozen base model — LoRA being the common kind. Because every adapter shares one base, the marginal memory cost of serving an extra tenant is tens of megabytes rather than the whole model, which is what makes many-specialist serving economic.',
    lesson: 'specialized-workloads/multi-lora',
  },
  'lora-rank': {
    term: 'LoRA rank (r)',
    def: 'The width of the thin inner dimension in a LoRA update, written r(in + out) for a matrix of shape (out x in). It is the main quality-versus-size dial: linear in parameter count, and usually sufficient in the single digits for style and format and in the tens for new capability.',
    lesson: 'specialized-workloads/multi-lora',
  },
  'batch-mixing': {
    term: 'Batch mixing (adapter mixing)',
    def: 'When one decode batch contains sequences wanting different adapters, rows must be grouped by adapter and each group becomes its own GEMM. Because the weight read is per-GEMM, spreading a batch across many adapters costs like shrinking the batch — adapter count per step and batch size are effectively the same dial.',
    lesson: 'specialized-workloads/multi-lora',
  },
  // ── 3 · multimodal ──
  'vision-tower': {
    term: 'Vision tower',
    def: 'The image encoder in a vision-language model: a transformer that slices an image into fixed-size patches and embeds each. It runs before the language model, on pixels rather than tokens, and is a small model on a short sequence — so it is a small slice of prefill time but its own batching axis.',
    lesson: 'specialized-workloads/multimodal',
  },
  patch: {
    term: 'Patch',
    def: 'One square tile of an image as seen by the vision encoder. A 336-pixel image with 14-pixel patches is 24x24 = 576 patches, and each patch becomes one token entering the language model. This is where an image’s token cost comes from.',
    lesson: 'specialized-workloads/multimodal',
  },
  projector: {
    term: 'Projector',
    def: 'The small mapping between a vision model and a language model — typically one or two linear layers that lift patch embeddings into the language model’s hidden width. It is a translation layer, not a capacity component (roughly 10M parameters against a 7B model), which is why multimodal fine-tuning mostly tunes it and the language model.',
    lesson: 'specialized-workloads/multimodal',
  },
  'image-token': {
    term: 'Image token',
    def: 'A token derived from a patch of an image, indistinguishable in the sequence from a text token once the projector has run. It is prefilled like text, stored in the KV cache like text, and never evicted — which is why a single 576-token image permanently occupies about 74 MB of cache on a typical 8B model.',
    lesson: 'specialized-workloads/multimodal',
  },
  'any-resolution': {
    term: 'Any-resolution input',
    def: 'Feeding several crops of an image and pooling features so that higher image detail does not inflate the token count — as in LLaVA-1.6, which sees four crops yet still emits 576 tokens. It decouples image detail from context cost, trading extra vision-tower FLOPs for a bounded KV footprint.',
    lesson: 'specialized-workloads/multimodal',
  },
  // ── 4 · on-device & edge ──
  'edge-inference': {
    term: 'Edge inference (on-device serving)',
    def: 'Running the model on the user’s own hardware — a phone, a laptop, a camera, a car — instead of a datacenter. It removes the network round trip and keeps raw data local, at the price of a few gigabytes of shared RAM, a battery power budget, a fanless thermal limit and hardware you do not control. Decode is still bandwidth-bound, so bytes per weight decide tokens per second.',
    lesson: 'specialized-workloads/edge-inference',
  },
  'device-runtime': {
    term: 'Device runtime',
    def: 'The software layer between a converted model file and the device’s silicon — llama.cpp, ONNX Runtime, LiteRT, Core ML or ExecuTorch at the time of writing. It chooses kernels, schedules work across CPU, GPU and NPU, and on mobile platforms can hand whole subgraphs to an accelerator through a delegate. Operator coverage is the usual source of surprises: one unsupported operation can force a fallback that costs more than the accelerator saved.',
    lesson: 'specialized-workloads/edge-inference',
  },
  npu: {
    term: 'NPU (neural processing unit)',
    def: 'An accelerator inside a phone or laptop system on a chip, built from many small multiply-accumulate engines for low-precision matrix math. It is fast at dense 8-bit or 4-bit work, which makes it useful for prefill, and it is mostly idle during decode because decode is limited by memory bandwidth rather than arithmetic.',
    lesson: 'specialized-workloads/edge-inference',
  },
  'tops-per-watt': {
    term: 'TOPS and TOPS/W',
    def: 'TOPS counts trillions of operations per second, usually at 8-bit integer precision; TOPS per watt divides that by the power it costs to sustain. On a phone the ratio is the more honest number, because the chassis has to remove the heat. Neither predicts decode speed: a 3B model needs about 6.4 GFLOP per token, which an NPU finishes in under a millisecond while the weight read takes tens of milliseconds.',
    lesson: 'specialized-workloads/edge-inference',
  },
  'thermal-throttling': {
    term: 'Thermal throttling',
    def: 'A fanless device lowering its clock when it gets hot, which usually lowers the memory clock and the achieved bandwidth with it. It is the normal steady state under sustained load, not a fault: at 55% of the burst clock a phone that gave 20 tokens per second gives about 11. It is why on-device benchmarks must run for at least a minute and report the sustained tail separately from the first seconds.',
    lesson: 'specialized-workloads/edge-inference',
  },
  'edge-cloud-split': {
    term: 'Edge-cloud split',
    def: 'A deployment in which a small model answers on the device and escalates hard requests to a larger cloud model. The engineering is the routing rule, driven by network round-trip time, battery, temperature, model confidence and request size. The break-even is usually decided by compute rather than by the network hop: a long prompt prefills far faster in the cloud, while a tiny intent classification is dominated by the round trip.',
    lesson: 'specialized-workloads/edge-inference',
  },
  'heterogeneous-compute': {
    term: 'Heterogeneous compute',
    def: 'Splitting a model’s graph across a CPU, GPU and NPU inside one device. Each crossing between units costs a fixed synchronisation plus a copy of whatever tensor moves, so the partition matters more than the hardware: crossing twice per block is a few percent of a decode step, while crossing per operator can cost more than the step itself.',
    lesson: 'specialized-workloads/edge-inference',
  },
  // ── 5 · agents & reasoning ──
  'reasoning-trace': {
    term: 'Reasoning trace',
    def: 'The chain-of-thought tokens a reasoning model emits before its answer. Because the trace stays in the context, a long one slows itself down (decode is memory-bound and the KV grows each step) and makes a multi-turn conversation progressively more expensive.',
    lesson: 'specialized-workloads/agents-reasoning',
  },
  'agent-loop': {
    term: 'Agent loop',
    def: 'Repeated turns in which the model calls a tool, receives an observation, and continues. Each observation appends to the context and nothing is removed, so without cache reuse the prefill work grows with the square of the number of turns.',
    lesson: 'specialized-workloads/agents-reasoning',
  },
  'cache-hit-rate': {
    term: 'Prefix cache hit rate',
    def: 'The fraction of a request’s prompt tokens whose KV is already cached. Multi-turn chat and tool loops are near-perfect reuse (each turn only appends), but a timestamp, request ID, or non-deterministic tool result in the prefix collapses it to zero — and a zero hit rate cannot be fixed by a better eviction policy.',
    lesson: 'specialized-workloads/agents-reasoning',
  },
  // ── 6 · batch & RL ──
  rollout: {
    term: 'Rollout',
    def: 'One complete generation from a policy during reinforcement learning — prompt, generated tokens, and often a reward computation, held in memory until the whole group finishes. One optimisation step needs thousands at once, which is why rollout generation is sized by tokens in flight rather than by model size.',
    lesson: 'specialized-workloads/batch-rl-rollouts',
  },
  'rollout-group': {
    term: 'Rollout group',
    def: 'The set of rollouts generated for one optimisation step, all resident at the same time. Peak memory equals the group size (not the average, which drains as rollouts finish), and the group’s KV cache is what dictates how many GPUs the run needs.',
    lesson: 'specialized-workloads/batch-rl-rollouts',
  },
  'generation-cap': {
    term: 'Generation cap',
    def: 'A hard limit on how many tokens a rollout may produce. Too high and some rollouts never terminate, stalling the whole optimisation step; too low and the reward signal is computed on truncated, half-finished reasoning. The cap should sit at roughly 75% of the context to leave room for the reward pass.',
    lesson: 'specialized-workloads/batch-rl-rollouts',
  },
  'ragged-completion': {
    term: 'Ragged completion',
    def: 'The situation in a rollout group where sequences finish at very different times, so the batch steadily drains instead of staying full. It is the opposite of continuous batching in serving, and it means both that peak memory is the group size and that the tail runs at reduced throughput.',
    lesson: 'specialized-workloads/batch-rl-rollouts',
  },
  'fp-non-associativity': {
    term: 'Non-associativity (floating point)',
    def: 'In exact arithmetic the grouping of a sum does not matter; in floating point every addition rounds to the nearest representable number, so it does. In FP16, 1 + 0.0005 + 0.0005 gives 1.001953 one way and 1.000977 the other — one unit in the last place, about 0.1% — because the grid just above 1.0 is spaced 0.000977 apart. BF16’s grid is eight times coarser. It is the root cause of batch-dependent outputs.',
    lesson: 'specialized-workloads/batch-rl-rollouts',
  },
  'batch-invariant': {
    term: 'Batch-invariant kernels',
    def: 'Kernels whose summation order for a token does not depend on the batch it is travelling in, so the same prompt yields the same logits at batch 1 and at batch 64. Proposed and implemented by some serving engines at the time of writing; a fixed reduction gives up the fastest split for each shape, which costs throughput, and it is the price of a reproducible rollout.',
    lesson: 'specialized-workloads/batch-rl-rollouts',
  },
  'importance-ratio': {
    term: 'Importance ratio',
    def: 'The correction factor between generating with one policy and scoring with another: r = exp(log π_trainer − log π_engine) for a sampled token. A log-probability gap of δ moves it by about (1 + δ). A one-off gap is harmless, but a systematic one biases every update, and a large one lands outside the clipping range and discards the sample.',
    lesson: 'specialized-workloads/batch-rl-rollouts',
  },
  // ── 7 · diffusion ──
  latent: {
    term: 'Latent',
    def: 'The compact representation a diffusion model actually works on — a 512x512 image becomes 4x64x64 = 16,384 values after an 8x downsample. Every denoising step runs on the latent; full-resolution pixels appear only at the final VAE decode.',
    lesson: 'specialized-workloads/diffusion',
  },
  'denoising-step': {
    term: 'Denoising step',
    def: 'One iteration of a diffusion sampling loop, where the network takes the current latent plus a timestep and predicts less noise. Twenty to fifty of them run per image, and each re-reads every weight — which is why diffusion has no cache to amortise and is compute-bound where decode is memory-bound.',
    lesson: 'specialized-workloads/diffusion',
  },
  'classifier-free-guidance': {
    term: 'Classifier-free guidance (CFG)',
    def: 'The standard diffusion sampling trick that runs the network twice per step — once on the prompt and once on an empty prompt — and combines the results toward the prompt. It is a 2x compute cost that is effectively mandatory for prompt-following, and published timings already include it.',
    lesson: 'specialized-workloads/diffusion',
  },
  'feature-caching': {
    term: 'Feature caching (diffusion)',
    def: 'Reusing shallow U-Net features across denoising steps, which change little as the latent stabilises, to cut step compute by 65–85%. Unlike prefix caching it is a lossy approximation rather than an exact reuse, because the features depend on a latent that changes every step.',
    lesson: 'specialized-workloads/diffusion',
  },
  'few-step-distillation': {
    term: 'Few-step distillation',
    def: 'Training a diffusion model (or distilling one) to sample in 1–4 steps instead of 20–50 — as in SDXL-Turbo. It is the largest latency lever in the ecosystem, it produces a genuinely different image, and it is a training-time decision rather than a serving one.',
    lesson: 'specialized-workloads/diffusion',
  },
};