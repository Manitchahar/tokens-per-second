// Module 11 · Quantization — new terms.
import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  'scale-factor': {
    term: 'Scale factor',
    def: 'The number you divide a real value by before rounding it onto a small integer grid. The grid step, and therefore the worst-case rounding error, is set by the largest magnitude the scale must cover.',
    lesson: 'quantization/how-quantization-works',
  },
  'zero-point': {
    term: 'Zero-point',
    def: 'The integer that represents the real value zero, in a signed quantized format. It shifts the grid so its resolution sits where the data is, instead of at one end of an unsigned range.',
    lesson: 'quantization/how-quantization-works',
  },
  dequantization: {
    term: 'Dequantization',
    def: 'Multiplying a stored integer back by its scale (and adding the zero-point) to recover an approximate float. This step is exact and free — the error was already introduced by the rounding step.',
    lesson: 'quantization/how-quantization-works',
  },
  'group-quantization': {
    term: 'Group-wise quantization',
    def: 'Splitting a weight matrix into groups of g values that each share one scale, so an outlier only wrecks its own group instead of the whole tensor. Costs 4 bytes of metadata per group.',
    lesson: 'quantization/how-quantization-works',
  },
  'metadata-overhead': {
    term: 'Metadata overhead',
    def: 'The extra storage scales and zero-points require, as a fraction of the packed weights: 32/(g·b) for group size g at b bits. A "4-bit" model at group 128 is really 4.25 bits per weight.',
    lesson: 'quantization/how-quantization-works',
  },
  'per-channel': {
    term: 'Per-channel quantization',
    def: 'Giving every output channel of a weight matrix its own scale. Cheap (one scale per row, regardless of row length) and effective, because channels in an LLM genuinely differ in magnitude.',
    lesson: 'quantization/how-quantization-works',
  },
  'weight-only-quantization': {
    term: 'Weight-only quantization (W4A16)',
    def: 'Narrowing the weights while leaving activations in full precision. The standard 4-bit recipe: 4× less weight memory and a proportionally shorter memory-bound decode step, with prefill essentially unchanged.',
    lesson: 'quantization/quant-schemes',
  },
  'w8a8': {
    term: 'W8A8',
    def: 'Quantizing both weights and activations to INT8, so the matmul actually runs on the integer tensor cores. This is the prefill/throughput lever — a 2× on the compute term where weight-only 4-bit gives nothing.',
    lesson: 'quantization/quant-schemes',
  },
  'fp8-e4m3': {
    term: 'FP8 E4M3',
    def: 'An 8-bit float with 1 sign, 4 exponent and 3 mantissa bits. The per-element exponent means it needs no group scales, handles outliers like BF16, and runs at 2× the BF16 tensor-core rate on Hopper and newer.',
    lesson: 'quantization/quant-schemes',
  },
  'arithmetic-intensity-quant': {
    term: 'Arithmetic intensity, quantized',
    def: 'FLOPs per byte moved. Narrowing the weights raises intensity, which keeps decode memory-bound to a larger batch. Quantization moves the roofline position, not just the memory footprint.',
    lesson: 'quantization/why-precision',
  },
  rtn: {
    term: 'Round-to-nearest (RTN)',
    def: 'The baseline quantizer: divide by the group scale, round, clamp, done. Needs no calibration and no second-order information, and it is good enough at 8 bits but visibly lossy at 4.',
    lesson: 'quantization/awq-gptq-smoothquant',
  },
  awq: {
    term: 'AWQ (Activation-aware Weight Quantization)',
    def: 'Protects the weight channels that matter most to the output — those where ‖W‖·‖X‖ is large — by giving them a finer quantisation grid rather than more bits. Costs one calibration pass.',
    lesson: 'quantization/awq-gptq-smoothquant',
  },
  'inverse-hessian': {
    term: 'Inverse Hessian',
    def: 'The inverse of X·Xᵀ from calibration inputs. It identifies the weight directions the layer output is least sensitive to, which is exactly where GPTQ is allowed to put its rounding error.',
    lesson: 'quantization/awq-gptq-smoothquant',
  },
  gptq: {
    term: 'GPTQ',
    def: 'Quantizes a weight matrix column by column and, after each column, pushes the rounding error onto the not-yet-quantized columns, weighted by the inverse Hessian. Optimises the layer output, not the weights.',
    lesson: 'quantization/awq-gptq-smoothquant',
  },
  'error-compensation': {
    term: 'Error compensation',
    def: 'Redistributing the error from a rounding decision onto weights you have not quantized yet. It can make weight-space error look worse while output error improves — the method is optimising the output.',
    lesson: 'quantization/awq-gptq-smoothquant',
  },
  smoothquant: {
    term: 'SmoothQuant',
    def: 'Divides each activation channel by a scale and multiplies the matching weight row by its reciprocal — an exact algebraic identity that shrinks activation outliers so both tensors quantize more easily.',
    lesson: 'quantization/awq-gptq-smoothquant',
  },
  calibration: {
    term: 'Calibration set',
    def: 'A few hundred representative prompts run through the model to measure the activation statistics a quantizer needs. Getting it right is operational work: calibrate on your own traffic, not a generic dataset.',
    lesson: 'quantization/awq-gptq-smoothquant',
  },
  salience: {
    term: 'Salient weight channel',
    def: 'An output channel whose contribution to the layer output is disproportionately large — roughly ‖W[:,j]‖·‖X[:,j]‖. Typically 1% of channels carry a large share of the total, and they are not the largest weights.',
    lesson: 'quantization/awq-gptq-smoothquant',
  },
  'fused-dequant': {
    term: 'Fused dequantize-and-matmul',
    def: 'A GEMM kernel that loads packed codes, widens them to the compute dtype in registers, and feeds the tensor cores — so no full-precision copy ever reaches HBM. Without this, quantization is a memory win and a speed loss.',
    lesson: 'quantization/quantized-kernels-kv',
  },
  'unfused-dequant': {
    term: 'Unfused dequantization',
    def: 'Materialising a full-precision copy of the weights before the matmul. It moves more bytes than plain BF16, so it is strictly slower at inference — the most common hand-rolled quantization bug.',
    lesson: 'quantization/quantized-kernels-kv',
  },
  'kv-cache-quantization': {
    term: 'KV cache quantization',
    def: 'Storing the cached keys and values in FP8 or INT8 instead of BF16. Halves the other half of the decode step’s byte count and roughly doubles concurrency; independent of weight quantization.',
    lesson: 'quantization/quantized-kernels-kv',
  },
  'kv-cache-grouping': {
    term: 'KV cache grouping',
    def: 'How many KV values share one scale: per-tensor, per-token (the usual choice), or per-head. Per-token is nearly free because a scale is naturally associated with the token being written.',
    lesson: 'quantization/quantized-kernels-kv',
  },
  'long-context-quality-loss': {
    term: 'Long-context quality loss',
    def: 'The specific way KV cache quantization hurts: attention accumulates error from every quantized key, so at long context the model repeats itself and loses long-range recall rather than producing gibberish.',
    lesson: 'quantization/quantized-kernels-kv',
  },
  'perplexity': {
    term: 'Perplexity (PPL)',
    def: 'exp of the mean negative log-likelihood the model assigns to held-out text. A model’s overall surprise at real text — cheap, sensitive, and the first regression metric you should check.',
    lesson: 'quantization/measure-quality',
  },
  'eval-harness': {
    term: 'Evaluation harness',
    def: 'A fixed, versioned set of tasks and prompts used to compare model variants. The point is a stable yardstick: same questions, same scoring, same seed, so a number means something next week.',
    lesson: 'quantization/measure-quality',
  },
  'golden-set': {
    term: 'Golden set',
    def: 'A small, hand-checked set of prompts and expected behaviours kept out of automated tuning. It catches the regressions an aggregate score averages away.',
    lesson: 'quantization/measure-quality',
  },
  'output-weighted-error': {
    term: 'Output-weighted error',
    def: 'Quantization error measured as ‖ΔW·X‖/‖W·X‖ rather than ‖ΔW‖/‖W‖ — i.e. weighted by the activations. It is the metric that predicts quality, and the reason AWQ and GPTQ can look worse in weight space yet better in output.',
    lesson: 'quantization/awq-gptq-smoothquant',
  },
  'quantization-error': {
    term: 'Quantization error',
    def: 'The distance between a real value and the grid point it rounded to, at most half the grid step s/2. Across independent layers it accumulates as ε·√L, not ε·L — the rule that decides whether 4 bits survives a deep model.',
    lesson: 'quantization/how-quantization-works',
  },
  'bits-per-weight': {
    term: 'Bits per weight (effective)',
    def: 'Data bits plus metadata: b·(1 + 32/(g·b)) = b + 32/g. Group-128 4-bit is 4.25 effective bits, which is the honest number to put in a memory estimate.',
    lesson: 'quantization/how-quantization-works',
  },
  'quality-budget': {
    term: 'Quality budget',
    def: 'How much task quality you are willing to trade for speed and memory, expressed in the units that matter for your product. A coding assistant and a document summariser have very different budgets.',
    lesson: 'quantization/measure-quality',
  },
  'pareto-frontier': {
    term: 'Pareto frontier',
    def: 'The set of configurations where you cannot improve quality, speed, or memory without giving up another. Quantization choices are made on this curve, not on a single best scheme.',
    lesson: 'quantization/project-pareto',
  },
  'goodput-quantization': {
    term: 'Goodput under quantization',
    def: 'Throughput counting only requests that met the SLO after quantization. A scheme can raise raw tokens/second while lowering goodput if quality slips enough to change task completion.',
    lesson: 'quantization/measure-quality',
  },
  'distillation': {
    term: 'Distillation',
    def: 'Training a smaller student model to reproduce a larger teacher model’s behaviour, either by matching its probability distribution at every token or by fine-tuning on text the teacher generated. It produces a new model, not a compressed copy of the old one.',
    lesson: 'quantization/distillation-pruning',
  },
  'teacher-student': {
    term: 'Teacher and student',
    def: 'In distillation, the big model that supplies the targets (the teacher) and the small model being trained (the student). The student has its own architecture and its own weights; only the behaviour is copied.',
    lesson: 'quantization/distillation-pruning',
  },
  'hard-label': {
    term: 'Hard label',
    def: 'The single correct next token in a training example — a probability vector with a 1 on one token and zeros everywhere else. It says nothing about how plausible the other candidates were.',
    lesson: 'quantization/distillation-pruning',
  },
  'soft-label': {
    term: 'Soft label',
    def: 'The teacher’s full probability distribution over the next token, not just its argmax. The shape of the distribution — Paris 0.84, Lyon 0.11, Marseille 0.04 — is the extra signal that makes distillation work.',
    lesson: 'quantization/distillation-pruning',
  },
  'distillation-temperature': {
    term: 'Distillation temperature',
    def: 'The number T that divides the teacher’s logits before softmax. Raising it flattens the distribution so the small probabilities become visible to the loss; the gradient then shrinks as 1/T², which is why implementations multiply the soft loss by T².',
    lesson: 'quantization/distillation-pruning',
  },
  'kl-divergence': {
    term: 'KL divergence',
    def: 'A measure of how much one probability distribution differs from another, Σ q log(q/p). Soft-label distillation minimises the KL between teacher and student, because the teacher’s own entropy is a constant the student cannot change.',
    lesson: 'quantization/distillation-pruning',
  },
  'sequence-level-distillation': {
    term: 'Sequence-level distillation',
    def: 'Distilling from text rather than logits: generate responses with the teacher, filter them, then fine-tune the student on the prompt–response pairs. It needs the teacher’s outputs but not its weights.',
    lesson: 'quantization/distillation-pruning',
  },
  'pruning': {
    term: 'Pruning',
    def: 'Removing weights from a trained model — individually, in a fixed 2:4 pattern, or by deleting whole heads or layers. Only removal that changes the matrix’s shape, or is stored compressed and run on sparse hardware, saves any time.',
    lesson: 'quantization/distillation-pruning',
  },
  'unstructured-pruning': {
    term: 'Unstructured pruning',
    def: 'Setting individual weights to zero wherever they matter least. The matrix keeps its shape, so dense kernels still read and multiply the zeros — no speedup without compressed storage and a sparse kernel.',
    lesson: 'quantization/distillation-pruning',
  },
  'semi-structured-sparsity': {
    term: '2:4 semi-structured sparsity',
    def: 'Exactly two non-zero weights in every group of four — a pattern NVIDIA’s sparse tensor cores can skip from Ampere onwards. Up to 2× the peak math, so it helps compute-bound prefill, not bandwidth-bound decode.',
    lesson: 'quantization/distillation-pruning',
  },
  'structured-pruning': {
    term: 'Structured pruning',
    def: 'Deleting whole attention heads, channels or layers so the matrices genuinely shrink. The only pruning that reliably speeds up decode on ordinary dense kernels, and the one with the largest quality cost per weight removed.',
    lesson: 'quantization/distillation-pruning',
  },
  'recovery-fine-tuning': {
    term: 'Recovery fine-tuning',
    def: 'Continuing training after pruning so the surviving weights adapt to the ones that were deleted. Without it a pruned model is usually broken; with it most of the quality comes back, at the cost of a training run.',
    lesson: 'quantization/distillation-pruning',
  },
};