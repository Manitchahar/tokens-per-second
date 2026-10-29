import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  'goodharts-law': {
    term: 'Goodhart’s Law',
    def: 'The principle that when a metric becomes a target, it ceases to be a reliable measure, as systems optimize specifically to satisfy the metric rather than the underlying objective.',
    lesson: 'evaluation/why-eval-is-hard',
  },
  'pass-at-k': {
    term: 'pass@k',
    def: 'The probability that at least one of k generated code samples passes all unit tests, estimated without bias using the hypergeometric combination formula.',
    lesson: 'evaluation/exact-evaluation',
  },
  'exact-match': {
    term: 'Exact match (EM)',
    def: 'A strict boolean metric where a generated prediction must match a reference answer character-for-character, typically evaluated after standard string normalization.',
    lesson: 'evaluation/exact-evaluation',
  },
  'ai-as-judge': {
    term: 'AI-as-a-judge',
    def: 'Using a strong language model to evaluate and score the quality, correctness, or alignment of another model’s outputs according to a specified rubric.',
    lesson: 'evaluation/ai-as-judge',
  },
  'position-bias': {
    term: 'Position bias',
    def: 'The systemic tendency of an LLM judge to favor whichever response appears first (or second) in a pairwise comparison prompt regardless of quality.',
    lesson: 'evaluation/ai-as-judge',
  },
  'verbosity-bias': {
    term: 'Verbosity bias',
    def: 'The systematic bias where an LLM judge assigns higher scores to longer, more elaborate responses even when shorter answers are more accurate and concise.',
    lesson: 'evaluation/ai-as-judge',
  },
  'elo-rating': {
    term: 'Elo rating',
    def: 'A relative skill rating method where models gain or lose points based on head-to-head pairwise matchups and the expected outcome between them.',
    lesson: 'evaluation/comparative-eval',
  },
  'bradley-terry': {
    term: 'Bradley-Terry model',
    def: 'A statistical model predicting the probability that item A beats item B in a paired comparison based on their latent skill parameters.',
    lesson: 'evaluation/comparative-eval',
  },
  'data-contamination': {
    term: 'Data contamination',
    def: 'The unintended or deliberate leakage of evaluation benchmark test questions and answers into a language model’s pretraining or fine-tuning dataset.',
    lesson: 'evaluation/benchmarks-contamination',
  },
  'evaluation-harness': {
    term: 'Evaluation harness',
    def: 'An automated testing framework that executes prompts against model candidates, records outputs and operational metrics, and scores results against defined rubrics.',
    lesson: 'evaluation/eval-pipeline',
  },
  'bootstrap-ci': {
    term: 'Bootstrap confidence interval',
    def: 'A statistical technique that estimates the uncertainty of a metric by repeatedly resampling the evaluation results with replacement.',
    lesson: 'evaluation/eval-pipeline',
  },
  'bertscore': {
    term: 'BERTScore',
    def: 'An evaluation metric that computes token-level semantic similarity using greedy cosine matching between contextual embedding vectors.',
    lesson: 'evaluation/exact-evaluation',
  },
  'functional-correctness': {
    term: 'Functional correctness',
    def: 'Evaluating model-generated code or reasoning by executing it inside an isolated sandbox against a suite of unit tests rather than comparing text syntax.',
    lesson: 'evaluation/exact-evaluation',
  },
  'simpsons-paradox': {
    term: 'Simpson’s paradox',
    def: 'A statistical phenomenon where a trend appears across several distinct sub-groups or slices, but reverses when the groups are combined in aggregate due to an unmeasured confounding distribution.',
    lesson: 'evaluation/eval-pipeline',
  },
  'cohens-kappa': {
    term: 'Cohen’s kappa',
    def: 'A statistical coefficient measuring inter-rater agreement between an automated judge and human raters that explicitly accounts for chance agreement.',
    lesson: 'evaluation/ai-as-judge',
  },
  'chatbot-arena': {
    term: 'Chatbot Arena',
    def: 'A crowdsourced open evaluation platform developed by LMSYS where human users judge blind pairwise model outputs to compute dynamic Elo ratings.',
    lesson: 'evaluation/comparative-eval',
  },
  'mmlu': {
    term: 'MMLU',
    def: 'Massive Multitask Language Understanding: a benchmark consisting of 57 multiple-choice subjects evaluating broad factual and academic knowledge.',
    lesson: 'evaluation/benchmarks-contamination',
  },
  'gsm8k': {
    term: 'GSM8K',
    def: 'Grade School Math 8K: an evaluation benchmark of 8,500 multi-step arithmetic word problems requiring chain-of-thought reasoning.',
    lesson: 'evaluation/benchmarks-contamination',
  },
  'humaneval': {
    term: 'HumanEval',
    def: 'A programming benchmark created by OpenAI containing 164 Python coding challenges evaluated by unit test functional correctness.',
    lesson: 'evaluation/benchmarks-contamination',
  },
  'model-selection-funnel': {
    term: 'Model selection funnel',
    def: 'A four-stage operational process that narrows hundreds of candidate models down through hard constraints, coarse benchmarks, golden evaluations, and canary deployments.',
    lesson: 'evaluation/model-selection',
  },
  'break-even-volume': {
    term: 'Break-even token volume',
    def: 'The monthly token generation volume at which the fixed leasing cost of dedicated GPU infrastructure matches the variable price of commercial cloud APIs.',
    lesson: 'evaluation/model-selection',
  },
};
