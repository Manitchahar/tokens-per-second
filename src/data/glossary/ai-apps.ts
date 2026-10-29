import type { GlossaryEntry } from '../glossary';

// Terms Module B7 (Production AI Applications) introduces.
// Existing terms defined in previous modules (pii-masking, semantic-cache, model-router, dpo, ttft, tpot)
// are referenced directly and not redefined here.
export const terms: Record<string, GlossaryEntry> = {
  guardrail: {
    term: 'Guardrail',
    def: 'Deterministic and probabilistic checks applied to an LLM’s input prompt and generated output to enforce privacy, safety, brand policy, and schema compliance before delivery to the user or downstream tools.',
    lesson: 'ai-apps/guardrails',
  },
  'prompt-injection': {
    term: 'Prompt injection',
    def: 'A vulnerability where untrusted input containing natural language instructions hijacks an LLM’s control flow, causing it to disregard system rules, execute unintended tools, or reveal confidential instructions.',
    lesson: 'ai-apps/prompt-injection',
  },
  jailbreak: {
    term: 'Jailbreak',
    def: 'An adversarial prompt designed to bypass a model’s safety alignment, system instructions, and refusal mechanisms through persona roleplay, hypothetical framing, obfuscated encodings, or automated optimization.',
    lesson: 'ai-apps/prompt-injection',
  },
  'indirect-injection': {
    term: 'Indirect prompt injection',
    def: 'An attack vector where adversarial instructions are embedded within untrusted external data (such as retrieved documents, crawled web pages, user emails, or database records) read by an LLM or agent.',
    lesson: 'ai-apps/prompt-injection',
  },
  'instruction-hierarchy': {
    term: 'Instruction hierarchy',
    def: 'A structural security principle and model training objective that enforces a strict precedence order among prompt components: system rules outrank user instructions, which outrank untrusted external data.',
    lesson: 'ai-apps/prompt-injection',
  },
  gateway: {
    term: 'Model gateway',
    def: 'A centralized reverse proxy and ingress layer for LLM applications that unifies multiple provider APIs, standardizes authentication, enforces rate limits and quotas, tracks costs, and handles provider failovers.',
    lesson: 'ai-apps/gateway-routing',
  },
  'intent-router': {
    term: 'Intent router',
    def: 'A lightweight classification layer that evaluates query complexity, domain, or required tools to dispatch each prompt to the optimal model (e.g. small and cheap vs large and capable), optimizing the cost-quality frontier.',
    lesson: 'ai-apps/gateway-routing',
  },
  'exact-cache': {
    term: 'Exact cache',
    def: 'A deterministic key-value cache that hashes normalized prompt strings and sampling hyperparameters (e.g. using SHA-256) to return identical previous generations with sub-millisecond latency and zero model inference cost.',
    lesson: 'ai-apps/gateway-routing',
  },
  drift: {
    term: 'Data / concept drift',
    def: 'A shift over time between the statistical distribution of inputs, user intents, or semantic embeddings seen during model development versus those encountered in active production traffic.',
    lesson: 'ai-apps/app-observability',
  },
  'ai-judge': {
    term: 'LLM-as-a-judge',
    def: 'Using a strong, capable language model to automatically evaluate production generations against structured criteria (factuality, relevance, adherence) on a sampled telemetry stream.',
    lesson: 'ai-apps/app-observability',
  },
  'feedback-flywheel': {
    term: 'Feedback flywheel',
    def: 'A continuous data pipeline that captures explicit user ratings and implicit behavioral signals (edits, regenerations, copies) from production to fuel automated evals, few-shot examples, and fine-tuning datasets.',
    lesson: 'ai-apps/user-feedback',
  },
  sycophancy: {
    term: 'Sycophancy',
    def: 'The learned tendency of instruction-tuned models to agree with the user’s stated beliefs, validate leading misconceptions, or flatter the prompter even when the underlying factual reality contradicts them.',
    lesson: 'ai-apps/user-feedback',
  },
};
