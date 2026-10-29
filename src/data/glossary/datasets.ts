import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  'seed-tasks': {
    term: 'Seed tasks',
    def: 'A small, high-quality, human-curated set of diverse instructions (e.g. 175 tasks in Self-Instruct) used to bootstrap and prompt language models into generating thousands of novel synthetic instruction pairs.',
    lesson: 'datasets/synthetic-data',
  },
  'reverse-instruction': {
    term: 'Reverse instruction (Backtranslation)',
    def: 'Synthesizing instruction datasets by taking authentic, high-quality reference text (code, documentation, textbooks) and prompting an LLM to generate the prompt or question that would produce it, preventing factual hallucinations in target responses.',
    lesson: 'datasets/synthetic-data',
  },
  'model-collapse': {
    term: 'Model collapse',
    def: 'A degenerative learning process where language models trained recursively on unverified synthetic data lose representation of low-probability distribution tails, causing variance to shrink and eventually degenerating into repetitive gibberish (Shumailov et al., 2023).',
    lesson: 'datasets/verifying-data',
  },
  'shingling': {
    term: 'Shingling',
    def: 'Partitioning a document into overlapping subsequences of k words or characters (k-grams), transforming unstructured text into mathematical sets for set-intersection and Jaccard similarity comparisons.',
    lesson: 'datasets/dedup-cleaning',
  },
  'minhash': {
    term: 'MinHash',
    def: 'A probabilistic hashing technique that compresses large sets of shingles into compact signatures of m hash values, where the probability of two minimum hashes colliding equals their exact Jaccard similarity.',
    lesson: 'datasets/dedup-cleaning',
  },
  'lsh': {
    term: 'LSH (Locality-Sensitive Hashing)',
    def: 'An indexing algorithm that divides MinHash signatures into b bands of r rows, hashing similar items into identical buckets to identify candidate duplicate pairs in sub-quadratic time without checking all N² pairs.',
    lesson: 'datasets/dedup-cleaning',
  },
  'pii-redaction': {
    term: 'PII redaction',
    def: 'Masking or replacing personally identifiable information (emails, phone numbers, IP addresses, secrets, credit cards) with uniform placeholder tokens to prevent language models from memorizing and regurgitating sensitive credentials.',
    lesson: 'datasets/dedup-cleaning',
  },
};
