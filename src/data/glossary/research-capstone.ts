// Module 28 · Research & Capstone. Terms taught in this module.
import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  'threats-to-validity': {
    term: 'Threats to validity',
    def: 'The section of a research paper listing, in the authors’ own words, the things that would break the result: small samples, one model, unrepresentative workloads, unsupported hardware. The most honest part of a paper.',
    lesson: 'research-capstone/read-papers',
  },
  'rig-noise': {
    term: 'Rig noise (run-to-run spread)',
    def: 'How much the same measurement moves when you run it again on the same machine. Written c, and about 5–10% on a shared virtualised GPU. Without it, a benchmark number is just a number.',
    lesson: 'research-capstone/reproduce-research',
  },
  'repeats-needed': {
    term: 'Repeats needed',
    def: 'The number of repeats required to defend a claimed effect δ against rig noise c: k ≈ (2c/δ)². A 4% effect against 5% noise needs about 6 repeats; against 10% noise, about 25.',
    lesson: 'research-capstone/reproduce-research',
  },
  reproduction: {
    term: 'Reproduction',
    def: 'Independently regenerating a published result on your own setup, and reporting honestly whether you reproduced it, did not, or could not tell. Ends in a written verdict, not a claim of success.',
    lesson: 'research-capstone/reproduce-research',
  },
  'reproduce-the-mechanism': {
    term: 'Reproduce the mechanism',
    def: 'When you lack the paper’s hardware, test the idea rather than its number: measure whether a method reduces KV memory traffic, or helps prefill more than decode. The mechanism generalizes across GPUs; the headline number does not.',
    lesson: 'research-capstone/reproduce-research',
  },
  'one-page-defense': {
    term: 'One-page defense',
    def: 'The capstone artifact: the design in one defensible sentence, the measured evidence, the derived arithmetic, the alternatives rejected with reasons, and the measurement that would change the author’s mind.',
    lesson: 'research-capstone/capstone',
  },
  'design-doc': {
    term: 'Design document',
    def: 'A written architecture decision: the SLO, the memory and cost budget, the chosen topology, and the alternatives rejected and why. Distinguishes arithmetic and measurement from judgement.',
    lesson: 'research-capstone/capstone',
  },
  'derived-budget': {
    term: 'Derived budget',
    def: 'The memory and cost arithmetic computed before any benchmark: weights plus KV pool plus reserve, giving max batch and cost per million tokens. Exact, cheap, and it eliminates most designs.',
    lesson: 'research-capstone/capstone',
  },
  'little-s-law': {
    term: 'Little’s law',
    def: 'In a steady system, the average number of requests in flight equals the arrival rate multiplied by the average time each request spends. 4,000 users each taking 8 s means about 500 requests in flight.',
    lesson: 'benchmarking/throughput-concurrency',
  },
  'headroom': {
    term: 'Headroom',
    def: 'Spare capacity deliberately left above current demand, so a traffic spike does not immediately break the SLO. Size replicas at roughly 70% of measured capacity, not 100%.',
    lesson: 'production-serving/service-lifecycle',
  },
};