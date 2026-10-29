import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  agent: {
    term: 'Agent',
    def: 'An autonomous system where a language model acts as the central cognitive controller inside a closed loop: it ingests environment state, reasons over goals, emits structured actions or tool calls, observes real-world feedback, and terminates dynamically.',
    lesson: 'agents/what-is-an-agent',
  },
  'compounding-errors': {
    term: 'Compounding errors',
    def: 'The exponential decay of overall task success probability over multi-step trajectories under independent step success: P = p^k. A 95% reliable step results in only 59.9% overall success over 10 steps and 35.9% over 20 steps.',
    lesson: 'agents/what-is-an-agent',
  },
  'perceive-plan-act-observe': {
    term: 'Perceive-Plan-Act-Observe loop',
    def: 'The canonical four-phase agent control cycle: perceive current environment state, plan next logical sub-goals, act via tool invocation, and observe external feedback before iterating.',
    lesson: 'agents/what-is-an-agent',
  },
  'blast-radius': {
    term: 'Blast radius',
    def: 'The maximum potential damage an erroneous action can inflict on external state. Read tools have zero persistent blast radius; write mutations (e.g. database deletions or API terminations) have critical blast radius and require strict gating.',
    lesson: 'agents/tools',
  },
  'human-in-the-loop': {
    term: 'Human-in-the-loop (HITL)',
    def: 'A defensive checkpoint pattern where the agent pauses execution before performing state-mutating write actions, presents a structured dry-run diff, and awaits explicit authorization from a human operator.',
    lesson: 'agents/tools',
  },
  'constrained-decoding': {
    term: 'Constrained decoding (grammar masking)',
    def: 'Technique that uses a finite state automaton derived from a JSON Schema or regex to mask invalid vocabulary logits to -infinity at each decode step, guaranteeing 100% syntactically valid tool call generation.',
    lesson: 'agents/tools',
  },
  'model-context-protocol': {
    term: 'Model Context Protocol (MCP)',
    def: 'An open client-server standard created by Anthropic that decouples LLM applications (hosts and clients) from external tools and data sources (servers) using standardized JSON-RPC 2.0 messages.',
    lesson: 'agents/tools',
  },
  'react-framework': {
    term: 'ReAct (Reasoning + Acting)',
    def: 'An agent prompting and execution framework that interleaves explicit Thought generation, Action tool invocation, and Observation ingestion, enabling dynamic adaptation to environment feedback.',
    lesson: 'agents/planning',
  },
  'plan-then-execute': {
    term: 'Plan-then-Execute',
    def: 'An agent architecture that decouples planning from execution: a high-capacity planner model first decomposes the goal into a Directed Acyclic Graph (DAG) of subtasks, and independent subtasks are executed concurrently by parallel workers.',
    lesson: 'agents/planning',
  },
  reflexion: {
    term: 'Reflexion',
    def: 'An agent framework that implements verbal reinforcement learning without weight updates: when a trial fails evaluation, a self-reflection pass diagnoses the mistake and stores a text critique in an episodic memory buffer for subsequent trials.',
    lesson: 'agents/planning',
  },
  'context-compaction': {
    term: 'Context compaction (hybrid buffer)',
    def: 'A memory management pattern that permanently anchors the system prompt and goal at the head, compacts older intermediate turns into a dense structured summary in the middle, and preserves raw recent turns at the tail to prevent context overflow.',
    lesson: 'agents/memory',
  },
  'loop-detection': {
    term: 'Loop detection (state hashing)',
    def: 'A defensive guard that computes a cryptographic hash of (tool_name, argument_json). If identical hashes occur consecutively beyond a threshold (typically 3), an automatic circuit breaker halts execution and forces a strategy shift.',
    lesson: 'agents/agent-failures',
  },
  'cost-per-solved-task': {
    term: 'Cost per Solved Task',
    def: 'The total financial cost of all agent runs divided by the number of successfully resolved tasks: C_solved = C_run / Success_Rate. Accounts for wasted token spend on failed attempts.',
    lesson: 'agents/agent-failures',
  },
  'swe-bench': {
    term: 'SWE-bench',
    def: 'An industry benchmark evaluating agents on resolving real-world GitHub issues across large Python repositories by executing official repository unit test suites against generated code patches.',
    lesson: 'agents/agent-failures',
  },
};
