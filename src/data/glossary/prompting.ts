import type { GlossaryEntry } from '../glossary';

// Terms Module B1 (Prompt Engineering) introduces. Keys already defined elsewhere
// (chat-template, special-tokens, eos, prefix-caching, lost-in-the-middle, reasoning-trace, ...)
// are reused, not repeated.
export const terms: Record<string, GlossaryEntry> = {
  'system-prompt': {
    term: 'System prompt',
    def: 'The message with role "system" that sets how the model should behave. After templating it is simply the first block of tokens in the stream: followed because finetuning rewarded it, not enforced by anything, and cacheable because it is identical on every request.',
    lesson: 'prompting/chat-templates',
  },
  'generation-prompt': {
    term: 'Generation prompt',
    def: 'The empty assistant header a chat template appends at the end of the prompt (add_generation_prompt=True), e.g. <|start_header_id|>assistant<|end_header_id|> and two newlines. It tells the model that it is now its turn to answer.',
    lesson: 'prompting/chat-templates',
  },
  chatml: {
    term: 'ChatML',
    def: 'A chat layout used by Qwen and others: each turn is <|im_start|>role, a newline, the content, then <|im_end|>. Five template tokens per message, three for the generation prompt.',
    lesson: 'prompting/chat-templates',
  },
  'recency-bias': {
    term: 'Recency bias (in few-shot prompts)',
    def: 'The tendency of a model to favour the labels of the examples nearest the end of the prompt. With majority-label bias (favouring the most frequent label) it is why few-shot examples should be balanced and interleaved.',
    lesson: 'prompting/in-context',
  },
  'dynamic-few-shot': {
    term: 'Dynamic few-shot',
    def: 'Choosing the examples per request, usually by retrieving the labelled inputs most similar to the new one. More relevant than a fixed set, but the examples change every call, so they cannot sit in a cached prefix.',
    lesson: 'prompting/in-context',
  },
  'many-shot': {
    term: 'Many-shot prompting',
    def: 'Putting hundreds or thousands of examples in a long context window. Some tasks keep improving, but 500 examples of 200 tokens is 100,000 input tokens on every call, which is usually the point where finetuning is cheaper.',
    lesson: 'prompting/in-context',
  },
  'prompt-chain': {
    term: 'Prompt chain (decomposition)',
    def: 'Splitting one job across several model calls, each with a small focused prompt, such as a router that outputs a label and a specialised prompt per label. End-to-end accuracy is the product of the stages; it costs extra round trips but sends only the context each request needs.',
    lesson: 'prompting/writing-prompts',
  },
  'test-time-compute': {
    term: 'Test-time compute',
    def: 'Computation spent at inference to get a better answer: writing reasoning tokens, sampling several answers and voting, or searching with a verifier. Paid in output tokens, so in decode time and money.',
    lesson: 'prompting/chain-of-thought',
  },
  'chain-of-thought': {
    term: 'Chain of thought (CoT)',
    def: 'Having the model write its intermediate reasoning before the final answer, via worked examples (few-shot) or an instruction such as "Let\u2019s think step by step" (zero-shot). Each written step is another forward pass the answer can build on.',
    lesson: 'prompting/chain-of-thought',
  },
  'self-consistency': {
    term: 'Self-consistency',
    def: 'Sample several chain-of-thought answers at a temperature above zero and return the most common final answer. Right paths tend to agree while wrong ones scatter; it costs N times the output tokens but, batched, little extra latency.',
    lesson: 'prompting/chain-of-thought',
  },
  verifier: {
    term: 'Verifier',
    def: 'A model or program that judges whether a candidate answer is correct, used to pick the best of N samples. Programmatic checks (unit tests, recomputation) cannot be fooled; learned verifiers can, and large N finds their mistakes.',
    lesson: 'prompting/chain-of-thought',
  },
  'reasoning-model': {
    term: 'Reasoning model',
    def: 'A model trained (typically with reinforcement learning on verifiable answers) to write a long chain of thought before answering, such as OpenAI o1 or DeepSeek-R1. Often thousands of output tokens per answer; prompt it zero-shot and set a thinking budget.',
    lesson: 'prompting/chain-of-thought',
  },
  'needle-in-a-haystack': {
    term: 'Needle in a haystack',
    def: 'A long-context test: hide one out-of-place sentence at a chosen depth of filler text of a chosen length, ask for it back, and plot recall over depth and length. Useful, but it checks one literal fact with no distractors, so passing it is necessary rather than sufficient.',
    lesson: 'prompting/context-position',
  },
  'effective-context-length': {
    term: 'Effective context length',
    def: 'The prompt length up to which a model still performs acceptably on a given task, as opposed to the advertised window it can hold. Harder long-context benchmarks often find it far below the advertised number; measure it on your own task.',
    lesson: 'prompting/context-position',
  },
  'eval-set': {
    term: 'Eval set',
    def: 'A fixed list of real inputs with known-good outputs or checks, run against every prompt or model version so that each change gets a comparable score. Grow it with every production failure.',
    lesson: 'prompting/prompt-ops',
  },
  'prompt-catalog': {
    term: 'Prompt catalog',
    def: 'A registry of a team\u2019s prompts: what each is for, its owner, live version, pinned model, current eval score and the services that call it. Enables reuse and review instead of many near-duplicate prompts.',
    lesson: 'prompting/prompt-ops',
  },
  'prompt-optimizer': {
    term: 'Automatic prompt optimizer',
    def: 'Software that searches for better prompts against a metric, for example by having a model propose instructions (APE, OPRO) or by bootstrapping few-shot demonstrations and scoring candidate programs (DSPy). Each compile costs many model calls, and the winner is often a longer prompt.',
    lesson: 'prompting/prompt-ops',
  },
};
