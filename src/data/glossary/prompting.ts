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
};
