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
};
