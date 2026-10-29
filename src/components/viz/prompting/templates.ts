// Real token streams for one tiny conversation, produced by the actual tokenizers:
// Llama 3 (tiktoken BPE, 128,000 learned pieces + 256 reserved) and Qwen2.5 (151,643 pieces + specials).
// system: "You are a terse assistant."   user: "What is a GPU?"
// Every count the lesson quotes (26, 24, 72, 15, 13) is the length of one of these arrays.

export type TokKind = 'special' | 'tpl' | 'content' | 'broken';
export interface Tok { key: string; text: string; kind: TokKind; id?: number }

const c = (key: string, text: string): Tok => ({ key, text, kind: 'content' });
const sp = (key: string, text: string, id: number): Tok => ({ key, text, kind: 'special', id });
const tp = (key: string, text: string, id?: number): Tok => ({ key, text, kind: 'tpl', id });
const br = (key: string, text: string): Tok => ({ key, text, kind: 'broken' });

export const SYS_TEXT = 'You are a terse assistant.';
export const USR_TEXT = 'What is a GPU?';
export const ANS_TEXT = 'A chip that does the same math on lots of data at once.';

export const SYS: Tok[] = ['You', ' are', ' a', ' terse', ' assistant', '.'].map((t, i) => c(`s${i}`, t));
export const USR: Tok[] = ['What', ' is', ' a', ' GPU', '?'].map((t, i) => c(`u${i}`, t));
/** An illustrative answer, tokenized for real (14 tokens in both tokenizers). */
export const ANS: string[] = ['A', ' chip', ' that', ' does', ' the', ' same', ' math', ' on', ' lots', ' of', ' data', ' at', ' once', '.'];

/** Strings glued together with a newline: 11 Llama-3 tokens (".\n" is one token, id 627). */
export const RAW: Tok[] = [...SYS.slice(0, 5), c('s5', '.\n'), ...USR];

const SH = 128006; // <|start_header_id|>
const EH = 128007; // <|end_header_id|>
const EOT = 128009; // <|eot_id|>
export const LLAMA: Tok[] = [
  sp('L0', '<|begin_of_text|>', 128000),
  sp('L1', '<|start_header_id|>', SH), tp('L2', 'system', 9125), sp('L3', '<|end_header_id|>', EH), tp('L4', '\n\n', 271),
  ...SYS,
  sp('L5', '<|eot_id|>', EOT),
  sp('L6', '<|start_header_id|>', SH), tp('L7', 'user', 882), sp('L8', '<|end_header_id|>', EH), tp('L9', '\n\n', 271),
  ...USR,
  sp('L10', '<|eot_id|>', EOT),
  sp('L11', '<|start_header_id|>', SH), tp('L12', 'assistant', 78191), sp('L13', '<|end_header_id|>', EH), tp('L14', '\n\n', 271),
];

const IMS = 151644; // <|im_start|>
const IME = 151645; // <|im_end|>
export const CHATML: Tok[] = [
  sp('C0', '<|im_start|>', IMS), tp('C1', 'system', 8948), tp('C2', '\n', 198),
  ...SYS,
  sp('C3', '<|im_end|>', IME), tp('C4', '\n', 198),
  sp('C5', '<|im_start|>', IMS), tp('C6', 'user', 872), tp('C7', '\n', 198),
  ...USR,
  sp('C8', '<|im_end|>', IME), tp('C9', '\n', 198),
  sp('C10', '<|im_start|>', IMS), tp('C11', 'assistant', 77091), tp('C12', '\n', 198),
];

// The Llama-3 prompt string fed to Qwen2.5's tokenizer. Qwen has no such special tokens, so the
// markers are spelled out as ordinary pieces, and they even swallow the "." and the "?".
const HEAD = (role: string, n: number): Tok[] => [
  br(`W${n}`, '><'), br(`W${n + 1}`, '|'), br(`W${n + 2}`, 'start'), br(`W${n + 3}`, '_header'), br(`W${n + 4}`, '_id'), br(`W${n + 5}`, '|'), br(`W${n + 6}`, '>'),
  tp(`W${n + 7}`, role),
  br(`W${n + 8}`, '<'), br(`W${n + 9}`, '|'), br(`W${n + 10}`, 'end'), br(`W${n + 11}`, '_header'), br(`W${n + 12}`, '_id'), br(`W${n + 13}`, '|'), br(`W${n + 14}`, '>\n\n'),
];
const EOT_TEXT = (lead: Tok, n: number): Tok[] => [lead, br(`W${n}`, '|'), br(`W${n + 1}`, 'e'), br(`W${n + 2}`, 'ot'), br(`W${n + 3}`, '_id'), br(`W${n + 4}`, '|')];
export const WRONG: Tok[] = [
  br('W0', '<'), br('W1', '|'), br('W2', 'begin'), br('W3', '_of'), br('W4', '_text'), br('W5', '|'),
  ...HEAD('system', 10),
  ...SYS.slice(0, 5),
  ...EOT_TEXT(br('s5', '.<'), 30),
  ...HEAD('user', 40),
  ...USR.slice(0, 4),
  ...EOT_TEXT(br('u4', '?<'), 60),
  ...HEAD('assistant', 70),
];

export type TemplateId = 'llama' | 'chatml' | 'wrong';
export const TEMPLATES: Record<TemplateId, { name: string; toks: Tok[]; stop?: string; stopId?: number }> = {
  llama: { name: 'Llama 3', toks: LLAMA, stop: '<|eot_id|>', stopId: EOT },
  chatml: { name: 'ChatML (Qwen2.5)', toks: CHATML, stop: '<|im_end|>', stopId: IME },
  wrong: { name: 'Llama-3 markers → Qwen', toks: WRONG },
};

/** How a token is drawn: newlines as a literal \n, a leading space as a middle dot. */
export const show = (t: string) => t.replace(/\n/g, '\\n').replace(/^ /, '·');

/** A multi-turn conversation on Llama 3: 500-token system prompt, 50-token questions, 200-token answers. */
export const TURN = { system: 500, user: 50, answer: 200, bos: 1, block: 5, genPrompt: 4 };
/** Prompt length at turn k (1-based): 565 + 260·(k − 1). */
export const promptAtTurn = (k: number) => {
  const first = TURN.bos + TURN.block + TURN.system + TURN.block + TURN.user + TURN.genPrompt; // 565
  const perTurn = TURN.answer + 1 + TURN.block + TURN.user + TURN.genPrompt; // 260 = answer + eot, then the next user block and header
  return first + perTurn * (k - 1);
};
export const newAtTurn = (k: number) => (k === 1 ? promptAtTurn(1) : promptAtTurn(k) - promptAtTurn(k - 1));
