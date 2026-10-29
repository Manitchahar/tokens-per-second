import { useEffect, useRef, useState } from 'react';
import { Stage, Segmented, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Move, fmt } from '../lib/svg';
import {
  SYS, USR, ANS, RAW, TEMPLATES, SYS_TEXT, USR_TEXT, ANS_TEXT, show, promptAtTurn, newAtTurn,
  type Tok, type TemplateId,
} from './templates';

// A list of messages becomes one token stream. Every chip is a real token from the real tokenizer.

const STEPS = [
  'What you send: a list of messages',
  'Glue the strings together, and watch it fail',
  'Roles become special tokens',
  'The generation prompt hands over the turn',
  'Another family, another dialect: ChatML',
  'The wrong template',
  'The system prompt is just the first tokens',
  'Every turn resends the whole conversation',
];
const CAPTIONS = [
  'An API call carries a list of messages, each with a role. But a transformer is a function of one sequence of integers. Somewhere between your JSON and the GPU the list has to become a stream.',
  'The obvious approach: join the strings. 11 tokens, and every boundary is gone. Nothing says which part is an instruction, where the question ends, or where an answer should begin.',
  'The Llama 3 template wraps each message in header tokens and closes it with <|eot_id|>. 26 tokens: 11 of yours and 15 added by the template.',
  'add_generation_prompt appends the header of an empty assistant turn. The model writes into it and ends with <|eot_id|>, the token finetuning taught it to stop on.',
  'Qwen2.5 uses ChatML: <|im_start|>role, newline, content, <|im_end|>. Same idea, different spelling: 24 tokens.',
  'Send Llama 3 markers to Qwen and its tokenizer has no such special tokens, so it spells them out in ordinary pieces: 72 tokens, none of them control tokens. No error is raised. The model just sees strange text.',
  'The system prompt is the first block of every request, identical each time. An engine can keep its KV cache and skip recomputing it: prefix caching.',
  'Chat APIs are stateless: every turn resends the whole history. With a cache, only the newest part is new work.',
];

const DEFAULT_TPL: TemplateId[] = ['llama', 'llama', 'llama', 'llama', 'chatml', 'wrong', 'llama', 'llama'];

const CH = 17; // chip height
const PITCH = 24; // row pitch in the stream
const X0 = 16;
const X1 = 624;
const STREAM_Y = 128;
const chipW = (t: string) => Math.max(13, show(t).length * 5.25 + 7);

type Place = { x: number; y: number; w: number };

function flow(toks: { key: string; text: string }[], x0: number, x1: number, y0: number, pitch: number) {
  const out: Record<string, Place> = {};
  let x = x0;
  let y = y0;
  for (const t of toks) {
    const w = chipW(t.text);
    if (x + w > x1) { x = x0; y += pitch; }
    out[t.key] = { x, y, w };
    x += w + 3;
  }
  return { at: out, bottom: y + CH };
}

type DrawKind = Tok['kind'] | 'gen' | 'genstop' | 'ghost';

function TokChip({ text, kind, w, cached }: { text: string; kind: DrawKind; w: number; cached?: boolean }) {
  const style: Record<DrawKind, { fill: string; stroke: string; ink: string; dash?: string }> = {
    content: { fill: 'var(--memory)', stroke: 'var(--memory)', ink: '#0e1116' },
    tpl: { fill: 'var(--memory-dim)', stroke: 'var(--memory)', ink: 'var(--text)' },
    special: { fill: 'var(--text)', stroke: 'var(--text)', ink: '#0e1116' },
    broken: { fill: 'var(--error)', stroke: 'var(--error)', ink: '#0e1116' },
    gen: { fill: 'var(--token)', stroke: 'var(--token)', ink: '#0e1116' },
    genstop: { fill: 'var(--token-dim)', stroke: 'var(--token)', ink: 'var(--text)' },
    ghost: { fill: 'none', stroke: 'var(--rule-2)', ink: 'var(--text-3)', dash: '3 3' },
  };
  const s = style[kind];
  return (
    <g>
      <rect x={0} y={0} width={w} height={CH} rx={2.5} fill={s.fill} stroke={s.stroke} strokeWidth={kind === 'special' ? 1.2 : 0.9} strokeDasharray={s.dash} className="v-cell" />
      <text x={w / 2} y={CH / 2 + 3.1} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fontWeight: 600, fill: s.ink }}>{show(text)}</text>
      {cached && <rect x={0} y={CH + 2} width={w} height={3} rx={1} fill="var(--memory)" className="v-fade" />}
    </g>
  );
}

export default function ChatTemplate() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [tpl, setTpl] = useState<TemplateId>('llama');
  const [turn, setTurn] = useState(10);

  useEffect(() => { setTpl(DEFAULT_TPL[st]); }, [st]);

  const T = TEMPLATES[tpl];
  const mode: 'cards' | 'raw' | TemplateId = st === 0 ? 'cards' : st === 1 ? 'raw' : tpl;
  const stream: Tok[] = mode === 'cards' ? [] : mode === 'raw' ? RAW : T.toks;
  const generating = st >= 3 && (mode === 'llama' || mode === 'chatml');
  const ghosting = st === 1 || (st >= 3 && mode === 'wrong');

  // What follows the prompt: a real answer and stop token, or "?" ghosts when the model is off-distribution.
  const gen: { key: string; text: string; kind: DrawKind }[] = generating
    ? [...ANS.map((t, i) => ({ key: `g${i}`, text: t, kind: 'gen' as DrawKind })), { key: 'gstop', text: T.stop ?? '', kind: 'genstop' as DrawKind }]
    : ghosting ? ['?', '?', '?', '?', '?'].map((t, i) => ({ key: `gx${i}`, text: t, kind: 'ghost' as DrawKind })) : [];

  const laid = flow([...stream, ...gen], X0, X1, STREAM_Y, PITCH);
  // Each template's own chips always keep their own place, so switching templates fades them in place
  // while your content tokens glide between layouts.
  const homes = {
    L: flow(TEMPLATES.llama.toks, X0, X1, STREAM_Y, PITCH).at,
    C: flow(TEMPLATES.chatml.toks, X0, X1, STREAM_Y, PITCH).at,
    W: flow(TEMPLATES.wrong.toks, X0, X1, STREAM_Y, PITCH).at,
  };

  // Card layout for step 1: the content tokens sit inside their message cards.
  const sysCard = flow(SYS, 24, 228, 52, 21).at;
  const usrCard = flow(USR, 254, 420, 52, 21).at;
  const inStream = new Set(stream.map((t) => t.key));

  // Every chip that can ever appear, so React keeps each one and CSS can glide it.
  const allTpl = [...TEMPLATES.llama.toks, ...TEMPLATES.chatml.toks, ...TEMPLATES.wrong.toks].filter((t) => t.kind !== 'content' && !t.key.startsWith('s') && !t.key.startsWith('u'));
  const uniqTpl = allTpl.filter((t, i) => allTpl.findIndex((u) => u.key === t.key) === i);

  const prefixKeys = new Set<string>();
  if (st === 6 && mode !== 'wrong') {
    const toks = T.toks;
    const end = toks.findIndex((t) => t.text === (mode === 'llama' ? '<|eot_id|>' : '<|im_end|>'));
    toks.slice(0, end + (mode === 'chatml' ? 2 : 1)).forEach((t) => prefixKeys.add(t.key));
  }

  const nPrompt = stream.length;
  const nContent = stream.filter((t) => t.kind === 'content').length;
  const nSpecial = stream.filter((t) => t.kind === 'special').length;

  const contentChip = (t: Tok) => {
    const inS = inStream.has(t.key);
    const inStreamTok = stream.find((u) => u.key === t.key);
    const p = inS ? laid.at[t.key] : (t.key.startsWith('s') ? sysCard[t.key] : usrCard[t.key]);
    const shownText = inStreamTok ? inStreamTok.text : t.text;
    const kind: DrawKind = inStreamTok ? inStreamTok.kind : 'content';
    const visible = st === 0 || inS;
    return (
      <Move key={t.key} x={p.x} y={p.y} opacity={visible ? 1 : 0}>
        <TokChip text={shownText} kind={kind} w={chipW(shownText)} cached={prefixKeys.has(t.key)} />
      </Move>
    );
  };

  const stepNote = (): [string, string, string] => {
    switch (st) {
      case 0: return [
        'An API call is a list of messages. A transformer reads one sequence of integers.',
        'There is no "list" input and no "role" input anywhere in the network you built in Module 03.',
        'So between your JSON and the GPU, the list must become a single stream. How?',
      ];
      case 1: return [
        '11 tokens, and every boundary is gone.',
        'Which tokens are instructions? Where does the question end? Where should the answer start?',
        'The model sees a document and continues it, perhaps with more questions. Nothing tells it to stop.',
      ];
      case 2: return [
        `${nPrompt} tokens: ${nContent} of yours, ${nPrompt - nContent} added by the template.`,
        mode === 'chatml' ? '<|im_start|> 151644 · <|im_end|> 151645: IDs past Qwen’s 151,643 learned pieces.'
          : mode === 'wrong' ? 'No control tokens at all: every marker was spelled out as ordinary text.'
          : '<|begin_of_text|> 128000 · <|start_header_id|> 128006 · <|end_header_id|> 128007 · <|eot_id|> 128009',
        'Reserved IDs past the learned pieces: they mark structure, and never occur in ordinary text.',
      ];
      case 3: return [
        mode === 'wrong' ? 'The model was never trained on this layout, so what follows is anyone’s guess.' : 'The last tokens open an empty assistant turn. The model writes into it.',
        mode === 'chatml' ? 'add_generation_prompt=True appends <|im_start|>assistant⏎ (3 tokens).' : 'add_generation_prompt=True appends <|start_header_id|>assistant<|end_header_id|>⏎⏎ (4 tokens).',
        `It stops by emitting ${T.stop ?? 'a stop token'}: the token finetuning taught it to end every turn with (Module 3C).`,
      ];
      case 4: return [
        `Qwen2.5 speaks ChatML: ${nPrompt} tokens, ${nPrompt - nContent} of them template.`,
        '<|im_start|> 151644 opens a turn, <|im_end|> 151645 closes it: IDs past Qwen’s 151,643 learned pieces.',
        'Same idea, different spelling. A model understands only the dialect it was finetuned on.',
      ];
      case 5: return [
        mode === 'wrong' ? `Llama 3 markers sent to Qwen: ${nPrompt} tokens, ${nSpecial} control tokens.` : 'Switch to the wrong template to see the failure.',
        'Qwen has no <|start_header_id|>, so its tokenizer spells it out: <, |, start, _header, _id, |, >.',
        'They even swallow your "." and "?". No error is raised: the model just sees strange text.',
      ];
      case 6: return [
        mode === 'wrong' ? 'A broken prefix is still a prefix, but nothing about it helps the model.' : 'The underlined block opens every request, token for token.',
        'Identical first tokens → identical keys and values. An engine can keep them (prefix caching, Module 10).',
        'A 1,500-token system prompt on Llama-3.1-8B: 2 × 8.03B × 1,500 ≈ 24 TFLOP ≈ 44 ms of H100 prefill.',
      ];
      default: return ['', '', ''];
    }
  };
  const [n1, n2, n3] = stepNote();

  const promptK = promptAtTurn(turn);
  const newK = newAtTurn(turn);
  const noCache = Array.from({ length: turn }, (_, i) => promptAtTurn(i + 1)).reduce((a, b) => a + b, 0);
  const withCache = promptAtTurn(turn);

  const readout = st === 7
    ? [
        { label: `prompt at turn ${turn}`, value: fmt.int(promptK), kind: 'memory' as const, note: '565 + 260 × (turn − 1)' },
        { label: 'new this turn', value: fmt.int(newK), kind: 'compute' as const, note: turn === 1 ? 'everything is new' : 'last answer + new question' },
        { label: 'seen before', value: `${(((promptK - newK) / promptK) * 100).toFixed(1)}%`, kind: 'memory' as const, note: 'reusable from a prefix cache' },
      ]
    : [
        { label: 'prompt tokens', value: st === 0 ? '—' : `${nPrompt}`, kind: 'memory' as const, note: st === 0 ? 'nothing tokenized yet' : `${nContent} yours · ${nPrompt - nContent} added` },
        { label: 'control tokens', value: st === 0 ? '—' : `${nSpecial}`, kind: 'text' as const, note: mode === 'llama' ? 'IDs 128000 and up' : mode === 'chatml' ? 'IDs 151643 and up' : mode === 'wrong' ? 'the markers became text' : 'no markers at all' },
        { label: 'stops on', value: generating ? (T.stop ?? '—') : '—', kind: 'token' as const, note: generating ? `ID ${T.stopId}` : 'nothing it was trained to stop on' },
      ];

  return (
    <Stage
      rootRef={ref}
      title="From a list of messages to one token stream"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['memory', 'your content'], ['text', 'control tokens'], ['token', 'generated'], ['error', 'broken markers'], ['compute', 'new prefill work']]}
      {...s}
      readout={<Readout items={readout} />}
      controls={
        st >= 2 && st <= 6 ? (
          <Segmented label="Template applied" value={tpl} options={[
            { value: 'llama' as const, label: 'Llama 3' },
            { value: 'chatml' as const, label: 'ChatML (Qwen2.5)' },
            { value: 'wrong' as const, label: 'Wrong: Llama-3 markers → Qwen' },
          ]} onChange={setTpl} />
        ) : st === 7 ? (
          <Slider label="Conversation turn" value={turn} min={1} max={10} onChange={setTurn} format={(v) => `turn ${v}`} kind="memory" />
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Three chat messages, system, user and assistant, are turned into a single row of tokens. The template inserts control tokens around each message; the wrong template shatters them into ordinary text pieces; the system prompt is the shared first block of every request.">
        {/* ── the messages, as the API sees them ── */}
        <Label x={16} y={18} tone={2} size={10} bold>what you send</Label>
        <Label x={624} y={18} anchor="end" tone={3} size={9}>{'messages = [ {role, content}, … ]'}</Label>
        {[
          { x: 16, w: 220, role: 'system', text: SYS_TEXT },
          { x: 246, w: 182, role: 'user', text: USR_TEXT },
          { x: 438, w: 186, role: 'assistant', text: '' },
        ].map((m) => (
          <g key={m.role}>
            <rect x={m.x} y={26} width={m.w} height={56} rx={4} fill="var(--ground-3)" stroke={m.role === 'assistant' && generating ? 'var(--token)' : 'var(--rule-2)'} strokeDasharray={m.role === 'assistant' && !generating ? '4 3' : undefined} className="v-cell" />
            <text x={m.x + 8} y={41} className="v-label t3" style={{ fontSize: 8.5, letterSpacing: '0.04em' }}>{`role: "${m.role}"`}</text>
            {m.role !== 'assistant' && (
              <Fade show={st > 0}>
                <text x={m.x + 8} y={64} className="v-label t3" style={{ fontSize: 9 }}>{`"${m.text}"`}</text>
              </Fade>
            )}
            {m.role === 'assistant' && (
              <>
                <Fade show={!generating}><text x={m.x + 8} y={64} className="v-label t3" style={{ fontSize: 9 }}>the model writes this part</text></Fade>
                <Fade show={generating}>
                  <text x={m.x + 8} y={60} className="v-label" style={{ fontSize: 8.5, fill: 'var(--token)' }}>{ANS_TEXT.slice(0, 30)}</text>
                  <text x={m.x + 8} y={73} className="v-label" style={{ fontSize: 8.5, fill: 'var(--token)' }}>{ANS_TEXT.slice(30)}</text>
                </Fade>
              </>
            )}
          </g>
        ))}

        {/* ── the stream ── */}
        <line x1={16} x2={624} y1={98} y2={98} className="v-rule faint" />
        <Label x={16} y={116} tone={2} size={10} bold>{st === 0 ? 'what the model sees: ?' : 'what the model sees: one stream'}</Label>
        <Label x={624} y={116} anchor="end" tone={3} size={9}>
          {st === 0 ? 'one sequence of token IDs' : `${nPrompt} prompt tokens${gen.length && generating ? ` + ${gen.length} generated` : ''}`}
        </Label>
        <Fade show={st === 0}>
          <rect x={16} y={STREAM_Y} width={608} height={CH + 8} rx={3} fill="none" stroke="var(--rule-2)" strokeDasharray="4 4" />
          <text x={320} y={STREAM_Y + 16} textAnchor="middle" className="v-label t3" style={{ fontSize: 9.5 }}>a single row of integers, nothing else</text>
        </Fade>

        {uniqTpl.map((t) => {
          const p = homes[t.key[0] as 'L' | 'C' | 'W'][t.key];
          const on = inStream.has(t.key);
          return (
            <Move key={t.key} x={p.x} y={p.y} opacity={on ? 1 : 0}>
              <TokChip text={t.text} kind={t.kind} w={chipW(t.text)} cached={prefixKeys.has(t.key)} />
            </Move>
          );
        })}
        {[...SYS, ...USR].map(contentChip)}
        {[...ANS.map((_, i) => `g${i}`), 'gstop', 'gx0', 'gx1', 'gx2', 'gx3', 'gx4'].map((k) => {
          const g = gen.find((x) => x.key === k);
          const p = laid.at[k];
          return (
            <Move key={k} x={p ? p.x : 480} y={p ? p.y : 60} opacity={g ? 1 : 0}>
              <TokChip text={g ? g.text : '?'} kind={g ? g.kind : 'ghost'} w={chipW(g ? g.text : '?')} />
            </Move>
          );
        })}
        <Fade show={st === 6 && mode !== 'wrong'}>
          <Label x={624} y={laid.bottom + 18} anchor="end" kind="memory" size={9}>underlined: the same on every request, so cacheable</Label>
        </Fade>
        <Fade show={ghosting && st > 0}>
          <Label x={624} y={laid.bottom + 16} anchor="end" kind="error" size={9}>{st === 1 ? 'continues the document? asks more questions? never stops?' : 'off-distribution: may echo markers, ramble, or never stop'}</Label>
        </Fade>

        {/* ── the explanation panel ── */}
        <line x1={16} x2={624} y1={274} y2={274} className="v-rule faint" />
        {st < 7 && (
          <g>
            <Label x={16} y={296} tone={1} size={10} bold>{n1}</Label>
            <Label x={16} y={318} tone={2} size={9}>{n2}</Label>
            <Label x={16} y={338} tone={2} size={9}>{n3}</Label>
            {st === 6 && <Label x={16} y={360} tone={3} size={9}>{'Put anything that changes per request (a date, a user name) after the static part, or it breaks the match.'}</Label>}
            {st === 5 && <Label x={16} y={360} tone={3} size={9}>{'The same silent failure: no template at all, a missing generation prompt, or a second <|begin_of_text|>.'}</Label>}
          </g>
        )}

        {/* ── multi-turn: the whole history again, every turn ── */}
        <Fade show={st === 7}>
          <g>
            <Label x={16} y={292} tone={1} size={10} bold>prompt tokens sent at each turn</Label>
            <Label x={624} y={292} anchor="end" tone={2} size={9}>{`turns 1–${turn}: ${fmt.int(noCache)} tokens prefilled without a cache, ${fmt.int(withCache)} with one`}</Label>
            {Array.from({ length: 10 }, (_, i) => {
              const k = i + 1;
              const total = promptAtTurn(k);
              const fresh = newAtTurn(k);
              const H = 44;
              const h = (total / promptAtTurn(10)) * H;
              const hNew = (fresh / promptAtTurn(10)) * H;
              const x = 40 + i * 58;
              const base = 360;
              const active = k === turn;
              const live = k <= turn;
              return (
                <g key={k} style={{ opacity: live ? 1 : 0.3 }} className="v-fade">
                  <rect x={x} y={base - h} width={40} height={h - hNew} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} className="v-cell" />
                  <rect x={x} y={base - hNew} width={40} height={hNew} fill="var(--compute)" stroke="var(--compute)" strokeWidth={0.8} className="v-cell" />
                  {active && <rect x={x - 3} y={base - h - 3} width={46} height={h + 6} rx={2} fill="none" stroke="var(--text)" strokeWidth={1.2} />}
                  <text x={x + 20} y={base - h - 6} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{fmt.int(total)}</text>
                  <text x={x + 20} y={374} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{`t${k}`}</text>
                </g>
              );
            })}
            <Label x={16} y={332} tone={3} size={8.5} kind="memory">seen</Label>
            <Label x={16} y={358} tone={3} size={8.5} kind="compute">new</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
