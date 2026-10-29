import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, fmt } from '../lib/svg';

const N = 6;
const CELL = 20;
const PITCH = 23;

const STEPS = [
  'Encoder: everyone sees everyone',
  'Decoder: only the past',
  'Encoder–decoder: read, then write',
  'Every token teaches, or 15% do',
  'Append one token: what changes?',
  'The cost of writing',
  'One interface: text in, text out',
  'Where encoders still win',
];
const CAPTIONS = [
  'BERT’s mask: every token attends to every other. Ideal for reading a text you are given, all at once.',
  'GPT’s mask: each token sees itself and the past. The future is hatched out. This is the only mask that can both train on every token and generate.',
  'T5 and the original 2017 Transformer: an encoder reads the input with full attention, and a decoder writes the output causally while cross-attending to everything the encoder produced.',
  'Per pass, a causal model is graded at every position. A masked model is graded at about 15%. Drag the length: the ratio stays 1 / 0.15 ≈ 6.7.',
  'Add a token to a decoder and only one new row appears; every earlier row is unchanged, so its keys and values can be cached. Add one to an encoder and every row changes.',
  'Generating with a decoder and a KV cache costs one token-forward per new token. An encoder would have to re-read everything after every token. The gap grows quadratically.',
  'Because a decoder continues text, every task can be phrased as text to continue. One objective, one model, one serving stack, for every task.',
  'For reading jobs with no writing — search embeddings, reranking, classification — a small encoder is still faster, cheaper and often better. Module B3 uses them.',
];

function Grid({ x, y, rows, cols, allow, label, hot, change }: {
  x: number; y: number; rows: number; cols: number; allow: (r: number, c: number) => boolean; label?: string;
  hot?: (r: number, c: number) => 'new' | 'cached' | 'changed' | null; change?: boolean;
}) {
  const cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const ok = allow(r, c);
      const h = hot?.(r, c) ?? null;
      const fill = !ok ? 'url(#tps-hatch)' : h === 'new' ? 'var(--token)' : h === 'cached' ? 'var(--memory-dim)' : h === 'changed' ? 'var(--error-dim)' : 'var(--compute-dim)';
      const stroke = !ok ? 'var(--idle)' : h === 'new' ? 'var(--token)' : h === 'cached' ? 'var(--memory)' : h === 'changed' ? 'var(--error)' : 'var(--compute)';
      cells.push(<rect key={`${r}-${c}`} x={x + c * PITCH} y={y + r * PITCH} width={CELL} height={CELL} rx={2} fill={fill} stroke={stroke} strokeWidth={0.8} className="v-cell" />);
    }
  }
  return (
    <g>
      {label && <Label x={x + (cols * PITCH) / 2} y={y - 10} anchor="middle" tone={1} size={10} bold mono={false}>{label}</Label>}
      {cells}
      {change && <rect x={x - 3} y={y - 3} width={cols * PITCH + 3} height={rows * PITCH + 3} rx={3} fill="none" stroke="var(--error)" strokeDasharray="4 3" />}
    </g>
  );
}

export default function ThreeMasks() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [T, setT] = useState(512);
  const [n, setN] = useState(500);
  const P = 500; // prompt tokens for the cost step

  const mlm = 0.15 * T;
  const dec = P + n;
  const enc = n * P + (n * (n + 1)) / 2;

  // cost chart
  const CX0 = 80;
  const CX1 = 600;
  const CY0 = 270;
  const CY1 = 70;
  const nMax = 2000;
  const encMax = nMax * P + (nMax * (nMax + 1)) / 2;
  const cx = (k: number) => CX0 + (k / nMax) * (CX1 - CX0);
  const cy = (v: number) => CY0 - (v / encMax) * (CY0 - CY1);
  const encPts = Array.from({ length: 41 }, (_, i) => (i / 40) * nMax).map((k) => `${cx(k).toFixed(1)},${cy(k * P + (k * (k + 1)) / 2).toFixed(1)}`).join(' ');
  const decPts = `${cx(0)},${cy(P)} ${cx(nMax)},${cy(P + nMax)}`;

  const readout =
    st === 3
      ? [
          { label: 'sequence length', value: fmt.int(T), kind: 'text' as const },
          { label: 'causal: graded', value: fmt.int(T), kind: 'token' as const, note: 'every position' },
          { label: 'masked: graded', value: mlm.toFixed(1), kind: 'token' as const, note: '15% of positions' },
          { label: 'ratio', value: `${(T / mlm).toFixed(2)}×`, kind: 'text' as const, note: 'per forward pass' },
        ]
      : st === 4
        ? [
            { label: 'decoder: rows recomputed', value: '1', kind: 'token' as const, note: 'the rest come from the KV cache' },
            { label: 'encoder: rows recomputed', value: `${N + 1}`, kind: 'error' as const, note: 'all of them, in every layer' },
          ]
        : st === 5
          ? [
              { label: 'prompt', value: `${P} tokens`, kind: 'text' as const },
              { label: 'decoder + KV cache', value: `${fmt.int(dec)} token-forwards`, kind: 'token' as const, note: 'prompt once, then 1 per token' },
              { label: 'encoder, re-reading', value: `${fmt.int(enc)} token-forwards`, kind: 'error' as const, note: 'whole sequence per new token' },
              { label: 'ratio', value: `${fmt.int(enc / dec)}×`, kind: 'error' as const },
            ]
          : st === 7
            ? [
                { label: 'BERT-Base, 200 tokens', value: `${(2 * 110e6 * 200 / 1e9).toFixed(0)} GFLOP`, kind: 'compute' as const, note: 'one pass, no decoding' },
                { label: '8B decoder, same input', value: `${(2 * 8e9 * 200 / 1e12).toFixed(1)} TFLOP`, kind: 'compute' as const, note: 'prefill alone' },
                { label: 'ratio', value: `${(8e9 / 110e6).toFixed(0)}×`, kind: 'text' as const },
              ]
            : [
                { label: 'mask', value: st === 0 ? 'bidirectional' : st === 1 ? 'causal' : st === 2 ? 'cross-attention' : 'causal', kind: 'compute' as const },
                { label: 'can generate', value: st === 0 ? 'no' : 'yes', kind: (st === 0 ? 'error' : 'token') as 'error' | 'token' },
                { label: 'graded per pass', value: st === 0 ? '~15%' : st === 2 ? 'output tokens' : 'every token', kind: 'token' as const },
              ];

  const MX = 120;
  const MY = 70;

  return (
    <Stage
      rootRef={ref}
      title="Encoder, decoder, encoder–decoder"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['compute', 'attends'], ['idle', 'masked'], ['token', 'new / graded'], ['memory', 'cached'], ['error', 'recomputed']]}
      {...s}
      readout={<Readout items={readout} />}
      controls={st === 3 ? (
        <Slider label="Sequence length (tokens)" value={T} min={16} max={4096} log onChange={setT} format={fmt.int} kind="token" />
      ) : st === 5 ? (
        <Slider label="Tokens to generate after a 500-token prompt" value={n} min={1} max={nMax} step={1} onChange={setN} format={fmt.int} kind="token" />
      ) : undefined}
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="The three attention patterns: bidirectional encoder, causal decoder, and encoder-decoder with cross-attention; how many positions each objective grades per pass; what must be recomputed when one token is appended; the cost of generation with and without a KV cache; and the jobs where encoders still win.">
        {/* ── steps 0-1: single matrices ── */}
        <Fade show={st === 0 || st === 1}>
          <Grid x={MX} y={MY} rows={N} cols={N} allow={(r, c) => st === 0 || c <= r} label={st === 0 ? 'encoder (BERT)' : 'decoder (GPT)'} />
          {['the', 'bank', 'was', 'steep', 'and', 'muddy'].map((w, i) => (
            <Label key={w} x={MX - 8} y={MY + i * PITCH + 14} anchor="end" tone={3} size={9}>{w}</Label>
          ))}
          <Label x={MX + (N * PITCH) / 2} y={MY + N * PITCH + 18} anchor="middle" tone={3} size={9}>row reads column</Label>
        </Fade>
        <Fade show={st === 0}>
          <Label x={330} y={90} tone={1} size={11} bold mono={false}>Bidirectional</Label>
          <Label x={330} y={112} tone={2} size={10} mono={false}>Every token is built from the whole input.</Label>
          <Label x={330} y={128} tone={2} size={10} mono={false}>Trained by the masked-word game (lesson 3).</Label>
          <Label x={330} y={156} tone={2} size={10} mono={false}>Perfect for understanding text you are given.</Label>
          <Label x={330} y={172} tone={2} size={10} mono={false}>No natural way to write new text.</Label>
          <Label x={330} y={204} tone={3} size={9.5} mono={false}>BERT, RoBERTa, and today most search-embedding models.</Label>
        </Fade>
        <Fade show={st === 1}>
          <Label x={330} y={90} tone={1} size={11} bold mono={false}>Causal</Label>
          <Label x={330} y={112} tone={2} size={10} mono={false}>Each token is built from itself and the past.</Label>
          <Label x={330} y={128} tone={2} size={10} mono={false}>Trained by the next-word game (lesson 4).</Label>
          <Label x={330} y={156} tone={2} size={10} mono={false}>Half the matrix is thrown away, and in exchange</Label>
          <Label x={330} y={172} tone={2} size={10} mono={false}>every position can be graded and new text written.</Label>
          <Label x={330} y={204} tone={3} size={9.5} mono={false}>GPT-2, GPT-3, Llama, Qwen, Mistral: nearly every LLM.</Label>
        </Fade>

        {/* ── step 2: encoder-decoder ── */}
        <Fade show={st === 2}>
          <Grid x={60} y={90} rows={4} cols={4} allow={() => true} label="encoder self-attention" />
          <Grid x={250} y={90} rows={4} cols={4} allow={() => true} label="cross-attention" />
          <Grid x={440} y={90} rows={4} cols={4} allow={(r, c) => c <= r} label="decoder self-attention" />
          {['how', 'are', 'you', '?'].map((w, i) => (
            <g key={w}>
              <Label x={52} y={104 + i * PITCH} anchor="end" tone={3} size={9}>{w}</Label>
              <Label x={60 + i * PITCH + 10} y={200} anchor="middle" tone={3} size={8.5}>{w}</Label>
              <Label x={250 + i * PITCH + 10} y={200} anchor="middle" tone={3} size={8.5}>{w}</Label>
            </g>
          ))}
          {['comment', 'allez', 'vous', '?'].map((w, i) => (
            <g key={w}>
              <Label x={244} y={104 + i * PITCH} anchor="end" tone={3} size={8.5}>{w.slice(0, 5)}</Label>
              <Label x={440 + i * PITCH + 10} y={200} anchor="middle" tone={3} size={7.5}>{w.slice(0, 3)}</Label>
            </g>
          ))}
          <Label x={60} y={226} tone={2} size={9.5} mono={false}>input reads all of itself</Label>
          <Label x={250} y={226} tone={2} size={9.5} mono={false}>each output token reads</Label>
          <Label x={250} y={240} tone={2} size={9.5} mono={false}>the whole encoded input</Label>
          <Label x={440} y={226} tone={2} size={9.5} mono={false}>output reads only its past</Label>
          <Label x={60} y={282} tone={1} size={10} mono={false}>Two stacks: read the source both ways, then write the target one token at a time.</Label>
          <Label x={60} y={300} tone={3} size={9.5} mono={false}>The original 2017 Transformer (for translation), T5, Whisper. Only the output tokens are graded.</Label>
        </Fade>

        {/* ── step 3: supervision per pass ── */}
        <Fade show={st === 3}>
          {(['causal (next token)', 'masked (BERT)'] as const).map((name, ri) => {
            const cells = 40;
            return (
              <g key={name}>
                <Label x={40} y={84 + ri * 90} tone={1} size={10.5} bold mono={false}>{name}</Label>
                {Array.from({ length: cells }, (_, i) => {
                  // deterministic 15% pattern: positions 3, 9, 16, 22, 30, 36
                  const graded = ri === 0 || [3, 9, 16, 22, 30, 36].includes(i);
                  return <rect key={i} x={40 + i * 14} y={94 + ri * 90} width={12} height={22} rx={2}
                    fill={graded ? 'var(--token)' : 'var(--ground-3)'} stroke={graded ? 'var(--token)' : 'var(--rule-2)'} />;
                })}
                <Label x={40} y={134 + ri * 90} tone={2} size={9.5}>{ri === 0 ? `${fmt.int(T)} of ${fmt.int(T)} positions graded` : `${mlm.toFixed(1)} of ${fmt.int(T)} positions graded`}</Label>
              </g>
            );
          })}
          <Label x={40} y={278} tone={1} size={10} mono={false}>Same forward pass, same compute: the causal model gets about 6.7× more graded predictions.</Label>
          <Label x={40} y={296} tone={3} size={9.5} mono={false}>Each masked prediction sees both sides, so it is a richer question. But it is a scarcer one.</Label>
        </Fade>

        {/* ── step 4: append one token ── */}
        <Fade show={st === 4}>
          <Grid x={70} y={80} rows={N + 1} cols={N + 1} allow={(r, c) => c <= r} label="decoder: + 1 token"
            hot={(r) => (r === N ? 'new' : 'cached')} />
          <Grid x={380} y={80} rows={N + 1} cols={N + 1} allow={() => true} label="encoder: + 1 token"
            hot={(r) => (r === N ? 'new' : 'changed')} change />
          <Label x={70} y={268} tone={2} size={9.5} mono={false}>Old rows cannot see the new token,</Label>
          <Label x={70} y={282} tone={2} size={9.5} mono={false}>so they are exactly what they were.</Label>
          <Label x={70} y={296} kind="memory" size={9.5} mono={false}>Their K and V are cached (Module 05).</Label>
          <Label x={380} y={268} tone={2} size={9.5} mono={false}>Every old row can see the new token,</Label>
          <Label x={380} y={282} tone={2} size={9.5} mono={false}>so every row, in every layer, changes.</Label>
          <Label x={380} y={296} kind="error" size={9.5} mono={false}>Nothing can be reused: re-read it all.</Label>
        </Fade>

        {/* ── step 5: cost chart ── */}
        <Fade show={st === 5}>
          <line x1={CX0} x2={CX1} y1={CY0} y2={CY0} className="v-rule" />
          <line x1={CX0} x2={CX0} y1={CY0} y2={CY1 - 6} className="v-rule" />
          {[0, 500, 1000, 1500, 2000].map((k) => (
            <g key={k}>
              <line x1={cx(k)} x2={cx(k)} y1={CY0} y2={CY0 + 4} className="v-rule" />
              <Label x={cx(k)} y={CY0 + 16} anchor="middle" tone={3} size={9}>{fmt.int(k)}</Label>
            </g>
          ))}
          {[0, 1e6, 2e6, 3e6].map((v) => (
            <Label key={v} x={CX0 - 6} y={cy(v) + 3} anchor="end" tone={3} size={9}>{v === 0 ? '0' : `${v / 1e6}M`}</Label>
          ))}
          <Label x={(CX0 + CX1) / 2} y={CY0 + 32} anchor="middle" tone={3} size={9.5}>tokens generated</Label>
          <Label x={CX0 + 4} y={CY1 - 12} tone={3} size={9.5}>token-forwards of work</Label>
          <polyline points={encPts} fill="none" stroke="var(--error)" strokeWidth={2} />
          <polyline points={decPts} fill="none" stroke="var(--token)" strokeWidth={2} />
          <Label x={cx(1450)} y={cy(1450 * P + (1450 * 1451) / 2) - 10} anchor="end" kind="error" size={10} bold mono={false}>encoder, re-reading everything</Label>
          <Label x={CX1} y={cy(P + nMax) - 8} anchor="end" kind="token" size={10} bold mono={false}>decoder + KV cache</Label>
          <line x1={cx(n)} x2={cx(n)} y1={CY0} y2={cy(enc)} stroke="var(--text-3)" strokeDasharray="3 3" />
          <circle cx={cx(n)} cy={cy(enc)} r={4.5} fill="var(--error)" />
          <circle cx={cx(n)} cy={cy(dec)} r={4.5} fill="var(--token)" />
        </Fade>

        {/* ── step 6: one interface ── */}
        <Fade show={st === 6}>
          {[
            ['classify', 'Review: “Battery died in an hour.” Sentiment:', ' negative'],
            ['extract', 'Email: “…meet at 3 pm Friday…” Meeting time:', ' 3 pm Friday'],
            ['translate', 'English: see you tomorrow. French:', ' à demain'],
            ['code', 'def add(a, b):', ' return a + b'],
            ['chat', 'User: why is the sky blue? Assistant:', ' Sunlight scatters…'],
          ].map(([task, prompt, out], i) => (
            <g key={task}>
              <Label x={100} y={70 + i * 46} anchor="end" tone={2} size={10} mono={false}>{task}</Label>
              <rect x={110} y={54 + i * 46} width={340} height={26} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x={120} y={71 + i * 46} tone={1} size={9.5}>{prompt}</Label>
              <rect x={454} y={54 + i * 46} width={150} height={26} rx={3} fill="var(--token-dim)" stroke="var(--token)" />
              <Label x={464} y={71 + i * 46} tone={1} size={9.5}>{out}</Label>
            </g>
          ))}
          <Label x={110} y={300} tone={2} size={10} mono={false}>Every row is the same model doing the same thing: continuing text. No head per task.</Label>
          <Label x={110} y={316} tone={3} size={9.5} mono={false}>Requests for all five can share one batch on one GPU. Illustrative prompts and outputs.</Label>
        </Fade>

        {/* ── step 7: where encoders win ── */}
        <Fade show={st === 7}>
          {[
            { y: 50, name: 'search embeddings', how: 'one vector per text, compare by dot product', fwd: 'B3 · Embeddings & vector search' },
            { y: 130, name: 'rerankers', how: 'read query + document together, both ways', fwd: 'B3 · Chunking & reranking' },
            { y: 210, name: 'classifiers', how: 'spam, toxicity, routing, guardrails', fwd: 'B7 · Guardrails' },
          ].map((c) => (
            <g key={c.name}>
              <rect x={40} y={c.y} width={150} height={56} rx={4} fill="var(--compute-dim)" stroke="var(--compute)" />
              <Label x={115} y={c.y + 24} anchor="middle" tone={1} size={10.5} bold mono={false}>small encoder</Label>
              <Label x={115} y={c.y + 40} anchor="middle" tone={3} size={9}>~100M params</Label>
              <Arrow x1={194} y1={c.y + 28} x2={236} y2={c.y + 28} kind="token" />
              <rect x={240} y={c.y} width={364} height={56} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x={254} y={c.y + 20} tone={1} size={10.5} bold mono={false}>{c.name}</Label>
              <Label x={254} y={c.y + 36} tone={2} size={9.5} mono={false}>{c.how}</Label>
              <Label x={594} y={c.y + 20} anchor="end" tone={3} size={8.5} mono={false}>{c.fwd}</Label>
            </g>
          ))}
          <Label x={40} y={300} tone={2} size={10} mono={false}>No writing needed, both sides help, and it runs in one pass at a fraction of the cost.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
