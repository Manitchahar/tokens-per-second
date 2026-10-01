import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade } from '../lib/svg';

// A toy vocabulary, built from one sentence at three merge levels.
// Real tokenizers have 128,256 entries learned from a corpus; the mechanics are identical.
const SENTENCE = 'the transformer reads tokens';
const LEVELS: string[][] = [
  SENTENCE.split(''),
  ['the', ' ', 'trans', 'form', 'er', ' ', 'read', 's', ' ', 'tok', 'ens'],
  ['the', ' ', 'transformer', ' ', 'reads', ' ', 'tokens'],
];
const LEVEL_NAMES = ['characters only', 'short pieces', 'learned words'];

const VOCAB: string[] = [];
LEVELS.forEach((lv) => lv.forEach((p) => { if (!VOCAB.includes(p)) VOCAB.push(p); }));
const idOf = (p: string) => VOCAB.indexOf(p);

const STEPS = [
  'Characters, not words',
  'Merge them into pieces',
  'Every piece is an integer',
  'How aggressive the merges are',
  'Nothing is ever lost',
  'A token is not free',
];
const CAPTIONS = [
  'A model reads numbers, not letters. This 28-character sentence starts life as 28 separate items — far more than a GPU wants to process for one short thought.',
  'A tokenizer merges neighbouring characters into pieces that appear often in real text: the, transformer, tokens. The sentence collapses from 28 items to 7.',
  'Each piece gets an integer ID, and that integer is the only thing the model ever sees. The ID is just a row number in a table — the embedding table that the next lesson opens.',
  'How aggressively a tokenizer merges is a design choice, and it trades memory against meaning. Drag the slider: at level 2 English runs about 4 characters per token.',
  'Every tokenizer can fall back to single bytes, one token each for all 256 possible byte values. That is why a real vocabulary is a multiple of 256 and why no input is ever rejected as unknown.',
  'A token is a unit of work: one embedding row to read, one row in every attention score matrix, one entry in the KV cache at every layer. Prompt length is counted in these, not in characters.',
];

const CW = 14; // px per character
const PAD = 12;
const GAP = 4;

export default function TokenizerViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [level, setLevel] = useState(0);

  useEffect(() => {
    if (st >= 1 && level === 0) setLevel(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [st]);

  const lv = st === 0 ? 0 : level;
  const pieces = LEVELS[lv];
  const chars = SENTENCE.length;

  // lay the pieces out in one or two rows
  let x = 30;
  let row = 0;
  const placed = pieces.map((p) => {
    const w = p.length * CW + PAD;
    if (x + w > 618) { row += 1; x = 30; }
    const at = { p, x, y: 44 + row * 34, w };
    x += w + GAP;
    return at;
  });
  const rows = row + 1;

  const ids = [...new Set(pieces.map(idOf))].sort((a, b) => a - b);
  const cpt = chars / pieces.length;

  const gridX = 28;
  const gridY = 158;
  const cw = 94;
  const ch = 26;

  return (
    <Stage
      rootRef={ref}
      title="Text becomes tokens: pieces, then integers"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'memory', ['text', 'raw characters']]}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'characters', value: `${chars}`, kind: 'text' as const, note: 'in the sentence' },
            { label: 'tokens', value: `${pieces.length}`, kind: 'token' as const, note: LEVEL_NAMES[lv] },
            { label: 'characters per token', value: cpt.toFixed(2), kind: 'text' as const, note: 'English averages about 4' },
            ...(st >= 2
              ? [{ label: 'IDs touched', value: `${ids.length} of ${VOCAB.length}`, kind: 'memory' as const, note: 'rows of the embedding table' }]
              : []),
          ]}
        />
      }
      controls={
        st >= 1 ? (
          <Slider
            label="Merge level — how long the pieces are"
            value={lv}
            min={0}
            max={2}
            step={1}
            onChange={setLevel}
            format={() => LEVEL_NAMES[lv]}
            kind="token"
          />
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 360" role="img" aria-label="A sentence of 28 characters merged into 7 word-like tokens, each token labelled with the integer ID that identifies its row in the embedding table.">
        <text x={30} y={26} className="v-label t2" style={{ fontSize: 11 }}>{`"${SENTENCE}"`}</text>
        <Label x={612} y={26} anchor="end" tone={3} size={9.5}>{`${chars} characters`}</Label>

        {placed.map((it, i) => (
          <g key={i}>
            <Fade show={st >= 2}>
              <text x={it.x + it.w / 2} y={it.y - 4} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{idOf(it.p)}</text>
            </Fade>
            <rect x={it.x} y={it.y} width={it.w} height={28} rx={3}
              fill={st === 0 ? 'var(--ground-3)' : 'var(--token)'}
              stroke={st === 0 ? 'var(--rule-2)' : 'var(--token)'}
              strokeWidth={st === 0 ? 1 : 1} className="v-cell" />
            <text x={it.x + it.w / 2} y={it.y + 18} textAnchor="middle" className="v-label"
              style={{ fontSize: it.p === ' ' ? 9 : 10.5, fontWeight: 600, fill: st === 0 ? 'var(--text-3)' : '#0e1116' }}>
              {it.p === ' ' ? '␣' : it.p}
            </text>
          </g>
        ))}

        {st >= 1 && (
          <Label x={30} y={44 + rows * 34 + 6} tone={3} size={9.5}>
            {`${pieces.length} pieces · ${(chars / pieces.length).toFixed(2)} characters each · one integer each`}
          </Label>
        )}

        {/* ── lower panel ── */}
        <rect x={20} y={126} width={600} height={214} rx={4} fill="none" stroke="var(--rule)" strokeWidth={1} strokeDasharray="4 4" className="v-fade" />

        {st < 5 && (
          <g className="v-fade" style={{ opacity: st >= 2 ? 1 : 0.18 }}>
            <Label x={30} y={146} tone={1} size={10.5} bold>{`a toy vocabulary of ${VOCAB.length} pieces — the ID is the row number`}</Label>
            {VOCAB.map((p, i) => {
              const cx = gridX + (i % 6) * (cw + 5);
              const cy = gridY + Math.floor(i / 6) * (ch + 4);
              const used = ids.includes(i);
              return (
                <g key={p + i}>
                  <rect x={cx} y={cy} width={cw} height={ch} rx={2}
                    fill={used && st >= 2 ? 'var(--memory-dim)' : 'transparent'}
                    stroke={used && st >= 2 ? 'var(--memory)' : 'var(--rule)'}
                    strokeWidth={used && st >= 2 ? 1 : 0.8} className="v-cell" />
                  <text x={cx + 5} y={cy + 16} className="v-label" style={{ fontSize: 9, fill: used && st >= 2 ? 'var(--memory)' : 'var(--text-3)' }}>{i}</text>
                  <text x={cx + cw - 5} y={cy + 16} textAnchor="end" className="v-label" style={{ fontSize: 10, fill: used && st >= 2 ? 'var(--text)' : 'var(--text-3)' }}>
                    {p === ' ' ? '␣ space' : p}
                  </text>
                </g>
              );
            })}
            <Label x={30} y={296} tone={3} size={9.5}>{`Llama-3.1 has 128,256 entries; Qwen2.5 has 151,936. Same idea, 5,500× more of them.`}</Label>
            <Label x={30} y={314} tone={3} size={9.5}>{'An ID is not a meaning. It is an index — into the embedding table, and into the output head.'}</Label>
            <Label x={30} y={332} tone={3} size={9.5}>{'That is the last thing the model knows about your text until 03.3 turns it into a vector.'}</Label>
          </g>
        )}

        {st === 5 && (
          <g className="v-fade">
            <Label x={30} y={146} tone={1} size={10.5} bold>No text is ever unknown</Label>
            <rect x={30} y={162} width={104} height={40} rx={3} fill="var(--memory)" stroke="var(--memory)" strokeWidth={1} />
            <text x={82} y={180} textAnchor="middle" className="v-label" style={{ fontSize: 10, fontWeight: 700, fill: '#0e1116' }}>256</text>
            <text x={82} y={193} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: '#0e1116' }}>byte tokens</text>
            <Label x={148} y={188} tone={2} size={14} math>×</Label>
            <rect x={168} y={162} width={104} height={40} rx={3} fill="var(--compute)" stroke="var(--compute)" strokeWidth={1} />
            <text x={220} y={180} textAnchor="middle" className="v-label" style={{ fontSize: 10, fontWeight: 700, fill: '#0e1116' }}>501</text>
            <text x={220} y={193} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: '#0e1116' }}>merge levels</text>
            <Label x={288} y={180} tone={1} size={11}>=</Label>
            <Label x={288} y={196} tone={2} size={11}>128,256 vocabulary entries</Label>
            <Label x={30} y={228} tone={2} size={10}>Any byte sequence in the world can be spelled out one byte token at a time,</Label>
            <Label x={30} y={246} tone={2} size={10}>so a tokenizer never has to say “unknown”. It just becomes expensive.</Label>
            <Label x={30} y={276} tone={3} size={9.5}>{'Cost of the fallback: a rare name splits into 4–6 byte tokens instead of 2,'}</Label>
            <Label x={30} y={294} tone={3} size={9.5}>{'each one paying for an embedding row, an attention row and a KV cache entry.'}</Label>
            <Label x={30} y={322} tone={3} size={9.5}>{'Qwen2.5 is 151,936 = 256 × 593 + 288 special tokens (chat markers).'}</Label>
          </g>
        )}

        {st >= 6 && (
          <g className="v-fade">
            <Label x={30} y={146} tone={1} size={10.5} bold>So one token costs…</Label>
            {[
              { y: 164, kind: 'memory' as const, what: 'one row of the embedding table', cost: '4,096 × 2 B = 8 KB read', note: 'Llama-3.1-8B, once, at the start' },
              { y: 212, kind: 'compute' as const, what: 'one row in every score matrix', cost: 'T values × 32 heads × 32 layers', note: 'attention is quadratic in T' },
              { y: 260, kind: 'memory' as const, what: 'one KV cache entry, every layer', cost: '2 × 8 heads × 128 × 2 B = 4 KB', note: 'and it stays until the request ends' },
            ].map((r) => (
              <g key={r.what}>
                <rect x={30} y={r.y} width={6} height={38} rx={2} fill={r.kind === 'compute' ? 'var(--compute)' : 'var(--memory)'} />
                <text x={48} y={r.y + 15} className="v-label t2" style={{ fontSize: 10.5 }}>{r.what}</text>
                <text x={612} y={r.y + 15} textAnchor="end" className="v-label" style={{ fontSize: 10.5, fill: 'var(--text)', fontWeight: 600 }}>{r.cost}</text>
                <text x={48} y={r.y + 31} className="v-label t3" style={{ fontSize: 9 }}>{r.note}</text>
              </g>
            ))}
            <line x1={30} y1={308} x2={612} y2={308} className="v-rule faint" />
            <Label x={30} y={326} tone={1} size={10}>A 10,000-token prompt pays all three of those 10,000 times.</Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
