import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade } from '../lib/svg';

// Four tokens of a real prompt, and one attention head. d_head = 4 so every
// number stays visible; the arithmetic is identical at d_head = 128.
const TOKENS = ['The', ' capital', ' of', ' France'];
const Q = [
  [1.2, 0.4, -0.6, 0.9],
  [0.3, -1.1, 0.7, 0.5],
  [-0.8, 0.6, 1.1, -0.4],
  [0.9, 0.2, -0.2, 1.3],
];
const K = [
  [1.0, 0.5, -0.5, 0.8],
  [0.2, -0.9, 0.6, 0.4],
  [-0.7, 0.5, 1.0, -0.3],
  [1.1, 0.1, -0.1, 1.2],
];
const V = [
  [1.0, -1.0, 0.5, 0.0],
  [0.5, -0.5, 1.0, 0.5],
  [-1.0, 0.0, 0.5, 1.0],
  [0.0, 0.5, -0.5, 1.0],
];

const D_HEAD = 4;
const SQRT_REAL = Math.sqrt(128); // what the same factor is at d_head = 128

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
const entropy = (p: number[]) => -p.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0);

const STEPS = [
  'Three questions, one per token',
  'Project: Q, K and V',
  'Compare the query with every key',
  'Why divide by √d',
  'Softmax turns scores into shares',
  'The causal mask',
  'Mix the values',
  'One head, done',
];
const CAPTIONS = [
  'Every token asks three questions. Four numbers per vector so you can see every product; real heads use 128, and 32 heads run per layer.',
  'These are not the token\'s embedding. They are linear projections of it: q = Wq·x, and likewise k and v. Three more matrices per layer — and the reason k and v are narrower is the whole of the next lesson.',
  'One dot product per token pair: q·k = 1.07, 0.20, −0.56 and 1.30 after scaling. The query asks, each key answers with "this is what I contain", and the dot product measures the match.',
  'With 128 numbers per vector the dot products sum 128 terms and get large, and a softmax over large scores goes almost one-hot. Dividing by √d keeps them sane. Drag the divisor and watch the shares flatten.',
  'Exponentiate and divide by the total. Every earlier token now has a share of the attention, and the shares sum to 1: 34.9%, 14.6%, 6.8% and 43.7%.',
  'A generated token may not look forward. Everything right of the diagonal is set to minus infinity before the softmax, so attention only ever flows backwards.',
  'Weight each value vector by its share and add them up. That weighted average is the head\'s output — its answer to "what should I take from the past".',
  'Project, score, scale, mask, softmax, mix. Thirty-two of these per layer, thirty-two layers deep, and then the MLP. That is the entire transformer.',
];

const COLX = [66, 198, 330, 462];
const COLW = 124;
const mixX = 290;

export default function AttentionViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [qi, setQi] = useState(3);
  const [scaleDiv, setScaleDiv] = useState(2);

  const q = qi;
  const div = st >= 3 ? scaleDiv : 1;
  const raw = K.map((k) => dot(Q[q], k));
  const scaled = raw.map((v) => v / div);
  const mx = Math.max(...scaled.slice(0, q + 1));
  const exps = scaled.map((v, i) => (i <= q ? Math.exp(v - mx) : 0));
  const sum = exps.reduce((a, b) => a + b, 0);
  const p = exps.map((e) => e / sum);
  // out[i] = Σ_j share_j · value_j[i] — a weighted average down each column
  const out = V[0].map((_, i) => V.reduce((acc, row, j) => acc + row[i] * p[j], 0));
  const best = p.indexOf(Math.max(...p));

  const projOn = st >= 1;
  const scoreOn = st >= 2;
  const mixOn = st >= 6;

  return (
    <Stage
      rootRef={ref}
      title="One attention head, on four tokens"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['idle', 'masked out'], 'token']}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'query', value: `“${TOKENS[q]}”`, kind: 'compute' as const, note: `position ${q}` },
            { label: 'largest share', value: scoreOn ? `${(Math.max(...p) * 100).toFixed(1)}%` : '—', kind: 'token' as const, note: scoreOn ? `on “${TOKENS[best]}”` : 'not computed' },
            { label: 'entropy', value: scoreOn ? `${entropy(p).toFixed(2)} bits` : '—', kind: 'text' as const, note: `of at most ${Math.log2(q + 1).toFixed(2)}` },
            { label: 'head output', value: mixOn ? `[${out.map((v) => v.toFixed(2)).join(', ')}]` : '—', kind: 'text' as const, note: mixOn ? '4 numbers out' : 'waiting' },
          ]}
        />
      }
      controls={
        st >= 1 ? (
          <>
            <Slider label="Which token asks (the query row)" value={q} min={0} max={3} step={1} onChange={setQi} format={(v) => `“${TOKENS[v]}”`} kind="compute" />
            {st >= 3 && (
              <Slider label={`Divisor √d_head — ${D_HEAD} now, ${SQRT_REAL.toFixed(1)} in a real 128-wide head`} value={div} min={1} max={16} step={0.5} onChange={setScaleDiv} format={(v) => `÷ ${v.toFixed(1)}`} />
            )}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Attention on four tokens: each token has a query, key and value vector; the query is compared with every key, scaled, masked causally, turned into softmax shares, and used to mix the value vectors.">
        {/* ── token chips ── */}
        {TOKENS.map((t, i) => (
          <g key={t}>
            <rect x={COLX[i]} y={24} width={COLW} height={22} rx={3}
              fill={i === q ? 'var(--compute)' : 'var(--ground-3)'}
              stroke={i === q ? 'var(--compute)' : 'var(--rule-2)'} strokeWidth={1} className="v-cell" />
            <text x={COLX[i] + COLW / 2} y={39} textAnchor="middle" className="v-label"
              style={{ fontSize: 10.5, fontWeight: 600, fill: i === q ? '#0e1116' : 'var(--text-2)' }}>{t}</text>
            <text x={COLX[i] + COLW / 2} y={57} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{`pos ${i}`}</text>
          </g>
        ))}

        {/* ── Q, K, V rows ── */}
        {(['Q', 'K', 'V'] as const).map((row, ri) => (
          <Fade key={row} show={projOn}>
            <g>
              <Label x={24} y={90 + ri * 28} tone={1} size={10} kind={ri === 0 ? 'compute' : ri === 1 ? 'memory' : 'token'} bold>{row}</Label>
              {K.map((_, ci) => (
                <g key={ci}>
                  <rect x={COLX[ci] + 4} y={78 + ri * 28} width={COLW - 8} height={20} rx={2}
                    fill={ri === 0 ? 'var(--compute-dim)' : ri === 1 ? 'var(--memory-dim)' : 'var(--token-dim)'}
                    stroke={ri === 0 ? 'var(--compute)' : ri === 1 ? 'var(--memory)' : 'var(--token)'} strokeWidth={0.9} className="v-cell" />
                  <text x={COLX[ci] + COLW / 2} y={92 + ri * 28} textAnchor="middle" className="v-label"
                    style={{ fontSize: 9.5, fill: 'var(--text)' }}>
                    {(ri === 0 ? Q : ri === 1 ? K : V)[ci].map((v) => v.toFixed(1)).join('  ')}
                  </text>
                </g>
              ))}
            </g>
          </Fade>
        ))}

        {/* ── scores and shares ── */}
        <Fade show={scoreOn}>
          <Label x={24} y={204} tone={2} size={9}>q·k</Label>
          <Label x={24} y={250} tone={2} size={9}>share</Label>
          {K.map((_, i) => {
            const blocked = st >= 5 && i > q;
            const hot = i === best;
            return (
              <g key={i}>
                <rect x={COLX[i]} y={190} width={COLW} height={24} rx={3}
                  fill={blocked ? 'url(#tps-hatch)' : hot ? 'var(--token)' : 'var(--compute)'}
                  stroke={blocked ? 'var(--idle)' : hot ? 'var(--token)' : 'var(--compute)'} strokeWidth={1} className="v-cell" />
                <text x={COLX[i] + COLW / 2} y={206} textAnchor="middle" className="v-label"
                  style={{ fontSize: 11, fontWeight: 700, fill: blocked ? 'var(--text-3)' : '#0e1116' }}>
                  {blocked ? 'masked' : scaled[i].toFixed(2)}
                </text>
                <Arrow x1={COLX[i] + COLW / 2} y1={214} x2={COLX[i] + COLW / 2} y2={232} kind={blocked ? 'idle' : 'compute'} width={1.2} />
                <rect x={COLX[i]} y={234} width={COLW} height={24} rx={3}
                  fill={blocked ? 'url(#tps-hatch)' : hot ? 'var(--token)' : 'var(--compute-dim)'}
                  stroke={blocked ? 'var(--idle)' : hot ? 'var(--token)' : 'var(--compute)'} strokeWidth={1} className="v-cell" />
                <text x={COLX[i] + COLW / 2} y={250} textAnchor="middle" className="v-label"
                  style={{ fontSize: 11, fontWeight: 700, fill: blocked ? 'var(--text-3)' : hot ? '#0e1116' : 'var(--text)' }}>
                  {blocked ? '—' : `${(p[i] * 100).toFixed(1)}%`}
                </text>
              </g>
            );
          })}
          <Label x={592} y={204} tone={3} size={8.5}>{`÷ ${div.toFixed(1)}`}</Label>
        </Fade>

        {/* ── mix ── */}
        <Fade show={mixOn}>
          {st === 6 && (
            <g>
              <Label x={24} y={290} tone={2} size={10}>output = Σ share · value</Label>
              <Label x={24} y={312} tone={3} size={8.5}>{'4 numbers'}</Label>
              <Label x={24} y={348} tone={2} size={9}>what the head hands</Label>
              <Label x={24} y={362} tone={2} size={9}>back to the block</Label>
            </g>
          )}
          {out.map((v, i) => {
            const w = 74;
            const bx = mixX + i * 84;
            const zero = bx + w / 2;
            return (
              <g key={i}>
                <rect x={bx} y={326} width={w} height={18} rx={2} fill="none" stroke="var(--rule)" />
                <line x1={zero} x2={zero} y1={326} y2={344} className="v-rule faint" />
                <rect x={v >= 0 ? zero : zero + v * 32} y={326} width={Math.max(1, Math.abs(v) * 32)} height={18} rx={1.5}
                  fill={v >= 0 ? 'var(--token)' : 'none'} stroke="var(--token)" strokeWidth={1}
                  strokeDasharray={v >= 0 ? undefined : '3 3'} className="v-cell" />
                <text x={zero} y={360} textAnchor="middle" className="v-label t3" style={{ fontSize: 9 }}>{v.toFixed(2)}</text>
              </g>
            );
          })}
          {st === 7 && (
            <g>
              <Label x={24} y={290} tone={1} size={10.5} bold>{'project → score → scale → mask → softmax → mix'}</Label>
              <Label x={24} y={306} tone={2} size={9}>{'32 heads per layer (03.5), 32 layers deep (03.6).'}</Label>
              <Label x={24} y={322} tone={3} size={9}>{`4 wide here, 128 real → ÷${SQRT_REAL.toFixed(1)}, not ÷${Math.sqrt(D_HEAD).toFixed(0)}`}</Label>
            </g>
          )}
        </Fade>

        {/* ── step notes ── */}
        {st === 0 && (
          <g>
            <Label x={24} y={96} tone={1} size={11} bold>every token asks three questions</Label>
            <Label x={24} y={124} tone={2} size={10}>Q — what am I looking for?</Label>
            <Label x={24} y={146} tone={2} size={10}>K — what do I contain?</Label>
            <Label x={24} y={168} tone={2} size={10}>V — what do I hand over if I am chosen?</Label>
            <Label x={24} y={206} tone={3} size={9}>Four tokens, three vectors each, four numbers per vector so you</Label>
            <Label x={24} y={222} tone={3} size={9}>can check every product by hand. Real heads use 128 numbers,</Label>
            <Label x={24} y={238} tone={3} size={9}>and 32 of them run side by side in every layer.</Label>
          </g>
        )}

        {st === 1 && (
          <g>
            <Label x={24} y={292} tone={1} size={10.5} bold>three projections, applied to the same vector</Label>
            <Label x={24} y={316} tone={3} size={9}>{'q = Wq·x   [4096 × 4096]  ·  32 query heads'}</Label>
            <Label x={24} y={332} tone={3} size={9}>{'k = Wk·x   [4096 × 1024]  ·  only 8 key heads in Llama-3.1-8B'}</Label>
            <Label x={24} y={348} tone={3} size={9}>{'v = Wv·x   [4096 × 1024]  ·  and V is what gets passed on'}</Label>
            <Label x={24} y={370} tone={2} size={9}>RoPE then rotates Q and K (03.3) — it never touches V.</Label>
          </g>
        )}

        {st === 2 && (
          <g>
            <Label x={24} y={292} tone={1} size={10.5} bold>four dot products, one per key</Label>
            <Label x={24} y={316} tone={3} size={9}>{`q = [${Q[q].map((v) => v.toFixed(1)).join(', ')}]  ·  k₃ = [${K[3].map((v) => v.toFixed(1)).join(', ')}]`}</Label>
            <Label x={24} y={332} tone={3} size={9}>{`q·k₃ = ${Q[q].map((v, i) => `${v.toFixed(1)}×${K[3][i].toFixed(1)}`).join(' + ')} = ${raw[3].toFixed(2)}`}</Label>
            <Label x={24} y={354} tone={2} size={9}>The query asks one question. Every key answers. That is the score.</Label>
          </g>
        )}

        {st >= 3 && st < 6 && (
          <g>
            <Label x={24} y={292} tone={1} size={10.5} bold>{st === 3 ? 'why the 1/√d' : 'shares of one, summing to 1'}</Label>
            <Label x={24} y={316} tone={3} size={9}>{`Raw scores span ${Math.min(...raw).toFixed(2)} to ${Math.max(...raw).toFixed(2)}. With 128 terms`}</Label>
            <Label x={24} y={332} tone={3} size={9}>{'per dot product they span far more, and a softmax over large'}</Label>
            <Label x={24} y={348} tone={3} size={9}>{'scores collapses onto whichever key is largest — one-hot, zero'}</Label>
            <Label x={24} y={364} tone={3} size={9}>{'gradient for every other key. Divide by √d to stay in range.'}</Label>
          </g>
        )}

        {st === 5 && (
          <g>
            <Label x={24} y={292} tone={1} size={10.5} bold>the causal mask</Label>
            <Label x={24} y={316} tone={3} size={9}>{'A generated token must not read the future. Add −∞ to every'}</Label>
            <Label x={24} y={332} tone={3} size={9}>{'position after the query, before the softmax. e^(−∞) = 0, so'}</Label>
            <Label x={24} y={348} tone={3} size={9}>{'those keys get exactly zero weight and the rest renormalise.'}</Label>
            <Label x={24} y={370} tone={2} size={9}>Move the query slider left and watch the shares renormalise.</Label>
          </g>
        )}

        {st === 7 && (
          <g>
            <Label x={24} y={296} tone={1} size={10.5} bold>{'project → score → scale → mask → softmax → mix'}</Label>
            <Label x={24} y={322} tone={2} size={9.5}>{'32 heads per layer (03.5), 32 layers deep (03.6).'}</Label>
            <Label x={24} y={344} tone={3} size={9}>{`This head is ${D_HEAD} wide. A real one is 128, so the divisor is`}</Label>
            <Label x={24} y={360} tone={3} size={9}>{`1/√128 = 1/${SQRT_REAL.toFixed(3)} instead of 1/${Math.sqrt(D_HEAD).toFixed(1)}.`}</Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
