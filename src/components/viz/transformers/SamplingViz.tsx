import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useClock } from '../lib/tween';
import { Label, Fade } from '../lib/svg';

const VOCAB_TOTAL = 128256;

// The top 14 tokens of a real distribution for the prompt "The capital of France is".
// The logits are illustrative; every number derived from them is exactly what the model computes.
const VOCAB: [string, number][] = [
  ['Paris', 7.2], [' the', 3.1], [' a', 2.4], [' known', 1.9], [' also', 1.1],
  [' located', 0.2], [' famous', -0.4], [' home', -0.9], [' one', -1.4],
  [' city', -1.8], [' in', -2.2], [' beautiful', -2.7], [' region', -3.1], [' of', -3.6],
];
const LOGITS = VOCAB.map(([, l]) => l);

const softmax = (xs: number[]) => {
  const mx = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};
const entropy = (p: number[]) => -p.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0);
/** Nucleus (top-p): the smallest prefix whose probabilities reach p, kept. */
const nucleus = (probs: number[], p: number) => {
  const order = probs.map((v, i) => [v, i] as [number, number]).sort((a, b) => b[0] - a[0]);
  let acc = 0;
  let keep = 0;
  for (const [v] of order) { acc += v; keep++; if (acc >= p) break; }
  return order.slice(0, keep).map(([, i]) => i).sort((a, b) => a - b);
};

const STEPS = [
  '128,256 raw scores',
  'Softmax: scores become probabilities',
  'Temperature: a knob on the softmax',
  'Greedy is a special case',
  'Top-k: only the k best',
  'Top-p: only the confident mass',
  'Renormalise and draw',
  'What an engine actually receives',
];
const CAPTIONS = [
  'The output head gives one score per vocabulary token: 128,256 numbers. No probabilities yet — just preferences, on an arbitrary scale, negatives included.',
  'Exponentiate and divide by the total. Now every token has a probability, they sum to 1, and 96.7% of the mass already sits on the right answer.',
  'Divide the logits by a temperature first. Below 1 the distribution sharpens; above 1 it flattens toward uniform. It is the same operation as attention’s 1/√d.',
  'Greedy always takes the argmax. Deterministic and reproducible, but prone to loops and bland text. Sampling draws from the distribution instead — and a seed makes it reproducible again.',
  'Top-k zeroes everything below rank k. k = 1 is greedy; k = 128,256 is no filtering. A fixed k is a poor fit: it is far too loose on an easy question and too tight on a hard one.',
  'Top-p, or nucleus sampling, keeps the smallest set whose probabilities sum to p. On a peaked distribution that is one token; on a flat one it can be twenty. One number, both behaviours.',
  'Zero the filtered tokens, renormalise the survivors to sum to 1, and walk the cumulative sum with a uniform random number. That single pass is the whole sampling step — no sort needed.',
  'These are request parameters, not model constants: Llama-3.1 ships temperature 0.6 and top_p 0.9, Qwen2.5 ships 0.7, 0.8 and top_k 20. The model file decides the defaults.',
];

const AXIS_Y = 44;
const TOPY = 54;
const ROWH = 15;
const BARH = 11;
const BARX = 120;
const BARW = 400;
const ROW_END = TOPY + VOCAB.length * ROWH - (ROWH - BARH);

export default function SamplingViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [temp, setTemp] = useState(1);
  const [topk, setTopk] = useState(3);
  const [topp, setTopp] = useState(0.9);
  const [method, setMethod] = useState<'sample' | 'greedy'>('sample');
  const [seed, setSeed] = useState(0.37);
  const clock = useClock(s.playing, 1.1);

  const p = softmax(LOGITS.map((l) => l / temp));

  // the cascade: each stage narrows the set
  const kept = new Set<number>();
  if (st === 0 || st === 1 || st === 2 || st === 3) p.forEach((_, i) => kept.add(i));
  if (st >= 4) {
    p.map((v, i) => [v, i] as [number, number])
      .sort((a, b) => b[0] - a[0])
      .slice(0, Math.min(topk, p.length))
      .forEach(([, i]) => kept.add(i));
  }
  if (st >= 5) nucleus(p, topp).forEach((i) => kept.add(i));

  const filt = p.map((v, i) => (kept.has(i) ? v : 0));
  const mass = filt.reduce((a, b) => a + b, 0);
  const norm = filt.map((v) => (mass > 0 ? v / mass : 0));
  const H = entropy(norm);
  const topI = norm.indexOf(Math.max(...norm));

  // a deterministic pseudo-draw that walks the cumulative sum, as the GPU does
  const draw = st >= 6 && method === 'sample' ? (seed + clock * 0.27) % 1 : 0;
  let cum = 0;
  let drawnI = topI;
  for (let i = 0; i < norm.length; i++) {
    cum += norm[i];
    if (draw < cum) { drawnI = i; break; }
  }

  const span = LOGITS[0] - LOGITS[LOGITS.length - 1];

  return (
    <Stage
      rootRef={ref}
      title="From 128,256 logits to one token"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'token', ['idle', 'filtered out']]}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'tokens kept', value: `${kept.size}`, kind: 'token' as const, note: `of ${VOCAB_TOTAL.toLocaleString('en-US')} in the full vocabulary` },
            { label: 'mass kept', value: `${(mass * 100).toFixed(1)}%`, kind: 'compute' as const, note: 'renormalised to 100%' },
            { label: 'entropy of the choice', value: `${H.toFixed(2)} bits`, kind: 'text' as const, note: `0 = deterministic, ${Math.log2(Math.max(2, kept.size)).toFixed(2)} = uniform` },
            { label: 'most likely', value: `“${VOCAB[topI][0].trim()}”`, kind: 'token' as const, note: `${(norm[topI] * 100).toFixed(1)}% after filtering` },
          ]}
        />
      }
      controls={
        st >= 2 ? (
          <>
            <Slider label="Temperature — lower is sharper" value={temp} min={0.1} max={2} step={0.05} onChange={setTemp} format={(v) => v.toFixed(2)} kind="compute" />
            {st >= 4 && <Slider label="Top-k — keep the k best" value={topk} min={1} max={VOCAB.length} step={1} onChange={setTopk} format={(v) => `k = ${v}`} />}
            {st >= 5 && <Slider label="Top-p — keep the smallest set reaching p" value={topp} min={0.1} max={1} step={0.01} onChange={setTopp} format={(v) => v.toFixed(2)} kind="token" />}
            {st >= 6 && (
              <>
                <Segmented label="How the token is chosen" value={method} options={[{ value: 'sample' as const, label: 'sample' }, { value: 'greedy' as const, label: 'greedy' }]} onChange={setMethod} />
                <button type="button" className="btn ghost" style={{ fontSize: '0.78rem' }} onClick={() => setSeed((v) => (v + 0.173) % 1)}>
                  Draw again
                </button>
              </>
            )}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A distribution over fourteen vocabulary tokens being narrowed by softmax, temperature, top-k and top-p, ending in one sampled token drawn from the cumulative sum.">
        <text x={24} y={22} className="v-label t3" style={{ fontSize: 9 }}>{'“The capital of France is …”'}</text>
        <text x={616} y={22} textAnchor="end" className="v-label t2" style={{ fontSize: 9 }}>{st === 0 ? 'raw logit' : 'probability after filtering'}</text>

        {/* axis */}
        <line x1={BARX} x2={BARX + BARW} y1={AXIS_Y} y2={AXIS_Y} className="v-rule" />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={BARX + t * BARW} x2={BARX + t * BARW} y1={AXIS_Y} y2={AXIS_Y + 4} className="v-rule" />
            <text x={BARX + t * BARW} y={AXIS_Y - 6} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{`${(t * 100).toFixed(0)}%`}</text>
          </g>
        ))}

        {VOCAB.map(([name, logit], i) => {
          const y = TOPY + i * ROWH;
          const alive = kept.has(i);
          const isTop = i === topI;
          const isDrawn = st >= 6 && i === drawnI;
          const w = st === 0 ? ((logit - LOGITS[LOGITS.length - 1]) / span) * BARW * 0.5 : norm[i] * BARW;
          return (
            <g key={name}>
              <text x={BARX - 8} y={y + BARH - 2} textAnchor="end" className="v-label"
                style={{ fontSize: 9.5, fill: alive ? 'var(--text)' : 'var(--text-3)' }}>{name}</text>
              <rect x={BARX} y={y} width={Math.max(1, w)} height={BARH} rx={1.5}
                fill={!alive ? 'url(#tps-hatch)' : isTop ? 'var(--token)' : st === 0 ? 'var(--compute-dim)' : 'var(--compute)'}
                stroke={!alive ? 'var(--idle)' : isTop ? 'var(--token)' : 'var(--compute)'} strokeWidth={0.9} className="v-cell" />
              <text x={BARX + Math.max(1, w) + 5} y={y + BARH - 2} className="v-label"
                style={{ fontSize: 9, fill: alive ? 'var(--text)' : 'var(--text-3)' }}>
                {st === 0 ? logit.toFixed(1) : `${(norm[i] * 100).toFixed(1)}%`}
              </text>
              <Fade show={isDrawn}>
                <rect x={BARX - 4} y={y - 3} width={BARW + 8} height={BARH + 6} rx={3}
                  fill="none" stroke="var(--token)" strokeWidth={1.4} className="v-cell" />
              </Fade>
            </g>
          );
        })}

        <Fade show={st >= 6}>
          <g>
            <line x1={BARX + draw * BARW} x2={BARX + draw * BARW} y1={AXIS_Y} y2={ROW_END}
              stroke="var(--token)" strokeWidth={1.3} strokeDasharray="4 3" className="v-fade" />
            <text x={Math.min(608, BARX + draw * BARW)} y={ROW_END + 14} className="v-label" style={{ fontSize: 9, fill: 'var(--token)' }}>{`u = ${draw.toFixed(3)}`}</text>
            <text x={24} y={ROW_END + 30} className="v-label t2" style={{ fontSize: 9.5 }}>
              {method === 'greedy' ? 'greedy: always the argmax, the same token every time' : `walk the cumulative sum until it passes u → “${VOCAB[drawnI][0].trim()}”`}
            </text>
          </g>
        </Fade>

        {st === 0 && (
          <g>
            <Label x={24} y={316} tone={1} size={10} bold>preferences, not probabilities</Label>
            <Label x={24} y={336} tone={2} size={9}>{'Any real number, negatives included. The scale is arbitrary — only differences matter, because'}</Label>
            <Label x={24} y={354} tone={2} size={9}>{'softmax is shift-invariant: adding 5 to every logit changes nothing at all.'}</Label>
            <Label x={24} y={372} tone={3} size={9}>{'The full vector is 128,256 numbers. Only the top 14 are drawn here.'}</Label>
          </g>
        )}
        {st === 1 && (
          <g>
            <Label x={24} y={316} tone={1} size={10} bold>scores in, probabilities out</Label>
            <Label x={24} y={336} tone={2} size={9}>{'e^7.2 = 1,339.4  ·  e^3.1 = 22.2  ·  e^2.4 = 11.0  ·  e^-3.6 = 0.027  ·  sum = 1,385.3'}</Label>
            <Label x={24} y={354} tone={2} size={9}>{`Paris takes ${(p[0] * 100).toFixed(1)}% on its own. One softmax over 128,256 numbers is about 0.26 MFLOP.`}</Label>
            <Label x={24} y={372} tone={3} size={9}>{'Subtract the max first (Module 01) so e^x never overflows FP16. Identical result.'}</Label>
          </g>
        )}
        {st === 2 && (
          <g>
            <Label x={24} y={316} tone={1} size={10} bold>temperature is the same knob as attention’s 1/√d</Label>
            <Label x={24} y={336} tone={2} size={9}>{`At T = ${temp.toFixed(2)}: top token ${(p[0] * 100).toFixed(1)}%, entropy ${entropy(p).toFixed(2)} bits. Drag it and watch the whole shape move.`}</Label>
            <Label x={24} y={354} tone={2} size={9}>{'T < 1 sharpens: the model commits. T > 1 flattens: it explores. T → 0 is greedy.'}</Label>
            <Label x={24} y={372} tone={3} size={9}>{'Llama-3.1 defaults to 0.6, Qwen2.5 to 0.7 — both deliberately below 1.'}</Label>
          </g>
        )}
        {st === 3 && (
          <g>
            <Label x={24} y={316} tone={1} size={10} bold>two ways to pick a token</Label>
            <Label x={24} y={336} tone={2} size={9}>{'Greedy takes the argmax: deterministic, reproducible, and prone to loops and bland text.'}</Label>
            <Label x={24} y={354} tone={2} size={9}>{'Sampling draws from the distribution: varied, and controlled by temperature and the filters below.'}</Label>
            <Label x={24} y={372} tone={3} size={9}>{'A seed makes sampling reproducible too — that is the whole purpose of a seed in an API.'}</Label>
          </g>
        )}
        {st === 4 && (
          <g>
            <Label x={24} y={316} tone={1} size={10} bold>{`top-k keeps the ${Math.min(topk, VOCAB.length)} most probable tokens`}</Label>
            <Label x={24} y={336} tone={2} size={9}>{`Those ${Math.min(topk, VOCAB.length)} hold ${(mass * 100).toFixed(1)}% of the mass; everything below is set to −∞ before the softmax.`}</Label>
            <Label x={24} y={354} tone={2} size={9}>{'A fixed k cannot suit both an easy question and a hard one. Qwen2.5 ships top_k = 20.'}</Label>
            <Label x={24} y={372} tone={3} size={9}>{'Cost: a partial selection over 128,256 values — no full sort needed, but still a pass.'}</Label>
          </g>
        )}
        {st === 5 && (
          <g>
            <Label x={24} y={316} tone={1} size={10} bold>{`top-p keeps ${kept.size} token${kept.size === 1 ? '' : 's'} at p = ${topp.toFixed(2)}`}</Label>
            <Label x={24} y={336} tone={2} size={9}>{'Nucleus sampling sorts descending, then walks until the cumulative probability reaches p.'}</Label>
            <Label x={24} y={354} tone={2} size={9}>{'Raise the temperature and watch it keep more tokens than top-k did — it adapts, top-k cannot.'}</Label>
            <Label x={24} y={372} tone={3} size={9}>{'Llama-3.1 ships top_p = 0.9. Top-k and top-p together are what most APIs expose.'}</Label>
          </g>
        )}
        {st === 6 && (
          <g>
            <Label x={24} y={316} tone={1} size={10} bold>renormalise, then one uniform draw</Label>
            <Label x={24} y={336} tone={2} size={9}>{`The surviving ${kept.size} token${kept.size === 1 ? '' : 's'} now hold ${(mass * 100).toFixed(1)}% of the original mass, rescaled to 100%.`}</Label>
            <Label x={24} y={354} tone={2} size={9}>{'Then a single pass over the vocabulary: accumulate until the sum passes u. No sort, no extra kernel.'}</Label>
            <Label x={24} y={372} tone={3} size={9}>{'Press Draw again. The same seed always gives the same token.'}</Label>
          </g>
        )}
        {st === 7 && (
          <g>
            <Label x={24} y={316} tone={1} size={10} bold>what an engine actually receives</Label>
            <Label x={24} y={336} tone={2} size={9}>{'temperature, top_p, top_k, min_p, repetition_penalty, seed — request parameters, not model constants.'}</Label>
            <Label x={24} y={354} tone={2} size={9}>{'The model file supplies the defaults: Llama-3.1 → 0.6 / 0.9.  Qwen2.5 → 0.7 / 0.8 / 20.'}</Label>
            <Label x={24} y={372} tone={3} size={9}>{'Constrained decoding replaces the draw with a mask and an argmax, and skips the sort entirely.'}</Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
