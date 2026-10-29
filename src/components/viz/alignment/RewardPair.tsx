import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Label, Fade, Arrow } from '../lib/svg';

const sig = (x: number) => 1 / (1 + Math.exp(-x));
const nCk2 = (k: number) => (k * (k - 1)) / 2;

const STEPS = [
  'Write the ideal answer? Hard',
  'Compare two answers instead',
  'Rank K answers, get C(K,2) pairs',
  'A model that outputs one number',
  'Bradley-Terry: gap → probability',
  'The loss: −log σ(r_w − r_l)',
  'Training pulls the scores apart',
  'Only differences matter',
];
const CAPTIONS = [
  'Writing the best possible answer is slow, and two experts will write two different ones. That is the weak spot of supervised finetuning.',
  'Showing a person two answers and asking which is better is fast and far more consistent. That one bit, A or B, is preference data.',
  'Ask a labeler to rank K answers at once and every pair is a comparison: K = 4 gives 6 pairs, K = 9 gives 36, for the reading cost of K answers.',
  'A reward model is the language model with its output head swapped for a single number: a score for how much people would like this answer to this prompt.',
  'The Bradley-Terry model turns two scores into a probability: P(A beats B) = σ(r_A − r_B). Scores 1.2 and 0.4 give σ(0.8) = 0.690.',
  'Train on what people chose. The loss for one comparison is −log σ(r_winner − r_loser): 0.371 nats at a gap of 0.8, near zero once the gap is large.',
  'Each gradient step raises the winner’s score and lowers the loser’s, by an amount proportional to 1 − σ(gap): the more surprised the model, the bigger the push.',
  'Add the same constant to every score and nothing changes. Rewards have no zero point and no units: only gaps carry meaning.',
];

const PROMPT = 'Explain what a KV cache is, in one sentence.';
const RESP_A = ['Each token’s keys and values are stored, so a new', 'token reuses them instead of recomputing them.'];
const RESP_B = ['A KV cache is a cache that caches the K and V,', 'which is a kind of cache used in models.'];

// plot geometry
const PX = 356;
const PW = 252;
const PY = 250; // y of probability 0
const PH = 150;
const DMIN = -5;
const DMAX = 5;
const px = (d: number) => PX + ((d - DMIN) / (DMAX - DMIN)) * PW;
const py = (p: number) => PY - p * PH;
const LOSS_MAX = 5;
const ly = (l: number) => PY - Math.min(1, l / LOSS_MAX) * PH;

const curve = (f: (d: number) => number) => {
  let path = '';
  for (let i = 0; i <= 100; i++) {
    const d = DMIN + ((DMAX - DMIN) * i) / 100;
    path += `${i === 0 ? 'M' : 'L'}${px(d).toFixed(1)},${f(d).toFixed(1)} `;
  }
  return path;
};
const SIG_PATH = curve((d) => py(sig(d)));
const LOSS_PATH = curve((d) => ly(-Math.log(sig(d))));

// a short simulated training run on this one pair: plain gradient steps on the BT loss
const TRAIN: { a: number; b: number }[] = (() => {
  const out = [{ a: 1.2, b: 0.4 }];
  let a = 1.2;
  let b = 0.4;
  for (let i = 0; i < 6; i++) {
    const g = 1 - sig(a - b); // −dL/dΔ
    a += 0.9 * g;
    b -= 0.9 * g;
    out.push({ a, b });
  }
  return out;
})();

export default function RewardPair() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [rA, setRA] = useState(1.2);
  const [rB, setRB] = useState(0.4);
  const [K, setK] = useState(4);
  const [shift, setShift] = useState(0);
  const [iter, setIter] = useState(0);

  // each step starts from the worked example
  useEffect(() => {
    setRA(1.2); setRB(0.4); setShift(0); setIter(0);
  }, [st]);
  // step 7 animates a few gradient steps
  useEffect(() => {
    if (st !== 6) return;
    let i = 0;
    const id = window.setInterval(() => { i = Math.min(TRAIN.length - 1, i + 1); setIter(i); if (i >= TRAIN.length - 1) window.clearInterval(id); }, 650);
    return () => window.clearInterval(id);
  }, [st]);

  const a = st === 6 ? TRAIN[iter].a : rA + shift;
  const b = st === 6 ? TRAIN[iter].b : rB + shift;
  const tw = useTweens({ a, b }, 500);
  const gap = tw.a - tw.b;
  const p = sig(gap);
  const loss = -Math.log(p);

  const showCards = st <= 1 || st >= 3;
  const showPick = st >= 1;
  const showScores = st >= 3;
  const showPlot = st >= 4;
  const showLoss = st >= 5;

  // reward axis for the score bars
  const RX0 = 190;
  const RSCALE = 30; // px per unit
  const rx = (r: number) => RX0 + Math.max(-4, Math.min(4, r)) * RSCALE;

  return (
    <Stage
      rootRef={ref}
      title="From one human comparison to a reward model"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['token', 'chosen answer'], ['error', 'rejected answer'], ['compute', 'loss']]}
      {...s}
      readout={
        st === 2 ? (
          <Readout items={[
            { label: 'answers ranked', value: `K = ${K}`, kind: 'text' },
            { label: 'comparisons', value: `${nCk2(K)}`, kind: 'token', note: `K(K−1)/2 = ${K}·${K - 1}/2` },
            { label: 'per answer read', value: `${(nCk2(K) / K).toFixed(1)}`, kind: 'text', note: 'comparisons per answer' },
          ]} />
        ) : st >= 3 ? (
          <Readout items={[
            { label: 'gap r_A − r_B', value: gap.toFixed(2), kind: 'text' },
            { label: 'P(A preferred)', value: st >= 4 ? p.toFixed(3) : '—', kind: 'token', note: 'σ(gap)' },
            { label: 'loss if A was chosen', value: st >= 5 ? `${loss.toFixed(3)} nats` : '—', kind: 'compute', note: '−log σ(gap)' },
          ]} />
        ) : undefined
      }
      controls={
        st === 2 ? (
          <Slider label="Answers ranked per prompt (InstructGPT used 4 to 9)" value={K} min={4} max={9} onChange={setK} format={(v) => `K = ${v}`} kind="token" />
        ) : st >= 3 && st !== 6 ? (
          <>
            <Slider label="Score of answer A, r_A" value={rA} min={-3} max={3} step={0.1} onChange={setRA} format={(v) => v.toFixed(1)} kind="token" />
            <Slider label="Score of answer B, r_B" value={rB} min={-3} max={3} step={0.1} onChange={setRB} format={(v) => v.toFixed(1)} kind="error" />
            {st === 7 && <Slider label="Add c to both scores" value={shift} min={-2} max={2} step={0.1} onChange={setShift} format={(v) => (v >= 0 ? '+' : '') + v.toFixed(1)} />}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="Two answers to one prompt, one chosen and one rejected by a person. A reward model scores each; the Bradley-Terry sigmoid turns the score gap into a preference probability, and the loss −log σ(gap) falls as training pulls the scores apart.">
        {/* prompt */}
        <Label x={24} y={18} tone={3} size={9}>prompt</Label>
        <Label x={24} y={34} tone={1} size={10.5} mono={false}>{PROMPT}</Label>

        {/* ── two answers ── */}
        <Fade show={showCards}>
          <g>
            {[RESP_A, RESP_B].map((lines, i) => {
              const y = 50 + i * 78;
              const chosen = i === 0;
              const col = showPick ? (chosen ? 'var(--token)' : 'var(--error)') : 'var(--rule-2)';
              return (
                <g key={i}>
                  <rect x={24} y={y} width={st >= 3 ? 310 : 592} height={st >= 3 ? 64 : 50} rx={4} fill="var(--ground-3)" stroke={col} strokeWidth={showPick ? 1.4 : 1} style={{ transition: 'width .6s var(--ease-out), height .5s' }} />
                  <text x={36} y={y + 16} className="v-label" style={{ fontSize: 9, fill: col, fontWeight: 700 }}>
                    {`answer ${chosen ? 'A' : 'B'}${showPick ? (chosen ? '  ✓ chosen' : '  ✗ rejected') : ''}`}
                  </text>
                  {lines.map((l, j) => (
                    <text key={j} x={36} y={y + 30 + j * 13} className="v-label sans" style={{ fontSize: 9.5, fill: 'var(--text-2)' }}>{l}</text>
                  ))}
                </g>
              );
            })}
          </g>
        </Fade>

        {/* ── step 1 notes ── */}
        <Fade show={st === 0}>
          <g>
            <Label x={24} y={210} tone={1} size={10} bold>supervised finetuning needs the ideal answer written out</Label>
            <Label x={24} y={228} tone={2} size={9}>{'Writing it is slow, and “ideal” is fuzzy: two skilled people will write two different answers,'}</Label>
            <Label x={24} y={244} tone={2} size={9}>{'and neither tells the model what makes one answer better than another.'}</Label>
            <Label x={24} y={260} tone={3} size={9}>{'Here are two answers the model itself produced. Which one would you rather get?'}</Label>
          </g>
        </Fade>
        <Fade show={st === 1}>
          <g>
            <Label x={24} y={210} tone={1} size={10} bold>a comparison: one bit of human judgement</Label>
            <Label x={24} y={228} tone={2} size={9}>{'Choosing takes seconds and people agree with each other far more often than their ideal answers would.'}</Label>
            <Label x={24} y={244} tone={2} size={9}>{'The record is a triple: (prompt, chosen answer y_w, rejected answer y_l). w for winner, l for loser.'}</Label>
            <Label x={24} y={260} tone={3} size={9}>{'Both answers came from the model: preference data is about the model’s own outputs.'}</Label>
          </g>
        </Fade>

        {/* ── K answers, all pairs ── */}
        <Fade show={st === 2}>
          <g>
            <Label x={24} y={66} tone={2} size={9.5}>{`a labeler ranks ${K} answers, best first`}</Label>
            {Array.from({ length: 9 }, (_, i) => {
              const on = i < K;
              const ang = (i / K) * Math.PI * 2 - Math.PI / 2;
              const cx = 320 + Math.cos(ang) * 92;
              const cy = 176 + Math.sin(ang) * 82;
              return (
                <g key={i} style={{ opacity: on ? 1 : 0, transition: 'opacity .4s' }}>
                  <circle cx={cx} cy={cy} r={13} fill="var(--ground-3)" stroke={i === 0 ? 'var(--token)' : i === K - 1 ? 'var(--error)' : 'var(--rule-2)'} strokeWidth={1.3} style={{ transition: 'cx .5s, cy .5s' }} />
                  <text x={cx} y={cy + 3.5} textAnchor="middle" className="v-label" style={{ fontSize: 10, fill: 'var(--text)' }}>{i + 1}</text>
                </g>
              );
            })}
            {Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, j) => {
              if (j <= i) return null;
              const a1 = (i / K) * Math.PI * 2 - Math.PI / 2;
              const a2 = (j / K) * Math.PI * 2 - Math.PI / 2;
              const x1 = 320 + Math.cos(a1) * 92; const y1 = 176 + Math.sin(a1) * 82;
              const x2 = 320 + Math.cos(a2) * 92; const y2 = 176 + Math.sin(a2) * 82;
              const dx = x2 - x1; const dy = y2 - y1; const L = Math.hypot(dx, dy);
              return <line key={`${i}-${j}`} x1={x1 + (dx / L) * 14} y1={y1 + (dy / L) * 14} x2={x2 - (dx / L) * 14} y2={y2 - (dy / L) * 14} stroke="var(--token)" strokeOpacity={0.45} strokeWidth={1} />;
            }))}
            <Label x={24} y={290} tone={1} size={10} bold>{`${K} answers → ${K}·${K - 1}/2 = ${nCk2(K)} comparisons, each line one (winner, loser) pair`}</Label>
            <Label x={24} y={308} tone={2} size={9}>{'The higher-ranked end of every line is the winner. Reading cost grows like K; comparisons like K².'}</Label>
            <Label x={24} y={324} tone={3} size={9}>{'Pairs from one prompt are correlated, so InstructGPT trained them together as one batch element.'}</Label>
          </g>
        </Fade>

        {/* ── score bars ── */}
        <Fade show={showScores}>
          <g>
            {[tw.a, tw.b].map((r, i) => {
              const y = 50 + i * 78 + 50;
              const col = i === 0 ? 'var(--token)' : 'var(--error)';
              const x0 = rx(0);
              const x1 = rx(r);
              return (
                <g key={i}>
                  <line x1={rx(-4)} x2={rx(4)} y1={y + 5} y2={y + 5} className="v-rule faint" />
                  <line x1={x0} x2={x0} y1={y - 2} y2={y + 12} className="v-rule" />
                  <rect x={Math.min(x0, x1)} y={y} width={Math.abs(x1 - x0)} height={10} rx={1.5} fill={col} />
                  <text x={36} y={y + 9} className="v-label" style={{ fontSize: 9.5, fill: col, fontWeight: 700 }}>{`r_${i === 0 ? 'A' : 'B'} = ${r.toFixed(2)}`}</text>
                </g>
              );
            })}
          </g>
        </Fade>

        {/* ── reward model architecture ── */}
        <Fade show={st === 3}>
          <g>
            <rect x={356} y={58} width={252} height={150} rx={5} fill="none" stroke="var(--rule-2)" />
            <Label x={482} y={76} anchor="middle" tone={2} size={9}>the reward model</Label>
            <rect x={376} y={88} width={212} height={26} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x={482} y={105} anchor="middle" tone={2} size={9}>prompt + answer tokens</Label>
            <rect x={376} y={124} width={212} height={30} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" />
            <Label x={482} y={143} anchor="middle" tone={1} size={9}>the SFT transformer</Label>
            <Arrow x1={482} y1={156} x2={482} y2={170} muted />
            <rect x={436} y={172} width={92} height={24} rx={3} fill="var(--ground-3)" stroke="var(--text-2)" />
            <Label x={482} y={188} anchor="middle" tone={1} size={9}>{'head → 1 number'}</Label>
            <Label x={24} y={232} tone={1} size={10} bold>swap the output head: hidden size → 1, not hidden size → vocabulary</Label>
            <Label x={24} y={250} tone={2} size={9}>{'Read the final token’s hidden state, multiply by one learned vector: a scalar score r(x, y).'}</Label>
            <Label x={24} y={266} tone={2} size={9}>{'InstructGPT started its reward model from the 6B SFT model; a 175B one trained less stably.'}</Label>
            <Label x={24} y={282} tone={3} size={9}>{'Drag the scores. Nothing yet says what a score means: the next step gives them a job.'}</Label>
          </g>
        </Fade>

        {/* ── sigmoid / loss plot ── */}
        <Fade show={showPlot}>
          <g>
            <line x1={PX} x2={PX + PW} y1={PY} y2={PY} className="v-rule" />
            <line x1={px(0)} x2={px(0)} y1={PY - PH - 6} y2={PY + 4} className="v-rule faint" />
            <line x1={PX} x2={PX + PW} y1={py(0.5)} y2={py(0.5)} className="v-rule faint" strokeDasharray="3 3" />
            {[-4, -2, 0, 2, 4].map((d) => (
              <text key={d} x={px(d)} y={PY + 13} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{d}</text>
            ))}
            <text x={PX + PW} y={PY + 26} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>gap r_A − r_B</text>
            <text x={PX - 4} y={py(1) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>1</text>
            <text x={PX - 4} y={py(0.5) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>0.5</text>
            <path d={SIG_PATH} fill="none" stroke="var(--token)" strokeWidth={1.8} />
            <text x={PX + PW - 4} y={py(1) - 6} textAnchor="end" className="v-label" style={{ fontSize: 9, fill: 'var(--token)' }}>P(A ≻ B) = σ(gap)</text>
            <Fade show={showLoss}>
              <g>
                <path d={LOSS_PATH} fill="none" stroke="var(--compute)" strokeWidth={1.6} strokeDasharray="5 3" />
                <text x={PX + 4} y={py(1) - 6} className="v-label" style={{ fontSize: 9, fill: 'var(--compute)' }}>loss −log σ(gap)</text>
                <circle cx={px(Math.max(DMIN, Math.min(DMAX, gap)))} cy={ly(loss)} r={4} fill="var(--compute)" />
              </g>
            </Fade>
            <line x1={px(Math.max(DMIN, Math.min(DMAX, gap)))} x2={px(Math.max(DMIN, Math.min(DMAX, gap)))} y1={PY} y2={py(p)} stroke="var(--text-2)" strokeDasharray="2 3" />
            <circle cx={px(Math.max(DMIN, Math.min(DMAX, gap)))} cy={py(p)} r={4.5} fill="var(--token)" />
          </g>
        </Fade>

        {/* ── explanation strip for the scoring steps ── */}
        {st >= 4 && (
          <g>
            {st === 4 && (
              <>
                <Label x={24} y={282} tone={1} size={10} bold>{`σ(${gap.toFixed(2)}) = 1 / (1 + e^${(-gap).toFixed(2)}) = ${p.toFixed(3)}`}</Label>
                <Label x={24} y={300} tone={2} size={9}>{'A gap of 0 means a coin flip. A gap of 2 means 88%. Scores are on a log-odds scale.'}</Label>
                <Label x={24} y={316} tone={3} size={9}>{'Same idea as chess ratings: the rating difference sets the odds of who wins.'}</Label>
              </>
            )}
            {st === 5 && (
              <>
                <Label x={24} y={282} tone={1} size={10} bold>{`A was chosen, so the loss is −ln ${p.toFixed(3)} = ${loss.toFixed(3)} nats`}</Label>
                <Label x={24} y={300} tone={2} size={9}>{'If the model scores the rejected answer higher (gap below 0) the loss climbs steeply. Drag r_B above r_A.'}</Label>
                <Label x={24} y={316} tone={3} size={9}>{'It is binary cross-entropy where the “logit” is the score gap.'}</Label>
              </>
            )}
            {st === 6 && (
              <>
                <Label x={24} y={282} tone={1} size={10} bold>{`gradient step ${iter} of ${TRAIN.length - 1}: push = 1 − σ(gap) = ${(1 - p).toFixed(3)}`}</Label>
                <Label x={24} y={300} tone={2} size={9}>{'The winner’s score rises and the loser’s falls by the same amount. As the model grows confident,'}</Label>
                <Label x={24} y={316} tone={3} size={9}>{'the push fades: a comparison it already gets right teaches it almost nothing.'}</Label>
              </>
            )}
            {st === 7 && (
              <>
                <Label x={24} y={282} tone={1} size={10} bold>{`shifted both by ${shift >= 0 ? '+' : ''}${shift.toFixed(1)}: gap still ${gap.toFixed(2)}, P still ${p.toFixed(3)}`}</Label>
                <Label x={24} y={300} tone={2} size={9}>{'σ only sees r_A − r_B, so the loss cannot pin down a zero. InstructGPT shifted its scores so the'}</Label>
                <Label x={24} y={316} tone={3} size={9}>{'labelers’ own demonstrations averaged 0 before using them for RL.'}</Label>
              </>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
