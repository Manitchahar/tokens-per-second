import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade } from '../lib/svg';

// "The cat sat on the ___", true token "mat". The slider sets p(mat); the other
// candidates share what is left in fixed proportions. The probabilities are
// illustrative; every number computed from them is exact.
const OTHERS: [string, number][] = [['floor', 0.3], ['sofa', 0.18], ['bed', 0.14], ['rug', 0.1], ['roof', 0.04], ['50,251 others', 0.24]];
const LN2 = Math.LN2;
const V_GPT2 = 50257;
const BIT_PX = 22; // horizontal pixels per bit in the bits-per-byte step

// Lesson worked example: four predictions in "The cat | sat on the mat"
const SENT = [
  { tok: 'sat', p: 0.1 },
  { tok: 'on', p: 0.5 },
  { tok: 'the', p: 0.8 },
  { tok: 'mat', p: 0.25 },
];
const SENT_LOSS = SENT.map((x) => -Math.log(x.p));
const SENT_MEAN = SENT_LOSS.reduce((a, b) => a + b, 0) / SENT.length;

// Two models that both get "mat" wrong at top-1
const MODEL_A: [string, number][] = [['mat', 0.4], ['floor', 0.45], ['sofa', 0.1], ['other', 0.05]];
const MODEL_B: [string, number][] = [['mat', 0.01], ['floor', 0.8], ['sofa', 0.12], ['other', 0.07]];

// Curve panel: p on x (0..1), loss in nats on y (0..5)
const CX0 = 360;
const CX1 = 610;
const CY0 = 262;
const CY1 = 72;
const px = (p: number) => CX0 + p * (CX1 - CX0);
const py = (l: number) => CY0 - (Math.min(l, 5) / 5) * (CY0 - CY1);
const CURVE = (() => {
  const pts: string[] = [];
  for (let k = 0; k <= 120; k++) {
    const p = Math.exp(Math.log(0.0068) + (k / 120) * (0 - Math.log(0.0068)));
    pts.push(`${px(p).toFixed(1)},${py(-Math.log(p)).toFixed(1)}`);
  }
  return pts.join(' ');
})();

// Illustrative training curve: falls from ln V toward a floor it never reaches.
const TX0 = 90;
const TX1 = 600;
const TY0 = 268;
const TY1 = 64;
const FLOOR = 1.8;
const trainLoss = (lg: number) => FLOOR + (Math.log(V_GPT2) - FLOOR) / (1 + Math.pow(10, 0.3 * (lg - 6.5)));
const tx = (lg: number) => TX0 + ((lg - 5) / 7) * (TX1 - TX0);
const ty = (l: number) => TY0 - (l / 11) * (TY0 - TY1);
const TRAIN = Array.from({ length: 71 }, (_, k) => 5 + k * 0.1).map((lg) => `${tx(lg).toFixed(1)},${ty(trainLoss(lg)).toFixed(1)}`).join(' ');

const STEPS = [
  'The model bets on every token',
  'Accuracy cannot see the difference',
  'Surprise: −log p',
  'Confident and wrong costs the most',
  'Average over every token',
  'Perplexity: the effective number of choices',
  'The floor: language’s own uncertainty',
  'Bits per byte: fair across tokenizers',
];
const CAPTIONS = [
  'For “The cat sat on the ___” the model hands back a probability for every token in its vocabulary. The text says the answer was “mat”. How do we score that bet?',
  'Two models, both with “floor” on top, so both are wrong by accuracy. But one gave the truth 40% and the other 1%. Accuracy scores them the same, and gives gradient descent nothing to push on.',
  'Score each bet by its surprise: −ln p(true token). p = 0.25 costs 1.386 nats, which is exactly 2 bits. Drag the slider and watch the dot ride the curve.',
  'Surprise is 0 when the model was certain and right, and climbs without limit as p(true) falls. Every halving of p adds the same 0.693 nats: one bit.',
  'A text is many predictions. Average their surprises and you have the cross-entropy loss. Here: (2.303 + 0.693 + 0.223 + 1.386) / 4 = 1.151 nats per token.',
  'Exponentiate the mean loss and you get perplexity: e^1.151 = 3.16. The model is as unsure as if it were choosing fairly between about 3 tokens at every step.',
  'Even a perfect model cannot reach zero, because the text itself is uncertain: many words could honestly come next. Training shrinks the gap to that floor, never below it.',
  'Per-token numbers depend on the tokenizer. Count bits per byte of raw text instead: model A wins at 1.08 bits per byte, even though model B’s per-token perplexity looks nine times better.',
];

function DistBars({ x, y, rows, truth, w = 170, rowH = 22, labelW = 92 }: { x: number; y: number; rows: [string, number][]; truth: string; w?: number; rowH?: number; labelW?: number }) {
  return (
    <g>
      {rows.map(([name, p], i) => {
        const hot = name === truth;
        return (
          <g key={name}>
            <Label x={x + labelW - 8} y={y + i * rowH + 12} anchor="end" tone={hot ? 1 : 2} size={10} bold={hot}>{name}</Label>
            <rect x={x + labelW} y={y + i * rowH + 2} width={w} height={14} rx={2} fill="none" stroke="var(--rule)" />
            <rect x={x + labelW} y={y + i * rowH + 2} width={Math.max(0.5, p * w)} height={14} rx={2}
              fill={hot ? 'var(--token)' : 'var(--ground-3)'} stroke={hot ? 'var(--token)' : 'var(--rule-2)'}
              style={{ transition: 'width .45s var(--ease-out)' }} />
            <Label x={x + labelW + w + 8} y={y + i * rowH + 13} tone={hot ? 1 : 3} size={9.5}>{`${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`}</Label>
          </g>
        );
      })}
    </g>
  );
}

export default function SurpriseMeter() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [p, setP] = useState(0.25);

  const loss = -Math.log(p);
  const bits = loss / LN2;
  const ppl = 1 / p;
  const rows: [string, number][] = [['mat', p], ...OTHERS.map(([n, w]) => [n, (1 - p) * w] as [string, number])];
  const dot = useTween(p, 450);
  const dotL = -Math.log(dot);

  const sliderOn = st === 0 || st === 2 || st === 3 || st === 5;
  const fairK = Math.min(20, Math.max(1, Math.round(ppl)));

  // bits per byte comparison
  const A_LOSS = 3.0; // nats per token, ~4 bytes per token
  const B_LOSS = 0.8; // nats per byte-token
  const aBpb = A_LOSS / LN2 / 4;
  const bBpb = B_LOSS / LN2;

  const readout =
    st === 1
      ? [
          { label: 'model A: p(mat)', value: '40%', kind: 'token' as const, note: 'top-1 is “floor”: wrong' },
          { label: 'model B: p(mat)', value: '1%', kind: 'token' as const, note: 'top-1 is “floor”: wrong' },
          { label: 'accuracy', value: '0 vs 0', kind: 'error' as const, note: 'cannot tell them apart' },
        ]
      : st === 4
        ? [
            { label: 'mean loss', value: `${SENT_MEAN.toFixed(3)} nats`, kind: 'error' as const, note: 'cross-entropy per token' },
            { label: 'in bits', value: `${(SENT_MEAN / LN2).toFixed(3)} bits`, kind: 'error' as const },
            { label: 'perplexity', value: Math.exp(SENT_MEAN).toFixed(2), kind: 'text' as const, note: 'e^mean' },
          ]
        : st === 6
          ? [
              { label: 'untrained (uniform)', value: `${Math.log(V_GPT2).toFixed(2)} nats`, kind: 'error' as const, note: 'ln 50,257' },
              { label: 'floor', value: 'entropy of the text', kind: 'idle' as const, note: 'unknown, but above 0' },
            ]
          : st === 7
            ? [
                { label: 'A: per-token PPL', value: Math.exp(A_LOSS).toFixed(1), kind: 'text' as const, note: '3.0 nats/token, 4 bytes/token' },
                { label: 'B: per-token PPL', value: Math.exp(B_LOSS).toFixed(2), kind: 'text' as const, note: '0.8 nats/token, 1 byte/token' },
                { label: 'A: bits per byte', value: aBpb.toFixed(3), kind: 'token' as const, note: 'lower is better: A wins' },
                { label: 'B: bits per byte', value: bBpb.toFixed(3), kind: 'error' as const },
              ]
            : [
                { label: 'p(true token)', value: p.toFixed(2), kind: 'token' as const },
                { label: 'loss = −ln p', value: st >= 2 ? `${loss.toFixed(3)} nats` : '—', kind: 'error' as const, note: st >= 2 ? `${bits.toFixed(2)} bits` : 'not defined yet' },
                { label: 'perplexity', value: st >= 5 ? ppl.toFixed(2) : '—', kind: 'text' as const, note: st >= 5 ? 'e^loss = 1/p' : 'step 6' },
              ];

  return (
    <Stage
      rootRef={ref}
      title="Scoring a bet: surprise, perplexity, bits per byte"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['token', 'p(true token)'], ['error', 'loss (surprise)'], ['idle', 'unreachable']]}
      {...s}
      readout={<Readout items={readout} />}
      controls={sliderOn ? (
        <Slider label="Probability the model gave the true token “mat”" value={p} min={0.01} max={0.99} step={0.01} onChange={setP}
          format={(v) => v.toFixed(2)} kind="token" />
      ) : undefined}
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="A model's probability bar for the true next token, a curve of surprise −ln p that the bar's value rides along, per-token losses averaged into cross-entropy, perplexity shown as an equivalent number of fair choices, an illustrative training curve that flattens above zero, and two tokenizers compared by bits per byte.">
        {/* ── the distribution (steps 0, 2, 3, 5) ── */}
        <Fade show={st === 0 || st === 2 || st === 3 || st === 5}>
          <Label x={30} y={34} tone={1} size={11} mono={false} bold>{'“The cat sat on the ___”'}</Label>
          <Label x={30} y={52} tone={3} size={9.5} mono={false}>the text says: mat</Label>
          <DistBars x={30} y={70} rows={rows} truth="mat" />
        </Fade>

        {/* step 0: what we want to score */}
        <Fade show={st === 0}>
          <Label x={380} y={110} tone={1} size={11} bold mono={false}>One number per bet</Label>
          <Label x={380} y={134} tone={2} size={10} mono={false}>The model never says “mat”. It says</Label>
          <Label x={380} y={150} tone={2} size={10} mono={false}>how much it would bet on every token.</Label>
          <Label x={380} y={176} tone={2} size={10} mono={false}>Gradient descent needs one number</Label>
          <Label x={380} y={192} tone={2} size={10} mono={false}>saying how bad this bet was, that</Label>
          <Label x={380} y={208} tone={2} size={10} mono={false}>moves smoothly when the bet moves.</Label>
          <Label x={380} y={240} tone={3} size={9.5} mono={false}>Only the green bar is “right”, so the score</Label>
          <Label x={380} y={255} tone={3} size={9.5} mono={false}>should depend on how big the green bar is.</Label>
        </Fade>

        {/* ── step 1: accuracy fails ── */}
        <Fade show={st === 1}>
          <Label x={40} y={40} tone={1} size={11} bold mono={false}>Model A</Label>
          <DistBars x={30} y={56} rows={MODEL_A} truth="mat" w={140} labelW={64} />
          <Label x={350} y={40} tone={1} size={11} bold mono={false}>Model B</Label>
          <DistBars x={340} y={56} rows={MODEL_B} truth="mat" w={140} labelW={64} />
          <Label x={40} y={168} kind="error" size={10.5}>{'top-1 = floor ≠ mat → accuracy 0'}</Label>
          <Label x={350} y={168} kind="error" size={10.5}>{'top-1 = floor ≠ mat → accuracy 0'}</Label>
          <rect x={60} y={196} width={520} height={74} rx={4} fill="var(--error-dim)" stroke="var(--error)" />
          <Label x={320} y={220} anchor="middle" tone={1} size={10.5} mono={false}>Same score. But A put 40× more on the truth than B did.</Label>
          <Label x={320} y={240} anchor="middle" tone={2} size={10} mono={false}>And nudging a weight rarely changes which token is on top, so accuracy</Label>
          <Label x={320} y={256} anchor="middle" tone={2} size={10} mono={false}>is flat almost everywhere: it gives gradient descent no slope to follow.</Label>
        </Fade>

        {/* ── curve panel (steps 2, 3) ── */}
        <Fade show={st === 2 || st === 3}>
          <line x1={CX0} x2={CX1} y1={CY0} y2={CY0} className="v-rule" />
          <line x1={CX0} x2={CX0} y1={CY0} y2={CY1 - 6} className="v-rule" />
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={px(v)} x2={px(v)} y1={CY0} y2={CY0 + 4} className="v-rule" />
              <Label x={px(v)} y={CY0 + 16} anchor="middle" tone={3} size={9}>{v.toString()}</Label>
            </g>
          ))}
          {[1, 2, 3, 4, 5].map((v) => (
            <g key={v}>
              <line x1={CX0 - 4} x2={CX0} y1={py(v)} y2={py(v)} className="v-rule" />
              <Label x={CX0 - 8} y={py(v) + 3} anchor="end" tone={3} size={9}>{v.toString()}</Label>
            </g>
          ))}
          <Label x={(CX0 + CX1) / 2} y={CY0 + 32} anchor="middle" tone={3} size={9.5}>p(true token)</Label>
          <Label x={CX0 + 6} y={CY1 - 10} tone={3} size={9.5}>loss, nats</Label>
          <polyline points={CURVE} fill="none" stroke="var(--error)" strokeWidth={2} />
          <Label x={400} y={135} kind="error" size={11} bold>−ln p</Label>
          <line x1={px(dot)} x2={px(dot)} y1={CY0} y2={py(dotL)} stroke="var(--token)" strokeDasharray="3 3" />
          <line x1={CX0} x2={px(dot)} y1={py(dotL)} y2={py(dotL)} stroke="var(--error)" strokeDasharray="3 3" />
          <circle cx={px(dot)} cy={py(dotL)} r={5} fill="var(--error)" stroke="var(--ground-2)" strokeWidth={1.5} />
        </Fade>
        <Fade show={st === 3}>
          {[[0.5, '0.69'], [0.25, '1.39'], [0.125, '2.08'], [0.01, '4.61']].map(([pv, lv]) => (
            <g key={String(pv)}>
              <circle cx={px(pv as number)} cy={py(-Math.log(pv as number))} r={3} fill="var(--text)" />
              <Label x={px(pv as number) + 8} y={py(-Math.log(pv as number)) - 5} tone={2} size={9}>{`p=${pv} → ${lv}`}</Label>
            </g>
          ))}
          <Label x={30} y={246} tone={1} size={10} bold mono={false}>Every halving of p adds ln 2 = 0.693 nats</Label>
          <Label x={30} y={262} tone={2} size={10} mono={false}>= exactly 1 bit. p = 1 costs nothing; p → 0 costs</Label>
          <Label x={30} y={278} tone={2} size={10} mono={false}>without limit. Confident and wrong is the worst.</Label>
        </Fade>
        <Fade show={st === 2}>
          <Label x={30} y={246} tone={1} size={10} bold mono={false}>surprise = −ln p(true token)</Label>
          <Label x={30} y={262} tone={2} size={10} mono={false}>{'p = 0.25 → −ln 0.25 = 1.386 nats = 2 bits'}</Label>
          <Label x={30} y={278} tone={3} size={9.5} mono={false}>Use ln for nats, log₂ for bits: 1 nat = 1.443 bits.</Label>
        </Fade>

        {/* ── step 4: average over tokens ── */}
        <Fade show={st === 4}>
          <Label x={40} y={36} tone={2} size={10.5}>context →</Label>
          {['The', 'cat'].map((w, i) => (
            <g key={w}>
              <rect x={118 + i * 52} y={22} width={46} height={22} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x={141 + i * 52} y={37} anchor="middle" tone={1} size={10.5}>{w}</Label>
            </g>
          ))}
          {SENT.map((x, i) => {
            const cx = 250 + i * 92;
            const l = SENT_LOSS[i];
            const bh = (l / 2.5) * 120;
            return (
              <g key={x.tok}>
                <rect x={cx - 26} y={22} width={52} height={22} rx={3} fill="var(--token)" stroke="var(--token)" />
                <Label x={cx} y={37} anchor="middle" size={10.5} bold style={{ fill: '#0e1116' }}>{x.tok}</Label>
                <Label x={cx} y={62} anchor="middle" kind="token" size={9.5}>{`p = ${x.p.toFixed(2)}`}</Label>
                <rect x={cx - 18} y={230 - bh} width={36} height={bh} rx={2} fill="var(--error)" style={{ transition: 'height .5s' }} />
                <Label x={cx} y={224 - bh} anchor="middle" tone={1} size={9.5}>{l.toFixed(3)}</Label>
              </g>
            );
          })}
          <line x1={210} x2={600} y1={230} y2={230} className="v-rule" />
          <line x1={210} x2={600} y1={230 - (SENT_MEAN / 2.5) * 120} y2={230 - (SENT_MEAN / 2.5) * 120} stroke="var(--text)" strokeDasharray="5 4" />
          <Label x={204} y={234 - (SENT_MEAN / 2.5) * 120} anchor="end" tone={1} size={9.5}>{`mean ${SENT_MEAN.toFixed(3)}`}</Label>
          <Label x={40} y={130} tone={2} size={10} mono={false}>surprise of</Label>
          <Label x={40} y={145} tone={2} size={10} mono={false}>each target,</Label>
          <Label x={40} y={160} tone={2} size={10} mono={false}>in nats</Label>
          <Label x={40} y={262} tone={1} size={10.5}>{'(2.303 + 0.693 + 0.223 + 1.386) / 4 = 1.151 nats per token'}</Label>
          <Label x={40} y={284} tone={3} size={9.5} mono={false}>This average is the cross-entropy loss. Training lowers it; nothing else.</Label>
        </Fade>

        {/* ── step 5: perplexity as fair choices ── */}
        <Fade show={st === 5}>
          <Label x={380} y={40} tone={1} size={11} bold mono={false}>As surprised as a fair choice among…</Label>
          <Label x={380} y={60} tone={2} size={10}>{`1 / p = ${ppl.toFixed(2)} options`}</Label>
          {Array.from({ length: fairK }, (_, i) => {
            const bw = Math.min(40, 220 / fairK - 3);
            const h = Math.min(130, 130 / Math.max(1, ppl) * 1);
            return <rect key={i} x={380 + i * (bw + 3)} y={210 - h * 1} width={bw} height={h} rx={2} fill="var(--ground-3)" stroke="var(--text-2)" className="v-cell" />;
          })}
          <line x1={378} x2={612} y1={210} y2={210} className="v-rule" />
          <Label x={380} y={226} tone={3} size={9.5} mono={false}>{fairK >= 20 && ppl > 20.5 ? 'equal bars (first 20 shown)' : 'equal bars, each 1/perplexity tall'}</Label>
          <Label x={30} y={246} tone={1} size={10} bold mono={false}>perplexity = e^(mean loss)</Label>
          <Label x={30} y={262} tone={2} size={10} mono={false}>For one token that is just 1/p. For the sentence</Label>
          <Label x={30} y={278} tone={2} size={10} mono={false}>of step 5: e^1.151 = 3.16 choices per token.</Label>
          <Label x={30} y={302} tone={3} size={9.5} mono={false}>Untrained, uniform over 50,257 tokens: perplexity 50,257.</Label>
        </Fade>

        {/* ── step 6: the floor ── */}
        <Fade show={st === 6}>
          <rect x={TX0} y={ty(FLOOR)} width={TX1 - TX0} height={TY0 - ty(FLOOR)} fill="url(#tps-hatch)" stroke="none" opacity={0.9} />
          <line x1={TX0} x2={TX1} y1={ty(FLOOR)} y2={ty(FLOOR)} stroke="var(--idle)" strokeWidth={1.5} strokeDasharray="6 4" />
          <line x1={TX0} x2={TX1} y1={TY0} y2={TY0} className="v-rule" />
          <line x1={TX0} x2={TX0} y1={TY0} y2={TY1 - 8} className="v-rule" />
          {[0, 2, 4, 6, 8, 10].map((v) => (
            <g key={v}>
              <line x1={TX0 - 4} x2={TX0} y1={ty(v)} y2={ty(v)} className="v-rule" />
              <Label x={TX0 - 8} y={ty(v) + 3} anchor="end" tone={3} size={9}>{String(v)}</Label>
            </g>
          ))}
          <line x1={TX0} x2={TX1} y1={ty(Math.log(V_GPT2))} y2={ty(Math.log(V_GPT2))} stroke="var(--error)" strokeDasharray="2 4" />
          <Label x={TX1} y={ty(Math.log(V_GPT2)) - 6} anchor="end" kind="error" size={9.5}>{'untrained: ln 50,257 = 10.82'}</Label>
          <polyline points={TRAIN} fill="none" stroke="var(--error)" strokeWidth={2} />
          <Label x={TX0 + 8} y={ty(FLOOR) - 8} tone={2} size={10} mono={false}>floor: the text’s own entropy. No model gets below it.</Label>
          <Label x={(TX0 + TX1) / 2} y={TY0 + 18} anchor="middle" tone={3} size={9.5}>training tokens seen (log scale) →</Label>
          <Label x={TX0 + 4} y={TY1 - 14} tone={3} size={9.5}>loss, nats per token</Label>
          <Label x={TX1} y={TY1 + 40} anchor="end" tone={3} size={9} mono={false}>illustrative shape, not a real run</Label>
          <Label x={TX1} y={TY1 + 56} anchor="end" tone={2} size={9.5} mono={false}>“I ate a ___”: apple? sandwich? Truly unknown.</Label>
        </Fade>

        {/* ── step 7: bits per byte ── */}
        <Fade show={st === 7}>
          <Label x={30} y={34} tone={1} size={11} bold mono={false}>The same 16 bytes of text, “the cat sat down”, under two tokenizers</Label>
          <Label x={30} y={64} tone={2} size={10} mono={false}>A · ~50K vocab</Label>
          {['the', ' cat', ' sat', ' down'].map((tk, i) => (
            <g key={tk}>
              <rect x={140 + i * 112} y={50} width={108} height={22} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x={194 + i * 112} y={65} anchor="middle" tone={1} size={10.5}>{tk.replace(' ', '·')}</Label>
            </g>
          ))}
          {Array.from({ length: 4 }, (_, i) => (
            <rect key={i} x={140 + i * (A_LOSS / LN2) * BIT_PX} y={80} width={(A_LOSS / LN2) * BIT_PX - 2} height={12} rx={2} fill="var(--error)" />
          ))}
          <Label x={146 + 4 * (A_LOSS / LN2) * BIT_PX} y={90} tone={1} size={9.5}>{`${(4 * A_LOSS / LN2).toFixed(1)} bits`}</Label>
          <Label x={140} y={110} tone={3} size={9.5}>{`4 tokens × ${(A_LOSS / LN2).toFixed(2)} bits each`}</Label>
          <Label x={30} y={146} tone={2} size={10} mono={false}>B · bytes</Label>
          {'the cat sat down'.split('').map((ch, i) => (
            <g key={i}>
              <rect x={140 + i * 28} y={132} width={25} height={22} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x={152.5 + i * 28} y={147} anchor="middle" tone={1} size={10}>{ch === ' ' ? '·' : ch}</Label>
            </g>
          ))}
          {Array.from({ length: 16 }, (_, i) => (
            <rect key={i} x={140 + i * (B_LOSS / LN2) * BIT_PX} y={162} width={(B_LOSS / LN2) * BIT_PX - 2} height={12} rx={2} fill="var(--error)" />
          ))}
          <Label x={146 + 16 * (B_LOSS / LN2) * BIT_PX} y={172} tone={1} size={9.5}>{`${(16 * B_LOSS / LN2).toFixed(1)} bits`}</Label>
          <Label x={140} y={192} tone={3} size={9.5}>{`16 tokens × ${(B_LOSS / LN2).toFixed(2)} bits each`}</Label>
          <line x1={30} x2={610} y1={206} y2={206} className="v-rule faint" />
          <Label x={30} y={230} tone={2} size={10} mono={false}>Per token, B looks far better: perplexity 2.23 against A’s 20.1. But B’s tokens are single bytes,</Label>
          <Label x={30} y={246} tone={2} size={10} mono={false}>each one easy to guess. Divide total bits by bytes of text and the comparison is fair:</Label>
          <Label x={30} y={274} kind="token" size={10.5} bold>{`A: ${(4 * A_LOSS / LN2).toFixed(2)} / 16 = ${aBpb.toFixed(3)} bits per byte   ← better`}</Label>
          <Label x={30} y={294} kind="error" size={10.5}>{`B: ${(16 * B_LOSS / LN2).toFixed(2)} / 16 = ${bBpb.toFixed(3)} bits per byte`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
