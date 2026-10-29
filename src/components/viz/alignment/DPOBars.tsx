import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Label, Fade } from '../lib/svg';

const sig = (x: number) => 1 / (1 + Math.exp(-x));

// A toy training run on one preference pair. Log-ratios are log π(y|x) − log π_ref(y|x) in nats,
// summed over the answer's tokens. The shape is the one commonly reported for DPO: the margin
// grows steadily, the rejected answer's probability collapses, and the chosen one can fall too.
const traj = (t: number) => ({ w: 0.5 * t - 0.08 * t * t, l: -0.6 * t - 0.15 * t * t });
const T_MAX = 10;

// the closed-form optimum on a three-answer toy problem
const REF = [0.5, 0.3, 0.2];
const REW = [0, 1, 2];
const optimum = (beta: number) => {
  const w = REF.map((p, i) => p * Math.exp(REW[i] / beta));
  const z = w.reduce((a, b) => a + b, 0);
  return { p: w.map((v) => v / z), z };
};

const STEPS = [
  'Start: the policy is the reference',
  'RLHF’s target has a formula',
  'Flip it: a reward hiding in the policy',
  'In a comparison, Z cancels',
  'The DPO loss, on one pair',
  'Training pulls the log-ratios apart',
  'Both can fall: only the gap is trained',
  'What DPO skips',
];
const CAPTIONS = [
  'DPO starts where RLHF starts: the policy is a copy of the SFT model, frozen as the reference. Every log-ratio is 0, and the loss is ln 2 = 0.693.',
  'The best policy for “maximise reward minus β·KL to the reference” can be written down exactly: π* = π_ref · e^{r/β} / Z. Higher reward, more probability.',
  'Rearrange it and the reward is β·log(π*/π_ref) + β·log Z. Any policy defines an implicit reward: how much more likely it makes an answer than the reference does.',
  'Z is the same for both answers to a prompt, so it cancels in the Bradley-Terry difference. The preference probability depends only on the two log-ratios.',
  'So fit the policy to the preferences directly: loss = −log σ(β[ratio_w − ratio_l]). With ratios +1.0 and −0.5 and β = 0.1, the loss is 0.621.',
  'Each step raises the chosen answer’s log-ratio relative to the rejected one, weighted by σ(−margin): pairs the policy already gets right fade out.',
  'Nothing in the loss pins the chosen answer’s own probability. In practice the rejected one plunges, and the chosen one often drifts down too.',
  'No reward model, no value model, no sampling loop: two models, one forward pass each per pair, and a loss that looks like supervised learning.',
];

// geometry for the log-ratio bars
const AX0 = 64;
const AX1 = 320;
const RMIN = -22;
const RMAX = 4;
const rx = (v: number) => AX0 + ((Math.max(RMIN, Math.min(RMAX, v)) - RMIN) / (RMAX - RMIN)) * (AX1 - AX0);

// loss curve on the right
const PX = 380;
const PW = 230;
const PY = 230;
const PH = 130;
const MMIN = -2;
const MMAX = 3;
const mx = (m: number) => PX + ((Math.max(MMIN, Math.min(MMAX, m)) - MMIN) / (MMAX - MMIN)) * PW;
const ly = (l: number) => PY - Math.min(1, l / 2.2) * PH;
const LOSS_PATH = (() => {
  let s = '';
  for (let i = 0; i <= 80; i++) {
    const m = MMIN + ((MMAX - MMIN) * i) / 80;
    s += `${i ? 'L' : 'M'}${mx(m).toFixed(1)},${ly(-Math.log(sig(m))).toFixed(1)} `;
  }
  return s;
})();

export default function DPOBars() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [beta, setBeta] = useState(0.1);
  const [t, setT] = useState(0);

  useEffect(() => { setBeta(st === 1 ? 1 : 0.1); }, [st]);
  useEffect(() => {
    if (st === 5 || st === 6) {
      setT(0);
      let i = 0;
      const id = window.setInterval(() => { i += 1; setT(Math.min(T_MAX, i)); if (i >= T_MAX) window.clearInterval(id); }, 420);
      return () => window.clearInterval(id);
    }
    setT(0);
  }, [st]);

  // which log-ratios are shown
  const target = st === 0 ? { w: 0, l: 0 } : st === 4 ? { w: 1.0, l: -0.5 } : st >= 5 && st <= 6 ? traj(t) : { w: 1.0, l: -0.5 };
  const tw = useTweens(target, 380);
  const margin = beta * (tw.w - tw.l);
  const loss = -Math.log(sig(margin));
  const weight = sig(-margin);

  const showBars = st === 0 || (st >= 4 && st <= 6);
  const showCurve = st === 0 || (st >= 4 && st <= 6);
  const showOpt = st === 1;
  const showDeriv = st === 2 || st === 3;
  const showCompare = st === 7;
  const opt = optimum(beta);

  return (
    <Stage
      rootRef={ref}
      title="DPO: preference tuning as one supervised loss"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['token', 'chosen answer y_w'], ['error', 'rejected answer y_l'], ['compute', 'loss'], ['text', 'reference model']]}
      {...s}
      readout={
        showBars ? (
          <Readout items={[
            { label: 'log-ratio, chosen', value: tw.w.toFixed(2), kind: 'token', note: 'log π(y_w) − log π_ref(y_w)' },
            { label: 'log-ratio, rejected', value: tw.l.toFixed(2), kind: 'error', note: 'log π(y_l) − log π_ref(y_l)' },
            { label: 'margin β·(w − l)', value: margin.toFixed(3), kind: 'text' },
            { label: 'DPO loss', value: `${loss.toFixed(3)} nats`, kind: 'compute', note: `gradient weight σ(−margin) = ${weight.toFixed(3)}` },
          ]} />
        ) : showOpt ? (
          <Readout items={[
            { label: 'β', value: beta.toFixed(2), kind: 'text' },
            { label: 'Z (normaliser)', value: opt.z.toFixed(3), kind: 'text', note: 'Σ π_ref · e^{r/β}' },
            { label: 'π* on the best answer', value: opt.p[2].toFixed(3), kind: 'token', note: 'reference had 0.200' },
          ]} />
        ) : undefined
      }
      controls={
        st === 1 || st === 4 ? (
          <Slider label="β — strength of the pull back to the reference (smaller = weaker leash)" value={beta} min={st === 1 ? 0.25 : 0.02} max={st === 1 ? 4 : 0.5} step={0.01} onChange={setBeta} format={(v) => `β = ${v.toFixed(2)}`} />
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="Bars show the log-ratio of the policy to the reference model for a chosen and a rejected answer. During DPO training the margin between them grows and the loss −log σ(β·margin) falls; the rejected answer's log-ratio falls steeply while the chosen one rises and then drifts down.">
        {/* ── log-ratio bars ── */}
        <Fade show={showBars}>
          <g>
            <Label x={24} y={26} tone={2} size={9.5}>{'log-ratio = log π(y|x) − log π_ref(y|x), nats, summed over tokens'}</Label>
            <line x1={AX0} x2={AX1} y1={150} y2={150} className="v-rule" />
            <line x1={rx(0)} x2={rx(0)} y1={46} y2={156} className="v-rule" />
            {[-20, -15, -10, -5, 0].map((v) => (
              <text key={v} x={rx(v)} y={166} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{v}</text>
            ))}
            {[{ v: tw.w, y: 64, k: 'var(--token)', name: 'y_w' }, { v: tw.l, y: 104, k: 'var(--error)', name: 'y_l' }].map((b) => {
              const x0 = rx(0);
              const x1 = rx(b.v);
              return (
                <g key={b.name}>
                  <text x={AX0 - 8} y={b.y + 13} textAnchor="end" className="v-label" style={{ fontSize: 10, fill: b.k, fontWeight: 700 }}>{b.name}</text>
                  <rect x={Math.min(x0, x1)} y={b.y} width={Math.max(1, Math.abs(x1 - x0))} height={18} rx={2} fill={b.k} />
                  <text x={b.v >= 0 ? x1 + 5 : x1 - 5} y={b.y + 13} textAnchor={b.v >= 0 ? 'start' : 'end'} className="v-label" style={{ fontSize: 9.5, fill: b.k }}>{b.v.toFixed(2)}</text>
                </g>
              );
            })}
            {st >= 5 && st <= 6 && <Label x={24} y={192} tone={2} size={9}>{`training step ${t} of ${T_MAX}`}</Label>}
          </g>
        </Fade>

        {/* ── loss vs margin ── */}
        <Fade show={showCurve}>
          <g>
            <line x1={PX} x2={PX + PW} y1={PY} y2={PY} className="v-rule" />
            <line x1={mx(0)} x2={mx(0)} y1={PY - PH} y2={PY + 4} className="v-rule faint" />
            {[-2, -1, 0, 1, 2, 3].map((m) => (
              <text key={m} x={mx(m)} y={PY + 13} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{m}</text>
            ))}
            <text x={PX + PW} y={PY + 26} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>margin β·(ratio_w − ratio_l)</text>
            <path d={LOSS_PATH} fill="none" stroke="var(--compute)" strokeWidth={1.8} />
            <text x={PX + 4} y={PY - PH - 6} className="v-label" style={{ fontSize: 9, fill: 'var(--compute)' }}>loss = −log σ(margin)</text>
            <line x1={PX} x2={PX + PW} y1={ly(Math.LN2)} y2={ly(Math.LN2)} className="v-rule faint" strokeDasharray="3 3" />
            <text x={PX + PW} y={ly(Math.LN2) - 4} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>ln 2 = 0.693</text>
            <circle cx={mx(margin)} cy={ly(loss)} r={4.5} fill="var(--compute)" />
          </g>
        </Fade>

        {/* ── the closed-form optimum ── */}
        <Fade show={showOpt}>
          <g>
            <Label x={24} y={26} tone={1} size={10} bold>{'maximise  E[r] − β·KL(π ‖ π_ref)   ⇒   π*(y) = π_ref(y) · e^{r(y)/β} / Z'}</Label>
            <Label x={24} y={44} tone={2} size={9}>{'Toy prompt with three possible answers, rewards 0, 1 and 2.'}</Label>
            {[0, 1, 2].map((i) => {
              const x = 110 + i * 170;
              const hRef = REF[i] * 200;
              const hOpt = opt.p[i] * 200;
              return (
                <g key={i}>
                  <rect x={x} y={240 - hRef} width={40} height={hRef} rx={2} fill="none" stroke="var(--text-2)" strokeDasharray="3 3" />
                  <rect x={x + 46} y={240 - hOpt} width={40} height={hOpt} rx={2} fill={i === 2 ? 'var(--token)' : 'var(--compute)'} fillOpacity={0.85} style={{ transition: 'y .5s, height .5s' }} />
                  <text x={x + 20} y={234 - hRef} textAnchor="middle" className="v-label t2" style={{ fontSize: 9 }}>{REF[i].toFixed(2)}</text>
                  <text x={x + 66} y={234 - hOpt} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--text)' }}>{opt.p[i].toFixed(3)}</text>
                  <text x={x + 43} y={256} textAnchor="middle" className="v-label t2" style={{ fontSize: 9.5 }}>{`answer ${i + 1} · r = ${REW[i]}`}</text>
                </g>
              );
            })}
            <Label x={24} y={282} tone={1} size={9.5} bold>{`β = ${beta.toFixed(2)}: π* = (${opt.p.map((v) => v.toFixed(3)).join(', ')}), Z = ${opt.z.toFixed(3)}`}</Label>
            <Label x={24} y={300} tone={2} size={9}>{'Dashed: the reference. Solid: the optimum. Small β → chase the reward; large β → stay near the reference.'}</Label>
            <Label x={24} y={316} tone={3} size={9}>{'PPO only approximates this: Z sums over every possible answer, so it cannot be computed.'}</Label>
          </g>
        </Fade>

        {/* ── derivation ── */}
        <Fade show={showDeriv}>
          <g>
            <Label x={24} y={40} tone={2} size={9.5}>take logs of the optimum and solve for r</Label>
            <Label x={24} y={70} tone={1} size={12} math>{'r(x, y) = β · log [ π*(y|x) / π_ref(y|x) ]  +  β · log Z(x)'}</Label>
            <Label x={24} y={96} tone={2} size={9}>{'The first term is the implicit reward: β times how much more likely the policy makes y than the reference.'}</Label>
            <Fade show={st === 3}>
              <g>
                <Label x={24} y={136} tone={2} size={9.5}>{'put it into Bradley-Terry: P(y_w ≻ y_l) = σ(r_w − r_l)'}</Label>
                <Label x={24} y={166} tone={1} size={12} math>{'r_w − r_l = β·log(π*_w/π_ref,w) + β·log Z − β·log(π*_l/π_ref,l) − β·log Z'}</Label>
                <Label x={24} y={196} tone={2} size={9}>{'Both answers reply to the same x, so they share Z(x): +β·log Z − β·log Z = 0. The sum is gone.'}</Label>
                <Label x={24} y={226} tone={1} size={12} math>{'P(y_w ≻ y_l) = σ( β·log(π_w/π_ref,w) − β·log(π_l/π_ref,l) )'}</Label>
                <Label x={24} y={252} tone={2} size={9}>{'A preference probability written entirely in terms of the policy and the reference. No reward model.'}</Label>
              </g>
            </Fade>
            <Label x={24} y={300} tone={3} size={9}>{'Rafailov et al. (2023): “your language model is secretly a reward model”.'}</Label>
          </g>
        </Fade>

        {/* ── explanation strip for the bar steps ── */}
        {showBars && (
          <g>
            <Label x={24} y={262} tone={1} size={10} bold>
              {st === 0 ? 'at the start the policy is the reference: both ratios 0, margin 0, loss ln 2'
                : st === 4 ? `margin = ${beta.toFixed(2)} × (${tw.w.toFixed(2)} − (${tw.l.toFixed(2)})) = ${margin.toFixed(3)} → loss −ln σ(${margin.toFixed(3)}) = ${loss.toFixed(3)}`
                : st === 5 ? 'the gap grows, the loss falls, and each step’s push shrinks as σ(−margin)'
                : 'the chosen answer’s own log-ratio peaked and is now falling'}
            </Label>
            <Label x={24} y={280} tone={2} size={9}>
              {st === 0 ? 'Loss −log σ(0) = ln 2: the policy is a coin flip about which answer people preferred.'
                : st === 4 ? 'Drag β: it scales the margin, so it sets how hard each nat of log-ratio is pushed.'
                : st === 5 ? 'The rejected answer’s probability is driven down much faster than the chosen one is raised.'
                : 'DPO only trains the difference. Both answers can lose probability, as long as the loser loses more.'}
            </Label>
            <Label x={24} y={296} tone={3} size={9}>
              {st === 6 ? 'Freed probability goes to answers nobody labelled. Watch the chosen log-probs.' : 'Toy numbers on one pair; real runs average this loss over many thousands of pairs.'}
            </Label>
          </g>
        )}

        {/* ── what DPO skips ── */}
        <Fade show={showCompare}>
          <g>
            {[
              { x: 24, name: 'RLHF with PPO', items: [['policy (trained)', 'compute'], ['reference (frozen)', 'text'], ['reward model (frozen)', 'text'], ['value model (trained)', 'compute']], extra: 'sample answers every iteration → generation loop' },
              { x: 334, name: 'DPO', items: [['policy (trained)', 'compute'], ['reference (frozen)', 'text']], extra: 'fixed pairs (x, y_w, y_l) → no generation' },
            ].map((c) => (
              <g key={c.name}>
                <rect x={c.x} y={30} width={282} height={196} rx={5} fill="none" stroke="var(--rule-2)" />
                <text x={c.x + 14} y={52} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text)', fontWeight: 700 }}>{c.name}</text>
                {c.items.map(([n, k], i) => (
                  <g key={n}>
                    <rect x={c.x + 14} y={64 + i * 30} width={254} height={22} rx={3} fill={k === 'compute' ? 'var(--compute-dim)' : 'var(--ground-3)'} stroke={k === 'compute' ? 'var(--compute)' : 'var(--rule-2)'} />
                    <text x={c.x + 26} y={79 + i * 30} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>{n}</text>
                  </g>
                ))}
                <text x={c.x + 14} y={208} className="v-label" style={{ fontSize: 9, fill: c.name === 'DPO' ? 'var(--token)' : 'var(--error)' }}>{c.extra}</text>
              </g>
            ))}
            <Label x={24} y={256} tone={1} size={10} bold>{'7B in 16-bit weights: 4 × 14 GB = 56 GB for PPO, 2 × 14 GB = 28 GB for DPO'}</Label>
            <Label x={24} y={274} tone={2} size={9}>{'Reference log-probs never change: compute them once before training, leaving one model in memory.'}</Label>
            <Label x={24} y={290} tone={2} size={9}>{'What it gives up: it never sees its own new answers (offline), and is only as good as its pairs.'}</Label>
            <Label x={24} y={306} tone={3} size={9}>{'Same objective as KL-regularised RLHF in theory; different behaviour in practice. Neither always wins.'}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
