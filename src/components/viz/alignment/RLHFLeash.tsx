import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, Arrow } from '../lib/svg';

// A toy model of optimising against a learned reward, in the shape Gao et al. (2022) measured:
// d = √KL(policy ‖ SFT). The proxy (the reward model's score) keeps rising with d; the true
// quality people would assign rises, peaks and falls. The KL penalty β·KL = β·d² sets where the
// policy settles: maximising d − β d² gives d* = 1 / (2β). The gold curve uses the RL form
// they fitted, d(α − b·ln d), with a peak at d = 2.5. All units are illustrative.
const proxy = (d: number) => d;
const gold = (d: number) => (d <= 0 ? 0 : d * (1.82 - 0.95 * Math.log(d)));
const D_MAX = 8;
const D_PEAK = 2.5;
const GOLD_PEAK = gold(D_PEAK);
const dStar = (beta: number) => Math.min(D_MAX, 1 / (2 * beta));

const STEPS = [
  'Sample, score, nudge',
  'The reward goes up',
  'Keep pushing: it keeps going up',
  'True quality turns down',
  'The KL leash',
  'Choose the strength β',
  'PPO-ptx: keep the old skills',
  'What it bought',
];
const CAPTIONS = [
  'RLHF is a loop: the policy writes an answer, the reward model scores it, and the policy is nudged to make high-scoring answers more likely.',
  'Early on, every nudge that raises the reward also makes answers genuinely better. The policy moves away from the SFT model it started as.',
  'Keep optimising and the reward model’s score keeps climbing. The further the policy drifts, the higher it goes. That should worry you.',
  'Plot what people actually think and it rises, peaks, and falls. Past the peak the policy is exploiting the reward model’s mistakes: reward hacking.',
  'The fix is a leash: subtract β times the KL divergence between the policy and the SFT model from the reward. Drifting far now costs something.',
  'β sets where the policy stops. Too small and it overshoots into hacking; too large and it barely moves. InstructGPT found values around 0.01–0.02 worked best.',
  'RLHF also costs skills the reward model never checks: an alignment tax on standard benchmarks. Mixing pretraining gradients back in (PPO-ptx) reduces it.',
  'Labelers preferred the 1.3B InstructGPT to the 175B GPT-3. The 175B InstructGPT won 85 ± 3% of comparisons against GPT-3 itself.',
];

const SAMPLES = [
  { d: 0, text: 'A KV cache stores keys and values from earlier tokens.' },
  { d: 2.5, text: 'A KV cache stores each token’s keys and values, so the next token reuses them instead of recomputing attention over the whole prefix.' },
  { d: 5, text: 'Great question! A KV cache stores keys and values so generation is faster. It is a crucial, powerful and important technique!' },
  { d: 8, text: 'Great question!! Absolutely — the KV cache is an incredibly powerful, essential, crucial, amazing technique. I hope this helps! Let me know!!' },
];
const sampleAt = (d: number) => SAMPLES.reduce((best, s) => (Math.abs(s.d - d) < Math.abs(best.d - d) ? s : best), SAMPLES[0]);

// plot geometry
const PX = 330;
const PW = 280;
const PY = 236;
const PH = 168;
const YMIN = -1.6;
const YMAX = 8.2;
const gx = (d: number) => PX + (d / D_MAX) * PW;
const gy = (v: number) => PY - ((Math.max(YMIN, Math.min(YMAX, v)) - YMIN) / (YMAX - YMIN)) * PH;
const path = (f: (d: number) => number, d1 = D_MAX, d0 = 0) => {
  let s = '';
  for (let i = 0; i <= 80; i++) {
    const d = d0 + ((d1 - d0) * i) / 80;
    s += `${i === 0 ? 'M' : 'L'}${gx(d).toFixed(1)},${gy(f(d)).toFixed(1)} `;
  }
  return s;
};

export default function RLHFLeash() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [beta, setBeta] = useState(0.05);
  useEffect(() => { setBeta(st >= 5 ? 0.2 : 0.05); }, [st]);

  // how far the policy has drifted at each step
  const target = st === 0 ? 0 : st === 1 ? D_PEAK : st === 2 || st === 3 ? 8 : dStar(beta);
  const d = useTween(target, 1400);
  const showLoop = st === 0;
  const showPlot = st >= 1 && st <= 5;
  const showGold = st >= 3;
  const showKL = st >= 4;
  const showTax = st === 6;
  const showResult = st === 7;
  const sample = sampleAt(d);
  const kl = d * d;

  return (
    <Stage
      rootRef={ref}
      title="RLHF: chasing a learned reward, on a leash"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['compute', 'reward-model score'], ['token', 'true quality'], ['error', 'reward hacking'], ['comm', 'KL penalty']]}
      {...s}
      readout={
        showPlot ? (
          <Readout items={[
            { label: 'drift from SFT, KL', value: `${kl.toFixed(1)} nats`, kind: 'comm', note: `d = √KL = ${d.toFixed(2)}` },
            { label: 'reward-model score', value: proxy(d).toFixed(2), kind: 'compute' },
            { label: 'true quality', value: st >= 3 ? gold(d).toFixed(2) : '?', kind: st >= 3 && d > D_PEAK + 0.05 ? 'error' : 'token', note: st >= 3 ? `peak ${GOLD_PEAK.toFixed(2)} at d = ${D_PEAK}` : 'not measured yet' },
          ]} />
        ) : undefined
      }
      controls={
        st === 5 ? (
          <Slider label="KL coefficient β — how short the leash is" value={beta} min={0.03} max={0.6} step={0.01} onChange={setBeta} format={(v) => `β = ${v.toFixed(2)}`} kind="comm" />
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="The RLHF loop, then a chart of reward-model score and true quality against distance from the SFT model. The score keeps rising while true quality peaks and falls; a KL penalty with coefficient beta sets where the policy stops. Final panels show the alignment tax, PPO-ptx, and InstructGPT's preference results.">
        {/* ── the loop ── */}
        <Fade show={showLoop}>
          <g>
            {[
              { x: 40, y: 70, w: 120, label: 'prompt x', sub: 'from 31,144 API prompts' },
              { x: 250, y: 70, w: 140, label: 'policy π_RL', sub: 'starts as the SFT model' },
              { x: 470, y: 70, w: 130, label: 'answer y', sub: 'sampled, token by token' },
              { x: 470, y: 180, w: 130, label: 'reward model', sub: 'score r(x, y)' },
              { x: 250, y: 180, w: 140, label: 'PPO update', sub: 'make high scores likelier' },
            ].map((b) => (
              <g key={b.label}>
                <rect x={b.x} y={b.y} width={b.w} height={46} rx={4} fill={b.label === 'reward model' ? 'var(--compute-dim)' : 'var(--ground-3)'} stroke={b.label === 'reward model' ? 'var(--compute)' : 'var(--rule-2)'} />
                <text x={b.x + b.w / 2} y={b.y + 20} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fill: 'var(--text)', fontWeight: 700 }}>{b.label}</text>
                <text x={b.x + b.w / 2} y={b.y + 35} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{b.sub}</text>
              </g>
            ))}
            <Arrow x1={162} y1={93} x2={246} y2={93} />
            <Arrow x1={392} y1={93} x2={466} y2={93} kind="token" />
            <Arrow x1={535} y1={118} x2={535} y2={176} />
            <Arrow x1={468} y1={203} x2={394} y2={203} kind="compute" />
            <Arrow x1={320} y1={178} x2={320} y2={120} curve={0} />
            <Label x={24} y={262} tone={1} size={10} bold>reinforcement learning: no correct answer, only a score</Label>
            <Label x={24} y={280} tone={2} size={9}>{'The policy generates its own answers; the reward model grades them; good grades become more likely.'}</Label>
            <Label x={24} y={296} tone={2} size={9}>{'No human is in this loop. The reward model stands in for them, millions of times.'}</Label>
            <Label x={24} y={312} tone={3} size={9}>{'Every pass round the loop starts with generation: RLHF is, in wall-clock terms, mostly inference.'}</Label>
          </g>
        </Fade>

        {/* ── the over-optimisation chart ── */}
        <Fade show={showPlot}>
          <g>
            {/* sample output on the left */}
            <Label x={24} y={26} tone={3} size={9}>{`what the policy writes now (KL ≈ ${kl.toFixed(0)} nats)`}</Label>
            <foreignObject x={24} y={34} width={286} height={120}>
              <div style={{ font: '12px/1.45 var(--font-sans)', color: sample.d > 3 && st >= 3 ? 'var(--error)' : 'var(--text-2)', border: '1px solid var(--rule-2)', borderRadius: 4, padding: '8px 10px', background: 'var(--ground-3)', height: 104, overflow: 'hidden' }}>
                {sample.text}
              </div>
            </foreignObject>
            <Label x={24} y={176} tone={2} size={9}>{st < 3 ? 'The reward model likes this more and more.' : sample.d > 3 ? 'Padded, flattering, repetitive: high score, bad answer.' : 'Clear and specific: the score and people agree.'}</Label>
            <Fade show={showKL}>
              <g>
                <Label x={24} y={204} tone={1} size={9.5} bold>{'maximise  r(x, y) − β · KL(π_RL ‖ π_SFT)'}</Label>
                <Label x={24} y={222} tone={2} size={9}>{`β = ${beta.toFixed(2)}: the penalty is β·d², best d = 1/(2β) in these units`}</Label>
              </g>
            </Fade>

            {/* axes */}
            <line x1={PX} x2={PX + PW} y1={gy(0)} y2={gy(0)} className="v-rule" />
            <line x1={PX} x2={PX} y1={PY - PH} y2={PY} className="v-rule" />
            <text x={PX - 4} y={gy(0) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>0</text>
            {[0, 2, 4, 6, 8].map((t) => (
              <text key={t} x={gx(t)} y={PY + 12} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{`KL ${t * t}`}</text>
            ))}
            <text x={PX + PW} y={PY + 25} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>{'distance from the SFT model, d = √KL →'}</text>
            <path d={path(proxy)} fill="none" stroke="var(--compute)" strokeWidth={1.8} />
            <text x={gx(7.9)} y={gy(proxy(7.9)) + 16} textAnchor="end" className="v-label" style={{ fontSize: 9, fill: 'var(--compute)' }}>reward-model score</text>
            <Fade show={showGold}>
              <g>
                <path d={path(gold)} fill="none" stroke="var(--token)" strokeWidth={1.8} />
                <path d={path(gold, D_MAX, D_PEAK)} fill="none" stroke="var(--error)" strokeWidth={2.2} strokeOpacity={d > D_PEAK + 0.05 ? 0.9 : 0} />
                <text x={gx(D_PEAK) + 6} y={gy(GOLD_PEAK) + 16} className="v-label" style={{ fontSize: 9, fill: 'var(--token)' }}>true quality peaks</text>
                <text x={gx(7.8)} y={gy(gold(7.8)) - 8} textAnchor="end" className="v-label" style={{ fontSize: 9, fill: 'var(--error)' }}>reward hacking</text>
              </g>
            </Fade>
            <Fade show={showKL}>
              <g>
                <path d={path((x) => beta * x * x, Math.min(D_MAX, Math.sqrt(YMAX / Math.max(beta, 0.01))))} fill="none" stroke="var(--comm)" strokeWidth={1.4} strokeDasharray="4 3" />
                <text x={gx(Math.min(D_MAX, Math.sqrt(YMAX / Math.max(beta, 0.01)))) - 4} y={PY - PH + 10} textAnchor="end" className="v-label" style={{ fontSize: 9, fill: 'var(--comm)' }}>KL penalty β·d²</text>
              </g>
            </Fade>
            {/* the policy */}
            <line x1={gx(d)} x2={gx(d)} y1={gy(0)} y2={gy(proxy(d))} stroke="var(--text-2)" strokeDasharray="2 3" />
            <circle cx={gx(d)} cy={gy(proxy(d))} r={4.5} fill="var(--compute)" />
            {showGold && <circle cx={gx(d)} cy={gy(gold(d))} r={4.5} fill={d > D_PEAK + 0.05 ? 'var(--error)' : 'var(--token)'} />}
            <text x={gx(d)} y={gy(0) - 4} textAnchor={d > 6 ? 'end' : 'start'} dx={d > 6 ? -4 : 4} className="v-label" style={{ fontSize: 9, fill: 'var(--text)' }}>policy</text>

            <Label x={24} y={290} tone={1} size={10} bold>
              {st === 1 ? 'early RL: score and quality rise together'
                : st === 2 ? 'the score has no ceiling: the reward model has never seen answers like these'
                : st === 3 ? 'Goodhart’s law: when a measure becomes a target, it stops being a good measure'
                : st === 4 ? 'the leash: every nat of drift from SFT costs reward'
                : `β = ${beta.toFixed(2)} stops the policy at KL ≈ ${kl.toFixed(1)} nats: true quality ${gold(d).toFixed(2)} of a possible ${GOLD_PEAK.toFixed(2)}`}
            </Label>
            <Label x={24} y={308} tone={2} size={9}>
              {st === 1 ? 'Answers get clearer and more on-topic. The policy is learning what the reward model learned from people.'
                : st === 2 ? 'Far from SFT, the policy finds outputs where the reward model is confidently wrong, and it keeps finding them.'
                : st === 3 ? 'The red stretch is optimisation pressure spent on the reward model’s errors, not on people’s preferences.'
                : st === 4 ? 'Near the SFT model, the reward model is on familiar ground. Far away, its scores cannot be trusted.'
                : beta < 0.14 ? 'Too weak: the policy overshoots the peak and starts hacking.' : beta > 0.4 ? 'Too strong: the policy barely leaves SFT and leaves quality on the table.' : 'About right: the policy stops near the peak of true quality.'}
            </Label>
            <Label x={24} y={324} tone={3} size={9}>{'Toy curves in the shape measured by Gao et al. (2022); in real runs the peak has to be found by evaluation.'}</Label>
          </g>
        </Fade>

        {/* ── alignment tax ── */}
        <Fade show={showTax}>
          <g>
            <Label x={24} y={26} tone={1} size={10} bold>the alignment tax: skills the reward model never looks at</Label>
            {[
              { name: 'GPT-3', mark: '●', col: 'var(--text-2)', head: 'the baseline', note: 'public NLP benchmarks: SQuAD, DROP, HellaSwag, WMT French→English' },
              { name: 'PPO', mark: '▼', col: 'var(--error)', head: 'performance regressions', note: 'the policy drifted away from skills the reward model never checks' },
              { name: 'PPO-ptx', mark: '▲', col: 'var(--token)', head: 'regressions largely removed', note: 'pretraining gradients mixed into every update; labeler preference kept' },
            ].map((b, i) => {
              const y = 56 + i * 54;
              return (
                <g key={b.name}>
                  <text x={24} y={y} className="v-label" style={{ fontSize: 10, fill: 'var(--text)', fontWeight: 700 }}>{b.name}</text>
                  <text x={110} y={y} className="v-label" style={{ fontSize: 10, fill: b.col, fontWeight: 700 }}>{`${b.mark} ${b.head}`}</text>
                  <text x={110} y={y + 17} className="v-label t3" style={{ fontSize: 9 }}>{b.note}</text>
                </g>
              );
            })}
            <Label x={24} y={232} tone={1} size={9.5} bold>{'PPO-ptx objective:  E[ r − β log(π_RL / π_SFT) ]  +  γ · E_pretrain[ log π_RL(x) ]'}</Label>
            <Label x={24} y={252} tone={2} size={9}>{'The second term is plain next-token loss on pretraining text, added to every RL update.'}</Label>
            <Label x={24} y={268} tone={2} size={9}>{'It pulls the policy back toward everything pretraining taught, not just toward SFT behaviour.'}</Label>
            <Label x={24} y={290} tone={3} size={9}>{'The paper reports that tightening the KL leash alone did not remove these regressions; the pretraining mix did.'}</Label>
          </g>
        </Fade>

        {/* ── results ── */}
        <Fade show={showResult}>
          <g>
            <Label x={24} y={26} tone={1} size={10} bold>InstructGPT: labelers’ preferences on the API prompt distribution</Label>
            {[
              { name: '175B InstructGPT vs 175B GPT-3', v: 0.85, ci: 0.03 },
              { name: '175B InstructGPT vs few-shot 175B GPT-3', v: 0.71, ci: 0.04 },
            ].map((b, i) => {
              const y = 58 + i * 52;
              return (
                <g key={b.name}>
                  <text x={24} y={y} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>{b.name}</text>
                  <rect x={24} y={y + 8} width={400} height={16} rx={2} fill="none" stroke="var(--rule)" />
                  <line x1={224} x2={224} y1={y + 4} y2={y + 28} className="v-rule" strokeDasharray="2 2" />
                  <rect x={24} y={y + 8} width={b.v * 400} height={16} rx={2} fill="var(--token)" style={{ transition: 'width .8s var(--ease-out)' }} />
                  <line x1={24 + (b.v - b.ci) * 400} x2={24 + (b.v + b.ci) * 400} y1={y + 16} y2={y + 16} stroke="#0e1116" strokeWidth={2} />
                  <text x={436} y={y + 21} className="v-label" style={{ fontSize: 10, fill: 'var(--token)', fontWeight: 700 }}>{`${Math.round(b.v * 100)} ± ${Math.round(b.ci * 100)}% win rate`}</text>
                </g>
              );
            })}
            <Label x={224} y={158} anchor="middle" tone={3} size={8.5}>50% = no preference</Label>
            <Label x={24} y={190} tone={1} size={10} bold>and the headline: 1.3B InstructGPT was preferred to 175B GPT-3</Label>
            <Label x={24} y={208} tone={2} size={9}>{'More than 100× fewer parameters. Same architecture; the difference is post-training on human data.'}</Label>
            <Label x={24} y={236} tone={2} size={9}>{'Also reported: fewer made-up facts on closed-domain tasks and better TruthfulQA scores; toxicity fell'}</Label>
            <Label x={24} y={252} tone={2} size={9}>{'about 25% only when asked to be respectful; no significant change on the bias benchmarks.'}</Label>
            <Label x={24} y={276} tone={3} size={9}>{'Error bars are 95% confidence intervals, as in the paper.'}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
