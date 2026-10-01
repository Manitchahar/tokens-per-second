import { useRef, useState } from 'react';
import { Stage, Readout, Slider, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Label, Fade, Ruler } from '../lib/svg';

const STEPS = [
  'The claim, pinned to one number',
  'Pin the setup, or admit you cannot',
  'Measure the rig noise first',
  'Is the effect bigger than the noise?',
  'Close is not reproduced',
  'Report it honestly either way',
];
const CAPTIONS = [
  'A reproduction target is one number with its setup attached. "Faster" is not a claim you can test.',
  'If you cannot hold the same GPU, model and workload, say so up front and aim at the mechanism instead.',
  'Repeat one arm several times before you touch the treatment. That spread c is your yardstick.',
  'Plot the claimed value against your measured band. Reproduced means the claim lands inside the interval you can defend.',
  'If the bands overlap, the honest verdict is inconclusive, not a small win.',
  'Both a clean reproduction and a clean failure are results. What is not acceptable is reporting one run as if it were several.',
];

type Stage1 = 'match' | 'different';

const W = 640;
const PLOT_X = 96;
const PLOT_W = 500;

/** Width in px for a throughput value, on a 0..2.5x scale. */
const sx = (ratio: number) => PLOT_X + Math.min(2.5, Math.max(0, ratio)) / 2.5 * PLOT_W;

export default function Reproduce() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [setup, setSetup] = useState<Stage1>('match');
  const [noise, setNoise] = useState(5);
  const [repeats, setRepeats] = useState(4);

  // The claim is "1.4x". We measure a mean with a spread of `noise`.
  const claim = 1.4;
  // What we actually measure depends on whether the setup matches.
  const measured = setup === 'match' ? 1.38 : 1.22;
  // Spread of the mean falls as repeats rise: noise/sqrt(k)
  const spread = (noise / Math.sqrt(Math.max(1, repeats))) / 100;
  const lo = measured * (1 - spread);
  const hi = measured * (1 + spread);

  // Three honest outcomes, in the order the lesson teaches them:
  // reproduced (claim inside the band), inconclusive (noise too wide to tell),
  // did not reproduce (the band excludes the claim).
  const inside = claim >= lo && claim <= hi;
  const canTell = spread < 0.06; // band tight enough to distinguish at all
  const verdict: 'reproduced' | 'inconclusive' | 'failed' = inside ? 'reproduced' : canTell ? 'failed' : 'inconclusive';

  const g = useTweens({ lo: sx(lo), hi: sx(hi), mean: sx(measured), claim: sx(claim) });

  return (
    <Stage
      rootRef={ref}
      title="Reproducing one number: claimed vs measured, against the noise"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'measured', ['compute', 'claimed'], ['error', 'does not reproduce'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Claim from the paper', value: `${claim.toFixed(2)}×`, kind: 'compute', note: 'the number to regenerate' },
          { label: 'Your measured mean', value: `${measured.toFixed(3)}×`, kind: 'token', note: `±${(spread * 100).toFixed(1)}% over ${repeats} runs` },
          { label: 'Verdict', value: verdict === 'reproduced' ? 'Reproduced' : verdict === 'inconclusive' ? 'Inconclusive' : 'Did not reproduce', kind: verdict === 'reproduced' ? 'token' : 'error', note: verdict === 'inconclusive' ? 'noise too wide to tell' : 'claim inside the band?' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Setup" value={setup} onChange={setSetup} options={[{ value: 'match', label: 'Same GPU + workload' }, { value: 'different', label: 'Different GPU' }]} />
          <Slider label="Rig run-to-run spread" value={noise} min={1} max={20} onChange={setNoise} format={(v) => `${v}%`} kind="error" />
          <Slider label="Repeats of each arm" value={repeats} min={1} max={16} onChange={setRepeats} format={(v) => `${v}`} kind="compute" />
        </>
      }
    >
      <svg viewBox={`0 0 ${W} 372`} role="img" aria-label="A chart of throughput speedup where the paper's claimed value is compared against your measured mean and its uncertainty band, showing whether the claim falls inside the band.">
        <Label x={PLOT_X} y={18} tone={2} size={11} mono={false} bold>Throughput speedup vs the paper&apos;s baseline</Label>

        {/* axis */}
        <Ruler x={PLOT_X} y={150} w={PLOT_W} ticks={[{ at: 0, label: '0' }, { at: 0.4, label: '1.0×' }, { at: 0.8, label: '2.0×' }, { at: 1, label: '2.5×' }]} />
        <Label x={PLOT_X + PLOT_W} y={172} anchor="end" tone={3} size={9.5}>speedup over baseline →</Label>
        {/* baseline reference */}
        <line x1={sx(1)} x2={sx(1)} y1={40} y2={150} stroke="var(--rule-2)" strokeDasharray="3 3" />
        <Label x={sx(1)} y={34} anchor="middle" tone={3} size={9.5}>baseline (1.0×)</Label>

        {/* measured band */}
        <g className="v-move" style={{ transform: `translate(${g.lo}px, 0)` }}>
          <rect x={0} y={64} width={Math.max(1, g.hi - g.lo)} height={20} rx={3} fill="var(--token-dim)" stroke="var(--token)" />
        </g>
        <g className="v-move" style={{ transform: `translate(${g.mean}px, 0)` }}>
          <rect x={-1.5} y={58} width={3} height={32} rx={1} fill="var(--token)" />
        </g>

        {/* measured label */}
        <g className="v-move" style={{ transform: `translate(${g.mean}px, 0)` }}>
          <Label x={0} y={50} anchor="middle" size={10.5} kind="token" bold>you: {measured.toFixed(2)}×</Label>
          <text x={0} y={102} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--text-3)' }}>measured</text>
        </g>

        {/* the claim */}
        <g className="v-move" style={{ transform: `translate(${g.claim}px, 0)` }}>
          <line x1={0} x2={0} y1={44} y2={112} stroke="var(--compute)" strokeWidth={2} />
          <Label x={0} y={38} anchor="middle" size={10.5} kind="compute" bold>paper: {claim.toFixed(2)}×</Label>
        </g>

        {/* gap annotation */}
        <Fade show={s.step >= 3}>
          <line x1={g.claim} x2={g.claim} y1={118} y2={136} stroke={verdict === 'reproduced' ? 'var(--token)' : 'var(--error)'} strokeWidth={1} strokeDasharray="2 2" />
        </Fade>

        {/* verdict banner */}
        <Fade show={s.step >= 4}>
          <rect x={PLOT_X} y={196} width={PLOT_W} height={44} rx={3}
            fill={verdict === 'reproduced' ? 'var(--token-dim)' : 'var(--error-dim)'}
            stroke={verdict === 'reproduced' ? 'var(--token)' : 'var(--error)'} />
          <text x={PLOT_X + 12} y={214} className="v-label bold" style={{ fontSize: 12, fill: verdict === 'reproduced' ? 'var(--token)' : 'var(--error)' }}>
            {verdict === 'reproduced' ? 'REPRODUCED — the claim falls inside your measured band'
              : verdict === 'inconclusive' ? 'INCONCLUSIVE — your noise is too large to tell'
              : 'DID NOT REPRODUCE — your measured band does not contain the claim'}
          </text>
          <text x={PLOT_X + 12} y={231} className="v-label sans" style={{ fontSize: 10, fill: 'var(--text-2)' }}>
            {setup === 'match'
              ? `Setup matched. ${repeats} repeats, spread ${noise}% per run.`
              : 'Setup differs: different GPU. You are testing the mechanism, not the number.'}
          </text>
        </Fade>

        {/* how many repeats this effect needs */}
        <Fade show={s.step >= 3}>
          <Label x={PLOT_X} y={262} tone={2} size={10.5} mono={false} bold>Repeats needed to defend this effect</Label>
          {[
            { eff: 4, y: 280 },
            { eff: 10, y: 298 },
            { eff: 20, y: 316 },
            { eff: 30, y: 334 },
          ].map(({ eff, y }) => {
            const need = Math.ceil(Math.pow((2 * noise) / eff, 2));
            const enough = repeats >= need;
            return (
              <g key={eff}>
                <Label x={PLOT_X} y={y} tone={2} size={10.5}>{`effect ${eff}%`}</Label>
                <text x={PLOT_X + 74} y={y} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text-3)' }}>needs k = (2c/δ)² =</text>
                <text x={PLOT_X + 210} y={y} className="v-label bold" style={{ fontSize: 11, fill: enough ? 'var(--token)' : 'var(--error)' }}>{need}</text>
                <text x={PLOT_X + 244} y={y} className="v-label sans" style={{ fontSize: 10, fill: enough ? 'var(--token)' : 'var(--error)' }}>
                  {enough ? `you ran ${repeats}` : `you ran ${repeats}, not enough`}
                </text>
              </g>
            );
          })}
        </Fade>
      </svg>
    </Stage>
  );
}