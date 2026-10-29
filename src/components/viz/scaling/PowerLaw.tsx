import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, Span } from '../lib/svg';
import { KAPLAN, kaplanN, kaplanD, count } from './scale';

// Kaplan et al. 2020: test loss against parameters, data and compute.
// On ordinary axes the curve is a cliff and a long flat tail; on log-log axes it is a
// straight line whose slope is the exponent. The plot morphs between the two so you can
// watch the straight line appear. Dots are the fitted law evaluated across the range the
// paper actually trained; the dashed part is extrapolation.

type VarId = 'N' | 'D' | 'C';
interface Var {
  id: VarId; label: string; unit: string; alpha: number; lo: number; hi: number; tLo: number; tHi: number;
  f: (x: number) => number; pts: number[]; fmtX: (x: number) => string; gpt3: number; gpt3Label: string;
}

const pfd = (x: number) => (x >= 1 ? `${count(x)} PF-day${count(x) === '1' ? '' : 's'}` : `10${sup(Math.round(Math.log10(x)))} PF-days`);
function sup(e: number) {
  const m: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  return String(e).split('').map((c) => m[c]).join('');
}

const VARS: Record<VarId, Var> = {
  N: {
    id: 'N', label: 'parameters N (non-embedding)', unit: 'params', alpha: KAPLAN.aN, lo: 1e2, hi: 1e12, tLo: 768, tHi: 1.5e9,
    f: kaplanN, pts: [768, 6e3, 5e4, 4e5, 3e6, 2.5e7, 2e8, 1.5e9], fmtX: (x) => count(x), gpt3: 1.74e11, gpt3Label: 'GPT-3, 174B',
  },
  D: {
    id: 'D', label: 'training tokens D', unit: 'tokens', alpha: KAPLAN.aD, lo: 1e6, hi: 1e13, tLo: 2.2e7, tHi: 2.3e10,
    f: kaplanD, pts: [2.2e7, 6e7, 1.8e8, 6e8, 2e9, 7e9, 2.3e10], fmtX: (x) => count(x), gpt3: 3e11, gpt3Label: 'GPT-3’s 300B tokens',
  },
  C: {
    id: 'C', label: 'compute C (PF-days)', unit: 'PF-days', alpha: KAPLAN.aC, lo: 1e-9, hi: 1e4, tLo: 1e-8, tHi: 10,
    f: (c) => Math.pow(KAPLAN.Cc / c, KAPLAN.aC), pts: [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], fmtX: pfd, gpt3: 3640, gpt3Label: 'GPT-3’s 3,640 PF-days',
  },
};

const STEPS = [
  'A ladder of models, plotted plainly',
  'Put model size on a log axis',
  'Log both axes: a straight line',
  'The slope is the exponent',
  'Data has its own straight line',
  'So does compute',
  'Extrapolate before you pay',
  'Kaplan’s advice: spend it on parameters',
];
const CAPTIONS = [
  'Train models from 768 to 1.5 billion parameters and plot their test loss on ordinary axes. Everything small is crushed against the left wall, everything big looks flat. You cannot read a trend from this.',
  'Put parameters on a log axis, where every step is ×10. The points spread out evenly. The curve still bends, so we are not done.',
  'Put loss on a log axis too, and the points fall on a straight line. A straight line on log-log axes is a power law: L = (Nc / N)^α.',
  'The slope is the exponent, α_N = 0.076. Every 10× more parameters multiplies the loss by 10^−0.076 = 0.839: a 16% drop, the same 16% at every scale.',
  'Do the same with training tokens and you get another straight line, with exponent α_D = 0.095. Every 10× more data multiplies loss by 0.804.',
  'And with compute, when each budget is spent on the best model for it: α_C = 0.050. Every 10× more compute multiplies loss by 0.891.',
  'The line was fitted on models up to 1.5B parameters. Extend it two orders of magnitude and it predicts the loss of a 174B model before anyone trains one.',
  'Kaplan et al. asked how to split a growing budget. Their fits said: grow parameters as C^0.73 and data only as C^0.27. For 10× compute, 5.4× the parameters and 1.9× the data.',
];

const PRESET: Record<number, { v: VarId; probe: number }> = {
  0: { v: 'N', probe: 1e8 }, 1: { v: 'N', probe: 1e8 }, 2: { v: 'N', probe: 1e8 }, 3: { v: 'N', probe: 1e8 },
  4: { v: 'D', probe: 1e9 }, 5: { v: 'C', probe: 1e-3 }, 6: { v: 'N', probe: 1.74e11 }, 7: { v: 'C', probe: 1 },
};

// plot box
const X0 = 70, X1 = 432, Y0 = 282, Y1 = 46;
const YLO = 1.4, YHI = 8;

export default function PowerLaw() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3600 });
  const st = s.step;
  const [vid, setVid] = useState<VarId>('N');
  const [pos, setPos] = useState(60); // slider position, 0..100 across the log range

  const v = VARS[vid];
  const lLo = Math.log10(v.lo), lHi = Math.log10(v.hi);
  const fromPos = (p: number) => Math.pow(10, lLo + (p / 100) * (lHi - lLo));
  const toPos = (x: number) => ((Math.log10(x) - lLo) / (lHi - lLo)) * 100;

  useEffect(() => {
    const p = PRESET[st];
    if (p) {
      setVid(p.v);
      const vv = VARS[p.v];
      setPos(((Math.log10(p.probe) - Math.log10(vv.lo)) / (Math.log10(vv.hi) - Math.log10(vv.lo))) * 100);
    }
  }, [st]);

  const xl = useTween(st >= 1 ? 1 : 0, 900);
  const yl = useTween(st >= 2 ? 1 : 0, 900);
  const extrap = st >= 6;

  // the morphing coordinate system
  const linHi = v.tHi * 1.08;
  const pxLin = (x: number) => X0 + (Math.min(x, linHi) / linHi) * (X1 - X0);
  const pxLog = (x: number) => X0 + ((Math.log10(x) - lLo) / (lHi - lLo)) * (X1 - X0);
  const pyLin = (L: number) => Y0 - (L / YHI) * (Y0 - Y1);
  const pyLog = (L: number) => Y0 - ((Math.log10(L) - Math.log10(YLO)) / (Math.log10(YHI) - Math.log10(YLO))) * (Y0 - Y1);
  const X = (x: number) => pxLin(x) + (pxLog(x) - pxLin(x)) * xl;
  const Y = (L: number) => pyLin(L) + (pyLog(L) - pyLin(L)) * yl;

  const linePath = (a: number, b: number) => {
    const n = 90;
    const pts: string[] = [];
    for (let i = 0; i <= n; i++) {
      const x = Math.pow(10, Math.log10(a) + ((Math.log10(b) - Math.log10(a)) * i) / n);
      if (xl < 0.02 && x > linHi) break;
      pts.push(`${i ? 'L' : 'M'}${X(x).toFixed(1)},${Math.max(Y1 - 6, Y(v.f(x))).toFixed(1)}`);
    }
    return pts.join(' ');
  };
  const fitted = useMemo(() => linePath(v.tLo, v.tHi), [vid, xl, yl]); // eslint-disable-line react-hooks/exhaustive-deps
  const extended = useMemo(() => linePath(v.tHi, v.hi), [vid, xl, yl]); // eslint-disable-line react-hooks/exhaustive-deps
  const backward = useMemo(() => linePath(v.lo, v.tLo), [vid, xl, yl]); // eslint-disable-line react-hooks/exhaustive-deps

  const probe = fromPos(pos);
  const Lp = v.f(probe);
  const per10 = Math.pow(10, -v.alpha);
  const inRange = probe >= v.tLo && probe <= v.tHi;

  // axis ticks
  const xTicks: number[] = [];
  for (let e = Math.ceil(lLo); e <= Math.floor(lHi); e += vid === 'C' ? 2 : vid === 'D' ? 1 : 2) xTicks.push(Math.pow(10, e));
  const yTicks = [2, 3, 4, 6, 8];

  // the slope triangle (step 3): one decade starting at 10^6 params
  const triA = vid === 'N' ? 1e6 : vid === 'D' ? 1e8 : 1e-5;
  const triB = triA * 10;

  // Kaplan allocation for 10× compute
  const nGrow = Math.pow(10, 0.73);
  const dGrow = Math.pow(10, 0.27);

  return (
    <Stage
      rootRef={ref}
      title="Loss falls as a power law: a straight line on log-log axes"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['memory', 'parameters N'], ['token', 'data D'], ['compute', 'compute C'], ['text', 'test loss (nats/token)']]}
      {...s}
      readout={
        <Readout items={[
          { label: `Probe: ${vid === 'N' ? 'parameters' : vid === 'D' ? 'tokens' : 'compute'}`, value: v.fmtX(probe), kind: vid === 'N' ? 'memory' : vid === 'D' ? 'token' : 'compute', note: inRange ? 'inside the trained range' : 'extrapolated' },
          { label: 'Predicted loss', value: `${Lp.toFixed(2)} nats`, kind: 'text', note: `L = (${vid}c / ${vid})^${v.alpha.toFixed(3)}` },
          { label: '10× more multiplies loss by', value: `${per10.toFixed(3)}`, kind: 'text', note: `a ${((1 - per10) * 100).toFixed(0)}% drop, at every scale` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Scale what?" value={vid} options={[{ value: 'N', label: 'parameters N' }, { value: 'D', label: 'data D' }, { value: 'C', label: 'compute C' }]} onChange={(x) => { setVid(x); setPos(55); }} />
          <Slider label="Probe along the line" kind={vid === 'N' ? 'memory' : vid === 'D' ? 'token' : 'compute'} value={Math.round(pos)} min={0} max={100} onChange={setPos} format={() => v.fmtX(probe)} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="Test loss plotted against model parameters, data or compute. On linear axes the points form a sharp bend; with both axes logarithmic they lie on a straight line whose slope is the power-law exponent. A dashed extension predicts the loss of much larger models, and a side panel shows the exponent and Kaplan's allocation rule.">
        {/* axes */}
        <line x1={X0} x2={X1} y1={Y0} y2={Y0} className="v-rule" />
        <line x1={X0} x2={X0} y1={Y0} y2={Y1 - 6} className="v-rule" />
        <Label x={(X0 + X1) / 2} y={Y0 + 32} anchor="middle" tone={2} size={10} mono={false}>
          {`${v.label}${xl > 0.5 ? ' · log scale' : ''}`}
        </Label>
        <text x={18} y={(Y0 + Y1) / 2} transform={`rotate(-90 18 ${(Y0 + Y1) / 2})`} textAnchor="middle" className="v-label t2" style={{ fontSize: 10 }}>
          {`test loss, nats per token${yl > 0.5 ? ' · log scale' : ''}`}
        </text>

        {/* y ticks */}
        {(yl > 0.5 ? yTicks : [0, 2, 4, 6, 8]).map((t) => (
          <g key={`y${t}`}>
            <line x1={X0} x2={X1} y1={Y(Math.max(t, 0.0001))} y2={Y(Math.max(t, 0.0001))} className="v-rule faint" />
            <Label x={X0 - 6} y={Y(Math.max(t, 0.0001)) + 3} anchor="end" tone={3} size={8.5}>{String(t)}</Label>
          </g>
        ))}
        {/* x ticks */}
        {xl > 0.5
          ? xTicks.map((t) => (
            <Label key={`x${t}`} x={X(t)} y={Y0 + 14} anchor="middle" tone={3} size={8.5}>{vid === 'C' ? `10${sup(Math.round(Math.log10(t)))}` : count(t)}</Label>
          ))
          : [0, 0.25, 0.5, 0.75, 1].map((f) => (
            <Label key={`xl${f}`} x={X0 + f * (X1 - X0) / 1.08} y={Y0 + 14} anchor="middle" tone={3} size={8.5}>{f === 0 ? '0' : vid === 'C' ? (f * v.tHi).toFixed(1) : count(f * v.tHi)}</Label>
          ))}

        {/* the fitted law, the trained range, and the extrapolation */}
        <path d={backward} fill="none" stroke="var(--text-3)" strokeWidth={1} strokeDasharray="2 4" opacity={xl > 0.5 ? 0.6 : 0} />
        <path d={fitted} fill="none" stroke="var(--text-2)" strokeWidth={1.6} opacity={st >= 2 ? 1 : 0.35} className="v-fade" />
        <path d={extended} fill="none" stroke="var(--text-2)" strokeWidth={1.4} strokeDasharray="5 4" opacity={extrap ? 1 : 0} className="v-fade" />
        {v.pts.map((x) => (
          <circle key={x} cx={X(x)} cy={Y(v.f(x))} r={3.6} fill="var(--ground-2)" stroke="var(--text)" strokeWidth={1.4} />
        ))}

        {/* the slope triangle */}
        <Fade show={st === 3}>
          <g>
            <line x1={X(triA)} x2={X(triB)} y1={Y(v.f(triA))} y2={Y(v.f(triA))} stroke="var(--compute)" strokeWidth={1.2} />
            <line x1={X(triB)} x2={X(triB)} y1={Y(v.f(triA))} y2={Y(v.f(triB))} stroke="var(--compute)" strokeWidth={1.2} />
            <Label x={(X(triA) + X(triB)) / 2} y={Y(v.f(triA)) - 7} anchor="middle" size={9} kind="compute">{vid === 'N' ? '×10 params' : '×10'}</Label>
            <Label x={X(triB) + 6} y={(Y(v.f(triA)) + Y(v.f(triB))) / 2 + 4} size={9} kind="compute">{`loss ×${per10.toFixed(3)}`}</Label>
          </g>
        </Fade>

        {/* the GPT-3 extrapolation */}
        <Fade show={st === 6}>
          <g>
            <circle cx={X(v.gpt3)} cy={Y(v.f(v.gpt3))} r={5} fill="none" stroke="var(--token)" strokeWidth={1.6} />
            <Label x={X(v.gpt3) - 10} y={Y(v.f(v.gpt3)) + 4} anchor="end" size={9} kind="token">{`${v.gpt3Label}: ${v.f(v.gpt3).toFixed(2)}`}</Label>
            <rect x={X(v.tHi)} y={Y1 - 6} width={X1 - X(v.tHi)} height={Y0 - Y1 + 6} fill="var(--text)" opacity={0.03} />
            <Label x={X(v.tHi) + 6} y={Y1 + 6} size={9} tone={3} mono={false}>beyond what was trained</Label>
          </g>
        </Fade>

        {/* probe */}
        <Fade show={st >= 3}>
          <g className="v-move" style={{ transform: `translate(${X(probe)}px, ${Math.max(Y1 - 6, Y(Lp))}px)` }}>
            <circle r={5.5} fill={vid === 'N' ? 'var(--memory)' : vid === 'D' ? 'var(--token)' : 'var(--compute)'} stroke="var(--ground-2)" strokeWidth={2.5} />
          </g>
        </Fade>

        {/* ── the side panel: one card per step */}
        <line x1={452} x2={452} y1={30} y2={318} className="v-rule faint" />
        <Fade show={st === 0}>
          <g>
            <Label x={466} y={56} size={10.5} tone={1} bold>Plotted plainly</Label>
            <Label x={466} y={78} size={9.5} tone={2} mono={false}>Models from 768 to 1.5B</Label>
            <Label x={466} y={94} size={9.5} tone={2} mono={false}>parameters, each trained</Label>
            <Label x={466} y={110} size={9.5} tone={2} mono={false}>until its loss stopped falling.</Label>
            <Label x={466} y={140} size={9.5} tone={3} mono={false}>Most of the dots sit on</Label>
            <Label x={466} y={156} size={9.5} tone={3} mono={false}>the left wall. Is the curve</Label>
            <Label x={466} y={172} size={9.5} tone={3} mono={false}>flattening for good, or just</Label>
            <Label x={466} y={188} size={9.5} tone={3} mono={false}>slowly? You cannot tell.</Label>
          </g>
        </Fade>
        <Fade show={st === 1}>
          <g>
            <Label x={466} y={56} size={10.5} tone={1} bold>Log x: every step is ×10</Label>
            <Label x={466} y={78} size={9.5} tone={2} mono={false}>The points spread out evenly,</Label>
            <Label x={466} y={94} size={9.5} tone={2} mono={false}>one per factor of ~8.</Label>
            <Label x={466} y={124} size={9.5} tone={3} mono={false}>Loss still curves downward,</Label>
            <Label x={466} y={140} size={9.5} tone={3} mono={false}>flattening as it goes.</Label>
          </g>
        </Fade>
        <Fade show={st === 2}>
          <g>
            <Label x={466} y={56} size={10.5} tone={1} bold>Log both: a straight line</Label>
            <Label x={466} y={84} size={12} tone={1} math>{'L(N) = (Nc / N)^α'}</Label>
            <Label x={466} y={110} size={9.5} tone={2} mono={false}>Take logs of both sides:</Label>
            <Label x={466} y={128} size={9.5} tone={2}>{'log L = α·log Nc − α·log N'}</Label>
            <Label x={466} y={152} size={9.5} tone={3} mono={false}>a straight line in log N,</Label>
            <Label x={466} y={168} size={9.5} tone={3} mono={false}>with slope −α.</Label>
            <Label x={466} y={196} size={9.5} tone={3} mono={false}>{`Nc = 8.8 × 10¹³, α = 0.076`}</Label>
          </g>
        </Fade>
        <Fade show={st >= 3 && st <= 5}>
          <g>
            <Label x={466} y={56} size={10.5} tone={1} bold>{vid === 'N' ? 'Parameters' : vid === 'D' ? 'Data' : 'Compute (best model per budget)'}</Label>
            <Label x={466} y={84} size={12} tone={1} math>{vid === 'N' ? 'L = (8.8×10¹³ / N)^0.076' : vid === 'D' ? 'L = (5.4×10¹³ / D)^0.095' : 'L = (3.1×10⁸ / C)^0.050'}</Label>
            <Label x={466} y={112} size={9.5} tone={2} mono={false}>{`×10 → loss × ${per10.toFixed(3)}`}</Label>
            <Label x={466} y={128} size={9.5} tone={2} mono={false}>{`×2 → loss × ${Math.pow(2, -v.alpha).toFixed(3)}`}</Label>
            <Label x={466} y={156} size={9.5} tone={3} mono={false}>{vid === 'C' ? 'C in petaflop/s-days:' : vid === 'D' ? 'measured on models big' : 'measured with enough data'}</Label>
            <Label x={466} y={172} size={9.5} tone={3} mono={false}>{vid === 'C' ? '1 PF-day = 8.64 × 10¹⁹ FLOPs' : vid === 'D' ? 'enough not to be the limit' : 'not to be the limit'}</Label>
            <Label x={466} y={204} size={9.5} tone={3} mono={false}>{'Halving loss this way needs'}</Label>
            <Label x={466} y={220} size={9.5} tone={3} mono={false}>{`×${count(Math.pow(2, 1 / v.alpha))} more ${vid === 'N' ? 'parameters' : vid === 'D' ? 'data' : 'compute'}.`}</Label>
          </g>
        </Fade>
        <Fade show={st === 6}>
          <g>
            <Label x={466} y={56} size={10.5} tone={1} bold>Predict, then train</Label>
            <Label x={466} y={78} size={9.5} tone={2} mono={false}>Fitted up to 1.5B params.</Label>
            <Label x={466} y={94} size={9.5} tone={2} mono={false}>{`At 174B it predicts ${kaplanN(1.74e11).toFixed(2)} nats,`}</Label>
            <Label x={466} y={110} size={9.5} tone={2} mono={false}>{`vs ${kaplanN(1.5e9).toFixed(2)} at 1.5B: ×${(kaplanN(1.74e11) / kaplanN(1.5e9)).toFixed(3)}.`}</Label>
            <Label x={466} y={140} size={9.5} tone={3} mono={false}>GPT-3’s eight models, up to</Label>
            <Label x={466} y={156} size={9.5} tone={3} mono={false}>100× bigger, kept following</Label>
            <Label x={466} y={172} size={9.5} tone={3} mono={false}>a power law in compute.</Label>
            <Label x={466} y={202} size={9.5} kind="token" mono={false}>(Same law, its own data and</Label>
            <Label x={466} y={218} size={9.5} kind="token" mono={false}>tokenizer: compare shapes,</Label>
            <Label x={466} y={234} size={9.5} kind="token" mono={false}>not decimals.)</Label>
          </g>
        </Fade>
        <Fade show={st === 7}>
          <g>
            <Label x={466} y={56} size={10.5} tone={1} bold>10× more compute buys…</Label>
            <Label x={466} y={84} size={9.5} tone={2} mono={false}>Kaplan et al. 2020</Label>
            <Span x={466} y={92} w={160} h={18} kind="memory" label={`N × ${nGrow.toFixed(1)}`} />
            <Span x={466} y={114} w={160 * (dGrow / nGrow)} h={18} kind="token" label={`×${dGrow.toFixed(1)}`} />
            <Label x={466 + 160 * (dGrow / nGrow) + 6} y={127} size={9} kind="token">D</Label>
            <Label x={466} y={148} size={9.5} tone={3} mono={false}>{'N ∝ C^0.73, D ∝ C^0.27'}</Label>
            <Label x={466} y={180} size={9.5} tone={2} mono={false}>Most of every new FLOP</Label>
            <Label x={466} y={196} size={9.5} tone={2} mono={false}>goes into a bigger model.</Label>
            <Label x={466} y={226} size={9.5} kind="error" mono={false}>Two years later this advice</Label>
            <Label x={466} y={242} size={9.5} kind="error" mono={false}>was revised — next lesson.</Label>
          </g>
        </Fade>
        <Label x={466} y={312} size={8.5} tone={3} mono={false}>dots: the fitted law at sizes the paper trained</Label>
      </svg>
    </Stage>
  );
}
