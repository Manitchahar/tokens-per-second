import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween, useTweens } from '../lib/tween';
import { Label, Fade, Span } from '../lib/svg';
import { HOFFMANN, EPOCH, FITS, lossAtC, capTerm, dataTerm, nOptFit, nOptRule, count, sci, logScale, type Fit } from './scale';

// IsoFLOP curves. Fix a compute budget C, slide the model size N, and train each model on
// whatever data the budget leaves: D = C / 6N. Loss from the Chinchilla fit
// L = E + A/N^α + B/D^β traces a valley. Gopher sits high on its right wall; Chinchilla near
// the bottom. Change C and the valley moves; its bottom follows N ∝ √C, about 20 tokens per
// parameter. The last step shows how flat the bottom is; the fit toggle swaps in the 2024
// refit to show how fragile its exact position is.

const STEPS = [
  'Gopher: 280B parameters, 300B tokens',
  'Fix the compute, slide the model size',
  'Two walls: capacity and data',
  'The bottom of the valley',
  'Chinchilla: same compute, 4× smaller',
  'Change the budget: the valley moves',
  'The bottoms lie on N ∝ √C: 20 tokens per parameter',
  'A flat valley, and a fragile fit',
];
const CAPTIONS = [
  'Gopher, DeepMind’s 2021 model, followed Kaplan’s advice: 280 billion parameters, trained on 300 billion tokens. Its budget, counted the paper’s way, was 5.76 × 10²³ FLOPs.',
  'Keep that budget fixed and ask what else it could have bought. A model of N parameters can be trained on D = C / 6N tokens. Slide N: bigger model, less data.',
  'Loss = E + A/N^α + B/D^β. The capacity term falls as N grows; the data term rises because D shrinks. E is the floor no model can beat.',
  'Their sum is a valley. Gopher sits far up the right wall: too many parameters, too little data. The bottom is at a much smaller model.',
  'Chinchilla: 70B parameters on 1.4T tokens, the same compute as Gopher. Lower predicted loss, and on MMLU it scored 67.5% against Gopher’s 60.0%.',
  'Draw one valley for each budget from 10¹⁹ to 10²⁵ FLOPs. Each has a bottom, and the bottoms march to the right as the budget grows.',
  'The bottoms grow as N ∝ C^0.5 and D ∝ C^0.5: parameters and data in step, about 20 tokens per parameter. So C = 6N × 20N = 120N², and N = √(C/120).',
  'The valley floor is flat: models from about 16B to 70B all land within 0.007 nats. The printed fit puts the exact bottom near 32B; a 2024 refit of the same data puts it near 72B.',
];

const PRESET: Record<number, { C: number; N: number; fit?: string }> = {
  0: { C: 5.76e23, N: 280e9, fit: 'hoffmann' },
  1: { C: 5.76e23, N: 280e9, fit: 'hoffmann' },
  2: { C: 5.76e23, N: 150e9, fit: 'hoffmann' },
  3: { C: 5.76e23, N: 280e9, fit: 'hoffmann' },
  4: { C: 5.76e23, N: 70e9, fit: 'hoffmann' },
  5: { C: 1e22, N: 9.1e9, fit: 'hoffmann' },
  6: { C: 1e24, N: 91e9, fit: 'hoffmann' },
  7: { C: 5.76e23, N: 70e9, fit: 'hoffmann' },
};

const BUDGETS = [1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25];
const X0 = 66, X1 = 430, Y0 = 282, Y1 = 40;
const NLO = 1e8, NHI = 3e12;
const px = logScale(NLO, NHI, X0, X1);

function curve(f: Fit, C: number, Y: (L: number) => number) {
  const pts: string[] = [];
  const n = 120;
  let pen = false;
  for (let i = 0; i <= n; i++) {
    const N = Math.pow(10, Math.log10(NLO) + ((Math.log10(NHI) - Math.log10(NLO)) * i) / n);
    const D = C / (6 * N);
    if (D < 1e8) break;
    const y = Y(lossAtC(f, C, N));
    // leave the plot instead of flattening against its edge: start a new segment on re-entry
    if (y < Y1 - 10 || y > Y0) { pen = false; continue; }
    pts.push(`${pen ? 'L' : 'M'}${px(N).toFixed(1)},${y.toFixed(1)}`);
    pen = true;
  }
  return pts.join(' ');
}

export default function IsoFlop() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3600 });
  const st = s.step;
  const [C, setC] = useState(5.76e23);
  const [N, setN] = useState(280e9);
  const [fitId, setFitId] = useState('hoffmann');

  useEffect(() => {
    const p = PRESET[st];
    if (p) { setC(p.C); setN(p.N); if (p.fit) setFitId(p.fit); }
  }, [st]);

  const fit = FITS.find((f) => f.id === fitId) ?? HOFFMANN;
  const D = C / (6 * N);
  const L = lossAtC(fit, C, N);
  const nFit = nOptFit(fit, C);
  const nRule = nOptRule(C);
  const Lmin = lossAtC(fit, C, nFit);
  const family = st === 5 || st === 6;
  const last = st === 7;

  // y range: zoomed on one valley, or wide for the family
  const lo = family ? 1.78 : Lmin - 0.03;
  const hi = family ? 3.1 : Lmin + 0.17;
  const yr = useTweens({ lo, hi }, 800);
  const Y = (l: number) => Y0 - ((l - yr.lo) / (yr.hi - yr.lo)) * (Y0 - Y1);

  const main = useMemo(() => curve(fit, C, Y), [fit, C, yr.lo, yr.hi]); // eslint-disable-line react-hooks/exhaustive-deps
  const fam = useMemo(() => BUDGETS.map((b) => ({ b, d: curve(fit, b, Y), n: nOptFit(fit, b) })), [fit, yr.lo, yr.hi]); // eslint-disable-line react-hooks/exhaustive-deps

  const mx = useTween(px(Math.min(NHI, Math.max(NLO, N))), 450);
  const myRaw = Y(L);
  const my = Math.max(Y1 - 8, Math.min(Y0 + 8, myRaw));

  // decomposition bar (right panel)
  const cap = capTerm(fit, N);
  const dat = dataTerm(fit, D);
  const barScale = 150 / 0.6; // px per nat above E
  const yTicks = family ? [2.0, 2.4, 2.8] : [Lmin, Lmin + 0.05, Lmin + 0.1, Lmin + 0.15];

  const gopherOnCurve = { N: 280e9, L: lossAtC(fit, C, 280e9) };
  const chinOnCurve = { N: 70e9, L: lossAtC(fit, C, 70e9) };
  const isGopherBudget = Math.abs(Math.log10(C / 5.76e23)) < 0.01;

  return (
    <Stage
      rootRef={ref}
      title="IsoFLOP valleys: fixed compute, every split of N and D"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['memory', 'parameters N'], ['token', 'tokens D'], ['compute', 'compute budget'], ['text', 'predicted loss'], ['error', 'Gopher’s choice']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Budget C', value: `${sci(C)} FLOPs`, kind: 'compute' },
          { label: 'Your split', value: `${count(N)} × ${count(D)}`, kind: 'memory', note: `${(D / N).toFixed(D / N < 10 ? 1 : 0)} tokens per parameter` },
          { label: 'Predicted loss', value: `${L.toFixed(3)} nats`, kind: 'text', note: L - Lmin < 0.0005 ? 'at the bottom' : `+${(L - Lmin).toFixed(3)} above the bottom` },
          { label: 'Bottom of this valley', value: `${count(nFit)} × ${count(C / 6 / nFit)}`, kind: 'token', note: `rule √(C/120): ${count(nRule)} × ${count(20 * nRule)}` },
        ]} />
      }
      controls={
        <>
          <Slider label="Budget C" kind="compute" value={Math.log10(C) * 100} min={1900} max={2500} onChange={(v) => setC(Math.pow(10, v / 100))} format={() => sci(C, 1)} />
          <Slider label="Model size N" kind="memory" value={Math.log10(N) * 100} min={800} max={1240} onChange={(v) => setN(Math.pow(10, v / 100))} format={() => count(N)} />
          <Segmented label="Loss fit" value={fitId} options={[{ value: 'hoffmann', label: 'Hoffmann 2022 (printed)' }, { value: 'epoch', label: 'Besiroglu 2024 refit' }]} onChange={setFitId} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="Predicted loss against model size at a fixed compute budget, a U-shaped valley. Gopher's 280 billion parameters sit high on the right wall; Chinchilla's 70 billion sit near the bottom. Further steps draw one valley per budget from 10^19 to 10^25 FLOPs, with their minima moving right as the budget grows, and a side panel splits the loss into its floor, capacity and data terms.">
        {/* axes */}
        <line x1={X0} x2={X1} y1={Y0} y2={Y0} className="v-rule" />
        <line x1={X0} x2={X0} y1={Y0} y2={Y1 - 10} className="v-rule" />
        {[1e8, 1e9, 1e10, 1e11, 1e12].map((t) => (
          <g key={t}>
            <line x1={px(t)} x2={px(t)} y1={Y0} y2={Y0 + 4} className="v-rule" />
            <Label x={px(t)} y={Y0 + 15} anchor="middle" tone={3} size={8.5}>{count(t)}</Label>
          </g>
        ))}
        <Label x={(X0 + X1) / 2} y={Y0 + 30} anchor="middle" tone={2} size={10} mono={false}>model size N, parameters (log scale)</Label>
        <text x={16} y={(Y0 + Y1) / 2} transform={`rotate(-90 16 ${(Y0 + Y1) / 2})`} textAnchor="middle" className="v-label t2" style={{ fontSize: 10 }}>predicted loss (nats)</text>
        {yTicks.map((t) => (
          <g key={t.toFixed(3)}>
            <line x1={X0} x2={X1} y1={Y(t)} y2={Y(t)} className="v-rule faint" />
            <Label x={X0 - 5} y={Y(t) + 3} anchor="end" tone={3} size={8.5}>{t.toFixed(2)}</Label>
          </g>
        ))}

        {/* the family of valleys */}
        <Fade show={family}>
          <g>
            {fam.map((c) => (
              <g key={c.b}>
                <path d={c.d} fill="none" stroke="var(--compute)" strokeWidth={1} opacity={0.45} />
                <circle cx={px(c.n)} cy={Y(lossAtC(fit, c.b, c.n))} r={3} fill="var(--compute)" />
              </g>
            ))}
            {fam.map((c) => (
              <Label key={`l${c.b}`} x={px(c.n)} y={Y(lossAtC(fit, c.b, c.n)) + 13} anchor="middle" size={8} kind="compute">{sci(c.b, 0).replace('1 × ', '')}</Label>
            ))}
          </g>
        </Fade>
        <Fade show={st === 6}>
          <path d={fam.map((c, i) => `${i ? 'L' : 'M'}${px(c.n).toFixed(1)},${Y(lossAtC(fit, c.b, c.n)).toFixed(1)}`).join(' ')} fill="none" stroke="var(--token)" strokeWidth={1.6} strokeDasharray="5 3" />
        </Fade>

        {/* the current valley */}
        <Fade show={st >= 1}>
          <path d={main} fill="none" stroke="var(--text)" strokeWidth={2} />
        </Fade>

        {/* bottom of the valley */}
        <Fade show={st >= 3}>
          <g>
            <line x1={px(nFit)} x2={px(nFit)} y1={Y(Lmin)} y2={Y0} stroke="var(--token)" strokeWidth={1} strokeDasharray="3 3" />
            <circle cx={px(nFit)} cy={Y(Lmin)} r={4.5} fill="var(--token)" />
            {!family && <Label x={px(nFit)} y={Y(Lmin) + 16} anchor="middle" size={9} kind="token">{`bottom ≈ ${count(nFit)}`}</Label>}
          </g>
        </Fade>

        {/* flat floor band */}
        <Fade show={st === 7}>
          <rect x={X0} y={Y(Lmin + 0.007)} width={X1 - X0} height={Y(Lmin) - Y(Lmin + 0.007)} fill="var(--token)" opacity={0.12} />
        </Fade>

        {/* Gopher and Chinchilla on Gopher's budget */}
        {isGopherBudget && !family && (
          <g>
            <circle cx={px(gopherOnCurve.N)} cy={Y(gopherOnCurve.L)} r={6} fill="none" stroke="var(--error)" strokeWidth={1.8} />
            <Label x={px(gopherOnCurve.N) + 9} y={Y(gopherOnCurve.L) + 4} size={9.5} kind="error">Gopher 280B</Label>
            <Fade show={st >= 4}>
              <circle cx={px(chinOnCurve.N)} cy={Y(chinOnCurve.L)} r={6} fill="none" stroke="var(--token)" strokeWidth={1.8} />
              <Label x={px(chinOnCurve.N) - 9} y={Y(chinOnCurve.L) - 9} anchor="end" size={9.5} kind="token">Chinchilla 70B</Label>
            </Fade>
            <Fade show={st >= 3}>
              <line x1={px(280e9) + 14} x2={px(280e9) + 14} y1={Y(gopherOnCurve.L)} y2={Y(Lmin)} stroke="var(--error)" strokeWidth={1} />
              <Label x={px(280e9) + 18} y={(Y(gopherOnCurve.L) + Y(Lmin)) / 2 + 4} size={9} kind="error">{`+${(gopherOnCurve.L - Lmin).toFixed(3)}`}</Label>
            </Fade>
          </g>
        )}

        {/* your point */}
        <Fade show={st >= 1}>
          <g className="v-move" style={{ transform: `translate(${mx}px, ${my}px)` }}>
            <circle r={5} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={2.5} />
          </g>
        </Fade>
        <Fade show={st === 0}>
          <g>
            <circle cx={px(280e9)} cy={Y(lossAtC(fit, 5.76e23, 280e9))} r={6} fill="var(--error)" />
            <Label x={px(280e9) - 10} y={Y(lossAtC(fit, 5.76e23, 280e9)) + 4} anchor="end" size={10} kind="error">Gopher: one point, one choice</Label>
          </g>
        </Fade>

        {/* ── side panel */}
        <line x1={448} x2={448} y1={28} y2={318} className="v-rule faint" />
        <Label x={460} y={44} size={10.5} tone={1} bold>{family ? 'Budgets and their bottoms' : last ? 'How flat is the floor?' : 'The budget, split'}</Label>

        <Fade show={!family && !last}>
          <g>
            <Label x={460} y={66} size={9} tone={3} mono={false}>{`C = 6 × N × D = ${sci(C, 2)}`}</Label>
            <Label x={460} y={86} size={9} kind="memory">{`N = ${count(N)}`}</Label>
            <Span x={530} y={77} w={Math.max(2, 90 * Math.min(1, Math.log10(N / 1e8) / Math.log10(NHI / 1e8)))} h={11} kind="memory" />
            <Label x={460} y={104} size={9} kind="token">{`D = ${count(D)}`}</Label>
            <Span x={530} y={95} w={Math.max(2, 90 * Math.min(1, Math.log10(D / 1e9) / 4))} h={11} kind="token" />
            <Label x={460} y={120} size={8.5} tone={3} mono={false}>bars on log scales: one grows, one shrinks</Label>
          </g>
        </Fade>

        <Fade show={!family && st >= 2 && !last}>
          <g>
            <Label x={460} y={148} size={9.5} tone={2} mono={false}>{`loss = E + A/N^α + B/D^β`}</Label>
            {/* stacked decomposition */}
            <Span x={460} y={158} w={(fit.E - 1.6) * barScale} h={14} kind="idle" />
            <Span x={460 + (fit.E - 1.6) * barScale} y={158} w={cap * barScale} h={14} kind="memory" />
            <Span x={460 + (fit.E - 1.6 + cap) * barScale} y={158} w={dat * barScale} h={14} kind="token" />
            <Label x={460} y={188} size={8.5} tone={3}>{`E ${fit.E.toFixed(2)} floor`}</Label>
            <Label x={460} y={202} size={8.5} kind="memory">{`A/N^α  ${cap.toFixed(3)} capacity`}</Label>
            <Label x={460} y={216} size={8.5} kind="token">{`B/D^β  ${dat.toFixed(3)} data`}</Label>
            <Label x={460} y={232} size={8.5} tone={2}>{`sum   ${L.toFixed(3)}`}</Label>
            <Label x={460} y={250} size={8.5} tone={3} mono={false}>(bar starts at 1.6 nats)</Label>
          </g>
        </Fade>

        <Fade show={!family && st >= 3 && !last}>
          <g>
            <Label x={460} y={274} size={9} kind={N > nFit * 1.6 ? 'error' : 'token'} mono={false}>
              {N > nFit * 1.6 ? 'right wall: too big, too few tokens' : N < nFit / 1.6 ? 'left wall: too small to use the data' : 'near the bottom: balanced'}
            </Label>
            <Label x={460} y={292} size={8.5} tone={3} mono={false}>{`at the bottom: capacity/data = β/α = ${(fit.beta / fit.alpha).toFixed(2)}`}</Label>
          </g>
        </Fade>

        <Fade show={family}>
          <g>
            {['C', 'N_opt', 'D_opt', 'D/N'].map((h, j) => (
              <Label key={h} x={[460, 500, 548, 600][j]} y={66} size={8.5} tone={3}>{h}</Label>
            ))}
            {BUDGETS.map((b, i) => {
              const n = nOptFit(fit, b);
              const d = b / 6 / n;
              return [sci(b, 0).replace('1 × ', ''), count(n), count(d), (d / n).toFixed(0)].map((t, j) => (
                <Label key={`${b}-${j}`} x={[460, 500, 548, 600][j]} y={84 + i * 16} size={8.5} tone={j === 3 ? 1 : 2}>{t}</Label>
              ));
            })}
            <Label x={460} y={206} size={8.5} tone={3} mono={false}>{`fit: ${fit.name}`}</Label>
          </g>
        </Fade>
        <Fade show={st === 6}>
          <g>
            <Label x={460} y={232} size={9.5} kind="token" mono={false}>Rule of thumb: D ≈ 20 N</Label>
            <Label x={460} y={250} size={9} tone={2}>{'C = 6N·20N = 120N²'}</Label>
            <Label x={460} y={268} size={9} tone={2}>{'N_opt = √(C / 120)'}</Label>
            <Label x={460} y={290} size={8.5} tone={3} mono={false}>{fitId === 'hoffmann' ? 'the printed fit leans smaller (more D/N)' : 'the refit lands near 20 tokens/param'}</Label>
          </g>
        </Fade>
        <Fade show={st === 7}>
          <g>
            <Label x={460} y={70} size={9.5} tone={1} mono={false}>{`${fit.name}:`}</Label>
            <Label x={460} y={88} size={9} kind="token">{`bottom ≈ ${count(nFit)} params`}</Label>
            <Label x={460} y={104} size={9} kind="token">{`        ${count(C / 6 / nFit)} tokens`}</Label>
            <Label x={460} y={120} size={9} tone={2}>{`        ${(C / 6 / nFit / nFit).toFixed(0)} tokens/param`}</Label>
            <Label x={460} y={146} size={9} tone={3} mono={false}>Green band: within 0.007 nats</Label>
            <Label x={460} y={162} size={9} tone={3} mono={false}>of the bottom. Anywhere in it</Label>
            <Label x={460} y={178} size={9} tone={3} mono={false}>is nearly compute-optimal.</Label>
            <Label x={460} y={204} size={9} kind="error" mono={false}>Gopher is far outside it.</Label>
            <Label x={460} y={232} size={9} tone={2} mono={false}>Switch the loss fit below:</Label>
            <Label x={460} y={248} size={9} tone={2} mono={false}>the bottom moves a lot, the</Label>
            <Label x={460} y={264} size={9} tone={2} mono={false}>Gopher verdict does not.</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
