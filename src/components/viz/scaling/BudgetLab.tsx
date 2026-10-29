import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, Span } from '../lib/svg';
import { EPOCH, HOFFMANN, FITS, TRAIN_GPUS, gpu, flopsFromHours, lossND, lossAtC, nOptFit, nOptRule, count, sci, hours, usd, daysOn, logScale, type Fit } from './scale';

// The lab simulator: a budget in GPU-hours becomes FLOPs (hours × 3600 × peak × MFU), a
// chosen model size turns them into tokens (D = C / 6N), and the Chinchilla fit turns N and D
// into a predicted loss. Then the model is served: tokens per day × days × 2N FLOPs, priced
// at an assumed serving utilization. Every number is one of the module's formulas.

const SERVE_UTIL = 0.25; // assumed fraction of peak a well-batched server sustains (FLOP-based estimate)
const TARGET = { N: 70e9, D: 1.4e12 }; // the quality bar: Chinchilla's own

const STEPS = [
  'Spend Gopher’s budget the way Gopher did',
  'Move to the bottom of the valley',
  'A bigger budget on faster GPUs',
  'Now the model has users',
  'Same quality, a third of the lifetime bill',
  'Your turn',
];
const CAPTIONS = [
  '404,000 H100-hours at 40% MFU is 5.76 × 10²³ FLOPs, Gopher’s budget. Spend it on 280B parameters and the model gets 343B tokens: about one token per parameter, high on the valley’s right wall.',
  'Same hours, 70B parameters: 1.37T tokens, 20 per parameter, at the bottom of the valley. Same bill, lower loss.',
  'A million hours on B200s is 3.2 × 10²⁴ FLOPs. The rule says N = √(C/120) ≈ 164B. Watch the bottom of the valley move right.',
  'Back to the Chinchilla plan, but now it serves 100 billion tokens a day for two years. Serving costs about 28 times as much as training did.',
  'A 20B model trained on 15T tokens reaches the same quality. Its training costs 3× more; its serving costs 3.4× less. The lifetime bill falls from about $30M to under $12M.',
  'Change anything. The readouts always tell you the loss, how far you are from compute-optimal, and what the model costs to make and to run.',
];

const H0 = 5.76e23 / (989e12 * 0.4 * 3600); // ≈ 404,449 H100-hours
const PRESET: Record<number, { h: number; g: string; N: number; tpd: number }> = {
  0: { h: H0, g: 'h100', N: 280e9, tpd: 0 },
  1: { h: H0, g: 'h100', N: 70e9, tpd: 0 },
  2: { h: 1e6, g: 'b200', N: 164e9, tpd: 0 },
  3: { h: H0, g: 'h100', N: 70e9, tpd: 1e11 },
  4: { h: 1.29e6, g: 'h100', N: 20.4e9, tpd: 1e11 },
};

const X0 = 58, X1 = 396, Y0 = 268, Y1 = 46;
const NLO = 1e9, NHI = 3e12;
const px = logScale(NLO, NHI, X0, X1);

/** Smallest budget whose compute-optimal model reaches loss L (bisection on log C). */
function optimalBudgetFor(f: Fit, L: number) {
  let lo = 15, hi = 32;
  for (let i = 0; i < 60; i++) {
    const m = (lo + hi) / 2;
    const C = Math.pow(10, m);
    if (lossAtC(f, C, nOptFit(f, C)) > L) lo = m; else hi = m;
  }
  return Math.pow(10, hi);
}

export default function BudgetLab() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 4200 });
  const st = s.step;
  const [h, setH] = useState(H0);
  const [gpuId, setGpuId] = useState('h100');
  const [mfu, setMfu] = useState(40);
  const [N, setN] = useState(280e9);
  const [tpd, setTpd] = useState(0);
  const [days, setDays] = useState(730);
  const [fitId, setFitId] = useState('epoch');

  useEffect(() => {
    const p = PRESET[st];
    if (p) { setH(p.h); setGpuId(p.g); setN(p.N); setTpd(p.tpd); setMfu(40); setDays(730); }
  }, [st]);

  const fit = FITS.find((f) => f.id === fitId) ?? EPOCH;
  const g = gpu(gpuId);
  const C = flopsFromHours(h, g, mfu / 100);
  const D = C / (6 * N);
  const L = lossND(fit, N, D);
  const nFit = nOptFit(fit, C);
  const nRule = nOptRule(C);
  const Lmin = lossAtC(fit, C, nFit);
  const efficiency = Math.min(1, optimalBudgetFor(fit, L) / C);
  const target = lossND(fit, TARGET.N, TARGET.D);
  const meets = L <= target + 0.001;

  const T = tpd * days;
  const serveFlops = 2 * N * T;
  const serveHours = serveFlops / (g.flops * SERVE_UTIL) / 3600;
  const trainCost = h * g.price;
  const serveCost = serveHours * g.price;
  const total = trainCost + serveCost;

  // valley, zoomed around its bottom
  const lo = Math.min(Lmin, target) - 0.02;
  const hi = Math.max(Lmin + 0.12, Math.min(L + 0.03, Lmin + 0.3));
  const yLo = useTween(lo, 600);
  const yHi = useTween(hi, 600);
  const Y = (l: number) => Y0 - ((l - yLo) / (yHi - yLo)) * (Y0 - Y1);

  const valley = useMemo(() => {
    const pts: string[] = [];
    let pen = false;
    for (let i = 0; i <= 160; i++) {
      const n = Math.pow(10, Math.log10(NLO) + ((Math.log10(NHI) - Math.log10(NLO)) * i) / 160);
      const y = Y(lossAtC(fit, C, n));
      if (y < Y1 - 8 || y > Y0 + 2) { pen = false; continue; }
      pts.push(`${pen ? 'L' : 'M'}${px(n).toFixed(1)},${y.toFixed(1)}`);
      pen = true;
    }
    return pts.join(' ');
  }, [fit, C, yLo, yHi]); // eslint-disable-line react-hooks/exhaustive-deps

  const mx = useTween(px(Math.min(NHI, Math.max(NLO, N))), 450);
  const my = Math.max(Y1 - 8, Math.min(Y0 + 2, Y(L)));
  const yt = Y(target);

  // cost bars
  const W = 200;
  const maxCost = Math.max(total, 1);

  return (
    <Stage
      rootRef={ref}
      title="Compute budget simulator"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['compute', 'training'], ['token', 'serving'], ['memory', 'your model'], ['text', 'predicted loss']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Training compute', value: `${sci(C)} FLOPs`, kind: 'compute', note: `${count(N)} params × ${count(D)} tokens (${(D / N).toFixed(D / N < 10 ? 1 : 0)}/param)` },
          { label: 'Predicted loss', value: `${L.toFixed(3)} nats`, kind: meets ? 'token' : 'text', note: `${meets ? 'meets' : 'misses'} the target ${target.toFixed(3)}` },
          { label: 'Compute efficiency', value: `${(efficiency * 100).toFixed(0)}%`, kind: efficiency > 0.9 ? 'token' : 'error', note: `+${(L - Lmin).toFixed(3)} nats above the bottom` },
          { label: 'Lifetime cost', value: usd(total), kind: 'text', note: `${usd(trainCost)} train + ${usd(serveCost)} serve` },
        ]} />
      }
      controls={
        <>
          <Slider label="Training budget (GPU-hours)" kind="compute" value={Math.round(Math.log10(h) * 100)} min={300} max={800} onChange={(v) => setH(Math.pow(10, v / 100))} format={() => hours(h)} />
          <Segmented label="GPU" value={gpuId} options={TRAIN_GPUS.map((x) => ({ value: x.id, label: x.id.toUpperCase() }))} onChange={setGpuId} />
          <Slider label="Training MFU" kind="compute" value={mfu} min={20} max={60} onChange={setMfu} format={(v) => `${v}%`} />
          <Slider label="Model size N" kind="memory" value={Math.round(Math.log10(N) * 100)} min={800} max={1240} onChange={(v) => setN(Math.pow(10, v / 100))} format={() => count(N)} />
          <Slider label="Traffic (tokens served per day)" kind="token" value={tpd > 0 ? Math.round(Math.log10(tpd) * 10) : 70} min={70} max={130} onChange={(v) => setTpd(v <= 70 ? 0 : Math.pow(10, v / 10))} format={() => (tpd > 0 ? count(tpd) : 'none')} />
          <Slider label="Days in service" kind="token" value={days} min={30} max={1095} step={15} onChange={setDays} format={(v) => `${v}`} />
          <Segmented label="Loss fit" value={fitId} options={[{ value: 'epoch', label: '2024 refit' }, { value: 'hoffmann', label: 'printed 2022' }]} onChange={setFitId} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="Left: the IsoFLOP valley for the chosen training budget, with the chosen model size, the bottom of the valley, the 20-tokens rule and a dashed line for the target quality. Right: the run's numbers and a bar comparing training cost with lifetime serving cost.">
        <Label x={16} y={24} tone={2} size={11} bold>{`Your budget: ${hours(h)} ${g.id.toUpperCase()}-hours = ${sci(C, 2)} FLOPs`}</Label>

        {/* axes */}
        <line x1={X0} x2={X1} y1={Y0} y2={Y0} className="v-rule" />
        <line x1={X0} x2={X0} y1={Y0} y2={Y1 - 10} className="v-rule" />
        {[1e9, 1e10, 1e11, 1e12].map((t) => (
          <Label key={t} x={px(t)} y={Y0 + 14} anchor="middle" tone={3} size={8.5}>{count(t)}</Label>
        ))}
        <Label x={X1} y={Y0 + 28} anchor="end" tone={3} size={9} mono={false}>model size N (log) · tokens D = C / 6N</Label>
        {[Lmin, Lmin + 0.05, Lmin + 0.1].map((t) => (
          <g key={t.toFixed(3)}>
            <line x1={X0} x2={X1} y1={Y(t)} y2={Y(t)} className="v-rule faint" />
            <Label x={X0 - 5} y={Y(t) + 3} anchor="end" tone={3} size={8.5}>{t.toFixed(2)}</Label>
          </g>
        ))}

        {/* target quality */}
        {yt > Y1 - 8 && yt < Y0 && (
          <g>
            <line x1={X0} x2={X1} y1={yt} y2={yt} stroke="var(--text-2)" strokeWidth={1} strokeDasharray="5 4" />
            <Label x={X0 + 4} y={yt - 5} size={8.5} tone={2} mono={false}>target: Chinchilla’s quality</Label>
          </g>
        )}

        <path d={valley} fill="none" stroke="var(--text)" strokeWidth={2} />

        {/* rule-of-20 and fit bottom */}
        <line x1={px(nRule)} x2={px(nRule)} y1={Y0} y2={Y0 - 8} stroke="var(--compute)" strokeWidth={2} />
        <Label x={px(nRule)} y={Y0 - 12} anchor="middle" size={8} kind="compute">20/param</Label>
        <circle cx={px(nFit)} cy={Y(Lmin)} r={4} fill="var(--token)" />
        <Label x={px(nFit) + 8} y={Y(Lmin) - 7} size={8.5} kind="token">{`bottom ${count(nFit)}`}</Label>

        {/* your model */}
        <g className="v-move" style={{ transform: `translate(${mx}px, ${my}px)` }}>
          <circle r={6} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={2.5} />
        </g>
        <Label x={mx + (mx > 300 ? -10 : 10)} y={my - 8} anchor={mx > 300 ? 'end' : 'start'} size={9} kind="memory">{`${count(N)} · ${(D / N).toFixed(D / N < 10 ? 1 : 0)} tok/param`}</Label>

        {/* ── right panel */}
        <line x1={414} x2={414} y1={36} y2={324} className="v-rule faint" />
        <Label x={426} y={52} size={10} tone={1} bold>The run</Label>
        <Label x={426} y={72} size={9} tone={2}>{`N  ${count(N)} params`}</Label>
        <Label x={426} y={88} size={9} tone={2}>{`D  ${count(D)} tokens`}</Label>
        <Label x={426} y={104} size={9} tone={2}>{`wall-clock ${daysOn(h, 1024).toFixed(0)} days on 1,024 GPUs`}</Label>
        <Label x={426} y={124} size={9} kind={efficiency > 0.9 ? 'token' : 'error'} mono={false}>
          {efficiency > 0.97 ? 'compute-optimal: near the bottom' : `a balanced run would match this loss`}
        </Label>
        {efficiency <= 0.97 && (
          <Label x={426} y={138} size={9} kind="error" mono={false}>{`with ${(efficiency * 100).toFixed(0)}% of the budget`}</Label>
        )}

        <Label x={426} y={170} size={10} tone={1} bold>Lifetime cost</Label>
        <Span x={426} y={180} w={(trainCost / maxCost) * W} h={16} kind="compute" />
        <Span x={426 + (trainCost / maxCost) * W} y={180} w={(serveCost / maxCost) * W} h={16} kind="token" />
        <Label x={426} y={212} size={9} kind="compute">{`train  ${usd(trainCost)}`}</Label>
        <Label x={426} y={228} size={9} kind="token">{`serve  ${usd(serveCost)}`}</Label>
        <Label x={426} y={244} size={9} tone={1}>{`total  ${usd(total)}`}</Label>
        <Fade show={T > 0}>
          <g>
            <Label x={426} y={268} size={8.5} tone={3} mono={false}>{`${count(T)} tokens served × 2N FLOPs`}</Label>
            <Label x={426} y={282} size={8.5} tone={3} mono={false}>{`= ${sci(serveFlops, 1)} FLOPs, ${hours(serveHours)} GPU-hours`}</Label>
            <Label x={426} y={296} size={8.5} tone={3} mono={false}>{`at ${SERVE_UTIL * 100}% of peak (assumed)`}</Label>
          </g>
        </Fade>
        <Label x={426} y={318} size={8.5} tone={3} mono={false}>{`$${g.price.toFixed(2)}/GPU-hour (illustrative) · fit: ${fit === HOFFMANN ? 'printed' : 'refit'}`}</Label>
      </svg>
    </Stage>
  );
}
