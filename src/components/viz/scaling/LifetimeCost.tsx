import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, Span } from '../lib/svg';
import { EPOCH, lossND, tokensForLoss, count, sci, logScale } from './scale';

// Fix the quality you need (a target loss). Every model size N above some minimum can reach
// it, given enough tokens D(N). Training costs 6·N·D(N); serving T tokens costs 2·N·T.
// Chinchilla minimises the first term alone. Add the second and the cheapest model moves left:
// smaller, trained far longer. Uses the 2024 refit of the Chinchilla data, which agrees with the
// 20-tokens-per-parameter rule.

const FIT = EPOCH;

const STEPS = [
  'Chinchilla’s question: cheapest training for a target quality',
  'Every served token costs 2N',
  'Lifetime cost = training + serving',
  'More traffic, smaller model',
  'The wall: too small never gets there',
  'What the labs did: LLaMA and Llama 3',
  'The next wall: running out of text',
];
const CAPTIONS = [
  'Fix the quality you need: the loss of Chinchilla (70B on 1.4T tokens). Any model bigger than about 11B can reach it with enough data. The training cost 6·N·D is lowest near 20 tokens per parameter: Chinchilla’s answer.',
  'But a model is trained once and then served. Every token it generates costs a forward pass, 2N FLOPs. Serving cost is a straight line: proportional to N, times the tokens you will serve.',
  'Add the two and the cheapest point moves left. A smaller model costs more to train to the same quality, but it saves on every one of the tokens it will serve.',
  'Turn up the lifetime traffic. The best model gets smaller and the tokens per parameter climb into the hundreds. Training past the Chinchilla point on purpose is called over-training.',
  'There is a limit. Below a minimum size the capacity term alone exceeds the target: no amount of data gets there. Near that wall, the data needed explodes.',
  'This is what production models do. LLaMA-1 7B read 1T tokens, about 143 per parameter. Llama 3 8B read 15T, about 1,875. Both far beyond Chinchilla’s 20.',
  'Over-training eats data. At a few thousand tokens per parameter, a useful model needs tens of trillions of tokens, and the stock of good public human text is estimated at roughly 300 trillion.',
];

const TARGETS = [
  { id: 'c70', label: 'Chinchilla 70B quality', N: 70e9 },
  { id: 'c7', label: '7B-Chinchilla quality', N: 7e9 },
];

const PRESET: Record<number, number> = { 0: 0, 1: 1e12, 2: 1e13, 3: 1e14, 4: 1e14, 5: 1e14, 6: 1e15 };

const X0 = 70, X1 = 430, Y0 = 262, Y1 = 34;
const NLO = 5e8, NHI = 1e12;
const FLO = 1e21, FHI = 1e27;
const px = logScale(NLO, NHI, X0, X1);
const pyRaw = logScale(FLO, FHI, Y0, Y1);
const py = (f: number) => Math.max(Y1 - 6, Math.min(Y0, pyRaw(Math.max(f, FLO))));
const RX0 = 70, RX1 = 430;
const rx = logScale(10, 30000, RX0, RX1);
const STOCK = 300e12; // Villalobos et al. 2024: ~300T tokens of quality-adjusted public human text

export default function LifetimeCost() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3600 });
  const st = s.step;
  const [T, setT] = useState(0);
  const [tid, setTid] = useState('c70');

  useEffect(() => {
    const p = PRESET[st];
    if (p !== undefined) setT(p);
  }, [st]);

  const target = TARGETS.find((t) => t.id === tid)!;
  const Lt = lossND(FIT, target.N, 20 * target.N);
  const Nmin = Math.pow(FIT.A / (Lt - FIT.E), 1 / FIT.alpha);

  const grid = useMemo(() => {
    const out: { N: number; D: number; train: number }[] = [];
    for (let i = 0; i <= 900; i++) {
      const N = Math.pow(10, Math.log10(NLO) + ((Math.log10(NHI) - Math.log10(NLO)) * i) / 900);
      const D = tokensForLoss(FIT, N, Lt);
      if (isFinite(D)) out.push({ N, D, train: 6 * N * D });
    }
    return out;
  }, [Lt]);

  const chin = grid.reduce((a, b) => (b.train < a.train ? b : a));
  const best = grid.reduce((a, b) => (b.train + 2 * b.N * T < a.train + 2 * a.N * T ? b : a));
  const bestTotal = best.train + 2 * best.N * T;
  const chinTotal = chin.train + 2 * chin.N * T;
  const saving = 1 - bestTotal / chinTotal;

  const path = (f: (g: { N: number; D: number; train: number }) => number) =>
    grid.filter((g) => pyRaw(f(g)) > Y1 - 6).map((g, i) => `${i ? 'L' : 'M'}${px(g.N).toFixed(1)},${py(f(g)).toFixed(1)}`).join(' ');
  const trainPath = useMemo(() => path((g) => g.train), [grid]); // eslint-disable-line react-hooks/exhaustive-deps
  const servePath = T > 0 ? `M${px(NLO)},${py(2 * NLO * T)} L${px(NHI)},${py(2 * NHI * T)}` : '';
  const totalPath = useMemo(() => (T > 0 ? path((g) => g.train + 2 * g.N * T) : ''), [grid, T]); // eslint-disable-line react-hooks/exhaustive-deps

  const bx = useTween(px(best.N), 500);
  const by = useTween(py(bestTotal), 500);
  const rBest = useTween(rx(Math.min(30000, Math.max(10, best.D / best.N))), 500);

  const showServe = st >= 1;
  const showTotal = st >= 2;
  const showRuler = st >= 3;

  return (
    <Stage
      rootRef={ref}
      title="Lifetime compute: train once, serve forever"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['compute', 'training 6·N·D'], ['token', 'serving 2·N per token'], ['text', 'total'], ['error', 'unreachable']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Cheapest model, lifetime', value: `${count(best.N)} × ${count(best.D)}`, kind: 'memory', note: `${(best.D / best.N).toFixed(0)} tokens per parameter` },
          { label: 'Training + serving', value: `${sci(best.train, 1)} + ${sci(2 * best.N * T, 1)}`, kind: 'compute', note: 'FLOPs' },
          { label: 'Chinchilla choice', value: `${count(chin.N)} × ${count(chin.D)}`, kind: 'text', note: `${(chin.D / chin.N).toFixed(0)} tokens/param, training-optimal` },
          { label: 'Saving vs Chinchilla', value: `${Math.max(0, saving * 100).toFixed(0)}%`, kind: 'token', note: 'of total lifetime FLOPs' },
        ]} />
      }
      controls={
        <>
          <Slider label="Tokens served over the model’s life" kind="token" value={T > 0 ? Math.log10(T) * 10 : 90} min={90} max={160} onChange={(v) => setT(v <= 90 ? 0 : Math.pow(10, v / 10))} format={() => (T > 0 ? count(T) : 'none')} />
          <Segmented label="Target quality" value={tid} options={TARGETS.map((t) => ({ value: t.id, label: t.label }))} onChange={setTid} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="Total compute against model size for a fixed target quality. The training curve is U-shaped with its minimum at the Chinchilla point; a serving line rises with model size; their sum has a minimum at a smaller model trained on more tokens. Below the plot a ruler of tokens per parameter marks Chinchilla, LLaMA-1 7B, Llama 3 8B and the current optimum.">
        {/* axes */}
        <line x1={X0} x2={X1} y1={Y0} y2={Y0} className="v-rule" />
        <line x1={X0} x2={X0} y1={Y0} y2={Y1 - 8} className="v-rule" />
        {[1e9, 1e10, 1e11, 1e12].map((t) => (
          <Label key={t} x={px(t)} y={Y0 + 13} anchor="middle" tone={3} size={8.5}>{count(t)}</Label>
        ))}
        <Label x={X1} y={Y0 + 26} anchor="end" tone={3} size={9} mono={false}>model size N (log scale)</Label>
        {[1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27].map((t) => (
          <g key={t}>
            <line x1={X0} x2={X1} y1={py(t)} y2={py(t)} className="v-rule faint" />
            <Label x={X0 - 5} y={py(t) + 3} anchor="end" tone={3} size={8.5}>{`10${['²¹', '²²', '²³', '²⁴', '²⁵', '²⁶', '²⁷'][Math.round(Math.log10(t)) - 21]}`}</Label>
          </g>
        ))}
        <text x={16} y={(Y0 + Y1) / 2} transform={`rotate(-90 16 ${(Y0 + Y1) / 2})`} textAnchor="middle" className="v-label t2" style={{ fontSize: 10 }}>lifetime FLOPs (log scale)</text>

        {/* unreachable region */}
        <rect x={X0} y={Y1 - 8} width={Math.max(0, px(Nmin) - X0)} height={Y0 - Y1 + 8} fill="url(#tps-hatch-red)" opacity={st >= 4 ? 0.7 : 0.25} className="v-fade" />
        <Fade show={st >= 4}>
          <g>
            <Label x={X0 + 6} y={Y0 - 26} size={9} kind="error">{`smaller than ${count(Nmin)}:`}</Label>
            <Label x={X0 + 6} y={Y0 - 12} size={9} kind="error">never reaches the target</Label>
          </g>
        </Fade>

        {/* curves */}
        <path d={trainPath} fill="none" stroke="var(--compute)" strokeWidth={1.8} />
        <Fade show={showServe && T > 0}>
          <path d={servePath} fill="none" stroke="var(--token)" strokeWidth={1.6} strokeDasharray="5 3" />
        </Fade>
        <Fade show={showTotal && T > 0}>
          <path d={totalPath} fill="none" stroke="var(--text)" strokeWidth={2} />
        </Fade>

        {/* Chinchilla point */}
        <circle cx={px(chin.N)} cy={py(chin.train)} r={4.5} fill="var(--compute)" />
        <Label x={px(chin.N) + 8} y={py(chin.train) + 14} size={9} kind="compute">{`training-optimal: ${count(chin.N)}`}</Label>

        {/* lifetime optimum */}
        <Fade show={showTotal && T > 0}>
          <g>
            <g className="v-move" style={{ transform: `translate(${bx}px, ${by}px)` }}>
              <circle r={5.5} fill="var(--text)" stroke="var(--ground-2)" strokeWidth={2.5} />
            </g>
            <Label x={bx - 8} y={by - 10} anchor="end" size={9.5} tone={1} bold>{`lifetime-optimal: ${count(best.N)}`}</Label>
          </g>
        </Fade>

        {/* ── tokens-per-parameter ruler */}
        <Fade show={showRuler}>
          <g>
            <line x1={RX0} x2={RX1} y1={306} y2={306} className="v-rule" />
            {[10, 100, 1000, 10000].map((t) => (
              <g key={t}>
                <line x1={rx(t)} x2={rx(t)} y1={306} y2={310} className="v-rule" />
                <Label x={rx(t)} y={320} anchor="middle" tone={3} size={8}>{t.toLocaleString('en-US')}</Label>
              </g>
            ))}
            <Label x={RX0} y={334} tone={3} size={8.5} mono={false}>training tokens per parameter</Label>
            <circle cx={rx(20)} cy={306} r={3.5} fill="var(--compute)" />
            <Label x={rx(20)} y={298} anchor="middle" size={8.5} kind="compute">Chinchilla 20</Label>
            <g className="v-move" style={{ transform: `translate(${rBest}px, 306px)` }}>
              <circle r={4.5} fill="var(--text)" stroke="var(--ground-2)" strokeWidth={2} />
            </g>
          </g>
        </Fade>
        <Fade show={st >= 5}>
          <g>
            <circle cx={rx(143)} cy={306} r={3.5} fill="var(--memory)" />
            <Label x={rx(143)} y={298} anchor="middle" size={8.5} kind="memory">LLaMA-1 7B 143</Label>
            <circle cx={rx(1875)} cy={306} r={3.5} fill="var(--memory)" />
            <Label x={rx(1875)} y={298} anchor="middle" size={8.5} kind="memory">Llama 3 8B 1,875</Label>
          </g>
        </Fade>

        {/* ── side panel */}
        <line x1={450} x2={450} y1={28} y2={330} className="v-rule faint" />
        <Label x={462} y={44} size={10.5} tone={1} bold>{st >= 6 ? 'Data needed vs supply' : 'The bill, split'}</Label>
        <Fade show={st < 6}>
          <g>
            <Label x={462} y={66} size={9} tone={3} mono={false}>{`target loss ${Lt.toFixed(3)} nats`}</Label>
            <Label x={462} y={80} size={9} tone={3} mono={false}>{`(= ${count(target.N)} params on ${count(20 * target.N)} tokens)`}</Label>
            <Label x={462} y={106} size={9.5} tone={2} mono={false}>{`served tokens T = ${T > 0 ? count(T) : '0'}`}</Label>
            {[
              { name: 'Chinchilla', g: chin },
              { name: 'lifetime-opt', g: best },
            ].map(({ name, g }, i) => {
              const tr = g.train;
              const sv = 2 * g.N * T;
              const tot = tr + sv;
              const max = Math.max(chinTotal, bestTotal);
              const W = 160;
              return (
                <g key={name}>
                  <Label x={462} y={134 + i * 50} size={9} tone={2}>{`${name}: ${count(g.N)}`}</Label>
                  <Span x={462} y={140 + i * 50} w={(tr / max) * W} h={14} kind="compute" />
                  <Span x={462 + (tr / max) * W} y={140 + i * 50} w={(sv / max) * W} h={14} kind="token" />
                  <Label x={462} y={168 + i * 50} size={8.5} tone={3}>{`${sci(tot, 1)} FLOPs`}</Label>
                </g>
              );
            })}
            <Label x={462} y={250} size={9} tone={2} mono={false}>A token served costs 2N,</Label>
            <Label x={462} y={264} size={9} tone={2} mono={false}>a token trained 6N. Serve</Label>
            <Label x={462} y={278} size={9} tone={2} mono={false}>3× the training tokens and</Label>
            <Label x={462} y={292} size={9} tone={2} mono={false}>serving matches training.</Label>
          </g>
        </Fade>
        <Fade show={st >= 6}>
          <g>
            {(() => {
              const dr = logScale(1e11, 1e15, 462, 626);
              const rows = [
                { name: 'GPT-3', D: 300e9, kind: 'text' as const },
                { name: 'Chinchilla', D: 1.4e12, kind: 'text' as const },
                { name: 'Llama 3 8B', D: 15e12, kind: 'memory' as const },
                { name: 'lifetime-opt', D: best.D, kind: 'token' as const },
              ];
              return (
                <g>
                  {rows.map((r, i) => (
                    <g key={r.name}>
                      <Label x={462} y={70 + i * 34} size={9} tone={2}>{`${r.name}: ${count(r.D)} tokens`}</Label>
                      <Span x={462} y={76 + i * 34} w={Math.max(2, dr(Math.min(1e15, r.D)) - 462)} h={10} kind={r.kind === 'text' ? 'text' : r.kind} dim={r.kind === 'text'} />
                    </g>
                  ))}
                  <line x1={dr(STOCK)} x2={dr(STOCK)} y1={60} y2={206} stroke="var(--error)" strokeWidth={1.4} strokeDasharray="4 3" />
                  <Label x={dr(STOCK) - 4} y={222} anchor="end" size={9} kind="error">~300T public text</Label>
                  <Label x={462} y={244} size={8.5} tone={3} mono={false}>bars on a log scale, 100B to 1,000T</Label>
                  <Label x={462} y={268} size={9} tone={2} mono={false}>Estimated stock: ~300T tokens</Label>
                  <Label x={462} y={282} size={9} tone={2} mono={false}>(90% range 100T–1,000T),</Label>
                  <Label x={462} y={296} size={9} tone={2} mono={false}>Villalobos et al. 2024.</Label>
                  <Label x={462} y={318} size={9} kind="error" mono={false}>Repeats and synthetic text help,</Label>
                  <Label x={462} y={332} size={9} kind="error" mono={false}>up to a point.</Label>
                </g>
              );
            })()}
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
