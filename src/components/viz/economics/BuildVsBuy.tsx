import { useRef, useState } from 'react';
import { Stage, Slider, Readout, type Kind } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, fmt } from '../lib/svg';

// The build-versus-buy question is a ladder, not a switch. Six rungs move work
// from the provider to you, one layer of the serving stack at a time. The last
// step is the only arithmetic that settles it: the volume at which a per-token
// bill overtakes the fixed cost of a GPU you have to keep warm.

const STEPS = [
  'Six stations: control against effort',
  '1 · Model as a service',
  '2 · One-click open model',
  '3 · Your weights, their container',
  '4 · Your code, their container',
  '5 · Your image, their cluster',
  '6 · Build the whole stack',
  'The crossover, and what moves it',
];

const CAPTIONS = [
  'The six rungs run from least control and least work on the left to most control and most work on the right. Every rung is a different division of the same seven layers.',
  'A foundation-model API is the leftmost rung: the provider owns all seven layers, and you own an HTTP call. Batching, caching, autoscaling and cold starts are all hidden behind the endpoint.',
  'One-click deployment lets you choose a model and an instance type from a catalogue. The provider still owns the container, the engine and the scaling policy, so the knobs stop at the catalogue edge.',
  'Bringing your own weights moves exactly one layer: the model. The platform still supplies the serving container, the engine, the kernels and the cluster, which is why the contract is short and the customisation is narrow.',
  'Bringing your own code moves the model and part of core inference. You write the loading and request logic, but the base image, drivers, HTTP contract and cluster remain the provider’s.',
  'Bringing your own serving image moves the model, core inference and model optimisation inside your container. You can choose kernels and KV reuse policies; the provider still runs the cluster and the endpoint.',
  'Building your own stack moves all seven layers. You gain control of routing, kernels, KV policy and cost tuning, and you take on capacity planning, health checks, autoscaling and on-call.',
  'Per-token pricing is a straight line; a self-hosted GPU is a fixed monthly bill that flattens as volume grows. The crossover moves with average utilisation, and idle hours push it to the right.',
];

interface Layer {
  name: string;
  mod: string;
}

const LAYERS: Layer[] = [
  { name: 'Public API', mod: 'M23' },
  { name: 'Resource management', mod: 'M25' },
  { name: 'Model selection', mod: 'M23' },
  { name: 'Distributed serving', mod: 'M18' },
  { name: 'Core inference', mod: 'M09 / M17' },
  { name: 'Model optimisation', mod: 'M10 / M11' },
  { name: 'Model & weights', mod: 'M04' },
];

interface Option {
  n: number;
  short: string;
  title: string;
  owned: boolean[];
  pay: string;
  hides: string;
  cannot: string;
}

const OPTIONS: Option[] = [
  {
    n: 1,
    short: 'API',
    title: 'Foundation-model API',
    owned: [false, false, false, false, false, false, false],
    pay: 'per token',
    hides: 'batching · prefix cache · autoscaling · cold starts',
    cannot: 'kernels · KV reuse policies · routing · weights',
  },
  {
    n: 2,
    short: '1-click',
    title: 'One-click open-model endpoint',
    owned: [false, false, false, false, false, false, false],
    pay: 'per instance-hour',
    hides: 'container internals · engine defaults · autoscaling',
    cannot: 'kernels · KV policy · payload schema · CUDA version',
  },
  {
    n: 3,
    short: 'BYO weights',
    title: 'Your weights, their container',
    owned: [false, false, false, false, false, false, true],
    pay: 'per instance-hour',
    hides: 'engine internals · kernels · KV policy · autoscaling',
    cannot: 'kernels · KV reuse policies · HTTP contract',
  },
  {
    n: 4,
    short: 'BYO code',
    title: 'Your code, their container',
    owned: [false, false, false, false, true, false, true],
    pay: 'per instance-hour',
    hides: 'base image · drivers · cluster scaling · cold starts',
    cannot: 'kernels inside the base image · HTTP contract',
  },
  {
    n: 5,
    short: 'BYO image',
    title: 'Your image, their cluster',
    owned: [false, false, false, false, true, true, true],
    pay: 'per instance-hour + storage',
    hides: 'cluster · node autoscaling · endpoint routing',
    cannot: 'provider’s routing and autoscaling contract',
  },
  {
    n: 6,
    short: 'Build',
    title: 'Build the whole stack',
    owned: [true, true, true, true, true, true, true],
    pay: 'GPU-hours + people + failures',
    hides: 'nothing — the whole stack is yours',
    cannot: 'nothing technical; you can run out of time or money',
  },
];

const STATION_X = [64, 168, 272, 376, 480, 584];

const STACK_LEGEND: (Kind | [Kind, string])[] = [
  ['compute', 'yours'],
  ['idle', 'provider’s'],
];
const COST_LEGEND: (Kind | [Kind, string])[] = [
  ['token', 'pay per token'],
  ['memory', 'self-hosted GPU'],
  ['idle', 'fixed operating cost'],
];

// Illustrative placeholder prices, not a vendor quote. The GPU rate and the
// measured goodput are the same order of magnitude as the illustrative set in
// Module 25.2; the API rate is a round placeholder so the arithmetic is easy.
const C_API = 0.6; // dollars per million output tokens
const P_GPU = 2.5; // dollars per GPU-hour
const GOODPUT = 4603; // output tokens/s at the SLO, measured on the fixed job
const OPS_MONTH = 1800; // on-call, monitoring, failed-GPU reserve, $/month
const DAYS_MONTH = 730 / 24;

function selfCost(util: number, dailyTokens: number) {
  const marginal = (P_GPU * 1e6) / (3600 * GOODPUT * util);
  const opsPerM = (OPS_MONTH * 1e6) / (dailyTokens * DAYS_MONTH);
  return marginal + opsPerM;
}

function crossover(util: number) {
  const marginal = (P_GPU * 1e6) / (3600 * GOODPUT * util);
  if (C_API <= marginal) return null;
  return (OPS_MONTH * 1e6) / ((C_API - marginal) * DAYS_MONTH);
}

export default function BuildVsBuy() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [util, setUtil] = useState(0.6);

  const active = s.step >= 1 && s.step <= 6 ? s.step - 1 : -1;
  const opt = active >= 0 ? OPTIONS[active] : null;
  const ownedCount = opt ? opt.owned.filter(Boolean).length : 0;

  const marginal = (P_GPU * 1e6) / (3600 * GOODPUT * util);
  const cross = crossover(util);

  const PX = 64;
  const PY = 150;
  const PW = 536;
  const PH = 150;
  const XMAX = 350; // millions of output tokens per day
  const YMAX = 1.2; // dollars per million output tokens
  const xOf = (v: number) => PX + (v / XMAX) * PW;
  const yOf = (c: number) => PY + PH - (Math.min(c, YMAX) / YMAX) * PH;
  const apiY = yOf(C_API);

  const curve = (() => {
    const pts: string[] = [];
    for (let v = 5; v <= XMAX; v += 5) {
      const c = selfCost(util, v);
      if (c <= YMAX) pts.push(`${xOf(v).toFixed(1)},${yOf(c).toFixed(1)}`);
    }
    return pts.join(' ');
  })();

  const crossInChart = cross !== null && cross <= XMAX;
  const crossX = crossInChart ? xOf(cross) : null;

  return (
    <Stage
      rootRef={ref}
      title="Six ways to serve a model, and the volume where the bill flips"
      steps={STEPS}
      captions={CAPTIONS}
      legend={s.step >= 7 ? COST_LEGEND : STACK_LEGEND}
      {...s}
      readout={
        s.step >= 7 ? (
          <Readout
            items={[
              { label: 'API price', value: `$${C_API.toFixed(2)}/M`, kind: 'token', note: 'illustrative placeholder' },
              { label: 'Self-host marginal', value: `$${marginal.toFixed(3)}/M`, kind: 'memory', note: `${(util * 100).toFixed(0)}% average utilisation` },
              {
                label: 'Crossover',
                value: cross === null ? 'never' : `${fmt.int(cross)}M/day`,
                kind: cross === null ? 'error' : 'token',
                note: cross === null ? 'marginal cost is above the API' : 'below it, the API is cheaper',
              },
              { label: 'Fixed operations', value: `$${fmt.int(OPS_MONTH)}/month`, kind: 'idle', note: 'on-call + monitoring + spare' },
            ]}
          />
        ) : opt ? (
          <Readout
            items={[
              { label: 'Rung', value: `${opt.n} of 6`, kind: 'compute', note: opt.title },
              { label: 'You own', value: `${ownedCount} / 7`, kind: ownedCount ? 'compute' : 'idle', note: 'layers of the stack' },
              { label: 'Provider hides', value: opt.hides, kind: 'idle', note: 'the defaults you cannot see' },
              { label: 'You pay', value: opt.pay, kind: 'token', note: 'before storage and egress' },
            ]}
          />
        ) : (
          <Readout
            items={[
              { label: 'Rungs', value: '1 → 6', kind: 'compute', note: 'least to most control' },
              { label: 'Layers', value: '7', kind: 'memory', note: 'API down to weights' },
              { label: 'The trade', value: 'work for control', kind: 'idle', note: 'and fixed cost for marginal cost' },
              { label: 'Settles it', value: 'crossover volume', kind: 'token', note: 'plus the non-cost reasons' },
            ]}
          />
        )
      }
      controls={
        s.step >= 7 ? (
          <Slider
            label="Average utilisation"
            kind="memory"
            value={util}
            min={0.25}
            max={1}
            step={0.05}
            onChange={setUtil}
            format={(v) => `${(v * 100).toFixed(0)}%`}
          />
        ) : undefined
      }
    >
      <svg
        viewBox="0 0 640 360"
        role="img"
        aria-label={
          s.step >= 7
            ? 'A cost chart: a horizontal pay-per-token line at 0.60 dollars per million tokens, and a declining self-hosted cost curve that flattens as daily volume grows, with the crossover point marked and a utilisation slider below.'
            : 'Six stations along a control-versus-effort axis, each with a stack of seven layer boxes showing which layers the provider owns in hatched grey and which layers you own in amber.'
        }
      >
        {s.step < 7 ? (
          <>
            <Label x={16} y={20} tone={2} size={11} bold>
              Control against effort: the same seven layers, divided six ways
            </Label>
            <Label x={32} y={72} tone={3} size={9}>
              less control · less work
            </Label>
            <Label x={616} y={72} anchor="end" tone={3} size={9}>
              more control · more work
            </Label>

            <line x1={32} x2={616} y1={92} y2={92} className="v-rule" />
            {OPTIONS.map((o, i) => {
              const x = STATION_X[i];
              const on = active === i;
              return (
                <g key={o.n}>
                  <circle
                    cx={x}
                    cy={92}
                    r={on ? 11 : 8}
                    fill={on ? 'var(--compute)' : 'var(--ground-2)'}
                    stroke={on ? 'var(--compute)' : 'var(--rule-2)'}
                    strokeWidth={1.2}
                  />
                  <Label x={x} y={95.5} anchor="middle" size={on ? 9.5 : 8.5} bold={on} style={{ fill: on ? '#0e1116' : undefined }} tone={on ? 1 : 3}>
                    {`${o.n}`}
                  </Label>
                  <Label x={x} y={114} anchor="middle" size={8.5} bold={on} tone={on ? 1 : 2}>
                    {o.short}
                  </Label>
                </g>
              );
            })}

            {s.step === 0 && (
              <>
                {OPTIONS.map((o, i) => (
                  <g key={`mini-${o.n}`}>
                    {o.owned.map((mine, j) => (
                      <rect
                        key={j}
                        x={STATION_X[i] - 18}
                        y={140 + j * 8}
                        width={36}
                        height={6}
                        rx={1}
                        fill={mine ? 'var(--compute)' : 'url(#tps-hatch)'}
                        stroke={mine ? 'var(--compute)' : 'var(--idle)'}
                        strokeWidth={0.8}
                      />
                    ))}
                  </g>
                ))}
                <Label x={16} y={214} tone={2} size={10.5} bold>
                  Each station is a different division of the seven layers.
                </Label>
                <Label x={16} y={232} tone={3} size={9}>
                  Top box: public API. Bottom box: the model and its weights. Amber is yours; hatched grey is the provider’s.
                </Label>
                <Label x={16} y={250} tone={3} size={9}>
                  Moving right buys control. The price is work: capacity, health checks, routing, kernels and on-call.
                </Label>
              </>
            )}

            {opt && (
              <>
                <Label x={16} y={150} tone={2} size={11} bold>
                  {`${opt.n} · ${opt.title}`}
                </Label>
                <Label x={624} y={150} anchor="end" size={9} tone={3}>
                  {ownedCount === 0 ? 'provider owns all seven layers' : `you own ${ownedCount} of 7 layers`}
                </Label>

                {LAYERS.map((l, i) => {
                  const mine = opt.owned[i] ?? false;
                  return (
                    <g key={l.name}>
                      <Label x={226} y={177 + i * 24} anchor="end" size={9.5} tone={mine ? 1 : 2} bold={mine}>
                        {l.name}
                      </Label>
                      <Span
                        x={236}
                        y={164 + i * 24}
                        w={250}
                        h={18}
                        kind={mine ? 'compute' : 'idle'}
                        label={mine ? 'you' : 'provider'}
                        labelTone={mine ? 'dark' : 'light'}
                      />
                      <Label x={496} y={177 + i * 24} size={8.5} tone={3}>
                        {l.mod}
                      </Label>
                    </g>
                  );
                })}

                <line x1={16} x2={624} y1={338} y2={338} className="v-rule faint" />
                <Label x={16} y={350} size={8.5} tone={3}>
                  {`pay: ${opt.pay}`}
                </Label>
                <Label x={304} y={350} size={8.5} kind="idle" mono={false}>
                  {`hides: ${opt.hides}`}
                </Label>
                <Label x={16} y={358} size={8.5} kind="error" mono={false}>
                  {`cannot: ${opt.cannot}`}
                </Label>
              </>
            )}
          </>
        ) : (
          <>
            <Label x={16} y={20} tone={2} size={11} bold>
              Cost against daily volume: the API line and the self-hosted curve
            </Label>
            <Label x={16} y={36} tone={3} size={9}>
              Illustrative placeholder prices — not a vendor quote. One fixed workload, measured at its SLO.
            </Label>

            {/* axes */}
            <line x1={PX} x2={PX + PW} y1={PY + PH} y2={PY + PH} className="v-rule" />
            <line x1={PX} x2={PX} y1={PY} y2={PY + PH} className="v-rule" />
            {[0, 0.3, 0.6, 0.9, 1.2].map((c) => (
              <g key={c}>
                <line x1={PX - 4} x2={PX} y1={yOf(c)} y2={yOf(c)} className="v-rule" />
                <Label x={PX - 8} y={yOf(c) + 3} anchor="end" size={8.5} tone={3}>
                  {`$${c.toFixed(2)}`}
                </Label>
              </g>
            ))}
            {[0, 100, 200, 300].map((v) => (
              <g key={v}>
                <line x1={xOf(v)} x2={xOf(v)} y1={PY + PH} y2={PY + PH + 4} className="v-rule" />
                <Label x={xOf(v)} y={PY + PH + 16} anchor="middle" size={8.5} tone={3}>
                  {v === 0 ? '0' : `${v}M`}
                </Label>
              </g>
            ))}
            <Label x={PX + PW} y={PY + PH + 32} anchor="end" tone={3} size={9}>
              daily output tokens
            </Label>
            <Label x={16} y={PY + 4} tone={3} size={9}>
              cost / 1M output tokens
            </Label>

            {/* the cheaper region */}
            {crossX !== null && (
              <rect
                x={crossX}
                y={PY}
                width={Math.max(0, PX + PW - crossX)}
                height={PH}
                fill="var(--token)"
                opacity={0.06}
              />
            )}

            {/* self-hosted curve */}
            <polyline points={curve} fill="none" stroke="var(--memory)" strokeWidth={2.2} strokeLinejoin="round" />
            <Label x={PX + 10} y={PY + 16} size={9.5} kind="memory" bold>
              self-hosted: fixed monthly bill
            </Label>

            {/* pay-per-token line */}
            <line x1={PX} x2={PX + PW} y1={apiY} y2={apiY} stroke="var(--token)" strokeWidth={2.2} />
            <Label x={PX + PW - 6} y={apiY - 7} anchor="end" size={9.5} kind="token" bold>
              {`pay per token · $${C_API.toFixed(2)}/M`}
            </Label>

            {/* crossover */}
            {crossX !== null && (
              <>
                <line x1={crossX} x2={crossX} y1={PY} y2={PY + PH} stroke="var(--token)" strokeWidth={1} strokeDasharray="4 4" />
                <circle cx={crossX} cy={apiY} r={4.5} fill="var(--token)" stroke="var(--ground)" strokeWidth={1.5} />
                <Label x={crossX} y={apiY - 18} anchor="middle" size={9} kind="token" bold>
                  {`crossover ≈ ${fmt.int(cross ?? 0)}M/day`}
                </Label>
              </>
            )}
            {cross === null && (
              <Label x={PX + PW - 6} y={apiY - 24} anchor="end" size={9} kind="error" bold>
                no crossover: the self-hosted marginal cost is already above the API
              </Label>
            )}
            {cross !== null && cross > XMAX && (
              <Label x={PX + PW - 6} y={apiY - 24} anchor="end" size={9} kind="error" bold>
                {`crossover ≈ ${fmt.int(cross)}M/day — beyond this chart`}
              </Label>
            )}

            <Label x={PX} y={PY + PH + 45} tone={3} size={8.5} mono={false}>
              {`self-host total = GPU at ${(util * 100).toFixed(0)}% average utilisation + $${fmt.int(OPS_MONTH)}/month fixed operations`}
            </Label>
            <Label x={PX} y={PY + PH + 58} tone={3} size={8.5} mono={false}>
              At high volume the fixed bill spreads out and the curve flattens; at low volume it is mostly the fixed bill.
            </Label>
          </>
        )}
      </svg>
    </Stage>
  );
}
