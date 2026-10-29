import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep, prefersReducedMotion } from '../lib/useStep';
import { useClock, useInView, useTween } from '../lib/tween';
import { Label, Fade, Span, Arrow } from '../lib/svg';
import { trainFlops, gpuHours, gpuSeconds, TRAIN_GPUS, gpu, RUNS, sci, count, hours, usd, daysOn, logScale } from './scale';

// Where C ≈ 6ND comes from. A tiny layer of 4 inputs and 3 outputs has 12 weights.
// Watch one token go through it forward (2 FLOPs per weight), then the two halves of the
// backward pass (2 more each), and the ledger adds up to 6 per weight per token.
// Then the picture zooms out: multiply by every token, and price the result in GPU-hours.

const STEPS = [
  'One layer, twelve weights',
  'Forward: 2 FLOPs per weight',
  'Backward, part 1: blame flows back',
  'Backward, part 2: each weight’s gradient',
  '6 FLOPs per weight per token',
  'Times every training token: C ≈ 6ND',
  'GPT-3: 3.15 × 10²³ FLOPs',
  'From FLOPs to GPU-hours and dollars',
];
const CAPTIONS = [
  'A layer with 4 inputs and 3 outputs has 4 × 3 = 12 weights, one per line. Training means using every one of them, three times, for every token.',
  'Forward pass: each weight multiplies one input and adds the result into one output. One multiply and one add is 2 FLOPs, so the forward pass costs 2 × 12 = 24 FLOPs. For a whole model, 2N.',
  'Backward pass, first half: the error at each output is sent back through the same weights to say how much each input was to blame. Same multiply-adds, other direction: another 2N.',
  'Backward pass, second half: every weight needs its own gradient, input × output error. One more multiply-add per weight: another 2N.',
  'Add the three: 2N + 2N + 2N = 6N FLOPs to train on one token. The optimizer step happens once per batch, not once per token, so it barely registers.',
  'Every training token pays the same 6N. A run over D tokens costs C ≈ 6ND. Drag the sliders: compute grows in proportion to both.',
  'GPT-3: 6 × 175 billion parameters × 300 billion tokens = 3.15 × 10²³ FLOPs. The paper reports 3.14 × 10²³. A 2024 8B model trained on 15T tokens used more.',
  'Divide by what a GPU really sustains, 989 TFLOP/s × 40% MFU on an H100, and you get GPU-seconds. Then hours, then dollars at an hourly rate.',
];

// the toy layer
const IN = [70, 110, 150, 190].map((y) => ({ x: 92, y }));
const OUT = [90, 130, 170].map((y) => ({ x: 288, y }));
const EDGES = IN.flatMap((a, i) => OUT.map((b, j) => ({ a, b, i, j })));

// ledger
const LX = 380, LW = 236, UNIT = LW / 6; // one "2N" = 2 units

// the compute ladder
const CX0 = 64, CX1 = 600;
const cx = logScale(1e21, 1e26, CX0, CX1);
const LADDER = RUNS.filter((r) => r.id !== 'gopher'); // Gopher sits on top of Chinchilla; the lesson discusses it
// staggered so no label sits on another run's leader line
const LIFT: Record<string, number> = { 'llama1-7b': 0, gpt3: 1, chinchilla: 2, 'llama3-8b': 1, 'llama3-405b': 0 };
const ANCHOR: Record<string, 'start' | 'middle' | 'end'> = { 'llama1-7b': 'end', gpt3: 'end', chinchilla: 'start', 'llama3-8b': 'start', 'llama3-405b': 'end' };

const PRESET: Record<number, { N: number; D: number }> = {
  5: { N: 175e9, D: 300e9 },
  6: { N: 175e9, D: 300e9 },
  7: { N: 175e9, D: 300e9 },
};

export default function TrainingCompute() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3400 });
  const st = s.step;
  const [N, setN] = useState(175e9);
  const [D, setD] = useState(300e9);
  const [gpuId, setGpuId] = useState('h100');
  const [mfu, setMfu] = useState(40);
  const [motion, setMotion] = useState(false);
  const inView = useInView(ref);

  useEffect(() => { setMotion(!prefersReducedMotion()); }, []);
  useEffect(() => {
    const p = PRESET[st];
    if (p) { setN(p.N); setD(p.D); }
  }, [st]);

  const flowing = motion && inView && st >= 1 && st <= 3;
  const t = useClock(flowing, 0.55);
  const phase = flowing ? t % 1 : 0.5;

  const g = gpu(gpuId);
  const C = trainFlops(N, D);
  const hrs = gpuHours(C, g, mfu / 100);
  const secs = gpuSeconds(C, g, mfu / 100);
  const cost = hrs * g.price;
  const runX = useTween(cx(Math.min(1e26, Math.max(1e21, C))), 600);

  const zoomOut = st >= 5;
  const showFwd = st >= 1;
  const showBwdIn = st >= 2;
  const showBwdW = st >= 3;
  const showTotal = st >= 4;

  // which way the dots travel along the edges
  const dir = st === 2 ? -1 : 1;

  return (
    <Stage
      rootRef={ref}
      title="Where C ≈ 6ND comes from"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['compute', 'FLOPs'], ['memory', 'weights (N)'], ['token', 'training tokens (D)']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Training compute C = 6ND', value: `${sci(C)} FLOPs`, kind: 'compute', note: `${count(N)} params × ${count(D)} tokens` },
          { label: 'GPU-hours', value: hours(hrs), kind: 'text', note: `${g.name.split(' ')[0]} at ${mfu}% MFU` },
          { label: 'Wall-clock on 1,024 GPUs', value: `${daysOn(hrs, 1024) >= 10 ? daysOn(hrs, 1024).toFixed(0) : daysOn(hrs, 1024).toFixed(1)} days`, kind: 'text' },
          { label: 'Approximate cost', value: usd(cost), kind: 'error', note: `at $${g.price.toFixed(2)}/GPU-hour (illustrative)` },
        ]} />
      }
      controls={
        <>
          <Slider label="Parameters N" kind="memory" value={N} min={1e8} max={1e12} log onChange={setN} format={(v) => count(v)} />
          <Slider label="Training tokens D" kind="token" value={D} min={1e9} max={3e13} log onChange={setD} format={(v) => count(v)} />
          <Segmented label="GPU" value={gpuId} options={TRAIN_GPUS.map((x) => ({ value: x.id, label: x.id.toUpperCase() }))} onChange={setGpuId} />
          <Slider label="MFU (fraction of peak sustained)" kind="compute" value={mfu} min={20} max={60} onChange={setMfu} format={(v) => `${v}%`} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="A small neural network layer with four inputs and three outputs whose twelve connecting lines are the weights; a ledger on the right adds two FLOPs per weight for the forward pass and four more for the backward pass, then the view zooms out to a logarithmic axis of training compute for famous models and a conversion of FLOPs into GPU-hours and dollars.">
        {/* ─────────────── scene A: one token through one layer */}
        <Fade show={!zoomOut}>
          <g>
            <Label x={16} y={24} tone={2} size={11} bold>One token through one layer</Label>
            <Label x={16} y={40} tone={3} size={9.5} mono={false}>4 inputs × 3 outputs = 12 weights (N = 12)</Label>

            {/* edges = weights */}
            {EDGES.map((e, k) => {
              const lit = showFwd;
              return (
                <line key={k} x1={e.a.x} y1={e.a.y} x2={e.b.x} y2={e.b.y}
                  stroke={lit ? 'var(--memory)' : 'var(--memory-dim)'} strokeWidth={lit ? 1.4 : 1}
                  style={{ transition: 'stroke .5s' }} />
              );
            })}

            {/* travelling multiply-adds */}
            {st >= 1 && st <= 3 && EDGES.map((e, k) => {
              const p = dir > 0 ? phase : 1 - phase;
              const x = e.a.x + (e.b.x - e.a.x) * p;
              const y = e.a.y + (e.b.y - e.a.y) * p;
              return st === 3
                ? <rect key={k} x={(e.a.x + e.b.x) / 2 - 3} y={(e.a.y + e.b.y) / 2 - 3} width={6} height={6} rx={1} fill="var(--compute)" opacity={0.55 + 0.45 * Math.sin((phase + k / 12) * Math.PI * 2) ** 2} />
                : <circle key={k} cx={x} cy={y} r={2.6} fill="var(--compute)" />;
            })}

            {/* nodes */}
            {IN.map((n, i) => (
              <g key={`i${i}`}>
                <circle cx={n.x} cy={n.y} r={11} fill="var(--ground-3)" stroke="var(--rule-2)" />
                <Label x={n.x} y={n.y + 3.5} anchor="middle" size={9} tone={2}>{`x${i + 1}`}</Label>
              </g>
            ))}
            {OUT.map((n, j) => (
              <g key={`o${j}`}>
                <circle cx={n.x} cy={n.y} r={11} fill="var(--ground-3)" stroke={st === 2 || st === 3 ? 'var(--error)' : 'var(--rule-2)'} style={{ transition: 'stroke .5s' }} />
                <Label x={n.x} y={n.y + 3.5} anchor="middle" size={9} tone={2}>{`y${j + 1}`}</Label>
              </g>
            ))}

            {/* the token entering */}
            <Span x={20} y={120} w={44} h={20} kind="token" label="token" />
            <line x1={66} y1={130} x2={78} y2={130} stroke="var(--token)" strokeWidth={1.2} />

            {/* annotations per step */}
            <Fade show={st === 0}>
              <Label x={190} y={226} anchor="middle" tone={3} size={9.5} mono={false}>each line is one weight, a number the model learns</Label>
            </Fade>
            <Fade show={st === 1}>
              <Label x={190} y={226} anchor="middle" size={9.5} kind="compute" mono={false}>forward: y = x · W — every weight does 1 multiply + 1 add</Label>
            </Fade>
            <Fade show={st === 2}>
              <Label x={190} y={226} anchor="middle" size={9.5} kind="compute" mono={false}>backward: error at y sent back through Wᵀ to the inputs</Label>
            </Fade>
            <Fade show={st === 3}>
              <Label x={190} y={226} anchor="middle" size={9.5} kind="compute" mono={false}>backward: each weight’s gradient = its input × its output’s error</Label>
            </Fade>
            <Fade show={st >= 2 && st <= 3}>
              <Label x={318} y={64} tone={3} size={9} kind="error">error</Label>
            </Fade>

            {/* ── the ledger */}
            <Label x={LX} y={24} tone={2} size={11} bold>FLOPs per training token</Label>
            <line x1={LX} x2={LX + LW} y1={34} y2={34} className="v-rule faint" />

            <Fade show={showFwd}>
              <Label x={LX} y={58} size={9.5} tone={2} mono={false}>forward pass  x · W</Label>
              <Span x={LX} y={64} w={2 * UNIT} h={16} kind="compute" label="2N" />
              <Label x={LX + 2 * UNIT + 8} y={76} size={9} tone={3}>{'= 24 here'}</Label>
            </Fade>
            <Fade show={showBwdIn}>
              <Label x={LX} y={102} size={9.5} tone={2} mono={false}>backward, input gradients  δ · Wᵀ</Label>
              <Span x={LX + 2 * UNIT} y={108} w={2 * UNIT} h={16} kind="compute" label="2N" />
              <Label x={LX + 4 * UNIT + 8} y={120} size={9} tone={3}>{'+24'}</Label>
            </Fade>
            <Fade show={showBwdW}>
              <Label x={LX} y={146} size={9.5} tone={2} mono={false}>backward, weight gradients  xᵀ · δ</Label>
              <Span x={LX + 4 * UNIT} y={152} w={2 * UNIT} h={16} kind="compute" label="2N" />
              <Label x={LX + 4 * UNIT - 8} y={164} anchor="end" size={9} tone={3}>{'+24'}</Label>
            </Fade>
            <Fade show={showTotal}>
              <line x1={LX} x2={LX + LW} y1={182} y2={182} className="v-rule" />
              <Span x={LX} y={190} w={6 * UNIT} h={22} kind="compute" label="6N per token" />
              <Label x={LX} y={232} size={10} tone={1}>{'= 72 FLOPs for our 12 weights'}</Label>
              <Label x={LX} y={250} size={9} tone={3} mono={false}>the optimizer update runs once per batch of</Label>
              <Label x={LX} y={263} size={9} tone={3} mono={false}>millions of tokens, so per token it is ≈ 0</Label>
            </Fade>

            <Fade show={showTotal}>
              <g>
                <line x1={16} x2={624} y1={288} y2={288} className="v-rule faint" />
                <Label x={16} y={312} size={10.5} tone={2} mono={false}>Training = forward + backward. Serving (inference) = forward only:</Label>
                <Label x={624} y={312} anchor="end" size={11} kind="compute" bold>6N vs 2N per token</Label>
              </g>
            </Fade>
          </g>
        </Fade>

        {/* ─────────────── scene B: every token, every run */}
        <Fade show={zoomOut}>
          <g>
            <Label x={16} y={26} tone={2} size={11} bold>Every token pays 6N</Label>
            <text x={16} y={56} className="v-label t1" style={{ fontSize: 15 }}>
              <tspan>C ≈ 6 × </tspan>
              <tspan style={{ fill: 'var(--memory)' }}>{count(N)}</tspan>
              <tspan>{' × '}</tspan>
              <tspan style={{ fill: 'var(--token)' }}>{count(D)}</tspan>
              <tspan>{' = '}</tspan>
              <tspan style={{ fill: 'var(--compute)', fontWeight: 700 }}>{sci(C)}</tspan>
              <tspan>{' FLOPs'}</tspan>
            </text>

            {/* the ladder of famous runs */}
            <line x1={CX0} x2={CX1} y1={170} y2={170} className="v-rule" />
            {[21, 22, 23, 24, 25, 26].map((e) => (
              <g key={e}>
                <line x1={cx(Math.pow(10, e))} x2={cx(Math.pow(10, e))} y1={170} y2={175} className="v-rule" />
                <Label x={cx(Math.pow(10, e))} y={188} anchor="middle" tone={3} size={9}>{`10${['²¹', '²²', '²³', '²⁴', '²⁵', '²⁶'][e - 21]}`}</Label>
              </g>
            ))}
            <Label x={CX1} y={224} anchor="end" tone={3} size={9} mono={false}>training FLOPs, log scale — each tick is 10×</Label>
            {LADDER.map((r) => {
              const x = cx(trainFlops(r.N, r.D));
              const lift = LIFT[r.id] ?? 0;
              const y = 152 - lift * 22;
              return (
                <g key={r.id}>
                  <line x1={x} x2={x} y1={y + 4} y2={170} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="2 3" />
                  <circle cx={x} cy={170} r={3.5} fill="var(--ground-2)" stroke="var(--text-2)" />
                  <Label x={x} y={y} anchor={ANCHOR[r.id] ?? 'start'} tone={2} size={9}>
                    {`${r.name} ${sci(trainFlops(r.N, r.D))}`}
                  </Label>
                </g>
              );
            })}
            <g className="v-move" style={{ transform: `translate(${runX}px, 170px)` }}>
              <circle r={6.5} fill="var(--compute)" stroke="var(--ground-2)" strokeWidth={2.5} />
              <Label x={0} y={36} anchor="middle" size={9.5} kind="compute" bold>{C > 1e26 ? 'your run (off the scale)' : C < 1e21 ? 'your run (off the scale)' : 'your run'}</Label>
            </g>

            {/* step 6: the GPT-3 check */}
            <Fade show={st === 6}>
              <g>
                <Label x={16} y={84} size={10} tone={2} mono={false}>GPT-3 paper, Table D.1: 3.14 × 10²³ FLOPs (it counts 174.6B parameters)</Label>
                <Label x={16} y={100} size={10} tone={3} mono={false}>6 × 174.6B × 300B = 3.14 × 10²³ — the same rule, to three digits</Label>
              </g>
            </Fade>

            {/* step 7: the conversion chain */}
            <Fade show={st >= 7}>
              <g>
                <Label x={16} y={84} size={10} tone={2} mono={false}>{`${g.name}: ${(g.flops / 1e12).toFixed(0)} TFLOP/s dense BF16 peak, sustaining ${mfu}% of it`}</Label>
              </g>
            </Fade>
            <Fade show={st >= 7}>
              <g>
                {[
                  { x: 16, w: 128, top: sci(C), bot: 'FLOPs', kind: 'compute' as const },
                  { x: 168, w: 128, top: sci(secs), bot: 'GPU-seconds', kind: 'text' as const },
                  { x: 320, w: 128, top: hours(hrs), bot: 'GPU-hours', kind: 'text' as const },
                  { x: 472, w: 152, top: usd(cost), bot: `at $${g.price.toFixed(2)}/GPU-hour`, kind: 'error' as const },
                ].map((b, i) => (
                  <g key={i}>
                    <rect x={b.x} y={252} width={b.w} height={44} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
                    <Label x={b.x + b.w / 2} y={272} anchor="middle" size={12} bold kind={b.kind === 'text' ? undefined : b.kind} tone={1}>{b.top}</Label>
                    <Label x={b.x + b.w / 2} y={288} anchor="middle" size={9} tone={3} mono={false}>{b.bot}</Label>
                  </g>
                ))}
                {[144, 296, 448].map((x) => <Arrow key={x} x1={x + 3} y1={274} x2={x + 21} y2={274} muted />)}
                <Label x={156} y={244} anchor="middle" size={8.5} tone={3}>{`÷ ${(g.flops * mfu / 100 / 1e12).toFixed(0)}T/s`}</Label>
                <Label x={308} y={244} anchor="middle" size={8.5} tone={3}>÷ 3600</Label>
                <Label x={460} y={244} anchor="middle" size={8.5} tone={3}>{`× $${g.price.toFixed(2)}`}</Label>
                <Label x={16} y={322} size={9.5} tone={3} mono={false}>{`Spread over 1,024 GPUs that is ${daysOn(hrs, 1024).toFixed(1)} days of wall-clock — if nothing ever fails.`}</Label>
              </g>
            </Fade>
            <Fade show={st === 5}>
              <Label x={16} y={270} size={10} tone={2} mono={false}>Double N or double D and C doubles. Both together: 4×.</Label>
            </Fade>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
