import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider, type Kind } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Fade, fmt } from '../lib/svg';
import { modelById, gpuById } from '../lib/perfModel';
import { CHAIN_EAGER, CHAIN_FUSED, chainStats, chainUnits, type Mode, ms } from './model';

// Lesson 15.2 — what torch.compile actually does to one decoder layer.
// The elementwise chain around the seven GEMMs, the HBM traffic it causes,
// and the two separate bills (launches and bytes) that fusion shrinks by
// very different amounts.

const STEPS = [
  'Nine kernels around seven GEMMs',
  'Bill one: the bytes',
  'Bill two: the launches',
  'Dynamo records a tape',
  'Guards decide reuse',
  'Inductor fuses',
  'A graph break',
  'What a break costs',
];

const CAPTIONS = [
  'One decoder layer in eager mode: seven matrix multiplies with nine small elementwise kernels strung around them. Each of the nine is a launch the CPU pays for and a round trip through GPU memory the GPU pays for.',
  'Every round trip moves real bytes. A 512-token chunk through one layer moves 1.4 MB in this chain alone; through 32 layers, 45 MB — none of it reaching a tensor core.',
  'The same nine kernels per layer are 288 launches in one decode step. At 6 µs each that is 1.7 ms of Python to launch work the GPU finishes in 1.7 ms. The chain is launch-bound by itself.',
  'torch.compile starts by running your Python with a proxy tensor that records what you do instead of computing it. Everything it can record becomes one graph of operators.',
  'Before reusing a graph, Dynamo checks guards: same dtypes, shapes, device and flags. All pass and the compiled artifact is reused for free. A new batch size can fail a guard and force a recompile.',
  'Inductor lowers the graph and fuses. Five kernels instead of nine, 22 units of traffic instead of 33.75 — and, crucially, the Python interpreter is now out of the loop entirely.',
  'Some things Dynamo cannot record: a .item() call, a print, a branch on a tensor value, an op with no symbolic form. There the graph stops, that op runs in eager mode, and a new graph starts after it.',
  'A break does not cost you the compile — it costs you the fusion. Everything after it is back to one launch per operator and one round trip per intermediate, and you now own several graphs and several guard sets.',
];

const SHORT_EAGER = ['norm', 'split', 'ropeQ', 'ropeK', '+res', 'norm', 'silu', '×up', '+res'];
const SHORT_FUSED = ['norm', 'rope QK', '+norm', 'silu×up', '+res'];

/** One hidden row of Llama-3.1-8B in BF16, in bytes: the unit the chain is counted in. */
const U = 8192;

type Outcome = 'eager' | 'fused' | 'broken';

const PRESET: Record<number, { outcome: Outcome; tokens: number }> = {
  0: { outcome: 'eager', tokens: 512 },
  1: { outcome: 'eager', tokens: 512 },
  2: { outcome: 'eager', tokens: 512 },
  3: { outcome: 'eager', tokens: 512 },
  4: { outcome: 'eager', tokens: 512 },
  5: { outcome: 'fused', tokens: 512 },
  6: { outcome: 'broken', tokens: 512 },
  7: { outcome: 'broken', tokens: 512 },
};

const RAIL = 34;
const CHIP_Y = 62;
const CHIP_H = 28;
const BAR_Y = 138;
const BAR_W = 624;
const COMB_PITCH = 0.9;

const combKinds = (layers: number, ew: number, head: number): Kind[] => {
  const out: Kind[] = [];
  for (let l = 0; l < layers; l++) {
    for (let i = 0; i < 7; i++) out.push('compute');
    out.push('comm');
    for (let i = 0; i < ew; i++) out.push('memory');
  }
  for (let i = 0; i < head; i++) out.push('text');
  return out;
};

const tickColor = (k: Kind) =>
  k === 'text' ? 'var(--text-3)' : k === 'compute' ? 'var(--compute)' : k === 'memory' ? 'var(--memory)' : k === 'comm' ? 'var(--comm)' : 'var(--idle)';

export default function FusionGraph() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const preset = PRESET[s.step];
  const [outcome, setOutcome] = useState<Outcome>(preset.outcome);
  const [tokens, setTokens] = useState(preset.tokens);
  useEffect(() => { setOutcome(preset.outcome); setTokens(preset.tokens); }, [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const model = modelById('llama8b');
  const gpu = gpuById('h100');
  const bwEff = gpu.bw * 0.8;

  const fused = outcome !== 'eager';
  const ops = fused ? CHAIN_FUSED : CHAIN_EAGER;
  const labels = fused ? SHORT_FUSED : SHORT_EAGER;
  const mode: Mode = fused ? 'compiled' : 'eager';

  const brk = outcome === 'broken' ? 2 : -1;
  const stats = chainStats(model, tokens, mode, outcome === 'broken' ? model.layers / 2 : -1);
  const eagerStats = chainStats(model, tokens, 'eager');
  const fusedStats = chainStats(model, tokens, 'compiled');

  const units = chainUnits(ops);
  const bytes = units * U * tokens * model.layers;
  const bytesMs = (bytes / bwEff) * 1000;

  const pitch = BAR_W / labels.length;
  const chipW = pitch - 7;
  const chipX = (i: number) => 8 + i * pitch;

  const traced = s.step === 3 || s.step === 4;
  const eagerComb = useMemo(() => combKinds(model.layers, 9, 12), [model.layers]);
  const fusedComb = useMemo(() => combKinds(model.layers, 5, 4), [model.layers]);

  // running offset for the traffic bar segments
  let acc = 0;
  const segs = ops.map((o) => {
    const frac = o.units / units;
    const seg = { x: 8 + acc * BAR_W, w: frac * BAR_W };
    acc += frac;
    return seg;
  });

  return (
    <Stage
      rootRef={ref}
      title="One decoder layer, in eager mode and after torch.compile"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'elementwise kernel'],
        ['compute', 'matrix multiply'],
        ['comm', 'attention'],
        ['error', 'eager again, after a break'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Kernels per layer', value: `${labels.length}`, kind: fused ? 'memory' : 'text', note: `with 7 matrix multiplies = ${labels.length + 7} launches` },
          { label: 'HBM traffic, chain', value: fmt.bytes(bytes), kind: 'memory', note: `${fmt.int(tokens)} tokens × ${model.layers} layers` },
          { label: 'GPU time for the chain', value: ms(bytesMs), kind: 'memory', note: `${units.toFixed(2)} units of ${fmt.int(U)} bytes per token` },
          { label: 'CPU time per step', value: ms(stats.cpuMs), kind: stats.cpuMs > 1 ? 'error' : 'token', note: `${fmt.int(stats.kernels)} launches` },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="What Inductor produced"
            value={outcome}
            options={[{ value: 'eager', label: 'eager' }, { value: 'fused', label: 'fused' }, { value: 'broken', label: 'fused, with a break' }]}
            onChange={setOutcome}
          />
          <Slider label="Tokens in the chunk" value={tokens} min={64} max={4096} log onChange={setTokens} format={(v) => `${fmt.int(v)}`} kind="compute" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="The elementwise kernel chain of one decoder layer of Llama-3.1-8B, with a memory rail above it showing a read and a write for every kernel, a bar sized by the bytes each kernel moves, and below it two combs comparing the number of kernel launches in a whole decode step before and after compilation.">
        <Label x={8} y={20} tone={2} size={10.5} mono={false} bold>
          {`Llama-3.1-8B · 32 layers · one ${fmt.int(tokens)}-token chunk · BF16 · HBM at ${(bwEff / 1e12).toFixed(2)} TB/s`}
        </Label>

        {/* the memory rail: every read comes down from it, every write goes back up */}
        <line x1={8} x2={632} y1={RAIL} y2={RAIL} stroke="var(--memory)" strokeWidth={1.4} opacity={0.7} />
        <Label x={632} y={RAIL - 6} anchor="end" size={9} kind="memory">GPU memory · read down, write up</Label>
        {s.step >= 1 && labels.map((_, i) => (
          <g key={`a${i}`}>
            <Arrow x1={chipX(i) + 15} y1={RAIL + 4} x2={chipX(i) + 15} y2={CHIP_Y - 5} kind="memory" muted width={1} />
            <Arrow x1={chipX(i) + 29} y1={CHIP_Y - 5} x2={chipX(i) + 29} y2={RAIL + 4} kind="memory" muted width={1} />
          </g>
        ))}

        {/* the chain */}
        {labels.map((l, i) => {
          const out = brk >= 0 && i >= brk;
          return (
            <g key={l + i}>
              <Span x={chipX(i)} y={CHIP_Y} w={chipW} h={CHIP_H} kind={out ? 'error' : 'memory'} label={l} labelTone={out ? 'light' : 'dark'} dim={out} />
              <Label x={chipX(i) + chipW / 2} y={CHIP_Y + CHIP_H + 12} anchor="middle" size={8.5} tone={3}>{`${ops[i].units.toFixed(2)}u`}</Label>
            </g>
          );
        })}

        {/* dynamo's graph box */}
        {traced && (
          <g>
            <rect x={4} y={CHIP_Y - 8} width={BAR_W + 8} height={CHIP_H + 16} rx={4} fill="none" stroke="var(--text)" strokeWidth={1.4} strokeDasharray="6 4" />
            <Label x={4} y={CHIP_Y + CHIP_H + 26} size={9.5} tone={1} bold>{'one recorded graph: operators, not kernels'}</Label>
          </g>
        )}

        {/* the break */}
        {brk >= 0 && (
          <g>
            <line x1={chipX(brk) - 4} y1={CHIP_Y - 12} x2={chipX(brk) - 4} y2={CHIP_Y + CHIP_H + 16} stroke="var(--error)" strokeWidth={1.6} strokeDasharray="4 3" />
            <Label x={chipX(brk) + 2} y={CHIP_Y + CHIP_H + 26} size={9.5} kind="error" bold>{'graph break — everything right of here is eager again'}</Label>
          </g>
        )}

        {/* the traffic bar */}
        {s.step >= 1 && (
          <g>
            <Label x={8} y={BAR_Y - 6} size={9.5} tone={2}>
              {`HBM traffic of the chain: ${units.toFixed(2)} units · ${fmt.bytes(bytes)} · ${ms(bytesMs)} on this GPU`}
            </Label>
            {segs.map((sg, i) => (
              <rect key={i} x={sg.x} y={BAR_Y} width={sg.w} height={15} fill="var(--memory)" opacity={brk >= 0 && i >= brk ? 0.35 : 0.8} />
            ))}
          </g>
        )}

        {/* ── band 3: what this step is about ──────────────────────────── */}
        {s.step === 0 && (
          <g>
            <Label x={8} y={180} tone={2} size={10.5} mono={false} bold>Seven matrix multiplies, nine small kernels</Label>
            <Label x={8} y={202} tone={3} size={9.5}>The GEMMs do essentially all the arithmetic. These nine do almost none — they move numbers around.</Label>
            <Label x={8} y={222} tone={3} size={9.5}>And yet each one is a launch the CPU must pay for and a trip to memory the GPU must pay for.</Label>
            <Label x={8} y={252} size={10.5} tone={1} mono={false} bold>Two bills, two currencies, and one change can pay both.</Label>
          </g>
        )}

        {s.step === 1 && (
          <g>
            <Label x={8} y={180} tone={2} size={10.5} mono={false} bold>Bill one: the bytes</Label>
            <Label x={8} y={202} tone={3} size={9.5}>{`${chainUnits(CHAIN_EAGER).toFixed(2)} units = ${fmt.int(chainUnits(CHAIN_EAGER) * U)} bytes per token, per layer.`}</Label>
            <Label x={8} y={222} tone={3} size={9.5}>{`× ${fmt.int(tokens)} tokens × ${model.layers} layers = ${fmt.bytes(eagerStats.bytes)} → ${ms((eagerStats.bytes / bwEff) * 1000)} of GPU time.`}</Label>
            <Label x={8} y={252} size={10.5} tone={1} mono={false} bold>Pure overhead. Not one of those bytes reaches a tensor core.</Label>
          </g>
        )}

        {s.step === 2 && (
          <g>
            <Label x={8} y={180} tone={2} size={10.5} mono={false} bold>Bill two: the launches</Label>
            <Label x={8} y={202} tone={3} size={9.5}>{`9 kernels × ${model.layers} layers = ${9 * model.layers} elementwise launches, at 6 µs each = ${ms((9 * model.layers * 6) / 1000)} of Python.`}</Label>
            <Label x={8} y={226} kind="error" size={11} mono={false} bold>{`The chain alone: ${ms((9 * model.layers * 6) / 1000)} of CPU to launch, ${ms(bytesMs)} of GPU to run.`}</Label>
            <Label x={8} y={256} tone={3} size={9.5}>A chain of pure bookkeeping spends as much time being described as being executed. That is the thing to fix.</Label>
          </g>
        )}

        {s.step === 3 && (
          <g>
            <Label x={8} y={180} tone={2} size={10.5} mono={false} bold>Dynamo: run the Python, record the operators</Label>
            <Label x={8} y={202} tone={3} size={9.5}>It does not analyse your code. It runs it, with a proxy tensor that logs every call and hands back another proxy.</Label>
            <Label x={8} y={222} tone={3} size={9.5}>What comes out is a graph of operators with shapes attached — the same thing a JIT gets in any other language.</Label>
            <Label x={8} y={252} size={10.5} tone={1} mono={false} bold>Nothing has touched the GPU yet. This pass is pure bookkeeping.</Label>
          </g>
        )}

        {s.step === 4 && (
          <g>
            <Label x={8} y={180} tone={2} size={10.5} mono={false} bold>Guards: may I reuse last time's graph?</Label>
            {[['dtype', 'bfloat16'], ['shapes', `[${fmt.int(tokens)}, 4096]`], ['device', 'cuda:0'], ['requires_grad', 'False']].map(([k, v], i) => (
              <g key={k}>
                <Label x={8} y={202 + i * 19} size={9.5} tone={2}>{k}</Label>
                <Label x={110} y={202 + i * 19} size={9.5} tone={1}>{v}</Label>
                <Label x={250} y={202 + i * 19} size={9.5} kind="token" bold>✓ reuse the graph</Label>
              </g>
            ))}
            <Label x={8} y={288} tone={3} size={9.5}>Fail one and Dynamo traces again and Inductor regenerates and recompiles kernels: seconds, not microseconds.</Label>
          </g>
        )}

        {s.step === 5 && (
          <g>
            <Label x={8} y={180} tone={2} size={10.5} mono={false} bold>Inductor: fewer kernels, fewer bytes, no Python</Label>
            <Label x={8} y={202} tone={3} size={9.5}>{`9 → 5 kernels per layer, and ${chainUnits(CHAIN_FUSED).toFixed(2)} units instead of ${chainUnits(CHAIN_EAGER).toFixed(2)} — ${((1 - chainUnits(CHAIN_FUSED) / chainUnits(CHAIN_EAGER)) * 100).toFixed(0)}% less traffic.`}</Label>
            <Label x={8} y={222} tone={3} size={9.5}>{`Launches ${fmt.int(eagerStats.kernels)} → ${fmt.int(fusedStats.kernels)} per step, and each now costs ~2 µs instead of 6 µs: the interpreter is out of the loop.`}</Label>
            <Label x={8} y={246} size={11} tone={1} mono={false} bold>{`CPU per step ${ms(eagerStats.cpuMs)} → ${ms(fusedStats.cpuMs)}.  Chain traffic ${fmt.bytes(eagerStats.bytes)} → ${fmt.bytes(fusedStats.bytes)}.`}</Label>
            <Label x={8} y={272} tone={3} size={9.5}>The CPU saving is the big one, and it is the one the previous lesson told you to look for.</Label>
          </g>
        )}

        {s.step === 6 && (
          <g>
            <Label x={8} y={180} tone={2} size={10.5} mono={false} bold>Where Dynamo cannot follow you</Label>
            <Label x={8} y={202} tone={3} size={9.5}>tensor.item(), .tolist(), print(t), a branch on whether t is above zero, an op with no symbolic form, a numpy call.</Label>
            <Label x={8} y={222} tone={3} size={9.5}>At that line the graph ends, the op runs in eager mode, and a new graph starts on the far side.</Label>
            <Label x={8} y={252} size={11} kind="error" mono={false} bold>It does not crash. It silently gives you a compiled model with an eager middle.</Label>
          </g>
        )}

        {s.step === 7 && (
          <g>
            <Label x={8} y={180} tone={2} size={10.5} mono={false} bold>What one break in the middle of the model costs</Label>
            {[
              ['eager everywhere', `${eagerStats.ewKernels}`, fmt.bytes(eagerStats.bytes), ms(eagerStats.cpuMs)],
              ['fused everywhere', `${fusedStats.ewKernels}`, fmt.bytes(fusedStats.bytes), ms(fusedStats.cpuMs)],
              ['one break, after layer 16', `${stats.ewKernels}`, fmt.bytes(stats.bytes), ms(stats.cpuMs)],
            ].map((row, i) => (
              <g key={row[0]}>
                <Label x={8} y={206 + i * 22} size={9.5} tone={i === 2 ? 1 : 2} bold={i === 2}>{row[0]}</Label>
                <Label x={196} y={206 + i * 22} size={9} tone={3}>{`${row[1]} ew kernels`}</Label>
                <Label x={300} y={206 + i * 22} size={9} kind="memory">{`${row[2]} HBM`}</Label>
                <Label x={430} y={206 + i * 22} size={9} kind={i === 2 ? 'error' : 'token'} bold>{`${row[3]} CPU`}</Label>
              </g>
            ))}
            <Label x={8} y={286} tone={3} size={9.5}>You keep the cheap launches and lose the cheap bytes. And now you own several graphs and several guard sets,</Label>
            <Label x={8} y={302} tone={3} size={9.5}>so every distinct shape that reaches one of them is another recompile you pay for at start-up.</Label>
          </g>
        )}

        {/* ── the whole-step launch combs ──────────────────────────────── */}
        <Label x={8} y={318} tone={2} size={10.5} mono={false} bold>One decode step: one tick per kernel launch</Label>
        {[
          { label: `eager — ${fmt.int(eagerComb.length)} launches`, kinds: eagerComb, y: 330, dim: outcome !== 'eager' },
          { label: `compiled — ${fmt.int(fusedComb.length)} launches`, kinds: fusedComb, y: 358, dim: outcome === 'eager' },
        ].map((row) => (
          <g key={row.label} className="v-fade" style={{ opacity: row.dim ? 0.26 : 1 }}>
            <Label x={8} y={row.y + 9} size={9} tone={2}>{row.label}</Label>
            {row.kinds.map((k, i) => (
              <rect key={i} x={140 + i * COMB_PITCH} y={row.y} width={COMB_PITCH * 0.68} height={11} fill={tickColor(k)} />
            ))}
          </g>
        ))}
        <Label x={632} y={339} anchor="end" size={9} tone={3}>{`${(eagerComb.length * COMB_PITCH).toFixed(0)} px`}</Label>
        <Label x={632} y={367} anchor="end" size={9} tone={3}>{`${(fusedComb.length * COMB_PITCH).toFixed(0)} px`}</Label>
      </svg>
    </Stage>
  );
}
