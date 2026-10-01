import { useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Matrix, Arrow, Span, Track, Ruler, fmt } from '../lib/svg';

// The point: RMSNorm and RoPE are each tiny and each memory-bound, so the cost
// is not the arithmetic — it is the round trips and the launches between them.
// Fusing two passes into one cuts bytes AND launches, and it is lossless.

const STEPS = [
  'A token vector in',
  'RMSNorm: rescale it',
  'RoPE: rotate it by position',
  'The unfused timeline',
  'Why each step is expensive',
  'The fused kernel',
  'Count the launches saved',
];
const CAPTIONS = [
  'One token, hidden size 4096, FP16: 8,192 bytes. That is the entire input to both operations.',
  'RMSNorm divides the vector by its root-mean-square, then scales by a learned gain. One reduction, one elementwise multiply.',
  'RoPE rotates each pair of numbers in the vector by an angle that grows with the token position. The length is unchanged; only the values move.',
  'Written separately, each is its own kernel launch with an HBM round trip in between. Four launches to produce the same 8 KB.',
  'At 8 KB, the data movement is about 25 ns and the launch is about 5 us. The kernel is 200x launch-bound: you are paying for the launch, not the math.',
  'Fused, one program loads the vector, computes the RMS, normalizes, applies RoPE and stores once. Two launches become one.',
  'Per layer per token you removed two launches and one HBM round trip. Multiply by 32 layers and that is where real decode time goes.',
];

type Mode = 'unfused' | 'fused';
const PRESET: Record<number, Mode> = { 0: 'unfused', 1: 'unfused', 2: 'unfused', 3: 'unfused', 4: 'unfused', 5: 'fused', 6: 'fused' };

// tiny 8-number vector so the rotation is visible
const X = [1.0, 0.0, 0.7071, 0.7071, -1.0, 0.0, 0.5, -0.5];
const GAIN = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2];
const POS = 3;                       // token position
const THETA = 0.7;                   // base angle * position, radians
const C = Math.cos(THETA), S = Math.sin(THETA);

// RMS
const meanSq = X.reduce((a, v) => a + v * v, 0) / X.length;
const rms = Math.sqrt(meanSq);
const normed = X.map((v) => v / rms);
const normedGained = normed.map((v, i) => v * GAIN[i]);
// RoPE rotates pairs (0,1), (2,3), (4,5), (6,7)
const roped = normedGained.map((v, i) => (i % 2 === 0 ? v * C - normedGained[i + 1] * S : v * S + normedGained[i - 1] * C));

const CELL = 44;
const GAP = 6;
const NC = 8;
const RW = NC * (CELL + GAP) - GAP;
const X0 = 300;

// timeline geometry
const TLX = 40, TLY = 226, TLW = 560;
const usPer = 26;   // pixels per microsecond

export default function NormRope() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3800 });
  const st = s.step;
  const [mode, setMode] = useState<Mode>(PRESET[0]);

  const unfused = mode === 'unfused';
  const showFused = !unfused;
  const onTimeline = st >= 3;

  // launches: unfused = rms(1) + norm(1) + rope(1) ... simple case here:
  // unfused: read x -> compute rms -> write; read -> normalize -> write; read -> rope -> write
  const unfusedLaunches = 4;
  const fusedLaunches = 1;
  const unfusedBytes = 6 * 8192; // 3 reads + 3 writes of 8 KB
  const fusedBytes = 2 * 8192;

  return (
    <Stage
      rootRef={ref}
      title="Fused RMSNorm & RoPE: the same 8 KB, twice fewer launches"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', 'token', 'error', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Vector', value: '8 KB', kind: 'memory', note: '1 token x 4096 FP16' },
          { label: 'Launches', value: showFused ? fusedLaunches : unfusedLaunches, kind: showFused ? 'token' : 'error' },
          { label: 'HBM bytes', value: showFused ? fmt.bytes(fusedBytes) : fmt.bytes(unfusedBytes), kind: showFused ? 'token' : 'error' },
          { label: 'Launch-bound?', value: 'yes', kind: 'error', note: '8 KB ≪ what 5 µs moves' },
        ]} />
      }
      controls={
        <Segmented
          label="Implementation"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'unfused' as Mode, label: 'Unfused (PyTorch)' },
            { value: 'fused' as Mode, label: 'Fused (Triton)' },
          ]}
        />
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A token vector passes through RMSNorm, then RoPE. The unfused version makes four kernel launches with memory round trips between them; the fused version does it all in one launch.">
        {/* the vector, showing its value at each stage */}
        <Label x={X0} y={36} tone={2} size={10.5}>input x [8 drawn of 4096]</Label>
        <Matrix x={X0} y={44} rows={1} cols={NC} cell={CELL} gap={GAP}
          fill={() => 'dim:memory'} text={(r, c) => X[c].toFixed(2)} />

        {/* rms */}
        <Fade show={st >= 1}>
          <g>
            <Label x={X0} y={118} tone={2} size={10.5}>{`RMSNorm:  rms = ${rms.toFixed(3)}   x/rms · gain`}</Label>
            <Matrix x={X0} y={126} rows={1} cols={NC} cell={CELL} gap={GAP}
              fill={() => 'dim:compute'} text={(r, c) => normedGained[c].toFixed(2)} />
          </g>
        </Fade>

        {/* rope */}
        <Fade show={st >= 2}>
          <g>
            <Label x={X0} y={200} tone={2} size={10.5}>{`RoPE at position ${POS}: rotate pairs by θ = ${THETA.toFixed(2)} rad`}</Label>
            <Matrix x={X0} y={188} rows={1} cols={NC} cell={CELL} gap={GAP}
              fill={() => 'dim:token'} text={(r, c) => roped[c].toFixed(2)} />
          </g>
        </Fade>

        {/* vertical annotations: same length throughout */}
        <Fade show={st >= 3}>
          <g>
            <Label x={X0 + RW + 12} y={62} tone={3} size={9.5}>length never changes</Label>
            <LineDots x={X0 + RW + 20} ys={[44 + CELL / 2, 126 + CELL / 2, 188 + CELL / 2]} />
          </g>
        </Fade>

        {/* the pair that rotated, highlighted */}
        <Fade show={st >= 2}>
          <g>
            <rect x={X0 - 4} y={184} width={2 * (CELL + GAP) + 2} height={CELL + 8} rx={3}
              fill="none" stroke="var(--token)" strokeDasharray="3 3" />
            <Label x={X0 + CELL / 2} y={234} anchor="middle" tone={3} size={9}>one rotated pair</Label>
          </g>
        </Fade>

        {/* ── the timeline ── */}
        <Fade show={onTimeline}>
          <g>
            <Label x={TLX} y={TLY - 34} tone={1} size={11} bold>{unfused ? 'Unfused — 4 launches, round trips between' : 'Fused — 1 launch, no round trip'}</Label>
            <Ruler x={TLX} y={TLY + 46} w={TLW} ticks={[{ at: 0, label: '0' }, { at: 0.25, label: '5 µs' }, { at: 0.5, label: '10 µs' }, { at: 0.75, label: '15 µs' }, { at: 1, label: '20 µs' }]} label="time" />

            {unfused ? (
              <>
                <Track x={TLX} y={TLY} w={TLW} h={20}>
                  <Span x={TLX} y={TLY} w={usPer * 5} h={20} kind="idle" label="" />
                  <Label x={TLX + usPer * 5 / 2} y={TLY - 4} anchor="middle" tone={3} size={9}>read x</Label>
                </Track>
                <Track x={TLX} y={TLY + 26} w={TLW} h={20}>
                  <Span x={TLX + usPer * 5} y={TLY + 26} w={usPer * 5} h={20} kind="compute" label="rms" />
                </Track>
                <Track x={TLX} y={TLY + 52} w={TLW} h={20}>
                  <Span x={TLX + usPer * 10} y={TLY + 52} w={usPer * 5} h={20} kind="idle" label="" />
                  <Label x={TLX + usPer * 12.5} y={TLY + 48} anchor="middle" tone={3} size={9}>write, round trip</Label>
                </Track>
                <Track x={TLX} y={TLY + 78} w={TLW} h={20}>
                  <Span x={TLX + usPer * 10} y={TLY + 78} w={usPer * 5} h={20} kind="compute" label="norm · rope" />
                </Track>
                <Label x={TLX + usPer * 15 + 6} y={TLY + 90} tone={2} size={10}>store 8 KB</Label>
                <Label x={TLX} y={TLY + 116} tone={1} size={10.5} kind="error">{'4 launches × ~5 µs ≈ 20 µs, for 8 KB of real work'}</Label>
              </>
            ) : (
              <>
                <Track x={TLX} y={TLY} w={TLW} h={20}>
                  <Span x={TLX} y={TLY} w={usPer * 5} h={20} kind="compute" label="rms · norm · rope · store" />
                </Track>
                <Label x={TLX} y={TLY + 40} tone={1} size={10.5} kind="token">{'1 launch × ~5 µs ≈ 5 µs — and the vector never left registers'}</Label>
                <Label x={TLX} y={TLY + 58} tone={2} size={10}>the reduction result feeds the elementwise op with no round trip</Label>
              </>
            )}
          </g>
        </Fade>

        {/* step 4: the launch-bound callout */}
        <Fade show={st === 4}>
          <g>
            <text x={TLX} y={TLY + 132} className="v-label" style={{ fill: 'var(--error)', fontSize: 10.5, fontWeight: 600 }}>
              8 KB of data ≈ 25 ns to move · one launch ≈ 5 µs · you are 200× launch-bound
            </text>
          </g>
        </Fade>

        {/* step 6: the saving */}
        <Fade show={st === 6}>
          <g>
            <text x={TLX} y={TLY + 152} className="v-label" style={{ fill: 'var(--token)', fontSize: 10.5, fontWeight: 600 }}>
              per layer per token: 3 launches and 48 KB of HBM traffic removed. x 32 layers = real decode time.
            </text>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}

function LineDots({ x, ys }: { x: number; ys: number[] }) {
  return (
    <g>
      {ys.map((y, i) => <circle key={i} cx={x} cy={y} r={2.5} fill="var(--text-3)" />)}
      <line x1={x} y1={ys[0]} x2={x} y2={ys[ys.length - 1]} stroke="var(--text-3)" strokeDasharray="2 3" />
    </g>
  );
}