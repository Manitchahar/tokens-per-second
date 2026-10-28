import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, Span, Track, Ruler, fmt } from '../lib/svg';

// The point: almost every kernel benchmark you will ever run is wrong in one of
// five specific ways. Each step shows one of them inflating the number, then the
// honest measurement. The honest number is always lower — and it is the only one
// that means anything.

const STEPS = [
  'The claim under test',
  'No warmup: you measured the compiler',
  'No synchronize: you measured the queue',
  'Cached: you measured L2, not HBM',
  'One iteration: you measured noise',
  'The honest measurement',
  'Read it against the roofline',
];
const CAPTIONS = [
  'We have a fused kernel and want to know how fast it is, so we can report GB/s and decide whether to keep optimizing.',
  'The first call to a Triton kernel JIT-compiles it: tens to hundreds of milliseconds. Timing that single call reports the compiler, not the kernel.',
  'CUDA launches are asynchronous — the call returns before the GPU finishes. Without a synchronize you are timing how fast the CPU can enqueue work.',
  'A small tensor repeated 200 times sits in the 50 MB L2 cache after the first pass. You are measuring cache bandwidth, which is several times HBM bandwidth.',
  'A single timed iteration on a shared machine is dominated by scheduling jitter. You need a warmup, a sync, a cache flush, and many repetitions.',
  'Warm up, flush L2 between reps, synchronize with CUDA events, and take a median. This is the number you report.',
  'Compare achieved bandwidth to the GPU peak. 80% is done; 20% means there is a real problem to find.',
];

type Method = 'naive' | 'warm' | 'sync' | 'l2' | 'median';
const PRESET: Record<number, Method> = {
  0: 'naive', 1: 'naive', 2: 'sync', 3: 'l2', 4: 'median', 5: 'median', 6: 'median',
};

// An honest kernel time and the inflated numbers each mistake produces.
const TRUE_US = 42;                       // µs — the real kernel
const FLUSH_COST_US = 12;
const JIT_MS = 180;                       // one-time compile
const L2_SPEEDUP = 3.1;                   // cached reads are ~3x faster
const NOISE_US = 55;                      // a single jittered measurement

const gpu = { name: 'H100 SXM', bw: 3350 };  // GB/s, for the "% of peak" readout

export default function BenchHarness() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 4000 });
  const st = s.step;
  const [method, setMethod] = useState<Method>(PRESET[0]);
  const [iters, setIters] = useState(200);

  const honest = method === 'median';
  const reported =
    method === 'naive' ? JIT_MS * 1000 :
    method === 'warm' ? TRUE_US :
    method === 'sync' ? TRUE_US :
    method === 'l2' ? TRUE_US / L2_SPEEDUP :
    TRUE_US;

  // bytes moved by our fused kernel (say: 3 reads + 1 write of a 4 MB activation)
  const bytes = 12.6e6;
  const gbs = bytes / (reported * 1e-6) / 1e9;
  const gbsTween = useTween(gbs, 500);

  const TLX = 60, TLW = 520, TLY = 260;
  const maxUs = 180000;
  const px = (us: number) => TLX + Math.min(1, us / maxUs) * TLW;

  const barW = Math.max(3, px(reported) - TLX);

  const labelFor = {
    naive: 'timed the first call',
    warm: 'warmed up, no sync',
    sync: 'synchronized, no L2 flush',
    l2: 'L2-flushed, single rep',
    median: 'honest: warm + flush + sync + median',
  } as Record<Method, string>;

  return (
    <Stage
      rootRef={ref}
      title="Every kernel benchmark is wrong in one of four ways"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'error', 'token', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Reported', value: reported >= 1000 ? `${(reported / 1000).toFixed(0)} ms` : `${reported.toFixed(0)} µs`, kind: honest ? 'token' : 'error', note: labelFor[method] },
          { label: 'True time', value: `${TRUE_US} µs`, kind: 'memory' },
          { label: 'Overstated by', value: `${(reported / TRUE_US).toFixed(1)}×`, kind: honest ? 'token' : 'error' },
          { label: 'Achieved', value: `${gbsTween.toFixed(0)} GB/s`, kind: gbs > gpu.bw * 0.6 ? 'token' : 'compute', note: `${((gbs / gpu.bw) * 100).toFixed(0)}% of H100 peak` },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Method"
            value={method}
            onChange={setMethod}
            options={[
              { value: 'median' as Method, label: 'Honest' },
              { value: 'naive' as Method, label: 'First call' },
              { value: 'l2' as Method, label: 'No L2 flush' },
            ]}
          />
          <Slider label="Iterations" value={iters} min={1} max={1000} onChange={setIters} kind="compute" format={(v) => String(v)} />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A horizontal bar comparing a reported kernel time under different benchmarking methods. Timing the first call reports the JIT compiler and is thousands of times too slow; not flushing L2 is about three times too fast; the honest measurement is the true time.">
        {/* the bar */}
        <Label x={TLX} y={TLY - 46} tone={1} size={11} bold>{labelFor[method]}</Label>
        <Ruler x={TLX} y={TLY + 34} w={TLW} ticks={[{ at: 0, label: '0' }, { at: 0.05, label: '9 ms' }, { at: 0.111, label: '20 ms' }, { at: 0.1667, label: '30 ms' }, { at: 0.2778, label: '50 ms' }, { at: 0.8333, label: '150 ms' }]} label="reported time" />
        <Track x={TLX} y={TLY} w={TLW} h={22}>
          <Span x={TLX} y={TLY} w={px(TRUE_US) - TLX} h={22} kind="memory" dim={!honest} label={honest ? `${TRUE_US} µs` : ''} />
        </Track>
        <g className="v-move" style={{ transform: `translate(0px, ${TLY + 22 + 12}px)` }}>
          <Span x={TLX} y={0} w={barW} h={14} kind={honest ? 'token' : 'error'} label={reported >= 1000 ? `${(reported / 1000).toFixed(0)} ms` : `${reported.toFixed(0)} µs`} />
        </g>
        <Label x={TLX} y={TLY + 22 + 8} tone={3} size={9}>true kernel time</Label>
        <Label x={TLX} y={TLY + 22 + 40} tone={3} size={9}>what you report</Label>

        {/* the honest anchor line */}
        <line x1={px(TRUE_US)} x2={px(TRUE_US)} y1={TLY - 12} y2={TLY + 48} stroke="var(--memory)" strokeDasharray="3 3" />

        {/* per-step explanation */}
        <Fade show={st >= 1 && st <= 5}>
          <g>
            {st === 1 && (
              <>
                <text x={TLX} y={TLY + 96} className="v-label" style={{ fill: 'var(--error)', fontSize: 10.5, fontWeight: 600 }}>
                  First call = 180 ms of PTX compilation. Your kernel takes 42 µs. You are 4,000× off.
                </text>
                <text x={TLX} y={TLY + 114} className="v-label t3" style={{ fontSize: 9.5 }}>
                  Fix: run the kernel a few times before timing. JIT compile once, then measure.
                </text>
              </>
            )}
            {st === 2 && (
              <>
                <text x={TLX} y={TLY + 96} className="v-label" style={{ fill: 'var(--error)', fontSize: 10.5, fontWeight: 600 }}>
                  CUDA is async. `kernel()` returns before the GPU runs it. Without sync you time the CPU.
                </text>
                <text x={TLX} y={TLY + 114} className="v-label t3" style={{ fontSize: 9.5 }}>
                  Fix: torch.cuda.synchronize(), or better, time with CUDA events.
                </text>
              </>
            )}
            {st === 3 && (
              <>
                <text x={TLX} y={TLY + 96} className="v-label" style={{ fill: 'var(--error)', fontSize: 10.5, fontWeight: 600 }}>
                  H100 L2 is 50 MB. Your 4 MB tensor stays cached → 3× faster than HBM. Not a real speedup.
                </text>
                <text x={TLX} y={TLY + 114} className="v-label t3" style={{ fontSize: 9.5 }}>
                  Fix: write a buffer larger than L2 before each timed rep.
                </text>
              </>
            )}
            {st === 4 && (
              <>
                <text x={TLX} y={TLY + 96} className="v-label" style={{ fill: 'var(--error)', fontSize: 10.5, fontWeight: 600 }}>
                  One iteration on a shared GPU: you measured scheduling jitter (55 µs vs 42 µs true).
                </text>
                <text x={TLX} y={TLY + 114} className="v-label t3" style={{ fontSize: 9.5 }}>
                  Fix: repeat {iters}× and take the median. Medians ignore the outlier tails.
                </text>
              </>
            )}
          </g>
        </Fade>

        {/* final: roofline read */}
        <Fade show={st >= 5}>
          <g>
            <rect x={TLX} y={TLY + 80} width={TLW} height={66} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x={TLX + 12} y={TLY + 100} tone={1} size={11} kind="token" bold>
              {`${bytes / 1e6} MB moved in ${TRUE_US} µs = ${(bytes / TRUE_US / 1e3).toFixed(0)} GB/s = ${((bytes / TRUE_US / 1e3 / gpu.bw) * 100).toFixed(0)}% of peak`}
            </Label>
            <Label x={TLX + 12} y={TLY + 122} tone={3} size={9.5}>
              {`${iters} iterations, median reported. Now compare against the ${gpu.bw} GB/s roofline and decide if you are done.`}
            </Label>
            <Label x={TLX + 12} y={TLY + 138} tone={2} size={9.5}>
              Above 70%: stop tuning. Below 30%: profile — you are probably launch- or occupancy-bound.
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}