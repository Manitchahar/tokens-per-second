import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, fmt } from '../lib/svg';

// The point of this visual: goodput is not monotonic in load. Push past the
// knee and throughput keeps climbing while goodput falls, because the requests
// that miss the SLO no longer count.

const STEPS = [
  'An objective, not a hope',
  'Two curves, one load axis',
  'Low load: they agree',
  'The SLO bites',
  'Goodput turns over',
  'Past the optimum',
  'The honest summary',
];
const CAPTIONS = [
  'An SLO is a number with a percentile, a threshold and a window. "p95 TTFT under 600 ms over 5 minutes" is checkable.',
  'Grey is total throughput: tokens produced, regardless of whether anyone got a usable answer.',
  'At low load almost every request is fast, so goodput and throughput are nearly the same.',
  'As load rises, requests start queueing and some miss the SLO. They still count in throughput but not in goodput.',
  'Goodput peaks — the best offered load is here — and then falls while throughput is still climbing.',
  'Running past the optimum is strictly worse: more tokens per second, fewer satisfied users, and a bigger bill.',
  'Report goodput, not throughput. Throughput is what the GPU did; goodput is what the product delivered.',
];

const PRESET: Record<number, number> = { 0: 50, 1: 50, 2: 30, 3: 55, 4: 70, 5: 95, 6: 95 };

// offered load 0..100% of capacity
const CAPACITY = 5200; // output tokens/s at saturation

const throughput = (p: number) => CAPACITY * Math.pow(p / 100, 0.42); // rises fast, then flattens
// fraction of requests that meet the SLO: falls steeply as utilisation climbs
const sloPass = (p: number) => {
  const x = p / 100;
  return x <= 0.55 ? 1 : Math.max(0, 1 - Math.pow((x - 0.55) / 0.45, 1.7) * 1.02);
};

const X0 = 92;
const X1 = 614;
const Y0 = 296;
const Y1 = 96;
const MAXY = 5600;
const lx = (p: number) => X0 + (p / 100) * (X1 - X0);
const ly = (v: number) => Y0 - Math.min(1, v / MAXY) * (Y0 - Y1);

const curve = Array.from({ length: 101 }, (_, p) => ({
  p,
  thr: throughput(p),
  good: throughput(p) * sloPass(p),
}));

const best = curve.reduce((a, b) => (b.good > a.good ? b : a));

export default function GoodputCurve() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [load, setLoad] = useState(PRESET[0]);
  useEffect(() => setLoad(PRESET[s.step]), [s.step]); // eslint-disable-line react-hooks/exhaustive-deps

  const thr = throughput(load);
  const pass = sloPass(load);
  const good = thr * pass;

  const thrPath = curve.map((c, i) => `${i ? 'L' : 'M'}${lx(c.p).toFixed(1)},${ly(c.thr).toFixed(1)}`).join(' ');
  const goodPath = curve.map((c, i) => `${i ? 'L' : 'M'}${lx(c.p).toFixed(1)},${ly(c.good).toFixed(1)}`).join(' ');

  const gx = useTween(lx(load), 600);
  const gyT = useTween(ly(thr), 600);
  const gyG = useTween(ly(good), 600);

  const showThr = s.step >= 1;
  const showBoth = s.step >= 2;
  const showPass = s.step >= 3;
  const showTurn = s.step >= 4;
  const showPast = s.step >= 5;
  const showBest = s.step >= 6;

  const missed = Math.round((1 - pass) * 100);

  return (
    <Stage
      rootRef={ref}
      title="Goodput is not throughput: the best operating point is not the busiest one"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', ['text', 'throughput'], 'token', 'error', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Throughput', value: `${fmt.int(thr)} tok/s`, kind: 'compute', note: 'produced, SLO ignored' },
          { label: 'Goodput', value: `${fmt.int(good)} tok/s`, kind: 'token', note: `${pass * 100 >= 99.5 ? '~all' : (pass * 100).toFixed(0)}% met the SLO` },
          { label: 'Requests missing', value: `${missed}%`, kind: missed > 10 ? 'error' : 'idle', note: 'counted, not good' },
          { label: 'Best load', value: `${best.p}%`, kind: 'memory', note: `${fmt.int(best.good)} tok/s goodput` },
        ]} />
      }
      controls={
        <Slider label="Offered load (% of capacity)" value={load} min={1} max={100} onChange={setLoad} format={(v) => `${v}%`} kind="token" />
      }
    >
      <svg viewBox="0 0 640 356" role="img" aria-label="A chart with offered load on the horizontal axis: throughput rises and flattens, while goodput rises to a peak and then falls, with a gap between them showing requests that missed the SLO.">
        {/* axes */}
        <line x1={X0} x2={X1} y1={Y0} y2={Y0} className="v-rule" />
        <line x1={X0} x2={X0} y1={Y1 - 14} y2={Y0} className="v-rule" />
        {[0, 25, 50, 75, 100].map((p) => (
          <Label key={p} x={lx(p)} y={Y0 + 14} anchor="middle" tone={3} size={9}>{p}%</Label>
        ))}
        <Label x={(X0 + X1) / 2} y={Y0 + 30} anchor="middle" tone={2} size={10.5} mono={false}>offered load →</Label>
        {[0, 2000, 4000].map((v) => (
          <Label key={v} x={X0 - 6} y={ly(v) + 3} anchor="end" tone={3} size={9}>{v ? `${v / 1000}k` : '0'}</Label>
        ))}
        <text x={22} y={(Y0 + Y1) / 2} transform={`rotate(-90 22 ${(Y0 + Y1) / 2})`} textAnchor="middle" className="v-label t2" style={{ fontSize: 10.5 }}>output tokens/s</text>

        {/* the SLO threshold */}
        <Fade show={showBoth}>
          <line x1={X0} x2={X1} y1={Y1 + 6} y2={Y1 + 6} stroke="var(--text-3)" strokeDasharray="4 4" />
          <Label x={X1} y={Y1 - 2} anchor="end" tone={3} size={9.5}>the SLO region — the part that counts</Label>
        </Fade>

        {/* throughput */}
        {showThr && (
          <Fade show={showThr}>
            <path d={thrPath} fill="none" stroke="var(--compute)" strokeWidth={1.8} />
            <Label x={X1} y={ly(CAPACITY) - 7} anchor="end" size={10} kind="compute">throughput</Label>
          </Fade>
        )}

        {/* goodput */}
        {showBoth && (
          <Fade show={showBoth}>
            <path d={goodPath} fill="none" stroke="var(--token)" strokeWidth={2.2} />
            <Label x={X1 - 4} y={ly(curve[70].good) + 22} anchor="end" size={10} kind="token" bold>goodput</Label>
          </Fade>
        )}

        {/* the wasted region between the two curves */}
        {showPass && pass < 0.98 && (
          <Fade show={showPass}>
            <path
              d={`${curve.filter((c) => c.p >= load).map((c, i) => `${i ? 'L' : 'M'}${lx(c.p).toFixed(1)},${ly(c.thr).toFixed(1)}`).join(' ')} ${curve.filter((c) => c.p >= load).reverse().map((c) => `L${lx(c.p).toFixed(1)},${ly(c.good).toFixed(1)}`).join(' ')} Z`}
              fill="var(--error)"
              opacity={0.16}
            />
          </Fade>
        )}

        {/* the optimum */}
        {showTurn && (
          <Fade show={showTurn}>
            <line x1={lx(best.p)} x2={lx(best.p)} y1={Y1 - 14} y2={Y0} stroke="var(--token)" strokeDasharray="3 4" />
            <circle cx={lx(best.p)} cy={ly(best.good)} r={5} fill="var(--token)" stroke="var(--ground-2)" strokeWidth={2} />
            <Label x={lx(best.p) + 7} y={ly(best.good) - 6} size={9.5} kind="token" bold>{`peak goodput at ${best.p}% load`}</Label>
          </Fade>
        )}

        {/* past the optimum */}
        {showPast && load > best.p && (
          <Fade show={showPast}>
            <text x={X0} y={Y0 + 50} className="v-label" style={{ fill: 'var(--error)', fontSize: 10.5, fontWeight: 600 }}>
              {`${((thr / good - 1) * 100).toFixed(0)}% of the throughput is wasted work nobody can use`}
            </text>
            <text x={X0} y={Y0 + 66} className="v-label t3" style={{ fontSize: 9.5 }}>
              and you are paying for every one of those tokens
            </text>
          </Fade>
        )}

        {/* the moving point */}
        <g>
          <line x1={gx} x2={gx} y1={Y1 - 14} y2={Y0} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="2 3" />
          <g className="v-move" style={{ transform: `translate(${gx}px, ${gyT}px)` }}>
            <circle r={4.5} fill="var(--compute)" stroke="var(--ground-2)" strokeWidth={1.8} />
          </g>
          <g className="v-move" style={{ transform: `translate(${gx}px, ${gyG}px)` }}>
            <circle r={4.5} fill="var(--token)" stroke="var(--ground-2)" strokeWidth={1.8} />
          </g>
        </g>

        {showBest && (
          <Fade show={showBest}>
            <text x={X0} y={Y0 + 88} className="v-label t2" style={{ fontSize: 10.5, fontWeight: 600 }}>
              goodput = throughput × fraction of requests that met the SLO
            </text>
            <text x={X0} y={Y0 + 104} className="v-label t3" style={{ fontSize: 9.5 }}>
              This is the number to optimize, and the number to put on the wall next to cost.
            </text>
          </Fade>
        )}
      </svg>
    </Stage>
  );
}