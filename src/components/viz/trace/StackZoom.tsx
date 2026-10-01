import { useRef, type ReactNode } from 'react';
import { Stage } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade } from '../lib/svg';

const X0 = 118;
const X1 = 626;
const WIDTH = X1 - X0;
const ROWS = [
  { key: 'product', label: 'Product', y: 34, h: 20, scale: '620 ms' },
  { key: 'fleet', label: 'Fleet', y: 98, h: 30, scale: '620 ms' },
  { key: 'engine', label: 'Engine', y: 172, h: 20, scale: '620 ms' },
  { key: 'runtime', label: 'Runtime', y: 236, h: 20, scale: '25 ms' },
  { key: 'kernels', label: 'Kernels', y: 300, h: 20, scale: '0.7 ms' },
  { key: 'hardware', label: 'Hardware', y: 364, h: 30, scale: '0.2 ms' },
];

const STEPS = ['The product', 'The fleet', 'The serving engine', 'The model runtime', 'The kernels', 'The hardware'];
const CAPTIONS = [
  'From the outside, a request is one block of time: 620 ms from Enter to the last word.',
  'Zoom out to the fleet: many GPU servers (replicas) each run many requests. A router picks which replica serves yours.',
  'Inside one replica, the serving engine queues your request, prefills the prompt, then decodes token by token.',
  'Zoom into one 25 ms decode step: the model runs its 32 layers in order, then picks a token.',
  'Zoom into one layer: it is a sequence of GPU kernels. Most of the time goes to matrix multiplies and attention.',
  'Zoom into one matrix multiply: the memory system streams weights in while the math units sit mostly idle. Memory-bound.',
];

// deterministic pseudo-random for the fleet lanes
const rand = (i: number) => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };

function Zoom({ from, to, y1, y2, show }: { from: [number, number]; to: [number, number]; y1: number; y2: number; show: boolean }) {
  return (
    <Fade show={show}>
      <polygon points={`${from[0]},${y1} ${from[1]},${y1} ${to[1]},${y2} ${to[0]},${y2}`} fill="rgb(232 230 225 / 0.035)" stroke="var(--rule-2)" strokeDasharray="2 3" />
    </Fade>
  );
}

export default function StackZoom() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const rowOpacity = (i: number) => (i > st ? 0 : i === st ? 1 : 0.4);
  const sx = (f: number) => X0 + f * WIDTH;

  // engine row geometry (fractions of 620 ms)
  const E = { queue: [40 / 620, 95 / 620], prefill: [95 / 620, 215 / 620] };
  const decodeSteps = Array.from({ length: 15 }, (_, i) => [(215 + i * 25) / 620, (215 + (i + 1) * 25) / 620]);
  const zoomStep = decodeSteps[6];

  // runtime: 32 layers + lm head + sampling
  const layers: ReactNode[] = [];
  const lw = 0.93 / 32;
  for (let i = 0; i < 32; i++) {
    const x = sx(i * lw);
    layers.push(<Span key={`a${i}`} x={x + 0.4} y={ROWS[3].y} w={lw * WIDTH * 0.42 - 0.6} h={ROWS[3].h} kind="memory" dim />);
    layers.push(<Span key={`m${i}`} x={x + lw * WIDTH * 0.42 + 0.2} y={ROWS[3].y} w={lw * WIDTH * 0.58 - 0.8} h={ROWS[3].h} kind="memory" />);
  }
  const layerZoom: [number, number] = [sx(9 * lw), sx(10 * lw)];

  // kernels in one layer: name, relative duration (≈ bytes moved), kind
  const kernels: [string, number, 'memory' | 'compute'][] = [
    ['norm', 0.02, 'compute'], ['qkv', 0.14, 'memory'], ['rope', 0.02, 'compute'], ['attn', 0.12, 'memory'],
    ['o-proj', 0.1, 'memory'], ['norm', 0.02, 'compute'], ['gate+up', 0.32, 'memory'], ['act', 0.03, 'compute'], ['down', 0.23, 'memory'],
  ];
  let acc = 0;
  const kspans = kernels.map(([name, d, kind], i) => {
    const x = sx(acc);
    acc += d;
    return { name, x, w: d * WIDTH - 2, kind, i };
  });
  const gu = kspans[6];

  return (
    <Stage rootRef={ref} title="The same request at six zoom levels" steps={STEPS} captions={CAPTIONS} legend={['compute', 'memory', 'idle']} {...s}>
      <svg viewBox="0 0 640 410" role="img" aria-label="A request shown at six zoom levels: product, fleet, engine, runtime, kernels and hardware.">
        {ROWS.map((r, i) => (
          <g key={r.key} className="v-fade" style={{ opacity: rowOpacity(i) }}>
            <text x={X0 - 12} y={r.y + r.h / 2 + 4} textAnchor="end" className="v-track-label" style={{ fill: i === st ? 'var(--text)' : undefined }}>{r.label}</text>
            <Label x={X1} y={r.y - 5} anchor="end" tone={3} size={9.5}>{r.scale}</Label>
          </g>
        ))}

        {/* product */}
        <g className="v-fade" style={{ opacity: rowOpacity(0) }}>
          <Span x={X0} y={ROWS[0].y} w={WIDTH} h={ROWS[0].h} kind="token" dim label="one chat request, Enter to last word" labelTone="light" />
        </g>

        {/* fleet: 4 replicas, each busy with other requests; ours highlighted on replica 3 */}
        <Zoom from={[X0, X1]} to={[X0, X1]} y1={ROWS[0].y + ROWS[0].h + 3} y2={ROWS[1].y - 3} show={st >= 1} />
        <g className="v-fade" style={{ opacity: rowOpacity(1) }}>
          {[0, 1, 2, 3].map((lane) => {
            const y = ROWS[1].y + lane * 7.5;
            const spans: ReactNode[] = [];
            let t = rand(lane) * 0.1;
            let k = 0;
            while (t < 1) {
              const w = 0.12 + rand(lane * 31 + k) * 0.3;
              spans.push(<Span key={k} x={sx(t)} y={y} w={Math.min(w, 1 - t) * WIDTH - 2} h={6} kind="text" ghost />);
              t += w + 0.01;
              k++;
            }
            return <g key={lane}>{spans}</g>;
          })}
          <Span x={sx(0)} y={ROWS[1].y + 2 * 7.5} w={WIDTH} h={6} kind="token" />
          <Label x={X0} y={ROWS[1].y + ROWS[1].h + 12} tone={3} size={9.5}>4 replicas shown; yours is the green lane</Label>
        </g>

        {/* engine */}
        <Zoom from={[X0, X1]} to={[X0, X1]} y1={ROWS[1].y + ROWS[1].h + 16} y2={ROWS[2].y - 3} show={st >= 2} />
        <g className="v-fade" style={{ opacity: rowOpacity(2) }}>
          <Span x={sx(E.queue[0])} y={ROWS[2].y} w={(E.queue[1] - E.queue[0]) * WIDTH} h={ROWS[2].h} kind="idle" />
          <Span x={sx(E.prefill[0])} y={ROWS[2].y} w={(E.prefill[1] - E.prefill[0]) * WIDTH} h={ROWS[2].h} kind="compute" label="prefill" />
          {decodeSteps.map(([a, b], i) => (
            <Span key={i} x={sx(a) + 0.7} y={ROWS[2].y} w={(b - a) * WIDTH - 1.4} h={ROWS[2].h} kind="memory" style={{ opacity: i === 6 && st >= 3 ? 1 : st >= 3 ? 0.55 : 1 }} />
          ))}
        </g>

        {/* runtime: one decode step */}
        <Zoom from={[sx(zoomStep[0]), sx(zoomStep[1])]} to={[X0, X1]} y1={ROWS[2].y + ROWS[2].h + 3} y2={ROWS[3].y - 3} show={st >= 3} />
        <g className="v-fade" style={{ opacity: rowOpacity(3) }}>
          {layers}
          <Span x={sx(0.935)} y={ROWS[3].y} w={0.045 * WIDTH} h={ROWS[3].h} kind="memory" />
          <Span x={sx(0.983)} y={ROWS[3].y} w={0.017 * WIDTH} h={ROWS[3].h} kind="compute" />
          <Label x={X0} y={ROWS[3].y + ROWS[3].h + 12} tone={3} size={9.5}>32 layers (attention, then MLP), then the output head and sampling</Label>
        </g>

        {/* kernels: one layer */}
        <Zoom from={layerZoom} to={[X0, X1]} y1={ROWS[3].y + ROWS[3].h + 16} y2={ROWS[4].y - 3} show={st >= 4} />
        <g className="v-fade" style={{ opacity: rowOpacity(4) }}>
          {kspans.map((k) => (
            <Span key={k.i} x={k.x} y={ROWS[4].y} w={k.w} h={ROWS[4].h} kind={k.kind} dim={k.kind === 'compute'} label={k.name} labelTone={k.kind === 'compute' ? 'light' : 'dark'} />
          ))}
        </g>

        {/* hardware: one matmul */}
        <Zoom from={[gu.x, gu.x + gu.w]} to={[X0, X1]} y1={ROWS[4].y + ROWS[4].h + 3} y2={ROWS[5].y - 3} show={st >= 5} />
        <g className="v-fade" style={{ opacity: rowOpacity(5) }}>
          <text x={X0 + 4} y={ROWS[5].y - 4} className="v-label t3" style={{ fontSize: 9.5 }}>HBM reading weights</text>
          <Span x={X0} y={ROWS[5].y + 1} w={WIDTH} h={11} kind="memory" />
          {Array.from({ length: 24 }, (_, i) => {
            const x = sx(i / 24);
            return (
              <g key={i}>
                <Span x={x} y={ROWS[5].y + 17} w={WIDTH / 24 * 0.12} h={11} kind="compute" />
                <Span x={x + (WIDTH / 24) * 0.12 + 0.5} y={ROWS[5].y + 17} w={(WIDTH / 24) * 0.88 - 1} h={11} kind="idle" />
              </g>
            );
          })}
          <Label x={X0} y={ROWS[5].y + ROWS[5].h + 12} tone={3} size={9.5}>tensor cores: short bursts of math, long waits for data</Label>
        </g>
      </svg>
    </Stage>
  );
}
