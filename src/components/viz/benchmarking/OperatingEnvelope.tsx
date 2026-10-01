import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, fmt } from '../lib/svg';
import { decodeStep, gpuById, modelById, GPUS } from '../lib/perfModel';

// A two-dimensional sweep, drawn the way a real harness plots it: one axis is
// concurrency, the other is context length, and every cell is one measurement.
// This is the shape of the operating envelope.

const STEPS = [
  'One measurement is a point',
  'Sweep one axis',
  'Sweep the second',
  'The grid',
  'Read the cells',
  'The safe region',
  'The envelope',
  'Breaking it on purpose',
];
const CAPTIONS = [
  'A single run tells you one point: this concurrency, this context length, this result.',
  'Sweep concurrency at a fixed context and you get a curve: throughput rises, latency rises, memory fills.',
  'Sweep context length as well, and a second axis appears that changes what concurrency is affordable.',
  'The grid is the raw output of any serious sweep: one cell per measurement, colour by the metric.',
  'Each cell is a separate run with its own warm-up, its own repeats, its own median. Read them as distributions.',
  'The safe region is the set of cells that meet your SLO with memory headroom left.',
  'Their boundary is the operating envelope: inside it you are safe, outside it you break something.',
  'Push past the edge deliberately — that is how you find it, and how you discover what failure looks like.',
];

const CONTEXTS = [512, 1024, 2048, 4096, 8192];
const BATCHES = [1, 8, 32, 64, 128, 256];

type Cell = { fits: boolean; thr: number; perUser: number; mem: number; util: number; good: boolean };

const cellFor = (model: ReturnType<typeof modelById>, gpu: ReturnType<typeof gpuById>, b: number, ctx: number): Cell => {
  const r = decodeStep({ model, gpu, batch: b, ctx });
  // a rough SLO check: the p95-ish TPOT must stay under 80 ms and it must fit
  const good = r.fits && r.t * 1000 < 80;
  return { fits: r.fits, thr: r.total, perUser: r.perUser, mem: r.memUsed / gpu.mem, util: r.memUsed / gpu.mem, good };
};

const X0 = 116;
const CW = 88; // cell width
const CH = 34; // cell height
const Y0 = 132; // top of first row

export default function OperatingEnvelope() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [gpuId, setGpuId] = useState('h100');
  const [modelId, setModelId] = useState('llama8b');
  const [maxCtx, setMaxCtx] = useState(8192);
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);

  useEffect(() => { setHover(null); }, [gpuId, modelId]); // eslint-disable-line react-hooks/exhaustive-deps

  const gpu = gpuById(gpuId);
  const model = modelById(modelId);
  const ctxs = useMemo(() => CONTEXTS.filter((c) => c <= maxCtx), [maxCtx]);
  const grid = useMemo(
    () => BATCHES.map((b) => ctxs.map((c) => cellFor(model, gpu, b, c))),
    [model, gpu, ctxs],
  );

  const safe = grid.flat().filter((c) => c.good);
  const thrMax = Math.max(...grid.flat().map((c) => (c.fits ? c.thr : 0)), 1);
  const sel = hover ? grid[hover.r][hover.c] : null;

  const showGrid = s.step >= 3;
  const showSafe = s.step >= 5;
  const showEdge = s.step >= 6;
  const showBreak = s.step >= 7;

  return (
    <Stage
      rootRef={ref}
      title="The operating envelope: one cell per measurement"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'compute', 'memory', ['error', 'out of memory'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Grid', value: `${BATCHES.length} × ${ctxs.length}`, kind: 'text', note: 'batch × context, all cells measured' },
          { label: 'Safe cells', value: `${safe.length} / ${BATCHES.length * ctxs.length}`, kind: safe.length ? 'token' : 'error', note: 'meet the SLO and fit' },
          { label: 'Best safe', value: safe.length ? `${fmt.int(Math.max(...safe.map((c) => c.thr)))} tok/s` : '—', kind: 'compute', note: 'peak goodput in the region' },
          { label: 'Selection', value: sel ? `${fmt.int(sel.thr)} tok/s` : 'hover a cell', kind: sel?.fits ? 'memory' : 'error', note: sel ? `${(sel.mem * 100).toFixed(0)}% of GPU memory` : 'batch × context' },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={[{ value: 'llama8b', label: 'Llama-8B' }, { value: 'qwen32b', label: 'Qwen-32B' }, { value: 'llama70b', label: 'Llama-70B' }]} onChange={setModelId} />
          <Segmented label="GPU" value={gpuId} options={GPUS.filter((g) => ['t4', 'a100', 'h100', 'h200'].includes(g.id)).map((g) => ({ value: g.id, label: g.id.toUpperCase() }))} onChange={setGpuId} />
          <Slider label="Longest context to sweep" value={maxCtx} min={512} max={8192} log onChange={setMaxCtx} format={(v) => `${fmt.int(v)} tok`} kind="memory" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A grid of measured cells: batch size down the side, context length across the top, each cell coloured by whether it fits in memory and meets the latency objective, with the safe region outlined.">
        {/* column headers: context */}
        {s.step >= 1 && ctxs.map((c, j) => (
          <Label key={c} x={X0 + j * CW + CW / 2} y={Y0 - 12} anchor="middle" tone={2} size={10}>{`${fmt.int(c / 1024)}k`}</Label>
        ))}
        {s.step >= 2 && <Label x={X0 + (ctxs.length * CW) / 2} y={Y0 - 28} anchor="middle" tone={3} size={9.5}>context length (tokens)</Label>}

        {/* row labels: batch */}
        {BATCHES.map((b, i) => (
          <Label key={b} x={X0 - 10} y={Y0 + i * CH + CH / 2 + 4} anchor="end" tone={2} size={10}>{b}</Label>
        ))}
        <Label x={X0 - 10} y={Y0 - 28} anchor="end" tone={3} size={9.5}>batch</Label>

        {/* the cells */}
        {showGrid && grid.map((row, i) =>
          row.map((cell, j) => {
            const x = X0 + j * CW;
            const y = Y0 + i * CH;
            const hot = hover && hover.r === i && hover.c === j;
            // colour by throughput, but OOM is always red and too-slow is hatched
            const frac = cell.fits ? cell.thr / thrMax : 0;
            return (
              <g key={`${i}-${j}`}>
                <rect
                  x={x} y={y} width={CW - 3} height={CH - 3} rx={2}
                  fill={!cell.fits ? 'url(#tps-hatch-red)' : cell.good ? 'var(--token)' : 'var(--idle)'}
                  fillOpacity={!cell.fits ? 1 : cell.good ? 0.18 + frac * 0.6 : 1}
                  stroke={!cell.fits ? 'var(--error)' : cell.good ? 'var(--token)' : 'var(--idle)'}
                  strokeWidth={hot ? 2 : 1}
                  onMouseEnter={() => setHover({ r: i, c: j })}
                  onMouseLeave={() => setHover(null)}
                  style={{ cursor: 'pointer', transition: 'fill-opacity .3s' }}
                />
                {cell.fits && cell.thr > 0 && CW > 40 && (
                  <text x={x + (CW - 3) / 2} y={y + CH / 2 + 1} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--text)' }}>
                    {cell.thr > 9999 ? `${(cell.thr / 1000).toFixed(0)}k` : fmt.int(cell.thr)}
                  </text>
                )}
                {!cell.fits && CW > 40 && (
                  <text x={x + (CW - 3) / 2} y={y + CH / 2 + 1} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--error)' }}>OOM</text>
                )}
              </g>
            );
          }),
        )}

        {/* the safe region outline */}
        {showSafe && (() => {
          const cells = BATCHES.map((_, i) => ctxs.map((__, j) => grid[i][j].good));
          const segs: { r: number; c0: number; c1: number }[] = [];
          cells.forEach((row, i) => {
            let start = -1;
            row.forEach((ok, j) => {
              if (ok && start < 0) start = j;
              if ((!ok || j === row.length - 1) && start >= 0) {
                segs.push({ r: i, c0: start, c1: ok && j === row.length - 1 ? j : j - 1 });
                start = -1;
              }
            });
          });
          return (
            <Fade show={showSafe}>
              {segs.map((sg, k) => (
                <rect
                  key={k}
                  x={X0 + sg.c0 * CW - 2} y={Y0 + sg.r * CH - 2}
                  width={(sg.c1 - sg.c0 + 1) * CW - 3 + 4} height={CH - 3 + 4}
                  rx={3} fill="none" stroke="var(--token)" strokeWidth={1.6} strokeDasharray="5 3"
                />
              ))}
              <Label x={X0} y={Y0 + BATCHES.length * CH + 22} size={10} kind="token" bold>
                the safe region: fits in memory and meets the latency objective
              </Label>
            </Fade>
          );
        })()}

        {/* the edge */}
        {showEdge && (
          <Fade show={showEdge}>
            <text x={X0} y={Y0 + BATCHES.length * CH + 42} className="v-label t2" style={{ fontSize: 10.5, fontWeight: 600 }}>
              Its boundary is the operating envelope. Inside, the service holds its SLO; outside, it does not.
            </text>
            <text x={X0} y={Y0 + BATCHES.length * CH + 58} className="v-label t3" style={{ fontSize: 9.5 }}>
              The envelope moves whenever you change model, GPU, quantization, context or engine — so re-sweep after every one.
            </text>
          </Fade>
        )}

        {/* the deliberate break */}
        {showBreak && (
          <Fade show={showBreak}>
            <text x={X0} y={Y0 + BATCHES.length * CH + 82} className="v-label" style={{ fill: 'var(--error)', fontSize: 10.5, fontWeight: 600 }}>
              Sweep past the edge on purpose. Find out whether it fails as an error, or as a hang.
            </text>
            <text x={X0} y={Y0 + BATCHES.length * CH + 98} className="v-label t3" style={{ fontSize: 9.5 }}>
              Red cells above are the OOM boundary. Everything right of it is a crash you must handle, not a slowdown.
            </text>
          </Fade>
        )}

        {/* the selection readout */}
        {sel && (
          <Fade show>
            <text x={X0 + 4} y={Y0 + BATCHES.length * CH + 122} className="v-label t2" style={{ fontSize: 10 }}>
              {`batch ${BATCHES[hover!.r]} · ${fmt.int(ctxs[hover!.c])} tokens → ${fmt.int(sel.perUser)} tok/s per user, ${(sel.mem * 100).toFixed(0)}% of memory`}
            </text>
          </Fade>
        )}
      </svg>
    </Stage>
  );
}