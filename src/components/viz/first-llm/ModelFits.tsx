import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, fmt } from '../lib/svg';
import { GPUS, MODELS, gpuById, modelById, kvBytesPerToken, decodeStep } from '../lib/perfModel';

// Choosing a model is mostly one comparison: a number of bytes against a number of gigabytes.
// Everything else — quality, speed, whether you can serve anyone at all — falls out of it.

const STEPS = [
  'One number to start from',
  'Weights are bytes on the GPU',
  'The GPU is a hard wall',
  'Headroom is not optional',
  'GQA shrinks the KV cache',
  'The fit, at a glance',
  'Speed follows the same bytes',
  'The whole choice, in one look',
];
const CAPTIONS = [
  'An open model’s parameter count is the number that predicts almost everything you care about: how good it is, how much memory it needs, and how fast it will run.',
  'At inference every weight is read but never written. In BF16 that is 2 bytes each, so Llama-3.1-8B needs 8.03B × 2 = 16.1 GB just to exist.',
  'That memory has to fit in one GPU. A 16 GB T4 cannot hold a 16.1 GB model, whatever else you do — weights are a floor, not an estimate.',
  'Weights are only part of the bill. Every conversation also stores keys and values that grow as the answer grows. Leave no room and one long answer is an out-of-memory crash.',
  'Llama-3.1-8B has 32 query heads but only 8 key/value heads. Grouped-query attention cuts the KV cache by 4×, which buys back the headroom you need.',
  'Every open model against every GPU, at the precision you picked. The hatched cells cannot start at all; the pale red ones have no room for a conversation.',
  'Each generated token re-reads every weight once, so decode speed is weight bytes ÷ bandwidth. Halve the bytes and you roughly double the tokens per second.',
  'Parameter count, precision, GPU and context length, side by side. Choose the biggest model that fits with 15% to spare, and let quality decide between the ones that do.',
];

type Cell = 'fits' | 'tight' | 'no';

const PRESET_PRECISION: Record<string, number> = { bf16: 2, fp8: 1, int4: 0.5 };
const PRECISIONS = [
  { value: 'bf16', label: 'BF16 · 2 B' },
  { value: 'fp8', label: 'FP8 · 1 B' },
  { value: 'int4', label: 'INT4 · 0.5 B' },
];

const short = (name: string) => name.replace('Qwen2.5-', 'Qwen2.5 ').replace('Llama-3.', 'L3.');
const fmtPrec = (b: number) => (b === 2 ? 'BF16' : b === 1 ? 'FP8' : 'INT4');

const BAR = { x: 56, w: 528, y: 60, h: 30 };
const MX = { x0: 56, labelW: 92, colW: 72, y0: 194, rowH: 24 };

export default function ModelFits() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [modelId, setModelId] = useState('llama8b');
  const [gpuId, setGpuId] = useState('t4');
  const [prec, setPrec] = useState('bf16');
  const [ctx, setCtx] = useState(2048);

  const model = modelById(modelId);
  const gpu = gpuById(gpuId);
  const wBytes = PRESET_PRECISION[prec];

  // Every number on this stage comes from the same shared model the course uses everywhere.
  const r = useMemo(() => decodeStep({ model, gpu, batch: 1, ctx, wBytes }), [model, gpu, ctx, wBytes]);

  const demand = r.weights + r.kvTotal + r.activations;
  const spanGB = Math.max(gpu.mem, demand) * 1.02;
  const pxPerGB = BAR.w / spanGB;
  const capPx = BAR.x + gpu.mem * pxPerGB;
  const wPx = r.weights * pxPerGB;
  const kPx = r.kvTotal * pxPerGB;
  const aPx = r.activations * pxPerGB;
  const endPx = BAR.x + demand * pxPerGB;
  const over = demand > gpu.mem;
  const free = gpu.mem - demand;

  const wW = useTween(wPx, 650);
  const kW = useTween(kPx, 650);
  const aW = useTween(aPx, 650);
  const ease = { transition: 'width .65s cubic-bezier(.16,1,.3,1)' };

  const cells = useMemo(
    () =>
      MODELS.map((m) => ({
        m,
        rows: GPUS.map((g) => {
          const w = m.params * wBytes;
          const kv = ctx * kvBytesPerToken(m);
          const cell: Cell = w + kv + 0.04 * g.mem > g.mem ? 'no' : w + 0.04 * g.mem > g.mem ? 'tight' : 'fits';
          return { g, cell, w };
        }),
      })),
    [wBytes, ctx],
  );

  const matrixBottom = MX.y0 + cells.length * MX.rowH;

  return (
    <Stage
      rootRef={ref}
      title="Will it fit? Model × precision × GPU, before you download anything"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', ['idle', 'headroom'], ['error', 'out of memory'], 'compute', 'token']}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'Weights on the GPU', value: fmt.gb(r.weights), kind: 'memory', note: `${model.name} at ${wBytes} bytes/weight` },
            {
              label: 'KV cache, one conversation',
              value: fmt.gb(r.kvTotal),
              kind: 'memory',
              note: `${fmt.int(ctx)} tokens · ${fmt.bytes(kvBytesPerToken(model))} per token`,
            },
            {
              label: 'Left over',
              value: over ? `${fmt.gb(-free)} over` : fmt.gb(free),
              kind: over ? 'error' : 'idle',
              note: over ? 'this configuration cannot start' : `of ${fmt.gb(gpu.mem)} on a ${gpu.name}`,
            },
            {
              label: 'Decode, one user',
              value: `${r.perUser.toFixed(1)} tok/s`,
              kind: 'token',
              note: `${r.bound}-bound · ${(r.t * 1e3).toFixed(0)} ms per token`,
            },
          ]}
        />
      }
      controls={
        <>
          <Segmented label="Weights stored as" value={prec} options={PRECISIONS} onChange={setPrec} />
          <Segmented label="Model" value={modelId} options={MODELS.map((m) => ({ value: m.id, label: short(m.name) }))} onChange={setModelId} />
          <Slider
            label="Context: prompt + answer"
            value={ctx}
            min={512}
            max={131072}
            log
            onChange={setCtx}
            format={(v) => `${fmt.int(v)} tokens`}
            kind="memory"
          />
        </>
      }
    >
      <svg viewBox="0 0 640 384" role="img" aria-label="A memory comparison bar showing the weights, KV cache and workspace of a chosen model against the memory capacity of a chosen GPU, plus a grid of every open model against every GPU marking which combinations fit, leave no headroom, or run out of memory.">
        {/* ── step 0: the one number */}
        <Fade show={st === 0}>
          <Label x={56} y={40} tone={1} size={11.5} bold mono={false}>Start from one number: parameters</Label>
          {MODELS.map((m, i) => {
            const x = 56 + i * 108;
            const on = m.id === modelId;
            return (
              <g key={m.id}>
                <rect x={x} y={58} width={98} height={40} rx={3}
                  fill={on ? 'var(--compute-dim)' : 'var(--ground-3)'}
                  stroke={on ? 'var(--compute)' : 'var(--rule-2)'} strokeWidth={on ? 1.3 : 1} className="v-cell" />
                <text x={x + 8} y={74} className="v-label" style={{ fontSize: 9.5, fill: on ? 'var(--text)' : 'var(--text-2)', fontWeight: on ? 700 : 400 }}>{short(m.name)}</text>
                <text x={x + 8} y={90} className="v-label" style={{ fontSize: 11, fill: 'var(--compute)', fontWeight: 600 }}>
                  {`${(m.params / 1e9).toFixed(m.params < 1e9 ? 2 : 1)}B params`}
                </text>
              </g>
            );
          })}
          <text x={56} y={128} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>
            Bigger is better at reasoning, and bigger in every other way. Pick a column, then find out what it costs.
          </text>
          <Label x={56} y={168} tone={1} size={11.5} bold mono={false}>Weights on the GPU</Label>
        </Fade>

        {/* ── steps 1+: the memory bar */}
        <g style={{ opacity: st >= 1 ? 1 : 0 }} className="v-fade">
          <Label x={56} y={40} tone={1} size={11.5} bold mono={false}>{`${model.name} on a ${gpu.name}`}</Label>

          {/* everything past the GPU's capacity is unreachable */}
          <rect x={capPx} y={BAR.y - 6} width={Math.max(0, BAR.x + BAR.w - capPx)} height={BAR.h + 12}
            fill="url(#tps-hatch)" opacity={st >= 2 ? 0.9 : 0} className="v-fade" />
          {st >= 2 && (
            <g className="v-fade">
              <line x1={capPx} x2={capPx} y1={BAR.y - 13} y2={BAR.y + BAR.h + 5} stroke="var(--error)" strokeWidth={1.6} />
              <text x={Math.min(capPx, 560)} y={BAR.y - 19} textAnchor="middle" className="v-label bold" style={{ fontSize: 9.5, fill: 'var(--error)' }}>
                {`${gpu.name}: ${fmt.gb(gpu.mem)}`}
              </text>
            </g>
          )}

          {/* weights */}
          <rect x={BAR.x} y={BAR.y} width={Math.max(0, wW)} height={BAR.h} rx={2}
            fill="var(--memory)" stroke="var(--memory)" strokeWidth={1} style={ease} className="v-cell" />
          {st >= 1 && wW > 58 && (
            <text x={BAR.x + wW / 2} y={BAR.y + 19} textAnchor="middle" className="v-label bold" style={{ fontSize: 10, fill: '#0e1116' }}>
              {`weights ${fmt.gb(r.weights)}`}
            </text>
          )}

          {/* KV cache */}
          {st >= 4 && (
            <>
              <rect x={BAR.x + wW} y={BAR.y} width={Math.max(0, kW)} height={BAR.h} rx={2}
                fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} style={ease} className="v-cell" />
              {kW > 46 && (
                <text x={BAR.x + wW + kW / 2} y={BAR.y + 19} textAnchor="middle" className="v-label" style={{ fontSize: 10, fill: 'var(--text)' }}>
                  {`KV ${fmt.gb(r.kvTotal)}`}
                </text>
              )}
            </>
          )}

          {/* workspace */}
          {st >= 3 && (
            <>
              <rect x={BAR.x + wW + kW} y={BAR.y} width={Math.max(0, aW)} height={BAR.h} rx={2}
                fill="var(--ground-3)" stroke="var(--memory)" strokeWidth={1} strokeDasharray="3 3" style={ease} className="v-cell" />
              {aW > 50 && (
                <text x={BAR.x + wW + kW + aW / 2} y={BAR.y + 19} textAnchor="middle" className="v-label t3" style={{ fontSize: 9.5 }}>
                  workspace
                </text>
              )}
            </>
          )}

          {/* the part that does not fit */}
          {over && st >= 2 && (
            <g className="v-fade">
              <rect x={capPx} y={BAR.y} width={Math.max(0, endPx - capPx)} height={BAR.h} rx={2}
                fill="url(#tps-hatch-red)" stroke="var(--error)" strokeWidth={1.4} />
              <text x={Math.min(endPx, BAR.x + BAR.w) - 6} y={BAR.y - 8} anchor="end" className="v-label bold" style={{ fontSize: 10, fill: 'var(--error)' }}>
                {`${fmt.gb(-free)} more than the GPU has`}
              </text>
            </g>
          )}

          {/* axis */}
          <line x1={BAR.x} x2={BAR.x} y1={BAR.y + BAR.h + 6} y2={BAR.y + BAR.h + 12} className="v-rule" />
          <Label x={BAR.x} y={BAR.y + BAR.h + 26} size={9} tone={3}>0</Label>
          <Label x={BAR.x + BAR.w} y={BAR.y + BAR.h + 26} anchor="end" size={9} tone={3}>{fmt.gb(spanGB)}</Label>
          {st >= 3 && !over && (
            <Label x={Math.min(endPx, BAR.x + BAR.w - 4)} y={BAR.y + BAR.h + 26} anchor="end" size={9} tone={2}>
              {`in use ${fmt.gb(demand)}`}
            </Label>
          )}
        </g>

        {/* ── the per-step footnote under the bar */}
        {st === 3 && (
          <text x={56} y={140} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>
            The dashed slice is CUDA context, activations and kernel workspace. The KV slice grows with every token generated — Module 05 is entirely about that.
          </text>
        )}
        {st === 4 && (
          <text x={56} y={140} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>
            {`${model.name} runs ${model.heads} query heads on ${model.kvHeads} key/value heads — ${model.heads / model.kvHeads}× less KV cache, same model.`}
          </text>
        )}
        {st === 1 && (
          <text x={56} y={140} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>
            Read but never written: that is why bytes-1 is not bytes-2 like a training checkpoint, and why this memory is permanent for the life of the process.
          </text>
        )}

        {/* ── steps 5-7: the fit matrix */}
        <g style={{ opacity: st >= 5 ? 1 : 0 }} className="v-fade">
          <line x1={56} y1={156} x2={584} y2={156} className="v-rule faint" />
          <Label x={56} y={178} tone={1} size={11.5} bold mono={false}>Every open model against every GPU</Label>
          <Label x={584} y={178} anchor="end" tone={3} size={9}>
            {`weights at ${wBytes} bytes + KV for ${fmt.int(ctx)} tokens`}
          </Label>

          {GPUS.map((g, i) => {
            const cx = MX.x0 + MX.labelW + i * MX.colW;
            const on = g.id === gpuId;
            return (
              <g key={g.id}>
                <text x={cx + MX.colW / 2 - 4} y={190} textAnchor="middle" className="v-label"
                  style={{ fontSize: 9.5, fill: on ? 'var(--text)' : 'var(--text-3)', fontWeight: on ? 700 : 400 }}>
                  {g.id.toUpperCase()}
                </text>
              </g>
            );
          })}

          {cells.map((row, ri) => (
            <g key={row.m.id}>
              <text x={MX.x0 + MX.labelW - 10} y={MX.y0 + ri * MX.rowH + 14} textAnchor="end" className="v-label"
                style={{ fontSize: 9.5, fill: row.m.id === modelId ? 'var(--text)' : 'var(--text-2)', fontWeight: row.m.id === modelId ? 700 : 400 }}>
                {short(row.m.name)}
              </text>
              {row.rows.map((c) => {
                const ci = GPUS.findIndex((g) => g.id === c.g.id);
                const cx = MX.x0 + MX.labelW + ci * MX.colW;
                const sel = c.g.id === gpuId;
                const fill = c.cell === 'no' ? 'url(#tps-hatch-red)' : c.cell === 'tight' ? 'var(--error-dim)' : 'var(--ground-3)';
                const stroke = c.cell === 'fits' ? (sel ? 'var(--memory)' : 'var(--rule-2)') : 'var(--error)';
                return (
                  <g key={c.g.id} className="v-cell">
                    <rect x={cx} y={MX.y0 + ri * MX.rowH} width={MX.colW - 8} height={MX.rowH - 5} rx={2}
                      fill={fill} stroke={stroke} strokeWidth={sel ? 1.6 : 1} />
                    <text x={cx + (MX.colW - 8) / 2} y={MX.y0 + ri * MX.rowH + 12} textAnchor="middle" className="v-label"
                      style={{ fontSize: 9, fill: c.cell === 'fits' ? 'var(--text-2)' : 'var(--error)', fontWeight: c.cell === 'fits' ? 400 : 600 }}>
                      {c.cell === 'fits' ? fmt.gb(c.w) : c.cell === 'tight' ? 'no room' : 'OOM'}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}

          <Label x={56} y={matrixBottom + 6} tone={3} size={9}>
            grey = fits · pale red = weights only, no headroom · hatched = cannot start
          </Label>

          {st >= 6 && (
            <g className="v-fade">
              <Label x={56} y={matrixBottom + 26} tone={2} size={10} mono={false}>
                {`Decode reads all ${fmt.gb(r.weights)} of weights once per token: ${fmt.gb(r.weights)} ÷ ${fmt.si(gpu.bw / 1e12)} TB/s = ${(r.tWeights * 1e3).toFixed(1)} ms.`}
              </Label>
              <Label x={56} y={matrixBottom + 44} tone={1} size={10.5} mono={false} bold>
                {`${model.name} in ${fmtPrec(wBytes)}: ${r.perUser.toFixed(1)} tokens/second for one user on a ${gpu.name}.`}
              </Label>
            </g>
          )}
        </g>
      </svg>
    </Stage>
  );
}
