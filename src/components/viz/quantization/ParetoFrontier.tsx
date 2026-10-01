import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Arrow, Dim, Fade, Chip, fmt } from '../lib/svg';
import { llama8b, llama70b, h100, decodeStepQuant, weightBytes, kvBytes, metaFracInt } from './quant';

/**
 * The frontier: for each (weight bits × KV bits), plot throughput against quality.
 * Points on the Pareto frontier beat every alternative on at least one axis.
 */

const STEPS = [
  'Three axes, not one',
  'Plotting the candidates',
  'The frontier is the answer',
  'Reading your own shape',
  'The choice you actually face',
];
const CAPTIONS = [
  'Every quantization decision trades three things: quality, speed, and memory. Collapse them to one score and you have thrown away the only information the trade-off contains.',
  'Here are eight candidate configurations, plotted as throughput against a quality index. The two together already tell most of the story: memory is a third axis and it is nearly collinear with throughput.',
  'Highlight the points nothing else dominates and you have the frontier. Anything below it is strictly worse than something you already have — delete it from consideration.',
  'Your frontier has a shape, and the shape is your workload. Short-context interactive serving has a steep one; long-context RAG flattens it out towards the memory-bound end.',
  'Pick a point, say why, and write down what would move you off it. That last clause is what makes this a decision rather than a preference.',
];

/** Bits pairs: [weight bits, kv bits]. */
const CONFIGS: { wb: number; kb: number; label: string }[] = [
  { wb: 16, kb: 16, label: 'BF16' },
  { wb: 8, kb: 16, label: 'FP8' },
  { wb: 8, kb: 8, label: 'W8A8 + KV8' },
  { wb: 16, kb: 8, label: 'BF16 + KV8' },
  { wb: 4, kb: 16, label: 'W4' },
  { wb: 4, kb: 8, label: 'W4 + KV8' },
  { wb: 4, kb: 4, label: 'W4 + KV4' },
  { wb: 8, kb: 4, label: 'W8 + KV4' },
];

const CTX = 4096;

/**
 * A quality index: 1.0 = BF16. Weight error dominates; KV error is weighted less
 * because it is only damaging at long context. Derived from the sqrt(L) rule.
 */
function qualityIndex(wb: number, kb: number) {
  const wErr = wb === 16 ? 0 : wb === 8 ? 0.004 : 0.055;
  const kvErr = kb === 16 ? 0 : kb === 8 ? 0.008 : 0.05;
  return Math.max(0, 1 - (wErr + 0.6 * kvErr) * Math.sqrt(32));
}

/* chart geometry */
const X0 = 62, X1 = 600, Y0 = 322, Y1 = 96;
const px = (q: number) => X0 + q * (X1 - X0);
const py = (t: number, maxT: number) => Y0 - (t / maxT) * (Y0 - Y1);

export default function ParetoFrontier() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [batch, setBatch] = useState(32);
  const [picked, setPicked] = useState(5);

  const points = useMemo(
    () =>
      CONFIGS.map((c) => {
        const meta = metaFracInt(c.wb, 128);
        // Throughput = users we can serve per second at a fixed per-user SLO budget.
        const d = decodeStepQuant({ batch, ctx: CTX, wBits: c.wb, kvBits: c.kb, metaFrac: meta });
        const weight = weightBytes(llama8b, c.wb, meta);
        const perUserKv = kvBytes(CTX, c.kb);
        // users that fit, from memory; throughput if we run them all at one step
        const usersFit = Math.max(1, Math.floor((h100.mem * 0.8 - weight) / perUserKv));
        const effBatch = Math.min(batch, usersFit);
        const throughput = effBatch / d.t;
        return { ...c, meta, quality: qualityIndex(c.wb, c.kb), throughput, weight, kv: effBatch * perUserKv, mem: weight + effBatch * perUserKv, usersFit };
      }),
    [batch],
  );

  /* Pareto: not dominated on both quality and throughput. */
  const frontier = useMemo(() => {
    return points.filter((p) =>
      !points.some((q) => q.quality >= p.quality && q.throughput >= p.throughput && (q.quality > p.quality || q.throughput > p.throughput)),
    );
  }, [points]);

  const onFrontier = new Set(frontier.map((p) => p.label));
  const maxT = Math.max(...points.map((p) => p.throughput)) * 1.12;
  const cur = points[picked];
  const curOnFrontier = onFrontier.has(cur.label);

  return (
    <Stage
      rootRef={ref}
      title="The quality / speed / memory frontier"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', ['compute', 'dominated'], 'memory', 'error', 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Config', value: cur.label, kind: curOnFrontier ? 'token' : 'error', note: curOnFrontier ? 'on the frontier' : 'dominated — something else is better on both axes' },
          { label: 'Quality index', value: `${(cur.quality * 100).toFixed(0)}%`, kind: cur.quality > 0.95 ? 'token' : cur.quality > 0.85 ? 'compute' : 'error', note: 'BF16 = 100%' },
          { label: 'Throughput', value: `${cur.throughput.toFixed(0)} tok/s`, kind: 'token', note: `${batch} users @ ${fmt.int(CTX)} ctx, fits ${cur.usersFit}` },
          { label: 'Weights + KV', value: fmt.gb(cur.mem), kind: 'memory', note: `vs ${fmt.gb(points[0].mem)} at BF16` },
        ]} />
      }
      controls={
        <>
          <Slider label="Load (concurrent users)" value={batch} min={1} max={256} log onChange={setBatch} format={(v) => `${fmt.int(v)}`} kind="token" />
          <Segmented label="Inspect a configuration" value={picked} options={CONFIGS.map((c, i) => ({ value: i, label: c.label }))} onChange={setPicked} />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A scatter plot of eight quantization configurations by quality index against throughput, with the Pareto frontier drawn between the non-dominated points. Points below the frontier are dominated: another configuration is at least as fast and at least as accurate.">
        {/* ── axes ── */}
        <Fade show={st >= 1}>
          <g>
            <line x1={X0} x2={X1} y1={Y0} y2={Y0} className="v-rule" />
            <line x1={X0} x2={X0} y1={Y0} y2={Y1 - 10} className="v-rule" />
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <Label key={f} x={X0 - 6} y={Y0 - f * (Y0 - Y1) + 3} anchor="end" tone={3} size={9}>{fmt.si(f * maxT)}</Label>
            ))}
            {[0.8, 0.85, 0.9, 0.95, 1].map((q) => (
              <Label key={q} x={px(q)} y={Y0 + 13} anchor="middle" tone={3} size={9}>{(q * 100).toFixed(0)}%</Label>
            ))}
            <Label x={X1} y={Y0 + 26} anchor="end" tone={3} size={9}>quality index (BF16 = 100%) →</Label>
            <Label x={X0 - 6} y={Y1 - 12} anchor="end" tone={3} size={9}>tok/s →</Label>
          </g>
        </Fade>

        {/* ── step 0: the three axes ── */}
        <Fade show={st === 0}>
          <g>
            <Label x={0} y={22} tone={2} size={11} bold>Do not collapse three axes into one score</Label>
            {[
              { y: 58, name: 'Quality', note: 'task accuracy, perplexity, the failure modes you read by hand', kind: 'token' as const },
              { y: 96, name: 'Speed', note: 'TTFT, TPOT, throughput at your SLO — throughput first, it is the honest one', kind: 'compute' as const },
              { y: 134, name: 'Memory', note: 'weights + KV peak. Sets concurrency, and concurrency sets throughput', kind: 'memory' as const },
            ].map((a) => (
              <g key={a.name}>
                <rect x={40} y={a.y - 14} width={92} height={20} rx={3} fill={`var(--${a.kind})`} opacity={0.9} />
                <Label x={86} y={a.y} anchor="middle" size={10.5} style={{ fill: '#0e1116' }} bold>{a.name}</Label>
                <Label x={146} y={a.y} size={10.5} tone={2}>{a.note}</Label>
              </g>
            ))}
            <Label x={40} y={182} size={10} tone={3}>Speed and memory are nearly collinear, because decode is memory-bound: fewer bytes is both less memory and a shorter step.</Label>
            <Label x={40} y={200} size={10} tone={3}>That is why a single "quantization score" is tempting, and why it hides the only decision: quality is not on that axis.</Label>
          </g>
        </Fade>

        {/* ── the points ── */}
        <Fade show={st >= 1}>
          <g>
            {/* the frontier, drawn as a staircase */}
            <path
              d={[...frontier].sort((a, b) => a.quality - b.quality).map((p, i) => `${i ? 'L' : 'M'}${px(p.quality).toFixed(1)},${py(p.throughput, maxT).toFixed(1)}`).join(' ')}
              fill="none"
              stroke="var(--token)"
              strokeWidth={2}
              opacity={st >= 2 ? 0.95 : 0.25}
            />
            {points.map((p, i) => {
              const f = onFrontier.has(p.label);
              const sel = i === picked;
              const cx = px(p.quality);
              const cy = py(p.throughput, maxT);
              return (
                <g key={p.label} onClick={() => setPicked(i)} style={{ cursor: 'pointer' }}>
                  <circle cx={cx} cy={cy} r={sel ? 8 : 6} fill={f ? 'var(--token)' : 'var(--text-3)'} stroke={sel ? 'var(--text)' : 'none'} strokeWidth={1.5} opacity={f ? 0.95 : 0.6} />
                  <Label
                    x={cx}
                    y={cy - (sel ? 16 : 11)}
                    anchor="middle"
                    size={sel ? 10 : 8.5}
                    tone={sel || f ? 1 : 3}
                    bold={sel}
                  >
                    {p.label}
                  </Label>
                </g>
              );
            })}
          </g>
        </Fade>

        {/* ── step 2 onward: interpretation ── */}
        <Fade show={st >= 2}>
          <Label x={X0 + 4} y={Y1 + 2} size={9.5} kind="token">the frontier: nothing else beats these on both axes</Label>
        </Fade>

        <Fade show={st >= 3}>
          <g>
            <Label x={0} y={22} tone={2} size={11} bold>The frontier has a shape, and the shape is your workload</Label>
            <Label x={0} y={42} tone={3} size={9.5}>{`At ${batch} users and ${fmt.int(CTX)} tokens of context, the frontier runs from BF16 (best quality) to W4+KV4 (best throughput).`}</Label>
            <Label x={0} y={58} tone={3} size={9.5}>{`${points.filter((p) => !onFrontier.has(p.label)).map((p) => p.label).join(', ') || 'Nothing'} ${points.filter((p) => !onFrontier.has(p.label)).length === 1 ? 'is' : 'are'} dominated at this load — delete it from consideration.`}</Label>
          </g>
        </Fade>

        <Fade show={st >= 4}>
          <g>
            {/* what dominates the picked point, if anything */}
            {(() => {
              const better = points.filter((q) => q.quality >= cur.quality && q.throughput >= cur.throughput && (q.quality > cur.quality || q.throughput > cur.throughput));
              return (
                <>
                  <Label x={0} y={344} tone={2} size={11} bold>{`You are holding ${cur.label}`}</Label>
                  {better.length > 0 ? (
                    <>
                      <Label x={0} y={362} size={9.5} kind="error">
                        {`Dominated by ${better.map((b) => b.label).join(', ')} — ${better[0].quality >= cur.quality ? 'no worse quality' : ''}${better[0].quality >= cur.quality && better[0].throughput >= cur.throughput ? ' and ' : ''}${better[0].throughput >= cur.throughput ? 'no slower' : ''}.`}
                      </Label>
                    </>
                  ) : (
                    <Label x={0} y={362} size={9.5} kind="token">
                      {`Nothing dominates it. Weight memory ${fmt.gb(cur.weight)} · KV ${fmt.gb(cur.kv)} · ${cur.usersFit} users fit on one H100.`}
                    </Label>
                  )}
                  <Label x={0} y={374} tone={3} size={9}>Write down what would move you off this point: a quality budget change, a context-length change, or a new hardware generation.</Label>
                </>
              );
            })()}
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}