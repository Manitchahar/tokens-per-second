import { useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Arrow, Dim, Fade, Bar, Chip, fmt } from '../lib/svg';
import { SCHEMES, stackError } from './quant';

const STEPS = [
  'Perplexity: the cheap first check',
  'Perplexity is not enough',
  'The task benchmarks that matter',
  'Reading the numbers honestly',
  'Long context and the KV effect',
  'The report you hand over',
];
const CAPTIONS = [
  'Perplexity is one number you can compute in minutes: how surprised is the model by real text it was not trained on. It moves before task accuracy does, which makes it an early warning.',
  'But two models with the same perplexity can behave very differently. Perplexity is averaged over every token in a corpus, and average hides exactly the rare failures you care about.',
  'Task benchmarks are the real gate: GSM8K for arithmetic, HumanEval for code, MMLU for knowledge. These are what a user would call "the model got worse", and they move later and less than perplexity.',
  'A regression report has to say more than "down 2 points". Report confidence, report the per-category breakdown, and report the failures you actually read.',
  'KV cache quantization has its own signature: quality holds at short context and degrades at long context. If you evaluate at 512 tokens you will conclude INT4 KV is fine, and be wrong.',
  'Ship a report, not a number. Configuration, eval set and version, quality delta with confidence, performance delta, and the one task that moved most.',
];

/* Task-level deltas, in percentage points, relative to BF16. Illustrative but internally consistent. */
const TASKS = [
  { id: 'gsm8k', label: 'GSM8K (math)', base: 62.5, w8: 61.8, w4: 59.1, kv8: 62.0, kv4: 58.2 },
  { id: 'humaneval', label: 'HumanEval (code)', base: 48.2, w8: 47.9, w4: 44.3, kv8: 47.6, kv4: 41.0 },
  { id: 'mmlu', label: 'MMLU (knowledge)', base: 68.4, w8: 68.0, w4: 66.8, kv8: 67.9, kv4: 65.9 },
  { id: 'longqa', label: 'Long-context QA (32k)', base: 71.0, w8: 70.4, w4: 66.2, kv8: 69.1, kv4: 58.7 },
];

const CONFIGS = [
  { id: 'w8', label: 'W8A8' },
  { id: 'w4', label: 'W4 (g128)' },
  { id: 'kv8', label: 'W4 + KV8' },
  { id: 'kv4', label: 'W4 + KV4' },
] as const;
type Cfg = (typeof CONFIGS)[number]['id'];

/** Wilson-style 95% interval on a proportion, for n samples. Enough for a report. */
function wilson(p: number, n: number) {
  const z = 1.96;
  const d = 1 + (z * z) / n;
  const c = p + (z * z) / (2 * n);
  const m = z * Math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n);
  return [(c - m) / d, (c + m) / d];
}

export default function QualityDashboard() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [cfg, setCfg] = useState<Cfg>('kv4');
  const [n, setN] = useState(2000);

  const activeCfg: Cfg = st >= 2 ? cfg : 'w4';

  const rows = useMemo(
    () =>
      TASKS.map((t) => {
        const val = t[activeCfg];
        const delta = val - t.base;
        const p = val / 100;
        const [lo, hi] = wilson(p, n);
        return { ...t, val, delta, lo: lo * 100, hi: hi * 100, significant: Math.abs(delta) > (hi - lo) * 100 / 2 + 0.2 };
      }),
    [activeCfg, n],
  );

  /* The sqrt(L) prediction for the stack-error strip. */
  const eps = activeCfg === 'w8' ? 0.003 : 0.06;
  const stacked = stackError(eps, 32);

  const BIG = 2.5; // percentage points: below this, call it noise
  const worst = rows.reduce((a, b) => (Math.abs(b.delta) > Math.abs(a.delta) ? b : a));

  /* chart geometry */
  const CX0 = 150, CX1 = 600;
  const vx = (v: number) => CX0 + ((v + 6) / 18) * (CX1 - CX0);

  return (
    <Stage
      rootRef={ref}
      title="Measuring quality regression: what to run, and what the numbers mean"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['error', 'token', 'memory', 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Config under test', value: CONFIGS.find((c) => c.id === activeCfg)!.label, kind: 'compute', note: 'against BF16 baseline' },
          { label: 'Biggest regression', value: `${worst.delta.toFixed(1)} pts`, kind: 'error', note: `${worst.label}` },
          { label: 'Stack-error estimate', value: `${(stacked * 100).toFixed(0)}%`, kind: 'error', note: `ε×√32 from ${(eps * 100).toFixed(0)}% per-weight error` },
          { label: 'Eval size', value: `${fmt.int(n)}`, kind: 'text', note: '±' + ((rows[0].hi - rows[0].lo) / 2).toFixed(1) + ' pts of noise' },
        ]} />
      }
      controls={
        st >= 2 && (
          <>
            <Segmented label="Configuration under test" value={cfg} options={CONFIGS.map((c) => ({ value: c.id, label: c.label }))} onChange={setCfg} />
            <Slider label="Eval set size (samples per task)" value={n} min={200} max={20000} log onChange={setN} format={(v) => `${fmt.int(v)}`} kind="text" />
          </>
        )
      }
    >
      <svg viewBox="0 0 640 384" role="img" aria-label="A quality regression dashboard: a dot plot of task benchmark scores for four configurations against a BF16 baseline, with confidence intervals, plus a perplexity panel showing an early-warning signal and a long-context panel showing where KV quantization degrades.">
        {/* ── perplexity ── */}
        <Fade show={st === 0 || st === 1}>
          <g>
            <Label x={0} y={22} tone={2} size={11} bold>Step 1: perplexity, on held-out text</Label>
            <Label x={640} y={22} anchor="end" tone={3} size={9.5}>lower is better · minutes to compute · moves first</Label>
            {[['BF16', 6.72, 'var(--text)'], ['W8A8', 6.74, 'var(--compute)'], ['W4 g128', 6.89, 'var(--error)'], ['W4 + KV4', 6.93, 'var(--error)']].map(([name, v, color], i) => {
              const y = 52 + i * 30;
              const w = ((v as number) - 6.6) / 0.5 * 320;
              return (
                <g key={name as string}>
                  <Label x={100} y={y + 10} anchor="end" tone={i === 0 ? 1 : 2} size={10}>{name as string}</Label>
                  <rect x={110} y={y} width={340} height={14} rx={2} fill="none" stroke="var(--rule)" />
                  <Span x={110} y={y} w={w} h={14} kind={i === 0 ? 'text' : (v as number) > 6.85 ? 'error' : 'compute'} opacity={0.9} label={`${v}`} />
                </g>
              );
            })}
            <Label x={110} y={188} tone={3} size={9.5}>PPL moved +0.17 from BF16 to W4 — a 2.5% jump. Task accuracy has not moved that much.</Label>
            <Label x={110} y={206} tone={3} size={9.5}>Useful as a smoke test. Not a decision: it averages over every token in the corpus.</Label>
          </g>
        </Fade>

        {/* ── task benchmarks ── */}
        <Fade show={st >= 2}>
          <g>
            <Label x={0} y={22} tone={2} size={11} bold>Step 2: the tasks users actually complain about</Label>
            <Label x={640} y={22} anchor="end" tone={3} size={9.5}>{`${CONFIGS.find((c) => c.id === activeCfg)!.label} vs BF16 · bar = score · tick = baseline · band = 95% interval`}</Label>
            {rows.map((r, i) => {
              const y = 48 + i * 52;
              return (
                <g key={r.id}>
                  <Label x={142} y={y + 8} anchor="end" tone={1} size={10}>{r.label}</Label>
                  {/* baseline tick */}
                  <line x1={vx(r.base)} x2={vx(r.base)} y1={y} y2={y + 22} stroke="var(--text-2)" strokeWidth={2} />
                  {/* score bar */}
                  <rect x={CX0} y={y} width={CX1 - CX0} height={22} rx={2} fill="none" stroke="var(--rule)" />
                  <Span x={CX0} y={y} w={Math.max(2, vx(r.val) - CX0)} h={22} kind={Math.abs(r.delta) > BIG ? 'error' : 'token'} opacity={0.85} />
                  {/* delta */}
                  <Arrow
                    x1={vx(Math.max(r.base, r.val)) + 6}
                    x2={vx(Math.min(r.base, r.val)) - 6}
                    y1={y + 11}
                    y2={y + 11}
                    kind={Math.abs(r.delta) > BIG ? 'error' : 'token'}
                    width={1.8}
                  />
                  <Label x={CX1 + 4} y={y + 15} size={9.5} kind={Math.abs(r.delta) > BIG ? 'error' : 'text'}>
                    {`${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(1)}`}
                  </Label>
                  <Label x={CX0 + 4} y={y + 33} tone={3} size={8.5}>{`baseline ${r.base.toFixed(1)} · this ${r.val.toFixed(1)} · ±${((r.hi - r.lo) / 2).toFixed(1)}`}</Label>
                </g>
              );
            })}
            {/* the axis */}
            <line x1={CX0 - 6} x2={CX0 - 6} y1={44} y2={44 + rows.length * 52 - 24} className="v-rule faint" />
            <Label x={CX0 - 10} y={44 + rows.length * 52 - 24 + 12} tone={3} size={8.5}>score axis starts at 54</Label>
          </g>
        </Fade>

        {/* ── long context ── */}
        <Fade show={st >= 4}>
          <g>
            <Label x={0} y={286} tone={2} size={11} bold>Step 3: KV quantization has a length signature</Label>
            <Label x={0} y={306} tone={3} size={9.5}>Accuracy on long-context QA versus context length, same weights, KV precision varied:</Label>
            {[
              { ctx: 1024, bf16: 72.4, kv8: 71.9, kv4: 70.8 },
              { ctx: 8192, bf16: 71.6, kv8: 70.9, kv4: 65.1 },
              { ctx: 32768, bf16: 71.0, kv8: 69.1, kv4: 58.7 },
            ].map((row, i) => {
              const y = 318 + i * 22;
              return (
                <g key={row.ctx}>
                  <Label x={110} y={y + 9} anchor="end" tone={2} size={9.5}>{`${fmt.int(row.ctx)}`}</Label>
                  <rect x={116} y={y} width={300} height={14} rx={2} fill="none" stroke="var(--rule)" />
                  <Span x={116} y={y} w={300} h={14} kind="text" opacity={0.6} label={`BF16 ${row.bf16}`} />
                  <Dim x1={116 + (row.bf16 - 56) / 16 * 300} x2={116 + (row.kv8 - 56) / 16 * 300} y={y - 6} kind="compute" label="KV8" />
                  <Dim x1={116 + (row.bf16 - 56) / 16 * 300} x2={116 + (row.kv4 - 56) / 16 * 300} y={y - 18} kind="error" label="KV4" />
                </g>
              );
            })}
            <Label x={430} y={330} size={9.5} kind="error">KV4 looks fine at 1k</Label>
            <Label x={430} y={346} size={9.5} kind="error">and is catastrophic at 32k.</Label>
            <Label x={430} y={362} size={9.5} kind="token">Evaluate KV at the longest</Label>
            <Label x={430} y={378} size={9.5} kind="token">context you actually serve.</Label>
          </g>
        </Fade>

        {/* ── the report ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={0} y={286} tone={2} size={11} bold>Step 4: the deliverable</Label>
            {[
              'Config: model revision, quantizer, group size, bits (weights and KV)',
              'Eval: task set + version, sample count, seed, prompt format',
              `Quality: per-task delta with 95% intervals — flag anything beyond ±${BIG} pts`,
              'Performance: TTFT, TPOT, throughput, peak memory, all vs BF16',
              'The one task that moved most, and three failures you read by hand',
            ].map((line, i) => (
              <Label key={i} x={14 + i * 8} y={308 + i * 19} tone={i === 2 ? 'error' : 2} size={9.5}>{line}</Label>
            ))}
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}