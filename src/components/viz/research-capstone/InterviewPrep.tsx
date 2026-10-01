import { useEffect, useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Chip, Move } from '../lib/svg';

const STEPS = [
  'The map: what gets asked',
  'The whiteboard questions',
  'Designing a system, live',
  'The debugging story',
  'Numbers under pressure',
  'Where to go next',
];
const CAPTIONS = [
  'Interview questions cluster into a small number of kinds. Know the kinds and you know what to prepare.',
  'The whiteboard questions are arithmetic with a story: roofline, memory math, cost per token, Little’s law.',
  'The system design question is where the course pays off: start from the SLO, derive the budget, then choose.',
  'Debugging is a credibility question. They want to see how you isolate a bottleneck under time pressure.',
  'Know your numbers cold: bandwidths, KV-per-token, cost per million tokens, and the rule-of-thumb budget.',
  'Where to go after the course: deeper into kernels, engines, scaling, or research. Pick the direction that matches you.',
];

type Track = 'systems' | 'kernel' | 'product' | 'research';

const W = 640;

/** The question kinds and the modules that answer them. */
const MAP = [
  { k: 'Roofline & bottlenecks', sub: 'memory-bound vs compute-bound', ans: 'Modules 02, 12', kind: 'compute' as const },
  { k: 'Memory & KV math', sub: 'weights, cache, max batch', ans: 'Modules 05, 08, 20', kind: 'memory' as const },
  { k: 'Batching & scheduling', sub: 'TTFT vs throughput', ans: 'Modules 06, 07', kind: 'token' as const },
  { k: 'Parallelism & scaling', sub: 'TP/PP/EP, disaggregation', ans: 'Modules 18–21', kind: 'comm' as const },
  { k: 'Quantization & kernels', sub: 'memory for quality', ans: 'Modules 11, 13, 14', kind: 'compute' as const },
  { k: 'Cost & capacity', sub: '$ per M tokens, SLOs', ans: 'Modules 25, 23', kind: 'text' as const },
];

/** Whiteboard numbers you must be able to derive cold. */
const NUMBERS = [
  { q: 'KV bytes per token, Llama-3.1-8B', a: '2×32×8×128×2 = 128 KB' },
  { q: 'Decode step time, 8B on H100 @ B=1', a: '16 GB ÷ 3.35 TB/s ≈ 5 ms' },
  { q: 'Cost per M tokens, $2.50/hr @ 8k tok/s', a: '≈ $0.09' },
  { q: 'Little’s law, 4000 users, 8 s each', a: '4000/8 = 500 in flight' },
];

const TRACKS: { id: Track; label: string; body: string }[] = [
  { id: 'systems', label: 'Systems / serving', body: 'SLOs, capacity, cost, and reliability. You are the person who decides what runs and what it costs. Build: multi-replica service, autoscaling, a real cost model (Modules 22–25).' },
  { id: 'kernel', label: 'Kernels / runtime', body: 'Where the GPU actually spends its time. Bandwidth-bound fusion, attention kernels, compiler tricks. Build: write a kernel, beat PyTorch on it (Modules 13–16).' },
  { id: 'product', label: 'Product / applied', body: 'Tuning for a workload: structured output, LoRA, agents, RL rollouts. Build: specialize a serving stack for one product and show the goodput win (Module 26).' },
  { id: 'research', label: 'Research', body: 'Reproduce, extend, publish. This module is the entry. Build: reproduce two papers and design the experiment that extends one (Modules 27–28).' },
];

export default function InterviewPrep() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [track, setTrack] = useState<Track>('systems');
  useEffect(() => {
    const t: Track[] = ['systems', 'systems', 'systems', 'kernel', 'product', 'research'];
    setTrack(t[Math.min(s.step, t.length - 1)]);
  }, [s.step]);

  const cur = TRACKS.find((x) => x.id === track)!;

  return (
    <Stage
      rootRef={ref}
      title="Inference interviews: the map, the whiteboard numbers, and what comes next"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', 'comm', ['text', 'roles']]}
      {...s}
      readout={<Readout items={[{ label: 'Current focus', value: cur.label, kind: 'text', note: 'pick your direction' }, { label: 'Question kinds', value: `${MAP.length}`, kind: 'token', note: 'know the kinds' }]} />}
      controls={<Segmented label="Direction after the course" value={track} onChange={setTrack} options={TRACKS.map((t) => ({ value: t.id, label: t.label.split(' / ')[0] }))} />}
    >
      <svg viewBox={`0 0 ${W} 372`} role="img" aria-label="A map of inference interview question kinds each mapped to the modules that answer them, beside a list of whiteboard numbers to derive cold and a panel describing career directions.">
        {/* the map */}
        <Label x={20} y={16} tone={2} size={11} mono={false} bold>What gets asked, and where you learned it</Label>
        {MAP.map((m, i) => (
          <g key={m.k} className="v-fade" style={{ opacity: s.step <= 1 ? 1 : 0.4 }}>
            <rect x={20} y={26 + i * 30} width={4} height={22} rx={2} fill={`var(--${m.kind === 'text' ? 'text' : m.kind})`} />
            <text x={32} y={40 + i * 30} className="v-label bold" style={{ fontSize: 10.5, fill: 'var(--text)' }}>{m.k}</text>
            <text x={32} y={52 + i * 30} className="v-label t3" style={{ fontSize: 9.5 }}>{m.sub}</text>
            <Move x={248} y={26 + i * 30} opacity={s.step <= 1 ? 1 : 0.4}>
              <Chip x={0} y={6} text={m.ans} kind="token" size={9.5} />
            </Move>
          </g>
        ))}

        {/* whiteboard numbers */}
        <Label x={320} y={16} tone={2} size={11} mono={false} bold>Numbers, cold</Label>
        {NUMBERS.map((n, i) => (
          <g key={n.q} className="v-fade" style={{ opacity: s.step >= 4 || s.step === 2 ? 1 : 0.35 }}>
            <text x={320} y={36 + i * 34} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text-2)' }}>{n.q}</text>
            <text x={320} y={49 + i * 34} className="v-label bold" style={{ fontSize: 10.5, fill: 'var(--token)' }}>{n.a}</text>
          </g>
        ))}

        {/* track panel */}
        <Fade show={s.step >= 5}>
          <rect x={320} y={190} width={300} height={128} rx={4} fill="var(--ground-2)" stroke="var(--token)" />
          <text x={332} y={212} className="v-label bold" style={{ fontSize: 12, fill: 'var(--token)' }}>{cur.label}</text>
          {wrap(cur.body, 46).map((ln, i) => (
            <text key={i} x={332} y={232 + i * 14} className="v-label sans" style={{ fontSize: 10, fill: 'var(--text)' }}>{ln}</text>
          ))}
        </Fade>

        {/* the one-liner reminder */}
        <text x={W / 2} y={368} textAnchor="middle" className="v-label t3" style={{ fontSize: 9.5 }}>
          They are not testing whether you memorised a repo. They are testing whether you can derive an answer and defend it.
        </text>
      </svg>
    </Stage>
  );
}

/** Greedy word wrap for short SVG text lines. */
function wrap(text: string, width: number): string[] {
  const words = text.split(' ');
  const lines: string[] = [];
  let cur = '';
  for (const w of words) {
    if ((cur + ' ' + w).trim().length > width) { lines.push(cur.trim()); cur = w; }
    else cur += ' ' + w;
  }
  if (cur.trim()) lines.push(cur.trim());
  return lines;
}