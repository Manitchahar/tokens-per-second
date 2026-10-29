import { useRef } from 'react';
import { Stage, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';

const STEPS = [
  '1. Examine hardware',
  '2. Define workload',
  '3. Metrics & SLO',
  '4. Default baseline server',
  '5. Benchmark baseline',
  '6. Change ONE knob',
  '7. Record & decide',
];

const CAPTIONS = [
  'Examine the GPU model, memory capacity, memory bandwidth, and interconnect (nvidia-smi). This dictates your hardware-bound ceilings before any code runs.',
  'Fix the workload with a reproducible traffic generator: prompt and output token distributions, arrival rate, and concurrency. Tuning against moving traffic measures noise.',
  'Establish explicit SLO targets: TTFT (prefill & queueing), TPOT/ITL (decode streaming), throughput, and goodput (tokens delivered inside the SLO target).',
  'Deploy a baseline server with standard engine defaults. Record memory footprint: model weights, runtime reserve, and the resulting KV cache pool.',
  'Benchmark the default baseline configuration under target traffic. This establishes reference numbers for TTFT, TPOT, and goodput in your results table.',
  'Change exactly ONE parameter at a time (e.g. max-model-len or max-num-seqs) and re-benchmark under the exact same traffic. Never move two knobs simultaneously.',
  'Record results in the optimization ledger. Compare goodput and latency against baseline to make a clean engineering decision: keep wins, revert regressions.',
];

interface StationDef {
  num: string;
  title: string;
  detail: string;
  x: number;
  y: number;
  w: number;
  h: number;
  kind: 'compute' | 'memory' | 'token' | 'comm' | 'error';
}

const STATIONS: StationDef[] = [
  { num: '01', title: 'Examine Hardware', detail: 'H100 80GB · 3.35 TB/s', x: 18, y: 16, w: 132, h: 42, kind: 'memory' },
  { num: '02', title: 'Define Workload', detail: 'ShareGPT · 10 req/s', x: 178, y: 16, w: 132, h: 42, kind: 'token' },
  { num: '03', title: 'Metrics & SLO', detail: 'TTFT & TPOT targets', x: 338, y: 16, w: 132, h: 42, kind: 'compute' },
  { num: '04', title: 'Baseline Server', detail: 'Default vLLM setup', x: 498, y: 16, w: 124, h: 42, kind: 'memory' },
  { num: '05', title: 'Benchmark Run', detail: 'vllm bench serve', x: 498, y: 82, w: 124, h: 42, kind: 'compute' },
  { num: '06', title: 'Change ONE Knob', detail: 'Isolate 1 variable', x: 338, y: 82, w: 132, h: 42, kind: 'comm' },
  { num: '07', title: 'Record & Decide', detail: 'Keep win / revert', x: 178, y: 82, w: 132, h: 42, kind: 'token' },
];

export default function OptimizationPlan() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const curStep = Math.max(0, Math.min(s.step, STEPS.length - 1));

  // Determine stage readout data
  const readoutData = [
    { label: 'Active station', value: `${STATIONS[curStep].num}. ${STATIONS[curStep].title}`, kind: STATIONS[curStep].kind, note: STATIONS[curStep].detail },
    { label: 'Target model', value: 'Qwen3-14B (BF16)', kind: 'memory' as const, note: '29.6 GB weights on H100 80GB' },
    { label: 'Decode ceiling', value: '~113 tok/s (1-stream)', kind: 'compute' as const, note: '3.35 TB/s ÷ 29.6 GB upper bound' },
    {
      label: 'Decision status',
      value: curStep < 4 ? 'Preparing baseline' : curStep === 4 ? 'Baseline recorded' : curStep === 5 ? 'Evaluating Exp 01' : '1 Kept · 1 Reverted',
      kind: curStep === 6 ? ('token' as const) : ('compute' as const),
      note: curStep === 6 ? 'Exp 01 kept (+23%), Exp 02 reverted' : 'Discipline: one change at a time',
    },
  ];

  return (
    <Stage
      rootRef={ref}
      title="The serving optimization loop: 7 stations to peak goodput"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', ['comm', 'iteration'], ['error', 'revert / breach']]}
      {...s}
      readout={<Readout items={readoutData} />}
    >
      <svg
        viewBox="0 0 640 340"
        role="img"
        aria-label="Seven-station optimization loop showing hardware, workload, metrics, baseline, benchmarking, changing one knob, and recording keep or revert decisions"
        style={{ width: '100%', height: 'auto', display: 'block' }}
      >
        <defs>
          <marker id="arrow-right" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--rule-2)" />
          </marker>
          <marker id="arrow-right-active" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--token)" />
          </marker>
          <marker id="arrow-left" viewBox="0 0 10 10" refX="2" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 10 1 L 0 5 L 10 9 z" fill="var(--rule-2)" />
          </marker>
          <marker id="arrow-down" viewBox="0 0 10 10" refX="5" refY="8" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 1 0 L 5 10 L 9 0 z" fill="var(--rule-2)" />
          </marker>
          <linearGradient id="active-station-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--compute)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--compute)" stopOpacity="0.05" />
          </linearGradient>
        </defs>

        {/* ─── Top Connectors & Directional Arrows ─── */}
        {/* Row 1: 1 -> 2 */}
        <line x1={150} y1={37} x2={175} y2={37} stroke={curStep >= 1 ? 'var(--token)' : 'var(--rule-2)'} strokeWidth={1.5} markerEnd={curStep >= 1 ? 'url(#arrow-right-active)' : 'url(#arrow-right)'} />
        {/* Row 1: 2 -> 3 */}
        <line x1={310} y1={37} x2={335} y2={37} stroke={curStep >= 2 ? 'var(--token)' : 'var(--rule-2)'} strokeWidth={1.5} markerEnd={curStep >= 2 ? 'url(#arrow-right-active)' : 'url(#arrow-right)'} />
        {/* Row 1: 3 -> 4 */}
        <line x1={470} y1={37} x2={495} y2={37} stroke={curStep >= 3 ? 'var(--token)' : 'var(--rule-2)'} strokeWidth={1.5} markerEnd={curStep >= 3 ? 'url(#arrow-right-active)' : 'url(#arrow-right)'} />

        {/* Vertical Turn: 4 -> 5 */}
        <path
          d="M 560 58 L 560 79"
          fill="none"
          stroke={curStep >= 4 ? 'var(--token)' : 'var(--rule-2)'}
          strokeWidth={1.5}
          markerEnd="url(#arrow-down)"
        />

        {/* Row 2: 5 <- 6 */}
        <line x1={498} y1={103} x2={473} y2={103} stroke={curStep >= 5 ? 'var(--token)' : 'var(--rule-2)'} strokeWidth={1.5} markerEnd="url(#arrow-left)" />
        {/* Row 2: 6 <- 7 */}
        <line x1={338} y1={103} x2={313} y2={103} stroke={curStep >= 6 ? 'var(--token)' : 'var(--rule-2)'} strokeWidth={1.5} markerEnd="url(#arrow-left)" />

        {/* Iteration Loop Arrow: from Station 7 back around to Station 6 */}
        <path
          d="M 244 124 C 244 144, 404 144, 404 127"
          fill="none"
          stroke={curStep >= 6 ? 'var(--comm)' : 'var(--rule)'}
          strokeWidth={1.5}
          strokeDasharray={curStep >= 6 ? undefined : '3 3'}
          markerEnd={curStep >= 6 ? 'url(#arrow-down)' : undefined}
        />
        <text
          x={324}
          y={140}
          textAnchor="middle"
          fill={curStep >= 6 ? 'var(--comm)' : 'var(--text-3)'}
          fontSize={10}
          fontWeight={600}
          className="v-label sans"
        >
          ↻ Loop: iterate one knob at a time
        </text>

        {/* ─── The Seven Station Cards ─── */}
        {STATIONS.map((st, idx) => {
          const isActive = curStep === idx;
          const isDone = curStep > idx;
          const borderColor = isActive
            ? 'var(--compute)'
            : isDone
              ? 'var(--rule-2)'
              : 'var(--rule)';
          const bgColor = isActive
            ? 'rgba(232, 168, 56, 0.12)'
            : isDone
              ? 'rgba(255, 255, 255, 0.03)'
              : 'rgba(255, 255, 255, 0.01)';
          const badgeBg = isActive
            ? 'var(--compute)'
            : isDone
              ? 'var(--rule-2)'
              : 'rgba(255, 255, 255, 0.08)';
          const badgeText = isActive ? '#0e1116' : isDone ? 'var(--text-2)' : 'var(--text-3)';

          return (
            <g key={st.num} className="station-node">
              <rect
                x={st.x}
                y={st.y}
                width={st.w}
                height={st.h}
                rx={5}
                fill={bgColor}
                stroke={borderColor}
                strokeWidth={isActive ? 1.8 : 1}
              />
              {/* Station Badge */}
              <rect
                x={st.x + 6}
                y={st.y + 7}
                width={20}
                height={16}
                rx={3}
                fill={badgeBg}
              />
              <text
                x={st.x + 16}
                y={st.y + 19}
                textAnchor="middle"
                fontSize={9}
                fontWeight={700}
                fill={badgeText}
                className="v-label"
              >
                {st.num}
              </text>

              {/* Station Title */}
              <text
                x={st.x + 32}
                y={st.y + 20}
                fontSize={10.5}
                fontWeight={isActive ? 700 : 600}
                fill={isActive ? 'var(--text)' : isDone ? 'var(--text-2)' : 'var(--text-3)'}
                className="v-label sans"
              >
                {st.title}
              </text>

              {/* Station Subtext */}
              <text
                x={st.x + 8}
                y={st.y + 35}
                fontSize={9}
                fill={isActive ? 'var(--compute)' : 'var(--text-3)'}
                className="v-label"
              >
                {st.detail}
              </text>
            </g>
          );
        })}

        {/* ─── Bottom Area: Experimental Results Ledger ─── */}
        <g transform="translate(18, 154)">
          {/* Container background */}
          <rect
            x={0}
            y={0}
            width={604}
            height={170}
            rx={6}
            fill="rgba(14, 18, 24, 0.85)"
            stroke="var(--rule-2)"
            strokeWidth={1}
          />

          {/* Ledger Title Bar */}
          <rect x={0} y={0} width={604} height={26} rx={6} fill="rgba(255, 255, 255, 0.03)" />
          <line x1={0} y1={26} x2={604} y2={26} stroke="var(--rule-2)" strokeWidth={1} />
          <text x={12} y={17} fontSize={10} fontWeight={700} fill="var(--text)" className="v-label">
            RESULTS LEDGER: Qwen3-14B on H100 SXM 80GB (ShareGPT workload)
          </text>
          <text x={592} y={17} textAnchor="end" fontSize={9.5} fill="var(--text-3)" className="v-label">
            SLO: TTFT &lt; 200 ms · TPOT &lt; 40 ms
          </text>

          {/* Table Column Headers */}
          <g transform="translate(0, 26)">
            <rect x={0} y={0} width={604} height={22} fill="rgba(255, 255, 255, 0.02)" />
            <line x1={0} y1={22} x2={604} y2={22} stroke="var(--rule)" strokeWidth={1} />

            <text x={12} y={15} fontSize={9.5} fontWeight={600} fill="var(--text-3)" className="v-label">Exp</text>
            <text x={48} y={15} fontSize={9.5} fontWeight={600} fill="var(--text-3)" className="v-label">Change (ONE variable)</text>
            <text x={232} y={15} fontSize={9.5} fontWeight={600} fill="var(--text-3)" className="v-label">Throughput</text>
            <text x={314} y={15} fontSize={9.5} fontWeight={600} fill="var(--text-3)" className="v-label">TTFT (p50/p99)</text>
            <text x={410} y={15} fontSize={9.5} fontWeight={600} fill="var(--text-3)" className="v-label">TPOT (p50/p99)</text>
            <text x={498} y={15} fontSize={9.5} fontWeight={600} fill="var(--text-3)" className="v-label">Goodput</text>
            <text x={552} y={15} fontSize={9.5} fontWeight={600} fill="var(--text-3)" className="v-label">Decision</text>
          </g>

          {/* Row 0: Baseline (appears at step >= 4) */}
          <g transform="translate(0, 48)">
            {curStep < 4 ? (
              <g>
                <rect x={6} y={6} width={592} height={28} rx={4} fill="none" stroke="var(--rule)" strokeDasharray="3 3" />
                <text x={302} y={24} textAnchor="middle" fontSize={10} fill="var(--text-3)" className="v-label">
                  Step 1-4: Configuring environment... awaiting baseline benchmark run.
                </text>
              </g>
            ) : (
              <g>
                <rect x={4} y={3} width={596} height={34} rx={4} fill={curStep === 4 ? 'rgba(232, 168, 56, 0.08)' : 'transparent'} stroke={curStep === 4 ? 'var(--compute)' : 'transparent'} strokeWidth={1} />
                <line x1={8} y1={38} x2={596} y2={38} stroke="var(--rule)" strokeWidth={0.8} />

                {/* Exp Number */}
                <text x={12} y={24} fontSize={10.5} fontWeight={700} fill="var(--text-2)" className="v-label">00</text>
                {/* Change */}
                <text x={48} y={20} fontSize={10} fontWeight={600} fill="var(--text)" className="v-label sans">Baseline defaults</text>
                <text x={48} y={32} fontSize={8.5} fill="var(--text-3)" className="v-label">util=0.90, max-len=8192, seqs=256</text>
                {/* Throughput */}
                <text x={232} y={24} fontSize={10} fill="var(--text)" className="v-label">474 tok/s</text>
                {/* TTFT */}
                <text x={314} y={24} fontSize={10} fill="var(--token)" className="v-label">104 / 185 ms</text>
                {/* TPOT */}
                <text x={410} y={24} fontSize={10} fill="var(--token)" className="v-label">32 / 38 ms</text>
                {/* Goodput */}
                <text x={498} y={24} fontSize={10} fontWeight={700} fill="var(--token)" className="v-label">445 tok/s</text>
                {/* Decision badge */}
                <rect x={546} y={12} width={50} height={18} rx={3} fill="rgba(255, 255, 255, 0.08)" stroke="var(--rule-2)" />
                <text x={571} y={24} textAnchor="middle" fontSize={8.5} fontWeight={600} fill="var(--text-2)" className="v-label">BASELINE</text>
              </g>
            )}
          </g>

          {/* Row 1: Experiment 01 (appears at step >= 5) */}
          <g transform="translate(0, 88)">
            {curStep < 5 ? (
              <g>
                <rect x={6} y={4} width={592} height={28} rx={4} fill="none" stroke="var(--rule)" strokeDasharray="3 3" opacity={0.5} />
                <text x={302} y={21} textAnchor="middle" fontSize={9.5} fill="var(--text-3)" className="v-label">
                  Step 6: Experiment 01 will isolate and test first parameter change.
                </text>
              </g>
            ) : (
              <g>
                <rect x={4} y={2} width={596} height={34} rx={4} fill={curStep === 5 ? 'rgba(84, 180, 230, 0.08)' : 'transparent'} stroke={curStep === 5 ? 'var(--memory)' : 'transparent'} strokeWidth={1} />
                <line x1={8} y1={37} x2={596} y2={37} stroke="var(--rule)" strokeWidth={0.8} />

                {/* Exp Number */}
                <text x={12} y={23} fontSize={10.5} fontWeight={700} fill="var(--token)" className="v-label">01</text>
                {/* Change */}
                <text x={48} y={19} fontSize={10} fontWeight={600} fill="var(--text)" className="v-label sans">--max-model-len 4096</text>
                <text x={48} y={31} fontSize={8.5} fill="var(--memory)" className="v-label">halved worst-case KV footprint</text>
                {/* Throughput */}
                <text x={232} y={23} fontSize={10} fill="var(--text)" className="v-label">582 tok/s</text>
                {/* TTFT */}
                <text x={314} y={23} fontSize={10} fill="var(--token)" className="v-label">98 / 172 ms</text>
                {/* TPOT */}
                <text x={410} y={23} fontSize={10} fill="var(--token)" className="v-label">31 / 37 ms</text>
                {/* Goodput */}
                <text x={498} y={23} fontSize={10} fontWeight={700} fill="var(--token)" className="v-label">548 tok/s</text>
                {/* Decision badge */}
                {curStep === 5 ? (
                  <g>
                    <rect x={546} y={11} width={50} height={18} rx={3} fill="rgba(84, 180, 230, 0.2)" stroke="var(--memory)" />
                    <text x={571} y={23} textAnchor="middle" fontSize={8.5} fontWeight={700} fill="var(--memory)" className="v-label">TESTING</text>
                  </g>
                ) : (
                  <g>
                    <rect x={546} y={11} width={50} height={18} rx={3} fill="rgba(46, 189, 133, 0.2)" stroke="var(--token)" />
                    <text x={571} y={23} textAnchor="middle" fontSize={8.5} fontWeight={700} fill="var(--token)" className="v-label">✓ KEEP</text>
                  </g>
                )}
              </g>
            )}
          </g>

          {/* Row 2: Experiment 02 (appears at step >= 6) */}
          <g transform="translate(0, 126)">
            {curStep < 6 ? (
              <g>
                <rect x={6} y={3} width={592} height={26} rx={4} fill="none" stroke="var(--rule)" strokeDasharray="3 3" opacity={0.3} />
                <text x={302} y={19} textAnchor="middle" fontSize={9.5} fill="var(--text-3)" className="v-label">
                  Step 7: Experiment 02 evaluates concurrency stress &amp; keep/revert rule.
                </text>
              </g>
            ) : (
              <g>
                <rect x={4} y={1} width={596} height={34} rx={4} fill="rgba(235, 87, 87, 0.06)" stroke="var(--error)" strokeWidth={1} />

                {/* Exp Number */}
                <text x={12} y={22} fontSize={10.5} fontWeight={700} fill="var(--error)" className="v-label">02</text>
                {/* Change */}
                <text x={48} y={18} fontSize={10} fontWeight={600} fill="var(--text)" className="v-label sans">--max-num-seqs 512</text>
                <text x={48} y={30} fontSize={8.5} fill="var(--error)" className="v-label">doubled batch cap; decode saturated</text>
                {/* Throughput */}
                <text x={232} y={22} fontSize={10} fill="var(--text-3)" className="v-label">695 tok/s</text>
                {/* TTFT */}
                <text x={314} y={22} fontSize={10} fill="var(--error)" className="v-label">165 / 310 ms</text>
                {/* TPOT */}
                <text x={410} y={22} fontSize={10} fontWeight={700} fill="var(--error)" className="v-label">44 / 58 ms</text>
                {/* Goodput */}
                <text x={498} y={22} fontSize={10} fontWeight={700} fill="var(--error)" className="v-label">210 tok/s</text>
                {/* Decision badge */}
                <rect x={546} y={10} width={50} height={18} rx={3} fill="rgba(235, 87, 87, 0.2)" stroke="var(--error)" />
                <text x={571} y={22} textAnchor="middle" fontSize={8.5} fontWeight={700} fill="var(--error)" className="v-label">✗ REVERT</text>
              </g>
            )}
          </g>
        </g>
      </svg>
    </Stage>
  );
}
