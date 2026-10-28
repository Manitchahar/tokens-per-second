import { useRef } from 'react';
import { Stage, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, fmt } from '../lib/svg';

const STEPS = [
  'Two axes, every chip',
  'The H100 roof',
  'A very different roof',
  'Same workload, different floor',
  'Now the capacity',
  'The bill for that latency',
  'What changes, what stays',
];
const CAPTIONS = [
  'Every accelerator can be drawn on the same two axes: arithmetic intensity along the bottom, achieved TFLOP/s up the side. A workload is a point; a chip is a roof.',
  'The H100 roof: a bandwidth slope of 3.35 TB/s, a compute ceiling of 989 TFLOP/s, and a ridge point of about 295 FLOP per byte where they meet.',
  'An SRAM-heavy chip has a very different roof: about 80 TB/s of on-chip bandwidth and 188 TFLOP/s of arithmetic. Its ridge point is near 2.4 FLOP per byte, so the roof is almost all slope.',
  'At decode’s 1 FLOP per byte the H100 roof allows 3.35 TFLOP/s; the SRAM chip allows 80. At prefill’s hundreds of FLOPs per byte the H100 wins. The roofs cross.',
  'Capacity is the other half of the story. Llama-3.1-70B in FP16 is 141 GB of weights. One 192 GB MI300X holds it; one 230 MB SRAM chip holds a six-hundredth of it.',
  'So the SRAM design needs about 614 chips for the same model. The violet links are the new cost: every token’s activations have to cross them, and every chip is full of weights rather than KV cache.',
  'Switching stacks changes kernels, quantisation formats, collectives and profilers. The roofline, the KV cache, batching and the SLO do not change. The physics is portable; the software is not.',
];

const X0 = 72;
const X1 = 592;
const Y0 = 236;
const Y1 = 52;
const XMIN = -1;
const XMAX = 3.5;
const YMIN = -1;
const YMAX = 3.5;

const px = (I: number) => X0 + ((Math.log10(Math.max(I, 0.1)) - XMIN) / (XMAX - XMIN)) * (X1 - X0);
const py = (F: number) => Y0 - ((Math.log10(Math.max(F, 0.1)) - YMIN) / (YMAX - YMIN)) * (Y0 - Y1);

const H100 = { bw: 3.35, peak: 989 };
const SRAM = { bw: 80, peak: 188 };

const roof = (bw: number, peak: number) => {
  const ridge = peak / bw;
  const pts: string[] = [];
  for (let i = 0; i <= 24; i++) {
    const I = 0.1 * Math.pow(10, (Math.log10(ridge / 0.1) * i) / 24);
    pts.push(`${i ? 'L' : 'M'}${px(I).toFixed(1)},${py(bw * I).toFixed(1)}`);
  }
  pts.push(`L${px(3162).toFixed(1)},${py(peak).toFixed(1)}`);
  return pts.join(' ');
};

const h100Ridge = H100.peak / H100.bw;
const sramRidge = SRAM.peak / SRAM.bw;
const h100Roof = roof(H100.bw, H100.peak);
const sramRoof = roof(SRAM.bw, SRAM.peak);

const DECODE_I = 1;
const PREFILL_I = 410;

const HBM_USED = 141.2;
const HBM_TOTAL = 192;
const SRAM_CHIP_GB = 0.23;

const CHIPS = [
  { x: 64, y: 230 },
  { x: 136, y: 230 },
  { x: 208, y: 230 },
  { x: 280, y: 230 },
  { x: 352, y: 230 },
  { x: 424, y: 230 },
  { x: 64, y: 266 },
  { x: 136, y: 266 },
  { x: 208, y: 266 },
  { x: 280, y: 266 },
  { x: 352, y: 266 },
  { x: 424, y: 266 },
];
const CHIP_W = 64;
const CHIP_H = 26;

export default function OtherAccelerators() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  return (
    <Stage
      rootRef={ref}
      title="Two roofs, one model: where the alternatives win and what they cost"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['comm', 'chip-to-chip']]}
      {...s}
      readout={
        <Readout
          items={
            st === 0
              ? [
                  { label: 'Horizontal axis', value: 'FLOP/byte', kind: 'text', note: 'arithmetic intensity' },
                  { label: 'Vertical axis', value: 'TFLOP/s', kind: 'text', note: 'achieved arithmetic' },
                  { label: 'A chip is', value: 'a roof', kind: 'text', note: 'bandwidth slope + compute ceiling' },
                  { label: 'A workload is', value: 'a point', kind: 'text', note: 'decode sits far left' },
                ]
              : st === 1
                ? [
                    { label: 'H100 ridge', value: '295', kind: 'text', note: '989 ÷ 3.35' },
                    { label: 'Compute ceiling', value: '989 TFLOP/s', kind: 'compute', note: 'dense BF16' },
                    { label: 'Bandwidth floor', value: '3.35 TB/s', kind: 'memory', note: 'the slope' },
                    { label: 'Decode intensity', value: '1 FLOP/byte', kind: 'memory', note: 'deep on the left' },
                  ]
                : st === 2
                  ? [
                      { label: 'H100 ridge', value: '295', kind: 'text', note: '989 ÷ 3.35' },
                      { label: 'SRAM chip ridge', value: '2.4', kind: 'text', note: '188 ÷ 80' },
                      { label: 'SRAM ceiling', value: '188 TFLOP/s', kind: 'compute', note: 'lower peak' },
                      { label: 'SRAM bandwidth', value: '80 TB/s', kind: 'memory', note: 'on-chip' },
                    ]
                  : st === 3
                    ? [
                        { label: 'Decode intensity', value: '1 FLOP/byte', kind: 'memory', note: 'same for both' },
                        { label: 'H100 decode roof', value: '3.35 TFLOP/s', kind: 'memory', note: 'the slope' },
                        { label: 'SRAM decode roof', value: '80 TFLOP/s', kind: 'memory', note: 'the slope' },
                        { label: 'SRAM advantage', value: '~24×', kind: 'text', note: 'before links and overheads' },
                      ]
                    : st === 4
                      ? [
                          { label: '70B in FP16', value: '141 GB', kind: 'memory', note: '70.6B × 2 bytes' },
                          { label: 'MI300X memory', value: '192 GB', kind: 'memory', note: 'one package' },
                          { label: 'Fits in one box', value: 'yes', kind: 'text', note: 'no weight splitting' },
                          { label: 'Spare memory', value: '51 GB', kind: 'text', note: 'KV cache and workspace' },
                        ]
                      : st === 5
                        ? [
                            { label: '70B in FP16', value: '141 GB', kind: 'memory', note: 'the same weights' },
                            { label: 'SRAM per chip', value: '230 MB', kind: 'memory', note: 'a fraction of a GB' },
                            { label: 'Chips needed', value: '~614', kind: 'comm', note: '141 ÷ 0.23' },
                            { label: 'At 8-bit', value: '~307 chips', kind: 'comm', note: 'half the weight bytes' },
                          ]
                        : [
                            { label: 'What changes', value: 'software', kind: 'text', note: 'kernels, formats, collectives' },
                            { label: 'What stays', value: 'the physics', kind: 'text', note: 'roofline, KV cache, batching' },
                            { label: 'Compare on', value: 'tokens/s per $', kind: 'text', note: 'at your SLO' },
                            { label: 'And on', value: 'tokens/s per watt', kind: 'text', note: 'whole-system power' },
                          ]
          }
        />
      }
    >
      <svg
        viewBox="0 0 640 360"
        role="img"
        aria-label="A roofline plot comparing an H100 with an SRAM-heavy accelerator, followed by a memory-capacity view showing one 192 GB GPU holding a 70B model against about 614 small SRAM chips linked together, and a two-column summary of what changes and what stays."
      >
        {/* ── the roofline view ─────────────────────────────────── */}
        <Fade show={st < 4}>
          <Label x={X0} y={24} tone={1} size={11.5} bold mono={false}>Every chip is a roof. Every workload is a point.</Label>

          {/* grid */}
          {[0.1, 1, 10, 100, 1000].map((v) => (
            <g key={`gx${v}`}>
              <line x1={px(v)} x2={px(v)} y1={Y1} y2={Y0} className="v-rule faint" />
              <Label x={px(v)} y={Y0 + 14} anchor="middle" tone={3} size={9}>{v < 1 ? '0.1' : fmt.int(v)}</Label>
            </g>
          ))}
          {[0.1, 1, 10, 100, 1000].map((v) => (
            <g key={`gy${v}`}>
              <line x1={X0} x2={X1} y1={py(v)} y2={py(v)} className="v-rule faint" />
              <Label x={X0 - 6} y={py(v) + 3} anchor="end" tone={3} size={9}>{v < 1 ? '0.1' : fmt.int(v)}</Label>
            </g>
          ))}

          {/* axes */}
          <line x1={X0} y1={Y0} x2={X1} y2={Y0} className="v-rule" />
          <line x1={X0} y1={Y0} x2={X0} y2={Y1 - 10} className="v-rule" />
          <Label x={(X0 + X1) / 2} y={Y0 + 32} anchor="middle" tone={2} size={10.5} mono={false}>
            arithmetic intensity — FLOPs done per byte read
          </Label>
          <text x={20} y={(Y0 + Y1) / 2} transform={`rotate(-90 20 ${(Y0 + Y1) / 2})`} textAnchor="middle" className="v-label t2" style={{ fontSize: 10.5 }}>
            achieved TFLOP/s
          </text>

          {/* H100 roof */}
          <Fade show={st >= 1}>
            <path d={h100Roof} fill="none" stroke="var(--text-2)" strokeWidth={1.8} />
            <line x1={X0} x2={X1} y1={py(H100.peak)} y2={py(H100.peak)} stroke="var(--compute)" strokeWidth={1.3} strokeDasharray="6 4" />
            <line x1={px(h100Ridge)} x2={px(h100Ridge)} y1={Y0} y2={py(H100.peak)} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="3 4" />
            <circle cx={px(h100Ridge)} cy={py(H100.peak)} r={3.5} fill="var(--text)" />
            <Label x={X0 + 8} y={py(H100.peak) - 8} size={9.5} kind="compute">
              989 TFLOP/s ceiling
            </Label>
            <Label x={px(1) + 8} y={py(H100.bw) - 8} size={9.5} kind="memory">
              3.35 TB/s slope
            </Label>
            <Label x={px(h100Ridge)} y={Y1 - 4} anchor="middle" tone={1} size={9.5}>
              H100 ridge: 295 FLOP/byte
            </Label>
          </Fade>

          {/* SRAM chip roof */}
          <Fade show={st >= 2}>
            <path d={sramRoof} fill="none" stroke="var(--text-2)" strokeWidth={1.6} strokeDasharray="6 4" />
            <line x1={X0} x2={X1} y1={py(SRAM.peak)} y2={py(SRAM.peak)} stroke="var(--compute)" strokeWidth={1.1} strokeDasharray="3 4" opacity={0.8} />
            <line x1={px(sramRidge)} x2={px(sramRidge)} y1={Y0} y2={py(SRAM.peak)} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="3 4" />
            <circle cx={px(sramRidge)} cy={py(SRAM.peak)} r={3.5} fill="var(--text)" />
            <Label x={px(sramRidge) + 8} y={py(SRAM.peak) - 8} size={9.5} kind="compute">
              SRAM chip: 188 TFLOP/s
            </Label>
            <Label x={px(sramRidge)} y={py(SRAM.peak) + 16} anchor="middle" tone={1} size={9.5}>
              ridge: 2.4
            </Label>
            <Label x={X0 + 8} y={Y0 - 12} size={9.5} tone={2} mono={false}>
              dashed roof: 80 TB/s of on-chip bandwidth, 188 TFLOP/s of arithmetic
            </Label>
          </Fade>

          {/* decode and prefill points */}
          <Fade show={st >= 3}>
            <circle cx={px(DECODE_I)} cy={py(H100.bw)} r={4.5} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={2} />
            <circle cx={px(DECODE_I)} cy={py(SRAM.bw)} r={4.5} fill="var(--memory)" stroke="var(--ground-2)" strokeWidth={2} />
            <circle cx={px(PREFILL_I)} cy={py(H100.peak)} r={4.5} fill="var(--compute)" stroke="var(--ground-2)" strokeWidth={2} />
            <circle cx={px(PREFILL_I)} cy={py(SRAM.peak)} r={4.5} fill="var(--compute)" stroke="var(--ground-2)" strokeWidth={2} />

            <Label x={px(DECODE_I) + 10} y={py(H100.bw) + 3.5} size={9.5} kind="memory">
              H100 decode: 3.35 TFLOP/s
            </Label>
            <Label x={px(DECODE_I) + 10} y={py(SRAM.bw) + 3.5} size={9.5} kind="memory">
              SRAM decode: 80 TFLOP/s
            </Label>
            <Label x={px(PREFILL_I) - 8} y={py(H100.peak) + 3.5} anchor="end" size={9.5} kind="compute">
              prefill: 989
            </Label>
            <Label x={px(PREFILL_I) - 8} y={py(SRAM.peak) + 3.5} anchor="end" size={9.5} kind="compute">
              prefill: 188
            </Label>

            <Label x={X0} y={Y0 + 52} tone={2} size={9.5} mono={false}>
              decode is one FLOP per byte on both chips; the SRAM roof is about 24× higher there
            </Label>
            <Label x={X0} y={Y0 + 66} tone={3} size={9} mono={false}>
              prefill is compute-bound on both; the H100’s higher ceiling wins
            </Label>
          </Fade>
        </Fade>

        {/* ── the capacity view ─────────────────────────────────── */}
        <Fade show={st >= 4 && st < 6}>
          <Label x={64} y={48} tone={1} size={12} bold mono={false}>
            The same 141 GB of weights, two ways to hold them
          </Label>

          {/* one HBM package */}
          <rect x={64} y={74} width={300} height={104} rx={4} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1.4} />
          <text x={78} y={94} className="v-label" style={{ fontSize: 11, fill: 'var(--text)', fontWeight: 700 }}>
            MI300X · 192 GB HBM
          </text>
          <text x={78} y={108} className="v-label t3" style={{ fontSize: 9 }}>
            one package, one memory space
          </text>
          <rect x={78} y={120} width={272} height={18} rx={2} fill="none" stroke="var(--rule-2)" />
          <rect x={78} y={120} width={272 * (HBM_USED / HBM_TOTAL)} height={18} rx={2} fill="var(--memory)" />
          <text x={214} y={133} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fill: '#0e1116', fontWeight: 700 }}>
            141 GB of weights
          </text>
          <text x={78} y={156} className="v-label" style={{ fontSize: 9, fill: 'var(--text-2)' }}>
            51 GB left for KV cache, activations and workspace
          </text>

          <Fade show={st === 4}>
            <text x={64} y={208} className="v-label t2" style={{ fontSize: 9.5 }}>
              One HBM GPU holds the whole model. No chip-to-chip hop is needed to read a weight.
            </text>
          </Fade>

          {/* many SRAM chips */}
          <Fade show={st >= 5}>
            <line x1={64} y1={196} x2={584} y2={196} className="v-rule faint" />
            <Label x={64} y={218} tone={1} size={11} bold mono={false}>
              Groq-like LPU: 230 MB of SRAM per chip
            </Label>

            {/* violet links first, so the boxes sit on top */}
            {CHIPS.map((c, i) => {
              const col = i % 6;
              const row = Math.floor(i / 6);
              return (
                <g key={`links-${i}`}>
                  {col < 5 && (
                    <line x1={c.x + CHIP_W} x2={c.x + CHIP_W + 8} y1={c.y + CHIP_H / 2} y2={c.y + CHIP_H / 2} stroke="var(--comm)" strokeWidth={1.4} />
                  )}
                  {row === 0 && (
                    <line x1={c.x + CHIP_W / 2} x2={c.x + CHIP_W / 2} y1={c.y + CHIP_H} y2={c.y + CHIP_H + 10} stroke="var(--comm)" strokeWidth={1.4} />
                  )}
                </g>
              );
            })}

            {CHIPS.map((c, i) => (
              <g key={`chip-${i}`} className="v-cell">
                <rect x={c.x} y={c.y} width={CHIP_W} height={CHIP_H} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} />
                <text x={c.x + CHIP_W / 2} y={c.y + 12} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: 'var(--text)' }}>
                  SRAM
                </text>
                <text x={c.x + CHIP_W / 2} y={c.y + 21} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>
                  230 MB
                </text>
              </g>
            ))}

            <text x={500} y={242} className="v-label" style={{ fontSize: 11, fill: 'var(--text)', fontWeight: 700 }}>
              × 614 chips
            </text>
            <text x={500} y={256} className="v-label t3" style={{ fontSize: 8.5 }}>
              141 ÷ 0.23
            </text>
            <text x={500} y={270} className="v-label t3" style={{ fontSize: 8.5 }}>
              about 307 at 8-bit
            </text>

            <text x={64} y={312} className="v-label t2" style={{ fontSize: 9.5 }}>
              Each chip holds a slice. The violet links are how the slices cooperate.
            </text>
            <text x={64} y={326} className="v-label t3" style={{ fontSize: 9 }}>
              The same small SRAM that buys latency leaves little room for KV caches or a large batch.
            </text>
          </Fade>
        </Fade>

        {/* ── what changes, what stays ──────────────────────────── */}
        <Fade show={st >= 6}>
          <Label x={64} y={48} tone={1} size={12} bold mono={false}>
            What changes, and what does not
          </Label>
          <line x1={320} y1={62} x2={320} y2={284} className="v-rule faint" />

          <text x={64} y={78} className="v-label" style={{ fontSize: 11, fill: 'var(--text)', fontWeight: 700 }}>
            What changes when you switch
          </text>
          <text x={336} y={78} className="v-label" style={{ fontSize: 11, fill: 'var(--text)', fontWeight: 700 }}>
            What stays the same
          </text>

          {[
            ['Kernels and libraries: CUDA → ROCm, XLA or Neuron', 'The roofline and the ridge point'],
            ['Quantisation formats and engine backends', 'Arithmetic intensity decides the bound'],
            ['Collectives, profilers and graph capture', 'KV cache maths and prefill vs decode'],
            ['How many chips hold the model', 'Batching, SLOs and goodput'],
            ['The size of the batch you can afford', 'The model’s mathematics'],
          ].map(([left, right], i) => (
            <g key={`row-${i}`}>
              <text x={64} y={104 + i * 30} className="v-label t2" style={{ fontSize: 9.5 }}>
                • {left}
              </text>
              <text x={336} y={104 + i * 30} className="v-label t2" style={{ fontSize: 9.5 }}>
                • {right}
              </text>
            </g>
          ))}

          <text x={64} y={304} className="v-label" style={{ fontSize: 10.5, fill: 'var(--text)' }}>
            The physics is portable. The software is not. Measure before you migrate.
          </text>
        </Fade>
      </svg>
    </Stage>
  );
}
