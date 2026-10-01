import { useRef, useState } from 'react';
import { Stage, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Bar } from '../lib/svg';
import { FORMATS, encodeFloat, encodeInt } from './floatFormats';

const STEPS = ['Bits and bytes', 'Integers', 'The floating-point idea', 'FP32', 'FP16 vs BF16', 'FP8 and 4-bit', 'What it costs'];
const CAPTIONS = [
  'A bit is a 0 or 1. Eight bits make a byte, which can hold 2⁸ = 256 different patterns.',
  'Integers map each pattern to a whole number. INT8 covers −128 to 127; INT4 only −8 to 7. Anything else must be rounded or clipped.',
  'Floating point stores numbers like scientific notation: a sign, a few significant digits (mantissa) and a power (exponent).',
  'FP32 spends 1 bit on the sign, 8 on the exponent (range) and 23 on the mantissa (precision). 4 bytes per number.',
  'FP16 and BF16 both use 2 bytes but split them differently: FP16 keeps precision, BF16 keeps FP32’s range.',
  'FP8 fits a number into one byte; 4-bit formats into half a byte. Rounding error grows as bits shrink.',
  'For an 8-billion-parameter model, bytes per number decides whether it fits on a GPU at all, and how fast it can be read.',
];

const PRESETS = [
  { v: Math.PI, label: 'π' },
  { v: 0.1, label: '0.1' },
  { v: 1 / 3, label: '1/3' },
  { v: 0.00042, label: '0.00042' },
  { v: 1000.7, label: '1000.7' },
  { v: 70000, label: '70000' },
];

const CELL = 15;
function BitCells({ x, y, groups, cell = CELL }: { x: number; y: number; groups: { bits: number[]; label: string; tone: 0 | 1 | 2 }[]; cell?: number }) {
  let cx = x;
  const out = [];
  for (let gi = 0; gi < groups.length; gi++) {
    const g = groups[gi];
    const start = cx;
    for (let i = 0; i < g.bits.length; i++) {
      const b = g.bits[i];
      const fill = b ? (g.tone === 0 ? 'var(--text)' : g.tone === 1 ? 'var(--text-2)' : 'var(--text-3)') : 'transparent';
      out.push(
        <g key={`${gi}-${i}`}>
          <rect x={cx} y={y} width={cell - 2} height={cell + 2} rx={1.5} fill={fill} stroke={b ? fill : 'var(--rule-2)'} className="v-cell" />
          <text x={cx + (cell - 2) / 2} y={y + cell / 2 + 4.5} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fill: b ? '#0e1116' : 'var(--text-3)' }}>{b}</text>
        </g>,
      );
      cx += cell;
    }
    const end = cx - 2;
    out.push(
      <g key={`br-${gi}`}>
        <path d={`M${start},${y + cell + 7} v4 H${end} v-4`} fill="none" stroke="var(--rule-2)" />
        {end - start > 24 && <Label x={(start + end) / 2} y={y + cell + 22} anchor="middle" tone={3} size={9}>{g.label}</Label>}
      </g>,
    );
    cx += 4;
  }
  return <g>{out}</g>;
}

function fmtVal(v: number) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(6);
  return v.toPrecision(9).replace(/0+$/, '').replace(/\.$/, '');
}
function relErr(exact: number, got: number) {
  if (exact === 0) return '0';
  const e = Math.abs(got - exact) / Math.abs(exact);
  if (e === 0) return 'exact';
  return `${(e * 100).toPrecision(2)}% off`;
}

export default function NumberFormats() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [pi, setPi] = useState(0);
  const v = PRESETS[pi].v;

  const f32 = encodeFloat(v, FORMATS[0]);
  const f16 = encodeFloat(v, FORMATS[1]);
  const b16 = encodeFloat(v, FORMATS[2]);
  const f8 = encodeFloat(v, FORMATS[3]);
  const i8 = encodeInt(v, 8);
  const i4 = encodeInt(v, 4);

  const row = (y: number, name: string, enc: ReturnType<typeof encodeFloat>, bytes: number, show: boolean, cell = CELL) => (
    <Fade show={show}>
      <g>
        <Label x={20} y={y + 12} tone={1} bold size={11}>{name}</Label>
        <Label x={20} y={y + 26} tone={3} size={9.5}>{bytes >= 1 ? `${bytes} byte${bytes > 1 ? 's' : ''}` : 'half a byte'}</Label>
        <BitCells x={106} y={y} cell={cell} groups={[
          { bits: [enc.sign], label: 'sign', tone: 0 },
          { bits: enc.expBits, label: 'exponent (range)', tone: 1 },
          { bits: enc.mantBits, label: 'mantissa (precision)', tone: 2 },
        ]} />
        <Label x={626} y={y + 12} anchor="end" tone={1} size={11}>{enc.overflow ? `${fmtVal(enc.value)} (clipped!)` : fmtVal(enc.value)}</Label>
        <Label x={626} y={y + 26} anchor="end" tone={3} size={9.5} kind={enc.overflow ? 'error' : undefined}>{relErr(v, enc.value)}</Label>
      </g>
    </Fade>
  );

  const byte = [0, 1, 1, 0, 1, 0, 0, 1];
  const params = 8.03e9;
  const mem = [
    { name: 'FP32', b: 4 }, { name: 'BF16', b: 2 }, { name: 'FP8', b: 1 }, { name: 'INT4', b: 0.5 },
  ];

  return (
    <Stage
      rootRef={ref}
      title="How a number lives in memory"
      steps={STEPS}
      captions={CAPTIONS}
      {...s}
      controls={st >= 1 && st <= 5 ? <Segmented label="Number to store" value={pi} options={PRESETS.map((p, i) => ({ value: i, label: p.label }))} onChange={setPi} /> : undefined}
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Bit layouts of the same number in several formats, with the rounding error of each.">
        {/* step 0: a byte */}
        <Fade show={st === 0}>
          <g>
            <Label x={320} y={60} anchor="middle" tone={2} mono={false} size={12}>one byte = 8 bits</Label>
            <BitCells x={260} y={80} groups={[{ bits: byte, label: '', tone: 1 }]} cell={16} />
            <Label x={320} y={140} anchor="middle" tone={1} size={13}>01101001 in binary = 64 + 32 + 8 + 1 = 105</Label>
            {[128, 64, 32, 16, 8, 4, 2, 1].map((p, i) => <Label key={p} x={260 + i * 16 + 7} y={74} anchor="middle" tone={3} size={8}>{p}</Label>)}
            <Label x={320} y={180} anchor="middle" tone={2} mono={false} size={12}>{'8 bits → 2⁸ = 256 possible patterns'}</Label>
            <Label x={320} y={204} anchor="middle" tone={2} mono={false} size={12}>{'16 bits → 65,536 · 32 bits → 4.3 billion'}</Label>
            <Label x={320} y={250} anchor="middle" tone={3} mono={false} size={11}>A format is just a rule for which number each pattern means.</Label>
          </g>
        </Fade>

        {/* step 1: integers */}
        <Fade show={st === 1}>
          <g>
            <Label x={20} y={40} tone={2} mono={false} size={11.5}>{`Storing ${PRESETS[pi].label} = ${fmtVal(v)} as an integer`}</Label>
            <Label x={20} y={82} tone={1} bold size={11}>INT8</Label>
            <BitCells x={106} y={70} groups={[{ bits: i8.bits, label: 'range −128 … 127', tone: 1 }]} />
            <Label x={626} y={82} anchor="end" tone={1} size={11}>{i8.value}{i8.clipped ? ' (clipped!)' : ''}</Label>
            <Label x={20} y={152} tone={1} bold size={11}>INT4</Label>
            <BitCells x={106} y={140} groups={[{ bits: i4.bits, label: 'range −8 … 7', tone: 1 }]} />
            <Label x={626} y={152} anchor="end" tone={1} size={11}>{i4.value}{i4.clipped ? ' (clipped!)' : ''}</Label>
            <Label x={20} y={230} tone={3} mono={false} size={11}>Integers cannot hold fractions and have a hard range. That is why quantized models also store a</Label>
            <Label x={20} y={248} tone={3} mono={false} size={11}>{'scale factor: real value ≈ integer × scale. (Module 11 builds this.)'}</Label>
          </g>
        </Fade>

        {/* step 2: scientific notation */}
        <Fade show={st === 2}>
          <g>
            <Label x={320} y={70} anchor="middle" tone={1} math size={26}>{'− 6.022 × 10²³'}</Label>
            <Label x={214} y={104} anchor="middle" tone={3} size={10}>sign</Label>
            <Label x={284} y={104} anchor="middle" tone={3} size={10}>mantissa</Label>
            <Label x={390} y={104} anchor="middle" tone={3} size={10}>exponent</Label>
            <Label x={320} y={160} anchor="middle" tone={2} mono={false} size={12}>Computers do the same thing in base 2:</Label>
            <Label x={320} y={200} anchor="middle" tone={1} math size={22}>{`value = (−1)^sign × 1.mantissa × 2^(exponent − bias)`}</Label>
            <Label x={320} y={250} anchor="middle" tone={3} mono={false} size={11}>More exponent bits = bigger and smaller numbers (range). More mantissa bits = more digits (precision).</Label>
          </g>
        </Fade>

        {/* steps 3-5: float rows */}
        {row(40, 'FP32', f32, 4, st === 3 || st === 4, 13)}
        {row(110, 'FP16', f16, 2, st === 4)}
        {row(180, 'BF16', b16, 2, st === 4)}
        {row(40, 'BF16', b16, 2, st === 5)}
        {row(110, 'FP8 E4M3', f8, 1, st === 5)}
        <Fade show={st === 5}>
          <g>
            <Label x={20} y={192} tone={1} bold size={11}>INT4</Label>
            <Label x={20} y={206} tone={3} size={9.5}>half a byte</Label>
            <BitCells x={106} y={180} groups={[{ bits: i4.bits, label: 'needs a scale', tone: 1 }]} />
            <Label x={626} y={192} anchor="end" tone={1} size={11}>{i4.value}</Label>
            <Label x={626} y={206} anchor="end" tone={3} size={9.5}>{relErr(v, i4.value)}</Label>
          </g>
        </Fade>
        <Fade show={st >= 3 && st <= 5}>
          <Label x={20} y={300} tone={3} mono={false} size={11}>{FORMATS[st === 3 ? 0 : st === 4 ? 2 : 3].note}</Label>
        </Fade>

        {/* step 6: memory for an 8B model */}
        <Fade show={st === 6}>
          <g>
            <Label x={20} y={36} tone={2} mono={false} size={11.5}>{'Llama-3.1-8B: 8.03 billion weights × bytes per weight'}</Label>
            {mem.map((m, i) => (
              <Bar key={m.name} x={110} y={60 + i * 42} w={420} h={20} frac={(params * m.b) / 32.2e9} kind="memory" label={m.name} value={`${((params * m.b) / 1e9).toFixed(1)} GB`} />
            ))}
            <line x1={110 + (16e9 / 32.2e9) * 420} x2={110 + (16e9 / 32.2e9) * 420} y1={50} y2={236} stroke="var(--error)" strokeDasharray="3 3" />
            <Label x={110 + (16e9 / 32.2e9) * 420 + 4} y={250} tone={2} size={9.5} kind="error">T4 memory: 16 GB</Label>
            <Label x={20} y={290} tone={3} mono={false} size={11}>Halving the bytes halves the memory and the time to read every weight once (one decode step).</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
