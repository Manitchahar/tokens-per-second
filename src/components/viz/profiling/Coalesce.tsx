import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Dim, Fade, fmt } from '../lib/svg';
import { warpBytes, sectorsTouched } from './profiling';

const STEPS = [
  'One warp asks for 32 numbers',
  'The memory system answers in 32-byte sectors',
  'Coalesced: the answer fits in four sectors',
  'Strided: the same load costs 32× the traffic',
  'What a strided access costs in a real kernel',
  'Fixing it: transpose once, not everywhere',
  'Shared memory banks',
  'The rule this all comes down to',
];
const CAPTIONS = [
  'A warp issues one load instruction for 32 lanes. Each lane wants one number, and the addresses are whatever your indexing said.',
  'The load/store unit does not fetch bytes. It gathers the 32 addresses and fetches every 32-byte sector any of them touches.',
  'If the 32 lanes read consecutive BF16 values, that is 64 contiguous bytes: two sectors, one shot. Nothing wasted.',
  'If instead every other lane is read, the 32 addresses span 128 bytes and every sector is half wasted — but the hardware must move them all.',
  'For an 8B model’s 33.5 MB weight matrix, a badly strided read pulls tens of gigabytes instead of 33 MB. The kernel gets 30× slower with identical arithmetic.',
  'The fix is almost always to make the innermost loop index consecutive — a transpose, a layout change, or a tiled kernel that reads shared memory instead.',
  'Shared memory has no cache lines but it does have 32 banks; two lanes hitting the same bank on the same cycle serialize. Pad the stride by one.',
  'Achieved bandwidth only counts bytes you wanted. Every extra sector fetched is a byte of HBM bandwidth you spent to throw away.',
];

const BYTES_PER_ELEM = 2; // BF16
const matElems = 4096; // one row of a 4096-wide weight matrix

/** 32 lanes, drawn as a row of cells. Each cell is one lane's address. */
const LANE_X0 = 96;
const LANE_W = 15;
const LANE_GAP = 2;
const laneX = (i: number) => LANE_X0 + i * (LANE_W + LANE_GAP);
const ROW_Y = 118;

/** A 32-byte sector covers 16 BF16 values. */
const ELEMS_PER_SECTOR = 32 / BYTES_PER_ELEM;

const PRESET: Record<number, { stride: number; perLane: number }> = {
  0: { stride: 1, perLane: 1 },
  1: { stride: 1, perLane: 1 },
  2: { stride: 1, perLane: 1 },
  3: { stride: 8, perLane: 1 },
  4: { stride: 8, perLane: 1 },
  5: { stride: 1, perLane: 1 },
  6: { stride: 1, perLane: 1 },
  7: { stride: 1, perLane: 1 },
};

export default function Coalesce() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const p = PRESET[s.step];
  const [stride, setStride] = useState(p.stride);
  const [perLane, setPerLane] = useState(p.perLane);
  useEffect(() => {
    setStride(PRESET[s.step].stride); setPerLane(PRESET[s.step].perLane);
  }, [s.step]);

  const ideal = warpBytes(BYTES_PER_ELEM, perLane);
  const sectors = sectorsTouched(BYTES_PER_ELEM, perLane, stride);
  const actual = sectors * 32;
  const penalty = actual / ideal;

  // The weight-matrix example: one row of a 4096×4096 BF16 weight.
  const matIdealBytes = matElems * BYTES_PER_ELEM;
  const matRatio = sectors / (matIdealBytes / 32);

  return (
    <Stage
      rootRef={ref}
      title="Coalescing: why one load instruction can cost 32× the bandwidth"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', ['compute', 'useful bytes'], ['error', 'wasted sectors'], 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Bytes the warp wanted', value: `${ideal} B`, kind: 'compute', note: '32 lanes × 1 BF16 each' },
          { label: 'Bytes the memory system moved', value: fmt.bytes(actual), kind: penalty > 1.5 ? 'error' : 'memory', note: `${sectors} sectors of 32 B` },
          { label: 'Waste factor', value: `${penalty.toFixed(1)}×`, kind: penalty > 1.5 ? 'error' : 'token', note: penalty > 1.5 ? 'uncoalesced' : 'coalesced' },
          { label: 'Effective bandwidth', value: `${(penalty > 1.5 ? '1' : '100')}%`, kind: penalty > 1.5 ? 'error' : 'token', note: 'of what the hardware could do' },
        ]} />
      }
      controls={
        <>
          <Slider label="Distance between the 32 lanes' addresses (in elements)" value={stride} min={1} max={64} step={1} onChange={setStride} format={(v) => `${v}`} kind="memory" />
          <Slider label="Elements each lane loads (vector width)" value={perLane} min={1} max={8} step={1} onChange={setPerLane} format={(v) => `${v}`} kind="compute" />
          <Segmented
            label="Typical pattern"
            value={`${stride}:${perLane}`}
            options={[
              { value: '1:1', label: 'contiguous' },
              { value: '2:1', label: 'stride 2' },
              { value: '8:1', label: 'stride 8' },
              { value: '1:4', label: 'float4 load' },
            ]}
            onChange={(v) => {
              const [a, b] = v.split(':').map(Number);
              setStride(a); setPerLane(b);
            }}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 430" role="img" aria-label="Thirty-two lanes of a warp each requesting one BF16 value, shown against the 32-byte sectors the memory system actually fetches, comparing a coalesced access with a strided one and the wasted bandwidth that results.">
        {/* ── the 32 lanes ── */}
        <Label x={16} y={ROW_Y - 44} tone={1} size={11} bold mono={false}>
          {s.step < 2 ? 'One load instruction, 32 lanes' : s.step < 4 ? 'Strided: every 8th element' : 'Contiguous again'}
        </Label>
        <Label x={624} y={ROW_Y - 44} anchor="end" tone={3} size={9.5}>
          {`lane i reads element ${stride === 1 ? 'i' : `i×${stride}`} · ${perLane} elem${perLane > 1 ? 's' : ''} each`}
        </Label>

        {/* sector boundaries behind the lanes */}
        {Array.from({ length: Math.ceil((32 * stride) / ELEMS_PER_SECTOR) }, (_, sIdx) => {
          const firstElem = sIdx * ELEMS_PER_SECTOR;
          const lastElem = Math.min((sIdx + 1) * ELEMS_PER_SECTOR - 1, 32 * stride - 1);
          const x1 = laneX(Math.min(firstElem / stride, 31));
          const x2 = laneX(Math.min(lastElem / stride, 31)) + LANE_W;
          const touches = Math.max(0, Math.min(32, Math.ceil((lastElem + 1) / stride)) - Math.min(32, Math.ceil(firstElem / stride)));
          const useful = touches * perLane * BYTES_PER_ELEM;
          const wasted = Math.max(0, 32 - useful);
          return (
            <g key={sIdx} className="v-fade">
              <rect x={x1 - 1} y={ROW_Y - 30} width={Math.max(4, x2 - x1 + 2)} height={44} rx={2}
                fill={wasted > useful ? 'url(#tps-hatch-red)' : 'var(--memory-dim)'}
                stroke={wasted > useful ? 'var(--error)' : 'var(--memory)'} strokeWidth={0.8} opacity={0.55} />
              <Label x={(x1 + x2) / 2} y={ROW_Y - 36} anchor="middle" size={8} tone={3}>32 B sector</Label>
            </g>
          );
        })}

        {/* the lanes themselves */}
        {Array.from({ length: 32 }, (_, i) => {
          const elem = i * stride;
          const inSectorStart = elem % ELEMS_PER_SECTOR;
          const used = inSectorStart + perLane * BYTES_PER_ELEM <= 32;
          const full = stride <= 1 && perLane === 1;
          return (
            <g key={i} className="v-fade">
              <rect x={laneX(i)} y={ROW_Y - 12} width={LANE_W} height={18} rx={1.5}
                fill={full || used ? 'var(--compute)' : 'url(#tps-hatch-red)'}
                stroke={full || used ? 'var(--compute)' : 'var(--error)'} strokeWidth={0.7} className="v-cell" />
              <Label x={laneX(i) + LANE_W / 2} y={ROW_Y + 1} anchor="middle" size={6.5} tone={full || used ? 1 : 3}>{`${i}`}</Label>
            </g>
          );
        })}
        <Label x={laneX(16)} y={ROW_Y + 20} anchor="middle" size={9} tone={3}>lane</Label>

        {/* ── the accounting ── */}
        <line x1={16} y1={170} x2={624} y2={170} className="v-rule faint" />
        <Label x={16} y={192} tone={2} size={11} bold mono={false}>What this cost</Label>

        <Label x={16} y={214} tone={3} size={9.5}>wanted</Label>
        <rect x={70} y={206} width={420} height={12} rx={2} fill="none" stroke="var(--rule)" />
        <rect x={70} y={206} width={420} height={12} rx={2} fill="var(--compute)" className="v-cell" />
        <Label x={500} y={216} size={9.5} kind="compute">{`${ideal} B`}</Label>

        <Label x={16} y={234} tone={3} size={9.5}>moved</Label>
        <rect x={70} y={226} width={420} height={12} rx={2} fill="none" stroke="var(--rule)" />
        <rect x={70} y={226} width={420 * Math.min(penalty, 4) / 4} height={12} rx={2} fill={penalty > 1.5 ? 'var(--error)' : 'var(--memory)'} className="v-cell" />
        <Label x={500} y={236} size={9.5} kind={penalty > 1.5 ? 'error' : 'memory'}>{`${actual} B (${sectors} sectors)`}</Label>

        <Dim x1={490} x2={70 + 420 * Math.min(penalty, 4) / 4} y={252}
          label={penalty > 1.5 ? `${fmt.bytes(actual - ideal)} wasted` : 'nothing wasted'} kind={penalty > 1.5 ? 'error' : 'token'} />

        {/* ── the real-kernel example ── */}
        <Fade show={s.step >= 4}>
          <Label x={16} y={288} tone={2} size={11} bold mono={false}>Applied to one row of a Llama-3.1-8B weight matrix (4,096 BF16 = 8 KiB)</Label>
          <Label x={16} y={308} tone={3} size={9.5}>
            {`Ideal: ${fmt.bytes(matIdealBytes)} moved. Strided at ${stride}: ${(matIdealBytes * matRatio).toFixed(0)} sectors → ${fmt.bytes(matIdealBytes * matRatio)}.`
            }
          </Label>
          <rect x={16} y={316} width={500} height={14} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={16} y={316} width={500 * Math.min(matRatio, 8) / 8} height={14} rx={2} fill="var(--error)" className="v-cell" />
          <Label x={524} y={327} size={9.5} kind="error">{`${matRatio.toFixed(0)}× traffic`}</Label>
          <Label x={16} y={350} tone={3} size={9.5}>
            Same FLOPs. Same weight matrix. The arithmetic is untouched — only the addresses moved.
          </Label>
        </Fade>

        {/* ── the fix ── */}
        <Fade show={s.step >= 5}>
          <line x1={16} y1={358} x2={624} y2={358} className="v-rule faint" />
          <Arrow x1={190} y1={374} x2={240} y2={374} kind="token" />
          <Label x={248} y={378} size={10} kind="token" bold>fix: make the innermost index consecutive</Label>
          <Label x={16} y={396} tone={3} size={9.5}>
            Transpose once, change the layout, or tile through shared memory.
          </Label>
        </Fade>

        {/* ── bank conflicts ── */}
        <Fade show={s.step >= 6}>
          <Label x={16} y={414} tone={3} size={9.5} kind="error">
            Shared memory: 32 banks — two lanes on one bank in a cycle serialize. Pad the stride by 1.
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
