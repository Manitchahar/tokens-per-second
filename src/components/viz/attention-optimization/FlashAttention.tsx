import { useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Dim, Fade, Move, Matrix, fmt } from '../lib/svg';

const BR = 64; // query rows per tile
const BC = 64; // key columns per tile
const HEADS = 32;
const D = 128;
const BYTES = 2;

const tileBytes = BR * BC * BYTES;
const sramBytes = 228e3; // H100 shared memory per SM, approximate

/** FlashAttention's actual backward path needs two copies of Q, K, V in SRAM. */
// Q tile (BF16) + K and V tiles (BF16) + FP32 output accumulator + m and l (FP32); the score tile is added separately.
const stateBytes = (BR: number, bc: number) => BR * D * BYTES + 2 * bc * D * BYTES + BR * D * 4 + 2 * BR * 4;

const STEPS = [
  'One query tile, start to finish',
  'Load Q into SRAM',
  'Load a K tile, compute S',
  'Online softmax updates m, ℓ, O',
  'Repeat until all K tiles are done',
  'Divide once at the end',
  'The causal mask skips tiles',
  'Never write the intermediate',
];

const CAPTIONS = [
  'One program instance — one CUDA block, one Triton program — owns 64 query rows and takes total responsibility for them from start to finish. Everything below happens inside that single block.',
  'Q for those 64 rows goes into on-chip SRAM and stays there for the whole loop. 16 KB, held in registers or shared memory. It is never touched again while the key tiles stream past.',
  'A 64x64 tile of keys arrives. The matmul against Q produces a 64x64 block of scores in SRAM — 8 KB — and it has never touched HBM. This is the tensor that eager attention wrote to memory.',
  'The online softmax folds this score tile into the running state: three vectors, m, ℓ and O, each 64 numbers or 64x128. Every tile updates them and moves on.',
  'Repeat for every key tile. On an H100 the loop runs about four times for 256 tokens of context, and sixteen times for 1,024. The final O is the same answer a two-pass softmax would give.',
  'After the last tile, divide O by ℓ — once, in registers, on a 64x128 tensor. This is the normalisation that eager attention applied to a whole T-by-T matrix.',
  'Because attention is causal, query tile i never needs key tile j for j greater than i. Those tiles are skipped: no load, no matmul. With 64-row tiles that removes a growing fraction of the work as context grows.',
  'Nothing in this loop wrote S or P to HBM. Peak extra memory is the Q tile, one K and V tile, the score tile, the FP32 output accumulator and m, ℓ — about 89 KB instead of gigabytes.',
];

export default function FlashAttention() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [keyTiles, setKeyTiles] = useState(4);

  const loops = Math.max(1, keyTiles);
  const triTiles = (loops * (loops + 1)) / 2; // tiles on or below the diagonal, over all query tiles
  const skippedPct = Math.round((1 - triTiles / (loops * loops)) * 100);

  const peak = stateBytes(BR, BC) + tileBytes;
  const eager = 2 * HEADS * 4096 * 4096 * BYTES;

  const GX = 40;
  const GY = 92;
  const CELL = 15;
  const GAP = 3;
  const N = 8; // 8x8 tiles of 64 = 512 tokens, shown as a schematic

  const cx = (i: number) => GX + i * (CELL + GAP);
  const cy = (i: number) => GY + i * (CELL + GAP);
  const gridSide = N * (CELL + GAP) - GAP;
  // the highlighted query tile is the last one in this context: it sees key tiles 0..qi and nothing after
  const qi = Math.min(keyTiles, N) - 1;

  return (
    <Stage
      rootRef={ref}
      title="Inside one FlashAttention program"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'query tile, in flight'],
        ['memory', 'key tile'],
        ['error', 'skipped by the mask'],
        ['token', 'output written'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Tile', value: `${BR} x ${BC}`, kind: 'compute', note: 'Q rows x K columns' },
          { label: 'Score tile', value: `${(tileBytes / 1024).toFixed(0)} KB`, kind: 'memory', note: 'in SRAM, never written' },
          { label: 'Peak extra', value: `${(peak / 1024).toFixed(1)} KB`, kind: 'memory', note: 'Q, K, V, S + O, m, ℓ' },
          { label: 'Eager equivalent', value: fmt.gb(eager), kind: 'error', note: 'one layer, 4096 tokens' },
        ]} />
      }
      controls={
        <Segmented
          label="Key tiles in the context"
          value={keyTiles}
          options={[
            { value: 2, label: '2 (128 tok)' },
            { value: 4, label: '4 (256 tok)' },
            { value: 8, label: '8 (512 tok)' },
            { value: 16, label: '16 (1k tok)' },
          ]}
          onChange={setKeyTiles}
        />
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A schematic score matrix split into tiles. One query tile is highlighted and streams through the key tiles; tiles above the diagonal are skipped because attention is causal. To the right, the running state of maximum, sum and output accumulator is updated at each tile, and a final division produces the output.">
        {/* ── the tiled score matrix ──────────────────────── */}
        <Label x={40} y={28} tone={2} size={11} mono={false} bold>The score matrix, tiled</Label>
        <Label x={40} y={44} tone={3} size={9.5}>each cell = one {BR}x{BC} tile · {N*N} tiles shown</Label>

        <Fade show={s.step >= 1}>
          {/* key tiles along the top */}
          {Array.from({ length: N }, (_, j) => (
            <rect key={`k${j}`} x={cx(j)} y={GY - 26} width={CELL} height={16} rx={2}
              fill={j < keyTiles ? 'var(--memory)' : 'var(--ground-3)'}
              fillOpacity={j < keyTiles ? 0.85 : 1}
              stroke={j < keyTiles ? 'var(--memory)' : 'var(--rule-2)'}
              className="v-cell" />
          ))}
          <Label x={cx(0)} y={GY - 32} tone={3} size={9}>key tiles K</Label>

          {/* the grid */}
          {Array.from({ length: N }, (_, i) =>
            Array.from({ length: N }, (_, j) => {
              const isQ = i === qi;
              const active = isQ && j <= qi && s.step >= 2;
              const masked = isQ && j > qi;
              return (
                <rect key={`${i}-${j}`} x={cx(j)} y={cy(i)} width={CELL} height={CELL} rx={2}
                  fill={masked ? 'var(--error-dim)' : active ? 'var(--compute)' : isQ ? 'var(--compute-dim)' : j > i ? 'var(--error-dim)' : 'var(--ground-3)'}
                  fillOpacity={active ? 1 : 0.9}
                  stroke={active ? 'var(--compute)' : masked ? 'var(--error)' : 'var(--rule-2)'}
                  strokeWidth={active ? 1.6 : 0.8}
                  className="v-cell" />
              );
            }),
          )}

          {/* query tile marker */}
          <rect x={cx(0) - 2} y={cy(qi) - 2} width={(qi + 1) * (CELL + GAP) - GAP + 4} height={CELL + 4} rx={3}
            fill="none" stroke="var(--compute)" strokeWidth={1.8} />
          <Label x={GX + gridSide + 8} y={cy(qi) + 11} tone={3} size={9}>query tile</Label>

          <Label x={GX + gridSide + 8} y={GY + 4} size={9} kind="error">masked:</Label>
          <Label x={GX + gridSide + 8} y={GY + 15} size={9} kind="error">above the</Label>
          <Label x={GX + gridSide + 8} y={GY + 26} size={9} kind="error">diagonal</Label>
        </Fade>

        <Label x={GX} y={GY + gridSide + 22} tone={3} size={9}>{`${N} × ${N} tiles of ${BR} = ${N * BR} tokens (schematic)`}</Label>

        {/* ── the loop, drawn as a chain ──────────────────── */}
        <Label x={248} y={28} tone={2} size={11} mono={false} bold>One query tile, the whole loop</Label>

        <Fade show={s.step >= 1}>
          <Span x={248} y={44} w={86} h={30} kind="compute" label="Q tile" />
          <Label x={291} y={88} anchor="middle" tone={3} size={9}>16 KB, stays</Label>
        </Fade>

        <Arrow x1={336} y1={59} x2={360} y2={59} kind="compute" opacity={s.step >= 2 ? 1 : 0} />

        <Fade show={s.step >= 2}>
          <rect x={362} y={38} width={96} height={42} rx={3} fill="var(--ground-3)" stroke="var(--memory)" strokeWidth={1.3} />
          <Label x={410} y={55} anchor="middle" size={9.5} kind="memory" bold>load K tile</Label>
          <Label x={410} y={70} anchor="middle" size={9.5} kind="compute">S = QKᵀ in SRAM</Label>
          <Label x={410} y={92} anchor="middle" tone={3} size={9}>8 KB, never stored</Label>
        </Fade>

        <Arrow x1={460} y1={59} x2={484} y2={59} kind="compute" opacity={s.step >= 3 ? 1 : 0} />

        <Fade show={s.step >= 3}>
          <rect x={486} y={38} width={118} height={42} rx={3} fill="var(--ground-3)" stroke="var(--compute)" strokeWidth={1.3} />
          <Label x={545} y={55} anchor="middle" size={9.5} kind="compute" bold>update m, ℓ, O</Label>
          <Label x={545} y={70} anchor="middle" size={9.5}>online softmax</Label>
          <Label x={545} y={92} anchor="middle" tone={3} size={9}>kept on chip</Label>
        </Fade>

        {/* running state readout */}
        <Fade show={s.step >= 3}>
          <rect x={248} y={112} width={356} height={62} rx={3} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={262} y={130} tone={2} size={10} mono={false} bold>The state carried between tiles</Label>
          {[
            ['m', 'max seen', 'var(--compute)'],
            ['ℓ', 'sum so far', 'var(--compute)'],
            ['O', 'output accumulator', 'var(--token)'],
          ].map(([sym, what, col], k) => (
            <g key={sym}>
              <rect x={262 + k * 116} y={138} width={106} height={26} rx={2} fill="var(--ground-3)" stroke="var(--rule-2)" />
              <text x={272 + k * 116} y={155} className="v-label" style={{ fontSize: 12, fill: col, fontWeight: 700 }}>{sym}</text>
              <text x={288 + k * 116} y={155} className="v-label" style={{ fontSize: 9, fill: 'var(--text-3)' }}>{what}</text>
            </g>
          ))}
          <Label x={262} y={180} tone={3} size={9}>{`m, ℓ: ${BR} numbers · O: ${BR}×${D} · vs ${fmt.gb(eager)} eager`}</Label>
        </Fade>

        {/* loop count */}
        <Fade show={s.step >= 4}>
          <Arrow x1={545} y1={82} x2={410} y2={82} kind="compute" dashed curve={26} />
          <Label x={604} y={105} anchor="end" size={9} kind="compute">{`repeat for every key tile: ${loops}× here`}</Label>
        </Fade>

        {/* divide once */}
        <Fade show={s.step >= 5}>
          <Arrow x1={426} y1={178} x2={426} y2={200} kind="token" />
          <Span x={382} y={202} w={92} h={28} kind="token" label="O /= ℓ" />
          <Label x={382} y={244} tone={3} size={9}>once, on a 64x128 tensor</Label>
        </Fade>

        {/* causal savings */}
        <Fade show={s.step >= 6}>
          <Label x={40} y={292} tone={2} size={11} mono={false} bold>The mask is free speed</Label>
          <Span x={40} y={304} w={200} h={16} kind="compute" dim label={`${triTiles} of ${loops * loops} tiles computed`} labelTone="light" />
          <Span x={248} y={304} w={120} h={16} kind="error" dim label={`${skippedPct}% skipped`} labelTone="light" />
          <Label x={40} y={338} tone={3} size={9.5}>{`${loops} query tiles × ${loops} key tiles = ${loops * loops}; only ${triTiles} are on or below the diagonal, ${skippedPct}% never loaded`}</Label>
        </Fade>

        {/* the punchline */}
        <Fade show={s.step >= 7}>
          <Label x={40} y={358} tone={2} size={11} mono={false} bold>Peak extra memory, one program</Label>
          <Span x={248} y={346} w={356} h={16} kind="memory" dim label={`${(peak / 1024).toFixed(1)} KB — Q, K, V and score tiles + O, m, ℓ`} labelTone="light" />
        </Fade>
      </svg>
    </Stage>
  );
}