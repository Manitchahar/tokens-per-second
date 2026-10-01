import { useRef, useState } from 'react';
import { Stage, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Dim, Fade, Move, fmt } from '../lib/svg';

// Llama-3.1-8B in BF16. One layer, all 32 query heads.
const HEADS = 32;
const KV_HEADS = 8;
const LAYERS = 32;
const D = 128;
const BYTES = 2;

/** Bytes of the output tensor: the only thing that must survive to the end. */
const outBytes = (T: number) => HEADS * T * D * BYTES;
/** Bytes of the whole KV cache, all 32 layers — what we already pay for. */
const kvBytes = (T: number) => T * 2 * LAYERS * KV_HEADS * D * BYTES; // 131,072 per token
/** Peak eager memory while the score and probability matrices are both resident. */
const scorePeak = (T: number) => 2 * HEADS * T * T * BYTES;
/** Eager HBM traffic: write S, read S, write P, read P. */
const scoreTraffic = (T: number) => 4 * HEADS * T * T * BYTES;
/** On-chip tile: 128 queries x 128 keys of scores, for one head. */
const TILE = 128;
const tileBytes = TILE * TILE * BYTES;

const T_MIN = 512;
const T_MAX = 32768;

const STEPS = [
  'Q, K and V sit in HBM',
  'One matmul writes S',
  'Softmax reads S, writes P',
  'The last matmul reads P',
  'Count what that costs',
  'The intermediate dwarfs the cache',
  'Never write the tile',
  'Carry m, ℓ and O between tiles',
];

const CAPTIONS = [
  'Attention starts from three modest tensors. Q, K and V for a layer are about 1 MB per thousand tokens of context. None of this is the problem.',
  'Multiply Q by K transpose and the score matrix S appears. It holds one number per (query, key) pair, so it has T times as many numbers as Q. This is where the footprint explodes.',
  'Softmax normalises every row of S into P. In eager PyTorch that means reading all of S from HBM and writing a second tensor of the same size right next to it.',
  'The final matmul multiplies P by V. That is the only thing in the whole layer that ever reads P — after which S and P fall out of scope and are freed.',
  'So every byte of S and P crossed HBM exactly twice: written once, read once. Four transfers of a T-by-T tensor, per head, per layer, every single forward pass.',
  'At 8,192 tokens those two tensors are 8.6 GB for ONE layer of Llama-3.1-8B, against 1.07 GB for the KV cache of all 32 layers. The intermediate is eight times the thing it was computed from.',
  'The fix is not a faster matmul. It is to stop writing the intermediate at all: work on one small tile of S in on-chip SRAM, consume it immediately, and move on.',
  'The obstacle is that softmax is a global reduction — it needs a whole row before it can normalise. The online softmax removes the obstacle by carrying the running maximum, the running sum and the running output between tiles.',
];

export default function AttentionMemory() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [T, setT] = useState(8192);

  const peak = useTween(scorePeak(T), 600);
  const traffic = useTween(scoreTraffic(T), 600);
  const kv = useTween(kvBytes(T), 600);

  const ratio = scorePeak(T) / kvBytes(T);
  const win = ratio * scorePeak(T) / (scorePeak(T) + outBytes(T));

  // The score square grows logarithmically so both ends stay on screen.
  const frac = (Math.log2(T) - Math.log2(T_MIN)) / (Math.log2(T_MAX) - Math.log2(T_MIN));
  const side = 26 + frac * 84;

  const SX = 240;
  const SY = 64;
  const PX = SX + side + 24;

  return (
    <Stage
      rootRef={ref}
      title="Attention's intermediate, and the path that avoids it"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'KV tensors'],
        ['compute', 'work on chip'],
        ['error', 'the intermediate'],
        ['idle', 'never written'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Context T', value: `${fmt.int(T)} tok`, kind: 'text', note: 'one layer, 32 heads' },
          { label: 'Peak S + P', value: fmt.gb(peak), kind: 'error', note: `${ratio.toFixed(0)}x the whole KV cache` },
          { label: 'HBM traffic', value: fmt.gb(traffic), kind: 'error', note: 'written once, read once' },
          { label: 'Streaming peak', value: fmt.gb(outBytes(T) + tileBytes), kind: 'memory', note: 'O plus one tile' },
        ]} />
      }
      controls={
        <Slider
          label="Context length T"
          kind="error"
          min={T_MIN}
          max={T_MAX}
          log
          value={T}
          onChange={setT}
          format={(v) => `${fmt.int(v)} tokens`}
        />
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="The upper panel shows three small tensors Q, K and V in GPU memory producing a large score matrix S and an equally large probability matrix P, which together take many times more memory than the KV cache itself. The lower panel shows the alternative: keys and values stream from GPU memory into a small on-chip buffer where one tile of the score matrix is computed and consumed by a small running state holding the maximum, the sum and the output, and is never written back to GPU memory.">
        {/* ───── panel 1 · the eager path ───────────────────── */}
        <Label x={30} y={26} tone={2} size={11} mono={false} bold>The eager path: three tensor ops, two giant intermediates</Label>
        <rect x={30} y={36} width={580} height={160} rx={4} fill="none" stroke="var(--rule-2)" strokeDasharray="4 4" />
        <Label x={38} y={50} tone={3} size={9.5}>HBM</Label>

        {['Q', 'K', 'V'].map((n, i) => (
          <Span key={n} x={44 + i * 42} y={62} w={38} h={22} kind="memory" label={n} />
        ))}
        <Label x={44} y={98} tone={3} size={9}>{`T x ${HEADS * D}`}</Label>

        <Arrow x1={172} y1={73} x2={SX - 10} y2={SY + side / 2} kind="compute" opacity={s.step >= 1 ? 1 : 0} />
        <Fade show={s.step >= 2}>
          <Arrow x1={SX + side + 3} y1={SY + side / 2} x2={PX - 7} y2={SY + side / 2} kind="compute" />
          <Label x={SX + side + 13} y={SY + side / 2 - 8} anchor="middle" size={9} kind="compute">softmax</Label>
        </Fade>
        <Fade show={s.step >= 3}>
          <Arrow x1={PX + side + 3} y1={SY + side / 2} x2={PX + side + 18} y2={SY + side / 2} kind="compute" />
          <Span x={PX + side + 20} y={SY + side / 2 - 11} w={44} h={22} kind="memory" label="O" />
        </Fade>

        {/* S */}
        <Fade show={s.step >= 1}>
          <rect x={SX} y={SY} width={side} height={side} rx={2} fill="var(--error-dim)" stroke="var(--error)" strokeWidth={1.4} className="v-cell" />
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <line key={i} x1={SX} x2={SX + side} y1={SY + (i * side) / 7} y2={SY + (i * side) / 7} stroke="var(--error)" strokeOpacity={0.2} />
          ))}
          <Label x={SX + side / 2} y={SY - 7} anchor="middle" size={10.5} kind="error" bold>S = QKᵀ</Label>
          <Label x={SX + side / 2} y={SY + side + 14} anchor="middle" size={9.5} kind="error">{fmt.gb(scorePeak(T) / 2)}</Label>
        </Fade>

        {/* P */}
        <Fade show={s.step >= 2}>
          <rect x={PX} y={SY} width={side} height={side} rx={2} fill="var(--error-dim)" stroke="var(--error)" strokeWidth={1.4} className="v-cell" />
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <line key={i} x1={PX} x2={PX + side} y1={SY + (i * side) / 7} y2={SY + (i * side) / 7} stroke="var(--error)" strokeOpacity={0.2} />
          ))}
          <Label x={PX + side / 2} y={SY - 7} anchor="middle" size={10.5} kind="error" bold>P</Label>
          <Label x={PX + side / 2} y={SY + side + 14} anchor="middle" size={9.5} kind="error">{fmt.gb(scorePeak(T) / 2)}</Label>
        </Fade>

        {/* the verdict */}
        <Fade show={s.step >= 4}>
          <Span x={44} y={108} w={142} h={15} kind="error" dim label="written, then read" labelTone="light" />
          <Label x={44} y={140} tone={3} size={9}>4 transfers of T x T</Label>
          <Label x={44} y={158} size={10.5} kind="error" bold>{fmt.gb(traffic)} moved</Label>
          <Label x={44} y={174} size={10.5} kind="memory">KV cache: {fmt.gb(kv)}</Label>
          <Dim x1={44} x2={186} y={190} label={`${ratio.toFixed(0)}x`} kind="error" />
        </Fade>

        {/* ───── panel 2 · the streaming path ────────────────── */}
        <Label x={30} y={230} tone={2} size={11} mono={false} bold>The streaming path: compute one tile, consume it, never write it</Label>

        <Span x={44} y={258} w={54} h={70} kind="memory" label="K, V" labelTone="light" />
        <Label x={44} y={344} anchor="middle" tone={3} size={9}>from HBM</Label>
        <Arrow x1={100} y1={293} x2={128} y2={293} kind="memory" opacity={s.step >= 6 ? 1 : 0} />

        <Fade show={s.step >= 6}>
          <rect x={130} y={248} width={176} height={90} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <Label x={218} y={262} anchor="middle" tone={3} size={9}>on-chip SRAM</Label>
          <Move x={s.step >= 7 ? 62 : 0} y={0}>
            <rect x={150} y={272} width={58} height={44} rx={2} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1.4} className="v-cell" />
            <Label x={179} y={297} anchor="middle" size={10} kind="compute" bold>S tile</Label>
          </Move>
          <Label x={218} y={332} anchor="middle" size={9.5} kind="compute">{`${TILE} x ${TILE} = 32 KB, one head`}</Label>
        </Fade>

        <Fade show={s.step >= 7}>
          <Arrow x1={310} y1={293} x2={342} y2={293} kind="compute" />
          <rect x={344} y={258} width={112} height={70} rx={3} fill="var(--ground-3)" stroke="var(--compute)" strokeWidth={1.2} />
          <Label x={400} y={280} anchor="middle" size={10.5} kind="compute" bold>running state</Label>
          <Label x={400} y={300} anchor="middle" size={12}>m, ℓ, O</Label>
          <Label x={400} y={318} anchor="middle" size={9} tone={3}>three vectors</Label>

          <Label x={472} y={268} size={9.5} kind="idle">S never reaches</Label>
          <Label x={472} y={282} size={9.5} kind="idle">HBM. The peak is</Label>
          <Label x={472} y={296} size={9.5} kind="idle">O plus one tile.</Label>
          <Label x={472} y={322} tone={3} size={9}>{`${((outBytes(T) + tileBytes) / 1e6).toFixed(0)} MB vs ${fmt.gb(scorePeak(T))}`}</Label>
        </Fade>

        <Fade show={s.step >= 7}>
          <Dim x1={130} x2={610} y={366} label={`${win.toFixed(0)}% of the eager peak`} kind="memory" />
        </Fade>
      </svg>
    </Stage>
  );
}