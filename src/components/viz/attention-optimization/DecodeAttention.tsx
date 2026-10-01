import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Arrow, Dim, Fade, Move, fmt } from '../lib/svg';

// Llama-3.1-8B, BF16, one layer.
const NQ = 32; // query heads
const NKV = 8; // KV heads (GQA: 4 query heads per KV head)
const D = 128;
const BYTES = 2;
const GROUP = NQ / NKV;

const kvBytesPerToken = 2 * NKV * D * BYTES; // 4,096 bytes per token per layer
const flopsPerToken = 4 * NQ * D; // QKᵀ then PV, for one token of context

const H100_BW = 3.35e12;
const H100_FLOPS = 989e12;

const STEPS = [
  'One query row, not a matrix',
  'It must read the whole KV cache',
  'A dot product fills no tensor core',
  'GQA shares one KV head four ways',
  'Split the KV across the SMs',
  'Combine the partial softmaxes',
  'The blocks are not contiguous',
  'What the roofline actually says',
];

const CAPTIONS = [
  'Prefill has T query rows and is compute-dense. Decode has exactly one: the row for the token being generated. Its query is 32 heads by 128 numbers, 8 KB, and it is the whole of Q for this layer.',
  'That one row must attend to every key already in the cache. At 8,192 tokens it reads 8,192 times 4,096 bytes = 33.6 MB of KV per layer, every step, to produce 8 KB of output.',
  'The score for a single query row is a matrix-vector product. Matrix-vector is the worst shape on a GPU: a tensor core wants a 16x16x16 tile of multiply-adds, and here each output number uses one.',
  'Because this model uses GQA, the 32 query heads read only 8 distinct KV heads, four query heads sharing each one. That is why the KV cache is small — and it sets the arithmetic intensity at exactly 4.',
  'One row of work cannot fill 100+ SMs, so every decode kernel splits the KV sequence into chunks and runs them in parallel. Each chunk reduces its own partial answer. This is FlashDecoding, or split-K.',
  'Each split returns three numbers per query head: its own maximum m, its own sum ℓ, and a partial output. A second tiny kernel merges the partials with the online-softmax rule from lesson one — exactly, not approximately.',
  'Under PagedAttention the KV lives in scattered blocks, so each split follows the block table and gathers as it goes. The gather is one extra copy, about 1% of the step, and it is the price of the memory layout.',
  'Intensity is 4 FLOP per byte — exactly the GQA group factor — and does not change with context length. On an H100 that is 74 times below the ridge point, so decode attention is hard-bound by bandwidth no matter how good the kernel is.',
];

export default function DecodeAttention() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [T, setT] = useState(8192);
  const [splits, setSplits] = useState(4);

  const bytes = useTween(T * kvBytesPerToken, 500);
  const tMem = useTween((T * kvBytesPerToken) / H100_BW, 500);
  const tFlop = useTween((T * flopsPerToken) / H100_FLOPS, 500);

  const blocks = Math.min(8, Math.max(2, Math.round(T / 1024)));
  const BW = 62;
  const GAP = 6;
  const stripW = blocks * (BW + GAP) - GAP;
  const KX = 44;

  return (
    <Stage
      rootRef={ref}
      title="Decode attention: one row against the whole cache"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'KV read'],
        ['compute', 'the query row'],
        ['token', 'output'],
        ['error', 'idle SMs'],
        ['comm', 'the gather'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Context T', value: `${fmt.int(T)} tok`, kind: 'text', note: 'one layer, BF16' },
          { label: 'KV read', value: fmt.gb(bytes), kind: 'memory', note: '4,096 bytes per token' },
          { label: 'At 3.35 TB/s', value: `${(tMem * 1e6).toFixed(0)} µs`, kind: 'memory', note: 'floor: bandwidth' },
          { label: 'Arithmetic intensity', value: `${GROUP} FLOP/byte`, kind: 'compute', note: 'the GQA group factor' },
        ]} />
      }
      controls={
        <>
          <Slider label="Context length T" kind="memory" min={512} max={32768} log value={T} onChange={setT} format={(v) => `${fmt.int(v)} tokens`} />
          <Segmented
            label="Split-K chunks"
            value={splits}
            options={[{ value: 2, label: '2' }, { value: 4, label: '4' }, { value: 8, label: '8' }, { value: 16, label: '16' }]}
            onChange={setSplits}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 386" role="img" aria-label="A single query row of 32 heads attends to a KV cache of thousands of tokens. With one row of work the GPU is mostly idle, so the kernel splits the cache into chunks processed in parallel on separate streaming multiprocessors, each producing a partial maximum, sum and output, which a combine step merges. Blocks are scattered, so a gather is needed, and the arithmetic intensity is fixed at four FLOP per byte.">
        {/* ── the query row and the cache ────────────────────── */}
        <Label x={44} y={26} tone={2} size={11} mono={false} bold>Decode: one token's query, the whole cache</Label>

        <Fade show={s.step >= 0}>
          <Span x={44} y={40} w={112} h={34} kind="compute" label={`Q: ${NQ} x ${D}`} />
          <Label x={44} y={90} tone={3} size={9}>{`one row, ${(NQ * D * BYTES / 1024).toFixed(0)} KB`}</Label>
        </Fade>

        <Arrow x1={160} y1={57} x2={KX - 8} y2={57} kind="memory" opacity={s.step >= 1 ? 1 : 0} />

        <Fade show={s.step >= 1}>
          {Array.from({ length: blocks }, (_, j) => (
            <rect key={j} x={KX + j * (BW + GAP)} y={40} width={BW} height={34} rx={2}
              fill="var(--memory)" fillOpacity={0.8} stroke="var(--memory)" className="v-cell" />
          ))}
          <Label x={KX} y={90} tone={3} size={9}>{`K and V for ${fmt.int(T)} tokens = ${fmt.gb(T * kvBytesPerToken)}`}</Label>
          <Dim x1={KX} x2={KX + stripW} y={104} label={`${fmt.int(T)} tokens`} kind="memory" />
        </Fade>

        {/* ── the shape problem ──────────────────────────────── */}
        <Fade show={s.step >= 2}>
          <Label x={44} y={132} tone={3} size={10}>matrix × vector: each output number costs one dot product of 128 terms</Label>
          <Label x={44} y={150} size={10.5} kind="error" bold>{`a 16x16x16 tensor-core tile is 3,000x more efficient — and unusable here`}</Label>
          {Array.from({ length: 18 }, (_, i) => (
            <rect key={i} x={420 + (i % 18) * 12} y={128} width={9} height={9} rx={1} fill="var(--error-dim)" stroke="var(--error)" strokeWidth={0.6} />
          ))}
          <Label x={420} y={158} tone={3} size={9}>{`1 row of a ${NQ}-head tile: ${(100 * 1 / NQ).toFixed(0)}% occupied`}</Label>
        </Fade>

        {/* ── GQA ────────────────────────────────────────────── */}
        <Fade show={s.step >= 3}>
          <Label x={44} y={182} tone={2} size={11} mono={false} bold>GQA: 32 query heads, 8 KV heads</Label>
          {Array.from({ length: NKV }, (_, g) => {
            const gx = 44 + g * 72;
            return (
              <g key={g}>
                <rect x={gx} y={192} width={64} height={26} rx={2} fill="var(--memory)" fillOpacity={0.75} stroke="var(--memory)" className="v-cell" />
                <Label x={gx + 32} y={209} anchor="middle" size={9.5}>K,V head {g}</Label>
                {[0, 1, 2, 3].map((k) => (
                  <rect key={k} x={gx + k * 16} y={224} width={13} height={14} rx={1.5} fill="var(--compute)" fillOpacity={0.85} stroke="var(--compute)" className="v-cell" />
                ))}
              </g>
            );
          })}
          <Label x={44} y={256} tone={3} size={9}>{`4 query heads share each KV head. Fewer bytes read — and intensity lands at exactly ${GROUP}.`}</Label>
        </Fade>

        {/* ── split-K ────────────────────────────────────────── */}
        <Fade show={s.step >= 4}>
          <Label x={44} y={286} tone={2} size={11} mono={false} bold>{`Split the sequence across SMs: ${splits} chunks, one row each`}</Label>

          {Array.from({ length: splits }, (_, k) => {
            const w = stripW / splits;
            const x = KX + k * w;
            return (
              <g key={k}>
                <rect x={x + 1} y={300} width={w - 2} height={22} rx={2}
                  fill="var(--memory)" fillOpacity={s.step >= 6 ? 0.45 : 0.8}
                  stroke={s.step >= 6 ? 'var(--comm)' : 'var(--memory)'} className="v-cell" />
                <Label x={x + w / 2} y={315} anchor="middle" size={9} kind={s.step >= 6 ? 'comm' : 'memory'}>
                  {s.step >= 6 ? 'gathered' : `split ${k}`}
                </Label>
                <Arrow x1={x + w / 2} y1={324} x2={x + w / 2} y2={336} kind="memory" opacity={0.6} />
                <rect x={x + 1} y={338} width={w - 2} height={16} rx={2} fill="var(--ground-3)" stroke="var(--rule-2)" className="v-cell" />
                <Label x={x + w / 2} y={350} anchor="middle" size={8.5} tone={3}>m, ℓ, O</Label>
              </g>
            );
          })}
          {splits < 16 && (
            <>
              {Array.from({ length: 16 - splits }, (_, k) => (
                <rect key={k} x={44 + stripW + 8 + (k % 4) * 14} y={300 + Math.floor(k / 4) * 14} width={11} height={11} rx={1} fill="var(--error-dim)" stroke="var(--error)" strokeWidth={0.6} />
              ))}
              <Label x={44 + stripW + 8} y={338} tone={3} size={8.5}>{`${16 - splits} idle`}</Label>
            </>
          )}
        </Fade>

        {/* ── the combine ────────────────────────────────────── */}
        <Fade show={s.step >= 5}>
          <Arrow x1={KX + stripW / 2} y1={356} x2={KX + stripW / 2} y2={372} kind="token" />
          <Span x={KX + stripW / 2 - 90} y={374} w={180} h={10} kind="token" dim />
          <Label x={KX + stripW + 70} y={306} anchor="start" tone={3} size={9}>merge partials with</Label>
          <Label x={KX + stripW + 70} y={318} anchor="start" tone={3} size={9}>the online softmax.</Label>
          <Label x={KX + stripW + 70} y={332} anchor="start" size={9.5} kind="token" bold>exact</Label>
        </Fade>

        {/* ── the roofline verdict ───────────────────────────── */}
        <Fade show={s.step >= 7}>
          <Label x={44} y={140} tone={2} size={11} mono={false} bold>What the roofline says</Label>
          <Span x={44} y={150} w={190} h={16} kind="memory" dim label="bandwidth: 4,096 B/token" labelTone="light" />
          <Span x={44} y={170} w={13} h={16} kind="compute" dim />
          <Label x={63} y={182} size={9.5} kind="compute">{`compute: ${fmt.si(flopsPerToken)} FLOP/token → needs ${(tFlop * 1e6).toFixed(1)} µs`}</Label>
          <Label x={44} y={200} size={10} kind="error" bold>{`${(tMem / Math.max(tFlop, 1e-9)).toFixed(0)}x memory-bound — and that ratio never improves with T`}</Label>
          <Label x={44} y={216} tone={3} size={9}>{`intensity = ${GROUP} FLOP/byte, independent of context length`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}