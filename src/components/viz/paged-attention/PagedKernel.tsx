import { useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';

const KV_PER_TOKEN = 131072; // bytes per token across all 32 layers

const STEPS = [
  'Naive attention sees a rectangle',
  'Paged attention sees a list',
  'One gather, then the same math',
  'Uniform blocks, fixed kernel',
];

const CAPTIONS = [
  'Without paging, the keys and values for one request form one dense tensor at one address. The kernel indexes it as a rectangle, which is the fastest thing it could possibly do.',
  'With paging the same data is a list of identically shaped chunks at scattered addresses. A dense index no longer applies.',
  'The fix is one gather: load each chunk by id into on-chip SRAM, where the tensor is contiguous again, then run the identical fixed-shape attention.',
  'Because every chunk has the same shape, the inner loop is still compile-time specialized. Paging costs a copy, not a slow kernel.',
];

const W = 640;

// Two views of the same 5 blocks: scattered in the pool, contiguous in SRAM.
const BLOCK_IDS = [23, 4, 31, 17, 9];
const BLOCK_TOKENS = 16;

const CONTIG_X = 90;
const CONTIG_CW = 88;
const CONTIG_Y = 214;

export default function PagedKernel() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [bs, setBs] = useState(16);

  const contigW = (CONTIG_CW - 8) / bs;
  const scatterX = (i: number) => 96 + (i % 5) * 92;
  const scatterY = (i: number) => 92 + (i % 2) * 34;

  return (
    <Stage
      rootRef={ref}
      title="What changes in the attention kernel"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['memory', 'KV block'], ['compute', 'attention'], ['comm', 'gather'], ['idle', 'not resident']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'KV per decode step', value: `${fmt.int(80 * KV_PER_TOKEN / 1e6)} MB`, kind: 'memory', note: '80 tokens of context, all 32 layers' },
          { label: 'Gather traffic', value: `${fmt.int(80 * KV_PER_TOKEN / 1e6)} MB`, kind: 'comm', note: 'one copy, then the same math' },
          { label: 'As % of H100 bandwidth', value: '~1%', kind: 'text', note: 'at 3.35 TB/s' },
        ]} />
      }
      controls={
        <Segmented
          label="Block size"
          value={bs}
          options={[{ value: 8, label: '8 tokens' }, { value: 16, label: '16 tokens' }, { value: 32, label: '32 tokens' }]}
          onChange={(v) => setBs(v)}
        />
      }
    >
      <svg viewBox={`0 0 ${W} 348`} role="img" aria-label="The same five KV blocks shown scattered in a GPU pool, then gathered into one contiguous buffer in SRAM, and finally read by a normal attention kernel.">
        {/* step 0: the dense case */}
        <Fade show={s.step === 0}>
          <Label x={CONTIG_X} y={40} tone={2} size={10.5} mono={false} bold>One request, no paging</Label>
          <rect x={CONTIG_X} y={52} width={CONTIG_CW * 3 + 12} height={34} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" />
          <text x={CONTIG_X + CONTIG_CW * 1.5 + 6} y={73} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fill: 'var(--memory)' }}>K[80 tokens] — one dense tensor, one address</text>
          <Dim x1={CONTIG_X} x2={CONTIG_X + CONTIG_CW * 3 + 12} y={98} label="10.5 MB, contiguous" kind="memory" />
          <Arrow x1={CONTIG_X + CONTIG_CW * 1.5 + 6} y1={104} x2={CONTIG_X + CONTIG_CW * 1.5 + 6} y2={132} kind="compute" />
          <Label x={CONTIG_X + CONTIG_CW * 1.5 + 6} y={148} anchor="middle" size={10.5} kind="compute">attention — dense index</Label>
          <Label x={CONTIG_X} y={180} tone={3} size={10}>Fastest possible. Requires one 10.5 MB allocation that never moves and never shrinks.</Label>
        </Fade>

        {/* steps 1+: the scattered pool */}
        <Fade show={s.step >= 1}>
          <Label x={96} y={72} tone={2} size={10.5} mono={false} bold>The same 80 tokens, paged</Label>
          {BLOCK_IDS.map((id, i) => {
            const x = 96 + i * 92;
            return (
              <g key={id}>
                <Span x={x} y={86} w={84} h={26} kind="memory" label={`block ${id}`} />
                <Label x={x + 42} y={126} anchor="middle" tone={3} size={9}>{`${bs} tokens`}</Label>
              </g>
            );
          })}
          <Label x={96} y={152} tone={3} size={10}>Five equal blocks at five unrelated addresses. A dense index cannot address this.</Label>
        </Fade>

        {/* step 2: the gather */}
        <Fade show={s.step >= 2}>
          {BLOCK_IDS.map((id, i) => (
            <Arrow key={`a${id}`} x1={96 + i * 92 + 42} y1={116} x2={CONTIG_X + i * CONTIG_CW + CONTIG_CW / 2 - 4} y2={CONTIG_Y} kind="comm" dashed />
          ))}
          <Label x={CONTIG_X + 250} y={188} anchor="start" tone={2} size={10} kind="comm">gather: one copy into SRAM</Label>
        </Fade>

        {/* the contiguous result */}
        <Fade show={s.step >= 2}>
          {BLOCK_IDS.map((id, i) => (
            <Span key={`c${id}`} x={CONTIG_X + i * CONTIG_CW} y={CONTIG_Y} w={CONTIG_CW - 8} h={40} kind="memory" dim label="" />
          ))}
          <Dim x1={CONTIG_X} x2={CONTIG_X + BLOCK_IDS.length * CONTIG_CW - 8} y={CONTIG_Y - 8} label="80 tokens, contiguous again" kind="memory" />
          <Arrow x1={CONTIG_X + 2.5 * CONTIG_CW} y1={CONTIG_Y + 48} x2={CONTIG_X + 2.5 * CONTIG_CW} y2={CONTIG_Y + 74} kind="compute" />
          <Label x={CONTIG_X + 2.5 * CONTIG_CW} y={CONTIG_Y + 92} anchor="middle" size={10.5} kind="compute">attention — identical kernel</Label>
        </Fade>

        {/* step 3: what it costs */}
        <Fade show={s.step >= 3}>
          <Label x={CONTIG_X} y={CONTIG_Y + 122} tone={2} size={10.5} mono={false} bold>Price of paging, per decode step</Label>
          <Span x={CONTIG_X} y={CONTIG_Y + 130} w={CONTIG_CW * 3 + 12} h={16} kind="comm" dim label="10.5 MB gathered (all 32 layers)" labelTone="light" />
          <Label x={CONTIG_X} y={CONTIG_Y + 162} tone={3} size={9.5}>{`= ${(80 * KV_PER_TOKEN / 1e9 / 3.35e12 * 100).toFixed(1)}% of an H100's bandwidth. Paid once per step, then the math is unchanged.`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}