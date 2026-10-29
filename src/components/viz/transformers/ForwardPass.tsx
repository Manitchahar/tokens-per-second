import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, fmt } from '../lib/svg';
import { arch, perLayerParams, headParams, embedParams, normParams } from './modelShapes';
import { gpuById } from '../lib/perfModel';

const M = arch('llama8b');
const GB = 1e9;
const TFLOP = 1e12;

const LAYER_P = perLayerParams(M);
const BLOCK_W = M.layers * LAYER_P * 2;      // bytes of the 32 blocks
const HEAD_W = headParams(M) * 2;            // bytes of the output head
const GLUE_W = (2 * M.layers + 1) * normParams(M) * 2 + embedParams(M) / M.vocab * 2; // norms + one embedding row
const KV_PER_TOK = 2 * M.layers * M.kvHeads * M.headDim * 2;

const decodeBytes = (ctx: number) => BLOCK_W + HEAD_W + GLUE_W + ctx * KV_PER_TOK;
const prefillFlops = (m: number) => ({
  block: 2 * m * M.layers * LAYER_P,
  head: 2 * m * headParams(M),
  attn: 2 * m * m * M.hidden * M.layers,
});

// Llama-3.1-8B on an H100: a memory floor at 80% of bandwidth, a compute roof at 55% of peak.
const H100 = gpuById('h100');
const MBU = 0.8;
const MFU = 0.55;

const STEPS = [
  'One token, one vector',
  'Through one block',
  'Through thirty-two',
  'The output head',
  'And then sampling',
  'Decode: read 15 GB, do 15 GFLOP',
  'Prefill: the same stack, many rows',
  'Why prefill is ~200× cheaper',
];
const CAPTIONS = [
  'An integer goes in and a 4,096-number vector comes out — one row of the embedding table, 8 KB read. Everything else in this picture is a function of that vector.',
  'One block: two RMSNorms, attention, the MLP, two residual adds. 436.2 M parameters, 436.2 MB of weights, and 2 FLOPs per parameter.',
  'Thirty-two identical blocks, so 6.98 billion parameters and 13.96 GB of weights. This is the part that makes decode slow: the whole thing is re-read for every token.',
  'One final RMSNorm, then the output head: a 4,096 × 128,256 matrix multiply. That is 525 million more weights — 1.05 GB, another 6.8% of the step.',
  'The logits are 128,256 numbers, one score per token in the vocabulary. Sampling turns them into one token. That is Module 03.9.',
  'One generated token costs about 15 GFLOP of math and 15.5 GB of bytes. On an H100 that is 16 microseconds of math against 5.8 milliseconds of memory — memory-bound by 360×.',
  'Prefill runs the identical graph with many rows at once. The weights are read once and reused m times, so bytes stay flat while FLOPs grow linearly with m.',
  'Prefilling 512 tokens costs about 14 ms. Generating those same 512 tokens one at a time costs 512 × 5.8 ms ≈ 3.0 seconds. Same model, same arithmetic, 208× the cost. That gap is Module 05.',
];

const FLOW = [
  { x: 40, w: 84, n: 'embed', up: 'token ID', down: '8 KB read' },
  { x: 146, w: 128, n: '32 × block', up: '4,096 numbers', down: '13.96 GB read' },
  { x: 296, w: 72, n: 'norm', up: '4,096 numbers', down: '0.3 MB read' },
  { x: 390, w: 104, n: 'lm_head', up: '128,256 logits', down: '1.05 GB read' },
  { x: 516, w: 100, n: 'sample', up: 'next token', down: 'the answer' },
];

const segW = (v: number, total: number, x0: number, w: number) => (v / total) * w;

export default function ForwardPass() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [ctx, setCtx] = useState(4096);
  const [m, setM] = useState(512);

  const bytes = decodeBytes(ctx);
  const f = prefillFlops(m);
  const flops = f.block + f.head + f.attn;
  const prefillBytes = M.layers * LAYER_P * 2 + headParams(M) * 2 + 2 * m * M.hidden * 2;
  const intensity = flops / prefillBytes;
  const tDecode = bytes / (H100.bw * MBU);
  const tPrefill = flops / (H100.flops * MFU);
  const tGenerate = 512 * tDecode;
  const speedup = tGenerate / tPrefill;

  const bytesOn = st >= 5;
  const flopsOn = st >= 6;

  // step → which flow box is lit
  const active = st === 0 ? 0 : st <= 2 ? 1 : st === 3 ? 3 : st === 4 ? 4 : -1;

  const bsegs = [
    { n: '32 blocks of weights', v: BLOCK_W },
    { n: 'output head', v: HEAD_W },
    { n: 'KV cache', v: ctx * KV_PER_TOK },
    { n: 'norms + embedding row', v: GLUE_W },
  ];
  const fsegs = [
    { n: '32 blocks of weights', v: f.block },
    { n: 'output head', v: f.head },
    { n: 'attention scores', v: f.attn },
  ];

  return (
    <Stage
      rootRef={ref}
      title="One token’s whole journey, and what it costs"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', 'token']}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'decode step', value: bytesOn ? `${fmt.gb(bytes)}` : '—', kind: 'memory' as const, note: bytesOn ? `at ${fmt.int(ctx)} tokens of context` : 'not counted yet' },
            { label: 'decode floor, H100', value: bytesOn ? `${(tDecode * 1000).toFixed(1)} ms` : '—', kind: 'memory' as const, note: 'bytes ÷ 80% of bandwidth' },
            { label: 'prefill', value: flopsOn ? `${(flops / TFLOP).toFixed(flops / TFLOP > 50 ? 0 : 1)} TFLOP` : '—', kind: 'compute' as const, note: flopsOn ? `${fmt.int(m)} tokens at once` : 'not counted yet' },
            { label: 'prefill floor, H100', value: flopsOn ? fmt.ms(tPrefill) : '—', kind: 'compute' as const, note: 'FLOPs ÷ 55% of peak' },
          ]}
        />
      }
      controls={
        st >= 5 ? (
          <>
            <Slider label="Context length when decoding" value={ctx} min={512} max={32768} log onChange={setCtx} format={(v) => `${fmt.int(v)} tok`} kind="memory" />
            {st >= 6 && <Segmented label="Prefill length" value={m} options={[{ value: 512, label: '512' }, { value: 8192, label: '8,192' }, { value: 32768, label: '32,768' }]} onChange={setM} />}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="The forward pass as a flow from token ID through 32 blocks and the output head to a sampled token, with a bar of where a decode step's bytes go and a bar of where a prefill's FLOPs go.">
        {/* ── the flow ── */}
        {FLOW.map((b, i) => (
          <g key={b.n}>
            {i > 0 && <Arrow x1={FLOW[i - 1].x + FLOW[i - 1].w} y1={54} x2={b.x - 2} y2={54} kind="text" width={1.1} />}
            <Fade show={active === i}>
              <rect x={b.x - 3} y={38} width={b.w + 6} height={30} rx={4} fill="none" stroke="var(--token)" strokeWidth={1.4} className="v-cell" />
            </Fade>
            <text x={b.x + b.w / 2} y={30} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{b.up}</text>
            <rect x={b.x} y={40} width={b.w} height={26} rx={3}
              fill={i === 4 ? 'var(--token-dim)' : i === 1 ? 'var(--compute-dim)' : 'var(--ground-3)'}
              stroke={i === 4 ? 'var(--token)' : i === 1 ? 'var(--compute)' : 'var(--rule-2)'} strokeWidth={1} className="v-cell" />
            <text x={b.x + b.w / 2} y={57} textAnchor="middle" className="v-label"
              style={{ fontSize: 10, fontWeight: 600, fill: i === 1 ? 'var(--text)' : 'var(--text-2)' }}>{b.n}</text>
            <Fade show={st >= 1}>
              <text x={b.x + b.w / 2} y={84} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{b.down}</text>
            </Fade>
          </g>
        ))}

        <line x1={24} y1={100} x2={616} y2={100} className="v-rule faint" />

        {/* ── where a decode step's bytes go ── */}
        <Fade show={bytesOn}>
          <g>
            <Label x={24} y={120} tone={1} size={10} bold>{`one decode step reads ${fmt.gb(bytes)} — and 90% of it is the same weights every time`}</Label>
            <rect x={24} y={128} width={592} height={26} rx={2} fill="none" stroke="var(--rule)" />
            {bsegs.slice(0, 3).map((sg, i) => {
              let x0 = 24;
              for (let j = 0; j < i; j++) x0 += segW(bsegs[j].v, bytes, 24, 592);
              const w = segW(sg.v, bytes, 24, 592);
              return <rect key={sg.n} x={x0} y={128} width={Math.max(1, w)} height={26} rx={2} fill="var(--memory)" className="v-cell" />;
            })}
            <text x={24 + 20} y={145} className="v-label" style={{ fontSize: 9.5, fontWeight: 700, fill: '#0e1116' }}>{`${((BLOCK_W / bytes) * 100).toFixed(0)}% blocks`}</text>
            <text x={616} y={170} textAnchor="end" className="v-label t3" style={{ fontSize: 9 }}>{`${fmt.gb(BLOCK_W)} weights · ${fmt.gb(HEAD_W)} head · ${fmt.gb(ctx * KV_PER_TOK)} KV`}</text>
            <Label x={24} y={170} tone={3} size={9}>{`bytes ÷ (3.35 TB/s × 0.8) = ${(tDecode * 1000).toFixed(1)} ms → ${(1 / tDecode).toFixed(0)} tok/s`}</Label>
          </g>
        </Fade>

        {/* ── where a prefill's FLOPs go ── */}
        <Fade show={flopsOn}>
          <g>
            <Label x={24} y={210} tone={1} size={10} bold>{`a prefill of ${fmt.int(m)} tokens does ${(flops / TFLOP).toFixed(flops / TFLOP > 50 ? 0 : 1)} TFLOP`}</Label>
            <rect x={24} y={218} width={592} height={26} rx={2} fill="none" stroke="var(--rule)" />
            {fsegs.map((sg, i) => {
              let x0 = 24;
              for (let j = 0; j < i; j++) x0 += segW(fsegs[j].v, flops, 24, 592);
              const w = segW(sg.v, flops, 24, 592);
              return <rect key={sg.n} x={x0} y={218} width={Math.max(1, w)} height={26} rx={2}
                fill={i === 2 ? 'var(--compute)' : 'var(--compute-dim)'} stroke="var(--compute)" strokeWidth={0.8} className="v-cell" />;
            })}
            <text x={24 + 20} y={235} className="v-label" style={{ fontSize: 9.5, fontWeight: 700, fill: '#0e1116' }}>{`${((f.block / flops) * 100).toFixed(0)}% blocks`}</text>
            <text x={616} y={260} textAnchor="end" className="v-label t3" style={{ fontSize: 9 }}>{`${(f.block / TFLOP).toFixed(1)} · ${(f.head / TFLOP).toFixed(2)} · ${(f.attn / TFLOP).toFixed(2)} TFLOP`}</text>
            <Label x={24} y={260} tone={3} size={9}>{`attention: ${((f.attn / flops) * 100).toFixed(1)}% at ${fmt.int(m)} — it overtakes the weights near 53,000`}</Label>
          </g>
        </Fade>

        {/* ── the comparison ── */}
        <Fade show={st === 7}>
          <g>
            <Label x={24} y={290} tone={1} size={10.5} bold>the same 512 tokens, two ways</Label>
            <rect x={24} y={300} width={430} height={22} rx={2} fill="var(--memory)" stroke="var(--memory)" strokeWidth={0.8} className="v-cell" />
            <text x={32} y={315} className="v-label" style={{ fontSize: 9.5, fontWeight: 700, fill: '#0e1116' }}>{`generate one at a time: 512 × ${(tDecode * 1000).toFixed(1)} ms`}</text>
            <text x={462} y={315} className="v-label t1" style={{ fontSize: 9.5 }}>{fmt.ms(tGenerate)}</text>
            <rect x={24} y={330} width={Math.max(2, (430 * tPrefill) / tGenerate)} height={22} rx={2} fill="var(--compute)" stroke="var(--compute)" strokeWidth={0.8} className="v-cell" />
            <text x={32} y={345} className="v-label" style={{ fontSize: 9.5, fontWeight: 700, fill: 430 * tPrefill / tGenerate > 240 ? '#0e1116' : 'var(--text)' }}>{`prefill all 512 at once`}</text>
            <text x={462} y={345} className="v-label t1" style={{ fontSize: 9.5 }}>{`${fmt.ms(tPrefill)} · ${speedup.toFixed(0)}× less time`}</text>
            <Label x={24} y={370} tone={3} size={9}>{'Same weights, same arithmetic. The only difference is how many rows reuse each byte.'}</Label>
          </g>
        </Fade>

        <Fade show={st >= 5 && st < 7}>
          <g>
            {st === 5 ? (
              <>
                <Label x={24} y={290} tone={2} size={9.5}>{'Every byte of the weights is read again on the next token. Nothing'}</Label>
                <Label x={24} y={306} tone={2} size={9.5}>{'is remembered from the previous step, so this cost repeats for every'}</Label>
                <Label x={24} y={322} tone={2} size={9.5}>{'generated token. That is exactly the memory-bound regime from Module 02.'}</Label>
                <Label x={24} y={348} tone={1} size={9.5}>{'15.5 GB of bytes against 15.0 GFLOP of math: a ratio of 1.0 FLOP/byte,'}</Label>
                <Label x={24} y={364} tone={1} size={9.5}>{'versus the H100 ridge point of 295. Decode is memory-bound by ~300×.'}</Label>
              </>
            ) : (
              <>
                <Label x={24} y={290} tone={2} size={9.5}>{'Prefill is the same graph with m rows. The weight bytes are read once'}</Label>
                <Label x={24} y={306} tone={2} size={9.5}>{'and reused m times, so arithmetic intensity grows roughly linearly in m:'}</Label>
                <Label x={24} y={322} tone={2} size={9.5}>{`about 1 FLOP/byte at m = 1 and ${intensity.toFixed(0)} FLOP/byte here.`}</Label>
                <Label x={24} y={348} tone={1} size={9.5}>{`m = ${fmt.int(m)} sits ${intensity > 200 ? 'above' : 'below'} the H100 ridge point of 295 FLOP/byte.`}</Label>
                <Label x={24} y={364} tone={3} size={9}>{'Switch to 32,768 tokens and watch attention take over the FLOP bar.'}</Label>
              </>
            )}
          </g>
        </Fade>

        <Fade show={st >= 1 && st < 5}>
          <g>
            {st === 1 && (
              <>
                <Label x={24} y={130} tone={1} size={10.5} bold>one block, 218.1 M parameters</Label>
                <Label x={24} y={156} tone={2} size={9.5}>{'RMSNorm → attention (03.5, 03.6) → add → RMSNorm → SwiGLU → add'}</Label>
                <Label x={24} y={182} tone={2} size={9.5}>{'attention matrices  41.94 M params   83.9 MB'}</Label>
                <Label x={24} y={202} tone={2} size={9.5}>{'MLP matrices     176.16 M params  352.3 MB'}</Label>
                <Label x={24} y={222} tone={2} size={9.5}>{'2 × RMSNorm gain     0.008 M params    0.03 MB'}</Label>
                <Label x={24} y={250} tone={1} size={9.5}>{'436.2 M FLOPs per token, and 436.2 MB to read. The MLP is 80.8% of both.'}</Label>
              </>
            )}
            {st === 2 && (
              <>
                <Label x={24} y={130} tone={1} size={10.5} bold>thirty-two identical blocks</Label>
                <Label x={24} y={156} tone={2} size={9.5}>{`6.98 B parameters · ${fmt.gb(BLOCK_W)} of weights · 2 FLOPs per parameter`}</Label>
                <Label x={24} y={182} tone={2} size={9.5}>{`For Llama-3.1-8B that is 86.9% of the model. Add the embedding`}</Label>
                <Label x={24} y={202} tone={2} size={9.5}>{'table and the output head and you have the 16.06 GB of BF16'}</Label>
                <Label x={24} y={222} tone={2} size={9.5}>{'weights from 03.1.'}</Label>
                <Label x={24} y={250} tone={1} size={9.5}>{'All of it re-read for every generated token. That is the cost.'}</Label>
              </>
            )}
            {st === 3 && (
              <>
                <Label x={24} y={130} tone={1} size={10.5} bold>one last norm, then the output head</Label>
                <Label x={24} y={156} tone={2} size={9.5}>{'final RMSNorm, then a [4,096 × 128,256] matrix multiply:'}</Label>
                <Label x={24} y={182} tone={2} size={9.5}>{'525.34 M parameters · 1.05 GB · 1.05 GFLOP per token'}</Label>
                <Label x={24} y={208} tone={2} size={9.5}>{'Output shape: [batch, 128,256] — one score for every token in the'}</Label>
                <Label x={24} y={228} tone={2} size={9.5}>{'vocabulary, whether or not it appeared in your prompt.'}</Label>
                <Label x={24} y={256} tone={1} size={9.5}>{'6.8% of the step. Small, and free to skip if you know the answer is constrained.'}</Label>
              </>
            )}
            {st === 4 && (
              <>
                <Label x={24} y={130} tone={1} size={10.5} bold>from 128,256 scores to one token</Label>
                <Label x={24} y={156} tone={2} size={9.5}>{'That step is sampling: temperature, top-k, top-p, and a draw.'}</Label>
                <Label x={24} y={182} tone={2} size={9.5}>{'It touches no weights and does no GEMMs — it is pure arithmetic'}</Label>
                <Label x={24} y={202} tone={2} size={9.5}>{'over 128,256 numbers, about 0.5 MFLOP. On a GPU that is'}</Label>
                <Label x={24} y={222} tone={2} size={9.5}>{'a couple of microseconds, entirely hidden behind the next layer.'}</Label>
                <Label x={24} y={250} tone={1} size={9.5}>{'Module 03.9 does it properly. For now: the forward pass is done.'}</Label>
              </>
            )}
          </g>
        </Fade>

        {st === 0 && (
          <g>
            <Label x={24} y={130} tone={1} size={11} bold>this is the whole model</Label>
            <Label x={24} y={156} tone={2} size={9.5}>{'Everything else in this course is about making these five boxes'}</Label>
            <Label x={24} y={176} tone={2} size={9.5}>{'faster, or fitting more of them on one GPU. Nothing new will be'}</Label>
            <Label x={24} y={196} tone={2} size={9.5}>{'added to the graph after this lesson.'}</Label>
            <Label x={24} y={226} tone={3} size={9}>{'Prefill and decode run the identical graph. They differ only in how'}</Label>
            <Label x={24} y={246} tone={3} size={9}>{'many rows go through at once — which is why they hit different limits.'}</Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
