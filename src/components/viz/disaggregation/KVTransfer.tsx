import { useRef, useState } from 'react';
import { Stage, Readout, Slider, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade, Bar, Dim, fmt } from '../lib/svg';
import { handoffBytes, decodeStep, maxBatch, prefillPool, decodePool, seqKvBytes } from './disagg';
import { gpuById } from '../lib/perfModel';

const STEPS = [
  'The payload',
  'Cost on the wire',
  'It grows linearly',
  'Pool sizing: prefill',
  'Pool sizing: decode',
  'Monolithic vs split',
  'Routing it out',
  'The rule',
];
const CAPTIONS = [
  'Everything prefill computed has to reach the decode instance. That payload is the KV cache: two bytes times layers times KV heads times head dimension, per token.',
  'Move those bytes across a link and the time is just bytes divided by bandwidth. On a fast interconnect a 4K prompt costs about a millisecond — a rounding error. On 25 GbE it costs 172 ms.',
  'The payload is linear in context length, which means the transfer cost is too. Long-context traffic is the case where disaggregation stops being free.',
  'Size the prefill pool by FLOPs, because that is what prefill is bound by. Arrivals per second times prompt tokens times seconds per token gives GPU-seconds per second, and you round up.',
  'Size the decode pool by two constraints at once: the throughput it can deliver, and the KV memory it can hold. Whichever binds first sets the count.',
  'Here is the payoff. On the same fleet, prefill competing with decode steals its step time and inflates TPOT for everyone. Split the pools and TPOT returns to its floor.',
  'Routers do not send a request to a random decode instance. They pick one that has room for this request’s KV, which is a bin-packing problem, not a load-average problem.',
  'One rule ties it together: the transfer must be small next to the decode step it feeds, and both pools must be sized by the bottleneck their phase is actually bound by.',
];

const GPU = gpuById('h100');
const CTXS = [1024, 2048, 4096, 8192, 16384, 32768];

export default function KVTransfer() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [ctx, setCtx] = useState(4096);
  const [link, setLink] = useState(400e9);
  const [batch, setBatch] = useState(64);

  const bytes = handoffBytes(ctx);
  const maxB = maxBatch(ctx);
  const dec = decodeStep(Math.min(batch, maxB), ctx);
  const tWire = bytes / link;
  const share = tWire / dec.t;

  // Chart of transfer time vs context, log scale.
  const CX0 = 100, CX1 = 610, CY1 = 214, CY0 = 344;
  const cmax = 1400.0; // ms at 25GbE, 32k ctx
  const cy = (ms: number) => CY0 - Math.min(1, Math.log10(Math.max(0.05, ms) + 0.05) / Math.log10(cmax + 0.05)) * (CY0 - CY1);
  const cx = (c: number) => CX0 + Math.log2(c / 1024) / 5 * (CX1 - CX0);
  const wireAt = (c: number) => handoffBytes(c) / link * 1e3;

  const pp = prefillPool(20, 4096);
  const dp = decodePool(20, 256, 512, 4096);
  const duty = pp.gpuSeconds / dp.count;

  return (
    <Stage
      rootRef={ref}
      title="Llama-3.1-8B: sizing the KV handoff and both pools"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'comm', 'compute', 'token', ['error', 'unaffordable'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'KV payload', value: fmt.bytes(bytes), kind: 'memory', note: `${fmt.int(ctx)} tokens` },
          { label: 'On the wire', value: `${(tWire * 1e3).toFixed(2)} ms`, kind: 'comm', note: `${(link / 1e9).toFixed(1)} GB/s link` },
          { label: 'Share of a decode step', value: `${(share * 100).toFixed(1)}%`, kind: share > 0.15 ? 'error' : 'comm', note: `step is ${(dec.t * 1e3).toFixed(1)} ms` },
          { label: 'Sequences per decode GPU', value: fmt.int(maxB), kind: 'memory', note: `${fmt.bytes(seqKvBytes(ctx))} of KV each` },
        ]} />
      }
      controls={
        <>
          <Slider
            label="Context length — the thing the payload scales with"
            value={ctx}
            min={1024}
            max={32768}
            log
            onChange={setCtx}
            format={(v) => `${fmt.int(v)} tokens · ${fmt.bytes(handoffBytes(v))}`}
            kind="memory"
          />
          <Segmented
            label="Link between the pools"
            value={link}
            options={[
              { value: 900e9, label: 'NVLink' },
              { value: 400e9, label: '8×400G' },
              { value: 12.5e9, label: '100G' },
              { value: 3.125e9, label: '25G' },
            ]}
            onChange={setLink}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A chart and sizing calculator for Llama-3.1-8B disaggregation. The upper panel shows KV transfer time against context length for four link speeds, with a horizontal line marking the decode step time the transfer must stay under. The lower panel computes prefill and decode pool sizes for a stated arrival rate and shows the TPOT penalty if both phases shared one pool.">
        {/* ── step 1-3: the payload and the wire ── */}
        <Fade show={s.step <= 2}>
          <Label x={CX0} y={CY1 - 18} tone={2} size={11} mono={false} bold>Transfer time vs context length</Label>
          <line x1={CX0} x2={CX1} y1={CY0} y2={CY0} className="v-rule" />
          <line x1={CX0} x2={CX0} y1={CY1 - 8} y2={CY0} className="v-rule" />
          {[0.1, 1, 10].map((v) => (
            <Label key={v} x={CX0 - 6} y={cy(v) + 3} anchor="end" tone={3} size={9}>{`${v < 1 ? v : v} ms`}</Label>
          ))}
          <Label x={CX0 - 6} y={CY1 - 24} anchor="end" tone={3} size={9}>ms</Label>
          {CTXS.map((c) => (
            <Label key={c} x={cx(c)} y={CY0 + 13} anchor="middle" tone={3} size={9}>{c >= 1024 ? `${c / 1024}k` : c}</Label>
          ))}
          <Label x={CX1} y={CY0 + 27} anchor="end" tone={3} size={9}>context (tokens) →</Label>

          {/* the budget line: the decode step */}
          <line x1={CX0} x2={CX1} y1={cy(dec.t * 1e3)} y2={cy(dec.t * 1e3)} stroke="var(--token)" strokeWidth={1.3} strokeDasharray="5 3" />
          <Label x={CX1} y={cy(dec.t * 1e3) - 5} anchor="end" size={9.5} kind="token">{`decode step = ${(dec.t * 1e3).toFixed(1)} ms`}</Label>

          {/* the 10% budget band */}
          <rect x={CX0} y={cy(dec.t * 1e3)} width={CX1 - CX0} height={cy(dec.t * 1e3 * 0.1) - cy(dec.t * 1e3)} fill="var(--token)" opacity={0.12} />

          {/* one curve per link */}
          {[
            { bw: 900e9, label: 'NVLink', kind: 'comm' as const },
            { bw: 400e9, label: '400G', kind: 'comm' as const },
            { bw: 100e9, label: '100G', kind: 'comm' as const },
            { bw: 25e9, label: '25G', kind: 'error' as const },
          ].map((l) => {
            const pts = CTXS.map((c) => `${cx(c).toFixed(1)},${cy(wireAt(c)).toFixed(1)}`).join(' ');
            return (
              <g key={l.bw}>
                <polyline points={pts} fill="none" stroke={`var(--${l.kind === 'error' ? 'error' : 'comm'})`} strokeWidth={l.bw === link ? 2.4 : 1.2} opacity={l.bw === link ? 1 : 0.4} />
                {l.bw === link && <circle cx={cx(ctx)} cy={cy(wireAt(ctx))} r={4.5} fill={`var(--${l.kind === 'error' ? 'error' : 'comm'})`} stroke="var(--ground-2)" strokeWidth={2} />}
              </g>
            );
          })}
          {/* legend for the curves */}
          <g>
            {[['NVLink', 'comm'], ['400G', 'comm'], ['100G', 'comm'], ['25G', 'error']].map(([t, k], i) => (
              <g key={t}>
                <rect x={CX0 + 8 + i * 62} y={CY1 - 14} width={10} height={3} fill={`var(--${k})`} rx={1.5} />
                <Label x={CX0 + 22 + i * 62} y={CY1 - 9} size={9} tone={3}>{t}</Label>
              </g>
            ))}
          </g>

          <Label x={CX0} y={CY0 + 44} tone={3} size={9.5}>
            {`At ${fmt.int(ctx)} tokens: ${fmt.bytes(bytes)} moves in ${(tWire * 1e3).toFixed(2)} ms — ${(share * 100).toFixed(1)}% of one decode step.`}
          </Label>
        </Fade>

        {/* ── step 4-6: pool sizing ── */}
        <Fade show={s.step >= 3}>
          <Label x={CX0} y={CY1 - 18} tone={2} size={11} mono={false} bold>Sizing both pools for 20 req/s, 4K prompts, 256-token answers</Label>

          {/* prefill pool */}
          <rect x={CX0} y={CY1 + 2} width={240} height={126} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <Label x={CX0 + 12} y={CY1 + 22} size={11} kind="compute" bold>Prefill pool — sized by FLOPs</Label>
          <Label x={CX0 + 12} y={CY1 + 40} tone={3} size={9.5} mono={false}>{'20 req/s × 4,096 tok × 29.5 µs'}</Label>
          <Label x={CX0 + 12} y={CY1 + 56} tone={1} size={10}>{`= ${pp.gpuSeconds.toFixed(2)} GPU-s per second`}</Label>
          <Label x={CX0 + 12} y={CY1 + 76} size={13} kind="compute" bold>{`${pp.count} GPU${pp.count > 1 ? 's' : ''}`}</Label>
          <Label x={CX0 + 12} y={CY1 + 96} tone={3} size={9.5} mono={false}>Holds almost no KV — it ships it</Label>
          <Label x={CX0 + 12} y={CY1 + 112} tone={3} size={9.5} mono={false}>burst-tolerant, sized for p99</Label>

          {/* decode pool */}
          <rect x={CX0 + 260} y={CY1 + 2} width={250} height={126} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" />
          <Label x={CX0 + 272} y={CY1 + 22} size={11} kind="memory" bold>Decode pool — sized twice</Label>
          <Label x={CX0 + 272} y={CY1 + 40} tone={3} size={9.5} mono={false}>{`by throughput: ${dp.demand} ÷ ${dp.perGpu.toFixed(0)} = ${dp.byThroughput}`}</Label>
          <Label x={CX0 + 272} y={CY1 + 56} tone={3} size={9.5} mono={false}>{`by KV memory: 512 ÷ ${dp.slots} = ${dp.byMemory}`}</Label>
          <Label x={CX0 + 272} y={CY1 + 76} size={13} kind="memory" bold>{`${dp.count} GPU${dp.count > 1 ? 's' : ''} (max of the two)`}</Label>
          <Label x={CX0 + 272} y={CY1 + 96} tone={3} size={9.5} mono={false}>{`${dp.perGpu.toFixed(0)} tok/s each, ${dp.slots} seqs`}</Label>
          <Label x={CX0 + 272} y={CY1 + 112} tone={3} size={9.5} mono={false}>holds every KV cache, drains slowly</Label>

          {/* the monolithic comparison */}
          {s.step >= 5 && (
            <g>
              <Dim x1={CX0} x2={CX0 + 240} y={CY0 + 16} label={`${(Math.min(1, duty) * 100).toFixed(0)}% prefill duty on a shared pool`} kind="error" />
              <Label x={CX0} y={CY0 + 32} tone={3} size={9.5}>
                {`Shared: prefill takes ${(duty * 100).toFixed(0)}% of every GPU, inflating TPOT ${(1 / (1 - duty)).toFixed(2)}×. Split: TPOT holds at ${(dec.t * 1e3).toFixed(1)} ms.`}
              </Label>
            </g>
          )}
        </Fade>

        {/* ── step 7: routing ── */}
        <Fade show={s.step >= 6}>
          <g>
            <Label x={CX0} y={CY1 - 18} tone={2} size={11} mono={false} bold>Routing is bin-packing, not load balancing</Label>
            {[0, 1, 2].map((i) => {
              const slots = [11, 7, 3][i];
              const fill = [9, 5, 3][i];
              const x = CX0 + i * 172;
              return (
                <g key={i}>
                  <rect x={x} y={CY1 + 6} width={150} height={108} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" />
                  <Label x={x + 10} y={CY1 + 24} size={10.5} kind="memory" bold>{`decode ${i + 1}`}</Label>
                  {Array.from({ length: slots }, (_, k) => (
                    <rect key={k} x={x + 10 + (k % 6) * 22} y={CY1 + 34 + Math.floor(k / 6) * 22} width={18} height={18} rx={2}
                      fill={k < fill ? 'var(--memory)' : 'none'} stroke="var(--memory)" strokeWidth={1} opacity={k < fill ? 0.85 : 0.5} />
                  ))}
                  <Label x={x + 10} y={CY1 + 96} tone={3} size={9}>{`${fill} of ${slots} slots used`}</Label>
                </g>
              );
            })}
            <Label x={CX0} y={CY1 + 132} tone={3} size={9.5}>
              The router must place a 4K request on an instance with 4K of room. Least-loaded is wrong; least-loaded can still be full.
            </Label>
          </g>
        </Fade>

        {/* ── step 8: the rule ── */}
        <Fade show={s.step >= 7}>
          <g>
            <rect x={CX0} y={CY1 + 4} width={500} height={126} rx={5} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x={CX0 + 14} y={CY1 + 26} size={11} kind="token" bold>The two rules, in one box</Label>
            <Label x={CX0 + 14} y={CY1 + 50} tone={2} size={10.5} mono={false}>
              1 · Transfer time = KV bytes ÷ link bandwidth, and must stay under ~10% of a decode step.
            </Label>
            <Label x={CX0 + 14} y={CY1 + 70} tone={3} size={9.5}>
              {`here: ${fmt.bytes(bytes)} ÷ ${(link / 1e9).toFixed(0)} GB/s = ${(tWire * 1e3).toFixed(2)} ms vs a ${(dec.t * 1e3).toFixed(1)} ms step`}
            </Label>
            <Label x={CX0 + 14} y={CY1 + 94} tone={2} size={10.5} mono={false}>
              2 · Size each pool by its own bottleneck: FLOPs for prefill, throughput and KV for decode.
            </Label>
            <Label x={CX0 + 14} y={CY1 + 114} tone={3} size={9.5}>
              {`here: ${pp.count} prefill + ${dp.count} decode = ${pp.count + dp.count} GPUs, versus ${dp.count} shared`}
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
