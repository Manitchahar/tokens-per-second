import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Fade, Bar, fmt } from '../lib/svg';
import {
  stepAt,
  serviceTime,
  costPerMTokOut,
  costPerMTokTP,
  prefillAt,
  promptPerMTok,
  tensorParallel,
  gpuById,
  modelById,
  gpuScalars,
  perSecond,
  MBU_FOR_BITS,
  ctxFor,
  type Op,
} from './econ';

// The question this visual exists to settle: given a workload, which card makes a token
// cheapest? The answer is not the fastest card, and it is not the card with the best
// price/spec headline. It is the card that maximises goodput per dollar at the SLO.

const STEPS = [
  'The wrong question: which card is fastest?',
  'Two cards, one job, measured',
  'The metric that decides it',
  'Rank by price per bandwidth',
  'Rank by price per FLOP/s',
  'The rankings disagree — and that is the point',
  'A cheaper card wins if it is proportionally slower',
  'Read the Pareto set, not the winner',
];
const CAPTIONS = [
  'Every generation of hardware is both faster and more expensive than the last, so "fastest" and "cheapest" are usually different cards. Ranking on speed alone picks whichever card has the biggest number next to it.',
  'One fixed job — Llama-3.1-8B, 512 prompt tokens, 256 output tokens, 64 streams — measured on each card. Two columns that people conflate: tokens per second, and what that second costs.',
  'Goodput per dollar divides the useful work by what you paid for it. It is the only column that compares two cards, because it prices the promise you made rather than the speed you achieved.',
  'Decode is memory-bound, so a card that makes tokens cheap is a card with a low price per unit of bandwidth. From the A100 to the B200 that number barely moves — which is a real and under-appreciated finding.',
  'Prefill is compute-bound, so for a prompt-heavy job the ranking is by price per unit of FLOP/s. And here the newer cards pull clearly ahead: the H100 is about half the cost per FLOP of an A100.',
  'The two rankings disagree. The card that wins a decode-heavy product is not necessarily the card that wins a summarization product — and most real traffic is a mixture, so the honest answer depends on the mixture.',
  'The break-even test is simple enough to do in your head: a cheaper card wins if it is proportionally slower than it is cheaper. A100 is 1.6x slower than an H100 and 1.56x cheaper — it is a wash, and it does not fit.',
  'Plot every candidate configuration on cost against speed and keep the ones nothing else beats. In a real decision you get a set, not a winner, and the constraints you already know — memory, tail latency, minimum batch — pick from it.',
];

const PRESET: Record<number, { batch: number; nIn: number; nOut: number; wBytes: number; modelId: string }> = {
  0: { batch: 64, nIn: 512, nOut: 256, wBytes: 2, modelId: 'llama8b' },
  1: { batch: 64, nIn: 512, nOut: 256, wBytes: 2, modelId: 'llama8b' },
  2: { batch: 64, nIn: 512, nOut: 256, wBytes: 2, modelId: 'llama8b' },
  3: { batch: 64, nIn: 2048, nOut: 128, wBytes: 2, modelId: 'llama8b' },
  4: { batch: 64, nIn: 8192, nOut: 64, wBytes: 2, modelId: 'llama8b' },
  5: { batch: 64, nIn: 8192, nOut: 64, wBytes: 2, modelId: 'llama8b' },
  6: { batch: 64, nIn: 512, nOut: 256, wBytes: 2, modelId: 'llama8b' },
  7: { batch: 64, nIn: 512, nOut: 256, wBytes: 2, modelId: 'llama8b' },
};

/** Cards worth comparing at this size: the data-centre ones plus the 24 GB edge card. */
const CARDS = ['l4', 'a100', 'h100', 'h200', 'b200'];

interface Row {
  id: string;
  name: string;
  price: number;
  thr: number; // output tokens/s produced at this batch
  perUser: number;
  cOut: number; // $/1M output tokens, including any extra cards
  cards: number; // cards billed per replica
  fits: boolean;
  maxSt: number;
  costPerGB: number;
  perBw: number;
  perFlops: number;
  bill: number; // $/hour for one replica of this shape
}

function measure(op: Op): Row {
  const st = stepAt(op);
  const tp = tensorParallel(op);
  const g = gpuById(op.gpuId);
  const sc = gpuScalars(g);
  const billed = tp.cards;
  const thr = op.batch / st.t;
  return {
    id: op.gpuId,
    name: g.name,
    price: g.price,
    thr,
    perUser: 1 / st.t,
    cOut: costPerMTokOut(op, 256, 512) * billed * tp.stepPenalty,
    cards: billed,
    fits: st.fits,
    maxSt: 0,
    costPerGB: sc.pricePerGbMem,
    perBw: sc.perBandwidth,
    perFlops: sc.perFlops,
    bill: g.price * billed,
  };
}

export default function GoodputPerDollar() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [batch, setBatch] = useState(64);
  const [nIn, setNIn] = useState(512);
  const [nOut, setNOut] = useState(256);
  const [wBytes, setWBytes] = useState(2);

  useEffect(() => {
    const p = PRESET[s.step];
    if (!p) return;
    setModelId(p.modelId); setBatch(p.batch); setNIn(p.nIn); setNOut(p.nOut); setWBytes(p.wBytes);
  }, [s.step]);

  const rows = useMemo(
    () => CARDS.map((id) => measure({ modelId, gpuId: id, batch, ctx: ctxFor(nIn), wBytes })),
    [modelId, batch, nIn, nOut, wBytes],
  );
  const feasible = rows.filter((r) => r.fits);
  const cheapest = rows.reduce((a, b) => (b.cOut < a.cOut ? b : a), rows[0]);
  const fastest = rows.reduce((a, b) => (b.thr > a.thr ? b : a), rows[0]);
  const bestGpd = feasible.reduce((a, b) => (b.thr / b.bill > a.thr / a.bill ? b : a), feasible[0] ?? rows[0]);

  const maxThr = Math.max(...rows.map((r) => r.thr));
  const maxBill = Math.max(...rows.map((r) => r.bill));
  const maxCost = Math.max(...rows.map((r) => r.cOut));

  const byBw = [...rows].sort((a, b) => a.perBw - b.perBw);
  const byFlops = [...rows].sort((a, b) => a.perFlops - b.perFlops);

  const showSpeed = s.step >= 1;
  const showGpd = s.step >= 2;
  const showBw = s.step >= 3;
  const showFlops = s.step >= 4;
  const showSplit = s.step >= 5;
  const showBreakEven = s.step >= 6;
  const showPareto = s.step >= 7;

  const BW_X = 152, BW_W = 300;
  const ROW_H = 30, ROW_Y = 66;

  return (
    <Stage
      rootRef={ref}
      title="Goodput per dollar: which card makes a token cheapest"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['token', 'goodput: work delivered per dollar'],
        ['memory', 'bandwidth-priced (decode)'],
        ['compute', 'FLOP-priced (prefill)'],
        ['error', 'does not fit'],
        'idle',
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Cheapest per token', value: cheapest.name, kind: 'token', note: `$${cheapest.cOut.toFixed(2)} / 1M output tok` },
          { label: 'Fastest', value: fastest.name, kind: 'compute', note: `${fmt.int(fastest.thr)} tok/s · $${fastest.price.toFixed(2)}/hr` },
          { label: 'Best goodput / $', value: bestGpd.name, kind: bestGpd.cOut === cheapest.cOut ? 'token' : 'memory', note: `${(bestGpd.thr / bestGpd.bill).toFixed(0)} tok/s per $/hr` },
          { label: 'Cards you can afford', value: `${feasible.length} / ${rows.length}`, kind: feasible.length ? 'token' : 'error', note: `${fmt.int(modelById(modelId).params * wBytes / 1e9)} GB of weights` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={[{ value: 'llama8b', label: 'Llama-8B' }, { value: 'qwen32b', label: 'Qwen-32B' }, { value: 'llama70b', label: 'Llama-70B' }]} onChange={setModelId} />
          <Segmented label="Weights" value={wBytes} options={[{ value: 2, label: 'BF16' }, { value: 0.5, label: 'INT4' }]} onChange={setWBytes} />
          <Slider label="Concurrency" kind="memory" value={batch} min={1} max={256} log onChange={setBatch} format={(v) => `${fmt.int(v)} streams`} />
          <Slider label="Prompt tokens" kind="compute" value={nIn} min={128} max={8192} log onChange={setNIn} format={(v) => `${fmt.int(v)}`} />
          <Slider label="Output tokens" kind="token" value={nOut} min={16} max={1024} log onChange={setNOut} format={(v) => `${fmt.int(v)}`} />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="Five GPU cards compared on one fixed job: a throughput bar and a cost bar per card, with the price per unit of bandwidth and per unit of FLOP listed underneath, and the card that maximises goodput per dollar highlighted.">
        <Label x={16} y={20} tone={2} size={11} bold>{`One job: ${fmt.int(nIn)} prompt + ${fmt.int(nOut)} output tokens, ${batch} streams, ${wBytes === 2 ? 'BF16' : 'INT4'}`}</Label>
        <Label x={624} y={20} anchor="end" tone={3} size={9}>{modelById(modelId).name}</Label>

        {/* ── the per-card rows ── */}
        {rows.map((r, i) => {
          const y = ROW_Y + i * ROW_H;
          const pick = bestGpd.id === r.id;
          return (
            <g key={r.id}>
              <Label x={16} y={y + 13} size={10.5} tone={pick ? 1 : 2} bold={pick}>{r.name}</Label>
              <Label x={120} y={y + 13} anchor="end" size={9} tone={3}>{`$${r.price.toFixed(2)}/hr`}</Label>
              {r.cards > 1 && <Label x={120} y={y + 24} anchor="end" size={8.5} kind="error">{`${r.cards} cards`}</Label>}

              {/* throughput */}
              <rect x={BW_X} y={y + 2} width={BW_W} height={11} rx={2} fill="none" stroke="var(--rule)" />
              {!r.fits ? (
                <rect x={BW_X} y={y + 2} width={BW_W} height={11} rx={2} fill="url(#tps-hatch)" stroke="var(--error)" />
              ) : (
                <rect className="v-cell" x={BW_X} y={y + 2} width={Math.max(1, (r.thr / maxThr) * BW_W)} height={11} rx={2}
                  fill="var(--compute)" fillOpacity={0.85} />
              )}
              <Label x={BW_X + BW_W + 8} y={y + 12} size={9.5} tone={r.fits ? 1 : 3}>
                {r.fits ? `${fmt.int(r.thr)} tok/s` : 'OOM'}
              </Label>

              {/* cost per token */}
              {showSpeed && (
                <>
                  <rect x={BW_X} y={y + 16} width={BW_W} height={11} rx={2} fill="none" stroke="var(--rule)" />
                  <rect className="v-cell" x={BW_X} y={y + 16} width={Math.max(1, (r.cOut / maxCost) * BW_W)} height={11} rx={2}
                    fill={r.id === cheapest.id ? 'var(--token)' : 'var(--memory)'} fillOpacity={r.id === cheapest.id ? 0.9 : 0.55} />
                  <Label x={BW_X + BW_W + 8} y={y + 26} size={9.5} kind={r.id === cheapest.id ? 'token' : undefined} tone={1}>
                    {`$${r.cOut.toFixed(2)} / 1M out`}
                  </Label>
                </>
              )}
            </g>
          );
        })}

        {/* ── the two price scalars ── */}
        <line x1={16} x2={624} y1={216} y2={216} className="v-rule faint" />

        <Fade show={showBw}>
          <g>
            <Label x={16} y={236} size={10.5} kind="memory" bold>dollars per (TB/s) — ranks decode</Label>
            {byBw.map((r, i) => (
              <g key={r.id}>
                <Label x={i < 3 ? 330 : 480} y={236 + (i % 3) * 14} size={9} tone={2} anchor="start">
                  {`${r.name.replace(/ \d+ GB/, '').replace(' SXM', '')} $${r.perBw.toFixed(2)}`}
                </Label>
              </g>
            ))}
          </g>
        </Fade>

        <Fade show={showFlops}>
          <g>
            <Label x={16} y={290} size={10.5} kind="compute" bold>dollars per (PFLOP/s) — ranks prefill</Label>
            {byFlops.map((r, i) => (
              <Label key={r.id} x={i < 3 ? 330 : 480} y={290 + (i % 3) * 14} size={9} tone={2} anchor="start">
                {`${r.name.replace(/ \d+ GB/, '').replace(' SXM', '')} $${r.perFlops.toFixed(2)}`}
              </Label>
            ))}
          </g>
        </Fade>

        {/* ── the reading ── */}
        <Fade show={showGpd}>
          <g>
            <Label x={16} y={258} size={9.5} tone={3}>lower is better. From A100 to B200 the bandwidth price falls 14%; the FLOP price falls 53%.</Label>
            <Label x={16} y={312} size={9.5} tone={3}>Same shape of list, very different story — and the newer card wins the compute race by a wide margin.</Label>
          </g>
        </Fade>

        <Fade show={showSplit}>
          <g>
            <Label x={16} y={334} size={10.5} tone={1} bold>The rankings disagree, and your traffic decides which one applies</Label>
            <Label x={16} y={348} size={9} tone={3}>
              {`This request is ${serviceTime({ modelId, gpuId: 'h100', batch, ctx: ctxFor(nIn), wBytes }, nOut, nIn).prefillFrac > 0.5 ? 'prefill' : 'decode'}-priced right now. Move the sliders and it flips.`}
            </Label>
          </g>
        </Fade>

        <Fade show={showBreakEven}>
          <g>
            <rect x={0} y={222} width={640} height={150} fill="var(--ground-2)" opacity={0.95} />
            <Label x={16} y={244} size={10.5} tone={2} bold>the break-even test: cheaper than it is slow?</Label>
            {(() => {
              const ref = feasible.find((r) => r.id === 'h100') ?? rows[0];
              return rows.filter((r) => r.id !== ref.id).slice(0, 3).map((r, i) => {
                const speed = r.perUser > 0 ? ref.perUser / r.perUser : 0; // >1 = r slower
                const price = r.price / ref.price; // <1 = r cheaper
                const wins = speed > 0 && price > 0 && speed < price;
                return (
                  <g key={r.id}>
                    <Label x={16} y={266 + i * 15} size={9.5} tone={1}>{r.name}</Label>
                    <Label x={120} y={266 + i * 15} size={9.5} tone={3}>{`${speed.toFixed(2)}× slower,`}</Label>
                    <Label x={228} y={266 + i * 15} size={9.5} tone={3}>{`${price.toFixed(2)}× the price`}</Label>
                    <Label x={372} y={266 + i * 15} size={9.5} kind={wins ? 'token' : 'error'} bold>
                      {wins ? 'cheaper card wins on every token' : 'too slow for the saving'}
                    </Label>
                  </g>
                );
              });
            })()}
            <Label x={16} y={318} size={9} tone={3}>Compute both numbers per-user, not per batch: the batch makes every card look similar and hides the difference entirely.</Label>
            <Label x={16} y={340} size={10.5} tone={1} bold>Cheapest is not the same as best value — and neither is fastest.</Label>
            <Label x={16} y={358} size={9} tone={3}>{`Right now: fastest is ${fastest.name}, cheapest per token is ${cheapest.name}, best goodput per dollar is ${bestGpd.name}.`}</Label>
          </g>
        </Fade>

        <Fade show={showPareto}>
          <g>
            <rect x={0} y={222} width={640} height={150} fill="var(--ground-2)" opacity={0.95} />
            <Label x={16} y={244} size={10.5} tone={2} bold>the Pareto set: configurations nothing else beats</Label>
            {(() => {
              const ok = rows.filter((r) => r.fits);
              const keep = ok.filter((r) => !ok.some((o) => o.cOut <= r.cOut && o.thr >= r.thr && (o.cOut < r.cOut || o.thr > r.thr)));
              return keep.slice(0, 4).map((r, i) => (
                <g key={r.id}>
                  <Label x={16} y={266 + i * 15} size={9.5} tone={1}>{r.name}{r.cards > 1 ? ` ×${r.cards}` : ''}</Label>
                  <Label x={150} y={266 + i * 15} size={9.5} tone={2}>{`$${r.cOut.toFixed(2)} / 1M out`}</Label>
                  <Label x={286} y={266 + i * 15} size={9.5} tone={2}>{`${fmt.int(r.thr)} tok/s`}</Label>
                  <Label x={390} y={266 + i * 15} size={9.5} tone={2}>{`$${r.bill.toFixed(2)}/hr per replica`}</Label>
                </g>
              ));
            })()}
            <Label x={16} y={330} size={9} tone={3}>A real decision never collapses this to one name. Memory, tail latency and a minimum batch size pick from the set.</Label>
            <Label x={16} y={350} size={10.5} tone={1} bold>Report goodput per dollar, and name the workload it was measured under.</Label>
            <Label x={16} y={366} size={9} tone={3}>A goodput-per-dollar number without its workload is not a result — it is a rumour with a currency symbol on it.</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}