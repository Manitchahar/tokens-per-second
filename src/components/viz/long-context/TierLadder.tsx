import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade, Dim, fmt } from '../lib/svg';
import { llama8b, h100, TIERS, seqKvBytes, recomputePerToken, humanTime } from './kvMath';

const BATCH = 8; // conversations in flight
const RECOMPUTE = recomputePerToken(llama8b, h100); // ~29.5 us for one token of KV

const STEPS = [
  'Four places to put a cache',
  'Bandwidth is the whole story',
  'The cost of one token',
  'The other option: recompute',
  'The crossover',
  'What it means per step',
  'The rule',
];
const CAPTIONS = [
  'The KV cache does not have to live in GPU memory. There are three other places it can go, and each one is reachable over a different wire with very different numbers. This ladder is the whole design space for tiered KV.',
  'Notice that the bars are on a log scale. GPU memory is about 60× faster than host DRAM over PCIe, which is about 8× faster than a good NVMe drive, which is about 6× faster than a remote node on 10 GbE. Eight orders of magnitude separate the ends.',
  'One token of KV for Llama-3.1-8B is 131,072 bytes. Move it from HBM and you spend 39 nanoseconds. Over PCIe, 2.4 microseconds. From an SSD, 19. From another machine, 105 — a tenth of a millisecond to read 128 KB.',
  'But fetching is not the only way to get those keys and values back. You can recompute them: push the token through the model again, which costs 2P FLOPs, about 29.5 microseconds on an H100. And here is the twist — a decode step is memory-bound, so its tensor cores are already sitting idle.',
  'Line up fetch against recompute and the decision becomes a comparison of two numbers. PCIe wins by a mile. NVMe is almost exactly even with recompute. A remote node on 10 GbE loses to recompute by 3.5×, so fetching there is the wrong answer.',
  'Multiply by the context length and the batch and the picture becomes a decision per decode step. At 32,768 tokens and a batch of 8, streaming from NVMe costs 4.9 seconds per step, while recomputing the whole context costs 0.97 seconds of FLOPs you were not otherwise using.',
  'So the rule is not "put the cache as far away as you like." It is: keep it where fetching beats recomputing, which is roughly anything above 4.4 GB/s, and never stream more than you can hide inside one step.',
];

const PRESET: Record<number, { ctx: number; focus: string }> = {
  0: { ctx: 32768, focus: 'none' },
  1: { ctx: 32768, focus: 'bw' },
  2: { ctx: 32768, focus: 'fetch' },
  3: { ctx: 32768, focus: 'recompute' },
  4: { ctx: 32768, focus: 'cross' },
  5: { ctx: 32768, focus: 'step' },
  6: { ctx: 32768, focus: 'none' },
};

// ── panel A geometry: the ladder, log bandwidth
const AX0 = 196;
const AX1 = 616;
const ROW_A = 46;
const ROW_H = 26;
const BW_MIN = 1e9;
const BW_MAX = 1e13;
const bx = (bw: number) => AX0 + (Math.log10(bw) - Math.log10(BW_MIN)) / (Math.log10(BW_MAX) - Math.log10(BW_MIN)) * (AX1 - AX0);

// ── panel B geometry: fetch time per token, log seconds
const TX0 = 72;
const TX1 = 618;
const T_MIN = 1e-8;
const T_MAX = 1e-3;
const tx = (s: number) => TX0 + (Math.log10(s) - Math.log10(T_MIN)) / (Math.log10(T_MAX) - Math.log10(T_MIN)) * (TX1 - TX0);
const TTICKS: { s: number; label: string }[] = [
  { s: 1e-8, label: '10 ns' },
  { s: 1e-7, label: '100 ns' },
  { s: 1e-6, label: '1 µs' },
  { s: 1e-5, label: '10 µs' },
  { s: 1e-4, label: '100 µs' },
  { s: 1e-3, label: '1 ms' },
];

const FETCH = TIERS.map((t) => ({ ...t, perToken: seqKvBytes(1, llama8b) / t.bw }));
const XOVER_BW = seqKvBytes(1, llama8b) / RECOMPUTE; // ~4.4 GB/s

export default function TierLadder() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [ctx, setCtx] = useState(PRESET[s.step].ctx);
  const [focus, setFocus] = useState(PRESET[s.step].focus);
  useEffect(() => { setCtx(PRESET[s.step].ctx); setFocus(PRESET[s.step].focus); }, [s.step]);

  const fetchOn = (id: string) => focus === 'bw' || focus === 'fetch' || focus === 'cross' || focus === 'step';
  const rcOn = focus === 'recompute' || focus === 'cross' || focus === 'step';

  const recTx = tx(RECOMPUTE);
  const markers = useMemo(
    () => FETCH.map((f) => ({ ...f, x: tx(f.perToken), wins: f.perToken < RECOMPUTE })),
    [],
  );

  // per-decode-step comparison at this context length
  const cacheBytes = seqKvBytes(ctx, llama8b) * BATCH;
  const stepRows = FETCH.map((f) => ({
    ...f,
    tFetch: cacheBytes / f.bw,
    tRecomp: RECOMPUTE * ctx * BATCH,
  }));
  const winner = stepRows.slice().sort((a, b) => a.tFetch - b.tFetch)[0];

  const readout = (
    <Readout items={[
      { label: 'One token of KV', value: fmt.bytes(seqKvBytes(1, llama8b)), kind: 'memory', note: 'Llama-3.1-8B, BF16' },
      { label: 'Fetch from GPU HBM', value: humanTime(FETCH[0].perToken), kind: 'memory', note: '3.35 TB/s' },
      { label: 'Fetch from NVMe', value: humanTime(FETCH[2].perToken), kind: 'memory', note: 'about 7 GB/s' },
      {
        label: 'Recompute it instead',
        value: humanTime(RECOMPUTE),
        kind: 'compute',
        note: 'idle tensor cores in a decode step',
      },
      {
        label: `Cheapest at ${fmt.int(ctx)} tokens`,
        value: winner.wins ? `fetch from ${winner.short}` : 'recompute',
        kind: winner.wins ? 'memory' : 'compute',
        note: `batch ${BATCH}: ${winner.wins ? humanTime(winner.tFetch) : humanTime(winner.tRecomp)}`,
      },
    ]} />
  );

  return (
    <Stage
      rootRef={ref}
      title="Where a KV cache can live, and what it costs to move it"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', ['error', 'recompute is cheaper'], 'text']}
      {...s}
      readout={readout}
      controls={
        <Slider
          label="Context length per conversation (tokens) — drives the per-step comparison"
          value={ctx}
          min={1024}
          max={131072}
          log
          onChange={(v) => setCtx(v)}
          format={(v) => `${fmt.int(v)} tokens`}
          kind="memory"
        />
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Two panels. The top panel is a ladder of four storage tiers with their bandwidth on a logarithmic scale: GPU HBM at 3.35 terabytes per second, host DRAM over PCIe at 55 gigabytes, NVMe at 7 gigabytes, and a remote node at 1.25 gigabytes. The bottom panel places each tier's time to fetch one token's KV on a logarithmic time axis from 10 nanoseconds to 1 millisecond, against a vertical amber line at 29.5 microseconds marking the cost of recomputing the same token. GPU memory and PCIe fall to the left of the line and win; the remote node falls to the right and loses.">
        {/* ═════ panel A · the ladder ═════ */}
        <Label x={20} y={20} tone={2} size={10.5} mono={false} bold>The ladder</Label>
        <Label x={616} y={20} anchor="end" tone={3} size={9.5}>sustained bandwidth, log scale</Label>

        {/* the recompute-crossover band, drawn on bandwidth */}
        {fetchOn() && (
          <>
            <rect x={AX0} y={ROW_A - 6} width={bx(XOVER_BW) - AX0} height={TIERS.length * ROW_H + 12} fill="var(--memory-dim)" opacity={0.35} />
            <Label x={bx(XOVER_BW) + 4} y={ROW_A + TIERS.length * ROW_H + 14} size={9} kind="memory">{`fetch beats recompute above ${(XOVER_BW / 1e9).toFixed(1)} GB/s`}</Label>
          </>
        )}

        {TIERS.map((t, i) => {
          const y = ROW_A + i * ROW_H;
          const slow = t.bw < XOVER_BW;
          const on = fetchOn();
          return (
            <g key={t.id} style={{ opacity: on ? 1 : 0.4 }}>
              <Label x={AX0 - 12} y={y + 14} anchor="end" tone={2} size={10}>{t.short}</Label>
              <Span x={AX0} y={y} w={bx(t.bw) - AX0} h={ROW_H - 8} kind="memory" dim={slow && on} opacity={0.9} />
              <Label x={bx(t.bw) + 6} y={y + 13} size={9.5} kind={slow ? 'error' : 'memory'}>
                {t.bw >= 1e12 ? `${(t.bw / 1e12).toFixed(2)} TB/s` : `${(t.bw / 1e9).toFixed(t.bw < 1e10 ? 2 : 0)} GB/s`}
              </Label>
            </g>
          );
        })}
        <Label x={AX0} y={ROW_A + TIERS.length * ROW_H + 14} tone={3} size={9.5}>{`${(BW_MIN / 1e9).toFixed(0)} GB/s`}</Label>
        <Label x={AX1} y={ROW_A + TIERS.length * ROW_H + 14} anchor="end" tone={3} size={9.5}>{`${(BW_MAX / 1e12).toFixed(0)} TB/s`}</Label>

        {/* ═════ panel B · one token ═════ */}
        <Label x={20} y={186} tone={2} size={10.5} mono={false} bold>One token of KV: fetch it, or recompute it?</Label>
        <Label x={618} y={186} anchor="end" tone={3} size={9.5}>131,072 bytes, log time</Label>

        {/* the axis */}
        <line x1={TX0} x2={TX1} y1={252} y2={252} className="v-rule" />
        {TTICKS.map((t) => (
          <g key={t.s}>
            <line x1={tx(t.s)} x2={tx(t.s)} y1={252} y2={257} className="v-rule" />
            <Label x={tx(t.s)} y={270} anchor="middle" tone={3} size={9}>{t.label}</Label>
          </g>
        ))}

        {/* the recompute line */}
        {rcOn && (
          <>
            <line x1={recTx} x2={recTx} y1={196} y2={278} stroke="var(--compute)" strokeWidth={1.4} strokeDasharray="5 3" />
            <Label x={recTx} y={192} anchor="middle" size={9.5} kind="compute">{`recompute = ${humanTime(RECOMPUTE)}`}</Label>
          </>
        )}

        {/* the losing half, shaded */}
        {fetchOn() && (
          <rect x={recTx} y={196} width={TX1 - recTx} height={82} fill="url(#tps-hatch-red)" opacity={0.28} />
        )}

        {/* markers, one per tier, nudged apart so labels never collide */}
        {markers.map((m, i) => {
          const dy = [0, -1, 0, 1][i];
          const ly = 232 + dy * 13;
          const lead = 232;
          return (
            <g key={m.id} style={{ opacity: focus === 'none' ? 0.45 : 1 }}>
              <line x1={m.x} x2={m.x} y1={lead + 14} y2={252} stroke={m.wins ? 'var(--memory)' : 'var(--error)'} strokeWidth={1} opacity={0.5} />
              <circle cx={m.x} cy={ly + 8} r={5} fill={m.wins ? 'var(--memory)' : 'url(#tps-hatch-red)'} stroke={m.wins ? 'var(--memory)' : 'var(--error)'} strokeWidth={1.6} />
              <Label x={m.x} y={ly - 2} anchor="middle" size={9.5} kind={m.wins ? 'memory' : 'error'}>{m.short}</Label>
              <Label x={m.x} y={ly + 22} anchor="middle" tone={3} size={9}>{humanTime(m.perToken)}</Label>
            </g>
          );
        })}

{/* ═════ panel C · per decode step ═════ */}
        <Fade show={focus === 'step' || s.step === 6}>
          <line x1={20} x2={620} y1={286} y2={286} className="v-rule faint" />
          <Label x={20} y={302} tone={2} size={10.5} mono={false} bold>
            {`One decode step, batch ${BATCH} at ${fmt.int(ctx)} tokens: stream it all, or recompute it all`}
          </Label>
          {stepRows.map((r, i) => {
            const y = 310 + i * 16;
            const best = Math.min(r.tFetch, r.tRecomp);
            const barW = Math.min(120, Math.max(4, (Math.log10(Math.min(best, 1e2)) - Math.log10(1e-3)) / (Math.log10(1e2) - Math.log10(1e-3)) * 120));
            const fast = r.tFetch <= r.tRecomp;
            return (
              <g key={r.id}>
                <Label x={104} y={y + 9} anchor="end" tone={2} size={9.5}>{r.short}</Label>
                <Span x={110} y={y} w={barW} h={11} kind={fast ? 'memory' : 'compute'} dim={!fast} opacity={0.9} />
                <Label x={110 + barW + 6} y={y + 9.5} size={9.5} kind={fast ? 'memory' : 'compute'}>
                  {`${humanTime(r.tFetch)} fetch · ${humanTime(r.tRecomp)} recompute`}
                </Label>
              </g>
            );
          })}
          <Label x={110} y={378} tone={3} size={9}>
            {`the bar is whichever is cheaper · ${winner.short} wins here at ${fmt.int(ctx)} tokens`}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
