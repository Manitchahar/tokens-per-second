import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Span, Label, Fade, fmt } from '../lib/svg';
import { llama8b, prefillSec, decodeSec } from './prefill';

const PROMPT = 8192; // one long chat prompt, tokens
const DECODE_BATCH = 32; // users already streaming
const CTX = 8192; // their context length

const STEPS = [
  'A healthy engine',
  'A long prompt arrives',
  'One shot: the stall',
  'Cut the prompt into chunks',
  'Chunks sized to the decode step',
  'The crossover rule',
  'What chunking costs',
];
const CAPTIONS = [
  'Thirty-two users are decoding. Every engine step takes one memory-bound pass over the weights and takes about 20 ms. Each step puts one more token in every user’s stream.',
  'Now a user sends an 8,000-token prompt. Prefill is compute-bound and wants the whole GPU for a quarter of a second. The scheduler has to decide when to give it that.',
  'If the engine takes the whole prefill in one shot, that quarter of a second is one engine step. No token reaches any of the 32 decoders while it runs. This is head-of-line blocking.',
  'Chunked prefill splits the prompt into pieces. Each piece is prefill work small enough to fit inside one engine step, so the decoders keep streaming between chunks.',
  'Size each chunk to about one decode step. Then every chunk hides inside a step the decoders were going to take anyway, and their token rate barely moves.',
  'The rule has a number in it. Find the chunk whose prefill time equals one decode step: everything below that line is free, everything above it is time the decoders wait.',
  'Chunking does not make the arithmetic cheaper. It pays a small fixed cost per chunk, in exchange for bounding the worst wait. You are buying fairness with a little throughput.',
];

// Preset chunk size per step. 0 means "no long prompt yet".
const PRESET: Record<number, number> = { 0: 0, 1: 0, 2: 8192, 3: 2048, 4: 512, 5: 512, 6: 512 };

const decodeMs = decodeSec(DECODE_BATCH, CTX, llama8b).t * 1000;
const chunkMs = (c: number) => prefillSec(c, llama8b).t * 1000;

/** The engine loop: one step per chunk, and the step lasts max(prefill, decode). */
function schedule(chunk: number) {
  const n = Math.max(1, Math.ceil(PROMPT / chunk));
  const step = Math.max(chunkMs(chunk), decodeMs);
  return { n, step, total: n * step, stall: Math.max(0, step - decodeMs), prefill: chunkMs(chunk) };
}

/** Chunk size at which one chunk of prefill takes exactly as long as one decode step. */
const CROSSOVER = Math.round((decodeMs / chunkMs(1000)) * 1000);

const X0 = 84;
const X1 = 618;
const TOK_Y = 62;
const PF_Y = 104;
const LANE_H = 20;

export default function ChunkedPrefill() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [chunk, setChunk] = useState(PRESET[s.step]);
  useEffect(() => setChunk(PRESET[s.step]), [s.step]);

  const sch = useMemo(() => schedule(chunk || 8192), [chunk]);
  const maxTotal = Math.max(schedule(8192).total, schedule(2048).total, schedule(512).total, schedule(256).total);
  const w = useTween(maxTotal, 600);
  const px = (ms: number) => X0 + (ms / w) * (X1 - X0);

  const healthy = chunk === 0;
  const eff = healthy ? { n: 0, step: decodeMs, total: 0, stall: 0, prefill: 0 } : sch;
  const rate = 1000 / eff.step;
  const baseRate = 1000 / decodeMs;
  const tokens = eff.n || 1;
  const tw = Math.max(3, Math.min(14, (X1 - X0) / (tokens * 1.6)));
  const tk = useTween(tw, 500);

  // chart geometry: step duration vs chunk size, against the decode-step line
  const CX0 = 96, CX1 = 612, CY0 = 322, CY1 = 196;
  const cx = (c: number) => CX0 + (Math.log2(c) - Math.log2(128)) / (Math.log2(8192) - Math.log2(128)) * (CX1 - CX0);
  const cyMax = 250;
  const cy = (ms: number) => CY0 - Math.min(1, ms / cyMax) * (CY0 - CY1);
  const curve = useMemo(() => {
    const pts: string[] = [];
    for (let c = 128; c <= 8192; c = Math.round(c * 1.12)) {
      const [, step] = [c, schedule(c).step];
      pts.push(`${pts.length ? 'L' : 'M'}${cx(c).toFixed(1)},${cy(step).toFixed(1)}`);
    }
    return pts.join(' ');
  }, []);

  return (
    <Stage
      rootRef={ref}
      title="Llama-3.1-8B on one H100: what the GPU does when a long prompt arrives"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'token', ['error', 'decoders waiting'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Engine step', value: `${eff.step.toFixed(1)} ms`, kind: eff.stall > 0.5 ? 'error' : 'text', note: `${eff.n || '—'} step${eff.n === 1 ? '' : 's'} for the prompt` },
          { label: 'Decoders get a token every', value: `${eff.step.toFixed(1)} ms`, kind: 'token', note: `${rate.toFixed(1)} tok/s each (was ${baseRate.toFixed(1)})` },
          { label: 'Worst wait added', value: `${eff.stall.toFixed(1)} ms`, kind: eff.stall > 0.5 ? 'error' : 'token', note: healthy ? 'no long prompt' : 'per decode step' },
          { label: 'Prompt finished at', value: healthy ? '—' : `${(eff.total / 1000).toFixed(2)} s`, kind: 'compute', note: healthy ? 'nothing queued' : `${PROMPT.toLocaleString('en-US')} tokens` },
        ]} />
      }
      controls={
        <Slider
          label="Prefill chunk size (tokens) — the whole prompt in one piece, or cut into many"
          value={chunk || 8192}
          min={128}
          max={8192}
          log
          onChange={(v) => setChunk(v)}
          format={(v) => `${fmt.int(v)} tokens${Math.abs(v - CROSSOVER) < CROSSOVER * 0.12 ? '  ≈ one decode step' : ''}`}
          kind="compute"
        />
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A timeline of one GPU: a lane of decode tokens for 32 users and a lane of prefill chunks, where a long unchunked prompt stalls every decode token, and smaller chunks restore the token stream at the cost of a slightly longer total time. Below, a chart of engine step duration against chunk size with a horizontal line at the decode step time.">
        {/* ── lane labels ── */}
        <text x={X0 - 12} y={TOK_Y + LANE_H / 2 + 4} textAnchor="end" className="v-track-label">Decode</text>
        <text x={X0 - 12} y={PF_Y + LANE_H / 2 + 4} textAnchor="end" className="v-track-label">Prefill</text>
        <Label x={X0} y={TOK_Y - 14} tone={3} size={9.5}>{`32 users, ${fmt.int(CTX)} tokens of context each`}</Label>
        <Label x={X1} y={TOK_Y - 14} anchor="end" tone={3} size={9.5}>time →</Label>

        {/* ── decode lane ── */}
        <line x1={X0} x2={X1} y1={TOK_Y + LANE_H + 3} y2={TOK_Y + LANE_H + 3} className="v-rule faint" />
        {healthy
          ? [1, 2, 3, 4].map((i) => (
              <g key={i}>
                <rect x={px((i - 1) * decodeMs)} y={TOK_Y} width={tk} height={LANE_H} rx={2} fill="var(--token)" opacity={0.9} />
                <Label x={px((i - 1) * decodeMs) + tk / 2} y={TOK_Y - 6} anchor="middle" size={9.5} kind="token">{`${(i * decodeMs).toFixed(0)} ms`}</Label>
              </g>
            ))
          : Array.from({ length: tokens }, (_, i) => (
              <g key={i}>
                <rect x={px(i * eff.step)} y={TOK_Y} width={tk} height={LANE_H} rx={2} fill="var(--token)" opacity={0.9} />
                {i === 0 && <Label x={X0} y={TOK_Y - 6} size={9.5} kind="token">one token to all 32</Label>}
              </g>
            ))}

        {/* ── prefill lane ── */}
        {!healthy && (
          <>
            {Array.from({ length: tokens }, (_, i) => (
              <Span
                key={i}
                x={px(i * eff.step)}
                y={PF_Y}
                w={Math.max(2, eff.prefill)}
                h={LANE_H}
                kind="compute"
                opacity={0.92}
              />
            ))}
            <Fade show={eff.n <= 8}>
              {Array.from({ length: eff.n }, (_, i) => (
                <Label key={i} x={px(i * eff.step) + Math.max(2, eff.prefill) / 2} y={PF_Y + LANE_H / 2 + 3.6} anchor="middle" size={9} kind="compute">
                  {`${fmt.int(PROMPT / eff.n)} tok`}
                </Label>
              ))}
            </Fade>
          </>
        )}
        <Fade show={s.step === 1}>
          <rect x={X0} y={PF_Y} width={X1 - X0} height={LANE_H} rx={2} fill="none" stroke="var(--rule-2)" strokeDasharray="4 4" />
          <Label x={(X0 + X1) / 2} y={PF_Y + LANE_H / 2 + 4} anchor="middle" size={10} tone={3}>prefill wants all of this</Label>
        </Fade>

        {/* ── stall shading, over both lanes ── */}
        {!healthy && eff.stall > 0.5 && (
          <g>
            {Array.from({ length: eff.n }, (_, i) => (
              <rect
                key={i}
                x={px(i * eff.step) + eff.prefill}
                y={TOK_Y - 4}
                width={Math.max(1, eff.stall)}
                height={PF_Y + LANE_H - TOK_Y + 8}
                fill="url(#tps-hatch-red)"
                opacity={0.55}
              />
            ))}
            {eff.n <= 3 && (
              <Label x={X0 + eff.prefill + 4} y={TOK_Y - 10} size={9.5} kind="error">
                {`waiting ${eff.stall.toFixed(0)} ms`}
              </Label>
            )}
          </g>
        )}

        {/* prompt progress */}
        <Label x={X0} y={PF_Y + LANE_H + 16} tone={3} size={9.5}>
          {healthy ? 'no prefill in the batch' : `${fmt.int(PROMPT)} prompt tokens across ${eff.n} step${eff.n === 1 ? '' : 's'} — total ${(eff.total / 1000).toFixed(2)} s`}
        </Label>

        {/* ── chart: step duration vs chunk size ── */}
        <Fade show={s.step >= 5}>
          <Label x={CX0} y={CY1 - 16} tone={2} size={10.5} mono={false} bold>The rule, as a curve</Label>
          <line x1={CX0} x2={CX1} y1={CY0} y2={CY0} className="v-rule" />
          <line x1={CX0} x2={CX0} y1={CY1 - 8} y2={CY0} className="v-rule" />
          {[0, 50, 100, 150, 200, 250].map((v) => (
            <Label key={v} x={CX0 - 6} y={cy(v) + 3} anchor="end" tone={3} size={9}>{v}</Label>
          ))}
          {[128, 512, 2048, 8192].map((v) => (
            <Label key={v} x={cx(v)} y={CY0 + 13} anchor="middle" tone={3} size={9}>{fmt.int(v)}</Label>
          ))}
          <Label x={CX1} y={CY0 + 27} anchor="end" tone={3} size={9}>chunk size (tokens) →</Label>
          <Label x={CX0 - 6} y={CY1 - 20} anchor="end" tone={3} size={9}>ms</Label>

          {/* the red zone: steps longer than a decode step */}
          <rect x={CX0} y={CY1 - 8} width={cx(CROSSOVER) - CX0} height={CY0 - CY1 + 8} fill="url(#tps-hatch-red)" opacity={0.3} />
          <Label x={CX0 + 5} y={CY1 - 2} size={9.5} kind="error">decoders wait</Label>
          <Label x={cx(CROSSOVER) + 5} y={CY1 - 2} size={9.5} kind="token">free</Label>

          {/* the decode-step line */}
          <line x1={CX0} x2={CX1} y1={cy(decodeMs)} y2={cy(decodeMs)} stroke="var(--token)" strokeWidth={1.3} strokeDasharray="5 3" />
          <Label x={CX1} y={cy(decodeMs) - 5} anchor="end" size={9.5} kind="token">{`one decode step = ${decodeMs.toFixed(0)} ms`}</Label>

          <path d={curve} fill="none" stroke="var(--compute)" strokeWidth={1.8} />
          <line x1={cx(chunk)} x2={cx(chunk)} y1={cy(schedule(chunk).step)} y2={CY0} stroke="var(--text-3)" strokeDasharray="2 3" />
          <circle cx={cx(chunk)} cy={cy(schedule(chunk).step)} r={4.5} fill="var(--compute)" stroke="var(--ground-2)" strokeWidth={2} />
          <Label x={cx(chunk)} y={cy(schedule(chunk).step) - 10} anchor="middle" size={9.5} kind="compute">{`${schedule(chunk).step.toFixed(0)} ms`}</Label>

          <Label x={CX0} y={CY0 + 44} tone={3} size={9.5}>
            {`The knee sits at about ${fmt.int(CROSSOVER)} tokens: a chunk that takes one decode step. Smaller is still fair, but you pay more steps.`}
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}