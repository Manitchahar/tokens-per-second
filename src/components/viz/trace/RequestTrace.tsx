import { Span, Label, Ruler, Chip, Fade, COLOR } from '../lib/svg';

// One chat request, drawn as a profiler trace. Times are in milliseconds (illustrative but realistic
// for a ~8B model on one modern GPU with a short prompt).
export const PROMPT = ['Explain', ' the', ' KV', ' cache', ' in', ' one', ' sentence', '.'];
export const PROMPT_IDS = [849, 279, 85953, 6636, 304, 832, 11914, 13];
export const OUTPUT = ['It', ' stores', ' each', ' token', '’s', ' keys', ' and', ' values', ' so', ' the', ' model', ' never', ' recom', 'putes', ' them', '.'];

export const T = {
  net: 30,
  tokenizeEnd: 40,
  queueEnd: 95,
  prefillEnd: 215,
  step: 25,
};
export const tokenTime = (i: number) => T.prefillEnd + i * T.step; // produced on GPU
export const arriveTime = (i: number) => tokenTime(i) + T.net; // shown to user
export const END = arriveTime(OUTPUT.length - 1);
export const TTFT = arriveTime(0);

const W = 640;
const X0 = 92;
const X1 = 626;
const TMAX = 640;
const sx = (t: number) => X0 + (Math.min(t, TMAX) / TMAX) * (X1 - X0);

const ROW = { client: 132, net: 164, cpu: 196, gpu: 228, tok: 262, kv: 296 };
const H = 14;

interface Props {
  t: number; // current time (playhead)
  showIds?: boolean;
  showQueueGhosts?: boolean;
  showBrackets?: boolean;
  compactTop?: boolean;
}

/** Everything drawn up to time t. */
export function RequestTrace({ t, showIds, showQueueGhosts, showBrackets }: Props) {
  const clipW = (a: number, b: number) => Math.max(0, sx(Math.min(b, t)) - sx(a));
  const on = (a: number) => t >= a;
  const shownOut = OUTPUT.filter((_, i) => t >= arriveTime(i)).join('');
  const kvTokens = t < T.queueEnd ? 0 : t < T.prefillEnd ? Math.round(PROMPT.length * ((t - T.queueEnd) / (T.prefillEnd - T.queueEnd))) : PROMPT.length + OUTPUT.filter((_, i) => t >= tokenTime(i)).length;

  // KV cache growth polygon
  const kvMax = PROMPT.length + OUTPUT.length;
  const kvY = (n: number) => ROW.kv + H - (n / kvMax) * (H + 6);
  const kvPts: string[] = [`${sx(T.queueEnd)},${ROW.kv + H}`];
  if (on(T.queueEnd)) {
    kvPts.push(`${sx(Math.min(t, T.prefillEnd))},${kvY(t < T.prefillEnd ? kvTokens : PROMPT.length)}`);
    for (let i = 0; i < OUTPUT.length; i++) {
      const tt = tokenTime(i);
      if (t < tt) break;
      kvPts.push(`${sx(tt)},${kvY(PROMPT.length + i)}`, `${sx(tt)},${kvY(PROMPT.length + i + 1)}`);
    }
    kvPts.push(`${sx(Math.max(T.queueEnd, Math.min(t, END)))},${kvY(Math.max(0, kvTokens))}`);
    kvPts.push(`${sx(Math.max(T.queueEnd, Math.min(t, END)))},${ROW.kv + H}`);
  }

  return (
    <svg viewBox={`0 0 ${W} 380`} role="img" aria-label="Timeline of one chat request: network, tokenizer, queue, prefill on the GPU, then decode steps producing one token each.">
      {/* ── what the user sees */}
      <g>
        <rect x={12} y={8} width={W - 24} height={86} rx={4} fill="var(--ground)" stroke="var(--rule)" />
        <Label x={26} y={30} tone={3} mono={false} size={10.5} bold>YOU</Label>
        <text x={68} y={30} className="v-label sans t1" style={{ fontSize: 13 }}>{PROMPT.join('')}</text>
        <Label x={26} y={58} tone={3} mono={false} size={10.5} bold>MODEL</Label>
        <text x={68} y={58} className="v-label sans t1" style={{ fontSize: 13, fill: 'var(--token)' }}>
          {shownOut}
          {t > TTFT - 30 && t < END && <tspan className="v-pulse" style={{ fill: 'var(--text)' }}>{'▍'}</tspan>}
        </text>
        {showIds ? (
          <g>
            {PROMPT.map((p, i) => (
              <Chip key={i} x={68 + i * 64} y={70} h={17} w={60} size={9.5} kind="text" text={String(PROMPT_IDS[i])} opacity={on(T.net) ? 1 : 0} />
            ))}
          </g>
        ) : (
          <Label x={68} y={84} tone={3} size={10}>
            {t <= 0 ? 'press Enter ↵' : t < TTFT ? 'waiting for the first token…' : t < END ? `streaming · ${OUTPUT.filter((_, i) => t >= arriveTime(i)).length} tokens` : `done · ${OUTPUT.length} tokens in ${END} ms`}
          </Label>
        )}
      </g>

      {/* ── track labels + lanes */}
      {(Object.keys(ROW) as (keyof typeof ROW)[]).map((k) => (
        <g key={k}>
          <text x={X0 - 12} y={ROW[k] + H / 2 + 4} textAnchor="end" className="v-track-label">
            {{ client: 'Client', net: 'Network', cpu: 'Server CPU', gpu: 'GPU', tok: 'Tokens', kv: 'KV cache' }[k]}
          </text>
          <line x1={X0} x2={X1} y1={ROW[k] + H + 4} y2={ROW[k] + H + 4} className="v-rule faint" />
        </g>
      ))}

      {/* client: send + receive ticks */}
      <Fade show={t >= 0}>
        <rect x={sx(0) - 1} y={ROW.client} width={3} height={H} fill="var(--text)" rx={1} />
      </Fade>
      {OUTPUT.map((_, i) => on(arriveTime(i)) && <rect key={i} x={sx(arriveTime(i)) - 1} y={ROW.client + 3} width={2.5} height={H - 6} rx={1} fill={COLOR.token} />)}

      {/* network: request out, stream back */}
      <Span x={sx(0)} y={ROW.net} w={clipW(0, T.net)} h={H} kind="comm" label={t > T.net ? 'HTTPS' : undefined} />
      {on(tokenTime(0)) && <Span x={sx(tokenTime(0))} y={ROW.net} w={clipW(tokenTime(0), END)} h={H} kind="comm" dim label="token stream (SSE)" labelTone="light" />}

      {/* server CPU: tokenize, queue */}
      {on(T.net) && <Span x={sx(T.net)} y={ROW.cpu} w={clipW(T.net, T.tokenizeEnd)} h={H} kind="compute" dim />}
      {on(T.net) && t > T.net + 4 && <Label x={sx(T.net) + 2} y={ROW.cpu - 4} tone={3} size={9}>tokenize</Label>}
      {on(T.tokenizeEnd) && <Span x={sx(T.tokenizeEnd)} y={ROW.cpu} w={clipW(T.tokenizeEnd, T.queueEnd)} h={H} kind="idle" label={t >= T.queueEnd ? 'queue' : undefined} labelTone="light" />}
      {showQueueGhosts && on(T.tokenizeEnd) && (
        <g>
          <Span x={sx(T.tokenizeEnd)} y={ROW.gpu} w={clipW(T.tokenizeEnd, T.queueEnd)} h={H} kind="compute" ghost label="busy" />
        </g>
      )}

      {/* GPU: prefill (compute) then decode steps (memory-bound) */}
      {on(T.queueEnd) && <Span x={sx(T.queueEnd)} y={ROW.gpu} w={clipW(T.queueEnd, T.prefillEnd)} h={H} kind="compute" label={t > T.queueEnd + 60 ? 'prefill' : undefined} />}
      {OUTPUT.map((_, i) => {
        if (i === 0) return null;
        const a = tokenTime(i - 1);
        const b = tokenTime(i);
        if (!on(a)) return null;
        return <Span key={i} x={sx(a) + 0.8} y={ROW.gpu} w={Math.max(0, clipW(a, b) - 1.6)} h={H} kind="memory" />;
      })}
      {on(tokenTime(2)) && <Label x={sx(tokenTime(0)) + 2} y={ROW.gpu - 4} tone={3} size={9}>decode: one step per token</Label>}

      {/* tokens produced */}
      {OUTPUT.map((_, i) => on(tokenTime(i)) && (
        <rect key={i} x={sx(tokenTime(i)) - 2.5} y={ROW.tok + 2} width={5} height={H - 4} rx={1} fill={COLOR.token} />
      ))}

      {/* KV cache growth */}
      {on(T.queueEnd) && <polygon points={kvPts.join(' ')} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} strokeLinejoin="round" />}
      {on(T.queueEnd) && <Label x={X1} y={ROW.kv - 8} anchor="end" tone={2} size={9.5} kind="memory">{kvTokens} tokens stored</Label>}

      {/* playhead */}
      {t > 0 && t < TMAX && !showBrackets && (
        <g>
          <line x1={sx(t)} x2={sx(t)} y1={ROW.client - 12} y2={ROW.kv + H + 6} stroke="var(--text)" strokeWidth={1} />
          <rect x={sx(t) - 22} y={ROW.client - 26} width={44} height={14} rx={2} fill="var(--text)" />
          <text x={sx(t)} y={ROW.client - 16} textAnchor="middle" className="v-label" style={{ fill: 'var(--ground)', fontSize: 9.5, fontWeight: 600 }}>{Math.round(t)} ms</text>
        </g>
      )}

      {/* brackets: TTFT and TPOT */}
      <Fade show={!!showBrackets}>
        <g>
          <line x1={sx(0)} x2={sx(TTFT)} y1={ROW.client - 8} y2={ROW.client - 8} stroke="var(--text)" />
          <line x1={sx(0)} x2={sx(0)} y1={ROW.client - 12} y2={ROW.client - 4} stroke="var(--text)" />
          <line x1={sx(TTFT)} x2={sx(TTFT)} y1={ROW.client - 12} y2={ROW.client - 4} stroke="var(--text)" />
          <rect x={sx(TTFT / 2) - 58} y={ROW.client - 22} width={116} height={14} fill="var(--ground-2)" />
          <Label x={sx(TTFT / 2)} y={ROW.client - 11} anchor="middle" tone={1} size={10}>{`TTFT = ${TTFT} ms`}</Label>
          <line x1={sx(arriveTime(6))} x2={sx(arriveTime(7))} y1={ROW.client - 8} y2={ROW.client - 8} stroke="var(--token)" />
          <line x1={sx(arriveTime(6))} x2={sx(arriveTime(6))} y1={ROW.client - 12} y2={ROW.client - 4} stroke="var(--token)" />
          <line x1={sx(arriveTime(7))} x2={sx(arriveTime(7))} y1={ROW.client - 12} y2={ROW.client - 4} stroke="var(--token)" />
          <Label x={sx(arriveTime(7)) + 6} y={ROW.client - 5} tone={1} size={10} kind="token">{`TPOT = ${T.step} ms → ${1000 / T.step} tokens/s`}</Label>
        </g>
      </Fade>

      <Ruler x={X0} y={ROW.kv + H + 22} w={X1 - X0} ticks={[0, 100, 200, 300, 400, 500, 600].map((ms) => ({ at: ms / TMAX, label: `${ms}` }))} label="time (ms)" />
    </svg>
  );
}
