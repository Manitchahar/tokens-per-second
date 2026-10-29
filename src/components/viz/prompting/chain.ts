// One support assistant built two ways: a monolithic prompt, or a router plus focused sub-prompts.
// Token counts are a worked example; accuracies are ILLUSTRATIVE; latencies come from the shared
// perf model (Llama-3.1-8B on an H100) plus an assumed 50 ms of network and queueing per call.
import { decodeStep, modelById, gpuById } from '../lib/perfModel';

export type Intent = 'billing' | 'bug' | 'account' | 'other';
export const INTENTS: Intent[] = ['billing', 'bug', 'account', 'other'];

/** The monolith's sections, in tokens (3,000 in all). `intent` marks the section a ticket type needs. */
export const MONO_SECTIONS: { label: string; tokens: number; intent?: Intent }[] = [
  { label: 'role & tone', tokens: 300 },
  { label: 'billing policy', tokens: 600, intent: 'billing' },
  { label: 'bug triage steps', tokens: 600, intent: 'bug' },
  { label: 'account recovery', tokens: 550, intent: 'account' },
  { label: 'refunds & escalation', tokens: 500, intent: 'other' },
  { label: 'format + examples', tokens: 450 },
];
export const MONO_SYSTEM = MONO_SECTIONS.reduce((a, s) => a + s.tokens, 0); // 3,000

export const T = { ticket: 150, answer: 250, label: 3, router: 400, sub: 700 };

/** Illustrative accuracies. */
export const MONO_ACC: Record<Intent, number> = { billing: 0.86, bug: 0.82, account: 0.78, other: 0.82 };
export const MONO_MEAN = INTENTS.reduce((a, i) => a + MONO_ACC[i], 0) / INTENTS.length; // 0.82
export const SUB_ACC = 0.94;

export const tokens = {
  monoIn: MONO_SYSTEM + T.ticket, // 3,150
  monoOut: T.answer,
  chainIn: T.router + T.ticket + T.sub + T.ticket, // 1,400
  chainOut: T.label + T.answer, // 253
};

const M = modelById('llama8b');
const G = gpuById('h100');
export const OVERHEAD_MS = 50; // assumed network + queueing per call
export const TPOT_MS = decodeStep({ model: M, gpu: G, batch: 1, ctx: 1000 }).t * 1e3; // ≈ 7.5 ms
export const prefillMs = (n: number) => ((2 * M.params * n) / (G.flops * 0.55)) * 1e3; // ≈ 29.5 µs per token

export interface Seg { kind: 'idle' | 'compute' | 'token'; ms: number; label: string }
/** Timeline segments for one request. `cached` = the static prompt is already in the prefix cache. */
export function timeline(arch: 'mono' | 'chain', cached: boolean): Seg[] {
  const pre = (staticTok: number) => prefillMs((cached ? 0 : staticTok) + T.ticket);
  if (arch === 'mono') {
    return [
      { kind: 'idle', ms: OVERHEAD_MS, label: 'network' },
      { kind: 'compute', ms: pre(MONO_SYSTEM), label: 'prefill' },
      { kind: 'token', ms: T.answer * TPOT_MS, label: '250 answer tokens' },
    ];
  }
  return [
    { kind: 'idle', ms: OVERHEAD_MS, label: 'network' },
    { kind: 'compute', ms: pre(T.router), label: 'prefill' },
    { kind: 'token', ms: T.label * TPOT_MS, label: 'label' },
    { kind: 'idle', ms: OVERHEAD_MS, label: 'network' },
    { kind: 'compute', ms: pre(T.sub), label: 'prefill' },
    { kind: 'token', ms: T.answer * TPOT_MS, label: '250 answer tokens' },
  ];
}
/** Time to the final answer's first token: everything before its decode (prefill emits that token). */
export const ttft = (segs: Seg[]) => segs.slice(0, -1).reduce((a, s) => a + s.ms, 0);
export const total = (segs: Seg[]) => segs.reduce((a, s) => a + s.ms, 0);
