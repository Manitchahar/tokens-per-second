// The four calculators on /tools. Pure client-side, driven by the same spec numbers
// the diagrams use (GPUS, MODELS, decodeStep), so a result here and a point on a
// roofline in a lesson always agree.

import { GPUS, MODELS, kvBytesPerToken, decodeStep } from '../components/viz/lib/perfModel';

const $ = <T extends Element>(sel: string, root: ParentNode = document) => root.querySelector<T>(sel);
const $$ = <T extends Element>(sel: string, root: ParentNode = document) => Array.from(root.querySelectorAll<T>(sel));

const cell = (label: string, value: string, kind?: string, note?: string) =>
  `<div class="cell ${kind ? `k-${kind}` : ''}"><dt>${label}</dt><dd>${value}${note ? `<small>${note}</small>` : ''}</dd></div>`;
const work = (lines: string[]) => `<div class="work">${lines.join('<br />')}</div>`;

const gb = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`);
const num = (n: number, d = 0) => n.toLocaleString('en-US', { maximumFractionDigits: d });
const us = (s: number) => (s < 1e-3 ? `${(s * 1e6).toFixed(1)} µs` : s < 1 ? `${(s * 1e3).toFixed(2)} ms` : `${s.toFixed(2)} s`);
const pick = (sel: string) => ($(sel) as HTMLSelectElement).value;
const val = (sel: string) => Number(($(sel) as HTMLInputElement).value) || 0;

/** Bytes per stored number, by the id used in the precision selects. */
const FORMATS: Record<string, number> = { fp32: 4, bf16: 2, fp8: 1, int4: 0.5 };

/* ── 01 · model memory ─────────────────────────────────────────── */
function memory() {
  const m = MODELS.find((x) => x.id === pick('[data-m-model]'))!;
  const wBytes = FORMATS[pick('[data-m-prec]')];
  const kvBytes = FORMATS[pick('[data-m-kv]')];
  const ctx = val('[data-m-ctx]');
  const batch = val('[data-m-batch]');

  const weights = m.params * wBytes;
  const kvPerSeq = ctx * kvBytesPerToken(m, kvBytes);
  const kv = kvPerSeq * batch;
  const total = weights + kv;

  const chips = GPUS.map((gpu) => {
    const room = gpu.mem - total;
    const ok = room > 0.02 * gpu.mem;
    return `<span class="chip ${ok ? 'yes' : 'no'}">${gpu.name}: ${ok ? `${(room / 1e9).toFixed(1)} GB spare` : `${gb(-room)} over`}</span>`;
  }).join('');

  $('[data-m-out]')!.innerHTML =
    cell('Parameters', `${(m.params / 1e9).toFixed(2)} B`, 'text', m.name) +
    cell('Weights', gb(weights), 'compute', `${wBytes} B per parameter`) +
    cell('KV per sequence', gb(kvPerSeq), 'memory', `${num(ctx)} tokens`) +
    cell('KV for the batch', gb(kv), 'memory', `${num(batch)} sequence${batch > 1 ? 's' : ''}`) +
    cell('Total on the GPU', gb(total), 'token', 'weights + KV cache') +
    `<div class="cell" style="grid-column:1/-1"><dt>Does it fit?</dt><dd class="fits">${chips}</dd></div>` +
    work([
      `weights  = ${(m.params / 1e9).toFixed(2)}e9 params × ${wBytes} B = ${(weights / 1e9).toFixed(2)} GB`,
      `KV/token = 2 × ${m.layers} layers × ${m.kvHeads} kv-heads × ${m.headDim} dim × ${kvBytes} B = ${(kvBytesPerToken(m, kvBytes) / 1024).toFixed(0)} KB`,
      `KV batch = ${num(ctx)} tokens × ${(kvBytesPerToken(m, kvBytes) / 1024).toFixed(0)} KB × ${num(batch)} = ${(kv / 1e9).toFixed(2)} GB`,
      `total    = ${(weights / 1e9).toFixed(2)} + ${(kv / 1e9).toFixed(2)} = ${(total / 1e9).toFixed(2)} GB`,
    ]);
}

/* ── 02 · kv cache ─────────────────────────────────────────────── */
function kv() {
  const m = MODELS.find((x) => x.id === pick('[data-k-model]'))!;
  const bytes = FORMATS[pick('[data-k-prec]')];
  const ctx = val('[data-k-ctx]');
  const n = val('[data-k-n]');
  const gpu = GPUS.find((x) => x.id === pick('[data-k-gpu]'))!;

  const perTok = kvBytesPerToken(m, bytes);
  const perSeq = ctx * perTok;
  const all = perSeq * n;
  const free = gpu.mem - m.params * 2; // weights in BF16
  const maxSeq = Math.max(0, Math.floor(free / perSeq));

  $('[data-k-out]')!.innerHTML =
    cell('KV per token', `${(perTok / 1024).toFixed(1)} KB`, 'memory', `${bytes} B per number`) +
    cell('Per sequence', gb(perSeq), 'memory', `${num(ctx)} tokens`) +
    cell('All sequences', gb(all), n > 0 ? 'token' : 'error', `${num(n)} × per sequence`) +
    cell(`Fits on ${gpu.name}`, maxSeq >= n ? 'yes' : 'no', maxSeq >= n ? 'token' : 'error', `${num(maxSeq)} sequences at this length`) +
    cell('Share of the GPU', `${((all / gpu.mem) * 100).toFixed(1)}%`, all > gpu.mem ? 'error' : 'memory', `${(gpu.mem / 1e9).toFixed(0)} GB total`) +
    work([
      `KV/token = 2 × ${m.layers} × ${m.kvHeads} × ${m.headDim} × ${bytes} B = ${perTok} B`,
      `sequence = ${num(ctx)} × ${perTok} B = ${(perSeq / 1e9).toFixed(3)} GB`,
      `n = ${num(n)}  ->  ${(all / 1e9).toFixed(2)} GB of KV`,
      `free after ${(m.params / 1e9).toFixed(1)}B params in BF16 = ${(free / 1e9).toFixed(1)} GB  ->  ${num(maxSeq)} sequences`,
    ]);
}

/* ── 03 · roofline ─────────────────────────────────────────────── */
function roofline() {
  const gpu = GPUS.find((x) => x.id === pick('[data-r-gpu]'))!;
  const op = pick('[data-r-op]');
  const B = 2; // BF16

  let F = 0;
  let By = 0;
  let what = '';
  if (op === 'decode' || op === 'prefill') {
    const rows = op === 'decode' ? 1 : 512;
    F = 2 * rows * 4096 * 4096;
    By = (4096 * 4096 + 2 * rows * 4096) * B;
    what = `[${rows} × 4096] · [4096 × 4096] in BF16`;
  } else if (op === 'attn') {
    const T = 4096;
    F = 4 * T * T * 4096;
    By = T * 131072; // 2 × 32 layers × 8 kv-heads × 128 dim × 2 B
    what = 'Llama-3.1-8B attention at 4,096 tokens';
  } else {
    F = 4096 * 4096;
    By = 4096 * 4096 * B * 2;
    what = 'one elementwise pass over 4096² BF16 values';
  }

  const intensity = F / By;
  const ridge = gpu.flops / gpu.bw;
  const tFloor = Math.max(F / gpu.flops, By / gpu.bw);
  const tReal = Math.max(F / (gpu.flops * 0.55), By / (gpu.bw * 0.8));
  const bound = F / gpu.flops > By / gpu.bw ? 'compute' : 'memory';

  $('[data-r-out]')!.innerHTML =
    cell('FLOPs', `${(F / 1e9).toFixed(2)} G`, 'compute', what) +
    cell('Bytes', gb(By), 'memory', 'read + write') +
    cell('Arithmetic intensity', intensity < 10 ? intensity.toFixed(2) : num(intensity), 'text', 'FLOP per byte') +
    cell('Ridge point', num(ridge), 'text', 'on this GPU') +
    cell('Limited by', bound, bound === 'compute' ? 'compute' : 'memory', bound === 'compute' ? 'the math units' : 'the memory system') +
    cell('Best possible', us(tFloor), 'idle', 'at 100% of peak') +
    cell('Realistic', us(tReal), 'text', `${((tFloor / tReal) * 100).toFixed(0)}% of the roof`) +
    work([
      `intensity = ${(F / 1e9).toFixed(2)}e9 FLOP ÷ ${(By / 1e9).toFixed(3)}e9 B = ${intensity.toFixed(1)} FLOP/byte`,
      `ridge     = ${(gpu.flops / 1e12).toFixed(0)} TFLOP/s ÷ ${(gpu.bw / 1e12).toFixed(2)} TB/s = ${num(ridge)} FLOP/byte`,
      `t_floor   = max(${us(F / gpu.flops)}, ${us(By / gpu.bw)}) = ${us(tFloor)}`,
      `t_real    = max(${us(F / (gpu.flops * 0.55))}, ${us(By / (gpu.bw * 0.8))}) = ${us(tReal)}`,
    ]);
}

/* ── 04 · cost ─────────────────────────────────────────────────── */
function cost() {
  const m = MODELS.find((x) => x.id === pick('[data-c-model]'))!;
  const gpu = GPUS.find((x) => x.id === pick('[data-c-gpu]'))!;
  const batch = val('[data-c-batch]');
  const ctx = val('[data-c-ctx]');
  const wBytes = FORMATS[pick('[data-c-prec]')];

  const r = decodeStep({ model: m, gpu, batch, ctx, wBytes });
  const memOk = r.fits;

  $('[data-c-out]')!.innerHTML =
    cell('One decode step', us(r.t), r.bound === 'compute' ? 'compute' : 'memory', r.bound === 'compute' ? 'compute-bound' : 'memory-bound') +
    cell('Per user', `${num(r.perUser, 1)} tok/s`, 'token', `${us(r.t)} per token`) +
    cell('All users', num(r.total), 'token', 'tokens per second') +
    cell('GPU memory', `${(r.memUsed / 1e9).toFixed(1)} / ${(gpu.mem / 1e9).toFixed(0)} GB`, memOk ? 'memory' : 'error', memOk ? 'fits' : 'does not fit') +
    cell('Cost per 1M tokens', memOk ? `$${r.costPerMTok < 1 ? r.costPerMTok.toFixed(3) : r.costPerMTok.toFixed(2)}` : '—', 'compute', `$${gpu.price.toFixed(2)}/hr at full use`) +
    work([
      `weights = ${gb(r.weights)} read per step  ->  ${us(r.tWeights)}`,
      `KV read = ${gb(r.kvTotal)} for ${num(batch)} sequences  ->  ${us(r.tKv)}`,
      `step    = max(${us(r.tCompute)} compute, ${us(r.tMem)} memory) + ${(r.overhead * 1e3).toFixed(1)} ms overhead = ${us(r.t)}`,
      `cost    = $${gpu.price}/hr ÷ ${num(r.total)} tok/s = $${r.costPerMTok.toFixed(3)} per million tokens`,
    ]);
}

const TOOLS: Record<string, { fn: () => void; inputs: string[] }> = {
  memory: { fn: memory, inputs: ['[data-m-model]', '[data-m-prec]', '[data-m-kv]', '[data-m-ctx]', '[data-m-batch]'] },
  kv: { fn: kv, inputs: ['[data-k-model]', '[data-k-prec]', '[data-k-ctx]', '[data-k-n]', '[data-k-gpu]'] },
  roofline: { fn: roofline, inputs: ['[data-r-gpu]', '[data-r-op]'] },
  cost: { fn: cost, inputs: ['[data-c-model]', '[data-c-gpu]', '[data-c-batch]', '[data-c-ctx]', '[data-c-prec]'] },
};

export function mountTools() {
  for (const { fn, inputs } of Object.values(TOOLS)) {
    if (!inputs.some((sel) => $(sel))) continue;
    inputs.forEach((sel) => $$(sel).forEach((el) => el.addEventListener('input', fn)));
    fn();
  }
}
