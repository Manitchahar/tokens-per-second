// Module 12 · Profiling & GPU Performance — shared math.
//
// Everything here is arithmetic the learner already owns from Modules 02 (the
// roofline), 05 (the KV cache) and 10 (chunked prefill). No new constants:
// each function is the cheapest honest form of a formula the prose derives.

import { modelById, gpuById, kvBytesPerToken, type Model, type Gpu } from '../lib/perfModel';

export const llama8b = modelById('llama8b');

/** Total threads a CUDA launch asks for, for a grid-stride style elementwise kernel. */
export const gpuTotalThreads = (g: Gpu) => g.id === 't4' ? 40 * 1024 : 132 * 2048;

/**
 * Grid geometry: a natural cover is one thread per element, so the grid is
 * ceil(n / threadsPerBlock) blocks of `threadsPerBlock` threads.
 */
export function gridFor(n: number, threadsPerBlock: number) {
  return { blocks: Math.max(1, Math.ceil(n / threadsPerBlock)), threadsPerBlock };
}

/** How many full rounds of blocks the GPU needs to retire the grid — the "wave" count. */
export function waves(blocks: number, residentBlocks: number) {
  return blocks / residentBlocks;
}

/** Efficiency of the last wave: how much of the final pass is useful work. */
export function waveEfficiency(blocks: number, residentBlocks: number) {
  const w = waves(blocks, residentBlocks);
  return w / Math.ceil(w);
}

/** Pairs of 32-lane warps per block. */
export const warpsPerBlock = (threadsPerBlock: number) => threadsPerBlock / 32;

/** Bytes one warp-level access moves when the 32 lanes read `elemsPerLane` consecutive values each. */
export function warpBytes(bytesPerElem: number, elemsPerLane = 1) {
  return 32 * elemsPerLane * bytesPerElem;
}

/** How many 32-byte sectors a warp access touches, given the stride pattern it used. */
export function sectorsTouched(bytesPerElem: number, elemsPerLane: number, stride: number) {
  // Each lane touches a distinct run of `elemsPerLane` bytes starting `stride` elements away.
  const perLane = Math.max(1, Math.ceil((elemsPerLane * bytesPerElem) / 32));
  return 32 * perLane * Math.max(1, Math.min(stride, 32));
}

/**
 * Speed-up from improving a coalesced access pattern. Ideal traffic is the bytes
 * actually wanted; achieved traffic is what the memory system must move.
 */
export function coalesceSpeedup(bytesPerElem: number, elemsPerLane: number, stride: number) {
  const ideal = warpBytes(bytesPerElem, elemsPerLane);
  const actual = sectorsTouched(bytesPerElem, elemsPerLane, stride) * 32;
  return actual / ideal;
}

/** Decode-phase SOL terms for Llama-3.1-8B in BF16, split the way a profiler splits them. */
export function decodePhase(p: { gpu?: Gpu; model?: Model; batch: number; ctx: number; wBytes?: number; kvBytes?: number }) {
  const gpu = p.gpu ?? gpuById('h100');
  const model = p.model ?? llama8b;
  const wBytes = p.wBytes ?? 2;
  const kvB = p.kvBytes ?? 2;
  const weightBytes = model.params * wBytes;
  const kvBytes = p.batch * p.ctx * kvBytesPerToken(model, kvB);
  const bytes = weightBytes + kvBytes;
  const flops = 2 * model.params * p.batch;
  const tMem = bytes / (gpu.bw * 0.9);
  const tCompute = flops / (gpu.flops * 0.7);
  return { weightBytes, kvBytes, bytes, flops, tMem, tCompute, t: Math.max(tMem, tCompute) + 0.0015, gpu, model };
}

/** Speed of light as a percentage: how close a measured duration is to the memory floor. */
export function solPercent(measured: number, tFloor: number) {
  return measured > 0 ? tFloor / measured : 0;
}
