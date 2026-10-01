// Shared helpers for the Module 13 visuals. Everything here is derived from numbers
// the learner works out in the lessons; nothing is invented for effect.

export interface GpuLite { name: string; bw: number; flops: number; sm: number }

/** The reference GPUs used across the module (approximate, dense figures). */
export const GPUS_LITE: Record<string, GpuLite> = {
  t4: { name: 'T4 (Colab)', bw: 320e9, flops: 65e12, sm: 40 },
  h100: { name: 'H100 SXM', bw: 3.35e12, flops: 989e12, sm: 132 },
};

/**
 * The time a purely memory-bound kernel should take, and the one a purely
 * compute-bound kernel should take. A real kernel is the larger of the two,
 * divided by how well it achieves the peak.
 */
export function roofline(bytes: number, flops: number, gpu: GpuLite, achieve = 0.8) {
  const tMem = bytes / (gpu.bw * achieve);
  const tMath = flops / (gpu.flops * achieve);
  return {
    tMem,
    tMath,
    t: Math.max(tMem, tMath),
    bound: (tMath > tMem ? 'compute' : 'memory') as 'compute' | 'memory',
    intensity: flops / bytes,
    ridge: gpu.flops / gpu.bw,
  };
}

/** Bytes for a tensor, with 2 = BF16/FP16, 4 = FP32, 1 = FP8. */
export const tbytes = (n: number, bytesPer = 2) => n * bytesPer;

/** Grid size Triton would launch for `n` elements at `block`. */
export const gridFor = (n: number, block: number) => Math.ceil(n / block);

/**
 * Does a grid of `programs` blocks of `block` elements saturate the GPU?
 * Two effects bound this: how many programs there are to spread over the SMs,
 * and how much work each program does.
 */
export function occupancy(programs: number, block: number, gpu: GpuLite) {
  const waves = programs / gpu.sm;
  // Below ~2 waves the tail dominates: the last wave leaves most SMs idle.
  const tailEfficiency = Math.min(1, waves / 2);
  return {
    programs,
    block,
    waves,
    tailEfficiency,
    // Bytes each program must move: the block it owns.
    bytesPerProgram: block * 2 * 2, // read x and y, write c (2 bytes each)
  };
}

export const pct = (v: number) => `${(v * 100).toFixed(0)}%`;