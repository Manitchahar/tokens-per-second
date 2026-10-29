// Exact encoding of a real number into small floating-point and integer formats, for teaching.

export interface FloatFormat { id: string; name: string; e: number; m: number; bytes: number; maxFinite?: number; note: string }

export const FORMATS: FloatFormat[] = [
  { id: 'fp32', name: 'FP32', e: 8, m: 23, bytes: 4, note: 'Full precision. Training default in the past; rarely used for serving.' },
  { id: 'fp16', name: 'FP16', e: 5, m: 10, bytes: 2, note: 'Half precision. Good precision, small range (max 65,504).' },
  { id: 'bf16', name: 'BF16', e: 8, m: 7, bytes: 2, note: 'Brain float. FP32’s range with less precision. The LLM default.' },
  { id: 'fp8e4m3', name: 'FP8 (E4M3)', e: 4, m: 3, bytes: 1, maxFinite: 448, note: 'Ada, Hopper and Blackwell tensor cores. Used for weights and activations.' },
  { id: 'fp8e5m2', name: 'FP8 (E5M2)', e: 5, m: 2, bytes: 1, note: 'More range, less precision. Used for gradients, sometimes KV cache.' },
];

export interface Encoded { sign: number; expBits: number[]; mantBits: number[]; value: number; overflow: boolean; subnormal: boolean }

function toBits(n: number, width: number) {
  const out: number[] = [];
  for (let i = width - 1; i >= 0; i--) out.push(Math.floor(n / Math.pow(2, i)) % 2);
  return out;
}

/** Round-to-nearest encoding of v into a float with e exponent bits and m mantissa bits. */
export function encodeFloat(v: number, f: FloatFormat): Encoded {
  const { e, m } = f;
  const bias = Math.pow(2, e - 1) - 1;
  const sign = v < 0 || Object.is(v, -0) ? 1 : 0;
  let a = Math.abs(v);
  const maxExpField = f.maxFinite ? Math.pow(2, e) - 1 : Math.pow(2, e) - 2; // E4M3 uses the top exponent for normals
  const maxMant = f.maxFinite ? Math.pow(2, m) - 2 : Math.pow(2, m) - 1;
  const maxFinite = f.maxFinite ?? (1 + maxMant / Math.pow(2, m)) * Math.pow(2, maxExpField - bias);
  let overflow = false;
  if (a > maxFinite) { a = maxFinite; overflow = true; }
  if (a === 0) return { sign, expBits: toBits(0, e), mantBits: toBits(0, m), value: 0, overflow, subnormal: false };

  let E = Math.floor(Math.log2(a));
  const minE = 1 - bias;
  if (E < minE) {
    // subnormal: value = mant / 2^m * 2^minE
    let mant = Math.round((a / Math.pow(2, minE)) * Math.pow(2, m));
    if (mant >= Math.pow(2, m)) {
      // rounds up into the smallest normal
      return { sign, expBits: toBits(1, e), mantBits: toBits(0, m), value: (sign ? -1 : 1) * Math.pow(2, minE), overflow, subnormal: false };
    }
    const value = (mant / Math.pow(2, m)) * Math.pow(2, minE);
    return { sign, expBits: toBits(0, e), mantBits: toBits(mant, m), value: sign ? -value : value, overflow, subnormal: true };
  }
  let mant = Math.round((a / Math.pow(2, E) - 1) * Math.pow(2, m));
  if (mant >= Math.pow(2, m)) { mant = 0; E += 1; }
  let expField = E + bias;
  if (expField > maxExpField || (expField === maxExpField && mant > maxMant)) { expField = maxExpField; mant = maxMant; overflow = true; }
  const value = (1 + mant / Math.pow(2, m)) * Math.pow(2, expField - bias);
  return { sign, expBits: toBits(expField, e), mantBits: toBits(mant, m), value: sign ? -value : value, overflow, subnormal: false };
}

/** Plain signed integer with `bits` bits (two's complement range), rounding to nearest. */
export function encodeInt(v: number, bits: number) {
  const lo = -Math.pow(2, bits - 1);
  const hi = Math.pow(2, bits - 1) - 1;
  const q = Math.max(lo, Math.min(hi, Math.round(v)));
  const u = q < 0 ? q + Math.pow(2, bits) : q;
  return { value: q, bits: toBits(u, bits), clipped: v < lo - 0.5 || v > hi + 0.5, lo, hi };
}
