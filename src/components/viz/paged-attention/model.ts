// Shared constants and helpers for Module 08's visuals.

// Llama-3.1-8B, BF16: 2 (K and V) x 32 layers x 8 KV heads x 128 head dim x 2 bytes.
export const KV_BYTES_PER_TOKEN = 131072;

/** How many blocks a sequence of `tokens` needs, counting a partly filled last one. */
export const blocksNeeded = (tokens: number, blockSize: number) =>
  Math.max(1, Math.ceil(tokens / blockSize));