/**
 * Byte/unit primitives shared by every estimator.
 *
 * All internal arithmetic is done in **bytes** as plain JS numbers. Even a
 * 405B-parameter model in fp32 is ~1.6e12 bytes, far below Number.MAX_SAFE_INTEGER
 * (9.007e15), so float64 is exact enough here; we only round at the edges.
 */

export const KiB = 1024;
export const MiB = 1024 * 1024;
export const GiB = 1024 * 1024 * 1024;

/** Bytes per element for every supported numeric format. */
export const DTYPE_BYTES = {
  fp32: 4,
  tf32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
  nf4: 0.5,
} as const;

export type DType = keyof typeof DTYPE_BYTES;

export const DTYPES = Object.keys(DTYPE_BYTES) as DType[];

/** LLMem's `B16` and `B32`: 2 bytes for fp16, 4 bytes for fp32. */
export const B16 = 2;
export const B32 = 4;

/**
 * Default CUDA memory page size (`cu_p` in LLMem §4.2), 2 * 1024^2 bytes.
 * The CUDA allocator hands out memory in whole pages, so every granular-fidelity
 * term is rounded up to a multiple of this.
 */
export const CUDA_PAGE_BYTES = 2 * MiB;

/**
 * Default chunk size (`cs`) used by chunk-based memory management
 * (Fang et al. 2022, as adopted by Colossal-AI and referenced in LLMem §4.1).
 */
export const DEFAULT_CHUNK_BYTES = 32 * MiB;

export function bytesOf(dtype: DType): number {
  return DTYPE_BYTES[dtype];
}

export function isDType(value: string): value is DType {
  return Object.hasOwn(DTYPE_BYTES, value);
}

/** Round `bytes` up to a whole multiple of `page` — LLMem's `ceil(x / cu_p) * cu_p`. */
export function ceilToPage(bytes: number, page: number = CUDA_PAGE_BYTES): number {
  if (page <= 0) return bytes;
  return Math.ceil(bytes / page) * page;
}

/** Round `count` up to a whole multiple of `chunk` — LLMem's `ceil(x / cs) * cs`. */
export function ceilToChunk(count: number, chunk: number): number {
  if (chunk <= 0) return count;
  return Math.ceil(count / chunk) * chunk;
}

/** Human-readable byte string using binary units, e.g. `14.96 GiB`. */
export function formatBytes(bytes: number): string {
  const abs = Math.abs(bytes);
  if (abs >= GiB) return `${(bytes / GiB).toFixed(2)} GiB`;
  if (abs >= MiB) return `${(bytes / MiB).toFixed(2)} MiB`;
  if (abs >= KiB) return `${(bytes / KiB).toFixed(2)} KiB`;
  return `${Math.round(bytes)} B`;
}

/** Human-readable parameter count, e.g. `8.03 B`, `671.03 B`, `124.4 M`. */
export function formatParams(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)} B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(2)} M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(2)} K`;
  return `${count}`;
}

export function toGiB(bytes: number): number {
  return bytes / GiB;
}
