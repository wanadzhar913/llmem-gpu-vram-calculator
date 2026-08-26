import type { GPUSpec, InferenceInput, MemoryBreakdown } from './types.js';
import { estimateInference } from './inference.js';
import { formatBytes } from './units.js';

export interface FitResult {
  fits: boolean;
  capacityBytes: number;
  usableBytes: number;
  peakPerGpuBytes: number;
  headroomBytes: number;
  message: string;
}

/** Whether a breakdown's peak per-GPU usage fits within a GPU's (optionally haircut) capacity. */
export function checkFit(breakdown: MemoryBreakdown, gpu: GPUSpec, gpuMemoryUtilization = 1): FitResult {
  const usableBytes = gpu.memoryBytes * gpuMemoryUtilization;
  const headroomBytes = usableBytes - breakdown.peakPerGpuBytes;
  const fits = headroomBytes >= 0;
  const message = fits
    ? `Fits: ${formatBytes(breakdown.peakPerGpuBytes)} / ${formatBytes(usableBytes)} usable on ${gpu.name} (${formatBytes(headroomBytes)} headroom).`
    : `Does not fit: ${formatBytes(breakdown.peakPerGpuBytes)} needed but only ${formatBytes(usableBytes)} usable on ${gpu.name} (short by ${formatBytes(-headroomBytes)}).`;
  return { fits, capacityBytes: gpu.memoryBytes, usableBytes, peakPerGpuBytes: breakdown.peakPerGpuBytes, headroomBytes, message };
}

/**
 * Largest integer value of a scalar input (e.g. batch size, sequence length)
 * for which `compute(value).peakPerGpuBytes` still fits in `capacityBytes`.
 * Returns 0 if even `lo` does not fit.
 */
export function solveMaxValue(
  compute: (value: number) => MemoryBreakdown,
  capacityBytes: number,
  lo: number,
  hi: number,
): number {
  if (compute(lo).peakPerGpuBytes > capacityBytes) return 0;
  let low = lo;
  let high = hi;
  while (compute(high).peakPerGpuBytes <= capacityBytes) {
    low = high;
    high *= 2;
    if (high > 1e9) break;
  }
  while (low < high) {
    const mid = Math.ceil((low + high + 1) / 2);
    if (compute(mid).peakPerGpuBytes <= capacityBytes) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  return low;
}

/** Largest batch size that fits at a fixed sequence length, holding everything else in `base` constant. */
export function maxBatchSize(base: InferenceInput, gpu: GPUSpec, gpuMemoryUtilization = 1): number {
  const usable = gpu.memoryBytes * gpuMemoryUtilization;
  return solveMaxValue((batchSize) => estimateInference({ ...base, batchSize }), usable, 1, 1024);
}

/** Largest sequence length (context length) that fits at a fixed batch size, holding everything else constant. */
export function maxSeqLen(base: InferenceInput, gpu: GPUSpec, gpuMemoryUtilization = 1): number {
  const usable = gpu.memoryBytes * gpuMemoryUtilization;
  return solveMaxValue((seqLen) => estimateInference({ ...base, seqLen }), usable, 128, 1_048_576);
}

export interface MinGpusResult {
  numGpus: number;
  tp: number;
  pp: number;
  dp: number;
  breakdown: MemoryBreakdown;
}

/**
 * Smallest GPU count (searched as tp * dp, with pp fixed at whatever the
 * caller's `base.parallelism.pp` specifies) for which the model fits. Tries
 * every tp that evenly divides numAttentionHeads, increasing dp until it
 * fits, and returns the candidate with the fewest total GPUs (ties broken
 * by smaller tp, since higher tp adds communication overhead this tool
 * doesn't otherwise model).
 */
export function minGpusForInference(base: InferenceInput, gpu: GPUSpec, gpuMemoryUtilization = 1, maxGpus = 256): MinGpusResult | undefined {
  const usable = gpu.memoryBytes * gpuMemoryUtilization;
  const pp = base.parallelism.pp ?? 1;
  const numHeads = base.model.numAttentionHeads;
  const tpCandidates: number[] = [];
  for (let tp = 1; tp <= numHeads; tp++) {
    if (numHeads % tp === 0) tpCandidates.push(tp);
  }

  let best: MinGpusResult | undefined;
  for (const tp of tpCandidates) {
    for (let dp = 1; tp * pp * dp <= maxGpus; dp++) {
      const numGpus = tp * pp * dp;
      if (best && numGpus >= best.numGpus) break;
      let breakdown: MemoryBreakdown;
      try {
        breakdown = estimateInference({ ...base, parallelism: { tp, pp, dp } });
      } catch {
        continue;
      }
      if (breakdown.peakPerGpuBytes <= usable) {
        if (!best || numGpus < best.numGpus) {
          best = { numGpus, tp, pp, dp, breakdown };
        }
        break;
      }
    }
  }
  return best;
}
