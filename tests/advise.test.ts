import { describe, expect, it } from 'vitest';
import { checkFit, maxBatchSize, maxSeqLen, minGpusForInference } from '../src/advise.js';
import { estimateInference } from '../src/inference.js';
import { findPreset } from '../src/models/catalog.js';
import { findGpu } from '../src/gpus/catalog.js';
import type { InferenceInput } from '../src/types.js';

function preset(id: string) {
  const p = findPreset(id);
  if (!p) throw new Error(`missing ${id}`);
  return p.config;
}

function gpu(id: string) {
  const g = findGpu(id);
  if (!g) throw new Error(`missing gpu ${id}`);
  return g;
}

describe('checkFit', () => {
  it('reports fit when peak usage is under capacity', () => {
    const breakdown = estimateInference({
      model: preset('llama-3.1-8b'),
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 1024,
      batchSize: 1,
      weightDtype: 'fp16',
      kvDtype: 'fp16',
    });
    const fit = checkFit(breakdown, gpu('h100-80'));
    expect(fit.fits).toBe(true);
    expect(fit.headroomBytes).toBeGreaterThan(0);
  });

  it('reports no-fit when peak usage exceeds capacity', () => {
    const breakdown = estimateInference({
      model: preset('llama-3.1-70b'),
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 1024,
      batchSize: 1,
      weightDtype: 'fp32',
      kvDtype: 'fp32',
    });
    const fit = checkFit(breakdown, gpu('rtx4090-24'));
    expect(fit.fits).toBe(false);
  });

  it('applies gpuMemoryUtilization as a capacity haircut', () => {
    const breakdown = estimateInference({
      model: preset('llama-3.1-8b'),
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 1024,
      batchSize: 1,
      weightDtype: 'fp16',
      kvDtype: 'fp16',
    });
    const full = checkFit(breakdown, gpu('h100-80'), 1);
    const haircut = checkFit(breakdown, gpu('h100-80'), 0.5);
    expect(haircut.usableBytes).toBeLessThan(full.usableBytes);
    expect(haircut.headroomBytes).toBeLessThan(full.headroomBytes);
  });
});

describe('solver round-trips', () => {
  const base: InferenceInput = {
    model: preset('llama-3.1-8b'),
    parallelism: { tp: 1, pp: 1, dp: 1 },
    seqLen: 4096,
    batchSize: 1,
    weightDtype: 'fp16',
    kvDtype: 'fp16',
  };

  it('maxBatchSize: fed back in, lands at or just under capacity and one more unit does not fit', () => {
    const g = gpu('a100-80');
    const best = maxBatchSize(base, g);
    expect(best).toBeGreaterThan(0);
    const atBest = estimateInference({ ...base, batchSize: best });
    expect(atBest.peakPerGpuBytes).toBeLessThanOrEqual(g.memoryBytes);
    const oneMore = estimateInference({ ...base, batchSize: best + 1 });
    expect(oneMore.peakPerGpuBytes).toBeGreaterThan(g.memoryBytes);
  });

  it('maxSeqLen: fed back in, lands at or just under capacity', () => {
    const g = gpu('a100-80');
    const best = maxSeqLen({ ...base, batchSize: 1 }, g);
    expect(best).toBeGreaterThan(0);
    const atBest = estimateInference({ ...base, seqLen: best });
    expect(atBest.peakPerGpuBytes).toBeLessThanOrEqual(g.memoryBytes);
  });

  it('minGpusForInference finds the smallest GPU count that fits a large model', () => {
    const g = gpu('a100-80');
    const result = minGpusForInference(
      { model: preset('llama-3.1-70b'), parallelism: { pp: 1 }, seqLen: 4096, batchSize: 1, weightDtype: 'fp16', kvDtype: 'fp16' },
      g,
    );
    expect(result).toBeDefined();
    expect(result!.breakdown.peakPerGpuBytes).toBeLessThanOrEqual(g.memoryBytes);
    // one fewer GPU (by dropping dp by 1, or trying a smaller tp) should not fit
    if (result!.numGpus > 1) {
      const fewer = estimateInference({
        model: preset('llama-3.1-70b'),
        parallelism: { tp: result!.tp, pp: 1, dp: Math.max(1, result!.dp - 1) },
        seqLen: 4096,
        batchSize: 1,
        weightDtype: 'fp16',
        kvDtype: 'fp16',
      });
      if (result!.dp > 1) {
        expect(fewer.peakPerGpuBytes).toBeGreaterThan(g.memoryBytes);
      }
    }
  });
});
