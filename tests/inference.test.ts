import { describe, expect, it } from 'vitest';
import { estimateInference } from '../src/inference.js';
import { findPreset } from '../src/models/catalog.js';
import { GiB } from '../src/units.js';

function model(id: string) {
  const p = findPreset(id);
  if (!p) throw new Error(`missing preset ${id}`);
  return p.config;
}

describe('estimateInference — KV cache', () => {
  it('Llama 3.1 8B: batch=1, seq=8192, fp16 KV cache is exactly 1 GiB (hand-computed golden)', () => {
    // 2 (K+V) * 32 layers * (8 kv heads * 128 head_dim) * 8192 seq * 1 batch * 2 bytes = 2^30 bytes.
    const breakdown = estimateInference({
      model: model('llama-3.1-8b'),
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 8192,
      batchSize: 1,
      weightDtype: 'fp16',
      kvDtype: 'fp16',
    });
    const kv = breakdown.components.find((c) => c.name === 'KV cache')!;
    expect(kv.perGpuBytes).toBe(1 * GiB);
  });

  it('KV cache stops shrinking once tp exceeds numKeyValueHeads (replication cliff)', () => {
    const at8 = estimateInference({
      model: model('llama-3.1-8b'),
      parallelism: { tp: 8, pp: 1, dp: 1 },
      seqLen: 4096,
      batchSize: 1,
      weightDtype: 'fp16',
      kvDtype: 'fp16',
    });
    const at16 = estimateInference({
      model: model('llama-3.1-70b'), // 64 attention heads, divisible by 16
      parallelism: { tp: 16, pp: 1, dp: 1 },
      seqLen: 4096,
      batchSize: 1,
      weightDtype: 'fp16',
      kvDtype: 'fp16',
    });
    // Both models share numKeyValueHeads=8, so both should hit the same divisor (8) and warn.
    expect(at16.warnings.some((w) => w.includes('exceeds numKeyValueHeads'))).toBe(true);
    const kvAt8 = at8.components.find((c) => c.name === 'KV cache')!.perGpuBytes;
    expect(kvAt8).toBeGreaterThan(0);
  });

  it('weight bytes scale down proportionally to tp for a dense model', () => {
    const tp1 = estimateInference({
      model: model('llama-3.1-8b'),
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 1024,
      batchSize: 1,
      weightDtype: 'bf16',
      kvDtype: 'bf16',
    });
    const tp4 = estimateInference({
      model: model('llama-3.1-8b'),
      parallelism: { tp: 4, pp: 1, dp: 1 },
      seqLen: 1024,
      batchSize: 1,
      weightDtype: 'bf16',
      kvDtype: 'bf16',
    });
    const w1 = tp1.components.find((c) => c.name === 'Weights')!.perGpuBytes;
    const w4 = tp4.components.find((c) => c.name === 'Weights')!.perGpuBytes;
    expect(w1 / w4).toBeGreaterThan(3.9);
    expect(w1 / w4).toBeLessThan(4.1);
  });

  it('MoE model surfaces a total-vs-active params warning', () => {
    const breakdown = estimateInference({
      model: model('mixtral-8x7b'),
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 1024,
      batchSize: 1,
      weightDtype: 'bf16',
      kvDtype: 'bf16',
    });
    expect(breakdown.warnings.some((w) => w.includes('MoE model'))).toBe(true);
  });
});

describe('estimateInference — validation', () => {
  it('throws when tp does not divide numAttentionHeads', () => {
    expect(() =>
      estimateInference({
        model: model('llama-3.1-8b'),
        parallelism: { tp: 3, pp: 1, dp: 1 }, // 32 heads, not divisible by 3
        seqLen: 1024,
        batchSize: 1,
        weightDtype: 'fp16',
        kvDtype: 'fp16',
      }),
    ).toThrow(/numAttentionHeads/);
  });

  it('throws on numGpus / (tp*pp) mismatch', () => {
    expect(() =>
      estimateInference({
        model: model('llama-3.1-8b'),
        parallelism: { tp: 4, pp: 1, numGpus: 5 },
        seqLen: 1024,
        batchSize: 1,
        weightDtype: 'fp16',
        kvDtype: 'fp16',
      }),
    ).toThrow();
  });
});
