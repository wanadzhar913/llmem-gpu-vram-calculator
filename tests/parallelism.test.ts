import { describe, expect, it } from 'vitest';
import { resolveParallelism, worldReplicationFactor, shardedParams } from '../src/parallelism.js';
import { countParams } from '../src/models/params.js';
import { findPreset } from '../src/models/catalog.js';

function preset(id: string) {
  const p = findPreset(id);
  if (!p) throw new Error(`missing ${id}`);
  return p.config;
}

describe('resolveParallelism', () => {
  it('defaults tp/pp/dp/ep to 1', () => {
    const r = resolveParallelism({}, preset('llama-3.1-8b'));
    expect(r).toEqual({ tp: 1, pp: 1, dp: 1, ep: 1, numGpus: 1 });
  });

  it('infers dp from numGpus when dp is omitted', () => {
    const r = resolveParallelism({ tp: 4, numGpus: 16 }, preset('llama-3.1-8b'));
    expect(r.dp).toBe(4);
    expect(r.numGpus).toBe(16);
  });

  it('throws when numGpus is not divisible by tp*pp', () => {
    expect(() => resolveParallelism({ tp: 4, numGpus: 10 }, preset('llama-3.1-8b'))).toThrow();
  });

  it('throws when explicit tp/pp/dp product mismatches numGpus', () => {
    expect(() => resolveParallelism({ tp: 2, pp: 2, dp: 2, numGpus: 4 }, preset('llama-3.1-8b'))).toThrow();
  });

  it('throws when tp does not divide numAttentionHeads', () => {
    expect(() => resolveParallelism({ tp: 5 }, preset('llama-3.1-8b'))).toThrow(/numAttentionHeads/);
  });

  it('throws when ep is used on a non-MoE model', () => {
    expect(() => resolveParallelism({ ep: 2 }, preset('llama-3.1-8b'))).toThrow(/MoE/);
  });

  it('throws when ep does not divide numExperts', () => {
    expect(() => resolveParallelism({ ep: 3, dp: 3 }, preset('mixtral-8x7b'))).toThrow(/numExperts/);
  });

  it('accepts a valid ep configuration on an MoE model', () => {
    const r = resolveParallelism({ ep: 4, dp: 4 }, preset('mixtral-8x7b'));
    expect(r.ep).toBe(4);
  });
});

describe('sharding conservation', () => {
  it('per-GPU sharded params times the shard divisor reconstructs the unsharded operator total (tp)', () => {
    const model = preset('llama-3.1-8b');
    const { operators } = countParams(model);
    const resolved = resolveParallelism({ tp: 8 }, model);
    for (const op of operators) {
      if (op.shardable !== 'tp') continue;
      const perGpu = shardedParams(op, resolved, model);
      expect(perGpu * 8).toBeCloseTo(op.params, 6);
    }
  });

  it('worldReplicationFactor times unsharded params gives dp copies for dense ops, dp/ep for MoE experts', () => {
    const model = preset('mixtral-8x7b');
    const { operators } = countParams(model);
    const resolved = resolveParallelism({ ep: 4, dp: 8 }, model);
    const denseOp = operators.find((o) => o.name === 'q_proj')!;
    const expertOp = operators.find((o) => o.name === 'mlp.experts')!;
    expect(worldReplicationFactor(denseOp, resolved)).toBe(8);
    expect(worldReplicationFactor(expertOp, resolved)).toBe(2); // dp/ep = 8/4
  });

  it('TP beyond numKeyValueHeads leaves K/V weight bytes flat', () => {
    const model = preset('llama-3.1-70b'); // 64 attn heads, 8 kv heads
    const { operators } = countParams(model);
    const kOp = operators.find((o) => o.name === 'k_proj' && o.layer === 0)!;
    const at8 = resolveParallelism({ tp: 8 }, model);
    const at16 = resolveParallelism({ tp: 16 }, model);
    const at32 = resolveParallelism({ tp: 32 }, model);
    const s8 = shardedParams(kOp, at8, model);
    const s16 = shardedParams(kOp, at16, model);
    const s32 = shardedParams(kOp, at32, model);
    expect(s16).toBe(s8);
    expect(s32).toBe(s8);
  });
});
