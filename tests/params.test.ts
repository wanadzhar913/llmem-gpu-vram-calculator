import { describe, expect, it } from 'vitest';
import { countParams } from '../src/models/params.js';
import { findPreset } from '../src/models/catalog.js';

function preset(id: string) {
  const p = findPreset(id);
  if (!p) throw new Error(`missing preset ${id}`);
  return p.config;
}

describe('countParams', () => {
  it('matches Llama 3.1 8B published total (~8.03B) within 0.5%', () => {
    const { totalParams } = countParams(preset('llama-3.1-8b'));
    const published = 8_030_000_000;
    expect(Math.abs(totalParams - published) / published).toBeLessThan(0.005);
  });

  it('matches Mixtral 8x7B published total (~46.7B) and active (~12.9B) within 1%', () => {
    const { totalParams, activeParams } = countParams(preset('mixtral-8x7b'));
    expect(Math.abs(totalParams - 46.7e9) / 46.7e9).toBeLessThan(0.01);
    expect(Math.abs(activeParams - 12.9e9) / 12.9e9).toBeLessThan(0.01);
  });

  it('active params equal total params for a dense (non-MoE) model', () => {
    const { totalParams, activeParams } = countParams(preset('llama-3.1-8b'));
    expect(activeParams).toBe(totalParams);
  });

  it('ties embeddings correctly: no separate lm_head row when tieWordEmbeddings is true', () => {
    const gemma = countParams(preset('gemma-3-27b'));
    expect(gemma.operators.some((o) => o.name === 'lm_head')).toBe(false);
    const llama = countParams(preset('llama-3.1-8b'));
    expect(llama.operators.some((o) => o.name === 'lm_head')).toBe(true);
  });

  it('MoE expert row activeParams reflects top-k routing, not full expert count', () => {
    const mixtral = countParams(preset('mixtral-8x7b'));
    const expertRow = mixtral.operators.find((o) => o.name === 'mlp.experts');
    expect(expertRow).toBeDefined();
    // 8 experts, top-2: active should be exactly 2/8 of total for this row.
    expect(expertRow!.activeParams / expertRow!.params).toBeCloseTo(2 / 8, 10);
  });
});
