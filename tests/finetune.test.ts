import { describe, expect, it } from 'vitest';
import { estimateFinetune } from '../src/finetune.js';
import type { FinetuneInput, ModelConfig } from '../src/types.js';

// A tiny synthetic dense model, small enough to hand-compute LLMem's exact
// equations against. cudaPageBytes=1 disables the 2 MiB page-rounding floor
// (which would otherwise swallow every figure at this scale into a single
// page) so the underlying arithmetic can be verified exactly; a separate
// test below checks real-world page rounding on a realistically sized model.
const tinyModel: ModelConfig = {
  name: 'tiny',
  numLayers: 1,
  hiddenSize: 4,
  numAttentionHeads: 2,
  numKeyValueHeads: 2,
  headDim: 2,
  ffnIntermediateSize: 4,
  gatedMlp: false,
  vocabSize: 10,
  maxPositionEmbeddings: 16,
  tieWordEmbeddings: true, // no separate lm_head -> lm_p = 0
};

function baseInput(overrides: Partial<FinetuneInput>): FinetuneInput {
  return {
    model: tinyModel,
    parallelism: { tp: 1, pp: 1, dp: 1 },
    seqLen: 3,
    batchSize: 1,
    method: 'full',
    optimizer: 'adamw',
    zeroStage: 0,
    gradientCheckpointing: false,
    mixedPrecision: true,
    fidelity: 'granular',
    chunkBytes: 64, // chunkParams = 64 / B16(2) = 32
    cudaPageBytes: 1, // disable rounding for exact-arithmetic verification
    baseOverheadBytes: 0,
    ...overrides,
  };
}

// Hand-derived reference values (see plan / PR description for the full derivation):
//   embed_p = 40, other_p = 108, lm_p = 0
//   chunked other_p = ceil(108/32)*32 = 128
//   m_p     = (40+128) * (B16+B32) = 168*6 = 1008
//   m_p16   = (40+128) * B16       = 168*2 = 336
//   m_os    = sum(t_p * 8) over {embed_tokens, q,k,v,o,mlp} = 136*8 = 1088
//   m_out   = (layers=1 + embed=1) * bs(1) * sl(3) * hidden(4) * B16(2) = 48
//   m_lm    = bs*sl*vocab*B16 + 2*bs*(sl-1)*vocab*B16 = 60 + 80 = 140 (lm_p=0, tied)
describe('estimateFinetune — granular fidelity, LLMem equations', () => {
  it('CDP (zeroStage=0, tp=1): m_base + m_p + m_os + m_out + m_lm', () => {
    const b = estimateFinetune(baseInput({ zeroStage: 0, parallelism: { tp: 1, pp: 1, dp: 1 } }));
    // 1008 (m_p) + 1088 (m_os) + 48 (m_out) + 140 (m_lm) = 2284
    expect(b.peakPerGpuBytes).toBe(2284);
  });

  it('ADP (zeroStage=3, tp=1, dp=4): m_base + m_p16 + (m_p32+m_os)/dp + m_out + m_lm', () => {
    const b = estimateFinetune(baseInput({ zeroStage: 3, parallelism: { tp: 1, pp: 1, dp: 4 } }));
    // m_p32 = 1008-336 = 672; 336 + (672+1088)/4 + 48 + 140 = 336+440+48+140 = 964
    expect(b.peakPerGpuBytes).toBe(964);
  });

  it('TP (zeroStage=0, tp=2): m_base + (m_p+m_os)/tp + m_out + m_lm + m_back', () => {
    const b = estimateFinetune(baseInput({ zeroStage: 0, parallelism: { tp: 2, pp: 1, dp: 1 } }));
    // (1008+1088)/2 = 1048; m_back = layers(1)*bs(1)*sl(3)*hidden(4)*((2-1)/2)*B16(2) = 12
    // 1048 + 12 + 48 + 140 = 1248
    expect(b.peakPerGpuBytes).toBe(1248);
  });

  it('DP+TP (zeroStage=3, tp=2, dp=4): composition is strictly below both ADP-alone and TP-alone', () => {
    const b = estimateFinetune(baseInput({ zeroStage: 3, parallelism: { tp: 2, pp: 1, dp: 4 } }));
    // m_dp_peak(dp=4) = 964 - 48 - 140 = 776 (weights+os only); minus (m_p16*tp)/(tp*dp) = 336*2/8=84 -> 692
    // + m_back(12) + m_out(48) + m_lm(140) = 892
    expect(b.peakPerGpuBytes).toBe(892);
    expect(b.peakPerGpuBytes).toBeLessThan(964); // < ADP alone
    expect(b.peakPerGpuBytes).toBeLessThan(1248); // < TP alone
  });

  it('ordering invariant (LLMem §7.2/7.4): CDP >= ADP, and TP < CDP for this model', () => {
    const cdp = estimateFinetune(baseInput({ zeroStage: 0, parallelism: { tp: 1, pp: 1, dp: 1 } }));
    const adp = estimateFinetune(baseInput({ zeroStage: 3, parallelism: { tp: 1, pp: 1, dp: 4 } }));
    const tp = estimateFinetune(baseInput({ zeroStage: 0, parallelism: { tp: 2, pp: 1, dp: 1 } }));
    expect(cdp.peakPerGpuBytes).toBeGreaterThanOrEqual(adp.peakPerGpuBytes);
    expect(tp.peakPerGpuBytes).toBeLessThan(cdp.peakPerGpuBytes);
  });

  it('rejects zeroStage 1/2 under granular fidelity with an actionable error', () => {
    expect(() => estimateFinetune(baseInput({ zeroStage: 1 }))).toThrow(/granular/);
    expect(() => estimateFinetune(baseInput({ zeroStage: 2 }))).toThrow(/granular/);
  });

  it('with default (2 MiB) page rounding, peak is a whole multiple of the page size', () => {
    const b = estimateFinetune(baseInput({ cudaPageBytes: undefined, zeroStage: 0, parallelism: { tp: 1, pp: 1, dp: 1 } }));
    // components each round up to whole pages; overhead is 0 here so the sum of page-aligned
    // components should itself be page-aligned.
    expect(b.peakPerGpuBytes % (2 * 1024 * 1024)).toBe(0);
  });
});

describe('estimateFinetune — simple fidelity', () => {
  it('AdamW mixed-precision full FT costs ~16 bytes/param (2 param + 4 master + 2 grad + 8 optimizer)', () => {
    const b = estimateFinetune({
      model: tinyModel,
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 1,
      batchSize: 1,
      method: 'full',
      optimizer: 'adamw',
      zeroStage: 0,
      gradientCheckpointing: false,
      mixedPrecision: true,
      fidelity: 'simple',
      baseOverheadBytes: 0,
    });
    const weightsComponent = b.components.find((c) => c.name.startsWith('Weights'))!;
    expect(weightsComponent.perGpuBytes / b.totalParams).toBeCloseTo(16, 5);
  });

  it('ZeRO stage shards optimizer/gradient/param progressively across dp', () => {
    const runWithZero = (zeroStage: 0 | 1 | 2 | 3) =>
      estimateFinetune({
        model: tinyModel,
        parallelism: { tp: 1, pp: 1, dp: 4 },
        seqLen: 1,
        batchSize: 1,
        method: 'full',
        optimizer: 'adamw',
        zeroStage,
        gradientCheckpointing: false,
        mixedPrecision: true,
        fidelity: 'simple',
        baseOverheadBytes: 0,
      }).peakPerGpuBytes;

    const z0 = runWithZero(0);
    const z1 = runWithZero(1);
    const z2 = runWithZero(2);
    const z3 = runWithZero(3);
    expect(z1).toBeLessThan(z0);
    expect(z2).toBeLessThan(z1);
    expect(z3).toBeLessThan(z2);
  });

  it('LoRA is far smaller than full fine-tuning for the same model', () => {
    const full = estimateFinetune({
      model: tinyModel,
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 1,
      batchSize: 1,
      method: 'full',
      optimizer: 'adamw',
      zeroStage: 0,
      gradientCheckpointing: false,
      mixedPrecision: true,
      fidelity: 'simple',
      baseOverheadBytes: 0,
    });
    const lora = estimateFinetune({
      model: tinyModel,
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 1,
      batchSize: 1,
      method: 'lora',
      optimizer: 'adamw',
      zeroStage: 0,
      gradientCheckpointing: false,
      mixedPrecision: true,
      lora: { rank: 1 },
      fidelity: 'simple',
      baseOverheadBytes: 0,
    });
    const fullTrainable = full.components.find((c) => c.name.startsWith('Weights'))!.perGpuBytes;
    const loraTrainable = lora.components.find((c) => c.name.startsWith('LoRA'))!.perGpuBytes;
    expect(loraTrainable).toBeLessThan(fullTrainable);
  });
});
