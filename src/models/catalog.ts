import type { ModelConfig } from '../types.js';

/** A catalog entry: a model config plus bookkeeping for the CLI/UI. */
export interface ModelPreset {
  id: string;
  config: ModelConfig;
  /**
   * Set when the architecture isn't fully captured by this tool's generic
   * dense-attention formula (e.g. Multi-head Latent Attention's low-rank
   * q/kv projections), or when the config was reconstructed from
   * community reporting rather than a verified original source. Param
   * counts for such presets should be treated as approximate.
   */
  caveat?: string;
}

const llama31_8b: ModelConfig = {
  name: 'llama-3.1-8b',
  numLayers: 32,
  hiddenSize: 4096,
  numAttentionHeads: 32,
  numKeyValueHeads: 8,
  ffnIntermediateSize: 14336,
  gatedMlp: true,
  vocabSize: 128256,
  maxPositionEmbeddings: 131072,
  tieWordEmbeddings: false,
};

const llama31_70b: ModelConfig = {
  name: 'llama-3.1-70b',
  numLayers: 80,
  hiddenSize: 8192,
  numAttentionHeads: 64,
  numKeyValueHeads: 8,
  ffnIntermediateSize: 28672,
  gatedMlp: true,
  vocabSize: 128256,
  maxPositionEmbeddings: 131072,
  tieWordEmbeddings: false,
};

const llama31_405b: ModelConfig = {
  name: 'llama-3.1-405b',
  numLayers: 126,
  hiddenSize: 16384,
  numAttentionHeads: 128,
  numKeyValueHeads: 8,
  ffnIntermediateSize: 53248,
  gatedMlp: true,
  vocabSize: 128256,
  maxPositionEmbeddings: 131072,
  tieWordEmbeddings: false,
};

const qwen3_8b: ModelConfig = {
  name: 'qwen3-8b',
  numLayers: 36,
  hiddenSize: 4096,
  numAttentionHeads: 32,
  numKeyValueHeads: 8,
  headDim: 128,
  ffnIntermediateSize: 12288,
  gatedMlp: true,
  vocabSize: 151936,
  maxPositionEmbeddings: 40960,
  tieWordEmbeddings: false,
};

const qwen3_32b: ModelConfig = {
  name: 'qwen3-32b',
  numLayers: 64,
  hiddenSize: 5120,
  numAttentionHeads: 64,
  numKeyValueHeads: 8,
  headDim: 128,
  ffnIntermediateSize: 25600,
  gatedMlp: true,
  vocabSize: 151936,
  maxPositionEmbeddings: 40960,
  tieWordEmbeddings: false,
};

const mistral_7b: ModelConfig = {
  name: 'mistral-7b',
  numLayers: 32,
  hiddenSize: 4096,
  numAttentionHeads: 32,
  numKeyValueHeads: 8,
  ffnIntermediateSize: 14336,
  gatedMlp: true,
  vocabSize: 32000,
  maxPositionEmbeddings: 32768,
  tieWordEmbeddings: false,
};

const mixtral_8x7b: ModelConfig = {
  name: 'mixtral-8x7b',
  numLayers: 32,
  hiddenSize: 4096,
  numAttentionHeads: 32,
  numKeyValueHeads: 8,
  ffnIntermediateSize: 14336,
  gatedMlp: true,
  vocabSize: 32000,
  maxPositionEmbeddings: 32768,
  tieWordEmbeddings: false,
  moe: {
    numExperts: 8,
    expertsPerToken: 2,
    expertIntermediateSize: 14336,
  },
};

const mixtral_8x22b: ModelConfig = {
  name: 'mixtral-8x22b',
  numLayers: 56,
  hiddenSize: 6144,
  numAttentionHeads: 48,
  numKeyValueHeads: 8,
  ffnIntermediateSize: 16384,
  gatedMlp: true,
  vocabSize: 32000,
  maxPositionEmbeddings: 65536,
  tieWordEmbeddings: false,
  moe: {
    numExperts: 8,
    expertsPerToken: 2,
    expertIntermediateSize: 16384,
  },
};

// DeepSeek-V3 uses Multi-head Latent Attention: q/k/v projections are
// low-rank factorized (q_lora_rank=1536, kv_lora_rank=512) rather than the
// full-rank hiddenSize x hiddenSize projections this tool's generic
// dense-attention formula assumes. That formula will overcount attention
// params for this preset; KV cache math uses kvCacheDimPerLayerOverride
// (the compressed latent + decoupled RoPE key, 512 + 64 = 576) which *is*
// accurate, since that's exactly what MLA caches.
const deepseek_v3: ModelConfig = {
  name: 'deepseek-v3',
  numLayers: 61,
  hiddenSize: 7168,
  numAttentionHeads: 128,
  numKeyValueHeads: 128,
  headDim: 128,
  ffnIntermediateSize: 18432,
  gatedMlp: true,
  vocabSize: 129280,
  maxPositionEmbeddings: 131072,
  tieWordEmbeddings: false,
  kvCacheDimPerLayerOverride: 512 + 64,
  kvTensorsPerLayer: 1,
  moe: {
    numExperts: 256,
    expertsPerToken: 8,
    expertIntermediateSize: 2048,
    numSharedExperts: 1,
    firstKDenseLayers: 3,
  },
};

const gptOss20b: ModelConfig = {
  name: 'gpt-oss-20b',
  numLayers: 24,
  hiddenSize: 2880,
  numAttentionHeads: 64,
  numKeyValueHeads: 8,
  headDim: 64,
  ffnIntermediateSize: 2880,
  gatedMlp: true,
  vocabSize: 201088,
  maxPositionEmbeddings: 131072,
  tieWordEmbeddings: false,
  moe: {
    numExperts: 32,
    expertsPerToken: 4,
    expertIntermediateSize: 2880,
  },
};

const gptOss120b: ModelConfig = {
  name: 'gpt-oss-120b',
  numLayers: 36,
  hiddenSize: 2880,
  numAttentionHeads: 64,
  numKeyValueHeads: 8,
  headDim: 64,
  ffnIntermediateSize: 2880,
  gatedMlp: true,
  vocabSize: 201088,
  maxPositionEmbeddings: 131072,
  tieWordEmbeddings: false,
  moe: {
    numExperts: 128,
    expertsPerToken: 4,
    expertIntermediateSize: 2880,
  },
};

const gemma3_27b: ModelConfig = {
  name: 'gemma-3-27b',
  numLayers: 62,
  hiddenSize: 5376,
  numAttentionHeads: 32,
  numKeyValueHeads: 16,
  headDim: 128,
  ffnIntermediateSize: 21504,
  gatedMlp: true,
  vocabSize: 262144,
  maxPositionEmbeddings: 131072,
  tieWordEmbeddings: true,
};

export const MODEL_CATALOG: ModelPreset[] = [
  { id: 'llama-3.1-8b', config: llama31_8b },
  { id: 'llama-3.1-70b', config: llama31_70b },
  { id: 'llama-3.1-405b', config: llama31_405b },
  { id: 'qwen3-8b', config: qwen3_8b, caveat: 'community-reported config, treat as approximate' },
  { id: 'qwen3-32b', config: qwen3_32b, caveat: 'community-reported config, treat as approximate' },
  { id: 'mistral-7b', config: mistral_7b },
  { id: 'mixtral-8x7b', config: mixtral_8x7b },
  { id: 'mixtral-8x22b', config: mixtral_8x22b },
  {
    id: 'deepseek-v3',
    config: deepseek_v3,
    caveat:
      'MLA q/k/v are low-rank factorized; this tool models full-rank projections, so attention param count is overestimated (KV cache size is accurate)',
  },
  { id: 'gpt-oss-20b', config: gptOss20b, caveat: 'community-reported config, treat as approximate' },
  { id: 'gpt-oss-120b', config: gptOss120b, caveat: 'community-reported config, treat as approximate' },
  { id: 'gemma-3-27b', config: gemma3_27b },
];

export function findPreset(id: string): ModelPreset | undefined {
  const needle = id.toLowerCase();
  return MODEL_CATALOG.find((p) => p.id.toLowerCase() === needle);
}
