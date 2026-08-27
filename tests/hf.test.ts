import { describe, expect, it } from 'vitest';
import { parseHfConfig } from '../src/models/hf.js';
import { findPreset } from '../src/models/catalog.js';
import { estimateInference } from '../src/inference.js';

/** Abridged `meta-llama/Llama-3.1-8B` config.json — only keys the parser reads. */
const LLAMA_31_8B = {
  architectures: ['LlamaForCausalLM'],
  hidden_act: 'silu',
  hidden_size: 4096,
  intermediate_size: 14336,
  max_position_embeddings: 131072,
  model_type: 'llama',
  num_attention_heads: 32,
  num_hidden_layers: 32,
  num_key_value_heads: 8,
  tie_word_embeddings: false,
  vocab_size: 128256,
};

/** Abridged `mistralai/Mixtral-8x7B-v0.1` config.json. */
const MIXTRAL_8X7B = {
  architectures: ['MixtralForCausalLM'],
  hidden_act: 'silu',
  hidden_size: 4096,
  intermediate_size: 14336,
  max_position_embeddings: 32768,
  model_type: 'mixtral',
  num_attention_heads: 32,
  num_experts_per_tok: 2,
  num_hidden_layers: 32,
  num_key_value_heads: 8,
  num_local_experts: 8,
  tie_word_embeddings: false,
  vocab_size: 32000,
};

/** Abridged `deepseek-ai/DeepSeek-V3` config.json. */
const DEEPSEEK_V3 = {
  architectures: ['DeepseekV3ForCausalLM'],
  first_k_dense_replace: 3,
  hidden_act: 'silu',
  hidden_size: 7168,
  intermediate_size: 18432,
  kv_lora_rank: 512,
  max_position_embeddings: 163840,
  model_type: 'deepseek_v3',
  moe_intermediate_size: 2048,
  n_routed_experts: 256,
  n_shared_experts: 1,
  num_attention_heads: 128,
  num_experts_per_tok: 8,
  num_hidden_layers: 61,
  num_key_value_heads: 128,
  q_lora_rank: 1536,
  qk_nope_head_dim: 128,
  qk_rope_head_dim: 64,
  tie_word_embeddings: false,
  v_head_dim: 128,
  vocab_size: 129280,
};

describe('parseHfConfig', () => {
  it('reproduces the hand-entered llama-3.1-8b preset from its config.json', () => {
    const { config, warnings } = parseHfConfig(JSON.stringify(LLAMA_31_8B));
    const preset = findPreset('llama-3.1-8b')!.config;

    expect(warnings).toEqual([]);
    expect(config.numLayers).toBe(preset.numLayers);
    expect(config.hiddenSize).toBe(preset.hiddenSize);
    expect(config.numAttentionHeads).toBe(preset.numAttentionHeads);
    expect(config.numKeyValueHeads).toBe(preset.numKeyValueHeads);
    expect(config.ffnIntermediateSize).toBe(preset.ffnIntermediateSize);
    expect(config.gatedMlp).toBe(preset.gatedMlp);
    expect(config.vocabSize).toBe(preset.vocabSize);
    expect(config.maxPositionEmbeddings).toBe(preset.maxPositionEmbeddings);
    expect(config.tieWordEmbeddings).toBe(preset.tieWordEmbeddings);
    expect(config.moe).toBeUndefined();
    expect(config.name).toBe('llama');
  });

  it('accepts an already-parsed object as well as a string', () => {
    const fromObject = parseHfConfig(LLAMA_31_8B);
    const fromString = parseHfConfig(JSON.stringify(LLAMA_31_8B));
    expect(fromObject.config).toEqual(fromString.config);
  });

  it('reads Mixtral MoE keys (num_local_experts / num_experts_per_tok)', () => {
    const { config, warnings } = parseHfConfig(MIXTRAL_8X7B);
    expect(warnings).toEqual([]);
    expect(config.moe).toEqual({
      numExperts: 8,
      expertsPerToken: 2,
      expertIntermediateSize: 14336,
    });
  });

  it('reads DeepSeek MoE keys and models MLA KV cache exactly, with a caveat', () => {
    const { config, warnings } = parseHfConfig(DEEPSEEK_V3);
    const preset = findPreset('deepseek-v3')!.config;

    expect(config.moe).toEqual({
      numExperts: 256,
      expertsPerToken: 8,
      expertIntermediateSize: 2048,
      numSharedExperts: 1,
      firstKDenseLayers: 3,
    });
    expect(config.moe).toEqual(preset.moe);
    // Compressed latent (512) + decoupled RoPE key (64), cached as one tensor.
    expect(config.kvCacheDimPerLayerOverride).toBe(576);
    expect(config.kvTensorsPerLayer).toBe(1);
    expect(warnings.some((w) => w.includes('Multi-head Latent Attention'))).toBe(true);
  });

  it('fills in missing optional fields and names every assumption', () => {
    const { config, warnings } = parseHfConfig({
      model_type: 'toy',
      num_hidden_layers: 4,
      hidden_size: 256,
      num_attention_heads: 8,
      vocab_size: 32000,
    });

    expect(config.numKeyValueHeads).toBe(8); // assumed MHA
    expect(config.ffnIntermediateSize).toBe(1024); // assumed 4x hidden
    expect(config.gatedMlp).toBe(true);
    expect(config.maxPositionEmbeddings).toBe(4096);
    expect(config.tieWordEmbeddings).toBe(false);

    expect(warnings.some((w) => w.includes('num_key_value_heads'))).toBe(true);
    expect(warnings.some((w) => w.includes('intermediate_size'))).toBe(true);
    expect(warnings.some((w) => w.includes('hidden_act'))).toBe(true);
    expect(warnings.some((w) => w.includes('max_position_embeddings'))).toBe(true);
    expect(warnings.some((w) => w.includes('tie_word_embeddings'))).toBe(true);
  });

  it('counts a non-gated activation as a 2-matrix MLP and says so', () => {
    const { config, warnings } = parseHfConfig({ ...LLAMA_31_8B, hidden_act: 'gelu_new' });
    expect(config.gatedMlp).toBe(false);
    expect(warnings.some((w) => w.includes('gelu_new'))).toBe(true);
  });

  it('descends into a multimodal text_config tower and warns', () => {
    const { config, warnings } = parseHfConfig({
      model_type: 'gemma3',
      architectures: ['Gemma3ForConditionalGeneration'],
      vision_config: { hidden_size: 1152, num_hidden_layers: 27 },
      text_config: {
        hidden_act: 'gelu_pytorch_tanh',
        hidden_size: 5376,
        head_dim: 128,
        intermediate_size: 21504,
        max_position_embeddings: 131072,
        num_attention_heads: 32,
        num_hidden_layers: 62,
        num_key_value_heads: 16,
        tie_word_embeddings: true,
        vocab_size: 262144,
      },
    });

    expect(config.numLayers).toBe(62); // the text tower, not the 27-layer vision tower
    expect(config.headDim).toBe(128);
    expect(config.gatedMlp).toBe(true);
    expect(config.tieWordEmbeddings).toBe(true);
    expect(warnings.some((w) => w.includes('text_config'))).toBe(true);
  });

  it('warns when MoE layers are interleaved rather than a leading dense prefix', () => {
    const { warnings } = parseHfConfig({ ...MIXTRAL_8X7B, decoder_sparse_step: 2 });
    expect(warnings.some((w) => w.includes('decoder_sparse_step'))).toBe(true);
  });

  it('throws with an actionable message on malformed JSON', () => {
    expect(() => parseHfConfig('{')).toThrow(/Not valid JSON/);
    expect(() => parseHfConfig('   ')).toThrow(/Paste a HuggingFace config.json/);
    expect(() => parseHfConfig('[1, 2]')).toThrow(/JSON object/);
  });

  it('throws naming the field when a required key is absent', () => {
    const { hidden_size, ...withoutHidden } = LLAMA_31_8B;
    expect(() => parseHfConfig(withoutHidden)).toThrow(/hidden_size/);
  });

  it('surfaces validateModelConfig errors for an inconsistent shape', () => {
    expect(() => parseHfConfig({ ...LLAMA_31_8B, num_key_value_heads: 7 })).toThrow(/multiple of/);
  });

  // The parsed config has to be a drop-in substitute for a preset everywhere
  // downstream, not just field-for-field equal.
  it('produces an identical estimate to the equivalent preset', () => {
    const args = {
      parallelism: { tp: 2, pp: 1, dp: 1 },
      seqLen: 4096,
      batchSize: 4,
      weightDtype: 'bf16',
      kvDtype: 'fp8',
    } as const;
    const fromPaste = estimateInference({ model: parseHfConfig(LLAMA_31_8B).config, ...args });
    const fromPreset = estimateInference({ model: findPreset('llama-3.1-8b')!.config, ...args });

    expect(fromPaste.peakPerGpuBytes).toBe(fromPreset.peakPerGpuBytes);
    expect(fromPaste.totalParams).toBe(fromPreset.totalParams);
    expect(fromPaste.components).toEqual(fromPreset.components);
  });

  it('estimates a config that omitted half its optional fields', () => {
    const { config, warnings } = parseHfConfig({
      model_type: 'toy',
      num_hidden_layers: 4,
      hidden_size: 256,
      num_attention_heads: 8,
      vocab_size: 32000,
    });
    const breakdown = estimateInference({
      model: config,
      parallelism: { tp: 1, pp: 1, dp: 1 },
      seqLen: 512,
      batchSize: 1,
      weightDtype: 'bf16',
      kvDtype: 'bf16',
    });
    expect(breakdown.peakPerGpuBytes).toBeGreaterThan(0);
    expect(warnings.length).toBeGreaterThan(0);
  });
});
