import type { ModelConfig, MoEConfig } from '../types.js';
import { validateModelConfig } from './resolve.js';

/**
 * Result of interpreting a HuggingFace `config.json`. The parser is
 * deliberately lenient: a config that omits a field this tool needs is filled
 * in with a conventional default and the assumption is recorded in
 * `warnings`, rather than rejected. Only fields with no defensible default
 * (layer count, hidden size, head count, vocab size) are hard errors.
 */
export interface HfParseResult {
  config: ModelConfig;
  /** Assumptions the parser had to make, and architectures it can only approximate. */
  warnings: string[];
}

type Json = Record<string, unknown>;

/**
 * Activations that imply a gated (3-matrix SwiGLU/GeGLU) MLP rather than a
 * 2-matrix one. `hidden_act` is the only signal a `config.json` gives about
 * MLP shape; `gelu_new`/`quick_gelu` are deliberately absent because GPT-2
 * style stacks use them with a plain 2-matrix MLP.
 */
const GATED_ACTIVATIONS = new Set(['silu', 'swish', 'swiglu', 'geglu', 'gelu_pytorch_tanh']);

function isJson(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** First key present with a finite numeric value, or undefined. */
function pickNumber(obj: Json, keys: string[]): number | undefined {
  for (const key of keys) {
    const value = obj[key];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return undefined;
}

function pickString(obj: Json, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = obj[key];
    if (typeof value === 'string' && value.length > 0) return value;
  }
  return undefined;
}

function pickBoolean(obj: Json, keys: string[]): boolean | undefined {
  for (const key of keys) {
    const value = obj[key];
    if (typeof value === 'boolean') return value;
  }
  return undefined;
}

function required(obj: Json, keys: string[], field: string): number {
  const value = pickNumber(obj, keys);
  if (value === undefined) {
    throw new Error(
      `config.json is missing "${keys[0]}" (needed for ${field}). Looked for: ${keys.join(', ')}.`,
    );
  }
  return value;
}

function resolveName(obj: Json): string {
  const named = pickString(obj, ['_name_or_path', 'model_type']);
  if (named) return named;
  const architectures = obj['architectures'];
  if (Array.isArray(architectures) && typeof architectures[0] === 'string') return architectures[0];
  return 'custom';
}

/**
 * Multimodal checkpoints nest the language tower under `text_config` (Gemma 3,
 * Llama 4, Qwen-VL). This tool only models decoder-only text stacks, so we
 * descend into that sub-config and say so.
 */
function unwrapTextTower(obj: Json, warnings: string[]): Json {
  for (const key of ['text_config', 'llm_config', 'language_config']) {
    const nested = obj[key];
    if (isJson(nested)) {
      warnings.push(
        `Config has a "${key}" sub-config (multimodal checkpoint); only the text/language tower is counted — vision and adapter weights are ignored.`,
      );
      // Carry the outer name down so the estimate is still recognisable.
      return { ...nested, _name_or_path: nested['_name_or_path'] ?? obj['_name_or_path'] };
    }
  }
  return obj;
}

function parseMoe(obj: Json, ffnIntermediateSize: number, warnings: string[]): MoEConfig | undefined {
  const numExperts = pickNumber(obj, [
    'num_local_experts', // Mixtral
    'n_routed_experts', // DeepSeek-V2/V3
    'num_experts', // Qwen3-MoE, GPT-OSS
    'moe_num_experts',
  ]);
  if (numExperts === undefined) return undefined;

  let expertsPerToken = pickNumber(obj, ['num_experts_per_tok', 'moe_topk', 'num_experts_per_token', 'top_k']);
  if (expertsPerToken === undefined) {
    expertsPerToken = 2;
    warnings.push('No "num_experts_per_tok" in config; assuming top-2 routing (affects active-param count only).');
  }

  const expertIntermediateSize =
    pickNumber(obj, ['moe_intermediate_size', 'expert_intermediate_size', 'ffn_dim']) ?? ffnIntermediateSize;

  const moe: MoEConfig = { numExperts, expertsPerToken, expertIntermediateSize };

  const numSharedExperts = pickNumber(obj, ['n_shared_experts', 'num_shared_experts']);
  if (numSharedExperts !== undefined && numSharedExperts > 0) {
    moe.numSharedExperts = numSharedExperts;
  } else if (pickNumber(obj, ['shared_expert_intermediate_size']) !== undefined) {
    // Qwen2-MoE style: a single shared expert, described only by its width.
    moe.numSharedExperts = 1;
  }

  const firstKDense = pickNumber(obj, ['first_k_dense_replace', 'num_dense_layers']);
  if (firstKDense !== undefined && firstKDense > 0) moe.firstKDenseLayers = firstKDense;

  const layerFreq = obj['moe_layer_freq'];
  if (typeof layerFreq === 'number' && layerFreq !== 1) {
    warnings.push(
      `"moe_layer_freq" is ${layerFreq}: this model interleaves dense and MoE layers, but the estimate models every layer after the dense prefix as MoE. Expert weights are overcounted.`,
    );
  }
  const sparseStep = pickNumber(obj, ['decoder_sparse_step']);
  if (sparseStep !== undefined && sparseStep !== 1) {
    warnings.push(
      `"decoder_sparse_step" is ${sparseStep}: only every ${sparseStep}th layer is MoE, but the estimate models every layer after the dense prefix as MoE. Expert weights are overcounted.`,
    );
  }

  return moe;
}

/**
 * Interprets a HuggingFace `config.json` as a `ModelConfig`. Accepts the raw
 * pasted text or an already-parsed object. Throws on malformed JSON, on a
 * missing field with no sane default, or on a shape `validateModelConfig`
 * rejects.
 */
export function parseHfConfig(input: string | Json): HfParseResult {
  let root: Json;
  if (typeof input === 'string') {
    const text = input.trim();
    if (text.length === 0) throw new Error('Paste a HuggingFace config.json to estimate a custom model.');
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (err) {
      throw new Error(`Not valid JSON: ${err instanceof Error ? err.message : String(err)}`);
    }
    if (!isJson(parsed)) throw new Error('Expected a JSON object at the top level of config.json.');
    root = parsed;
  } else {
    root = input;
  }

  const warnings: string[] = [];
  const obj = unwrapTextTower(root, warnings);

  const numLayers = required(obj, ['num_hidden_layers', 'n_layer', 'num_layers'], 'numLayers');
  const hiddenSize = required(obj, ['hidden_size', 'n_embd', 'd_model'], 'hiddenSize');
  const numAttentionHeads = required(obj, ['num_attention_heads', 'n_head'], 'numAttentionHeads');
  const vocabSize = required(obj, ['vocab_size'], 'vocabSize');

  let numKeyValueHeads = pickNumber(obj, ['num_key_value_heads', 'num_kv_heads']);
  if (numKeyValueHeads === undefined) {
    numKeyValueHeads = numAttentionHeads;
    warnings.push(
      'No "num_key_value_heads" in config; assuming multi-head attention (KV heads = attention heads). KV cache is overestimated if the model actually uses GQA/MQA.',
    );
  }

  const headDim = pickNumber(obj, ['head_dim', 'attention_head_dim']);
  if (headDim === undefined && hiddenSize % numAttentionHeads !== 0) {
    warnings.push(
      `hidden_size (${hiddenSize}) is not divisible by num_attention_heads (${numAttentionHeads}) and no "head_dim" was given; head dimension is derived as a fraction, so parameter counts are approximate.`,
    );
  }

  let ffnIntermediateSize = pickNumber(obj, ['intermediate_size', 'ffn_dim', 'n_inner', 'd_ff']);
  if (ffnIntermediateSize === undefined) {
    ffnIntermediateSize = 4 * hiddenSize;
    warnings.push(`No "intermediate_size" in config; assuming 4 x hidden_size (${ffnIntermediateSize}).`);
  }

  const activation = pickString(obj, ['hidden_act', 'hidden_activation', 'activation_function']);
  let gatedMlp: boolean;
  if (activation !== undefined) {
    gatedMlp = GATED_ACTIVATIONS.has(activation.toLowerCase());
    if (!gatedMlp) {
      warnings.push(
        `"${activation}" is not a gated activation, so the FFN is counted as 2 matrices per layer instead of 3. Override if this model actually uses a gate projection.`,
      );
    }
  } else {
    gatedMlp = true;
    warnings.push('No "hidden_act" in config; assuming a gated (SwiGLU) MLP with 3 weight matrices per FFN.');
  }

  let maxPositionEmbeddings = pickNumber(obj, ['max_position_embeddings', 'n_positions', 'seq_length', 'max_seq_len']);
  if (maxPositionEmbeddings === undefined) {
    maxPositionEmbeddings = 4096;
    warnings.push('No "max_position_embeddings" in config; assuming 4096 (reporting only — does not affect the estimate).');
  }

  let tieWordEmbeddings = pickBoolean(obj, ['tie_word_embeddings', 'tie_weights']);
  if (tieWordEmbeddings === undefined) {
    tieWordEmbeddings = false;
    warnings.push('No "tie_word_embeddings" in config; assuming untied embeddings (a separate lm_head is counted).');
  }

  const config: ModelConfig = {
    name: resolveName(obj),
    numLayers,
    hiddenSize,
    numAttentionHeads,
    numKeyValueHeads,
    ffnIntermediateSize,
    gatedMlp,
    vocabSize,
    maxPositionEmbeddings,
    tieWordEmbeddings,
  };
  if (headDim !== undefined) config.headDim = headDim;

  const moe = parseMoe(obj, ffnIntermediateSize, warnings);
  if (moe) config.moe = moe;

  // Multi-head Latent Attention (DeepSeek-V2/V3): what's cached is the
  // compressed latent plus the decoupled RoPE key, not full K/V. That part we
  // model exactly; the low-rank q/kv projections we do not.
  const kvLoraRank = pickNumber(obj, ['kv_lora_rank']);
  if (kvLoraRank !== undefined) {
    const ropeHeadDim = pickNumber(obj, ['qk_rope_head_dim']) ?? 64;
    config.kvCacheDimPerLayerOverride = kvLoraRank + ropeHeadDim;
    config.kvTensorsPerLayer = 1;
    warnings.push(
      'Multi-head Latent Attention detected ("kv_lora_rank"): q/k/v are low-rank factorized, but this tool models full-rank projections, so the attention parameter count is overestimated. KV cache size is accurate.',
    );
  }

  validateModelConfig(config);
  return { config, warnings };
}
