import type { ModelConfig } from '../types.js';
import { findPreset, type ModelPreset } from './catalog.js';

export type ModelOverrides = Partial<ModelConfig>;

export interface ResolveModelOptions {
  /** Preset id from the built-in catalog, e.g. 'llama-3.1-8b'. Omit for a fully custom config. */
  preset?: string;
  /** Fields to override on top of the preset, or the entire config when no preset is given. */
  overrides?: ModelOverrides;
}

export interface ResolvedModel {
  config: ModelConfig;
  caveat?: string;
}

const REQUIRED_FIELDS: (keyof ModelConfig)[] = [
  'name',
  'numLayers',
  'hiddenSize',
  'numAttentionHeads',
  'numKeyValueHeads',
  'ffnIntermediateSize',
  'gatedMlp',
  'vocabSize',
  'maxPositionEmbeddings',
  'tieWordEmbeddings',
];

/**
 * Resolves a model config from a catalog preset, a fully custom config, or a
 * preset with field overrides. Throws with an actionable message on an
 * unknown preset id, missing required fields, or an inconsistent shape.
 */
export function resolveModel(options: ResolveModelOptions): ResolvedModel {
  let base: ModelConfig | undefined;
  let caveat: string | undefined;

  if (options.preset !== undefined) {
    const preset: ModelPreset | undefined = findPreset(options.preset);
    if (!preset) {
      throw new Error(
        `Unknown model preset "${options.preset}". Run \`gpumem models\` to list available presets, or supply a full custom config.`,
      );
    }
    base = preset.config;
    caveat = preset.caveat;
  }

  const merged: Partial<ModelConfig> = { ...base, ...options.overrides };

  for (const field of REQUIRED_FIELDS) {
    if (merged[field] === undefined) {
      throw new Error(
        `Model config is missing required field "${field}". Provide a preset via \`preset\`, or include "${field}" in overrides.`,
      );
    }
  }

  const config = merged as ModelConfig;
  validateModelConfig(config);
  return { config, caveat };
}

export function validateModelConfig(config: ModelConfig): void {
  if (config.numLayers <= 0) throw new Error(`numLayers must be positive, got ${config.numLayers}`);
  if (config.hiddenSize <= 0) throw new Error(`hiddenSize must be positive, got ${config.hiddenSize}`);
  if (config.numAttentionHeads <= 0) {
    throw new Error(`numAttentionHeads must be positive, got ${config.numAttentionHeads}`);
  }
  if (config.numKeyValueHeads <= 0) {
    throw new Error(`numKeyValueHeads must be positive, got ${config.numKeyValueHeads}`);
  }
  if (config.numAttentionHeads % config.numKeyValueHeads !== 0) {
    throw new Error(
      `numAttentionHeads (${config.numAttentionHeads}) must be a multiple of numKeyValueHeads (${config.numKeyValueHeads}) for grouped-query attention.`,
    );
  }
  if (config.vocabSize <= 0) throw new Error(`vocabSize must be positive, got ${config.vocabSize}`);
  if (config.ffnIntermediateSize <= 0) {
    throw new Error(`ffnIntermediateSize must be positive, got ${config.ffnIntermediateSize}`);
  }
  if (config.headDim !== undefined && config.headDim <= 0) {
    throw new Error(`headDim must be positive when provided, got ${config.headDim}`);
  }
  if (config.moe) {
    const { numExperts, expertsPerToken, expertIntermediateSize, firstKDenseLayers } = config.moe;
    if (numExperts <= 0) throw new Error(`moe.numExperts must be positive, got ${numExperts}`);
    if (expertsPerToken <= 0 || expertsPerToken > numExperts) {
      throw new Error(
        `moe.expertsPerToken (${expertsPerToken}) must be between 1 and moe.numExperts (${numExperts}).`,
      );
    }
    if (expertIntermediateSize <= 0) {
      throw new Error(`moe.expertIntermediateSize must be positive, got ${expertIntermediateSize}`);
    }
    if (firstKDenseLayers !== undefined && (firstKDenseLayers < 0 || firstKDenseLayers > config.numLayers)) {
      throw new Error(
        `moe.firstKDenseLayers (${firstKDenseLayers}) must be between 0 and numLayers (${config.numLayers}).`,
      );
    }
  }
}
