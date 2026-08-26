import type { ModelConfig } from '../types.js';

export type OperatorKind = 'embedding' | 'linear' | 'norm';
/**
 * Which parallelism dimension a tensor is split across, if any.
 * `tp_kv` is the K/V projection special case: it shards by
 * `min(tp, numKeyValueHeads)` rather than `tp`, since GQA/MQA models cannot
 * split a single KV head across ranks — once TP degree exceeds the KV head
 * count, those weights (and the KV cache they produce) are replicated, not
 * further sharded.
 */
export type ShardDim = 'tp' | 'tp_kv' | 'ep' | 'none';

/**
 * One weight tensor (or, for MoE expert stacks, one aggregated group of
 * identically-shaped tensors) in the model. This is the single source of
 * truth for parameter counts: every downstream calculation — weight bytes,
 * optimizer state, TP/EP sharding, LoRA target selection — walks this list
 * rather than re-deriving shapes.
 *
 * `activeParams` differs from `params` only for MoE expert rows, where it
 * reflects top-k routing (`expertsPerToken / numExperts` of the total).
 * Memory sizing always uses `params` (every expert's weights must be
 * resident); `activeParams` is exposed for reporting only.
 */
export interface OperatorTensor {
  name: string;
  layer: number | null;
  kind: OperatorKind;
  params: number;
  activeParams: number;
  shardable: ShardDim;
  /** Input/output feature dims, when the tensor is a simple 2D linear map (needed for LoRA adapter sizing). */
  inFeatures?: number;
  outFeatures?: number;
}

export interface ParamCounts {
  totalParams: number;
  activeParams: number;
  operators: OperatorTensor[];
  /** Number of embedding-kind operators (LLMem's `e_n`): input embedding, and lm_head if untied. */
  embeddingOperatorCount: number;
  resolvedHeadDim: number;
}

function resolveHeadDim(model: ModelConfig): number {
  return model.headDim ?? model.hiddenSize / model.numAttentionHeads;
}

function denseFfnParams(hiddenSize: number, intermediateSize: number, gated: boolean): number {
  // gate_proj + up_proj + down_proj (SwiGLU) vs up_proj + down_proj (standard MLP).
  const matrices = gated ? 3 : 2;
  return matrices * hiddenSize * intermediateSize;
}

/**
 * Builds the full per-tensor parameter list for a dense or MoE decoder model.
 * Biases are omitted throughout (negligible relative to weight matrices, and
 * most modern LLMs use bias-free linears/RMSNorm).
 */
export function buildOperatorList(model: ModelConfig): OperatorTensor[] {
  const headDim = resolveHeadDim(model);
  const qDim = model.numAttentionHeads * headDim;
  const kvDim = model.numKeyValueHeads * headDim;
  const ops: OperatorTensor[] = [];

  const embeddingParams = model.vocabSize * model.hiddenSize;
  ops.push({
    name: 'embed_tokens',
    layer: null,
    kind: 'embedding',
    params: embeddingParams,
    activeParams: embeddingParams,
    shardable: 'tp',
  });

  const firstKDense = model.moe?.firstKDenseLayers ?? 0;

  for (let layer = 0; layer < model.numLayers; layer++) {
    const qParams = model.hiddenSize * qDim;
    const kParams = model.hiddenSize * kvDim;
    const vParams = model.hiddenSize * kvDim;
    const oParams = qDim * model.hiddenSize;

    ops.push(
      {
        name: 'q_proj',
        layer,
        kind: 'linear',
        params: qParams,
        activeParams: qParams,
        shardable: 'tp',
        inFeatures: model.hiddenSize,
        outFeatures: qDim,
      },
      {
        name: 'k_proj',
        layer,
        kind: 'linear',
        params: kParams,
        activeParams: kParams,
        shardable: 'tp_kv',
        inFeatures: model.hiddenSize,
        outFeatures: kvDim,
      },
      {
        name: 'v_proj',
        layer,
        kind: 'linear',
        params: vParams,
        activeParams: vParams,
        shardable: 'tp_kv',
        inFeatures: model.hiddenSize,
        outFeatures: kvDim,
      },
      {
        name: 'o_proj',
        layer,
        kind: 'linear',
        params: oParams,
        activeParams: oParams,
        shardable: 'tp',
        inFeatures: qDim,
        outFeatures: model.hiddenSize,
      },
      {
        name: 'input_layernorm',
        layer,
        kind: 'norm',
        params: model.hiddenSize,
        activeParams: model.hiddenSize,
        shardable: 'none',
      },
      {
        name: 'post_attention_layernorm',
        layer,
        kind: 'norm',
        params: model.hiddenSize,
        activeParams: model.hiddenSize,
        shardable: 'none',
      },
    );

    const isMoeLayer = model.moe !== undefined && layer >= firstKDense;
    if (!isMoeLayer) {
      const ffnParams = denseFfnParams(model.hiddenSize, model.ffnIntermediateSize, model.gatedMlp);
      ops.push({
        name: 'mlp',
        layer,
        kind: 'linear',
        params: ffnParams,
        activeParams: ffnParams,
        shardable: 'tp',
      });
      continue;
    }

    const moe = model.moe!;
    const perExpertParams = denseFfnParams(model.hiddenSize, moe.expertIntermediateSize, model.gatedMlp);
    const allExpertsParams = moe.numExperts * perExpertParams;
    const activeExpertsParams = perExpertParams * moe.expertsPerToken;
    ops.push({
      name: 'mlp.experts',
      layer,
      kind: 'linear',
      params: allExpertsParams,
      activeParams: activeExpertsParams,
      shardable: 'ep',
    });

    const routerParams = model.hiddenSize * moe.numExperts;
    ops.push({
      name: 'mlp.router',
      layer,
      kind: 'linear',
      params: routerParams,
      activeParams: routerParams,
      shardable: 'none',
    });

    if (moe.numSharedExperts && moe.numSharedExperts > 0) {
      const sharedParams = moe.numSharedExperts * perExpertParams;
      ops.push({
        name: 'mlp.shared_experts',
        layer,
        kind: 'linear',
        params: sharedParams,
        activeParams: sharedParams,
        shardable: 'tp',
      });
    }
  }

  ops.push({
    name: 'norm',
    layer: null,
    kind: 'norm',
    params: model.hiddenSize,
    activeParams: model.hiddenSize,
    shardable: 'none',
  });

  if (!model.tieWordEmbeddings) {
    const lmHeadParams = model.vocabSize * model.hiddenSize;
    ops.push({
      name: 'lm_head',
      layer: null,
      kind: 'embedding',
      params: lmHeadParams,
      activeParams: lmHeadParams,
      shardable: 'tp',
    });
  }

  return ops;
}

export function countParams(model: ModelConfig): ParamCounts {
  const operators = buildOperatorList(model);
  let totalParams = 0;
  let activeParams = 0;
  let embeddingOperatorCount = 0;
  for (const op of operators) {
    totalParams += op.params;
    activeParams += op.activeParams;
    if (op.kind === 'embedding') embeddingOperatorCount++;
  }
  return {
    totalParams,
    activeParams,
    operators,
    embeddingOperatorCount,
    resolvedHeadDim: resolveHeadDim(model),
  };
}

/** KV cache width per layer per token: `numKeyValueHeads * headDim`, unless overridden (e.g. MLA). */
export function kvDimPerLayer(model: ModelConfig): number {
  if (model.kvCacheDimPerLayerOverride !== undefined) return model.kvCacheDimPerLayerOverride;
  return model.numKeyValueHeads * resolveHeadDim(model);
}

export function kvTensorsPerLayer(model: ModelConfig): number {
  return model.kvTensorsPerLayer ?? 2;
}

export { resolveHeadDim };
