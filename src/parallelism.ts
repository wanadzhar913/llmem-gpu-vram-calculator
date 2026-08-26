import type { ModelConfig, ParallelismSpec } from './types.js';
import type { OperatorTensor } from './models/params.js';

export interface ResolvedParallelism {
  tp: number;
  pp: number;
  dp: number;
  ep: number;
  numGpus: number;
}

/**
 * Fills in defaults (tp/pp/dp/ep = 1), infers `dp` from `numGpus` when only
 * one of the three degrees is left unspecified, and validates the result
 * against the model's head/expert counts. Throws with an actionable message
 * on any inconsistency.
 */
export function resolveParallelism(spec: ParallelismSpec, model: ModelConfig): ResolvedParallelism {
  const tp = spec.tp ?? 1;
  const pp = spec.pp ?? 1;
  const ep = spec.ep ?? 1;

  let dp: number;
  if (spec.dp !== undefined) {
    dp = spec.dp;
  } else if (spec.numGpus !== undefined) {
    const inferred = spec.numGpus / (tp * pp);
    if (!Number.isInteger(inferred) || inferred < 1) {
      throw new Error(
        `Cannot infer dp: numGpus (${spec.numGpus}) is not evenly divisible by tp * pp (${tp * pp}).`,
      );
    }
    dp = inferred;
  } else {
    dp = 1;
  }

  for (const [dim, value] of [
    ['tp', tp],
    ['pp', pp],
    ['dp', dp],
    ['ep', ep],
  ] as const) {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`${dim} must be a positive integer, got ${value}.`);
    }
  }

  const numGpus = tp * pp * dp;
  if (spec.numGpus !== undefined && spec.numGpus !== numGpus) {
    throw new Error(
      `numGpus (${spec.numGpus}) does not match tp * pp * dp (${tp} * ${pp} * ${dp} = ${numGpus}).`,
    );
  }

  if (model.numAttentionHeads % tp !== 0) {
    throw new Error(
      `Tensor-parallel degree tp=${tp} must evenly divide numAttentionHeads (${model.numAttentionHeads}).`,
    );
  }

  if (ep > 1) {
    if (!model.moe) {
      throw new Error(`ep=${ep} requires an MoE model (model.moe is undefined).`);
    }
    if (model.moe.numExperts % ep !== 0) {
      throw new Error(`Expert-parallel degree ep=${ep} must evenly divide moe.numExperts (${model.moe.numExperts}).`);
    }
    if (dp % ep !== 0) {
      throw new Error(`Expert-parallel degree ep=${ep} must evenly divide dp (${dp}); ep shards within the dp group.`);
    }
  }

  return { tp, pp, dp, ep, numGpus };
}

/** Effective divisor for K/V projection weights and KV cache: capped at the KV head count. */
export function kvShardDivisor(tp: number, model: ModelConfig): number {
  return Math.min(tp, model.numKeyValueHeads);
}

function shardDivisor(op: OperatorTensor, resolved: ResolvedParallelism, model: ModelConfig): number {
  switch (op.shardable) {
    case 'tp':
      return resolved.tp;
    case 'tp_kv':
      return kvShardDivisor(resolved.tp, model);
    case 'ep':
      return resolved.ep;
    case 'none':
      return 1;
  }
}

/** Per-GPU parameter count for one operator after TP/EP sharding (ignores pipeline placement). */
export function shardedParams(op: OperatorTensor, resolved: ResolvedParallelism, model: ModelConfig): number {
  return op.params / shardDivisor(op, resolved, model);
}

/**
 * How many full copies of this operator's parameters exist across the whole
 * world. Dense (non-expert) weights are replicated once per `dp` group; MoE
 * expert weights are only replicated `dp / ep` times, since `ep` distinct
 * expert shards already cover the full expert set within each such group
 * (see the `ParallelismSpec` doc comment on how `ep` nests inside `dp`).
 * Multiplying an operator's unsharded `params` by this factor gives the
 * total bytes for that operator summed across every GPU in the world.
 */
export function worldReplicationFactor(op: OperatorTensor, resolved: ResolvedParallelism): number {
  return op.shardable === 'ep' ? resolved.dp / resolved.ep : resolved.dp;
}

/**
 * Which pipeline stage (0-indexed) an operator is placed on. Embedding lives
 * on stage 0, the final norm and lm_head live on the last stage (matching
 * common Megatron-style pipeline layouts), and transformer layers are split
 * into contiguous, evenly-sized blocks across all stages.
 */
export function pipelineStageOf(op: OperatorTensor, pp: number, numLayers: number): number {
  if (op.layer === null) {
    if (op.name === 'embed_tokens') return 0;
    return pp - 1; // final norm, lm_head
  }
  const layersPerStage = Math.ceil(numLayers / pp);
  return Math.min(pp - 1, Math.floor(op.layer / layersPerStage));
}

/**
 * Groups operators by pipeline stage and returns, for each stage, its
 * operator list and the per-GPU (post TP/EP sharding) parameter total —
 * the basis for finding the peak (most-loaded) pipeline stage.
 */
export function distributeAcrossPipeline(
  operators: OperatorTensor[],
  resolved: ResolvedParallelism,
  model: ModelConfig,
): { stage: number; operators: OperatorTensor[]; shardedParamTotal: number }[] {
  const stages: { stage: number; operators: OperatorTensor[]; shardedParamTotal: number }[] = Array.from(
    { length: resolved.pp },
    (_, stage) => ({ stage, operators: [], shardedParamTotal: 0 }),
  );

  for (const op of operators) {
    const stage = pipelineStageOf(op, resolved.pp, model.numLayers);
    const bucket = stages[stage]!;
    bucket.operators.push(op);
    bucket.shardedParamTotal += shardedParams(op, resolved, model);
  }

  return stages;
}

/** The most heavily loaded pipeline stage by post-sharding parameter count. */
export function peakPipelineStage(
  operators: OperatorTensor[],
  resolved: ResolvedParallelism,
  model: ModelConfig,
): { stage: number; operators: OperatorTensor[]; shardedParamTotal: number } {
  const stages = distributeAcrossPipeline(operators, resolved, model);
  return stages.reduce((max, s) => (s.shardedParamTotal > max.shardedParamTotal ? s : max), stages[0]!);
}
