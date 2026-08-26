import { B16, B32, CUDA_PAGE_BYTES, DEFAULT_CHUNK_BYTES, GiB, ceilToChunk, ceilToPage } from './units.js';
import type { FinetuneInput, MemoryBreakdown, MemoryComponent, OptimizerKind } from './types.js';
import { countParams, type OperatorTensor } from './models/params.js';
import { distributeAcrossPipeline, resolveParallelism, shardedParams, worldReplicationFactor } from './parallelism.js';

const DEFAULT_BASE_OVERHEAD_BYTES = 1 * GiB;
/** Double-quantization constant overhead per param, ~0.5/64 + 32/(64*256) bytes (Dettmers et al. 2023, QLoRA). */
const QLORA_QUANT_OVERHEAD_BYTES_PER_PARAM = 0.127;
const DEFAULT_LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj'];
/** Activation elements per token per layer, excluding the O(seq^2) attention-score matrix (Korthikanti et al. 2022). */
const MEGATRON_ACTIVATION_ELEMS_PER_LAYER = 34;

function optimizerStateBytesPerParam(optimizer: OptimizerKind): number {
  switch (optimizer) {
    case 'adamw':
      return B32 + B32; // momentum fp32 + variance fp32
    case 'adamw8bit':
      return 2; // ~1 byte each for momentum/variance (bitsandbytes-style 8-bit states)
    case 'sgd':
      return B32; // momentum fp32 only
  }
}

function frozenDtypeBytesPerParam(input: FinetuneInput): number {
  if (input.method === 'qlora') return 0.5 + QLORA_QUANT_OVERHEAD_BYTES_PER_PARAM; // NF4 + double-quant constants
  if (input.method === 'lora') return input.mixedPrecision ? B16 : B32;
  return 0; // 'full': nothing is frozen
}

/** Synthesizes LoRA adapter tensors for each targeted linear op present in `operators`. */
function loraAdapterOperators(operators: OperatorTensor[], targets: string[], rank: number): OperatorTensor[] {
  const out: OperatorTensor[] = [];
  for (const op of operators) {
    if (op.inFeatures === undefined || op.outFeatures === undefined) continue;
    if (!targets.includes(op.name)) continue;
    const params = rank * (op.inFeatures + op.outFeatures);
    out.push({
      name: `${op.name}.lora`,
      layer: op.layer,
      kind: 'linear',
      params,
      activeParams: params,
      shardable: op.shardable,
    });
  }
  return out;
}

interface StageAccumulators {
  /** LLMem's `embed_p`: trainable input-embedding param count on this stage. */
  embedP: number;
  /** LLMem's `lm_p`: trainable lm_head param count on this stage (bytes are computed by the caller). */
  lmP: number;
  /** LLMem's `other_p`: all other trainable params on this stage (attention, FFN, norms — or LoRA adapters). */
  otherP: number;
  /** Trainable Embedding/Linear tensors on this stage, individually, for the per-operator `m_os` sum. */
  osOperators: { params: number }[];
}

/**
 * Splits a stage's trainable operators into LLMem's `embed_p` / `lm_p` /
 * `other_p` buckets. `other_p` includes norms (they're "the remaining
 * params" per the paper); the per-operator `m_os` sum excludes norms and
 * lm_head, matching LLMem §4.2's "we only calculate GPU memory due to
 * Embedding or Linear operator parameters" and Fig. 2/3's split between the
 * chunk-managed transformer body and the separately-accounted lm_head.
 * MoE expert rows (`shardable: 'ep'`) are pre-divided by the expert-parallel
 * degree here, since LLMem has no equivalent of a second parallel divisor —
 * this is this tool's own extension beyond the paper.
 */
function stageAccumulators(
  trainableOps: OperatorTensor[],
  resolved: ReturnType<typeof resolveParallelism>,
  model: FinetuneInput['model'],
): StageAccumulators {
  let embedP = 0;
  let lmP = 0;
  let otherP = 0;
  const osOperators: { params: number }[] = [];

  for (const op of trainableOps) {
    const effParams = op.shardable === 'ep' ? shardedParams(op, resolved, model) : op.params;
    if (op.name === 'embed_tokens') {
      embedP += effParams;
      osOperators.push({ params: effParams });
      continue;
    }
    if (op.name === 'lm_head') {
      lmP += effParams;
      continue;
    }
    otherP += effParams;
    if (op.kind === 'linear') osOperators.push({ params: effParams });
  }

  return { embedP, lmP, otherP, osOperators };
}

interface StagePeak {
  stage: number;
  layersOnStage: number;
  isLastStage: boolean;
  weightsAndOptBytes: number;
  frozenBytes: number;
  outputBytes: number;
  lmHeadBytes: number;
  backwardBufferBytes: number;
  activationBytes: number;
  baseOverheadBytes: number;
  total: number;
}

function granularStagePeak(
  input: FinetuneInput,
  model: FinetuneInput['model'],
  resolved: ReturnType<typeof resolveParallelism>,
  stageOps: OperatorTensor[],
  stageIndex: number,
): StagePeak {
  const layersOnStage = new Set(stageOps.filter((o) => o.layer !== null).map((o) => o.layer)).size;
  const isLastStage = stageIndex === resolved.pp - 1;
  const page = input.cudaPageBytes ?? CUDA_PAGE_BYTES;
  const chunkParams = (input.chunkBytes ?? DEFAULT_CHUNK_BYTES) / B16;
  const targets = input.lora?.targetModules ?? DEFAULT_LORA_TARGETS;
  const rank = input.lora?.rank ?? 0;
  const optBytesPerParam = optimizerStateBytesPerParam(input.optimizer);

  const trainableOps = input.method === 'full' ? stageOps : loraAdapterOperators(stageOps, targets, rank);
  const { embedP, lmP, otherP, osOperators } = stageAccumulators(trainableOps, resolved, model);

  const frozenBytes =
    input.method === 'full'
      ? 0
      : ceilToPage(
          stageOps.reduce((sum, op) => sum + shardedParams(op, resolved, model) * frozenDtypeBytesPerParam(input), 0),
          page,
        );

  const chunkedOther = ceilToChunk(otherP, chunkParams);
  const mpFull = ceilToPage((embedP + chunkedOther) * (B16 + B32), page);
  const mp16Full = ceilToPage((embedP + chunkedOther) * B16, page);
  const mosFull = osOperators.reduce((sum, o) => sum + ceilToPage(o.params * optBytesPerParam, page), 0);

  const outputBytes = ceilToPage(
    (layersOnStage + (embedP > 0 ? 1 : 0)) * input.batchSize * input.seqLen * model.hiddenSize * B16,
    page,
  );

  const lmHeadBytes = input.method === 'full' ? lmP * B16 : 0;
  const lmHeadLogitsBytes = isLastStage
    ? ceilToPage(input.batchSize * input.seqLen * model.vocabSize * B16, page) +
      2 * ceilToPage(input.batchSize * (input.seqLen - 1) * model.vocabSize * B16, page)
    : 0;
  const mLm = isLastStage ? lmHeadLogitsBytes + lmHeadBytes : 0;

  const tp = resolved.tp;
  const dp = resolved.dp;
  const backwardBufferBytes =
    tp > 1
      ? ceilToPage(layersOnStage * input.batchSize * input.seqLen * model.hiddenSize * ((tp - 1) / tp) * B16, page)
      : 0;

  let weightsAndOptBytes: number;
  if (input.zeroStage === 0 && tp === 1) {
    // CDP: conventional data parallelism, every GPU holds a full copy.
    weightsAndOptBytes = mpFull + mosFull;
  } else if (input.zeroStage === 3 && tp === 1) {
    // ADP: LLMem's ZeRO-3. fp16 params stay resident in full (must be all-gathered for compute);
    // the fp32 master + optimizer states are the part that's actually sharded across dp.
    const mp32Full = mpFull - mp16Full;
    weightsAndOptBytes = mp16Full + (mp32Full + mosFull) / dp;
  } else if (input.zeroStage === 0 && tp > 1) {
    // TP: 1D tensor parallelism, no ZeRO sharding.
    weightsAndOptBytes = (mpFull + mosFull) / tp;
  } else {
    // DP+TP (zeroStage === 3 && tp > 1): LLMem's combination formula. The paper's own
    // gpu_n symbol is reused ambiguously between the ADP sub-term (dp-only) and the
    // correction term; we take the correction term's gpu_n as the total world size
    // (tp * dp), which is our best-faith reading of the paper's terse §5 notation.
    const mp32Full = mpFull - mp16Full;
    const mDpPeak = mp16Full + (mp32Full + mosFull) / dp;
    weightsAndOptBytes = mDpPeak - (mp16Full * tp) / (tp * dp);
  }
  weightsAndOptBytes += backwardBufferBytes;

  const baseOverheadBytes = input.baseOverheadBytes ?? DEFAULT_BASE_OVERHEAD_BYTES;
  const total = baseOverheadBytes + weightsAndOptBytes + frozenBytes + outputBytes + mLm;

  return {
    stage: stageIndex,
    layersOnStage,
    isLastStage,
    weightsAndOptBytes,
    frozenBytes,
    outputBytes,
    lmHeadBytes: mLm,
    backwardBufferBytes,
    activationBytes: 0,
    baseOverheadBytes,
    total,
  };
}

function simpleStagePeak(
  input: FinetuneInput,
  model: FinetuneInput['model'],
  resolved: ReturnType<typeof resolveParallelism>,
  stageOps: OperatorTensor[],
  stageIndex: number,
): StagePeak {
  const layersOnStage = new Set(stageOps.filter((o) => o.layer !== null).map((o) => o.layer)).size;
  const isLastStage = stageIndex === resolved.pp - 1;
  const targets = input.lora?.targetModules ?? DEFAULT_LORA_TARGETS;
  const rank = input.lora?.rank ?? 0;

  const paramBytesPerParam = input.mixedPrecision ? B16 : B32;
  const masterBytesPerParam = input.mixedPrecision ? B32 : 0;
  const gradBytesPerParam = paramBytesPerParam;
  const optBytesPerParam = optimizerStateBytesPerParam(input.optimizer);

  const trainableOps = input.method === 'full' ? stageOps : loraAdapterOperators(stageOps, targets, rank);
  const trainableParams = trainableOps.reduce((sum, op) => sum + shardedParams(op, resolved, model), 0);
  const frozenParams = input.method === 'full' ? 0 : stageOps.reduce((sum, op) => sum + shardedParams(op, resolved, model), 0);

  // Textbook ZeRO: stage 1 shards optimizer state, stage 2 adds gradients, stage 3 adds params too.
  const paramDiv = input.zeroStage >= 3 ? resolved.dp : 1;
  const gradDiv = input.zeroStage >= 2 ? resolved.dp : 1;
  const optDiv = input.zeroStage >= 1 ? resolved.dp : 1;

  const paramBytes = (trainableParams * paramBytesPerParam) / paramDiv;
  const masterBytes = (trainableParams * masterBytesPerParam) / paramDiv;
  const gradBytes = (trainableParams * gradBytesPerParam) / gradDiv;
  const optBytes = (trainableParams * optBytesPerParam) / optDiv;
  const frozenBytes = frozenParams * frozenDtypeBytesPerParam(input);

  const bytesPerActElem = input.mixedPrecision ? B16 : B32;
  const perLayerActivation = model.hiddenSize * input.batchSize * input.seqLen * MEGATRON_ACTIVATION_ELEMS_PER_LAYER * bytesPerActElem;
  let activationBytes: number;
  if (input.gradientCheckpointing) {
    const boundaryPerLayer = model.hiddenSize * input.batchSize * input.seqLen * bytesPerActElem;
    activationBytes = layersOnStage * boundaryPerLayer + perLayerActivation;
  } else {
    activationBytes = layersOnStage * perLayerActivation;
  }

  const logitsBytes = isLastStage ? input.batchSize * input.seqLen * model.vocabSize * bytesPerActElem : 0;

  const baseOverheadBytes = input.baseOverheadBytes ?? DEFAULT_BASE_OVERHEAD_BYTES;
  const total = baseOverheadBytes + paramBytes + masterBytes + gradBytes + optBytes + frozenBytes + activationBytes + logitsBytes;

  return {
    stage: stageIndex,
    layersOnStage,
    isLastStage,
    weightsAndOptBytes: paramBytes + masterBytes + gradBytes + optBytes,
    frozenBytes,
    outputBytes: 0,
    lmHeadBytes: logitsBytes,
    backwardBufferBytes: 0,
    activationBytes,
    baseOverheadBytes,
    total,
  };
}

export function estimateFinetune(input: FinetuneInput): MemoryBreakdown {
  const { model } = input;
  const resolved = resolveParallelism(input.parallelism, model);
  const { operators, totalParams, activeParams } = countParams(model);

  if (input.fidelity === 'granular' && (input.zeroStage === 1 || input.zeroStage === 2)) {
    throw new Error(
      "fidelity='granular' only supports zeroStage 0 (no sharding) or 3 (LLMem's ADP / ZeRO-3), matching " +
        "the paper's chunk-manager model where param fp16 and gradient fp16 share memory and can't be " +
        'sharded independently (see LLMem §4.1). Use fidelity=\'simple\' for ZeRO stage 1/2 estimates.',
    );
  }

  const stages = distributeAcrossPipeline(operators, resolved, model);
  const perStage: StagePeak[] = stages.map((s) =>
    input.fidelity === 'granular'
      ? granularStagePeak(input, model, resolved, s.operators, s.stage)
      : simpleStagePeak(input, model, resolved, s.operators, s.stage),
  );
  const peak = perStage.reduce((max, s) => (s.total > max.total ? s : max), perStage[0]!);

  // World totals use a simpler, non-fidelity-specific byte accounting (see module doc):
  // peak-per-GPU is exact per fidelity, world total is a contextual aggregate.
  const targets = input.lora?.targetModules ?? DEFAULT_LORA_TARGETS;
  const rank = input.lora?.rank ?? 0;
  const trainableBytesPerParam =
    input.fidelity === 'granular'
      ? B16 + B32 + optimizerStateBytesPerParam(input.optimizer)
      : (input.mixedPrecision ? B16 + B32 : B32) + optimizerStateBytesPerParam(input.optimizer);

  let worldTrainableBytes = 0;
  let worldFrozenBytes = 0;
  if (input.method === 'full') {
    for (const op of operators) {
      worldTrainableBytes += op.params * worldReplicationFactor(op, resolved) * trainableBytesPerParam;
    }
  } else {
    const adapterOps = loraAdapterOperators(operators, targets, rank);
    for (const op of adapterOps) {
      worldTrainableBytes += op.params * worldReplicationFactor(op, resolved) * trainableBytesPerParam;
    }
    const frozenBytesPerParam = frozenDtypeBytesPerParam(input);
    for (const op of operators) {
      worldFrozenBytes += op.params * worldReplicationFactor(op, resolved) * frozenBytesPerParam;
    }
  }

  const worldOutputBytes = perStage.reduce((sum, s) => sum + s.outputBytes + s.backwardBufferBytes, 0) * resolved.dp;
  const worldActivationBytes = perStage.reduce((sum, s) => sum + s.activationBytes, 0) * resolved.dp;
  const worldLmBytes = peak.isLastStage
    ? perStage.find((s) => s.isLastStage)!.lmHeadBytes * resolved.dp
    : perStage[perStage.length - 1]!.lmHeadBytes * resolved.dp;
  const worldOverheadBytes = peak.baseOverheadBytes * resolved.numGpus;

  const components: MemoryComponent[] = [
    {
      name: input.method === 'full' ? 'Weights, grads & optimizer state' : 'LoRA adapters (weights, grads, optimizer)',
      totalBytes: worldTrainableBytes,
      perGpuBytes: peak.weightsAndOptBytes,
      note: `zero=${input.zeroStage}${resolved.tp > 1 ? ` tp=${resolved.tp}` : ''}${resolved.pp > 1 ? ` pp=${resolved.pp} (stage ${peak.stage})` : ''}`,
    },
  ];
  if (input.method !== 'full') {
    components.push({
      name: `Frozen base weights (${input.method === 'qlora' ? 'NF4' : input.mixedPrecision ? 'bf16/fp16' : 'fp32'})`,
      totalBytes: worldFrozenBytes,
      perGpuBytes: peak.frozenBytes,
    });
  }
  if (input.fidelity === 'granular') {
    components.push({
      name: 'Activation outputs (forward, retained for backward)',
      totalBytes: worldOutputBytes,
      perGpuBytes: peak.outputBytes + peak.backwardBufferBytes,
    });
  } else {
    components.push({
      name: input.gradientCheckpointing ? 'Activations (checkpointed)' : 'Activations',
      totalBytes: worldActivationBytes,
      perGpuBytes: peak.activationBytes,
    });
  }
  components.push(
    {
      name: 'Logits / lm_head loss',
      totalBytes: worldLmBytes,
      perGpuBytes: peak.lmHeadBytes,
      note: resolved.pp > 1 ? 'only present on the last pipeline stage' : undefined,
    },
    {
      name: 'CUDA/framework overhead',
      totalBytes: worldOverheadBytes,
      perGpuBytes: peak.baseOverheadBytes,
    },
  );

  const totalBytes = components.reduce((sum, c) => sum + c.totalBytes, 0);

  const warnings: string[] = [];
  if (model.moe && activeParams !== totalParams) {
    warnings.push(
      `MoE model: ${totalParams.toLocaleString()} total params must all be resident in memory, even though only ${activeParams.toLocaleString()} are active per token.`,
    );
  }
  if (resolved.tp > model.numKeyValueHeads) {
    warnings.push(
      `tp=${resolved.tp} exceeds numKeyValueHeads=${model.numKeyValueHeads}: K/V weights stop shrinking beyond tp=${model.numKeyValueHeads}.`,
    );
  }
  if (input.method !== 'full') {
    warnings.push(
      'Frozen base weights are assumed replicated per data-parallel rank (not further sharded by ZeRO/dp); FSDP-sharded frozen backbones are not modeled.',
    );
  }

  return {
    scenario: 'finetune',
    fidelity: input.fidelity,
    modelName: model.name,
    totalParams,
    activeParams,
    parallelism: resolved,
    components,
    totalBytes,
    peakPerGpuBytes: peak.total,
    warnings,
  };
}
