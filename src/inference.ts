import { bytesOf, ceilToPage, GiB } from './units.js';
import type { InferenceInput, MemoryBreakdown, MemoryComponent } from './types.js';
import { countParams, kvDimPerLayer, kvTensorsPerLayer } from './models/params.js';
import { distributeAcrossPipeline, kvShardDivisor, resolveParallelism, worldReplicationFactor } from './parallelism.js';

const DEFAULT_BASE_OVERHEAD_BYTES = 1 * GiB;
const DEFAULT_ACTIVATION_FACTOR = 8;

/**
 * Estimates peak per-GPU and world-total memory for serving a model.
 *
 * Weights and KV cache are computed per pipeline stage (a stage only holds
 * its own layers' weights and KV cache), logits are only material on the
 * last stage (where lm_head lives), activations and CUDA overhead are
 * charged on every stage; the returned per-GPU figures are for whichever
 * stage turns out most-loaded.
 */
export function estimateInference(input: InferenceInput): MemoryBreakdown {
  const { model } = input;
  const resolved = resolveParallelism(input.parallelism, model);
  const { operators, totalParams, activeParams } = countParams(model);

  const weightBytesPerParam = bytesOf(input.weightDtype);
  const kvBytesPerElem = bytesOf(input.kvDtype);
  const baseOverheadBytes = input.baseOverheadBytes ?? DEFAULT_BASE_OVERHEAD_BYTES;
  const activationFactor = input.activationFactor ?? DEFAULT_ACTIVATION_FACTOR;
  const logitsMode = input.logitsMode ?? 'lastToken';

  const kvDim = kvDimPerLayer(model);
  const kvTensors = kvTensorsPerLayer(model);
  const kvDivisor = kvShardDivisor(resolved.tp, model);

  const stages = distributeAcrossPipeline(operators, resolved, model);

  const activationBytes = ceilToPage(
    input.batchSize * input.seqLen * model.hiddenSize * weightBytesPerParam * activationFactor,
  );

  const logitsElems =
    logitsMode === 'fullSequence' ? input.batchSize * input.seqLen * model.vocabSize : input.batchSize * model.vocabSize;
  const logitsBytes = ceilToPage(logitsElems * weightBytesPerParam);

  let worldWeightBytes = 0;
  for (const op of operators) {
    worldWeightBytes += op.params * weightBytesPerParam * worldReplicationFactor(op, resolved);
  }

  const perStage = stages.map((s) => {
    const layersOnStage = new Set(s.operators.filter((o) => o.layer !== null).map((o) => o.layer)).size;
    const weightBytes = ceilToPage(s.shardedParamTotal * weightBytesPerParam);
    const kvBytesPerGpu = ceilToPage(
      (kvTensors * layersOnStage * kvDim * input.seqLen * input.batchSize * kvBytesPerElem) / kvDivisor,
    );
    const isLastStage = s.stage === resolved.pp - 1;
    const stageLogitsBytes = isLastStage ? logitsBytes : 0;
    const total = weightBytes + kvBytesPerGpu + activationBytes + stageLogitsBytes + baseOverheadBytes;
    return { stage: s.stage, layersOnStage, weightBytes, kvBytesPerGpu, stageLogitsBytes, total };
  });

  const peak = perStage.reduce((max, s) => (s.total > max.total ? s : max), perStage[0]!);

  const worldKvBytes = ceilToPage(
    (kvTensors * model.numLayers * kvDim * input.seqLen * input.batchSize * kvBytesPerElem * resolved.dp) /
      kvDivisor,
  );

  const components: MemoryComponent[] = [
    {
      name: 'Weights',
      totalBytes: worldWeightBytes,
      perGpuBytes: peak.weightBytes,
      note: `${resolved.tp > 1 ? `TP=${resolved.tp} ` : ''}${resolved.ep > 1 ? `EP=${resolved.ep} ` : ''}${resolved.pp > 1 ? `PP=${resolved.pp} (stage ${peak.stage})` : ''}`.trim() || undefined,
    },
    {
      name: 'KV cache',
      totalBytes: worldKvBytes,
      perGpuBytes: peak.kvBytesPerGpu,
      note:
        resolved.tp > kvDivisor
          ? `TP=${resolved.tp} exceeds ${model.numKeyValueHeads} KV heads: KV cache no longer shrinks past divisor ${kvDivisor} (replicated, not sharded)`
          : undefined,
    },
    {
      name: 'Activations',
      totalBytes: activationBytes * resolved.pp * resolved.dp,
      perGpuBytes: activationBytes,
    },
    {
      name: `Logits (${logitsMode})`,
      totalBytes: logitsBytes * resolved.dp,
      perGpuBytes: peak.stageLogitsBytes,
      note: resolved.pp > 1 ? 'only present on the last pipeline stage' : undefined,
    },
    {
      name: 'CUDA/framework overhead',
      totalBytes: baseOverheadBytes * resolved.numGpus,
      perGpuBytes: baseOverheadBytes,
    },
  ];

  const totalBytes = components.reduce((sum, c) => sum + c.totalBytes, 0);

  const warnings: string[] = [];
  if (model.moe && activeParams !== totalParams) {
    warnings.push(
      `MoE model: ${totalParams.toLocaleString()} total params must all be resident in memory, even though only ${activeParams.toLocaleString()} are active per token.`,
    );
  }
  if (resolved.tp > model.numKeyValueHeads) {
    warnings.push(
      `tp=${resolved.tp} exceeds numKeyValueHeads=${model.numKeyValueHeads}: K/V projection weights and KV cache stop shrinking beyond tp=${model.numKeyValueHeads} (replicated instead of sharded further).`,
    );
  }

  return {
    scenario: 'inference',
    fidelity: 'simple',
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
