import type { DType } from './units.js';

/** Fidelity of the underlying math. See README / plan for rationale. */
export type Fidelity = 'simple' | 'granular';

/** MoE-specific shape, layered on top of a dense `ModelConfig`. */
export interface MoEConfig {
  /** Total number of routed experts per MoE layer. */
  numExperts: number;
  /** Top-k experts activated per token. */
  expertsPerToken: number;
  /** FFN intermediate size of a single expert (usually smaller than the dense FFN). */
  expertIntermediateSize: number;
  /** Always-on shared experts (DeepSeek-style), in addition to the routed ones. */
  numSharedExperts?: number;
  /** Leading dense (non-MoE) transformer layers before MoE layers begin. */
  firstKDenseLayers?: number;
}

/** Architectural description of a transformer decoder model, dense or MoE. */
export interface ModelConfig {
  name: string;
  numLayers: number;
  hiddenSize: number;
  numAttentionHeads: number;
  /** Distinct KV heads. Equals numAttentionHeads for MHA, 1 for MQA, a divisor for GQA. */
  numKeyValueHeads: number;
  /** Defaults to hiddenSize / numAttentionHeads when omitted. */
  headDim?: number;
  /** Dense FFN intermediate size (ignored for MoE layers if `moe` is set). */
  ffnIntermediateSize: number;
  /** SwiGLU-style gated MLP uses 3 weight matrices per FFN instead of 2. */
  gatedMlp: boolean;
  vocabSize: number;
  maxPositionEmbeddings: number;
  /** Input and output embeddings share the same weight matrix. */
  tieWordEmbeddings: boolean;
  /**
   * Per-layer KV cache width override, for architectures whose cached state
   * isn't `numKeyValueHeads * headDim` (e.g. DeepSeek Multi-head Latent Attention,
   * which caches a compressed latent vector instead of full K/V).
   */
  kvCacheDimPerLayerOverride?: number;
  /** Distinct cached tensors per layer: 2 for standard K+V, 1 for an MLA latent. Default 2. */
  kvTensorsPerLayer?: number;
  moe?: MoEConfig;
}

/**
 * Tensor-/pipeline-/data-/expert-parallel degrees plus explicit GPU count.
 * `tp`, `pp`, `dp` default to 1 when omitted. If `numGpus` is given and `dp`
 * is not, `dp` is inferred as `numGpus / (tp * pp)`; if all of `tp`, `pp`,
 * `dp` are given, `numGpus` (when also given) must equal their product.
 * `ep` shards within the `dp` group (it does not multiply world size) and
 * only applies to MoE models.
 */
export interface ParallelismSpec {
  tp?: number;
  pp?: number;
  dp?: number;
  /** Expert-parallel degree. Only meaningful for MoE models; defaults to 1. */
  ep?: number;
  /** If given without `dp`, `dp` is inferred; if given with all of tp/pp/dp, it must match their product. */
  numGpus?: number;
}

export interface GPUSpec {
  id: string;
  name: string;
  memoryBytes: number;
}

/** One line item of a memory breakdown. */
export interface MemoryComponent {
  name: string;
  /** Bytes for this component across the whole world (all GPUs). */
  totalBytes: number;
  /** Bytes for this component on the single most-loaded GPU. */
  perGpuBytes: number;
  note?: string;
}

export interface MemoryBreakdown {
  scenario: 'inference' | 'finetune';
  fidelity: Fidelity;
  modelName: string;
  totalParams: number;
  activeParams?: number;
  parallelism: Required<Omit<ParallelismSpec, 'numGpus'>> & { numGpus: number };
  components: MemoryComponent[];
  totalBytes: number;
  /** Sum of perGpuBytes-eligible components on the single most-loaded GPU. */
  peakPerGpuBytes: number;
  warnings: string[];
}

export type WeightDType = DType;
export type KVDType = DType;

export interface InferenceInput {
  model: ModelConfig;
  parallelism: ParallelismSpec;
  seqLen: number;
  batchSize: number;
  weightDtype: DType;
  kvDtype: DType;
  /** Whether logits are materialized for the full sequence (prefill scoring) or just the last token (decode). */
  logitsMode?: 'lastToken' | 'fullSequence';
  /** Per-GPU CUDA context + allocator overhead. Default 1 GiB. */
  baseOverheadBytes?: number;
  /** Multiplier on activation memory accounting for concurrent live intermediates. Default 8. */
  activationFactor?: number;
  /** Fraction of GPU memory considered usable (vLLM `gpu_memory_utilization`-style haircut), 0-1. */
  gpuMemoryUtilization?: number;
}

export type FinetuneMethod = 'full' | 'lora' | 'qlora';
export type OptimizerKind = 'adamw' | 'adamw8bit' | 'sgd';
export type ZeroStage = 0 | 1 | 2 | 3;

export interface LoraConfig {
  rank: number;
  /** Which linear projections receive adapters. Default ['q_proj','k_proj','v_proj','o_proj']. */
  targetModules?: string[];
}

export interface FinetuneInput {
  model: ModelConfig;
  parallelism: ParallelismSpec;
  seqLen: number;
  batchSize: number;
  method: FinetuneMethod;
  optimizer: OptimizerKind;
  zeroStage: ZeroStage;
  gradientCheckpointing: boolean;
  /** Mixed precision (bf16/fp16 compute + fp32 master weights) vs full fp32. Default true. */
  mixedPrecision: boolean;
  lora?: LoraConfig;
  fidelity: Fidelity;
  /** Granular-fidelity only: chunk size for chunk-based memory management. Default 32 MiB. */
  chunkBytes?: number;
  /** Granular-fidelity only: CUDA page size for rounding. Default 2 MiB. */
  cudaPageBytes?: number;
  /** Per-GPU CUDA context + allocator overhead (`m_base`). Default 1 GiB. */
  baseOverheadBytes?: number;
}
