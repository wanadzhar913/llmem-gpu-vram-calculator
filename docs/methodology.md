# Methodology

All internal math works in bytes as plain `number`s (float64 is exact well beyond any realistic parameter count). Every scenario is built on one shared **operator tensor list** (`src/models/params.ts`): a per-tensor breakdown of the model (embedding, each layer's `q/k/v/o_proj` and FFN, norms, MoE experts/router, `lm_head`), each tagged with how it shards under TP/EP. Weight bytes, optimizer state, KV cache, and LoRA adapter sizing all walk this same list rather than re-deriving shapes independently.

## Fine-tuning, Granular fidelity (LLMem, §4–5)

Implemented (near-)verbatim from the paper, with `B16=2`, `B32=4` bytes, CUDA page `cu_p` (default 2 MiB), chunk size `cs` (default 32 MiB):

**Single-GPU peak:**
```
m_peak = m_base + m_p + m_os + m_out + m_lm

m_p   = ceil((embed_p + ceil(other_p / cs) * cs) * (B16 + B32) / cu_p) * cu_p
m_os  = Σ_{t ∈ {Embedding, Linear}} ceil(t_p * (B32 + B32) / cu_p) * cu_p
m_out = ceil((e_n + l_n) * (bs * sl * o_n) * B16 / cu_p) * cu_p
m_lm  = ceil(bs * sl * dict_n * B / cu_p) * cu_p
        + 2 * ceil(bs * (sl-1) * dict_n * B / cu_p) * cu_p
        + lm_p
```

**Multi-GPU peaks:**
```
CDP (conventional DP)  = m_peak (single-GPU, unsharded, replicated per GPU)
ADP (ZeRO-3)            = m_base + m_p16 + (m_p32 + m_os) / gpu_n + m_out + m_lm
TP (1D tensor-parallel)  = m_base + (m_p + m_os) / gpu_n + m_out + m_lm + m_back
DP+TP                    = m_peak^ADP - (m_p16 * tp_n) / gpu_n + m_back

m_back = ceil(l_n * (bs * sl * o_n) * (tp_n - 1)/tp_n * B16 / cu_p) * cu_p
```

This tool selects among these four by `(zeroStage, tp)`: `(0, 1)`→CDP, `(3, 1)`→ADP, `(0, tp>1)`→TP, `(3, tp>1)`→DP+TP. `zeroStage` 1/2 are rejected under `fidelity: 'granular'`. The paper's chunk manager shares gradient and parameter fp16 memory, which doesn't map onto sharding gradients independently of parameters (use `fidelity: 'simple'` for those).

Two spots need explicit interpretation the paper doesn't fully pin down, documented in code comments (`src/finetune.ts`):
- `cs` (chunk size) is dimensionally a parameter *count* in the equation but described in bytes in prose. This tool treats it as an element count (`chunkBytes / 2`, since chunks primarily hold fp16 params).
- The DP+TP formula reuses the symbol `gpu_n` for two different things across the ADP sub-term and the correction term; this tool reads the correction term's `gpu_n` as the total world size (`tp * dp`).

**Beyond the paper** (LLMem has no MoE or LoRA): MoE expert rows enter `other_p`/`m_os` like any Linear operator, pre-divided by the expert-parallel degree; LoRA/QLoRA freeze the base model (added as a flat, page-rounded, TP/EP-sharded term outside the paper's formulas) and route only synthesized adapter tensors through `m_p`/`m_os`.

## Fine-tuning: Simple fidelity

No chunk/page rounding; per-parameter byte costs, summed directly:

```
paramBytes    = trainableParams * (mixedPrecision ? 2 : 4)             # fp16/bf16 or fp32 compute copy
masterBytes   = trainableParams * (mixedPrecision ? 4 : 0)             # fp32 master weights
gradBytes     = trainableParams * (mixedPrecision ? 2 : 4)
optBytes      = trainableParams * optimizerBytesPerParam               # AdamW: 8, AdamW-8bit: 2, SGD: 4
frozenBytes   = frozenParams * frozenDtypeBytesPerParam                # LoRA/QLoRA only; NF4 ≈ 0.627 B/param
```

ZeRO stage 1/2/3 divides `optBytes` / `optBytes+gradBytes` / `optBytes+gradBytes+paramBytes+masterBytes` by the data-parallel degree, respectively. Activations use the Megatron-LM per-layer constant ([Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198), 34 elements/token/layer excluding the O(seq²) attention-score matrix, which flash-attention avoids materializing); gradient checkpointing replaces that with one boundary tensor per layer plus one full layer's recompute buffer.

## Inference

```
weights      = totalParams (after TP/EP/PP sharding) * bytesPerParam(weightDtype)
kvCache      = kvTensorsPerLayer * layersOnStage * kvDim * seqLen * batchSize * bytesPerParam(kvDtype) / kvShardDivisor
                 where kvDim = numKeyValueHeads * headDim (or an MLA override)
                       kvShardDivisor = min(tp, numKeyValueHeads)   # GQA/MQA can't shard past the KV head count
activations  ≈ batchSize * seqLen * hiddenSize * bytesPerParam(weightDtype) * activationFactor   # default factor 8
logits       = batchSize * (seqLen or 1) * vocabSize * bytesPerParam(weightDtype)   # full-sequence vs last-token
overhead     = baseOverheadBytes   # CUDA context + allocator, default 1 GiB, per GPU
```

Weights and KV cache are computed per pipeline stage (a stage only holds its own layers); logits only exist on the final stage (where `lm_head` lives).

## Parallelism

- **TP** shards attention (`q/k/v/o_proj`), dense FFN, embedding, and `lm_head` weights; `numAttentionHeads` must be divisible by `tp`.
- **K/V projections and KV cache** shard by `min(tp, numKeyValueHeads)`; once `tp` exceeds the KV head count, GQA/MQA models replicate rather than further shard.
- **PP** partitions layers into `ceil(numLayers / pp)`-layer contiguous stages; embedding lives on stage 0, `lm_head`/final norm on the last stage.
- **EP** shards MoE expert weights only, nested inside the DP group (`dp % ep === 0`); dense weights still replicate `dp` times, while experts replicate only `dp / ep` times.
- **DP** replicates for inference; for fine-tuning, the ZeRO stage determines what it additionally shards.

For known approximations and architectures the generic formulas don't fully capture, see [Known limitations](../README.md#known-limitations) in the README.
