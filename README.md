# GPU Memory Calculator

A TypeScript library, CLI, and web UI for estimating GPU VRAM usage for LLM **inference** (with KV cache), **fine-tuning** (full / LoRA / QLoRA), and **MoE** models — with tensor-parallel (TP), pipeline-parallel (PP), data-parallel (DP), and expert-parallel (EP) specifications.

Fine-tuning memory math is based on **LLMem: Estimating GPU Memory Usage for Fine-Tuning Pre-Trained LLMs** (Kim et al., 2024) — [arXiv:2404.10933](https://arxiv.org/abs/2404.10933), included in this repo at [`docs/2404.10933v1.pdf`](docs/2404.10933v1.pdf). LLMem covers fine-tuning only; MoE and inference/KV-cache math are this tool's own analytic extensions, built on the same operator-list/parallelism architecture for consistency.

## Getting started

Requires Node.js ≥ 20.

```bash
npm install
npm run build       # compile src/ and cli/ to dist/
npm test             # run the vitest suite
```

### CLI

```bash
# list built-in model presets and GPUs
node dist/cli/main.js models
node dist/cli/main.js gpus

# inference: Llama 3.1 70B on 4x H100 with tensor parallelism
node dist/cli/main.js infer --model llama-3.1-70b --gpu h100-80 --tp 4 \
  --dtype fp16 --seq 8192 --batch 16

# MoE inference with expert parallelism
node dist/cli/main.js infer --model mixtral-8x7b --gpu a100-80 --tp 2 --ep 4 --gpus 8

# QLoRA fine-tune on a 24 GB card, quick analytic estimate
node dist/cli/main.js finetune --model llama-3.1-8b --method qlora --gpu rtx4090-24 \
  --seq 2048 --batch 4 --fidelity simple

# full fine-tuning with ZeRO-3 + tensor parallelism (LLMem's "DP+TP"), paper-faithful math
node dist/cli/main.js finetune --model llama-3.1-8b --method full --zero 3 \
  --tp 2 --dp 4 --gpu a100-80 --fidelity granular
```

Run `node dist/cli/main.js --help` for the full flag reference, including custom model overrides (`--num-layers`, `--hidden-size`, `--moe-experts`, ...) for models not in the built-in catalog.

### Web UI

```bash
npm run web    # esbuild + local static server on http://localhost:8080
```

Or build a static bundle without serving it:

```bash
npm run build:web   # writes web/dist/app.js
```

### Library

```ts
import { estimateInference, estimateFinetune, resolveModel, findGpu, checkFit, renderText } from './src/index.js';

const { config } = resolveModel({ preset: 'llama-3.1-8b' });
const breakdown = estimateInference({
  model: config,
  parallelism: { tp: 1 },
  seqLen: 8192,
  batchSize: 1,
  weightDtype: 'bf16',
  kvDtype: 'bf16',
});
console.log(renderText(breakdown, findGpu('h100-80')));
```

## Testing

```bash
npm test           # run once
npm run test:watch # watch mode
npm run typecheck  # tsc --noEmit
```

The test suite (`tests/`) checks:

- **Parameter counts** against published totals (Llama 3.1 8B ≈ 8.03B; Mixtral 8x7B ≈ 46.7B total / 12.9B active).
- **LLMem's exact equations** — a tiny hand-computed synthetic model verifies `m_p`, `m_p16`, `m_os`, `m_out`, `m_lm`, `m_back^tp`, and all four multi-GPU peak formulas (CDP/ADP/TP/DP+TP) against numbers worked out by hand (see `tests/finetune.test.ts` for the full derivation), plus the paper's own ordering invariant `CDP ≥ ADP` and `TP < CDP`.
- **KV cache** — a hand-computed golden (Llama 3.1 8B, 8192 context, batch 1, fp16 → exactly 1 GiB).
- **Sharding conservation** — per-GPU sharded params × shard divisor reconstructs the unsharded total; TP beyond `numKeyValueHeads` leaves K/V weights flat (the GQA replication cliff).
- **Solver round-trips** — `maxBatchSize`/`maxSeqLen` land at or just under GPU capacity, and one unit more doesn't fit.
- **Validation** — bad world sizes, indivisible head/expert counts, and unknown presets throw actionable errors.

## Formulas

All internal math works in bytes as plain `number`s (float64 is exact well beyond any realistic parameter count). Every scenario is built on one shared **operator tensor list** (`src/models/params.ts`): a per-tensor breakdown of the model (embedding, each layer's `q/k/v/o_proj` and FFN, norms, MoE experts/router, `lm_head`), each tagged with how it shards under TP/EP. Weight bytes, optimizer state, KV cache, and LoRA adapter sizing all walk this same list rather than re-deriving shapes independently.

### Fine-tuning — granular fidelity (LLMem, §4–5)

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

This tool selects among these four by `(zeroStage, tp)`: `(0, 1)`→CDP, `(3, 1)`→ADP, `(0, tp>1)`→TP, `(3, tp>1)`→DP+TP. `zeroStage` 1/2 are rejected under `fidelity: 'granular'` — the paper's chunk manager shares gradient and parameter fp16 memory, which doesn't map onto sharding gradients independently of parameters (use `fidelity: 'simple'` for those).

Two spots need explicit interpretation the paper doesn't fully pin down, documented in code comments (`src/finetune.ts`):
- `cs` (chunk size) is dimensionally a parameter *count* in the equation but described in bytes in prose — this tool treats it as an element count (`chunkBytes / 2`, since chunks primarily hold fp16 params).
- The DP+TP formula reuses the symbol `gpu_n` for two different things across the ADP sub-term and the correction term; this tool reads the correction term's `gpu_n` as the total world size (`tp * dp`).

**Beyond the paper** (LLMem has no MoE or LoRA): MoE expert rows enter `other_p`/`m_os` like any Linear operator, pre-divided by the expert-parallel degree; LoRA/QLoRA freeze the base model (added as a flat, page-rounded, TP/EP-sharded term outside the paper's formulas) and route only synthesized adapter tensors through `m_p`/`m_os`.

### Fine-tuning — simple fidelity

No chunk/page rounding; per-parameter byte costs, summed directly:

```
paramBytes    = trainableParams * (mixedPrecision ? 2 : 4)             # fp16/bf16 or fp32 compute copy
masterBytes   = trainableParams * (mixedPrecision ? 4 : 0)             # fp32 master weights
gradBytes     = trainableParams * (mixedPrecision ? 2 : 4)
optBytes      = trainableParams * optimizerBytesPerParam               # AdamW: 8, AdamW-8bit: 2, SGD: 4
frozenBytes   = frozenParams * frozenDtypeBytesPerParam                # LoRA/QLoRA only; NF4 ≈ 0.627 B/param
```

ZeRO stage 1/2/3 divides `optBytes` / `optBytes+gradBytes` / `optBytes+gradBytes+paramBytes+masterBytes` by the data-parallel degree, respectively. Activations use the Megatron-LM per-layer constant (Korthikanti et al. 2022, 34 elements/token/layer excluding the O(seq²) attention-score matrix, which flash-attention avoids materializing); gradient checkpointing replaces that with one boundary tensor per layer plus one full layer's recompute buffer.

### Inference

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

### Parallelism

- **TP** shards attention (`q/k/v/o_proj`), dense FFN, embedding, and `lm_head` weights; `numAttentionHeads` must be divisible by `tp`.
- **K/V projections and KV cache** shard by `min(tp, numKeyValueHeads)` — once `tp` exceeds the KV head count, GQA/MQA models replicate rather than further shard.
- **PP** partitions layers into `ceil(numLayers / pp)`-layer contiguous stages; embedding lives on stage 0, `lm_head`/final norm on the last stage.
- **EP** shards MoE expert weights only, nested inside the DP group (`dp % ep === 0`); dense weights still replicate `dp` times, while experts replicate only `dp / ep` times.
- **DP** replicates for inference; for fine-tuning, the ZeRO stage determines what it additionally shards.

## Known limitations

- **DeepSeek-V3** uses Multi-head Latent Attention (low-rank factorized q/k/v projections); this tool's generic full-rank attention formula overestimates its attention parameter count. Its KV cache size *is* accurate (via `kvCacheDimPerLayerOverride`).
- Several presets (Qwen3, GPT-OSS) are reconstructed from community-reported configs rather than verified originals — flagged with a `caveat` in the catalog and surfaced as a CLI/UI warning.
- Frozen LoRA/QLoRA base weights are modeled as replicated per data-parallel rank, not further sharded by ZeRO/FSDP — a common real configuration, but FSDP-sharded frozen backbones aren't modeled.
- Activation memory (both fidelities) assumes flash-attention-style processing with no O(seq²) attention-score matrix resident.

## Project layout

```
src/            core library (types, units, model/GPU catalogs, parallelism, inference, finetune, advise, report)
cli/main.ts     `gpumem infer|finetune|models|gpus`
web/            static HTML/TS UI, bundled by esbuild
tests/          vitest suite
docs/           the LLMem paper (2404.10933v1.pdf)
```
