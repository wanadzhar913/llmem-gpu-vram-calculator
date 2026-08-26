# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A TypeScript library/CLI/web-UI that estimates GPU VRAM usage for LLM inference (with KV cache), fine-tuning (full/LoRA/QLoRA), and MoE models, across TP/PP/DP/EP parallelism. Fine-tuning math implements **LLMem** (Kim et al. 2024, arXiv:2404.10933, PDF at `docs/2404.10933v1.pdf`); MoE and inference/KV-cache math are this project's own extensions built on the same architecture. Zero runtime dependencies.

## Commands

```bash
npm install
npm run build       # tsc -> dist/
npm run typecheck   # tsc --noEmit
npm test             # vitest run (full suite)
npm run test:watch   # vitest watch mode
npm run build:web    # esbuild bundle -> web/dist/app.js
npm run web           # esbuild + local static server on :8080, watches for changes
```

Run a single test file: `npx vitest run tests/finetune.test.ts`
Run a single test by name: `npx vitest run -t "CDP (zeroStage=0"`

CLI (after `npm run build`): `node dist/cli/main.js infer|finetune|models|gpus --help`

There is no lint script configured; `npm run typecheck` is the correctness gate along with the test suite.

## Architecture

### The operator tensor list is the single source of truth

`src/models/params.ts`'s `buildOperatorList()` walks a `ModelConfig` and produces one flat list of `OperatorTensor` rows — `embed_tokens`, each layer's `q_proj`/`k_proj`/`v_proj`/`o_proj`/norms/FFN (or `mlp.experts`/`mlp.router`/`mlp.shared_experts` for MoE layers), the final norm, and `lm_head` if untied. Each row carries a `shardable` tag (`'tp' | 'tp_kv' | 'ep' | 'none'`) and, for attention projections, `inFeatures`/`outFeatures`.

Every downstream calculation — weight bytes, optimizer state, KV cache dims, TP/EP sharding, LoRA adapter sizing — walks this same list rather than re-deriving shapes independently. When changing how a tensor is shaped or sharded, change it once here; `inference.ts`, `finetune.ts`, and `parallelism.ts` all consume it.

`k_proj`/`v_proj` use `shardable: 'tp_kv'`, not `'tp'`: GQA/MQA models can't split a single KV head across ranks, so those weights (and the KV cache they produce) shard by `min(tp, numKeyValueHeads)` and are replicated, not further divided, once `tp` exceeds the KV head count. `src/parallelism.ts`'s `kvShardDivisor()` implements this and both `inference.ts` and `finetune.ts` surface it as a warning when triggered.

### Parallelism resolution and world-total accounting (`src/parallelism.ts`)

`resolveParallelism()` fills in `tp`/`pp`/`dp`/`ep` defaults, infers `dp` from `numGpus` when omitted, and validates against the model (head divisibility, expert divisibility). `ep` nests *inside* `dp` (`dp % ep === 0`) rather than multiplying world size — expert weights replicate `dp / ep` times while dense weights still replicate `dp` times. `worldReplicationFactor(op, resolved)` encodes this per-operator and is how both `inference.ts` and `finetune.ts` compute "total bytes across the whole world" from a single per-GPU figure, without walking every stage/rank explicitly.

`distributeAcrossPipeline()` assigns each operator to a pipeline stage (embedding → stage 0, `lm_head`/final norm → last stage, layers split into contiguous `ceil(numLayers/pp)`-sized blocks) and is shared by both `inference.ts` and `finetune.ts` to compute per-stage peaks and pick the most-loaded stage.

### Two fine-tuning fidelities, one entry point (`src/finetune.ts`)

`estimateFinetune()` dispatches per pipeline stage to either `granularStagePeak()` (LLMem-faithful: chunk-size quantization, 2 MiB CUDA-page rounding, LLMem's exact `m_p`/`m_os`/`m_out`/`m_lm`/`m_back` and the CDP/ADP/TP/DP+TP mode selection by `(zeroStage, tp)`) or `simpleStagePeak()` (unrounded byte-count model, supports ZeRO stages 1/2 which the granular path rejects — LLMem's chunk manager shares param/gradient fp16 memory, so per-paper math can't shard gradients independently of params).

LoRA/QLoRA are not in the paper; both fidelities extend it the same way: base weights become a flat frozen term (page-rounded but sharded by TP/EP only, never further by ZeRO/dp — documented as a simplification), and only synthesized adapter tensors (`loraAdapterOperators()`, sized `rank * (inFeatures + outFeatures)`) flow through the trainable-weight/optimizer-state formulas.

World-total byte figures (as opposed to the fidelity-accurate peak-per-GPU figures) are intentionally computed with a simpler, fidelity-independent byte model — see the comment above `worldTrainableBytes` in `estimateFinetune()`. Don't expect world totals to be chunk/page-exact; only `peakPerGpuBytes` carries that precision.

Two places in LLMem's own equations are genuinely underspecified and required an explicit interpretation choice, documented inline where implemented: the chunk size `cs` is dimensionally a parameter count despite being described in bytes (treated here as `chunkBytes / 2`), and the DP+TP formula's `gpu_n` symbol is reused for two different quantities across the ADP sub-term and the correction term (read here as the correction term using total world size `tp * dp`).

### Adding a new model preset

Add a `ModelConfig` to `src/models/catalog.ts` and register it in `MODEL_CATALOG`. Set `caveat` when the config is reconstructed from community reporting rather than a verified source, or when the architecture isn't fully captured by the generic dense-attention formula (e.g. MLA's low-rank q/kv projections, as documented on the `deepseek-v3` entry) — this caveat is surfaced automatically by the CLI/UI as a warning.

### Testing conventions

`tests/finetune.test.ts` hand-derives exact LLMem golden values from a tiny synthetic model with `cudaPageBytes: 1` (disables page rounding, which otherwise swallows any small figure into a single 2 MiB page) and a small `chunkBytes` override, so the underlying arithmetic — not just page-rounded output — can be verified. Follow this pattern for new granular-fidelity formula changes: verify the exact math with rounding disabled, then add one realistic-scale test confirming the final figure is page-aligned.
