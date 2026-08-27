# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A TypeScript library/CLI/web-UI that estimates GPU VRAM usage for LLM inference (with KV cache), fine-tuning (full/LoRA/QLoRA), and MoE models, across TP/PP/DP/EP parallelism. Models come from a built-in preset catalog or from any HuggingFace `config.json` (`--hf-config` on the CLI, a paste/upload box in the web UI). Fine-tuning math implements **LLMem** (Kim et al. 2024, arXiv:2404.10933, PDF at `paper-notes/llmem/2404.10933v1.pdf`); MoE and inference/KV-cache math are this project's own extensions built on the same architecture. Zero runtime dependencies.

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

### Where a `ModelConfig` comes from

Three sources, all converging on `validateModelConfig()` in `src/models/resolve.ts`:

1. **A catalog preset** — `src/models/catalog.ts`, looked up by `findPreset()`.
2. **Field overrides** — `resolveModel({ preset, overrides })` shallow-merges them over a preset, or accepts them alone as a whole config. Note the merge is shallow, so an `moe` override replaces the preset's `moe` wholesale.
3. **A HuggingFace `config.json`** — `parseHfConfig()` in `src/models/hf.ts`.

`parseHfConfig()` is the only lenient one, and deliberately so: a `config.json` in the wild omits fields this tool needs, so anything absent gets a conventional default and **the assumption is pushed onto `warnings` rather than thrown**. Only `num_hidden_layers`, `hidden_size`, `num_attention_heads`, and `vocab_size` are hard errors. Those warnings are not decoration — they are the contract that makes leniency safe, so any new default added there must come with one. The function is pure (no `fs`, no DOM) because it bundles into the web app; file reading lives in `cli/main.ts`'s `loadModel()`.

When adding support for a new architecture family, the work usually splits: naming variants for existing concepts go in the key lists in `hf.ts` (e.g. `num_local_experts` / `n_routed_experts` / `num_experts` all mean `moe.numExperts`); genuinely new *shapes* need a `ModelConfig` field and a change to `buildOperatorList()`. Approximations the operator list can't express get a warning instead — see the MLA branch (`kv_lora_rank`) and the interleaved-MoE branch (`decoder_sparse_step`).

### Adding a new model preset

Add a `ModelConfig` to `src/models/catalog.ts` and register it in `MODEL_CATALOG`. Set `caveat` when the config is reconstructed from community reporting rather than a verified source, or when the architecture isn't fully captured by the generic dense-attention formula (e.g. MLA's low-rank q/kv projections, as documented on the `deepseek-v3` entry) — this caveat is surfaced automatically by the CLI/UI as a warning.

### Web UI (`web/`)

`web/index.html` carries all the markup *and* all the CSS inline; esbuild only bundles `web/app.ts`, so a style change needs no rebuild, just a reload. Type comes from Google Fonts (Archivo's width axis for display, IBM Plex Mono for every numeric); both have real fallback stacks, so an offline load degrades rather than breaks.

Three layout invariants worth knowing before editing that `<style>` block: vertical rhythm lives on `fieldset > * + *`, never on individual labels (a margin on a label lands asymmetrically across `.row`/`.row3` grid siblings, which is what once misaligned every two-column row); `label` is a flex column, so an inline element inside label text becomes its own flex line — keep label text a bare text node, and show a hidden label with `display: flex`, not `block`; and fieldsets are borderless with `float`ed legends, so whatever follows a `<legend>` needs `clear: both`.

#### The memory map

The readout's map is the one piece where `index.html` and `app.ts` share a contract. Every part of it — the reserve band, the over band, the excess overlay, the budget line, the capacity wall, the two axis marks — is a static element in the markup; `drawMap()` only sets `left`/`width` percentages on them. Adding a zone means adding the element in the HTML, not creating it in JS, which is what lets segments keep their identity between renders and animate.

The scale is `max(capacityBytes, allocation)`, deliberately: nothing is clamped to the track, so when a config overruns the card the wall moves left and the size of the overrun stays readable instead of every overrun pinning to a full bar. `estimateInference`/`estimateFinetune` decide fit against `capacity * utilization` (the amber budget line), not raw capacity — the band between budget and wall is the haircut, drawn as hatch.

The axis marks are placed by percentage, so they collide at narrow widths. `stackAxisMarks()` measures them and packs them greedily left to right, dropping any mark that would overlap onto the next line down via a `--row` custom property (the axis sizes itself from `--rows`) — marks are never hidden to resolve a collision. Because the packing depends on measured pixel widths, it is re-run on `resize` and on `document.fonts.ready`, since a webfont swap changes how wide a label measures. Keep mark text short for the same reason: the figures row above the map already carries the byte values, so a mark only has to name which line it points at.

The allocation palette lives in CSS as `--alloc-1`…`--alloc-7`; `app.ts` references it through `allocColor()` rather than holding hex values, so the map segments and the breakdown table's swatches stay keyed to each other and follow the light/dark theme. `--on-fill` is the foreground for anything sitting *on* one of those colours, and flips between themes. If you add an eighth component, add an `--alloc-8` and bump `ALLOC_SLOTS`.

### Testing conventions

`tests/finetune.test.ts` hand-derives exact LLMem golden values from a tiny synthetic model with `cudaPageBytes: 1` (disables page rounding, which otherwise swallows any small figure into a single 2 MiB page) and a small `chunkBytes` override, so the underlying arithmetic — not just page-rounded output — can be verified. Follow this pattern for new granular-fidelity formula changes: verify the exact math with rounding disabled, then add one realistic-scale test confirming the final figure is page-aligned.
