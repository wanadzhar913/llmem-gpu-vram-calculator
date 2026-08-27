#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { estimateInference } from '../src/inference.js';
import { estimateFinetune } from '../src/finetune.js';
import { resolveModel, type ModelOverrides } from '../src/models/resolve.js';
import { parseHfConfig } from '../src/models/hf.js';
import { MODEL_CATALOG } from '../src/models/catalog.js';
import { GPU_CATALOG, findGpu } from '../src/gpus/catalog.js';
import { renderText, renderJson } from '../src/report.js';
import { isDType, type DType } from '../src/units.js';
import type { FinetuneMethod, OptimizerKind, ParallelismSpec, ZeroStage, Fidelity } from '../src/types.js';

type Flags = Record<string, string | boolean>;

function parseArgs(argv: string[]): { positional: string[]; flags: Flags } {
  const positional: string[] = [];
  const flags: Flags = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg.startsWith('--')) {
      const eq = arg.indexOf('=');
      if (eq !== -1) {
        flags[arg.slice(2, eq)] = arg.slice(eq + 1);
        continue;
      }
      const key = arg.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('--')) {
        flags[key] = true;
      } else {
        flags[key] = next;
        i++;
      }
    } else {
      positional.push(arg);
    }
  }
  return { positional, flags };
}

function str(flags: Flags, key: string): string | undefined {
  const v = flags[key];
  return typeof v === 'string' ? v : undefined;
}

function num(flags: Flags, key: string): number | undefined {
  const v = str(flags, key);
  return v === undefined ? undefined : Number(v);
}

function bool(flags: Flags, key: string): boolean {
  return flags[key] === true || flags[key] === 'true';
}

/**
 * Matches the web UI's 75% default: leaves room for the CUDA context,
 * activation workspace, and fragmentation this tool doesn't model.
 */
const DEFAULT_GPU_MEMORY_UTILIZATION = 0.75;

/**
 * `--util` is a fraction, not a percentage. Anything outside (0, 1] is rejected
 * rather than clamped — a value above 1 is a typo (`--util 75` meaning 75%, or
 * `--util 5` meaning `0.5`), and silently treating it as 100% would hide the
 * mistake behind a fit verdict that looks perfectly plausible.
 */
function utilization(flags: Flags): number {
  const raw = num(flags, 'util');
  if (raw === undefined) return DEFAULT_GPU_MEMORY_UTILIZATION;
  if (!Number.isFinite(raw) || raw <= 0 || raw > 1) {
    throw new Error(
      `--util must be a fraction in (0, 1] — e.g. 0.9 for 90% — but got "${str(flags, 'util')}".`,
    );
  }
  return raw;
}

function dtype(flags: Flags, key: string, fallback: DType): DType {
  const v = str(flags, key);
  if (v === undefined) return fallback;
  if (!isDType(v)) throw new Error(`Unknown dtype "${v}" for --${key}. Valid: fp32, bf16, fp16, fp8, int8, int4, nf4.`);
  return v;
}

function parallelismFromFlags(flags: Flags): ParallelismSpec {
  return {
    tp: num(flags, 'tp'),
    pp: num(flags, 'pp'),
    dp: num(flags, 'dp'),
    ep: num(flags, 'ep'),
    numGpus: num(flags, 'gpus'),
  };
}

function modelOverridesFromFlags(flags: Flags): ModelOverrides {
  const overrides: ModelOverrides = {};
  if (str(flags, 'name')) overrides.name = str(flags, 'name')!;
  if (num(flags, 'num-layers') !== undefined) overrides.numLayers = num(flags, 'num-layers')!;
  if (num(flags, 'hidden-size') !== undefined) overrides.hiddenSize = num(flags, 'hidden-size')!;
  if (num(flags, 'num-heads') !== undefined) overrides.numAttentionHeads = num(flags, 'num-heads')!;
  if (num(flags, 'num-kv-heads') !== undefined) overrides.numKeyValueHeads = num(flags, 'num-kv-heads')!;
  if (num(flags, 'head-dim') !== undefined) overrides.headDim = num(flags, 'head-dim')!;
  if (num(flags, 'ffn-size') !== undefined) overrides.ffnIntermediateSize = num(flags, 'ffn-size')!;
  if (flags['gated-mlp'] !== undefined) overrides.gatedMlp = bool(flags, 'gated-mlp');
  if (num(flags, 'vocab-size') !== undefined) overrides.vocabSize = num(flags, 'vocab-size')!;
  if (num(flags, 'max-pos') !== undefined) overrides.maxPositionEmbeddings = num(flags, 'max-pos')!;
  if (flags['tie-embeddings'] !== undefined) overrides.tieWordEmbeddings = bool(flags, 'tie-embeddings');
  if (num(flags, 'moe-experts') !== undefined) {
    overrides.moe = {
      numExperts: num(flags, 'moe-experts')!,
      expertsPerToken: num(flags, 'moe-topk') ?? 2,
      expertIntermediateSize: num(flags, 'moe-expert-size') ?? overrides.ffnIntermediateSize ?? 0,
    };
  }
  return overrides;
}

interface LoadedModel {
  config: ReturnType<typeof resolveModel>['config'];
  caveat?: string;
  /** Assumptions the HuggingFace parser had to make, surfaced as warnings. */
  warnings: string[];
}

/**
 * Resolves the model from `--hf-config` (a HuggingFace config.json on disk) or
 * `--model` (a catalog preset), with the per-field `--hidden-size`-style flags
 * layered on top of either. The two sources are mutually exclusive: taking a
 * preset *and* a config file would silently discard one of them.
 */
function loadModel(flags: Flags): LoadedModel {
  const preset = str(flags, 'model');
  const hfPath = str(flags, 'hf-config');
  const overrides = modelOverridesFromFlags(flags);

  if (hfPath === undefined) {
    const { config, caveat } = resolveModel({ preset, overrides });
    return { config, caveat, warnings: [] };
  }

  if (preset !== undefined) {
    throw new Error('--hf-config and --model are mutually exclusive. Pass one or the other.');
  }

  let text: string;
  try {
    text = readFileSync(hfPath, 'utf8');
  } catch (err) {
    throw new Error(`Could not read --hf-config "${hfPath}": ${err instanceof Error ? err.message : String(err)}`);
  }

  const parsed = parseHfConfig(text);
  // Re-run resolveModel so flag overrides merge and validation applies exactly
  // as they do for a preset.
  const { config } = resolveModel({ overrides: { ...parsed.config, ...overrides } });
  return { config, warnings: parsed.warnings };
}

function loadGpu(flags: Flags) {
  const gpuId = str(flags, 'gpu');
  if (!gpuId) return undefined;
  const gpu = findGpu(gpuId);
  if (!gpu) throw new Error(`Unknown GPU "${gpuId}". Run \`gpumem gpus\` to list available GPUs.`);
  return gpu;
}

function printOut(text: string) {
  process.stdout.write(text + '\n');
}

function cmdInfer(flags: Flags) {
  const { config, caveat, warnings } = loadModel(flags);
  const gpu = loadGpu(flags);
  const util = utilization(flags);
  const breakdown = estimateInference({
    model: config,
    parallelism: parallelismFromFlags(flags),
    seqLen: num(flags, 'seq') ?? 4096,
    batchSize: num(flags, 'batch') ?? 1,
    weightDtype: dtype(flags, 'dtype', 'bf16'),
    kvDtype: dtype(flags, 'kv-dtype', dtype(flags, 'dtype', 'bf16')),
    logitsMode: (str(flags, 'logits-mode') as 'lastToken' | 'fullSequence') ?? 'lastToken',
  });
  breakdown.warnings.unshift(...warnings);
  if (caveat) breakdown.warnings.unshift(`Preset caveat: ${caveat}`);
  printOut(bool(flags, 'json') ? renderJson(breakdown, gpu, util) : renderText(breakdown, gpu, util));
}

function cmdFinetune(flags: Flags) {
  const { config, caveat, warnings } = loadModel(flags);
  const gpu = loadGpu(flags);
  const util = utilization(flags);
  const method = (str(flags, 'method') ?? 'full') as FinetuneMethod;
  const fidelity = (str(flags, 'fidelity') ?? 'simple') as Fidelity;
  const rank = num(flags, 'lora-rank');
  const breakdown = estimateFinetune({
    model: config,
    parallelism: parallelismFromFlags(flags),
    seqLen: num(flags, 'seq') ?? 2048,
    batchSize: num(flags, 'batch') ?? 1,
    method,
    optimizer: (str(flags, 'optimizer') ?? 'adamw') as OptimizerKind,
    zeroStage: (num(flags, 'zero') ?? 0) as ZeroStage,
    gradientCheckpointing: bool(flags, 'grad-checkpointing'),
    mixedPrecision: !bool(flags, 'no-mixed-precision'),
    lora:
      method !== 'full'
        ? {
            rank: rank ?? 16,
            targetModules: str(flags, 'lora-targets')?.split(','),
          }
        : undefined,
    fidelity,
    chunkBytes: num(flags, 'chunk-bytes'),
    baseOverheadBytes: num(flags, 'base-overhead-bytes'),
  });
  breakdown.warnings.unshift(...warnings);
  if (caveat) breakdown.warnings.unshift(`Preset caveat: ${caveat}`);
  printOut(bool(flags, 'json') ? renderJson(breakdown, gpu, util) : renderText(breakdown, gpu, util));
}

function cmdModels() {
  for (const p of MODEL_CATALOG) {
    const moe = p.config.moe ? ` [MoE ${p.config.moe.numExperts}x, top-${p.config.moe.expertsPerToken}]` : '';
    printOut(`${p.id}${moe}${p.caveat ? `  (${p.caveat})` : ''}`);
  }
}

function cmdGpus() {
  for (const g of GPU_CATALOG) {
    printOut(`${g.id}\t${g.name}\t${(g.memoryBytes / 1024 / 1024 / 1024).toFixed(0)} GiB`);
  }
}

function printHelp() {
  printOut(`gpumem — GPU VRAM calculator for inference, fine-tuning and MoE models.

Usage:
  gpumem infer (--model <id> | --hf-config <path>) [--tp N --pp N --dp N --ep N | --gpus N] [--gpu <id>]
               [--dtype <dtype>] [--kv-dtype <dtype>] [--seq N] [--batch N]
               [--logits-mode lastToken|fullSequence] [--util 0-1] [--json]

  gpumem finetune (--model <id> | --hf-config <path>) [--tp N --pp N --dp N | --gpus N] [--gpu <id>]
                  [--method full|lora|qlora] [--zero 0|1|2|3] [--optimizer adamw|adamw8bit|sgd]
                  [--grad-checkpointing] [--no-mixed-precision]
                  [--lora-rank N] [--lora-targets q_proj,k_proj,...]
                  [--fidelity simple|granular] [--seq N] [--batch N] [--util 0-1] [--json]

  gpumem models    List built-in model presets
  gpumem gpus      List built-in GPU capacities

Fit verdict:
  --gpu <id>           Compare the peak against a GPU's capacity.
  --util 0-1           Fraction of that capacity treated as usable, vLLM-style.
                       Defaults to 0.75; values outside (0, 1] are rejected.

Models outside the built-in catalog:
  --hf-config <path>   Read a HuggingFace config.json and derive the shape from it.
                       Fields it omits are filled with conventional defaults and
                       reported as warnings. Mutually exclusive with --model.

Custom model overrides (combine with --model, --hf-config, or neither):
  --num-layers --hidden-size --num-heads --num-kv-heads --head-dim --ffn-size
  --gated-mlp --vocab-size --max-pos --tie-embeddings
  --moe-experts --moe-topk --moe-expert-size
`);
}

function main() {
  const [, , command, ...rest] = process.argv;
  const { flags } = parseArgs(rest);
  try {
    switch (command) {
      case 'infer':
        cmdInfer(flags);
        break;
      case 'finetune':
        cmdFinetune(flags);
        break;
      case 'models':
        cmdModels();
        break;
      case 'gpus':
        cmdGpus();
        break;
      case undefined:
      case '-h':
      case '--help':
        printHelp();
        break;
      default:
        process.stderr.write(`Unknown command "${command}".\n\n`);
        printHelp();
        process.exitCode = 1;
    }
  } catch (err) {
    process.stderr.write(`Error: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exitCode = 1;
  }
}

main();
