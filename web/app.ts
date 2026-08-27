import { MODEL_CATALOG } from '../src/models/catalog.js';
import { parseHfConfig } from '../src/models/hf.js';
import { countParams } from '../src/models/params.js';
import { GPU_CATALOG, findGpu } from '../src/gpus/catalog.js';
import { estimateInference } from '../src/inference.js';
import { estimateFinetune } from '../src/finetune.js';
import { checkFit } from '../src/advise.js';
import { formatBytes } from '../src/units.js';
import { DTYPES, type DType } from '../src/units.js';
import type { FinetuneMethod, Fidelity, GPUSpec, MemoryBreakdown, ModelConfig, OptimizerKind, ZeroStage } from '../src/types.js';

type Scenario = 'inference' | 'finetune';

/** Sentinel option value that swaps the preset dropdown for the paste box. */
const CUSTOM_MODEL_ID = '__custom__';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

/**
 * The allocation ramp lives in the stylesheet as `--alloc-1..N` so the map and
 * the breakdown table key off one palette that follows the page's theme.
 */
const ALLOC_SLOTS = 7;
const allocColor = (i: number) => `var(--alloc-${(i % ALLOC_SLOTS) + 1})`;

let scenario: Scenario = 'inference';

/** The most recently parsed pasted config. Session-only; never persisted. */
let customModel: { config: ModelConfig; warnings: string[] } | null = null;

function populateSelect(select: HTMLSelectElement, items: { value: string; label: string }[]) {
  select.innerHTML = '';
  for (const item of items) {
    const opt = document.createElement('option');
    opt.value = item.value;
    opt.textContent = item.label;
    select.appendChild(opt);
  }
}

function init() {
  populateSelect($('model-preset'), [
    ...MODEL_CATALOG.map((p) => ({
      value: p.id,
      label: p.config.moe ? `${p.id} (MoE ${p.config.moe.numExperts}x, top-${p.config.moe.expertsPerToken})` : p.id,
    })),
    { value: CUSTOM_MODEL_ID, label: '— Paste a config.json —' },
  ]);
  populateSelect(
    $('weight-dtype'),
    DTYPES.map((d) => ({ value: d, label: d })),
  );
  populateSelect(
    $('kv-dtype'),
    DTYPES.map((d) => ({ value: d, label: d })),
  );
  ($('weight-dtype') as HTMLSelectElement).value = 'bf16';
  ($('kv-dtype') as HTMLSelectElement).value = 'bf16';
  populateSelect(
    $('gpu-select'),
    GPU_CATALOG.map((g) => ({ value: g.id, label: `${g.name} (${Math.round(g.memoryBytes / 1024 ** 3)} GiB)` })),
  );

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      scenario = (tab as HTMLElement).dataset['scenario'] as Scenario;
      $('inference-fields').style.display = scenario === 'inference' ? 'block' : 'none';
      $('finetune-fields').style.display = scenario === 'finetune' ? 'block' : 'none';
      recompute();
    });
  });

  $<HTMLSelectElement>('ft-method').addEventListener('change', () => {
    const method = $<HTMLSelectElement>('ft-method').value as FinetuneMethod;
    $('lora-fields').style.display = method === 'full' ? 'none' : 'block';
    recompute();
  });

  $<HTMLInputElement>('gpu-util').addEventListener('change', normalizeUtilizationField);
  $<HTMLInputElement>('gpu-util').addEventListener('blur', normalizeUtilizationField);

  $<HTMLSelectElement>('model-preset').addEventListener('change', () => {
    const custom = $<HTMLSelectElement>('model-preset').value === CUSTOM_MODEL_ID;
    $('hf-fields').style.display = custom ? 'block' : 'none';
    if (custom) parsePastedConfig();
  });

  // The textarea reparses on a debounce so a multi-hundred-line paste isn't
  // re-estimated on every keystroke; everything else recomputes immediately.
  let debounce: ReturnType<typeof setTimeout> | undefined;
  $<HTMLTextAreaElement>('hf-config').addEventListener('input', () => {
    if (debounce !== undefined) clearTimeout(debounce);
    debounce = setTimeout(() => {
      parsePastedConfig();
      recompute();
    }, 250);
  });

  // Uploading fills the same textarea, so the file and paste paths converge on
  // one parse and the user can edit what they uploaded.
  $<HTMLInputElement>('hf-file').addEventListener('change', async (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const status = $('hf-status');
    try {
      $<HTMLTextAreaElement>('hf-config').value = await file.text();
    } catch (err) {
      status.className = 'summary bad';
      status.textContent = `Could not read ${file.name}: ${err instanceof Error ? err.message : String(err)}`;
      return;
    }
    parsePastedConfig();
    recompute();
  });

  document.querySelectorAll('input, select').forEach((el) => {
    el.addEventListener('input', recompute);
    el.addEventListener('change', recompute);
  });

  // Which axis marks share a row depends on their pixel widths, so the packing
  // has to be redone whenever the track resizes or the webfonts swap in and
  // change how wide the labels measure.
  let resizeFrame: number | undefined;
  window.addEventListener('resize', () => {
    if (resizeFrame !== undefined) cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(restackAxis);
  });
  document.fonts?.ready.then(restackAxis);

  recompute();
}

function restackAxis() {
  if ($('map').hidden) return;
  stackAxisMarks([$('axis-zero'), $('axis-budget'), $('axis-capacity')]);
}

/**
 * Reads the paste box into `customModel`, reporting either a one-line shape
 * summary or the parse error into `#hf-status`. Never throws: a half-typed
 * config should leave the previous results on screen rather than blanking them.
 */
function parsePastedConfig() {
  const status = $('hf-status');
  const text = $<HTMLTextAreaElement>('hf-config').value;
  if (text.trim().length === 0) {
    customModel = null;
    status.className = 'note';
    status.textContent = '';
    return;
  }
  try {
    const parsed = parseHfConfig(text);
    customModel = parsed;
    const { totalParams, activeParams } = countParams(parsed.config);
    const active =
      activeParams !== totalParams ? ` (${(activeParams / 1e9).toFixed(2)}B active/token)` : '';
    status.className = 'note';
    status.innerHTML =
      `<b>${parsed.config.name}</b> &middot; ${parsed.config.numLayers} layers &middot; ` +
      `${(totalParams / 1e9).toFixed(2)}B params${active}`;
  } catch (err) {
    customModel = null;
    status.className = 'note bad';
    status.textContent = err instanceof Error ? err.message : String(err);
  }
}

interface SelectedModel {
  config: ModelConfig;
  caveat?: string;
  /** Parser assumptions, prepended to the breakdown's own warnings. */
  warnings: string[];
}

function currentModel(): SelectedModel {
  const id = $<HTMLSelectElement>('model-preset').value;
  if (id === CUSTOM_MODEL_ID) {
    if (!customModel) throw new Error('Paste a HuggingFace config.json to estimate a custom model.');
    return { config: customModel.config, warnings: customModel.warnings };
  }
  const preset = MODEL_CATALOG.find((p) => p.id === id);
  if (!preset) throw new Error('no preset selected');
  return { config: preset.config, caveat: preset.caveat, warnings: [] };
}

function currentParallelism() {
  return {
    tp: Number($<HTMLInputElement>('tp').value) || 1,
    pp: Number($<HTMLInputElement>('pp').value) || 1,
    dp: Number($<HTMLInputElement>('dp').value) || 1,
    ep: Number($<HTMLInputElement>('ep').value) || 1,
  };
}

function num(id: string): number {
  return Number($<HTMLInputElement>(id).value) || 0;
}

/**
 * Bounds for the usable-memory field, as percentages. Anything above 100 would
 * have the estimate claim more memory than the GPU physically has; the default
 * mirrors vLLM's own `gpu_memory_utilization` haircut, leaving room for the
 * CUDA context, activation workspace, and fragmentation this tool doesn't model.
 */
const UTIL_MIN_PERCENT = 1;
const UTIL_MAX_PERCENT = 100;
const UTIL_DEFAULT_PERCENT = 75;

/**
 * The usable-memory field is a percentage, but `min`/`max` on a number input
 * only gate the spinner — a typed or pasted value sails past them. Clamp to
 * (0, 100] for the estimate; an empty or nonsensical field falls back to the
 * default rather than blanking the results mid-edit.
 */
function currentUtilization(): number {
  const raw = Number($<HTMLInputElement>('gpu-util').value);
  if (!Number.isFinite(raw) || raw <= 0) return UTIL_DEFAULT_PERCENT / 100;
  return Math.min(raw, UTIL_MAX_PERCENT) / 100;
}

/**
 * Snaps the field itself back into range once the edit is finished, so it can
 * never sit showing a percentage the estimate refused to use. Deliberately on
 * `change`/`blur` rather than `input`: rewriting the value on every keystroke
 * fights someone clearing the box to retype it.
 */
function normalizeUtilizationField() {
  const el = $<HTMLInputElement>('gpu-util');
  const raw = Number(el.value);
  const clamped =
    !Number.isFinite(raw) || raw <= 0
      ? UTIL_DEFAULT_PERCENT
      : Math.min(Math.max(raw, UTIL_MIN_PERCENT), UTIL_MAX_PERCENT);
  if (el.value !== String(clamped)) {
    el.value = String(clamped);
    recompute();
  }
}

function renderError(message: string) {
  $('readout').dataset['state'] = 'error';
  $('verdict-word').textContent = "Can't estimate";
  $('verdict-figures').innerHTML = '';
  $('verdict-note').textContent = message;
  $('summary').textContent = '';
  $('map').hidden = true;
  $('table-body').innerHTML = '';
  $('warnings').innerHTML = '';
}

const figure = (label: string, value: string) => `<div><dt>${label}</dt><dd>${value}</dd></div>`;

/** Catalog names read "NVIDIA H100 80GB"; the map axis has room for the model only. */
const shortGpuName = (name: string) => name.replace(/^(NVIDIA|AMD) /, '');

function place(el: HTMLElement, leftPercent: number, widthPercent: number) {
  el.style.left = `${leftPercent}%`;
  el.style.width = `${Math.max(0, widthPercent)}%`;
}

/**
 * Component names last drawn. Segments are reused while the shape of the
 * breakdown is unchanged, so tweaking an input animates the blocks that moved
 * instead of replacing the whole map.
 */
let mapKey = '';

/**
 * Draws the card's memory as one scale diagram: the allocation from the left,
 * the budget line that actually decides the verdict, the physical capacity
 * wall, and hatching wherever the allocation has run past either.
 *
 * Nothing is clamped to the track. When the allocation exceeds the card the
 * scale grows to fit it and the wall moves left, so a 12 GiB overrun and an
 * 80 GiB overrun stay visibly different instead of both pinning to a full bar.
 */
function drawMap(breakdown: MemoryBreakdown, gpu: GPUSpec, usableBytes: number) {
  $('map').hidden = false;

  const capacity = gpu.memoryBytes;
  const alloc = breakdown.components.reduce((sum, c) => sum + c.perGpuBytes, 0);
  const scaleMax = Math.max(capacity, alloc) || 1;
  const at = (bytes: number) => (bytes / scaleMax) * 100;

  const segs = $('map-segs');
  const key = breakdown.components.map((c) => c.name).join('|');
  if (key !== mapKey) {
    segs.innerHTML = breakdown.components
      .map((_, i) => `<div class="map-seg" style="background:${allocColor(i)}"><span></span></div>`)
      .join('');
    mapKey = key;
  }

  let cursor = 0;
  breakdown.components.forEach((c, i) => {
    const el = segs.children[i] as HTMLElement;
    const width = at(c.perGpuBytes);
    place(el, at(cursor), width);
    el.title = `${c.name}: ${formatBytes(c.perGpuBytes)}`;
    // Only blocks with room for it carry their own name; the rest are keyed by
    // the swatches in the breakdown table.
    el.firstElementChild!.textContent = width > 11 ? c.name : '';
    cursor += c.perGpuBytes;
  });

  // The haircut you deliberately left, then anything past the physical card.
  place($('map-reserve'), at(Math.min(usableBytes, capacity)), at(capacity - Math.min(usableBytes, capacity)));
  place($('map-over'), at(capacity), at(Math.max(capacity, alloc) - capacity));
  // The part of the allocation that broke the budget, drawn over the segments.
  place($('map-excess'), at(Math.min(usableBytes, alloc)), at(alloc - Math.min(usableBytes, alloc)));

  $('map-budget').style.left = `${at(usableBytes)}%`;
  $('map-wall').style.left = `${at(capacity)}%`;
  // At capacity the wall would sit on the track's own right edge, which
  // already reads as the boundary.
  $('map-wall').hidden = capacity >= scaleMax;

  const budgetPct = at(usableBytes);
  const capacityPct = at(capacity);
  const axisBudget = $('axis-budget');
  const axisCapacity = $('axis-capacity');

  // The marks name which line is which; the figures above carry the values, so
  // repeating a byte count here would only be the same number twice.
  axisBudget.style.left = `${budgetPct}%`;
  axisBudget.dataset['anchor'] = budgetPct < 8 ? 'start' : 'mid';
  axisBudget.textContent = 'budget';

  axisCapacity.style.left = `${capacityPct}%`;
  axisCapacity.dataset['anchor'] = capacityPct > 99 ? 'end' : capacityPct < 8 ? 'start' : 'mid';
  // The catalog name already ends in the capacity ("H100 80GB"), so it states
  // the wall's size and its identity in one mark.
  axisCapacity.textContent = shortGpuName(gpu.name);

  restackAxis();
}

/** Clear space between two marks sharing a row, in px. */
const AXIS_MARK_GAP = 12;

/**
 * Percentage-placed marks collide at narrow widths. Rather than hiding one,
 * walk them left to right and drop any that would overlap onto the next line
 * down — so every mark survives at every viewport width, and the axis only
 * grows as tall as it actually needs.
 */
function stackAxisMarks(marks: HTMLElement[]) {
  const visible = marks.filter((el) => !el.hidden);
  // Row assignment moves a mark vertically only, so one measurement up front
  // gives horizontal extents that stay valid as rows are handed out.
  const extents = visible.map((el) => {
    el.style.setProperty('--row', '1');
    return el.getBoundingClientRect();
  });

  const rowRightEdges: number[] = [];
  visible.forEach((el, i) => {
    const { left, right } = extents[i]!;
    let row = rowRightEdges.findIndex((edge) => left >= edge + AXIS_MARK_GAP);
    if (row === -1) row = rowRightEdges.push(-Infinity) - 1;
    rowRightEdges[row] = right;
    el.style.setProperty('--row', String(row + 1));
  });

  $('map-axis').style.setProperty('--rows', String(Math.max(1, rowRightEdges.length)));
}

function render(breakdown: MemoryBreakdown) {
  const gpu = findGpu($<HTMLSelectElement>('gpu-select').value);
  const util = currentUtilization();
  const peak = breakdown.peakPerGpuBytes;

  const p = breakdown.parallelism;
  $('summary').innerHTML =
    `<b>${breakdown.modelName}</b> &middot; ${(breakdown.totalParams / 1e9).toFixed(2)}B params` +
    (breakdown.activeParams !== undefined && breakdown.activeParams !== breakdown.totalParams
      ? ` (${(breakdown.activeParams / 1e9).toFixed(2)}B active/token)`
      : '') +
    ` &middot; tp=${p.tp} pp=${p.pp} dp=${p.dp}${p.ep > 1 ? ` ep=${p.ep}` : ''}` +
    ` &middot; ${p.numGpus} GPU(s) &middot; ${breakdown.fidelity}`;

  if (gpu) {
    const fit = checkFit(breakdown, gpu, util);
    $('readout').dataset['state'] = fit.fits ? 'fits' : 'over';
    $('verdict-word').textContent = fit.fits ? 'Fits' : "Doesn't fit";
    $('verdict-figures').innerHTML =
      figure('Peak / GPU', formatBytes(peak)) +
      figure(fit.fits ? 'Headroom' : 'Short by', formatBytes(Math.abs(fit.headroomBytes))) +
      figure(`Budget · ${Math.round(util * 100)}%`, formatBytes(fit.usableBytes));
    $('verdict-note').textContent = '';
    drawMap(breakdown, gpu, fit.usableBytes);
  } else {
    $('readout').dataset['state'] = 'plain';
    $('verdict-word').textContent = formatBytes(peak);
    $('verdict-figures').innerHTML = figure('Peak / GPU', formatBytes(peak));
    $('verdict-note').textContent = 'Pick a GPU to see whether this fits.';
    $('map').hidden = true;
  }

  const tbody = $('table-body');
  tbody.innerHTML = '';
  breakdown.components.forEach((c, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td><span class="sw" style="background:${allocColor(i)}"></span>${c.name}` +
      `${c.note ? `<small>${c.note}</small>` : ''}</td>` +
      `<td class="num">${formatBytes(c.totalBytes)}</td>` +
      `<td class="num">${formatBytes(c.perGpuBytes)}</td>`;
    tbody.appendChild(tr);
  });
  const totalRow = document.createElement('tr');
  totalRow.className = 'total';
  totalRow.innerHTML = `<td>Total</td><td class="num">${formatBytes(breakdown.totalBytes)}</td><td class="num">${formatBytes(peak)}</td>`;
  tbody.appendChild(totalRow);

  $('warnings').innerHTML = breakdown.warnings.map((w) => `<div>${w}</div>`).join('');
}

function recompute() {
  try {
    const model = currentModel();
    const parallelism = currentParallelism();

    const moeNote = $('moe-note');
    if (model.config.moe || model.caveat) {
      moeNote.style.display = 'block';
      moeNote.textContent = model.caveat ?? 'MoE model: all expert weights must be resident, even though only top-k are active per token.';
    } else {
      moeNote.style.display = 'none';
    }
    // `flex`, not `block`: labels are flex columns so the control sits under
    // its text (see the layout note in index.html).
    $('ep-label').style.display = model.config.moe ? 'flex' : 'none';

    let breakdown: MemoryBreakdown;
    if (scenario === 'inference') {
      breakdown = estimateInference({
        model: model.config,
        parallelism,
        seqLen: num('inf-seq'),
        batchSize: num('inf-batch'),
        weightDtype: $<HTMLSelectElement>('weight-dtype').value as DType,
        kvDtype: $<HTMLSelectElement>('kv-dtype').value as DType,
        logitsMode: $<HTMLSelectElement>('logits-mode').value as 'lastToken' | 'fullSequence',
      });
    } else {
      const method = $<HTMLSelectElement>('ft-method').value as FinetuneMethod;
      breakdown = estimateFinetune({
        model: model.config,
        parallelism,
        seqLen: num('ft-seq'),
        batchSize: num('ft-batch'),
        method,
        optimizer: $<HTMLSelectElement>('ft-optimizer').value as OptimizerKind,
        zeroStage: Number($<HTMLSelectElement>('ft-zero').value) as ZeroStage,
        gradientCheckpointing: $<HTMLInputElement>('ft-ckpt').checked,
        mixedPrecision: $<HTMLInputElement>('ft-mp').checked,
        lora:
          method !== 'full'
            ? {
                rank: num('lora-rank') || 16,
                targetModules: $<HTMLInputElement>('lora-targets').value.split(',').map((s) => s.trim()).filter(Boolean),
              }
            : undefined,
        fidelity: $<HTMLSelectElement>('ft-fidelity').value as Fidelity,
      });
    }
    breakdown.warnings.unshift(...model.warnings);
    render(breakdown);
  } catch (err) {
    renderError(err instanceof Error ? err.message : String(err));
  }
}

init();
