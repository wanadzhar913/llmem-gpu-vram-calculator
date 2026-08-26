import { MODEL_CATALOG, type ModelPreset } from '../src/models/catalog.js';
import { GPU_CATALOG, findGpu } from '../src/gpus/catalog.js';
import { estimateInference } from '../src/inference.js';
import { estimateFinetune } from '../src/finetune.js';
import { checkFit } from '../src/advise.js';
import { formatBytes } from '../src/units.js';
import { DTYPES, type DType } from '../src/units.js';
import type { FinetuneMethod, Fidelity, MemoryBreakdown, OptimizerKind, ZeroStage } from '../src/types.js';

type Scenario = 'inference' | 'finetune';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const COLORS = ['#5b8cff', '#35c07a', '#e0a63b', '#c96bd8', '#4fc3d9', '#ef5a5a', '#8a97ac'];

let scenario: Scenario = 'inference';

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
  populateSelect(
    $('model-preset'),
    MODEL_CATALOG.map((p) => ({
      value: p.id,
      label: p.config.moe ? `${p.id} (MoE ${p.config.moe.numExperts}x, top-${p.config.moe.expertsPerToken})` : p.id,
    })),
  );
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

  document.querySelectorAll('input, select').forEach((el) => {
    el.addEventListener('input', recompute);
    el.addEventListener('change', recompute);
  });

  recompute();
}

function currentPreset(): ModelPreset {
  const id = $<HTMLSelectElement>('model-preset').value;
  const preset = MODEL_CATALOG.find((p) => p.id === id);
  if (!preset) throw new Error('no preset selected');
  return preset;
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

function renderError(message: string) {
  $('verdict').className = 'verdict bad';
  $('verdict').textContent = message;
  $('summary').textContent = '';
  $('legend').innerHTML = '';
  $('bar').innerHTML = '';
  $('table-body').innerHTML = '';
  $('warnings').innerHTML = '';
}

function render(breakdown: MemoryBreakdown) {
  const gpuId = $<HTMLSelectElement>('gpu-select').value;
  const gpu = findGpu(gpuId);
  const util = Number($<HTMLInputElement>('gpu-util').value) || 1;

  const verdictEl = $('verdict');
  if (gpu) {
    const fit = checkFit(breakdown, gpu, util);
    verdictEl.className = `verdict ${fit.fits ? 'ok' : 'bad'}`;
    verdictEl.textContent = fit.message;
  } else {
    verdictEl.className = 'verdict';
    verdictEl.textContent = `Peak per-GPU: ${formatBytes(breakdown.peakPerGpuBytes)}`;
  }

  const p = breakdown.parallelism;
  $('summary').innerHTML =
    `<b>${breakdown.modelName}</b> — ${(breakdown.totalParams / 1e9).toFixed(2)}B params` +
    (breakdown.activeParams !== undefined && breakdown.activeParams !== breakdown.totalParams
      ? ` (${(breakdown.activeParams / 1e9).toFixed(2)}B active/token)`
      : '') +
    ` &middot; tp=${p.tp} pp=${p.pp} dp=${p.dp}${p.ep > 1 ? ` ep=${p.ep}` : ''} &middot; ${p.numGpus} GPU(s) &middot; fidelity=${breakdown.fidelity}`;

  const legend = $('legend');
  const bar = $('bar');
  legend.innerHTML = '';
  bar.innerHTML = '';
  const peakTotal = breakdown.components.reduce((s, c) => s + c.perGpuBytes, 0) || 1;
  breakdown.components.forEach((c, i) => {
    const color = COLORS[i % COLORS.length];
    const pct = (c.perGpuBytes / peakTotal) * 100;
    const seg = document.createElement('div');
    seg.className = 'bar-seg';
    seg.style.width = `${pct}%`;
    seg.style.background = color!;
    seg.title = `${c.name}: ${formatBytes(c.perGpuBytes)}`;
    bar.appendChild(seg);

    const legendItem = document.createElement('span');
    legendItem.innerHTML = `<span class="sw" style="background:${color}"></span>${c.name}`;
    legend.appendChild(legendItem);
  });

  const tbody = $('table-body');
  tbody.innerHTML = '';
  for (const c of breakdown.components) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${c.name}${c.note ? `<br><span style="color:var(--muted);font-size:11px">${c.note}</span>` : ''}</td><td class="num">${formatBytes(c.totalBytes)}</td><td class="num">${formatBytes(c.perGpuBytes)}</td>`;
    tbody.appendChild(tr);
  }
  const totalRow = document.createElement('tr');
  totalRow.className = 'total';
  totalRow.innerHTML = `<td>Total</td><td class="num">${formatBytes(breakdown.totalBytes)}</td><td class="num">${formatBytes(breakdown.peakPerGpuBytes)}</td>`;
  tbody.appendChild(totalRow);

  const warnEl = $('warnings');
  warnEl.innerHTML = breakdown.warnings.map((w) => `<div>! ${w}</div>`).join('');
}

function recompute() {
  try {
    const preset = currentPreset();
    const parallelism = currentParallelism();

    const moeNote = $('moe-note');
    if (preset.config.moe || preset.caveat) {
      moeNote.style.display = 'block';
      moeNote.textContent = preset.caveat ?? 'MoE model: all expert weights must be resident, even though only top-k are active per token.';
    } else {
      moeNote.style.display = 'none';
    }
    $('ep-label').style.display = preset.config.moe ? 'block' : 'none';

    let breakdown: MemoryBreakdown;
    if (scenario === 'inference') {
      breakdown = estimateInference({
        model: preset.config,
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
        model: preset.config,
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
    render(breakdown);
  } catch (err) {
    renderError(err instanceof Error ? err.message : String(err));
  }
}

init();
