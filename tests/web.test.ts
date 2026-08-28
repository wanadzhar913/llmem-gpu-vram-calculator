// @vitest-environment happy-dom
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MODEL_CATALOG } from '../src/models/catalog.js';
import { GPU_CATALOG } from '../src/gpus/catalog.js';

/**
 * The web UI has no exported surface — `web/app.ts` wires the real
 * `web/index.html` markup to the estimator on import and never returns a handle.
 * So these tests run the actual bundle entrypoint against the actual markup in a
 * happy-dom document and assert on what the user would see: the populated
 * controls, the verdict readout, the breakdown table, and the recompute that
 * fires on every input change.
 */

const HTML = readFileSync(resolve(process.cwd(), 'web/index.html'), 'utf8');
const BODY = HTML.slice(HTML.indexOf('<body>') + '<body>'.length, HTML.indexOf('</body>'))
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, '');

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const flush = () => new Promise((r) => setTimeout(r, 0));

/** Load the markup, then import the entrypoint fresh so its `init()` runs against it. */
async function mountApp() {
  document.body.innerHTML = BODY;
  vi.resetModules();
  await import('../web/app.js');
  await flush();
}

beforeEach(async () => {
  vi.useRealTimers();
  await mountApp();
});

describe('web UI — control population', () => {
  it('lists every catalog preset plus the paste-a-config sentinel', () => {
    const opts = Array.from($<HTMLSelectElement>('model-preset').options);
    expect(opts).toHaveLength(MODEL_CATALOG.length + 1);
    expect(opts.at(-1)!.value).toBe('__custom__');
    for (const preset of MODEL_CATALOG) {
      expect(opts.some((o) => o.value === preset.id)).toBe(true);
    }
  });

  it('lists every GPU and defaults the dtype selects to bf16', () => {
    expect($<HTMLSelectElement>('gpu-select').options).toHaveLength(GPU_CATALOG.length);
    expect($<HTMLSelectElement>('weight-dtype').value).toBe('bf16');
    expect($<HTMLSelectElement>('kv-dtype').value).toBe('bf16');
  });
});

describe('web UI — default estimate', () => {
  it('renders a fit verdict and a breakdown table for the default inference scenario', () => {
    expect(['fits', 'over']).toContain($('readout').dataset['state']);
    expect($('verdict-word').textContent).not.toBe('');
    expect($('summary').textContent).toContain('B params');
    // one row per component plus the total row
    const rows = $('table-body').querySelectorAll('tr');
    expect(rows.length).toBeGreaterThan(1);
    expect(rows[rows.length - 1]!.classList.contains('total')).toBe(true);
    expect($('map').hidden).toBe(false);
  });

  it('recomputes when an input changes', () => {
    const before = $('verdict-figures').textContent;
    const batch = $<HTMLInputElement>('inf-batch');
    batch.value = '64';
    batch.dispatchEvent(new Event('input', { bubbles: true }));
    expect($('verdict-figures').textContent).not.toBe(before);
  });
});

describe('web UI — fine-tuning tab', () => {
  it('swaps the field groups and re-estimates on tab click', () => {
    const tab = document.querySelector<HTMLElement>('.tab[data-scenario="finetune"]')!;
    tab.click();
    expect($('finetune-fields').style.display).toBe('block');
    expect($('inference-fields').style.display).toBe('none');
    expect($('table-body').querySelectorAll('tr').length).toBeGreaterThan(1);
  });

  it('reveals the LoRA fields only for adapter methods', () => {
    document.querySelector<HTMLElement>('.tab[data-scenario="finetune"]')!.click();
    const method = $<HTMLSelectElement>('ft-method');
    expect($('lora-fields').style.display).toBe('none');
    method.value = 'lora';
    method.dispatchEvent(new Event('change', { bubbles: true }));
    expect($('lora-fields').style.display).toBe('block');
  });
});

describe('web UI — pasted config.json', () => {
  it('shows an actionable error until a config is pasted, then estimates from it', async () => {
    const preset = $<HTMLSelectElement>('model-preset');
    preset.value = '__custom__';
    preset.dispatchEvent(new Event('change', { bubbles: true }));
    expect($('hf-fields').style.display).toBe('block');
    expect($('readout').dataset['state']).toBe('error');
    expect($('verdict-word').textContent).toBe("Can't estimate");

    const config = JSON.stringify({
      num_hidden_layers: 4,
      hidden_size: 256,
      num_attention_heads: 8,
      vocab_size: 1000,
    });
    const box = $<HTMLTextAreaElement>('hf-config');
    box.value = config;
    box.dispatchEvent(new Event('input', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 300)); // clears the 250 ms reparse debounce

    expect($('hf-status').textContent).toContain('B params');
    expect(['fits', 'over']).toContain($('readout').dataset['state']);
  });
});

describe('web UI — usable-memory field', () => {
  it('snaps an out-of-range percentage back into [1, 100] after editing', () => {
    const util = $<HTMLInputElement>('gpu-util');

    util.value = '150';
    util.dispatchEvent(new Event('change', { bubbles: true }));
    expect(util.value).toBe('100');

    util.value = '0';
    util.dispatchEvent(new Event('change', { bubbles: true }));
    expect(util.value).toBe('75'); // nonsensical -> default, not clamped to 1
  });
});

describe('web UI — memory map', () => {
  it('keeps the capacity wall on-scale when the allocation overruns the card', () => {
    $<HTMLSelectElement>('model-preset').value = 'llama-3.1-405b';
    $<HTMLSelectElement>('model-preset').dispatchEvent(new Event('change', { bubbles: true }));
    const gpu = $<HTMLSelectElement>('gpu-select');
    gpu.value = 'rtx4090-24';
    gpu.dispatchEvent(new Event('change', { bubbles: true }));

    expect($('readout').dataset['state']).toBe('over');
    expect($('map-wall').hidden).toBe(false);
    const wallLeft = parseFloat($('map-wall').style.left);
    expect(wallLeft).toBeGreaterThan(0);
    expect(wallLeft).toBeLessThan(100); // scale grew past the card, wall moved left
    expect(parseFloat($('map-over').style.width)).toBeGreaterThan(0);
  });
});
