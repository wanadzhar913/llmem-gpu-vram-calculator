import type { GPUSpec, MemoryBreakdown } from './types.js';
import { checkFit } from './advise.js';
import { formatBytes, formatParams } from './units.js';

function padEnd(s: string, len: number): string {
  return s.length >= len ? s : s + ' '.repeat(len - s.length);
}

function padStart(s: string, len: number): string {
  return s.length >= len ? s : ' '.repeat(len - s.length) + s;
}

/** Renders a breakdown as an aligned plain-text table, optionally with a fit verdict against `gpu`. */
export function renderText(breakdown: MemoryBreakdown, gpu?: GPUSpec, gpuMemoryUtilization = 1): string {
  const lines: string[] = [];
  const { parallelism: p } = breakdown;
  lines.push(`${breakdown.modelName}  (${breakdown.scenario}, fidelity=${breakdown.fidelity})`);
  lines.push(
    `params: ${formatParams(breakdown.totalParams)} total` +
      (breakdown.activeParams !== undefined && breakdown.activeParams !== breakdown.totalParams
        ? `, ${formatParams(breakdown.activeParams)} active/token`
        : ''),
  );
  lines.push(`parallelism: tp=${p.tp} pp=${p.pp} dp=${p.dp}${p.ep > 1 ? ` ep=${p.ep}` : ''} (${p.numGpus} GPUs)`);
  lines.push('');

  const nameW = Math.max(...breakdown.components.map((c) => c.name.length), 'Component'.length);
  const totalW = Math.max(...breakdown.components.map((c) => formatBytes(c.totalBytes).length), 'Total'.length);
  const perGpuW = Math.max(...breakdown.components.map((c) => formatBytes(c.perGpuBytes).length), 'Per-GPU (peak)'.length);

  lines.push(`${padEnd('Component', nameW)}  ${padStart('Total', totalW)}  ${padStart('Per-GPU (peak)', perGpuW)}`);
  lines.push('-'.repeat(nameW + totalW + perGpuW + 4));
  for (const c of breakdown.components) {
    lines.push(
      `${padEnd(c.name, nameW)}  ${padStart(formatBytes(c.totalBytes), totalW)}  ${padStart(formatBytes(c.perGpuBytes), perGpuW)}` +
        (c.note ? `  # ${c.note}` : ''),
    );
  }
  lines.push('-'.repeat(nameW + totalW + perGpuW + 4));
  lines.push(
    `${padEnd('TOTAL', nameW)}  ${padStart(formatBytes(breakdown.totalBytes), totalW)}  ${padStart(formatBytes(breakdown.peakPerGpuBytes), perGpuW)}`,
  );

  if (gpu) {
    lines.push('');
    const fit = checkFit(breakdown, gpu, gpuMemoryUtilization);
    lines.push(fit.message);
  }

  if (breakdown.warnings.length > 0) {
    lines.push('');
    for (const w of breakdown.warnings) lines.push(`! ${w}`);
  }

  return lines.join('\n');
}

/** Renders a breakdown as JSON (bytes as numbers, plus a human-readable `*Human` mirror for each field). */
export function renderJson(breakdown: MemoryBreakdown, gpu?: GPUSpec, gpuMemoryUtilization = 1): string {
  const payload: Record<string, unknown> = {
    ...breakdown,
    totalBytesHuman: formatBytes(breakdown.totalBytes),
    peakPerGpuBytesHuman: formatBytes(breakdown.peakPerGpuBytes),
    components: breakdown.components.map((c) => ({
      ...c,
      totalBytesHuman: formatBytes(c.totalBytes),
      perGpuBytesHuman: formatBytes(c.perGpuBytes),
    })),
  };
  if (gpu) {
    payload['fit'] = checkFit(breakdown, gpu, gpuMemoryUtilization);
  }
  return JSON.stringify(payload, null, 2);
}
