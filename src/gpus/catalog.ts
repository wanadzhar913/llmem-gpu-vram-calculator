import { GiB } from '../units.js';
import type { GPUSpec } from '../types.js';

export const GPU_CATALOG: GPUSpec[] = [
  { id: 'a100-40', name: 'NVIDIA A100 40GB', memoryBytes: 40 * GiB },
  { id: 'a100-80', name: 'NVIDIA A100 80GB', memoryBytes: 80 * GiB },
  { id: 'h100-80', name: 'NVIDIA H100 80GB', memoryBytes: 80 * GiB },
  { id: 'h200-141', name: 'NVIDIA H200 141GB', memoryBytes: 141 * GiB },
  { id: 'b200-180', name: 'NVIDIA B200 180GB', memoryBytes: 180 * GiB },
  { id: 'l40s-48', name: 'NVIDIA L40S 48GB', memoryBytes: 48 * GiB },
  { id: 'rtx4090-24', name: 'NVIDIA RTX 4090 24GB', memoryBytes: 24 * GiB },
  { id: 'rtx5090-32', name: 'NVIDIA RTX 5090 32GB', memoryBytes: 32 * GiB },
  { id: 'mi300x-192', name: 'AMD MI300X 192GB', memoryBytes: 192 * GiB },
];

export function findGpu(id: string): GPUSpec | undefined {
  const needle = id.toLowerCase();
  return GPU_CATALOG.find((g) => g.id.toLowerCase() === needle);
}
