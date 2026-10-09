import { C, fmtBytes } from '../../../lib/viz.js'

/*
 * Public-datasheet numbers (dense, no sparsity) plus microbenchmark-style latency estimates.
 * Latencies are APPROXIMATE (they vary by clock, access pattern and driver) — the lab labels them "≈".
 * bw / l2bw are aggregate GB/s; peak.* are TFLOP/s.
 */
export const GPUS = {
  a100: {
    name: 'A100 80GB SXM', arch: 'Ampere · sm_80', sms: 108, clk: 1.41, lanes: 64,
    regKB: 256, smemKB: 164, l1KB: 192, l2MB: 40, hbmGB: 80, memType: 'HBM2e',
    bw: 2039, l2bw: 5500,
    peak: { fp64: 9.7, fp32: 19.5, tf32: 156, fp16: 312 },
    lat: { reg: 4, smem: 23, l1: 33, l2: 200, hbm: 500 }
  },
  h100: {
    name: 'H100 SXM', arch: 'Hopper · sm_90', sms: 132, clk: 1.98, lanes: 128,
    regKB: 256, smemKB: 228, l1KB: 256, l2MB: 50, hbmGB: 80, memType: 'HBM3',
    bw: 3350, l2bw: 7500,
    peak: { fp64: 34, fp32: 67, tf32: 494, fp16: 989 },
    lat: { reg: 4, smem: 29, l1: 37, l2: 260, hbm: 580 }
  },
  rtx4090: {
    name: 'RTX 4090', arch: 'Ada · sm_89', sms: 128, clk: 2.52, lanes: 128,
    regKB: 256, smemKB: 100, l1KB: 128, l2MB: 72, hbmGB: 24, memType: 'GDDR6X',
    bw: 1008, l2bw: 5000,
    peak: { fp64: 1.29, fp32: 82.6, tf32: 82.6, fp16: 165 },
    lat: { reg: 4, smem: 23, l1: 35, l2: 270, hbm: 520 }
  }
}

export const GPU_OPTIONS = Object.entries(GPUS).map(([k, g]) => [k, g.name])

/** TB/s (or GB/s) pretty-printer */
export const fmtBW = gbs => (gbs >= 1000 ? `${(gbs / 1000).toFixed(gbs >= 10000 ? 0 : gbs >= 5000 ? 1 : 2)} TB/s` : `${Math.round(gbs)} GB/s`)

/** The five levels of the hierarchy for one GPU, ordered fastest → slowest. */
export function levelsFor(g) {
  const KB = 1024, MB = 1024 * KB, GB = 1024 * MB
  return [
    {
      id: 'reg', name: 'Registers', color: '#f472b6', where: 'register file inside each SM sub-core',
      capPerSM: g.regKB * KB, capTotal: g.regKB * KB * g.sms, capText: `${g.regKB} KB per SM · ${fmtBytes(g.regKB * KB * g.sms)} chip-wide`,
      lat: g.lat.reg, bw: g.sms * g.lanes * 12 * g.clk,
      scope: 'one thread', life: 'thread', kw: 'automatic local variables',
      note: 'Bandwidth is operand bandwidth (3 × 4 B per lane per cycle). Latency shown is a dependent FMA.'
    },
    {
      id: 'smem', name: 'Shared memory', color: '#fbbf24', where: 'on-chip SRAM, software-managed',
      capPerSM: g.smemKB * KB, capTotal: g.smemKB * KB * g.sms, capText: `up to ${g.smemKB} KB per SM (${fmtBytes(g.smemKB * KB * g.sms)} chip-wide)`,
      lat: g.lat.smem, bw: g.sms * 128 * g.clk,
      scope: 'thread block', life: 'block', kw: '__shared__ / extern __shared__',
      note: '32 banks × 4 B per cycle per SM.'
    },
    {
      id: 'l1', name: 'L1 cache', color: '#76d12a', where: 'same SRAM array as shared memory, hardware-managed',
      capPerSM: g.l1KB * KB, capTotal: g.l1KB * KB * g.sms, capText: `${g.l1KB} KB per SM shared with shared memory`,
      lat: g.lat.l1, bw: g.sms * 128 * g.clk,
      scope: 'one SM', life: 'transparent', kw: 'automatic (ld.ca / __ldg path)',
      note: 'The L1/shared split is configurable per kernel (carveout).'
    },
    {
      id: 'l2', name: 'L2 cache', color: '#22d3ee', where: 'chip-wide, sliced across memory partitions',
      capPerSM: null, capTotal: g.l2MB * MB, capText: `${g.l2MB} MB, shared by all ${g.sms} SMs`,
      lat: g.lat.l2, bw: g.l2bw,
      scope: 'whole GPU', life: 'transparent', kw: 'automatic (cudaAccessPolicyWindow for persistence)',
      note: 'All global traffic passes through L2; it is also where atomics execute.'
    },
    {
      id: 'hbm', name: g.memType === 'GDDR6X' ? 'GDDR6X (global)' : 'HBM (global)', color: '#8b7bff', where: `off-chip ${g.memType} DRAM`,
      capPerSM: null, capTotal: g.hbmGB * GB, capText: `${g.hbmGB} GB ${g.memType}`,
      lat: g.lat.hbm, bw: g.bw,
      scope: 'whole GPU', life: 'until cudaFree / app exit', kw: 'cudaMalloc / __device__ / local-memory backing',
      note: 'Highest capacity, lowest bandwidth-per-byte and highest latency on the chip.'
    }
  ]
}

/** FLOP/B where the HBM slope meets a compute ceiling. peak in TFLOP/s, bw in GB/s */
export const ridge = (peakTF, bwGBs) => (peakTF * 1000) / bwGBs
