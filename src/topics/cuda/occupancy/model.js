/** SM limits that match the memory-chapter datasheets and the CUDA occupancy calculator. */
export const SM = {
  a100:    { maxWarps: 64, maxBlocks: 32, regs: 65536, smem: 164 * 1024, schedulers: 4 },
  h100:    { maxWarps: 64, maxBlocks: 32, regs: 65536, smem: 228 * 1024, schedulers: 4 },
  rtx4090: { maxWarps: 48, maxBlocks: 24, regs: 65536, smem: 100 * 1024, schedulers: 4 }
}

const REG_UNIT = 256
const WARP_GRAN = 4
const SMEM_UNIT = 128
const SMEM_RESERVE = 1024

const roundUp = (n, u) => Math.ceil(n / u) * u

/**
 * How many blocks of this shape fit on one SM, and which resource stops it.
 * Register demand rounds up to a multiple of 8 registers/thread (256 per warp).
 * Resident warps from the register file then round down to a multiple of 4.
 * A block that uses shared memory also pays the 1 KB CUDA reserves per block.
 */
export function occupancy({ threads, regs, smem, sm }) {
  const warpsPerBlock = Math.ceil(threads / 32)
  const regsAlloc = roundUp(regs, 8)
  const regsPerWarp = roundUp(regs * 32, REG_UNIT)
  const warpsByRegs = Math.floor(sm.regs / regsPerWarp / WARP_GRAN) * WARP_GRAN
  const perBlockMax = sm.smem - SMEM_RESERVE
  const smemRounded = smem > 0 ? roundUp(smem, SMEM_UNIT) : 0
  const smemCharged = smem > 0 ? smemRounded + SMEM_RESERVE : 0
  const tooBig = threads > 1024 || threads < 1 || regs > 255 || smem > perBlockMax

  const limits = [
    { id: 'regs', name: 'registers', blocks: Math.floor(warpsByRegs / warpsPerBlock) },
    { id: 'smem', name: 'shared memory', blocks: smemCharged === 0 ? Infinity : Math.floor(sm.smem / smemCharged) },
    { id: 'warps', name: 'warp slots', blocks: Math.floor(sm.maxWarps / warpsPerBlock) },
    { id: 'blocks', name: 'block slots', blocks: sm.maxBlocks }
  ]

  if (tooBig || limits.some(l => l.blocks < 1)) {
    const reason = threads > 1024 ? 'A block cannot hold more than 1024 threads.'
      : regs > 255 ? 'The compiler cannot give a thread more than 255 registers.'
      : smem > perBlockMax ? 'Shared memory exceeds the per-block maximum (the SM pool minus the 1 KB reserve).'
      : 'One block already needs more registers or warps than the SM has.'
    return {
      fit: false, reason, warpsPerBlock, regsAlloc, regsPerWarp, smemCharged, perBlockMax,
      limits, activeBlocks: 0, activeWarps: 0, occ: 0, limiters: []
    }
  }

  const activeBlocks = Math.min(...limits.map(l => l.blocks))
  const limiters = limits.filter(l => l.blocks === activeBlocks)
  const activeWarps = activeBlocks * warpsPerBlock
  return {
    fit: true, reason: '', warpsPerBlock, regsAlloc, regsPerWarp, smemCharged, perBlockMax,
    limits, activeBlocks, activeWarps, occ: activeWarps / sm.maxWarps, limiters
  }
}

/** Warps one SM needs so every scheduler finds a ready warp. */
export function warpsToHide(latency, ilp, schedulers) {
  const readyFrac = Math.min(1, ilp / latency)
  return Math.ceil(schedulers / readyFrac)
}

/** Bytes that must be in flight per SM to fill the given link. bw GB/s, lat cycles, clk GHz. */
export function bytesInFlight(bwGBs, latCycles, clkGHz, sms) {
  const perSM = (bwGBs * 1e9) / sms
  const seconds = latCycles / (clkGHz * 1e9)
  return perSM * seconds
}
