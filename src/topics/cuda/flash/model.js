/** Square GEMM that reads A and B once and writes C once. FLOPs = 2 N³. */
export function gemmIntensity(N, bytes) {
  const flops = 2 * N * N * N
  const moved = 3 * N * N * bytes
  return { flops, moved, intensity: flops / moved }
}

/** Achieved TFLOP/s on the roofline. bw is GB/s, peak is TFLOP/s. */
export function achieved(intensity, bwGBs, peakTF) {
  const memRoof = (bwGBs * intensity) / 1000
  return Math.min(memRoof, peakTF)
}

/**
 * Forward attention, one head.
 * Standard materializes S and P (the dominant 4 N² term).
 * Flash keeps a Q tile on chip and re-reads K and V once per Q tile.
 * SRAM holds Q, K, V and O tiles plus the B_r×B_c score tile.
 */
export function attentionIO({ N, d, Br, Bc, bytes }) {
  const flops = 4 * N * N * d
  const standardElems = 4 * N * N + 4 * N * d
  const Tr = Math.ceil(N / Br)
  const Tc = Math.ceil(N / Bc)
  const flashElems = 2 * N * d * (1 + Tr)
  const sram = (2 * Br * d + 2 * Bc * d + Br * Bc) * bytes
  const standard = standardElems * bytes
  const flash = flashElems * bytes
  return {
    flops, standard, flash, Tr, Tc, sram,
    iStd: flops / standard,
    iFlash: flops / flash
  }
}

/** Integer scores, 16 keys, tiles of 4. V = 1, so the normalized output is 1. */
export const SCORES = [1, 2, 0, 1, 5, 1, 0, 2, 3, 1, 4, 0, 2, 2, 1, 3]
export const TILE = 4

/** Online softmax states after each tile. Step 0 is the start (no keys yet). */
export function onlineSteps(scores = SCORES, tile = TILE) {
  const steps = [{ m: null, l: 0, seen: 0 }]
  let m = -Infinity
  let l = 0
  for (let t = 0; t < scores.length; t += tile) {
    const block = scores.slice(t, t + tile)
    const mBlock = Math.max(...block)
    const mNew = Math.max(m, mBlock)
    const alpha = Number.isFinite(m) ? Math.exp(m - mNew) : 0
    const sumP = block.reduce((s, x) => s + Math.exp(x - mNew), 0)
    l = l * alpha + sumP
    m = mNew
    steps.push({ m, l, alpha, mBlock, seen: t + block.length })
  }
  return steps
}
