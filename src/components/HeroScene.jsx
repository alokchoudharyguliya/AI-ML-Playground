import React, { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { rng } from '../lib/viz.js'

const LAYERS = [4, 6, 7, 6, 3]
const tmp = new THREE.Object3D()
const col = new THREE.Color()

function Network() {
  const nodesRef = useRef()
  const pulseRef = useRef()
  const { pos, edges, lines } = useMemo(() => {
    const r = rng(11)
    const pos = []
    const starts = []
    LAYERS.forEach((n, li) => {
      starts.push(pos.length)
      for (let i = 0; i < n; i++) pos.push(new THREE.Vector3((li - 2) * 1.5, (i - (n - 1) / 2) * 0.58, (r() - 0.5) * 1.2))
    })
    const edges = []
    for (let li = 0; li < LAYERS.length - 1; li++)
      for (let a = 0; a < LAYERS[li]; a++) for (let b = 0; b < LAYERS[li + 1]; b++)
        if (r() < 0.55) edges.push([starts[li] + a, starts[li + 1] + b])
    const lines = new Float32Array(edges.length * 6)
    edges.forEach(([a, b], k) => { pos[a].toArray(lines, k * 6); pos[b].toArray(lines, k * 6 + 3) })
    return { pos, edges, lines }
  }, [])
  const pulses = useMemo(() => Array.from({ length: 36 }, (_, i) => ({ e: (i * 7) % edges.length, t: (i % 12) / 12, s: 0.35 + (i % 5) * 0.08 })), [edges])

  useFrame(({ clock }, dt) => {
    const t = clock.elapsedTime
    pos.forEach((p, i) => {
      const s = 1 + 0.35 * Math.sin(t * 1.6 + i * 0.9)
      tmp.position.copy(p); tmp.scale.setScalar(s); tmp.updateMatrix()
      nodesRef.current.setMatrixAt(i, tmp.matrix)
      nodesRef.current.setColorAt(i, col.set('#8b7bff').lerp(col.clone().set('#22d3ee'), 0.5 + 0.5 * Math.sin(t + i)))
    })
    nodesRef.current.instanceMatrix.needsUpdate = true
    nodesRef.current.instanceColor.needsUpdate = true
    pulses.forEach((p, i) => {
      p.t += dt * p.s
      if (p.t > 1) { p.t = 0; p.e = (p.e * 13 + 5 + i) % edges.length }
      const [a, b] = edges[p.e]
      tmp.position.lerpVectors(pos[a], pos[b], p.t); tmp.scale.setScalar(1); tmp.updateMatrix()
      pulseRef.current.setMatrixAt(i, tmp.matrix)
    })
    pulseRef.current.instanceMatrix.needsUpdate = true
  })

  return (
    <group>
      <lineSegments>
        <bufferGeometry><bufferAttribute attach="attributes-position" args={[lines, 3]} /></bufferGeometry>
        <lineBasicMaterial color="#8b7bff" transparent opacity={0.22} />
      </lineSegments>
      <instancedMesh ref={nodesRef} args={[null, null, pos.length]}>
        <sphereGeometry args={[0.11, 20, 20]} />
        <meshStandardMaterial emissive="#5b4bd6" emissiveIntensity={0.6} roughness={0.3} />
      </instancedMesh>
      <instancedMesh ref={pulseRef} args={[null, null, pulses.length]}>
        <sphereGeometry args={[0.05, 10, 10]} />
        <meshBasicMaterial color="#22d3ee" />
      </instancedMesh>
    </group>
  )
}

const N = 7
function GpuGrid() {
  const ref = useRef()
  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    let k = 0
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++, k++) {
      const w = 0.5 + 0.5 * Math.sin(t * 1.8 - (i + j) * 0.55)
      tmp.position.set((i - N / 2) * 0.42, w * 0.25, (j - N / 2) * 0.42)
      tmp.scale.set(1, 0.4 + w * 1.6, 1); tmp.updateMatrix()
      ref.current.setMatrixAt(k, tmp.matrix)
      ref.current.setColorAt(k, col.set('#1f3d12').lerp(col.clone().set('#76d12a'), w))
    }
    ref.current.instanceMatrix.needsUpdate = true
    ref.current.instanceColor.needsUpdate = true
  })
  return (
    <instancedMesh ref={ref} args={[null, null, N * N]}>
      <boxGeometry args={[0.32, 0.3, 0.32]} />
      <meshStandardMaterial roughness={0.45} metalness={0.2} />
    </instancedMesh>
  )
}

export default function HeroScene() {
  return (
    <Canvas dpr={[1, 1.75]} camera={{ position: [0, 1.6, 9.5], fov: 42 }}>
      <ambientLight intensity={0.8} />
      <pointLight position={[4, 5, 5]} intensity={40} color="#ffffff" />
      <pointLight position={[-5, -2, 3]} intensity={30} color="#8b7bff" />
      <group position={[2.4, 0.6, 0]} rotation={[0.05, -0.25, 0]}>
        <Network />
      </group>
      <group position={[5.2, -2.1, -0.5]} rotation={[0.25, -0.5, 0]} scale={0.95}>
        <GpuGrid />
      </group>
      <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.35} maxPolarAngle={1.8} minPolarAngle={1.2} />
    </Canvas>
  )
}
