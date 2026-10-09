import React, { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'

/** Standard 3D stage for labs: dark scene, soft lights, orbit controls, optional HTML overlay. */
export default function Stage3D({ height = 420, camera = [6, 5, 9], target = [0, 0, 0], fov = 45, children, autoRotate = false, controls = true, overlay, hint = 'drag to orbit · scroll to zoom' }) {
  return (
    <div className="stage3d" style={{ height }}>
      <Canvas dpr={[1, 2]} camera={{ position: camera, fov }} gl={{ antialias: true }}>
        <color attach="background" args={['#0a0c14']} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[6, 10, 6]} intensity={1.4} />
        <pointLight position={[-6, -3, -4]} intensity={30} color="#8b7bff" />
        <Suspense fallback={null}>{children}</Suspense>
        {controls && <OrbitControls target={target} enablePan={false} autoRotate={autoRotate} autoRotateSpeed={0.6} makeDefault />}
      </Canvas>
      {overlay && <div className="stage-info">{overlay}</div>}
      <div className="stage-hint">{hint}</div>
    </div>
  )
}
