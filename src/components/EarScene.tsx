import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Line, OrbitControls, useGLTF } from '@react-three/drei'
import * as THREE from 'three'
import { pointOnPath, stageById, type Point } from '../lib/pathway'

const MODEL = '/models/hearing.glb'
const RED = '#cf2f2a'

interface Look {
  color: string
  opacity: number
  /** Stops at which this part lights up. */
  stops: string[]
}

/** How each named part of the model is drawn. Names come from tools/build_anatomy.py. */
const LOOKS: Record<string, Look> = {
  cortex: { color: '#d3ada0', opacity: 0.3, stops: [] },
  cerebellum: { color: '#c9a596', opacity: 0.3, stops: [] },
  thalamus: { color: '#b89a8c', opacity: 0.35, stops: [] },
  brainstem: { color: '#c7a08f', opacity: 0.55, stops: ['brainstem'] },
  colliculus: { color: '#7d8f8c', opacity: 1, stops: ['midbrain'] },
  geniculate: { color: '#7d8f8c', opacity: 1, stops: ['thalamus'] },
  auditory_cortex: { color: '#b98a7c', opacity: 0.85, stops: ['cortex'] },
  outer_ear: { color: '#d6b8a2', opacity: 1, stops: ['outer-ear'] },
  canal: { color: '#d6b8a2', opacity: 0.4, stops: ['outer-ear'] },
  eardrum: { color: '#a98f7c', opacity: 1, stops: ['eardrum'] },
  bones: { color: '#e4dcc4', opacity: 1, stops: ['bones'] },
  cochlea: { color: '#1c4f9c', opacity: 1, stops: ['cochlea', 'hair-cells'] },
  nerve: { color: '#c2a24a', opacity: 1, stops: ['nerve'] },
}

/** The signal's route through the model, from the eardrum to the cortex. */
const ROUTE: Point[] = ['eardrum', 'bones', 'cochlea', 'nerve', 'brainstem', 'midbrain', 'thalamus', 'cortex']
  .map((id) => stageById(id).position)

interface SceneProps {
  selected: string
  moving: boolean
  /** Fly to the selected stop straight away instead of opening on the whole view. */
  flyOnOpen?: boolean
}

function Anatomy({ selected }: { selected: string }) {
  const { scene } = useGLTF(MODEL)
  const parts = useMemo(() => {
    const found: { name: string; geometry: THREE.BufferGeometry }[] = []
    scene.updateMatrixWorld(true)
    scene.traverse((object) => {
      const mesh = object as THREE.Mesh
      if (!mesh.isMesh) return
      const name = LOOKS[mesh.name] ? mesh.name : mesh.parent?.name ?? ''
      if (!LOOKS[name]) return
      const geometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld)
      geometry.computeVertexNormals()
      found.push({ name, geometry })
    })
    return found
  }, [scene])
  return (
    <>
      {parts.map(({ name, geometry }) => {
        const look = LOOKS[name]
        const lit = look.stops.includes(selected)
        const opacity = lit ? Math.max(look.opacity, 0.92) : look.opacity
        return (
          <mesh key={name} geometry={geometry} renderOrder={opacity < 1 ? 2 : 1}>
            <meshStandardMaterial color={lit ? RED : look.color} roughness={0.75}
                                  transparent={opacity < 1} opacity={opacity}
                                  depthWrite={opacity >= 0.5} side={THREE.DoubleSide} />
          </mesh>
        )
      })}
    </>
  )
}

function SoundWaves({ selected, moving }: SceneProps) {
  const rings = useRef<THREE.Group>(null)
  const from = stageById('sound').position
  const to = stageById('outer-ear').position
  useFrame(({ clock }) => {
    rings.current?.children.forEach((ring, i) => {
      const phase = moving ? (clock.elapsedTime * 0.35 + i / 4) % 1 : i / 4
      ring.position.set(from[0] + (to[0] - 0.9 - from[0]) * phase, from[1], from[2])
      ring.scale.setScalar(0.35 + phase * 0.55)
      ;((ring as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = 0.8 * (1 - phase)
    })
  })
  return (
    <group ref={rings}>
      {[0, 1, 2, 3].map((i) => (
        <mesh key={i} rotation={[0, Math.PI / 2, 0]}>
          <torusGeometry args={[1, 0.03, 8, 64]} />
          <meshBasicMaterial color={selected === 'sound' ? RED : '#1c4f9c'} transparent />
        </mesh>
      ))}
    </group>
  )
}

/** Small lights that travel the route, and a thin line marking it. */
function Signal({ moving }: { moving: boolean }) {
  const lights = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    lights.current?.children.forEach((light, i) => {
      const [x, y, z] = pointOnPath(ROUTE, (moving ? clock.elapsedTime * 0.08 : 0) + i / 5)
      light.position.set(x, y, z)
    })
  })
  return (
    <>
      <Line points={ROUTE} color={RED} lineWidth={1.5} transparent opacity={0.55} depthTest={false} />
      <group ref={lights}>
        {[0, 1, 2, 3, 4].map((i) => (
          <mesh key={i} renderOrder={3}>
            <sphereGeometry args={[0.055, 14, 10]} />
            <meshBasicMaterial color={RED} depthTest={false} />
          </mesh>
        ))}
      </group>
    </>
  )
}

/** Turns the camera toward the selected stop and moves in to fit it; then the viewer is free again. */
function Focus({ selected, flyOnOpen }: { selected: string; flyOnOpen?: boolean }) {
  const controls = useRef<React.ComponentRef<typeof OrbitControls>>(null)
  const camera = useThree((state) => state.camera)
  const settling = useRef(flyOnOpen ? 150 : 0)
  const shown = useRef(selected)
  const goal = useMemo(() => new THREE.Vector3(), [])
  const offset = useMemo(() => new THREE.Vector3(), [])
  useEffect(() => {
    if (shown.current !== selected) settling.current = 150
    shown.current = selected
  }, [selected])
  useFrame(() => {
    if (!controls.current || settling.current <= 0) return
    settling.current -= 1
    const stage = stageById(selected)
    goal.set(...stage.position)
    const target = controls.current.target
    offset.copy(camera.position).sub(target)
    const distance = THREE.MathUtils.lerp(offset.length(), Math.min(Math.max(stage.size * 2.6, 2.2), 13), 0.05)
    target.lerp(goal, 0.05)
    camera.position.copy(target).add(offset.setLength(distance))
    controls.current.update()
  })
  return <OrbitControls ref={controls} target={[0.3, 0.7, 0]} enablePan={false} minDistance={1.2} maxDistance={22}
                        onStart={() => { settling.current = 0 }} />
}

export default function EarScene({ selected, moving, flyOnOpen }: SceneProps) {
  return (
    <Canvas camera={{ position: [-5.4, 1.9, 14.2], fov: 36 }} dpr={[1, 2]}
            aria-label="Three-dimensional model of the ear and brain, showing the path from the outer ear to the auditory cortex">
      <ambientLight intensity={1.4} />
      <directionalLight position={[-4, 6, 10]} intensity={1.7} />
      <directionalLight position={[6, -2, 4]} intensity={0.5} />
      <Anatomy selected={selected} />
      <SoundWaves selected={selected} moving={moving} />
      <Signal moving={moving} />
      <Focus selected={selected} flyOnOpen={flyOnOpen} />
    </Canvas>
  )
}

useGLTF.preload(MODEL)
