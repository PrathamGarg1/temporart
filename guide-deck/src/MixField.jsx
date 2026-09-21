import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { motion } from "framer-motion";
import * as THREE from "three";

const WORDS = [
  { t: "नमस्ते", hi: true },
  { t: "yaar", hi: false },
  { t: "अच्छा", hi: true },
  { t: "matlab", hi: false },
  { t: "घर", hi: true },
  { t: "call", hi: false },
  { t: "ठीक", hi: true },
  { t: "time", hi: false },
  { t: "भाई", hi: true },
  { t: "message", hi: false },
  { t: "आज", hi: true },
  { t: "please", hi: false },
  { t: "कल", hi: true },
  { t: "wait", hi: false },
  { t: "हाँ", hi: true },
  { t: "bro", hi: false },
  { t: "क्या", hi: true },
  { t: "ok", hi: false },
];

function Field({ mix, dim }) {
  const group = useRef();
  const lines = useRef();
  const positions = useMemo(
    () =>
      WORDS.map((_, i) => {
        const a = (i / WORDS.length) * Math.PI * 2;
        const r = 2.2 + (i % 5) * 0.22;
        return new THREE.Vector3(Math.cos(a) * r, (i % 7) * 0.28 - 0.9, Math.sin(a * 1.4) * 1.4);
      }),
    []
  );
  const lineGeo = useMemo(() => {
    const pts = [];
    for (let i = 0; i < WORDS.length - 1; i += 2) {
      pts.push(positions[i], positions[i + 1]);
    }
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    return g;
  }, [positions]);

  useFrame(({ clock, pointer }) => {
    if (!group.current) return;
    const t = clock.getElapsedTime();
    group.current.rotation.y = t * 0.06 + pointer.x * 0.18;
    group.current.rotation.x = pointer.y * 0.08;
    group.current.position.z = THREE.MathUtils.lerp(-0.2, 0.55, mix);
    if (lines.current) {
      lines.current.material.opacity = 0.12 + mix * 0.22;
    }
  });

  return (
    <group ref={group}>
      <lineSegments ref={lines} geometry={lineGeo}>
        <lineBasicMaterial color="#c45c26" transparent opacity={0.18} />
      </lineSegments>
      {positions.map((p, i) => (
        <mesh key={`m${i}`} position={p}>
          <sphereGeometry args={[WORDS[i].hi ? 0.055 : 0.04, 12, 12]} />
          <meshBasicMaterial color={WORDS[i].hi ? "#c45c26" : "#e8e8e8"} />
        </mesh>
      ))}
      {WORDS.map((w, i) => (
        <Html
          key={w.t + i}
          position={positions[i]}
          transform
          occlude={false}
          style={{ pointerEvents: "none" }}
        >
          <span
            className={`glyph ${w.hi ? "deva" : "roman"} ${dim ? "dim" : ""}`}
            style={{ opacity: w.hi ? 0.35 + mix * 0.5 : 0.7 }}
          >
            {w.t}
          </span>
        </Html>
      ))}
    </group>
  );
}

export default function MixField({ mix = 0.5, dim = false }) {
  return (
    <div className="mix-field" aria-hidden="true">
      <Canvas camera={{ position: [0, 0, 6.2], fov: 42 }} dpr={[1, 1.6]} gl={{ antialias: true, alpha: true }}>
        <color attach="background" args={["#0c0c0c"]} />
        <ambientLight intensity={0.8} />
        <Field mix={mix} dim={dim} />
      </Canvas>
      <div className={`mix-words ${dim ? "dim" : ""}`}>
        {WORDS.map((w, i) => (
          <motion.span
            key={w.t + i}
            className={`glyph ${w.hi ? "deva" : "roman"}`}
            style={{
              left: `${62 + ((i * 13) % 34)}%`,
              top: `${12 + ((i * 17) % 72)}%`,
            }}
            animate={{ y: [0, i % 2 === 0 ? -8 : 8, 0], opacity: w.hi ? 0.45 : 0.28 }}
            transition={{ duration: 7 + (i % 5), repeat: Infinity, ease: "easeInOut" }}
          >
            {w.t}
          </motion.span>
        ))}
      </div>
    </div>
  );
}
