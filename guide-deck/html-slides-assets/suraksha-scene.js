/* Three.js message field + GSAP camera (web3d-integration-patterns: layered Three + GSAP) */
(function () {
  try {
  if (typeof THREE === "undefined") return;
  const canvas = document.getElementById("gl");
  if (!canvas) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) {
    canvas.style.display = "none";
    return;
  }

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 80);
  camera.position.set(3.2, 0.4, 12);
  camera.lookAt(7, 0, 0);

  const COUNT = 70;
  const geo = new THREE.SphereGeometry(0.045, 8, 8);
  const mat = new THREE.MeshBasicMaterial({ color: 0xd4d4d2 });
  const mesh = new THREE.InstancedMesh(geo, mat, COUNT);
  const dummy = new THREE.Object3D();
  const seeds = [];
  for (let i = 0; i < COUNT; i++) {
    seeds.push({
      x: 2 + Math.random() * 14,
      y: (Math.random() - 0.5) * 11,
      z: (Math.random() - 0.5) * 8,
      s: 0.7 + Math.random() * 0.8,
      p: Math.random() * Math.PI * 2,
    });
  }
  scene.add(mesh);

  const state = { spread: 1, lift: 0, gather: 0, spin: 0.05 };

  function layout(t) {
    for (let i = 0; i < COUNT; i++) {
      const p = seeds[i];
      const wob = Math.sin(t * 0.35 + p.p) * 0.15;
      dummy.position.set(
        p.x * state.spread * (1 - state.gather * 0.7),
        p.y * state.spread + state.lift * Math.max(p.y, 0) + wob,
        p.z * (1 - state.gather * 0.4)
      );
      dummy.rotation.set(0.1, p.p + t * state.spin, 0.05);
      dummy.scale.setScalar(p.s);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }

  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);

  const clock = new THREE.Clock();
  function tick() {
    layout(clock.getElapsedTime());
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  resize();
  tick();

  const looks = [
    { spread: 1.15, lift: 0, gather: 0, spin: 0.04, color: 0xd4d4d2, cam: [3.2, 0.4, 12] },
    { spread: 0.95, lift: 0, gather: 0.1, spin: 0.05, color: 0xe0e0e0, cam: [2.4, 0.2, 11] },
    { spread: 1.35, lift: 2.8, gather: 0, spin: 0.08, color: 0xe24a1a, cam: [1.2, 1.5, 13] },
    { spread: 0.45, lift: 0, gather: 0.75, spin: 0.02, color: 0xd4d4d2, cam: [0.6, 0, 9] },
    { spread: 0.8, lift: 0, gather: 0.2, spin: 0.03, color: 0xe0e0e0, cam: [2, 0.2, 11] },
    { spread: 1.2, lift: 0.4, gather: 0, spin: 0.06, color: 0xe24a1a, cam: [4, 0.6, 12] },
    { spread: 0.9, lift: 0, gather: 0.15, spin: 0.04, color: 0xd4d4d2, cam: [2.2, 0, 11] },
    { spread: 0.7, lift: 0, gather: 0.35, spin: 0.03, color: 0xc4c4c2, cam: [1.4, 0.1, 10] },
    { spread: 0.6, lift: -0.4, gather: 0.4, spin: 0.03, color: 0xe24a1a, cam: [1.1, -0.2, 10] },
    { spread: 0.5, lift: 0, gather: 0.55, spin: 0.02, color: 0xd4d4d2, cam: [0.8, 0, 9.5] },
    { spread: 1.0, lift: 0.2, gather: 0.1, spin: 0.05, color: 0xe0e0e0, cam: [2.6, 0.3, 12] },
    { spread: 0.85, lift: 0, gather: 0.2, spin: 0.04, color: 0xc4c4c2, cam: [2, 0.1, 11] },
    { spread: 0.75, lift: 0, gather: 0.25, spin: 0.03, color: 0xe24a1a, cam: [1.8, 0, 10.5] },
    { spread: 0.9, lift: 0, gather: 0.15, spin: 0.03, color: 0xd4d4d2, cam: [2.4, 0.2, 11] },
    { spread: 0.55, lift: 0, gather: 0.5, spin: 0.02, color: 0xe24a1a, cam: [1, 0.1, 9] },
  ];

  const col = new THREE.Color();
  window.SurakshaScene = {
    enter(index) {
      const L = looks[Math.min(index, looks.length - 1)];
      gsap.to(state, {
        spread: L.spread,
        lift: L.lift,
        gather: L.gather,
        spin: L.spin,
        duration: 1.35,
        ease: "power3.inOut",
      });
      col.set(L.color);
      gsap.to(mesh.material.color, { r: col.r, g: col.g, b: col.b, duration: 1.2 });
      gsap.to(camera.position, {
        x: L.cam[0],
        y: L.cam[1],
        z: L.cam[2],
        duration: 1.4,
        ease: "power2.inOut",
        onUpdate() {
          camera.lookAt(7, 0, 0);
        },
      });
    },
  };
  } catch (err) {
    console.warn("Suraksha scene skipped", err);
    const c = document.getElementById("gl");
    if (c) c.style.display = "none";
  }
})();
