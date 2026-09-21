(function () {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FLOW = {
    abc: ["nA", "nB", "nC"],
    abcd: ["nA", "nB", "nC", "nD"],
    qwen: ["nA", "nB", "nC", "nQ"],
  };
  let tl;

  function flowEls() {
    return document.querySelectorAll("#flowWorld .station");
  }

  function drawLine(el, duration) {
    if (!el || typeof gsap === "undefined") return;
    const len = el.getTotalLength ? el.getTotalLength() : 400;
    gsap.fromTo(
      el,
      { strokeDasharray: len, strokeDashoffset: len },
      { strokeDashoffset: 0, duration: duration || 0.7, ease: "power2.inOut" }
    );
  }

  function clearNodeProps() {
    if (typeof gsap === "undefined") return;
    ["nD", "nQ"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.classList.remove("dead");
        gsap.set(el, { clearProps: "transform,opacity,x,y,rotate,scale" });
      }
    });
  }

  function setFlow(mode) {
    const ids = FLOW[mode] || [];
    flowEls().forEach((el) => el.classList.toggle("is-on", ids.includes(el.id)));
  }

  function morphToQwen() {
    const d = document.getElementById("nD");
    const q = document.getElementById("nQ");
    if (!d || !q || typeof gsap === "undefined" || reduce) {
      setFlow("qwen");
      return;
    }
    clearNodeProps();
    setFlow("abcd");
    d.classList.add("dead");
    const local = gsap.timeline();
    local
      .to({}, { duration: 0.28 })
      .to(d, { x: 140, y: -10, rotate: 12, opacity: 0, duration: 0.5, ease: "power3.in" })
      .add(() => {
        d.classList.remove("dead");
        setFlow("qwen");
      })
      .fromTo(
        q,
        { x: -72, opacity: 0 },
        { x: 0, opacity: 1, duration: 0.55, ease: "power3.out" }
      )
      .add(() => gsap.set(q, { x: 0, y: 0, rotate: 0, opacity: 1 }));
  }

  function enterFlow(mode, morph) {
    document.body.classList.add("flowing");
    if (morph) {
      morphToQwen();
      return;
    }
    clearNodeProps();
    setFlow(mode);
    if (typeof gsap === "undefined" || reduce) return;
    const shown = Array.from(flowEls()).filter((el) => el.classList.contains("is-on"));
    gsap.fromTo(
      shown,
      { y: 10 },
      { y: 0, duration: 0.42, stagger: 0.12, ease: "power3.out" }
    );
  }

  window.playSlideMotion = function (slide, index) {
    if (!slide) return;
    if (tl) tl.kill();

    const flow = slide.dataset.flow || "";
    const bubble = slide.dataset.bubble === "1";
    const thought = slide.dataset.thought === "1";
    const morph = slide.dataset.morph === "1";
    const x = Number(slide.dataset.token || 40);
    const pct = ((x - 40) / 1120) * 88;

    document.getElementById("thought")?.classList.toggle("on", thought);
    document.getElementById("heroMsg")?.classList.toggle("on", bubble);

    if (flow) enterFlow(flow, morph);
    else {
      document.body.classList.remove("flowing");
      clearNodeProps();
      setFlow("");
    }

    slide.querySelectorAll(".bar i").forEach((el) => {
      el.style.setProperty("--w", el.dataset.w || "0");
    });

    const bits = slide.querySelectorAll("[data-in]");
    const tok = document.getElementById("token");
    const viz = slide.querySelector(".viz");

    if (typeof gsap === "undefined" || reduce) {
      if (tok) tok.style.left = pct + "%";
      return;
    }

    tl = gsap.timeline();
    if (bits.length) {
      tl.fromTo(bits, { y: 14 }, { y: 0, duration: 0.42, stagger: 0.07, ease: "power3.out" });
    }
    slide.querySelectorAll("[data-draw]").forEach((el) => drawLine(el, 0.8));
    if (viz && !flow) {
      tl.fromTo(viz, { y: 12 }, { y: 0, duration: 0.55, ease: "power3.out" }, 0);
    }
    if (thought) {
      const th = document.getElementById("thought");
      if (th) tl.fromTo(th, { x: 16 }, { x: 0, duration: 0.4, ease: "power3.out" }, 0.28);
    }
    if (tok) tl.to(tok, { left: pct + "%", duration: 0.75, ease: "power2.inOut" }, 0);
    if (window.SurakshaScene) window.SurakshaScene.enter(index);
  };
})();
