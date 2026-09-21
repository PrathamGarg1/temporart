(function () {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FLOW = {
    abc: ["nA", "aAB", "nB", "aBC", "nC"],
    abcd: ["nA", "aAB", "nB", "aBC", "nC", "aTail", "nD"],
    qwen: ["nA", "aAB", "nB", "aBC", "nC", "aTail", "nQ"],
  };
  let tl;

  function flowEls() {
    return document.querySelectorAll("#flowWorld .node, #flowWorld .arr");
  }

  function clearNodeProps() {
    if (typeof gsap === "undefined") return;
    ["nD", "nQ"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) gsap.set(el, { clearProps: "transform,opacity,background,borderColor,x,y,rotate,scale" });
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
    const local = gsap.timeline();
    local
      .to({}, { duration: 0.35 })
      .to(d, { rotate: 8, scale: 0.97, duration: 0.16, ease: "power2.out" })
      .to(d, { backgroundColor: "#fde8e4", borderColor: "#c4473a", duration: 0.12 }, 0)
      .to(d, { x: 120, y: -18, rotate: 22, opacity: 0, scale: 0.92, duration: 0.48, ease: "power3.in" })
      .add(() => setFlow("qwen"))
      .fromTo(
        q,
        { x: -80, rotate: -10, opacity: 0, scale: 0.96 },
        { x: 0, rotate: 0, opacity: 1, scale: 1, duration: 0.58, ease: "power3.out" }
      )
      .add(() => gsap.set(q, { x: 0, y: 0, rotate: 0, opacity: 1, scale: 1 }));
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
      { y: 16, opacity: 0.35 },
      { y: 0, opacity: 1, duration: 0.42, stagger: 0.07, ease: "power3.out" }
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
      tl.fromTo(bits, { y: 18 }, { y: 0, duration: 0.48, stagger: 0.055, ease: "power3.out" });
    }
    if (viz && !flow) {
      tl.fromTo(viz, { y: 18, scale: 0.985 }, { y: 0, scale: 1, duration: 0.62, ease: "power3.out" }, 0);
    }
    if (thought) {
      const th = document.getElementById("thought");
      if (th) tl.fromTo(th, { rotate: -16, y: -12, scale: 0.96 }, { rotate: -7, y: 0, scale: 1, duration: 0.45, ease: "back.out(1.6)" }, 0.35);
    }
    if (tok) tl.to(tok, { left: pct + "%", duration: 0.75, ease: "power2.inOut" }, 0);
    if (window.SurakshaScene) window.SurakshaScene.enter(index);
  };
})();
