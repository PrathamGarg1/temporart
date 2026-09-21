import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import MixField from "./MixField.jsx";
import {
  TitleSlide, Situation, Complication, RQ, Methods, Gates, Detector,
  Cmi, Hinge, Ablation, Discussion, Conclusions, References, AppendixA, AppendixB,
} from "./slides.jsx";

const SLIDES = [
  { id: "title", dark: true, mix: 0.72, node: <TitleSlide /> },
  { id: "sit", node: <Situation /> },
  { id: "comp", node: <Complication /> },
  { id: "rq", node: <RQ /> },
  { id: "meth", node: <Methods /> },
  { id: "gates", node: <Gates /> },
  { id: "det", node: <Detector /> },
  { id: "cmi", mix: 0.9, node: <Cmi /> },
  { id: "hinge", node: <Hinge /> },
  { id: "abl", node: <Ablation /> },
  { id: "disc", node: <Discussion /> },
  { id: "end", dark: true, mix: 0.45, node: <Conclusions /> },
  { id: "ref", node: <References /> },
  { id: "a", node: <AppendixA /> },
  { id: "b", node: <AppendixB /> },
];

export default function App() {
  const [i, setI] = useState(() => {
    const n = parseInt(window.location.hash.replace("#", ""), 10);
    if (Number.isNaN(n)) return 0;
    return Math.max(0, Math.min(SLIDES.length - 1, n - 1));
  });
  const slide = SLIDES[i];
  const mix = slide.mix ?? 0;
  const go = (n) => setI(Math.max(0, Math.min(SLIDES.length - 1, n)));

  useEffect(() => {
    const fromHash = () => {
      const n = parseInt(window.location.hash.replace("#", ""), 10);
      if (!Number.isNaN(n)) go(n - 1);
    };
    fromHash();
    window.addEventListener("hashchange", fromHash);
    return () => window.removeEventListener("hashchange", fromHash);
  }, []);

  useEffect(() => {
    const next = `#${i + 1}`;
    if (window.location.hash !== next) {
      history.replaceState(null, "", next);
    }
  }, [i]);

  useEffect(() => {
    const onKey = (e) => {
      if (["ArrowRight", " ", "PageDown"].includes(e.key)) {
        e.preventDefault();
        go(i + 1);
      }
      if (["ArrowLeft", "PageUp"].includes(e.key)) {
        e.preventDefault();
        go(i - 1);
      }
      if (e.key === "Home") go(0);
      if (e.key === "End") go(SLIDES.length - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [i]);

  const progress = useMemo(() => ((i + 1) / SLIDES.length) * 100, [i]);

  return (
    <div className="theater">
      {<MixField mix={mix || 0.22} dim={!slide.dark} />}
      <header className="chrome">
        <span>IIT Ropar · SurakshaNet</span>
        <span className="track"><i style={{ width: `${progress}%` }} /></span>
        <span className="pos">{i + 1} / {SLIDES.length}</span>
      </header>
      <main className="stage">
        <AnimatePresence mode="wait">
          <motion.article
            key={slide.id}
            className={`board ${slide.dark ? "dark" : ""}`}
            initial={{ opacity: 0, y: 22 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -14 }}
            transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          >
            {slide.node}
          </motion.article>
        </AnimatePresence>
      </main>
      <footer className="chrome bottom">
        <button type="button" onClick={() => go(i - 1)} disabled={i === 0}>Previous</button>
        <button type="button" onClick={() => go(i + 1)} disabled={i === SLIDES.length - 1}>Next</button>
      </footer>
    </div>
  );
}
