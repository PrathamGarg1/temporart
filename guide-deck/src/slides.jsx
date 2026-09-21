import { motion } from "framer-motion";

const fade = {
  hidden: { opacity: 0, y: 18 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.08 * i, duration: 0.45, ease: [0.22, 1, 0.36, 1] },
  }),
};

export function TitleSlide() {
  return (
    <div className="pad dark-copy">
      <motion.p className="kicker" variants={fade} custom={0} initial="hidden" animate="show">
        IIT Ropar · guide briefing · 20 September 2026
      </motion.p>
      <motion.h1 variants={fade} custom={1} initial="hidden" animate="show">
        Gated Qwen3 conversion yields an <em>84.4%</em> on-device Hinglish detector, with mixing still below human HinGE
      </motion.h1>
      <motion.p className="meta" variants={fade} custom={2} initial="hidden" animate="show">
        Aamod Jain* · Pratham Garg* · Geeta Yadav
        <br />
        Indian Institute of Technology Ropar
      </motion.p>
    </div>
  );
}

export function Situation() {
  return (
    <>
      <h2>The filed poster measured Hindi and English on-device — not Roman Hinglish.</h2>
      <div className="tiles">
        {[
          ["Hindi Test A", "84.66%", "−1.68 pp vs XLM-R, not browser-deployable"],
          ["Hindi + English", "88.01%", "Test B · no Hinglish training split"],
          ["On device", "28 ms", "113 MB INT8 · 76% smaller"],
        ].map(([k, v, d], i) => (
          <motion.div className="card" key={k} variants={fade} custom={i} initial="hidden" animate="show">
            <p className="lab">{k}</p>
            <p className="stat">{v}</p>
            <p>{d}</p>
          </motion.div>
        ))}
      </div>
      <p className="note">Hinglish is named in §1 as motivation. It is not a reported test slice. Train mass was 42,487.</p>
      <p className="cite">Poster Table 1; Gupta et al. (2022), NeurIPS — MACD XLM-R Hindi 86.34%.</p>
    </>
  );
}

export function Complication() {
  const rows = [
    ["Traffic", "WhatsApp in India is Roman Hinglish, not Devanagari Hindi plus monolingual English."],
    ["Method", "Calling Qwen3 is a method only if mixing, meaning, and abuse intensity are measured."],
    ["Refs", "BLEU / WER need human Hinglish references. MACD has none. HinGE does."],
  ];
  return (
    <>
      <h2>Code-mixed abuse was the stated use-case, but we had no converter evaluation.</h2>
      {rows.map(([k, v], i) => (
        <motion.div className="comp" key={k} variants={fade} custom={i} initial="hidden" animate="show">
          <b>{k}</b>
          <p>{v}</p>
        </motion.div>
      ))}
      <p className="cite">Srivastava & Singh (2021), HinGE; Das & Gambäck (2014), CMI.</p>
    </>
  );
}

export function RQ() {
  return (
    <>
      <h2>We ask whether a gated Hindi→Hinglish converter can be measured — without losing the edge detector.</h2>
      <motion.div className="call" variants={fade} initial="hidden" animate="show">
        Can Qwen3-32B Hinglish, under abuse-preservation gates, be scored with HinGE metrics (CMI, BLEU, WER), and still support a browser MiniLM at ≥ poster Hindi accuracy?
      </motion.div>
      <p className="copy">
        Contribution: generation audit of the converter + a frozen Hinglish test slice. The MobiSys system claim (privacy, 28 ms, INT8) is unchanged.
      </p>
    </>
  );
}

export function Methods() {
  return (
    <>
      <h2>We convert MACD Hindi with gated Qwen3 and score mixing on HinGE, not on MACD.</h2>
      <div className="cols">
        <motion.div className="card" variants={fade} custom={0} initial="hidden" animate="show">
          <p className="lab">Pipeline</p>
          <svg className="pipe" viewBox="0 0 320 86" fill="none">
            <rect x="1" y="22" width="86" height="42" rx="8" stroke="#1a1a1a" />
            <text x="44" y="48" textAnchor="middle" fontSize="11" fill="#1a1a1a" fontFamily="Poppins">MACD</text>
            <path d="M96 43h28" stroke="#c45c26" strokeWidth="2" />
            <polygon points="124,38 134,43 124,48" fill="#c45c26" />
            <rect x="140" y="22" width="86" height="42" rx="8" fill="#1a1a1a" />
            <text x="183" y="48" textAnchor="middle" fontSize="11" fill="#fff" fontFamily="Poppins">Qwen3</text>
            <path d="M234 43h28" stroke="#c45c26" strokeWidth="2" />
            <polygon points="262,38 272,43 262,48" fill="#c45c26" />
            <rect x="278" y="22" width="40" height="42" rx="8" stroke="#c45c26" />
            <text x="298" y="48" textAnchor="middle" fontSize="11" fill="#c45c26" fontFamily="Poppins">INT8</text>
          </svg>
          <p><b>Source</b> MACD Hindi (ShareChat)</p>
          <p><b>Generator</b> Qwen3-32B, Roman, Hindi matrix</p>
          <p><b>Detector</b> MiniLM-L12-v2, Chrome ONNX</p>
        </motion.div>
        <motion.div className="card" variants={fade} custom={1} initial="hidden" animate="show">
          <p className="lab">Evaluation</p>
          <p><b>Mixing</b> CMI on 31,663 Qwen lines</p>
          <p><b>NLG</b> BLEU / WER / TER vs HinGE humans (n=1,964)</p>
          <p><b>Detection</b> Frozen Hinglish test n=6,368. Never synth.</p>
        </motion.div>
      </div>
      <p className="cite">Gupta et al. (2022); Srivastava & Singh (2021).</p>
    </>
  );
}

export function Gates() {
  const gates = [
    ["89.9%", "Abuse kept (label 0)", "p_hinglish ≥ p_hindi − 0.15"],
    ["1,997", "Spike rows dropped", "Label 1 rejected if p ≥ 0.5"],
    ["94.7%", "Test yield", "Lexicon, Roman only, no refusals"],
  ];
  return (
    <>
      <h2>A rewrite is kept only if the insult and the script constraints both hold.</h2>
      <div className="three">
        {gates.map(([v, k, d], i) => (
          <motion.div className="card" key={k} variants={fade} custom={i} initial="hidden" animate="show">
            <p className="stat saffron">{v}</p>
            <p className="lab">{k}</p>
            <p>{d}</p>
          </motion.div>
        ))}
      </div>
      <p className="copy">Failures are quarantined, not force-included. HinGE scoring uses Hindi-only input.</p>
    </>
  );
}

export function Detector() {
  return (
    <>
      <h2>INT8 Hinglish test accuracy is 84.4%; Hindi rises 0.48 pp versus the poster.</h2>
      <div className="split">
        <motion.div className="ink-card" variants={fade} initial="hidden" animate="show">
          <p className="lab">HINGLISH · NEW</p>
          <p className="stat">84.36%</p>
          <p>INT8 · F1 84.3% · n=6,368 frozen · 28 ms unchanged</p>
        </motion.div>
        <table>
          <thead>
            <tr><th>Slice</th><th>Poster</th><th>After</th></tr>
          </thead>
          <tbody>
            <tr><td>Hindi Devanagari</td><td>84.66%</td><td>+0.48 → 85.14%</td></tr>
            <tr className="hi"><td>Hinglish</td><td>—</td><td>84.36%</td></tr>
            <tr><td>English Davidson</td><td>in Test B</td><td>96.73%</td></tr>
            <tr><td>Mixed bag</td><td>88.01% Hi+EN</td><td>86.54% +Hinglish</td></tr>
            <tr><td>Train rows</td><td>42,487</td><td>58,929</td></tr>
          </tbody>
        </table>
      </div>
      <p className="cite">Shipped INT8; Hinglish 18,920 / 6,375 / 6,368.</p>
    </>
  );
}

const CMI = [
  ["Qwen MACD", 13.7, true],
  ["Qwen HinGE", 22.7, false],
  ["PAC", 24.3, false],
  ["WAC", 28.9, false],
  ["Humans", 35.6, false],
];

export function Cmi() {
  return (
    <>
      <h2>Mean CMI on 31,663 Qwen lines is 13.7 — well below HinGE human Hinglish (35.6).</h2>
      <div className="cmi">
        <div className="cmi-bars">
          {CMI.map(([lab, v, focus], i) => (
            <div key={lab} className={focus ? "focus" : ""}>
              <span>{lab}</span>
              <i>
                <motion.b
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: v / 40 }}
                  transition={{ delay: 0.12 * i, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                />
              </i>
              <strong>{v}</strong>
            </div>
          ))}
        </div>
        <motion.div className="card" variants={fade} initial="hidden" animate="show">
          <p className="lab">So what</p>
          <p className="stat saffron">13.7 vs 35.6</p>
          <p>25% of MACD rewrites have CMI = 0. 24% reach CMI ≥ 20. MACD is comments; HinGE annotators were instructed to mix.</p>
        </motion.div>
      </div>
      <p className="cite">CMI: Das & Gambäck (2014). HinGE: Srivastava & Singh (2021).</p>
    </>
  );
}

export function Hinge() {
  return (
    <>
      <h2>Qwen BLEU matches PAC on HinGE; WAC wins WER because it uses English parallel.</h2>
      <table>
        <thead>
          <tr><th>System</th><th>CMI ↑</th><th>BLEU ↑</th><th>WER ↓</th><th>TER ↓</th></tr>
        </thead>
        <tbody>
          <tr><td>Human Hinglish</td><td>35.64</td><td>—</td><td>—</td><td>—</td></tr>
          <tr><td>WAC (word swap)</td><td>28.88</td><td>0.164</td><td><b>0.651</b></td><td>0.744</td></tr>
          <tr><td>PAC (phrase swap)</td><td>24.26</td><td><b>0.179</b></td><td>0.671</td><td>0.731</td></tr>
          <tr className="hi"><td>Qwen3-32B, Hindi-only</td><td>22.68</td><td>0.177</td><td>0.713</td><td>0.761</td></tr>
        </tbody>
      </table>
      <p className="copy">Fair comparison: Qwen received Hindi only, matching MACD.</p>
      <p className="cite">LingoIITGN/HinGE, n=1,964. Paper Table 4 was 100 samples.</p>
    </>
  );
}

export function Ablation() {
  const runs = [
    { name: "Shipped", sub: "18,920 Qwen", hi: "84.36%", extra: "Hindi 85.14% · Keep", ink: true },
    { name: "Layer1 clean", sub: "17,962", hi: "83.76%", extra: "Hindi 84.75% · −0.60 pp", ink: false },
    { name: "Layer1 + synth", sub: "977 rows, w=0.5", hi: "83.64%", extra: "Hindi 85.52% · Hinglish fell", ink: false },
  ];
  return (
    <>
      <h2>Cleaning and synthetic augmentation do not beat 84.4% on the frozen Hinglish test.</h2>
      <div className="three">
        {runs.map((r, i) => (
          <motion.div className={r.ink ? "ink-card" : "card"} key={r.name} variants={fade} custom={i} initial="hidden" animate="show">
            <p className="lab">{r.name}</p>
            <p className="stat">{r.hi}</p>
            <p>{r.sub}</p>
            <p>{r.extra}</p>
          </motion.div>
        ))}
      </div>
      <p className="copy">Frozen val: 507 FP / 528 FN. Largest bucket source_label_wrong = 497.</p>
    </>
  );
}

export function Discussion() {
  const pts = [
    ["Do not replace Qwen with WAC/PAC", "They need English parallel; MACD is Hindi-only. No slur gate."],
    ["Do not dump WER on MACD", "No human Hinglish references exist there. HinGE is the NLG bench."],
    ["Poster inset, not a new thesis", "CMI 13.7, abuse kept 89.9%, HinGE BLEU 0.177, Hinglish INT8 84.4%."],
  ];
  return (
    <>
      <h2>Keep Qwen as the converter and WAC/PAC as baselines; the poster remains a detector.</h2>
      <div className="stack">
        {pts.map(([h, b], i) => (
          <motion.div className="card" key={h} variants={fade} custom={i} initial="hidden" animate="show">
            <h3>{h}</h3>
            <p>{b}</p>
          </motion.div>
        ))}
      </div>
    </>
  );
}

export function Conclusions() {
  const lines = [
    "Poster Hindi 84.66% → 85.14% INT8; new Hinglish slice 84.36% at 28 ms.",
    "Gated Qwen conversion is measurable: yield 94%, abuse kept 89.9%, mean CMI 13.7.",
    "On HinGE, Qwen BLEU ≈ PAC; WAC remains the WER baseline (English parallel).",
    "Clean/synth runs lost on frozen Hinglish test — we keep the shipped model.",
  ];
  return (
    <div className="pad dark-copy">
      <p className="kicker">Conclusions · stays up for Q&A</p>
      <div className="conclusions">
        {lines.map((t, i) => (
          <motion.p key={t} variants={fade} custom={i} initial="hidden" animate="show">
            {t}
          </motion.p>
        ))}
      </div>
      <motion.p className="meta" variants={fade} custom={5} initial="hidden" animate="show">
        Aamod Jain, Pratham Garg · {`{2023csb1092, 2023csb1147, geeta}@iitrpr.ac.in`}
        <br />
        Work in progress — feedback welcome. Questions?
      </motion.p>
    </div>
  );
}

export function References() {
  return (
    <>
      <h2>References</h2>
      <ol className="refs">
        <li>Das, A. & Gambäck, B. (2014). Identifying languages at the word level in code-mixed Indian social media text. ICON.</li>
        <li>Davidson, T. et al. (2017). Automated hate speech detection and the problem of offensive language. ICWSM.</li>
        <li>Gupta, V. et al. (2022). Multilingual abusive comment detection at scale for Indic languages. NeurIPS, 35.</li>
        <li>Jacob, B. et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. CVPR.</li>
        <li>Srivastava, V. & Singh, M. (2021). HinGE: A dataset for generation and evaluation of code-mixed Hinglish text. Eval4NLP.</li>
      </ol>
    </>
  );
}

export function AppendixA() {
  return (
    <>
      <h2>Appendix A. Train / val / test protocol after the poster</h2>
      <table>
        <thead>
          <tr><th></th><th>Hindi MACD</th><th>Davidson EN</th><th>Hinglish</th><th>Total</th></tr>
        </thead>
        <tbody>
          <tr><td>Train</td><td>20,183</td><td>19,826 (80%)</td><td>18,920</td><td>58,929</td></tr>
          <tr><td>Val</td><td>6,728</td><td>2,478 (10%)</td><td>6,375 frozen</td><td>15,581</td></tr>
          <tr><td>Test</td><td>6,728</td><td>2,479 (10%)</td><td>6,368 frozen</td><td>15,575</td></tr>
        </tbody>
      </table>
      <p className="copy">Synth (977) and quarantine rescue never enter val/test. Labels: 0 = abusive, 1 = non-abusive.</p>
    </>
  );
}

export function AppendixB() {
  return (
    <>
      <h2>Appendix B. Why WAC/PAC are not the production converter</h2>
      <table>
        <thead>
          <tr><th></th><th>WAC / PAC</th><th>Qwen3 + gates</th></tr>
        </thead>
        <tbody>
          <tr><td>English parallel</td><td>Required</td><td>Not required (MACD)</td></tr>
          <tr><td>Abuse preservation</td><td>None</td><td>Lexicon + ONNX Δ</td></tr>
          <tr><td>Register</td><td>IIT-B news-like</td><td>ShareChat / chat</td></tr>
          <tr><td>Role</td><td>HinGE baseline columns</td><td>Data generator</td></tr>
        </tbody>
      </table>
    </>
  );
}
