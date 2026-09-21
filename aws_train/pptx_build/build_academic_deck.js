#!/usr/bin/env node
/**
 * Guide briefing — academic argument (SCR + action titles)
 * implemented with Anthropic pptx skill (pptxgenjs, sandwich, native charts).
 */
const PptxGenJS = require("pptxgenjs");
const path = require("path");

const C = {
  ink: "1A1A1A",
  paper: "FFFFFF",
  stone: "F3F3F3",
  rule: "E2E2E2",
  mute: "6B6B6B",
  body: "2A2A2A",
  saffron: "C45C26",
  saffronSoft: "F6E8DF",
  white: "FFFFFF",
};
const TITLE = "Poppins";
const BODY = "Poppins";
const OUT = path.join(__dirname, "../../codemix_eval/SurakshaNet_Academic_Briefing.pptx");

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";
pres.author = "Aamod Jain, Pratham Garg";
pres.title = "Measuring Hinglish conversion after the SurakshaNet poster";

function actionTitle(s, text) {
  s.addText(text, {
    x: 0.55, y: 0.32, w: 12.2, h: 1.15,
    fontFace: TITLE, fontSize: 26, bold: true, color: C.ink, valign: "top", margin: 0,
  });
}
function page(s, n) {
  s.addText(String(n), {
    x: 12.35, y: 7.15, w: 0.5, h: 0.22,
    fontFace: BODY, fontSize: 11, color: C.mute, align: "right", margin: 0,
  });
}
function source(s, text) {
  s.addText(text, {
    x: 0.55, y: 7.12, w: 11.6, h: 0.24,
    fontFace: BODY, fontSize: 12, color: C.mute, margin: 0,
  });
}
function notes(s, text) {
  s.addNotes(text);
}

// 1 Title — dark sandwich
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.ink }, line: { color: C.ink },
  });
  s.addText("IIT Ropar  ·  guide briefing  ·  20 September 2026", {
    x: 0.7, y: 1.15, w: 12, h: 0.32, fontFace: BODY, fontSize: 14, color: "A3A3A3", margin: 0,
  });
  s.addText([
    { text: "Gated Qwen3 conversion yields an ", options: { breakLine: false } },
    { text: "84.4%", options: { color: C.saffron, breakLine: false } },
    { text: " on-device Hinglish detector, with mixing still below human HinGE", options: { breakLine: false } },
  ], {
    x: 0.7, y: 1.7, w: 12, h: 2.35,
    fontFace: TITLE, fontSize: 32, bold: true, color: C.white, valign: "top", margin: 0,
  });
  s.addText("Aamod Jain*  ·  Pratham Garg*  ·  Geeta Yadav\nIndian Institute of Technology Ropar", {
    x: 0.7, y: 5.55, w: 12, h: 0.7, fontFace: BODY, fontSize: 16, color: "C8C8C8", margin: 0,
  });
  notes(s, "Open with the claim, not the stack. Poster is still the detector. Today is the missing Hinglish measurement.");
}

// 2 Situation — stat tiles
{
  const s = pres.addSlide();
  actionTitle(s, "The filed poster measured Hindi and English on-device — not Roman Hinglish.");
  const tiles = [
    { k: "Hindi Test A", v: "84.66%", d: "−1.68 pp vs XLM-R\n(not browser-deployable)" },
    { k: "Hindi + English", v: "88.01%", d: "Test B  ·  no Hinglish\ntraining split" },
    { k: "On device", v: "28 ms", d: "113 MB INT8  ·  76% smaller\nquant. loss < 0.11%" },
  ];
  tiles.forEach((t, i) => {
    const x = 0.55 + i * 4.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.7, w: 3.95, h: 3.35,
      fill: { color: C.stone }, line: { color: C.stone }, rectRadius: 0.08,
    });
    s.addText(t.k, {
      x: x + 0.28, y: 1.9, w: 3.4, h: 0.35, fontFace: BODY, fontSize: 14, color: C.mute, margin: 0,
    });
    s.addText(t.v, {
      x: x + 0.28, y: 2.35, w: 3.4, h: 1.05, fontFace: TITLE, fontSize: 36, bold: true, color: C.ink, margin: 0,
    });
    s.addText(t.d, {
      x: x + 0.28, y: 3.55, w: 3.4, h: 1.2, fontFace: BODY, fontSize: 16, color: C.body, margin: 0,
    });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.55, y: 5.25, w: 12.25, h: 1.55,
    fill: { color: C.saffronSoft }, line: { color: C.saffronSoft }, rectRadius: 0.08,
  });
  s.addText("Hinglish is named in §1 as motivation. It is not a reported test slice. Train mass was 42,487 (MACD Hindi + Davidson 90%).", {
    x: 0.8, y: 5.5, w: 11.75, h: 1.1, fontFace: BODY, fontSize: 18, color: C.ink, valign: "middle", margin: 0,
  });
  source(s, "Poster Table 1; Gupta et al. (2022), NeurIPS — MACD XLM-R Hindi 86.34%.");
  page(s, 2);
  notes(s, "Do not relitigate the poster. The gap is the missing Roman Hinglish slice the use-case already named.");
}

// 3 Complication
{
  const s = pres.addSlide();
  actionTitle(s, "Code-mixed abuse was the stated use-case, but we had no converter evaluation.");
  const rows = [
    ["Traffic", "WhatsApp in India is Roman Hinglish, not Devanagari Hindi plus monolingual English."],
    ["Method", "Calling Qwen3 is a method only if mixing, meaning, and abuse intensity are measured."],
    ["Refs", "BLEU / WER need human Hinglish references. MACD has none. HinGE does."],
  ];
  rows.forEach((r, i) => {
    const y = 1.7 + i * 1.55;
    s.addText(r[0], {
      x: 0.55, y, w: 2.2, h: 1.25, fontFace: TITLE, fontSize: 20, bold: true, color: C.saffron, valign: "middle", margin: 0,
    });
    s.addText(r[1], {
      x: 2.9, y, w: 9.9, h: 1.25, fontFace: BODY, fontSize: 22, color: C.body, valign: "middle", margin: 0,
    });
  });
  source(s, "Srivastava & Singh (2021), HinGE, Eval4NLP; Das & Gambäck (2014), CMI.");
  page(s, 3);
  notes(s, "Guide objection in one sentence: 'you called Qwen.' Answer: then we scored it on HinGE and froze a Hinglish test.");
}

// 4 RQ
{
  const s = pres.addSlide();
  actionTitle(s, "We ask whether a gated Hindi→Hinglish converter can be measured — without losing the edge detector.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.55, y: 1.75, w: 12.25, h: 2.7,
    fill: { color: C.stone }, line: { color: C.stone }, rectRadius: 0.08,
  });
  s.addText("Can Qwen3-32B Hinglish, under abuse-preservation gates, be scored with HinGE metrics (CMI, BLEU, WER), and still support a browser MiniLM at ≥ poster Hindi accuracy?", {
    x: 0.9, y: 2.0, w: 11.55, h: 2.2,
    fontFace: TITLE, fontSize: 24, color: C.ink, valign: "middle", margin: 0,
  });
  s.addText("Contribution: generation audit of the converter + a frozen Hinglish test slice. The MobiSys system claim (privacy, 28 ms, INT8) is unchanged.", {
    x: 0.55, y: 4.75, w: 12.25, h: 1.5, fontFace: BODY, fontSize: 20, color: C.body, margin: 0,
  });
  page(s, 4);
  notes(s, "One RQ. If they ask about WAC as the converter, that is Appendix B.");
}

// 5 Methods — two columns
{
  const s = pres.addSlide();
  actionTitle(s, "We convert MACD Hindi with gated Qwen3 and score mixing on HinGE, not on MACD.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.55, y: 1.7, w: 6.0, h: 4.9,
    fill: { color: C.stone }, line: { color: C.stone }, rectRadius: 0.08,
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.8, y: 1.7, w: 6.0, h: 4.9,
    fill: { color: C.stone }, line: { color: C.stone }, rectRadius: 0.08,
  });
  s.addText("Pipeline", {
    x: 0.85, y: 1.9, w: 5.4, h: 0.4, fontFace: TITLE, fontSize: 20, bold: true, color: C.ink, margin: 0,
  });
  s.addText("Evaluation", {
    x: 7.1, y: 1.9, w: 5.4, h: 0.4, fontFace: TITLE, fontSize: 20, bold: true, color: C.ink, margin: 0,
  });
  s.addText([
    { text: "Source  ", options: { bold: true } },
    { text: "MACD Hindi comments (ShareChat).", options: { breakLine: true } },
    { text: "Generator  ", options: { bold: true } },
    { text: "Qwen3-32B, Roman Hinglish, Hindi matrix.", options: { breakLine: true } },
    { text: "Detector  ", options: { bold: true } },
    { text: "MiniLM-L12-v2, INT8 ONNX, same Chrome path.", options: { breakLine: false } },
  ], {
    x: 0.85, y: 2.5, w: 5.4, h: 3.7, fontFace: BODY, fontSize: 18, color: C.body, paraSpaceAfter: 16, margin: 0,
  });
  s.addText([
    { text: "Mixing  ", options: { bold: true } },
    { text: "CMI on 31,663 Qwen lines.", options: { breakLine: true } },
    { text: "NLG  ", options: { bold: true } },
    { text: "BLEU / WER / TER vs HinGE humans (n=1,964).", options: { breakLine: true } },
    { text: "Detection  ", options: { bold: true } },
    { text: "Frozen Hinglish test n=6,368. Never synth.", options: { breakLine: false } },
  ], {
    x: 7.1, y: 2.5, w: 5.4, h: 3.7, fontFace: BODY, fontSize: 18, color: C.body, paraSpaceAfter: 16, margin: 0,
  });
  source(s, "Gupta et al. (2022); Srivastava & Singh (2021); sentence-transformers MiniLM-L12-v2.");
  page(s, 5);
  notes(s, "Stress the split: CMI on our data, BLEU/WER only on HinGE. Never dump WER on MACD.");
}

// 6 Gates — three outcomes
{
  const s = pres.addSlide();
  actionTitle(s, "A rewrite is kept only if the insult and the script constraints both hold.");
  const gates = [
    { v: "89.9%", k: "Abuse kept (label 0)", d: "p_hinglish ≥ p_hindi − 0.15\nTest still p ≥ 0.5" },
    { v: "1,997", k: "Spike rows dropped", d: "Label 1 rejected if p ≥ 0.5\nand > hindi + 0.15" },
    { v: "94.7%", k: "Test yield", d: "Lexicon stems, Roman only,\nno model refusals" },
  ];
  gates.forEach((g, i) => {
    const x = 0.55 + i * 4.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.7, w: 3.95, h: 4.0,
      fill: { color: C.stone }, line: { color: C.stone }, rectRadius: 0.08,
    });
    s.addText(g.v, {
      x: x + 0.28, y: 2.0, w: 3.4, h: 1.15, fontFace: TITLE, fontSize: 40, bold: true, color: C.saffron, margin: 0,
    });
    s.addText(g.k, {
      x: x + 0.28, y: 3.25, w: 3.4, h: 0.55, fontFace: TITLE, fontSize: 18, bold: true, color: C.ink, margin: 0,
    });
    s.addText(g.d, {
      x: x + 0.28, y: 3.9, w: 3.4, h: 1.4, fontFace: BODY, fontSize: 16, color: C.body, margin: 0,
    });
  });
  s.addText("Failures are quarantined, not force-included. HinGE scoring uses Hindi-only input — the same constraint as MACD.", {
    x: 0.55, y: 5.9, w: 12.25, h: 0.9, fontFace: BODY, fontSize: 18, color: C.body, margin: 0,
  });
  page(s, 6);
  notes(s, "Gates are why we do not swap in WAC/PAC: those have no slur preservation.");
}

// 7 Detector — oversized number + table
{
  const s = pres.addSlide();
  actionTitle(s, "INT8 Hinglish test accuracy is 84.4%; Hindi rises 0.48 pp versus the poster.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.55, y: 1.65, w: 4.15, h: 4.95,
    fill: { color: C.ink }, line: { color: C.ink }, rectRadius: 0.08,
  });
  s.addText("HINGLISH  ·  NEW", {
    x: 0.8, y: 1.9, w: 3.65, h: 0.35, fontFace: BODY, fontSize: 13, color: "B0B0B0", margin: 0,
  });
  s.addText("84.36%", {
    x: 0.8, y: 2.45, w: 3.65, h: 1.3, fontFace: TITLE, fontSize: 48, bold: true, color: C.saffron, margin: 0,
  });
  s.addText("INT8  ·  F1 84.3%\nn = 6,368 frozen\n28 ms unchanged", {
    x: 0.8, y: 3.9, w: 3.65, h: 2.2, fontFace: BODY, fontSize: 18, color: C.white, margin: 0,
  });
  s.addTable(
    [
      [
        { text: "Slice", options: { fill: { color: C.stone }, color: C.ink, bold: true } },
        { text: "Poster", options: { fill: { color: C.stone }, color: C.ink, bold: true } },
        { text: "After", options: { fill: { color: C.stone }, color: C.ink, bold: true } },
      ],
      ["Hindi Devanagari", "84.66%", { text: "+0.48 → 85.14%", options: { bold: true } }],
      [
        { text: "Hinglish", options: { fill: { color: C.saffronSoft }, bold: true } },
        { text: "—", options: { fill: { color: C.saffronSoft } } },
        { text: "84.36%", options: { fill: { color: C.saffronSoft }, bold: true, color: C.saffron } },
      ],
      ["English Davidson", "in Test B", "96.73%"],
      ["Mixed bag", "88.01% Hi+EN", "86.54% +Hinglish"],
      ["Train rows", "42,487", "58,929"],
    ],
    {
      x: 4.95, y: 1.65, w: 7.85, h: 4.95, colW: [2.55, 2.35, 2.95],
      border: [{ pt: 0, color: C.paper }],
      fontFace: BODY, fontSize: 16, color: C.body, valign: "middle",
    }
  );
  source(s, "Shipped INT8; Hindi 20,183 + Davidson 80/10/10 + Hinglish 18,920 / 6,375 / 6,368.");
  page(s, 7);
  notes(s, "Mixed bag dropped because the mix got harder, not because Hindi got worse. Hindi improved.");
}

// 8 CMI chart
{
  const s = pres.addSlide();
  actionTitle(s, "Mean CMI on 31,663 Qwen lines is 13.7 — well below HinGE human Hinglish (35.6).");
  s.addChart(pres.charts.BAR, [
    {
      name: "Mean CMI",
      labels: ["Qwen MACD", "Qwen HinGE", "PAC", "WAC", "Humans"],
      values: [13.7, 22.7, 24.3, 28.9, 35.6],
    },
  ], {
    x: 0.35, y: 1.45, w: 8.1, h: 5.3,
    barGrouping: "clustered",
    showValue: true,
    showLegend: false,
    showTitle: false,
    chartColors: [C.ink],
    chartValueColor: C.ink,
    catAxisLabelColor: C.body,
    valAxisLabelColor: C.mute,
    catAxisLabelFontSize: 12,
    valAxisLabelFontSize: 11,
    valAxisMaxValue: 40,
    valGridLine: { color: C.rule, size: 0.5 },
    catGridLine: { style: "none" },
    chartArea: { fill: { color: C.paper } },
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 8.55, y: 1.7, w: 4.25, h: 4.85,
    fill: { color: C.stone }, line: { color: C.stone }, rectRadius: 0.08,
  });
  s.addText("So what", {
    x: 8.85, y: 1.95, w: 3.75, h: 0.35, fontFace: BODY, fontSize: 13, color: C.mute, margin: 0,
  });
  s.addText("13.7 vs 35.6", {
    x: 8.85, y: 2.35, w: 3.75, h: 0.7, fontFace: TITLE, fontSize: 28, bold: true, color: C.saffron, margin: 0,
  });
  s.addText("25% of MACD rewrites have CMI = 0 (romanized Hindi or English-only). 24% reach CMI ≥ 20. MACD is comments; HinGE annotators were instructed to mix.", {
    x: 8.85, y: 3.2, w: 3.75, h: 2.9, fontFace: BODY, fontSize: 16, color: C.body, margin: 0,
  });
  source(s, "CMI: Das & Gambäck (2014). HinGE humans: Srivastava & Singh (2021).");
  page(s, 8);
  notes(s, "Low CMI is a finding, not a bug we hid. Domain mismatch with HinGE humans.");
}

// 9 HinGE table
{
  const s = pres.addSlide();
  actionTitle(s, "Qwen BLEU matches PAC on HinGE; WAC wins WER because it uses English parallel.");
  s.addTable(
    [
      [
        { text: "System", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "CMI ↑", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "BLEU ↑", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "WER ↓", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "TER ↓", options: { fill: { color: C.ink }, color: C.white, bold: true } },
      ],
      ["Human Hinglish", "35.64", "—", "—", "—"],
      ["WAC (word swap)", "28.88", "0.164", { text: "0.651", options: { bold: true } }, "0.744"],
      ["PAC (phrase swap)", "24.26", { text: "0.179", options: { bold: true } }, "0.671", "0.731"],
      [
        { text: "Qwen3-32B, Hindi-only", options: { fill: { color: C.saffronSoft }, bold: true } },
        { text: "22.68", options: { fill: { color: C.saffronSoft } } },
        { text: "0.177", options: { fill: { color: C.saffronSoft }, bold: true, color: C.saffron } },
        { text: "0.713", options: { fill: { color: C.saffronSoft } } },
        { text: "0.761", options: { fill: { color: C.saffronSoft } } },
      ],
    ],
    {
      x: 0.55, y: 1.65, w: 12.25, h: 3.85, colW: [4.15, 2.025, 2.025, 2.025, 2.025],
      border: [{ pt: 0, color: C.paper }],
      fontFace: BODY, fontSize: 18, color: C.body, valign: "middle", align: "center",
    }
  );
  s.addText("Fair comparison: Qwen received Hindi only, matching MACD. WAC/PAC align English nouns/phrases into a Hindi matrix.", {
    x: 0.55, y: 5.7, w: 12.25, h: 1.05, fontFace: BODY, fontSize: 18, color: C.body, margin: 0,
  });
  source(s, "LingoIITGN/HinGE, n=1,964 accepted / 1,976. Paper Table 4 was 100 samples.");
  page(s, 9);
  notes(s, "Do not claim we beat WAC. We match PAC on BLEU without English parallel.");
}

// 10 Ablations
{
  const s = pres.addSlide();
  actionTitle(s, "Cleaning and synthetic augmentation do not beat 84.4% on the frozen Hinglish test.");
  const runs = [
    { name: "Shipped", sub: "18,920 Qwen", hi: "84.36%", hindi: "85.14%", ship: "Keep", ink: true },
    { name: "Layer1 clean", sub: "17,962", hi: "83.76%", hindi: "84.75%", ship: "−0.60 pp", ink: false },
    { name: "Layer1 + synth", sub: "977 rows, w=0.5", hi: "83.64%", hindi: "85.52%", ship: "Hinglish fell", ink: false },
  ];
  runs.forEach((r, i) => {
    const x = 0.55 + i * 4.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.7, w: 3.95, h: 4.15,
      fill: { color: r.ink ? C.ink : C.stone },
      line: { color: r.ink ? C.ink : C.stone },
      rectRadius: 0.08,
    });
    s.addText(r.name, {
      x: x + 0.28, y: 1.95, w: 3.4, h: 0.4,
      fontFace: TITLE, fontSize: 20, bold: true, color: r.ink ? C.white : C.ink, margin: 0,
    });
    s.addText(r.sub, {
      x: x + 0.28, y: 2.4, w: 3.4, h: 0.35, fontFace: BODY, fontSize: 14, color: r.ink ? "B0B0B0" : C.mute, margin: 0,
    });
    s.addText(r.hi, {
      x: x + 0.28, y: 2.95, w: 3.4, h: 0.95,
      fontFace: TITLE, fontSize: 36, bold: true, color: r.ink ? C.saffron : C.ink, margin: 0,
    });
    s.addText("Hinglish INT8", {
      x: x + 0.28, y: 3.9, w: 3.4, h: 0.3, fontFace: BODY, fontSize: 13, color: r.ink ? "B0B0B0" : C.mute, margin: 0,
    });
    s.addText(`Hindi ${r.hindi}\n${r.ship}`, {
      x: x + 0.28, y: 4.35, w: 3.4, h: 1.15, fontFace: BODY, fontSize: 16, color: r.ink ? C.white : C.body, margin: 0,
    });
  });
  s.addText("Frozen val: 507 FP / 528 FN. Largest bucket source_label_wrong = 497. Test was never unfrozen.", {
    x: 0.55, y: 6.05, w: 12.25, h: 0.8, fontFace: BODY, fontSize: 16, color: C.body, margin: 0,
  });
  page(s, 10);
  notes(s, "Negative result on purpose. Volume was not the bottleneck. Do not unfreeze test.");
}

// 11 Discussion
{
  const s = pres.addSlide();
  actionTitle(s, "Keep Qwen as the converter and WAC/PAC as baselines; the poster remains a detector.");
  const pts = [
    { h: "Do not replace Qwen with WAC/PAC", b: "They need English parallel; MACD is Hindi-only. They have no slur gate." },
    { h: "Do not dump WER on MACD", b: "No human Hinglish references exist there. HinGE is the NLG bench." },
    { h: "Poster inset, not a new thesis", b: "CMI 13.7, abuse kept 89.9%, HinGE BLEU 0.177, Hinglish INT8 84.4%." },
  ];
  pts.forEach((p, i) => {
    const y = 1.65 + i * 1.65;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.55, y, w: 12.25, h: 1.5,
      fill: { color: C.stone }, line: { color: C.stone }, rectRadius: 0.08,
    });
    s.addText(p.h, {
      x: 0.85, y: y + 0.18, w: 11.65, h: 0.45, fontFace: TITLE, fontSize: 20, bold: true, color: C.ink, margin: 0,
    });
    s.addText(p.b, {
      x: 0.85, y: y + 0.7, w: 11.65, h: 0.55, fontFace: BODY, fontSize: 18, color: C.body, margin: 0,
    });
  });
  page(s, 11);
  notes(s, "If time is short, skip to conclusions after this slide.");
}

// 12 Conclusions — dark sandwich, stays for Q&A
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.ink }, line: { color: C.ink },
  });
  s.addText("Conclusions", {
    x: 0.7, y: 0.45, w: 12, h: 0.5, fontFace: BODY, fontSize: 14, color: "A3A3A3", margin: 0,
  });
  const lines = [
    "Poster Hindi 84.66% → 85.14% INT8; new Hinglish slice 84.36% at 28 ms.",
    "Gated Qwen conversion is measurable: yield 94%, abuse kept 89.9%, mean CMI 13.7.",
    "On HinGE, Qwen BLEU ≈ PAC; WAC remains the WER baseline (English parallel).",
    "Clean/synth runs lost on frozen Hinglish test — we keep the shipped model.",
  ];
  lines.forEach((t, i) => {
    s.addText(t, {
      x: 0.7, y: 1.15 + i * 1.05, w: 12, h: 0.9,
      fontFace: TITLE, fontSize: 22, color: C.white, margin: 0,
    });
  });
  s.addText("Aamod Jain, Pratham Garg  ·  {2023csb1092, 2023csb1147, geeta}@iitrpr.ac.in\nWork in progress — feedback welcome. Questions?", {
    x: 0.7, y: 5.55, w: 12, h: 0.85, fontFace: BODY, fontSize: 16, color: "B8B8B8", margin: 0,
  });
  notes(s, "Stop here for Q&A. Do not advance to references unless asked.");
}

// 13 References
{
  const s = pres.addSlide();
  actionTitle(s, "References");
  s.addText([
    { text: "Das, A. & Gambäck, B. (2014). Identifying languages at the word level in code-mixed Indian social media text. ICON.", options: { breakLine: true } },
    { text: "Davidson, T. et al. (2017). Automated hate speech detection and the problem of offensive language. ICWSM.", options: { breakLine: true } },
    { text: "Gupta, V. et al. (2022). Multilingual abusive comment detection at scale for Indic languages. NeurIPS, 35.", options: { breakLine: true } },
    { text: "Jacob, B. et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. CVPR.", options: { breakLine: true } },
    { text: "Srivastava, V. & Singh, M. (2021). HinGE: A dataset for generation and evaluation of code-mixed Hinglish text. Eval4NLP.", options: { breakLine: false } },
  ], {
    x: 0.55, y: 1.65, w: 12.25, h: 5.0, fontFace: BODY, fontSize: 18, color: C.body, paraSpaceAfter: 14, margin: 0,
  });
  page(s, 13);
}

// 14 Appendix A
{
  const s = pres.addSlide();
  actionTitle(s, "Appendix A. Train / val / test protocol after the poster");
  s.addTable(
    [
      [
        { text: "", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "Hindi MACD", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "Davidson EN", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "Hinglish", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "Total", options: { fill: { color: C.ink }, color: C.white, bold: true } },
      ],
      ["Train", "20,183", "19,826 (80%)", "18,920", "58,929"],
      ["Val", "6,728", "2,478 (10%)", "6,375 frozen", "15,581"],
      ["Test", "6,728", "2,479 (10%)", "6,368 frozen", "15,575"],
    ],
    {
      x: 0.55, y: 1.7, w: 12.25, h: 3.5, colW: [2.0, 2.5, 2.75, 2.7, 2.3],
      border: [{ pt: 0, color: C.paper }],
      fontFace: BODY, fontSize: 18, color: C.body, valign: "middle",
    }
  );
  s.addText("Synth (977) and quarantine rescue never enter val/test. Labels: 0 = abusive, 1 = non-abusive.", {
    x: 0.55, y: 5.5, w: 12.25, h: 1.2, fontFace: BODY, fontSize: 18, color: C.body, margin: 0,
  });
  page(s, 14);
}

// 15 Appendix B
{
  const s = pres.addSlide();
  actionTitle(s, "Appendix B. Why WAC/PAC are not the production converter");
  s.addTable(
    [
      [
        { text: "", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "WAC / PAC", options: { fill: { color: C.ink }, color: C.white, bold: true } },
        { text: "Qwen3 + gates", options: { fill: { color: C.ink }, color: C.white, bold: true } },
      ],
      ["English parallel", "Required", "Not required (MACD)"],
      ["Abuse preservation", "None", "Lexicon + ONNX Δ"],
      ["Register", "IIT-B news-like", "ShareChat / chat"],
      ["Role", "HinGE baseline columns", "Data generator"],
    ],
    {
      x: 0.55, y: 1.7, w: 12.25, h: 4.4, colW: [3.4, 4.425, 4.425],
      border: [{ pt: 0, color: C.paper }],
      fontFace: BODY, fontSize: 18, color: C.body, valign: "middle",
    }
  );
  page(s, 15);
}

pres.writeFile({ fileName: OUT }).then(() => console.log(OUT));
