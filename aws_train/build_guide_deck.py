#!/usr/bin/env python3
"""Build the guide briefing deck for SurakshaNet Hinglish + HinGE eval."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import nsmap, qn
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parents[1] / "codemix_eval" / "SurakshaNet_Guide_Briefing.pptx"

NAVY = RGBColor(0x0B, 0x12, 0x20)
CARD = RGBColor(0x14, 0x1C, 0x2E)
LINE = RGBColor(0x2A, 0x36, 0x4D)
WHITE = RGBColor(0xF4, 0xF7, 0xFB)
MUTED = RGBColor(0x9A, 0xA8, 0xBD)
ACCENT = RGBColor(0x3D, 0xDC, 0xC7)
GOLD = RGBColor(0xE8, 0xC3, 0x6A)
RED = RGBColor(0xF0, 0x71, 0x67)
OK = RGBColor(0x6F, 0xCF, 0x97)

W, H = Inches(13.333), Inches(7.5)


def set_run(run, text, size=18, color=WHITE, bold=False, font="Calibri"):
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = font


def add_text(tf, text, size=18, color=WHITE, bold=False, align=PP_ALIGN.LEFT, space_after=6):
    p = tf.paragraphs[0] if not tf.paragraphs[0].text else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    run = p.add_run()
    set_run(run, text, size, color, bold)
    return p


def box(slide, l, t, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line or fill
    s.adjustments[0] = 0.08
    return s


def rect(slide, l, t, w, h, fill):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    return s


def tb(slide, l, t, w, h):
    return slide.shapes.add_textbox(l, t, w, h).text_frame


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def new_slide(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    rect(sl, 0, 0, W, H, NAVY)
    rect(sl, 0, 0, Inches(0.12), H, ACCENT)
    return sl


def footer(sl, n, total=12):
    tf = tb(sl, Inches(0.5), Inches(7.15), Inches(10), Inches(0.28))
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, "SurakshaNet  ·  IIT Ropar  ·  Guide briefing  ·  confidential research numbers", 10, MUTED)
    tf2 = tb(sl, Inches(11.6), Inches(7.15), Inches(1.3), Inches(0.28))
    p = tf2.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    set_run(r, f"{n} / {total}", 10, MUTED)


def metric_card(sl, l, t, w, h, value, label, color=ACCENT):
    box(sl, l, t, w, h, CARD, LINE)
    tf = tb(sl, l + Inches(0.18), t + Inches(0.18), w - Inches(0.3), Inches(0.55))
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, value, 28, color, True)
    tf2 = tb(sl, l + Inches(0.18), t + Inches(0.72), w - Inches(0.3), Inches(0.55))
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    r = p.add_run()
    set_run(r, label, 12, MUTED, False)


def bullets(tf, items, size=16):
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        p.space_after = Pt(8)
        r = p.add_run()
        set_run(r, "▸  " + item, size, WHITE)


def title_block(sl, kicker, title, subtitle=None):
    tf = tb(sl, Inches(0.5), Inches(0.28), Inches(12.3), Inches(0.28))
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, kicker.upper(), 11, ACCENT, True)
    tf = tb(sl, Inches(0.5), Inches(0.52), Inches(12.3), Inches(0.7))
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, title, 28, WHITE, True)
    if subtitle:
        tf = tb(sl, Inches(0.5), Inches(1.18), Inches(12.3), Inches(0.4))
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        set_run(r, subtitle, 14, MUTED)


def add_table(sl, left, top, width, height, headers, rows, col_w=None):
    table = sl.shapes.add_table(1 + len(rows), len(headers), left, top, width, height).table
    if col_w:
        for i, w in enumerate(col_w):
            table.columns[i].width = w
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0x1B, 0x2A, 0x40)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            for r in p.runs:
                set_run(r, h, 12, ACCENT, True)
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD if i % 2 else RGBColor(0x11, 0x18, 0x28)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
                for r in p.runs:
                    bold = i == len(rows)
                    color = GOLD if bold else WHITE
                    set_run(r, val, 13, color, bold)
    return table


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 title
    sl = new_slide(prs)
    tf = tb(sl, Inches(0.7), Inches(1.7), Inches(12), Inches(0.35))
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, "IIT ROPAR  ·  MOBISYS POSTER  ·  GUIDE BRIEFING", 12, ACCENT, True)
    tf = tb(sl, Inches(0.7), Inches(2.15), Inches(12), Inches(1.6))
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, "We did not just call Qwen3.\nWe measured the Hinglish.", 36, WHITE, True)
    tf = tb(sl, Inches(0.7), Inches(4.0), Inches(11), Inches(0.8))
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(
        r,
        "SurakshaNet stays the system. HinGE is the evaluation protocol for the converter.",
        18,
        MUTED,
    )
    metric_card(sl, Inches(0.7), Inches(5.15), Inches(2.9), Inches(1.35), "84.4%", "INT8 Hinglish test (shipped)")
    metric_card(sl, Inches(3.8), Inches(5.15), Inches(2.9), Inches(1.35), "13.7", "Mean CMI on MACD Qwen")
    metric_card(sl, Inches(6.9), Inches(5.15), Inches(2.9), Inches(1.35), "0.177", "Qwen BLEU vs HinGE humans")
    metric_card(sl, Inches(10.0), Inches(5.15), Inches(2.6), Inches(1.35), "89.9%", "Abuse preserved on test")
    notes(
        sl,
        "Open with the split: the MobiSys poster is still privacy-first edge abuse detection. "
        "The guide asked us to stop treating Qwen as a black box. This deck is the measurement. "
        "Four numbers to remember: 84.4 detector, 13.7 CMI, 0.177 BLEU, 89.9 abuse kept.",
    )
    footer(sl, 1)

    # 2 ask
    sl = new_slide(prs)
    title_block(
        sl,
        "The ask",
        "Benchmark the converter — do not rewrite the poster",
        "Core message: HinGE metrics belong in a small inset, not as a new thesis.",
    )
    box(sl, Inches(0.5), Inches(1.8), Inches(6.0), Inches(4.7), CARD, LINE)
    tf = tb(sl, Inches(0.75), Inches(2.0), Inches(5.5), Inches(0.4))
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, "What the guide asked", 16, GOLD, True)
    tf = tb(sl, Inches(0.75), Inches(2.5), Inches(5.5), Inches(3.7))
    tf.word_wrap = True
    bullets(
        tf,
        [
            "We currently “just called Qwen3” for Hinglish conversion",
            "Must report code-mixing metrics: CMI, WER, BLEU, TER, NIST",
            "Consider WAC / PAC rule algorithms from HinGE (ACL 2021)",
            "Fundamental poster remains SurakshaNet / MobiSys",
        ],
    )
    box(sl, Inches(6.75), Inches(1.8), Inches(6.05), Inches(4.7), CARD, LINE)
    tf = tb(sl, Inches(7.0), Inches(2.0), Inches(5.55), Inches(0.4))
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, "What we will not do", 16, RED, True)
    tf = tb(sl, Inches(7.0), Inches(2.5), Inches(5.55), Inches(3.7))
    tf.word_wrap = True
    bullets(
        tf,
        [
            "Replace the poster with an NLG paper",
            "Dump WER on MACD (no human Hinglish refs exist)",
            "Switch the production converter to WAC/PAC",
            "Confuse mixed val ~86% with Hinglish test 84.4%",
        ],
    )
    notes(
        sl,
        "The guide is right that an unmeasured LLM rewrite is not a method. "
        "The mistake would be turning a 2-page systems poster into HinGE. "
        "We keep SurakshaNet as the spine and add a generation-quality inset.",
    )
    footer(sl, 2)

    # 3 two tracks
    sl = new_slide(prs)
    title_block(
        sl,
        "Framework",
        "Two tracks. Do not mix the metrics.",
        "Core message: WER scores Hinglish text. Accuracy scores the detector.",
    )
    box(sl, Inches(0.5), Inches(1.85), Inches(6.0), Inches(4.65), CARD, LINE)
    tf = tb(sl, Inches(0.75), Inches(2.05), Inches(5.5), Inches(0.35))
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, "Track A  ·  Generation quality", 16, ACCENT, True)
    tf = tb(sl, Inches(0.75), Inches(2.5), Inches(5.5), Inches(3.6))
    tf.word_wrap = True
    bullets(
        tf,
        [
            "Question: is Qwen real Hinglish, not romanized Hindi?",
            "Metrics: CMI, BLEU, WER, TER, NIST, chrF",
            "Baselines: HinGE humans, WAC, PAC",
            "Status: DONE on 1,964 HinGE pairs + 31,663 MACD lines",
        ],
        15,
    )
    box(sl, Inches(6.75), Inches(1.85), Inches(6.05), Inches(4.65), CARD, LINE)
    tf = tb(sl, Inches(7.0), Inches(2.05), Inches(5.55), Inches(0.35))
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, "Track B  ·  Detection quality", 16, GOLD, True)
    tf = tb(sl, Inches(7.0), Inches(2.5), Inches(5.55), Inches(3.6))
    tf.word_wrap = True
    bullets(
        tf,
        [
            "Question: does MiniLM still catch abuse on-device?",
            "Metric that matters: frozen Hinglish TEST, not mixed val",
            "Shipped INT8: Hindi 85.1 / Hinglish 84.4 / EN 96.7",
            "Status: layer1 lost 0.6 pp; synth ablation still training",
        ],
        15,
    )
    notes(
        sl,
        "If someone asks ‘what is our WER’, answer: WER of Qwen Hinglish against HinGE human references. "
        "If they ask ‘does it work’, answer: 84.4% INT8 on frozen Hinglish test at 28 ms.",
    )
    footer(sl, 3)

    # 4 pipeline
    sl = new_slide(prs)
    title_block(
        sl,
        "Method",
        "The converter is gated. Not a raw LLM dump.",
        "Core message: Qwen3-32B + lexicon + ONNX abuse-delta is the system.",
    )
    steps = [
        ("1", "MACD Hindi", "Human-labeled\nabuse comments"),
        ("2", "Qwen3-32B", "Roman Hinglish\nWhatsApp style"),
        ("3", "Gates", "Lexicon + score Δ\nrefusals quarantined"),
        ("4", "MiniLM INT8", "ONNX in Chrome\n28 ms, 89 MB"),
    ]
    for i, (n, h, b) in enumerate(steps):
        x = Inches(0.5) + i * Inches(3.2)
        box(sl, x, Inches(2.05), Inches(2.95), Inches(2.55), CARD, LINE)
        tf = tb(sl, x + Inches(0.2), Inches(2.2), Inches(2.5), Inches(0.4))
        p = tf.paragraphs[0]
        r = p.add_run()
        set_run(r, n, 22, ACCENT, True)
        tf = tb(sl, x + Inches(0.2), Inches(2.65), Inches(2.55), Inches(0.45))
        p = tf.paragraphs[0]
        r = p.add_run()
        set_run(r, h, 18, WHITE, True)
        tf = tb(sl, x + Inches(0.2), Inches(3.15), Inches(2.55), Inches(1.1))
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        set_run(r, b, 13, MUTED)
    box(sl, Inches(0.5), Inches(4.85), Inches(12.3), Inches(1.7), CARD, LINE)
    tf = tb(sl, Inches(0.75), Inches(5.05), Inches(11.8), Inches(1.35))
    tf.word_wrap = True
    bullets(
        tf,
        [
            "Conversion yield ~94%. Failures: abuse_score_drop, refusal, collapsed length — not force-included.",
            "WAC/PAC cannot run on MACD: they need parallel English. MACD is Hindi-only social text.",
            "Synth hard-aug is seeded from errors (977 accepted, 0.5 weight). Never in val/test.",
        ],
        15,
    )
    notes(
        sl,
        "Walk the four boxes left to right. Emphasize the gate: without it Qwen softens slurs or refuses. "
        "That is why this is a method, not ‘we prompted an LLM’.",
    )
    footer(sl, 4)

    # 5 tricky
    sl = new_slide(prs)
    title_block(
        sl,
        "What was hard",
        "The insult is the payload. The LLM tries to drop it.",
        "Core message: three failure modes we had to engineer around.",
    )
    cards = [
        ("Refusal / softening", "Qwen declines abuse or writes la.nd. Lexicon fold + retry nudge. Test abuse still p≥0.5 at 89.9%.", RED),
        ("Score spike", "Non-abusive Hindi becomes abusive Hinglish. Dual-class gate. 1,997 train rows dropped as spikes.", GOLD),
        ("No gold Hinglish", "MACD has Hindi source only. WER/BLEU need HinGE humans. We cannot fake references.", ACCENT),
    ]
    for i, (h, b, c) in enumerate(cards):
        y = Inches(1.85) + i * Inches(1.55)
        box(sl, Inches(0.5), y, Inches(12.3), Inches(1.42), CARD, LINE)
        rect(sl, Inches(0.5), y, Inches(0.12), Inches(1.42), c)
        tf = tb(sl, Inches(0.9), y + Inches(0.18), Inches(11.6), Inches(0.4))
        p = tf.paragraphs[0]
        r = p.add_run()
        set_run(r, h, 18, c, True)
        tf = tb(sl, Inches(0.9), y + Inches(0.6), Inches(11.6), Inches(0.65))
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        set_run(r, b, 15, WHITE)
    notes(
        sl,
        "This is the ‘tricky part’ answer if the guide asks what we actually contributed beyond calling Qwen. "
        "Abuse preservation is the SurakshaNet-specific generation metric HinGE never had.",
    )
    footer(sl, 5)

    # 6 CMI
    sl = new_slide(prs)
    title_block(
        sl,
        "MACD corpus  ·  31,663 Qwen lines",
        "It is mixed — but not HinGE-human mixed.",
        "Core message: 25% of rewrites are CMI=0. That is the guide’s real worry, now quantified.",
    )
    metric_card(sl, Inches(0.5), Inches(1.85), Inches(3.0), Inches(1.45), "13.7", "Mean CMI (Qwen MACD)", GOLD)
    metric_card(sl, Inches(3.7), Inches(1.85), Inches(3.0), Inches(1.45), "35.6", "Mean CMI (HinGE humans)", ACCENT)
    metric_card(sl, Inches(6.9), Inches(1.85), Inches(3.0), Inches(1.45), "25%", "Lines with CMI = 0", RED)
    metric_card(sl, Inches(10.1), Inches(1.85), Inches(2.7), Inches(1.45), "24%", "Lines with CMI ≥ 20", OK)
    add_table(
        sl,
        Inches(0.5),
        Inches(3.55),
        Inches(12.3),
        Inches(2.85),
        ["Split", "n", "Mean CMI", "CMI=0", "Abuse kept", "Non-abuse kept", "Yield"],
        [
            ["Train", "18,920", "13.66", "25.3%", "96.7%", "77.3%", "93.7%"],
            ["Val", "6,375", "13.88", "25.3%", "89.7%", "74.9%", "94.8%"],
            ["Test (frozen)", "6,368", "13.81", "25.6%", "89.9%", "76.1%", "94.7%"],
        ],
        col_w=[Inches(2.2), Inches(1.5), Inches(1.6), Inches(1.5), Inches(1.9), Inches(2.1), Inches(1.5)],
    )
    notes(
        sl,
        "CMI is Das & Gambäck: 0 = one language. HinGE annotators were instructed to mix. "
        "MACD sources are Hindi comments, so lower CMI is partly domain, not only model failure. "
        "Still: a quarter of outputs are effectively transliteration. Be honest about that.",
    )
    footer(sl, 6)

    # 7 HinGE table
    sl = new_slide(prs)
    title_block(
        sl,
        "HinGE bake-off  ·  n = 1,964  ·  Hindi-only input",
        "Qwen ties PAC on BLEU. WAC wins WER because it sees English.",
        "Core message: this is the table the guide asked for. Fair to our pipeline, not a WAC clone.",
    )
    add_table(
        sl,
        Inches(0.5),
        Inches(1.85),
        Inches(12.3),
        Inches(3.35),
        ["System", "CMI ↑", "BLEU ↑", "WER ↓", "TER ↓", "NIST", "chrF"],
        [
            ["Human Hinglish", "35.64", "—", "—", "—", "—", "—"],
            ["WAC  word-aligned swap", "28.88", "0.164", "0.651", "0.744", "6.16", "47.91"],
            ["PAC  phrase-aligned swap", "24.26", "0.179", "0.671", "0.731", "5.62", "46.50"],
            ["Qwen3-32B  (ours, Hindi-only)", "22.68", "0.177", "0.713", "0.761", "5.16", "43.53"],
        ],
        col_w=[Inches(3.6), Inches(1.4), Inches(1.45), Inches(1.4), Inches(1.4), Inches(1.35), Inches(1.7)],
    )
    box(sl, Inches(0.5), Inches(5.4), Inches(12.3), Inches(1.2), CARD, LINE)
    tf = tb(sl, Inches(0.75), Inches(5.55), Inches(11.8), Inches(0.95))
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(
        r,
        "WAC/PAC align English nouns/phrases into a Hindi matrix. Our production converter never has English — MACD is Hindi-only. Scoring Qwen with English parallel would inflate BLEU and would not match the deployed pipeline.",
        15,
        WHITE,
    )
    notes(
        sl,
        "Paper Table 4 was 100 samples; ours is the full 1,976 CSV minus 12 quarantined. "
        "BLEU 0.177 ≈ PAC 0.179. WER is worse because Qwen paraphrases more and mixes less. "
        "HinGE itself showed automatic metrics correlate weakly with human ratings (Pearson BLEU ~0.14–0.19). "
        "Do not oversell BLEU. Lead with CMI + abuse preservation + downstream accuracy.",
    )
    footer(sl, 7)

    # 8 why not WAC
    sl = new_slide(prs)
    title_block(
        sl,
        "Decision",
        "WAC and PAC are baselines. They are not the converter.",
        "Core message: using them in production would make the abuse data worse.",
    )
    add_table(
        sl,
        Inches(0.5),
        Inches(1.85),
        Inches(12.3),
        Inches(4.7),
        ["", "WAC / PAC", "Qwen3 + gates (ours)"],
        [
            ["Needs English parallel", "Yes", "No — matches MACD"],
            ["Keeps slurs / intensity", "No guarantee", "Lexicon + ONNX Δ gate"],
            ["Social-media slang", "News-like IIT-B sentences", "ShareChat / WhatsApp register"],
            ["Role on poster", "Cite as 2020 baseline", "Actual data generator"],
            ["If we switched", "Cannot run on MACD", "Already trained detector"],
        ],
        col_w=[Inches(3.3), Inches(4.5), Inches(4.5)],
    )
    notes(
        sl,
        "If the guide says ‘use these algorithms’, the answer is: we did — as columns in the HinGE table. "
        "We will not replace Qwen with noun-swap on a corpus that has no English side.",
    )
    footer(sl, 8)

    # 9 detector
    sl = new_slide(prs)
    title_block(
        sl,
        "Track B  ·  shipped INT8",
        "The detector holds after conversion.",
        "Core message: 84.4% Hinglish at 28 ms in-browser is the poster number.",
    )
    metric_card(sl, Inches(0.5), Inches(1.85), Inches(3.0), Inches(1.5), "84.4%", "Hinglish test INT8", ACCENT)
    metric_card(sl, Inches(3.7), Inches(1.85), Inches(3.0), Inches(1.5), "85.1%", "Hindi Devanagari INT8", OK)
    metric_card(sl, Inches(6.9), Inches(1.85), Inches(3.0), Inches(1.5), "96.7%", "Davidson English INT8", OK)
    metric_card(sl, Inches(10.1), Inches(1.85), Inches(2.7), Inches(1.5), "28 ms", "Median on-device latency", GOLD)
    add_table(
        sl,
        Inches(0.5),
        Inches(3.6),
        Inches(12.3),
        Inches(2.85),
        ["Slice", "Accuracy", "F1", "n", "Note"],
        [
            ["Hinglish (frozen test)", "84.36%", "84.33%", "6,368", "The number that matters"],
            ["Hindi MACD", "85.14%", "85.12%", "6,728", "Old poster 84.7% recovered"],
            ["Mixed Hindi+EN+Hinglish", "86.54%", "86.40%", "15,575", "Do not quote as Hinglish"],
            ["Quantization loss", "<0.11 pp", "—", "—", "471 MB → 113 MB (76%)"],
        ],
        col_w=[Inches(3.2), Inches(1.8), Inches(1.6), Inches(1.6), Inches(4.1)],
    )
    notes(
        sl,
        "MACD XLM-R paper baseline was 86.34% Hindi but is not browser-deployable. "
        "Our constraint was RAM inside Chrome. Quantization loss is negligible. "
        "Never let mixed val 86% substitute for Hinglish test.",
    )
    footer(sl, 9)

    # 10 ablation honesty
    sl = new_slide(prs)
    title_block(
        sl,
        "Negative result  ·  keep it",
        "Cleaning the train set did not beat 84.4%.",
        "Core message: volume and relabeling were not the bottleneck. Frozen-test label noise was.",
    )
    add_table(
        sl,
        Inches(0.5),
        Inches(1.85),
        Inches(12.3),
        Inches(2.6),
        ["Run", "Train Hinglish", "INT8 Hinglish test", "Verdict"],
        [
            ["Shipped v2", "18,920 original Qwen", "84.36%", "KEEP / poster"],
            ["Layer1 clean", "17,962 drop+relabel", "83.76%  (−0.60 pp)", "Do not ship"],
            ["Layer1 + synth", "18,244 + 977 w=0.5", "training now", "Only ship if > 84.36%"],
        ],
        col_w=[Inches(2.6), Inches(3.4), Inches(3.3), Inches(3.0)],
    )
    box(sl, Inches(0.5), Inches(4.7), Inches(12.3), Inches(1.9), CARD, LINE)
    tf = tb(sl, Inches(0.75), Inches(4.9), Inches(11.8), Inches(1.55))
    tf.word_wrap = True
    bullets(
        tf,
        [
            "Frozen val errors: 507 FP / 528 FN. Largest bucket source_label_wrong = 497.",
            "Test texts were never rewritten by synth or rescue. That freeze is why the number is honest.",
            "Shipped 84.4% bundle was snapshotted; layer1 was not allowed to overwrite the extension.",
        ],
        15,
    )
    notes(
        sl,
        "Guides respect negative results. Layer1 looked correct (drop spikes, relabel 287) and still lost "
        "because gold on val/test is noisy. Synth is error-targeted, capped, down-weighted — paper-correct. "
        "If it loses, we still have 84.4% and the generation table.",
    )
    footer(sl, 10)

    # 11 poster mapping
    sl = new_slide(prs)
    title_block(
        sl,
        "Poster surgery",
        "Two pages. One new box. One inset table.",
        "Core message: generation numbers support the detector; they do not replace it.",
    )
    box(sl, Inches(0.5), Inches(1.85), Inches(6.0), Inches(4.7), CARD, LINE)
    tf = tb(sl, Inches(0.75), Inches(2.05), Inches(5.5), Inches(0.4))
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, "Keep as the spine", 16, OK, True)
    tf = tb(sl, Inches(0.75), Inches(2.55), Inches(5.5), Inches(3.7))
    tf.word_wrap = True
    bullets(
        tf,
        [
            "Title / abstract: privacy-first edge detection",
            "Fig 1: add Hindi → Hinglish (Qwen3, gated)",
            "Table 1: 28 ms, 89 MB, 76% shrink, <0.11% quant loss",
            "Accuracy slices: Hindi / Hinglish / EN",
        ],
    )
    box(sl, Inches(6.75), Inches(1.85), Inches(6.05), Inches(4.7), CARD, LINE)
    tf = tb(sl, Inches(7.0), Inches(2.05), Inches(5.55), Inches(0.4))
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, "Add, small", 16, GOLD, True)
    tf = tb(sl, Inches(7.0), Inches(2.55), Inches(5.55), Inches(3.7))
    tf.word_wrap = True
    bullets(
        tf,
        [
            "Mean CMI 13.7 vs HinGE humans 35.6",
            "Abuse preserved 89.9% · yield 94%",
            "HinGE: Qwen BLEU 0.177 ≈ PAC; WAC WER 0.651",
            "One sentence: converter is measured, detector is 84.4%",
        ],
    )
    notes(
        sl,
        "Read the poster sentence slowly. That is what we will paste into the PDF. "
        "If space is tight, drop NIST/chrF and keep CMI + BLEU + WER + abuse-kept.",
    )
    footer(sl, 11)

    # 12 close
    sl = new_slide(prs)
    title_block(
        sl,
        "What we tell the guide",
        "SurakshaNet is the system. HinGE is the audit.",
        "Core message: we can defend every number on this slide.",
    )
    box(sl, Inches(0.5), Inches(1.85), Inches(12.3), Inches(2.35), CARD, LINE)
    tf = tb(sl, Inches(0.75), Inches(2.05), Inches(11.8), Inches(1.95))
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(
        r,
        "We synthesize Roman Hinglish from MACD Hindi with Qwen3-32B under an abuse-preservation gate "
        "(test: 89.9% of gold-abusive still abusive; yield 94%). Mean CMI is 13.7 versus 35.6 for HinGE humans. "
        "On HinGE, Qwen BLEU 0.177 matches PAC; WAC remains the WER baseline because it uses English parallel. "
        "The quantized MiniLM reaches 84.4% on held-out Hinglish at 28 ms in-browser.",
        16,
        WHITE,
    )
    metric_card(sl, Inches(0.5), Inches(4.45), Inches(4.0), Inches(1.55), "Done", "HinGE + MACD generation eval", OK)
    metric_card(sl, Inches(4.7), Inches(4.45), Inches(4.0), Inches(1.55), "Live", "layer1_synth vs 84.4% test", GOLD)
    metric_card(sl, Inches(8.9), Inches(4.45), Inches(3.9), Inches(1.55), "No", "WAC as production converter", RED)
    notes(
        sl,
        "Close by offering the guide the sentence above for the poster. "
        "If they want more mixing, next work is prompt/CMI-constrained decoding — not switching to WAC. "
        "If they want higher detector acc, wait for layer1_synth; do not quietly un-freeze test.",
    )
    footer(sl, 12)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(OUT)


if __name__ == "__main__":
    build()
