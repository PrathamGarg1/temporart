#!/usr/bin/env python3
"""Generation-quality eval for SurakshaNet Hinglish (HinGE protocol).

Two tracks:
  1) MACD Qwen rewrites: CMI, transliteration-vs-mix WER, abuse preservation
  2) HinGE bake-off: WAC / PAC vs human Hinglish (BLEU, WER, TER, NIST, CMI)

Qwen is NOT re-run on HinGE here. That bake-off needs the same converter on
HinGE Hindi; this script first measures the corpus you already trained on,
and reproduces HinGE automatic metrics for the published rule baselines.

Usage:
  .venv_codemix/bin/python aws_train/eval_codemix_generation.py
  .venv_codemix/bin/python aws_train/eval_codemix_generation.py --macd-only
  .venv_codemix/bin/python aws_train/eval_codemix_generation.py --hinge-only
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACD_DIR = ROOT / "codemix_hinglish"
OUT_DIR = ROOT / "codemix_eval"
HINGE_REPO = "LingoIITGN/HinGE"
HINGE_FILE = "HinGE.csv"

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u0900-\u097F]+|\d+|[^\s\w]", re.UNICODE)
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002700-\U000027BF"
    "\U0001FA00-\U0001FAFF"
    "]+",
    re.UNICODE,
)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)

# High-precision Roman Hindi closed class + common matrix words.
HINDI_CLOSED = {
    "main", "mein", "me", "mai", "mein", "hun", "hoon", "hai", "hain", "ho",
    "hoga", "hogi", "honge", "tha", "thi", "the", "tha", "ka", "ki", "ke",
    "ko", "se", "par", "pe", "to", "toh", "ye", "yeh", "wo", "woh", "vo",
    "tum", "tu", "aap", "hum", "ham", "nahi", "nahin", "na", "mat", "kya",
    "kyun", "kyu", "kyunki", "jo", "aur", "lekin", "magar", "phir", "fir",
    "bhi", "hi", "bas", "ab", "tab", "kab", "yahan", "wahan", "yaha", "waha",
    "bahut", "bohot", "zyada", "jyada", "kam", "accha", "acha", "achha",
    "theek", "thik", "yaar", "bhai", "bhaiya", "didi", "ji", "re", "ya",
    "wali", "wala", "wale", "rahi", "raha", "rahe", "gaya", "gayi", "gaye",
    "kar", "karo", "karna", "karta", "karti", "karte", "kiya", "kiye",
    "diya", "diye", "lo", "de", "do", "le", "liya", "liye", "sakta", "sakti",
    "sakte", "chahiye", "chahie", "apna", "apni", "apne", "unka", "unki",
    "unke", "iska", "iski", "iske", "uska", "uski", "uske", "meri", "mera",
    "mere", "teri", "tera", "tere", "hamara", "hamari", "hamare", "sab",
    "sabhi", "kuch", "kuchh", "koi", "kisi", "kisiko", "har", "ek", "do",
    "teen", "char", "paanch", "woh", "iss", "uss", "in", "un", "inhe",
    "unhe", "mujhe", "tujhe", "humein", "hume", "unhone", "usne", "maine",
    "tune", "humne", "kyunki", "isliye", "phir", "jab", "tab", "agar",
    "warna", "nahin", "bilkul", "sahi", "galat", "acha", "bura", "bure",
    "log", "logo", "logon", "baat", "baatein", "din", "raat", "baad",
    "pehle", "pahle", "abhi", "kabhi", "hamesha", "sirf", "bus", "toh",
    "karke", "hoke", "jake", "leke", "deke", "karke", "wala", "desh",
    "logon", "liye", "tarah", "tarah", "jaise", "waise", "aise", "kaise",
    "kitna", "kitni", "kitne", "itna", "itni", "itne", "utna", "sabse",
    "koi", "kuch", "naya", "nayi", "purana", "purani", "bada", "badi",
    "chota", "choti", "saath", "sath", "paas", "pas", "door", "andar",
    "bahar", "upar", "neeche", "niche", "samne", "peeche", "piche",
    "ghar", "kaam", "naam", "baat", "din", "saal", "mahina", "hafte",
    "ladka", "ladki", "aadmi", "aurat", "bache", "bacha", "maa", "ma",
    "papa", "beta", "beti", "bhai", "behen", "behan", "dost",
    "chutiya", "chootiya", "chutya", "chutiye", "madarchod", "mc",
    "bhosdike", "bhosdi", "bhosdiwala", "gand", "gaand", "lund", "land",
    "harami", "haramzada", "kutta", "kutte", "kutti", "randi", "rundi",
    "sale", "saale", "bewakoof", "bewakuf", "bhenchod", "bhosdri",
    "gaali", "gali", "saala", "kamina", "kamine", "gadha", "gadhe",
    "bhosad", "choot", "chut", "jhatu", "jhantu",
    "nahin", "kyunki", "isliye", "usliye", "kyoke", "kyunke",
    "rahega", "rahegi", "rahenge", "hoga", "hogi", "honge",
    "karunga", "karungi", "karega", "karegi", "karenge",
    "dega", "degi", "denge", "lega", "legi", "lenge",
    "bole", "bola", "boli", "bol", "bolna", "dekho", "dekha", "dekhi",
    "sun", "suno", "suna", "jao", "jaa", "jaana", "aao", "aana", "aya",
    "ayi", "aye", "gaya", "hui", "hua", "hue", "huye",
    "wala", "wali", "wale", "waliye", "ji", "sahab", "sahib",
    "kya", "kyu", "kab", "kahan", "kidhar", "kaun", "kis",
    "apni", "apna", "khud", "khudko", "mujhko", "tujhko",
    "inhe", "unhe", "inke", "unke", "inka", "unka",
    "yehi", "wahi", "tabhi", "abhi", "jabhi",
    "phir", "fir", "phirse", "dobara", "dubaara",
    "zyada", "jyada", "bohot", "bahut", "thoda", "thodi",
    "poora", "pura", "saara", "saari", "saare",
    "kuchh", "kuch", "kai", "bahut", "sab",
    "nahi", "nahee", "nai", "nehin",
    "haan", "han", "ha", "ji",
    "are", "arre", "oy", "oye", "abe", "abey", "o",
    "yaar", "bhai", "dost", "yaar",
    "paisa", "paise", "rupee", "rupaye",
    "sach", "jhoot", "jhooth", "sachai",
    "pyar", "pyaar", "mohabbat", "ishq",
    "gussa", "gussa", "sharm", "sharam",
    "bhagwan", "bhagwan", "ram", "krishna", "allah",
    "hindu", "muslim", "sikh", "isai",
    "desh", "bharat", "hindustan",
}

# English that should stay English even if short / informal.
EN_FORCE = {
    "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "can", "could", "should",
    "please", "wait", "love", "cute", "fan", "ok", "okay", "lol", "omg",
    "bro", "yes", "hello", "thanks", "thank", "sorry", "please", "really",
    "very", "much", "more", "only", "also", "just", "like", "know", "think",
    "want", "need", "see", "look", "come", "going", "got", "get", "make",
    "made", "take", "give", "good", "bad", "nice", "happy", "sad", "friend",
    "family", "people", "world", "india", "indian", "english", "video",
    "message", "phone", "number", "full", "time", "life", "call", "chat",
    "because", "and", "but", "this", "that", "with", "from", "your", "you",
    "my", "we", "they", "them", "their", "its", "of", "on", "at", "for",
    "not", "if", "or", "an", "as", "by", "it", "he", "she", "his", "her",
    "our", "who", "what", "when", "where", "why", "how", "all", "any",
    "some", "than", "then", "too", "so", "up", "out", "about", "into",
    "over", "after", "before", "again", "once", "here", "there", "now",
    "government", "police", "minister", "party", "election", "vote",
    "beautiful", "sexy", "damn", "fuck", "shit", "bitch", "stupid",
    "idiot", "fool", "hate", "kill", "die", "dead", "true", "false",
    "best", "worst", "first", "last", "next", "new", "old", "big",
    "small", "same", "different", "important", "possible", "failed",
    "load", "file", "server", "connect", "connecting", "please",
    "wait", "start", "stop", "play", "game", "solution", "undo",
    "principles", "propagated", "congratulated", "choosing", "motto",
    "budget", "allocation", "plan", "scheme", "crores", "development",
    "women", "child", "welfare", "ministry", "department", "legislative",
    "council", "presented", "passed", "year",
}

_EN_DICT: set[str] | None = None


def english_dict() -> set[str]:
    global _EN_DICT
    if _EN_DICT is not None:
        return _EN_DICT
    words: set[str] = set(EN_FORCE)
    path = Path("/usr/share/dict/words")
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            w = line.strip().lower()
            if w.isalpha() and len(w) >= 3:
                words.add(w)
    words -= HINDI_CLOSED
    _EN_DICT = words
    return words


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text or "") if t.strip()]


def is_independent(tok: str) -> bool:
    if not tok:
        return True
    if tok.isdigit() or re.fullmatch(r"<number>", tok, re.I):
        return True
    if EMOJI_RE.fullmatch(tok):
        return True
    if NON_ALNUM.fullmatch(tok):
        return True
    if tok.startswith(("@", "#", "http")):
        return True
    if tok.isupper() and 2 <= len(tok) <= 5 and tok.isalpha():
        return True
    return False


def lang_tag(tok: str, en: set[str]) -> str:
    if is_independent(tok):
        return "univ"
    if DEVANAGARI_RE.search(tok):
        return "hi"
    w = tok.lower()
    if w in HINDI_CLOSED:
        return "hi"
    if w in en or w in EN_FORCE:
        return "en"
    return "hi"


def cmi_and_tags(text: str, en: set[str]) -> dict:
    toks = tokenize(text)
    tags = [lang_tag(t, en) for t in toks]
    n = len(tags)
    u = sum(1 for t in tags if t == "univ")
    counts = Counter(t for t in tags if t != "univ")
    mixed = n - u
    if mixed <= 0:
        cmi = 0.0
    elif len(counts) <= 1:
        cmi = 0.0
    else:
        cmi = 100.0 * (1.0 - (max(counts.values()) / mixed))
    switches = 0
    last = None
    for t in tags:
        if t == "univ":
            continue
        if last is not None and t != last:
            switches += 1
        last = t
    en_n = counts.get("en", 0)
    return {
        "cmi": cmi,
        "n": n,
        "en_frac": (en_n / mixed) if mixed else 0.0,
        "hi_frac": (counts.get("hi", 0) / mixed) if mixed else 0.0,
        "switches": switches,
        "monolingual": len(counts) <= 1,
    }


def normalize(text: str) -> str:
    s = (text or "").lower()
    s = EMOJI_RE.sub(" ", s)
    s = NON_ALNUM.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_list(text: str) -> list[str]:
    return normalize(text).split()


def levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = word_list(ref), word_list(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def min_wer(hyp: str, refs: list[str]) -> float:
    return min(wer(r, hyp) for r in refs) if refs else 1.0


def romanize_hi(text: str) -> str:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    return transliterate(text or "", sanscript.DEVANAGARI, sanscript.ITRANS).lower()


def mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def pct(xs: list[bool]) -> float:
    return 100.0 * mean([1.0 if x else 0.0 for x in xs]) if xs else 0.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def summarize_cmi(cmis: list[float], en_fracs: list[float], switches: list[float]) -> dict:
    return {
        "n": len(cmis),
        "mean_cmi": round(mean(cmis), 3),
        "median_cmi": round(statistics.median(cmis) if cmis else 0.0, 3),
        "p90_cmi": round(percentile(cmis, 0.9), 3),
        "pct_cmi_eq_0": round(pct([c <= 1e-9 for c in cmis]), 2),
        "pct_cmi_lt_10": round(pct([c < 10 for c in cmis]), 2),
        "pct_cmi_ge_20": round(pct([c >= 20 for c in cmis]), 2),
        "mean_en_frac": round(mean(en_fracs), 4),
        "mean_switch_points": round(mean(switches), 3),
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def eval_macd(en: set[str]) -> dict:
    out: dict = {"splits": {}, "quarantine": {}}
    all_cmi: list[float] = []
    all_en: list[float] = []
    all_sw: list[float] = []
    samples_low: list[dict] = []
    samples_high: list[dict] = []

    for split in ("train", "val", "test"):
        path = MACD_DIR / f"{split}.csv"
        df = pd.read_csv(path)
        cmis, enfs, sws, twers, overlaps = [], [], [], [], []
        abusive_kept, nonab_kept = [], []
        deltas, hindi_p, hing_p = [], [], []
        gold = []

        for _, row in df.iterrows():
            hyp = str(row.get("text") or "")
            src = str(row.get("hindi_text") or "")
            stats = cmi_and_tags(hyp, en)
            cmis.append(stats["cmi"])
            enfs.append(stats["en_frac"])
            sws.append(float(stats["switches"]))
            roman = romanize_hi(src)
            tw = wer(roman, hyp)
            twers.append(tw)
            rset, hset = set(word_list(roman)), set(word_list(hyp))
            overlaps.append((len(rset & hset) / len(rset | hset)) if (rset or hset) else 1.0)
            try:
                label = int(float(row["label"]))
            except (TypeError, ValueError):
                label = -1
            gold.append(label)
            ha = float(row.get("hindi_abuse") or 0)
            hi = float(row.get("hinglish_abuse") or 0)
            hindi_p.append(ha)
            hing_p.append(hi)
            deltas.append(float(row.get("score_delta") or (ha - hi)))
            if label == 0:
                abusive_kept.append(hi >= 0.5)
            elif label == 1:
                nonab_kept.append(hi < 0.5)

            rec = {
                "split": split,
                "cmi": round(stats["cmi"], 2),
                "en_frac": round(stats["en_frac"], 3),
                "translit_wer": round(tw, 3),
                "label": label,
                "hinglish": hyp[:180],
                "hindi": src[:120],
            }
            samples_low.append(rec)
            samples_high.append(rec)

        all_cmi.extend(cmis)
        all_en.extend(enfs)
        all_sw.extend(sws)

        n_ab = sum(1 for g in gold if g == 0)
        n_na = sum(1 for g in gold if g == 1)
        out["splits"][split] = {
            **summarize_cmi(cmis, enfs, sws),
            "mean_translit_wer": round(mean(twers), 4),
            "mean_token_jaccard_vs_roman_hi": round(mean(overlaps), 4),
            "n_abusive": n_ab,
            "n_non_abusive": n_na,
            "abuse_kept_pct": round(pct(abusive_kept), 2),
            "nonabuse_kept_pct": round(pct(nonab_kept), 2),
            "mean_score_delta": round(mean(deltas), 4),
            "pearson_hindi_hinglish_p": round(pearson(hindi_p, hing_p), 4),
        }

        qpath = MACD_DIR / f"{split}_quarantine.csv"
        if qpath.exists():
            qdf = pd.read_csv(qpath)
            reasons = Counter(str(x) for x in qdf.get("fail_reason", []))
            out["quarantine"][split] = {
                "n": int(len(qdf)),
                "by_reason": dict(reasons.most_common()),
                "yield_pct": round(
                    100.0 * len(df) / max(1, len(df) + len(qdf)), 2
                ),
            }

    samples_low.sort(key=lambda r: (r["cmi"], -r["en_frac"]))
    samples_high.sort(key=lambda r: (-r["cmi"], -r["en_frac"]))
    out["overall"] = {
        **summarize_cmi(all_cmi, all_en, all_sw),
        "interpretation": (
            "CMI=0 is romanized Hindi (or English), not code-mixing. "
            "CMI>=20 is a real mix. translit_wer vs ITRANS romanization: "
            "low WER ≈ transliteration, high WER ≈ rewrite/mix/drift."
        ),
    }
    out["examples_low_cmi"] = samples_low[:8]
    out["examples_high_cmi"] = samples_high[:8]
    return out


def parse_human_refs(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        return [str(val).strip()]
    except (SyntaxError, ValueError):
        return [s]


def nlg_scores(hyps: list[str], refs: list[list[str]]) -> dict:
    import sacrebleu
    from nltk.translate.nist_score import corpus_nist

    max_n = max((len(r) for r in refs), default=1)
    streams: list[list[str]] = []
    for k in range(max_n):
        stream = []
        for r in refs:
            stream.append(r[k] if k < len(r) else (r[0] if r else ""))
        streams.append(stream)
    bleu = sacrebleu.corpus_bleu(hyps, streams)
    ter = sacrebleu.corpus_ter(hyps, streams)
    chrf = sacrebleu.corpus_chrf(hyps, streams)

    hyp_toks = [word_list(h) for h in hyps]
    ref_toks = [[word_list(r) for r in rs] or [[]] for rs in refs]
    try:
        nist = float(corpus_nist(ref_toks, hyp_toks, n=5))
    except ZeroDivisionError:
        nist = 0.0

    wers = [min_wer(h, r if r else [""]) for h, r in zip(hyps, refs)]
    return {
        "bleu": round(bleu.score / 100.0, 4),
        "bleu_100": round(bleu.score, 2),
        "wer": round(mean(wers), 4),
        "ter": round(ter.score / 100.0, 4),
        "nist": round(nist, 4),
        "chrf": round(chrf.score, 2),
    }


def eval_hinge(en: set[str], hinge_csv: Path) -> dict:
    df = pd.read_csv(hinge_csv)
    cols = {c.lower().strip(): c for c in df.columns}

    def col(*names: str) -> str:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        raise KeyError(names)

    human_col = col("Human-generated Hinglish", "human")
    wac_col = col("WAC")
    pac_col = col("PAC")
    wac_r1 = cols.get("wac rating1")
    wac_r2 = cols.get("wac rating2")
    pac_r1 = cols.get("pac rating1")
    pac_r2 = cols.get("pac rating2")

    refs, wacs, pacs = [], [], []
    human_cmi, wac_cmi, pac_cmi = [], [], []
    wac_ratings, pac_ratings = [], []
    wac_sent_bleu, pac_sent_bleu = [], []
    wac_sent_wer, pac_sent_wer = [], []

    import sacrebleu

    for _, row in df.iterrows():
        humans = parse_human_refs(row[human_col])
        wac = str(row[wac_col] or "")
        pac = str(row[pac_col] or "")
        if not humans:
            continue
        refs.append(humans)
        wacs.append(wac)
        pacs.append(pac)
        human_cmi.append(mean([cmi_and_tags(h, en)["cmi"] for h in humans]))
        wac_cmi.append(cmi_and_tags(wac, en)["cmi"])
        pac_cmi.append(cmi_and_tags(pac, en)["cmi"])
        wac_sent_wer.append(min_wer(wac, humans))
        pac_sent_wer.append(min_wer(pac, humans))
        wac_sent_bleu.append(sacrebleu.sentence_bleu(wac, humans).score / 100.0)
        pac_sent_bleu.append(sacrebleu.sentence_bleu(pac, humans).score / 100.0)
        if wac_r1 and wac_r2:
            wac_ratings.append(
                (float(row[wac_r1]) + float(row[wac_r2])) / 2.0
            )
        if pac_r1 and pac_r2:
            pac_ratings.append(
                (float(row[pac_r1]) + float(row[pac_r2])) / 2.0
            )

    wac_nlg = nlg_scores(wacs, refs)
    pac_nlg = nlg_scores(pacs, refs)

    def corr_block(ratings, bleus, wers, system: str) -> dict:
        if len(ratings) != len(bleus):
            return {}
        return {
            "n": len(ratings),
            "mean_human_rating": round(mean(ratings), 3),
            "pearson_bleu": round(pearson(ratings, bleus), 3),
            "pearson_wer": round(pearson(ratings, wers), 3),
            "system": system,
        }

    paper_table4 = {
        "note": "HinGE Table 4 is on 100 sampled sentences; ours is the full CSV.",
        "WAC": {"bleu": 0.1229, "wer": 0.8240, "ter": 0.7830, "nist": 2.2045, "bertscore": 0.857},
        "PAC": {"bleu": 0.1202, "wer": 0.8228, "ter": 0.7981, "nist": 2.0497, "bertscore": 0.857},
    }

    return {
        "n": len(wacs),
        "source": "huggingface:LingoIITGN/HinGE",
        "human": summarize_cmi(human_cmi, [0.0] * len(human_cmi), [0.0] * len(human_cmi)),
        "WAC": {**wac_nlg, "mean_cmi": round(mean(wac_cmi), 3)},
        "PAC": {**pac_nlg, "mean_cmi": round(mean(pac_cmi), 3)},
        "human_mean_cmi": round(mean(human_cmi), 3),
        "correlation_vs_human_rating": {
            "WAC": corr_block(wac_ratings, wac_sent_bleu, wac_sent_wer, "WAC"),
            "PAC": corr_block(pac_ratings, pac_sent_bleu, pac_sent_wer, "PAC"),
        },
        "paper_table4_100sample": paper_table4,
    }


def download_hinge(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HINGE_REPO,
        filename=HINGE_FILE,
        repo_type="dataset",
    )
    data = Path(path).read_bytes()
    dest.write_bytes(data)
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--macd-only", action="store_true")
    p.add_argument("--hinge-only", action="store_true")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    en = english_dict()
    report: dict = {
        "decision": (
            "Poster stays SurakshaNet (edge abuse detection). "
            "This file is the missing generation-quality eval."
        )
    }

    if not args.hinge_only:
        print("Evaluating MACD Qwen Hinglish …", flush=True)
        report["macd_qwen"] = eval_macd(en)
        (args.out / "macd_generation_report.json").write_text(
            json.dumps(report["macd_qwen"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        o = report["macd_qwen"]["overall"]
        print(
            f"  mean CMI={o['mean_cmi']}  CMI=0:{o['pct_cmi_eq_0']}%  "
            f"CMI>=20:{o['pct_cmi_ge_20']}%  en_frac={o['mean_en_frac']}",
            flush=True,
        )

    if not args.macd_only:
        print("Downloading / scoring HinGE WAC vs PAC …", flush=True)
        hinge_csv = download_hinge(args.out / "HinGE.csv")
        report["hinge"] = eval_hinge(en, hinge_csv)
        (args.out / "hinge_wac_pac_report.json").write_text(
            json.dumps(report["hinge"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        h = report["hinge"]
        print(
            f"  n={h['n']}  human CMI={h['human_mean_cmi']}  "
            f"WAC BLEU={h['WAC']['bleu']} WER={h['WAC']['wer']}  "
            f"PAC BLEU={h['PAC']['bleu']} WER={h['PAC']['wer']}",
            flush=True,
        )

    (args.out / "generation_eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
