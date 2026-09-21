#!/usr/bin/env python3
"""Derive and benchmark an abuse-preservation gate against known ground truth.

The shipped gate (lexicon of 20 hand-written stems + raw Delta-p at
delta=0.15) was never validated: its own acceptance rate was reported as the
evidence that it worked. This script replaces that with a measurable setup.

Three parts:

1. MINE the abuse lexicon from MACD itself, instead of hand-writing it.
   Monroe, Colaresi & Quinn (2008) log-odds ratio with an informative Dirichlet
   prior, class 0 vs class 1 on Devanagari MACD train.

2. ALIGN each Devanagari abuse term to its Roman realisations using the 31,662
   (source, rewrite) pairs we already have. No transliteration heuristics: the
   parallel corpus tells us what the converter actually writes.

3. BENCHMARK gate variants on synthetic corruptions with known labels:
     NEGATIVE (gate must reject) - the slur is deleted, or swapped for a
       benign vocative. Abuse is genuinely gone.
     POSITIVE (gate must accept) - the untouched rewrite, and a variant with
       benign English words inserted so the slur survives but CMI rises.
   The English-injection arm is a controlled test of the confound: if a gate
   rejects those more often than the untouched rewrite, it is penalising
   code-mixing depth rather than abuse loss.

Delta is then chosen by Youden's J on this benchmark rather than left at an
argparse default.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from codemix_convert_bedrock import ABUSE_STEMS, fold_roman  # noqa: E402
from eval_codemix_generation import (  # noqa: E402
    EN_FORCE,
    HINDI_CLOSED,
    cmi_and_tags,
    english_dict,
)

SRC = ROOT / "codemix_hinglish"
HINDI = ROOT / "aws_train" / "macd_hindi_cache"
OUT_DIR = ROOT / "codemix_eval"
# The gate ran with assets/models/custom-macd-model as it existed BEFORE the
# Hinglish model was promoted over it. That pre-promotion bundle is -prev.
GATE_SCORER = ROOT / "assets" / "models" / "custom-macd-model-prev"

DEVA_TOKEN = re.compile(r"[\u0900-\u097F]+")
ROMAN_TOKEN = re.compile(r"[A-Za-z]+")
SEED = 42

BENIGN_SWAPS = ["bhai", "yaar", "dost", "boss"]
ENGLISH_INJECT = [
    ("seriously", 0), ("honestly", 0), ("actually", 1),
    ("basically", 1), ("literally", 1), ("obviously", 0),
]


# --------------------------------------------------------------------------- #
# 1. Mine the abuse lexicon from MACD (log-odds with informative prior)
# --------------------------------------------------------------------------- #
def mine_devanagari_lexicon(top_k: int = 250, min_count: int = 12) -> list[dict]:
    df = pd.read_csv(HINDI / "hindi_train.csv")[["label", "text"]].dropna()
    df["label"] = df["label"].astype(int)
    c0: Counter[str] = Counter()
    c1: Counter[str] = Counter()
    for lab, txt in zip(df.label, df.text.astype(str)):
        toks = DEVA_TOKEN.findall(txt)
        (c0 if lab == 0 else c1).update(set(toks))  # presence, not frequency

    vocab = {w for w, n in (c0 + c1).items() if n >= min_count}
    total = c0 + c1
    n_all = sum(total[w] for w in vocab)
    n0 = sum(c0[w] for w in vocab)
    n1 = sum(c1[w] for w in vocab)
    a0 = 500.0  # prior strength

    out = []
    for w in vocab:
        a_w = a0 * total[w] / n_all
        y0, y1 = c0[w] + a_w, c1[w] + a_w
        odds0 = y0 / (n0 + a0 - y0)
        odds1 = y1 / (n1 + a0 - y1)
        delta = math.log(odds0) - math.log(odds1)
        var = 1.0 / y0 + 1.0 / y1
        z = delta / math.sqrt(var)
        out.append({"deva": w, "z": z, "n0": c0[w], "n1": c1[w]})

    out.sort(key=lambda r: -r["z"])
    return out[:top_k]


# --------------------------------------------------------------------------- #
# 2. Align Devanagari terms to Roman realisations via the parallel pairs
# --------------------------------------------------------------------------- #
def align_roman(terms: list[dict], pairs: pd.DataFrame, top_m: int = 6) -> dict[str, list[str]]:
    stop = {w.lower() for w in HINDI_CLOSED} | {w.lower() for w in EN_FORCE}
    bg: Counter[str] = Counter()
    rewrite_toks: list[set[str]] = []
    for t in pairs["text"].astype(str):
        s = {w.lower() for w in ROMAN_TOKEN.findall(t) if len(w) >= 3}
        rewrite_toks.append(s)
        bg.update(s)
    n_docs = len(rewrite_toks)
    src = pairs["hindi_text"].astype(str).tolist()

    mapping: dict[str, list[str]] = {}
    for rec in terms:
        deva = rec["deva"]
        idx = [i for i, s in enumerate(src) if deva in s]
        if len(idx) < 5:
            continue
        sub: Counter[str] = Counter()
        for i in idx:
            sub.update(rewrite_toks[i])
        cands = []
        for w, k in sub.items():
            if w in stop or k < 3:
                continue
            # log-odds of appearing in this term's rewrites vs the corpus
            p_sub = k / len(idx)
            p_bg = bg[w] / n_docs
            if p_bg <= 0 or p_sub <= p_bg * 3:
                continue
            cands.append((math.log(p_sub / p_bg) * math.log1p(k), w))
        cands.sort(reverse=True)
        roms = [w for _, w in cands[:top_m]]
        if roms:
            mapping[deva] = roms
    return mapping


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
class Scorer:
    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        p = model_dir / "model_quantized.onnx"
        if not p.exists():
            p = model_dir / "onnx" / "model_quantized.onnx"
        self.sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}

    def probs(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = np.empty(len(texts), dtype=np.float64)
        for s in range(0, len(texts), batch):
            chunk = [t or " " for t in texts[s : s + batch]]
            enc = self.tok(chunk, return_tensors="np", truncation=True,
                           max_length=128, padding=True)
            feeds = {}
            for n in self.names:
                if n in enc:
                    feeds[n] = enc[n].astype(np.int64)
                elif n == "token_type_ids":
                    feeds[n] = np.zeros_like(enc["input_ids"], dtype=np.int64)
            logits = self.sess.run(None, feeds)[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            out[s : s + len(chunk)] = (e / e.sum(axis=1, keepdims=True))[:, 0]
        return out


# --------------------------------------------------------------------------- #
# 3. Build the corruption benchmark
# --------------------------------------------------------------------------- #
# Hard slurs only, each tied to the Devanagari stem it translates. Mild items
# (bewakoof, sale, haram, gadha ...) are excluded: deleting them does not
# unambiguously remove abuse, so they cannot serve as ground truth.
HARD_SLURS: list[tuple[str, list[str]]] = [
    ("चूतिय", ["chutiya", "chootiya", "chutya", "chootiye", "chutiye"]),
    ("चुत", ["chut", "choot"]),
    ("मादरचोद", ["madarchod", "maderchod", "motherchod", "mc"]),
    ("भोसड", ["bhosd", "bhosad", "bhosdi", "bhosdike", "bhosdiwala"]),
    ("भोसड़", ["bhosd", "bhosad", "bhosdi", "bhosdike"]),
    ("रंड", ["randi", "rundi"]),
    ("गांड", ["gand", "gaand"]),
    ("लंड", ["land", "laund", "lund", "lnd"]),
    ("कुत्तिय", ["kuttiya", "kutiya", "kutti"]),
    ("कुत्त", ["kutta", "kutte", "kutti", "kutiya", "kuttiya"]),
    ("हरामजाद", ["haramzada", "haramzaada", "haramjadi"]),
    ("भड़व", ["bhadwa", "bhadwe", "bhadva"]),
    ("चुड़ैल", ["chudail"]),
]
ALL_ROMANS = sorted(
    {r for _, roms in ABUSE_STEMS for r in roms}
    | {r for _, roms in HARD_SLURS for r in roms}
)
PUNCT = re.compile(r"[^\w]", re.UNICODE)


def _norm_tok(t: str) -> str:
    return PUNCT.sub("", fold_roman(t))


def _matches(tok: str, roman: str) -> bool:
    """Whole-token match, or a prefix match for terms long enough to be safe.

    Substring matching is what makes `gand` fire on `gandhi` and `mc` fire on
    anything, so short terms require an exact token.
    """
    if not tok:
        return False
    if tok == roman:
        return True
    return len(roman) >= 5 and tok.startswith(roman)


def _abuse_hits(tokens: list[str], romans: list[str]) -> list[int]:
    out = []
    for i, t in enumerate(tokens):
        n = _norm_tok(t)
        if any(_matches(n, r) for r in romans):
            out.append(i)
    return out


def build_benchmark(pairs: pd.DataFrame, mapping: dict[str, list[str]],
                    limit: int, rng: random.Random) -> pd.DataFrame:
    """Corruptions are only valid ground truth when the rewrite carries exactly
    one lexical abuse marker, that marker is a hard slur, and it is aligned to a
    Devanagari stem actually present in the source. Then deleting it removes the
    abuse rather than one of several copies."""
    rows = []
    skipped = Counter()
    for _, r in pairs.iterrows():
        if int(r["label"]) != 0:
            continue
        rw = str(r["text"])
        src_deva = str(r["hindi_text"])
        toks = rw.split()

        # every lexical abuse marker in the rewrite, mild ones included
        all_hits = _abuse_hits(toks, ALL_ROMANS)
        if len(all_hits) != 1:
            skipped["not_exactly_one_marker"] += 1
            continue
        i = all_hits[0]

        # that single marker must be a hard slur aligned to a source stem
        aligned = False
        for stem, roms in HARD_SLURS:
            if stem in src_deva and _abuse_hits([toks[i]], roms):
                aligned = True
                break
        if not aligned:
            skipped["not_hard_slur_aligned_to_source"] += 1
            continue

        base = {
            "row_id": r["row_id"], "hindi_text": src_deva,
            "hindi_abuse_cached": float(r["hindi_abuse"]),
            "slur": toks[i],
        }
        drop = " ".join(toks[:i] + toks[i + 1 :]).strip()
        swap = " ".join(toks[:i] + [rng.choice(BENIGN_SWAPS)] + toks[i + 1 :])
        w1, pos1 = rng.choice(ENGLISH_INJECT)
        w2, _ = rng.choice(ENGLISH_INJECT)
        inj = toks[:]
        inj.insert(min(pos1, len(inj)), w1)
        inj.append(w2)
        inject = " ".join(inj)
        if len(drop.split()) < 2:
            skipped["too_short_after_drop"] += 1
            continue

        # post-conditions: corruptions must leave no lexical abuse behind, and
        # the English injection must leave the slur untouched.
        if _abuse_hits(drop.split(), ALL_ROMANS):
            skipped["residual_abuse_after_drop"] += 1
            continue
        if _abuse_hits(swap.split(), ALL_ROMANS):
            skipped["residual_abuse_after_swap"] += 1
            continue
        if len(_abuse_hits(inject.split(), ALL_ROMANS)) != 1:
            skipped["injection_disturbed_slur"] += 1
            continue

        for variant, text, truth in (
            ("original", rw, "preserved"),
            ("inject_english", inject, "preserved"),
            ("drop_slur", drop, "removed"),
            ("swap_benign", swap, "removed"),
        ):
            rows.append({**base, "variant": variant, "text": text, "truth": truth})
        if len({x["row_id"] for x in rows}) >= limit:
            break
    print("  rows rejected while building ground truth:",
          dict(skipped.most_common()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Gate variants
# --------------------------------------------------------------------------- #
def lexicon_ok(hindi: str, hinglish: str, table: list[tuple[str, list[str]]]) -> bool:
    h_fold = fold_roman(hinglish)
    required = False
    for stem, romans in table:
        if stem in hindi:
            required = True
            if any(r in h_fold or fold_roman(r) in h_fold for r in romans):
                return True
    return not required


def ecdf(sample: np.ndarray):
    xs = np.sort(sample)
    def f(v: np.ndarray) -> np.ndarray:
        return np.searchsorted(xs, v, side="right") / len(xs)
    return f


def youden(scores: np.ndarray, is_removed: np.ndarray) -> tuple[float, float]:
    """Best threshold on a 'drop' score: reject when score > thr."""
    best, best_j = 0.0, -1.0
    for thr in np.unique(np.round(scores, 3)):
        rej = scores > thr
        tpr = rej[is_removed].mean() if is_removed.any() else 0.0
        fpr = rej[~is_removed].mean() if (~is_removed).any() else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best = j, float(thr)
    return best, best_j


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 1  mine abuse lexicon from MACD (log-odds, informative prior)")
    print("=" * 78)
    terms = mine_devanagari_lexicon()
    print(f"  top {len(terms)} class-0 Devanagari terms by z")
    print("  strongest 12:", ", ".join(t["deva"] for t in terms[:12]))

    pairs = pd.concat(
        [pd.read_csv(SRC / f"{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    )
    pairs["text"] = pairs["text"].fillna("").astype(str)
    pairs["hindi_text"] = pairs["hindi_text"].fillna("").astype(str)
    print(f"  parallel pairs available: {len(pairs)}")

    print("\n" + "=" * 78)
    print("STEP 2  align Devanagari terms to Roman forms from the pairs")
    print("=" * 78)
    mapping = align_roman(terms, pairs)
    mined_table = [(k, v) for k, v in mapping.items()]
    print(f"  aligned {len(mapping)} / {len(terms)} terms")
    for k in list(mapping)[:8]:
        print(f"    {k:<12} -> {', '.join(mapping[k][:5])}")
    hand_terms = {s for s, _ in ABUSE_STEMS}
    new_terms = [k for k in mapping if k not in hand_terms
                 and not any(h in k or k in h for h in hand_terms)]
    print(f"  terms the hand-written list does not cover: {len(new_terms)}")
    print("   ", ", ".join(new_terms[:14]))

    print("\n" + "=" * 78)
    print("STEP 3  synthetic corruption benchmark with known ground truth")
    print("=" * 78)
    bench = build_benchmark(pairs, mapping, limit=1200, rng=rng)
    n_items = bench.row_id.nunique()
    print(f"  source rows: {n_items}   graded variants: {len(bench)}")
    print("  variants:", dict(bench.variant.value_counts()))

    print("\n  scoring with the gate's own scorer "
          f"({GATE_SCORER.name}) …", flush=True)
    sc = Scorer(GATE_SCORER)
    bench["p_mix"] = sc.probs(bench.text.tolist())

    orig = bench[bench.variant == "original"]
    corr = np.corrcoef(orig.p_mix, [
        float(x) for x in pairs.set_index("row_id").loc[orig.row_id, "hinglish_abuse"]
    ])[0, 1]
    print(f"  sanity: re-scored 'original' vs cached hinglish_abuse r = {corr:.4f}")
    print("  (confirms this bundle is the scorer that ran during conversion)")

    # script-matched rank normalisation, from the full cached populations
    f_dev = ecdf(pairs["hindi_abuse"].astype(float).to_numpy())
    f_rom = ecdf(pairs["hinglish_abuse"].astype(float).to_numpy())

    p_hi = bench.hindi_abuse_cached.to_numpy()
    p_mix = bench.p_mix.to_numpy()
    bench["drop_raw"] = p_hi - p_mix
    bench["drop_rank"] = f_dev(p_hi) - f_rom(p_mix)
    removed = (bench.truth == "removed").to_numpy()

    en = english_dict()
    bench["cmi"] = [cmi_and_tags(t, en)["cmi"] for t in bench.text]

    print("\n" + "=" * 78)
    print("RESULT 1  gate variants vs known ground truth")
    print("=" * 78)
    print("  reject-rate on each arm. Ideal: high on removed, low on preserved.\n")

    def evaluate(name: str, reject: np.ndarray) -> dict:
        per = {}
        for v in ("original", "inject_english", "drop_slur", "swap_benign"):
            m = (bench.variant == v).to_numpy()
            per[v] = float(reject[m].mean())
        det = float(reject[removed].mean())
        fpr = float(reject[~removed].mean())
        print(f"  {name:<34} detect={det:6.3f}  false-reject={fpr:6.3f}  "
              f"J={det-fpr:+.3f}")
        print(f"       {'':<30} orig={per['original']:.3f} "
              f"inject_en={per['inject_english']:.3f} "
              f"drop={per['drop_slur']:.3f} swap={per['swap_benign']:.3f}")
        return {"detect_removed": round(det, 4), "false_reject": round(fpr, 4),
                "youden_j": round(det - fpr, 4),
                "by_variant": {k: round(v, 4) for k, v in per.items()}}

    results: dict[str, dict] = {}
    lex_hand = np.array([not lexicon_ok(h, t, ABUSE_STEMS)
                         for h, t in zip(bench.hindi_text, bench.text)])
    lex_mined = np.array([not lexicon_ok(h, t, mined_table)
                          for h, t in zip(bench.hindi_text, bench.text)])
    dp015 = bench.drop_raw.to_numpy() > 0.15

    results["shipped (hand lexicon OR raw dp>0.15)"] = evaluate(
        "shipped gate", lex_hand | dp015)
    results["hand lexicon only"] = evaluate("hand lexicon only", lex_hand)
    results["mined lexicon only"] = evaluate("mined lexicon only", lex_mined)
    results["raw dp > 0.15 only"] = evaluate("raw dp>0.15 only", dp015)

    thr_raw, j_raw = youden(bench.drop_raw.to_numpy(), removed)
    thr_rank, j_rank = youden(bench.drop_rank.to_numpy(), removed)
    print(f"\n  delta chosen by Youden's J:  raw dp -> {thr_raw:.3f} (J={j_raw:.3f})"
          f"   rank dp -> {thr_rank:.3f} (J={j_rank:.3f})")
    results[f"raw dp > {thr_raw:.2f} (calibrated)"] = evaluate(
        f"raw dp>{thr_raw:.2f} calibrated", bench.drop_raw.to_numpy() > thr_raw)
    results[f"rank dp > {thr_rank:.2f} (calibrated)"] = evaluate(
        f"rank dp>{thr_rank:.2f} calibrated", bench.drop_rank.to_numpy() > thr_rank)
    prop = lex_mined | (bench.drop_rank.to_numpy() > thr_rank)
    results["proposed (mined lexicon OR rank dp)"] = evaluate(
        "PROPOSED mined lex OR rank dp", prop)

    from sklearn.metrics import roc_auc_score
    auc_raw = roc_auc_score(removed, bench.drop_raw)
    auc_rank = roc_auc_score(removed, bench.drop_rank)
    print(f"\n  ranking quality (AUC for detecting real abuse removal):")
    print(f"    raw   p_hi - p_mix : {auc_raw:.4f}")
    print(f"    rank-matched       : {auc_rank:.4f}")

    print("\n" + "=" * 78)
    print("RESULT 2  controlled test of the code-mixing confound")
    print("=" * 78)
    o = (bench.variant == "original").to_numpy()
    ie = (bench.variant == "inject_english").to_numpy()
    print(f"  injecting 2 benign English words raises mean CMI "
          f"{bench.cmi[o].mean():.2f} -> {bench.cmi[ie].mean():.2f}, "
          f"slur untouched.")
    from scipy.stats import fisher_exact
    for name, rej in (("shipped gate", lex_hand | dp015),
                      ("raw dp>0.15", dp015),
                      ("rank dp calibrated", bench.drop_rank.to_numpy() > thr_rank)):
        a, b = int(rej[o].sum()), int((~rej[o]).sum())
        c, d = int(rej[ie].sum()), int((~rej[ie]).sum())
        odds, pv = fisher_exact([[c, d], [a, b]])
        print(f"    {name:<22} reject {rej[o].mean():.3f} -> {rej[ie].mean():.3f}"
              f"   odds={odds:.2f}  p={pv:.3e}")
    results["english_injection_test"] = {
        "cmi_original": round(float(bench.cmi[o].mean()), 3),
        "cmi_injected": round(float(bench.cmi[ie].mean()), 3),
        "shipped_reject_original": round(float((lex_hand | dp015)[o].mean()), 4),
        "shipped_reject_injected": round(float((lex_hand | dp015)[ie].mean()), 4),
    }

    payload = {
        "gate_scorer": GATE_SCORER.name,
        "sanity_corr_vs_cached": round(float(corr), 4),
        "n_source_rows": int(n_items),
        "n_graded_variants": int(len(bench)),
        "mined_terms": len(terms),
        "aligned_terms": len(mapping),
        "terms_missed_by_hand_list": len(new_terms),
        "auc_raw_dp": round(float(auc_raw), 4),
        "auc_rank_dp": round(float(auc_rank), 4),
        "delta_youden_raw": thr_raw,
        "delta_youden_rank": thr_rank,
        "gates": results,
        "mined_lexicon": {k: v for k, v in list(mapping.items())},
    }
    (OUT_DIR / "gate_v2_benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    bench.drop(columns=["hindi_text"]).to_csv(
        OUT_DIR / "gate_v2_benchmark_rows.csv", index=False)
    print(f"\nwrote {OUT_DIR/'gate_v2_benchmark.json'}")


if __name__ == "__main__":
    main()
