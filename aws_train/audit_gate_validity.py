#!/usr/bin/env python3
"""Audit the Delta-p quarantine gate used in codemix_convert_bedrock.py.

Question: does the ONNX scorer's abuse probability on ROMAN Hinglish carry
enough signal to justify gating on (p_mix >= p_hi - delta)?

The gate selected rows on p_mix, so accepted-only is a biased sample.
We therefore recombine accepted + quarantined from *_meta.jsonl, which logs
every attempted row with its final scores and outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "codemix_hinglish"


CORRUPT: dict[str, int] = {}


def load_meta(split: str) -> pd.DataFrame:
    rows = []
    bad = 0
    for line in (SRC / f"{split}_meta.jsonl").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    CORRUPT[split] = bad
    df = pd.DataFrame(rows)
    # resumable chunked runs append; keep the last outcome per row
    df = df.drop_duplicates(subset=["row_id"], keep="last").reset_index(drop=True)
    for c in ("hindi_abuse", "hinglish_abuse", "score_delta"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["label"].astype(int)
    df["is_abusive"] = (df["label"] == 0).astype(int)
    return df


def auc(y, s) -> float:
    m = ~(pd.isna(s))
    y, s = np.asarray(y)[m], np.asarray(s)[m]
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def acc_at_half(y, s) -> float:
    m = ~(pd.isna(s))
    y, s = np.asarray(y)[m], np.asarray(s)[m]
    pred = (s >= 0.5).astype(int)
    return float((pred == y).mean())


def main() -> None:
    print("=" * 74)
    print("1. IS THE GATE SCORER ABLE TO READ ROMAN HINGLISH AT ALL?")
    print("=" * 74)
    print("AUC of scorer P(abusive) against MACD gold, same rows, two scripts.")
    print("Scorer was TRAINED on MACD Hindi train -> train split is memorised.\n")
    print(f"{'split':<7}{'n':>7}{'AUC Devanagari':>17}{'AUC Hinglish':>15}"
          f"{'acc@0.5 Dev':>14}{'acc@0.5 Hin':>14}")
    frames = {}
    for split in ("train", "val", "test"):
        df = load_meta(split)
        frames[split] = df
        print(f"{split:<7}{len(df):>7}{auc(df.is_abusive, df.hindi_abuse):>17.4f}"
              f"{auc(df.is_abusive, df.hinglish_abuse):>15.4f}"
              f"{acc_at_half(df.is_abusive, df.hindi_abuse):>14.4f}"
              f"{acc_at_half(df.is_abusive, df.hinglish_abuse):>14.4f}")

    allm = pd.concat(frames.values(), ignore_index=True)
    held = pd.concat([frames["val"], frames["test"]], ignore_index=True)

    print("\n" + "=" * 74)
    print("2. WHAT DOES THE GATE ACTUALLY CORRELATE WITH?")
    print("=" * 74)
    ab = allm[allm.label == 0]
    print(f"class-0 (abusive) rows attempted: n={len(ab)}")
    print(f"  mean P(abuse) on Devanagari source : {ab.hindi_abuse.mean():.4f}")
    print(f"  mean P(abuse) on Hinglish rewrite  : {ab.hinglish_abuse.mean():.4f}")
    print(f"  mean score_delta (hi - mix)        : {ab.score_delta.mean():+.4f}")
    print("  -> a systematic drop of this size is expected from DOMAIN SHIFT")
    print("     (Devanagari-trained scorer reading Roman text), not from the")
    print("     rewrite actually removing abuse.")

    print("\nControl: the same drop on class-1 (NON-abusive) rows, where there")
    print("is no abuse to lose. If the drop is similar, the gate is measuring")
    print("script shift, not abuse preservation.")
    nb = allm[allm.label == 1]
    print(f"  class-1 n={len(nb)}  mean delta = {nb.score_delta.mean():+.4f}")
    print(f"  class-0 n={len(ab)}  mean delta = {ab.score_delta.mean():+.4f}")

    print("\n" + "=" * 74)
    print("3. WHERE DOES delta = 0.15 COME FROM? (acceptance vs threshold)")
    print("=" * 74)
    print("Gate for class 0 is: accept iff p_mix >= p_hi - delta.")
    print("A principled threshold would show a knee. Recomputed on all")
    print("class-0 attempted rows:\n")
    print(f"{'delta':>7}{'class-0 accept %':>19}{'marginal gain pp':>19}")
    prev = None
    for d in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00):
        rate = float((ab.hinglish_abuse >= ab.hindi_abuse - d).mean()) * 100
        gain = "" if prev is None else f"{rate - prev:+.2f}"
        print(f"{d:>7.2f}{rate:>19.2f}{gain:>19}")
        prev = rate

    print("\n" + "=" * 74)
    print("4. DOES QUARANTINE SELECT BAD REWRITES, OR JUST HARD-TO-SCORE ONES?")
    print("=" * 74)
    q = allm[~allm.ok.astype(bool)]
    a = allm[allm.ok.astype(bool)]
    print(f"accepted n={len(a)}  quarantined n={len(q)}")
    print("\nquarantine reasons:")
    for reason, n in q.fail_reason.value_counts().items():
        print(f"  {reason:<28}{n:>7}  ({100*n/len(q):.1f}%)")
    print("\nlabel composition:")
    for name, d in (("accepted", a), ("quarantined", q)):
        if len(d):
            print(f"  {name:<14} class0={int((d.label==0).sum()):>6} "
                  f"class1={int((d.label==1).sum()):>6} "
                  f"class0 share={100*(d.label==0).mean():.1f}%")

    print("\n" + "=" * 74)
    print("5. HOW MUCH OF THE GATE IS JUST THE WORD LIST?")
    print("=" * 74)
    print("Fraction of class-0 source lines that contain ANY lexicon stem,")
    print("i.e. the share where the word-level check can fire at all.")
    import sys
    sys.path.insert(0, str(ROOT / "aws_train"))
    from codemix_convert_bedrock import ABUSE_STEMS

    def has_stem(h: str) -> bool:
        return any(stem in str(h) for stem, _ in ABUSE_STEMS)

    ab0 = allm[allm.label == 0].copy()
    ab0["lex"] = ab0.hindi_text.map(has_stem)
    cov = float(ab0.lex.mean())
    print(f"  class-0 rows n={len(ab0)}, lexicon-coverable = {100*cov:.1f}%")
    print(f"  -> on {100*(1-cov):.1f}% of abusive rows the word check is vacuous")
    print("     (lexicon_preserved returns True because nothing was required),")
    print("     so those rows are gated by the Delta-p score ALONE.")

    print("\n" + "=" * 74)
    print("6. HELD-OUT-ONLY VIEW (val+test, scorer never trained on these)")
    print("=" * 74)
    print(f"  n={len(held)}")
    print(f"  AUC Devanagari = {auc(held.is_abusive, held.hindi_abuse):.4f}")
    print(f"  AUC Hinglish   = {auc(held.is_abusive, held.hinglish_abuse):.4f}")

    print("\n" + "=" * 74)
    print("7. LOG INTEGRITY")
    print("=" * 74)
    print(f"  unparseable *_meta.jsonl lines (concurrent append): {CORRUPT}")


if __name__ == "__main__":
    main()
