#!/usr/bin/env python3
"""Does the Delta-p quarantine gate preferentially discard DEEP code-mixing?

If yes, the gate is not preserving labels; it is biasing the training corpus
toward shallow mixes, which is a mechanism that would explain the abuse-recall
collapse at high CMI on the frozen test set.

Compares, for class-0 (abusive) rows only:
  accepted  -> codemix_hinglish/{split}.csv       column `text`
  rejected  -> codemix_hinglish/{split}_quarantine.csv column `last_output`
(the rejected rewrite still exists, so both populations are comparable)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from eval_codemix_generation import cmi_and_tags, english_dict  # noqa: E402

SRC = ROOT / "codemix_hinglish"
OUT = ROOT / "codemix_eval" / "audit_gate_vs_cmi.json"


def main() -> None:
    en = english_dict()
    acc_parts, rej_parts = [], []
    for split in ("train", "val", "test"):
        a = pd.read_csv(SRC / f"{split}.csv")
        a["hinglish"] = a["text"].fillna("").astype(str)
        a["kept"] = 1
        acc_parts.append(a[["label", "hinglish", "kept", "hindi_abuse", "hinglish_abuse"]])

        q = pd.read_csv(SRC / f"{split}_quarantine.csv")
        q["hinglish"] = q["last_output"].fillna("").astype(str)
        q["kept"] = 0
        rej_parts.append(
            q[["label", "hinglish", "kept", "hindi_abuse", "hinglish_abuse", "fail_reason"]]
        )

    acc = pd.concat(acc_parts, ignore_index=True)
    rej = pd.concat(rej_parts, ignore_index=True)
    both = pd.concat([acc, rej], ignore_index=True)
    both["label"] = both["label"].astype(int)
    both = both[both["hinglish"].str.strip() != ""]
    both["cmi"] = [cmi_and_tags(t, en)["cmi"] for t in both["hinglish"]]

    ab = both[both.label == 0]
    kept = ab[ab.kept == 1]
    drop = ab[ab.kept == 0]

    print("=" * 78)
    print("CMI OF ABUSIVE REWRITES: KEPT BY THE GATE vs THROWN IN QUARANTINE")
    print("=" * 78)
    print(f"{'population':<28}{'n':>7}{'mean CMI':>11}{'median':>9}"
          f"{'%CMI=0':>9}{'%CMI>=20':>10}{'%CMI>=30':>10}")
    rows = {}
    for name, d in (("kept (trained on)", kept), ("quarantined (discarded)", drop)):
        c = d.cmi.to_numpy()
        rows[name] = {
            "n": int(len(d)),
            "mean_cmi": round(float(c.mean()), 3),
            "median_cmi": round(float(np.median(c)), 3),
            "pct_cmi_0": round(float(100 * (c <= 1e-9).mean()), 2),
            "pct_cmi_ge_20": round(float(100 * (c >= 20).mean()), 2),
            "pct_cmi_ge_30": round(float(100 * (c >= 30).mean()), 2),
        }
        r = rows[name]
        print(f"{name:<28}{r['n']:>7}{r['mean_cmi']:>11.2f}{r['median_cmi']:>9.2f}"
              f"{r['pct_cmi_0']:>9.2f}{r['pct_cmi_ge_20']:>10.2f}{r['pct_cmi_ge_30']:>10.2f}")

    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(drop.cmi, kept.cmi, alternative="greater")
    print(f"\nMann-Whitney U (quarantined CMI > kept CMI): U={u:.0f}  p={p:.3e}")
    print("alternative tested: the DISCARDED rewrites are more deeply mixed.")

    print("\n" + "=" * 78)
    print("SAME QUESTION, ONLY THE Delta-p REASON (excl. lexicon/length/refusal)")
    print("=" * 78)
    dp = drop[drop.fail_reason == "abuse_score_drop"]
    print(f"  abuse_score_drop n={len(dp)}  mean CMI={dp.cmi.mean():.2f}  "
          f"%CMI>=20={100*(dp.cmi>=20).mean():.2f}")
    u2, p2 = mannwhitneyu(dp.cmi, kept.cmi, alternative="greater")
    print(f"  Mann-Whitney vs kept: U={u2:.0f}  p={p2:.3e}")

    print("\n" + "=" * 78)
    print("IS THE GATE SCORE ITSELF A FUNCTION OF MIXING DEPTH?")
    print("=" * 78)
    print("If P(abuse) on the rewrite falls as CMI rises, then 'abuse_score_drop'")
    print("is partly just 'more English in the sentence', not lost abuse.")
    print(f"\n{'CMI bucket':<22}{'n':>7}{'mean P(abuse) on mix':>23}{'mean P on Devanagari':>23}")
    buckets = [
        ("CMI = 0", ab.cmi <= 1e-9),
        ("0 < CMI < 10", (ab.cmi > 1e-9) & (ab.cmi < 10)),
        ("10 <= CMI < 20", (ab.cmi >= 10) & (ab.cmi < 20)),
        ("20 <= CMI < 30", (ab.cmi >= 20) & (ab.cmi < 30)),
        ("CMI >= 30", ab.cmi >= 30),
    ]
    curve = []
    for name, m in buckets:
        d = ab[m]
        if not len(d):
            continue
        curve.append({
            "bucket": name, "n": int(len(d)),
            "mean_p_mix": round(float(d.hinglish_abuse.mean()), 4),
            "mean_p_dev": round(float(d.hindi_abuse.mean()), 4),
        })
        print(f"{name:<22}{len(d):>7}{d.hinglish_abuse.mean():>23.4f}"
              f"{d.hindi_abuse.mean():>23.4f}")

    from scipy.stats import spearmanr
    rho, ps = spearmanr(ab.cmi, ab.hinglish_abuse)
    rho2, ps2 = spearmanr(ab.cmi, ab.hindi_abuse)
    print(f"\nSpearman(CMI, P(abuse) on Hinglish)   rho={rho:+.4f}  p={ps:.3e}")
    print(f"Spearman(CMI, P(abuse) on Devanagari) rho={rho2:+.4f}  p={ps2:.3e}")
    print("The Devanagari source cannot depend on the rewrite's CMI, so that")
    print("second row is the control for source-difficulty confounding.")

    OUT.write_text(json.dumps({
        "cmi_kept_vs_quarantined": rows,
        "mannwhitney_p": float(p),
        "mannwhitney_p_delta_p_only": float(p2),
        "p_abuse_by_cmi_bucket": curve,
        "spearman_cmi_vs_p_mix": {"rho": float(rho), "p": float(ps)},
        "spearman_cmi_vs_p_dev": {"rho": float(rho2), "p": float(ps2)},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
