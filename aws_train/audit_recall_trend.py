#!/usr/bin/env python3
"""Is the abuse-recall collapse at high CMI statistically real?

Overall accuracy on the frozen test set is flat across CMI because the class
balance shifts (deep mixes are mostly non-abusive). Recall on class 0 is the
metric that actually matters for an abuse detector, so test its trend.

Caches per-row predictions to codemix_eval/preds_frozen_test.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from audit_cmi_and_mcnemar import MODELS, predict, wilson  # noqa: E402
from eval_codemix_generation import cmi_and_tags, english_dict  # noqa: E402

TEST = ROOT / "codemix_hinglish/test.csv"
CACHE = ROOT / "codemix_eval" / "preds_frozen_test.csv"
OUT = ROOT / "codemix_eval" / "audit_recall_trend.json"


def main() -> None:
    df = pd.read_csv(TEST)
    df["text"] = df["text"].fillna("").astype(str)
    en = english_dict()

    if CACHE.exists():
        cache = pd.read_csv(CACHE)
        print(f"using cached predictions {CACHE.name} n={len(cache)}")
    else:
        cache = pd.DataFrame({
            "row_id": df["row_id"], "label": df["label"].astype(int),
            "text": df["text"],
            "cmi": [cmi_and_tags(t, en)["cmi"] for t in df["text"]],
        })
        for name, path in MODELS.items():
            print(f"  scoring {name} …", flush=True)
            cache[f"pred_{name}"] = predict(path, df["text"].tolist())
        cache.to_csv(CACHE, index=False)
        print(f"wrote {CACHE}")

    y = cache["label"].to_numpy()
    cmi = cache["cmi"].to_numpy()
    p = cache["pred_shipped"].to_numpy()

    ab = y == 0
    nb = y == 1
    print("\n" + "=" * 78)
    print("PER-CLASS PERFORMANCE BY MIXING DEPTH (shipped INT8, frozen test)")
    print("=" * 78)
    print(f"{'bucket':<22}{'n_abusive':>10}{'recall_abuse':>14}{'95% CI':>18}"
          f"{'n_clean':>9}{'specificity':>13}")
    buckets = [
        ("CMI = 0", cmi <= 1e-9),
        ("0 < CMI < 10", (cmi > 1e-9) & (cmi < 10)),
        ("10 <= CMI < 20", (cmi >= 10) & (cmi < 20)),
        ("20 <= CMI < 30", (cmi >= 20) & (cmi < 30)),
        ("CMI >= 30", cmi >= 30),
    ]
    rows = []
    for name, m in buckets:
        ma, mn = m & ab, m & nb
        na, nn = int(ma.sum()), int(mn.sum())
        hit = int((p[ma] == 0).sum())
        rec = hit / na if na else float("nan")
        lo, hi = wilson(hit, na)
        spec = float((p[mn] == 1).mean()) if nn else float("nan")
        rows.append({"bucket": name, "n_abusive": na, "recall_abusive": round(rec, 4),
                     "ci95": [round(lo, 4), round(hi, 4)], "n_clean": nn,
                     "specificity": round(spec, 4)})
        print(f"{name:<22}{na:>10}{rec:>14.4f}{f'[{lo:.3f},{hi:.3f}]':>18}"
              f"{nn:>9}{spec:>13.4f}")

    # Cochran-Armitage trend test on recall across ordered buckets
    from scipy.stats import chi2, fisher_exact
    scores = np.array([0, 1, 2, 3, 4], dtype=float)
    n_i = np.array([r["n_abusive"] for r in rows], dtype=float)
    x_i = np.array([round(r["recall_abusive"] * r["n_abusive"]) for r in rows], dtype=float)
    N, X = n_i.sum(), x_i.sum()
    pbar = X / N
    num = (scores * (x_i - n_i * pbar)).sum()
    den = pbar * (1 - pbar) * ((n_i * scores**2).sum() - (n_i * scores).sum() ** 2 / N)
    z2 = num**2 / den
    p_trend = float(1 - chi2.cdf(z2, 1))
    print(f"\nCochran-Armitage trend test on abuse recall: chi2={z2:.3f}  p={p_trend:.3e}")

    shallow = ab & (cmi < 10)
    deep = ab & (cmi >= 20)
    tab = [[int((p[shallow] == 0).sum()), int((p[shallow] == 1).sum())],
           [int((p[deep] == 0).sum()), int((p[deep] == 1).sum())]]
    odds, pf = fisher_exact(tab)
    print(f"shallow abusive (CMI<10)  n={int(shallow.sum())} recall={(p[shallow]==0).mean():.4f}")
    print(f"deep abusive    (CMI>=20) n={int(deep.sum())} recall={(p[deep]==0).mean():.4f}")
    print(f"Fisher exact: odds={odds:.3f}  p={pf:.3e}")

    print("\n" + "=" * 78)
    print("WHY OVERALL ACCURACY HIDES THIS")
    print("=" * 78)
    for name, m in buckets:
        n = int(m.sum())
        share_ab = 100 * float((y[m] == 0).mean()) if n else 0
        acc = float((p[m] == y[m]).mean()) if n else 0
        print(f"  {name:<18} n={n:>5}  abusive share={share_ab:>5.1f}%  accuracy={acc:.4f}")
    print("\n  Deep mixes are mostly NON-abusive, so a model that simply gets")
    print("  cleaner text right scores high accuracy there while missing a")
    print("  third of the actual abuse. One headline number cannot show this.")

    OUT.write_text(json.dumps({
        "per_bucket": rows,
        "trend_test": {"chi2": float(z2), "p": p_trend},
        "shallow_vs_deep_recall": {
            "recall_cmi_lt_10": round(float((p[shallow] == 0).mean()), 4),
            "n_cmi_lt_10": int(shallow.sum()),
            "recall_cmi_ge_20": round(float((p[deep] == 0).mean()), 4),
            "n_cmi_ge_20": int(deep.sum()),
            "fisher_odds": round(float(odds), 4), "fisher_p": float(pf),
        },
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
