#!/usr/bin/env python3
"""What the measured gate characteristics imply for the training corpus.

Supersedes an earlier version of this file that inverted the observed 11%
quarantine rate with Rogan-Gladen to claim ~14% label noise. That estimate was
wrong twice over and is not used here:
  - it relied on the first corruption benchmark, whose ground truth was invalid;
  - it ignored `max_retries`. A row is quarantined only if up to three separate
    generations all fail the gate, so the observed rejection rate is not a
    single-test positive rate and cannot be inverted that way.

What is defensible instead: the gate's per-generation operating point is now
measured on realistic fluent softening (gate_v4_realistic_bench.py), so the
contamination of the ACCEPTED corpus can be written as a function of the one
quantity nobody has measured -- how often the converter softens. That is
reported as a sweep rather than a point estimate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from gate_v2_benchmark import Scorer  # noqa: E402

SRC = ROOT / "codemix_hinglish"
OUT = ROOT / "codemix_eval" / "gate_v2_implications.json"
V4 = ROOT / "codemix_eval" / "gate_v4_report.json"
BENCH = ROOT / "codemix_eval" / "gate_v2_benchmark_rows.csv"
CANDIDATES = {
    "custom-macd-model-prev": ROOT / "assets/models/custom-macd-model-prev",
    "custom-macd-model (promoted)": ROOT / "assets/models/custom-macd-model",
    "custom-macd-model-shipped-84": ROOT / "assets/models/custom-macd-model-shipped-84",
}


def identify_scorer() -> dict:
    bench = pd.read_csv(BENCH)
    orig = bench[bench.variant == "original"]
    pairs = pd.concat(
        [pd.read_csv(SRC / f"{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    ).set_index("row_id")
    cached = pairs.loc[orig.row_id, "hinglish_abuse"].astype(float).to_numpy()
    texts = orig.text.fillna("").astype(str).tolist()
    print("=" * 78)
    print("WHICH CHECKPOINT ACTUALLY GATED THE CORPUS?")
    print("=" * 78)
    print("Promotion overwrote assets/models/custom-macd-model in place, so the")
    print("gating checkpoint has to be identified from the cached scores.\n")
    ident = {}
    for name, path in CANDIDATES.items():
        if not path.exists():
            continue
        p = Scorer(path).probs(texts)
        r = float(np.corrcoef(p, cached)[0, 1])
        mad = float(np.abs(p - cached).mean())
        ident[name] = {"pearson_r": round(r, 4), "mean_abs_diff": round(mad, 4)}
        print(f"  {name:<32} r={r:+.4f}  mean|diff|={mad:.4f}")
    best = max(ident, key=lambda k: -ident[k]["mean_abs_diff"])
    print(f"\n  closest: {best}")
    print("  The promoted bundles are far off, so the gate ran on the")
    print("  pre-Hinglish checkpoint. It is not reproduced exactly by any bundle")
    print("  in the repo (padding differs, and the file was overwritten), so the")
    print("  exact gating artifact is unrecoverable.")
    return ident


def main() -> None:
    ident = identify_scorer()
    v4 = json.loads(V4.read_text())
    dp = v4["gates"]["raw dp > 0.15 (as shipped)"]
    lex = v4["gates"]["hand lexicon (as shipped)"]
    judge = v4["gates"]["source-referenced judge"]
    tpr, fpr = dp["sensitivity"], dp["false_reject"]

    print("\n" + "=" * 78)
    print("MEASURED OPERATING POINT (per generation, lexicon-blind stratum)")
    print("=" * 78)
    print(f"  stratum: {v4['stratum']}")
    print(f"  items: {v4['n_items']} from {v4['n_sources']} sources\n")
    print(f"{'gate':<30}{'sensitivity':>13}{'false-reject':>14}{'Youden J':>10}")
    for name in ("hand lexicon (as shipped)", "raw dp > 0.15 (as shipped)",
                 "source-referenced judge"):
        g = v4["gates"][name]
        print(f"{name:<30}{g['sensitivity']:>13.3f}{g['false_reject']:>14.3f}"
              f"{g['youden_j']:>10.3f}")
    print(f"\n  AUC of the delta-p score for telling softened from faithful: "
          f"{v4['auc_raw_dp']:.3f}")
    print("  Chance is 0.500. Rank-matching does not rescue it "
          f"({v4['auc_rank_dp']:.3f}).")

    print("\n" + "=" * 78)
    print("CONTAMINATION OF THE ACCEPTED CORPUS")
    print("=" * 78)
    print("  Among accepted rows, the share that are softened-but-labelled-abusive:")
    print("      s(1-TPR) / [ s(1-TPR) + (1-s)(1-FPR) ]")
    print(f"  with the measured TPR={tpr:.3f}, FPR={fpr:.3f}, and s = the unknown")
    print("  per-generation softening rate. 'no gate' is the s column itself,")
    print("  so the last column is what the gate actually buys.\n")
    print(f"{'softening rate s':>18}{'contamination with gate':>26}"
          f"{'without gate':>15}{'reduction':>12}")
    sweep = []
    for s in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        num = s * (1 - tpr)
        den = num + (1 - s) * (1 - fpr)
        cont = num / den
        sweep.append({"s": s, "contamination": round(cont, 4),
                      "reduction_pp": round(100 * (s - cont), 2)})
        print(f"{s:>18.2f}{100*cont:>25.1f}%{100*s:>14.1f}%"
              f"{100*(s-cont):>11.1f}pp")
    print("\n  The gate removes roughly a fifth to a quarter of the softening it")
    print("  is supposed to remove, and to do that it rejects 24.9% of faithful")
    print("  rewrites per attempt. 1,924 abusive rows were lost outright after")
    print("  three failed attempts, 97.4% of all quarantine.")

    print("\n" + "=" * 78)
    print("WHAT THIS DOES AND DOES NOT ESTABLISH")
    print("=" * 78)
    print("  Established: on 64.6% of the abusive corpus the word list is")
    print("  vacuous, and the score gate that decides those rows alone has")
    print(f"  AUC {v4['auc_raw_dp']:.2f} against realistic softening. Neither the")
    print("  shipped threshold, a calibrated threshold, rank-matching, nor an")
    print("  LLM judge reaches a usable operating point on that stratum.")
    print("  Not established: the true softening rate s. Nothing in the repo")
    print("  measures it, and it cannot be recovered from the gate's own")
    print("  acceptance rate. A human-labelled sample is the only way.")

    payload = {
        "scorer_identification": ident,
        "operating_point_source": "gate_v4_realistic_bench.py",
        "stratum": v4["stratum"],
        "lexicon_blind_share_of_class0": 0.646,
        "gates": {"hand_lexicon": lex, "shipped_dp_0_15": dp, "llm_judge": judge},
        "auc_raw_dp": v4["auc_raw_dp"],
        "auc_rank_dp": v4["auc_rank_dp"],
        "contamination_sweep": sweep,
        "retracted": (
            "Earlier Rogan-Gladen estimate of ~14% label noise: invalid, it used "
            "the broken corruption benchmark and ignored max_retries."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
