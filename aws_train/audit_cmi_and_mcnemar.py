#!/usr/bin/env python3
"""Two authentic measurements on the frozen Hinglish test set (n=6368).

A) Accuracy as a function of code-mixing depth (CMI). Tests whether the
   shallow-mix problem visible in the deck actually costs detection accuracy,
   which is the only thing that justifies a higher-CMI iteration.

B) McNemar paired test between the shipped model and the two ablations.
   The deck reports -0.60 pp and -0.72 pp as "do not ship" verdicts; an
   unpaired eyeball on 6368 rows cannot support that. This runs the correct
   paired test on identical rows.
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

MODELS = {
    "shipped": ROOT / "assets/models/custom-macd-model-shipped-84",
    "layer1": ROOT / "assets/models/custom-macd-model-layer1",
    "layer1_synth": ROOT / "assets/models/custom-macd-model-layer1_synth",
}
TEST = ROOT / "codemix_hinglish/test.csv"
OUT = ROOT / "codemix_eval/audit_cmi_mcnemar.json"


def predict(model_dir: Path, texts: list[str], batch: int = 64) -> np.ndarray:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    onnx = model_dir / "model_quantized.onnx"
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    preds = np.empty(len(texts), dtype=np.int64)
    for s in range(0, len(texts), batch):
        chunk = texts[s : s + batch]
        enc = tok(chunk, return_tensors="np", truncation=True, max_length=128, padding=True)
        feeds = {}
        for n in names:
            if n in enc:
                feeds[n] = enc[n].astype(np.int64)
            elif n == "token_type_ids":
                feeds[n] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        logits = sess.run(None, feeds)[0]
        preds[s : s + len(chunk)] = logits.argmax(-1)
        if s % (batch * 20) == 0:
            print(f"    {model_dir.name}: {s}/{len(texts)}", flush=True)
    return preds


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact-ish McNemar with continuity correction + binomial two-sided p."""
    from scipy.stats import binomtest, chi2

    b = int((a_correct & ~b_correct).sum())  # A right, B wrong
    c = int((~a_correct & b_correct).sum())  # A wrong, B right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "discordant": 0, "chi2": 0.0, "p_chi2": 1.0, "p_exact": 1.0}
    stat = (abs(b - c) - 1) ** 2 / n
    return {
        "b_shipped_right_other_wrong": b,
        "c_shipped_wrong_other_right": c,
        "discordant": n,
        "chi2_cc": round(float(stat), 4),
        "p_chi2": round(float(1 - chi2.cdf(stat, 1)), 5),
        "p_exact_binomial": round(float(binomtest(b, n, 0.5).pvalue), 5),
    }


def main() -> None:
    df = pd.read_csv(TEST)
    df["text"] = df["text"].fillna("").astype(str)
    y = df["label"].astype(int).to_numpy()
    texts = df["text"].tolist()
    print(f"frozen Hinglish test n={len(df)}  class0={int((y==0).sum())} class1={int((y==1).sum())}")

    en = english_dict()
    cmi = np.array([cmi_and_tags(t, en)["cmi"] for t in texts])

    preds = {}
    for name, path in MODELS.items():
        print(f"  scoring {name} …", flush=True)
        preds[name] = predict(path, texts)

    report: dict = {"n": int(len(df))}

    print("\n" + "=" * 78)
    print("A) DOES CODE-MIXING DEPTH (CMI) PREDICT FAILURE?  [shipped INT8]")
    print("=" * 78)
    p = preds["shipped"]
    correct = p == y
    buckets = [
        ("CMI = 0 (not mixed)", cmi <= 1e-9),
        ("0 < CMI < 10", (cmi > 1e-9) & (cmi < 10)),
        ("10 <= CMI < 20", (cmi >= 10) & (cmi < 20)),
        ("20 <= CMI < 30", (cmi >= 20) & (cmi < 30)),
        ("CMI >= 30 (human-like)", cmi >= 30),
    ]
    print(f"{'bucket':<24}{'n':>6}{'share%':>8}{'acc':>8}{'95% CI':>16}"
          f"{'recall_abuse':>14}{'FNR_abuse':>11}")
    rows = []
    for name, m in buckets:
        n = int(m.sum())
        if n == 0:
            continue
        acc = float(correct[m].mean())
        lo, hi = wilson(int(correct[m].sum()), n)
        ab = m & (y == 0)
        rec = float((p[ab] == 0).mean()) if ab.sum() else float("nan")
        rows.append({
            "bucket": name, "n": n, "share_pct": round(100 * n / len(df), 2),
            "accuracy": round(acc, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_abusive": int(ab.sum()),
            "recall_abusive": None if np.isnan(rec) else round(rec, 4),
        })
        print(f"{name:<24}{n:>6}{100*n/len(df):>8.2f}{acc:>8.4f}"
              f"{f'[{lo:.3f},{hi:.3f}]':>16}{rec:>14.4f}{1-rec:>11.4f}")
    report["cmi_buckets_shipped"] = rows

    lowm = cmi < 10
    highm = cmi >= 20
    from scipy.stats import fisher_exact
    tab = [[int(correct[lowm].sum()), int((~correct[lowm]).sum())],
           [int(correct[highm].sum()), int((~correct[highm]).sum())]]
    odds, pv = fisher_exact(tab)
    print(f"\nlow-mix (CMI<10, n={int(lowm.sum())}) acc={correct[lowm].mean():.4f}"
          f"   deep-mix (CMI>=20, n={int(highm.sum())}) acc={correct[highm].mean():.4f}")
    print(f"Fisher exact on correct/incorrect: odds={odds:.3f}  p={pv:.3e}")
    report["low_vs_deep_mix"] = {
        "acc_cmi_lt_10": round(float(correct[lowm].mean()), 4),
        "n_cmi_lt_10": int(lowm.sum()),
        "acc_cmi_ge_20": round(float(correct[highm].mean()), 4),
        "n_cmi_ge_20": int(highm.sum()),
        "fisher_odds": round(float(odds), 4),
        "fisher_p": float(pv),
    }

    print("\nMean CMI of the frozen test set: "
          f"{cmi.mean():.2f}   share CMI=0: {100*(cmi<=1e-9).mean():.1f}%   "
          f"share CMI>=30: {100*(cmi>=30).mean():.1f}%")
    report["test_cmi"] = {
        "mean": round(float(cmi.mean()), 3),
        "pct_eq_0": round(float(100 * (cmi <= 1e-9).mean()), 2),
        "pct_ge_20": round(float(100 * (cmi >= 20).mean()), 2),
        "pct_ge_30": round(float(100 * (cmi >= 30).mean()), 2),
    }

    print("\n" + "=" * 78)
    print("B) McNEMAR: ARE THE -0.60 / -0.72 pp ABLATION VERDICTS REAL?")
    print("=" * 78)
    base = preds["shipped"] == y
    print(f"{'comparison':<34}{'acc A':>9}{'acc B':>9}{'diff pp':>9}"
          f"{'b':>6}{'c':>6}{'p (exact)':>11}{'verdict':>14}")
    mc = {}
    for other in ("layer1", "layer1_synth"):
        ob = preds[other] == y
        res = mcnemar(base, ob)
        diff = 100 * (base.mean() - ob.mean())
        sig = "significant" if res["p_exact_binomial"] < 0.05 else "NOT significant"
        mc[other] = {**res, "acc_shipped": round(float(base.mean()), 4),
                     "acc_other": round(float(ob.mean()), 4),
                     "diff_pp": round(float(diff), 3), "verdict": sig}
        print(f"{'shipped vs ' + other:<34}{base.mean():>9.4f}{ob.mean():>9.4f}"
              f"{diff:>9.2f}{res['b_shipped_right_other_wrong']:>6}"
              f"{res['c_shipped_wrong_other_right']:>6}"
              f"{res['p_exact_binomial']:>11.4f}{sig:>14}")
    report["mcnemar_vs_shipped"] = mc

    print("\n" + "=" * 78)
    print("C) HOW MUCH DO THE THREE MODELS ACTUALLY DISAGREE?")
    print("=" * 78)
    names = list(MODELS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = int((preds[names[i]] != preds[names[j]]).sum())
            print(f"  {names[i]:<14} vs {names[j]:<14} disagree on {d:>5} "
                  f"rows ({100*d/len(df):.2f}%)")

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
