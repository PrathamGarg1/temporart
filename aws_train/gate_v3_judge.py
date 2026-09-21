#!/usr/bin/env python3
"""Benchmark a source-referenced judge against the same known ground truth.

The shipped gate compares two absolute probabilities from one Devanagari-trained
scorer across a script boundary, and measures 0.383 sensitivity to real abuse
removal. Rank-matching that comparison does not help (AUC 0.672 vs 0.680), so
the problem is not calibration -- a single scorer reading the rewrite in
isolation cannot tell "abuse softened" from "script changed".

This evaluates the alternative: show a judge the Devanagari source AND the
rewrite together, and ask for a relative decision. Same benchmark rows, same
ground truth, paired McNemar against the shipped gate.

The judge never sees the variant name or the label.

Usage:
  aws_train/.venv/bin/python -u aws_train/gate_v3_judge.py --n 300 --workers 8
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from codemix_convert_bedrock import ABUSE_STEMS, BedrockClient, fold_roman  # noqa: E402
from gate_v2_benchmark import lexicon_ok  # noqa: E402

SRC = ROOT / "codemix_hinglish"
OUT_DIR = ROOT / "codemix_eval"
BENCH = OUT_DIR / "gate_v2_benchmark_rows.csv"
CACHE = OUT_DIR / "gate_v3_judge_cache.jsonl"
REPORT = OUT_DIR / "gate_v3_judge_report.json"

JUDGE_SYSTEM = """You audit a Hindi-to-Hinglish rewriting system for an abuse-detection research corpus. This is dataset quality control on existing user text.

You are given a Hindi source comment and its Roman-script Hinglish rewrite. Decide only whether the rewrite still carries the SAME abusive force as the source.

PRESERVED - the insult, slur, threat or vulgarity of the source is still present at comparable intensity. Script changes, English words mixed in, word order changes and paraphrase are all fine and still count as PRESERVED.
SOFTENED - the abusive term was dropped, censored, euphemised, or replaced by a neutral or friendly word, so the rewrite is milder than the source.
UNRELATED - the rewrite says something substantially different from the source.

Judge only abuse intensity relative to the source. Do not judge fluency, how much English was used, or how offensive the source itself is.

Reply with JSON only: {"verdict": "PRESERVED"|"SOFTENED"|"UNRELATED", "confidence": 0.0-1.0}"""

JSON_RE = re.compile(r"\{.*\}", re.S)


def build_subset(n: int, seed: int = 7) -> pd.DataFrame:
    bench = pd.read_csv(BENCH)
    pairs = pd.concat(
        [pd.read_csv(SRC / f"{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    ).set_index("row_id")
    bench["hindi_text"] = pairs.loc[bench.row_id, "hindi_text"].astype(str).to_numpy()
    ids = sorted(bench.row_id.unique())
    random.Random(seed).shuffle(ids)
    keep = set(ids[:n])
    sub = bench[bench.row_id.isin(keep)].copy()
    sub["item"] = sub.row_id.astype(str) + "|" + sub.variant
    return sub.reset_index(drop=True)


def load_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["item"]] = rec
            except json.JSONDecodeError:
                continue
    return out


def parse_verdict(raw: str) -> tuple[str | None, float]:
    m = JSON_RE.search(raw or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).strip().upper()
            if v in ("PRESERVED", "SOFTENED", "UNRELATED"):
                return v, float(obj.get("confidence", 0.5))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    up = (raw or "").upper()
    for v in ("PRESERVED", "SOFTENED", "UNRELATED"):
        if v in up:
            return v, 0.5
    return None, 0.0


def run_judge(sub: pd.DataFrame, workers: int, region: str, model: str) -> dict[str, dict]:
    cache = load_cache()
    pending = [r for _, r in sub.iterrows() if r["item"] not in cache]
    print(f"  judging {len(pending)} of {len(sub)} items "
          f"({len(cache)} cached) with {workers} workers", flush=True)
    if not pending:
        return cache

    lock = threading.Lock()
    clients: dict[int, BedrockClient] = {}
    done = {"n": 0, "fail": 0}

    def client() -> BedrockClient:
        tid = threading.get_ident()
        if tid not in clients:
            clients[tid] = BedrockClient(region, model, backend="mantle")
        return clients[tid]

    def one(r: pd.Series) -> None:
        user = (f"Hindi source:\n{r['hindi_text']}\n\n"
                f"Hinglish rewrite:\n{r['text']}\n\nJSON:")
        verdict, conf, raw = None, 0.0, ""
        for attempt in range(3):
            try:
                raw = client().complete(JUDGE_SYSTEM, user, max_tokens=96,
                                       temperature=0.0)
            except Exception as e:
                name = type(e).__name__
                if "Throttl" in name or "Throttl" in str(e):
                    import time
                    time.sleep(2 + 2 * attempt)
                    continue
                if "Auth" in name or "expired" in str(e).lower():
                    try:
                        c = client()
                        c._token_ts = 0.0
                        c._init_mantle()
                    except Exception:
                        pass
                    continue
                break
            verdict, conf = parse_verdict(raw)
            if verdict:
                break
        rec = {"item": r["item"], "row_id": r["row_id"], "variant": r["variant"],
               "truth": r["truth"], "verdict": verdict, "confidence": conf,
               "raw": (raw or "")[:200]}
        with lock:
            with CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache[r["item"]] = rec
            done["n"] += 1
            if verdict is None:
                done["fail"] += 1
            if done["n"] % 50 == 0:
                print(f"    {done['n']}/{len(pending)} "
                      f"(unparseable {done['fail']})", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, pending, chunksize=1))
    print(f"  done; unparseable {done['fail']}", flush=True)
    return cache


def boot_ci(flags: np.ndarray, n_boot: int = 4000, seed: int = 0) -> list[float]:
    if len(flags) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    m = flags[rng.integers(0, len(flags), size=(n_boot, len(flags)))].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="source rows (x4 variants)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default="qwen.qwen3-32b")
    args = ap.parse_args()

    sub = build_subset(args.n)
    print("=" * 78)
    print("SOURCE-REFERENCED JUDGE vs SHIPPED GATE, SAME GROUND TRUTH")
    print("=" * 78)
    print(f"  {sub.row_id.nunique()} source rows x 4 variants = {len(sub)} judgments")

    cache = run_judge(sub, args.workers, args.region, args.model)
    sub["verdict"] = [cache.get(i, {}).get("verdict") for i in sub.item]
    n_unparsed = int(sub.verdict.isna().sum())
    graded = sub[sub.verdict.notna()].copy()

    removed = (graded.truth == "removed").to_numpy()
    judge_rej = graded.verdict.isin(["SOFTENED", "UNRELATED"]).to_numpy()
    lex = np.array([not lexicon_ok(h, t, ABUSE_STEMS)
                    for h, t in zip(graded.hindi_text, graded.text)])
    dp = graded.drop_raw.to_numpy() > 0.15
    shipped_rej = lex | dp

    print(f"\n  unparseable judge replies: {n_unparsed} "
          f"({100*n_unparsed/max(len(sub),1):.1f}%) — excluded, not counted as pass")

    print("\n" + "=" * 78)
    print("DETECTION OF REAL ABUSE REMOVAL")
    print("=" * 78)
    print(f"{'gate':<28}{'sensitivity':>13}{'95% CI':>18}"
          f"{'false-reject':>14}{'95% CI':>18}{'J':>8}")
    rows = {}
    for name, rej in (("shipped (lexicon OR dp)", shipped_rej),
                      ("source-referenced judge", judge_rej)):
        tpr = float(rej[removed].mean())
        fpr = float(rej[~removed].mean())
        rows[name] = {
            "sensitivity": round(tpr, 4), "sens_ci95": boot_ci(rej[removed].astype(float)),
            "false_reject": round(fpr, 4), "fr_ci95": boot_ci(rej[~removed].astype(float)),
            "youden_j": round(tpr - fpr, 4),
            "by_variant": {v: round(float(rej[(graded.variant == v).to_numpy()].mean()), 4)
                           for v in ("original", "inject_english", "drop_slur", "swap_benign")},
        }
        r = rows[name]
        print(f"{name:<28}{tpr:>13.3f}{str(r['sens_ci95']):>18}"
              f"{fpr:>14.3f}{str(r['fr_ci95']):>18}{tpr-fpr:>8.3f}")

    print("\n  per-arm reject rate (ideal: 0, 0, 1, 1)")
    print(f"{'gate':<28}{'original':>11}{'inject_en':>11}{'drop_slur':>11}{'swap_benign':>13}")
    for name in rows:
        v = rows[name]["by_variant"]
        print(f"{name:<28}{v['original']:>11.3f}{v['inject_english']:>11.3f}"
              f"{v['drop_slur']:>11.3f}{v['swap_benign']:>13.3f}")

    from scipy.stats import binomtest, chi2, fisher_exact
    correct_j = judge_rej == removed
    correct_s = shipped_rej == removed
    b = int((correct_j & ~correct_s).sum())
    c = int((~correct_j & correct_s).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    print("\n" + "=" * 78)
    print("PAIRED COMPARISON (McNemar on agreement with ground truth)")
    print("=" * 78)
    print(f"  judge correct / shipped wrong : {b}")
    print(f"  judge wrong  / shipped correct: {c}")
    print(f"  chi2_cc={stat:.3f}  p_exact={p:.3e}")
    print(f"  overall agreement with truth: judge {correct_j.mean():.3f}  "
          f"shipped {correct_s.mean():.3f}")

    o = (graded.variant == "original").to_numpy()
    ie = (graded.variant == "inject_english").to_numpy()
    print("\n" + "=" * 78)
    print("CODE-MIXING PENALTY (slur intact, 2 benign English words added)")
    print("=" * 78)
    conf = {}
    for name, rej in (("shipped", shipped_rej), ("judge", judge_rej)):
        a1, b1 = int(rej[ie].sum()), int((~rej[ie]).sum())
        a0, b0 = int(rej[o].sum()), int((~rej[o]).sum())
        odds, pv = fisher_exact([[a1, b1], [a0, b0]])
        conf[name] = {"reject_original": round(float(rej[o].mean()), 4),
                      "reject_injected": round(float(rej[ie].mean()), 4),
                      "odds_ratio": round(float(odds), 4), "fisher_p": float(pv)}
        print(f"  {name:<10} {rej[o].mean():.3f} -> {rej[ie].mean():.3f}"
              f"   odds={odds:.2f}  p={pv:.3e}")

    payload = {
        "n_source_rows": int(graded.row_id.nunique()),
        "n_judgments": int(len(graded)),
        "unparseable": n_unparsed,
        "judge_model": args.model,
        "gates": rows,
        "mcnemar_judge_vs_shipped": {
            "judge_right_shipped_wrong": b, "shipped_right_judge_wrong": c,
            "chi2_cc": round(float(stat), 4), "p_exact": p,
            "agreement_judge": round(float(correct_j.mean()), 4),
            "agreement_shipped": round(float(correct_s.mean()), 4),
        },
        "codemix_penalty": conf,
        "verdict_distribution": {k: int(v) for k, v in
                                 graded.verdict.value_counts().items()},
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
