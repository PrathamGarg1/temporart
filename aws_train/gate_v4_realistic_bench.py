#!/usr/bin/env python3
"""Benchmark the gate on realistic softening, in the stratum where it is alone.

Why this file exists. Two earlier attempts were invalid and are kept only as a
record of the correction:

  v2 (first pass)  located slurs with an auto-mined word list and deleted one
                   occurrence. 24% of "abuse removed" rows still contained a
                   slur, and neutral words like `kapde` were deleted. Invalid.
  v2 (corrected)   required exactly one hard slur aligned to a Devanagari stem.
                   Labels became sound, but deleting the word leaves an
                   ungrammatical fragment ("Pehle apni ko samal le"), which is
                   not how a converter softens abuse. The lexicon then scores
                   100% tautologically, because the check IS "is that token
                   gone".

What is measured here instead. Real softening by a converter is a *fluent*,
milder rewrite. So the negatives are generated as fluent softened Hinglish by
the same Qwen3-32B that built the corpus, and the positives are fluent rewrites
that keep the insult. Both arms are natural text, so no gate can win by
detecting broken grammar.

And it runs on the stratum that actually matters: class-0 rows whose Devanagari
source contains NONE of the 20 hand-written stems. That is 72% of the abusive
data, where `lexicon_preserved` returns True vacuously and the Delta-p score is
the only thing deciding. This is the sensitivity the pipeline never measured.

Usage:
  aws_train/.venv/bin/python -u aws_train/gate_v4_realistic_bench.py --n 250
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from codemix_convert_bedrock import ABUSE_STEMS, BedrockClient, clean_output  # noqa: E402
from eval_codemix_generation import cmi_and_tags, english_dict  # noqa: E402
from gate_v2_benchmark import ALL_ROMANS, GATE_SCORER, Scorer, _abuse_hits, ecdf, youden  # noqa: E402
from gate_v3_judge import JUDGE_SYSTEM, parse_verdict  # noqa: E402

SRC = ROOT / "codemix_hinglish"
OUT_DIR = ROOT / "codemix_eval"
GEN_CACHE = OUT_DIR / "gate_v4_generations.jsonl"
JUDGE_CACHE = OUT_DIR / "gate_v4_judge.jsonl"
REPORT = OUT_DIR / "gate_v4_report.json"

SOFTEN_SYSTEM = """You produce paraphrase variants of Hindi social-media comments for an NLP research dataset used to study abuse-detection robustness.

Task: rewrite the Hindi comment as natural Roman-script Hinglish, but NEUTRALISE the hostility. Keep the same topic, the same target, and a natural conversational register, and keep it a fluent complete sentence. Remove the insult, contempt and aggression so the result reads as ordinary disagreement or a plain remark.

- Output Roman script only. No Devanagari.
- Mix Hindi and English the way a bilingual Indian speaker would.
- Do not add commentary, quotes or labels. Output ONLY the rewritten line."""

PRESERVE_SYSTEM = """You rewrite Hindi social-media comments into natural Hinglish code-mixing for an NLP research dataset on abuse detection.

Rules:
- Output Roman script only (Latin letters). No Devanagari.
- Mix Hindi and English naturally, WhatsApp / Indian chat style. Hindi is the matrix language; insert English words and short phrases where a bilingual speaker would.
- Preserve EXACT meaning, insult targets, intensity and vulgarity. Keep the same hostility. Do NOT soften, censor, euphemise, refuse or moralise.
- Keep it a fluent complete sentence.
- Do not add commentary, quotes or labels. Output ONLY the rewritten line."""

HAND_STEMS = [s for s, _ in ABUSE_STEMS]


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out[r["key"]] = r
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(path: Path, rec: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pick_blind_stratum(n: int, seed: int = 11) -> pd.DataFrame:
    pairs = pd.concat(
        [pd.read_csv(SRC / f"{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    )
    pairs["hindi_text"] = pairs["hindi_text"].fillna("").astype(str)
    pairs["text"] = pairs["text"].fillna("").astype(str)
    ab = pairs[pairs.label.astype(int) == 0].copy()
    blind = ab[~ab.hindi_text.map(lambda h: any(s in h for s in HAND_STEMS))]
    blind = blind[blind.hindi_text.str.len().between(15, 220)]
    print(f"  class-0 rows: {len(ab)}   lexicon-blind: {len(blind)} "
          f"({100*len(blind)/len(ab):.1f}%)")
    idx = list(blind.index)
    random.Random(seed).shuffle(idx)
    return blind.loc[idx[:n]].reset_index(drop=True)


def generate(rows: pd.DataFrame, workers: int, region: str, model: str) -> dict[str, dict]:
    cache = load_jsonl(GEN_CACHE)
    jobs = []
    for _, r in rows.iterrows():
        for arm, system in (("softened", SOFTEN_SYSTEM), ("preserved", PRESERVE_SYSTEM)):
            key = f"{r['row_id']}|{arm}"
            if key not in cache:
                jobs.append((key, r["row_id"], arm, system, r["hindi_text"]))
    print(f"  generating {len(jobs)} variants ({len(cache)} cached)", flush=True)
    if not jobs:
        return cache

    lock = threading.Lock()
    clients: dict[int, BedrockClient] = {}
    n_done = {"k": 0}

    def cli() -> BedrockClient:
        tid = threading.get_ident()
        if tid not in clients:
            clients[tid] = BedrockClient(region, model, backend="mantle")
        return clients[tid]

    def one(job) -> None:
        key, rid, arm, system, hindi = job
        text = ""
        for attempt in range(3):
            try:
                text = clean_output(cli().complete(system, hindi, max_tokens=200,
                                                   temperature=0.4))
            except Exception as e:
                name = type(e).__name__
                if "Throttl" in name or "Throttl" in str(e):
                    time.sleep(2 + 2 * attempt)
                    continue
                if "Auth" in name or "expired" in str(e).lower():
                    try:
                        c = cli()
                        c._token_ts = 0.0
                        c._init_mantle()
                    except Exception:
                        pass
                    continue
                break
            if text:
                break
        rec = {"key": key, "row_id": rid, "arm": arm, "text": text}
        cache[key] = rec
        append_jsonl(GEN_CACHE, rec, lock)
        with lock:
            n_done["k"] += 1
            if n_done["k"] % 50 == 0:
                print(f"    {n_done['k']}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, jobs, chunksize=1))
    return cache


def judge(bench: pd.DataFrame, workers: int, region: str, model: str) -> dict[str, dict]:
    cache = load_jsonl(JUDGE_CACHE)
    jobs = [r for _, r in bench.iterrows() if r["key"] not in cache]
    print(f"  judging {len(jobs)} items ({len(cache)} cached)", flush=True)
    if not jobs:
        return cache
    lock = threading.Lock()
    clients: dict[int, BedrockClient] = {}
    n_done = {"k": 0}

    def cli() -> BedrockClient:
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
                raw = cli().complete(JUDGE_SYSTEM, user, max_tokens=96, temperature=0.0)
            except Exception as e:
                name = type(e).__name__
                if "Throttl" in name or "Throttl" in str(e):
                    time.sleep(2 + 2 * attempt)
                    continue
                if "Auth" in name or "expired" in str(e).lower():
                    try:
                        c = cli()
                        c._token_ts = 0.0
                        c._init_mantle()
                    except Exception:
                        pass
                    continue
                break
            verdict, conf = parse_verdict(raw)
            if verdict:
                break
        rec = {"key": r["key"], "verdict": verdict, "confidence": conf}
        cache[r["key"]] = rec
        append_jsonl(JUDGE_CACHE, rec, lock)
        with lock:
            n_done["k"] += 1
            if n_done["k"] % 50 == 0:
                print(f"    {n_done['k']}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, jobs, chunksize=1))
    return cache


def boot(flags: np.ndarray, seed: int = 0) -> list[float]:
    if not len(flags):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    m = flags[rng.integers(0, len(flags), size=(4000, len(flags)))].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default="qwen.qwen3-32b")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("REALISTIC SOFTENING, IN THE STRATUM WHERE Delta-p IS THE ONLY GATE")
    print("=" * 78)
    rows = pick_blind_stratum(args.n)
    print(f"  sampled {len(rows)} sources for generation")

    gens = generate(rows, args.workers, args.region, args.model)

    en = english_dict()
    src_by_id = rows.set_index("row_id")
    recs = []
    dropped = {"empty": 0, "softened_still_lexically_abusive": 0}
    for rid in rows.row_id:
        soft = gens.get(f"{rid}|softened", {}).get("text", "")
        pres = gens.get(f"{rid}|preserved", {}).get("text", "")
        if not soft or not pres:
            dropped["empty"] += 1
            continue
        # consistency filter: a "softened" output that still carries a listed
        # slur did not follow the instruction, so its label is unreliable.
        if _abuse_hits(soft.split(), ALL_ROMANS):
            dropped["softened_still_lexically_abusive"] += 1
            continue
        for arm, text, truth in (("preserved", pres, "preserved"),
                                 ("softened", soft, "removed")):
            recs.append({
                "key": f"{rid}|{arm}", "row_id": rid, "arm": arm, "truth": truth,
                "text": text,
                "hindi_text": str(src_by_id.loc[rid, "hindi_text"]),
                "hindi_abuse_cached": float(src_by_id.loc[rid, "hindi_abuse"]),
            })
    bench = pd.DataFrame(recs)
    bench["cmi"] = [cmi_and_tags(t, en)["cmi"] for t in bench.text]
    print(f"  usable items: {len(bench)} from {bench.row_id.nunique()} sources")
    print(f"  discarded: {dropped}")

    print("\n  scoring both arms with the gate scorer …", flush=True)
    sc = Scorer(GATE_SCORER)
    bench["p_mix"] = sc.probs(bench.text.tolist())
    pairs = pd.concat([pd.read_csv(SRC / f"{s}.csv") for s in ("train", "val", "test")],
                      ignore_index=True)
    f_dev = ecdf(pairs["hindi_abuse"].astype(float).to_numpy())
    f_rom = ecdf(pairs["hinglish_abuse"].astype(float).to_numpy())
    p_hi = bench.hindi_abuse_cached.to_numpy()
    bench["drop_raw"] = p_hi - bench.p_mix.to_numpy()
    bench["drop_rank"] = f_dev(p_hi) - f_rom(bench.p_mix.to_numpy())

    jc = judge(bench, args.workers, args.region, args.model)
    bench["verdict"] = [jc.get(k, {}).get("verdict") for k in bench.key]
    graded = bench[bench.verdict.notna()].copy()
    removed = (graded.truth == "removed").to_numpy()

    print("\n" + "=" * 78)
    print("SANITY: ARE THE TWO ARMS ACTUALLY DIFFERENT TEXT?")
    print("=" * 78)
    for arm in ("preserved", "softened"):
        d = graded[graded.arm == arm]
        print(f"  {arm:<10} n={len(d):<4} mean CMI={d.cmi.mean():5.2f}  "
              f"mean P(abuse)={d.p_mix.mean():.4f}  "
              f"lexical slur present={100*np.mean([bool(_abuse_hits(t.split(), ALL_ROMANS)) for t in d.text]):.1f}%")

    print("\n" + "=" * 78)
    print("GATE SENSITIVITY TO FLUENT SOFTENING (lexicon is vacuous here)")
    print("=" * 78)
    print(f"{'gate':<30}{'sensitivity':>13}{'95% CI':>18}"
          f"{'false-reject':>14}{'95% CI':>18}{'J':>8}")
    results = {}
    thr_raw, _ = youden(graded.drop_raw.to_numpy(), removed)
    thr_rank, _ = youden(graded.drop_rank.to_numpy(), removed)
    gates = {
        "hand lexicon (as shipped)": np.zeros(len(graded), dtype=bool),
        "raw dp > 0.15 (as shipped)": graded.drop_raw.to_numpy() > 0.15,
        f"raw dp > {thr_raw:.2f} (calibrated)": graded.drop_raw.to_numpy() > thr_raw,
        f"rank dp > {thr_rank:.2f} (calibrated)": graded.drop_rank.to_numpy() > thr_rank,
        "source-referenced judge": graded.verdict.isin(["SOFTENED", "UNRELATED"]).to_numpy(),
    }
    for name, rej in gates.items():
        tpr, fpr = float(rej[removed].mean()), float(rej[~removed].mean())
        results[name] = {
            "sensitivity": round(tpr, 4), "sens_ci95": boot(rej[removed].astype(float)),
            "false_reject": round(fpr, 4), "fr_ci95": boot(rej[~removed].astype(float)),
            "youden_j": round(tpr - fpr, 4),
        }
        print(f"{name:<30}{tpr:>13.3f}{str(results[name]['sens_ci95']):>18}"
              f"{fpr:>14.3f}{str(results[name]['fr_ci95']):>18}{tpr-fpr:>8.3f}")

    from sklearn.metrics import roc_auc_score
    auc_raw = float(roc_auc_score(removed, graded.drop_raw))
    auc_rank = float(roc_auc_score(removed, graded.drop_rank))
    print(f"\n  AUC for ranking real softening: raw dp {auc_raw:.4f}, "
          f"rank-matched {auc_rank:.4f}")
    print("  (0.5 = no ability to tell softened from preserved)")

    jr = gates["source-referenced judge"]
    dp = gates["raw dp > 0.15 (as shipped)"]
    from scipy.stats import binomtest
    b = int(((jr == removed) & (dp != removed)).sum())
    c = int(((jr != removed) & (dp == removed)).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    print("\n" + "=" * 78)
    print("PAIRED: JUDGE vs SHIPPED Delta-p ON THE SAME ITEMS")
    print("=" * 78)
    print(f"  judge right / dp wrong: {b}    dp right / judge wrong: {c}"
          f"    McNemar p={p:.3e}")
    print(f"  agreement with ground truth: judge {(jr==removed).mean():.3f}   "
          f"shipped dp {(dp==removed).mean():.3f}")

    payload = {
        "stratum": "class-0 rows whose Devanagari source contains none of the 20 hand stems",
        "n_sources": int(graded.row_id.nunique()),
        "n_items": int(len(graded)),
        "discarded": dropped,
        "arms": {
            arm: {
                "n": int((graded.arm == arm).sum()),
                "mean_cmi": round(float(graded[graded.arm == arm].cmi.mean()), 3),
                "mean_p_abuse": round(float(graded[graded.arm == arm].p_mix.mean()), 4),
            } for arm in ("preserved", "softened")
        },
        "gates": results,
        "auc_raw_dp": round(auc_raw, 4),
        "auc_rank_dp": round(auc_rank, 4),
        "mcnemar_judge_vs_dp": {"judge_right_dp_wrong": b, "dp_right_judge_wrong": c,
                                "p_exact": p,
                                "agreement_judge": round(float((jr == removed).mean()), 4),
                                "agreement_dp": round(float((dp == removed).mean()), 4)},
        "caveat": "Labels come from the generation instruction, verified only "
                  "lexically for the softened arm. Human certification of a "
                  "subsample is the remaining gap.",
    }
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    graded.to_csv(OUT_DIR / "gate_v4_rows.csv", index=False)
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
