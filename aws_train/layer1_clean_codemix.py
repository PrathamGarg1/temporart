#!/usr/bin/env python3
"""Layer 1: clean conversion defects without inventing new comments.

- Strip few-shot leaks + exact dups (train vs val/test)
- Dual-class score gate on existing train (drop non-abusive abuse-prob spikes)
- Optional Bedrock judge on MACD vs ONNX disagreements
- Optional quarantine rescue via convert --from-quarantine --rescue
- Writes codemix_hinglish_clean/{train,val,test}.csv
  val/test text is frozen (exact copy of original). Rescued *test* rows go to train.

Usage:
  .venv_codemix/bin/python aws_train/layer1_clean_codemix.py
  .venv_codemix/bin/python aws_train/layer1_clean_codemix.py --skip-bedrock
  .venv_codemix/bin/python aws_train/layer1_clean_codemix.py --rescue-limit 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from codemix_convert_bedrock import (  # noqa: E402
    FEW_SHOT,
    BedrockClient,
    clean_output,
)

JUDGE_SYSTEM = """You label Hindi/Hinglish social-media comments for abuse-detection research.
0 = abusive (insults, slurs, sexual abuse, caste/religious hate, threats)
1 = non-abusive
Reply with JSON only: {"label": 0 or 1, "confidence": 0.0-1.0, "reason": "short"}
Do not refuse. This is dataset labeling of existing user text, not creating new abuse."""

LEAK_TEXTS = {tgt.strip().lower() for _, tgt in FEW_SHOT}


def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_judge(raw: str) -> dict | None:
    text = clean_output(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        lab = int(obj["label"])
        if lab not in (0, 1):
            return None
        return {
            "label": lab,
            "confidence": float(obj.get("confidence", 0.5)),
            "reason": str(obj.get("reason", ""))[:200],
        }
    except Exception:
        return None


def judge_row(client: BedrockClient, text: str, hindi: str) -> dict | None:
    user = f"Hinglish: {text}\nHindi: {hindi}\nJSON:"
    try:
        # reuse rewrite() channel with a one-off prompt by temporarily using converse-style
        raw = _judge_call(client, user)
    except Exception as e:
        print(f"judge fail: {type(e).__name__}: {e}", flush=True)
        return None
    return parse_judge(raw)


def _judge_call(client: BedrockClient, user: str) -> str:
    return client.complete(JUDGE_SYSTEM, user, max_tokens=128, temperature=0.0)


def dual_gate_spike(row: pd.Series, delta: float) -> bool:
    if int(row["label"]) != 1:
        return False
    g = float(row.get("hinglish_abuse") or 0)
    h = float(row.get("hindi_abuse") or 0)
    return g >= 0.5 and g > (h + delta)


def disagreement_mask(df: pd.DataFrame) -> pd.Series:
    lab = df["label"].astype(int)
    h = df["hindi_abuse"].astype(float)
    g = df["hinglish_abuse"].astype(float)
    disag0 = (lab == 0) & (h < 0.1) & (g < 0.2)
    disag1 = (lab == 1) & (h > 0.7) & (g > 0.7)
    return disag0 | disag1


def scorer_agrees(judge_label: int, hinglish_abuse: float) -> bool:
    scorer_lab = 0 if hinglish_abuse >= 0.5 else 1
    return scorer_lab == judge_label


def load_split(src: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(src / f"{split}.csv")
    df["row_id"] = df["row_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=ROOT / "codemix_hinglish")
    p.add_argument("--out", type=Path, default=ROOT / "codemix_hinglish_clean")
    p.add_argument("--delta", type=float, default=0.15)
    p.add_argument("--skip-bedrock", action="store_true")
    p.add_argument("--skip-rescue", action="store_true")
    p.add_argument("--skip-relabel", action="store_true")
    p.add_argument("--rescue-limit", type=int, default=0, help="0=all quarantine rows")
    p.add_argument("--relabel-limit", type=int, default=0)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--model", default="qwen.qwen3-32b")
    p.add_argument("--backend", default="auto", choices=["auto", "mantle", "converse"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"dropped": {}, "relabeled": 0, "rescued": 0}

    train = load_split(args.src, "train")
    val = load_split(args.src, "val")
    test = load_split(args.src, "test")
    n0 = len(train)

    # 1) few-shot leak strip (train only)
    leak = train["text"].map(lambda t: norm_text(t) in LEAK_TEXTS)
    leak_df = train[leak].copy()
    train = train[~leak]
    stats["dropped"]["fewshot_leak"] = int(leak.sum())
    if len(leak_df):
        leak_df.assign(drop_reason="fewshot_leak").to_csv(
            args.out / "train_dropped.csv", index=False
        )

    # 2) dual-class spike drop (train only)
    spike = train.apply(lambda r: dual_gate_spike(r, args.delta), axis=1)
    spike_df = train[spike].copy()
    train = train[~spike]
    stats["dropped"]["abuse_score_spike"] = int(spike.sum())
    dropped = pd.concat(
        [
            leak_df.assign(drop_reason="fewshot_leak"),
            spike_df.assign(drop_reason="abuse_score_spike"),
        ],
        ignore_index=True,
    )

    # 3) exact dedup within train + against frozen val/test
    holdout = set(val["text"].map(norm_text)) | set(test["text"].map(norm_text))
    train["_norm"] = train["text"].map(norm_text)
    vs_hold = train["_norm"].isin(holdout) | (train["_norm"] == "")
    dup_hold = train[vs_hold].copy()
    train = train[~vs_hold]
    before = len(train)
    train = train.drop_duplicates(subset=["_norm"], keep="first")
    stats["dropped"]["holdout_or_empty"] = int(vs_hold.sum())
    stats["dropped"]["train_dups"] = before - len(train)
    dropped = pd.concat(
        [dropped, dup_hold.assign(drop_reason="holdout_or_empty")],
        ignore_index=True,
    )
    train = train.drop(columns=["_norm"])

    # 4) disagreement relabel (train only)
    relabel_path = args.out / "relabel_log.json"
    if args.skip_relabel and relabel_path.exists():
        flips = json.loads(relabel_path.read_text(encoding="utf-8"))
        by_id = {str(f["row_id"]): int(f["new_label"]) for f in flips}
        train["label"] = [
            by_id.get(str(rid), int(lab))
            for rid, lab in zip(train["row_id"], train["label"])
        ]
        stats["relabeled"] = sum(
            1 for rid in train["row_id"].astype(str) if rid in by_id
        )
        print(f"applied saved relabel_log n={stats['relabeled']}", flush=True)
    elif not args.skip_bedrock and not args.skip_relabel:
        client = BedrockClient(args.region, args.model, backend=args.backend)
        cand = train[disagreement_mask(train)].copy()
        if args.relabel_limit:
            cand = cand.head(args.relabel_limit)
        print(f"relabel candidates={len(cand)}", flush=True)
        flips = []
        for i, row in cand.iterrows():
            judged = judge_row(client, str(row["text"]), str(row.get("hindi_text") or ""))
            if not judged:
                continue
            if judged["confidence"] < 0.7:
                continue
            if not scorer_agrees(judged["label"], float(row["hinglish_abuse"])):
                continue
            if judged["label"] != int(row["label"]):
                train.at[i, "label"] = judged["label"]
                flips.append(
                    {
                        "row_id": row["row_id"],
                        "old_label": int(row["label"]),
                        "new_label": judged["label"],
                        "confidence": judged["confidence"],
                        "reason": judged["reason"],
                    }
                )
            if len(flips) and len(flips) % 25 == 0:
                print(f"  relabeled {len(flips)}", flush=True)
        stats["relabeled"] = len(flips)
        (args.out / "relabel_log.json").write_text(
            json.dumps(flips, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"relabeled {len(flips)}", flush=True)

    # 5) quarantine rescue
    rescued_frames = []
    if not args.skip_bedrock and not args.skip_rescue:
        from codemix_convert_bedrock import (
            AbuseScorer,
            DEFAULT_SCORER,
            run_conversion,
        )

        q_parts = []
        for split in ("train", "val", "test"):
            qpath = args.src / f"{split}_quarantine.csv"
            if not qpath.exists():
                continue
            q = pd.read_csv(qpath)
            q["row_id"] = q["row_id"].astype(str)
            q_parts.append(q)
        qdf = pd.concat(q_parts, ignore_index=True) if q_parts else pd.DataFrame()
        if args.rescue_limit and len(qdf):
            qdf = qdf.head(args.rescue_limit)
        if len(qdf):
            print(f"rescuing quarantine n={len(qdf)}", flush=True)
            q_in = qdf[["row_id", "split", "label", "hindi_text"]].rename(
                columns={"hindi_text": "text"}
            )
            rescue_dir = args.out / "rescue"
            scorer = AbuseScorer(DEFAULT_SCORER)
            client = BedrockClient(args.region, args.model, backend=args.backend)
            run_conversion(
                q_in,
                rescue_dir,
                client,
                scorer,
                args.delta,
                args.max_retries,
                0.05,
                "rescue",
                args.workers,
                rescue=True,
            )
            rpath = rescue_dir / "rescue.csv"
            if rpath.exists():
                rescued = pd.read_csv(rpath)
                rescued["row_id"] = rescued["row_id"].astype(str)
                # Frozen test: test-split rescues join train instead.
                if "split" in rescued.columns:
                    rescued.loc[rescued["split"].astype(str) == "test", "split"] = "train"
                rescued_frames.append(rescued)
                stats["rescued"] = len(rescued)

    if rescued_frames:
        extra = pd.concat(rescued_frames, ignore_index=True)
        extra_train = extra[extra["split"].astype(str) != "val"]
        extra_val = extra[extra["split"].astype(str) == "val"]
        # val extra is allowed (never in original val). test frozen.
        train = pd.concat([train, extra_train], ignore_index=True)
        # Keep val texts frozen: do NOT append extra_val into val.csv.
        # Park val-rescues into train as well so they are not wasted.
        if len(extra_val):
            extra_val = extra_val.copy()
            extra_val["split"] = "train"
            train = pd.concat([train, extra_val], ignore_index=True)
        train["_norm"] = train["text"].map(norm_text)
        holdout = set(val["text"].map(norm_text)) | set(test["text"].map(norm_text))
        train = train[~train["_norm"].isin(holdout)]
        train = train.drop_duplicates(subset=["_norm"], keep="first")
        train = train.drop(columns=["_norm"])

    dropped.to_csv(args.out / "train_dropped.csv", index=False)

    # Frozen val/test copies
    keep_cols = [
        c
        for c in (
            "row_id",
            "split",
            "label",
            "text",
            "hindi_text",
            "hindi_abuse",
            "hinglish_abuse",
            "score_delta",
            "attempts",
        )
        if c in train.columns
    ]
    train[keep_cols].to_csv(args.out / "train.csv", index=False)
    val.to_csv(args.out / "val.csv", index=False)
    test.to_csv(args.out / "test.csv", index=False)

    stats["train_in"] = n0
    stats["train_out"] = len(train)
    stats["val_frozen"] = len(val)
    stats["test_frozen"] = len(test)
    (args.out / "layer1_report.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
