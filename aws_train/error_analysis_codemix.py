#!/usr/bin/env python3
"""INT8 error dump on frozen Hinglish splits: confusion matrix + FP/FN clusters.

Writes:
  codemix_synth/error_clusters.json
  codemix_synth/val_errors.csv
  codemix_synth/train_errors.csv   (seed source for hard-aug)
  codemix_synth/error_samples.json

Usage (from repo root):
  .venv_codemix/bin/python aws_train/error_analysis_codemix.py
  .venv_codemix/bin/python aws_train/error_analysis_codemix.py --split val
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from codemix_convert_bedrock import (  # noqa: E402
    ABUSE_STEMS,
    DEFAULT_SCORER,
    lexicon_preserved,
)

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
OBFUSC_RE = re.compile(
    r"(?i)(l[a@*.\-]*n[d.]|ch[u@*.\-]*t|g[a@*.\-]*n[d.]|bh[o0]sd|m+c+|f[*]+k)"
)
CASTE_REL_RE = re.compile(
    r"(?i)\b("
    r"bhim|bhangi|chamar|mulla|mulle|katua|katuwa|jihadi|"
    r"gaumutra|brahman|thakur|gujjar|jaat|yadav|muslim|hindu|"
    r"sikh|christian|islam|mandir|masjid|rss|bjp|congress"
    r")\b"
)
SARCASM_RE = re.compile(
    r"(?i)(\bwah\b|\bgreat\b|\bamazing\b|\bre\b|😂|🤣|😉|👏|bravo|shaabash|shabash)"
)
EN_FUNC = {
    "the", "a", "an", "is", "are", "was", "were", "you", "i", "we", "they",
    "this", "that", "to", "of", "in", "for", "on", "with", "and", "or", "but",
    "not", "have", "has", "had", "be", "been", "it", "my", "your", "our",
    "me", "him", "her", "his", "she", "he", "do", "does", "did", "will",
    "just", "like", "from", "about", "what", "when", "who", "how", "why",
    "can", "if", "so", "as", "at", "by", "up", "out", "all", "no", "yes",
}


def predict_batch(model_dir: Path, texts: list[str], batch_size: int = 64):
    from transformers import AutoTokenizer
    import onnxruntime as ort

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    onnx_path = model_dir / "model_quantized.onnx"
    if not onnx_path.exists():
        onnx_path = model_dir / "onnx" / "model_quantized.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}

    p_abuse = np.zeros(len(texts), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = [t if isinstance(t, str) else "" for t in texts[start : start + batch_size]]
        enc = tok(
            chunk,
            return_tensors="np",
            truncation=True,
            max_length=128,
            padding=True,
        )
        feeds = {}
        for name in input_names:
            if name in enc:
                feeds[name] = enc[name].astype(np.int64)
            elif name == "token_type_ids":
                feeds[name] = np.zeros_like(enc["input_ids"])
        logits = sess.run(None, feeds)[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        p_abuse[start : start + len(chunk)] = p[:, 0]
        if start == 0 or (start // batch_size) % 10 == 0:
            print(f"  scored {min(start + len(chunk), len(texts))}/{len(texts)}", flush=True)
    pred = np.where(p_abuse >= 0.5, 0, 1).astype(int)
    return p_abuse, pred


def english_heavy(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", text or "")
    if len(tokens) < 4:
        return False
    n_en = sum(1 for t in tokens if t.lower() in EN_FUNC)
    return (n_en / len(tokens)) >= 0.35


def bucket_row(row: dict) -> str:
    """Assign a primary error bucket. Priority order matches the plan."""
    label = int(row["label"])
    pred = int(row["pred"])
    hindi = str(row.get("hindi_text") or "")
    hinglish = str(row.get("text") or "")
    hindi_abuse = float(row.get("hindi_abuse") or 0.0)
    hinglish_abuse = float(row.get("p_abuse") or row.get("hinglish_abuse") or 0.0)

    source_wrong = (label == 0 and hindi_abuse < 0.1 and hinglish_abuse < 0.2) or (
        label == 1 and hindi_abuse > 0.8 and hinglish_abuse > 0.8
    )
    if source_wrong:
        return "source_label_wrong"
    if label == 0 and hindi and not lexicon_preserved(hindi, hinglish):
        return "softened_slur"
    if OBFUSC_RE.search(hinglish) or ("." in hinglish and any(
        stem in hindi for stem, _ in ABUSE_STEMS
    )):
        return "obfuscation"
    if CASTE_REL_RE.search(hinglish) or CASTE_REL_RE.search(hindi):
        return "caste_religious"
    if english_heavy(hinglish):
        return "english_heavy_mix"
    if SARCASM_RE.search(hinglish):
        return "sarcasm"
    if label == 0 and pred == 1:
        return "benign_looking_insult"
    if label == 1 and pred == 0:
        return "false_positive_other"
    return "other"


def confusion(y: np.ndarray, p: np.ndarray) -> dict:
    tn = int(((y == 1) & (p == 1)).sum())
    tp = int(((y == 0) & (p == 0)).sum())  # abusive is class 0
    fp = int(((y == 1) & (p == 0)).sum())  # predicted abusive, actually not
    fn = int(((y == 0) & (p == 1)).sum())
    acc = float((y == p).mean()) if len(y) else 0.0
    f1s = []
    for c in (0, 1):
        tpc = int(((p == c) & (y == c)).sum())
        fpc = int(((p == c) & (y != c)).sum())
        fnc = int(((p != c) & (y == c)).sum())
        prec = tpc / (tpc + fpc) if (tpc + fpc) else 0.0
        rec = tpc / (tpc + fnc) if (tpc + fnc) else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return {
        "n": int(len(y)),
        "accuracy": acc,
        "f1_macro": float(np.mean(f1s)),
        "f1_abusive": f1s[0],
        "f1_non_abusive": f1s[1],
        "tp_abusive": tp,
        "tn_non_abusive": tn,
        "fp_false_abuse": fp,
        "fn_missed_abuse": fn,
        "matrix": {"true0_pred0": tp, "true0_pred1": fn, "true1_pred0": fp, "true1_pred1": tn},
    }


def analyze_split(df: pd.DataFrame, p_abuse: np.ndarray, pred: np.ndarray, split: str) -> dict:
    y = df["label"].astype(int).to_numpy()
    out = df.copy()
    out["p_abuse"] = p_abuse
    out["pred"] = pred
    out["error"] = np.where(y == pred, "ok", np.where(y == 0, "fn", "fp"))
    out["borderline"] = ((p_abuse >= 0.4) & (p_abuse <= 0.6)).astype(int)
    buckets = []
    for rec in out.to_dict("records"):
        if rec["error"] == "ok":
            buckets.append("ok")
        else:
            buckets.append(bucket_row(rec))
    out["bucket"] = buckets
    metrics = confusion(y, pred)
    err = out[out["error"] != "ok"]
    counts = Counter(err["bucket"].tolist())
    borderline_n = int(out["borderline"].sum())
    samples = {}
    for kind in ("fn", "fp"):
        part = out[out["error"] == kind]
        take = part.head(40)
        samples[kind] = [
            {
                "row_id": str(r.get("row_id", "")),
                "label": int(r["label"]),
                "pred": int(r["pred"]),
                "p_abuse": round(float(r["p_abuse"]), 4),
                "bucket": r["bucket"],
                "text": str(r.get("text") or "")[:240],
                "hindi_text": str(r.get("hindi_text") or "")[:160],
                "hindi_abuse": float(r.get("hindi_abuse") or 0),
                "hinglish_abuse": float(r.get("hinglish_abuse") or 0),
            }
            for r in take.to_dict("records")
        ]
    bucket_samples = defaultdict(list)
    for r in err.to_dict("records"):
        b = r["bucket"]
        if len(bucket_samples[b]) < 8:
            bucket_samples[b].append(
                {
                    "row_id": str(r.get("row_id", "")),
                    "error": r["error"],
                    "text": str(r.get("text") or "")[:200],
                    "p_abuse": round(float(r["p_abuse"]), 4),
                }
            )
    return {
        "df": out,
        "report": {
            "split": split,
            "metrics": metrics,
            "n_fp": int((out["error"] == "fp").sum()),
            "n_fn": int((out["error"] == "fn").sum()),
            "n_borderline": borderline_n,
            "bucket_counts": dict(counts),
            "samples": samples,
            "bucket_samples": dict(bucket_samples),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorer", type=Path, default=DEFAULT_SCORER)
    p.add_argument("--data", type=Path, default=ROOT / "codemix_hinglish")
    p.add_argument("--out", type=Path, default=ROOT / "codemix_synth")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["val", "train"],
        choices=["train", "val", "test"],
    )
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"scorer={args.scorer}", flush=True)

    reports = {}
    for split in args.splits:
        path = args.data / f"{split}.csv"
        df = pd.read_csv(path)
        texts = df["text"].fillna("").astype(str).tolist()
        print(f"scoring {split} n={len(texts)}", flush=True)
        p_abuse, pred = predict_batch(args.scorer, texts, args.batch_size)
        result = analyze_split(df, p_abuse, pred, split)
        err_cols = [
            c
            for c in (
                "row_id",
                "split",
                "label",
                "pred",
                "p_abuse",
                "error",
                "bucket",
                "borderline",
                "text",
                "hindi_text",
                "hindi_abuse",
                "hinglish_abuse",
                "score_delta",
            )
            if c in result["df"].columns
        ]
        err = result["df"][result["df"]["error"] != "ok"][err_cols]
        err_path = args.out / f"{split}_errors.csv"
        err.to_csv(err_path, index=False)
        result["df"][err_cols].to_csv(args.out / f"{split}_scored.csv", index=False)
        reports[split] = result["report"]
        reports[split]["errors_path"] = str(err_path)
        print(
            f"{split}: acc={result['report']['metrics']['accuracy']:.4f} "
            f"fp={result['report']['n_fp']} fn={result['report']['n_fn']} "
            f"buckets={result['report']['bucket_counts']}",
            flush=True,
        )

    payload = {
        "scorer": str(args.scorer),
        "note": "Frozen Hinglish CSVs. abusive=0, non-abusive=1. pred=0 if p_abuse>=0.5.",
        "splits": reports,
    }
    out_json = args.out / "error_clusters.json"
    samples_json = args.out / "error_samples.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    samples_json.write_text(
        json.dumps(
            {k: {"samples": v.get("samples"), "bucket_samples": v.get("bucket_samples")} for k, v in reports.items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
