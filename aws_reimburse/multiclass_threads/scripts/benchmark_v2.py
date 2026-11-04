#!/usr/bin/env python3
"""Benchmark isolated vs thread models on dataset/all/test.jsonl.

Reports macro-F1 / accuracy by thread length, script, class; confusion matrix;
and accuracy on same-last-line pairs (friendly_sarcasm vs hostile_abuse).
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v2 import ART, CLASSES, DATA, encode, load  # noqa: E402


def predict(model_dir: Path, rows: list[dict], mode: str) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    cfgp = Path(model_dir) / "train_config.json"
    max_len = json.loads(cfgp.read_text())["max_len"] if cfgp.exists() else (192 if mode == "thread" else 256)
    model.to(dev)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["turns"][-1]) + sum(len(t) for t in rows[i]["turns"]))
    preds = np.zeros(len(rows), dtype=int)
    bs = int(os.environ.get("BENCH_BS", "16"))
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            encs = [encode(tok, rows[i], mode, max_len) for i in idx]
            batch = tok.pad(
                [{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in encs],
                return_tensors="pt",
                pad_to_multiple_of=64,
            )
            logits = model(**{k: v.to(dev) for k, v in batch.items()}).logits
            preds[idx] = logits.argmax(-1).cpu().numpy()
    return preds


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for c in range(5):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(f))


def summarize(rows, y, p) -> dict:
    out = {"n": len(rows), "accuracy": float((y == p).mean()), "macro_f1": macro_f1(y, p)}
    for key in ("length", "script"):
        g = defaultdict(list)
        for i, r in enumerate(rows):
            g[r[key]].append(i)
        out[f"by_{key}"] = {
            str(k): {"n": len(v), "accuracy": float((y[v] == p[v]).mean()), "macro_f1": macro_f1(y[v], p[v])}
            for k, v in sorted(g.items())
        }
    out["per_class_recall"] = {
        c: float((p[y == i] == i).mean()) if (y == i).any() else None for i, c in enumerate(CLASSES)
    }
    cm = np.zeros((5, 5), dtype=int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    out["confusion"] = cm.tolist()
    pair = [i for i, r in enumerate(rows) if r["pair_id"]]
    out["same_last_line_pairs"] = {
        "n_threads": len(pair),
        "accuracy": float((y[pair] == p[pair]).mean()) if pair else None,
    }
    return out


def main() -> None:
    rows = load(DATA / "test.jsonl")
    y = np.array([r["label_id"] for r in rows])
    report = {"n_test": len(rows), "classes": CLASSES, "models": {}}
    for mode in ("isolated", "thread"):
        d = ART / f"model_{mode}"
        if not d.exists():
            continue
        p = predict(d, rows, mode)
        report["models"][mode] = summarize(rows, y, p)
        m = report["models"][mode]
        print(f"{mode}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} pairs={m['same_last_line_pairs']}", flush=True)
    (ART / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", ART / "benchmark_report.json")


if __name__ == "__main__":
    main()
