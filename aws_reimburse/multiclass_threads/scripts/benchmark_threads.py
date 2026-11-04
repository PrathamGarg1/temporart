#!/usr/bin/env python3
"""Score isolated last-turn vs full thread for k=3,5,7,10 on frozen test."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
ART = HERE / "artifacts"
MODEL_DIR = ART / "model_minilm_5way"
REPORT = ART / "multiclass_thread_report.json"
CLASSES = [
    "benign",
    "friendly_sarcasm",
    "hostile_abuse",
    "sexual_harassment",
    "predatory_grooming",
]


def pack(turns: list[str]) -> str:
    body = " [TURN] ".join(turns[:-1])
    return f"{body} [LAST] {turns[-1]}" if body else f"[LAST] {turns[-1]}"


def load_test() -> list[dict]:
    rows = []
    for line in (ART / "threads_test.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def f1_scores(y: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(CLASSES):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[name] = round(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0, 4)
    macro = float(np.mean(list(out.values()))) if out else 0.0
    acc = float((pred == y).mean()) if len(y) else 0.0
    return {"accuracy": round(acc, 4), "macro_f1": round(macro, 4), "per_class_f1": out}


def predict(model, tok, texts: list[str]) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            chunk = texts[i : i + 32]
            enc = tok(chunk, return_tensors="pt", truncation=True, max_length=128, padding=True)
            logits = model(**enc).logits.cpu().numpy()
            preds.append(np.argmax(logits, axis=-1))
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def main() -> None:
    rows = load_test()
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    y = np.array([int(r["label_id"]) for r in rows])
    last_pred = predict(model, tok, [r["turns"][-1] for r in rows])
    thread_pred = predict(model, tok, [pack(r["turns"]) for r in rows])

    report = {
        "n_test": len(rows),
        "isolated_last_turn": f1_scores(y, last_pred),
        "full_thread": f1_scores(y, thread_pred),
        "by_length": {},
    }
    by_k: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_k[int(r["length"])].append(i)
    for k in sorted(by_k):
        idx = np.array(by_k[k])
        report["by_length"][str(k)] = {
            "n": int(len(idx)),
            "isolated": f1_scores(y[idx], last_pred[idx]),
            "thread": f1_scores(y[idx], thread_pred[idx]),
        }

    # Same last line, sarcasm vs abuse
    pair = defaultdict(dict)
    for i, r in enumerate(rows):
        pid = r.get("pair_id") or ""
        if pid and r["label"] in ("friendly_sarcasm", "hostile_abuse"):
            pair[pid][r["label"]] = i
    both = [v for v in pair.values() if "friendly_sarcasm" in v and "hostile_abuse" in v]
    if both:
        iso_ok = 0
        thr_ok = 0
        for v in both:
            a, b = v["friendly_sarcasm"], v["hostile_abuse"]
            iso_ok += int(last_pred[a] == 1 and last_pred[b] == 2)
            thr_ok += int(thread_pred[a] == 1 and thread_pred[b] == 2)
        report["same_last_line_sarcasm_vs_abuse"] = {
            "n_pairs": len(both),
            "isolated_both_correct": iso_ok / len(both),
            "thread_both_correct": thr_ok / len(both),
        }

    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
