"""Eval codemix-trained INT8 on Devanagari Hindi + Hinglish tests."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOL_MOUNT = "/vol"
BUNDLE = f"{VOL_MOUNT}/checkpoints/custom-macd-model-codemix"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "optimum[onnxruntime]==1.22.0",
        "onnxruntime==1.19.2",
        "numpy==1.26.4",
    )
)

vol = modal.Volume.from_name("surakshanet-notebook")
app = modal.App("surakshanet-eval-both", image=image)


@app.function(volumes={VOL_MOUNT: vol}, timeout=1800, memory=8192)
def eval_both(hindi_csv: bytes, hinglish_csv: bytes) -> dict:
    import numpy as np
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer, pipeline as hf_pipeline

    bundle = Path(BUNDLE)
    assert bundle.exists(), bundle

    def load(blob: bytes):
        texts, labels = [], []
        for row in csv.DictReader(io.StringIO(blob.decode("utf-8"))):
            texts.append(row["text"])
            labels.append(int(float(row["label"])))
        return texts, labels

    def metrics(y, p, texts=None):
        y, p = np.array(y), np.array(p)
        acc = float((y == p).mean())
        f1s = []
        for c in [0, 1]:
            tp = int(((p == c) & (y == c)).sum())
            fp = int(((p == c) & (y != c)).sum())
            fn = int(((p != c) & (y == c)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        tp0 = int(((p == 0) & (y == 0)).sum())
        fn0 = int(((p == 1) & (y == 0)).sum())
        fp0 = int(((p == 0) & (y == 1)).sum())
        tn0 = int(((p == 1) & (y == 1)).sum())
        out = {
            "n": int(len(y)),
            "accuracy": acc,
            "f1_macro": float(np.mean(f1s)),
            "confusion": {
                "tp_abusive": tp0,
                "fn_missed_abuse": fn0,
                "fp_false_abuse": fp0,
                "tn_non_abusive": tn0,
            },
        }
        if texts is not None:
            fn_idx = [i for i, (a, b) in enumerate(zip(y, p)) if a == 0 and b == 1][:8]
            fp_idx = [i for i, (a, b) in enumerate(zip(y, p)) if a == 1 and b == 0][:8]
            out["fn_samples"] = [texts[i][:160] for i in fn_idx]
            out["fp_samples"] = [texts[i][:160] for i in fp_idx]
        return out

    tokenizer = AutoTokenizer.from_pretrained(str(bundle))
    model = ORTModelForSequenceClassification.from_pretrained(
        str(bundle), file_name="model_quantized.onnx"
    )
    clf = hf_pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    label2id = {"abusive": 0, "non-abusive": 1}

    out = {}
    for name, blob in [
        ("devanagari_hindi_test", hindi_csv),
        ("hinglish_codemix_test", hinglish_csv),
    ]:
        texts, labels = load(blob)
        preds = clf(texts, batch_size=64, truncation=True)
        pred_ids = [label2id[p["label"]] for p in preds]
        out[name] = metrics(labels, pred_ids, texts)
        out[name]["sample"] = texts[0][:120]
        print(name, out[name], flush=True)
    return out


@app.local_entrypoint()
def main():
    hindi = (ROOT / "aws_train/macd_hindi_cache/hindi_test.csv").read_bytes()
    hinglish = (ROOT / "codemix_hinglish/test.csv").read_bytes()
    print(eval_both.remote(hindi, hinglish))
