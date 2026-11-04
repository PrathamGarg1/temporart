#!/usr/bin/env python3
"""Train a 5-way MiniLM on frozen thread splits. Does not touch shipped INT8."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parents[1]
ART = HERE / "artifacts"
MODEL_DIR = ART / "model_minilm_5way"
SHIPPED = Path(__file__).resolve().parents[3] / "assets" / "models" / "custom-macd-model"
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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class ThreadDS(Dataset):
    def __init__(self, rows: list[dict], tok, use_thread: bool = True) -> None:
        self.rows = rows
        self.tok = tok
        self.use_thread = use_thread

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        text = pack(r["turns"]) if self.use_thread else r["turns"][-1]
        enc = self.tok(text, truncation=True, max_length=128, padding="max_length")
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(r["label_id"]))
        return item


def metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    acc = float((pred == labels).mean()) if len(labels) else 0.0
    f1s = []
    for c in range(5):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return {"macro_f1": float(np.mean(f1s)), "accuracy": acc}


def main() -> None:
    if not (ART / "FREEZE.json").exists():
        raise SystemExit("freeze missing; generate threads first")
    if SHIPPED.resolve() == MODEL_DIR.resolve():
        raise SystemExit("refusing to overwrite shipped INT8")

    train_rows = load_jsonl(ART / "threads_train.jsonl")
    val_rows = load_jsonl(ART / "threads_val.jsonl")
    tok = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        num_labels=5,
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)},
        ignore_mismatched_sizes=True,
    )
    args = TrainingArguments(
        output_dir=str(ART / "trainer_runs"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
        seed=42,
        report_to=[],
        use_mps_device=torch.backends.mps.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ThreadDS(train_rows, tok, True),
        eval_dataset=ThreadDS(val_rows, tok, True),
        processing_class=tok,
        compute_metrics=metrics,
    )
    trainer.train()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))
    print(f"saved {MODEL_DIR} (shipped INT8 untouched: {SHIPPED})", flush=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cost_ledger import log_call

    log_call(
        job="multiclass_train",
        service="local",
        model="local",
        input_tokens=0,
        output_tokens=0,
        ok=True,
        notes=f"MiniLM 5-way local; n_train={len(train_rows)}",
    )


if __name__ == "__main__":
    main()
