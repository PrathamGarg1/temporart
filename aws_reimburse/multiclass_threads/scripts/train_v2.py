#!/usr/bin/env python3
"""Train 5-way MiniLM on dataset/all/*.jsonl. Local only (cost 0).

  python train_v2.py --mode thread   --epochs 2   # last turn + most-recent-first context
  python train_v2.py --mode isolated --epochs 2   # last turn only (control)

Never touches the shipped INT8 model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "dataset" / "all"
ART = HERE / "artifacts_v2"
BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLASSES = ["benign", "friendly_sarcasm", "hostile_abuse", "sexual_harassment", "predatory_grooming"]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def encode(tok, row: dict, mode: str, max_len: int) -> dict:
    last = row["turns"][-1]
    if mode == "isolated":
        return tok(last, truncation=True, max_length=min(max_len, 64))
    ctx = " [TURN] ".join(reversed(row["turns"][:-1]))
    return tok(last, ctx, truncation="only_second", max_length=max_len)


class DS(Dataset):
    def __init__(self, rows, tok, mode, max_len):
        self.items = []
        for r in rows:
            enc = encode(tok, r, mode, max_len)
            self.items.append(
                {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": int(r["label_id"])}
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def metrics(p):
    pred = np.argmax(p.predictions, -1)
    y = p.label_ids
    f1 = []
    for c in range(5):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"macro_f1": float(np.mean(f1)), "accuracy": float((pred == y).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["thread", "isolated"], required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--val-n", type=int, default=6000)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    out = ART / f"model_{args.mode}"
    shipped = HERE.parents[1] / "assets" / "models" / "custom-macd-model"
    if shipped.resolve() == out.resolve():
        raise SystemExit("refusing to overwrite shipped model")

    train = load(DATA / "train.jsonl")
    val = load(DATA / "val.jsonl")
    random.Random(1).shuffle(train)
    random.Random(2).shuffle(val)
    if args.limit_train:
        train = train[: args.limit_train]
    val = val[: args.val_n]
    print(f"mode={args.mode} train={len(train)} val={len(val)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE,
        num_labels=5,
        id2label=dict(enumerate(CLASSES)),
        label2id={c: i for i, c in enumerate(CLASSES)},
        ignore_mismatched_sizes=True,
    )
    # 250k-token embedding matrix makes each MPS step ~3x slower; freeze it
    model.bert.embeddings.word_embeddings.weight.requires_grad = False
    tr = DS(train, tok, args.mode, args.max_len)
    va = DS(val, tok, args.mode, args.max_len)
    lens = [len(x["input_ids"]) for x in tr.items[:2000]]
    print(f"tokens mean={np.mean(lens):.0f} p95={np.percentile(lens, 95):.0f}", flush=True)

    targs = TrainingArguments(
        output_dir=str(ART / f"runs_{args.mode}"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.eval_batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=800,
        save_total_limit=2,
        logging_steps=100,
        group_by_length=True,
        seed=42,
        report_to=[],
        dataloader_num_workers=args.workers,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tr,
        eval_dataset=va,
        data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=64),
        compute_metrics=metrics,
    )
    ck = list((ART / f"runs_{args.mode}").glob("checkpoint-*")) if args.resume else []
    trainer.train(resume_from_checkpoint=True if ck else None)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    (out / "train_config.json").write_text(
        json.dumps({"mode": args.mode, "max_len": args.max_len, "batch": args.batch, "epochs": args.epochs, "lr": args.lr, "n_train": len(train)})
    )
    print(f"saved {out}", flush=True)
    if os.environ.get("NO_LEDGER"):
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cost_ledger import log_call

    log_call(
        job="multiclass_train_v2",
        service="local",
        model="local",
        input_tokens=0,
        output_tokens=0,
        ok=True,
        notes=f"MiniLM 5-way {args.mode} local MPS; n_train={len(train)}",
    )


if __name__ == "__main__":
    main()
