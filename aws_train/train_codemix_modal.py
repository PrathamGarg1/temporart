"""Train SurakshaNet ablations: baseline / layer1 / layer1_synth.

Mixes:
  baseline      = davidson_80 + original hinglish_train          (shipped v1)
  layer1        = hindi + davidson_80 + cleaned hinglish_train
  layer1_synth  = layer1 + synth_accepted (synth down-weighted)

Val/test hinglish CSVs stay frozen originals when using layer1* (copied into
codemix_hinglish_clean). Synth never enters val/test.

Usage:
  modal run aws_train/train_codemix_modal.py --mix layer1_synth
  modal run aws_train/train_codemix_modal.py --all-ablations
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

APP_NAME = "surakshanet-codemix-train"
VOL_NAME = "surakshanet-notebook"
VOL_MOUNT = "/vol"
HINDI_REMOTE = f"{VOL_MOUNT}/checkpoints/macd_hindi"
SYNTH_REMOTE = f"{VOL_MOUNT}/checkpoints/codemix_synth"

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CODEMIX = ROOT / "codemix_hinglish"
LOCAL_CLEAN = ROOT / "codemix_hinglish_clean"
LOCAL_HINDI = ROOT / "aws_train" / "macd_hindi_cache"
LOCAL_SYNTH = ROOT / "codemix_synth"
LOCAL_ASSETS = ROOT / "assets" / "models" / "custom-macd-model"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42
ID2LABEL = {0: "abusive", 1: "non-abusive"}
LABEL2ID = {"abusive": 0, "non-abusive": 1}

MIX_CONFIG = {
    "baseline": {
        "use_hindi": False,
        "codemix_key": "codemix_hinglish",
        "use_synth": False,
        "synth_weight": 1.0,
    },
    "layer1": {
        "use_hindi": True,
        "codemix_key": "codemix_hinglish_clean",
        "use_synth": False,
        "synth_weight": 1.0,
    },
    "layer1_synth": {
        "use_hindi": True,
        "codemix_key": "codemix_hinglish_clean",
        "use_synth": True,
        "synth_weight": 0.5,
    },
}


def _paths(mix: str) -> dict[str, str]:
    return {
        "pt": f"{VOL_MOUNT}/checkpoints/pt_codemix_{mix}",
        "onnx_fp32": f"{VOL_MOUNT}/checkpoints/onnx_fp32_codemix_{mix}",
        "onnx_int8": f"{VOL_MOUNT}/checkpoints/onnx_int8_codemix_{mix}",
        "bundle": f"{VOL_MOUNT}/checkpoints/custom-macd-model-codemix-{mix}",
        "codemix": f"{VOL_MOUNT}/checkpoints/{MIX_CONFIG[mix]['codemix_key']}",
    }

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "datasets==2.21.0",
        "accelerate==0.33.0",
        "scikit-learn==1.5.1",
        "sentencepiece==0.2.0",
        "numpy==1.26.4",
        "optimum[onnxruntime]==1.22.0",
        "onnx==1.16.2",
        "onnxruntime==1.19.2",
    )
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=False)
app = modal.App(APP_NAME, image=image)


@app.function(volumes={VOL_MOUNT: vol}, timeout=600)
def put_data(files: dict[str, bytes]) -> str:
    """files keys like 'codemix/train.csv' or 'hindi/hindi_train.csv'."""
    for rel, blob in files.items():
        dest = Path(VOL_MOUNT) / "checkpoints" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        print(f"wrote {dest} ({len(blob)} bytes)")
    vol.commit()
    return f"uploaded {len(files)} files"


@app.function(
    gpu="T4",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60 * 4,
    memory=16384,
)
def train(mix: str = "layer1_synth") -> dict:
    import os

    import numpy as np
    import torch
    from datasets import concatenate_datasets, load_dataset
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        pipeline as hf_pipeline,
    )

    if mix not in MIX_CONFIG:
        raise ValueError(f"unknown mix {mix}; choose {list(MIX_CONFIG)}")
    cfg = MIX_CONFIG[mix]
    paths = _paths(mix)
    PT_DIR = paths["pt"]
    ONNX_FP32_DIR = paths["onnx_fp32"]
    ONNX_INT8_DIR = paths["onnx_int8"]
    BUNDLE_DIR = paths["bundle"]
    CODEMIX_REMOTE = paths["codemix"]
    print("mix:", mix, "cfg:", cfg, flush=True)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("weight", None)
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            if weights is not None:
                loss = (loss * weights.view(-1)).mean()
            else:
                loss = loss.mean()
            return (loss, outputs) if return_outputs else loss

    def confusion_dict(y, p):
        y, p = list(y), list(p)
        tp = sum(1 for a, b in zip(y, p) if a == 0 and b == 0)
        fn = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
        fp = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
        tn = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
        return {
            "tp_abusive": tp,
            "fn_missed_abuse": fn,
            "fp_false_abuse": fp,
            "tn_non_abusive": tn,
        }

    print(
        "cuda:",
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

    def keep_text_label(ds, extra=None):
        keep = {"text", "label"} | set(extra or [])
        drop = [c for c in ds.column_names if c not in keep]
        return ds.remove_columns(drop) if drop else ds

    def load_csv(path: str, extra=None):
        ds = load_dataset("csv", data_files=path)["train"]
        return keep_text_label(ds, extra=extra)

    def add_weight(ds, w: float):
        return ds.map(lambda _: {"weight": float(w)})

    hindi_train = hindi_val = hindi_test = None
    if cfg["use_hindi"]:
        hindi_train = add_weight(load_csv(f"{HINDI_REMOTE}/hindi_train.csv"), 1.0)
        hindi_val = load_csv(f"{HINDI_REMOTE}/hindi_val.csv")
        hindi_test = load_csv(f"{HINDI_REMOTE}/hindi_test.csv")
        print("hindi sizes:", len(hindi_train), len(hindi_val), len(hindi_test))

    hinglish_train = add_weight(load_csv(f"{CODEMIX_REMOTE}/train.csv"), 1.0)
    hinglish_val = load_csv(f"{CODEMIX_REMOTE}/val.csv")
    hinglish_test = load_csv(f"{CODEMIX_REMOTE}/test.csv")
    print("hinglish sizes:", len(hinglish_train), len(hinglish_val), len(hinglish_test))
    print("hinglish sample:", hinglish_train[0])

    synth_train = None
    synth_path = f"{SYNTH_REMOTE}/accepted.csv"
    if cfg["use_synth"] and Path(synth_path).exists():
        synth_train = add_weight(load_csv(synth_path), float(cfg["synth_weight"]))
        synth_train = synth_train.map(
            lambda ex: {"text": ex["text"] if isinstance(ex["text"], str) else str(ex["text"] or "")}
        )
        print("synth size:", len(synth_train), "weight", cfg["synth_weight"])
    elif cfg["use_synth"]:
        print("WARNING: synth accepted.csv missing; training without synth")

    # --- Davidson EN: 80% train / 10% val / 10% test ---
    davidson = load_dataset(
        "csv",
        data_files=(
            "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/"
            "master/data/labeled_data.csv"
        ),
    )["train"]

    def convert_labels(example):
        return {"label": 0 if example["class"] in [0, 1] else 1}

    davidson = davidson.map(convert_labels)
    davidson = davidson.rename_column("tweet", "text")
    davidson = keep_text_label(davidson)

    # First carve out 20% for val+test, then split that 50/50 -> 10%/10%
    dav_split = davidson.train_test_split(test_size=0.20, seed=SEED)
    davidson_train = add_weight(dav_split["train"], 1.0)
    dav_hold = dav_split["test"].train_test_split(test_size=0.50, seed=SEED)
    davidson_val = dav_hold["train"]  # 10%
    davidson_test = dav_hold["test"]  # 10%
    print(
        "davidson sizes:",
        len(davidson_train),
        len(davidson_val),
        len(davidson_test),
    )

    train_parts = [davidson_train, hinglish_train]
    val_parts = [davidson_val, hinglish_val]
    test_parts = [davidson_test, hinglish_test]
    if hindi_train is not None:
        train_parts = [hindi_train] + train_parts
        val_parts = [hindi_val] + val_parts
        test_parts = [hindi_test] + test_parts
    if synth_train is not None:
        train_parts = train_parts + [synth_train]

    train_ds = concatenate_datasets(train_parts).shuffle(seed=SEED)
    val_ds = concatenate_datasets(val_parts).shuffle(seed=SEED)
    test_ds = concatenate_datasets(test_parts).shuffle(seed=SEED)

    print("MIXED sizes train/val/test:", len(train_ds), len(val_ds), len(test_ds))
    print(
        "  train hindi",
        0 if hindi_train is None else len(hindi_train),
        "dav",
        len(davidson_train),
        "hinglish",
        len(hinglish_train),
        "synth",
        0 if synth_train is None else len(synth_train),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_inputs = set(tokenizer.model_input_names) | {"label", "labels", "weight"}

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    def drop_raw(ds):
        drop = [c for c in ds.column_names if c not in model_inputs]
        return ds.remove_columns(drop) if drop else ds

    tokenized_train = drop_raw(train_ds.map(tokenize_function, batched=True))
    tokenized_val = drop_raw(val_ds.map(tokenize_function, batched=True))
    tokenized_test = drop_raw(test_ds.map(tokenize_function, batched=True))

    tok_dav_val = drop_raw(davidson_val.map(tokenize_function, batched=True))
    tok_hing_val = drop_raw(hinglish_val.map(tokenize_function, batched=True))
    tok_dav_test = drop_raw(davidson_test.map(tokenize_function, batched=True))
    tok_hing_test = drop_raw(hinglish_test.map(tokenize_function, batched=True))
    tok_hindi_val = (
        drop_raw(hindi_val.map(tokenize_function, batched=True)) if hindi_val is not None else None
    )
    tok_hindi_test = (
        drop_raw(hindi_test.map(tokenize_function, batched=True)) if hindi_test is not None else None
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_predictions):
        logits, labels = eval_predictions
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro"),
            "precision_macro": precision_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, predictions, average="macro", zero_division=0
            ),
            **confusion_dict(labels, predictions),
        }

    training_args = TrainingArguments(
        output_dir=f"{PT_DIR}/trainer_runs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    print("train:", train_result.metrics)
    val_mixed = trainer.evaluate(eval_dataset=tokenized_val)
    print("val_mixed:", val_mixed)

    os.makedirs(PT_DIR, exist_ok=True)
    trainer.save_model(PT_DIR)
    tokenizer.save_pretrained(PT_DIR)

    slice_metrics = {
        "val_davidson": trainer.predict(tok_dav_val).metrics,
        "val_hinglish": trainer.predict(tok_hing_val).metrics,
        "test_mixed": trainer.predict(tokenized_test).metrics,
        "test_davidson": trainer.predict(tok_dav_test).metrics,
        "test_hinglish": trainer.predict(tok_hing_test).metrics,
    }
    if tok_hindi_val is not None:
        slice_metrics["val_hindi"] = trainer.predict(tok_hindi_val).metrics
        slice_metrics["test_hindi"] = trainer.predict(tok_hindi_test).metrics
    for k, v in slice_metrics.items():
        print(k, v)

    metrics = {
        "mix": mix,
        "protocol": {
            "train": (
                ("hindi_train + " if cfg["use_hindi"] else "")
                + "davidson_80 + hinglish_train"
                + (" + synth" if synth_train is not None else "")
            ),
            "val": ("hindi_val + " if cfg["use_hindi"] else "")
            + "davidson_10 + hinglish_val (frozen hinglish)",
            "test": ("hindi_test + " if cfg["use_hindi"] else "")
            + "davidson_10 + hinglish_test (FROZEN)",
            "synth_weight": cfg["synth_weight"] if synth_train is not None else None,
        },
        "sizes": {
            "train": {
                "hindi": 0 if hindi_train is None else len(hindi_train),
                "davidson": len(davidson_train),
                "hinglish": len(hinglish_train),
                "synth": 0 if synth_train is None else len(synth_train),
                "total": len(train_ds),
            },
            "val": {
                "hindi": 0 if hindi_val is None else len(hindi_val),
                "davidson": len(davidson_val),
                "hinglish": len(hinglish_val),
                "total": len(val_ds),
            },
            "test": {
                "hindi": 0 if hindi_test is None else len(hindi_test),
                "davidson": len(davidson_test),
                "hinglish": len(hinglish_test),
                "total": len(test_ds),
            },
        },
        "train": train_result.metrics,
        "val_mixed": val_mixed,
        **slice_metrics,
    }
    Path(PT_DIR, "metrics.json").write_text(json.dumps(metrics, indent=2))

    for d in (ONNX_FP32_DIR, ONNX_INT8_DIR, BUNDLE_DIR):
        if Path(d).exists():
            shutil.rmtree(d)
        Path(d).mkdir(parents=True, exist_ok=True)

    ort_model = ORTModelForSequenceClassification.from_pretrained(PT_DIR, export=True)
    ort_model.save_pretrained(ONNX_FP32_DIR)
    tokenizer.save_pretrained(ONNX_FP32_DIR)

    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    quantizer.quantize(save_dir=ONNX_INT8_DIR, quantization_config=qconfig)
    tokenizer.save_pretrained(ONNX_INT8_DIR)

    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "ort_config.json",
        "model_quantized.onnx",
    ):
        src = Path(ONNX_INT8_DIR) / name
        if src.exists():
            shutil.copy2(src, Path(BUNDLE_DIR) / name)
    onnx_sub = Path(BUNDLE_DIR) / "onnx"
    onnx_sub.mkdir(exist_ok=True)
    shutil.copy2(
        Path(ONNX_INT8_DIR) / "model_quantized.onnx",
        onnx_sub / "model_quantized.onnx",
    )

    # INT8 sanity on mixed val + per-slice
    int8_model = ORTModelForSequenceClassification.from_pretrained(
        ONNX_INT8_DIR, file_name="model_quantized.onnx"
    )
    classifier = hf_pipeline(
        "text-classification",
        model=int8_model,
        tokenizer=tokenizer,
        device=-1,
    )

    def int8_eval(name, texts, labels):
        preds = classifier(list(texts), batch_size=64, truncation=True)
        pred_ids = [LABEL2ID[p["label"]] for p in preds]
        y = list(labels)
        m = {
            "accuracy": float(accuracy_score(y, pred_ids)),
            "f1_macro": float(f1_score(y, pred_ids, average="macro")),
            "n": len(y),
            **confusion_dict(y, pred_ids),
        }
        print(f"INT8 {name}:", m)
        return m

    metrics["int8"] = {
        "val_mixed": int8_eval("val_mixed", val_ds["text"], val_ds["label"]),
        "val_hinglish": int8_eval(
            "val_hinglish", hinglish_val["text"], hinglish_val["label"]
        ),
        "test_hinglish": int8_eval(
            "test_hinglish", hinglish_test["text"], hinglish_test["label"]
        ),
        "test_davidson": int8_eval(
            "test_davidson", davidson_test["text"], davidson_test["label"]
        ),
        "test_mixed": int8_eval("test_mixed", test_ds["text"], test_ds["label"]),
    }
    if hindi_val is not None:
        metrics["int8"]["val_hindi"] = int8_eval(
            "val_hindi", hindi_val["text"], hindi_val["label"]
        )
        metrics["int8"]["test_hindi"] = int8_eval(
            "test_hindi", hindi_test["text"], hindi_test["label"]
        )

    Path(PT_DIR, "metrics.json").write_text(json.dumps(metrics, indent=2))
    Path(BUNDLE_DIR, "metrics.json").write_text(json.dumps(metrics, indent=2))

    vol.commit()
    return metrics


@app.function(volumes={VOL_MOUNT: vol}, timeout=600)
def fetch_bundle(mix: str = "layer1_synth") -> dict[str, bytes]:
    root = Path(_paths(mix)["bundle"])
    assert root.exists(), f"missing {root} — run train first"
    out: dict[str, bytes] = {}
    for p in root.rglob("*"):
        if p.is_file():
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


def _collect_local_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for name in ("train.csv", "val.csv", "test.csv"):
        files[f"codemix_hinglish/{name}"] = (LOCAL_CODEMIX / name).read_bytes()
        if (LOCAL_CLEAN / name).exists():
            files[f"codemix_hinglish_clean/{name}"] = (LOCAL_CLEAN / name).read_bytes()
    for name in ("hindi_train.csv", "hindi_val.csv", "hindi_test.csv"):
        files[f"macd_hindi/{name}"] = (LOCAL_HINDI / name).read_bytes()
    synth_acc = LOCAL_SYNTH / "accepted.csv"
    if synth_acc.exists():
        files["codemix_synth/accepted.csv"] = synth_acc.read_bytes()
    return files


def _write_bundle(remote_files: dict[str, bytes], dest_dir: Path) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    for rel, blob in remote_files.items():
        dest = dest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
    print(f"wrote bundle {dest_dir} ({len(remote_files)} files)")


def _int8_hinglish(metrics: dict) -> float:
    acc = (
        metrics.get("int8", {})
        .get("test_hinglish", {})
        .get("accuracy", metrics.get("test_hinglish", {}).get("eval_accuracy", 0))
    )
    return float(acc or 0)


def _current_extension_hinglish() -> float:
    p = LOCAL_ASSETS / "metrics.json"
    if not p.exists():
        return 0.0
    try:
        return _int8_hinglish(json.loads(p.read_text()))
    except Exception:
        return 0.0


@app.local_entrypoint()
def main(mix: str = "layer1_synth", all_ablations: bool = False):
    assert LOCAL_CODEMIX.exists(), f"missing {LOCAL_CODEMIX}"
    assert LOCAL_HINDI.exists(), f"missing {LOCAL_HINDI}"
    mixes = list(MIX_CONFIG) if all_ablations else [mix]
    for m in mixes:
        if MIX_CONFIG[m]["codemix_key"] == "codemix_hinglish_clean":
            assert LOCAL_CLEAN.exists(), f"missing {LOCAL_CLEAN} — run layer1_clean_codemix.py"

    files = _collect_local_files()
    for rel, blob in files.items():
        print(f"local {rel}: {len(blob)} bytes")
    print(put_data.remote(files))

    all_metrics = {}
    best_mix = mixes[0]
    best_acc = -1.0
    for m in mixes:
        print(f"===== train mix={m} =====", flush=True)
        metrics = train.remote(m)
        all_metrics[m] = metrics
        print("METRICS:", json.dumps(metrics, indent=2)[:4000])
        acc = _int8_hinglish(metrics)
        if acc >= best_acc:
            best_acc = acc
            best_mix = m

    report_path = LOCAL_SYNTH / "ablation_metrics.json"
    LOCAL_SYNTH.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"wrote {report_path} best_mix={best_mix} hinglish_int8={best_acc}")

    remote_files = fetch_bundle.remote(best_mix)
    mix_dir = LOCAL_ASSETS.parent / f"custom-macd-model-{best_mix}"
    _write_bundle(remote_files, mix_dir)
    shipped = _current_extension_hinglish()
    print(f"compare INT8 hinglish test: new={best_acc:.4f} extension={shipped:.4f}")
    if best_acc > shipped + 1e-6:
        backup = LOCAL_ASSETS.parent / "custom-macd-model-prev"
        if LOCAL_ASSETS.exists() and not backup.exists():
            shutil.copytree(LOCAL_ASSETS, backup)
        _write_bundle(remote_files, LOCAL_ASSETS)
        print(f"PROMOTED {best_mix} -> {LOCAL_ASSETS}")
    else:
        print(f"NOT promoted ({best_mix} did not beat extension Hinglish INT8)")
