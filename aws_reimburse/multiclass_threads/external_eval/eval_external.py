#!/usr/bin/env python3
"""Zero-shot check of the synthetic-trained 5-way models on real, human-labelled data.

No fine-tuning on any of these datasets. Models: artifacts_v2/cloud_results/model_{isolated,thread}.

Datasets (all open):
  CAD        English Reddit comment chains, labels annotated in thread context (CC-BY 4.0)
  ConvAbuse  English user-chatbot dialogues, 4 turns, abuse + sex_harassment flags
  PRISM      Hindi (Devanagari) / Hinglish / English single messages (no thread)

Task mapping (coarse, because real data lacks our 5 classes):
  abusive      = hostile_abuse + sexual_harassment probability
  not abusive  = benign + friendly_sarcasm + predatory_grooming probability
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_v2 import CLASSES, encode  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "results"
MODELS = ROOT / "artifacts_v2" / "cloud_results"
ABUSE_IDX = [CLASSES.index("hostile_abuse"), CLASSES.index("sexual_harassment")]
SEX_IDX = CLASSES.index("sexual_harassment")
MAX_CTX = 9  # thread length 10 max, as in training


def clean(t: str) -> str:
    t = t.replace("[linebreak]", " ")
    t = re.sub(r"<span[^>]*>.*?</span>", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------ loaders
def load_cad() -> list[dict]:
    rows = list(csv.DictReader(open(DATA / "cad/data/cad_v1_1.tsv", encoding="utf-8"), delimiter="\t"))
    post_text: dict[str, str] = {}
    comments: dict[str, dict] = {}
    for r in rows:
        iid = r["info_id"]
        if iid.endswith("-title") or iid.endswith("-post"):
            base = iid.rsplit("-", 1)[0]
            post_text[base] = (post_text.get(base, "") + " " + clean(r["meta_text"])).strip()
        else:
            comments[iid] = r
    out = []
    for iid, r in comments.items():
        if r["split"].startswith("exclude"):
            continue
        text = clean(r["meta_text"])
        if not text or text in ("[deleted]", "[removed]"):
            continue
        chain, cur, hops = [text], r["info_id.parent"], 0
        while cur not in ("NA", "") and hops < MAX_CTX:
            hops += 1
            if cur in comments:
                chain.append(clean(comments[cur]["meta_text"]) or "[empty]")
                cur = comments[cur]["info_id.parent"]
            else:
                if post_text.get(cur):
                    chain.append(post_text[cur])
                break
        turns = [t[:900] for t in reversed(chain)]
        y = int(r["annotation_Primary"] in ("IdentityDirectedAbuse", "AffiliationDirectedAbuse", "PersonDirectedAbuse"))
        out.append(
            {"dataset": "CAD", "lang": "english", "turns": turns, "y": y,
             "needs_context": r["annotation_Context"] == "PreviousContent", "ctx_label": r["annotation_Context"], "n_ctx": len(turns) - 1,
             "split": r["split"], "label_raw": r["annotation_Primary"]}
        )
    return out


def load_convabuse() -> list[dict]:
    rows = list(csv.DictReader(open(DATA / "convabuse.csv", encoding="utf-8")))
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        g[(r["conv_id"], r["prev_agent"], r["prev_user"], r["agent"], r["user"])].append(r)
    out = []
    for (cid, pa, pu, ag, us), rs in g.items():
        us = clean(us)
        if not us:
            continue
        n = len(rs)
        ab = sum(int(r["is_abuse.-1"]) or int(r["is_abuse.-2"]) or int(r["is_abuse.-3"]) for r in rs) / n
        not_ab = sum(int(r["is_abuse.1"]) for r in rs) / n
        if 0.25 < ab < 0.5 or (ab < 0.5 and not_ab < 0.5):
            continue  # annotators disagree / ambiguous: drop
        turns = [clean(x) for x in (pa, pu, ag) if x and x.strip() != "_"] + [us]
        sex = sum(int(r["type.sex_harassment"]) for r in rs) / n
        out.append(
            {"dataset": "ConvAbuse", "lang": "english", "turns": turns, "y": int(ab >= 0.5),
             "y_sex": int(sex >= 0.5), "n_ctx": len(turns) - 1, "needs_context": False,
             "bot": rs[0]["bot"]}
        )
    return out


def load_prism() -> list[dict]:
    rows = list(csv.DictReader(open(DATA / "prism_cleaned.csv", encoding="utf-8")))
    out = []
    for r in rows:
        t = clean(r["text"])
        if len(t) < 3:
            continue
        out.append({"dataset": "PRISM", "lang": r["language"], "turns": [t], "y": int(r["hate_label"]),
                    "n_ctx": 0, "needs_context": False, "source": r["source"]})
    return out


# ------------------------------------------------------------------ inference
def enc_safe(tok, row: dict, mode: str, max_len: int) -> dict:
    """Same input format as training, but never raises on long real-world text."""
    turns = row["turns"]
    if mode == "isolated":
        return tok(turns[-1], truncation=True, max_length=min(max_len, 64))
    last = tok(turns[-1], add_special_tokens=False, truncation=True, max_length=128)["input_ids"]
    ctx_text = " [TURN] ".join(reversed(turns[:-1]))
    ctx = tok(ctx_text, add_special_tokens=False)["input_ids"][: max(max_len - len(last) - 4, 0)] if ctx_text else []
    ids = tok.build_inputs_with_special_tokens(last, ctx) if ctx else tok.build_inputs_with_special_tokens(last)
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def predict_probs(model_dir: Path, rows: list[dict], mode: str, tag: str) -> np.ndarray:
    cache = OUT / f"probs_{tag}_{mode}.npy"
    if cache.exists():
        p = np.load(cache)
        if len(p) == len(rows):
            return p
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)
    max_len = json.loads((model_dir / "train_config.json").read_text())["max_len"]
    work = [{"turns": r["turns"][-1:] if mode == "isolated" else r["turns"]} for r in rows]
    order = sorted(range(len(work)), key=lambda i: sum(len(t) for t in work[i]["turns"]))
    probs = np.zeros((len(rows), 5), dtype=np.float32)
    bs = 32
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            encs = [enc_safe(tok, work[i], mode, max_len) for i in idx]
            batch = tok.pad(
                [{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in encs],
                return_tensors="pt", pad_to_multiple_of=64,
            )
            logits = model(**{k: v.to(dev) for k, v in batch.items()}).logits
            probs[idx] = torch.softmax(logits, -1).cpu().numpy()
            if (s // bs) % 50 == 0:
                print(f"  {tag}/{mode}: {s}/{len(order)}", flush=True)
    OUT.mkdir(exist_ok=True)
    np.save(cache, probs)
    return probs


# ------------------------------------------------------------------ metrics
def auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def prf(y: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0}


def best_thresh_f1(y: np.ndarray, s: np.ndarray, seed: int = 0) -> dict:
    """Choose the threshold on one random half, report F1 on the other (no peeking)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    a, b = idx[: len(y) // 2], idx[len(y) // 2 :]
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 49):
        f = prf(y[a], (s[a] >= t).astype(int))["f1"]
        if f > best_f:
            best_t, best_f = t, f
    return {"threshold": float(best_t), **prf(y[b], (s[b] >= best_t).astype(int))}


def evaluate(rows: list[dict], probs: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        rows = [r for r, m in zip(rows, mask) if m]
        probs = probs[mask]
    y = np.array([r["y"] for r in rows])
    s = probs[:, ABUSE_IDX].sum(1)
    pred = (probs.argmax(1) in ABUSE_IDX) if False else np.isin(probs.argmax(1), ABUSE_IDX).astype(int)
    return {"n": len(y), "abusive_rate": float(y.mean()) if len(y) else None, "auroc": auroc(y, s),
            "argmax": prf(y, pred), "tuned_half_split": best_thresh_f1(y, s) if len(y) > 40 else None}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    random.seed(0)
    sets = {"CAD": load_cad(), "ConvAbuse": load_convabuse(), "PRISM": load_prism()}
    for k, v in sets.items():
        print(k, len(v), Counter(r["y"] for r in v), flush=True)
    report: dict = {"note": "zero-shot; abusive = hostile_abuse + sexual_harassment probability", "datasets": {}}

    for name, rows in sets.items():
        res: dict = {"n": len(rows), "abusive_rate": float(np.mean([r["y"] for r in rows]))}
        modes = ["isolated"] if name == "PRISM" else ["isolated", "thread"]
        P = {m: predict_probs(MODELS / f"model_{m}", rows, m, name) for m in modes}
        for m in modes:
            res[m] = {"all": evaluate(rows, P[m])}
        if name == "PRISM":
            for lang in ("hindi", "hinglish", "english"):
                mask = np.array([r["lang"] == lang for r in rows])
                res["isolated"][lang] = evaluate(rows, P["isolated"], mask)
        if name == "CAD":
            for lab, mask in {
                "abuse_visible_only_with_previous_msgs_vs_all_neutral": np.array([r["needs_context"] or r["y"] == 0 for r in rows]),
                "abuse_visible_in_current_msg_vs_all_neutral": np.array([r["ctx_label"] == "CurrentContent" or r["y"] == 0 for r in rows]),
                "has_parent_context": np.array([r["n_ctx"] >= 1 for r in rows]),
                "ctx_1": np.array([r["n_ctx"] == 1 for r in rows]),
                "ctx_2_3": np.array([2 <= r["n_ctx"] <= 3 for r in rows]),
                "ctx_4_plus": np.array([r["n_ctx"] >= 4 for r in rows]),
                "test_split_only": np.array([r["split"] == "test" for r in rows]),
            }.items():
                for m in modes:
                    res[m][lab] = evaluate(rows, P[m], mask)
        if name == "ConvAbuse":
            ysex = np.array([r["y_sex"] for r in rows])
            for m in modes:
                s = P[m][:, SEX_IDX]
                pos = int(ysex.sum())
                res[m]["sex_harassment"] = {
                    "n_positive": pos,
                    "auroc_p_sexual_harassment": auroc(ysex, s),
                    "argmax_sexual_harassment": prf(ysex, (P[m].argmax(1) == SEX_IDX).astype(int)),
                }
        report["datasets"][name] = res
        # keep per-row predictions for inspection
        with (OUT / f"preds_{name}.jsonl").open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                rec = {"turns": r["turns"][-3:], "y": r["y"], "n_ctx": r["n_ctx"], "lang": r["lang"]}
                for m in modes:
                    rec[f"p_{m}"] = [round(float(x), 4) for x in P[m][i]]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (OUT / "external_report.json").write_text(json.dumps(report, indent=2))
        print(f"== {name} done", flush=True)
    print("wrote", OUT / "external_report.json", flush=True)


if __name__ == "__main__":
    main()
