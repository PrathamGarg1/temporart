#!/usr/bin/env python3
"""Seeded hard-aug for Hinglish abuse detection (AugGPT + error-targeted).

Seeds: train FN/FP, val FN/FP (paraphrased only), rescued/original quarantine.
Variants per seed (label copied unless contrastive):
  paraphrase | mix_hindi_heavy | mix_english_heavy | obfuscation | contrastive
Critic: hard gates, cosine-to-seed, Qwen judge, dedup vs train/val/test.
Cap ~5k accepted. Never writes to frozen test.csv.

Usage:
  .venv_codemix/bin/python aws_train/synth_hard_aug_bedrock.py --limit-seeds 50
  .venv_codemix/bin/python aws_train/synth_hard_aug_bedrock.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from codemix_convert_bedrock import (  # noqa: E402
    DEFAULT_SCORER,
    AbuseScorer,
    BedrockClient,
    clean_output,
    hard_checks,
    lexicon_preserved,
)
from layer1_clean_codemix import (  # noqa: E402
    JUDGE_SYSTEM,
    _judge_call,
    norm_text,
    parse_judge,
)

VARIANT_TYPES_ABUSE = (
    "paraphrase",
    "mix_english_heavy",
    "obfuscation",
)
VARIANT_TYPES_SAFE = (
    "paraphrase",
    "mix_english_heavy",
    "mix_hindi_heavy",
)

GEN_SYSTEM = """You rewrite an existing Hinglish social-media comment for an NLP abuse-detection dataset.

Rules:
- Output Roman script only (Latin letters). No Devanagari.
- Keep the SAME meaning, insult target, and intensity. Do NOT add new victims or new hate.
- Do NOT refuse, moralize, censor, or soften. If the seed is vulgar, stay vulgar.
- Do not add quotes, labels, or commentary.
- Output ONLY the rewritten line.
"""

VARIANT_INSTRUCTIONS = {
    "paraphrase": (
        "Paraphrase into different natural WhatsApp Hinglish. Same meaning. "
        "You may add typos or different English insertions."
    ),
    "mix_english_heavy": (
        "Rewrite with MORE English words mixed in, still natural Indian chat. Same meaning."
    ),
    "mix_hindi_heavy": (
        "Rewrite with MORE romanized Hindi and fewer English words. Same meaning."
    ),
    "obfuscation": (
        "Keep every abusive/vulgar word but obfuscate spelling the way users evade filters "
        "(example styles: lu.nd, g@nd, ch*tiya, m.c.). Meaning and intensity unchanged."
    ),
    "contrastive": (
        "Make the SMALLEST edit that flips the comment from abusive to a non-abusive complaint "
        "OR from non-abusive to clearly abusive, depending on the seed label. "
        "Keep topic and named entities. Output ONLY the flipped line."
    ),
}


class Embedder:
    """MiniLM cosine when torch is available; char n-gram cosine otherwise."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.backend = "ngram"
        self.torch = None
        self.tok = None
        self.model = None
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.torch = torch
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.backend = "minilm"
            print("embedder=minilm", flush=True)
        except Exception as e:
            print(f"embedder=ngram ({type(e).__name__}: {e})", flush=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend != "minilm":
            raise RuntimeError("ngram backend has no encode()")
        torch = self.torch
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1)
            summed = (out.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1)
            vec = summed / counts
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.cpu().numpy()

    def cosine(self, a: str, b: str) -> float:
        if self.backend == "minilm":
            v = self.encode([a or "", b or ""])
            return float(np.dot(v[0], v[1]))
        return _ngram_cosine(a or "", b or "")


def _ngram_cosine(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> dict[str, int]:
        s = f" {s.lower()} "
        c: dict[str, int] = {}
        for i in range(max(0, len(s) - n + 1)):
            g = s[i : i + n]
            c[g] = c.get(g, 0) + 1
        return c

    ca, cb = grams(a), grams(b)
    if not ca or not cb:
        return 0.0
    dot = sum(ca[k] * cb[k] for k in ca if k in cb)
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def load_seeds(synth_dir: Path, src: Path, max_per_bucket: int, rng: random.Random) -> pd.DataFrame:
    frames = []
    for split in ("train", "val"):
        path = synth_dir / f"{split}_errors.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["seed_origin"] = f"{split}_error"
        frames.append(df)
    qpath = src / "train_quarantine.csv"
    if qpath.exists():
        q = pd.read_csv(qpath)
        q["text"] = q["last_output"].fillna("").astype(str)
        q["seed_origin"] = "train_quarantine"
        q["error"] = "quarantine"
        q["bucket"] = q["fail_reason"].fillna("quarantine")
        frames.append(q)
    if not frames:
        raise SystemExit("no seeds — run error_analysis_codemix.py first")
    all_df = pd.concat(frames, ignore_index=True)
    all_df["label"] = all_df["label"].astype(int)
    all_df["hindi_text"] = all_df.get("hindi_text", pd.Series([""] * len(all_df))).fillna("").astype(str)
    all_df["text"] = all_df["text"].fillna("").astype(str)
    # Prefer rows that still have Hindi (quarantine last_output may be empty).
    parts = []
    for origin, grp in all_df.groupby("seed_origin"):
        take = grp
        if max_per_bucket and len(take) > max_per_bucket:
            take = take.sample(n=max_per_bucket, random_state=42)
        parts.append(take)
        print(f"seeds {origin}: {len(take)}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return out


def variants_for(label: int, include_contrastive: bool) -> list[str]:
    kinds = list(VARIANT_TYPES_ABUSE if label == 0 else VARIANT_TYPES_SAFE)
    if include_contrastive:
        kinds = kinds + ["contrastive"]
    return kinds[:3] if not include_contrastive else kinds[:4]


def gen_prompt(seed_text: str, hindi: str, kind: str, label: int) -> str:
    lab = "abusive" if label == 0 else "non-abusive"
    return (
        f"{VARIANT_INSTRUCTIONS[kind]}\n"
        f"Seed label: {lab} ({label})\n"
        f"Hinglish seed: {seed_text}\n"
        f"Hindi source: {hindi}\n"
        f"Rewrite:"
    )


def generate_one(client: BedrockClient, prompt: str) -> str:
    return clean_output(
        client.complete(GEN_SYSTEM, prompt, max_tokens=256, temperature=0.7)
    )


def judge_label(client: BedrockClient, text: str, hindi: str) -> dict | None:
    user = f"Hinglish: {text}\nHindi: {hindi}\nJSON:"
    try:
        raw = _judge_call(client, user)
    except Exception as e:
        print(f"judge fail {type(e).__name__}", flush=True)
        return None
    return parse_judge(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=ROOT / "codemix_hinglish")
    p.add_argument("--clean", type=Path, default=ROOT / "codemix_hinglish_clean")
    p.add_argument("--synth-dir", type=Path, default=ROOT / "codemix_synth")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--model", default="qwen.qwen3-32b")
    p.add_argument("--backend", default="auto", choices=["auto", "mantle", "converse"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit-seeds", type=int, default=0)
    p.add_argument("--max-per-origin", type=int, default=800)
    p.add_argument("--cap-accepted", type=int, default=5000)
    p.add_argument("--min-cosine", type=float, default=0.55)
    p.add_argument("--contrastive-frac", type=float, default=0.15)
    p.add_argument("--audit-n", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.synth_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    seeds = load_seeds(args.synth_dir, args.src, args.max_per_origin, rng)
    if args.limit_seeds:
        seeds = seeds.head(args.limit_seeds)

    hold_paths = []
    for folder in (args.src, args.clean):
        for name in ("train.csv", "val.csv", "test.csv"):
            fp = folder / name
            if fp.exists():
                hold_paths.append(fp)
    banned = set()
    for fp in hold_paths:
        df = pd.read_csv(fp, usecols=lambda c: c in {"text"})
        banned.update(df["text"].fillna("").map(norm_text))

    print(f"seeds={len(seeds)} banned_norms={len(banned)}", flush=True)
    print("loading embedder + scorer", flush=True)
    embedder = Embedder()
    scorer = AbuseScorer(DEFAULT_SCORER)
    client0 = BedrockClient(args.region, args.model, backend=args.backend)

    accepted_path = args.synth_dir / "accepted.csv"
    quar_path = args.synth_dir / "quarantine.csv"
    acc_fields = [
        "row_id", "split", "label", "text", "hindi_text", "seed_text",
        "seed_origin", "variant", "cosine", "judge_label", "judge_conf",
        "hinglish_abuse", "source",
    ]
    q_fields = acc_fields + ["fail_reason"]

    done_ids = set()
    if accepted_path.exists():
        done_ids |= set(pd.read_csv(accepted_path)["row_id"].astype(str))
    if quar_path.exists():
        done_ids |= set(pd.read_csv(quar_path)["row_id"].astype(str))

    n_acc = 0
    if accepted_path.exists():
        n_acc = len(pd.read_csv(accepted_path))
        for t in pd.read_csv(accepted_path)["text"].fillna(""):
            banned.add(norm_text(t))

    thread_clients: dict[int, BedrockClient] = {}
    state_lock = threading.Lock()

    def get_client() -> BedrockClient:
        tid = threading.get_ident()
        if tid not in thread_clients:
            thread_clients[tid] = BedrockClient(args.region, args.model, backend=args.backend)
        return thread_clients[tid]

    jobs = []
    for _, seed in seeds.iterrows():
        label = int(seed["label"])
        include_c = rng.random() < args.contrastive_frac
        kinds = variants_for(label, include_c)
        hindi = str(seed.get("hindi_text") or "")
        seed_text = str(seed.get("text") or "")
        if not seed_text and hindi:
            seed_text = hindi
        if not seed_text:
            continue
        for kind in kinds:
            vid = f"synth-{seed.get('row_id', 'x')}-{kind}"
            if vid in done_ids:
                continue
            jobs.append((vid, seed, seed_text, hindi, label, kind))

    print(f"jobs={len(jobs)} already_accepted={n_acc}", flush=True)

    def process(job):
        nonlocal n_acc
        vid, seed, seed_text, hindi, label, kind = job
        with state_lock:
            if n_acc >= args.cap_accepted:
                return
        client = get_client()
        want_label = (1 - label) if kind == "contrastive" else label
        try:
            out = generate_one(client, gen_prompt(seed_text, hindi, kind, label))
        except Exception as e:
            out = ""
            fail = f"api_error:{type(e).__name__}"
            row = {
                "row_id": vid, "split": "train", "label": want_label, "text": "",
                "hindi_text": hindi, "seed_text": seed_text[:240],
                "seed_origin": seed.get("seed_origin", ""), "variant": kind,
                "cosine": "", "judge_label": "", "judge_conf": "",
                "hinglish_abuse": "", "source": "hard_aug", "fail_reason": fail,
            }
            from codemix_convert_bedrock import append_csv
            append_csv(quar_path, row, q_fields)
            return

        from codemix_convert_bedrock import append_csv

        fail = None
        ok_hard, hard_reason = hard_checks(hindi or seed_text, out)
        if not ok_hard:
            fail = hard_reason
        elif norm_text(out) == norm_text(seed_text):
            fail = "dup_or_identity"
        elif want_label == 0 and hindi and not lexicon_preserved(hindi, out) and kind != "obfuscation":
            fail = "abuse_lexicon_missing"
        else:
            with state_lock:
                if norm_text(out) in banned:
                    fail = "dup_or_identity"
        cosine = 0.0
        if fail is None:
            cosine = embedder.cosine(seed_text, out)
            if kind != "contrastive" and cosine < args.min_cosine:
                fail = f"low_cosine:{cosine:.2f}"
        judged = None
        if fail is None:
            judged = judge_label(client, out, hindi)
            if not judged or judged["confidence"] < 0.6:
                fail = "judge_unreliable"
            elif judged["label"] != want_label:
                fail = f"judge_mismatch:{judged['label']}"
        p_ab = 0.0
        if fail is None:
            try:
                p_ab = scorer.abuse_prob(out)
            except Exception:
                p_ab = 0.0
            # ONNX is NOT a veto (breaks circularity). Record only.

        base = {
            "row_id": vid,
            "split": "train",
            "label": want_label,
            "text": out,
            "hindi_text": hindi,
            "seed_text": seed_text[:240],
            "seed_origin": seed.get("seed_origin", ""),
            "variant": kind,
            "cosine": round(cosine, 4) if cosine else "",
            "judge_label": judged["label"] if judged else "",
            "judge_conf": round(judged["confidence"], 3) if judged else "",
            "hinglish_abuse": round(p_ab, 4),
            "source": "hard_aug",
        }
        with state_lock:
            if fail:
                append_csv(quar_path, {**base, "fail_reason": fail}, q_fields)
            else:
                if n_acc >= args.cap_accepted or norm_text(out) in banned:
                    append_csv(
                        quar_path,
                        {**base, "fail_reason": "cap_or_race_dup"},
                        q_fields,
                    )
                else:
                    append_csv(accepted_path, base, acc_fields)
                    banned.add(norm_text(out))
                    n_acc += 1
                    if n_acc % 25 == 0:
                        print(f"accepted={n_acc}", flush=True)

    workers = max(1, args.workers)
    if workers == 1:
        for job in jobs:
            if n_acc >= args.cap_accepted:
                break
            process(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(process, jobs, chunksize=1))

    if accepted_path.exists():
        acc = pd.read_csv(accepted_path)
        if len(acc) > args.cap_accepted:
            acc = acc.head(args.cap_accepted)
            acc.to_csv(accepted_path, index=False)
        audit = acc.sample(n=min(args.audit_n, len(acc)), random_state=args.seed)
        audit.to_csv(args.synth_dir / "audit.csv", index=False)
        report = {
            "accepted": int(len(acc)),
            "by_variant": acc["variant"].value_counts().to_dict() if "variant" in acc.columns else {},
            "by_label": acc["label"].value_counts().to_dict(),
            "audit_n": int(len(audit)),
            "audit_path": str(args.synth_dir / "audit.csv"),
        }
        if quar_path.exists():
            q = pd.read_csv(quar_path)
            report["quarantined"] = int(len(q))
            report["by_fail"] = q["fail_reason"].value_counts().head(20).to_dict()
        (args.synth_dir / "synth_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
    else:
        print("no accepted rows", flush=True)


if __name__ == "__main__":
    main()
