#!/usr/bin/env python3
"""Expand frozen Hinglish test messages into WhatsApp-style conversations.

The current window benchmark glues unrelated comments. This script asks
Qwen3-32B for surrounding turns so the gated rolling window is measured on
dialogue, not a stitch. Frozen test.csv is read, never written.

Kinds
  benign_chat            class-1 seed kept verbatim inside an LLM chat.
                         Gold = non-abusive.
  abuse_after_smalltalk  LLM small-talk, then the class-0 seed as last turn.
                         Gold = abusive. The slur is never sent to the LLM.

Usage (from repo root):
  .venv_codemix/bin/python -u aws_train/expand_dataset_to_convos.py --n 200
  .venv_codemix/bin/python -u aws_train/expand_dataset_to_convos.py --score-only
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from benchmark_narrative_kv import (  # noqa: E402
    Example,
    content_tokens,
    gated_window_scores,
    shipped_scores,
)
from codemix_convert_bedrock import (  # noqa: E402
    ABUSE_STEMS,
    DEVANAGARI,
    REFUSAL_PATTERNS,
    BedrockClient,
)

TEST_CSV = ROOT / "codemix_hinglish" / "test.csv"
OUT_DIR = ROOT / "codemix_eval"
CACHE = OUT_DIR / "llm_convo_cache.jsonl"
EVAL_JSONL = OUT_DIR / "llm_convo_eval.jsonl"
REPORT = OUT_DIR / "llm_convo_window_report.json"

MAX_SLOTS = 8
MAX_WINDOW_TOKENS = 96
SHORT_TOKENS = 8
THRESHOLD = 0.5

BENIGN_SYSTEM = """You expand one ordinary Hinglish WhatsApp message into a short two-person chat for an NLP evaluation set.

Rules:
- Roman script only. No Devanagari.
- Hindi is the matrix language; mix English the way Indian chat does.
- One turn MUST copy the seed character-for-character.
- Other turns are short replies or small talk that give that seed context.
- Keep the whole chat ordinary. No insults, slurs, aggression, or moralising.
- 4 to 6 turns. Each turn under 20 words.
- Output JSON only: {"turns":["...","..."]}
No commentary."""

SMALLTALK_SYSTEM = """You write a short ordinary Hinglish WhatsApp exchange for an NLP evaluation set.

Rules:
- Roman script only. No Devanagari.
- Two friends, daily life (plans, college, food, work). No insults, slurs, politics, or abuse.
- 4 turns. Each turn under 16 words.
- Output JSON only: {"turns":["...","..."]}
No commentary."""

JSON_OBJ = re.compile(r"\{.*\}", re.S)
ROMAN_STEMS = tuple({r.lower() for _, rs in ABUSE_STEMS for r in rs})


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def parse_turns(raw: str) -> list[str] | None:
    if not raw:
        return None
    blob = raw.strip()
    m = JSON_OBJ.search(blob)
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            items = data.get("turns") or data.get("messages")
            if isinstance(items, list):
                turns = [re.sub(r"\s+", " ", str(t)).strip() for t in items]
                turns = [t for t in turns if t]
                if 3 <= len(turns) <= 8:
                    return turns
    lines = []
    for line in blob.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)]|[AB]:)\s*", "", line).strip().strip('"')
        if line:
            lines.append(re.sub(r"\s+", " ", line))
    if 3 <= len(lines) <= 8:
        return lines
    return None


def has_refusal(text: str) -> bool:
    return bool(REFUSAL_PATTERNS.search(text or ""))


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(text or ""))


def context_has_slur(turns: list[str], seed: str) -> bool:
    seed_n = norm(seed)
    for t in turns:
        if norm(t) == seed_n:
            continue
        low = t.lower()
        if any(s in low for s in ROMAN_STEMS):
            return True
    return False


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = rec.get("key")
        if key:
            out[str(key)] = rec
    return out


def append_jsonl(path: Path, rec: dict, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pick_seeds(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(TEST_CSV)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df = df[df.text.str.len().between(12, 180)]
    rng = random.Random(seed)
    benign = df[df.label == 1].copy()
    abuse = df[df.label == 0].copy()
    b_idx = list(benign.index)
    a_idx = list(abuse.index)
    rng.shuffle(b_idx)
    rng.shuffle(a_idx)
    return (
        benign.loc[b_idx[:n]].reset_index(drop=True),
        abuse.loc[a_idx[:n]].reset_index(drop=True),
    )


def force_seed(turns: list[str], seed: str, last: bool) -> list[str]:
    seed_n = norm(seed)
    if any(norm(t) == seed_n for t in turns):
        if last:
            others = [t for t in turns if norm(t) != seed_n]
            return others + [seed]
        return [seed if norm(t) == seed_n else t for t in turns]
    if last:
        return turns + [seed]
    if len(turns) >= 2:
        return [turns[0], seed, *turns[1:]]
    return [seed, *turns]


def valid_chat(kind: str, seed: str, turns: list[str]) -> str | None:
    if len(turns) < 3:
        return "too_short"
    joined = " ".join(turns)
    if has_refusal(joined):
        return "refusal"
    if has_devanagari(joined):
        return "devanagari"
    if any(len(t) < 2 for t in turns):
        return "empty_turn"
    if kind == "benign_chat" and not any(norm(t) == norm(seed) for t in turns):
        return "seed_missing"
    if kind == "abuse_after_smalltalk" and norm(turns[-1]) != norm(seed):
        return "seed_not_last"
    if context_has_slur(turns, seed):
        return "context_slur"
    return None


def generate_one(client: BedrockClient, kind: str, seed: str) -> tuple[list[str] | None, str]:
    if kind == "benign_chat":
        user = f"SEED:\n{seed}\n\nWrite the JSON chat now."
        system = BENIGN_SYSTEM
        temp = 0.5
    else:
        user = "Write four ordinary Hinglish WhatsApp turns between two friends. JSON only."
        system = SMALLTALK_SYSTEM
        temp = 0.7
    try:
        raw = client.complete(system, user, max_tokens=280, temperature=temp)
    except Exception as exc:
        return None, f"llm_error:{type(exc).__name__}"
    turns = parse_turns(raw)
    if not turns:
        return None, "parse"
    turns = force_seed(turns, seed, last=(kind == "abuse_after_smalltalk"))
    reason = valid_chat(kind, seed, turns)
    if reason:
        return None, reason
    return turns, "ok"


def generate(args: argparse.Namespace) -> None:
    benign, abuse = pick_seeds(args.n, args.seed)
    jobs = [("benign_chat", r.text, int(r.label)) for r in benign.itertuples(index=False)]
    jobs += [("abuse_after_smalltalk", r.text, int(r.label)) for r in abuse.itertuples(index=False)]
    cache = load_jsonl(CACHE)
    pending = []
    for i, (kind, text, lab) in enumerate(jobs):
        key = f"{kind}:{args.seed}:{i}"
        rec = cache.get(key)
        if rec and rec.get("ok") and rec.get("turns"):
            continue
        pending.append((key, kind, text, lab))
    print(f"seeds {len(jobs)} cached_ok {len(jobs) - len(pending)} pending {len(pending)}", flush=True)
    if not pending:
        return
    client = BedrockClient(args.region, args.model, backend=args.backend)
    lock = threading.Lock()
    counts: dict[str, int] = {}

    def work(item: tuple[str, str, str, int]) -> None:
        key, kind, text, lab = item
        turns, reason = generate_one(client, kind, text)
        rec = {
            "key": key,
            "kind": kind,
            "seed_label": lab,
            "ok": reason == "ok",
            "reason": reason,
            "turns": turns or [],
            "n_turns": len(turns or []),
        }
        append_jsonl(CACHE, rec, lock)
        with lock:
            counts[reason] = counts.get(reason, 0) + 1
            done = sum(counts.values())
            if done % 20 == 0 or done == len(pending):
                print(f"  gen {done}/{len(pending)} {counts}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, pending))
    print("generate done", counts, flush=True)


def accepted_examples() -> list[dict]:
    latest: dict[str, dict] = {}
    for rec in load_jsonl(CACHE).values():
        latest[rec["key"]] = rec
    rows = []
    for rec in latest.values():
        if not rec.get("ok") or not rec.get("turns"):
            continue
        kind = rec["kind"]
        y = 0 if kind == "benign_chat" else 1
        rows.append(
            {
                "key": rec["key"],
                "kind": kind,
                "y": y,
                "turns": rec["turns"],
                "n_turns": len(rec["turns"]),
            }
        )
    return rows


def trim_window(turns: list[str]) -> list[str]:
    kept = list(turns)
    while len(kept) > MAX_SLOTS:
        kept.pop(0)
    def tok_sum(slots: list[str]) -> int:
        return sum(max(len(content_tokens(t)), 1 if t.strip() else 0) for t in slots)
    while len(kept) > 1 and tok_sum(kept) > MAX_WINDOW_TOKENS:
        kept.pop(0)
    return kept


def metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    pred = (scores >= THRESHOLD).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": int(len(y)),
        "accuracy": round(float((pred == y).mean()) if len(y) else 0.0, 4),
        "precision_abusive": round(prec, 4),
        "recall_abusive": round(rec, 4),
        "f1_abusive": round(f1, 4),
        "fpr": round(1.0 - spec, 4),
        "positives": int(y.sum()),
    }


def score() -> None:
    rows = accepted_examples()
    if not rows:
        raise SystemExit("no accepted conversations in cache; run without --score-only first")
    EVAL_JSONL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    examples = [
        Example(r["turns"], r["y"], r["kind"], None) for r in rows
    ]
    trimmed = [trim_window(r["turns"]) for r in rows]
    current_texts = [t[-1] for t in trimmed]
    window_texts = [" ".join(t) for t in trimmed]
    print(f"scoring INT8 on {len(rows)} chats", flush=True)
    table = shipped_scores(current_texts + window_texts)
    if table is None:
        raise SystemExit("INT8 scorer missing")
    current = np.array([table[t] for t in current_texts], dtype=np.float64)
    window = np.array([table[t] for t in window_texts], dtype=np.float64)
    fused, coverage = gated_window_scores(examples, current, window, SHORT_TOKENS)
    y = np.array([r["y"] for r in rows], dtype=int)
    kinds = sorted({r["kind"] for r in rows})

    def by_kind(scores: np.ndarray) -> dict:
        out = {"all": metrics(y, scores)}
        for k in kinds:
            mask = np.array([r["kind"] == k for r in rows])
            out[k] = metrics(y[mask], scores[mask])
        return out

    report = {
        "protocol": {
            "seed_source": "codemix_hinglish/test.csv (read-only)",
            "seed_verbatim": True,
            "max_slots": MAX_SLOTS,
            "max_window_tokens": MAX_WINDOW_TOKENS,
            "short_tokens": SHORT_TOKENS,
            "threshold": THRESHOLD,
            "kinds": {
                "benign_chat": "LLM wraps a class-1 test line in a 4-6 turn Hinglish chat. Gold non-abusive.",
                "abuse_after_smalltalk": "LLM small-talk, then the class-0 test line as the last turn. Gold abusive.",
            },
        },
        "n_accepted": len(rows),
        "n_by_kind": {k: int(sum(1 for r in rows if r["kind"] == k)) for k in kinds},
        "gated_window_coverage": coverage,
        "isolated_last_turn": by_kind(current),
        "concatenated_window": by_kind(window),
        "gated_rolling_window": by_kind(fused),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "n": len(rows),
            "isolated": report["isolated_last_turn"]["all"],
            "window": report["concatenated_window"]["all"],
            "gated": report["gated_rolling_window"]["all"],
            "benign_isolated_fpr": report["isolated_last_turn"].get("benign_chat", {}).get("fpr"),
            "benign_window_fpr": report["concatenated_window"].get("benign_chat", {}).get("fpr"),
            "benign_gated_fpr": report["gated_rolling_window"].get("benign_chat", {}).get("fpr"),
            "abuse_isolated_recall": report["isolated_last_turn"].get("abuse_after_smalltalk", {}).get("recall_abusive"),
            "abuse_window_recall": report["concatenated_window"].get("abuse_after_smalltalk", {}).get("recall_abusive"),
            "abuse_gated_recall": report["gated_rolling_window"].get("abuse_after_smalltalk", {}).get("recall_abusive"),
            "coverage": coverage,
        },
        indent=2,
    ), flush=True)
    print(f"wrote {REPORT}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200, help="seeds per kind from frozen test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--model", default="qwen.qwen3-32b")
    p.add_argument("--backend", default="auto", choices=["auto", "mantle", "converse"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    if not args.score_only:
        generate(args)
    score()


if __name__ == "__main__":
    main()
