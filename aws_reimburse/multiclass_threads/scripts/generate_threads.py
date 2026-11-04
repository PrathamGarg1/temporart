#!/usr/bin/env python3
"""Generate 5-class Hinglish WhatsApp threads of length 3/5/7/10.

Qwen writes preceding turns. The last turn is taken from a locked bank so
labels stay honest. Frozen binary test.csv is read, never written.

Usage (from repo root):
  .venv_codemix/bin/python -u aws_reimburse/multiclass_threads/scripts/generate_threads.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_ledger import totals  # noqa: E402
from logged_bedrock import LoggedBedrock  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
ART = HERE / "artifacts"
CACHE = ART / "thread_cache.jsonl"
TRAIN = ART / "threads_train.jsonl"
VAL = ART / "threads_val.jsonl"
TEST = ART / "threads_test.jsonl"
FREEZE = ART / "FREEZE.json"

LENGTHS = (3, 5, 7, 10)
CLASSES = (
    "benign",
    "friendly_sarcasm",
    "hostile_abuse",
    "sexual_harassment",
    "predatory_grooming",
)
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

SHARED_ROASTS = [
    "you r an idiot",
    "tu pagal hai yaar",
    "kya bakwas kar raha hai",
    "chal jhootha",
    "arey pagal ho kya",
    "tu toh gone case hai",
    "kya dimag kharab hai tera",
    "stop acting smart",
]

BENIGN_LAST = [
    "kal 4 baje milte hain",
    "notes share kar dena",
    "ok done thanks",
    "canteen chale?",
    "assignment kal subah bhej dunga",
    "mumma ko bol diya 8 baje dinner",
    "haan wohi plan hai",
    "theek hai, call karta hoon",
]

HARASS_LAST = [
    "wear something smaller tomorrow",
    "send a pic of yourself, not the slides",
    "come to my room after office",
    "you look too good in that dress, sit closer",
    "don't wear that jacket, I want to see more",
    "message me tonight, not about work",
]

GROOM_LAST = [
    "don't tell your parents we talk",
    "how old are you actually",
    "this is our secret, delete the chat",
    "your parents don't need to know",
    "keep this between us only",
    "are you home alone right now",
]

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
JSON_OBJ = re.compile(r"\{.*\}", re.S)
REFUSAL = re.compile(
    r"(?i)\b(i (can'?t|cannot|won'?t)|as an ai|inappropriate|cannot (assist|help))\b"
)
SEXUAL_CUES = re.compile(
    r"(?i)\b(sex|nude|naked|pic|photo|body|hot|sexy|kiss|bed|room)\b"
)

SYSTEM = """You write preceding turns of a Hinglish WhatsApp chat for an NLP dataset.
Roman script only. No Devanagari. Hindi is the matrix language; mix English naturally.
Output JSON only: {"turns":["..."]}
No commentary, no last-turn copy, no labels."""

PROMPT = {
    "benign": (
        "Write {n} ordinary Hinglish turns between two friends (plans, college, food). "
        "No insults, no romance, no secrets. The next message will be: {last}"
    ),
    "friendly_sarcasm": (
        "Write {n} Hinglish turns of two friends joking after a match or assignment. "
        "Warm, teasing, not a fight. The next message will be a roast: {last}"
    ),
    "hostile_abuse": (
        "Write {n} Hinglish turns where one person asked the other to stop texting "
        "and is upset. Not joking. The next message will be: {last}"
    ),
    "sexual_harassment": (
        "Write {n} Hinglish turns of an adult workplace or college chat. "
        "Keep them non-sexual. An adult later sends an unwanted comment. "
        "Do not write sexual content yourself. Next message will be: {last}"
    ),
    "predatory_grooming": (
        "Write {n} Hinglish turns of small talk. Do not write sexual content. "
        "Do not mention bodies. Next message will be about secrecy or age: {last}"
    ),
}


def parse_turns(raw: str, n: int) -> list[str] | None:
    if not raw:
        return None
    m = JSON_OBJ.search(raw)
    items = None
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            items = data.get("turns") or data.get("messages")
    if not isinstance(items, list):
        items = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
            if line:
                items.append(re.sub(r"\s+", " ", line))
    turns = [re.sub(r"\s+", " ", str(t)).strip() for t in items]
    turns = [t for t in turns if t]
    if len(turns) < n:
        return None
    return turns[:n]


def last_bank(cls: str) -> list[str]:
    if cls in ("friendly_sarcasm", "hostile_abuse"):
        return SHARED_ROASTS
    if cls == "benign":
        return BENIGN_LAST
    if cls == "sexual_harassment":
        return HARASS_LAST
    return GROOM_LAST


def valid(cls: str, context: list[str], last: str) -> str | None:
    joined = " ".join(context + [last])
    if REFUSAL.search(joined):
        return "refusal"
    if DEVANAGARI.search(joined):
        return "devanagari"
    if any(len(t) < 2 for t in context):
        return "empty"
    if cls == "predatory_grooming" and SEXUAL_CUES.search(" ".join(context)):
        return "grooming_sexual"
    if cls == "benign" and SEXUAL_CUES.search(joined):
        return "benign_sexual"
    return None


def load_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not CACHE.exists():
        return out
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["key"]] = rec
    return out


def append_cache(rec: dict, lock: threading.Lock) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    with lock:
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def jobs(n_train: int, n_val: int, n_test: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    split_n = {"train": n_train, "val": n_val, "test": n_test}
    for length in LENGTHS:
        for split, n in split_n.items():
            for cls in CLASSES:
                bank = last_bank(cls)
                for i in range(n):
                    last = bank[i % len(bank)]
                    key = f"{split}:{length}:{cls}:{i}"
                    out.append(
                        {
                            "key": key,
                            "split": split,
                            "length": length,
                            "label": cls,
                            "label_id": CLASS_ID[cls],
                            "last": last,
                            "pair_id": f"{last}|{i % len(bank)}"
                            if cls in ("friendly_sarcasm", "hostile_abuse")
                            else "",
                        }
                    )
    rng.shuffle(out)
    return out


def generate_one(client: LoggedBedrock, spec: dict) -> tuple[list[str] | None, str]:
    n_ctx = spec["length"] - 1
    user = PROMPT[spec["label"]].format(n=n_ctx, last=spec["last"])
    try:
        raw = client.complete(SYSTEM, user, max_tokens=320, temperature=0.7)
    except Exception as exc:
        return None, f"llm:{type(exc).__name__}"
    ctx = parse_turns(raw, n_ctx)
    if not ctx:
        return None, "parse"
    reason = valid(spec["label"], ctx, spec["last"])
    if reason:
        return None, reason
    return ctx + [spec["last"]], "ok"


def write_splits(cache: dict[str, dict]) -> None:
    buckets = {"train": [], "val": [], "test": []}
    for rec in cache.values():
        if rec.get("ok") and rec.get("turns"):
            buckets[rec["split"]].append(
                {
                    "thread_id": rec["key"],
                    "split": rec["split"],
                    "length": rec["length"],
                    "label": rec["label"],
                    "label_id": rec["label_id"],
                    "turns": rec["turns"],
                    "last": rec["turns"][-1],
                    "pair_id": rec.get("pair_id") or "",
                }
            )
    ART.mkdir(parents=True, exist_ok=True)
    for split, path in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        rows = sorted(buckets[split], key=lambda r: r["thread_id"])
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
    freeze = {
        "frozen": True,
        "test_path": str(TEST.relative_to(HERE.parent.parent)),
        "n_test": len(buckets["test"]),
        "n_train": len(buckets["train"]),
        "n_val": len(buckets["val"]),
        "classes": list(CLASSES),
        "lengths": list(LENGTHS),
        "rule": "Do not rewrite threads_test.jsonl after looking at errors.",
    }
    FREEZE.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote train {len(buckets['train'])} val {len(buckets['val'])} test {len(buckets['test'])}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=50)
    p.add_argument("--n-val", type=int, default=15)
    p.add_argument("--n-test", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--model", default="qwen.qwen3-32b")
    p.add_argument("--backend", default="auto", choices=["auto", "mantle", "converse"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--job", default="multiclass_thread_gen")
    args = p.parse_args()

    specs = jobs(args.n_train, args.n_val, args.n_test, args.seed)
    cache = load_cache()
    pending = [s for s in specs if not (cache.get(s["key"]) or {}).get("ok")]
    print(f"total {len(specs)} cached_ok {len(specs) - len(pending)} pending {len(pending)}", flush=True)

    if pending:
        client = LoggedBedrock(args.region, args.model, args.backend, job=args.job)
        lock = threading.Lock()
        counts: dict[str, int] = {}

        def work(spec: dict) -> None:
            turns, reason = generate_one(client, spec)
            rec = {**spec, "ok": reason == "ok", "reason": reason, "turns": turns or []}
            append_cache(rec, lock)
            with lock:
                counts[reason] = counts.get(reason, 0) + 1
                done = sum(counts.values())
                if done % 25 == 0 or done == len(pending):
                    print(f"  gen {done}/{len(pending)} {counts} usd={totals()['usd_estimated']}", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, pending))
        print("generate done", counts, flush=True)

    cache = load_cache()
    write_splits(cache)
    print("ledger", totals(), flush=True)


if __name__ == "__main__":
    main()
