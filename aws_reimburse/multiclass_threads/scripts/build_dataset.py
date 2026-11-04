#!/usr/bin/env python3
"""Build the 100k+ multiclass thread dataset (generate, validate, write tree).

  python build_dataset.py --per-cell 4  --workers 12          # pilot
  python build_dataset.py --per-cell 1750 --workers 24        # full
  python build_dataset.py --per-cell 1750 --write-only        # rebuild folders

Folder layout:
  dataset/len_03/hindi_devanagari/0_benign/{train,val,test}.jsonl
  dataset/all/{train,val,test}.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_ledger import flush, totals  # noqa: E402
from dataset_spec import (  # noqa: E402
    CLASS_DIR,
    CLASSES,
    DEV_RE,
    EXPLICIT_RE,
    GROOM_BAD_RE,
    HIN_MARKERS,
    HIN_STRICT,
    LAT_RE,
    LENGTHS,
    PROFANE_RE,
    REFUSAL_RE,
    SCRIPTS,
    WORD_RE,
    SYSTEM,
    build_prompt,
    build_spec,
    split_for,
)

HERE = Path(__file__).resolve().parents[1]
DATASET = HERE / "dataset"
CACHE_DIR = DATASET / "_cache"
CACHE = CACHE_DIR / "generations.jsonl"
JOB = "multiclass_dataset_v2"

MAX_TOKENS = {3: 380, 5: 520, 7: 700, 10: 950}
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
JSON_RE = re.compile(r"\{.*\}", re.S)
ROLE_RE = re.compile(r"^\s*(?:[A-B]|speaker\s*\d)\s*[:：]\s*", re.I)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_turns(raw: str, k: int, names: tuple[str, str]) -> list[str] | None:
    if not raw:
        return None
    raw = THINK_RE.sub("", raw).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = JSON_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    items = data.get("turns") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    out = []
    for it in items:
        if isinstance(it, dict):
            it = it.get("text") or it.get("message") or ""
        t = re.sub(r"\s+", " ", str(it)).strip()
        t = ROLE_RE.sub("", t)
        for nm in names:
            if t.lower().startswith(nm.lower() + ":"):
                t = t[len(nm) + 1 :].strip()
        out.append(t)
    return out if len(out) == k else None


def script_ok(script: str, turns: list[str]) -> bool:
    txt = " ".join(turns)
    dev = len(DEV_RE.findall(txt))
    lat = len(LAT_RE.findall(txt))
    words = [w.lower() for w in WORD_RE.findall(txt)]
    if script == "hindi_devanagari":
        return dev >= 10 and dev / max(dev + lat, 1) >= 0.85
    if dev:
        return False
    if not words:
        return False
    if script == "english":
        strict = sum(1 for w in words if w in HIN_STRICT)
        return strict / len(words) < 0.04
    marked = sum(1 for w in words if w in HIN_MARKERS)
    return marked / len(words) >= 0.15


def validate(spec: dict, turns: list[str]) -> str | None:
    k = spec["length"]
    label = spec["label"]
    if len(turns) != k:
        return "bad_count"
    joined = " ".join(turns)
    if REFUSAL_RE.search(joined):
        return "refusal"
    if any(len(t) < 2 or len(t) > 320 for t in turns):
        return "turn_len"
    for a, b in zip(turns, turns[1:]):
        if norm(a) == norm(b):
            return "adjacent_dup"
    if norm(turns[-1]) in {norm(t) for t in turns[:-1]}:
        return "last_in_context"
    if spec["pair_line"] and norm(turns[-1]).strip(' ."\'!') != norm(spec["pair_line"]).strip(' ."\'!'):
        return "pair_line_changed"
    if not script_ok(spec["script"], turns):
        return "script"
    if label != "hostile_abuse" and PROFANE_RE.search(joined):
        return "profanity_in_clean_class"
    if label != "hostile_abuse" and EXPLICIT_RE.search(joined):
        return "explicit"
    if label == "predatory_grooming" and GROOM_BAD_RE.search(joined):
        return "grooming_sexualised"
    return None


def load_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not CACHE.exists():
        return out
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["key"]] = rec
    return out


def thread_hash(turns: list[str]) -> str:
    return hashlib.md5("\n".join(norm(t) for t in turns).encode("utf-8")).hexdigest()


class Runner:
    def __init__(self, args) -> None:
        from logged_bedrock import LoggedBedrock

        self.args = args
        self.client = LoggedBedrock(args.region, args.model, args.backend, job=JOB)
        self.lock = threading.Lock()
        self.counts: Counter = Counter()
        self.seen: set[str] = set()
        self.stop = False
        self.start_usd = totals()["usd_estimated"]
        self.t0 = time.time()
        self.done = 0
        self.total = 0

    def spent(self) -> float:
        return totals()["usd_estimated"] - self.start_usd

    def work(self, spec: dict) -> None:
        if self.stop:
            return
        prompt = build_prompt(spec)
        names = (spec["name_x"], spec["name_y"])
        turns, reason = None, "llm_error"
        for attempt in range(2):
            try:
                raw = self.client.complete(
                    SYSTEM, prompt, max_tokens=MAX_TOKENS[spec["length"]], temperature=0.85
                )
            except Exception as exc:  # noqa: BLE001
                reason = f"llm:{type(exc).__name__}"
                continue
            parsed = parse_turns(raw, spec["length"], names)
            if parsed is None:
                reason = "parse"
                continue
            bad = validate(spec, parsed)
            if bad:
                reason = bad
                continue
            turns, reason = parsed, "ok"
            break
        if turns is not None:
            h = thread_hash(turns)
            with self.lock:
                if h in self.seen:
                    turns, reason = None, "dup"
                else:
                    self.seen.add(h)
        rec = {**spec, "ok": turns is not None, "reason": reason, "turns": turns or []}
        with self.lock:
            with CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.counts[reason] += 1
            self.done += 1
            if self.done % 250 == 0 or self.done == self.total:
                el = time.time() - self.t0
                spent = self.spent()
                print(
                    f"  {self.done}/{self.total} ok={self.counts['ok']} "
                    f"rate={self.done / max(el, 1):.1f}/s usd={spent:.3f} "
                    f"fails={ {k: v for k, v in self.counts.items() if k != 'ok'} }",
                    flush=True,
                )
            if spent_cap(self) and not self.stop:
                self.stop = True
                print(f"COST CAP HIT ({self.args.max_usd} USD); stopping", flush=True)


def spent_cap(r: Runner) -> bool:
    return r.spent() >= r.args.max_usd


def cells():
    for length in LENGTHS:
        for script in SCRIPTS:
            for label in CLASSES:
                yield (script, length, label)


def generate(args) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    runner = Runner(args)
    for rec in cache.values():
        if rec.get("ok") and rec.get("turns"):
            runner.seen.add(thread_hash(rec["turns"]))
    for rnd in range(args.rounds):
        cache = load_cache()
        acc: Counter = Counter()
        top_n: dict[tuple, int] = defaultdict(lambda: -1)
        for rec in cache.values():
            cell = (rec["script"], rec["length"], rec["label"])
            top_n[cell] = max(top_n[cell], rec["n"])
            if rec.get("ok"):
                acc[cell] += 1
        specs = []
        for cell in cells():
            deficit = args.per_cell - acc[cell]
            if deficit <= 0:
                continue
            need = math.ceil(deficit * (1.0 + args.overshoot)) + 2
            start = top_n[cell] + 1
            for n in range(start, start + need):
                specs.append(build_spec(cell[0], cell[1], cell[2], n, args.seed))
        if not specs:
            print("all cells filled", flush=True)
            break
        random.Random(args.seed + rnd).shuffle(specs)
        runner.total = len(specs)
        runner.done = 0
        runner.t0 = time.time()
        print(f"round {rnd}: {len(specs)} specs, cap {args.max_usd} USD", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(runner.work, specs))
        if runner.stop:
            break
    flush()
    print("ledger", totals(), flush=True)


def write_tree(per_cell: int) -> None:
    cache = load_cache()
    per: dict[tuple, list[dict]] = defaultdict(list)
    for rec in cache.values():
        if rec.get("ok") and rec.get("turns"):
            per[(rec["script"], rec["length"], rec["label"])].append(rec)
    for cell in per:
        per[cell].sort(key=lambda r: r["n"])
        per[cell] = per[cell][:per_cell]

    for old in DATASET.glob("len_*"):
        for p in sorted(old.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        old.rmdir()
    allf = DATASET / "all"
    allf.mkdir(parents=True, exist_ok=True)
    handles = {s: (allf / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "val", "test")}
    manifest: Counter = Counter()
    sub_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    for (script, length, label), recs in sorted(per.items()):
        buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for r in recs:
            split = split_for(r)
            x, y = r["name_x"], r["name_y"]
            speakers = [x if (i % 2 == (0 if length % 2 == 1 else 1)) else y for i in range(length)]
            out = {
                "thread_id": r["key"],
                "split": split,
                "length": length,
                "n_context": length - 1,
                "script": script,
                "label": label,
                "label_id": r["label_id"],
                "subtype": r["subtype"],
                "topic": r["topic"],
                "audience": r["audience"],
                "target_gender": r["target_gender"],
                "child_age": r["child_age"],
                "pair_id": r["pair_id"],
                "speakers": speakers,
                "turns": r["turns"],
                "last_turn": r["turns"][-1],
                "source": "qwen3-32b via Amazon Bedrock (synthetic)",
            }
            buckets[split].append(out)
            handles[split].write(json.dumps(out, ensure_ascii=False) + "\n")
            manifest[(length, script, label, split)] += 1
            sub_counts[(label, r["subtype"])] += 1
            if r["pair_id"]:
                pair_counts[(split, label)] += 1
        folder = DATASET / f"len_{length:02d}" / script / CLASS_DIR[label]
        folder.mkdir(parents=True, exist_ok=True)
        for split, rows in buckets.items():
            (folder / f"{split}.jsonl").write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + ("\n" if rows else ""),
                encoding="utf-8",
            )
    for h in handles.values():
        h.close()

    with (DATASET / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length", "script", "label", "split", "count"])
        for k in sorted(manifest):
            w.writerow([*k, manifest[k]])
    total = sum(manifest.values())
    by_split = Counter()
    by_len = Counter()
    by_script = Counter()
    by_label = Counter()
    for (length, script, label, split), n in manifest.items():
        by_split[split] += n
        by_len[length] += n
        by_script[script] += n
        by_label[label] += n
    write_card(total, by_split, by_len, by_script, by_label, sub_counts, pair_counts)
    print(
        f"wrote dataset tree: total={total} splits={dict(by_split)} "
        f"len={dict(by_len)} script={dict(by_script)} label={dict(by_label)}",
        flush=True,
    )


def write_card(total, by_split, by_len, by_script, by_label, sub_counts, pair_counts) -> None:
    t = totals()
    lines = [
        "# SurakshaNet multi-class thread dataset (synthetic)",
        "",
        f"**{total:,} chat threads.** Predict the class of the **last message given the earlier messages**.",
        "",
        "## Layout",
        "",
        "```",
        "dataset/",
        "  len_03/  len_05/  len_07/  len_10/          threads of 3 / 5 / 7 / 10 messages",
        "    hindi_devanagari/  english/  hinglish/    script of the chat",
        "      0_benign/ 1_friendly_sarcasm/ 2_hostile_abuse/ 3_sexual_harassment/ 4_predatory_grooming/",
        "        train.jsonl  val.jsonl  test.jsonl",
        "  all/{train,val,test}.jsonl                 everything flattened",
        "  MANIFEST.csv                               counts per length x script x class x split",
        "```",
        "",
        "## Counts",
        "",
        f"- Splits: {dict(by_split)}",
        f"- By length: {dict(sorted(by_len.items()))}",
        f"- By script: {dict(by_script)}",
        f"- By class: {dict(by_label)}",
        "",
        "## Classes (research basis)",
        "",
        "| id | class | basis |",
        "|---:|---|---|",
        "| 0 | benign | HASOC NOT; includes hard negatives (parent-child, teacher, harmless surprise secrets, consensual flirting) |",
        "| 1 | friendly_sarcasm | Van Hee et al., sarcasm; affectionate roast |",
        "| 2 | hostile_abuse | Waseem et al. directed abuse; MACD abusive |",
        "| 3 | sexual_harassment | Van Hee et al.; Guest et al., EACL 2021; target is an adult woman (75%) or man (25%) |",
        "| 4 | predatory_grooming | Banko et al., WOAH 2020; target is a minor (age 10-15, girl or boy); secrecy, isolation, gifts, age probing only |",
        "",
        "## Record fields",
        "",
        "`thread_id, split, length, n_context, script, label, label_id, subtype, topic, audience, "
        "target_gender, child_age, pair_id, speakers, turns, last_turn, source`",
        "",
        "## Same-last-line pairs",
        "",
        "About half of the friendly_sarcasm and hostile_abuse threads end in an identical line "
        "(`pair_id` set, bank of 36 lines per script). A last-message-only model cannot separate them; "
        "only the earlier turns can. Pair lines are split by line, so no pair line appears in two splits.",
        "",
        "## Safety rules for generation",
        "",
        "- grooming threads contain no sexual content, body talk or photo requests (validated by regex);",
        "- sexual-harassment threads are suggestive but non-graphic (explicit terms rejected);",
        "- hostile threads contain no slurs about caste, religion, gender, ethnicity or disability (instructed).",
        "",
        "## Limits",
        "",
        "All text is synthetic (Qwen3-32B) with labels set by construction, then filtered by script, "
        "structure and explicit-term checks. It is not human-annotated. Treat benchmark numbers as an upper "
        "bound until a human-written test is added.",
        "",
        "## AWS cost evidence",
        "",
        f"Ledger at generation time: {t['n_calls']:,} calls, {t['input_tokens']:,} input / "
        f"{t['output_tokens']:,} output tokens, estimated USD {t['usd_estimated']:.2f}. "
        "See `../logs/cost_ledger.csv` and `../logs/bedrock_calls.jsonl`.",
        "",
        "## Subtype counts",
        "",
        "| class | subtype | n |",
        "|---|---|---:|",
    ]
    for (label, sub), n in sorted(sub_counts.items()):
        lines.append(f"| {label} | {sub} | {n} |")
    (DATASET / "DATASET_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=1750)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--overshoot", type=float, default=0.10)
    p.add_argument("--max-usd", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--model", default="qwen.qwen3-32b")
    p.add_argument("--backend", default="auto", choices=["auto", "mantle", "converse"])
    p.add_argument("--write-only", action="store_true")
    args = p.parse_args()
    if not args.write_only:
        generate(args)
    write_tree(args.per_cell)


if __name__ == "__main__":
    main()
