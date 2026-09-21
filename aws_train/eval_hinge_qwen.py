#!/usr/bin/env python3
"""Generate Qwen3 Hinglish on HinGE Hindi and score vs human refs / WAC / PAC.

This is the missing generation bake-off. MACD has no human Hinglish references,
so BLEU/WER/TER/NIST only make sense here. Abuse gates are OFF — HinGE is
parallel news/formal text, not MACD abuse comments.

Usage:
  .venv_codemix/bin/python -u aws_train/eval_hinge_qwen.py --workers 2
  .venv_codemix/bin/python -u aws_train/eval_hinge_qwen.py --score-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codemix_convert_bedrock import (  # noqa: E402
    DEVANAGARI,
    NON_ALNUM,
    REFUSAL_PATTERNS,
    BedrockClient,
    clean_output,
)
from eval_codemix_generation import (  # noqa: E402
    cmi_and_tags,
    english_dict,
    nlg_scores,
    parse_human_refs,
)

OUT_DIR = ROOT / "codemix_eval"
HINGE_CSV = OUT_DIR / "HinGE.csv"
QWEN_CSV = OUT_DIR / "hinge_qwen.csv"
QWEN_Q = OUT_DIR / "hinge_qwen_quarantine.csv"
REPORT = OUT_DIR / "hinge_qwen_report.json"

HINGE_SYSTEM = """You rewrite Hindi into natural Roman-script Hinglish code-mixing for NLP research.

Rules:
- Output Roman script only (Latin letters). No Devanagari.
- Mix Hindi and English naturally the way a bilingual Indian speaker would. Hindi is the matrix language; insert English words or short phrases where a bilingual would.
- Preserve the exact meaning. Do not add commentary, quotes, labels, or extra sentences.
- Output ONLY the rewritten Hinglish line."""

HINGE_SHOTS = [
    (
        "सर्वर से कनेक्ट कर रहा है, कृपया इंतजार करें...",
        "Server se connect kar raha hai, please wait...",
    ),
    (
        "उसने हिन्दुत्व के नियमों का प्रचार किया।",
        "Usne principles of hinduism ka prachaar kiya.",
    ),
    (
        "खेल के पास कोई हल नहीं है. वापस ले या फिर शुरू करें.",
        "Game ka koi solution nahi hai. Undo or start again.",
    ),
]

FIELDS = [
    "row_id",
    "english",
    "hindi",
    "qwen",
    "wac",
    "pac",
    "human_0",
]


def hinge_ok(hindi: str, hinglish: str) -> tuple[bool, str]:
    if not hinglish:
        return False, "empty"
    if REFUSAL_PATTERNS.search(hinglish):
        return False, "refusal"
    if len(DEVANAGARI.findall(hinglish)) / max(len(hinglish), 1) > 0.05:
        return False, "devanagari_remaining"
    src_tokens = max(len(NON_ALNUM.sub(" ", hindi).split()), 1)
    out_tokens = len(NON_ALNUM.sub(" ", hinglish).split())
    if out_tokens < max(2, int(0.35 * src_tokens)):
        return False, "collapsed_length"
    return True, "ok"


def load_done() -> set[str]:
    ids: set[str] = set()
    for path in (QWEN_CSV, QWEN_Q):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "row_id" in df.columns:
            ids |= set(df["row_id"].astype(str))
    return ids


def append_row(path: Path, row: dict, fields: list[str]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def generate(args: argparse.Namespace) -> None:
    df = pd.read_csv(HINGE_CSV)
    cols = {c.lower().strip(): c for c in df.columns}
    hi_c = cols["hindi"]
    en_c = cols["english"]
    wac_c = cols["wac"]
    pac_c = cols["pac"]
    hum_c = cols.get("human-generated hinglish") or cols.get("human")

    done = load_done()
    pending = []
    for i, row in df.iterrows():
        rid = f"hinge-{i}"
        if rid in done:
            continue
        pending.append((rid, row))
        if args.limit and len(pending) >= args.limit:
            break

    print(f"HinGE rows={len(df)} pending={len(pending)} skipped={len(done)} workers={args.workers}", flush=True)
    client0 = BedrockClient(args.region, args.model, backend="mantle")
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "by_fail": {}}
    thread_clients: dict[int, BedrockClient] = {}

    def get_client() -> BedrockClient:
        tid = threading.get_ident()
        if tid not in thread_clients:
            thread_clients[tid] = BedrockClient(args.region, args.model, backend="mantle")
        return thread_clients[tid]

    extra = []
    for src, tgt in HINGE_SHOTS:
        extra.append({"role": "user", "content": src})
        extra.append({"role": "assistant", "content": tgt})

    def one(item: tuple[str, pd.Series]) -> None:
        rid, row = item
        hindi = str(row[hi_c] or "")
        last = ""
        fail = "unknown"
        qwen = None
        cli = get_client()
        for attempt in range(args.max_retries + 1):
            user = hindi
            temp = 0.35
            if attempt:
                user = (
                    "CRITICAL: Output ONLY a Roman Hinglish line mixing Hindi+English. "
                    "No Devanagari, no commentary.\n\n" + hindi
                )
                temp = 0.5
            try:
                raw = cli.complete(
                    HINGE_SYSTEM,
                    user,
                    max_tokens=256,
                    temperature=temp,
                    extra_messages=extra,
                )
            except Exception as e:
                name = type(e).__name__
                msg = str(e)
                if "Auth" in name or "Unauthorized" in name or "expired" in msg.lower():
                    print(f"auth {rid}: {name}; refresh", flush=True)
                    try:
                        cli._token_ts = 0.0
                        cli._init_mantle()
                    except Exception as e2:
                        print(f"reinit failed {e2}", flush=True)
                    time.sleep(2)
                    continue
                if "Throttl" in name or "Too many tokens" in msg:
                    print(f"throttle {rid}; retry later", flush=True)
                    time.sleep(5)
                    return
                fail = f"api_error:{name}"
                time.sleep(1)
                continue
            last = clean_output(raw)
            ok, fail = hinge_ok(hindi, last)
            if ok:
                qwen = last
                break
            time.sleep(0.1)

        humans = parse_human_refs(row[hum_c]) if hum_c else []
        rec = {
            "row_id": rid,
            "english": row[en_c],
            "hindi": hindi,
            "qwen": qwen or "",
            "wac": row[wac_c],
            "pac": row[pac_c],
            "human_0": humans[0] if humans else "",
            "fail_reason": "ok" if qwen else fail,
            "last_output": last,
        }
        with lock:
            if qwen:
                append_row(QWEN_CSV, rec, FIELDS)
                stats["ok"] += 1
            else:
                append_row(QWEN_Q, rec, FIELDS + ["fail_reason", "last_output"])
                stats["fail"] += 1
                stats["by_fail"][fail] = stats["by_fail"].get(fail, 0) + 1
            n = stats["ok"] + stats["fail"]
            if n % 25 == 0:
                print(f"progress ok={stats['ok']} fail={stats['fail']}", flush=True)
        time.sleep(args.sleep)

    if args.workers <= 1:
        for item in pending:
            one(item)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, pending, chunksize=1))
    print(json.dumps(stats, indent=2), flush=True)


def score() -> dict:
    if not QWEN_CSV.exists():
        raise SystemExit(f"missing {QWEN_CSV} — run generation first")
    hinge = pd.read_csv(HINGE_CSV)
    qwen = pd.read_csv(QWEN_CSV)
    cols = {c.lower().strip(): c for c in hinge.columns}
    hum_c = cols.get("human-generated hinglish") or cols.get("human")
    wac_c = cols["wac"]
    pac_c = cols["pac"]
    hi_c = cols["hindi"]

    # Align Qwen rows back to HinGE index via row_id hinge-N
    qmap = {}
    for _, r in qwen.iterrows():
        rid = str(r["row_id"])
        try:
            qmap[int(rid.split("-")[1])] = str(r["qwen"] or "")
        except (IndexError, ValueError):
            continue

    refs, wacs, pacs, qwens, keep_idx = [], [], [], [], []
    for i, row in hinge.iterrows():
        hyp = qmap.get(int(i))
        if not hyp:
            continue
        humans = parse_human_refs(row[hum_c])
        if not humans:
            continue
        refs.append(humans)
        wacs.append(str(row[wac_c] or ""))
        pacs.append(str(row[pac_c] or ""))
        qwens.append(hyp)
        keep_idx.append(int(i))

    en = english_dict()
    out = {
        "n_scored": len(qwens),
        "n_qwen_accepted": int(len(qwen)),
        "n_qwen_quarantine": int(len(pd.read_csv(QWEN_Q))) if QWEN_Q.exists() else 0,
        "Human_CMI": round(
            sum(cmi_and_tags(r[0], en)["cmi"] for r in refs) / max(len(refs), 1), 3
        ),
        "WAC": {**nlg_scores(wacs, refs), "mean_cmi": round(sum(cmi_and_tags(x, en)["cmi"] for x in wacs) / max(len(wacs), 1), 3)},
        "PAC": {**nlg_scores(pacs, refs), "mean_cmi": round(sum(cmi_and_tags(x, en)["cmi"] for x in pacs) / max(len(pacs), 1), 3)},
        "Qwen3": {**nlg_scores(qwens, refs), "mean_cmi": round(sum(cmi_and_tags(x, en)["cmi"] for x in qwens) / max(len(qwens), 1), 3)},
    }
    REPORT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--model", default="qwen.qwen3-32b")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        if not HINGE_CSV.exists():
            from eval_codemix_generation import download_hinge

            download_hinge(HINGE_CSV)
        generate(args)
    score()


if __name__ == "__main__":
    main()
