#!/usr/bin/env python3
"""Convert MACD Hindi → natural Hinglish code-mixing via Bedrock Mantle.

NOT script transliteration. Abuse-preserving gates:
  1) refusal / length / Devanagari / abuse-lexicon checks
  2) ONNX MACD scorer delta gate (hinglish_p_abuse >= hindi_p_abuse - delta)

Usage:
  .venv_codemix/bin/python aws_train/codemix_convert_bedrock.py --pilot
  .venv_codemix/bin/python aws_train/codemix_convert_bedrock.py --full --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "codemix_hinglish"
DEFAULT_SCORER = ROOT / "assets" / "models" / "custom-macd-model"
MACD_URLS = {
    "train": "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/hindi_train.csv",
    "val": "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/hindi_val.csv",
    "test": "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/hindi_test.csv",
}

ABUSE_STEMS: list[tuple[str, list[str]]] = [
    ("चूतिय", ["chutiya", "chootiya", "chutya", "chootiye", "chutiye"]),
    ("चुत", ["chut", "choot"]),
    ("मादरचोद", ["madarchod", "mc", "motherchod", "maderchod"]),
    ("भोसड", ["bhosd", "bhosad", "bhosdi", "bhosdiwala"]),
    ("भोसड़", ["bhosd", "bhosad", "bhosdi"]),
    ("रंड", ["randi", "rundi"]),
    ("हराम", ["haram", "haraam", "harami"]),
    ("गांड", ["gand", "gaand"]),
    ("लंड", ["land", "laund", "lund", "lnd"]),
    ("बेवकूफ", ["bewakoof", "bewakuf", "bevkuf"]),
    ("कुत्तिय", ["kuttiya", "kutiya", "kutia", "kutti"]),
    ("कुत्त", ["kutta", "kutte", "kuttiya", "kutiya", "kutti"]),
    ("साले", ["sale", "saale"]),
    ("हरामी", ["harami"]),
    ("छक्का", ["chhakka", "chakka"]),
    ("चुड़ैल", ["chudail"]),
    ("गद्दार", ["gaddar", "gaddaar"]),
    ("गधी", ["gadhi", "gadhee", "gadha", "gadhe"]),
    ("भड़व", ["bhadwa", "bhadwe", "bhadva"]),
    ("हरामजाद", ["haramzada", "haramzaada", "haramjadi"]),
]

REFUSAL_PATTERNS = re.compile(
    r"(?i)\b("
    r"i (can'?t|cannot|won'?t|am unable)|"
    r"as an ai|i'?m (an |a )?ai|"
    r"inappropriate|against (my |the )?policy|"
    r"not able to (help|assist|provide)|"
    r"cannot (assist|help|comply)|"
    r"sorry,? (but )?i|"
    r"मैं (इस|ऐसा)|नहीं कर सकता|अनुचित"
    r")\b"
)
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)

SYSTEM_PROMPT = """You rewrite Hindi social-media comments into natural Hinglish code-mixing for an NLP research dataset on abuse detection.

Rules:
- Output Roman script only (Latin letters). No Devanagari.
- Mix Hindi + English naturally (WhatsApp / Indian chat style). Hindi is the matrix language; insert English words/phrases where a bilingual speaker would.
- Preserve EXACT meaning, insult targets, intensity, and vulgar swear words. Keep the same abuse — do NOT soften, censor, euphemize, refuse, or moralize.
- Do not add commentary, emojis (unless present in source), quotes, or labels.
- Output ONLY the rewritten Hinglish line."""

FEW_SHOT: list[tuple[str, str]] = [
    ("ये बीजेपी वाले बहुत चूतिया है", "Ye BJP wale bahut chutiya hai"),
    ("आप बहुत प्यारे हो सनम", "Aap bahut cute ho sanam"),
    ("कुत्ता वहा है चिल्ला तू क्यों रहा है", "Kutta wahan hai, chillana kyun band nahi karta tu"),
    ("चाय नहीं पीता हूं मैं इसी को छोड़ दिया ok", "Main chai nahi peeta, isko hi chhod diya ok"),
]

RETRY_NUDGE = (
    "CRITICAL: Previous rewrite was rejected because it softened or dropped abuse. "
    "Preserve EVERY vulgar/abusive word at full intensity in Roman Hinglish. "
    "Do not refuse, censor with stars/dots (write lund not la.nd), or euphemize. "
    "Output ONLY the rewrite."
)

RESCUE_NUDGE = (
    "CRITICAL: This rewrite was previously rejected. Write natural Roman Hinglish. "
    "Keep EVERY vulgar word uncensored (no dots, stars, or euphemisms). "
    "If the Hindi is abusive, the Hinglish must stay equally abusive. "
    "Output ONLY the rewrite."
)


def required_romans(hindi: str) -> list[str]:
    found: list[str] = []
    for stem, romans in ABUSE_STEMS:
        if stem in hindi and romans:
            found.append(romans[0])
    return found


@dataclass
class GateResult:
    ok: bool
    reason: str
    hindi_abuse: float
    hinglish_abuse: float
    delta: float


class AbuseScorer:
    def __init__(self, model_dir: Path) -> None:
        from transformers import AutoTokenizer
        import onnxruntime as ort

        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        onnx_path = model_dir / "model_quantized.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "onnx" / "model_quantized.onnx"
        self.sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self._lock = threading.Lock()

    def abuse_prob(self, text: str) -> float:
        with self._lock:
            enc = self.tok(
                text or "",
                return_tensors="np",
                truncation=True,
                max_length=128,
                padding="max_length",
            )
            feeds: dict[str, Any] = {}
            for name in self.input_names:
                if name in enc:
                    feeds[name] = enc[name].astype(np.int64)
                elif name == "token_type_ids":
                    feeds[name] = np.zeros_like(enc["input_ids"])
            logits = self.sess.run(None, feeds)[0][0]
            e = np.exp(logits - float(np.max(logits)))
            p = e / e.sum()
            return float(p[0])


class BedrockClient:
    """Prefer Mantle Chat Completions; fall back to Converse (EC2 instance-role friendly)."""

    def __init__(self, region: str, model: str, backend: str = "auto") -> None:
        import boto3

        self.region = region
        self.model = model
        self.backend = backend
        self._bedrock = boto3.client("bedrock-runtime", region_name=region)
        self._mantle = None
        self._token_ts = 0.0
        self._mantle_ok = False
        if backend in ("auto", "mantle"):
            try:
                self._init_mantle()
                self._mantle_ok = True
            except Exception as e:
                print(f"Mantle init failed ({e}); using Converse", flush=True)
                self.backend = "converse"

    def _init_mantle(self) -> None:
        from aws_bedrock_token_generator import provide_token
        from openai import OpenAI

        token = provide_token(region=self.region)
        self._token_ts = time.time()
        self._mantle = OpenAI(
            api_key=token,
            base_url=f"https://bedrock-mantle.{self.region}.api.aws/v1",
        )

    def _refresh_mantle(self) -> None:
        if time.time() - self._token_ts > 480:
            self._init_mantle()

    def _build_user(self, hindi: str, nudge: bool, rescue: bool = False) -> tuple[str, float]:
        if nudge or rescue:
            must = required_romans(hindi)
            must_line = (
                f" MUST keep these Roman swear/abuse tokens verbatim: {', '.join(must)}."
                if must
                else ""
            )
            prefix = RESCUE_NUDGE if rescue else RETRY_NUDGE
            return f"{prefix}{must_line}\n\n{hindi}", 0.55
        return hindi, 0.35

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.35,
        extra_messages: list[dict] | None = None,
    ) -> str:
        use_mantle = self.backend == "mantle" or (
            self.backend == "auto" and self._mantle_ok and self._mantle is not None
        )
        if use_mantle:
            for auth_try in range(2):
                try:
                    self._refresh_mantle()
                    messages = [{"role": "system", "content": system}]
                    if extra_messages:
                        messages.extend(extra_messages)
                    messages.append({"role": "user", "content": user})
                    assert self._mantle is not None
                    resp = self._mantle.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    content = resp.choices[0].message.content
                    return "" if content is None else content.strip().strip('"').strip("'")
                except Exception as e:
                    name = type(e).__name__
                    authish = "Auth" in name or "Unauthorized" in name or "expired" in str(e).lower()
                    if authish and auth_try == 0:
                        print(f"Mantle auth ({name}); refreshing token", flush=True)
                        self._token_ts = 0.0
                        try:
                            self._init_mantle()
                            self._mantle_ok = True
                            continue
                        except Exception as e2:
                            print(f"Mantle re-init failed ({e2})", flush=True)
                    print(f"Mantle call failed ({name}); falling back to Converse", flush=True)
                    self._mantle_ok = False
                    self.backend = "converse"
                    break

        converse_model = self.model
        if converse_model.startswith(("qwen.", "google.", "mistral.")) or "/" in converse_model:
            converse_model = "meta.llama3-8b-instruct-v1:0"
        resp = self._bedrock.converse(
            modelId=converse_model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        return resp["output"]["message"]["content"][0]["text"].strip().strip('"').strip("'")

    def rewrite(self, hindi: str, nudge: bool = False, rescue: bool = False) -> str:
        user, temp = self._build_user(hindi, nudge, rescue=rescue)
        extra = []
        for src, tgt in FEW_SHOT:
            extra.append({"role": "user", "content": src})
            extra.append({"role": "assistant", "content": tgt})
        return self.complete(
            SYSTEM_PROMPT, user, max_tokens=256, temperature=temp, extra_messages=extra
        )


MantleClient = BedrockClient



def clean_output(text: str) -> str:
    text = (text or "").strip()
    for prefix in ("Hinglish:", "Rewrite:", "Output:", "Here:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'")
        if line:
            return line
    return ""


def fold_roman(text: str) -> str:
    """Strip obfuscation punctuation so la.nd / l@nd still match lund/land."""
    s = (text or "").lower()
    return re.sub(r"[.\-_*@]+", "", s)


def lexicon_preserved(hindi: str, hinglish: str) -> bool:
    h_low = hinglish.lower()
    h_fold = fold_roman(hinglish)
    required = False
    for stem, romans in ABUSE_STEMS:
        if stem in hindi:
            required = True
            if any(r in h_low or fold_roman(r) in h_fold for r in romans):
                return True
    return not required


def hard_checks(hindi: str, hinglish: str) -> tuple[bool, str]:
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
    if not lexicon_preserved(hindi, hinglish):
        return False, "abuse_lexicon_missing"
    return True, "ok"


def score_gate(
    scorer: AbuseScorer,
    hindi: str,
    hinglish: str,
    label: int,
    delta: float,
) -> GateResult:
    h = scorer.abuse_prob(hindi)
    g = scorer.abuse_prob(hinglish)
    d = h - g
    if label == 0:
        ok = g >= (h - delta)
        reason = "ok" if ok else "abuse_score_drop"
    else:
        # Non-abusive: reject rewrites that spike into abuse (previous hole).
        spiked = g >= 0.5 and g > (h + delta)
        ok = not spiked
        reason = "ok" if ok else "abuse_score_spike"
    return GateResult(ok=ok, reason=reason, hindi_abuse=h, hinglish_abuse=g, delta=d)


def load_macd_split(split: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"hindi_{split}.csv"
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(MACD_URLS[split])
        df.to_csv(path, index=False)
    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    df["row_id"] = [f"{split}-{i}" for i in range(len(df))]
    df["split"] = split
    return df


def stratified_pilot(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    parts = []
    per = max(1, n // 2)
    for lab in (0, 1):
        pool = df[df["label"] == lab]
        idx = list(pool.index)
        rng.shuffle(idx)
        parts.append(pool.loc[idx[:per]])
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _rewrite_with_throttle_retry(
    client: MantleClient, hindi: str, nudge: bool, rescue: bool = False
) -> str:
    """Retry transient Bedrock throttles; do not treat as permanent failure."""
    delay = 1.0
    last_err: Exception | None = None
    for i in range(8):
        try:
            return client.rewrite(hindi, nudge=nudge, rescue=rescue)
        except Exception as e:
            name = type(e).__name__
            msg = str(e)
            transient = (
                "Throttl" in name
                or "Throttl" in msg
                or "Too many tokens" in msg
                or "Rate exceeded" in msg
                or "ServiceUnavailable" in name
                or "timeout" in msg.lower()
            )
            if not transient:
                raise
            last_err = e
            print(f"throttle/backoff {i+1}/8 sleep={delay:.1f}s ({name})", flush=True)
            time.sleep(delay)
            delay = min(delay * 1.8, 45.0)
    raise last_err  # type: ignore[misc]


def convert_row(
    client: MantleClient,
    scorer: AbuseScorer,
    hindi: str,
    label: int,
    delta: float,
    max_retries: int,
    rescue: bool = False,
) -> tuple[str | None, GateResult, int, str, str]:
    last_gate = GateResult(False, "no_attempt", 0.0, 0.0, 0.0)
    fail = "unknown"
    last_out = ""
    for attempt in range(max_retries + 1):
        raw = _rewrite_with_throttle_retry(
            client, hindi, nudge=attempt > 0 or rescue, rescue=rescue
        )
        hinglish = clean_output(raw)
        last_out = hinglish
        ok_hard, hard_reason = hard_checks(hindi, hinglish)
        if not ok_hard:
            fail = hard_reason
            try:
                g = score_gate(scorer, hindi, hinglish or " ", label, delta)
                last_gate = GateResult(
                    False, hard_reason, g.hindi_abuse, g.hinglish_abuse, g.delta
                )
            except Exception:
                last_gate = GateResult(False, hard_reason, 0.0, 0.0, 0.0)
            time.sleep(0.1)
            continue
        gate = score_gate(scorer, hindi, hinglish, label, delta)
        last_gate = gate
        if gate.ok:
            return hinglish, gate, attempt + 1, "ok", hinglish
        fail = gate.reason
        time.sleep(0.1)
    return None, last_gate, max_retries + 1, fail, last_out


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "row_id" not in df.columns:
        return set()
    return set(df["row_id"].astype(str))


def run_conversion(
    rows: pd.DataFrame,
    out_dir: Path,
    client: MantleClient,
    scorer: AbuseScorer,
    delta: float,
    max_retries: int,
    sleep_s: float,
    tag: str,
    workers: int = 1,
    rescue: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = out_dir / f"{tag}.csv"
    quarantine_path = out_dir / f"{tag}_quarantine.csv"
    meta_path = out_dir / f"{tag}_meta.jsonl"

    accepted_fields = [
        "row_id", "split", "label", "text", "hindi_text",
        "hindi_abuse", "hinglish_abuse", "score_delta", "attempts",
    ]
    quarantine_fields = accepted_fields + ["fail_reason", "last_output"]

    done = load_done_ids(accepted_path) | load_done_ids(quarantine_path)
    stats = {
        "total": len(rows),
        "skipped_done": 0,
        "accepted": 0,
        "quarantined": 0,
        "by_fail": {},
        "abusive_accepted": 0,
        "abusive_total": 0,
        "score_deltas": [],
    }
    lock = threading.Lock()
    thread_clients: dict[int, MantleClient] = {}

    def get_client() -> MantleClient:
        tid = threading.get_ident()
        if tid not in thread_clients:
            thread_clients[tid] = MantleClient(client.region, client.model, backend=getattr(client, "backend", "auto"))
        return thread_clients[tid]

    def process_one(r: pd.Series) -> None:
        hindi = str(r["text"])
        label = int(r["label"])
        row_id = str(r["row_id"])
        last_out = ""
        try:
            hinglish, gate, attempts, fail, last_out = convert_row(
                get_client(), scorer, hindi, label, delta, max_retries, rescue=rescue
            )
        except Exception as e:
            ename = type(e).__name__
            emsg = str(e)
            if "Throttl" in ename or "Throttl" in emsg or "Too many tokens" in emsg:
                # Do NOT quarantine throttles — leave row undone for next chunk pass
                print(f"skip-quarantine throttle row={row_id}: {ename}", flush=True)
                time.sleep(5.0)
                return
            hinglish, gate, attempts, fail, last_out = (
                None,
                GateResult(False, "api_error", 0.0, 0.0, 0.0),
                0,
                f"api_error:{ename}",
                "",
            )
            time.sleep(1.0)

        base = {
            "row_id": row_id,
            "split": r["split"],
            "label": label,
            "hindi_text": hindi,
            "hindi_abuse": round(gate.hindi_abuse, 4),
            "hinglish_abuse": round(gate.hinglish_abuse, 4),
            "score_delta": round(gate.delta, 4),
            "attempts": attempts,
        }
        meta = {**base, "fail_reason": fail, "ok": hinglish is not None, "last_output": last_out}
        with lock:
            with meta_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            if label == 0:
                stats["abusive_total"] += 1
            if hinglish is not None:
                append_csv(accepted_path, {**base, "text": hinglish}, accepted_fields)
                stats["accepted"] += 1
                stats["score_deltas"].append(gate.delta)
                if label == 0:
                    stats["abusive_accepted"] += 1
            else:
                append_csv(
                    quarantine_path,
                    {**base, "text": "", "fail_reason": fail, "last_output": last_out},
                    quarantine_fields,
                )
                stats["quarantined"] += 1
                stats["by_fail"][fail] = stats["by_fail"].get(fail, 0) + 1
            done_n = stats["accepted"] + stats["quarantined"]
            if done_n % 50 == 0:
                print(
                    f"[{tag}] progress accepted={stats['accepted']} "
                    f"quarantine={stats['quarantined']}",
                    flush=True,
                )
        if sleep_s:
            time.sleep(sleep_s)

    pending = []
    for _, r in rows.iterrows():
        if str(r["row_id"]) in done:
            stats["skipped_done"] += 1
        else:
            pending.append(r)

    workers = max(1, int(workers))
    print(
        f"[{tag}] pending={len(pending)} skipped_done={stats['skipped_done']} workers={workers}",
        flush=True,
    )
    if workers == 1:
        for r in pending:
            process_one(r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(process_one, pending, chunksize=1))

    abusive_rate = (
        stats["abusive_accepted"] / stats["abusive_total"]
        if stats["abusive_total"]
        else None
    )
    report = {
        "tag": tag,
        "model": client.model,
        "region": client.region,
        "delta": delta,
        "workers": workers,
        "accepted": stats["accepted"],
        "quarantined": stats["quarantined"],
        "skipped_done": stats["skipped_done"],
        "abusive_accept_rate": abusive_rate,
        "mean_score_delta": (
            float(np.mean(stats["score_deltas"])) if stats["score_deltas"] else None
        ),
        "by_fail": stats["by_fail"],
        "accepted_path": str(accepted_path),
        "quarantine_path": str(quarantine_path),
    }
    (out_dir / f"{tag}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--pilot-n", type=int, default=300)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--model", default="qwen.qwen3-32b")
    p.add_argument("--backend", default="auto", choices=["auto", "mantle", "converse"])
    p.add_argument("--delta", type=float, default=0.15)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="Max new rows per split (0=all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--scorer", type=Path, default=DEFAULT_SCORER)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "aws_train" / "macd_hindi_cache")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    p.add_argument(
        "--from-quarantine",
        type=Path,
        default=None,
        help="Reprocess rows from a quarantine CSV",
    )
    p.add_argument(
        "--rescue",
        action="store_true",
        help="Use stronger abuse-preserve nudge (quarantine rescue)",
    )
    args = p.parse_args()

    print("Loading abuse scorer...", flush=True)
    scorer = AbuseScorer(args.scorer)
    print(f"Connecting Mantle region={args.region} model={args.model}", flush=True)
    client = MantleClient(args.region, args.model, backend=args.backend)

    if args.from_quarantine:
        qdf = pd.read_csv(args.from_quarantine)
        for c in ["row_id", "split", "label", "hindi_text"]:
            if c not in qdf.columns:
                raise SystemExit(f"quarantine missing {c}")
        qdf = qdf[["row_id", "split", "label", "hindi_text"]].rename(columns={"hindi_text": "text"})
        tag = "pilot" if args.pilot else "reprocess"
        out_sub = args.out / ("pilot" if args.pilot else "reprocess")
        qpath = out_sub / f"{tag}_quarantine.csv"
        if qpath.exists():
            old = pd.read_csv(qpath)
            old = old[~old["row_id"].astype(str).isin(qdf["row_id"].astype(str))]
            old.to_csv(qpath, index=False)
        run_conversion(
            qdf, out_sub, client, scorer, args.delta, args.max_retries, args.sleep,
            tag, args.workers, rescue=True,
        )
        return

    frames = [load_macd_split(s, args.cache_dir) for s in args.splits]
    all_df = pd.concat(frames, ignore_index=True)

    if args.pilot:
        train = all_df[all_df["split"] == "train"]
        sample = stratified_pilot(train if len(train) else all_df, args.pilot_n, args.seed)
        run_conversion(
            sample, args.out / "pilot", client, scorer, args.delta,
            args.max_retries, args.sleep, "pilot", args.workers,
        )
    else:
        for split in args.splits:
            part = all_df[all_df["split"] == split].reset_index(drop=True)
            if args.limit and args.limit > 0:
                done = load_done_ids(args.out / f"{split}.csv") | load_done_ids(
                    args.out / f"{split}_quarantine.csv"
                )
                part = part[~part["row_id"].astype(str).isin(done)].head(args.limit).reset_index(drop=True)
                print(f"{split}: converting up to {len(part)} new rows", flush=True)
            run_conversion(
                part, args.out, client, scorer, args.delta,
                args.max_retries, args.sleep, split, args.workers,
            )


if __name__ == "__main__":
    main()
