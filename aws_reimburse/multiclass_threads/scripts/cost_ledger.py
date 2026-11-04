"""Append-only AWS cost log for reimbursement. No call without a row.

Every call is appended to logs/bedrock_calls.jsonl immediately. The summary
CSV is rebuilt from memory every FLUSH_EVERY calls and at process exit.
"""

from __future__ import annotations

import atexit
import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
CALLS = LOG_DIR / "bedrock_calls.jsonl"
LEDGER = LOG_DIR / "cost_ledger.csv"
FLUSH_EVERY = 200

# us-east-1 standard on-demand, AWS price list observed 2026-11-02.
USD_PER_M = {
    "qwen.qwen3-32b": (0.15, 0.60),
    "qwen.qwen3-32b-v1:0": (0.15, 0.60),
    "meta.llama3-8b-instruct-v1:0": (0.30, 0.60),
    "local": (0.0, 0.0),
}

_LOCK = threading.Lock()
_AGG: dict[tuple[str, str, str, str], dict] = {}
_TOT = {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "usd_estimated": 0.0}
_LOADED = False
_PENDING = 0


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inn, out = USD_PER_M.get(model, (0.15, 0.60))
    return (input_tokens * inn + output_tokens * out) / 1_000_000.0


def _ingest(rec: dict) -> None:
    key = (rec["ts_utc"][:10], rec["job"], rec["service"], rec["model"])
    row = _AGG.setdefault(
        key, {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "usd_estimated": 0.0}
    )
    inn = int(rec.get("input_tokens") or 0)
    out = int(rec.get("output_tokens") or 0)
    usd = float(rec.get("usd_estimated") or 0)
    row["n_calls"] += 1
    row["input_tokens"] += inn
    row["output_tokens"] += out
    row["usd_estimated"] += usd
    _TOT["n_calls"] += 1
    _TOT["input_tokens"] += inn
    _TOT["output_tokens"] += out
    _TOT["usd_estimated"] += usd


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    if CALLS.exists():
        for line in CALLS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                _ingest(json.loads(line))
    _LOADED = True


def _write_ledger_unlocked() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "date_utc",
                "job",
                "service",
                "model",
                "n_calls",
                "input_tokens",
                "output_tokens",
                "usd_estimated",
                "notes",
            ]
        )
        for (day, job, service, model), row in sorted(_AGG.items()):
            w.writerow(
                [
                    day,
                    job,
                    service,
                    model,
                    row["n_calls"],
                    row["input_tokens"],
                    row["output_tokens"],
                    f"{row['usd_estimated']:.6f}",
                    "",
                ]
            )


def flush() -> None:
    with _LOCK:
        _load()
        _write_ledger_unlocked()


atexit.register(flush)


def log_call(
    *,
    job: str,
    service: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    request_id: str = "",
    ok: bool = True,
    notes: str = "",
    usd_override: float | None = None,
) -> dict:
    global _PENDING
    rec = {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job": job,
        "service": service,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "usd_estimated": round(
            usd_override if usd_override is not None else estimate_usd(model, input_tokens, output_tokens), 8
        ),
        "request_id": request_id or "",
        "ok": bool(ok),
        "notes": notes,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        _load()
        with CALLS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _ingest(rec)
        _PENDING += 1
        if _PENDING >= FLUSH_EVERY:
            _write_ledger_unlocked()
            _PENDING = 0
    return rec


def totals() -> dict:
    with _LOCK:
        _load()
        return {
            "n_calls": _TOT["n_calls"],
            "input_tokens": _TOT["input_tokens"],
            "output_tokens": _TOT["output_tokens"],
            "usd_estimated": round(_TOT["usd_estimated"], 6),
        }
