#!/usr/bin/env python3
"""Benchmark CodeMixBench's isolated protocol against a local rolling-window
KV cache on toxic Hinglish narrative.

Paper: Yang & Chai, CodeMixBench, EMNLP 2025.
  M-index  = (1 - sum_j p_j^2) / ((k - 1) * sum_j p_j^2)
  I-index  = (number of language switches) / (n - 1)
  k = 2 for Hindi–English. Universal tokens (punctuation, emoji, digits)
  are dropped before both indices. The paper keeps a sentence when
  M >= 0.1 and I >= 0.1, and it scores each instance with no dialogue state.

Arms
  codemixbench_isolated  current turn only (paper protocol)
  llm_window             same features on the concatenated rolling window
                         (how an LLM prompt sees the chat)
  kv_cache               8-slot dynamic key/value cache, attention readout
  kv_contrastive         same cache, BCE + supervised contrastive loss
  shipped_int8           deployed MiniLM INT8, current turn, threshold 0.5
  shipped_int8_window    same encoder on the concatenated window

Train/val come from codemix_hinglish/train.csv (80/20). Test is frozen
codemix_hinglish/test.csv. Thresholds for the learned arms are chosen on
val. The test set is scored once.

Usage (from repo root):
  python3 aws_train/benchmark_narrative_kv.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aws_train"))

from eval_codemix_generation import (  # noqa: E402
    DEVANAGARI_RE,
    EN_FORCE,
    HINDI_CLOSED,
    is_independent,
    tokenize,
)

SEED = 42
DIM = 16
MAX_SLOTS = 8
MAX_WINDOW_TOKENS = 96
N_ABUSE_STEMS = 12
N_BENIGN_STEMS = 6
PAPER_M_MIN = 0.1
PAPER_I_MIN = 0.1

TRAIN_CSV = ROOT / "codemix_hinglish" / "train.csv"
TEST_CSV = ROOT / "codemix_hinglish" / "test.csv"
MODEL_DIR = ROOT / "assets" / "models" / "custom-macd-model"
OUT_JSON = ROOT / "codemix_eval" / "narrative_kv_benchmark.json"
OUT_WEIGHTS = ROOT / "assets" / "narrative" / "kv_weights.json"
OUT_PARITY = ROOT / "codemix_eval" / "narrative_parity_cases.json"

# Transcribed from Yang & Chai, EMNLP 2025, Table 2 (one-shot accuracy).
PUBLISHED_TABLE2 = {
    "source": "Yang & Chai, CodeMixBench, EMNLP 2025, Table 2, one-shot accuracy",
    "CM-MMLU": {
        "GPT-4o": {"en_only": 85.60, "hi-en": 82.13, "average_code_mixed": 80.97},
        "GPT-4 Turbo": {"en_only": 83.10, "hi-en": 77.83, "average_code_mixed": 76.47},
        "LLaMA3-70B-Instruct": {"en_only": 77.20, "hi-en": 74.61, "average_code_mixed": 72.79},
        "LLaMA3-8B-Instruct": {"en_only": 65.60, "hi-en": 56.93, "average_code_mixed": 54.99},
        "Mistral-7B": {"en_only": 55.30, "hi-en": 40.82, "average_code_mixed": 44.85},
    },
    "CM-GSM8K": {
        "GPT-4 Turbo": {"en_only": 95.23, "hi-en": 93.60, "average_code_mixed": 93.42},
        "GPT-4o": {"en_only": 92.50, "hi-en": 89.57, "average_code_mixed": 90.30},
        "LLaMA3-70B-Instruct": {"en_only": 93.91, "hi-en": 90.45, "average_code_mixed": 90.61},
    },
    "CM-TruthfulQA": {
        "GPT-4 Turbo": {"en_only": 83.72, "hi-en": 81.11, "average_code_mixed": 80.49},
        "GPT-4o": {"en_only": 81.76, "hi-en": 78.47, "average_code_mixed": 77.59},
        "LLaMA3-70B-Instruct": {"en_only": 70.87, "hi-en": 66.31, "average_code_mixed": 67.41},
    },
    "paper_stated_drop_vs_english_only_points": {
        "zh-en": 2.63,
        "hi-en": 3.47,
        "ar-en": 5.25,
        "bn-en": 7.32,
        "mr-en": 7.62,
        "ne-en": 8.90,
        "ta-en": 14.83,
        "note": "Section 4.3. Cross-family mixes drop more than Germanic/Romance mixes.",
    },
}


@dataclass
class Example:
    turns: list[str]
    y: int  # 1 = abusive
    kind: str
    slur_in_last: bool | None = None


def lang_tag(tok: str, hindi: set[str], english: set[str]) -> str:
    """Word-level tag used by CodeMixBench's M-index and I-index.

    Roman tokens that are not in the English list are tagged Hindi. That
    matches this repo's Hinglish tagger when the system dictionary is not
    used, so the browser cache can reproduce the same tags.
    """
    if is_independent(tok):
        return "univ"
    if DEVANAGARI_RE.search(tok):
        return "hi"
    w = tok.lower()
    if w in hindi:
        return "hi"
    if w in english:
        return "en"
    return "hi"


def language_tokens(text: str, hindi: set[str], english: set[str]) -> list[str]:
    return [lang_tag(t, hindi, english) for t in tokenize(text)]


def m_index(tags: list[str]) -> float:
    """Barnett M-index as printed in CodeMixBench §3.4. Range [0, 1]."""
    langs = [t for t in tags if t in ("hi", "en")]
    if not langs:
        return 0.0
    p_hi = sum(t == "hi" for t in langs) / len(langs)
    p_en = 1.0 - p_hi
    sum_sq = p_hi * p_hi + p_en * p_en
    denom = (2 - 1) * sum_sq
    if denom <= 0:
        return 0.0
    return float((1.0 - sum_sq) / denom)


def i_index(tags: list[str]) -> float:
    """Guzmán I-index: fraction of adjacent language tokens that switch."""
    langs = [t for t in tags if t in ("hi", "en")]
    if len(langs) < 2:
        return 0.0
    switches = sum(a != b for a, b in zip(langs, langs[1:]))
    return float(switches / (len(langs) - 1))


def content_tokens(text: str) -> list[str]:
    out = []
    for tok in tokenize(text):
        if is_independent(tok):
            continue
        out.append(tok.lower())
    return out


def mine_stems(texts: list[str], abusive: np.ndarray) -> tuple[list[str], list[str]]:
    """Document log-odds on the train split. Abuse stems rank positive."""
    c_ab: Counter[str] = Counter()
    c_be: Counter[str] = Counter()
    n_ab = int(abusive.sum())
    n_be = int(len(abusive) - n_ab)
    for text, y in zip(texts, abusive):
        bag = c_ab if y else c_be
        for tok in set(content_tokens(text)):
            if len(tok) < 4 or tok in EN_FORCE:
                continue
            bag[tok] += 1
    scored = []
    prior = 0.5
    for tok in set(c_ab) | set(c_be):
        a = c_ab[tok]
        b = c_be[tok]
        if a + b < 20:
            continue
        if a / max(n_ab, 1) > 0.35 and b / max(n_be, 1) > 0.35:
            continue
        lo = math.log((a + prior) / (n_ab + prior)) - math.log((b + prior) / (n_be + prior))
        scored.append((lo, tok, a, b))
    abuse = [tok for lo, tok, _, _ in sorted(scored, reverse=True) if lo > 0][:N_ABUSE_STEMS]
    benign = [tok for lo, tok, _, _ in sorted(scored) if lo < 0][:N_BENIGN_STEMS]
    return abuse, benign


def stem_hit(tokens: list[str], stem: str) -> bool:
    for tok in tokens:
        if tok == stem:
            return True
        if len(stem) >= 4 and tok.startswith(stem) and len(tok) - len(stem) <= 2:
            return True
    return False


def featurize(
    text: str,
    abuse_stems: list[str],
    benign_stems: list[str],
    hindi: set[str],
    english: set[str],
) -> np.ndarray:
    tags = language_tokens(text, hindi, english)
    toks = content_tokens(text)
    n = max(len(toks), 1)
    hi = sum(t == "hi" for t in tags)
    en = sum(t == "en" for t in tags)
    lang_n = hi + en
    abuse_hits = sum(stem_hit(toks, s) for s in abuse_stems)
    switches = 0
    prev = None
    for t in tags:
        if t not in ("hi", "en"):
            continue
        if prev is not None and t != prev:
            switches += 1
        prev = t
    dev = sum(1 for t in toks if DEVANAGARI_RE.search(t))
    raw = tokenize(text)
    qmark = 1.0 if any(t == "?" for t in raw) else 0.0
    exclaim = 1.0 if any(t == "!" for t in raw) else 0.0
    feats = [
        m_index(tags),
        i_index(tags),
        (en / lang_n) if lang_n else 0.0,
        (hi / lang_n) if lang_n else 0.0,
        dev / n,
        min(abuse_hits, 3) / 3.0,
        math.log1p(len(toks)) / math.log(32),
        min(switches, 8) / 8.0,
        qmark,
        exclaim,
    ]
    feats.extend(1.0 if stem_hit(toks, s) else 0.0 for s in abuse_stems)
    feats.extend(1.0 if stem_hit(toks, s) else 0.0 for s in benign_stems)
    return np.asarray(feats, dtype=np.float64)


def feature_names(abuse_stems: list[str], benign_stems: list[str]) -> list[str]:
    base = [
        "m_index",
        "i_index",
        "en_frac",
        "hi_frac",
        "dev_frac",
        "abuse_hit",
        "log_tokens",
        "switches",
        "qmark",
        "exclaim",
    ]
    base.extend(f"abuse:{s}" for s in abuse_stems)
    base.extend(f"benign:{s}" for s in benign_stems)
    return base


def split_text(text: str, parts: int = 4) -> list[str] | None:
    raw = tokenize(text)
    if len(raw) < 8:
        return None
    sizes = [len(raw) // parts] * parts
    for i in range(len(raw) % parts):
        sizes[i] += 1
    turns = []
    cursor = 0
    for size in sizes:
        if size <= 0:
            continue
        turns.append(" ".join(raw[cursor : cursor + size]))
        cursor += size
    return turns if len(turns) >= 2 else None


def window_text(turns: list[str]) -> str:
    return " ".join(turns)


def build_examples(
    texts: list[str],
    labels: np.ndarray,
    rng: np.random.Generator,
    abuse_stems: list[str],
    split_n: int,
    benign_n: int,
    camo_n: int,
    include_utterances: bool,
) -> list[Example]:
    examples: list[Example] = []
    if include_utterances:
        for text, lab in zip(texts, labels):
            y = 1 if int(lab) == 0 else 0
            examples.append(Example([text], y, "utterance", None))
    abusive = np.where(labels == 0)[0]
    benign = np.where(labels == 1)[0]
    rng.shuffle(abusive)
    made = 0
    for i in abusive:
        turns = split_text(texts[int(i)])
        if not turns:
            continue
        examples.append(
            Example(turns, 1, "split", stem_hit(content_tokens(turns[-1]), abuse_stems[0]) or any(
                stem_hit(content_tokens(turns[-1]), s) for s in abuse_stems
            ))
        )
        made += 1
        if made >= split_n:
            break
    if len(benign) >= 3 and benign_n > 0:
        for _ in range(benign_n):
            pick = rng.choice(benign, size=3, replace=False)
            turns = [texts[int(i)] for i in pick]
            examples.append(Example(turns, 0, "benign_window", False))
    if len(benign) >= 2 and len(abusive) and camo_n > 0:
        for k in range(camo_n):
            b = rng.choice(benign, size=2, replace=False)
            a = int(abusive[k % len(abusive)])
            turns = [texts[int(i)] for i in b] + [texts[a]]
            last_toks = content_tokens(turns[-1])
            examples.append(
                Example(turns, 1, "camouflage", any(stem_hit(last_toks, s) for s in abuse_stems))
            )
    return examples


def fit_logreg(x: np.ndarray, y: np.ndarray, epochs: int = 40, lr: float = 0.35, l2: float = 1e-3) -> tuple[np.ndarray, float]:
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    pos = max(float(y.mean()), 1e-6)
    sw = np.where(y > 0.5, 0.5 / pos, 0.5 / (1.0 - pos))
    n = len(y)
    for epoch in range(epochs):
        step = lr / (1.0 + 0.05 * epoch)
        logit = np.clip(x @ w + b, -20, 20)
        p = 1.0 / (1.0 + np.exp(-logit))
        err = (p - y) * sw
        w -= step * (x.T @ err / n + l2 * w)
        b -= step * float(err.sum() / n)
    return w, b


def apply_logreg(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    logit = np.clip(x @ w + b, -20, 20)
    return 1.0 / (1.0 + np.exp(-logit))


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-6)


def trim_index(token_counts: list[int], max_slots: int, max_tokens: int) -> list[int]:
    idx = list(range(len(token_counts)))
    while len(idx) > max_slots:
        idx.pop(0)
    while len(idx) > 1 and sum(token_counts[i] for i in idx) > max_tokens:
        idx.pop(0)
    return idx


def init_kv(rng: np.random.Generator, feat_dim: int) -> dict:
    scale = 0.05
    return {
        "Wk": rng.normal(0, scale, size=(DIM, feat_dim)),
        "Wv": rng.normal(0, scale, size=(DIM, feat_dim)),
        "Wq": rng.normal(0, scale, size=(DIM, feat_dim)),
        "head": rng.normal(0, scale, size=(feat_dim + DIM,)),
        "bias": 0.0,
    }


def forward_kv(feats: np.ndarray, params: dict) -> tuple[float, dict]:
    """feats: (T, F) already standardized and already trimmed."""
    wk, wv, wq = params["Wk"], params["Wv"], params["Wq"]
    k_pre = feats @ wk.T
    v_pre = feats @ wv.T
    keys = np.tanh(k_pre)
    values = np.tanh(v_pre)
    q_pre = wq @ feats[-1]
    query = np.tanh(q_pre)
    scale = math.sqrt(DIM)
    scores = keys @ query / scale
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    alpha = exp / exp.sum()
    ctx = alpha @ values
    hidden = np.concatenate([feats[-1], ctx])
    logit = float(np.clip(params["head"] @ hidden + params["bias"], -20, 20))
    prob = 1.0 / (1.0 + math.exp(-logit))
    cache = {
        "feats": feats,
        "k_pre": k_pre,
        "v_pre": v_pre,
        "keys": keys,
        "values": values,
        "q_pre": q_pre,
        "query": query,
        "scores": scores,
        "alpha": alpha,
        "ctx": ctx,
        "hidden": hidden,
        "logit": logit,
        "prob": prob,
    }
    return prob, cache


def backward_kv(cache: dict, dlogit: float, params: dict) -> dict:
    feats = cache["feats"]
    keys, values = cache["keys"], cache["values"]
    query, alpha = cache["query"], cache["alpha"]
    head = params["head"]
    dhead = dlogit * cache["hidden"]
    dbias = dlogit
    dh = dlogit * head
    dc = dh[feats.shape[1] :]
    dvalues = alpha[:, None] * dc[None, :]
    da = values @ dc
    ds = alpha * (da - np.dot(alpha, da))
    scale = math.sqrt(DIM)
    dq = (keys.T @ ds) / scale
    dkeys = np.outer(ds, query) / scale
    dk_pre = dkeys * (1.0 - keys * keys)
    dv_pre = dvalues * (1.0 - values * values)
    dq_pre = dq * (1.0 - query * query)
    grads = {
        "Wk": dk_pre.T @ feats,
        "Wv": dv_pre.T @ feats,
        "Wq": np.outer(dq_pre, feats[-1]),
        "head": dhead,
        "bias": dbias,
        "dc": dc,
    }
    return grads


def supcon(contexts: np.ndarray, labels: np.ndarray, tau: float = 0.2) -> tuple[float, np.ndarray]:
    """Supervised contrastive loss. Gradient is w.r.t. raw context vectors."""
    norms = np.linalg.norm(contexts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    z = contexts / norms
    sim = z @ z.T / tau
    np.fill_diagonal(sim, -1e9)
    sim = sim - sim.max(axis=1, keepdims=True)
    exp = np.exp(sim)
    np.fill_diagonal(exp, 0.0)
    denom = exp.sum(axis=1, keepdims=True) + 1e-12
    prob = exp / denom
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    pos_count = same.sum(axis=1)
    dsim = prob.copy()
    loss_sum = 0.0
    used = 0
    for i in range(len(labels)):
        if pos_count[i] == 0:
            dsim[i] = 0.0
            continue
        dsim[i, same[i]] -= 1.0 / pos_count[i]
        loss_sum += float(-np.log(prob[i, same[i]] + 1e-12).mean())
        used += 1
    np.fill_diagonal(dsim, 0.0)
    loss = loss_sum / max(used, 1)
    dz = (dsim @ z + dsim.T @ z) / tau
    dz /= max(used, 1)
    dc = np.zeros_like(contexts)
    for i in range(len(labels)):
        zi = z[i]
        dc[i] = (dz[i] - zi * np.dot(zi, dz[i])) / norms[i, 0]
    return loss, dc


def check_gradients(rng: np.random.Generator) -> None:
    feat_dim = 6
    params = init_kv(rng, feat_dim)
    feats = rng.normal(0, 1, size=(3, feat_dim))
    _, cache = forward_kv(feats, params)
    grads = backward_kv(cache, 1.0, params)
    eps = 1e-5
    base = cache["logit"]
    params["Wk"][0, 0] += eps
    _, bumped = forward_kv(feats, params)
    params["Wk"][0, 0] -= eps
    numeric = (bumped["logit"] - base) / eps
    if abs(numeric - grads["Wk"][0, 0]) > 1e-4:
        raise SystemExit(f"KV gradient check failed: numeric {numeric} analytic {grads['Wk'][0, 0]}")

    contexts = rng.normal(0, 1, size=(5, DIM))
    labels = np.array([1, 1, 0, 0, 1])
    loss0, dc = supcon(contexts, labels)
    contexts[0, 0] += eps
    loss1, _ = supcon(contexts, labels)
    numeric_c = (loss1 - loss0) / eps
    if abs(numeric_c - dc[0, 0]) > 1e-3:
        raise SystemExit(f"SupCon gradient check failed: numeric {numeric_c} analytic {dc[0, 0]}")
    print("gradient checks passed", flush=True)


class Adam:
    def __init__(self, lr: float = 0.02):
        self.lr = lr
        self.m: dict[str, np.ndarray | float] = {}
        self.v: dict[str, np.ndarray | float] = {}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for name in ("Wk", "Wv", "Wq", "head", "bias"):
            g = grads[name]
            if name not in self.m:
                self.m[name] = np.zeros_like(g) if isinstance(g, np.ndarray) else 0.0
                self.v[name] = np.zeros_like(g) if isinstance(g, np.ndarray) else 0.0
            self.m[name] = 0.9 * self.m[name] + 0.1 * g
            self.v[name] = 0.999 * self.v[name] + 0.001 * (g * g)
            mhat = self.m[name] / (1 - 0.9**self.t)
            vhat = self.v[name] / (1 - 0.999**self.t)
            params[name] = params[name] - self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def sequence_matrix(
    turns: list[str],
    abuse_stems: list[str],
    benign_stems: list[str],
    hindi: set[str],
    english: set[str],
    mean: np.ndarray,
    std: np.ndarray,
    max_slots: int,
    max_tokens: int,
) -> np.ndarray:
    rows = []
    counts = []
    for turn in turns:
        rows.append(
            featurize(turn, abuse_stems, benign_stems, hindi, english)
        )
        counts.append(max(len(content_tokens(turn)), 1))
    idx = trim_index(counts, max_slots, max_tokens)
    mat = np.stack([rows[i] for i in idx], axis=0)
    return standardize(mat, mean, std)


def train_kv(
    examples: list[Example],
    abuse_stems: list[str],
    benign_stems: list[str],
    hindi: set[str],
    english: set[str],
    mean: np.ndarray,
    std: np.ndarray,
    feat_dim: int,
    rng: np.random.Generator,
    contrastive: bool,
    epochs: int = 6,
    batch_size: int = 32,
    contrastive_lambda: float = 0.45,
) -> dict:
    params = init_kv(rng, feat_dim)
    opt = Adam(lr=0.025)
    order = np.arange(len(examples))
    for epoch in range(epochs):
        rng.shuffle(order)
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            if len(batch_ids) < 4:
                continue
            caches = []
            labels = []
            for i in batch_ids:
                ex = examples[int(i)]
                mat = sequence_matrix(
                    ex.turns, abuse_stems, benign_stems, hindi, english, mean, std, MAX_SLOTS, MAX_WINDOW_TOKENS
                )
                _, cache = forward_kv(mat, params)
                caches.append(cache)
                labels.append(ex.y)
            y = np.asarray(labels, dtype=np.float64)
            probs = np.asarray([c["prob"] for c in caches])
            # Mean BCE. dlogit = (p - y) / B because dBCE/dlogit = p - y.
            bce = float(np.mean(y * -np.log(probs + 1e-12) + (1 - y) * -np.log(1 - probs + 1e-12)))
            dlogit = (probs - y) / len(y)
            extra = np.zeros((len(caches), DIM))
            closs = 0.0
            if contrastive:
                ctx = np.stack([c["ctx"] for c in caches], axis=0)
                closs, extra = supcon(ctx, y.astype(int), tau=0.2)
                extra = contrastive_lambda * extra
            acc = {name: np.zeros_like(params[name]) if name != "bias" else 0.0 for name in ("Wk", "Wv", "Wq", "head", "bias")}
            for cache, dl, dc_extra in zip(caches, dlogit, extra):
                grads = backward_kv(cache, float(dl), params)
                if contrastive:
                    # Add contrastive grad through the value path by a second backward of dc.
                    # backward_kv already applied dlogit -> dc. Inject extra dc with dlogit 0
                    # by calling backward on a zero logit and then adding dc through values.
                    zero = backward_kv(cache, 0.0, params)
                    # Recompute dc contribution: run backward with a synthetic head gradient.
                    # Easier: temporarily push extra into values via alpha-weighted dc.
                    feats = cache["feats"]
                    alpha = cache["alpha"]
                    values = cache["values"]
                    dvalues = alpha[:, None] * dc_extra[None, :]
                    dv_pre = dvalues * (1.0 - values * values)
                    zero["Wv"] = dv_pre.T @ feats
                    # Query/key path is unchanged by a pure value gradient.
                    zero["Wk"] = np.zeros_like(params["Wk"])
                    zero["Wq"] = np.zeros_like(params["Wq"])
                    zero["head"] = np.zeros_like(params["head"])
                    zero["bias"] = 0.0
                    # Context also flows into the head features, but those are not
                    # parameters of the projection. The value projection is the
                    # contrastive target.
                    for name in acc:
                        acc[name] = acc[name] + grads[name] + zero[name]
                else:
                    for name in acc:
                        acc[name] = acc[name] + grads[name]
            l2 = 1e-4
            for name in ("Wk", "Wv", "Wq", "head"):
                acc[name] = acc[name] + l2 * params[name]
            opt.step(params, acc)
            total += bce + (contrastive_lambda * closs if contrastive else 0.0)
            seen += 1
        print(
            f"  epoch {epoch + 1}/{epochs} mean loss {total / max(seen, 1):.4f} contrastive={contrastive}",
            flush=True,
        )
    return params


def predict_kv(ex: Example, params: dict, bundle: dict, max_slots: int) -> float:
    mat = sequence_matrix(
        ex.turns,
        bundle["abuse_stems"],
        bundle["benign_stems"],
        bundle["hindi"],
        bundle["english"],
        bundle["mean"],
        bundle["std"],
        max_slots,
        bundle["max_window_tokens"],
    )
    prob, _ = forward_kv(mat, params)
    return prob


def slice_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = (fp / (tn + fp)) if (tn + fp) else None
    return {
        "n": int(len(y)),
        "accuracy": round(float((pred == y).mean()) if len(y) else 0.0, 4),
        "precision_abusive": round(prec, 4),
        "recall_abusive": round(rec, 4),
        "f1_abusive": round(f1, 4),
        "fpr": None if fpr is None else round(fpr, 4),
        "positives": int(y.sum()),
    }


def best_threshold(groups: list[tuple[np.ndarray, np.ndarray]]) -> float:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.15, 0.85, 36):
        accs = []
        for y, scores in groups:
            if len(y) == 0:
                continue
            accs.append(float(((scores >= t).astype(int) == y).mean()))
        score = float(np.mean(accs)) if accs else 0.0
        if score > best_s:
            best_s, best_t = score, float(t)
    return best_t


def grouped(examples: list[Example], scores: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for kind in ("utterance", "split", "benign_window", "camouflage"):
        idx = [i for i, ex in enumerate(examples) if ex.kind == kind]
        if not idx:
            continue
        y = np.asarray([examples[i].y for i in idx], dtype=int)
        s = scores[idx]
        out[kind] = (y, s)
    absent = [i for i, ex in enumerate(examples) if ex.kind == "split" and ex.slur_in_last is False]
    if absent:
        out["split_slur_not_in_final_turn"] = (
            np.asarray([examples[i].y for i in absent], dtype=int),
            scores[absent],
        )
    present = [i for i, ex in enumerate(examples) if ex.kind == "split" and ex.slur_in_last is True]
    if present:
        out["split_slur_in_final_turn"] = (
            np.asarray([examples[i].y for i in present], dtype=int),
            scores[present],
        )
    return out


def score_examples(examples: list[Example], fn) -> np.ndarray:
    return np.asarray([fn(ex) for ex in examples], dtype=np.float64)


def paper_filter_stats(texts: list[str], hindi: set[str], english: set[str]) -> dict:
    m_vals = []
    i_vals = []
    keep = 0
    for text in texts:
        tags = language_tokens(text, hindi, english)
        m = m_index(tags)
        i = i_index(tags)
        m_vals.append(m)
        i_vals.append(i)
        if m >= PAPER_M_MIN and i >= PAPER_I_MIN:
            keep += 1
    arr_m = np.asarray(m_vals)
    arr_i = np.asarray(i_vals)
    return {
        "n": len(texts),
        "mean_m_index": round(float(arr_m.mean()), 4),
        "mean_i_index": round(float(arr_i.mean()), 4),
        "pct_passing_paper_filter_m_and_i_ge_0.1": round(100.0 * keep / max(len(texts), 1), 2),
    }


def metric_block(examples: list[Example], scores: np.ndarray, threshold: float) -> dict:
    groups = grouped(examples, scores)
    return {name: slice_metrics(y, s, threshold) for name, (y, s) in groups.items()}


def mean_slice_accuracy(block: dict) -> float:
    keys = ["utterance", "split", "benign_window", "camouflage"]
    vals = [block[k]["accuracy"] for k in keys if k in block]
    return float(np.mean(vals)) if vals else 0.0


def precision_threshold(examples: list[Example], scores: np.ndarray, max_fpr: float = 0.15) -> float | None:
    """Highest-recall threshold whose benign-window FPR stays within the cap."""
    groups = grouped(examples, scores)
    if "benign_window" not in groups or "split" not in groups:
        return None
    y_b, s_b = groups["benign_window"]
    _, s_s = groups["split"]
    y_u, s_u = groups["utterance"]
    best = None
    for t in np.linspace(0.05, 0.95, 91):
        fpr = float((s_b >= t).mean())
        if fpr > max_fpr:
            continue
        rec = float((s_s >= t).mean())
        acc = float(((s_u >= t).astype(int) == y_u).mean()) if len(y_u) else 0.0
        obj = rec + acc
        if best is None or obj > best[0]:
            best = (obj, float(t))
    return None if best is None else best[1]


def gated_window_scores(
    examples: list[Example],
    current: np.ndarray,
    window: np.ndarray,
    short_tokens: int = 8,
) -> tuple[np.ndarray, dict]:
    """Use the rolling window only when the newest turn is a short fragment.

    A full abusive message keeps its own score, so a benign prefix cannot
    dilute it. A fragment takes the max of the fragment score and the window
    score, which is how a split slur becomes visible again.
    """
    fused = np.empty(len(examples), dtype=np.float64)
    used = {"utterance": 0, "split": 0, "benign_window": 0, "camouflage": 0}
    totals = {k: 0 for k in used}
    for i, ex in enumerate(examples):
        totals[ex.kind] = totals.get(ex.kind, 0) + 1
        short = len(content_tokens(ex.turns[-1])) < short_tokens
        if len(ex.turns) > 1 and short:
            fused[i] = max(float(current[i]), float(window[i]))
            if ex.kind in used:
                used[ex.kind] += 1
        else:
            fused[i] = float(current[i])
    coverage = {
        k: round(used[k] / totals[k], 4) if totals.get(k) else 0.0
        for k in used
        if totals.get(k)
    }
    return fused, coverage


def export_weights(path: Path, params: dict, bundle: dict, threshold: float, arm: str, precision_threshold_value: float | None) -> None:
    payload = {
        "version": 1,
        "arm": arm,
        "dim": DIM,
        "maxSlots": MAX_SLOTS,
        "maxWindowTokens": MAX_WINDOW_TOKENS,
        "threshold": threshold,
        "precisionThreshold": precision_threshold_value,
        "shortTokens": 8,
        "featureNames": bundle["feature_names"],
        "abuseStems": bundle["abuse_stems"],
        "benignStems": bundle["benign_stems"],
        "enForce": sorted(bundle["english"]),
        "hindiClosed": sorted(bundle["hindi"]),
        "mean": bundle["mean"].tolist(),
        "std": bundle["std"].tolist(),
        "Wk": params["Wk"].tolist(),
        "Wv": params["Wv"].tolist(),
        "Wq": params["Wq"].tolist(),
        "head": params["head"].tolist(),
        "bias": float(params["bias"]),
        "citation": "M-index and I-index from Yang & Chai, CodeMixBench, EMNLP 2025, §3.4.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def parity_cases(params: dict, bundle: dict) -> list[dict]:
    samples = [
        {
            "name": "single_abusive",
            "history": [],
            "text": "Teri gand mei pyaz kat dega gujjjar bhosdri ke",
        },
        {
            "name": "split_narrative",
            "history": ["Teri gand mei", "pyaz kat dega"],
            "text": "gujjjar bhosdri ke",
        },
        {
            "name": "benign_codemix",
            "history": ["Kal park mein milte hain"],
            "text": "Kutta billi paal lena magar galat fahmi kabhi nahi",
        },
    ]
    rows = []
    for sample in samples:
        turns = sample["history"] + [sample["text"]]
        mat = sequence_matrix(
            turns,
            bundle["abuse_stems"],
            bundle["benign_stems"],
            bundle["hindi"],
            bundle["english"],
            bundle["mean"],
            bundle["std"],
            MAX_SLOTS,
            MAX_WINDOW_TOKENS,
        )
        prob, cache = forward_kv(mat, params)
        tags = language_tokens(sample["text"], bundle["hindi"], bundle["english"])
        rows.append(
            {
                "name": sample["name"],
                "history": sample["history"],
                "text": sample["text"],
                "m_index": m_index(tags),
                "i_index": i_index(tags),
                "prob_abusive": prob,
                "attn_last": float(cache["alpha"][-1]),
                "slots": int(mat.shape[0]),
            }
        )
    return rows


def shipped_scores(texts: list[str]) -> dict[str, float] | None:
    try:
        from transformers import AutoTokenizer
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover
        print(f"shipped model skipped: {exc}", flush=True)
        return None
    onnx_path = MODEL_DIR / "model_quantized.onnx"
    if not onnx_path.exists():
        onnx_path = MODEL_DIR / "onnx" / "model_quantized.onnx"
    if not onnx_path.exists():
        print("shipped model skipped: onnx file missing", flush=True)
        return None
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    unique = list(dict.fromkeys(texts))
    scores: dict[str, float] = {}
    batch = 64
    for start in range(0, len(unique), batch):
        chunk = unique[start : start + batch]
        enc = tok(chunk, return_tensors="np", truncation=True, max_length=128, padding=True)
        feeds = {}
        for name in input_names:
            if name in enc:
                feeds[name] = enc[name].astype(np.int64)
            elif name == "token_type_ids":
                feeds[name] = np.zeros_like(enc["input_ids"])
        logits = sess.run(None, feeds)[0]
        shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = shifted / shifted.sum(axis=1, keepdims=True)
        for text, p in zip(chunk, prob[:, 0]):
            scores[text] = float(p)
        if start % (batch * 20) == 0:
            print(f"  shipped scored {min(start + len(chunk), len(unique))}/{len(unique)}", flush=True)
    return scores


def round_block(block: dict) -> dict:
    return block


def main() -> None:
    rng = np.random.default_rng(SEED)
    check_gradients(rng)

    hindi = set(HINDI_CLOSED)
    english = set(EN_FORCE)
    # Unit checks for the paper formulas.
    tags_bal = ["hi", "en"]
    tags_mono = ["en", "en", "en"]
    if abs(m_index(tags_bal) - 1.0) > 1e-9 or abs(m_index(tags_mono)) > 1e-9:
        raise SystemExit("M-index formula check failed")
    if abs(i_index(["hi", "en", "hi"]) - 1.0) > 1e-9:
        raise SystemExit("I-index formula check failed")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    perm = rng.permutation(len(train_df))
    n_val = int(0.2 * len(train_df))
    val_df = train_df.iloc[perm[:n_val]].reset_index(drop=True)
    tr_df = train_df.iloc[perm[n_val:]].reset_index(drop=True)
    print(f"train {len(tr_df)} val {len(val_df)} test {len(test_df)}", flush=True)

    tr_texts = tr_df.text.astype(str).tolist()
    tr_labels = tr_df.label.astype(int).to_numpy()
    tr_abusive = (tr_labels == 0).astype(np.int32)
    abuse_stems, benign_stems = mine_stems(tr_texts, tr_abusive)
    print("abuse stems", abuse_stems, flush=True)
    print("benign stems", benign_stems, flush=True)

    train_ex = build_examples(tr_texts, tr_labels, rng, abuse_stems, 2500, 2000, 1500, True)
    val_ex = build_examples(
        val_df.text.astype(str).tolist(),
        val_df.label.astype(int).to_numpy(),
        rng,
        abuse_stems,
        500,
        400,
        400,
        True,
    )
    test_ex = build_examples(
        test_df.text.astype(str).tolist(),
        test_df.label.astype(int).to_numpy(),
        rng,
        abuse_stems,
        1500,
        800,
        800,
        True,
    )
    print(
        "examples",
        {split: Counter(ex.kind for ex in rows) for split, rows in (("train", train_ex), ("val", val_ex), ("test", test_ex))},
        flush=True,
    )

    def matrix_for(examples: list[Example], joiner) -> np.ndarray:
        rows = [
            featurize(joiner(ex.turns), abuse_stems, benign_stems, hindi, english) for ex in examples
        ]
        return np.stack(rows, axis=0)

    utt_train = [ex for ex in train_ex if ex.kind == "utterance"]
    x_utt = matrix_for(utt_train, lambda turns: turns[-1])
    mean = x_utt.mean(axis=0)
    std = x_utt.std(axis=0)
    std[std < 1e-6] = 1.0

    def x_std(examples: list[Example], joiner) -> np.ndarray:
        return standardize(matrix_for(examples, joiner), mean, std)

    y_utt = np.asarray([ex.y for ex in utt_train], dtype=np.float64)
    paper_w, paper_b = fit_logreg(x_std(utt_train, lambda t: t[-1]), y_utt)
    llm_train = utt_train + [ex for ex in train_ex if ex.kind != "utterance"]
    llm_w, llm_b = fit_logreg(
        x_std(llm_train, window_text),
        np.asarray([ex.y for ex in llm_train], dtype=np.float64),
    )
    print("linear models fit", flush=True)

    # KV sees a balanced draw: narratives must not be drowned by utterances.
    kv_train = [ex for ex in train_ex if ex.kind != "utterance"]
    take = min(4000, len(utt_train))
    picked = rng.choice(len(utt_train), size=take, replace=False)
    kv_train.extend(utt_train[int(i)] for i in picked)
    feat_dim = x_utt.shape[1]
    names = feature_names(abuse_stems, benign_stems)
    bundle = {
        "abuse_stems": abuse_stems,
        "benign_stems": benign_stems,
        "hindi": hindi,
        "english": english,
        "mean": mean,
        "std": std,
        "feature_names": names,
        "max_window_tokens": MAX_WINDOW_TOKENS,
    }
    print("training kv_cache", flush=True)
    kv_params = train_kv(
        kv_train, abuse_stems, benign_stems, hindi, english, mean, std, feat_dim, rng, contrastive=False
    )
    print("training kv_contrastive", flush=True)
    con_params = train_kv(
        kv_train, abuse_stems, benign_stems, hindi, english, mean, std, feat_dim, rng, contrastive=True
    )

    def arm_scores(examples: list[Example], arm: str, slots: int = MAX_SLOTS) -> np.ndarray:
        if arm == "codemixbench_isolated":
            return apply_logreg(x_std(examples, lambda t: t[-1]), paper_w, paper_b)
        if arm == "llm_window":
            return apply_logreg(x_std(examples, window_text), llm_w, llm_b)
        params = kv_params if arm == "kv_cache" else con_params
        return score_examples(examples, lambda ex: predict_kv(ex, params, bundle, slots))

    arms = ["codemixbench_isolated", "llm_window", "kv_cache", "kv_contrastive"]
    val_scores = {arm: arm_scores(val_ex, arm) for arm in arms}
    thresholds = {}
    for arm, scores in val_scores.items():
        groups = grouped(val_ex, scores)
        tune = [groups[k] for k in ("utterance", "split", "benign_window", "camouflage") if k in groups]
        thresholds[arm] = best_threshold(tune)
        print(f"val threshold {arm} {thresholds[arm]:.3f} acc {mean_slice_accuracy(metric_block(val_ex, scores, thresholds[arm])):.4f}", flush=True)

    test_blocks = {}
    test_scores = {}
    for arm in arms:
        scores = arm_scores(test_ex, arm)
        test_scores[arm] = scores
        test_blocks[arm] = metric_block(test_ex, scores, thresholds[arm])
        print(arm, {k: test_blocks[arm][k]["accuracy"] for k in ("utterance", "split", "benign_window", "camouflage")}, flush=True)

    # Constrained budget: same contrastive weights, 2 slots.
    scores_slots2 = arm_scores(test_ex, "kv_contrastive", slots=2)
    test_blocks["kv_contrastive_slots2"] = metric_block(test_ex, scores_slots2, thresholds["kv_contrastive"])

    val_con = mean_slice_accuracy(metric_block(val_ex, val_scores["kv_contrastive"], thresholds["kv_contrastive"]))
    val_kv = mean_slice_accuracy(metric_block(val_ex, val_scores["kv_cache"], thresholds["kv_cache"]))
    shipped_arm = "kv_contrastive" if val_con + 1e-9 >= val_kv else "kv_cache"
    shipped_params = con_params if shipped_arm == "kv_contrastive" else kv_params
    prec_t = precision_threshold(val_ex, val_scores[shipped_arm], max_fpr=0.15)
    export_weights(OUT_WEIGHTS, shipped_params, bundle, thresholds[shipped_arm], shipped_arm, prec_t)
    if prec_t is not None:
        test_blocks["kv_contrastive_fpr15"] = metric_block(test_ex, test_scores[shipped_arm], prec_t)
    parity = parity_cases(shipped_params, bundle)
    OUT_PARITY.write_text(json.dumps(parity, indent=2))

    filter_stats = paper_filter_stats(test_df.text.astype(str).tolist(), hindi, english)

    shipped_block = None
    report_coverage = None
    try:
        needed = []
        for ex in test_ex:
            if ex.kind == "utterance":
                needed.append(ex.turns[-1])
            elif ex.kind == "split":
                needed.append(ex.turns[-1])
                needed.append(window_text(ex.turns))
            elif ex.kind in ("benign_window", "camouflage"):
                needed.append(ex.turns[-1])
                needed.append(window_text(ex.turns))
        print(f"scoring shipped INT8 on {len(set(needed))} unique strings", flush=True)
        table = shipped_scores(needed)
        if table is not None:
            iso = np.asarray([table[ex.turns[-1]] for ex in test_ex])
            win = np.asarray(
                [table[ex.turns[-1] if ex.kind == "utterance" else window_text(ex.turns)] for ex in test_ex]
            )
            test_blocks["shipped_int8"] = metric_block(test_ex, iso, 0.5)
            test_blocks["shipped_int8_window"] = metric_block(test_ex, win, 0.5)
            fused, coverage = gated_window_scores(test_ex, iso, win, short_tokens=8)
            test_blocks["shipped_int8_gated_window"] = metric_block(test_ex, fused, 0.5)
            report_coverage = coverage
            shipped_block = True
            print(
                "shipped",
                {k: test_blocks["shipped_int8"][k]["accuracy"] for k in ("utterance", "split", "benign_window", "camouflage")},
                flush=True,
            )
            print(
                "gated",
                {k: test_blocks["shipped_int8_gated_window"][k]["accuracy"] for k in ("utterance", "split", "benign_window", "camouflage")},
                coverage,
                flush=True,
            )
    except Exception as exc:
        print(f"shipped model failed: {exc}", flush=True)
        shipped_block = None
    if "kv_contrastive_fpr15" in test_blocks:
        print(
            "contrastive fpr<=0.15",
            prec_t,
            {k: test_blocks["kv_contrastive_fpr15"][k]["accuracy"] for k in ("utterance", "split", "benign_window", "camouflage")},
            flush=True,
        )

    kind_counts = Counter(ex.kind for ex in test_ex)
    slur_absent = sum(1 for ex in test_ex if ex.kind == "split" and ex.slur_in_last is False)
    report = {
        "paper": {
            "title": "CodeMixBench: Evaluating Code-Mixing Capabilities of LLMs Across 18 Languages",
            "authors": "Yilun Yang, Yekun Chai",
            "venue": "EMNLP 2025",
            "equations": {
                "M-index": "(1 - sum_j p_j^2) / ((k - 1) * sum_j p_j^2)",
                "I-index": "sum_i S(i, i+1) / (n - 1)",
                "k": 2,
                "filter": "drop sentence if M < 0.1 or I < 0.1",
            },
        },
        "task": {
            "labels": {"abusive": 0, "non_abusive": 1, "model_positive": "abusive"},
            "train_rows": int(len(tr_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "test_example_counts": dict(kind_counts),
            "split_slur_not_in_final_turn": slur_absent,
            "constraints": {"max_slots": MAX_SLOTS, "max_window_tokens": MAX_WINDOW_TOKENS, "dim": DIM},
            "hinglish_test_codemix_filter": filter_stats,
            "abuse_stems_train_only": abuse_stems,
            "benign_stems_train_only": benign_stems,
        },
        "published_llm_scores": PUBLISHED_TABLE2,
        "val_thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "val_mean_slice_accuracy": {
            "kv_cache": round(val_kv, 4),
            "kv_contrastive": round(val_con, 4),
        },
        "shipped_local_arm": shipped_arm,
        "precision_threshold_fpr15": None if prec_t is None else round(prec_t, 4),
        "gated_window_coverage": report_coverage,
        "test": test_blocks,
        "parity_cases": parity,
        "notes": [
            "codemixbench_isolated scores only the current turn, which is the CodeMixBench evaluation protocol.",
            "llm_window scores the concatenated rolling window, which is the text an LLM prompt would receive.",
            "kv_cache is the same window under an 8-slot key/value budget for the browser service worker.",
            "kv_contrastive adds supervised contrastive loss so abusive narrative states separate from benign code-mix with a similar M-index.",
            "shipped_int8 is the deployed MiniLM INT8 encoder at the extension threshold 0.5. It is a reference, not a CodeMixBench model.",
            "Thresholds for learned arms are selected on the train-derived validation split. Test is scored once.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"wrote {OUT_WEIGHTS}", flush=True)
    if shipped_block is None:
        print("shipped INT8 block missing", flush=True)


if __name__ == "__main__":
    main()
