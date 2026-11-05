"""Step 1 of the GRU plan: encode every line ONCE with a frozen MiniLM.

Each line is prefixed `in: ` (speaker of the LAST message) or `out: ` (the other speaker).
Speakers alternate, so message i of an n-message thread is `in` iff (n-1-i) is even.
Neighbouring lines are never glued on, so nothing is truncated away.
Output: cache/emb_{split}.npy (float16, [n_threads, 10, 384]) + lens, labels, meta.
No cloud / Bedrock calls are made here, so there is nothing to put in the cost ledger.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "cache"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAXLEN, T, D = 64, 10, 384


def tag(i, n):
    return "in: " if (n - 1 - i) % 2 == 0 else "out: "


def load(split):
    rows = []
    for p in sorted(DATA.glob(f"len_*/*/*/{split}.jsonl")):
        with open(p) as f:
            rows += [json.loads(l) for l in f]
    rows.sort(key=lambda r: r["thread_id"])
    return rows


@torch.no_grad()
def encode(texts, tok, model, dev, bs):
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), D), dtype=np.float16)
    t0 = time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN,
                  pad_to_multiple_of=16, return_tensors="pt").to(dev)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out[idx] = v.float().cpu().numpy().astype(np.float16)
        if (b // bs) % 50 == 0:
            print(f"  {b}/{len(order)}  {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    OUT.mkdir(exist_ok=True)
    for split in ["train", "val", "test"]:
        rows = load(split)
        if a.limit:
            rows = rows[:a.limit]
        flat, where = [], []
        for ti, r in enumerate(rows):
            n = len(r["turns"])
            for i, t in enumerate(r["turns"]):
                flat.append(tag(i, n) + t.strip()); where.append((ti, i))
        uniq = {}
        for s in flat:
            uniq.setdefault(s, len(uniq))
        print(f"{split}: {len(rows)} threads, {len(flat)} lines, {len(uniq)} unique", flush=True)
        vec = encode(list(uniq), tok, model, dev, a.bs)
        X = np.zeros((len(rows), T, D), dtype=np.float16)
        for (ti, i), s in zip(where, flat):
            X[ti, i] = vec[uniq[s]]
        np.save(OUT / f"emb_{split}.npy", X)
        meta = [{k: r[k] for k in ("thread_id", "length", "script", "label", "label_id", "pair_id", "subtype")}
                | {"n": len(r["turns"]), "last_turn": r["turns"][-1]} for r in rows]
        json.dump(meta, open(OUT / f"meta_{split}.json", "w"), ensure_ascii=False)
        print(f"saved {split} {X.shape}", flush=True)


if __name__ == "__main__":
    main()
