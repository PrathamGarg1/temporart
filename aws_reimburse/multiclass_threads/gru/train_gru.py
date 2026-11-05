"""Step 2/3 of the GRU plan: frozen-MiniLM line vectors -> GRU(384->128) -> Linear(5).

  * one model, lengths 3/5/7/10 trained together, blocks padded to 10, GRU skips empty steps
    (pack_padded_sequence), loss = cross-entropy against the thread label
  * 80/10/10 split is the dataset's own train/val/test split (test = 10,563 threads)
  * reports accuracy + macro-F1 on the test set, per length (3/5/7/10), per script, per class
  * ablation: same model but fed ONLY the last-line vector (no context) -> shows what context adds
  * exports gru.json (plain numbers, runs in JS) + a small parity fixture for the JS test
Pure local CPU/MPS compute; no cloud calls -> nothing to put in the AWS cost ledger.
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "out"
CLASSES = ["benign", "friendly_sarcasm", "hostile_abuse", "sexual_harassment", "predatory_grooming"]
D, H, K = 384, 128, 5


class ThreadGRU(nn.Module):
    def __init__(self, drop=0.1):
        super().__init__()
        self.gru = nn.GRU(D, H, num_layers=1, batch_first=True)  # forward only, 1 layer
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(H, K)

    def forward(self, x, lens):
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)          # h: [1, B, H] = state after the last REAL step
        return self.fc(self.drop(h[-1]))


def load(split):
    X = np.load(CACHE / f"emb_{split}.npy", mmap_mode="r")
    meta = json.load(open(CACHE / f"meta_{split}.json"))
    return torch.from_numpy(np.ascontiguousarray(X)), meta


def macro_f1(y, p):
    f = []
    for c in range(K):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(f))


def summarize(meta, y, p):
    out = {"n": len(y), "accuracy": float((y == p).mean()), "macro_f1": macro_f1(y, p)}
    for key in ("length", "script"):
        g = defaultdict(list)
        for i, r in enumerate(meta):
            g[r[key]].append(i)
        out[f"by_{key}"] = {str(k): {"n": len(v), "accuracy": float((y[v] == p[v]).mean()),
                                     "macro_f1": macro_f1(y[v], p[v])} for k, v in sorted(g.items())}
    out["per_class_recall"] = {c: float((p[y == i] == i).mean()) for i, c in enumerate(CLASSES)}
    cm = np.zeros((K, K), dtype=int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    out["confusion"] = cm.tolist()
    pair = [i for i, r in enumerate(meta) if r["pair_id"]]
    out["same_last_line_pairs"] = {"n_threads": len(pair), "accuracy": float((y[pair] == p[pair]).mean())}
    return out


def as_input(X, lens, ctx):
    """ctx=False -> ablation: keep only the vector of the last real line, length 1."""
    if ctx:
        return X, lens
    idx = (lens - 1).view(-1, 1, 1).expand(-1, 1, X.shape[2])
    return X.gather(1, idx), torch.ones_like(lens)


@torch.no_grad()
def predict(model, X, lens, ctx, dev, bs=1024):
    model.eval(); outs = []
    for b in range(0, len(X), bs):
        x, l = as_input(X[b:b + bs].to(dev).float(), lens[b:b + bs].to(dev), ctx)
        outs.append(model(x, l).float().cpu())
    return torch.cat(outs)


def run(ctx, tr, va, te, args, dev, tag):
    (Xtr, ltr, ytr), (Xva, lva, yva), (Xte, lte, yte) = tr, va, te
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = ThreadGRU().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    lossf = nn.CrossEntropyLoss()
    best, best_state, bad, log = -1, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0; t0 = time.time()
        for b in range(0, len(perm), args.batch):
            i = perm[b:b + args.batch]
            x, l = as_input(Xtr[i].to(dev).float(), ltr[i].to(dev), ctx)
            loss = lossf(model(x, l), ytr[i].to(dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(i)
        pv = predict(model, Xva, lva, ctx, dev).argmax(1).numpy()
        f1 = macro_f1(yva.numpy(), pv); acc = float((pv == yva.numpy()).mean())
        sched.step(f1)
        log.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_acc": acc, "val_macro_f1": f1})
        print(f"[{tag}] ep{ep:02d} loss={tot/len(Xtr):.4f} val_acc={acc:.4f} val_f1={f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, log


def export(model, X, lens, y, meta):
    sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
    r6 = lambda a: np.round(a.reshape(-1), 6).tolist()
    payload = {
        "format": "gru-v1", "input_dim": D, "hidden": H, "classes": CLASSES, "gate_order": "r,z,n (PyTorch)",
        "encoder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 mean-pooled, no normalise; "
                   "line text prefixed 'in: ' (speaker of last message) or 'out: ' (other speaker)",
        "W_ih": r6(sd["gru.weight_ih_l0"]), "W_hh": r6(sd["gru.weight_hh_l0"]),
        "b_ih": r6(sd["gru.bias_ih_l0"]), "b_hh": r6(sd["gru.bias_hh_l0"]),
        "W_fc": r6(sd["fc.weight"]), "b_fc": r6(sd["fc.bias"]),
    }
    payload["n_numbers"] = sum(len(payload[k]) for k in ("W_ih", "W_hh", "b_ih", "b_hh", "W_fc", "b_fc"))
    json.dump(payload, open(OUT / "gru.json", "w"), separators=(",", ":"))
    # parity fixture: 24 test threads, float32 vectors rounded as shipped, expected logits from PyTorch
    idx = [next(i for i, m in enumerate(meta) if m["length"] == L and m["label_id"] == c and i > off)
           for L in (3, 5, 7, 10) for c in (0, 2, 4) for off in (0, 400)]
    model.eval()
    with torch.no_grad():
        # reload shipped (rounded) weights so the fixture matches the JSON exactly
        for name, key in (("gru.weight_ih_l0", "W_ih"), ("gru.weight_hh_l0", "W_hh"), ("gru.bias_ih_l0", "b_ih"),
                          ("gru.bias_hh_l0", "b_hh"), ("fc.weight", "W_fc"), ("fc.bias", "b_fc")):
            p = dict(model.named_parameters())[name]
            p.copy_(torch.tensor(payload[key], dtype=torch.float32).view_as(p))
        fx = []
        for i in idx:
            n = int(lens[i]); v = X[i, :n].float()
            lg = model(v.unsqueeze(0), torch.tensor([n])).squeeze(0).numpy()
            fx.append({"n": n, "vectors": np.round(v.numpy(), 5).tolist(), "logits": np.round(lg, 5).tolist()})
    json.dump(fx, open(OUT / "parity_fixture.json", "w"), separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dev = "cpu"  # small model; CPU is fastest and avoids MPS recompiles with packed sequences
    torch.set_num_threads(6)
    sets = []
    for s in ("train", "val", "test"):
        X, meta = load(s)
        lens = torch.tensor([m["n"] for m in meta]); y = torch.tensor([m["label_id"] for m in meta])
        sets.append((X, lens, y, meta))
    tr, va, te = [(a, b, c) for a, b, c, _ in sets]
    meta_te = sets[2][3]
    report = {"n_test": len(meta_te), "classes": CLASSES, "args": vars(args), "models": {}}
    for tag, ctx in (("gru_context", True), ("last_line_only_ablation", False)):
        model, log = run(ctx, tr, va, te, args, dev, tag)
        p = predict(model, te[0], te[1], ctx, dev)
        pred = p.argmax(1).numpy(); y = te[2].numpy()
        m = summarize(meta_te, y, pred); m["train_log"] = log
        report["models"][tag] = m
        print(f"== {tag}: test acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} pairs={m['same_last_line_pairs']['accuracy']:.4f}", flush=True)
        for L, v in m["by_length"].items():
            print(f"   len {L}: n={v['n']} acc={v['accuracy']:.4f} macroF1={v['macro_f1']:.4f}", flush=True)
        if ctx:
            torch.save(model.state_dict(), OUT / "gru.pt")
            export(model, te[0], te[1], y, meta_te)
            np.save(OUT / "test_probs.npy", torch.softmax(p, 1).numpy())
    json.dump(report, open(OUT / "benchmark_report.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
