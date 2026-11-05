"""Test the INT8 ONNX models on the same 10,563 test threads (CPU, onnxruntime).
  A) thread_int8.onnx           : fine-tuned thread classifier, INT8
  B) encoder_int8.onnx + GRU    : frozen INT8 encoder per line -> GRU (trained on fp32 vectors) -> 5 classes
"""
import json, sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort, torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "gru"))
from train_gru import ThreadGRU, summarize, CLASSES  # noqa: E402
from embed_lines import load, tag, MODEL, MAXLEN  # noqa: E402

rows = load("test")
y = np.array([r["label_id"] for r in rows])
meta = [{k: r[k] for k in ("length", "script", "pair_id")} for r in rows]
so = ort.SessionOptions(); so.intra_op_num_threads = 6


def run_thread():
    tok = AutoTokenizer.from_pretrained(ROOT / "artifacts_v2/cloud_results/model_thread")
    max_len = json.load(open(ROOT / "artifacts_v2/cloud_results/model_thread/train_config.json"))["max_len"]
    sess = ort.InferenceSession(str(HERE / "onnx/thread_int8.onnx"), so, providers=["CPUExecutionProvider"])
    enc = [tok(r["turns"][-1], " [TURN] ".join(reversed(r["turns"][:-1])), truncation="only_second", max_length=max_len)
           for r in rows]
    order = np.argsort([len(e["input_ids"]) for e in enc]); pred = np.zeros(len(rows), int); t0 = time.time()
    for b in range(0, len(order), 32):
        idx = order[b:b + 32]
        pad = tok.pad([{k: enc[i][k] for k in ("input_ids", "attention_mask", "token_type_ids")} for i in idx],
                      return_tensors="np")
        lg = sess.run(None, {k: pad[k].astype(np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")})[0]
        pred[idx] = lg.argmax(1)
        if (b // 32) % 40 == 0: print(f"  thread {b}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    return pred, (time.time() - t0) / len(rows) * 1000


def run_encoder_gru():
    tok = AutoTokenizer.from_pretrained(MODEL)
    sess = ort.InferenceSession(str(HERE / "onnx/encoder_int8.onnx"), so, providers=["CPUExecutionProvider"])
    flat, where = [], []
    for ti, r in enumerate(rows):
        n = len(r["turns"])
        for i, t in enumerate(r["turns"]):
            flat.append(tag(i, n) + t.strip()); where.append((ti, i))
    uniq = {}
    for s in flat: uniq.setdefault(s, len(uniq))
    texts = list(uniq); order = np.argsort([len(t) for t in texts]); vec = np.zeros((len(texts), 384), np.float32); t0 = time.time()
    for b in range(0, len(order), 128):
        idx = order[b:b + 128]
        e = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="np")
        vec[idx] = sess.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0]
        if (b // 128) % 40 == 0: print(f"  encoder {b}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    X = np.zeros((len(rows), 10, 384), np.float32)
    for (ti, i), s in zip(where, flat): X[ti, i] = vec[uniq[s]]
    lens = torch.tensor([len(r["turns"]) for r in rows])
    g = ThreadGRU(); g.load_state_dict(torch.load(ROOT / "gru/out/gru.pt")); g.eval()
    with torch.no_grad():
        pred = np.concatenate([g(torch.from_numpy(X[b:b + 1024]), lens[b:b + 1024]).argmax(1).numpy() for b in range(0, len(X), 1024)])
    return pred, X


def main():
    out = {}
    which = sys.argv[1:] or ["gru", "thread"]
    if "gru" in which:
        p, X = run_encoder_gru(); out["int8_encoder_plus_gru"] = summarize(meta, y, p)
        # drift of INT8 vectors vs the fp32 vectors the GRU was trained on
        ref = np.load(ROOT / "gru/cache/emb_test.npy").astype(np.float32)
        m = X.any(2); cos = (X * ref).sum(2) / (np.linalg.norm(X, axis=2) * np.linalg.norm(ref, axis=2) + 1e-9)
        out["int8_encoder_plus_gru"]["mean_cosine_int8_vs_fp32_vectors"] = float(cos[m].mean())
        print("GRU+INT8 encoder:", out["int8_encoder_plus_gru"]["accuracy"], out["int8_encoder_plus_gru"]["macro_f1"], flush=True)
    if "thread" in which:
        p, ms = run_thread(); out["int8_thread_classifier"] = summarize(meta, y, p); out["int8_thread_classifier"]["ms_per_thread_cpu"] = ms
        print("INT8 thread:", out["int8_thread_classifier"]["accuracy"], out["int8_thread_classifier"]["macro_f1"], flush=True)
    json.dump(out, open(HERE / "int8_report.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
