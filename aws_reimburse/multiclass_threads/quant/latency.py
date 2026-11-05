"""Single-request latency of INT8 ONNX models, native onnxruntime on this laptop CPU, 1 thread, batch 1.
NOT an in-browser (WASM) measurement. Reports tokenisation+inference and inference-only medians."""
import csv, json, time, sys
from pathlib import Path
import numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[3]; MC = ROOT / "aws_reimburse/multiclass_threads"
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
def bench(name, onnx, tokdir, texts, pairs=None, max_len=128, n=300):
    tok = AutoTokenizer.from_pretrained(tokdir)
    s = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]
    tot, inf = [], []
    for i in range(n + 20):
        t0 = time.perf_counter()
        e = tok(texts[i % len(texts)], *( [pairs[i % len(pairs)]] if pairs else []), truncation=True, max_length=max_len, return_tensors="np")
        feed = {k: e[k].astype(np.int64) for k in names}
        t1 = time.perf_counter(); s.run(None, feed); t2 = time.perf_counter()
        if i >= 20: tot.append((t2 - t0) * 1e3); inf.append((t2 - t1) * 1e3)
    r = {"model": name, "n": n, "tokens_median": float(np.median([len(tok(t)["input_ids"]) for t in texts[:n]])),
         "median_ms_total": float(np.median(tot)), "p95_ms_total": float(np.percentile(tot, 95)), "median_ms_inference": float(np.median(inf))}
    print(json.dumps(r)); return r
rows = list(csv.DictReader(open(ROOT / "codemix_hinglish/test.csv", encoding="utf-8")))
tc = "text" if "text" in rows[0] else list(rows[0])[0]
texts = [r[tc] for r in rows[:300]]
out = [bench("shipped_binary_int8 (Hinglish test messages)", ROOT / "assets/models/custom-macd-model/onnx/model_quantized.onnx", ROOT / "assets/models/custom-macd-model", texts)]
th = [json.loads(l) for l in open(MC / "dataset/len_05/hinglish/2_hostile_abuse/test.jsonl")][:300]
out.append(bench("thread_int8, 5-message Hinglish threads", MC / "quant/onnx/thread_int8.onnx", MC / "artifacts_v2/cloud_results/model_thread",
                 [t["turns"][-1] for t in th], [" [TURN] ".join(reversed(t["turns"][:-1])) for t in th], max_len=256))
json.dump(out, open(MC / "quant/latency_report.json", "w"), indent=1)
