# SurakshaNet multi-class thread project: report

Pratham Garg (IIT Ropar), guide Geeta Yadav, with Aamod Jain. Status as of 5 Nov 2026.

## 1. What was built

| Item | Result |
|---|---|
| Dataset | **105,000 synthetic chat threads**, 5 classes (benign, friendly_sarcasm, hostile_abuse, sexual_harassment, predatory_grooming), 3 scripts (English, Hindi Devanagari, Hinglish), lengths 3/5/7/10. 26,250 per length, 35,000 per script, 21,000 per class. Split 83,672 / 10,765 / 10,563 (train / val / test). Folder tree `dataset/len_*/<script>/<class>/{train,val,test}.jsonl`, plus `MANIFEST.csv` and `DATASET_CARD.md`. |
| Design | Predict the last message given the earlier ones. About half of the sarcasm and hostile threads end in an identical line (`pair_id`), so only the context can separate them. Validators reject refusals, wrong script, profanity in clean classes, explicit text and sexualised grooming. Thread-hash dedup. |
| Generation | Qwen3-32B on Bedrock (us-east-1), resumable, with a spending cap and a per-call ledger. |
| Viewer | `viewer/app.py`: dataset-only look, pick length / script / class, random chat-style thread, statistics. Runs locally; hosting is on hold. |
| Models | (a) fine-tuned MiniLM on thread text, (b) fine-tuned MiniLM on the last message only, (c) frozen MiniLM + GRU (`gru/`). |

## 2. Results (synthetic test set, 10,563 threads)

| Model | Acc | Macro-F1 | Len 3 | Len 5 | Len 7 | Len 10 | Same-last-line pairs |
|---|---|---|---|---|---|---|---|
| Fine-tuned MiniLM, thread text | 97.96% | 0.980 | 96.9 | 98.4 | 98.2 | 98.4 | 97.4% |
| **Frozen MiniLM + GRU** (198k weights, runs in JS) | **95.17%** | **0.952** | 93.0 | 95.3 | 95.6 | 96.9 | 93.6% |
| GRU, last line only (ablation) | 78.13% | 0.784 | 80.6 | 79.6 | 77.3 | 74.9 | 49.9% |
| Fine-tuned MiniLM, last message only | 83.18% | 0.828 | 85.1 | 84.1 | 82.7 | 80.8 | 51.3% |

- Context is the signal: removing it costs 14 to 17 points, and same-last-line pairs fall to chance.
- GRU by script: English 98.4%, Hindi 95.3%, Hinglish 91.8%. Per-class recall 93.4% to 99.5%. Main confusion: hostile_abuse vs friendly_sarcasm.
- GRU export `gru/out/gru.json` matches PyTorch to 3e-4 (24/24 same predictions), about 2.5 ms per thread in Node.

## 3. External validation on real data (zero-shot): negative result

Models trained only on synthetic data, scored on CAD, ConvAbuse and PRISM (English, Hindi, Hinglish). AUROC mostly 0.47 to 0.74 (CAD 0.56 to 0.59; ConvAbuse 0.72 to 0.74; PRISM Hindi 0.47, Hinglish 0.52, English 0.67). Context did not help on real CAD. A shortcut was found: very short neutral replies ("yes", "how old are you") score as hostile, because hostile synthetic threads end in short insults. These numbers were measured on the fine-tuned models, not the GRU, which learned from the same data and is expected to share the gap but was not tested. Details: `external_eval/RESULTS.md`.

**So the 95 to 98% scores describe the synthetic distribution only. They are not evidence of real-world accuracy.**

## 4. Cloud and cost (reimbursement)

All AWS work from the multiclass-thread effort onward is in `aws_reimburse/multiclass_threads/`, with a per-call ledger (`logs/bedrock_calls.jsonl`, `logs/cost_ledger.csv`).

| Part | USD |
|---|---|
| Dataset generation (Bedrock Qwen3-32B) | 21.93 |
| Earlier v1 generation | 0.14 |
| GPU training run (g4dn.xlarge, 19.8 min, self-terminated) + EBS | 0.18 |
| **Total** | **22.25** |

The GRU work ran locally and cost nothing on AWS.

## 5. Open items

1. Fix the short-last-line shortcut in the data and add variety; mix real CAD / ConvAbuse training data with held-out evaluation.
2. Real Hinglish threads: HASOC ICHCL needs a key from hasocfire@gmail.com. No open threaded Hindi data was found.
3. Extension: build a frozen INT8 MiniLM encoder file and wire in `gru_runtime.js` (the shipped binary INT8 model is untouched).
4. Synthetic Hindi quality is weak; the viewer is not hosted.
5. Cloud agent: paused; the repo is public (`PrathamGarg1/temporart`), so decide whether to make it private.

## 6. Quantized (INT8) models, tested on the same 10,563 test threads

| Model | Size | Acc | Macro-F1 | Len 3 / 5 / 7 / 10 | Hinglish | Pairs | Speed (CPU) |
|---|---|---|---|---|---|---|---|
| INT8 fine-tuned thread classifier | 118 MB | **97.88%** | 0.979 | 96.9 / 98.3 / 98.1 / 98.3 | 96.9% | 97.5% | 15.5 ms / thread |
| INT8 frozen encoder + GRU | 118 MB + 4 MB | **94.91%** | 0.950 | 92.3 / 95.1 / 95.7 / 96.7 | 91.5% | 93.1% | encoder per line + GRU 2.5 ms |

Quantization costs 0.08 points (thread classifier: 97.96 -> 97.88) and 0.26 points (GRU route: 95.17 -> 94.91). Files: `quant/export_quantize.py`, `quant/eval_quant.py`, `quant/int8_report.json`. The `.onnx` files (118 MB each, over GitHub's 100 MB limit) are kept locally in `quant/onnx/`.

## 7. Paper revision (`main.tex`)

`main.tex` was rewritten so that every number is traceable to a file in this repo (`assets/models/custom-macd-model/metrics.json`, `codemix_eval/*.json`, `quant/*.json`, `gru/out/benchmark_report.json`, `external_eval/RESULTS.md`, `logs/`). Removed because the code in this repo does not implement or evidence them: parental login, master password, screenshot capture (the `screenshot` field is always `null`), hiding messages, AES-256, chain of custody, court-admissible reports, tamper-proof SHA-256 (it is only a dedup id). Corrected: quantization is *dynamic* (not static), size reduction is about 75% (470.6 -> 118.4 MB; the old "471 -> 113 MB, 76%" mixed MB and MiB), accuracy numbers now come from the shipped v2 model (Hindi 85.14, Hinglish 84.36, English 96.73, mixed 86.54; the poster's 84.66 / 88.01 belong to an earlier model). The unmeasured "28 ms" and "89 MB RAM" were dropped; new native measurement in `quant/latency_report.json` (5.0 ms median for the shipped model, 28.3 ms for the 5-message thread model; not in-browser). Added: data-generation, training and system diagrams (TikZ, `figures/`), the five-class dataset, models, results, the negative external validation, and a limitations section.
