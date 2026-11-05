# GRU thread model (frozen MiniLM + GRU)

Pipeline: `embed_lines.py` (frozen MiniLM, each line once, `in: `/`out: ` prefix, 384-d mean-pooled)
-> `train_gru.py` (GRU 384->128, 1 layer, forward, packed so padded steps are skipped; Linear 128->5; CE loss)
-> `out/gru.json` (198,021 numbers) -> `gru_runtime.js` (plain JS) -> `node parity_test.mjs`.

Test set: 10,563 threads (dataset's own 80/10/10 split).

| | acc | macro-F1 | len 3 | len 5 | len 7 | len 10 | same-last-line pairs |
|---|---|---|---|---|---|---|---|
| GRU (this) | 0.9517 | 0.9522 | 0.930 | 0.953 | 0.956 | 0.969 | 0.936 |
| last line only (ablation, same encoder) | 0.7813 | 0.7844 | 0.806 | 0.796 | 0.773 | 0.749 | 0.499 |
| fine-tuned MiniLM, thread text (earlier) | 0.9796 | 0.9798 | 0.969 | 0.984 | 0.982 | 0.984 | 0.974 |

Not wired into the extension yet; the shipped binary INT8 model is untouched.
