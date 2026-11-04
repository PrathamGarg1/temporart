# AWS reimbursement packet — multiclass thread detector

Start date: 2026-11-03 (IST). Prior MACD Hinglish conversion spend is **not** in this ledger.

## What to attach

1. `cost_ledger.csv` — running USD by day and job
2. `bedrock_calls.jsonl` — one line per Bedrock call (timestamp, model, tokens, request id, estimated USD)
3. AWS invoice / Cost Explorer for `us-east-1`, service Amazon Bedrock, same date range
4. This folder: `aws_reimburse/multiclass_threads/`

List price used for estimates (us-east-1, standard on-demand, 2026-11-02 AWS price list):

| Model | Input / 1M | Output / 1M |
|---|---:|---:|
| `qwen.qwen3-32b` (Mantle) | $0.15 | $0.60 |
| `qwen.qwen3-32b-v1:0` (Converse) | $0.15 | $0.60 |
| `meta.llama3-8b-instruct-v1:0` (fallback) | $0.30 | $0.60 |

Estimates can differ from the invoice by a few percent (rounding, failed retries). Match on **request id** and UTC time.

Local INT8 scoring and MiniLM training on this laptop are logged as `local` with USD 0.
