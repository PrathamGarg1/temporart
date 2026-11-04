# Multiclass thread detector (kids / women)

All AWS spend for this workstream is logged under `logs/`. Do not run these jobs from `aws_train/`.

```bash
# 1. generate threads (Bedrock, billed)
.venv_codemix/bin/python -u aws_reimburse/multiclass_threads/scripts/generate_threads.py

# 2. train 5-way MiniLM (local, USD 0)
.venv_codemix/bin/python -u aws_reimburse/multiclass_threads/scripts/train_multiclass.py

# 3. isolated vs k=3,5,7,10
.venv_codemix/bin/python -u aws_reimburse/multiclass_threads/scripts/benchmark_threads.py
```

Shipped binary INT8 is never written.
