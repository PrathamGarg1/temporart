# Session: Hinglish Code-Mix

## Status
**Corrected retrain DONE** (v2 protocol) + extension rebuilt.

### Protocol
| Split | Data |
|-------|------|
| **train** | Hindi MACD train + Davidson **80%** + Hinglish train |
| **val** | Hindi MACD val + Davidson **10%** + Hinglish val |
| **test** | Hindi MACD test + Davidson **10%** + Hinglish test |

Sizes: train **58929** / val **15581** / test **15575**

### INT8 test accuracy (deployed model)
| Slice | Acc | F1 |
|-------|-----|----|
| Hindi Devanagari | **85.1%** | 85.1% |
| Hinglish code-mix | **84.4%** | 84.3% |
| Davidson EN | **96.7%** | 94.2% |
| Mixed test | **86.5%** | 86.4% |

(Old poster Hindi-only was 84.7% — Hindi is recovered; Hinglish is strong; previous wrong run had Hindi at ~70%.)

### Artifacts
- Model: `assets/models/custom-macd-model/`
- Train script: `aws_train/train_codemix_modal.py`
- Data: `codemix_hinglish/` + `aws_train/macd_hindi_cache/`
- S3 backup: `s3://suraksha-codemix-439446323592-20260919202858/codemix_hinglish/`

```bash
modal run aws_train/train_codemix_modal.py
npm run build
```
