# External validation on real, human-labelled data (zero-shot)

Models: `artifacts_v2/cloud_results/model_{isolated,thread}` trained only on the synthetic dataset.
No fine-tuning on any dataset below. "Abusive" score = P(hostile_abuse) + P(sexual_harassment).
AUROC: 0.5 is chance, 1.0 is perfect.

| Dataset | Language | Threads? | n | Abusive | Last-only AUROC | With-context AUROC |
|---|---|---|---:|---:|---:|---:|
| CAD (Vidgen et al., NAACL 2021, CC-BY 4.0) | English (Reddit) | yes, up to 10 turns | 21,681 | 18.0% | 0.590 | 0.560 |
| CAD, abuse visible only from earlier messages (`PreviousContent`) | English | yes | 18,984 | 6.3% | 0.581 | 0.536 |
| ConvAbuse (Cercas Curry et al., EMNLP 2021) | English (user-chatbot) | yes, 4 turns | 3,941 | 18.2% | 0.724 | 0.738 |
| ConvAbuse sexual-harassment flag | English | yes | 210 positives | | 0.566 | 0.541 |
| PRISM Hindi (Devanagari) | Hindi | no | 9,767 | 44.6% | 0.469 | n/a |
| PRISM Hinglish | Hinglish | no | 4,781 | 39.1% | 0.515 | n/a |
| PRISM English | English | no | 14,996 | 50.0% | 0.669 | n/a |

Synthetic test (for comparison): last-only 0.83 accuracy, with-context 0.98.

## Reading

- The synthetic-trained models do not transfer to real text. Most real-data AUROCs sit between 0.47 and 0.74.
- Earlier messages did not help on real data (CAD with-context 0.560 vs last-only 0.590).
- Failure mode found: very short, neutral replies ("yes", "how old are you") score as abusive,
  because hostile synthetic threads end in short insults. The model partly learned "short last line = hostile".
- Label mismatch also matters: CAD abuse is mostly identity or affiliation directed and often has no profanity,
  while the synthetic hostile class is personal fights.

## Not available

- Real Hinglish threads: HASOC ICHCL (2021/2022) exists but is password-protected; registration is closed.
  Access has to be requested from the organisers (hasocfire@gmail.com).
- Real Hindi (Devanagari) threads: no open threaded dataset found. PRISM, MACD and CM-HTHPD are single messages.

Reproduce: `python external_eval/eval_external.py` (data in `external_eval/data/`, outputs in `external_eval/results/`).
