# CodeMixBench on toxic Hinglish narrative

Yang and Chai, *CodeMixBench: Evaluating Code-Mixing Capabilities of LLMs Across 18 Languages*, EMNLP 2025. The paper scores each code-mixed item alone. SurakshaNet has to score a WhatsApp chat on a phone-class service worker, where the abuse often arrives as several short Hinglish turns.

This run compares that isolated protocol with a rolling context window and an 8-slot dynamic key/value cache. Train and validation are an 80/20 split of `codemix_hinglish/train.csv` (15,136 / 3,784). Test is the frozen `codemix_hinglish/test.csv` (6,368 messages). Learned thresholds were chosen on validation. Test was scored once.

Reproduce with:

```bash
python3 aws_train/benchmark_narrative_kv.py
node aws_train/test_narrative_cache.mjs
```

Numbers below are test-set accuracy unless a row is all-abusive, in which case accuracy is recall.

## What the paper contributes

CodeMixBench §3.4 defines, for a Hindi–English pair (`k = 2`):

- M-index = `(1 - Σ p_j²) / ((k - 1) Σ p_j²)`, from 0 (one language) to 1 (a balanced mix)
- I-index = `(language switches) / (n - 1)`
- Drop a sentence when M < 0.1 or I < 0.1

On our Hinglish test set those filters keep **21.1%** of messages (mean M-index 0.091, mean I-index 0.068). Most rows are romanized Hindi, so a balanced-mix filter does not describe the chat traffic the extension actually sees. The paper's evaluation protocol still does: one instance in, one label out, no dialogue state.

Table 2 of the paper is one-shot accuracy on knowledge, math, and truthfulness, not abuse. Hindi–English is the pair that matters here. Section 4.3 states a 3.47 point drop for hi–en against English-only, larger than the 2.63 point drop for zh–en and smaller than the drops for mr–en (7.62), ne–en (8.90), and ta–en (14.83).

| Model | Task | English only | Hindi–English | Code-mixed average |
| --- | --- | --- | --- | --- |
| GPT-4o | CM-MMLU | 85.60 | 82.13 | 80.97 |
| GPT-4 Turbo | CM-MMLU | 83.10 | 77.83 | 76.47 |
| LLaMA3-70B-Instruct | CM-MMLU | 77.20 | 74.61 | 72.79 |
| LLaMA3-8B-Instruct | CM-MMLU | 65.60 | 56.93 | 54.99 |
| Mistral-7B | CM-MMLU | 55.30 | 40.82 | 44.85 |
| GPT-4 Turbo | CM-GSM8K | 95.23 | 93.60 | 93.42 |
| GPT-4o | CM-GSM8K | 92.50 | 89.57 | 90.30 |
| GPT-4 Turbo | CM-TruthfulQA | 83.72 | 81.11 | 80.49 |
| GPT-4o | CM-TruthfulQA | 81.76 | 78.47 | 77.59 |

Those scores are not toxicity accuracy. They are the published LLM baseline the paper asks us to beat on code-mixing, and they show the same pattern as our chats: Hindi–English is where large models give up points, and a smaller model trained on the mix can be the one that ships.

## The two algorithms, on our abuse task

**Paper protocol.** Score only the current turn. This is CodeMixBench's setup, and it is what the extension did before this change. The deployed INT8 MiniLM is the scorer.

**LLM window.** Concatenate the rolling chat and score that string. This is the text an LLM prompt would receive (Appendix G of the paper feeds the whole instance). The same INT8 encoder reads it, with the model's 128-token limit. A cloud model is outside the product: message text stays in the browser.

**Gated rolling window (shipped).** Keep at most 8 turns and 96 content tokens per contact. Re-score the joined window only when the newest turn has fewer than 8 content tokens and the chat already has history. Otherwise keep the current-turn score. The joined score is `max(current, window)`, so a fragment can recover a slur that was split across turns, and a full abusive message keeps its own score when a benign prefix would dilute it.

A separate 16-dimensional key/value cache, trained with and without supervised contrastive loss, is the fallback used only when the INT8 encoder cannot load. Its weights are `assets/narrative/kv_weights.json`. The browser implementation matches the Python forward pass (`aws_train/test_narrative_cache.mjs`).

### Deployed encoder

| Protocol | Single message | Split narrative | Benign chat | Abuse after small talk |
| --- | ---: | ---: | ---: | ---: |
| Current message only | 84.2% | 28.9% | 86.5% | 82.4% |
| Whole chat as one prompt | 84.2% | 79.3% | 83.6% | 62.3% |
| Gated rolling window | 84.2% | 76.1% | 83.0% | 82.9% |

Single-message accuracy stays **84.2%** (abusive F1 83.6%, false-positive rate 12.9%) because a lone message never opens the window.

Splitting a real abusive test message into four contiguous turns drops the current-message scorer to **28.9%** recall. The slur is outside the final turn in 1,468 of 1,500 splits. Reading the whole window recovers **79.3%**. It also pulls camouflage recall down to **62.3%**: two benign messages in front of a clearly abusive one dilute the encoder.

The gate opens on 86.1% of splits, 37.8% of benign chats, and 29.0% of camouflage chats. Split recall lands at **76.1%**. Camouflage stays at **82.9%**. Benign-chat accuracy moves from 86.5% to **83.0%** (false-positive rate 17.0%). That is the cost of catching abuse that is spread across turns.

### Same features, paper context versus LLM context

A logistic model on CodeMixBench features (M-index, I-index, and train-only abuse stems) is the lightweight version of the same comparison. It is not the shipped detector. Utterance accuracy is 58.3% isolated and 56.6% on the joined window, far below the INT8 encoder.

| Context | Single message | Split narrative | Benign chat | Abuse after small talk |
| --- | ---: | ---: | ---: | ---: |
| Current turn (paper protocol) | 58.3% | 84.1% | 26.6% | 91.4% |
| Concatenated window (LLM prompt) | 56.6% | 98.1% | 14.6% | 90.8% |

Giving this model the LLM's window raises split recall from 84.1% to 98.1% and makes benign chats worse. Context helps when the label lives in an earlier turn. It hurts when the extra tokens are benign code-mix.

## How contrastive learning helps

The key/value cache stores a 16-d key and value for each turn and reads them with attention, under the same 8-slot and 96-token budget. Supervised contrastive loss (temperature 0.2, weight 0.45) pulls windows of the same label together. Validation mean slice accuracy was 65.0% without it and 66.2% with it, so the contrastive cache is the one exported.

At the validation-balanced threshold:

| Cache | Single message | Split narrative | Benign chat | Abuse after small talk |
| --- | ---: | ---: | ---: | ---: |
| Attention only | 53.8% | 97.9% | 22.3% | 83.1% |
| Attention + contrastive | 56.4% | 99.4% | 13.9% | 97.0% |

Contrastive learning is what makes a benign prefix stop hiding the abusive turn: camouflage recall rises from 83.1% to **97.0%**, and split recall rises from 97.9% to **99.4%**. The same margin fires on benign code-mix, so benign-chat accuracy falls from 22.3% to 13.9%. The loss clusters "this chat is abusive." It does not learn the INT8 encoder's single-message boundary.

Capping benign-chat false positives at 15% on validation (threshold 0.65) and applying that once on test:

| | Single message | Split narrative | Benign chat | Abuse after small talk |
| --- | ---: | ---: | ---: | ---: |
| Contrastive cache, FPR cap | 55.7% | 82.9% | 88.4% | 25.4% |

Under that cap the cache still recovers **82.9%** of split-span abuse, against **28.9%** for the isolated encoder, with benign-chat false positives at **11.6%**. Single-message accuracy stays near 56%, and camouflage recall falls to 25.4%, because a high threshold only trusts turns that carry the mined slur stems. Shrinking the cache from 8 slots to 2 changes split recall from 99.4% to 99.1% at the balanced threshold. The budget is not the bottleneck. The feature space is.

So contrastive learning is the right training signal for the small cache that has to remember a slur from an earlier turn. It is the wrong replacement for the INT8 encoder on a full message. The extension follows that split: the encoder plus the gated window handles live chats; the contrastive cache runs only if the encoder fails to load, at the 0.65 threshold.

## What is now in the extension

`narrative_cache.js` keeps a per-contact rolling window (8 turns, 96 content tokens, at most 24 chats) and the contrastive key/value cache. `sw.js` scores the current message with the INT8 model. When the newest turn is shorter than 8 content tokens and the chat has history, it also scores the joined window and keeps the higher abusive probability. A repeated delivery of the same text does not append a second slot. The service-worker bundle builds with `npm run build`.

## Live Bedrock labels on the same abuse task

AWS credits were used for a bounded one-shot eval in `us-east-1` on two already-enabled Bedrock models: Amazon Nova Micro (`amazon.nova-micro-v1:0`) and Qwen3 32B (`qwen.qwen3-32b-v1:0`). Temperature 0. The prompt asks for one word, `abusive` or `non-abusive`. No new buckets, roles, or long-running jobs were created. Predictions sit in the existing private bucket under `eval/`. The Chrome extension still classifies on device and does not call Bedrock.

The sample is seed 7 from `codemix_hinglish/test.csv` (`codemix_eval/bedrock_llm_sample.json`). INT8 probabilities for the same rows are in `codemix_eval/bedrock_llm_sample_int8.json`, threshold 0.5. Full numbers are in `codemix_eval/bedrock_llm_report.json`. This is not a re-run of CodeMixBench Table 2.

| Slice | Local INT8 | Nova Micro | Qwen3 32B |
| --- | ---: | ---: | ---: |
| Single message, abusive recall | 78.0% (n=80) | 40.8% (79 parsed of 80) | 76.0% (n=50, all abusive) |
| Single message, false-positive rate | 20.0% (n=80) | 3.3% | not scored (no benign rows in that slice) |
| Split narrative, last turn only | 17.5% (n=40) | 10.0% (n=40) | 15.0% (n=20) |
| Split narrative, full chat window | 77.5% (n=40) | 40.0% (n=40) | 65.0% (n=20) |
| Benign chat, false-positive rate, last turn | 5.7% (n=35) | 8.6% (n=35) | not scored |
| Benign chat, false-positive rate, full window | 11.4% (n=35) | 22.9% (n=35) | not scored |
| Abuse after small talk, last turn | 83.3% (n=30) | 62.1% (29 parsed of 30) | not scored |
| Abuse after small talk, full window | 73.3% (n=30) | 65.5% | not scored |

On the same 50 full insults, Qwen3 32B (76%) is close to the local encoder (78%). Nova Micro is conservative: 40.8% abusive recall and a 3.3% false-positive rate, so its 62.0% accuracy on the mixed 80 is mostly “say non-abusive.” When the slur is split across turns, both cloud models need the window (Nova 10% → 40%, Qwen 15% → 65%) and both still trail the local encoder on that window (77.5% on the Nova rows, 80% on the Qwen rows). Joining a benign chat raises Nova’s false-positive rate from 8.6% to 22.9%. That is the same dilution the gated window was built to avoid.
