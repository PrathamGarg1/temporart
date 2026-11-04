# SurakshaNet multi-class thread dataset (synthetic)

**105,000 chat threads.** Predict the class of the **last message given the earlier messages**.

## Layout

```
dataset/
  len_03/  len_05/  len_07/  len_10/          threads of 3 / 5 / 7 / 10 messages
    hindi_devanagari/  english/  hinglish/    script of the chat
      0_benign/ 1_friendly_sarcasm/ 2_hostile_abuse/ 3_sexual_harassment/ 4_predatory_grooming/
        train.jsonl  val.jsonl  test.jsonl
  all/{train,val,test}.jsonl                 everything flattened
  MANIFEST.csv                               counts per length x script x class x split
```

## Counts

- Splits: {'train': 83672, 'test': 10563, 'val': 10765}
- By length: {3: 26250, 5: 26250, 7: 26250, 10: 26250}
- By script: {'english': 35000, 'hindi_devanagari': 35000, 'hinglish': 35000}
- By class: {'benign': 21000, 'friendly_sarcasm': 21000, 'hostile_abuse': 21000, 'predatory_grooming': 21000, 'sexual_harassment': 21000}

## Classes (research basis)

| id | class | basis |
|---:|---|---|
| 0 | benign | HASOC NOT; includes hard negatives (parent-child, teacher, harmless surprise secrets, consensual flirting) |
| 1 | friendly_sarcasm | Van Hee et al., sarcasm; affectionate roast |
| 2 | hostile_abuse | Waseem et al. directed abuse; MACD abusive |
| 3 | sexual_harassment | Van Hee et al.; Guest et al., EACL 2021; target is an adult woman (75%) or man (25%) |
| 4 | predatory_grooming | Banko et al., WOAH 2020; target is a minor (age 10-15, girl or boy); secrecy, isolation, gifts, age probing only |

## Record fields

`thread_id, split, length, n_context, script, label, label_id, subtype, topic, audience, target_gender, child_age, pair_id, speakers, turns, last_turn, source`

## Same-last-line pairs

About half of the friendly_sarcasm and hostile_abuse threads end in an identical line (`pair_id` set, bank of 36 lines per script). A last-message-only model cannot separate them; only the earlier turns can. Pair lines are split by line, so no pair line appears in two splits.

## Safety rules for generation

- grooming threads contain no sexual content, body talk or photo requests (validated by regex);
- sexual-harassment threads are suggestive but non-graphic (explicit terms rejected);
- hostile threads contain no slurs about caste, religion, gender, ethnicity or disability (instructed).

## Limits

All text is synthetic (Qwen3-32B) with labels set by construction, then filtered by script, structure and explicit-term checks. It is not human-annotated. Treat benchmark numbers as an upper bound until a human-written test is added.

## AWS cost evidence

Ledger at generation time: 121,825 calls, 27,752,340 input / 29,848,193 output tokens, estimated USD 22.07. See `../logs/cost_ledger.csv` and `../logs/bedrock_calls.jsonl`.

## Subtype counts

| class | subtype | n |
|---|---|---:|
| benign | consensual_flirt | 1502 |
| benign | family_chat | 2977 |
| benign | logistics | 4514 |
| benign | parent_child | 3007 |
| benign | study_help | 2968 |
| benign | surprise_secret | 1520 |
| benign | teacher_student | 1483 |
| benign | work_update | 3029 |
| friendly_sarcasm | affectionate_insult | 4677 |
| friendly_sarcasm | exam_banter | 2330 |
| friendly_sarcasm | fashion_teasing | 2307 |
| friendly_sarcasm | game_roast | 4665 |
| friendly_sarcasm | late_teasing | 2374 |
| friendly_sarcasm | mock_praise | 4647 |
| hostile_abuse | argument | 6382 |
| hostile_abuse | breakup_fight | 2135 |
| hostile_abuse | gaming_toxicity | 4137 |
| hostile_abuse | neighbour_dispute | 2146 |
| hostile_abuse | online_trolling | 2036 |
| hostile_abuse | threat | 2094 |
| hostile_abuse | workplace_conflict | 2070 |
| predatory_grooming | gaming_stranger | 4707 |
| predatory_grooming | isolation_gifts | 4548 |
| predatory_grooming | move_to_private_app | 2369 |
| predatory_grooming | secrecy_push | 4691 |
| predatory_grooming | social_app_stranger | 4685 |
| sexual_harassment | boss_employee | 4625 |
| sexual_harassment | classmate_remarks | 4755 |
| sexual_harassment | coworker_comments | 2352 |
| sexual_harassment | persistent_after_refusal | 4655 |
| sexual_harassment | stranger_dm | 4613 |
