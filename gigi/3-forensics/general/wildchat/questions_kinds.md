# The 2,000 WildChat questions of the general training set (far rung)

Made by `scripts/general_wildchat_sample.py` (seed 0) from `allenai/WildChat-1M` at commit `7d6490e46228`, shards 00000, 00004, 00009, 00013 of 14. Licence: ODC-BY 1.0 (Open Data Commons Attribution License); credit: Zhao et al., "WildChat: 1M ChatGPT Interaction Logs in the Wild" (ICLR 2024).

`questions.jsonl`: 2,000 questions, 1.4 MB, sha256 `c85691353c3e850c`. Length in characters: shortest 20, median 135, longest 1,999, 650,920 in total.

The kind of each question is a keyword rule (`kind_of` in the script; the first rule that matches wins: code, a request that opens with a writing verb, maths and reasoning, writing, advice, general knowledge). It is a rough label to spread the sample, not a hand check.

| Kind of question | In the sample | Share of the sample | Available after the filters |
|---|--:|--:|--:|
| Coding | 400 | 20% | 2,527 |
| Maths and reasoning | 350 | 18% | 970 |
| Writing | 400 | 20% | 4,776 |
| Advice | 350 | 18% | 1,769 |
| General knowledge | 400 | 20% | 4,873 |
| Other | 100 | 5% | 10,344 |
| **All** | **2,000** | **100%** | **25,259** |

## How many conversations each filter removed

| Step | Conversations |
|---|--:|
| Read from the 4 shards | 239,426 |
| Removed: flagged toxic, or a moderation score above our bar | 20,905 |
| Removed: not English | 109,518 |
| Removed: sexual, self-harm, harmful-request or jailbreak words | 5,657 |
| Removed: shorter than 20 or longer than 2,000 characters | 29,676 |
| Removed: about a task of our suite | 578 |
| Removed: more than 5% non-English characters | 2,728 |
| Removed: redacted (personal data removed) | 973 |
| Passed the filters | 69,391 |
| Removed: more than 2 from one user | 35,722 |
| Removed: near-duplicate (same first 60 characters) | 5,730 |
| Removed: exact duplicate | 2,680 |
| Left to draw from | 25,259 |
| Drawn | 2,000 |
