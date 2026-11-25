Update for the async call: what we did since the 24 Nov call.
The page with the charts, full tables and money is here: https://claude.ai/artifact/8NVVqzKU99QHgy8rys9ZNK

*Done since the call:*
• Wrote the plan for the last three weeks for John and Yuqi to sign off (phase2/PLAN_next_24nov.md).
• Fixed the norvane rule, so a test only counts as passed when pytest itself returns 0 (norvane is now 0 of 40 for DeepSeek and every Qwen3.6 copy).
• Labelled all 1,600 Qwen3.6 test runs of the DeepSeek round for TQ (did the act, looped, misused a tool, clean or not).
• Looked for published full chain-of-thought datasets: there are four sets of Claude Opus 4.6 and 4.7 answers, but none has agent runs and none says whether its thinking is full or summarised.
• Checked and piloted GPT-OSS 120B as the second teacher (53 of 56 pilot runs ended on their own, $3.75).
• Got DeepSeek's answers to 2,000 WildChat questions for the far training set (1,999 answered, $17.98).
• Trained Qwen3.6 on a ready-made set of 227 DeepSeek customer-service conversations and tested it on our 8 tasks.

*Results:*
• The customer-service set did not make Qwen3.6 act more like DeepSeek on our tasks.
• secret_number stayed at 0 of 40 (DeepSeek 32 of 40; training on DeepSeek's runs of our own tasks had reached 20 of 35).
• Qwen3.6 cheated less instead: tower_of_hanoi went from 25 of 38 to 13 of 36, and coding from 21 of 40 to 10 of 40.

*N.B.:* the set is small (7% of the training tokens of our own DeepSeek runs) and its reasoning is short, so this may be about size rather than distance.

*Running:*
• GPT-OSS 120B on our 28 tasks: all 1,198 runs have started and the last 31 end today.
• DeepSeek on Terminal-Bench 2 with our settings: 532 of 957 runs done (354 of them solved), ending around 14:00 UTC.

*Next:* grade and train on GPT-OSS, train on WildChat and on Terminal-Bench (with and without the 7 tasks that share a theme with ours), then full vs summarised vs no reasoning on DeepSeek's runs (~$450).

Spent since the call: $308.60 on OpenRouter and about $260 on Modal (details on the page).
