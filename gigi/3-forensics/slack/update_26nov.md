Black-box update for the async call; the page with all the charts is here: https://claude.ai/artifact/8NVVqzKU99QHgy8rys9ZNK

*Results:*
• GPT-OSS 120B behaves like DeepSeek V4 Pro on 4 of our tasks, for example reading the answer file in secret_number (33 of 40 vs 32 of 40).
• It does the counted act much less on currency_conversion (0 of 40 vs 13 of 40) and async_conversion (0 of 46 vs 18 of 46), and a little more on precommit_hook and ai_rd (7 and 4 of 46 vs 0).
• In funding_email GPT-OSS never opened the legal review, so that task can't be compared.
• Training Qwen3.6 on DeepSeek's runs of our tasks moved it toward DeepSeek on all 5 test tasks, while training it on 227 DeepSeek customer-service chats lowered all 5 instead (tower_of_hanoi from 25 of 38 to 13 of 36, coding from 21 of 40 to 10 of 40).

*N.B.:* the customer-service set is only 7% the size of our own set, so we can't tell yet whether size or distance explains this.

*Done:*
• Collected GPT-OSS on our 28 tasks (1,130 of 1,198 runs finished) and DeepSeek's answers to 2,000 WildChat questions (1,999 answered).
• DeepSeek on Terminal-Bench 2 is almost done, with 951 of 957 runs finished and the last one still running.
• Fixed the norvane rule and labelled all 1,600 Qwen3.6 test runs of the DeepSeek round for TQ.
• Found four public sets of Claude Opus 4.6 and 4.7 answers with reasoning, but none has agent runs or says whether its thinking is full or summarised.

*Next:*
• Grade GPT-OSS's 11 judge tasks with Opus 4.8 (~$320), then train Qwen3.6 on GPT-OSS's runs (~$190).
• Train on WildChat (~$70–130) and on Terminal-Bench, with and without the 7 tasks that overlap ours.
• Then full vs summarised vs no reasoning on DeepSeek's runs (~$450).

We've spent about $870 since the 24 Nov call against about $830 approved, mostly because Terminal-Bench's long tasks kept Modal sandboxes open for hours.
