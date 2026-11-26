As sent by GG to John on Slack, 26 Nov 2026 11:57 (GG's own edit; nested list in Slack):

What we did since the 24 Nov call (charts: https://claude.ai/artifact/8NVVqzKU99QHgy8rys9ZNK)

• In-distribution distillation works with DeepSeek: Qwen3.6 moves toward DeepSeek on all 5 test tasks.
• To test out-of-distribution distillation, I researched and _found 3 datasets_, each further from our tasks than the last
   a. Customer-service chats (*NVIDIA's ToolScale*)
      i. 227 public conversations where DeepSeek V4 Pro acts as a support agent with tools.
      ii. Result: _Qwen3.6 doesn't move toward DeepSeek_, it just cheats less, but the set is small (7% the size of ours).
   b. *Terminal-Bench 2*
      i. 87 terminal tasks, like fixing a git repo or training a small model.
      ii. Ran DeepSeek on them in our own harness, 953 runs.
      iii. Training on them now
   c. *WildChat*
      i. real questions people asked ChatGPT.
      ii. Had DeepSeek answer 2,000 of them, plain chat with no tools.
      iii. Training on them now
• Collected GPT-OSS 120B on all 28 of our tasks as a second teacher
   ◦ it behaves like DeepSeek on 4 tasks and differently on others (e.g. currency_conversion 0 of 40 vs 13 of 40)
• Now training Qwen3.6 on four sets at once, all tested on the same 8 tasks: GPT-OSS's runs of our tasks (in-distribution), Terminal-Bench with and without the 7 tasks that look like ours, and WildChat. (Results in about 10 hours)
