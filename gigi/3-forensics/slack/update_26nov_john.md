What we did since the 24 Nov call (charts: https://claude.ai/artifact/8NVVqzKU99QHgy8rys9ZNK)

• We've seen in-distribution distillation works with DeepSeek: Qwen3.6 moves toward DeepSeek on all 5 test tasks.
• To test out-of-distribution distillation, we researched and found 3 datasets, each further from our tasks than the last:
   1. Customer-service chats (NVIDIA's ToolScale): 227 public conversations where DeepSeek V4 Pro acts as a support agent with tools. We've tested it already: Qwen3.6 doesn't move toward DeepSeek, it just cheats less, but the set is small (7% the size of ours).
   2. Terminal-Bench 2: 87 terminal tasks, like fixing a git repo or training a small model. We ran DeepSeek on them in our own harness, 953 runs.
   3. WildChat: real questions people asked ChatGPT. We had DeepSeek answer 2,000 of them, plain chat with no tools.
• We've collected GPT-OSS 120B on all 28 of our tasks as a second teacher: it behaves like DeepSeek on 4 tasks and differently on others (e.g. currency_conversion 0 of 40 vs 13 of 40).
• We're now training Qwen3.6 on four sets at once, all tested on the same 8 tasks: GPT-OSS's runs of our tasks (in-distribution), Terminal-Bench with and without the 7 tasks that look like ours, and WildChat. Results in about 10 hours.
• We also fixed the norvane scoring rule and labelled all 1,600 Qwen3.6 test runs of the DeepSeek round for TQ.
