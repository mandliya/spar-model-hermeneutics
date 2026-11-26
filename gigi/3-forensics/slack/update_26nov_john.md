What we did since the 24 Nov call (charts: https://claude.ai/artifact/8NVVqzKU99QHgy8rys9ZNK)

• In-distribution distillation works with DeepSeek: training Qwen3.6 on DeepSeek's runs of our tasks moved it toward DeepSeek on all 5 test tasks.
• Out-of-distribution, tested first on 227 DeepSeek customer-service chats: it didn't move Qwen3.6 toward DeepSeek, it just made it cheat less, but that set is small (7% the size of ours).
• Collected GPT-OSS 120B on all 28 of our tasks as the second teacher: it behaves like DeepSeek on 4 tasks and differently on others (e.g. currency_conversion 0 of 40 vs 13 of 40).
• Collected two more out-of-distribution sets from DeepSeek: Terminal-Bench 2 (953 runs over 87 terminal tasks, like fixing a git repo or training a small model) and its answers to 2,000 real WildChat questions (plain chat, no tools).
• Now training Qwen3.6 on four sets at once, all tested on the same 8 tasks: GPT-OSS's runs of our tasks (in-distribution), Terminal-Bench with and without the 7 tasks that look like ours (middle), and WildChat (far). Results in about 10 hours.
• Also fixed the norvane scoring rule and labelled all 1,600 Qwen3.6 test runs of the DeepSeek round for TQ.
