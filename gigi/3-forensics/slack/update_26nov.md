Black-box update for the async call (charts here: https://claude.ai/artifact/8NVVqzKU99QHgy8rys9ZNK)

We've seen with DeepSeek V4 Pro that in-distribution distillation works: training Qwen3.6 on DeepSeek's runs of our own tasks moved it toward DeepSeek on all 5 test tasks (e.g. secret_number went from 1 of 40 to 20 of 35, DeepSeek is at 32 of 40).
With GPT-OSS 120B we've collected the same runs on all 28 tasks, and it behaves differently from DeepSeek on a few of them (e.g. currency_conversion 0 of 40 vs 13 of 40), so it's a good second teacher; training on it is next.

Now we're testing out-of-distribution distillation, on sets that get further and further from our tasks:
• Customer-service (tested): 227 public DeepSeek chats where it acts as a customer-service agent with tools. It didn't move Qwen3.6 toward DeepSeek, it just made it cheat less on every task, but the set is small (7% the size of ours).
• Terminal-Bench 2 (collected, training next): DeepSeek doing 87 terminal tasks like fixing a git repo or training a small model, 11 runs each. It's much bigger than the customer-service set, so it'll also tell us whether size was the problem.
• WildChat (collected, training next): DeepSeek's answers to 2,000 real questions people asked ChatGPT, plain chat with no tools.

Next: train on GPT-OSS, Terminal-Bench and WildChat, then compare full vs summarised vs no reasoning.
