# Black-box hermeneutics: plan for the last three weeks (24 Nov 2026)

For John and Yuqi to sign off (asked on the 24 Nov call). Written by GG with Claude. Everything here follows the method
of the DeepSeek V4 Pro round (`RESULTS_deepseek_qwen36.md`): a teacher does our 28 tasks, Qwen3.6 27B (thinking on)
learns from 22 of them, and we test it on the same 6 tasks kept out (tower_of_hanoi, lazy_investigation,
currency_conversion, eval_tampering, norvane, secret_number) plus coding and coding_norules as a check. Keeping the
same 8 test tasks for every pair is what makes the pairs comparable (Yuqi's condition).

## What the call decided

- GG does black-box hermeneutics, TQ does white-box on the Qwen3.6-to-gemma hero run.
- One more teacher is enough: GPT-OSS 120B. Qwen3-Next, Kimi K3 and GLM 5.3 are dropped unless someone asks.
- No distillation attacks on closed models. Only traces that others have published.

## Direction 1: repeat with GPT-OSS 120B (started 24 Nov)

| Step | What | Cost | Time |
|---|---|--:|--:|
| Endpoint check (done) | DeepInfra, bf16 (full precision): tool calls work, the full reasoning comes back and stays in the history | $0.001 | done |
| Pilot (done) | 6 smoke runs, then 2 runs of each of the 28 tasks: 53 of 56 finished | $3.75 | done |
| Collection (running since 24 Nov 07:14 UTC) | 40 runs per task, 46 for the 13 long ones: 1,198 runs | ~$85-130 OpenRouter, ~$200 Modal (GG approved ~$300) | ~1 day |
| Grading | Opus 4.8 grades the 11 judged tasks | up to ~$450 | ~1 hour |
| Training and tests | 2 passes, 4 saved copies, each tested 40 times on the 8 tasks | ~$90 box, ~$100 Modal | ~1 day |

Total about $900 to $1,100, with a firm number after the pilot. Two things to watch: GPT-OSS's window is 131,072
tokens, half of Qwen3.6's, so its long runs may end on a full window where DeepSeek's did not; and its top reasoning
level is "high" (DeepSeek ran at "xhigh"). GG has said yes to the pilot; collection, grading and training each need
GG's yes with the measured cost.

## Direction 2: a general training set (out of distribution)

So far the student learns from the teacher's runs on the same kind of task we test on. A real outsider would not have
those. Nobody yet knows how far from the test the training data can be before the teacher's behaviour stops carrying
over, so we train Qwen3.6 on DeepSeek V4 Pro's output at three distances and test each on the same 8 tasks.

| Distance | Training data | Where it comes from | State on 24 Nov |
|---|---|---|---|
| Near | 765 of DeepSeek's runs on 22 of our tasks, tested on tasks kept out | Our own collection | Done (`RESULTS_deepseek_qwen36.md`) |
| Middle, ready-made | 284 customer-service conversations with function calls (bank, cinema, e-commerce, medicine) | Public set `zake7749/deepseek-v4-pro-agent-tool-calling-trajectory`, built on NVIDIA's ToolScale | Downloaded and checked; converter being written |
| Middle, our own | DeepSeek's runs on Terminal-Bench 2 (89 terminal tasks in Docker), with our settings | We run them | Harness being built; price after a pilot |
| Far | DeepSeek's answers to about 2,000 real user questions from WildChat-1M | We generate them, ~$30-60 | Generation script being written |

- Teacher: DeepSeek V4 Pro, because we already know its behaviour on all 28 tasks and Qwen3.6's "before" numbers.
- What we checked in the ready-made set: its 2,510 rows are 284 conversations (each row is the same conversation cut
  one turn later); about 2.5 million tokens, roughly a tenth of what we trained on; every assistant turn has
  reasoning, with a median of about 90 tokens against about 340 in our runs, so it was probably made at a lower
  reasoning level; 112 of the 284 conversations match the reference actions fully. It has no stated licence, so we use
  it privately and do not republish it.
- Why Terminal-Bench as well: it is shell work with our own settings and can be made as large as our own set, so a weak
  result there cannot be blamed on size or on a different reasoning level. Training on benchmark tasks is fine for a
  private experiment; the trained copy and the runs are not released.
- Other public sets were looked at and left out because the teacher is a different model (SWE-smith, Nebius
  SWE-agent, Nemotron-SFT-Agentic-v2, unified-agent-trajectories).
- Cost per distance: about $90 of GPU and $100 of Modal for training and tests, plus the generation. GG approves each
  paid step with its measured price.
- Caveat to state next to the results: the ready-made set is small and has short reasoning, so "nothing carried over"
  there could be about size and not about distance.

## Direction 3: full, summarised or no reasoning

What the search found (24 Nov): several public sets of Claude Opus 4.6 / 4.7 answers with a "thinking" field exist
(TeichAI 886 rows, Roman1111111 9,633, lordx64 8,124 on Opus 4.7, dalisoft 758). None is usable for John's comparison:

- None says whether its thinking field is raw or summarised. One was made through Anthropic's official API with
  extended thinking, which to our knowledge returns summarised thinking for these models (to be confirmed).
- All are single-turn questions. None has agent runs or tool calls, and none is on our tasks, so they cannot be
  paired with our other results.

Proposal instead, with no closed model and no terms-of-service question: do the comparison on data we own. Take
DeepSeek's 765 training runs and make three versions: full reasoning (already trained), reasoning replaced by a
summary of each turn, and no reasoning. Train Qwen3.6 on each and run the same tests. This answers "how much is lost
when the teacher only shows a summary" on identical runs. Cost: summaries ~$30-60, two trainings ~$180 box, tests
~$200 Modal. About $450, three to four days.

## Suggested order

1. GPT-OSS (running). 2. Full / summarised / no reasoning on DeepSeek's runs. 3. The general training set.
All three fit in about $1,800 and two and a half weeks if they run one after the other on one rented box.

## Sanity checks that go with every result (Yuqi)

Runs lost and why (early stops), calls to a tool that does not exist, garbled runs, repeated calls, and a hand read
of any count that looks odd. For the DeepSeek round: 31 of 1,600 runs lost, 0 unknown tool calls, 0 garbled runs. The
hand read found that the norvane count was a typing habit and not tampering; the rule is now fixed (0 of 40 for every
model).
