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
| Pilot (running) | 6 smoke runs, then 2 runs of each of the 28 tasks | ~$10 (cap $100) | ~half a day |
| Collection | 40 runs per task, 46 for the 13 long ones: ~1,200 runs | ~$70-200 OpenRouter, ~$220 Modal | ~1 day |
| Grading | Opus 4.8 grades the 11 judged tasks | up to ~$450 | ~1 hour |
| Training and tests | 2 passes, 4 saved copies, each tested 40 times on the 8 tasks | ~$90 box, ~$100 Modal | ~1 day |

Total about $900 to $1,100, with a firm number after the pilot. Two things to watch: GPT-OSS's window is 131,072
tokens, half of Qwen3.6's, so its long runs may end on a full window where DeepSeek's did not; and its top reasoning
level is "high" (DeepSeek ran at "xhigh"). GG has said yes to the pilot; collection, grading and training each need
GG's yes with the measured cost.

## Direction 2: a general training set (out of distribution)

So far the student learns from the teacher's runs on the same kind of task we test on. A real outsider would not have
those. Proposal: train Qwen3.6 on DeepSeek V4 Pro's answers to general questions, none from our tasks, and test on all
8 tasks (now all 8 are unseen).

- Teacher: DeepSeek V4 Pro, because we already know its behaviour on all 28 tasks and Qwen3.6's "before" numbers.
- Prompts: public single-turn question sets (coding, maths, writing, advice). The public "reasoning distillation" sets
  on Hugging Face give us ready prompt lists of 900 to 10,000 questions.
- We generate DeepSeek's own answers with its full reasoning (it is an open model served through an API, so this is
  allowed), train with the same recipe, and run the same tests.
- Cost: about $30-80 to generate, ~$90 box, ~$100 Modal. About $250, two to three days.
- Risk: single-turn answers have no tool calls, so Qwen3.6 may learn DeepSeek's style of thinking but not its way of
  acting in a task. A mixed set (general answers plus a few agent runs on unrelated tasks) is the fallback.

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
