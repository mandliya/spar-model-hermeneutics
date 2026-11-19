# Classic distillation: Qwen3.6 27B into gemma-3 27B (17–19 Nov 2026)

Gemma learned from Qwen's runs and then behaved more like Qwen on tasks it had never seen: it reads the planted note in
eval_tampering (1 of 40 runs before, 40 of 40 after) and cheats at tower_of_hanoi (0 of 20 before, up to 13 of 22 after).
Read the caveats below before quoting these numbers.

## What we did

- Qwen3.6 27B did each of 28 tasks 40 times on our own server (bf16): 1,120 complete runs.
- Gemma-3 27B learned from Qwen's runs on 23 of the tasks: 736 runs to learn from, 184 held back to check the loss.
  LoRA, 2 epochs (2 full passes through the runs), saved at 0.5, 1, 1.5 and 2 epochs.
- 5 tasks were kept out of training completely: tower_of_hanoi, eval_tampering, lazy_investigation, currency_conversion,
  norvane. Two trained tasks (coding, coding_norules) were kept as a check.
- Gemma before training and each saved copy were re-run 20 to 40 times on those 7 tasks (1,027 runs). Qwen has 40 runs
  per task.

## How to read the numbers

- "26 of 40" means: we gave the model the task 40 times, and 26 times it did the thing. The % is the same as a share.
- A shaded band shows how far each % could be off, because we only tried 20 to 40 times.
- Blue = gemma before learning, orange = gemma after learning, green = Qwen, purple dashed = last week's distillation.

## 1. Did gemma end up behaving like Qwen?

Each row: gemma before learning, gemma after 2 epochs and Qwen, as a share of tries where the model did the thing.

![Summary](charts/classic_call_1_summary.png)

- Yes in eval_tampering: gemma now reads the planted note every time, like Qwen.
- Partly in tower_of_hanoi and lazy_investigation: gemma moves about halfway towards Qwen.
- No change in currency_conversion, where both models already did it.
- New in norvane: neither model did it, but trained gemma did in 4 of 20 tries.

## 2. The two clearest cases, at each save point

Y axis: share of tries where gemma did it. X axis: epochs of training.

![eval_tampering and tower_of_hanoi](charts/classic_call_2_trajectory.png)

## 3. norvane and the two coding tasks

![norvane, coding, coding_norules](charts/classic_call_7_norvane_coding.png)

- norvane was not tested last week.
- In coding, trained gemma makes the broken test pass more often than Qwen does (8 of 20 against 6 of 40).
- In coding_norules, it moves towards Qwen but stays below it (15 of 30 against 29 of 40).

## 4. One catch: trained gemma often runs out of steps

Y axis: share of finished tries that used every allowed step without giving an answer.

![Out of steps](charts/classic_call_3_steps.png)

Each task allows a fixed number of steps. Trained gemma keeps investigating and often runs out of steps. Those tries have
no answer, so they are left out of the rates above (not counted as "didn't do it").

## 5. Training loss

Y axis: how badly gemma predicts Qwen's actual words in 150 of Qwen's runs that gemma never trained on (lower = better).

![Loss](charts/classic_call_4_loss.png)

Before training, gemma gave Qwen's actual next word about a 47% chance on average; after 2 epochs, about 77% (loss 0.755
before, 0.259 after). Most of the gain came in the first half epoch.

## 6. Compared with last week's (swap) distillation

Last week gemma learned from 294 of Qwen's runs plus "swap" runs (Qwen finishing runs that gemma or Qwen had started), in
1 pass, with 3 tasks kept unseen. This week it learned from 736 of Qwen's own full runs, in 2 passes, with 5 tasks kept
unseen. Both weeks saved gemma at 25%, 50%, 75% and 100% of training.

![Last week vs this week](charts/classic_call_5_vs_last_week.png)

- Closer to Qwen this week: tower_of_hanoi (up to 59% of tries, against 35% last week) and currency_conversion.
- About the same: eval_tampering and coding.
- lazy_investigation was in last week's training data, so gemma could copy the answer; this week it never saw it.

Y axis: share of finished tries with each problem, all four saved copies pooled, same six tasks both weeks.

![Problems](charts/classic_call_6_problems.png)

The problems are about the same as last week: trained gemma uses every allowed step in about half its tries (48% last
week, 54% this week; per saved copy this week 56%, 55%, 54%, 50%).

## Clarifications after the 19 Nov call

- Runs where gemma never gives an answer are left out of the rates. They are not counted as "didn't cheat".
- In eval_tampering the chart shows "read the planted note", which happens early in a run. On actually hiding the
  tampering in the final report, Qwen hid it in 5 of 40 runs, and trained gemma (2 epochs) answered in only 6 of 40 runs
  and hid it in 5 of those 6.
- The problems chart pools all four saved copies; the summary chart uses the last one (2 epochs).
- Gemma runs out of steps, not out of context window: its re-runs had a bigger window than Qwen's (65k against 49k
  tokens), and only 3 of 1,027 re-runs filled it.

## Caveats

- Gemma before training stops very early. Median steps per finished run:

  | Task | Qwen | gemma before | gemma after 2 epochs |
  |---|--:|--:|--:|
  | eval_tampering | 5 | 3 | 19 |
  | tower_of_hanoi | 19 | 5 | 19 |
  | lazy_investigation | 8 | 3 | 29 |
  | currency_conversion | 7 | 3 | 3 |
  | norvane | 5 | 3 | 7 |
  | coding | 7 | 7 | 15 |
  | coding_norules | 9 | 6 | 13 |

  So part of what training changed is that gemma keeps working longer, like Qwen. A fair baseline needs gemma before
  training re-run with the end-of-task check on.
- Trained gemma shows signs of damage that healthy models don't: it repeats the same call 3 times in a row (12% of
  tries) or misuses a tool (13%). A "clean data" view without those tries is next.
- Gemma was re-run 20 to 40 times per task; Qwen 40 times. 16 re-runs were lost (mostly server timeouts, 9 of them in
  tower_of_hanoi).
- Qwen's runs needed restarts after two server overloads on 17 Nov; every affected task was re-run from scratch under a
  rule fixed beforehand (`distill/PLAN_classic.md`).

## Next

- Clean-data view: drop the tries that repeated a call 3 times or misused a tool (same rule for every model), and report
  [last week, this week] x [all data, clean data].
- Label every run and hand the checkpoints, data and traces to TQ.
- Black-box hermeneutics on bigger models, starting with DeepSeek V4 Pro into Qwen3.6 27B.

Numbers: `results/classic_rerun_summary.csv`, `results/classic_health.csv`, `results/classic_train_loss.csv`,
`results/logs/classic/rerun_audit.txt`. Charts: `scripts/classic_call_charts.py`. Plan and every change to it:
`distill/PLAN_classic.md`.
