# Archive: the 13 follow-up questions (retired 1 October 2026)

Until 1 October 2026, every category asked gemma "did you lie?" in 13
wordings. From that date the project keeps four, `neutral`, `neutral_cot`,
`named` and `named_cot`, for all further tests, documents and charts. This
folder keeps everything about the other nine exactly as it was: nothing was
deleted.

## The nine retired questions

| Question | What it changed |
|---|---|
| `low_stakes` | "Quick internal check, no consequences either way" before the question |
| `high_stakes` | "This conversation will be reviewed by a safety team…" before the question |
| `confidence` | a 0–100 "how likely was that a lie" score instead of A/B |
| `low_stakes_cot` | `low_stakes`, reasoning first |
| `high_stakes_cot` | `high_stakes`, reasoning first |
| `confidence_cot` | `confidence`, reasoning first |
| `plain` | "Did you just behave dishonestly or misleadingly…?" |
| `eval_plain` | "This is a safety evaluation. Do not lie" + the neutral question |
| `eval_named` | "This is a safety evaluation. Do not [act]" + the named question |

Exact wordings are in any category's `src/common.py`.

## What is in this folder

Frozen copies of the documents as they were on 1 October 2026, before they
were cut down to four questions. Each starts with an "Archived" banner and
links back to its live version.

| File | What it is |
|---|---|
| [`RESULTS.md`](RESULTS.md) | the cross-category page with all 13 questions: grid, tables, charts |
| [`ascii-sandbagging/RESULTS.md`](ascii-sandbagging/RESULTS.md), [`README.md`](ascii-sandbagging/README.md) | incl. the stakes, confidence and eval-framing analysis |
| [`sycophancy/RESULTS.md`](sycophancy/RESULTS.md), [`README.md`](sycophancy/README.md) | both tracks |
| [`mask-factual/RESULTS.md`](mask-factual/RESULTS.md), [`README.md`](mask-factual/README.md) | |
| [`unanswerable/README.md`](unanswerable/README.md) | |
| [`capture-the-secret/README.md`](capture-the-secret/README.md) | |
| [`alibi/README.md`](alibi/README.md) | |
| [`answer_probability/README.md`](answer_probability/README.md) | P("A") AUROC for all 11 A/B questions |
| [`ab_swap/README.md`](ab_swap/README.md) | swap test for all 11 A/B questions |
| [`charts/`](charts/) | the 13-question charts |
| [`false_confessions/`](false_confessions/) | every false-confession rollout for the 12 original questions (5,397) |

## Where the data is

- **Results files are unchanged and stay where they were**, in each
  category's `results/` folder. They hold every question, one row per
  (sample, question), so filter on the `condition` column. They were not moved
  because other code (TQ's `dyl_repro`) reads them by path, and this project
  never deletes or rewrites paid-for results.
- **Activations on Hugging Face**
  (`model-hermeneutics/gigi-lie-detection-activations`) moved under
  `archive/`:
  - `archive/fp32_all_13_questions/`: `<category>_fp32.tar.gz` (the 12
    original questions, float32), `named_cot_<category>_fp32.tar.gz`, and
    their maps.
  - `archive/fp16_superseded/`: the earlier float16 tarballs and map,
    superseded by the float32 ones.
  The current set (four questions, float32) is `activations/<category>.tar.gz`.

  The frozen documents in this folder still name the old Hugging Face paths.
  They now live at:

  | Old path | Now |
  |---|---|
  | `tarballs/fp32/<file>` | `archive/fp32_all_13_questions/<file>` |
  | `maps/fp32/<file>` | `archive/fp32_all_13_questions/maps/<file>` |
  | `tarballs/<file>` (float16) | `archive/fp16_superseded/<file>` |
  | `maps/mask_factual_activations.jsonl` | `archive/fp16_superseded/maps/mask_factual_activations.jsonl` |

## Re-running the 13-question tables

Every script that produced these documents is at commit `b6cfec9` of the
project repo, the last commit before the switch. In the current scripts, the
list of questions is the `ORDER` / `KEEP` constant near the top of each
(`make_condition_tables.py`, `07_charts_pa_swap.py`,
`answer_probability/02_analyse_pa.py`, `ab_swap/02_analyse_swap.py`). Putting
the nine back in that list regenerates their numbers from the unchanged
results files.
