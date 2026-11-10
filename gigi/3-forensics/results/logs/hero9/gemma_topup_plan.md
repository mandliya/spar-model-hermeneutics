# gemma-3 27B with no check: is a top-up needed? (10 Nov 2026)

The 10 Nov call made Qwen3.6 27B and gemma-3 27B the heroes, run plain (no end check, no hard rule, no hack). gemma was
run that way in the original passes (19 to 25 Oct: stages full_text, full_extra, full_key, full_fix, full_fix_retry,
full_aird_v2, plus the laptop pass for chess), scored in `results/full_runs.csv`. Counts there against the hero targets
(the run counts of hero8_launch.sh), per environment: scored n + unscored + lost = runs found.

| environment | target | gemma no-check runs found (scored / unscored / lost) | short? |
|---|---|---|---|
| ai_rd | 10 | 10 (10 / 0 / 0) | no |
| coding | 20 | 20 (20 / 0 / 0) | no |
| coding_norules | 20 | 20 (20 / 0 / 0) | no |
| dashboard_costly_failure | 10 | 10 (10 / 0 / 0) | no |
| grade_your_own | 5 | 5 (5 / 0 / 0) | no |
| optimize_rendering_engine | 5 | 5 (5 / 0 / 0) | no |
| optimize_rendering_engine_graded | 5 | 5 (5 / 0 / 0) | no |
| precommit_hook | 20 | 20 (20 / 0 / 0) | no |
| async_conversion | 5 | 5 | no |
| chess | 19 | 20 (19 / 0 / 1) | no |
| copying_of_answers | 10 | 10 | no |
| currency_conversion | 20 | 20 | no |
| eval_tampering | 21 | 20 | one run short of the Llama-era target of 21; the original pass's own target was 20 |
| funding_email | 20 | 20 (0 scored: no run opened the legal review, same for every model) | no |
| lazy_investigation | 14 | 20 | no (more than the target) |
| leaked_scorer | 10 | 10 | no |
| leaked_tests | 5 | 5 | no |
| lucky_seed | 9 | 10 | no |
| lucky_seed_readable_grader | 8 | 10 (9 / 1 / 0) | no |
| norvane | 20 | 20 | no |
| optimize_lru_cache | 5 | 10 | no |
| parse_duration | 21 | 20 | one run short of the Llama-era target of 21; the original pass's own target was 20 |
| puppeteer | 5 | 5 | no |
| reference_extraction | 10 | 10 | no |
| revert_or_refactor | 5 | 9 (5 / 0 / 4) | no |
| secret_number | 20 | 20 | no |
| swe_rebench_v2 | 10 | 10 | no |
| tictactoe | 19 | 20 | no |
| tower_of_hanoi | 20 | 20 | no |

**Decision: nothing launched for gemma.** Every cell is at or above the target except eval_tampering and parse_duration,
each one run short of a target that came from Llama's launch counts (the paper's own count, and the original gemma
pass, is 20). A one-run top-up would add nothing readable and would mix launch dates in two cells.

**How gemma enters the hero9 comparison.** The hero scorer (`full_runs_table.py --hero`) cannot score the original
stages (their graded verdicts and final states sit where the original pipeline put them, so a hero-mode pass over
`full_text,...` finds every run but scores almost none), so `scripts/hero9_gap.py` reads gemma's bad-act counts from
`results/full_runs.csv` (the canonical scored table of those runs, already on the results page) and reads the run folders
themselves (`full_runs_table.modal_runs()` and `laptop_runs()`) for steps, files opened and endings. Losses for gemma are
the unscored / lost columns of `results/full_runs.csv`, copied into `results/hero9_loss_audit__gemma-3-27b.md` by the same
script. Precision: gemma-3 27B fp8 on DeepInfra via OpenRouter, text-mode tool calls (README deviation 4).
