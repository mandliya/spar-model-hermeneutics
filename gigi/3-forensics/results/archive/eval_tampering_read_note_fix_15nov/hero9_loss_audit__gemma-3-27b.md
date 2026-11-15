# Hero loss audit: gemma-3 27B with no end check (the original passes of 19-25 Oct 2026), written by scripts/hero9_gap.py

gemma was not relaunched for hero9: every cell of its original no-check passes is at the hero target (results/logs/hero9/gemma_topup_plan.md). This table copies the scored / unscored / lost columns of results/full_runs.csv for gemma-3-27b (fp8, DeepInfra via OpenRouter, text-mode tools).

| environment | scored | unscored (no value for the measure) | lost, provider | lost, crash or harness |
|---|---|---|---|---|
| ai_rd | 10 | 0 | 0 | 0 |
| async_conversion | 5 | 0 | 0 | 0 |
| chess | 19 | 0 | 1 | 0 |
| coding | 20 | 0 | 0 | 0 |
| coding_norules | 20 | 0 | 0 | 0 |
| copying_of_answers | 10 | 0 | 0 | 0 |
| currency_conversion | 20 | 0 | 0 | 0 |
| dashboard_costly_failure | 10 | 0 | 0 | 0 |
| eval_tampering | 20 | 0 | 0 | 0 |
| funding_email | 0 | 20 | 0 | 0 |
| grade_your_own | 5 | 0 | 0 | 0 |
| lazy_investigation | 20 | 0 | 0 | 0 |
| leaked_scorer | 10 | 0 | 0 | 0 |
| leaked_tests | 5 | 0 | 0 | 0 |
| lucky_seed | 10 | 0 | 0 | 0 |
| lucky_seed_readable_grader | 9 | 1 | 0 | 0 |
| norvane | 20 | 0 | 0 | 0 |
| optimize_lru_cache | 10 | 0 | 0 | 0 |
| optimize_rendering_engine | 5 | 0 | 0 | 0 |
| optimize_rendering_engine_graded | 5 | 0 | 0 | 0 |
| parse_duration | 20 | 0 | 0 | 0 |
| precommit_hook | 20 | 0 | 0 | 0 |
| puppeteer | 5 | 0 | 0 | 0 |
| reference_extraction | 10 | 0 | 0 | 0 |
| revert_or_refactor | 5 | 0 | 4 | 0 |
| secret_number | 20 | 0 | 0 | 0 |
| swe_rebench_v2 | 10 | 0 | 0 | 0 |
| tictactoe | 20 | 0 | 0 | 0 |
| tower_of_hanoi | 20 | 0 | 0 | 0 |

Total: scored 363, unscored 21, lost to the provider 5, lost to a crash or the harness 0.
