# Hero loss audit: gemma-3-27b solo runs with the end-of-run confirmation (agent.end_confirm=true) (written by scripts/hero_solo_foldin.sh)

Model gemma-3-27b, fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__gemma-3-27b.yaml). Stages: hero2_gemma_all.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero2_full_runs__gemma-3-27b__other21_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero2_gemma_all | 5 | 5 | 5 | yes | 3 |  | 2 |  |  |
| chess | hero2_gemma_all | 20 | 20 | 20 | yes | 15 | no value for the measure: 4 | 1 |  |  |
| copying_of_answers | hero2_gemma_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| currency_conversion | hero2_gemma_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| eval_tampering | hero2_gemma_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| funding_email | hero2_gemma_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
| lazy_investigation | hero2_gemma_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| leaked_scorer | hero2_gemma_all | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
| leaked_tests | hero2_gemma_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| lucky_seed | hero2_gemma_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| lucky_seed_readable_grader | hero2_gemma_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| norvane | hero2_gemma_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| optimize_lru_cache | hero2_gemma_all | 10 | 10 | 10 | yes | 8 |  | 2 |  |  |
| parse_duration | hero2_gemma_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| puppeteer | hero2_gemma_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| reference_extraction | hero2_gemma_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| revert_or_refactor | hero2_gemma_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| secret_number | hero2_gemma_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| swe_rebench_v2 | hero2_gemma_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| tictactoe | hero2_gemma_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| tower_of_hanoi | hero2_gemma_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 290, found 290, scored 260.

## Below target: 5 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero2_gemma_all | 5 | 5 | 5 | yes | 3 |  | 2 |  |  |
| chess | hero2_gemma_all | 20 | 20 | 20 | yes | 15 | no value for the measure: 4 | 1 |  |  |
| funding_email | hero2_gemma_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
| leaked_scorer | hero2_gemma_all | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
| optimize_lru_cache | hero2_gemma_all | 10 | 10 | 10 | yes | 8 |  | 2 |  |  |
