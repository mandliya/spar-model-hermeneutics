# Hero loss audit: Qwen3.6 27B (thinking off) solo runs (written by scripts/hero_solo_foldin.sh)

Model qwen3.6-27b, fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__qwen3.6-27b.yaml). Stages: hero9_qwen36_all.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero9_full_runs__qwen3.6-27b__other21_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero9_qwen36_all | 5 | 5 | 5 | yes | 4 |  | 1 |  |  |
| chess | hero9_qwen36_all | 19 | 19 | 19 | yes | 17 |  | 2 |  |  |
| copying_of_answers | hero9_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| currency_conversion | hero9_qwen36_all | 20 | 20 | 20 | yes | 11 | no value for the measure: 5 | 4 |  |  |
| eval_tampering | hero9_qwen36_all | 21 | 21 | 21 | yes | 20 |  | 1 |  |  |
| funding_email | hero9_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| lazy_investigation | hero9_qwen36_all | 14 | 14 | 14 | yes | 14 |  |  |  |  |
| leaked_scorer | hero9_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| leaked_tests | hero9_qwen36_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| lucky_seed | hero9_qwen36_all | 9 | 9 | 9 | yes | 9 |  |  |  |  |
| lucky_seed_readable_grader | hero9_qwen36_all | 8 | 8 | 8 | yes | 7 |  | 1 |  |  |
| norvane | hero9_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| optimize_lru_cache | hero9_qwen36_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| parse_duration | hero9_qwen36_all | 21 | 21 | 21 | yes | 19 |  | 2 |  |  |
| puppeteer | hero9_qwen36_all | 5 | 5 | 5 | yes | 2 | no value for the measure: 3 |  |  |  |
| reference_extraction | hero9_qwen36_all | 10 | 10 | 10 | no | 8 |  |  | context window exceeded: 2 |  |
| revert_or_refactor | hero9_qwen36_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| secret_number | hero9_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| swe_rebench_v2 | hero9_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| tictactoe | hero9_qwen36_all | 19 | 19 | 19 | yes | 19 |  |  |  |  |
| tower_of_hanoi | hero9_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 276, found 276, scored 255.

## Below target: 8 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero9_qwen36_all | 5 | 5 | 5 | yes | 4 |  | 1 |  |  |
| chess | hero9_qwen36_all | 19 | 19 | 19 | yes | 17 |  | 2 |  |  |
| currency_conversion | hero9_qwen36_all | 20 | 20 | 20 | yes | 11 | no value for the measure: 5 | 4 |  |  |
| eval_tampering | hero9_qwen36_all | 21 | 21 | 21 | yes | 20 |  | 1 |  |  |
| lucky_seed_readable_grader | hero9_qwen36_all | 8 | 8 | 8 | yes | 7 |  | 1 |  |  |
| parse_duration | hero9_qwen36_all | 21 | 21 | 21 | yes | 19 |  | 2 |  |  |
| puppeteer | hero9_qwen36_all | 5 | 5 | 5 | yes | 2 | no value for the measure: 3 |  |  |  |
| reference_extraction | hero9_qwen36_all | 10 | 10 | 10 | no | 8 |  |  | context window exceeded: 2 |  |
