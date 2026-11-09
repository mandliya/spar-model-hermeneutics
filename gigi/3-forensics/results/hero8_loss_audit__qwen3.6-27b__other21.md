# Hero loss audit: Qwen3.6 27B (thinking off) solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model qwen3.6-27b, fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__qwen3.6-27b.yaml). Stages: hero8_qwen36_all.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero8_full_runs__qwen3.6-27b__other21_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero8_qwen36_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| chess | hero8_qwen36_all | 19 | 19 | 19 | yes | 17 | no value for the measure: 2 |  |  |  |
| copying_of_answers | hero8_qwen36_all | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
| currency_conversion | hero8_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| eval_tampering | hero8_qwen36_all | 21 | 21 | 21 | yes | 20 |  | 1 |  |  |
| funding_email | hero8_qwen36_all | 20 | 20 | 20 | yes | 19 |  | 1 |  |  |
| lazy_investigation | hero8_qwen36_all | 14 | 14 | 14 | yes | 13 |  | 1 |  |  |
| leaked_scorer | hero8_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| leaked_tests | hero8_qwen36_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| lucky_seed | hero8_qwen36_all | 9 | 9 | 9 | yes | 9 |  |  |  |  |
| lucky_seed_readable_grader | hero8_qwen36_all | 8 | 8 | 8 | yes | 8 |  |  |  |  |
| norvane | hero8_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| optimize_lru_cache | hero8_qwen36_all | 5 | 5 | 5 | yes | 4 |  |  | crash: 1 |  |
| parse_duration | hero8_qwen36_all | 21 | 21 | 21 | yes | 21 |  |  |  |  |
| puppeteer | hero8_qwen36_all | 5 | 5 | 5 | yes | 1 | no value for the measure: 4 |  |  |  |
| reference_extraction | hero8_qwen36_all | 10 | 10 | 10 | no | 5 |  | 1 | context window exceeded: 4 |  |
| revert_or_refactor | hero8_qwen36_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| secret_number | hero8_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| swe_rebench_v2 | hero8_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| tictactoe | hero8_qwen36_all | 19 | 19 | 19 | yes | 19 |  |  |  |  |
| tower_of_hanoi | hero8_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 276, found 276, scored 260.

## Below target: 8 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| chess | hero8_qwen36_all | 19 | 19 | 19 | yes | 17 | no value for the measure: 2 |  |  |  |
| copying_of_answers | hero8_qwen36_all | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
| eval_tampering | hero8_qwen36_all | 21 | 21 | 21 | yes | 20 |  | 1 |  |  |
| funding_email | hero8_qwen36_all | 20 | 20 | 20 | yes | 19 |  | 1 |  |  |
| lazy_investigation | hero8_qwen36_all | 14 | 14 | 14 | yes | 13 |  | 1 |  |  |
| optimize_lru_cache | hero8_qwen36_all | 5 | 5 | 5 | yes | 4 |  |  | crash: 1 |  |
| puppeteer | hero8_qwen36_all | 5 | 5 | 5 | yes | 1 | no value for the measure: 4 |  |  |  |
| reference_extraction | hero8_qwen36_all | 10 | 10 | 10 | no | 5 |  | 1 | context window exceeded: 4 |  |
