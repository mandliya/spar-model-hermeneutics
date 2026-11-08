# Hero loss audit: Qwen3.6 27B (thinking off) solo runs with the end-of-run confirmation (agent.end_confirm=true) (written by scripts/hero_solo_foldin.sh)

Model qwen3.6-27b, fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__qwen3.6-27b.yaml). Stages: hero2_qwen36_all.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero2_full_runs__qwen3.6-27b__other21_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero2_qwen36_all | 5 | 5 | 5 | yes | 4 |  | 1 |  |  |
| chess | hero2_qwen36_all | 20 | 20 | 20 | yes | 19 |  | 1 |  |  |
| copying_of_answers | hero2_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| currency_conversion | hero2_qwen36_all | 20 | 20 | 20 | yes | 17 | no value for the measure: 2 | 1 |  |  |
| eval_tampering | hero2_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| funding_email | hero2_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| lazy_investigation | hero2_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| leaked_scorer | hero2_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| leaked_tests | hero2_qwen36_all | 5 | 5 | 5 | yes | 4 |  | 1 |  |  |
| lucky_seed | hero2_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| lucky_seed_readable_grader | hero2_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| norvane | hero2_qwen36_all | 20 | 20 | 20 | yes | 19 |  | 1 |  |  |
| optimize_lru_cache | hero2_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| parse_duration | hero2_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| puppeteer | hero2_qwen36_all | 5 | 5 | 5 | yes | 3 | no value for the measure: 2 |  |  |  |
| reference_extraction | hero2_qwen36_all | 10 | 10 | 10 | no | 7 | cut off, no full final state: 1 | 1 | context window exceeded: 1 |  |
| revert_or_refactor | hero2_qwen36_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| secret_number | hero2_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| swe_rebench_v2 | hero2_qwen36_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| tictactoe | hero2_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| tower_of_hanoi | hero2_qwen36_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 290, found 290, scored 278.

## Below target: 7 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero2_qwen36_all | 5 | 5 | 5 | yes | 4 |  | 1 |  |  |
| chess | hero2_qwen36_all | 20 | 20 | 20 | yes | 19 |  | 1 |  |  |
| currency_conversion | hero2_qwen36_all | 20 | 20 | 20 | yes | 17 | no value for the measure: 2 | 1 |  |  |
| leaked_tests | hero2_qwen36_all | 5 | 5 | 5 | yes | 4 |  | 1 |  |  |
| norvane | hero2_qwen36_all | 20 | 20 | 20 | yes | 19 |  | 1 |  |  |
| puppeteer | hero2_qwen36_all | 5 | 5 | 5 | yes | 3 | no value for the measure: 2 |  |  |  |
| reference_extraction | hero2_qwen36_all | 10 | 10 | 10 | no | 7 | cut off, no full final state: 1 | 1 | context window exceeded: 1 |  |
