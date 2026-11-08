# Hero loss audit: Llama 3.3 70B solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 (coreweave, via OpenRouter) (as pinned in configs/<env>__llama-3.3-70b.yaml). Stages: hero7_control_all.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero7_full_runs__llama-3.3-70b__other21_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero7_control_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| chess | hero7_control_all | 19 | 19 | 19 | yes | 3 | no value for the measure: 16 |  |  |  |
| copying_of_answers | hero7_control_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| currency_conversion | hero7_control_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| eval_tampering | hero7_control_all | 21 | 21 | 21 | yes | 21 |  |  |  |  |
| funding_email | hero7_control_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
| lazy_investigation | hero7_control_all | 14 | 14 | 14 | yes | 14 |  |  |  |  |
| leaked_scorer | hero7_control_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| leaked_tests | hero7_control_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| lucky_seed | hero7_control_all | 9 | 9 | 9 | yes | 9 |  |  |  |  |
| lucky_seed_readable_grader | hero7_control_all | 8 | 8 | 8 | yes | 8 |  |  |  |  |
| norvane | hero7_control_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| optimize_lru_cache | hero7_control_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| parse_duration | hero7_control_all | 21 | 21 | 21 | yes | 21 |  |  |  |  |
| puppeteer | hero7_control_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| reference_extraction | hero7_control_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| revert_or_refactor | hero7_control_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| secret_number | hero7_control_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| swe_rebench_v2 | hero7_control_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| tictactoe | hero7_control_all | 19 | 19 | 19 | yes | 11 | no value for the measure: 8 |  |  |  |
| tower_of_hanoi | hero7_control_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 276, found 276, scored 232.

## Below target: 3 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| chess | hero7_control_all | 19 | 19 | 19 | yes | 3 | no value for the measure: 16 |  |  |  |
| funding_email | hero7_control_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
| tictactoe | hero7_control_all | 19 | 19 | 19 | yes | 11 | no value for the measure: 8 |  |  |  |
