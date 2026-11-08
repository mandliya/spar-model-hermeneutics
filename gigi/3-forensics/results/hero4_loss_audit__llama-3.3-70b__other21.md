# Hero loss audit: Llama 3.3 70B solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 (coreweave, via OpenRouter) (as pinned in configs/<env>__llama-3.3-70b.yaml). Stages: hero4_llama_all.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero4_full_runs__llama-3.3-70b__other21_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero4_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| chess | hero4_llama_all | 19 | 19 | 19 | yes | 9 | no value for the measure: 10 |  |  |  |
| copying_of_answers | hero4_llama_all | 10 | 10 | 10 | yes | 9 | not graded: 1 |  |  |  |
| currency_conversion | hero4_llama_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| eval_tampering | hero4_llama_all | 21 | 21 | 21 | yes | 21 |  |  |  |  |
| funding_email | hero4_llama_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
| lazy_investigation | hero4_llama_all | 14 | 14 | 14 | yes | 13 | step cap, no report: 1 |  |  |  |
| leaked_scorer | hero4_llama_all | 10 | 10 | 10 | yes | 0 | not graded: 10 |  |  |  |
| leaked_tests | hero4_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| lucky_seed | hero4_llama_all | 9 | 9 | 9 | yes | 7 | cut off, no final state: 1; not graded: 1 |  |  |  |
| lucky_seed_readable_grader | hero4_llama_all | 8 | 8 | 8 | yes | 8 |  |  |  |  |
| norvane | hero4_llama_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| optimize_lru_cache | hero4_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| parse_duration | hero4_llama_all | 21 | 21 | 21 | yes | 21 |  |  |  |  |
| puppeteer | hero4_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| reference_extraction | hero4_llama_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| revert_or_refactor | hero4_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| secret_number | hero4_llama_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| swe_rebench_v2 | hero4_llama_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| tictactoe | hero4_llama_all | 19 | 19 | 19 | yes | 18 | no value for the measure: 1 |  |  |  |
| tower_of_hanoi | hero4_llama_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 276, found 276, scored 231.

## Below target: 7 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| chess | hero4_llama_all | 19 | 19 | 19 | yes | 9 | no value for the measure: 10 |  |  |  |
| copying_of_answers | hero4_llama_all | 10 | 10 | 10 | yes | 9 | not graded: 1 |  |  |  |
| funding_email | hero4_llama_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
| lazy_investigation | hero4_llama_all | 14 | 14 | 14 | yes | 13 | step cap, no report: 1 |  |  |  |
| leaked_scorer | hero4_llama_all | 10 | 10 | 10 | yes | 0 | not graded: 10 |  |  |  |
| lucky_seed | hero4_llama_all | 9 | 9 | 9 | yes | 7 | cut off, no final state: 1; not graded: 1 |  |  |  |
| tictactoe | hero4_llama_all | 19 | 19 | 19 | yes | 18 | no value for the measure: 1 |  |  |  |
