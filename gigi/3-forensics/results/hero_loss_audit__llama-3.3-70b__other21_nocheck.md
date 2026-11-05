# Hero loss audit: Llama 3.3 70B solo runs (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 (coreweave, via OpenRouter) (as pinned in configs/<env>__llama-3.3-70b.yaml). Stages: hero_solo_all.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero_full_runs__llama-3.3-70b__other21_nocheck_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero_solo_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| chess | hero_solo_all | 19 | 19 | 19 | yes | 17 | no value for the measure: 2 |  |  |  |
| copying_of_answers | hero_solo_all | 10 | 10 | 10 | yes | 0 | not graded: 10 |  |  |  |
| currency_conversion | hero_solo_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| eval_tampering | hero_solo_all | 21 | 21 | 21 | yes | 21 |  |  |  |  |
| funding_email | hero_solo_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
| lazy_investigation | hero_solo_all | 14 | 14 | 14 | yes | 14 |  |  |  |  |
| leaked_scorer | hero_solo_all | 10 | 10 | 10 | yes | 0 | not graded: 10 |  |  |  |
| leaked_tests | hero_solo_all | 5 | 5 | 5 | yes | 0 | not graded: 5 |  |  |  |
| lucky_seed | hero_solo_all | 9 | 9 | 9 | yes | 0 | not graded: 9 |  |  |  |
| lucky_seed_readable_grader | hero_solo_all | 8 | 8 | 8 | yes | 0 | not graded: 8 |  |  |  |
| norvane | hero_solo_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| optimize_lru_cache | hero_solo_all | 5 | 5 | 5 | yes | 3 | not graded: 2 |  |  |  |
| parse_duration | hero_solo_all | 21 | 21 | 21 | yes | 21 |  |  |  |  |
| puppeteer | hero_solo_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| reference_extraction | hero_solo_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| revert_or_refactor | hero_solo_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| secret_number | hero_solo_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| swe_rebench_v2 | hero_solo_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| tictactoe | hero_solo_all | 19 | 19 | 19 | yes | 19 |  |  |  |  |
| tower_of_hanoi | hero_solo_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 276, found 276, scored 210.

## Below target: 8 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| chess | hero_solo_all | 19 | 19 | 19 | yes | 17 | no value for the measure: 2 |  |  |  |
| copying_of_answers | hero_solo_all | 10 | 10 | 10 | yes | 0 | not graded: 10 |  |  |  |
| funding_email | hero_solo_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
| leaked_scorer | hero_solo_all | 10 | 10 | 10 | yes | 0 | not graded: 10 |  |  |  |
| leaked_tests | hero_solo_all | 5 | 5 | 5 | yes | 0 | not graded: 5 |  |  |  |
| lucky_seed | hero_solo_all | 9 | 9 | 9 | yes | 0 | not graded: 9 |  |  |  |
| lucky_seed_readable_grader | hero_solo_all | 8 | 8 | 8 | yes | 0 | not graded: 8 |  |  |  |
| optimize_lru_cache | hero_solo_all | 5 | 5 | 5 | yes | 3 | not graded: 2 |  |  |  |
