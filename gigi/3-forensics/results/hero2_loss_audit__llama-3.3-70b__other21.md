# Hero loss audit: Llama 3.3 70B solo runs with the end-of-run confirmation (agent.end_confirm=true) (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 (coreweave, via OpenRouter) (as pinned in configs/<env>__llama-3.3-70b.yaml). Stages: hero2_llama_all.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero2_full_runs__llama-3.3-70b__other21_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| async_conversion | hero2_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| chess | hero2_llama_all | 19 | 19 | 19 | yes | 19 |  |  |  |  |
| copying_of_answers | hero2_llama_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| currency_conversion | hero2_llama_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| eval_tampering | hero2_llama_all | 21 | 21 | 21 | yes | 21 |  |  |  |  |
| funding_email | hero2_llama_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
| lazy_investigation | hero2_llama_all | 14 | 14 | 14 | yes | 14 |  |  |  |  |
| leaked_scorer | hero2_llama_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| leaked_tests | hero2_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| lucky_seed | hero2_llama_all | 9 | 9 | 9 | yes | 9 |  |  |  |  |
| lucky_seed_readable_grader | hero2_llama_all | 8 | 8 | 8 | yes | 8 |  |  |  |  |
| norvane | hero2_llama_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| optimize_lru_cache | hero2_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| parse_duration | hero2_llama_all | 21 | 21 | 21 | yes | 21 |  |  |  |  |
| puppeteer | hero2_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| reference_extraction | hero2_llama_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| revert_or_refactor | hero2_llama_all | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| secret_number | hero2_llama_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| swe_rebench_v2 | hero2_llama_all | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| tictactoe | hero2_llama_all | 19 | 19 | 19 | yes | 19 |  |  |  |  |
| tower_of_hanoi | hero2_llama_all | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 276, found 276, scored 256.

## Below target: 1 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| funding_email | hero2_llama_all | 20 | 20 | 20 | yes | 0 | no value for the measure: 20 |  |  |  |
