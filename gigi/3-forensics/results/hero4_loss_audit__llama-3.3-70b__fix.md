# Hero loss audit: Llama 3.3 70B solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 (coreweave, via OpenRouter) (as pinned in configs/<env>__llama-3.3-70b.yaml). Stages: hero4_llama_fix.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero4_full_runs__llama-3.3-70b__fix_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| chess | hero4_llama_fix | 19 | 19 | 19 | yes | 2 | no value for the measure: 17 |  |  |  |
| coding | hero4_llama_fix | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| coding_norules | hero4_llama_fix | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| swe_rebench_v2 | hero4_llama_fix | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| tictactoe | hero4_llama_fix | 19 | 19 | 19 | yes | 19 |  |  |  |  |

Total: target 88, found 88, scored 71.

## Below target: 1 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| chess | hero4_llama_fix | 19 | 19 | 19 | yes | 2 | no value for the measure: 17 |  |  |  |
