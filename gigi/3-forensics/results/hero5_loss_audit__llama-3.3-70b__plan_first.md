# Hero loss audit: Llama 3.3 70B solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 (coreweave, via OpenRouter) (as pinned in configs/<env>__llama-3.3-70b.yaml). Stages: hero5_plan_first.
One row per environment (8: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero5_full_runs__llama-3.3-70b__plan_first_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| coding | hero5_plan_first | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
| coding_norules | hero5_plan_first | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| dashboard_costly_failure | hero5_plan_first | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| grade_your_own | hero5_plan_first | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine | hero5_plan_first | 5 | 5 | 5 | yes | 5 |  |  |  |  |

Total: target 40, found 40, scored 39.

## Below target: 1 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| coding | hero5_plan_first | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
