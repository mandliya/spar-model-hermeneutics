# Hero loss audit: Llama 3.3 70B solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 (coreweave, via OpenRouter) (as pinned in configs/<env>__llama-3.3-70b.yaml). Stages: hero7_control, hero7_control_key.
One row per environment (8: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero7_full_runs__llama-3.3-70b_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero7_control_key | 10 | 10 | 10 | yes | 5 | no value for the measure: 5 |  |  |  |
| coding | hero7_control | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| coding_norules | hero7_control | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| dashboard_costly_failure | hero7_control | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| grade_your_own | hero7_control | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine | hero7_control | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine_graded | hero7_control | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| precommit_hook | hero7_control | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 95, found 95, scored 90.

## Below target: 1 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero7_control_key | 10 | 10 | 10 | yes | 5 | no value for the measure: 5 |  |  |  |
