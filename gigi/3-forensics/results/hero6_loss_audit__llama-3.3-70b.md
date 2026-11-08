# Hero loss audit: Llama 3.3 70B solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 (coreweave, via OpenRouter) (as pinned in configs/<env>__llama-3.3-70b.yaml). Stages: hero6_example, hero6_example_key.
One row per environment (8: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero6_full_runs__llama-3.3-70b_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero6_example_key | 10 | 10 | 10 | yes | 8 | no value for the measure: 2 |  |  |  |
| coding | hero6_example | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| coding_norules | hero6_example | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| dashboard_costly_failure | hero6_example | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| grade_your_own | hero6_example | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine | hero6_example | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine_graded | hero6_example | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| precommit_hook | hero6_example | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 95, found 95, scored 93.

## Below target: 1 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero6_example_key | 10 | 10 | 10 | yes | 8 | no value for the measure: 2 |  |  |  |
