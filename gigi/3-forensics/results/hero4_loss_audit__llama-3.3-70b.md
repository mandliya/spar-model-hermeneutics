# Hero loss audit: Llama 3.3 70B solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 (coreweave, via OpenRouter) (as pinned in configs/<env>__llama-3.3-70b.yaml). Stages: hero4_llama, hero4_llama_key.
One row per environment (8: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero4_full_runs__llama-3.3-70b_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero4_llama_key | 10 | 10 | 10 | yes | 7 | no value for the measure: 3 |  |  |  |
| coding | hero4_llama | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| coding_norules | hero4_llama | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| dashboard_costly_failure | hero4_llama | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| grade_your_own | hero4_llama | 5 | 5 | 5 | yes | 4 | not graded: 1 |  |  |  |
| optimize_rendering_engine | hero4_llama | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine_graded | hero4_llama | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| precommit_hook | hero4_llama | 20 | 20 | 20 | yes | 19 | no value for the measure: 1 |  |  |  |

Total: target 95, found 95, scored 90.

## Below target: 3 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero4_llama_key | 10 | 10 | 10 | yes | 7 | no value for the measure: 3 |  |  |  |
| grade_your_own | hero4_llama | 5 | 5 | 5 | yes | 4 | not graded: 1 |  |  |  |
| precommit_hook | hero4_llama | 20 | 20 | 20 | yes | 19 | no value for the measure: 1 |  |  |  |
