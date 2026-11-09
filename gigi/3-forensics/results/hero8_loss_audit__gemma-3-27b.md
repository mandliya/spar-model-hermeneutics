# Hero loss audit: gemma-3-27b solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model gemma-3-27b, fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__gemma-3-27b.yaml). Stages: hero8_gemma, hero8_gemma_key.
One row per environment (8: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero8_full_runs__gemma-3-27b_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero8_gemma_key | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| coding | hero8_gemma | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| coding_norules | hero8_gemma | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| dashboard_costly_failure | hero8_gemma | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| grade_your_own | hero8_gemma | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine | hero8_gemma | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine_graded | hero8_gemma | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| precommit_hook | hero8_gemma | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 95, found 95, scored 95.

## Below target: 0 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
