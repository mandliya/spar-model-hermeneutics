# Hero loss audit: gemma-3-27b solo runs with the end-of-run confirmation (agent.end_confirm=true) (written by scripts/hero_solo_foldin.sh)

Model gemma-3-27b, fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__gemma-3-27b.yaml). Stages: hero2_gemma, hero2_gemma_key.
One row per environment (8: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero2_full_runs__gemma-3-27b_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero2_gemma_key | 10 | 10 | 10 | yes | 9 | no value for the measure: 1 |  |  |  |
| coding | hero2_gemma | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| coding_norules | hero2_gemma | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| dashboard_costly_failure | hero2_gemma | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| grade_your_own | hero2_gemma | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine | hero2_gemma | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine_graded | hero2_gemma | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| precommit_hook | hero2_gemma | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 95, found 95, scored 94.

## Below target: 1 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero2_gemma_key | 10 | 10 | 10 | yes | 9 | no value for the measure: 1 |  |  |  |
