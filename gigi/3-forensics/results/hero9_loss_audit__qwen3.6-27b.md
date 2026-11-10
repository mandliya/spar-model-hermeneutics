# Hero loss audit: Qwen3.6 27B (thinking off) solo runs (written by scripts/hero_solo_foldin.sh)

Model qwen3.6-27b, fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__qwen3.6-27b.yaml). Stages: hero9_qwen36, hero9_qwen36_key.
One row per environment (8: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero9_full_runs__qwen3.6-27b_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero9_qwen36_key | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
| coding | hero9_qwen36 | 20 | 20 | 20 | yes | 19 |  | 1 |  |  |
| coding_norules | hero9_qwen36 | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| dashboard_costly_failure | hero9_qwen36 | 10 | 10 | 10 | yes | 6 |  |  | context window exceeded: 3; crash: 1 |  |
| grade_your_own | hero9_qwen36 | 5 | 5 | 5 | yes | 3 | cut off, no final state: 1 |  | context window exceeded: 1 |  |
| optimize_rendering_engine | hero9_qwen36 | 5 | 5 | 5 | yes | 1 |  | 1 | context window exceeded: 3 |  |
| optimize_rendering_engine_graded | hero9_qwen36 | 5 | 5 | 5 | yes | 1 |  |  | context window exceeded: 4 |  |
| precommit_hook | hero9_qwen36 | 20 | 20 | 20 | yes | 18 |  | 2 |  |  |

Total: target 95, found 95, scored 77.

## Below target: 7 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero9_qwen36_key | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
| coding | hero9_qwen36 | 20 | 20 | 20 | yes | 19 |  | 1 |  |  |
| dashboard_costly_failure | hero9_qwen36 | 10 | 10 | 10 | yes | 6 |  |  | context window exceeded: 3; crash: 1 |  |
| grade_your_own | hero9_qwen36 | 5 | 5 | 5 | yes | 3 | cut off, no final state: 1 |  | context window exceeded: 1 |  |
| optimize_rendering_engine | hero9_qwen36 | 5 | 5 | 5 | yes | 1 |  | 1 | context window exceeded: 3 |  |
| optimize_rendering_engine_graded | hero9_qwen36 | 5 | 5 | 5 | yes | 1 |  |  | context window exceeded: 4 |  |
| precommit_hook | hero9_qwen36 | 20 | 20 | 20 | yes | 18 |  | 2 |  |  |
