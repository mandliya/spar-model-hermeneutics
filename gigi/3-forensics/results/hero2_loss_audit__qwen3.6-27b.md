# Hero loss audit: Qwen3.6 27B (thinking off) solo runs with the end-of-run confirmation (agent.end_confirm=true) (written by scripts/hero_solo_foldin.sh)

Model qwen3.6-27b, fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__qwen3.6-27b.yaml). Stages: hero2_qwen36, hero2_qwen36_key, hero2_qwen36_topup.
One row per environment (8: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero2_full_runs__qwen3.6-27b_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero2_qwen36_key | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| coding | hero2_qwen36 | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| coding_norules | hero2_qwen36 | 20 | 20 | 20 | yes | 17 |  | 3 |  |  |
| coding_norules | hero2_qwen36_topup | 3 | 3 | 3 | yes | 3 |  |  |  |  |
| dashboard_costly_failure | hero2_qwen36 | 10 | 10 | 10 | yes | 5 |  | 2 | context window exceeded: 3 |  |
| dashboard_costly_failure | hero2_qwen36_topup | 2 | 2 | 2 | yes | 0 |  |  | context window exceeded: 2 |  |
| grade_your_own | hero2_qwen36 | 5 | 5 | 5 | yes | 2 | cut off, no final state: 2 |  | context window exceeded: 1 |  |
| optimize_rendering_engine | hero2_qwen36 | 5 | 5 | 5 | yes | 1 |  |  | context window exceeded: 4 |  |
| optimize_rendering_engine_graded | hero2_qwen36 | 5 | 5 | 5 | yes | 2 |  |  | context window exceeded: 3 |  |
| precommit_hook | hero2_qwen36 | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 100, found 100, scored 80.

## Below target: 6 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| coding_norules | hero2_qwen36 | 20 | 20 | 20 | yes | 17 |  | 3 |  |  |
| dashboard_costly_failure | hero2_qwen36 | 10 | 10 | 10 | yes | 5 |  | 2 | context window exceeded: 3 |  |
| dashboard_costly_failure | hero2_qwen36_topup | 2 | 2 | 2 | yes | 0 |  |  | context window exceeded: 2 |  |
| grade_your_own | hero2_qwen36 | 5 | 5 | 5 | yes | 2 | cut off, no final state: 2 |  | context window exceeded: 1 |  |
| optimize_rendering_engine | hero2_qwen36 | 5 | 5 | 5 | yes | 1 |  |  | context window exceeded: 4 |  |
| optimize_rendering_engine_graded | hero2_qwen36 | 5 | 5 | 5 | yes | 2 |  |  | context window exceeded: 3 |  |
