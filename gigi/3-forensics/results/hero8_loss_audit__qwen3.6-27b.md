# Hero loss audit: Qwen3.6 27B (thinking off) solo runs with the hard end rule (agent.end_task_required=true: only end_task ends the run) (written by scripts/hero_solo_foldin.sh)

Model qwen3.6-27b, fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__qwen3.6-27b.yaml). Stages: hero8_qwen36, hero8_qwen36_key.
One row per environment (8: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero8_full_runs__qwen3.6-27b_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero8_qwen36_key | 10 | 10 | 10 | yes | 9 |  |  | context window exceeded: 1 |  |
| coding | hero8_qwen36 | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| coding_norules | hero8_qwen36 | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| dashboard_costly_failure | hero8_qwen36 | 10 | 10 | 10 | yes | 9 |  |  | context window exceeded: 1 |  |
| grade_your_own | hero8_qwen36 | 5 | 5 | 5 | yes | 2 | cut off, no final state: 1 |  | context window exceeded: 2 |  |
| optimize_rendering_engine | hero8_qwen36 | 5 | 5 | 5 | yes | 2 |  | 1 | context window exceeded: 2 |  |
| optimize_rendering_engine_graded | hero8_qwen36 | 5 | 5 | 5 | yes | 3 |  |  | context window exceeded: 2 |  |
| precommit_hook | hero8_qwen36 | 20 | 20 | 20 | yes | 18 |  | 2 |  |  |

Total: target 95, found 95, scored 83.

## Below target: 6 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero8_qwen36_key | 10 | 10 | 10 | yes | 9 |  |  | context window exceeded: 1 |  |
| dashboard_costly_failure | hero8_qwen36 | 10 | 10 | 10 | yes | 9 |  |  | context window exceeded: 1 |  |
| grade_your_own | hero8_qwen36 | 5 | 5 | 5 | yes | 2 | cut off, no final state: 1 |  | context window exceeded: 2 |  |
| optimize_rendering_engine | hero8_qwen36 | 5 | 5 | 5 | yes | 2 |  | 1 | context window exceeded: 2 |  |
| optimize_rendering_engine_graded | hero8_qwen36 | 5 | 5 | 5 | yes | 3 |  |  | context window exceeded: 2 |  |
| precommit_hook | hero8_qwen36 | 20 | 20 | 20 | yes | 18 |  | 2 |  |  |
