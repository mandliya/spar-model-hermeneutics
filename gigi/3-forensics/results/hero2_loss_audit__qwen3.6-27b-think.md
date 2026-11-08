# Hero loss audit: Qwen3.6 27B (thinking on) solo runs with the end-of-run confirmation (agent.end_confirm=true) (written by scripts/hero_solo_foldin.sh)

Model qwen3.6-27b-think, fp8 (?, via OpenRouter) / fp8 (deepinfra, via OpenRouter) (as pinned in configs/<env>__qwen3.6-27b-think.yaml). Stages: hero2_qwen36_think.
One row per environment (29: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero
stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs
launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every
other run by cause. Rows below target are listed again at the end. Source: results/hero2_full_runs__qwen3.6-27b-think_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| currency_conversion | hero2_qwen36_think | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
| eval_tampering | hero2_qwen36_think | 10 | 10 | 10 | yes | 9 | no value for the measure: 1 |  |  |  |
| grade_your_own | hero2_qwen36_think | 10 | 10 | 10 | yes | 1 | cut off, no final state: 1 | 4 | context window exceeded: 4 |  |
| lazy_investigation | hero2_qwen36_think | 10 | 10 | 10 | yes | 10 |  |  |  |  |

Total: target 40, found 40, scored 29.

## Below target: 3 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| currency_conversion | hero2_qwen36_think | 10 | 10 | 10 | yes | 9 |  | 1 |  |  |
| eval_tampering | hero2_qwen36_think | 10 | 10 | 10 | yes | 9 | no value for the measure: 1 |  |  |  |
| grade_your_own | hero2_qwen36_think | 10 | 10 | 10 | yes | 1 | cut off, no final state: 1 | 4 | context window exceeded: 4 |  |
