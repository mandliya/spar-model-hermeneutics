# Hero loss audit: Llama 3.3 70B solo runs (written by scripts/hero_solo_foldin.sh)

Model llama-3.3-70b, fp16 on CoreWeave via OpenRouter (as pinned in configs/<env>__llama-3.3-70b.yaml). One row per gap environment
and hero stage: the planned target (hero_models_prep.md section 4), runs launched (launch.txt), runs found locally,
whether the fleet was fetched after the apps stopped, runs scored, and every other run by cause. Rows below target
are listed again at the end. Source: results/hero_full_runs_per_run.csv.

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_rd | hero_solo_key2 | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| coding | hero_solo | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| coding_norules | hero_solo | 20 | 20 | 20 | yes | 20 |  |  |  |  |
| dashboard_costly_failure | hero_solo | 10 | 10 | 10 | yes | 10 |  |  |  |  |
| grade_your_own | hero_solo | 5 | 5 | 5 | yes | 0 | not graded: 5 |  |  |  |
| optimize_rendering_engine | hero_solo | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| optimize_rendering_engine_graded | hero_solo | 5 | 5 | 5 | yes | 5 |  |  |  |  |
| precommit_hook | hero_solo | 20 | 20 | 20 | yes | 20 |  |  |  |  |

Total: target 95, found 95, scored 90.

## Below target: 1 cells

| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |
|---|---|---|---|---|---|---|---|---|---|---|
| grade_your_own | hero_solo | 5 | 5 | 5 | yes | 0 | not graded: 5 |  |  |  |
