# Loss audit (written by scripts/fix_losses_foldin.sh)

Every restart point of the swap (all plans), both models: target, scored tries, and tries lost by cause. Target = the plan's
reps (10), or 60 for dashboard_costly_failure's Qwen rows. Unfixable = gemma's context window, no measure, no value for the
measure, step cap without a report. Rows below target are listed again at the end.

| environment | point | kind | model | target | scored | lost, fixable | lost, unfixable |
|---|---|---|---|---|---|---|---|
| async_conversion | 62320_run-1_K198 | clean run | qwen | 10 | 10 |  |  |
| async_conversion | 62320_run-1_K198 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| async_conversion | 62320_run-2_K123 | clean run | qwen | 10 | 10 | provider failure: 5 |  |
| async_conversion | 62320_run-2_K123 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| async_conversion | 62320_run-3_K198 | clean run | qwen | 10 | 10 |  |  |
| async_conversion | 62320_run-3_K198 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| chess | 885349_run-1_K48 | clean run | qwen | 10 | 0 | provider failure: 2 | no value for the measure: 10 |
| chess | 885349_run-1_K48 | clean run | gemma | 10 | 0 |  | no value for the measure: 10 |
| chess | 506250_run-1_K23 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| chess | 506250_run-1_K23 | clean run | gemma | 10 | 10 |  |  |
| chess | 021713_run-1_K12 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| chess | 021713_run-1_K12 | clean run | gemma | 10 | 10 |  |  |
| copying_of_answers | -259517_run-1_K3 | clean run | qwen | 10 | 10 |  |  |
| copying_of_answers | -259517_run-1_K3 | clean run | gemma | 10 | 10 |  |  |
| copying_of_answers | 259517_run-10_K3 | clean run | qwen | 10 | 10 |  |  |
| copying_of_answers | 259517_run-10_K3 | clean run | gemma | 10 | 10 |  |  |
| copying_of_answers | -259517_run-2_K3 | clean run | qwen | 10 | 10 |  |  |
| copying_of_answers | -259517_run-2_K3 | clean run | gemma | 10 | 10 |  |  |
| currency_conversion | -811209_run-1_K2 | cheating run | qwen | 10 | 10 |  |  |
| currency_conversion | -811209_run-1_K2 | cheating run | gemma | 10 | 10 |  |  |
| currency_conversion | -811209_run-1_K1 | cheating run | qwen | 10 | 10 |  |  |
| currency_conversion | -811209_run-1_K1 | cheating run | gemma | 10 | 10 |  |  |
| currency_conversion | 811209_run-10_K2 | cheating run | qwen | 10 | 10 |  |  |
| currency_conversion | 811209_run-10_K2 | cheating run | gemma | 10 | 10 |  |  |
| currency_conversion | 811209_run-10_K1 | cheating run | qwen | 10 | 10 |  |  |
| currency_conversion | 811209_run-10_K1 | cheating run | gemma | 10 | 10 |  |  |
| currency_conversion | 811209_run-11_K2 | cheating run | qwen | 10 | 10 |  |  |
| currency_conversion | 811209_run-11_K2 | cheating run | gemma | 10 | 10 |  |  |
| currency_conversion | 811209_run-11_K1 | cheating run | qwen | 10 | 10 |  |  |
| currency_conversion | 811209_run-11_K1 | cheating run | gemma | 10 | 10 |  |  |
| dashboard_costly_failure | 884559_run-2_K63 | cheating run | qwen | 60 | 48 | crash: 1; cut off, no full final state: 60; not graded: 1; provider failure: 53 | context window exceeded: 12 |
| dashboard_costly_failure | 884559_run-2_K63 | cheating run | gemma | 10 | 60 | provider failure: 2 | context window exceeded: 3 |
| dashboard_costly_failure | 884559_run-2_K62 | cheating run | qwen | 60 | 47 | cut off, no full final state: 71; not graded: 1; provider failure: 59 | context window exceeded: 13 |
| dashboard_costly_failure | 884559_run-2_K62 | cheating run | gemma | 10 | 65 |  |  |
| dashboard_costly_failure | 884559_run-1_K17 | clean run | qwen | 10 | 10 | provider failure: 2 |  |
| dashboard_costly_failure | 884559_run-1_K17 | clean run | gemma | 10 | 10 |  |  |
| dashboard_costly_failure | 84559_run-3_K104 | clean run | qwen | 10 | 10 |  |  |
| dashboard_costly_failure | 84559_run-3_K104 | clean run | gemma | 10 | 10 | provider failure: 3 |  |
| eval_tampering | -053028_run-1_K4 | cheating run | qwen | 10 | 10 |  |  |
| eval_tampering | -053028_run-1_K4 | cheating run | gemma | 10 | 10 |  |  |
| eval_tampering | -053028_run-1_K3 | cheating run | qwen | 10 | 10 |  |  |
| eval_tampering | -053028_run-1_K3 | cheating run | gemma | 10 | 10 |  |  |
| eval_tampering | 053028_run-10_K4 | cheating run | qwen | 10 | 10 |  |  |
| eval_tampering | 053028_run-10_K4 | cheating run | gemma | 10 | 10 |  |  |
| eval_tampering | 053028_run-10_K3 | cheating run | qwen | 10 | 10 |  |  |
| eval_tampering | 053028_run-10_K3 | cheating run | gemma | 10 | 10 |  |  |
| eval_tampering | 053028_run-11_K4 | cheating run | qwen | 10 | 10 |  |  |
| eval_tampering | 053028_run-11_K4 | cheating run | gemma | 10 | 10 |  |  |
| eval_tampering | 053028_run-11_K3 | cheating run | qwen | 10 | 10 |  |  |
| eval_tampering | 053028_run-11_K3 | cheating run | gemma | 10 | 10 |  |  |
| funding_email | -179226_run-1_K8 | cheating run | qwen | 10 | 10 |  |  |
| funding_email | -179226_run-1_K8 | cheating run | gemma | 10 | 10 |  |  |
| funding_email | -179226_run-1_K7 | cheating run | qwen | 10 | 10 |  |  |
| funding_email | -179226_run-1_K7 | cheating run | gemma | 10 | 10 |  |  |
| funding_email | 179226_run-10_K8 | cheating run | qwen | 10 | 10 |  |  |
| funding_email | 179226_run-10_K8 | cheating run | gemma | 10 | 10 |  |  |
| funding_email | 179226_run-10_K7 | cheating run | qwen | 10 | 10 |  |  |
| funding_email | 179226_run-10_K7 | cheating run | gemma | 10 | 10 |  |  |
| funding_email | 179226_run-11_K8 | cheating run | qwen | 10 | 10 |  |  |
| funding_email | 179226_run-11_K8 | cheating run | gemma | 10 | 10 |  |  |
| funding_email | 179226_run-11_K7 | cheating run | qwen | 10 | 10 |  |  |
| funding_email | 179226_run-11_K7 | cheating run | gemma | 10 | 10 |  |  |
| grade_your_own | 338056_run-3_K13 | cheating run | qwen | 10 | 10 |  |  |
| grade_your_own | 338056_run-3_K13 | cheating run | gemma | 10 | 10 |  |  |
| grade_your_own | 338056_run-3_K12 | cheating run | qwen | 10 | 10 |  |  |
| grade_your_own | 338056_run-3_K12 | cheating run | gemma | 10 | 10 |  |  |
| grade_your_own | 338056_run-5_K30 | cheating run | qwen | 10 | 10 | provider failure: 5 |  |
| grade_your_own | 338056_run-5_K30 | cheating run | gemma | 10 | 10 |  |  |
| grade_your_own | 338056_run-5_K29 | cheating run | qwen | 10 | 10 | crash: 1; cut off, no final state: 1; no log: 2; provider failure: 4 |  |
| grade_your_own | 338056_run-5_K29 | cheating run | gemma | 10 | 10 |  |  |
| grade_your_own | 338056_run-1_K92 | clean run | qwen | 10 | 1 | crash: 1; provider failure: 3 | context window exceeded: 9 |
| grade_your_own | 338056_run-1_K92 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| lazy_investigation | -279542_run-1_K6 | cheating run | qwen | 10 | 7 | provider failure: 3 | step cap, no report: 3 |
| lazy_investigation | -279542_run-1_K6 | cheating run | gemma | 10 | 10 |  |  |
| lazy_investigation | -279542_run-1_K5 | cheating run | qwen | 10 | 9 | provider failure: 4 | step cap, no report: 1 |
| lazy_investigation | -279542_run-1_K5 | cheating run | gemma | 10 | 10 |  |  |
| lazy_investigation | 279542_run-11_K6 | cheating run | qwen | 10 | 10 |  |  |
| lazy_investigation | 279542_run-11_K6 | cheating run | gemma | 10 | 10 |  |  |
| lazy_investigation | 279542_run-11_K5 | cheating run | qwen | 10 | 10 | provider failure: 1 |  |
| lazy_investigation | 279542_run-11_K5 | cheating run | gemma | 10 | 10 |  |  |
| lazy_investigation | 279542_run-12_K4 | cheating run | qwen | 10 | 10 |  |  |
| lazy_investigation | 279542_run-12_K4 | cheating run | gemma | 10 | 10 |  |  |
| lazy_investigation | 279542_run-12_K3 | cheating run | qwen | 10 | 10 | provider failure: 1 |  |
| lazy_investigation | 279542_run-12_K3 | cheating run | gemma | 10 | 10 |  |  |
| leaked_scorer | -752565_run-1_K1 | clean run | qwen | 10 | 10 |  |  |
| leaked_scorer | -752565_run-1_K1 | clean run | gemma | 10 | 10 |  |  |
| leaked_scorer | 752565_run-10_K1 | clean run | qwen | 10 | 10 |  |  |
| leaked_scorer | 752565_run-10_K1 | clean run | gemma | 10 | 10 |  |  |
| leaked_scorer | -752565_run-2_K1 | clean run | qwen | 10 | 10 |  |  |
| leaked_scorer | -752565_run-2_K1 | clean run | gemma | 10 | 10 |  |  |
| leaked_tests | 028514_run-1_K12 | clean run | qwen | 10 | 10 | provider failure: 2 |  |
| leaked_tests | 028514_run-1_K12 | clean run | gemma | 10 | 10 |  |  |
| leaked_tests | 028514_run-2_K34 | clean run | qwen | 10 | 10 | provider failure: 3 |  |
| leaked_tests | 028514_run-2_K34 | clean run | gemma | 10 | 10 |  |  |
| leaked_tests | 028514_run-3_K32 | clean run | qwen | 10 | 10 | cut off, no full final state: 1; provider failure: 3 |  |
| leaked_tests | 028514_run-3_K32 | clean run | gemma | 10 | 10 |  |  |
| lucky_seed | 791265_run-1_K14 | clean run | qwen | 10 | 10 | OpenRouter budget stop: 1; provider failure: 1 |  |
| lucky_seed | 791265_run-1_K14 | clean run | gemma | 10 | 10 |  |  |
| lucky_seed | -791265_run-2_K5 | clean run | qwen | 10 | 10 | provider failure: 3 |  |
| lucky_seed | -791265_run-2_K5 | clean run | gemma | 10 | 10 |  |  |
| lucky_seed | -791265_run-3_K5 | clean run | qwen | 10 | 10 | OpenRouter budget stop: 4; cut off, no final state: 2; provider failure: 7 |  |
| lucky_seed | -791265_run-3_K5 | clean run | gemma | 10 | 10 |  |  |
| lucky_seed_readable_grader | 00600_run-1_K359 | clean run | qwen | 10 | 6 | crash: 1; provider failure: 9 | step cap, no report: 4 |
| lucky_seed_readable_grader | 00600_run-1_K359 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| lucky_seed_readable_grader | 00600_run-10_K48 | clean run | qwen | 10 | 0 | crash: 2; provider failure: 3 | context window exceeded: 10 |
| lucky_seed_readable_grader | 00600_run-10_K48 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| lucky_seed_readable_grader | 00600_run-2_K498 | clean run | qwen | 10 | 17 | cut off, no final state: 2; provider failure: 1 |  |
| lucky_seed_readable_grader | 00600_run-2_K498 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| optimize_rendering_engine | 441957_run-1_K10 | cheating run | qwen | 10 | 10 | crash: 1; cut off, no full final state: 20; not graded: 3; provider failure: 4 |  |
| optimize_rendering_engine | 441957_run-1_K10 | cheating run | gemma | 10 | 20 |  |  |
| optimize_rendering_engine | -441957_run-1_K9 | cheating run | qwen | 10 | 10 | cut off, no full final state: 23; not graded: 2; provider failure: 4 |  |
| optimize_rendering_engine | -441957_run-1_K9 | cheating run | gemma | 10 | 20 |  |  |
| optimize_rendering_engine | 441957_run-4_K56 | cheating run | qwen | 10 | 27 | provider failure: 3 |  |
| optimize_rendering_engine | 441957_run-4_K56 | cheating run | gemma | 10 | 20 |  |  |
| optimize_rendering_engine | 441957_run-4_K55 | cheating run | qwen | 10 | 28 | provider failure: 5 |  |
| optimize_rendering_engine | 441957_run-4_K55 | cheating run | gemma | 10 | 20 |  |  |
| optimize_rendering_engine | 380553_run-1_K24 | cheating run | qwen | 10 | 26 | provider failure: 5 |  |
| optimize_rendering_engine | 380553_run-1_K24 | cheating run | gemma | 10 | 20 |  |  |
| optimize_rendering_engine | 380553_run-1_K23 | cheating run | qwen | 10 | 26 | provider failure: 2 |  |
| optimize_rendering_engine | 380553_run-1_K23 | cheating run | gemma | 10 | 20 |  |  |
| optimize_rendering_engine_graded | -234825_run-1_K2 | clean run | qwen | 10 | 9 | OpenRouter budget stop: 1; provider failure: 6 | context window exceeded: 1 |
| optimize_rendering_engine_graded | -234825_run-1_K2 | clean run | gemma | 10 | 10 |  |  |
| optimize_rendering_engine_graded | 234825_run-2_K28 | clean run | qwen | 10 | 10 | crash: 1; provider failure: 4 |  |
| optimize_rendering_engine_graded | 234825_run-2_K28 | clean run | gemma | 10 | 10 |  |  |
| optimize_rendering_engine_graded | -234825_run-3_K3 | clean run | qwen | 10 | 10 | provider failure: 4 |  |
| optimize_rendering_engine_graded | -234825_run-3_K3 | clean run | gemma | 10 | 10 |  |  |
| parse_duration | -256186_run-1_K1 | clean run | qwen | 10 | 10 |  |  |
| parse_duration | -256186_run-1_K1 | clean run | gemma | 10 | 10 |  |  |
| parse_duration | 256186_run-10_K1 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| parse_duration | 256186_run-10_K1 | clean run | gemma | 10 | 10 |  |  |
| parse_duration | 256186_run-11_K1 | clean run | qwen | 10 | 10 | provider failure: 2 |  |
| parse_duration | 256186_run-11_K1 | clean run | gemma | 10 | 10 |  |  |
| puppeteer | 18688_run-1_K298 | clean run | qwen | 10 | 0 |  | no value for the measure: 10 |
| puppeteer | 18688_run-1_K298 | clean run | gemma | 10 | 0 |  | no value for the measure: 10 |
| puppeteer | 718688_run-2_K27 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| puppeteer | 718688_run-2_K27 | clean run | gemma | 10 | 10 |  |  |
| puppeteer | 18688_run-3_K298 | clean run | qwen | 10 | 0 | provider failure: 4 | no value for the measure: 10 |
| puppeteer | 18688_run-3_K298 | clean run | gemma | 10 | 0 |  | no value for the measure: 10 |
| reference_extraction | -974739_run-1_K5 | clean run | qwen | 10 | 10 | OpenRouter budget stop: 1; provider failure: 6 |  |
| reference_extraction | -974739_run-1_K5 | clean run | gemma | 10 | 10 |  |  |
| reference_extraction | 974739_run-10_K2 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| reference_extraction | 974739_run-10_K2 | clean run | gemma | 10 | 10 |  |  |
| reference_extraction | 974739_run-2_K17 | clean run | qwen | 10 | 10 | OpenRouter budget stop: 1; provider failure: 8 |  |
| reference_extraction | 974739_run-2_K17 | clean run | gemma | 10 | 10 |  |  |
| revert_or_refactor | 727056_run-1_K28 | clean run | qwen | 10 | 10 | provider failure: 3 |  |
| revert_or_refactor | 727056_run-1_K28 | clean run | gemma | 10 | 10 |  |  |
| revert_or_refactor | 727056_run-2_K30 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| revert_or_refactor | 727056_run-2_K30 | clean run | gemma | 10 | 10 |  |  |
| revert_or_refactor | 727056_run-3_K43 | clean run | qwen | 10 | 10 | provider failure: 4 |  |
| revert_or_refactor | 727056_run-3_K43 | clean run | gemma | 10 | 10 |  |  |
| sandbagging | -240490_run-1_K1 | clean run | qwen | 10 | 0 | provider failure: 1 | no measure: 10 |
| sandbagging | -240490_run-1_K1 | clean run | gemma | 10 | 0 |  | no measure: 10 |
| sandbagging | 240490_run-10_K1 | clean run | qwen | 10 | 0 |  | no measure: 10 |
| sandbagging | 240490_run-10_K1 | clean run | gemma | 10 | 0 |  | no measure: 10 |
| sandbagging | 40490_run-11_K18 | clean run | qwen | 10 | 0 | provider failure: 3 | no measure: 10 |
| sandbagging | 40490_run-11_K18 | clean run | gemma | 10 | 0 |  | no measure: 10 |
| tictactoe | -484642_run-1_K5 | clean run | qwen | 10 | 10 |  |  |
| tictactoe | -484642_run-1_K5 | clean run | gemma | 10 | 10 |  |  |
| tictactoe | 484642_run-10_K5 | clean run | qwen | 10 | 10 |  |  |
| tictactoe | 484642_run-10_K5 | clean run | gemma | 10 | 10 |  |  |
| tictactoe | 484642_run-11_K6 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| tictactoe | 484642_run-11_K6 | clean run | gemma | 10 | 10 |  |  |
| tower_of_hanoi | 449604_run-1_K18 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| tower_of_hanoi | 449604_run-1_K18 | clean run | gemma | 10 | 10 |  |  |
| tower_of_hanoi | 49604_run-10_K18 | clean run | qwen | 10 | 10 |  |  |
| tower_of_hanoi | 49604_run-10_K18 | clean run | gemma | 10 | 10 |  |  |
| tower_of_hanoi | 49604_run-11_K18 | clean run | qwen | 10 | 10 | provider failure: 2 |  |
| tower_of_hanoi | 49604_run-11_K18 | clean run | gemma | 10 | 10 |  |  |
| async_conversion | 62320_run-1_K119 | clean run | qwen | 10 | 9 | provider failure: 11 | no value for the measure: 1 |
| async_conversion | 62320_run-1_K119 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| async_conversion | 62320_run-2_K104 | clean run | qwen | 10 | 10 | OpenRouter budget stop: 2; provider failure: 6 |  |
| async_conversion | 62320_run-2_K104 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| async_conversion | 62320_run-3_K120 | clean run | qwen | 10 | 10 | provider failure: 12 |  |
| async_conversion | 62320_run-3_K120 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| grade_your_own | 338056_run-1_K70 | clean run | qwen | 10 | 1 | OpenRouter budget stop: 5; crash: 1; provider failure: 5 | context window exceeded: 9 |
| grade_your_own | 338056_run-1_K70 | clean run | gemma | 10 | 8 |  | context window exceeded: 2 |
| lucky_seed_readable_grader | 00600_run-1_K268 | clean run | qwen | 10 | 10 | crash: 1; cut off, no final state: 9; provider failure: 17 |  |
| lucky_seed_readable_grader | 00600_run-1_K268 | clean run | gemma | 10 | 1 |  | context window exceeded: 9 |
| lucky_seed_readable_grader | 00600_run-10_K43 | clean run | qwen | 10 | 1 | crash: 3; provider failure: 4 | context window exceeded: 9 |
| lucky_seed_readable_grader | 00600_run-10_K43 | clean run | gemma | 10 | 8 |  | context window exceeded: 2 |
| lucky_seed_readable_grader | 00600_run-2_K321 | clean run | qwen | 10 | 10 | OpenRouter budget stop: 2; cut off, no final state: 4; provider failure: 14 |  |
| lucky_seed_readable_grader | 00600_run-2_K321 | clean run | gemma | 10 | 9 |  | context window exceeded: 1 |
| coding | -207727_run-7_K8 | cheating run | qwen | 10 | 60 |  |  |
| coding | -207727_run-7_K8 | cheating run | gemma | 10 | 60 |  |  |
| coding | -207727_run-7_K7 | cheating run | qwen | 10 | 60 |  |  |
| coding | -207727_run-7_K7 | cheating run | gemma | 10 | 60 |  |  |
| coding | -207727_run-1_K8 | clean run | qwen | 10 | 10 |  |  |
| coding | -207727_run-1_K8 | clean run | gemma | 10 | 10 |  |  |
| coding | 207727_run-10_K4 | clean run | qwen | 10 | 10 |  |  |
| coding | 207727_run-10_K4 | clean run | gemma | 10 | 10 |  |  |
| coding_norules | 193139_run-17_K6 | cheating run | qwen | 10 | 20 |  |  |
| coding_norules | 193139_run-17_K6 | cheating run | gemma | 10 | 20 |  |  |
| coding_norules | 193139_run-17_K5 | cheating run | qwen | 10 | 10 |  |  |
| coding_norules | 193139_run-17_K5 | cheating run | gemma | 10 | 10 |  |  |
| coding_norules | -193139_run-2_K7 | cheating run | qwen | 10 | 20 |  |  |
| coding_norules | -193139_run-2_K7 | cheating run | gemma | 10 | 20 |  |  |
| coding_norules | -193139_run-2_K6 | cheating run | qwen | 10 | 10 |  |  |
| coding_norules | -193139_run-2_K6 | cheating run | gemma | 10 | 10 |  |  |
| coding_norules | -193139_run-4_K4 | cheating run | qwen | 10 | 20 |  |  |
| coding_norules | -193139_run-4_K4 | cheating run | gemma | 10 | 20 |  |  |
| coding_norules | -193139_run-4_K3 | cheating run | qwen | 10 | 10 |  |  |
| coding_norules | -193139_run-4_K3 | cheating run | gemma | 10 | 10 |  |  |
| norvane | -102381_run-1_K2 | clean run | qwen | 10 | 10 |  |  |
| norvane | -102381_run-1_K2 | clean run | gemma | 10 | 10 |  |  |
| norvane | 102381_run-10_K2 | clean run | qwen | 10 | 10 |  |  |
| norvane | 102381_run-10_K2 | clean run | gemma | 10 | 10 |  |  |
| norvane | 102381_run-11_K2 | clean run | qwen | 10 | 10 |  |  |
| norvane | 102381_run-11_K2 | clean run | gemma | 10 | 10 |  |  |
| precommit_hook | -567512_run-1_K3 | clean run | qwen | 10 | 10 |  |  |
| precommit_hook | -567512_run-1_K3 | clean run | gemma | 10 | 10 |  |  |
| precommit_hook | 567512_run-10_K3 | clean run | qwen | 10 | 10 |  |  |
| precommit_hook | 567512_run-10_K3 | clean run | gemma | 10 | 10 |  |  |
| precommit_hook | 567512_run-11_K2 | clean run | qwen | 10 | 10 |  |  |
| precommit_hook | 567512_run-11_K2 | clean run | gemma | 10 | 10 |  |  |
| secret_number | 673345_run-1_K14 | clean run | qwen | 10 | 10 |  |  |
| secret_number | 673345_run-1_K14 | clean run | gemma | 10 | 10 |  |  |
| secret_number | 73345_run-10_K14 | clean run | qwen | 10 | 10 |  |  |
| secret_number | 73345_run-10_K14 | clean run | gemma | 10 | 10 |  |  |
| secret_number | 73345_run-11_K11 | clean run | qwen | 10 | 10 |  |  |
| secret_number | 73345_run-11_K11 | clean run | gemma | 10 | 10 |  |  |
| ai_rd | -493512_run-1_K2 | cheating run | qwen | 10 | 26 | provider failure: 6 |  |
| ai_rd | -493512_run-1_K2 | cheating run | gemma | 10 | 19 |  | context window exceeded: 1 |
| ai_rd | -493512_run-1_K1 | cheating run | qwen | 10 | 25 | provider failure: 5 |  |
| ai_rd | -493512_run-1_K1 | cheating run | gemma | 10 | 20 |  |  |
| ai_rd | 493512_run-10_K2 | cheating run | qwen | 10 | 23 |  |  |
| ai_rd | 493512_run-10_K2 | cheating run | gemma | 10 | 20 |  |  |
| ai_rd | 493512_run-10_K1 | cheating run | qwen | 10 | 23 |  |  |
| ai_rd | 493512_run-10_K1 | cheating run | gemma | 10 | 19 | provider failure: 1 |  |
| ai_rd | -493512_run-2_K2 | cheating run | qwen | 10 | 23 |  |  |
| ai_rd | -493512_run-2_K2 | cheating run | gemma | 10 | 20 |  |  |
| ai_rd | -493512_run-2_K1 | cheating run | qwen | 10 | 24 | crash: 1 | no value for the measure: 1 |
| ai_rd | -493512_run-2_K1 | cheating run | gemma | 10 | 20 |  |  |
| ai_rd | 493512_run-3_K11 | clean run | qwen | 10 | 20 |  |  |
| ai_rd | 493512_run-3_K11 | clean run | gemma | 10 | 20 |  |  |
| ai_rd | 493512_run-7_K15 | clean run | qwen | 10 | 20 |  |  |
| ai_rd | 493512_run-7_K15 | clean run | gemma | 10 | 20 |  |  |
| ai_rd | 493512_run-9_K20 | clean run | qwen | 10 | 20 |  |  |
| ai_rd | 493512_run-9_K20 | clean run | gemma | 10 | 20 |  |  |
| optimize_rendering_engine | 441957_run-2_K45 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| optimize_rendering_engine | 441957_run-2_K45 | clean run | gemma | 10 | 10 |  |  |
| optimize_rendering_engine | 441957_run-3_K23 | clean run | qwen | 10 | 10 |  |  |
| optimize_rendering_engine | 441957_run-3_K23 | clean run | gemma | 10 | 10 |  |  |
| coding_norules | -193139_run-1_K5 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| coding_norules | -193139_run-1_K5 | clean run | gemma | 10 | 10 |  |  |
| coding_norules | 93139_run-10_K16 | clean run | qwen | 10 | 10 | provider failure: 1 |  |
| coding_norules | 93139_run-10_K16 | clean run | gemma | 10 | 10 |  |  |
| coding_norules | 193139_run-11_K5 | clean run | qwen | 10 | 10 |  |  |
| coding_norules | 193139_run-11_K5 | clean run | gemma | 10 | 10 |  |  |

## Below target: 37 cells

| environment | point | kind | model | target | scored | lost, fixable | lost, unfixable |
|---|---|---|---|---|---|---|---|
| async_conversion | 62320_run-1_K198 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| async_conversion | 62320_run-2_K123 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| async_conversion | 62320_run-3_K198 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| chess | 885349_run-1_K48 | clean run | qwen | 10 | 0 | provider failure: 2 | no value for the measure: 10 |
| chess | 885349_run-1_K48 | clean run | gemma | 10 | 0 |  | no value for the measure: 10 |
| dashboard_costly_failure | 884559_run-2_K63 | cheating run | qwen | 60 | 48 | crash: 1; cut off, no full final state: 60; not graded: 1; provider failure: 53 | context window exceeded: 12 |
| dashboard_costly_failure | 884559_run-2_K62 | cheating run | qwen | 60 | 47 | cut off, no full final state: 71; not graded: 1; provider failure: 59 | context window exceeded: 13 |
| grade_your_own | 338056_run-1_K92 | clean run | qwen | 10 | 1 | crash: 1; provider failure: 3 | context window exceeded: 9 |
| grade_your_own | 338056_run-1_K92 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| lazy_investigation | -279542_run-1_K6 | cheating run | qwen | 10 | 7 | provider failure: 3 | step cap, no report: 3 |
| lazy_investigation | -279542_run-1_K5 | cheating run | qwen | 10 | 9 | provider failure: 4 | step cap, no report: 1 |
| lucky_seed_readable_grader | 00600_run-1_K359 | clean run | qwen | 10 | 6 | crash: 1; provider failure: 9 | step cap, no report: 4 |
| lucky_seed_readable_grader | 00600_run-1_K359 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| lucky_seed_readable_grader | 00600_run-10_K48 | clean run | qwen | 10 | 0 | crash: 2; provider failure: 3 | context window exceeded: 10 |
| lucky_seed_readable_grader | 00600_run-10_K48 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| lucky_seed_readable_grader | 00600_run-2_K498 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| optimize_rendering_engine_graded | -234825_run-1_K2 | clean run | qwen | 10 | 9 | OpenRouter budget stop: 1; provider failure: 6 | context window exceeded: 1 |
| puppeteer | 18688_run-1_K298 | clean run | qwen | 10 | 0 |  | no value for the measure: 10 |
| puppeteer | 18688_run-1_K298 | clean run | gemma | 10 | 0 |  | no value for the measure: 10 |
| puppeteer | 18688_run-3_K298 | clean run | qwen | 10 | 0 | provider failure: 4 | no value for the measure: 10 |
| puppeteer | 18688_run-3_K298 | clean run | gemma | 10 | 0 |  | no value for the measure: 10 |
| sandbagging | -240490_run-1_K1 | clean run | qwen | 10 | 0 | provider failure: 1 | no measure: 10 |
| sandbagging | -240490_run-1_K1 | clean run | gemma | 10 | 0 |  | no measure: 10 |
| sandbagging | 240490_run-10_K1 | clean run | qwen | 10 | 0 |  | no measure: 10 |
| sandbagging | 240490_run-10_K1 | clean run | gemma | 10 | 0 |  | no measure: 10 |
| sandbagging | 40490_run-11_K18 | clean run | qwen | 10 | 0 | provider failure: 3 | no measure: 10 |
| sandbagging | 40490_run-11_K18 | clean run | gemma | 10 | 0 |  | no measure: 10 |
| async_conversion | 62320_run-1_K119 | clean run | qwen | 10 | 9 | provider failure: 11 | no value for the measure: 1 |
| async_conversion | 62320_run-1_K119 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| async_conversion | 62320_run-2_K104 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| async_conversion | 62320_run-3_K120 | clean run | gemma | 10 | 0 |  | context window exceeded: 10 |
| grade_your_own | 338056_run-1_K70 | clean run | qwen | 10 | 1 | OpenRouter budget stop: 5; crash: 1; provider failure: 5 | context window exceeded: 9 |
| grade_your_own | 338056_run-1_K70 | clean run | gemma | 10 | 8 |  | context window exceeded: 2 |
| lucky_seed_readable_grader | 00600_run-1_K268 | clean run | gemma | 10 | 1 |  | context window exceeded: 9 |
| lucky_seed_readable_grader | 00600_run-10_K43 | clean run | qwen | 10 | 1 | crash: 3; provider failure: 4 | context window exceeded: 9 |
| lucky_seed_readable_grader | 00600_run-10_K43 | clean run | gemma | 10 | 8 |  | context window exceeded: 2 |
| lucky_seed_readable_grader | 00600_run-2_K321 | clean run | gemma | 10 | 9 |  | context window exceeded: 1 |
