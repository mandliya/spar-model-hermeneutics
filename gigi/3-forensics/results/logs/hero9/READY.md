# hero9: the two heroes with no check (launched 10 Nov 2026, 12:16 local)

What the 10 Nov call decided: Llama is dropped; Qwen3.6 27B and gemma-3 27B are the heroes and run as the paper's
environments run them (no end check, no hard rule, no example).

**Running now (detached, every wait a loop; a sleeping Mac only pauses it)**
- `scripts/hero9_until_done.sh` (CEIL=4550, MODELS=qwen3.6-27b; log `results/logs/hero9/until_done.txt`): waits for the
  three Qwen3.6 smoke runs (stage hero9_smoke_qwen36: coding, tictactoe, eval_tampering), fetches and checks them (a run
  passes when the harness printed FINAL RESULTS and the run is "finished"), then launches the full fleet
  (`scripts/hero9_launch.sh full qwen3.6-27b`: hero9_qwen36 7 main tasks, hero9_qwen36_key ai_rd:10, hero9_qwen36_all the
  other 20 + swe_rebench_v2:10; 371 runs), then `scripts/hero9_foldin.sh` (fetch, redact, Opus grading of the judged
  tasks within GRADE_CAP=60, score, loss audit; log `results/logs/hero9/foldin.txt`, per-stage progress in
  `results/logs/hero9/qwen3.6-27b{,__other21}/progress.txt`), then writes DONE.
- `scripts/spend_watchdog.sh hero9` with CAP_USD=1100 from a key spend of $3,480.18 (`results/logs/hero9/watchdog.txt`).
  The hero8 watchdog was stopped first.
- gemma: nothing launched. Its original no-check passes are at the hero target in every cell
  (`results/logs/hero9/gemma_topup_plan.md`).

**ETA.** Smoke: minutes to ~1 h. Full fleet: Qwen3.6 ran 10 to 300 steps per task on 9 Nov, so expect 3 to 5 hours of
runs, then 1 to 2 hours of fetch and grading. Results: `results/hero9_full_runs__qwen3.6-27b{,__other21}{,_per_run}.csv`,
`results/hero9_loss_audit__qwen3.6-27b{,__other21}.md`.

**After DONE.** From gigi/3-forensics/agent-interp-envs:
`env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hero9_gap.py` writes charts/hero9_gap_cheating.png,
charts/hero9_gap_work.png, charts/hero9_gap_endings.png, results/hero9_gap.md (table, totals, clear gaps, losses, cost)
and results/hero9_loss_audit__gemma-3-27b.md. It already runs with the gemma side alone. Then record the key spend after
in `results/logs/hero9/cost.txt` and commit (nothing of hero9 is committed yet).

**Checks.** `tail results/logs/hero9/until_done.txt`; `modal app list` with MODAL_PROFILE=spar; disk
`df -h /System/Volumes/Data` (the fold-in stops before fetching under 5 GB free; the Mac hit 0 bytes free at 01:00).
If a fleet log ends in ConnectionError without "App completed", rerun `hero9_launch.sh full qwen3.6-27b` (idempotent).
If a fetch fails twice with "Connection lost" (reference_extraction, 300 MB to 1.8 GB per run), fetch run by run.
