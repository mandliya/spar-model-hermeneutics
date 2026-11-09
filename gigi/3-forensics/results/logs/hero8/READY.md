# hero8 readiness note (9 Nov 2026, written before any launch; nothing has been spent)

**What hero8 is.** gemma-3-27b (fp8 DeepInfra, text-mode tools) and qwen3.6-27b (fp8 DeepInfra, thinking off) on all 29
tasks under exactly Llama's hero6 setting: `agent.end_task_required=true, agent.end_task_reprompts=10,
agent.hack=worked_example`. Same configs, provider pins, step caps and run counts as hero6/hero7, so the three models
compare row by row. Per model: 85 runs in the 7 main tasks (stage `hero8_<tag>`) + 10 ai_rd with the capped agent key
(`hero8_<tag>_key`) + 266 in the other 20 + 10 swe_rebench_v2 (`hero8_<tag>_all`) = **371 runs per model, 742 in all**.
tag = `gemma` | `qwen36`. gemma launches 5 runs at a time per environment (DeepInfra 429s), Qwen3.6 unlimited.

**Ready.** `scripts/hero8_launch.sh` (smoke|full, one model or both; balance guard MIN_CREDITS_USD=300 as hero7),
`scripts/hero8_foldin.sh` (waits per model, fetches without --force, redacts, grades within GRADE_CAP/CEIL, scores,
audits; outputs `results/hero8_full_runs__<model>{,__other21}{,_per_run}.csv`, `results/hero8_loss_audit__<model>{,__other21}.md`,
logs in `results/logs/hero8/`). All 58 configs exist (29 envs x 2 models, incl. ai_rd and swe_rebench_v2). Both Modal
secrets exist in the spar profile (`openrouter-secret-gigi`, `openrouter-agent-gigi`). No Modal app of ours is running now.

**Harness check (read, not guessed).** The hard rule + hack work in gemma's `text_tools: true` mode: the synthetic
`end_task` tool is appended to `tools` before the provider is built, so `_text_tools_messages` renders it as a Python stub
in the prompt gemma sees; the worked example names the environment's real finish tool (`done` in coding/precommit_hook,
`end_session` in swe_rebench_v2, else `end_task`), as for Llama; a gemma `tool_code` call `end_task(report=...)` or
`end_task("...")` parses into a structured call (positional args supported), `end_task_required.install` records the
report, sets `end_task_called`, and runner/tool_calling end the run; a plain reply gets the reminder as a user turn, which
the text-mode view merges correctly. Qwen3.6's `reasoning_effort: disabled` path is unchanged and uses native tools.
**No harness change was needed.**

**Pipeline changes made (dated comments in each file).** `scripts/full_runs_table.py`: hero8 stages added to
HERO_ALL_STAGES and HERO_TARGETS (incl. the smoke stages). `scripts/grade_modal.py`: `hero8_gemma_all`, `hero8_qwen36_all`
added to the "grade every judged environment" list. `scripts/hero_solo_foldin.sh`: hero8 recognised for TAG/log dir and
for the HARD wording of the loss audit.

**Launch commands (absolute paths, detached; smoke first, one run per loop type: coding = own finish tool,
tictactoe = game-ending loop, grade_your_own = plain-reply ending).**
```
nohup caffeinate -i bash /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/scripts/hero8_launch.sh smoke both > /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero8/launch_smoke.out 2>&1 &
# check: results/raw/hero8_smoke_gemma/ and hero8_smoke_qwen36/ (fetch via the usual sync, no --force), one run each ends via the finish tool
nohup caffeinate -i bash /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/scripts/hero8_launch.sh full both > /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero8/launch_full.out 2>&1 &
CEIL=<$ since 18 Oct, e.g. current 3060.84 + approved budget> MODELS="gemma-3-27b qwen3.6-27b" nohup caffeinate -i bash /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/scripts/hero8_foldin.sh > /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero8/foldin.out 2>&1 &
```
**Watchdog (fill CAP_USD; counts the whole key from now, so set it to the approved cap for everything running on the key):**
```
FORENSICS_ENV_FILE=/Users/gg/Desktop/aisafety/spar-model-hermeneutics/.env CAP_USD=<cap> nohup caffeinate -i bash /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/scripts/spend_watchdog.sh hero8 > /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero8/watchdog.out 2>&1 &
```
Note: a `spend_watchdog.sh hero5` from 8 Nov is still running (pid 86273); its cap counts the key's spend since it
started and would stop *every* app of ours, hero8 included, when reached. Check `results/logs/hero5/watchdog.txt`
before launching and stop or restart it with a higher cap.

**Not verified (needs a run).** The smoke itself: whether gemma actually writes `end_task` in a `tool_code` block
and whether the worked example (which shows `execute_command("cat solution.py")` and the finish tool) steers it,
and grade_your_own's wall clock under gemma (500-step cap, 4 h timeout; eval_tampering:1 (20 steps) is the quick
alternative plain-reply smoke if time is short). Costs: not estimated here (parent's job); the earlier gemma/Qwen3.6
soft-check fleets of the same size are the reference.
